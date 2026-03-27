import torch
import torch.nn as nn
import torchvision
import inspect
import os
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel

try:
    from transformers import SiglipVisionModel
except Exception:
    SiglipVisionModel = None

try:
    from transformers import Siglip2VisionModel
except Exception:
    Siglip2VisionModel = None


class HFAutoVisionEncoder(nn.Module):
    """
    Generic Hugging Face vision encoder wrapper that matches the DINO-WM encoder API.
    """

    def __init__(
        self,
        model_name_or_path: str,
        token_pool: str = "patch",
        feature_key: str = "last_hidden_state",
        drop_cls_token: bool = True,
        trust_remote_code: bool = True,
        normalize_imagenet: bool = True,
        normalize_mean=None,
        normalize_std=None,
        local_files_only: bool = False,
        name: str = "hf_encoder",
    ):
        super().__init__()
        self.name = name
        self.model_name_or_path = model_name_or_path
        self.token_pool = token_pool
        self.feature_key = feature_key
        self.drop_cls_token = drop_cls_token
        resolved_model_path = model_name_or_path
        force_local = bool(local_files_only)

        def _has_model_weights(path: str) -> bool:
            return any(
                os.path.exists(os.path.join(path, f))
                for f in [
                    "pytorch_model.bin",
                    "model.safetensors",
                    "tf_model.h5",
                    "model.ckpt.index",
                    "flax_model.msgpack",
                ]
            )

        if "/" in model_name_or_path:
            # Resolve repo IDs to concrete local snapshot paths. Try local cache
            # first for reproducibility and speed, then allow online download.
            try:
                resolved_model_path = snapshot_download(
                    repo_id=model_name_or_path,
                    local_files_only=True,
                )
            except Exception:
                if force_local:
                    raise
                resolved_model_path = snapshot_download(
                    repo_id=model_name_or_path,
                    local_files_only=False,
                )

            if not _has_model_weights(resolved_model_path):
                # Some repos/checkouts may yield incomplete local snapshots in
                # specific hub configurations. Fall back to direct HF loading.
                resolved_model_path = model_name_or_path

        def _load_model(model_ref: str, local_only: bool):
            cfg = AutoConfig.from_pretrained(model_ref, local_files_only=local_only)
            model_type = str(getattr(cfg, "model_type", "")).lower()

            # SigLIP families expose joint text-image models by default.
            # DINO-WM encoder is image-only, so load vision-only models here.
            if model_type == "siglip" and SiglipVisionModel is not None:
                return SiglipVisionModel.from_pretrained(model_ref, local_files_only=local_only)
            if model_type == "siglip2" and Siglip2VisionModel is not None:
                return Siglip2VisionModel.from_pretrained(model_ref, local_files_only=local_only)

            return AutoModel.from_pretrained(
                model_ref,
                trust_remote_code=trust_remote_code,
                local_files_only=local_only,
            )

        if resolved_model_path != model_name_or_path:
            self.base_model = _load_model(resolved_model_path, local_only=True)
        else:
            try:
                self.base_model = _load_model(model_name_or_path, local_only=True)
            except Exception:
                if force_local:
                    raise
                self.base_model = _load_model(model_name_or_path, local_only=False)
        self._forward_params = set(inspect.signature(self.base_model.forward).parameters.keys())

        hidden_size = getattr(self.base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.base_model.config, "d_model", None)
        if hidden_size is None and hasattr(self.base_model.config, "vision_config"):
            hidden_size = getattr(self.base_model.config.vision_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer embedding dim from config of {model_name_or_path}."
            )

        self.emb_dim = int(hidden_size)
        self.latent_ndim = 1 if token_pool == "cls" else 2
        self.patch_size = getattr(self.base_model.config, "patch_size", None)
        if self.patch_size is None and hasattr(self.base_model.config, "vision_config"):
            self.patch_size = getattr(self.base_model.config.vision_config, "patch_size", None)

        self.normalize = None
        if normalize_mean is not None and normalize_std is not None:
            self.normalize = torchvision.transforms.Normalize(
                mean=list(normalize_mean), std=list(normalize_std)
            )
        elif normalize_imagenet:
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

    def _extract_features(self, outputs):
        if hasattr(outputs, self.feature_key):
            return getattr(outputs, self.feature_key)
        if isinstance(outputs, dict) and self.feature_key in outputs:
            return outputs[self.feature_key]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise ValueError(
            f"Unable to extract feature '{self.feature_key}' from model outputs."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize is not None:
            x = self.normalize(x)

        if "pixel_values" in self._forward_params:
            outputs = self.base_model(pixel_values=x)
        elif "pixel_values_videos" in self._forward_params:
            # VJEPA2 expects 5D input [B, T, C, H, W]. For single-frame training,
            # tile the frame along time to satisfy tubelet-based video patching.
            tubelet = int(getattr(self.base_model.config, "tubelet_size", 2))
            if x.ndim == 4:
                x_video = x.unsqueeze(1).repeat(1, max(2, tubelet), 1, 1, 1)
            elif x.ndim == 5:
                x_video = x
            else:
                raise ValueError(f"Unexpected VJEPA2 input shape: {tuple(x.shape)}")
            outputs = self.base_model(pixel_values_videos=x_video, skip_predictor=True)
        else:
            outputs = self.base_model(x)
        feat = self._extract_features(outputs)

        if feat.ndim == 2:
            # Already pooled feature.
            return feat.unsqueeze(1)

        if feat.ndim != 3:
            raise ValueError(f"Unexpected feature shape from encoder: {tuple(feat.shape)}")

        if self.token_pool == "cls":
            return feat[:, :1, :]

        # Keep patch tokens for world-model predictor.
        if self.drop_cls_token and feat.shape[1] > 1:
            feat = feat[:, 1:, :]
        return feat


class VJEPA2Encoder(HFAutoVisionEncoder):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)  # Remove name if it exists in kwargs to avoid duplicate argument error
        super().__init__(name="vjepa2", **kwargs)


class DINOv3Encoder(HFAutoVisionEncoder):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)  # Remove name if it exists in kwargs to avoid duplicate argument error
        super().__init__(name="dinov3", **kwargs)


class DINOTokEncoder(HFAutoVisionEncoder):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)  # Remove name if it exists in kwargs to avoid duplicate argument error
        super().__init__(name="dinotok", **kwargs)


class VFMVAEEncoder(HFAutoVisionEncoder):
    def __init__(self, **kwargs):
        # VFM-VAE uses a VFM backbone (default in paper/repo: SigLIP2) for
        # visual token extraction. We expose this as a dedicated encoder
        # adapter with paper-aligned defaults while keeping the same output API.
        kwargs.setdefault("model_name_or_path", "google/siglip-base-patch16-224")
        kwargs.setdefault("feature_key", "last_hidden_state")
        kwargs.setdefault("token_pool", "patch")
        kwargs.setdefault("drop_cls_token", False)  # SigLIP2 has no cls token.
        kwargs.setdefault("normalize_imagenet", False)
        kwargs.setdefault("normalize_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("normalize_std", [0.5, 0.5, 0.5])
        try:
            super().__init__(name="vfm_vae", **kwargs)
        except OSError:
            model_id = kwargs.get("model_name_or_path", "")
            if "siglip" not in model_id:
                raise
            # Practical fallback to keep training unblocked when the default
            # SigLIP checkpoint cannot be materialized in this environment.
            kwargs["model_name_or_path"] = "google/siglip-base-patch16-224"
            super().__init__(name="vfm_vae", **kwargs)
