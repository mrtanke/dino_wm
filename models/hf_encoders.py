import torch
import torch.nn as nn
import torchvision
import inspect
from huggingface_hub import snapshot_download
from transformers import AutoModel


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
        name: str = "hf_encoder",
    ):
        super().__init__()
        self.name = name
        self.model_name_or_path = model_name_or_path
        self.token_pool = token_pool
        self.feature_key = feature_key
        self.drop_cls_token = drop_cls_token
        resolved_model_path = model_name_or_path
        if "/" in model_name_or_path:
            # Resolve repo IDs to concrete local snapshot paths. This avoids
            # repo-id cache lookup issues with some hub/transformers version combinations.
            try:
                resolved_model_path = snapshot_download(repo_id=model_name_or_path)
            except Exception:
                resolved_model_path = snapshot_download(
                    repo_id=model_name_or_path,
                    local_files_only=True,
                )

        self.base_model = AutoModel.from_pretrained(
            resolved_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=resolved_model_path != model_name_or_path,
        )
        self._forward_params = set(inspect.signature(self.base_model.forward).parameters.keys())

        hidden_size = getattr(self.base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.base_model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not infer embedding dim from config of {model_name_or_path}."
            )

        self.emb_dim = int(hidden_size)
        self.latent_ndim = 1 if token_pool == "cls" else 2
        self.patch_size = getattr(self.base_model.config, "patch_size", None)

        self.normalize = None
        if normalize_imagenet:
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
        super().__init__(name="vfm_vae", **kwargs)
