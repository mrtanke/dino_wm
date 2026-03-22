import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

"""
A compatibility adapter
"""

class HFPatchEncoder(nn.Module):
    """Generic HuggingFace vision encoder adapter for patch-token latents."""

    def __init__(
        self,
        name,
        model_name,
        feature_key="last_hidden_state",
        use_cls_token=False,
        patch_size=14,
        trust_remote_code=False,
        normalize=False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "transformers is required for HFPatchEncoder. Please install it first."
            )

        self.name = name
        self.model_name = model_name
        self.feature_key = feature_key
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.normalize = normalize

        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        cfg = getattr(self.base_model, "config", None)
        self.emb_dim = self._infer_emb_dim(cfg)
        self.latent_ndim = 1 if self.use_cls_token else 2

    @staticmethod
    def _infer_emb_dim(cfg):
        if cfg is None:
            raise ValueError("Model config is missing; cannot infer embedding dimension.")
        for attr in ["hidden_size", "embed_dim", "d_model", "dim"]:
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)
        raise ValueError("Unable to infer embedding dimension from model config.")

    @staticmethod
    def _looks_like_cls_prefixed(tokens):
        # Heuristic: cls + square-grid patch tokens.
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            return False
        n_patch = tokens.shape[1] - 1
        side = int(math.isqrt(n_patch))
        return side * side == n_patch

    def _extract_feature_tensor(self, outputs):
        if isinstance(outputs, dict):
            if self.feature_key in outputs:
                return outputs[self.feature_key]
            if "last_hidden_state" in outputs:
                return outputs["last_hidden_state"]
        if hasattr(outputs, self.feature_key):
            return getattr(outputs, self.feature_key)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def _to_patch_tokens(self, features):
        if isinstance(features, (tuple, list)):
            features = features[0]

        if isinstance(features, dict):
            for key in [
                "x_norm_patchtokens",
                "x_patchtokens",
                "patch_tokens",
                "last_hidden_state",
            ]:
                if key in features:
                    features = features[key]
                    break
            else:
                raise ValueError("Could not find token tensor in model outputs dict.")

        if not torch.is_tensor(features):
            raise ValueError("Encoder output is not a tensor.")

        if features.ndim == 4:
            # Convert (B, C, H, W) to patch-token sequence.
            features = rearrange(features, "b c h w -> b (h w) c")
        elif features.ndim == 2:
            features = features.unsqueeze(1)
        elif features.ndim != 3:
            raise ValueError(f"Unsupported encoder output shape: {tuple(features.shape)}")

        if self.use_cls_token:
            if features.shape[1] > 1:
                features = features[:, :1, :]
        else:
            if self._looks_like_cls_prefixed(features):
                features = features[:, 1:, :]

        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)

        return features

    def forward(self, x):
        try:
            outputs = self.base_model(pixel_values=x, return_dict=True)
        except TypeError:
            outputs = self.base_model(x)
        except Exception:
            if hasattr(self.base_model, "forward_features"):
                outputs = self.base_model.forward_features(x)
            else:
                raise

        features = self._extract_feature_tensor(outputs)
        tokens = self._to_patch_tokens(features)
        return tokens
