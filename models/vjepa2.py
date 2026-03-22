from models.hf_vision import HFPatchEncoder


class VJEPA2Encoder(HFPatchEncoder):
    def __init__(
        self,
        model_name="facebook/vjepa2-vitl16",
        feature_key="last_hidden_state",
        use_cls_token=False,
        patch_size=16,
        trust_remote_code=True,
        normalize=False,
    ):
        super().__init__(
            name="vjepa2",
            model_name=model_name,
            feature_key=feature_key,
            use_cls_token=use_cls_token,
            patch_size=patch_size,
            trust_remote_code=trust_remote_code,
            normalize=normalize,
        )
