"""Reference loader for the Qwen-Image-Edit transformer.

The repo is a diffusers checkpoint (config.json `_class_name` = QwenImageTransformer2DModel,
sharded `diffusion_pytorch_model-*.safetensors`), not a transformers one, so AutoModel cannot
load it. diffusers' own QwenImageTransformer2DModel implements the architecture and loads every
tensor group in the checkpoint (img_in, txt_norm, txt_in, time_text_embed, transformer_blocks,
norm_out, proj_out) with the real weights.
"""

REFERENCE_LOADER_CONTRACT = 2


def load_reference_model(model_id: str):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model, loaded from whatever real format the repo actually ships."""
    import os

    import torch
    from diffusers import QwenImageTransformer2DModel

    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        model = QwenImageTransformer2DModel.from_pretrained(model_id, **kwargs)
    else:
        # Hub pipeline repo (e.g. Qwen/Qwen-Image-Edit) keeps the transformer in a subfolder.
        try:
            model = QwenImageTransformer2DModel.from_pretrained(model_id, **kwargs)
        except (OSError, EnvironmentError):
            model = QwenImageTransformer2DModel.from_pretrained(model_id, subfolder="transformer", **kwargs)
    model.eval()
    return model
