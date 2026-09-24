"""Reference loader for the Qwen-Image(-Edit) VAE.

The repo is a diffusers checkpoint (config.json `_class_name: AutoencoderKLQwenImage`
+ diffusion_pytorch_model.safetensors), not a transformers model. The architecture
lives in diffusers, so we load it natively with the real weights. The checkpoint's
top-level groups (encoder, decoder, quant_conv, post_quant_conv) are all members of
AutoencoderKLQwenImage and are loaded strictly.
"""

REFERENCE_LOADER_CONTRACT = 2


def load_reference_model(model_id: str):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model, loaded from whatever real format the repo actually ships."""
    import torch
    from diffusers import AutoencoderKLQwenImage

    model = AutoencoderKLQwenImage.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    model.eval()
    return model
