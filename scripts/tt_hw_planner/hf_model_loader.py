"""HF model loading helpers for trust_remote_code / custom AutoModel repos.

Some custom models (e.g. ACE-Step) register only ``AutoModel`` in
``config.auto_map`` and their ``__init__`` paths call third-party code
(``vector_quantize_pytorch.ResidualFSQ``) that performs real tensor
asserts. Transformers 5.x builds models on the **meta** device during
``from_pretrained``, which makes those asserts fail with::

    RuntimeError: Tensor.item() cannot be called on meta tensors

For those models we instantiate on CPU via ``ModelCls(config)`` and load
hub weights with ``load_state_dict`` afterward.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Tuple


def is_meta_tensor_load_error(exc: BaseException) -> bool:
    """True when ``from_pretrained`` failed due to meta-device init."""
    return isinstance(exc, RuntimeError) and "meta tensors" in str(exc).lower()


def _resolve_auto_model_class(model_id: str, config: Any) -> Optional[type]:
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel")
    if not class_ref:
        return None
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(class_ref, model_id)


def load_hub_state_dict(model_id: str, *, trust_remote_code: bool = True) -> dict:
    """Download/read checkpoint shard(s) for ``model_id`` into a CPU state dict."""
    from transformers.modeling_utils import _get_resolved_checkpoint_files, load_state_dict

    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=model_id,
        variant=None,
        gguf_file=None,
        use_safetensors=True,
        user_agent=None,
        is_remote_code=trust_remote_code,
        download_kwargs={"trust_remote_code": trust_remote_code},
    )
    if not checkpoint_files:
        raise OSError(f"No checkpoint files resolved for {model_id!r}")

    if sharded_metadata is not None:
        state_dict: dict = {}
        for path in checkpoint_files:
            state_dict.update(load_state_dict(path))
        return state_dict
    return load_state_dict(checkpoint_files[0])


def load_via_auto_map_cpu_init(
    model_id: str,
    *,
    torch_dtype: Any = None,
    trust_remote_code: bool = True,
    verbose: bool = False,
) -> Tuple[Any, str]:
    """Load a custom ``AutoModel`` via CPU init + manual weight load.

    Returns ``(model, loader_label)``. Raises on failure.
    """
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_cls = _resolve_auto_model_class(model_id, config)
    if model_cls is None:
        raise ValueError(f"{model_id!r} has no AutoModel entry in config.auto_map")

    label = "AutoModel+cpu_init"
    if verbose:
        print(
            f"  [hf-loader] {model_id} via {label} (config.auto_map AutoModel)",
            file=sys.stderr,
            flush=True,
        )

    model = model_cls(config)
    state_dict = load_hub_state_dict(model_id, trust_remote_code=trust_remote_code)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose and (missing or unexpected):
        print(
            f"  [hf-loader] {label}: load_state_dict missing={len(missing)} " f"unexpected={len(unexpected)}",
            file=sys.stderr,
            flush=True,
        )

    if torch_dtype is not None:
        from scripts.tt_hw_planner.activation_diff import _torch_dtype_from_string

        dtype = _torch_dtype_from_string(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        if dtype is not None:
            model = model.to(dtype=dtype)

    try:
        model.eval()
    except Exception:
        pass
    return model, label


def try_load_via_auto_map(
    model_id: str,
    *,
    torch_dtype: Any = None,
    trust_remote_code: bool = True,
    verbose: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Prefer ``config.auto_map['AutoModel']`` when present.

    Tries ``from_pretrained`` first; on meta-tensor init failure falls
    back to CPU init + hub weights. Returns ``(None, None)`` when the
    model has no custom AutoModel mapping (caller should use cascade).
    """
    import transformers

    try:
        config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        return None, None

    if not _resolve_auto_model_class(model_id, config):
        return None, None

    from scripts.tt_hw_planner.activation_diff import _torch_dtype_from_string

    dtype = _torch_dtype_from_string(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    model_cls = _resolve_auto_model_class(model_id, config)
    assert model_cls is not None
    try:
        model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        label = "AutoModel"
        if verbose:
            print(f"  [hf-loader] {model_id} via {label}", file=sys.stderr, flush=True)
        try:
            model.eval()
        except Exception:
            pass
        return model, label
    except Exception as exc:
        if not is_meta_tensor_load_error(exc):
            if verbose:
                print(
                    f"  [hf-loader] AutoModel.from_pretrained failed: " f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        if verbose:
            print(
                f"  [hf-loader] AutoModel.from_pretrained meta-tensor init failed; " f"retrying CPU init + hub weights",
                file=sys.stderr,
                flush=True,
            )
        return load_via_auto_map_cpu_init(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )


def verify_hf_model_instantiable(model_id: str) -> Tuple[bool, str]:
    """Structural load check for custom AutoModel repos (no weight I/O)."""
    import transformers

    try:
        config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        installed = getattr(transformers, "__version__", "unknown")
        return False, f"transformers (v{installed}) can't load `{model_id}`: {exc}"

    model_cls = _resolve_auto_model_class(model_id, config)
    if model_cls is None:
        return True, ""

    try:
        model_cls(config)
        return True, ""
    except Exception as exc:
        return False, (
            f"transformers can load config for `{model_id}` but CPU init of "
            f"{model_cls.__name__} failed: {type(exc).__name__}: {exc}"
        )
