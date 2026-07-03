import torch

import ttnn


def _coerce_to_torch(x):
    try:
        import ttnn as _ttnn

        if isinstance(x, _ttnn.Tensor):
            import torch as _torch

            t = _ttnn.to_torch(x)
            # Bug Y fix (2026-05-23 live-run sam2-hiera-tiny)
            if t.is_floating_point():
                if t.dtype != _torch.float32:
                    t = t.to(_torch.float32)
            elif t.dtype != _torch.bool:
                t = t.to(_torch.long)
            return t
    except Exception:
        pass
    if isinstance(x, tuple):
        return tuple(_coerce_to_torch(e) for e in x)
    if isinstance(x, list):
        return [_coerce_to_torch(e) for e in x]
    if isinstance(x, dict):
        return {k: _coerce_to_torch(v) for k, v in x.items()}
    return x


class Lambda:
    def __init__(self, device, torch_module):
        self.device = device
        self.torch_module = torch_module.eval()

    def _pick_tensor(self, value):
        if torch.is_tensor(value):
            return value
        if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
            return value.last_hidden_state
        if isinstance(value, dict):
            for v in value.values():
                t = self._pick_tensor(v)
                if t is not None:
                    return t
            return None
        if isinstance(value, (list, tuple)):
            for v in value:
                t = self._pick_tensor(v)
                if t is not None:
                    return t
            return None
        return None

    def __call__(self, *args, **kwargs):
        # decoder.proj_in.0 is `lambda x: x.transpose(1, 2)`: captured I/O is
        # (1, 50, 192) -> (1, 192, 50), a swap of the last two axes. The
        # harness feeds the primary arg as a ttnn tensor already on device;
        # only fall back to `ttnn.from_torch` if a raw torch tensor sneaks in.
        x = args[0] if args else next(iter(kwargs.values()))
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        return ttnn.transpose(x, 1, 2)


def build(device, torch_module):
    return Lambda(device, torch_module)


_instance = None


def lambda_shim(*args, **kwargs):
    global _instance
    if _instance is None:
        raise RuntimeError(
            "Synthesized TTNN module requires `build(device, torch_module)`. "
            "Call it from the PCC test's `_build_ttnn_port`."
        )
    return _instance(*args, **kwargs)
