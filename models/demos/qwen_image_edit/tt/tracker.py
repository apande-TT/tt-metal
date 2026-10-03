# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Invocation counter for the graduated stubs (Gate 2).

`track(name, obj, methods)` counts every call of the listed methods on that one stub object (per-block
instances included). `__call__` cannot be shadowed on an instance, so for it the object's class is
swapped for a one-off subclass whose __call__ counts and delegates; the object, its weights and its
behaviour are unchanged. Counting is a Python increment: it adds no device op and no host compute.
"""

from __future__ import annotations

from collections import Counter

# The graduated modules of the three bring-ups (status NEW + a last_good snapshot), per component.
GRADUATED = {
    "text_encoder": (
        "vision_patch_embed",
        "v_l_vision_block",
        "v_l_patch_merger",
        "vision_transformer_pretrained_model",
        "v_l_decoder_layer",
        "language_model_layers_0_mlp",
        "v_l_text_model",
    ),
    "transformer": (
        "timesteps",
        "timestep_embedding",
        "qwen_timestep_proj_embeddings",
        "qwen_embed_rope",
        "qwen_image_transformer_block",
        "feed_forward",
        "ada_layer_norm_continuous",
    ),
    "vae": (
        "qwen_image_encoder3d",
        "qwen_image_decoder3d",
        "qwen_image_causal_conv3d",
        "qwen_image_residual_block",
        "qwen_image_r_m_s",
        "qwen_image_mid_block",
        "qwen_image_attention_block",
        "qwen_image_resample",
        "zero_pad2d",
        "qwen_image_up_block",
        "qwen_image_upsample",
    ),
}
ALL_GRADUATED = tuple(n for names in GRADUATED.values() for n in names)


class Tracker:
    def __init__(self):
        self.calls = Counter()
        self.objects = Counter()
        self.enabled = True
        self._classes = {}  # (class, name) -> counting subclass, shared so a block list stays same-typed

    def track(self, name, obj, methods=("__call__",)):
        self.objects[name] += 1
        for m in methods:
            if m == "__call__":
                obj.__class__ = self._counting_class(type(obj), name)
            else:
                bound = getattr(obj, m)

                def counted(*a, _bound=bound, **k):
                    if self.enabled:
                        self.calls[name] += 1
                    return _bound(*a, **k)

                setattr(obj, m, counted)
        return obj

    def _counting_class(self, cls, name):
        key = (cls, name)
        if key not in self._classes:
            inner, tracker = cls.__call__, self

            def __call__(self_, *a, **k):
                if tracker.enabled:
                    tracker.calls[name] += 1
                return inner(self_, *a, **k)

            self._classes[key] = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})
        return self._classes[key]

    def reset(self):
        self.calls.clear()

    def missing(self, names=ALL_GRADUATED):
        return [n for n in names if self.calls[n] == 0]

    def report(self, names=ALL_GRADUATED):
        return {n: int(self.calls[n]) for n in names}
