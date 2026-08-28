# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-specific PCC-harness inputs for the FLUX.2-klein-9B *transformer*.

Why this file exists
--------------------
The generated per-component tests build their synthetic inputs from
``_make_arg_for``, which knows transformers/ViT naming conventions
(``pixel_values``, ``hidden_states``, ``input_ids``, ...).  FLUX.2 is a
**diffusers MMDiT**, and four of its input conventions are invisible to that
builder, so the *torch reference* raises and the test SKIPs before the stub is
ever called:

1. ``temb_mod`` / ``temb_mod_img`` / ``temb_mod_txt`` are PACKED AdaLN
   modulation vectors of width ``3 * mod_param_sets * dim`` at rank 2
   (24576 for the double-stream blocks, 12288 for the single-stream ones).
   The generic builder hands them a ``(1, 64, 4096)`` activation;
   ``Flux2Modulation.split`` then chunks 4096 into 3 or 6 unequal pieces and
   the broadcast against the normed hidden states raises.
2. ``encoder_hidden_states`` is REQUIRED by the double-stream block and by
   ``Flux2Attention`` (``added_kv_proj_dim`` is not None, so the processor
   unconditionally projects it).  The generic builder returns ``None`` for that
   name, and for ``Flux2Attention`` drops it entirely because it is optional in
   the signature -- either way the processor dereferences ``None``.
   The SINGLE-stream block is the exception: it documents
   ``encoder_hidden_states=None`` as "already concatenated", so ``None`` is
   correct there.
3. ``AdaLayerNormContinuous.conditioning_embedding`` must be RANK 2
   ``(B, cond_dim)``: the module does ``chunk(emb, 2, dim=1)`` then
   ``[:, None, :]``.  A rank-3 ``(1, 64, 4096)`` splits the TOKEN axis instead.
4. ``Flux2PosEmbed.ids`` is an ``(S, 4)`` position table and ``Timesteps``
   asserts a 1-D input; both get a ``(1, 64, C)`` activation and raise.

Everything here is *inputs only* -- it never touches an assertion, a PCC
threshold, or a stub.

Sequence layout
---------------
One shared layout keeps the rope tables consistent with the activations:
``N_TXT`` text tokens followed by ``N_IMG`` image tokens, matching what
``Flux2Transformer2DModel.forward`` builds (text first, then image).  Text ids
are all-zero (that is what the Flux pipelines pass); image ids carry a square
row/col grid on rope axes 1 and 2.
"""

from __future__ import annotations

import torch

DIM = 4096
HEAD_DIM = 128
N_IMG = 64
N_TXT = 32
FF_INNER = 12288  # dim * mlp_ratio(3.0)


class _Omit:
    """Marker asking the caller to drop the arg entirely (its ``_OMIT``)."""


OMIT = _Omit()


def _grid_ids(n_txt: int, n_img: int) -> torch.Tensor:
    """The ``(S, 4)`` rope id table for ``[txt, img]``.

    Text ids are zeros (Flux pipelines pass ``torch.zeros(L, axes)``); image ids
    use a square grid on axes 1/2 so positions stay small and every token is
    distinct.
    """
    ids = torch.zeros(n_txt + n_img, 4, dtype=torch.float32)
    side = max(1, int(round(n_img**0.5)))
    idx = torch.arange(n_img, dtype=torch.float32)
    ids[n_txt:, 1] = torch.div(idx, side, rounding_mode="floor")
    ids[n_txt:, 2] = idx % side
    return ids


def _rope(model, n_txt: int, n_img: int):
    """``(cos, sin)`` of shape ``(n_txt + n_img, 128)`` from the model's own
    ``Flux2PosEmbed`` -- the exact tables the real forward would feed the
    blocks."""
    with torch.no_grad():
        return model.pos_embed(_grid_ids(n_txt, n_img))


def _mod(width: int) -> torch.Tensor:
    """A packed rank-2 AdaLN modulation vector.

    Scaled down to 0.1: real ``Flux2Modulation`` outputs are small, and
    ``(1 + scale)`` staying near 1 keeps the block in the regime it was trained
    for instead of amplifying one arbitrary channel.
    """
    return torch.randn(1, width, dtype=torch.float32) * 0.1


# Extra forward args that are optional in the signature but that the module
# genuinely needs -- the generated test only builds an arg when it is required
# OR well-known, so these names have to be promoted per component.
_EXTRA_WELL_KNOWN = {
    "flux2_attention": ("encoder_hidden_states", "image_rotary_emb"),
    "encoder_stack": ("image_rotary_emb",),
    "flux2_transformer_block": ("image_rotary_emb",),
    "flux2_parallel_self_attention": ("image_rotary_emb",),
    "self_attention": ("image_rotary_emb",),
    "flux2_single_transformer_block": ("image_rotary_emb",),
}


def extra_well_known(component: str) -> tuple:
    return _EXTRA_WELL_KNOWN.get(component, ())


# (component, arg_name) -> builder(model, torch_module).  A component absent
# from the table falls through to the generic `_make_arg_for`.
_ARGS = {
    # AdaLayerNormContinuous(4096, 4096): rank-2 conditioning vector.
    ("ada_layer_norm_continuous", "conditioning_embedding"): lambda m, tm: torch.randn(1, DIM),
    # Flux2Attention (double stream) -- text stream is mandatory.
    ("flux2_attention", "encoder_hidden_states"): lambda m, tm: torch.randn(1, N_TXT, DIM),
    ("flux2_attention", "image_rotary_emb"): lambda m, tm: _rope(m, N_TXT, N_IMG),
    # Flux2TransformerBlock (double stream), reached under two component names.
    ("flux2_transformer_block", "encoder_hidden_states"): lambda m, tm: torch.randn(1, N_TXT, DIM),
    ("flux2_transformer_block", "temb_mod_img"): lambda m, tm: _mod(6 * DIM),
    ("flux2_transformer_block", "temb_mod_txt"): lambda m, tm: _mod(6 * DIM),
    ("flux2_transformer_block", "image_rotary_emb"): lambda m, tm: _rope(m, N_TXT, N_IMG),
    ("encoder_stack", "encoder_hidden_states"): lambda m, tm: torch.randn(1, N_TXT, DIM),
    ("encoder_stack", "temb_mod_img"): lambda m, tm: _mod(6 * DIM),
    ("encoder_stack", "temb_mod_txt"): lambda m, tm: _mod(6 * DIM),
    ("encoder_stack", "image_rotary_emb"): lambda m, tm: _rope(m, N_TXT, N_IMG),
    # Flux2ParallelSelfAttention (single stream) -- self-attention only, so the
    # sequence is just the already-concatenated stream.
    ("flux2_parallel_self_attention", "image_rotary_emb"): lambda m, tm: _rope(m, 0, N_IMG),
    ("self_attention", "image_rotary_emb"): lambda m, tm: _rope(m, 0, N_IMG),
    # Flux2SingleTransformerBlock: encoder_hidden_states=None is CORRECT here
    # (the block documents hidden_states as already concatenated).
    ("flux2_single_transformer_block", "temb_mod"): lambda m, tm: _mod(3 * DIM),
    ("flux2_single_transformer_block", "image_rotary_emb"): lambda m, tm: _rope(m, 0, N_IMG),
    # Flux2SwiGLU halves its last dim; feed it the real ff inner width.
    ("flux2_swi_g_l_u", "x"): lambda m, tm: torch.randn(1, N_IMG, 2 * FF_INNER),
    # Flux2PosEmbed consumes the id table itself.
    ("flux2_pos_embed", "ids"): lambda m, tm: _grid_ids(0, N_IMG),
    # Timesteps / Flux2TimestepGuidanceEmbeddings take a 1-D timestep.
    # 256.0 is on the scale the pipeline uses (`timestep * 1000`) and spreads
    # the sinusoid across many periods, so the test is not degenerate.
    ("timesteps", "timesteps"): lambda m, tm: torch.tensor([256.0]),
    ("flux2_timestep_guidance_embeddings", "timestep"): lambda m, tm: torch.tensor([256.0]),
    # guidance_embeds=false in this checkpoint, so `guidance_embedder` is None
    # and the module ignores this arg -- but it is required, so it must be
    # present and must be None.
    ("flux2_timestep_guidance_embeddings", "guidance"): lambda m, tm: None,
}


def make_arg(component: str, arg_name: str, *, model, torch_module):
    """``(handled, value)`` for one forward arg of one component."""
    fn = _ARGS.get((component, arg_name))
    if fn is None:
        return False, None
    return True, fn(model, torch_module)


def to_device(extra: dict, stage):
    """Stage every floating-point tensor in ``extra`` on device via ``stage``.

    The generated template forwards non-primary kwargs as host ``torch.Tensor``
    objects.  A native stub would then have to call ``ttnn.from_torch`` inside
    its forward, and ``models/common/native_probe.py`` counts the ``detach`` /
    ``to`` / ``__dlpack__`` that happens there as torch compute with
    ``max_torch_ops=0`` -- refusing graduation to a genuinely pure-ttnn port.
    Ints, bools, ``None`` and non-tensor scalars pass through untouched.
    """
    return {k: _stage_value(v, stage) for k, v in extra.items()}


def _stage_value(v, stage):
    if isinstance(v, torch.Tensor):
        return stage(v) if v.is_floating_point() else v
    if isinstance(v, tuple):
        return tuple(_stage_value(e, stage) for e in v)
    if isinstance(v, list):
        return [_stage_value(e, stage) for e in v]
    return v
