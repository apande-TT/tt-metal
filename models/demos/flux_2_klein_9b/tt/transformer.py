# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The DIFFUSION TRANSFORMER stage of the FLUX.2-klein-9B TT pipeline.

One denoise step of `diffusers.Flux2Transformer2DModel`, composed entirely out of
the 18 graduated bring-up ports under
``models/tt_dit/pipelines/flux_2_klein_9b_transformer/_stubs``.  Every stub body is
used verbatim (Gate 1 byte-compares them); this module only decides *where* each
port sits, marshals real activations between them, and fills the two holes no
bring-up component was bound to.

Routing
-------
Each graduated stub occupies its own POSITION in the reference model, and every
position's output is the real activation the next position consumes -- no
reference tensor is ever injected at a joint, and no port is called just to be
counted.  The first two blocks of each stack are DECOMPOSED (their sub-modules
are separate ports wired together here, mirroring `Flux2TransformerBlock.forward`
and `Flux2SingleTransformerBlock.forward` op for op); the remaining blocks are
composite ports.

    patch_embed                          x_embedder
    flux2_pos_embed                      pos_embed  (txt_ids, then img_ids)
    flux2_timestep_guidance_embeddings   time_guidance_embed          -> temb_d
    timesteps -> timestep_embedding      time_guidance_embed.{time_proj,
                                         timestep_embedder}           -> temb_s
    flux2_modulation x3                  double_stream_modulation_img / _txt,
                                         single_stream_modulation
    layer                                the affine-free LayerNorms of the four
                                         decomposed blocks (10 positions)
    flux2_attention                      transformer_blocks[0].attn,
                                         transformer_blocks[1].attn
    flux2_swi_g_l_u                      transformer_blocks[0].ff.act_fn
    flux2_feed_forward                   transformer_blocks[0].ff_context
    mlp                                  transformer_blocks[1].ff, .ff_context
    flux2_transformer_block              transformer_blocks[2..6]
    encoder_stack                        transformer_blocks[7]
    self_attention                       single_transformer_blocks[0].attn
    flux2_parallel_self_attention         single_transformer_blocks[1].attn
    flux2_single_transformer_block        single_transformer_blocks[2..23]
    ada_layer_norm_continuous            norm_out
    decoder_head                         proj_out

Two positions have no graduated stub and are implemented natively in ttnn here
(the plan records both up front): ``context_embedder`` (Linear 12288 -> 4096) and
``transformer_blocks[0].ff.linear_in`` / ``.linear_out``, the two matmuls that
bracket the `flux2_swi_g_l_u` port.

The batch axis
--------------
The stage carries B INDEPENDENT samples on the LEADING axis and runs them as ONE
program per denoise step -- never a python loop over samples.  What differs per
sample is exactly what the caller supplies: `hidden_states` `(B, N_img, 128)` and
`encoder_hidden_states` `(B, L_txt, 12288)`, out to `(B, N_img, 128)`.

What is SHARED stays leading-dim 1 and broadcasts, because it genuinely is one
value for the whole batch: the denoise schedule is the same for all B samples, so
`temb`, the three modulation vectors derived from it and the `(S, 128)` rope
tables are computed once in `pin` at batch 1.  `H.mod_chunks` keeps them at
`(1, 1, dim)`, which broadcasts over `(B, N, dim)`; the rope tables broadcast over
`(B, heads, S, head_dim)`.  Nothing about that is a fake batch axis -- collapsing
`temb` to B copies would compute the identical number B times.

B is a SEPARATE axis from the TP=8 weight sharding: every `mesh_partition` /
`all_reduce` / `all_gather` still splits the FEATURE axis, and the batch rides
through them untouched.

The graduated stub bodies were brought up at B=1 and a few of them wrote that 1
into a `ttnn.slice` end bound or into the rank-4 view they hand to
`nlp_create_qkv_heads`.  Those bounds now come off the tensor; every such edit is
declared in `tt/batch_patches/transformer.json` and re-derived by Gate 1 (see
`tt/stubs.py`).  They had to be fixed rather than left alone because a hardcoded
leading 1 does not RAISE at B=32 -- it keeps sample 0 and silently drops 1..31.

Precision notes that are load-bearing
-------------------------------------
* The timestep reaches the sinusoid as an ABSOLUTE angle (`t * 1000`), so it is
  carried to the device in fp32.  It is rounded to bfloat16 FIRST, on the host,
  because the reference does `timestep.to(hidden_states.dtype) * 1000` with
  bf16 `hidden_states` -- so bf16(t)*1000 IS the reference value, and fp32 only
  keeps that value from being destroyed a second time inside the sinusoid.
* The `(S, 4)` rope id tables are staged in fp32 for the same reason: the rope
  argument is an absolute angle and `flux2_pos_embed` typecasts what it is given.
"""

from __future__ import annotations

import importlib.util
import os

import torch

import ttnn
from models.demos.flux_2_klein_9b.tt import stubs
from models.demos.flux_2_klein_9b.tt.depth import stack_depth

_STAGE = "transformer"


def _load_shared_helpers():
    """Import the stubs' shared ``_flux2_ttnn.py`` helper module by path.

    Same `spec_from_file_location` trick the stubs themselves use.  This module
    only READS it (weight staging, mesh width, packed-block re-interleaving); it
    is part of the bring-up directory and must not be edited.
    """
    path = os.path.join(stubs.bringup_dir(_STAGE), "_stubs", "_flux2_ttnn.py")
    spec = importlib.util.spec_from_file_location("_f2k_shared_flux2_ttnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_shared_helpers()


# --------------------------------------------------------------------------- utils


def _resolve_depth(specific, generic, total):
    """`double_layers`/`single_layers` override `layers`, which defaults to all.

    Never 0, and never below `depth.MIN_DISCOVERABLE_STACK`: a stack the profiler's
    walk cannot see is worse than a deep one (see `tt/depth.py`).  Capped builds keep
    the DECOMPOSED blocks (they are the first ones in each stack), so the smallest cap
    gives both decomposed blocks per stack plus one composite port.
    """
    return stack_depth(specific if specific is not None else generic, total)


class _ReplicatedLinear:
    """`nn.Linear` as one replicated matmul (used for `ff.linear_in`, whose full
    packed 24576-wide output is what `flux2_swi_g_l_u` slices its halves out of)."""

    def __init__(self, device, linear):
        self.cfg = H.compute_config()
        self.w = H.matmul_weight(linear, device)
        self.b = H.bias_vector(linear, device)

    def __call__(self, x):
        return ttnn.linear(x, self.w, bias=self.b, compute_kernel_config=self.cfg)


class _RowParallelLinear:
    """`nn.Linear` split on its CONTRACTION axis + one `all_reduce`.

    Each chip takes its own slice of the replicated activation with
    `mesh_partition` (a local view, no fabric traffic), matmuls its own rows, and
    the partials are summed -- a matmul over a concatenated contraction axis IS
    the sum of the per-block matmuls.  A bias is added ONCE after the reduce.
    """

    def __init__(self, device, linear):
        self.cfg = H.compute_config()
        w = linear.weight.detach().float().t().contiguous()  # (in, out)
        tp = H.mesh_width(device)
        self.tp = tp if tp > 1 and w.shape[0] % tp == 0 else 1
        self.w = H.stage(w, device, shard_dim=0 if self.tp > 1 else None)
        self.b = H.bias_vector(linear, device)

    def __call__(self, x):
        if self.tp > 1:
            x = ttnn.mesh_partition(x, dim=-1)
        out = ttnn.linear(x, self.w, compute_kernel_config=self.cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out)
        if self.b is not None:
            out = ttnn.add(out, self.b)
        return out


# --------------------------------------------------------------------------- blocks


class _StageBlock:
    """Common base for every element of `double_blocks` / `single_blocks`.

    A structure walk over either list sees ONE concrete type, and both lists
    share this base: the decomposed and composite variants are the same class
    with different fields, not different classes, so nothing downstream has to
    branch on which blocks happen to be decomposed at a given depth.

    `__init__` records WHICH block this is; `build()` stages its ports.  Keeping the
    two apart is what lets `build_pipeline` hand back a pipeline whose stacks are
    already walkable without a single weight having been written to the device.
    """

    #: "double" | "single"
    kind = None

    def __init__(self, stage, index, hf_block, mode):
        self.stage = stage
        self.index = index
        self.hf = hf_block
        #: "decomposed" | "composite"
        self.mode = mode
        self._built = False

    def __repr__(self):
        return f"<{type(self).__name__} {self.kind}[{self.index}] {self.mode}>"

    def build(self):
        """Stage this position's ports.  Idempotent."""
        if not self._built:
            self._build()
            self._built = True
        return self

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError


class _DoubleBlock(_StageBlock):
    """One `Flux2TransformerBlock` position -- decomposed for index 0 and 1,
    a single composite port after that."""

    kind = "double"

    def __init__(self, stage, index, hf_block, *, composite_name=None):
        super().__init__(stage, index, hf_block, "composite" if composite_name else "decomposed")
        self._composite_name = composite_name

    def _build(self):
        stage, index, hf_block = self.stage, self.index, self.hf
        composite_name = self._composite_name
        device = stage.device
        prefix = f"transformer_blocks[{index}]"

        if composite_name is not None:
            self.block = stage._bind(composite_name, prefix, hf_block)
            return

        # --- decomposed: one port per sub-module, wired as Flux2TransformerBlock.forward
        self.norm1 = stage._layer_port(f"{prefix}.norm1", hf_block.norm1)
        self.norm1_context = stage._layer_port(f"{prefix}.norm1_context", hf_block.norm1_context)
        self.norm2 = stage._layer_port(f"{prefix}.norm2", hf_block.norm2)
        self.norm2_context = stage._layer_port(f"{prefix}.norm2_context", hf_block.norm2_context)
        self.attn = stage._bind("flux2_attention", f"{prefix}.attn", hf_block.attn)

        if index == 0:
            # `ff` around the flux2_swi_g_l_u port: linear_in replicated to the
            # full packed width (the port slices [gate | up] itself and
            # mesh_partitions each half), linear_out row-parallel over exactly
            # the inner features the port's all_gather handed back.
            self.ff_linear_in = _ReplicatedLinear(device, hf_block.ff.linear_in)
            self.ff_act = stage._bind("flux2_swi_g_l_u", f"{prefix}.ff.act_fn", hf_block.ff.act_fn)
            self.ff_linear_out = _RowParallelLinear(device, hf_block.ff.linear_out)
            self.ff = None
            self.ff_context = stage._bind("flux2_feed_forward", f"{prefix}.ff_context", hf_block.ff_context)
        else:
            self.ff_act = None
            self.ff = stage._bind("mlp", f"{prefix}.ff", hf_block.ff)
            self.ff_context = stage._bind("mlp", f"{prefix}.ff_context", hf_block.ff_context)

    # -- forward
    def _feed_forward_image(self, x_n):
        if self.ff is not None:
            out = self.ff(x_n)
            self.stage._consumed(out)
            return out
        packed = self.ff_linear_in(x_n)
        hidden = self.ff_act(packed)
        out = self.ff_linear_out(hidden)
        # `flux2_swi_g_l_u`'s result is consumed by a native matmul, which the
        # ledger cannot see (it only observes port -> port hand-offs).
        self.stage._consumed(hidden)
        return out

    def __call__(self, hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt, rope):
        if self.mode == "composite":
            return self.block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=temb_mod_img,
                temb_mod_txt=temb_mod_txt,
                image_rotary_emb=rope,
            )

        stage = self.stage
        x, c = hidden_states, encoder_hidden_states

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = H.mod_chunks(temb_mod_img, 6)
        (
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = H.mod_chunks(temb_mod_txt, 6)
        # `Flux2Modulation.split` is a plain chunk of the modulation port's output.
        stage._consumed([temb_mod_img, temb_mod_txt])

        x_n = stage._modulate(self.norm1, x, scale_msa, shift_msa)
        c_n = stage._modulate(self.norm1_context, c, c_scale_msa, c_shift_msa)

        # The processor returns the IMAGE stream first -- the opposite of
        # `Flux2TransformerBlock.forward`'s own (txt, img) return order.
        attn_img, attn_txt = self.attn(hidden_states=x_n, encoder_hidden_states=c_n, image_rotary_emb=rope)
        stage._consumed([attn_img, attn_txt])

        x = ttnn.add(x, ttnn.multiply(attn_img, gate_msa))
        ff_img = self._feed_forward_image(stage._modulate(self.norm2, x, scale_mlp, shift_mlp))
        x = ttnn.add(x, ttnn.multiply(ff_img, gate_mlp))

        c = ttnn.add(c, ttnn.multiply(attn_txt, c_gate_msa))
        ff_txt = self.ff_context(stage._modulate(self.norm2_context, c, c_scale_mlp, c_shift_mlp))
        stage._consumed(ff_txt)
        c = ttnn.add(c, ttnn.multiply(ff_txt, c_gate_mlp))

        return c, x


class _SingleBlock(_StageBlock):
    """One `Flux2SingleTransformerBlock` position, operating on the concatenated
    `[text | image]` stream (the model runs every single block with
    `encoder_hidden_states=None` and `split_hidden_states=False`)."""

    kind = "single"

    def __init__(self, stage, index, hf_block, *, composite_name=None, attn_name=None):
        super().__init__(stage, index, hf_block, "composite" if composite_name else "decomposed")
        self._composite_name = composite_name
        self._attn_name = attn_name

    def _build(self):
        stage, index, hf_block = self.stage, self.index, self.hf
        prefix = f"single_transformer_blocks[{index}]"

        if self._composite_name is not None:
            self.block = stage._bind(self._composite_name, prefix, hf_block)
            return

        self.norm = stage._layer_port(f"{prefix}.norm", hf_block.norm)
        self.attn = stage._bind(self._attn_name, f"{prefix}.attn", hf_block.attn)

    def __call__(self, hidden_states, temb_mod, rope):
        if self.mode == "composite":
            return self.block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=temb_mod,
                image_rotary_emb=rope,
                split_hidden_states=False,
            )

        shift, scale, gate = H.mod_chunks(temb_mod, 3)
        self.stage._consumed(temb_mod)
        x_n = self.stage._modulate(self.norm, hidden_states, scale, shift)
        attn_out = self.attn(hidden_states=x_n, image_rotary_emb=rope)
        self.stage._consumed(attn_out)
        return ttnn.add(hidden_states, ttnn.multiply(attn_out, gate))


# ----------------------------------------------------------------------- the stage


class Flux2TransformerStage:
    """`Flux2Transformer2DModel` as one on-device denoise step.

    Args:
        device: a `ttnn.MeshDevice` (the stage is built for TP = mesh width).
        hf_transformer: the loaded `Flux2Transformer2DModel`, bf16.  Used only to
            source weights at build time; it is never called in the forward path.
        ledger: optional `stubs.Ledger`; every graduated port is registered at
            its position so the e2e test can prove it ran inside the real path.
        layers: cap on BOTH stacks (None = full depth, floored at
            `depth.MIN_DISCOVERABLE_STACK` -- see `tt/depth.py`).
        double_layers / single_layers: per-stack override of `layers`.

    Constructing the stage lays out both block stacks and stages NO weight; `build()`
    does the device work and every entry point calls it.
    """

    def __init__(
        self,
        device,
        hf_transformer,
        *,
        ledger=None,
        layers=None,
        double_layers=None,
        single_layers=None,
    ):
        self.device = device
        self.hf = hf_transformer
        self.ledger = ledger
        self.cfg = H.compute_config()
        self.tp = H.mesh_width(device)
        #: set by `pin(warmup=True)`; the pipeline uses it before trace capture.
        self._warmed_up = False

        hf = hf_transformer
        self.inner_dim = int(hf.inner_dim)

        n_double = len(hf.transformer_blocks)
        n_single = len(hf.single_transformer_blocks)
        self.num_double = _resolve_depth(double_layers, layers, n_double)
        self.num_single = _resolve_depth(single_layers, layers, n_single)
        #: True when the cap is so tight that a decomposed variant was dropped.
        self.capped_below_decomposed = self.num_double < 2 or self.num_single < 2

        # ------------------------------------------------------------------- blocks
        # BOTH stacks are laid out here, in __init__, and staged later by build():
        # this stage's two sections have to be visible to a structure walk over a
        # freshly-constructed pipeline, and they cannot be if the lists only appear
        # once ~9 B of weights have been pushed to the device.
        self._double_blocks = [
            _DoubleBlock(
                self,
                i,
                hf.transformer_blocks[i],
                composite_name=(
                    None if i < 2 else ("encoder_stack" if i == n_double - 1 else "flux2_transformer_block")
                ),
            )
            for i in range(self.num_double)
        ]
        self._single_blocks = [
            _SingleBlock(
                self,
                i,
                hf.single_transformer_blocks[i],
                composite_name=None if i < 2 else "flux2_single_transformer_block",
                attn_name=("self_attention", "flux2_parallel_self_attention")[i] if i < 2 else None,
            )
            for i in range(self.num_single)
        ]
        self._built = False

    def build(self):
        """Stage every port: embedders, timestep path, modulations, both block
        stacks and the tail.  Idempotent, and called by every entry point."""
        if self._built:
            return self
        hf, device = self.hf, self.device

        # ---------------------------------------------------------------- embedders
        self._patch_embed = self._bind("patch_embed", "x_embedder", hf.x_embedder)
        # No bring-up component was bound to `context_embedder`; implemented
        # natively, row-parallel over the 12288-wide contraction axis.
        self._context_embedder = _RowParallelLinear(device, hf.context_embedder)

        # `pos_embed` is one parameter-free module called at TWO positions, and
        # both results are real (they are the halves of the concatenated rope
        # table).  One port object, two bound positions.
        pos_port = self._build("flux2_pos_embed", hf.pos_embed)
        self._pos_embed_txt = self._wrap("flux2_pos_embed", "pos_embed(txt_ids)", pos_port)
        self._pos_embed_img = self._wrap("flux2_pos_embed", "pos_embed(img_ids)", pos_port)

        # ------------------------------------------------------------ timestep path
        # temb_d: the composite, feeding both double-stream modulations and norm_out.
        self._time_guidance = self._bind(
            "flux2_timestep_guidance_embeddings", "time_guidance_embed", hf.time_guidance_embed
        )
        # temb_s: the DECOMPOSED route through the same module's two children,
        # feeding the single-stream modulation.
        self._timesteps = self._bind("timesteps", "time_guidance_embed.time_proj", hf.time_guidance_embed.time_proj)
        self._timestep_embedding = self._bind(
            "timestep_embedding",
            "time_guidance_embed.timestep_embedder",
            hf.time_guidance_embed.timestep_embedder,
        )

        self._mod_img = self._bind("flux2_modulation", "double_stream_modulation_img", hf.double_stream_modulation_img)
        self._mod_txt = self._bind("flux2_modulation", "double_stream_modulation_txt", hf.double_stream_modulation_txt)
        self._mod_single = self._bind("flux2_modulation", "single_stream_modulation", hf.single_stream_modulation)

        # ------------------------------------------------------------------- blocks
        # The lists were laid out in __init__; this stages their ports, in order.
        for block in self._double_blocks:
            block.build()
        for block in self._single_blocks:
            block.build()

        # --------------------------------------------------------------------- tail
        self._norm_out = self._bind("ada_layer_norm_continuous", "norm_out", hf.norm_out)
        self._proj_out = self._bind("decoder_head", "proj_out", hf.proj_out)
        self._built = True
        return self

    # ------------------------------------------------------------------- properties

    @property
    def staged(self):
        """True once `build()` has put weights on the device.  The pipeline reads this
        to decide whether releasing this stage frees anything."""
        return self._built

    @property
    def double_blocks(self):
        """The `Flux2TransformerBlock` positions, in order (plain list, one type).

        A COPY, so a caller cannot resize the stack the walk reads; `_double_blocks`
        is the list itself and is what `step` iterates.
        """
        return list(self._double_blocks)

    @property
    def single_blocks(self):
        """The `Flux2SingleTransformerBlock` positions, in order."""
        return list(self._single_blocks)

    # ------------------------------------------------------------ port plumbing

    def _build(self, name, torch_module):
        """Build a graduated port from its verbatim stub body."""
        return stubs.load_stub_module(_STAGE, name).build(self.device, torch_module)

    def _wrap(self, name, position, port):
        if self.ledger is None:
            return port
        return self.ledger.bind(_STAGE, name, position, port)

    def _bind(self, name, position, torch_module):
        return self._wrap(name, position, self._build(name, torch_module))

    def _layer_port(self, position, torch_module):
        """A `layer` port for one affine-free `nn.LayerNorm` position.

        These norms hold no parameters, so a single port body would serve every
        position -- but each position is bound separately so the ledger records
        where the port actually ran.
        """
        return self._bind("layer", position, torch_module)

    def _consumed(self, obj):
        """Record that a bound port's output was consumed by a plain ttnn op.

        The ledger infers consumption by matching a port's output tensor ids
        against the arguments of the NEXT bound port, so a port that hands its
        result to a native op (`ttnn.concat` for the rope halves, the modulation
        multiply after `layer`, the row-parallel `linear_out` after
        `flux2_swi_g_l_u`) would read as unconsumed even though its value is
        carried all the way to the noise prediction.  `Ledger.mark_final` is the
        public entry point for "whatever produced this was consumed".
        """
        if self.ledger is not None:
            self.ledger.mark_final(obj)

    def _modulate(self, norm_port, x, scale, shift):
        """`layer(norm)(x) * (1 + scale) + shift` -- the AdaLN step of both block
        kinds, with the norm itself supplied by the graduated `layer` port."""
        normed = norm_port(x)
        out = ttnn.add(ttnn.multiply(normed, ttnn.add(scale, 1.0)), shift)
        self._consumed(normed)
        return out

    # ----------------------------------------------------------------- host staging

    @staticmethod
    def _stage_activation(x, device, name):
        """An activation as a replicated device tensor.

        `ttnn` in means `ttnn` through: at the pipeline joint this IS the previous
        stage's real output and must not be round-tripped.  A torch tensor is
        accepted for standalone use (`__call__` refuses it -- see there).
        """
        if isinstance(x, ttnn.Tensor):
            return x
        if isinstance(x, torch.Tensor):
            return H.as_device(x, device)
        raise TypeError(f"{name} must be a ttnn.Tensor or a torch.Tensor, got {type(x).__name__}")

    def _stage_timestep(self, timestep, device=None):
        """`timestep * 1000` as a replicated `(1, 1)` fp32 device tensor.

        The reference computes `timestep.to(hidden_states.dtype) * 1000` with bf16
        `hidden_states`, so the bf16 rounding is part of the reference value and is
        reproduced here on the host.  It is then carried in fp32 because the
        sinusoid treats it as an absolute angle: at ~1e3 one bfloat16 ulp is
        several radians.
        """
        if isinstance(timestep, torch.Tensor):
            value = float(timestep.detach().reshape(-1)[0])
        else:
            value = float(timestep)
        scaled = (torch.tensor([value], dtype=torch.bfloat16) * 1000).to(torch.float32)
        return H.as_device(scaled.reshape(1, 1), device or self.device, dtype=ttnn.float32)

    def _stage_ids(self, ids, device=None):
        """An `(S, 4)` rope id table as a replicated fp32 device tensor.

        `flux2_pos_embed` typecasts what it is handed and forms an absolute rope
        angle from it, so the table must arrive in fp32.
        """
        if isinstance(ids, ttnn.Tensor):
            return ids
        table = ids.detach().float()
        while table.dim() > 2:
            table = table[0]
        return H.as_device(table.contiguous(), device or self.device, dtype=ttnn.float32)

    # ------------------------------------------------------- pin: all the host work

    def pin(
        self,
        device,
        *,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        warmup=False,
    ):
        """Upload everything `step` will read into persistent device buffers.

        Every host-side operation of a denoise step happens here: staging the ids
        and the timestep, and running the ports whose result is per-TIMESTEP
        rather than per-block -- `flux2_pos_embed` (twice, for the rope table of
        the full joint sequence), the two timestep routes, and the three
        modulations.  That is the right factoring for the denoise loop anyway:
        those values are constant across all 32 blocks, and re-deriving them
        inside `step` would put host work in the traced region.

        `hidden_states` / `encoder_hidden_states` may be torch OR ttnn (the
        pipeline hands over ttnn -- the real output of the previous stage);
        `img_ids` / `txt_ids` / `timestep` are host prep and may be torch.

        Both activations carry the batch of INDEPENDENT samples on their leading
        axis and must agree on it.  Everything else pinned here is per-TIMESTEP,
        not per-sample, and is deliberately left at leading dim 1 so it broadcasts
        over the batch: one schedule drives all B samples (see the module
        docstring), so `temb`, the modulations and the rope tables are computed
        once and read B times rather than B times over.

        Args:
            warmup: run one throwaway `step` before returning.  Some ports build
                host-side tables lazily on their first call, and tt-metal wants a
                completed forward before `begin_trace_capture`, so the pipeline
                should pin with `warmup=True` once before capturing a trace.

        Returns:
            dict: the resident buffers, the only thing `step` is allowed to read.
        """
        self.build()
        resident = {
            "hidden_states": self._stage_activation(hidden_states, device, "hidden_states"),
            "encoder_hidden_states": self._stage_activation(encoder_hidden_states, device, "encoder_hidden_states"),
        }

        # --- timestep embedding + modulation parameters (per timestep, not per block)
        t_dev = self._stage_timestep(timestep, device)
        temb_d = self._time_guidance(t_dev)

        proj = self._timesteps(t_dev)
        # The reference feeds `timesteps_proj.to(timestep.dtype)` into the
        # embedder, and `timestep` is bf16 there.
        proj_bf16 = ttnn.typecast(proj, ttnn.bfloat16)
        self._consumed(proj)
        temb_s = self._timestep_embedding(proj_bf16)

        resident["temb"] = temb_d
        resident["mod_img"] = self._mod_img(temb_d)
        resident["mod_txt"] = self._mod_txt(temb_d)
        resident["mod_single"] = self._mod_single(temb_s)

        # --- rope for the whole joint sequence: text tokens LEAD it
        txt_cos, txt_sin = self._pos_embed_txt(self._stage_ids(txt_ids, device))
        img_cos, img_sin = self._pos_embed_img(self._stage_ids(img_ids, device))
        resident["rope"] = (
            ttnn.typecast(ttnn.concat([txt_cos, img_cos], dim=0), ttnn.bfloat16),
            ttnn.typecast(ttnn.concat([txt_sin, img_sin], dim=0), ttnn.bfloat16),
        )
        self._consumed([txt_cos, txt_sin, img_cos, img_sin])

        # The traced shape is whatever was pinned; `step` runs at exactly this one.
        resident["txt_len"] = int(resident["encoder_hidden_states"].shape[-2])
        resident["img_len"] = int(resident["hidden_states"].shape[-2])
        # The two streams are the SAME B samples seen from two sides, so a
        # mismatch is a wiring bug the joint sequence would otherwise absorb
        # silently (the shorter one would just broadcast).
        resident["batch"] = int(resident["hidden_states"].shape[0])
        if int(resident["encoder_hidden_states"].shape[0]) != resident["batch"]:
            raise ValueError(
                f"hidden_states carries {resident['batch']} sample(s) but encoder_hidden_states "
                f"carries {int(resident['encoder_hidden_states'].shape[0])} -- the image and text "
                f"streams of one denoise step are the same samples and must share a leading axis"
            )

        if warmup:
            self.step(resident)
            self._warmed_up = True

        return resident

    # --------------------------------------------------- step: the traced forward

    def step(self, resident):
        """One denoise step over pinned buffers -- no host work, no torch.

        Reads only `resident`: the embedders, the 32 blocks, `norm_out` and
        `proj_out`, at the fixed shape that was pinned.

        All B samples go through as ONE program: every op below is shape-polymorphic
        in the leading axis, and the only place that axis is named is the slice that
        drops the text tokens, where the bound is read off the tensor.

        Returns:
            ttnn `(B, N_img, 128)` noise prediction.
        """
        rope = resident["rope"]
        mod_img, mod_txt, mod_single = resident["mod_img"], resident["mod_txt"], resident["mod_single"]
        txt_len = resident["txt_len"]

        # 1. input projections
        x = self._patch_embed(resident["hidden_states"])
        c = self._context_embedder(resident["encoder_hidden_states"])

        # 2. double stream
        for block in self._double_blocks:
            c, x = block(x, c, mod_img, mod_txt, rope)

        # 3. single stream over the concatenated [text | image] sequence
        self._consumed([c, x])
        x = ttnn.concat([c, x], dim=-2)
        for block in self._single_blocks:
            x = block(x, mod_single, rope)

        # 4. drop the text tokens, then the output head
        batch, total, width = x.shape[0], x.shape[-2], x.shape[-1]
        self._consumed(x)
        x = ttnn.slice(x, [0, txt_len, 0], [batch, total, width])

        x = self._norm_out(x, conditioning_embedding=resident["temb"])
        return self._proj_out(x)

    # --------------------------------------------------------------------- forward

    def __call__(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        """One denoise step: `pin` then `step`, so there is only ever one body.

        Args:
            hidden_states: ttnn `(B, N_img, 128)` packed latents -- B INDEPENDENT
                samples stacked on the leading axis, run as one program.
            encoder_hidden_states: ttnn `(B, L, 12288)` prompt embeddings -- the
                real output of the text stage, one distinct prompt per sample.
            timestep: scalar (or 1-element tensor) in the pipeline's `t/1000`
                scale; `* 1000` happens inside `pin`, as it does inside the
                reference's own forward.
            img_ids / txt_ids: `(S, 4)` float position tables (host prep).

        Returns:
            ttnn `(B, N_img, 128)` noise prediction.
        """
        for name, value in (
            ("hidden_states", hidden_states),
            ("encoder_hidden_states", encoder_hidden_states),
        ):
            if not isinstance(value, ttnn.Tensor):
                raise TypeError(
                    f"{name} must be a ttnn.Tensor at the pipeline joint (the real output of the "
                    f"previous TT stage), got {type(value).__name__}"
                )
        return self.step(
            self.pin(
                self.device,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )
        )
