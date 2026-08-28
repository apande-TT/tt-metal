# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN pipeline for `black-forest-labs/FLUX.2-klein-9B`.

Both `demo/` and `tests/e2e/` import and call the heads in this file, so a green
test guarantees a working demo -- there is exactly one copy of the wiring.

Four heads, all four of them real task outputs of this checkpoint:

    run_text_to_image     text            -> image     (Flux2KleinPipeline.__call__)
    run_image_edit        text + refs     -> image      (multi-reference editing)
    run_text_generation   text            -> text       (the text_encoder IS a Qwen3ForCausalLM)
    run_vae_roundtrip     image           -> image      (AutoencoderKLFlux2's own codec)

Every stage is fed the previous TT stage's real device output.  No reference tensor
is ever injected at a joint, and no HF submodule is called on the forward path --
`reference.py` owns the goldens, and the only HF use here is reading `.config` and
sourcing weights at build time.

The latent plumbing between the transformer and the VAE (unpack, BatchNorm
de-normalisation, unpatchify -- and their inverse for reference images) runs ON
DEVICE, built out of ops this build actually supports: a 0/1 selection matmul for
the channel de-interleave, a ROW_MAJOR view reshape, `ttnn.upsample` and a 0/1
sub-pixel mask.  A `ttnn.reshape` that splits a TILED axis does not compile here, so
every rank change happens in ROW_MAJOR.
"""

from __future__ import annotations

import contextlib
import gc

import torch

import ttnn
from models.demos.flux_2_klein_9b import host_inputs as L
from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.tt.stubs import Ledger, captured_dir

#: Phases derived from Source A's module graph (model_index.json) plus the one
#: AutoModel head inside it.  Not is_encoder_decoder and not a bare ForCausalLM, so
#: the phases are the diffusion pipeline's own: text conditioning, reference-image
#: conditioning, one denoise step, the latent decode -- plus prefill/decode for the
#: Qwen3ForCausalLM head.
PIPELINE_STAGES = ["encode_text", "vae_encode", "denoise", "vae_decode", "prefill", "decode"]

#: How many INDEPENDENT samples one call carries on the leading axis.
#:
#: This checkpoint has no autoregressive image contract -- there is no per-step KV
#: cache to index and no decode tile to fill -- so the axis the 32 samples stack on is
#: the one the model's own loop repeats over: the denoise step.  One transformer
#: program per step processes all 32 latents together, and the 32 rows are genuinely
#: independent (different prompts, different noise, different reference images).  What
#: they share is only the weights and the iteration count, which is why the timestep
#: conditioning stays leading-dim 1 and broadcasts rather than being tiled 32 ways.
#:
#: The text->text head batches the same way, over 32 different prompts decoded in
#: lockstep; the VAE head over 32 different images.
BATCH = 32

_COMPUTE_CFG = None


def _compute_config():
    global _COMPUTE_CFG
    if _COMPUTE_CFG is None:
        _COMPUTE_CFG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _COMPUTE_CFG


def _mesh_width(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:  # noqa: BLE001
            pass
    return 1


def _replicate(t: torch.Tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    mapper = ttnn.ReplicateTensorToMesh(device) if _mesh_width(device) > 1 else None
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        mesh_mapper=mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch(x, device) -> torch.Tensor:
    """One chip's copy of a replicated device tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


class Flux2KleinTtPipeline:
    """The resident pipeline object: stages, heads, and the per-stage trace contract."""

    # ------------------------------------------------------------------ build
    def __init__(
        self,
        device,
        *,
        model=None,
        layers=None,
        encode_text_layers=None,
        vae_encode_layers=None,
        denoise_layers=None,
        vae_decode_layers=None,
        prefill_layers=None,
        decode_layers=None,
        ledger=None,
        trace_capacity=None,
        batch: int = BATCH,
        **_demo_kwargs,  # text/prompt/language/... accepted and ignored for call compatibility
    ) -> None:
        self.device = device
        #: the leading-axis width the trace contract and the self-tests drive; a head
        #: called with N prompts uses N, so this is the DEFAULT, not a constraint
        self.batch = int(batch)
        self.ledger = ledger if ledger is not None else Ledger()
        self.layers = layers
        #: one knob per stack; None falls back to `layers`, and `layers=None` means all
        self.stage_layers = {
            "encode_text": encode_text_layers,
            "vae_encode": vae_encode_layers,
            "denoise": denoise_layers,
            "vae_decode": vae_decode_layers,
            "prefill": prefill_layers,
            "decode": decode_layers,
        }
        self._hf = dict(model) if isinstance(model, dict) else {}
        self._stages: dict[str, object] = {}
        self._trace_state: dict[str, dict] = {}
        self.trace_capacity = dict(trace_capacity or {})
        self.scheduler = R.load_scheduler()
        #: per-instance: these cache DEVICE tensors, so they must never be shared
        self._latent_helpers: dict = {}
        # Lay out every stage -- and therefore every repeated block stack this
        # checkpoint declares -- before returning.  Structure only: not one weight
        # reaches the device here, so this stays cheap and needs no particular mesh.
        for name in self.STAGE_OBJECTS:
            self._stage(name)

    #: Per-stage batch ceiling, measured on this device.  Empty: no stage needs one.
    #: `vae_decode` was capped at 16 while `ttnn.group_norm` threw "beyond max L1 size" at 32
    #: (its static circular buffers wanted 1412288 B of the chip's 1499136 B).  `tt_dit`'s
    #: `GroupNorm` now catches that throw at program-BUILD time and re-chunks until it fits,
    #: so the constraint no longer holds: 32 runs in one pass, and faster (135.4s vs 154.0s).
    #: Re-measure before re-adding an entry here.
    STAGE_BATCH_CAP = {}

    def stage_batch(self, stage: str) -> int:
        """The batch this stage can actually run, which is `self.batch` unless the device
        caps it.  The trace contract asks for this rather than `self.batch` so a stage is
        driven at a shape it can hold instead of failing to capture at all."""
        return min(self.batch, self.STAGE_BATCH_CAP.get(stage, self.batch))

    def _depth(self, stage: str):
        override = self.stage_layers.get(stage)
        return override if override is not None else self.layers

    # ---- HF pieces (weights + config only; never called on the forward path)
    def hf_text_encoder(self):
        if "text_encoder" not in self._hf:
            self._hf["text_encoder"] = R.load_text_encoder()
        return self._hf["text_encoder"]

    def hf_transformer(self):
        if "transformer" not in self._hf:
            self._hf["transformer"] = R.load_transformer()
        return self._hf["transformer"]

    def hf_vae(self):
        if "vae" not in self._hf:
            self._hf["vae"] = R.load_vae()
        return self._hf["vae"]

    @property
    def hf(self):
        """The HF reference stays reachable from the built object: it is ground truth
        for how many sections this model has and how deep each one is."""
        return self._hf

    # ---- stages
    #
    # Constructing a stage lays out its repeated block stacks and stages NO weight;
    # each stage's own `build()` (called by its every entry point) does the device
    # work.  That split is what lets `build_pipeline` hand back an object whose
    # sections are already discoverable -- the profiler sizes and marks a model by
    # WALKING it (`find_all_stacks`), and a stack that only appears once ~9 B of
    # weights have been pushed is a stack the walk never sees.  See `tt/depth.py`.
    def _stage(self, name: str):
        """The stage object for `name`, constructed (structure only) if absent."""
        if name in self._stages:
            return self._stages[name]
        if name == "encode_text":
            from models.demos.flux_2_klein_9b.tt.text_encoder import Flux2PromptEmbedStage

            built = Flux2PromptEmbedStage(
                self.device, self.hf_text_encoder(), ledger=self.ledger, layers=self._depth("encode_text")
            )
        elif name == "causal_lm":
            from models.demos.flux_2_klein_9b.tt.text_encoder import Qwen3CausalLmStage

            depth = self._depth("prefill")
            if depth is None:
                depth = self._depth("decode")
            built = Qwen3CausalLmStage(self.device, self.hf_text_encoder(), ledger=self.ledger, layers=depth)
        elif name == "denoise":
            from models.demos.flux_2_klein_9b.tt.transformer import Flux2TransformerStage

            built = Flux2TransformerStage(
                self.device, self.hf_transformer(), ledger=self.ledger, layers=self._depth("denoise")
            )
        elif name == "vae":
            from models.demos.flux_2_klein_9b.tt.vae import Flux2VaeStage

            depth = self._depth("vae_decode")
            if depth is None:
                depth = self._depth("vae_encode")
            built = Flux2VaeStage(self.device, self.hf_vae(), ledger=self.ledger, layers=depth)
        else:
            raise KeyError(f"unknown stage {name!r}")
        self._stages[name] = built
        return built

    #: every stage this pipeline owns, in execution order
    STAGE_OBJECTS = ("encode_text", "causal_lm", "denoise", "vae")

    def text_stage(self):
        return self._stage("encode_text")

    def causal_lm_stage(self):
        return self._stage("causal_lm")

    def transformer_stage(self):
        return self._stage("denoise")

    def vae_stage(self):
        return self._stage("vae")

    #: pipeline stage -> the bring-up directory whose stubs it builds
    _LEDGER_KEY = {
        "encode_text": "text_encoder",
        "causal_lm": "text_encoder",
        "denoise": "transformer",
        "vae": "vae",
    }

    def _reclaim_program_cache(self) -> None:
        """Give back the L1_SMALL a released stage's programs were still holding.

        `ttnn.conv2d` keeps its halo / reader-index buffers in L1_SMALL, and the
        PROGRAM CACHE owns them -- so they outlive every Python reference to the port
        that built them, and dropping the stage frees none of them.  Measured on this
        pipeline's VAE stage, one 1x8 device at the bring-up's l1_small_size of 24576 B:

            fresh device                  0 B/bank     0 cached programs
            4 encode routes at B=1     8224 B/bank   126
            + 3 decode routes at B=1  19328 B/bank   219
            clear_program_cache           0 B/bank     0

        At 19328 of 24576 the next convolution cannot get its halo, and it does not
        matter how small the request is: the trace selftest's `vae_decode` died asking
        for 2048 B per bank against 864 B free, and every stage after the first used to
        die asking for 16 B against 0.  A released stage's programs will never run
        again, so this reclaims dead residency rather than dropping live work; the
        stage that builds next pays a program build it was going to pay anyway.

        MUST NOT run while a captured trace is live -- a trace replays exactly the
        programs it captured.  Every caller in this file releases its trace first
        (`trace_capture_selftest` does so even on its failure path); a caller that
        cannot must pass `reclaim=False`.
        """
        clear = getattr(self.device, "clear_program_cache", None)
        if clear is not None:
            clear()

    def release_stage(self, *names: str, reclaim: bool = True) -> None:
        """Drop a built stage so its device weights are freed.

        The text trunk and the transformer are ~9 B parameters each; on a 1x8 mesh
        both fit, but the head methods free the text stage once `prompt_embeds` --
        the real device tensor, not a copy of the golden -- has been produced, so a
        run stays comfortable and so `trace_capture_selftest` can size its region
        from one stage at a time.
        """
        targets = list(names) if names else list(self._stages)
        dropped = [self._stages.pop(name) for name in targets if name in self._stages]

        # A pinned trace state holds the stage object, its resident input buffer, the
        # built port that owns the staged weights and the warm-up output -- so a stage
        # released while its `_trace_state` entry survives frees NOTHING.  That is what
        # turned one stage's cost into every later stage's failure:
        # `trace_capture_selftest` walks six stages in a row on one device, the VAE's
        # conv halo stayed in L1_SMALL across all of them, and denoise, vae_decode,
        # prefill and decode each died asking for 16 B per bank of a region reporting
        # "24576 B allocated, free: 0 B".  Releasing a stage now releases what it pinned.
        if names:
            freed = {id(stage) for stage in dropped}
            for key in [k for k, v in self._trace_state.items() if id(v.get("stage")) in freed]:
                self._trace_state.pop(key, None)
        else:
            self._trace_state.clear()

        # The ledger holds a wrapper per port, and each wrapper holds the built port
        # -- i.e. its staged weights.  Dropping the stage object alone frees nothing.
        # Only drop a bring-up stage's ports once no LIVE pipeline stage still maps to
        # it (encode_text and causal_lm both source from `text_encoder`).
        #
        # LIVE means STAGED, not merely constructed.  Every stage object is laid out
        # up front now (so the stacks are walkable), and counting a laid-out-but-empty
        # stage as live would pin the ports of its bring-up directory forever:
        # releasing encode_text would free nothing, because causal_lm -- structure
        # only, no weights -- also maps to `text_encoder`.
        still_live = {
            self._LEDGER_KEY[n]
            for n, stage in self._stages.items()
            if n in self._LEDGER_KEY and getattr(stage, "staged", True)
        }
        freeing = {self._LEDGER_KEY[n] for n in targets if n in self._LEDGER_KEY} - still_live
        if freeing:
            self.ledger.drop_ports(*sorted(freeing))
        self._latent_helpers = {}
        gc.collect()
        if reclaim:
            self._reclaim_program_cache()

    # ------------------------------------------------- device latent plumbing
    def _unpatchify_helpers(self, h: int, w: int, channels: int, packed: int):
        """0/1 selection matrices and sub-pixel masks for the on-device unpatchify.

        `unpatchify` maps packed channel ``4c + 2*p1 + p2`` at grid (i, j) to output
        channel ``c`` at pixel ``(2i + p1, 2j + p2)``.  Selection is a
        ``(packed, channels)`` 0/1 matmul; the spatial scatter is a nearest-neighbour
        2x upsample masked to the (p1, p2) sub-lattice.  Both are exact in bfloat16.
        """
        key = ("unpatch", h, w, channels, packed)
        hit = self._latent_helpers.get(key)
        if hit is not None:
            return hit
        selects, masks = [], []
        for p1 in (0, 1):
            for p2 in (0, 1):
                sel = torch.zeros(packed, channels)
                for c in range(channels):
                    sel[4 * c + 2 * p1 + p2, c] = 1.0
                selects.append(_replicate(sel, self.device))
                mask = torch.zeros(1, 2 * h, 2 * w, channels)
                mask[:, p1::2, p2::2, :] = 1.0
                masks.append(_replicate(mask, self.device))
        self._latent_helpers[key] = (selects, masks)
        return selects, masks

    def _patchify_permutation(self, channels: int, packed: int):
        """`(packed, packed)` 0/1 matrix taking ``[p1][p2][c]`` order (what four
        strided slices concatenate to) into the model's ``4c + 2*p1 + p2`` order."""
        key = ("patch_perm", channels, packed)
        hit = self._latent_helpers.get(key)
        if hit is not None:
            return hit
        perm = torch.zeros(packed, packed)
        for k, (p1, p2) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            for c in range(channels):
                perm[k * channels + c, 4 * c + 2 * p1 + p2] = 1.0
        out = _replicate(perm, self.device)
        self._latent_helpers[key] = out
        return out

    def _bn_vectors(self, packed: int):
        """The pipeline's top-level BatchNorm statistics, as `(1, 1, packed)` rows."""
        key = ("bn", packed)
        hit = self._latent_helpers.get(key)
        if hit is not None:
            return hit
        mean, std = L.bn_stats(self.hf_vae())
        out = (
            _replicate(mean.reshape(1, 1, -1), self.device),
            _replicate(std.reshape(1, 1, -1), self.device),
        )
        self._latent_helpers[key] = out
        return out

    def _latents_to_nchw(self, x, h: int, w: int, channels: int = 32):
        """(B, h*w, 4C) packed tokens -> (B, C, 2h, 2w) NCHW, all on device.

        This is the pipeline's `unpack_latents_with_ids` + BatchNorm denormalise +
        `_unpatchify_latents`, in that order.  The unpack is a pure reshape because
        the kept tokens always carry the row-major latent grid ids (the reference
        scatter is the identity for them) -- `_assert_row_major_ids` checks that.
        """
        packed = int(x.shape[-1])
        mean, std = self._bn_vectors(packed)
        x = ttnn.add(ttnn.multiply(x, std), mean)

        selects, masks = self._unpatchify_helpers(h, w, channels, packed)
        acc = None
        for sel, mask in zip(selects, masks):
            plane = ttnn.matmul(x, sel, compute_kernel_config=_compute_config())  # (B, N, C)
            plane = ttnn.to_layout(plane, ttnn.ROW_MAJOR_LAYOUT)
            plane = ttnn.reshape(plane, (int(x.shape[0]), h, w, channels))
            plane = ttnn.upsample(plane, 2)  # (B, 2h, 2w, C), nearest
            plane = ttnn.to_layout(plane, ttnn.TILE_LAYOUT)
            plane = ttnn.multiply(plane, mask)
            acc = plane if acc is None else ttnn.add(acc, plane)
        return ttnn.permute(acc, (0, 3, 1, 2))

    def _nchw_to_latents(self, x, *, channels: int = 32):
        """(B, C, 2h, 2w) NCHW -> (B, h*w, 4C) packed tokens, all on device.

        `_patchify_latents` + BatchNorm normalise + `_pack_latents`: four strided
        slices on the sub-pixel lattice, concatenated and permuted into the model's
        channel order.
        """
        b, _, hh, ww = (int(d) for d in x.shape)
        h, w = hh // 2, ww // 2
        nhwc = ttnn.permute(x, (0, 2, 3, 1))  # (B, 2h, 2w, C)
        planes = []
        for p1 in (0, 1):
            for p2 in (0, 1):
                plane = ttnn.slice(nhwc, [0, p1, p2, 0], [b, hh, ww, channels], [1, 2, 2, 1])
                plane = ttnn.to_layout(plane, ttnn.ROW_MAJOR_LAYOUT)
                plane = ttnn.reshape(plane, (b, h * w, channels))
                planes.append(ttnn.to_layout(plane, ttnn.TILE_LAYOUT))
        stacked = ttnn.concat(planes, dim=-1)  # (B, N, 4C) in [p1][p2][c] order
        packed = int(stacked.shape[-1])
        out = ttnn.matmul(
            stacked, self._patchify_permutation(channels, packed), compute_kernel_config=_compute_config()
        )
        mean, std = self._bn_vectors(packed)
        return ttnn.divide(ttnn.subtract(out, mean), std)

    @staticmethod
    def _assert_row_major_ids(ids: torch.Tensor, h: int, w: int) -> None:
        """The reference unpack scatters tokens by their (h, w) ids; for the denoised
        latents those ids are exactly the row-major grid, so the scatter is the
        identity and the device unpack can be a reshape.  Fail loudly if that ever
        stops holding rather than silently producing a transposed image."""
        expect = L.latent_ids(1, h, w)[0]
        if not torch.equal(ids[0].to(expect.dtype), expect):
            raise AssertionError("latent ids are not the row-major grid; the device unpack would be wrong")

    def _euler_step(self, sample, model_output, dt: float):
        """`FlowMatchEulerDiscreteScheduler.step` with `stochastic_sampling: false`:
        upcast, `sample + dt * model_output`, cast back to the model dtype."""
        acc = ttnn.add(
            ttnn.typecast(sample, ttnn.float32),
            ttnn.multiply(ttnn.typecast(model_output, ttnn.float32), dt),
        )
        return ttnn.typecast(acc, ttnn.bfloat16)

    # ------------------------------------------------------------ head: T2I
    def run_text_to_image(
        self,
        prompt,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 8,
        max_sequence_length: int = 512,
        seed: int = 0,
        latents: torch.Tensor | None = None,
        staged: bool = True,
        return_intermediates: bool = False,
    ):
        """text -> image.  The real Flux2Klein task, chained over the graduated stubs."""
        batch = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device

        # --- input encoding (host): the pipeline's own chat template + padding
        input_ids, attention_mask = R.text_inputs(prompt, max_sequence_length)
        text_len = int(input_ids.shape[1])

        # --- TT stage 1: prompt embeddings (27 Qwen3 layers, taps at 9/18/27)
        prompt_embeds = self.text_stage()(input_ids, attention_mask)
        if staged:
            self.release_stage("encode_text")

        # --- latent grid + ids (host layout prep)
        lh, lw = L.latent_grid(height, width)
        txt_ids = L.text_ids(batch, text_len)
        img_ids = L.latent_ids(batch, lh, lw)
        self._assert_row_major_ids(img_ids, lh, lw)

        if latents is None:
            latents = R.make_latents(batch, height, width, seed)
        x = _replicate(L.pack_latents(latents.to(torch.bfloat16)), device)
        n_latent_tokens = int(x.shape[-2])

        timesteps, sigmas = L.schedule(self.scheduler, num_inference_steps, image_seq_len=n_latent_tokens)
        deltas = L.euler_deltas(sigmas)

        # --- TT stage 2: the denoise loop
        transformer = self.transformer_stage()
        steps = []
        for i, t in enumerate(timesteps):
            noise_pred = transformer(x, prompt_embeds, float(t) / 1000.0, img_ids, txt_ids)
            noise_pred = ttnn.slice(noise_pred, [0, 0, 0], [batch, n_latent_tokens, int(noise_pred.shape[-1])])
            if return_intermediates:
                steps.append(_to_torch(noise_pred, device).float())
            x = self._euler_step(x, noise_pred, deltas[i])
        if staged:
            self.release_stage("denoise")

        # --- on-device latent unpack + bn denormalise + unpatchify
        latent_nchw = self._latents_to_nchw(x, lh, lw)

        # --- TT stage 3: VAE decode (composite `decoder` port).
        # `AutoencoderKLFlux2._decode` is `decoder(post_quant_conv(z))` and the graduated
        # `decoder` port is the `decoder` SUBMODULE, so post_quant_conv is ours to apply.
        vae = self.vae_stage()
        image = vae.decode(vae.post_quant_conv(latent_nchw))
        self.ledger.mark_final(image)
        out = _to_torch(image, device).float()
        if return_intermediates:
            return out, {"prompt_embeds": prompt_embeds, "noise_preds": steps, "latents": latent_nchw}
        return out

    # ------------------------------------------------------------ head: edit
    def run_image_edit(
        self,
        prompt,
        images,
        *,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 2,
        max_sequence_length: int = 128,
        seed: int = 0,
        latents: torch.Tensor | None = None,
        staged: bool = True,
    ):
        """text + N reference images -> image (FLUX.2 multi-reference editing).

        Reference i is encoded by a DIFFERENT graduated route -- the composite
        `encoder`, its alias `encoder_stack`, then the block-wise decomposition --
        and every one of those latents is concatenated into the denoised stream, so
        each route's output feeds the final image.
        """
        batch = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device
        images = list(images)

        input_ids, attention_mask = R.text_inputs(prompt, max_sequence_length)
        text_len = int(input_ids.shape[1])

        # reference images: real preprocessing, then the three VAE encode routes.
        # `images` is a list of reference SLOTS, not of samples -- slot i is encoded by
        # route i and lands at rope coordinate T = 10*(i+1) for every row.  A slot may
        # hold one image (shared by the whole batch) or B images, one per sample, so
        # the 32 rows can carry 32 different references through the same three routes.
        vae = self.vae_stage()
        routes = [vae.encode, vae.encode_alias, vae.encode_blockwise]
        ref_tokens, ref_shapes = [], []
        for i, image in enumerate(images):
            pixel = _slot_pixels(image, batch, height, width)
            moments = routes[i % len(routes)](_replicate(pixel.to(torch.bfloat16), device))
            mode = vae.moments_to_mode(vae.quant_conv(moments))
            packed = self._nchw_to_latents(mode)  # patchify + bn normalise + pack
            ref_shapes.append((int(mode.shape[-2]) // 2, int(mode.shape[-1]) // 2))
            ref_tokens.append(packed)
        if staged:
            self.release_stage("vae")

        prompt_embeds = self.text_stage()(input_ids, attention_mask)
        if staged:
            self.release_stage("encode_text")

        lh, lw = L.latent_grid(height, width)
        txt_ids = L.text_ids(batch, text_len)
        latent_ids = L.latent_ids(batch, lh, lw)
        self._assert_row_major_ids(latent_ids, lh, lw)
        ref_ids = L.image_ids(ref_shapes).repeat(batch, 1, 1)
        img_ids = L.joint_image_ids(latent_ids, ref_ids)

        if latents is None:
            latents = R.make_latents(batch, height, width, seed)
        x = _replicate(L.pack_latents(latents.to(torch.bfloat16)), device)
        n_latent_tokens = int(x.shape[-2])
        ref_stream = ttnn.concat(ref_tokens, dim=-2) if len(ref_tokens) > 1 else ref_tokens[0]

        timesteps, sigmas = L.schedule(self.scheduler, num_inference_steps, image_seq_len=n_latent_tokens)
        deltas = L.euler_deltas(sigmas)

        transformer = self.transformer_stage()
        for i, t in enumerate(timesteps):
            model_input = ttnn.concat([x, ref_stream], dim=-2)
            noise_pred = transformer(model_input, prompt_embeds, float(t) / 1000.0, img_ids, txt_ids)
            noise_pred = ttnn.slice(noise_pred, [0, 0, 0], [batch, n_latent_tokens, int(noise_pred.shape[-1])])
            x = self._euler_step(x, noise_pred, deltas[i])
        if staged:
            self.release_stage("denoise")

        latent_nchw = self._latents_to_nchw(x, lh, lw)
        vae = self.vae_stage()  # post_quant_conv sits OUTSIDE the graduated `decoder` port
        image = vae.decode_alias(vae.post_quant_conv(latent_nchw))
        self.ledger.mark_final(image)
        return _to_torch(image, device).float()

    # -------------------------------------------------- head: text generation
    def run_text_generation(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        return_ids: bool = False,
        return_logits: bool = False,
    ):
        """text -> text.  Greedy decode of the Qwen3ForCausalLM head under the
        model's OWN stop rule (`generation_config.eos_token_id`), with
        `max_new_tokens` as the safety cap that both sides share.

        `return_logits` adds the per-step last-row logits -- a measurement tap for
        the PCC gate; the argmax and the id concat stay on device either way.

        `prompt` may be a LIST: the rows are left-padded into one `(B, L)` block and
        decoded in lockstep, one program per step for all B streams, each row cut at
        its OWN first stop id.  The loop runs until every row has stopped or the
        shared safety cap is reached.
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        stop_ids = R.stop_token_ids()
        if len(prompts) == 1:
            input_ids, attention_mask = R.chat_prompt_ids(prompts[0]), None
        else:
            input_ids, attention_mask, _ = R.chat_prompt_ids_batch(prompts)
        out = self.causal_lm_stage().generate(
            input_ids,
            max_new_tokens,
            stop_ids,
            collect_logits=return_logits,
            attention_mask=attention_mask,
        )
        new_ids, logits = out if isinstance(out, tuple) else (out, [])
        self.ledger.mark_final(new_ids)
        tokenizer = R.load_tokenizer()
        if isinstance(prompt, str):
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            text = [tokenizer.decode(row, skip_special_tokens=True) for row in new_ids]
        result = (text,)
        if return_ids:
            result += (new_ids,)
        if return_logits:
            result += (logits,)
        return result[0] if len(result) == 1 else result

    # ------------------------------------------------- head: VAE round trip
    def run_vae_roundtrip(self, image, *, height: int = 256, width: int = 256):
        """image -> image through the checkpoint's own latent codec, on the fully
        DECOMPOSED encode and decode routes (every VAE sub-block stub at its own
        position).  The decode is fed the encode's real output.

        `image` may be a LIST of B images: they are stacked on the leading axis and the
        VAE stage runs them in chunks of the width one program carries, joined on
        device (see `tt/vae.py::_CHUNK_START`)."""
        device = self.device
        pixel = _slot_pixels(image, None, height, width)
        vae = self.vae_stage()
        moments = vae.encode_decomposed(_replicate(pixel.to(torch.bfloat16), device))
        mode = vae.moments_to_mode(vae.quant_conv(moments))
        recon = vae.decode_decomposed(vae.post_quant_conv(mode))
        self.ledger.mark_final(recon)
        return _to_torch(recon, device).float()

    # =========================================================== trace contract
    #
    # Per stage: <stage>_trace_setup(inputs) pins the variable dim to a fixed
    # capacity C and pre-uploads the padded input plus every shape-dependent
    # constant into persistent buffers OUTSIDE the trace; <stage>_trace_step() is
    # one host-op-free forward at that fixed shape reading only those buffers;
    # <stage>_trace_inputs() is the zero-arg seam that assembles exactly the value
    # _trace_setup takes; <stage>_trace_items() states how many items one step
    # retires, which is the only input to the stage's arithmetic ceiling.

    _DEFAULT_CAPACITY = {
        "encode_text": 128,
        "prefill": 128,
        "decode": 128,
        "denoise": 128 + 256,  # text capacity + packed latent tokens for 256x256
        # 256 / 32: the smallest grid ttnn.group_norm's DRAM rule allows here, and the
        # shape the graduated VAE stubs were PCC'd at
        "vae_encode": 256,
        "vae_decode": 32,
    }

    def capacity(self, stage: str) -> int:
        return int(self.trace_capacity.get(stage, self._DEFAULT_CAPACITY[stage]))

    # ---- inputs seams (model-specific assembly behind a fixed, generic name)
    def encode_text_trace_inputs(self):
        """The same real text input the e2e test and the demo use, at this pipeline's
        batch width -- `self.batch` DISTINCT prompts, so the traced step is the one the
        heads actually run and not a narrower single-sample shape."""
        ids, mask = R.text_inputs(R.batch_prompts(self.batch), self.capacity("encode_text"))
        return {"input_ids": ids, "attention_mask": mask}

    def prefill_trace_inputs(self):
        """The causal-LM head's prefill takes the same real text input the
        prompt-embed trunk does -- one checkpoint, one tokenizer, one chat template --
        so this assembles it the same way, at the prefill stage's own capacity.

        Spelled out as its own method rather than aliased to
        `encode_text_trace_inputs`: the perf engine binds this seam by NAME, and an
        alias is not a definition to anything reading the source.
        """
        ids, mask = R.text_inputs(R.batch_text_prompts(self.batch), self.capacity("prefill"))
        return {"input_ids": ids, "attention_mask": mask}

    def decode_trace_inputs(self):
        ids, mask, _ = R.chat_prompt_ids_batch(R.batch_text_prompts(self.batch))
        return {"input_ids": ids, "attention_mask": mask}

    def denoise_trace_inputs(self):
        """The stage's inputs at this stage's pinned capacity.

        The bring-up captured PER COMPONENT, so there is no captured tensor at the
        transformer's own input boundary -- `_captured/flux2_transformer_block` holds
        post-`x_embedder` activations (4096 wide).  What IS taken from the captures
        is the shape law (`3 * 4096 == joint_attention_dim`) and the reference
        timestep; the image side is built the way the pipeline builds it, from the
        seeded initial latents and the real position ids, so this is the same input
        the e2e test and the demo feed.
        """
        cap = self.capacity("denoise")
        text_len = self.capacity("encode_text")
        n_img = cap - text_len
        if n_img <= 0:
            raise ValueError(
                f"denoise capacity {cap} leaves no image tokens once {text_len} text tokens are "
                f"taken out -- the joint length must stay text + grid**2 (see _smaller_capacity)"
            )
        grid = int(round(n_img**0.5))
        if grid * grid != n_img:
            raise ValueError(f"denoise capacity {cap} - text {text_len} = {n_img} is not a square grid")

        captured = torch.load(f"{captured_dir('transformer', 'flux2_transformer_block')}/kwargs.pt", weights_only=False)
        width = int(captured["hidden_states"].shape[-1])  # 4096 == inner_dim
        timestep = torch.load(
            f"{captured_dir('transformer', 'flux2_timestep_guidance_embeddings')}/args.pt",
            weights_only=False,
        )[0]

        size = grid * 2 * L.VAE_SCALE_FACTOR  # the image size this token count comes from
        b = self.batch
        hidden = L.pack_latents(R.batch_latents(b, size, size, 0))
        gen = torch.Generator("cpu").manual_seed(0)
        return {
            "hidden_states": hidden,
            "encoder_hidden_states": torch.randn(b, text_len, 3 * width, generator=gen, dtype=torch.float32).to(
                torch.bfloat16
            ),
            # ONE timestep for the whole batch: the 32 samples share the resolution and
            # the step count, so they share the flow-match schedule.  This is the axis
            # that legitimately stays 1 and broadcasts.
            "timestep": float(timestep.reshape(-1)[0]),
            "img_ids": L.latent_ids(b, grid, grid),
            "txt_ids": L.text_ids(b, text_len),
        }

    def vae_encode_trace_inputs(self):
        """The captured HF-golden image, zero-padded to this stage's pinned grid.

        The bring-up captured a 224x224 sample but `ttnn.group_norm`'s DRAM grid rule
        makes 224 unrunnable here (its 28x28 latent gives Ht=25, which no core grid
        divides), so the values come from the capture and the SHAPE comes from the
        pinned capacity -- which is what trace capture needs.
        """
        args = torch.load(f"{captured_dir('vae', 'encoder')}/args.pt", weights_only=False)
        return {"pixel_values": _tile_batch(_pad_spatial(args[0], self.capacity("vae_encode")), self.batch)}

    def vae_decode_trace_inputs(self):
        args = torch.load(f"{captured_dir('vae', 'decoder')}/args.pt", weights_only=False)
        return {
            "latents": _tile_batch(_pad_spatial(args[0], self.capacity("vae_decode")), self.stage_batch("vae_decode"))
        }

    # ---- item counts (the arithmetic ceiling is 2 x params x items)
    #
    # Each of these is the TOTAL one call of `<stage>_trace_step` retires, BATCH
    # INCLUDED -- a step that runs 32 samples at once retires 32x the items of a
    # single-sample step, and a stage that under-reports is handed a compute roof that
    # much too small and then mis-diagnosed as memory-bound.
    def encode_text_trace_items(self):
        return self.batch * self.capacity("encode_text")

    def prefill_trace_items(self):
        return self.batch * self.capacity("prefill")

    def decode_trace_items(self):
        return self.batch  # one token per stream per step, `batch` streams in lockstep

    def denoise_trace_items(self):
        # every joint token of every sample goes through all 32 blocks
        return self.batch * self.capacity("denoise")

    def vae_encode_trace_items(self):
        c = self.capacity("vae_encode")
        return self.batch * (c // 8) * (c // 8)  # latent positions retired

    def vae_decode_trace_items(self):
        c = self.capacity("vae_decode")
        return self.stage_batch("vae_decode") * c * c

    # ---- setup / step
    def encode_text_trace_setup(self, inputs):
        cap = self.capacity("encode_text")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        ids, mask = _pad_ids(ids, mask, cap)
        stage = self.text_stage()
        state = {"stage": stage, "resident": stage.pin(ids, mask)}
        self._trace_state["encode_text"] = state
        return state

    def encode_text_trace_step(self):
        state = self._trace_state["encode_text"]
        return state["stage"].step(state["resident"])

    def prefill_trace_setup(self, inputs):
        cap = self.capacity("prefill")
        ids, mask = _pad_ids(inputs["input_ids"], inputs.get("attention_mask"), cap)
        stage = self.causal_lm_stage()
        state = {"stage": stage, "resident": stage.pin_prefill(ids, mask)}
        self._trace_state["prefill"] = state
        return state

    def prefill_trace_step(self):
        state = self._trace_state["prefill"]
        return state["stage"].prefill_step(state["resident"])

    def decode_trace_setup(self, inputs):
        cap = self.capacity("decode")
        ids, mask = _pad_ids(inputs["input_ids"], inputs.get("attention_mask"), cap)
        stage = self.causal_lm_stage()
        state = {"stage": stage, "resident": stage.pin_decode(ids, mask, cap)}
        self._trace_state["decode"] = state
        return state

    def decode_trace_step(self):
        state = self._trace_state["decode"]
        return state["stage"].decode_step(state["resident"])

    def denoise_trace_setup(self, inputs):
        stage = self.transformer_stage()
        state = {"stage": stage, "resident": stage.pin(self.device, **inputs)}
        self._trace_state["denoise"] = state
        return state

    def denoise_trace_step(self):
        state = self._trace_state["denoise"]
        return state["stage"].step(state["resident"])

    def vae_encode_trace_setup(self, inputs):
        stage = self.vae_stage()
        state = {"stage": stage, "resident": stage.pin_encode(inputs["pixel_values"])}
        self._trace_state["vae_encode"] = state
        return state

    def vae_encode_trace_step(self):
        state = self._trace_state["vae_encode"]
        return state["stage"].encode_step(state["resident"])

    def vae_decode_trace_setup(self, inputs):
        stage = self.vae_stage()
        state = {"stage": stage, "resident": stage.pin_decode(inputs["latents"])}
        self._trace_state["vae_decode"] = state
        return state

    def vae_decode_trace_step(self):
        state = self._trace_state["vae_decode"]
        return state["stage"].decode_step(state["resident"])

    # ---------------------------------------------------------- self tests
    #: A capture that overflows the device's trace region says so in the message; the
    #: answer is a smaller pinned capacity, not a dropped stage.  (What runs out here
    #: is the trace REGION; every stage's parameters are staged once at build time and
    #: stay resident for the whole run, at every capacity.)
    _OVERFLOW_MARKERS = (
        "trace_region",
        "trace region",
        "Out of Memory",
        "out of memory",
        "Not enough space",
        "TT_THROW",
        "allocate",
    )

    #: Allocator failures a smaller pinned capacity CANNOT fix, because what ran out is
    #: not the trace region.  An exhausted L1_SMALL halo reservation and a circular-buffer
    #: clash are fixed costs of the step's convolutions at ANY capacity: the denoise
    #: capture failed asking for 16 B per bank of an L1_SMALL region that was already
    #: 24576/24576 full, and retrying it at 192, 144 and 132 spent three more captures to
    #: read back the identical message.  Naming them here keeps the ladder for the one
    #: failure it is for, and lets the real cause reach the report unshrunken.
    _NOT_A_REGION_OVERFLOW = ("L1_SMALL buffer", "circular buffer", "Statically allocated circular")

    def _is_region_overflow(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}"
        if any(m in text for m in self._NOT_A_REGION_OVERFLOW):
            return False
        return any(m in text for m in self._OVERFLOW_MARKERS)

    #: Stages whose pinned capacity is a JOINT length -- text tokens plus a SQUARE image
    #: grid -- mapped to the stage that owns the text half.  Their capacity cannot simply
    #: be halved: `denoise` at 384 is 128 text + 16x16 image, and halving the total to 96
    #: asks for -32 image tokens, whose square root is complex.  These shrink by halving
    #: the grid SIDE, which is the only move that keeps `cap == text + grid**2` true.
    _JOINT_CAPACITY = {"denoise": "encode_text"}

    #: A grid side below this is not a smaller image, it is an absent one: the VAE packs
    #: 2x2 latent patches, so side 1 leaves a single patch with no spatial structure for
    #: the position ids to index.
    _MIN_GRID_SIDE = 2

    def _smaller_capacity(self, stage: str):
        """The next pinned capacity DOWN for a stage whose capture did not fit, or None
        when no smaller shape still satisfies that stage's own shape law."""
        current = self.capacity(stage)
        text_stage = self._JOINT_CAPACITY.get(stage)
        if text_stage is None:
            smaller = current // 2
            return smaller if smaller >= 32 else None
        text_len = self.capacity(text_stage)
        n_img = current - text_len
        if n_img <= 0:
            return None
        side = int(round(n_img**0.5)) // 2
        if side < self._MIN_GRID_SIDE:
            return None
        return text_len + side * side

    def trace_capture_selftest(self, device=None, stages=None, pcc_target: float = 0.99):
        """Capture ONE step per stage in begin/end_trace_capture, execute it, PCC it
        against the eager step, then RELEASE the trace before the next stage.

        The trace region is sized from the LARGEST stage (pinned capacity x layers).
        If a capture still overflows it, the stage's capacity C is HALVED and the
        capture retried, and the fallback is PRINTED -- a stage is never silently
        dropped, and the report carries the capacity actually used alongside the one
        it was asked for.
        """
        device = device or self.device
        report = {}
        for stage in stages or PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup", None)
            step = getattr(self, f"{stage}_trace_step", None)
            seam = getattr(self, f"{stage}_trace_inputs", None)
            if not (setup and step and seam):
                report[stage] = {"ok": False, "reason": "stage does not expose the trace contract"}
                continue
            asked = self.capacity(stage)
            entry = {}
            attempts = []
            while True:
                entry = {"capacity": self.capacity(stage), "items": getattr(self, f"{stage}_trace_items")()}
                if entry["capacity"] != asked:
                    entry["requested_capacity"] = asked
                if attempts:
                    entry["attempts"] = list(attempts)
                tid = None
                ended = False
                try:
                    setup(seam())
                    eager = _to_torch(step(), device).float()
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    traced = step()
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    ended = True
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                    got = _to_torch(traced, device).float()
                    entry["pcc"] = R.pcc(eager, got)
                    entry["ok"] = entry["pcc"] >= pcc_target
                    break
                except Exception as exc:  # noqa: BLE001
                    entry["ok"] = False
                    entry["reason"] = f"{type(exc).__name__}: {exc}"[:400]
                    # keep EVERY attempt: the retry used to overwrite `reason`, so a stage
                    # that failed at its real capacity for one cause and then again at a
                    # shrunken one for another reported only the second, and the cause
                    # that actually mattered was gone from the report.
                    attempts.append({"capacity": entry["capacity"], "reason": entry["reason"]})
                    entry["attempts"] = list(attempts)
                    smaller = self._smaller_capacity(stage)
                    if not self._is_region_overflow(exc) or smaller is None:
                        break
                    # never a silent drop: say what shrank and why
                    print(
                        f"trace fallback: {stage} did not fit the trace region at capacity "
                        f"{self.capacity(stage)} ({type(exc).__name__}: {exc}); retrying at {smaller}"
                    )
                    self.trace_capacity[stage] = smaller
                    self.release_stage()
                finally:
                    # A capture that raised BETWEEN begin and end leaves the device in
                    # capture mode with its trace buffer still allocated.  Every later
                    # stage then runs against an allocator that never got that region
                    # back, which is how one stage's failure turned into an out-of-memory
                    # for all of the stages after it (and for the next test to share the
                    # device).  Close and release on every path, including the failing one.
                    if tid is not None:
                        if not ended:
                            with contextlib.suppress(Exception):
                                ttnn.end_trace_capture(device, tid, cq_id=0)
                        with contextlib.suppress(Exception):
                            ttnn.release_trace(device, tid)
            report[stage] = entry
            self.release_stage()
        report["all_ok"] = all(v.get("ok") for k, v in report.items() if k != "all_ok")
        return report

    def host_op_selftest(self, heads=None, **head_kwargs):
        """The authoritative fully-on-device check: run each head's model math under
        `observe_host_ops()` with input encoding and weight build done OUTSIDE the
        observed region, and return the observer's verdict per head."""
        import sys

        sys.path.insert(0, R.__file__.rsplit("/models/", 1)[0])
        from scripts.tt_hw_planner import host_op_observer

        report = {}
        for name, thunk in (heads or self._observable_heads(**head_kwargs)).items():
            prepared = thunk["prepare"]()
            # ONE warm-up forward, still OUTSIDE the observed region.  Several routes
            # prepare their conv weights lazily on first use (ttnn's conv/group-norm
            # weight prep is host torch work), and that is weight build, not model
            # math -- observing it would report a fully-on-device forward as host
            # compute.  The observed call below is a steady-state forward.
            thunk["forward"](prepared)
            with host_op_observer.observe_host_ops() as ops:
                thunk["forward"](prepared)
            report[name] = host_op_observer.verdict(list(ops))
            self.release_stage()
        report["all_on_device"] = all(v["on_device"] for v in report.values())
        return report

    def _observable_heads(self, height=256, width=256, num_inference_steps=1, max_sequence_length=128, image=None):
        """Each head split into (encoding + weight build) and (pure device math)."""
        device = self.device

        b = self.batch

        def prep_t2i():
            ids, mask = R.text_inputs(R.batch_prompts(b), max_sequence_length)
            lh, lw = L.latent_grid(height, width)
            latents = R.batch_latents(b, height, width, 0)
            packed = _replicate(L.pack_latents(latents.to(torch.bfloat16)), device)
            timesteps, sigmas = L.schedule(self.scheduler, num_inference_steps, image_seq_len=lh * lw)
            text = self.text_stage()
            resident_text = text.pin(ids, mask)
            transformer = self.transformer_stage()
            vae = self.vae_stage()
            self._unpatchify_helpers(lh, lw, 32, 128)
            self._bn_vectors(128)

            # The denoise stage's ENCODING -- the position-id tables and the per-step
            # timestep scalar -- is pinned here, once per step, exactly as the trace
            # contract requires.  `Flux2TransformerStage.pin` is where the last two
            # host ops of this head live (a bf16 round of the timestep and an `ids[0]`
            # select), and they are per-step input encoding, not model math.
            #
            # The residents carry the WARM-UP prompt embeddings.  That is the text
            # stage's own deterministic output, not a golden -- the real chain in
            # `run_text_to_image` feeds `text.step`'s live tensor straight into the
            # transformer, and Gate 3 is what proves that wiring.  Here the point is
            # only the host-op verdict.
            warm_embeds = text.step(resident_text)
            txt_ids = L.text_ids(b, int(ids.shape[1]))
            img_ids = L.latent_ids(b, lh, lw)
            residents = [
                transformer.pin(
                    device,
                    hidden_states=packed,
                    encoder_hidden_states=warm_embeds,
                    timestep=float(t) / 1000.0,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                )
                for t in timesteps
            ]
            return {
                "text": text,
                "resident_text": resident_text,
                "transformer": transformer,
                "vae": vae,
                "residents": residents,
                "x": packed,
                "lh": lh,
                "lw": lw,
                "dt": L.euler_deltas(sigmas),
            }

        def fwd_t2i(p):
            p["text"].step(p["resident_text"])  # the text stage's real device forward
            x = p["x"]
            n = int(x.shape[-2])
            rows = int(x.shape[0])
            for i, resident in enumerate(p["residents"]):
                pred = p["transformer"].step(resident)
                pred = ttnn.slice(pred, [0, 0, 0], [rows, n, int(pred.shape[-1])])
                x = self._euler_step(x, pred, p["dt"][i])
            vae = p["vae"]
            return vae.decode(vae.post_quant_conv(self._latents_to_nchw(x, p["lh"], p["lw"])))

        def prep_text():
            ids, mask, _ = R.chat_prompt_ids_batch(R.batch_text_prompts(b))
            stage = self.causal_lm_stage()
            return {"stage": stage, "resident": stage.pin_prefill(ids, mask)}

        def fwd_text(p):
            return p["stage"].prefill_step(p["resident"])

        def prep_vae():
            img = image if image is not None else R.batch_images(b, 256)
            pixel = _slot_pixels(img, None, 256, 256)
            stage = self.vae_stage()
            self._patchify_permutation(32, 128)
            self._bn_vectors(128)
            return {"stage": stage, "x": _replicate(pixel.to(torch.bfloat16), device)}

        def fwd_vae(p):
            stage = p["stage"]
            moments = stage.encode_decomposed(p["x"])
            mode = stage.moments_to_mode(stage.quant_conv(moments))
            return stage.decode_decomposed(stage.post_quant_conv(mode))

        return {
            "text_to_image": {"prepare": prep_t2i, "forward": fwd_t2i},
            "text_generation": {"prepare": prep_text, "forward": fwd_text},
            "vae_roundtrip": {"prepare": prep_vae, "forward": fwd_vae},
        }


def _slot_pixels(images, batch, height: int, width: int) -> torch.Tensor:
    """`host_inputs.slot_pixels` bound to this pipeline's image preprocessor.

    The body lives in `host_inputs` with the rest of the host-side input encoding --
    resize/crop/scale and the row `cat` are prep, not forward path, and `tt/` holds
    only the forward path.
    """
    return L.slot_pixels(images, batch, height, width, R.preprocess_image)


def _pad_ids(ids: torch.Tensor, mask: torch.Tensor | None, capacity: int):
    """`host_inputs.pad_ids` at this checkpoint's pad id."""
    return L.pad_ids(ids, mask, capacity, R.pad_token_id())


def _tile_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    """Widen a captured `(1, ...)` reference tensor to the pipeline's batch.

    The bring-up captured ONE sample, so the batched trace input repeats it.  That is
    fine here and only here: a trace step is measured for its SHAPE and its
    host-op-freeness, not for telling samples apart.  What proves the batch axis
    carries 32 INDEPENDENT samples is the PCC gate, which feeds 32 distinct real
    inputs and scores each row against its own golden.
    """
    b = int(batch)
    if int(x.shape[0]) == b:
        return x
    if int(x.shape[0]) != 1:
        raise ValueError(f"cannot widen a captured tensor with leading dim {x.shape[0]} to {b}")
    return x.repeat(b, *([1] * (x.dim() - 1)))


def _pad_spatial(x: torch.Tensor, size: int) -> torch.Tensor:
    """Crop or zero-pad an NCHW tensor's H and W to `size` (pure layout prep)."""
    out = torch.zeros(x.shape[0], x.shape[1], size, size, dtype=x.dtype)
    h = min(int(x.shape[-2]), size)
    w = min(int(x.shape[-1]), size)
    out[:, :, :h, :w] = x[:, :, :h, :w]
    return out


def _demo_image(size: int = 256):
    """A deterministic RGB test image, so a demo/selftest run needs no asset."""
    from PIL import Image

    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = (xx * 255 // max(size - 1, 1)).to(torch.uint8)
    g = (yy * 255 // max(size - 1, 1)).to(torch.uint8)
    b = (((xx // 16 + yy // 16) % 2) * 200 + 27).to(torch.uint8)
    return Image.fromarray(torch.stack([r, g, b], dim=-1).numpy(), mode="RGB")


#: The depth the two module-level self-tests run at, and the trace region that fits
#: it.  `tests/e2e/test_trace_contract.py` uses the same cap, so these hooks run the
#: configuration that is actually under test.  What they prove -- that one step reads
#: only resident buffers, fires no host op, and replays identically from a captured
#: trace -- is a property of the step, not of how many times its block repeats; the
#: per-stage tests observe every stage at FULL depth, and how deep trace is MEASURED
#: is optimize's business (it sizes the region itself).
SELFTEST_LAYERS = 4
TRACE_SELFTEST_REGION = 90 * 1024 * 1024


def _merge_host_op_verdicts(report: dict) -> dict:
    """One verdict for the whole pipeline out of the per-head reports.

    The observer's own shape (`on_device` / `host_ops` / `n_host_ops` / `reason`), so a
    caller that has a single head's verdict and a caller that has the pipeline's read
    the same fields.  A head that fires a host op names it here: an aggregate that only
    said "false" would leave the next reader to re-run the whole thing to find out which.
    """
    heads = {name: v for name, v in report.items() if name != "all_on_device"}
    host_ops = sorted({op for v in heads.values() for op in (v.get("host_ops") or [])})
    offenders = sorted(name for name, v in heads.items() if not v.get("on_device"))
    return {
        "on_device": bool(report.get("all_on_device")) and not host_ops,
        "host_ops": host_ops,
        "n_host_ops": len(host_ops),
        "heads": {
            name: {"on_device": bool(v.get("on_device")), "n_host_ops": int(v.get("n_host_ops", 0))}
            for name, v in sorted(heads.items())
        },
        "reason": (
            "fully on device: %d head(s) ran with no host aten op" % len(heads)
            if not host_ops
            else "host compute in %s: %s" % (", ".join(offenders), ", ".join(host_ops[:12]))
        ),
    }


def host_op_selftest(device=None, *, layers=SELFTEST_LAYERS, **head_kwargs) -> dict:
    """MODULE-LEVEL, zero-arg: is the pipeline's forward path fully on device?

    This is the seam the emit-e2e host-op observer calls
    (`scripts/tt_hw_planner/_host_op_probe.py` imports `tt.pipeline` and calls
    `host_op_selftest()` with no arguments), so it has to be able to answer without a
    device being handed to it.  When one is passed it is used as-is; when none is, the
    RUNNER-side opener in `models/demos/flux_2_klein_9b/mesh.py` provides the same 1x8
    mesh the demos and the e2e fixtures use.  That opener deliberately lives outside
    `tt/`: nothing in the pipeline package opens a device, and this function is an
    entry point into it, not a stage of it.

    What this hook adds is the CHAIN: each head's stages observed together, one head's
    device output feeding the next stage, at `SELFTEST_LAYERS`.  Depth is not what it
    is measuring -- every stage is separately observed at FULL depth, inside the
    observed region, by `tests/e2e/test_stage_transformer.py::test_step_is_host_op_free`,
    `test_stage_text_encoder.py::test_pinned_step_is_host_op_free` and
    `test_stage_vae.py::test_trace_steps_host_op_free`.  Pass `layers=None` to run the
    chain at full depth too.

    Returns the observer's verdict dict for the pipeline as a whole; the per-head
    breakdown is under `"heads"`.
    """
    if device is not None:
        return _merge_host_op_verdicts(build_pipeline(device, layers=layers).host_op_selftest(**head_kwargs))

    from models.demos.flux_2_klein_9b.mesh import VAE_L1_SMALL, open_flux_mesh

    R.ensure_flux_imports()
    # the observed heads include `vae_roundtrip`, which holds the encoder's conv
    # programs and the decoder's at once -- see `mesh.VAE_L1_SMALL`
    with open_flux_mesh(l1_small_size=VAE_L1_SMALL) as mesh:
        return _merge_host_op_verdicts(build_pipeline(mesh, layers=layers).host_op_selftest(**head_kwargs))


def trace_capture_selftest(device=None, *, stages=None, pcc_target: float = 0.99, layers=SELFTEST_LAYERS, report=False):
    """MODULE-LEVEL, zero-arg: does a REAL trace capture of every stage succeed?

    The seam `scripts/tt_hw_planner/_trace_capture_probe.py` calls (it imports
    `tt.pipeline`, calls `trace_capture_selftest()` with no arguments and takes the
    result as a bool), so it must be able to open its own mesh -- with a trace region,
    which is the whole point -- via the runner-side opener in `mesh.py`.

    Per stage: pin outside the capture, run one eager step, capture and replay one
    step inside begin/end_trace_capture, PCC the two, release the trace.  Returns
    True only if every stage did all of that; pass `report=True` for the per-stage
    dict (that is what the e2e test asserts on).
    """
    if device is not None:
        got = build_pipeline(device, layers=layers).trace_capture_selftest(device, stages=stages, pcc_target=pcc_target)
        return got if report else bool(got.get("all_ok"))

    from models.demos.flux_2_klein_9b.mesh import VAE_L1_SMALL, open_flux_mesh

    R.ensure_flux_imports()
    # `vae_encode` and `vae_decode` are both in the walk, so the same halo the e2e
    # VAE runners take (see `mesh.VAE_L1_SMALL`)
    with open_flux_mesh(l1_small_size=VAE_L1_SMALL, trace_region_size=TRACE_SELFTEST_REGION) as mesh:
        got = build_pipeline(mesh, layers=layers).trace_capture_selftest(mesh, stages=stages, pcc_target=pcc_target)
    return got if report else bool(got.get("all_ok"))


def build_pipeline(
    device,
    model=None,
    layers=None,
    *,
    encode_text_layers=None,
    vae_encode_layers=None,
    denoise_layers=None,
    vae_decode_layers=None,
    prefill_layers=None,
    decode_layers=None,
    batch: int = BATCH,
    **kwargs,
) -> Flux2KleinTtPipeline:
    """CONSTRUCT AND RETURN the resident pipeline object -- the single entry the perf
    harness calls to obtain the object carrying PIPELINE_STAGES and the per-stage
    trace hooks.  It does not run anything, and it stages no weight (see
    `Flux2KleinTtPipeline._stage`).

    `layers` is the default depth for every repeated block (None = all layers, never
    0, and never below `depth.MIN_DISCOVERABLE_STACK`).  This checkpoint has five
    sections, so ONE number is not enough: each stage of PIPELINE_STAGES that owns a
    repeated stack takes its own `<stage>_layers` override, and each is spelled out in
    this signature rather than left to `**kwargs` -- a caller (or a tool) can only
    discover a knob that is actually declared, and a docstring promise is not a knob.
    `None` on any of them falls back to `layers`.

    `batch` is the leading-axis width the trace contract and the self-tests drive --
    32 independent samples per call.  It is a DEFAULT, not a constraint: a head called
    with N prompts runs at N.

    Remaining demo kwargs (prompt, height, ...) are accepted and ignored: the resident
    build takes its shapes from the config, not from a prompt.
    """
    return Flux2KleinTtPipeline(
        device,
        model=model,
        layers=layers,
        encode_text_layers=encode_text_layers,
        vae_encode_layers=vae_encode_layers,
        denoise_layers=denoise_layers,
        vae_decode_layers=vae_decode_layers,
        prefill_layers=prefill_layers,
        decode_layers=decode_layers,
        batch=batch,
        **kwargs,
    )
