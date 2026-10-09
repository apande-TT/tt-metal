# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN forward of Qwen-Image-Edit (image + instruction -> edited image).

Both demo/demo_image_edit.py and tests/e2e/test_e2e_image_edit.py call `run_image_edit`, which is the
HF QwenImageEditPipeline.__call__ chain on the graduated ports:

    encode_inputs (host, tt/inputs.py)   processor, image resize, seeded noise, sigma table
    vision_encode   qwen2.5-VL vision tower on the condition image
    text_encode     token embed + image embeddings -> 28-layer LM -> drop the 64 template tokens
    vae_encode      VAE encoder -> quant_conv -> mode -> normalise -> 2x2 pack
    denoise x N     cat[latents, image latents] -> 60-block transformer -> FlowMatch Euler step
    vae_decode      unpack -> denormalise -> post_quant_conv -> VAE decoder -> image in [0, 1]

N is the scheduler's own schedule (num_inference_steps timesteps); nothing caps it. Everything between
the uploaded encoded inputs and the output image runs on device.
"""

from __future__ import annotations

import contextlib
import time

import torch

import ttnn
from models.demos.qwen_image_edit.tt import inputs as I
from models.demos.qwen_image_edit.tt import lanes_ttl, round_ttl
from models.demos.qwen_image_edit.tt.guard_kernel import guard_tail, split_limbs, sum_parts, vote_tail
from models.demos.qwen_image_edit.tt.text_encoder import TtQwenTextEncoder
from models.demos.qwen_image_edit.tt.transformer import TtQwenImageTransformer
from models.demos.qwen_image_edit.tt.vae import TtQwenVAE
from models.demos.qwen_image_edit_text_encoder._stubs import attention as _te_attention
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs import _resident as _vae_resident

# Stages from the HF reference (model_index.json): the Qwen2.5-VL text_encoder is a vision tower + an
# LM, the vae encodes the condition image and decodes the result, and the transformer is the step the
# scheduler repeats.
PIPELINE_STAGES = ["vision_encode", "text_encode", "vae_encode", "denoise", "vae_decode"]

_SCHED_COLS = 32  # schedule table width: col 0 = t / 1000, col 1 = sigma_next - sigma


def _replicated(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, **kw)


def to_host(t, device):
    """Replicated device tensor -> torch (the first device's copy)."""
    if isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[: t.shape[0]]
    return ttnn.to_torch(t)


_HF = {}


def load_hf_pipeline():
    """The HF reference (float32; the checkpoint is bf16, so this is an exact upcast). The TT build reads
    its config and weights; it is also the ground truth for section structure."""
    if "pipe" not in _HF:
        from diffusers import QwenImageEditPipeline

        _HF["pipe"] = QwenImageEditPipeline.from_pretrained(I.MODEL_ID, torch_dtype=torch.float32)
    return _HF["pipe"]


class Uploaded:
    """Device-resident encoded inputs of one call."""


class DenoiseState:
    """Persistent device buffers of the denoising loop (the trace reads only these)."""


class TtQwenImageEditPipeline:
    def __init__(
        self,
        device,
        hf=None,
        layers=None,
        vision_encode_layers=None,
        text_encode_layers=None,
        denoise_layers=None,
        tracker=None,
        precise=True,
    ):
        self.device = device
        # the batch every <stage>_trace_inputs() builds (the perf adapter reads it back via resolve_batch)
        self.batch_size = I.batch_size_from_env()
        self.hf = hf if hf is not None else load_hf_pipeline()
        self.tracker = tracker
        # Exact-lane products of every precise port (text encoder, transformer, VAE encoder convs) in
        # their "guarded" form: exact lanes on the leading bf16 limb only, glitch-guarded by the dense
        # bound (see text-encoder _stubs/attention.py). Measured on this T3K against the HF float32
        # golden, B=2 then B=32: prompt embeddings 2.20e-4 -> 2.07e-4 rel, image latents 8.56e-6 ->
        # 8.53e-6, one teacher-forced denoise step's velocity 1.3e-4..4.0e-4 -> 1.8e-4..3.5e-4 (the same
        # floor), while B=32 per-pass cost fell vision 68 -> 14 s, text 86 -> 24 s, VAE encode 228 -> 144 s,
        # denoise step 257 -> 83 s. The stubs' own default stays "median3" (their PCC tests unchanged).
        if precise:
            _te_attention.EXACT_MODE = "guarded"
        # the guarded tail (~9 float32 passes per precise product) as one fused kernel, bit-identical, while
        # the text encoder runs (see _fused_tail)
        self.fused_tail = guard_tail if precise else None
        # and their lead limb's 8 exact lanes formed in one pass
        self.lane_split = lanes_ttl.lanes8 if precise else None
        # and their trailing (lo) limb's product -- and the guard's tie-break product -- at HiFi2
        self.lo_fidelity = ttnn.MathFidelity.HiFi2 if precise else ttnn.MathFidelity.HiFi4
        # and their TP all-reduce's sum of the gathered float32 partials in one pass (no slice copies)
        self.sum_parts = sum_parts if precise else None
        # and their inputs' two bf16 limbs in one pass
        self.split_limbs = split_limbs if precise else None
        # and their batched attention products on the full core grid
        self.bmm_config = _te_attention.bmm_program_config if precise else None
        if precise:
            lanes_ttl.prepare(device)
        # likewise the VAE encoder's exact-conv vote tail (~8 float32 passes per conv) while it encodes
        self.vote_tail = vote_tail if precise else None
        # and its hi limbs' bf16 rounding, one pass instead of two typecasts
        self.round_bf16 = round_ttl.round_bf16 if precise else None
        if precise:
            round_ttl.prepare(device)
        pick = lambda v: layers if v is None else v  # noqa: E731
        self.text_encoder = TtQwenTextEncoder(
            device,
            self.hf.text_encoder,
            vision_layers=pick(vision_encode_layers),
            text_layers=pick(text_encode_layers),
            tracker=tracker,
        )
        self.transformer = TtQwenImageTransformer(
            device, self.hf.transformer, layers=pick(denoise_layers), tracker=tracker, precise=precise
        )
        # The VAE encoder / decoder are not homogeneous repeats (every level changes channels and
        # resolution), so a depth cap leaves them whole.
        self.vae = TtQwenVAE(device, self.hf.vae, tracker=tracker)
        self._trace = {}
        self._active_stage = None
        self.stage_times = {}

    # ---- encoded inputs -> device (outside the forward) --------------------------------------------
    def upload_inputs(self, enc: I.EncodedInputs):
        d = self.device
        up = Uploaded()
        up.enc = enc
        up.B = enc.batch
        up.te = self.text_encoder.prepare(enc.vl)
        up.vae_image = _replicated(d, enc.vae_image)
        up.latents = _replicated(d, enc.latents)
        up.latent_hw = enc.latent_hw
        up.img_shapes = enc.img_shapes
        up.timesteps = enc.timesteps.tolist()  # host floats for the step log (iterating a tensor is aten.unbind)
        # FlowMatchEulerDiscreteScheduler.step: x += (sigma_next - sigma) * v, all float32; the
        # transformer gets t / 1000 (QwenImageEditPipeline passes timestep / 1000)
        n = len(enc.timesteps)
        sched = torch.zeros(n, _SCHED_COLS, dtype=torch.float32)
        sched[:, 0] = enc.timesteps / 1000
        sched[:, 1] = enc.sigmas[1 : n + 1] - enc.sigmas[:n]
        up.sched = sched
        return up

    # ---- stages (device only) ----------------------------------------------------------------------
    @staticmethod
    @contextlib.contextmanager
    def _hook(module, name, fn):
        """module.name = fn for the duration (a port's optional fused-kernel hook)."""
        prev = getattr(module, name)
        setattr(module, name, fn)
        try:
            yield
        finally:
            setattr(module, name, prev)

    @contextlib.contextmanager
    def _fused_tail(self):
        """The text encoder's precise-product hooks (fused kernels, the lo limb's fidelity, the batched products'
        program configs) for the vision tower and the LM."""
        hooks = {
            "GUARD_TAIL": self.fused_tail,
            "LANE_SPLIT": self.lane_split,
            "LO_FIDELITY": self.lo_fidelity,
            "NN_FIDELITY": self.lo_fidelity,
            "SUM_PARTS": self.sum_parts,
            "SPLIT_LIMBS": self.split_limbs,
            "BMM_CONFIG": self.bmm_config,
        }
        with contextlib.ExitStack() as stack:
            for name, fn in hooks.items():
                stack.enter_context(self._hook(_te_attention, name, fn))
            yield

    def vision_encode(self, up):
        with self._fused_tail():
            return self.text_encoder.encode_vision(up.te)

    def text_encode(self, up, image_embeds):
        with self._fused_tail():
            return self.text_encoder.encode_text(up.te, image_embeds)

    def vae_encode(self, up):
        with self._hook(_vae_resident, "VOTE_TAIL", self.vote_tail), self._hook(
            _vae_resident, "ROUND_BF16", self.round_bf16
        ):
            return self.vae.encode(up.vae_image)

    def denoise_setup_state(self, up, prompt_embeds, image_latents):
        """Persistent buffers: latents (updated in place), image latents, prompt embeddings, RoPE tables,
        and the schedule table (rolled on device by one row per step)."""
        d = self.device
        st = DenoiseState()
        st.B = up.B
        st.lat = ttnn.clone(up.latents)
        st.image_latents = image_latents
        st.prompt = prompt_embeds
        st.sched = _replicated(d, up.sched)
        st.n = up.sched.shape[0]
        st.rotary = self.transformer.rotary(up.img_shapes, prompt_embeds.shape[1])
        return st

    def denoise_step(self, st):
        """One denoising step on the persistent buffers: transformer forward + Euler update."""
        B, S, C = st.lat.shape[0], st.lat.shape[1], st.lat.shape[2]
        t = ttnn.repeat(ttnn.slice(st.sched, [0, 0], [1, 1]), (B, 1))  # [B, 1] = timestep / 1000
        dt = ttnn.reshape(ttnn.slice(st.sched, [0, 1], [1, 2]), (1, 1, 1))
        x_in = ttnn.concat([st.lat, st.image_latents], dim=1)
        v = self.transformer(x_in, t, st.prompt, st.rotary)
        v = ttnn.slice(v, [0, 0, 0], [B, S, C])  # noise_pred[:, : latents.size(1)]
        ttnn.copy(ttnn.add(st.lat, ttnn.multiply(v, dt)), st.lat)
        n, w = st.n, _SCHED_COLS
        ttnn.copy(
            ttnn.concat([ttnn.slice(st.sched, [1, 0], [n, w]), ttnn.slice(st.sched, [0, 0], [1, w])], dim=0), st.sched
        )
        return st.lat

    def vae_decode(self, up, latents):
        h, w = up.latent_hw
        return self.vae.decode(latents, h, w)

    # ---- the chained forward -------------------------------------------------------------------------
    def _sync_print(self, msg, t0=None):
        ttnn.synchronize_device(self.device)
        dt = "" if t0 is None else f" ({time.time() - t0:.1f} s)"
        print(f"[qwen_image_edit] {msg}{dt}", flush=True)

    def denoise(self, st, timesteps, use_trace=False, on_step=None):
        """The scheduler's loop: one step per timestep, all B samples in one program per step. Each step
        is synchronised and reported as it completes (a silent run looks hung)."""
        tid = None
        n = len(timesteps)
        t0 = time.time()
        try:
            for i, t in enumerate(timesteps):
                if tid is None:
                    self.denoise_step(st)
                    if use_trace and i + 1 < n:
                        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
                        self.denoise_step(st)
                        ttnn.end_trace_capture(self.device, tid, cq_id=0)
                        # the capture recorded the step without running it
                else:
                    ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
                self._sync_print(f"denoise step {i + 1}/{n} t={float(t):.2f}", t0)
                if on_step is not None:
                    on_step(i, st)
        finally:
            if tid is not None:
                ttnn.release_trace(self.device, tid)
        return st.lat

    def forward(self, up, use_trace=False, on_step=None):
        """encoded inputs (on device) -> dict(image [B, 3, H, W] in [0, 1], latents, prompt_embeds, image_latents)."""
        t0 = time.time()
        img_emb = self.vision_encode(up)
        self._sync_print("vision_encode done", t0)
        prompt = self.text_encode(up, img_emb)
        self._sync_print("text_encode done", t0)
        image_latents = self.vae_encode(up)
        self.vae.release_buffers()
        self._sync_print("vae_encode done", t0)
        st = self.denoise_setup_state(up, prompt, image_latents)
        latents = self.denoise(st, up.timesteps, use_trace=use_trace, on_step=on_step)
        image = self.vae_decode(up, latents)
        self.vae.release_buffers()
        self._sync_print("vae_decode done", t0)
        return {"image": image, "latents": latents, "prompt_embeds": prompt, "image_latents": image_latents}

    # ---- trace contract (per stage) --------------------------------------------------------------------
    def _stage_inputs(self, B=None):
        """Host inputs of every stage for the published example at batch B, from the HF golden's recorded
        stage tensors (reference/golden.py) when they are cached for these seeds."""
        from models.demos.qwen_image_edit.reference import golden as G

        B = I.batch_size_from_env() if B is None else B
        enc = I.encode_inputs(B)
        try:
            g = G.load_golden(enc, build=False)
        except FileNotFoundError:
            g = None
        return enc, g

    def _stage_reference(self, name):
        enc, g = self._stage_inputs()
        if g is not None:
            return enc, g
        # no cached golden for these seeds: run the upstream TT stages once on the same encoded inputs
        up = self.upload_inputs(enc)
        img = self.vision_encode(up)
        prompt = self.text_encode(up, img)
        lat = self.vae_encode(up)
        self.vae.release_buffers()
        return enc, {
            "image_embeds": to_host(img, self.device),
            "prompt_embeds": to_host(prompt, self.device),
            "image_latents": to_host(lat, self.device),
            "latents_final": enc.latents,
        }

    def _stage_upload(self, enc):
        """ONE device upload of an encoded input, shared by every stage's setup. The perf harness runs
        every <stage>_trace_setup before capturing any stage; with one upload per setup the five copies
        (~130 MB per chip at B=32) left denoise and vae_decode short of DRAM (bank_manager TT_FATAL).
        Each stage gets a shallow copy, so the tensors a setup attaches stay that stage's own."""
        import copy

        cache = self.__dict__.setdefault("_uploads", {})
        k = enc.key()
        if k not in cache:
            cache[k] = self.upload_inputs(enc)
        return copy.copy(cache[k])

    def _enter_stage(self, name):
        """Free the previous stage's per-shape scratch (the VAE's halo buffers) when a DIFFERENT stage's
        step starts. The perf harness captures + replays the stages back to back with no hook between
        them, so the halo buffers vae_encode cached stayed resident and at B=32 denoise / vae_decode then
        failed allocating (bank_manager TT_FATAL). The previous stage's trace is already released by
        then; repeated calls of the SAME stage keep them (its trace replays against them)."""
        if self._active_stage != name:
            self.vae.release_buffers()
            self._active_stage = name

    # vision_encode
    def vision_encode_trace_inputs(self):
        return self._stage_inputs()[0]

    def vision_encode_trace_setup(self, inputs):
        self._trace["vision_encode"] = self._stage_upload(inputs)

    def vision_encode_trace_step(self):
        self._enter_stage("vision_encode")
        return self.vision_encode(self._trace["vision_encode"])

    def vision_encode_trace_items(self):
        enc = (
            self._trace["vision_encode"].enc
            if "vision_encode" in self._trace
            else I.encode_inputs(I.batch_size_from_env())
        )
        return int(enc.vl["pixel_values"].shape[0])  # patch tokens over all B images

    # text_encode
    def text_encode_trace_inputs(self):
        enc, g = self._stage_reference("text_encode")
        emb = g.get("image_embeds")
        return {"enc": enc, "image_embeds": emb}

    def text_encode_trace_setup(self, inputs):
        up = self._stage_upload(inputs["enc"])
        emb = inputs.get("image_embeds")
        up.image_embeds = self.vision_encode(up) if emb is None else _replicated(self.device, emb.to(torch.float32))
        self._trace["text_encode"] = up

    def text_encode_trace_step(self):
        self._enter_stage("text_encode")
        up = self._trace["text_encode"]
        return self.text_encode(up, up.image_embeds)

    def text_encode_trace_items(self):
        up = self._trace.get("text_encode")
        ids = up.enc.vl["input_ids"] if up is not None else I.encode_inputs(I.batch_size_from_env()).vl["input_ids"]
        return int(ids.numel())  # B x sequence tokens through the 28 layers

    # vae_encode
    def vae_encode_trace_inputs(self):
        return self._stage_inputs()[0]

    def vae_encode_trace_setup(self, inputs):
        self._trace["vae_encode"] = self._stage_upload(inputs)

    def vae_encode_trace_step(self):
        self._enter_stage("vae_encode")
        return self.vae_encode(self._trace["vae_encode"])

    def vae_encode_trace_items(self):
        up = self._trace.get("vae_encode")
        enc = up.enc if up is not None else I.encode_inputs(I.batch_size_from_env())
        return enc.batch * self._conv_items("encoder", (1, 3, 1, enc.height, enc.width))

    # denoise
    def denoise_trace_inputs(self):
        enc, g = self._stage_reference("denoise")
        return {"enc": enc, "prompt_embeds": g["prompt_embeds"], "image_latents": g["image_latents"]}

    def denoise_trace_setup(self, inputs):
        enc = inputs["enc"]
        up = self._stage_upload(enc)
        il = inputs["image_latents"].to(torch.float32)
        if il.dim() == 5:  # golden records the unpacked [B, C, 1, h, w] image latents
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline

            b, c, _, h, w = il.shape
            il = QwenImageEditPipeline._pack_latents(il, b, c, h, w)
        st = self.denoise_setup_state(
            up, _replicated(self.device, inputs["prompt_embeds"].to(torch.float32)), _replicated(self.device, il)
        )
        self._trace["denoise"] = (up, st)

    def denoise_trace_step(self):
        self._enter_stage("denoise")
        return self.denoise_step(self._trace["denoise"][1])

    def denoise_trace_items(self):
        if "denoise" in self._trace:
            st = self._trace["denoise"][1]
            return int(st.B * (st.lat.shape[1] + st.image_latents.shape[1] + st.prompt.shape[1]))
        enc = I.encode_inputs(I.batch_size_from_env())
        s_img = enc.latents.shape[1] * len(enc.img_shapes)
        return int(enc.batch * (s_img + enc.vl["input_ids"].shape[1] - I.PROMPT_TEMPLATE_DROP))

    # vae_decode
    def vae_decode_trace_inputs(self):
        enc, g = self._stage_reference("vae_decode")
        lat = g["step_latents"][-1] if "step_latents" in g else g["latents_final"]
        return {"enc": enc, "latents": lat}

    def vae_decode_trace_setup(self, inputs):
        up = self._stage_upload(inputs["enc"])
        up.final_latents = _replicated(self.device, inputs["latents"].to(torch.float32))
        self._trace["vae_decode"] = up

    def vae_decode_trace_step(self):
        self._enter_stage("vae_decode")
        up = self._trace["vae_decode"]
        return self.vae_decode(up, up.final_latents)

    def vae_decode_trace_items(self):
        up = self._trace.get("vae_decode")
        enc = up.enc if up is not None else I.encode_inputs(I.batch_size_from_env())
        h, w = enc.latent_hw
        return enc.batch * self._conv_items("decoder", (1, int(self.hf.vae.config.z_dim), 1, h, w))

    def _conv_items(self, part, shape):
        """Items one image retires in a conv stack, in the 2 x params x items sense: each conv applies its
        weights once per output voxel, so items = sum(weights x output voxels) / sum(weights). Read from
        the HF module's own conv output shapes (one hook pass on a zero input, cached)."""
        key = (part, tuple(shape))
        cache = self.__dict__.setdefault("_items_cache", {})
        if key not in cache:
            mod = getattr(self.hf.vae, part)
            convs = [m for m in mod.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))]
            work, hooks = [], []
            for m in convs:
                hooks.append(
                    m.register_forward_hook(lambda m, i, o: work.append((m.weight.numel(), o.numel() // o.shape[1])))
                )
            try:
                with torch.no_grad():
                    mod(torch.zeros(shape))
            finally:
                for h in hooks:
                    h.remove()
            params = sum(m.weight.numel() for m in convs)
            cache[key] = max(1, int(round(sum(p * v for p, v in work) / params)))
        return cache[key]


def build_pipeline(
    device,
    model=None,
    layers=None,
    vision_encode_layers=None,
    text_encode_layers=None,
    denoise_layers=None,
    tracker=None,
    precise=True,
    **_demo_kwargs,
):
    """Construct and RETURN the resident pipeline object (it is not run here).

    layers caps every repeated stack (None = full depth); <stage>_layers overrides it for that stage's
    stack (vision tower, LM, transformer). model: an HF QwenImageEditPipeline to build from (default:
    the checkpoint, float32). Demo kwargs (prompt, image, ...) are accepted and ignored."""
    return TtQwenImageEditPipeline(
        device,
        hf=model,
        layers=layers,
        vision_encode_layers=vision_encode_layers,
        text_encode_layers=text_encode_layers,
        denoise_layers=denoise_layers,
        tracker=tracker,
        precise=precise,
    )


def run_image_edit(pipe, enc: I.EncodedInputs, use_trace=False, on_step=None, observe_host_ops=False):
    """THE chained forward shared by demo and test: encoded inputs -> edited images (device tensor).

    observe_host_ops: run the forward (everything after the input upload) under the host-op observer and
    return its verdict as out["host_ops"]."""
    up = pipe.upload_inputs(enc)
    if not observe_host_ops:
        return pipe.forward(up, use_trace=use_trace, on_step=on_step)
    from scripts.tt_hw_planner import host_op_observer

    with host_op_observer.observe_host_ops() as ops:
        out = pipe.forward(up, use_trace=use_trace, on_step=on_step)
    out["host_ops"] = host_op_observer.verdict(list(ops))
    return out


# ---- self-tests (zero-arg entry points for the harness probes) ---------------------------------------
def _open_default_mesh():
    from models.demos.qwen_image_edit.demo.mesh import open_mesh

    return open_mesh()


def _close_default_mesh(device):
    from models.demos.qwen_image_edit.demo.mesh import close_mesh

    close_mesh(device)


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def trace_capture_selftest(device=None, layers=2, batch=2):
    """For EACH stage: set up, run it eagerly (compile + reference), capture ONE step, replay it, compare,
    release the trace before the next stage. True only if every stage captured and matched.

    batch: whether a stage captures does not depend on B, and at the e2e batch (32) the VAE stages alone
    run ~4 min per pass (x3 passes here). 2 keeps the VAE's 2-row batch split; the stage inputs are
    built at the batch the env names, so it is pinned for the duration of the selftest."""
    import os

    from models.experimental.perf_automation.agent.perf_adapter import BATCH_ENV

    own = device is None
    device = _open_default_mesh() if own else device
    ok = True
    prev_batch = os.environ.get(BATCH_ENV)
    os.environ[BATCH_ENV] = str(int(batch))
    try:
        pipe = build_pipeline(device, layers=layers)
        for stage in PIPELINE_STAGES:
            setup = getattr(pipe, f"{stage}_trace_setup")
            step = getattr(pipe, f"{stage}_trace_step")
            setup(getattr(pipe, f"{stage}_trace_inputs")())
            snap = None
            if stage == "denoise":
                st = pipe._trace["denoise"][1]
                snap = (ttnn.clone(st.lat), ttnn.clone(st.sched))
            ref = to_host(step(), device)
            if snap is not None:  # rewind the in-place state so the replay starts where the eager step did
                ttnn.copy(snap[0], st.lat)
                ttnn.copy(snap[1], st.sched)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = to_host(out, device)
            ttnn.release_trace(device, tid)
            pcc = _pcc(got, ref)
            print(f"[trace_selftest] {stage}: captured + replayed, trace-vs-eager pcc {pcc:.6f}", flush=True)
            ok = ok and pcc >= 0.9999
            pipe.vae.release_buffers()
            pipe._trace.pop(stage, None)
    finally:
        if prev_batch is None:
            os.environ.pop(BATCH_ENV, None)
        else:
            os.environ[BATCH_ENV] = prev_batch
        if own:
            _close_default_mesh(device)
    return bool(ok)


def host_op_selftest(device=None, layers=2, batch=2, on_step=None):
    """Run the whole forward (encoded inputs -> image) under the host-op observer. Input encoding,
    weight build and the input upload happen OUTSIDE the observed region; every stage runs inside it."""
    from scripts.tt_hw_planner import host_op_observer

    own = device is None
    device = _open_default_mesh() if own else device
    try:
        pipe = build_pipeline(device, layers=layers)
        enc = I.encode_inputs(batch)
        pipe.forward(pipe.upload_inputs(enc))  # warm-up: program compile + per-shape device buffers
        up = pipe.upload_inputs(enc)
        with host_op_observer.observe_host_ops() as ops:
            out = pipe.forward(up, on_step=on_step)
        ttnn.synchronize_device(device)
        v = host_op_observer.verdict(list(ops))
        v["output_shape"] = list(out["image"].shape)
        return v
    finally:
        if own:
            _close_default_mesh(device)
