# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit on TT: the ONE chained forward shared by the demo and the e2e test.

    image, instruction --(HF processors, host)--> encoded inputs --(prepare, upload once)--> device inputs
    run_image_edit(device inputs):
        img_emb        = vision tower(pixels)                                  [vision_encode]
        pe, neg_pe     = text model(ids with img_emb spliced)                  [text_encode]  (cond | uncond rows)
        image_latents  = pack(normalize(quant_conv(vae encoder(image))))       [vae_encode]
        rope_c, rope_u = qwen_embed_rope(img_shapes, len(pe)), (.., len(neg_pe))
        for t in schedule:                                                     [denoise] x num_inference_steps
            x      = cat[latents, image_latents]
            eps_c  = transformer(x, pe, t), eps_u = transformer(x, neg_pe, t)  (the latent part)
            eps    = true-CFG(eps_c, eps_u) with norm rescale
            latents += (sigma_next - sigma) * eps                              (FlowMatch Euler)
        image          = vae decoder(post_quant_conv(denormalize(unpack(latents))))  [vae_decode]

Every stage consumes the previous stage's device tensor; nothing is read back to the host until the
final image.

Mesh: Galaxy, 32 Wormhole chips as an 8x4 mesh (build_pipeline reshapes a 1x32 / 4x8 handle in place).
  vision_encode  replicated over the 8 rows, TP=4 over the 4 columns (graduated)
  text_encode    TP=4 over the columns; the 28 LM layers are stage-split over the rows
                 (rows 0-3: layers 0-13, rows 4-7: layers 14-27), exact all_gather hand-off on axis 0
  vae_encode     batch over the 8-axis, W over the 4-axis (the re-graduated VAE ports)
  denoise        transformer TP=8 over mesh axis 0; the batch is split over mesh axis 1 (DP=4, B/4
                 samples per column: mesh_partition of the replicated text / VAE outputs)
  vae_decode     latents all_gathered over axis 1 -> batch over the 8-axis, W over the 4-axis
"""
from __future__ import annotations

import gc
import os
import time

import torch

import ttnn
from models.demos.qwen_image_edit.mesh import DEVICE_PARAMS, MESH_SHAPE  # noqa: F401 (re-exported)
from models.demos.qwen_image_edit.tt.inputs import EditConfig, EncodedInputs, encode_inputs
from models.demos.qwen_image_edit.tt.text_encoder import TtQwenTextEncoder
from models.demos.qwen_image_edit.tt.tracker import InvocationTracker
from models.demos.qwen_image_edit.tt.transformer import TtQwenImageTransformer
from models.demos.qwen_image_edit.tt.vae import TtQwenVAE
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl as _tr_ccl

PIPELINE_STAGES = ["vision_encode", "text_encode", "vae_encode", "denoise", "vae_decode"]

# The 25 graduated modules (status NEW + last_good snapshot) across the three bring-ups.
GRADUATED = {
    "text_encoder": [
        "vision_patch_embed",
        "v_l_vision_block",
        "v_l_patch_merger",
        "vision_transformer_pretrained_model",
        "v_l_decoder_layer",
        "language_model_layers_0_mlp",
        "v_l_text_model",
    ],
    "vae": [
        "qwen_image_encoder3d",
        "qwen_image_decoder3d",
        "qwen_image_causal_conv3d",
        "qwen_image_residual_block",
        "qwen_image_resample",
        "zero_pad2d",
        "qwen_image_mid_block",
        "qwen_image_attention_block",
        "qwen_image_r_m_s",
        "qwen_image_up_block",
        "qwen_image_upsample",
    ],
    "transformer": [
        "timesteps",
        "timestep_embedding",
        "qwen_timestep_proj_embeddings",
        "qwen_embed_rope",
        "qwen_image_transformer_block",
        "feed_forward",
        "ada_layer_norm_continuous",
    ],
}
GRADUATED_ALL = [n for v in GRADUATED.values() for n in v]

# Galaxy layout (e2e_plan.json "decisions"): the mesh is 8x4; the transformer's TP axis is mesh axis 0
# (TP=8, 24 heads / 8 = 3 per chip) and the denoise batch is split over mesh axis 1 (DP=4).
GALAXY_SHAPE = (8, 4)
TP_AXIS = 0
DP_AXIS = 1


def normalize_mesh(device):
    """Reshape a 32-chip mesh handed as 1x32 or 4x8 to the 8x4 grid, in place, before anything is
    allocated on it (the same device.reshape the VAE ports' physical_grid uses). Returns the device."""
    shape = tuple(device.shape)
    if device.get_num_devices() == GALAXY_SHAPE[0] * GALAXY_SHAPE[1] and shape != GALAXY_SHAPE:
        device.reshape(ttnn.MeshShape(*GALAXY_SHAPE))
    return device


# Per-stage batch ceilings (images per program). None -> the whole batch in one program.
# Empty on the 8x4 Galaxy: every stage ran all 32 samples in one program (2026-09-25, B=32 demo +
# full-depth trace test); the T3K ceilings (vae_decode 16, text 2 x 32) did not reproduce here.
STAGE_MAX_BATCH = {}


def _replicated(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )


def _dp_sharded(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    """Host input -> device, dim 0 (the batch) split over the DP mesh axis, replicated over the TP axis."""
    dims = [None, None]
    dims[DP_AXIS] = 0
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=tuple(dims))
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)


def to_host(t):
    """Replicated device tensor -> torch (device 0's copy)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


class PreparedInputs:
    """Device-resident encoded inputs for one batched image_edit call (uploaded once, outside the forward)."""


class QwenImageEditTT:
    """Resident TT pipeline. Build with build_pipeline()."""

    def __init__(
        self,
        device,
        hf_pipe,
        layers=None,
        vision_encode_layers=None,
        text_encode_layers=None,
        denoise_layers=None,
        cfg: EditConfig | None = None,
        precise_transformer: bool = True,
    ):
        self.device = normalize_mesh(device)
        self.hf = hf_pipe  # HF reference kept reachable: ground truth for section structure / depth
        self.cfg = cfg or EditConfig()
        self.batch_size = int(self.cfg.batch)  # images per image_edit call (what the trace hooks run)
        self.dp = tuple(device.shape)[DP_AXIS]  # denoise batch split (samples per column = B / dp)
        self.tp = tuple(device.shape)[TP_AXIS]  # transformer tensor parallelism
        self.tracker = InvocationTracker()
        pick = lambda v: layers if v is None else v  # noqa: E731
        t0 = time.time()
        self.text_encoder = TtQwenTextEncoder(
            device,
            hf_pipe.text_encoder,
            text_layers=pick(text_encode_layers),
            vision_layers=pick(vision_encode_layers),
            tracker=self.tracker,
        )
        self.vae = TtQwenVAE(device, hf_pipe.vae, tracker=self.tracker)
        with _tr_ccl.tp_axis(TP_AXIS):  # TP=8 over mesh axis 0, replicated over axis 1 (see _ccl)
            self.transformer = TtQwenImageTransformer(
                device, hf_pipe.transformer, layers=pick(denoise_layers), tracker=self.tracker
            )
        # Precise transformer (see _stubs/_precise.py): 2-limb activations + exact-lane QK^T. Measured at
        # B=2, full depth: CFG noise PCC 0.99993 -> 0.9999956 per forward. Over the 50-step trajectory
        # that moved the worst samples' image PCC from 0.52-0.64 to 0.76 (sample 30) and from ~0.85 to
        # 0.988 (sample 25).
        from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise as _tr_precise

        _tr_precise.ENABLED = bool(precise_transformer)
        self.build_seconds = time.time() - t0
        # repeated stacks (plain lists of same-typed elements), one per stage that owns one
        self.stacks = {
            "vision_encode": self.text_encoder.visual.blocks,
            "text_encode": self.text_encoder.text_model.layers,
            "denoise": self.transformer.transformer_blocks,
            "vae_encode": self.vae.encoder.down_blocks,
            "vae_decode": self.vae.decoder.up_blocks,
        }
        self._ts = {}  # per-stage resident trace inputs, keyed by stage name
        self.trace_denoise = True  # replay one captured scheduler step for steps 1..N-1 (see _denoise_loop)

    # ------------------------------------------------------------------ input encoding / upload
    def encode(self, cfg: EditConfig | None = None, images=None, prompts=None, seeds=None) -> EncodedInputs:
        return encode_inputs(cfg or self.cfg, images=images, prompts=prompts, seeds=seeds)

    def prepare(self, enc: EncodedInputs) -> PreparedInputs:
        d = self.device
        p = PreparedInputs()
        p.enc = enc
        p.B = enc.latents.shape[0]
        assert p.B % self.dp == 0, f"batch {p.B} must split evenly over the {self.dp} DP columns"
        p.B_local = p.B // self.dp
        p.te = self.text_encoder.prepare(enc)
        p.vae_image = _replicated(d, enc.vae_image, ttnn.float32)
        p.latents = _replicated(d, enc.latents.to(torch.float32), ttnn.float32)  # (vae_decode trace input)
        p.latents_dp = _dp_sharded(d, enc.latents.to(torch.float32))  # the denoise loop's initial noise
        p.S_lat = enc.latents.shape[1]
        p.lat_h = 2 * (enc.height // 16)
        p.lat_w = 2 * (enc.width // 16)
        p.img_shapes = list(enc.img_shapes)
        p.cfg_scale = float(enc.cfg.true_cfg_scale)
        p.do_cfg = p.cfg_scale > 1.0 and enc.cfg.negative_prompt is not None
        ts = enc.timesteps.to(torch.float32)
        sig = enc.sigmas.to(torch.float32)
        p.num_steps = int(ts.numel())
        # per step: t / 1000 for every sample (DP-split like the latents), and the Euler step size
        p.timesteps = [_dp_sharded(d, torch.full((p.B,), float(t) / 1000.0)) for t in ts.tolist()]
        p.dts = [_replicated(d, torch.full((1, 1, 1), float(sig[i + 1] - sig[i]))) for i in range(p.num_steps)]

        def joint_mask(m):
            if m is None:
                return None
            full = torch.cat([m.to(torch.float32), torch.ones(p.B, 2 * p.S_lat)], dim=1)
            return _dp_sharded(d, full.reshape(p.B, 1, 1, -1))

        p.joint_mask_cond = joint_mask(p.te.mask_cond)
        p.joint_mask_uncond = joint_mask(p.te.mask_uncond)
        return p

    # ------------------------------------------------------------------ stages (device only)
    def vision_encode(self, p):
        return self.text_encoder.encode_vision(p.te)

    def text_encode(self, p, img_emb):
        return self.text_encoder.encode_text(p.te, img_emb)

    def vae_encode(self, p):
        return self.vae.encode(p.vae_image)

    def rope(self, p, txt_len):
        return self.transformer.rope(p.img_shapes, txt_len)

    def dp_split(self, x):
        """Replicated device tensor -> this column's B/dp samples (dim 0 split over the DP axis)."""
        return ttnn.mesh_partition(x, dim=0, cluster_axis=DP_AXIS) if self.dp > 1 else x

    def dp_gather(self, x):
        """DP-split device tensor -> the whole batch on every chip (exact all_gather over the DP axis)."""
        if self.dp == 1:
            return x
        return ttnn.all_gather(x, dim=0, cluster_axis=DP_AXIS, num_links=1, topology=ttnn.Topology.Linear)

    def denoise_step(self, p, i, latents, image_latents, pe, neg_pe, rope_c, rope_u, t=None, dt=None):
        """One FlowMatch-Euler step with true CFG on this column's B/dp samples (every tensor here is
        DP-split over mesh axis 1). t / dt default to step i's persistent buffers."""
        t = p.timesteps[i] if t is None else t
        dt = p.dts[i] if dt is None else dt
        x = ttnn.concat([latents, image_latents], dim=1)
        B, S = latents.shape[0], p.S_lat
        eps = self.transformer(x, pe, t, rope_c, p.joint_mask_cond)
        eps = ttnn.slice(eps, [0, 0, 0], [B, S, eps.shape[-1]])
        if p.do_cfg:
            neg = self.transformer(x, neg_pe, t, rope_u, p.joint_mask_uncond)
            neg = ttnn.slice(neg, [0, 0, 0], [B, S, neg.shape[-1]])
            comb = ttnn.add(neg, ttnn.multiply(ttnn.subtract(eps, neg), p.cfg_scale))
            cond_norm = ttnn.sqrt(ttnn.sum(ttnn.multiply(eps, eps), dim=-1, keepdim=True))
            comb_norm = ttnn.sqrt(ttnn.sum(ttnn.multiply(comb, comb), dim=-1, keepdim=True))
            eps = ttnn.multiply(comb, ttnn.divide(cond_norm, comb_norm))
        return ttnn.add(latents, ttnn.multiply(eps, dt))

    def _denoise_loop(self, p, n, latents, image_latents, pe, neg_pe, rope_c, rope_u, on_step=None):
        """The scheduler loop. Step 0 runs eagerly (it compiles every program the step uses); the step is
        then captured ONCE as a device trace over persistent (latents, t, dt) buffers and replayed for
        steps 1..n-1. Each replay is fed by device-to-device copies of that step's pre-uploaded t / dt
        and the previous step's output, so the loop never touches the host. self.trace_denoise=False
        runs every step eagerly (same ops, same order: the replay matches eager to PCC 1.0)."""
        if n <= 0:
            return latents
        latents = self.denoise_step(p, 0, latents, image_latents, pe, neg_pe, rope_c, rope_u)
        if on_step is not None:
            on_step(0, latents)
        if n == 1 or not self.trace_denoise:
            for i in range(1, n):
                latents = self.denoise_step(p, i, latents, image_latents, pe, neg_pe, rope_c, rope_u)
                if on_step is not None:
                    on_step(i, latents)
            return latents
        d = self.device
        lat_buf = latents
        t_buf = ttnn.clone(p.timesteps[1])
        dt_buf = ttnn.clone(p.dts[1])
        tid = ttnn.begin_trace_capture(d, cq_id=0)
        out = self.denoise_step(p, 1, lat_buf, image_latents, pe, neg_pe, rope_c, rope_u, t=t_buf, dt=dt_buf)
        ttnn.end_trace_capture(d, tid, cq_id=0)
        try:
            for i in range(1, n):
                if i > 1:
                    ttnn.copy(out, lat_buf)
                    ttnn.copy(p.timesteps[i], t_buf)
                    ttnn.copy(p.dts[i], dt_buf)
                ttnn.execute_trace(d, tid, cq_id=0, blocking=False)
                if on_step is not None:
                    on_step(i, out)
            result = ttnn.clone(out)
        finally:
            ttnn.release_trace(d, tid)
        for t in (t_buf, dt_buf, lat_buf, out):
            ttnn.deallocate(t)
        return result

    def vae_decode(self, p, latents):
        return self.vae.decode(latents, p.lat_h, p.lat_w, max_batch=STAGE_MAX_BATCH.get("vae_decode"))

    # ------------------------------------------------------------------ the chain
    def run_image_edit(self, p, num_steps=None, on_step=None):
        """The full image_edit forward on device -> image [B, 3, H, W] in [0, 1] (device tensor)."""
        img_emb = self.vision_encode(p)
        pe, neg_pe = self.text_encode(p, img_emb)
        image_latents = self.vae_encode(p)
        rope_c = self.rope(p, pe.shape[1])
        rope_u = self.rope(p, neg_pe.shape[1])
        n = p.num_steps if num_steps is None else num_steps
        # denoise: the batch is split over mesh axis 1 (DP), the transformer is TP over axis 0
        pe_l, neg_l, img_lat_l = self.dp_split(pe), self.dp_split(neg_pe), self.dp_split(image_latents)
        latents = self._denoise_loop(p, n, p.latents_dp, img_lat_l, pe_l, neg_l, rope_c, rope_u, on_step)
        latents = self.dp_gather(latents)  # the whole batch on every chip for the decode
        self.last_latents = latents
        self.steps_run = n
        return self.vae_decode(p, latents)

    def image_edit(self, enc: EncodedInputs):
        """Host convenience: upload, run the chain, read the image back as torch [B, 3, H, W]."""
        p = self.prepare(enc)
        out = self.run_image_edit(p)
        return to_host(out).to(torch.float32)

    # ================================================================== trace contract (per stage)
    # Stages come from the HF config: the text encoder (Qwen2_5_VLForConditionalGeneration) has a
    # vision_config and a text_config -> [vision_encode, text_encode]; the VAE encodes the condition
    # image -> [vae_encode]; the transformer runs once per scheduler step -> [denoise]; the VAE
    # decodes the final latent -> [vae_decode]. Every variable dim is pinned by the fixed image area
    # (vision patches, latent tokens) and by the padded text capacity C (tile multiple of the longest
    # prompt), so each stage replays at one shape.

    def _trace_inputs_common(self):
        if getattr(self, "_trace_enc", None) is None:
            self._trace_enc = encode_inputs(self.cfg)
        return self._trace_enc

    def _trace_prepared(self, inputs):
        key = id(inputs)
        if getattr(self, "_trace_p_key", None) != key:
            self._trace_p = self.prepare(inputs)
            self._trace_p_key = key
        return self._trace_p

    # ---- vision_encode
    def vision_encode_trace_inputs(self):
        return self._trace_inputs_common()

    def vision_encode_trace_setup(self, inputs):
        self._ts["vision_encode"] = {"p": self._trace_prepared(inputs)}

    def vision_encode_trace_step(self):
        return self.vision_encode(self._ts["vision_encode"]["p"])

    def vision_encode_trace_items(self):
        p = self._trace_prepared(self._trace_inputs_common())
        return int(p.B * p.te.vision_consts.s)  # patches through the 32 vision blocks

    # ---- text_encode
    def text_encode_trace_inputs(self):
        return self._trace_inputs_common()

    def text_encode_trace_setup(self, inputs):
        p = self._trace_prepared(inputs)
        img = self.vision_encode(p)  # the stage's image-embedding input, resident
        self._ts["text_encode"] = {"p": p, "img": img}

    def text_encode_trace_step(self):
        s = self._ts["text_encode"]
        pe, neg = self.text_encode(s["p"], s["img"])
        return pe

    def text_encode_trace_items(self):
        p = self._trace_prepared(self._trace_inputs_common())
        return int(2 * p.B * p.te.s_pad)  # prompts + negatives, padded capacity, through 28 layers

    # ---- vae_encode
    def vae_encode_trace_inputs(self):
        return self._trace_inputs_common()

    def vae_encode_trace_setup(self, inputs):
        self.vae.release_after_stage = False  # halo buffers must stay put while a trace references them
        self._ts["vae_encode"] = {"p": self._trace_prepared(inputs)}

    def vae_encode_trace_step(self):
        return self.vae_encode(self._ts["vae_encode"]["p"])

    def vae_encode_trace_items(self):
        return int(self._trace_prepared(self._trace_inputs_common()).B * self._vae_items("encode"))

    # ---- denoise (one scheduler step: cond + uncond transformer, true CFG, Euler update)
    def denoise_trace_inputs(self):
        return self._trace_inputs_common()

    def denoise_trace_setup(self, inputs):
        p = self._trace_prepared(inputs)
        img = self.vision_encode(p)
        pe, neg = self.text_encode(p, img)
        lat_img = self.vae_encode(p)
        self._ts["denoise"] = {
            "p": p,
            "pe": self.dp_split(pe),  # the step runs on DP-split inputs, as in run_image_edit
            "neg": self.dp_split(neg),
            "img_lat": self.dp_split(lat_img),
            "rope_c": self.rope(p, pe.shape[1]),  # RoPE tables: shape-dependent constants, pre-uploaded
            "rope_u": self.rope(p, neg.shape[1]),
            "latents": p.latents_dp,
            "t": ttnn.clone(p.timesteps[0]),  # persistent step inputs
            "dt": ttnn.clone(p.dts[0]),
        }

    def denoise_trace_step(self):
        s = self._ts["denoise"]
        return self.denoise_step(
            s["p"], 0, s["latents"], s["img_lat"], s["pe"], s["neg"], s["rope_c"], s["rope_u"], t=s["t"], dt=s["dt"]
        )

    def denoise_trace_items(self):
        p = self._trace_prepared(self._trace_inputs_common())
        lc, lu = p.te.len_cond, p.te.len_uncond
        n_img = 2 * p.S_lat  # noise latent + condition latent tokens
        return int(p.B * (n_img + lc) + (p.B * (n_img + lu) if p.do_cfg else 0))  # tokens through the 60 blocks

    # ---- vae_decode
    def vae_decode_trace_inputs(self):
        return self._trace_inputs_common()

    def vae_decode_trace_setup(self, inputs):
        self.vae.release_after_stage = False
        self._ts["vae_decode"] = {"p": self._trace_prepared(inputs)}

    def vae_decode_trace_step(self):
        p = self._ts["vae_decode"]["p"]
        return self.vae_decode(p, p.latents)

    def vae_decode_trace_items(self):
        return int(self._trace_prepared(self._trace_inputs_common()).B * self._vae_items("decode"))

    def _vae_items(self, which):
        """Per-image items for the conv VAE: FLOPs / (2 x params), i.e. the parameter-weighted mean
        number of output positions each weight is applied at (from the HF module at the pinned size)."""
        cache = getattr(self, "_vae_items_cache", {})
        if which in cache:
            return cache[which]
        vae = self.hf.vae
        h = w = None
        enc = self._trace_inputs_common()
        h, w = enc.height, enc.width
        macs = [0]
        params = sum(p.numel() for p in (vae.encoder if which == "encode" else vae.decoder).parameters())

        def hook(m, a, o):
            if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d)):
                macs[0] += m.weight.numel() * (o.numel() // o.shape[1] // o.shape[0])

        mod = vae.encoder if which == "encode" else vae.decoder
        hs = [m.register_forward_hook(hook) for m in mod.modules() if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d))]
        try:
            with torch.no_grad():
                if which == "encode":
                    vae.encode(torch.zeros(1, 3, 1, h, w))
                else:
                    vae.decode(torch.zeros(1, vae.config.z_dim, 1, h // 8, w // 8))
        finally:
            for hh in hs:
                hh.remove()
        cache[which] = max(1, macs[0] // max(1, params))
        self._vae_items_cache = cache
        return cache[which]

    def _trace_readback(self, t):
        """Whole stage output on host: device 0's copy, plus the other DP columns for a DP-split tensor
        (the denoise step's output), concatenated along the batch."""
        t = t[0] if isinstance(t, tuple) else t
        shape = tuple(self.device.shape)
        devs = ttnn.get_device_tensors(t)
        cols = [ttnn.to_torch(devs[c]).to(torch.float32) for c in range(shape[1])]
        if all(torch.equal(cols[0], c) for c in cols[1:]):
            return cols[0]
        return torch.cat(cols, dim=0)

    def trace_capture_selftest(self, device=None, pcc_threshold=0.999):
        """Per stage: eager reference, then capture ONE step, execute it, compare, release the trace
        before the next stage (stage traces never co-reside). True only if every stage captured and
        matched. trace_region_size comes from DEVICE_PARAMS (sized for the largest stage, denoise)."""
        from models.common.utility_functions import comp_pcc

        d = device or self.device
        ok_all = True
        self.trace_report = {}
        for stage in PIPELINE_STAGES:
            inputs = getattr(self, f"{stage}_trace_inputs")()
            getattr(self, f"{stage}_trace_setup")(inputs)
            step = getattr(self, f"{stage}_trace_step")
            ref = self._trace_readback(step())  # eager warm-up: compiles, allocates halo buffers
            tid = None
            try:
                tid = ttnn.begin_trace_capture(d, cq_id=0)
                out = step()
                ttnn.end_trace_capture(d, tid, cq_id=0)
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[0][:200]
                print(
                    f"[trace] {stage}: capture FAILED ({msg}); this stage's capacity is pinned by the config "
                    f"(image area / prompt length), so there is no smaller C to fall back to",
                    flush=True,
                )
                self.trace_report[stage] = {"captured": False, "error": msg}
                ok_all = False
                if tid is not None:
                    try:
                        ttnn.release_trace(d, tid)
                    except Exception:  # noqa: BLE001
                        pass
                self._ts.pop(stage, None)
                self.vae.release_after_stage = True
                self.vae.release_buffers()
                gc.collect()
                continue
            ttnn.execute_trace(d, tid, cq_id=0, blocking=True)
            got = self._trace_readback(out)
            _, pcc = comp_pcc(ref, got, pcc_threshold)
            ttnn.release_trace(d, tid)
            passed = float(pcc) >= pcc_threshold
            self.trace_report[stage] = {"captured": True, "pcc_vs_eager": float(pcc)}
            print(f"[trace] {stage}: captured, replay PCC vs eager {float(pcc):.6f}", flush=True)
            ok_all = ok_all and passed
            # the trace is gone: drop this stage's resident state before the next stage sets up
            del out
            self._ts.pop(stage, None)
            self.vae.release_after_stage = True
            self.vae.release_buffers()
            gc.collect()
        self.vae.release_after_stage = True
        self.vae.release_buffers()
        self._ts = {}
        return ok_all

    def host_op_selftest(self, p=None, num_steps=None, on_step=None):
        """Run the image_edit forward (encoded + uploaded inputs -> image, every stage) under
        host_op_observer. Input encoding and the weight build happen outside the observed region.
        on_step is run_image_edit's per-scheduler-step callback, passed through unchanged.
        Returns (verdict, output device tensor)."""
        from scripts.tt_hw_planner import host_op_observer

        p = p if p is not None else self.prepare(self._trace_inputs_common())
        with host_op_observer.observe_host_ops() as ops:
            out = self.run_image_edit(p, num_steps=num_steps, on_step=on_step)
            ttnn.synchronize_device(self.device)
        return host_op_observer.verdict(list(ops)), out


def load_hf_reference(dtype=torch.float32):
    from diffusers import QwenImageEditPipeline

    from models.demos.qwen_image_edit.tt.inputs import MODEL_ID

    return QwenImageEditPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)


def build_pipeline(
    device, model=None, layers=None, vision_encode_layers=None, text_encode_layers=None, denoise_layers=None, **kwargs
):
    """Construct and RETURN the resident TT pipeline object (weights on device; nothing is run).

    model: an HF QwenImageEditPipeline (float32); loaded from the hub when None.
    layers: default depth cap for EVERY repeated stack (None = all layers); per-stack overrides
        vision_encode_layers / text_encode_layers / denoise_layers fall back to `layers`.
    Other kwargs (prompt, image, ...) are accepted and ignored: shapes come from the config.
    A 32-chip mesh handed as 1x32 or 4x8 is reshaped to 8x4 in place before anything is allocated.
    """
    normalize_mesh(device)
    if layers is None and os.environ.get("TT_PERF_LAYERS"):
        layers = int(os.environ["TT_PERF_LAYERS"])
    hf = model if model is not None else load_hf_reference()
    cfg = kwargs.get("cfg")
    return QwenImageEditTT(
        device,
        hf,
        layers=layers,
        vision_encode_layers=vision_encode_layers,
        text_encode_layers=text_encode_layers,
        denoise_layers=denoise_layers,
        cfg=cfg if isinstance(cfg, EditConfig) else None,
        precise_transformer=kwargs.get("precise_transformer", True),
    )


# ====================================================================== harness entry points (zero-arg)
# The emit-e2e probes import this module and call these with no arguments, in a fresh process. They are
# standalone self-tests, so they open (and close) the mesh themselves via models/demos/qwen_image_edit/mesh.py;
# the pipeline proper only ever runs on the device handed to build_pipeline. They check properties of
# the CODE PATH (no host aten op in the forward; every stage captures and replays), which do not depend
# on depth or batch, so they build shallow (2 of each repeated stack) at B=8 -- the smallest batch the
# 8x4 layout takes: one image per mesh row in the batch-parallel VAE, 2 samples per DP column in the
# denoise -- for 2 scheduler steps (step 0 eager + one traced replay) to stay inside the probes' budget.
SELFTEST_LAYERS = 2
SELFTEST_CFG = dict(batch=8, num_inference_steps=2)


def _selftest_pipeline(mesh):
    cfg = EditConfig(**SELFTEST_CFG)
    return build_pipeline(mesh, model=load_hf_reference(), layers=SELFTEST_LAYERS, cfg=cfg), cfg


def host_op_selftest():
    """The full image_edit forward (every stage, traced denoise loop) under host_op_observer -> verdict."""
    from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh

    mesh = open_mesh()
    try:
        pipe, cfg = _selftest_pipeline(mesh)
        verdict, _ = pipe.host_op_selftest(pipe.prepare(pipe.encode(cfg)))
        return verdict
    finally:
        close_mesh(mesh)


def trace_capture_selftest():
    """Every PIPELINE_STAGES step captured, replayed and compared against eager -> True iff all match."""
    from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh

    mesh = open_mesh()
    try:
        pipe, _ = _selftest_pipeline(mesh)
        ok = pipe.trace_capture_selftest()
        print(f"[trace] report {pipe.trace_report}", flush=True)
        return ok
    finally:
        close_mesh(mesh)
