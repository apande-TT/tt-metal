# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stage test for the FLUX.2-klein-9B VAE: all seven routes of ``tt/vae.py`` against
the real ``AutoencoderKLFlux2`` golden, on a 1x8 mesh at TP=8.

The input is a real image (a deterministic gradient-and-shapes PIL image) pushed
through the pipeline's own ``Flux2ImageProcessor``, not a random tensor.

Resolution is 256x256, not the captured 224x224.  ``ttnn.group_norm``'s DRAM grid
check (``groupnorm.cpp::validate_dram_grid``) requires ``Ht = ceil(N*H*W/32)`` to
be a nonzero multiple of ``num_virtual_rows``, and tt_dit's ``GroupNorm`` pins
``CoreGrid(8, 8)``.  A 224x224 image has a 28x28 latent, ``Ht = 25``, which no
core grid divides -- the 512-channel mid block fails before any arithmetic.  The
graduated stubs were PCC'd at 256x256 / 32x32 for exactly this reason (see
``models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder.py``'s
``_NCHW_INPUT`` comment).  ``tt/vae.py::check_resolution`` states the rule.

The mesh device and the built stage are MODULE-scoped: the seven routes together
build ~20 graduated ports, and reopening the device (and reloading every port's
weights) per test function would multiply that by eight.

BATCH
-----
The second half of this file drives the same routes at a leading batch of 32
INDEPENDENT samples and compares EACH sample to its own golden.  Those tests
currently FAIL, and they are left failing on purpose: everything in ``tt/vae.py``
carries the batch, but ``models/tt_dit/layers/normalization.py::GroupNorm.forward``
asks ``ttnn.group_norm`` for ``num_out_blocks=-1``, and the op's auto-heuristic
sizes the per-core chunk from ``H*W*C`` with the batch nowhere in it -- so at any
``B > 1`` the 256x256 norms overflow L1 with

    Statically allocated circular buffers ... grow to 1967296 B which is beyond
    max L1 size of 1499136 B          (128 channels, 256x256, replicated, B=2)

The stage can pass a chunk count to the two norms it owns, but not to the ones a
graduated stub owns (tt_dit calls those as ``self.norm1.forward(h)``), and
rebinding a ``.forward`` under ``tt/`` is itself a gate.

With that one line applied in-process --

    F2K_VAE_UPSTREAM_GROUPNORM_FIX=1 F2K_VAE_L1_SMALL=65536 pytest ... -k batch

-- ``encode``, ``encode_alias`` and ``encode_blockwise`` all pass at B=32 with a
worst per-sample PCC of 0.9989 and a worst cross-sample correlation of 0.91, i.e.
32 real independent samples.  ``encode_decomposed`` and the three decode routes
then stop on a SECOND and unrelated batch limit -- an L1-buffer / circular-buffer
clash at B=32 that this stage has no lever on (see ``mesh_device``).
"""

from __future__ import annotations

import gc
import importlib
import math
import os

import pytest
import torch

from models.demos.flux_2_klein_9b import reference

# Side-load the Flux2-capable diffusers (and its huggingface_hub) before anything
# else can bind the older one.
reference.ensure_flux_imports()

import ttnn  # noqa: E402
from models.demos.flux_2_klein_9b.tt.stubs import Ledger, graduated_components, load_stub_module  # noqa: E402
from models.demos.flux_2_klein_9b.tt.vae import (  # noqa: E402
    Flux2VaeStage,
    _group_norm_out_blocks,
    check_batch,
    check_resolution,
    legal_batch,
)

TP = 8
IMAGE_SIZE = 256
#: per-chip L1_SMALL reservation.  The bring-up default (24576) is what every B=1
#: test here is green at; the batched conv halo needs more (see `mesh_device`), so
#: it is overridable rather than raised -- raising it regresses B=1's neighbours.
L1_SMALL_SIZE = int(os.environ.get("F2K_VAE_L1_SMALL", "24576"))
LATENT_SIZE = IMAGE_SIZE // 8
PCC_TARGET = 0.98

# --------------------------------------------------------------------------- batch
#
# The stage's target is 32 INDEPENDENT samples per call -- 32 different images at
# `encode*`, 32 different latents at `decode*`, one program, no python loop over
# samples.  A hardcoded leading 1 does not raise at B=32, it keeps sample 0 and
# drops the rest, so every batch test below compares EACH sample to ITS OWN
# golden and additionally requires each TT sample's best-matching golden to be
# its own.
BATCH_TARGET = 32
BATCH_SIZES = tuple(int(b) for b in os.environ.get("F2K_VAE_BATCH", str(BATCH_TARGET)).split(","))
BATCH_PCC_TARGET = 0.99

#: Opt-in, in-process shim for the ONE upstream line that stands between this
#: stage and B=32.  It is NOT part of the stage and is off by default; it exists
#: so the batch results below can be reproduced (and so the batch gates go green
#: by themselves) the moment tt_dit lands the same change.
#:
#: `models/tt_dit/layers/normalization.py::GroupNorm.forward` reshapes
#: `[B, H, W, C]` to `[B, 1, H*W, C]` and asks `ttnn.group_norm` for
#: `num_out_blocks=-1`.  The op then sizes its per-core chunk from
#: `shape[1] * shape[2] * shape[3]` -- `H*W*C`, with the batch nowhere in it -- so
#: at B>1 each core's block grows while the chunking stays put and the statically
#: allocated circular buffers run past L1.  `tt/vae.py::_group_norm_out_blocks` is
#: that same heuristic with the batch restored; at B=1 it returns the same -1.
_UPSTREAM_GROUPNORM_FIX = os.environ.get("F2K_VAE_UPSTREAM_GROUPNORM_FIX") == "1"

if _UPSTREAM_GROUPNORM_FIX:  # pragma: no cover - diagnostic path
    from models.tt_dit.layers import normalization as _tt_dit_norm

    _stock_group_norm_forward = _tt_dit_norm.GroupNorm.forward

    def _batch_aware_group_norm_forward(self, x, num_out_blocks=-1, compute_kernel_config=None):
        if num_out_blocks == -1:
            num_out_blocks = _group_norm_out_blocks(self, x)
        return _stock_group_norm_forward(self, x, num_out_blocks, compute_kernel_config)

    _tt_dit_norm.GroupNorm.forward = _batch_aware_group_norm_forward
    print("F2K_VAE_UPSTREAM_GROUPNORM_FIX=1: GroupNorm.forward chunks per batch", flush=True)

# The pipeline builds ``Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)``
# and ``vae_scale_factor = 2 ** (len(block_out_channels) - 1) = 8``.
IMAGE_PROCESSOR_SCALE_FACTOR = 16


# --------------------------------------------------------------------------- input


def _deterministic_image(height: int, width: int, *, seed: int = 0):
    """A fixed-seed RGB gradient with a few hard edges, as a PIL image.

    Host input prep.  Hard edges matter: a pure gradient is almost perfectly
    reconstructed by any autoencoder and would make PCC insensitive to real
    errors in the high-frequency path (the 3x3 convs and the downsamplers).
    """
    from PIL import Image

    generator = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0.0, 1.0, height).view(height, 1).expand(height, width)
    xs = torch.linspace(0.0, 1.0, width).view(1, width).expand(height, width)

    img = torch.stack([xs, ys, (1.0 - xs) * ys], dim=-1).clone()

    # a filled square, a ring and a diagonal stripe -- all deterministic
    yy = torch.arange(height).view(height, 1).expand(height, width).float()
    xx = torch.arange(width).view(1, width).expand(height, width).float()
    square = (yy > height * 0.15) & (yy < height * 0.45) & (xx > width * 0.55) & (xx < width * 0.85)
    radius = ((yy - height * 0.7) ** 2 + (xx - width * 0.3) ** 2).sqrt()
    ring = (radius > min(height, width) * 0.14) & (radius < min(height, width) * 0.2)
    stripe = (yy + xx).remainder(48.0) < 12.0

    img[square] = torch.tensor([0.95, 0.15, 0.05])
    img[ring] = torch.tensor([0.05, 0.2, 0.95])
    img[stripe] = img[stripe] * 0.35

    noise = torch.rand((height, width, 3), generator=generator)
    img = (img * 0.9 + noise * 0.1).clamp(0.0, 1.0)
    return Image.fromarray((img * 255.0).round().to(torch.uint8).numpy(), mode="RGB")


def _image_processor():
    """``Flux2ImageProcessor`` without instantiating the whole 9 B pipeline.

    ``reference.image_processor()`` reaches it through ``load_pipeline()``, which
    would also load the text encoder and the transformer -- 18 GB of host weights
    this stage has no use for.  Seeding the cache with the same object the
    pipeline would have built keeps ``reference.preprocess_image`` honest.
    """
    if "image_processor" not in reference._CACHE:
        module = importlib.import_module("diffusers.pipelines.flux2.image_processor")
        reference._CACHE["image_processor"] = module.Flux2ImageProcessor(vae_scale_factor=IMAGE_PROCESSOR_SCALE_FACTOR)
    return reference._CACHE["image_processor"]


# --------------------------------------------------------------------------- goldens


@torch.no_grad()
def _hf_reference_encode(hf_vae, pixel_values):
    """``AutoencoderKLFlux2.encode``: ``DiagonalGaussian(quant_conv(encoder(x)))``.

    Returns the three tensors the TT boundaries land on -- the raw encoder output
    (what ``encode*`` returns), the post-``quant_conv`` moments and the mode.
    """
    px = pixel_values.to(torch.float32)
    pre_quant = hf_vae.encoder(px)
    posterior = hf_vae.encode(px).latent_dist
    return pre_quant, posterior.parameters, posterior.mode()


@torch.no_grad()
def _hf_reference_decode(hf_vae, latents):
    """``AutoencoderKLFlux2.decode`` -- ``decoder(post_quant_conv(z))``."""
    return hf_vae.decode(latents.to(torch.float32), return_dict=False)[0]


@torch.no_grad()
def _hf_reference_roundtrip(pixel_values):
    """The Call-4 golden, straight from Source A."""
    return reference.hf_vae_roundtrip(pixel_values.to(torch.float32))[0]


# --------------------------------------------------------------------------- readback


def _to_torch(tensor, device):
    """Read one device's copy of a replicated ttnn tensor back to host fp32."""
    ttnn.synchronize_device(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    out = ttnn.to_torch(tensor, mesh_composer=composer)
    num_devices = device.get_num_devices()
    if out.shape[0] > 1 and out.shape[0] % num_devices == 0:
        out = out[: out.shape[0] // num_devices]
    return out.to(torch.float32)


def _report(name, tt, golden, label="stage PCC"):
    """`label` exists so a deliberately-ablated run reports under `ablated_corr`
    instead of `stage PCC` -- there a LOW score is the pass condition, and it must
    not read as one of this stage's measured PCCs."""
    value = reference.pcc(tt, golden)
    print(f"{label}={value:.6f}  route={name} shape={tuple(tt.shape)}", flush=True)
    return value


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def mesh_device():
    """``MeshShape(1, 8)`` + ``FABRIC_1D`` -- the opening of
    ``models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/test_decoder_sharded.py``,
    lifted to module scope (see the module docstring).

    TP=8 is a HARD requirement, not a preference: the graduated stubs shard their
    channels across exactly this mesh, so a narrower machine cannot run them at all.
    Too few devices is therefore a failure of this gate, not a reason to pass it
    quietly -- an unrun stage test proves nothing about the stage.

    ``l1_small_size`` is the per-chip L1_SMALL reservation the conv halo allocates
    from, and it has to be sized for the BATCH, not just the resolution: the
    bring-up convention of 24576 covers B=1 but at B=32 the encoder's first conv
    asks for 3200 B per bank and gets

        Out of Memory: Not enough space to allocate 204800 B L1_SMALL buffer
        across 64 banks ... (allocated: 23776 B, free: 800 B)

    so this fixture reserves ``L1_SMALL_SIZE``, default 24576 (the bring-up value,
    which every B=1 test here is green at) and overridable with ``F2K_VAE_L1_SMALL``.

    It is left at the B=1 value rather than raised because the batched window is
    measured and awkward.  L1_SMALL is carved off the same 1.5 MB the circular
    buffers and the L1-resident activations share, so:

      * the B=32 encoder needs ~64 KB.  Below that the halo runs out -- at 45056
        "allocated: 44896 B, free: 160 B", at 40960 a conv wants 2048 B per bank
        with 1568 free, and the demand tracks whatever it is given;
      * but at 65536 the B=32 DECODER programs come back with "static circular
        buffers ... clash with L1 buffers ... L1 buffer allocated at 1412096 and
        static circular buffer region ends at 1412288", and so does the rank-3
        attention test that is green at 24576.

    Chunking the group norms 4x finer left that region end at 1412288 to the byte,
    so the clashing buffers are not the norms\' and this stage cannot move them.
    Reproduce the B=32 encode numbers with ``F2K_VAE_L1_SMALL=65536``.
    """
    available = min(ttnn.get_num_devices(), ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size())
    assert available >= TP, (
        f"the VAE stage is graduated at TP={TP} and needs {TP} devices; this machine offers "
        f"{available}. Run the gate on a 1x8 mesh."
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, TP), l1_small_size=L1_SMALL_SIZE)
    yield device
    for submesh in device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def hf_vae():
    return reference.load_vae()


@pytest.fixture(scope="module")
def pixel_values():
    check_resolution(IMAGE_SIZE, IMAGE_SIZE)
    _image_processor()
    image = _deterministic_image(IMAGE_SIZE, IMAGE_SIZE)
    px = reference.preprocess_image(image, IMAGE_SIZE, IMAGE_SIZE).to(torch.float32)
    assert tuple(px.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(px.shape)
    return px


@pytest.fixture(scope="module")
def golden(hf_vae, pixel_values):
    pre_quant, moments, mode = _hf_reference_encode(hf_vae, pixel_values)
    assert tuple(mode.shape) == (1, 32, LATENT_SIZE, LATENT_SIZE), tuple(mode.shape)
    image = _hf_reference_decode(hf_vae, mode)
    assert tuple(image.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(image.shape)
    return {"pre_quant": pre_quant, "moments": moments, "mode": mode, "image": image}


@pytest.fixture(scope="module")
def ledger():
    return Ledger()


@pytest.fixture(scope="module")
def stage(mesh_device, hf_vae, ledger):
    return Flux2VaeStage(mesh_device, hf_vae, ledger=ledger)


# --------------------------------------------------------------------------- encode


def _run_encode(stage, route, pixel_values, golden):
    """The pipeline's own encode call: ``moments_to_mode(quant_conv(encode*(px)))``."""
    pre_quant = getattr(stage, route)(pixel_values)
    stage.mark_final(pre_quant)
    moments = stage.quant_conv(pre_quant)
    mode = stage.moments_to_mode(moments)

    host_pre = _to_torch(pre_quant, stage.device)
    host_mode = _to_torch(mode, stage.device)
    assert tuple(host_pre.shape) == (1, 64, LATENT_SIZE, LATENT_SIZE), tuple(host_pre.shape)
    assert tuple(host_mode.shape) == (1, 32, LATENT_SIZE, LATENT_SIZE), tuple(host_mode.shape)
    _report(f"{route}:pre_quant_conv", host_pre, golden["pre_quant"])
    _report(f"{route}:moments", _to_torch(moments, stage.device), golden["moments"])
    return host_mode


def test_encode_composite_pcc(stage, pixel_values, golden):
    """``encoder`` port + native ``quant_conv``, then ``moments_to_mode``."""
    host_mode = _run_encode(stage, "encode", pixel_values, golden)
    value = _report("encode", host_mode, golden["mode"])
    assert value >= PCC_TARGET, f"encode PCC {value} < {PCC_TARGET}"


def test_encode_alias_pcc(stage, pixel_values, golden):
    """``encoder_stack`` port: the same ``diffusers.Encoder`` as a second component,
    so it must match both the HF golden and this stage's own ``encode``."""
    host_alias = _run_encode(stage, "encode_alias", pixel_values, golden)
    host_encode = _run_encode(stage, "encode", pixel_values, golden)

    same = _report("encode_alias-vs-encode", host_alias, host_encode)
    value = _report("encode_alias", host_alias, golden["mode"])
    assert same >= 0.9999, f"encode_alias vs encode PCC {same} -- the same module must agree"
    assert value >= PCC_TARGET, f"encode_alias PCC {value} < {PCC_TARGET}"


def test_encode_blockwise_pcc(stage, pixel_values, golden):
    """``patch_embed`` -> 4x ``down_encoder_block2_d`` -> ``u_net_mid_block2_d``."""
    host_mode = _run_encode(stage, "encode_blockwise", pixel_values, golden)
    value = _report("encode_blockwise", host_mode, golden["mode"])
    assert value >= PCC_TARGET, f"encode_blockwise PCC {value} < {PCC_TARGET}"


def test_encode_decomposed_pcc(stage, pixel_values, golden):
    """``resnet_block2_d`` x2 + ``downsample2_d`` + ``down_encoder_block2_d`` x3 +
    ``resnet_block2_d`` + ``attention`` + ``resnet_block2_d``."""
    host_mode = _run_encode(stage, "encode_decomposed", pixel_values, golden)
    value = _report("encode_decomposed", host_mode, golden["mode"])
    assert value >= PCC_TARGET, f"encode_decomposed PCC {value} < {PCC_TARGET}"


# --------------------------------------------------------------------------- decode


def _run_decode(stage, route, latents, golden):
    """The pipeline's own decode call: ``decode*(post_quant_conv(latents))``."""
    image = getattr(stage, route)(stage.post_quant_conv(latents))
    stage.mark_final(image)
    host = _to_torch(image, stage.device)
    assert tuple(host.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(host.shape)
    return host


def test_decode_composite_pcc(stage, golden):
    """``post_quant_conv`` + ``decoder`` port, driven by the HF reference latents."""
    host = _run_decode(stage, "decode", golden["mode"], golden)
    value = _report("decode", host, golden["image"])
    assert value >= PCC_TARGET, f"decode PCC {value} < {PCC_TARGET}"


def test_decode_alias_pcc(stage, golden):
    """``decoder_head`` port -- the same ``diffusers.Decoder``, second component."""
    host = _run_decode(stage, "decode_alias", golden["mode"], golden)
    value = _report("decode_alias", host, golden["image"])
    assert value >= PCC_TARGET, f"decode_alias PCC {value} < {PCC_TARGET}"


def test_decode_decomposed_pcc(stage, golden):
    """``mlp`` + ``self_attention`` + ``mlp`` -> ``up_decoder_block2_d`` -> ``layer``
    -> ``mlp`` x3 + ``upsample2_d`` -> ``up_decoder_block2_d``."""
    host = _run_decode(stage, "decode_decomposed", golden["mode"], golden)
    value = _report("decode_decomposed", host, golden["image"])
    assert value >= PCC_TARGET, f"decode_decomposed PCC {value} < {PCC_TARGET}"


# --------------------------------------------------------------------------- call 4


def test_roundtrip_pcc(stage, pixel_values):
    """The real Call-4 chain, end to end on device.

    TT ``encode_decomposed`` -> ``moments_to_mode`` -> TT ``decode_decomposed``.
    The decode is fed the encode's OWN output; the reference latents are never
    injected at the joint, so an encoder error cannot be hidden.
    """
    moments = stage.encode_decomposed(pixel_values)
    stage.mark_final(moments)
    latents = stage.moments_to_mode(stage.quant_conv(moments))
    image = stage.decode_decomposed(stage.post_quant_conv(latents))
    stage.mark_final(image)

    host = _to_torch(image, stage.device)
    golden_image = _hf_reference_roundtrip(pixel_values)

    value = _report("roundtrip", host, golden_image)
    assert value >= PCC_TARGET, f"roundtrip PCC {value} < {PCC_TARGET}"


def test_trace_steps_host_op_free(stage, pixel_values, golden):
    """``encode_step`` / ``decode_step`` must fire no host aten op.

    ``pin_*`` absorbs all of it: the input staging, ``ttnn.prepare_conv_weights``
    for every conv, and ``CCLManager``'s per-shape ping-pong buffers (each built
    from a ``torch.empty``) plus its per-axis semaphores.  Everything there is
    cached by shape and the step re-runs identical shapes.
    """
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    encode_resident = stage.pin_encode(pixel_values)
    decode_resident = stage.pin_decode(stage.post_quant_conv(golden["mode"]))

    with observe_host_ops() as ops:
        encoded = stage.encode_step(encode_resident)
        decoded = stage.decode_step(decode_resident)
        ttnn.synchronize_device(stage.device)
    result = verdict(list(ops))
    print(f"stage PCC=n/a  route=trace_steps host_ops={result['n_host_ops']} {result['reason']}", flush=True)

    stage.mark_final(encoded)
    stage.mark_final(decoded)
    assert tuple(encoded.shape) == (1, 64, LATENT_SIZE, LATENT_SIZE), tuple(encoded.shape)
    assert tuple(decoded.shape) == (1, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(decoded.shape)

    # the pinned steps must still be numerically the composite routes
    host_mode = _to_torch(stage.moments_to_mode(stage.quant_conv(encoded)), stage.device)
    encode_pcc = _report("encode_step", host_mode, golden["mode"])
    decode_pcc = _report("decode_step", _to_torch(decoded, stage.device), golden["image"])

    assert result["on_device"], result["reason"]
    assert encode_pcc >= PCC_TARGET, f"encode_step PCC {encode_pcc} < {PCC_TARGET}"
    assert decode_pcc >= PCC_TARGET, f"decode_step PCC {decode_pcc} < {PCC_TARGET}"


def test_all_vae_stubs_routed(stage, ledger):
    """Coverage: after the routes above, every graduated VAE stub has actually run
    inside a real forward path."""
    print(ledger.table(), flush=True)
    expected = set(graduated_components("vae"))
    routed = set(ledger.routed()["vae"])
    assert not (expected - routed), f"unrouted VAE stubs: {sorted(expected - routed)}"

    idle = [row["name"] for row in ledger.rows() if row["calls"] == 0]
    assert not idle, f"bound but never called: {sorted(idle)}"

    # `downstream` is object-identity based, so a native op between two ports
    # breaks the chain -- reported, never asserted (see Ledger's docstring).
    print(f"stage PCC=n/a  route=coverage no_downstream={ledger.no_downstream()}", flush=True)


def test_decode_decomposed_ablation(stage, ledger, golden):
    """Load-bearing proof: neutralise a routed port and the head's PCC must fall.

    ``self_attention`` (``decoder.mid_block.attentions.0``) is the ablation target
    because it is shape-preserving, so an identity override is a legal drop-in and
    the only thing that changes is the arithmetic.  Coverage alone cannot tell a
    forward path from a sweep; this can.
    """
    latents = stage.post_quant_conv(golden["mode"])

    baseline = _report("ablation:baseline", _to_torch(stage.decode_decomposed(latents), stage.device), golden["image"])

    targets = ledger.ports("vae", "self_attention")
    assert targets, "self_attention was never bound"
    for port in targets:
        port.override(lambda x, *a, **k: x)
    try:
        ablated = _report(
            "ablation:self_attention-identity",
            _to_torch(stage.decode_decomposed(latents), stage.device),
            golden["image"],
            label="ablated_corr",
        )
    finally:
        ledger.restore_all()

    restored = _report("ablation:restored", _to_torch(stage.decode_decomposed(latents), stage.device), golden["image"])
    assert ablated < baseline - 0.005, (
        f"ablating self_attention left PCC at {ablated} vs baseline {baseline} -- "
        "the port's result is not reaching the head's output"
    )
    assert restored >= PCC_TARGET, f"restore failed: PCC {restored} < {PCC_TARGET}"


# ======================================================================= BATCH
#
# Everything below drives the SAME routes at a leading batch of B independent
# samples.  Three things are asserted that a mean PCC would not catch:
#
#   * per-sample PCC against THAT sample's own golden, printed for every sample;
#   * every TT sample's best-matching golden is its own (a dropped batch axis
#     makes row 0 the best match for every row);
#   * the samples are pairwise distinct, so the check above cannot be satisfied
#     by 32 copies of one image.


def _distinct_image(height: int, width: int, index: int):
    """One of a sequence of VISIBLY different RGB images -- not one image + noise.

    Same construction as ``_deterministic_image`` (gradient plus hard edges, which
    is what makes PCC sensitive to the high-frequency path), but the gradient
    direction, the square, the ring, the stripe period and the channel order all
    move with ``index``.  Samples that differ only by 10% noise would correlate at
    ~0.99 with each other and could not distinguish "32 samples" from "sample 0
    broadcast 32 times".
    """
    from PIL import Image

    generator = torch.Generator().manual_seed(1000 + index)
    ys = torch.linspace(0.0, 1.0, height).view(height, 1).expand(height, width)
    xs = torch.linspace(0.0, 1.0, width).view(1, width).expand(height, width)

    angle = 2.0 * math.pi * (index % 8) / 8.0
    ramp = (xs * math.cos(angle) + ys * math.sin(angle) + 1.0) * 0.5
    img = torch.stack([ramp, 1.0 - ramp, (ramp * (1.0 - ramp)) * 4.0], dim=-1).clone()
    img = img[..., [(index + c) % 3 for c in range(3)]]

    yy = torch.arange(height).view(height, 1).expand(height, width).float()
    xx = torch.arange(width).view(1, width).expand(height, width).float()
    cy = height * (0.2 + 0.6 * ((index * 5 % 7) / 6.0))
    cx = width * (0.2 + 0.6 * ((index * 3 % 5) / 4.0))
    side = min(height, width) * (0.10 + 0.12 * ((index % 4) / 3.0))
    square = (yy > cy - side) & (yy < cy + side) & (xx > cx - side) & (xx < cx + side)
    radius = ((yy - (height - cy)) ** 2 + (xx - (width - cx)) ** 2).sqrt()
    ring = (radius > min(height, width) * 0.14) & (radius < min(height, width) * 0.2)
    period = 16.0 + 8.0 * (index % 6)
    stripe = (yy + xx * (1 + index % 3)).remainder(period) < period / 3.0

    img[square] = torch.tensor([0.95, 0.15, 0.05])
    img[ring] = torch.tensor([0.05, 0.2, 0.95])
    img[stripe] = img[stripe] * 0.35

    noise = torch.rand((height, width, 3), generator=generator)
    img = (img * 0.9 + noise * 0.1).clamp(0.0, 1.0)
    return Image.fromarray((img * 255.0).round().to(torch.uint8).numpy(), mode="RGB")


def _batch_pixel_values(batch: int) -> torch.Tensor:
    """``(batch, 3, H, W)`` -- ``batch`` DIFFERENT real images, one stacked tensor."""
    check_resolution(IMAGE_SIZE, IMAGE_SIZE, batch)
    _image_processor()
    px = torch.cat(
        [
            reference.preprocess_image(_distinct_image(IMAGE_SIZE, IMAGE_SIZE, i), IMAGE_SIZE, IMAGE_SIZE)
            for i in range(batch)
        ],
        dim=0,
    ).to(torch.float32)
    assert tuple(px.shape) == (batch, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(px.shape)
    return px


#: batch -> {"px", "pre_quant", "moments", "mode", "image"}.  Built on demand and
#: only AFTER the device call it is compared against, so a route that cannot run
#: at this batch does not pay for a 32-sample torch VAE first.
_BATCH_GOLDEN: dict[int, dict] = {}


def _batch_golden(hf_vae, batch: int) -> dict:
    hit = _BATCH_GOLDEN.get(batch)
    if hit is not None:
        return hit
    px = _batch_pixel_values(batch)
    pre_quant, moments, mode = _hf_reference_encode(hf_vae, px)
    image = _hf_reference_decode(hf_vae, mode)
    assert tuple(mode.shape) == (batch, 32, LATENT_SIZE, LATENT_SIZE), tuple(mode.shape)
    assert tuple(image.shape) == (batch, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(image.shape)
    hit = {"px": px, "pre_quant": pre_quant, "moments": moments, "mode": mode, "image": image}
    _BATCH_GOLDEN[batch] = hit
    return hit


#: At most ONE batched stage is alive at a time.  A built port bakes
#: ``prepare_conv_weights(..., batch_size=b)`` into every conv it owns, so batches
#: cannot share ports; keeping two batches' worth of VAE weights resident as well
#: as the module-scoped B=1 stage's would be three copies for no reason.
_BATCH_STAGE: dict[int, tuple] = {}


def _batch_stage(mesh_device, hf_vae, batch: int):
    hit = _BATCH_STAGE.get(batch)
    if hit is not None:
        return hit[0]
    for other in list(_BATCH_STAGE):
        _, ledger = _BATCH_STAGE.pop(other)
        ledger.drop_ports("vae")
    gc.collect()
    ledger = Ledger()
    stage = Flux2VaeStage(mesh_device, hf_vae, ledger=ledger)
    _BATCH_STAGE[batch] = (stage, ledger)
    return stage


def _corr_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[B, B]`` of Pearson correlations: row i = TT sample i against every golden."""
    x = a.reshape(a.shape[0], -1).to(torch.float64)
    y = b.reshape(b.shape[0], -1).to(torch.float64)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x @ y.T


def _report_batch(route: str, tt: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    """Print every sample's own PCC and return the full TT-vs-golden matrix."""
    corr = _corr_matrix(tt, golden)
    for i, value in enumerate(corr.diagonal().tolist()):
        print(f"stage PCC={value:.6f}  route={route}[sample {i}] shape={tuple(tt.shape[1:])}", flush=True)
    return corr


def _assert_batched(route: str, tt: torch.Tensor, golden: torch.Tensor, batch: int) -> None:
    assert tuple(tt.shape) == tuple(golden.shape), f"{route}: {tuple(tt.shape)} vs {tuple(golden.shape)}"
    corr = _report_batch(route, tt, golden)
    own = corr.diagonal()

    # 1. every sample matches its own golden
    worst = int(own.argmin())
    assert float(own[worst]) >= BATCH_PCC_TARGET, (
        f"{route} at B={batch}: sample {worst} PCC {float(own[worst]):.6f} < {BATCH_PCC_TARGET}; "
        f"per-sample PCCs {[round(v, 5) for v in own.tolist()]}"
    )

    # 2. the samples really are 32 different results, not one broadcast
    tt_self = _corr_matrix(tt, tt)
    off = tt_self - torch.eye(batch, dtype=tt_self.dtype) * 2.0
    if batch > 1:
        assert float(off.max()) < 0.999, (
            f"{route} at B={batch}: two output samples are identical (max off-diagonal "
            f"self-correlation {float(off.max()):.6f}) -- the batch axis is not being carried"
        )

    # 3. and each one is matched to ITS OWN golden, not to sample 0's
    picked = corr.argmax(dim=1)
    mismatched = [(i, int(j)) for i, j in enumerate(picked.tolist()) if i != j]
    assert not mismatched, (
        f"{route} at B={batch}: sample(s) {mismatched} correlate better with another sample's "
        f"golden than with their own -- samples are being mixed or dropped"
    )
    print(
        f"stage PCC=n/a  route={route} batch={batch} min_per_sample={float(own.min()):.6f} "
        f"max_cross={float((corr - torch.eye(batch, dtype=corr.dtype) * 2.0).max()):.6f}",
        flush=True,
    )


def _batched_route(stage, route: str, x, batch: int):
    """Run one route once at ``batch``, turning the known blocker into a clear failure."""
    try:
        return getattr(stage, route)(x)
    except RuntimeError as error:
        if "circular buffers" not in str(error):
            raise
        blocked = stage.blocked_group_norms(batch)
        raise AssertionError(
            f"{route} at B={batch} died in ttnn.group_norm: {str(error).strip().splitlines()[0]}\n"
            f"This is the batch-blind chunking heuristic described in this module's "
            f"_UPSTREAM_GROUPNORM_FIX note -- GroupNorm.forward asks for num_out_blocks=-1 and the "
            f"op sizes the chunk from H*W*C with no batch in it. Norms this stage cannot pass a "
            f"chunk count to: {len(blocked)}. Re-run with F2K_VAE_UPSTREAM_GROUPNORM_FIX=1 to "
            f"confirm nothing else blocks this batch."
        ) from error


# ------------------------------------------------------------------ host-only


def test_legal_batch_matches_the_group_norm_multicast_rule(expect_error):
    """``ttnn.group_norm`` needs num_virtual_rows < N or divisible by N, for the
    four periods this checkpoint's norms run at -- so 3 is not a legal batch and
    32 is.  Checked on the host: at B=3 the op raises inside the mid block, minutes
    into a build, with a message about grids rather than about the batch."""
    assert [b for b in range(1, 65) if legal_batch(b)] == [1, 2, 4, 8, 16, 32, 64]
    assert legal_batch(BATCH_TARGET)
    assert legal_batch(96) and legal_batch(100)  # every period is below the batch
    check_batch(BATCH_TARGET)
    with expect_error(ValueError, "leading batch of 3"):
        check_batch(3)
    with expect_error(ValueError, "leading batch of 3"):
        check_resolution(IMAGE_SIZE, IMAGE_SIZE, 3)


def test_batch_inputs_are_distinct_images():
    """The batch really is B different images; if it were not, every per-sample
    assertion below would be satisfied by a stage that dropped the batch axis."""
    px = _batch_pixel_values(BATCH_TARGET)
    corr = _corr_matrix(px, px)
    off = corr - torch.eye(BATCH_TARGET, dtype=corr.dtype) * 2.0
    print(f"stage PCC=n/a  route=batch_inputs max_pairwise_corr={float(off.max()):.6f}", flush=True)
    assert float(off.max()) < 0.95, f"input images are not distinct enough: {float(off.max())}"


# -------------------------------------------------------------------- on device


@pytest.mark.parametrize("batch", BATCH_SIZES)
@pytest.mark.parametrize("route", ["encode", "encode_alias", "encode_blockwise", "encode_decomposed"])
def test_encode_batch_pcc(mesh_device, hf_vae, route, batch):
    """One call, ``batch`` different images in, ``batch`` different modes out.

    Also exercises ``quant_conv`` and ``moments_to_mode`` at the same batch -- the
    slice bound in ``moments_to_mode`` is the classic hardcoded leading 1.
    """
    stage = _batch_stage(mesh_device, hf_vae, batch)
    px = _batch_pixel_values(batch)

    pre_quant = _batched_route(stage, route, px, batch)
    stage.mark_final(pre_quant)
    moments = stage.quant_conv(pre_quant)
    mode = stage.moments_to_mode(moments)

    host_pre = _to_torch(pre_quant, stage.device)
    host_moments = _to_torch(moments, stage.device)
    host_mode = _to_torch(mode, stage.device)
    assert tuple(host_pre.shape) == (batch, 64, LATENT_SIZE, LATENT_SIZE), tuple(host_pre.shape)
    assert tuple(host_mode.shape) == (batch, 32, LATENT_SIZE, LATENT_SIZE), tuple(host_mode.shape)

    golden = _batch_golden(hf_vae, batch)
    _assert_batched(f"{route}:pre_quant_conv", host_pre, golden["pre_quant"], batch)
    _assert_batched(f"{route}:moments", host_moments, golden["moments"], batch)
    _assert_batched(route, host_mode, golden["mode"], batch)


@pytest.mark.parametrize("batch", BATCH_SIZES)
@pytest.mark.parametrize("route", ["decode", "decode_alias", "decode_decomposed"])
def test_decode_batch_pcc(mesh_device, hf_vae, route, batch):
    """One call, ``batch`` different latents in, ``batch`` different images out.

    The latents are the HF reference modes for the same ``batch`` distinct images,
    so no two of them are the same latent.  ``post_quant_conv`` runs at the batch too.
    """
    stage = _batch_stage(mesh_device, hf_vae, batch)
    golden = _batch_golden(hf_vae, batch)

    latents = stage.post_quant_conv(golden["mode"])
    image = _batched_route(stage, route, latents, batch)
    stage.mark_final(image)

    host = _to_torch(image, stage.device)
    assert tuple(host.shape) == (batch, 3, IMAGE_SIZE, IMAGE_SIZE), tuple(host.shape)
    _assert_batched(route, host, golden["image"], batch)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_roundtrip_batch_pcc(mesh_device, hf_vae, batch):
    """The Call-4 chain at a leading batch: TT encode -> mode -> TT decode.

    The decode eats the encode's own output, so a batch axis dropped anywhere in
    the encoder cannot be repaired by injecting reference latents at the joint.
    """
    stage = _batch_stage(mesh_device, hf_vae, batch)
    px = _batch_pixel_values(batch)

    moments = _batched_route(stage, "encode_decomposed", px, batch)
    stage.mark_final(moments)
    latents = stage.moments_to_mode(stage.quant_conv(moments))
    image = _batched_route(stage, "decode_decomposed", stage.post_quant_conv(latents), batch)
    stage.mark_final(image)

    host = _to_torch(image, stage.device)
    golden_image = _hf_reference_roundtrip(_batch_pixel_values(batch))
    _assert_batched("roundtrip", host, golden_image, batch)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_trace_steps_batch(mesh_device, hf_vae, batch):
    """``pin_encode`` / ``pin_decode`` pin the BATCH as well as the shapes.

    The pinned batch is part of the program: the warm-up forward is what bakes
    each conv's prepared weight, and that weight comes out of
    ``prepare_conv_weights(..., batch_size=b)``.
    """
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    stage = _batch_stage(mesh_device, hf_vae, batch)
    golden = _batch_golden(hf_vae, batch)
    px = _batch_pixel_values(batch)

    try:
        encode_resident = stage.pin_encode(px)
        decode_resident = stage.pin_decode(stage.post_quant_conv(golden["mode"]))
    except RuntimeError as error:
        if "circular buffers" not in str(error):
            raise
        raise AssertionError(f"pin_* at B={batch}: {str(error).strip().splitlines()[0]}") from error

    assert encode_resident["batch"] == batch, encode_resident["batch"]
    assert decode_resident["batch"] == batch, decode_resident["batch"]

    with observe_host_ops() as ops:
        encoded = stage.encode_step(encode_resident)
        decoded = stage.decode_step(decode_resident)
        ttnn.synchronize_device(stage.device)
    result = verdict(list(ops))
    print(
        f"stage PCC=n/a  route=trace_steps_batch{batch} host_ops={result['n_host_ops']} {result['reason']}", flush=True
    )

    stage.mark_final(encoded)
    stage.mark_final(decoded)
    host_mode = _to_torch(stage.moments_to_mode(stage.quant_conv(encoded)), stage.device)
    _assert_batched(f"encode_step:B{batch}", host_mode, golden["mode"], batch)
    _assert_batched(f"decode_step:B{batch}", _to_torch(decoded, stage.device), golden["image"], batch)
    assert result["on_device"], result["reason"]


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_attention_stub_rank3_keeps_samples_independent(mesh_device, hf_vae, batch):
    """The two attention stubs' rank-3 ``(B, L, C)`` entry, at a real batch.

    This is the one declared batch patch in ``tt/batch_patches/vae.json``: the
    graduated bodies inserted the spatial axis at 0, which makes the BATCH the H
    axis, so ``VaeAttention``'s ``reshape([n, 1, h*w, c])`` flattens all B samples
    into ONE attention sequence and every sample attends to every other sample.
    Moving the insert to axis 1 keeps them independent.  Golden is the diffusers
    ``Attention`` module on the same rank-3 batch.

    It runs at the mid block's own spatial extent (32x32 -> L=1024), which is small
    enough that the group-norm chunking that blocks the full routes does not apply
    here -- so this is a real B=32 result.
    """
    length = LATENT_SIZE * LATENT_SIZE
    cases = {
        "attention": hf_vae.encoder.mid_block.attentions[0],
        "self_attention": hf_vae.decoder.mid_block.attentions[0],
    }
    for name, torch_module in cases.items():
        channels = int(torch_module.group_norm.num_channels)
        torch.manual_seed(7)
        host = torch.randn(batch, length, channels, dtype=torch.float32)
        # Each sample is its own draw AND has its own first two moments, so a
        # dropped batch axis would show up in the group-norm statistics as well as
        # in the signal.  The spread is kept modest because the activation is
        # staged in bfloat16 and a large constant offset would cost mantissa bits
        # to cancellation, not to any effect this test is measuring.
        host = host * torch.linspace(0.8, 1.2, batch).view(batch, 1, 1)
        host = host + torch.linspace(-0.2, 0.2, batch).view(batch, 1, 1)

        with torch.no_grad():
            expected = torch_module(host)

        port = load_stub_module("vae", name).build(mesh_device, torch_module)
        tt = ttnn.from_torch(
            host.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = port(tt)
        assert tuple(out.shape) == (batch, length, channels), tuple(out.shape)
        _assert_batched(f"{name}:rank3", _to_torch(out, mesh_device), expected, batch)
        del port
        gc.collect()


def test_batch_stage_is_released(mesh_device, hf_vae):
    """The batched stage is dropped once the batch tests are done, so the module's
    own B=1 stage is not left sharing the device with a second set of weights."""
    for batch in list(_BATCH_STAGE):
        _, ledger = _BATCH_STAGE.pop(batch)
        ledger.drop_ports("vae")
    _BATCH_GOLDEN.clear()
    gc.collect()
    assert not _BATCH_STAGE


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-vv"]))
