# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HOST-side input encoding for FLUX.2-klein-9B: latent layout, position ids, the
flow-match schedule, id padding and the host->device staging of token ids.

This module sits OUTSIDE ``tt/`` on purpose, and the split is the same one the trace
contract and the host-op observer draw: ``tt/`` is the DEVICE FORWARD PATH, and input
encoding happens before it and outside it.  Everything in here runs on the host, by
design, once per input -- so keeping it in the pipeline package would make every
reader (and every scanner) of ``tt/`` have to re-derive which ``torch`` calls are the
forward and which are the prep.  None of it is inside a traced step, none of it is in
a decode loop, and none of it touches model weights.

Two families:

* **Layout / schedule** -- pure index, shape and scale arithmetic lifted from
  ``diffusers.pipelines.flux2.pipeline_flux2_klein.Flux2KleinPipeline``.  It is
  reimplemented rather than called off the pipeline object so the TT forward path
  never reaches into an HF object, and ``tests/e2e/test_layout_parity.py`` asserts
  every function here is bit-identical to the reference's own staticmethod.
* **Staging** -- padding a prompt out to a pinned capacity, and uploading token ids
  as ``uint32`` / ``ROW_MAJOR`` (never bfloat16, see ``stage_ids``), plus the
  device->host readback the stop-token test needs.

The one piece of maths that does touch a tensor value -- the flow-match Euler
update and the BatchNorm latent (de)normalisation -- lives on the DEVICE in
``tt/pipeline.py``; this module only hands over the scalars.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn

VAE_SCALE_FACTOR = 8  # 2 ** (len(block_out_channels) - 1) for [128,256,512,512]
#: latent H/W must be divisible by the 2x2 packing, so the pipeline uses 2x this
LATENT_MULTIPLE = VAE_SCALE_FACTOR * 2


# ------------------------------------------------------------------ layout


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,4C,H/2,W/2)."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,4C,H,W) -> (B,C,2H,2W)."""
    b, c, h, w = latents.shape
    x = latents.reshape(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // 4, h * 2, w * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,H*W,C)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scatter (B,N,C) tokens back to (B,C,height,width) using their h/w ids."""
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        flat = h_ids * width + w_ids
        out = torch.zeros((height * width, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat.unsqueeze(1).expand(-1, ch), data)
        out_list.append(out.view(height, width, ch).permute(2, 0, 1))
    return torch.stack(out_list, dim=0)


# ------------------------------------------------------------- position ids


def text_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """(B,L,4) -- t/h/w all zero, the 4th axis carries the token index."""
    out = []
    for _ in range(batch_size):
        out.append(torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len)))
    return torch.stack(out)


def latent_ids(batch_size: int, height: int, width: int) -> torch.Tensor:
    """(B,H*W,4) for the packed latent grid: t=0, h, w, l=0."""
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return ids.unsqueeze(0).expand(batch_size, -1, -1)


def image_ids(shapes: list[tuple[int, int]], scale: int = 10) -> torch.Tensor:
    """(1,N_total,4) for a SEQUENCE of reference latents.

    Each reference gets its own T coordinate ``scale + scale*i`` so the transformer
    can tell the references apart; ``shapes`` is the list of (h, w) of each
    reference's PATCHIFIED latent.
    """
    out = []
    for i, (h, w) in enumerate(shapes):
        t = torch.tensor([scale + scale * i])
        out.append(torch.cartesian_prod(t, torch.arange(h), torch.arange(w), torch.arange(1)))
    return torch.cat(out, dim=0).unsqueeze(0)


def latent_grid(height: int, width: int) -> tuple[int, int]:
    """Packed-latent grid (h, w) for a target image size, per ``prepare_latents``."""
    h = 2 * (int(height) // LATENT_MULTIPLE)
    w = 2 * (int(width) // LATENT_MULTIPLE)
    return h // 2, w // 2


# --------------------------------------------------------------- schedule


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """``compute_empirical_mu`` -- the dynamic-shift parameter the Klein pipeline
    feeds ``set_timesteps``; the schedule is wrong without it."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def schedule(scheduler, num_inference_steps: int, image_seq_len: int):
    """(timesteps, sigmas) from the model's own scheduler, exactly as the pipeline
    obtains them: default sigmas ``linspace(1, 1/N, N)`` plus dynamic shifting."""
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    return scheduler.timesteps, scheduler.sigmas


def euler_deltas(sigmas: torch.Tensor) -> list[float]:
    """``dt = sigma_next - sigma`` per step (stochastic_sampling is false for this
    checkpoint, so the update is the plain Euler ``sample + dt * model_output``)."""
    return [float(sigmas[i + 1] - sigmas[i]) for i in range(len(sigmas) - 1)]


# ------------------------------------------------------- bn latent statistics


def bn_stats(vae) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-level BatchNorm the pipeline uses to (de)normalise packed latents."""
    mean = vae.bn.running_mean.detach().reshape(1, -1, 1, 1).float()
    std = torch.sqrt(vae.bn.running_var.detach().reshape(1, -1, 1, 1).float() + vae.config.batch_norm_eps)
    return mean, std


# --------------------------------------------------------------- id staging


def joint_image_ids(latent_ids_table: torch.Tensor, ref_ids_table: torch.Tensor) -> torch.Tensor:
    """The edit head's image-side id table: the denoised grid FIRST, then every
    reference latent's grid, along the token axis.  Order matters -- the transformer
    reads position from this table alone, and the stream it is paired with is
    ``[latents | references]``."""
    return torch.cat([latent_ids_table, ref_ids_table], dim=1)


def slot_pixels(images, batch, height: int, width: int, preprocess) -> torch.Tensor:
    """The pipeline's image front end over one or many images -> ``(B, C, H, W)``.

    ``batch=None`` takes the batch from ``images`` itself; an int broadcasts a single
    shared image across the batch and otherwise requires one image per row.
    ``preprocess`` is ``reference.preprocess_image`` (``Flux2ImageProcessor.preprocess``),
    passed in rather than imported so this module keeps no dependency on the HF side.

    This lives here and not in ``tt/`` for the reason at the top of this file: resize,
    crop, scale and the row ``cat`` are input ENCODING, they run once per input on the
    host, and none of it belongs in the file that holds the device forward path.
    """
    items = list(images) if isinstance(images, (list, tuple)) else [images]
    if batch is not None:
        if len(items) == 1:
            items = items * int(batch)
        if len(items) != int(batch):
            raise ValueError(f"reference slot holds {len(items)} images but the batch is {batch}")
    return torch.cat([preprocess(im, height, width) for im in items], dim=0)


def pad_ids(ids: torch.Tensor, mask: torch.Tensor | None, capacity: int, pad_id: int):
    """Pin the sequence axis to ``capacity``, masking the padded positions so the
    output on ``[0:real_len]`` is unchanged.

    Right-padding, because that is what both the tokenizer and every ``pin_*`` in the
    pipeline assume: the real tokens are the leading run.
    """
    length = int(ids.shape[1])
    if mask is None:
        mask = torch.ones_like(ids)
    if length > capacity:
        return ids[:, :capacity], mask[:, :capacity]
    pad = capacity - length
    ids = torch.cat([ids, torch.full((ids.shape[0], pad), int(pad_id), dtype=ids.dtype)], dim=1)
    mask = torch.cat([mask, torch.zeros((mask.shape[0], pad), dtype=mask.dtype)], dim=1)
    return ids, mask


# ------------------------------------------------------- host <-> device


def is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids_fn = getattr(device, "get_device_ids", None)
    if callable(ids_fn):
        try:
            return max(1, len(ids_fn()))
        except Exception:
            pass
    return 1


def replicate_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if is_mesh(device) else None


def to_host(tensor, device) -> torch.Tensor:
    """Read a REPLICATED device tensor back to host.  Pure marshalling.

    Uses the ``ConcatMeshToTensor`` composer the bring-up harness settled on --
    bare ``to_torch`` / ``get_device_tensors()[0]`` can busy-loop on a mesh --
    then drops the duplicate shards, since every chip holds the same rows.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    try:
        ttnn.synchronize_device(device)
    except Exception:
        pass
    if not is_mesh(device):
        return ttnn.to_torch(tensor)
    host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    n = num_devices(device)
    if n > 1 and host.shape[0] > 1 and host.shape[0] % n == 0:
        host = host[: host.shape[0] // n]
    return host


def stage_ids(tokens, device):
    """Token ids on device as uint32 / ROW_MAJOR -- never bfloat16.

    bfloat16 carries 8 mantissa bits, so it cannot represent an id above 256
    exactly; a vocab of 151936 needs 18.  Casting ids silently gathers a
    neighbouring embedding row.
    """
    if not isinstance(tokens, torch.Tensor):
        return tokens  # already staged
    return ttnn.from_torch(
        tokens.reshape(1, -1).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=replicate_mapper(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
