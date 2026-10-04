# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the short prefill SwiGLU: `minimal_matmul(x, w_gu, fuse_swiglu=True)` re-laid over
the whole grid, bit for bit (kernels in tt/cpp_swiglu_mm_kernels).

minimal_matmul spreads its weight over grid.x = 11 column readers (M on the grid rows, N on the columns), so at
the text prefix's 5 tile rows it runs on 55 cores and streams the 32 MB bf4_b gate / up weight through 11 of
them (~126 GB/s, 252 us). Here each of ncores cores owns ALL rows of W fused (gate, up) column tiles and
streams just those columns of every K row, so every core reads weight.

The arithmetic is minimal_matmul's, tile for tile (see compute.cpp): the same K blocks (its K_block_size)
accumulate in a fresh 16-bit DEST at LoFi, are packed into a Float16_b accumulator (packer L1 accumulation from
the second block), and the same copy / silu (approximate) / SFPU multiply / bf16 pack produces each output tile.
The ORDER of the K blocks is minimal_matmul's too: it walks each core's N blocks in a snake (K forward, then
backward to reuse the resident in0 block, then forward ...), so an output tile in an odd N block of its
minimal_matmul core accumulates its blocks last to first -- and bf16 accumulation is order-sensitive. Each core
here owns tiles of ONE such N block and streams its K blocks in that block's order.

The data movement, measured per call (160 x 3072 x 18432, 96 cores; the math alone runs ~85 us):
  * weight: a RE-LAID copy (`relayout`) -- the same fp32 gate / up pairs, tiles permuted before the same
    conversion (each tile converts on its own), so every tile is bit-identical to minimal_matmul's. With
    kb > 0 it holds one tile row per K block and each core's kb x W block sits on one DRAM bank at consecutive
    addresses (interleaved page p lives on bank p % banks): one read per K block (reader_w_block.cpp); cores
    spread over the banks c % banks (grouping a bank's readers by K direction put neighbours on one bank:
    136 vs 116 us). The odd grid columns read it over the other NoC (_SPLIT_NOC: 116 -> 98 us).
  * x: MULTICAST (send_x.cpp, from a core outside the 96) -- each core reading all of x itself cost ~35 us,
    full-depth buffering or a rotated read order did not help; the multicast writes each K block to the same
    offset of every core's whole-x CB, and each core consumes its blocks in its own K order (recv_x_writer.cpp).
  252 us (minimal_matmul) -> 176 (interleaved tile reads) -> 137 (bank-contiguous rows) -> 123 (x multicast)
  -> 116 (bank-contiguous blocks) -> 98 (both NoCs).

Taken for <= 8 tile rows (the text prefix); on unless VOXTRAL_CPP_SWIGLU_MM=0.
"""
from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_swiglu_mm_kernels"
_READER_W = str(_DIR / "reader_w.cpp")
_READER_W_BANK = str(_DIR / "reader_w_bank.cpp")
_READER_W_BLOCK = str(_DIR / "reader_w_block.cpp")
_SEND_X = str(_DIR / "send_x.cpp")
_RECV_X = str(_DIR / "recv_x_writer.cpp")
_READER_X = str(_DIR / "reader_x_writer.cpp")
_COMPUTE = str(_DIR / "compute.cpp")

_TILE = 32
_MAX_MT = 8
_DEPTH = 4
# K blocks of weight read per barrier (the weight CB holds _DEPTH blocks): 2 measured slower than 1.
_BATCH = 1
# Odd grid columns read their weight over the other NoC (multicast form).
_SPLIT_NOC = True
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SWIGLU_MM", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_swiglu_mm: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def _rows(x):
    rows = 1
    for d in list(x.shape)[:-1]:
        rows *= int(d)
    return rows


def _reversed(n, pairs, n_block, grid_x):
    """Whether minimal_matmul (fuse_swiglu, N_block_size `n_block`, `grid_x` column cores) accumulates output
    tile n's K blocks last to first: pairs are padded to a multiple of grid_x and split evenly over the columns,
    each column walks its N blocks (n_block // 2 output tiles each) alternating K forward / backward."""
    per = -(-pairs // grid_x)
    return ((n % per) // (n_block // 2)) % 2 == 1


def _split(device, pairs, n_block, grid_x):
    """The most cores (<= the grid) that split the output column tiles evenly, with <= 8 fused tiles a core and
    every core's tiles in one K order."""
    grid = device.compute_with_storage_grid_size()
    cap = int(grid.x) * int(grid.y)
    for c in range(min(cap, pairs), 0, -1):
        pw = pairs // c
        if pairs % c or 2 * pw > 8:
            continue
        if all(
            len({_reversed(n, pairs, n_block, grid_x) for n in range(i * pw, (i + 1) * pw)}) == 1 for i in range(c)
        ):
            return c
    return None


class Relaid:
    """The fused weight re-laid for `ncores` cores over `banks` DRAM banks: core c (bank, slot group q from
    `bmap[c]`) owns original fused tiles [c * wf, (c + 1) * wf). BLOCKED (kb > 0): tile row blk holds K block
    blk, the core's kb x wf block (K-row major) at its bank's slots q * kb * wf ...; else the [K, 2N] columns
    are permuted, the core's wf tiles of each K row at its bank's slots q * wf ...."""

    def __init__(self, tensor, k, wn, ncores, banks, wf, kb, n_block, grid_x, bmap):
        self.tensor, self.k, self.wn = tensor, k, wn
        self.ncores, self.banks, self.wf, self.kb = ncores, banks, wf, kb
        self.n_block, self.grid_x, self.bmap = n_block, grid_x, bmap


def _bank_map(ncores, banks):
    """core c -> (bank c % banks, slot group c // banks): neighbouring cores load different banks (grouping the
    backward-K cores onto their own banks put neighbours on one bank: 136 vs 116 us)."""
    return [(c % banks, c // banks) for c in range(ncores)]


def relayout(pairs, device, make, n_block, grid_x, kb=0):
    """A Relaid copy of the `[K, 2N]` torch gate / up pairs, converted by `make` (the SAME conversion as the
    minimal_matmul weight, so each tile is bit-identical), or None when the shape has no bank-even split.
    `kb` > 0: one contiguous block per core per K block of kb."""
    if not enabled():
        return None
    import torch

    k, wn = int(pairs.shape[0]), int(pairs.shape[1])
    if k % _TILE or wn % (2 * _TILE) or (kb and (k // _TILE) % kb):
        return None
    kt, ntf = k // _TILE, wn // _TILE
    ncores = _split(device, ntf // 2, n_block, grid_x)
    banks = int(device.dram_grid_size().x)
    if ncores is None or ncores % banks or ntf % banks:
        return None
    wf = ntf // ncores
    bmap = _bank_map(ncores, banks)
    if not kb:
        idx = torch.empty(ntf, dtype=torch.long)
        for c in range(ncores):
            b, q = bmap[c]
            for i in range(wf):
                idx[(q * wf + i) * banks + b] = c * wf + i
        relaid = pairs.reshape(k, ntf, _TILE)[:, idx].reshape(k, wn).contiguous()
        return Relaid(make(relaid), k, wn, ncores, banks, wf, 0, n_block, grid_x, bmap)
    nb = kt // kb
    rows = torch.empty(kb * ntf, dtype=torch.long)
    cols = torch.empty(kb * ntf, dtype=torch.long)
    for c in range(ncores):
        b, q = bmap[c]
        for kk in range(kb):
            for i in range(wf):
                j = (q * kb * wf + kk * wf + i) * banks + b
                rows[j], cols[j] = kk, c * wf + i
    tiles = pairs.reshape(kt, _TILE, ntf, _TILE).permute(0, 2, 1, 3)  # [kt, ntf, 32, 32]
    blk_rows = torch.arange(nb).unsqueeze(1) * kb + rows.unsqueeze(0)  # [nb, kb * ntf]
    relaid = tiles[blk_rows, cols.unsqueeze(0).expand(nb, -1)]  # [nb, kb * ntf, 32, 32]
    relaid = relaid.permute(0, 2, 1, 3).reshape(nb * _TILE, kb * ntf * _TILE).contiguous()
    return Relaid(make(relaid), k, wn, ncores, banks, wf, kb, n_block, grid_x, bmap)


def supports(x, w, n_block, grid_x) -> bool:
    """x: bf8_b / bf16 TILE interleaved `[..., rows, K]`, rows a multiple of 32 up to 256; w: the bf4_b / bf8_b
    tile-pair-interleaved `[K, 2N]` minimal_matmul fuse_swiglu weight, interleaved (or its Relaid copy)."""
    try:
        if not enabled():
            return False
        if isinstance(w, Relaid):
            if (w.n_block, w.grid_x) != (n_block, grid_x):
                return False
            wk, wn, w = w.k, w.wn, w.tensor
        else:
            wk, wn = int(w.shape[-2]), int(w.shape[-1])
        k = int(x.shape[-1])
        rows = _rows(x)
        return (
            x.layout == ttnn.TILE_LAYOUT
            and w.layout == ttnn.TILE_LAYOUT
            and not x.is_sharded()
            and not w.is_sharded()
            and x.dtype in (ttnn.bfloat8_b, ttnn.bfloat16)
            and w.dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
            and [int(d) for d in x.padded_shape][-1] == k
            and rows % _TILE == 0
            and 1 <= rows // _TILE <= _MAX_MT
            and wk == k
            and k % _TILE == 0
            and wn % (2 * _TILE) == 0
            and n_block % 2 == 0
            and _split(x.device(), wn // (2 * _TILE), n_block, grid_x) is not None
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _cb(cores, index, dtype, tiles):
    size = _TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=tiles * size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
    )


def apply(x, w, kb, n_block, grid_x, memory_config=None):
    """`silu(x @ gate) * (x @ up)` bf16 `[..., rows, N]` (L1 interleaved unless `memory_config`), exactly as
    minimal_matmul(fuse_swiglu=True, K_block_size=kb, N_block_size=n_block, grid_x column cores, LoFi, 16-bit
    DEST, packer L1 acc) computes it."""
    device = x.device()
    relaid = w if isinstance(w, Relaid) else None
    if relaid is not None:
        if relaid.kb and relaid.kb != kb:
            raise RuntimeError(f"cpp_swiglu_mm: the re-laid weight has K blocks of {relaid.kb}, not {kb}")
        w, wn = relaid.tensor, relaid.wn
    else:
        wn = int(w.shape[-1])
    shape = [int(d) for d in x.shape]
    rows, k = _rows(x), shape[-1]
    mt, kt = rows // _TILE, k // _TILE
    ntf = wn // _TILE
    nt = ntf // 2
    if kt % kb:
        raise RuntimeError(f"cpp_swiglu_mm: K tiles {kt} not a multiple of the K block {kb}")
    nb = kt // kb
    ncores = _split(device, nt, n_block, grid_x)
    pw = nt // ncores
    wf = 2 * pw
    sw = wf
    grid = device.compute_with_storage_grid_size()
    gx = int(grid.x)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape[:-1] + [nt * _TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        memory_config or ttnn.L1_MEMORY_CONFIG,
    )
    xa, wa, ya = x.buffer_address(), w.buffer_address(), y.buffer_address()
    rw, rx, rc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    w_by, x_by = {}, {}
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        rev = int(_reversed(c * pw, nt, n_block, grid_x))
        if relaid is not None:
            bank, group = relaid.bmap[c]
            w_by[(cx, cy)] = [wa, bank, group * (relaid.kb or 1) * wf, rev]
        else:
            w_by[(cx, cy)] = [wa, c * wf, rev]
        rw[cx][cy] = w_by[(cx, cy)]
        rx[cx][cy] = [xa, ya, c * pw, rev, c]
        rc[cx][cy] = []
    if relaid is not None:
        if (relaid.ncores, relaid.wf) != (ncores, wf):
            raise RuntimeError("cpp_swiglu_mm: the re-laid weight was split for another core count")
        if relaid.kb:
            w_kernel, w_args = _READER_W_BLOCK, [kb, nb, wf, kb * ntf // relaid.banks]
        else:
            w_kernel, w_args = _READER_W_BANK, [kb, nb, wf, ntf // relaid.banks, _BATCH if nb % _BATCH == 0 else 1]
    else:
        w_kernel, w_args = _READER_W, [kb, nb, wf, ntf] + _accessor_args(w)
    gy = int(grid.y)
    # MULTICAST x from a core outside the compute cores (send_x.cpp) when there is one: read 96 times over the
    # NoC, x cost ~35 us of a ~150 us call.
    mcast = ncores < gx * gy
    kcores, sems = cores, []
    if mcast:
        full, rem = divmod(ncores, gx)
        rects = []
        if full:
            rects.append((ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1), gx * full))
        if rem:
            rects.append((ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full), rem))
        sender = ttnn.CoreCoord(gx - 1, gy - 1)
        snoc = device.worker_core_from_logical_core(sender)
        send_args = [xa]
        for lo, hi, n in rects + [(None, None, 0)] * (2 - len(rects)):
            if n:
                plo, phi = device.worker_core_from_logical_core(lo), device.worker_core_from_logical_core(hi)
                send_args += [int(plo.x), int(plo.y), int(phi.x), int(phi.y), n]
            else:
                send_args += [0, 0, 0, 0, 0]
        rs = ttnn.RuntimeArgs()
        rs[int(sender.x)][int(sender.y)] = send_args
        send_cores = ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)])
        kcores = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi) for lo, hi, _ in rects] + [ttnn.CoreRange(sender, sender)])
        sems = [
            ttnn.SemaphoreDescriptor(id=0, core_ranges=kcores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=1, core_ranges=kcores, initial_value=0),
        ]
        for c in range(ncores):
            cy, cx = divmod(c, gx)
            x_by[(cx, cy)] = [ya, c * pw, int(_reversed(c * pw, nt, n_block, grid_x)), int(snoc.x), int(snoc.y)]
            rx[cx][cy] = x_by[(cx, cy)]
            rc[cx][cy] = [int(_reversed(c * pw, nt, n_block, grid_x))]
        x_kernel = ttnn.KernelDescriptor(
            kernel_source=_RECV_X,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[mt, kb, nb, pw, nt] + _accessor_args(y),
            runtime_args=rx,
            config=ttnn.WriterConfigDescriptor(),
        )
    else:
        x_kernel = ttnn.KernelDescriptor(
            kernel_source=_READER_X,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[mt, kb, nb, pw, kt, nt] + _accessor_args(x) + _accessor_args(y),
            runtime_args=rx,
            config=ttnn.WriterConfigDescriptor(),
        )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=w_kernel,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=w_args,
            runtime_args=rw,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        x_kernel,
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[mt, kb, nb, wf, sw, int(mcast)],
            runtime_args=rc,
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    cfg = kernels[2].config
    # minimal_matmul's prefix config: LoFi, 16-bit half-sync DEST, approximate SFPU (the default), packer L1 acc
    # (switched per block in the kernel).
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = True
    cfg.dst_full_sync_en = False
    if mcast and _SPLIT_NOC:
        # Odd grid columns swap RISCs: their weight stream runs on the other NoC (and x / the output on this
        # one), so the 32 MB weight comes in over both NoCs.
        def _cols(parity):
            ranges = []
            for col in range(parity, gx, 2):
                last = full - 1 + (1 if col < rem else 0)
                if last >= 0:
                    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(col, 0), ttnn.CoreCoord(col, last)))
            return ttnn.CoreRangeSet(ranges)

        def _args(table, parity):
            ra = ttnn.RuntimeArgs()
            for (cx, cy), v in table.items():
                if cx % 2 == parity:
                    ra[cx][cy] = v
            return ra

        w_desc, x_desc = kernels[0], kernels[1]
        split = []
        for parity, w_cfg, x_cfg in (
            (0, ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()),
            (1, ttnn.WriterConfigDescriptor(), ttnn.ReaderConfigDescriptor()),
        ):
            split.append(
                ttnn.KernelDescriptor(
                    kernel_source=w_kernel,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_cols(parity),
                    compile_time_args=w_args,
                    runtime_args=_args(w_by, parity),
                    config=w_cfg,
                )
            )
            split.append(
                ttnn.KernelDescriptor(
                    kernel_source=_RECV_X,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_cols(parity),
                    compile_time_args=[mt, kb, nb, pw, nt] + _accessor_args(y),
                    runtime_args=_args(x_by, parity),
                    config=x_cfg,
                )
            )
        kernels = split + [kernels[2]]
    if mcast:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_SEND_X,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=send_cores,
                compile_time_args=[mt, kb, nb, kt] + _accessor_args(x),
                runtime_args=rs,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
    # Every CB on the multicaster too, so x's CB sits at the same address there as on the compute cores.
    cbs = [
        # x: the whole activation when multicast (one write per block, read in place), else a 2-block ring.
        _cb(kcores, 0, x.dtype, (nb if mcast else 2) * mt * kb),
        _cb(kcores, 1, w.dtype, _DEPTH * kb * wf),
        _cb(kcores, 16, ttnn.bfloat16, mt * pw),
        _cb(kcores, 24, ttnn.bfloat16, mt * wf),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    ttnn.generic_op([x, w, y], desc)
    return y
