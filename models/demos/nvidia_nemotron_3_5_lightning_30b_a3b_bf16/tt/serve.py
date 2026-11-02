# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The serving pipeline for NVIDIA-Nemotron-3.5-Lightning-30B-A3B on the QB2 mesh: up to 32 users, each on its own
device row with its own state, sampled on device.

    ids -> embedding -> 52 x [norm -> mamba2 mixer | attention | MoE, + residual] -> norm_f -> lm_head -> sampler

Every layer of a kind runs one implementation: the Mamba2 mixer stub (_stubs/nemotron_h_mamba2_mixer.py), the MoE
stub (_stubs/nemotron_h_mo_e.py) and the attention below (ttnn SDPA over a per-user K,V cache).

Per-user state, one device row per user (B = 32 rows):
  mamba      ssm (G N, B, H P / G) fp32 (users as tile rows, see SSMStep) and the conv window prev1..3 (1, B, conv_dim)
             fp32, stepped in place: the conv by the stub's decode helper (_stubs/_ssm_cache.py), the recurrence by
             tt/kernels/ssm_step_*.cpp
  attention  K, V caches (B, 1, KV_CAPACITY, head_dim) bf16 (each chip holds the kv head its query heads read), written
             at each row's own position (cur_pos)

Prefill runs the prompts of one call as one batch, each padded after its end to the longest (a multiple of 32). The
padding never reaches a real position: attention is causal, and the Mamba2 step size dt is zeroed at padded
positions, which leaves the SSD state exactly as the prompt's last real token left it. Prompts longer than one SSD
chunk (128) run chunk by chunk, each chunk starting from the state (and conv window) the previous one left. Each
prompt's state goes into its own user's row only (tt/kernels/ssm_rows.cpp, kv_rows.cpp); every other row is left as
it is, so requests join and leave while the others decode.
"""
from __future__ import annotations

import os
import struct
import time

import torch

import ttnn
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import _dram_mm, _sparse_moe, _ssm_cache
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_mamba2_mixer as _mamba_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_mo_e as _moe_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import nemotron_h_r_m_s_norm as _rmsnorm_stub
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt.sampler import _KERNELS, Sampler, _cb, _first_cores

BATCH = 32
KV_CAPACITY = int(os.environ.get("NEMOTRON_KV_CAPACITY", "2048"))
# Candidates the device sampler keeps per row: generation_config sets top_p 0.95 and no top_k.
SAMPLER_TOP_K = 128
SSD_CHUNK = 128
TILE = ttnn.TILE_SIZE


def _mapper(device):
    return ttnn.ReplicateTensorToMesh(device)


def read_host(t) -> torch.Tensor:
    """Chip 0's copy of a device tensor that is the same on every chip."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _f32(t):
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)


def _ckc(fidelity=ttnn.MathFidelity.HiFi4):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _cores(n, grid):
    return _first_cores(n, grid)


def _tile_bytes(dtype):
    return {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.uint32: 4096, ttnn.int32: 4096}[dtype]


def kv_rows(device, k, v, k_cache, v_cache, slots):
    """K, V [n, 1, T, D] bf16 -> positions [0, T) of the cache rows slots[i] (tt/kernels/kv_rows.cpp)."""
    n, hkv, T, D = (int(s) for s in k.shape)
    w, rt_, ct = D // TILE, T // TILE, int(k_cache.shape[-2]) // TILE
    total = 2 * n * hkv * rt_ * w
    g = device.compute_with_storage_grid_size()
    per = -(-total // (g.x * g.y))
    xy, cores = _cores(-(-total // per), g)
    io = [k, v, k_cache, v_cache]
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(xy):
        rt[cx][cy] = [t.buffer_address() for t in io] + [i * per, min(total, (i + 1) * per)] + [int(s) for s in slots]
    slots_l1 = min(per, 4)
    tile = _tile_bytes(ttnn.bfloat16)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "kv_rows.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[n, hkv, rt_, w, ct, tile, slots_l1]
        + [a for t in io for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = _cb(0, ttnn.bfloat16, tile, slots_l1, cores)
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))


def _accessors(*ts):
    return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]


def _kernel(name, cores, ct, rt, config):
    return ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / name),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=config,
    )


class SSMStep:
    """The Mamba2 decode recurrence of one layer as one program (tt/kernels/ssm_step_{reader,compute,writer}.cpp),
    straight from the conv output: S <- S exp(dt A) + B (x dt), y = C S, every product in fp32. The state is held with
    the users as tile rows, (G N, 32, H P / G): the per-user scalars broadcast along tile rows, so no input is split
    into heads. Each core owns one group, NC column tiles and a range of state rows n; its y over those rows is one
    slab of the (R, 32, H P) partials, summed by the caller."""

    NC = 2  # column tiles per core: fewer means less of x, dt, B, C read per core (it is bound by DRAM traffic)

    def __init__(self, device, m, batch):
        if batch != TILE:
            raise ValueError(f"the decode state keeps the {batch} users as one tile's rows: needs {TILE}")
        self.device = device
        G, N, H, P = m.n_groups, m.ssm_state_size, m.num_heads, m.head_dim
        HPG, CPH = H // G, P // TILE
        CPG = HPG * CPH
        nc = self.NC
        grid = device.compute_with_storage_grid_size()
        cols = G * (CPG // nc)
        R = max(-(-N // 32), min(N, grid.x * grid.y // cols))  # the reader holds <= 2 B / C tiles: <= 32 rows each
        bounds = [N * r // R for r in range(R + 1)]
        self.work = [
            (g, q * nc, bounds[r], bounds[r + 1], r) for r in range(R) for g in range(G) for q in range(CPG // nc)
        ]
        self.xy, self.cores = _cores(len(self.work), grid)
        self.R, self.G, self.N, self.HPG, self.CPH, self.CPG = R, G, N, HPG, CPH, CPG
        self.x_tiles, self.bc_tiles = H * P // TILE, G * N // TILE
        pages = [(0, 2 * nc), (1, nc), (2, nc), (3, nc), (4, 2), (6, nc), (7, 6), (16, 2 * nc), (17, nc)]
        self.cbs = [_cb(i, ttnn.float32, _tile_bytes(ttnn.float32), n, self.cores) for i, n in pages]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(
            [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        )
        self.compute_cfg = cfg
        # before / after the recurrence (pre, post): one core per conv-channel tile plus one for dt; one per y tile
        self.I, self.conv_dim, self.H, self.K = m.intermediate_size, m.conv_dim, H, m.conv_k
        self.pre_xy, self.pre_cores = _cores(self.conv_dim // TILE + 1, grid)
        self.post_xy, self.post_cores = _cores(self.I // TILE, grid)
        tb = _tile_bytes(ttnn.float32)
        self.pre_cbs = [
            _cb(i, ttnn.float32, tb, n, self.pre_cores) for i, n in ((0, self.K), (1, self.K + 1), (16, 1), (17, 1))
        ]
        self.post_cbs = [_cb(i, ttnn.float32, tb, n, self.post_cores) for i, n in ((0, R), (1, 3), (16, 1))]

    def _io(self, name, xy, cores, ct, reads, writes, roles, cbs, tensors):
        """One program of tiles_in.cpp / name / tiles_out.cpp: per core, its reads (cb, src, tile[, dst, tile]),
        writes (cb, dst, tile) and role (the compute kernel's runtime arg 0)."""
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addr = lambda t: t.buffer_address() if t is not None else 0  # noqa: E731
        for (cx, cy), r, w, role in zip(xy, reads, writes, roles):
            rd[cx][cy] = [len(r)] + [
                v
                for e in r
                for v in (e[0], addr(e[1]), e[2], addr(e[3]) if len(e) > 3 else 0, e[4] if len(e) > 3 else 0)
            ]
            cp[cx][cy] = [role]
            wr[cx][cy] = [len(w)] + [v for e in w for v in (e[0], addr(e[1]), e[2])]
        acc = _accessors(tensors[0])
        kernels = [
            _kernel("tiles_in.cpp", cores, acc, rd, ttnn.ReaderConfigDescriptor()),
            _kernel(name, cores, ct, cp, self.compute_cfg),
            _kernel("tiles_out.cpp", cores, acc, wr, ttnn.WriterConfigDescriptor()),
        ]
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def pre(self, proj, m, conv_consts, dt_bias_full, A_full):
        """proj (1, 32, gate | x B C | dt) the in_proj output -> hc = silu(conv) (1, 32, conv_dim), the conv window
        (m._state prev1..) advanced; dt = softplus(dt + dt_bias) and exp(dt A), (1, 32, H) (mamba_pre_compute.cpp)."""
        K, CT = self.K, self.conv_dim // TILE
        XT, DT = self.I // TILE, (self.I + self.conv_dim) // TILE
        mk = lambda w: ttnn.empty(
            [1, TILE, w], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
        )  # noqa: E731
        hc, dt, dec = mk(self.conv_dim), mk(self.H), mk(self.H)
        win = [proj] + [m._state[f"prev{s}"] for s in range(1, K)]
        reads, writes = [], []
        for t in range(CT):
            r = [(0, win[s], XT + t if s == 0 else t) + ((win[s + 1], t) if s + 1 < K else ()) for s in range(K)]
            reads.append(r + [(1, conv_consts, s * CT + t) for s in range(K + 1)])
            writes.append([(16, hc, t)])
        reads.append([(0, proj, DT), (1, dt_bias_full, 0), (1, A_full, 0)])
        writes.append([(16, dt, 0), (17, dec, 0)])
        self._io(
            "mamba_pre_compute.cpp",
            self.pre_xy,
            self.pre_cores,
            [K],
            reads,
            writes,
            [0] * CT + [1],
            self.pre_cbs,
            [proj, *win[1:], conv_consts, dt_bias_full, A_full, hc, dt, dec],
        )
        return hc, dt, dec

    def post(self, parts, hc, proj, D_full):
        """(sum of the partials + D x) silu(gate) -> (1, 32, I) (mamba_post_compute.cpp); x, gate read from hc, proj."""
        y = ttnn.empty([1, TILE, self.I], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        YT = self.I // TILE
        reads = [
            [(0, parts, r * YT + c) for r in range(self.R)] + [(1, hc, c), (1, D_full, c), (1, proj, c)]
            for c in range(YT)
        ]
        writes = [[(16, y, c)] for c in range(YT)]
        self._io(
            "mamba_post_compute.cpp",
            self.post_xy,
            self.post_cores,
            [self.R],
            reads,
            writes,
            [0] * YT,
            self.post_cbs,
            [parts, hc, proj, D_full, y],
        )
        return y

    def state_shape(self, batch):
        return (self.G * self.N, batch, self.CPG * TILE)

    def __call__(self, S, hc, dt, dec):
        """S the state (stepped in place), hc (1, 32, conv_dim) the conv output, dt / dec (1, 32, H) softplus(dt +
        dt_bias) and exp(dt A) -> y partials (R, 32, H P) fp32: y = their sum + D x."""
        nc = self.NC
        y = ttnn.empty(
            [self.R, TILE, self.G * self.CPG * TILE], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addr = [t.buffer_address() for t in (S, hc, dt, dec)]
        for (cx, cy), (g, c0, n0, n1, r) in zip(self.xy, self.work):
            rd[cx][cy] = addr + [g, c0, n0, n1]
            cp[cx][cy] = [n1 - n0]
            wr[cx][cy] = [addr[0], y.buffer_address(), g, c0, n0, n1, r]
        kernels = [
            _kernel(
                "ssm_step_reader.cpp",
                self.cores,
                [nc, self.N, self.x_tiles, self.bc_tiles, self.HPG, self.CPH] + _accessors(S, hc, dt, dec),
                rd,
                ttnn.ReaderConfigDescriptor(),
            ),
            _kernel("ssm_step_compute.cpp", self.cores, [nc], cp, self.compute_cfg),
            _kernel(
                "ssm_step_writer.cpp",
                self.cores,
                [nc, self.N, self.CPG, self.G * self.CPG] + _accessors(S, y),
                wr,
                ttnn.WriterConfigDescriptor(),
            ),
        ]
        ttnn.generic_op([S, hc, dt, dec, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=self.cbs))
        return y

    def rows(self, src, dst, slots):
        """Prefill: the SSD states src (n, H, N, P) fp32 -> rows slots[i] of the decode state dst
        (tt/kernels/ssm_rows.cpp); every other row is left as it is."""
        n, H = int(src.shape[0]), int(src.shape[1])
        total = n * self.G * self.N * self.CPG
        g = self.device.compute_with_storage_grid_size()
        per = -(-total // (g.x * g.y))
        xy, cores = _cores(-(-total // per), g)
        rt = ttnn.RuntimeArgs()
        # one slot per state row: a program cached for another prompt count must find as many runtime args
        rows = [int(s) for s in slots] + [0] * (int(dst.shape[1]) - n)
        for i, (cx, cy) in enumerate(xy):
            rt[cx][cy] = [src.buffer_address(), dst.buffer_address(), i * per, min(total, (i + 1) * per)] + rows
        slots_l1 = min(per, 32)
        kernel = _kernel(
            "ssm_rows.cpp",
            cores,
            [H, self.N, self.CPH, self.HPG, self.G, slots_l1] + _accessors(src, dst),
            rt,
            ttnn.ReaderConfigDescriptor(),
        )
        cb = _cb(0, ttnn.float32, slots_l1 * 2 * 64, 1, cores)
        ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))


class Router:
    """The MoE gate for one tile row of tokens: the stub's fp32 logits and sigmoid, then the top-k and this chip's
    routing weights in one program (tt/kernels/router_topk.cpp), one core per row; the same selection and weights as
    the stub's _route, without its all-pairs rank."""

    def __init__(self, device, m):
        self.device, self.m = device, m
        tp = m._TP if m._shard else 1
        chip = torch.arange(tp, dtype=torch.int32).repeat_interleave(TILE).reshape(1, 1, tp * TILE).expand(1, TILE, -1)
        mapper = (
            ttnn.ShardTensor2dMesh(device, mesh_shape=list(device.shape), dims=(None, 2))
            if m._shard
            else _mapper(device)
        )
        self.chip = ttnn.from_torch(
            chip.contiguous(), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
        )
        self.xy, self.cores = _cores(TILE, device.compute_with_storage_grid_size())
        scale = struct.unpack("<I", struct.pack("<f", m.routed_scaling_factor))[0]
        self.ct = [m.n_routed_experts, m.top_k, m._Eloc, int(m.norm_topk_prob), scale]
        self.cb = _cb(0, ttnn.float32, 2048, 1, self.cores)

    def __call__(self, hs):
        """hs (1, 32, hidden) fp32 -> (1, 32, Eloc) fp32."""
        m = self.m
        scores = ttnn.sigmoid(ttnn.matmul(hs, m._gate_w, compute_kernel_config=m.ckc))
        choice = ttnn.add(scores, m._bias)
        out = ttnn.empty([1, TILE, m._Eloc], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        io = [choice, scores, self.chip, out]
        rt = ttnn.RuntimeArgs()
        for b, (cx, cy) in enumerate(self.xy):
            rt[cx][cy] = [t.buffer_address() for t in io] + [b]
        kernel = _kernel("router_topk.cpp", self.cores, self.ct + _accessors(*io), rt, ttnn.ReaderConfigDescriptor())
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[self.cb]))
        ttnn.deallocate(choice)
        ttnn.deallocate(scores)
        return out


def _shards(device, t):
    """A width-sharded L1 tensor's shards in tile order: (their cores' NoC coordinates, tiles per shard, L1 address)."""
    spec = t.memory_config().shard_spec
    cores = ttnn.corerange_to_cores(spec.grid, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
    xy = [device.worker_core_from_logical_core(c) for c in cores]
    return [(c.x, c.y) for c in xy], int(spec.shape[1]) // TILE, t.buffer_address()


def _runs(shards, t0, n, page, cb):
    """Tiles [t0, t0 + n) of one tile row of a width-sharded tensor as (cb, x, y, L1 address, tiles), one per shard."""
    xy, per, base = shards
    runs, t = [], t0
    while t < t0 + n:
        c, o = divmod(t, per)
        k = min(per - o, t0 + n - t)
        runs.append((cb, *xy[c], base + o * page, k))
        t += k
    return runs


def _fp32_dst_cfg(fp32_cbs):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in fp32_cbs:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    return cfg


class ExpertAct:
    """MoE decode between an up and a down matmul as one program (tt/kernels/l1_in.cpp, moe_act_compute.cpp,
    l1_out.cpp): the first n_tiles tiles of the up matmul's L1 output relu(x W_up), squared and, with scale, times each
    row's routing weight for the tile's expert (group tiles per expert), written straight into the down matmul's L1
    input for a (k, n) weight; in place of moving the output to DRAM, slicing off the padding, expanding the weights
    to the bank's width, the multiply and resharding for the down matmul."""

    def __init__(self, device, k, n, group, dtype_in, dtype_out, scale):
        self.device, self.k, self.dtype_out, self.scale = device, k, dtype_out, scale
        self.cfg_out = _dram_mm.in0_memcfg(device, k, n)
        grid = device.compute_with_storage_grid_size()
        n_tiles = k // TILE
        groups = n_tiles // group
        split = max(1, min(group, grid.x * grid.y // groups))
        self.work = [
            (g, g * group + group * s // split, group * (s + 1) // split - group * s // split)
            for g in range(groups)
            for s in range(split)
        ]
        self.xy, self.cores = _cores(len(self.work), grid)
        most = max(w[2] for w in self.work)
        self.page_in, self.page_out = _tile_bytes(dtype_in), _tile_bytes(dtype_out)
        self.cbs = [
            _cb(0, dtype_in, self.page_in, most, self.cores),
            _cb(16, dtype_out, self.page_out, most, self.cores),
        ]
        if scale:
            self.cbs += [_cb(1, ttnn.float32, 4096, 1, self.cores), _cb(2, ttnn.float32, 4096 + 64, 1, self.cores)]
        self.compute_cfg = _fp32_dst_cfg([1])

    def __call__(self, up, W=None):
        """up the up matmul's raw output, W (1, 32, experts) fp32 the routing weights (with scale) -> the down
        matmul's input (1, 1, 32, k), width-sharded in L1."""
        out = ttnn.empty(
            [1, 1, TILE, self.k],
            dtype=self.dtype_out,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.cfg_out,
        )
        src, dst = _shards(self.device, up), _shards(self.device, out)
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        w_addr = W.buffer_address() if self.scale else 0
        for (cx, cy), (e, t0, n) in zip(self.xy, self.work):
            r, o = _runs(src, t0, n, self.page_in, 0), _runs(dst, t0, n, self.page_out, 16)
            rd[cx][cy] = [w_addr, e, len(r)] + [v for run in r for v in run]
            cp[cx][cy] = [n]
            wr[cx][cy] = [len(o)] + [v for run in o for v in run]
        kernels = [
            _kernel(
                "l1_in.cpp",
                self.cores,
                [int(self.scale)] + (_accessors(W) if self.scale else []),
                rd,
                ttnn.ReaderConfigDescriptor(),
            ),
            _kernel("moe_act_compute.cpp", self.cores, [int(self.scale)], cp, self.compute_cfg),
            _kernel("l1_out.cpp", self.cores, [], wr, ttnn.WriterConfigDescriptor()),
        ]
        io = [up, W, out] if self.scale else [up, out]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=self.cbs))
        return out


class TilesAdd:
    """a + b over the first n columns of two fp32 width-sharded L1 tensors (the down matmuls' raw outputs) -> (1, 32, n)
    fp32 interleaved, one tile per core (tt/kernels/l1_in.cpp, tiles_add_compute.cpp, tiles_out.cpp)."""

    def __init__(self, device, n):
        self.device, self.n = device, n
        self.xy, self.cores = _cores(n // TILE, device.compute_with_storage_grid_size())
        self.cbs = [_cb(i, ttnn.float32, 4096, 1, self.cores) for i in (0, 1, 16)]
        self.compute_cfg = _fp32_dst_cfg([0, 1])

    def __call__(self, a, b):
        out = ttnn.empty([1, TILE, self.n], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        sa, sb = _shards(self.device, a), _shards(self.device, b)
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for t, (cx, cy) in enumerate(self.xy):
            r = _runs(sa, t, 1, 4096, 0) + _runs(sb, t, 1, 4096, 1)
            rd[cx][cy] = [0, 0, len(r)] + [v for run in r for v in run]
            cp[cx][cy] = [1]
            wr[cx][cy] = [1, 16, out.buffer_address(), t]
        kernels = [
            _kernel("l1_in.cpp", self.cores, [0], rd, ttnn.ReaderConfigDescriptor()),
            _kernel("tiles_add_compute.cpp", self.cores, [], cp, self.compute_cfg),
            _kernel("tiles_out.cpp", self.cores, _accessors(out), wr, ttnn.WriterConfigDescriptor()),
        ]
        ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=self.cbs))
        return out


class MoEGlue:
    """The kernels the MoE stub's decode runs between its matmuls (_decode_glued)."""

    def __init__(self, device, m):
        hid, ni, sh = int(m._up_dram[0].shape[-2]), m._Eloc * m._inter, int(m._sh_up_dram.shape[-1])
        self.x_cfg, self.sh_x_cfg = _dram_mm.in0_memcfg(device, hid, ni), _dram_mm.in0_memcfg(device, hid, sh)
        self.act = ExpertAct(device, ni, hid, m._inter // TILE, ttnn.bfloat8_b, ttnn.bfloat8_b, True)
        self.sh_act = ExpertAct(device, sh, hid, sh // TILE, ttnn.bfloat16, ttnn.bfloat16, False)
        self.add = TilesAdd(device, hid)


class Step:
    """What one forward needs besides the activations."""

    def __init__(
        self, mode, cur_pos=None, n=0, Tp=0, lens=None, slots=None, valid=None, valid_rows=None, gather=None, keep=None
    ):
        self.mode, self.cur_pos = mode, cur_pos
        self.n, self.Tp, self.lens, self.slots = n, Tp, lens, slots
        self.valid = valid  # prefill: (n, Tp, 1) fp32, 1 at real positions, 0 at padding
        self.valid_rows = valid_rows  # the same as (n Tp, 1)
        self.gather = gather  # prefill: (1, 32*(K-1), n*Tp) fp32 one-hot: prompt b's row len_b - s -> device row slot_b
        self.keep = keep  # prefill: (1, B, 1) fp32, 0 on the prefilled rows, 1 elsewhere
        self._consts = {}


# --------------------------------------------------------------------------------------------------- mamba
class MambaLayer:
    def __init__(self, pipe, hf_block):
        dev = pipe.device
        self.device = dev
        self.norm = _rmsnorm_stub.build(dev, hf_block.norm)
        self.m = _mamba_stub.build(dev, hf_block.mixer)
        m = self.m
        B = pipe.batch
        rep = _mapper(dev)
        z = lambda shape: ttnn.from_torch(  # noqa: E731
            torch.zeros(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep
        )
        if pipe._ssm_step is None:
            pipe._ssm_step = SSMStep(dev, m, B)
        self.step = pipe._ssm_step
        m._state = {"ssm": z(self.step.state_shape(B))}
        for s in range(1, m.conv_k):
            m._state[f"prev{s}"] = z((1, B, m.conv_dim))
        m._cache_mode = "decode"
        m._out_dtype = ttnn.float32
        m._consts = pipe._mamba_consts  # every mixer's shapes are the same: one copy of the per-shape constants
        self._carry = pipe._mamba_carry
        # per-head A and D as rows over this chip's heads (D over its heads' P columns), sharded like in_proj's heads
        mix = hf_block.mixer
        heads = ttnn.ShardTensor2dMesh(dev, mesh_shape=list(dev.shape), dims=(None, 2)) if m._shard else rep
        row = lambda t: ttnn.from_torch(  # noqa: E731
            t.float().reshape(1, 1, -1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=heads,
        )
        self.A_row = row(-torch.exp(mix.A_log.detach().float()))
        self.D_row = row(mix.D.detach().float().repeat_interleave(m.head_dim))
        # the decode kernels' per-column constants, rows replicated (tiles that broadcast as they are)
        full = lambda t, w: ttnn.add(z((1, B, w)), t)  # noqa: E731
        conv = [full(t, m.conv_dim) for t in m._conv_taps]
        conv.append(full(m._conv_bias, m.conv_dim) if m._conv_bias is not None else z((1, B, m.conv_dim)))
        self.conv_consts = ttnn.concat(conv, dim=-1)  # taps by lag, then the bias
        self.dt_bias_full, self.A_full = full(m._dt_bias, m.num_heads), full(self.A_row, m.num_heads)
        self.D_full = full(self.D_row, m.intermediate_size)

    def __call__(self, x, st):
        h = self.norm(x, dtype=ttnn.float32)
        y = self._decode(h) if st.mode == "decode" else self._prefill(h, st)
        return ttnn.add(x, y, dtype=ttnn.float32)

    def _decode(self, hs):
        """hs (1, B, hidden) fp32, one token per row -> the mixer output (1, B, hidden) fp32, each row's state stepped."""
        m = self.m
        B = int(hs.shape[1])
        pc = _dram_mm.mcast1d_config(self.device, int(hs.shape[-1]), int(m._w_in.shape[-1]))
        proj = ttnn.matmul(hs, m._w_in, compute_kernel_config=m.ckc, program_config=pc)  # gate | x B C | dt
        hc, dt, dec = self.step.pre(proj, m, self.conv_consts, self.dt_bias_full, self.A_full)
        parts = self.step(m._state["ssm"], hc, dt, dec)
        y = self.step.post(parts, hc, proj, self.D_full)
        for t in (proj, hc, dt, dec, parts):
            ttnn.deallocate(t)
        y = _ssm_cache.group_rms(self.device, y, m._norm_w_full, m.norm_group_size, m.norm_eps, m._consts, m.ckc)
        return m._finish(y, 1, B)

    # -- prefill ----------------------------------------------------------------------------------------------
    def _chunk_consts(self, n, Tc, Tprev):
        """carry[s] (n, Tc, 32): row t picks the previous chunk's pre-conv row t - s (< 0) out of `prev`; sel
        (n, 32, Tprev): prev's row j = the previous chunk's row Tprev - 1 - j (j < K - 1)."""
        key = (n, Tc, Tprev)
        if key not in self._carry:
            K = self.m.conv_k
            carry = []
            for s in range(K):
                c = torch.zeros(Tc, TILE)
                for t in range(min(s, Tc)):
                    c[t, s - t - 1] = 1.0
                carry.append(self.m._dev(c.reshape(1, Tc, TILE).repeat(n, 1, 1)))
            sel = torch.zeros(TILE, Tprev)
            for j in range(K - 1):
                sel[j, Tprev - 1 - j] = 1.0
            self._carry[key] = (carry, self.m._dev(sel.reshape(1, TILE, Tprev).repeat(n, 1, 1)))
        return self._carry[key]

    def _ssd_chunk(self, hbc, dt, n, Tc, prev, S0):
        """One SSD chunk: hbc (n, Tc, conv_dim) pre-conv, dt (n, Tc, H) (0 at padding), prev (n, 32, conv_dim) the
        previous chunk's last K - 1 pre-conv rows (None: sequence start), S0 (n, H, N, P) the state entering the chunk
        (None: zero) -> (Y (n, H, Tc, P), the state leaving the chunk)."""
        m = self.m
        H, P, N, G, I = m.num_heads, m.head_dim, m.ssm_state_size, m.n_groups, m.intermediate_size
        consts = m._get_consts(n, Tc)
        carry = self._chunk_consts(n, Tc, Tc)[0] if prev is not None else None
        conv = ttnn.multiply(hbc, m._conv_taps[0])
        for s in range(1, m.conv_k):
            shifted = ttnn.matmul(consts["shifts"][s], hbc, compute_kernel_config=m.ckc)
            if carry is not None:
                shifted = ttnn.add(shifted, ttnn.matmul(carry[s], prev, compute_kernel_config=m.ckc))
            conv = ttnn.addcmul(conv, shifted, m._conv_taps[s])
            ttnn.deallocate(shifted)
        if m._conv_bias is not None:
            conv = ttnn.add(conv, m._conv_bias)
        hbc = ttnn.silu(conv)
        ttnn.deallocate(conv)
        Xf = ttnn.slice(hbc, [0, 0, 0], [n, Tc, I])
        Bf = ttnn.slice(hbc, [0, 0, I], [n, Tc, I + G * N])
        Cf = ttnn.slice(hbc, [0, 0, I + G * N], [n, Tc, I + 2 * G * N])
        ttnn.deallocate(hbc)
        X = m._to_heads(Xf, n, Tc, H, P)
        Bh = ttnn.repeat_interleave(m._to_heads(Bf, n, Tc, G, N), H // G, dim=1)  # (n, H, Tc, N)
        Ch = ttnn.repeat_interleave(m._to_heads(Cf, n, Tc, G, N), H // G, dim=1)
        dt_h = m._to_heads(dt, n, Tc, H, 1)  # (n, H, Tc, 1)
        X_disc = ttnn.multiply(X, dt_h)
        D_res = ttnn.multiply(X, m._D)
        a = ttnn.multiply(dt_h, m._A)
        A_cum = _ssm_cache.heads_matmul(consts["tril"], a, m.ckc)  # (n, H, Tc, 1) inclusive cumsum

        # the state leaving the chunk: S0 decayed over the chunk + the chunk's own inputs, decayed to its end
        last = ttnn.slice(A_cum, [0, 0, Tc - 1, 0], [n, H, Tc, 1])
        w = ttnn.exp(ttnn.subtract(last, A_cum))
        S = _ssm_cache.heads_matmul(ttnn.transpose(ttnn.multiply(Bh, w), -2, -1), X_disc, m.ckc)  # (n, H, N, P)
        if S0 is not None:
            S = ttnn.add(S, ttnn.multiply(S0, ttnn.exp(last)))

        # within the chunk: Y = (C B^T * L) X dt + D x, plus what S0 contributes: exp(A_cum_t) C_t S0
        Dmat = ttnn.add(ttnn.subtract(A_cum, ttnn.transpose(A_cum, -2, -1)), consts["neg"])
        L = ttnn.exp(Dmat)
        CB = _ssm_cache.heads_matmul(
            ttnn.typecast(Ch, ttnn.bfloat16), ttnn.typecast(ttnn.transpose(Bh, -2, -1), ttnn.bfloat16), m.ckc
        )
        Y = ttnn.add(_ssm_cache.heads_matmul(ttnn.multiply(CB, L), X_disc, m.ckc), D_res)
        if S0 is not None:
            Y = ttnn.add(Y, ttnn.multiply(_ssm_cache.heads_matmul(Ch, S0, m.ckc), ttnn.exp(A_cum)))
        for t in (X, Bh, Ch, dt_h, X_disc, D_res, a, A_cum, last, w, Dmat, L, CB):
            ttnn.deallocate(t)
        return Y, S

    def _prefill(self, hs, st):
        """hs (n, Tp, hidden) fp32 -> the mixer output (n, Tp, hidden) fp32; prompt i's state into row slots[i]."""
        m, n, Tp = self.m, st.n, st.Tp
        H, P, I, C, K = m.num_heads, m.head_dim, m.intermediate_size, m.conv_dim, m.conv_k
        cg = self.device.compute_with_storage_grid_size()
        proj = _dram_mm.from_rows(
            ttnn.matmul(
                _dram_mm.as_rows(hs),
                m._w_in,
                compute_kernel_config=m._proj_ckc,
                core_grid=ttnn.CoreGrid(y=cg.y, x=cg.x),
            ),
            n,
            Tp,
        )
        ttnn.deallocate(hs)
        gate = ttnn.slice(proj, [0, 0, 0], [n, Tp, I])
        hbc = ttnn.slice(proj, [0, 0, I], [n, Tp, I + C])
        dt = ttnn.slice(proj, [0, 0, I + C], [n, Tp, I + C + H])
        ttnn.deallocate(proj)
        dt = ttnn.multiply(m._softplus(ttnn.add(dt, m._dt_bias)), st.valid)

        # each prompt's conv window at its own end, into its row of prev1..3 (other rows kept)
        win = ttnn.matmul(st.gather, ttnn.reshape(hbc, [1, n * Tp, C]), compute_kernel_config=m.ckc)  # (1, 32(K-1), C)
        B = int(st.keep.shape[1])
        for s in range(1, K):
            row = ttnn.slice(win, [0, (s - 1) * B, 0], [1, s * B, C])
            new = ttnn.add(ttnn.multiply(m._state[f"prev{s}"], st.keep), row)
            ttnn.copy(new, m._state[f"prev{s}"])
            ttnn.deallocate(new)
            ttnn.deallocate(row)
        ttnn.deallocate(win)

        ys, S, prev = [], None, None
        for c0 in range(0, Tp, SSD_CHUNK):
            Tc = min(SSD_CHUNK, Tp - c0)
            hbc_c = ttnn.slice(hbc, [0, c0, 0], [n, c0 + Tc, C]) if Tc != Tp else hbc
            dt_c = ttnn.slice(dt, [0, c0, 0], [n, c0 + Tc, H]) if Tc != Tp else dt
            Y, S_new = self._ssd_chunk(hbc_c, dt_c, n, Tc, prev, S)
            if S is not None:
                ttnn.deallocate(S)
            S = S_new
            ys.append(Y)
            if c0 + Tc < Tp:
                sel = self._chunk_consts(n, SSD_CHUNK, Tc)[1]
                if prev is not None:
                    ttnn.deallocate(prev)
                prev = ttnn.matmul(sel, hbc_c, compute_kernel_config=m.ckc)  # (n, 32, C)
        ttnn.deallocate(hbc)
        self.step.rows(S, m._state["ssm"], st.slots)
        ttnn.deallocate(S)
        Y = ys[0] if len(ys) == 1 else ttnn.concat(ys, dim=2)

        y = _ssm_cache.from_heads(Y, n, Tp, H, P)
        ttnn.deallocate(Y)
        y = ttnn.multiply(y, ttnn.silu(gate))
        ttnn.deallocate(gate)
        gs = m.norm_group_size
        ng = I // gs
        y = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [n, Tp * ng, gs])
        y = ttnn.rms_norm(
            ttnn.to_layout(y, ttnn.TILE_LAYOUT), weight=m._norm_ones, epsilon=m.norm_eps, compute_kernel_config=m.ckc
        )
        y = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [n, Tp, I])
        y = ttnn.multiply(ttnn.to_layout(y, ttnn.TILE_LAYOUT), m._norm_w_full)
        return m._finish(y, n, Tp)


# ----------------------------------------------------------------------------------------------- attention
class AttentionLayer:
    """NemotronH attention (no RoPE, GQA 32 q / 2 kv heads): head-parallel over the TP axis, each chip its q heads and
    the kv head they read; o_proj row-parallel + all_reduce. Prefill: ttnn SDPA (causal); decode: flash-decode over
    each row's own cache up to its position."""

    def __init__(self, pipe, hf_block):
        dev = self.device = pipe.device
        self.norm = _rmsnorm_stub.build(dev, hf_block.norm)
        a = hf_block.mixer
        D = self.head_dim = int(a.head_dim)
        Hq = int(a.q_proj.out_features) // D
        Hkv = int(a.k_proj.out_features) // D
        tp = self.tp = pipe.tp
        self.hq = Hq // tp
        self.scale = float(getattr(a, "scaling", D**-0.5))
        Wq = a.q_proj.weight.detach().float().t()  # (hidden, Hq*D)
        Wk = a.k_proj.weight.detach().float().t()
        Wv = a.v_proj.weight.detach().float().t()
        Wo = a.o_proj.weight.detach().float().t()  # (Hq*D, hidden)
        qkv, wo = [], []
        for d in range(tp):
            qh = range(d * self.hq, (d + 1) * self.hq)
            kvh = (d * self.hq) // (Hq // Hkv)
            qcols = [c for h in qh for c in range(h * D, (h + 1) * D)]
            kvcols = list(range(kvh * D, (kvh + 1) * D))
            qkv.append(torch.cat([Wq[:, qcols], Wk[:, kvcols], Wv[:, kvcols]], dim=1))
            wo.append(Wo[qcols, :])
        mesh = list(dev.shape)
        shard = lambda t, dim: ttnn.from_torch(  # noqa: E731
            t.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=mesh, dims=(None, dim)),
        )
        self.w_qkv = shard(torch.cat(qkv, dim=1), 1)  # (hidden, (hq + 2) D) per chip
        self.w_o = shard(torch.cat(wo, dim=0), 0)  # (hq D, hidden) per chip
        B, cap = pipe.batch, pipe.capacity
        zeros = torch.zeros(B, 1, cap, D)
        self.k_cache = ttnn.from_torch(
            zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=_mapper(dev)
        )
        self.v_cache = ttnn.from_torch(
            zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=_mapper(dev)
        )
        self.ckc = _ckc()
        grid = dev.compute_with_storage_grid_size()
        self.sdpa_grid = ttnn.CoreCoord(min(8, grid.x), min(8, grid.y))
        self.heads_mem = ttnn.create_sharded_memory_config(
            shape=(TILE, D),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # nlp_concat_heads_decode wants one core per user in a single rectangle from (0, 0)
        self.out_mem = ttnn.create_sharded_memory_config(
            shape=(-(-self.hq // TILE) * TILE, D),
            core_grid=ttnn.CoreGrid(y=B // 8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_grid, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )

    def __call__(self, x, st):
        h = self.norm(x, dtype=ttnn.bfloat16)
        y = self._decode(h, st) if st.mode == "decode" else self._prefill(h, st)
        return ttnn.add(x, y, dtype=ttnn.float32)

    def _o_proj(self, attn_rows):
        out = ttnn.matmul(attn_rows, self.w_o, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out

    def _decode(self, h, st):
        B, hid = int(h.shape[1]), int(h.shape[2])
        qkv = ttnn.matmul(
            ttnn.reshape(h, [1, 1, B, hid]),
            self.w_qkv,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=self.hq, num_kv_heads=1, memory_config=self.heads_mem
        )
        ttnn.deallocate(qkv)
        ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=st.cur_pos)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=st.cur_pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=st.cur_pos,
            scale=self.scale,
            program_config=self.decode_pc,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        out = ttnn.to_memory_config(out, self.out_mem)
        cat = ttnn.experimental.nlp_concat_heads_decode(out, num_heads=self.hq)  # (1, 1, B, hq D)
        ttnn.deallocate(out)
        cat = ttnn.to_memory_config(cat, ttnn.L1_MEMORY_CONFIG)
        return ttnn.reshape(self._o_proj(cat), [1, B, hid])

    def _prefill(self, h, st):
        n, Tp, hid, D = st.n, st.Tp, int(h.shape[2]), self.head_dim
        qkv = ttnn.matmul(_dram_mm.as_rows(h), self.w_qkv, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        qkv = ttnn.reshape(qkv, [n, 1, Tp, int(qkv.shape[-1])])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.hq, num_kv_heads=1, transpose_k_heads=False
        )
        ttnn.deallocate(qkv)
        kv_rows(self.device, k, v, self.k_cache, self.v_cache, st.slots)
        chunk = next(c for c in (256, 128, 64, 32) if Tp % c == 0)
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False
        )
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale, program_config=pc, compute_kernel_config=self.ckc
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        cat = _ssm_cache.from_heads(out, n, Tp, self.hq, D)  # (n, Tp, hq D)
        ttnn.deallocate(out)
        y = self._o_proj(_dram_mm.as_rows(cat))
        return _dram_mm.from_rows(y, n, Tp)


# ------------------------------------------------------------------------------------------------------ MoE
# the expert matmuls at decode's precision (fp32 accumulation, bf16 expert outputs): a prompt's MoE output then does
# not depend on which other prompts share its prefill
_EXPERT_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _expert_slots(device, W):
    """Each local expert's tokens, picked on the host from W (T, Eloc), this chip's routing weights: C slots per
    expert (the busiest expert's token count up to a power of two: few programs), its routed tokens first, the rest
    weight 0 so they add nothing. -> C, the slots' token ids (uint32 row-major and fp32 tile, (1, Eloc C)) and
    weights (Eloc, C, 1). ttnn.topk over the tokens runs out of L1 once C reaches a few hundred."""
    per_chip = [ttnn.to_torch(t).reshape(-1, int(W.shape[-1])) for t in ttnn.get_device_tensors(W)]
    T = per_chip[0].shape[0]
    most = max(int((w > 0).sum(0).max()) for w in per_chip)
    C = min(T, max(TILE, 1 << (max(1, most) - 1).bit_length()))
    vals, idx = zip(*(torch.topk(w.T.float(), C, dim=-1) for w in per_chip))
    idx = torch.cat(idx).reshape(len(per_chip), -1)
    vals = torch.cat(vals).reshape(-1, C, 1)
    to_chips = ttnn.ShardTensorToMesh(device, dim=0)
    idx_rm = ttnn.from_torch(
        idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=to_chips
    )
    idx_f = ttnn.from_torch(
        idx.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=to_chips
    )
    w3 = ttnn.from_torch(vals, dtype=W.dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=to_chips)
    return C, idx_rm, idx_f, w3


def _routed_mix(device, hs, W, up_b, down_cat, ckc, arange_cache):
    """_stubs/_sparse_moe.routed_mix at decode's precision. hs (T, H); W (T, Eloc) this chip's routing weights;
    up_b (Eloc, H, I); down_cat (Eloc I, H) -> this chip's partial mixture (T, H) fp32."""
    bmm = _sparse_moe.bmm_config
    T, H = int(hs.shape[-2]), int(hs.shape[-1])
    Eloc, I = int(W.shape[-1]), int(up_b.shape[-1])
    C, idx_rm, idx_f, w3 = _expert_slots(device, W)
    table = ttnn.to_layout(ttnn.typecast(hs, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    xe = ttnn.reshape(ttnn.embedding(idx_rm, table, layout=ttnn.TILE_LAYOUT), [Eloc, C, H])
    ttnn.deallocate(table)
    ttnn.deallocate(idx_rm)
    act = ttnn.matmul(
        xe,
        up_b,
        compute_kernel_config=_EXPERT_CKC,
        dtype=ttnn.bfloat8_b,
        program_config=bmm(
            device, C, H, I, xe.dtype, up_b.dtype, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
    )
    ttnn.deallocate(xe)
    act = ttnn.multiply(act, w3, dtype=ttnn.bfloat8_b, input_tensor_a_activations=[ttnn.UnaryOpType.SQUARE])
    ttnn.deallocate(w3)
    ye = ttnn.matmul(
        act,
        ttnn.reshape(down_cat, [Eloc, I, H]),
        compute_kernel_config=_EXPERT_CKC,
        dtype=ttnn.bfloat16,
        program_config=bmm(device, C, I, H, act.dtype, down_cat.dtype),
    )
    ttnn.deallocate(act)
    ar = arange_cache.get(T)
    if ar is None:
        ar = arange_cache[T] = ttnn.from_torch(
            torch.arange(T, dtype=torch.float32).reshape(T, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=_mapper(device),
        )
    onehot = ttnn.eq(ar, idx_f, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(idx_f)
    out = ttnn.matmul(
        onehot,
        ttnn.reshape(ye, [Eloc * C, H]),
        compute_kernel_config=ckc,
        dtype=ttnn.float32,
        program_config=bmm(device, T, Eloc * C, H, onehot.dtype, ye.dtype, ttnn.float32, 800 * 1024),
    )
    ttnn.deallocate(onehot)
    ttnn.deallocate(ye)
    return out


class MoELayer:
    def __init__(self, pipe, hf_block):
        self.norm = _rmsnorm_stub.build(pipe.device, hf_block.norm)
        self.moe = _moe_stub.build(pipe.device, hf_block.mixer)
        for name in ("_up_cat",):  # the stub's dense-prefill bank: prefill runs its sparse path
            t = getattr(self.moe, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self.moe, name, None)
        self.router = Router(pipe.device, self.moe) if self.moe._Eloc == TILE else None
        dram = self.router is not None and getattr(self.moe, "_sh_up_dram", None) is not None
        self.glue = MoEGlue(pipe.device, self.moe) if dram else None

    def __call__(self, x, st):
        h = self.norm(x, dtype=ttnn.float32)
        if st.mode == "decode" or st.n * st.Tp <= TILE:  # one tile row: the stub's dense bank, every expert
            one_row = self.router is not None and int(h.shape[0]) * int(h.shape[1]) == TILE
            y = self.moe(h, W_use=self.router(h), glue=self.glue) if one_row else self.moe(h)
        else:
            n, Tp, hid = st.n, st.Tp, int(h.shape[2])
            y = ttnn.reshape(self._prefill(ttnn.reshape(h, [n * Tp, hid]), st), [n, Tp, hid])
        return ttnn.add(x, y, dtype=ttnn.float32)

    def _route(self, hs):
        """The stub's router (fp32 all-pairs rank top-k) -> this chip's routing weights (T, Eloc)."""
        m = self.moe
        T, E = int(hs.shape[0]), m.n_routed_experts
        scores = ttnn.sigmoid(ttnn.matmul(hs, m._gate_w, compute_kernel_config=m.ckc))
        choice = ttnn.reshape(ttnn.add(scores, m._bias), [1, T, 1, E])
        gt = ttnn.gt(choice, ttnn.transpose(choice, 2, 3), dtype=ttnn.bfloat8_b)
        rank = ttnn.reshape(ttnn.sum(gt, dim=3), [T, E])
        ttnn.deallocate(gt)
        ttnn.deallocate(choice)
        W = ttnn.multiply(scores, ttnn.typecast(ttnn.lt(rank, float(m.top_k)), ttnn.float32))
        ttnn.deallocate(rank)
        ttnn.deallocate(scores)
        if m.norm_topk_prob:
            W = ttnn.multiply(W, ttnn.reciprocal(ttnn.add(ttnn.sum(W, dim=-1, keepdim=True), 1e-20)))
        W = ttnn.multiply(W, m.routed_scaling_factor)
        return ttnn.matmul(W, m._sel, compute_kernel_config=m.ckc) if m._shard else W

    def _prefill(self, hs, st):
        """hs (n Tp, hidden) fp32. The stub's sparse expert mix with padding routed nowhere and every expert taking all
        the tokens routed to it: its capacity is read from the routing, so no token is ever dropped."""
        m = self.moe
        W = ttnn.multiply(self._route(hs), st.valid_rows)  # (T, Eloc), 0 at padding
        out = _routed_mix(m.device, hs, W, m._up_b, m._down_cat, m.ckc, m._arange)
        ttnn.deallocate(W)
        hs_bf = ttnn.typecast(hs, ttnn.bfloat16)
        s_act = ttnn.square(ttnn.linear(hs_bf, m._sh_up, compute_kernel_config=m._expert_ckc, activation="relu"))
        ttnn.deallocate(hs_bf)
        s_down = ttnn.matmul(s_act, m._sh_down, compute_kernel_config=m._expert_ckc, dtype=ttnn.float32)
        ttnn.deallocate(s_act)
        out = ttnn.add(out, s_down)
        ttnn.deallocate(s_down)
        if m._shard:
            out = ttnn.all_reduce(out, cluster_axis=m._tp_axis, topology=ttnn.Topology.Linear)
        return out


# ------------------------------------------------------------------------------------------------- pipeline
class Settings:
    def __init__(self, top_k, top_p, temperature, stop_ids, pad_id):
        self.top_k, self.top_p, self.temperature, self.stop_ids, self.pad_id = (
            top_k,
            top_p,
            temperature,
            stop_ids,
            pad_id,
        )


def _settings():
    from transformers import GenerationConfig

    g = GenerationConfig.from_pretrained(_hf_ref.HF_MODEL_ID)
    eos = g.eos_token_id if isinstance(g.eos_token_id, (list, tuple)) else [g.eos_token_id]
    return Settings(
        top_k=SAMPLER_TOP_K,
        top_p=float(g.top_p if g.top_p is not None else 1.0),
        temperature=float(g.temperature if g.temperature is not None else 1.0),
        stop_ids=tuple(int(e) for e in eos),
        pad_id=int(g.pad_token_id if g.pad_token_id is not None else 0),
    )


class NemotronServe:
    def __init__(self, device, model, batch=BATCH, capacity=KV_CAPACITY, log=print):
        self.device = device
        self.config = model.config
        self.batch = int(batch)
        self.capacity = int(capacity)
        self.tp = int(list(device.shape)[-1])
        self.settings = _settings()
        cfg = self.config
        self.hidden = int(cfg.hidden_size)
        t0 = time.time()
        self._mamba_consts, self._mamba_carry = {}, {}
        self._ssm_step = None
        self.layers = []
        for i, blk in enumerate(model.model.layers):
            bt = str(blk.block_type).lower()
            if bt in ("mamba", "mamba2", "linear_attention"):
                layer = MambaLayer(self, blk)
            elif bt in ("attention", "full_attention", "self_attention"):
                layer = AttentionLayer(self, blk)
            elif bt == "moe":
                layer = MoELayer(self, blk)
            else:
                raise ValueError(f"block type {bt!r}")
            self.layers.append(layer)
            log(f"[build] layer {i} {type(layer).__name__} {time.time() - t0:.0f}s", flush=True)
        emb = model.model.embeddings.weight.detach().to(torch.bfloat16).contiguous().clone()
        self.embed_w = ttnn.from_torch(
            emb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=_mapper(device)
        )
        self.final_norm = _rmsnorm_stub.build(device, model.model.norm_f)
        lm = model.lm_head.weight.detach().float().t().contiguous()  # (hidden, vocab)
        self.vocab = int(lm.shape[1])
        self.lm_w = ttnn.from_torch(
            lm,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=list(device.shape), dims=(None, 1)),
        )
        self.head_ckc = _ckc(ttnn.MathFidelity.HiFi2)
        s = self.settings
        self.sampler = Sampler(
            device, self.batch, s.top_k, s.top_p, s.temperature, vocab=self.vocab, tp=self.tp, tp_axis=1
        )
        B = self.batch
        rm = ttnn.ROW_MAJOR_LAYOUT
        self._write("tok_u32", torch.zeros((1, B), dtype=torch.int32), ttnn.uint32, rm)
        self._write("cur_pos", torch.zeros((B,), dtype=torch.int32), ttnn.int32, rm)
        self._write("pos_col", torch.zeros((1, 1, B, 1)), ttnn.float32)
        self._write("u_table", torch.rand((1, 1, B, self.capacity)), ttnn.float32)
        self.prefill_len = None
        log(
            f"[build] {len(self.layers)} layers, {B} rows, KV capacity {self.capacity}, TP {self.tp} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ---------------------------------------------------------------------------------------------- helpers
    def _up(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=self.device, mesh_mapper=_mapper(self.device)
        )

    def _write(self, name, t, dtype, layout=ttnn.TILE_LAYOUT):
        """Host data into the resident buffer `name`, IN PLACE when it exists with this shape (a captured trace keeps
        reading the same addresses), else allocate it."""
        host = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=_mapper(self.device))
        cur = getattr(self, name, None)
        if cur is not None and list(cur.shape) == list(host.shape) and cur.dtype == host.dtype:
            ttnn.copy_host_to_device_tensor(host, cur)
        else:
            setattr(self, name, ttnn.to_device(host, self.device))

    def _embed(self, ids, shape):
        h = ttnn.to_layout(ttnn.embedding(ids, self.embed_w), ttnn.TILE_LAYOUT)
        return ttnn.typecast(ttnn.reshape(h, shape), ttnn.float32)

    def _stack(self, h, st):
        for layer in self.layers:
            h = layer(h, st)
        return h

    def _head_local(self, x):
        """Final-norm state (1, B, hidden) -> this chip's vocab slice of the fp32 logits [1, 1, B, V / tp]."""
        B = int(x.shape[-2])
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, B, self.hidden]), ttnn.bfloat16)
        pc = _dram_mm.mcast1d_config(self.device, self.hidden, int(self.lm_w.shape[-1]))
        return ttnn.matmul(x, self.lm_w, compute_kernel_config=self.head_ckc, program_config=pc, dtype=ttnn.float32)

    def _gather(self, local):
        if self.tp == 1:
            return local
        return ttnn.all_gather(local, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)

    def _commit(self, tok):
        t = ttnn.typecast(tok, ttnn.uint32)
        t = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [1, self.batch])
        ttnn.copy(t, self.tok_u32)

    def _advance(self):
        ttnn.plus_one(self.cur_pos)
        ttnn.copy(ttnn.add(self.pos_col, 1.0), self.pos_col)

    # ---------------------------------------------------------------------------------------------- prefill
    def prefill_rows(self, prompts, slots):
        """Prompts run as one batch -> prompt i's state in device row slots[i] only; every other row is left as it is.
        Returns the gathered logits [1, 1, B, V], row i at prompt i's last position (rows past the prompts: copies of
        row 0). The caller owns the decode counters (set_decode_rows)."""
        B, H, n = self.batch, self.hidden, len(prompts)
        slots = [int(s) for s in slots]
        if not 1 <= n <= B or len(slots) != n or len(set(slots)) != n or not all(0 <= s < B for s in slots):
            raise ValueError(f"{n} prompts for slots {slots} (distinct rows in 0..{B - 1})")
        lens = [len(p) for p in prompts]
        if min(lens) < 1 or max(lens) >= self.capacity:
            raise ValueError(f"prompt lengths {min(lens)}..{max(lens)}: need 1..{self.capacity - 1}")
        Tp = max(TILE, -(-max(lens) // TILE) * TILE)
        ids = torch.full((n, Tp), self.settings.pad_id, dtype=torch.int32)
        valid = torch.zeros((n, Tp, 1))
        for i, p in enumerate(prompts):
            ids[i, : lens[i]] = torch.as_tensor(list(p), dtype=torch.int32)
            valid[i, : lens[i]] = 1.0
        K = self.layers[0].m.conv_k if isinstance(self.layers[0], MambaLayer) else 4
        gather = torch.zeros((1, B * (K - 1), n * Tp))
        keep = torch.ones((1, B, 1))
        for i, (T, slot) in enumerate(zip(lens, slots)):
            keep[0, slot, 0] = 0.0
            for s in range(1, K):
                if T - s >= 0:
                    gather[0, (s - 1) * B + slot, i * Tp + T - s] = 1.0
        self.prefill_len = Tp
        st = Step(
            "prefill",
            n=n,
            Tp=Tp,
            lens=lens,
            slots=slots,
            valid=self._up(valid, ttnn.float32),
            valid_rows=self._up(valid.reshape(n * Tp, 1), ttnn.float32),
            gather=self._up(gather, ttnn.float32),
            keep=self._up(keep, ttnn.float32),
        )
        h = self._embed(self._up(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), [n, Tp, H])
        hn = self.final_norm(self._stack(h, st), dtype=ttnn.float32)
        for t in (st.valid, st.valid_rows, st.gather, st.keep):
            ttnn.deallocate(t)
        # each prompt's last position: the 32-row block holding it, read back, the row picked on the host
        blocks = []
        for i, T in enumerate(lens):
            tr = (T - 1) // TILE * TILE
            blocks.append(ttnn.slice(hn, [i, tr, 0], [i + 1, tr + TILE, H]))
        cat = blocks[0] if n == 1 else ttnn.concat(blocks, dim=1)
        host = read_host(cat).reshape(n, TILE, H)
        for t in {id(t): t for t in blocks + [cat, hn]}.values():  # a full-range slice is its input
            ttnn.deallocate(t)
        last = host[torch.arange(n), torch.as_tensor([(T - 1) % TILE for T in lens])]
        last = torch.cat([last, last[:1].expand(B - n, H)]).reshape(1, B, H)
        x = self._up(last, ttnn.float32)
        local = self._head_local(x)
        ttnn.deallocate(x)
        logits = self._gather(local)
        if logits is not local:
            ttnn.deallocate(local)
        return logits

    # ----------------------------------------------------------------------------------------------- decode
    def serve_rows(self):
        """Serving: each row samples by its own request's rule (set_sampling); call before decode is compiled."""
        s, B = self.settings, self.batch
        self.set_sampling([s.top_k] * B, [s.top_p] * B, [1.0 / s.temperature] * B)
        self.sampler.serve(self.samp_prm, self.samp_inv_t)

    def set_sampling(self, top_k, top_p, inv_temp):
        B = self.batch
        prm = torch.zeros((1, 1, B, 32), dtype=torch.float32)
        prm[0, 0, :, 0] = torch.as_tensor(top_k, dtype=torch.float32)
        prm[0, 0, :, 1] = torch.as_tensor(top_p, dtype=torch.float32)
        self._write("samp_prm", prm, ttnn.float32)
        self._write("samp_inv_t", torch.as_tensor(inv_temp, dtype=torch.float32).reshape(1, 1, B, 1), ttnn.float32)

    def set_uniforms(self, table):
        self._write("u_table", torch.as_tensor(table, dtype=torch.float32).reshape(1, 1, self.batch, -1), ttnn.float32)

    def set_decode_rows(self, tokens, positions=None, active=None):
        """Each row's next input token and (when given) its position ([B] host values), IN PLACE. Without positions the
        device's own counters stand; without tokens, the ids the last sampled step left on device. `active` is
        accepted for the adapter's contract: decode runs every expert for every row, so idle rows cost nothing extra."""
        B, rm = self.batch, ttnn.ROW_MAJOR_LAYOUT
        if tokens is not None:
            self._write("tok_u32", torch.as_tensor(tokens, dtype=torch.int32).reshape(1, B), ttnn.uint32, rm)
        if positions is None:
            return
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(B)
        self._write("cur_pos", pos, ttnn.int32, rm)
        self._write("pos_col", pos.float().reshape(1, 1, B, 1), ttnn.float32)

    def _decode_hidden(self):
        st = Step("decode", cur_pos=self.cur_pos)
        h = self._embed(self.tok_u32, [1, self.batch, self.hidden])
        return self.final_norm(self._stack(h, st), dtype=ttnn.float32)

    def decode_tokens(self):
        """One token per row, sampled on device from this chip's vocab slice: ids [1, 1, B, 1] fp32."""
        local = self._head_local(self._decode_hidden())
        tok = self.sampler(local if self.sampler.sharded else self._gather(local), self.u_table, self.pos_col)
        self._commit(tok)
        self._advance()
        return tok

    def decode_step(self, state=None) -> dict:
        """One step returning the gathered logits [1, 1, B, V] (the host samples; it writes the next tokens)."""
        logits = self._gather(self._head_local(self._decode_hidden()))
        self._advance()
        return {"logits": logits}

    def decode_prefill(self, input_ids, seeds=None) -> dict:
        """The same prompt into every row (warmup / tests): rows' counters set to its end."""
        prompt = [int(t) for t in torch.as_tensor(input_ids).reshape(-1, torch.as_tensor(input_ids).shape[-1])[0]]
        logits = self.prefill_rows([prompt] * self.batch, list(range(self.batch)))
        tok = read_host(logits).reshape(self.batch, -1)[:, : self.vocab].argmax(-1)
        self.set_decode_rows(tok, [len(prompt)] * self.batch)
        return {"logits": logits, "tokens": tok}


def build_pipeline(device, model=None, layers=None, **kwargs):
    os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")
    if model is None:
        dtype, why = _hf_ref.choose_reference_dtype(layers)
        print(f"[ref] reference build precision -> {why}", flush=True)
        model = _hf_ref.load_reference(layers, dtype=dtype)
    return NemotronServe(
        device,
        model,
        batch=int(kwargs.get("batch", BATCH)),
        capacity=int(kwargs.get("capacity", KV_CAPACITY)),
        log=kwargs.get("log", print),
    )
