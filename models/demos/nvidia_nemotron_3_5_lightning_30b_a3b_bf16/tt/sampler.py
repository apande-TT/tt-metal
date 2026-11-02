# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device sampling for the serving pipeline (tt/serve.py): each row's temperature / top_k / top_p, among the row's
top `top_k` candidates, from one uniform per position. The same sampler as models/demos/kolibri_1 (its kernels are in
tt/kernels: topk_merge.cpp, sample_tail.cpp, sample_tail_rows.cpp)."""
from __future__ import annotations

from pathlib import Path

import torch

import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"


def _first_cores(n, grid):
    """The first n cores of the grid in row-major order, as (x, y) and as rectangles: a set of single-core ranges
    costs ~0.4 us per core to dispatch, a rectangle about the same as one core."""
    xy = [(i % grid.x, i // grid.x) for i in range(n)]
    rows, rem = divmod(n, grid.x)
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1))] if rows else []
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rem - 1, rows)))
    return xy, ttnn.CoreRangeSet(ranges)


def _cb(index, fmt, page, pages, cores):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _topk_tree(n_tiles: int, k_tiles: int, cores: int, cap: int = 40):
    """Group counts, stage by stage, of a top-k tree over a row n_tiles tiles wide (None: no tree).

    ttnn's top-k for k > 64 sorts each tile row on one core, so one [B, V] row of logits is ~17 us per tile on a
    single core. Each stage splits the row into g groups of w tiles, one tile row per group (g <= cores, so every
    group gets its own core); g * k_tiles tiles survive into the next stage. The first stage takes the narrowest
    groups that fit the grid, at most `cap` tiles wide when the row allows it. A row of at most 20 tiles is left to
    one final top-k."""
    plan = []
    while n_tiles > 20:
        fits = [w for w in range(2 * k_tiles, n_tiles) if n_tiles % w == 0 and n_tiles // w <= cores]
        if not fits:
            return None
        w = next((w for w in fits if w >= min(cap, fits[-1])), fits[0]) if not plan else fits[0]
        plan.append(n_tiles // w)
        n_tiles, cap = (n_tiles // w) * k_tiles, 10
    return plan or None


class Sampler:
    """generation_config's rule on device, fp32 throughout: top_k on logits/T, top_p over those k (exclusive
    cumulative probability < p keeps a token), then inverse CDF in TOKEN-ID order over the kept set against the
    step's uniform.

    The top-k runs as a tree (_topk_tree). A [B, N] tile tensor and a [B*g, N/g] one have the same DRAM pages in the
    same order (page = tile row * row tiles + tile col, and B is whole tile rows), so each regrouping is a metadata
    view -- no data moves -- and every stage carries the token ids along as top-k labels. The inverse CDF then needs
    only the <= k kept tokens: sorted by token id and cumulated in that order, which is the full-vocabulary cumsum
    with its zero terms dropped (the same non-zero terms, added in the same order)."""

    def __init__(
        self, device, batch: int, top_k: int, top_p: float, temperature: float, vocab: int = 0, tp: int = 1, tp_axis=1
    ) -> None:
        self.device = device
        self.top_k, self.top_p, self.temperature = int(top_k), float(top_p), float(temperature)
        self.row_prm = self.row_inv_temp = None  # serve(): each row's own rule
        self.k_pad = -(-self.top_k // 32) * 32
        mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        # The merge and tail kernels (tt/kernels/topk_merge.cpp, sample_tail.cpp): one core per user row.
        g = device.compute_with_storage_grid_size()
        self.count_xy, self.count_cores = _first_cores(batch, g)
        self.top_p_q32 = min(round(self.top_p * 2**32), 2**32 - 1)
        cores = g.x * g.y
        # Vocab-sharded logits (tp chips, each its contiguous vocab / tp slice): each chip runs the tree over its own
        # slice and the chips' top-k lists are gathered and merged. The top k of the union of the slices' top k is
        # the row's top k.
        self.tp, self.tp_axis = (tp, tp_axis) if tp > 1 and vocab % (tp * 32) == 0 else (1, tp_axis)
        local = vocab // self.tp
        tree_ok = vocab > 0 and batch % 32 == 0 and local % 32 == 0
        cap = 40 if self.tp == 1 else 16
        self.plan = _topk_tree(local // 32, self.k_pad // 32, cores, cap) if tree_ok else None
        if self.plan is None or self.k_pad != self.top_k:
            self.tp = 1  # the sharded path is the single-merge tree; anything else samples the gathered logits
            self.plan = _topk_tree(vocab // 32, self.k_pad // 32, cores) if tree_ok else None
        self.token_index = None
        if self.plan:
            vocab_index = torch.arange(vocab, dtype=torch.int32).expand(1, 1, batch, -1).contiguous()
            if self.tp > 1:  # each chip's slice of the global ids
                mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=list(device.shape), dims=(None, -1))
            self.token_index = ttnn.from_torch(
                vocab_index, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
            )

    @property
    def sharded(self) -> bool:
        """True: __call__ takes this chip's vocab slice of the logits, not the gathered row."""
        return self.tp > 1

    def serve(self, prm, inv_temp):
        """Serving: each row samples by its own request's rule, read from resident buffers the caller rewrites in
        place -- prm [1, 1, B, 32] fp32 (column 0: top_k, 1..self.top_k, 1 = greedy; column 1: top_p, >= 1 = no cut)
        and inv_temp [1, 1, B, 1] fp32 (1 / temperature; 1.0 on greedy rows). The candidates stay the row's top
        self.top_k, so a larger or unset top_k samples among those."""
        self.row_prm, self.row_inv_temp = prm, inv_temp

    def _merge(self, x, labels, groups):
        """x, labels [1, 1, B, groups * k] (each group's top k, largest first) -> the row's top k as a groups-way
        merge, one core per user row (tt/kernels/topk_merge.cpp)."""
        B, k = x.shape[-2], self.top_k
        vals = ttnn.empty([1, 1, B, k], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        ids = ttnn.empty([1, 1, B, k], dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(self.count_xy):
            rt[cx][cy] = [t.buffer_address() for t in (x, labels, vals, ids)] + [i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "topk_merge.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.count_cores,
            compile_time_args=[groups, k]
            + [a for t in (x, labels, vals, ids) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        leaves = 1 << (groups - 1).bit_length()  # the kernel's winner tree: P keys + 2P nodes
        scratch = 64 + 2 * groups * k * 4 + 2 * k * 4 + (-(-groups * 4 // 64) * 64) + 3 * leaves * 4
        cb = _cb(0, ttnn.uint32, -(-scratch // 32) * 32, 1, self.count_cores)
        ttnn.generic_op([x, labels, vals, ids], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return vals, ids

    def _top_k(self, x):
        """x [1, 1, B, V] -> (values [1, 1, B, k] largest first, their token ids [1, 1, B, k] uint32)."""
        if not self.plan:
            return ttnn.topk(x, k=self.top_k, dim=-1, largest=True, sorted=True)
        B, n = x.shape[-2], x.shape[-1]
        labels = self.token_index
        if self.k_pad == self.top_k:  # first stage on ttnn, then one merge of its sorted group lists
            g = self.plan[0]
            grouped = [1, 1, B * g, n // g]
            # Group lists to L1: the merge reads them back one 64 B face row at a time.
            x, labels = ttnn.topk(
                ttnn.experimental.view(x, grouped),
                k=self.k_pad,
                dim=-1,
                largest=True,
                sorted=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                indices_tensor=ttnn.experimental.view(labels, grouped),
            )
            flat = [1, 1, B, g * self.k_pad]
            vals, ids = self._merge(ttnn.experimental.view(x, flat), ttnn.experimental.view(labels, flat), g)
            if self.tp == 1:
                return vals, ids
            # Each chip's top k of its slice, in chip order (= ascending token ids, the merge's tie order).
            vals = ttnn.all_gather(vals, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
            ids = ttnn.all_gather(ids, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
            return self._merge(vals, ids, self.tp)
        for g in self.plan:
            grouped = [1, 1, B * g, n // g]
            x, labels = ttnn.topk(
                ttnn.experimental.view(x, grouped),
                k=self.k_pad,
                dim=-1,
                largest=True,
                sorted=True,
                indices_tensor=ttnn.experimental.view(labels, grouped),
            )
            n = g * self.k_pad
            x, labels = ttnn.experimental.view(x, [1, 1, B, n]), ttnn.experimental.view(labels, [1, 1, B, n])
        return ttnn.topk(x, k=self.top_k, dim=-1, largest=True, sorted=True, indices_tensor=labels)

    def _tail(self, vals, e, ids, u_table, col):
        """vals / e = exp(vals - vals[:, :1]) / token ids [1, 1, B, k] (descending), u_table [1, 1, B, C], col
        [1, 1, B, 1] (each user's position, fp32) -> token ids [1, 1, B, 1] fp32 (exact integers): top_p cut, the kept
        candidates in token-id order and the inverse CDF at u_table[b, col[b]], one core per user row with exact
        integer sums (tt/kernels/sample_tail.cpp)."""
        B, k = vals.shape[-2], vals.shape[-1]
        out = ttnn.empty([1, 1, B, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        rt = ttnn.RuntimeArgs()
        rows = self.row_prm is not None
        args = (e, vals, ids, u_table, col) + ((self.row_prm,) if rows else ()) + (out,)
        for i, (cx, cy) in enumerate(self.count_xy):
            rt[cx][cy] = [t.buffer_address() for t in args] + [i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / ("sample_tail_rows.cpp" if rows else "sample_tail.cpp")),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.count_cores,
            compile_time_args=[k // ttnn.TILE_SIZE, u_table.shape[-1] // ttnn.TILE_SIZE, self.top_p_q32]
            + [a for t in args for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = 64 + 3 * k * 4 + 3 * 64 + k * 4 + 2 * k * 8 + (64 if rows else 0)
        cb = _cb(0, ttnn.uint32, -(-scratch // 32) * 32, 1, self.count_cores)
        ttnn.generic_op(list(args), ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return out

    def __call__(self, logits, u_table, col):
        """logits [1, 1, B, V] fp32 (this chip's [1, 1, B, V / tp] slice when sharded), u_table [1, 1, B, C] fp32 (each
        user's uniform per position), col [1, 1, B, 1] fp32 (the position each user samples at) -> token ids
        [1, 1, B, 1] fp32 (exact integers), the same on every chip."""
        if self.row_inv_temp is not None:
            x = ttnn.multiply(logits, self.row_inv_temp)
        else:
            x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        vals, tok_ids = self._top_k(x)
        B = vals.shape[-2]
        vmax = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, B, 1])
        e = ttnn.exp(ttnn.subtract(vals, vmax))
        return self._tail(vals, e, tok_ids, u_table, col)
