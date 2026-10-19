# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How a Wan VAE stack (encoder / decoder port) lays a batch of BTHWC activations over a 2-D mesh.

batch_parallel:
  False   H over mesh axis 0, W over axis 1 (spatial: every conv exchanges halos)
  True    batch over axis 0, W over axis 1
  "full"  batch over axis 0, then axis 1: whole images per chip, no H / W partition, so no conv needs
          a halo exchange or width masking; a batch that does not divide the mesh is padded with zero
          images, dropped again on the way back
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig

H_AXIS = 0
W_AXIS = 1


def mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


def _wide_rows(t, min_width=32):
    """(b, 1, rows, w): t's per-entry elements in row-major pages w wide, for a row-major t whose own rows are
    narrower than min_width; None when that does not apply."""
    s = list(t.shape)
    if t.layout != ttnn.ROW_MAJOR_LAYOUT or len(s) < 2 or s[-1] >= min_width:
        return None
    per = 1
    for d in s[1:]:
        per *= d
    w = next((w for w in (1024, 512, 256, 128, 64, 32) if per % w == 0), None)
    return None if w is None else [s[0], 1, per // w, w]


class BatchSplit:
    def __init__(self, device, batch_parallel=False):
        shape = mesh_shape(device)
        # mesh axes the batch is split over, in partition order (gathered back in reverse)
        self.axes = [H_AXIS] if (batch_parallel and shape[H_AXIS] > 1) else []
        if batch_parallel == "full" and shape[W_AXIS] > 1:
            self.axes.append(W_AXIS)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1 if H_AXIS in self.axes else shape[H_AXIS], mesh_axis=H_AXIS),
            width_parallel=ParallelFactor(factor=1 if W_AXIS in self.axes else shape[W_AXIS], mesh_axis=W_AXIS),
        )
        self.factor = 1
        for a in self.axes:
            self.factor *= shape[a]

    def scatter(self, x):
        """Replicated row-major BTHWC -> this chip's part. Returns (part, pad_b)."""
        pc = self.parallel_config
        H, W = x.shape[2], x.shape[3]
        assert (
            H % pc.height_parallel.factor == 0 and W % pc.width_parallel.factor == 0
        ), f"{H}x{W} must divide the {pc.height_parallel.factor}x{pc.width_parallel.factor} spatial mesh"
        pad_b = -x.shape[0] % self.factor
        if pad_b:
            x = ttnn.pad(x, [(0, pad_b), (0, 0), (0, 0), (0, 0), (0, 0)], value=0.0)
        part = self._batch_part(x) if self.axes else None
        if part is not None:
            x = part
        for a in self.axes if part is None else ():
            x = ttnn.mesh_partition(x, dim=0, cluster_axis=a)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return x, pad_b

    _COPY_READER = "models/demos/qwen_image_edit/tt/kernels/chunk_copy_reader.cpp"
    _COPY_WRITER = "models/demos/qwen_image_edit/tt/kernels/chunk_copy_writer.cpp"
    _CHUNK = 64

    def _batch_part(self, x):
        """The batch partitions over self.axes in ONE device copy: each chip copies its own images' pages of the
        replicated row-major x (exact data movement; one program per chip, its page offset a runtime arg).
        None when x is not a plain interleaved row-major tensor on a mesh."""
        device = x.device()
        if not isinstance(device, ttnn.MeshDevice) or x.layout != ttnn.ROW_MAJOR_LAYOUT or x.is_sharded():
            return None
        shape = mesh_shape(device)
        B = x.shape[0]
        b = B // self.factor
        per_item = 1
        for d in list(x.shape)[1:-1]:
            per_item *= d
        pages = b * per_item
        y = ttnn.empty([b] + list(x.shape)[1:], dtype=x.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        page_bytes = x.shape[-1] * (4 if x.dtype == ttnn.float32 else 2)
        grid = device.compute_with_storage_grid_size()
        n_cores = min(grid.x * grid.y, pages)
        per, extra = divmod(pages, n_cores)
        work, start = [], 0
        for i in range(n_cores):
            cnt = per + (1 if i < extra else 0)
            work.append((ttnn.CoreCoord(i % grid.x, i // grid.x), start, cnt))
            start += cnt
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in work])
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * self._CHUNK * page_bytes,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=x.dtype, page_size=page_bytes)],
            )
        ]
        mesh_pd = ttnn.MeshProgramDescriptor()
        for r in range(shape[0]):
            for c in range(shape[1]):
                idx = (r, c)
                chunk = 0
                for a in self.axes:  # mesh_partition order: the first axis splits the batch, the next its parts
                    chunk = chunk * shape[a] + idx[a]
                src0 = chunk * pages
                rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for core, st, cnt in work:
                    rd[core.x][core.y] = [x.buffer_address(), src0 + st, cnt]
                    wr[core.x][core.y] = [y.buffer_address(), st, cnt]
                kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=self._COPY_READER,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores,
                        compile_time_args=ttnn.TensorAccessorArgs(x).get_compile_time_args() + [self._CHUNK],
                        runtime_args=rd,
                        config=ttnn.ReaderConfigDescriptor(),
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=self._COPY_WRITER,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores,
                        compile_time_args=ttnn.TensorAccessorArgs(y).get_compile_time_args() + [self._CHUNK],
                        runtime_args=wr,
                        config=ttnn.WriterConfigDescriptor(),
                    ),
                ]
                coord = ttnn.MeshCoordinate(r, c)
                mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                    kernels=kernels, semaphores=[], cbs=cbs
                )
        ttnn.generic_op([x, y], mesh_pd)
        return y

    def gather(self, out, ccl_manager, B, pad_b):
        """This chip's BTHWC part -> the replicated [B, ...] whole."""
        pc = self.parallel_config
        if pc.width_parallel.factor > 1:
            out = ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        if pc.height_parallel.factor > 1:
            out = ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        if self.axes:
            # a narrow row-major output (the decoder's 3-channel image: 12-byte pages) is gathered as wide pages
            # of the same per-image elements, then viewed back: a pure relayout, far fewer CCL pages
            shape = list(out.shape)
            wide = _wide_rows(out)
            if wide is not None:
                out = ttnn.reshape(out, wide)
            for a in reversed(self.axes):
                out = ccl_manager.all_gather(out, dim=0, mesh_axis=a, use_hyperparams=False)
            if wide is not None:
                out = ttnn.reshape(out, [out.shape[0]] + shape[1:])
        if pad_b:
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (B,) + tuple(out.shape)[1:])
        return out
