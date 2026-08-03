# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tt-lang rung for the long-prefill QKV projection (MinimalMatmul 1024 x 3840 x 8192).

MEASURED OUT -- this rung is blocked by the toolchain, not by the kernel design, and the block is
fatal for THIS op specifically. Probe it with::

    TTLANG_COMPILE_ONLY=1 python -m models.demos.multimodal.gemma3.tt.ttl_qkv_matmul

The kernel must live in a real file: ``inspect.getsource`` is what ttl reads, so a heredoc cannot be
probed. The probe is device-free and takes ~2 s, which is the right budget for this rung.

Adapted from GUIDELINES/11's matmul template with the ttl-1.0.1 divergences this tree has already
paid for elsewhere applied up front: ``ttl.block`` does not exist and a block's ``.shape`` is not
introspectable, so the accumulator cannot be zero-filled -- the first k-step is PEELED and stores
``a_blk @ b_blk`` directly (an ``if kt == 0`` inside the loop would force the k-loop to unroll at
trace time). The grid is the real device grid rather than the template's single-core ``(1, 1)``.

THE BLOCKER CHAIN, each step reproduced on this op:

1. At the tensors' NATURAL rank 4, the constraints contradict each other. A dataflow buffer's rank
   must match its tensor's, so the blocks are rank 4 -- but ``@`` rejects them::

       shape mismatch between (1, 1) bf16 tensor and tensor<1x1x1x1x!ttcore.tile<32x32, bf16>>
       error: lhs must be rank 2, got rank 4              (with a rank-2 accumulator instead)

   There is no rank at which both the buffer-rank rule and the matmul-arity rule hold for a rank-4
   tensor, so the op has to be reshaped to rank 2 first.

2. At rank 2 the ranks clear and it lands on the wall that actually matters here::

       error: element type mismatch: lhs has '!ttcore.tile<32x32, bf16>'
                                     but rhs has '!ttcore.tile<32x32, bfp_bf4>'

   ttl 1.0.1 cannot express a mixed-precision matmul, and this op IS one: the dtype rung already
   walked wqkv to bfloat4_b against a bfloat16 activation. The only way past is upcasting the weight
   back to bfloat16 -- which quadruples the weight read on a projection whose cost is that read, so
   it would undo a banked win to unblock a rung that then still has to beat ttnn.

3. With a matching bfloat16 weight the kernel DOES lower ("PROBE: lowered"). That is not a pass:
   ``TTLANG_COMPILE_ONLY`` skips the metal JIT, and on device trisc0 then fails on ttl's
   ``mm_block_init``/``mm_block_init_short``, which this tree renamed to ``matmul_block_init``
   (``grep -r mm_block_init tt_metal/`` = zero hits, and the ``_short`` variant exists under no
   name).

The C++ Metalium rung is unaffected by all three -- hand-written kernels call ``matmul_block_init``
and ``matmul_tiles``, which this tree does declare.
"""

import os

import ttnn

try:
    import ttl
except ImportError:  # the rung is unavailable rather than broken
    ttl = None

TILE = 32

# The QKV projection's real shape and dtype contract: a bfloat16 activation against the bfloat4_b
# weight the dtype rung already walked wqkv down to. Rank 2, per blocker (1).
M, K, N = 1024, 3840, 8192
GRID = (10, 8)


if ttl is not None:

    @ttl.operation(grid=GRID)
    def qkv_matmul(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        m_tiles, n_tiles, k_tiles = a.shape[-2] // TILE, b.shape[-1] // TILE, a.shape[-1] // TILE
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read():
            for mt in range(m_tiles):
                for nt in range(n_tiles):
                    for kt in range(k_tiles):
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            ta = ttl.copy(a[mt, kt], a_blk)
                            tb = ttl.copy(b[kt, nt], b_blk)
                            ta.wait()
                            tb.wait()

        @ttl.compute()
        def compute():
            for _ in range(m_tiles):
                for _ in range(n_tiles):
                    # PEEL the first k-step: there is no ttl.block.fill to zero an accumulator with.
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(a_blk @ b_blk)
                    for _ in range(k_tiles - 1):
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, acc_dfb.wait() as pre:
                            with acc_dfb.reserve() as acc_blk:
                                acc_blk.store(pre + a_blk @ b_blk)
                    with acc_dfb.wait() as acc_blk:
                        with y_dfb.reserve() as y_blk:
                            y_blk.store(acc_blk)

        @ttl.datamovement()
        def write():
            for mt in range(m_tiles):
                for nt in range(n_tiles):
                    with y_dfb.wait() as y_blk:
                        ttl.copy(y_blk, y[mt, nt]).wait()


def _probe(weight_dtype=ttnn.bfloat4_b):
    """Device-free lowering probe. Reproduces the rung's blocker without burning a device round.

    Defaults to the op's REAL weight dtype, which is the configuration that matters; pass
    ``ttnn.bfloat16`` to reach blocker (3) instead of stopping at (2).
    """
    import torch

    if ttl is None:
        print("PROBE: ttl not importable — rung unavailable")
        return
    os.environ.setdefault("TTLANG_COMPILE_ONLY", "1")
    a = ttnn.from_torch(torch.zeros(M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch.zeros(K, N), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    y = ttnn.from_torch(torch.zeros(M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    try:
        qkv_matmul(a, b, y)
        print("PROBE: lowered")
    except Exception as exc:  # the blocker is the result we are after
        print("PROBE FAILED: {}: {}".format(type(exc).__name__, exc))


if __name__ == "__main__":
    _probe()
