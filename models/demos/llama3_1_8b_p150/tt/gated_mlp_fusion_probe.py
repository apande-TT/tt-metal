"""Gate+up (w1/w3) matmul fusion for the decode MLP -- MEASURED and REJECTED. Not wired in.

The catalogued QKV lever (GUIDELINES/03 section 1) says to fuse projections that share an input
into ONE matmul: "Splitting into three matmuls adds two ops and underutilizes the grid." The gated
MLP looks like the same shape of win -- w1 (gate) and w3 (up) are both [dim, hidden], both read the
SAME activation x, and both feed one elementwise combine -- so the fused form is:

    W13 = concat(w3, w1, dim=-1)          # [4096, 28672]
    w2_in = ttnn.swiglu(x @ W13, -1)      # swiglu = A * silu(B) over the two halves

Concat order is [w3, w1], NOT [w1, w3]: ttnn.swiglu applies silu to the SECOND half, and Llama's
MLP is down(silu(x@w1) * (x@w3)), so w1 must land in the second half.

MEASURED on the decode shape M=32, K=4096, N=14336, bf4_b weights, bf8_b intermediates:

    form                                        PCC         ms/call
    2 matmuls + fused-SILU mul (current model)   0.987542    0.2408
    1 matmul (2N wide) + ttnn.swiglu             0.987539    0.2523   -> 4.8% SLOWER

The fusion is mathematically EXACT (the two PCCs agree to 5 decimal places, so the concat order and
half assignment are right). It is still a loss, for two reasons that also explain why the QKV
analogy does not carry:

1. The matmul saves nothing. It is DRAM-bandwidth bound on the weights, and concatenating w1 and w3
   does not change how many weight bytes must be read -- it is the same 66 MB either way. The QKV
   fusion wins on grid utilisation at large M; here M is one tile, so there is no occupancy to gain.
2. The combine gets MORE expensive, not less. The model does not run a bare multiply it could fold
   away -- it already runs ONE op, `ttnn.mul(a, b, input_tensor_a_activations=[SILU])`, with the
   activation fused into the multiply. `ttnn.swiglu` is a composite that slices the wide tensor into
   two halves and then does silu+multiply, so it replaces one fused op with strictly more work.

So the op count is 3 -> 2 on paper but the wall time goes up. The cheaper fusion (activation folded
into the multiply) is already applied, and it is the one that matters.

Run directly to reproduce the table.
"""
import time

import torch

import ttnn

M, K, N = 32, 4096, 14336


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        x = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
        w1 = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        w3 = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        gold = (torch.nn.functional.silu(x.float() @ w1.float()) * (x.float() @ w3.float())).flatten()

        kw = dict(device=dev, layout=ttnn.TILE_LAYOUT)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, **kw)
        t1 = ttnn.from_torch(w1.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat4_b, **kw)
        t3 = ttnn.from_torch(w3.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat4_b, **kw)
        t13 = ttnn.from_torch(
            torch.cat([w3, w1], dim=-1).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat4_b, **kw
        )

        def pcc(t):
            return torch.corrcoef(torch.stack([gold, ttnn.to_torch(t).float().flatten()]))[0, 1].item()

        def base():
            a = ttnn.linear(tx, t1, dtype=ttnn.bfloat8_b)
            b = ttnn.linear(tx, t3, dtype=ttnn.bfloat8_b)
            o = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return o

        def fused():
            h = ttnn.linear(tx, t13, dtype=ttnn.bfloat8_b)
            o = ttnn.swiglu(h, -1)
            ttnn.deallocate(h)
            return o

        for label, fn in (("BASE", base), ("FUSED", fused)):
            o = fn()
            print("%s_PCC=%.6f" % (label, pcc(o)))
            ttnn.deallocate(o)
            fn()
            ttnn.synchronize_device(dev)
            t0 = time.monotonic()
            for _ in range(10):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            print("%s_MS=%.4f" % (label, (time.monotonic() - t0) * 100.0))
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
