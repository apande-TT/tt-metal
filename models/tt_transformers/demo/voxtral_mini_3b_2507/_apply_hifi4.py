"""Raise every graduated stub to the SAME compute config the graduated
`llama_model` body already uses: MathFidelity.HiFi4 with fp32 destination
accumulation.

Why: the bring-up graduated each stub in isolation against a per-component PCC
target, and only `llama_model` picked an explicit compute kernel config.  In the
composed 30-layer + 32-layer chain the default (lower) fidelity compounds --
measured on device: the audio tower drops to 0.9936 where torch bf16 holds
0.9997, and the lm_head alone mis-ranks 22/379 argmax positions that HiFi4 gets
right.  This is a precision repair, not a behaviour change: same ops, same order,
same weights.

Adds `compute_kernel_config=_HIFI4_CFG` to every ttnn.linear / rms_norm /
layer_norm / matmul / scaled_dot_product_attention[_decode] call that does not
already carry one.  Uses the AST to find each call's closing paren, so
multi-line calls are handled correctly.
"""
from __future__ import annotations

import ast
import pathlib
import sys

STUBS = pathlib.Path(__file__).resolve().parent / "_stubs"

TARGETS = {
    "linear",
    "matmul",
    "rms_norm",
    "layer_norm",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_decode",
}

CFG_SRC = """
_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
"""


def _attr_name(node):
    cur = node.func
    if isinstance(cur, ast.Attribute):
        return cur.attr
    return None


def _is_ttnn_call(node):
    cur = node.func
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id == "ttnn"


def patch(path: pathlib.Path) -> int:
    src = path.read_text()
    if "_HIFI4_CFG" in src and "compute_kernel_config=_HIFI4_CFG" in src:
        already = True
    else:
        already = False
    tree = ast.parse(src)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_ttnn_call(node):
            continue
        if _attr_name(node) not in TARGETS:
            continue
        if any(k.arg == "compute_kernel_config" for k in node.keywords):
            continue
        edits.append((node.end_lineno, node.end_col_offset))
    if not edits:
        return 0

    lines = src.splitlines(keepends=True)
    # apply from the end so earlier offsets stay valid
    for lineno, col in sorted(edits, reverse=True):
        line = lines[lineno - 1]
        assert line[col - 1] == ")", f"{path.name}:{lineno} expected ')' got {line[col-1]!r}"
        head, tail = line[: col - 1], line[col - 1 :]
        # find the last non-whitespace char BEFORE the closing paren, which may
        # live on an earlier line for a multi-line call
        prev = head.rstrip()
        i = lineno - 2
        while not prev and i >= 0:
            prev = lines[i].rstrip()
            i -= 1
        sep = "" if prev.endswith((",", "(")) else ", "
        lines[lineno - 1] = f"{head}{sep}compute_kernel_config=_HIFI4_CFG{tail}"
    out = "".join(lines)

    if not already and "_HIFI4_CFG =" not in out:
        marker = "import ttnn\n"
        idx = out.index(marker) + len(marker)
        out = out[:idx] + CFG_SRC + out[idx:]

    ast.parse(out)  # must still be valid
    path.write_text(out)
    return len(edits)


def main():
    names = sys.argv[1:] or [p.stem for p in sorted(STUBS.glob("*.py")) if not p.stem.startswith("_")]
    total = 0
    for n in names:
        p = STUBS / f"{n}.py"
        if not p.is_file():
            print(f"  skip {n}: no such stub")
            continue
        k = patch(p)
        total += k
        print(f"  {n:32s} {k} call site(s) raised to HiFi4+fp32acc")
    print(f"total: {total} call sites")


if __name__ == "__main__":
    main()
