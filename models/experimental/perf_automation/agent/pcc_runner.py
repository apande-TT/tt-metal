"""PCC measurement for GATE_PCC (PLAN 8.6) — single-stage e2e.

parse_pcc() is deterministic and unit-tested. run_pcc() runs the model's
end-to-end PCC test on hardware and is the injectable default (ctx.deps["pcc_runner"]);
it is exercised live, not in unit tests. TBD(pcc-parse): the regex assumes the
test prints a "PCC: <float>" style number — refine per the real test's output.
"""

from __future__ import annotations

import os
import re
import subprocess

from . import gitio

_PCC_RE = re.compile(r"(?i)pcc\D{0,12}(-?\d\.\d+)")


def parse_pcc(text: str):
    """Last 'pcc ... <float>' occurrence in the test output, or None."""
    matches = _PCC_RE.findall(text or "")
    return float(matches[-1]) if matches else None


def run_pcc(ctx) -> dict:
    """Run the e2e PCC test, parse the measured PCC, compare the manifest threshold.

    Returns {status: ok|pcc_low|crash, pcc?, error?}. A parsed number below
    threshold is pcc_low (expected pytest non-zero exit); an unparseable result
    or an exception is crash.
    """
    entry = ctx.manifest["pathmap"]["pcc"]["end_to_end"]
    test = str(ctx.model_root() / entry["path"])
    threshold = entry["threshold"]
    env = dict(os.environ)
    vd = ctx.manifest.get("config", {}).get("visible_devices")
    if vd is not None:
        env["TT_VISIBLE_DEVICES"] = str(vd)
        env["TT_METAL_VISIBLE_DEVICES"] = str(vd)
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "-o", "addopts=", test, "-sv"],
            cwd=str(gitio.repo_root(ctx.model_root())),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except Exception as exc:  # timeout, OS error, etc.
        return {"status": "crash", "error": str(exc)}
    out = (r.stdout or "") + (r.stderr or "")
    pcc = parse_pcc(out)

    # A SKIPPED e2e test verified NOTHING -- never accept it as correct just because a stale
    # "pcc=..." string happened to be in the log (the seamless SKIP-mislabel pattern).
    if re.search(r"\b[1-9]\d*\s+skipped\b", out, re.IGNORECASE) and not re.search(r"\b[1-9]\d*\s+passed\b", out):
        return {"status": "crash", "error": "e2e PCC test SKIPPED (correctness NOT verified): " + out.strip()[-800:]}

    if pcc is None:
        return {"status": "crash", "error": out.strip()[-800:]}

    # HONOR THE TEST'S OWN VERDICT -- do not just trust the scraped PCC float. The e2e test
    # asserts MORE than logits PCC (no runtime fallbacks, all modules invoked, exact token
    # match). If pytest exited non-zero while PCC>=threshold, one of those OTHER gates failed
    # -- the model is numerically close but wrong (e.g. wrong tokens). Banking that as "ok" is
    # exactly how a broken pipeline gets reported correct. A below-threshold PCC is the ONE
    # expected non-zero exit -> pcc_low (repairable); anything else is a real failure.
    if r.returncode != 0 and pcc >= threshold:
        return {
            "status": "crash",
            "error": (
                "e2e test FAILED a correctness gate despite PCC>=threshold "
                "(token-match / modules-invoked / runtime-fallback): " + out.strip()[-800:]
            ),
        }
    return (
        {"status": "ok", "pcc": pcc} if (r.returncode == 0 and pcc >= threshold) else {"status": "pcc_low", "pcc": pcc}
    )
