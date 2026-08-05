"""Re-stamp _stubs/_e2e_repairs.json with the CURRENT sha256 of each repaired stub.

The reasons are fixed; only the digests are refreshed.  Run this after any
deliberate edit to a repaired stub so the Phase-0 inventory check keeps its
teeth (an UNdeclared divergence must still fail).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

S = pathlib.Path(__file__).resolve().parent / "_stubs"
MANIFEST = S / "_e2e_repairs.json"


def main():
    doc = json.loads(MANIFEST.read_text())
    for name, entry in doc["repairs"].items():
        live = (S / f"{name}.py").read_bytes()
        snap = (S / f"{name}.py.{entry['snapshot']}").read_bytes()
        entry["snapshot_sha256"] = hashlib.sha256(snap).hexdigest()
        entry["repaired_sha256"] = hashlib.sha256(live).hexdigest()
        print(f"{name}: repaired={entry['repaired_sha256'][:12]} snapshot={entry['snapshot_sha256'][:12]}")
    MANIFEST.write_text(json.dumps(doc, indent=2))
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
