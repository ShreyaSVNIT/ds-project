#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = []

for p in sorted(ROOT.rglob("*.csv*")):
    if "data/raw" in str(p) or ".git" in str(p):
        continue
    rel = p.relative_to(ROOT)
    lines.append(f"{rel}\t{p.stat().st_size} bytes")

print("\n".join(lines[:30]))
(ROOT / "SHA256SUMS.txt").write_text("\n".join(lines), encoding="utf-8")

