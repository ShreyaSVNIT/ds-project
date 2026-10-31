#!/usr/bin/env python3
"""Write SHA-256 checksums for tracked repository files."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "SHA256SUMS.txt"

try:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"Unable to list tracked files with Git: {exc}") from exc

lines = []
for relative_name in tracked:
    if relative_name == manifest.name:
        continue
    path = ROOT / relative_name
    if not path.is_file():
        raise SystemExit(f"Tracked file is missing: {relative_name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {relative_name}")

manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} tracked-file checksums to {manifest}")
