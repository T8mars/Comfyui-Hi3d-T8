"""Fetch and verify the pinned upstream source for a code-only release build."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'upstream-source-manifest.json').read_text(encoding='utf-8'))
target = ROOT / 'upstream/Hi3Dv1.0'
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/hi3d-official/Hi3Dv1.0.git', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', manifest['revision']], check=True)
for name, expected in manifest['files'].items():
    assert hashlib.sha256((target / name).read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest['files'])} pinned upstream source files")
