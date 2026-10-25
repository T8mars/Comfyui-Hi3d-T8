"""Fetch and verify the pinned upstream source for a code-only release build."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
ROOT = parser.parse_args().root.resolve()
manifest = json.loads((ROOT / 'upstream-source-manifest.json').read_text(encoding='utf-8'))
target = ROOT / 'upstream/Hi3Dv1.0'
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/hi3d-official/Hi3Dv1.0.git', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', manifest['revision']], check=True)
for name, expected in manifest['files'].items():
    path = target / name
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        # The validated Windows checkout recorded CRLF bytes. Reproduce those
        # exact bytes on Linux only if the pinned hash proves equivalence.
        windows = content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        assert hashlib.sha256(windows).hexdigest() == expected, name
        path.write_bytes(windows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest['files'])} pinned upstream source files")
