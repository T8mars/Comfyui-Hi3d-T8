"""A Windows Git checkout must preserve pinned notice and native-source hashes."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest


def test_autocrlf_checkout_preserves_pinned_third_party_and_native_bytes(tmp_path):
    git = shutil.which('git')
    if not git:
        pytest.skip('Git required for portable source byte verification')
    root = Path(__file__).parents[1]
    repository = tmp_path / 'source'
    repository.mkdir()
    names = (
        'licenses/asset-studio/models/p3sam/README.md',
        'hi3d_workbench/web/vendor/three/loaders/RGBELoader.js',
        'tools/native/autoremesher_cli.cpp',
    )
    expected = {}
    for name in ('.gitattributes', *names):
        payload = (root / name).read_bytes()
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()

    def run(*arguments, cwd=repository):
        return subprocess.run([git, '-c', 'core.autocrlf=true', '-c', 'core.safecrlf=false',
                               *arguments], cwd=cwd, check=True, capture_output=True).stdout

    run('init', '-q')
    run('add', '--', '.gitattributes', *names)
    for name in names:
        assert hashlib.sha256(run('show', ':' + name)).hexdigest() == expected[name]
    run('-c', 'user.name=Portable source test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'Test pinned source bytes')
    checkout = tmp_path / 'checkout'
    run('clone', '-q', '--no-local', str(repository), str(checkout), cwd=tmp_path)
    for name in names:
        assert hashlib.sha256((checkout / name).read_bytes()).hexdigest() == expected[name]
