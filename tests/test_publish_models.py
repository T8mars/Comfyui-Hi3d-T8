import hashlib

import pytest

from tools.publish_models import studio_components


def fixture_spec(tmp_path):
    folder = tmp_path / 'example'
    folder.mkdir()
    contents = {'LICENSE': b'upstream license\n', 'weights.safetensors': b'pinned model fixture'}
    files = []
    for name, data in contents.items():
        (folder / name).write_bytes(data)
        algorithm = 'git-sha1' if name == 'LICENSE' else 'sha256'
        expected = (hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                    if algorithm == 'git-sha1' else hashlib.sha256(data).hexdigest())
        files.append({'name': name, 'size': len(data), 'hash': expected, 'algorithm': algorithm})
    return {'example': {'repo': 'upstream/example', 'revision': 'fixed-revision',
                        'license': 'MIT', 'gated': False, 'files': files}}


def test_studio_publication_preserves_pinned_hashes_and_provenance(tmp_path):
    specs = fixture_spec(tmp_path)
    groups, catalogue, manifest = studio_components(tmp_path, specs)
    assert len(groups['studio/example']) == 2
    assert set(catalogue) == {'studio/example/LICENSE', 'studio/example/weights.safetensors'}
    public = manifest['components']['example']
    assert public['repo'] == 'upstream/example'
    assert public['revision'] == 'fixed-revision'
    assert public['license'] == 'MIT'
    for name, entry in public['files'].items():
        assert set(entry) == {'bytes', 'hash', 'algorithm'}
        assert entry['hash'] == next(e['hash'] for e in specs['example']['files'] if e['name'] == name)


def test_studio_publication_rejects_same_length_corrupt_weight(tmp_path):
    specs = fixture_spec(tmp_path)
    path = tmp_path / 'example/weights.safetensors'
    path.write_bytes(b'x' * path.stat().st_size)
    with pytest.raises(AssertionError):
        studio_components(tmp_path, specs)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/absolute', 'folder\\outside'])
def test_studio_publication_rejects_escaping_paths(tmp_path, name):
    specs = fixture_spec(tmp_path)
    specs['example']['files'][0]['name'] = name
    with pytest.raises(AssertionError):
        studio_components(tmp_path, specs)
