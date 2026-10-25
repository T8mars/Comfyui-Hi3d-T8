"""CPU tests for UV provenance, bounded fallback and verified caching."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from hi3d_workbench.engines import mesh_uv


def hard_box():
    mesh = trimesh.creation.box()
    mesh.unmerge_vertices()
    mesh.vertex_normals = np.repeat(mesh.face_normals, 3, axis=0)
    return mesh


def arrays_for(faces):
    mapping = np.asarray(faces, dtype=np.int64).reshape(-1)
    uv_faces = np.arange(len(mapping), dtype=np.int64).reshape(-1, 3)
    side = int(np.ceil(np.sqrt(len(faces))))
    base = np.column_stack((np.arange(len(faces)) % side, np.arange(len(faces)) // side)) / side
    uv = (base[:, None, :]+np.array([[[.1, .1], [.8, .1], [.1, .8]]])/side).reshape(-1, 2).astype(np.float32)
    return mapping, uv_faces, uv


def fake_community(monkeypatch, calls=None):
    from hi3d_workbench.engines import community_uv
    def unwrap(vertices, faces, resolution, padding, notify):
        if calls is not None:
            calls.append((vertices.copy(), faces.copy(), resolution, padding))
        notify('uv_segment', '聚类运行中', device='cuda', current=4, unit='iterations')
        notify('uv_parameterize', '参数化完成', device='cuda', current=3, total=3, unit='charts')
        return arrays_for(faces)
    monkeypatch.setattr(community_uv, 'unwrap', unwrap)
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', lambda *args, **kw: (_ for _ in ()).throw(
        ImportError('native runtime intentionally unavailable in community/cache test')))


def test_original_index_community_path_keeps_every_hard_corner(monkeypatch):
    fake_community(monkeypatch)
    source = hard_box()
    before = [np.asarray(value).copy() for value in (source.vertices, source.faces, source.vertex_normals)]
    result, policy = mesh_uv.ensure_uv(source, backend='community')
    mapping = source.faces.reshape(-1)
    assert policy == 'community_GPU_PEC_original_indices'
    assert np.array_equal(result.vertices, source.vertices[mapping])
    assert np.array_equal(result.vertex_normals, source.vertex_normals[mapping])
    mesh_uv._require_complete_faces(source.faces, mapping[result.faces])
    assert all(np.array_equal(a, b) for a, b in zip(before, (source.vertices, source.faces, source.vertex_normals)))


def test_existing_uv_is_preserved_without_importing_any_inference_backend(monkeypatch):
    source = hard_box()
    source.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(source.vertices), 2)))
    monkeypatch.setattr(mesh_uv, '_valid_source_faces', lambda *args: pytest.fail('existing UV should return immediately'))
    result, policy = mesh_uv.ensure_uv(source)
    assert result is source
    assert policy == 'preserved_existing'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reverse', 'negative_index', 'outside_index', 'nan_uv', 'outside_uv', 'collapsed_uv', 'reused_uv'])
def test_community_output_is_rejected_if_provenance_or_uv_is_invalid(monkeypatch, mutation):
    from hi3d_workbench.engines import community_uv
    def invalid(vertices, faces, *args):
        mapping, new_faces, uv = arrays_for(faces)
        if mutation == 'missing': new_faces = new_faces[:-1]
        if mutation == 'duplicate': new_faces = np.vstack((new_faces, new_faces[0]))
        if mutation == 'reverse': new_faces[0] = new_faces[0, ::-1]
        if mutation == 'negative_index': mapping[0] = -1
        if mutation == 'outside_index': new_faces[0, 0] = len(mapping)
        if mutation == 'nan_uv': uv[0, 0] = np.nan
        if mutation == 'outside_uv': uv[0, 0] = 5.
        if mutation == 'collapsed_uv': uv[:3] = uv[0]
        if mutation == 'reused_uv': uv[3:6] = uv[:3]
        return mapping, new_faces, uv
    monkeypatch.setattr(community_uv, 'unwrap', invalid)
    with pytest.raises(RuntimeError):
        mesh_uv.ensure_uv(hard_box(), backend='community')


def test_only_explicit_zero_area_faces_can_be_removed(monkeypatch):
    calls = []
    fake_community(monkeypatch, calls)
    source = hard_box()
    source = trimesh.Trimesh(source.vertices, np.vstack((source.faces, [0, 0, 1])),
                             vertex_normals=source.vertex_normals, process=False)
    events = []
    result, _ = mesh_uv.ensure_uv(source, backend='community', emit=lambda **event: events.append(event))
    assert len(result.faces) == len(source.faces)-1
    assert len(calls[0][1]) == len(source.faces)-1
    assert events[-1]['removed_degenerate_faces'] == 1


def test_progress_is_phase_local_and_pec_iterations_do_not_claim_percentage(monkeypatch):
    fake_community(monkeypatch)
    events = []
    mesh_uv.ensure_uv(hard_box(), emit=lambda **event: events.append(event))
    segment = next(event for event in events if event['phase'] == 'uv_segment')
    assert segment['progress_current'] == 4
    assert segment['progress_unit'] == 'iterations'
    assert 'progress_total' not in segment
    assert segment['device'] == 'cuda'
    assert all(event['stage'] == 'unwrapping' for event in events)


def test_cache_reuses_only_verified_uv_and_reconstructs_source_attributes(monkeypatch, tmp_path):
    calls = []
    fake_community(monkeypatch, calls)
    source = hard_box()
    first, policy = mesh_uv.ensure_uv(source, cache_dir=tmp_path)
    events = []
    second, second_policy = mesh_uv.ensure_uv(source, cache_dir=tmp_path, emit=lambda **event: events.append(event))
    assert len(calls) == 1
    assert second_policy == 'cached_'+policy
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.vertex_normals, second.vertex_normals)
    assert np.array_equal(first.visual.uv, second.visual.uv)
    assert events[-1]['cache_hit'] is True
    with np.load(next(tmp_path.glob('*.npz')), allow_pickle=False) as saved:
        assert set(saved.files) == {'mapping', 'faces', 'uv', 'metadata'}


@pytest.mark.parametrize('change', ['vertices', 'normals', 'winding', 'texture_size', 'backend', 'algorithm'])
def test_cache_identity_binds_all_geometry_and_atlas_inputs(monkeypatch, tmp_path, change):
    calls = []
    fake_community(monkeypatch, calls)
    source = hard_box()
    mesh_uv.ensure_uv(source, cache_dir=tmp_path)
    options = {'cache_dir': tmp_path}
    if change == 'vertices': source.vertices[0, 0] += .125
    if change == 'normals': source.vertex_normals = -source.vertex_normals
    if change == 'winding': source.faces[0] = source.faces[0, ::-1]
    if change == 'texture_size': options['texture_size'] = 4096
    if change == 'backend': options['backend'] = 'community'
    if change == 'algorithm': monkeypatch.setattr(mesh_uv, 'UV_ALGORITHM_VERSION', 'next-algorithm')
    mesh_uv.ensure_uv(source, **options)
    assert len(calls) == 2


@pytest.mark.parametrize('corruption', ['not_zip', 'payload', 'identity', 'lost_face'])
def test_corrupt_cache_is_reported_and_rebuilt(monkeypatch, tmp_path, corruption):
    calls = []
    fake_community(monkeypatch, calls)
    source = hard_box()
    mesh_uv.ensure_uv(source, cache_dir=tmp_path)
    path = next(tmp_path.glob('*.npz'))
    if corruption == 'not_zip':
        path.write_bytes(b'broken compressed cache')
    else:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        metadata = json.loads(str(arrays['metadata'].item()))
        if corruption == 'payload': arrays['uv'][0, 0] += .01
        if corruption == 'identity': metadata['identity'] = 'wrong identity'
        if corruption == 'lost_face':
            arrays['faces'] = arrays['faces'][:-1]
            metadata['payload_sha256'] = mesh_uv._digest_arrays((arrays['mapping'], arrays['faces'], arrays['uv']))
        arrays['metadata'] = np.asarray(json.dumps(metadata))
        with path.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    events = []
    mesh_uv.ensure_uv(source, cache_dir=tmp_path, emit=lambda **event: events.append(event))
    assert len(calls) == 2
    assert any(event['phase'] == 'uv_cache' and event.get('cache_hit') is False for event in events)


def test_atomic_cache_writes_are_safe_for_simultaneous_identical_requests(monkeypatch, tmp_path):
    fake_community(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: mesh_uv.ensure_uv(hard_box(), cache_dir=tmp_path), range(2)))
    assert len(list(tmp_path.glob('*.npz'))) == 1
    assert not list(tmp_path.glob('*.tmp'))
    assert np.array_equal(results[0][0].vertices, results[1][0].vertices)


def test_corner_mapping_preserves_ambiguous_overlapping_shells_and_opposite_normals():
    box = trimesh.creation.box()
    source = trimesh.Trimesh(np.vstack((box.vertices, box.vertices)),
                             np.vstack((box.faces, box.faces+len(box.vertices))), process=False)
    source.vertex_normals = np.vstack((box.vertex_normals, -box.vertex_normals))
    # The native mesh merged both coincident shells but retained every face
    # occurrence. UV seams create further output vertices.
    native_indices = np.vstack((box.faces, box.faces)).reshape(-1)
    native_positions = box.vertices[native_indices]
    native_faces = np.arange(len(native_indices)).reshape(-1, 3)
    uv = arrays_for(native_faces)[2]
    mapping, faces, out_uv = mesh_uv._corner_mapped_arrays(source, native_positions, native_faces, uv)
    mesh_uv._require_complete_faces(source.faces, mapping[faces])
    assert set(mapping.tolist()) == set(range(len(source.vertices)))
    result = mesh_uv._result_from_arrays(source, mapping, faces, out_uv)
    assert np.array_equal(result.vertex_normals, source.vertex_normals[mapping])
    assert np.array_equal(result.vertices, source.vertices[mapping])


@pytest.mark.parametrize('mutation', ['reverse', 'missing', 'position'])
def test_corner_mapping_never_hides_invalid_geometry(mutation):
    source = hard_box()
    positions = source.vertices.copy()
    faces = source.faces.copy()
    uv = np.zeros((len(positions), 2), dtype=np.float32)
    if mutation == 'reverse': faces[0] = faces[0, ::-1]
    if mutation == 'missing': faces = faces[:-1]
    if mutation == 'position': positions[0, 0] += 1e-4
    with pytest.raises(RuntimeError):
        mesh_uv._corner_mapped_arrays(source, positions, faces, uv)


def test_topology_partitions_cover_every_face_once_and_are_bounded():
    source = trimesh.creation.icosphere(subdivisions=2)
    chunks = mesh_uv._topology_chunks(source.faces, max_faces=31)
    assert all(0 < len(chunk) <= 31 for chunk in chunks)
    assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(len(source.faces)))


def test_cpu_fallback_preserves_normals_and_packs_all_partitions_once(monkeypatch):
    source = hard_box()
    monkeypatch.setattr(mesh_uv, '_CPU_CHUNK_FACES', 4)
    added, operations = [], []
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def numpy(self): return self.value
    class Atlas:
        def __init__(self): self.meshes = []
        def add_mesh(self, vertices, faces, normals):
            self.meshes.append((vertices.value, faces.value, normals.value))
            added.append(self.meshes[-1])
        def compute_charts(self, **options):
            assert options['normal_seam_weight'] > 1000
            callback = self._create_progress_callback()
            assert callback('Compute charts', 25)
            assert callback('Compute charts', 100)
            operations.append('compute')
        def pack_charts(self, **options):
            assert options['resolution'] == 4096
            assert options['texels_per_unit'] == 0.
            self._create_progress_callback()('Pack charts', 100)
            operations.append('pack')
        def get_mesh(self, index):
            vertices, faces, _ = self.meshes[index]
            uv = arrays_for(source.faces)[2]
            start = sum(len(v) for v, _, _ in self.meshes[:index])
            uv = uv[start:start+len(vertices)]
            return Tensor(np.arange(len(vertices))), Tensor(faces), Tensor(uv)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(from_numpy=Tensor))
    monkeypatch.setitem(sys.modules, 'cumesh', SimpleNamespace(Atlas=Atlas))
    events = []
    result, policy = mesh_uv.ensure_uv(source, backend='cpu', texture_size=4096,
                                      emit=lambda **event: events.append(event))
    assert policy == 'native_xatlas_partitioned_original_indices'
    assert len(added) == 3
    assert operations == ['compute', 'pack']
    assert len(result.faces) == len(source.faces)
    assert any(event['phase'] == 'uv_charts' and event.get('progress_total') == 100 for event in events)
    # Every input hard-edge vertex remains represented with its authored normal.
    original = sorted(map(tuple, np.column_stack((source.vertices, source.vertex_normals))))
    output = sorted(map(tuple, np.column_stack((result.vertices, result.vertex_normals))))
    assert original == output


def test_auto_reports_each_failure_and_runs_each_backend_at_most_once(monkeypatch):
    from hi3d_workbench.engines import community_uv
    calls = []
    def native(*args, **kw):
        calls.append('cumesh')
        raise RuntimeError('native source identity rejected')
    def community(*args):
        calls.append('community')
        raise RuntimeError('specific kernel failure')
    def cpu(mesh, faces, *args):
        calls.append('cpu')
        return arrays_for(faces)
    monkeypatch.setattr(community_uv, 'unwrap', community)
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', native)
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', cpu)
    events = []
    _, policy = mesh_uv.ensure_uv(hard_box(), emit=lambda **event: events.append(event))
    assert policy == 'native_xatlas_partitioned_original_indices'
    assert calls == ['cumesh', 'community', 'cpu']
    assert any('specific kernel failure' in event.get('fallback_reason', '') for event in events)
    assert any('native source identity rejected' in event.get('fallback_reason', '') for event in events)
    transitions = [(event['failed_backend'], event['next_backend']) for event in events if event['phase'] == 'uv_fallback']
    assert transitions == [('cumesh', 'community'), ('community', 'cpu')]


def test_invalid_backend_or_resolution_is_not_silently_accepted():
    with pytest.raises(ValueError, match='后端'):
        mesh_uv.ensure_uv(hard_box(), backend='guess')
    with pytest.raises(ValueError, match='尺寸'):
        mesh_uv.ensure_uv(hard_box(), texture_size=-1)


def test_pec_reports_monotone_iteration_count_not_invalid_upstream_merge_number():
    from hi3d_workbench.engines.community_uv import _segment_callback
    events = []
    callback = _segment_callback(lambda phase, message, **values: events.append((phase, values)))
    for done in (312, 277, 274805):
        callback(done, 973926)
    assert [event[1]['current'] for event in events] == [1, 2, 3]
    assert all(event[1]['unit'] == 'iterations' and 'total' not in event[1] for event in events)
    assert all('charts_remaining' not in event[1] for event in events)


def test_local_repair_splits_only_collapsed_uv_face_and_keeps_every_source_corner():
    from hi3d_workbench.engines.community_uv import _repair_local_faces
    source = hard_box()
    mapping, faces, uv = arrays_for(source.faces)
    # One non-degenerate 3D face projected to a UV line.
    uv[faces[5], 1] = .5
    pixels = uv.astype(np.float64)*2048
    events = []
    result = _repair_local_faces(source.vertices, mapping, faces, pixels, 2048, 2048, 4,
                                 lambda phase, message, **values: events.append((phase, values)))
    new_mapping, new_faces, new_uv, report = result
    mesh_uv._verify_arrays(source, source.faces, new_mapping, new_faces, new_uv)
    assert report['repaired_faces'] == 1
    assert report['repair_passes'] == 1
    assert np.array_equal(new_mapping[new_faces], mapping[faces])
    assert report['minimum_height_adjusted_faces'] == 0
    assert all(phase == 'uv_repair' for phase, _ in events)
    # The added island is above the occupied old atlas, with a padding gap.
    kept = np.delete(np.arange(len(faces)), 5)
    assert new_uv[new_faces[5], 1].max() < new_uv[new_faces[kept], 1].min()


def test_float32_roundoff_and_clipped_boundary_slivers_are_repaired_without_dropping_faces():
    from hi3d_workbench.engines.community_uv import _repair_local_faces
    source = hard_box()
    mapping, faces, uv = arrays_for(source.faces)
    pixels = uv.astype(np.float64)*2048
    # A point outside the nominal old atlas must grow the common bound,
    # rather than clip an entire boundary triangle to one line.
    pixels[faces[0]] = [[2048.1, 30.], [2048.2, 31.], [2048.3, 30.]]
    # Another chart has genuinely vanished at final float32 precision.
    pixels[faces[1]] = [[100., 100.], [100.+1e-12, 100.], [100., 100.+1e-12]]
    new_mapping, new_faces, new_uv, report = _repair_local_faces(
        source.vertices, mapping, faces, pixels, 2048, 2048, 4, lambda *args, **values: None)
    mesh_uv._verify_arrays(source, source.faces, new_mapping, new_faces, new_uv)
    assert report['repaired_faces'] >= 1
    assert np.array_equal(new_mapping[new_faces], mapping[faces])
    assert np.max(new_uv) <= 1 and np.min(new_uv) >= 0


def test_extreme_source_sliver_gets_minimum_texel_area_without_geometry_change():
    from hi3d_workbench.engines.community_uv import _repair_local_faces
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [.4, 1e-13, 0.]])
    source = trimesh.Trimesh(vertices, [[0, 1, 2]], process=False)
    new_mapping, new_faces, uv, report = _repair_local_faces(
        vertices, np.arange(3), np.array([[0, 1, 2]]), np.zeros((3, 2)),
        2048, 2048, 4, lambda *args, **values: None)
    mesh_uv._verify_arrays(source, source.faces, new_mapping, new_faces, uv)
    assert report['minimum_height_adjusted_faces'] == 1
    triangle = uv[new_faces[0]].astype(np.float64)
    area = abs(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))*.5
    assert area*2048**2 > 1.
    assert np.array_equal(vertices[new_mapping][new_faces], vertices[source.faces])


def test_large_scale_collapsed_layout_is_rejected_instead_of_hidden_cpu_rerun(monkeypatch):
    from hi3d_workbench.engines import community_uv
    source = trimesh.creation.icosphere(subdivisions=1)
    mapping, faces, _ = arrays_for(source.faces)
    with pytest.raises(community_uv.LocalUVRepairError, match='安全修复范围'):
        community_uv._repair_local_faces(source.vertices, mapping, faces, np.zeros((len(mapping), 2)),
                                         2048, 2048, 4, lambda *args, **values: None)
    monkeypatch.setattr(community_uv, 'unwrap', lambda *args: (_ for _ in ()).throw(
        community_uv.LocalUVRepairError('unsafe structural layout')))
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', lambda *args, **kw: (_ for _ in ()).throw(
        RuntimeError('test native rejected geometry')))
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', lambda *args: pytest.fail('must not run CPU unwrap'))
    with pytest.raises(community_uv.LocalUVRepairError):
        mesh_uv.ensure_uv(source)


def test_explicit_diagnostics_preserve_raw_arrays_for_replay(monkeypatch, tmp_path):
    from hi3d_workbench.engines.community_uv import _save_diagnostics
    monkeypatch.setenv('HI3D_UV_DIAGNOSTICS_DIR', str(tmp_path))
    raw = {'mapping': np.arange(3), 'faces': np.array([[0, 1, 2]]), 'uv': np.zeros((3, 2))}
    _save_diagnostics(raw, {'postpack_zero_faces': 1})
    with np.load(tmp_path/'community-raw.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['uv'], raw['uv'])
        assert json.loads(str(saved['metadata'].item()))['postpack_zero_faces'] == 1
    assert json.loads((tmp_path/'community-raw.json').read_text())['postpack_zero_faces'] == 1


def test_legacy_native_receives_filtered_faces_without_reindexing_original_vertices(monkeypatch):
    source = hard_box()
    source = trimesh.Trimesh(np.vstack((source.vertices, [[7., 8., 9.]])),
                             np.vstack((source.faces, [[0, 0, 1]])), process=False)
    valid_faces, removed = mesh_uv._valid_source_faces(source.vertices, source.faces)
    assert removed == 1
    captured = []
    def native(value, notify, **options):
        captured.append(value)
        assert np.array_equal(value.vertices, source.vertices)
        assert np.array_equal(value.vertex_normals, source.vertex_normals)
        assert np.array_equal(value.faces, valid_faces)
        mapping, faces, uv = arrays_for(value.faces)
        return mesh_uv._result_from_arrays(value, mapping, faces, uv), 'test_native', mapping
    monkeypatch.setattr(mesh_uv, '_legacy_cumesh', native)
    _, arrays, policy = mesh_uv._legacy_arrays(source, valid_faces, lambda *args, **values: None)
    assert len(captured) == 1 and policy == 'test_native'
    mesh_uv._verify_arrays(source, valid_faces, *arrays)


@pytest.mark.parametrize('failure', [ImportError('missing numba'), OSError('LLVM DLL unavailable'), RuntimeError('broken optional compiler')])
def test_optional_numba_import_failures_use_upstream_torch_and_restore_module_state(monkeypatch, failure):
    from hi3d_workbench.engines.community_uv import _load_pack_api
    previous = SimpleNamespace(original=True)
    monkeypatch.setitem(sys.modules, 'numba', previous)
    events = []
    cli = SimpleNamespace(args=SimpleNamespace(disable_dynamic_vram=False))
    pack = SimpleNamespace(_HAVE_NUMBA_PACK=False)
    def importer(name):
        if name == 'comfy.cli_args': return cli
        if name == 'numba': raise failure
        assert name == 'comfy_extras.mesh3d.uv_unwrap.pack'
        assert sys.modules['numba'] is None
        assert cli.args.disable_dynamic_vram is True
        return pack
    assert _load_pack_api(lambda phase, message, **values: events.append((phase, values)), importer) is pack
    assert sys.modules['numba'] is previous
    assert events[0][1]['packing_backend'] == 'torch'
    assert type(failure).__name__ in events[0][1]['packing_fallback_reason']


def test_optional_numba_failed_pack_import_still_restores_module_state(monkeypatch):
    from hi3d_workbench.engines.community_uv import _load_pack_api
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    cli = SimpleNamespace(args=SimpleNamespace())
    def importer(name):
        if name == 'comfy.cli_args': return cli
        if name == 'numba': raise OSError('numba failed')
        raise ImportError('pack dependency failed independently')
    with pytest.raises(ImportError, match='independently'):
        _load_pack_api(lambda *args, **values: None, importer)
    assert 'numba' not in sys.modules


def test_packing_runtime_reports_numba_cpu_and_actual_torch_device(monkeypatch):
    from hi3d_workbench.engines.community_uv import _packing_runtime
    assert _packing_runtime(SimpleNamespace(_HAVE_NUMBA_PACK=True)) == ('numba', 'cpu', 'cpu')
    class Device:
        type = 'cuda'
        def __str__(self): return 'cuda:0'
    mm = SimpleNamespace(get_torch_device=lambda: Device())
    monkeypatch.setitem(sys.modules, 'comfy', SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    assert _packing_runtime(SimpleNamespace(_HAVE_NUMBA_PACK=False)) == ('torch', 'cuda', 'cuda:0')


@pytest.mark.parametrize('coincident', [False, True])
def test_exact_or_mirrored_uv_reuse_is_split_without_losing_source_occurrences(coincident):
    from hi3d_workbench.engines.community_uv import _repair_local_faces
    if coincident:
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        source = trimesh.Trimesh(np.vstack((vertices, vertices)), [[0, 1, 2], [5, 4, 3]],
                                 vertex_normals=np.array([[0., 0., 1.]]*3+[[0., 0., -1.]]*3), process=False)
    else:
        source = hard_box()
    mapping, faces, uv = arrays_for(source.faces)
    uv[faces[1]] = uv[faces[0]][::-1]
    new_mapping, new_faces, new_uv, report = _repair_local_faces(
        source.vertices, mapping, faces, uv.astype(np.float64)*2048, 2048, 2048, 4,
        lambda *args, **values: None)
    mesh_uv._verify_arrays(source, source.faces, new_mapping, new_faces, new_uv)
    assert np.array_equal(new_mapping[new_faces], mapping[faces])
    assert report['repaired_duplicate_uv_faces'] == 1
    assert report['repaired_zero_uv_faces'] == 0
    assert report['repaired_coincident_uv_faces'] == int(coincident)
    assert report['repaired_unrelated_uv_collisions'] == int(not coincident)
    assert report['source_double_sided_coincident_pairs'] == int(coincident)


def test_legacy_uv_repair_uses_requested_atlas_size_and_preserves_source_attributes(monkeypatch):
    source = hard_box()
    mapping, faces, uv = arrays_for(source.faces)
    uv[faces[0], 1] = .5
    options_seen = []
    def native(value, notify, **options):
        options_seen.append(options)
        return mesh_uv._result_from_arrays(value, mapping, faces, uv), 'native_test', mapping
    monkeypatch.setattr(mesh_uv, '_legacy_cumesh', native)
    result, arrays, policy = mesh_uv._legacy_arrays(source, source.faces, lambda *args, **values: None,
                                                    texture_size=4096, padding=4)
    assert options_seen == [{'texture_size': 4096, 'padding': 4}]
    assert policy == 'native_test_local_repair'
    mesh_uv._verify_arrays(source, source.faces, *arrays)
    assert np.array_equal(result.vertices, source.vertices[arrays[0]])
    assert np.array_equal(result.vertex_normals, source.vertex_normals[arrays[0]])


def test_invalid_native_geometry_is_rejected_before_local_uv_repair(monkeypatch):
    from hi3d_workbench.engines import community_uv
    source = hard_box()
    mapping, faces, uv = arrays_for(source.faces)
    monkeypatch.setattr(mesh_uv, '_legacy_cumesh', lambda value, notify, **options:
        (mesh_uv._result_from_arrays(value, mapping, faces[:-1], uv), 'native_bad', mapping))
    monkeypatch.setattr(community_uv, '_repair_local_faces', lambda *args: pytest.fail('cannot repair bad geometry'))
    with pytest.raises(RuntimeError, match='丢失'):
        mesh_uv._legacy_arrays(source, source.faces, lambda *args, **values: None)


def test_corner_mapping_defers_only_layout_check_until_bounded_repair():
    source = hard_box()
    uv = arrays_for(source.faces)[2]
    uv[source.faces[0], 1] = .5
    with pytest.raises(RuntimeError, match='零面积'):
        mesh_uv._corner_mapped_arrays(source, source.vertices, source.faces, uv)
    arrays = mesh_uv._corner_mapped_arrays(source, source.vertices, source.faces, uv, validate_layout=False)
    mesh_uv._require_complete_faces(source.faces, arrays[0][arrays[1]])
    with pytest.raises(RuntimeError, match='零面积'):
        mesh_uv._verify_arrays(source, source.faces, *arrays)


def test_native_atlas_callbacks_use_distinct_phases_and_do_not_patch_the_class():
    from hi3d_workbench.engines.native_uv import atlas_progress
    class Atlas: pass
    atlas = Atlas()
    events = []
    atlas_progress(atlas, lambda phase, message, **values: events.append((phase, values)))
    callback = atlas._create_progress_callback()
    for category in ['AddMesh', 'ComputeCharts', 'PackCharts', 'BuildOutputMeshes']:
        assert callback(category, 0)
        assert callback(category, 100)
    assert [phase for phase, _ in events] == ['uv_addmesh']*2+['uv_charts']*2+['uv_pack']*2+['uv_assemble']*2
    assert '_create_progress_callback' not in Atlas.__dict__
    assert all(values['device'] == 'cpu' for _, values in events)


def test_native_thin_adapter_keeps_sdk_math_and_reports_gpu_and_cpu_work(monkeypatch):
    from hi3d_workbench.engines.native_uv import unwrap
    operations, events = [], []
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def __getitem__(self, item):
            if isinstance(item, Tensor): item = item.value
            result = self.value[item]
            return Tensor(result) if isinstance(result, np.ndarray) else result
        def __add__(self, value): return Tensor(self.value+value)
        def __sub__(self, value): return Tensor(self.value-value)
        def cpu(self): return self
        @property
        def shape(self): return self.value.shape
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    class Native:
        def remove_degenerate_faces(self): operations.append('clean')
        def compute_charts(self, **options):
            assert options == {}
            operations.append('gpu_cluster')
        def read(self): return Tensor(vertices), Tensor([[0, 1, 2], [1, 3, 2]])
        def read_atlas_charts(self):
            return 2, Tensor([0, 1]), Tensor([0, 1, 2, 1, 3, 2]), Tensor([[0, 1, 2], [3, 4, 5]]), Tensor([0, 3, 6]), Tensor([0, 1, 2])
    class Atlas:
        def __init__(self): self.meshes = []
        def add_mesh(self, vertices, faces):
            self.meshes.append((vertices, faces))
        def compute_charts(self, **options):
            assert options == {'verbose': True}
            self._create_progress_callback()('ComputeCharts', 100)
            operations.append('cpu_parameterize')
        def pack_charts(self, **options):
            assert options == {'resolution': 4096, 'padding': 4, 'texels_per_unit': 0., 'verbose': True}
            self._create_progress_callback()('PackCharts', 100)
            operations.append('cpu_pack')
        def get_mesh(self, index):
            return Tensor([0, 1, 2]), self.meshes[index][1], Tensor(np.array([[.1, .1], [.2, .1], [.1, .2]])+[index*.3, 0.])
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cat=lambda values, dim=0: Tensor(np.concatenate([v.value for v in values], axis=dim))))
    monkeypatch.setitem(sys.modules, 'cumesh', SimpleNamespace(Atlas=Atlas))
    output = unwrap(Native(), notify=lambda phase, message, **values: events.append((phase, values)), texture_size=4096, padding=4)
    assert operations == ['clean', 'gpu_cluster', 'cpu_parameterize', 'cpu_pack']
    assert np.array_equal(output[3].value, [0, 1, 2, 1, 3, 2])
    assert np.array_equal(output[0].value, vertices[output[3].value])
    assert any(phase == 'uv_segment' and values['device'] == 'cuda' for phase, values in events)
    assert any(phase == 'uv_pack' and values['device'] == 'cpu' for phase, values in events)


def test_native_algorithm_source_changes_invalidate_uv_cache(monkeypatch):
    from pathlib import Path
    source = hard_box()
    old, _ = mesh_uv._cache_identity(source, 2048, 'auto', 4)
    read = Path.read_bytes
    def changed(path):
        value = read(path)
        return value+b'\n# next native adapter version\n' if path.name == 'native_uv.py' else value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    new, _ = mesh_uv._cache_identity(source, 2048, 'auto', 4)
    assert old != new


def test_unsafe_native_repair_stops_before_community_or_cpu(monkeypatch):
    from hi3d_workbench.engines import community_uv
    monkeypatch.setattr(community_uv, 'unwrap', lambda *args: pytest.fail('cannot bypass native structural rejection'))
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', lambda *args, **kw: (_ for _ in ()).throw(
        community_uv.LocalUVRepairError('over budget native UV')))
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', lambda *args: pytest.fail('cannot hide unsafe repair in CPU unwrap'))
    with pytest.raises(community_uv.LocalUVRepairError, match='over budget'):
        mesh_uv.ensure_uv(hard_box())


def test_auto_prefers_verified_native_without_loading_community_or_cpu(monkeypatch):
    from hi3d_workbench.engines import community_uv
    calls = []
    def native(mesh, faces, notify, **options):
        calls.append(('cumesh', options))
        arrays = arrays_for(faces)
        return mesh_uv._result_from_arrays(mesh, *arrays), arrays, 'native_verified'
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', native)
    monkeypatch.setattr(community_uv, 'unwrap', lambda *args: pytest.fail('native success must not load community'))
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', lambda *args: pytest.fail('native success must not use CPU safety path'))
    _, policy = mesh_uv.ensure_uv(hard_box(), texture_size=4096)
    assert policy == 'native_verified'
    assert calls == [('cumesh', {'texture_size': 4096, 'padding': 4})]


@pytest.mark.parametrize('failure', [RuntimeError('native changed winding'), ImportError('native wheel missing'), OSError('native DLL unavailable')])
def test_auto_native_rejection_uses_community_once_then_stops(monkeypatch, failure):
    from hi3d_workbench.engines import community_uv
    calls = []
    def native(*args, **kw):
        calls.append('cumesh')
        raise failure
    def community(vertices, faces, *args):
        calls.append('community')
        return arrays_for(faces)
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', native)
    monkeypatch.setattr(community_uv, 'unwrap', community)
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', lambda *args: pytest.fail('community succeeded'))
    _, policy = mesh_uv.ensure_uv(hard_box())
    assert policy == 'community_GPU_PEC_original_indices'
    assert calls == ['cumesh', 'community']


def test_native_geometry_cannot_cross_final_guard_before_community_fallback(monkeypatch):
    from hi3d_workbench.engines import community_uv
    calls = []
    def native(mesh, faces, *args, **kw):
        calls.append('cumesh')
        mapping, triangles, uv = arrays_for(faces)
        triangles[0] = triangles[0, ::-1]
        return None, (mapping, triangles, uv), 'unverified_native'
    def community(vertices, faces, *args):
        calls.append('community')
        return arrays_for(faces)
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', native)
    monkeypatch.setattr(community_uv, 'unwrap', community)
    _, policy = mesh_uv.ensure_uv(hard_box())
    assert calls == ['cumesh', 'community']
    assert policy == 'community_GPU_PEC_original_indices'


@pytest.mark.parametrize('backend, expected', [('cumesh', 'cumesh'), ('community', 'community'), ('comfy_gpu', 'community'), ('cpu', 'cpu')])
def test_explicit_backend_failure_never_changes_selected_backend(monkeypatch, backend, expected):
    from hi3d_workbench.engines import community_uv
    calls = []
    def reject(name):
        def call(*args, **kw):
            calls.append(name)
            raise RuntimeError('explicit '+name+' failed')
        return call
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', reject('cumesh'))
    monkeypatch.setattr(community_uv, 'unwrap', reject('community'))
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', reject('cpu'))
    with pytest.raises(RuntimeError, match='explicit '+expected):
        mesh_uv.ensure_uv(hard_box(), backend=backend)
    assert calls == [expected]


def test_native_rejection_then_community_structural_repair_failure_stops_before_cpu(monkeypatch):
    from hi3d_workbench.engines import community_uv
    calls = []
    def native(*args, **kw):
        calls.append('cumesh')
        raise RuntimeError('native source provenance rejected')
    def community(*args):
        calls.append('community')
        raise community_uv.LocalUVRepairError('community over repair budget')
    monkeypatch.setattr(mesh_uv, '_legacy_arrays', native)
    monkeypatch.setattr(community_uv, 'unwrap', community)
    monkeypatch.setattr(mesh_uv, '_chunked_atlas', lambda *args: pytest.fail('must preserve fail-stop'))
    with pytest.raises(community_uv.LocalUVRepairError, match='over repair budget'):
        mesh_uv.ensure_uv(hard_box())
    assert calls == ['cumesh', 'community']
