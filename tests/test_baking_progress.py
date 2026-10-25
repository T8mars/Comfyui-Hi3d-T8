"""CPU-only tests for observing the unchanged official four-inpaint body."""
from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest

from hi3d_workbench.engines.baking_progress import call_baking_with_progress


class FakeCV2:
    INPAINT_TELEA = object()
    def __init__(self, error=None):
        self.calls, self.results = [], []
        self.error = error
    def inpaint(self, source, mask, radius, flags):
        self.calls.append((source, mask, radius, flags))
        if self.error is not None and len(self.calls) == 2:
            raise self.error
        result = np.asarray(source)+radius
        self.results.append(result)
        return result


def pipeline_type(token):
    class Pipeline:
        def __init__(self):
            self.invocations = []
            self.mask = object()
        def postprocess_mesh(self, mesh, pbr_voxel, resolution=1024, texture_size=1024, *, marker=token):
            self.invocations.append((self, mesh, pbr_voxel, resolution, texture_size, marker, token))
            color = cv2.inpaint(pbr_voxel[0], self.mask, 3, cv2.INPAINT_TELEA)
            metallic = cv2.inpaint(pbr_voxel[1], self.mask, 1, cv2.INPAINT_TELEA)
            roughness = cv2.inpaint(pbr_voxel[2], self.mask, 1, cv2.INPAINT_TELEA)
            alpha = cv2.inpaint(pbr_voxel[3], self.mask, 1, cv2.INPAINT_TELEA)
            return color, metallic, roughness, alpha
    return Pipeline


def setup_original(monkeypatch, *, error=None):
    fake = FakeCV2(error)
    token = object()
    pipeline = pipeline_type(token)()
    function = pipeline.postprocess_mesh.__func__
    monkeypatch.setitem(function.__globals__, 'cv2', fake)
    mesh = SimpleNamespace(visual=SimpleNamespace(uv=object()))
    arrays = tuple(np.full((2, 3), index, dtype=np.float32) for index in range(4))
    return fake, pipeline, mesh, arrays, function


def test_four_original_cv2_calls_receive_identical_objects_and_return_identical_pixels(monkeypatch):
    fake, pipeline, mesh, arrays, function = setup_original(monkeypatch)
    original_globals = function.__globals__
    before = dict(original_globals)
    defaults, kwdefaults, closure, code = function.__defaults__, function.__kwdefaults__, function.__closure__, function.__code__
    marker = object()
    expected = pipeline.postprocess_mesh(mesh, arrays, 512, texture_size=4096, marker=marker)
    expected_calls = list(fake.calls)
    fake.calls.clear(); fake.results.clear()
    events = []
    result = call_baking_with_progress(pipeline.postprocess_mesh,
                                       lambda stage, **values: events.append((stage, values)),
                                       mesh, arrays, 512, texture_size=4096, marker=marker)
    assert len(pipeline.invocations) == 2  # baseline once, observed original body once
    assert pipeline.invocations[0] == pipeline.invocations[1]
    assert len(fake.calls) == 4
    for actual, expected_call in zip(fake.calls, expected_calls):
        assert all(actual[i] is expected_call[i] for i in (0, 1, 3))
        assert actual[2] == expected_call[2]
    assert all(np.array_equal(actual, wanted) for actual, wanted in zip(result, expected))
    assert all(actual is returned for actual, returned in zip(result, fake.results))
    assert function.__globals__ is original_globals
    assert original_globals['cv2'] is fake
    assert before.keys() == original_globals.keys()
    assert all(original_globals[key] is value for key, value in before.items())
    assert function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
    assert function.__closure__ is closure and function.__code__ is code
    assert events[0][1]['phase'] == 'texture_raster' and events[0][1]['device'] == 'cuda'
    inpaint = [values for _, values in events if values['phase'] == 'texture_inpaint']
    assert [item['progress_current'] for item in inpaint] == [0, 1, 1, 2, 2, 3, 3, 4]
    assert all(item['device'] == 'cpu' and item['progress_total'] == 4 for item in inpaint)


def test_original_inpaint_exception_is_not_wrapped_and_failed_texture_is_not_completed(monkeypatch):
    error = RuntimeError('original cv2 failure')
    fake, pipeline, mesh, arrays, function = setup_original(monkeypatch, error=error)
    namespace = function.__globals__
    events = []
    with pytest.raises(RuntimeError, match='original cv2 failure') as caught:
        call_baking_with_progress(pipeline.postprocess_mesh, lambda stage, **event: events.append(event), mesh, arrays)
    assert caught.value is error
    assert len(fake.calls) == 2 and len(fake.results) == 1
    assert len(pipeline.invocations) == 1
    assert namespace['cv2'] is fake and function.__globals__ is namespace
    inpaint = [event for event in events if event['phase'] == 'texture_inpaint']
    assert [event['progress_current'] for event in inpaint] == [0, 1, 1]


def test_supported_method_without_existing_uv_does_not_guess_initial_cuda(monkeypatch):
    _, pipeline, mesh, arrays, _ = setup_original(monkeypatch)
    mesh.visual.uv = None
    events = []
    call_baking_with_progress(pipeline.postprocess_mesh, lambda stage, **event: events.append(event), mesh, arrays)
    assert events[0]['phase'] == 'texture_bake' and 'device' not in events[0]
    assert all(event.get('device') != 'cuda' for event in events)
    assert events[-1]['progress_current'] == 4


def test_unsupported_method_runs_the_original_once_with_no_fabricated_cuda_phase():
    marker, result = object(), object()
    class Other:
        def __init__(self): self.calls = []
        def postprocess_mesh(self, mesh, *, value):
            self.calls.append((mesh, value))
            return result
    pipeline = Other()
    mesh = SimpleNamespace(visual=SimpleNamespace(uv=object()))
    events = []
    actual = call_baking_with_progress(pipeline.postprocess_mesh, lambda stage, **event: events.append(event), mesh, value=marker)
    assert actual is result and pipeline.calls == [(mesh, marker)]
    assert len(events) == 1 and events[0]['phase'] == 'texture_bake' and 'device' not in events[0]


def test_decorated_method_falls_back_without_reimplementing_or_unwrapping_it(monkeypatch):
    fake, pipeline, mesh, arrays, _ = setup_original(monkeypatch)
    original = pipeline.postprocess_mesh
    calls = []
    @wraps(original)
    def decorated(*args, **kwargs):
        calls.append('original decorator executed')
        return original(*args, **kwargs)
    events = []
    result = call_baking_with_progress(decorated, lambda stage, **event: events.append(event), mesh, arrays)
    assert calls == ['original decorator executed'] and len(fake.calls) == 4
    assert all(value is returned for value, returned in zip(result, fake.results))
    assert len(events) == 1 and 'device' not in events[0]


def test_keyword_mesh_and_original_defaults_are_preserved(monkeypatch):
    _, pipeline, mesh, arrays, _ = setup_original(monkeypatch)
    events = []
    call_baking_with_progress(pipeline.postprocess_mesh, lambda stage, **event: events.append(event), mesh=mesh, pbr_voxel=arrays)
    invocation = pipeline.invocations[0]
    assert invocation[1] is mesh and invocation[2] is arrays
    assert invocation[3:5] == (1024, 1024)
    assert invocation[5] is invocation[6]
    assert events[0]['phase'] == 'texture_raster'
