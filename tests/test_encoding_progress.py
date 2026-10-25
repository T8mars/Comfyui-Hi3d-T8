"""CPU-only protocol tests: progress hooks never substitute official math."""
from types import SimpleNamespace

import pytest

from hi3d_workbench.engines.encoding_progress import observe_shape_encoding


class HookModel:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else object()
        self.error = error
        self.hooks = {}
        self.removed = []
        self.calls = []
        self.next_key = 0

    def register_forward_pre_hook(self, callback):
        self.next_key += 1
        key = self.next_key
        self.hooks[key] = callback
        def remove():
            self.removed.append(key)
            self.hooks.pop(key)
        return SimpleNamespace(remove=remove)

    def __call__(self, *args):
        for hook in list(self.hooks.values()):
            assert hook(self, args) is None, 'observer must never replace inputs'
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result


def test_gpu_converter_preparation_is_not_reported_as_cpu_voxelization():
    model=HookModel();trace=[]
    with observe_shape_encoding(SimpleNamespace(models={'shape_slat_encoder':model}),
            lambda stage,**event:trace.append(event),voxel_backend='cuda:validated-build'):
        assert trace[-1]['device']=='cpu' and 'CPU 网格体素化' not in trace[-1]['message']
        model(object())
    assert trace[-1]['phase']=='shape_encoding' and trace[-1]['device']=='cuda'


def test_lazy_checkpoint_loading_restores_cpu_voxelization_before_method_runs():
    from hi3d_workbench.engines.model_loading import LazyModel
    trace=[]
    model=HookModel()
    lazy=LazyModel('shape_slat_encoder',('progress-fixture',),lambda:model,
        lambda stage,**event:trace.append(event))
    with observe_shape_encoding(SimpleNamespace(models={'shape_slat_encoder':lazy}),
            lambda stage,**event:trace.append(event)):
        assert trace[-1]['phase']=='voxelize' and trace[-1]['device']=='cpu'
        lazy(object())
    assert trace[-1]['phase']=='shape_encoding' and trace[-1]['device']=='cuda'
    assert not model.hooks


def test_cpu_preparation_then_actual_encoder_forward_preserves_parameters_and_result():
    model = HookModel()
    pipeline = SimpleNamespace(models={'shape_slat_encoder': model})
    mesh, resolution, sparse, intersections = object(), object(), object(), object()
    trace, calls = [], []
    def emit(stage, **values):
        trace.append((stage, values))
    def official(value, *, size):
        assert value is mesh and size is resolution
        calls.append((value, size))
        assert trace[-1][1]['phase'] == 'voxelize'
        return model(sparse, intersections)
    with observe_shape_encoding(pipeline, emit):
        result = official(mesh, size=resolution)
    assert result is model.result
    assert calls == [(mesh, resolution)]
    assert model.calls == [(sparse, intersections)]
    assert [event[1]['phase'] for event in trace] == ['voxelize', 'shape_encoding']
    assert [event[1]['device'] for event in trace] == ['cpu', 'cuda']
    assert all(event[0] == 'encoding_mesh' for event in trace)
    assert not model.hooks and model.removed == [1]


@pytest.mark.parametrize('failure_phase', ['voxelize', 'encoder'])
def test_failures_always_remove_only_the_temporary_observer(failure_phase):
    error = RuntimeError('official inference failed')
    model = HookModel(error=error if failure_phase == 'encoder' else None)
    preserved = model.register_forward_pre_hook(lambda module, inputs: None)
    pipeline = SimpleNamespace(models={'shape_slat_encoder': model})
    trace = []
    with pytest.raises(RuntimeError, match='official inference failed') as caught:
        with observe_shape_encoding(pipeline, lambda stage, **event: trace.append(event)):
            if failure_phase == 'voxelize':
                raise error
            model(object())
    assert caught.value is error
    assert len(model.hooks) == 1
    assert model.removed == [2]
    assert [event['phase'] for event in trace] == (['voxelize'] if failure_phase == 'voxelize'
                                                 else ['voxelize', 'shape_encoding'])
    preserved.remove()


@pytest.mark.parametrize('models', [{}, {'shape_slat_encoder': object()}])
def test_missing_hook_never_guesses_a_cuda_transition(models):
    trace = []
    result = object()
    with observe_shape_encoding(SimpleNamespace(models=models), lambda stage, **event: trace.append(event)):
        actual = result
    assert actual is result
    assert all(event['device'] == 'cpu' and event['phase'] == 'voxelize' for event in trace)
    assert trace[-1]['encoding_observer_available'] is False


def test_registration_error_does_not_replace_or_skip_the_official_method():
    def unavailable(_hook):
        raise OSError('optional observer unavailable')
    model = SimpleNamespace(register_forward_pre_hook=unavailable)
    trace, calls = [], []
    with observe_shape_encoding(SimpleNamespace(models={'shape_slat_encoder': model}),
                                lambda stage, **event: trace.append(event)):
        calls.append('official method still ran')
    assert calls == ['official method still ran']
    assert all(event['device'] == 'cpu' for event in trace)
    assert 'OSError' in trace[-1]['encoding_observer_reason']


def test_observer_is_not_left_on_the_model_between_independent_encodes():
    model = HookModel()
    pipeline = SimpleNamespace(models={'shape_slat_encoder': model})
    trace = []
    for _ in range(2):
        with observe_shape_encoding(pipeline, lambda stage, **event: trace.append(event)):
            assert model(object()) is model.result
    assert [event['phase'] for event in trace] == ['voxelize', 'shape_encoding']*2
    assert model.removed == [1, 2] and not model.hooks
    model(object())
    assert len(trace) == 4


def test_multiple_forward_calls_do_not_reset_the_same_phase_progress():
    model = HookModel()
    trace = []
    with observe_shape_encoding(SimpleNamespace(models={'shape_slat_encoder': model}),
                                lambda stage, **event: trace.append(event)):
        model(object())
        model(object())
    assert len(model.calls) == 2
    assert [event['phase'] for event in trace] == ['voxelize', 'shape_encoding']
