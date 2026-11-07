import pytest
from test_workbench import testroot


def test_real_condition_forward_progress_does_not_change_values_or_leave_hooks():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from hi3d_workbench.studio.xpart_sampling import condition_progress
    geo=torch.nn.Linear(2,2);obj=torch.nn.Linear(2,2);semantic=torch.nn.Linear(2,2)
    holder=SimpleNamespace(geo_encoder=geo,obj_encoder=SimpleNamespace(encoder=obj),seg_feat_encoder=semantic)
    value=torch.ones(2,2);expected=semantic(obj(geo(value)));events=[]
    def emit(stage,message,**details):events.append(details['progress_current'])
    with condition_progress(holder,emit):actual=semantic(obj(geo(value)))
    torch.testing.assert_close(actual,expected)
    assert events==[0,1,1,2,2,3]
    events.clear()
    with pytest.raises(RuntimeError,match='fixture failure'):
        with condition_progress(holder,emit):
            geo(value);raise RuntimeError('fixture failure')
    count=len(events);geo(value);obj(value);semantic(value)
    assert len(events)==count
