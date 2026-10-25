"""Mixed terminal displays must not hide real structured progress events."""
import pytest

from hi3d_workbench.events import event_spans,iter_events


@pytest.mark.parametrize('line,expected',[
    ('HI3D_EVENT {"stage":"sampling","step":0,"steps":12}\n',[{'stage':'sampling','step':0,'steps':12}]),
    ('\rSampling 50%|### HI3D_EVENT {"stage":"sampling","step":6} 50%|###\x1b[0m\n',[{'stage':'sampling','step':6}]),
    ('display HI3D_EVENT [] HI3D_EVENT {"stage":"uv","progress_current":3} trailing',[{'stage':'uv','progress_current':3}]),
    ('HI3D_EVENT {bad JSON HI3D_EVENT {"stage":"valid"} tail',[{'stage':'valid'}]),
    ('HI3D_EVENT {"stage":"one"}xx HI3D_EVENT {"stage":"two"}',[{'stage':'one'},{'stage':'two'}]),
    ('HI3D_EVENT {"stage":"good","message":"literal HI3D_EVENT inside JSON"}',[{'stage':'good','message':'literal HI3D_EVENT inside JSON'}]),
    ('HI3D_EVENT {"stage":42}',[]),('HI3D_EVENT {"stage":"  "}',[]),
    ('HI3D_EVENT {"stage":"bad","step":NaN}',[]),
    ('HI3D_EVENT {"stage":"bad","metrics":{"value":Infinity}}',[]),
    ('HI3D_EVENT {"stage":"bad","metrics":[1e999]}',[]),
    ('HI3D_EVENT {"stage":"bad","step":-Infinity} HI3D_EVENT {"stage":"recovered"}',[{'stage':'recovered'}]),
    ('normal log without any structured event',[]),
])
def test_inline_event_parser_handles_terminal_noise_and_bad_payloads(line,expected):
    assert list(iter_events(line))==expected


def test_event_spans_leave_surrounding_display_and_json_strings_intact():
    line='prefix\rHI3D_EVENT {"stage":"uv","message":"brace } in string"}suffix\x1b[0m'
    spans=list(event_spans(line))
    assert len(spans)==1
    start,end,event=spans[0]
    assert line[:start]=='prefix\r' and line[end:]=='suffix\x1b[0m'
    assert event['message']=='brace } in string'


def test_bad_nested_json_does_not_break_following_events():
    line='HI3D_EVENT '+('['*1100)+' HI3D_EVENT {"stage":"next"}'
    assert list(iter_events(line))==[{'stage':'next'}]
