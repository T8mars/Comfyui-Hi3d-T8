import io
import json

from test_workbench import testroot
from hi3d_workbench.studio.native import stream_output


def test_same_percent_with_new_native_stage_is_still_reported(tmp_path):
    import io
    events=[]
    stream_output(io.StringIO('72% done. Projecting vertices\n72% done. Cleaning up triangles\n'),
        tmp_path,report=lambda stage,message,**fields:events.append((message,fields)),output=lambda value:None)
    assert [message for message,fields in events]==['将顶点投影回原表面','整理残余三角面']
    assert all(fields['progress_current']==72 for message,fields in events)


def test_verbose_native_stream_keeps_progress_and_errors_without_console_flood(tmp_path):
    lines = ['0% starting\n'] + ['debug loop\n'] * 10000 + ['25.5% projecting\n'] * 100
    lines += ['ERROR: native diagnostic\n', '100% done\n']
    ticks = iter(index / 10000 for index in range(len(lines) + 10))
    printed, progress = [], []
    result = stream_output(io.StringIO(''.join(lines)), tmp_path,
        clock=lambda: next(ticks), output=printed.append, report=lambda *a, **kw: progress.append(kw))
    assert result['read_records'] == len(lines)
    assert len(printed) < 30 and 'ERROR: native diagnostic' in printed
    assert [event['progress_current'] for event in progress] == [0, 25.5, 100]
    saved = json.loads((tmp_path / 'native-output.json').read_text(encoding='utf-8'))
    assert saved['tail'][-1] == '100% done' and len(saved['tail']) <= 200


def test_long_native_line_is_drained_in_bounded_records(tmp_path):
    printed = []
    data = 'x' * (2 * 1024 * 1024) + '\n100% done\n'
    stream_output(io.StringIO(data), tmp_path, clock=lambda: 0,
                  output=printed.append, report=lambda *a, **kw: None)
    assert max(map(len, printed)) <= 500
    assert (tmp_path / 'native-output.json').stat().st_size < 150000
