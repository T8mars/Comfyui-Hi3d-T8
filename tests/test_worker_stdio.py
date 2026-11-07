import json
import os
from pathlib import Path
import subprocess
import sys

from test_workbench import testroot


def test_worker_wire_encoding_ignores_inherited_legacy_codepage():
    code = '''import sys,json
for stream in (sys.stdin,sys.stdout,sys.stderr):
    stream.reconfigure(encoding='gbk')
from hi3d_workbench.common import configure_stdio
configure_stdio()
value=json.loads(sys.stdin.readline())
print(json.dumps({'message':value['message'],'path':value['path']},ensure_ascii=False))
sys.stderr.write(value['message'])
'''
    value = {'message': '中文进度 🎬', 'path': 'E:/角色 素材/模型.glb'}
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
        input=(json.dumps(value, ensure_ascii=False) + '\n').encode('utf-8'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONIOENCODING': 'gbk'}, timeout=15)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert json.loads(result.stdout.decode('utf-8')) == value
    assert result.stderr.decode('utf-8') == value['message']
