"""Actual abrupt-parent termination of a dedicated, harmless tool fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows local bundle lifecycle')
def test_native_child_dies_when_its_wrapper_crashes():
    project = Path(__file__).resolve().parents[1]
    script = '''import json,subprocess,sys,psutil
from hi3d_workbench.studio.ownership import ProcessContainment
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)','hi3d-owned-process-fixture'])
guard=ProcessContainment(child)
print(json.dumps({'pid':child.pid,'created':psutil.Process(child.pid).create_time()}),flush=True)
child.wait()
'''
    wrapper = subprocess.Popen([sys.executable, '-u', '-c', script], cwd=project,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    created = None
    try:
        line = wrapper.stdout.readline()
        assert line, wrapper.stderr.read()
        info = json.loads(line)
        child = psutil.Process(info['pid'])
        created = info['created']
        assert child.create_time() == created and child.is_running()
        # This wrapper was created by this exact test. No user service or model
        # process is searched for or stopped by the lifecycle check.
        wrapper.terminate()
        wrapper.wait(timeout=10)
        child.wait(timeout=10)
        assert not child.is_running()
    finally:
        if wrapper.poll() is None:
            wrapper.kill(); wrapper.wait(timeout=10)
        wrapper.stdout.close(); wrapper.stderr.close()
        if child is not None:
            try:
                if child.is_running() and child.create_time() == created:
                    assert 'hi3d-owned-process-fixture' in child.cmdline()
                    child.kill(); child.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
