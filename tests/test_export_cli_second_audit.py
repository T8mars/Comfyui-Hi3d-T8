"""New export transaction and actual CLI port regressions; no model execution."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
import requests
import trimesh
import numpy as np
from PIL import Image
from test_server_operations import isolated_server


@pytest.mark.parametrize('failure',['mesh','dimensions'])
@pytest.mark.parametrize('source_kind',['coarse','textured'])
def test_failed_export_removes_its_partial_copy_and_preserves_original(isolated_server,monkeypatch,failure,source_kind):
    server,client=isolated_server
    folder=server.manager.root/'export-failure';folder.mkdir()
    source=folder/(source_kind+'.glb')
    if source_kind=='textured':
        mesh=trimesh.creation.box(extents=[1,2,4])
        uv=(np.asarray(mesh.vertices)[:,:2]+[.5,1])/[1,2]
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new('RGBA',(2,2),(180,70,120,255)),
            metallicRoughnessTexture=Image.new('RGB',(2,2),(0,100,190)))
        mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
        mesh.export(source)
    else:source.write_bytes(trimesh.creation.box().export(file_type='glb'))
    original=source.read_bytes()
    exports=server.DATA/'exports';retained=exports/'retained';retained.mkdir(parents=True)
    (retained/'model.glb').write_bytes(b'existing exported model')
    if failure=='mesh':
        def fail(mesh,target,height_mm):
            target.write_bytes(b'incomplete mesh')
            raise ValueError('export write failed')
        if source_kind=='textured':
            from hi3d_workbench import asset_pipeline
            monkeypatch.setattr(asset_pipeline,'export_textured_glb',fail)
        else:monkeypatch.setattr(server,'export_mesh',fail)
    else:
        write=server.atomic_json
        def fail(path,value):
            if path.name=='dimensions.json':raise ValueError('dimensions write failed')
            return write(path,value)
        monkeypatch.setattr(server,'atomic_json',fail)
    response=client.post('/api/jobs/export-failure/export',json={'source':source_kind,'format':'glb','height_mm':80})
    assert response.status_code==400
    assert source.read_bytes()==original
    assert list(exports.iterdir())==[retained]
    assert (retained/'model.glb').read_bytes()==b'existing exported model'


def test_cli_override_is_reported_as_the_actual_listening_port(tmp_path):
    # Reserve a free loopback port; neither the user service nor its data is used.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    config=tmp_path/'config.local.json'
    config.write_text(json.dumps({'port':8378}),encoding='utf-8')
    env={**os.environ,'HI3D_WORKBENCH_ROOT':str(tmp_path)}
    process=subprocess.Popen([sys.executable,'-m','hi3d_workbench','--port',str(port)],cwd=Path(__file__).resolve().parents[1],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        health=None
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail(process.communicate()[1].decode('utf-8',errors='replace'))
            try:
                response=requests.get(f'http://127.0.0.1:{port}/api/health',timeout=.25)
                if response.status_code==200:health=response.json();break
            except requests.RequestException:pass
            time.sleep(.05)
        assert health is not None,'Disposable CLI service failed to start'
        assert health['port']==port
        assert requests.get(f'http://127.0.0.1:{port}/api/health',headers={'host':'127.0.0.1:12345'},timeout=2).json()['port']==port
        assert json.loads(config.read_text(encoding='utf-8'))=={'port':8378},'CLI overrides must not rewrite user config'
    finally:
        process.terminate()
        try:process.communicate(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.communicate(timeout=5)
