"""Gaussian view ordering is checked independently of WebGL/browser acceptance."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_gaussians_sort_back_to_front_after_camera_rotation():
    node=shutil.which('node')
    if not node:pytest.skip('Node is required for the JavaScript math check')
    path=Path(__file__).resolve().parents[1]/'hi3d_workbench/web/splat-sort-worker.js'
    script="""const fs=require('fs'),vm=require('vm');
const results=[],self={postMessage:data=>results.push(Array.from(data.order))};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),{self,Float32Array,Uint32Array});
self.onmessage({data:{type:'init',positions:new Float32Array([2,0,-1,0,0,-5,-2,0,-3]).buffer}});
const matrix=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
self.onmessage({data:{type:'sort',matrix}});
matrix[2]=1;matrix[10]=0;
self.onmessage({data:{type:'sort',matrix}});
console.log(JSON.stringify(results));"""
    result=subprocess.run([node,'-e',script,str(path)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[[1,2,0],[2,1,0]]
