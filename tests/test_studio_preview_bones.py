"""Exercise preview comparison with the shipped Three.js bone/world transforms."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_comparison_rebinds_bone_tracking_and_keeps_joint_edits(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for preview bone regression')
    web = Path(__file__).parents[1] / 'hi3d_workbench/web'
    source = (web / 'studio.js').read_text(encoding='utf-8')
    bones = source[source.index('function clearBones('):source.index('function selectedBone(')]
    compare_start = source.index('function setComparing(') if 'function setComparing(' in source else source.index("$('compare').onclick=")
    compare = source[compare_start:source.index("$('reset-camera').onclick=")]
    script = tmp_path / 'studio-preview-bones.mjs'
    script.write_text('import * as THREE from ' + json.dumps((web / 'vendor/three/three.module.js').as_uri()) + ';\n' + r"""
import assert from 'node:assert/strict';
const fields={};
const $=id=>fields[id] ||= {value:'',textContent:'',children:[],
  classList:{toggle(){}},replaceChildren(...items){this.children=items;},append(item){this.children.push(item);}};
function Option(text,value){return {text,value};}
let sourceInfo={bones:[{index:0,node:7,name:'hip',position:[0,1,0],parent:-1},
  {index:1,node:8,name:'knee',position:[0,0,0],parent:0}]};
let model=null,gltf=null,comparing=false,loadedUrl=null,showBones=true;
let sourceUrl='/original.glb',resultUrl='/result.glb',boneEdits=new Map();
let boneGroup=new THREE.Group(),boneMarkers=[],boneLines=[],boneTargets=[];
const transform={detach(){}},availability=()=>{};
$('action').value='rig';
let currentRoot;
async function showModel(url){
  // A new GLTF scene owns new node instances. The old scene is disposed.
  boneTargets=[];model=new THREE.Group();currentRoot=new THREE.Bone();
  const knee=new THREE.Bone();knee.position.set(0,-1,0);currentRoot.add(knee);
  currentRoot.position.set(5,6,7);model.add(currentRoot);
  const node=url===resultUrl?7:70;
  gltf={parser:{associations:new Map([[currentRoot,{nodes:node}],[knee,{nodes:node+1}]])}};
  boneGroup.visible=showBones&&!comparing;loadedUrl=url;return true;
}
""" + bones + '\n' + compare + r"""
await showModel(resultUrl);drawBones(sourceInfo.bones);
const firstRoot=currentRoot;
await $('compare').onclick();await $('compare').onclick();
assert.notEqual(currentRoot,firstRoot);
assert.ok(boneTargets[0]===currentRoot,'returning from comparison must track the new GLTF nodes');
currentRoot.position.y=8;updateAnimatedBones();
assert.deepEqual(boneMarkers[0].position.toArray(),[5,8,7]);
assert.deepEqual(boneMarkers[1].position.toArray(),[5,7,7]);
assert.deepEqual([...boneLines[0].line.geometry.attributes.position.array],[5,8,7,5,7,7]);
assert.equal(boneGroup.visible,true);
$('action').value='edit_rig';boneEdits.set(0,{index:0,position:[8,9,10]});
await $('compare').onclick();await $('compare').onclick();
assert.deepEqual(boneMarkers[0].position.toArray(),[8,9,10],'comparison must not discard a pending joint edit');
assert.equal(boneEdits.size,1);
assert.equal($('compare').textContent,'查看原模型');
""", encoding='utf-8')
    result = subprocess.run([node, '--experimental-default-type=module', str(script)],
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
