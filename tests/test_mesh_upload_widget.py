from pathlib import Path
import shutil
import subprocess

import pytest


def test_mesh_upload_widget_preserves_file_dropdown_and_canvas_serialization(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js required for upload widget regression')
    source=(Path(__file__).parents[1]/'web/hi3d_mesh_upload.js').read_text(encoding='utf-8')
    source='\n'.join(line for line in source.splitlines() if not line.startswith('import '))
    script=tmp_path/'upload-widget.cjs'
    script.write_text('''
const assert=require('node:assert/strict');
let extension,requests=[],messages=[],dirty=0;
const app={registerExtension:value=>extension=value,graph:{setDirtyCanvas:()=>dirty++},extensionManager:{toast:{add:value=>messages.push(value)}}};
let failure=false;
const api={fetchApi:async (url,options)=>{
  requests.push([url,options]);
  return {ok:!failure,status:failure?500:200,text:async ()=>'fixture error',json:async()=>({name:'粗模 (1).glb',subfolder:'hi3d',type:'input'})};
}};
''' + source + '''
class Node {
  constructor(){this.widgets=[{name:'mesh_file',value:'请先上传粗模',options:{values:['请先上传粗模']}}];}
  addWidget(type,name,value,callback,options){const widget={type,name,value,callback,options};this.widgets.push(widget);return widget;}
  computeSize(){return [340,140];} setSize(value){this.size=value;}
}
extension.beforeRegisterNodeDef(Node,{name:'Hi3DLoadMesh'});
(async()=>{
 const legacy={nodes:[
  {id:1,type:'Hi3DBackend',widgets_values:['http://127.0.0.1:8378'],inputs:[],outputs:[]},
  {id:3,type:'Hi3DLoadMesh',widgets_values:['E:/old/coarse.glb'],inputs:[{name:'backend',link:1}]},
  {id:4,type:'Hi3DRefine',widgets_values:[512,25,23,'fixed',true,'sdpa'],inputs:[{name:'backend',type:'HI3D_BACKEND',link:2}],outputs:[{name:'mesh'},{name:'report'},{name:'job_id'}]}
 ],links:[[1,1,0,3,0,'HI3D_BACKEND'],[2,1,0,4,0,'HI3D_BACKEND']]};
 extension.beforeConfigureGraph(legacy);
 assert.equal(legacy.nodes[0].type,'Hi3DModelLoader');assert.equal(legacy.nodes[2].inputs[0].name,'model');
 assert.deepEqual(legacy.nodes[0].widgets_values,['',true,'sdpa',false,false]);
 assert.equal(legacy.nodes[1].widgets_values[0],'请先上传粗模');assert.equal(legacy.links.length,1);
 assert(!JSON.stringify(legacy).includes('http://'));assert.equal(legacy.nodes[2].outputs.length,2);
 messages=[];
 const node=new Node();node.onNodeCreated();
 const button=node.widgets[1];assert.equal(button.options.serialize,false);assert.equal(button.serializeValue(),undefined);
 const file=new File([new Uint8Array([1,2,3])],'粗模.glb');
 assert.equal(node.onDragDrop({dataTransfer:{files:[file]}}),true);
 await new Promise(setImmediate);
 assert.equal(requests.length,1);assert.equal(requests[0][0],'/upload/image');
 const form=requests[0][1].body;
 assert.equal(form.get('subfolder'),'hi3d');assert.equal(form.get('type'),'input');assert.equal(form.get('image').name,'粗模.glb');
 assert.equal(node.widgets[0].value,'hi3d/粗模 (1).glb');
 assert(node.widgets[0].options.values.includes('hi3d/粗模 (1).glb'));
 assert.equal(button.name,'选择文件上传粗模');
 assert.equal(node.onDragDrop({dataTransfer:{files:[new File(['x'],'unsafe.gltf')]}}),false);
 failure=true;node.onDragDrop({dataTransfer:{files:[file]}});await new Promise(setImmediate);
 assert.equal(messages.length,1);assert.match(messages[0].detail,/500/);
 assert.equal(button.name,'选择文件上传粗模');assert(dirty>=4);
 console.log('UPLOAD_WIDGET_TESTS_PASSED');
})().catch(error=>{console.error(error);process.exitCode=1;});
''',encoding='utf-8')
    result=subprocess.run([node,str(script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
