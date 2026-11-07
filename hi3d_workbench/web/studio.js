import * as THREE from 'three';
import {GLTFLoader} from './vendor/three/loaders/GLTFLoader.js';
import {RGBELoader} from './vendor/three/loaders/RGBELoader.js';
import {OrbitControls} from './vendor/three/controls/OrbitControls.js';
import {TransformControls} from './vendor/three/controls/TransformControls.js';

const root = document.querySelector('#studio');
root.innerHTML = await (await fetch('/static/studio.html')).text();
const $ = id => root.querySelector('#studio-' + id);
let token = document.querySelector('meta[name="hi3d-token"]').content;
let state = null, health = null, currentJob = null, busy = false, motionAsset = null, motionValid=false;
let sourceInfo = null, sourceRevision = 0, selectionRevision = 0, resultUrl = null, loadedUrl = null, activeClip = null;
let sourceUrl = null, comparing = false, wireframe = false, showBones = true, playing = false;
let partEdits = [], boneEdits = new Map(), constraints = [], facePaint = false, polling = false;
let partAttachments = new Map();
const activeStates = new Set(['queued','running','cancelling']);
const actions = {simplify:'保材质减面',retopology:'四边面重拓扑',segment:'AI 自动拆件',edit_parts:'部件修正',complete_parts:'部件补全',rig:'自动骨骼与蒙皮',edit_rig:'关节修正',attach_parts:'部件骨骼固定',retarget:'动作导入与重定向',motion:'AI 单人动作'};
const presets = {
  punch:'A person performs a controlled punch and returns to a fighting stance.',
  kick:'A person in a fighting stance performs a front kick and returns to the starting stance.',
  dodge:'A person in a fighting stance ducks and sidesteps to dodge an attack, then returns to the starting stance.',
  walk:'A person walks forward at a steady pace.',
  run:'A person runs forward at a steady pace.',
  wave:'A person stands still and waves with the right hand.'
};
const presetSeeds = {punch:23,kick:42,dodge:23,walk:23,run:23,wave:23};
function message(text, error=false) { $('model-message').textContent=text; $('model-message').style.color=error?'var(--danger)':''; }
async function api(path, options={}, retry=true) {
  const response=await fetch(path,{...options,headers:{'x-hi3d-token':token,...options.headers}});
  if(response.status===403 && retry){token=(await (await fetch('/api/session')).json()).token;return api(path,options,false);}
  if(!response.ok){let text=await response.text();try{text=JSON.parse(text).detail;}catch{}throw new Error(typeof text==='string'?text:JSON.stringify(text));}
  return response.json();
}
const post=(path,body)=>api(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
function selectedFeature(){const action=$('action').value;return action==='rig'?'rig_'+$('rig-engine').value:['edit_rig','attach_parts'].includes(action)?'retarget':action==='edit_parts'?'simplify':action;}
function availability(){
  const action=$('action').value, feature=state?.features?.[selectedFeature()];
  const working=Boolean(health?.active || health?.operation || health?.recovery_blocked || activeStates.has(currentJob?.state));
  const needsRig=['edit_rig','attach_parts','retarget','motion'].includes(action),missingRig=needsRig&&!sourceInfo?.bones?.length;
  const missingAttachment=action==='attach_parts'&&(!sourceInfo?.parts?.parts?.length||!partAttachments.size);
  $('run').disabled=busy||working||state?.busy||!$('source').value||!sourceInfo||!feature?.ready||missingRig||missingAttachment||(action==='retarget'&&(!motionAsset||!motionValid));
  $('run').textContent=busy?'正在提交…':`开始${actions[action]} →`;
  $('download').disabled=busy||working||state?.busy||!(feature?.components?.length);
  $('verify').disabled=busy||working||state?.busy;
  $('download').textContent=state?.busy?'正在准备组件…':'准备所需模型';
  $('readiness').textContent=state?.busy?(state.download?.message||'模型组件正在准备，完成后可运行'):!feature?'正在读取组件状态…':feature.ready?'✓ 所需模型与环境已就绪':!feature.runtime_ready?'所需运行环境未就绪，请安装 / 更新资产加工环境。':`需准备模型：${feature.missing_components.join('、')}`;
  if(!state?.busy&&feature?.ready){if(missingRig)$('readiness').textContent='请先选择已绑定骨架的角色，或执行「自动骨骼与蒙皮」。';else if(action==='retarget'&&!motionValid)$('readiness').textContent=motionAsset?'动作解析失败，请检查骨架和坐标轴设置，或重新上传。':'请上传 BVH、FBX 或带动作的 GLB。';}
  if(!missingRig&&missingAttachment)$('readiness').textContent='请先保存拆件结果、绑定骨架，再添加部件与骨骼的对应关系。';
  for(const id of ['attach-part','attach-bone','attach-ground','add-attachment'])$(id).disabled=busy||working||missingRig||!sourceInfo?.parts?.parts?.length;
  for(const button of $('attachment-list').querySelectorAll('button'))button.disabled=busy||working;
  const editDisabled=busy||working||comparing||!model||!sourceInfo?.parts?.parts?.length;
  for(const control of [...$('parts-list').querySelectorAll('input,button'),$('add-part'),$('merge-parts'),$('paint-faces'),$('paint-target')])control.disabled=editDisabled;
  $('mesh-file').disabled=busy;
  $('motion-file').disabled=busy||working;
  $('motion-up').disabled=busy||working||!motionAsset||motionAsset.filename!=='motion.bvh';
  $('motion-armature').disabled=busy||working||!(motionAsset?.armatures?.length>1);
  $('motion-source-clip').disabled=busy||working||!motionValid;
  $('cancel').disabled=!currentJob||!activeStates.has(currentJob.state);
}
function changeAction(){
  const action=$('action').value;
  root.querySelectorAll('[data-studio-actions]').forEach(element=>{element.hidden=!element.dataset.studioActions.split(' ').includes(action);});
  $('count-field').hidden=$('budget-mode').value!=='count';$('ratio-field').hidden=$('budget-mode').value!=='ratio';
  changeRetopology();
  transform.setMode('translate');
  if(action!=='edit_rig')transform.detach();
  else{
    resetMotionCarrier();
    playing=false;mixer?.stopAllAction();$('play').textContent='播放';
    for(const bone of sourceInfo?.bones||[])if(boneMarkers[bone.index])boneMarkers[bone.index].position.fromArray(boneEdits.get(bone.index)?.position||bone.position);
    updateBoneLines();
  }
  for(const id of ['clip','play','timeline'])$(id).disabled=action==='edit_rig'||!clips.length;
  for(const select of $('mapping').querySelectorAll('select[data-side=source]'))select.disabled=action==='motion';
  availability();
}
function changeRetopology(){const quad=$('retopology-engine').value==='quadriflow';$('autoremesher-options').hidden=quad;$('quadriflow-options').hidden=!quad;}
$('retopology-engine').onchange=changeRetopology;
$('action').onchange=changeAction;$('budget-mode').onchange=changeAction;$('rig-engine').onchange=availability;
$('motion-preset').onchange=()=>{const preset=$('motion-preset').value;$('prompt').value=presets[preset];$('seed').value=presetSeeds[preset];};

const viewport=$('viewport'),scene=new THREE.Scene();
scene.background=new THREE.Color('#f9f2f7');
const camera=new THREE.PerspectiveCamera(40,1,.001,10000),renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
viewport.append(renderer.domElement);
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;
const transform=new TransformControls(camera,renderer.domElement);scene.add(transform.getHelper());
transform.addEventListener('dragging-changed',event=>{orbit.enabled=!event.value;});
scene.add(new THREE.HemisphereLight(0xffffff,0x8b7395,2));
const key=new THREE.DirectionalLight(0xfff6ee,3);key.position.set(3,6,5);scene.add(key);
const fill=new THREE.DirectionalLight(0xd7e6ff,1.2);fill.position.set(-4,2,-3);scene.add(fill);
const pmrem=new THREE.PMREMGenerator(renderer);
new RGBELoader().load('/static/environments/studio_small_09.hdr',texture=>{const environment=pmrem.fromEquirectangular(texture);scene.environment=environment.texture;texture.dispose();},undefined,()=>{});
const loader=new GLTFLoader(),raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
let model=null,gltf=null,mixer=null,clips=[],boneGroup=new THREE.Group(),boneMarkers=[],boneLines=[],boneTargets=[],paintPreview=null,carrierBase=new THREE.Matrix4();
scene.add(boneGroup);
const clock=new THREE.Clock();
function resize(){const width=viewport.clientWidth,height=viewport.clientHeight;if(width&&height){renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}}
new ResizeObserver(resize).observe(viewport);
function resetCamera(){if(!model)return;const box=new THREE.Box3().setFromObject(model),size=box.getSize(new THREE.Vector3()).length(),center=box.getCenter(new THREE.Vector3());orbit.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(size*.9,size*.4,size*1.3));camera.near=Math.max(size/10000,.00001);camera.far=Math.max(size*100,100);camera.updateProjectionMatrix();orbit.update();}
function releaseScene(value){
  const geometries=new Set(),materials=new Set(),textures=new Set();
  value?.traverse(object=>{if(object.geometry)geometries.add(object.geometry);for(const material of [].concat(object.material||[])){materials.add(material);for(const item of Object.values(material))if(item?.isTexture)textures.add(item);}});
  for(const geometry of geometries)geometry.dispose();for(const material of materials)material.dispose();for(const texture of textures){texture.dispose();texture.source?.data?.close?.();}
}
function clearPaintPreview(){if(paintPreview){scene.remove(paintPreview);releaseScene(paintPreview);paintPreview=null;}}
function disposeModel(){clearPaintPreview();if(model){mixer?.stopAllAction();mixer?.uncacheRoot(model);releaseScene(model);scene.remove(model);}model=null;gltf=null;mixer=null;activeClip=null;boneTargets=[];transform.detach();}
async function showModel(url,reset=true){
  if(!url)return false;if(url===loadedUrl)return true;
  const revision=++sourceRevision;$('view-empty').hidden=false;$('view-empty').textContent='正在加载三维预览…';
  try{
    const loaded=await loader.loadAsync(url);
    if(revision!==sourceRevision){releaseScene(loaded.scene);return false;}
    disposeModel();gltf=loaded;model=loaded.scene;scene.add(model);loadedUrl=url;
    model.updateMatrix();carrierBase.copy(model.matrix);
    model.traverse(object=>{if(object.isMesh){object.userData.originalMaterials=object.material;for(const material of [].concat(object.material))material.wireframe=wireframe;}});
    clips=loaded.animations;mixer=clips.length?new THREE.AnimationMixer(model):null;
    if(mixer){
      mixer.addEventListener('finished',()=>{playing=false;$('play').textContent='播放';});
      mixer.addEventListener('loop',event=>{
        const metadata=gltf?.parser?.json?.extras?.studio_motion;
        if(!metadata?.loop||metadata.root_motion!=='original'||!Array.isArray(metadata.cycle_transform))return;
        let count=Math.abs(event.loopDelta),step=new THREE.Matrix4().fromArray(metadata.cycle_transform),result=new THREE.Matrix4();
        if(!Number.isSafeInteger(count))return;if(event.loopDelta<0)step.invert();
        while(count){if(count%2)result.multiply(step);step.multiply(step.clone());count=Math.floor(count/2);}
        model.updateMatrix();model.matrix.multiply(result);model.matrix.decompose(model.position,model.quaternion,model.scale);
      });
    }
    $('clip').replaceChildren(...clips.map((clip,index)=>new Option(clip.name||`动作 ${index+1}`,String(index))));
    const editing=$('action').value==='edit_rig';
    $('clip').disabled=editing||!clips.length;$('play').disabled=editing||!clips.length;$('timeline').disabled=editing||!clips.length;
    if(clips.length&&!editing)selectClip(0);else{playing=false;$('play').textContent='播放';$('time').textContent='0.00 s';}
    $('view-empty').hidden=true;resize();if(reset)resetCamera();
    boneGroup.visible=showBones&&!comparing;
    return true;
  }catch(error){if(revision===sourceRevision){$('view-empty').hidden=false;$('view-empty').textContent='三维预览失败：'+error.message;message(error.message,true);}return false;}
}
function resetMotionCarrier(){if(model)carrierBase.decompose(model.position,model.quaternion,model.scale);}
function selectClip(index){if(!mixer||!clips[index])return;mixer.stopAllAction();resetMotionCarrier();activeClip=mixer.clipAction(clips[index]);const looping=Boolean(gltf?.parser?.json?.extras?.studio_motion?.loop);activeClip.setLoop(looping?THREE.LoopRepeat:THREE.LoopOnce,looping?Infinity:1);activeClip.clampWhenFinished=true;activeClip.reset().play();activeClip.paused=false;playing=false;$('play').textContent='播放';$('timeline').max=String(clips[index].duration);$('timeline').value='0';mixer.setTime(0);activeClip.paused=true;updateAnimatedBones();}
$('clip').onchange=()=>selectClip(+$('clip').value);
$('play').onclick=()=>{if(activeClip){if(!playing&&activeClip.time>=activeClip.getClip().duration-1e-6){resetMotionCarrier();activeClip.reset().play();}playing=!playing;activeClip.paused=!playing;$('play').textContent=playing?'暂停':'播放';}};
$('timeline').oninput=()=>{if(mixer){playing=false;resetMotionCarrier();activeClip.enabled=true;activeClip.paused=false;mixer.setTime(+$('timeline').value);activeClip.paused=true;$('time').textContent=(+$('timeline').value).toFixed(2)+' s';$('play').textContent='播放';}};
$('wireframe').onclick=()=>{wireframe=!wireframe;$('wireframe').classList.toggle('active',wireframe);model?.traverse(object=>{if(object.isMesh)for(const material of [].concat(object.material))material.wireframe=wireframe;});};
$('bones').onclick=()=>{showBones=!showBones;boneGroup.visible=showBones&&!comparing;$('bones').classList.toggle('active',showBones);};
$('compare').onclick=async()=>{comparing=!comparing;loadedUrl=null;$('compare').classList.toggle('active',comparing);await showModel(comparing?sourceUrl:resultUrl||sourceUrl,false);};
$('reset-camera').onclick=resetCamera;
function animate(){requestAnimationFrame(animate);const dt=Math.min(clock.getDelta(),.1);if(!root.classList.contains('active'))return;resize();orbit.update();if(mixer&&playing){mixer.update(dt);$('timeline').value=String(activeClip.time);$('time').textContent=activeClip.time.toFixed(2)+' s';}updateAnimatedBones();renderer.render(scene,camera);}
animate();

function clearBones(){transform.detach();for(const child of [...boneGroup.children]){boneGroup.remove(child);child.geometry?.dispose();child.material?.dispose();}boneMarkers=[];boneLines=[];boneTargets=[];}
function drawBones(bones){
  clearBones();$('bone-select').replaceChildren(new Option('选择一个关节',''));
  const span=model?new THREE.Box3().setFromObject(model).getSize(new THREE.Vector3()).length():1;
  for(const bone of bones){
    const marker=new THREE.Mesh(new THREE.SphereGeometry(Math.max(span*.006,.0001),10,8),new THREE.MeshBasicMaterial({color:0xc72e76,depthTest:false,transparent:true,opacity:.9}));
    marker.position.fromArray(bone.position);marker.userData.boneIndex=bone.index;marker.renderOrder=10;boneGroup.add(marker);boneMarkers[bone.index]=marker;
    for(const [object,association] of gltf?.parser?.associations||[])if(association.nodes===bone.node&&object.isObject3D){boneTargets[bone.index]=object;break;}
    $('bone-select').append(new Option(bone.name,String(bone.index)));
  }
  for(const bone of bones){if(bone.parent<0)continue;const geometry=new THREE.BufferGeometry().setFromPoints([boneMarkers[bone.parent].position,boneMarkers[bone.index].position]);const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0xb74b81,depthTest:false}));line.renderOrder=9;boneGroup.add(line);boneLines.push({line,parent:bone.parent,index:bone.index});}
  $('bone-select').disabled=!bones.length;boneGroup.visible=showBones&&!comparing;
}
function updateAnimatedBones(){
  if(!model||comparing||$('action').value==='edit_rig'||boneEdits.size)return;
  model.updateMatrixWorld(true);
  for(let index=0;index<boneTargets.length;index++)if(boneTargets[index]&&boneMarkers[index])boneTargets[index].getWorldPosition(boneMarkers[index].position);
  updateBoneLines();
}
function updateBoneLines(){for(const item of boneLines){const data=item.line.geometry.attributes.position;data.setXYZ(0,...boneMarkers[item.parent].position.toArray());data.setXYZ(1,...boneMarkers[item.index].position.toArray());data.needsUpdate=true;}}
function selectedBone(){const value=$('bone-select').value;return value===''?null:+value;}
function selectBone(index){
  $('bone-select').value=index===null?'':String(index);
  for(const axis of ['x','y','z'])$('bone-'+axis).disabled=index===null;
  if(index===null){transform.detach();return;}
  const marker=boneMarkers[index];if(!marker)return;
  for(const axis of ['x','y','z'])$('bone-'+axis).value=marker.position[axis].toFixed(5);
  if($('action').value==='edit_rig')transform.attach(marker);
  $('bone-info').textContent=`${sourceInfo?.bones?.[index]?.name||'关节'} · 修改关节后请保存，蒙皮可按需重新计算。`;
}
$('bone-select').onchange=()=>selectBone(selectedBone());
function rememberBone(){const index=selectedBone();if(index===null)return;const marker=boneMarkers[index];boneEdits.set(index,{index,position:marker.position.toArray()});updateBoneLines();selectBone(index);}
transform.addEventListener('objectChange',rememberBone);
for(const axis of ['x','y','z'])$('bone-'+axis).onchange=()=>{const index=selectedBone();if(index===null)return;const value=+$('bone-'+axis).value;if(Number.isFinite(value)){boneMarkers[index].position[axis]=value;rememberBone();}};

function partOf(object){let item=object;while(item){if(Number.isInteger(item.userData?.studio_part))return item.userData.studio_part;item=item.parent;}return null;}
function partObjects(id){const values=[];model?.traverse(object=>{if(object.isMesh&&partOf(object)===id)values.push(object);});return values;}
function renderParts(parts){
  const list=$('parts-list');list.replaceChildren();$('paint-target').replaceChildren();
  const values=parts?.parts||[];
  $('add-part').disabled=!values.length;$('merge-parts').disabled=!values.length;$('paint-faces').disabled=!values.length;$('paint-target').disabled=!values.length;
  for(const part of values){
    const row=document.createElement('div');row.className='studio-part-row';row.dataset.part=String(part.id);
    const selected=document.createElement('input');selected.type='checkbox';selected.setAttribute('aria-label','选中 '+part.name);selected.dataset.part=String(part.id);
    const name=document.createElement('input');name.type='text';name.value=part.name;name.maxLength=80;name.onchange=()=>{part.name=name.value;partEdits.push({operation:'rename',part:part.id,name:name.value});editCount();};
    const visible=document.createElement('button');visible.textContent='◉';visible.title='显示 / 隐藏部件';visible.onclick=()=>{const objects=partObjects(part.id);const show=!objects.some(object=>object.visible);for(const object of objects)object.visible=show;visible.textContent=show?'◉':'○';};
    row.append(selected,name,visible);list.append(row);$('paint-target').append(new Option(part.name,String(part.id)));
  }
  if(!values.length)list.textContent='自动拆件后可编辑部件。';
  renderAttachments();
}
function renderAttachments(){
  $('attach-part').replaceChildren(...(sourceInfo?.parts?.parts||[]).map(part=>new Option(part.name,String(part.id))));
  $('attach-bone').replaceChildren(...(sourceInfo?.bones||[]).map(bone=>new Option(bone.name,String(bone.index))));
  const list=$('attachment-list');list.replaceChildren();
  for(const [part,assignment] of partAttachments){const bone=assignment.bone;
    const row=document.createElement('p'),label=document.createElement('span'),remove=document.createElement('button');
    label.textContent=`${sourceInfo?.parts?.parts.find(item=>item.id===part)?.name||part} → ${sourceInfo?.bones?.find(item=>item.index===bone)?.name||bone}${assignment.avoid_ground?' · 避免穿地':''} `;
    remove.textContent='移除';remove.type='button';remove.onclick=()=>{partAttachments.delete(part);renderAttachments();availability();};
    row.append(label,remove);list.append(row);
  }
}
$('add-attachment').onclick=()=>{const part=+$('attach-part').value,bone=+$('attach-bone').value;
  if(!sourceInfo?.parts?.parts.some(item=>item.id===part)||!sourceInfo?.bones?.some(item=>item.index===bone))return;
  partAttachments.set(part,{bone,avoid_ground:$('attach-ground').checked});renderAttachments();availability();};
function editCount(){$('part-edit-count').textContent=partEdits.length?`${partEdits.length} 项修正尚未保存`:'当前没有未保存修正。';}
function canEditParts(){return !busy&&!comparing&&!health?.active&&!health?.operation&&!activeStates.has(currentJob?.state)&&model&&$('view-empty').hidden;}
$('add-part').onclick=()=>{const parts=sourceInfo?.parts?.parts;if(!parts?.length)return;const id=Math.max(...parts.map(part=>part.id))+1,name=`新部件 ${id+1}`;parts.push({id,name,faces:0});partEdits.push({operation:'create',part:id,name});renderParts(sourceInfo.parts);$('paint-target').value=String(id);$('action').value='edit_parts';changeAction();editCount();};
$('merge-parts').onclick=()=>{const parts=[...$('parts-list').querySelectorAll('input[type=checkbox]:checked')].map(input=>+input.dataset.part);if(parts.length<2){message('请至少选中两个部件。');return;}partEdits.push({operation:'merge',parts,name:'合并部件'});editCount();$('action').value='edit_parts';changeAction();};
$('paint-faces').onclick=()=>{facePaint=!facePaint;$('paint-faces').classList.toggle('active',facePaint);$('view-info').textContent=facePaint?'点击模型上的三角面，将它修正到所选部件。':'拖动旋转、滚轮缩放；点选部件或骨骼。';};
let downPoint=null;
function paintSelection(hit){
  if(!canEditParts()){message('请等待任务结束，并切换到当前模型预览后再修正。');return;}
  const target=+$('paint-target').value,percentage=+$('brush-radius').value;
  if(!Number.isFinite(percentage)||percentage<0||percentage>20){message('笔刷半径应在 0–20% 之间。',true);return;}
  const radius=new THREE.Box3().setFromObject(model).getSize(new THREE.Vector3()).length()*percentage/100;
  const selected=[],preview=[],a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),center=new THREE.Vector3();
  let total=0,overflow=false;model.updateMatrixWorld(true);
  model.traverse(object=>{if(!object.isMesh||!object.visible||partOf(object)===null)return;const association=gltf.parser.associations.get(object);if(association?.meshes===undefined)return;
    const indices=object.geometry.index,count=indices?indices.count/3:object.geometry.attributes.position.count/3,faces=[];
    for(let face=0;face<count;face++){
      if(percentage===0&&(object!==hit.object||face!==hit.faceIndex))continue;
      object.getVertexPosition(indices?indices.getX(face*3):face*3,a).applyMatrix4(object.matrixWorld);
      object.getVertexPosition(indices?indices.getX(face*3+1):face*3+1,b).applyMatrix4(object.matrixWorld);
      object.getVertexPosition(indices?indices.getX(face*3+2):face*3+2,c).applyMatrix4(object.matrixWorld);
      center.copy(a).add(b).add(c).multiplyScalar(1/3);
      if(percentage===0||center.distanceToSquared(hit.point)<=radius*radius){faces.push(face);total++;if(preview.length<45000)preview.push(...a.toArray(),...b.toArray(),...c.toArray());if(faces.length>100000||total>200000){overflow=true;break;}}
    }
    if(faces.length)selected.push({operation:'faces',mesh:association.meshes,primitive:association.primitives||0,faces,target_part:target});
  });
  if(overflow){message('区域过大，请减小笔刷半径后再修正。',true);return;}if(!total)return;
  if(partEdits.length+selected.length>500){message('请先保存已有修正，再继续涂选区域。');return;}
  partEdits.push(...selected);clearPaintPreview();const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(preview,3));
  paintPreview=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0xd9478c,transparent:true,opacity:.5,side:THREE.DoubleSide,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}));scene.add(paintPreview);editCount();
  $('view-info').textContent=`已选择 ${total.toLocaleString()} 个三角面，保存后移入所选部件。`;$('action').value='edit_parts';changeAction();
}
renderer.domElement.addEventListener('pointerdown',event=>{downPoint=[event.clientX,event.clientY];});
renderer.domElement.addEventListener('pointerup',event=>{
  if(!model||transform.dragging||!downPoint||Math.hypot(event.clientX-downPoint[0],event.clientY-downPoint[1])>5)return;
  const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
  if(showBones&&!comparing){const hit=raycaster.intersectObjects(boneMarkers.filter(Boolean),false)[0];if(hit){selectBone(hit.object.userData.boneIndex);return;}}
  const hit=raycaster.intersectObject(model,true).find(hit=>hit.object.isMesh&&hit.object.visible);if(!hit)return;
  const id=partOf(hit.object);
  if(facePaint&&id!==null&&hit.faceIndex!==undefined){paintSelection(hit);}
  else if(id!==null){for(const row of $('parts-list').children)row.classList.toggle('active',+row.dataset.part===id);}
});

const mappingRoles=['hips','spine','chest','neck','head','left_shoulder','left_upper_arm','left_forearm','left_hand','right_shoulder','right_upper_arm','right_forearm','right_hand','left_thigh','left_shin','left_foot','left_toe','right_thigh','right_shin','right_foot','right_toe'];
function renderMapping(){
  const container=$('mapping');container.replaceChildren();
  const roleNames={hips:'骨盆',spine:'脊柱',chest:'胸腔',neck:'颈部',head:'头部',left_shoulder:'左肩',right_shoulder:'右肩',left_upper_arm:'左上臂',right_upper_arm:'右上臂',left_forearm:'左前臂',right_forearm:'右前臂',left_hand:'左手',right_hand:'右手',left_thigh:'左大腿',right_thigh:'右大腿',left_shin:'左小腿',right_shin:'右小腿',left_foot:'左脚',right_foot:'右脚',left_toe:'左脚趾',right_toe:'右脚趾'};
  const header=document.createElement('div');header.className='studio-mapping-row';for(const text of ['部位','来源动作','目标角色']){const span=document.createElement('span');span.textContent=text;header.append(span);}container.append(header);
  for(const role of mappingRoles){const row=document.createElement('div');row.className='studio-mapping-row';const text=document.createElement('span');text.textContent=roleNames[role]||role;row.append(text);for(const side of ['source','target']){const info=side==='source'?motionAsset:sourceInfo;const select=document.createElement('select');select.dataset.role=role;select.dataset.side=side;select.setAttribute('aria-label',`${side==='source'?'来源':'目标'} ${roleNames[role]||role}`);select.append(new Option(info?.mapping?.[role]?`自动 · ${info.mapping[role]}`:'自动识别',''));for(const bone of info?.bones||[])select.append(new Option(bone.name,bone.name));select.disabled=side==='source'&&$('action').value==='motion';row.append(select);}container.append(row);}
}
function renderConstraints(){
  const list=$('constraints');list.replaceChildren();
  constraints.forEach((value,index)=>{const row=document.createElement('div');row.className='studio-constraint-row';const label=document.createElement('span');label.textContent=value.type==='root'?'路径点':'关键姿态';row.append(label);for(const key of ['time',...(value.type==='root'?['x','z']:[])]){const input=document.createElement('input');input.type='number';input.step='.1';input.value=String(value[key]??0);input.title=key;input.onchange=()=>{value[key]=+input.value;};row.append(input);}const remove=document.createElement('button');remove.textContent='删除';remove.onclick=()=>{constraints.splice(index,1);renderConstraints();};row.append(remove);list.append(row);});
}
$('add-waypoint').onclick=()=>{constraints.push({type:'root',time:Math.min(+$('duration').value,1+constraints.length),x:0,z:0});renderConstraints();};
$('add-pose').onclick=()=>{if(!boneMarkers.length){message('当前模型没有骨架，请先绑定。');return;}constraints.push({type:'pose',time:Math.min(+$('duration').value,1+constraints.length),joints:boneMarkers.map(marker=>marker.position.toArray())});renderConstraints();};

async function refreshSources(selected=null){const revision=selectionRevision,previous=selected||$('source').value;const values=await api('/api/studio/sources');if(revision!==selectionRevision)return;$('source').replaceChildren(new Option('选择一个模型',''),...values.map(row=>new Option(row.name,row.value)));$('source').value=values.some(row=>row.value===previous)?previous:'';availability();}
async function sourceChanged(){
  const revision=++selectionRevision,source=$('source').value;++sourceRevision;
  if(currentJob&&!activeStates.has(currentJob.state)){
    currentJob=null;$('progress-box').hidden=true;$('downloads').replaceChildren();
    $('result-summary').textContent='已选择新模型，加工记录中的原成品仍可打开。';$('log').textContent='尚未运行。';
  }
  sourceInfo=null;partAttachments.clear();renderAttachments();if(!source){disposeModel();loadedUrl=null;sourceUrl=null;resultUrl=null;busy=false;drawBones([]);renderParts(null);availability();return;}busy=true;availability();
  try{const info=await api('/api/studio/source?source='+encodeURIComponent(source));if(revision!==selectionRevision)return;sourceInfo=info;sourceUrl=info.url;resultUrl=null;partEdits=[];boneEdits.clear();constraints=[];renderConstraints();comparing=false;loadedUrl=null;const shown=await showModel(sourceUrl);if(revision!==selectionRevision)return;if(!shown){disposeModel();loadedUrl=null;}drawBones(info.bones);renderParts(info.parts);renderMapping();editCount();$('source-info').textContent=`${info.faces.toLocaleString()} 三角面 · ${info.materials} 材质 · ${info.bones.length} 关节`;}
  catch(error){if(revision===selectionRevision)message(error.message,true);}finally{if(revision===selectionRevision){busy=false;availability();}}
}
$('source').onchange=sourceChanged;$('refresh-sources').onclick=()=>refreshSources().catch(error=>message(error.message,true));
async function uploadMesh(file){busy=true;availability();message('正在上传并检查模型…');try{const form=new FormData();form.append('file',file);const item=await api('/api/assets',{method:'POST',body:form});await refreshSources('asset:'+item.id);await sourceChanged();message('模型已保存，可选择加工方式。');}catch(error){message(error.message,true);}finally{busy=false;availability();}}
$('mesh-file').onchange=()=>{if($('mesh-file').files[0])void uploadMesh($('mesh-file').files[0]);};
for(const eventName of ['dragenter','dragover'])$('mesh-drop').addEventListener(eventName,event=>{event.preventDefault();$('mesh-drop').classList.add('drag');});
$('mesh-drop').addEventListener('dragleave',()=>$('mesh-drop').classList.remove('drag'));
$('mesh-drop').addEventListener('drop',event=>{event.preventDefault();$('mesh-drop').classList.remove('drag');const file=event.dataTransfer.files[0];if(file)void uploadMesh(file);});
function motionInformation(){
  $('motion-name').textContent=motionAsset?`${motionAsset.name} · ${motionAsset.bones.length} 关节 · ${motionAsset.clips.length} 片段`:'支持 BVH、FBX 和带动作的 GLB。';
  $('motion-armature').replaceChildren(...(motionAsset?.armatures?.length?motionAsset.armatures.map(name=>new Option(name,name)):[new Option('使用文件骨架','')]));$('motion-armature').value=motionAsset?.armature||'';$('motion-armature').disabled=!(motionAsset?.armatures?.length>1);
  $('motion-source-clip').replaceChildren(...(motionAsset?.clips||[]).map(clip=>new Option(`${clip.name} · ${clip.duration.toFixed(2)} 秒`,String(clip.index))));$('motion-source-clip').disabled=!motionAsset;renderMapping();
}
$('motion-file').onchange=async()=>{const file=$('motion-file').files[0];if(!file||busy)return;busy=true;motionAsset=null;motionValid=false;$('motion-up').value='Y';availability();$('motion-name').textContent='正在上传并解析动作骨架…';try{const form=new FormData();form.append('file',file);motionAsset=await api('/api/studio/upload-motion',{method:'POST',body:form});motionValid=true;motionInformation();}catch(error){motionInformation();message(error.message,true);}finally{busy=false;availability();}};
async function refreshMotion(){if(!motionAsset)return;busy=true;motionValid=false;availability();try{motionAsset=await api('/api/studio/motion-source?asset='+motionAsset.id+'&up_axis='+$('motion-up').value+'&armature='+encodeURIComponent($('motion-armature').value));motionValid=true;motionInformation();}catch(error){message(error.message,true);}finally{busy=false;availability();}}
$('motion-up').onchange=refreshMotion;$('motion-armature').onchange=refreshMotion;
function requestBody(){
  const action=$('action').value;
  const body={source:$('source').value,action,rig_engine:$('rig-engine').value,seed:+$('seed').value,
    max_error:+$('max-error').value/100,priority:$('priority').value,target_quads:+$('target-quads').value,
    retopology_engine:$('retopology-engine').value,retopology_surface:$('retopology-surface').value,
    material_projection:$('material-projection').value,preserve_sharp:$('preserve-sharp').checked,edge_scaling:+$('edge-scaling').value,
    sharp_angle:+$('sharp-angle').value,adaptivity:+$('adaptivity').value,texture_size:+$('texture-size').value,uv_backend:$('uv-backend').value,
    segment_points:+$('segment-points').value,segment_prompts:+$('segment-prompts').value,
    part_guidance:+$('part-guidance').value,part_steps:+$('part-steps').value,part_resolution:+$('part-resolution').value,
    part_edits:partEdits,bone_edits:[...boneEdits.values()],part_attachments:[...partAttachments].map(([part,assignment])=>({part,...assignment})),use_skeleton:$('use-skeleton').checked,release_contacts:$('release-contacts').checked,
    motion_asset:action==='retarget'?motionAsset?.id||null:null,motion_clip:+$('motion-source-clip').value||0,motion_armature:$('motion-armature').value||null,motion_up:$('motion-up').value,prompt:$('prompt').value,duration:+$('duration').value,
    text_device:$('text-device').value,speed:+$('speed').value,loop:$('loop').checked,
    root_motion:$('root-motion').value,foot_lock:$('foot-lock').checked,rest_pose:$('rest-pose').value,
    motion_scale:$('motion-scale').value===''?null:+$('motion-scale').value,
    source_mapping:action==='retarget'?Object.fromEntries([...$('mapping').querySelectorAll('select[data-side=source]')].filter(select=>select.value).map(select=>[select.dataset.role,select.value])):{},
    joint_mapping:Object.fromEntries([...$('mapping').querySelectorAll('select[data-side=target]')].filter(select=>select.value).map(select=>[select.dataset.role,select.value])),constraints};
  if(action==='simplify'){if($('budget-mode').value==='count')body.target_faces=+$('target-faces').value;else body.ratio=+$('ratio').value/100;}
  return body;
}
$('run').onclick=async()=>{if($('run').disabled)return;for(const input of root.querySelectorAll('input,textarea'))if(!input.closest('[hidden]')&&!input.reportValidity())return;busy=true;availability();try{currentJob=await post('/api/studio/jobs',requestBody());$('downloads').replaceChildren();$('result-summary').textContent='任务已保存，正在启动。';await renderJob(currentJob);}catch(error){message(error.message,true);}finally{busy=false;availability();}};
$('cancel').onclick=async()=>{if(!currentJob)return;try{await post('/api/jobs/'+currentJob.id+'/cancel',{});await poll();}catch(error){message(error.message,true);}};
async function renderJob(job){
  currentJob=job;const selection=selectionRevision,active=activeStates.has(job.state);$('progress-box').hidden=false;$('progress-label').textContent=job.message||job.state;
  $('cancel').hidden=!active;$('elapsed').textContent=`${Math.max(0,Math.floor((active?Date.now()/1000:job.updated||job.created)-job.created))} 秒`;
  const progress=job.progress||{},measured=Number.isFinite(progress.current)&&Number.isFinite(progress.total)&&progress.total>0;
  $('progress').hidden=!measured;if(measured){$('progress').max=progress.total;$('progress').value=progress.current;}
  $('progress-detail').textContent=[progress.device==='cuda'?'CUDA / GPU':progress.device==='cpu'?'CPU':'',measured?`${progress.current} / ${progress.total} ${progress.unit||''}`:'当前阶段未报告可量化总量',active&&job.last_progress_at?`距上次进展 ${Math.floor(Date.now()/1000-job.last_progress_at)} 秒`:''].filter(Boolean).join(' · ');
  const names={'output.glb':'下载 GLB','rig-original.glb':'原始绑定 GLB','output.fbx':'下载 FBX','output.obj':'四边面 OBJ','parts.zip':'独立部件 ZIP','original-segmentation.glb':'原始拆件 GLB','original-parts.zip':'原始拆件 ZIP','motion.bvh':'动作 BVH','report.json':'运行报告'};
  $('downloads').replaceChildren(...Object.entries(names).filter(([name])=>job.available_files?.includes(name)).map(([name,label])=>{const link=document.createElement('a');link.href=`/api/studio/jobs/${job.id}/files/${name}`;link.download=`Hi3D-${job.id.slice(0,8)}-${name}`;link.textContent=label;return link;}));
  if(job.state==='succeeded'){
    const report=job.report||{},details=report.details||{};
    const quadInfo=Number.isFinite(details.quad_faces)?` · ${Number(details.quad_faces).toLocaleString()} 四边面 / ${Number((details.polygons||details.quad_faces)-details.quad_faces).toLocaleString()} 其他多边形`:'';
    const notices=[...new Set([details.message,details.notice,...(Array.isArray(details.warnings)?details.warnings:[])].filter(value=>typeof value==='string'&&value.trim()))];
    $('result-summary').textContent=`${actions[job.options.action]||'加工'}完成 · ${Number(report.input_faces||0).toLocaleString()} → ${Number(report.output_faces||0).toLocaleString()} 三角面${quadInfo} · ${report.skins||0} 骨架 · ${report.animations?.length||0} 动作 · ${report.seconds||0} 秒。${notices.join(' ')}`;
  }else if(['failed','cancelled','interrupted'].includes(job.state)){$('result-summary').textContent=job.message||'任务停止；已有文件保留。';}
  if(!active&&job.available_files?.includes('output.glb')&&resultUrl!==`/api/jobs/${job.id}/files/output.glb`){
    try{
      const key=`job:${job.id}:output.glb`,url=`/api/jobs/${job.id}/files/output.glb`;
      const info=await api('/api/studio/source?source='+encodeURIComponent(key));if(currentJob?.id!==job.id||selection!==selectionRevision)return;
      comparing=false;const rendered=await showModel(url);if(currentJob?.id!==job.id||selection!==selectionRevision)return;
      if(rendered){sourceInfo=info;resultUrl=url;sourceUrl=`/api/jobs/${job.id}/files/coarse.glb`;
        partEdits=[];boneEdits.clear();partAttachments.clear();editCount();await refreshSources(key);
        if(currentJob?.id!==job.id||selection!==selectionRevision)return;drawBones(info.bones);renderParts(info.parts);renderMapping();}
    }catch(error){if(currentJob?.id!==job.id||selection!==selectionRevision)return;message('成品已保存，预览信息读取失败：'+error.message,true);}
  }
  try{const response=await fetch(`/api/jobs/${job.id}/log`);if(response.ok){const text=await response.text();if(currentJob?.id===job.id&&selection===selectionRevision)$('log').textContent=text.slice(-30000);}}catch{}
  availability();
}
async function history(){const jobs=(await api('/api/jobs')).filter(job=>job.options?.mode==='studio');const list=$('history');list.replaceChildren();for(const job of jobs.slice(0,40)){const row=document.createElement('div');row.className='studio-history-row';const text=document.createElement('div');text.textContent=`${actions[job.options.action]||job.options.action} · ${job.message||job.state}`;const date=document.createElement('small');date.textContent=new Date(job.created*1000).toLocaleString();text.append(date);const button=document.createElement('button');button.textContent='打开';button.onclick=async()=>{await renderJob(await api('/api/jobs/'+job.id));};row.append(text,button);list.append(row);}}
$('refresh-history').onclick=()=>history().catch(error=>message(error.message,true));
$('download').onclick=async()=>{try{await post('/api/studio/models/download',{selection:selectedFeature()});message('已开始准备所需组件，可在此查看进度。');await poll();}catch(error){message(error.message,true);}};
$('verify').onclick=async()=>{busy=true;availability();try{const value=await post('/api/studio/models/verify',{selection:selectedFeature()});message(value.verification_errors?.length?'校验未通过：'+value.verification_errors.join('、'):'所选组件校验完成。',!!value.verification_errors?.length);}catch(error){message(error.message,true);}finally{busy=false;await poll();}};
async function poll(){
  if(polling)return;polling=true;
  try{state=await api('/api/studio/status');if(currentJob&&(activeStates.has(currentJob.state)||(currentJob.available_files?.includes('output.glb')&&resultUrl!==`/api/jobs/${currentJob.id}/files/output.glb`))){
      const id=currentJob.id,selection=selectionRevision,job=await api('/api/jobs/'+id);
      if(currentJob?.id===id&&selection===selectionRevision)await renderJob(job);
    }
    if(state.download?.state==='downloading'){const download=state.download;message(`${download.message} · ${((download.file_bytes||0)/1048576).toFixed(0)} / ${((download.file_total||0)/1048576).toFixed(0)} MiB`);}availability();
  }catch(error){message(error.message,true);}finally{polling=false;}
}
window.addEventListener('hi3d:health',event=>{health=event.detail;availability();});
window.addEventListener('hi3d:studio-job',event=>{void renderJob(event.detail);});
await refreshSources();await history();changeAction();await poll();setInterval(()=>{if(root.classList.contains('active')||currentJob&&activeStates.has(currentJob.state))void poll();},2500);
