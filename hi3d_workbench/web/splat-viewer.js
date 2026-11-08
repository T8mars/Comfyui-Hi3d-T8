import * as THREE from 'three';
import {OrbitControls} from './vendor/three/controls/OrbitControls.js';

// The original 3DGS covariance, orientation, opacity and SH0 colour are drawn
// as sorted anisotropic Gaussian ellipses. No mesh/point-cloud conversion.
const status=document.querySelector('#status');
let renderer,controls,geometry,material,sortWorker;
function readPly(buffer) {
  const bytes=new Uint8Array(buffer),limit=Math.min(bytes.length,16384);
  let end=-1;
  const marker=new TextEncoder().encode('end_header\n');
  outer:for(let i=0;i<limit-marker.length;i++){for(let j=0;j<marker.length;j++)if(bytes[i+j]!==marker[j])continue outer;end=i+marker.length;break;}
  if(end<0)throw new Error('高斯 PLY 头部不完整');
  const header=new TextDecoder().decode(bytes.subarray(0,end));
  if(!header.includes('format binary_little_endian 1.0'))throw new Error('只支持标准二进制高斯 PLY');
  const count=Number(/element vertex (\d+)/.exec(header)?.[1]),props=[...header.matchAll(/^property float (\w+)$/gm)].map(x=>x[1]);
  if(!Number.isInteger(count)||count<1||count>1048576||end+count*props.length*4!==bytes.length)throw new Error('高斯数量或文件长度不正确');
  const names=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity','f_dc_0','f_dc_1','f_dc_2'];
  const offsets=names.map(n=>props.indexOf(n));if(offsets.some(n=>n<0))throw new Error('此 PLY 缺少高斯属性，不能当作网格 PLY 预览');
  const rows=new Float32Array(count*14),view=new DataView(buffer),bounds=new THREE.Box3();
  for(let i=0;i<count;i++){
    for(let j=0;j<14;j++){const v=view.getFloat32(end+(i*props.length+offsets[j])*4,true);if(!Number.isFinite(v))throw new Error('高斯属性包含无效数值');rows[i*14+j]=v;}
    for(let j=3;j<6;j++)rows[i*14+j]=Math.exp(rows[i*14+j]);
    rows[i*14+10]=1/(1+Math.exp(-rows[i*14+10]));
    for(let j=11;j<14;j++)rows[i*14+j]=Math.max(0,Math.min(1,.5+.28209479177387814*rows[i*14+j]));
    bounds.expandByPoint(new THREE.Vector3(rows[i*14],rows[i*14+1],rows[i*14+2]));
  }
  return {count,rows,bounds};
}
async function main() {
  const source=new URLSearchParams(location.search).get('src');
  if(!/^\/api\/jobs\/[a-f0-9]{32}\/files\/splat\.ply$/.test(source||''))throw new Error('缺少有效的任务高斯地址');
  const response=await fetch(source);if(!response.ok)throw new Error('读取高斯失败：'+response.status);
  const {count,rows,bounds}=readPly(await response.arrayBuffer());
  let referenceDirection=null;
  try{const r=await fetch(source.replace('splat.ply','idol-camera.json'));if(r.ok){const d=(await r.json()).gltf_direction;if(Array.isArray(d)&&d.length===3&&d.every(Number.isFinite)&&Math.hypot(...d)>.001)referenceDirection=new THREE.Vector3(...d).normalize();}}catch{}
  renderer=new THREE.WebGLRenderer({antialias:false,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#171c27');document.body.append(renderer.domElement);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,.001,1000),group=new THREE.Group();scene.add(group);
  const center=bounds.getCenter(new THREE.Vector3()),extent=Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
  if(!Number.isFinite(extent)||extent<=0)throw new Error('高斯包围框无效');
  group.rotation.x=Math.PI;group.position.set(-center.x,center.y,center.z);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,0);
  const reset=()=>{if(referenceDirection)camera.position.copy(referenceDirection).multiplyScalar(extent*2.5);else camera.position.set(extent*1.5,extent*.7,extent*2);controls.target.set(0,0,0);controls.update();};document.querySelector('#reset').onclick=reset;
  geometry=new THREE.InstancedBufferGeometry();geometry.setIndex([0,1,2,0,2,3]);
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([-3,-3,0,3,-3,0,3,3,0,-3,3,0],3));
  const specs={center:3,scale:3,rotation:4,color:4};for(const [name,size] of Object.entries(specs))geometry.setAttribute('g_'+name,new THREE.InstancedBufferAttribute(new Float32Array(count*size),size));geometry.instanceCount=count;
  material=new THREE.ShaderMaterial({transparent:true,depthTest:false,depthWrite:false,uniforms:{viewport:{value:new THREE.Vector2()}},
    vertexShader:`attribute vec3 g_center;attribute vec3 g_scale;attribute vec4 g_rotation;attribute vec4 g_color;uniform vec2 viewport;varying vec2 ellipse;varying vec4 rgba;
    void main(){vec4 c=modelViewMatrix*vec4(g_center,1.);if(c.z>=-.001){gl_Position=vec4(2.,2.,2.,1.);return;}
    vec4 q=normalize(vec4(g_rotation.yzw,g_rotation.x));vec3 x=vec3(1.-2.*(q.y*q.y+q.z*q.z),2.*(q.x*q.y+q.z*q.w),2.*(q.x*q.z-q.y*q.w));vec3 y=vec3(2.*(q.x*q.y-q.z*q.w),1.-2.*(q.x*q.x+q.z*q.z),2.*(q.y*q.z+q.x*q.w));vec3 z=vec3(2.*(q.x*q.z+q.y*q.w),2.*(q.y*q.z-q.x*q.w),1.-2.*(q.x*q.x+q.y*q.y));
    mat3 a=mat3(modelViewMatrix)*mat3(x*g_scale.x,y*g_scale.y,z*g_scale.z);float iz=-1./c.z;vec3 jx=vec3(projectionMatrix[0][0]*viewport.x*.5*iz,0.,projectionMatrix[0][0]*viewport.x*.5*c.x*iz*iz);vec3 jy=vec3(0.,projectionMatrix[1][1]*viewport.y*.5*iz,projectionMatrix[1][1]*viewport.y*.5*c.y*iz*iz);
    vec3 ax=vec3(dot(jx,a[0]),dot(jx,a[1]),dot(jx,a[2]));vec3 ay=vec3(dot(jy,a[0]),dot(jy,a[1]),dot(jy,a[2]));float xx=dot(ax,ax)+.3,xy=dot(ax,ay),yy=dot(ay,ay)+.3;float mid=(xx+yy)*.5;float d=length(vec2((xx-yy)*.5,xy));float l1=max(mid+d,.1),l2=max(mid-d,.1);vec2 e1=abs(xy)>1e-5?normalize(vec2(xy,l1-xx)):(xx>=yy?vec2(1.,0.):vec2(0.,1.));vec2 e2=vec2(-e1.y,e1.x);vec2 offset=position.x*sqrt(l1)*e1+position.y*sqrt(l2)*e2;vec4 p=projectionMatrix*c;p.xy+=offset*2./viewport*p.w;gl_Position=p;ellipse=position.xy;rgba=g_color;}`,
    fragmentShader:`precision highp float;varying vec2 ellipse;varying vec4 rgba;void main(){float r=dot(ellipse,ellipse);if(r>9.)discard;float a=min(.99,rgba.a*exp(-.5*r));if(a<.003)discard;gl_FragColor=vec4(rgba.rgb,a);}`});
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;group.add(mesh);
  const mv=new THREE.Matrix4();let dirty=true,lastSort=0,sortPending=false;
  controls.addEventListener('change',()=>{dirty=true;});reset();
  function applyOrder(order){
    const p=geometry.getAttribute('g_center'),s=geometry.getAttribute('g_scale'),q=geometry.getAttribute('g_rotation'),c=geometry.getAttribute('g_color');
    for(let k=0;k<count;k++){const i=order[k]*14;p.setXYZ(k,rows[i],rows[i+1],rows[i+2]);s.setXYZ(k,rows[i+3],rows[i+4],rows[i+5]);q.setXYZW(k,rows[i+6],rows[i+7],rows[i+8],rows[i+9]);c.setXYZW(k,rows[i+11],rows[i+12],rows[i+13],rows[i+10]);}
    for(const a of [p,s,q,c])a.needsUpdate=true;
  }
  const centers=new Float32Array(count*3);for(let i=0;i<count;i++)centers.set(rows.subarray(i*14,i*14+3),i*3);
  sortWorker=new Worker(new URL('./splat-sort-worker.js',import.meta.url),{type:'module'});
  sortWorker.onmessage=({data})=>{if(data.type!=='sorted')return;applyOrder(data.order);sortPending=false;
    status.textContent=count.toLocaleString()+' 个高斯 · 原始外观';window.__hi3dSplat={count,representation:'anisotropic_gaussian',ready:true};};
  sortWorker.onerror=()=>{status.textContent='高斯排序失败，请重新打开预览';window.__hi3dSplat={ready:false,error:'sort_worker_failed'};};
  sortWorker.postMessage({type:'init',positions:centers.buffer},[centers.buffer]);
  function sort(){scene.updateMatrixWorld(true);camera.updateMatrixWorld(true);mv.multiplyMatrices(camera.matrixWorldInverse,mesh.matrixWorld);
    sortPending=true;dirty=false;sortWorker.postMessage({type:'sort',matrix:Array.from(mv.elements)});
  }
  const resize=()=>{const w=innerWidth,h=innerHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.getDrawingBufferSize(material.uniforms.viewport.value);dirty=true;};addEventListener('resize',resize);resize();
  status.textContent=count.toLocaleString()+' 个高斯 · 准备视图';
  renderer.domElement.addEventListener('webglcontextlost',()=>{status.textContent='显卡上下文丢失，请重新打开预览';});
  renderer.setAnimationLoop(time=>{controls.update();if(dirty&&!sortPending&&time-lastSort>100){sort();lastSort=time;}renderer.render(scene,camera);});
  window.__hi3dSplat={count,representation:'anisotropic_gaussian',ready:false};
}
main().catch(error=>{status.textContent='预览失败：'+error.message;window.__hi3dSplat={ready:false,error:error.message};});
addEventListener('pagehide',()=>{sortWorker?.terminate();renderer?.setAnimationLoop(null);controls?.dispose();geometry?.dispose();material?.dispose();renderer?.dispose();});
