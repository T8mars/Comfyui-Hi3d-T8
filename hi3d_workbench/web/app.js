const $ = selector => document.querySelector(selector);
let token = $('meta[name="hi3d-token"]').content;
let inputImage = null, inputMesh = null, currentJob = null, health = null;
let online = false, pending = null, exportPending = false, pollBusy = false;
let epoch = 0, shownJob = null, sessionReady = false, sessionDefaultsApplied = false, toastTimer, logsKey = '';
const activeStates = new Set(['running', 'queued', 'cancelling']);
const stageLabels = {queued:'任务已保存，等待启动', loading:'正在加载本地模型', conditioning:'正在处理参考图片', voxelizing:'正在准备粗模型', sampling:'正在生成几何细节', decoding:'正在解码几何', extracting:'正在提取网格', postprocessing:'正在整理几何', exporting:'正在保存结果', complete:'精修完成，可以对比导出', failed:'精修失败，输入与日志已保留', cancelled:'任务已停止', interrupted:'任务已中断'};
const stateLabels = {succeeded:'已完成', failed:'失败', running:'运行中', queued:'等待启动', cancelled:'已停止', interrupted:'已中断', cancelling:'停止中'};
function toast(message) {
  clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 7000);
}
class ApiError extends Error {
  constructor(message, status = 0, network = false) { super(message); this.status = status; this.network = network; }
}
async function api(path, options = {}, retry = true) {
  const {timeoutMs = options.method === 'POST' ? 600000 : path === '/api/health' || path === '/api/session' ? 8000 : 15000, responseType = 'json', ...init} = options;
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
  const response = await fetch(path, {...init, signal:controller.signal, headers:{'x-hi3d-token':token, ...init.headers}});
  if (response.status === 403 && retry && path !== '/api/session') {
    const session = await api('/api/session', {}, false); token = session.token;
    return api(path, options, false);
  }
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail; } catch {}
    if (Array.isArray(detail)) detail = detail.map(x => `${x.loc?.at(-1) || '参数'}：${x.msg}`).join('；');
    throw new ApiError(typeof detail === 'string' ? detail : `请求失败（${response.status}）`, response.status);
  }
  return responseType === 'text' ? await response.text() : responseType === 'blob' ? await response.blob() : await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new ApiError('本地请求超时，请检查启动窗口，稍后重试。', 0, true);
    if (error instanceof TypeError) throw new ApiError('无法连接工作台，请确认启动窗口仍在运行。', 0, true);
    throw error;
  } finally { clearTimeout(timeout); }
}
const post = (path, body = {}) => api(path, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)});
function unavailable(error) {
  online = false; $('#connection-alert').hidden = false;
  $('#connection-message').textContent = `工作台暂不可用：${error.message}。恢复后会自动重连。`;
  $('#health-badge').textContent = '● 工作台暂不可用'; $('#health-badge').classList.add('offline'); availability();
}
function showError(error) { if (error.network) unavailable(error); toast(error.message); }
function page(name) {
  document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === name));
  document.querySelectorAll('[data-page]').forEach(el => {
    el.classList.toggle('active', el.dataset.page === name);
    el.setAttribute('aria-current', el.dataset.page === name ? 'page' : 'false');
  });
  const text = {workspace:['让粗模型，多一点细节。','上传同一物体的参考图和粗模型，用 Hi3D 精修几何，再对比导出。'],history:['每一次尝试，都保留下来。','打开旧任务对比结果，或复用输入继续调整。'],models:['准备好环境，再安心创作。','模型文件保存在本机，推理使用本地 GPU。']};
  $('#page-title').textContent = text[name][0]; $('#page-subtitle').textContent = text[name][1];
  updateDock();
  if (name === 'history') void history();
}
document.querySelectorAll('[data-page]').forEach(button => { button.onclick = () => page(button.dataset.page); });
$('#health-badge').onclick = () => page('models');
function gpuBusy() { return Boolean(health?.active || health?.operation || health?.recovery_blocked || activeStates.has(currentJob?.state)); }
function hasJobFile(job, name) { return Array.isArray(job?.available_files) ? job.available_files.includes(name) : ['coarse.glb','input.png'].includes(name); }
function inputBusy() { return !online || Boolean(pending) || gpuBusy(); }
function availability() {
  const busy = inputBusy(), checking = health?.model_checking, downloading = health?.models.download.state === 'downloading';
  const isHistory = Boolean(currentJob), ready = Boolean(health?.models.ready);
  $('#run').disabled = busy || checking || downloading || !ready || !inputImage || !inputMesh || isHistory;
  $('#run').textContent = pending === 'submitting' ? '正在校验文件并提交…' : '开始 Hi3D 精修 →';
  for (const id of ['image-file','mesh-file','load-sample','new-task']) $('#'+id).disabled = busy;
  $('#load-sample').textContent = pending === 'sample' ? '正在载入图片与粗模型…' : '载入样例，熟悉流程 →';
  $('#mask-open').disabled = busy || !inputImage;
  $('#mask-apply').disabled = busy;
  for (const id of ['resolution','steps','seed','offload','attention']) $('#'+id).disabled = busy || isHistory;
  $('#reuse').hidden = !isHistory || activeStates.has(currentJob.state);
  $('#reuse').disabled = busy || !hasJobFile(currentJob,'input.png') || !hasJobFile(currentJob,'coarse.glb');
  $('#reuse').textContent = pending === 'reuse' ? '正在准备可编辑输入…' : '复用这次输入和参数';
  $('#cancel').hidden = !currentJob || !activeStates.has(currentJob.state);
  $('#cancel').disabled = !online || Boolean(pending) || currentJob?.state === 'cancelling';
  $('#active-job').hidden = !health?.active;
  $('#active-label').textContent = currentJob && currentJob.id === health?.active ? `当前任务：${stateLabels[currentJob.state] || '运行中'}` : '有一个精修任务正在运行';
  $('#show-active').disabled = !online || Boolean(pending);
  $('#cancel-active').disabled = !online || Boolean(pending) || Boolean(currentJob && currentJob.id === health?.active && currentJob.state === 'cancelling');
  let help = !online ? '工作台连接已断开，恢复连接后可以继续。' : pending === 'submitting' ? '正在完整校验模型文件，请等待。提交期间不会重复创建任务。' : pending ? '正在处理，请稍候…' : health?.active ? '精修运行中，可在上方查看或停止任务。' : health?.operation ? '正在检查 GPU 或准备任务，请稍候。' : checking ? '正在完整校验本地模型文件。首次启动或文件变化后会自动检查，请稍候。' : downloading ? '正在下载必需组件，可在「模型与环境」查看进度。' : !ready ? '本地模型尚未就绪，请前往「模型与环境」查看原因并补齐或校验。' : isHistory ? '当前正在查看记录。点击「复用这次输入和参数」即可调整后再次精修。' : !inputImage || !inputMesh ? '准备一张参考图片和同物体粗模型，或点击「载入样例」。' : '输入已齐全。建议先用 512 档、默认 25 步试一次。';
  $('#run-help').textContent = help;
  for (const [id, ok, name] of [['ready-image', Boolean(inputImage || currentJob && hasJobFile(currentJob,'input.png')), '参考图片'], ['ready-mesh', Boolean(inputMesh || currentJob && hasJobFile(currentJob,'coarse.glb')), '粗模型'], ['ready-models', online && ready, '本地模型']]) {
    $('#'+id).textContent = `${ok ? '✓' : '○'} ${name}`; $('#'+id).classList.toggle('ready', ok);
  }
  const modelBusy = !online || Boolean(pending) || gpuBusy() || checking || downloading;
  for (const id of ['download-models','verify-models','diagnose']) $('#'+id).disabled = modelBusy;
  $('#verify-models').textContent = checking || pending === 'verify' ? '正在完整校验…' : '重新完整校验';
  $('#diagnose').textContent = health?.operation === 'diagnosing' || pending === 'diagnose' ? '正在检查 GPU…' : '检查 GPU 是否可用';
  const phase = currentJob?.state === 'succeeded' ? 3 : inputImage && inputMesh || currentJob ? 2 : 1;
  for (const [index, id] of ['step-input','step-run','step-export'].entries()) {
    $('#'+id).classList.toggle('current', phase === index+1); $('#'+id).classList.toggle('done', phase > index+1);
  }
  exportAvailability();
  $('#mask-apply').textContent = pending === 'mask-saving' ? '正在保存蒙版…' : '应用透明蒙版';
  updateDock();
}
function dockTarget() { return currentJob ? (activeStates.has(currentJob.state) ? $('#cancel') : $('#reuse')) : $('#run'); }
function updateDock() {
  const target = dockTarget(), rect = target.getBoundingClientRect();
  $('#action-dock').hidden = !$('#workspace').classList.contains('active') || $('#mask-dialog').open || (!target.hidden && rect.top >= 0 && rect.bottom <= innerHeight-12);
  $('#quick-action').textContent = target.textContent; $('#quick-action').disabled = target.disabled;
  $('#quick-action').className = target.id === 'cancel' ? 'danger' : 'primary';
  $('#dock-title').textContent = currentJob ? (activeStates.has(currentJob.state) ? 'Hi3D 正在精修' : '继续调整？复用这次输入') : inputImage && inputMesh ? '输入已齐全，可以开始' : '下一步：准备图片和粗模型';
  $('#dock-help').textContent = !online ? '连接断开，请检查启动窗口' : pending ? '正在处理，请稍候…' : currentJob ? '原任务记录与文件会保留' : health?.model_checking ? '正在自动校验模型文件…' : '第一次建议使用 512 档和省显存模式';
}
$('#quick-action').onclick = () => dockTarget().click();
const dockObserver = new IntersectionObserver(updateDock);
for (const id of ['run','reuse','cancel']) dockObserver.observe($('#'+id));
window.addEventListener('resize', updateDock); window.addEventListener('scroll', updateDock, {passive:true});
$('#mask-dialog').addEventListener('close', updateDock);
function exportAvailability() {
  const files = currentJob?.available_files || [], completed = currentJob && !activeStates.has(currentJob.state);
  const refined = completed && files.includes('refined.glb'), coarse = completed && files.includes('coarse.glb');
  const source = $('#export-source'); source.options[0].disabled = !refined; source.options[1].disabled = !coarse;
  if (source.value === 'refined' && !refined && coarse) source.value = 'coarse';
  $('#export').disabled = !online || exportPending || Boolean(pending) || !(source.value === 'refined' ? refined : coarse);
  $('#export').textContent = exportPending ? '正在导出…' : '导出模型 ↓';
  for (const id of ['export-source','export-format','height']) $('#'+id).disabled = exportPending;
  $('#report').disabled = !online || !files.includes('report.json');
  $('#export-status').textContent = refined ? '精修结果与输入粗模均可导出' : coarse ? '此任务没有精修结果，可导出输入粗模型' : '完成后可导出精修版本';
}
function setView(id, url) {
  const view = $('#'+id);
  const empty = $('#'+id+'-empty');
  if (!url) {
    view.removeAttribute('src'); delete view.dataset.loading; delete view.dataset.loadError; empty.hidden = false;
    if (id === 'before') { empty.querySelector('b').textContent = '从一个粗模型开始'; empty.querySelector('p').textContent = '拖入模型，或点击上方「载入样例」'; }
  } else if (view.getAttribute('src') !== url) {
    view.dataset.loading = 'true'; delete view.dataset.loadError; empty.hidden = false;
    empty.querySelector('b').textContent = id === 'before' ? '正在加载粗模型预览…' : '正在加载精修结果预览…';
    view.setAttribute('src', url);
  }
}
function resetResult() {
  setView('after', null); $('#conditioning').hidden = true; $('#conditioning').removeAttribute('src');
  $('#dimension-link').hidden = true; $('#dimension-link').removeAttribute('href'); logsKey = '';
  for (const id of ['face-stat','time-stat','memory-stat','res-stat']) $('#'+id).textContent = '—';
  $('#progress').hidden = true; $('#elapsed').textContent = ''; $('#log').textContent = '还没有任务。';
  $('#after-title').textContent = '精修完成后，结果会出现在这里';
  $('#after-hint').textContent = '左右拖动旋转，滚轮缩放；两侧视角同步';
  $('#export-source').value = 'refined'; $('#height').value = '';
  $('#geometry-summary').className = ''; $('#geometry-summary').textContent = '上传粗模型后显示几何检查。';
  $('#geometry-info').textContent = '包括连通件、边界边与法向方向。自交、壁厚和实际打印强度需在专门软件中检查。';
}
function applyOptions(options) {
  for (const key of ['resolution','steps','seed','attention']) if (options[key] !== undefined) $('#'+key).value = options[key];
  if (options.offload !== undefined) $('#offload').checked = options.offload;
}
function renderDraft() {
  if (inputImage) { $('#image-thumb').src = inputImage.url; $('#image-thumb').hidden = false; }
  else { $('#image-thumb').hidden = true; $('#image-thumb').removeAttribute('src'); }
  $('#image-name').textContent = inputImage?.name || '未选择图片';
  $('#mesh-name').textContent = inputMesh?.name || '未选择粗模型';
  setView('before', inputMesh?.url); if (inputMesh?.geometry) geometry(inputMesh.geometry);
  $('#view-status').textContent = inputMesh ? '中性材质预览 · 拖动旋转，滚轮缩放' : '上传粗模型后可旋转查看';
  $('#progress-label').textContent = inputImage && inputMesh ? '输入已准备好，可以开始精修' : '先准备图片和粗模型';
  $('#stage-detail').textContent = '请确认图片和模型为同一物体，主体与朝向匹配。'; availability();
}
function draftWith(image, mesh) {
  epoch++; currentJob = null; shownJob = null; inputImage = image; inputMesh = mesh; resetResult(); renderDraft();
}
$('#new-task').onclick = () => { if (!inputBusy()) draftWith(null, null); };
async function upload(file, kind) {
  if (kind === 'image' && !/\.(png|jpe?g|webp)$/i.test(file.name)) throw new Error('参考图片请选择 PNG、JPG 或 WebP 文件。');
  if (kind === 'mesh' && !/\.(glb|ply|stl)$/i.test(file.name)) throw new Error('粗模型请选择 GLB、PLY 或 STL 文件。');
  const form = new FormData(); form.append('file', file);
  const item = await api('/api/assets', {method:'POST', body:form});
  if (item.kind !== kind) throw new Error('文件类型与输入栏不匹配。'); return item;
}
async function handleFile(file, kind) {
  if (!file || inputBusy()) return false;
  pending = 'upload'; availability();
  try {
    const item = await upload(file, kind);
    draftWith(kind === 'image' ? item : inputImage, kind === 'mesh' ? item : inputMesh);
    toast('输入已保存到本机。'); return true;
  } catch (error) { showError(error); return false; }
  finally { pending = null; availability(); }
}
for (const [id, kind] of [['image-file','image'],['mesh-file','mesh']]) {
  $('#'+id).onchange = async event => { const file = event.target.files[0]; event.target.value = ''; await handleFile(file, kind); };
}
for (const [id, kind] of [['image-drop','image'],['mesh-drop','mesh']]) {
  const el = $('#'+id);
  el.ondragover = event => { event.preventDefault(); if (!inputBusy()) el.classList.add('drag'); };
  el.ondragleave = () => el.classList.remove('drag');
  el.ondrop = event => { event.preventDefault(); el.classList.remove('drag'); if (inputBusy()) { toast('当前操作完成后可以更换输入。'); return; } void handleFile(event.dataTransfer.files[0], kind); };
}
$('#load-sample').onclick = async () => {
  if (inputBusy()) return; pending = 'sample'; availability();
  try {
    const items = [];
    for (const [name, kind, type] of [['sculpted-vase.png','image','image/png'],['sculpted-vase.glb','mesh','model/gltf-binary']]) {
      const blob = await api('/api/sample/'+name, {responseType:'blob'});
      items.push(await upload(new File([blob], name, {type}), kind));
    }
    draftWith(...items); toast('样例已载入。它用于熟悉流程，不是美术效果基准。');
  } catch (error) { showError(error); }
  finally { pending = null; availability(); }
};
function geometry(value) {
  const g = value, issues = g.boundary_edges || g.nonmanifold_edges || g.degenerate_faces;
  $('#geometry-summary').className = issues ? 'warning' : 'good';
  $('#geometry-summary').textContent = `${Number(g.faces).toLocaleString()} 个三角面 · ${g.components} 个连通件。${issues ? '检测到开放边界或异常几何，打印前请进一步检查。' : '当前几何检查未发现边界或非流形边；打印前仍需检查壁厚和自交。'}`;
  $('#geometry-info').textContent = `${Number(g.vertices).toLocaleString()} 顶点 · ${g.boundary_edges} 条边界边 · ${g.nonmanifold_edges} 条非流形边 · ${g.degenerate_faces ?? '未统计'} 个退化面 · 法向方向${g.winding_consistent ? '一致' : '需检查'} · ${g.watertight ? '封闭网格' : '非封闭网格'}。未检测：${(g.unchecked || ['自交','壁厚']).join('、')}。`;
}
for (const view of [$('#before'), $('#after')]) {
  view.addEventListener('load', () => {
    if (!view.loaded || !view.hasAttribute('src')) return;
    delete view.dataset.loading; delete view.dataset.loadError; $('#'+view.id+'-empty').hidden = true;
    for (const material of view.model.materials) {
      const pbr = material.pbrMetallicRoughness; pbr.setBaseColorFactor([.38,.37,.43,1]); pbr.setMetallicFactor(0); pbr.setRoughnessFactor(.8);
    }
  });
  view.addEventListener('error', () => { if (view.hasAttribute('src')) {
    view.dataset.loadError = 'true'; delete view.dataset.loading;
    const empty = $('#'+view.id+'-empty'); empty.hidden = false; empty.querySelector('b').textContent = '三维预览加载失败';
    toast('三维预览加载失败。请检查模型文件；仍可查看任务日志与导出文件。');
  } });
}
function syncView(source, target) {
  source.addEventListener('camera-change', event => {
    if (event.detail.source !== 'user-interaction' || !target.loaded) return;
    const c = source.getCameraOrbit(), s = source.getDimensions(), t = target.getDimensions();
    const ratio = Math.hypot(t.x,t.y,t.z) / Math.max(Math.hypot(s.x,s.y,s.z), 1e-12);
    target.cameraOrbit = `${c.theta}rad ${c.phi}rad ${c.radius * ratio}m`; target.fieldOfView = `${source.getFieldOfView()}deg`;
    const pan = source.getCameraTarget(), a = source.getBoundingBoxCenter(), b = target.getBoundingBoxCenter();
    target.cameraTarget = `${b.x+(pan.x-a.x)*ratio}m ${b.y+(pan.y-a.y)*ratio}m ${b.z+(pan.z-a.z)*ratio}m`;
  });
}
syncView($('#before'), $('#after')); syncView($('#after'), $('#before'));
$('#rotate').onchange = event => { for (const view of [$('#before'),$('#after')]) view.autoRotate = event.target.checked; };
$('#reset-camera').onclick = () => { for (const view of [$('#before'),$('#after')]) { view.cameraOrbit = '35deg 72deg auto'; view.fieldOfView = 'auto'; view.cameraTarget = 'auto'; } };
$('#run').onclick = async () => {
  if ($('#run').disabled) return;
  for (const id of ['steps','seed']) if (!$('#'+id).checkValidity()) { $('#advanced').open = true; $('#'+id).reportValidity(); $('#'+id).focus(); return; }
  pending = 'submitting'; availability();
  try {
    const job = await post('/api/jobs', {image:inputImage.id, mesh:inputMesh.id, resolution:+$('#resolution').value, steps:+$('#steps').value, seed:+$('#seed').value, offload:$('#offload').checked, attention:$('#attention').value});
    epoch++; inputImage = null; inputMesh = null; renderJob(job); await loadArtifacts(job, epoch); await refresh();
  } catch (error) { showError(error); }
  finally { pending = null; availability(); }
};
async function cancelJob(id) {
  if (!id || !online || pending) return; pending = 'cancel'; availability();
  try { const job = await post(`/api/jobs/${id}/cancel`); if (job && currentJob?.id === id) renderJob({...currentJob, ...job}); await refresh(); }
  catch (error) { showError(error); } finally { pending = null; availability(); }
}
$('#cancel').onclick = () => cancelJob(currentJob?.id);
$('#cancel-active').onclick = () => cancelJob(health?.active);
$('#show-active').onclick = () => { if (health?.active) void openJob(health.active); };
function duration(seconds) { const s = Math.max(0, Math.round(seconds)); return s >= 60 ? `${Math.floor(s/60)} 分 ${s%60} 秒` : `${s} 秒`; }
function renderJob(job) {
  const changed = shownJob !== job.id; currentJob = job;
  if (changed) {
    resetResult(); shownJob = job.id; inputImage = null; inputMesh = null; applyOptions(job.options);
  }
  setView('before', hasJobFile(job,'coarse.glb') ? `/api/jobs/${job.id}/files/coarse.glb` : null);
  $('#image-thumb').hidden = !hasJobFile(job,'input.png');
  const imageUrl = `/api/jobs/${job.id}/files/input.png`;
  if (hasJobFile(job,'input.png') && $('#image-thumb').getAttribute('src') !== imageUrl) $('#image-thumb').src = imageUrl;
  if (!hasJobFile(job,'input.png')) $('#image-thumb').removeAttribute('src');
  $('#image-name').textContent = hasJobFile(job,'input.png') ? '此任务的参考图片' : '参考图片文件已缺失';
  $('#mesh-name').textContent = hasJobFile(job,'coarse.glb') ? '此任务的输入粗模型' : '输入粗模型文件已缺失';
  const displayStage = {succeeded:'complete',failed:'failed',cancelled:'cancelled',interrupted:'interrupted'}[job.state] || job.stage;
  $('#progress-label').textContent = job.state === 'cancelling' ? '正在停止任务并释放显存…' : stageLabels[displayStage] || stateLabels[job.state] || job.state;
  $('#stage-detail').textContent = job.message || '';
  $('#elapsed').textContent = duration((activeStates.has(job.state) ? Date.now()/1000 : job.updated || job.created) - job.created);
  const event = job.event || {};
  if (job.stage === 'sampling' && job.state === 'running' && event.step && event.num_inference_steps) {
    $('#progress').hidden = false; $('#progress').max = event.num_inference_steps; $('#progress').value = event.step;
    $('#stage-detail').textContent = `第 ${event.pass_index+1} / ${event.num_passes} 轮 · 本轮采样 ${event.step} / ${event.num_inference_steps}。这是当前轮进度，后续还需解码与保存。`;
  } else $('#progress').hidden = true;
  $('#view-status').textContent = '查看历史任务 · 复用后可修改输入与参数';
  if (job.state === 'succeeded' && job.report) {
    const report = job.report; setView('after', hasJobFile(job,'refined.glb') ? `/api/jobs/${job.id}/files/refined.glb` : null);
    $('#face-stat').textContent = report.geometry.faces.toLocaleString(); $('#time-stat').textContent = duration(report.seconds);
    $('#memory-stat').textContent = `${(report.peak_allocated_mb/1024).toFixed(2)} GiB`; $('#res-stat').textContent = report.actual_resolution;
    $('#view-status').textContent = report.compatibility_port ? 'Windows SDPA 兼容路径 · 两侧视角同步' : 'Hi3D 模型与采样路径 · 两侧视角同步'; geometry(report.geometry);
    if (!hasJobFile(job,'refined.glb')) {
      $('#after-title').textContent = '此任务的精修文件已缺失'; $('#after-hint').textContent = '仍可查看保留的报告；输入齐全时可复用后重试';
      $('#stage-detail').textContent = '任务曾完成，但当前目录找不到精修文件。下方指标来自保存的运行报告。';
    }
  } else {
    setView('after', null);
    $('#after-title').textContent = activeStates.has(job.state) ? 'Hi3D 正在精修，请稍候' : '这次任务没有生成精修结果';
    $('#after-hint').textContent = activeStates.has(job.state) ? '可在下方查看当前阶段与日志' : '展开日志查看原因；复用输入后可以重试';
    if (!activeStates.has(job.state)) $('#stage-detail').textContent = (job.message || stateLabels[job.state]) + '。输入与日志已保留，可复用后重试。';
  }
  availability();
}
async function loadArtifacts(job, selectedEpoch) {
  const files = job.available_files || [], key = `${job.id}:${job.updated}:${files.join(',')}`;
  if (files.includes('conditioning.png')) { $('#conditioning').src = `/api/jobs/${job.id}/files/conditioning.png`; $('#conditioning').hidden = false; }
  else { $('#conditioning').hidden = true; $('#conditioning').removeAttribute('src'); }
  if (!files.includes('worker.log')) { $('#log').textContent = '日志文件缺失或尚未生成。'; return; }
  if (logsKey === key) return;
  const text = await api(`/api/jobs/${job.id}/files/worker.log`, {responseType:'text',timeoutMs:10000});
  if (epoch === selectedEpoch && currentJob?.id === job.id) { $('#log').textContent = text.slice(-24000) || '任务日志暂时为空。'; logsKey = key; }
}
async function openJob(id) {
  if (pending) return; pending = 'opening'; const selectedEpoch = ++epoch; availability();
  try {
    const job = await api('/api/jobs/'+id); if (epoch !== selectedEpoch) return;
    renderJob(job); page('workspace'); await loadArtifacts(job, selectedEpoch);
  } catch (error) { showError(error); } finally { pending = null; availability(); }
}
$('#reuse').onclick = async () => {
  if (!currentJob || inputBusy()) return; pending = 'reuse'; availability();
  try { const result = await post(`/api/jobs/${currentJob.id}/reuse`); draftWith(result.image, result.mesh); applyOptions(result.options); toast('输入和参数已复制到新任务。调整后即可再次精修，原记录保留。'); }
  catch (error) { showError(error); } finally { pending = null; availability(); }
};
$('#report').onclick = () => { if (currentJob && !$('#report').disabled) window.open(`/api/jobs/${currentJob.id}/files/report.json`, '_blank', 'noopener'); };
$('#export-source').onchange = exportAvailability;
for (const id of ['export-source','export-format','height']) $('#'+id).addEventListener('change', () => { $('#dimension-link').hidden = true; });
$('#export').onclick = async () => {
  if ($('#export').disabled || !$('#height').reportValidity()) return;
  exportPending = true; const jobId = currentJob.id, selectedEpoch = epoch, format = $('#export-format').value; availability();
  try {
    const result = await post(`/api/jobs/${jobId}/export`, {source:$('#export-source').value, format, height_mm:$('#height').value ? +$('#height').value : null});
    const link = document.createElement('a'); link.href = result.url; link.download = `Hi3D-${format === 'glb' ? 'model' : 'geometry'}.${format}`; link.click();
    if (epoch === selectedEpoch) { $('#dimension-link').href = result.dimensions; $('#dimension-link').hidden = false; }
    toast('模型已导出。尺寸说明可一并下载；打印前请在切片器中核对单位。');
  } catch (error) { showError(error); } finally { exportPending = false; availability(); }
};
let historyEpoch = 0;
async function history() {
  const selectedEpoch = ++historyEpoch; $('#refresh-history').disabled = true;
  $('#history-list').textContent = '正在读取任务记录…';
  try {
    const jobs = await api('/api/jobs'); if (historyEpoch !== selectedEpoch) return;
    const list = $('#history-list'); list.replaceChildren();
    if (!jobs.length) { list.textContent = '还没有任务。回到几何工作台，载入样例开始第一次精修。'; return; }
    for (const job of jobs) {
      const row = document.createElement('div'); row.className = 'history-item'; row.dataset.jobId = job.id; const info = document.createElement('div'); info.className = 'grow';
      const title = document.createElement('b'); title.textContent = `Hi3D ${job.options?.resolution || '—'} · ${job.options?.steps || '—'} 步 · 种子 ${job.options?.seed ?? '—'}`;
      const time = document.createElement('time'); time.textContent = new Date(job.created*1000).toLocaleString();
      const description = document.createElement('p'); description.textContent = job.state === 'succeeded' ? '打开查看结果、导出或复用输入' : '打开查看日志与保留的输入'; info.append(title, document.createElement('br'), time, description);
      const state = document.createElement('span'); state.className = 'state-pill'; state.textContent = stateLabels[job.state] || job.state;
      const button = document.createElement('button'); button.textContent = '打开任务 →'; button.onclick = () => openJob(job.id); row.append(info, state, button); list.append(row);
    }
  } catch (error) { showError(error); $('#history-list').textContent = '无法读取记录，请检查连接后点击刷新。'; }
  finally { $('#refresh-history').disabled = false; }
}
$('#refresh-history').onclick = history;
function renderHealth() {
  const recoveryBlocked = Boolean(health.recovery_blocked), recoveryErrors = health.recovery_errors || [];
  $('#recovery-alert').hidden = !recoveryBlocked && !recoveryErrors.length;
  $('#recovery-alert').textContent = recoveryBlocked ? `检测到重启前仍在运行的工作进程，已暂停新任务。请等待进程退出；涉及进程：${(health.recovery_workers || []).map(item => item.pid).join('、')}。` : `有 ${recoveryErrors.length} 条历史记录无法读取，已跳过；其他记录仍可使用。详情：${recoveryErrors.map(item => typeof item === 'string' ? item : JSON.stringify(item)).join('；')}`;
  $('#connection-alert').hidden = online; $('#health-badge').classList.toggle('offline', !online);
  $('#health-badge').textContent = !online ? '● 连接已断开' : health.model_checking ? '◌ 正在校验模型…' : health.operation === 'diagnosing' ? '◌ 正在检查 GPU…' : health.active ? '● 本地精修运行中' : health.models.ready ? '● 本地模型已就绪' : '○ 模型文件待准备';
  const cards = $('#model-cards'); cards.replaceChildren();
  for (const [key, component] of Object.entries(health.models.components)) {
    const card = document.createElement('div'); card.className = 'model-card'; const title = document.createElement('h3'); title.textContent = {hi3d:'Hi3D V1.0',dino:'DINOv2 Large',clip:'CLIP ViT-L/14'}[key] || key;
    const repo = document.createElement('small'); repo.textContent = component.repo; const license = document.createElement('p'); license.className = 'hint'; license.textContent = component.license;
    const state = document.createElement('div'); state.className = 'model-state'; state.textContent = component.message || (component.verified ? '✓ 完整校验通过' : '需要准备或校验'); card.append(title, repo, license, state); cards.append(card);
  }
  $('#download-state').textContent = health.verification_error ? '校验出错：'+health.verification_error : health.model_checking ? '正在完整校验文件，请稍候…' : health.models.download.message;
  if (health.diagnostics && pending !== 'diagnose') $('#diagnostics').textContent = JSON.stringify(health.diagnostics, null, 2);
  availability();
}
async function refresh() {
  const session = sessionReady ? null : await api('/api/session');
  if (session) { token = session.token; sessionReady = true; if (!sessionDefaultsApplied && !currentJob) $('#attention').value = session.default_attention || 'flash'; sessionDefaultsApplied = true; document.querySelectorAll('.app-version').forEach(el => { el.textContent = session.version; }); }
  health = await api('/api/health'); online = true; renderHealth();
}
async function tick() {
  if (pollBusy) return; pollBusy = true;
  try {
    try { await refresh(); } catch (error) { unavailable(error); return; }
    const id = currentJob?.id, selectedEpoch = epoch;
    if (id) {
      let job;
      try { job = await api('/api/jobs/'+id); }
      catch (error) {
        if ([400,404].includes(error.status) && epoch === selectedEpoch && currentJob?.id === id && health.active !== id) { draftWith(null,null); toast('此任务记录已不存在或不可读取，已返回新任务。原文件与其他记录不受影响。'); return; }
        throw error;
      }
      if (epoch === selectedEpoch && currentJob?.id === id) { renderJob(job); await loadArtifacts(job, selectedEpoch); }
    } else if (health.active && !pending && !inputImage && !inputMesh) {
      const job = await api('/api/jobs/'+health.active);
      if (epoch === selectedEpoch && !currentJob && !pending) { renderJob(job); await loadArtifacts(job, selectedEpoch); }
    }
  } catch (error) {
    if (error.network) unavailable(error);
    else { $('#stage-detail').textContent = error.message; }
  } finally { pollBusy = false; }
}
$('#reconnect').onclick = () => { sessionReady = false; void tick(); };
for (const [id, kind, action] of [['download-models','download',async () => { const result = await post('/api/models/download'); toast(result.started ? '开始下载 Hi3D 与必需编码器，请在此查看进度。' : '已有下载进程，请等待完成。'); }],['verify-models','verify',async () => { const result = await post('/api/models/verify'); toast(result.ok ? '全部模型文件 SHA256 校验通过。' : '校验未通过：'+result.errors.join('，')); }],['diagnose','diagnose',async () => { $('#diagnostics').textContent = '正在检查 GPU 基础内核，首次编译可能需要几分钟…'; const result = await post('/api/diagnostics'); $('#diagnostics').textContent = JSON.stringify(result,null,2); toast(result.kernel_probe === 'passed' ? 'GPU 基础内核检查通过，可以尝试精修。' : 'GPU 检查发现问题，请查看诊断详情。'); }]]) {
  $('#'+id).onclick = async () => { if ($('#'+id).disabled) return; pending = kind; availability(); try { await action(); await refresh(); } catch (error) { showError(error); } finally { pending = null; availability(); } };
}
// The background tool edits alpha locally without another model.
let original = null, drawing = false, last = null;
const canvas = $('#mask-canvas'), ctx = canvas.getContext('2d', {willReadFrequently:true});
$('#mask-open').onclick = async () => {
  if (!inputImage || inputBusy()) return; pending = 'mask-loading'; availability();
  try {
    const image = new Image(); image.src = `/api/assets/${inputImage.id}/image.png`; await image.decode(); original = image;
    canvas.width = image.width; canvas.height = image.height; ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(image,0,0); $('#mask-dialog').showModal(); updateDock();
  } catch { toast('参考图片无法读取，请重新选择图片后再试。'); } finally { pending = null; availability(); }
};
$('#mask-close').onclick = () => $('#mask-dialog').close();
$('#mask-reset').onclick = () => { if (original) { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(original,0,0); } };
function point(event) { const r = canvas.getBoundingClientRect(); return [(event.clientX-r.left)*canvas.width/r.width,(event.clientY-r.top)*canvas.height/r.height]; }
function paint(event) {
  if (!drawing) return; const p = point(event), size = +$('#brush-size').value*canvas.width/canvas.getBoundingClientRect().width;
  if ($('#brush-mode').value === 'erase') {
    ctx.globalCompositeOperation = 'destination-out'; ctx.lineCap = 'round'; ctx.lineWidth = size; ctx.beginPath(); ctx.moveTo(...(last || p)); ctx.lineTo(...p); ctx.stroke(); ctx.globalCompositeOperation = 'source-over';
  } else { ctx.save(); ctx.beginPath(); ctx.arc(...p,size/2,0,Math.PI*2); ctx.clip(); ctx.clearRect(p[0]-size/2,p[1]-size/2,size,size); ctx.drawImage(original,0,0); ctx.restore(); }
  last = p;
}
canvas.onpointerdown = event => { drawing = true; last = null; canvas.setPointerCapture(event.pointerId); paint(event); };
canvas.onpointermove = paint; canvas.onpointerup = canvas.onpointercancel = () => { drawing = false; last = null; };
$('#mask-apply').onclick = async () => {
  if (inputBusy()) return;
  pending = 'mask-saving'; availability();
  try {
    const selectedEpoch = epoch, imageId = inputImage.id;
    const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/png'));
    if (!blob) throw new Error('无法保存蒙版，请缩小图片后重试。');
    const item = await upload(new File([blob],'reference-masked.png',{type:'image/png'}),'image');
    if (epoch === selectedEpoch && inputImage?.id === imageId) { draftWith(item,inputMesh); $('#mask-dialog').close(); toast('透明蒙版已保存。'); }
  }
  catch (error) { showError(error); }
  finally { pending = null; availability(); }
};
availability(); void tick(); setInterval(tick, 2500);
