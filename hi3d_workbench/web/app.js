const $ = selector => document.querySelector(selector);
let token = $('meta[name="hi3d-token"]').content;
let inputImage = null, inputMesh = null, currentJob = null, health = null;
let online = false, pending = null, exportPending = false, pollBusy = false;
let epoch = 0, shownJob = null, sessionReady = false, sessionDefaultsApplied = false, toastTimer, logsKey = '';
let selectedMode = 'image_to_asset', previewManual = false, exportSourceManual = false;
let comparisonJob = null, comparisonCandidates = [], comparisonsFor = '';
let workbenchStopped = false;
const activeStates = new Set(['running', 'queued', 'cancelling']);
const stageLabels = {queued:'任务已保存，等待启动', loading:'正在加载本地模型', conditioning:'正在处理参考图片', foreground:'正在处理图片背景', voxelizing:'正在准备粗模型', sampling:'正在生成几何细节', decoding:'正在解码几何', extracting:'正在提取网格', postprocessing:'正在整理几何', exporting:'正在保存结果', complete:'任务完成，可以预览导出', failed:'任务失败，已有阶段文件与日志已保留', cancelled:'任务已停止', interrupted:'任务已中断'};
const stateLabels = {succeeded:'已完成', failed:'失败', running:'运行中', queued:'等待启动', cancelled:'已停止', interrupted:'已中断', cancelling:'停止中'};
Object.assign(stageLabels, {initializing:'初始化推理运行时', encoding_mesh:'准备法线与 UV', encoding_reference:'编码参考条件', baking:'烘焙 PBR 材质'});
const modeLabels = {image_to_asset:'图片建模', texture:'已有模型贴图', refine:'模型精修'};
const modelVersions = ['textured','refined','coarse','generated'];
const versionLabels = {textured:'PBR 材质结果', refined:'Hi3D 精修几何', coarse:'粗模型', generated:'原始生成网格'};
const qualityPresets = {
  light: {geometry:'hunyuan_mini',geometryResolution:256,geometrySteps:5,texture:'trellis2',textureResolution:512,textureSteps:12,textureSize:2048},
  balanced: {geometry:'trellis2',geometryResolution:512,geometrySteps:12,texture:'trellis2',textureResolution:1024,textureSteps:12,textureSize:4096},
  detail: {geometry:'trellis2',geometryResolution:1024,geometrySteps:12,texture:'ultratex_int8',textureResolution:1024,textureSteps:25,textureSize:4096}
};
function needsMesh(mode = selectedMode) { return mode !== 'image_to_asset'; }
function wantsRefine(options = null) { return options ? (options.mode || 'refine') === 'refine' || options.mode === 'image_to_asset' && options.refine_geometry === true : selectedMode === 'refine' || selectedMode === 'image_to_asset' && $('#refine-geometry').checked; }
function wantsTexture(options = null) { return options ? options.mode === 'texture' || Boolean(options.generate_texture ?? options.mode === 'image_to_asset') : selectedMode === 'texture' || $('#generate-texture').checked; }
function inputsReady() { return Boolean(inputImage && (!needsMesh() || inputMesh)); }
function downloadingModels() { return health?.models?.download?.state === 'downloading' || health?.pipeline_models?.download?.state === 'downloading'; }
function checkingModels() { return Boolean(health?.model_checking || health?.pipeline_model_checking || health?.pipeline_models?.checking); }
function renderHealthBadge() {
  if (!health || !online) return;
  $('#health-badge').textContent = downloadingModels() ? '◌ 正在下载组件…' : checkingModels() ? '◌ 正在校验模型…' : health.operation === 'diagnosing' ? '◌ 正在检查 GPU…' : health.active ? '● 本地任务运行中' : !missingModels().length ? '● 所选流程已就绪' : '○ 所选流程待准备';
}
function missingModels() {
  const missing = [], pipeline = health?.pipeline_models;
  if (wantsRefine() && !health?.models?.ready) missing.push('Hi3D 核心');
  const geometry = pipeline?.engines?.coarse?.[$('#geometry-engine').value];
  const texture = pipeline?.engines?.texture?.[$('#texture-engine').value];
  if (selectedMode === 'image_to_asset' && !(geometry?.ready ?? ($('#geometry-engine').value === 'hunyuan_mini' && pipeline?.coarse_ready))) missing.push(geometry?.name || '图片建模组件');
  if (wantsTexture() && !(texture?.ready ?? ($('#texture-engine').value === 'trellis2' && pipeline?.texture_ready))) missing.push(texture?.name || 'PBR 贴图组件');
  if ((selectedMode === 'image_to_asset' || wantsTexture()) && $('#background').value === 'auto' && pipeline?.background_ready === false && inputImage?.has_transparency !== true) missing.push('自动前景组件');
  if ((selectedMode === 'image_to_asset' || wantsTexture()) && !pipeline?.runtime?.ready) missing.push('扩展运行环境');
  return missing;
}
function syncEngines(resetStage = null) {
  for (const [stage, prefix] of [['coarse','geometry'],['texture','texture']]) {
    const choice = health?.pipeline_models?.engines?.[stage]?.[$('#'+prefix+'-engine').value];
    if (!choice) continue;
    const resolutions = $('#'+prefix+'-resolution'), current = +resolutions.value;
    if (resolutions.dataset.values !== JSON.stringify(choice.resolutions)) {
      resolutions.replaceChildren(...choice.resolutions.map(value => new Option(String(value),String(value))));
      resolutions.dataset.values = JSON.stringify(choice.resolutions);
      resolutions.value = String(choice.resolutions.includes(current) ? current : choice.default_resolution);
    }
    if (resetStage === stage) {
      resolutions.value = String(choice.default_resolution);
      $('#'+prefix+'-steps').value = String(choice.default_steps);
    }
    $('#'+prefix+'-engine-hint').textContent = `${choice.description} ${choice.ready ? '✓ 已就绪' : choice.runtime_ready ? '需下载所选模型。' : '需更新扩展运行环境并准备模型。'}`;
    if (stage === 'coarse') $('#geometry-steps').readOnly = choice.id === 'hunyuan_mini';
  }
}
function syncModeUI() {
  const imageMode = selectedMode === 'image_to_asset', texture = wantsTexture(), refine = wantsRefine();
  document.querySelectorAll('[data-mode]').forEach(button => { const active = button.dataset.mode === selectedMode; button.classList.toggle('active',active); button.setAttribute('aria-pressed',String(active)); });
  $('#mesh-input-group').hidden = imageMode; $('#ready-mesh').hidden = imageMode;
  $('#refine-geometry-field').hidden = !imageMode; $('#generate-texture-field').hidden = selectedMode === 'texture';
  $('#refine-settings').hidden = !refine; $('#texture-settings').hidden = !texture;
  $('#geometry-settings').hidden = !imageMode;
  $('#trellis-cleanup-field').hidden = !imageMode || $('#geometry-engine').value !== 'trellis2';
  $('#hi3d-steps-field').hidden = !refine; $('#hi3d-attention-field').hidden = !refine;
  $('#advanced-fields').classList.toggle('seed-only',!refine);
  $('#advanced-summary').textContent = refine ? '高级设置 · 种子 / Hi3D 步数与兼容模式' : '高级设置 · 随机种子';
  $('#hi3d-advanced-hint').textContent = refine ? '步数与 Attention 只作用于 Hi3D 几何精修，不改变图片建模或贴图阶段的采样设置。Hi3D 默认 25 步；相同种子便于对比，Windows 兼容路径的数值与效果需单独评价。' : '当前流程不运行 Hi3D，已隐藏精修专用的步数与 Attention。随机种子仍用于当前生成流程；图片建模与贴图使用各自的采样设置。';
  $('#input-step-hint').textContent = imageMode ? '一张主体完整的参考图片' : '参考图片 + 同物体模型';
  $('#run-step-title').textContent = modeLabels[selectedMode];
  $('#mode-description').textContent = imageMode ? '默认先生成粗模型并添加 PBR 贴图。Hi3D 精修可选，会重建几何；卡通可先保形，再按需尝试。' : selectedMode === 'texture' ? '上传参考图片与已有模型，保留几何并生成 PBR 材质；使用 GLB 导出贴图结果。' : '用 Hi3D 重建已有粗模型并补充几何细节，可选择在精修后继续生成 PBR 贴图。';
  $('#input-help').textContent = imageMode ? '图片建模不需要粗模型。图片尽量背景简单、主体完整；自动处理背景或手动清理透明区域后再开始。' : '从建模软件导出 GLB、PLY 或 STL。图片与模型应为同一物体，主体与朝向匹配；贴图过程不会替代几何精修。';
}
function stageName(stage = '') {
  for (const [prefix,label] of [['coarse_','图片建模'],['texture_','PBR 贴图']]) if (stage.startsWith(prefix)) return `${label} · ${stageLabels[stage.slice(prefix.length)] || '正在处理'}`;
  return stageLabels[stage] || stage || '正在处理';
}
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
  const text = {workspace:['从一张图片，开始 3D 创作。','图片建模、Hi3D 几何精修与 PBR 贴图，在本地逐阶段完成，保留中间结果。'],history:['每一次尝试，都保留下来。','打开旧任务查看阶段结果，或复用输入继续调整。'],models:['准备好环境，再安心创作。','按流程分别准备 Hi3D 核心、图片建模与 PBR 贴图组件。']};
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
  const busy = inputBusy(), checking = checkingModels(), downloading = downloadingModels();
  const isHistory = Boolean(currentJob), missing = missingModels(), ready = !missing.length;
  syncModeUI();
  $('#run').disabled = busy || checking || downloading || !ready || !inputsReady() || isHistory;
  $('#run').textContent = pending === 'submitting' ? '正在校验文件并提交…' : `开始${modeLabels[selectedMode]} →`;
  for (const id of ['image-file','mesh-file','load-sample','new-task']) $('#'+id).disabled = busy;
  $('#load-sample').textContent = pending === 'sample' ? '正在载入样例输入…' : '载入样例，熟悉流程 →';
  $('#mask-open').disabled = busy || !inputImage;
  $('#mask-apply').disabled = busy;
  for (const id of ['resolution','steps','seed','offload','attention','background','refine-geometry','generate-texture','geometry-engine','geometry-resolution','geometry-steps','trellis-cleanup','texture-engine','texture-resolution','texture-steps','texture-size']) $('#'+id).disabled = busy || isHistory;
  document.querySelectorAll('[data-quality-preset]').forEach(button => {
    button.disabled=busy || isHistory;
    const p=qualityPresets[button.dataset.qualityPreset];
    const selected=$('#generate-texture').checked && $('#geometry-engine').value===p.geometry && +$('#geometry-resolution').value===p.geometryResolution && +$('#geometry-steps').value===p.geometrySteps && $('#texture-engine').value===p.texture && +$('#texture-resolution').value===p.textureResolution && +$('#texture-steps').value===p.textureSteps && +$('#texture-size').value===p.textureSize;
    button.setAttribute('aria-pressed',String(selected));button.classList.toggle('active',selected);
  });
  for (const id of ['steps','attention']) $('#'+id).disabled = busy || isHistory || !wantsRefine();
  document.querySelectorAll('[data-mode]').forEach(button => { button.disabled = busy || isHistory; });
  $('#reuse').hidden = !isHistory || activeStates.has(currentJob.state);
  $('#reuse').disabled = busy || !hasJobFile(currentJob,'input.png') || needsMesh(currentJob?.options?.mode || 'refine') && !hasJobFile(currentJob,'coarse.glb');
  $('#reuse').textContent = pending === 'reuse' ? '正在准备可编辑输入…' : '复用这次输入和参数';
  $('#cancel').hidden = !currentJob || !activeStates.has(currentJob.state);
  $('#cancel').disabled = !online || Boolean(pending) || currentJob?.state === 'cancelling';
  $('#active-job').hidden = !health?.active;
  $('#active-label').textContent = currentJob && currentJob.id === health?.active ? `当前任务：${modeLabels[selectedMode]} · ${stateLabels[currentJob.state] || '运行中'}` : '有一个任务正在运行';
  $('#show-active').disabled = !online || Boolean(pending);
  $('#cancel-active').disabled = !online || Boolean(pending) || Boolean(currentJob && currentJob.id === health?.active && currentJob.state === 'cancelling');
  let help = !online ? '工作台连接已断开，恢复连接后可以继续。' : pending === 'submitting' ? '正在校验所需模型与输入，请等待。提交期间不会重复创建任务。' : pending ? '正在处理，请稍候…' : health?.active ? '任务运行中，可在上方查看或停止。' : health?.recovery_blocked ? '重启前的工作进程仍在退出，请稍候。' : health?.operation ? '正在检查 GPU 或准备任务，请稍候。' : downloading ? '正在下载组件，期间暂停新任务；可在「模型与环境」查看进度。' : checking ? '正在校验本地模型文件，请稍候。' : isHistory ? '当前正在查看记录。复用输入后可更改流程与参数，中间模型也可预览和导出。' : !ready ? `尚需准备：${missing.join('、')}。请前往「模型与环境」补齐并校验。` : !inputsReady() ? needsMesh() ? '准备一张参考图片和同物体模型，或点击「载入样例」。' : '准备一张参考图片即可开始，或点击「载入样例」。' : '输入与所需组件已就绪。先使用默认设置；可随时停止并保留已完成的阶段。';
  $('#run-help').textContent = help;
  for (const [id, ok, name] of [['ready-image', Boolean(inputImage || currentJob && hasJobFile(currentJob,'input.png')), '参考图片'], ['ready-mesh', Boolean(inputMesh || currentJob && hasJobFile(currentJob,'coarse.glb')), '模型'], ['ready-models', online && ready && !downloading && !checking, '所需模型与运行环境']]) {
    $('#'+id).textContent = `${ok ? '✓' : '○'} ${name}`; $('#'+id).classList.toggle('ready', ok);
  }
  const modelBusy = !online || Boolean(pending) || gpuBusy() || checking || downloading;
  for (const id of ['download-models','verify-models','diagnose','download-pipeline-all','download-pipeline-coarse','download-pipeline-texture','verify-pipeline-models']) $('#'+id).disabled = modelBusy;
  document.querySelectorAll('[data-engine-download]').forEach(button => { button.disabled = modelBusy; });
  $('#refresh-comparisons').disabled = !online || Boolean(pending) || !currentJob;
  $('#comparison-job').disabled = !online || Boolean(pending) || !currentJob;
  $('#verify-models').textContent = checking || pending === 'verify' ? '正在完整校验…' : '重新完整校验';
  $('#verify-pipeline-models').textContent = pending === 'pipeline-verify' || health?.pipeline_model_checking || health?.pipeline_models?.checking ? '正在校验扩展组件…' : '校验扩展组件';
  $('#diagnose').textContent = health?.operation === 'diagnosing' || pending === 'diagnose' ? '正在检查 GPU…' : '检查 GPU 是否可用';
  const phase = currentJob && !activeStates.has(currentJob.state) && modelVersions.some(version => hasJobFile(currentJob,version+'.glb')) ? 3 : inputsReady() || currentJob ? 2 : 1;
  for (const [index, id] of ['step-input','step-run','step-export'].entries()) {
    $('#'+id).classList.toggle('current', phase === index+1); $('#'+id).classList.toggle('done', phase > index+1);
  }
  exportAvailability();
  $('#mask-apply').textContent = pending === 'mask-saving' ? '正在保存蒙版…' : '应用透明蒙版';
  renderHealthBadge();
  updateDock();
}
function dockTarget() { return currentJob ? (activeStates.has(currentJob.state) ? $('#cancel') : $('#reuse')) : $('#run'); }
function updateDock() {
  const target = dockTarget(), rect = target.getBoundingClientRect();
  $('#action-dock').hidden = !$('#workspace').classList.contains('active') || $('#mask-dialog').open || (!target.hidden && rect.top >= 0 && rect.bottom <= innerHeight-12);
  $('#quick-action').textContent = target.textContent; $('#quick-action').disabled = target.disabled;
  $('#quick-action').className = target.id === 'cancel' ? 'danger' : 'primary';
  $('#dock-title').textContent = currentJob ? (activeStates.has(currentJob.state) ? `${modeLabels[selectedMode]}正在运行` : '继续调整？复用这次输入') : inputsReady() ? '输入已齐全，可以开始' : needsMesh() ? '下一步：准备图片和模型' : '下一步：准备参考图片';
  $('#dock-help').textContent = !online ? '连接断开，请检查启动窗口' : pending ? '正在处理，请稍候…' : currentJob ? '原任务记录与阶段文件会保留' : downloadingModels() ? '模型下载中，暂停新任务' : checkingModels() ? '正在校验模型文件…' : `当前流程：${modeLabels[selectedMode]}`;
}
$('#quick-action').onclick = () => dockTarget().click();
const dockObserver = new IntersectionObserver(updateDock);
for (const id of ['run','reuse','cancel']) dockObserver.observe($('#'+id));
window.addEventListener('resize', updateDock); window.addEventListener('scroll', updateDock, {passive:true});
$('#mask-dialog').addEventListener('close', updateDock);
function exportAvailability() {
  const files = currentJob?.available_files || [], completed = currentJob && !activeStates.has(currentJob.state);
  const versions = modelVersions.filter(version => completed && files.includes(version+'.glb'));
  const source = $('#export-source'); for (const option of source.options) option.disabled = !versions.includes(option.value);
  if ((!exportSourceManual || !versions.includes(source.value)) && versions.length) source.value = versions[0];
  if (!versions.length) source.value = '';
  const textured = source.value === 'textured', format = $('#export-format');
  for (const option of format.options) option.disabled = textured && option.value !== 'glb';
  if (textured) format.value = 'glb';
  $('#export').disabled = !online || exportPending || Boolean(pending) || !versions.includes(source.value);
  $('#export').textContent = exportPending ? '正在导出…' : '导出模型 ↓';
  for (const id of ['export-source','export-format','height-axis','height']) $('#'+id).disabled = exportPending || !versions.length;
  $('#report').disabled = !online || !files.includes('report.json');
  $('#export-status').textContent = versions.length ? `可导出：${versions.map(version => versionLabels[version]).join(' / ')}` : activeStates.has(currentJob?.state) ? '任务结束后可导出已保留的模型' : '有阶段模型后即可导出';
  $('#export-material-hint').textContent = !versions.length ? '阶段文件生成且任务结束后可选择导出。PBR 材质使用 GLB；STL / PLY 仅包含几何。' : textured ? '当前版本包含 PBR 材质，仅支持 GLB。若需 STL / PLY，请选择粗模或精修几何；这些格式不包含贴图。' : '当前导出几何版本。若已有 PBR 结果，选择「PBR 材质结果 + GLB」可保留材质。STL / PLY 不含贴图。';
}
function setView(id, url) {
  const view = $('#'+id);
  const empty = $('#'+id+'-empty');
  if (!url) {
    view.removeAttribute('src'); delete view.dataset.loading; delete view.dataset.loadError; empty.hidden = false;
    if (id === 'before') { empty.querySelector('b').textContent = '粗模型会出现在这里'; empty.querySelector('p').textContent = '图片建模会自动生成；其他模式请上传模型'; }
  } else if (view.getAttribute('src') !== url) {
    view.dataset.loading = 'true'; delete view.dataset.loadError; empty.hidden = false;
    empty.querySelector('b').textContent = id === 'before' ? '正在加载粗模型预览…' : '正在加载阶段结果预览…';
    view.setAttribute('src', url);
  }
}
function resetResult() {
  comparisonJob = null; comparisonCandidates = []; comparisonsFor = '';
  $('#comparison-job').replaceChildren(new Option('当前任务的阶段模型',''));
  $('#comparison-version').replaceChildren(new Option('粗模型','coarse'));
  $('#comparison-hint').textContent = '可从历史任务选择其他引擎的结果，使用同步视角对比。';
  setView('after', null); $('#conditioning').hidden = true; $('#conditioning').removeAttribute('src');
  $('#dimension-link').hidden = true; $('#dimension-link').removeAttribute('href'); logsKey = '';
  for (const id of ['face-stat','time-stat','memory-stat','res-stat']) $('#'+id).textContent = '—';
  $('#progress').hidden = true; $('#elapsed').textContent = ''; $('#log').textContent = '还没有任务。';
  $('#after-title').textContent = '阶段结果会出现在这里';
  $('#after-hint').textContent = '左右拖动旋转，滚轮缩放；两侧视角同步';
  previewManual = false; exportSourceManual = false;
  $('#result-preview').value = ''; $('#result-preview').disabled = true;
  $('#result-view-label').textContent = '生成结果'; $('#preview-hint').textContent = '阶段完成后即可查看已保留的模型。';
  $('#stage-summary').replaceChildren(); $('#stage-summary').hidden = true;
  $('#export-source').value = ''; $('#height').value = ''; $('#height-axis').value = 'Y';
  $('#geometry-summary').className = ''; $('#geometry-summary').textContent = '上传粗模型后显示几何检查。';
  $('#geometry-info').textContent = '包括连通件、边界边与法向方向。自交、壁厚和实际打印强度需在专门软件中检查。';
}
function applyOptions(options = {}) {
  selectedMode = Object.hasOwn(modeLabels,options.mode) ? options.mode : 'refine';
  for (const key of ['resolution','steps','seed','attention']) if (options[key] !== undefined) $('#'+key).value = options[key];
  if (options.offload !== undefined) $('#offload').checked = options.offload;
  $('#refine-geometry').checked = options.refine_geometry === true;
  $('#generate-texture').checked = Boolean(options.generate_texture ?? selectedMode === 'image_to_asset');
  $('#geometry-engine').value = options.geometry_engine || 'hunyuan_mini';
  $('#trellis-cleanup').checked = options.geometry_cleanup !== false;
  $('#texture-engine').value = options.texture_engine || 'trellis2';
  syncEngines('coarse'); syncEngines('texture');
  $('#geometry-resolution').value = String(options.geometry_resolution || 256);
  $('#geometry-steps').value = String(options.geometry_steps || ($('#geometry-engine').value === 'hunyuan_mini' ? 5 : 50));
  $('#texture-steps').value = String(options.texture_steps || 12);
  $('#texture-resolution').value = String(options.texture_resolution || 512); $('#texture-size').value = String(options.texture_size || 2048);
  $('#background').value = options.background || 'auto'; syncModeUI();
}
function renderComparison(job) {
  const source = comparisonJob || job, selector = $('#comparison-version');
  const available = modelVersions.filter(version => hasJobFile(source,version+'.glb'));
  const previous = selector.value;
  selector.replaceChildren(...available.map(version => new Option(versionLabels[version],version)));
  selector.value = available.includes(previous) ? previous : available.includes('coarse') && !comparisonJob ? 'coarse' : available[0] || '';
  selector.disabled = !available.length;
  const version = selector.value;
  $('#before').dataset.surface = version === 'textured' || version === 'coarse' && source.options?.input_has_materials ? 'pbr' : 'geometry';
  setView('before',version ? `/api/jobs/${encodeURIComponent(source.id)}/files/${version}.glb` : null);
  $('#before-view-label').textContent = comparisonJob ? `对照 · ${versionLabels[version] || '无模型'}` : versionLabels[version] || '粗模型';
  const target = $('#comparison-hint');
  if (!comparisonJob) { target.textContent = '当前任务的阶段对比；可从历史任务选择其他引擎的结果。'; return; }
  const imageA = source.input_sha256, imageB = job.report?.input_sha256;
  const meshA = source.texture_mesh_sha256, meshB = job.report?.texture_mesh_sha256;
  target.textContent = typeof imageA === 'string' && imageA === imageB
    ? version === 'textured' && $('#result-preview').value === 'textured' && typeof meshA === 'string' && meshA === meshB
      ? '同一参考图、同一贴图输入网格，可以直接比较材质效果。'
      : '参考图一致；几何或阶段不同，请按整条流程的结果比较。'
    : '参考图不同或尚无一致性记录，仅供目视对照。';
}
$('#refresh-comparisons').onclick = () => { if (currentJob) { comparisonsFor=''; void loadComparisons(currentJob,epoch); } };
function updatePreviewLighting() {
  const environments = {neutral:'neutral',soft:'/static/environments/studio_small_09.hdr',contrast:'/static/environments/studio_small_08.hdr'};
  const environment = environments[$('#preview-light').value] || 'neutral';
  const exposure = Math.max(.4,Math.min(1.6,+$('#preview-exposure').value || 1));
  for (const view of [$('#before'),$('#after')]) {
    view.setAttribute('environment-image',environment); view.setAttribute('exposure',String(exposure));
  }
}
$('#preview-light').onchange = updatePreviewLighting;
$('#preview-exposure').oninput = updatePreviewLighting;
async function loadComparisons(job, selectedEpoch) {
  if (comparisonsFor === job.id) return;
  comparisonsFor = job.id;
  try {
    const candidates = await api(`/api/jobs/${encodeURIComponent(job.id)}/comparisons`);
    if (epoch !== selectedEpoch || currentJob?.id !== job.id) return;
    comparisonCandidates = candidates.filter(candidate => candidate.id !== job.id);
    const selector = $('#comparison-job'), previous = selector.value;
    selector.replaceChildren(new Option('当前任务的阶段模型',''));
    const image = job.report?.input_sha256;
    comparisonCandidates.sort((a,b) => Number(b.input_sha256 === image && typeof image === 'string') - Number(a.input_sha256 === image && typeof image === 'string'));
    for (const candidate of comparisonCandidates) {
      const options = candidate.options || {};
      const geometry = health?.pipeline_models?.engines?.coarse?.[options.geometry_engine || 'hunyuan_mini']?.name || options.geometry_engine || 'Hi3D';
      const texture = health?.pipeline_models?.engines?.texture?.[options.texture_engine || 'trellis2']?.name || options.texture_engine || 'TRELLIS.2';
      const same = typeof image === 'string' && image === candidate.input_sha256 ? '同图 · ' : '';
      const label = `${same}${options.mode === 'image_to_asset' ? geometry+' / ' : ''}${candidate.available_files.includes('textured.glb') ? texture : '几何'} · ${new Date(candidate.created*1000).toLocaleString()}`;
      selector.append(new Option(label,candidate.id));
    }
    selector.value = comparisonCandidates.some(candidate => candidate.id === previous) ? previous : '';
    comparisonJob = comparisonCandidates.find(candidate => candidate.id === selector.value) || null;
    renderComparison(job);
  } catch(error) {
    if (epoch === selectedEpoch && currentJob?.id === job.id) { comparisonsFor = ''; $('#comparison-hint').textContent = '对照记录暂时无法读取，可稍后刷新。'; }
  }
}
$('#comparison-job').onchange = () => {
  comparisonJob = comparisonCandidates.find(candidate => candidate.id === $('#comparison-job').value) || null;
  $('#comparison-version').value = comparisonJob?.available_files.includes('textured.glb') ? 'textured' : 'coarse';
  if (currentJob) renderComparison(currentJob);
};
$('#comparison-version').onchange = () => { if (currentJob) renderComparison(currentJob); };
function renderDraft() {
  if (inputImage) { $('#image-thumb').src = inputImage.url; $('#image-thumb').hidden = false; }
  else { $('#image-thumb').hidden = true; $('#image-thumb').removeAttribute('src'); }
  $('#image-name').textContent = inputImage?.name || '未选择图片';
  $('#mesh-name').textContent = inputMesh?.name || '未选择粗模型';
  $('#before').dataset.surface = inputMesh?.has_materials ? 'pbr' : 'geometry';
  setView('before', needsMesh() ? inputMesh?.url : null); if (needsMesh() && inputMesh?.geometry) geometry(inputMesh.geometry);
  $('#view-status').textContent = needsMesh() && inputMesh ? '几何预览 · 拖动旋转，滚轮缩放' : '图片建模会先生成粗模型，再完成所选后续阶段';
  $('#progress-label').textContent = inputsReady() ? `输入已准备好，可以开始${modeLabels[selectedMode]}` : needsMesh() ? '先准备图片和模型' : '先准备一张参考图片';
  $('#stage-detail').textContent = needsMesh() ? '请确认图片和模型为同一物体，主体与朝向匹配。' : '一张参考图片即可开始。按选择依次生成粗模、精修几何与 PBR 贴图；各阶段结果保存在任务目录。'; availability();
}
function draftWith(image, mesh) {
  epoch++; currentJob = null; shownJob = null; inputImage = image; inputMesh = mesh; resetResult(); renderDraft();
}
$('#new-task').onclick = () => { if (!inputBusy()) draftWith(null, null); };
document.querySelectorAll('[data-mode]').forEach(button => { button.onclick = () => {
  if (inputBusy() || currentJob) return;
  selectedMode = button.dataset.mode; $('#refine-geometry').checked = false; $('#generate-texture').checked = selectedMode === 'image_to_asset';
  resetResult(); renderDraft();
}; });
for (const id of ['refine-geometry','generate-texture']) $('#'+id).onchange = availability;
$('#geometry-engine').onchange = () => { syncEngines('coarse'); availability(); };
$('#texture-engine').onchange = () => { syncEngines('texture'); availability(); };
document.querySelectorAll('[data-quality-preset]').forEach(button => { button.onclick=() => {
  if (button.disabled) return;
  const p=qualityPresets[button.dataset.qualityPreset];
  $('#geometry-engine').value=p.geometry;$('#texture-engine').value=p.texture;
  syncEngines('coarse');syncEngines('texture');
  $('#geometry-resolution').value=String(p.geometryResolution);$('#geometry-steps').value=String(p.geometrySteps);
  $('#texture-resolution').value=String(p.textureResolution);$('#texture-steps').value=String(p.textureSteps);$('#texture-size').value=String(p.textureSize);
  $('#generate-texture').checked=true;$('#trellis-cleanup').checked=true;$('#offload').checked=true;
  availability();
}; });
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
    const samples = [['sculpted-vase.png','image','image/png']];
    if (needsMesh()) samples.push(['sculpted-vase.glb','mesh','model/gltf-binary']);
    for (const [name, kind, type] of samples) {
      const blob = await api('/api/sample/'+name, {responseType:'blob'});
      items.push(await upload(new File([blob], name, {type}), kind));
    }
    draftWith(items[0],items[1] || null); toast('样例已载入。它用于熟悉流程，不是美术效果基准。');
  } catch (error) { showError(error); }
  finally { pending = null; availability(); }
};
function validMetric(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
function completeGeometry(value) {
  return value && ['vertices','faces','components','boundary_edges','nonmanifold_edges','degenerate_faces'].every(key => validMetric(value[key]) && Number.isInteger(value[key])) && typeof value.winding_consistent === 'boolean' && typeof value.watertight === 'boolean';
}
function geometry(value) {
  if (!completeGeometry(value)) {
    $('#geometry-summary').className = 'warning';
    $('#geometry-summary').textContent = '几何检查报告不完整，暂无法判断边界、流形与封闭性。';
    $('#geometry-info').textContent = '部分几何指标未保存或无法读取。现有模型仍可预览和导出；打印前请在专门软件中检查几何、壁厚与自交。';
    return false;
  }
  const g = value, issues = g.boundary_edges || g.nonmanifold_edges || g.degenerate_faces;
  $('#geometry-summary').className = issues ? 'warning' : 'good';
  $('#geometry-summary').textContent = `${Number(g.faces).toLocaleString()} 个三角面 · ${g.components} 个连通件。${issues ? '检测到开放边界或异常几何，打印前请进一步检查。' : '当前几何检查未发现边界或非流形边；打印前仍需检查壁厚和自交。'}`;
  $('#geometry-info').textContent = `${Number(g.vertices).toLocaleString()} 顶点 · ${g.boundary_edges} 条边界边 · ${g.nonmanifold_edges} 条非流形边 · ${g.degenerate_faces} 个退化面 · 法向方向${g.winding_consistent ? '一致' : '需检查'} · ${g.watertight ? '封闭网格' : '非封闭网格'}。未检测：${(Array.isArray(g.unchecked) ? g.unchecked : ['自交','壁厚']).join('、')}。`;
  if (g.topology_basis) $('#geometry-info').textContent += '拓扑按相同坐标合并 UV 接缝顶点统计，仅用于检查，交付网格未修改。';
  return true;
}
for (const view of [$('#before'), $('#after')]) {
  view.addEventListener('load', () => {
    if (!view.loaded || !view.hasAttribute('src')) return;
    delete view.dataset.loading; delete view.dataset.loadError; $('#'+view.id+'-empty').hidden = true;
    if (view.dataset.surface === 'pbr') return;
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
  const numericFields = ['seed', ...(wantsRefine() ? ['steps'] : []), ...(selectedMode === 'image_to_asset' ? ['geometry-steps'] : []), ...(wantsTexture() ? ['texture-steps'] : [])];
  for (const id of numericFields) if (!$('#'+id).checkValidity() || !$('#'+id).value) { $('#advanced').open = true; $('#'+id).reportValidity(); $('#'+id).focus(); return; }
  pending = 'submitting'; availability();
  try {
    const job = await post('/api/jobs', {mode:selectedMode, image:inputImage.id, mesh:needsMesh() ? inputMesh.id : null, refine_geometry:wantsRefine(), generate_texture:wantsTexture(), geometry_engine:$('#geometry-engine').value, geometry_cleanup:$('#trellis-cleanup').checked, geometry_resolution:+$('#geometry-resolution').value, geometry_steps:+$('#geometry-steps').value, texture_engine:$('#texture-engine').value, texture_steps:+$('#texture-steps').value, texture_resolution:+$('#texture-resolution').value, texture_size:+$('#texture-size').value, background:$('#background').value, resolution:+$('#resolution').value, steps:wantsRefine() ? +$('#steps').value : 25, seed:+$('#seed').value, offload:$('#offload').checked, attention:$('#attention').value});
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
function renderPreview(job) {
  const available = modelVersions.filter(version => hasJobFile(job,version+'.glb')), selector = $('#result-preview');
  for (const option of selector.options) option.disabled = !available.includes(option.value);
  selector.disabled = !available.length;
  if ((!previewManual || !available.includes(selector.value)) && available.length) selector.value = available[0];
  if (!available.length) selector.value = '';
  const version = available.includes(selector.value) ? selector.value : null;
  $('#after').dataset.surface = version === 'textured' ? 'pbr' : 'geometry';
  setView('after',version ? `/api/jobs/${job.id}/files/${version}.glb` : null);
  $('#result-view-label').textContent = version ? versionLabels[version] : '生成结果';
  $('#after').alt = version ? `${versionLabels[version]}三维预览` : '生成结果三维预览';
  $('#preview-hint').textContent = version === 'textured' ? '显示模型原始 PBR 材质；GLB 导出保留贴图。' : version ? '显示中性材质以检查几何；可以切换查看其他已完成阶段。' : '阶段完成后即可查看已保留的模型。';
  return version;
}
$('#result-preview').onchange = () => { previewManual = true; if (currentJob) { renderPreview(currentJob); renderComparison(currentJob); } };
function renderStages(stages) {
  const container = $('#stage-summary'); container.replaceChildren();
  const values = Array.isArray(stages) ? stages.map((value,index) => [value?.name || value?.stage || String(index+1),value]) : stages && typeof stages === 'object' ? Object.entries(stages) : [];
  for (const [name,value] of values) {
    const card = document.createElement('div'), title = document.createElement('b'), detail = document.createElement('small');
    title.textContent = {coarse:'图片建模',refine:'Hi3D 精修',hi3d:'Hi3D 精修',texture:'PBR 贴图',background:'背景处理'}[name] || name;
    const state = value && typeof value === 'object' ? value : {}, pieces = [];
    if (state.state || state.status) pieces.push(stateLabels[state.state || state.status] || state.state || state.status);
    if (validMetric(state.seconds)) pieces.push(duration(state.seconds));
    else if (validMetric(state.duration)) pieces.push(duration(state.duration));
    detail.textContent = pieces.join(' · ') || '阶段信息已记录'; card.append(title,detail); container.append(card);
  }
  container.hidden = !values.length;
}
function renderJob(job) {
  const changed = shownJob !== job.id; currentJob = job;
  if (changed) {
    resetResult(); shownJob = job.id; inputImage = null; inputMesh = null; applyOptions(job.options);
    comparisonJob = null; comparisonCandidates = []; comparisonsFor = '';
    $('#comparison-job').replaceChildren(new Option('当前任务的阶段模型',''));
    $('#comparison-version').value = 'coarse';
  }
  $('#image-thumb').hidden = !hasJobFile(job,'input.png');
  const imageUrl = `/api/jobs/${job.id}/files/input.png`;
  if (hasJobFile(job,'input.png') && $('#image-thumb').getAttribute('src') !== imageUrl) $('#image-thumb').src = imageUrl;
  if (!hasJobFile(job,'input.png')) $('#image-thumb').removeAttribute('src');
  $('#image-name').textContent = hasJobFile(job,'input.png') ? '此任务的参考图片' : '参考图片文件已缺失';
  $('#mesh-name').textContent = hasJobFile(job,'coarse.glb') ? selectedMode === 'image_to_asset' ? '此任务生成的粗模型' : '此任务的输入粗模型' : selectedMode === 'image_to_asset' ? '此任务尚无粗模型' : '输入粗模型文件已缺失';
  const displayStage = {succeeded:'complete',failed:'failed',cancelled:'cancelled',interrupted:'interrupted'}[job.state] || job.stage;
  $('#progress-label').textContent = job.state === 'cancelling' ? '正在停止任务并释放显存…' : stageName(displayStage) || stateLabels[job.state] || job.state;
  $('#stage-detail').textContent = job.message || '';
  $('#elapsed').textContent = duration((activeStates.has(job.state) ? Date.now()/1000 : job.updated || job.created) - job.created);
  const event = job.event || {};
  if (job.state === 'running' && validMetric(event.step) && event.step > 0 && validMetric(event.num_inference_steps) && event.num_inference_steps > 0) {
    $('#progress').hidden = false; $('#progress').max = event.num_inference_steps; $('#progress').value = event.step;
    const pass = validMetric(event.pass_index) && validMetric(event.num_passes) ? `第 ${event.pass_index+1} / ${event.num_passes} 轮 · ` : '';
    $('#stage-detail').textContent = `${job.message ? job.message+' · ' : ''}${pass}本轮采样 ${event.step} / ${event.num_inference_steps}。这是当前阶段进度，后续还需处理与保存。`;
  } else $('#progress').hidden = true;
  const report = job.report && typeof job.report === 'object' ? job.report : {}, version = renderPreview(job);
  renderComparison(job);
  $('#face-stat').textContent = validMetric(report.geometry?.faces) ? report.geometry.faces.toLocaleString() : '—';
  $('#time-stat').textContent = validMetric(report.seconds) ? duration(report.seconds) : '—';
  $('#memory-stat').textContent = validMetric(report.peak_allocated_mb) ? `${(report.peak_allocated_mb/1024).toFixed(2)} GiB` : '—';
  $('#res-stat').textContent = wantsRefine(job.options || {}) && validMetric(report.actual_resolution) && report.actual_resolution > 0 ? report.actual_resolution : '—';
  renderStages(report.stages || job.stages);
  $('#view-status').textContent = `${modeLabels[selectedMode]} · ${report.compatibility_port ? 'Windows SDPA 兼容路径 · ' : ''}两侧视角同步`;
  $('#after-title').textContent = version ? versionLabels[version] : activeStates.has(job.state) ? '正在生成阶段结果，请稍候' : '此任务没有可用的阶段模型';
  $('#after-hint').textContent = version ? '可切换已完成的阶段；任务停止或失败后仍保留可用文件' : '可在下方查看任务阶段与日志';
  const complete = report.geometry ? geometry(report.geometry) : false;
  if (!activeStates.has(job.state)) {
    if (!report.geometry) geometry(null);
    if (job.state === 'succeeded') {
      const absent = [];
      if (wantsRefine(job.options || {}) && !hasJobFile(job,'refined.glb')) absent.push('Hi3D 精修文件');
      if (wantsTexture(job.options || {}) && !hasJobFile(job,'textured.glb')) absent.push('PBR 材质文件');
      if (absent.length) $('#stage-detail').textContent = `任务记录已完成，但当前目录缺少${absent.join('、')}。现有阶段模型仍可预览和导出。`;
      else if (!complete || !validMetric(report.seconds) || !validMetric(report.peak_allocated_mb)) $('#stage-detail').textContent = '运行报告不完整，缺失指标显示为 —。现有阶段模型仍可预览和导出，几何质量需另行检查。';
    } else $('#stage-detail').textContent = `${job.message || stateLabels[job.state] || '任务未完成'}。已生成的阶段文件与日志保留，可查看、导出或复用输入。`;
  }
  availability();
}
async function loadArtifacts(job, selectedEpoch) {
  if (epoch !== selectedEpoch || currentJob?.id !== job.id) return;
  if (comparisonsFor !== job.id) void loadComparisons(job,selectedEpoch);
  const files = job.available_files || [], key = `${job.id}:${job.updated}:${files.join(',')}`;
  const condition = files.includes('foreground.png') ? 'foreground.png' : files.includes('conditioning.png') ? 'conditioning.png' : null;
  if (condition) { $('#conditioning').src = `/api/jobs/${job.id}/files/${condition}`; $('#conditioning').hidden = false; }
  else { $('#conditioning').hidden = true; $('#conditioning').removeAttribute('src'); }
  if (!files.includes('worker.log')) { $('#log').textContent = '日志文件缺失或尚未生成。'; return; }
  if (logsKey === key) return;
  try {
    const text = await api(`/api/jobs/${job.id}/files/worker.log`, {responseType:'text',timeoutMs:10000});
    if (epoch === selectedEpoch && currentJob?.id === job.id) { $('#log').textContent = text.slice(-24000) || '任务日志暂时为空。'; logsKey = key; }
  } catch (error) {
    const detail = error.network ? '本次日志请求中断或超时' : error.message.replace(/[。.!！]+$/u,'');
    if (epoch === selectedEpoch && currentJob?.id === job.id) $('#log').textContent = `日志暂时无法读取：${detail}。稍后会自动重试；现有预览与导出仍可使用。`;
  }
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
  try { const result = await post(`/api/jobs/${currentJob.id}/reuse`); applyOptions(result.options); draftWith(result.image, result.mesh || null); toast('输入和参数已复制到新任务。调整流程后即可再次运行，原记录与阶段文件保留。'); }
  catch (error) { showError(error); } finally { pending = null; availability(); }
};
$('#report').onclick = () => { if (currentJob && !$('#report').disabled) window.open(`/api/jobs/${currentJob.id}/files/report.json`, '_blank', 'noopener'); };
$('#export-source').onchange = () => { exportSourceManual = true; exportAvailability(); };
for (const id of ['export-source','export-format','height-axis','height']) $('#'+id).addEventListener('change', () => { $('#dimension-link').hidden = true; });
$('#export').onclick = async () => {
  if ($('#export').disabled || !$('#height').reportValidity()) return;
  exportPending = true; const jobId = currentJob.id, selectedEpoch = epoch, format = $('#export-format').value; availability();
  try {
    const result = await post(`/api/jobs/${jobId}/export`, {source:$('#export-source').value, format, height_axis:$('#height-axis').value, height_mm:$('#height').value ? +$('#height').value : null});
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
      const title = document.createElement('b'), mode = job.options?.mode || 'refine';
      const geometryName = health?.pipeline_models?.engines?.coarse?.[job.options?.geometry_engine || 'hunyuan_mini']?.name || job.options?.geometry_engine || 'Hunyuan mini';
      const textureName = health?.pipeline_models?.engines?.texture?.[job.options?.texture_engine || 'trellis2']?.name || job.options?.texture_engine || 'TRELLIS.2 PBR';
      title.textContent = `${modeLabels[mode] || mode}${mode === 'image_to_asset' ? ' · '+geometryName : ''}${wantsRefine(job.options || {}) ? ' · Hi3D '+(job.options?.resolution || '—') : ''}${wantsTexture(job.options || {}) ? ' · '+textureName : ''} · 种子 ${job.options?.seed ?? '—'}`;
      const time = document.createElement('time'); time.textContent = new Date(job.created*1000).toLocaleString();
      const description = document.createElement('p'); description.textContent = job.state === 'succeeded' ? '打开查看阶段结果、导出或复用输入' : '打开查看日志、已保留阶段与输入'; info.append(title, document.createElement('br'), time, description);
      const state = document.createElement('span'); state.className = 'state-pill'; state.textContent = stateLabels[job.state] || job.state;
      const button = document.createElement('button'); button.textContent = '打开任务 →'; button.onclick = () => openJob(job.id); row.append(info, state, button); list.append(row);
    }
  } catch (error) { if (historyEpoch === selectedEpoch) { showError(error); $('#history-list').textContent = '无法读取记录，请检查连接后点击刷新。'; } }
  finally { if (historyEpoch === selectedEpoch) $('#refresh-history').disabled = false; }
}
$('#refresh-history').onclick = history;
function renderHealth() {
  const recoveryBlocked = Boolean(health.recovery_blocked), recoveryErrors = health.recovery_errors || [];
  $('#recovery-alert').hidden = !recoveryBlocked && !recoveryErrors.length;
  $('#recovery-alert').textContent = recoveryBlocked ? `检测到重启前仍在运行的工作进程，已暂停新任务。请等待进程退出；涉及进程：${(health.recovery_workers || []).map(item => item.pid).join('、')}。` : `有 ${recoveryErrors.length} 条历史记录无法读取，已跳过；其他记录仍可使用。详情：${recoveryErrors.map(item => typeof item === 'string' ? item : JSON.stringify(item)).join('；')}`;
  $('#connection-alert').hidden = online; $('#health-badge').classList.toggle('offline', !online);
  renderHealthBadge();
  const cards = $('#model-cards'); cards.replaceChildren();
  for (const [key, component] of Object.entries(health.models?.components || {})) {
    const card = document.createElement('div'); card.className = 'model-card'; const title = document.createElement('h3'); title.textContent = {hi3d:'Hi3D V1.0',dino:'DINOv2 Large',clip:'CLIP ViT-L/14'}[key] || key;
    const repo = document.createElement('small'); repo.textContent = component.repo; const license = document.createElement('p'); license.className = 'hint'; license.textContent = component.license;
    const state = document.createElement('div'); state.className = 'model-state'; state.textContent = component.message || (component.verified ? '✓ 完整校验通过' : '需要准备或校验'); card.append(title, repo, license, state); cards.append(card);
  }
  $('#download-state').textContent = health.verification_error ? '校验出错：'+health.verification_error : health.model_checking ? '正在完整校验文件，请稍候…' : health.models?.download?.message || 'Hi3D 核心尚无下载任务。';
  const pipeline = health.pipeline_models, pipelineCards = $('#pipeline-model-cards'); pipelineCards.replaceChildren();
  for (const [stage, choices] of Object.entries(pipeline?.engines || {})) for (const [id, choice] of Object.entries(choices)) {
    const card = document.createElement('div'); card.className = 'model-card';
    const title = document.createElement('h3'); title.textContent = choice.name;
    const description = document.createElement('p'); description.className = 'hint'; description.textContent = choice.description;
    const state = document.createElement('div'); state.className = 'model-state';
    state.textContent = `${stage === 'coarse' ? '图片建模' : 'PBR 贴图'} · ${choice.ready ? '✓ 已就绪' : choice.runtime_ready ? '待准备模型' : '需要更新扩展环境与模型'}`;
    const button = document.createElement('button'); button.dataset.engineDownload = `${stage}:${id}`;
    button.textContent = choice.ready ? '检查 / 补齐此引擎' : '下载此引擎模型';
    button.onclick = () => downloadEngine(stage,id);
    card.append(title,description,state,button); pipelineCards.append(card);
  }
  const componentCards = $('#pipeline-component-cards'); componentCards.replaceChildren();
  for (const [key,component] of Object.entries(pipeline?.components || {})) {
    const card = document.createElement('div'); card.className = 'model-card';
    const title = document.createElement('h3'); title.textContent = {background:'自动前景处理',coarse:'Hunyuan mini 权重',texture:'TRELLIS.2 贴图权重',texture_encoder:'DINOv3 图像编码器',coarse_hunyuan21:'Hunyuan3D 2.1 几何权重',coarse_trellis2:'TRELLIS.2 几何权重',trellis_structure:'TRELLIS 空间结构解码器',texture_ultratex:'UltraTex 颜色 / 材质适配器',ultratex_backbone:'UltraTex FP8 Base 9B 底座',ultratex_backbone_int8:'UltraTex INT8 ConvRot Base 9B 底座',ultratex_text_encoder:'UltraTex 图像解码器与文本编码器'}[key] || key;
    const repo = document.createElement('small'); repo.textContent = component.repo || '';
    const license = document.createElement('p'); license.className = 'hint'; license.textContent = component.license || '许可见组件目录';
    const state = document.createElement('div'); state.className = 'model-state';
    state.textContent = component.ready ? '✓ 组件就绪' : component.message || `待准备${Array.isArray(component.missing) && component.missing.length ? ' · 缺少 '+component.missing.length+' 个文件' : ''}`;
    card.append(title,repo,license,state); componentCards.append(card);
  }
  if (!pipeline) pipelineCards.textContent = '当前服务未提供扩展组件状态，请更新并重启完整工作台。Hi3D 几何精修仍可单独使用。';
  $('#pipeline-download-state').textContent = pipeline?.download?.message || '扩展组件尚无下载任务。';
  const coverage = pipeline?.engine_coverage;
  $('#pipeline-runtime-state').textContent = pipeline?.runtime?.ready ? `✓ 扩展运行环境就绪。${coverage ? '引擎已就绪 '+coverage.ready+' / '+coverage.total+'。' : ''}各阶段依次运行，所需组件按任务选择。` : `扩展运行环境尚未就绪。${pipeline?.runtime?.message || pipeline?.runtime?.error || '请完成扩展组件安装后重新检查。'}`;
  syncEngines();
  if (health.diagnostics && pending !== 'diagnose') $('#diagnostics').textContent = JSON.stringify(health.diagnostics, null, 2);
  availability();
}
async function downloadEngine(stage,id) {
  if (!online || pending || gpuBusy() || downloadingModels() || checkingModels()) return;
  pending = 'pipeline-download'; availability();
  try {
    await post('/api/pipeline/models/download',{selection:`engine:${stage}:${id}`});
    toast('所选引擎的下载请求已提交。'); await refresh();
  } catch(error) { showError(error); }
  finally { pending = null; availability(); }
}
async function refresh() {
  const session = sessionReady ? null : await api('/api/session');
  if (session) { token = session.token; sessionReady = true; $('#shutdown-workbench').disabled = !session.can_shutdown; if (!sessionDefaultsApplied && !currentJob) $('#attention').value = session.default_attention || 'flash'; sessionDefaultsApplied = true; document.querySelectorAll('.app-version').forEach(el => { el.textContent = session.version; }); }
  health = await api('/api/health'); if (workbenchStopped) return; online = true; renderHealth();
  if (Date.now() - lastUpdateRefresh > 15000) { lastUpdateRefresh = Date.now(); void refreshUpdates(); }
}
let lastUpdateRefresh = 0;
async function refreshUpdates() {
  try {
    const value = await api('/api/updates');
    $('#auto-update').checked = value.auto_update;
    $('#auto-update').disabled = value.source_checkout;
    $('#update-state').textContent = value.source_checkout ? '当前为源码 Git 目录，请使用 Git 更新。独立整合包支持自动下载与安装。' : value.message;
    $('#check-updates').disabled = ['checking','downloading'].includes(value.phase);
  } catch (error) { $('#update-state').textContent = '更新状态暂时不可用：'+error.message; }
}
$('#check-updates').onclick = async () => {
  $('#check-updates').disabled = true;
  try { await post('/api/updates/check'); $('#update-state').textContent = '已开始检查，下载完成后会提示下次启动安装。'; }
  catch (error) { showError(error); }
  finally { lastUpdateRefresh = 0; }
};
$('#auto-update').onchange = async () => {
  try { await post('/api/updates/preferences',{auto_update:$('#auto-update').checked}); await refreshUpdates(); }
  catch (error) { showError(error); await refreshUpdates(); }
};
async function tick() {
  if (pollBusy || workbenchStopped) return; pollBusy = true;
  try {
    try { await refresh(); } catch (error) { if (!workbenchStopped) unavailable(error); return; }
    if (workbenchStopped) return;
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
$('#shutdown-workbench').onclick = async () => {
  if ((health?.active || health?.operation) && !window.confirm('关闭工作台并停止当前任务？已保存的输入和阶段结果会保留。')) return;
  const button = $('#shutdown-workbench'); button.disabled = true;
  try {
    await post('/api/shutdown'); workbenchStopped = true; online = false; availability();
    button.disabled = true; button.textContent = '已请求关闭';
    $('#connection-alert').hidden = true;
    $('#health-badge').textContent = '正在关闭工作台';
    toast('已请求停止服务并释放推理进程。启动窗口显示“本次启动已结束”后，可关闭窗口和此页面。');
  } catch (error) { button.disabled = false; showError(error); }
};
for (const [id, kind, action] of [['download-models','download',async () => { const result = await post('/api/models/download'); toast(result.started ? '开始下载 Hi3D 与必需编码器，请在此查看进度。' : '已有下载进程，请等待完成。'); }],['verify-models','verify',async () => { const result = await post('/api/models/verify'); toast(result.ok ? '全部模型文件 SHA256 校验通过。' : '校验未通过：'+result.errors.join('，')); }],['diagnose','diagnose',async () => { $('#diagnostics').textContent = '正在检查 GPU 基础内核，首次编译可能需要几分钟…'; const result = await post('/api/diagnostics'); $('#diagnostics').textContent = JSON.stringify(result,null,2); toast(result.kernel_probe === 'passed' ? 'GPU 基础内核检查通过，可以尝试精修。' : 'GPU 检查发现问题，请查看诊断详情。'); }]]) {
  $('#'+id).onclick = async () => { if ($('#'+id).disabled) return; pending = kind; availability(); try { await action(); await refresh(); } catch (error) { showError(error); } finally { pending = null; availability(); } };
}
for (const [id,selection] of [['download-pipeline-complete','complete'],['download-pipeline-all','all'],['download-pipeline-coarse','coarse'],['download-pipeline-texture','texture']]) {
  $('#'+id).onclick = async () => {
    if ($('#'+id).disabled) return; pending = 'pipeline-download'; availability();
    try { const result = await post('/api/pipeline/models/download',{selection}); toast(result.started === false ? '已有下载任务，请查看组件状态。' : '扩展组件下载请求已提交，请在此查看进度。'); await refresh(); }
    catch (error) { showError(error); } finally { pending = null; availability(); }
  };
}
$('#verify-pipeline-models').onclick = async () => {
  if ($('#verify-pipeline-models').disabled) return; pending = 'pipeline-verify'; availability();
  try {
    const result = await post('/api/pipeline/models/verify');
    toast(result.ok === false ? '扩展组件校验未通过：'+(result.errors || []).join('，') : result.ok || result.ready ? '扩展组件校验通过。' : '检查已完成，请查看各组件与运行环境状态。');
    await refresh();
  } catch (error) { showError(error); } finally { pending = null; availability(); }
};
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
    ctx.globalCompositeOperation = 'destination-out'; ctx.lineCap = 'round'; ctx.lineWidth = size; ctx.beginPath();
    if (!last || last[0] === p[0] && last[1] === p[1]) { ctx.arc(...p,size/2,0,Math.PI*2); ctx.fill(); }
    else { ctx.moveTo(...last); ctx.lineTo(...p); ctx.stroke(); }
    ctx.globalCompositeOperation = 'source-over';
  } else {
    const start = last || p, radius = size/2, angle = Math.atan2(p[1]-start[1], p[0]-start[0]);
    ctx.save(); ctx.beginPath();
    if (start[0] === p[0] && start[1] === p[1]) ctx.arc(...p,radius,0,Math.PI*2);
    else {
      const dx = Math.sin(angle)*radius, dy = Math.cos(angle)*radius;
      ctx.moveTo(start[0]+dx,start[1]-dy); ctx.lineTo(p[0]+dx,p[1]-dy);
      ctx.arc(...p,radius,angle-Math.PI/2,angle+Math.PI/2);
      ctx.lineTo(start[0]-dx,start[1]+dy); ctx.arc(...start,radius,angle+Math.PI/2,angle+Math.PI*1.5); ctx.closePath();
    }
    ctx.clip(); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(original,0,0); ctx.restore();
  }
  last = p;
}
canvas.onpointerdown = event => { drawing = true; last = null; canvas.setPointerCapture(event.pointerId); paint(event); };
canvas.onpointermove = paint; canvas.onpointerup = canvas.onpointercancel = canvas.onlostpointercapture = () => { drawing = false; last = null; };
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
