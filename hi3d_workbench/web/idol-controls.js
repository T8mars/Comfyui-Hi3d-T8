const $ = selector => document.querySelector(selector);
const fields = ['gaussians','cfg','decode-seed','extract-mesh','crop','level','smooth','min-component','min-opacity','video','sampling','align-mesh','mesh-backend','solid','fill'];
const defaults = [262144,3,0,true,'synthetic',.6,0,500,.02,true,'native_euler',true,'solid',true,20];
export function installIdolControls(changed) {
  $('#geometry-engine').append(new Option('Idol · 高斯与彩色网格','idol'));
  $('#geometry-settings').insertAdjacentHTML('beforeend', `<fieldset id="idol-settings" hidden><legend>Idol 高斯生成</legend>
    <label class="field">采样方案<select id="idol-sampling"><option value="native_euler">原生 Euler 时间表与引导公式</option><option value="comfy">Comfy 官方模板 · DPM++ 2M</option></select><small>两种方案的引导值口径不同；Euler 建议 50 步。随机数与编码使用本地 Comfy 后端。</small></label>
    <div class="field-row"><label class="field">高斯数量<select id="idol-gaussians"><option>32768</option><option>131072</option><option selected>262144</option><option>524288</option><option>1048576</option></select><small>数量控制密度；更多点不保证更多细节。</small></label>
    <label class="field">引导强度<input id="idol-cfg" type="number" min="0" max="20" step="0.1" value="3"></label></div>
    <div class="field-row"><label class="field">参考图类型<select id="idol-crop"><option value="synthetic">生成图／渲染图</option><option value="real">真实照片 · 留出更多边距</option></select></label>
    <label class="field">高斯解码种子<input id="idol-decode-seed" type="number" min="0" max="9007199254740991" value="0"></label></div>
    <label class="toggle"><input id="idol-extract-mesh" type="checkbox" checked><span><b>同时提取彩色网格</b><small>生成顶点色 GLB，可继续精修、生成 PBR 或进行资产加工。</small></span></label>
    <div id="idol-mesh-settings"><label class="field">网格提取<select id="idol-mesh-backend"><option value="solid">内部壳层处理 · 新版官方算法</option><option value="legacy">原始密度表面 · 旧版对照</option></select></label>
    <label class="toggle"><input id="idol-solid" type="checkbox" checked><span><b>处理内部壳层</b><small>减少内部碎片；有孔物体仍需检查开口。</small></span></label>
    <label class="field">开口保留强度<input id="idol-fill" type="number" min="0" max="1000" step="1" value="20"><small>较小数值更容易封住杯口和衣领，请对照原始高斯。</small></label>
    <div class="field-row"><label class="field">表面阈值<input id="idol-level" type="number" min="0.01" max="2" step="0.01" value="0.6"></label>
    <label class="field">表面平滑次数<input id="idol-smooth" type="number" min="0" max="60" value="0"></label></div>
    <div class="field-row"><label class="field">最小部件顶点数<input id="idol-min-component" type="number" min="0" max="100000" value="500"><small>设为 0 保留全部部件。</small></label>
    <label class="field">最低不透明度<input id="idol-min-opacity" type="number" min="0" max="1" step="0.01" value="0.02"></label></div>
    <label class="toggle"><input id="idol-align-mesh" type="checkbox" checked><span><b>按参考视角对齐网格正面</b><small>使用预测相机，只旋转导出的网格；原始高斯保持。</small></span></label></div>
    <label class="toggle"><input id="idol-video" type="checkbox" checked><span><b>保存环绕展示视频</b><small>静态物体的 3 秒展示，512 × 512；人物动作需后续绑定与动作制作。</small></span></label>
    <p class="hint">原始高斯与提取网格分别保存。彩色网格采用无光照顶点色，PBR 贴图由所选贴图引擎另行生成。上方几何分辨率控制网格提取精度。</p></fieldset>`);
  $('#stage-summary').insertAdjacentHTML('afterend', `<section id="idol-result" hidden><h3>原始高斯</h3><p id="idol-result-status" class="hint"></p><div id="idol-downloads" class="button-row"></div>
    <details id="idol-preview-details"><summary>旋转查看原始高斯</summary><iframe id="idol-preview" title="Idol 原始高斯三维预览" style="width:100%;height:520px;border:0" loading="lazy"></iframe></details>
    <video id="idol-preview-video" controls preload="none" style="max-width:100%;max-height:420px" hidden></video></section>`);
  for (const field of fields) $('#idol-'+field).addEventListener('change',changed);
  $('#idol-sampling').addEventListener('change',()=>{$('#geometry-steps').value=$('#idol-sampling').value==='native_euler'?'50':'20';changed();});
  $('#idol-extract-mesh').addEventListener('change',()=>{
    if (!$('#idol-extract-mesh').checked) {$('#refine-geometry').checked=false;$('#generate-texture').checked=false;}
    changed();
  });
  $('#idol-preview-details').addEventListener('toggle',()=>{
    const frame=$('#idol-preview');
    if ($('#idol-preview-details').open && frame.dataset.url && frame.getAttribute('src')!==frame.dataset.url) frame.src=frame.dataset.url;
  });
}
export function idolOptions() {
  return Object.fromEntries(fields.map((field,i)=>['idol_'+field.replaceAll('-','_'),
    typeof defaults[i]==='boolean' ? $('#idol-'+field).checked : typeof defaults[i]==='number' ? +$('#idol-'+field).value : $('#idol-'+field).value]));
}
export function restoreIdolOptions(options) {
  fields.forEach((field,i)=>{const value=options['idol_'+field.replaceAll('-','_')]??defaults[i],node=$('#idol-'+field);if(typeof defaults[i]==='boolean')node.checked=value;else node.value=value;});
}
export function syncIdolControls(active,locked) {
  $('#idol-settings').hidden=!active;
  for(const field of fields) $('#idol-'+field).disabled=locked;
  const splatOnly=active&&!$('#idol-extract-mesh').checked;
  $('#idol-mesh-settings').hidden=!$('#idol-extract-mesh').checked;
  if(splatOnly){$('#refine-geometry').disabled=true;$('#generate-texture').disabled=true;}
}
export function resetIdolResult() {
  $('#idol-result').hidden=true;
  $('#idol-preview').removeAttribute('src');delete $('#idol-preview').dataset.url;
  $('#idol-preview-video').pause();$('#idol-preview-video').removeAttribute('src');$('#idol-preview-video').hidden=true;
}
export function renderIdolResult(job) {
  const files=job.available_files||[],exists=files.includes('splat.ply');
  $('#idol-result').hidden=!exists;
  if(!exists){resetIdolResult();return;}
  $('#idol-result-status').textContent=`${(job.options?.idol_gaussians||262144).toLocaleString()} 个高斯 · 原始外观；与下方网格／PBR 分开查看。`;
  const base=`/api/jobs/${encodeURIComponent(job.id)}/files/`;
  const target='/static/splat-viewer.html?src='+encodeURIComponent(base+'splat.ply');
  const frame=$('#idol-preview');frame.dataset.url=target;
  if($('#idol-preview-details').open && frame.getAttribute('src')!==target)frame.src=target;
  const links=files.filter(name=>['splat.ply','splat.spz','splat.ksplat','splat.json','idol-input.png','splat-preview.mp4','idol-camera.json'].includes(name));
  $('#idol-downloads').replaceChildren(...links.map(name=>{const a=document.createElement('a');a.textContent='下载 '+name;a.href=base+name;a.download=name;return a;}));
  const video=$('#idol-preview-video'),videoUrl=base+'splat-preview.mp4';video.hidden=!files.includes('splat-preview.mp4');
  if(!video.hidden&&video.getAttribute('src')!==videoUrl)video.src=videoUrl;
}
