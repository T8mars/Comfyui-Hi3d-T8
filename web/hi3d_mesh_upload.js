import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const supported = (file) => /\.(glb|ply|stl|npz)$/i.test(file?.name || '');

function migrateLegacyCanvas(graph) {
  const backends = graph.nodes?.filter(node => node.type === 'Hi3DBackend') || [];
  const refiners = graph.nodes?.filter(node => node.type === 'Hi3DRefine') || [];
  if (backends.length !== 1 || refiners.length !== 1 || !Array.isArray(graph.links)) return false;
  const backend = backends[0], refine = refiners[0];
  const modelLink = graph.links.find(link => link[1] === backend.id && link[3] === refine.id);
  if (!modelLink) return false;
  const old = refine.widgets_values || [];
  if (!Array.isArray(old) || old.length < 6) return false;
  backend.type = 'Hi3DModelLoader'; backend.title = 'Hi3D · 加载模型（独立）';
  backend.inputs = []; backend.size = [360, 220];
  backend.widgets_values = ['', old[4] !== false, old[5] || 'sdpa', false, false];
  backend.widgets_values_named = { model_root: '', cpu_offload: old[4] !== false, attention: old[5] || 'sdpa', keep_model_loaded: false, download_missing: false };
  backend.properties = { ...backend.properties, 'Node name for S&R': 'Hi3DModelLoader' };
  backend.outputs = [{ name: 'model', type: 'HI3D_MODEL', links: [modelLink[0]] }];
  const removed = new Set(graph.links.filter(link => link[1] === backend.id && link !== modelLink || link[1] === refine.id && link[2] === 2).map(link => link[0]));
  graph.links = graph.links.filter(link => !removed.has(link[0])); modelLink[5] = 'HI3D_MODEL';
  for (const input of refine.inputs || []) if (input.name === 'backend') { input.name = 'model'; input.type = 'HI3D_MODEL'; delete input.localized_name; delete input.label; }
  refine.widgets_values = old.slice(0, 4); delete refine.widgets_values_named;
  refine.outputs = refine.outputs?.slice(0, 2);
  for (const node of graph.nodes) {
    if (node.type === 'Hi3DLoadMesh') {
      node.inputs = []; node.widgets_values = ['请先上传粗模'];
      node.widgets_values_named = { mesh_file: '请先上传粗模' };
    } else for (const input of node.inputs || []) if (removed.has(input.link)) input.link = null;
  }
  return true;
}

app.registerExtension({
  name: 'T8star.Hi3D.MeshUpload',
  beforeConfigureGraph(graph) {
    if (migrateLegacyCanvas(graph)) app.extensionManager?.toast?.add({ severity: 'info', summary: '已转换为独立 Hi3D 节点', detail: '旧后端地址已移除，请在粗模节点重新上传文件，或导入新版示例。', life: 9000 });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'Hi3DLoadMesh') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      const field = node.widgets.find(widget => widget.name === 'mesh_file');
      let busy = false;
      const upload = async (file) => {
        if (busy || !supported(file)) return false;
        busy = true;
        button.name = '正在上传粗模…'; app.graph.setDirtyCanvas(true, true);
        try {
          const form = new FormData();
          form.append('image', file); form.append('type', 'input'); form.append('subfolder', 'hi3d');
          const response = await api.fetchApi('/upload/image', { method: 'POST', body: form });
          if (!response.ok) throw new Error(`上传失败（${response.status}）：${await response.text()}`);
          const value = await response.json();
          const name = value.subfolder ? `${value.subfolder}/${value.name}` : value.name;
          const choices = field.options.values;
          if (Array.isArray(choices) && !choices.includes(name)) choices.push(name);
          field.value = name; field.callback?.(name);
        } catch (error) {
          const toast = app.extensionManager?.toast;
          if (toast) toast.add({ severity: 'error', summary: '粗模上传失败', detail: error.message, life: 7000 });
          else window.alert(error.message);
        } finally {
          busy = false; button.name = '选择文件上传粗模'; app.graph.setDirtyCanvas(true, true);
        }
        return true;
      };
      const button = node.addWidget('button', '选择文件上传粗模', null, () => {
        const picker = document.createElement('input'); picker.type = 'file'; picker.accept = '.glb,.ply,.stl,.npz';
        picker.onchange = () => { if (picker.files?.[0]) void upload(picker.files[0]); };
        picker.click();
      }, { serialize: false });
      button.serializeValue = () => undefined;
      const dragOver = node.onDragOver;
      node.onDragOver = function (event) {
        return [...(event.dataTransfer?.files || [])].some(supported) || [...(event.dataTransfer?.items || [])].some(item => item.kind === 'file') || dragOver?.call(this, event);
      };
      const drop = node.onDragDrop;
      node.onDragDrop = function (event) {
        const file = [...(event.dataTransfer?.files || [])].find(supported);
        if (file) { void upload(file); return true; }
        return drop?.call(this, event) || false;
      };
      this.setSize(this.computeSize());
    };
  }
});
