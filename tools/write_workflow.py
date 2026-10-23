import json
from pathlib import Path

def node(i,t,pos,size,inputs,outputs,widgets):
    return {'id':i,'type':t,'pos':pos,'size':size,'flags':{},'order':i-1,'mode':0,'inputs':inputs,'outputs':outputs,'properties':{'Node name for S&R':t},'widgets_values':widgets}
nodes=[
node(1,'Hi3DBackend',[30,30],[300,80],[],[{'name':'HI3D_BACKEND','type':'HI3D_BACKEND','links':[1,2]}],['http://127.0.0.1:8378']),
node(2,'LoadImage',[30,180],[300,340],[],[{'name':'IMAGE','type':'IMAGE','links':[3]},{'name':'MASK','type':'MASK','links':[4]}],['reference.png','image']),
node(3,'Hi3DLoadMesh',[380,30],[350,110],[{'name':'backend','type':'HI3D_BACKEND','link':1}],[{'name':'MESH','type':'MESH','links':[5]}],['C:/your-model/coarse.glb']),
node(4,'Hi3DRefine',[800,110],[330,330],[{'name':'backend','type':'HI3D_BACKEND','link':2},{'name':'image','type':'IMAGE','link':3},{'name':'mesh','type':'MESH','link':5},{'name':'mask','type':'MASK','link':4}],[{'name':'refined_mesh','type':'MESH','links':[6]},{'name':'report_json','type':'STRING','links':None},{'name':'job_id','type':'STRING','links':None}],[512,25,0,'fixed',True,'sdpa']),
node(5,'SaveGLB',[1200,120],[300,130],[{'name':'mesh','type':'MESH','link':6}],[],['Hi3D/refined'])]
doc={'last_node_id':5,'last_link_id':6,'nodes':nodes,'links':[[1,1,0,3,0,'HI3D_BACKEND'],[2,1,0,4,0,'HI3D_BACKEND'],[3,2,0,4,1,'IMAGE'],[4,2,1,4,3,'MASK'],[5,3,0,4,2,'MESH'],[6,4,0,5,0,'MESH']],'groups':[],'config':{},'extra':{'ds':{'scale':.85,'offset':[30,60]}},'version':.4}
Path('examples/Hi3D-Refine.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
