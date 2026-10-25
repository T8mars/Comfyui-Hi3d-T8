import json
from pathlib import Path

def node(i,t,pos,size,inputs,outputs,widgets):
 return {"id":i,"type":t,"pos":pos,"size":size,"flags":{},"order":i-1,"mode":0,"inputs":inputs,"outputs":outputs,"properties":{"Node name for S&R":t},"widgets_values":widgets}
nodes=[
 node(1,"Hi3DModelLoader",[30,30],[360,220],[],[{"name":"model","type":"HI3D_MODEL","links":[1]}],["",True,"sdpa",False,False]),
 node(2,"LoadImage",[30,300],[320,400],[],[{"name":"IMAGE","type":"IMAGE","links":[2]},{"name":"MASK","type":"MASK","links":[3]}],["reference.png","image"]),
 node(3,"Hi3DLoadMesh",[440,30],[340,140],[],[{"name":"mesh","type":"MESH","links":[4]}],["请先上传粗模"]),
 node(4,"Hi3DRefine",[830,150],[340,250],[{"name":"model","type":"HI3D_MODEL","link":1},{"name":"image","type":"IMAGE","link":2},{"name":"mesh","type":"MESH","link":4},{"name":"mask","type":"MASK","link":3}],[{"name":"refined_mesh","type":"MESH","links":[5]},{"name":"report_json","type":"STRING","links":None}],[512,25,0,"fixed"]),
 node(5,"SaveGLB",[1230,150],[310,140],[{"name":"mesh","type":"MESH","link":5}],[],["Hi3D/refined"])]
doc={"last_node_id":5,"last_link_id":5,"nodes":nodes,"links":[[1,1,0,4,0,"HI3D_MODEL"],[2,2,0,4,1,"IMAGE"],[3,2,1,4,3,"MASK"],[4,3,0,4,2,"MESH"],[5,4,0,5,0,"MESH"]],"groups":[],"config":{},"extra":{"ds":{"scale":.85,"offset":[30,60]}},"version":.4}
Path("examples/Hi3D-Refine.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
