"""Restore shading normals from the matching source geometry into a new GLB.

Positions, UVs, faces and generated material pixels are retained. No inference
is rerun. Original files are never overwritten.
"""
from pathlib import Path
import argparse,json,time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--textured',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('use a new output path')
    import numpy as np
    from hi3d_workbench.asset_pipeline import load_texture_input,inspect_textured_glb
    from hi3d_workbench.engines.mesh_uv import match_source_vertices
    from hi3d_workbench.common import atomic_json,digest
    started=time.monotonic()
    original,_=load_texture_input(a.geometry);painted,_=load_texture_input(a.textured)
    inspect_textured_glb(a.textured)
    before=np.asarray(painted.vertex_normals).copy()
    _,normals=match_source_vertices(original,np.asarray(painted.vertices))
    painted.vertex_normals=normals
    a.output.parent.mkdir(parents=True,exist_ok=True);painted.export(a.output)
    restored,_=load_texture_input(a.output)
    assert np.array_equal(restored.vertices,painted.vertices)
    assert np.array_equal(restored.faces,painted.faces)
    assert np.array_equal(restored.visual.uv,painted.visual.uv)
    for key in ['baseColorTexture','metallicRoughnessTexture']:
        assert np.array_equal(np.asarray(getattr(restored.visual.material,key)),np.asarray(getattr(painted.visual.material,key)))
    info=inspect_textured_glb(a.output)
    stage=a.textured.with_suffix('.stage.json')
    inference=json.loads(stage.read_text(encoding='utf-8')) if stage.exists() else {}
    cosine=np.sum(before*normals,axis=1)/(np.linalg.norm(before,axis=1)*np.linalg.norm(normals,axis=1)+1e-12)
    result={'status':'succeeded','operation':'shading_normal_correction','texture_inference_rerun':False,
            'geometry_generated':False,'texture_generated':True,'material_pixels_unchanged':True,
            'positions_faces_uv_unchanged':True,'seconds':round(time.monotonic()-started,3),
            'source_textured_sha256':digest(a.textured),'normal_template_sha256':digest(a.geometry),
            'fraction_original_normals_cosine_below_09':float(np.mean(cosine<.9)),
            'output_sha256':digest(a.output),'original_inference':inference,'material_validation':info}
    atomic_json(a.output.with_suffix('.stage.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='original_inference'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
