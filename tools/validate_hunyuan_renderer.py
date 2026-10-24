"""CUDA position-map round trip through the real Hunyuan camera and UV baker.

This is an adapter diagnostic, not AI texture-quality evidence.
"""
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('use a new diagnostic output path')
    import torch
    from types import SimpleNamespace
    from hi3d_workbench.asset_pipeline import load_texture_input
    from hi3d_workbench.common import atomic_json,digest
    from hi3d_workbench.engines.run_stage import source_dir
    from hi3d_workbench.engines.hunyuan_renderer import load_renderer
    source=source_dir('hunyuan21')
    Renderer=load_renderer(source)
    from utils.pipeline_utils import ViewProcessor
    mesh,policy=load_texture_input(args.mesh)
    if getattr(mesh.visual,'uv',None) is None:p.error('diagnostic requires an existing UV map')
    renderer=Renderer(default_resolution=512,texture_size=1024,bake_mode='back_sample',raster_mode='cr')
    renderer.load_mesh(mesh=mesh)
    processor=ViewProcessor(SimpleNamespace(bake_exp=4),renderer)
    elevations=[0,0,0,0,90,-90];azimuths=[0,90,180,270,0,180]
    images=processor.render_position_multiview(elevations,azimuths)
    baked,mask=processor.bake_from_multiview(images,elevations,azimuths,[1,.1,.5,.1,.05,.05])
    expected=renderer.uv_feature_map(.5-renderer.vtx_pos/renderer.scale_factor)
    if expected.ndim==4:expected=expected[0]
    valid=mask.squeeze(-1)>0
    errors=(baked[valid]-expected[valid]).abs()
    result={'passed':bool(errors.mean()<.015 and torch.quantile(errors.flatten(),.95)<.03),
            'metric':'normalized XYZ color round trip on trusted texels; not an AI material score',
            'input_sha256':digest(args.mesh),'uv_policy':policy,'view_resolution':512,'atlas_resolution':1024,
            'mean_absolute_error':float(errors.mean()),'p95_absolute_error':float(torch.quantile(errors.flatten(),.95)),
            'trusted_texels':int(valid.sum()),'implementation_sha256':digest(ROOT/'hi3d_workbench/engines/hunyuan_renderer.py')}
    atomic_json(args.output,result);print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
