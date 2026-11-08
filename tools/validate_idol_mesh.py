"""Run a real fixed upstream mesh conversion on a saved Idol splat."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=384)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    from hi3d_workbench.engines.idol import initialize,mesher,release
    initialize()
    import torch
    from io import BytesIO
    from comfy_api.latest import Types
    from comfy_extras.nodes_gaussian_splat import File3DToSplat
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
    from hi3d_workbench.common import atomic_json,digest
    from hi3d_workbench.asset_pipeline import inspect_asset_geometry
    splat=File3DToSplat.execute(Types.File3D(BytesIO(args.input.read_bytes()),file_format='ply'))[0]
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():mesh=mesher('solid').execute(splat,args.resolution,5,3,.6,500,.02,2.,solid=True,fill=20.)[0]
    data=mesh_item_to_glb_bytes(mesh,0)
    if not data:raise ValueError('No surface')
    args.output.write_bytes(data)
    atomic_json(args.output.with_suffix('.validation.json'),{'seconds':time.monotonic()-started,
        'input_sha256':digest(args.input),'output_sha256':digest(args.output),
        'geometry':inspect_asset_geometry(args.output),'torch_peak_mib':torch.cuda.max_memory_allocated()/2**20})
    release();print('Converted real saved splat:',args.output)


if __name__=='__main__':main()
