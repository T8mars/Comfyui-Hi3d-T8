"""Validate a copied offline bundle without downloading or using source weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def verify(root,phase,check_only=False):
    from hi3d_workbench import common
    root=Path(root).resolve()
    if (common.ROOT!=root or common.MODELS!=root/'models'
            or not Path(common.__file__).resolve().is_relative_to(root)
            or not Path(sys.executable).resolve().is_relative_to(root)):
        raise RuntimeError('Portable verification must use the copied application, interpreter and models')
    if phase=='core':
        from hi3d_workbench import models
        result=None if check_only else models.verify_all()
        state=models.status()
        ready=state.get('ready') is True
        errors=[] if result is None else result.get('errors',[])
    elif phase=='pipeline':
        from hi3d_workbench import pipeline_models as models
        if models.MODEL_ROOT!=root/'models/extensions':raise RuntimeError('Wrong pipeline model directory')
        result=None if check_only else models.verify()
        state=models.status()
        ready=state.get('ready') is True
        # Optional engines included in the manifest must be valid as well.
        manifest=common.read_json(models.MODEL_ROOT/'manifest.json',{})
        invalid=[key for key in manifest.get('components',{}) if not state.get('components',{}).get(key,{}).get('ready')]
        errors=([] if result is None else result.get('verification_errors',[]))+invalid
    elif phase=='studio':
        from hi3d_workbench.studio import models
        if models.model_root()!=root/'models/studio':raise RuntimeError('Wrong asset model directory')
        result=None if check_only else models.verify()
        state=models.status()
        errors=([] if result is None else result.get('verification_errors',[]))
        errors += [key for key,row in state['components'].items() if not row['ready']]
        ready=not errors and all(row['ready'] for row in state['features'].values())
    else:
        raise ValueError('Unknown portable verification phase')
    receipt={'phase':phase,'check_only':check_only,'ready':ready and not errors,
             'errors':errors,'root':str(root),'python':str(Path(sys.executable).resolve())}
    common.atomic_json(root/'data/bundle-validation'/(phase+('-check' if check_only else '-verify')+'.json'),receipt)
    if not receipt['ready']:raise RuntimeError('Copied model verification failed: '+json.dumps(receipt,ensure_ascii=True))
    return receipt


if __name__=='__main__':
    # Embedded runtimes include the bundle root in their relative ._pth file.
    from hi3d_workbench.common import configure_stdio
    configure_stdio();parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['core','pipeline','studio'],required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.phase,args.check_only),ensure_ascii=False),flush=True)
