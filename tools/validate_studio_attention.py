"""Compare the real packed backend with an independent float64 oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hi3d_workbench.studio.attention import packed_sdpa


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(31)
    checks=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        for amplitude in [1.,100000.]:
            qkv=torch.randn(20,3,2,16,device=device);qkv[:,2]*=amplitude
            offsets=[0,7,14,20]
            actual=packed_sdpa(qkv,offsets,scale=.25)
            expected=[]
            for start,end in zip(offsets,offsets[1:]):
                q,k,v=qkv[start:end].double().permute(1,2,0,3).unbind(0)
                expected.append((torch.softmax(q @ k.transpose(-1,-2)*.25,dim=-1) @ v).transpose(0,1))
            expected=torch.cat(expected).float();error=float((actual-expected).abs().max())
            if not torch.isfinite(actual).all() or error>=amplitude*3e-6:
                raise RuntimeError(f'{device} packed attention differs from float64 oracle: {error}')
            checks.append({'device':device,'value_amplitude':amplitude,'max_absolute_error':error,
                           'amplitude_normalized_error':error/amplitude,'points':20,'patch_lengths':[7,7,6]})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'checks':checks,'torch':torch.__version__},indent=2),encoding='utf-8')
    print(json.dumps(checks))


if __name__=='__main__':main()
