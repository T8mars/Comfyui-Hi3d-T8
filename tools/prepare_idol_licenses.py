"""Retain the exact upstream notices for the independently licensed Idol chain."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'Idol-Apache-2.0.txt':'https://raw.githubusercontent.com/speridlabs/idol-sdk/4702a7f00850e34b6e83a2170cc123c3e9570744/LICENSE',
    'DINOv3-LICENSE.md':'https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/c807c9eeea853df70aec4069e6f56b28ddc82acc/LICENSE.md',
    'BiRefNet-MIT.txt':'https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/f7079632d681ecf12161fcea8905682934e59e51/LICENSE',
    'TripoSplat-MIT.txt':'https://raw.githubusercontent.com/VAST-AI-Research/TripoSplat/d8db9e018b413dd9c4a9fe22463781bf98e8e68d/LICENSE',
    'FLUX2-autoencoder-README.md':'https://raw.githubusercontent.com/black-forest-labs/flux2/50fe5162777813d869182b139e83b10743caef15/README.md',
    'Comfy-Idol-README.md':'https://huggingface.co/Comfy-Org/Idol/resolve/3d050f4bc705ca1202a0ae4d21d2d97987521307/README.md',
}


def main():
    sys.path.insert(0,str(ROOT))
    from hi3d_workbench.common import atomic_json
    folder=ROOT/'licenses/idol';folder.mkdir(parents=True,exist_ok=True)
    files={}
    for name,url in SOURCES.items():
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read(1024*1024)
        if not data or data.lstrip().startswith(b'<!DOCTYPE'):raise ValueError('Invalid notice response: '+name)
        (folder/name).write_bytes(data)
        files[name]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    core=ROOT/'.pipelineenv/sources/comfy_core/LICENSE'
    if not core.is_file():raise FileNotFoundError(core)
    data=core.read_bytes();(folder/'ComfyUI-GPL-3.0.txt').write_bytes(data)
    files['ComfyUI-GPL-3.0.txt']={'source':'Comfy-Org/ComfyUI@7cebca52227c63b4180ceed448cbd7ad22903b6d/LICENSE',
                              'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    atomic_json(folder/'manifest.json',{'scope':'Idol and its independent model/code components; not a blanket license',
        'flux2_vae':'The autoencoder is Apache-2.0 as stated separately by BFL; Idol uses no FLUX generator or text encoder.',
        'files':files})
    print(json.dumps({'saved':len(files),'folder':str(folder)}))


if __name__=='__main__':main()
