"""Real CPU/CUDA float64 oracle for X-Part's query-bounded attention."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from hi3d_workbench.studio.attention import bounded_sdpa


def main():
    results=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        for amplitude in [1.,100000.]:
            torch.manual_seed(23)
            q=torch.randn(2,3,17,8,device=device)
            k=torch.randn(2,3,37,8,device=device)
            v=torch.randn(2,3,37,8,device=device)*amplitude
            for masked in [False,True]:
                mask=torch.rand(17,37,device=device)>.2 if masked else None
                expected=torch.nn.functional.scaled_dot_product_attention(q.double(),k.double(),v.double(),attn_mask=mask,scale=.4)
                actual=bounded_sdpa(q,k,v,attn_mask=mask,scale=.4,matrix_elements=700)
                error=float((actual.double()-expected).abs().max()/max(1.,float(expected.abs().max())))
                assert error<2e-6,(device,amplitude,masked,error)
                results.append({'device':device,'value_amplitude':amplitude,'masked':masked,'normalized_max_error':error})
    path=ROOT/'data/asset-studio-validation/cross-attention-check.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2))
    print(json.dumps(results))


if __name__=='__main__':main()
