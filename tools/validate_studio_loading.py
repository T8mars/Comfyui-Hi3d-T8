"""Exercise real pinned Transformers loading progress with a tiny local model."""
import json
import copy
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime-assets/engines/kimodo-v510'),str(ROOT)]
import torch
from transformers import LlamaConfig,LlamaModel,core_model_loading
from hi3d_workbench.studio.loading_progress import weight_loading,merge_adapter


def main():
    folder=ROOT/'data/asset-studio-validation'
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='loader-fixture-',dir=folder) as temporary:
        config=LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
        torch.manual_seed(23);original=LlamaModel(config)
        original.save_pretrained(temporary)
        events=[];logging=core_model_loading.logging
        with weight_loading(lambda stage,message,**values:events.append({'stage':stage,'message':message,**values}),'cpu'):
            loaded=LlamaModel.from_pretrained(temporary,local_files_only=True,dtype=torch.float32)
        assert core_model_loading.logging is logging
        for key,value in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key],value,atol=0,rtol=0)
        counts=[event['progress_current'] for event in events]
        assert counts[0]==0 and counts[-1]==events[-1]['progress_total']
        assert counts==sorted(counts) and len(events)>4
        try:
            with weight_loading(lambda *a,**k:None,'cpu'):
                raise RuntimeError('fixture')
        except RuntimeError:pass
        assert core_model_loading.logging is logging
        from peft import get_peft_model,LoraConfig
        adapter=get_peft_model(copy.deepcopy(original),LoraConfig(r=2,lora_alpha=4,
            target_modules=['q_proj','v_proj'],init_lora_weights=False))
        reference=copy.deepcopy(adapter).merge_and_unload()
        layers=[layer for layer in adapter.modules() if hasattr(layer,'lora_A')]
        merged=merge_adapter(adapter,lambda stage,message,**values:events.append(
            {'stage':stage,'message':message,**values}),'cpu')
        for key,value in reference.state_dict().items():
            torch.testing.assert_close(merged.state_dict()[key],value,atol=0,rtol=0)
        assert all('merge' not in layer.__dict__ for layer in layers)
        merges=[event for event in events if event['stage']=='asset_motion_text_adapter']
        assert merges[0]['progress_current']==0 and merges[-1]['progress_current']==len(layers)==2
        # A failed merge must also restore the actual bound layer method.
        failed=get_peft_model(copy.deepcopy(original),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj']))
        failed_layer=next(layer for layer in failed.modules() if hasattr(layer,'lora_A'))
        old=failed_layer.merge
        def explode(*a,**kw):raise RuntimeError('intentional merge failure')
        failed_layer.merge=explode
        try:
            merge_adapter(failed,lambda *a,**kw:None,'cpu')
        except RuntimeError as error:assert str(error)=='intentional merge failure'
        else:raise AssertionError('Merge failure was hidden')
        assert failed_layer.merge is explode
        failed_layer.merge=old
        result={'status':'passed','checkpoint':'tiny local Llama fixture','parameters_unchanged':True,
                'exception_restored_logger':True,'adapter_merge_unchanged':True,
                'adapter_exception_restored':True,'events':events}
        (folder/'studio-loading-progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'status':'passed','events':len(events),'materialized_tensors':counts[-1]}))


if __name__=='__main__':main()
