import pytest


def test_uv_report_preserves_cache_params_phases_and_failure_reasons(monkeypatch,tmp_path):
    from hi3d_workbench import common
    from hi3d_workbench.engines import mesh_uv,uv_runtime
    monkeypatch.setattr(common,'DATA',tmp_path)
    received=[];out=object();mesh=object()
    def ensure(value,**options):
        assert value is mesh
        assert options['cache_dir']==tmp_path/'uv-cache'
        assert options['texture_size']==4096
        options['emit'](stage='unwrapping',phase='uv_fallback',device='cpu',fallback_reason='known failure')
        options['emit'](stage='unwrapping',phase='uv_pack',device='cuda',packing_backend='torch',progress_current=1,progress_total=3)
        options['emit'](stage='unwrapping',phase='uv_validate',device='cpu',removed_degenerate_faces=2,local_uv_repaired_faces=1,uv_repair={'repaired_faces':1})
        return out,'verified'
    monkeypatch.setattr(mesh_uv,'ensure_uv',ensure)
    result,policy,report=uv_runtime.prepare(mesh,{'texture_size':4096},lambda stage,**event:received.append((stage,event)))
    assert result is out and policy=='verified'
    assert report['seconds']>=0 and report['fallback_reasons']==['known failure']
    assert report['details']['packing_backend']=='torch'
    assert report['details']['uv_repair']=={'repaired_faces':1}
    assert [phase['device'] for phase in report['phases']]==['cpu','cuda','cpu']
    assert received[1][0]=='unwrapping' and received[1][1]['progress_total']==3


def test_existing_uv_report_requires_no_fabricated_gpu_phase(monkeypatch):
    from hi3d_workbench.engines import mesh_uv,uv_runtime
    mesh=object()
    monkeypatch.setattr(mesh_uv,'ensure_uv',lambda value,**kw:(value,'preserved_existing'))
    result,policy,report=uv_runtime.prepare(mesh,{},lambda *a,**kw:pytest.fail('existing UV has no worker callbacks'))
    assert result is mesh and policy=='preserved_existing' and report['phases']==[]
