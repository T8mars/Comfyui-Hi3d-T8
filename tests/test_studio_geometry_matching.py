import pytest
import torch

from test_workbench import testroot
from hi3d_workbench.studio.geometry_matching import nearest_indices, install_xpart_conditioning


def oracle(query, reference):
    return ((query.double()[:, :, None] - reference.double()[:, None]) ** 2).sum(-1).argmin(-1)


@pytest.mark.parametrize('budget',[1,13,127,100000])
def test_nearest_matching_checks_every_point_with_bounded_temporaries(monkeypatch,budget):
    torch.manual_seed(13);query=torch.randn(2,13,3);reference=torch.randn(1,47,3)
    expected=oracle(query,reference)
    original=torch.cdist;allocations=[];progress=[]
    def tracked(left,right,**kwargs):
        result=original(left,right,**kwargs)
        allocations.append(result.numel())
        assert kwargs['compute_mode']=='donot_use_mm_for_euclid_dist'
        return result
    monkeypatch.setattr(torch,'cdist',tracked)
    actual=nearest_indices(query,reference,matrix_elements=budget,progress=lambda *args:progress.append(args))
    assert torch.equal(actual,expected)
    assert allocations and max(allocations)<=budget
    assert progress[-1]==(26,26)


@pytest.mark.parametrize('budget',[5,13,10000])
def test_nearest_matching_resolves_close_samples_and_first_ties_across_blocks(budget):
    query=torch.tensor([[[.3,.3,.3]]]);reference=torch.ones(1,47,3)
    reference[0,0]=query[0,0]+1e-6
    reference[0,3]=query[0,0];reference[0,41]=query[0,0]
    assert nearest_indices(query,reference,matrix_elements=budget).item()==3
    assert torch.equal(nearest_indices(query,reference,matrix_elements=budget),oracle(query,reference))


def test_nearest_matching_preserves_empty_query_batch_shape():
    progress=[]
    result=nearest_indices(torch.empty(2,0,3),torch.ones(1,8,3),progress=lambda *row:progress.append(row))
    assert result.shape==(2,0) and result.dtype==torch.int64 and progress==[(0,0)]


@pytest.mark.parametrize('problem',['nan','empty_reference','different_dtype','batch','budget'])
def test_nearest_matching_rejects_invalid_input_before_search(problem):
    query=torch.ones(2,3,3);reference=torch.ones(2,4,3);options={}
    if problem=='nan':query[0,0,0]=float('nan')
    elif problem=='empty_reference':reference=torch.empty(2,0,3)
    elif problem=='different_dtype':reference=reference.double()
    elif problem=='batch':reference=torch.ones(3,4,3)
    else:options['matrix_elements']=0
    with pytest.raises(ValueError,match='最近点匹配'):
        nearest_indices(query,reference,**options)


def test_conditioner_retains_full_batch_encoders_features_projections_and_rng():
    torch.manual_seed(7);parts=torch.randn(2,6,6);surface=torch.randn(1,11,6).expand(2,-1,-1)
    local_query=torch.randn(2,5,3);global_query=torch.randn(2,3,3)
    local_features=torch.randn(2,5,2);global_features=torch.randn(2,3,2)
    semantics=torch.randn(1,11,3);calls=[];events=[]
    class Geo(torch.nn.Module):
        def forward(self,a,b,**kwargs):
            assert a is parts and b is surface and len(a)==2 and kwargs=={'return_local_pc_info':True}
            calls.append('all-parts-geo');return local_features,(local_query,)
    class Obj(torch.nn.Module):
        def encode_shape(self,value,**kwargs):
            assert value is surface and kwargs=={'return_pc_info':True}
            calls.append('all-parts-object');return global_features,(global_query,)
    class Seg(torch.nn.Module):
        def forward(self,points,normals):
            assert torch.equal(points,surface[:1,:,:3]) and torch.equal(normals,surface[:1,:,3:6])
            calls.append('one-shared-object-semantics');return semantics
    class Conditioner(torch.nn.Module):
        def __init__(self):
            super().__init__();self.use_image=False;self.use_geo=True;self.use_obj=True;self.use_seg_feat=True
            self.geo_encoder=Geo();self.obj_encoder=Obj();self.seg_feat_encoder=Seg()
            self.geo_out_proj=torch.nn.Linear(5,4);self.obj_out_proj=torch.nn.Linear(5,4)
    model=Conditioner().eval();expected={}
    for key,query,features,projection in [('geo_cond',local_query,local_features,model.geo_out_proj),
                                        ('obj_cond',global_query,global_features,model.obj_out_proj)]:
        index=oracle(query,surface[...,:3])
        selected=semantics.expand(2,-1,-1).gather(1,index[...,None].expand(-1,-1,3))
        expected[key]=projection(torch.cat([features,selected],-1))
    rng=torch.get_rng_state()
    install_xpart_conditioning(model,lambda stage,message,**kwargs:events.append((stage,message,kwargs)))
    actual=model(parts,surface)
    assert calls==['all-parts-geo','all-parts-object','one-shared-object-semantics']
    assert torch.equal(torch.get_rng_state(),rng)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert len(model._hi3d_matching_report)==2 and all(row['all_reference_points_tested'] for row in model._hi3d_matching_report)
    assert events[-1][2]['progress_current']==events[-1][2]['progress_total']==10
