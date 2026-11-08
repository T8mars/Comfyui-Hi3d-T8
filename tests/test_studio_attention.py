import numpy as np
import pytest
import torch

from test_workbench import testroot
from hi3d_workbench.studio.attention import install_sonata_attention, packed_sdpa
from hi3d_workbench.studio.parts_models import validate_p3sam_features, validate_p3sam_probabilities


class Point(dict):
    __getattr__=dict.__getitem__
    __setattr__=dict.__setitem__


class Serialized(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enable_flash=True;self.enable_rpe=False
        self.num_heads=2;self.channels=4;self.order_index=0;self.scale=.37
        self.qkv=torch.nn.Linear(4,12);self.proj=torch.nn.Linear(4,4)
        self.proj_drop=torch.nn.Identity()

    def get_padding_and_inverse(self,point):
        return torch.tensor([0,1,2,3,4,3]),torch.arange(5),torch.tensor([0,3,6],dtype=torch.int32)

    def forward(self,point):
        raise AssertionError('The native half-precision attention must not be called')


def test_sonata_replacement_preserves_actual_padding_order_inverse_scale_and_projection():
    torch.manual_seed(8)
    layer=Serialized().eval();features=torch.randn(5,4)
    order=torch.tensor([2,0,4,1,3]);pad=torch.tensor([0,1,2,3,4,3])
    point=Point(feat=features.clone(),serialized_order=order[None],serialized_inverse=torch.argsort(order)[None])
    qkv=layer.qkv(features)[order[pad]].reshape(2,3,3,2,2).double()
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
    expected=(torch.softmax((q@k.transpose(-2,-1))*.37,dim=-1)@v).transpose(1,2).reshape(6,4)
    expected=expected[torch.argsort(order)].float()
    expected=layer.proj(expected)
    assert install_sonata_attention(layer,Serialized,label='P3-SAM')==1
    result=layer(point)
    assert result is point
    torch.testing.assert_close(result.feat,expected,atol=2e-7,rtol=2e-6)


def test_packed_attention_with_short_final_patch_matches_all_key_float64_reference():
    torch.manual_seed(11)
    qkv=torch.randn(10,3,2,4);offsets=[0,3,8,10]
    expected=[]
    for start,stop in zip(offsets,offsets[1:]):
        q,k,v=qkv[start:stop].double().permute(1,2,0,3)
        expected.append((torch.softmax((q@k.transpose(-2,-1))*.2,dim=-1)@v).transpose(0,1))
    actual=packed_sdpa(qkv,offsets,scale=.2)
    torch.testing.assert_close(actual,torch.cat(expected).float(),atol=3e-7,rtol=2e-6)


@pytest.mark.parametrize('state',['training','rpe','non_flash'])
def test_sonata_replacement_rejects_incompatible_configurations(state):
    layer=Serialized().eval()
    if state=='training':layer.train()
    elif state=='rpe':layer.enable_rpe=True
    else:layer.enable_flash=False
    with pytest.raises(RuntimeError,match='不兼容'):
        install_sonata_attention(layer,Serialized,label='P3-SAM')


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1,1.1])
def test_p3sam_rejects_bad_probabilities_before_boolean_mask_conversion(bad):
    values=[torch.full((2,5),.5) for _ in range(3)]+[torch.full((2,3),.5)]
    validate_p3sam_probabilities(values)
    values[1][0,0]=bad
    with pytest.raises(RuntimeError,match='掩码概率'):
        validate_p3sam_probabilities(values)


def test_p3sam_feature_guard_does_not_accept_nonfinite_or_mismatched_backbone_features():
    valid=torch.ones(5,4)
    assert validate_p3sam_features(valid,5) is valid
    for value in [np.ones((5,4)),torch.ones(4,4),torch.ones(5),torch.full((5,4),float('nan'))]:
        with pytest.raises(RuntimeError,match='点云特征'):
            validate_p3sam_features(value,5)
