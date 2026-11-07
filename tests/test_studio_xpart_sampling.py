import numpy as np
import pytest

from test_workbench import testroot
from hi3d_workbench.studio.xpart_sampling import guided_prediction


def test_guidance_keeps_part_attention_groups_separate_and_shares_embedding_randomness():
    torch=pytest.importorskip('torch')
    x=torch.tensor([[[1.]],[[2.]]]);condition={'geo_cond':torch.tensor([[[2.]],[[3.]]])}
    calls=[]
    class Model:
        def __call__(self,latents,timestep,context,**kwargs):
            token=torch.randperm(50)[0].float();calls.append((len(latents),token.item()))
            return latents+latents.sum(0,keepdim=True)+token+context['geo_cond']
    torch.manual_seed(23);offset=torch.randperm(50)[0].float();expected_state=torch.get_rng_state()
    expected=x+x.sum(0,keepdim=True)+offset+7.5*condition['geo_cond']
    torch.manual_seed(23)
    actual=guided_prediction(Model(),x,torch.zeros(2),condition,boxes=torch.zeros(1,2,2,3),
        num_tokens=torch.ones(1,2,dtype=torch.int64),guidance=7.5)
    torch.testing.assert_close(actual,expected)
    assert len(calls)==2 and calls[0]==calls[1] and calls[0][0]==2
    assert torch.equal(torch.get_rng_state(),expected_state)


def test_failed_unconditional_branch_restores_the_conditional_rng_state():
    torch=pytest.importorskip('torch')
    calls=[]
    def model(x,t,c,**kwargs):
        calls.append(1);torch.randperm(20)
        if len(calls)==2:raise RuntimeError('fixture failure')
        return x
    torch.manual_seed(5);torch.randperm(20);expected=torch.get_rng_state()
    torch.manual_seed(5)
    with pytest.raises(RuntimeError,match='fixture failure'):
        guided_prediction(model,torch.ones(2,1,1),torch.zeros(2),{'value':torch.ones(2,1,1)},
            boxes=torch.zeros(1,2,2,3),num_tokens=torch.ones(1,2),guidance=7.5)
    assert torch.equal(torch.get_rng_state(),expected)


@pytest.mark.parametrize('guidance',[np.nan,np.inf,16,-2])
def test_guidance_rejects_invalid_values_before_model_execution(guidance):
    torch=pytest.importorskip('torch')
    def unexpected(*args,**kwargs):raise AssertionError('Unexpected model call')
    with pytest.raises(ValueError,match='引导强度'):
        guided_prediction(unexpected,torch.ones(2,1,1),torch.zeros(2),{},boxes=None,num_tokens=None,guidance=guidance)
