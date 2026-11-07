import numpy as np

from test_workbench import testroot
from hi3d_workbench.studio.material_projection import texture_sample,linear,srgb,frame,_raster_impl


def test_gltf_texture_origin_filtering_and_wrapping_are_not_flipped():
    image=np.array([[[255,0,0,255],[0,255,0,255]],[[0,0,255,255],[255,255,255,255]]],dtype=np.uint8)
    points=np.array([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
    np.testing.assert_allclose(texture_sample(image,points),image.reshape(-1,4)/255,atol=1e-7)
    np.testing.assert_allclose(texture_sample(image,np.array([[0,.25]]),{'wrapS':10497}),[[.5,.5,0,1]],atol=1e-7)
    np.testing.assert_allclose(texture_sample(image,np.array([[0,.25]]),{'wrapS':33071}),[[1,0,0,1]],atol=1e-7)
    np.testing.assert_allclose(texture_sample(image,np.array([[-.25,.25]]),{'wrapS':33648}),[[1,0,0,1]],atol=1e-7)


def test_srgb_interpolation_is_linear_but_alpha_is_not_gamma_corrected():
    image=np.array([[[0,0,0,0],[255,255,255,255]]],dtype=np.uint8)
    value=texture_sample(image,np.array([[.5,.5]]),color=True)
    np.testing.assert_allclose(value,[[.5,.5,.5,.5]],atol=1e-7)
    np.testing.assert_allclose(srgb(value[:,:3]),.73535698,atol=1e-7)
    samples=np.linspace(0,1,21)
    np.testing.assert_allclose(srgb(linear(samples)),samples,atol=1e-8)


def test_uv_raster_reconstructs_every_texel_centre_and_a_subpixel_triangle():
    triangles=np.array([[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]],dtype=float)
    face,bary=_raster_impl(triangles,8)
    assert (face>=0).all()
    weights=np.concatenate((bary,1-bary.sum(2,keepdims=True)),axis=2)
    actual=np.einsum('yxi,yxij->yxj',weights,triangles[face])
    y,x=np.mgrid[:8,:8]
    np.testing.assert_allclose(actual,np.stack([(x+.5)/8,(y+.5)/8],axis=-1),atol=1e-7)
    face,bary=_raster_impl(np.array([[[.001,.001],[.002,.001],[.001,.002]]]),8)
    assert (face>=0).sum()==1
    np.testing.assert_allclose(bary[0,0],[1/3,1/3],atol=1e-7)


def test_tangent_frame_retains_mirrored_uv_handedness():
    points=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    uv=np.array([[[0.,0],[1,0],[0,1]]]);normal=np.array([[0.,0,1]])
    t,b=frame(points,uv,normal)
    np.testing.assert_allclose(t,[[1,0,0]]);np.testing.assert_allclose(b,[[0,1,0]])
    uv[:,:,1]*=-1
    t,b=frame(points,uv,normal)
    np.testing.assert_allclose(t,[[1,0,0]]);np.testing.assert_allclose(b,[[0,-1,0]])
