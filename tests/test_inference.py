import numpy as np
import torch
from bearing_diagnosis.models import BearingCNN
from bearing_diagnosis.training import predict_cnn
from bearing_diagnosis.evaluation import perturb

def test_model_shape_and_probabilities():
    torch.set_num_threads(1);model=BearingCNN();x=np.random.default_rng(1).normal(size=(3,4096)).astype('float32')
    p=predict_cnn(model,x,{'mean':0.,'std':1.})
    assert p.shape==(3,4);assert np.allclose(p.sum(1),1)

def test_noise_snr_and_determinism():
    rng=np.random.default_rng(7);x=rng.normal(size=(80,4096)).astype('float32');x-=x.mean(1,keepdims=True)
    a=perturb(x,'noise',10,np.random.default_rng(5));b=perturb(x,'noise',10,np.random.default_rng(5))
    assert np.array_equal(a,b)
    assert abs(10*np.log10(np.mean(x*x)/np.mean((a-x)**2))-10)<.1

def test_missing_segments():
    x=np.tile(np.sin(np.arange(4096)),(4,1)).astype('float32')
    a=perturb(x,'missing',.1,np.random.default_rng(5))
    assert a.shape==x.shape and np.isfinite(a).all()
    assert (np.abs(np.diff(a,axis=1))<1e-10).sum()>=4*(round(4096*.1)-1)


def test_upload_rejects_objects_and_multiple_columns():
    import io,pytest
    from bearing_diagnosis.signal import parse_upload
    with pytest.raises(ValueError): parse_upload(b'1,2\n'*200,'bad.csv')
    buffer=io.BytesIO();np.save(buffer,np.array([{'a':1}],dtype=object))
    with pytest.raises(ValueError): parse_upload(buffer.getvalue(),'bad.npy')
    x=np.arange(200,dtype=float)
    parsed=parse_upload(('signal\n'+'\n'.join(map(str,x))).encode(),'ok.csv')
    assert np.array_equal(x,parsed)
