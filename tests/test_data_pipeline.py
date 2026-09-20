import numpy as np
import pytest
from bearing_diagnosis.signal import validate_signal,preprocess,windows
from bearing_diagnosis.training import fit_normalizer,normalize
from bearing_diagnosis.config import load_config

@pytest.mark.parametrize('x',[np.ones(200),np.zeros((2,200)),np.array([1,np.nan]*100),np.arange(10)])
def test_bad_signal(x):
    with pytest.raises(ValueError): validate_signal(x)

def test_resample_and_windows():
    cfg=load_config('configs/quick.yaml');t=np.arange(48000)/48000;x=np.sin(2*np.pi*1000*t)+3
    y=preprocess(x,48000,cfg);assert len(y)==12000
    chunks,starts,means=windows(y,4096,0)
    assert starts.tolist()==[0,4096];assert np.max(np.abs(chunks.mean(axis=1)))<1e-6
    assert np.allclose(means,3,atol=.01)

def test_training_only_normalization():
    train=np.arange(20,dtype=float).reshape(2,10);test=train+1000
    stats=fit_normalizer(train);assert stats['mean']==9.5
    assert abs(normalize(train,stats).mean())<1e-6
    assert normalize(test,stats).mean()>100
