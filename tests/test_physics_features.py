import numpy as np
import pytest
from bearing_diagnosis.physics import bearing_frequencies
from bearing_diagnosis.signal import spectrum
from bearing_diagnosis.features import extract
from bearing_diagnosis.config import load_config

def test_frequency_equations_and_scaling():
    f=bearing_frequencies(1800);double=bearing_frequencies(3600)
    assert f['BPFO']/30==pytest.approx(3.5848,abs=1e-4)
    assert f['BPFI']/30==pytest.approx(5.4152,abs=1e-4)
    assert 2*f['BSF']/30==pytest.approx(4.7135,abs=1e-4)
    for key in f: assert double[key]==pytest.approx(2*f[key])

@pytest.mark.parametrize('kwargs',[{'rpm':0},{'rpm':1800,'ball_diameter':2},{'rpm':float('nan')},{'rpm':1800,'rolling_elements':2.5}])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError): bearing_frequencies(**kwargs)

def test_parseval_and_frequency():
    fs=12000;n=4096;t=np.arange(n)/fs;hz=fs/n*100;x=2*np.sin(2*np.pi*hz*t)
    f,p=spectrum(x,fs);assert np.sum(p)*(f[1]-f[0])==pytest.approx(np.mean(x*x))
    features=extract(x,1800,load_config('configs/quick.yaml'))
    assert features['dominant_frequency']==pytest.approx(hz)
    assert features['rms']==pytest.approx(np.sqrt(2))
    assert sum(v for k,v in features.items() if k.startswith('band_'))==pytest.approx(1)
