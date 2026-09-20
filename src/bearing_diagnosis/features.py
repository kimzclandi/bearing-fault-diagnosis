import numpy as np
from scipy.stats import skew, kurtosis
from .signal import spectrum, envelope_spectrum
from .physics import bearing_frequencies

def extract(x,rpm,cfg):
    x=np.asarray(x,dtype=np.float64); eps=1e-12
    std=x.std(); rms=np.sqrt(np.mean(x*x)); peak=np.max(np.abs(x)); absolute=np.mean(np.abs(x))
    f,p=spectrum(x,cfg['sampling_rate']); df=f[1]-f[0]; total=p.sum()*df
    out={'mean':float(x.mean()),'std':std,'rms':rms,'peak_to_peak':np.ptp(x),
         'skewness':float(skew(x)) if std>eps else 0.,'kurtosis':float(kurtosis(x,fisher=False)) if std>eps else 0.,
         'crest_factor':peak/(rms+eps),'shape_factor':rms/(absolute+eps),
         'impulse_factor':peak/(absolute+eps),'clearance_factor':peak/(np.mean(np.sqrt(np.abs(x)))**2+eps),
         'spectral_energy':total,'dominant_frequency':f[np.argmax(p)],
         'spectral_centroid':np.sum(f*p)/(p.sum()+eps)}
    for lo,hi in [(0,500),(500,1500),(1500,3000),(3000,6000)]:
        mask=(f>=lo)&(f<hi if hi<6000 else f<=hi)
        out[f'band_{lo}_{hi}_ratio']=p[mask].sum()*df/(total+eps)
    ef,ep=envelope_spectrum(x,cfg['sampling_rate'],cfg['envelope_band'])
    denom=ep.sum()+eps; tol=max(cfg['frequency_tolerance_hz'],df)
    for name,hz in bearing_frequencies(rpm,**cfg['geometry']).items():
        for harmonic in [1,2,3]:
            out[f'physics_{name}_h{harmonic}']=ep[np.abs(ef-hz*harmonic)<=tol].sum()/denom
    return out

def extract_batch(x,rpm,cfg):
    rows=[extract(a,float(r),cfg) for a,r in zip(x,rpm)]
    return np.array([list(r.values()) for r in rows],dtype=np.float64),list(rows[0])
