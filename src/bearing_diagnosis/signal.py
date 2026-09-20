from math import gcd
import numpy as np
from scipy.signal import resample_poly, butter, sosfiltfilt, hilbert, periodogram

def validate_signal(x, minimum=128):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or len(x)<minimum: raise ValueError(f'Expected a 1D signal with at least {minimum} samples.')
    if not np.isfinite(x).all(): raise ValueError('Signal contains NaN or Inf.')
    if np.std(x)<1e-12: raise ValueError('Signal is constant or has negligible variation.')
    return x

def preprocess(x, fs, cfg):
    x=validate_signal(x)
    if not np.isfinite(fs) or int(fs)!=fs or fs<2: raise ValueError('Sampling rate must be a positive integer.')
    target=cfg['sampling_rate']; fs=int(fs)
    if fs<target: raise ValueError('Input sampling rate below protocol rate; upsampling is disabled.')
    if fs!=target:
        g=gcd(fs,target); x=resample_poly(x,target//g,fs//g)
    if cfg['filter_band'] is not None: x=bandpass(x,target,cfg['filter_band'])
    return x

def bandpass(x,fs,band):
    lo,hi=band
    if not 0<lo<hi<fs/2: raise ValueError('Band must lie strictly inside (0, Nyquist).')
    return sosfiltfilt(butter(4,[lo,hi],btype='bandpass',fs=fs,output='sos'),x)

def windows(x,size,overlap=0,limit=None):
    if not 0<=overlap<1 or size<2: raise ValueError('Invalid window parameters.')
    step=max(1,round(size*(1-overlap)))
    starts=np.arange(0,len(x)-size+1,step,dtype=int)
    if limit is not None and len(starts)>limit:
        starts=starts[np.linspace(0,len(starts)-1,limit,dtype=int)]
    if not len(starts): raise ValueError('Signal shorter than one window after resampling.')
    chunks=np.stack([x[s:s+size] for s in starts])
    means=chunks.mean(axis=1)
    return (chunks-means[:,None]).astype(np.float32),starts,means

def spectrum(x,fs):
    return periodogram(x,fs=fs,window='boxcar',detrend=False,scaling='density')

def envelope_spectrum(x,fs,band):
    envelope=np.abs(hilbert(bandpass(x,fs,band)))
    return spectrum(envelope-envelope.mean(),fs)


def parse_upload(content, filename):
    import io
    if len(content)>20*1024*1024: raise ValueError('File exceeds the 20 MB limit.')
    if filename.lower().endswith('.npy'):
        x=np.load(io.BytesIO(content),allow_pickle=False)
    elif filename.lower().endswith('.csv'):
        text=content.decode('utf-8-sig');lines=text.splitlines()
        if lines and lines[0].strip()=='signal': text='\n'.join(lines[1:])
        x=np.loadtxt(io.StringIO(text),delimiter=',',ndmin=1)
    else: raise ValueError('Expected CSV or NPY.')
    return validate_signal(x)
