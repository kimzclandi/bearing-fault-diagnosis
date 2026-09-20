"""Explicit synthetic input for interface checks, never experimental evidence."""
import argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',default='data/processed/synthetic_example.csv');a=p.parse_args()
rng=np.random.default_rng(42);fs=12000;t=np.arange(fs)/fs
x=.02*rng.normal(size=len(t))+.1*np.sin(2*np.pi*30*t)
for onset in range(0,len(t),110):
    k=np.arange(min(200,len(t)-onset));x[onset:onset+len(k)]+=.4*np.exp(-k/30)*np.sin(2*np.pi*3000*k/fs)
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
np.savetxt(path,x,delimiter=',',header='signal',comments='')
print(f'Synthetic signal only: {path}; sampling rate={fs} Hz; no ground-truth experimental result.')
