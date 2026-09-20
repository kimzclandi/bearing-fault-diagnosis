from pathlib import Path
import io,json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from bearing_diagnosis.config import ROOT,LABELS
from bearing_diagnosis.signal import parse_upload,validate_signal,preprocess,windows,spectrum,envelope_spectrum
from bearing_diagnosis.features import extract
from bearing_diagnosis.physics import bearing_frequencies
from bearing_diagnosis.training import load_cnn,predict_cnn
from bearing_diagnosis.explain import saliency

st.set_page_config(page_title='Bearing vibration diagnostics',layout='wide')
st.title('Bearing vibration diagnostics')
st.caption('Held-out-condition analysis with physical frequency markers. Health deviation is not remaining useful life.')
runs=sorted([p for p in (ROOT/'outputs').glob('*') if (p/'models/normalizer.json').exists()],key=lambda p:p.stat().st_mtime)
if not runs:
    st.info('No trained model found. Run: python scripts/run_all.py --config configs/quick.yaml --download')
    st.stop()
run=Path(st.selectbox('Experiment',runs,format_func=lambda p:p.name,index=len(runs)-1))
cfg=json.loads((run/'config.json').read_text());seed=cfg['seeds'][0]
import torch
torch.set_num_threads(cfg['threads'])
mode=st.radio('Input',['Local example','Upload'])
st.write('Upload: one numeric column in CSV (optional header signal), or a 1D numeric NPY array. At least one 4096-point window at 12 kHz is required. NPY object arrays are rejected. No automatic channel selection.')
try:
    if mode=='Local example':
        meta=pd.read_csv(run/'metrics/windows.csv');test=meta[meta.split=='test']
        idx=st.selectbox('Window',test.index.tolist(),format_func=lambda i:str(meta.loc[i,'sample_id']))
        with np.load(run/'cache/dataset.npz') as data: x=data['X'][idx].copy()
        from scipy.io import loadmat
        source_table=pd.read_csv(ROOT/cfg['manifest'],dtype={'source_id':str})
        source_id=str(meta.loc[idx,'source_id']);source=source_table[source_table.source_id==source_id].iloc[0]
        record=loadmat(ROOT/cfg['data_dir']/(source_id+'.mat'))
        from bearing_diagnosis.data import select_channel
        key,_,_=select_channel(record,source_id)
        raw=record[key].squeeze()
        original_fs=int(source.sampling_rate);original_start=round(meta.loc[idx,'start']*original_fs/cfg['sampling_rate'])
        original=raw[original_start:original_start+round(cfg['window_size']*original_fs/cfg['sampling_rate'])]
        rpm=float(meta.loc[idx,'rpm']);st.caption(f'Sample {meta.loc[idx,"sample_id"]}; reference class: {LABELS[int(meta.loc[idx,"label"])]}')
    else:
        upload=st.file_uploader('Signal file',type=['csv','npy'])
        fs=int(st.number_input('Sampling rate (Hz)',min_value=12000,value=12000,step=1000))
        rpm=float(st.number_input('Rotation speed (rpm)',min_value=1.,value=1750.))
        if upload is None: st.stop()
        raw=parse_upload(upload.getvalue(),upload.name)
        raw=validate_signal(raw);clean=preprocess(raw,fs,cfg)
        chunks,starts,_=windows(clean,cfg['window_size'],0)
        selected=st.slider('Window index',0,len(chunks)-1,0) if len(chunks)>1 else 0
        x=chunks[selected]
        original_start=int(starts[selected]*fs/cfg['sampling_rate']);original_end=original_start+round(cfg['window_size']*fs/cfg['sampling_rate'])
        original=raw[original_start:original_end];original_fs=fs
        st.caption(f'Analyzing window {selected} of {len(chunks)}; original sampling rate {fs} Hz.')
    validate_signal(x,cfg['window_size'])
    geometry=cfg['geometry'].copy()
    with st.expander('Bearing geometry: use one common length unit'):
        geometry['rolling_elements']=int(st.number_input('Rolling elements',min_value=2,value=int(geometry['rolling_elements'])))
        geometry['ball_diameter']=st.number_input('Ball diameter',min_value=.0001,value=float(geometry['ball_diameter']),format='%.4f')
        geometry['pitch_diameter']=st.number_input('Pitch diameter',min_value=.0001,value=float(geometry['pitch_diameter']),format='%.4f')
        geometry['contact_angle_deg']=st.number_input('Contact angle (degrees)',min_value=0.,max_value=89.,value=float(geometry['contact_angle_deg']))
    freq=bearing_frequencies(rpm,**geometry)
    st.caption('Geometry controls change the physical overlay only. Model features retain the recorded training geometry.')
    fig,ax=plt.subplots(figsize=(11,2.5));ax.plot(np.arange(len(original))/original_fs,original,lw=.6);ax.set_title('Original acquired signal segment');ax.set_xlabel('Time (s)');st.pyplot(fig);plt.close(fig)
    features=extract(x,rpm,cfg);bundle=joblib.load(run/f'models/rf_{seed}.joblib')
    F=np.array(list(features.values()))[None];rf=bundle['model'].predict_proba(F[:,bundle['columns']])[0]
    model,stats=load_cnn(run,seed);cnn=predict_cnn(model,x[None],stats)[0]
    for label,prob in [('Random Forest',rf),('1D CNN',cnn)]:
        st.write(f'{label}: **{LABELS[int(prob.argmax())]}**, uncalibrated confidence {prob.max():.3f}')
    st.dataframe(pd.DataFrame({'class':LABELS,'Random Forest':rf,'1D CNN':cnn}),hide_index=True)
    fig,axes=plt.subplots(3,1,figsize=(11,8));t=np.arange(len(x))/cfg['sampling_rate']
    axes[0].plot(t,x,lw=.6);axes[0].set_xlabel('Time (s)');axes[0].set_title('Input window after preprocessing')
    f,p=spectrum(x,cfg['sampling_rate']);axes[1].plot(f,p);axes[1].set_xlabel('Frequency (Hz)');axes[1].set_title('Vibration PSD')
    ef,ep=envelope_spectrum(x,cfg['sampling_rate'],cfg['envelope_band']);axes[2].plot(ef,ep)
    for color,(name,hz) in zip(['tab:orange','tab:green','tab:red','tab:purple'],freq.items()):
        for harmonic in [1,2,3]: axes[2].axvline(hz*harmonic,ls='--',alpha=.65,color=color,label=name if harmonic==1 else None)
    axes[2].set_xlim(0,600);axes[2].set_title('Envelope PSD and ideal fault frequencies');axes[2].set_xlabel('Frequency (Hz)');axes[2].legend()
    fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.dataframe(pd.DataFrame({'feature':list(features),'value':list(features.values())}),hide_index=True)
    st.write('Ideal fault frequencies (Hz)',freq)
    fig,ax=plt.subplots(figsize=(11,2.5));ax.plot(t,saliency(model,x,stats),lw=.7);ax.set_xlabel('Time (s)');ax.set_ylabel('Absolute gradient');st.pyplot(fig);plt.close(fig)
    st.caption('Input gradients show local sensitivity to the predicted logit; they do not prove a causal fault mechanism.')
except (ValueError,KeyError,OSError,TypeError,EOFError,UnicodeError) as exc:
    st.error(f'Cannot analyze this input: {exc}')
