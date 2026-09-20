import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import load_dataset
from .evaluation import savefig
from .config import LABELS,save_json

def health_index(cfg,out):
    _,F,y,m=load_dataset(out);names=json.loads((out/'cache/feature_names.json').read_text())
    cols=[i for i,n in enumerate(names) if n in ['rms','kurtosis'] or n.startswith('physics_')]
    reference=F[(m.split.eq('train').values)&(y==0)][:,cols]
    center=np.median(reference,axis=0);scale=np.maximum(1.4826*np.median(np.abs(reference-center),axis=0),1e-6)
    # Log compression preserves rank and limits domination by a single extreme feature.
    scores=np.mean(np.log1p(np.abs((F[:,cols]-center)/scale)),axis=1)
    threshold=float(np.quantile(scores[(m.split.eq('train').values)&(y==0)],.99))
    frame=m.copy();frame['health_deviation']=scores;frame['time_s']=frame.start/cfg['sampling_rate']
    frame['causal_smoothed']=frame.groupby('source_id').health_deviation.transform(lambda s:s.ewm(alpha=.2,adjust=False).mean())
    frame['above_reference']=scores>threshold;frame.to_csv(out/'metrics/health_index.csv',index=False)
    save_json(out/'models/health_reference.json',{'features':[names[i] for i in cols],'center':center.tolist(),
              'scale':scale.tolist(),'threshold':threshold,'meaning':'Deviation from training normal reference; not RUL.'})
    summary=frame[frame.split=='test'].groupby('label').agg(median_index=('health_deviation','median'),
              above_reference_fraction=('above_reference','mean')).reset_index();summary['class']=summary.label.map(dict(enumerate(LABELS)))
    summary.to_csv(out/'metrics/health_summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for label,ax in enumerate(axes.flat):
        group=frame[(frame.split=='test')&(frame.label==label)];source=group.source_id.iloc[0];group=group[group.source_id==source]
        ax.plot(group.time_s,group.health_deviation,alpha=.4,label='window');ax.plot(group.time_s,group.causal_smoothed,label='causal EWMA')
        ax.axhline(threshold,color='r',ls='--',label='training normal q99');ax.set_title(f'{LABELS[label]} / source {source}')
        ax.set_xlabel('Within-record time (s)');ax.set_ylabel('Health deviation');ax.legend(fontsize=8)
    savefig(out/'figures/health_index.png')
