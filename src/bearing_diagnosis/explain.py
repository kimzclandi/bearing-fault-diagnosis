import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .data import load_dataset
from .training import load_cnn,normalize
from .signal import envelope_spectrum
from .physics import bearing_frequencies
from .config import LABELS,save_json
from .evaluation import savefig

def saliency(model,x,stats):
    tensor=torch.from_numpy(normalize(x[None],stats))[:,None,:].requires_grad_(True)
    model.eval();logits=model(tensor);target=int(logits.argmax(1).item())
    gradient=torch.autograd.grad(logits[0,target],tensor)[0]
    return np.abs(gradient.detach().numpy()[0,0])/stats['std']

def explain(cfg,out):
    X,F,y,m=load_dataset(out);seed=cfg['seeds'][0]
    names=json.loads((out/'cache/feature_names.json').read_text())
    bundle=joblib.load(out/f'models/rf_{seed}.joblib')
    importance=pd.DataFrame({'feature':names,'importance':bundle['model'][-1].feature_importances_}).sort_values('importance')
    importance.to_csv(out/'metrics/feature_importance.csv',index=False)
    importance.tail(15).plot.barh(x='feature',y='importance',legend=False,figsize=(8,5));plt.title('Random Forest impurity importance')
    savefig(out/'figures/feature_importance.png')
    pred=pd.read_csv(out/f'predictions/cnn_{seed}.csv');model,stats=load_cnn(out,seed);cases=[]
    for kind,subset in [('success',pred[pred.label==pred.predicted]),('failure',pred[pred.label!=pred.predicted])]:
        if subset.empty:
            cases.append({'kind':kind,'available':False,'reason':'No matching case in the clean test predictions.'});continue
        row=subset.sort_values(['confidence','sample_id'],ascending=[False,True]).iloc[0]
        idx=int(np.flatnonzero(m.sample_id.eq(row.sample_id).values)[0]);x=X[idx];sens=saliency(model,x,stats)
        ef,ep=envelope_spectrum(x,cfg['sampling_rate'],cfg['envelope_band']);freq=bearing_frequencies(row.rpm,**cfg['geometry'])
        fig,axes=plt.subplots(3,1,figsize=(10,8));t=np.arange(len(x))/cfg['sampling_rate']
        axes[0].plot(t,x,lw=.6);axes[0].set_ylabel('Acceleration');axes[0].set_title(f'{kind}: {row.sample_id}; true={LABELS[int(row.label)]}, predicted={LABELS[int(row.predicted)]}')
        axes[1].plot(t,sens,lw=.6);axes[1].set_ylabel('Absolute input gradient');axes[1].set_xlabel('Time (s)')
        axes[2].plot(ef,ep,lw=.8)
        for color,(name,hz) in zip(['tab:orange','tab:green','tab:red','tab:purple'],freq.items()):
            for harmonic in [1,2,3]: axes[2].axvline(hz*harmonic,alpha=.65,ls='--',color=color,label=name if harmonic==1 else None)
        axes[2].set_xlim(0,600);axes[2].set_xlabel('Envelope frequency (Hz)');axes[2].set_ylabel('PSD');axes[2].legend(ncol=4)
        savefig(out/f'figures/case_{kind}.png')
        pd.DataFrame({'time_s':t,'signal':x,'absolute_gradient':sens}).to_csv(out/f'metrics/saliency_{kind}.csv',index=False)
        cases.append({'kind':kind,'available':True,'sample_id':row.sample_id,'true':LABELS[int(row.label)],
                      'predicted':LABELS[int(row.predicted)],'confidence':float(row.confidence),
                      'physics_features':{n:float(F[idx,j]) for j,n in enumerate(names) if n.startswith('physics_')},
                      'interpretation':'Gradient measures local logit sensitivity, not causal fault evidence. Frequency peaks require speed and resonance assumptions.'})
    save_json(out/'reports/cases.json',cases)
