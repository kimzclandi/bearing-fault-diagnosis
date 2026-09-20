import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from .config import LABELS,save_json
from .data import load_dataset
from .features import extract_batch
from .training import load_cnn,predict_cnn

def metrics(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,labels=range(4),average='macro',zero_division=0)
    return dict(accuracy=float(accuracy_score(y,p)),precision_macro=float(precision),recall_macro=float(recall),f1_macro=float(f1))

def savefig(path):
    plt.tight_layout();plt.savefig(path,dpi=150,bbox_inches='tight');plt.close()

def predict_all(cfg,out,X,F,seed):
    result={}
    for name in ['rf','rf_without_physics']:
        bundle=joblib.load(out/f'models/{name}_{seed}.joblib')
        result[name]=bundle['model'].predict_proba(F[:,bundle['columns']])
    model,stats=load_cnn(out,seed)
    result['cnn']=predict_cnn(model,X,stats)
    return result

def evaluate(cfg,out):
    X,F,y,m=load_dataset(out);test=m.split.eq('test').values
    rows=[];per_file=[]
    for seed in cfg['seeds']:
        results=predict_all(cfg,out,X[test],F[test],seed)
        for name,probs in results.items():
            p=probs.argmax(1); rows.append({'model':name,'seed':seed,**metrics(y[test],p)})
            report=classification_report(y[test],p,labels=range(4),target_names=LABELS,output_dict=True,zero_division=0)
            save_json(out/f'metrics/classification_{name}_{seed}.json',report)
            frame=m[test].copy();frame['predicted']=p;frame['confidence']=probs.max(1)
            for k,label in enumerate(LABELS): frame['p_'+label]=probs[:,k]
            frame.to_csv(out/f'predictions/{name}_{seed}.csv',index=False)
            for source,group in frame.groupby('source_id'):
                per_file.append({'model':name,'seed':seed,'source_id':source,'n':len(group),
                                 'accuracy':float((group.label==group.predicted).mean())})
            if seed==cfg['seeds'][0]:
                fig,axes=plt.subplots(1,2,figsize=(10,4))
                for ax,norm in zip(axes,[None,'true']):
                    cm=confusion_matrix(y[test],p,labels=range(4),normalize=norm)
                    ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False,values_format='.2f' if norm else 'd')
                    ax.set_title(name+(' / fraction' if norm else ' / count'))
                savefig(out/f'figures/confusion_{name}.png')
    table=pd.DataFrame(rows);table.to_csv(out/'metrics/comparison.csv',index=False)
    pd.DataFrame(per_file).to_csv(out/'metrics/per_file.csv',index=False)
    table.groupby('model')[['accuracy','f1_macro']].agg(['mean','std']).to_csv(out/'metrics/comparison_summary.csv')
    for seed in cfg['seeds']:
        h=pd.read_csv(out/f'metrics/history_{seed}.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for col in ['train_loss','val_loss']: axes[0].plot(h.epoch,h[col],label=col)
        for col in ['train_accuracy','val_accuracy']: axes[1].plot(h.epoch,h[col],label=col)
        for ax in axes: ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
        axes[0].set_title('Weighted training / unweighted validation loss');axes[1].set_title('Accuracy')
        savefig(out/f'figures/training_{seed}.png')
    print(table.to_string(index=False),flush=True)

def perturb(x,kind,level,rng):
    x=x.copy()
    if kind=='noise':
        std=np.sqrt(np.mean(x*x,axis=1,keepdims=True))*10**(-level/20)
        x+=rng.normal(size=x.shape)*std
    elif kind=='missing':
        n=max(1,round(x.shape[1]*level))
        for row in x:
            start=int(rng.integers(0,len(row)-n+1));row[start:start+n]=0
    else: raise ValueError('Unknown perturbation.')
    return (x-x.mean(axis=1,keepdims=True)).astype(np.float32)

def robustness(cfg,out):
    X,F,y,m=load_dataset(out);test=m.split.eq('test').values;X=X[test];F=F[test];y=y[test];rpm=m.loc[test,'rpm'].values
    clean=pd.read_csv(out/'metrics/comparison.csv');rows=[]
    for kind,levels in [('noise',cfg['noise_snr_db']),('missing',cfg['missing_fraction'])]:
        for level in levels:
            for perturb_seed in cfg['robustness_seeds']:
                a=perturb(X,kind,level,np.random.default_rng(perturb_seed));features,_=extract_batch(a,rpm,cfg)
                for seed in cfg['seeds']:
                    for name,probs in predict_all(cfg,out,a,features,seed).items():
                        score=metrics(y,probs.argmax(1));baseline=clean[(clean.model==name)&(clean.seed==seed)].iloc[0]
                        rows.append({'model':name,'seed':seed,'perturbation':kind,'level':level,'perturb_seed':perturb_seed,
                            **score,'f1_change':score['f1_macro']-baseline.f1_macro})
                        pred=m[test].copy();pred['predicted']=probs.argmax(1);pred['confidence']=probs.max(1)
                        pred.to_csv(out/f'predictions/robust_{name}_{seed}_{kind}_{level}_{perturb_seed}.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics/robustness.csv',index=False)
    summary=df.groupby(['model','perturbation','level']).agg(f1_mean=('f1_macro','mean'),f1_std=('f1_macro','std'),
                f1_change_mean=('f1_change','mean'),accuracy_mean=('accuracy','mean')).reset_index()
    summary.to_csv(out/'metrics/robustness_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axes,['noise','missing']):
        for name,group in summary[summary.perturbation==kind].groupby('model'):
            ax.errorbar(group.level,group.f1_mean,yerr=group.f1_std,label=name,marker='o',capsize=3)
        ax.set_title(kind);ax.set_ylabel('Macro F1');ax.set_xlabel('SNR (dB)' if kind=='noise' else 'Missing fraction');ax.legend();ax.grid(alpha=.2)
    savefig(out/'figures/robustness.png')
