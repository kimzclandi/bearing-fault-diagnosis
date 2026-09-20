import json, random, time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from .models import BearingCNN
from .data import load_dataset
from .config import save_json

def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

def fit_normalizer(x):
    return {'mean':float(x.mean()),'std':max(float(x.std()),1e-8)}

def normalize(x,stats): return ((x-stats['mean'])/stats['std']).astype(np.float32)

def train_baseline(cfg,out):
    _,F,y,m=load_dataset(out);train=m.split.eq('train').values;val=m.split.eq('val').values
    names=json.loads((out/'cache/feature_names.json').read_text()); summary=[]
    for seed in cfg['seeds']:
        for variant in ['rf','rf_without_physics']:
            cols=np.array([i for i,n in enumerate(names) if variant=='rf' or not n.startswith('physics_')])
            model=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=cfg['rf_trees'],random_state=seed,
                class_weight='balanced',n_jobs=cfg['threads'],min_samples_leaf=2))
            t=time.perf_counter();model.fit(F[train][:,cols],y[train]); elapsed=time.perf_counter()-t
            # Ordered tree accumulation prevents near-tie class flips across worker schedules.
            model[-1].n_jobs=1
            pred=model.predict(F[val][:,cols]);joblib.dump({'model':model,'columns':cols},out/f'models/{variant}_{seed}.joblib')
            summary.append({'model':variant,'seed':seed,'validation_macro_f1':f1_score(y[val],pred,average='macro'),
                            'validation_accuracy':accuracy_score(y[val],pred),'train_seconds':elapsed})
    pd.DataFrame(summary).to_csv(out/'metrics/baseline_validation.csv',index=False)

def predict_cnn(model,x,stats,batch_size=64,device='cpu'):
    model.eval(); probs=[]
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            a=torch.from_numpy(normalize(x[start:start+batch_size],stats))[:,None,:].to(device)
            probs.append(torch.softmax(model(a),1).cpu().numpy())
    return np.concatenate(probs)

def train_deep(cfg,out):
    X,_,y,m=load_dataset(out);train=m.split.eq('train').values;val=m.split.eq('val').values
    stats=fit_normalizer(X[train]);save_json(out/'models/normalizer.json',stats)
    device=cfg['device']
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable; use cpu.')
    for seed in cfg['seeds']:
        seed_all(seed,cfg['threads']);model=BearingCNN().to(device)
        tx=torch.from_numpy(normalize(X[train],stats))[:,None,:];ty=torch.from_numpy(y[train]).long()
        loader=DataLoader(TensorDataset(tx,ty),batch_size=cfg['batch_size'],shuffle=True,
                          generator=torch.Generator().manual_seed(seed),num_workers=0)
        counts=np.bincount(y[train],minlength=4);weights=len(ty)/(4*counts)
        lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        best=(-1.,float('-inf'));bad=0;history=[];begin=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            model.train();total=0.;loss_weight=0.;correct=0;seen=0
            for xbatch,ybatch in loader:
                xbatch,ybatch=xbatch.to(device),ybatch.to(device)
                optimizer.zero_grad();logits=model(xbatch);loss=lossfn(logits,ybatch);loss.backward();optimizer.step()
                batch_weight=lossfn.weight[ybatch].sum().item();total+=loss.item()*batch_weight;loss_weight+=batch_weight;correct+=(logits.argmax(1)==ybatch).sum().item();seen+=len(ybatch)
            p=predict_cnn(model,X[val],stats,cfg['batch_size'],device)
            vl=float(-np.log(np.clip(p[np.arange(val.sum()),y[val]],1e-9,1)).mean())
            vf=float(f1_score(y[val],p.argmax(1),average='macro',zero_division=0))
            history.append({'epoch':epoch,'train_loss':total/loss_weight,'val_loss':vl,'train_accuracy':correct/seen,
                            'val_accuracy':accuracy_score(y[val],p.argmax(1)),'val_macro_f1':vf})
            key=(vf,-vl)
            if key>best:
                best=key;bad=0
                torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'seed':seed,
                            'best_epoch':epoch,'normalizer':stats,'config':cfg},out/f'models/cnn_{seed}.pt')
            else: bad+=1
            print(f'CNN seed={seed} epoch={epoch}: val_F1={vf:.4f} val_loss={vl:.4f}',flush=True)
            if bad>=cfg['patience']: break
        pd.DataFrame(history).to_csv(out/f'metrics/history_{seed}.csv',index=False)
        save_json(out/f'metrics/training_{seed}.json',{'seconds':time.perf_counter()-begin,'epochs':len(history),
                  'best_validation_macro_f1':best[0],'train_loss_weighting':'inverse class frequency; val_loss unweighted'})

def load_cnn(out,seed):
    c=torch.load(out/f'models/cnn_{seed}.pt',map_location='cpu',weights_only=True)
    model=BearingCNN();model.load_state_dict(c['state_dict']);model.eval()
    return model,c['normalizer']
