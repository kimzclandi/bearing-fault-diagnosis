"""Recompute every inference condition and compare discrete outputs exactly."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from bearing_diagnosis.config import ROOT,save_json
from bearing_diagnosis.data import load_dataset
from bearing_diagnosis.features import extract_batch
from bearing_diagnosis.evaluation import predict_all,perturb
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
cfg=json.loads((out/'config.json').read_text())
import torch
torch.set_num_threads(cfg['threads'])
X,F,_,m=load_dataset(out);test=m.split.eq('test').values;X=X[test];F=F[test];rpm=m.loc[test,'rpm'].values
count=0;max_difference=0.
def compare(filename,probabilities):
    global count,max_difference
    saved=pd.read_csv(out/'predictions'/filename)
    assert np.array_equal(saved.predicted,probabilities.argmax(1)),filename
    assert saved.sample_id.tolist()==m.loc[test,'sample_id'].tolist()
    difference=np.max(np.abs(saved.confidence.values-probabilities.max(1)));max_difference=max(max_difference,float(difference))
    assert np.allclose(saved.confidence.values,probabilities.max(1),atol=1e-7,rtol=1e-6)
    count+=1
for seed in cfg['seeds']:
    for name,probs in predict_all(cfg,out,X,F,seed).items():compare(f'{name}_{seed}.csv',probs)
for kind,levels in [('noise',cfg['noise_snr_db']),('missing',cfg['missing_fraction'])]:
    for level in levels:
        for perturb_seed in cfg['robustness_seeds']:
            a=perturb(X,kind,level,np.random.default_rng(perturb_seed));features,_=extract_batch(a,rpm,cfg)
            for seed in cfg['seeds']:
                for name,probs in predict_all(cfg,out,a,features,seed).items():
                    compare(f'robust_{name}_{seed}_{kind}_{level}_{perturb_seed}.csv',probs)
result={'status':'passed','prediction_tables_recomputed':count,'discrete_predictions':'exact match','max_confidence_difference':max_difference,
        'probability_atol':1e-7,'probability_rtol':1e-6,'scope':'same-machine re-inference of all clean and perturbed test windows'}
save_json(out/'reports/reproducibility.json',result);print(json.dumps(result,indent=2))
