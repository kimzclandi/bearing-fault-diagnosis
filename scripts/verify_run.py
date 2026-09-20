"""Independently validate saved predictions, split membership and preprocessing state."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from bearing_diagnosis.config import ROOT,save_json,digest
from bearing_diagnosis.data import check_duplicates

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
cfg=json.loads((out/'config.json').read_text())
sources=json.loads((out/'metrics/sources.json').read_text());check_duplicates(sources)
for row in sources:
    assert row['sha256']==digest(ROOT/cfg['data_dir']/(str(row['source_id'])+'.mat'))
meta=pd.read_csv(out/'metrics/windows.csv',dtype={'source_id':str})
assert meta.sample_id.is_unique
assert meta.groupby('source_id').split.nunique().max()==1
assert ((meta.end-meta.start)==cfg['window_size']).all()
assert set(meta[meta.split=='train'].load_hp)=={0,1}
assert set(meta[meta.split=='val'].load_hp)=={2}
assert set(meta[meta.split=='test'].load_hp)=={3}
expected=meta[meta.split=='test'].reset_index(drop=True)
count=0

def check_predictions(path,row):
    global count
    df=pd.read_csv(path,dtype={'source_id':str})
    assert df.sample_id.tolist()==expected.sample_id.tolist()
    assert np.array_equal(df.label,expected.label)
    assert np.isfinite(df.confidence).all() and df.confidence.between(0,1).all()
    for name,fn in [('accuracy',accuracy_score),('f1_macro',f1_score),('precision_macro',precision_score),('recall_macro',recall_score)]:
        value=fn(df.label,df.predicted) if name=='accuracy' else fn(df.label,df.predicted,labels=range(4),average='macro',zero_division=0)
        assert np.isclose(value,row[name],atol=1e-12),str(path)+': '+name
    probability_columns=['p_'+s for s in ['normal','inner','outer','ball']]
    if set(probability_columns)<=set(df.columns):
        probs=df[probability_columns].values
        assert np.allclose(probs.sum(1),1,atol=1e-6)
        assert np.array_equal(probs.argmax(1),df.predicted)
    count+=1

for row in pd.read_csv(out/'metrics/comparison.csv').to_dict('records'):
    check_predictions(out/f'predictions/{row["model"]}_{row["seed"]}.csv',row)
for row in pd.read_csv(out/'metrics/robustness.csv').to_dict('records'):
    level=float(row['level'])
    suffix=str(int(level)) if row['perturbation']=='noise' else str(level)
    check_predictions(out/f'predictions/robust_{row["model"]}_{row["seed"]}_{row["perturbation"]}_{suffix}_{row["perturb_seed"]}.csv',row)
with np.load(out/'cache/dataset.npz') as data:
    train=data['X'][meta.split.eq('train').values]
    stats=json.loads((out/'models/normalizer.json').read_text())
    assert np.isclose(stats['mean'],train.mean()) and np.isclose(stats['std'],train.std())
    assert len(data['X'])==len(meta) and len(data['F'])==len(meta)
    assert np.isfinite(data['X']).all() and np.isfinite(data['F']).all()
    assert np.array_equal(data['y'],meta.label)
health=pd.read_csv(out/'metrics/health_index.csv')
assert health.sample_id.tolist()==meta.sample_id.tolist()
reference=json.loads((out/'models/health_reference.json').read_text())
normal=health[(health.split=='train')&(health.label==0)]
assert np.isclose(reference['threshold'],np.quantile(normal.health_deviation,.99))
result={'status':'passed','source_files':len(sources),'windows':len(meta),'prediction_tables_checked':count,
        'checks':['raw checksums','file-disjoint conditions','sample identity','recomputed metrics','probability sums','training-only normalization','health reference threshold']}
save_json(out/'reports/verification.json',result)
print(json.dumps(result,indent=2))
