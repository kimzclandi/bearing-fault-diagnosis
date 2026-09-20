from pathlib import Path
import hashlib, time
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from .config import ROOT, digest, save_json
from .signal import validate_signal, preprocess, windows
from .features import extract_batch

def manifest(cfg):
    table=pd.read_csv(ROOT/cfg['manifest'],dtype={'source_id':str},keep_default_na=False)
    expected=table.load_hp.map({0:'train',1:'train',2:'val',3:'test'})
    if not table.split.equals(expected) or table.source_id.duplicated().any():
        raise ValueError('Manifest violates fixed file/condition split.')
    for split in ['train','val','test']:
        if set(table.loc[table.split==split,'label'])!={0,1,2,3}: raise ValueError('Missing class in split.')
    return table

def download_one(r,folder):
    """Validated byte-range transfer; only complete MATLAB files become visible."""
    path=folder/(r['source_id']+'.mat')
    if not path.exists():
        temp=path.with_suffix('.part')
        with requests.Session() as session:
            for attempt in range(5):
                try:
                    head=session.head(r['url'],timeout=(20,60),allow_redirects=True)
                    head.raise_for_status();total=int(head.headers['Content-Length']);break
                except (requests.RequestException,KeyError,ValueError):
                    if attempt==4: raise
                    time.sleep(2**attempt)
            if temp.exists() and temp.stat().st_size>total: temp.unlink()
            offset=temp.stat().st_size if temp.exists() else 0
            while offset<total:
                end=min(offset+1024*1024-1,total-1)
                for attempt in range(5):
                    try:
                        response=session.get(r['url'],headers={'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'},timeout=(20,90))
                        response.raise_for_status()
                        if response.status_code==206:
                            expected=f'bytes {offset}-{end}/{total}'
                            if response.headers.get('Content-Range')!=expected: raise ValueError('Incorrect byte range response.')
                            content=response.content
                            if len(content)!=end-offset+1: raise ValueError('Truncated byte range.')
                            with temp.open('ab') as f: f.write(content)
                            offset+=len(content)
                        elif response.status_code==200 and len(response.content)==total:
                            temp.write_bytes(response.content);offset=total
                        else: raise ValueError('Server did not deliver a complete file or requested range.')
                        break
                    except (requests.RequestException,ValueError):
                        if attempt==4: raise
                        time.sleep(2**attempt)
            if temp.stat().st_size!=total: raise ValueError('File size mismatch.')
            loadmat(temp);temp.replace(path)
    loadmat(path)
    result={'source_id':r['source_id'],'sha256':digest(path),'bytes':path.stat().st_size,'url':r['url']}
    print(f"Downloaded/verified {r['name']}: {path.stat().st_size} bytes",flush=True)
    return result

def download(cfg):
    from concurrent.futures import ThreadPoolExecutor
    folder=ROOT/cfg['data_dir'];folder.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipt=list(pool.map(lambda row:download_one(row,folder),manifest(cfg).to_dict('records')))
    save_json(folder/'download_receipt.json',receipt)
    lock=ROOT/'data/manifests/checksums.json'
    if lock.exists():
        import json
        expected={r['source_id']:r['sha256'] for r in json.loads(lock.read_text())}
        for row in receipt:
            if expected.get(row['source_id'])!=row['sha256']: raise ValueError('Data differs from the recorded checksum lock.')


def check_duplicates(records):
    for key in ['sha256','signal_sha256']:
        seen={}
        for r in records:
            value=r[key]
            if value in seen: raise ValueError(f'Duplicate {key}: {seen[value]} and {r["source_id"]}')
            seen[value]=r['source_id']

def select_channel(mat, source_id):
    expected=f'X{int(source_id):03d}_DE_time'
    candidates=[k for k in mat if k.endswith('_DE_time')]
    if expected in candidates:
        key=expected
    elif len(candidates)==1:
        key=candidates[0]
    else:
        raise ValueError(f'Ambiguous drive-end channels for source {source_id}: {candidates}')
    prefix=key.removesuffix('_DE_time')
    rpm_keys=[k for k in mat if k.upper().endswith('RPM') and k.startswith(prefix)]
    return key,rpm_keys,[k for k in candidates if k!=key]

def prepare(cfg,out):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    signals=[]; metadata=[]; sources=[]
    for r in manifest(cfg).to_dict('records'):
        path=ROOT/cfg['data_dir']/(r['source_id']+'.mat')
        if not path.is_file(): raise FileNotFoundError(f'{path}; run download_data.py first.')
        mat=loadmat(path);key,rpm_keys,ignored=select_channel(mat,r['source_id'])
        raw=validate_signal(mat[key].squeeze())
        rpm=float(np.asarray(mat[rpm_keys[0]]).squeeze()) if rpm_keys else float(r['rpm_nominal'])
        if not 100<rpm<10000: raise ValueError('Implausible speed.')
        clean=preprocess(raw,int(r['sampling_rate']),cfg)
        x,starts,means=windows(clean,cfg['window_size'],cfg['overlap'],cfg['max_windows_per_file'])
        source={**r,'sha256':digest(path),'signal_sha256':hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest(),
                'raw_samples':len(raw),'processed_samples':len(clean),'windows':len(x),'rpm':rpm,
                'rpm_source':'mat' if rpm_keys else 'official_nominal','channel':key,'ignored_de_channels':ignored,
                'channel_selection':'exact_id' if key==f'X{int(r["source_id"]):03d}_DE_time' else 'sole_channel_fallback',
                'crest_warning':bool(np.max(np.abs(raw-raw.mean()))/(raw.std()+1e-12)>30),
                'uncovered_tail_samples':int((len(clean)-cfg['window_size'])%max(1,round(cfg['window_size']*(1-cfg['overlap']))))}
        sources.append(source)
        for start,mean in zip(starts,means):
            metadata.append({'sample_id':f'{r["source_id"]}:{start}','source_id':r['source_id'],'split':r['split'],
                'label':r['label'],'load_hp':r['load_hp'],'rpm':rpm,'sampling_rate':cfg['sampling_rate'],
                'start':int(start),'end':int(start+cfg['window_size']),'raw_mean':float(mean)})
        signals.extend(x)
    check_duplicates(sources)
    meta=pd.DataFrame(metadata); X=np.stack(signals); F,names=extract_batch(X,meta.rpm.values,cfg)
    if not np.isfinite(F).all(): raise ValueError('Invalid extracted features.')
    np.savez_compressed(cache/'dataset.npz',X=X,F=F,y=meta.label.values)
    meta.to_csv(out/'metrics/windows.csv',index=False)
    pd.DataFrame(F,columns=names).assign(sample_id=meta.sample_id.values).to_csv(out/'metrics/features.csv',index=False)
    save_json(out/'metrics/sources.json',sources);save_json(cache/'feature_names.json',names)
    print(f'Prepared {len(X)} windows; split counts: {meta.split.value_counts().to_dict()}',flush=True)

def load_dataset(out):
    a=np.load(out/'cache/dataset.npz'); meta=pd.read_csv(out/'metrics/windows.csv',dtype={'source_id':str})
    return a['X'],a['F'],a['y'],meta
