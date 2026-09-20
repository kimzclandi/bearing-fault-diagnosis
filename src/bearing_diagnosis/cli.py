import argparse,datetime,importlib.metadata,json,platform,sys,time,traceback,subprocess
from pathlib import Path
from .config import ROOT,load_config,save_json,digest

def execute(stage,cfg,out):
    from . import data,training,evaluation,explain,health,report
    stages={'download':lambda:data.download(cfg),'prepare':lambda:data.prepare(cfg,out),
       'baseline':lambda:training.train_baseline(cfg,out),'deep':lambda:training.train_deep(cfg,out),
       'evaluate':lambda:evaluation.evaluate(cfg,out),'explain':lambda:explain.explain(cfg,out),
       'health':lambda:health.health_index(cfg,out),'robustness':lambda:evaluation.robustness(cfg,out),
       'report':lambda:report.report(cfg,out)}
    start=time.perf_counter()
    try:
        stages[stage]();status={'stage':stage,'status':'passed','seconds':time.perf_counter()-start}
    except Exception as exc:
        status={'stage':stage,'status':'failed','error':str(exc),'seconds':time.perf_counter()-start}
        save_json(out/f'logs/{stage}.json',status);raise
    save_json(out/f'logs/{stage}.json',status);print(json.dumps(status),flush=True)

def main(stage='all'):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/quick.yaml');parser.add_argument('--run-dir')
    parser.add_argument('--download',action='store_true');args=parser.parse_args()
    cfg=load_config(args.config)
    out=Path(args.run_dir) if args.run_dir else Path('outputs')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    if not out.is_absolute(): out=ROOT/out
    if stage=='all' and out.exists(): raise FileExistsError('Use a new run directory to preserve previous evidence.')
    for folder in ['figures','metrics','models','logs','reports','predictions','cache']: (out/folder).mkdir(parents=True,exist_ok=True)
    cfgpath=out/'config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=cfg: raise ValueError('Run configuration differs from saved configuration.')
    save_json(cfgpath,cfg)
    if not (out/'provenance.json').exists():
        save_json(out/'provenance.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python':sys.version,'platform':platform.platform(),'command':sys.argv,'data_kind':'real_cwru',
          'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','torch','matplotlib','streamlit']},
          'manifest_sha256':digest(ROOT/cfg['manifest']),
          'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src').rglob('*.py')}})
    stages=(['download'] if args.download else [])+['prepare','baseline','deep','evaluate','explain','health','robustness','report'] if stage=='all' else [stage]
    if stage=='all':
        result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
        (out/'logs/tests.txt').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError('Preflight tests failed; see logs/tests.txt.')
    for item in stages: execute(item,cfg,out)
    print(f'Artifacts: {out}',flush=True)
