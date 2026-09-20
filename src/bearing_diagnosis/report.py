import json
from pathlib import Path
import pandas as pd
from .config import save_json

def markdown_table(df):
    cols=list(df.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)

def report(cfg,out):
    compare=pd.read_csv(out/'metrics/comparison.csv');robust=pd.read_csv(out/'metrics/robustness_summary.csv')
    health=pd.read_csv(out/'metrics/health_summary.csv');cases=json.loads((out/'reports/cases.json').read_text())
    text='# Measured experiment report\n\n'
    text+='All values below come from saved predictions. This experiment is a held-out-load evaluation, not an independent-device validation.\n\n'
    text+='## Classification\n\n'+markdown_table(compare)+'\n\n'
    for seed in cfg['seeds']:
        group=compare[compare.seed==seed].set_index('model')
        delta=group.loc['rf','f1_macro']-group.loc['rf_without_physics','f1_macro']
        text+=f'Physics feature ablation, seed {seed}: macro F1 difference (with minus without) = {delta:+.4f}. This measures this fixed protocol only.\n\n'
    text+='## Robustness\n\n'+markdown_table(robust)+'\n\n'
    worst=robust.loc[robust.f1_change_mean.idxmin()]
    text+=f'Largest observed mean change: {worst.model}, {worst.perturbation}, level {worst.level}: {worst.f1_change_mean:+.4f} macro F1. Noise can mask impulses and dropped intervals can remove evidence; these are plausible mechanisms, not established causal explanations.\n\n'
    text+='## Cases and interpretation\n\n'
    for case in cases:
        if case['available']:
            text+=f'- {case["kind"]}: {case["sample_id"]}, true {case["true"]}, predicted {case["predicted"]}, confidence {case["confidence"]:.4f}.\n'
        else: text+=f'- {case["kind"]}: {case["reason"]}\n'
    text+='\nCases are chosen deterministically as the highest-confidence correct/incorrect clean CNN prediction. Feature importance favors correlated and high-cardinality features. Input gradients measure local sensitivity, not causation. The marked frequencies are ideal kinematic estimates.\n\n'
    text+='## Health deviation\n\n'+markdown_table(health)+'\n\n'
    text+='The threshold uses training normal windows only. Above-reference fractions for normal data measure observed false alarms on this condition. Fault and normal distributions may overlap. Within-record fluctuations are not natural degradation histories; no RUL or supervised regression scores are available.\n\n'
    text+='## Limits\n\nNormal sampling rate is a declared 48 kHz convention, not an embedded per-file measurement. Labels are manifest-derived. Shared bearing identity across loads cannot be excluded. Window metrics contain dependent observations and do not establish field reliability. Training and validation loss use different weighting, as recorded in training metadata.\n'
    (out/'reports/experiment.md').write_text(text)
    index={str(p.relative_to(out)):p.stat().st_size for p in out.rglob('*') if p.is_file() and 'models' not in p.parts and 'cache' not in p.parts}
    save_json(out/'reports/artifact_index.json',index)
