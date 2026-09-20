from pathlib import Path
import hashlib, json
import yaml
ROOT = Path(__file__).resolve().parents[2]
LABELS = ["normal", "inner", "outer", "ball"]
def load_config(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    if cfg['sampling_rate'] != 12000: raise ValueError('This protocol requires 12000 Hz.')
    if cfg['window_size'] < 128 or not 0 <= cfg['overlap'] < 1: raise ValueError('Invalid window configuration.')
    return cfg
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
