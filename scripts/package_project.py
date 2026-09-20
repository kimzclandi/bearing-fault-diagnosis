"""Create a source-and-results archive without raw data, weights, or environments."""
import argparse,hashlib,zipfile
from pathlib import Path
from bearing_diagnosis.config import ROOT
p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT.parent/'bearing-fault-diagnosis.zip'));args=p.parse_args()
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
include_dirs={'src','scripts','app','tests','configs','docs','notebooks','.streamlit','.github'}
include_files={'README.md','README.en.md','requirements.txt','requirements-lock.txt','pyproject.toml','Dockerfile','.dockerignore','.gitignore'}
paths=[]
for f in ROOT.rglob('*'):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(ROOT);parts=rel.parts
    if '__pycache__' in parts or '.pytest_cache' in parts or any(x.endswith('.egg-info') for x in parts) or f.suffix in ['.pyc','.part']:continue
    include=(len(parts)==1 and parts[0] in include_files) or parts[0] in include_dirs
    if parts[0]=='data':include=parts[1]=='manifests' or rel.as_posix()=='data/README.md'
    if parts[0]=='outputs':include=len(parts)>2 and parts[1] in ['quick-verified','full-verified','qa'] and not any(x in parts for x in ['models','cache','logs'])
    if include:paths.append(f)
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for f in sorted(paths):archive.write(f,Path('bearing-fault-diagnosis')/f.relative_to(ROOT))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert not any('/raw/' in name or '/models/' in name or '/cache/' in name or '/.venv/' in name for name in archive.namelist())
sha=hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix('.zip.sha256').write_text(f'{sha}  {output.name}\n')
print(f'Archive: {output}; files={len(paths)}; bytes={output.stat().st_size}; SHA256={sha}')
