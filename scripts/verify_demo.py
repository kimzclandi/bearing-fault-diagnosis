"""Exercise the trained interface and CSV/NPY validation without a browser."""
from pathlib import Path
from unittest.mock import patch
import io,json
import numpy as np
from streamlit.testing.v1 import AppTest
from bearing_diagnosis.config import ROOT,save_json

app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=40).run()
assert not app.exception and not app.error and len(app.dataframe)>=2
checks={'local_example':'passed'}
app.radio[0].set_value('Upload').run();assert not app.exception and not app.error
checks['upload_empty_state']='passed'
class Upload(io.BytesIO):
    def __init__(self,content,name):super().__init__(content);self.name=name
x=np.sin(2*np.pi*100*np.arange(4096)/12000)
buf=io.BytesIO();np.save(buf,x)
inputs=[('numeric_npy',Upload(buf.getvalue(),'signal.npy'),False),
        ('numeric_csv',Upload(('signal\n'+'\n'.join(map(str,x))).encode(),'signal.csv'),False),
        ('nonfinite_csv',Upload(('signal\n'+'\n'.join(['nan']*200)).encode(),'invalid.csv'),True)]
for name,upload,invalid in inputs:
    with patch('streamlit.file_uploader',return_value=upload): app.run()
    assert not app.exception
    assert bool(app.error)==invalid
    if not invalid: assert len(app.dataframe)>=2
    checks[name]='passed'
save_json(ROOT/'outputs/qa/demo_automated.json',checks);print(json.dumps(checks,indent=2))
