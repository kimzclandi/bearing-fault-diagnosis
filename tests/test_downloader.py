from scipy.io import savemat
import io
import numpy as np
from bearing_diagnosis.data import download_one

def test_range_resume_and_atomic_completion(tmp_path,monkeypatch):
    from bearing_diagnosis import data
    buffer=io.BytesIO();savemat(buffer,{'X_DE_time':np.arange(300000,dtype=float)})
    content=buffer.getvalue();calls=[]
    class Response:
        status_code=206
        def __init__(self,body=b'',headers=None):self.content=body;self.headers=headers or {}
        def raise_for_status(self):pass
    class Session:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def head(self,*args,**kwargs):return Response(headers={'Content-Length':str(len(content))})
        def get(self,url,headers,**kwargs):
            start,end=map(int,headers['Range'][6:].split('-'));calls.append(start)
            return Response(content[start:end+1],{'Content-Range':f'bytes {start}-{end}/{len(content)}'})
    monkeypatch.setattr(data.requests,'Session',Session)
    (tmp_path/'sample.part').write_bytes(content[:12345])
    result=download_one({'source_id':'sample','name':'sample','url':'https://example.invalid/sample.mat'},tmp_path)
    assert calls[0]==12345 and len(calls)>=2
    assert (tmp_path/'sample.mat').read_bytes()==content
    assert not (tmp_path/'sample.part').exists()
    assert result['bytes']==len(content)
