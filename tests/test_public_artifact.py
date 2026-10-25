import hashlib
import pytest


@pytest.mark.parametrize('resume,range_ignored',[(False,False),(True,False),(True,True)])
def test_public_artifact_streams_and_recovers_partial_download(tmp_path,monkeypatch,resume,range_ignored):
    from hi3d_workbench.pipeline_models import _download_artifact
    data=b'verified model bytes'
    target=tmp_path/'model.bin'
    if resume:target.with_name('model.bin.tmp').write_bytes(data[:8])
    class Response:
        status_code=200 if not resume or range_ignored else 206
        headers={'Content-Range':f'bytes 8-{len(data)-1}/{len(data)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield data[8:] if resume and not range_ignored else data
        @property
        def content(self):raise AssertionError('Large model must not be buffered in memory')
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    _download_artifact({'url':'https://example.com/public','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},target)
    assert target.read_bytes()==data


def test_public_artifact_never_publishes_oversized_response(tmp_path,monkeypatch):
    from hi3d_workbench.pipeline_models import _download_artifact
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield b'too many bytes'
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    with pytest.raises(ValueError,match='超过'):
        _download_artifact({'url':'https://example.com/public','bytes':2,'sha256':'0'*64},tmp_path/'model.bin')
    assert not (tmp_path/'model.bin').exists()
