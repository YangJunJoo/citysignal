import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from data_downloader.data_downloader import download_csv, _response_plan, DownloadIntegrityError


class Response:
    def __init__(self, status=200, body=b'abcdef', headers=None, interrupt=False):
        self.status_code = status
        self.body = body
        self.headers = CaseInsensitiveDict(headers or {})
        self.interrupt = interrupt
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))
    def iter_content(self, chunk_size):
        yield self.body
        if self.interrupt:
            raise requests.ConnectionError('interrupted')
    def close(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers_sent = []
    def get(self, url, **kwargs):
        self.headers_sent.append(kwargs.get('headers', {}))
        return next(self.responses)
    def mount(self, *args):
        pass
    def close(self):
        pass


def run_download(path, responses, retries=2):
    session = Session(responses)
    with patch('data_downloader.data_downloader.requests.Session', return_value=session), patch('data_downloader.data_downloader.time.sleep'):
        ok = download_csv('https://example.test/data.csv', path, max_retries=retries)
    return ok, session


def partial(path, validator='"v1"'):
    Path(str(path)+'.tmp').write_bytes(b'abc')
    Path(str(path)+'.tmp.json').write_text(json.dumps({'url':'https://example.test/data.csv', 'etag':validator,'total_size':6}))


def test_full_download_and_evidence(tmp_path):
    path = tmp_path/'data.csv'
    ok, _ = run_download(path,[Response(headers={'Content-Length':'6','ETag':'"v1"'})])
    assert ok and path.read_bytes() == b'abcdef'
    evidence = json.loads(Path(str(path)+'.download.json').read_text())
    assert evidence['byte_count'] == 6 and evidence['http_status'] == 200


def test_interrupted_download_validated_resume(tmp_path):
    path = tmp_path/'data.csv'
    ok, session = run_download(path,[Response(body=b'abc',headers={'Content-Length':'6','ETag':'"v1"'},interrupt=True),
                     Response(206,b'def',{'Content-Length':'3','Content-Range':'bytes 3-5/6','ETag':'"v1"'})])
    assert ok and path.read_bytes() == b'abcdef'
    assert session.headers_sent[1]['Range'] == 'bytes=3-'
    assert session.headers_sent[1]['If-Range'] == '"v1"'


@pytest.mark.parametrize('headers', [
    {'Content-Length':'3','Content-Range':'bytes 2-4/6','ETag':'"v1"'},
    {'Content-Length':'3','ETag':'"v1"'},
    {'Content-Length':'4','Content-Range':'bytes 3-5/6','ETag':'"v1"'},
    {'Content-Length':'3','Content-Range':'bytes 3-5/6','ETag':'"v2"'},
    {'Content-Length':'3','Content-Range':'bytes 3-5/6'},
    {'Content-Length':'4','Content-Range':'bytes 3-6/7','ETag':'"v1"'},
])
def test_invalid_resume_preserves_existing_final(tmp_path,headers):
    path=tmp_path/'data.csv'; path.write_bytes(b'original');partial(path)
    ok,_=run_download(path,[Response(206,b'def',headers)],retries=1)
    assert not ok and path.read_bytes() == b'original'
    assert not Path(str(path)+'.tmp').exists()


def test_200_response_to_range_replaces_partial(tmp_path):
    path=tmp_path/'data.csv';partial(path)
    ok,_=run_download(path,[Response(200,b'NEWALL',{'Content-Length':'6','ETag':'"v2"'})])
    assert ok and path.read_bytes() == b'NEWALL'


def test_416_restarts_with_full_request(tmp_path):
    path=tmp_path/'data.csv';partial(path)
    ok,session=run_download(path,[Response(416),Response(200,b'abcdef',{'Content-Length':'6'})])
    assert ok and path.read_bytes() == b'abcdef'
    assert 'Range' not in session.headers_sent[1]


def test_unversioned_partial_is_not_resumed(tmp_path):
    path=tmp_path/'data.csv';Path(str(path)+'.tmp').write_bytes(b'old')
    ok,session=run_download(path,[Response(200,b'abcdef',{'Content-Length':'6'})])
    assert ok and 'Range' not in session.headers_sent[0]
    assert path.read_bytes() == b'abcdef'


def test_404_persists_http_status(tmp_path):
    path=tmp_path/'data.csv';path.write_bytes(b'original')
    ok,_=run_download(path,[Response(404)])
    assert not ok and path.read_bytes() == b'original'
    assert json.loads(Path(str(path)+'.download.json').read_text())['http_status'] == 404


def test_incomplete_200_never_publishes(tmp_path):
    path=tmp_path/'data.csv';path.write_bytes(b'original')
    ok,_=run_download(path,[Response(200,b'abc',{'Content-Length':'6'})],retries=1)
    assert not ok and path.read_bytes() == b'original'
