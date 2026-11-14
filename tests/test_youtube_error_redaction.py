"""YouTube Data API failures must not print the API key.

googleapiclient's HttpError text includes the request URL, and that URL carries
the key as a query parameter, so the error lines in research/lib/youtube.py must
print a summary rather than the raw exception.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SECRET = "AIzaSyFAKE-test-key-0123456789"
URL = (f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet&id=vid"
       f"&key={SECRET}&alt=json")


class _FakeHttpError(Exception):
    """Mimics googleapiclient.errors.HttpError: a resp plus a URL in str()."""

    def __init__(self):
        self.resp = types.SimpleNamespace(status=403, reason="Forbidden")
        super().__init__(f'<HttpError 403 when requesting {URL} returned "quota">')


def test_summary_uses_status_and_reason_only():
    from research.lib import youtube

    out = youtube._api_error_summary(_FakeHttpError())
    assert SECRET not in out
    assert "403" in out and "Forbidden" in out


def test_summary_redacts_key_from_plain_exceptions():
    from research.lib import youtube

    out = youtube._api_error_summary(RuntimeError(f"connection failed for {URL}"))
    assert SECRET not in out
    assert "key=REDACTED" in out


def _install_fake_client(monkeypatch, exc: Exception):
    class _Call:
        def execute(self):
            raise exc

    class _Resource:
        def list(self, **_kw):
            return _Call()

    class _Client:
        def videos(self):
            return _Resource()

        def commentThreads(self):
            return _Resource()

    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *a, **kw: _Client()
    pkg = types.ModuleType("googleapiclient")
    pkg.discovery = discovery
    httplib2 = types.ModuleType("httplib2")
    httplib2.Http = lambda *a, **kw: object()
    monkeypatch.setitem(sys.modules, "googleapiclient", pkg)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setitem(sys.modules, "httplib2", httplib2)


def test_metadata_and_comments_errors_do_not_print_key(monkeypatch, capsys):
    from research.lib import youtube

    monkeypatch.setattr(youtube, "YOUTUBE_API_KEY", lambda: SECRET)
    _install_fake_client(monkeypatch, _FakeHttpError())
    assert youtube.get_video_metadata("vid") is None
    assert youtube.get_top_comments("vid") == []
    printed = capsys.readouterr().out
    assert SECRET not in printed
    assert "metadata error" in printed and "comments error" in printed
    assert "HTTP 403" in printed
