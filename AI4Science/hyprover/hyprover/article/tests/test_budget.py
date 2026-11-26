"""Offline checks for article network deadlines and worker cancellation."""
import fcntl
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hyprover.article import agent, budget, pdf_sources, references
from hyprover.article.models import ArticleConfig, Cancelled
from hyprover.integrations import arxiv


def test_model_headers_use_only_the_job_time_left(monkeypatch):
    now = [59.9]
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(agent, "backend", lambda *_, **__: ("https://fixture.invalid/v1", "fixture-key"))
    trace = []

    class Connection:
        def __init__(self, host, port, timeout):
            trace.append(timeout)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            now[0] = 60
            return SimpleNamespace(status=200)

        def close(self):
            trace.append("closed")

    monkeypatch.setattr(agent.http.client, "HTTPSConnection", Connection)
    with pytest.raises(budget.ArticleDeadlineExceeded, match="total_time_budget"):
        agent.Agent(ArticleConfig(timeout_seconds=1800), deadline=60)("fixture", {}, "fixture")

    assert trace == [pytest.approx(0.1), "closed"]


def test_model_cancel_interrupts_a_blocked_response_body(monkeypatch):
    monkeypatch.setattr(agent, "backend", lambda *_, **__: ("https://fixture.invalid/v1", "fixture-key"))
    cancelled, reading, finished, rescued = (threading.Event() for _ in range(4))

    # A socket pair lets the adapter block in a real read without any server
    # or model call. The second guard keeps a failing test from hanging.
    reader, peer = socket.socketpair()

    class Response:
        status = 200

        def getheader(self, name, default=None):
            return "application/json"

        def read(self, count):
            reading.set()
            return reader.recv(count)

    class Connection:
        sock = reader

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            reader.close()

    def cancel():
        if reading.wait(1):
            cancelled.set()
        if not finished.wait(2):
            rescued.set()
            peer.shutdown(socket.SHUT_RDWR)

    monkeypatch.setattr(agent.http.client, "HTTPSConnection", Connection)
    worker = threading.Thread(target=cancel)
    with reader, peer:
        worker.start()
        try:
            with pytest.raises(Cancelled):
                agent.Agent(ArticleConfig(), cancelled=cancelled.is_set)("fixture", {}, "fixture")
        finally:
            finished.set()
            worker.join()
    assert not rescued.is_set()


def test_pdf_download_and_parser_share_the_same_remaining_deadline(monkeypatch):
    now = [59.5]
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(references, "_public_addresses", lambda _: ["8.8.8.8"])
    monkeypatch.setattr(pdf_sources.shutil, "which", lambda name: "/fixture/" + name)
    seen = []

    class Response:
        status = 200

        def getheader(self, name, default=None):
            return "application/pdf"

        def read(self, count):
            result = b"%PDF-1.7\nfixture" if now[0] == 59.5 else b""
            now[0] = 59.8
            return result

    class Connection:
        sock = None

        def __init__(self, host, addresses, timeout):
            seen.append(timeout)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            seen.append("closed")

    def parse(command, shared, **kwargs):
        seen.append(shared.remaining())
        now[0] = 60
        shared.remaining()

    monkeypatch.setattr(references, "_PinnedHTTPS", Connection)
    monkeypatch.setattr(pdf_sources, "run_command", parse)
    reader = references.SourceReader(ArticleConfig(source_timeout=60), True, deadline=60)
    with pytest.raises(budget.ArticleDeadlineExceeded, match="total_time_budget"):
        reader.get_text("https://numdam.org/item/fixture.pdf")
    assert seen == [pytest.approx(0.5), pytest.approx(0.2), "closed"]


def test_cancel_kills_and_reaps_a_local_subprocess(tmp_path):
    pid_file = tmp_path / "child.pid"
    cancel_at = time.monotonic() + 2
    command = [sys.executable, "-c", "import os, pathlib, sys, time; "
               "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(5)", str(pid_file)]
    operation = budget.Budget(6, "fixture_timeout",
                              cancelled=lambda: pid_file.is_file() or time.monotonic() >= cancel_at)
    with pytest.raises(Cancelled):
        budget.run_command(command, operation, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_article_arxiv_cancellation_in_shared_pacing_releases_the_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    monkeypatch.setenv("HYPROVER_RUNTIME", str(tmp_path))
    path = arxiv._cache_dir()
    path.mkdir()
    lock = path / "request.lock"
    mark = str(time.time())
    lock.write_text(mark)
    cancelled = threading.Event()
    fake_clock = SimpleNamespace(monotonic=lambda: 1.0, sleep=lambda _: cancelled.set())
    monkeypatch.setattr(budget, "time", fake_clock)
    monkeypatch.setattr(references, "_public_addresses", lambda _: pytest.fail("unexpected network"))

    reader = references.SourceReader(ArticleConfig(), True, deadline=60, cancelled=cancelled.is_set)
    with pytest.raises(Cancelled):
        reader.search_arxiv("id:2301.12345", 1)

    assert lock.read_text() == mark
    with lock.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("status", [200, 429, 503, "deadline"])
def test_article_arxiv_transport_uses_the_job_time_and_shares_cache_and_cooldown(monkeypatch, tmp_path, status):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    monkeypatch.setenv("HYPROVER_RUNTIME", str(tmp_path))
    now = [59.8]
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(references, "_public_addresses", lambda _: ["8.8.8.8"])
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *_, **__: pytest.fail("unexpected legacy transport"))
    calls = []

    class Connection:
        reason = "fixture"
        headers = {"Retry-After": "35"}

        def __init__(self, host, addresses, timeout):
            assert host == "export.arxiv.org" and timeout == pytest.approx(0.2)

        def request(self, method, path, headers):
            assert method == "GET" and path.startswith("/api/query?") and headers["Accept"] == "application/atom+xml"
            calls.append(path)

        def getresponse(self):
            self.status = 200 if status == "deadline" else status
            return self

        def read(self, limit):
            assert self.status == 200 and limit == 2097153
            if status == "deadline":
                now[0] = 60
            return b'<feed xmlns="http://www.w3.org/2005/Atom"/>'

        def close(self):
            pass

    monkeypatch.setattr(references, "_PinnedHTTPS", Connection)
    reader = references.SourceReader(ArticleConfig(), True, deadline=60)

    if status == "deadline":
        with pytest.raises(budget.ArticleDeadlineExceeded, match="total_time_budget"):
            reader.search_arxiv("id:2301.12345", 1)
        assert len(calls) == 1 and not list(arxiv._cache_dir().glob("*.json"))
        return
    first = reader.search_arxiv("id:2301.12345", 1)
    if status == 200:
        assert first["cached"] is False
        assert arxiv.search("id:2301.12345", 1)["cached"] is True
    else:
        assert first["http_status"] == status and first["retry_after_seconds"] == 35
        later = arxiv.search("id:2312.12345", 1)
        assert later["retry_deferred"] is True and later["http_status"] == status
    assert len(calls) == 1
