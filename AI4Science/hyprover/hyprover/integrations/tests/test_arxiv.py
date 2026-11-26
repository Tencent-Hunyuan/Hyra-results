"""Offline public arXiv API contract and deployment-wide rate-limit tests."""
import errno
import fcntl
import http.client
import io
import json
import multiprocessing
import os
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
import socket
import ssl
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from hyprover.integrations import arxiv

FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2301.12345v2</id><title> A sample\n paper </title>
<author><name>A. Researcher</name></author><summary> An abstract, not a theorem. </summary>
<published>2023-01-01T00:00:00Z</published><updated>2023-01-02T00:00:00Z</updated>
</entry></feed>'''


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_RUNTIME", str(tmp_path))
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    calls, sleeps = [], []
    monkeypatch.setattr(arxiv.time, "time", lambda: 10000.0)
    monkeypatch.setattr(arxiv.time, "sleep", sleeps.append)
    def request(req, timeout):
        calls.append(req)
        assert req.get_header("Accept") == "application/atom+xml"
        assert req.get_header("User-agent") == "hyprover/0.1.0"
        assert timeout == 30
        return io.BytesIO(FEED)
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", request)
    return calls, sleeps


def test_metadata_and_bounded_request(transport):
    calls, _ = transport
    result = arxiv.search("convex body", 3)
    assert result["count"] == 1 and result["is_verified"] is False
    assert result["result_type"] == "paper_metadata"
    row = result["results"][0]
    assert set(row) == set(arxiv.RESULT_FIELDS)
    assert row["title"] == "A sample paper"
    assert row["authors"] == ["A. Researcher"]
    assert row["url"] == "https://arxiv.org/abs/2301.12345v2"
    assert "theorem" not in row
    query = parse_qs(urlsplit(calls[0].full_url).query)
    assert query["search_query"] == ['all:"convex" AND all:"body"']
    assert query["max_results"] == ["3"]


@pytest.mark.parametrize("query,limit", [("", 10), (None, 10), ("x" * 1001, 10), ("x", True), ("x", 0), ("x", 21), ("x", "3")])
def test_bad_inputs_never_call_network(transport, query, limit):
    assert "error" in arxiv.search(query, limit)
    assert not transport[0]


def test_disabled_search_never_calls_network(transport, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    assert arxiv.search("x")["error"] == "literature_search_disabled_for_this_run"
    assert not transport[0]


def test_fielded_query_and_shared_rate_limit(transport):
    calls, sleeps = transport
    arxiv.search('ti:"convex bodies"')
    arxiv.search('au:"Gardner"')
    assert len(calls) == 2 and sleeps == [3.0]
    assert parse_qs(urlsplit(calls[0].full_url).query)["search_query"] == ['ti:"convex bodies"']


def test_cache_deduplicates_and_is_bounded(transport, monkeypatch):
    import os
    arxiv.search("same")
    for path in arxiv._cache_dir().glob("*.json"):
        os.utime(path, (10000, 10000))
    assert arxiv.search("same")["cached"] is True
    assert len(transport[0]) == 1
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 2)
    for q in ("one", "two", "three"):
        arxiv.search(q)
    assert len(list(arxiv._cache_dir().glob("*.json"))) == 2


def test_busy_lock_does_not_call_network(transport):
    cache = arxiv._cache_dir()
    cache.mkdir()
    with (cache / "request.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = arxiv.search("x")
    assert result["error"] == "search_busy" and result["retry_after_seconds"] == 3
    assert not transport[0]


@pytest.mark.parametrize("payload", [b"not xml", b"<html/>", b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>', FEED.replace(b"arxiv.org/abs", b"example.com/abs"), FEED.replace(b"/abs/2301.12345v2", b"/api/errors#bad"), b"x" * (2 * 1024 * 1024 + 1)])
def test_invalid_responses_are_not_empty_success(transport, monkeypatch, payload):
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    result = arxiv.search("x")
    assert "error" in result and result["results"] == []


@pytest.mark.parametrize("exception,code", [(TimeoutError(), "search_unavailable"), (urllib.error.URLError("offline"), "search_unavailable"), (urllib.error.HTTPError(arxiv.ENDPOINT, 429, "slow down", {}, None), "http_error")])
def test_errors_are_explicit(transport, monkeypatch, exception, code):
    def fail(*a, **k):
        raise exception
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", fail)
    result = arxiv.search("x")
    assert result["error"] == code and result["results"] == []
    if code == "http_error":
        assert result["http_status"] == 429 and result["retry_after_seconds"] == 60


def test_incomplete_http_body_returns_error(transport, monkeypatch):
    class Truncated(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b"partial", 100)
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: Truncated())
    assert arxiv.search("x")["error"] == "search_unavailable"


@pytest.mark.parametrize("failure", ["write", "cleanup"])
def test_cache_errors_preserve_successful_results(transport, monkeypatch, failure):
    if failure == "write":
        def fail(*args, **kwargs):
            raise OSError(errno.ENOSPC, "disk full")
        monkeypatch.setattr(arxiv.tempfile, "mkstemp", fail)
    else:
        original = Path.unlink
        def unlink(path, *args, **kwargs):
            if path.suffix == ".tmp":
                raise OSError("cleanup failed")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(Path, "replace", lambda *a: (_ for _ in ()).throw(OSError("replace failed")))
    result = arxiv.search("x")
    assert result["count"] == 1 and "error" not in result
    assert result["cache_warning"] == ("cache_write_failed" if failure == "write" else "cache_cleanup_failed")


def test_utf16_xml_cannot_bypass_entity_rejection(transport, monkeypatch):
    payload = '<!DOCTYPE feed [<!ENTITY x "bad">]><feed/>'.encode("utf-16")
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    assert arxiv.search("x")["error"] == "invalid_arxiv_response"


def test_empty_feed_is_valid(transport, monkeypatch):
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'<feed xmlns="http://www.w3.org/2005/Atom"/>'))
    result = arxiv.search("unmatched")
    assert result["count"] == 0 and "error" not in result


def _make_cache_fresh():
    for path in arxiv._cache_dir().glob("*.json"):
        if len(path.stem) == 64:
            os.utime(path, (10000, 10000))


def _http_failure(monkeypatch, status=429, headers=None):
    calls = []
    def request(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(arxiv.ENDPOINT, status, "limited", headers or {}, io.BytesIO())
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", request)
    return calls


@pytest.mark.parametrize("status", [429, 503])
def test_shared_cooldown_blocks_other_queries(transport, monkeypatch, status):
    calls = _http_failure(monkeypatch, status)
    first = arxiv.search("first")
    assert first["http_status"] == status and first["retry_after_seconds"] == 60
    assert first["retry_source"] == "exponential_backoff"
    monkeypatch.setattr(arxiv.time, "time", lambda: 10010.1)
    second = arxiv.search("different worker query")
    assert second["retry_deferred"] is True
    assert second["retry_after_seconds"] == 50
    assert second["error_stage"] == "cooldown"
    assert second["http_status"] == status
    assert len(calls) == 1


@pytest.mark.parametrize("header,expected,source", [
    ("120", 120, "retry_after"),
    (format_datetime(datetime.fromtimestamp(10120, timezone.utc), usegmt=True), 120, "retry_after"),
    ("7200", 7200, "retry_after"),
    ("0", 3, "retry_after"),
    ("-10", 60, "exponential_backoff"),
    ("NaN", 60, "exponential_backoff"),
    ("invalid", 60, "exponential_backoff"),
    (format_datetime(datetime.fromtimestamp(9990, timezone.utc), usegmt=True), 60, "exponential_backoff"),
])
def test_retry_after_and_fallback(transport, monkeypatch, header, expected, source):
    calls = _http_failure(monkeypatch, headers={"Retry-After": header})
    result = arxiv.search("x")
    assert result["retry_after_seconds"] == expected and result["retry_source"] == source
    state = json.loads((arxiv._cache_dir() / "cooldown.json").read_text())
    assert state["retry_until"] == 10000 + expected
    assert len(calls) == 1


def test_backoff_increases_caps_and_resets(transport, monkeypatch):
    calls = _http_failure(monkeypatch)
    now = 10000
    for delay in (60, 120, 240, 480, 960, 1920, 3600, 3600):
        monkeypatch.setattr(arxiv.time, "time", lambda: now)
        result = arxiv.search("x")
        assert result["retry_after_seconds"] == delay
        now += delay
    assert len(calls) == 8
    monkeypatch.setattr(arxiv.time, "time", lambda: now)
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(FEED))
    assert arxiv.search("x")["count"] == 1
    state = json.loads((arxiv._cache_dir() / "cooldown.json").read_text())
    assert state["retry_until"] == 0 and state["failures"] == 0
    _http_failure(monkeypatch)
    assert arxiv.search("new query")["retry_after_seconds"] == 60


@pytest.mark.parametrize("empty", [False, True])
def test_fresh_cache_bypasses_busy_lock_and_cooldown(transport, monkeypatch, empty):
    if empty:
        monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(
            b'<feed xmlns="http://www.w3.org/2005/Atom"/>'))
    assert "error" not in arxiv.search("cached")
    _make_cache_fresh()
    calls = _http_failure(monkeypatch)
    arxiv.search("uncached")
    with (arxiv._cache_dir() / "request.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = arxiv.search("cached")
    assert result["cached"] is True and result["count"] == (0 if empty else 1)
    assert "error" not in result and len(calls) == 1


def test_cache_rechecked_after_lock_acquisition(transport, monkeypatch):
    original = arxiv._cached_results
    arxiv.search("cached")
    _make_cache_fresh()
    reads = []
    def miss_once(path, limit):
        reads.append(path)
        return None if len(reads) == 1 else original(path, limit)
    monkeypatch.setattr(arxiv, "_cached_results", miss_once)
    assert arxiv.search("cached")["cached"] is True
    assert len(transport[0]) == 1 and len(reads) == 2


def test_full_fresh_cache_keeps_existing_files(transport, monkeypatch):
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 1)
    arxiv.search("first")
    _make_cache_fresh()
    before = {p.name: p.read_bytes() for p in arxiv._cache_dir().glob("*.json")}
    def no_deletes(*args, **kwargs):
        pytest.fail("fresh cache files must not be evicted")
    monkeypatch.setattr(Path, "unlink", no_deletes)
    result = arxiv.search("second")
    assert result["count"] == 1 and result["cache_warning"] == "cache_full_write_skipped"
    assert {p.name: p.read_bytes() for p in arxiv._cache_dir().glob("*.json")} == before


def test_expired_cache_reclaimed_but_not_other_files(transport, monkeypatch, tmp_path):
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 2)
    arxiv.search("old")
    cache = arxiv._cache_dir()
    expired = next(cache.glob("*.json"))
    os.utime(expired, (1000, 1000))
    arxiv.search("fresh")
    fresh = next(p for p in cache.glob("*.json") if p != expired)
    os.utime(fresh, (10000, 10000))
    unrelated = cache / "notes.json"
    unrelated.write_text("keep")
    other = tmp_path / "external.json"
    other.write_text("keep")
    link = cache / ("f" * 64 + ".json")
    link.symlink_to(other)
    directory = cache / ("e" * 64 + ".json")
    directory.mkdir()
    result = arxiv.search("new")
    assert result["count"] == 1 and "cache_warning" not in result
    assert not expired.exists() and fresh.exists()
    assert unrelated.read_text() == other.read_text() == "keep"
    assert link.is_symlink() and directory.is_dir()
    _make_cache_fresh()
    assert arxiv.search("new")["cached"] is True
    assert len(transport[0]) == 3


def test_expired_cache_not_served_during_cooldown(transport, monkeypatch):
    arxiv.search("old")
    for p in arxiv._cache_dir().glob("*.json"):
        os.utime(p, (1000, 1000))
    calls = _http_failure(monkeypatch)
    arxiv.search("different")
    result = arxiv.search("old")
    assert result["retry_deferred"] is True and result["results"] == []
    assert len(calls) == 1


@pytest.mark.parametrize("error,category", [
    (TimeoutError("private details"), "timeout"),
    (urllib.error.URLError(socket.gaierror(-2, "private details")), "dns"),
    (urllib.error.URLError(ssl.SSLError("private details")), "tls"),
    (urllib.error.URLError("private details"), "transport"),
])
def test_safe_network_error_classification(transport, monkeypatch, error, category):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", fail)
    result = arxiv.search("x")
    assert result["error"] == "search_unavailable"
    assert result["error_stage"] == "network" and result["error_type"] == category
    assert "private details" not in json.dumps(result)


@pytest.mark.parametrize("error,category", [
    (OSError(errno.ENOSPC, "private details"), "disk_full"),
    (PermissionError(errno.EACCES, "private details"), "permission_denied"),
])
def test_cache_setup_errors_are_not_network_errors(transport, monkeypatch, error, category):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(Path, "mkdir", fail)
    result = arxiv.search("x")
    assert result["error_stage"] == "cache_lock" and result["error_type"] == category
    assert not transport[0] and "private details" not in json.dumps(result)


@pytest.mark.parametrize("state", ["not json", "[]", '{"retry_until": NaN}',
                                   '{"retry_until": 0, "failures": -1, "http_status": 429}',
                                   json.dumps({"retry_until": 10**400, "failures": 1, "http_status": 429})])
def test_corrupt_cooldown_fails_closed(transport, state):
    cache = arxiv._cache_dir()
    cache.mkdir()
    (cache / "cooldown.json").write_text(state)
    result = arxiv.search("x")
    assert result["error_type"] == "invalid_cooldown_state"
    assert result["error_stage"] == "rate_limit_state" and not transport[0]


def test_cooldown_write_failure_keeps_http_diagnostic(transport, monkeypatch):
    calls = _http_failure(monkeypatch)
    def fail(*a, **k):
        raise OSError(errno.ENOSPC, "private details")
    monkeypatch.setattr(arxiv, "_write_cooldown", fail)
    result = arxiv.search("x")
    assert result["http_status"] == 429 and result["cooldown_warning"] == "cooldown_write_failed"
    assert result["cooldown_error_type"] == "disk_full" and len(calls) == 1
    monkeypatch.setattr(arxiv.time, "time", lambda: 20000)
    blocked = arxiv.search("later different query")
    assert blocked["error_type"] == "cooldown_persistence_failed"
    assert blocked["error_stage"] == "rate_limit_state" and len(calls) == 1


def test_nonretryable_http_error_does_not_set_cooldown(transport, monkeypatch):
    calls = _http_failure(monkeypatch, 400)
    result = arxiv.search("x")
    assert result["http_status"] == 400 and result["retry_after_seconds"] is None
    assert not (arxiv._cache_dir() / "cooldown.json").exists()
    assert len(calls) == 1


def _child_cooldown_search(runtime, output):
    os.environ["HYPROVER_RUNTIME"] = runtime
    os.environ["HYPROVER_LITERATURE_SEARCH"] = "1"
    arxiv.time.time = lambda: 10001
    def no_network(*a, **k):
        raise AssertionError("a separate process must respect the shared cooldown")
    arxiv.urllib.request.urlopen = no_network
    output.put(arxiv.search("another process"))


@pytest.mark.parametrize("persistence_failure", [False, True])
def test_cooldown_survives_new_process(transport, monkeypatch, persistence_failure):
    _http_failure(monkeypatch)
    if persistence_failure:
        def fail(*a, **k):
            raise OSError(errno.ENOSPC, "disk full")
        monkeypatch.setattr(arxiv, "_write_cooldown", fail)
    arxiv.search("parent")
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_child_cooldown_search,
                              args=(os.environ["HYPROVER_RUNTIME"], output))
    process.start()
    try:
        result = output.get(timeout=15)
        process.join(timeout=15)
        assert process.exitcode == 0
        if persistence_failure:
            assert result["error_type"] == "cooldown_persistence_failed"
        else:
            assert result["retry_deferred"] is True and result["retry_after_seconds"] == 59
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        output.close()
        output.join_thread()


def test_cache_target_symlink_is_not_replaced(transport, tmp_path, monkeypatch):
    arxiv.search("linked")
    cache = arxiv._cache_dir()
    target = next(cache.glob("*.json"))
    target.unlink()
    outside = tmp_path / "protected.json"
    outside.write_text("unchanged")
    target.symlink_to(outside)
    result = arxiv.search("linked")
    assert result["count"] == 1 and result["cache_warning"] == "unsafe_cache_target"
    assert target.is_symlink() and outside.read_text() == "unchanged"


def test_fixed_temporary_symlinks_are_never_used(transport, tmp_path, monkeypatch):
    arxiv.search("cached")
    cache = arxiv._cache_dir()
    target = next(cache.glob("*.json"))
    os.utime(target, (1000, 1000))
    outside = tmp_path / "protected"
    outside.write_text("unchanged")
    links = [target.with_suffix(".tmp"), cache / "cooldown.tmp"]
    for link in links:
        link.symlink_to(outside)
    assert arxiv.search("cached")["count"] == 1
    _http_failure(monkeypatch)
    result = arxiv.search("limited")
    assert result["http_status"] == 429 and "cooldown_warning" not in result
    assert outside.read_text() == "unchanged" and all(p.is_symlink() for p in links)


@pytest.mark.parametrize("control", ["request.lock", "cooldown.json"])
def test_control_symlink_does_not_follow_or_modify_target(transport, tmp_path, control):
    cache = arxiv._cache_dir()
    cache.mkdir()
    outside = tmp_path / "protected"
    outside.write_text("unchanged")
    link = cache / control
    link.symlink_to(outside)
    result = arxiv.search("x")
    assert result["error"] == "search_unavailable" and not transport[0]
    assert outside.read_text() == "unchanged" and link.is_symlink()


def test_failed_fetch_does_not_evict_expired_cache(transport, monkeypatch):
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 1)
    arxiv.search("old")
    cache = arxiv._cache_dir()
    old = next(cache.glob("*.json"))
    os.utime(old, (1000, 1000))
    before = old.read_bytes()
    _http_failure(monkeypatch)
    assert arxiv.search("new")["http_status"] == 429
    assert old.read_bytes() == before


def test_legacy_request_timestamp_still_throttles(transport):
    cache = arxiv._cache_dir()
    cache.mkdir()
    (cache / "request.lock").write_text("9999")
    assert arxiv.search("x")["count"] == 1
    assert transport[1] == [2.0]


def test_expired_slot_reuse_does_not_require_delete(transport, monkeypatch):
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 1)
    arxiv.search("old")
    cache = arxiv._cache_dir()
    old = next(cache.glob("*.json"))
    os.utime(old, (1000, 1000))
    def no_delete(*args, **kwargs):
        raise RecursionError("filesystem trash integration failed")
    monkeypatch.setattr(Path, "unlink", no_delete)
    result = arxiv.search("new")
    assert result["count"] == 1 and "cache_warning" not in result
    _make_cache_fresh()
    assert arxiv.search("new")["cached"] is True
    assert len(list(cache.glob("*.json"))) == 1


def test_failed_slot_replacement_never_serves_old_data_as_fresh(transport, monkeypatch):
    monkeypatch.setattr(arxiv, "_CACHE_LIMIT", 1)
    arxiv.search("old")
    cache = arxiv._cache_dir()
    old = next(cache.glob("*.json"))
    os.utime(old, (1000, 1000))
    original = Path.replace
    def fail(path, target):
        if path.suffix == ".tmp":
            raise OSError("replacement failed")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", fail)
    result = arxiv.search("new")
    assert result["count"] == 1 and result["cache_warning"] == "cache_write_failed"
    target = next(cache.glob("*.json"))
    assert arxiv._cached_results(target, 10) is None


def test_cache_runtime_failure_keeps_successful_result(transport, monkeypatch):
    def fail(*a, **k):
        raise RecursionError("filesystem integration failed")
    monkeypatch.setattr(Path, "replace", fail)
    result = arxiv.search("x")
    assert result["count"] == 1 and "error" not in result
    assert result["cache_error_type"] == "filesystem_runtime_error"
