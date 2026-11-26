"""Direct arXiv Atom API lookup: bibliographic metadata and abstracts, not proofs."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import socket
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ENDPOINT = "https://export.arxiv.org/api/query"
RESULT_FIELDS = ("title", "authors", "abstract", "arxiv_id", "published", "updated", "url", "pdf_url")
_ATOM = "{http://www.w3.org/2005/Atom}"
_MAX_BYTES = 2 * 1024 * 1024
_CACHE_TTL = 3600
_CACHE_LIMIT = 128
_INTERVAL = 3.0
_BACKOFF_INITIAL = 60
_BACKOFF_MAX = 3600
_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?\Z")


def _parse(data: bytes, limit: int) -> list[dict]:
    xml = data.decode("utf-8-sig")
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("unsafe_xml")
    root = ET.fromstring(xml)
    if root.tag != _ATOM + "feed":
        raise ValueError("invalid_atom_feed")
    results = []
    for entry in root.findall(_ATOM + "entry"):
        def text(name):
            return " ".join((entry.findtext(_ATOM + name) or "").split())
        url = urllib.parse.urlsplit(text("id"))
        identifier = url.path.removeprefix("/abs/")
        if (url.hostname not in ("arxiv.org", "export.arxiv.org") or url.scheme not in ("http", "https")
                or not url.path.startswith("/abs/") or not _ID.fullmatch(identifier)):
            raise ValueError("invalid_arxiv_entry")
        title, abstract = text("title"), text("summary")
        if not title or not abstract:
            raise ValueError("incomplete_arxiv_entry")
        results.append(dict(title=title, abstract=abstract, arxiv_id=identifier,
                            authors=[" ".join((author.findtext(_ATOM + "name") or "").split())
                                     for author in entry.findall(_ATOM + "author")],
                            published=text("published"), updated=text("updated"),
                            url="https://arxiv.org/abs/" + identifier,
                            pdf_url="https://arxiv.org/pdf/" + identifier))
        if len(results) >= limit:
            break
    return results


def _cache_dir() -> Path:
    runtime = Path(os.environ.get("HYPROVER_RUNTIME", str(Path(__file__).resolve().parents[2] / "runtime")))
    return runtime / "arxiv-search-cache"


def _cached_results(path: Path, limit: int) -> list[dict] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        stat = path.stat()
        if not 0 <= time.time() - stat.st_mtime < _CACHE_TTL or stat.st_size > _MAX_BYTES:
            return None
        rows = json.loads(path.read_text())
        if (isinstance(rows, list) and len(rows) <= limit
                and all(isinstance(row, dict) and set(row) == set(RESULT_FIELDS) for row in rows)):
            return rows
    except (OSError, ValueError):
        pass
    return None


def _error_type(exc: BaseException) -> str:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, TimeoutError):
        return "timeout"
    if isinstance(reason, socket.gaierror):
        return "dns"
    if isinstance(reason, ssl.SSLError):
        return "tls"
    if isinstance(reason, PermissionError):
        return "permission_denied"
    if isinstance(reason, OSError) and reason.errno in (errno.ENOSPC, errno.EDQUOT):
        return "disk_full"
    if isinstance(reason, RuntimeError):
        return "filesystem_runtime_error"
    return "transport" if isinstance(exc, (urllib.error.URLError, http.client.HTTPException)) else "io"


def _load_cooldown(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("invalid_cooldown_state")
    try:
        state = json.loads(path.read_text())
    except FileNotFoundError:
        return {"retry_until": 0, "failures": 0, "http_status": 429}
    try:
        valid = (isinstance(state, dict)
                 and type(state.get("retry_until")) in (int, float)
                 and math.isfinite(float(state["retry_until"])) and state["retry_until"] >= 0
                 and type(state.get("failures")) is int and 0 <= state["failures"] <= 16
                 and state.get("http_status") in (429, 503))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("invalid_cooldown_state")
    return state


def _write_cooldown(path: Path, state: dict) -> None:
    fd, name = tempfile.mkstemp(prefix="cooldown-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _retry_delay(header: str | None, now: float, failures: int) -> tuple[float, str]:
    if header:
        try:
            value = header.strip()
            if re.fullmatch(r"[0-9]+", value):
                delay = float(value)
            else:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = date.timestamp() - now
            if math.isfinite(delay) and delay >= 0 and math.isfinite(now + delay):
                return max(_INTERVAL, delay), "retry_after"
        except (ValueError, TypeError, OverflowError):
            pass
    return min(_BACKOFF_MAX, _BACKOFF_INITIAL * 2 ** min(failures - 1, 6)), "exponential_backoff"


def _store_cache(target: Path, results: list[dict], success: dict) -> None:
    temporary = None
    try:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            success["cache_warning"] = "unsafe_cache_target"
            return
        files = [p for p in target.parent.glob("*.json")
                 if re.fullmatch(r"[0-9a-f]{64}\.json", p.name)
                 and not p.is_symlink() and p.is_file()]
        recycle = None
        if not target.exists() and len(files) >= _CACHE_LIMIT:
            expired = sorted((p for p in files if time.time() - p.stat().st_mtime >= _CACHE_TTL),
                             key=lambda p: p.stat().st_mtime)
            if not expired or len(files) > _CACHE_LIMIT:
                success["cache_warning"] = "cache_full_write_skipped"
                return
            recycle = expired[0]
        fd, name = tempfile.mkstemp(prefix=target.stem + "-", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w") as handle:
            json.dump(results, handle, ensure_ascii=False)
        if recycle is not None:
            recycle.replace(target)
        temporary.replace(target)
    except (OSError, RuntimeError) as exc:
        success.update(cache_warning="cache_write_failed", cache_error_type=_error_type(exc))
    finally:
        try:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        except (OSError, RuntimeError) as exc:
            success.update(cache_warning="cache_cleanup_failed", cache_error_type=_error_type(exc))


def search(query: str, num_results: int = 10, timeout: float = 30, *,
           fetch: Callable[[urllib.request.Request, int], bytes] | None = None,
           pace: Callable[[float], None] | None = None) -> dict:
    """Search paper metadata. Plain keywords or arXiv fielded query syntax are accepted.

    Calls share a filesystem lock, a three-second request interval and a bounded
    one-hour cache within this deployment. An internal caller can supply a bounded
    transport and sleeper without changing the shared request/cache policy.
    """
    out = {"query": query, "count": 0, "results": [], "endpoint": ENDPOINT,
           "source": "arxiv", "result_type": "paper_metadata", "is_verified": False}
    if os.environ.get("HYPROVER_LITERATURE_SEARCH", "1") == "0":
        return dict(out, error="literature_search_disabled_for_this_run")
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        return dict(out, error="invalid_query")
    if type(num_results) is not int or not 1 <= num_results <= 20:
        return dict(out, error="num_results_must_be_1_to_20")
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        return dict(out, error="invalid_timeout")
    q = query.strip()
    expression = q if ":" in q else " AND ".join('all:"' + word.replace('"', '') + '"' for word in q.split())
    key = hashlib.sha256(json.dumps([expression, num_results]).encode()).hexdigest()
    cache = _cache_dir()
    target = cache / (key + ".json")
    stage = "cache_lock"
    try:
        if pace is not None:
            pace(0)
        cached = _cached_results(target, num_results)
        if cached is not None:
            return dict(out, count=len(cached), results=cached, cached=True)
        cache.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(cache / "request.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(out, error="search_busy", error_stage=stage, retry_after_seconds=3)
            cached = _cached_results(target, num_results)
            if cached is not None:
                return dict(out, count=len(cached), results=cached, cached=True)
            stage = "rate_limit_state"
            lock.seek(0)
            if lock.read(1) == "!":
                return dict(out, error="search_unavailable", error_stage=stage,
                            error_type="cooldown_persistence_failed")
            cooldown_path = cache / "cooldown.json"
            try:
                cooldown = _load_cooldown(cooldown_path)
            except ValueError:
                return dict(out, error="search_unavailable", error_stage=stage,
                            error_type="invalid_cooldown_state")
            remaining = cooldown["retry_until"] - time.time()
            if remaining > 0:
                return dict(out, error="http_error", error_stage="cooldown",
                            http_status=cooldown["http_status"], retry_deferred=True,
                            retry_after_seconds=math.ceil(remaining), retry_source="shared_cooldown")
            lock.seek(0)
            try:
                last_request = float(lock.read() or "0")
                remaining = _INTERVAL - (time.time() - last_request) if math.isfinite(last_request) else _INTERVAL
            except ValueError:
                remaining = _INTERVAL
            if remaining > 0:
                (pace or time.sleep)(min(remaining, _INTERVAL))
            elif pace is not None:
                pace(0)
            lock.seek(0)
            lock.truncate()
            lock.write(str(time.time()))
            lock.flush()
            params = urllib.parse.urlencode({"search_query": expression, "start": 0,
                                             "max_results": num_results, "sortBy": "relevance"})
            req = urllib.request.Request(ENDPOINT + "?" + params,
                                         headers={"Accept": "application/atom+xml", "User-Agent": "hyprover/0.1.0"})
            stage = "network"
            try:
                if fetch is None:
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        data = response.read(_MAX_BYTES + 1)
                else:
                    data = fetch(req, _MAX_BYTES + 1)
            except urllib.error.HTTPError as exc:
                result = dict(out, error="http_error", error_stage=stage,
                              http_status=exc.code, retry_after_seconds=None)
                try:
                    if exc.code in (429, 503):
                        failures = min(cooldown["failures"] + 1, 16)
                        now = time.time()
                        delay, source = _retry_delay(exc.headers.get("Retry-After") if exc.headers else None,
                                                    now, failures)
                        result.update(retry_after_seconds=math.ceil(delay), retry_source=source)
                        try:
                            lock.seek(0)
                            timestamp = lock.read()
                            lock.seek(0)
                            lock.write("!")
                            lock.flush()
                            _write_cooldown(cooldown_path, {"retry_until": now + delay,
                                                           "failures": failures, "http_status": exc.code})
                            lock.seek(0)
                            lock.write(timestamp)
                            lock.truncate()
                            lock.flush()
                        except (OSError, RuntimeError) as error:
                            result.update(cooldown_warning="cooldown_write_failed",
                                          cooldown_error_type=_error_type(error))
                finally:
                    exc.close()
                return result
            stage = "parse"
            if len(data) > _MAX_BYTES:
                return dict(out, error="response_too_large", error_stage=stage)
            try:
                results = _parse(data, num_results)
            except (ET.ParseError, ValueError):
                return dict(out, error="invalid_arxiv_response", error_stage="parse")
            success = dict(out, count=len(results), results=results, cached=False)
            if cooldown["failures"]:
                try:
                    _write_cooldown(cooldown_path, {"retry_until": 0, "failures": 0, "http_status": 429})
                except (OSError, RuntimeError) as exc:
                    success.update(cooldown_warning="cooldown_reset_failed", cooldown_error_type=_error_type(exc))
            _store_cache(target, results, success)
            return success
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        return dict(out, error="search_unavailable", error_stage=stage, error_type=_error_type(exc))
