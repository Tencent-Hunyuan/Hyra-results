"""Bibliographic provenance, bounded retrieval, and explicit citation obligations."""
from __future__ import annotations

import copy
import http.client
import io
import ipaddress
import json
import re
import socket
import ssl
import tarfile
import time
import unicodedata
import urllib.error
from collections.abc import Callable
from contextlib import contextmanager
from html.parser import HTMLParser
from itertools import chain
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

from hyprover.core._util import utc_now
from hyprover.integrations import arxiv

from .budget import ArticleDeadlineExceeded, Budget, OperationTimeout
from .evidence import units
from . import citation_policy as policy
from .models import (DEFAULT_SOURCE_TEXT_BYTES, FORMAT, ArticleConfig, ArticleError, Cancelled,
                     EvidenceChanged, digest, nonempty_text, read_json, read_text,
                     require_keys, safe_path, string_list, text_hash)

_DOI = re.compile(r"10\.\d{4,9}/[^\s<>]+\Z", re.I)
_ARXIV = re.compile(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?\Z")
_MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024
_NUMDAM_HOSTS = frozenset({"numdam.org", "www.numdam.org"})


def normal(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(c for c in value if c.isalnum())


def identifiers(metadata: dict) -> tuple[str, str]:
    doi = str(metadata.get("doi") or "").removeprefix("https://doi.org/").strip().lower()
    aid = str(metadata.get("arxiv") or metadata.get("arxiv_id") or "").strip()
    aid = aid.removeprefix("https://arxiv.org/abs/").removeprefix("arXiv:")
    if (doi and not _DOI.fullmatch(doi)) or (aid and not _ARXIV.fullmatch(aid)):
        raise ArticleError("invalid_reference_identifier")
    return doi, aid


def reference_id(metadata: dict) -> str:
    doi, aid = identifiers(metadata)
    identity = {"doi": doi, "arxiv": aid} if doi or aid else {
        "title": normal(metadata.get("title", "")), "authors": metadata.get("authors", []),
        "year": metadata.get("year"), "edition": metadata.get("edition")}
    return "ref-" + digest(identity)[:20]


def source_url(url: object) -> str | None:
    """Upgrade known Numdam links, and otherwise require a regular HTTPS source."""
    if not isinstance(url, str) or len(url) > 4096:
        return None
    try:
        parsed = urlsplit(url)
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return None
        if parsed.scheme == "http" and parsed.hostname in _NUMDAM_HOSTS and parsed.port is None:
            parsed = parsed._replace(scheme="https")
        if parsed.scheme == "https" and parsed.port in (None, 443):
            return parsed._replace(fragment="").geturl()
    except ValueError:
        return None
    return None


def merge_source_urls(row: dict, urls) -> None:
    retained = list(row.get("source_urls", []))
    for candidate in urls:
        url = source_url(candidate)
        if url and url not in retained:
            retained.append(url)
    if retained:
        row["source_urls"] = retained


def add_reference(ledger: dict, metadata: dict, *, alias: str = "") -> str:
    if not isinstance(metadata, dict) or not metadata.get("title"):
        raise ArticleError("reference_title_required")
    rid = reference_id(metadata)
    row = ledger["references"].get(rid)
    if row is None:
        ledger["references"][rid] = {"id": rid, "declared": dict(metadata), "metadata": {},
            "metadata_verified": False, "metadata_sources": [], "aliases": [], "source_ids": [],
            "conflicts": [], "related_versions": []}
        row = ledger["references"][rid]
    elif any(row["declared"].get(k) and metadata.get(k)
             and normal(row["declared"][k]) != normal(metadata[k]) for k in ("title", "year", "edition")):
        row["conflicts"].append({"kind": "conflicting_declared_metadata", "candidate": metadata})
    if alias and alias not in row["aliases"]:
        row["aliases"].append(alias)
    merge_source_urls(row, [metadata.get(key) for key in ("url", "source_url", "fulltext_url")])
    doi, aid = identifiers(metadata)
    for other_id, other in ledger["references"].items():
        if other_id == rid:
            continue
        other_doi, other_aid = identifiers(other["declared"])
        if ((doi and doi == other_doi) or (aid and other_aid
                and re.sub(r"v\d+$", "", aid) == re.sub(r"v\d+$", "", other_aid))):
            if other_id not in row["related_versions"]:
                row["related_versions"].append(other_id)
            if rid not in other["related_versions"]:
                other["related_versions"].append(rid)
    return rid


def external_obligations(ledger: dict) -> dict:
    canonical = policy.canonical_ids(ledger)
    internal_groups = {canonical[oid] for oid in ledger.get("internal_resolutions", {}) if oid in canonical}
    return {oid: row for oid, row in ledger["obligations"].items() if canonical[oid] not in internal_groups}


def bind_reference(ledger: dict, obligation_id: str, reference_id: str) -> None:
    if obligation_id not in ledger["obligations"] or reference_id not in ledger["references"]:
        raise ArticleError("unknown_reference_obligation_binding")
    if obligation_id not in external_obligations(ledger):
        raise ArticleError("internally_resolved_obligation_cannot_gain_external_reference")
    canonical = policy.canonical_ids(ledger)[obligation_id]
    members = policy.groups(ledger)[canonical]["member_ids"]
    for oid in members:
        row = ledger["obligations"][oid]
        if reference_id not in row["reference_ids"]:
            row["reference_ids"].append(reference_id)


def seed(evidence: dict) -> dict:
    ledger = {"format": FORMAT, "evidence_digest": evidence["digest"], "references": {},
              "obligations": {}, "sources": {}, "checks": {}, "exclusions": {},
              "searches": [], "errors": [], "discovery_complete": False,
              "stop_reason": "not_started", "unresolved_candidates": []}
    for fact in evidence["facts"]:
        for index, metadata in enumerate(fact["external_refs"]):
            alias = fact["fact_id"] + ":" + str(metadata.get("key", index))
            try:
                rid = add_reference(ledger, metadata, alias=alias)
            except ArticleError as exc:
                ledger["errors"].append({"kind": str(exc), "alias": alias})
                continue
            oid = "obligation-" + digest(alias)[:20]
            ledger["obligations"][oid] = {"id": oid, "description": str(metadata.get("cited_for") or
                "Reference declared by the verified proof; identify its precise use or justify exclusion."),
                "source_units": [fact["fact_id"] + ":statement"] + [
                    fact["fact_id"] + ":proof:" + span["id"] for span in fact["ranges"]], "reference_ids": [rid],
                "kind": "mathematical", "origin": "verified_fact", "critical": True}
    return ledger


def add_obligations(ledger: dict, rows: list, evidence: dict) -> None:
    if not isinstance(rows, list):
        raise ArticleError("citation_obligations_must_be_list")
    known = units(evidence)
    for row in rows:
        require_keys(row, {"id", "description", "source_units", "reference_ids", "kind", "critical"})
        nonempty_text(row["id"])
        nonempty_text(row["description"])
        string_list(row["source_units"])
        string_list(row["reference_ids"])
        if (set(row["source_units"]) - known.keys() or set(row["reference_ids"]) - ledger["references"].keys()
                or row["kind"] not in ("mathematical", "historical", "comparison", "background")
                or type(row["critical"]) is not bool):
            raise ArticleError("invalid_citation_obligation")
        if row["id"] in ledger["obligations"]:
            raise ArticleError("citation_obligation_id_already_exists")
        ledger["obligations"][row["id"]] = dict(row, origin="discovery")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if not self.skip and tag in ("p", "div", "section", "h1", "h2", "h3", "li", "br", "tr"):
            self.parts.append("\n")
        if not self.skip and tag == "math":
            values = dict(attrs)
            if values.get("alttext"):
                self.parts.append(" " + values["alttext"] + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def source_error(exc: Exception) -> str:
    if isinstance(exc, ArticleError) and re.fullmatch(r"[a-z][a-z0-9_]{0,159}", str(exc)):
        return str(exc)
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "source_timeout"
    if isinstance(exc, UnicodeError):
        return "source_text_decoding_failed"
    if isinstance(exc, ssl.SSLError):
        return "source_tls_error"
    if isinstance(exc, OSError):
        return "source_network_or_file_error"
    return "source_response_or_extraction_invalid"


def is_pdf(prefix: bytes, content_type: str) -> bool:
    return prefix.startswith(b"%PDF") or "application/pdf" in content_type.lower()


def extract_text(raw: bytes, content_type: str, max_bytes: int | None = None) -> str:
    if max_bytes is not None and len(raw) > max_bytes:
        raise ArticleError("source_too_large")
    if is_pdf(raw, content_type):
        from .pdf_sources import extract
        return extract(raw, max_bytes)
    if raw.startswith(b"\x1f\x8b"):
        expanded_limit = max_bytes if max_bytes is not None else DEFAULT_SOURCE_TEXT_BYTES
        pieces, total = [], 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as archive:
            for index, member in enumerate(archive):
                if (index >= 64 or member.size > expanded_limit or member.issym() or member.islnk()
                        or Path(member.name).is_absolute() or ".." in Path(member.name).parts):
                    raise ArticleError("unsafe_source_archive")
                total += member.size
                if total > expanded_limit:
                    raise ArticleError("source_archive_too_large")
                if member.isfile() and member.name.endswith((".tex", ".bbl")):
                    handle = archive.extractfile(member)
                    if handle is not None:
                        pieces.append("\nSOURCE MEMBER: " + member.name + "\n" + handle.read().decode("utf-8"))
        if not pieces:
            raise ArticleError("source_archive_has_no_readable_text")
        return "\n".join(pieces)
    text = raw.decode("utf-8-sig")
    if "html" in content_type.lower() or text.lstrip().lower().startswith(("<!doctype html", "<html")):
        parser = _HTMLText()
        parser.feed(text)
        text = "".join(parser.parts)
    if not text.strip() or "\x00" in text:
        raise ArticleError("source_is_not_text")
    return text


def _public_addresses(host: str) -> list[str]:
    addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
    if not addresses or any(not ipaddress.ip_address(addr).is_global for addr in addresses):
        raise ArticleError("nonpublic_source_address")
    return addresses


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, address: list[str], timeout: float):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self.addresses = address

    def connect(self):
        deadline = time.monotonic() + self.timeout
        failure = None
        for index, address in enumerate(self.addresses):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            raw = None
            try:
                budget = remaining / (len(self.addresses) - index)
                raw = socket.create_connection((address, 443), budget)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("source_connect_timeout")
                raw.settimeout(remaining)
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                return
            except OSError as exc:
                failure = exc
                if raw is not None:
                    raw.close()
        raise failure or TimeoutError("source_connect_timeout")


class SourceReader:
    """Only this trusted adapter accesses sources; model sessions have no tools."""
    def __init__(self, config: ArticleConfig, allowed: bool, *, tick=lambda: None,
                 deadline: float | None = None, cancelled=None):
        self.config, self.allowed, self.tick = config, allowed, tick
        self.deadline, self.cancelled = deadline, cancelled
        self.requests = 0

    def _budget(self):
        return Budget(self.config.source_timeout, "source_timeout",
                      deadline=self.deadline, cancelled=self.cancelled)

    def remaining(self) -> float:
        return self._budget().remaining()

    def search_arxiv(self, query: str, num_results: int) -> dict:
        """Use the shared arXiv cache and throttle with article-bound network I/O."""
        self.tick()
        budget = Budget(min(120, self.config.source_timeout), "source_timeout",
                        deadline=self.deadline, cancelled=self.cancelled)

        def pace(seconds):
            try:
                budget.wait(seconds)
            except OperationTimeout as exc:
                raise TimeoutError("source_timeout") from exc

        def fetch(request, limit):
            try:
                with self._response(request.full_url, budget=budget, headers=dict(request.header_items()),
                                    count_source=False, preserve_http_error=True) as (response, _, _, _):
                    return response.read(limit)
            except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
                raise
            except OperationTimeout as exc:
                raise TimeoutError("source_timeout") from exc
            except ArticleError as exc:
                raise urllib.error.URLError(exc) from exc

        return arxiv.search(query, num_results=num_results, timeout=budget.remaining(), fetch=fetch, pace=pace)

    def get(self, url: str) -> tuple[bytes, str, str]:
        """Download data that callers need in memory, such as a Crossref response."""
        with self._response(url) as (response, update_socket, _budget, final):
            raw = io.BytesIO()
            for block in self._chunks(response, update_socket):
                raw.write(block)
            return raw.getvalue(), response.getheader("Content-Type", ""), final

    def get_text(self, url: str) -> tuple[str, str]:
        """Extract remote source text, staging PDFs outside process memory."""
        with self._response(url) as (response, update_socket, budget, final):
            blocks = self._chunks(response, update_socket)
            first = next(blocks, b"")
            content_type = response.getheader("Content-Type", "")
            if is_pdf(first, content_type):
                from .pdf_sources import extract_chunks
                return extract_chunks(chain((first,), blocks), budget=budget), final
            raw = io.BytesIO(first)
            for block in blocks:
                raw.write(block)
            return extract_text(raw.getvalue(), content_type, self.config.max_source_bytes), final

    def _chunks(self, response, update_socket):
        max_bytes = self.config.max_source_bytes
        received = 0
        while True:
            self.tick()
            update_socket()
            size = 65536 if max_bytes is None else min(65536, max_bytes + 1 - received)
            block = response.read(size)
            if not block:
                return
            received += len(block)
            if max_bytes is not None and received > max_bytes:
                raise ArticleError("source_too_large")
            yield block

    @contextmanager
    def _response(self, url: str, *, budget: Budget | None = None, headers=None, count_source: bool = True,
                  preserve_http_error: bool = False):
        if not self.allowed:
            raise ArticleError("literature_search_disabled_for_this_run")
        if count_source:
            if self.requests >= self.config.max_sources:
                raise ArticleError("article_source_budget_exhausted")
            self.requests += 1
        if budget is None:
            budget = self._budget()
        request_headers = {"User-Agent": "hyprover-article/1"} if headers is None else dict(headers)
        request_headers["Accept-Encoding"] = "identity"
        for _ in range(4):
            self.tick()
            normalized = source_url(url)
            if not normalized:
                raise ArticleError("source_url_not_allowed")
            url = normalized
            parsed = urlsplit(url)
            if parsed.hostname not in self.config.allowed_source_hosts:
                raise ArticleError("source_url_not_allowed")
            address = budget.resolve(_public_addresses, parsed.hostname)
            conn = _PinnedHTTPS(parsed.hostname, address, budget.remaining())
            try:
                with budget.watch_http(conn) as update_socket:
                    update_socket()
                    conn.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""),
                                 headers=request_headers)
                    update_socket()
                    response = conn.getresponse()
                    update_socket()
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.getheader("Location")
                        if not location:
                            raise ArticleError("invalid_source_redirect")
                        url = urljoin(url, location)
                        continue
                    if response.status != 200:
                        if preserve_http_error:
                            raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
                        raise ArticleError("source_http_" + str(response.status))
                    yield response, update_socket, budget, url
                    return
            finally:
                conn.close()
        raise ArticleError("too_many_source_redirects")


def source_kind(text: str, origin: str) -> str:
    parsed = urlsplit(origin)
    if (parsed.hostname in ("api.crossref.org", "export.arxiv.org") and "/api/" in parsed.path
            or parsed.hostname == "api.crossref.org" or parsed.path.startswith("/abs/")
            or re.search(r"/(?:abstract|abs)(?:/|$)", parsed.path, re.I)):
        return "metadata"
    stripped = text.lstrip()
    if stripped.startswith(("{", "[", "<feed", "<?xml")):
        return "metadata"
    if re.search(r"(?:^|\n)\s*abstract\b", text, re.I) and not re.search(
            r"(?:^|\n)\s*(?:1[. ]+Introduction|Theorem\s+\d|Lemma\s+\d|Proof[.:])|\\\\begin\{(?:theorem|proof)\}", text, re.I):
        return "metadata"
    return "unconfirmed"


def source_span(text: str, quotation: str) -> dict:
    nonempty_text(quotation)
    start = text.find(quotation)
    if start >= 0:
        return {"start": start, "end": start + len(quotation), "match": "exact"}
    parts = quotation.split()
    if not parts:
        raise ArticleError("source_quote_not_found")
    match = re.search(r"\s+".join(re.escape(part) for part in parts), text)
    if match is None:
        raise ArticleError("source_quote_not_found")
    return {"start": match.start(), "end": match.end(), "match": "whitespace"}


def source_classification(source: dict, rid: str) -> dict | None:
    report = source.get("classifications", {}).get(rid, source.get("classification"))
    if (isinstance(report, dict) and report.get("reference_id") == rid
            and report.get("source_id") == source["id"] and report.get("source_sha256") == source["sha256"]):
        return report
    return None


def source_is_fulltext(ledger: dict, sid: str, rid: str) -> bool:
    source = ledger["sources"][sid]
    if source["kind"] != "fulltext" or sid not in ledger["references"][rid]["source_ids"]:
        return False
    report = source_classification(source, rid)
    if report is not None:
        return (report.get("document_type") == "fulltext" and report.get("identity_supported") is True
                and report.get("outside_abstract") is True)
    if source.get("classification") or source.get("classifications"):
        return False
    if "fulltext_reference_id" in source:
        return source["fulltext_reference_id"] == rid
    # Older trusted preclassified sources can only be reused if they have one owner.
    return not any(other != rid and sid in ref["source_ids"] for other, ref in ledger["references"].items())


def confirm_source(ledger: dict, sid: str, rid: str, report: dict, *, start: int = 0, end: int | None = None) -> None:
    require_keys(report, {"source_id", "source_sha256", "reference_id", "document_type", "identity_supported",
                          "body_quote", "locator", "outside_abstract"})
    source = ledger["sources"][sid]
    if (report["source_id"] != sid or report["reference_id"] != rid or rid not in ledger["references"]
            or sid not in ledger["references"][rid]["source_ids"]
            or report["source_sha256"] != source["sha256"] or source["kind"] == "metadata"
            or source_kind(source["text"], source["origin"]) == "metadata"):
        raise ArticleError("source_identity_or_type_mismatch")
    if (report["document_type"] not in ("fulltext", "abstract", "metadata", "unknown")
            or type(report["identity_supported"]) is not bool or type(report["outside_abstract"]) is not bool):
        raise ArticleError("invalid_source_classification")
    if end is None:
        end = len(source["text"])
    if not 0 <= start < end <= len(source["text"]):
        raise ArticleError("invalid_source_classification_range")
    classified = dict(source.get("classifications", {}))
    old = source.get("classification")
    if old and old.get("reference_id"):
        classified.setdefault(old["reference_id"], old)
    if report["document_type"] != "fulltext" or not report["identity_supported"] or not report["outside_abstract"]:
        if not source_is_fulltext(ledger, sid, rid):
            classified[rid] = dict(report)
            source["classifications"] = classified
            if source["kind"] != "fulltext":
                source["classification"] = classified[rid]
        return
    try:
        spans = {key: source_span(source["text"][start:end], report[key]) for key in ("body_quote", "locator")}
    except ArticleError as exc:
        raise ArticleError("source_classification_has_no_body_evidence") from exc
    canonical = dict(report)
    for key, span in spans.items():
        span["start"] += start
        span["end"] += start
        canonical[key] = source["text"][span["start"]:span["end"]]
    classified[rid] = canonical
    source.update(kind="fulltext", classification=canonical, classification_original=dict(report),
                  classification_spans=spans, classifications=classified)


def add_source(ledger: dict, rid: str, text: str, *, origin: str, kind: str, version: str = "") -> str:
    if rid not in ledger["references"] or kind not in ("metadata", "fulltext", "unconfirmed"):
        raise ArticleError("invalid_reference_source")
    sid = "source-" + digest([origin, text_hash(text), kind, version])[:20]
    if kind == "fulltext" and sid not in ledger["references"][rid]["source_ids"]:
        sid = "source-" + digest([rid, origin, text_hash(text), kind, version])[:20]
    if sid not in ledger["sources"]:
        def serialized_text_size(value: str) -> int:
            return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        previous_bytes = sum(serialized_text_size(s["text"]) for s in ledger["sources"].values())
        if previous_bytes + serialized_text_size(text) > _MAX_TOTAL_SOURCE_BYTES:
            raise ArticleError("article_total_source_budget_exhausted")
    ledger["sources"].setdefault(sid, {"id": sid, "origin": origin, "kind": kind,
        "version": version, "text": text, "sha256": text_hash(text), "retrieved_at": utc_now()})
    if kind == "fulltext" and "fulltext_reference_id" not in ledger["sources"][sid]:
        peers = any(other != rid and sid in row["source_ids"] for other, row in ledger["references"].items())
        if not peers:
            ledger["sources"][sid]["fulltext_reference_id"] = rid
    if sid not in ledger["references"][rid]["source_ids"]:
        ledger["references"][rid]["source_ids"].append(sid)
    return sid


def _metadata_match(declared: dict, found: dict) -> bool:
    if not found.get("title") or not found.get("authors") or not found.get("year"):
        return False
    if any(declared.get(k) and normal(declared[k]) != normal(found.get(k, "")) for k in ("title", "year", "edition")):
        return False
    authors = declared.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in re.split(r";| and ", authors)]
    actual = " ".join(found["authors"])
    return all(normal(str(a).split()[-1]) in normal(actual) for a in authors if str(a).split())


def corroborate(ledger: dict, rid: str, found: dict, sid: str) -> None:
    row = ledger["references"][rid]
    if not _metadata_match(row["declared"], found):
        row["conflicts"].append({"kind": "metadata_mismatch_or_incomplete", "source_id": sid})
        return
    row["metadata"] = found
    row["metadata_verified"] = True
    if sid not in row["metadata_sources"]:
        row["metadata_sources"].append(sid)


def crossref_metadata(message: dict) -> dict:
    date = next((message[k].get("date-parts", [[]])[0] for k in
                 ("published-print", "published-online", "published", "issued") if message.get(k)), [])
    return {"title": (message.get("title") or [""])[0],
        "authors": [" ".join(filter(None, (a.get("given"), a.get("family")))) for a in message.get("author", [])],
        "year": date[0] if date else None, "doi": message.get("DOI", ""),
        "venue": (message.get("container-title") or [""])[0], "volume": message.get("volume", ""),
        "number": message.get("issue", ""), "pages": message.get("page", ""),
        "publisher": message.get("publisher", ""), "type": message.get("type", "article")}


def merge_crossref_source_urls(row: dict, item: dict) -> None:
    link = item.get("link")
    links = [entry.get("URL") for entry in link if isinstance(entry, dict)] if isinstance(link, list) else []
    resource = item.get("resource")
    primary = resource.get("primary") if isinstance(resource, dict) else None
    merge_source_urls(row, [*links, primary.get("URL") if isinstance(primary, dict) else None])


def _recover_crossref_urls(ledger: dict, row: dict) -> None:
    """Reuse links in DOI metadata captured by older runs without refetching it."""
    for sid in row["metadata_sources"]:
        source = ledger["sources"][sid]
        if urlsplit(source["origin"]).hostname != "api.crossref.org":
            continue
        try:
            payload = json.loads(source["text"])
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(item := payload.get("message", payload), dict):
            merge_crossref_source_urls(row, item)


def retrieve(ledger: dict, reader: SourceReader, *, queries: list[str] | None = None,
             supplied: list[dict] | None = None, project: Path | None = None,
             search=None, checkpoint=lambda: None, reference_ids: set[str] | None = None,
             fulltext_ids: set[str] | None = None, supplement_ids: set[str] | None = None,
             classify: Callable[[str, str], None] | None = None) -> None:
    """Keep unavailable sources explicit; never promote search abstracts to evidence."""
    queries = string_list(queries or [])
    if not reader.allowed:
        ledger["stop_reason"] = "literature_search_disabled_for_this_run"
        return

    def search_metadata(query, num_results):
        if search is None:
            return reader.search_arxiv(query, num_results)
        return search(query, num_results=num_results, timeout=min(120, reader.remaining()))

    from . import literature
    queries = literature.topical_queries(queries, ledger)
    done_queries = {s["query"] for s in ledger["searches"]}
    for rid, row in list(ledger["references"].items()):
        if reference_ids is not None and rid not in reference_ids:
            continue
        doi, aid = identifiers(row["declared"])
        if not doi and not aid:
            literature.lookup(ledger, rid, reader, checkpoint)
    for query in queries:
        reader.tick()
        if query in done_queries:
            continue
        if len(ledger["searches"]) >= reader.config.max_queries:
            ledger["unresolved_candidates"].append({"query": query, "reason": "query_budget"})
            continue
        result = search_metadata(query, 5)
        reader.tick()
        ledger["searches"].append({"query": query, "result": result, "retrieved_at": utc_now()})
        done_queries.add(query)
        checkpoint()
    for rid, row in list(ledger["references"].items()):
        if reference_ids is not None and rid not in reference_ids:
            continue
        reader.tick()
        doi, aid = identifiers(row["declared"])
        if not row["metadata_verified"] and doi:
            try:
                url = "https://api.crossref.org/works/" + quote(doi, safe="")
                raw, _, final = reader.get(url)
                item = json.loads(raw)["message"]
                found = crossref_metadata(item)
                if str(found["doi"]).lower() != doi:
                    raise ArticleError("doi_response_identifier_mismatch")
                sid = add_source(ledger, rid, raw.decode("utf-8"), origin=final, kind="metadata")
                corroborate(ledger, rid, found, sid)
            except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
                raise
            except (ValueError, OSError, KeyError, TypeError, http.client.HTTPException) as exc:
                ledger["errors"].append({"reference_id": rid, "kind": "doi_unavailable", "error_type": type(exc).__name__,
                                         "error": source_error(exc)})
        if aid and not row["metadata_verified"]:
            query = "id:" + aid
            cached = next((s["result"] for s in ledger["searches"] if s["query"] == query), None)
            known = [r for s in ledger["searches"] for r in s["result"].get("results", [])
                     if r.get("arxiv_id") == aid or (not re.search(r"v\d+$", aid)
                         and re.sub(r"v\d+$", "", r.get("arxiv_id", "")) == aid)]
            if known:
                cached = {"results": known}
            if cached is None and reader.requests < reader.config.max_sources:
                reader.requests += 1
                checkpoint()
                cached = search_metadata(query, 1)
                reader.tick()
                ledger["searches"].append({"query": query, "result": cached, "retrieved_at": utc_now(), "purpose": "identity"})
            if cached:
                for result in cached.get("results", []):
                    actual = result.get("arxiv_id", "")
                    if (re.sub(r"v\d+$", "", actual) != re.sub(r"v\d+$", "", aid)
                            or (re.search(r"v\d+$", aid) and actual != aid)):
                        continue
                    found = {"title": result["title"], "authors": result["authors"],
                             "year": result["published"][:4], "arxiv": actual, "type": "misc", "venue": "arXiv"}
                    sid = add_source(ledger, rid, json.dumps(result, ensure_ascii=False),
                                     origin=result["url"], kind="metadata", version=actual)
                    corroborate(ledger, rid, found, sid)
        if fulltext_ids is not None and rid not in fulltext_ids:
            checkpoint()
            continue
        supplement = rid in (supplement_ids or set())
        if classify:
            for sid in row["source_ids"]:
                if ledger["sources"][sid]["kind"] != "metadata" and not source_is_fulltext(ledger, sid, rid):
                    classify(rid, sid)
        if aid and (supplement or not any(source_is_fulltext(ledger, s, rid) for s in row["source_ids"])):
            version = row["metadata"].get("arxiv", aid)
            for endpoint in ("html", "src", "pdf"):
                url = "https://arxiv.org/" + endpoint + "/" + version
                requests = ledger.setdefault("retrieval_requests", {})
                key = digest([rid, "fulltext", url])
                previous = requests.get(key, {})
                if (previous.get("state") == "complete" or previous.get("attempts", 0) >= 2
                        or any(ledger["sources"][sid]["origin"] == url for sid in row["source_ids"])):
                    continue
                if reader.requests >= reader.config.max_sources:
                    break
                record = {"reference_id": rid, "url": url, "attempts": previous.get("attempts", 0) + 1, "state": "pending"}
                requests[key] = record
                checkpoint()
                try:
                    text, final = reader.get_text(url)
                    sid = add_source(ledger, rid, text, origin=final, kind=source_kind(text, final), version=version)
                    record.update(state="complete", source_id=sid)
                except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
                    raise
                except (ValueError, OSError, http.client.HTTPException, tarfile.TarError) as exc:
                    record.update(state="unavailable", error_type=type(exc).__name__, error=source_error(exc))
                    ledger["errors"].append({"reference_id": rid, "kind": "fulltext_unavailable", "error_type": type(exc).__name__,
                                             "error": source_error(exc)})
                    continue
                finally:
                    checkpoint()
                if classify and ledger["sources"][sid]["kind"] != "metadata" and not source_is_fulltext(ledger, sid, rid):
                    classify(rid, sid)
                if not supplement and source_is_fulltext(ledger, sid, rid):
                    break
        if row["metadata_verified"]:
            _recover_crossref_urls(ledger, row)
        literature.publisher_sources(ledger, rid, reader, checkpoint, supplement=supplement, classify=classify)
        checkpoint()
    for spec in supplied or []:
        reader.tick()
        require_keys(spec, {"reference"}, {"path", "url", "version"})
        rid = spec["reference"]
        if rid not in ledger["references"]:
            matches = [r for r, v in ledger["references"].items() if any(
                a == rid or a.rsplit(":", 1)[-1] == rid for a in v["aliases"])]
            if len(matches) != 1:
                raise ArticleError("ambiguous_or_unknown_supplied_reference")
            rid = matches[0]
        if ("path" in spec) == ("url" in spec):
            raise ArticleError("source_requires_exactly_one_location")
        try:
            if "path" in spec:
                if project is None or Path(spec["path"]).parts[:1] != ("literature",):
                    raise ArticleError("local_sources_must_be_inside_project_literature")
                path = safe_path(project, spec["path"])
                text = read_text(path, reader.config.local_source_max_bytes)
                text = extract_text(text.encode(), "text/html" if path.suffix in (".htm", ".html") else "text/plain",
                                    reader.config.local_source_max_bytes)
                origin = "project:" + spec["path"]
            else:
                text, origin = reader.get_text(spec["url"])
            sid = add_source(ledger, rid, text, origin=origin, kind=source_kind(text, origin), version=spec.get("version", ""))
            declared = ledger["references"][rid]["declared"]
            authors = declared.get("authors", [])
            if (isinstance(authors, list) and authors and declared.get("year")
                    and normal(declared["title"]) in normal(text)
                    and all(normal(a) in normal(text) for a in authors)
                    and str(declared["year"]) in text):
                found = {key: declared[key] for key in ("title", "authors", "year")}
                for key in ("doi", "arxiv", "edition", "venue", "publisher", "address", "series"):
                    if declared.get(key) and str(declared[key]) in text:
                        found[key] = declared[key]
                corroborate(ledger, rid, found, sid)
        except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
            raise
        except (ValueError, OSError, http.client.HTTPException, tarfile.TarError) as exc:
            ledger["errors"].append({"reference_id": rid, "kind": "supplied_source_unavailable", "error_type": type(exc).__name__,
                                     "error": source_error(exc)})
        checkpoint()
    ledger["stop_reason"] = "bounded_search_completed" if not ledger["unresolved_candidates"] else "budget_or_unresolved_candidates"


def apply_checks(ledger: dict, rows: list[dict]) -> None:
    if not isinstance(rows, list):
        raise ArticleError("citation_checks_must_be_list")
    for row in rows:
        require_keys(row, {"obligation_id", "reference_id", "source_id", "quote", "locator", "premise_comparison", "supported"})
        oid, rid, sid = row["obligation_id"], row["reference_id"], row["source_id"]
        if oid not in ledger["obligations"] or rid not in ledger["references"] or sid not in ledger["sources"]:
            raise ArticleError("unknown_citation_evidence")
        source, ref = ledger["sources"][sid], ledger["references"][rid]
        if (rid not in ledger["obligations"][oid]["reference_ids"] or sid not in ref["source_ids"]
                or not source_is_fulltext(ledger, sid, rid) or source["sha256"] != text_hash(source["text"])
                or type(row["supported"]) is not bool):
            raise ArticleError("invalid_citation_evidence")
        for key in ("quote", "locator", "premise_comparison"):
            nonempty_text(row[key])
        try:
            spans = {key: source_span(source["text"], row[key]) for key in ("quote", "locator")}
        except ArticleError as exc:
            raise ArticleError("citation_quote_or_locator_not_in_source") from exc
        canonical = dict(row)
        for key, span in spans.items():
            canonical[key] = source["text"][span["start"]:span["end"]]
        cid = "cite-" + digest(canonical)[:20]
        ledger["checks"][cid] = dict(canonical, id=cid, source_sha256=source["sha256"],
                                    start=spans["quote"]["start"], end=spans["quote"]["end"],
                                    original_quote=row["quote"], original_locator=row["locator"], matching=spans)


def apply_standard_checks(ledger: dict, rows: list[dict]) -> None:
    if not isinstance(rows, list):
        raise ArticleError("standard_reference_checks_must_be_list")
    target, ledger = ledger, copy.deepcopy(ledger)
    grouped, canonical = policy.groups(ledger), policy.canonical_ids(ledger)
    seen = set()
    for row in rows:
        require_keys(row, {"obligation_id", "reference_id", "rationale", "standard_result", "supported"})
        oid, rid = row["obligation_id"], row["reference_id"]
        if not isinstance(oid, str) or not isinstance(rid, str) or oid not in canonical or rid not in ledger["references"]:
            raise ArticleError("unknown_standard_reference")
        obligation, reference = grouped[canonical[oid]], ledger["references"][rid]
        if type(row["standard_result"]) is not bool or type(row["supported"]) is not bool:
            raise ArticleError("standard_reference_flags_must_be_boolean")
        if policy.essential(obligation):
            raise ArticleError("standard_reference_cannot_discharge_essential_result")
        if obligation["kind"] != "mathematical" or rid not in obligation["reference_ids"]:
            raise ArticleError("standard_reference_check_out_of_scope")
        key = (canonical[oid], rid)
        if key in seen:
            raise ArticleError("duplicate_standard_reference_check")
        seen.add(key)
        nonempty_text(row["rationale"])
        ledger.setdefault("standard_assessments", {})[digest(row)] = dict(row, policy_revision=policy.POLICY_REVISION)
        if not row["standard_result"]:
            for member in obligation["member_ids"]:
                ledger["obligations"][member]["critical"] = True
            obligation["critical"] = True
            continue
        if not row["supported"]:
            continue
        if not reference["metadata_verified"] or reference["conflicts"] or not reference["metadata_sources"]:
            raise ArticleError("standard_reference_metadata_not_verified")
        sid = reference["metadata_sources"][0]
        source = ledger["sources"][sid]
        cid = "cite-" + digest(["standard", row, source["sha256"]])[:20]
        ledger["checks"][cid] = {"id": cid, "obligation_id": oid, "reference_id": rid, "source_id": sid,
            "source_sha256": source["sha256"], "evidence_level": "standard_reference", "quote": "", "locator": "",
            "start": 0, "end": 0, "premise_comparison": row["rationale"], "supported": True,
            "standard_result": True, "policy_revision": policy.POLICY_REVISION}
    for key in ("obligations", "checks", "standard_assessments"):
        if key in ledger:
            target[key] = ledger[key]


def apply_background_checks(ledger: dict, rows: list[dict]) -> None:
    if not isinstance(rows, list):
        raise ArticleError("background_checks_must_be_array")
    for row in rows:
        require_keys(row, {"obligation_id", "reference_id", "rationale", "supported"})
        obligation = ledger["obligations"][row["obligation_id"]]
        ref = ledger["references"][row["reference_id"]]
        if (obligation["kind"] != "background" or row["reference_id"] not in obligation["reference_ids"]
                or type(row["supported"]) is not bool):
            raise ArticleError("background_citation_cannot_support_theorem_or_priority")
        nonempty_text(row["rationale"])
        if not row["supported"]:
            continue
        if not ref["metadata_verified"] or ref["conflicts"] or not ref["metadata_sources"]:
            raise ArticleError("background_metadata_not_verified")
        sid = ref["metadata_sources"][0]
        cid = "cite-" + digest(["background", row, ledger["sources"][sid]["sha256"]])[:20]
        ledger["checks"][cid] = {"id": cid, "obligation_id": row["obligation_id"], "reference_id": row["reference_id"],
            "source_id": sid, "source_sha256": ledger["sources"][sid]["sha256"], "evidence_level": "background_reference",
            "quote": "", "locator": "", "start": 0, "end": 0, "premise_comparison": row["rationale"],
            "supported": True, "policy_revision": policy.POLICY_REVISION}


def usable_check(ledger: dict, cid: str) -> bool:
    row = ledger["checks"].get(cid)
    if not row:
        return False
    ref = ledger["references"].get(row.get("reference_id"))
    source = ledger["sources"].get(row.get("source_id"))
    obligation = ledger["obligations"].get(row.get("obligation_id"))
    if (not ref or not source or not obligation or row.get("supported") is not True
            or ref.get("metadata_verified") is not True or ref.get("conflicts")
            or row.get("reference_id") not in obligation["reference_ids"]
            or row.get("source_id") not in ref["source_ids"]
            or row.get("source_sha256") != text_hash(source["text"])):
        return False
    if row.get("evidence_level") == "background_reference":
        return (obligation["kind"] == "background" and row.get("policy_revision") == policy.POLICY_REVISION
                and row["source_id"] in ref["metadata_sources"] and bool(row.get("premise_comparison"))
                and row.get("quote") == "" and row.get("locator") == "")
    if row.get("evidence_level") == "standard_reference":
        group = policy.groups(ledger)[policy.canonical_ids(ledger)[row["obligation_id"]]]
        return (not policy.essential(group) and row.get("standard_result") is True
                and row.get("policy_revision") == policy.POLICY_REVISION
                and row.get("source_id") in ref["metadata_sources"]
                and bool(row.get("premise_comparison")) and row.get("quote") == "" and row.get("locator") == "")
    return (source_is_fulltext(ledger, row["source_id"], row["reference_id"])
            and bool(row.get("quote")) and bool(row.get("locator"))
            and row["locator"] in source["text"]
            and source["text"][row["start"]:row["end"]] == row["quote"])


def apply_internal_resolution(ledger: dict, evidence: dict, report: dict) -> None:
    require_keys(report, {"obligation_id", "supported_internally", "proof_units", "reason"})
    oid = report["obligation_id"]
    if (oid not in ledger["obligations"] or ledger["obligations"][oid]["reference_ids"]
            or ledger["obligations"][oid]["kind"] != "mathematical"):
        raise ArticleError("external_reference_cannot_be_excluded_as_internal")
    if type(report["supported_internally"]) is not bool:
        raise ArticleError("invalid_internal_citation_resolution")
    if not report["supported_internally"]:
        return
    nonempty_text(report["reason"])
    known = units(evidence)
    for uid in string_list(report["proof_units"], nonempty=True):
        if uid not in known or known[uid]["kind"] not in ("proof", "bridge"):
            raise ArticleError("internal_resolution_requires_actual_proof_units")
    ledger.setdefault("internal_resolutions", {})[oid] = dict(report,
        evidence_digest=evidence["digest"], policy_revision=policy.POLICY_REVISION)


def optional_context_candidate(ledger: dict, row: dict, used: set[str] | None) -> bool:
    """Advisory-only leads: no essential dependency and no identity/evidence failure.

    An unresolved lead may carry the selection agent's generic critical flag even when no
    critical obligation depends on it. Downgrade only that bookkeeping uncertainty to a
    visible warning. Critical/historical/comparison obligations, verified-fact substitutions,
    metadata conflicts and failed evidence tasks remain blocking.
    """
    if (row.get("reason") != "focused_selection_uncertain"
            or any(key in row for key in ("stage", "error", "failures", "material_digest"))):
        return False
    rid = row.get("reference_id")
    reference = ledger["references"].get(rid)
    if not reference or reference["conflicts"]:
        return False
    related = [o for o in policy.groups(ledger).values() if rid in o["reference_ids"]]
    if any(policy.essential(o) or o.get("origin") == "verified_fact" for o in related):
        return False
    if used is not None:
        candidate_metadata = reference["metadata"] or reference["declared"]
        candidate_doi, candidate_arxiv = identifiers(candidate_metadata)
        for cid in used:
            check = ledger["checks"].get(cid)
            if not check or check["reference_id"] not in ledger["references"]:
                return False
            other = ledger["references"][check["reference_id"]]
            doi, arxiv = identifiers(other["metadata"])
            matches = (check["reference_id"] == rid or (candidate_doi and doi == candidate_doi)
                       or (candidate_arxiv and arxiv == candidate_arxiv)
                       or normal(other["metadata"].get("title")) == normal(candidate_metadata.get("title")))
            if matches and not usable_check(ledger, cid):
                return False
    return True


def coverage(ledger: dict, used: set[str] | None = None) -> dict:
    grouped, canonical = policy.groups(ledger), policy.canonical_ids(ledger)
    valid = {cid for cid in ledger["checks"] if usable_check(ledger, cid)}
    if used is not None:
        valid &= used
    covered = {canonical[ledger["checks"][cid]["obligation_id"]] for cid in valid}
    excluded = {canonical[oid] for oid, reason in ledger["exclusions"].items() if oid in canonical
                and not policy.essential(grouped[canonical[oid]]) and isinstance(reason, str) and reason.strip()}
    internal = {canonical[oid] for oid, row in ledger.get("internal_resolutions", {}).items()
                if oid in canonical and not grouped[canonical[oid]]["reference_ids"]
                and row.get("policy_revision") == policy.POLICY_REVISION
                and row.get("evidence_digest") == ledger["evidence_digest"]
                and row.get("supported_internally") is True and row.get("proof_units") and row.get("reason")}
    missing = sorted(set(grouped) - covered - excluded - internal)
    blocking = [oid for oid in missing if policy.essential(grouped[oid])]
    advisory = [oid for oid in missing if not policy.essential(grouped[oid])]
    candidates = ledger["unresolved_candidates"]
    optional = [r for r in candidates if optional_context_candidate(ledger, r, used)]
    advisory_candidates = [dict(r, assessment="advisory_only_no_essential_dependency_or_conflict", blocking=False)
                           for r in optional]
    blocking_candidates = [r for r in candidates if r.get("reason") != "query_budget"
                           and r.get("critical", True) is not False and r not in optional]
    focus = ledger.get("focus")
    if focus and (not focus.get("selection_done") or focus.get("search_complete") is not True):
        blocking_candidates = [*blocking_candidates, {"reason": "focused_search_incomplete", "critical": True}]
    return {"policy_revision": policy.POLICY_REVISION, "identified": len(grouped),
            "raw_identified": len(ledger["obligations"]), "duplicates_collapsed": len(ledger["obligations"]) - len(grouped),
            "covered": len(covered), "excluded": sorted(excluded), "internally_supported": sorted(internal), "missing": missing,
            "blocking_missing": blocking, "advisory_missing": advisory,
            "complete": not blocking and not blocking_candidates and ledger["discovery_complete"] is True
                and not any("alias" in error for error in ledger["errors"]),
            "all_identified_resolved": not missing and not candidates,
            "discovery_complete": ledger["discovery_complete"], "unresolved_candidates": candidates,
            "blocking_candidates": blocking_candidates, "advisory_candidates": advisory_candidates,
            "scope": "identified_obligations_only",
            "stop_reason": ledger["stop_reason"], "retrieval_errors": ledger["errors"]}


def validate(ledger: dict, evidence: dict) -> None:
    require_keys(ledger, {"format", "evidence_digest", "references", "obligations", "sources", "checks", "exclusions",
                         "searches", "errors", "discovery_complete", "stop_reason", "unresolved_candidates"},
                 {"policy_revision", "internal_resolutions", "retrieval_requests", "standard_assessments", "focus"})
    if ledger["format"] != FORMAT or ledger["evidence_digest"] != evidence["digest"]:
        raise ArticleError("stale_reference_ledger")
    known = units(evidence)
    for oid, row in ledger.get("internal_resolutions", {}).items():
        if (row.get("obligation_id") != oid or oid not in ledger["obligations"]
                or ledger["obligations"][oid]["reference_ids"] or row.get("evidence_digest") != evidence["digest"]
                or row.get("policy_revision") != policy.POLICY_REVISION or row.get("supported_internally") is not True
                or not row.get("reason") or not row.get("proof_units")
                or any(uid not in known or known[uid]["kind"] not in ("proof", "bridge") for uid in row["proof_units"])):
            raise ArticleError("invalid_internal_citation_resolution")
    for sid, source in ledger["sources"].items():
        if source["id"] != sid or source["sha256"] != text_hash(source["text"]):
            raise ArticleError("reference_source_changed")
    for cid, row in ledger["checks"].items():
        if row["id"] != cid or row["obligation_id"] not in ledger["obligations"]:
            raise ArticleError("invalid_reference_check")
        if row["source_sha256"] != ledger["sources"][row["source_id"]]["sha256"]:
            raise ArticleError("reference_check_source_changed")


def supplied_sources(project: Path, relative: str | None) -> dict:
    if relative is None:
        return {"path": None, "sha256": None, "sources": []}
    text = read_text(safe_path(project, relative), 2 * 1024 * 1024)
    payload = json.loads(text)
    require_keys(payload, {"sources"})
    if not isinstance(payload["sources"], list):
        raise ArticleError("supplied_sources_must_be_list")
    return {"path": relative, "sha256": text_hash(text), "sources": payload["sources"]}


def assert_sources_current(ledger: dict, project: Path, supplied: dict) -> None:
    if supplied.get("path") and supplied_sources(project, supplied["path"])["sha256"] != supplied["sha256"]:
        raise ArticleError("supplied_source_manifest_changed")
    for source in ledger["sources"].values():
        if source["sha256"] != text_hash(source["text"]):
            raise ArticleError("reference_source_changed")
        if source["origin"].startswith("project:"):
            path = safe_path(project, source["origin"].removeprefix("project:"))
            raw = read_text(path, 8 * 1024 * 1024)
            current = extract_text(raw.encode(), "text/html" if path.suffix in (".htm", ".html") else "text/plain", 8 * 1024 * 1024)
            if text_hash(current) != source["sha256"]:
                raise ArticleError("local_reference_source_changed")
