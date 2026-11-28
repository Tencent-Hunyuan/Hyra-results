import io
import json
import tarfile

import pytest

from hyprover.article import evidence, references as refs
from hyprover.article.models import ArticleConfig, ArticleError, digest
from .conftest import publish


META = {"key": "sample", "authors": ["Ada Example"], "title": "A fixture monograph", "year": 2020,
        "cited_for": "The fixture identity"}
TEXT = "A fixture monograph\nAda Example\n2020\nTheorem 2.1. Under hypothesis H, the identity follows.\n"


_REAL_GET = refs.SourceReader.get


@pytest.fixture(autouse=True)
def offline_sources(monkeypatch):
    def offline(reader, url):
        if not reader.allowed:
            raise ArticleError("literature_search_disabled_for_this_run")
        raise OSError("offline fixture")
    monkeypatch.setattr(refs.SourceReader, "get", offline)
    monkeypatch.setattr(refs.SourceReader, "get_text", offline)


def ledger(project):
    fid = publish(project, refs=[META])
    frozen = evidence.capture(project.root, [fid])
    out = refs.seed(frozen)
    return frozen, out, next(iter(out["references"])), next(iter(out["obligations"]))


def test_supplied_metadata_and_abstract_cannot_be_promoted_to_fulltext(project):
    _, out, rid, oid = ledger(project)
    (project.root / "literature").mkdir()
    (project.root / "literature/metadata.json").write_text(json.dumps(dict(META, abstract="A claimed Theorem 1.")))
    refs.retrieve(out, refs.SourceReader(ArticleConfig(), True),
                  supplied=[{"reference": "sample", "path": "literature/metadata.json"}], project=project.root)
    source = next(iter(out["sources"].values()))
    assert source["kind"] == "metadata"
    with pytest.raises(ArticleError, match="source_identity_or_type_mismatch"):
        refs.confirm_source(out, source["id"], rid, {"source_id": source["id"], "source_sha256": source["sha256"],
            "reference_id": rid, "document_type": "fulltext", "identity_supported": True,
            "body_quote": "A claimed Theorem 1.", "locator": "Theorem 1", "outside_abstract": True})
    assert refs.source_kind("Abstract\nThe summary of a result.", "https://arxiv.org/abs/2301.12345") == "metadata"
    assert not refs.coverage(out)["complete"]


def test_unconfirmed_source_requires_original_body_and_identity(project):
    _, out, rid, _ = ledger(project)
    sid = refs.add_source(out, rid, TEXT, origin="project:literature/book.txt", kind="unconfirmed")
    report = {"source_id": sid, "source_sha256": out["sources"][sid]["sha256"], "reference_id": rid,
              "document_type": "fulltext", "identity_supported": True, "body_quote": "invented quotation",
              "locator": "Theorem 2.1", "outside_abstract": True}
    with pytest.raises(ArticleError, match="no_body_evidence"):
        refs.confirm_source(out, sid, rid, report)
    report["body_quote"] = "Under hypothesis H, the identity follows."
    refs.confirm_source(out, sid, rid, report)
    assert out["sources"][sid]["kind"] == "fulltext"


def test_legacy_source_confirmation_is_bound_to_the_reference(project):
    frozen, out, rid, oid = ledger(project)
    other_meta = {**META, "title": "Another fixture monograph"}
    other = refs.add_reference(out, other_meta)
    out["obligations"][oid]["reference_ids"].append(other)
    origin = "https://www.ams.org/shared"
    sid = refs.add_source(out, rid, TEXT, origin=origin, kind="unconfirmed")
    assert refs.add_source(out, other, TEXT, origin=origin, kind="unconfirmed") == sid
    refs.corroborate(out, rid, META, refs.add_source(out, rid, "First metadata", origin="fixture", kind="metadata"))
    refs.corroborate(out, other, other_meta, refs.add_source(out, other, "Other metadata", origin="fixture", kind="metadata"))
    source = out["sources"][sid]
    quote = "Under hypothesis H, the identity follows."
    classification = {"source_id": sid, "source_sha256": source["sha256"], "reference_id": rid,
                      "document_type": "fulltext", "identity_supported": True, "outside_abstract": True,
                      "body_quote": quote, "locator": "Theorem 2.1"}
    refs.confirm_source(out, sid, rid, classification)
    source.pop("classifications")  # load the older single-report representation
    row = {"obligation_id": oid, "reference_id": rid, "source_id": sid, "quote": quote, "locator": "Theorem 2.1",
           "premise_comparison": "Exact fixture conditions.", "supported": True}
    refs.apply_checks(out, [row])
    cid, = out["checks"]
    assert refs.usable_check(out, cid)
    with pytest.raises(ArticleError, match="invalid_citation_evidence"):
        refs.apply_checks(out, [dict(row, reference_id=other)])
    old_bad = dict(out["checks"][cid], id="saved-wrong-citation", reference_id=other)
    out["checks"][old_bad["id"]] = old_bad
    refs.validate(out, frozen)
    assert not refs.usable_check(out, old_bad["id"])
    refs.confirm_source(out, sid, other, dict(classification, reference_id=other, identity_supported=False))
    assert refs.usable_check(out, cid)
    assert not refs.usable_check(out, old_bad["id"])


def test_fulltext_not_metadata_is_required(project):
    frozen, out, rid, oid = ledger(project)
    sid = refs.add_source(out, rid, TEXT, origin="project:literature/book.txt", kind="fulltext", version="2020")
    row = {"obligation_id": oid, "reference_id": rid, "source_id": sid,
           "quote": "Under hypothesis H, the identity follows.", "locator": "Theorem 2.1",
           "premise_comparison": "Fixture hypothesis H is present.", "supported": True}
    refs.apply_checks(out, [row])
    cid = next(iter(out["checks"]))
    assert not refs.usable_check(out, cid)
    refs.corroborate(out, rid, META, sid)
    assert refs.usable_check(out, cid)
    assert not refs.coverage(out)["complete"]
    out["discovery_complete"] = True
    assert refs.coverage(out, {cid})["complete"]
    assert refs.coverage(out, set())["missing"] == [oid]
    refs.validate(out, frozen)


def test_fabricated_quote_locator_and_abstract_are_rejected(project):
    _, out, rid, oid = ledger(project)
    sid = refs.add_source(out, rid, TEXT, origin="fixture", kind="fulltext")
    row = {"obligation_id": oid, "reference_id": rid, "source_id": sid,
           "quote": "Invented theorem", "locator": "Theorem 2.1", "premise_comparison": "Checked.", "supported": True}
    with pytest.raises(ArticleError, match="not_in_source"):
        refs.apply_checks(out, [row])
    row.update(quote="Under hypothesis H, the identity follows.", locator="Theorem 99")
    with pytest.raises(ArticleError, match="not_in_source"):
        refs.apply_checks(out, [row])
    out["sources"][sid]["kind"] = "metadata"
    row["locator"] = "Theorem 2.1"
    with pytest.raises(ArticleError, match="invalid_citation_evidence"):
        refs.apply_checks(out, [row])


def test_versions_remain_separate_and_conflicts_are_visible(project):
    _, out, _, _ = ledger(project)
    a = refs.add_reference(out, dict(META, arxiv="2301.12345v1"))
    b = refs.add_reference(out, dict(META, arxiv="2301.12345v2"))
    assert a != b and b in out["references"][a]["related_versions"]
    assert refs.add_reference(out, dict(META, arxiv="2301.12345v1", year=2021)) == a
    assert out["references"][a]["conflicts"]
    with pytest.raises(ArticleError, match="identifier"):
        refs.add_reference(out, dict(META, doi="file:///etc/passwd"))


def test_disabled_retrieval_has_no_network_or_local_source_reads(project, monkeypatch):
    _, out, _, _ = ledger(project)
    def forbidden(*args, **kwargs):
        pytest.fail("disabled retrieval accessed a resource")
    monkeypatch.setattr(refs, "read_text", forbidden)
    reader = refs.SourceReader(ArticleConfig(), False)
    refs.retrieve(out, reader, queries=["fixture topic"], supplied=[{"reference": "sample", "path": "literature/book.txt"}],
                  project=project.root, search=forbidden)
    assert out["sources"] == {} and out["searches"] == []
    with pytest.raises(ArticleError, match="disabled"):
        reader.get("https://arxiv.org/html/2301.12345")


def test_operator_supplied_book_and_freshness(project):
    _, out, rid, _ = ledger(project)
    (project.root / "literature").mkdir()
    path = project.root / "literature/book.txt"
    path.write_text(TEXT)
    refs.retrieve(out, refs.SourceReader(ArticleConfig(), True),
                  supplied=[{"reference": "sample", "path": "literature/book.txt", "version": "2020"}], project=project.root)
    assert out["references"][rid]["metadata_verified"] is True
    refs.assert_sources_current(out, project.root, {"path": None})
    path.write_text(TEXT + "changed")
    with pytest.raises(ArticleError, match="changed"):
        refs.assert_sources_current(out, project.root, {"path": None})


def test_source_protocol_hosts_and_addresses_fail_closed(monkeypatch):
    monkeypatch.setattr(refs.SourceReader, "get", _REAL_GET)
    reader = refs.SourceReader(ArticleConfig(), True)
    for url in ("http://arxiv.org/a", "file:///etc/passwd", "https://localhost/a", "https://user@arxiv.org/a", "https://arxiv.org:444/a"):
        with pytest.raises(ArticleError, match="not_allowed"):
            reader.get(url)
    monkeypatch.setattr(refs.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ArticleError, match="nonpublic"):
        reader.get("https://arxiv.org/a")


def test_large_pdf_download_requires_an_explicit_size_limit_to_reject(monkeypatch):
    former_limit = 2 * 1024 * 1024
    raw = b"%PDF-1.7\n" + b"p" * former_limit
    url = "https://arxiv.org/pdf/2301.12345"
    monkeypatch.setattr(refs.SourceReader, "get", _REAL_GET)
    monkeypatch.setattr(refs, "_public_addresses", lambda host: ["8.8.8.8"])

    class Response(io.BytesIO):
        status = 200

        def getheader(self, name, default=None):
            headers = {"Content-Type": "application/pdf", "Content-Length": str(len(raw))}
            return headers.get(name, default)

    class Connection:
        sock = None

        def request(self, method, path, **kwargs):
            assert (method, path) == ("GET", "/pdf/2301.12345")

        def getresponse(self):
            return Response(raw)

        def close(self):
            pass

    monkeypatch.setattr(refs, "_PinnedHTTPS", lambda *args: Connection())
    default = ArticleConfig()
    assert default.to_dict()["max_source_bytes"] is None
    assert refs.SourceReader(default, True).get(url) == (raw, "application/pdf", url)

    saved = ArticleConfig.from_dict({"max_source_bytes": former_limit})
    with pytest.raises(ArticleError, match="source_too_large"):
        refs.SourceReader(saved, True).get(url)


def test_default_archive_expansion_still_has_a_size_budget():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for index in range(3):
            raw = b"x" * (1024 * 1024)
            info = tarfile.TarInfo(f"fixture-{index}.tex")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))

    with pytest.raises(ArticleError, match="source_archive_too_large"):
        refs.extract_text(data.getvalue(), "application/gzip", ArticleConfig().max_source_bytes)


def test_unsafe_archive_and_binary_sources_are_not_executed():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        info = tarfile.TarInfo("../../outside.tex")
        raw = b"untrusted content"
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    with pytest.raises(ArticleError, match="unsafe_source_archive"):
        refs.extract_text(data.getvalue(), "application/gzip", 4096)
    with pytest.raises(ArticleError, match="invalid_pdf_signature"):
        refs.extract_text(b"%PDF binary", "application/pdf", 4096)
    text = refs.extract_text(b"<html><script>do not execute</script><p>Actual theorem</p></html>", "text/html", 4096)
    assert "Actual theorem" in text and "execute" not in text


def test_search_budget_and_metadata_never_certify_proof(project):
    _, out, _, _ = ledger(project)
    calls = []
    def search(query, **kwargs):
        calls.append(query)
        return {"results": [{"abstract": "Not a theorem certificate."}], "is_verified": False}
    refs.retrieve(out, refs.SourceReader(ArticleConfig(max_queries=1), True),
                  queries=["query one", "query two"], search=search)
    assert calls == ["query one"]
    assert out["unresolved_candidates"] and not out["checks"]


def test_doi_metadata_requires_the_requested_identity(project):
    _, out, _, _ = ledger(project)
    rid = refs.add_reference(out, dict(META, doi="10.1234/example"))
    class Reader(refs.SourceReader):
        def get(self, url):
            payload = {"message": {"DOI": "10.9999/wrong", "title": [META["title"]],
                "author": [{"given": "Ada", "family": "Example"}], "published": {"date-parts": [[2020]]}}}
            return json.dumps(payload).encode(), "application/json", url
    refs.retrieve(out, Reader(ArticleConfig(), True))
    assert not out["references"][rid]["metadata_verified"]
    assert out["errors"]


def test_exact_doi_uses_saved_and_new_crossref_original_links(project):
    _, out, _, _ = ledger(project)
    rid = refs.add_reference(out, dict(META, doi="10.1234/example"))
    url = "https://numdam.org/item/fixture.pdf"
    calls = []

    class Reader(refs.SourceReader):
        def get(self, query):
            calls.append(query)
            payload = {"message": {"DOI": "10.1234/example", "title": [META["title"]],
                       "author": [{"given": "Ada", "family": "Example"}], "published": {"date-parts": [[2020]]},
                       "link": [{"URL": url.replace("https:", "http:")}]}}
            return json.dumps(payload).encode(), "application/json", query

        def get_text(self, source):
            calls.append(source)
            return TEXT, source

    config = ArticleConfig()
    assert "numdam.org" in config.allowed_source_hosts
    refs.retrieve(out, Reader(config, True), reference_ids={rid}, fulltext_ids=set())
    assert len(calls) == 1 and calls[0].startswith("https://api.crossref.org/works/")
    assert out["references"][rid]["metadata_verified"]
    refs.retrieve(out, Reader(config, True), reference_ids={rid}, fulltext_ids={rid})
    assert calls[1:] == [url]
    assert out["references"][rid]["source_urls"] == [url]
