import copy
import json

import pytest

from hyprover.article import citation_policy as policy, evidence, literature, references as refs
from hyprover.article.models import ArticleConfig, ArticleError, digest
from .conftest import publish
from .test_reference_refresh import ITEM, META, referenced


def ledger_fixture(project):
    frozen = evidence.capture(project.root, [publish(project)])
    return frozen, refs.seed(frozen)


class Reader(refs.SourceReader):
    def __init__(self, *, fail=False, max_sources=48):
        super().__init__(ArticleConfig(max_sources=max_sources), True)
        self.urls = []
        self.fail = fail

    def get(self, url):
        assert self.requests < self.config.max_sources
        self.urls.append(url)
        self.requests += 1
        if self.fail:
            raise OSError("fixture source unavailable")
        return b"Theorem 2. A fixture statement. Proof. A fixture argument.", "text/plain", url

    def get_text(self, url):
        raw, content_type, final = self.get(url)
        return refs.extract_text(raw, content_type), final


def test_same_reference_keeps_new_urls_without_overwriting_original_evidence(project):
    frozen, ledger, article, cid = referenced(project)
    check = copy.deepcopy(ledger["checks"][cid])
    rid = check["reference_id"]
    row = ledger["references"][rid]
    original = copy.deepcopy(row["declared"])
    source = copy.deepcopy(ledger["sources"][check["source_id"]])
    first, second = "https://projecteuclid.org/original", "https://www.ams.org/alternative"
    refs.merge_source_urls(row, [first])
    assert refs.add_reference(ledger, {**META, "fulltext_url": second}) == rid
    assert row["declared"] == original and row["source_urls"] == [first, second]
    refs.merge_source_urls(row, [second, "file:///private", "https://user:password@www.ams.org/secret"])
    assert row["source_urls"] == [first, second]
    assert ledger["checks"][cid] == check and ledger["sources"][check["source_id"]] == source
    assert refs.usable_check(ledger, cid)
    refs.validate(ledger, frozen)


@pytest.mark.parametrize("kind", ["unconfirmed", "fulltext"])
def test_old_document_does_not_block_new_publisher_url_in_supplement_mode(project, kind):
    frozen, ledger = ledger_fixture(project)
    rid = refs.add_reference(ledger, META)
    old = "https://projecteuclid.org/old"
    sid = refs.add_source(ledger, rid, "Old content lacks the needed theorem.", origin=old, kind=kind)
    snapshot = copy.deepcopy(ledger["sources"][sid])
    skipped = [old, "https://www.ams.org/done1", "https://www.ams.org/done2"]
    new = "https://projecteuclid.org/new-original"
    refs.merge_source_urls(ledger["references"][rid], [*skipped, new])
    for url in skipped[1:]:
        ledger.setdefault("retrieval_requests", {})[digest([rid, "fulltext", url])] = {"state": "complete", "attempts": 1}
    reader = Reader()
    literature.publisher_sources(ledger, rid, reader)
    assert reader.urls == ([new] if kind == "unconfirmed" else [])
    literature.publisher_sources(ledger, rid, reader, supplement=True)
    literature.publisher_sources(ledger, rid, reader, supplement=True)
    assert reader.urls == [new]
    assert ledger["sources"][sid] == snapshot
    assert not ledger["checks"]
    assert all(s["kind"] != "fulltext" for k, s in ledger["sources"].items() if k != sid)
    refs.validate(ledger, frozen)


def test_new_publisher_failures_and_network_budget_remain_bounded(project):
    _, ledger = ledger_fixture(project)
    rid = refs.add_reference(ledger, {**META, "fulltext_url": "https://www.ams.org/unavailable"})
    reader = Reader(fail=True)
    for _ in range(5):
        literature.publisher_sources(ledger, rid, reader, supplement=True)
    assert len(reader.urls) == 2
    limited = Reader(max_sources=0)
    refs.merge_source_urls(ledger["references"][rid], ["https://www.ams.org/new"])
    literature.publisher_sources(ledger, rid, limited, supplement=True)
    assert not limited.urls
    reader.allowed = False
    literature.publisher_sources(ledger, rid, reader, supplement=True)
    assert len(reader.urls) == 2


def test_arxiv_supplement_fetches_new_endpoint_without_redownloading_old_source(project):
    frozen, ledger = ledger_fixture(project)
    metadata = {"title": "Fixture preprint", "authors": ["A. Writer"], "year": 2023, "arxiv": "2301.12345v1"}
    rid = refs.add_reference(ledger, metadata)
    meta = refs.add_source(ledger, rid, "Fixture identity", origin="https://arxiv.org/abs/2301.12345v1", kind="metadata")
    refs.corroborate(ledger, rid, metadata, meta)
    old = refs.add_source(ledger, rid, "Abstract-only old HTML", origin="https://arxiv.org/html/2301.12345v1", kind="unconfirmed")
    before = copy.deepcopy(ledger["sources"][old])
    reader = Reader()
    refs.retrieve(ledger, reader, reference_ids={rid}, fulltext_ids={rid}, supplement_ids={rid})
    refs.retrieve(ledger, reader, reference_ids={rid}, fulltext_ids={rid}, supplement_ids={rid})
    assert reader.urls == ["https://arxiv.org/src/2301.12345v1", "https://arxiv.org/pdf/2301.12345v1"]
    assert ledger["sources"][old] == before and not ledger["checks"]
    refs.validate(ledger, frozen)


def test_arxiv_falls_through_metadata_and_stops_after_confirmed_body(project):
    _, ledger = ledger_fixture(project)
    meta = {"title": "Fixture preprint", "authors": ["A. Writer"], "year": 2023, "arxiv": "2301.12345v1"}
    rid = refs.add_reference(ledger, meta)
    sid = refs.add_source(ledger, rid, "Fixture identity", origin="https://arxiv.org/abs/2301.12345v1", kind="metadata")
    refs.corroborate(ledger, rid, meta, sid)

    class MetadataFirst(Reader):
        def get_text(self, url):
            if "/html/" in url:
                self.urls.append(url)
                self.requests += 1
                return "Abstract\nA summary of the fixture.", url
            return super().get_text(url)

    reader = MetadataFirst()

    def classify(reference, source):
        refs.confirm_source(ledger, source, reference, {"source_id": source,
            "source_sha256": ledger["sources"][source]["sha256"], "reference_id": reference,
            "document_type": "fulltext", "identity_supported": True, "outside_abstract": True,
            "body_quote": "A fixture statement.", "locator": "Theorem 2"})

    refs.retrieve(ledger, reader, reference_ids={rid}, classify=classify)
    assert reader.urls == ["https://arxiv.org/html/2301.12345v1", "https://arxiv.org/src/2301.12345v1"]
    assert any(refs.source_is_fulltext(ledger, sid, rid) for sid in ledger["references"][rid]["source_ids"])


def test_focused_search_preserves_old_urls_and_only_clears_completed_skips(project):
    _, ledger = ledger_fixture(project)
    rid = refs.add_reference(ledger, META)
    first, second = "https://www.ams.org/old-source", "https://www.ams.org/new-source"
    refs.merge_source_urls(ledger["references"][rid], [first])
    ledger["focus"] = {"revision": 1, "searches": {str(i): {"state": "empty", "attempts": 1} for i in range(6)},
                       "skipped_queries": ["previously-skipped", "still-untried"]}
    item = copy.deepcopy(ITEM)
    item["resource"] = {"primary": {"URL": second}}
    reader = Reader()

    def get(url):
        reader.requests += 1
        return json.dumps({"message": {"items": [item]}}).encode(), "application/json", url

    reader.get = get
    literature.focused_search(ledger, reader, ["previously-skipped"], review=True)
    assert ledger["references"][rid]["source_urls"] == [first, second]
    assert ledger["focus"]["skipped_queries"] == ["still-untried"]
    assert ledger["focus"]["resolved_skipped_queries"] == ["previously-skipped"]
    assert not ledger["focus"]["search_complete"]
    literature.focused_search(ledger, reader, ["still-untried"], review=True)
    assert ledger["focus"]["search_complete"]


def test_bind_reference_keeps_duplicate_obligations_grouped_and_old_checks_usable(project):
    frozen, ledger, article, cid = referenced(project)
    oid = ledger["checks"][cid]["obligation_id"]
    duplicate = dict(ledger["obligations"][oid], id="duplicate", reference_ids=list(ledger["obligations"][oid]["reference_ids"]))
    ledger["obligations"]["duplicate"] = duplicate
    check = copy.deepcopy(ledger["checks"][cid])
    rid = refs.add_reference(ledger, {"title": "Alternative source", "doi": "10.1234/alternative"})
    assert len(policy.groups(ledger)) == 1
    refs.bind_reference(ledger, "duplicate", rid)
    assert len(policy.groups(ledger)) == 1
    assert ledger["obligations"][oid]["reference_ids"] == duplicate["reference_ids"]
    assert rid in duplicate["reference_ids"]
    assert ledger["checks"][cid] == check and refs.usable_check(ledger, cid)
    refs.validate(ledger, frozen)


def test_internal_proof_resolution_protects_entire_duplicate_group(project):
    frozen, ledger = ledger_fixture(project)
    fid = frozen["targets"][0]
    row = {"id": "internal", "description": "An original proof step", "source_units": [fid + ":proof:part-0001"],
           "reference_ids": [], "kind": "mathematical", "critical": True}
    refs.add_obligations(ledger, [row, dict(row, id="duplicate")], frozen)
    refs.apply_internal_resolution(ledger, frozen, {"obligation_id": "internal", "supported_internally": True,
        "proof_units": [fid + ":proof:part-0001"], "reason": "Established by the supplied proof."})
    rid = refs.add_reference(ledger, META)
    before = copy.deepcopy(ledger)
    assert not refs.external_obligations(ledger)
    for oid in ("internal", "duplicate"):
        with pytest.raises(ArticleError, match="internally_resolved"):
            refs.bind_reference(ledger, oid, rid)
    assert ledger == before
    refs.validate(ledger, frozen)
