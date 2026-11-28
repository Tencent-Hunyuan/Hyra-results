import json
from urllib.parse import parse_qs, urlsplit

from hyprover.article import evidence, references as refs, literature
from hyprover.article.models import ArticleConfig
from .conftest import publish


def test_book_title_author_lookup_without_doi_and_no_abstract_promotion(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = refs.seed(frozen)
    rid = refs.add_reference(ledger, {"title": "A real book", "authors": ["Ada Example"], "year": 2004})
    calls = []
    class Reader(refs.SourceReader):
        def get(self, url):
            calls.append(url)
            item = {"title": ["A real book"], "author": [{"given": "Ada", "family": "Example"}],
                    "published": {"date-parts": [[2004]]}, "DOI": "10.1234/book", "type": "book",
                    "publisher": "Publisher", "resource": {"primary": {"URL": "https://www.cambridge.org/book"}}}
            return json.dumps({"message": {"items": [item]}}).encode(), "application/json", url
    reader = Reader(ArticleConfig(), True)
    literature.lookup(ledger, rid, reader)
    literature.lookup(ledger, rid, reader)
    assert len(calls) == 1
    query = parse_qs(urlsplit(calls[0]).query)
    assert query["query.title"] == ["A real book"] and query["query.author"] == ["Ada Example"]
    assert ledger["references"][rid]["metadata_verified"]
    assert ledger["references"][rid]["metadata"]["doi"] == "10.1234/book"
    assert {s["kind"] for s in ledger["sources"].values()} == {"metadata"}
    assert not ledger["checks"]


def test_title_search_mismatch_not_selected_and_failures_retry_at_most_twice(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = refs.seed(frozen)
    rid = refs.add_reference(ledger, {"title": "Expected title", "authors": ["A. Writer"], "year": 2000})
    calls = []
    class Reader(refs.SourceReader):
        def get(self, url):
            calls.append(url)
            raise OSError("offline")
    reader = Reader(ArticleConfig(), True)
    for _ in range(4):
        literature.lookup(ledger, rid, reader)
    assert len(calls) == 2 and not ledger["references"][rid]["metadata_verified"]
    reader.allowed = False
    literature.lookup(ledger, rid, reader)
    assert len(calls) == 2


def test_source_connection_tries_only_validated_addresses_within_one_budget(monkeypatch):
    calls = []
    class Socket:
        def settimeout(self, value):
            assert value > 0
        def close(self):
            pass
    class Context:
        def wrap_socket(self, raw, server_hostname):
            assert server_hostname == "publisher.example"
            return raw
    def connect(address, timeout):
        calls.append((address, timeout))
        if len(calls) == 1:
            raise TimeoutError("first address unreachable")
        return Socket()
    monkeypatch.setattr(refs.socket, "create_connection", connect)
    conn = refs._PinnedHTTPS("publisher.example", ["8.8.8.8", "1.1.1.1"], 10)
    conn._context = Context()
    conn.connect()
    assert [row[0][0] for row in calls] == ["8.8.8.8", "1.1.1.1"]
    assert calls[0][1] <= 5 and calls[1][1] <= 10


def test_book_identity_queries_are_not_all_word_and_arxiv_queries():
    ledger = {"references": {"book": {"declared": {"title": "Convex Optimization"}}}}
    assert literature.topical_queries(["Boyd Convex Optimization 2004 bibliography", "log concave score bounds", "log concave score bounds"], ledger) == ["log concave score bounds"]
