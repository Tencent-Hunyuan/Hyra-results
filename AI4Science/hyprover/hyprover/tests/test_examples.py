"""The bundled research snapshot is sanitized, linked, and not a runtime backup."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from hyprover.core import KNOWLEDGE_FORMAT, KnowledgeStore, compute_fact_id
from hyprover.core.knowledge_markdown import attachment_body

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "research-notebook"


def test_example_is_a_sanitized_historical_snapshot():
    data = json.loads((EXAMPLE / "project.json").read_text())
    assert data["knowledge_format"] == KNOWLEDGE_FORMAT
    assert data["synthetic_example"] is False
    assert data["static_snapshot"] is True
    assert data["workers"] == []
    assert data["source_run"]["worker_count"] == 8
    assert data["formal_certificate"] is False
    assert data["blind_research_certified"] is False
    snapshot = KnowledgeStore(EXAMPLE).snapshot()
    assert snapshot.problem_id == data["problem_id"] == "arxiv-2606-19865"
    assert len(snapshot.facts) == 13
    assert len(snapshot.entries) == 47
    assert {e.kind for e in snapshot.entries.values()} >= {
        "overview", "route", "conjecture", "experience", "feedback", "report",
    }
    completion = data["completion"]
    assert completion["state"] == "completed" and completion["outcome"] == "passed"
    assert len(snapshot.topological(completion["target_fact_ids"])) == completion["checked_facts"] == 9
    assert completion["checked_jobs"] == 10
    assert snapshot.entries[completion["feedback_id"]].display_id == "V018"
    for fact in snapshot.facts.values():
        assert fact.verification["verifier"] == "independent-llm"
        assert "Historical source-run verdict" in fact.verification["provenance"]
        assert fact.verification["formal_certificate"] is False


def test_example_ids_match_proof_content_and_proofs_are_separate():
    store = KnowledgeStore(EXAMPLE)
    snapshot = store.snapshot()
    main = store.path.read_text()
    for fid in snapshot.topological():
        fact = store.read_fact(fid, snapshot=snapshot)
        assert fid == compute_fact_id(problem_id=fact.problem_id, predecessors=fact.predecessors,
                                      glossary_introduces=fact.glossary_introduces,
                                      statement=fact.statement, proof=fact.proof)
        assert fact.statement in main
        assert fact.proof not in main
    conjectures = [e for e in snapshot.entries.values() if e.kind == "conjecture"]
    assert conjectures and all(e.status == "proved" for e in conjectures)
    for entry in conjectures:
        support = entry.attributes["resolution_fact_id"]
        assert support in entry.links and snapshot.facts[support].status == "verified"


def test_example_navigation_links_resolve():
    readme = EXAMPLE.parent / "README.md"
    navigation_link = re.compile(r"(?<![\w])\[[^\]\n]*\]\(([^)\n]+)\)")
    for document in [readme, *EXAMPLE.rglob("*.md")]:
        boundary = ROOT if document == readme else EXAMPLE
        for target in navigation_link.findall(document.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                assert parsed.scheme == "https" and parsed.netloc == "arxiv.org", target
                continue
            name, _, fragment = target.partition("#")
            dest = (document.parent / name).resolve() if name else document.resolve()
            assert dest.is_relative_to(boundary.resolve()), target
            assert dest.is_file() or (document == readme and dest.is_dir()), (document, target)
            if fragment:
                assert f"id=\"{fragment}\"" in dest.read_text(), (document, target)


def test_read_only_example_needs_no_worker_or_runtime_state():
    before = {p.relative_to(EXAMPLE): p.read_bytes() for p in EXAMPLE.rglob("*") if p.is_file()}
    hits = KnowledgeStore(EXAMPLE).search("", limit=100)
    assert any(hit["is_fact"] for hit in hits)
    assert any(not hit["is_fact"] for hit in hits)
    assert all("proof" not in hit for hit in hits)
    after = {p.relative_to(EXAMPLE): p.read_bytes() for p in EXAMPLE.rglob("*") if p.is_file()}
    assert before == after
    assert not (EXAMPLE / "workers").exists()


def test_retired_operation_examples_are_not_shipped():
    assert not (ROOT / "examples/ops/main-agent-tmux.sh").exists()
    assert not (ROOT / "examples/ops/watchdog.sh").exists()


def test_historical_reports_and_reviews_remain_linked_nonfacts():
    store = KnowledgeStore(EXAMPLE)
    snapshot = store.snapshot()
    reports = [entry for entry in snapshot.entries.values() if entry.kind == "report"]
    assert len(reports) == 22
    assert all(entry.status in {"recorded", "archived"} for entry in reports)
    subtypes = {entry.attributes.get("subtype") for entry in snapshot.entries.values()
                if entry.kind == "experience"}
    assert subtypes == {"review"}
    default_ids = {row["id"] for row in store.search("", limit=100)}
    assert not default_ids.intersection(entry.entry_id for entry in reports)
    records = {**snapshot.facts, **snapshot.entries}
    for entry in snapshot.entries.values():
        if entry.kind not in {"report", "experience"}:
            continue
        assert entry.entry_id not in snapshot.facts
        if entry.attributes.get("subtype") == "review":
            assert entry.attributes["audits"]
        for audit in entry.attributes.get("audits", ()):
            assert audit["target_id"] in entry.links
            assert audit["target_id"] in records
            assert audit["scope"].strip() and audit["not_covered"].strip()
            assert 1 <= audit["revision"] <= records[audit["target_id"]].revision
        assert store.entry_detail(entry.entry_id)["is_fact"] is False


def test_archive_has_only_referenced_public_material():
    snapshot = KnowledgeStore(EXAMPLE).snapshot()
    expected = {"project.json", "PROBLEM.md", "KNOWLEDGE.md"}
    expected.update(f.proof_path for f in snapshot.facts.values())
    expected.update(f.intuition_path for f in snapshot.facts.values() if f.intuition_path)
    expected.update(e.details_path for e in snapshot.entries.values() if e.details_path)
    paths = list(EXAMPLE.rglob("*"))
    assert not any(p.is_symlink() for p in paths)
    assert {p.relative_to(EXAMPLE).as_posix() for p in paths if p.is_file()} == expected
    assert len(expected) == 39


def test_archive_metadata_and_feedback_are_sanitized():
    snapshot = KnowledgeStore(EXAMPLE).snapshot()
    forbidden_keys = {
        "api_key", "authorization", "token", "password", "pid", "session_id",
        "run_id", "request_id", "record_request", "fingerprint", "evidence_digest",
        "full_revalidation_run", "coverage_verification_run", "report_path",
        "resolved_at", "verified_at", "project_dir", "hostname", "endpoint",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(key.lower() for key in value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(json.loads((EXAMPLE / "project.json").read_text()))
    authors = {"main_agent", "verifier", "max", *(f"max{i}" for i in range(2, 9))}
    for record in [*snapshot.facts.values(), *snapshot.entries.values()]:
        assert record.author in authors
        assert record.created_at == ""
        assert getattr(record, "updated_at", "") == ""
        inspect(record.to_dict())
    feedback = []
    for entry in snapshot.entries.values():
        if entry.kind == "feedback":
            data = json.loads(attachment_body((EXAMPLE / entry.details_path).read_text()))
            inspect(data)
            assert data["formal_certificate"] is False
            assert "not rerun on export" in data["provenance"]
            feedback.append(data)
    assert len(feedback) == 18
    assert any(data["accepted"] is False for data in feedback)
    assert any(data.get("applied") is False and data.get("action") == "revoke" for data in feedback)
    completion = next(data for data in feedback if data.get("format") == "hyprover-completion-v1")
    assert completion["accepted"] is True and completion["outcome"] == "passed"
    assert completion["checked_facts"] == 9 and completion["checked_jobs"] == 10


def test_archive_has_no_private_addresses_secrets_or_runtime_identifiers():
    forbidden = re.compile(
        r"\b(?:woa|tencent)\."
        r"|\b(?:[a-z0-9-]+\.)+(?:internal|corp|local|lan)\b"
        r"|\b(?:\d{1,3}\.){3}\d{1,3}\b"
        r"|\blocalhost\b|sk-[a-z0-9_-]{12,}|bearer\s+[a-z0-9._?=&/-]{8,}"
        r"|/(?:data|root|home|Users|tmp|opt|var|usr|workspace|mnt)/|[a-z]:\\"
        r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
        r"|\b[0-9a-f]{64}\b|\b\d{8}T\d{6}Z_[0-9a-f]{12}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b",
        re.IGNORECASE,
    )
    for path in (ROOT / "examples").rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix in {".md", ".json"}
        text = path.read_text()
        assert not forbidden.search(text), path
        for url in re.findall(r"https?://[^\s<>\"\)]+", text):
            parsed = urlsplit(url)
            assert parsed.scheme == "https" and parsed.netloc == "arxiv.org", path
