"""Build synthetic data in temporary directories for offline pipeline tests.

These records exercise dependencies, citations and prompt isolation. They are
opaque test markers, not mathematical claims or a bundled research project.
"""

import json
from pathlib import Path

from hyprover.core import KNOWLEDGE_FORMAT, KnowledgeStore, compute_fact_id
from hyprover.core.schema import Fact


INPUT_ID = compute_fact_id(
    problem_id="fixture-project", predecessors=[],
    glossary_introduces={"sample_symbol": "An opaque test token."},
    statement="The input fixture statement defines sample_symbol.",
    proof="input proof marker used only in full content; see \\cite{AC24}.",
)
SUPPORT_ID = compute_fact_id(
    problem_id="fixture-project", predecessors=[], glossary_introduces={},
    statement="The support fixture statement is available.",
    proof="support proof marker used only in full content.",
)
TARGET_ID = compute_fact_id(
    problem_id="fixture-project", predecessors=[INPUT_ID, SUPPORT_ID], glossary_introduces={},
    statement="the target fixture statement follows from both inputs.",
    proof="target proof marker combines both inputs; see \\cite{Exm20}.",
)


def create_project(path: Path) -> Path:
    path = Path(path)
    store = KnowledgeStore.initialize(path, "fixture-project")
    (path / "project.json").write_text(json.dumps({
        "project": "fixture-project", "knowledge_format": KNOWLEDGE_FORMAT,
        "synthetic_fixture": True, "workers": [],
    }), encoding="utf-8")
    (path / "PROBLEM.md").write_text(
        "# Pipeline fixture\n\n**Goal (verbatim).** Exercise the target fixture statement.\n",
        encoding="utf-8",
    )
    records = [
        Fact(
            fact_id=INPUT_ID, problem_id="fixture-project", author="test-worker",
            predecessors=[], statement="The input fixture statement defines sample_symbol.",
            proof="input proof marker used only in full content; see \\cite{AC24}.",
            glossary_introduces={"sample_symbol": "An opaque test token."},
            external_refs=[{
                "key": "AC24", "authors": ["A. Author", "B. Coauthor"],
                "title": "Synthetic input reference", "venue": "Fixture Journal",
                "year": "2024", "cited_for": "Testing citation propagation",
            }],
        ),
        Fact(
            fact_id=SUPPORT_ID, problem_id="fixture-project", author="test-worker",
            predecessors=[], statement="The support fixture statement is available.",
            proof="support proof marker used only in full content.",
        ),
        Fact(
            fact_id=TARGET_ID, problem_id="fixture-project", author="test-worker",
            predecessors=[INPUT_ID, SUPPORT_ID],
            statement="the target fixture statement follows from both inputs.",
            proof="target proof marker combines both inputs; see \\cite{Exm20}.",
            external_refs=[{
                "key": "Exm20", "authors": ["C. Example"],
                "title": "Synthetic target reference", "venue": "Fixture Notes",
                "year": "2020", "cited_for": "Testing target closure",
            }],
        ),
    ]
    for record in records:
        store.publish(record, {"verdict": "correct", "verifier": "synthetic-fixture",
                               "provenance": "offline test only; no verifier process"})
    return path
