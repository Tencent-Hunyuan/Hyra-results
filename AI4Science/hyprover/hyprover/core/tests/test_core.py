"""Schema helpers and the agent-readable notation reference."""

from __future__ import annotations

import json
from importlib import resources

from hyprover.core import clean_external_refs, compute_fact_id


def test_fact_id_compatibility_vectors():
    base = dict(problem_id="P", predecessors=[], glossary_introduces={}, statement="A", proof="Proof")
    for changes, expected in [
        ({}, "cdfa7ed261ed7400"),
        (dict(statement=" \tA\u2003\n", proof="Proof\r\n"), "cdfa7ed261ed7400"),
        (dict(problem_id="题", predecessors=["b", "a", "a"],
              glossary_introduces={"β": " x\n y ", "α": "<s>"},
              statement="∀ x\u00a0∈ R", proof='引号"，反斜线\\，控制\x00'), "5f270954d09b1dd5"),
        (dict(predecessors=["b", "a"]), "e50e68f9288104bb"),
        (dict(predecessors=["b", "a", "a"]), "1a75fed5abbf4bba"),
        (dict(glossary_introduces={1: "a", "1": "z"}), "f806fe2d7f41ccd3"),
        (dict(statement="é"), "3205d8a1a73a2da2"),
        (dict(statement="e\u0301"), "422f140ed5f883fd"),
    ]:
        payload = base | changes
        assert compute_fact_id(**payload) == expected
        payload["predecessors"] = list(reversed(payload["predecessors"]))
        payload["glossary_introduces"] = dict(reversed(list(payload["glossary_introduces"].items())))
        assert compute_fact_id(**payload) == expected


def test_schema_clean_external_refs_extra_keys():
    out = clean_external_refs([{"note": "z", "title": "T", "key": "K", "aardvark": 1}])
    assert out == [{"key": "K", "title": "T", "aardvark": 1, "note": "z"}]
    assert list(out[0]) == ["key", "title", "aardvark", "note"]


def test_universal_notation_reference_remains_packaged_data():
    def unique_object(pairs):
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys)), "Duplicate notation reference key"
        return dict(pairs)

    raw = resources.files("hyprover.core").joinpath("glossary_global.json").read_text(encoding="utf-8")
    reference = json.loads(raw, object_pairs_hook=unique_object)
    assert reference["version"] == 1 and reference["terms"]
    assert isinstance(reference["description"], str) and reference["description"].strip()
    for term, row in reference["terms"].items():
        assert isinstance(term, str) and term.strip()
        assert isinstance(row["kind"], str) and row["kind"].strip()
        assert isinstance(row["definition"], str) and row["definition"].strip()
        aliases = row["aliases"]
        assert isinstance(aliases, list)
        assert all(isinstance(alias, str) and alias.strip() for alias in aliases)
        assert len(aliases) == len(set(aliases))
    assert {
        "Z", "N", "T_{Y/Z}", "forall", "exists", "emptyset", "A subseteq B",
        "f^{-1}(S)", "n!", "binom(n,k)", "ker(T)", "sup(A)", "O(g)", "E[X]",
    } <= reference["terms"].keys()
