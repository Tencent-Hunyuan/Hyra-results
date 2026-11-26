import copy
import json
import shutil
import subprocess

import pytest

from hyprover.article import evidence, field_recovery, render, runtime, response_validation
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleError, digest, read_json, write_json
from hyprover.article.pipeline import _BLOCK_CONTRACT, _LEGACY_BLOCK_CONTRACT
from .test_recovery import make_pipeline
from .test_audit import FakeAgent
from .test_citation_policy import material
from .test_pipeline import article_fixture


def damaged_article(pipe):
    result = FakeAgent(pipe.evidence)("compose_paper", {"reference_digest": digest(pipe.ledger)}, "").value
    proof = result["article"]["sections"][3]["blocks"][0]
    proof["kind"] = "paragraph"
    proof["parts"][0]["text"] += "\x10"
    return result


def patches(fields):
    return {"patches": [{"path": row["path"], "before_digest": row["before_digest"],
                        "value": "proof" if row["path"][-1] == "kind" else row["value"].replace("\x10", "")}
                       for row in fields]}


def test_targeted_patch_preserves_all_unrequested_fields_and_original_cache(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original = damaged_article(pipe)
    original_copy = copy.deepcopy(original)
    contract = _BLOCK_CONTRACT + "Fixture publication contract"
    request = {"fixture": True}
    legacy_key = digest([pipe.fingerprint, "compose_paper", request,
                         _LEGACY_BLOCK_CONTRACT + "Fixture publication contract", False])
    path = pipe.root / "calls" / (legacy_key + ".json")
    path.parent.mkdir(exist_ok=True)
    write_json(path, {"request_digest": legacy_key, "value": original, "value_digest": digest(original),
                     "session_id": "cached-writer", "reviewer": False, "task": "compose_paper"})
    before = path.read_bytes()
    calls = []

    def agent(task, data, contract):
        assert task == "repair_article_fields"
        assert "response" not in data and "validation_material" not in data
        calls.append(task)
        return Reply("field-writer", patches(data["fields"]))

    pipe.agent = agent
    validation = {"evidence": pipe.evidence, "ledger": pipe.ledger}
    reply = pipe.ask("compose_paper", request, contract, validation_material=validation)
    response_validation.validate("compose_paper", reply.value, validation)
    assert reply.session_id == "cached-writer"
    assert calls == ["repair_article_fields"]
    assert original == original_copy and path.read_bytes() == before
    assert pipe.checkpoint["calls"] == 1
    assert list((pipe.root / "response-repairs").glob("*.json"))
    again = pipe.ask("compose_paper", request, contract, validation_material=validation)
    assert again.value == reply.value and len(calls) == 1


@pytest.mark.parametrize("damage", ["extra", "stale", "duplicate", "kind", "reference", "unsafe", "empty"])
def test_patch_rejects_scope_and_binding_changes(project, tmp_path, damage):
    pipe = make_pipeline(project, tmp_path)
    original = damaged_article(pipe)
    fields = field_recovery.plan(original, pipe.evidence)
    response = patches(fields)
    if damage == "extra":
        response["patches"][0]["path"] = ["title", "text"]
    elif damage == "stale":
        response["patches"][0]["before_digest"] = "changed"
    elif damage == "duplicate":
        response["patches"][1] = copy.deepcopy(response["patches"][0])
    elif damage == "kind":
        response["patches"][0]["value"] = "theorem"
    elif damage == "reference":
        response["patches"][1]["value"] += " [[cite:invented]]"
    elif damage == "unsafe":
        response["patches"][1]["value"] = r"\input{secret}"
    else:
        response["patches"] = []
    with pytest.raises((ArticleError, TypeError)):
        field_recovery.apply(original, fields, response)


def test_legal_arrows_do_not_relax_unsafe_tex():
    render.safe_tex(r"\(x_n\uparrow x,\quad y_n\downarrow y\)")
    for text in (r"\input{secret}", r"\write18{cmd}", r"\def\x{1}", "x\x10y"):
        with pytest.raises(ArticleError):
            render.safe_tex(text)


@pytest.mark.skipif(not all(shutil.which(n) for n in ("bwrap", "pdflatex", "prlimit", "pdftotext")),
                    reason="Optional isolated TeX runtime unavailable")
@pytest.mark.parametrize("count", [1, 2])
def test_real_template_nonempty_bibliography_has_terminated_last_item(project, tmp_path, count):
    frozen, ledger, rid, _ = material(project)
    article = article_fixture(frozen, ledger)
    references = {}
    checks = {}
    for i in range(count):
        key = "ref-" + str(i)
        references[key] = dict(ledger["references"][rid], id=key)
        checks["check-" + str(i)] = {"reference_id": key}
    outputs = render.render(article, {**ledger, "checks": {}})
    outputs["references.tex"], outputs["references.bib"] = render.bibliography(
        {"references": references, "checks": checks}, set(checks))
    assert "\n\n\\end{thebibliography}" in outputs["references.tex"]
    root = tmp_path / "compiled"
    root.mkdir()
    render.write(root, outputs)
    result = render.compile_pdf(root)
    assert result["compiled"], (root / "compile.log").read_text()
    text = subprocess.run(["pdftotext", str(root / "paper.pdf"), "-"], check=True,
                          capture_output=True, text=True).stdout
    assert "Real analysis" in text


def test_legacy_bibliography_migration_is_narrow_idempotent_and_no_model(project, monkeypatch):
    from .conftest import prepare_legacy_review, publish
    fid = publish(project)
    root = prepare_legacy_review(project.root, [fid])
    state = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root), truncated=True))
    assert state["state"] == "needs_review"
    original_article = (root / "article.json").read_bytes()
    original_render = render.render
    tex = "\\begin{thebibliography}{99}\n\\bibitem{x} Fixture.\n\n\\end{thebibliography}\n"

    def revised(article, ledger):
        outputs = original_render(article, ledger)
        outputs["references.tex"] = tex
        return outputs

    monkeypatch.setattr(render, "render", revised)
    (root / "references.tex").write_text(tex.replace("\n\n\\end", "\n\\end"))
    monkeypatch.setattr(render, "compile_pdf", lambda p, **_: {"compiled": True})
    first = runtime.compile_article(project.root, root.name)
    assert first["state"] == "needs_review" and first["compilation"]["compiled"]
    assert (root / "article.json").read_bytes() == original_article
    assert read_json(root / "audit.json")["receipts"] == {}
    history = root / "typesetting-revisions/bibliography-paragraph-v1/references.tex"
    saved = history.read_text()
    assert saved != tex and (root / "references.tex").read_text() == tex
    assert runtime.compile_article(project.root, root.name)["compilation"]["compiled"]
    assert history.read_text() == saved
    (root / "paper.tex").write_text((root / "paper.tex").read_text() + "changed")
    with pytest.raises(ArticleError, match="rendered_sources_changed"):
        runtime.compile_article(project.root, root.name)
