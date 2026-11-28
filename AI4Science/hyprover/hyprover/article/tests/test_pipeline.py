import json

import pytest

from hyprover.article import evidence, references, render
from hyprover.article.models import FORMAT, ArticleError, digest
from .conftest import publish


def block(identifier, text, sources, kind="paragraph", **extra):
    return dict(id=identifier, kind=kind, text=text, evidence=sources, citations=[], crossrefs=[], **extra)


def article_fixture(frozen, ledger):
    fid = frozen["targets"][0]
    statement = fid + ":statement"
    proof = fid + ":proof:part-0001"
    return {"format": FORMAT, "evidence_digest": frozen["digest"], "reference_digest": digest(ledger),
        "title": {"text": "A fixture result", "evidence": [statement]},
        "abstract": block("abstract", "We establish the fixture result under its stated hypotheses.", [statement]),
        "keywords": ["fixture"], "msc": ["00A00"],
        "sections": [
            {"id": "intro", "role": "introduction", "title": "Introduction", "blocks": [
                block("result", frozen["facts"][0]["statement"], [statement], "theorem")]},
            {"id": "prelim", "role": "preliminaries", "title": "Preliminaries", "blocks": [
                block("notation", "We retain the hypotheses of the result.", [statement])]},
            {"id": "core", "role": "core", "title": "The fixture argument", "blocks": [
                block("method", "We use the given argument.", [proof])]},
            {"id": "proofs", "role": "main_proofs", "title": "Proofs of the main results", "blocks": [
                block("proof", "We prove the stated result.", [statement], "proof", proves="result", parts=[
                    block("proof-part", frozen["facts"][0]["proof"], [proof])])]},
        ]}


def fixture(project):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    ledger = references.seed(frozen)
    ledger["discovery_complete"] = True
    ledger["stop_reason"] = "no_external_results_identified"
    return frozen, ledger, article_fixture(frozen, ledger)


def test_fixed_template_author_and_no_declarations(project):
    _, ledger, article = fixture(project)
    outputs = render.render(article, ledger)
    tex = outputs["paper.tex"]
    assert r"\documentclass[11pt,a4paper]{amsart}" in tex
    assert r"{\bfseries hyprover\par}" + "\n" + r"\end{center}" in tex
    assert r"\begin{minipage}{14cm}" in tex and r"\maketitle" not in tex
    assert tex.index(r"\bfseries hyprover") < tex.index(r"\textbf{Abstract.}") < tex.index("Subject Classification")
    assert tex.index("geometry}") < tex.index(r"\textwidth=16 true cm")
    assert r"\mathtoolsset{showonlyrefs}" in tex
    assert r"\numberwithin{equation}{section}" in tex
    assert "@@HYPROVER_" not in tex
    for forbidden in ("Conflict of Interest", "Data Availability", "Declaration on the use of AI", r"\thanks", "Funding"):
        assert forbidden not in tex
    assert outputs["references.tex"] == ""
    _, profile = render.template()
    assert profile["author"] == "hyprover" and "affiliation" not in profile
    assert len(profile["source_main_tex_sha256"]) == 64
    assert profile["visual_comparison_performed"] is False


@pytest.mark.parametrize("text", [r"\input{/etc/passwd}", r"\write18{command}", r"\csname input\endcsname",
    r"\catcode1=0", r"\directlua{os.execute()}", r"\begin{document}text\end{document}",
    r"\def\a{b}", "^^5cinput", "Conflict of Interest. None.", "Text % hidden", r"\cite{unknown}"])
def test_untrusted_tex_and_declarations_are_rejected(text):
    with pytest.raises(ArticleError):
        render.safe_tex(text)


def test_math_cross_references_and_missing_safe_compiler(project, monkeypatch):
    _, ledger, article = fixture(project)
    proof = article["sections"][3]["blocks"][0]["parts"][0]
    proof["text"] = r"For \(x\in\R\), use Theorem [[ref:result]]."
    proof["crossrefs"] = ["result"]
    assert r"\ref{b:result}" in render.render(article, ledger)["paper.tex"]
    proof["text"] = "See [[ref:missing]]."
    with pytest.raises(ArticleError, match="dangling"):
        render.render(article, ledger)
    monkeypatch.setattr(render.shutil, "which", lambda _: None)
    assert render.compile_pdf(project.root) == {"compiled": False, "error": "safe_tex_environment_unavailable"}


@pytest.mark.parametrize("timeout", [None, 37])
@pytest.mark.parametrize("timed_out", [False, True])
def test_compile_time_limits_and_explicit_override(tmp_path, monkeypatch, timeout, timed_out):
    from pathlib import Path
    from unittest.mock import Mock
    from hyprover.article.budget import OperationTimeout

    for name in ("paper.tex", "references.tex"):
        (tmp_path / name).write_text("Synthetic compilation input")
    monkeypatch.setattr(render.shutil, "which", lambda name: "/synthetic/" + name)

    budgets = []
    def compile_round(command, budget, **kwargs):
        budgets.append(budget.remaining())
        if timed_out:
            raise OperationTimeout(budget.error)
        work = Path(command[command.index("--bind") + 1])
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                   b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>"]
        pdf, offsets = b"%PDF-1.4\n", [0]
        for index, body in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 4\n0000000000 65535 f \n"
        pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
        pdf += f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        (work / "paper.pdf").write_bytes(pdf)
        (work / "paper.log").write_text("Output written on paper.pdf (1 page, 300 bytes).\n")
        return 0

    run = Mock(side_effect=compile_round)
    monkeypatch.setattr(render, "run_command", run)
    result = render.compile_pdf(tmp_path, **({} if timeout is None else {"timeout": timeout}))
    assert run.call_count == (1 if timed_out else 3)
    assert all(0 < remaining <= (270 if timeout is None else timeout) for remaining in budgets)
    assert budgets == sorted(budgets, reverse=True)
    assert len({id(call.args[1]) for call in run.call_args_list}) == 1
    for call in run.call_args_list:
        command = call.args[0]
        assert "--cpu=180" in command
        assert "--as=1073741824" in command and "--fsize=33554432" in command and "--nproc=64" in command
        assert "--unshare-all" in command and "-no-shell-escape" in command
    if timed_out:
        assert result == {"compiled": False, "error": "isolated_tex_compilation_unavailable_or_timed_out"}
        assert not (tmp_path / "paper.pdf").exists()
    else:
        assert result["compiled"] is True
        assert Path(result["path"]).read_bytes().startswith(b"%PDF-")
        assert result["pages"] == 1 and result["isolated"]
        assert render.current_pdf(tmp_path) == result
        (tmp_path / "references.tex").write_text("Changed source")
        assert render.current_pdf(tmp_path) is None
    assert not list(tmp_path.glob(".tex-*"))


def test_citations_use_only_checked_locators_and_verified_metadata(project):
    frozen, ledger, article = fixture(project)
    rid = references.add_reference(ledger, {"title": "Example book", "authors": ["A. Example"], "year": 2020})
    oid = "book-obligation"
    ledger["obligations"][oid] = {"id": oid, "reference_ids": [rid]}
    sid = references.add_source(ledger, rid, "Theorem 1. Every fixture has the stated property.", origin="fixture", kind="fulltext")
    references.corroborate(ledger, rid, {"title": "Example book", "authors": ["A. Example"], "year": 2020, "type": "book"}, sid)
    references.apply_checks(ledger, [{"obligation_id": oid, "reference_id": rid, "source_id": sid,
        "quote": "Every fixture has the stated property.", "locator": "Theorem 1", "premise_comparison": "Fixture.", "supported": True}])
    cid = next(iter(ledger["checks"]))
    target = article["sections"][1]["blocks"][0]
    target["text"] = "Use the stated property [[cite:" + cid + "]]."
    target["citations"] = [cid]
    outputs = render.render(article, ledger)
    assert r"\cite[Theorem 1]{" + rid + "}" in outputs["paper.tex"]
    assert r"\bibitem{" + rid + "}" in outputs["references.tex"]
    assert "@book{" + rid in outputs["references.bib"]
    ledger["references"][rid]["metadata_verified"] = False
    with pytest.raises(ArticleError, match="unverified"):
        render.render(article, ledger)
