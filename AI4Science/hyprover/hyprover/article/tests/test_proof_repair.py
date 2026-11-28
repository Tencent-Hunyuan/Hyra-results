"""Exercise the article workflow with a scripted candidate and independent verdict."""

import pytest

from hyprover.article import delivery, evidence, proof_repair, render, runtime
from hyprover.article.agent import Reply
from hyprover.article.models import digest, read_json, write_json
from .conftest import publish
from .test_audit import FakeAgent
from .test_publication import fixture


STATEMENT = "The fixture property holds at every point."
SOURCE_PROOF = "At the transition, both pieces agree with the defining expression."
NARROWED = "Assume the junction condition. Then the fixture property holds at every point."
OMISSION = "The transition is left unverified."
SUPPLEMENTED = (SOURCE_PROOF
                + " Therefore the defining property extends to the transition point.")


def _blocks(article):
    return {block["id"]: block for block in render.walk_blocks(article)}


def test_compact_proof_review_keeps_the_complete_target_and_exact_premises(project):
    frozen, _, _, _, article = fixture(project)
    predecessor = frozen["targets"][0]
    target = publish(project, statement=STATEMENT, proof=SOURCE_PROOF, predecessors=[predecessor])
    frozen = evidence.capture(project.root, [target])

    material = proof_repair.target_material(article, frozen, target, include_dependency_proofs=False)
    source = {unit["id"]: unit["text"] for unit in material["original_units"]}
    assert not material["dependency_proofs_included"]
    assert source[target + ":statement"] == STATEMENT
    assert source[predecessor + ":statement"] == frozen["facts"][0]["statement"]
    assert "".join(text for uid, text in source.items() if uid.startswith(target + ":proof:")) == SOURCE_PROOF
    assert not any(uid.startswith(predecessor + ":proof:") for uid in source)


class RepairAgent(FakeAgent):
    def __init__(self, root, outcome):
        super().__init__(evidence.load(root))
        self.root = root
        self.outcome = outcome

    def __call__(self, task, material, contract):
        if task == "compose_paper":
            reply = super().__call__(task, material, contract)
            blocks = _blocks(reply.value["article"])
            blocks["result"]["text"] = NARROWED
            blocks["proof-part"]["text"] = OMISSION
            return reply
        if task not in {"detect_proof_gaps", "propose_proof_repair", "verify_proof_repair"}:
            return super().__call__(task, material, contract)
        self.calls.append(task)
        response = {"binding": material["binding"]}
        if task == "detect_proof_gaps":
            fact_id = self.frozen["targets"][0]
            response.update(materials_complete=True, findings=[{
                "kind": "scope",
                "block_ids": ["result", "proof-part"],
                "source_unit_ids": [fact_id + ":statement", fact_id + ":proof:part-0001"],
                "issue": "The theorem adds a junction hypothesis and the proof omits that case.",
            }])
        elif task == "propose_proof_repair":
            blocks = {block["id"]: block for block in material["blocks"]}
            response.update(patches=[
                {"block_id": bid, "before_digest": blocks[bid]["before_digest"], "text": replacement}
                for bid, replacement in (("result", STATEMENT), ("proof-part", SUPPLEMENTED))
            ], reason="Restore the stated scope and supply the missing junction argument.")
        else:
            saved = _blocks(read_json(self.root / "article.json"))
            assert saved["result"]["text"] == NARROWED
            assert saved["proof-part"]["text"] == OMISSION
            checked = {block["id"]: block for block in material["candidate_blocks"]}
            assert checked["result"]["text"] == STATEMENT
            assert checked["proof-part"]["text"] == SUPPLEMENTED
            assert any(unit["text"] == SOURCE_PROOF for unit in material["original_units"])
            accepted = self.outcome != "rejected"
            response.update(candidate_digest=material["candidate_digest"],
                            materials_complete=True, proof_complete=accepted,
                            scope_preserved=accepted, no_new_premises=True,
                            verdict="verified" if accepted else "rejected",
                            reason="Fixture verdict for the supplied proof.")
            if self.outcome == "same_session":
                return Reply(self.candidate_session, response)
        session = f"repair-session-{len(self.calls)}"
        if task == "propose_proof_repair":
            self.candidate_session = session
        return Reply(session, response)


@pytest.mark.parametrize("outcome", ["passed", "rejected", "same_session"])
def test_only_independently_verified_proof_changes_enter_the_article(project, outcome):
    fact_id = publish(project, statement=STATEMENT, proof=SOURCE_PROOF)
    notebook = (project.root / "KNOWLEDGE.md").read_bytes()
    root = runtime.prepare(project.root, [fact_id])
    agent = RepairAgent(root, outcome)
    status = runtime.work(project.root, root.name, agent=agent)
    assert "verify_proof_repair" in agent.calls, status
    receipt = read_json(root / "proof-repair.json")
    article = read_json(root / "article.json")
    assert receipt["article_digest"] == digest(article)
    assert receipt["evidence_digest"] == agent.frozen["digest"]
    blocks = _blocks(article)
    manuscript = (root / "paper.tex").read_text()
    assert (project.root / "KNOWLEDGE.md").read_bytes() == notebook
    if outcome == "passed":
        assert status["state"] == "ready", status
        assert blocks["result"]["text"] == STATEMENT
        assert SUPPLEMENTED in manuscript
        assert NARROWED not in manuscript
        assert runtime.recheck(project.root, root.name)["state"] == "ready"
        write_json(root / "proof-repair.json", dict(receipt, article_digest="a different manuscript"))
        rechecked = runtime.recheck(project.root, root.name)
        assert rechecked["state"] == "needs_review"
        assert rechecked["error"] == "proof_repair_receipt_missing_or_stale"
    else:
        assert status["state"] == "needs_review", status
        assert blocks["result"]["text"] == NARROWED
        assert OMISSION in manuscript
        assert SUPPLEMENTED not in manuscript
        assert runtime.recheck(project.root, root.name)["state"] == "needs_review"
        assert delivery.export(project.root, root.name)["state"] == "needs_review"
