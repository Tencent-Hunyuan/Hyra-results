"""Versioned citation requirements; routine tools are not external proof gaps."""
from __future__ import annotations

import re

POLICY_REVISION = 2


def signature(row: dict) -> tuple:
    description = re.sub(r"\s+", " ", row["description"]).strip().casefold()
    return description, tuple(sorted(row["reference_ids"])), row["kind"], tuple(sorted(row["source_units"]))


def groups(ledger: dict) -> dict[str, dict]:
    """Keep every original obligation ID, but count identical requirements once."""
    result, indexed = {}, {}
    for oid, row in ledger["obligations"].items():
        key = signature(row)
        canonical = indexed.setdefault(key, oid)
        if canonical not in result:
            result[canonical] = dict(row, id=canonical, member_ids=[], source_units=[], critical=False)
        combined = result[canonical]
        combined["member_ids"].append(oid)
        combined["critical"] = combined["critical"] or row.get("critical", True) is not False
        combined["source_units"] = sorted(set(combined["source_units"]) | set(row["source_units"]))
    return result


def canonical_ids(ledger: dict) -> dict[str, str]:
    return {oid: canonical for canonical, row in groups(ledger).items() for oid in row["member_ids"]}


def essential(row: dict) -> bool:
    return row.get("critical", True) is not False or row.get("kind") in ("historical", "comparison")


def policy_text() -> str:
    return (
        "Citation policy v2: substantive external theorems, borrowed core arguments, historical attribution "
        "and priority/comparison claims require original-source evidence, precise locators and applicability checks. "
        "Routine textbook mathematics may use an independently assessed standard reference with verified "
        "bibliographic identity; no invented page number, verbatim quotation, or separate citation at every use "
        "is required. Such a standard-reference check does NOT certify a specific external theorem. "
        "Missing nonessential textbook references are advisory, not proof gaps. Inspect the classification: "
        "a substantive external result mislabeled routine is a blocking finding. Identical obligations count "
        "once; one relevant citation can cover repeated uses. A proof absent from a chunk but present in the "
        "complete evidence index is not missing evidence or a new literature obligation. "
        "Do not manufacture citations to remove warnings. Critical unresolved sources still prevent ready."
    )
