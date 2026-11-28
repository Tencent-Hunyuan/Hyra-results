"""Role -> allowed-tools table — the permission skeleton of the whole system.

This is the single, first-class source of truth for separation of duties: which
MCP tools each agent role may even see. It is data (easy to audit / change), not
logic scattered across the server.

Invariants:
  - ``main`` reasons and coordinates but has no ``fact_submit``.
  - The verifier model gets literature search only; its service owns publication.
  - ``worker`` submits candidates, not verdicts or direct fact writes.
All roles get literature search; worker and main use ``knowledge_search`` and
role-limited ``knowledge_record``. These are application-level permissions, not
OS-level isolation from arbitrary shell actions under a shared account.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# All tools the gateway can expose (names must match the functions registered in
# server.py). Kept here so the role table and the implementation can't drift.
ALL_TOOLS: Tuple[str, ...] = (
    "knowledge_record",
    "knowledge_search",
    "fact_submit",
    "fact_revoke",
    "fact_review",
    "search_arxiv_theorems",
)

ROLE_TOOLS: Dict[str, Tuple[str, ...]] = {
    "worker": ("knowledge_record", "knowledge_search", "fact_submit", "fact_review", "search_arxiv_theorems"),
    "main": ("knowledge_record", "knowledge_search", "fact_revoke", "fact_review", "search_arxiv_theorems"),
    "verifier": ("search_arxiv_theorems",),
    "all": ALL_TOOLS,
}


def tools_for(role: str) -> List[str]:
    """The tool names a given role may use. An unknown / misconfigured role falls
    back to the most-restrictive read-only set (``verifier``), so a typo can never
    grant write access (fail-closed); the full set requires the explicit
    ``"all"`` key (dev use)."""
    return list(ROLE_TOOLS.get(role, ROLE_TOOLS["verifier"]))
