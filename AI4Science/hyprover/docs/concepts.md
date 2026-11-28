# Research Concepts

HyProver coordinates ongoing research around a mathematical problem supplied by the operator. The original statement is stored in `PROBLEM.md`. The main agent maintains the overall proof strategy, exploratory subagents investigate possible approaches, workers derive specific results, and the verification service publishes independently reviewed proofs to the shared notebook. Each result can be traced to the premises it uses. [Architecture](ARCHITECTURE.md)

## Roles

| Role | Responsibilities |
| --- | --- |
| Operator | Supplies the problem, budget, and run limits, confirms results, and decides whether to publish them externally |
| Main agent | Works through the overall argument, compares routes, identifies critical gaps, assigns tasks, and consolidates progress from the shared notebook |
| Exploratory subagent | Investigates approaches, possible counterexamples, or literature. The main agent evaluates these ideas and assigns promising statements to workers for independent investigation |
| Worker | Develops proofs, shares progress, and submits complete statements, proofs, and dependencies with `fact_submit` |
| Verifier and verification service | The verifier reviews the candidate proof in an independent model session. The service rechecks premises and record revisions, publishes facts, and records feedback |

Exploratory findings can help the main agent assign work. Reusable proof premises come from published facts that remain active. The main agent maintains the overview and formal routes, while workers submit proofs and propose alternatives. The service handles publication and verification feedback.

## Turning Proof Gaps into Intermediate Conjectures

Start with `PROBLEM.md`. State what the current route has established and what the next step needs. If a theorem gives a conditional result, check its hypotheses one by one and identify those the current construction has yet to establish. Express a key gap as an intermediate conjecture that can be checked independently. If a proof route is already clear, proceed directly to the proof.

1. **State the complete conjecture.** Put the objects, definitions, quantifiers, parameter ranges, hypotheses, and intended conclusion in the `conjecture` record's `body`. Explain which premises follow from existing facts and which require separate proofs.
2. **Explain how it fits into the route.** Use `details` to identify where the conclusion would be used, which hypothesis of a downstream theorem it would discharge, or how it would connect existing results to the original goal. Link the route, predecessor facts, and any conjectures that remain unresolved.
3. **Assign it to a worker for independent investigation.** The worker can derive a complete proof, investigate a specific obstacle, or look for a counterexample satisfying the hypotheses. Submit a completed proof via `fact_submit`. The verification service handles independent review and publication. Once published, the fact can be cited in later arguments.

<a id="notebook-layout"></a>

## Research Notebook Structure

`KNOWLEDGE.md` is the shared entry point for the main agent and workers. It provides access to published facts, lets readers trace proof dependencies, and supports handoffs between rounds. The architecture guide describes the section order, display labels, and links to attachments. [Research Notebook Structure](ARCHITECTURE.md#notebook-layout)

### Experience and Audits

Use `attributes.subtype` to identify the purpose of new or revised experience records.

| Subtype | Use | Example |
| --- | --- | --- |
| `method` | A transferable approach, improvement, or precaution | Write inputs and outputs in consistent notation, then check the normalization of each downstream formula |
| `boundary` | An identified unmet condition, limit of applicability, or limit of extrapolation | An estimate establishes pointwise convergence, and exchanging limits still requires a controlling estimate |
| `review` | A specific question, error, or correction concerning a record | Review the boundary measure-zero argument in step 6 of `F011` and record the revision and scope checked |

Record the current round's attempts, blockers, and handoffs in a `report`. If a failure has a sufficiently specific and reusable explanation, record it separately as experience. Use `report` for periodic comprehensive audits and for checks that find no specific issue. Ordinary progress reports may omit audit fields.

Each round, the main agent and workers explicitly read relevant `report` records and service `feedback` before continuing. Authors can edit their own reports, and the main agent can archive reports written by others. [Tool Reference](cli-and-tools.md)

Audit reports and `review` experience use `attributes.audits` to record the target's internal ID, revision, the scope checked, and any areas left unchecked. Include the target in `links`. Use the current revision for a new audit, or set `historical=true` when reviewing an older revision. Create a separate record when checking a new revision so the original audit retains its baseline. Reassess whether the audit still applies when notified of a revision change or revocation. [Data Model](../hyprover/core/DATA_MODEL.md)

## Fact Publication and Review

Workers submit self-contained candidate proofs and explain how they use the conclusions and hypotheses of each predecessor. The fact ID is computed from the problem ID, predecessors, new glossary definitions, statement, and proof. Active facts and their dependencies form a directed acyclic graph. Later proofs can therefore cite exact facts, and readers can review them along the same dependency chain.

When a candidate proof addresses an existing conjecture, copy the search result's `source_ref` into the `source` argument to `fact_submit`. The service checks that the specified revision matches the submitted statement and marks the conjecture `proved` once the fact is published. An independent lemma can be submitted directly. If a candidate is returned for revision, the conjecture remains open.

A fact can be used as a premise once the same receipt contains `accepted=true`, `published=true`, and a valid `fact_id`. Verification is an independent LLM review. Important results warrant human review of the complete argument.

Request an independent review to raise a specific concern about an existing fact. Revocation preserves the history, invalidates proofs that depend on the revoked fact, and reopens affected resolved conjectures. An independently published stronger result that covers the old hypotheses can be recommended as a replacement. Existing proofs retain their original premises. [Data Model](../hyprover/core/DATA_MODEL.md)

## Running the Research

Keep well-supported alternatives available. The main agent can assign parallel work on different proof methods, critical hypotheses, counterexamples, or literature. If a construction stalls, record the exact hypothesis or derivation causing the problem. Decide whether to repair the construction, pursue another route, or keep testing for a genuine counterexample. Use the latest facts to update the overall argument and identify the step that still separates it from the original goal.

- **Half-hour review.** Use the original problem, shared notebook, and worker status to check the progress, evidence, and critical obstacles on each credible route. Note when a paused route should be resumed. Continue, refine, or reassign work according to the actual gaps.
- **Four-hour comprehensive audit.** Write a `report` with a time and revision baseline, compare progress, obstacles, and staffing across routes, and link the relevant `route` decisions. Update `overview` when the mathematical synthesis changes substantively. Incorporate important findings as they arrive.
- **Ongoing coordination.** The interactive main agent uses a persistent Goal to track an active, unfinished project and keeps the existing review schedule. In background mode, the controller starts subsequent main-agent sessions at the configured interval. A resumed run continues from the saved routes, assignments, and shared notebook.

Completion requires published facts that cover every objective and a complete argument connecting those facts to the original problem. When the background main agent proposes completion, the controller records the proposal or arranges independent review according to the completion settings. If the review passes and the evidence remains current, the controller stops the workers and saves the result. The operator confirms the result and decides whether to publish it externally. [Review and Completion](operating-guide.md#review-facts-and-close-a-project)
