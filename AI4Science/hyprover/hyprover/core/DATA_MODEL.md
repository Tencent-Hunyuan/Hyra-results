# Research Notebook Data Model

Each project keeps its shared state in one `KNOWLEDGE.md` file. Published facts contain complete mathematical statements. Research entries capture routes, progress, and reviews. Full proofs and longer notes are available as linked attachments. [Notebook layout](../../docs/ARCHITECTURE.md#notebook-layout) · [Interface reference](INTERFACES.md)

## Storage and Compatibility

| Path | Contents |
| --- | --- |
| `<project>/KNOWLEDGE.md` | Shared notebook with fact statements, research entries, and attachment links |
| `<project>/knowledge/proofs/<fact_id>.md` | Proof attachments indexed by fact ID |
| `<project>/knowledge/notes/` | Optional intuition, longer notes, and verification feedback attachments |
| `<worker>/local_memory/` | Drafts maintained by that worker |

New projects use `hyprover-knowledge-v1`. If there is a `project.json`, its `knowledge_format` must match the notebook. Existing notebooks in the earlier v1 format remain readable and writable with their original storage markers. To continue work from a `fact_graph` or `global_memory` project, create a new project and keep the original directory as an archive.

Display IDs use a letter for the record type, as in `F001` for a fact, `O001` for an overview, `R001` for a route, `C001` for a conjecture, `E001` for experience, `L001` for a lead, `P001` for a report, and `V001` for feedback. The store assigns display IDs under the write lock and retains them through edits and revocations. Proof dependencies and edit parameters use internal IDs. Supported query parameters also accept display IDs.

Earlier notebooks without `display_version` can be read as stored. A regular update writes display version 1. Use `refresh_display(titles=..., expected_revision=...)` to refresh display metadata or update fact titles against a notebook revision. A refresh that changes display metadata increments only the notebook revision. Record revisions, attachment paths, and anchors remain valid.

## Facts and Proofs

A `FactRecord` contains a complete statement and immutable metadata. Include the quantifiers, hypotheses, definitions, parameter ranges, and conclusion so later proofs can check when the fact applies.

| Field | Meaning |
| --- | --- |
| `fact_id` | Content ID represented as 16 lowercase hexadecimal characters |
| `problem_id`, `author` | Problem identifier and submitting author |
| `statement` | Complete mathematical statement |
| `predecessors`, `glossary_introduces` | Prerequisite facts and new notation definitions. Proof dependencies form a directed acyclic graph |
| `proof_path`, `intuition_path` | Relative paths to the proof and optional intuition attachment |
| `external_refs`, `verification` | Structured bibliography and verification record for publication |
| `status` | `verified` or `revoked` |
| `revision`, `created_at`, `revocation`, `supersession` | Record revision, creation time, and revocation and supersession records |
| `display_id`, `title` | Display ID and navigation title |

Snapshots and `get` return `FactRecord` objects. `proof` loads the attachment on demand and verifies its content ID. `read_fact` also loads any optional intuition note and assembles a complete `Fact` with `proof` and `intuition`. A `Fact` can also represent a proof candidate awaiting verification. The project glossary combines the `glossary_introduces` mappings of valid facts. The store checks for conflicting definitions of the same name.

`compute_fact_id` encodes the problem identifier, sorted predecessors, sorted glossary additions, and the statement and proof with whitespace collapsed. It takes the first 16 hexadecimal characters of the SHA-256 digest. The encoding uses UTF-8, sorted JSON keys, default separators, and literal non-ASCII characters. Whitespace normalization applies to the statement and proof only. Publication rejects duplicate predecessors. When reusing an ID, the store still checks the statement, proof, and dependencies. [Content ID implementation](schema.py)

The content ID excludes the author, intuition, bibliography, title, and lifecycle metadata. Common `external_refs` fields include `key`, `authors`, `title`, `arxiv`, `year`, `venue`, `doi`, and `cited_for`. `set_external_refs` updates the bibliography and increments the fact revision. Citations written in the body count as proof content. Metadata uses UTF-8 JSON with string object keys and finite numbers.

## Non-Fact Entries

A `KnowledgeEntry` includes `entry_id`, `kind`, `title`, `body`, `author`, `status`, `revision`, `links`, `details_path`, `attributes`, `created_at`, `updated_at`, and `display_id`. Internal IDs consist of `entry-` followed by 16 hexadecimal characters. Put a searchable summary in `body` and use `details` for longer material saved in an attachment. `links` associates existing records. `predecessors` specifies the premises used in a proof.

| Kind | Statuses | Purpose and maintainer |
| --- | --- | --- |
| `overview` | `current` | The main agent summarizes the fixed goal, known results, and unresolved gaps |
| `route` | `proposed`, `planned`, `active`, `parked`, `failed`, `completed` | The main agent maintains adopted routes. Workers can propose routes |
| `conjecture` | `open`, `proved`, `refuted`, `reopened` | A complete statement awaiting examination. Ordinary updates allow `open` and `reopened` |
| `experience` | `open`, `archived` | Reusable methods, applicability limits, or specific reviews |
| `lead` | `open`, `archived` | Literature, analogies, and research leads |
| `report` | `recorded`, `archived` | Progress, handoffs, and audits. Authors edit their own reports, and the main agent can archive other reports |
| `feedback` | `recorded` | Verification results and suggestions for repair written by the verification service |

When creating an experience entry or editing its body, links, or semantic attributes, set `attributes.subtype` to `method` for a technique, `boundary` for conditions and limits, or `review` for a specific question or error. Associate `review` entries with the records examined using the fields below. Ordinary progress reports may omit audit fields.

| Field in each `attributes.audits` item | Meaning |
| --- | --- |
| `target_id` | Target's internal ID. Include it in the entry's `links` as well |
| `revision` | Positive integer revision of the record examined |
| `scope`, `not_covered` | What the audit checked and what it left unchecked. Use `"none"` for `not_covered` when the entire scope is covered |
| `historical` | Set to `true` when examining an older revision |

The entry retains the audit baseline. Create a new linked entry when auditing a newer revision. Search flags records that have been revised, revoked, or superseded, so readers can tell which version the audit applies to. [Audit parameter example](../../docs/cli-and-tools.md#recording-and-editing)

When updating an entry, provide its `entry_id` and current `expected_revision`, and keep the entry kind. A creation retry returns the original record if the project, author, kind, `request_id`, and payload all match. A different payload for the same request returns a conflict. The service maintains `research_policy_version`, `record_request`, `resolution_fact_id`, `invalidated_by`, and `resolved_at`.

Earlier unclassified experience entries remain readable as stored. Status-only updates can omit the subtype. When editing the content under the current policy, choose an explicit subtype and supply a validated baseline for any earlier `audits`. Historical attributes without a policy version remain readable as stored. Create a new entry if a reserved field conflicts.

## Publication, Revocation, and Supersession

1. The worker sends the full statement, proof, and direct predecessors through `fact_submit`. The service checks the dependency closure and provides the independent session with the direct predecessors' statements, definitions, and proof paths.
2. The verification session examines the reasoning and the conditions under which it applies. An acceptable report has a `correct` verdict, empty `critical_errors` and `gaps` lists, and an empty `repair_hints` string.
3. The service acquires the write lock, rereads the notebook, and checks the content ID, valid predecessors, and conjecture revision before writing the proof and notebook. A publication receipt contains `accepted=true`, `published=true`, and a valid `fact_id` together. An exact retry of the same published candidate is supported. [Submission interface](INTERFACES.md#fact_submit)

When a submission names a conjecture as its source, the service checks the source before verification and again before committing. The conjecture must be `open` or `reopened`, its revision must match, and its body must exactly match the candidate statement after collapsing whitespace. Publication sets the conjecture to `proved`, increments its revision, and records `resolution_fact_id` and the fact association.

`fact_review` starts a new session to assess a specific concern. Under the write lock, the service rechecks the target revision, valid evidence, and referenced proofs. The committed change records the requester, rationale, request and run identifiers, and evidence revisions. The service writes verification feedback separately. An exact retry of an already committed request is supported.

- `revoke` marks the target and valid facts that depend on it as `revoked`, increments their revisions, and retains the proofs in the history. Entries that reference those facts receive `invalidated_by`. Conjectures resolved by those facts become `reopened`. The main agent can use these records to revisit routes and articles.
- `supersede` gives priority to an independently published stronger fact covering every hypothesis and the full scope of the earlier result. The service checks the replacement's independence and rejects supersession cycles. `supersession` stores `replacement_id`, `reason`, `superseded_at`, and `review`. The earlier fact remains `verified`, and existing proofs retain their original predecessors. If the replacement or its implication evidence is revoked, search can recommend the earlier fact while it remains valid. [HTTP review endpoint](../verify/README.md#post-verifyreview)

## Snapshots and Concurrency

`KnowledgeSnapshot` stores `problem_id`, `revision`, `facts`, `entries`, `display_version`, and `storage_format`. Snapshots and their records are immutable, so related reads can share one snapshot. `active_facts()` returns valid facts. `topological(targets)` returns the targets' dependency closure in dependency order.

| Revision read | Subsequent use |
| --- | --- |
| Notebook `snapshot_revision` | Pass as `expected_snapshot_revision` when paging. Restart the query after a notebook change |
| Entry `revision` | Pass as `expected_revision` when editing, or as `source.revision` when submitting a proof for a conjecture |
| Fact `revision` | Pass as the target revision for `fact_review`. Audit entries can reference revisions of both facts and research entries |

`search` returns entry summaries or full fact statements. `search_page` also returns the page position, total count, and snapshot revision. An empty query browses by kind and time. Search can locate a record by internal ID or display ID. Other text queries apply literal filtering. [Browsing and pagination examples](../../docs/cli-and-tools.md#search-and-pagination)

The store rereads the snapshot under `knowledge/.write.lock`, validates the current revisions, writes attachments, and atomically replaces `KNOWLEDGE.md`. The notebook records the committed state. An interruption can leave unreferenced attachments. A retry syncs visible files and their directories again. A successful write preserves existing attachment contents. The store rejects path escapes and symbolic links. [Python interface](INTERFACES.md#python-interface)

If an atomic replacement succeeds but the durability check reports an error, the publication service rereads the fact, proof, verification run, and conjecture association. If it can see the same commit, it may report success with a `durability_warning`. Feedback is written separately. A failed feedback write is reported as `feedback_error`.
