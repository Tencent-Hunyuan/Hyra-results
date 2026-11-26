# Research Notebook and Verification Interfaces

This page lists MCP tool parameters and receipts, as well as the Python interface used by the implementation. The MCP session configuration binds the role. The verification service commits fact publication and review changes. [Architecture](../../docs/ARCHITECTURE.md) · [Data model](DATA_MODEL.md) · [CLI and MCP examples](../../docs/cli-and-tools.md)

## MCP Tools and Roles

| Role | Tools |
| --- | --- |
| `worker` | `knowledge_record`, `knowledge_search`, `fact_submit`, `fact_review`, `search_arxiv_theorems` |
| `main` | `knowledge_record`, `knowledge_search`, `fact_revoke`, `fact_review`, `search_arxiv_theorems` |
| `verifier` | `search_arxiv_theorems` |

The deployment configuration controls literature search. Workers use their bound project and author identity. The main agent can select a `project` where the tool supports it. [Gateway and environment variables](../gateway/README.md)

### `knowledge_record`

| Use | Fields |
| --- | --- |
| Required | `kind`, `title`, `body` |
| Optional | `entry_id`, `expected_revision`, `status`, `links`, `details`, `attributes`, `project`, `request_id` |
| Returns | `id`, `entry_id`, `kind`, `display_id`, `title`, `is_fact: false`, `revision`, `status` |

When updating an entry, provide its internal `entry_id` and current `expected_revision`. Read the current record if there is a revision conflict. Use `request_id` when retrying creation. Given the same project, author, kind, and request ID, an unchanged payload returns the original record and a different payload returns `record_request_conflict`.

The main agent maintains the overview and adopted routes. Workers may propose routes and record conjectures, experience, leads, and reports. Authors edit their own reports. The main agent can archive other reports. The verification service writes `feedback`. [Experience, audits, and links](DATA_MODEL.md#non-fact-entries)

### `knowledge_search`

| Parameter | Purpose |
| --- | --- |
| `query=""` | An empty query browses by kind. A nonempty query first tries an internal ID or display ID, then applies a case-insensitive literal substring filter |
| `kinds`, `subtype` | Select record kinds and filter experience by subtype. Use `kinds=["report"]` for reports |
| `author`, `linked_to` | Filter by author and associated record. `linked_to` accepts an internal ID or a display ID |
| `include_superseded=false`, `include_archived=false` | Include superseded facts or archived entries. An exact ID match can still locate a superseded fact while it remains valid. Kind and archive filters still apply |
| `sort_by="updated"` | Sort newest first by update time or, with `created`, by creation time. The earlier `relevance` value is accepted and reported as `updated` |
| `limit=10`, `offset=0`, `expected_snapshot_revision` | Pagination and snapshot guard. `limit` accepts 1–1000 |
| `project` | Select a project as the main agent |

Returns `query`, `results`, `total`, `offset`, `limit`, `has_more`, `sort_by`, and `snapshot_revision`. For the next page, use `offset + len(results)` and the snapshot revision from the first page. Restart from the first page if the query returns `search_snapshot_conflict`.

Results include a fact statement or research summary along with `id`, `kind`, `is_fact`, `link`, `display_id`, `title`, and `display_name`. Facts also have `predecessor_labels`, `preferred_fact_id`, and `superseded`. Entries include `related_labels`, `invalidated_links`, and any applicable audit notices. Default queries include feedback. Use `kinds=["fact"]` to look for valid premises and follow attachments for complete proofs or notes.

The MCP result for a conjecture also includes `source_ref={"id": entry_id, "revision": revision}`. The revision belongs to that conjecture entry and can bind a subsequent submission to it.

### `fact_submit`

Provide `statement` and `proof`. Optional fields are `predecessors`, `glossary_introduces`, `intuition`, `external_refs`, `title`, and `source`. The optional title is a single line of up to 160 characters for navigation.

When proving an existing conjecture, pass the returned `source_ref` unchanged as `source`. The object accepts only the internal `id` and a positive integer `revision`. Numeric strings and booleans are invalid revisions. Omit the source for an independent lemma. For compatibility, clients may supply `source_id` and `source_revision` together as the alternative form. The gateway converts the source object into these two HTTP fields. The verification service checks the conjecture's status, revision, and statement after whitespace normalization. If the revision is stale, reread the conjecture and confirm the proof still applies.

Publication is confirmed by a receipt containing `accepted=true`, `published=true`, and a valid 16-character hexadecimal `fact_id`. An unconfirmed publication returns `accepted=false`, `published=false`, and `fact_id=null`. `write_error`, `feedback_error`, and `durability_warning` describe issues with committing the fact, recording feedback, or confirming durability. Refer to `raw_verification` for diagnostics. An exact retry can return `deduplicated=true`. [HTTP request and receipt](../verify/README.md#post-verifyfact)

### `fact_review`

The main agent and workers can pass `fact_id`, `action` (`revoke` or `supersede`), `reason`, and `expected_revision`, with `replacement_id` and `evidence_fact_ids` as needed. The main agent can also supply `project`.

The gateway calls `POST /verify/review` and reads back the committed review from the notebook. `accepted=true` and `applied=true` confirm the change. A replacement must be a stronger, independently published fact. The review includes the proof under dispute, and the evidence dependency closure excludes the target and its dependents. [HTTP review endpoint](../verify/README.md#post-verifyreview)

### `fact_revoke`

The main agent uses this tool for emergencies authorized by the operator. Pass `fact_id`, `reason`, and optionally `project`. The response is `{"revoked": [...]}` and includes the target and any facts revoked along with it. Use `fact_review` for routine corrections.

### `search_arxiv_theorems`

Accepts `query` and `num_results`, which defaults to 10 and accepts 1–20. Returns paper metadata and abstracts. A failed search returns an error. `HYPROVER_LITERATURE_SEARCH=0` disables the tool. [Literature search and source review](../../docs/literature-search.md)

## HTTP Verification Service

| Endpoint | Purpose |
| --- | --- |
| `POST /verify` | Review a `statement` and `proof`, and return a verdict, report, and repair hints |
| `POST /verify/fact` | Verify and publish a candidate. Requires `project_dir`, `author`, `statement`, and `proof` |
| `POST /verify/review` | Independently check the rationale for revocation or supersession and commit any approved change |
| `GET /health` | Return service status and the process ID |

The service documentation lists complete request fields, responses, and error codes. [HTTP verification endpoints](../verify/README.md)

## Python Interface

`KnowledgeStore(project_dir)` opens an existing project. `KnowledgeStore.initialize(project_dir, problem_id=None)` initializes a new notebook and attachment directories. Initialization refuses to overwrite an existing notebook or older layout.

| Method | Purpose |
| --- | --- |
| `snapshot()` | Read an immutable snapshot |
| `list()`, `exists(fact_id)` | List valid facts or check whether a fact is valid |
| `get(fact_id, include_revoked=False)` | Read a `FactRecord` or return `None` |
| `proof(fact_id, snapshot=None)`, `read_fact(fact_id, snapshot=None)` | Load and check the proof on demand. `read_fact` assembles a complete `Fact` |
| `predecessors(fact_id)`, `descendants(fact_id)` | Inspect direct predecessors or valid transitive dependents |
| `glossary()` | Read the glossary assembled from valid facts |
| `external_refs(fact_id)`, `set_external_refs(fact_id, external_refs)` | Read or update structured bibliographic references |
| `search(...)`, `search_page(...)`, `entry_detail(entry_id)` | Return summaries, return a page of results, or load an entry and its attachment on demand |
| `refresh_display(titles=None, expected_revision=None)` | Maintain display metadata against a notebook revision. Changes increment the notebook revision, while record revisions and attachments are preserved |
| `record(kind, title, body, author, role, ...)` | Create a non-fact entry or update it against a revision |
| `publish(candidate, verification, source_id=None, source_revision=None)` | Atomically publish a fact within the verification service |
| `revoke(fact_id, reason)`, `apply_review(review)` | Revoke a fact and its dependents or commit a reviewed change |

Use `KnowledgeSnapshot.active_facts()` and `topological(targets=None)` to query dependencies. Share a snapshot across related reads. Use `plain` or a record's `.to_dict()` to obtain a serializable copy.

Proof attachments are addressed by fact ID. Longer notes usually use `<entry_id>-<revision>.md`. If an interruption leaves a file with the same name but different content, the new filename adds a 16-character content digest. An exact retry syncs visible files and their directories again. [Storage and concurrency](DATA_MODEL.md#snapshots-and-concurrency)
