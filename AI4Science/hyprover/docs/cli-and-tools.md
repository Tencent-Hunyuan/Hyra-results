# CLI and MCP Tools

Run `./bin/hyprover` from the repository root. The command loads `scripts/env.sh`.

## Project Commands

| Command | Purpose |
| --- | --- |
| `hyprover list [--json]` | List projects and active workers |
| `hyprover new <project> [--roles max:8] [--model M]` | Create a project and configure its workers |
| `hyprover main <start\|status\|stop> <project>` | Manage the background main agent |
| `hyprover assign <project>/<worker> (--task "…" \| --file P \| --stdin)` | Replace that worker's `TASK.md` |
| `hyprover start <project>[/<worker>]` | Start worker loops |
| `hyprover status <project>[/<worker>] [--json]` | Show runtime status, round counts, and recent activity |
| `hyprover stop <project>[/<worker>] [--force]` | Stop at the end of the current round. `--force` attempts to stop the process group immediately |
| `hyprover article <generate\|status\|resume\|stop\|compile\|recheck> …` | Write, review, or compile manuscripts |

`start` validates the project configuration and notebook format before launching. New projects use `hyprover-knowledge-v1`. Existing notebooks with the original v1 identifier continue to be read and written in their original format. `stop` can also stop workers still running on older projects. In `status`, `waiting` means a worker is waiting for a change in relevant input. `stuck?` flags an unusually long round for investigation. [Operations Guide](operating-guide.md) · [Article Generation](article.md)

## MCP Tools

`HYPROVER_ROLE` determines which tools a session can use. The main agent and workers can record, search, and request fact reviews. Workers submit proofs, and the main agent can revoke facts. When literature retrieval is permitted, the main agent, workers, and verifiers can search the literature. The verifier's MCP tool is limited to literature search.

A main agent without a bound project directory selects the project with `project`. Workers use the project and author bound at startup. See the literature guide for other ways to connect search providers. [Literature Search Guide](literature-search.md)

| Tool | Parameters (`?` means optional) | Purpose |
| --- | --- | --- |
| `knowledge_record` | `kind, title, body, entry_id?, expected_revision?, status?, links?, details?, attributes?, request_id?, project?` | Create or update a research record |
| `knowledge_search` | `query?, limit?, kinds?, subtype?, author?, linked_to?, include_archived?, include_superseded?, sort_by?, offset?, expected_snapshot_revision?, project?` | Browse and filter the notebook |
| `fact_submit` | `statement, proof, predecessors?, glossary_introduces?, intuition?, external_refs?, title?, source?` | Submit a proof for independent verification and publication |
| `fact_revoke` | `fact_id, reason, project?` | Revoke a fact in an emergency with operator authorization |
| `fact_review` | `fact_id, action, reason, expected_revision, replacement_id?, evidence_fact_ids?, project?` | Request an independent review of a proposed revocation or replacement |
| `search_arxiv_theorems` | `query, num_results?` (1–20) | Find arXiv paper metadata and abstracts |

### Recording and Editing

- The main agent maintains `overview` and formal `route` records. Workers can write conjectures, experience, leads, reports, and proposed routes. Authors can edit their own reports. The main agent can archive reports written by others, and the verification service writes `feedback`.
- Edits use the internal `entry_id` and the record's `expected_revision`. When creating a record, you can pass a stable `request_id`. Reuse the key and the same payload when retrying the original request. The key makes creation idempotent for a given project, author, and record type. Reusing it with different content returns `record_request_conflict`.
- Put a searchable summary in `body` and longer explanations in `details`. Put the internal IDs of related records in `links`. [Experience Subtypes](concepts.md#experience-and-audits)
- Reports can record progress, handoffs, or audits. Use the `review` subtype of experience for concrete questions or errors concerning a specific target. Audit reports and `review` experience use `attributes.audits` to record the target's revision, the scope checked, and any areas left unchecked.

The following `knowledge_record` example records a proof audit. Replace the sample ID and revision with the ones you read.

```json
{
  "kind": "report",
  "title": "Check the scope of the section argument",
  "body": "Checked the conditions under which the proof applies to nonsmooth convex bodies. The final equality case still needs a separate check.",
  "links": ["0123456789abcdef"],
  "attributes": {
    "audits": [{
      "target_id": "0123456789abcdef",
      "revision": 2,
      "scope": "All hypotheses in the statement and the application to nonsmooth convex bodies in Section 2 of the proof",
      "not_covered": "Conditions for equality"
    }]
  }
}
```

Set `historical: true` for an old revision and `not_covered: "none"` when every relevant item was checked. Create a new record when reviewing a new revision so the original audit retains its baseline. The service maintains `research_policy_version` and `record_request`. [Data Model and Compatibility](../hyprover/core/DATA_MODEL.md#non-fact-entries)

### Search and Pagination

Start with an empty `query` to browse potentially relevant categories, then open linked attachments as needed. Enter an internal ID or display label to find a specific record. Other nonempty queries use a case-insensitive literal substring match against titles, statements, glossary terms, and record bodies. The search also supports non-English text, including Chinese. Judge relevance from the records themselves and return to category browsing if a literal query finds no match.

- Read relevant `report` and `feedback` records each round. Use `kinds=["fact"]` to find proof premises, `kinds=["report"]` to see shared progress, and `kinds=["feedback"]` to focus on verification results. Default results include feedback. Use `include_archived=true` for archived records and `include_superseded=true` for active facts with a recommended replacement.
- `subtype` restricts results to `experience`. If you also pass `kinds`, use `["experience"]`. You can also filter by `author` and `linked_to`. `linked_to` accepts an internal ID or display label.
- `limit` defaults to 10 and accepts values from 1 to 1,000. `sort_by` orders results from newest to oldest by update time (`updated`, the default) or creation time (`created`). The legacy value `relevance` is treated as `updated`.
- When `has_more` is true, start the next page at `offset + len(results)` and keep the same filters. Send the previous page's `snapshot_revision` as `expected_snapshot_revision`. If the service returns `search_snapshot_conflict`, start again from the first page.

For example, call `knowledge_search` with `{"query":"","kinds":["report"],"limit":20}`. If it returns 20 results with `snapshot_revision=41` and `has_more=true`, add `offset=20` and `expected_snapshot_revision=41` to the same arguments for the next page. The main agent should supply the same `project` where applicable.

Each row includes `kind`, `id`, `is_fact`, and `link`, along with the full fact statement or a record summary. Display labels such as `F001` and `C001` help locate records. Use internal IDs for edits and references. Pass arrays, objects, and positive integers as JSON values of the declared type. Use `null` to leave an optional value unset. The string `"null"` is ordinary text.

| Common error | What to do |
| --- | --- |
| `entry_revision_conflict` | Read the record again and edit against its current content and revision |
| `record_request_conflict` | Check whether you are retrying the original request or creating a new record |
| `search_snapshot_conflict` | Start from the first page using a fresh snapshot |

### Submitting Proofs

Submit the complete statement and proof. List the active facts actually used in `predecessors` and define new notation in `glossary_introduces`. The worker environment supplies the project and author. `title` accepts a single-line navigation title of up to 160 characters. Put the complete statement in `statement`.

When proving an existing conjecture, first read the full statement, then pass that result's `source_ref` unchanged as `source`. It contains the conjecture's internal ID and the positive integer revision that was read. Search pagination uses the separate `snapshot_revision` field. The example shows the shape of `source`. Submit the actual values returned by the search.

```json
{"id": "entry-0123456789abcdef", "revision": 3}
```

The service checks the conjecture's status, revision, and exact statement both before verification and at publication. Matching normalizes whitespace in the statement and conjecture body. If the revision is stale, read the conjecture again and check the proof against it. Omit `source` for an independent lemma. Legacy clients can pass both `source_id` and `source_revision` in place of `source`. The gateway translates the `source` object to those two HTTP fields.

Publication is complete when the same receipt contains `accepted: true`, `published: true`, and a 16-digit lowercase hexadecimal `fact_id`. If the receipt only has `correct`, check `write_error` or the publication status. An exact retry of an already published result can return `deduplicated: true` after the service confirms the publication. `durability_warning` means the committed record can be read but a persistence step failed. `feedback_error` means feedback was not recorded. See `raw_verification` or the linked feedback for the full model result. If a candidate is not published, continue investigating the conjecture or revise the proof.

### Reviewing Facts

Send `fact_review` with the target's current revision and a specific reason. For `revoke`, describe a proof error or substantive gap. An applied revocation withdraws the target and its dependent proofs. For `supersede`, provide a `replacement_id` and show that it covers the old conclusion under all the original hypotheses. Existing proofs remain valid. Supply independent facts through `evidence_fact_ids` if applicable. A change takes effect when the same receipt contains `accepted: true` and `applied: true`.

In an emergency, the main agent can use `fact_revoke` with operator authorization and inspect `revoked` for affected facts. Use the latest notebook to reassess the affected proof routes and any generated manuscripts.

See the references for more fields, errors, and HTTP interfaces. [Core Interfaces](../hyprover/core/INTERFACES.md) · [Verification Service](../hyprover/verify/README.md)

## Service Commands

| Service | Default address | Purpose |
| --- | --- | --- |
| verify | `127.0.0.1:8091` | One-off checks, proof publication, and fact review |
| dashboard | `127.0.0.1:8099` | Read-only project dashboard |

Use `scripts/services.sh` to start, check, or stop services. [Verification Service and Dashboard](operating-guide.md#services) · [Configuration](configuration.md)
