# Proof Verification and Fact Publication Service

The service starts an independent Codex session for a candidate proof or a review request. The verifier returns its assessment, and the service checks the publication conditions and updates the shared research notebook. Verification is an independent model review of a written proof. Important results still warrant human review. [System architecture](../../docs/ARCHITECTURE.md) · [Verifier role](../../agents/contracts/verifier.md)

## Startup and Configuration

```bash
bash scripts/services.sh up verify
bash scripts/services.sh status
```

After loading the deployment environment, you can also start the service in the foreground with `python -m hyprover.verify`. `GET /health` returns `status="ok"` and the service process's `pid`. The management scripts use the PID to distinguish deployments on the same host. [Start the services and dashboard](../../docs/operating-guide.md#services)

| Setting | Purpose and default |
| --- | --- |
| `VERIFY_HOST`, `VERIFY_PORT` | Listen on `127.0.0.1:8091` by default. If `VERIFY_PORT` is unset, the service can use `PORT` |
| `HYPROVER_AGENTS_ROOT` | Root directory for projects in publication and review requests |
| `VERIFY_AGENT_HOME` | Verification session working directory. The module defaults to `hyprover/verify/agent/` when run directly. If `AGENTS.md` is absent, it links the verifier instructions from the repository |
| `VERIFIER_RESULTS_DIR` | Where verification run records are stored. Defaults to `hyprover/verify/runs/` when unset. The management scripts default to `runtime/verify-runs/` |
| `HYPROVER_CODEX_BIN` | Codex executable |
| `HYPROVER_VERIFY_MODEL`, `HYPROVER_VERIFY_EFFORT` | Verifier model and reasoning effort. When unset, the service checks the main agent settings and then the shared defaults |
| `CODEX_TIMEOUT_SECONDS` | Time limit for one verification session. The module entry point defaults to 900 seconds. Library calls default to `0`, which disables the time limit |
| `HYPROVER_VERIFY_TIMEOUT` | HTTP request timeout for the MCP gateway. Defaults to 3600 seconds. `0` disables the timeout |

The gateway connects through `HYPROVER_VERIFY_URL`. Supply an HTTP or HTTPS base URL, optionally ending in `/verify` or `/verify/fact`. The gateway resolves the actual publication and review paths from that address. Omit the query string and fragment. [Full configuration](../../docs/configuration.md)

## Verification and Publication Endpoints

### `POST /verify`

Submit a nonblank `statement` and `proof`. This endpoint returns a review without writing to the notebook. A normal response includes `verdict`, `verification_report` (`summary`, `critical_errors`, and `gaps`), and `repair_hints`. Invalid input returns HTTP 422, session startup or output errors return 500, and a verification timeout returns 504.

### `POST /verify/fact`

| Request | Fields |
| --- | --- |
| Required | `project_dir`, `author`, `statement`, `proof` |
| Optional | `problem_id`, `predecessors` (list), `glossary_introduces` (definition map), `intuition`, `external_refs` (list of objects), `source_id`, `source_revision`, `title` (up to 160 characters) |

The service determines the problem identifier from the project notebook. `project_dir` must be an absolute path to a direct subdirectory of `HYPROVER_AGENTS_ROOT`, with no symbolic links in the path. Publication and review share this project boundary. An invalid project or author returns HTTP 400. An unconfigured project root returns 503. Missing or extra request fields and blank proof text return 422.

1. The service validates the candidate content ID, dependency closure, and source conjecture against one snapshot. To associate a conjecture through MCP, pass the search result's `source_ref` as `source`. The gateway turns it into the HTTP fields `source_id` and `source_revision`. The revision must be a positive integer. The conjecture body and candidate statement must agree after collapsing whitespace. Omit the source fields for an independent lemma.
2. A new session reviews the candidate proof and its predecessors. An acceptable verification report has `verdict="correct"`, a string `summary`, empty `critical_errors` and `gaps` lists, and an empty `repair_hints` string.
3. Under the write lock, the service rechecks that the dependencies and source revision remain valid, then commits the fact and sets the associated conjecture to `proved`. If committing fails, consult the receipt. The original conjecture remains available for another attempt.

| Receipt | Meaning |
| --- | --- |
| `accepted=true`, `published=true`, 16-character lowercase hexadecimal `fact_id` | Publication is confirmed. Retrieve the fact by ID |
| `accepted=false`, `published=false`, `fact_id=null` | Use `stage`, `verdict`, `verification_report`, and `error` to locate the failure |
| `write_error` | Writing failed after verification. The response can still include `verdict="correct"` |
| `deduplicated=true` | The service found a valid fact with the same content and reused its publication record |
| `durability_warning` | The service reread the same visible commit after the durability check reported an error |
| `feedback_error` | The feedback write failed. Consult the publication receipt for fact status |

A response can also contain `run_id`, `feedback_id`, `repair_hints`, and `raw_verification`. Through `fact_reviews`, the verifier can propose up to three reviews about the dependency closure. The service handles those reviews separately before writing the candidate, then validates its predecessors against the latest notebook. [Publication fields and tool calls](../core/INTERFACES.md#fact_submit)

<a id="recheck-published-facts"></a>

## Review Published Facts

### `POST /verify/review`

The request requires `project_dir`, `author`, `fact_id`, `action`, `reason`, and a positive integer `expected_revision`. The `supersede` action also requires `replacement_id`. Use `evidence_fact_ids` to provide other valid facts. The evidence dependency closure must exclude the target fact and its dependents. The review includes the proof under dispute as the object of the review.

| Action | Effect after applying |
| --- | --- |
| `revoke` | Revokes the target and its valid dependents, reopens conjectures resolved by those facts, and retains the original attachments in the history |
| `supersede` | Recommends an independently published stronger fact covering every hypothesis and the full scope of the earlier result. The earlier fact remains valid, and existing proofs retain their original predecessors |

After independent verification approves the request, the service rechecks the target, evidence revisions, and proof content, then commits under the write lock. Confirm the change with `accepted=true` and `applied=true`. Use `stage`, `verdict`, and `error` to diagnose changes that were not applied. The receipt can also include `fact_id`, `action`, `replacement_id`, `revoked`, `request_id`, `run_id`, and `verification_report`. An exact retry can return the committed review receipt. [Fact review records](../core/DATA_MODEL.md#publication-revocation-and-supersession)
