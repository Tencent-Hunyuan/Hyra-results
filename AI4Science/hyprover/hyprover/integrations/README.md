# arXiv Literature Search Adapter

`arxiv.py` queries the public arXiv Atom API using the Python standard library and returns paper metadata and abstracts. The main agent, workers, and verifiers can use `search_arxiv_theorems` when search is enabled. The project guide explains how to switch to another index, a private database, or Codex web search. [Literature search and source review](../../docs/literature-search.md)

<a id="queries-and-results"></a>

## Query and Response

| Field | Rules |
| --- | --- |
| `query` | 1–1000 characters with at least one non-whitespace character. Accepts field syntax such as `ti:convex AND au:Gardner`. Plain keywords are searched across all fields using AND |
| `num_results` | An integer from 1 through 20, with a default of 10 |
| Response metadata | `query`, `count`, `results`, `endpoint`, `source="arxiv"`, `result_type="paper_metadata"`, and `is_verified=false`. Successful responses also include `cached` |
| Paper record | `title`, `authors`, `abstract`, `arxiv_id`, `published`, `updated`, `url`, and `pdf_url` |

The client waits up to 30 seconds by default, reads up to 2 MiB, and rejects XML containing `DOCTYPE` or `ENTITY`. Input, network, HTTP, and parsing failures return an empty result with an `error` field. Check `error` before interpreting the result count. Follow the source link to read the paper and check the theorem's hypotheses and conclusion.

`HYPROVER_LITERATURE_SEARCH=0` disables both client queries and the MCP tool.

<a id="rate-limits-and-recovery"></a>

## Shared Cache and Request Spacing

Processes using the same `HYPROVER_RUNTIME` share the request lock and cache in `arxiv-search-cache/`. Live requests are at least three seconds apart. Successful responses, including empty results, are cached for one hour, with a limit of 128 queries. Valid cache entries remain readable during a cooldown. Deployments using different directories manage request rates independently.

HTTP 429 and 503 start a shared cooldown, recorded in `cooldown.json`. The client uses a valid `Retry-After` value in seconds or as an HTTP date. Otherwise it backs off exponentially from 60 seconds to at most one hour. A successful live request resets the backoff counter.

| Diagnostic | How to interpret it |
| --- | --- |
| `search_busy` | The query missed the cache and another process holds the lock. Wait for `retry_after_seconds` |
| `http_error`, `http_status` | Check the HTTP status. A cache miss during cooldown also returns `retry_deferred=true` and the remaining wait. `retry_source` identifies the source of the backoff |
| `cache_warning=cache_full_write_skipped` | The live response is usable. Cache writes use atomic replacement. At capacity, the client only clears expired query files |
| `error_stage`, `error_type`, `cache_warning`, `cooldown_warning` | Use the stage to distinguish network, TLS, parsing, path access, and state write failures |
| `invalid_cooldown_state`, `cooldown_persistence_failed` | Cooldown data is damaged or a state write was interrupted. A `!` marker in `request.lock` identifies the interrupted write. After restoring storage, the operator can inspect and restore cooldown state under the same lock |

<a id="tests"></a>

## Offline Tests

`python -m pytest hyprover/integrations/` checks queries, rate limits, and recovery using simulated HTTP responses and temporary caches.
