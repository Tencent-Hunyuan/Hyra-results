# Configuration reference

Configure models, services, and research permissions through environment variables or project settings. New projects default to the `gpt-6-astra` model, `max` reasoning effort, and `max:8` workers. [Installation guide](getting-started.md)

## Configuration load order

```text
config/codex.env → config/hyprover.env → runtime/runtime.env → defaults for unset variables
```

Later files can override earlier values, including variables already exported in the shell. Set model access in `config/codex.env` and deployment overrides in `config/hyprover.env`. The installer generates `runtime/runtime.env` with machine-specific paths. The wrappers in `bin/` load these files. When running a Python module directly, first run `. ./scripts/env.sh` from the repository root. Commit only the `*.env.example` templates.

## Models and backends

| Setting | Default or purpose |
| --- | --- |
| `CODEX_BACKEND` | Defaults to `api` for the Responses API. Use `chatgpt` for Codex account sign-in |
| `CODEX_API_BASE_URL` | API base URL without `/responses` |
| `HYPROVER_CODEX_API_KEY` | API key read from the environment at runtime |
| `CODEX_API_MODEL` | Defaults to `gpt-6-astra` and is used for probes in API mode |
| `HYPROVER_MAIN_MODEL` / `HYPROVER_MAIN_EFFORT` | Default research model and reasoning effort. See the precedence below |
| `HYPROVER_WORKER_MODEL` / `HYPROVER_WORKER_EFFORT` | Model for new workers and reasoning-effort override at runtime |
| `HYPROVER_VERIFY_MODEL` / `HYPROVER_VERIFY_EFFORT` | Verifier model and reasoning effort. Defaults to the research settings |
| `HYPROVER_CODEX_BIN` | Path to the Codex executable. Resolution checks the repository wrapper before `PATH` |
| `CODEX_HOME` | Defaults to `codex-home/` under the deployment runtime directory |

### API and account sign-in

The installation guide covers setup and switching backends. Use these variables to override Codex's defaults for API retries and streaming timeouts. After changing them, rerun `bash scripts/setup-codex.sh api`. [Enable the selected backend](getting-started.md#4-enable-the-selected-backend)

| Variable | Purpose |
| --- | --- |
| `CODEX_API_REQUEST_MAX_RETRIES` | Maximum request retries |
| `CODEX_API_STREAM_MAX_RETRIES` | Maximum streaming connection retries |
| `CODEX_API_STREAM_IDLE_TIMEOUT_MS` | Streaming idle timeout in milliseconds |

Manuscript generation also uses the Responses API. If a deployment uses account sign-in for research, you can select the API backend for the manuscript process alone. [API setup for manuscript generation](article.md#generate-a-manuscript) · [Manuscript budgets](article.md#set-budgets-and-sources)

### Model precedence and worker count

- The shared model is resolved in order from `HYPROVER_MAIN_MODEL`, its compatibility alias `HYPROVER_CODEX_MODEL`, `CODEX_API_MODEL`, and the built-in default. The `env.sh` script provides the `CODEX_API_MODEL` fallback. Shared reasoning effort is resolved in order from `HYPROVER_MAIN_EFFORT`, `HYPROVER_CODEX_EFFORT`, and `max`. Independent verification gives its own model and effort variables priority.
- Interactive `./bin/codex` uses `HYPROVER_MAIN_MODEL`. Codex configuration such as the repository's `.codex/config.toml` controls reasoning effort. The `codex exec` command retains model arguments passed by the caller.
- When creating a project, use an option such as `--roles max:2` to change the worker count or configure a mix of reasoning efforts. The model comes from `--model` first, then `HYPROVER_WORKER_MODEL`, then the shared model. Project creation writes the model and effort for each worker to its `.role` file. Existing workers read their model from `.role`, and `HYPROVER_WORKER_EFFORT` can override their effort.

## Paths and services

| Variable | Default | Purpose |
| --- | --- | --- |
| `HYPROVER_RUNTIME` | Repository `runtime/` directory | Deployment runtime directory. Bootstrap still installs dependencies in the repository |
| `HYPROVER_AGENTS_ROOT` | `$HYPROVER_RUNTIME/projects` | Root directory for research projects |
| `VERIFIER_RESULTS_DIR` | `$HYPROVER_RUNTIME/verify-runs` | Verification run records |
| `HYPROVER_PY` | Python from the deployment's virtual environment, the configured path, or `python3` | Interpreter for the engine and services |
| `VERIFY_PORT` / `DASHBOARD_PORT` | `8091` / `8099` | Verification service and dashboard ports |
| `VERIFY_HOST` | `127.0.0.1` | Bind address for the verification service |
| `HYPROVER_VERIFY_URL` | `http://127.0.0.1:${VERIFY_PORT}/verify` | Verification service URL. The gateway sends publication and review requests to `/verify/fact` and `/verify/review`, respectively |

When changing a port, update the service, its clients, and its probes. Use distinct ports for deployments on the same host. [Port settings for multiple deployments](operating-guide.md#services)

## Wait times, retries, and time limits

Times are in seconds. Restart the affected processes after changing these settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HYPROVER_MAX_ROUNDS` | `0` | Maximum rounds per worker process. Use 0 for no limit |
| `HYPROVER_ROUND_HARD_TIMEOUT` | `14400` | Timeout for one worker round. Use 0 for no limit |
| `HYPROVER_MAX_CONSEC_FAILURES` | `5` | Consecutive worker failure limit. Use 0 for no limit. Timeouts do not count |
| `HYPROVER_ROUND_BEAT` | `5` | Default wait before retrying a failed worker round |
| `HYPROVER_RETRY_INTERVAL` | Workers inherit `ROUND_BEAT`, the background main agent defaults to `30` | Wait after a failure. Workers allow 0. The background main agent requires a positive value |
| `HYPROVER_MAIN_INTERVAL` | `300` | Wait between background main-agent sessions |
| `HYPROVER_MAIN_MAX_CONSEC_FAILURES` | `5` | Consecutive main-agent failure limit. Use 0 for no limit |
| `CODEX_TIMEOUT_SECONDS` | `900` in the verification service entry point, `0` in the library | Timeout for verifier subprocesses. Use 0 for no limit |
| `HYPROVER_VERIFY_TIMEOUT` | `3600` | Verification HTTP timeout. Use 0 for no limit |
| `HYPROVER_MCP_TOOL_TIMEOUT` | `3600` | MCP tool timeout for workers and the background main agent. Requires a positive value |

A worker enters `waiting` after a successful round. Every three seconds, the host checks locally for changes to assignments, official approaches, the main agent's overview, facts, and substantive verification feedback. A relevant change starts the next round. Routine reports, logs, and rewrites with identical content leave the worker waiting. Failure retry settings are independent of this polling interval. When setting verification timeouts, account for generation, HTTP, and MCP, and give the outer layers enough time to wait. The operator manages the spending budget separately. [Execution module](../hyprover/execution/README.md)

## arXiv search load control

The built-in adapter shares its cache and request rate limit within a runtime directory. You can replace arXiv with a different literature source or use Codex's built-in search. [Literature search guide](literature-search.md) · [arXiv caching and rate limits](../hyprover/integrations/README.md)

## Research permissions

- Set `HYPROVER_LITERATURE_SEARCH=0` before starting a session to disallow external search. Managed sessions disable web search automatically. For interactive sessions, disable it explicitly. [Run without external search](literature-search.md#run-without-external-search)
- The `allow_math_computation` setting in `project.json` defaults to `false`. Set it to `true` only when the operator explicitly authorizes computation.
- Set `allow_early_completion` to `false` in `project.json` to send the background main agent's completion candidate to the operator for review while research continues. [Review facts and close a project](operating-guide.md#review-facts-and-close-a-project)
