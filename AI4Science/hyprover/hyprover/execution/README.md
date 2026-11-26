# Worker Workspaces and Execution Loop

`hyprover.execution` initializes worker workspaces, runs each research round in a Codex session, and watches shared inputs to decide when to start the next round. The orchestration module controls projects and starts and stops processes. [Project operations guide](../../docs/operating-guide.md) · [Orchestration module](../orchestration/README.md)

<a id="workspaces-and-rounds"></a>

## Workspaces

`HYPROVER_AGENTS_ROOT` sets the project root. Its default is `runtime/projects/` under the working directory. Each project shares `PROBLEM.md`, `project.json`, `KNOWLEDGE.md`, and the `knowledge/` directory. New projects use the `hyprover-knowledge-v1` notebook format.

| File under `workers/<worker>/` | Purpose |
| --- | --- |
| `TASK.md` | Current assignment, updated with `hyprover assign` |
| `AGENTS.md` | Link to the worker role instructions |
| `.role`, `.codex/config.toml` | Worker identity, model, reasoning effort, and MCP configuration. Startup rewrites the MCP configuration to bind the project, author, and current Python interpreter |
| `local_memory/` | Drafts, attempts, and material the worker keeps for subsequent rounds |
| `.status.json`, `logs/` | Process and round status, along with output for each round |

<a id="input-changes-and-wakeups"></a>

## Research Rounds and Waiting

1. At startup, the host validates the project notebook and reads the assignment and research inputs. Once the inputs are available, it runs the first `codex exec` round. The worker uses its bound identity to access the notebook and submit proofs through MCP.
2. After a successful round, the worker enters `waiting`. Every three seconds the host checks the task text, the main agent's overview, adopted routes, facts, and substantive verification feedback. The host calls the model when these inputs change. Ordinary reports, proposed worker routes, logs, and rewrites with the same text do not trigger another round on their own.
3. The host compares against the inputs collected before the current round started. A substantive change during that round can therefore trigger the next one. If reading the inputs fails, the host retains the last valid baseline, records `input_error`, and continues checking locally. The status also includes `waiting_since`, `wake_reason`, and `input_fingerprint`.
4. A failed round is retried according to `HYPROVER_RETRY_INTERVAL`, or `HYPROVER_ROUND_BEAT` when the first setting is absent. The round limit, deadline, stop requests, and consecutive failure policy control when execution ends. [Time limits and retries](../../docs/configuration.md#wait-times-retries-and-time-limits)

The execution loop detects publication results in the round log. It records a fact in status only when one JSON receipt contains `accepted=true`, `published=true`, and a valid 16-character hexadecimal `fact_id`. [Submitting proofs](../../docs/cli-and-tools.md#submitting-proofs)

<a id="stop-and-restart"></a>

## Stop and Resume

`hyprover stop PROJECT/WORKER` requests that the worker stop at the end of its round. A forced stop targets the worker's separate process group. Use `status` to confirm that the old process has exited before starting the next round in the same workspace. Stop requests remain available when the notebook is damaged. [Stop and resume a project](../../docs/operating-guide.md#stop-and-resume)

<a id="tests"></a>

## Source Files and Tests

| File | Responsibility |
| --- | --- |
| `layout.py` | Resolve directories and validate the notebook and startup conditions |
| `scaffold.py` | Create workspaces, generate MCP configuration, and launch processes |
| `loop.py` | Run Codex sessions and record timeouts, stops, and retries |
| `wakeup.py` | Compute fingerprints for detecting input changes |

`python -m pytest hyprover/execution/` checks initialization, round execution, waiting, and stopping with simulated Codex processes.
