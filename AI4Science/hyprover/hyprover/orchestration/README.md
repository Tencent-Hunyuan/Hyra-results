# Project and Main Agent Orchestration

`cli.py` handles project commands, `controller.py` schedules the background main agent and completion reviews, and `sequence.py` runs projects in a configured order. Invoke project commands through `./bin/hyprover` or `python -m hyprover.orchestration`. [Project operations guide](../../docs/operating-guide.md) · [CLI parameters](../../docs/cli-and-tools.md)

## Projects and Workers

`new PROJECT` creates `project.json`, an empty research notebook, directories for proofs and notes, and the worker workspaces. Fill in the project's `PROBLEM.md` and add `RUN_POLICY.md` as needed. Use an option such as `--roles max:2` to set the worker count, and `--model` to set the worker model. [Model and worker settings](../../docs/configuration.md)

`assign PROJECT/WORKER` replaces that worker's `TASK.md`. Supply the task through `--task`, `--file`, or `--stdin`. `start PROJECT[/WORKER]` uses a startup lock to prevent duplicate instances and archives the worker's previous end-of-round stop marker. Assignment and startup validate the notebook format. Use `list` to view projects, `status` to check processes and rounds, and `stop` to request a stop at the end of a round. Stop requests remain available when the notebook is unreadable. [Worker workspaces and execution loop](../execution/README.md)

## Background Main Agent

Start the verification service first. All project processes use the same `HYPROVER_AGENTS_ROOT`, and each connects to its model backend according to the deployment configuration.

```bash
./bin/hyprover main start demo
./bin/hyprover main status demo
./bin/hyprover main stop demo
```

The same three subcommands are available through `python -m hyprover.orchestration.controller`. Each project's background main agent runs in a separate process with a project lock to prevent duplicate starts. Subsequent sessions continue from the shared notebook and the previous decision.

| Main agent state or decision | Behavior |
| --- | --- |
| `continue` | The host enters `waiting` and starts the next round at `next_pass_at`. The default interval is `HYPROVER_MAIN_INTERVAL=300` seconds |
| Session failed | The host enters `retrying` and tries again after `HYPROVER_RETRY_INTERVAL`, which defaults to 30 seconds. When consecutive failures reach `HYPROVER_MAIN_MAX_CONSEC_FAILURES`, the host enters `blocked` and stops the workers. The default limit is five failures |
| Main agent reports `blocked` | By default, the host stops the project's workers. With the consecutive failure limit set to `0`, it treats the block as a failure and continues retrying |
| `complete` | Independently review the complete proof dependencies and reasoning for the original target, recheck that the evidence remains current, then stop the workers and write `main/result.json`. With `allow_early_completion=false`, the host records the completion candidate for operator review |

`main/status.json` records rounds and status, `main/supervisor.log` contains host logs, and `main/logs/` contains session logs and decisions. If a process has vanished while its recorded state is still active, `main status` reports `interrupted`.

`main stop` records a persistent stop request and arranges for that project's workers to exit. The verification service remains available to other projects. Once startup conditions are met, `main start` clears the main agent's own stop marker and resumes from existing state. After an unexpected exit, check for remaining processes, stop markers, and the project deadline before resuming. [Stop and resume a project](../../docs/operating-guide.md#stop-and-resume)

<a id="serial-queue-and-tests"></a>

## Serial Project Queue

`python -m hyprover.orchestration.sequence start queue.json` starts the queue. Use `status`, `stop`, or `resume` with the same file to inspect the queue, request a stop, or resume an eligible queue. This entry point relies on Linux process identity checks. The queue configuration lists `stages` in order. Each stage specifies an initialized `project`, `time_limit_seconds`, and `worker_count`. Each project uses its own `run-settings.json`, verification port, working directory, and run log directory. See all options with `python -m hyprover.orchestration.sequence --help`.

## Offline Tests

`python -m pytest hyprover/orchestration/` tests assignment, process control, main agent state, and completion using simulated Codex sessions and temporary projects.
