# Project operations guide

After installing and configuring HyProver, run the following commands from the repository root. These examples use `demo` as the project name and `max` as the worker name. [Installation guide](getting-started.md)

## Create a project

```bash
. ./scripts/env.sh
./bin/hyprover new demo
cp -n templates/project/PROBLEM.md "$HYPROVER_AGENTS_ROOT/demo/PROBLEM.md"
```

Fill in the project's `PROBLEM.md` using the template. Include the complete statement with its quantifiers and assumptions, definitions and notation, objectives and stopping conditions, background and references, and any research constraints. You can also record the budget, deadline, and additional constraints in the project's `RUN_POLICY.md`. [Problem template](../templates/project/PROBLEM.md)

The `new` command creates the project structure and eight workers. Use the start commands below to run them. For a first trial, use `--roles max:1` when creating the project to set up a single worker.

The default project directory is `runtime/projects/demo/`. It contains the project configuration in `project.json`, the shared research notebook in `KNOWLEDGE.md`, proofs and supporting notes in `knowledge/`, and each worker's assignment and status in `workers/`.

<a id="services"></a>

## Start the verification service and dashboard

```bash
bash scripts/services.sh up verify
bash scripts/services.sh up dashboard demo
bash scripts/services.sh status
bash scripts/services.sh logs verify -f
```

Start `verify` before starting workers. It independently verifies and publishes `fact_submit` submissions and listens on `127.0.0.1:8091` by default. The optional dashboard provides a read-only view of the project. It listens on `127.0.0.1:8099` by default and is accessible through SSH port forwarding.

The `services.sh` script uses `setsid` to detach services, allowing them to continue running after the SSH connection closes. Process IDs and startup requests are recorded in `$HYPROVER_RUNTIME/run/`, with logs in `$HYPROVER_RUNTIME/logs/`. To run a service in the foreground, execute its `start-*.sh` script directly.

When you run multiple deployments on one host, give each deployment distinct `VERIFY_PORT` and `DASHBOARD_PORT` values in `config/hyprover.env` and set `HYPROVER_VERIFY_URL` accordingly. The `services.sh status`, `services.sh test`, and `doctor.sh` commands compare the PID from `/health` with `runtime/run/verify.pid`. If they report `FOREIGN`, choose a different port for the current deployment.

## Choose how to coordinate research

### Run the main agent in the background

```bash
./bin/hyprover main start demo
./bin/hyprover main status demo
./bin/hyprover status demo --json
```

The main agent plans mathematical approaches, assigns tasks, and reviews progress. The background controller starts subsequent main-agent sessions at the interval set by `HYPROVER_MAIN_INTERVAL`, which defaults to 300 seconds. It handles retries and completion review. The project's `main/logs/` directory contains session logs.

Give the main agent the original problem, any established definitions, and available materials. It assigns intermediate claims that need checking and adapts the research plan based on published results and feedback from failed submissions. For example, if you already have a conditional construction, the next assignment could check whether the objects in the original problem satisfy its hypotheses.

### Coordinate work manually or interactively

Use `./bin/codex` to start an interactive main-agent session. You can also assign tasks directly.

```bash
./bin/hyprover assign demo/max \
  --task 'Read ../../PROBLEM.md and ../../KNOWLEDGE.md. Investigate one subgoal and submit a complete proof for verification.'
./bin/hyprover start demo/max
```

Each worker runs in its own directory. Use `assign --file` or `assign --stdin` for longer assignments. The `start demo` command starts all workers in the project. Use one coordination method per project so assignments do not overwrite one another.

## Run research and record results

1. Use `knowledge_search` to browse full fact statements, approaches, `report` entries, and verification `feedback`. Open proofs and longer notes as needed.
2. Use `knowledge_record` to save conjectures, experiences, leads, and progress updates. Supply the current `expected_revision` when editing an existing entry.
3. Workers use `fact_submit` to submit a complete statement, a proof, and the IDs of prerequisite facts. To prove an existing conjecture, copy its `source_ref` from the search result into `source` without modification. Standalone lemmas can omit `source`.
4. A verification receipt that includes `accepted=true`, `published=true`, and a valid `fact_id` confirms publication. Subsequent proofs can then cite the fact as a premise.

For all parameters and pagination options, see the [tool reference](cli-and-tools.md). To change sources or retrieval permissions, see the [literature search guide](literature-search.md). If a project requires computational checks, the operator sets `"allow_math_computation": true` in `project.json`.

## Check progress and waiting workers

Use `./bin/hyprover status demo --json` to inspect each worker and `./bin/hyprover main status demo` to inspect the background main agent. The dashboard displays the same project's shared research notebook and run status.

A worker enters `waiting` after finishing a round. About every three seconds, the host checks for changes to its assignment, the main agent's overview, official approaches, facts, and substantive verification feedback. It starts the next model call only after a relevant change. A round already in progress finishes first. Retry settings govern failures and timeouts. The `stuck?` indicator means the current round has run for more than 1.5 times its hard timeout. Inspect the process and logs to diagnose the delay. [Troubleshooting](#troubleshooting)

## Review facts and close a project

Use `fact_review` to request an independent review of a specific concern about an existing fact. Use `revoke` to revoke verification for the fact and its dependents. Use `supersede` to recommend an existing stronger result that applies under the original assumptions. The main agent can also use `fact_revoke` for emergency revocation with the operator's authorization.

When the background main agent proposes completion, the controller independently reviews each proof dependency and checks how the proofs establish the original objective. If the review passes and the evidence remains valid, it stops the workers and saves `main/result.json`. With `allow_early_completion` set to `false` in `project.json`, `main status` records the path to the completion candidate for human review and research continues. Review the complete argument yourself before relying on the results.

You can run [manuscript generation](article.md) separately after project completion or for selected published facts.

<a id="stop-and-resume"></a>

## Stop and resume a run

| Action | Command |
| --- | --- |
| Stop a worker at the end of its round | `./bin/hyprover stop demo/max` |
| Check whether the worker stopped | `./bin/hyprover status demo/max --json` |
| Resume one worker | `./bin/hyprover start demo/max` |
| Resume all workers under manual coordination | `./bin/hyprover start demo` |
| Stop the background main agent and its project workers | `./bin/hyprover main stop demo` |
| Resume the background main agent | `./bin/hyprover main start demo` |
| Stop the verification service | `bash scripts/services.sh down verify` |
| Stop the verification service and dashboard registered with the service manager | `bash scripts/services.sh down all` |

The background controller can also stop project workers when it reaches its retry limit or the main agent reports a blocker. To preserve work in the current round, first coordinate with the main agent to pause assignments, then request a graceful stop for each worker. Shut down the services after verification requests finish.

The `./bin/hyprover stop demo/max --force` command attempts to terminate the worker's process group immediately and may discard unsaved work. If the command reports `killed`, use the recorded PID to confirm that the old process and its children exited before restarting. Workers resume from their saved assignments and research notebook.

<a id="service-health"></a>

## Check the deployment

| Command | What it checks |
| --- | --- |
| `bash scripts/setup-codex.sh status` | API provider configuration or account sign-in status |
| `bash scripts/services.sh status` | Service status and which deployment owns the verifier process |
| `bash scripts/check-codex.sh` | In API mode, sends a real Responses request using `CODEX_API_MODEL` and inspects recent logs. In account mode, runs `codex login status`. Both modes write to `runtime/logs/codex-health.jsonl` |
| `bash scripts/doctor.sh` | Configuration, dependencies, Node.js, the Codex wrapper, the model backend, and the verification service. Review each `FAIL` or `warn` result |

In API mode, `doctor.sh`, `services.sh test`, and the recovery script can send billable requests. If the probe uses a different model from the research agents, check separately that the research model and reasoning effort are available.

In API mode, a successful probe means the current Responses request completed. Use `warn` entries in recent logs to investigate previous calls. In account mode, the probe checks the current sign-in status.

<a id="recover-or-update"></a>

## Recover or update a deployment

Before upgrading, record process status, back up projects and custom configuration, and stop the run at an appropriate round boundary. After a host restart, confirm that the old services and the main-agent, worker, and model subprocesses have exited. Then run the recovery command.

```bash
bash scripts/recover.sh
```

The recovery script reruns setup, clears stale service PIDs, and replays `$HYPROVER_RUNTIME/run/autostart`. Setup may rebuild the Python environment or regenerate the API provider configuration. If the verification service is missing from `autostart`, run `bash scripts/services.sh up verify` separately and check that it is ready.

Check the project's stop flags, deadline, and completion state before explicitly restarting unfinished research. For a background main agent, run `./bin/hyprover main start demo`. For manual coordination, run `./bin/hyprover start demo`. If only the main agent exited unexpectedly and the services are still running, you can restart that project after its remaining model subprocesses exit. The main agent picks up the saved research notebook and assignments. Workers can continue using their own local materials.

Processes pick up code and configuration changes when they restart. Keep the research data and control markers. Startup checks that the notebook format in `project.json` matches `KNOWLEDGE.md`, so restore both from the same backup.

<a id="troubleshooting"></a>

## Troubleshooting a deployment

| Symptom | What to check |
| --- | --- |
| A fact does not appear in the notebook | Inspect the verification service, logs, and receipt. Confirm that `accepted` and `published` are both `true` |
| A model call fails | Check the Responses base URL, model, effort, account permissions, usage limits, and `runtime/logs/codex-health.jsonl` |
| A worker keeps waiting | Check its current assignment and new shared inputs |
| A worker reports `input_error` | Check the input error in the status output and the corresponding notebook files and attachments |
| A project fails to start | Check the problem statement, notebook format, stop status, deadline, and any remaining model subprocesses |
| The dashboard is unreachable | Check the project name, service status, port, and SSH forwarding |

[Configuration reference](configuration.md)
