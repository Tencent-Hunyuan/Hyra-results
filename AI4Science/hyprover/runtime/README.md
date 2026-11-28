# Local runtime directory

By default, HyProver stores research projects, run records, and deployment files in the repository's `runtime/` directory. Git ignores project data and files generated for a particular machine.

## Research data and service records

| Location | Contents |
| --- | --- |
| `projects/<project>/PROBLEM.md` | Problem statement, definitions, and targets supplied by the operator |
| `projects/<project>/KNOWLEDGE.md`, `projects/<project>/knowledge/` | Shared research notebook and the proof and note attachments it links to |
| `projects/<project>/workers/` | Each worker's assignment, status, and working materials |
| `projects/<project>/main/` | Background main-agent status and completion receipts |
| `projects/<project>/articles/` | Article-generation runs, manuscripts, and review records |
| `logs/` | Service logs and model connection checks in `codex-health.jsonl` |
| `run/` | Service PIDs and the `autostart` record |
| `verify-runs/` | Verification reports and evidence for completion reviews |

Set `HYPROVER_RUNTIME` to change the directory services use at runtime. Use `HYPROVER_AGENTS_ROOT` and `VERIFIER_RESULTS_DIR` to set the project root and the verification results directory individually. Use the configured locations when backing up data. [Paths and services](../docs/configuration.md#paths-and-services)

## Files created during installation

The installation script sets up Node.js, the Python virtual environment, and the Codex CLI in the repository's `runtime/` directory. It writes `runtime.env` for the command wrappers. Local Codex settings and sign-in state live in `CODEX_HOME`, which defaults to `codex-home/` under the deployment's runtime directory. [Installation guide](../docs/getting-started.md)

After a restart, bring the verification service back first. Check project status before resuming the main agent or workers. [Recovering or updating a deployment](../docs/operating-guide.md#recover-or-update)
