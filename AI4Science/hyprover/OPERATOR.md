# Operator preferences

Read this file alongside `AGENTS.md` when starting a session. Keep long-term preferences here and record project progress in the project notebook. Store credentials in local `config/*.env` files.

## Preferences

| Setting | Value |
| --- | --- |
| Preferred name | To be filled in |
| Language | Use English for replies, code, role instructions, problem statements, and worker assignments by default |
| Time zone | To be filled in for scheduling runs |
| Notifications | Specify which events to report and where |
| Budget and run limits | Agree on these before each run |
| Worker configuration | Choose a model, reasoning effort, and number of workers that fit the backend quota. Set `HYPROVER_WORKER_MODEL` to override the model |
| Stopping | Prefer `hyprover stop <project>` so workers can finish their current rounds |

## Projects

Add links to projects here as you create them. Each project's problem statement, results, and proof dependencies live under `runtime/projects/<project>/`.

Record verified facts, conditional results, conjectures, and leads in their corresponding notebook sections. Keep the original goal and stopping conditions in view. When a route stalls, record the evidence and adjust the plan. Get the operator's authorization before publishing results or sending reports to others.
