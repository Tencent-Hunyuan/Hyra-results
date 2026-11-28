# Read-Only Research Dashboard

The dashboard displays a project's shared research notebook, research entries, and proof dependency graph. Each request uses a consistent notebook snapshot. The dashboard loads proofs and longer notes when you open their fact or entry details.

## Startup

From the repository root, run `bash scripts/services.sh up dashboard demo`, where `demo` is the project name. After setting up the environment, you can also start the service in the foreground with `python -m hyprover.observability --project PROJECT_DIR`.

The service takes the first available project from `--project`, `HYPROVER_DASHBOARD_PROJECT`, or `HYPROVER_PROJECT_DIR`, in that order. It validates the directory and notebook format at startup and listens on `127.0.0.1:8099` by default. Use SSH port forwarding to access it remotely. [Start the services and dashboard](../../docs/operating-guide.md#services)

## HTTP Endpoints

| Request | Response |
| --- | --- |
| `GET /api/overview` | Project, fact, author, and entry counts. Verification feedback is grouped into `published` and `not_published` |
| `GET /api/factgraph` | Current valid fact nodes, proof dependencies, and graph depth |
| `GET /api/fact/{fact_id}` | Complete statement, content ID-checked proof, and direct predecessors |
| `GET /api/channels` | Counts by entry kind, including shared reports |
| `GET /api/channel/{kind}` | Paginated results for the selected kind |
| `GET /api/entry/{entry_id}` | Entry body and extended notes |
| `GET /`, `GET /static/*` | Dashboard page and static assets |

Channel queries support `query`, `subtype`, `author`, `linked_to`, `include_archived`, `sort_by`, `offset`, and `limit`. When paging, pass the first page's `snapshot_revision` as `expected_snapshot_revision`. If the notebook changes and the service returns HTTP 409, restart from the first page. Unknown or revoked facts and unknown entries or kinds return 404. Notebook validation and attachment read failures return 400. [Search and pagination](../core/INTERFACES.md#knowledge_search)

## Tests

`python -m pytest hyprover/observability/` verifies the HTTP reads. To test the dashboard scripts, install the optional dependencies from `tests/package.json` and run `npm test` in `tests/`. Pytest can run these tests as well. Set `HYPROVER_NODE` to the Node executable and `NODE_PATH` to the dependency installation directory.
