# MCP Gateway for Research Tools

`hyprover.gateway` exposes the `hyprover-core` MCP service over standard input and output. It delegates notebook operations to `KnowledgeStore` and forwards proof submissions and fact reviews to the verification service. [HyProver architecture](../../docs/ARCHITECTURE.md)

<a id="startup-and-identity"></a>

## Startup and Session Configuration

`./bin/hyprover-mcp` loads the deployment environment and defaults to role `main` and author `main_agent`. Workers and verifiers use the roles bound by their own launch configuration. After loading the environment, you can also run `python -m hyprover.gateway` directly. This entry point defaults to `verifier` when the role is unspecified.

| Setting | Effect |
| --- | --- |
| `HYPROVER_ROLE` | Selects the tool set. Unknown values use the verifier's tool set. `all` is available for development tests |
| `HYPROVER_AUTHOR` | Binds the request author |
| `HYPROVER_PROJECT_DIR` | Binds the absolute project directory |
| `HYPROVER_AGENTS_ROOT` | Project root. The main agent can use `project` to select one of its direct subdirectories |
| `HYPROVER_PROBLEM_ID` | Optional problem identifier for proof submissions |
| `HYPROVER_VERIFY_URL`, `HYPROVER_VERIFY_TIMEOUT` | Verification service address and HTTP timeout |
| `HYPROVER_LITERATURE_SEARCH` | Setting this to `0` removes the arXiv search tool and rejects calls to it |

The gateway captures the session configuration at MCP tool registration and uses the same role and identity for each call. A main agent without a bound project passes the project name to notebook or fact review tools. Worker tools use the project bound at startup. [Configuration reference](../../docs/configuration.md)

<a id="common-entry-points"></a>

## Tools by Role

| Role | Research operations |
| --- | --- |
| Main agent | Browse the notebook, maintain the overview and adopted routes, and request fact reviews. Emergency revocations require operator authorization |
| Worker | Browse and record entries, submit candidate proofs, and request fact reviews |
| Verifier | Search the literature when the research settings permit it |

`fact_submit` calls `POST /verify/fact`. A publication receipt has `accepted=true`, `published=true`, and a valid `fact_id`. `fact_review` calls `POST /verify/review` and rereads the notebook to confirm the change was committed. The interface reference provides the parameters for browsing, editing, and searching. [Tool parameters and results](../core/INTERFACES.md) · [HTTP verification endpoints](../verify/README.md) · [Literature search guide](../../docs/literature-search.md)
