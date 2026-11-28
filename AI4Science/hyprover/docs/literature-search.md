# Literature search and source verification

HyProver ships with a public arXiv search adapter as an example. You can replace it with another public index or private database, or use Codex's built-in web search. Record search results as `lead` entries and check the original source and its assumptions before citing a result.

## Use the built-in arXiv adapter

When search is allowed, the main agent, workers, and verifiers can call `search_arxiv_theorems(query, num_results=10)`. For example, pass `query="ti:convex AND au:Gardner"` and `num_results=5`. The tool returns titles, authors, abstracts, and links from the public API. If the response contains `error`, handle that error. A successful query with `count=0` found no matches. [Parameters, caching, and rate limits](../hyprover/integrations/README.md)

## Switch to a public or private database

You can replace the default MCP data source in your deployment branch while keeping the tool name and role permissions.

1. Add an adapter under `hyprover/integrations/`, for example `bibliography.py`. Implement `search(query: str, num_results: int = 10) -> dict` to handle query translation, authentication, and timeouts. Limit `num_results` to the range 1–20. Return `query`, `count`, `results`, `source`, `"result_type": "paper_metadata"`, and `"is_verified": false`. For each result, include the title, authors, abstract if available, a stable identifier, and a source link. On failure, return `error`, `count=0`, and an empty `results` list.
2. In `hyprover/gateway/server.py`, replace the current import with `from hyprover.integrations.bibliography import search as _arxiv_search`. Keep the `search_arxiv_theorems` call and registration entry point. Update the tool description with the data source and query syntax, and retain role checks and the `HYPROVER_LITERATURE_SEARCH` check. [Gateway source](../hyprover/gateway/server.py)
3. For a private database, use read-only credentials and confirm that the authorization allows results to be sent to the model. Keep secrets in the local environment and list the variable names in `mcp_servers.hyprover.env_vars` in the Codex stdio configuration each session uses. Worker configuration is generated at startup. The background main agent and verifiers have their own startup configurations. [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
4. Use mock responses to check success, no matches, request failures, and behavior when retrieval is disallowed. After deployment, restart sessions that use the gateway. If you change the tool name, update the session instructions for each role.

You can also expose the data source as a separate MCP service and configure and check its permissions for each session. [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

## Use only Codex's built-in web search

If external search is allowed and your backend supports it, launch the interactive main-agent session from the repository root.

```bash
./bin/codex -c 'web_search="live"' \
  -c 'mcp_servers.hyprover.disabled_tools=["search_arxiv_theorems"]'
```

This command removes arXiv from the current session's tool list and keeps the notebook tools. To switch the background main agent, workers, and verifiers, set `web_search = "live"` in the effective configurations for those sessions. You can update the deployment's `$CODEX_HOME/config.toml` and check project overrides. To remove arXiv across the deployment, omit the registration of `search_arxiv_theorems` in the gateway's `build_app`. Update each role's search instructions, restart sessions, and check web search and the remaining tools. [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)

## Run without external search

Before starting, set `HYPROVER_LITERATURE_SEARCH=0` in `config/hyprover.env` and open new sessions to load the configuration. This disables the built-in arXiv tool. Codex sessions started by HyProver's Python launchers also disable built-in web search and receive instructions to use only approved materials. For an interactive session, pass `-c 'web_search="disabled"'` to `./bin/codex` as well. Disable any separately configured search tools. Model access continues to use the deployment's existing configuration.

## Find references for manuscript generation

Manuscript generation uses a separate source reader to check citations. Within its budget and permissions, the reader can query Crossref, arXiv, and approved hosts for original sources. It can also read original sources from the source manifest. `HYPROVER_LITERATURE_SEARCH` controls both network search and use of the source manifest. The reader follows the manuscript run's configuration when research sessions switch MCP sources or use built-in web search. [Manuscript source manifest](article.md#set-budgets-and-sources)
