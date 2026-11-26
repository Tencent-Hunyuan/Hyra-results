"""Run the hyprover gateway as a stdio MCP server: ``python -m hyprover.gateway``.

Role is taken from ``HYPROVER_ROLE`` (env). Launched by ``bin/hyprover-mcp`` (main) and
by each worker's ``.codex/config.toml`` (worker) / the verifier's ``-c`` override.
"""

from .server import build_app

if __name__ == "__main__":
    build_app().run()
