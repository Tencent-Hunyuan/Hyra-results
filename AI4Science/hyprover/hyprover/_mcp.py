"""The ``FastMCP`` server class, resolved across mcp versions.

mcp 2.0 removed ``mcp.server.fastmcp`` and renamed the class to ``MCPServer``
(``mcp.server.mcpserver``). The surface hyprover uses — ``FastMCP(name)``,
``app.tool(name=...)(fn)``, ``app.run()`` — is the same on both, so the research
gateway imports the class from here to support either version.

Resolved by capability (try the import) rather than by version number: mcp exposes
no ``__version__``, and the 2.0 pre-releases already dropped the old module while
still comparing as ``< 2.0.0``, so a version test picks the wrong branch.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from pydantic import ConfigDict

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
except ImportError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.utilities.func_metadata import FuncMetadata


class _WireMetadata(FuncMetadata):
    """MCP arguments are already JSON values; never decode their strings again."""

    def pre_parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


def public_error_code(exc, public_errors=frozenset()):
    """Expose only allowlisted business codes, never arbitrary exception text."""
    code = str(exc).partition(":")[0] if isinstance(exc, ValueError) else ""
    if code in public_errors:
        return code
    return "permission_denied" if isinstance(exc, PermissionError) else "internal_tool_error"


def register_tool(app, name, fn, *, public_errors=frozenset()):
    """Register a synchronous tool with lossless arguments and safe failures.

    Only this registration is adapted, not the SDK or other services. Both SDK
    generations validate via FuncMetadata.pre_parse_json and arg_model. Retain
    their generated schemas and output conversion; fail at startup if those
    capabilities change rather than silently running with lossy coercion.
    """
    @wraps(fn)
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise ToolError(public_error_code(exc, public_errors)) from exc

    app.tool(name=name)(guarded)
    manager = getattr(app, "_tool_manager", None)
    tool = manager.get_tool(name) if manager is not None else None
    metadata = getattr(tool, "fn_metadata", None)
    if not isinstance(metadata, FuncMetadata) or not callable(getattr(metadata, "pre_parse_json", None)):
        raise RuntimeError("unsupported_mcp_tool_metadata")
    model = metadata.arg_model
    strict_model = type(model.__name__, (model,), {
        "model_config": ConfigDict(**(dict(model.model_config) | {"strict": True, "extra": "forbid"})),
    })
    fields = {key: getattr(metadata, key) for key in type(metadata).model_fields}
    tool.fn_metadata = _WireMetadata(**(fields | {"arg_model": strict_model}))
    return guarded


__all__ = ["FastMCP", "register_tool"]
