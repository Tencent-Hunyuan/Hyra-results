import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from hyprover.core import Fact, KnowledgeStore
from hyprover.gateway import server, tools_for


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    store = KnowledgeStore.initialize(root / "P", "P")
    for key, value in {
        "HYPROVER_AGENTS_ROOT": str(root), "HYPROVER_PROJECT_DIR": str(store.root),
        "HYPROVER_AUTHOR": "worker", "HYPROVER_ROLE": "worker",
        "HYPROVER_PROBLEM_ID": "P", "HYPROVER_VERIFY_URL": "http://verify/verify",
    }.items():
        monkeypatch.setenv(key, value)
    return store


def registered(monkeypatch, role):
    app = server.build_app(role)
    # These direct-call tests exercise role binding; transport tests below also
    # exercise the error boundary, SDK validation and JSON-RPC dispatch.
    return {tool.name: tool.fn.__wrapped__ for tool in app._tool_manager.list_tools()}


@pytest.mark.parametrize("role", ["worker", "main", "verifier", "unknown"])
def test_literature_disabled_hides_tool_and_blocks_direct_calls(monkeypatch, role):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    monkeypatch.setattr(server, "_arxiv_search", lambda *a, **k: pytest.fail("network invoked"))
    assert "search_arxiv_theorems" not in registered(monkeypatch, role)
    with pytest.raises(PermissionError, match="literature_search_disabled"):
        server.search_arxiv_theorems("forbidden query")


def test_literature_policy_is_bound_to_instance(monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    call = server._bind_role(server.search_arxiv_theorems, "verifier")
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    with pytest.raises(PermissionError, match="literature_search_disabled"):
        call("forbidden query")


def test_role_table_and_retired_tools():
    assert "fact_submit" in tools_for("worker")
    assert "fact_submit" not in tools_for("main")
    assert "fact_revoke" in tools_for("main")
    assert tools_for("verifier") == ["search_arxiv_theorems"]
    assert "search_arxiv_papers" not in server._TOOLS
    assert "paper metadata/abstracts, not theorem statements" in server.search_arxiv_theorems.__doc__
    assert tools_for("unknown") == tools_for("verifier")
    for role in ("main", "worker", "verifier", "all"):
        assert not {"gm_add", "gm_search", "fact_search"}.intersection(tools_for(role))
    assert all(not hasattr(server, name) for name in ("gm_add", "gm_search", "fact_search"))


@pytest.mark.parametrize("role", ["worker", "main", "verifier", "unknown"])
def test_registered_tools_match_role(monkeypatch, role):
    tools = registered(monkeypatch, role)
    assert set(tools) == set(tools_for(role))


def test_knowledge_record_search_are_nonfactual(project):
    result = server.knowledge_record(kind="conjecture", title="Integer conjecture", body="Integer arithmetic result")
    entry = project.snapshot().entries[result["entry_id"]]
    assert entry.author == "worker" and entry.status == "open"
    hits = server.knowledge_search("Integer")
    assert hits["results"][0]["is_fact"] is False
    assert not project.list()


@pytest.mark.parametrize("kind", ["fact", "feedback", "overview"])
def test_worker_cannot_record_privileged_kinds(project, kind):
    with pytest.raises((ValueError, PermissionError)):
        server.knowledge_record(kind=kind, title="Not allowed", body="Attempted privilege promotion")


def test_worker_cannot_promote_conjecture(project):
    with pytest.raises(ValueError):
        server.knowledge_record(kind="conjecture", title="Not proved", body="No independent verification", status="proved")


def test_tool_arguments_cannot_supply_identity_or_verdict():
    for fn in (server.knowledge_record, server.fact_submit):
        parameters = inspect.signature(fn).parameters
        assert not {"role", "author", "verdict", "verification", "context", "published"}.intersection(parameters)


def test_submit_forwards_candidate_without_local_publication(project, monkeypatch):
    captured = {}
    def verify(payload):
        captured.update(payload)
        return {"accepted": True, "published": True, "fact_id": "1234567890abcdef", "verdict": "correct"}
    monkeypatch.setattr(server, "_verify", verify)
    before = project.path.read_bytes()
    result = server.fact_submit(statement="Integer arithmetic", proof="Independent written argument",
                                glossary_introduces={"q": "an integer"}, title="Integer identity")
    assert result["accepted"] is True and result["published"] is True
    assert captured["project_dir"] == str(project.root) and captured["author"] == "worker"
    assert captured["glossary_introduces"] == {"q": "an integer"}
    assert captured["title"] == "Integer identity"
    assert "verdict" not in captured and "role" not in captured
    assert project.path.read_bytes() == before


@pytest.mark.parametrize("response", [
    {"verdict": "correct"},
    {"accepted": True, "published": False, "fact_id": "1234567890abcdef"},
    {"accepted": True, "fact_id": "1234567890abcdef"},
    {"accepted": True, "published": True, "fact_id": None},
    {"accepted": True, "published": True, "fact_id": "not-a-fact-id"},
    ["not", "an", "object"],
])
def test_unconfirmed_publication_fails_closed(project, monkeypatch, response):
    monkeypatch.setattr(server, "_verify", lambda payload: response)
    result = server.fact_submit(statement="Some statement", proof="Some proof")
    assert result["accepted"] is False and result["published"] is False
    assert result["fact_id"] is None and not project.list()


def test_transport_error_is_clean(project, monkeypatch):
    def verify(payload):
        raise RuntimeError("service unavailable")
    monkeypatch.setattr(server, "_verify", verify)
    result = server.fact_submit(statement="Some statement", proof="Some proof")
    assert not result["accepted"] and not result["published"]
    assert result["fact_id"] is None and "service unavailable" in result["error"]


def test_registered_worker_role_cannot_follow_mutated_environment(project, monkeypatch):
    tools = registered(monkeypatch, "worker")
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    with pytest.raises(ValueError):
        tools["knowledge_record"](kind="overview", title="No privilege escalation", body="Worker remains worker")
    with pytest.raises(PermissionError):
        tools["knowledge_search"]("anything", project="P")


def test_registered_tools_bind_author_project_and_root(project, monkeypatch, tmp_path):
    tools = registered(monkeypatch, "worker")
    other = KnowledgeStore.initialize(tmp_path / "other-projects" / "Q", "Q")
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(other.root))
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(other.root.parent))
    monkeypatch.setenv("HYPROVER_AUTHOR", "impostor")
    monkeypatch.setenv("HYPROVER_PROBLEM_ID", "Q")
    result = tools["knowledge_record"](kind="conjecture", title="Bound identity", body="Remains in original notebook")
    assert result["entry_id"] in project.snapshot().entries
    assert project.snapshot().entries[result["entry_id"]].author == "worker"
    assert not other.snapshot().entries
    captured = {}
    monkeypatch.setattr(server, "_verify", lambda payload: captured.update(payload) or {"accepted": False, "published": False})
    tools["fact_submit"](statement="Some statement", proof="Some proof")
    assert captured["project_dir"] == str(project.root)
    assert captured["author"] == "worker" and captured["problem_id"] == "P"


def test_role_context_is_reset_after_tool_failure(project, monkeypatch):
    tools = registered(monkeypatch, "worker")
    monkeypatch.setenv("HYPROVER_ROLE", "verifier")
    with pytest.raises(ValueError):
        tools["knowledge_record"](kind="overview", title="Not permitted", body="Worker cannot author overview")
    assert server._role() == "verifier"


@pytest.mark.parametrize("bad", ["../P", "a/b", "", "/absolute", ".", ".."])
def test_bad_project_names_rejected(project, monkeypatch, bad):
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    with pytest.raises((RuntimeError, ValueError, OSError)):
        server.knowledge_search("anything", project=bad)


@pytest.mark.parametrize("shape", ["same-root-link", "outside-link", "parent-link", "traversal", "nested"])
def test_project_path_aliases_and_escape_are_rejected(project, monkeypatch, tmp_path, shape):
    root = project.root.parent
    if shape == "same-root-link":
        supplied = root / "alias"
        supplied.symlink_to(project.root, target_is_directory=True)
    elif shape == "outside-link":
        outside = tmp_path / "outside"
        outside.mkdir()
        supplied = root / "alias"
        supplied.symlink_to(outside, target_is_directory=True)
    elif shape == "parent-link":
        alias = tmp_path / "alias-root"
        alias.symlink_to(root, target_is_directory=True)
        supplied = alias / "P"
    elif shape == "nested":
        supplied = project.root / "nested"
        supplied.mkdir()
    else:
        supplied = project.root / ".." / "P"
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(supplied))
    with pytest.raises((RuntimeError, ValueError, OSError)):
        server._project()


def test_main_can_select_valid_named_project(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    assert server._project("P") == project.root
    with pytest.raises((RuntimeError, ValueError, OSError)):
        server._project("missing")


@pytest.mark.parametrize("suffix", ["", "/verify", "/verify/fact", "/verify/fact/"])
def test_http_endpoint_and_complete_payload(project, monkeypatch, suffix):
    captured = {}
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"accepted":false,"published":false}'
    def open_request(request, timeout):
        captured.update(url=request.full_url, payload=json.loads(request.data), timeout=timeout)
        return Response()
    monkeypatch.setattr(server.urllib.request, "urlopen", open_request)
    monkeypatch.setenv("HYPROVER_VERIFY_URL", "http://verify" + suffix)
    monkeypatch.setenv("HYPROVER_VERIFY_TIMEOUT", "not-an-int")
    payload = {"statement": "A statement", "proof": "An argument", "project_dir": str(project.root)}
    result = server._verify(payload)
    assert not result["published"]
    assert captured == {"url": "http://verify/verify/fact", "payload": payload, "timeout": 3600}
    server._verify(payload, operation="review")
    assert captured == {"url": "http://verify/verify/review", "payload": payload, "timeout": 3600}
    monkeypatch.setenv("HYPROVER_VERIFY_TIMEOUT", "0")
    server._verify(payload)
    assert captured["timeout"] is None


@pytest.mark.parametrize("url", ["", "file:///tmp/local", "http://verify?a=b", "http://verify#fragment"])
def test_invalid_verify_endpoint_rejected(project, monkeypatch, url):
    monkeypatch.setenv("HYPROVER_VERIFY_URL", url)
    with pytest.raises(RuntimeError):
        server._verify({"statement": "s", "proof": "p"})


def test_main_revocation_cascades(project, monkeypatch):
    first = Fact("", "P", "worker", [], "First statement", "First complete proof")
    base = project.publish(first, {"verdict": "correct"})
    second = Fact("", "P", "worker", [base], "Second statement", "Second proof using first result")
    child = project.publish(second, {"verdict": "correct"})
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    assert set(server.fact_revoke(base, "Counterexample found")["revoked"]) == {base, child}
    assert not project.list()


def test_search_arxiv_delegates_offline(monkeypatch):
    monkeypatch.setattr(server, "_arxiv_search", lambda query, num_results: {"query": query, "count": num_results})
    assert server.search_arxiv_theorems("integer identity", 3) == {"query": "integer identity", "count": 3}


@pytest.mark.parametrize("response", [
    {"error": "http_error", "http_status": 429, "error_stage": "cooldown",
     "retry_after_seconds": 120, "retry_deferred": True, "count": 0, "results": []},
    {"error": "search_unavailable", "error_stage": "cache_lock", "error_type": "disk_full",
     "count": 0, "results": []},
    {"cached": False, "count": 1, "results": [{"title": "Synthetic metadata"}],
     "cache_warning": "cache_full_write_skipped"},
])
def test_search_arxiv_preserves_diagnostics(monkeypatch, response):
    monkeypatch.setattr(server, "_arxiv_search", lambda query, num_results: response)
    assert server.search_arxiv_theorems("integer identity", 1) == response


def test_report_retry_and_filtered_search_use_real_store(project):
    report = server.knowledge_record("report", "Synthetic check", "No mathematical certification.",
                                     details="Long synthetic detail.", request_id="gateway-report-1")
    revision = project.snapshot().revision
    assert server.knowledge_record("report", "Synthetic check", "No mathematical certification.",
                                   details="Long synthetic detail.", request_id="gateway-report-1") == report
    assert project.snapshot().revision == revision
    assert server.knowledge_search("Synthetic check")["results"] == []
    page = server.knowledge_search("Synthetic check", kinds=["report"], author="worker",
                                   limit=1, offset=0, sort_by="updated")
    assert page["total"] == 1 and page["has_more"] is False
    assert page["snapshot_revision"] == revision
    assert page["results"][0]["id"] == report["entry_id"]
    assert page["results"][0]["is_fact"] is False
    with pytest.raises(ValueError):
        server.knowledge_record("report", "Synthetic check", "Different payload.",
                                details="Long synthetic detail.", request_id="gateway-report-1")
    assert project.snapshot().revision == revision


def test_experience_subtype_is_explicit_and_search_implies_kind(project):
    before = project.path.read_bytes()
    with pytest.raises(ValueError):
        server.knowledge_record("experience", "Missing type", "Not automatically classified.")
    assert project.path.read_bytes() == before
    method = server.knowledge_record("experience", "Method", "Technique with prerequisites.",
                                     attributes={"subtype": "method"})
    server.knowledge_record("experience", "Boundary", "A scoped limitation.",
                            attributes={"subtype": "boundary"})
    result = server.knowledge_search("", subtype="method", limit=1)
    assert result["total"] == 1 and result["results"][0]["id"] == method["entry_id"]
    with pytest.raises(ValueError):
        server.knowledge_search("", kinds=["report"], subtype="method")


def test_registered_report_retry_binds_identity_and_project(project, monkeypatch, tmp_path):
    tools = registered(monkeypatch, "worker")
    other = KnowledgeStore.initialize(tmp_path / "other" / "Q", "Q")
    first = tools["knowledge_record"]("report", "Bound report", "Synthetic report.", request_id="bound-report")
    monkeypatch.setenv("HYPROVER_AUTHOR", "impostor")
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(other.root))
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(other.root.parent))
    retry = tools["knowledge_record"]("report", "Bound report", "Synthetic report.", request_id="bound-report")
    assert retry["entry_id"] == first["entry_id"]
    assert project.snapshot().entries[first["entry_id"]].author == "worker"
    assert not other.snapshot().entries
    with pytest.raises(TypeError):
        tools["knowledge_record"]("report", "Impersonation", "Rejected.", author="impostor")


def test_gateway_browses_without_query_or_relevance_scores(project):
    first = server.knowledge_record("conjecture", "Positive curvature", "A geometric condition.")
    second = server.knowledge_record("conjecture", "Shape recovery", "A different formulation.")
    page = server.knowledge_search(kinds=["conjecture"], limit=1)
    assert page["sort_by"] == "updated" and page["total"] == 2 and page["has_more"]
    more = server.knowledge_search(kinds=["conjecture"], limit=1, offset=1,
                                   expected_snapshot_revision=page["snapshot_revision"])
    rows = page["results"] + more["results"]
    assert {row["id"] for row in rows} == {first["id"], second["id"]}
    assert all("score" not in row for row in rows)
    assert server.knowledge_search(first["display_id"].lower())["results"][0]["id"] == first["id"]


def test_gateway_search_rejects_stale_snapshot(project):
    server.knowledge_record("report", "First", "Synthetic report.")
    page = server.knowledge_search("", kinds=["report"], limit=1)
    server.knowledge_record("report", "Second", "Another synthetic report.")
    with pytest.raises(ValueError, match="search_snapshot_conflict"):
        server.knowledge_search("", kinds=["report"], offset=1,
                                expected_snapshot_revision=page["snapshot_revision"])


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def mcp_session(app):
    lowlevel = getattr(app, "_lowlevel_server", None) or getattr(app, "_mcp_server")
    with anyio.fail_after(30):
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as group:
                group.start_soon(lowlevel.run, *server_streams, lowlevel.create_initialization_options())
                async with ClientSession(*client_streams) as client:
                    await client.initialize()
                    yield client
                group.cancel_scope.cancel()


def wire_result(result):
    return result.model_dump(mode="json", by_alias=True)


def wire_data(result):
    wire = wire_result(result)
    assert not wire.get("isError"), wire
    return json.loads(wire["content"][0]["text"])


def wire_error(result, code=None):
    wire = wire_result(result)
    assert wire["isError"], wire
    text = "\n".join(block.get("text", "") for block in wire["content"])
    if code is not None:
        assert text.endswith(": " + code) or text == code, text
    return text


@pytest.mark.anyio
async def test_mcp_query_is_optional_for_model_led_browsing(project):
    first = project.record("conjecture", "First formulation", "A synthetic candidate.", "worker", "worker")
    second = project.record("conjecture", "Another formulation", "A related idea without shared terms.", "worker", "worker")
    before = project.path.read_bytes()
    async with mcp_session(server.build_app("worker")) as client:
        tools = (await client.list_tools()).tools
        schema = next(wire_result(tool)["inputSchema"] for tool in tools if tool.name == "knowledge_search")
        assert "query" not in schema.get("required", [])
        page = wire_data(await client.call_tool("knowledge_search", {"kinds": ["conjecture"], "limit": 1}))
        assert page["query"] == "" and page["sort_by"] == "updated" and page["has_more"]
        more = wire_data(await client.call_tool("knowledge_search", {
            "kinds": ["conjecture"], "limit": 1, "offset": 1,
            "expected_snapshot_revision": page["snapshot_revision"],
        }))
        rows = page["results"] + more["results"]
        assert {row["id"] for row in rows} == {first.entry_id, second.entry_id}
        assert all("score" not in row for row in rows) and not more["has_more"]
    assert project.path.read_bytes() == before


@pytest.mark.anyio
@pytest.mark.parametrize("request_id", ["null", "true", "false", "123", "normal-key"])
async def test_mcp_creation_retry_preserves_string_keys(project, request_id):
    payload = {"kind": "report", "title": "Synthetic", "body": "No research performed.",
               "request_id": request_id}
    async with mcp_session(server.build_app("worker")) as client:
        first = wire_data(await client.call_tool("knowledge_record", payload))
        revision = project.snapshot().revision
        second = wire_data(await client.call_tool("knowledge_record", payload))
        assert second == first
        assert project.snapshot().revision == revision
        assert len(project.snapshot().entries) == 1
        page = wire_data(await client.call_tool("knowledge_search", {"query": "", "kinds": ["report"]}))
        assert isinstance(page["results"], list) and page["total"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("author", ["null", "[]", "{}", '["worker"]', '"worker"', "true", "123"])
async def test_mcp_author_filter_preserves_every_string(project, author):
    target = project.record(kind="report", title="Target", body="Synthetic target.", author=author, role="worker")
    project.record(kind="report", title="Other", body="Synthetic decoy.", author="other", role="worker")
    async with mcp_session(server.build_app("worker")) as client:
        page = wire_data(await client.call_tool("knowledge_search", {
            "query": "", "kinds": ["report"], "author": author,
        }))
        assert page["total"] == 1 and page["results"][0]["id"] == target.entry_id
        unfiltered = wire_data(await client.call_tool("knowledge_search", {
            "query": "", "kinds": ["report"], "author": None,
        }))
        assert unfiltered["total"] == 2


@pytest.mark.anyio
@pytest.mark.parametrize("name,field,arguments", [
    ("knowledge_search", "limit", {"query": ""}),
    ("knowledge_search", "offset", {"query": ""}),
    ("knowledge_search", "expected_snapshot_revision", {"query": ""}),
    ("knowledge_record", "expected_revision", {"kind": "report", "title": "Synthetic", "body": "Synthetic."}),
    ("fact_submit", "source_revision", {"statement": "Synthetic", "proof": "Not a proof."}),
    ("fact_review", "expected_revision", {"fact_id": "a" * 16, "action": "revoke", "reason": "Synthetic."}),
    ("search_arxiv_theorems", "num_results", {"query": "Synthetic"}),
])
@pytest.mark.parametrize("value", [True, False, "1", 1.5])
async def test_mcp_integer_arguments_reject_coercion(project, monkeypatch, name, field, arguments, value):
    monkeypatch.setattr(server, "_verify", lambda *a, **k: pytest.fail("verification was invoked"))
    monkeypatch.setattr(server, "_arxiv_search", lambda *a, **k: pytest.fail("network was invoked"))
    before = project.path.read_bytes()
    async with mcp_session(server.build_app("worker")) as client:
        wire_error(await client.call_tool(name, dict(arguments, **{field: value})))
    assert project.path.read_bytes() == before


@pytest.mark.anyio
async def test_mcp_conflicts_are_actionable_and_atomic(project):
    payload = {"kind": "report", "title": "Synthetic", "body": "Synthetic report.", "request_id": "null"}
    async with mcp_session(server.build_app("worker")) as client:
        first = wire_data(await client.call_tool("knowledge_record", payload))
        page = wire_data(await client.call_tool("knowledge_search", {"query": "", "kinds": ["report"]}))
        before = project.path.read_bytes()
        wire_error(await client.call_tool("knowledge_record", dict(payload, body="Changed.")), "record_request_conflict")
        wire_error(await client.call_tool("knowledge_record", {
            "kind": "report", "title": "Synthetic", "body": "Changed.",
            "entry_id": first["entry_id"], "expected_revision": first["revision"] + 1,
        }), "entry_revision_conflict")
        assert project.path.read_bytes() == before
        wire_data(await client.call_tool("knowledge_record", dict(payload, request_id="another")))
        wire_error(await client.call_tool("knowledge_search", {
            "query": "", "kinds": ["report"], "expected_snapshot_revision": page["snapshot_revision"],
        }), "search_snapshot_conflict")


@pytest.mark.anyio
@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError])
async def test_mcp_unknown_exception_never_discloses_internal_text(project, monkeypatch, error):
    secret = "/private/project/config.json contains confidential report text"
    def broken(project=None):
        raise error(secret)
    monkeypatch.setattr(server, "_knowledge", broken)
    async with mcp_session(server.build_app("worker")) as client:
        result = await client.call_tool("knowledge_search", {"query": ""})
        wire_error(result, "internal_tool_error")
        assert secret not in json.dumps(wire_result(result))
        assert "/private" not in json.dumps(wire_result(result))


@pytest.mark.anyio
@pytest.mark.parametrize("code", [
    "search_snapshot_conflict", "metadata_must_be_finite_json", "json_metadata_keys_must_be_strings",
    "invalid_json_unicode", "external_refs_must_be_list",
])
async def test_mcp_public_error_strips_private_suffix(project, monkeypatch, code):
    def broken(project=None):
        raise ValueError(f"{code}: /private/config and secret body")
    monkeypatch.setattr(server, "_knowledge", broken)
    async with mcp_session(server.build_app("worker")) as client:
        result = await client.call_tool("knowledge_search", {"query": ""})
        wire_error(result, code)
        assert "/private" not in json.dumps(wire_result(result))
        assert "secret body" not in json.dumps(wire_result(result))


@pytest.mark.anyio
@pytest.mark.parametrize("role", ["worker", "main", "verifier", "unknown", "all"])
async def test_mcp_discovery_role_and_schema_unchanged(project, role):
    app = server.build_app(role)
    baseline = server.FastMCP("schema-baseline")
    for name in tools_for(role):
        baseline.tool(name=name)(server._bind_role(server._TOOLS[name], role))
    async with mcp_session(app) as client:
        tools = wire_result(await client.list_tools())["tools"]
        assert {tool["name"] for tool in tools} == set(tools_for(role))
        for tool in tools:
            original = baseline._tool_manager.get_tool(tool["name"])
            assert tool["inputSchema"] == original.parameters
            assert tool.get("outputSchema") == original.output_schema


@pytest.mark.anyio
async def test_mcp_bound_role_identity_permissions_and_context(project, monkeypatch, tmp_path):
    app = server.build_app("worker")
    other = KnowledgeStore.initialize(tmp_path / "other" / "Q", "Q")
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    monkeypatch.setenv("HYPROVER_AUTHOR", "impostor")
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(other.root))
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(other.root.parent))
    payload = {"kind": "report", "title": "Bound", "body": "Synthetic report."}
    async with mcp_session(app) as client:
        record = wire_data(await client.call_tool("knowledge_record", payload))
        assert project.snapshot().entries[record["entry_id"]].author == "worker"
        assert not other.snapshot().entries
        wire_error(await client.call_tool("knowledge_record", dict(payload, kind="overview")), "role_cannot_record_kind")
        wire_error(await client.call_tool("knowledge_search", {"query": "", "project": "P"}), "permission_denied")
        for key in ("role", "author", "context", "published", "verdict"):
            wire_error(await client.call_tool("knowledge_record", dict(payload, **{key: "impostor"})))
        wire_error(await client.call_tool("fact_revoke", {"fact_id": "a" * 16, "reason": "Synthetic"}))
    assert server._CALL_ROLE.get() is None and server._CALL_CONFIG.get() is None
    assert len(project.snapshot().entries) == 1


@pytest.mark.anyio
async def test_mcp_concurrent_identical_and_conflicting_creation_keys(project):
    payload = {"kind": "report", "title": "Concurrent", "body": "Synthetic.", "request_id": "null"}
    async with mcp_session(server.build_app("worker")) as client:
        results = await asyncio.gather(*(client.call_tool("knowledge_record", payload) for _ in range(20)))
        records = [wire_data(result) for result in results]
        assert all(record == records[0] for record in records)
        assert len(project.snapshot().entries) == 1
        competing = await asyncio.gather(*(
            client.call_tool("knowledge_record", dict(payload, request_id="competing", body=f"Synthetic {i}."))
            for i in range(8)
        ))
        assert sum(not wire_result(result)["isError"] for result in competing) == 1
        for result in competing:
            if wire_result(result)["isError"]:
                wire_error(result, "record_request_conflict")
        assert len(project.snapshot().entries) == 2


@pytest.mark.anyio
async def test_mcp_sdk_direct_call_is_strict_without_wire_validation(project):
    from hyprover._mcp import ToolError

    app = server.build_app("worker")
    for field in ("limit", "offset", "expected_snapshot_revision"):
        for value in (True, False, "1", 1.5):
            with pytest.raises(ToolError):
                await app.call_tool("knowledge_search", {"query": "", field: value})
    record = app._tool_manager.get_tool("knowledge_record")
    assert record.fn_metadata.pre_parse_json({"request_id": "null"}) == {"request_id": "null"}
    assert not project.snapshot().entries


@pytest.mark.anyio
async def test_mcp_source_schema_and_search_reference_forwarding(project, monkeypatch):
    captured = []
    monkeypatch.setattr(server, "_verify", lambda payload: captured.append(payload) or {
        "accepted": False, "published": False, "fact_id": None,
    })
    guess = project.record("conjecture", "Source", "Synthetic statement", "worker", "worker")
    project.record("report", "Decoy", "Not a conjecture", "worker", "worker")
    before = project.path.read_bytes()
    async with mcp_session(server.build_app("worker")) as client:
        tools = wire_result(await client.list_tools())["tools"]
        schema = next(tool["inputSchema"] for tool in tools if tool["name"] == "fact_submit")
        assert "source" not in schema.get("required", [])
        assert {"source_id", "source_revision"} <= schema["properties"].keys()
        source_schema = next(option for option in schema["properties"]["source"]["anyOf"]
                             if option.get("type") != "null")
        if "$ref" in source_schema:
            source_schema = schema["$defs"][source_schema["$ref"].rsplit("/", 1)[1]]
        assert set(source_schema["required"]) == {"id", "revision"}
        assert source_schema["additionalProperties"] is False
        assert source_schema["properties"]["id"]["type"] == "string"
        assert source_schema["properties"]["revision"]["type"] == "integer"
        assert source_schema["properties"]["revision"]["minimum"] == 1
        page = wire_data(await client.call_tool("knowledge_search", {"query": guess.display_id}))
        row = page["results"][0]
        reference = {"id": guess.entry_id, "revision": guess.revision}
        assert row["source_ref"] == reference
        assert reference["id"] != row["display_id"]
        assert reference["revision"] != page["snapshot_revision"]
        for fields in ({"source": row["source_ref"]},
                       {"source_id": reference["id"], "source_revision": reference["revision"]},
                       {}, {"source": None},
                       {"source": reference, "source_id": None, "source_revision": None},
                       {"source": None, "source_id": reference["id"], "source_revision": reference["revision"]},
                       {"source": None, "source_id": None, "source_revision": None}):
            result = wire_data(await client.call_tool("fact_submit", {
                "statement": row["body"], "proof": "Synthetic proof", **fields,
            }))
            assert not result["accepted"]
            assert captured[-1]["project_dir"] == str(project.root)
            assert "source" not in captured[-1] and "source_ref" not in captured[-1]
        assert captured[0] == captured[1] == captured[4] == captured[5]
        assert captured[0]["source_id"] == guess.entry_id
        assert captured[0]["source_revision"] == guess.revision
        assert captured[2] == captured[3] == captured[6]
        assert captured[2]["source_id"] is None and captured[2]["source_revision"] is None
    assert project.path.read_bytes() == before


INVALID_SOURCES = [
    {}, {"id": "entry-example"}, {"revision": 1},
    {"id": "", "revision": 1}, {"id": " \n", "revision": 1},
    {"id": 17, "revision": 1}, {"id": "entry-example", "revision": None},
    {"id": "entry-example", "revision": 0}, {"id": "entry-example", "revision": -1},
    {"id": "entry-example", "revision": True}, {"id": "entry-example", "revision": "1"},
    {"id": "entry-example", "revision": 1.0},
    {"id": "entry-example", "revision": 1, "project": "other"},
    {"id": "entry-example", "revision": 1, "author": "other"},
    {"id": "entry-example", "revision": 1, "extra": "ignored?"},
    '{"id":"entry-example","revision":1}', [], False,
]


@pytest.mark.anyio
@pytest.mark.parametrize("source", INVALID_SOURCES)
async def test_source_is_strict_on_python_sdk_and_mcp_entries(project, monkeypatch, source):
    from hyprover._mcp import ToolError

    monkeypatch.setattr(server, "_verify", lambda *a, **k: pytest.fail("verification was invoked"))
    before = project.path.read_bytes()
    arguments = {"statement": "Synthetic", "proof": "Not a proof", "source": source}
    result = server.fact_submit(**arguments)
    assert result["error"] == "invalid_source_reference"
    assert result["accepted"] is False and result["published"] is False and result["fact_id"] is None
    app = server.build_app("worker")
    with pytest.raises(ToolError):
        await app.call_tool("fact_submit", arguments)
    async with mcp_session(app) as client:
        wire_error(await client.call_tool("fact_submit", arguments))
    assert project.path.read_bytes() == before


@pytest.mark.anyio
@pytest.mark.parametrize("fields,code", [
    ({"source_id": "entry-example"}, "source_requires_positive_integer_revision"),
    ({"source_revision": 1}, "source_revision_without_source"),
    ({"source_id": "", "source_revision": 1}, "invalid_source_reference"),
    ({"source_id": "entry-example", "source_revision": 0}, "source_requires_positive_integer_revision"),
    ({"source": {"id": "entry-example", "revision": 1}, "source_id": "entry-example"}, "source_arguments_conflict"),
    ({"source": {"id": "entry-example", "revision": 1}, "source_revision": 1}, "source_arguments_conflict"),
    ({"source": {"id": "entry-example", "revision": 1}, "source_id": "entry-example", "source_revision": 1}, "source_arguments_conflict"),
    ({"source": {"id": "entry-example", "revision": 1}, "source_id": "entry-other", "source_revision": 2}, "source_arguments_conflict"),
])
async def test_source_forms_never_mix_or_fill_missing_values(project, monkeypatch, fields, code):
    monkeypatch.setattr(server, "_verify", lambda *a, **k: pytest.fail("verification was invoked"))
    before = project.path.read_bytes()
    arguments = {"statement": "Synthetic", "proof": "Not a proof", **fields}
    assert server.fact_submit(**arguments)["error"] == code
    async with mcp_session(server.build_app("worker")) as client:
        result = wire_data(await client.call_tool("fact_submit", arguments))
        assert result["error"] == code
        assert not result["accepted"] and not result["published"] and result["fact_id"] is None
    assert project.path.read_bytes() == before


def test_source_model_and_legacy_direct_types(project, monkeypatch):
    captured = []
    monkeypatch.setattr(server, "_verify", lambda payload: captured.append(payload) or {"accepted": False})
    server.fact_submit("Statement", "Proof", source=server.SourceRef(id="entry-example", revision=3))
    assert captured[0]["source_id"] == "entry-example" and captured[0]["source_revision"] == 3
    for revision in (True, False, "1", 1.0):
        result = server.fact_submit("Statement", "Proof", source_id="entry-example", source_revision=revision)
        assert result["error"] == "source_requires_positive_integer_revision"
    constructed = server.SourceRef.model_construct(id="entry-example", revision=True)
    assert server.fact_submit("Statement", "Proof", source=constructed)["error"] == "invalid_source_reference"
    assert len(captured) == 1


def test_search_source_reference_only_on_conjectures_and_uses_read_revision(project, monkeypatch):
    guess = project.record("conjecture", "Source", "Synthetic", "worker", "worker")
    project.record("report", "Progress", "Synthetic", "worker", "worker")
    project.record("lead", "Literature", "Synthetic", "worker", "worker")
    project.publish(Fact("", "P", "worker", [], "Other statement", "Synthetic proof"), {"verdict": "correct"})
    original = KnowledgeStore.search_page

    def read_then_update(store, *args, **kwargs):
        page = original(store, *args, **kwargs)
        store.record("conjecture", "Revised", "Revised statement", "worker", "worker",
                     entry_id=guess.entry_id, expected_revision=guess.revision)
        return page

    monkeypatch.setattr(KnowledgeStore, "search_page", read_then_update)
    page = server.knowledge_search(kinds=["conjecture", "report", "lead", "fact"])
    assert len(page["results"]) == 4
    for row in page["results"]:
        if row["kind"] == "conjecture":
            assert row["source_ref"] == {"id": guess.entry_id, "revision": guess.revision}
        else:
            assert "source_ref" not in row
    assert project.snapshot().entries[guess.entry_id].revision == guess.revision + 1


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["fact_submit", "fact_review"])
async def test_mcp_failure_envelope_does_not_leak_caught_exception(project, monkeypatch, name):
    private = "/private/verify/config.json: unpublished report body"
    def broken(*args, **kwargs):
        raise RuntimeError(private)
    monkeypatch.setattr(server, "_verify", broken)
    fact_id = project.publish(Fact("", "P", "worker", [], "Synthetic statement", "Synthetic proof"),
                              {"verdict": "correct"})
    arguments = ({"statement": "Synthetic", "proof": "Synthetic"} if name == "fact_submit" else
                 {"fact_id": fact_id, "action": "revoke", "reason": "Synthetic", "expected_revision": 1})
    async with mcp_session(server.build_app("worker")) as client:
        result = await client.call_tool(name, arguments)
        data = wire_data(result)
        assert data["accepted"] is False
        assert data["error"] == "internal_tool_error"
        assert private not in json.dumps(wire_result(result))
    assert private in getattr(server, name)(**arguments)["error"]


@pytest.mark.anyio
async def test_mcp_json_null_is_distinct_from_literal_null_request_id(project):
    payload = {"kind": "report", "title": "Synthetic", "body": "Synthetic.", "request_id": None}
    async with mcp_session(server.build_app("worker")) as client:
        records = [wire_data(await client.call_tool("knowledge_record", payload)) for _ in range(2)]
        assert records[0]["entry_id"] != records[1]["entry_id"]
        assert len(project.snapshot().entries) == 2


@pytest.mark.anyio
async def test_mcp_record_ownership_and_archive_permissions(project, monkeypatch):
    record = project.record(kind="report", title="Owned", body="Synthetic report.", author="owner", role="worker")
    monkeypatch.setenv("HYPROVER_AUTHOR", "another")
    payload = {"kind": "report", "title": "Owned", "body": "Changed report.",
               "entry_id": record.entry_id, "expected_revision": record.revision}
    for role, code in (("worker", "entry_ownership_conflict"), ("main", "report_author_required")):
        async with mcp_session(server.build_app(role)) as client:
            wire_error(await client.call_tool("knowledge_record", payload), code)
    async with mcp_session(server.build_app("main")) as client:
        archived = wire_data(await client.call_tool("knowledge_record", dict(payload, body=record.body, status="archived")))
        assert archived["status"] == "archived"
    assert project.snapshot().entries[record.entry_id].body == record.body


@pytest.mark.anyio
@pytest.mark.parametrize("directory", ["notes", "project"])
async def test_mcp_retry_does_not_claim_success_before_directory_sync(project, monkeypatch, directory):
    from hyprover.core import _util

    target = project.root / "knowledge/notes" if directory == "notes" else project.root
    original_sync = _util.fsync_directory
    attempts = []
    def fail_directory(path):
        if Path(path) == target:
            attempts.append(path)
            raise OSError("/private/project: injected directory sync failure")
        return original_sync(path)
    payload = {"kind": "report", "title": "Durable retry", "body": "Synthetic.",
               "details": "Synthetic note.", "request_id": "null"}
    async with mcp_session(server.build_app("worker")) as client:
        with monkeypatch.context() as patch:
            patch.setattr(_util, "fsync_directory", fail_directory)
            for _ in range(2):
                result = await client.call_tool("knowledge_record", payload)
                wire_error(result, "internal_tool_error")
                assert "/private" not in json.dumps(wire_result(result))
            assert len(attempts) == 2
        recovered = wire_data(await client.call_tool("knowledge_record", payload))
        retry = wire_data(await client.call_tool("knowledge_record", payload))
        assert recovered == retry
        snapshot = project.snapshot()
        assert len(snapshot.entries) == 1 and snapshot.revision == 1
        assert project.entry_detail(recovered["entry_id"])["details"] == payload["details"]
        assert len(list((project.root / "knowledge/notes").glob("*.md"))) == 1
