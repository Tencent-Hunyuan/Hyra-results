"""Offline checks for the read-only unified notebook dashboard."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hyprover.core import Fact, KnowledgeStore
from hyprover.observability import app as obs_app
from hyprover.observability.app import build_channel, build_channels, build_factgraph, build_overview


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    store = KnowledgeStore.initialize(root, "P")
    a = store.publish(Fact("", "P", "worker_high", [], "Base lemma.", "Proof of base."), {"verdict": "correct"})
    b = store.publish(Fact("", "P", "worker_high", [a], "Middle result.", "Proof of middle."), {"verdict": "correct"})
    c = store.publish(Fact("", "P", "worker_xhigh", [b], "Top result.", "Proof of top.", intuition="the deep one"), {"verdict": "correct"})
    d = store.publish(Fact("", "P", "worker_high", [], "Standalone result.", "Standalone proof."), {"verdict": "correct"})
    store.record("route", "First route", "Investigate the next bridge.", "main", "main", status="planned")
    entry = store.record("route", "Second route", "An alternative mechanism.", "main", "main", details="Long route explanation.")
    store.record("feedback", "Accepted", "Synthetic accepted proof.", "verifier", "verifier", attributes={"published": True})
    store.record("feedback", "Rejected", "A failed attempt, not a refutation.", "verifier", "verifier", attributes={"published": False})
    monkeypatch.setenv("HYPROVER_DASHBOARD_PROJECT", str(root))
    return root, store, [a, b, c, d], entry


def test_overview_counts(project):
    root, _, _, _ = project
    overview = build_overview(root)
    assert overview["facts"] == 4
    assert overview["facts_with_predecessors"] == 2
    assert overview["facts_by_author"]["worker_high"] == 3
    assert overview["channel_counts"]["route"] == 2
    assert overview["verdicts"] == {"published": 1, "not_published": 1}


def test_graph_preserves_dependencies_without_proof_bodies(project):
    root, _, ids, _ = project
    graph = build_factgraph(root)
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert set(nodes) == set(ids)
    assert [nodes[fid]["depth"] for fid in ids] == [0, 1, 2, 0]
    assert graph["max_depth"] == 2
    assert graph["edges"] == [{"source": ids[0], "target": ids[1]}, {"source": ids[1], "target": ids[2]}]
    assert all("proof" not in node and "intuition" not in node for node in nodes.values())


def test_channels_have_nonfact_labels_and_latest_first(project):
    root, _, _, entry = project
    channels = build_channels(root)["channels"]
    assert len(channels) == 7
    assert next(ch for ch in channels if ch["kind"] == "report")["count"] == 0
    result = build_channel("route", root)
    assert result["count"] == 2
    assert result["entries"][0]["entry_id"] == entry.entry_id
    assert all(e["is_fact"] is False for e in result["entries"])
    with pytest.raises(KeyError):
        build_channel("nope", root)


def test_http_details_are_lazy_and_readonly(project, monkeypatch):
    root, store, ids, entry = project
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    client = TestClient(obs_app)
    original = Path.open
    reads = []
    def track(path, *args, **kwargs):
        if "proofs" in path.parts or "notes" in path.parts:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", track)
    for route in ("overview", "factgraph", "channels", "channel/route"):
        assert client.get("/api/" + route).status_code == 200
    assert reads == []
    response = client.get("/api/fact/" + ids[2])
    assert response.status_code == 200
    assert response.json()["proof"] == "Proof of top."
    assert response.json()["display_id"] == "F003"
    assert response.json()["display_name"] == "F003 · Top result."
    assert response.json()["predecessor_labels"][ids[1]] == "F002 · Middle result."
    assert response.json()["intuition"] == "the deep one"
    assert len(reads) == 2
    response = client.get("/api/entry/" + entry.entry_id)
    assert response.json()["details"] == "Long route explanation."
    assert response.json()["is_fact"] is False
    assert client.get("/api/channel/nope").status_code == 404
    assert client.get("/api/fact/not-a-fact").status_code == 404
    assert client.get("/api/entry/not-an-entry").status_code == 404
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_revoked_facts_disappear_and_associations_warn(project):
    root, store, ids, _ = project
    entry = store.record("experience", "Caution", "Needs the base.", "w", "worker",
                         links=[ids[0]], attributes={"subtype": "boundary"})
    store.revoke(ids[0], "Synthetic audit failure")
    assert {n["id"] for n in build_factgraph(root)["nodes"]} == {ids[3]}
    assert build_channel("experience", root)["entries"][0]["invalidated_links"] == [ids[0]]
    assert TestClient(obs_app).get("/api/fact/" + ids[0]).status_code == 404


def test_supersession_is_visible_but_does_not_break_old_dependencies(project):
    from hyprover.verify.review import review_fact
    root, store, ids, _ = project
    result = review_fact(store, ids[0], action="supersede", replacement_id=ids[3],
        reason="Synthetic implication and strengthening fixture, not mathematical evidence.",
        expected_revision=store.get(ids[0]).revision, author="w", allocator=lambda _: "obs-review",
        verifier=lambda **_: {"verdict": "correct", "verification_report": {
            "summary": "Synthetic review", "critical_errors": [], "gaps": []}, "repair_hints": ""})
    assert result["applied"]
    graph = build_factgraph(root)
    assert {n["id"] for n in graph["nodes"]} == set(ids)
    detail = TestClient(obs_app).get("/api/fact/" + ids[0]).json()
    assert detail["preferred_fact_id"] == ids[3]
    assert detail["supersession"]["replacement_id"] == ids[3]
    overview = build_overview(root)
    assert overview["verdicts"]["review_applied"] == 1
    assert overview["verdicts"]["not_published"] == 1
    store.revoke(ids[3], "Synthetic withdrawal")
    detail = TestClient(obs_app).get("/api/fact/" + ids[0]).json()
    assert detail["preferred_fact_id"] == ids[0]


def test_missing_and_legacy_notebooks_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_DASHBOARD_PROJECT", str(tmp_path))
    (tmp_path / "fact_graph").mkdir()
    before = sorted(tmp_path.iterdir())
    with pytest.raises(ValueError, match="unsupported_knowledge_format"):
        build_overview(tmp_path)
    assert TestClient(obs_app).get("/api/factgraph").status_code == 400
    assert sorted(tmp_path.iterdir()) == before


def test_previous_layout_dashboard_is_readonly_and_readable(project, monkeypatch):
    from dataclasses import replace
    from hyprover.core import knowledge_markdown as md
    root, store, ids, entry = project
    snap = store.snapshot()
    old = replace(snap, display_version=0,
                  facts={fid: replace(f, title="", display_id="") for fid, f in snap.facts.items()},
                  entries={eid: replace(e, display_id="") for eid, e in snap.entries.items()})
    store.path.write_text(md.serialize(old), encoding="utf-8")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original = Path.open
    def guarded(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        assert path.parent not in (root / "knowledge/proofs", root / "knowledge/notes")
        return original(path, mode, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        graph = build_factgraph(root)
        assert {n["display_id"] for n in graph["nodes"]} == {"F001", "F002", "F003", "F004"}
        rows = build_channel("route", root)["entries"]
        assert all(row["display_name"].startswith("R00") for row in rows)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_empty_new_notebook_is_valid(tmp_path):
    KnowledgeStore.initialize(tmp_path, "P")
    assert build_overview(tmp_path)["facts"] == 0
    assert build_factgraph(tmp_path)["nodes"] == []
    assert build_channel("route", tmp_path)["entries"] == []


def test_corrupt_notebook_is_reported(project):
    root, store, _, _ = project
    store.path.write_text("corrupt", encoding="utf-8")
    assert TestClient(obs_app).get("/api/overview").status_code == 400


def test_reports_are_separate_nonfacts_and_count_only_visible_entries(project, monkeypatch):
    root, store, ids, _ = project
    report = store.record("report", "Public explanation", "Synthetic report summary.", "writer", "main",
                          links=[ids[0]], details="Synthetic long explanation.")
    store.record("report", "Archived explanation", "Archived summary.", "writer", "main", status="archived")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    client = TestClient(obs_app)
    original = Path.open
    def summaries_only(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        assert "proofs" not in path.parts and "notes" not in path.parts
        return original(path, mode, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", summaries_only)
        page = client.get("/api/channel/report").json()
        assert page["total"] == page["count"] == 1
        assert page["entries"][0]["entry_id"] == report.entry_id
        assert page["entries"][0]["is_fact"] is False
        assert page["entries"][0]["related_labels"][ids[0]].startswith("F001")
        assert "details" not in page["entries"][0]
        assert client.get("/api/overview").json()["channel_counts"]["report"] == 1
        assert next(ch for ch in client.get("/api/channels").json()["channels"] if ch["kind"] == "report")["count"] == 1
        assert len(client.get("/api/factgraph").json()["nodes"]) == 4
        assert client.get("/api/channel/report", params={"include_archived": "true"}).json()["total"] == 2
    assert client.get("/api/entry/" + report.entry_id).json()["details"] == "Synthetic long explanation."
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_experience_filters_and_linked_display_ids(project):
    root, store, ids, _ = project
    wanted = store.record("experience", "Bridge method", "Synthetic keyword method.", "alice", "worker",
                          links=[ids[0]], attributes={"subtype": "method"})
    store.record("experience", "Bridge boundary", "Synthetic keyword boundary.", "bob", "worker",
                 links=[ids[0]], attributes={"subtype": "boundary"})
    store.record("experience", "Review", "Synthetic keyword review.", "alice", "worker",
                 links=[ids[1]], attributes={"subtype": "review", "audits": [{"target_id": ids[1],
                     "revision": store.get(ids[1]).revision, "scope": "Synthetic local check.",
                     "not_covered": "No mathematical verification performed."}]})
    client = TestClient(obs_app)
    for linked in (ids[0], "F001"):
        response = client.get("/api/channel/experience", params={"query": "keyword", "subtype": "method",
                              "author": "alice", "linked_to": linked, "sort_by": "created"})
        assert response.status_code == 200
        assert [row["entry_id"] for row in response.json()["entries"]] == [wanted.entry_id]
    assert client.get("/api/channel/experience", params={"subtype": "boundary"}).json()["total"] == 1
    assert client.get("/api/channel/experience", params={"subtype": "review"}).json()["total"] == 1


def test_channel_pagination_snapshot_conflicts_and_refresh(project):
    root, store, _, _ = project
    for i in range(5):
        store.record("report", f"Report {i}", "Synthetic report.", "writer", "main")
    client = TestClient(obs_app)
    first = client.get("/api/channel/report", params={"limit": 2}).json()
    assert first["total"] == first["count"] == 5
    assert first["offset"] == 0 and first["limit"] == 2 and first["has_more"]
    params = {"limit": 2, "offset": 2, "expected_snapshot_revision": first["snapshot_revision"]}
    second = client.get("/api/channel/report", params=params).json()
    assert second == client.get("/api/channel/report", params=params).json()
    assert not {e["entry_id"] for e in first["entries"]} & {e["entry_id"] for e in second["entries"]}
    last = client.get("/api/channel/report", params={**params, "offset": 4}).json()
    assert len(last["entries"]) == 1 and not last["has_more"]
    store.record("report", "New report", "Synthetic new summary.", "writer", "main")
    stale = client.get("/api/channel/report", params=params)
    assert stale.status_code == 409 and stale.json()["detail"] == "search_snapshot_conflict"
    assert client.get("/api/channel/report").json()["total"] == 6
    assert next(ch for ch in client.get("/api/channels").json()["channels"] if ch["kind"] == "report")["count"] == 6


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 1001}, {"offset": -1}, {"sort_by": "invalid"},
    {"subtype": "legacy"}, {"linked_to": "unknown-object"},
])
def test_invalid_channel_filters_fail_closed(project, params):
    assert TestClient(obs_app).get("/api/channel/experience", params=params).status_code == 400


def test_subtype_is_not_a_report_filter(project):
    assert TestClient(obs_app).get("/api/channel/report", params={"subtype": "method"}).status_code == 400


def test_missing_report_attachment_retains_summary(project):
    root, store, _, _ = project
    report = store.record("report", "Missing detail", "Keep this summary.", "writer", "main", details="Gone later.")
    (root / report.details_path).unlink()
    response = TestClient(obs_app).get("/api/entry/" + report.entry_id)
    assert response.status_code == 200
    assert response.json()["body"] == "Keep this summary."
    assert response.json()["details"] == ""
    assert response.json()["details_error"] == "missing_attachment"
    assert response.json()["is_fact"] is False


def test_unsafe_report_attachment_fails_closed(project, monkeypatch):
    from dataclasses import replace
    root, store, _, _ = project
    report = store.record("report", "Unsafe detail", "Synthetic summary.", "writer", "main", details="Safe detail.")
    snapshot = store.snapshot()
    unsafe = replace(report, details_path="../outside.md")
    snapshot = replace(snapshot, entries={**snapshot.entries, report.entry_id: unsafe})
    monkeypatch.setattr(KnowledgeStore, "snapshot", lambda self: snapshot)
    response = TestClient(obs_app).get("/api/entry/" + report.entry_id)
    assert response.status_code == 400
    assert "details" not in response.json()


def test_http_queries_do_not_select_another_project(project, tmp_path):
    root, store, _, _ = project
    other = tmp_path / "other"
    other_store = KnowledgeStore.initialize(other, "Other")
    other_store.record("report", "Other project", "Should not be visible.", "writer", "main")
    response = TestClient(obs_app).get("/api/channel/report", params={"project": str(other), "project_dir": str(other)})
    assert response.status_code == 200 and response.json()["total"] == 0


def test_legacy_experience_is_a_warning_not_a_fourth_subtype(project, monkeypatch):
    from dataclasses import replace
    root, store, _, _ = project
    entry = store.record("experience", "Historical note", "Unclassified original.", "writer", "worker",
                         attributes={"subtype": "method"})
    snapshot = store.snapshot()
    legacy = replace(entry, attributes={"subtype": "untrusted historical label"})
    snapshot = replace(snapshot, entries={**snapshot.entries, entry.entry_id: legacy})
    monkeypatch.setattr(KnowledgeStore, "snapshot", lambda self: snapshot)
    client = TestClient(obs_app)
    row = client.get("/api/channel/experience").json()["entries"][0]
    assert row["legacy_experience"] is True and row["is_fact"] is False
    assert row["attributes"]["subtype"] == "untrusted historical label"
    for subtype in ("method", "boundary", "review"):
        assert client.get("/api/channel/experience", params={"subtype": subtype}).json()["total"] == 0


def test_report_audit_warnings_match_summary_and_detail(project):
    root, store, ids, _ = project
    report = store.record("report", "Scoped synthetic report", "Not verification.", "writer", "main",
        links=[ids[0]], attributes={"audits": [{"target_id": ids[0], "revision": store.get(ids[0]).revision,
            "scope": "Synthetic metadata checks.", "not_covered": "Proof correctness."}]})
    store.revoke(ids[0], "Synthetic withdrawal")
    client = TestClient(obs_app)
    row = client.get("/api/channel/report").json()["entries"][0]
    detail = client.get("/api/entry/" + report.entry_id).json()
    assert row["warnings"] == detail["warnings"]
    assert row["audits"] == detail["audits"]
    assert row["audits"][0]["scope"] == "Synthetic metadata checks."
    assert row["audits"][0]["not_covered"] == "Proof correctness."
    assert row["audits"][0]["revoked"] is True
    assert row["audits"][0]["version_changed"] is True
    assert row["is_fact"] is False


def test_dashboard_node_dom_regressions():
    """Optional real-DOM client tests; not a substitute for browser layout/loading tests."""
    import os
    import shutil
    import subprocess

    node = os.environ.get("HYPROVER_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js 22 and optional jsdom/markdown-it/katex test dependencies required")
    dependencies = subprocess.run(
        [node, "-e", "require('jsdom');require('markdown-it');require('katex')"],
        capture_output=True, text=True, timeout=15,
    )
    if dependencies.returncode:
        if os.environ.get("HYPROVER_NODE"):
            pytest.fail("Node DOM dependencies unavailable: " + dependencies.stderr)
        pytest.skip("Optional DOM dependencies unavailable; set NODE_PATH and HYPROVER_NODE to run")
    for arguments in (["-e", _DASHBOARD_NODE_TESTS], ["--test", "dashboard.test.cjs"]):
        result = subprocess.run(
            [node, *arguments], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "# fail 0" in result.stdout, result.stdout + result.stderr


_DASHBOARD_NODE_TESTS = r"""
'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM, VirtualConsole } = require('jsdom');
const staticRoot = path.resolve('../static');
const html = fs.readFileSync(path.join(staticRoot, 'index.html'), 'utf8');
const app = fs.readFileSync(path.join(staticRoot, 'app.js'), 'utf8');
const scripts = {
  markdownit: fs.readFileSync(require.resolve('markdown-it/dist/markdown-it.min.js'), 'utf8'),
  katex: fs.readFileSync(require.resolve('katex/dist/katex.min.js'), 'utf8'),
};
const A = '0123456789abcdef', B = 'fedcba9876543210', E = 'entry-aaaaaaaaaaaaaaaa';
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const defer = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const fact = (id, proof = 'Current proof ' + id) => ({ id, display_name: id === A ? 'Fact A' : 'Fact B', author: 'writer', predecessors: [], statement: 'Statement ' + id, proof });
const entry = (body = 'Synthetic report', extra = {}) => ({ id: E, entry_id: E, kind: 'report', title: 'Report', body, author: 'writer', revision: 1, status: 'open', ...extra });
const page = (entries, extra = {}) => ({ entries, offset: 0, limit: 20, total: entries.length, has_more: false, snapshot_revision: 1, ...extra });
const overview = () => ({ project: 'fixture', updated_at: 1, facts: 2, facts_with_predecessors: 0, channel_counts: { report: 1 }, facts_by_author: { writer: 2 }, verdicts: {} });
async function settle() { for (let i = 0; i < 5; i++) await new Promise((done) => setImmediate(done)); }
function baseAPI(url) {
  if (url.pathname === '/api/overview') return response(overview());
  if (url.pathname === '/api/channels') return response({ channels: ['overview', 'experience', 'report'].map((kind) => ({ kind, count: 1 })) });
  if (url.pathname === '/api/factgraph') return response({ nodes: [fact(A), fact(B)], edges: [], max_depth: 0 });
  if (url.pathname.startsWith('/api/fact/')) return response(fact(url.pathname.split('/').pop()));
  if (url.pathname.startsWith('/api/channel/')) return response(page([entry()]));
  if (url.pathname === '/api/entry/' + E) return response(entry());
  return response({ detail: 'unknown object' }, 404);
}
async function boot(t, handler = () => undefined, libraries = true) {
  const errors = [], calls = [], console = new VirtualConsole();
  console.on('jsdomError', (error) => errors.push(error.message));
  const dom = new JSDOM(html, { url: 'http://dashboard.test/', runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole: console });
  const w = dom.window;
  w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  w.HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new w.Event('close')); };
  w.fetch = async (href) => {
    const url = new URL(href, w.location.href); calls.push(url);
    const value = await handler(url, calls.length);
    return value === undefined ? baseAPI(url) : value;
  };
  if (libraries) { w.eval(scripts.markdownit); w.eval(scripts.katex); }
  t.after(() => { w.close(); assert.deepEqual(errors, []); });
  require('node:vm').runInContext(app, dom.getInternalVMContext(), { filename: 'app.js' });
  await settle();
  const clickTab = async (name) => { w.document.querySelector(`[data-tab="${name}"]`).click(); await settle(); };
  return { w, document: w.document, calls, clickTab };
}
function loadRenderer(w, name) {
  w.eval(scripts[name]);
  const fragment = name === 'markdownit' ? 'markdown-it' : name;
  w.document.querySelector(`script[src*="${fragment}"]`).dispatchEvent(new w.Event('load'));
}

test('local client works without CDN; late Markdown and KaTeX refresh existing cards', async (t) => {
  const body = '**Visible** `$` and $x_a$';
  const { w, document, clickTab } = await boot(t, (url) => url.pathname.startsWith('/api/channel/') ? response(page([entry(body)])) : undefined, false);
  assert.equal(document.querySelectorAll('#ov-cards .card').length, 3);
  for (const script of document.querySelectorAll('script[data-renderer]')) { assert.ok(script.hasAttribute('async')); assert.ok(!script.hasAttribute('defer')); }
  assert.equal(document.querySelector('link[data-renderer-style]').getAttribute('media'), 'print');
  await clickTab('report');
  const rendered = document.querySelector('#report-list .entry-evidence');
  assert.equal(rendered.textContent, body);
  assert.ok(rendered.classList.contains('plain-text'));
  loadRenderer(w, 'markdownit');
  assert.equal(rendered.querySelector('strong').textContent, 'Visible');
  assert.equal(rendered.querySelector('code').textContent, '$');
  assert.equal(rendered.querySelector('.math-source').textContent, '$x_a$');
  loadRenderer(w, 'katex');
  assert.ok(rendered.querySelector('.katex'));
  assert.equal(document.querySelectorAll('#report-list .entry').length, 1);
  await clickTab('graph');
  assert.equal(document.querySelectorAll('#graph button').length, 2);
  document.querySelector('#graph button').click(); await settle();
  assert.match(document.querySelector('#fact-detail').textContent, /Current proof/);
});

test('math respects code, link targets, references and escaping; trust and HTML remain off', async (t) => {
  const body = '`$` and $x_a<y$ [link](https://example.org/$x$)\n\n```text\n$x$\n```\n\n\\(z_1\\) and \\[x+y\\]\n\n$\\invalidCommand{x}$\n\n<img src=x onerror=alert(1)>\n\n$\\href{javascript:alert(1)}{bad}$\n\n[bad](javascript:alert(1))\n\n$left [ref] $z$\n\n[ref]: https://example.org/$q$';
  const { document, clickTab } = await boot(t, (url) => url.pathname.startsWith('/api/channel/') ? response(page([entry(body)])) : undefined);
  await clickTab('report');
  const rendered = document.querySelector('#report-list .entry-evidence');
  assert.equal(rendered.querySelector('code').textContent, '$');
  assert.equal(rendered.querySelector('pre code').textContent.trim(), '$x$');
  assert.equal(rendered.querySelector('a[href="https://example.org/$x$"]').textContent, 'link');
  assert.ok(rendered.querySelector('a[href="https://example.org/$q$"]'));
  assert.ok(rendered.querySelectorAll('.katex').length >= 4);
  assert.match(rendered.querySelector('.math-source').textContent, /invalidCommand/);
  assert.equal(rendered.querySelector('img,script,[onerror],a[href^="javascript:"]'), null);
  assert.match(rendered.textContent, /<img src=x/);
});

test('internal anchors and digest attachments navigate APIs only; unsafe paths never navigate', async (t) => {
  const attachment = `knowledge/notes/${E}-2-0123456789abcdef.md`;
  const body = `[note](KNOWLEDGE.md#${E}) [proof](knowledge/proofs/${A}.md) [anchor](#fact-${A}) [digest](${attachment}) [old](knowledge/notes/${E}-1.md) [unknown](#entry-bbbbbbbbbbbbbbbb) [traversal](knowledge/proofs/../secret.md) [encoded](knowledge/proofs/%2e%2e/secret.md) [external](https://example.org/$x$) [evil](javascript:alert(1))`;
  const { w, document, calls, clickTab } = await boot(t, (url) => {
    if (url.pathname.startsWith('/api/channel/')) return response(page([entry(body)]));
    if (url.pathname === '/api/entry/' + E) return response(entry('Current note', { revision: 2, details_path: attachment, details: `[Back](../../KNOWLEDGE.md#${E})` }));
  });
  await clickTab('report');
  const link = (label) => [...document.querySelectorAll('#report-list a')].find((node) => node.textContent === label);
  for (const label of ['traversal', 'encoded']) assert.equal(link(label).getAttribute('href'), null);
  assert.equal(link('external').getAttribute('href'), 'https://example.org/$x$');
  link('note').click(); await settle();
  assert.match(document.querySelector('#entry-dialog-body').textContent, /Current note/);
  document.querySelector('#entry-close').click();
  link('digest').click(); await settle();
  assert.ok(document.querySelector('#entry-dialog-body a[href="#' + E + '"]'));
  document.querySelector('#entry-close').click();
  link('old').click(); await settle();
  assert.match(document.querySelector('#entry-dialog-body').textContent, /not the current revision/);
  document.querySelector('#entry-close').click();
  link('unknown').click(); await settle();
  assert.match(document.querySelector('#entry-dialog-body').textContent, /unavailable/);
  document.querySelector('#entry-close').click();
  link('proof').click(); await settle();
  assert.ok(document.querySelector('#tab-graph').classList.contains('active'));
  assert.match(document.querySelector('#fact-detail').textContent, /Current proof/);
  assert.ok(calls.every((url) => url.pathname.startsWith('/api/')));
  assert.equal(w.location.href, 'http://dashboard.test/');
  for (const value of ['../KNOWLEDGE.md#' + E, 'knowledge/proofs/' + A + '.md?x=1', 'knowledge/notes/' + E + '-0.md', 'knowledge/notes/' + E + '-2-nothex.md']) assert.equal(w.notebookTarget(value), null);
});

test('a completed selected proof is removed after successful graph refresh revokes its fact', async (t) => {
  let revoked = false;
  const { document, clickTab } = await boot(t, (url) => url.pathname === '/api/factgraph' ? response({ nodes: revoked ? [fact(B)] : [fact(A), fact(B)], edges: [], max_depth: 0 }) : undefined);
  await clickTab('graph');
  document.querySelector('#graph button').click(); await settle();
  assert.match(document.querySelector('#fact-detail').textContent, /Current proof/);
  revoked = true;
  await clickTab('overview'); await clickTab('graph');
  assert.doesNotMatch(document.querySelector('#fact-detail').textContent, /Current proof/);
  assert.match(document.querySelector('#fact-detail').textContent, /no longer verified/);
  assert.equal(document.querySelectorAll('#graph button').length, 1);
});

test('late proof responses cannot resurrect revoked facts or overwrite newer selection', async (t) => {
  let delayed = null, revoked = false;
  const { document, clickTab } = await boot(t, (url) => {
    if (url.pathname === '/api/factgraph') return response({ nodes: revoked ? [fact(B)] : [fact(A), fact(B)], edges: [], max_depth: 0 });
    if (url.pathname === '/api/fact/' + A && delayed) return delayed.promise;
  });
  await clickTab('graph');
  delayed = defer();
  document.querySelectorAll('#graph button')[0].click(); await settle();
  document.querySelectorAll('#graph button')[1].click(); await settle();
  delayed.resolve(response(fact(A, 'OLD RESPONSE'))); await settle();
  assert.match(document.querySelector('#fact-detail').textContent, /Fact B/);
  assert.doesNotMatch(document.querySelector('#fact-detail').textContent, /OLD RESPONSE/);
  delayed = defer();
  document.querySelectorAll('#graph button')[0].click(); await settle();
  revoked = true;
  await clickTab('overview'); await clickTab('graph');
  delayed.resolve(response(fact(A, 'REVOKED RESPONSE'))); await settle();
  assert.match(document.querySelector('#fact-detail').textContent, /no longer verified/);
  assert.doesNotMatch(document.querySelector('#fact-detail').textContent, /REVOKED RESPONSE/);
});

test('resource errors remain independent and older success cannot hide a newer failure', async (t) => {
  let overviewFail = true, graphFail = true, overviewDelay = null;
  const { w, document, clickTab } = await boot(t, (url) => {
    if (url.pathname === '/api/overview') return overviewDelay ? overviewDelay.promise : response(overviewFail ? { detail: 'failed' } : overview(), overviewFail ? 500 : 200);
    if (url.pathname === '/api/factgraph' && graphFail) return response({ detail: 'failed' }, 500);
  });
  assert.match(document.querySelector('#conn-banner').textContent, /Overview/);
  assert.equal(document.querySelector('#conn-banner').hidden, false);
  await clickTab('graph'); await w.refreshCounts();
  assert.match(document.querySelector('#conn-banner').textContent, /Overview.*Fact Graph/);
  overviewFail = false;
  await clickTab('overview');
  assert.doesNotMatch(document.querySelector('#conn-banner').textContent, /Overview/);
  assert.match(document.querySelector('#conn-banner').textContent, /Fact Graph/);
  graphFail = false; await clickTab('graph');
  assert.equal(document.querySelector('#conn-banner').hidden, true);
  const late = defer(); overviewDelay = late;
  const old = w.loadOverview(); await settle(); overviewDelay = null; overviewFail = true;
  await w.loadOverview(); late.resolve(response(overview())); await old;
  await w.refreshCounts();
  assert.equal(document.querySelector('#conn-banner').hidden, false);
  assert.match(document.querySelector('#conn-banner').textContent, /Overview/);
});

test('60 rapid tab/filter changes discard out-of-order pages and details', async (t) => {
  const held = [];
  let hold = false;
  const { w, document, clickTab } = await boot(t, (url) => {
    if (url.pathname.startsWith('/api/channel/')) {
      const value = response(page([entry(url.searchParams.get('query') || 'initial')]));
      if (hold) { const wait = defer(); held.push({ wait, value }); return wait.promise; }
      return value;
    }
  });
  await clickTab('report'); hold = true;
  for (let i = 0; i < 60; i++) {
    document.querySelector('[data-tab="' + (i % 2 ? 'report' : 'memory') + '"]').click();
    const prefix = i % 2 ? 'report' : 'mem';
    const form = document.querySelector('#' + prefix + '-browser form');
    form.elements.namedItem('query').value = 'revision-' + i;
    form.dispatchEvent(new w.Event('submit', { cancelable: true }));
  }
  await settle();
  assert.ok(held.length >= 120);
  for (const item of held.reverse()) item.wait.resolve(item.value);
  await settle();
  assert.equal(document.querySelector('#report-list .entry-evidence').textContent.trim(), 'revision-59');
  assert.equal(document.querySelectorAll('#report-list .entry').length, 1);
  assert.equal(document.querySelector('#report-list').getAttribute('aria-busy'), 'false');
  assert.equal(document.querySelector('[data-tab="report"]').getAttribute('aria-pressed'), 'true');
});

test('snapshot conflicts return to first page and detail replaces the whole current card', async (t) => {
  const requests = [];
  let revision = 1;
  const { w, document, clickTab } = await boot(t, (url) => {
    if (url.pathname.startsWith('/api/channel/')) {
      requests.push(url);
      if (url.searchParams.get('offset') === '20') { revision = 2; return response({ detail: 'search_snapshot_conflict' }, 409); }
      return response(page([entry('Summary v' + revision, { revision, details_path: `knowledge/notes/${E}-${revision}.md` })], { total: 25, has_more: true, snapshot_revision: revision }));
    }
    if (url.pathname === '/api/entry/' + E) return response(entry('Summary v3', { revision: 3, title: 'Changed title', details_path: `knowledge/notes/${E}-3.md`, details: 'Explanation v3', warnings: [{ code: 'target_changed' }] }));
  });
  await clickTab('report');
  document.querySelectorAll('#report-pager button')[1].click(); await settle();
  assert.equal(requests.at(-2).searchParams.get('expected_snapshot_revision'), '1');
  assert.equal(requests.at(-1).searchParams.get('offset'), '0');
  assert.equal(requests.at(-1).searchParams.has('expected_snapshot_revision'), false);
  assert.match(document.querySelector('#report-browser .browser-note').textContent, /first page/);
  document.querySelector('#report-list .more').click(); await settle();
  const card = document.querySelector('#report-list .entry');
  assert.match(card.textContent, /Changed title/);
  assert.match(card.textContent, /Summary v3/);
  assert.match(card.textContent, /Explanation v3/);
  assert.match(card.textContent, /target changed/);
  assert.doesNotMatch(card.textContent, /Summary v2/);
  assert.equal(document.querySelectorAll('#report-list .entry').length, 1);
});
"""
