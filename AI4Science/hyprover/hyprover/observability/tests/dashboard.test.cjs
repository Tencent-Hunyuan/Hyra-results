'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');
const markdownit = require('markdown-it');
const katex = require('katex');

const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const fid = '1111111111111111';
const eid = 'entry-2222222222222222';
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(setImmediate); };
const response = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
function entry(changes = {}) {
  return { id: eid, entry_id: eid, kind: 'report', title: 'Synthetic report', display_name: 'P001 · Synthetic report',
    body: 'Synthetic summary', author: 'alice', status: 'recorded', revision: 1, links: [],
    details_path: '', attributes: { research_policy_version: 1 }, audits: [], warnings: [],
    updated_at: '2026-10-29T00:00:00Z', is_fact: false, ...changes };
}
function page(entries, changes = {}) {
  return { kind: 'report', count: entries.length, total: entries.length, entries,
    offset: 0, limit: 20, has_more: false, snapshot_revision: 1, ...changes };
}
function app(t, options = {}) {
  const dom = new JSDOM(html, { url: 'http://localhost/', runScripts: 'outside-only', pretendToBeVisual: true });
  const w = dom.window;
  const calls = [];
  const state = { revoked: false, overviewError: false, rows: [entry()], custom: null };
  w.setInterval = () => 0;
  w.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  w.HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new w.Event('close')); };
  if (options.renderers) { w.markdownit = markdownit; w.katex = katex; }
  w.fetch = async (url) => {
    calls.push(String(url));
    if (state.custom) { const custom = state.custom(String(url)); if (custom !== undefined) return custom; }
    const u = new URL(url, 'http://localhost/');
    if (u.pathname === '/api/overview') return state.overviewError ? response({ detail: 'Synthetic outage' }, 503)
      : response({ project: 'Synthetic', facts: state.revoked ? 0 : 1, facts_with_predecessors: 0,
        facts_by_author: { alice: 1 }, channel_counts: { report: state.rows.length, experience: 0 }, verdicts: {}, updated_at: 1 });
    if (u.pathname === '/api/channels') return response({ channels: [
      { kind: 'overview', count: 0 }, { kind: 'experience', count: 0 }, { kind: 'report', count: state.rows.length },
    ] });
    if (u.pathname === '/api/factgraph') return response({ nodes: state.revoked ? [] : [{ id: fid, display_id: 'F001',
      display_name: 'Synthetic fact', author: 'alice', depth: 0 }], edges: [], max_depth: 0 });
    if (u.pathname === '/api/fact/' + fid) return state.revoked ? response({ detail: 'unknown or revoked fact' }, 404)
      : response({ id: fid, display_name: 'Synthetic fact', statement: 'SYNTHETIC STATEMENT', proof: 'OLD VERIFIED PROOF', predecessors: [] });
    if (u.pathname === '/api/entry/' + eid) return response(state.rows[0]);
    if (u.pathname === '/api/channel/report') return response(page(state.rows));
    if (u.pathname.startsWith('/api/channel/')) return response(page([], { kind: u.pathname.split('/').at(-1) }));
    return response({ detail: 'No such route' }, 404);
  };
  vm.runInContext(script, dom.getInternalVMContext());
  t.after(async () => { await flush(); w.close(); });
  return { w, state, calls, document: w.document };
}

test('same-origin interface starts with no CDN renderers and external resources do not defer it', async t => {
  const { w, document } = app(t);
  await flush();
  assert.equal(document.querySelectorAll('#ov-cards .card').length, 3);
  for (const library of document.querySelectorAll('script[data-renderer]')) {
    assert(library.hasAttribute('async'));
    assert(!library.hasAttribute('defer'));
  }
  assert.equal(document.querySelector('link[data-renderer-style]').media, 'print');
  w.switchTab('report');
  await flush();
  assert.match(document.querySelector('#report-list').textContent, /Synthetic report/);
});

test('math tokenization preserves code spans, fences and URLs containing dollars', t => {
  const { w, document } = app(t, { renderers: true });
  const node = document.createElement('div');
  w.mdmath(node, '`$` and $x$\n\n[link](https://example.org/$x$)\n\n```text\n$x$\n```');
  assert.equal(node.querySelector('code').textContent, '$');
  assert.equal(node.querySelector('pre code').textContent.trim(), '$x$');
  assert.equal(node.querySelectorAll('.katex').length, 1);
  assert.equal(node.querySelector('a').getAttribute('href'), 'https://example.org/$x$');
  assert(!node.textContent.includes('HYPROMATH'));
});

test('renderer failures and untrusted HTML never become executable markup', t => {
  const { w, document } = app(t, { renderers: true });
  const node = document.createElement('div');
  w.mdmath(node, '<img src=x onerror="window.pwned=1"> [bad](javascript:alert(1)) $\\notARealCommand$');
  assert.equal(node.querySelector('img'), null);
  assert.equal(node.querySelector('[onerror]'), null);
  assert.equal(node.querySelector('a[href^="javascript:"]'), null);
  assert.match(node.textContent, /notARealCommand/);
  assert.equal(w.pwned, undefined);
});

test('late renderers restore safely displayed original text', async t => {
  const { w, document } = app(t);
  const node = document.createElement('div'); document.body.appendChild(node);
  w.mdmath(node, '**Bold** and $x$ <img onerror="bad()">');
  assert.equal(node.querySelector('strong'), null);
  w.markdownit = markdownit; w.katex = katex;
  document.querySelector('script[data-renderer]').dispatchEvent(new w.Event('load'));
  assert.equal(node.querySelector('strong').textContent, 'Bold');
  assert.equal(node.querySelectorAll('.katex').length, 1);
  assert.equal(node.querySelector('img'), null);
});

test('revoked selected fact is cleared on a successful graph refresh', async t => {
  const { w, state, document } = app(t);
  await flush();
  w.switchTab('graph'); await flush();
  await w.showFact(fid);
  assert.match(document.querySelector('#fact-detail').textContent, /OLD VERIFIED PROOF/);
  state.revoked = true;
  w.switchTab('overview'); await flush();
  w.switchTab('graph'); await flush();
  assert(!document.querySelector('#fact-detail').textContent.includes('OLD VERIFIED PROOF'));
  assert.match(document.querySelector('#fact-detail').textContent, /no longer verified/);
});

test('channel success cannot hide an Overview error', async t => {
  const { w, state, document } = app(t);
  await flush();
  state.overviewError = true;
  await w.loadOverview();
  await w.refreshCounts();
  const banner = document.querySelector('#conn-banner');
  assert.equal(banner.hidden, false);
  assert.match(banner.textContent, /Overview/);
  state.overviewError = false;
  await w.loadOverview();
  assert.equal(banner.hidden, true);
});

test('notebook links use controlled APIs; arbitrary paths do not become file access', async t => {
  const { w, calls, document } = app(t, { renderers: true });
  await flush();
  const node = document.createElement('div'); document.body.appendChild(node);
  w.mdmath(node, `[note](KNOWLEDGE.md#${eid}) [proof](knowledge/proofs/${fid}.md) [unsafe](../secrets.md)`);
  const links = node.querySelectorAll('a');
  links[0].click(); await flush();
  assert(calls.includes('/api/entry/' + eid));
  assert.equal(document.querySelector('#entry-dialog').open, true);
  assert.equal(links[2].hasAttribute('href'), false);
  assert(!calls.some(url => url.startsWith('/KNOWLEDGE.md')));
  assert.equal(w.notebookTarget('../private/file.md'), null);
  assert.equal(w.notebookTarget('knowledge/notes/%2e%2e/private.md'), null);
});

test('recovery note filenames and attachment backlinks resolve to the owning entry', t => {
  const { w } = app(t);
  const file = `knowledge/notes/${eid}-2-abcdef0123456789.md`;
  const target = w.notebookTarget(file);
  assert(target);
  assert.equal(target.id, eid);
  assert.equal(target.attachmentPath, file);
  assert.equal(w.notebookTarget(`../../KNOWLEDGE.md#${eid}`, file).id, eid);
});

test('100 out-of-order filter responses cannot overwrite the latest list', async t => {
  const { w, state, document } = app(t);
  await flush(); w.switchTab('report'); await flush();
  const pending = [];
  state.custom = url => url.startsWith('/api/channel/report?') ? new Promise(resolve => pending.push(resolve)) : undefined;
  const promises = [];
  const started = performance.now();
  for (let i = 0; i < 100; i++) {
    document.querySelector('#report-browser [name=query]').value = `query-${i}`;
    promises.push(w.loadChannel('report'));
  }
  assert.equal(pending.length, 100);
  pending[99](response(page([entry({ display_name: 'LATEST REQUEST', body: 'Latest body' })])));
  await promises[99];
  for (let i = 98; i >= 0; i--) pending[i](response(page([entry({ display_name: `STALE ${i}` })])));
  await Promise.all(promises);
  const text = document.querySelector('#report-list').textContent;
  assert.match(text, /LATEST REQUEST/);
  assert(!text.includes('STALE'));
  console.log('UI_STRESS_METRICS ' + JSON.stringify({ responses: 100, stale_responses_ignored: 99,
    elapsed_ms: Math.round(performance.now() - started), environment: 'jsdom, not a real browser' }));
});

test('snapshot conflict returns to the first page with a visible explanation', async t => {
  const { w, state, document } = app(t);
  await flush(); w.switchTab('report'); await flush();
  state.custom = url => {
    if (!url.startsWith('/api/channel/report?')) return undefined;
    const params = new URL(url, 'http://localhost/').searchParams;
    return Number(params.get('offset')) > 0 ? response({ detail: 'search_snapshot_conflict' }, 409)
      : response(page([entry({ display_name: 'REFRESHED FIRST PAGE' })], { snapshot_revision: 2 }));
  };
  await w.loadChannel('report', { offset: 20, paginate: true });
  assert.match(document.querySelector('#report-list').textContent, /REFRESHED FIRST PAGE/);
  assert.match(document.querySelector('#report-browser').textContent, /Notebook changed/);
});

test('details update replaces the summary and warnings from the same response', async t => {
  const { w, state, document } = app(t);
  state.rows = [entry({ body: 'OLD SUMMARY', details_path: `knowledge/notes/${eid}-1.md` })];
  await flush(); w.switchTab('report'); await flush();
  state.rows = [entry({ body: 'CURRENT SUMMARY', revision: 2, details: 'CURRENT DETAILS',
    details_path: `knowledge/notes/${eid}-2.md`, warnings: [{ code: 'audit_target_updated' }] })];
  document.querySelector('#report-list button.more').click(); await flush();
  const text = document.querySelector('#report-list').textContent;
  assert(!text.includes('OLD SUMMARY'));
  assert.match(text, /CURRENT SUMMARY/);
  assert.match(text, /CURRENT DETAILS/);
  assert.match(text, /audit target updated/);
});
