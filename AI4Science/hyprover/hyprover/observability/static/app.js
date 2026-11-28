/* hyprover dashboard client. Overview / Fact Graph (echarts DAG) / Research notebook
   (per-channel). Read-only; polls the FastAPI app in app.py. */
'use strict';
const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = (s) => (s == null ? '' : String(s));

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const error = new Error('Request failed (' + r.status + ')');
    error.status = r.status;
    try { error.message = (await r.json()).detail || error.message; } catch (_) { /* keep status */ }
    throw error;
  }
  return r.json();
}
const resourceErrors = new Set();
function connError(resource, on) {
  if (on) resourceErrors.add(resource); else resourceErrors.delete(resource);
  const banner = $('#conn-banner');
  if (!banner) return;
  banner.hidden = resourceErrors.size === 0;
  banner.textContent = 'Cannot refresh: ' + [...resourceErrors].join(', ') + '. Previously loaded data may be stale.';
}

// ---- markdown + math ----------------------------------------------------- //
let _md = null;
const markdownSources = new WeakMap();
function mathInline(state, silent) {
  const start = state.pos, source = state.src;
  let open, close, display = false;
  if (source.startsWith('$$', start)) { open = close = '$$'; display = true; }
  else if (source[start] === '$') { open = close = '$'; }
  else if (source.startsWith('\\[', start)) { open = '\\['; close = '\\]'; display = true; }
  else if (source.startsWith('\\(', start)) { open = '\\('; close = '\\)'; }
  else return false;
  const begin = start + open.length;
  if (!display && /\s/.test(source[begin] || ' ')) return false;
  for (let end = begin; end < state.posMax; end++) {
    if (source.startsWith(close, end)) {
      if (end === begin || (!display && /\s/.test(source[end - 1]))) return false;
      if (!silent) {
        const token = state.push('notebook_math', 'span', 0);
        token.content = source.slice(begin, end);
        token.markup = source.slice(start, end + close.length);
        token.meta = { display };
      }
      state.pos = end + close.length;
      return true;
    }
    // Do not consume Markdown code or links; ordinary inequalities remain math.
    if (source[end] === '`' || (!display && source[end] === '\n')) return false;
    if (source[end] === '<' && /^<(?:[a-z][a-z\d+.-]*:|[^\s<>]+@)/i.test(source.slice(end))) return false;
    if (source[end] === '[') {
      const labelEnd = state.md.helpers.parseLinkLabel(state, end, true);
      const reference = state.md.utils.normalizeReference(source.slice(end + 1, labelEnd));
      if (labelEnd >= 0 && (source[labelEnd + 1] === '(' || source[labelEnd + 1] === '['
          || (state.env.references && Object.hasOwn(state.env.references, reference)))) return false;
    }
    if (source[end] === '\\') end++;
  }
  return false;
}
function md() {
  if (!_md && window.markdownit) {
    _md = window.markdownit({ html: false, linkify: true, breaks: false });
    _md.inline.ruler.before('escape', 'notebook_math', mathInline);
    _md.renderer.rules.notebook_math = (tokens, index) => {
      const token = tokens[index];
      if (window.katex) {
        try { return window.katex.renderToString(token.content, { displayMode: token.meta.display, throwOnError: true, trust: false }); }
        catch (_) { /* retain the original formula */ }
      }
      return '<span class="math-source">' + _md.utils.escapeHtml(token.markup) + '</span>';
    };
  }
  return _md;
}
function notebookTarget(href, sourcePath = '') {
  if (typeof href !== 'string' || /[%\\\\\u0000-\u0020\u007f]/.test(href)) return null;
  let path = href;
  if (/^knowledge\/(?:proofs\/[0-9a-f]{16}|notes\/(?:[0-9a-f]{16}-intuition|entry-[0-9a-f]{16}-[1-9][0-9]*(?:-[0-9a-f]{16})?))\.md$/.test(sourcePath)
      && path.startsWith('../../KNOWLEDGE.md#')) path = path.slice(6);
  const anchor = /^(?:(?:\.\/|\/)?KNOWLEDGE\.md)?#(fact-[0-9a-f]{16}|entry-[0-9a-f]{16})$/.exec(path);
  if (anchor) return { kind: anchor[1].startsWith('fact-') ? 'fact' : 'entry', id: anchor[1].replace(/^fact-/, '') };
  const proof = /^(?:\.\/|\/)?knowledge\/proofs\/([0-9a-f]{16})\.md$/.exec(path);
  if (proof) return { kind: 'fact', id: proof[1] };
  const intuition = /^(?:\.\/|\/)?knowledge\/notes\/([0-9a-f]{16})-intuition\.md$/.exec(path);
  if (intuition) return { kind: 'fact', id: intuition[1] };
  const note = /^(?:\.\/|\/)?(knowledge\/notes\/(entry-[0-9a-f]{16})-[1-9][0-9]*(?:-[0-9a-f]{16})?\.md)$/.exec(path);
  return note ? { kind: 'entry', id: note[2], attachmentPath: note[1] } : null;
}
function openNotebookTarget(target) {
  if (target.kind === 'fact') { switchTab('graph'); showFact(target.id); }
  else showEntry(target.id, target.attachmentPath);
}
function mdmath(node, text, sourcePath = '') {
  text = esc(text);
  markdownSources.set(node, { text, sourcePath });
  node.classList.add('md');
  node.classList.remove('plain-text');
  const fallback = () => { node.textContent = text; node.classList.add('plain-text'); };
  try {
    const m = md();
    if (!m) { fallback(); return; }
    node.innerHTML = m.render(text);
    for (const link of node.querySelectorAll('a[href]')) {
      const href = link.getAttribute('href'), target = notebookTarget(href, sourcePath);
      if (target) {
        link.setAttribute('href', '#' + (target.kind === 'fact' ? 'fact-' : '') + target.id);
        link.onclick = (event) => { event.preventDefault(); openNotebookTarget(target); };
      } else if (!/^(?:https?:\/\/|mailto:|\/\/)/i.test(href)) {
        link.removeAttribute('href');
        link.setAttribute('aria-disabled', 'true');
        link.title = 'Unsupported notebook link';
      } else link.setAttribute('rel', 'noopener noreferrer');
    }
  } catch (_) { fallback(); }
}
function refreshRenderers() {
  document.querySelectorAll('.md').forEach((node) => {
    const source = markdownSources.get(node);
    if (source) mdmath(node, source.text, source.sourcePath);
  });
  if (window.echarts && graphData && $('#tab-graph').classList.contains('active')) renderGraph(graphData);
}

// ---- tabs ---------------------------------------------------------------- //
function switchTab(name) {
  invalidateBrowsers();
  factRequest++;
  graphRequest++;
  overviewRequest++;
  document.querySelectorAll('.nav-link').forEach((a) => {
    a.classList.toggle('active', a.dataset.tab === name);
    a.setAttribute('aria-pressed', String(a.dataset.tab === name));
  });
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === 'tab-' + name));
  if (name === 'overview') loadOverview();
  if (name === 'graph') loadGraph();
  if (name === 'memory') loadMemory();
  if (name === 'report') { refreshCounts(); loadChannel('report'); }
  if (name === 'graph' && graphChart) setTimeout(() => graphChart.resize(), 50);
}
document.querySelectorAll('.nav-link').forEach((a) => (a.onclick = () => switchTab(a.dataset.tab)));

// ---- overview ------------------------------------------------------------ //
function bars(container, obj, colorFn) {
  container.innerHTML = '';
  const entries = Object.entries(obj).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map((e) => e[1]));
  for (const [k, v] of entries) {
    const row = el('div', 'bar-row');
    row.appendChild(el('div', 'bar-label', k));
    const track = el('div', 'bar-track');
    const fill = el('div', 'bar-fill');
    fill.style.width = (v / max * 100).toFixed(1) + '%';
    if (colorFn) fill.style.background = colorFn(k);
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el('div', 'bar-val', String(v)));
    container.appendChild(row);
  }
}
let overviewRequest = 0;
async function loadOverview() {
  const request = ++overviewRequest;
  try {
    const d = await api('/api/overview');
    if (request !== overviewRequest) return;
    connError('Overview', false);
    $('#project-badge').textContent = d.project || '';
    $('#refresh-note').textContent = 'updated ' + new Date(d.updated_at * 1000).toLocaleTimeString();
    const cards = [
      ['Verified facts', d.facts, `${d.facts_with_predecessors} with predecessors`],
      ['Non-factual entries', Object.values(d.channel_counts).reduce((a, b) => a + b, 0), `${Object.keys(d.channel_counts).length} channels`],
      ['Verification submissions', (d.verdicts.published || 0) + (d.verdicts.not_published || 0), `${d.verdicts.published || 0} published · ${d.verdicts.not_published || 0} not published`],
    ];
    const c = $('#ov-cards'); c.innerHTML = '';
    for (const [k, v, sub] of cards) {
      const card = el('div', 'card');
      card.appendChild(el('div', 'k', k));
      card.appendChild(el('div', 'v', String(v)));
      card.appendChild(el('div', 'sub', sub));
      c.appendChild(card);
    }
    bars($('#ov-channels'), d.channel_counts);
    bars($('#ov-authors'), d.facts_by_author);
    const vd = $('#ov-verdicts'); vd.innerHTML = '';
    const chips = el('div', 'chips');
    const vmap = { correct: 'var(--green)', wrong: 'var(--red)', error: 'var(--orange)' };
    for (const [k, v] of Object.entries(d.verdicts)) {
      const chip = el('div', 'chip');
      const dot = el('span', 'dot'); dot.style.background = vmap[k] || 'var(--text-muted)';
      chip.appendChild(dot); chip.appendChild(document.createTextNode(`${k}: ${v}`));
      chips.appendChild(chip);
    }
    vd.appendChild(chips);
  } catch (e) { if (request === overviewRequest) connError('Overview', true); }
}

// ---- fact graph (echarts) ------------------------------------------------ //
let graphChart = null;
let graphData = null;
let graphRequest = 0;
let factById = {};
let selectedFactId = null;
// importance = dependency depth (longest path from an axiom/leaf up to a fact).
// Continuous shade: the deeper a fact, the darker — shallow=light, deep=dark.
function depthColor(depth, maxDepth) {
  const t = maxDepth > 0 ? Math.min(1, depth / maxDepth) : 0;
  return `hsl(248, ${(50 + t * 26).toFixed(0)}%, ${(80 - t * 52).toFixed(0)}%)`;
}
async function loadGraph() {
  const request = ++graphRequest;
  factRequest++;
  if (selectedFactId) $('#fact-detail').textContent = 'Checking whether the selected fact is still verified…';
  try {
    const d = await api('/api/factgraph');
    if (request !== graphRequest) return;
    graphData = d;
    factById = Object.fromEntries(d.nodes.map((node) => [node.id, node]));
    if (selectedFactId) {
      if (Object.hasOwn(factById, selectedFactId)) showFact(selectedFactId);
      else {
        factRequest++;
        selectedFactId = null;
        $('#fact-detail').textContent = 'Fact unavailable or no longer verified.';
      }
    }
    renderGraph(d);
    connError('Fact Graph', false);
  } catch (e) {
    if (request !== graphRequest) return;
    connError('Fact Graph', true);
    graphData = null; factById = {}; factRequest++;
    $('#fact-detail').textContent = 'Cannot confirm the current facts. Reload Fact Graph to retry.';
  }
}
function renderGraph(d) {
    $('#graph-stat').textContent = `${d.nodes.length} facts · ${d.edges.length} edges · max depth ${d.max_depth}`;
    if (!window.echarts) {
      const fallback = $('#graph'); fallback.replaceChildren(el('p', 'section-note', 'Graph renderer unavailable. Select a verified fact below.'));
      d.nodes.forEach((fact) => {
        const button = el('button', 'pred-link', fact.display_name || fact.id);
        button.onclick = () => showFact(fact.id); fallback.appendChild(button);
      });
      return;
    }
    const maxD = d.max_depth || 1;
    const nodes = d.nodes.map((n) => {
      const dp = n.depth || 0;
      return {
        id: n.id, name: n.display_id || n.id.slice(0, 7),
        symbolSize: 6 + Math.min(16, dp * 2.5),        // bigger = deeper, but capped small so circles don't overlap when zoomed out
        itemStyle: { color: depthColor(dp, maxD) },
        depth: dp, author: n.author,                   // no per-node label (would clutter)
      };
    });
    const links = d.edges.map((e) => ({ source: e.source, target: e.target }));
    if (!graphChart) { $('#graph').replaceChildren(); graphChart = window.echarts.init($('#graph')); }
    graphChart.setOption({
      tooltip: {
        formatter: (p) => {
          if (p.dataType !== 'node') return '';
          const fact = factById[p.data.id] || {};
          const tip = el('div');
          tip.appendChild(el('b', null, fact.display_name || p.data.name));
          tip.appendChild(el('div', null, `by ${p.data.author} · dependency depth: ${p.data.depth}`));
          tip.appendChild(el('div', null, esc(fact.statement).slice(0, 160)));
          return tip;
        },
      },
      series: [{
        type: 'graph', layout: 'force', roam: true, draggable: true,
        force: { repulsion: 160, edgeLength: 80, gravity: 0.06 },
        label: { show: false }, emphasis: { focus: 'adjacency', label: { show: false } },
        lineStyle: { color: '#cbd5e1', width: 1, opacity: 0.55, curveness: 0.05 },
        edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 5,
        data: nodes, links: links,
      }],
    });
    graphChart.off('click');
    graphChart.on('click', (p) => { if (p.dataType === 'node') showFact(p.data.id); });
    graphChart.resize();
}
let factRequest = 0;
async function showFact(id) {
  const request = ++factRequest;
  selectedFactId = id;
  const d = $('#fact-detail'); d.textContent = 'Loading proof details…';
  let f;
  try { f = await api('/api/fact/' + encodeURIComponent(id)); }
  catch (error) { if (request === factRequest) d.textContent = 'Fact unavailable or no longer verified.'; return; }
  if (request !== factRequest) return;
  d.innerHTML = '';
  d.appendChild(el('div', 'fid', f.display_name || f.title || f.id));
  const metadata = el('details');
  metadata.appendChild(el('summary', 'muted', 'Internal ID'));
  metadata.appendChild(el('code', null, f.id));
  d.appendChild(metadata);
  if (f.author || f.problem_id) d.appendChild(el('div', 'muted', `${f.author || '?'} · ${f.problem_id || ''}`));
  const addSec = (h, txt, sourcePath = '') => { if (!txt) return; d.appendChild(el('div', 'sec-h', h)); const m = el('div', 'math'); mdmath(m, txt, sourcePath); d.appendChild(m); };
  if (f.supersession && f.supersession.replacement_id) {
    const usable = f.preferred_fact_id !== f.id;
    d.appendChild(el('div', 'muted', usable
      ? '由更强结果替代推荐；原结论仍有效。'
      : '替代依据已失效，原结论仍有效。'));
    if (usable) {
      const next = el('button', 'pred-link', f.preferred_fact_label);
      next.onclick = () => showFact(f.preferred_fact_id); d.appendChild(next);
    }
    addSec('Replacement reason', f.supersession.reason);
  }
  addSec('Statement', f.statement);
  addSec('Proof', f.proof, `knowledge/proofs/${f.id}.md`);
  addSec('Intuition', f.intuition, `knowledge/notes/${f.id}-intuition.md`);
  if (f.predecessors.length) {
    d.appendChild(el('div', 'sec-h', `Predecessors (${f.predecessors.length})`));
    const wrap = el('div');
    f.predecessors.forEach((p) => {
      const a = el('button', 'pred-link', (f.predecessor_labels || {})[p] || p.slice(0, 10));
      a.title = p; a.onclick = () => showFact(p); wrap.appendChild(a);
    });
    d.appendChild(wrap);
  }
}

// ---- research notes and public reports ----------------------------------- //
const LONGFORM = new Set(['overview', 'route', 'feedback', 'experience', 'report']);
const browsers = {
  mem: { kind: 'overview', request: 0, snapshot: null, offset: 0 },
  report: { kind: 'report', request: 0, snapshot: null, offset: 0 },
};
let countsRequest = 0;
let entryRequest = 0;
function invalidateBrowsers() {
  Object.values(browsers).forEach((state) => { state.request++; });
  entryRequest++;
  if ($('#entry-dialog').open) $('#entry-dialog').close();
}
function activeBrowser(prefix) {
  return $('#tab-' + (prefix === 'mem' ? 'memory' : 'report')).classList.contains('active');
}
function updateSubtabs() {
  document.querySelectorAll('#mem-subtabs .subtab').forEach((button) => {
    const selected = button.dataset.kind === browsers.mem.kind;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}
async function refreshCounts() {
  const request = ++countsRequest;
  try {
    const d = await api('/api/channels');
    if (request !== countsRequest) return;
    const subtabs = $('#mem-subtabs');
    for (const channel of d.channels) {
      if (channel.kind === 'report') { $('#report-count').textContent = String(channel.count); continue; }
      let button = [...subtabs.children].find((item) => item.dataset.kind === channel.kind);
      if (!button) {
        button = el('button', 'subtab'); button.type = 'button'; button.dataset.kind = channel.kind;
        button.appendChild(document.createTextNode(channel.kind));
        button.appendChild(el('span', 'cnt'));
        button.onclick = () => loadChannel(channel.kind);
        subtabs.appendChild(button);
      }
      button.querySelector('.cnt').textContent = String(channel.count);
    }
    updateSubtabs();
    $('#refresh-note').textContent = 'counts updated ' + new Date().toLocaleTimeString();
    connError('Channel counts', false);
  } catch (_) { if (request === countsRequest) connError('Channel counts', true); }
}
function setupBrowser(prefix) {
  const state = browsers[prefix];
  if (state.form) return;
  const container = $('#' + prefix + '-browser');
  const form = el('form', 'filters'); form.setAttribute('aria-label', prefix === 'report' ? 'Report filters' : 'Note filters');
  const field = (name, label, options) => {
    const wrap = el('label', null, label);
    const control = el(options ? 'select' : 'input'); control.name = name;
    if (options) options.forEach(([value, text]) => { const option = el('option', null, text); option.value = value; control.appendChild(option); });
    else control.type = 'search';
    wrap.appendChild(control); form.appendChild(wrap); return control;
  };
  field('query', 'Literal text or ID (optional)');
  field('author', 'Author (exact)');
  field('linked_to', 'Related object ID');
  const subtype = field('subtype', 'Experience type', [['', 'All types'], ['method', 'method'], ['boundary', 'boundary'], ['review', 'review']]);
  subtype.parentElement.hidden = true;
  field('sort_by', 'Sort', [['updated', 'Recently updated'], ['created', 'Recently created']]);
  field('limit', 'Per page', [['20', '20'], ['50', '50'], ['100', '100']]);
  const archived = el('label', 'check-filter');
  const checkbox = el('input'); checkbox.type = 'checkbox'; checkbox.name = 'include_archived';
  archived.append(checkbox, document.createTextNode('Include archived')); form.appendChild(archived);
  const apply = el('button', null, 'Apply filters'); apply.type = 'submit';
  const refresh = el('button', null, 'Refresh'); refresh.type = 'button';
  const reset = el('button', null, 'Reset'); reset.type = 'button';
  form.append(apply, refresh, reset);
  form.onsubmit = (event) => { event.preventDefault(); loadChannel(state.kind); };
  form.onchange = (event) => { if (event.target.matches('select, [type="checkbox"]')) loadChannel(state.kind); };
  refresh.onclick = () => { refreshCounts(); loadChannel(state.kind); };
  reset.onclick = () => { form.reset(); loadChannel(state.kind); };
  state.form = form;
  state.note = el('div', 'browser-note'); state.note.setAttribute('role', 'status');
  state.hint = el('p', 'section-note');
  container.append(state.hint, form, state.note);
}
function loadMemory() {
  refreshCounts();
  loadChannel(browsers.mem.kind);
}
function renderPager(prefix, page, busy = false) {
  const state = browsers[prefix], pager = $('#' + prefix + '-pager');
  const status = el('span', 'page-status', busy ? 'Loading…' : page
    ? `${page.total ? page.offset + 1 : 0}–${page.offset + page.entries.length} of ${page.total}` : '');
  status.setAttribute('role', 'status');
  const previous = el('button', null, 'Previous'); previous.type = 'button';
  const next = el('button', null, 'Next'); next.type = 'button';
  previous.disabled = busy || !page || page.offset === 0;
  next.disabled = busy || !page || !page.has_more;
  previous.onclick = () => loadChannel(state.kind, { offset: Math.max(0, page.offset - page.limit), paginate: true });
  next.onclick = () => loadChannel(state.kind, { offset: page.offset + page.limit, paginate: true });
  pager.replaceChildren(status, previous, next);
}
async function loadChannel(kind, { offset = 0, paginate = false, notice = '' } = {}) {
  const prefix = kind === 'report' ? 'report' : 'mem';
  const state = browsers[prefix]; setupBrowser(prefix);
  if (state.kind !== kind) { state.form.reset(); state.snapshot = null; }
  state.kind = kind;
  const subtype = state.form.elements.namedItem('subtype');
  subtype.disabled = kind !== 'experience'; subtype.parentElement.hidden = subtype.disabled;
  state.hint.textContent = kind === 'experience'
    ? 'NOT A FACT. method / boundary / review are the three experience types. Historical unclassified entries are shown with a warning, not a fourth type.'
    : kind === 'report' ? '' : 'Research notes are NOT A FACT.';
  if (!paginate) { state.filters = new URLSearchParams(new FormData(state.form)); state.snapshot = null; }
  const params = new URLSearchParams(state.filters);
  params.set('include_archived', state.filters.has('include_archived') ? 'true' : 'false');
  for (const key of ['subtype', 'author', 'linked_to']) if (!params.get(key)) params.delete(key);
  params.set('offset', String(offset));
  if (state.snapshot != null) params.set('expected_snapshot_revision', String(state.snapshot));
  const request = ++state.request;
  const current = () => request === state.request && activeBrowser(prefix);
  const list = $('#' + prefix + '-list');
  list.replaceChildren(el('div', 'empty', 'Loading…')); list.setAttribute('aria-busy', 'true');
  state.note.textContent = notice;
  renderPager(prefix, null, true); updateSubtabs();
  try {
    const page = await api('/api/channel/' + encodeURIComponent(kind) + '?' + params);
    if (!current()) return;
    state.snapshot = page.snapshot_revision; state.offset = page.offset;
    list.replaceChildren();
    if (!page.entries.length) list.appendChild(el('div', 'empty', 'No entries match these filters.'));
    for (const entry of page.entries) list.appendChild(renderEntry(entry, current));
    renderPager(prefix, page);
  } catch (error) {
    if (!current()) return;
    if (error.status === 409 && !notice) {
      state.snapshot = null;
      refreshCounts();
      return loadChannel(kind, { paginate: true, notice: 'Notebook changed. Returned to the first page to avoid missing or duplicate entries.' });
    }
    list.replaceChildren(el('div', 'empty', 'Cannot load entries: ' + error.message));
    renderPager(prefix, null);
  } finally { if (current()) list.setAttribute('aria-busy', 'false'); }
}
function relatedButton(id, label) {
  const button = el('button', 'pred-link', label || id); button.type = 'button'; button.title = id;
  button.onclick = () => {
    if (/^[0-9a-f]{16}$/.test(id)) {
      if ($('#entry-dialog').open) $('#entry-dialog').close();
      switchTab('graph'); showFact(id);
    } else showEntry(id);
  };
  return button;
}
function renderEntry(entry, current = () => true, detailed = false) {
  const card = el('article', 'entry' + (LONGFORM.has(entry.kind) ? ' longform' : ''));
  const head = el('div', 'entry-head');
  head.append(el('span', 'entry-author', entry.author || '?'), el('span', 'tag', 'NOT A FACT'));
  if (entry.status) head.appendChild(el('span', 'tag', entry.status));
  const attributes = entry.attributes || {};
  if (entry.kind === 'experience' && attributes.research_policy_version === 1
      && ['method', 'boundary', 'review'].includes(attributes.subtype)) {
    head.appendChild(el('span', 'tag', attributes.subtype));
  }
  head.appendChild(el('span', 'entry-ts', (entry.updated_at || '').slice(0, 19).replace('T', ' ')));
  card.append(head, el('h3', 'entry-claim', entry.display_name || entry.title));
  if (entry.legacy_experience) card.appendChild(el('p', 'entry-warning', 'Historical unclassified experience — not assigned to method, boundary or review.'));
  if (entry.links && entry.links.length) {
    const related = el('div', 'muted', 'Related: ');
    entry.links.forEach((id) => related.appendChild(relatedButton(id, (entry.related_labels || {})[id])));
    card.appendChild(related);
  }
  if (entry.invalidated_links && entry.invalidated_links.length) {
    const labels = entry.invalidated_links.map((id) => (entry.related_labels || {})[id] || id);
    card.appendChild(el('p', 'entry-warning', 'Referenced facts revoked: ' + labels.join(', ')));
  }
  for (const warning of entry.warnings || []) {
    const target = warning.target_id ? ' · ' + ((entry.related_labels || {})[warning.target_id] || warning.target_id) : '';
    card.appendChild(el('p', 'entry-warning', esc(warning.code).replace(/_/g, ' ') + target));
  }
  for (const audit of entry.audits || []) {
    const section = el('section', 'audit-info');
    section.appendChild(el('strong', null, 'Reported review scope · NOT A FACT / not certification'));
    section.appendChild(relatedButton(audit.target_id, audit.display_name));
    const fields = [['Reviewed revision', audit.revision], ['Current revision', audit.current_revision],
      ['Scope', audit.scope], ['Not covered', audit.not_covered], ['Target status', audit.status]];
    const dl = el('dl');
    for (const [label, value] of fields) {
      dl.append(el('dt', null, label), el('dd', null, Array.isArray(value) ? value.join('\n') : esc(value)));
    }
    section.appendChild(dl);
    if (audit.historical) section.appendChild(el('p', 'entry-warning', 'Historical review; not a review of the current target.'));
    if (audit.version_changed) section.appendChild(el('p', 'entry-warning', 'Target version changed since this review.'));
    if (audit.revoked) section.appendChild(el('p', 'entry-warning', 'Reviewed target has been revoked.'));
    card.appendChild(section);
  }
  const body = el('div', 'entry-evidence'); mdmath(body, entry.body); card.appendChild(body);
  const details = el('div', 'entry-details');
  if (detailed) appendDetails(details, entry);
  if (entry.details_path && (!detailed || entry.details_error)) {
    const more = el('button', 'more', entry.details_error ? 'Retry explanation' : 'Read explanation');
    more.type = 'button'; more.setAttribute('aria-expanded', 'false');
    more.onclick = async () => {
      more.disabled = true;
      try {
        const full = await api('/api/entry/' + encodeURIComponent(entry.entry_id || entry.id));
        if (!current() || !card.isConnected) return;
        const replacement = renderEntry(full, current, true);
        if (full.revision !== entry.revision) {
          const notice = el('p', 'entry-warning', 'Entry changed since the list loaded. Summary, explanation and review warnings now show the same current revision. Refresh the list to update its order and counts.');
          notice.setAttribute('role', 'status'); replacement.prepend(notice);
        }
        card.replaceWith(replacement);
        replacement.tabIndex = -1;
        replacement.focus({ preventScroll: true });
      } catch (error) {
        if (!current() || !card.isConnected) return;
        details.replaceChildren(el('p', 'entry-warning', 'Explanation unavailable. Summary retained. ' + error.message));
        more.disabled = false; more.textContent = 'Retry explanation';
      }
    };
    card.appendChild(more);
  }
  card.appendChild(details);
  return card;
}
function appendDetails(container, entry) {
  container.replaceChildren();
  if (entry.details_error) {
    container.appendChild(el('p', 'entry-warning', 'Explanation attachment is missing. Summary retained.'));
  } else if (entry.details) {
    const note = el('div', 'entry-evidence'); mdmath(note, entry.details, entry.details_path); container.appendChild(note);
  }
}
async function showEntry(id, attachmentPath = '') {
  const request = ++entryRequest, dialog = $('#entry-dialog'), body = $('#entry-dialog-body');
  if (!dialog.open) dialog.showModal();
  body.replaceChildren(el('p', 'empty', 'Loading related note…'));
  const current = () => request === entryRequest && dialog.open;
  try {
    const entry = await api('/api/entry/' + encodeURIComponent(id));
    if (!current()) return;
    if (attachmentPath && entry.details_path !== attachmentPath) {
      body.replaceChildren(el('p', 'entry-warning', 'This explanation attachment is not the current revision. Open the current note instead.'), relatedButton(id, entry.display_name));
      return;
    }
    body.replaceChildren(renderEntry(entry, current, true));
  } catch (error) {
    if (current()) body.replaceChildren(el('p', 'entry-warning', 'Related note unavailable: ' + error.message));
  }
}
$('#entry-close').onclick = () => { entryRequest++; $('#entry-dialog').close(); };
$('#entry-dialog').addEventListener('close', () => { entryRequest++; });

// ---- init + polling ------------------------------------------------------ //
document.querySelectorAll('script[data-renderer]').forEach((script) => script.addEventListener('load', refreshRenderers));
document.querySelectorAll('link[data-renderer-style]').forEach((link) => {
  if (link.sheet) link.media = 'all';
  else link.addEventListener('load', () => { link.media = 'all'; }, { once: true });
});
loadOverview();
refreshCounts();
window.addEventListener('resize', () => { if (graphChart) graphChart.resize(); });
setInterval(() => {
  const active = document.querySelector('.nav-link.active');
  if (active && active.dataset.tab === 'overview') loadOverview();
  refreshCounts();
}, 15000);
