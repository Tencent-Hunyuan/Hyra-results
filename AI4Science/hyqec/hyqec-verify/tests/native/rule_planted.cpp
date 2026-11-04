// Node-level stratum for the S0/S1/S2 rules at FULL buffer sizes, in any build variant (owner requirement 3,
// review finding 01). Each case builds a code TOGETHER WITH a node and a valid zero-syndrome completion of weight
// <= cap - w that respects the node's domains, so a prune of the node (packing or S0/S1/S2) is a false prune.
// Sizes are scaled to the variant's NMAX/MMAX/WMAX/CMAX so that every capacity limit is crossed (choices,
// local generators, occurrences, the anticommuting list, |A| > 64, S1's 31-variable bitmask, more than 2048
// touched qubits, the S2 table). Every node is evaluated on one reused state and on a fresh state (decisions
// must agree) and the cross-call scratch is checked after every call. Limit hits are counted by
// STABCERT_RULE_LIMIT. Usage: rule-planted DIR SEED CASES. Test-only.
#include <map>
#include <string>
static std::map<std::string, long> limit_hits;
#define STABCERT_RULE_LIMIT(name) (++limit_hits[name])
#define STABCERT_SOURCE_SHA256 "test-only-rule-planted"
#define STABCERT_RULES 1
#define main unused_native_main
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
#undef main
#include <memory>
#include <random>

static int dirty(const Code& c, const Dfs& st) {
  int count = 0;
  for (int q = 0; q < c.n; q++) count += st.qcnt[q] || st.qlast[q] || st.qmk[q] || st.qown[q];
  for (int g = 0; g < c.m; g++) count += st.gmap[g] != -1 || st.kmask[g];
  for (int i = 0; i < (c.n + 31) / 32; i++) count += st.used[i] != 0;
  return count;
}

struct Case {
  int n = 0;
  std::vector<std::vector<std::pair<int, int>>> rows;   // (qubit, label 1..3), ascending qubits
  std::vector<int> lab;                                  // node domains (0..7)
  int cap = 0;
  bool planted = true;                                   // a completion of weight <= cap - w exists
};

static int anti(int a, int b) { return a && b && a != b; }

// Build one planted case. r packed rows g_i = {d_i} U B_i (d_i decided, B_i disjoint candidate blocks);
// completion L = P_v + one anticommuting qubit per block + `extra` further qubits; every other row is random
// and then made to commute with L by toggling one label on supp(L).
static Case planted(std::mt19937_64& rng, int r, int slack, int extra, int block, int w, int nfill, int mrand,
                    int rowmax) {
  Case k;
  std::vector<int> lab;                  // node domains
  std::vector<int> L;                    // completion labels (0 = I)
  std::vector<int> deg;                  // column degree
  auto newq = [&](int l, int dom) { lab.push_back(dom); L.push_back(l); deg.push_back(0); return (int)lab.size() - 1; };
  auto rl = [&]() { return 1 + (int)(rng() % 3); };
  std::vector<int> D;
  for (int i = 0; i < w; i++) { int l = rl(); D.push_back(newq(l, l)); }          // decided, label l
  std::vector<std::vector<int>> B(r);
  std::vector<int> X;
  for (int i = 0; i < r; i++) for (int j = 0; j < block; j++) B[i].push_back(newq(0, 4));
  for (int i = 0; i < nfill; i++) {
    int u = rng() % 8, dom = u < 4 ? 4 : u < 6 ? 4 + rl() : 0;                     // free, restricted, or I
    X.push_back(newq(0, dom));
  }
  std::vector<std::vector<std::pair<int, int>>> rows;
  auto add = [&](std::vector<std::pair<int, int>> g) {
    std::sort(g.begin(), g.end());
    for (auto& e : g) deg[e.first]++;
    rows.push_back(g);
  };
  // Packed rows: one decided qubit with a label differing from P_v's (so the row anticommutes with P_v), plus B_i.
  for (int i = 0; i < r; i++) {
    int d = D[rng() % w];
    int gl = L[d] % 3 + 1;                                                        // != L[d]
    std::vector<std::pair<int, int>> g{{d, gl}};
    for (int q : B[i]) g.push_back({q, rl()});
    // completion qubit x_i in B_i with a label anticommuting with g_i there: g_i then commutes with L
    int x = rng() % block, q = B[i][x], gq = g[1 + x].second, lx = gq % 3 + 1;
    L[q] = lx; lab[q] = rng() % 2 ? 4 : 4 + lx;
    add(g);
  }
  // Extras: outside every block (a filler qubit) or a second qubit of a block with a label equal to g_j's there.
  for (int e = 0; e < extra; e++) {
    if (!X.empty() && rng() % 2) {
      int q = X[rng() % X.size()]; if (L[q]) continue;
      int l = rl(); L[q] = l; lab[q] = rng() % 2 ? 4 : 4 + l;
    } else {
      int i = rng() % r, x = rng() % block, q = B[i][x]; if (L[q]) continue;
      int l = rows[i][1 + x].second; L[q] = l; lab[q] = rng() % 2 ? 4 : 4 + l;
    }
  }
  int nq = lab.size();
  std::vector<int> supp; for (int q = 0; q < nq; q++) if (L[q]) supp.push_back(q);
  // Random rows over everything, biased to the blocks and decided qubits so A_v and the local-generator and
  // occurrence tables grow; each made to commute with L.
  for (int t = 0; t < mrand; t++) {
    int wt = 2 + rng() % (rowmax - 1);
    std::vector<std::pair<int, int>> g; std::vector<char> in(nq, 0);
    for (int tries = 0; (int)g.size() < wt && tries < 8 * wt; tries++) {
      int u = rng() % 4, q;
      if (u == 0) q = D[rng() % w];
      else if (u == 1) q = B[rng() % r][rng() % block];
      else q = rng() % nq;
      if (in[q] || deg[q] >= CMAX) continue;
      in[q] = 1; g.push_back({q, rl()});
    }
    if (g.size() < 2) continue;
    int par = 0; for (auto& e : g) par ^= anti(e.second, L[e.first]);
    if (par) {
      int fix = -1;
      for (size_t j = 0; j < g.size(); j++) if (L[g[j].first]) { fix = j; break; }
      if (fix < 0) continue;                                                        // cannot happen if par
      auto& e = g[fix]; e.second = anti(e.second, L[e.first]) ? L[e.first] : L[e.first] % 3 + 1;
    }
    add(g);
  }
  k.n = nq; k.rows = rows; k.lab = lab;
  int wtL = 0; for (int q : supp) wtL++;
  k.cap = w + r + slack;                    // budget r + slack >= completion weight r + (#extras placed)
  if (wtL - w > r + slack) k.planted = false;  // (never: extras <= slack by construction below)
  return k;
}

// Structured S1 cases (slack 1, one extra completion qubit E outside the blocks), blocks of WMAX - 2 qubits.
//  4b (S1 filter, nzl = 0; the packing is exactly the r block rows g_i = {d} U B_i):
//   - "even killer" rows {q, fillers of domain I...} for block qubits q != x_i, weight up to WMAX: even (no decided
//     qubit), flippable only by variable i; added first, so they are the first killers of most values and their
//     heavy supports make the filter's union of touched qubits exceed 2048 at NMAX 4096;
//   - "need" rows {d} U B_i (whole block: same candidate set as g_i, which wins the packing tie by index; x_i's
//     label anticommutes with L): odd, only variable i flips them; every value that does not flip one is killed,
//     which fills S1's killer lists (CSP_MAXOCC);
//   - a row {x_i, E}: x_i's planted value flips it and only E can repair it, so the no-E analysis wipes variable i
//     and S1's filter must find E (the planted completion is P_v + x_1..x_r + E).
//  4c (zl): small need rows {d, b, x_i} plus rows {d, E, fillers...} that no packed variable can flip -> more than
//     AMAX zl entries during S1's recorded analysis. (With a planted completion, more than AMAX unflippable odd rows
//     cannot all be unpacked: this limit is crossed at the perturbed, truth-free nodes, which are still checked for
//     reused == fresh and clean scratch.)
// Every row commutes with L (an even number of anticommuting overlaps with L).
static Case structured(std::mt19937_64& rng, int r, int nvars, int nneed, int neven, int evenw, int nzl, int zlw,
                       int nfill, bool small_need, int w, int nlate) {
  int block = WMAX - 2;
  Case k;
  std::vector<int> lab, L, deg;
  auto newq = [&](int l, int dom) { lab.push_back(dom); L.push_back(l); deg.push_back(0); return (int)lab.size() - 1; };
  auto rl = [&]() { return 1 + (int)(rng() % 3); };
  auto other = [&](int l) { return l % 3 + 1; };                        // a label anticommuting with l
  std::vector<int> D; for (int i = 0; i < w; i++) { int l = rl(); D.push_back(newq(l, l)); }
  std::vector<std::vector<int>> B(r); std::vector<int> x(r);
  for (int i = 0; i < r; i++) for (int j = 0; j < block; j++) B[i].push_back(newq(0, 4));
  int lE = rl(); int E = newq(lE, rng() % 2 ? 4 : 4 + lE);
  // fillers: domain I (4b), or also restricted domains (4c: they are candidates, which lets zl rows stay unpacked
  // when small need rows win the greedy packing)
  std::vector<int> F; for (int i = 0; i < nfill; i++) F.push_back(newq(0, small_need && rng() % 2 ? 4 + rl() : 0));
  auto add = [&](std::vector<std::pair<int, int>> g) {
    std::sort(g.begin(), g.end());
    for (auto& e : g) if (deg[e.first] >= CMAX) return;
    for (auto& e : g) deg[e.first]++;
    k.rows.push_back(g);
  };
  auto fillers = [&](std::vector<std::pair<int, int>>& g, int want) {
    std::vector<char> in(lab.size(), 0); for (auto& e : g) in[e.first] = 1;
    for (int tries = 0; (int)g.size() < want && tries < 4 * want && !F.empty(); tries++) {
      int q = F[rng() % F.size()]; if (in[q]) continue; in[q] = 1; g.push_back({q, rl()});
    }
  };
  std::vector<std::vector<std::pair<int, int>>> packed(r);
  for (int i = 0; i < r; i++) {
    int d = D[i % w];
    packed[i].push_back({d, other(L[d])});
    for (int q : B[i]) packed[i].push_back({q, rl()});
    int xi = rng() % block; x[i] = B[i][xi];
    L[x[i]] = other(packed[i][1 + xi].second); lab[x[i]] = rng() % 2 ? 4 : 4 + L[x[i]];
    add(packed[i]);
  }
  for (int i = 0; i < nvars && i < r; i++)
    for (int t = 0; t < neven; t++) {
      int q = B[i][rng() % block]; if (q == x[i]) continue;
      std::vector<std::pair<int, int>> g{{q, rl()}}; fillers(g, evenw); add(g);
    }
  for (int i = 0; i < nvars && i < r; i++) {
    for (int t = 0; t < nneed; t++) {
      int d = D[rng() % w];
      std::vector<std::pair<int, int>> g{{d, other(L[d])}};
      if (small_need) {                        // 4c: {d, b, x_i}
        int b = B[i][rng() % block]; if (b == x[i]) continue;
        g.push_back({b, rl()}); g.push_back({x[i], other(L[x[i]])});
      } else
        for (int q : B[i]) g.push_back({q, q == x[i] ? other(L[x[i]]) : rl()});
      add(g);
    }
    add({{x[i], other(L[x[i]])}, {E, other(lE)}});
  }
  for (int t = 0; t < nzl; t++) {
    int d = D[rng() % w];
    std::vector<std::pair<int, int>> g{{d, other(L[d])}, {E, other(lE)}}; fillers(g, zlw); add(g);
  }
  // 4d: late rows {d, f} with a restricted filler f and a label at f differing from f's domain label: odd, and
  // beyond the first AMAX anticommuting rows the greedy packing considers, so they stay unpacked and unowned:
  // more than AMAX zl entries in S1's recorded analysis even when column degrees are 64. (No planted completion
  // can flip them all: truth-free nodes, checked for reused == fresh and clean scratch.)
  for (int t = 0; t < nlate; t++) {
    int d = D[rng() % w], f = newq(0, 0); int lf = rl(); lab[f] = 4 + lf;
    add({{d, other(L[d])}, {f, other(lf)}});
  }
  k.n = lab.size(); k.lab = lab; k.cap = w + r + 1;
  k.planted = nlate == 0;      // late rows {d, f} need one completion qubit each: no low-weight completion
  return k;
}

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: rule-planted DIR SEED CASES\n"); return 2; }
  std::string dir = argv[1]; std::mt19937_64 rng(std::stoull(argv[2])); int cases = std::stoi(argv[3]);
  long calls = 0, false_prunes = 0, mismatches = 0, dirty_calls = 0, results[5] = {0, 0, 0, 0, 0}, skipped = 0;
  long maxn = 0, maxm = 0;
  std::vector<Case> cases_run;
  for (int t = 0; t < cases; t++) {
    if (cases_run.size() > 4) cases_run.erase(cases_run.begin());
    // Shape scaled to the build: big blocks cross the choice tables, many packed rows cross S1's bitmask,
    // many random rows on decided/block qubits cross the anticommuting list and the GAC tables.
    int kind = t % 5;
    int r, slack, block;
    if (kind == 4) {
      // 4a: zl-heavy (crosses the zl list); 4b: S1 filter (2-4 big blocks: records fit, killer lists and the
      // touched union grow; 5-6 blocks: the records table overflows)
      bool s1 = rng() % 2 != 0;                 // 4b, else 4c
      // r scaled to the row width: S1's records table (CSP_MAXCH values) overflows at about 1024 / (2 (WMAX - 2))
      // blocks, so narrow builds also get 8-17 blocks (records fit, many wiped variables fill the killer lists)
      // and 18-30 blocks (records overflow).
      int band = rng() % 3, rmax_n = (int)std::min<long>(40, (NMAX - 64) / (WMAX - 2));
      r = band == 0 ? 2 + rng() % 5 : band == 1 ? 8 + rng() % 10 : 18 + rng() % 13;
      if (r > rmax_n) r = std::max(2, rmax_n);
      // decided qubits: their column degree (<= CMAX) bounds the need rows, so narrow-column builds get more
      int w = CMAX <= 64 ? 3 + rng() % 8 : 3;
      if (w + r + 1 > DMAXW - 2) r = DMAXW - 2 - w - 1;
      int nlate = !s1 && rng() % 2 ? AMAX + 20 + rng() % AMAX : 0;
      slack = 1; block = WMAX - 2;
      int nvars = s1 ? 1 + rng() % r : 1 + rng() % 2;
      int nneed = s1 ? std::min((CMAX - 3) * std::max(1, w / 3), 40 + (int)(rng() % 160)) : 20 + (int)(rng() % 40);
      int neven = s1 ? (int)(rng() % (WMAX - 2)) : 0, evenw = 2 + rng() % (WMAX - 1);
      int nzl = s1 ? 0 : (rng() % 2 ? CMAX - 2 * nvars - 2 : 10 + rng() % 40), zlw = 2 + rng() % (WMAX - 1);
      long room = NMAX - w - 1 - (long)r * block - nlate;
      int nfill = room > 0 ? (int)std::min<long>(room, rng() % 2 ? room : 200 + rng() % 400) : 0;
      Case built = structured(rng, r, nvars, nneed, neven, evenw, nzl, zlw, nfill, !s1, w, nlate);
      if ((int)built.rows.size() > MMAX || built.n > NMAX) { skipped++; continue; }
      cases_run.push_back(std::move(built));
    } else {
      block = kind == 0 ? WMAX - 1 : kind == 1 ? 2 + rng() % 6 : 2 + rng() % (WMAX - 2);   // row weight block + 1 <= WMAX
      r = kind == 1 ? 32 + rng() % 12 : 1 + rng() % (kind == 0 ? 10 : 20);
      slack = rng() % 3; int extra = rng() % (slack + 1);
      int w = 1 + rng() % 3;
      if (w + r + slack > DMAXW - 2) r = DMAXW - 2 - w - slack;
      int nfill = std::min<long>(NMAX - w - (long)r * block - 1, rng() % 2 ? NMAX / 2 : 64 + rng() % 256);
      if (nfill < 0) { skipped++; continue; }
      int mrand = std::min<long>(MMAX - r - 1, kind == 3 ? MMAX - r - 1 : 50 + rng() % std::max<long>(1, MMAX / 4));
      int rowmax = std::max(2, std::min(WMAX, kind == 3 ? WMAX : 8 + (int)(rng() % 24)));
      cases_run.push_back(planted(rng, r, slack, extra, block, w, nfill, mrand, rowmax));
    }
    Case& k = cases_run.back();
    std::string path = dir + "/planted.inst";
    {
      std::ofstream f(path);
      f << k.n << ' ' << k.rows.size() << " 0\n";
      for (auto& g : k.rows) { f << g.size(); for (auto& e : g) f << ' ' << e.first << ' ' << e.second; f << '\n'; }
      for (int q = 0; q < k.n; q++) f << q << ' ';
      f << '\n' << k.n; for (int q = 0; q < k.n; q++) f << ' ' << q;
      f << '\n';
    }
    HostCode h = load(path.c_str());
    maxn = std::max<long>(maxn, h.n); maxm = std::max<long>(maxm, h.m);
    std::unique_ptr<Dfs> reused(new Dfs());
    for (int prune = 1; prune <= 3; prune++) {
      h.prune = prune; Code c = h.view();
      reused->init(c, k.cap);
      // Node sequence on the reused state: the planted node, a perturbed node (one block qubit forced to I,
      // truth unknown: only reuse == fresh is checked), and the planted node again.
      for (int step = 0; step < 3; step++) {
        std::vector<int> lab = k.lab;
        bool truth = step != 1;
        if (step == 1) { for (int q = 0; q < k.n; q++) if (lab[q] == 4 && rng() % 4 == 0) { lab[q] = 0; } }
        for (int q = 0; q < c.n; q++) if (reused->lab[q] != lab[q]) reused->change(c, q, reused->lab[q], lab[q]);
        if (reused->dead || reused->w + 1 > reused->cap) continue;
        int a = reused->csp_prune(c), d = dirty(c, *reused);
        std::unique_ptr<Dfs> fresh(new Dfs()); fresh->init(c, k.cap);
        for (int q = 0; q < c.n; q++) if (fresh->lab[q] != lab[q]) fresh->change(c, q, fresh->lab[q], lab[q]);
        int b = fresh->csp_prune(c);
        calls++; results[a]++; mismatches += a != b; dirty_calls += d > 0;
        if (truth && k.planted && a != 0) false_prunes++;
        if (a != b || d || (truth && k.planted && a)) {
          fprintf(stderr, "PROBLEM case %d prune=%d step=%d r=%d slack=%d block=%d n=%d m=%d: reused=%d fresh=%d dirty=%d\n",
                  t, prune, step, r, slack, block, h.n, h.m, a, b, d);
          if (false_prunes + mismatches + dirty_calls > 5) goto done;
        }
      }
    }
  }
done:
  printf("{\"NMAX\":%d,\"MMAX\":%d,\"WMAX\":%d,\"CMAX\":%d,\"calls\":%ld,\"false_prunes\":%ld,\"mismatches\":%ld,"
         "\"dirty_calls\":%ld,\"skipped\":%ld,\"max_n\":%ld,\"max_m\":%ld,\"results\":{\"none\":%ld,\"packing\":%ld,"
         "\"s0\":%ld,\"s1\":%ld,\"s2\":%ld},\"limits\":{",
         NMAX, MMAX, WMAX, CMAX, calls, false_prunes, mismatches, dirty_calls, skipped, maxn, maxm,
         results[0], results[1], results[2], results[3], results[4]);
  bool comma = false;
  for (auto& e : limit_hits) { printf("%s\"%s\":%ld", comma ? "," : "", e.first.c_str(), e.second); comma = true; }
  printf("}}\n");
  return false_prunes || mismatches || dirty_calls ? 1 : 0;
}
