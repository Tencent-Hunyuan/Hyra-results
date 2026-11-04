// Node-level regressions for the count-only S0/S1/S2 rules (docs/SOUNDNESS.md section 5.1).
// Each case fixes a code and a node (qubit domains, cap) taken from an independent review, calls
// the shipped rule code directly and compares with complete enumeration of the node's completions.
// This translation unit is test-only and never part of a release binary.
#define STABCERT_SOURCE_SHA256 "test-only-rule-cases"
#ifndef STABCERT_RULES
#define STABCERT_RULES 1   // the rules build: every policy, including the opt-in S0/S1/S2
#endif
#define main unused_native_main
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
#undef main
#include <iostream>
#include <functional>
#include <sstream>

static std::string directory;

// Generators as label vectors (0=I 1=X 2=Z 3=Y); written in the plain instance format and
// loaded through the shipped parser, without logicals (the rules never read them).
static HostCode make(const char* name, const std::vector<std::vector<int>>& G) {
  int n = G.empty() ? 0 : G[0].size();
  std::string path = directory + "/" + name + ".inst";
  std::ofstream f(path);
  f << n << ' ' << G.size() << " 0\n";
  for (const auto& g : G) {
    int r = 0; for (int l : g) r += l != 0;
    f << r; for (int q = 0; q < n; ++q) if (g[q]) f << ' ' << q << ' ' << g[q];
    f << '\n';
  }
  for (int q = 0; q < n; ++q) f << q << ' ';
  f << '\n' << n; for (int q = 0; q < n; ++q) f << ' ' << q;
  f << '\n';
  f.close();
  return load(path.c_str());
}

// Node with domains labs (4 = free, 5..7 = {I, label-4}, 0..3 fixed) and the given cap.
static void node(const Code& c, Dfs& st, int cap, const std::vector<int>& labs) {
  st.init(c, cap);
  for (int q = 0; q < c.n; ++q) if (labs[q] != 4) st.change(c, q, 4, labs[q]);
}

// Number of zero-syndrome completions of weight <= cap (complete enumeration over the domains).
static long completions(const std::vector<std::vector<int>>& G, int cap, const std::vector<int>& labs) {
  int n = labs.size(); long count = 0;
  std::vector<int> x(n, 0);
  std::function<void(int, int)> rec = [&](int q, int w) {
    if (w > cap) return;
    if (q == n) {
      for (const auto& g : G) {
        int odd = 0;
        for (int p = 0; p < n; ++p) odd ^= g[p] && x[p] && g[p] != x[p];
        if (odd) return;
      }
      ++count; return;
    }
    int a = labs[q];
    std::vector<int> options = a <= 3 ? std::vector<int>{a} : a == 4 ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{0, a - 4};
    for (int l : options) { x[q] = l; rec(q + 1, w + (l != 0)); }
  };
  rec(0, 0);
  return count;
}

static int failures = 0;
static void report(const char* name, std::ostringstream& row, bool ok) {
  std::cout << "{\"case\":\"" << name << "\"," << row.str() << ",\"ok\":" << (ok ? "true" : "false") << "}\n";
  failures += !ok;
}

// Soundness at one node: whatever a policy prunes must have no completion of weight <= cap.
static void rules_at(const char* name, const std::vector<std::vector<int>>& G, int cap,
                     const std::vector<int>& labs, int expect_completions, const std::vector<int>& expect_why,
                     long expect_s1_calls = 0) {
  HostCode h = make(name, G); long s1_calls = 0;
  long zero = completions(G, cap, labs);
  std::ostringstream row; row << "\"cap\":" << cap << ",\"completions\":" << zero << ",\"why\":[";
  bool ok = expect_completions < 0 ? true : (zero > 0) == (expect_completions > 0);
  for (int prune = 1; prune <= 3; ++prune) {
    h.prune = prune; Code c = h.view(); Dfs st; node(c, st, cap, labs);
    int why = st.dead ? -1 : st.csp_prune(c);
    row << (prune > 1 ? "," : "") << why;
    s1_calls += st.s1_calls;
    ok = ok && !(why > 0 && zero > 0) && why == expect_why[prune - 1];
  }
  row << "],\"s1_calls\":" << s1_calls;
  report(name, row, ok && s1_calls >= expect_s1_calls);
}

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: rule-cases DIRECTORY\n"; return 2; }
  directory = argv[1];
  // the S2 author (S2 lemma, tests/extra_rows.cpp): a representative X0 flips row 0 only; extras X1, X2
  // must jointly repair odd row 1 and even row 2. Blocking qubit 3 leaves no completion.
  for (int block = 0; block < 2; ++block) {
    const char* name = block ? "s2-extra-only-rows-blocked" : "s2-extra-only-rows-pair";
    std::vector<std::vector<int>> G = {{2, 2, 0, 0}, {2, 0, 2, 0}, {0, 0, 2, 2}};
    std::vector<int> labs = {1, 5, 5, block ? 0 : 5};
    HostCode h = make(name, G); h.prune = 3; Code c = h.view(); Dfs st; node(c, st, 4, labs);
    st.packed[0] = 0; int cut = s2_extra(c, st, 1); long zero = completions(G, 4, labs);
    std::ostringstream row; row << "\"s2_extra\":" << cut << ",\"completions\":" << zero;
    report(name, row, cut == block && (cut ? zero == 0 : zero > 0));
  }
  // the S2 author: one representative and two extras flip the same packed row; the other rows force
  // X1 = X2 = X3, so the triple case of the S2 lemma is essential.
  for (int block = 0; block < 2; ++block) {
    const char* name = block ? "s2-triple-blocked" : "s2-triple-anticommuting";
    std::vector<std::vector<int>> G = {{2, 2, 2, 2}, {0, 2, 2, 0}, {0, 0, 2, 2}};
    std::vector<int> labs = {1, 5, 5, block ? 0 : 5};
    HostCode h = make(name, G); h.prune = 3; Code c = h.view(); Dfs st; node(c, st, 4, labs);
    st.packed[0] = 0; int cut = s2_extra(c, st, 1); long zero = completions(G, 4, labs);
    std::ostringstream row; row << "\"s2_extra\":" << cut << ",\"completions\":" << zero;
    report(name, row, cut == block && (cut ? zero == 0 : zero > 0));
  }
  // the S1 reviewer (S1 review reproducer random_repro): the unfixed S1 owners cache pruned this node
  // although it has zero-syndrome completions (witness 1341). No policy may prune it.
  rules_at("s1-stale-cache-node", {{2, 1, 0, 1, 2, 2, 2}, {0, 2, 1, 0, 0, 1, 0}, {0, 1, 2, 3, 0, 1, 0}},
           5, {4, 4, 3, 7, 1, 1, 4}, 1, {0, 0, 0}, 1);
  // the S0 reviewer (S0 review reproducer gac_incomplete): tight packing r = 3 whose parity system is an
  // inconsistent XOR cycle. The node has no completion, but GAC cannot refute it: a documented
  // limit (S0 is incomplete). If this ever prunes, update docs/SOUNDNESS.md section 5.1.
  {
    std::vector<std::vector<int>> G;
    for (auto qs : std::vector<std::vector<int>>{{0, 1, 2}, {0, 3, 4}, {0, 5, 6}, {1, 3}, {3, 5}, {0, 1, 5}}) {
      std::vector<int> g(7); for (int q : qs) g[q] = 2; G.push_back(g);
    }
    std::vector<int> labs = {1, 4, 4, 4, 4, 4, 4};
    HostCode h = make("s0-gac-incomplete", G); Code c = h.view(); Dfs st; node(c, st, 4, labs);
    int packing = st.bound(c);
    std::ostringstream row; row << "\"greedy_packing\":" << packing;
    report("s0-gac-incomplete-packing", row, packing == 3);
    rules_at("s0-gac-incomplete", G, 4, labs, 0, {0, 0, 0});
  }
  // the S0 reviewer (S0 review reproducer mis_not_maximum): mis_pack is a threshold search. With target
  // r = 1 it stops at a size-3 packing although the maximum independent set has size 4.
  {
    std::vector<std::pair<int, int>> E = {{2, 0}, {2, 1}, {3, 0}, {4, 1}, {4, 2}, {4, 3}, {5, 0}, {5, 2},
                                          {6, 0}, {6, 2}, {7, 5}, {7, 6}, {8, 0}, {8, 1}, {8, 3}, {8, 4}};
    int n = 1 + E.size();
    std::vector<std::vector<int>> G(9, std::vector<int>(n));
    for (auto& g : G) g[0] = 2;
    for (int j = 0; j < (int)E.size(); ++j) { G[E[j].first][j + 1] = 2; G[E[j].second][j + 1] = 2; }
    std::vector<int> labs(n, 4); labs[0] = 1;
    HostCode h = make("s0-mis-threshold", G); h.prune = 1; Code c = h.view(); Dfs st; node(c, st, 2, labs);
    int result = st.mis_pack(c, 1, 0), alpha = 0;
    for (int s = 0; s < (1 << 9); ++s) {
      bool ok = true; for (auto e : E) if ((s >> e.first & 1) && (s >> e.second & 1)) ok = false;
      if (ok) alpha = std::max(alpha, __builtin_popcount((unsigned)s));
    }
    std::ostringstream row; row << "\"mis_pack\":" << result << ",\"maximum\":" << alpha;
    report("s0-mis-threshold", row, result == 3 && alpha == 4);
  }
  return failures != 0;
}
