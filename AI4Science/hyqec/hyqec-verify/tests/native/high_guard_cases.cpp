// Deterministic boundary cases for the fixed-constant and list guards of the S0/S1/S2 rules, by the independent rules reviewer
// (a review fixture), ported to the STABCERT_RULE_LIMIT counters. Each node
// has an explicit zero-syndrome X-type witness of weight <= cap (all rows are Z-only), so the rules must not prune;
// three repeated calls on one state must leave the scratch clean; the named guard must (or must not) fire.
// Build: -DNMAX=4096 -DMMAX=4096 -DWMAX=128 -DCMAX=128. Test-only.
#include <map>
#include <string>
static std::map<std::string, long> limit_hits;
#define STABCERT_RULE_LIMIT(name) (++limit_hits[name])
#define STABCERT_SOURCE_SHA256 "test-only-high-guard-cases"
#define STABCERT_RULES 1
#define main unused_native_main
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
#undef main
#include <memory>

static std::string dir;
static int failures = 0;

static HostCode make(const std::string& name, int n, const std::vector<std::vector<int>>& rows) {
  std::string path = dir + "/" + name + ".inst";
  std::ofstream f(path); f << n << ' ' << rows.size() << " 0\n";
  for (auto& row : rows) { f << row.size(); for (int q : row) f << ' ' << q << " 2"; f << '\n'; }
  for (int q = 0; q < n; q++) f << q << ' ';
  f << '\n' << n; for (int q = 0; q < n; q++) f << ' ' << q;
  f << '\n'; f.close();
  return load(path.c_str());
}

static void check(const char* name, HostCode h, int cap, int prune, const std::vector<int>& witness,
                  const char* guard, bool expect_hit, const std::vector<int>& decided = {0}) {
  h.prune = prune; Code c = h.view(); std::unique_ptr<Dfs> st(new Dfs()); st->init(c, cap);
  for (int q = 0; q < c.n; q++)
    st->change(c, q, st->lab[q], std::find(decided.begin(), decided.end(), q) != decided.end() ? 1 : 5);
  bool ok = !st->dead && st->w + 1 <= cap && (int)witness.size() <= cap;
  for (int g = 0; g < c.m; g++) {
    int parity = 0;
    for (int p = 0; p < c.gw[g]; p++) parity ^= std::find(witness.begin(), witness.end(), c.gq[g * c.W + p]) != witness.end();
    ok = ok && !parity;
  }
  long before = limit_hits[guard];
  int cuts = 0, dirty = 0;
  for (int repeat = 0; repeat < 3; repeat++) {
    cuts += st->csp_prune(c) != 0;
    for (int q = 0; q < c.n; q++) dirty += st->qcnt[q] || st->qlast[q] || st->qmk[q] || st->qown[q];
    for (int g = 0; g < c.m; g++) dirty += st->gmap[g] != -1 || st->kmask[g];
  }
  bool hit = limit_hits[guard] > before;
  ok = ok && !cuts && !dirty && hit == expect_hit;
  printf("{\"case\":\"%s\",\"n\":%d,\"m\":%d,\"cap\":%d,\"cuts\":%d,\"dirty\":%d,\"guard\":\"%s\",\"hit\":%s,\"ok\":%s}\n",
         name, c.n, c.m, cap, cuts, dirty, guard, hit ? "true" : "false", ok ? "true" : "false");
  failures += !ok;
}

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: high-guard-cases DIR\n"); return 2; }
  dir = argv[1];
  std::vector<std::vector<int>> rows;
  // h = 32 packed rows at cap 34: S1's 31-variable bitmask declines (no prune); witness X0..X33
  for (int q = 1; q <= 32; q++) rows.push_back({0, q});
  rows.push_back({0, 1});
  for (int q = 1; q <= 32; q++) rows.push_back({q, 33});
  std::vector<int> all; for (int q = 0; q < 34; q++) all.push_back(q);
  check("h32-bitmask", make("h32", 34, rows), 34, 2, all, "s1_bitmask_h_gt_31", true);
  // |A_v| = 97 duplicate rows: mis_pack declines (> 64) and the greedy list is truncated (> AMAX = 96)
  rows.assign(97, {0, 1});
  check("nA97-mis", make("nA97", 2, rows), 3, 2, {0, 1}, "mis_nA_gt_64", true);
  check("nA97-amax", make("nA97b", 2, rows), 3, 2, {0, 1}, "greedy_amax", true);
  // S2 table guard boundary: active at n = 1365 (3n = 4095 <= 4096), declines at n = 1366
  rows.assign(3, {0, 1});
  check("S2-n1365", make("s2-1365", 1365, rows), 4, 3, {0, 1}, "s2_table", false);
  check("S2-n1366", make("s2-1366", 1366, rows), 4, 3, {0, 1}, "s2_table", true);
  // need list beyond AMAX entries
  rows.assign(120, {0, 1});
  check("need-list-full", make("need120", 2, rows), 3, 2, {0, 1}, "need_amax", true);
  // zero (zl) list beyond AMAX entries
  rows.assign(96, {0, 2}); for (int i = 0; i < 97; i++) rows.push_back({1, 3});
  check("zero-list-full", make("zero97", 4, rows), 4, 2, {0, 1, 2, 3}, "zl_amax", true, {0, 1});
  return failures ? 1 : 0;
}
