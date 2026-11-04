// review finding 01 (review of d2552e0): at the 4096-w128 limits an S1 qubit-filter pass touched more qubits than its
// 2048-entry cleanup list held; the dirty entries made the next csp_prune call prune (S1) a node that has the
// weight-3 completion X0 X2049 X2050. Reproducer by the rules reviewer (wide_s1_probe.cpp), kept as a permanent regression:
// every call must leave the cross-call scratch clean, and the second node must agree with a freshly initialised
// state (no prune). Build: -DNMAX=4096 -DMMAX=4096 -DWMAX=128 -DCMAX=128 (rules build); test-only.
#define STABCERT_SOURCE_SHA256 "test-only-wide-s1-regression"
#define STABCERT_RULES 1
#define main unused_native_main
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
#undef main
#include <memory>

static int dirty(const Code& c, const Dfs& st) {
  int count = 0;
  for (int q = 0; q < c.n; q++) count += st.qcnt[q] || st.qlast[q] || st.qmk[q] || st.qown[q];
  for (int g = 0; g < c.m; g++) count += st.gmap[g] != -1 || st.kmask[g];
  return count;
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) { fprintf(stderr, "usage: wide-s1-regression INSTANCE_OUT [BLOCKS WIDTH]\n"); return 2; }
  // `blocks` disjoint blocks of `width` qubits and anchor qubit 0 (default 17 x 128: qubits 1..2176); Z-only
  // generators: two identical packed rows Z0 Z(first qubit of each block), and one all-Z row per block. the rules reviewer's
  // boundary sweep: 16 x 128 and 32 x 64 (2048 touched qubits) were clean before the fix; 33 x 64, 17 x 128 and
  // 31 x 128 (2112, 2176, 3968) were dirty and false-pruned.
  const int blocks = argc == 4 ? std::stoi(argv[2]) : 17, width = argc == 4 ? std::stoi(argv[3]) : 128;
  const int n = 1 + blocks * width;
  std::ofstream f(argv[1]);
  f << n << ' ' << blocks + 2 << " 0\n";
  for (int repeat = 0; repeat < 2; repeat++) {
    f << blocks + 1 << " 0 2";
    for (int i = 0; i < blocks; i++) f << ' ' << 1 + i * width << " 2";
    f << '\n';
  }
  for (int i = 0; i < blocks; i++) {
    f << width;
    for (int j = 0; j < width; j++) f << ' ' << 1 + i * width + j << " 2";
    f << '\n';
  }
  for (int q = 0; q < n; q++) f << q << ' ';
  f << '\n' << n;
  for (int q = 0; q < n; q++) f << ' ' << q;
  f << '\n'; f.close();
  HostCode h = load(argv[1]); h.prune = 2; Code c = h.view();
  std::unique_ptr<Dfs> st(new Dfs()); st->init(c, 3);
  for (int q = 0; q < n; q++) st->change(c, q, st->lab[q], q == 0 ? 1 : 5);   // X0 fixed, others {I, X}
  int first = st->csp_prune(c), stale = dirty(c, *st);
  for (int i = 0; i < blocks - 1; i++) { int q = 1 + i * width; st->change(c, q, st->lab[q], 0); }
  const int qx = 1 + (blocks - 1) * width, qe = qx + 1;
  bool valid = true;
  for (int g = 0; g < c.m; g++) {
    int parity = 0;
    for (int p = 0; p < c.gw[g]; p++) { int q = c.gq[g * c.W + p]; parity ^= (q == 0 || q == qx || q == qe); }
    valid = valid && !parity;
  }
  int second = st->csp_prune(c), stale2 = dirty(c, *st);
  std::unique_ptr<Dfs> clean(new Dfs()); clean->init(c, 3);
  for (int q = 0; q < n; q++) clean->change(c, q, clean->lab[q], st->lab[q]);
  int fresh = clean->csp_prune(c);
  printf("{\"n\":%d,\"W\":%d,\"first_cut\":%d,\"stale_after_first\":%d,\"second_cut\":%d,\"stale_after_second\":%d,"
         "\"fresh_cut\":%d,\"valid_X_witness\":[0,%d,%d],\"commutes\":%s}\n",
         n, c.W, first, stale, second, stale2, fresh, qx, qe, valid ? "true" : "false");
  return first == 0 && stale == 0 && second == 0 && stale2 == 0 && fresh == 0 && valid ? 0 : 1;
}
