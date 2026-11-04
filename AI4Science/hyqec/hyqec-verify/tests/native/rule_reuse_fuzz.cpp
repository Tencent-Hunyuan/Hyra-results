// Large-n stratum for the S0/S1/S2 rules (review finding 01): state that persists between rule calls must never
// change a decision. For codes beyond the cut audit's enumeration oracle (n > 2048 at the 4096-w128 limits),
// a long random sequence of nodes is evaluated twice: on ONE reused search state (as the real search does)
// and on a freshly initialised state with the same domains. The two csp_prune results must agree, and the
// cross-call scratch must be clean after every call. Test-only; build -DNMAX=4096 -DMMAX=4096 -DWMAX=128 -DCMAX=128.
#define STABCERT_SOURCE_SHA256 "test-only-rule-reuse-fuzz"
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
  return count;
}

// the rules reviewer's family: anchor qubit 0, `blocks` disjoint blocks of `width` qubits, `dup` identical packed rows
// Z0 Z(first qubit of each block), one all-Z row per block, plus `extra` random Z rows of weight <= width.
static std::string family(const std::string& dir, int blocks, int width, int dup, int extra, std::mt19937_64& rng) {
  int n = 1 + blocks * width;
  std::vector<std::vector<int>> rows;
  for (int r = 0; r < dup; r++) {
    std::vector<int> g{0};
    for (int i = 0; i < blocks; i++) g.push_back(1 + i * width);
    rows.push_back(g);
  }
  for (int i = 0; i < blocks; i++) {
    std::vector<int> g;
    for (int j = 0; j < width; j++) g.push_back(1 + i * width + j);
    rows.push_back(g);
  }
  for (int e = 0; e < extra; e++) {
    int w = 2 + rng() % (width - 1);
    std::vector<int> g;
    while ((int)g.size() < w) { int q = rng() % n; if (std::find(g.begin(), g.end(), q) == g.end()) g.push_back(q); }
    std::sort(g.begin(), g.end()); rows.push_back(g);
  }
  std::string path = dir + "/family-" + std::to_string(blocks) + "x" + std::to_string(width) + "-d" +
                     std::to_string(dup) + "-e" + std::to_string(extra) + ".inst";
  std::ofstream f(path);
  f << n << ' ' << rows.size() << " 0\n";
  for (auto& g : rows) { f << g.size(); for (int q : g) f << ' ' << q << " 2"; f << '\n'; }
  for (int q = 0; q < n; q++) f << q << ' ';
  f << '\n' << n; for (int q = 0; q < n; q++) f << ' ' << q;
  f << '\n';
  return path;
}

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: rule-reuse-fuzz DIR SEED NODES_PER_CODE\n"); return 2; }
  std::string dir = argv[1]; std::mt19937_64 rng(std::stoull(argv[2])); int per = std::stoi(argv[3]);
  struct Shape { int blocks, width, dup, extra; };
  // n > 2048 (the overflow regime; S2's table guard disables S2 above n = 1365) and n <= 1365 (S2 active).
  std::vector<Shape> shapes = {{17, 128, 2, 0}, {17, 128, 2, 40}, {18, 120, 3, 20}, {33, 64, 2, 30}, {40, 60, 1, 60},
                               {10, 128, 2, 0}, {10, 128, 2, 30}, {9, 120, 3, 15}, {20, 64, 2, 25}};
  long calls = 0, mismatches = 0, dirty_calls = 0, results[5] = {0, 0, 0, 0, 0};
  for (const Shape& s : shapes) {
    HostCode h = load(family(dir, s.blocks, s.width, s.dup, s.extra, rng).c_str());
    for (int prune = 1; prune <= 3; prune++) {
      h.prune = prune; Code c = h.view();
      std::unique_ptr<Dfs> reused(new Dfs());
      int cap = 0;
      for (int it = 0; it < per; it++) {
        // Random node: qubit 0 decided X; up to cap-2 further decided X; every block's first qubit I or {I,X};
        // every other qubit {I,X} or (rarely) I. Same cap within a run of nodes, as in a real task.
        if (it % 50 == 0) {
          cap = 2 + rng() % 3; reused->init(c, cap);
          // start the owners/E stamps just below their wrap, so the wrap-and-reset paths run in this test
          reused->curstamp = reused->estamp = 0xffffffffu - (uint32_t)(rng() % 64);
        }
        std::vector<int> lab(c.n, 5);
        lab[0] = 1;
        // Removal rates 0, 1/16, 1/3 so that some nodes keep every block alive: the S1 filter's union of
        // killer supports then exceeds 2048 qubits (the finding 01 overflow regime) at widths 120-128.
        static const int rates[3] = {0, 16, 3};
        int rate = rates[rng() % 3];
        for (int i = 0; i < s.blocks; i++) if (rate && rng() % rate == 0) lab[1 + i * s.width] = 0;
        for (int q = 1; q < c.n; q++) if (rng() % 256 == 0) lab[q] = 0;
        int extra_w = rng() % 2 ? 0 : rng() % (cap - 1);
        for (int k = 0; k < extra_w; k++) lab[1 + rng() % (c.n - 1)] = 1;
        for (int q = 0; q < c.n; q++) if (reused->lab[q] != lab[q]) reused->change(c, q, reused->lab[q], lab[q]);
        if (reused->dead || reused->w + 1 > reused->cap) continue;
        int a = reused->csp_prune(c);
        int d = dirty(c, *reused);
        std::unique_ptr<Dfs> fresh(new Dfs()); fresh->init(c, cap);
        for (int q = 0; q < c.n; q++) if (fresh->lab[q] != lab[q]) fresh->change(c, q, fresh->lab[q], lab[q]);
        int b = fresh->csp_prune(c);
        calls++; results[a]++; dirty_calls += d > 0; mismatches += a != b;
        if (a != b || d) {
          fprintf(stderr, "MISMATCH shape %dx%d prune=%d it=%d cap=%d reused=%d fresh=%d dirty=%d\n",
                  s.blocks, s.width, prune, it, cap, a, b, d);
          if (mismatches + dirty_calls > 5) break;
        }
      }
    }
  }
  printf("{\"calls\":%ld,\"mismatches\":%ld,\"dirty_calls\":%ld,\"results\":{\"none\":%ld,\"packing\":%ld,\"s0\":%ld,\"s1\":%ld,\"s2\":%ld}}\n",
         calls, mismatches, dirty_calls, results[0], results[1], results[2], results[3], results[4]);
  return mismatches || dirty_calls ? 1 : 0;
}
