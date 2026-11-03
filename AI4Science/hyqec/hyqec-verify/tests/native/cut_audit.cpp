// Instrument the actual CPU DFS, but derive the exact oracle independently.
// This translation unit is test-only and never part of a release binary.
struct Code;
struct Dfs;
static void audit_cut(const Code&, const Dfs&, const char*);
static void audit_scratch(const Code&, const Dfs&);
#define STABCERT_CUT_HOOK(c, st, why) audit_cut(c, st, why)
// After every S0/S1/S2 rule pass: all scratch that must be clean between calls is clean (review finding 01).
#define STABCERT_RULE_SCRATCH_HOOK(c, st) audit_scratch(c, st)
#define STABCERT_SOURCE_SHA256 "test-only-cut-audit"
// Count every capacity limit the rule code hits (s012.inc / s2.inc STABCERT_RULE_LIMIT; absent in production).
#include <map>
#include <string>
static std::map<std::string, long> limit_hits;
#define STABCERT_RULE_LIMIT(name) (++limit_hits[name])
#ifndef STABCERT_RULES
#define STABCERT_RULES 1   // the rules build: every policy, including the opt-in S0/S1/S2
#endif
#define main unused_native_main
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
#undef main
#include <array>
#include <map>
#include <iostream>

struct Basis {
  std::array<uint64_t, 64> row{};
  int rank = 0;
  uint64_t reduce(uint64_t x) const {
    while (x) {
      uint64_t pivot = row[63 - __builtin_clzll(x)];
      if (!pivot) break;
      x ^= pivot;
    }
    return x;
  }
  void add(uint64_t x) {
    x = reduce(x);
    if (x) { row[63 - __builtin_clzll(x)] = x; ++rank; }
  }
};

struct Exact {
  int n, distance;
  uint64_t mask;
  std::vector<std::vector<uint64_t>> logical;
  std::vector<uint64_t> ordered;
  std::vector<size_t> weight_end;
  std::vector<std::array<std::vector<uint64_t>, 4>> labels;
  explicit Exact(const Code& c) : n(c.n), distance(c.n + 1),
      mask((UINT64_C(1) << n) - 1), logical(n + 1) {
    if (n > 18) throw std::runtime_error("oracle is limited to n <= 18");
    Basis stabilizer, checks;
    for (int g = 0; g < c.m; ++g) {
      uint64_t v = 0;
      for (int p = 0; p < c.gw[g]; ++p) {
        int q = c.gq[g*c.W+p], l = c.gl[g*c.W+p];
        v |= uint64_t(l & 1) << q; v |= uint64_t(l >> 1) << (n+q);
      }
      stabilizer.add(v);
      checks.add((v >> n) | ((v & mask) << n));
    }
    std::vector<uint64_t> kernel;
    for (int free = 0; free < 2*n; ++free) if (!checks.row[free]) {
      uint64_t v = UINT64_C(1) << free;
      // Rows have their highest bit at the pivot, so solve upward.
      for (int p = 0; p < 2*n; ++p)
        if (checks.row[p] && (__builtin_popcountll(checks.row[p] & v) & 1))
          v |= UINT64_C(1) << p;
      kernel.push_back(v);
    }
    for (auto g : stabilizer.row) if (g)
      for (auto check : checks.row) if (__builtin_popcountll(g & check) & 1)
        throw std::runtime_error("noncommuting oracle input");
    Basis normalizer = stabilizer;
    for (int l = 0; l < c.nl; ++l) {
      uint64_t v = 0;
      for (int q = 0; q < n; ++q) {
        int a = c.L[l*n+q];
        v |= uint64_t(a & 1) << q; v |= uint64_t(a >> 1) << (n+q);
      }
      for (auto check : checks.row) if (__builtin_popcountll(v & check) & 1)
        throw std::runtime_error("invalid supplied logical basis");
      normalizer.add(v);
    }
    if (c.nl != 2*(n-stabilizer.rank) || normalizer.rank != 2*n-stabilizer.rank)
      throw std::runtime_error("incomplete supplied logical basis");
    uint64_t v = 0;
    for (uint64_t i = 1; i < (UINT64_C(1) << kernel.size()); ++i) {
      v ^= kernel[__builtin_ctzll(i)];
      if ((c.domain == 1 && (v >> n)) || (c.domain == 2 && (v & mask))) continue;
      if (!stabilizer.reduce(v)) continue;
      int w = __builtin_popcountll((v | (v >> n)) & mask);
      logical[w].push_back(v); distance = std::min(distance, w);
    }
    // Independently scan all Paulis at the small end; this cross-checks the
    // nullspace enumeration rather than merely comparing two DFS routines.
    if (n <= 7) {
      std::vector<size_t> counts(n+1);
      for (uint64_t p = 1; p < (UINT64_C(1) << (2*n)); ++p) {
        if ((c.domain == 1 && (p >> n)) || (c.domain == 2 && (p & mask))) continue;
        bool commutes = true;
        for (auto check : checks.row) if (__builtin_popcountll(p & check) & 1) { commutes = false; break; }
        if (commutes && stabilizer.reduce(p)) ++counts[__builtin_popcountll((p | (p >> n)) & mask)];
      }
      for (int w = 1; w <= n; ++w) if (counts[w] != logical[w].size())
        throw std::runtime_error("nullspace oracle disagrees with all-Pauli scan");
    }
    weight_end.resize(n+1);
    for (int w = 1; w <= n; ++w) {
      ordered.insert(ordered.end(), logical[w].begin(), logical[w].end());
      weight_end[w] = ordered.size();
    }
    labels.resize(n);
    for (auto& q : labels) for (auto& l : q) l.resize((ordered.size()+63)/64);
    for (size_t i = 0; i < ordered.size(); ++i) for (int q = 0; q < n; ++q) {
      int l = ((ordered[i] >> q) & 1) | (((ordered[i] >> (n+q)) & 1) << 1);
      labels[q][l][i/64] |= UINT64_C(1) << (i%64);
    }
  }
  uint64_t compatible(const Dfs& st) const {
    // Exact intersection over the exhaustively enumerated logicals. Bitmaps
    // make repeated domain queries cheap without sampling or dropping a cut.
    size_t count = weight_end[std::max(0, std::min(n, st.cap))];
    for (size_t word = 0; word < (count+63)/64; ++word) {
      uint64_t candidates = ~UINT64_C(0);
      if (word == count/64 && count%64) candidates >>= 64-count%64;
      for (int q = 0; q < n && candidates; ++q) {
        int a = st.lab[q];
        if (a < 4) candidates &= labels[q][a][word];
        else if (a > 4) candidates &= labels[q][0][word] | labels[q][a-4][word];
      }
      if (candidates) return ordered[64*word+__builtin_ctzll(candidates)];
    }
    return 0;
  }
};

static const Exact* oracle = nullptr;
static bool checking = false;
// --large: codes beyond the enumeration oracle (n > 2048 at the 4096-w128 limits). Every cut is still re-searched
// with packing disabled from rebuilt state, and whole distances are compared with the reviewed greedy search.
static bool large_mode = false;
static std::string input_name, policy_name;
static std::map<std::string, uint64_t> cuts;
static uint64_t baseline_nodes = 0;

static void audit_cut(const Code& c, const Dfs& state, const char* reason) {
  if (checking || (!oracle && !large_mode)) return;
  ++cuts[policy_name + "/" + reason];
  uint64_t witness = oracle ? oracle->compatible(state) : 0;
  if (witness) {
    std::ostringstream msg;
    msg << "unsafe " << reason << " policy=" << policy_name << " input=" << input_name
        << " cap=" << state.cap << " witness=" << witness << " domains=";
    for (int q = 0; q < c.n; ++q) msg << int(state.lab[q]);
    throw std::runtime_error(msg.str());
  }
  // Rebuild every cache from the same domains; copying the old caches would
  // conceal precisely the stale-state failure this differential test targets.
  Code baseline = c; baseline.greedy = baseline.coverage_bound = false; baseline.prune = 0;
  Dfs fresh; fresh.init(baseline, state.cap);
  for (int q = 0; q < c.n; ++q)
    if (fresh.lab[q] != state.lab[q]) fresh.change(baseline, q, fresh.lab[q], state.lab[q]);
  checking = true;
  int found = fresh.dead ? (1 << 30) : fresh.run(baseline, [&] { return state.cap; });
  checking = false;
  baseline_nodes += fresh.nodes;
  if (found <= state.cap)
    throw std::runtime_error("packing-none subtree contradicts cut: " + input_name);
}

static uint64_t scratch_checks = 0;
static void audit_scratch(const Code& c, const Dfs& st) {
  ++scratch_checks;
  for (int q = 0; q < c.n; ++q)
    if (st.qcnt[q] || st.qlast[q] || st.qmk[q] || st.qown[q])
      throw std::runtime_error("dirty rule scratch at qubit " + std::to_string(q) + " policy=" + policy_name + " input=" + input_name);
  for (int g = 0; g < c.m; ++g)
    if (st.gmap[g] != -1 || st.kmask[g])
      throw std::runtime_error("dirty rule scratch at generator " + std::to_string(g) + " policy=" + policy_name + " input=" + input_name);
  for (int i = 0; i < (c.n + 31) / 32; ++i)
    if (st.used[i]) throw std::runtime_error("dirty packing scratch policy=" + policy_name + " input=" + input_name);
}

static int search(Code c, const HostCode& h, int cap) {
  int best = c.n + 1;
  if (!cap) return best;
  for (int oi = 0; oi < (int)h.reps.size(); ++oi) for (int l = 1; l <= 3; ++l) {
    if (c.domain && c.domain != l) continue;
    Dfs st; st.init(c, std::min(cap, best-1));
    if (st.root(c, h.orbit.data(), h.reps[oi], oi, l))
      best = std::min(best, st.run(c, [&] { return std::min(cap, best-1); }));
  }
  return best;
}

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3) throw std::runtime_error("usage: cut-audit MANIFEST [--inject-unsafe-cut | --large]");
    large_mode = argc == 3 && std::string(argv[2]) == "--large";
    std::ifstream manifest(argv[1]);
    int random_cap, cases = 0;
    while (manifest >> input_name >> random_cap) {
      HostCode h = load(input_name.c_str()); Code c = h.view();
      if (large_mode) {
        // Reference: the reviewed greedy search (no rule code) at the manifest cap; every policy must agree.
        policy_name = "reference"; c.greedy = true; c.coverage_bound = false; c.prune = 0;
        checking = true; int expected = search(c, h, random_cap); checking = false;
        for (const char* policy : {"greedy", "greedy+s0", "greedy+s0s1", "greedy+s0s1s2"}) {
          policy_name = policy; c.prune = policy_name == "greedy+s0" ? 1 : policy_name == "greedy+s0s1" ? 2 : policy_name == "greedy+s0s1s2" ? 3 : 0;
          if (search(c, h, random_cap) != expected)
            throw std::runtime_error("distance mismatch: " + input_name + " policy=" + policy_name);
        }
        ++cases; std::cerr << "large case " << input_name << " n=" << c.n << " cap=" << random_cap << " d<=cap: "
                           << (expected <= random_cap ? std::to_string(expected) : std::string("none")) << '\n';
        continue;
      }
      Exact exact(c); oracle = &exact;
      if (argc == 3) {
        if (std::string(argv[2]) != "--inject-unsafe-cut") throw std::runtime_error("unknown test option");
        // Ensure the hook rejects a deliberately discarded real logical.
        Dfs st; st.init(c, c.n);
        uint64_t v = exact.logical.at(exact.distance).at(0);
        for (int q = 0; q < c.n; ++q)
          st.change(c, q, st.lab[q], ((v >> q) & 1) | (((v >> (c.n+q)) & 1) << 1));
        audit_cut(c, st, "self-test");
        throw std::runtime_error("injected unsafe cut was not detected");
      }
      for (const char* policy : {"none", "greedy", "greedy+b1b3", "greedy+s0", "greedy+s0s1", "greedy+s0s1s2"}) {
        policy_name = policy; c.greedy = policy_name != "none";
        c.coverage_bound = policy_name == "greedy+b1b3";
        c.prune = policy_name == "greedy+s0" ? 1 : policy_name == "greedy+s0s1" ? 2 : policy_name == "greedy+s0s1s2" ? 3 : 0;
        for (int cap : {random_cap, c.n}) {
          int actual = search(c, h, cap);
          int expected = exact.distance <= cap ? exact.distance : c.n+1;
          if (actual != expected)
            throw std::runtime_error("distance mismatch: " + input_name + " policy=" + policy_name);
        }
      }
      oracle = nullptr; ++cases;
      if (cases % 100 == 0) {
        uint64_t count = 0; for (const auto& entry : cuts) count += entry.second;
        std::cerr << "cases=" << cases << " cuts=" << count << " baseline_nodes=" << baseline_nodes << '\n';
      }
    }
    if (!manifest.eof()) throw std::runtime_error("malformed audit manifest");
    std::cout << "{\"cases\":" << cases << ",\"baseline_nodes\":" << baseline_nodes
              << ",\"scratch_checks\":" << scratch_checks << ",\"limits\":{";
    bool lcomma = false;
    for (const auto& entry : limit_hits) {
      if (lcomma) std::cout << ',';
      lcomma = true; std::cout << '"' << entry.first << "\":" << entry.second;
    }
    std::cout << "},\"cuts\":{";
    bool comma = false;
    for (const auto& entry : cuts) {
      if (comma) std::cout << ',';
      comma = true; std::cout << '"' << entry.first << "\":" << entry.second;
    }
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
