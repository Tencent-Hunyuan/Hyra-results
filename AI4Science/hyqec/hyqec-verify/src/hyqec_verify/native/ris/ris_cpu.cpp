// CPU/OpenMP qubit-aligned RIS, with the same rank-1/rank-2 chain updates
// as ris.cu. This is an upper-bound search; Python verifies every witness.
#include "../common/platform.h"
#include "input.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <numeric>
#include <random>

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Options {
  double seconds = 10;
  long long max_steps = 0;
  u64 seed = 1;
  int threads = 1, steps = -1, target = -1, report_le = 0, ring_cap = 4096;
  bool chain = true, hamming = false;
};
struct Reports {
  const RisInput& d;
  const Options& o;
  std::atomic<int> best;
  std::atomic<int> ring_size{0};
  std::mutex mutex;
  std::vector<u64> hits;
  std::vector<std::pair<int, std::vector<u64>>> ring;
  Reports(const RisInput& input, const Options& options) : d(input), o(options),
      best((o.hamming ? 2*d.n : d.n)+1), hits(best.load(), 0) {}
  void check(const u64* a, const u64* b, std::vector<u64>& local_hits) {
    int w = 0;
    for (int i = 0; i < d.nq; i++) {
      u64 x = a[i] ^ (b ? b[i] : 0), z = a[d.nq+i] ^ (b ? b[d.nq+i] : 0);
      w += o.hamming ? __builtin_popcountll(x)+__builtin_popcountll(z) : __builtin_popcountll(x|z);
    }
    if (!w || w > std::max(best.load(std::memory_order_relaxed), o.report_le)) return;
    u64 syndrome = 0;
    for (int i = d.nw; i < d.stride; i++) syndrome |= a[i] ^ (b ? b[i] : 0);
    if (!syndrome) return;
    // Hits are far more frequent than improvements. Each search owns its
    // histogram; only a new witness/ring slot needs the shared report lock.
    local_hits[w]++;
    bool may_save = o.report_le && w <= o.report_le && ring_size.load(std::memory_order_relaxed) < o.ring_cap;
    if (w >= best.load(std::memory_order_relaxed) && !may_save) return;
    std::lock_guard<std::mutex> lock(mutex);
    bool improvement = w < best.load();
    bool save = o.report_le && w <= o.report_le && (int)ring.size() < o.ring_cap;
    if (!improvement && !save) return;
    std::vector<u64> witness(d.nw);
    for (int i = 0; i < d.nw; i++) witness[i] = a[i] ^ (b ? b[i] : 0);
    if (save) { ring.emplace_back(w, witness); ring_size.store(ring.size(), std::memory_order_relaxed); }
    if (improvement) {
      best.store(w);
      std::printf("{\"event\":\"witness\",\"w\":%d,", w);
      ris_print_vec(witness.data(), d.n, d.nq); std::printf("}\n"); std::fflush(stdout);
    }
  }
};

struct Search {
  const RisInput& d;
  Reports& reports;
  std::mt19937_64 rng;
  std::vector<u64> matrix, buffer;
  std::vector<u64> local_hits;
  std::vector<int> pivot, colrow, perm, free[2], full[2];
  std::vector<std::pair<int,int>> slots;
  u64 inits = 0, steps = 0, attempts = 0, failed = 0;
  Search(const RisInput& data, Reports& rep, u64 seed) : d(data), reports(rep), rng(seed),
      buffer(2*d.stride), local_hits(rep.hits.size(), 0), pivot(d.m), colrow(2*d.n), perm(d.n) {}
  u64* row(int r) { return matrix.data()+(size_t)r*d.stride; }
  int bit(const u64* a, int col) const {
    int q = col < d.n ? col : col-d.n;
    return (a[(col < d.n ? 0 : d.nq)+q/64] >> (q%64)) & 1;
  }
  void score(int a, int b) {
    reports.check(row(a), nullptr, local_hits);
    if (b >= 0) { reports.check(row(b), nullptr, local_hits); reports.check(row(a), row(b), local_hits); }
  }
  void init() {
    matrix = d.rows;
    std::fill(pivot.begin(), pivot.end(), -1);
    std::fill(colrow.begin(), colrow.end(), -1);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = d.n-1; i > 0; i--) std::swap(perm[i], perm[rng()%(i+1)]);
    uint32_t salt = (uint32_t)rng();
    for (int t = 0; t < 2*d.n; t++) {
      int q = perm[t/2];
      int first = (((uint32_t)q*2654435761u ^ salt) >> 7) & 1;
      int col = ((t&1)^first) ? d.n+q : q;
      int selected = -1;
      for (int r = 0; r < d.m; r++) if (pivot[r] < 0 && bit(row(r), col)) { selected = r; break; }
      if (selected < 0) continue;
      pivot[selected] = col; colrow[col] = selected;
      const u64* source = row(selected);
      for (int r = 0; r < d.m; r++) if (r != selected && bit(row(r), col)) {
        u64* target = row(r);
        for (int w = 0; w < d.stride; w++) target[w] ^= source[w];
      }
    }
    for (int c = 0; c < 2; c++) { free[c].clear(); full[c].clear(); }
    slots.clear();
    for (int q = 0; q < d.n; q++) {
      int cls = d.colnz[q]+d.colnz[d.n+q];
      if (!cls) continue;
      int x = colrow[q], z = colrow[d.n+q], count = (x >= 0)+(z >= 0);
      if (count == 0) free[cls-1].push_back(q);
      else {
        if (count == cls) full[cls-1].push_back(q);
        slots.emplace_back(x >= 0 ? x : z, x >= 0 && z >= 0 ? z : -1);
      }
    }
    inits++;
    for (auto slot : slots) score(slot.first, slot.second);
  }
  void step() {
    attempts++;
    int e0 = full[0].empty() ? 0 : free[0].size();
    int e1 = full[1].empty() ? 0 : free[1].size();
    if (!e0 && !e1) { failed++; return; }
    int cls = rng()%(e0+e1) < (u64)e1 ? 1 : 0;
    auto& f = free[cls]; auto& p = full[cls];
    int fi = rng()%f.size(), qin = f[fi];
    for (int tries = 0; tries < 16; tries++) {
      int pi = rng()%p.size(), qout = p[pi];
      int r1, r2 = -1, c1, c2 = -1;
      if (cls == 1) {
        r1 = colrow[qout]; r2 = colrow[d.n+qout]; c1 = qin; c2 = d.n+qin;
        const u64 *a = row(r1), *b = row(r2);
        int aa = bit(a,c1), ab = bit(a,c2), ba = bit(b,c1), bb = bit(b,c2);
        if (!((aa&bb)^(ab&ba))) continue;
        for (int i = 0; i < d.stride; i++) {
          buffer[i] = (bb ? a[i] : 0) ^ (ab ? b[i] : 0);
          buffer[d.stride+i] = (ba ? a[i] : 0) ^ (aa ? b[i] : 0);
        }
        colrow[qout] = colrow[d.n+qout] = -1;
        colrow[c1] = r1; colrow[c2] = r2;
      } else {
        int cout = colrow[qout] >= 0 ? qout : d.n+qout;
        c1 = d.colnz[qin] ? qin : d.n+qin; r1 = colrow[cout];
        if (!bit(row(r1),c1)) continue;
        std::copy(row(r1), row(r1)+d.stride, buffer.begin());
        colrow[cout] = -1; colrow[c1] = r1;
      }
      pivot[r1] = c1; if (r2 >= 0) pivot[r2] = c2;
      f[fi] = qout; p[pi] = qin;
      for (auto slot : slots) {
        bool changed = false;
        for (int r : {slot.first, slot.second}) {
          if (r < 0) continue;
          u64* target = row(r);
          if (r == r1 || r == r2) {
            const u64* source = buffer.data() + (r == r1 ? 0 : d.stride);
            std::copy(source, source+d.stride, target); changed = true;
          } else {
            int a = bit(target,c1), b = c2 >= 0 ? bit(target,c2) : 0;
            if (a || b) {
              for (int w = 0; w < d.stride; w++) target[w] ^= (a ? buffer[w] : 0) ^ (b ? buffer[d.stride+w] : 0);
              changed = true;
            }
          }
        }
        if (changed) score(slot.first,slot.second);
      }
      steps++; return;
    }
    failed++;
  }
};

static int main_impl(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    stabcert::version("stabcert-ris", "{\"NMAX\":16000}"); return 0;
  }
  Options o; o.threads = std::min(16, omp_get_max_threads());
  const char* path = nullptr;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next = [&]() { if (++i >= argc) throw std::invalid_argument("missing option value"); return std::string(argv[i]); };
    if (arg == "--seconds") o.seconds = ris_real(next());
    else if (arg == "--max-steps") o.max_steps = ris_integer(next());
    else if (arg == "--steps-per-round") o.steps = ris_int(next());
    else if (arg == "--seed") o.seed = (u64)ris_integer(next());
    else if (arg == "--threads") o.threads = ris_int(next());
    else if (arg == "--target") o.target = ris_int(next());
    else if (arg == "--report-le") o.report_le = ris_int(next());
    else if (arg == "--ring-cap") o.ring_cap = ris_int(next());
    else if (arg == "--no-chain") o.chain = false;
    else if (arg == "--hamming") o.hamming = true;
    else if (arg == "--quiet") {}
    else if (arg == "--device") { if (ris_int(next()) != 0) throw std::invalid_argument("CPU RIS has no CUDA device"); }
    else if (arg.empty() || arg[0] == '-' || path) throw std::invalid_argument("unknown argument: " + arg);
    else path = argv[i];
  }
  if (!path || o.threads < 1 || o.steps < -1 || o.max_steps < 0 || o.report_le < 0 || o.ring_cap < 1)
    throw std::invalid_argument("usage: stabcert-ris INPUT [--seconds S] [--threads T] [--max-steps N]");
#ifndef _OPENMP
  o.threads = 1;
#endif
  RisInput input = read_ris(path);
  if (o.steps < 0) o.steps = 20*input.m;
  Reports reports(input, o);
  std::vector<std::array<u64,4>> stats(o.threads);
  double start = now_s();
  std::atomic<long long> issued(0);
  std::printf("{\"event\":\"start\",\"n\":%d,\"m\":%d,\"threads\":%d,\"seed\":%llu,\"no_chain\":%s}\n",
      input.n, input.m, o.threads, o.seed, stabcert::json_bool(!o.chain));
  std::fflush(stdout);
  #pragma omp parallel num_threads(o.threads)
  {
    int tid = omp_get_thread_num();
    Search search(input, reports, o.seed + (u64)tid);
    auto stop = [&]() { return now_s()-start >= o.seconds || (o.target >= 0 && reports.best.load() < o.target); };
    auto reserve = [&]() { return !o.max_steps || issued.fetch_add(1) < o.max_steps; };
    while (!stop()) {
      if ((!o.chain || !o.steps) && !reserve()) break;
      search.init();
      if (o.chain && o.steps) {
        bool exhausted = false;
        for (int i = 0; i < o.steps; i++) {
          if (stop() || !reserve()) { exhausted = true; break; }
          search.step();
        }
        if (exhausted) break;
      }
    }
    stats[tid] = {search.steps, search.inits, search.attempts, search.failed};
    std::lock_guard<std::mutex> lock(reports.mutex);
    for (size_t w = 0; w < search.local_hits.size(); w++) reports.hits[w] += search.local_hits[w];
  }
  for (const auto& hit : reports.ring) {
    std::printf("{\"event\":\"ring\",\"w\":%d,",hit.first);
    ris_print_vec(hit.second.data(),input.n,input.nq); std::printf("}\n");
  }
  u64 steps=0,inits=0,attempts=0,failed=0;
  for (auto s : stats) { steps+=s[0]; inits+=s[1]; attempts+=s[2]; failed+=s[3]; }
  int best = reports.best.load();
  std::printf("{\"event\":\"done\",\"interface\":1,\"seconds\":%.6f,\"launches\":%llu,\"steps\":%llu,\"inits\":%llu,\"attempted_steps\":%llu,\"failed_swaps\":%llu,\"best\":%d,\"hits_by_weight\":{",
      now_s()-start,inits,steps,inits,attempts,failed,best <= (o.hamming ? 2*input.n : input.n) ? best : -1);
  bool first = true;
  for (size_t w=1; w<reports.hits.size(); w++) if (reports.hits[w]) { std::printf("%s\"%zu\":%llu",first ? "" : ",",w,reports.hits[w]); first=false; }
  std::printf("}}\n"); return 0;
}
int main(int argc,char** argv) {
  try { return main_impl(argc,argv); }
  catch (const std::invalid_argument& e) { std::fprintf(stderr,"input error: %s\n",e.what()); return 2; }
  catch (const std::out_of_range& e) { std::fprintf(stderr,"input error: %s\n",e.what()); return 2; }
  catch (const std::exception& e) { std::fprintf(stderr,"error: %s\n",e.what()); return 1; }
}
