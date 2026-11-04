// Test-only build of the native binary (never part of a release binary) that also dumps every prune node of the
// rules pass to $X1_DUMP: reason, cap, w, state fingerprint, the entry state lab[0..n) and the packing PK the prover
// holds (packed[0..h)). test_coverage_v2.py compares PK with cov2_ref.packing_select (BNB_CERT_SPEC_v2 6.1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
struct Code;
struct Dfs;
static void x1_dump(const Code&, const Dfs&, const char*);
#define STABCERT_CUT_HOOK(c, st, why) x1_dump(c, st, why)
#ifndef STABCERT_RULES
#define STABCERT_RULES 1
#endif
#include "../../src/hyqec_verify/native/bnb/bnb.cu"
static std::mutex x1_mu;
static FILE* x1_file() {
  static FILE* f = [] { const char* p = getenv("X1_DUMP"); return p ? fopen(p, "w") : (FILE*)nullptr; }();
  return f;
}
static void x1_dump(const Code& c, const Dfs& st, const char* why) {
  int r = st.cap - st.w, h;
  if (!c.prune) return;
  if (!strcmp(why, "packing")) h = r + 1;
  else if (!strcmp(why, "s0")) h = r;
  else if (!strcmp(why, "s1")) h = r - 1;
  else if (!strcmp(why, "s2")) h = r - 2;
  else return;
  FILE* f = x1_file(); if (!f) return;
  // X1_DUMP_EVERY=k: nodes with |A| <= 64 are sampled 1 in k (|A| > 64 always dumped); default all
  static long every = [] { const char* e = getenv("X1_DUMP_EVERY"); return e ? atol(e) : 1L; }();
  // X1_DUMP_MAX=k: stop after k dumped nodes (with --emit-cert the top tree is walked first)
  static long most = [] { const char* e = getenv("X1_DUMP_MAX"); return e ? atol(e) : -1L; }();
  static long seen = 0, dumped = 0;
  std::lock_guard<std::mutex> lock(x1_mu);
  if (st.nA <= 64 && (seen++ % every)) return;
  if (most >= 0 && dumped >= most) return;
  dumped++;
  fprintf(f, "%s %d %d %016llx ", why, st.cap, st.w, (unsigned long long)st.state_fp);
  for (int q = 0; q < c.n; q++) fputc('0' + st.lab[q], f);
  for (int i = 0; i < h; i++) fprintf(f, " %d", st.packed[i]);
  fputc('\n', f);
}
