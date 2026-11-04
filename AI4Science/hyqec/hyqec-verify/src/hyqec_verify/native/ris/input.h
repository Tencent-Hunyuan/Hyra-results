#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

using u64 = unsigned long long;
struct RisInput {
  int n, nq, m, nl, nw, nsw, stride;
  std::vector<uint8_t> colnz;
  std::vector<u64> rows;
};

inline RisInput read_ris(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::invalid_argument("cannot open RIS input");
  auto bytes = input.tellg(); input.seekg(0);
  char magic[8]; input.read(magic, 8);
  if (!input || std::string(magic, 8) != "PAULIRS2") throw std::invalid_argument("expected PAULIRS2 header");
  auto word = [&](int length) {
    u64 value = 0;
    for (int b = 0; b < length; b++) {
      int ch = input.get();
      if (ch < 0) throw std::invalid_argument("truncated RIS input");
      value |= (u64)(uint8_t)ch << (8*b);
    }
    return value;
  };
  RisInput d;
  d.n = (int32_t)word(4); d.nq = (int32_t)word(4); d.m = (int32_t)word(4); d.nl = (int32_t)word(4);
  if (d.n <= 0 || d.n > 16000 || d.nq != (d.n+63)/64 || d.m <= 0 || d.m > 2*d.n || d.nl <= 0 || d.nl > 2*d.n)
    throw std::invalid_argument("RIS limits: require 1 <= n <= 16000, nq=ceil(n/64), 1 <= m,nl <= 2n");
  d.nw = 2*d.nq; d.nsw = (d.nl+63)/64; d.stride = d.nw+d.nsw;
  size_t expected = 24 + (size_t)2*d.n + (size_t)8*d.m*d.stride;
  if (bytes < 0 || (size_t)bytes != expected) throw std::invalid_argument("RIS input length mismatch (truncated or trailing data)");
  d.colnz.resize(2*d.n); input.read((char*)d.colnz.data(), d.colnz.size());
  for (auto c : d.colnz) if (c > 1) throw std::invalid_argument("column flags must be zero or one");
  d.rows.resize((size_t)d.m*d.stride);
  for (auto& w : d.rows) w = word(8);
  std::vector<uint8_t> actual(2*d.n, 0);
  for (int r = 0; r < d.m; r++) {
    const u64* row = d.rows.data() + (size_t)r*d.stride;
    if (d.n%64 && ((row[d.nq-1] | row[d.nw-1]) >> (d.n%64))) throw std::invalid_argument("nonzero Pauli padding bits");
    if (d.nl%64 && (row[d.stride-1] >> (d.nl%64))) throw std::invalid_argument("nonzero logical-syndrome padding bits");
    for (int q = 0; q < d.n; q++) {
      actual[q] |= (row[q/64] >> (q%64)) & 1;
      actual[d.n+q] |= (row[d.nq+q/64] >> (q%64)) & 1;
    }
  }
  if (actual != d.colnz) throw std::invalid_argument("column flags disagree with search rows");
  return d;
}

inline long long ris_integer(const std::string& s) {
  size_t end = 0; long long n = std::stoll(s, &end);
  if (end != s.size()) throw std::invalid_argument("invalid integer");
  return n;
}
inline int ris_int(const std::string& s) {
  long long n = ris_integer(s);
  if (n < std::numeric_limits<int>::min() || n > std::numeric_limits<int>::max()) throw std::invalid_argument("integer out of range");
  return (int)n;
}
inline double ris_real(const std::string& s) {
  size_t end = 0; double n = std::stod(s, &end);
  if (end != s.size() || !(n >= 0) || n > 1e12) throw std::invalid_argument("invalid nonnegative finite number");
  return n;
}
inline void ris_print_vec(const u64* v, int n, int nq) {
  std::printf("\"x\":["); bool first = true;
  for (int q = 0; q < n; q++) if ((v[q/64] >> (q%64)) & 1) { std::printf("%s%d", first ? "" : ",", q); first = false; }
  std::printf("],\"z\":["); first = true;
  for (int q = 0; q < n; q++) if ((v[nq+q/64] >> (q%64)) & 1) { std::printf("%s%d", first ? "" : ",", q); first = false; }
  std::printf("]");
}
