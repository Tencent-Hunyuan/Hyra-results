// Checkpoint bookkeeping. Digests are computed/re-checked by the Python caller.
#pragma once
#include <fstream>
#include <sstream>
#include <set>
#include <filesystem>
#include <sys/file.h>
#include <unistd.h>
#include <cerrno>
#include <zlib.h>

// Fault injection is supplied only by the test compiler's -include header.
#ifndef STABCERT_CHECKPOINT_WRITE
#define STABCERT_CHECKPOINT_WRITE(data, size, file, kind) std::fwrite(data, 1, size, file)
#endif
#ifndef STABCERT_CHECKPOINT_FLUSH
#define STABCERT_CHECKPOINT_FLUSH(file, kind) std::fflush(file)
#endif
#ifndef STABCERT_CHECKPOINT_FSYNC
#define STABCERT_CHECKPOINT_FSYNC(fd, kind) ::fsync(fd)
#endif
#ifndef STABCERT_CHECKPOINT_CLOSE
#define STABCERT_CHECKPOINT_CLOSE(file, kind) std::fclose(file)
#endif

struct Piece {
  long long a, b, nodes, maxn, rf;
  std::vector<long long> ab;
};

[[noreturn]] static void checkpoint_error(const std::string& reason) {
  std::fprintf(stderr, "checkpoint rejected: %s\n", reason.c_str());
  std::exit(6);
}

[[noreturn]] static void checkpoint_io_error(const char* operation) {
  std::fprintf(stderr, "checkpoint %s failed: %s\n", operation, std::strerror(errno));
  std::fflush(stderr);
  // An I/O error can occur in consume() while another GPU slot is active.
  // Avoid CUDA exit handlers racing those workers. Unflushed checkpoint
  // bytes are uncommitted and safely replayed by the next process.
  std::_Exit(1);
}

// zlib's IEEE CRC-32 protects against accidental corruption, not hostile edits.
// Hash the exact header (including newline), then each payload plus newline.
static uLong checkpoint_crc(uLong crc, const std::string& bytes) {
  for (size_t p = 0; p < bytes.size();) {
    uInt n = (uInt)std::min(bytes.size() - p, (size_t)std::numeric_limits<uInt>::max());
    crc = crc32(crc, reinterpret_cast<const Bytef*>(bytes.data() + p), n);
    p += n;
  }
  return crc;
}

static std::string checkpoint_checksum(uLong crc) {
  char suffix[17];
  std::snprintf(suffix, sizeof suffix, " crc32=%08lx", crc);
  return suffix;
}

class Checkpoint {
  FILE* file = nullptr;
  std::string path;
  size_t valid_bytes = 0;
  bool torn_tail = false;
public:
  bool resumed = false;
  std::vector<std::vector<Piece>> previous;
  Found saved_find;
  Checkpoint(const std::string& name, const std::string& header, int cap, int n, int domain) : path(name) {
    if (path.empty()) return;
    const std::string header_line = header + "\n";
    chain = checkpoint_crc(crc32(0, Z_NULL, 0), header_line);
    // The separate lock survives atomic/torn-tail handling and never alters the checkpoint.
    std::string lockname = path + ".lock";
    lockfd = ::open(lockname.c_str(), O_CREAT | O_RDWR, 0600);
    if (lockfd < 0 || flock(lockfd, LOCK_EX | LOCK_NB)) checkpoint_error("file is already in use or cannot lock");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      if (std::filesystem::exists(path)) checkpoint_error("cannot read file");
      pending_header = header_line;
      return;
    }
    resumed = true;
    std::string line;
    std::getline(input, line);
    if (input.bad()) checkpoint_io_error("read");
    // Only the exact prefix of this run's header can be a recoverable torn
    // header. This includes an empty file and a header missing just its newline.
    if (input.eof() && line.size() < header_line.size() && header_line.compare(0, line.size(), line) == 0) {
      pending_header = header_line;
      torn_tail = true;
      resumed = false;
      return;
    }
    if (line.rfind("H stabcert-bnb-ckpt 1 ", 0) == 0)
      checkpoint_error("version-1 checkpoints lack integrity checks; start a new checkpoint file");
    if (!input || input.eof() || line != header)
#ifdef STABCERT_RULES
      checkpoint_error("header mismatch (instance/domain/cap/split/delta/stop_first/source/binary/gen_tasks/bound)");
#else
      checkpoint_error("header mismatch (instance/domain/cap/split/delta/stop_first/source/binary/gen_tasks)");
#endif
    valid_bytes = line.size() + 1;
    int last_round = -1;
    while (std::getline(input, line)) {
      if (input.eof()) { torn_tail = true; break; } // incomplete final write is not a committed record
      valid_bytes += line.size() + 1;
      if (saved_find.w < (1 << 30)) checkpoint_error("records after final find");
      size_t checksum = line.rfind(" crc32=");
      if (checksum == std::string::npos) checkpoint_error("missing record checksum");
      std::string payload = line.substr(0, checksum);
      chain = checkpoint_crc(chain, payload + "\n");
      if (line.substr(checksum) != checkpoint_checksum(chain)) checkpoint_error("record checksum mismatch");
      line = std::move(payload);
      std::istringstream row(line);
      char tag; if (!(row >> tag)) checkpoint_error("empty record");
      if (tag == 'F') {
        int w; if (!(row >> w) || w < 1 || w > cap) checkpoint_error("invalid final find weight");
        saved_find.w = w;
        std::set<int> seen;
        for (int i = 0; i < w; i++) {
          int q, l;
          if (!(row >> q >> l) || q < 0 || q >= n || l < 1 || l > 3 || (domain && l != domain) || !seen.insert(q).second)
            checkpoint_error("invalid final witness");
          saved_find.wit.push_back({q, l});
        }
      } else if (tag == 'C') {
        int r; long long nab; Piece p;
        if (!(row >> r >> p.a >> p.b >> p.nodes >> p.maxn >> p.rf >> nab) || r < 0 || r >= DMAXW || r < last_round || r > last_round + 1 || p.a < 0 || p.b <= p.a || p.nodes < 0 || p.maxn < 0 || p.maxn > p.nodes || p.rf != 0 || nab < 0 || nab > p.b-p.a)
          checkpoint_error("invalid chunk record");
        last_round = r;
        long long prev = -1;
        for (long long i = 0; i < nab; i++) {
          long long t;
          if (!(row >> t) || t < p.a || t >= p.b || t <= prev) checkpoint_error("invalid aborted task index");
          p.ab.push_back(t); prev = t;
        }
        if ((int)previous.size() <= r) previous.resize(r+1);
        previous[r].push_back(std::move(p));
      } else checkpoint_error("unknown record type");
      std::string extra; if (row >> extra) checkpoint_error("trailing record data");
    }
    if (input.bad()) checkpoint_io_error("read");
  }
  // Called only after all records and task indices have been checked.
  void activate() {
    if (path.empty()) return;
    if (torn_tail) std::filesystem::resize_file(path, valid_bytes);
    file = std::fopen(path.c_str(), "ab");
    if (!file) checkpoint_io_error("open for append");
    if (!pending_header.empty()) write_line(pending_header, 'H');
    if (saved_find.w < (1 << 30)) {
      // A previous process may have died after fflush but before fsync. Make
      // the existing terminal record durable before re-emitting its witness.
      sync_file('F');
      wrote_find = true;
    } else if (pending_header.empty()) {
      // Complete C records may also have survived a crash before their sync.
      sync_file('C');
    }
  }
  void append(const std::string& payload, bool final_find = false) {
    if (!file) return;
    if (wrote_find) checkpoint_error("append after terminal find");
    chain = checkpoint_crc(chain, payload + "\n");
    write_line(payload + checkpoint_checksum(chain) + "\n", final_find ? 'F' : 'C');
  }
  void find(const Found& f) {
    if (!file || f.w >= (1 << 30) || wrote_find) return;
    std::ostringstream line; line << "F " << f.w;
    for (auto p : f.wit) line << ' ' << p.first << ' ' << p.second;
    // F is terminal: the finding piece never receives a C record. Until F
    // is complete, that piece remains replayable at the original bound.
    append(line.str(), true); wrote_find = true;
    STABCERT_CHECKPOINT_HOOK("find-committed");
  }
  bool has_file() const { return !path.empty(); }
  // Explicitly called before done; the destructor is only a fallback.
  void close() {
    if (!file) return;
    FILE* closing = file; file = nullptr;
    if (STABCERT_CHECKPOINT_CLOSE(closing, wrote_find ? 'F' : 'C')) checkpoint_io_error("close");
  }
  ~Checkpoint() { close(); if (lockfd >= 0) ::close(lockfd); }
private:
  void write_line(const std::string& line, char kind) {
    if (kind == 'F') STABCERT_CHECKPOINT_HOOK("before-find-write");
    if (kind == 'C') STABCERT_CHECKPOINT_HOOK("before-chunk-write");
    if (STABCERT_CHECKPOINT_WRITE(line.data(), line.size(), file, kind) != line.size()) checkpoint_io_error("write");
    if (kind == 'F') STABCERT_CHECKPOINT_HOOK("after-find-write");
    if (kind == 'C') STABCERT_CHECKPOINT_HOOK("after-chunk-write");
    if (STABCERT_CHECKPOINT_FLUSH(file, kind)) checkpoint_io_error("flush");
    if (kind == 'F') STABCERT_CHECKPOINT_HOOK("after-find-flush");
    if (kind == 'C') STABCERT_CHECKPOINT_HOOK("after-chunk-flush");
    sync_file(kind);
  }
  void sync_file(char kind) {
    if (kind == 'F') STABCERT_CHECKPOINT_HOOK("before-find-fsync");
    if (kind == 'C') STABCERT_CHECKPOINT_HOOK("before-chunk-fsync");
    int rc;
    do { rc = STABCERT_CHECKPOINT_FSYNC(fileno(file), kind); } while (rc && errno == EINTR);
    if (rc) checkpoint_io_error("fsync");
    if (kind == 'F') STABCERT_CHECKPOINT_HOOK("after-find-fsync");
    if (kind == 'C') STABCERT_CHECKPOINT_HOOK("after-chunk-fsync");
  }
  uLong chain = 0;
  int lockfd = -1;
  bool wrote_find = false;
  std::string pending_header;
};

static void validate_checkpoint(const HostCode& h, const Batch& initial, const Checkpoint& ck,
                                int cap, int delta, int threads) {
  // Reconstruct recorded task numbering before any append: malformed later rounds
  // must not cause earlier valid rounds to be appended first.
  Batch tasks = initial;
  for (size_t round = 0; round < ck.previous.size(); round++) {
    std::vector<char> covered(tasks.count(), 0);
    std::vector<uint32_t> roots;
    std::vector<std::vector<uint8_t>> paths;
    std::vector<int> splits;
    for (const Piece& p : ck.previous[round]) {
      if (p.b > tasks.count()) checkpoint_error("chunk task index out of range");
      for (long long t = p.a; t < p.b; t++) {
        if (covered[t]) checkpoint_error("overlapping chunks");
        covered[t] = 1;
      }
      for (long long t : p.ab) {
        roots.push_back(tasks.root[t]);
        paths.emplace_back(tasks.path.begin() + t*tasks.stride, tasks.path.begin() + t*tasks.stride + tasks.len[t]);
        int split = std::min(cap, tasks.len[t]+1+delta);
        if (split <= tasks.len[t]+1) checkpoint_error("aborted task cannot be split further");
        splits.push_back(split);
      }
    }
    if (round+1 < ck.previous.size()) {
      if (std::find(covered.begin(), covered.end(), 0) != covered.end()) checkpoint_error("next round after incomplete round");
      long long nodes = 0, fail = 0; Found found;
      tasks = expand_all(h, cap, roots, paths, splits, threads, nodes, fail, found);
      if (fail || found.w < (1 << 30)) checkpoint_error("recorded re-expansion cannot be reproduced");
    }
  }
  if (ck.saved_find.w < (1 << 30)) {
    Code c = h.view(); Dfs st; st.init(c, cap);
    for (auto p : ck.saved_find.wit) st.change(c, p.first, st.lab[p.first], p.second);
    if (st.anyA(c) || !st.nontrivial(c)) checkpoint_error("saved witness is not a logical");
  }
}
