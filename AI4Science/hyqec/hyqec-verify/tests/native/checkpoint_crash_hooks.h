// Included explicitly by fault-injection builds, never by package builds.
#pragma once
#ifdef __CUDA_ARCH__
#define STABCERT_CHECKPOINT_HOOK(point) ((void)0)
#else
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace checkpoint_test {
inline void hook(const char* point) {
  std::fprintf(stderr, "CKPT_TEST %s\n", point);
  std::fflush(stderr);
  const char* selected = std::getenv("STABCERT_TEST_KILL_POINT");
  if (selected && !std::strcmp(selected, point)) {
    ::kill(::getpid(), SIGKILL);
    ::_exit(99); // A test fails unless the observed status is SIGKILL.
  }
}
inline bool fault_is(char kind, const char* name) {
  const char* selected = std::getenv("STABCERT_TEST_IO_KIND");
  const char* fault = std::getenv("STABCERT_TEST_IO_FAULT");
  return selected && selected[0] == kind && fault && !std::strcmp(fault, name);
}
inline size_t write_record(const char* data, size_t size, FILE* file, char kind) {
  const char* variable = kind == 'F' ? "STABCERT_TEST_PARTIAL_FIND_BYTES" :
                         kind == 'C' ? "STABCERT_TEST_PARTIAL_CHUNK_BYTES" : "STABCERT_TEST_PARTIAL_HEADER_BYTES";
  if (const char* text = std::getenv(variable)) {
    size_t count = std::strtoul(text, nullptr, 10);
    if (count > size) ::_exit(98);
    if (std::fwrite(data, 1, count, file) != count || std::fflush(file)) ::_exit(97);
    hook(kind == 'F' ? "partial-find-write" : kind == 'C' ? "partial-chunk-write" : "partial-header-write");
  }
  if (fault_is(kind, "short-write")) {
    size_t written = std::fwrite(data, 1, size / 2, file);
    errno = EIO;
    return written;
  }
  if (fault_is(kind, "write-ENOSPC")) { errno = ENOSPC; return 0; }
  return std::fwrite(data, 1, size, file);
}
inline int flush_record(FILE* file, char kind) {
  if (fault_is(kind, "flush-EIO")) { errno = EIO; return EOF; }
  if (fault_is(kind, "flush-ENOSPC")) { errno = ENOSPC; return EOF; }
  return std::fflush(file);
}
inline int sync_record(int fd, char kind) {
  hook(kind == 'F' ? "fsync-called" : kind == 'C' ? "chunk-fsync-called" : "header-fsync-called");
  const char* fault = kind == 'F' ? std::getenv("STABCERT_TEST_FSYNC_FAULT") : nullptr;
  if ((fault && !std::strcmp(fault, "EIO")) || fault_is(kind, "fsync-EIO")) { errno = EIO; return -1; }
  if (fault_is(kind, "fsync-ENOSPC")) { errno = ENOSPC; return -1; }
  static bool interrupted = false;
  if (((fault && !std::strcmp(fault, "EINTR")) || fault_is(kind, "fsync-EINTR")) && !interrupted) {
    interrupted = true; errno = EINTR; return -1;
  }
  return ::fsync(fd);
}
inline int close_record(FILE* file, char kind) {
  int rc = std::fclose(file);
  if (fault_is(kind, "close-EIO")) { errno = EIO; return EOF; }
  if (fault_is(kind, "close-ENOSPC")) { errno = ENOSPC; return EOF; }
  return rc;
}
}
#define STABCERT_CHECKPOINT_HOOK(point) checkpoint_test::hook(point)
#define STABCERT_CHECKPOINT_WRITE(data, size, file, kind) checkpoint_test::write_record(data, size, file, kind)
#define STABCERT_CHECKPOINT_FLUSH(file, kind) checkpoint_test::flush_record(file, kind)
#define STABCERT_CHECKPOINT_CLOSE(file, kind) checkpoint_test::close_record(file, kind)
#define STABCERT_CHECKPOINT_FSYNC(fd, kind) checkpoint_test::sync_record(fd, kind)
#endif
