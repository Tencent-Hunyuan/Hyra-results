#pragma once
#include <cstdio>
#ifndef STABCERT_SOURCE_SHA256
#define STABCERT_SOURCE_SHA256 "unrecorded"
#endif
#ifndef STABCERT_CUDA_ARCH
#define STABCERT_CUDA_ARCH "unknown"
#endif
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
#endif
namespace stabcert {
inline const char* json_bool(bool value) { return value ? "true" : "false"; }
inline void version(const char* name, const char* limits) {
#ifdef __CUDACC__
  const char* cuda = "true";
  const char* arch = "\"" STABCERT_CUDA_ARCH "\"";
#else
  const char* cuda = "false";
  const char* arch = "null";
#endif
#ifdef _OPENMP
  const char* openmp = "true";
#else
  const char* openmp = "false";
#endif
  std::printf("{\"name\":\"%s\",\"interface\":1,\"source_sha256\":\"%s\","
              "\"cuda\":%s,\"arch\":%s,\"openmp\":%s,\"limits\":%s,"
              "\"warning\":%s}\n", name, STABCERT_SOURCE_SHA256, cuda, arch,
              openmp, limits,
#ifdef _OPENMP
              "null"
#else
              "\"built without OpenMP; one CPU thread\""
#endif
  );
}
}
