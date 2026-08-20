#ifndef BENCH_PLATFORM_H
#define BENCH_PLATFORM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
typedef struct _stat64 bench_stat_t;
#else
typedef struct stat bench_stat_t;
#endif

/* Small host-OS surface used by the main executable.  The POSIX
 * implementation is intentionally just the calls the Linux build already
 * made; Windows maps the same semantics onto the MSVC CRT/Win32 API. */
double bench_monotonic_ms(void);
int bench_sync_stream(FILE *f);
int64_t bench_tell(FILE *f);
int bench_seek(FILE *f, uint64_t off);
int bench_truncate(FILE *f, uint64_t off);
int bench_fstat_stream(FILE *f, bench_stat_t *st);
int bench_stat_path(const char *path, bench_stat_t *st);
int bench_is_regular_mode(unsigned short mode);
int bench_stdout_is_tty(void);
int bench_path_exists(const char *path);
int bench_atomic_replace(const char *src, const char *dst);
int bench_sync_parent(const char *path);
int bench_lock_create(const char *path);
int bench_process_exists(long pid);
long bench_getpid(void);
int bench_fd_write(int fd, const void *buf, size_t n);
int bench_fd_close(int fd);
int bench_localtime(const time_t *t, struct tm *out);
int64_t bench_getline(char **line, size_t *cap, FILE *f);
void bench_fast_exit(int code);

#ifdef __cplusplus
}
#endif

/* Host arithmetic helpers used by CUDA translation units.  MSVC has no
 * unsigned __int128, but x64 exposes the same operations as intrinsics. */
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
typedef struct { uint64_t lo, hi; } bench_u128_t;
static __forceinline uint64_t bench_mulhi_u64(uint64_t a, uint64_t b)
{
#if defined(_M_X64)
    return __umulh(a, b);
#else
    const uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    const uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    const uint64_t p00 = a0 * b0;
    const uint64_t p01 = a0 * b1;
    const uint64_t p10 = a1 * b0;
    const uint64_t p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}
static __forceinline bench_u128_t bench_mul_u64_wide(uint64_t a, uint64_t b)
{
    bench_u128_t r;
#if defined(_M_X64)
    r.lo = _umul128(a, b, &r.hi);
#else
    const uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    const uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    const uint64_t p00 = a0 * b0;
    const uint64_t p01 = a0 * b1;
    const uint64_t p10 = a1 * b0;
    const uint64_t p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    r.lo = (p00 & 0xffffffffULL) | (mid << 32);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
    return r;
}
static __forceinline uint64_t bench_div_u128_u64(uint64_t hi, uint64_t lo,
                                                  uint64_t d, uint64_t *rem)
{
#if defined(_M_X64)
    return _udiv128(hi, lo, d, rem);
#else
    /* The CUDA sieve's Windows target is x64.  Keep a compile-time fallback
     * for other MSVC targets rather than exposing __int128 syntax to cl. */
    uint64_t q = 0, r = hi;
    for (int i = 63; i >= 0; --i) {
        const uint64_t carry = r >> 63;
        r = (r << 1) | ((lo >> i) & 1ULL);
        if (carry || r >= d) { r -= d; q |= 1ULL << i; }
    }
    *rem = r;
    return q;
#endif
}
#else
typedef struct { uint64_t lo, hi; } bench_u128_t;
static inline uint64_t bench_mulhi_u64(uint64_t a, uint64_t b)
{
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
}
static inline bench_u128_t bench_mul_u64_wide(uint64_t a, uint64_t b)
{
    const unsigned __int128 v = (unsigned __int128)a * (unsigned __int128)b;
    bench_u128_t r = { (uint64_t)v, (uint64_t)(v >> 64) };
    return r;
}
static inline uint64_t bench_div_u128_u64(uint64_t hi, uint64_t lo,
                                           uint64_t d, uint64_t *rem)
{
    const unsigned __int128 v = ((unsigned __int128)hi << 64) | lo;
    *rem = (uint64_t)(v % d);
    return (uint64_t)(v / d);
}
#endif

static inline uint32_t bench_u128_take32(bench_u128_t *v)
{
    const uint32_t out = (uint32_t)v->lo;
    v->lo = (v->lo >> 32) | (v->hi << 32);
    v->hi >>= 32;
    return out;
}

static inline int bench_u128_nonzero(bench_u128_t v)
{
    return v.lo != 0 || v.hi != 0;
}

static inline unsigned bench_popcount32(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (unsigned)((x * 0x01010101u) >> 24);
}

static inline char *bench_strdup(const char *s)
{
    const size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

#endif
