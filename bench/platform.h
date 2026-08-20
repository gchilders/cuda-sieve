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

/* The longest path this layer will handle. ckpt.h's CKPT_PATH_MAX is defined
 * from this, so the two cannot drift: raising it there raises the buffers
 * here. Kept in the platform header because platform.c must not include
 * ckpt.h -- the dependency runs the other way. */
#define BENCH_PATH_MAX 2048

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
int bench_seek_end(FILE *f);
int bench_truncate(FILE *f, uint64_t off);
int bench_fstat_stream(FILE *f, bench_stat_t *st);

/* Durable file identity -- the "is this still the same file?" test.
 *
 * NOT st_dev/st_ino from bench_stat_t: MSVC's _stat64 reports st_ino as a
 * hardcoded 0 and st_dev as a bare drive index, so a (dev, ino) comparison on
 * Windows silently reduces to "some file on the same drive" and matches any
 * peer. Windows keeps the real answer behind GetFileInformationByHandle, so
 * the identity has to come from a handle rather than a stat buffer.
 *
 * On POSIX these are exactly st_dev and st_ino, so a Linux checkpoint written
 * before this existed still compares equal. `volume` and `index` are opaque:
 * compare them, never interpret them. An all-zero id means the filesystem
 * would not supply one -- bench_file_id_valid() is false for it, and a caller
 * gating a destructive action MUST treat that as "unknown", never as a
 * match. */
typedef struct { uint64_t volume, index; } bench_file_id_t;
int bench_file_id_stream(FILE *f, bench_file_id_t *id);
int bench_file_id_path(const char *path, bench_file_id_t *id);
int bench_file_id_valid(const bench_file_id_t *id);
int bench_file_id_equal(const bench_file_id_t *a, const bench_file_id_t *b);
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
int bench_gmtime(const time_t *t, struct tm *out);
int64_t bench_getline(char **line, size_t *cap, FILE *f);
void bench_fast_exit(int code);

/* Clean-stop request hook.
 *
 * The caller wants one thing -- "the operator asked this run to stop" -- and
 * the two platforms deliver it through unrelated mechanisms, so the mechanism
 * belongs here rather than at the call site.
 *
 * POSIX: SIGINT and SIGTERM, with the previous dispositions saved and put
 * back by bench_stop_hook_remove().
 *
 * Windows: a console control handler. signal(SIGTERM, ...) compiles and
 * succeeds under MSVC but the Win32 CRT never raises SIGTERM -- only raise()
 * can -- so the SIGTERM registration this replaces was dead code, and ^C was
 * the only thing that ever reached it. The handler covers CTRL_C_EVENT,
 * CTRL_BREAK_EVENT and the CTRL_CLOSE/LOGOFF/SHUTDOWN events.
 *
 * WHAT WINDOWS STILL CANNOT DO: TerminateProcess -- which is how most job
 * queues, and the BOINC client, stop a task -- runs no handler at all. There
 * is no way to intercept it, so on Windows --stop-file is the only reliable
 * way to ask for a checkpointed stop. The close/logoff/shutdown events also
 * grant only a few seconds before the process dies, which may not be long
 * enough to drain a cofactor queue.
 *
 * The callback runs on a separate thread on Windows and in signal context on
 * POSIX. It must touch nothing but a volatile sig_atomic_t flag. */
/* Run-time dynamic loading, for the optional NVML binding in runlog.c.
 *
 * bench_dlsym writes THROUGH a slot rather than returning the address: neither
 * dlsym's void * nor GetProcAddress's FARPROC converts to a function pointer
 * in a way ISO C guarantees, and memcpy into the target slot is the portable
 * spelling that compiles to nothing. Pass the address of the function pointer
 * you want filled. A name that does not resolve leaves the slot NULL. */
void *bench_dlopen(const char *name);
void  bench_dlsym(void *lib, const char *name, void *slot);
void  bench_dlclose(void *lib);

typedef void (*bench_stop_handler_t)(void);
int  bench_stop_hook_install(bench_stop_handler_t fn);
void bench_stop_hook_remove(void);

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

/* Keep the builtin where the compiler has one. This replaced a direct
 * __builtin_popcount, and an unconditional SWAR body costs the Linux build
 * ~12 ALU ops per word where it had been a single POPCNT -- a portability
 * shim is not a reason to give that up on the platform that already had it.
 * __builtin_popcount is not an unconditional instruction: GCC and clang emit
 * POPCNT when the target allows it (the Makefile builds -march=native) and
 * call their own fallback otherwise, which is exactly the right behaviour.
 *
 * MSVC deliberately keeps the SWAR body. Its __popcnt intrinsic compiles to
 * the instruction unconditionally, with no fallback, so a binary built with
 * it faults on any CPU without SSE4.2 -- and unlike -march=native there is no
 * compile-time target here that rules that out. This path was never the
 * regression; it is the one the shim was written for. */
#if defined(__GNUC__) || defined(__clang__)
static inline unsigned bench_popcount32(uint32_t x)
{
    return (unsigned)__builtin_popcount(x);
}
#else
static inline unsigned bench_popcount32(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (unsigned)((x * 0x01010101u) >> 24);
}
#endif

static inline char *bench_strdup(const char *s)
{
    const size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

#endif
