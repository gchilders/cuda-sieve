/* GPU factor-base generation.
 *
 * The standalone fbgen_gpu executable remains a timing/debug harness for the
 * ordinary-prime GPU stages.  The same translation unit also exports the
 * production afb_build_gpu() path used by bench when --fb1 is omitted.
 *
 * Ordinary primes and algebraic roots are generated in large GPU segments.
 * The complete builder then delegates only the tiny exact population -- primes
 * that can contribute p^k entries plus ramified/projective exception primes --
 * to fbgen.c's native p-adic/Hensel implementation.  The two streams are merged
 * into exactly the fb_t representation consumed by the sieve, including logp,
 * ispow and maxbits metadata.  Thus the standalone timing mode can still omit
 * powers for clean kernel measurements while the production path does not.
 *
 * The standalone harness also generates the rational prime root -Y0/Y1 mod p
 * for timing.  Production bench already has an exact in-process rational factor
 * base builder, so afb_build_gpu() is deliberately algebraic-only.
 *
 * The modular root finder is a direct GPU-oriented implementation of the same
 * x^p-x / gcd / deterministic splitting construction used by fbgen.c.  The
 * default arithmetic backend is now 32-bit Montgomery reduction: the repeated
 * polynomial multiplies use mul.lo/mul.hi rather than uint64 % uint32.  The
 * original remainder backend remains available with --alg-backend legacy for
 * same-binary A/B timing and regression checks.  Degree <=6 and <=8 inputs
 * default to fixed-capacity kernel specializations with a stack-free splitter;
 * --alg-kernel generic keeps the original degree-generic implementation as the
 * reference/fallback path.
 */
#include "bench.h"
#include "platform.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#define GPU_FB_BIG_LIMBS 20
#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)
#define GPU_FB_BRUTE_ROOT_LIMIT 67u
#define GPU_FB_DEFAULT_SEG_ODDS (1u << 23)  /* 8M odds = ~16M integers */
/* counts/offsets and the compact-root count are uint32_t.  A prime contributes
 * at most degree+1 <= GPU_FB_MAX_ROOTS entries, so keep every segment small
 * enough that even the pessimistic slot count cannot wrap.  This also keeps
 * CUB item counts and the (n+255)/256 launch arithmetic comfortably in range. */
#define GPU_FB_MAX_SEG_ODDS (UINT32_MAX / GPU_FB_MAX_ROOTS)

static int gpu_fb_normalize_segment_odds(uint32_t *segment_odds)
{
    if (!segment_odds) { errno = EINVAL; return -1; }
    if (!*segment_odds) *segment_odds = GPU_FB_DEFAULT_SEG_ODDS;
    if (*segment_odds > GPU_FB_MAX_SEG_ODDS) {
        fprintf(stderr,
                "fbgen_gpu: --segment-odds %u exceeds safe maximum %u\n",
                *segment_odds, (uint32_t)GPU_FB_MAX_SEG_ODDS);
        errno = ERANGE;
        return -1;
    }
    return 0;
}

template <typename T>
static int vector_resize_nothrow(std::vector<T> *v, size_t n)
{
    try { v->resize(n); }
    catch (const std::bad_alloc &) { errno = ENOMEM; return -1; }
    return 0;
}

#ifndef FBGEN_GPU_LIBRARY
/* These live in fbgen.c and are intentionally the same serializer used by
 * native fbgen.  Keep them out of bench.h: production bench never writes an
 * algebraic FB text file, and its fbgen_gpu_lib.o excludes all code below that
 * needs these entry points. */
extern "C" int fbgen_write_header(FILE *out, const poly_t *P,
                                   uint32_t lim, int maxbits);
extern "C" int fbgen_write_prime_entries(FILE *out,
                                          const fbgen_exact_entry_t *v,
                                          size_t n);
#endif

#define CUDA_OR_DIE(x) do {                                                     \
    cudaError_t _e = (x);                                                       \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "fbgen_gpu: CUDA %s failed at %s:%d: %s\n",           \
                #x, __FILE__, __LINE__, cudaGetErrorString(_e));                \
        goto fail;                                                              \
    }                                                                           \
} while (0)

typedef struct {
    uint32_t v[GPU_FB_BIG_LIMBS];
    int n;
    int neg;
} gpu_big_t;

typedef struct {
    int deg;
    uint32_t c[BENCH_NCOEFF];
} dpoly_t;

__constant__ gpu_big_t c_alg[BENCH_NCOEFF];
__constant__ int c_alg_deg;
__constant__ gpu_big_t c_y0, c_y1;

static int host_big_parse(gpu_big_t *a, const char *s)
{
    memset(a, 0, sizeof(*a));
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-' || *s == '+') { a->neg = (*s == '-'); s++; }
    if (!*s) return -1;
    for (; *s; s++) {
        uint64_t carry;
        int i;
        if (*s < '0' || *s > '9') return -1;
        carry = (uint32_t)(*s - '0');
        for (i = 0; i < a->n; i++) {
            uint64_t t = (uint64_t)a->v[i] * 10u + carry;
            a->v[i] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) {
            if (a->n == GPU_FB_BIG_LIMBS) return -1;
            a->v[a->n++] = (uint32_t)carry;
        }
    }
    while (a->n && !a->v[a->n - 1]) a->n--;
    if (!a->n) a->neg = 0;
    return 0;
}

#ifndef FBGEN_GPU_LIBRARY
/* Standalone validation/output helpers.  Production bench never evaluates the
 * parsed decimal polynomial on the host; excluding these from fbgen_gpu_lib.o
 * keeps the library build warning-free without suppressing diagnostics. */
static uint32_t host_dec_mod(const char *s, uint32_t p)
{
    uint64_t r = 0;
    int neg = 0;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-' || *s == '+') { neg = (*s == '-'); s++; }
    while (*s >= '0' && *s <= '9') {
        r = (r * 10u + (uint32_t)(*s - '0')) % p;
        s++;
    }
    if (neg && r) r = p - r;
    return (uint32_t)r;
}

static uint32_t host_poly_eval_mod(const poly_t *P, uint32_t x, uint32_t p)
{
    uint64_t r = 0;
    for (int k = P->deg; k >= 0; k--)
        r = (r * x + host_dec_mod(P->cs[k][0] ? P->cs[k] : "0", p)) % p;
    return (uint32_t)r;
}
#endif

__device__ __forceinline__ uint32_t d_big_mod(const gpu_big_t *a, uint32_t p)
{
    uint64_t r = 0;
    for (int i = a->n - 1; i >= 0; i--)
        r = ((r << 32) | a->v[i]) % p;
    if (a->neg && r) r = p - r;
    return (uint32_t)r;
}

__device__ __forceinline__ uint32_t d_mod_add(uint32_t a, uint32_t b, uint32_t p)
{
    /* a,b < p.  Work entirely in 32 bits: on wrap, subtracting p in the
     * wrapped word is exactly (a+b)-p because the true sum is below 2p. */
    uint32_t z = a + b;
    if (z < a || z >= p) z -= p;
    return z;
}

__device__ __forceinline__ uint32_t d_mod_sub(uint32_t a, uint32_t b, uint32_t p)
{
    return a >= b ? a - b : p - (b - a);
}

/* Legacy remainder multiply retained for the rational root and for the
 * --alg-backend legacy comparison path. */
__device__ __forceinline__ uint32_t d_mod_mul(uint32_t a, uint32_t b, uint32_t p)
{
    return (uint32_t)(((uint64_t)a * b) % p);
}

/* Independent deterministic primality gate for ordinary q=p entries emitted
 * by the complete GPU builder.  This deliberately does not share the segmented
 * sieve or Montgomery root arithmetic: for 32-bit n, bases 2,7,61 are a
 * deterministic Miller-Rabin witness set. */
__device__ __forceinline__ bool d_mr32_pass(uint32_t n, uint32_t d,
                                             unsigned s, uint32_t a)
{
    if (a % n == 0u) return true;
    uint32_t x = 1u, b = a % n;
    uint32_t e = d;
    while (e) {
        if (e & 1u) x = d_mod_mul(x, b, n);
        e >>= 1;
        if (e) b = d_mod_mul(b, b, n);
    }
    if (x == 1u || x == n - 1u) return true;
    for (unsigned r = 1; r < s; r++) {
        x = d_mod_mul(x, x, n);
        if (x == n - 1u) return true;
    }
    return false;
}

__device__ __forceinline__ bool d_is_prime32_independent(uint32_t n)
{
    if (n < 2u) return false;
    if ((n & 1u) == 0u) return n == 2u;
    if (n % 3u == 0u) return n == 3u;
    uint32_t d = n - 1u;
    unsigned s = 0;
    while ((d & 1u) == 0u) { d >>= 1; s++; }
    return d_mr32_pass(n, d, s, 2u) &&
           d_mr32_pass(n, d, s, 7u) &&
           d_mr32_pass(n, d, s, 61u);
}

/* Per-prime 32-bit Montgomery context, R = 2^32.  `one` is R mod p (the
 * Montgomery representation of 1), and r2 is R^2 mod p in ordinary form.
 *
 * The setup deliberately contains no integer division.  For odd p, Newton's
 * iteration gives -p^-1 mod 2^32 in five multiplies.  Sixty-four modular
 * doublings produce R mod p and R^2 mod p.  That fixed setup cost replaces the
 * hundreds to thousands of uint64 %% p operations in the polynomial powers.
 */
typedef struct {
    uint32_t p;
    uint32_t n0inv;
    uint32_t one;
    uint32_t r2;
} mont32_t;

__device__ __forceinline__ mont32_t d_mont_setup(uint32_t p)
{
    mont32_t M;
    uint32_t inv = p;
    uint32_t x = 1;
    M.p = p;
    #pragma unroll
    for (int i = 0; i < 5; i++) inv *= 2u - p * inv;
    M.n0inv = 0u - inv;
    M.one = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        x = d_mod_add(x, x, p);
        if (i == 31) M.one = x;
    }
    M.r2 = x;
    return M;
}

/* REDC(a*b), with a allowed to be any 32-bit word and b < p.  The latter is
 * enough to keep a*b < R*p, so the Montgomery result is below 2p and one
 * subtraction suffices.  tlo + (tlo*n0inv mod R)*p is exactly 0 mod R; its
 * carry into the high word is therefore simply (tlo != 0), avoiding another
 * low-half multiply by p. */
__device__ __forceinline__ uint32_t d_mont_mul(uint32_t a, uint32_t b,
                                               const mont32_t *M)
{
    const uint32_t tlo = a * b;
    const uint32_t thi = __umulhi(a, b);
    const uint32_t m = tlo * M->n0inv;
    const uint32_t mph = __umulhi(m, M->p);
    uint32_t u = thi + mph;
    uint32_t carry = (u < thi);
    const uint32_t before = u;
    u += (tlo != 0u);
    carry |= (u < before);
    if (carry) return u - M->p;  /* represents (2^32 + u) - p */
    if (u >= M->p) u -= M->p;
    return u;
}

__device__ __forceinline__ uint32_t d_mont_from(uint32_t a, const mont32_t *M)
{
    return d_mont_mul(a, M->r2, M);
}

__device__ __forceinline__ uint32_t d_mont_to(uint32_t a, const mont32_t *M)
{
    return d_mont_mul(a, 1u, M);
}

/* Reduce an arbitrary signed base-2^32 integer directly into Montgomery form.
 * If r represents the prefix value times R, appending one 32-bit limb w gives
 * (prefix*R + w)*R.  Multiplication by r2 advances the prefix by one radix and
 * multiplication of w by r2 converts that word to Montgomery form. */
__device__ __forceinline__ uint32_t d_big_mod_mont(const gpu_big_t *a,
                                                   const mont32_t *M)
{
    uint32_t r = 0;
    for (int i = a->n - 1; i >= 0; i--) {
        r = d_mont_mul(r, M->r2, M);
        r = d_mod_add(r, d_mont_mul(a->v[i], M->r2, M), M->p);
    }
    if (a->neg && r) r = M->p - r;
    return r;
}

template <bool MONT>
__device__ __forceinline__ uint32_t d_ctx_mul(uint32_t a, uint32_t b,
                                              const mont32_t *M)
{
    if constexpr (MONT) return d_mont_mul(a, b, M);
    else return (uint32_t)(((uint64_t)a * b) % M->p);
}

template <bool MONT>
__device__ __forceinline__ uint32_t d_ctx_from(uint32_t a, const mont32_t *M)
{
    if constexpr (MONT) return d_mont_from(a, M);
    else return a;
}

template <bool MONT>
__device__ __forceinline__ uint32_t d_ctx_to(uint32_t a, const mont32_t *M)
{
    if constexpr (MONT) return d_mont_to(a, M);
    else return a;
}

template <bool MONT>
__device__ __forceinline__ uint32_t d_ctx_big(const gpu_big_t *a,
                                              const mont32_t *M)
{
    if constexpr (MONT) return d_big_mod_mont(a, M);
    else return d_big_mod(a, M->p);
}

template <bool MONT>
__device__ __forceinline__ uint32_t d_ctx_one(const mont32_t *M)
{
    if constexpr (MONT) return M->one;
    else return 1u;
}

/* Binary extended GCD for an odd prime modulus.  This avoids a p-2 power just
 * to make a polynomial monic.  All ordinary coefficient state stays in [0,p). */
__device__ uint32_t d_mod_inv(uint32_t a, uint32_t p)
{
    if (a == 0) return 0;
    if (p == 2) return 1;
    uint32_t u = a, v = p, x1 = 1, x2 = 0;
    while (u != 1 && v != 1) {
        while (!(u & 1u)) {
            u >>= 1;
            x1 = !(x1 & 1u) ? (x1 >> 1)
                              : (uint32_t)(((uint64_t)x1 + p) >> 1);
        }
        while (!(v & 1u)) {
            v >>= 1;
            x2 = !(x2 & 1u) ? (x2 >> 1)
                              : (uint32_t)(((uint64_t)x2 + p) >> 1);
        }
        if (u >= v) {
            u -= v;
            x1 = x1 >= x2 ? x1 - x2 : (uint32_t)((uint64_t)x1 + p - x2);
        } else {
            v -= u;
            x2 = x2 >= x1 ? x2 - x1 : (uint32_t)((uint64_t)x2 + p - x1);
        }
    }
    return u == 1 ? x1 : x2;
}

__device__ __forceinline__ void d_poly_zero(dpoly_t *a)
{
    a->deg = -1;
    #pragma unroll
    for (int i = 0; i < BENCH_NCOEFF; i++) a->c[i] = 0;
}

__device__ __forceinline__ void d_poly_norm(dpoly_t *a)
{
    while (a->deg >= 0 && a->c[a->deg] == 0) a->deg--;
}

template <bool MONT>
__device__ void d_poly_make_monic(dpoly_t *a, const mont32_t *M)
{
    const uint32_t one = d_ctx_one<MONT>(M);
    if (a->deg < 0 || a->c[a->deg] == one) return;
    uint32_t lead = d_ctx_to<MONT>(a->c[a->deg], M);
    uint32_t z = d_ctx_from<MONT>(d_mod_inv(lead, M->p), M);
    for (int i = 0; i <= a->deg; i++) a->c[i] = d_ctx_mul<MONT>(a->c[i], z, M);
}

/* b must be monic. */
template <bool MONT>
__device__ dpoly_t d_poly_rem_monic(dpoly_t a, const dpoly_t *b,
                                    const mont32_t *M)
{
    while (a.deg >= b->deg) {
        uint32_t q = a.c[a.deg];
        int shift = a.deg - b->deg;
        /* b is monic, so the leading cancellation is known exactly.  Do not
         * spend a modular multiply computing q*1 only to subtract q from q. */
        for (int i = 0; i < b->deg; i++)
            a.c[i + shift] = d_mod_sub(a.c[i + shift],
                                       d_ctx_mul<MONT>(q, b->c[i], M), M->p);
        a.c[a.deg] = 0;
        d_poly_norm(&a);
    }
    return a;
}

/* b must be monic and divide a exactly. */
template <bool MONT>
__device__ dpoly_t d_poly_divexact_monic(dpoly_t a, const dpoly_t *b,
                                         const mont32_t *M, uint32_t *failed)
{
    dpoly_t q;
    d_poly_zero(&q);
    while (a.deg >= b->deg) {
        uint32_t z = a.c[a.deg];
        int shift = a.deg - b->deg;
        q.c[shift] = z;
        if (shift > q.deg) q.deg = shift;
        for (int i = 0; i < b->deg; i++)
            a.c[i + shift] = d_mod_sub(a.c[i + shift],
                                       d_ctx_mul<MONT>(z, b->c[i], M), M->p);
        a.c[a.deg] = 0;
        d_poly_norm(&a);
    }
    if (a.deg >= 0) *failed = 1;
    d_poly_norm(&q);
    return q;
}

template <bool MONT>
__device__ dpoly_t d_poly_gcd(dpoly_t a, dpoly_t b, const mont32_t *M)
{
    if (a.deg >= 0) d_poly_make_monic<MONT>(&a, M);
    if (b.deg >= 0) d_poly_make_monic<MONT>(&b, M);
    while (b.deg >= 0) {
        dpoly_t r = d_poly_rem_monic<MONT>(a, &b, M);
        a = b;
        b = r;
        if (b.deg >= 0) d_poly_make_monic<MONT>(&b, M);
    }
    return a;
}

/* mod must be monic. */
template <bool MONT>
__device__ dpoly_t d_poly_mul_rem(const dpoly_t *a, const dpoly_t *b,
                                  const dpoly_t *mod, const mont32_t *M)
{
    uint32_t t[2 * BENCH_MAX_DEGREE + 1];
    dpoly_t r;
    #pragma unroll
    for (int i = 0; i < 2 * BENCH_MAX_DEGREE + 1; i++) t[i] = 0;
    int tdeg = -1;
    for (int i = 0; i <= a->deg; i++) for (int j = 0; j <= b->deg; j++) {
        uint32_t z = d_ctx_mul<MONT>(a->c[i], b->c[j], M);
        t[i + j] = d_mod_add(t[i + j], z, M->p);
        if (t[i + j] && i + j > tdeg) tdeg = i + j;
    }
    while (tdeg >= mod->deg) {
        uint32_t q = t[tdeg];
        int shift = tdeg - mod->deg;
        for (int i = 0; i < mod->deg; i++)
            t[i + shift] = d_mod_sub(t[i + shift],
                                     d_ctx_mul<MONT>(q, mod->c[i], M), M->p);
        t[tdeg] = 0;
        while (tdeg >= 0 && !t[tdeg]) tdeg--;
    }
    d_poly_zero(&r);
    r.deg = tdeg;
    for (int i = 0; i <= tdeg; i++) r.c[i] = t[i];
    return r;
}

/* Polynomial squaring is the dominant operation in x^p mod F.  A generic
 * product computes both a_i*a_j and a_j*a_i; exploit symmetry so a degree-d
 * square uses (d+1)(d+2)/2 modular multiplies instead of (d+1)^2. */
template <bool MONT>
__device__ dpoly_t d_poly_sqr_rem(const dpoly_t *a, const dpoly_t *mod,
                                  const mont32_t *M)
{
    uint32_t t[2 * BENCH_MAX_DEGREE + 1];
    dpoly_t r;
    #pragma unroll
    for (int i = 0; i < 2 * BENCH_MAX_DEGREE + 1; i++) t[i] = 0;
    int tdeg = -1;
    for (int i = 0; i <= a->deg; i++) {
        uint32_t z = d_ctx_mul<MONT>(a->c[i], a->c[i], M);
        t[2 * i] = d_mod_add(t[2 * i], z, M->p);
        if (t[2 * i] && 2 * i > tdeg) tdeg = 2 * i;
        for (int j = i + 1; j <= a->deg; j++) {
            z = d_ctx_mul<MONT>(a->c[i], a->c[j], M);
            z = d_mod_add(z, z, M->p);
            t[i + j] = d_mod_add(t[i + j], z, M->p);
            if (t[i + j] && i + j > tdeg) tdeg = i + j;
        }
    }
    while (tdeg >= mod->deg) {
        uint32_t q = t[tdeg];
        int shift = tdeg - mod->deg;
        for (int i = 0; i < mod->deg; i++)
            t[i + shift] = d_mod_sub(t[i + shift],
                                     d_ctx_mul<MONT>(q, mod->c[i], M), M->p);
        t[tdeg] = 0;
        while (tdeg >= 0 && !t[tdeg]) tdeg--;
    }
    d_poly_zero(&r);
    r.deg = tdeg;
    for (int i = 0; i <= tdeg; i++) r.c[i] = t[i];
    return r;
}

template <bool MONT, bool SYMMETRIC>
__device__ dpoly_t d_poly_pow_rem(dpoly_t a, uint32_t e,
                                  const dpoly_t *mod, const mont32_t *M)
{
    dpoly_t r;
    d_poly_zero(&r);
    r.deg = 0;
    r.c[0] = d_ctx_one<MONT>(M);
    a = d_poly_rem_monic<MONT>(a, mod, M);
    while (e) {
        if (e & 1u) r = d_poly_mul_rem<MONT>(&r, &a, mod, M);
        e >>= 1;
        if (e) {
            if constexpr (SYMMETRIC) a = d_poly_sqr_rem<MONT>(&a, mod, M);
            else a = d_poly_mul_rem<MONT>(&a, &a, mod, M);
        }
    }
    return r;
}

template <bool MONT>
__device__ uint32_t d_poly_eval(const dpoly_t *f, uint32_t x,
                                const mont32_t *M)
{
    uint32_t r = 0;
    uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 0; i--)
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), f->c[i], M->p);
    return r;
}

/* Split a squarefree product of linear factors without device recursion. */
template <bool MONT, bool SYMMETRIC>
__device__ int d_split_linear(const dpoly_t *input, const mont32_t *M,
                              uint32_t *roots, uint32_t *failed)
{
    dpoly_t stack[BENCH_MAX_DEGREE];
    int sp = 0, nr = 0;
    if (input->deg <= 0) return 0;
    stack[sp++] = *input;
    while (sp) {
        dpoly_t f = stack[--sp];
        if (f.deg == 1) {
            /* Every factor placed on this stack is monic. */
            uint32_t c0 = d_ctx_to<MONT>(f.c[0], M);
            roots[nr++] = c0 ? M->p - c0 : 0;
            continue;
        }
        dpoly_t g;
        uint32_t a;
        int found = 0;
        for (a = 0; a < M->p; a++) {
            dpoly_t lin, h;
            d_poly_zero(&lin);
            lin.deg = 1;
            lin.c[0] = d_ctx_from<MONT>(a, M);
            lin.c[1] = d_ctx_one<MONT>(M);
            h = d_poly_pow_rem<MONT, SYMMETRIC>(lin, (M->p - 1) / 2, &f, M);
            if (h.deg < 0) { h.deg = 0; h.c[0] = 0; }
            h.c[0] = d_mod_sub(h.c[0], d_ctx_one<MONT>(M), M->p);
            d_poly_norm(&h);
            g = d_poly_gcd<MONT>(f, h, M);
            if (g.deg > 0 && g.deg < f.deg) { found = 1; break; }
        }
        if (!found || sp + 2 > BENCH_MAX_DEGREE) {
            *failed = 1;
            return nr;
        }
        dpoly_t other = d_poly_divexact_monic<MONT>(f, &g, M, failed);
        if (*failed) return nr;
        d_poly_make_monic<MONT>(&g, M);
        d_poly_make_monic<MONT>(&other, M);
        stack[sp++] = g;
        stack[sp++] = other;
    }
    return nr;
}

template <bool MONT, bool SYMMETRIC>
__device__ int d_alg_roots_prime(uint32_t p, uint32_t *roots, uint32_t *failed)
{
    dpoly_t f;
    mont32_t M;
    M.p = p;
    M.n0inv = M.one = M.r2 = 0;
    if constexpr (MONT) M = d_mont_setup(p);

    uint32_t lead = d_ctx_big<MONT>(&c_alg[c_alg_deg], &M);
    int nr = 0;
    d_poly_zero(&f);
    f.deg = c_alg_deg;
    for (int i = 0; i <= c_alg_deg; i++) f.c[i] = d_ctx_big<MONT>(&c_alg[i], &M);
    d_poly_norm(&f);

    if (f.deg < 0) {
        /* A primitive input polynomial should never vanish identically mod p. */
        *failed = 1;
        return 0;
    }
    if (f.deg == 1) {
        uint32_t c0 = d_ctx_to<MONT>(f.c[0], &M);
        uint32_t c1 = d_ctx_to<MONT>(f.c[1], &M);
        uint32_t inv = d_mod_inv(c1, p);
        uint32_t z = d_ctx_mul<MONT>(d_ctx_from<MONT>(c0 ? p - c0 : 0, &M),
                                     d_ctx_from<MONT>(inv, &M), &M);
        roots[nr++] = d_ctx_to<MONT>(z, &M);
    } else if (p <= GPU_FB_BRUTE_ROOT_LIMIT) {
        for (uint32_t x = 0; x < p; x++)
            if (d_poly_eval<MONT>(&f, x, &M) == 0) roots[nr++] = x;
    } else {
        dpoly_t x, xp, linear;
        d_poly_make_monic<MONT>(&f, &M);
        d_poly_zero(&x);
        x.deg = 1;
        x.c[1] = d_ctx_one<MONT>(&M);
        xp = d_poly_pow_rem<MONT, SYMMETRIC>(x, p, &f, &M);
        if (xp.deg < 1) xp.deg = 1;
        xp.c[1] = d_mod_sub(xp.c[1], d_ctx_one<MONT>(&M), p);
        d_poly_norm(&xp);
        linear = d_poly_gcd<MONT>(f, xp, &M);
        nr = d_split_linear<MONT, SYMMETRIC>(&linear, &M, roots, failed);
        if (*failed) return nr;
    }

    /* Homogeneous projective root.  At the prime level this exists exactly
     * when the original leading coefficient vanishes mod p. */
    if (lead == 0) roots[nr++] = p;
    if (nr > GPU_FB_MAX_ROOTS) {
        *failed = 1;
        return 0;
    }
    /* Deterministic output, matching fbgen's sorted affine roots and placing
     * the projective encoding p last naturally. */
    for (int i = 1; i < nr; i++) {
        uint32_t v = roots[i];
        int j = i;
        while (j && roots[j - 1] > v) { roots[j] = roots[j - 1]; j--; }
        roots[j] = v;
    }
    return nr;
}


/* Fixed-capacity algebraic root path.
 *
 * The original generic dpoly_t path is intentionally retained below as the
 * reference/fallback implementation.  This path instantiates the polynomial
 * capacity at compile time (currently 6 and 8) and avoids the generic
 * dpoly_t stack[BENCH_MAX_DEGREE] used by d_split_linear().  Instead it isolates
 * one linear factor, emits that root, divides it from the remaining squarefree
 * product, and repeats.  That trades a little repeated splitting in uncommon
 * many-root cases for substantially less per-thread local state.
 *
 * Access helpers deliberately turn dynamic coefficient indices into unrolled
 * constant-index selects.  This gives ptxas a chance to scalarize the small
 * coefficient vectors rather than materializing them as local-memory arrays.
 */
template <int CAP>
struct fpoly_t {
    int deg;
    uint32_t c[CAP + 1];
};

template <int CAP>
__device__ __forceinline__ uint32_t fp_get(const fpoly_t<CAP> *a, int idx)
{
    uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (idx == i) v = a->c[i];
    return v;
}

template <int CAP>
__device__ __forceinline__ void fp_set(fpoly_t<CAP> *a, int idx, uint32_t v)
{
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (idx == i) a->c[i] = v;
}

template <int CAP>
__device__ __forceinline__ void fp_zero(fpoly_t<CAP> *a)
{
    a->deg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) a->c[i] = 0;
}

template <int CAP>
__device__ __forceinline__ void fp_copy(fpoly_t<CAP> *dst, const fpoly_t<CAP> *src)
{
    dst->deg = src->deg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) dst->c[i] = src->c[i];
}

template <int CAP>
__device__ __forceinline__ void fp_norm(fpoly_t<CAP> *a)
{
    while (a->deg >= 0 && fp_get(a, a->deg) == 0) a->deg--;
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_make_monic(fpoly_t<CAP> *a, const mont32_t *M)
{
    const uint32_t one = d_ctx_one<MONT>(M);
    if (a->deg < 0) return;
    const uint32_t lc = fp_get(a, a->deg);
    if (lc == one) return;
    const uint32_t lead = d_ctx_to<MONT>(lc, M);
    const uint32_t z = d_ctx_from<MONT>(d_mod_inv(lead, M->p), M);
    #pragma unroll
    for (int i = 0; i <= CAP; i++)
        if (i <= a->deg) a->c[i] = d_ctx_mul<MONT>(a->c[i], z, M);
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_rem_monic(fpoly_t<CAP> *out,
                                              const fpoly_t<CAP> *ain,
                                              const fpoly_t<CAP> *b,
                                              const mont32_t *M)
{
    fpoly_t<CAP> a;
    fp_copy(&a, ain);
    while (a.deg >= b->deg) {
        const uint32_t q = fp_get(&a, a.deg);
        const int shift = a.deg - b->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < b->deg) {
                const int k = i + shift;
                const uint32_t av = fp_get(&a, k);
                const uint32_t bv = b->c[i];
                fp_set(&a, k, d_mod_sub(av, d_ctx_mul<MONT>(q, bv, M), M->p));
            }
        }
        fp_set(&a, a.deg, 0);
        fp_norm(&a);
    }
    fp_copy(out, &a);
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_divexact_monic(fpoly_t<CAP> *qout,
                                                   const fpoly_t<CAP> *ain,
                                                   const fpoly_t<CAP> *b,
                                                   const mont32_t *M,
                                                   uint32_t *failed)
{
    fpoly_t<CAP> a, q;
    fp_copy(&a, ain);
    fp_zero(&q);
    while (a.deg >= b->deg) {
        const uint32_t z = fp_get(&a, a.deg);
        const int shift = a.deg - b->deg;
        fp_set(&q, shift, z);
        if (shift > q.deg) q.deg = shift;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < b->deg) {
                const int k = i + shift;
                const uint32_t av = fp_get(&a, k);
                fp_set(&a, k, d_mod_sub(av,
                                        d_ctx_mul<MONT>(z, b->c[i], M), M->p));
            }
        }
        fp_set(&a, a.deg, 0);
        fp_norm(&a);
    }
    if (a.deg >= 0) *failed = 1;
    fp_norm(&q);
    fp_copy(qout, &q);
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_gcd(fpoly_t<CAP> *out,
                                        const fpoly_t<CAP> *ain,
                                        const fpoly_t<CAP> *bin,
                                        const mont32_t *M)
{
    fpoly_t<CAP> a, b, r;
    fp_copy(&a, ain);
    fp_copy(&b, bin);
    if (a.deg >= 0) fp_make_monic<CAP, MONT>(&a, M);
    if (b.deg >= 0) fp_make_monic<CAP, MONT>(&b, M);
    while (b.deg >= 0) {
        fp_rem_monic<CAP, MONT>(&r, &a, &b, M);
        fp_copy(&a, &b);
        fp_copy(&b, &r);
        if (b.deg >= 0) fp_make_monic<CAP, MONT>(&b, M);
    }
    fp_copy(out, &a);
}

template <int CAP>
struct fwide_t {
    uint32_t c[2 * CAP + 1];
};

template <int CAP>
__device__ __forceinline__ uint32_t fw_get(const fwide_t<CAP> *a, int idx)
{
    uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) if (idx == i) v = a->c[i];
    return v;
}

template <int CAP>
__device__ __forceinline__ void fw_set(fwide_t<CAP> *a, int idx, uint32_t v)
{
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) if (idx == i) a->c[i] = v;
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_mul_rem(fpoly_t<CAP> *out,
                                            const fpoly_t<CAP> *a,
                                            const fpoly_t<CAP> *b,
                                            const fpoly_t<CAP> *mod,
                                            const mont32_t *M)
{
    fwide_t<CAP> t;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) t.c[i] = 0;

    int tdeg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) {
        if (i <= a->deg) {
            #pragma unroll
            for (int j = 0; j <= CAP; j++) {
                if (j <= b->deg) {
                    const uint32_t z = d_ctx_mul<MONT>(a->c[i], b->c[j], M);
                    t.c[i + j] = d_mod_add(t.c[i + j], z, M->p);
                    if (t.c[i + j] && i + j > tdeg) tdeg = i + j;
                }
            }
        }
    }

    while (tdeg >= mod->deg) {
        const uint32_t q = fw_get(&t, tdeg);
        const int shift = tdeg - mod->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < mod->deg) {
                const int k = i + shift;
                const uint32_t tv = fw_get(&t, k);
                fw_set(&t, k, d_mod_sub(tv,
                                        d_ctx_mul<MONT>(q, mod->c[i], M), M->p));
            }
        }
        fw_set(&t, tdeg, 0);
        do { tdeg--; } while (tdeg >= 0 && fw_get(&t, tdeg) == 0);
    }

    fp_zero(out);
    out->deg = tdeg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (i <= tdeg) out->c[i] = t.c[i];
}

template <int CAP, bool MONT>
__device__ __forceinline__ void fp_sqr_rem(fpoly_t<CAP> *out,
                                            const fpoly_t<CAP> *a,
                                            const fpoly_t<CAP> *mod,
                                            const mont32_t *M)
{
    fwide_t<CAP> t;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) t.c[i] = 0;

    int tdeg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) {
        if (i <= a->deg) {
            uint32_t z = d_ctx_mul<MONT>(a->c[i], a->c[i], M);
            t.c[2 * i] = d_mod_add(t.c[2 * i], z, M->p);
            if (t.c[2 * i] && 2 * i > tdeg) tdeg = 2 * i;
            #pragma unroll
            for (int j = i + 1; j <= CAP; j++) {
                if (j <= a->deg) {
                    z = d_ctx_mul<MONT>(a->c[i], a->c[j], M);
                    z = d_mod_add(z, z, M->p);
                    t.c[i + j] = d_mod_add(t.c[i + j], z, M->p);
                    if (t.c[i + j] && i + j > tdeg) tdeg = i + j;
                }
            }
        }
    }

    while (tdeg >= mod->deg) {
        const uint32_t q = fw_get(&t, tdeg);
        const int shift = tdeg - mod->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < mod->deg) {
                const int k = i + shift;
                const uint32_t tv = fw_get(&t, k);
                fw_set(&t, k, d_mod_sub(tv,
                                        d_ctx_mul<MONT>(q, mod->c[i], M), M->p));
            }
        }
        fw_set(&t, tdeg, 0);
        do { tdeg--; } while (tdeg >= 0 && fw_get(&t, tdeg) == 0);
    }

    fp_zero(out);
    out->deg = tdeg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (i <= tdeg) out->c[i] = t.c[i];
}

template <int CAP, bool MONT, bool SYMMETRIC>
__device__ __forceinline__ void fp_pow_rem(fpoly_t<CAP> *out,
                                            const fpoly_t<CAP> *ain,
                                            uint32_t e,
                                            const fpoly_t<CAP> *mod,
                                            const mont32_t *M)
{
    fpoly_t<CAP> r, a, tmp;
    fp_zero(&r);
    r.deg = 0;
    r.c[0] = d_ctx_one<MONT>(M);
    fp_rem_monic<CAP, MONT>(&a, ain, mod, M);
    while (e) {
        if (e & 1u) {
            fp_mul_rem<CAP, MONT>(&tmp, &r, &a, mod, M);
            fp_copy(&r, &tmp);
        }
        e >>= 1;
        if (e) {
            if constexpr (SYMMETRIC) fp_sqr_rem<CAP, MONT>(&tmp, &a, mod, M);
            else fp_mul_rem<CAP, MONT>(&tmp, &a, &a, mod, M);
            fp_copy(&a, &tmp);
        }
    }
    fp_copy(out, &r);
}

template <int CAP, bool MONT>
__device__ __forceinline__ uint32_t fp_eval(const fpoly_t<CAP> *f, uint32_t x,
                                             const mont32_t *M)
{
    uint32_t r = 0;
    const uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 0; i--)
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), fp_get(f, i), M->p);
    return r;
}

template <int CAP, bool MONT>
__device__ __forceinline__ uint32_t fp_deriv_eval(const fpoly_t<CAP> *f,
                                                   uint32_t x,
                                                   const mont32_t *M)
{
    uint32_t r = 0;
    const uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 1; i--) {
        const uint32_t ii = d_ctx_from<MONT>((uint32_t)i, M);
        const uint32_t term = d_ctx_mul<MONT>(fp_get(f, i), ii, M);
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), term, M->p);
    }
    return r;
}

/* Find any proper factor of a squarefree product of linear factors. */
template <int CAP, bool MONT, bool SYMMETRIC>
__device__ __forceinline__ int fp_find_split(fpoly_t<CAP> *gout,
                                              const fpoly_t<CAP> *f,
                                              const mont32_t *M)
{
    for (uint32_t a = 0; a < M->p; a++) {
        fpoly_t<CAP> lin, h, g;
        fp_zero(&lin);
        lin.deg = 1;
        lin.c[0] = d_ctx_from<MONT>(a, M);
        lin.c[1] = d_ctx_one<MONT>(M);
        fp_pow_rem<CAP, MONT, SYMMETRIC>(&h, &lin, (M->p - 1u) / 2u, f, M);
        if (h.deg < 0) { h.deg = 0; h.c[0] = 0; }
        h.c[0] = d_mod_sub(h.c[0], d_ctx_one<MONT>(M), M->p);
        fp_norm(&h);
        fp_gcd<CAP, MONT>(&g, f, &h, M);
        if (g.deg > 0 && g.deg < f->deg) {
            fp_copy(gout, &g);
            return 1;
        }
    }
    return 0;
}

/* Stack-free root extraction for the fixed-capacity path.  `remaining` always
 * contains every not-yet-emitted root.  To get the next root, repeatedly take
 * one proper factor until a linear factor remains, emit it, and divide that
 * linear factor from `remaining`.  The common case has zero or one root, so it
 * pays no repeated-splitting penalty while avoiding CAP polynomial stack slots
 * for every thread. */
template <int CAP, bool MONT, bool SYMMETRIC>
__device__ __forceinline__ int fp_split_linear(fpoly_t<CAP> *input,
                                                const mont32_t *M,
                                                uint32_t *roots,
                                                uint32_t *failed)
{
    fpoly_t<CAP> remaining, target, g, lin, next;
    int nr = 0;
    if (input->deg <= 0) return 0;
    fp_copy(&remaining, input);
    fp_make_monic<CAP, MONT>(&remaining, M);

    while (remaining.deg > 0) {
        fp_copy(&target, &remaining);
        while (target.deg > 1) {
            if (!fp_find_split<CAP, MONT, SYMMETRIC>(&g, &target, M)) {
                *failed = 1;
                return nr;
            }
            fp_make_monic<CAP, MONT>(&g, M);
            fp_copy(&target, &g);
        }

        const uint32_t c0 = d_ctx_to<MONT>(target.c[0], M);
        const uint32_t root = c0 ? M->p - c0 : 0;
        roots[nr++] = root;
        if (nr > CAP) { *failed = 1; return 0; }

        fp_zero(&lin);
        lin.deg = 1;
        lin.c[0] = d_ctx_from<MONT>(root ? M->p - root : 0, M);
        lin.c[1] = d_ctx_one<MONT>(M);
        fp_divexact_monic<CAP, MONT>(&next, &remaining, &lin, M, failed);
        if (*failed) return nr;
        fp_copy(&remaining, &next);
        if (remaining.deg >= 0) fp_make_monic<CAP, MONT>(&remaining, M);
    }
    return nr;
}

template <int CAP, bool MONT, bool SYMMETRIC>
__device__ __forceinline__ int d_alg_roots_prime_fixed(uint32_t p,
                                                        uint32_t *roots,
                                                        uint32_t *failed,
                                                        uint8_t *special = nullptr)
{
    fpoly_t<CAP> f;
    mont32_t M;
    M.p = p;
    M.n0inv = M.one = M.r2 = 0;
    if constexpr (MONT) M = d_mont_setup(p);

    const uint32_t lead = d_ctx_big<MONT>(&c_alg[c_alg_deg], &M);
    int nr = 0;
    fp_zero(&f);
    f.deg = c_alg_deg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++)
        if (i <= c_alg_deg) f.c[i] = d_ctx_big<MONT>(&c_alg[i], &M);
    fp_norm(&f);

    if (f.deg < 0) {
        *failed = 1;
        return 0;
    }
    if (f.deg == 1) {
        const uint32_t c0 = d_ctx_to<MONT>(f.c[0], &M);
        const uint32_t c1 = d_ctx_to<MONT>(f.c[1], &M);
        const uint32_t inv = d_mod_inv(c1, p);
        const uint32_t z = d_ctx_mul<MONT>(
            d_ctx_from<MONT>(c0 ? p - c0 : 0, &M), d_ctx_from<MONT>(inv, &M), &M);
        roots[nr++] = d_ctx_to<MONT>(z, &M);
    } else if (p <= GPU_FB_BRUTE_ROOT_LIMIT) {
        for (uint32_t x = 0; x < p; x++)
            if (fp_eval<CAP, MONT>(&f, x, &M) == 0) roots[nr++] = x;
    } else {
        fpoly_t<CAP> x, xp, linear;
        fp_make_monic<CAP, MONT>(&f, &M);
        fp_zero(&x);
        x.deg = 1;
        x.c[1] = d_ctx_one<MONT>(&M);
        fp_pow_rem<CAP, MONT, SYMMETRIC>(&xp, &x, p, &f, &M);
        if (xp.deg < 1) xp.deg = 1;
        xp.c[1] = d_mod_sub(xp.c[1], d_ctx_one<MONT>(&M), p);
        fp_norm(&xp);
        fp_gcd<CAP, MONT>(&linear, &f, &xp, &M);
        nr = fp_split_linear<CAP, MONT, SYMMETRIC>(&linear, &M, roots, failed);
        if (*failed) return nr;
    }

    if (lead == 0) roots[nr++] = p;
    if (nr > GPU_FB_MAX_ROOTS) {
        *failed = 1;
        return 0;
    }
    if (special) {
        uint8_t sp = lead == 0;
        /* A repeated affine root is exactly the case where the q=p entry can
         * carry a valuation increment other than (1,0).  Projective roots are
         * exceptional for the same reason.  Mark only those primes; the host
         * then reruns the native exact p-adic code for a tiny exception set. */
        for (int i = 0; i < nr && !sp; i++)
            if (roots[i] < p && fp_deriv_eval<CAP, MONT>(&f, roots[i], &M) == 0)
                sp = 1;
        *special = sp;
    }
    for (int i = 1; i < nr; i++) {
        const uint32_t v = roots[i];
        int j = i;
        while (j && roots[j - 1] > v) { roots[j] = roots[j - 1]; j--; }
        roots[j] = v;
    }
    return nr;
}

template <int CAP, bool MONT, bool SYMMETRIC>
__global__ void k_alg_roots_fixed(const uint32_t *primes, uint32_t n,
                                  uint32_t *rootbuf, uint32_t *counts,
                                  uint32_t *failures)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        const uint32_t p = primes[t];
        const int nr = (MONT && p == 2u)
            ? d_alg_roots_prime_fixed<CAP, false, SYMMETRIC>(p, roots, &bad)
            : d_alg_roots_prime_fixed<CAP, MONT, SYMMETRIC>(p, roots, &bad);
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = i < nr ? roots[i] : 0u;
    }
}

template <int CAP, bool MONT>
__global__ void k_alg_roots_fixed_mark(const uint32_t *primes, uint32_t n,
                                       uint32_t *rootbuf, uint32_t *counts,
                                       uint8_t *special, uint32_t *failures)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        uint8_t sp = 0;
        const uint32_t p = primes[t];
        const int nr = (MONT && p == 2u)
            ? d_alg_roots_prime_fixed<CAP, false, false>(p, roots, &bad, &sp)
            : d_alg_roots_prime_fixed<CAP, MONT, false>(p, roots, &bad, &sp);
        /* A q=p record is trusted by FB_VALIDATE_GENERATED_PRIME_POWERS, so do
         * not let the same GPU sieve that proposed p be its only primality
         * proof.  Check only candidates that would actually emit an ideal. */
        if (!bad && nr && !d_is_prime32_independent(p)) bad = 1u;
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        special[t] = bad ? 1u : sp;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = (!bad && i < nr) ? roots[i] : 0u;
    }
}

__global__ void k_make_odds(uint32_t lo, uint32_t n, uint32_t *values)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) values[t] = (uint32_t)((uint64_t)lo + 2ull * t);
}

__global__ void k_mark_composites(uint32_t lo, uint32_t n,
                                  const uint32_t *base, uint32_t nbase,
                                  uint8_t *isprime)
{
    /* One block per base prime.  A one-thread-per-prime sieve makes p=3 one
     * serial thread write roughly a third of the entire segment; spreading a
     * prime's progression across the block is dramatically less pathological. */
    uint32_t k = blockIdx.x;
    if (k >= nbase) return;
    uint32_t p = base[k];
    if (p == 2) return;
    uint64_t hi = (uint64_t)lo + 2ull * (n - 1u);
    uint64_t first = (uint64_t)p * p;
    if (first < lo) first = ((uint64_t)lo + p - 1u) / p * p;
    if (!(first & 1u)) first += p;
    if (first > hi) return;
    uint64_t idx = ((first - lo) >> 1) + (uint64_t)threadIdx.x * p;
    uint64_t step = (uint64_t)p * blockDim.x;
    for (; idx < n; idx += step) isprime[idx] = 0;
}

template <bool MONT, bool SYMMETRIC>
__global__ void k_alg_roots(const uint32_t *primes, uint32_t n,
                            uint32_t *rootbuf, uint32_t *counts,
                            uint32_t *failures)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        const uint32_t p = primes[t];
        /* Montgomery arithmetic requires an odd modulus; p=2 is one entry and
         * stays on the legacy path.  The branch is uniform for all but the
         * first lane of the first segment. */
        int nr = (MONT && p == 2u) ? d_alg_roots_prime<false, SYMMETRIC>(p, roots, &bad)
                                   : d_alg_roots_prime<MONT, SYMMETRIC>(p, roots, &bad);
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = i < nr ? roots[i] : 0u;
    }
}

__global__ void k_rational_roots(const uint32_t *primes, uint32_t n,
                                 uint32_t *roots, uint32_t *failures)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t p = primes[t];
        uint32_t y0 = d_big_mod(&c_y0, p);
        uint32_t y1 = d_big_mod(&c_y1, p);
        if (!y1) {
            if (!y0) { atomicAdd(failures, 1u); roots[t] = 0; }
            else roots[t] = p;
        } else {
            uint32_t inv = d_mod_inv(y1, p);
            roots[t] = d_mod_mul(y0 ? p - y0 : 0, inv, p);
        }
    }
}

__global__ void k_total_roots(const uint32_t *counts, const uint32_t *offsets,
                              uint32_t n, uint32_t *total)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *total = n ? offsets[n - 1] + counts[n - 1] : 0;
}

__global__ void k_scatter_alg(const uint32_t *primes, uint32_t n,
                              const uint32_t *rootbuf, const uint32_t *counts,
                              const uint32_t *offsets,
                              uint32_t *out_p, uint32_t *out_r)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t off = offsets[t], nr = counts[t];
        for (uint32_t k = 0; k < nr; k++) {
            out_p[off + k] = primes[t];
            out_r[off + k] = rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + k];
        }
    }
}

__global__ void k_scatter_alg_mark(const uint32_t *primes, uint32_t n,
                                   const uint32_t *rootbuf, const uint32_t *counts,
                                   const uint32_t *offsets, const uint8_t *special,
                                   uint32_t *out_p, uint32_t *out_r,
                                   uint8_t *out_special)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t off = offsets[t], nr = counts[t];
        for (uint32_t k = 0; k < nr; k++) {
            out_p[off + k] = primes[t];
            out_r[off + k] = rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + k];
            out_special[off + k] = special[t];
        }
    }
}

__global__ void k_upper_bound(const uint32_t *v, uint32_t n, uint32_t lim,
                              uint32_t *out)
{
    if (blockIdx.x || threadIdx.x) return;
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (v[mid] <= lim) lo = mid + 1;
        else hi = mid;
    }
    *out = lo;
}

static double now_s(void)
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

#ifndef FBGEN_GPU_LIBRARY
typedef struct {
    cudaEvent_t a, b;
} evtimer_t;

static int timer_init(evtimer_t *T)
{
    cudaError_t e;
    T->a = T->b = NULL;
    e = cudaEventCreate(&T->a);
    if (e != cudaSuccess) return -1;
    e = cudaEventCreate(&T->b);
    if (e != cudaSuccess) { cudaEventDestroy(T->a); T->a = NULL; return -1; }
    return 0;
}

static float timer_finish(evtimer_t *T)
{
    float ms = 0;
    cudaEventRecord(T->b);
    cudaEventSynchronize(T->b);
    cudaEventElapsedTime(&ms, T->a, T->b);
    return ms;
}
#endif

/* Host representation used while assembling the complete generated factor
 * base.  Ordinary q=p entries are already sorted by the segmented prime
 * stream, so keep them in three SoA arrays.  Proper powers are tiny and are
 * merged backward at the end, avoiding a second full-size factor-base copy. */
typedef struct {
    uint32_t *q, *r;
    uint8_t *logp;
    size_t n, cap;
} afb_base_vec_t;

typedef struct {
    uint32_t q, r;
    uint8_t logp;
} afb_power_entry_t;

static void afb_base_free(afb_base_vec_t *v)
{
    if (!v) return;
    free(v->q); free(v->r); free(v->logp);
    memset(v, 0, sizeof(*v));
}

static int afb_base_reserve(afb_base_vec_t *v, size_t need)
{
    size_t cap;
    uint32_t *q, *r;
    uint8_t *lp;
    if (need <= v->cap) return 0;
    cap = v->cap ? v->cap : (1u << 20);
    while (cap < need) {
        size_t grow = cap / 4u;            /* 25%, not 2x at multi-GB sizes */
        if (grow < (1u << 20)) grow = (1u << 20);
        if (cap > SIZE_MAX - grow) { cap = need; break; }
        cap += grow;
    }
    if (cap < need || cap > SIZE_MAX / sizeof(*v->q)) return -1;
    q = (uint32_t *)realloc(v->q, cap * sizeof(*v->q));
    if (!q) return -1;
    v->q = q;
    r = (uint32_t *)realloc(v->r, cap * sizeof(*v->r));
    if (!r) return -1;
    v->r = r;
    lp = (uint8_t *)realloc(v->logp, cap * sizeof(*v->logp));
    if (!lp) return -1;
    v->logp = lp;
    v->cap = cap;
    return 0;
}

static int afb_base_push(afb_base_vec_t *v, uint32_t q, uint32_t r, uint8_t logp)
{
    if (v->n == UINT32_MAX || afb_base_reserve(v, v->n + 1u)) return -1;
    v->q[v->n] = q;
    v->r[v->n] = r;
    v->logp[v->n] = logp;
    v->n++;
    return 0;
}

static int afb_append_exact(const poly_t *P, uint32_t p, int maxbits,
                            double scale, afb_base_vec_t *base,
                            std::vector<afb_power_entry_t> *power)
{
    fbgen_exact_entry_t *e = NULL;
    size_t n = 0;
    if (fbgen_exact_prime_entries(P, p, maxbits, &e, &n)) return -1;
    for (size_t i = 0; i < n; i++) {
        uint8_t lp;
        if (fb_log_delta_checked(p, e[i].n1, e[i].n0, scale, &lp)) {
            free(e);
            return -1;
        }
        if (e[i].q == p) {
            if (afb_base_push(base, e[i].q, e[i].r, lp)) {
                free(e);
                return -1;
            }
        } else {
            try { power->push_back({e[i].q, e[i].r, lp}); }
            catch (const std::bad_alloc &) {
                free(e);
                errno = ENOMEM;
                return -1;
            }
        }
    }
    free(e);
    return 0;
}

static int afb_finish_merge(afb_base_vec_t *base,
                            std::vector<afb_power_entry_t> *power,
                            int maxbits, fb_t *fb)
{
    const size_t nbase = base->n;
    const size_t npow = power->size();
    const size_t ntotal = nbase + npow;
    if (!fb || ntotal == 0 || ntotal > UINT32_MAX) return -1;

    try {
        std::stable_sort(power->begin(), power->end(),
            [](const afb_power_entry_t &a, const afb_power_entry_t &b) {
                return a.q < b.q;
            });
    } catch (const std::bad_alloc &) {
        errno = ENOMEM;
        return -1;
    }
    if (afb_base_reserve(base, ntotal)) return -1;
    uint8_t *ispow = (uint8_t *)malloc(ntotal ? ntotal : 1u);
    if (!ispow) return -1;

    size_t i = nbase, j = npow, k = ntotal;
    while (k) {
        if (j && (!i || (*power)[j - 1].q >= base->q[i - 1])) {
            const afb_power_entry_t &e = (*power)[--j];
            --k;
            base->q[k] = e.q;
            base->r[k] = e.r;
            base->logp[k] = e.logp;
            ispow[k] = 1;
        } else {
            --i; --k;
            base->q[k] = base->q[i];
            base->r[k] = base->r[i];
            base->logp[k] = base->logp[i];
            ispow[k] = 0;
        }
    }

    memset(fb, 0, sizeof(*fb));
    fb->n = (uint32_t)ntotal;
    fb->primes = base->q;
    fb->roots = base->r;
    fb->logp = base->logp;
    fb->ispow = ispow;
    fb->maxbits = maxbits;
    base->q = base->r = NULL;
    base->logp = NULL;
    base->n = base->cap = 0;
    if (fb_validate(fb, FB_VALIDATE_GENERATED_PRIME_POWERS,
                    "afb_build_gpu") != 0) {
        fb_free(fb);
        return -1;
    }
    return 0;
}

typedef struct {
    uint64_t ordinary_roots;
    uint64_t exact_primes;
} gpu_fb_generate_stats_t;

template <typename T>
static cudaError_t gpu_fb_free(T **p)
{
    T *q;
    if (!p || !*p) return cudaSuccess;
    q = *p;
    *p = NULL;
    return cudaFree(q);
}

/* In-memory consumer for production bench.  All CUDA prime/root generation is
 * shared with the standalone writer below; this consumer only decides how an
 * already-generated prime is represented in fb_t. */
struct afb_memory_sink_t {
    const poly_t *P;
    int maxbits;
    double scale;
    afb_base_vec_t *base;
    std::vector<afb_power_entry_t> *power;

    int begin(void) { return 0; }

    int exact(uint32_t p)
    {
        if (afb_append_exact(P, p, maxbits, scale, base, power)) {
            fprintf(stderr, "afb_build_gpu: exact branch generation failed for p=%u\n", p);
            return -1;
        }
        return 0;
    }

    int ordinary(uint32_t p, const uint32_t *roots, size_t n)
    {
        uint8_t lp;
        if (fb_log_delta_checked(p, 1, 0, scale, &lp)) {
            fprintf(stderr, "afb_build_gpu: p=%u log increment is not representable\n", p);
            return -1;
        }
        for (size_t i = 0; i < n; i++) {
            if (afb_base_push(base, p, roots[i], lp)) {
                fprintf(stderr, "afb_build_gpu: out of host memory\n");
                return -1;
            }
        }
        return 0;
    }

    int checkpoint(void) { return 0; }
    int finish(void) { return 0; }
};

/* Single source of truth for complete GPU factor-base generation.
 *
 * This owns every correctness-critical step common to production in-memory
 * generation and standalone --out generation: device selection, coefficient
 * upload, segmented prime sieve, deterministic primality gate, CAP=6/CAP=8
 * algebraic roots, count scan, scatter, exact-prime classification, progress,
 * and all CUDA/host temporary ownership.  Sink methods are host-only and are
 * inlined by NVCC; FBGEN_GPU_LIBRARY instantiates only afb_memory_sink_t, so
 * production bench contains no serializer path or per-prime indirect call.
 *
 * Sink contract:
 *   begin()                    once, after GPU setup and before p=2
 *   exact(p)                   exact/Hensel representation for one prime
 *   ordinary(p, roots, n)      simple q=p roots already sorted for that prime
 *   checkpoint()               once per nonempty generated segment
 *   finish()                   once after final cudaDeviceSynchronize()
 */
template <typename Sink>
static int gpu_fb_generate_complete(const poly_t *P, uint32_t lim, int maxbits,
                                    int device, uint32_t segment_odds,
                                    int verbose, const char *who, Sink *sink,
                                    gpu_fb_generate_stats_t *stats)
{
    gpu_big_t h_alg[BENCH_NCOEFF];
    uint32_t *h_base = NULL;
    size_t nbase_sz = 0;
    uint32_t nbase = 0, sqrtlim;
    uint32_t *d_base = NULL, *d_values = NULL, *d_primes = NULL;
    uint8_t *d_flags = NULL, *d_special = NULL;
    uint32_t *d_nprime = NULL, *d_counts = NULL, *d_offsets = NULL;
    uint32_t *d_failures = NULL, *d_total = NULL;
    uint32_t *d_rootbuf = NULL, *d_out_p = NULL, *d_out_r = NULL;
    uint8_t *d_out_special = NULL;
    void *d_temp = NULL;
    size_t select_temp = 0, scan_temp = 0, temp_bytes = 0;
    cudaDeviceProp prop;
    std::vector<uint32_t> hp, hr;
    std::vector<uint8_t> hs;
    uint64_t ordinary_seen = 0, exact_primes = 0;
    uint32_t next_progress = 10;
    uint64_t pow_bound;
    uint32_t power_prime_max;
    int dev = device;
    int rc = -1;

    if (stats) memset(stats, 0, sizeof(*stats));
    if (!P || !sink || lim < 2 || maxbits < 1 || maxbits > 31 ||
        P->deg < 1 || P->deg > BENCH_MAX_DEGREE) {
        fprintf(stderr, "%s: invalid polynomial/bound/maxbits\n", who);
        return -1;
    }
    if (gpu_fb_normalize_segment_odds(&segment_odds)) return -1;
    pow_bound = UINT64_C(1) << maxbits;

    memset(h_alg, 0, sizeof(h_alg));
    for (int k = 0; k <= P->deg; k++) {
        if (host_big_parse(&h_alg[k], P->cs[k][0] ? P->cs[k] : "0")) {
            fprintf(stderr, "%s: coefficient c%d exceeds fixed parser capacity\n", who, k);
            return -1;
        }
    }

    if (dev >= 0) {
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) {
            fprintf(stderr, "%s: cudaSetDevice(%d): %s\n",
                    who, dev, cudaGetErrorString(e));
            return -1;
        }
    } else if (cudaGetDevice(&dev) != cudaSuccess) {
        fprintf(stderr, "%s: cannot query current CUDA device\n", who);
        return -1;
    }
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        fprintf(stderr, "%s: cannot query CUDA device %d\n", who, dev);
        return -1;
    }
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_alg, h_alg, sizeof(h_alg)));
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_alg_deg, &P->deg, sizeof(P->deg)));

    sqrtlim = (uint32_t)floor(sqrt((double)lim));
    while ((uint64_t)(sqrtlim + 1u) * (sqrtlim + 1u) <= lim) sqrtlim++;
    while ((uint64_t)sqrtlim * sqrtlim > lim) sqrtlim--;
    h_base = prime_list_build(sqrtlim, &nbase_sz);
    if (!h_base || nbase_sz > UINT32_MAX) {
        fprintf(stderr, "%s: base-prime sieve failed\n", who);
        goto fail;
    }
    nbase = (uint32_t)nbase_sz;

    power_prime_max = (uint32_t)floor(sqrt((double)pow_bound));
    while ((uint64_t)(power_prime_max + 1u) * (power_prime_max + 1u) <= pow_bound)
        power_prime_max++;
    while ((uint64_t)power_prime_max * power_prime_max > pow_bound)
        power_prime_max--;

    if (nbase) {
        CUDA_OR_DIE(cudaMalloc((void **)&d_base, (size_t)nbase * sizeof(*d_base)));
        CUDA_OR_DIE(cudaMemcpy(d_base, h_base, (size_t)nbase * sizeof(*d_base),
                               cudaMemcpyHostToDevice));
    }
    CUDA_OR_DIE(cudaMalloc((void **)&d_values, (size_t)segment_odds * sizeof(*d_values)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_primes, (size_t)segment_odds * sizeof(*d_primes)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_flags, (size_t)segment_odds));
    CUDA_OR_DIE(cudaMalloc((void **)&d_special, (size_t)segment_odds));
    CUDA_OR_DIE(cudaMalloc((void **)&d_nprime, sizeof(*d_nprime)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_counts, (size_t)segment_odds * sizeof(*d_counts)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_offsets, (size_t)segment_odds * sizeof(*d_offsets)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_failures, sizeof(*d_failures)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_total, sizeof(*d_total)));
    CUDA_OR_DIE(cub::DeviceSelect::Flagged(NULL, select_temp, d_values, d_flags,
                                           d_primes, d_nprime, segment_odds));
    CUDA_OR_DIE(cub::DeviceScan::ExclusiveSum(NULL, scan_temp, d_counts, d_offsets,
                                              segment_odds));
    temp_bytes = std::max(select_temp, scan_temp);
    CUDA_OR_DIE(cudaMalloc(&d_temp, temp_bytes));

    if (sink->begin()) goto fail;

    /* p=2 is always exact.  Besides powers, this preserves the native
     * ramified/projective metadata without adding an even-modulus GPU path. */
    if (sink->exact(2u)) goto fail;
    exact_primes++;

    for (uint64_t lo64 = 3; lo64 <= lim; ) {
        uint64_t hi64 = lo64 + 2ull * (segment_odds - 1u);
        if (hi64 > lim) hi64 = lim;
        if (!(hi64 & 1u)) hi64--;
        if (hi64 < lo64) break;
        const uint32_t lo = (uint32_t)lo64;
        const uint32_t n = (uint32_t)(((hi64 - lo64) >> 1) + 1u);
        uint32_t nprime = 0, nroots = 0, failures = 0;
        uint32_t blocks = std::min<uint32_t>((n + 255u) / 256u,
                                             (uint32_t)prop.multiProcessorCount * 8u);
        if (!blocks) blocks = 1;

        k_make_odds<<<blocks, 256>>>(lo, n, d_values);
        CUDA_OR_DIE(cudaGetLastError());
        CUDA_OR_DIE(cudaMemset(d_flags, 1, n));
        if (nbase) {
            k_mark_composites<<<nbase, 128>>>(lo, n, d_base, nbase, d_flags);
            CUDA_OR_DIE(cudaGetLastError());
        }
        CUDA_OR_DIE(cub::DeviceSelect::Flagged(d_temp, select_temp, d_values,
                                               d_flags, d_primes, d_nprime, n));
        CUDA_OR_DIE(cudaMemcpy(&nprime, d_nprime, sizeof(nprime), cudaMemcpyDeviceToHost));
        if (!nprime) { lo64 = hi64 + 2u; continue; }

        const size_t slots = (size_t)nprime * GPU_FB_MAX_ROOTS;
        CUDA_OR_DIE(cudaMalloc((void **)&d_rootbuf, slots * sizeof(*d_rootbuf)));
        CUDA_OR_DIE(cudaMemset(d_failures, 0, sizeof(*d_failures)));
        {
            const uint32_t ablocks = std::min<uint32_t>((nprime + 127u) / 128u,
                                                        (uint32_t)prop.multiProcessorCount * 8u);
            if (P->deg <= 6)
                k_alg_roots_fixed_mark<6, true><<<ablocks, 128>>>(
                    d_primes, nprime, d_rootbuf, d_counts, d_special, d_failures);
            else
                k_alg_roots_fixed_mark<8, true><<<ablocks, 128>>>(
                    d_primes, nprime, d_rootbuf, d_counts, d_special, d_failures);
        }
        CUDA_OR_DIE(cudaGetLastError());
        CUDA_OR_DIE(cudaMemcpy(&failures, d_failures, sizeof(failures),
                               cudaMemcpyDeviceToHost));
        if (failures) {
            fprintf(stderr,
                    "%s: algebraic root/primality validation failed for %u candidates in [%u,%u]\n",
                    who, failures, lo, (uint32_t)hi64);
            goto fail;
        }

        CUDA_OR_DIE(cub::DeviceScan::ExclusiveSum(d_temp, scan_temp, d_counts,
                                                   d_offsets, nprime));
        k_total_roots<<<1,1>>>(d_counts, d_offsets, nprime, d_total);
        CUDA_OR_DIE(cudaGetLastError());
        CUDA_OR_DIE(cudaMemcpy(&nroots, d_total, sizeof(nroots), cudaMemcpyDeviceToHost));
        ordinary_seen += nroots;

        if (nroots) {
            if (vector_resize_nothrow(&hp, nroots) ||
                vector_resize_nothrow(&hr, nroots) ||
                vector_resize_nothrow(&hs, nroots)) {
                fprintf(stderr, "%s: out of host memory for %u roots\n", who, nroots);
                goto fail;
            }
            CUDA_OR_DIE(cudaMalloc((void **)&d_out_p, (size_t)nroots * sizeof(*d_out_p)));
            CUDA_OR_DIE(cudaMalloc((void **)&d_out_r, (size_t)nroots * sizeof(*d_out_r)));
            CUDA_OR_DIE(cudaMalloc((void **)&d_out_special, (size_t)nroots));
            k_scatter_alg_mark<<<std::min<uint32_t>((nprime + 255u) / 256u,
                                                    (uint32_t)prop.multiProcessorCount * 8u), 256>>>(
                d_primes, nprime, d_rootbuf, d_counts, d_offsets, d_special,
                d_out_p, d_out_r, d_out_special);
            CUDA_OR_DIE(cudaGetLastError());
            CUDA_OR_DIE(cudaMemcpy(hp.data(), d_out_p, (size_t)nroots * sizeof(*d_out_p),
                                   cudaMemcpyDeviceToHost));
            CUDA_OR_DIE(cudaMemcpy(hr.data(), d_out_r, (size_t)nroots * sizeof(*d_out_r),
                                   cudaMemcpyDeviceToHost));
            CUDA_OR_DIE(cudaMemcpy(hs.data(), d_out_special, (size_t)nroots,
                                   cudaMemcpyDeviceToHost));

            for (uint32_t a = 0; a < nroots; ) {
                const uint32_t p = hp[a];
                uint32_t b = a + 1u;
                while (b < nroots && hp[b] == p) b++;
                if (p <= power_prime_max || hs[a]) {
                    if (sink->exact(p)) goto fail;
                    exact_primes++;
                } else if (sink->ordinary(p, hr.data() + a, (size_t)(b - a))) {
                    goto fail;
                }
                a = b;
            }
        }

        CUDA_OR_DIE(gpu_fb_free(&d_out_special));
        CUDA_OR_DIE(gpu_fb_free(&d_out_r));
        CUDA_OR_DIE(gpu_fb_free(&d_out_p));
        CUDA_OR_DIE(gpu_fb_free(&d_rootbuf));

        if (sink->checkpoint()) goto fail;
        if (verbose) {
            uint32_t pct = (uint32_t)((hi64 * 100ull) / lim);
            if (pct >= next_progress) {
                fprintf(stderr, "%s: %u%% through prime range\n", who, pct);
                while (next_progress <= pct && next_progress <= 100) next_progress += 10;
            }
        }
        lo64 = hi64 + 2u;
    }

    CUDA_OR_DIE(cudaDeviceSynchronize());
    if (sink->finish()) goto fail;
    if (stats) {
        stats->ordinary_roots = ordinary_seen;
        stats->exact_primes = exact_primes;
    }
    rc = 0;

fail:
    (void)gpu_fb_free(&d_out_special);
    (void)gpu_fb_free(&d_out_r);
    (void)gpu_fb_free(&d_out_p);
    (void)gpu_fb_free(&d_rootbuf);
    (void)gpu_fb_free(&d_temp);
    (void)gpu_fb_free(&d_total);
    (void)gpu_fb_free(&d_failures);
    (void)gpu_fb_free(&d_offsets);
    (void)gpu_fb_free(&d_counts);
    (void)gpu_fb_free(&d_nprime);
    (void)gpu_fb_free(&d_special);
    (void)gpu_fb_free(&d_flags);
    (void)gpu_fb_free(&d_primes);
    (void)gpu_fb_free(&d_values);
    (void)gpu_fb_free(&d_base);
    free(h_base);
    return rc;
}

extern "C" int afb_build_gpu(const poly_t *P, uint32_t lim, int maxbits,
                              double scale, int device, uint32_t segment_odds,
                              int verbose, fb_t *fb)
{
    afb_base_vec_t base = {};
    std::vector<afb_power_entry_t> power;
    gpu_fb_generate_stats_t stats = {};
    afb_memory_sink_t sink;
    double t0 = now_s();
    int rc = -1;

    if (!P || !fb || lim < 2 || maxbits < 1 || maxbits > 31 ||
        !std::isfinite(scale) || scale <= 0.0 || P->deg < 1 ||
        P->deg > BENCH_MAX_DEGREE) {
        fprintf(stderr, "afb_build_gpu: invalid polynomial/bound/scale/maxbits\n");
        return -1;
    }
    memset(fb, 0, sizeof(*fb));

    /* For an irreducible polynomial the average number of roots per prime is
     * one. Reserve a small margin to avoid reallocating multi-GB arrays in the
     * normal case; reducible/unusual jobs still grow safely through reserve(). */
    if (lim >= 100u) {
        double est = (double)lim / (log((double)lim) - 1.1) * 1.10;
        if (est > 0.0 && est < (double)UINT32_MAX)
            (void)afb_base_reserve(&base, (size_t)est);
    }

    sink.P = P;
    sink.maxbits = maxbits;
    sink.scale = scale;
    sink.base = &base;
    sink.power = &power;
    if (gpu_fb_generate_complete(P, lim, maxbits, device, segment_odds,
                                 verbose, "afb_build_gpu", &sink, &stats))
        goto fail;

    if (afb_finish_merge(&base, &power, maxbits, fb)) {
        fprintf(stderr, "afb_build_gpu: final merge/validation failed\n");
        goto fail;
    }
    if (verbose) {
        uint32_t npow = 0;
        for (uint32_t i = 0; i < fb->n; i++) npow += !!fb->ispow[i];
        fprintf(stderr,
                "afb_build_gpu: %u ideals through %u (%u prime-power); "
                "%llu ordinary GPU roots, %llu exact primes; %.3f s wall\n",
                fb->n, lim, npow, (unsigned long long)stats.ordinary_roots,
                (unsigned long long)stats.exact_primes, now_s() - t0);
    }
    rc = 0;

fail:
    afb_base_free(&base);
    if (rc) fb_free(fb);
    return rc;
}


#ifndef FBGEN_GPU_LIBRARY
/* Standalone cached-file path.  Serialization remains compiled out of
 * fbgen_gpu_lib.o.  The GPU generation loop itself is shared above so the
 * production and --out paths cannot drift in sieve/root/scan semantics. */
static int afb_write_exact_prime(FILE *out, const poly_t *P, uint32_t p,
                                 int maxbits, uint64_t *nentry)
{
    fbgen_exact_entry_t *e = NULL;
    size_t n = 0;
    if (fbgen_exact_prime_entries(P, p, maxbits, &e, &n)) return -1;
    if (fbgen_write_prime_entries(out, e, n)) {
        free(e);
        return -1;
    }
    if (nentry) *nentry += n;
    free(e);
    return 0;
}

static int afb_write_ordinary_prime(FILE *out, uint32_t p,
                                    const uint32_t *roots, size_t n,
                                    uint64_t *nentry)
{
    fbgen_exact_entry_t e[GPU_FB_MAX_ROOTS];
    if (n > GPU_FB_MAX_ROOTS) return -1;
    for (size_t i = 0; i < n; i++) {
        e[i].q = p;
        e[i].r = roots[i];
        e[i].n1 = 1;
        e[i].n0 = 0;
    }
    if (fbgen_write_prime_entries(out, e, n)) return -1;
    if (nentry) *nentry += n;
    return 0;
}

struct afb_stream_sink_t {
    FILE *out;
    const poly_t *P;
    uint32_t lim;
    int maxbits;
    uint64_t entries;

    int begin(void)
    {
        if (fbgen_write_header(out, P, lim, maxbits)) {
            fprintf(stderr, "fbgen_gpu: cannot write factor-base header: %s\n",
                    strerror(errno ? errno : EIO));
            return -1;
        }
        return 0;
    }

    int exact(uint32_t p)
    {
        if (afb_write_exact_prime(out, P, p, maxbits, &entries)) {
            fprintf(stderr, "fbgen_gpu: exact branch generation/write failed for p=%u\n", p);
            return -1;
        }
        return 0;
    }

    int ordinary(uint32_t p, const uint32_t *roots, size_t n)
    {
        if (afb_write_ordinary_prime(out, p, roots, n, &entries)) {
            fprintf(stderr, "fbgen_gpu: ordinary entry write failed for p=%u\n", p);
            return -1;
        }
        return 0;
    }

    int checkpoint(void)
    {
        if (ferror(out)) {
            fprintf(stderr, "fbgen_gpu: factor-base output write failed: %s\n",
                    strerror(errno ? errno : EIO));
            return -1;
        }
        return 0;
    }

    int finish(void)
    {
        if (fflush(out)) {
            fprintf(stderr, "fbgen_gpu: factor-base output flush failed: %s\n",
                    strerror(errno));
            return -1;
        }
        return 0;
    }
};

static int afb_write_gpu_stream(FILE *out, const poly_t *P, uint32_t lim,
                                int maxbits, int device, uint32_t segment_odds,
                                int verbose, uint64_t *nentry_out)
{
    afb_stream_sink_t sink;
    gpu_fb_generate_stats_t stats = {};
    double t0 = now_s();

    if (!out || !P || lim < 2 || maxbits < 1 || maxbits > 31 ||
        P->deg < 1 || P->deg > BENCH_MAX_DEGREE) {
        fprintf(stderr, "fbgen_gpu: invalid polynomial/bound/maxbits for --out\n");
        return -1;
    }
    sink.out = out;
    sink.P = P;
    sink.lim = lim;
    sink.maxbits = maxbits;
    sink.entries = 0;

    if (gpu_fb_generate_complete(P, lim, maxbits, device, segment_odds,
                                 verbose, "fbgen_gpu", &sink, &stats))
        return -1;

    if (verbose)
        fprintf(stderr,
                "fbgen_gpu: wrote %llu exact-format entries through %u; "
                "%llu exact primes; %.3f s wall\n",
                (unsigned long long)sink.entries, lim,
                (unsigned long long)stats.exact_primes, now_s() - t0);
    if (nentry_out) *nentry_out = sink.entries;
    return 0;
}

static int files_byte_equal(const char *a_path, const char *b_path)
{
    unsigned char a[32768], b[32768];
    FILE *fa = fopen(a_path, "rb"), *fb = NULL;
    if (!fa) {
        fprintf(stderr, "fbgen_gpu: cannot open %s: %s\n", a_path, strerror(errno));
        return -1;
    }
    fb = fopen(b_path, "rb");
    if (!fb) {
        fprintf(stderr, "fbgen_gpu: cannot open %s: %s\n", b_path, strerror(errno));
        fclose(fa);
        return -1;
    }
    for (;;) {
        size_t na = fread(a, 1, sizeof(a), fa);
        size_t nb = fread(b, 1, sizeof(b), fb);
        if (na != nb || (na && memcmp(a, b, na))) {
            fclose(fb); fclose(fa);
            return 0;
        }
        if (na < sizeof(a)) {
            int bad = ferror(fa) || ferror(fb);
            fclose(fb); fclose(fa);
            return bad ? -1 : 1;
        }
    }
}

static int compare_fb_t(const fb_t *got, const fb_t *ref, const char *label)
{
    if (!got || !ref) return -1;
    if (got->n != ref->n) {
        fprintf(stderr, "fbgen_gpu: %s count mismatch: GPU %u, reference %u\n",
                label, got->n, ref->n);
        return -1;
    }
    for (uint32_t i = 0; i < got->n; i++) {
        if (got->primes[i] != ref->primes[i] || got->roots[i] != ref->roots[i] ||
            got->logp[i] != ref->logp[i] || got->ispow[i] != ref->ispow[i]) {
            fprintf(stderr,
                    "fbgen_gpu: %s mismatch at %u: "
                    "GPU=(%u,%u,log=%u,pow=%u) ref=(%u,%u,log=%u,pow=%u)\n",
                    label, i, got->primes[i], got->roots[i], (unsigned)got->logp[i],
                    (unsigned)got->ispow[i], ref->primes[i], ref->roots[i],
                    (unsigned)ref->logp[i], (unsigned)ref->ispow[i]);
            return -1;
        }
    }
    if (got->maxbits && ref->maxbits && got->maxbits != ref->maxbits) {
        fprintf(stderr,
                "fbgen_gpu: %s maxbits mismatch: GPU %d, reference %d\n",
                label, got->maxbits, ref->maxbits);
        return -1;
    }
    fprintf(stderr, "fbgen_gpu: %s comparison: PASS (%u entries)\n", label, got->n);
    return 0;
}

static int compare_loaded_fb_files(const char *got_path, const char *ref_path,
                                   double scale)
{
    fb_t got = {}, ref = {};
    int rc;
    if (fb_load_cado(got_path, scale, &got) != 0) return -1;
    if (fb_load_cado(ref_path, scale, &ref) != 0) {
        fb_free(&got);
        return -1;
    }
    rc = compare_fb_t(&got, &ref, "loaded fb_t");
    fb_free(&ref);
    fb_free(&got);
    return rc;
}

static void usage(FILE *f)
{
    fputs(
        "usage: fbgen_gpu --poly FILE [options]\n"
        "  --alim N          algebraic prime bound [job alim]\n"
        "  --rlim N          rational prime bound  [job rlim]\n"
        "  --lim N           set both bounds to N\n"
        "  --segment-odds N  odd candidates per GPU segment [8388608; max 477218588]\n"
        "  --check N         validate up to N generated prime entries on host [1000]\n"
        "  --alg-backend B   algebraic arithmetic: montgomery (default) or legacy\n"
        "  --alg-square S    polynomial square: generic (default) or symmetric\n"
        "  --alg-kernel K    root kernel: auto (default), generic, cap6, or cap8\n"
        "  --out FILE        write complete fbgen-compatible algebraic roots file;\n"
        "                    uses --alim/--lim, stages FILE.part, then atomically replaces FILE\n"
        "  --complete        build the complete algebraic fb_t in memory, including powers\n"
        "  --maxbits N       power bound for --out/--complete [15]\n"
        "  --scale X         log-byte scale for loaded-fb comparisons [1.0]\n"
        "  --compare-fb F    compare against fbgen text (byte + loaded fb_t with --out)\n"
        "  --device N        CUDA device [0]\n"
        "\n"
        "Default benchmark mode times ORDINARY-PRIME entries only.\n"
        "--out uses the production hybrid generator and the native fbgen serializer;\n"
        "its file is intended to be byte-identical to native fbgen output and can\n"
        "be reused by bench --fb1 to avoid regenerating the FB at every startup.\n"
        "--complete uses the production hybrid builder: GPU ordinary roots plus\n"
        "the native exact CPU Hensel/ramified path for exceptional entries.\n"
        "Supported polynomial degree is 1..8 (CAP=6 through degree 6, CAP=8 for 7-8).\n",
        f);
}

int main(int argc, char **argv)
{
    const char *poly_path = NULL, *out_path = NULL;
    uint32_t alim = 0, rlim = 0, both_lim = 0;
    uint32_t seg_odds = GPU_FB_DEFAULT_SEG_ODDS;
    uint32_t check_left = 1000;
    int alg_legacy = 0;
    int alg_symmetric = 0;
    int alg_kernel = 0; /* 0 auto, 1 generic, 6 cap6, 8 cap8 */
    int complete = 0, complete_maxbits = 15;
    double complete_scale = 1.0;
    const char *compare_fb = NULL;
    int dev = 0;
    poly_t P;
    gpu_big_t h_alg[BENCH_NCOEFF], h_y0, h_y1;
    uint32_t *h_base = NULL;
    size_t nbase_sz = 0;
    uint32_t nbase = 0;
    uint32_t *d_base = NULL, *d_values = NULL, *d_primes = NULL;
    uint8_t *d_flags = NULL;
    uint32_t *d_nprime = NULL, *d_counts = NULL, *d_offsets = NULL;
    uint32_t *d_rroots = NULL, *d_failures = NULL, *d_total = NULL;
    uint32_t *d_rootbuf = NULL, *d_out_p = NULL, *d_out_r = NULL;
    void *d_temp = NULL;
    size_t select_temp = 0, scan_temp = 0, temp_bytes = 0;
    cudaDeviceProp prop;
    std::vector<uint32_t> check_p, check_c, check_roots, check_rroots;
    evtimer_t T = {};
    double t0, t_base0, t_base1, t_total;
    double ms_make = 0, ms_mark = 0, ms_select = 0;
    double ms_alg = 0, ms_scan = 0, ms_scatter = 0, ms_rat = 0;
    uint64_t prime_total = 0, alg_ideal_total = 0, rat_ideal_total = 0;
    uint64_t checked_alg = 0, checked_rat = 0;
    uint32_t maxlim, sqrtlim, next_progress = 10;
    int rc = 1;

    for (int i = 1; i < argc; i++) {
        uint64_t v;
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) poly_path = argv[++i];
        else if (!strcmp(argv[i], "--alim") && i + 1 < argc) {
            if (bench_parse_u64_decimal(argv[++i], &v) || v > UINT32_MAX) { usage(stderr); return 2; }
            alim = (uint32_t)v;
        } else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) {
            if (bench_parse_u64_decimal(argv[++i], &v) || v > UINT32_MAX) { usage(stderr); return 2; }
            rlim = (uint32_t)v;
        } else if (!strcmp(argv[i], "--lim") && i + 1 < argc) {
            if (bench_parse_u64_decimal(argv[++i], &v) || v > UINT32_MAX) { usage(stderr); return 2; }
            both_lim = (uint32_t)v;
        } else if (!strcmp(argv[i], "--segment-odds") && i + 1 < argc) {
            if (bench_parse_u64_decimal(argv[++i], &v) || !v || v > UINT32_MAX) { usage(stderr); return 2; }
            seg_odds = (uint32_t)v;
        } else if (!strcmp(argv[i], "--check") && i + 1 < argc) {
            if (bench_parse_u64_decimal(argv[++i], &v) || v > UINT32_MAX) { usage(stderr); return 2; }
            check_left = (uint32_t)v;
        } else if (!strcmp(argv[i], "--alg-backend") && i + 1 < argc) {
            const char *b = argv[++i];
            if (!strcmp(b, "montgomery") || !strcmp(b, "mont")) alg_legacy = 0;
            else if (!strcmp(b, "legacy")) alg_legacy = 1;
            else {
                fprintf(stderr, "fbgen_gpu: --alg-backend must be montgomery or legacy\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--alg-square") && i + 1 < argc) {
            const char *q = argv[++i];
            if (!strcmp(q, "symmetric") || !strcmp(q, "sym")) alg_symmetric = 1;
            else if (!strcmp(q, "generic") || !strcmp(q, "mul")) alg_symmetric = 0;
            else {
                fprintf(stderr, "fbgen_gpu: --alg-square must be symmetric or generic\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--alg-kernel") && i + 1 < argc) {
            const char *k = argv[++i];
            if (!strcmp(k, "auto")) alg_kernel = 0;
            else if (!strcmp(k, "generic")) alg_kernel = 1;
            else if (!strcmp(k, "cap6")) alg_kernel = 6;
            else if (!strcmp(k, "cap8")) alg_kernel = 8;
            else {
                fprintf(stderr, "fbgen_gpu: --alg-kernel must be auto, generic, cap6, or cap8\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--complete")) {
            complete = 1;
        } else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) {
            char *e = NULL;
            long x = strtol(argv[++i], &e, 10);
            if (!e || *e || x < 1 || x > 31) { usage(stderr); return 2; }
            complete_maxbits = (int)x;
        } else if (!strcmp(argv[i], "--scale") && i + 1 < argc) {
            char *e = NULL;
            complete_scale = strtod(argv[++i], &e);
            if (!e || *e || !std::isfinite(complete_scale) || complete_scale <= 0.0) {
                usage(stderr); return 2;
            }
        } else if (!strcmp(argv[i], "--compare-fb") && i + 1 < argc) {
            compare_fb = argv[++i];
            complete = 1;
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            char *e = NULL;
            long x = strtol(argv[++i], &e, 10);
            if (!e || *e || x < 0 || x > INT32_MAX) { usage(stderr); return 2; }
            dev = (int)x;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout); return 0;
        } else {
            fprintf(stderr, "fbgen_gpu: unknown or incomplete option: %s\n", argv[i]);
            usage(stderr); return 2;
        }
    }
    if (!poly_path) { usage(stderr); return 2; }
    if (poly_load(poly_path, &P)) return 2;
    if ((alg_kernel == 6 && P.deg > 6) || (alg_kernel == 8 && P.deg > 8)) {
        fprintf(stderr, "fbgen_gpu: polynomial degree %d exceeds forced cap%d kernel\n",
                P.deg, alg_kernel);
        return 2;
    }
    if (alg_symmetric && (alg_kernel == 6 || alg_kernel == 8)) {
        fprintf(stderr,
                "fbgen_gpu: fixed-capacity kernels currently use generic squaring; "
                "use --alg-kernel generic for --alg-square symmetric\n");
        return 2;
    }
    if (both_lim) alim = rlim = both_lim;
    if (!alim) alim = P.alim;
    if (!rlim) rlim = P.rlim;
    if (alim < 2 || ((!out_path && !complete) && rlim < 2)) {
        fprintf(stderr, out_path || complete
                ? "fbgen_gpu: alim must be >= 2 (job or CLI)\n"
                : "fbgen_gpu: alim and rlim must be >= 2 (job or CLI)\n");
        return 2;
    }

    if (gpu_fb_normalize_segment_odds(&seg_odds)) return 2;

    if (out_path) {
        FILE *out = stdout;
        char *stage_path = NULL;
        uint64_t nentry = 0;
        int wrc;
        if (compare_fb && !strcmp(out_path, "-")) {
            fprintf(stderr, "fbgen_gpu: --compare-fb with --out - is not supported\n");
            return 2;
        }
        if (compare_fb && !strcmp(out_path, compare_fb)) {
            fprintf(stderr, "fbgen_gpu: --out and --compare-fb must name different files\n");
            return 2;
        }
        if (strcmp(out_path, "-")) {
            size_t n = strlen(out_path) + sizeof(".part");
            stage_path = (char *)malloc(n);
            if (!stage_path) {
                fprintf(stderr, "fbgen_gpu: out of memory creating output path\n");
                return 1;
            }
            snprintf(stage_path, n, "%s.part", out_path);
            out = fopen(stage_path, "wb");
            if (!out) {
                fprintf(stderr, "fbgen_gpu: cannot open %s: %s\n",
                        stage_path, strerror(errno));
                free(stage_path);
                return 1;
            }
        }
        wrc = afb_write_gpu_stream(out, &P, alim, complete_maxbits, dev,
                                   seg_odds, 1, &nentry);
        if (out != stdout && fclose(out)) wrc = -1;
        if (stage_path) {
            /* Validate the staged bytes before publishing them.  Besides preserving
             * an existing good cache on mismatch, this makes path aliases safe:
             * --compare-fb can never be overwritten before it is consulted. */
            if (!wrc && compare_fb) {
                int same = files_byte_equal(stage_path, compare_fb);
                if (same < 0) wrc = -1;
                else if (!same) {
                    fprintf(stderr, "fbgen_gpu: byte comparison FAILED: %s != %s\n",
                            stage_path, compare_fb);
                    wrc = -1;
                } else {
                    fprintf(stderr, "fbgen_gpu: byte comparison: PASS (%s == %s)\n",
                            stage_path, compare_fb);
                    if (compare_loaded_fb_files(stage_path, compare_fb, complete_scale))
                        wrc = -1;
                }
            }
            if (!wrc && bench_atomic_replace(stage_path, out_path)) {
                fprintf(stderr, "fbgen_gpu: cannot replace %s with %s: %s\n",
                        out_path, stage_path, strerror(errno));
                wrc = -1;
            }
            if (wrc) remove(stage_path);
            free(stage_path);
        }
        if (wrc) return 1;
        if (strcmp(out_path, "-"))
            fprintf(stderr, "fbgen_gpu: complete factor base written to %s (%llu entries)\n",
                    out_path, (unsigned long long)nentry);
        return 0;
    }

    maxlim = std::max(alim, rlim);

    if (complete) {
        fb_t got = {}, ref = {};
        if (afb_build_gpu(&P, alim, complete_maxbits, complete_scale, dev,
                          seg_odds, 1, &got) != 0)
            return 1;
        uint32_t npow = 0;
        for (uint32_t i = 0; i < got.n; i++) npow += !!got.ispow[i];
        printf("complete algebraic factor base: %u ideals through %u "
               "(%u prime, %u prime-power), maxbits=%d, scale=%.6f\n",
               got.n, alim, got.n - npow, npow, got.maxbits, complete_scale);
        if (compare_fb) {
            if (fb_load_cado(compare_fb, complete_scale, &ref) != 0) {
                fb_free(&got);
                return 1;
            }
            if (compare_fb_t(&got, &ref, "complete fb_t")) {
                fb_free(&ref); fb_free(&got);
                return 1;
            }
            fb_free(&ref);
        }
        fb_free(&got);
        return 0;
    }

    memset(h_alg, 0, sizeof(h_alg));
    for (int k = 0; k <= P.deg; k++)
        if (host_big_parse(&h_alg[k], P.cs[k][0] ? P.cs[k] : "0")) {
            fprintf(stderr, "fbgen_gpu: coefficient c%d exceeds fixed parser capacity\n", k);
            return 2;
        }
    if (host_big_parse(&h_y0, P.y0s) || host_big_parse(&h_y1, P.y1s)) {
        fprintf(stderr, "fbgen_gpu: Y0/Y1 exceeds fixed parser capacity\n");
        return 2;
    }

    CUDA_OR_DIE(cudaSetDevice(dev));
    CUDA_OR_DIE(cudaGetDeviceProperties(&prop, dev));
    if (timer_init(&T)) { fprintf(stderr, "fbgen_gpu: cannot create CUDA events\n"); goto fail; }
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_alg, h_alg, sizeof(h_alg)));
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_alg_deg, &P.deg, sizeof(P.deg)));
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_y0, &h_y0, sizeof(h_y0)));
    CUDA_OR_DIE(cudaMemcpyToSymbol(c_y1, &h_y1, sizeof(h_y1)));

    t0 = now_s();
    sqrtlim = (uint32_t)floor(sqrt((double)maxlim));
    while ((uint64_t)(sqrtlim + 1u) * (sqrtlim + 1u) <= maxlim) sqrtlim++;
    while ((uint64_t)sqrtlim * sqrtlim > maxlim) sqrtlim--;
    t_base0 = now_s();
    h_base = prime_list_build(sqrtlim, &nbase_sz);
    t_base1 = now_s();
    if (!h_base || nbase_sz > UINT32_MAX) {
        fprintf(stderr, "fbgen_gpu: base-prime sieve failed\n"); goto fail;
    }
    nbase = (uint32_t)nbase_sz;

    if (nbase) {
        CUDA_OR_DIE(cudaMalloc((void **)&d_base, (size_t)nbase * sizeof(*d_base)));
        CUDA_OR_DIE(cudaMemcpy(d_base, h_base, (size_t)nbase * sizeof(*d_base), cudaMemcpyHostToDevice));
    }
    CUDA_OR_DIE(cudaMalloc((void **)&d_values, (size_t)seg_odds * sizeof(*d_values)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_primes, (size_t)seg_odds * sizeof(*d_primes)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_flags, (size_t)seg_odds));
    CUDA_OR_DIE(cudaMalloc((void **)&d_nprime, sizeof(*d_nprime)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_counts, (size_t)seg_odds * sizeof(*d_counts)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_offsets, (size_t)seg_odds * sizeof(*d_offsets)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_rroots, (size_t)seg_odds * sizeof(*d_rroots)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_failures, sizeof(*d_failures)));
    CUDA_OR_DIE(cudaMalloc((void **)&d_total, sizeof(*d_total)));

    CUDA_OR_DIE(cub::DeviceSelect::Flagged(NULL, select_temp, d_values, d_flags,
                                           d_primes, d_nprime, seg_odds));
    CUDA_OR_DIE(cub::DeviceScan::ExclusiveSum(NULL, scan_temp, d_counts, d_offsets,
                                              seg_odds));
    temp_bytes = std::max(select_temp, scan_temp);
    CUDA_OR_DIE(cudaMalloc(&d_temp, temp_bytes));

    printf("=== GPU factor-base generation benchmark ===\n");
    printf("device: %s, %d SMs, %.1f GB global memory\n",
           prop.name, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / 1.0e9);
    printf("polynomial: %s, degree %d\n", poly_path, P.deg);
    printf("bounds: algebraic %u, rational %u; max %u\n", alim, rlim, maxlim);
    printf("segment: %u odd candidates (%.1f MiB byte flags)\n",
           seg_odds, (double)seg_odds / (1024.0 * 1024.0));
    printf("base primes: %u through sqrt(maxlim)=%u (CPU setup %.3f ms)\n",
           nbase, sqrtlim, (t_base1 - t_base0) * 1000.0);
    printf("note: ordinary-prime entries only; p^k/Hensel entries are not included\n");
    printf("algebraic arithmetic: %s\n",
           alg_legacy ? "legacy uint64 remainder" : "32-bit Montgomery (division-free hot loop)");
    printf("algebraic squaring: %s\n",
           alg_symmetric ? "symmetric triangular product" : "generic multiply");
    {
        const int selected = alg_kernel ? alg_kernel
                                        : (alg_symmetric ? 1 :
                                           (P.deg <= 6 ? 6 : (P.deg <= 8 ? 8 : 1)));
        printf("algebraic kernel: %s\n\n",
               selected == 6 ? "fixed-capacity cap6 (stack-free splitter)" :
               selected == 8 ? "fixed-capacity cap8 (stack-free splitter)" :
                               "generic reference/fallback");
    }

    /* Account for p=2 separately; the segmented GPU sieve is odd-only. */
    if (maxlim >= 2) prime_total = 1;
    if (rlim >= 2) {
        uint32_t y0 = host_dec_mod(P.y0s, 2), y1 = host_dec_mod(P.y1s, 2);
        if (!y0 && !y1) {
            fprintf(stderr, "fbgen_gpu: rational polynomial vanishes identically mod 2\n");
            goto fail;
        }
        rat_ideal_total++;
    }
    if (alim >= 2) {
        int n2 = 0, any2 = 0;
        for (int k = 0; k <= P.deg; k++) any2 |= host_dec_mod(P.cs[k], 2) != 0;
        if (!any2) {
            fprintf(stderr, "fbgen_gpu: algebraic polynomial vanishes identically mod 2\n");
            goto fail;
        }
        for (uint32_t x = 0; x < 2; x++) if (!host_poly_eval_mod(&P, x, 2)) n2++;
        if (!host_dec_mod(P.cs[P.deg], 2)) n2++;  /* projective */
        alg_ideal_total += (uint64_t)n2;
    }

    for (uint64_t lo64 = 3; lo64 <= maxlim; ) {
        uint64_t hi64 = lo64 + 2ull * (seg_odds - 1u);
        if (hi64 > maxlim) hi64 = maxlim;
        if (!(hi64 & 1u)) hi64--;
        if (hi64 < lo64) break;
        uint32_t lo = (uint32_t)lo64;
        uint32_t n = (uint32_t)(((hi64 - lo64) >> 1) + 1u);
        uint32_t nprime = 0, nalg = 0, nrat = 0, nroots = 0, failures = 0;
        uint32_t blocks = std::min<uint32_t>((n + 255u) / 256u,
                                             (uint32_t)prop.multiProcessorCount * 8u);
        if (!blocks) blocks = 1;

        cudaEventRecord(T.a);
        k_make_odds<<<blocks, 256>>>(lo, n, d_values);
        CUDA_OR_DIE(cudaGetLastError());
        ms_make += timer_finish(&T);

        CUDA_OR_DIE(cudaMemset(d_flags, 1, n));
        if (nbase) {
            cudaEventRecord(T.a);
            k_mark_composites<<<nbase, 128>>>(lo, n, d_base, nbase, d_flags);
            CUDA_OR_DIE(cudaGetLastError());
            ms_mark += timer_finish(&T);
        }

        cudaEventRecord(T.a);
        CUDA_OR_DIE(cub::DeviceSelect::Flagged(d_temp, select_temp, d_values, d_flags,
                                               d_primes, d_nprime, n));
        ms_select += timer_finish(&T);
        CUDA_OR_DIE(cudaMemcpy(&nprime, d_nprime, sizeof(nprime), cudaMemcpyDeviceToHost));
        prime_total += nprime;
        if (!nprime) { lo64 = hi64 + 2u; continue; }

        /* CUB preserves input order, so one tiny device-side binary search gives
         * the prefix length belonging to each side when the bounds differ. */
        k_upper_bound<<<1,1>>>(d_primes, nprime, alim, d_total);
        CUDA_OR_DIE(cudaGetLastError());
        CUDA_OR_DIE(cudaMemcpy(&nalg, d_total, sizeof(nalg), cudaMemcpyDeviceToHost));
        k_upper_bound<<<1,1>>>(d_primes, nprime, rlim, d_total);
        CUDA_OR_DIE(cudaGetLastError());
        CUDA_OR_DIE(cudaMemcpy(&nrat, d_total, sizeof(nrat), cudaMemcpyDeviceToHost));

        if (nrat) {
            CUDA_OR_DIE(cudaMemset(d_failures, 0, sizeof(*d_failures)));
            cudaEventRecord(T.a);
            k_rational_roots<<<std::min<uint32_t>((nrat + 255u) / 256u,
                                                  (uint32_t)prop.multiProcessorCount * 8u), 256>>>(
                d_primes, nrat, d_rroots, d_failures);
            CUDA_OR_DIE(cudaGetLastError());
            ms_rat += timer_finish(&T);
            CUDA_OR_DIE(cudaMemcpy(&failures, d_failures, sizeof(failures), cudaMemcpyDeviceToHost));
            if (failures) {
                fprintf(stderr, "fbgen_gpu: %u rational prime(s) divide both Y0 and Y1\n", failures);
                goto fail;
            }
            rat_ideal_total += nrat;
        }

        if (nalg) {
            size_t slots = (size_t)nalg * GPU_FB_MAX_ROOTS;
            CUDA_OR_DIE(cudaMalloc((void **)&d_rootbuf, slots * sizeof(*d_rootbuf)));
            CUDA_OR_DIE(cudaMemset(d_failures, 0, sizeof(*d_failures)));

            cudaEventRecord(T.a);
            {
                const uint32_t blocks = std::min<uint32_t>((nalg + 127u) / 128u,
                                                           (uint32_t)prop.multiProcessorCount * 8u);
                const int selected = alg_kernel ? alg_kernel
                                                : (alg_symmetric ? 1 :
                                                   (P.deg <= 6 ? 6 : (P.deg <= 8 ? 8 : 1)));
#define LAUNCH_FIXED(CAP, MONT) \
                k_alg_roots_fixed<CAP, MONT, false><<<blocks, 128>>>( \
                    d_primes, nalg, d_rootbuf, d_counts, d_failures)
#define LAUNCH_GENERIC(MONT, SYM) \
                k_alg_roots<MONT, SYM><<<blocks, 128>>>( \
                    d_primes, nalg, d_rootbuf, d_counts, d_failures)
                if (selected == 6) {
                    if (alg_legacy) LAUNCH_FIXED(6, false);
                    else            LAUNCH_FIXED(6, true);
                } else if (selected == 8) {
                    if (alg_legacy) LAUNCH_FIXED(8, false);
                    else            LAUNCH_FIXED(8, true);
                } else {
                    if (alg_legacy) {
                        if (alg_symmetric) LAUNCH_GENERIC(false, true);
                        else               LAUNCH_GENERIC(false, false);
                    } else {
                        if (alg_symmetric) LAUNCH_GENERIC(true, true);
                        else               LAUNCH_GENERIC(true, false);
                    }
                }
#undef LAUNCH_GENERIC
#undef LAUNCH_FIXED
            }
            CUDA_OR_DIE(cudaGetLastError());
            ms_alg += timer_finish(&T);
            CUDA_OR_DIE(cudaMemcpy(&failures, d_failures, sizeof(failures), cudaMemcpyDeviceToHost));
            if (failures) {
                fprintf(stderr, "fbgen_gpu: algebraic root finder failed for %u prime(s) in [%u,%u]\n",
                        failures, lo, (uint32_t)hi64);
                goto fail;
            }

            cudaEventRecord(T.a);
            CUDA_OR_DIE(cub::DeviceScan::ExclusiveSum(d_temp, scan_temp, d_counts,
                                                       d_offsets, nalg));
            k_total_roots<<<1,1>>>(d_counts, d_offsets, nalg, d_total);
            CUDA_OR_DIE(cudaGetLastError());
            ms_scan += timer_finish(&T);
            CUDA_OR_DIE(cudaMemcpy(&nroots, d_total, sizeof(nroots), cudaMemcpyDeviceToHost));
            alg_ideal_total += nroots;

            if (nroots) {
                CUDA_OR_DIE(cudaMalloc((void **)&d_out_p, (size_t)nroots * sizeof(*d_out_p)));
                CUDA_OR_DIE(cudaMalloc((void **)&d_out_r, (size_t)nroots * sizeof(*d_out_r)));
                cudaEventRecord(T.a);
                k_scatter_alg<<<std::min<uint32_t>((nalg + 255u) / 256u,
                                                   (uint32_t)prop.multiProcessorCount * 8u), 256>>>(
                    d_primes, nalg, d_rootbuf, d_counts, d_offsets, d_out_p, d_out_r);
                CUDA_OR_DIE(cudaGetLastError());
                ms_scatter += timer_finish(&T);
            }

            if (check_left) {
                uint32_t m = std::min(check_left, nalg);
                uint32_t mr = std::min(m, nrat);
                if (vector_resize_nothrow(&check_p, m) ||
                    vector_resize_nothrow(&check_c, m) ||
                    vector_resize_nothrow(&check_roots, (size_t)m * GPU_FB_MAX_ROOTS) ||
                    vector_resize_nothrow(&check_rroots, mr)) {
                    fprintf(stderr, "fbgen_gpu: out of host memory for --check buffers\n");
                    goto fail;
                }
                CUDA_OR_DIE(cudaMemcpy(check_p.data(), d_primes, (size_t)m * 4, cudaMemcpyDeviceToHost));
                CUDA_OR_DIE(cudaMemcpy(check_c.data(), d_counts, (size_t)m * 4, cudaMemcpyDeviceToHost));
                CUDA_OR_DIE(cudaMemcpy(check_roots.data(), d_rootbuf,
                                       (size_t)m * GPU_FB_MAX_ROOTS * 4,
                                       cudaMemcpyDeviceToHost));
                if (mr) CUDA_OR_DIE(cudaMemcpy(check_rroots.data(), d_rroots, (size_t)mr * 4,
                                               cudaMemcpyDeviceToHost));
                for (uint32_t q = 0; q < m; q++) {
                    uint32_t p = check_p[q];
                    for (uint32_t k = 0; k < check_c[q]; k++) {
                        uint32_t r = check_roots[(size_t)q * GPU_FB_MAX_ROOTS + k];
                        if (r == p) {
                            if (host_dec_mod(P.cs[P.deg], p) != 0) {
                                fprintf(stderr, "fbgen_gpu: bad projective root p=%u\n", p);
                                goto fail;
                            }
                        } else if (r >= p || host_poly_eval_mod(&P, r, p) != 0) {
                            fprintf(stderr, "fbgen_gpu: bad algebraic root p=%u r=%u\n", p, r);
                            goto fail;
                        }
                        if (k && check_roots[(size_t)q * GPU_FB_MAX_ROOTS + k - 1] >= r) {
                            fprintf(stderr, "fbgen_gpu: duplicate/unsorted roots at p=%u\n", p);
                            goto fail;
                        }
                    }
                    if (p <= GPU_FB_BRUTE_ROOT_LIMIT) {
                        uint32_t brute = 0;
                        for (uint32_t x = 0; x < p; x++)
                            if (!host_poly_eval_mod(&P, x, p)) brute++;
                        if (!host_dec_mod(P.cs[P.deg], p)) brute++;
                        if (brute != check_c[q]) {
                            fprintf(stderr, "fbgen_gpu: incomplete small-p roots p=%u gpu=%u brute=%u\n",
                                    p, check_c[q], brute);
                            goto fail;
                        }
                    }
                    if (q < mr) {
                        uint32_t y0 = host_dec_mod(P.y0s, p), y1 = host_dec_mod(P.y1s, p);
                        uint32_t r = check_rroots[q];
                        int ok = r == p ? (!y1 && y0)
                                        : (r < p && ((uint64_t)y1 * r + y0) % p == 0);
                        if (!ok) {
                            fprintf(stderr, "fbgen_gpu: bad rational root p=%u r=%u\n", p, r);
                            goto fail;
                        }
                    }
                }
                checked_alg += m;
                checked_rat += mr;
                check_left -= m;
            }

            if (d_out_p) { CUDA_OR_DIE(cudaFree(d_out_p)); d_out_p = NULL; }
            if (d_out_r) { CUDA_OR_DIE(cudaFree(d_out_r)); d_out_r = NULL; }
            if (d_rootbuf) { CUDA_OR_DIE(cudaFree(d_rootbuf)); d_rootbuf = NULL; }
        }

        {
            uint32_t pct = (uint32_t)((hi64 * 100ull) / maxlim);
            if (pct >= next_progress) {
                fprintf(stderr, "fbgen_gpu: %u%% through prime range\n", pct);
                while (next_progress <= pct && next_progress <= 100) next_progress += 10;
            }
        }
        lo64 = hi64 + 2u;
    }

    CUDA_OR_DIE(cudaDeviceSynchronize());
    t_total = now_s() - t0;

    printf("--- generated ordinary-prime factor base ---\n");
    printf("primes through max bound             %12llu\n", (unsigned long long)prime_total);
    printf("rational prime ideals                %12llu\n", (unsigned long long)rat_ideal_total);
    printf("algebraic prime ideals               %12llu\n", (unsigned long long)alg_ideal_total);
    printf("\n--- GPU stage time (summed CUDA events) ---\n");
    printf("candidate generation                 %10.3f ms\n", ms_make);
    printf("segmented composite marking          %10.3f ms\n", ms_mark);
    printf("prime compaction                      %10.3f ms\n", ms_select);
    printf("rational roots                       %10.3f ms\n", ms_rat);
    printf("algebraic roots                      %10.3f ms\n", ms_alg);
    printf("algebraic count scan                 %10.3f ms\n", ms_scan);
    printf("algebraic pair scatter               %10.3f ms\n", ms_scatter);
    printf("GPU stages total                     %10.3f ms\n",
           ms_make + ms_mark + ms_select + ms_rat + ms_alg + ms_scan + ms_scatter);
    printf("CPU base-prime setup                 %10.3f ms\n", (t_base1 - t_base0) * 1000.0);
    printf("alloc/sync/host overhead             %10.3f ms\n",
           t_total * 1000.0 - (ms_make + ms_mark + ms_select + ms_rat + ms_alg +
                               ms_scan + ms_scatter) - (t_base1 - t_base0) * 1000.0);
    printf("wall startup total                   %10.3f ms\n", t_total * 1000.0);
    printf("throughput                           %10.3f M primes/s\n",
           t_total > 0 ? (double)prime_total / t_total / 1.0e6 : 0.0);
    printf("\nvalidation: %llu algebraic and %llu rational prime entries checked on host;"
           " small-p completeness checked\n",
           (unsigned long long)checked_alg, (unsigned long long)checked_rat);
    printf("limitation: prime powers/Hensel lifts are excluded from this prototype\n");
    rc = 0;

fail:
    if (T.a) cudaEventDestroy(T.a);
    if (T.b) cudaEventDestroy(T.b);
    cudaFree(d_out_r); cudaFree(d_out_p); cudaFree(d_rootbuf);
    cudaFree(d_temp); cudaFree(d_total); cudaFree(d_failures); cudaFree(d_rroots);
    cudaFree(d_offsets); cudaFree(d_counts); cudaFree(d_nprime); cudaFree(d_flags);
    cudaFree(d_primes); cudaFree(d_values); cudaFree(d_base);
    free(h_base);
    return rc;
}
#endif /* !FBGEN_GPU_LIBRARY */
