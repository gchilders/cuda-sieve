/* Franke-Kleinjung p-lattice reduction and walk.
 *
 * Ported from CADO-NFS sieve/las-plattice.hpp and
 * sieve/las-reduce-plattice-simplistic.hpp so that the benchmark walks the
 * same positions a real lattice siever does. Marked __host__ __device__ so
 * verify_cpu.c and the GPU kernels run bit-identical code.
 *
 * Proposition 1 of [FrKl05]: a basis <(i0,j0),(i1,j1)> of the p-lattice
 * inside the q-lattice with j0,j1 > 0, -I < i0 <= 0 <= i1 < I, i1-i0 >= I.
 * i0 is stored negated as mi0.
 *
 * Positions are encoded x = (i + I/2) + (j << logI). For I15e, I*J = 2^29,
 * so uint32 suffices for x and for both increments; CADO uses uint64 because
 * it must also handle I=16 with rescaled J.
 */
#ifndef CUDA_SIEVE_PLATTICE_CUH
#define CUDA_SIEVE_PLATTICE_CUH

#include <stdint.h>

#ifdef __CUDACC__
#define PL_FN __host__ __device__ static inline
#else
#define PL_FN static inline
#endif

/* Walk parameters, 16 bytes. Produced once per (p,r) per special-q. */
typedef struct {
    uint32_t inc_warp;   /* (j0 << logI) + i0 , always > 0 as an offset */
    uint32_t inc_step;   /* (j1 << logI) + i1 , or PL_VERTICAL          */
    uint32_t bound_warp; /* add warp when (x & (I-1)) >= bound_warp     */
    uint32_t bound_step; /* add step when (x & (I-1)) <  bound_step     */
} plat_t;

#define PL_VERTICAL 0xFFFFFFFFu

typedef struct { uint32_t mi0, j0, i1, j1; } pl_basis_t;

PL_FN void pl_reduce_with_vertical_vector(pl_basis_t *B, uint32_t I)
{
    const uint32_t a = (I - 1) % B->mi0;
    uint32_t mi2 = (I - 1) - a - B->i1;
    const uint32_t j2 = B->j1;
    if (B->mi0 + mi2 < I) mi2 += B->mi0;
    B->i1  = B->mi0;
    B->j1  = B->j0;
    B->mi0 = mi2;
    B->j0  = j2;
}

/* CADO's reduce_plattice_simplistic, verbatim in structure. */
PL_FN void pl_reduce(pl_basis_t *B, uint32_t I)
{
    /* needs_special_treatment: i1 == 0 || (j1 > 1 && mi0 < I).
     * On the initial basis j1 == 1, so this is exactly r == 0. */
    if (B->i1 == 0 || (B->j1 > 1 && B->mi0 < I)) {
        pl_reduce_with_vertical_vector(B, I);
        return;
    }
    for (;;) {
        if (B->i1 < I) {
            if (B->i1 == 0) {
                B->j0 = B->j1 - B->j0;
                pl_reduce_with_vertical_vector(B, I);
                return;
            }
            {
                uint32_t a = (B->mi0 + B->i1 - I) / B->i1;
                B->mi0 -= a * B->i1;
                B->j0  += a * B->j1;
            }
            return;
        }
        {
            uint32_t k = B->mi0 / B->i1;
            B->mi0 -= k * B->i1;
            B->j0  += k * B->j1;
        }
        if (B->mi0 < I) {
            if (B->mi0 == 0) {
                uint32_t t = B->i1;
                B->mi0 = t;
                B->i1  = B->j0; B->j0 = B->j1; B->j1 = B->i1;
                B->i1  = 0;
                pl_reduce_with_vertical_vector(B, I);
                return;
            }
            {
                uint32_t a = (B->mi0 + B->i1 - I) / B->mi0;
                B->i1 -= a * B->mi0;
                B->j1 += a * B->j0;
            }
            return;
        }
        {
            uint32_t k = B->i1 / B->mi0;
            B->i1 -= k * B->mi0;
            B->j1 += k * B->j0;
        }
    }
}

/* Build the walk parameters for prime p with transformed root r. */
PL_FN plat_t pl_make(uint32_t p, uint32_t r, int logI)
{
    const uint32_t I = 1u << logI;
    pl_basis_t B;
    plat_t P;
    B.mi0 = p; B.j0 = 0; B.i1 = r; B.j1 = 1;   /* initial_basis, affine */
    pl_reduce(&B, I);

    P.bound_step = B.mi0;
    if (B.i1 >> logI) {                         /* vertical line */
        P.bound_warp = 0;
        P.inc_step   = PL_VERTICAL;
    } else {
        P.bound_warp = I - B.i1;
        P.inc_step   = (B.j1 << logI) + B.i1;
    }
    /* inc_warp = (j0 << logI) + i0 with i0 = -mi0; positive because
     * the post-conditions give mi0 < I <= j0*I whenever j0 >= 1. */
    P.inc_warp = (B.j0 << logI) - B.mi0;
    return P;
}

/* Advance x to the next lattice position. */
PL_FN uint32_t pl_next(uint32_t x, const plat_t *P, uint32_t Imask)
{
    const uint32_t i = x & Imask;
    if (i >= P->bound_warp) x += P->inc_warp;
    if (i <  P->bound_step) x += P->inc_step;
    return x;
}

/* The lattice always contains (i,j) = (0,0), i.e. x = I/2, and for j == 0
 * that is its *only* point (i == 0 mod p and |i| < I/2 < p forces i == 0).
 * pl_start returns it; brute-force verification enumerates from here. */
PL_FN uint32_t pl_start(int logI) { return 1u << (logI - 1); }

/* Sieving starts at j == 1: (0,0) is not a relation, and counting it would
 * add one spurious record *per prime* -- 6.8M of them on this factor base,
 * which is 4x the entire true update count at small sieve areas. Every
 * throughput path must use this, not pl_start. */
PL_FN uint32_t pl_first(const plat_t *P, int logI)
{
    return pl_next(1u << (logI - 1), P, (1u << logI) - 1);
}

/* ---- modular arithmetic used by the root transform ------------------- */

/* Binary extended Euclid. p odd, 0 < a < p. Returns a^-1 mod p, or 0 if
 * a == 0. Divergent by nature; the design doc's recommended replacement is
 * a branch-free 2-adic (Hensel/REDC) inverse, which is variant "TH". */
PL_FN uint32_t pl_invmod(uint32_t a, uint32_t p)
{
    uint32_t u = a, v = p, x1 = 1, x2 = 0;
    if (a == 0) return 0;
    while (u != 1 && v != 1) {
        while ((u & 1) == 0) {
            u >>= 1;
            x1 = (x1 & 1) ? (uint32_t)((x1 + p) >> 1) : (x1 >> 1);
        }
        while ((v & 1) == 0) {
            v >>= 1;
            x2 = (x2 & 1) ? (uint32_t)((x2 + p) >> 1) : (x2 >> 1);
        }
        if (u >= v) { u -= v; x1 = (x1 >= x2) ? x1 - x2 : x1 + p - x2; }
        else        { v -= u; x2 = (x2 >= x1) ? x2 - x1 : x2 + p - x1; }
    }
    return (u == 1) ? x1 : x2;
}

/* Inverse modulo an arbitrary prime power.
 *
 * pl_invmod above is binary extended Euclid and requires an ODD modulus -- its
 * halving step (x+p)>>1 is only exact when p is odd. GGNFS's .afb.0 has no
 * even entries so this never mattered; CADO's factor base carries the whole
 * ladder 2,4,8,...,32768, and feeding one to pl_invmod does not fail, it
 * LOOPS FOREVER. Route powers of two to a 2-adic Newton iteration instead:
 * x <- x*(2 - a*x) doubles the number of correct bits each step, and x = a is
 * already correct mod 8 for odd a, so five steps cover 32 bits. */
PL_FN uint32_t pl_invmod_any(uint32_t a, uint32_t m)
{
    if (m & 1u) return pl_invmod(a, m);
    if (!(a & 1u)) return 0;              /* not invertible mod 2^k */
    {
        uint32_t inv = a;
        int i;
        for (i = 0; i < 5; i++) inv *= 2u - a * inv;
        return inv & (m - 1u);
    }
}

/* Transformed root: the r' with i == r'*j (mod p) over the q-lattice.
 *
 * BEWARE THE FIELD NAMING. CADO's qlattice groups by coordinate --
 * a = a0*i + a1*j, b = b0*i + b1*j -- and `fb_root_in_qlattice` therefore
 * returns (R*b1 - a1)/(a0 - R*b0). qlat_t groups by *vector*: (a0,a1) and
 * (b0,b1) are the two basis vectors, so a = i*a0 + j*b0 and b = i*a1 + j*b1
 * (this is what qlat_build constructs and what norm_setup consumes). Copying
 * CADO's formula across the two conventions silently transposes the basis --
 * a1 and b0 swap -- which yields a perfectly valid lattice of the right
 * density, so record counts and CPU/GPU cross-checks all still agree. It is
 * simply a *different* lattice from the one the norms describe, and nothing
 * short of parity against las would catch it.
 *
 * In this struct's convention, a == r*b (mod p) becomes
 *     i*(a0 - r*a1) == j*(r*b1 - b0)   (mod p)
 * so r' = (r*b1 - b0) / (a0 - r*a1).
 *
 * Returns p (an out-of-range sentinel) when the denominator vanishes: the
 * transformed root is itself projective. */
PL_FN uint32_t pl_transform(uint32_t p, uint32_t r,
                            int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
    const int64_t P = (int64_t)p;
    int64_t num = ((int64_t)r * b1 - b0) % P;
    int64_t den = (a0 - (int64_t)r * a1) % P;
    if (num < 0) num += P;
    if (den < 0) den += P;
    if (den == 0) return p;
    return (uint32_t)(( (uint64_t)num * pl_invmod_any((uint32_t)den, p) ) % (uint64_t)p);
}

#define PL_ROWS 0xFFFFFFFFu

/* General transformed root, valid for prime POWERS as well as primes.
 *
 * With a prime modulus the denominator is either invertible or zero. With
 * q = p^k there is a third case: p | D but q does not, and then D has no
 * inverse mod q at all -- binary Euclid does not report failure, it spins
 * forever on gcd > 1. That is exactly what CADO's factor base triggers, since
 * makefb emits 49, 121, 169, ... alongside the plain primes.
 *
 * ROOT ENCODING. Both CADO and GGNFS store a root of q in [0, 2q). A value
 * r < q is AFFINE and means a == r*b (mod q). A value r >= q is PROJECTIVE
 * with reciprocal rr = r - q, and means
 *
 *     a * rr == b   (mod q).
 *
 * rr == 0 is the classical "b == 0 (mod q)" case, and it is the ONLY case that
 * arises for a prime q -- which is why treating every projective root as
 * rr == 0 survives a prime-only factor base. It is wrong for prime powers.
 * A projective ideal above p lifts to p^k as a point of P^1(Z/p^k) whose
 * reciprocal is divisible by p but not by p^k, so the ladder above a
 * projective prime carries NONZERO reciprocals: `c183.fb1` opens with
 * `4:4,3: 6` (q = 4, rr = 2) and has 35 such entries, worth 4.7e8 updates and
 * 1.54 scaled log units per cell. Forcing rr to 0 keeps the density (both are
 * index-q sublattices) and moves every one of those updates to the wrong
 * congruence -- invisible to a density gate like sum(logp/q), and invisible to
 * a CPU replay that shares this function. Pass the reciprocal.
 *
 * Solve it in general. The condition becomes i*D == j*N (mod q) with
 *     affine (a == r*b):      D = a0 - r*a1,     N = r*b1 - b0
 *     projective (a*rr == b): D = rr*a0 - a1,    N = b1 - rr*b0
 * (at rr == 0 the projective pair is (-a1, b1), the negation of the old
 * (a1, -b1) -- same ratio, so this generalises the previous case rather than
 * replacing it). D and N cannot both be divisible by p, or p would divide the
 * determinant, which is the special-q: in the projective case
 * rr*det == D*b1 + a1*N, and p | rr would force p | a1 and p | b1. So with
 * g = gcd(D, q):
 *
 * PRECONDITION: q is a prime power. That is all a factor base ever contains,
 * and the parameterisation below genuinely needs it -- the step "solutions
 * exist only when g | j" uses gcd(N, g) == 1, which follows from "p divides at
 * most one of D and N" and holds only for a single prime p. For a composite
 * modulus with two prime factors (q = 200, D even, N divisible by 5) the
 * solution set is a CRT combination and this form is wrong. `fbtest` gates
 * both halves of that: the transform against the definition over prime-power
 * moduli, and every loaded factor base for being prime powers only.
 *
 *     solutions exist only when g | j, and then, writing j = g*j',
 *     i == rt * j'  (mod m),   m = q/g,   rt = N * (D/g)^-1  (mod m).
 *
 * g == 1 is the ordinary affine case. g == q gives m == 1, meaning "every i,
 * on every q-th row" -- which is precisely the old PL_ROWS special case, now
 * subsumed rather than special. */
PL_FN uint32_t pl_gcd32(uint32_t a, uint32_t b)
{
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

/* Returns the modulus m; writes the root and the row divisor g. */
PL_FN uint32_t pl_transform_gen(uint32_t q, uint32_t r, int is_proj,
                                int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                uint32_t *rt_out, uint32_t *g_out)
{
    const int64_t Q = (int64_t)q;
    int64_t D = is_proj ? (((int64_t)r * a0 - a1) % Q) : ((a0 - (int64_t)r * a1) % Q);
    int64_t N = is_proj ? ((b1 - (int64_t)r * b0) % Q) : (((int64_t)r * b1 - b0) % Q);
    uint32_t g, m, dg;
    if (D < 0) D += Q;
    if (N < 0) N += Q;
    g = pl_gcd32((uint32_t)D, q);
    m = q / g;
    if (m == 1) { *rt_out = 0; *g_out = g; return 1; }
    dg = (uint32_t)((uint64_t)D / g);
    *rt_out = (uint32_t)(((uint64_t)((uint32_t)N % m)
                          * pl_invmod_any(dg % m, m)) % (uint64_t)m);
    *g_out = g;
    return m;
}

/* Transform a root in the factor base's OWN encoding -- affine below q,
 * projective with reciprocal r - q at or above it. Every consumer of a loaded
 * factor base should call this and nothing else: it is the single place the
 * encoding is decoded, it is power-safe, and it cannot hang. */
PL_FN uint32_t pl_transform_enc(uint32_t q, uint32_t r_enc,
                                int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                uint32_t *rt_out, uint32_t *g_out)
{
    const int isp = (r_enc >= q);
    return pl_transform_gen(q, isp ? r_enc - q : r_enc, isp,
                            a0, a1, b0, b1, rt_out, g_out);
}

/* Projective factor-base root (GGNFS and CADO both encode it as root >= p):
 * the condition is b == 0 (mod p) rather than a == r*b. Kept for the
 * prime-modulus paths; pl_transform_gen handles it uniformly. */

PL_FN uint32_t pl_transform_proj(uint32_t p,
                                 int64_t a0, int64_t a1, int64_t b0, int64_t b1)
{
    const int64_t P = (int64_t)p;
    int64_t d = a1 % P, num = (-b1) % P;
    (void)a0; (void)b0;
    if (d < 0) d += P;
    if (num < 0) num += P;
    if (d == 0) return PL_ROWS;
    return (uint32_t)(( (uint64_t)num * pl_invmod_any((uint32_t)d, p) ) % (uint64_t)p);
}

#endif
