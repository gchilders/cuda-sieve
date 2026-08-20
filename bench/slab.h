/* slab.h -- geometry helpers for j-slabbing the production sieve.
 *
 * A slab keeps the hot sieve position x = j_local*I + (i + I/2) in uint32_t
 * and below 2^31, while j_base carries the global row origin outside the walk.
 * The ordinary path never calls any of this from a device hot loop: kernels are
 * specialised on SLABBED=false and the compiler removes every slab operation.
 */
#ifndef CUDA_SIEVE_SLAB_H
#define CUDA_SIEVE_SLAB_H

#include <stdint.h>

#ifdef __CUDACC__
#define SLAB_HD __host__ __device__ static inline
#else
#define SLAB_HD static inline
#endif

typedef struct {
    uint32_t jmax;       /* maximum rows in one slab                    */
    uint32_t nslab;      /* ceil(J / jmax)                              */
    int      enabled;    /* nslab > 1                                  */
} slab_plan_t;

/* Position-space limit. 2^31 itself is a valid exclusive endpoint in a
 * uint32_t; individual positions are in [0, 2^31). */
static inline uint32_t slab_area_jmax(int logI)
{
    if (logI < 0 || logI > 30) return 0;
    return (uint32_t)(((uint64_t)1 << 31) >> logI);
}

/* The direct small-prime predicate uses
 *
 *     w = rt * j_local + hi,    1 <= hi <= I,
 *
 * and td_mod's multiply/shift reciprocal is exact for w < 2^31.  A transformed
 * prime has rt < m <= p, so max_prime is a conservative bound on rt+1.
 * Return the largest slab height whose ENTIRE local row range satisfies that
 * invariant. No small-prime entries means the TD bound is irrelevant.
 */
static inline uint32_t slab_td_jmax(int logI, uint32_t max_prime)
{
    const uint64_t I = (logI >= 0 && logI <= 30) ? ((uint64_t)1 << logI) : 0;
    uint64_t max_rt, room, rows;
    if (!I || I > 0x7fffffffull) return 0;
    if (max_prime <= 1) return 0xffffffffu;
    max_rt = (uint64_t)max_prime - 1u;
    room = 0x7fffffffull - I;
    rows = 1u + room / max_rt;
    return rows > 0xffffffffull ? 0xffffffffu : (uint32_t)rows;
}

/* Build the host-side schedule. forced_j == 0 means auto. A nonzero value is
 * deliberately strict: it is a regression/testing knob, not permission to
 * violate either the position or direct-TD arithmetic bound. */
static inline int slab_make_plan(int logI, uint32_t J, uint32_t max_small_prime,
                                 uint32_t forced_j, slab_plan_t *P)
{
    uint32_t amax, tmax, jmax;
    uint64_t n;
    if (!P || !J) return -1;
    amax = slab_area_jmax(logI);
    tmax = slab_td_jmax(logI, max_small_prime);
    if (!amax || !tmax) return -1;
    jmax = amax < tmax ? amax : tmax;
    if (forced_j) {
        if (forced_j > jmax) return -1;
        jmax = forced_j;
    }
    if (jmax > J) jmax = J;
    if (!jmax) return -1;
    n = ((uint64_t)J + jmax - 1u) / jmax;
    if (n > 0xffffffffull) return -1;
    P->jmax = jmax;
    P->nslab = (uint32_t)n;
    P->enabled = (P->nslab > 1u);
    return 0;
}

SLAB_HD uint32_t slab_rows_at(const slab_plan_t *P, uint32_t J, uint32_t slab)
{
    const uint64_t base = (uint64_t)slab * P->jmax;
    const uint64_t left = base < J ? (uint64_t)J - base : 0;
    return left > P->jmax ? P->jmax : (uint32_t)left;
}

SLAB_HD uint32_t slab_jbase_at(const slab_plan_t *P, uint32_t slab)
{
    return (uint32_t)((uint64_t)slab * P->jmax);
}

/* Advance the small-prime congruence target by delta_j rows. If cst represents
 *
 *     rt*j_local + hi == cst (mod m)
 *
 * at one slab origin, the returned target represents the same global
 * congruence after that origin moves forward by delta_j. This runs once per
 * small-prime ENTRY per slab, not once per survivor x entry test. */
SLAB_HD uint32_t slab_phase_cst(uint32_t cst, uint32_t rt, uint32_t m,
                                uint32_t delta_j)
{
    uint32_t d;
    if (m <= 1u) return cst;
    d = (uint32_t)(((uint64_t)rt * delta_j) % m);
    return cst >= d ? cst - d : cst + (m - d);
}

#undef SLAB_HD
#endif
