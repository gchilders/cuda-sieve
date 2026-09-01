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

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SLAB_HD __host__ __device__ static inline
#else
#define SLAB_HD static inline
#endif

typedef struct {
    uint32_t jmax;       /* maximum rows in one slab                    */
    uint32_t nslab;      /* ceil(J / jmax)                              */
    int      enabled;    /* nslab > 1                                  */
} slab_plan_t;

/* Trial division ranks the survivor bitmap in groups of 8 x 32-bit words =
 * 256 positions. Every slab, including the final tail, must contain a whole
 * number of those groups. For logI >= 8 every complete j row already does. */
#define SLAB_TD_GROUP_POS 256u

/* Performance policy: once a full sieve reaches 2^30 positions, auto mode
 * targets slabs no larger than 2^29 positions.  Two independent benchmarks
 * (Ampere RTX 3090 and Blackwell RTX 5070) found this working set near the
 * fill/TD crossover; an L40 with a larger L2 cache instead preferred 2^30 for
 * maximum throughput. This is a generic performance/memory tuning cap, not a
 * universal speed optimum. Explicit --slab-j overrides it, while the 2^31
 * position and direct-TD bounds below remain mandatory.
 *
 * gfx1103 (HIP build) does NOT confirm 2^29 -- it is measurably worse here,
 * not just "not proven better". Swept 2^26..2^31-position slabs at logI=16,
 * J=65536 (33-special-q sample, oracle/c183.poly): 2^27 gave 3889 ms/q,
 * 2^29 (this constant's CUDA-tuned value) gave 5664 ms/q -- 2^29 is 45.6%
 * SLOWER than 2^27 on this hardware, and 2^30 (the L40's preferred point)
 * is catastrophic here (12794 ms/q, 3.3x worse than 2^27). Consistent with
 * gfx1103's small 2 MB L2 (vs. tens of MB on the NVIDIA cards this was
 * tuned against) and its UMA/shared-DDR5 memory subsystem -- a smaller
 * working set matters much more here. BENCH_HIP_BUILD-gated rather than
 * changed outright, so the CUDA build's own tuned value is untouched; see
 * HIP_TUNING_PLAN.md for the full sweep data.
 *
 * TRIGGER moves too, and not by guesswork: TRIGGER_LOG2 = TARGET_LOG2 + 1
 * on the CUDA side is not a coincidence -- slab_perf_jmax() computes rows =
 * TARGET/I independent of J, so a sieve at EXACTLY the trigger (area ==
 * 2*TARGET) gets nslab = ceil(J / (TARGET/I)) = ceil(2) = 2: a clean
 * minimal split right at the boundary, by construction. Leaving TRIGGER at
 * 30 while only dropping TARGET to 27 breaks that invariant (an 8:1 ratio
 * instead of 2:1) and silently reopens the exact problem TARGET was fixed
 * for, just at a smaller size: confirmed by testing a sieve at exactly
 * 2^28 positions (inside the resulting gap) with the old inherited trigger
 * -- it runs unsplit (fill 123.6 ms) since 2^28 < 2^30, versus a forced
 * 2-way split into 2^27 chunks (fill 64.7 ms, 14% faster overall) that
 * TRIGGER=28 would have produced automatically. Restoring TRIGGER = TARGET
 * + 1 keeps the same clean-split-at-boundary property this constant pair
 * was designed around, just recentered on gfx1103's own working set. */
#if defined(BENCH_HIP_BUILD)
#define SLAB_PERF_TRIGGER_LOG2 28u
#define SLAB_PERF_TARGET_LOG2  27u
#else
#define SLAB_PERF_TRIGGER_LOG2 30u
#define SLAB_PERF_TARGET_LOG2  29u
#endif

static inline uint32_t slab_row_quantum(int logI)
{
    if (logI < 0 || logI > 30) return 0;
    return logI >= 8 ? 1u : (1u << (8 - logI));
}

static inline int slab_rows_shape_ok(int logI, uint32_t rows)
{
    const uint32_t q = slab_row_quantum(logI);
    return q && rows && (rows % q) == 0;
}

/* Position-space limit. 2^31 itself is a valid exclusive endpoint in a
 * uint32_t; individual positions are in [0, 2^31).
 *
 * plat_t's walk increments are intentionally NOT a slab-plan constraint. They
 * are uint64_t (plattice.cuh) because a reduced increment can exceed 2^32 even
 * when every slab-local position is below 2^31. The planner only has to bound
 * quantities that remain 32-bit: local x and the direct-TD reciprocal input. */
static inline uint32_t slab_area_jmax(int logI)
{
    if (logI < 0 || logI > 30) return 0;
    return (uint32_t)(((uint64_t)1 << 31) >> logI);
}

/* Return the auto-mode performance cap in rows. UINT32_MAX means that the
 * geometry is below the performance-slabbing trigger and should not be split
 * for performance alone.  If one row itself exceeds 2^29 positions, one row
 * is the smallest representable slab and the correctness caps still apply. */
static inline uint32_t slab_perf_jmax(int logI, uint32_t J)
{
    uint64_t I, area, rows;
    if (logI < 0 || logI > 30 || !J) return 0;
    I = (uint64_t)1 << logI;
    area = I * (uint64_t)J;
    if (area < ((uint64_t)1 << SLAB_PERF_TRIGGER_LOG2)) return 0xffffffffu;
    rows = ((uint64_t)1 << SLAB_PERF_TARGET_LOG2) / I;
    if (!rows) rows = 1u;
    return rows > 0xffffffffull ? 0xffffffffu : (uint32_t)rows;
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
    uint32_t amax, tmax, perf_jmax, jmax, quantum;
    uint64_t n;
    if (!P || !J) return -1;
    quantum = slab_row_quantum(logI);
    if (!quantum || !slab_rows_shape_ok(logI, J)) return -1;
    amax = slab_area_jmax(logI);
    tmax = slab_td_jmax(logI, max_small_prime);
    if (!amax || !tmax) return -1;
    jmax = amax < tmax ? amax : tmax;
    if (forced_j) {
        const uint32_t requested = forced_j < J ? forced_j : J;
        if (requested > jmax || !slab_rows_shape_ok(logI, requested)) return -1;
        jmax = requested;
    } else {
        perf_jmax = slab_perf_jmax(logI, J);
        if (!perf_jmax) return -1;
        if (perf_jmax < jmax) jmax = perf_jmax;
        jmax -= jmax % quantum;
    }
    if (jmax > J) jmax = J;
    if (!jmax || !slab_rows_shape_ok(logI, jmax)) return -1;
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
