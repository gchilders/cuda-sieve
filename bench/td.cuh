/* Trial division: exact norms, large primes from the resieve, small primes by
 * direct test.
 *
 * The split is the one prototype.md argues for, and it inverts the sieve's own
 * reasoning:
 *
 *   p >= bkthresh (bucketed) -- RESIEVE. Hits are rare and survivors sparse,
 *       so re-walking the factor base and filtering against the survivor
 *       bitmap is cheap. Layout A, measured at 8.55 ms/q.
 *   p <  bkthresh (line-sieved) -- DIRECT TEST per survivor. The small sieve
 *       carries 6.84e9 updates per q; resieving that is ~40 ms, worse than
 *       everything else in the post-sieve budget combined. Testing instead
 *       costs one congruence per (survivor, entry).
 *
 * The direct test is the hot loop -- ~3,500 entries against ~841K survivors is
 * 3e9 tests -- so it is written to avoid integer division entirely. See
 * td_magic_build: each entry carries a multiply-shift reciprocal valid over
 * the range this kernel actually produces, which turns the modulo into
 * __umulhi + shift + mad.
 */
#ifndef CUDA_SIEVE_TD_CUH
#define CUDA_SIEVE_TD_CUH

#include "bigint.cuh"
#include "prp.cuh"

/* ---- exact homogeneous form ------------------------------------------- */

/* F(a,b) = sum_k c[k] a^k b^(deg-k), magnitude and sign kept apart because
 * bn_t is a magnitude type. The RATIONAL side is just deg == 1 with
 * c[1] = Y1, c[0] = Y0 -- G(a,b) = Y1*a + Y0*b is the same homogeneous form,
 * so one kernel covers both sides. */
typedef struct {
    bn_t    c[BENCH_NCOEFF];
    int32_t sign[BENCH_NCOEFF];
    int32_t deg;
} tdpoly_t;

/* ---- small-prime direct-test entry ------------------------------------- */

/* The predicate is the one verify_cpu.c calls hits_pred: solutions exist only
 * on rows j == 0 (mod g), and there i == rt*(j/g) (mod m), with m = p/g.
 *
 * Rewritten to keep the reduced value NONNEGATIVE so a single unsigned
 * multiply-shift reciprocal suffices:
 *
 *     w = rt*j' + (Ihalf - i)          in [1, 2^30), since i < Ihalf
 *     hit  <=>  w mod m == Ihalf mod m
 *
 * `cst` holds Ihalf mod m. m == 1 ("every i, on every g-th row") has no valid
 * 32-bit magic and is flagged by magic == 0; it is rare and the branch is
 * uniform across the warp, since every thread walks the same entry list. */
typedef struct {
    uint64_t recip;    /* Barrett reciprocal of p = m*g, for the division   */
    uint32_t m;        /* reduced modulus p/g                        */
    uint32_t rt;       /* transformed root mod m                     */
    uint32_t g;        /* row divisor; 1 in the ordinary affine case */
    uint32_t cst;      /* Ihalf mod m                                */
    uint32_t magic;    /* multiply-shift reciprocal of m; 0 if m == 1 */
    uint32_t sh;
} tdsmall_t;

/* Multiply-shift reciprocal for divisors below 2^15, exact for dividends below
 * 2^30 -- which is all this kernel produces (rt*j' < 2^15 * 2^14, plus 2^15).
 *
 *   m a power of two 2^k, k >= 1 : magic = 2^(32-k), sh = 0
 *   otherwise                    : sh = floor(log2 m), magic = 2^(32+sh)/m + 1
 *
 * The general case needs magic < 2^32 and (magic*m - 2^(32+sh)) * 2^30 <
 * 2^(32+sh). Both follow from 2^sh < m < 2^(sh+1): the first because
 * 2^(32+sh)/m < 2^32, the second because the error is at most m and
 * m * 2^30 < 2^32 * 2^sh whenever 2^sh > m/4. */
static inline void td_magic_build(uint32_t m, uint32_t *magic, uint32_t *sh)
{
    if (m == 1) { *magic = 0; *sh = 0; return; }
    if ((m & (m - 1)) == 0) {                 /* power of two */
        int k = 0; while ((1u << k) != m) k++;
        *magic = 1u << (32 - k);
        *sh = 0;
        return;
    }
    {
        int s = 0;
        while ((1u << (s + 1)) <= m) s++;      /* s = floor(log2 m) */
        *magic = (uint32_t)(((uint64_t)1 << (32 + s)) / m + 1);
        *sh = (uint32_t)s;
    }
}

#if defined(__CUDACC__)

__device__ __forceinline__ uint32_t td_mod(uint32_t w, uint32_t m,
                                           uint32_t magic, uint32_t sh)
{
    uint32_t q = __umulhi(w, magic) >> sh;
    return w - q * m;
}

/* Divide p out of N to full multiplicity.
 *
 * One pass produces the quotient AND the remainder, so a factor costs two
 * passes -- one that succeeds, one that discovers it is done -- rather than
 * the three a separate mod-then-divide would take. The pass walks only the
 * limbs still in use, which matters: the norm loses ~110 bits over a
 * factorisation and the tail limbs go to zero early.
 *
 * `fac`/`nf` optionally record the factor once per division, so multiplicity
 * comes out as repetition -- the form a relation record wants. Recording is a
 * template parameter rather than a null check so the measured path carries no
 * stores at all. */
template <int RECORD>
__device__ __forceinline__ void td_divide_out(bn_t *N, uint32_t p, uint64_t recip,
                                              uint32_t *fac, uint32_t *nf,
                                              uint32_t fmax)
{
    for (;;) {
        bn_t q = *N;
        int top = bn_top(&q);
        if (top < 0) return;
        if (bn_divmod_u32_pre(&q, p, recip, top)) return;
        *N = q;
        if (RECORD && *nf < fmax) fac[(*nf)++] = p;
        else if (RECORD) (*nf)++;          /* count past the cap so it shows */
    }
}

/* Per-q candidate bookkeeping, on device.
 *
 * Under inline cofactorisation the host used to read back every candidate's
 * coordinates, cofactors, factor counts and two 64-entry factor arrays -- about
 * 618 bytes each -- purely to count them and check for overflow, because the
 * queue had already taken everything it needed straight from device memory.
 * Three counters replace all of it.
 *
 * out[0] = records that are already relations (both sides within lpb)
 * out[1] = records still needing cofactorisation
 * out[2] = records whose factor list overflowed TD_FMAX -- these must FAIL the
 *          run, since the enqueue clamps the count and the missing factors
 *          would leave the norm undivided with nothing downstream to notice.
 */
__global__ void k_cand_stats(uint32_t n,
                             const uint8_t *__restrict bits0,
                             const uint8_t *__restrict bits1,
                             const uint32_t *__restrict fn0,
                             const uint32_t *__restrict fn1,
                             uint32_t lpb0, uint32_t lpb1, uint32_t fmax,
                             uint32_t *__restrict out)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        if (fn0[t] > fmax || fn1[t] > fmax) { atomicAdd(&out[2], 1u); continue; }
        if ((uint32_t)bits0[t] <= lpb0 && (uint32_t)bits1[t] <= lpb1)
            atomicAdd(&out[0], 1u);
        else
            atomicAdd(&out[1], 1u);
    }
}

/* ---- survivor rank ----------------------------------------------------- */

/* The compacted survivor list is written in rank order over the two-sided
 * bitmap, so a position's index in that list is a function of the bitmap
 * alone. That is what lets the resieve scatter a recovered prime straight
 * into its survivor's slot without sorting 1.9M (x,p) pairs or building a
 * 2 GB position-indexed map.
 *
 * gbase[] holds the exclusive prefix over groups of TD_GROUP_W words, so a
 * lookup reads at most TD_GROUP_W-1 extra words. 8 words = 256 positions
 * keeps that bounded while holding the table to 8 MB on the full area. */
#define TD_GROUP_W 8
#define TD_GROUP_X (TD_GROUP_W * 32)

__device__ __forceinline__ uint32_t td_rank(const uint32_t *__restrict bits,
                                            const uint32_t *__restrict gbase,
                                            uint32_t x)
{
    const uint32_t wx = x >> 5;
    const uint32_t w0 = (x / TD_GROUP_X) * TD_GROUP_W;
    uint32_t r = gbase[x / TD_GROUP_X];
    for (uint32_t w = w0; w < wx; w++) r += __popc(bits[w]);
    return r + __popc(bits[wx] & ((1u << (x & 31u)) - 1u));
}

__global__ void k_group_counts(const uint32_t *__restrict bits,
                               uint32_t ngroup, uint32_t *__restrict cnt)
{
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < ngroup; g += stride) {
        uint32_t s = 0;
        for (uint32_t k = 0; k < TD_GROUP_W; k++) s += __popc(bits[g * TD_GROUP_W + k]);
        cnt[g] = s;
    }
}

/* ---- exclusive scan ---------------------------------------------------- */
/* Hand-written rather than CUB: this project builds with nothing but nvcc and
 * libm, and the scan is 40 lines. Three passes -- block-local scan, scan of
 * the block totals, then add the offsets back. */

#define TD_SCAN_BLK 256

__global__ void k_scan_pass1(const uint32_t *__restrict in, uint32_t n,
                             uint32_t *__restrict out, uint32_t *__restrict bsum)
{
    __shared__ uint32_t s[TD_SCAN_BLK];
    uint32_t i = blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    s[threadIdx.x] = (i < n) ? in[i] : 0u;
    __syncthreads();
    for (uint32_t off = 1; off < TD_SCAN_BLK; off <<= 1) {
        uint32_t v = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += v;
        __syncthreads();
    }
    if (i < n) out[i] = s[threadIdx.x] - ((i < n) ? in[i] : 0u);   /* exclusive */
    if (threadIdx.x == TD_SCAN_BLK - 1) bsum[blockIdx.x] = s[threadIdx.x];
}

/* One block scans the block totals in place. nb is at most a few thousand. */
__global__ void k_scan_pass2(uint32_t *__restrict bsum, uint32_t nb)
{
    __shared__ uint32_t s[1024];
    __shared__ uint32_t running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    for (uint32_t base = 0; base < nb; base += 1024) {
        uint32_t i = base + threadIdx.x;
        uint32_t v = (i < nb) ? bsum[i] : 0u;
        s[threadIdx.x] = v;
        __syncthreads();
        for (uint32_t off = 1; off < 1024; off <<= 1) {
            uint32_t t = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
            __syncthreads();
            s[threadIdx.x] += t;
            __syncthreads();
        }
        if (i < nb) bsum[i] = running + s[threadIdx.x] - v;         /* exclusive */
        __syncthreads();
        if (threadIdx.x == 1023) running += s[1023];
        __syncthreads();
    }
}

__global__ void k_scan_pass3(uint32_t *__restrict out, uint32_t n,
                             const uint32_t *__restrict bsum)
{
    uint32_t i = blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    if (i < n) out[i] += bsum[blockIdx.x];
}

/* ---- rank-ordered compaction ------------------------------------------- */

/* Same output as k_intersect_compact's atomic path, but at a deterministic
 * index: survivor s is the s'th set bit of the two-sided bitmap. Two things
 * come from that. The resieve can find a position's slot with td_rank instead
 * of a sort, and the emitted list is reproducible run to run, which is what
 * makes a bit-identical regression test on it mean anything. */
__global__ void k_emit_ranked(const uint32_t *__restrict bits,
                              const uint32_t *__restrict gbase,
                              uint32_t nword, uint32_t logI,
                              int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                              uint32_t *__restrict out_x,
                              int64_t *__restrict out_a,
                              int64_t *__restrict out_b,
                              uint32_t cap)
{
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < nword; w += stride) {
        uint32_t m = bits[w];
        if (!m) continue;
        /* rank of this word's first position: the group base plus the words
         * of the group that precede it. */
        uint32_t slot = gbase[w / TD_GROUP_W];
        for (uint32_t k = (w / TD_GROUP_W) * TD_GROUP_W; k < w; k++)
            slot += __popc(bits[k]);
        while (m) {
            uint32_t k = __ffs(m) - 1;
            m &= m - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            uint32_t j = x >> logI;
            if (slot < cap) {
                out_x[slot] = x;
                out_a[slot] = (int64_t)i * a0 + (int64_t)j * b0;
                out_b[slot] = (int64_t)i * a1 + (int64_t)j * b1;
            }
            slot++;
        }
    }
}

/* ---- resieve that scatters into per-survivor lists --------------------- *
 *
 * k_resieve_rewalk with the output changed from an unordered (x,p) array to a
 * per-survivor list. PROPER PRIME POWERS ARE SKIPPED: if p^2 hits a position
 * then p hits it too and p is in the same factor base, so the base prime is
 * always recorded and multiplicity is recovered by repeated division. A p^2
 * entry recorded as if it were prime would be divided out as p^2, which is
 * wrong whenever the true multiplicity is odd.
 *
 * UNROLL is the whole performance story of this kernel, so it is worth stating
 * what it is fixing.
 *
 * The walk is neither arithmetic-bound nor bandwidth-bound. It runs 315.5M
 * steps in 6.69 ms, which is 325 lane-cycles for a step of about eight
 * instructions, at full occupancy (40 registers, 48 warps/SM, no spills). What
 * costs is the summary probe: one scattered dependent load per step, and with
 * one load in flight per warp, 48 warps cannot cover L2 latency.
 *
 * A COARSE PRE-FILTER ABOVE THE SUMMARY WAS TRIED AND LOST at every
 * granularity from 256 to 4096 positions per bit -- 6.69 ms became 6.75 to
 * 7.82 -- even though at 256 it rejected 96.5% of steps before they reached
 * the summary. That is the measurement that identifies the bottleneck:
 * removing 96.5% of the probes did not help, so the probes were never the
 * throughput cost. Rejecting a step still costs one dependent load, and the
 * steps that pass now pay two in series.
 *
 * The fix for latency is more loads in flight, not fewer loads. Walking UNROLL
 * positions ahead and issuing all UNROLL summary probes before branching on
 * any of them multiplies memory-level parallelism per warp by UNROLL. The walk
 * itself is pure register arithmetic, so running ahead costs nothing and needs
 * no speculation.
 */
template <int UNROLL>
__global__ void k_resieve_scatter(const plat_t *__restrict plat,
                                  const uint32_t *__restrict primes,
                                  const uint8_t *__restrict ispow,
                                  uint32_t n, uint32_t xmax, int logI,
                                  const uint32_t *__restrict summary,
                                  const uint32_t *__restrict bits,
                                  const uint32_t *__restrict gbase,
                                  uint32_t *__restrict plist,
                                  uint32_t *__restrict pcnt,
                                  uint32_t K,
                                  unsigned long long *__restrict noverflow,
                                  int log_gran)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t NONE = 0xFFFFFFFFu;      /* x < 2^29, so this cannot collide */
    uint32_t stride = gridDim.x * blockDim.x;
    unsigned long long ovf = 0;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        plat_t P = plat[k];
        if (P.inc_warp == 0xFFFFFFFFu) continue;
        if (ispow && ispow[k]) continue;
        uint32_t p = primes[k];
        uint32_t x = pl_first(&P, logI);

        while (x < xmax) {
            uint32_t xs[UNROLL], hit[UNROLL];

            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                if (x < xmax) { xs[u] = x; x = pl_next(x, &P, Imask); }
                else xs[u] = NONE;
            }
            /* every one of these is independent, so they issue together */
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                uint32_t sb = xs[u] >> log_gran;
                hit[u] = (xs[u] == NONE) ? 0u
                                         : ((summary[sb >> 5] >> (sb & 31u)) & 1u);
            }
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                if (!hit[u]) continue;
                uint32_t xv = xs[u];
                if (!((bits[xv >> 5] >> (xv & 31u)) & 1u)) continue;
                uint32_t idx = td_rank(bits, gbase, xv);
                uint32_t slot = atomicAdd(&pcnt[idx], 1u);
                if (slot < K) plist[(size_t)idx * K + slot] = p;
                else ovf++;
            }
        }
    }
    if (ovf) atomicAdd(noverflow, ovf);
}

/* ---- the trial-division kernel ----------------------------------------- */

#define TDF_NORM_OVERFLOW  1u
#define TDF_LIST_TRUNCATED 2u

/* One thread per survivor. The norm is built in registers, divided down in
 * place, and only the residual cofactor is written out -- a 224-bit norm never
 * touches global memory.
 *
 * The small-prime table is STAGED IN SHARED MEMORY in tiles. Every thread
 * walks the same entry list, so the address is warp-uniform and the read looks
 * free -- it is not. Reading the entries straight from global cost 40 ms
 * against an arithmetic floor near 1 ms: at 3,633 entries x 841K survivors the
 * kernel issues 3e9 loads of 24 B each, and load issue, not arithmetic, is
 * what it spends its time on. Shared memory turns each into a broadcast.
 *
 * The survivor loop is the OUTER one because the norm has to stay in registers
 * across the whole entry list, and it runs a uniform iteration count so every
 * thread reaches each __syncthreads. */
#define TD_TILE 512
#define TD_MAXHIT 16      /* buffered small-prime hits; ~7 per survivor typical */
/* Recorded factors per survivor, multiplicity included. 32 was not enough: a
 * 224-bit algebraic norm carrying a high power of a small prime records one
 * entry per division, and a candidate at q=120000007 exceeded it. Exceeding
 * the cap fails the run rather than emitting a truncated factorisation, so the
 * observed maximum is reported to keep the margin visible. */
#define TD_FMAX  64

/* DIVIDE == 0 runs the identical hit test but skips the big-integer division
 * it triggers, and counts the hits instead. That is the only way to tell
 * whether this kernel is spending its time on 3e9 congruence tests or on the
 * few million 256-bit divisions those tests find -- the two have completely
 * different fixes and the fused number cannot distinguish them. */
/* SELECT == 1 runs over a LIST of survivor indices rather than all of them.
 * Inputs stay indexed by the survivor index s = sel[t]; outputs are written at
 * the compacted index t. That is what lets the recording pass -- the only pass
 * that writes a 64-word factor list per thread -- run over the ~1,900 joint
 * candidates instead of the ~240,000 survivors, which is the difference between
 * a 61 MB device array read back per side and a 600 KB one. The dense form is
 * still what the measured passes use; recording is not on the hot path. */
template <int DIVIDE, int RECORD, int SELECT>
__global__ void k_td(const int64_t *__restrict A, const int64_t *__restrict B,
                     const uint32_t *__restrict X,
                     const uint32_t *__restrict sel, uint32_t n, int logI,
                     const tdpoly_t *__restrict P,
                     uint32_t sq,
                     const uint32_t *__restrict plist,
                     const uint32_t *__restrict pcnt, uint32_t K,
                     const tdsmall_t *__restrict sm, uint32_t nsm,
                     bn_t *__restrict cof, uint8_t *__restrict cofbits,
                     uint32_t *__restrict flags,
                     unsigned long long *__restrict nhit,
                     uint32_t *__restrict fac, uint32_t *__restrict faccnt,
                     uint32_t fmax)
{
    __shared__ tdsmall_t tile[TD_TILE];

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint32_t nthread = gridDim.x * blockDim.x;
    const uint32_t iters = (n + nthread - 1) / nthread;
    const int deg = P->deg;
    unsigned long long hits = 0;

    for (uint32_t it = 0; it < iters; it++) {
        const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x + it * nthread;
        const bool active = (t < n);
        const uint32_t s = (SELECT && active) ? sel[t] : t;
        const int64_t a = active ? A[s] : 0;
        const int64_t b = active ? B[s] : 0;
        const uint64_t ua = (uint64_t)(a < 0 ? -a : a);
        const uint64_t ub = (uint64_t)(b < 0 ? -b : b);
        const int sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;
        uint32_t myflags = 0;
        uint32_t nf = 0;
        uint32_t *myfac = (RECORD && active) ? fac + (size_t)t * fmax : NULL;
        bn_t N;

        /* ---- exact norm ---- */
        {
            bns_t acc; bns_zero(&acc);
            for (int k = 0; k <= deg; k++) {
                bn_t term = P->c[k];
                int s = P->sign[k];
                if (bn_is_zero(&term)) continue;
                for (int e = 0; e < k; e++) {
                    if (bn_mul_u64(&term, ua)) myflags |= TDF_NORM_OVERFLOW;
                    s *= sa;
                }
                for (int e = 0; e < deg - k; e++) {
                    if (bn_mul_u64(&term, ub)) myflags |= TDF_NORM_OVERFLOW;
                    s *= sb;
                }
                if (bns_addmag(&acc, &term, s)) myflags |= TDF_NORM_OVERFLOW;
            }
            N = acc.m;
        }

        /* ---- the special-q, on its own side ---- */
        if (active && sq)
            td_divide_out<RECORD>(&N, sq, bn_recip_u32(sq), myfac, &nf, fmax);

        /* ---- large primes recovered by the resieve ---- */
        if (active) {
            uint32_t c = pcnt[s];
            if (c > K) { c = K; myflags |= TDF_LIST_TRUNCATED; }
            for (uint32_t k = 0; k < c; k++) {
                uint32_t p = plist[(size_t)s * K + k];
                td_divide_out<RECORD>(&N, p, bn_recip_u32(p), myfac, &nf, fmax);
            }
        }

        /* ---- small primes, direct test ----
         *
         * Hits are BUFFERED rather than divided where they are found. A hit is
         * a per-lane event but the division is a whole-warp one: with ~7 hits
         * per survivor spread over 3,633 entries, a warp stalls on roughly 44
         * separate divisions, each with most lanes idle. Buffering turns that
         * into one dense division loop per survivor where every lane has work,
         * and the entry loop keeps its uniform control flow. */
        {
            const uint32_t x = active ? X[s] : 0u;
            const int32_t  i = (int32_t)(x & Imask) - Ihalf;
            const uint32_t j = x >> logI;
            const uint32_t hi = (uint32_t)(Ihalf - i);      /* in (0, 2^logI] */
            uint32_t hitp[TD_MAXHIT];
            uint64_t hitr[TD_MAXHIT];       /* carry the reciprocal with it:
                                             * recomputing it later would put a
                                             * 64-bit division back in the hot
                                             * path, which is what this whole
                                             * arrangement exists to avoid */
            uint32_t nh = 0;

            for (uint32_t base = 0; base < nsm; base += TD_TILE) {
                const uint32_t cnt = min((uint32_t)TD_TILE, nsm - base);
                __syncthreads();
                for (uint32_t k = threadIdx.x; k < cnt; k += blockDim.x)
                    tile[k] = sm[base + k];
                __syncthreads();
                if (!active) continue;
                for (uint32_t e = 0; e < cnt; e++) {
                    const uint32_t m = tile[e].m, g = tile[e].g;
                    uint32_t jp = j;
                    if (g > 1) {
                        if (j % g) continue;
                        jp = j / g;
                    }
                    if (tile[e].magic) {
                        uint32_t w = tile[e].rt * jp + hi;
                        if (td_mod(w, m, tile[e].magic, tile[e].sh) != tile[e].cst)
                            continue;
                    }
                    /* magic == 0 means m == 1: the row condition was the whole
                     * test, and it has already passed. */
                    hits++;
                    if (DIVIDE) {
                        hitp[nh] = m * g; hitr[nh] = tile[e].recip; nh++;
                        if (nh == TD_MAXHIT) {          /* drain and carry on */
                            for (uint32_t z = 0; z < TD_MAXHIT; z++)
                                td_divide_out<RECORD>(&N, hitp[z], hitr[z],
                                                      myfac, &nf, fmax);
                            nh = 0;
                        }
                    }
                }
            }
            if (DIVIDE)
                for (uint32_t z = 0; z < nh; z++)
                    td_divide_out<RECORD>(&N, hitp[z], hitr[z],
                                          myfac, &nf, fmax);
        }

        if (active) {
            cof[t] = N;
            cofbits[t] = (uint8_t)bn_bits(&N);
            if (RECORD) faccnt[t] = nf;
            if (myflags) atomicOr(flags, myflags);
        }
    }
    if (nhit && hits) atomicAdd(nhit, hits);
}

/* ---- cofactor classification ------------------------------------------- */

/* Applies CADO's check_leftover_norm per survivor. This is the stage that
 * turns hundreds of thousands of survivors into the ~1,851 that are worth
 * handing to a cofactorizer, and the 2026-08-04 harness work established that
 * it is real work the siever has to do, not "a small input adapter". */
__global__ void k_classify(const bn_t *__restrict cof,
                           const uint8_t *__restrict cofbits,
                           const int64_t *__restrict B, uint32_t n,
                           uint32_t lpb, uint32_t mfb, double lim,
                           uint8_t *__restrict status)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        bn_t c = cof[t];
        /* b == 0 is the lattice point (i,j) = (0,1): a = q, b = 0. It passes
         * gcd(i,j) == 1 and both norm bounds, and its cofactors classify
         * cleanly -- but (a,0) is not a relation, and it turned up as a
         * spurious candidate on the first full run. las carries it through
         * after_sieve too and drops it later. */
        if (B[t] == 0) { status[t] = COF_DEGENERATE; continue; }
        status[t] = (uint8_t)cof_classify(&c, cofbits[t], lpb, mfb, lim);
    }
}

/* ---- joint acceptance, compacted on device ----------------------------- *
 *
 * A record is only worth recording if BOTH sides classified it as a
 * cofactorisation candidate; one side accepting is not a candidate at all. The
 * join used to happen on the host over every survivor, which meant reading back
 * both sides' whole factor matrices to reach ~1,900 rows. Doing the intersection
 * here reduces the readback to those rows.
 *
 * Ordered, not atomic: a prefix scan over the accept flags gives each candidate
 * a deterministic slot, so the emitted batch is byte-reproducible run to run.
 * That property is what makes diffing this path against the two-process one a
 * real regression test, so it is not worth trading for one atomicAdd. */
__global__ void k_accept_flags(const uint8_t *__restrict st0,
                               const uint8_t *__restrict st1,
                               uint32_t n, uint32_t *__restrict flag)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        uint32_t a = (st0[t] == COF_ACCEPT || st0[t] == COF_SPLIT);
        uint32_t b = (st1[t] == COF_ACCEPT || st1[t] == COF_SPLIT);
        flag[t] = a & b;
    }
}

__global__ void k_scatter_sel(const uint32_t *__restrict flag,
                              const uint32_t *__restrict off,
                              uint32_t n, uint32_t *__restrict sel,
                              uint32_t cap, uint32_t *__restrict nacc)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        if (flag[t] && off[t] < cap) sel[off[t]] = t;
        if (t == n - 1) *nacc = off[t] + flag[t];
    }
}

__global__ void k_gather_ab(const int64_t *__restrict A,
                            const int64_t *__restrict B,
                            const uint32_t *__restrict sel, uint32_t n,
                            int64_t *__restrict oa, int64_t *__restrict ob)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        oa[t] = A[sel[t]]; ob[t] = B[sel[t]];
    }
}

#endif  /* __CUDACC__ */
#endif  /* CUDA_SIEVE_TD_CUH */
