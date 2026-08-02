/* GPU kernels for the bucket-fill benchmark.
 *
 * Stages, matching the design doc's cost pillars:
 *   (T) transform + plattice : one modular inverse per distinct prime per
 *                              special-q, then the FK reduction. Producer
 *                              only -- deliberately NOT fused with scatter
 *                              (msieve-s experiment #11: fusing cost +25%).
 *   (a) FILL_ATOMIC          : one global atomicAdd per record, 16K-way.
 *   (c) FILL_TWOLEVEL        : level 1 stages records in shared memory and
 *                              flushes full cache lines with one atomic per
 *                              flush; level 2 splits super-buckets into
 *                              regions the same way.
 */
#include "bench.h"
#include "plattice.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return -1; } } while (0)

/* ---- stage T: root transform + plattice reduction --------------------- */

/* The transform runs through pl_transform_enc, not pl_transform, for three
 * reasons that all bite on CADO's factor base and none of which bite on
 * GGNFS's:
 *   - a projective entry (root >= q) must keep its reciprocal, and the affine
 *     formula would reduce it to a bogus affine root instead;
 *   - q = 2^15 sits exactly at the default bkthresh, so an EVEN modulus
 *     reaches this kernel, and binary-Euclid pl_invmod cannot invert mod 2^k;
 *   - raising -maxbits puts odd prime powers here, where a non-invertible
 *     denominator makes binary Euclid spin forever ON THE DEVICE.
 * pl_transform_enc handles all three; the first two are wrong answers and the
 * third is a hang, so none of them would have shown up as a failed gate.
 *
 * What is still not expressible is g > 1: hits confined to every g-th row,
 * which is not a plat_t walk. Those emit an empty walk, and the kernel
 * accumulates the number of positions thereby dropped so the loss is a printed
 * number rather than a silence. With the default bkthresh = I >= J it is
 * exactly zero: g > 1 needs q | (rows), and every bucketed q exceeds J. */
__global__ void k_transform(const uint32_t *__restrict primes,
                            const uint32_t *__restrict roots,
                            plat_t *__restrict out,
                            uint32_t n, int logI, uint32_t J,
                            int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                            uint32_t *__restrict nproj,
                            unsigned long long *__restrict nlost)
{
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        const uint32_t q = primes[k];
        uint32_t rt, g, m = pl_transform_enc(q, roots[k], a0, a1, b0, b1, &rt, &g);
        if (g > 1) {                         /* rows only: emit an empty walk */
            plat_t P; P.inc_warp = 0xFFFFFFFFu; P.inc_step = PL_VERTICAL;
            P.bound_warp = 0; P.bound_step = 0;
            out[k] = P;
            atomicAdd(nproj, 1u);
            atomicAdd(nlost, (unsigned long long)((J / g) * ((1u << logI) / m)));
        } else {
            out[k] = pl_make(m, rt, logI);
        }
    }
}

/* ---- stage (a): naive single-level atomic append ---------------------- */

/* One atomicAdd per record into a 2^log_nbuckets-way split. This is the
 * baseline the design doc says to beat. */
template <int RECBYTES>
__global__ void k_fill_atomic(const plat_t *__restrict plat,
                              const uint16_t *__restrict slice,
                              uint32_t n, uint32_t xmax, int logI,
                              int log_region,
                              uint32_t *__restrict cursor,
                              uint8_t *__restrict out, uint32_t cap,
                              uint32_t *__restrict overflow)
{
    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t offmask = (1u << log_region) - 1;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
        plat_t P = plat[k];
        if (P.inc_warp == 0xFFFFFFFFu) continue;
        /* one read per prime, not per record: the slice hint is what the apply
         * kernel turns back into log p (and what resieve would use to recover
         * the prime itself), exactly as CADO's shorthint does */
        const uint32_t sl = slice[k];
        for (uint32_t x = pl_first(&P, logI); x < xmax; x = pl_next(x, &P, Imask)) {
            uint32_t b = x >> log_region;
            uint32_t slot = atomicAdd(&cursor[b], 1u);
            if (slot >= cap) { atomicAdd(overflow, 1u); continue; }
            size_t at = ((size_t)b * cap + slot) * RECBYTES;
            if (RECBYTES == 2) {
                *(uint16_t *)(out + at) = (uint16_t)(x & offmask);
            } else if (RECBYTES == 4) {
                /* 16-bit offset + 16-bit slice hint (CADO shorthint shape) */
                *(uint32_t *)(out + at) = (x & offmask) | (sl << 16);
            } else {
                *(uint64_t *)(out + at) = (uint64_t)(x & offmask) | ((uint64_t)sl << 32);
            }
        }
    }
}

/* ---- small-prime line sieve, fused into apply -------------------------- */

/* Primes below bkthresh are not bucketed: they hit so often that a record per
 * hit costs more than recomputing the entry point per region. On this job they
 * are only 1,969 factor-base entries but **2.92e9 updates per special-q, 9.4x
 * the entire bucket-sieve volume** -- and 84% of that comes from the 52 entries
 * with p < 64. So the load balance across primes, not the update count, is what
 * has to be engineered.
 *
 * The region is chosen to lie inside a single j-row (log_region <= logI), which
 * makes the entry point one multiply and one remainder: within a row, hits are
 * the arithmetic progression i == rt*j (mod p). No walk state has to be carried
 * between regions, so every block is independent.
 *
 * Three tiers, sized so each entry's hit count matches the number of threads
 * assigned to it:
 *   p <   64  (52 entries, 84% of updates): the whole block, one entry at a time
 *   p < 1024  (165 entries):                one warp per entry
 *   p >= 1024 (1752 entries, <=16 hits):    one thread per entry
 */
#define SS_BLOCK_CUT   64u
#define SS_WARP_CUT  1024u

template <int CELLBITS, int ATOMIC>
__device__ __forceinline__ void ss_add(uint32_t *S, uint32_t c, uint32_t lp)
{
    const uint32_t CPW = 32 / CELLBITS;
    uint32_t v = lp << ((c % CPW) * CELLBITS);
    if (ATOMIC) atomicAdd(&S[c / CPW], v);
    else        S[c / CPW] += v;
}

/* First offset within [0,width) whose i is congruent to rt*j (mod p). */
__device__ __forceinline__ uint32_t ss_first(uint32_t p, uint32_t rt,
                                             uint32_t j, int32_t ilo)
{
    const int32_t base = (int32_t)(((uint64_t)rt * j) % p);   /* i mod p */
    int32_t c = (base - ilo) % (int32_t)p;
    return (uint32_t)(c < 0 ? c + (int32_t)p : c);
}

template <int CELLBITS, int ATOMIC>
__device__ void sieve_small(uint32_t *S, uint32_t region, int logI, int log_region,
                            const uint32_t *__restrict sp,
                            const uint32_t *__restrict srt,
                            const uint32_t *__restrict sg,
                            const uint16_t *__restrict slp,
                            uint32_t nsmall, uint32_t nblk, uint32_t nwrp,
                            uint32_t tid, uint32_t nth)
{
    const uint32_t width = 1u << log_region;
    const uint32_t x0    = region << log_region;
    const uint32_t j     = x0 >> logI;
    const int32_t  ilo   = (int32_t)(x0 & ((1u << logI) - 1)) - (int32_t)(1u << (logI - 1));
    const uint32_t warp = tid >> 5, lane = tid & 31, nwarps = nth >> 5;

    /* Every entry is (m, rt, g): hits this row only when g | j, and then at
     * i == rt*(j/g) (mod m). g == 1 is the ordinary case; m == 1 means every
     * position in the row. See pl_transform_gen. */
    #define SS_ROW(e, first, step)                                            \
        do {                                                                  \
            const uint32_t m = sp[e], g = sg[e], lp = slp[e];                 \
            if (g > 1 && (j % g)) break;                                      \
            {                                                                 \
                const uint32_t c0 = ss_first(m, srt[e], g > 1 ? j / g : j, ilo); \
                for (uint32_t c = c0 + (first) * m; c < width; c += (step) * m)\
                    ss_add<CELLBITS,ATOMIC>(S, c, lp);                        \
            }                                                                 \
        } while (0)

    for (uint32_t e = 0; e < nblk; e++)                     /* whole block   */
        SS_ROW(e, tid, nth);
    for (uint32_t e = nblk + warp; e < nwrp; e += nwarps)   /* one warp      */
        SS_ROW(e, lane, 32u);
    for (uint32_t e = nwrp + tid; e < nsmall; e += nth)     /* one thread    */
        SS_ROW(e, 0u, 1u);
    #undef SS_ROW
}

/* ---- stage (A): apply -- accumulate logs into a shared-memory region ---- */

/* One block owns one bucket region for its entire life: it initialises the
 * cells to the log-norm bound in shared memory, accumulates every log p that
 * landed in the region, scans for survivors, and writes only the survivors
 * back. The region itself never touches global memory in either direction.
 * That is the structural advantage over CPU las, which must stream the region
 * through cache.
 *
 * Sign convention. GPU atomics are add-only, and a 16-bit half-word subtract
 * would borrow into its neighbour. So instead of "start at the norm and
 * subtract logs until small", we start at CINIT - T(x) and add, where
 * T(x) = log2|F(a,b)| - log2(q) - allowance. A position is a survivor exactly
 * when its cell reaches CINIT. Identical test, no borrow, no CAS.
 *
 * CELLBITS 16 is the doc's recommendation and is exactly correct: an
 * accumulated log cannot reach 65536. CELLBITS 8 packs four cells per word,
 * halving the shared memory a region needs -- it is measured here only to
 * price what correctness costs, since a byte cell overflows into its
 * neighbour (accumulated logs do exceed 255) and cannot be used for real.
 */
template <int CELLBITS, int ATOMIC, int NORMMODE>
__global__ void k_apply(const uint32_t *__restrict buckets,
                        const uint32_t *__restrict cnt, uint32_t cap,
                        int logI, int log_region,
                        const uint16_t *__restrict slice_logp, uint32_t nslice,
                        norm_t N, uint32_t CINIT, uint32_t THRESH, uint32_t tconst,
                        uint8_t *__restrict dump,
                        uint32_t *__restrict surv, uint32_t *__restrict nsurv,
                        uint32_t maxsurv, uint16_t *__restrict dbg_cells,
                        uint32_t dbg_region,
                        const uint32_t *__restrict sp, const uint32_t *__restrict srt,
                        const uint32_t *__restrict sg, const uint16_t *__restrict slp,
                        uint32_t nsmall, uint32_t nblk, uint32_t nwrp)
{
    extern __shared__ uint32_t sm[];
    const uint32_t ncell  = 1u << log_region;
    const uint32_t nword  = ncell / (32 / CELLBITS);
    const uint32_t CPW    = 32 / CELLBITS;          /* cells per 32-bit word */
    uint32_t *S   = sm;
    uint16_t *lut = (uint16_t *)(sm + nword);

    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = 1 << (logI - 1);
    const uint32_t xbase = b << log_region;

    for (uint32_t i = tid; i < nslice; i += nth) lut[i] = slice_logp[i];

    /* ---- init: log-norm bound, computed in shared memory ---- */
    for (uint32_t w = tid; w < nword; w += nth) {
        uint32_t word = 0;
        #pragma unroll
        for (uint32_t c = 0; c < CPW; c++) {
            uint32_t t;
            if (NORMMODE == NORM_CONST) {
                t = tconst;
            } else {
                const uint32_t x = xbase + w * CPW + c;
                const float fi = (float)((int32_t)(x & Imask) - Ihalf);
                const float fj = (float)(x >> logI);
                const float u = fmaf(N.ua, fi, N.ub * fj);
                const float v = fmaf(N.va, fi, N.vb * fj);
                float acc = N.d[N.deg], vp = 1.0f;
                #pragma unroll
                for (int k = 6; k >= 0; k--)
                    if (k < N.deg) { vp *= v; acc = fmaf(acc, u, N.d[k] * vp); }
                float s = fabsf(acc);
                /* las: S = fb_log(|F|) = floor(log2|F| * scale + 0.5).
                 * las clamps here at 255 because its cell IS a byte, and that
                 * is the whole reason `scale` exists (1.28 * 196.61 = 251.7).
                 * Our cell is 16 bits, so the ceiling is CINIT, not 255, and
                 * scale becomes a free parameter rather than a constraint --
                 * at scale 1.28 las discards 0.39 bits of resolution per
                 * position that we can keep. Clamping at 255 threw that away
                 * and, worse, would have silently flattened every norm above
                 * 255/scale into one bucket the moment anyone raised scale. */
                float lg = N.scale * (N.log2M + __log2f(fmaxf(s, 1e-30f)) - N.bias);
                int ti = (int)floorf(lg + 0.5f);
                const uint32_t TMAX = (CELLBITS == 8) ? 255u : CINIT;
                t = (ti < 0) ? 0u : ((uint32_t)ti > TMAX ? TMAX : (uint32_t)ti);
            }
            word |= (CINIT - t) << (c * CELLBITS);
        }
        S[w] = word;
    }
    __syncthreads();

    /* ---- small primes: line-sieved straight into the same shared region ---- */
    if (nsmall)
        sieve_small<CELLBITS, ATOMIC>(S, b, logI, log_region, sp, srt, sg, slp,
                                      nsmall, nblk, nwrp, tid, nth);

    /* ---- apply ---- */
    uint32_t n = cnt[b];
    if (n > cap) n = cap;
    for (uint32_t i = tid; i < n; i += nth) {
        const uint32_t r = buckets[(size_t)b * cap + i];
        const uint32_t off = r & (ncell - 1);
        const uint32_t lp  = lut[(r >> 16) & (nslice - 1)];
        const uint32_t w   = off / CPW;
        const uint32_t sh  = (off % CPW) * CELLBITS;
        if (ATOMIC) atomicAdd(&S[w], lp << sh);
        else        S[w] += lp << sh;               /* racy: speed-of-light probe */
    }
    __syncthreads();

    /* ---- threshold scan ---- */
    const uint32_t CMASK = (CELLBITS == 16) ? 0xFFFFu : 0xFFu;
    for (uint32_t w = tid; w < nword; w += nth) {
        const uint32_t word = S[w];
        #pragma unroll
        for (uint32_t c = 0; c < CPW; c++) {
            const uint32_t v = (word >> (c * CELLBITS)) & CMASK;
            if (v >= THRESH) {
                uint32_t at = atomicAdd(nsurv, 1u);
                if (at < maxsurv) surv[at] = xbase + w * CPW + c;
            }
            if (dump) {
                /* las's byte: S = max(T - sum(logp), 0), and our cell holds
                 * CINIT - T + sum, so S = CINIT - cell. */
                int32_t sv = (int32_t)CINIT - (int32_t)v;
                dump[xbase + w * CPW + c] =
                    (uint8_t)(sv < 0 ? 0 : (sv > 255 ? 255 : sv));
            }
        }
    }

    /* ---- optional: dump one region for the CPU cross-check ---- */
    if (dbg_cells && b == dbg_region && CELLBITS == 16)
        for (uint32_t w = tid; w < nword; w += nth) {
            dbg_cells[2 * w]     = (uint16_t)(S[w] & 0xFFFFu);
            dbg_cells[2 * w + 1] = (uint16_t)(S[w] >> 16);
        }
}

/* ---- stage (c) level 1: shared-memory staged split into super-buckets -- */

/* NBUF open output streams per block. The doc's rule: fan-out per pass is
 * bounded by how many buffers fit in shared memory, and each flush is one
 * atomic and one full-cache-line store, not one atomic per record. */
#define L1_NBUF   128
#define L1_CAP     64        /* uint32 slots = 256 B; 128*64*4 = 32 KB static smem */
#define L1_RUNMAX   8        /* max positions aggregated per reservation   */
#define L1_FLUSH  (L1_CAP - L1_RUNMAX)

/* Run-aggregated variant. A (p,r) walk emits positions in monotone
 * increasing order, so consecutive hits usually land in the SAME
 * super-bucket -- for the smallest bucket-sieved primes, which produce most
 * of the volume, many in a row do. Collect a run and reserve it with ONE
 * shared atomic, which also cuts the number of block barriers by the same
 * factor. The barrier-per-record version this replaces spent 30% of its
 * issue slots stalled on barriers at 7.5% of DRAM peak. */
__global__ __launch_bounds__(512, 3)   /* 3 x 33 KB = 99 KB of 100 KB; 1536 thr = 100% occ */
void k_fill_l1(const plat_t *__restrict plat,
               uint32_t n, uint32_t xmax, int logI, int log_super,
               uint32_t *__restrict cursor,
               uint32_t *__restrict out, uint32_t cap,
               uint32_t *__restrict overflow)
{
    __shared__ uint32_t buf[L1_NBUF * L1_CAP];
    __shared__ uint32_t cnt[L1_NBUF];
    __shared__ uint32_t base[L1_NBUF];

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    for (uint32_t i = tid; i < L1_NBUF; i += nth) cnt[i] = 0;
    __syncthreads();

    const uint32_t stride = gridDim.x * nth;
    uint32_t k = blockIdx.x * nth + tid;
    plat_t P; P.inc_warp = 0xFFFFFFFFu;
    uint32_t x = 0;
    int active = 0;

    for (;;) {
        if (!active) {                       /* pick up the next prime */
            while (k < n) {
                P = plat[k];
                if (P.inc_warp != 0xFFFFFFFFu) { x = pl_first(&P, logI); active = (x < xmax); }
                k += stride;
                if (active) break;
            }
        }
        if (!__syncthreads_or(active)) break;

        if (active) {
            /* gather a run of consecutive positions in one super-bucket */
            uint32_t run[L1_RUNMAX];
            const uint32_t b = x >> log_super;
            uint32_t nrun = 0, xn = x;
            do {
                run[nrun++] = xn;
                xn = pl_next(xn, &P, Imask);
            } while (nrun < L1_RUNMAX && xn < xmax && (xn >> log_super) == b);

            uint32_t slot = atomicAdd(&cnt[b], nrun);
            if (slot + nrun <= L1_CAP) {
                for (uint32_t t = 0; t < nrun; t++) buf[b * L1_CAP + slot + t] = run[t];
                x = xn;
                active = (x < xmax);
            } else {
                atomicSub(&cnt[b], nrun);    /* keep cnt exact; retry after flush */
            }
        }
        __syncthreads();

        /* flush buffers at/over the high-water mark. One warp per buffer. */
        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < L1_NBUF; b += nwarp) {
            uint32_t c = cnt[b];
            if (c < L1_FLUSH) continue;
            if (lane == 0) base[b] = atomicAdd(&cursor[b], c);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + c <= cap) {
                for (uint32_t t = lane; t < c; t += 32)
                    out[(size_t)b * cap + dst + t] = buf[b * L1_CAP + t];
            } else if (lane == 0) atomicAdd(overflow, c);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
        __syncthreads();
    }

    /* drain partial buffers */
    {
        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < L1_NBUF; b += nwarp) {
            uint32_t c = cnt[b];
            if (c == 0) continue;
            if (lane == 0) base[b] = atomicAdd(&cursor[b], c);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + c <= cap) {
                for (uint32_t t = lane; t < c; t += 32)
                    out[(size_t)b * cap + dst + t] = buf[b * L1_CAP + t];
            } else if (lane == 0) atomicAdd(overflow, c);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
    }
}

/* ---- stage (c) level 2: split one super-bucket into regions ----------- */

#define L2_NBUF 128
#define L2_CAP   64

template <int RECBYTES>
__global__ __launch_bounds__(512, 3)
void k_fill_l2(const uint32_t *__restrict in, const uint32_t *__restrict incnt,
               uint32_t in_cap, int log_region, int log_super,
               uint32_t *__restrict cursor, uint8_t *__restrict out,
               uint32_t cap, uint32_t *__restrict overflow)
{
    __shared__ uint32_t buf[L2_NBUF * L2_CAP];
    __shared__ uint32_t cnt[L2_NBUF];
    __shared__ uint32_t base[L2_NBUF];

    const uint32_t sb = blockIdx.x;              /* one block per super-bucket */
    uint32_t nrec = incnt[sb];
    if (nrec > in_cap) nrec = in_cap;   /* L1 cursor may overrun on overflow */
    const uint32_t offmask = (1u << log_region) - 1;
    const uint32_t regions_per_super = 1u << (log_super - log_region);
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    for (uint32_t i = tid; i < L2_NBUF; i += nth) cnt[i] = 0;
    __syncthreads();

    uint32_t idx = tid;
    uint32_t pending = 0, prec = 0;
    for (;;) {
        int have = 0;
        if (pending) { have = 1; }
        else if (idx < nrec) { prec = in[(size_t)sb * in_cap + idx]; have = 1; }
        if (!__syncthreads_or(have)) break;

        if (have) {
            uint32_t b = (prec >> log_region) & (regions_per_super - 1);
            uint32_t slot = atomicAdd(&cnt[b], 1u);
            if (slot < L2_CAP) {
                buf[b * L2_CAP + slot] = prec;
                pending = 0; idx += nth;
            } else pending = 1;
        }
        __syncthreads();

        const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
        for (uint32_t b = warp; b < regions_per_super; b += nwarp) {
            uint32_t c = cnt[b];
            if (c < L2_CAP) continue;
            uint32_t gb = sb * regions_per_super + b;
            if (lane == 0) base[b] = atomicAdd(&cursor[gb], L2_CAP);
            __syncwarp();
            uint32_t dst = base[b];
            if (dst + L2_CAP <= cap) {
                for (uint32_t t = lane; t < L2_CAP; t += 32) {
                    uint32_t v = buf[b * L2_CAP + t];
                    size_t at = ((size_t)gb * cap + dst + t) * RECBYTES;
                    if (RECBYTES == 2) *(uint16_t *)(out + at) = (uint16_t)(v & offmask);
                    else               *(uint32_t *)(out + at) = (v & offmask);
                }
            } else if (lane == 0) atomicAdd(overflow, L2_CAP);
            __syncwarp();
            if (lane == 0) cnt[b] = 0;
        }
        __syncthreads();
    }

    const uint32_t warp = tid >> 5, lane = tid & 31, nwarp = nth >> 5;
    for (uint32_t b = warp; b < regions_per_super; b += nwarp) {
        uint32_t c = cnt[b];
        if (c == 0) continue;
        if (c > L2_CAP) c = L2_CAP;
        uint32_t gb = sb * regions_per_super + b;
        if (lane == 0) base[b] = atomicAdd(&cursor[gb], c);
        __syncwarp();
        uint32_t dst = base[b];
        if (dst + c <= cap) {
            for (uint32_t t = lane; t < c; t += 32) {
                uint32_t v = buf[b * L2_CAP + t];
                size_t at = ((size_t)gb * cap + dst + t) * RECBYTES;
                if (RECBYTES == 2) *(uint16_t *)(out + at) = (uint16_t)(v & offmask);
                else               *(uint32_t *)(out + at) = (v & offmask);
            }
        } else if (lane == 0) atomicAdd(overflow, c);
        __syncwarp();
    }
}

/* ---- host driver ------------------------------------------------------ */

struct dev_bufs {
    uint32_t *primes, *roots, *cursor, *overflow, *nproj, *l1, *l1cnt;
    unsigned long long *nlost;
    plat_t   *plat;
    uint8_t  *out;
    uint16_t *slice, *slice_logp;
    uint32_t *surv, *nsurv;
    uint16_t *dbg;
    uint32_t *sp, *srt, *sg;
    uint16_t *slp;
    uint8_t  *dumpbuf;
};

static float time_kernel(cudaEvent_t a, cudaEvent_t b)
{ float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }

/* Cut the (sorted) factor base into slices of constant log p. This is CADO's
 * scheme: the bucket record carries a slice index, and log p is a property of
 * the slice, so the apply kernel needs only a tiny shared-memory table. Slices
 * are capped so the count stays a power of two we can mask with. */
static uint32_t build_slices(const fb_t *fb, uint16_t *slice,
                             uint16_t **logp_tab, uint32_t *nslice_pow2)
{
    uint32_t ns = 0, k;
    uint16_t *tab = (uint16_t *)malloc(65536 * 2);
    int cur = -1;
    for (k = 0; k < fb->n; k++) {
        int lp = fb->logp[k];        /* las's per-ideal increment, already scaled */
        if (lp != cur || (ns && (k % 262144u) == 0)) {
            cur = lp; tab[ns++] = (uint16_t)lp;
        }
        slice[k] = (uint16_t)(ns - 1);
    }
    { uint32_t p2 = 1; while (p2 < ns) p2 <<= 1; *nslice_pow2 = p2;
      for (uint32_t i = ns; i < p2; i++) tab[i] = 0; }
    *logp_tab = tab;
    return ns;
}

extern "C" int run_bench(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                         const poly_t *POLY, const bench_cfg_t *cfg)
{
    const uint32_t I = 1u << cfg->logI;
    const uint32_t xmax = I * cfg->J;
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const int log_super = log_region + 7;             /* 128 regions/super */
    const uint32_t nsuper = xmax >> log_super;

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));
    printf("  device memory: %.2f GB free of %.2f GB\n",
           freeB / 1073741824.0, totalB / 1073741824.0);

    dev_bufs D; memset(&D, 0, sizeof(D));
    CK(cudaMalloc(&D.primes, (size_t)fb->n * 4));
    CK(cudaMalloc(&D.roots,  (size_t)fb->n * 4));
    CK(cudaMalloc(&D.plat,   (size_t)fb->n * sizeof(plat_t)));
    CK(cudaMalloc(&D.overflow, 4));
    CK(cudaMalloc(&D.nproj, 4));
    CK(cudaMalloc(&D.nlost, 8));
    CK(cudaMemcpy(D.primes, fb->primes, (size_t)fb->n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.roots,  fb->roots,  (size_t)fb->n * 4, cudaMemcpyHostToDevice));

    /* ---- slices: bucket record hint -> log p ---- */
    uint16_t *hslice = (uint16_t *)malloc((size_t)fb->n * 2), *hlogp = NULL;
    uint32_t nslice_pow2 = 1;
    uint32_t nslice = build_slices(fb, hslice, &hlogp, &nslice_pow2);
    printf("  factor base cut into %u slices (padded to %u), log p in [%u,%u] bits\n",
           nslice, nslice_pow2, hlogp[0], hlogp[nslice - 1]);
    CK(cudaMalloc(&D.slice, (size_t)fb->n * 2));
    CK(cudaMalloc(&D.slice_logp, (size_t)nslice_pow2 * 2));
    CK(cudaMemcpy(D.slice, hslice, (size_t)fb->n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.slice_logp, hlogp, (size_t)nslice_pow2 * 2, cudaMemcpyHostToDevice));

    /* ---- small primes: transform on the host (a few thousand entries) and
     * split into the three load-balance tiers. Entries arrive sorted by p, so
     * the tier boundaries are just two indices. ---- */
    uint32_t nsmall = 0, nblk = 0, nwrp = 0;
    uint32_t *hsp = NULL, *hsrt = NULL, *hsg = NULL; uint16_t *hslp = NULL;
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t i, k = 0, nrow = 0, nprj = 0;
        hsp  = (uint32_t *)malloc((size_t)fbs->n * 4);
        hsrt = (uint32_t *)malloc((size_t)fbs->n * 4);
        hslp = (uint16_t *)malloc((size_t)fbs->n * 2);
        hsg = (uint32_t *)malloc((size_t)fbs->n * 4);
        for (i = 0; i < fbs->n; i++) {
            uint32_t q = fbs->primes[i], r = fbs->roots[i], rt, g, m;
            m = pl_transform_enc(q, r, L->a0, L->a1, L->b0, L->b1, &rt, &g);
            if (g > 1) nrow++;
            if (r >= q) nprj++;
            hsp[k] = m; hsrt[k] = rt; hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        nsmall = k;
        /* Tier by the EFFECTIVE modulus m, not by q: an entry with q = 32768
         * and g = 32768 has m = 1 and hits every position in its rows, so
         * leaving it in the thread-per-entry tier would hand one thread the
         * whole region. Sorting by m puts every entry in the tier sized for
         * the number of hits it actually produces. */
        for (i = 1; i < nsmall; i++) {
            uint32_t mm = hsp[i], rr = hsrt[i], gg = hsg[i]; uint16_t ll = hslp[i];
            int32_t z = (int32_t)i - 1;
            while (z >= 0 && hsp[z] > mm) {
                hsp[z+1] = hsp[z]; hsrt[z+1] = hsrt[z];
                hsg[z+1] = hsg[z]; hslp[z+1] = hslp[z]; z--;
            }
            hsp[z+1] = mm; hsrt[z+1] = rr; hsg[z+1] = gg; hslp[z+1] = ll;
        }
        for (i = 0; i < nsmall && hsp[i] < SS_BLOCK_CUT; i++) nblk = i + 1;
        for (i = 0; i < nsmall && hsp[i] < SS_WARP_CUT;  i++) nwrp = i + 1;
        CK(cudaMalloc(&D.sp,  (size_t)nsmall * 4));
        CK(cudaMalloc(&D.srt, (size_t)nsmall * 4));
        CK(cudaMalloc(&D.sg,  (size_t)nsmall * 4));
        CK(cudaMalloc(&D.slp, (size_t)nsmall * 2));
        CK(cudaMemcpy(D.sp,  hsp,  (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.srt, hsrt, (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.sg,  hsg,  (size_t)nsmall * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(D.slp, hslp, (size_t)nsmall * 2, cudaMemcpyHostToDevice));
        {   double upd = 0; uint32_t xm = (1u << cfg->logI) * cfg->J;
            for (i = 0; i < nsmall; i++) upd += (double)xm / hsp[i] / hsg[i];
            printf("  small sieve: %u entries (%u block-tier m<%u, %u warp-tier m<%u,"
                   " %u thread-tier), %u with a row divisor, %u projective,"
                   " %.3e updates\n",
                   nsmall, nblk, SS_BLOCK_CUT, nwrp - nblk, SS_WARP_CUT,
                   nsmall - nwrp, nrow, nprj, upd);
        }
    }

    /* ---- norm initialisation constants ---- */
    norm_t N; memset(&N, 0, sizeof(N));
    norm_setup(&N, POLY, L, cfg->logI, cfg->J, cfg->scale, cfg->side == 1);
    const uint32_t CINIT = (cfg->cell_bits == 16) ? 4096u : 255u;
    /* las's survivor test is S = max(T - sum, 0) <= bound with
     * bound = round(scale * lambda * lpb). Ours holds CINIT - T + sum, so
     * S = CINIT - cell and the test becomes cell >= CINIT - bound. */
    const uint32_t BOUND = (uint32_t)(cfg->scale * cfg->allowance + 0.5);
    uint32_t tconst;
    {
        float t = norm_target_host(&N, 0, cfg->J / 2);
        int ti = (int)(t + 0.5f);
        tconst = (ti < 1) ? 1u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
        printf("  init T at (i=0, j=J/2) = %u; survivor bound = %u"
               " (scale %.3f x %.2f bits)\n", tconst, BOUND, cfg->scale, cfg->allowance);
    }

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4);

    int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    float t_trans = 0, t_fill = 0, t_l1 = 0, t_l2 = 0;

    /* ---- stage T ---- */
    CK(cudaMemset(D.nproj, 0, 4));
    CK(cudaMemset(D.nlost, 0, 8));
    cudaEventRecord(e0);
    for (int rep = 0; rep < cfg->reps; rep++)
        k_transform<<<blocks, cfg->threads>>>(D.primes, D.roots, D.plat, fb->n,
            cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, D.nproj, D.nlost);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    t_trans = time_kernel(e0, e1) / cfg->reps;

    uint32_t hproj = 0;
    unsigned long long hlost = 0;
    CK(cudaMemcpy(&hproj, D.nproj, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hlost, D.nlost, 8, cudaMemcpyDeviceToHost));
    hproj /= cfg->reps;
    hlost /= (unsigned)cfg->reps;

    /* ---- expected record count, for sizing ---- */
    double exp_rec = 0;
    for (uint32_t i = 0; i < fb->n; i++) exp_rec += (double)xmax / fb->primes[i];
    uint64_t est = (uint64_t)(exp_rec * 1.15) + 4096;

    printf("  transformed roots: %u row-confined (g > 1), %llu positions lost%s\n",
           hproj, hlost, hlost ? "  ** move these to the small tier **" : "");
    printf("  analytic records : %.3e   (sized with 1.15x margin)\n", exp_rec);

    uint32_t cap = 0;               /* records per region, set by the fill path */
    if (cfg->fill_mode == FILL_ATOMIC) {
        cap = (uint32_t)(est / nregion) + 256;
        size_t need = (size_t)nregion * cap * cfg->record_bytes;
        printf("  single-level: %u buckets x cap %u x %d B = %.2f GB\n",
               nregion, cap, cfg->record_bytes, need / 1073741824.0);
        if (need + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(cudaMalloc(&D.cursor, (size_t)nregion * 4));
        CK(cudaMalloc(&D.out, need));
        CK(cudaMemset(D.overflow, 0, 4));
        cudaEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
            if (cfg->record_bytes == 2)
                k_fill_atomic<2><<<blocks, cfg->threads>>>(D.plat, D.slice, fb->n, xmax,
                    cfg->logI, log_region, D.cursor, D.out, cap, D.overflow);
            else if (cfg->record_bytes == 4)
                k_fill_atomic<4><<<blocks, cfg->threads>>>(D.plat, D.slice, fb->n, xmax,
                    cfg->logI, log_region, D.cursor, D.out, cap, D.overflow);
            else
                k_fill_atomic<8><<<blocks, cfg->threads>>>(D.plat, D.slice, fb->n, xmax,
                    cfg->logI, log_region, D.cursor, D.out, cap, D.overflow);
        }
        cudaEventRecord(e3);
        CK(cudaEventSynchronize(e3));
        CK(cudaGetLastError());
        t_fill = time_kernel(e2, e3) / cfg->reps;
    } else {
        /* Both levels stage their fan-out in a fixed number of shared buffers,
         * so the split has to fit: L1 needs nsuper <= L1_NBUF and L2 needs
         * regions_per_super <= L2_NBUF. With 128 each, two-level tops out at
         * 128*128 = 16384 regions -- exactly I15e at region 2^15, and no more.
         * Past that it silently indexed cnt[] out of bounds and livelocked in
         * the retry loop. Refuse instead: the operating point that actually
         * won (region 2^14, 32768 regions) is out of reach for a two-level
         * split at these buffer sizes and would need a third level. */
        if (nsuper > L1_NBUF || (1u << (log_super - log_region)) > L2_NBUF) {
            printf("  two-level cannot express this split: %u super-buckets"
                   " (max %u) x %u regions each (max %u).\n"
                   "  Use --mode atomic, which is 2.7x faster anyway"
                   " (RESULTS.md finding 1).\n",
                   nsuper, L1_NBUF, 1u << (log_super - log_region), L2_NBUF);
            return 1;
        }
        uint32_t l1cap = (uint32_t)(est / nsuper) + 4096;
        uint32_t l2cap = (uint32_t)(est / nregion) + 256;
        size_t need1 = (size_t)nsuper * l1cap * 4;
        size_t need2 = (size_t)nregion * l2cap * cfg->record_bytes;
        printf("  two-level: L1 %u super x cap %u x 4 B = %.2f GB;"
               " L2 %u regions x cap %u x %d B = %.2f GB\n",
               nsuper, l1cap, need1 / 1073741824.0,
               nregion, l2cap, cfg->record_bytes, need2 / 1073741824.0);
        if (need1 + need2 + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(cudaMalloc(&D.l1, need1));
        CK(cudaMalloc(&D.l1cnt, (size_t)nsuper * 4));
        CK(cudaMalloc(&D.cursor, (size_t)nregion * 4));
        CK(cudaMalloc(&D.out, need2));
        CK(cudaMemset(D.overflow, 0, 4));

        cudaEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.l1cnt, 0, (size_t)nsuper * 4));
            k_fill_l1<<<blocks, 512>>>(D.plat, fb->n, xmax, cfg->logI, log_super,
                D.l1cnt, D.l1, l1cap, D.overflow);
        }
        cudaEventRecord(e3);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(cudaMemset(D.cursor, 0, (size_t)nregion * 4));
            if (cfg->record_bytes == 2)
                k_fill_l2<2><<<nsuper, 512>>>(D.l1, D.l1cnt, l1cap, log_region,
                    log_super, D.cursor, D.out, l2cap, D.overflow);
            else
                k_fill_l2<4><<<nsuper, 512>>>(D.l1, D.l1cnt, l1cap, log_region,
                    log_super, D.cursor, D.out, l2cap, D.overflow);
        }
        cudaEventRecord(e4);
        CK(cudaEventSynchronize(e4));
        CK(cudaGetLastError());
        t_l1 = time_kernel(e2, e3) / cfg->reps;
        t_l2 = time_kernel(e3, e4) / cfg->reps;
        t_fill = t_l1 + t_l2;
    }

    uint32_t ovf = 0;
    CK(cudaMemcpy(&ovf, D.overflow, 4, cudaMemcpyDeviceToHost));

    /* ---- count what actually landed ---- */
    uint64_t landed = 0;
    {
        uint32_t *h = (uint32_t *)malloc((size_t)nregion * 4);
        CK(cudaMemcpy(h, D.cursor, (size_t)nregion * 4, cudaMemcpyDeviceToHost));
        uint32_t mx = 0;
        for (uint32_t i = 0; i < nregion; i++) { landed += h[i]; if (h[i] > mx) mx = h[i]; }
        printf("  records landed   : %llu   (max bucket %u, mean %.0f -> imbalance %.2fx)\n",
               (unsigned long long)landed, mx, (double)landed / nregion,
               mx / ((double)landed / nregion));
        free(h);
    }
    if (ovf) printf("  ** OVERFLOW: %u records dropped -- resize and re-run **\n", ovf);

    /* ---- stage A: apply ---- */
    float t_apply = 0;
    if (cfg->stage != STAGE_FILL) {
        if (cfg->fill_mode != FILL_ATOMIC || cfg->record_bytes != 4) {
            printf("\n  apply needs single-level 4 B records (--mode atomic"
                   " --record-bytes 4); skipping\n");
        } else {
            const int CB = cfg->cell_bits;
            const uint32_t ncell = 1u << log_region;
            const size_t smem = (size_t)ncell * CB / 8 + (size_t)nslice_pow2 * 2;
            const uint32_t maxsurv = 1u << 22;
            int athr = cfg->apply_threads ? cfg->apply_threads : 512;
            /* region 0 is the j=0 row and is legitimately almost empty --
             * gating on it would check nothing. Use a mid-range region. */
            const uint32_t dbgreg = nregion / 2;

            CK(cudaMalloc(&D.surv, (size_t)maxsurv * 4));
            CK(cudaMalloc(&D.nsurv, 4));
            if (cfg->dump) {
                CK(cudaMalloc(&D.dumpbuf, (size_t)xmax));
                printf("  dumping the region in las byte convention to %s"
                       " (%.0f MB)\n", cfg->dump, xmax / 1048576.0);
            }
            if (cfg->verify) CK(cudaMalloc(&D.dbg, (size_t)ncell * 2));

            printf("\n  apply: %u regions x %u cells x %d bit = %zu B smem/block,"
                   " %d threads, %s, norm=%s\n",
                   nregion, ncell, CB, smem, athr,
                   cfg->apply_atomic ? "smem atomicAdd" : "PLAIN (racy probe)",
                   cfg->norm_mode == NORM_CONST ? "const" : "horner");
            if (smem > 101376) {
                printf("  SKIP: %zu B exceeds the 101376 B opt-in limit\n", smem);
                goto after_apply;
            }

#define LAUNCH_APPLY(CBV, AT, NM)                                              \
            do {                                                               \
                CK(cudaFuncSetAttribute(k_apply<CBV, AT, NM>,                  \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));  \
                cudaEventRecord(e3);                                           \
                for (int rep = 0; rep < cfg->reps; rep++) {                    \
                    CK(cudaMemset(D.nsurv, 0, 4));                             \
                    k_apply<CBV, AT, NM><<<nregion, athr, smem>>>(             \
                        (const uint32_t *)D.out, D.cursor, cap,                \
                        cfg->logI, log_region, D.slice_logp, nslice_pow2,      \
                        N, CINIT, CINIT - BOUND, tconst, D.dumpbuf,            \
                        D.surv, D.nsurv, maxsurv,                              \
                        D.dbg, dbgreg, D.sp, D.srt, D.sg, D.slp,                     \
                        nsmall, nblk, nwrp);                                   \
                }                                                              \
                cudaEventRecord(e4);                                           \
                CK(cudaEventSynchronize(e4));                                  \
                CK(cudaGetLastError());                                        \
                t_apply = time_kernel(e3, e4) / cfg->reps;                      \
            } while (0)

            if (CB == 16) {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 1, NORM_CONST);
                    else                              LAUNCH_APPLY(16, 1, NORM_HORNER);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 0, NORM_CONST);
                    else                              LAUNCH_APPLY(16, 0, NORM_HORNER);
                }
            } else {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 1, NORM_CONST);
                    else                              LAUNCH_APPLY(8, 1, NORM_HORNER);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 0, NORM_CONST);
                    else                              LAUNCH_APPLY(8, 0, NORM_HORNER);
                }
            }
#undef LAUNCH_APPLY

            if (cfg->dump) {
                uint8_t *h = (uint8_t *)malloc((size_t)xmax);
                FILE *fo = fopen(cfg->dump, "wb");
                CK(cudaMemcpy(h, D.dumpbuf, (size_t)xmax, cudaMemcpyDeviceToHost));
                if (!fo) { perror(cfg->dump); }
                else { fwrite(h, 1, (size_t)xmax, fo); fclose(fo); }
                free(h);
            }
            uint32_t hs = 0;
            CK(cudaMemcpy(&hs, D.nsurv, 4, cudaMemcpyDeviceToHost));
            printf("  survivors: %u of %u positions (1 in %.3e)%s\n", hs, xmax,
                   hs ? (double)xmax / hs : 0.0,
                   hs > maxsurv ? "  ** list truncated **" : "");

            if (cfg->verify && CB == 16) {
                uint32_t *hrec = (uint32_t *)malloc((size_t)cap * 4);
                uint32_t *hcnt = (uint32_t *)malloc((size_t)nregion * 4);
                uint16_t *hgpu = (uint16_t *)malloc((size_t)ncell * 2);
                uint16_t *href = (uint16_t *)malloc((size_t)ncell * 2);
                CK(cudaMemcpy(hcnt, D.cursor, (size_t)nregion * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hrec, D.out + (size_t)dbgreg * cap * 4, (size_t)cap * 4,
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hgpu, D.dbg, (size_t)ncell * 2, cudaMemcpyDeviceToHost));
                uint32_t nr = hcnt[dbgreg] > cap ? cap : hcnt[dbgreg];
                uint32_t rs = verify_apply_region(hrec, nr, hlogp, &N, cfg->logI,
                        log_region, dbgreg, cfg->norm_mode, CINIT, tconst,
                        hsp, hsrt, hsg, hslp, nsmall, href);
                uint32_t bad = 0, first = 0xFFFFFFFFu;
                for (uint32_t i = 0; i < ncell; i++)
                    if (hgpu[i] != href[i]) { if (!bad) first = i; bad++; }
                printf("  [verify] region %u: %u records replayed on CPU, %u cells differ",
                       dbgreg, nr, bad);
                if (bad) printf("  (first at cell %u: gpu %u ref %u)",
                                first, hgpu[first], href[first]);
                printf("\n  [verify] region %u survivors: cpu %u\n", dbgreg, rs);
                free(hrec); free(hcnt); free(hgpu); free(href);
            }
        }
    }
after_apply:

    printf("\n  %-26s %8.3f ms\n", "transform + plattice (T)", t_trans);
    if (cfg->fill_mode == FILL_ATOMIC)
        printf("  %-26s %8.3f ms\n", "fill: atomic single-level", t_fill);
    else {
        printf("  %-26s %8.3f ms\n", "fill L1: -> super-buckets", t_l1);
        printf("  %-26s %8.3f ms\n", "fill L2: -> regions", t_l2);
        printf("  %-26s %8.3f ms\n", "fill total", t_fill);
    }
    if (t_apply > 0)
        printf("  %-26s %8.3f ms\n", "apply (init+add+scan)", t_apply);
    printf("  %-26s %8.3f ms  <-- vs 182 ms (tie CPU) / 56 ms (TD floor)\n",
           "SIEVE CHAIN ms/special-q", t_trans + t_fill + t_apply);
    if (landed) {
        printf("  %-26s %8.2f\n", "ns per record (fill)", t_fill * 1e6 / landed);
        if (t_apply > 0)
            printf("  %-26s %8.2f\n", "ns per record (apply)", t_apply * 1e6 / landed);
    }

    free(hslice); free(hlogp); free(hsp); free(hsrt); free(hsg); free(hslp);
    cudaFree(D.primes); cudaFree(D.roots); cudaFree(D.plat); cudaFree(D.cursor);
    cudaFree(D.out); cudaFree(D.overflow); cudaFree(D.nproj); cudaFree(D.nlost);
    cudaFree(D.l1); cudaFree(D.l1cnt); cudaFree(D.slice); cudaFree(D.slice_logp);
    cudaFree(D.surv); cudaFree(D.nsurv); cudaFree(D.dbg);
    cudaFree(D.sp); cudaFree(D.srt); cudaFree(D.sg); cudaFree(D.slp); cudaFree(D.dumpbuf);
    return 0;
}
