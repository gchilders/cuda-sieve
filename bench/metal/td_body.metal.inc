/* Optional buffer arguments: metal_rt supplies bit i = "argument i is bound". */
constant uint mtl_bound_mask [[function_constant(0)]];
constant bool mtl_bound_2 = (mtl_bound_mask & (1u << 2)) != 0;
constant bool mtl_bound_3 = (mtl_bound_mask & (1u << 3)) != 0;
constant bool mtl_bound_14 = (mtl_bound_mask & (1u << 14)) != 0;
constant bool mtl_bound_16 = (mtl_bound_mask & (1u << 16)) != 0;
constant bool mtl_bound_17 = (mtl_bound_mask & (1u << 17)) != 0;
constant bool mtl_bound_18 = (mtl_bound_mask & (1u << 18)) != 0;


/* Move the direct-test congruence origin forward by delta_j global rows. This
 * is deliberately a separate tiny kernel: it performs one 64-bit modulo per
 * small-prime ENTRY per slab so the billions-of-tests k_td loop stays on its
 * original 32-bit multiply/high-multiply sequence. */
kernel void k_tdsmall_advance(
    device tdsmall_t * sm [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    constant uint32_t &delta_j [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        tdsmall_t v = sm[t];
        if (v.magic) v.cst = slab_phase_cst(v.cst, v.rt, v.m, delta_j);
        sm[t] = v;
    }
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
static inline void td_divide_out(thread bn_t *N, uint32_t p, uint64_t recip,
                                              device uint32_t *fac, thread uint32_t *nf,
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
kernel void k_cand_stats(
    constant uint32_t &n [[buffer(0)]],
    device const uint8_t * bits0 [[buffer(1)]],
    device const uint8_t * bits1 [[buffer(2)]],
    device const uint32_t * fn0 [[buffer(3)]],
    device const uint32_t * fn1 [[buffer(4)]],
    constant uint32_t &lpb0 [[buffer(5)]],
    constant uint32_t &lpb1 [[buffer(6)]],
    constant uint32_t &fmax [[buffer(7)]],
    device uint32_t * out [[buffer(8)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
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

static inline uint32_t td_rank(const device uint32_t * bits,
                                            const device uint32_t * gbase,
                                            uint32_t x)
{
    const uint32_t wx = x >> 5;
    const uint32_t w0 = (x / TD_GROUP_X) * TD_GROUP_W;
    uint32_t r = gbase[x / TD_GROUP_X];
    for (uint32_t w = w0; w < wx; w++) r += __popc(bits[w]);
    return r + __popc(bits[wx] & ((1u << (x & 31u)) - 1u));
}

kernel void k_group_counts(
    device const uint32_t * bits [[buffer(0)]],
    constant uint32_t &ngroup [[buffer(1)]],
    device uint32_t * cnt [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t gg = bench_grid_thread_x(); gg < ngroup; gg += stride) {
        const uint32_t g = (uint32_t)gg;
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

kernel void k_scan_pass1(
    device const uint32_t * in [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t * out [[buffer(2)]],
    device uint32_t * bsum [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    threadgroup uint32_t s[TD_SCAN_BLK];
    const uint64_t ii = (uint64_t)blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    const uint32_t i = (uint32_t)ii;
    s[threadIdx.x] = (ii < n) ? in[i] : 0u;
    __syncthreads();
    for (uint32_t off = 1; off < TD_SCAN_BLK; off <<= 1) {
        uint32_t v = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += v;
        __syncthreads();
    }
    if (ii < n) out[i] = s[threadIdx.x] - in[i];   /* exclusive */
    if (threadIdx.x == TD_SCAN_BLK - 1) bsum[blockIdx.x] = s[threadIdx.x];
}


/* One block scans the block totals in place. nb is at most a few thousand. */
kernel void k_scan_pass2(
    device uint32_t * bsum [[buffer(0)]],
    constant uint32_t &nb [[buffer(1)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    threadgroup uint32_t s[1024];
    threadgroup uint32_t running;
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


kernel void k_scan_pass3(
    device uint32_t * out [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device const uint32_t * bsum [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t ii = (uint64_t)blockIdx.x * TD_SCAN_BLK + threadIdx.x;
    if (ii < n) out[(uint32_t)ii] += bsum[blockIdx.x];
}


/* ---- rank-ordered compaction ------------------------------------------- */

/* Same output as k_intersect_compact's atomic path, but at a deterministic
 * index: survivor s is the s'th set bit of the two-sided bitmap. Two things
 * come from that. The resieve can find a position's slot with td_rank instead
 * of a sort, and the emitted list is reproducible run to run, which is what
 * makes a bit-identical regression test on it mean anything. */
template <bool SLABBED>
static inline void k_emit_ranked_body(
    device const uint32_t * bits,
    device const uint32_t * gbase,
    constant uint32_t &nword,
    constant uint32_t &logI,
    constant int64_t &a0,
    constant int64_t &a1,
    constant int64_t &b0,
    constant int64_t &b1,
    device uint32_t * out_x,
    device int64_t * out_a,
    device int64_t * out_b,
    constant uint32_t &cap,
    constant uint32_t &j_base,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint64_t stride = bench_grid_stride_x();

    for (uint64_t ww = bench_grid_thread_x(); ww < nword; ww += stride) {
        const uint32_t w = (uint32_t)ww;
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
            const uint32_t jlocal = x >> logI;
            uint32_t j = jlocal;
            if constexpr (SLABBED) j += j_base;
            if (slot < cap) {
                out_x[slot] = x;
                out_a[slot] = (int64_t)i * a0 + (int64_t)j * b0;
                out_b[slot] = (int64_t)i * a1 + (int64_t)j * b1;
            }
            slot++;
        }
    }
}

kernel void k_emit_ranked_0(
    device const uint32_t * bits [[buffer(0)]],
    device const uint32_t * gbase [[buffer(1)]],
    constant uint32_t &nword [[buffer(2)]],
    constant uint32_t &logI [[buffer(3)]],
    constant int64_t &a0 [[buffer(4)]],
    constant int64_t &a1 [[buffer(5)]],
    constant int64_t &b0 [[buffer(6)]],
    constant int64_t &b1 [[buffer(7)]],
    device uint32_t * out_x [[buffer(8)]],
    device int64_t * out_a [[buffer(9)]],
    device int64_t * out_b [[buffer(10)]],
    constant uint32_t &cap [[buffer(11)]],
    constant uint32_t &j_base [[buffer(12)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_emit_ranked_body<false>(bits, gbase, nword, logI, a0, a1, b0, b1, out_x, out_a, out_b, cap, j_base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_emit_ranked_1(
    device const uint32_t * bits [[buffer(0)]],
    device const uint32_t * gbase [[buffer(1)]],
    constant uint32_t &nword [[buffer(2)]],
    constant uint32_t &logI [[buffer(3)]],
    constant int64_t &a0 [[buffer(4)]],
    constant int64_t &a1 [[buffer(5)]],
    constant int64_t &b0 [[buffer(6)]],
    constant int64_t &b1 [[buffer(7)]],
    device uint32_t * out_x [[buffer(8)]],
    device int64_t * out_a [[buffer(9)]],
    device int64_t * out_b [[buffer(10)]],
    constant uint32_t &cap [[buffer(11)]],
    constant uint32_t &j_base [[buffer(12)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_emit_ranked_body<true>(bits, gbase, nword, logI, a0, a1, b0, b1, out_x, out_a, out_b, cap, j_base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
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
template <int UNROLL, bool SLABBED>
static inline void k_resieve_scatter_body(
    device const plat_t * plat,
    device const uint32_t * primes,
    device const uint8_t * ispow,
    constant uint32_t &n,
    constant uint32_t &xmax,
    constant int &logI,
    device const uint32_t * summary,
    device const uint32_t * bits,
    device const uint32_t * gbase,
    device uint32_t * plist,
    device uint32_t * pcnt,
    constant uint32_t &K,
    device uint32_t * noverflow,
    constant int &log_gran,
    device const uint64_t * walk_cur,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t NONE = 0xFFFFFFFFu;      /* local x < 2^31, cannot collide */
    const uint64_t stride = bench_grid_stride_x();
    ulong ovf = 0;

    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        if (ispow && ispow[k]) continue;
        uint32_t p = primes[k];

        /* Resieve never carries walk state forward, so the only slab
         * difference is WHERE the walk starts -- one loop serves both. The
         * walk itself is 64-bit in both, for the reason measured in
         * k_fill_atomic: pl_next's saturating add is a branch per increment
         * and pl_next64's is not. The run-ahead buffer stays uint32 because
         * every in-slab position is below 2^31. */
        uint64_t x;
        if constexpr (SLABBED) x = walk_cur[k];   /* may already be past xmax */
        else                   x = pl_first64(&P, logI);
        while (x < xmax) {
            uint32_t xs[UNROLL], hit[UNROLL];

            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                if (x < xmax) { xs[u] = (uint32_t)x; x = pl_next64(x, &P, Imask); }
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
    if (ovf) atomicAdd64(noverflow, ovf);
}

kernel void k_resieve_scatter_1_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    device const uint8_t * ispow [[buffer(2), function_constant(mtl_bound_2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &xmax [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const uint32_t * summary [[buffer(6)]],
    device const uint32_t * bits [[buffer(7)]],
    device const uint32_t * gbase [[buffer(8)]],
    device uint32_t * plist [[buffer(9)]],
    device uint32_t * pcnt [[buffer(10)]],
    constant uint32_t &K [[buffer(11)]],
    device uint32_t * noverflow [[buffer(12)]],
    constant int &log_gran [[buffer(13)]],
    device const uint64_t * walk_cur [[buffer(14), function_constant(mtl_bound_14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint8_t * ispow_opt = nullptr;
    if (mtl_bound_2) ispow_opt = ispow;
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_14) walk_cur_opt = walk_cur;
    k_resieve_scatter_body<1, false>(plat, primes, ispow_opt, n, xmax, logI, summary, bits, gbase, plist, pcnt, K, noverflow, log_gran, walk_cur_opt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_resieve_scatter_2_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    device const uint8_t * ispow [[buffer(2), function_constant(mtl_bound_2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &xmax [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const uint32_t * summary [[buffer(6)]],
    device const uint32_t * bits [[buffer(7)]],
    device const uint32_t * gbase [[buffer(8)]],
    device uint32_t * plist [[buffer(9)]],
    device uint32_t * pcnt [[buffer(10)]],
    constant uint32_t &K [[buffer(11)]],
    device uint32_t * noverflow [[buffer(12)]],
    constant int &log_gran [[buffer(13)]],
    device const uint64_t * walk_cur [[buffer(14), function_constant(mtl_bound_14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint8_t * ispow_opt = nullptr;
    if (mtl_bound_2) ispow_opt = ispow;
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_14) walk_cur_opt = walk_cur;
    k_resieve_scatter_body<2, false>(plat, primes, ispow_opt, n, xmax, logI, summary, bits, gbase, plist, pcnt, K, noverflow, log_gran, walk_cur_opt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_resieve_scatter_4_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    device const uint8_t * ispow [[buffer(2), function_constant(mtl_bound_2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &xmax [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const uint32_t * summary [[buffer(6)]],
    device const uint32_t * bits [[buffer(7)]],
    device const uint32_t * gbase [[buffer(8)]],
    device uint32_t * plist [[buffer(9)]],
    device uint32_t * pcnt [[buffer(10)]],
    constant uint32_t &K [[buffer(11)]],
    device uint32_t * noverflow [[buffer(12)]],
    constant int &log_gran [[buffer(13)]],
    device const uint64_t * walk_cur [[buffer(14), function_constant(mtl_bound_14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint8_t * ispow_opt = nullptr;
    if (mtl_bound_2) ispow_opt = ispow;
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_14) walk_cur_opt = walk_cur;
    k_resieve_scatter_body<4, false>(plat, primes, ispow_opt, n, xmax, logI, summary, bits, gbase, plist, pcnt, K, noverflow, log_gran, walk_cur_opt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_resieve_scatter_8_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    device const uint8_t * ispow [[buffer(2), function_constant(mtl_bound_2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &xmax [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const uint32_t * summary [[buffer(6)]],
    device const uint32_t * bits [[buffer(7)]],
    device const uint32_t * gbase [[buffer(8)]],
    device uint32_t * plist [[buffer(9)]],
    device uint32_t * pcnt [[buffer(10)]],
    constant uint32_t &K [[buffer(11)]],
    device uint32_t * noverflow [[buffer(12)]],
    constant int &log_gran [[buffer(13)]],
    device const uint64_t * walk_cur [[buffer(14), function_constant(mtl_bound_14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint8_t * ispow_opt = nullptr;
    if (mtl_bound_2) ispow_opt = ispow;
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_14) walk_cur_opt = walk_cur;
    k_resieve_scatter_body<8, false>(plat, primes, ispow_opt, n, xmax, logI, summary, bits, gbase, plist, pcnt, K, noverflow, log_gran, walk_cur_opt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_resieve_scatter_4_1(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    device const uint8_t * ispow [[buffer(2), function_constant(mtl_bound_2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &xmax [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const uint32_t * summary [[buffer(6)]],
    device const uint32_t * bits [[buffer(7)]],
    device const uint32_t * gbase [[buffer(8)]],
    device uint32_t * plist [[buffer(9)]],
    device uint32_t * pcnt [[buffer(10)]],
    constant uint32_t &K [[buffer(11)]],
    device uint32_t * noverflow [[buffer(12)]],
    constant int &log_gran [[buffer(13)]],
    device const uint64_t * walk_cur [[buffer(14), function_constant(mtl_bound_14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint8_t * ispow_opt = nullptr;
    if (mtl_bound_2) ispow_opt = ispow;
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_14) walk_cur_opt = walk_cur;
    k_resieve_scatter_body<4, true>(plat, primes, ispow_opt, n, xmax, logI, summary, bits, gbase, plist, pcnt, K, noverflow, log_gran, walk_cur_opt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
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
/* Overridable so a build whose threadgroup budget cannot afford the default
 * can shrink it. DEFAULT UNCHANGED, so the CUDA build is unaffected; only a
 * build passing -DTD_TILE sees anything different.
 *
 * Why it matters on Apple: tdsmall_t is 32 bytes, so 512 of them is 16,384 B
 * of THREADGROUP memory -- half of Apple's whole 32,768 B budget, capping k_td
 * at ~2 threadgroups per core. A 1080 Ti has 96 KB per SM and fits six. See
 * METAL_PORT_PLAN.md section 8q. */
#ifndef TD_TILE
#define TD_TILE 512
#endif
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
template <int DIVIDE, int RECORD, int SELECT, bool SLABBED>
static inline void k_td_body(
    device const int64_t * A,
    device const int64_t * B,
    device const uint32_t * X,
    device const uint32_t * sel,
    constant uint32_t &n,
    constant int &logI,
    device const tdpoly_t * P,
    constant uint32_t &sq,
    device const uint32_t * plist,
    device const uint32_t * pcnt,
    constant uint32_t &K,
    device const tdsmall_t * sm,
    constant uint32_t &nsm,
    device bn_t * cof,
    device uint8_t * cofbits,
    device uint32_t * flags,
    device uint32_t * nhit,
    device uint32_t * fac,
    device uint32_t * faccnt,
    constant uint32_t &fmax,
    constant uint32_t &j_base,
    threadgroup tdsmall_t *tile,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint64_t nthread = bench_grid_stride_x();
    const uint64_t iters = ((uint64_t)n + nthread - 1) / nthread;
    const int deg = P->deg;
    ulong hits = 0;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t tt = bench_grid_thread_x() + it * nthread;
        const bool active = (tt < n);
        const uint32_t t = (uint32_t)tt;
        const uint32_t s = (SELECT && active) ? sel[t] : t;
        const int64_t a = active ? A[s] : 0;
        const int64_t b = active ? B[s] : 0;
        const uint64_t ua = (uint64_t)(a < 0 ? -a : a);
        const uint64_t ub = (uint64_t)(b < 0 ? -b : b);
        const int sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;
        uint32_t myflags = 0;
        uint32_t nf = 0;
        device uint32_t *myfac = (RECORD && active) ? fac + (size_t)t * fmax : NULL;
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
            const uint32_t jlocal = x >> logI;
            uint32_t jglobal = jlocal;
            if constexpr (SLABBED) jglobal += j_base;
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
                    const tdsmall_t te = tile[e];
                    const uint32_t m = te.m, g = te.g;
                    /* Ordinary prime entries have g==1 and use slab-local j
                     * with cst pre-phased once per slab. g>1 implies m==1 for
                     * a prime modulus, so only its global-row condition remains. */
                    uint32_t jp = jlocal;
                    if (g > 1) {
                        if (jglobal % g) continue;
                        jp = jglobal / g;
                    }
                    if (te.magic) {
                        uint32_t w = te.rt * jp + hi;
                        if (td_mod_magic(w, m, te.magic, te.sh) != te.cst)
                            continue;
                    }
                    /* magic == 0 means m == 1: the row condition was the whole
                     * test, and it has already passed. */
                    hits++;
                    if (DIVIDE) {
                        hitp[nh] = m * g; hitr[nh] = te.recip; nh++;
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
    if (nhit && hits) atomicAdd64(nhit, hits);
}

kernel void k_td_0_0_0_0(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<0, 0, 0, false>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_0_0_0(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 0, 0, false>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_0_0_1(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 0, 0, true>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_1_0_0(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 1, 0, false>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_1_0_1(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 1, 0, true>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_1_1_0(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 1, 1, false>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_1_1_1_1(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3), function_constant(mtl_bound_3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * nhit [[buffer(16), function_constant(mtl_bound_16)]],
    device uint32_t * fac [[buffer(17), function_constant(mtl_bound_17)]],
    device uint32_t * faccnt [[buffer(18), function_constant(mtl_bound_18)]],
    constant uint32_t &fmax [[buffer(19)]],
    constant uint32_t &j_base [[buffer(20)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    device const uint32_t * sel_opt = nullptr;
    if (mtl_bound_3) sel_opt = sel;
    device uint32_t * nhit_opt = nullptr;
    if (mtl_bound_16) nhit_opt = nhit;
    device uint32_t * fac_opt = nullptr;
    if (mtl_bound_17) fac_opt = fac;
    device uint32_t * faccnt_opt = nullptr;
    if (mtl_bound_18) faccnt_opt = faccnt;
    k_td_body<1, 1, 1, true>(A, B, X, sel_opt, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, nhit_opt, fac_opt, faccnt_opt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* ---- the recording pass, one WARP per candidate ------------------------ *
 *
 * `k_td` is the right shape for the two dense passes, which run over every
 * survivor and are throughput-bound on `n`. It is the WRONG shape for the
 * recording pass, and slabbing is what made that visible.
 *
 * That pass runs `SELECT=1` over the ~1,900-5,300 joint candidates, and one
 * thread per candidate means one thread marching the entire `nsm` entry list
 * (6,726 on a c194). Measured per launch across a slab sweep it costs a dead
 * flat ~0.65 ms whether the launch carries 5,342 candidates or 577 -- it is
 * latency-bound on `nsm`, not throughput-bound on `nacc`, which is why sizing
 * the grid to `ceil(nacc/threads)` was tried and measured nothing (RESULTS
 * finding 74). Two launches per slab then made the pass grow at ~1.33 ms per
 * slab, and that per-slab tax is the entire reason the slab-size curve turns
 * back up below 2^29 positions.
 *
 * Here a whole warp takes one candidate and the 32 lanes stride the entry
 * list, so the march is ~32x shorter for the same candidate. Three things make
 * that cheap rather than delicate:
 *
 *   - the norm stays in ONE lane. `bn_t` is 256 bits and the divisions are
 *     inherently sequential on N, so shuffling it between lanes would cost
 *     more than it saves. Lane 0 owns N, `nf` and the factor list; the other
 *     lanes only ever evaluate the congruence. Their idling is free: the
 *     serial part was one thread's work before this change too.
 *   - the hit mask comes back from `__ballot_sync` in ASCENDING LANE ORDER,
 *     and lane index is entry index within the 32-entry chunk, so walking it
 *     with __ffs divides factors out in the same entry order `k_td` uses. That
 *     is a CONVENIENCE, not the correctness argument -- an earlier version of
 *     this comment claimed otherwise and was wrong. `td_divide_out` loops
 *     until p no longer divides, so an entry consumes the whole power of its
 *     prime whichever entry reaches it first, and distinct primes commute:
 *     the recorded multiset is order-invariant. Both emitters sort before
 *     writing anyway (`cf_emit_sorted`, cofac.cuh:1613; `std::sort`,
 *     pipeline.cuh:1002), so relation TEXT cannot see the order either.
 *     Order is load-bearing in exactly one place -- when `nf` exceeds `fmax`
 *     the tail is dropped, so which factors survive depends on it -- and that
 *     path is a hard error the host reports as "raise TD_FMAX". A
 *     lanes-per-item rewrite of the dense pass (STATUS item 18) is therefore
 *     NOT constrained to preserve this order.
 *   - the shared tile is unchanged and still block-wide: every warp in the
 *     block walks the same entry list, so one staged tile serves all of them
 *     and the `__syncthreads` pair stays uniform (`iters` is computed from the
 *     warp count, so every warp makes the same number of passes).
 *
 * No hit buffer is needed. `k_td` buffers into `hitp[TD_MAXHIT]` because a hit
 * is a per-lane event there while the division is a whole-warp one; here the
 * ballot has already done the compaction, and lane 0's division loop skips
 * empty chunks in a single predicate.
 *
 * DIVIDE=0 and the dense SELECT=0 form have no counterpart here on purpose:
 * this kernel exists for the one launch that was paying the tax, and leaving
 * the diagnostics on `k_td` keeps a second, independent implementation of the
 * same predicate in the tree.
 *
 * WHAT COVERS THIS KERNEL, precisely -- `--verify` does NOT. pipe_td_verify
 * launches `k_td<1,1,0>` into its own d_fac/d_faccnt (pipeline.cuh:796) and
 * runs BEFORE the `if (!nacc) return 0;` that guards phase 3, so it never
 * reads C->d_cfac[side], the only array this kernel writes. A --verify run
 * staying green says nothing about the warp path. The actual gates are
 * out-of-process: cofcheck.sh's A/B against `--td-record-scalar` at two slab
 * shapes, and `--check-relations` rebuilding norms from the emitted factor
 * lists. Anyone adding an in-binary gate for this kernel has to compare
 * d_cfac after phase 3, not before it. */
template <bool SLABBED>
static inline void k_td_record_warp_body(
    device const int64_t * A,
    device const int64_t * B,
    device const uint32_t * X,
    device const uint32_t * sel,
    constant uint32_t &n,
    constant int &logI,
    device const tdpoly_t * P,
    constant uint32_t &sq,
    device const uint32_t * plist,
    device const uint32_t * pcnt,
    constant uint32_t &K,
    device const tdsmall_t * sm,
    constant uint32_t &nsm,
    device bn_t * cof,
    device uint8_t * cofbits,
    device uint32_t * flags,
    device uint32_t * fac,
    device uint32_t * faccnt,
    constant uint32_t &fmax,
    constant uint32_t &j_base,
    threadgroup tdsmall_t *tile,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = (int32_t)(1u << (logI - 1));
    const uint32_t lane  = threadIdx.x & 31u;
    const uint32_t wpb   = blockDim.x >> 5;            /* warps per block */
    const uint64_t nwarp = (uint64_t)gridDim.x * wpb;
    const uint64_t w0    = (uint64_t)blockIdx.x * wpb + (threadIdx.x >> 5);
    const uint64_t iters = ((uint64_t)n + nwarp - 1) / nwarp;
    const int deg = P->deg;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t tt = w0 + it * nwarp;
        /* Warp-uniform: `tt` derives from `w0`, which is built from
         * (threadIdx.x >> 5), so every lane of a warp shares it. The
         * unconditional __ballot_sync below depends on this -- if `active`
         * were ever per-lane, lanes that took the `continue` would still be
         * named in the mask. Do not derive `tt` from a per-lane quantity. */
        const bool active = (tt < n);
        const bool leader = active && lane == 0;
        const uint32_t t = (uint32_t)tt;
        const uint32_t s = active ? sel[t] : 0u;
        uint32_t myflags = 0;
        uint32_t nf = 0;
        device uint32_t *myfac = leader ? fac + (size_t)t * fmax : NULL;
        bn_t N;

        /* Everything that touches N runs on lane 0 alone. The other lanes are
         * masked off for exactly the stretch that used to be the whole
         * kernel's serial cost, so this costs nothing they were not already
         * waiting through. */
        if (leader) {
            const int64_t a = A[s], b = B[s];
            const uint64_t ua = (uint64_t)(a < 0 ? -a : a);
            const uint64_t ub = (uint64_t)(b < 0 ? -b : b);
            const int sa = (a < 0) ? -1 : 1, sb = (b < 0) ? -1 : 1;

            /* ---- exact norm ---- */
            bns_t acc; bns_zero(&acc);
            for (int k = 0; k <= deg; k++) {
                bn_t term = P->c[k];
                int sg = P->sign[k];
                if (bn_is_zero(&term)) continue;
                for (int e = 0; e < k; e++) {
                    if (bn_mul_u64(&term, ua)) myflags |= TDF_NORM_OVERFLOW;
                    sg *= sa;
                }
                for (int e = 0; e < deg - k; e++) {
                    if (bn_mul_u64(&term, ub)) myflags |= TDF_NORM_OVERFLOW;
                    sg *= sb;
                }
                if (bns_addmag(&acc, &term, sg)) myflags |= TDF_NORM_OVERFLOW;
            }
            N = acc.m;

            /* ---- the special-q, on its own side ---- */
            if (sq) td_divide_out<1>(&N, sq, bn_recip_u32(sq), myfac, &nf, fmax);

            /* ---- large primes recovered by the resieve ---- */
            uint32_t c = pcnt[s];
            if (c > K) { c = K; myflags |= TDF_LIST_TRUNCATED; }
            for (uint32_t k = 0; k < c; k++) {
                const uint32_t p = plist[(size_t)s * K + k];
                td_divide_out<1>(&N, p, bn_recip_u32(p), myfac, &nf, fmax);
            }
        }

        /* ---- small primes, direct test, 32 entries at a time ---- */
        {
            const uint32_t x = active ? X[s] : 0u;
            const int32_t  i = (int32_t)(x & Imask) - Ihalf;
            const uint32_t jlocal = x >> logI;
            uint32_t jglobal = jlocal;
            if constexpr (SLABBED) jglobal += j_base;
            const uint32_t hi = (uint32_t)(Ihalf - i);      /* in (0, 2^logI] */

            for (uint32_t base = 0; base < nsm; base += TD_TILE) {
                const uint32_t cnt = min((uint32_t)TD_TILE, nsm - base);
                __syncthreads();
                for (uint32_t k = threadIdx.x; k < cnt; k += blockDim.x)
                    tile[k] = sm[base + k];
                __syncthreads();
                if (!active) continue;
                for (uint32_t c = 0; c < cnt; c += 32) {
                    const uint32_t e = c + lane;
                    bool hit = false;
                    if (e < cnt) {
                        const tdsmall_t te = tile[e];
                        const uint32_t m = te.m, g = te.g;
                        uint32_t jp = jlocal;
                        hit = true;
                        if (g > 1) {
                            if (jglobal % g) hit = false;
                            else jp = jglobal / g;
                        }
                        if (hit && tile[e].magic) {
                            const uint32_t w = te.rt * jp + hi;
                            if (td_mod_magic(w, m, tile[e].magic, tile[e].sh)
                                != tile[e].cst) hit = false;
                        }
                    }
                    /* Ascending lane order == ascending entry order == the
                     * order k_td divides in. Convenient for diffing factor
                     * arrays, but NOT required: see the header -- the multiset
                     * is order-invariant and both emitters sort. */
                    uint32_t mask = __ballot_sync(0xffffffffu, hit);
                    if (lane == 0) {
                        while (mask) {
                            const uint32_t z = c + (uint32_t)__ffs(mask) - 1u;
                            mask &= mask - 1u;
                            td_divide_out<1>(&N, tile[z].m * tile[z].g,
                                             tile[z].recip, myfac, &nf, fmax);
                        }
                    }
                }
            }
        }

        if (leader) {
            cof[t] = N;
            cofbits[t] = (uint8_t)bn_bits(&N);
            faccnt[t] = nf;
            if (myflags) atomicOr(flags, myflags);
        }
    }
}

kernel void k_td_record_warp_0(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * fac [[buffer(16)]],
    device uint32_t * faccnt [[buffer(17)]],
    constant uint32_t &fmax [[buffer(18)]],
    constant uint32_t &j_base [[buffer(19)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    k_td_record_warp_body<false>(A, B, X, sel, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, fac, faccnt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_td_record_warp_1(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * X [[buffer(2)]],
    device const uint32_t * sel [[buffer(3)]],
    constant uint32_t &n [[buffer(4)]],
    constant int &logI [[buffer(5)]],
    device const tdpoly_t * P [[buffer(6)]],
    constant uint32_t &sq [[buffer(7)]],
    device const uint32_t * plist [[buffer(8)]],
    device const uint32_t * pcnt [[buffer(9)]],
    constant uint32_t &K [[buffer(10)]],
    device const tdsmall_t * sm [[buffer(11)]],
    constant uint32_t &nsm [[buffer(12)]],
    device bn_t * cof [[buffer(13)]],
    device uint8_t * cofbits [[buffer(14)]],
    device uint32_t * flags [[buffer(15)]],
    device uint32_t * fac [[buffer(16)]],
    device uint32_t * faccnt [[buffer(17)]],
    constant uint32_t &fmax [[buffer(18)]],
    constant uint32_t &j_base [[buffer(19)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup tdsmall_t tile[TD_TILE];
    k_td_record_warp_body<true>(A, B, X, sel, n, logI, P, sq, plist, pcnt, K, sm, nsm, cof, cofbits, flags, fac, faccnt, fmax, j_base, tile, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* ---- cofactor classification ------------------------------------------- */

/* Applies CADO's check_leftover_norm per survivor. This is the stage that
 * turns hundreds of thousands of survivors into the ~1,851 that are worth
 * handing to a cofactorizer, and the 2026-08-04 harness work established that
 * it is real work the siever has to do, not "a small input adapter". */
kernel void k_classify(
    device const bn_t * cof [[buffer(0)]],
    device const uint8_t * cofbits [[buffer(1)]],
    device const int64_t * B [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &lpb [[buffer(4)]],
    constant uint32_t &mfb [[buffer(5)]],
    constant sf64 &lim [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
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
kernel void k_accept_flags(
    device const uint8_t * st0 [[buffer(0)]],
    device const uint8_t * st1 [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    device uint32_t * flag [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        uint32_t a = (st0[t] == COF_ACCEPT || st0[t] == COF_SPLIT);
        uint32_t b = (st1[t] == COF_ACCEPT || st1[t] == COF_SPLIT);
        flag[t] = a & b;
    }
}


kernel void k_scatter_sel(
    device const uint32_t * flag [[buffer(0)]],
    device const uint32_t * off [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    device uint32_t * sel [[buffer(3)]],
    constant uint32_t &cap [[buffer(4)]],
    device uint32_t * nacc [[buffer(5)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t] && off[t] < cap) sel[off[t]] = t;
        if (t == n - 1) *nacc = off[t] + flag[t];
    }
}


kernel void k_gather_ab(
    device const int64_t * A [[buffer(0)]],
    device const int64_t * B [[buffer(1)]],
    device const uint32_t * sel [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    device int64_t * oa [[buffer(4)]],
    device int64_t * ob [[buffer(5)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        oa[t] = A[sel[t]]; ob[t] = B[sel[t]];
    }
}


