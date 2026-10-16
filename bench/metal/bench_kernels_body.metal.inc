/* Optional buffer arguments: see metal/gen_bench_kernels.py.
 * metal_rt supplies bit i = "kernel argument i is bound". */
constant uint mtl_bound_mask [[function_constant(0)]];
constant bool mtl_bound_9 = (mtl_bound_mask & (1u << 9)) != 0;
constant bool mtl_bound_10 = (mtl_bound_mask & (1u << 10)) != 0;
constant bool mtl_bound_11 = (mtl_bound_mask & (1u << 11)) != 0;
constant bool mtl_bound_12 = (mtl_bound_mask & (1u << 12)) != 0;
constant bool mtl_bound_13 = (mtl_bound_mask & (1u << 13)) != 0;
constant bool mtl_bound_14 = (mtl_bound_mask & (1u << 14)) != 0;
constant bool mtl_bound_16 = (mtl_bound_mask & (1u << 16)) != 0;
constant bool mtl_bound_25 = (mtl_bound_mask & (1u << 25)) != 0;
constant bool mtl_bound_26 = (mtl_bound_mask & (1u << 26)) != 0;

#ifndef ROWRT_NONE
#define ROWRT_NONE 0xFFFFFFFFu
#endif

template <bool SLABBED>
static inline void k_transform_body(
    device const uint32_t * primes,
    device const uint32_t * roots,
    device plat_t * out,
    constant uint32_t &n,
    constant int &logI,
    constant uint32_t &J,
    constant int64_t &a0,
    constant int64_t &a1,
    constant int64_t &b0,
    constant int64_t &b1,
    device uint32_t * nproj,
    device uint32_t * nlost,
    device uint64_t * walk_cur,
    device uint32_t * rt_out,
    constant uint32_t &n_rt,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const uint32_t q = primes[k];
        uint32_t rt, g, m = pl_transform_enc(q, roots[k], a0, a1, b0, b1, &rt, &g);
        /* --fill-rows: the row tier walks i == rt*j (mod q) row by row, which
         * is the same lattice pl_make reduces whenever g == 1 (then m == q). */
        if (k < n_rt) rt_out[k] = (g > 1) ? ROWRT_NONE : rt;
        if (g > 1) {                         /* rows only: emit an empty walk */
            plat_t P; P.inc_warp = PL_INVALID; P.inc_step = PL_VERTICAL;
            P.bound_warp = 0; P.bound_step = 0;
            out[k] = P;
            if constexpr (SLABBED) walk_cur[k] = UINT64_MAX;
            atomicAdd(nproj, 1u);
            atomicAdd64(nlost, (ulong)(J / g) * ((1u << logI) / m));
        } else {
            const plat_t P = pl_make(m, rt, logI);
            out[k] = P;
            if constexpr (SLABBED) walk_cur[k] = pl_first64(&P, logI);
        }
    }
}

kernel void k_transform_0(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * roots [[buffer(1)]],
    device plat_t * out [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant uint32_t &J [[buffer(5)]],
    constant int64_t &a0 [[buffer(6)]],
    constant int64_t &a1 [[buffer(7)]],
    constant int64_t &b0 [[buffer(8)]],
    constant int64_t &b1 [[buffer(9)]],
    device uint32_t * nproj [[buffer(10)]],
    device uint32_t * nlost [[buffer(11)]],
    device uint64_t * walk_cur [[buffer(12), function_constant(mtl_bound_12)]],
    device uint32_t * rt_out [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &n_rt [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_12) walk_cur_opt = walk_cur;
    device uint32_t * rt_out_opt = nullptr;
    if (mtl_bound_13) rt_out_opt = rt_out;
    k_transform_body<false>(primes, roots, out, n, logI, J, a0, a1, b0, b1, nproj, nlost, walk_cur_opt, rt_out_opt, n_rt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_transform_1(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * roots [[buffer(1)]],
    device plat_t * out [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant uint32_t &J [[buffer(5)]],
    constant int64_t &a0 [[buffer(6)]],
    constant int64_t &a1 [[buffer(7)]],
    constant int64_t &b0 [[buffer(8)]],
    constant int64_t &b1 [[buffer(9)]],
    device uint32_t * nproj [[buffer(10)]],
    device uint32_t * nlost [[buffer(11)]],
    device uint64_t * walk_cur [[buffer(12), function_constant(mtl_bound_12)]],
    device uint32_t * rt_out [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &n_rt [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_12) walk_cur_opt = walk_cur;
    device uint32_t * rt_out_opt = nullptr;
    if (mtl_bound_13) rt_out_opt = rt_out;
    k_transform_body<true>(primes, roots, out, n, logI, J, a0, a1, b0, b1, nproj, nlost, walk_cur_opt, rt_out_opt, n_rt, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* ---- positions gcd(i,j) already rules out (--sieve-skip, finding 100) ----
 *
 * x = j*I + (i + I/2) and I/2 is even, so parity(i) == parity(x). Every stage
 * that skips a both-even position -- fill, apply's init, the small sieve and
 * the scan -- asks these two helpers, because they must agree exactly: fill
 * dropping a record for a cell apply still scans in would lose relations
 * silently, and no gate compares the four sites with each other.
 *
 * pos_row is the GLOBAL row. Slabbed kernels add j_base; unslabbed ones do not,
 * so they stay correct whatever j_base a caller passes. */
template <bool SLABBED>
static inline uint32_t pos_row(uint32_t x, int logI, uint32_t j_base)
{
    uint32_t j = x >> logI;
    if constexpr (SLABBED) j += j_base;
    return j;
}

static inline int pos_both_even(uint32_t x, uint32_t j)
{
    return ((x | j) & 1u) == 0u;
}

/* --fill-gate: the side filled SECOND stores a record only where the side
 * filled first has a survivor within the same 2^gate_shift positions. A
 * position the first side rejected is dropped by k_intersect_compact whatever
 * the second side's sum is, so the records that matter are all still written
 * and the relations are unchanged. gate == NULL is the ungated fill. */
static inline int fill_gate_drops(const device uint32_t * gate,
                                               int gate_shift, uint32_t x)
{
    if (!gate) return 0;
    const uint32_t g = x >> gate_shift;
    return !((gate[g >> 5] >> (g & 31u)) & 1u);
}

/* ---- stage (a): naive single-level atomic append ---------------------- */

/* One atomicAdd per record into a 2^log_nbuckets-way split. This is the
 * baseline the design doc says to beat. */
template <int RECBYTES, bool SLABBED, typename SLICE_T = uint16_t>
static inline void k_fill_atomic_body(
    device const plat_t * plat,
    device const SLICE_T * slice,
    constant uint32_t &n,
    constant uint32_t &xmax,
    constant int &logI,
    constant int &log_region,
    device uint32_t * cursor,
    device uint8_t * out,
    constant uint32_t &cap,
    device uint32_t * overflow,
    device const uint64_t * walk_cur,
    device uint64_t * walk_next,
    constant int &sieve_skip,
    constant uint32_t &j_base,
    device const uint32_t * gate,
    constant int &gate_shift,
    constant int &sl_shift,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint64_t stride = bench_grid_stride_x();
    /* --sieve-skip: 3 | i  <=>  (x & Imask) == I/2 (mod 3), since i = (x & Imask) - I/2. */
    const uint32_t ih3 = (1u << (logI - 1)) % 3u;

    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) {
            if constexpr (SLABBED) walk_next[k] = walk_cur[k];
            continue;
        }
        /* one read per prime, not per record: the slice hint is what the apply
         * kernel turns back into log p (and what resieve would use to recover
         * the prime itself), exactly as CADO's shorthint does */
        const uint32_t sl = slice[k];
        /* MEASURED 2026-08-24: the walk runs in 64 bits in BOTH specialisations,
         * and the legacy 32-bit form is the slow one. pl_next routes every
         * increment through pl_add32_sat -- `if (hi || x > UINT32_MAX - lo)` --
         * which is a BRANCH per increment, up to two per step, in the hottest
         * loop in the program. pl_next64 is a plain 64-bit add: two IADDs, no
         * control flow. Converting the slabbed path to 32-bit cost 6.0% of fill
         * (116.8 -> 123.8 ms/q, n=4 vs n=5, non-overlapping ranges, c147
         * I14/J65536) -- that experiment ran on a CONTENDED card and its
         * absolutes are ~2x the idle ones, so treat the 6.0% as directional
         * only; it was not repeated once the card freed up. Converting the
         * unslabbed path to 64-bit is what actually pays -- but ONLY there.
         * The slabbed path was already 64-bit before this change, so it gains
         * nothing: measured -6.1% on c147 I14/J8192 (unslabbed) against -0.7%
         * on c147 I14/J65536 and +0.3% on c194 I16/J32768 (both slabbed).
         * Every production geometry at I16 and above is slabbed, so this
         * conversion buys real jobs nothing; it is kept because it also removes
         * a duplicated loop and because sub-2^30 geometries do benefit.
         *
         * Correctness of the narrowing, which is the part to read before
         * touching the walk width: local positions stay below 2^31 either way,
         * so (uint32_t)x is exact for the record and the bucket index.
         *
         * What survives from the 32-bit attempt is the shape: one loop here
         * instead of two. An explicit `x >= xmax` early-out does NOT survive --
         * it was load-bearing only while x was 32-bit and could not hold the
         * continuation. With a 64-bit x the loop below simply does not execute
         * and the store after it writes the identical value, so the branch was
         * pure added divergence. (An exhausted walk holds UINT64_MAX and decays
         * by xmax per slab; pre-existing, and stays far above xmax for any
         * reachable slab count.) */
        uint64_t x;
        if constexpr (SLABBED) x = walk_cur[k];   /* may already be past xmax */
        else                   x = pl_first64(&P, logI);
        for (; x < xmax; x = pl_next64(x, &P, Imask)) {
            const uint32_t xl = (uint32_t)x;
            /* --sieve-skip: a position k_intersect_compact's gcd(i,j) test will
             * reject gets no record. Fill is L2-bound with the ALU nearly idle,
             * so the test is free and every skipped record is one write and one
             * cursor atomic fewer. parity(i) == parity(x) because I/2 is even. */
            if (sieve_skip) {
                const uint32_t j = pos_row<SLABBED>(xl, logI, j_base);
                if (pos_both_even(xl, j)) continue;
                if (sieve_skip > 1 && (j % 3u) == 0u && ((xl & Imask) % 3u) == ih3)
                    continue;
            }
            /* The bucket resieve reads records only at two-sided survivors,
             * whose gate group always has a side-1 survivor, so it loses
             * nothing to the gate either. */
            if (fill_gate_drops(gate, gate_shift, xl)) continue;
            const uint32_t b = xl >> log_region;
            const uint32_t slot = atomicAdd(&cursor[b], 1u);
            if (slot >= cap) { atomicAdd(overflow, 1u); continue; }
            const size_t at = ((size_t)b * cap + slot) * RECBYTES;
            if (RECBYTES == 2) {
                *(device uint16_t *)(out + at) = (uint16_t)(xl & offmask);
            } else if (RECBYTES == 4) {
                *(device uint32_t *)(out + at) = (xl & offmask) | (sl << sl_shift);
            } else {
                *(device uint64_t *)(out + at) = (uint64_t)(xl & offmask) | ((uint64_t)sl << 32);
            }
        }
        /* x is already the exact first hit at/after the slab end. */
        if constexpr (SLABBED) walk_next[k] = x - xmax;
    }
}

kernel void k_fill_atomic_2_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint16_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<2, false>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_atomic_4_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint16_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<4, false>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_atomic_8_0(
    device const plat_t * plat [[buffer(0)]],
    device const uint16_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<8, false>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_atomic_4_1(
    device const plat_t * plat [[buffer(0)]],
    device const uint16_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<4, true>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_atomic_4_0_uint32_t(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<4, false, uint32_t>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_atomic_4_1_uint32_t(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * slice [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint8_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    device const uint64_t * walk_cur [[buffer(10), function_constant(mtl_bound_10)]],
    device uint64_t * walk_next [[buffer(11), function_constant(mtl_bound_11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint64_t * walk_cur_opt = nullptr;
    if (mtl_bound_10) walk_cur_opt = walk_cur;
    device uint64_t * walk_next_opt = nullptr;
    if (mtl_bound_11) walk_next_opt = walk_next;
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_atomic_body<4, true, uint32_t>(plat, slice, n, xmax, logI, log_region, cursor, out, cap, overflow, walk_cur_opt, walk_next_opt, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* ---- --fill-rows: the smallest bucketed moduli, row by row ------------ *
 *
 * k_fill_atomic gives each modulus ONE thread, and that thread visits all of
 * its xmax/q hits in sequence, each a cursor atomic whose returned slot the
 * store must wait for. The smallest bucketed modulus is ~I, so its thread
 * runs xmax/I = J dependent atomic->store steps while the average thread
 * runs a few hundred: the kernel's tail is a handful of warps on an
 * otherwise idle card. (C143: fill +60% from I14 to I13 for +14% records.)
 *
 * For q >= I a row holds at most one hit, at i + I/2 == (rt*j + I/2) mod q,
 * so the strip can be enumerated by row instead of by FK step, and the rows
 * split into bands that start independently. A warp takes 32 consecutive
 * moduli over one band of rows; its hits in a row land in one or two
 * regions, so one atomic per region per row reserves slots for all of them
 * (CADO's fill_row_kernel does the same). Same records as the FK walk; only
 * their order inside a bucket changes, which k_apply's sum does not see.
 *
 * rts[k] is k_transform's rt_out: the root, or ROWRT_NONE where k_fill_atomic
 * would skip the entry (PL_INVALID). The row tier needs no walk state; when
 * k_resieve_scatter still walks it (--no-resieve-rows), k_rows_walk_next
 * supplies the cursor. j_base counts only under SLABBED, as in pos_row. */
template <bool SLABBED, typename SLICE_T = uint16_t>
static inline void k_fill_rows_body(
    device const uint32_t * primes,
    device const uint32_t * rts,
    device const SLICE_T * slice,
    constant uint32_t &n,
    constant uint32_t &nbands,
    constant uint32_t &rows,
    constant int &logI,
    constant int &log_region,
    device uint32_t * cursor,
    device uint8_t * out,
    constant uint32_t &cap,
    device uint32_t * overflow,
    constant int &sieve_skip,
    constant uint32_t &j_base,
    device const uint32_t * gate,
    constant int &gate_shift,
    constant int &sl_shift,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint32_t I = 1u << logI;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint32_t jb = SLABBED ? j_base : 0u;
    const uint32_t ih3 = (I >> 1) % 3u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    /* whole warps only (blockDim % 32 == 0), so this exit is warp-uniform and
     * the full-mask ballot below stays legal */
    if (warp >= (uint64_t)((n + 31u) >> 5) * nbands) return;
    const uint32_t band = (uint32_t)(warp % nbands);
    const uint32_t k = (uint32_t)(warp / nbands) * 32u + lane;
    const uint32_t r0 = (uint32_t)((uint64_t)band * rows / nbands);
    const uint32_t r1 = (uint32_t)((uint64_t)(band + 1) * rows / nbands);

    uint32_t q = 1, rt = 0, sl = 0;
    const bool active = k < n && rts[k] != ROWRT_NONE;
    if (active) { q = primes[k]; rt = rts[k]; sl = slice[k]; }
    uint32_t off = active
        ? (uint32_t)(((uint64_t)rt * (jb + r0) + (I >> 1)) % q) : I;

    for (uint32_t r = r0; r < r1; r++) {
        const uint32_t j = jb + r;
        /* (0,0) is the only lattice point of row 0 and is never sieved */
        bool hit = off < I && j != 0;
        if (hit && sieve_skip) {
            /* row bases are multiples of I, so parity(x) == parity(off) and
             * x & Imask == off: the same tests as k_fill_atomic */
            if (((off | j) & 1u) == 0u) hit = false;
            else if (sieve_skip > 1 && j % 3u == 0u && off % 3u == ih3) hit = false;
        }
        const uint32_t xl = (r << logI) + off;
        if (hit && fill_gate_drops(gate, gate_shift, xl)) hit = false;
        /* every lane runs every row (the band is warp-uniform), so the
         * full-mask ballot is legal */
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, hit);
        if (hit) {
            const uint32_t b = xl >> log_region;
            uint32_t slot;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
            /* __match_any_sync needs sm_70; before it, one atomic per hit */
            (void)mask; (void)lane;
            slot = atomicAdd(&cursor[b], 1u);
#else
            const uint32_t peers = __match_any_sync(mask, b);
            const uint32_t leader = __ffs(peers) - 1;
            slot = 0;
            if (lane == leader) slot = atomicAdd(&cursor[b], __popc(peers));
            slot = __shfl_sync(peers, slot, leader) +
                   __popc(peers & ((1u << lane) - 1u));
#endif
            if (slot >= cap) atomicAdd(overflow, 1u);
            else *(device uint32_t *)(out + ((size_t)b * cap + slot) * 4) =
                     (xl & offmask) | (sl << sl_shift);
        }
        if (active) { off += rt; if (off >= q) off -= q; }
    }
}

kernel void k_fill_rows_0(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * rts [[buffer(1)]],
    device const uint16_t * slice [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &nbands [[buffer(4)]],
    constant uint32_t &rows [[buffer(5)]],
    constant int &logI [[buffer(6)]],
    constant int &log_region [[buffer(7)]],
    device uint32_t * cursor [[buffer(8)]],
    device uint8_t * out [[buffer(9)]],
    constant uint32_t &cap [[buffer(10)]],
    device uint32_t * overflow [[buffer(11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_rows_body<false>(primes, rts, slice, n, nbands, rows, logI, log_region, cursor, out, cap, overflow, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_rows_1(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * rts [[buffer(1)]],
    device const uint16_t * slice [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &nbands [[buffer(4)]],
    constant uint32_t &rows [[buffer(5)]],
    constant int &logI [[buffer(6)]],
    constant int &log_region [[buffer(7)]],
    device uint32_t * cursor [[buffer(8)]],
    device uint8_t * out [[buffer(9)]],
    constant uint32_t &cap [[buffer(10)]],
    device uint32_t * overflow [[buffer(11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_rows_body<true>(primes, rts, slice, n, nbands, rows, logI, log_region, cursor, out, cap, overflow, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_rows_0_uint32_t(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * rts [[buffer(1)]],
    device const uint32_t * slice [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &nbands [[buffer(4)]],
    constant uint32_t &rows [[buffer(5)]],
    constant int &logI [[buffer(6)]],
    constant int &log_region [[buffer(7)]],
    device uint32_t * cursor [[buffer(8)]],
    device uint8_t * out [[buffer(9)]],
    constant uint32_t &cap [[buffer(10)]],
    device uint32_t * overflow [[buffer(11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_rows_body<false, uint32_t>(primes, rts, slice, n, nbands, rows, logI, log_region, cursor, out, cap, overflow, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_rows_1_uint32_t(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * rts [[buffer(1)]],
    device const uint32_t * slice [[buffer(2)]],
    constant uint32_t &n [[buffer(3)]],
    constant uint32_t &nbands [[buffer(4)]],
    constant uint32_t &rows [[buffer(5)]],
    constant int &logI [[buffer(6)]],
    constant int &log_region [[buffer(7)]],
    device uint32_t * cursor [[buffer(8)]],
    device uint8_t * out [[buffer(9)]],
    constant uint32_t &cap [[buffer(10)]],
    device uint32_t * overflow [[buffer(11)]],
    constant int &sieve_skip [[buffer(12)]],
    constant uint32_t &j_base [[buffer(13)]],
    device const uint32_t * gate [[buffer(14), function_constant(mtl_bound_14)]],
    constant int &gate_shift [[buffer(15)]],
    constant int &sl_shift [[buffer(16)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device const uint32_t * gate_opt = nullptr;
    if (mtl_bound_14) gate_opt = gate;
    k_fill_rows_body<true, uint32_t>(primes, rts, slice, n, nbands, rows, logI, log_region, cursor, out, cap, overflow, sieve_skip, j_base, gate_opt, gate_shift, sl_shift, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* Slabbed --fill-rows with --no-resieve-rows: k_resieve_scatter replays
 * walk_cur for the row tier as for every other entry, so each needs the first
 * lattice point at or past the slab end, as k_fill_atomic leaves it for the
 * rest. Stepping row by row to it can take ~q - I rows (rt == 1), which is the
 * single-thread tail the tier exists to remove, so pl_rows_to_hit jumps there
 * in O(log q). row_end is the GLOBAL row after this slab. */
kernel void k_rows_walk_next(
    device const uint32_t * primes [[buffer(0)]],
    device const uint32_t * rts [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &row_end [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    device const uint64_t * walk_cur [[buffer(5)]],
    device uint64_t * walk_next [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t I = 1u << logI;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const uint32_t rt = rts[k];
        if (rt == ROWRT_NONE) { walk_next[k] = walk_cur[k]; continue; }
        const uint32_t q = primes[k];
        const uint32_t off = (uint32_t)(((uint64_t)rt * row_end + (I >> 1)) % q);
        const uint64_t t = pl_rows_to_hit(off, rt, q, I);
        if (t == UINT64_MAX) { walk_next[k] = UINT64_MAX; continue; }
        const uint32_t o = (uint32_t)((off + (uint64_t)rt * (t % q)) % q);
        walk_next[k] = (t << logI) + o;   /* x - xmax, as k_fill_atomic */
    }
}


/* --fill-gate-shift > 0: fold the first side's survivor bitmap to one bit per
 * 2^shift positions, set when any of them survived. A superset of the bitmap,
 * so the gated fill stores somewhat more records than shift 0 would but the
 * gate is 2^shift times smaller to keep in L2. One thread per gate word. */
kernel void k_gate_fold(
    device const uint32_t * bits [[buffer(0)]],
    device uint32_t * gate [[buffer(1)]],
    constant int &shift [[buffer(2)]],
    constant uint32_t &ngword [[buffer(3)]],
    constant uint32_t &nbitword [[buffer(4)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t span = 1u << shift;          /* bitmap words per gate word */
    const uint32_t per = 32u >> shift;          /* gate bits per bitmap word */
    const uint32_t low = span == 32u ? 0xFFFFFFFFu : (1u << span) - 1u;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t w = bench_grid_thread_x(); w < ngword; w += stride) {
        uint32_t g = 0;
        for (uint32_t k = 0; k < span; k++) {
            /* the last gate word may cover past the bitmap's end: no
             * survivors there, so no gate bits */
            const uint64_t wi = (w << shift) + k;
            const uint32_t word = wi < nbitword ? bits[wi] : 0u;
            for (uint32_t c = 0; c < per; c++)
                if ((word >> (c << shift)) & low) g |= 1u << (k * per + c);
        }
        gate[w] = g;
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
 * assigned to it. The cut is on the entry's effective MODULUS m = q/g, not on
 * the prime: row-divisor and projective entries have a large q and a small m.
 * (Entry counts below are from an early job; c183's algebraic side has 95
 * block-tier entries.)
 *   m <   64  (52 entries, 84% of updates): the whole block
 *   m < 1024  (165 entries):                one warp per entry
 *   m >= 1024 (1752 entries, <=16 hits):    one thread per entry
 * The two lower tiers share each entry's setup across a warp by shuffle
 * (RESULTS findings 101-102); see sieve_small.
 */
#define SS_BLOCK_CUT   64u
#define SS_WARP_CUT  1024u

template <int CELLBITS, int ATOMIC>
static inline void ss_add(threadgroup uint32_t *S, uint32_t c, uint32_t lp)
{
    const uint32_t CPW = 32 / CELLBITS;
    uint32_t v = lp << ((c % CPW) * CELLBITS);
    if (ATOMIC) atomicAdd(&S[c / CPW], v);
    else        S[c / CPW] += v;
}

/* First offset within [0,width) whose i is congruent to rt*j (mod p). TWO
 * selects the two-step reduction, for the thread tier's tail range only (see
 * the magic path below and ss_bound_ok in td.cuh). */
template <bool TWO = false>
static inline uint32_t ss_first(uint32_t p, uint32_t rt,
                                             uint32_t j, int32_t ilo,
                                             uint32_t magic, uint32_t kshift)
{
    /* Magic path: no integer division at all, just __umulhi + shift + mad.
     *
     * ONE table lookup, not three. Only `magic` has to be stored -- it needs a
     * 64-bit divide to build, so the host has to. The other two values used to
     * be stored beside it and are now derived here for free, because the extra
     * global loads cost more than the arithmetic they saved (measured: the
     * three-value form left two thirds of the available win on the table,
     * concentrated in the thread tier that has the fewest hits to amortize a
     * load over):
     *
     *   bias  p << SS_KSHIFT(logI), the same macro the host builds its bound
     *         against. Any multiple of p that is >= Ihalf will do -- it only
     *         has to make the numerator non-negative without moving the
     *         residue -- and p >= 2 gives p<<(logI-2) >= 2^(logI-1) == Ihalf.
     *         The stored "least multiple" was never worth a load; neither was
     *         a per-p shift, which is smaller still but costs more per hit
     *         than the coverage is worth (measured; see SS_KSHIFT in td.cuh).
     *   sh    floor(log2 p) == 31 - clz(p), one instruction.
     *
     * Powers of two take an AND instead: td_magic_build has to special-case
     * them (their general-form magic would be 2^32, which does not fit), and
     * the AND is cheaper than the reciprocal anyway. p == 1 never reaches this
     * branch -- ss_magic_build refuses m < 2 outright, so it takes the
     * fallback below (t % 1 == 0). Do not relax that guard on the theory that
     * the AND covers p == 1: td_magic_build's power-of-two case would then
     * evaluate 1u << 32, which is undefined.
     *
     * The uint32 arithmetic is exact because the true value is non-negative
     * and < 2^31; the host refuses the magic (magic == 0) for any entry where
     * it could not prove that, matching tdsmall_t's convention in td.cuh.
     * td_mod_magic is exact only for dividends < 2^31, which is that bound. */
    if (magic) {
        if constexpr (TWO) {
            /* TWO-STEP: the host could not prove rt*j + bias - ilo < 2^31
             * (at I16/J32768, every p above ~43,690), but did prove rt*j and
             * (rt*j mod p) + bias - ilo each are. Same magic, applied twice:
             * a few instructions, against the 64-bit remainder call below
             * that these entries took in every region (findings 110-111). A power
             * of two needs no bound at all -- p divides 2^32, so the AND is
             * exact even if the 32-bit sum wraps. */
            if ((p & (p - 1)) == 0)
                return (rt * j + (p << kshift) - (uint32_t)ilo) & (p - 1);
            const uint32_t sh = 31u - (uint32_t)__clz((int)p);
            const uint32_t r  = td_mod_magic(rt * j, p, magic, sh);
            return td_mod_magic(r + (p << kshift) - (uint32_t)ilo, p, magic, sh);
        }
        const uint32_t w = rt * j + (p << kshift) - (uint32_t)ilo;
        if ((p & (p - 1)) == 0) return w & (p - 1);
        return td_mod_magic(w, p, magic, 31u - (uint32_t)__clz((int)p));
    }
    /* ONE reduction, not two. base == rt*j (mod p), so (base - ilo) mod p is
     * just (rt*j - ilo) mod p -- the intermediate reduction to `base` buys
     * nothing. The bias makes the numerator unsigned so the sign fixup goes
     * too: p<<20 is a multiple of p (leaves the residue alone) and is >= 2^21
     * for any p >= 2, while |ilo| <= 2^(logI-1) <= 2^19 because main() pins
     * logI to [2,20]. No overflow: the uint32 product is < 2^64 and the bias
     * is < 2^52 for any bkthresh a real job sets.
     *
     * Motivation is AMD-specific: there is no integer-divide instruction, so
     * every % here is a software sequence. Removing one is removing real ALU
     * work, not one op. */
    const uint64_t t = (uint64_t)rt * j + ((uint64_t)p << 20) - (uint64_t)(int64_t)ilo;
    return (uint32_t)(t % p);
}

/* ss_magic_build (the host side of the above) and SS_KSHIFT live in td.cuh,
 * next to the td_magic_build they call -- shared by both builds rather than
 * forked, and reachable from pipeline.cuh without depending on include order. */

/* One entry's per-row setup, shared by all three tiers of sieve_small: returns
 * the first hit c0 (< 2m) and sets *mm_out to the step, or to 0 when the entry
 * has no hits in this row -- g does not divide j, or under --sieve-skip an even
 * modulus whose hits in an even row all fall on even i. */
template <bool TWO = false>
static inline uint32_t ss_setup(uint32_t e,
                                             const device uint32_t * sp,
                                             const device uint32_t * srt,
                                             const device uint32_t * sg,
                                             const device uint32_t * smag,
                                             uint32_t j, int32_t ilo,
                                             uint32_t kshift, int evenrow,
                                             thread uint32_t *mm_out)
{
    /* All four loads before the g test, so they go out together: an entry
     * costs one L2 round trip, not two. The tables do not fit in the L1 left
     * beside apply's shared memory, and the thread tier pays this once per
     * entry per region (finding 110). An entry with g not dividing j loads
     * two words it does not use; they share its lines anyway. */
    const uint32_t m = sp[e], g = sg[e], rt = srt[e];
    const uint32_t mg = smag ? smag[e] : 0u;
    *mm_out = 0;
    if (g > 1 && (j % g)) return 0;
    uint32_t c0 = ss_first<TWO>(m, rt, g > 1 ? j / g : j, ilo, mg, kshift);
    uint32_t mm = m;
    if (evenrow) {
        if (m & 1u) { if (!(c0 & 1u)) c0 += m; mm = 2u * m; }
        else if (!(c0 & 1u)) return 0;
    }
    *mm_out = mm;
    return c0;
}

/* The tier split, for BOTH callers (the pipeline and run_bench). hsp must be
 * sorted by modulus; the counts are the leading runs below each cut, which is
 * what sieve_small's packing relies on: every e < nblk has m < SS_BLOCK_CUT
 * and every e < nwrp has m < SS_WARP_CUT. One copy, so the two callers cannot
 * drift apart and hand the block tier a modulus its 8-bit fields cannot hold. */
template <int CELLBITS, int ATOMIC, bool SLABBED = false, bool SSTAIL = true>
static inline void sieve_small(threadgroup uint32_t *S, uint32_t region, int logI, int log_region,
                            const device uint32_t * sp,
                            const device uint32_t * srt,
                            const device uint32_t * sg,
                            const device uint16_t * slp,
                            const device uint32_t * smag,
                            uint32_t nsmall, uint32_t nblk, uint32_t nwrp,
                            uint32_t nsplit,
                            uint32_t tid, uint32_t nth, uint32_t j_base,
                            int skipeven)
{
    const uint32_t width = 1u << log_region;
    const uint32_t x0    = region << log_region;
    const uint32_t jlocal = x0 >> logI;
    uint32_t j = jlocal;
    if constexpr (SLABBED) j += j_base;
    const int32_t  ilo   = (int32_t)(x0 & ((1u << logI) - 1)) - (int32_t)(1u << (logI - 1));
    /* Loop-invariant, so it is derived once here rather than per hit: the
     * host proves its bound against SS_KSHIFT(logI) too, so both sides read
     * the same definition. */
    const uint32_t kshift = SS_KSHIFT(logI);
    const uint32_t warp = tid >> 5, lane = tid & 31, nwarps = nth >> 5;
    /* --sieve-skip: in an even row only odd i can ever be a relation -- the
     * row half of pos_both_even -- and parity(c) == parity(i) because ilo is
     * even (x0 is even, I/2 is even). An odd modulus alternates parity, so
     * start at the first odd hit and step 2m; an even one never changes it, so
     * the entry either keeps every hit or contributes nothing to this row.
     * 2m cannot wrap: bench_main caps --bkthresh, and so every line-sieved
     * modulus, at 2^30. */
    const int evenrow = skipeven && !(j & 1u);

    /* Callers launch whole warps -- bench_main refuses an --apply-threads that
     * is not a multiple of 32 -- and every tier below depends on it: the two
     * shuffle tiers need full warps, and the warp tier's stride nth >> 5 would
     * be 0 otherwise. */

    /* Every entry is (m, rt, g): hits this row only when g | j, and then at
     * i == rt*(j/g) (mod m). g == 1 is the ordinary case; m == 1 means every
     * position in the row. See pl_transform_gen. ss_setup is that per-row
     * setup, shared by all three tiers so they cannot sieve different position
     * sets. */
    #define SS_SETUP(e, MM) ss_setup((e), sp, srt, sg, smag, j, ilo, kshift, evenrow, &(MM))

    /* WHOLE-BLOCK TIER (m < SS_BLOCK_CUT), setup shared by shuffle. An entry's
     * setup is the same for every thread in the block, and the per-warp form
     * computed it in all nth/32 warps for every entry: ncu on the c183
     * algebraic side (95 entries) put that replicated setup at ~19% of
     * k_apply's samples, against ~15% for the hits (finding 101). Lane k of
     * each warp now sets up entry base+k and the warp walks those 32 entries
     * with ONE __shfl_sync each: ceil(nblk/32) setups per warp, not nblk.
     *
     * One register carries it all: m < SS_BLOCK_CUT <= 128 bounds c0 (< 2m
     * after the even-row shift) and the step mm (<= 2m) to 8 bits each, and the
     * log is 16. mm == 0 means "no hits in this row". The host guarantees
     * m < SS_BLOCK_CUT for every e < nblk (ss_tiers); the static_assert keeps
     * the constant inside the packing. Cell sums are integer adds, so the
     * reordering is output-identical.
     *
     * Leftover hits always land on the lowest tids, so warp 0 runs the most
     * iterations of each entry. Rotating the start thread per entry evens that
     * out and was measured SLOWER (apply +3.6%, finding 102): leave it. */
    static_assert(SS_BLOCK_CUT <= 128u, "block-tier fields are packed in 8 bits");
    for (uint32_t base = 0; base < nblk; base += 32u) {
        const uint32_t e = base + lane;
        uint32_t pk = 0;
        if (e < nblk) {
            uint32_t mm;
            const uint32_t c0 = SS_SETUP(e, mm);
            if (mm) pk = c0 | (mm << 8) | ((uint32_t)slp[e] << 16);
        }
        const uint32_t ne = (nblk - base < 32u) ? nblk - base : 32u;
        for (uint32_t k = 0; k < ne; k++) {
            const uint32_t q  = __shfl_sync(0xFFFFFFFFu, pk, k);
            const uint32_t mm = (q >> 8) & 0xFFu;
            if (!mm) continue;
            const uint32_t lp = q >> 16;
            for (uint32_t c = (q & 0xFFu) + tid * mm; c < width; c += nth * mm)
                ss_add<CELLBITS,ATOMIC>(S, c, lp);
        }
    }

    /* WARP TIER (SS_BLOCK_CUT <= m < SS_WARP_CUT), setup shared by shuffle
     * too. Warp w still owns entries nblk + w + i*nwarps, but instead of
     * running each entry's setup as a whole-warp sequence for its 0.5-8 hits
     * per lane, lane k sets up the warp's k-th next entry and the warp walks
     * them: apply −7.0% on c183 (finding 102). c0 and mm are < 2m < 2^16. */
    {
        static_assert(SS_WARP_CUT <= 32768u, "warp-tier c0/mm are packed in 16 bits");
        for (uint32_t base = 0; ; base += 32u) {
            const uint32_t e0 = nblk + warp + base * nwarps;
            if (e0 >= nwrp) break;
            const uint32_t e = e0 + lane * nwarps;
            uint32_t pk = 0, lp = 0;
            if (e < nwrp) {
                uint32_t mm;
                const uint32_t c0 = SS_SETUP(e, mm);
                if (mm) { pk = c0 | (mm << 16); lp = slp[e]; }
            }
            const uint32_t left = (nwrp - e0 + nwarps - 1) / nwarps;
            const uint32_t ne = left < 32u ? left : 32u;
            for (uint32_t k = 0; k < ne; k++) {
                const uint32_t q  = __shfl_sync(0xFFFFFFFFu, pk, k);
                const uint32_t kl = __shfl_sync(0xFFFFFFFFu, lp, k);
                const uint32_t mm = q >> 16;
                if (!mm) continue;
                for (uint32_t c = (q & 0xFFFFu) + lane * mm; c < width; c += 32u * mm)
                    ss_add<CELLBITS,ATOMIC>(S, c, kl);
            }
        }
    }

    /* THREAD TIER (m >= SS_WARP_CUT): one thread per entry, so the setup is
     * already lane-parallel. The log is loaded with the setup's four words,
     * not after c0 is known: three dependent L2 round trips per entry became
     * one. At I16 this tier has ~6,500 entries per side, each paid in every
     * quarter-row region; the hoist is apply -2.7% there, -2.0% at c183, and
     * no more because apply is issue-bound and co-resident blocks were
     * already hiding most of the wait (finding 110).
     *
     * THE TWO-STEP TAIL. Entries at or past nsplit (ss_two_step_from) are
     * the large moduli whose one-step numerator the host could not bound --
     * at I16/J32768 every m above ~43,690 -- and take the two-step setup
     * instead of the 64-bit fallback. SSTAIL is a template constant: false
     * (no tail, nsplit == nsmall: every geometry up to logI 15 at the default
     * J) folds the select away and runs the one-step loop alone; true tests
     * e >= nsplit per entry, which is warp-uniform except in the one warp
     * that straddles it.
     *
     * One thread-tier loop per kernel instance, ON PURPOSE. Two loops in one
     * kernel -- back to back over [nwrp, nsplit) and [nsplit, nsmall), or as
     * the two arms of a runtime branch -- made ptxas compile one of them
     * without reconvergence (a YIELD and a plain BSSY in its hit loop): lanes
     * drifted apart and the block waited at the scan's barrier, apply +24%
     * at I16 for the same instructions. And testing e < nsplit in the I15
     * loop, where it is always true, cost apply +0.8% at c183 (finding 111).
     * An edit here should recount YIELD/BSSY per k_apply instance
     * (oracle/ss-two-step-20261014/sass-structure.txt).
     *
     * The one-step instance TRAPS on a table with a tail. Its entries past
     * nsplit carry magics proved only for the two-step form, and the
     * one-step reduction would give them silently wrong residues
     * (ss_magic_build); a launch site that skipped ss_needs_tail must fail
     * loudly instead. */
    if constexpr (!SSTAIL) {
        if (nsplit < nsmall) __trap();
    }
    for (uint32_t e = nwrp + tid; e < nsmall; e += nth) {
        const uint32_t lp = slp[e];
        uint32_t mm;
        const uint32_t c0 = (SSTAIL && e >= nsplit)
            ? ss_setup<true>(e, sp, srt, sg, smag, j, ilo, kshift, evenrow, &mm)
            : SS_SETUP(e, mm);
        if (!mm) continue;
        for (uint32_t c = c0; c < width; c += mm)
            ss_add<CELLBITS,ATOMIC>(S, c, lp);
    }
    #undef SS_SETUP
}

/* ---- stage (A): apply -- accumulate logs into a shared-memory region ---- */

/* One position's survivor test. Factored out because the threshold scan below
 * has two shapes (warp-ballot and the small-region fallback) and a predicate
 * that drifts between them would produce results that depend on --region and
 * --apply-threads -- which the --verify gate does not vary, since it compares
 * CELLS rather than survivor counts.
 *
 * las's `not_both_even` filter: x = j*I + (i + I/2) and I/2 is a power of two
 * >= 2, so parity(i) == parity(x) and parity(j) == parity(x >> logI). Both even
 * means (a,b) are both even and the relation is a duplicate of (a/2, b/2); las
 * marks these 255 so they can never survive. Off by default so that every
 * survivor count recorded before 2026-08-03 still reproduces -- except that
 * k_apply ORs in its sieve_skip argument, because the cells --sieve-skip
 * leaves unsieved must never be scanned in. */
/* The per-cell side effects that accompany the predicate: the gate-5 probe and
 * the optional dump byte. They live here for the same reason apply_keep does --
 * the threshold scan has two shapes and anything duplicated between them drifts.
 * Splitting the predicate out but leaving these inline was the first version of
 * this refactor and it left both explanatory comments stranded in the fallback,
 * which is the branch that essentially never runs. */
static inline void apply_cell_side_effects(
        uint32_t x, uint32_t v, uint32_t CINIT,
        uint32_t probe_x, device uint32_t * probe_out,
        device uint8_t * dump)
{
    /* Gate 5's GPU half: this value has been through the root transform, the FK
     * walk, the tiering, the bucket fill, the small sieve and this kernel's
     * apply. probe_out[0] is the initialised norm, so the sieved log sum the
     * PIPELINE actually produced is v - (CINIT - probe_out[0]). */
    if (probe_out && x == probe_x) probe_out[1] = v;
    /* las's byte: S = max(T - sum(logp), 0), and our cell holds
     * CINIT - T + sum, so S = CINIT - cell. */
    if (dump) {
        int32_t sv = (int32_t)CINIT - (int32_t)v;
        dump[x] = (uint8_t)(sv < 0 ? 0 : (sv > 255 ? 255 : sv));
    }
}

template <bool SLABBED>
static inline int apply_keep(uint32_t x, uint32_t v,
                                          uint32_t THRESH, int logI,
                                          uint32_t j_base, int not_both_even)
{
    return (v >= THRESH)
        && !(not_both_even && pos_both_even(x, pos_row<SLABBED>(x, logI, j_base)));
}

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
template <int CELLBITS, int ATOMIC, int NORMMODE, bool SLABBED, bool SSTAIL = true>
static inline void k_apply_body(
    device const uint32_t * buckets,
    device const uint32_t * cnt,
    constant uint32_t &cap,
    constant int &logI,
    constant int &log_region,
    device const uint16_t * slice_logp,
    constant uint32_t &nslice,
    constant norm_t &N,
    constant uint32_t &CINIT,
    constant uint32_t &THRESH,
    constant uint32_t &tconst,
    device uint8_t * dump,
    device uint32_t * nsurv,
    device uint16_t * dbg_cells,
    constant uint32_t &dbg_region,
    device const uint32_t * sp,
    device const uint32_t * srt,
    device const uint32_t * sg,
    device const uint16_t * slp,
    device const uint32_t * smag,
    constant uint32_t &nsmall,
    constant uint32_t &nblk,
    constant uint32_t &nwrp,
    constant uint32_t &nsplit,
    constant uint32_t &probe_x,
    device uint32_t * probe_out,
    device uint32_t * survbits,
    constant int &not_both_even,
    constant int &sieve_skip,
    constant uint32_t &j_base,
    constant int &lut_shift,
    threadgroup uint32_t *sm,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    
    const uint32_t ncell  = 1u << log_region;
    const uint32_t nword  = ncell / (32 / CELLBITS);
    const uint32_t CPW    = 32 / CELLBITS;          /* cells per 32-bit word */
    threadgroup uint32_t *S   = sm;

    const uint32_t b = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = 1 << (logI - 1);
    const uint32_t xbase = b << log_region;
    /* --sieve-skip: both-even cells get no norm, no small-prime hits and are
     * never scanned in. The region lies in one row -- bench_main refuses
     * --region > --logI under --pipeline -- so the row is one value per block. */
    const uint32_t jblk = pos_row<SLABBED>(xbase, logI, j_base);
    const int drop_both_even = not_both_even || sieve_skip;


    /* ---- init: log-norm bound, computed in shared memory ---- */
    for (uint32_t w = tid; w < nword; w += nth) {
        uint32_t word = 0;
        #pragma unroll
        for (uint32_t c = 0; c < CPW; c++) {
            uint32_t t;
            const uint32_t x = xbase + w * CPW + c;
            /* Warp-uniform: cell c has the same parity in every word, because
             * xbase and CPW are even. */
            if (sieve_skip && pos_both_even(x, jblk)) {
                t = CINIT;                       /* cell starts at 0 */
            } else if (NORMMODE == NORM_CONST) {
                t = tconst;
            } else {
                const int32_t  ii = (int32_t)(x & Imask) - Ihalf;
                const uint32_t jlocal = x >> logI;
                uint32_t jj = jlocal;
                if constexpr (SLABBED) jj += j_base;
                const float fi = (float)ii;
                const float fj = (float)jj;
                const float u = fmaf(N.ua, fi, N.ub * fj);
                const float v = fmaf(N.va, fi, N.vb * fj);
                float acc = N.d[N.deg], vp = 1.0f;
                /* The same Horner on |.|, which bounds the cancellation. The
                 * normalisation gives fp32 ample dynamic RANGE, but near a root
                 * line of F the terms are O(1) and cancel to something tiny, so
                 * what is left is mostly rounding noise. Measured before this
                 * guard: 144 of 63,497 positions in a band along the three real
                 * root lines rounded to the wrong sieve-log value, -3.31 to
                 * +2.57 units and BOTH SIGNS -- false survivors and lost
                 * relations alike. */
                float aabs = fabsf(N.d[N.deg]), vpa = 1.0f;
                const float au = fabsf(u), av = fabsf(v);
                #pragma unroll
                for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                    if (k < N.deg) {
                        vp  *= v;  acc  = fmaf(acc,  u,  N.d[k] * vp);
                        vpa *= av; aabs = fmaf(aabs, au, fabsf(N.d[k]) * vpa);
                    }
                float s = fabsf(acc);
                if (s < NORM_CANCEL_TOL * aabs) {
                    /* redo in fp64: (a,b) exactly in int64, sum at 2^-53.
                     * Doubles are 1/64 rate on this part, but this fires on
                     * well under one cell in a thousand. */
                    /* Apple GPUs have no `double`; softfp64.h is bit-exact
                     * binary64 in integer ops (Phase 2 gate). Same
                     * computation, same rounding, not an approximation. */
                    const sf64 a = sf_from_i64((long)ii * N.a0 + (long)jj * N.b0);
                    const sf64 b = sf_from_i64((long)ii * N.a1 + (long)jj * N.b1);
                    const sf64 ud = sf_div(a, N.A), vd = sf_div(b, N.B);
                    sf64 accd = N.dd[N.deg], vpd = SF_ONE_D;
                    #pragma unroll
                    for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                        if (k < N.deg) {
                            vpd = sf_mul(vpd, vd);
#ifdef NORM_FP64_NO_FMA
                            accd = sf_add(sf_mul(accd, ud), sf_mul(N.dd[k], vpd));
#else
                            accd = sf_fma(accd, ud, sf_mul(N.dd[k], vpd));
#endif
                        }
                    s = sf_to_f32(sf_abs(accd));
                }
                /* las: S = fb_log(|F|) = floor(log2|F| * scale + 0.5).
                 * las clamps here at 255 because its cell IS a byte, and that
                 * is the whole reason `scale` exists (1.28 * 196.61 = 251.7).
                 * Our cell is 16 bits, so the ceiling is CINIT, not 255, and
                 * scale becomes a free parameter rather than a constraint --
                 * at scale 1.28 las discards 0.39 bits of resolution per
                 * position that we can keep. Clamping at 255 threw that away
                 * and, worse, would have silently flattened every norm above
                 * 255/scale into one bucket the moment anyone raised scale. */
                /* log2f, NOT __log2f. The fast intrinsic is ~2 ulp and has no
                 * host equivalent, so the CPU replay could not mirror this
                 * expression exactly and "0 cells differ" was measuring two
                 * slightly different functions. Accurate log2f is reproducible
                 * on both sides; norm init was 1.76 ms of a 27 ms apply, so the
                 * difference is affordable and correctness is not. */
#ifdef NORM_FAST_LOG2
#warning "NORM_FAST_LOG2: __log2f breaks CPU parity; --verify and dumpcmp will \
report spurious cell mismatches. Pricing builds only, never for relations."
                /* PRICING SWITCH, not a shipping option: __log2f is one MUFU
                 * instruction against log2f's software sequence, and the point
                 * of building with it is to measure what the accurate version
                 * costs -- see the paragraph above for why the accurate one is
                 * the default.
                 *
                 * This SILENTLY INVALIDATES THE PARITY GATE, which is why the
                 * #warning above exists. poly.c's norm_target_host mirrors the
                 * device expression deliberately -- same float type, same
                 * clamp, same log2f -- so that "0 cells differ" means
                 * something. Building with this breaks the mirror on the
                 * device side only, and the ~2 ulp difference shows up as
                 * scattered mismatches at threshold boundaries that read
                 * exactly like a kernel regression. */
                float lg = N.scale * (N.log2M + __log2f(fmaxf(s, 1e-30f)) - N.bias);
#else
                float lg = N.scale * (N.log2M + log2f(fmaxf(s, 1e-30f)) - N.bias);
#endif
                int ti = (int)floorf(lg + 0.5f);
                const uint32_t TMAX = (CELLBITS == 8) ? 255u : CINIT;
                t = (ti < 0) ? 0u : ((uint32_t)ti > TMAX ? TMAX : (uint32_t)ti);
            }
            if (probe_out && x == probe_x) probe_out[0] = t;
            word |= (CINIT - t) << (c * CELLBITS);
        }
        S[w] = word;
    }
    __syncthreads();

    /* ---- small primes: line-sieved straight into the same shared region ---- */
    if (nsmall)
        sieve_small<CELLBITS, ATOMIC, SLABBED, SSTAIL>(S, b, logI, log_region, sp, srt, sg, slp,
                                               smag,
                                               nsmall, nblk, nwrp, nsplit, tid, nth,
                                               j_base,
                                               sieve_skip);

    /* ---- apply ---- */
    uint32_t n = cnt[b];
    if (n > cap) n = cap;
    for (uint32_t i = tid; i < n; i += nth) {
        const uint32_t r = buckets[(size_t)b * cap + i];
        const uint32_t off = r & (ncell - 1);
        const uint32_t lp  = slice_logp[(r >> lut_shift) & (nslice - 1)];
        const uint32_t w   = off / CPW;
        const uint32_t sh  = (off % CPW) * CELLBITS;
        if (ATOMIC) atomicAdd(&S[w], lp << sh);
        else        S[w] += lp << sh;               /* racy: speed-of-light probe */
    }
    __syncthreads();

    /* ---- threshold scan ----
     *
     * A WARP covers 32 consecutive positions, which is exactly one survbits
     * word, so the bitmap is written with ONE unconditional 32-bit store per
     * group instead of one atomicOr per survivor. Adjacent lanes read the same
     * shared word (CPW cells share a word), so the read broadcasts rather than
     * conflicting.
     *
     * MEASURED idle, paired, relations byte-identical throughout:
     *
     *   c147 I14/J65536  n=3   apply 83.3  -> 72.5   -12.9%
     *   c147 I14/J8192   n=3   apply 10.54 -> 9.13   -13.4%
     *   c194 I16/J32768  n=7   apply 158.1 -> 149.6   -5.4%
     *
     * **c194 is the representative job and it gets less than half the c147
     * win.** The gap is unexplained: c194 has 2x the area and 7x the two-sided
     * survivors yet a smaller ABSOLUTE saving (8.00 vs 10.78 ms), so the win
     * tracks neither positions nor survivors. Do not quote -13% as the figure.
     * (An earlier contended-card pass reported -15.0%/-9.8%; superseded.)
     *
     * ncu profiling on 2026-08-25 (finding 75) ruled out the obvious causes
     * without replacing them: all three geometries profile nearly identically
     * (DRAM 8.5-8.9%, IPC 2.88-2.91, occupancy 66.3-66.5%) and per-launch
     * instruction counts differ by 3% for the same 2^29 positions. It also
     * showed this scan is only ~3% of the kernel's instructions, so the win
     * was contention on the global atomics, not instruction count -- there is
     * no more to get by shaving the scan itself.
     *
     * The store is only race-free across blocks when a block owns whole
     * survbits words -- log_region >= 5 -- so smaller regions keep the atomicOr
     * form. (The full-warp half of the guard is belt-and-braces: bench_main
     * already rejects an --apply-threads that is not a multiple of 32, but this
     * kernel is also launched directly by the standalone harness.) This is why
     * the callers still clear survbits: the fast path makes the clear
     * redundant, the fallback does not.
     *
     * **CALLER CONTRACT, unenforceable from inside the kernel.** The fast path
     * OVERWRITES survbits words rather than OR-ing into them, which is correct
     * only when (a) blockIdx.x maps 1:1 onto regions, so `xbase = b <<
     * log_region` makes each block's words private, and (b) exactly one launch
     * writes a given bitmap. Both callers satisfy this today (`<<<nregion,
     * athr>>>`, one launch per side per slab). A caller that grid-strides
     * regions -- the pattern k_purge uses -- or that accumulates two slabs into
     * one bitmap would get a bitmap where only the last-written words are
     * correct, and only when log_region >= 5, so it would pass --region 4 and
     * fail the default --region 14. The kernel cannot check this: it never
     * sees nregion.
     *
     * nsurv is accumulated per warp and committed once. The old per-survivor
     * atomicAdd had its RETURN VALUE consumed by surv[at] = x, which forces a
     * serialising atom.global.add on one address; the list it indexed was
     * write-only in both callers and is gone, so what is left is a single
     * non-returning red.global.add per warp.
     */
    const uint32_t CMASK = (CELLBITS == 16) ? 0xFFFFu : 0xFFu;
    if (ncell >= 32u && nth >= 32u && (nth & 31u) == 0u) {
        const uint32_t lane = tid & 31u;
        const uint32_t ngrp = ncell >> 5;
        uint32_t nlocal = 0;
        for (uint32_t g = tid >> 5; g < ngrp; g += (nth >> 5)) {
            const uint32_t off = (g << 5) + lane;
            const uint32_t x   = xbase + off;
            const uint32_t v   =
                (S[off / CPW] >> ((off % CPW) * CELLBITS)) & CMASK;
            const int keep = apply_keep<SLABBED>(x, v, THRESH, logI,
                                                 j_base, drop_both_even);
            const uint32_t mask = __ballot_sync(0xFFFFFFFFu, keep);
            if (lane == 0) {
                if (survbits) survbits[x >> 5] = mask;
                nlocal += __popc(mask);
            }
            apply_cell_side_effects(x, v, CINIT, probe_x, probe_out, dump);
        }
        if (lane == 0 && nlocal) atomicAdd(nsurv, nlocal);
    } else {
        /* Fallback: log_region < 5, or a launch whose block is not whole warps.
         * survbits words are then shared between blocks, so the store has to be an
         * atomicOr into a pre-cleared bitmap. */
        for (uint32_t w = tid; w < nword; w += nth) {
            const uint32_t word = S[w];
            #pragma unroll
            for (uint32_t c = 0; c < CPW; c++) {
                const uint32_t v = (word >> (c * CELLBITS)) & CMASK;
                const uint32_t x = xbase + w * CPW + c;
                if (apply_keep<SLABBED>(x, v, THRESH, logI, j_base,
                                        drop_both_even)) {
                    atomicAdd(nsurv, 1u);
                    if (survbits) atomicOr(&survbits[x >> 5], 1u << (x & 31u));
                }
                apply_cell_side_effects(x, v, CINIT, probe_x, probe_out, dump);
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

kernel void k_apply_16_0_0_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 0, 0, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_0_1_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 0, 1, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_1_0_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 1, 0, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_1_1_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 1, 1, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_8_0_0_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<8, 0, 0, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_8_0_1_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<8, 0, 1, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_8_1_0_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<8, 1, 0, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_8_1_1_0_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<8, 1, 1, false, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_1_1_0_0(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 1, 1, false, false>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_1_1_1_1(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 1, 1, true, true>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_apply_16_1_1_1_0(
    device const uint32_t * buckets [[buffer(0)]],
    device const uint32_t * cnt [[buffer(1)]],
    constant uint32_t &cap [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint16_t * slice_logp [[buffer(5)]],
    constant uint32_t &nslice [[buffer(6)]],
    constant norm_t &N [[buffer(7)]],
    constant uint32_t &CINIT [[buffer(8)]],
    constant uint32_t &THRESH [[buffer(9)]],
    constant uint32_t &tconst [[buffer(10)]],
    device uint8_t * dump [[buffer(11), function_constant(mtl_bound_11)]],
    device uint32_t * nsurv [[buffer(12)]],
    device uint16_t * dbg_cells [[buffer(13), function_constant(mtl_bound_13)]],
    constant uint32_t &dbg_region [[buffer(14)]],
    device const uint32_t * sp [[buffer(15)]],
    device const uint32_t * srt [[buffer(16)]],
    device const uint32_t * sg [[buffer(17)]],
    device const uint16_t * slp [[buffer(18)]],
    device const uint32_t * smag [[buffer(19)]],
    constant uint32_t &nsmall [[buffer(20)]],
    constant uint32_t &nblk [[buffer(21)]],
    constant uint32_t &nwrp [[buffer(22)]],
    constant uint32_t &nsplit [[buffer(23)]],
    constant uint32_t &probe_x [[buffer(24)]],
    device uint32_t * probe_out [[buffer(25), function_constant(mtl_bound_25)]],
    device uint32_t * survbits [[buffer(26), function_constant(mtl_bound_26)]],
    constant int &not_both_even [[buffer(27)]],
    constant int &sieve_skip [[buffer(28)]],
    constant uint32_t &j_base [[buffer(29)]],
    constant int &lut_shift [[buffer(30)]],
    threadgroup uint32_t *sm [[threadgroup(0)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint8_t * dump_opt = nullptr;
    if (mtl_bound_11) dump_opt = dump;
    device uint16_t * dbg_cells_opt = nullptr;
    if (mtl_bound_13) dbg_cells_opt = dbg_cells;
    device uint32_t * probe_out_opt = nullptr;
    if (mtl_bound_25) probe_out_opt = probe_out;
    device uint32_t * survbits_opt = nullptr;
    if (mtl_bound_26) survbits_opt = survbits;
    k_apply_body<16, 1, 1, true, false>(buckets, cnt, cap, logI, log_region, slice_logp, nslice, N, CINIT, THRESH, tconst, dump_opt, nsurv, dbg_cells_opt, dbg_region, sp, srt, sg, slp, smag, nsmall, nblk, nwrp, nsplit, probe_x, probe_out_opt, survbits_opt, not_both_even, sieve_skip, j_base, lut_shift, sm, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


/* ---- stage (c) level 1: shared-memory staged split into super-buckets -- */

/* NBUF open output streams per block. The doc's rule: fan-out per pass is
 * bounded by how many buffers fit in shared memory, and each flush is one
 * atomic and one full-cache-line store, not one atomic per record. */
#define L1_NBUF   128
#define L1_CAP     61        /* uint32 slots = 256 B; 128*64*4 = 32 KB static smem */
#define L1_RUNMAX   8        /* max positions aggregated per reservation   */
#define L1_FLUSH  (L1_CAP - L1_RUNMAX)

/* Run-aggregated variant. A (p,r) walk emits positions in monotone
 * increasing order, so consecutive hits usually land in the SAME
 * super-bucket -- for the smallest bucket-sieved primes, which produce most
 * of the volume, many in a row do. Collect a run and reserve it with ONE
 * shared atomic, which also cuts the number of block barriers by the same
 * factor. The barrier-per-record version this replaces spent 30% of its
 * issue slots stalled on barriers at 7.5% of DRAM peak. */
kernel void k_fill_l1(
    device const plat_t * plat [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    constant uint32_t &xmax [[buffer(2)]],
    constant int &logI [[buffer(3)]],
    constant int &log_super [[buffer(4)]],
    device uint32_t * cursor [[buffer(5)]],
    device uint32_t * out [[buffer(6)]],
    constant uint32_t &cap [[buffer(7)]],
    device uint32_t * overflow [[buffer(8)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS
    threadgroup uint _sync_or_flag[1];

    threadgroup uint32_t buf[L1_NBUF * L1_CAP];
    threadgroup uint32_t cnt[L1_NBUF];
    threadgroup uint32_t base[L1_NBUF];

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;

    for (uint32_t i = tid; i < L1_NBUF; i += nth) cnt[i] = 0;
    __syncthreads();

    const uint64_t stride = bench_grid_product_u64(gridDim.x, nth);
    uint64_t k = bench_grid_product_u64(blockIdx.x, nth) + tid;
    plat_t P; P.inc_warp = PL_INVALID;
    /* This walk stays 32-bit, unlike k_fill_atomic's. Converting it to
     * pl_next64 was tried on 2026-08-24 and REVERTED: __launch_bounds__(512, 3)
     * above caps this kernel at 40 registers, HEAD fits in exactly 40 with no
     * spill, and the two extra 64-bit live values push it over -- ptxas honours
     * the bound by spilling rather than by dropping to 2 blocks/SM.
     *
     *     HEAD:      40 regs, 0 stack,  0 spill st,  0 spill ld
     *     converted: 40 regs, 8 stack, 12 spill st,  8 spill ld
     *
     * The conversion existed only to keep the fill-strategy A/B honest, and
     * buying that with local-memory traffic in one strategy defeats its own
     * purpose. If this kernel is ever revisited, raise or drop the launch bound
     * first and re-measure; do not widen the walk underneath it. */
    uint32_t x = 0;
    int active = 0;

    for (;;) {
        if (!active) {                       /* pick up the next prime */
            while (k < n) {
                P = plat[(uint32_t)k];
                if (P.inc_warp != PL_INVALID) { x = pl_first(&P, logI); active = (x < xmax); }
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
#define L2_CAP   61

template <int RECBYTES>
static inline void k_fill_l2_body(
    device const uint32_t * in,
    device const uint32_t * incnt,
    constant uint32_t &in_cap,
    constant int &log_region,
    constant int &log_super,
    device uint32_t * cursor,
    device uint8_t * out,
    constant uint32_t &cap,
    device uint32_t * overflow,
    threadgroup uint *_sync_or_flag,
    threadgroup uint32_t *buf,
    threadgroup uint32_t *cnt,
    threadgroup uint32_t *base,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

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
                    if (RECBYTES == 2) *(device uint16_t *)(out + at) = (uint16_t)(v & offmask);
                    else               *(device uint32_t *)(out + at) = (v & offmask);
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
                if (RECBYTES == 2) *(device uint16_t *)(out + at) = (uint16_t)(v & offmask);
                else               *(device uint32_t *)(out + at) = (v & offmask);
            }
        } else if (lane == 0) atomicAdd(overflow, c);
        __syncwarp();
    }
}

kernel void k_fill_l2_2(
    device const uint32_t * in [[buffer(0)]],
    device const uint32_t * incnt [[buffer(1)]],
    constant uint32_t &in_cap [[buffer(2)]],
    constant int &log_region [[buffer(3)]],
    constant int &log_super [[buffer(4)]],
    device uint32_t * cursor [[buffer(5)]],
    device uint8_t * out [[buffer(6)]],
    constant uint32_t &cap [[buffer(7)]],
    device uint32_t * overflow [[buffer(8)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup uint _sync_or_flag[1];
    threadgroup uint32_t buf[L2_NBUF * L2_CAP];
    threadgroup uint32_t cnt[L2_NBUF];
    threadgroup uint32_t base[L2_NBUF];
    k_fill_l2_body<2>(in, incnt, in_cap, log_region, log_super, cursor, out, cap, overflow, _sync_or_flag, buf, cnt, base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_fill_l2_4(
    device const uint32_t * in [[buffer(0)]],
    device const uint32_t * incnt [[buffer(1)]],
    constant uint32_t &in_cap [[buffer(2)]],
    constant int &log_region [[buffer(3)]],
    constant int &log_super [[buffer(4)]],
    device uint32_t * cursor [[buffer(5)]],
    device uint8_t * out [[buffer(6)]],
    constant uint32_t &cap [[buffer(7)]],
    device uint32_t * overflow [[buffer(8)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    threadgroup uint _sync_or_flag[1];
    threadgroup uint32_t buf[L2_NBUF * L2_CAP];
    threadgroup uint32_t cnt[L2_NBUF];
    threadgroup uint32_t base[L2_NBUF];
    k_fill_l2_body<4>(in, incnt, in_cap, log_region, log_super, cursor, out, cap, overflow, _sync_or_flag, buf, cnt, base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


static inline uint32_t bgcd(uint32_t u, uint32_t v)
{
    /* gcd(u,0) = u, which is what makes j = 0 fall out correctly: only
     * i = +-1 is primitive on that row. */
    if (!u) return v;
    if (!v) return u;
    int s = __ffs(u | v) - 1;
    u >>= __ffs(u) - 1;
    do {
        v >>= __ffs(v) - 1;
        if (u > v) { uint32_t t = u; u = v; v = t; }
        v -= u;
    } while (v);
    return u << s;
}


/* ---- recovery: the A/B ------------------------------------------------- *
 *
 * A: re-walk the factor base and test each hit against a hierarchical
 *    survivor filter. Touches no bucket memory at all.
 * B: stream the retained bucket array and keep the records that land on a
 *    survivor -- CADO's `purge`. Our 4 B record carries a slice index rather
 *    than a within-slice offset, so it cannot name the prime; that is a
 *    correctness gap, NOT a cost one. The memory traffic being measured here
 *    is exactly the traffic layout B would have, because the record is 4 B
 *    either way and every record is read exactly once.
 */
/* One summary bit per `wper` words of the full bitmap. wper == 2 reproduces
 * k_build_summary's 1-bit-per-64-positions table. Whole output words are built
 * by one thread, including a zero-padded partial final word, so unlike
 * k_build_summary this needs neither atomics nor a cudaMemset between slabs. */
kernel void k_build_summary_g(
    device const uint32_t * bits [[buffer(0)]],
    constant uint32_t &nword [[buffer(1)]],
    constant uint32_t &wper [[buffer(2)]],
    device uint32_t * summary [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t span = wper * 32u;
    const uint32_t nsw = (nword + span - 1u) / span;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ww = bench_grid_thread_x(); ww < nsw; ww += stride) {
        const uint32_t w = (uint32_t)ww;
        uint32_t out = 0;
        for (uint32_t b = 0; b < 32; b++) {
            uint32_t o = 0;
            const uint32_t base = (w * 32u + b) * wper;
            if (base >= nword) break;
            for (uint32_t k = 0; k < wper && base + k < nword; k++)
                o |= bits[base + k];
            if (o) out |= 1u << b;
        }
        summary[w] = out;
    }
}


kernel void k_build_summary(
    device const uint32_t * bits [[buffer(0)]],
    constant uint32_t &nword [[buffer(1)]],
    device uint32_t * summary [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    /* one summary bit per 64 positions == per 2 words of the full bitmap */
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ss = bench_grid_thread_x(); ss < nword / 2; ss += stride) {
        const uint32_t s = (uint32_t)ss;
        uint32_t occupied = (bits[2 * s] | bits[2 * s + 1]) != 0u;
        if (occupied) atomicOr(&summary[s >> 5], 1u << (s & 31u));
    }
}


kernel void k_resieve_rewalk(
    device const plat_t * plat [[buffer(0)]],
    device const uint32_t * primes [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    device const uint32_t * summary [[buffer(5)]],
    device const uint32_t * bits [[buffer(6)]],
    device uint32_t * out_x [[buffer(7)]],
    device uint32_t * out_p [[buffer(8)]],
    constant uint32_t &cap [[buffer(9)]],
    device uint32_t * nout [[buffer(10)]],
    device uint32_t * nprobe [[buffer(11)]],
    device uint32_t * npass1 [[buffer(12)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const uint64_t stride = bench_grid_stride_x();
    ulong probes = 0, pass1 = 0;
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        uint32_t p = primes[k];
        for (uint64_t x = pl_first64(&P, logI); x < xmax; x = pl_next64(x, &P, Imask)) {
            probes++;
            const uint32_t xn = (uint32_t)x;   /* narrow ONCE: every probe
                                                * below indexes with it, and a
                                                * 64-bit index is pure extra
                                                * address math on the dependent
                                                * load this loop waits on. */
            uint32_t sb = xn >> 6;
            if (!((summary[sb >> 5] >> (sb & 31u)) & 1u)) continue;
            pass1++;
            if (!((bits[xn >> 5] >> (xn & 31u)) & 1u)) continue;
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap) { out_x[slot] = xn; out_p[slot] = p; }
        }
    }
    if (probes) atomicAdd64(nprobe, probes);
    if (pass1)  atomicAdd64(npass1, pass1);
}


kernel void k_purge(
    device const uint32_t * recs [[buffer(0)]],
    device const uint32_t * cursor [[buffer(1)]],
    constant uint32_t &nregion [[buffer(2)]],
    constant uint32_t &cap [[buffer(3)]],
    constant int &log_region [[buffer(4)]],
    device const uint32_t * bits [[buffer(5)]],
    device uint32_t * out_x [[buffer(6)]],
    constant uint32_t &cap_out [[buffer(7)]],
    device uint32_t * nout [[buffer(8)]],
    device uint32_t * nread [[buffer(9)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t offmask = (1u << log_region) - 1;
    ulong rd = 0;
    for (uint32_t b = blockIdx.x; b < nregion; b += gridDim.x) {
        uint32_t nrec = cursor[b];
        if (nrec > cap) nrec = cap;
        rd += (threadIdx.x == 0) ? nrec : 0;
        for (uint32_t t = threadIdx.x; t < nrec; t += blockDim.x) {
            uint32_t rec = recs[(size_t)b * cap + t];
            uint32_t x = (b << log_region) | (rec & offmask);
            if (!((bits[x >> 5] >> (x & 31u)) & 1u)) continue;
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap_out) out_x[slot] = x;
        }
    }
    if (rd) atomicAdd64(nread, rd);
}



/* Slice cut for LAYOUT B. Two differences from fb_build_slices(), and both
 * are forced by the record format rather than chosen:
 *
 *  - the cap is 65536, not 262144, because the record's high 16 bits hold the
 *    offset WITHIN the slice. CADO has the same constraint and asserts on it
 *    (`fb.hpp:356`: size() <= numeric_limits<slice_offset_t>::max()).
 *  - the slice start indices are kept, because the fill is launched one slice
 *    at a time and the purge needs them to turn (slice, offset) back into a
 *    factor-base index.
 *
 * The tighter cap is what makes B's segment count ~3x the 39 that the looser
 * cut produces -- see the launch-cost discussion.
 */
/* build_slices_b: host-side, see bench_kernels.cu */


/* Fill one slice. Identical walk to k_fill_atomic<4>, but the record carries
 * the offset of the prime within this slice instead of a slice index, which is
 * what lets the purge name the prime. Slice identity comes from segmentation:
 * the driver snapshots cursor[] after each launch, giving per-(bucket, slice)
 * boundaries for free -- no counting pass. */
kernel void k_fill_segmented(
    device const plat_t * plat [[buffer(0)]],
    constant uint32_t &kbeg [[buffer(1)]],
    constant uint32_t &kend [[buffer(2)]],
    constant uint32_t &xmax [[buffer(3)]],
    constant int &logI [[buffer(4)]],
    constant int &log_region [[buffer(5)]],
    device uint32_t * cursor [[buffer(6)]],
    device uint32_t * out [[buffer(7)]],
    constant uint32_t &cap [[buffer(8)]],
    device uint32_t * overflow [[buffer(9)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t Imask = (1u << logI) - 1;
    const uint32_t offmask = (1u << log_region) - 1;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = (uint64_t)kbeg + bench_grid_thread_x();
         kk < kend; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        plat_t P = plat[k];
        if (P.inc_warp == PL_INVALID) continue;
        const uint32_t soff = k - kbeg;          /* < 65536 by construction */
        for (uint64_t x64 = pl_first64(&P, logI); x64 < xmax;
             x64 = pl_next64(x64, &P, Imask)) {
            const uint32_t x = (uint32_t)x64;
            uint32_t b = x >> log_region;
            uint32_t slot = atomicAdd(&cursor[b], 1u);
            if (slot >= cap) { atomicAdd(overflow, 1u); continue; }
            out[(size_t)b * cap + slot] = (x & offmask) | (soff << 16);
        }
    }
}


/* Snapshot cursor[] into column `sl` of the bucket-major boundary table.
 * This replaced a cudaMemcpy2D: a 2D copy 4 bytes wide over 32768 rows is
 * pathological (it was costing ~0.7 ms per slice, 88 ms over 126 slices),
 * whereas this is a plain coalesced read and a strided write. */
kernel void k_snapshot_bounds(
    device const uint32_t * cursor [[buffer(0)]],
    device uint32_t * bounds [[buffer(1)]],
    constant uint32_t &nregion [[buffer(2)]],
    constant uint32_t &nslice [[buffer(3)]],
    constant uint32_t &sl [[buffer(4)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t bb = bench_grid_thread_x(); bb < nregion; bb += stride) {
        const uint32_t b = (uint32_t)bb;
        bounds[(size_t)b * nslice + sl] = cursor[b];
    }
}


/* Purge, layout B: stream the retained records, keep the ones on a survivor,
 * and name the prime. bounds[b*nslice + s] is the cursor value after slice s
 * finished bucket b, so the slice a slot belongs to is found by binary search
 * over that row -- contiguous per bucket, which is why the table is laid out
 * bucket-major. */
kernel void k_purge_prime(
    device const uint32_t * recs [[buffer(0)]],
    device const uint32_t * cursor [[buffer(1)]],
    device const uint32_t * bounds [[buffer(2)]],
    device const uint32_t * starts [[buffer(3)]],
    device const uint32_t * primes [[buffer(4)]],
    constant uint32_t &nregion [[buffer(5)]],
    constant uint32_t &cap [[buffer(6)]],
    constant uint32_t &nslice [[buffer(7)]],
    constant int &log_region [[buffer(8)]],
    device const uint32_t * bits [[buffer(9)]],
    device uint32_t * out_x [[buffer(10)]],
    device uint32_t * out_p [[buffer(11)]],
    constant uint32_t &cap_out [[buffer(12)]],
    device uint32_t * nout [[buffer(13)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint32_t offmask = (1u << log_region) - 1;
    for (uint32_t b = blockIdx.x; b < nregion; b += gridDim.x) {
        uint32_t nrec = cursor[b];
        if (nrec > cap) nrec = cap;
        device const uint32_t *row = bounds + (size_t)b * nslice;
        for (uint32_t t = threadIdx.x; t < nrec; t += blockDim.x) {
            uint32_t rec = recs[(size_t)b * cap + t];
            uint32_t x = (b << log_region) | (rec & offmask);
            if (!((bits[x >> 5] >> (x & 31u)) & 1u)) continue;
            /* which slice does slot t belong to? row[] is non-decreasing. */
            uint32_t lo = 0, hi = nslice - 1;
            while (lo < hi) { uint32_t mid = (lo + hi) >> 1;
                              if (row[mid] <= t) lo = mid + 1; else hi = mid; }
            uint32_t p = primes[starts[lo] + (rec >> 16)];
            uint32_t slot = atomicAdd(nout, 1u);
            if (slot < cap_out) { out_x[slot] = x; out_p[slot] = p; }
        }
    }
}


template <int AGG, bool SLABBED>
static inline void k_intersect_compact_body(
    device const uint32_t * A,
    device const uint32_t * B,
    constant uint32_t &nword,
    constant uint32_t &logI,
    constant int64_t &a0,
    constant int64_t &a1,
    constant int64_t &b0,
    constant int64_t &b1,
    constant int64_t &q,
    device uint32_t * out_x,
    device int64_t  * out_a,
    device int64_t  * out_b,
    constant uint32_t &cap,
    device uint32_t * nout,
    device uint32_t * npre,
    device uint32_t * twosided,
    device uint32_t * nqb,
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
    const uint32_t lane = threadIdx.x & 31u;
    ulong pre = 0;
    /* Counted separately so the gcd(i,j)-only population stays recoverable.
     * That is the population CADO's after_sieve holds and dumpcmp --and
     * reproduces, so the parity gate must be able to name it; folding the
     * q|b rejections into the survivor count silently compares two different
     * populations, which is RESULTS finding 40's mistake. */
    ulong qbrej = 0;

    /* Round the trip count up so every lane of a warp runs the same number of
     * iterations: the warp-aggregated atomic below needs a converged warp, and
     * lanes past the end simply contribute nothing. */
    const uint64_t tid = bench_grid_thread_x();
    const uint64_t iters = ((uint64_t)nword + stride - 1) / stride;

    for (uint64_t it = 0; it < iters; it++) {
        const uint64_t ww = tid + it * stride;
        const uint32_t w = (uint32_t)ww;
        uint32_t m = (ww < nword) ? (A[w] & B[w]) : 0u;
        uint32_t keep = 0;
        pre += __popc(m);
        while (m) {
            uint32_t k = __ffs(m) - 1;
            m &= m - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            uint32_t jlocal = x >> logI;
            uint32_t j = jlocal;
            if constexpr (SLABBED) j += j_base;
            uint32_t ai = (uint32_t)(i < 0 ? -i : i);
            if (bgcd(ai, j) != 1) continue;
            /* the second test: q | b makes (a,b) = q*(a',b'). One 64-bit
             * modulo per two-sided survivor, which is ~1 position in 400. */
            if (q > 1 && ((int64_t)i * a1 + (int64_t)j * b1) % q == 0) {
                qbrej++;
                continue;
            }
            keep |= 1u << k;
        }

        /* Warp-aggregated allocation: one atomicAdd per warp instead of one
         * per survivor. Survivors are ~1 in 400 positions, so a warp covering
         * 32 words holds only a few -- but they all serialise on the same
         * global counter, and that, not the 134 MB of bitmap, is what this
         * kernel spends its time on. */
        /* Each thread owns a whole 32-position word, so the primitive
         * two-sided bitmap can be stored outright -- no atomics needed. This
         * is what the resieve filters against. */
        if (twosided && ww < nword) twosided[w] = keep;

        uint32_t slot;
        if (AGG) {
            uint32_t n = __popc(keep), scan = n;
            for (int off = 1; off < 32; off <<= 1) {
                uint32_t v = __shfl_up_sync(0xffffffffu, scan, off);
                if (lane >= (uint32_t)off) scan += v;
            }
            uint32_t total = __shfl_sync(0xffffffffu, scan, 31);
            uint32_t base = 0;
            if (lane == 31 && total) base = atomicAdd(nout, total);
            base = __shfl_sync(0xffffffffu, base, 31);
            slot = base + (scan - n);   /* exclusive prefix */
        } else {
            /* the naive form, kept for the A/B: one global atomic per
             * non-empty word. Measured 0.567 ms against 0.447 ms aggregated,
             * so the ~10 lines above buy 21%. */
            slot = keep ? atomicAdd(nout, __popc(keep)) : 0;
        }

        while (keep) {
            uint32_t k = __ffs(keep) - 1;
            keep &= keep - 1;
            uint32_t x = (w << 5) + k;
            int32_t  i = (int32_t)(x & Imask) - Ihalf;
            uint32_t jlocal = x >> logI;
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
    /* one atomic per thread, not one per survivor */
    if (pre) atomicAdd64(npre, pre);
    if (nqb && qbrej) atomicAdd64(nqb, qbrej);
}

kernel void k_intersect_compact_1_0(
    device const uint32_t * A [[buffer(0)]],
    device const uint32_t * B [[buffer(1)]],
    constant uint32_t &nword [[buffer(2)]],
    constant uint32_t &logI [[buffer(3)]],
    constant int64_t &a0 [[buffer(4)]],
    constant int64_t &a1 [[buffer(5)]],
    constant int64_t &b0 [[buffer(6)]],
    constant int64_t &b1 [[buffer(7)]],
    constant int64_t &q [[buffer(8)]],
    device uint32_t * out_x [[buffer(9), function_constant(mtl_bound_9)]],
    device int64_t  * out_a [[buffer(10), function_constant(mtl_bound_10)]],
    device int64_t  * out_b [[buffer(11), function_constant(mtl_bound_11)]],
    constant uint32_t &cap [[buffer(12)]],
    device uint32_t * nout [[buffer(13)]],
    device uint32_t * npre [[buffer(14)]],
    device uint32_t * twosided [[buffer(15)]],
    device uint32_t * nqb [[buffer(16), function_constant(mtl_bound_16)]],
    constant uint32_t &j_base [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint32_t * out_x_opt = nullptr;
    if (mtl_bound_9) out_x_opt = out_x;
    device int64_t  * out_a_opt = nullptr;
    if (mtl_bound_10) out_a_opt = out_a;
    device int64_t  * out_b_opt = nullptr;
    if (mtl_bound_11) out_b_opt = out_b;
    device uint32_t * nqb_opt = nullptr;
    if (mtl_bound_16) nqb_opt = nqb;
    k_intersect_compact_body<1, false>(A, B, nword, logI, a0, a1, b0, b1, q, out_x_opt, out_a_opt, out_b_opt, cap, nout, npre, twosided, nqb_opt, j_base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_intersect_compact_1_1(
    device const uint32_t * A [[buffer(0)]],
    device const uint32_t * B [[buffer(1)]],
    constant uint32_t &nword [[buffer(2)]],
    constant uint32_t &logI [[buffer(3)]],
    constant int64_t &a0 [[buffer(4)]],
    constant int64_t &a1 [[buffer(5)]],
    constant int64_t &b0 [[buffer(6)]],
    constant int64_t &b1 [[buffer(7)]],
    constant int64_t &q [[buffer(8)]],
    device uint32_t * out_x [[buffer(9), function_constant(mtl_bound_9)]],
    device int64_t  * out_a [[buffer(10), function_constant(mtl_bound_10)]],
    device int64_t  * out_b [[buffer(11), function_constant(mtl_bound_11)]],
    constant uint32_t &cap [[buffer(12)]],
    device uint32_t * nout [[buffer(13)]],
    device uint32_t * npre [[buffer(14)]],
    device uint32_t * twosided [[buffer(15)]],
    device uint32_t * nqb [[buffer(16), function_constant(mtl_bound_16)]],
    constant uint32_t &j_base [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device uint32_t * out_x_opt = nullptr;
    if (mtl_bound_9) out_x_opt = out_x;
    device int64_t  * out_a_opt = nullptr;
    if (mtl_bound_10) out_a_opt = out_a;
    device int64_t  * out_b_opt = nullptr;
    if (mtl_bound_11) out_b_opt = out_b;
    device uint32_t * nqb_opt = nullptr;
    if (mtl_bound_16) nqb_opt = nqb;
    k_intersect_compact_body<1, true>(A, B, nword, logI, a0, a1, b0, b1, q, out_x_opt, out_a_opt, out_b_opt, cap, nout, npre, twosided, nqb_opt, j_base, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


typedef struct {
    uint32_t w, m, magic, sh, ref;
} td_mod_case_t;

kernel void k_verify_td_mod_cases(
    device const td_mod_case_t * v [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t * first_bad [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const td_mod_case_t c = v[k];
        if (td_mod_magic(c.w, c.m, c.magic, c.sh) != c.ref)
            atomicMin(first_bad, k);
    }
}


typedef struct {
    uint32_t p, rt, j, magic, kshift, two, ref;
    int32_t  ilo;
} ss_first_case_t;

kernel void k_verify_ss_first_cases(
    device const ss_first_case_t * v [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t * first_bad [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t kk = bench_grid_thread_x(); kk < n; kk += stride) {
        const uint32_t k = (uint32_t)kk;
        const ss_first_case_t c = v[k];
        const uint32_t got = c.two
            ? ss_first<true>(c.p, c.rt, c.j, c.ilo, c.magic, c.kshift)
            : ss_first<false>(c.p, c.rt, c.j, c.ilo, c.magic, c.kshift);
        if (got != c.ref) atomicMin(first_bad, k);
    }
}

