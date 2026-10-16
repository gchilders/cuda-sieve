/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * CUDA's device-side vocabulary, expressed in MSL, so that ported kernel
 * bodies can stay textually identical to their CUDA originals.
 *
 * That is the whole design goal, and it is worth stating plainly: a kernel
 * body that differs from the original only in its address-space qualifiers
 * cannot have acquired a transcription bug, and this port's acceptance gates
 * are byte-comparisons where a single wrong cell changes the answer. Every
 * shim below exists so some line of bench_kernels.cu needs no edit.
 *
 * Phase 0 established that every warp assumption in the source holds on Apple
 * silicon: SIMD width 32, ascending simd_ballot lane order, and
 * simd_shuffle_up matching CUDA's __shfl_up_sync semantics exactly.
 */
#ifndef CUDA_SIEVE_CUDA_MSL_COMPAT_H
#define CUDA_SIEVE_CUDA_MSL_COMPAT_H

#include <metal_stdlib>
using namespace metal;

/* ---- limits <stdint.h> would provide ----------------------------------- */

#define UINT32_MAX 0xffffffffu
#define UINT64_MAX 0xfffffffffffffffful
#define INT32_MAX  0x7fffffff
#define INT64_MAX  0x7fffffffffffffffl

/* ---- launch configuration ---------------------------------------------- */

struct cu_dim { uint x; };
#define CUDA_KERNEL_IDS                                     \
    const cu_dim blockIdx  = { _bid };                      \
    const cu_dim threadIdx = { _lid };                      \
    const cu_dim blockDim  = { _bdim };                     \
    const cu_dim gridDim   = { _gdim };                     \
    (void)blockIdx; (void)threadIdx; (void)blockDim; (void)gridDim;

/* bench.h:895. CUDA keeps the cast before the multiply so a large grid cannot
 * wrap; k_fill_l1 calls it directly rather than through the stride helpers. */
static inline ulong bench_grid_product_u64(uint a, uint b)
{
    return (ulong)a * (ulong)b;
}

/* bench.h's grid-stride helpers. CUDA keeps the cast before the multiply so a
 * large grid cannot wrap; MSL's builtins are already the flattened values. */
#define bench_grid_thread_x()  ((ulong)_tid)
#define bench_grid_stride_x()  ((ulong)_ntid)

/* ---- barriers ---------------------------------------------------------- */

#define __syncthreads() threadgroup_barrier(mem_flags::mem_threadgroup)
/* CUDA's __syncwarp() also fences shared memory for the warp. */
#define __syncwarp(...)  simdgroup_barrier(mem_flags::mem_threadgroup)

/* CUDA's __syncthreads_or(pred): a barrier AND a block-wide "is pred true for
 * any thread?" vote. MSL has no block-wide vote, so it is built from
 * threadgroup memory. The write of 1 is idempotent, so the race between
 * voters is benign, and the two barriers reproduce the barrier half of the
 * CUDA semantics exactly.
 *
 * The scratch word is named `_sync_or_flag` and is declared by whichever
 * kernel needs it -- directly for a plain kernel, and by the wrapper for a
 * templated one, since MSL forbids threadgroup declarations inside a
 * non-kernel function. Declaring it as a one-element ARRAY means both spell
 * the call site identically. */
static inline int cu_syncthreads_or(threadgroup uint *flag, bool pred, uint lid)
{
    if (lid == 0) *flag = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pred) *flag = 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int r = (int)*flag;
    /* Third barrier: without it, a thread that races round the loop could
     * reset the flag for the next vote while a slower one is still reading
     * this one. Every caller does have a barrier further down its loop body,
     * so this is belt-and-braces -- but a vote that is subtly wrong shows up
     * as misplaced records with a correct total, which is the worst failure
     * mode to debug. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
#define __syncthreads_or(p) cu_syncthreads_or(_sync_or_flag, (p), _lid)

/* ---- warp intrinsics --------------------------------------------------- */

/* The mask argument is CUDA's; Metal has no partial-mask form, and every
 * call site in this tree passes full-mask. */
#define __ballot_sync(mask, pred) \
    ((uint)((simd_vote::vote_t)simd_ballot((pred))))
#define __shfl_sync(mask, v, lane)   simd_shuffle((v), (ushort)(lane))
#define __shfl_up_sync(mask, v, d)   simd_shuffle_up((v), (ushort)(d))
#define __activemask()               ((uint)0xffffffffu)

#define __popc(x)    ((int)popcount((uint)(x)))
#define __clz(x)     ((int)clz((uint)(x)))
#define __umulhi(a, b) mulhi((uint)(a), (uint)(b))
/* CUDA's __ffs is 1-based and returns 0 for 0; ctz(0) is undefined. */
static inline int __ffs(uint x) { return x ? (int)ctz(x) + 1 : 0; }

/* CUDA's __trap(): end the kernel, now, with an error. MSL has no trap
 * instruction and no abort, so this takes the one fault macOS reports BY
 * NAME -- a store through a null device pointer fails the command buffer with
 * MTLCommandBufferErrorPageFault, which metal_rt prints as `page fault`
 * (plan 9z-i). Loud and fatal, which is what the call sites want; the
 * alternative of making it a no-op would turn a detected impossibility into
 * silently wrong residues. `volatile` so the store cannot be folded away. */
static inline void __trap()
{
    volatile device uint *p = (volatile device uint *)0;
    *p = 0xdeadbeefu;
}

/* sm_70's __match_any_sync: the mask of participating lanes holding the same
 * value as this one. Metal has no such instruction at any family, so this is
 * the standard ballot/shuffle emulation -- O(distinct values), not O(32).
 *
 * THE MASK ARGUMENT IS IGNORED, as it is for __ballot_sync above, and for the
 * same reason: Metal's SIMD operations act on the ACTIVE lanes. That is also
 * what makes this correct at the one call site -- k_fill_rows calls it inside
 * `if (hit)`, having passed exactly the ballot of hit lanes, so CUDA's
 * participating set and Metal's active set are the same lanes.
 *
 * `todo` is uniform across those lanes, so the loop trip count is uniform and
 * every participant reaches every simd_ballot and every simd_shuffle. The
 * leader is drawn from `todo`, so it is always a participating lane -- a
 * shuffle from an inactive lane would be undefined. */
static inline uint __match_any_sync(uint mask, uint value)
{
    (void)mask;
    uint todo = (uint)((simd_vote::vote_t)simd_ballot(true));
    uint result = 0;
    while (todo) {
        const ushort leader = (ushort)ctz(todo);
        const uint v = simd_shuffle(value, leader);
        const bool eq = (v == value);
        const uint peers = (uint)((simd_vote::vote_t)simd_ballot(eq));
        if (eq) result = peers;
        todo &= ~peers;
    }
    return result;
}

/* ---- fp32 math under CUDA's names -------------------------------------- *
 *
 * Phase 0 measured Metal's fp32 divide and fma as bit-exact against the host
 * wherever neither side is subnormal, so these are renames, not
 * approximations. log2f is the exception and is deliberately NOT metal::log2:
 * that disagrees with the host on 50.03% of inputs by up to 3 ULP, which
 * would move sieve cells. It maps to portable_log2.h's pl_log2f, which
 * computes the same sequence on host and device by construction.
 */
#define fmaf(a, b, c) fma((a), (b), (c))
#define fabsf(x)      fabs((x))
#define fmaxf(a, b)   fmax((a), (b))
#define fminf(a, b)   fmin((a), (b))
#define floorf(x)     floor((x))
#define sqrtf(x)      sqrt((x))
#define log2f(x)      pl_log2f((x))

/* ---- atomics ----------------------------------------------------------- */

/* Metal requires the atomic TYPE at the access rather than at the operation,
 * so a plain device uint* is reinterpreted here. Standard Metal idiom, and
 * confined to these helpers. */
static inline uint atomicAdd(device uint *p, uint v)
{ return atomic_fetch_add_explicit((device atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicSub(device uint *p, uint v)
{ return atomic_fetch_sub_explicit((device atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicOr(device uint *p, uint v)
{ return atomic_fetch_or_explicit((device atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicMin(device uint *p, uint v)
{ return atomic_fetch_min_explicit((device atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicAdd(threadgroup uint *p, uint v)
{ return atomic_fetch_add_explicit((threadgroup atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicOr(threadgroup uint *p, uint v)
{ return atomic_fetch_or_explicit((threadgroup atomic_uint *)p, v, memory_order_relaxed); }

static inline uint atomicSub(threadgroup uint *p, uint v)
{ return atomic_fetch_sub_explicit((threadgroup atomic_uint *)p, v, memory_order_relaxed); }

/* A 64-bit atomic add, on a device with NO 64-bit atomics of any kind
 * (Phase 0: __HAVE_ATOMIC_ULONG__ is never defined on this toolchain).
 *
 * The counter is kept as two 32-bit words, low first, and the carry out of
 * the low add is folded into the high add. Each word is updated atomically
 * and the pair is eventually consistent, which is exactly right for what this
 * is used for -- k_transform's `nlost` diagnostic -- and would NOT be right
 * for anything a later computation depends on. There is only one such site.
 *
 * The host needs no change at all: two little-endian uint32 words at one
 * address ARE a little-endian uint64, so the existing 8-byte readback of
 * these counters reconstructs the value for free.
 *
 * Named atomicAdd64 rather than overloading atomicAdd because an untyped
 * literal (atomicAdd(&cursor[b], L2_CAP)) would otherwise be ambiguous
 * between the 32- and 64-bit forms. The generator rewrites the call sites. */
static inline void atomicAdd64(device uint *p, ulong v)
{
    const uint lo = (uint)v, hi = (uint)(v >> 32);
    const uint old = atomic_fetch_add_explicit((device atomic_uint *)p, lo,
                                               memory_order_relaxed);
    const uint carry = (old + lo < old) ? 1u : 0u;
    atomic_fetch_add_explicit((device atomic_uint *)(p + 1), hi + carry,
                              memory_order_relaxed);
}

#endif  /* CUDA_SIEVE_CUDA_MSL_COMPAT_H */
