/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * fbgen_gpu.cu's device half, as MSL.
 *
 * The body is fbgen_gpu_body.metal.inc, generated from fbgen_gpu.cu by
 * metal/gen_fbgen_metal.py and then hand-fixed. Regenerating is deliberately
 * NOT part of the build: the generator is a porting aid, not a source of
 * truth, and the .inc is committed and reviewed like any other source. The
 * script exists so that a future CUDA-side change to this file's device code
 * can be re-diffed against it rather than re-ported from memory.
 *
 * Three things the transformation does, all forced by MSL:
 *
 *  1. Address spaces. MSL rejects an unqualified pointer parameter outright
 *     ("pointer type must have explicit address space qualifier"), so every
 *     `T *` parameter is qualified. Almost all of them are `thread`: the
 *     original code builds polynomials in registers and passes their
 *     addresses, which is exactly the pattern that maps cleanly. The three
 *     exceptions -- d_big_mod, d_big_mod_mont and d_ctx_big -- take a
 *     `constant` gpu_big_t, because their only callers pass &c_alg[i],
 *     &c_y0 or &c_y1.
 *
 *  2. __constant__ globals become kernel parameters. CUDA's c_alg /
 *     c_alg_deg / c_y0 / c_y1 are file-scope; MSL has no such thing, so they
 *     are bound as buffers AFTER the kernel's own parameters (keeping the
 *     "CUDA parameter i -> [[buffer(i)]]" rule intact for the real ones) and
 *     threaded down into d_alg_roots_prime and d_alg_roots_prime_fixed.
 *
 *  3. Kernel bodies are UNCHANGED. Rather than rewriting every
 *     blockIdx.x * blockDim.x + threadIdx.x, the ids arrive as MSL attribute
 *     parameters and FB_KERNEL_IDS re-exposes them under CUDA's names. A
 *     kernel body that is textually identical to the CUDA original cannot
 *     have acquired a transcription bug, which matters more here than
 *     elegance: this file must produce byte-identical output.
 */
#include <metal_stdlib>
using namespace metal;

#define BENCH_MAX_DEGREE 8
#define BENCH_NCOEFF (BENCH_MAX_DEGREE + 1)
#define GPU_FB_BIG_LIMBS 20
#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)
#define GPU_FB_BRUTE_ROOT_LIMIT 67u

typedef struct {
    uint32_t v[GPU_FB_BIG_LIMBS];
    int n;
    int neg;
} gpu_big_t;

typedef struct {
    int deg;
    uint32_t c[BENCH_NCOEFF];
} dpoly_t;

/* CUDA's launch-configuration variables, re-exposed under their own names so
 * the kernel bodies below need no edits at all. */
struct fb_dim { uint x; };
#define FB_KERNEL_IDS                          \
    const fb_dim blockIdx  = { _bid };         \
    const fb_dim threadIdx = { _lid };         \
    const fb_dim blockDim  = { _bdim };        \
    const fb_dim gridDim   = { _gdim };        \
    (void)blockIdx; (void)threadIdx; (void)blockDim; (void)gridDim; \
    (void)_tid; (void)_ntid;

/* CUDA's atomicAdd, so the kernel bodies need no edit. Metal requires the
 * atomic TYPE at the access, not just the operation, and reinterpreting a
 * device uint* as device atomic_uint* is the standard idiom for that. Only
 * ever applied here to the `failures` counters, which are plain uint32
 * everywhere else they are touched. */
static inline void atomicAdd(device uint32_t *p, uint32_t v)
{
    atomic_fetch_add_explicit((device atomic_uint *)p, v, memory_order_relaxed);
}

#include "fbgen_gpu_body.metal.inc"
