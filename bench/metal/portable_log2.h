/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * A log2f that is bit-identical on the host and on the GPU, by construction.
 *
 * WHY THIS EXISTS. bench_kernels.cu:543 already records that the sieve needs
 * the accurate log2f and not the __log2f intrinsic, because the result is
 * rounded to a sieve cell and CPU parity depends on it. CUDA and glibc agree
 * there because both can evaluate a float log2 in fp64 internally. Apple GPUs
 * have no fp64, and the Phase 0 probe measured metal::log2 disagreeing with
 * the host's log2f on 50.03% of inputs by up to 3 ULP -- with
 * metal::precise::log2 returning bit-identical results to plain log2, so the
 * precise:: namespace is not a fix.
 *
 * So the only way to get one answer on both sides is to compute the same
 * sequence of operations on both sides. That is all this file is.
 *
 * WHY IT IS ALLOWED TO USE / AND fma. A Phase 0 probe compared 2^20 random
 * fp32 divisions and fmas between the M3 and the host: among results where
 * neither side was subnormal, ZERO differed. Apple's GPU flushes subnormals
 * to zero, but is otherwise correctly rounded for both operations. Every
 * intermediate below is bounded well inside the normal range -- f is in
 * [1/sqrt2, sqrt2), f+1 is in [1.7, 2.5), |t| <= 0.172 -- so FTZ is
 * unreachable here and both operations are exact on both sides.
 *
 * METHOD. Reduce x = 2^e * f with f in [1/sqrt2, sqrt2), then
 *
 *     log2(f) = (2/ln2) * t * (1 + t^2/3 + t^4/5 + t^6/7 + t^8/9),
 *     t = (f-1)/(f+1)
 *
 * The series is atanh's, so it converges on |t| <= 0.172 fast: the first
 * omitted term is t^10/11 <= 2.1e-9, an order of magnitude under fp32's
 * 6e-8 epsilon, and the polynomial is evaluated as an explicit fma chain so
 * neither compiler can reassociate it into something different from the
 * other.
 *
 * Phase 7 decides the SCOPE of this file: Metal-only, or also compiled into
 * the CUDA build behind -DNORM_PORTABLE_LOG2 so both can be compared exactly.
 * Nothing here presumes that decision.
 */
#ifndef CUDA_SIEVE_PORTABLE_LOG2_H
#define CUDA_SIEVE_PORTABLE_LOG2_H

#include "msl_compat.h"

#define PL_FN SF_FN

/* Mantissa field of sqrt(2): the reduction threshold. */
#define PL_SQRT2_MANT 0x3504F3u

PL_FN float pl_log2f(float x)
{
    sf_u32 b = sf_f2b(x);

    if (b >= 0x7f800000u) {              /* +inf, or negative / NaN */
        if (b == 0x7f800000u) return x;                     /* +inf  */
        return sf_b2f(0x7fc00000u);                         /* NaN   */
    }

    sf_i32 e = 0;
    if (b < 0x00800000u) {               /* zero or subnormal */
        if (b == 0u) return sf_b2f(0xff800000u);            /* -inf  */
        /* Scale by 2^24. Exact: it only moves the exponent field. */
        x = x * 16777216.0f;
        b = sf_f2b(x);
        e = -24;
    }

    e += (sf_i32)((b >> 23) & 0xffu) - 127;
    sf_u32 m = b & 0x7fffffu;

    /* Reduce to f in [1/sqrt2, sqrt2) by borrowing one power of two. Done on
     * the bit pattern, so it is exact and branch-symmetric. */
    float f;
    if (m >= PL_SQRT2_MANT) { e += 1; f = sf_b2f(m | 0x3f000000u); }  /* [0.707,1)   */
    else                    {         f = sf_b2f(m | 0x3f800000u); }  /* [1, 1.4142) */

    const float t = (f - 1.0f) / (f + 1.0f);
    const float u = t * t;

    /* 1 + u/3 + u^2/5 + u^3/7 + u^4/9, as an explicit fma chain. */
    float p = 0.111111111111111111f;                      /* 1/9 */
    p = sf_fmaf(u, p, 0.142857142857142857f);             /* 1/7 */
    p = sf_fmaf(u, p, 0.2f);                              /* 1/5 */
    p = sf_fmaf(u, p, 0.333333333333333333f);             /* 1/3 */
    p = sf_fmaf(u, p, 1.0f);

    /* 2/ln(2). Parenthesised so the order is fixed on both compilers. */
    const float lg = (t * p) * 2.88539008177792681f;

    return (float)e + lg;
}

#endif  /* CUDA_SIEVE_PORTABLE_LOG2_H */
