/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * The two device-side fp64 sequences the siever actually needs, rewritten on
 * softfp64.h. Kept in one header so Phase 6 lifts a tested sequence rather
 * than re-deriving it, and so the Phase 2 gate tests the real call sites and
 * not just the primitives underneath them.
 *
 * Both mirror their CUDA originals operation for operation, in the same order.
 * Where the original is a plain C expression that nvcc is free to contract
 * into an fma (-fmad=true is nvcc's default), the note below says whether the
 * contraction could change the result.
 */
#ifndef CUDA_SIEVE_SF_SITES_H
#define CUDA_SIEVE_SF_SITES_H

#include "softfp64.h"

/* 2^32 as a binary64. */
#define SF_TWO32 ((sf64)0x41f0000000000000ull)

/* prp.cuh:175 bn_to_double:
 *
 *     double d = 0.0;
 *     for (i = BN_LIMBS-1; i >= 0; i--) d = d * 4294967296.0 + (double)x->v[i];
 *
 * CONTRACTION IS HARMLESS HERE, which is worth stating rather than assuming:
 * nvcc may fuse this into fma(d, 2^32, v[i]), but multiplying by a power of
 * two is exact, so the fused and unfused forms round identically. The limb
 * count is a template parameter so the caller's BN_LIMBS drives it. */
SF_FN sf64 sf_bn_to_double(const SF_BN_PTR sf_u32 *v, int nlimbs)
{
    sf64 d = 0;
    for (int i = nlimbs - 1; i >= 0; i--)
        d = sf_add(sf_mul(d, SF_TWO32), sf_from_u32(v[i]));
    return d;
}

/* prp.cuh:199 cof_classify's gap test:
 *
 *     double nd = bn_to_double(n);
 *     double kB = lim * lim;
 *     for (klpb = lpb; klpb < bits; klpb += lpb, kB *= lim)
 *         if (nd < kB) return COF_REJECT_GAP;
 *
 * Pure multiplies and one comparison, so there is nothing for a compiler to
 * contract. Returns 1 when the original returns COF_REJECT_GAP. */
SF_FN int sf_cof_gap_test(sf64 nd, int bits, unsigned lpb, sf64 lim)
{
    sf64 kB = sf_mul(lim, lim);
    for (unsigned klpb = lpb; klpb < (unsigned)bits; klpb += lpb, kB = sf_mul(kB, lim))
        if (sf_lt(nd, kB)) return 1;
    return 0;
}

#endif  /* CUDA_SIEVE_SF_SITES_H */
