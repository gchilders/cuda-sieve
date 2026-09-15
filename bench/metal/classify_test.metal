/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * cof_classify on the device, for comparison against prp.cuh's own fp64 on
 * the host. Phase 2 tested the softfp64 PRIMITIVES device-vs-host and the
 * assembled call sites host-vs-fp64 -- but never cof_classify as prp_msl.h
 * assembles it, ON THE DEVICE. That gap is what this closes.
 */
#include "cuda_msl_compat.h"
#include "softfp64.h"
#include "sf_sites.h"
#include "bigint_msl.h"
#include "prp_msl.h"

kernel void k_classify_probe(device uint8_t *out [[buffer(0)]],
                             device const uint32_t *limbs [[buffer(1)]],
                             device const uint8_t *bits [[buffer(2)]],
                             constant uint32_t &lpb [[buffer(3)]],
                             constant uint32_t &mfb [[buffer(4)]],
                             constant sf64 &lim [[buffer(5)]],
                             constant uint32_t &n [[buffer(6)]],
                             uint t [[thread_position_in_grid]])
{
    if (t >= n) return;
    bn_t c;
    for (int i = 0; i < BN_LIMBS; i++) c.v[i] = limbs[t * BN_LIMBS + i];
    out[t] = (uint8_t)cof_classify(&c, (int)bits[t], lpb, mfb, lim);
}
