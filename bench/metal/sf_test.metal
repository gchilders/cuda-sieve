/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 2 gate, part 2: the device half. Compiles the SAME softfp64.h and
 * portable_log2.h the host test compiles, and evaluates the same operations
 * on the GPU, so the comparison is between two builds of one source rather
 * than between two implementations.
 */
#include "softfp64.h"
#include "portable_log2.h"

using namespace metal;

kernel void k_sf(device ulong *o_add  [[buffer(0)]],
                 device ulong *o_sub  [[buffer(1)]],
                 device ulong *o_mul  [[buffer(2)]],
                 device ulong *o_div  [[buffer(3)]],
                 device ulong *o_fma  [[buffer(4)]],
                 device uint  *o_f32  [[buffer(5)]],
                 device ulong *o_i64  [[buffer(6)]],
                 device uint  *o_lg2  [[buffer(7)]],
                 device const ulong *a [[buffer(8)]],
                 device const ulong *b [[buffer(9)]],
                 device const ulong *c [[buffer(10)]],
                 uint t [[thread_position_in_grid]])
{
    sf64 x = a[t], y = b[t], z = c[t];
    o_add[t] = sf_add(x, y);
    o_sub[t] = sf_sub(x, y);
    o_mul[t] = sf_mul(x, y);
    o_div[t] = sf_div(x, y);
    o_fma[t] = sf_fma(x, y, z);
    o_f32[t] = sf_f2b(sf_to_f32(x));
    o_i64[t] = sf_from_i64((sf_i64)z);
    /* log2 over the float reinterpretation of the low word, forced positive
     * and normal, which is the shape k_apply feeds it. */
    uint fb = (uint)(z & 0x7fffffffu);
    if ((fb >> 23) == 0u || (fb >> 23) == 0xffu) fb = 0x3f800000u;
    o_lg2[t] = sf_f2b(pl_log2f(sf_b2f(fb)));
}
