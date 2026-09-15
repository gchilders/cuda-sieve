/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 2 gate, part 3: the two real call sites, against the CUDA headers'
 * own hardware-fp64 versions, on the host. `prp.cuh` is compiled unmodified,
 * so the reference is literally the code the CUDA build runs.
 */
#include "prp.cuh"
#include "metal/sf_sites.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static sf_u64 d2b(double d) { sf_u64 b; memcpy(&b, &d, 8); return b; }
static double b2d(sf_u64 b) { double d; memcpy(&d, &b, 8); return d; }

static sf_u64 rs = 0xDEADBEEF12345678ull;
static sf_u64 rnd(void)
{ rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1Dull; }

int main(void)
{
    long nbad_bn = 0, nbad_gap = 0, ngap_hit = 0, n = 0;

    for (int trial = 0; trial < 400000; trial++) {
        bn_t x;
        /* Cofactor-shaped values: mostly small (the real residuals are under
         * ~128 bits), with a tail of wide ones to exercise the rounding. */
        int limbs = 1 + (int)(rnd() % (trial % 8 ? 4 : BN_LIMBS));
        memset(&x, 0, sizeof x);
        for (int i = 0; i < limbs && i < BN_LIMBS; i++) x.v[i] = (uint32_t)rnd();

        double ref = bn_to_double(&x);
        sf64   got = sf_bn_to_double(x.v, BN_LIMBS);
        if (d2b(ref) != got) {
            if (nbad_bn++ < 5)
                printf("  bn_to_double limbs=%d ref=%016llx got=%016llx\n",
                       limbs, (unsigned long long)d2b(ref), (unsigned long long)got);
        }
        n++;

        /* Gap test, over the lpb / lim / bits combinations the siever uses. */
        unsigned lpb  = 29 + (unsigned)(rnd() % 6);          /* 29..34   */
        uint32_t lim  = 1u << (20 + (rnd() % 8));            /* fb bound */
        int bits      = (int)(lpb + 1 + (rnd() % (2 * lpb)));
        double dlim   = (double)lim;

        int ref_gap = 0;
        { double nd = ref, kB = dlim * dlim;
          for (unsigned k = lpb; k < (unsigned)bits; k += lpb, kB *= dlim)
              if (nd < kB) { ref_gap = 1; break; } }
        int got_gap = sf_cof_gap_test(got, bits, lpb, sf_from_u32(lim));
        if (ref_gap) ngap_hit++;
        if (ref_gap != got_gap) {
            if (nbad_gap++ < 5)
                printf("  gap lpb=%u lim=%u bits=%d ref=%d got=%d nd=%.17g\n",
                       lpb, lim, bits, ref_gap, got_gap, ref);
        }
    }

    printf("bn_to_double : %ld cases, %ld mismatches\n", n, nbad_bn);
    printf("gap test     : %ld cases, %ld mismatches (%ld took the GAP branch)\n",
           n, nbad_gap, ngap_hit);
    if (nbad_bn || nbad_gap) { printf("SF SITES GATE: FAIL\n"); return 1; }
    printf("SF SITES GATE: PASS\n");
    return 0;
}
