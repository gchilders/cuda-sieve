/* Single-threaded ground truth for the lattice walk.
 *
 * The design doc's first correctness gate is "fill+apply parity against our
 * own reference on the real FB". This is that reference. It shares
 * plattice.cuh with the GPU, so it validates the walk *algorithm* against
 * brute force, not the GPU against itself. */
#include "bench.h"
#include "plattice.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Brute-force enumerate {x : i == r*j mod p} and compare against pl_*(). */
static int check_one(uint32_t p, uint32_t r, int logI, uint32_t J)
{
    const uint32_t I = 1u << logI, Imask = I - 1;
    const uint32_t xmax = I * J;
    uint32_t *ref = (uint32_t *)malloc((size_t)(xmax / p + J + 16) * 4);
    uint32_t nref = 0, x, n = 0;
    plat_t P;
    int64_t j;

    for (j = 0; j < (int64_t)J; j++) {
        /* i in [-I/2, I/2) with i == r*j (mod p) */
        int64_t base = ((int64_t)r * j) % p;          /* in [0,p) */
        int64_t i;
        for (i = base; i < (int64_t)I / 2; i += p)
            if (i >= -(int64_t)I / 2) ref[nref++] = (uint32_t)(i + I / 2 + (j << logI));
        for (i = base - p; i >= -(int64_t)I / 2; i -= p)
            ref[nref++] = (uint32_t)(i + I / 2 + (j << logI));
    }
    /* ref is grouped by j and each j-group is unsorted; sort by value */
    for (uint32_t a = 1; a < nref; a++) {            /* insertion sort, small */
        uint32_t v = ref[a]; int32_t b = (int32_t)a - 1;
        while (b >= 0 && ref[b] > v) { ref[b + 1] = ref[b]; b--; }
        ref[b + 1] = v;
    }

    P = pl_make(p, r, logI);
    for (x = pl_start(logI); x < xmax; x = pl_next(x, &P, Imask)) {
        if (n >= nref || ref[n] != x) {
            fprintf(stderr,
                "verify_walk MISMATCH p=%u r=%u logI=%d J=%u: step %u walk=%u ref=%u\n",
                p, r, logI, J, n, x, n < nref ? ref[n] : 0xFFFFFFFFu);
            free(ref); return -1;
        }
        n++;
    }
    if (n != nref) {
        fprintf(stderr, "verify_walk SHORT p=%u r=%u: walk %u of %u points\n",
                p, r, n, nref);
        free(ref); return -1;
    }
    free(ref);
    return 0;
}

/* Gate the root transform against its *definition*, which is the one thing the
 * walk check and the GPU/CPU cross-check cannot see: a position (i,j) is hit by
 * (p,r) exactly when the (a,b) it maps to satisfies a == r*b (mod p). Checking
 * the walk only proves we enumerate whatever lattice pl_transform named; this
 * proves it named the right one. Returns 0 on success. */
int verify_transform(const qlat_t *L, int ncheck)
{
    static const uint32_t ps[] = {3,5,7,11,13,17,101,1009,65537,1000003};
    int k, bad = 0;
    for (k = 0; k < (int)(sizeof ps / sizeof ps[0]) && k < ncheck; k++) {
        uint32_t p = ps[k], r;
        for (r = 0; r <= p; r += (p > 32 ? p / 8 + 1 : 1)) {
            uint32_t rt = (r == p) ? pl_transform_proj(p, L->a0, L->a1, L->b0, L->b1)
                                   : pl_transform(p, r, L->a0, L->a1, L->b0, L->b1);
            int32_t i; uint32_t j;
            if (rt == PL_ROWS || rt >= p) continue;   /* covered by other cases */
            for (j = 1; j < 40; j++) {
                /* the walk claims i == rt*j (mod p) is a hit; check the norm */
                i = (int32_t)(((uint64_t)rt * j) % p);
                {
                    int64_t a = (int64_t)i * L->a0 + (int64_t)j * L->b0;
                    int64_t b = (int64_t)i * L->a1 + (int64_t)j * L->b1;
                    int64_t lhs = (r == p) ? b : (a - (int64_t)r * b);
                    if (lhs % (int64_t)p != 0) {
                        fprintf(stderr,
                            "verify_transform MISMATCH p=%u r=%u rt=%u at (i=%d,j=%u)\n",
                            p, r, rt, i, j);
                        bad++; break;
                    }
                }
            }
            if (r == p) break;
        }
        /* also exercise the projective entry explicitly */
        {
            uint32_t rt = pl_transform_proj(p, L->a0, L->a1, L->b0, L->b1);
            uint32_t j;
            if (rt != PL_ROWS)
                for (j = 1; j < 40; j++) {
                    int32_t i = (int32_t)(((uint64_t)rt * j) % p);
                    int64_t b = (int64_t)i * L->a1 + (int64_t)j * L->b1;
                    if (b % (int64_t)p != 0) {
                        fprintf(stderr, "verify_transform PROJ MISMATCH p=%u rt=%u\n", p, rt);
                        bad++; break;
                    }
                }
        }
    }
    return bad ? -1 : 0;
}

int verify_walk(int logI, uint32_t J, int nprimes)
{
    /* Primes just above I are the interesting regime: that is where the
     * FK reduction actually has work to do. */
    const uint32_t I = 1u << logI;
    uint32_t p, checked = 0;
    for (p = I + 1; checked < (uint32_t)nprimes && p < 40u * I; p += 2) {
        uint32_t d, isprime = 1;
        for (d = 3; (uint64_t)d * d <= p; d += 2) if (p % d == 0) { isprime = 0; break; }
        if (!isprime) continue;
        /* a spread of roots including the r == 0 special case */
        for (uint32_t k = 0; k < 5; k++) {
            uint32_t r = (uint32_t)(((uint64_t)p * k) / 5);
            if (check_one(p, r, logI, J) != 0) return -1;
        }
        checked++;
    }
    return (int)checked;
}

/* Ground truth for the apply kernel: replay one region's bucket records into a
 * 16-bit cell array exactly as k_apply should, and apply the same threshold.
 * Reproduces the GPU's *sign convention* too -- init to CINIT - T, add, test
 * >= CINIT -- so a mismatch means a real kernel bug and not a modelling
 * difference. */
uint32_t verify_apply_region(const uint32_t *records, uint32_t nrec,
                             const uint16_t *slice_logp,
                             const norm_t *N, int logI, int log_region,
                             uint32_t region, int norm_mode,
                             uint32_t Cinit, uint32_t tconst,
                             const uint32_t *sp, const uint32_t *srt,
                             const uint32_t *sg, const uint16_t *slp,
                             uint32_t nsmall, uint16_t *cells)
{
    const uint32_t ncell = 1u << log_region;
    const uint32_t Imask = (1u << logI) - 1;
    const int32_t  Ihalf = 1 << (logI - 1);
    const uint32_t xbase = region << log_region;
    uint32_t c, surv = 0;

    for (c = 0; c < ncell; c++) {
        uint32_t t;
        if (norm_mode == NORM_CONST) t = tconst;
        else {
            uint32_t x = xbase + c;
            float lg = norm_target_host(N, (int32_t)(x & Imask) - Ihalf, x >> logI);
            int ti = (int)floorf(lg + 0.5f);
            t = (ti < 0) ? 0u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
        }
        cells[c] = (uint16_t)(Cinit - t);
    }
    /* small primes, replayed independently of the GPU's tiering: a flat loop
     * over every entry, so a bug in the block/warp/thread split shows up as a
     * cell difference rather than hiding behind matching code */
    {
        const uint32_t j = xbase >> logI;
        const int32_t ilo = (int32_t)(xbase & Imask) - Ihalf;
        uint32_t e;
        for (e = 0; e < nsmall; e++) {
            uint32_t m = sp[e], rt = srt[e], g = sg[e], lp = slp[e], off, jj;
            if (g > 1 && (j % g)) continue;
            jj = (g > 1) ? j / g : j;
            {
                int32_t base = (int32_t)(((uint64_t)rt * jj) % m);
                int32_t s = (base - ilo) % (int32_t)m;
                if (s < 0) s += (int32_t)m;
                for (off = (uint32_t)s; off < ncell; off += m) cells[off] += (uint16_t)lp;
            }
        }
    }
    for (c = 0; c < nrec; c++) {
        uint32_t r = records[c];
        cells[r & (ncell - 1)] += slice_logp[r >> 16];
    }
    for (c = 0; c < ncell; c++) if (cells[c] >= Cinit) surv++;
    return surv;
}

uint64_t verify_count_updates(const fb_t *fb, const qlat_t *L,
                              int logI, uint32_t J)
{
    const uint32_t I = 1u << logI, Imask = I - 1;
    const uint32_t xmax = I * J;
    uint64_t total = 0, nproj = 0;
    uint32_t k;

    for (k = 0; k < fb->n; k++) {
        uint32_t p = fb->primes[k];
        uint32_t rt = pl_transform(p, fb->roots[k], L->a0, L->a1, L->b0, L->b1);
        plat_t P;
        uint32_t x;
        if (rt >= p) { nproj++; continue; }
        P = pl_make(p, rt, logI);
        for (x = pl_first(&P, logI); x < xmax; x = pl_next(x, &P, Imask)) total++;
    }
    if (nproj)
        fprintf(stderr, "  (cpu ref: %llu projective transformed roots skipped)\n",
                (unsigned long long)nproj);
    return total;
}
