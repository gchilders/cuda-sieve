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
#include <string.h>
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

/* Does the factor-base entry (q, r_enc) hit position (i,j)? Straight from the
 * definition, with no lattice algebra: map (i,j) to (a,b) and test the
 * congruence the encoding names. This is the ground truth. */
static int hits_def(uint32_t q, uint32_t r_enc, const qlat_t *L,
                    int32_t i, int32_t j)
{
    const int64_t a = (int64_t)i * L->a0 + (int64_t)j * L->b0;
    const int64_t b = (int64_t)i * L->a1 + (int64_t)j * L->b1;
    const int64_t lhs = (r_enc >= q) ? (a * (int64_t)(r_enc - q) - b)
                                     : (a - (int64_t)r_enc * b);
    return lhs % (int64_t)q == 0;
}

/* What pl_transform_enc claims: solutions exist only on rows j == 0 (mod g),
 * and there i == rt*(j/g) (mod m). */
static int hits_pred(uint32_t m, uint32_t rt, uint32_t g, int32_t i, int32_t j)
{
    int64_t d;
    if (g > 1 && (j % (int32_t)g)) return 0;
    if (m == 1) return 1;
    d = (int64_t)i - (int64_t)rt * (g > 1 ? j / (int32_t)g : j);
    return d % (int64_t)m == 0;
}

/* Gate the root transform against its definition, by SET EQUALITY over a small
 * box. Two things make this stronger than the check it replaces:
 *
 *   - the old test verified only that each predicted hit satisfies the
 *     congruence -- one direction. A transform naming a proper sublattice
 *     (right congruence, half the hits) passed it. This compares both
 *     directions, so a missing hit fails too.
 *   - it drives pl_transform_enc through the encoding, so PROJECTIVE roots
 *     with a nonzero reciprocal are covered. That is the case the factor base
 *     actually contains (c183.fb1 has 35 of them) and the case that was wrong.
 *
 * Every PRIME POWER 2..QMAX and every root in [0, 2q) is exercised: primes,
 * powers, even moduli, affine and projective. The transform is pure lattice
 * algebra and needs no (q,r) to be a genuine root of anything -- it only needs
 * q coprime to the determinant, which is the special-q.
 *
 * Composite moduli with two distinct prime factors are deliberately NOT
 * checked: pl_transform_gen does not claim them (see its precondition), and no
 * factor base contains them. `fb_check_prime_powers` is the gate that keeps
 * that assumption true. */
int verify_transform(const qlat_t *L, int ncheck)
{
    const int32_t IL = 32, JH = 32;
    uint32_t q;
    int bad = 0;
    const uint32_t QMAX = (ncheck > 2 ? (uint32_t)ncheck : 200u);

    for (q = 2; q <= QMAX; q++) {
        uint32_t r_enc;
        if (!fb_is_prime_power(q)) continue;  /* outside the contract */
        if (L->q % q == 0) continue;          /* q | det: no such ideal */
        for (r_enc = 0; r_enc < 2 * q; r_enc++) {
            uint32_t rt, g;
            const uint32_t m = pl_transform_enc(q, r_enc, L->a0, L->a1,
                                                L->b0, L->b1, &rt, &g);
            int32_t i, j;
            uint64_t nd = 0, np = 0;
            for (j = 0; j < JH; j++)
                for (i = -IL; i < IL; i++) {
                    const int d = hits_def(q, r_enc, L, i, j);
                    const int p = hits_pred(m, rt, g, i, j);
                    nd += (unsigned)d; np += (unsigned)p;
                    if (d != p) {
                        fprintf(stderr,
                            "verify_transform MISMATCH q=%u r=%u%s -> (m=%u,rt=%u,g=%u)"
                            " at (i=%d,j=%d): definition=%d predicted=%d\n",
                            q, r_enc, r_enc >= q ? " [projective]" : "",
                            m, rt, g, i, j, d, p);
                        bad++;
                        goto next_root;
                    }
                }
            if (nd == 0 || np == 0) {          /* an empty lattice is a red flag */
                fprintf(stderr, "verify_transform EMPTY q=%u r=%u\n", q, r_enc);
                bad++;
            }
        next_root: ;
        }
    }
    return bad ? -1 : 0;
}

/* The same gate, driven by a real factor base rather than synthetic moduli:
 * every entry's transform is checked by set equality over a small box. This is
 * what catches an encoding that the loader got wrong but the algebra handles.
 * Returns the number of entries checked, or -1. */
int verify_fb_transform(const fb_t *fb, const qlat_t *L, uint32_t maxq)
{
    const int32_t IL = 24, JH = 24;
    uint32_t k, checked = 0;
    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k], r_enc = fb->roots[k];
        uint32_t rt, g, m;
        int32_t i, j;
        if (q > maxq) continue;
        if (L->q % q == 0) continue;
        m = pl_transform_enc(q, r_enc, L->a0, L->a1, L->b0, L->b1, &rt, &g);
        for (j = 0; j < JH; j++)
            for (i = -IL; i < IL; i++)
                if (hits_def(q, r_enc, L, i, j) != hits_pred(m, rt, g, i, j)) {
                    fprintf(stderr,
                        "verify_fb_transform MISMATCH entry %u: q=%u r=%u%s"
                        " -> (m=%u,rt=%u,g=%u) at (i=%d,j=%d)\n",
                        k, q, r_enc, r_enc >= q ? " [projective]" : "",
                        m, rt, g, i, j);
                    return -1;
                }
        checked++;
    }
    return (int)checked;
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
            /* ceiling is the cell's, not the byte's -- see k_apply */
            const uint32_t TMAX = (Cinit > 255u) ? Cinit : 255u;
            t = (ti < 0) ? 0u : ((uint32_t)ti > TMAX ? TMAX : (uint32_t)ti);
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

/* Reference record count, and -- when `per_region` is non-NULL -- the reference
 * count for every region.
 *
 * A global total is a weak gate: it cannot tell "right total, wrong region"
 * from "right". The transposed-basis bug this project already wrote up was
 * exactly that shape, and so was the projective-reciprocal bug (same density,
 * wrong congruence). Comparing all 32K region counts against the GPU's cursors
 * gates PLACEMENT, at the same host cost.
 *
 * This shares pl_transform_enc with the device, so it validates the walk and
 * the fill, not the transform -- verify_transform is what gates that, against
 * the definition. */
uint64_t verify_count_updates(const fb_t *fb, const qlat_t *L,
                              int logI, uint32_t J,
                              int log_region, uint32_t *per_region)
{
    const uint32_t I = 1u << logI, Imask = I - 1;
    const uint32_t xmax = I * J;
    uint64_t total = 0, nrow = 0;
    uint32_t k;

    if (per_region)
        memset(per_region, 0, (size_t)(xmax >> log_region) * sizeof *per_region);

    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k];
        uint32_t rt, g, m;
        plat_t P;
        uint32_t x;
        m = pl_transform_enc(q, fb->roots[k], L->a0, L->a1, L->b0, L->b1, &rt, &g);
        if (g > 1) { nrow++; continue; }      /* the kernel drops these too */
        P = pl_make(m, rt, logI);
        for (x = pl_first(&P, logI); x < xmax; x = pl_next(x, &P, Imask)) {
            total++;
            if (per_region) per_region[x >> log_region]++;
        }
    }
    if (nrow)
        fprintf(stderr, "  (cpu ref: %llu row-confined transformed roots skipped)\n",
                (unsigned long long)nrow);
    return total;
}
