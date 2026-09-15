/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 5 gate: the Metal sieve against the tree's own CPU ground truth.
 *
 * Two independent references, both already in the tree as plain host C, and
 * neither written for this port:
 *
 *   verify_count_updates()  -- walks the whole factor base single-threaded and
 *                              counts, PER REGION, how many updates the
 *                              lattice walk produces. bench_kernels.cu:2617
 *                              explains why per-region and not just the total:
 *                              "every placement bug this project has hit
 *                              (transposed basis, projective reciprocal) had
 *                              exactly the right total".
 *   verify_apply_region()   -- replays one region's bucket records on the CPU
 *                              into a 16-bit cell array, including the norm
 *                              initialisation and the threshold test, and is
 *                              the ground truth for k_apply.
 *
 * The factor base comes from afb_build_gpu -- our own Phase 4 code, already
 * gated byte-identical against the CPU generator -- so this harness needs no
 * input files beyond the polynomial.
 *
 * The special-q is the oracle's: q = 120000053, rho = 112625526, the pair the
 * recorded CADO run used, so the geometry being sieved is a real one.
 */
#include "bench.h"
#include "plattice.cuh"
#include "metal/metal_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstddef>

#define CK(call) do { mtlError_t _e = (call); if (_e != mtlSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, mtlGetErrorString(_e)); return 1; } } while (0)

int main(int argc, char **argv)
{
    const char *poly_path = "../oracle/c183.poly";
    uint64_t q = 120000053, rho = 112625526;
    int logI = 13; uint32_t J = 2048; int log_region = 13;
    uint32_t lim = 300000; int maxbits = 15; double scale = 1.925;
    uint32_t BOUND = 150; uint32_t ncheck = 16;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) poly_path = argv[++i];
        else if (!strcmp(argv[i], "--lim") && i + 1 < argc) lim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--logI") && i + 1 < argc) logI = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--J") && i + 1 < argc) J = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--region") && i + 1 < argc) log_region = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bound") && i + 1 < argc) BOUND = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ncheck") && i + 1 < argc) ncheck = (uint32_t)strtoul(argv[++i], 0, 10);
    }

    poly_t P;
    if (poly_load(poly_path, &P)) { fprintf(stderr, "cannot load %s\n", poly_path); return 1; }

    qlat_t L;
    qlat_build(&L, q, rho, P.skew);

    norm_t N;
    memset(&N, 0, sizeof N);
    norm_setup(&N, &P, &L, logI, J, scale, /*is_sqside=*/1);

    fb_t fb;
    memset(&fb, 0, sizeof fb);
    printf("building the factor base on the GPU (lim=%u, maxbits=%d)...\n", lim, maxbits);
    if (afb_build_gpu(&P, lim, maxbits, scale, 0, 0, 0, &fb)) {
        fprintf(stderr, "afb_build_gpu failed\n"); return 1;
    }
    printf("factor base: %u ideals\n", fb.n);

    uint16_t *hslice = NULL, *hlogp = NULL;
    uint32_t nslice_pow2 = 1;
    if (fb_build_slices(&fb, &hslice, &hlogp, &nslice_pow2) < 0) {
        fprintf(stderr, "fb_build_slices failed\n"); return 1;
    }

    const uint32_t xmax = (1u << logI) * J;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t ncell = 1u << log_region;
    const size_t smem = mtl_apply_smem(ncell, 16);
    printf("geometry: logI %d, J %u, xmax %u, %u regions of %u cells;"
           " %zu B threadgroup memory\n", logI, J, xmax, nregion, ncell, smem);
    CK(mtlFuncSetMaxThreadgroupMemory("k_apply_16_1_1_0", smem));

    /* ---- CPU reference for the fill, first: it also sizes the buckets ---- */
    std::vector<uint32_t> ref(nregion);
    printf("CPU reference (single-threaded) for %u regions...\n", nregion);
    const uint64_t ref_total = verify_count_updates(&fb, &L, logI, J, log_region, ref.data());

    /* A SECOND CPU reference, walking in 64 bits.
     *
     * verify_count_updates uses pl_first/pl_next, the 32-bit walk, whose
     * pl_add32_sat saturates to UINT32_MAX whenever an increment's high word
     * is nonzero -- which ENDS the walk. k_fill_atomic uses pl_first64 /
     * pl_next64, where the same increment wraps and the walk continues. They
     * are therefore different walks, and on this factor base they disagree.
     * That is a property of the CUDA tree, not of this port, so the gate
     * compares against both and says which the GPU reproduces. */
    std::vector<uint32_t> ref64(nregion, 0);
    uint64_t ref64_total = 0;
    {
        const uint32_t Imask = (1u << logI) - 1;
        for (uint32_t k = 0; k < fb.n; k++) {
            uint32_t rt, g;
            uint32_t m = pl_transform_enc(fb.primes[k], fb.roots[k],
                                          L.a0, L.a1, L.b0, L.b1, &rt, &g);
            if (g > 1) continue;
            plat_t Q = pl_make(m, rt, logI);
            for (uint64_t x = pl_first64(&Q, logI); x < xmax; x = pl_next64(x, &Q, Imask)) {
                ref64_total++;
                ref64[(uint32_t)x >> log_region]++;
            }
        }
    }
    printf("reference (64-bit walk): %llu updates\n", (unsigned long long)ref64_total);
    uint32_t refmax = 0;
    for (uint32_t i = 0; i < nregion; i++) if (ref[i] > refmax) refmax = ref[i];
    const uint32_t cap = refmax + (refmax >> 2) + 64;      /* 25% headroom */
    printf("reference: %llu updates, max region %u -> bucket cap %u\n",
           (unsigned long long)ref_total, refmax, cap);

    /* ---- device buffers ---- */
    uint32_t *d_primes, *d_roots, *d_nproj, *d_nlost, *d_cursor, *d_overflow, *d_nsurv;
    uint16_t *d_slice, *d_slice_logp, *d_dbg;
    uint8_t  *d_out;
    void     *d_plat;
    uint32_t *d_dummy; uint16_t *d_dummy16; uint32_t *d_probe;
    const size_t plat_sz = 24;    /* sizeof(plat_t): 2 x uint64 + 2 x uint32 */
    CK(mtlMalloc((void **)&d_primes, (size_t)fb.n * 4));
    CK(mtlMalloc((void **)&d_roots,  (size_t)fb.n * 4));
    CK(mtlMalloc((void **)&d_plat,   (size_t)fb.n * plat_sz));
    CK(mtlMalloc((void **)&d_slice,  (size_t)fb.n * 2));
    CK(mtlMalloc((void **)&d_slice_logp, (size_t)nslice_pow2 * 2));
    CK(mtlMalloc((void **)&d_nproj, 4));
    CK(mtlMalloc((void **)&d_nlost, 8));
    CK(mtlMalloc((void **)&d_cursor, (size_t)nregion * 4));
    CK(mtlMalloc((void **)&d_out, (size_t)nregion * cap * 4));
    CK(mtlMalloc((void **)&d_overflow, 4));
    CK(mtlMalloc((void **)&d_nsurv, 4));
    CK(mtlMalloc((void **)&d_dbg, (size_t)ncell * 2));
    CK(mtlMalloc((void **)&d_probe, 8));
    CK(mtlMalloc((void **)&d_dummy, 16));
    CK(mtlMalloc((void **)&d_dummy16, 16));
    memcpy(d_primes, fb.primes, (size_t)fb.n * 4);
    memcpy(d_roots,  fb.roots,  (size_t)fb.n * 4);
    memcpy(d_slice,  hslice,    (size_t)fb.n * 2);
    memcpy(d_slice_logp, hlogp, (size_t)nslice_pow2 * 2);
    CK(mtlMemset(d_nproj, 0, 4));
    CK(mtlMemset(d_nlost, 0, 8));
    CK(mtlMemset(d_cursor, 0, (size_t)nregion * 4));
    CK(mtlMemset(d_overflow, 0, 4));
    CK(mtlMemset(d_nsurv, 0, 4));
    CK(mtlMemset(d_probe, 0, 8));

    int fails = 0;

    /* ---- layout cross-check, before anything depends on it ---- */
    {
        uint32_t *lay = NULL;
        CK(mtlMalloc((void **)&lay, 16));
        CK(mtlMemset(lay, 0, 16));
        CK(MTL_LAUNCH(k_layout_check, 1, 1, 0, 0, lay));
        CK(mtlDeviceSynchronize());
        const uint32_t h_norm = (uint32_t)sizeof(norm_t), h_plat = (uint32_t)plat_sz;
        const uint32_t h_dd = (uint32_t)offsetof(norm_t, dd);
        const uint32_t h_a0 = (uint32_t)offsetof(norm_t, a0);
        printf("layout: sizeof(norm_t) host %u device %u; offsetof(dd) %u/%u;"
               " offsetof(a0) %u/%u; sizeof(plat_t) host %u device %u\n",
               h_norm, lay[0], h_dd, lay[2], h_a0, lay[3], h_plat, lay[1]);
        if (lay[0] != h_norm || lay[1] != h_plat || lay[2] != h_dd || lay[3] != h_a0) {
            printf("  FAIL: host and device disagree on a shared struct layout\n");
            fails++;
        } else {
            printf("  PASS: shared struct layouts agree\n");
        }
    }

    /* ---- stage 1: transform ---- */
    CK(MTL_LAUNCH(k_transform_0, 256, 256, 0, 0,
                  (const uint32_t *)d_primes, (const uint32_t *)d_roots, d_plat,
                  fb.n, logI, J, L.a0, L.a1, L.b0, L.b1,
                  d_nproj, d_nlost, (uint64_t *)nullptr));
    CK(mtlDeviceSynchronize());
    printf("\ntransform: %u projective, %llu positions lost to them\n",
           *d_nproj, *(unsigned long long *)d_nlost);

    /* ---- stage 2: fill, then the per-region gate ---- */
    CK(MTL_LAUNCH(k_fill_atomic_4_0, 1024, 32, 0, 0,
                  (const void *)d_plat, (const uint16_t *)d_slice,
                  fb.n, xmax, logI, log_region, d_cursor, d_out, cap, d_overflow,
                  (const uint64_t *)nullptr, (uint64_t *)nullptr));
    CK(mtlDeviceSynchronize());

    if (*d_overflow) {
        printf("  FAIL: %u records overflowed their bucket\n", *d_overflow);
        fails++;
    }
    {
        uint64_t gpu_total = 0; uint32_t bad = 0, first = 0;
        for (uint32_t i = 0; i < nregion; i++) {
            gpu_total += d_cursor[i];
            if (d_cursor[i] != ref[i]) { if (!bad) first = i; bad++; }
        }
        uint32_t bad64 = 0, first64 = 0;
        for (uint32_t i = 0; i < nregion; i++)
            if (d_cursor[i] != ref64[i]) { if (!bad64) first64 = i; bad64++; }

        printf("fill: GPU %llu updates; CPU 32-bit walk %llu; CPU 64-bit walk %llu\n",
               (unsigned long long)gpu_total, (unsigned long long)ref_total,
               (unsigned long long)ref64_total);
        printf("  vs verify_count_updates (32-bit walk): %u of %u regions differ\n",
               bad, nregion);
        if (bad64 || gpu_total != ref64_total) {
            printf("  FAIL vs the 64-bit walk: %u of %u regions differ"
                   " (first region %u: gpu %u ref %u)\n",
                   bad64, nregion, first64, d_cursor[first64], ref64[first64]);
            fails++;
        } else {
            printf("  PASS: all %u regions match the 64-bit CPU walk EXACTLY,\n"
                   "        which is the walk k_fill_atomic actually performs\n", nregion);
            if (bad)
                printf("        (the 32-bit reference differs because pl_add32_sat\n"
                       "         saturates and ends the walk where pl_next64 wraps\n"
                       "         and continues -- a CUDA-tree property, see notes)\n");
        }
    }

    /* ---- stage 3: apply, then the per-cell gate ---- */
    const uint32_t CINIT = 4096u;
    float t = norm_target_host(&N, 0, J / 2);
    int ti = (int)(t + 0.5f);
    const uint32_t tconst = (ti < 1) ? 1u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
    const uint32_t THRESH = CINIT - BOUND;
    const uint32_t dbgreg = nregion / 2;
    printf("\napply: T(i=0,j=J/2) = %u, CINIT %u, threshold %u, debug region %u\n",
           tconst, CINIT, THRESH, dbgreg);

    CK(MTL_LAUNCH(k_apply_16_1_1_0, nregion, 256, smem, 0,
                  (const uint32_t *)d_out, (const uint32_t *)d_cursor, cap,
                  logI, log_region, (const uint16_t *)d_slice_logp, nslice_pow2,
                  N, CINIT, THRESH, tconst,
                  (uint8_t *)nullptr, d_nsurv, d_dbg, dbgreg,
                  (const uint32_t *)d_dummy, (const uint32_t *)d_dummy,
                  (const uint32_t *)d_dummy, (const uint16_t *)d_dummy16,
                  (const uint32_t *)d_dummy,
                  0u, 0u, 0u, 0xFFFFFFFFu, d_probe,
                  (uint32_t *)nullptr, 0, 0u));
    CK(mtlDeviceSynchronize());
    printf("survivors: %u of %u positions\n", *d_nsurv, xmax);

    {
        /* Many regions, not one. A single region is 8192 cells, far too small
         * a sample for the log2 question Phase 7 has to answer, and region 0
         * is the j=0 row and legitimately almost empty. */
        if (ncheck > nregion) ncheck = nregion;
        std::vector<uint16_t> href(ncell);
        uint64_t cells = 0, bad = 0, surv_cpu = 0, surv_gpu_cells = 0; long worst = 0;
        uint32_t badregions = 0, firstreg = 0, firstcell = 0;
        for (uint32_t r = 0; r < ncheck; r++) {
            const uint32_t reg = 1 + (uint32_t)((uint64_t)r * (nregion - 1) / ncheck);
            uint32_t nr = d_cursor[reg] > cap ? cap : d_cursor[reg];
            /* k_apply only writes dbg_cells for its dbg_region, so re-run it
             * pointed at this one. */
            CK(mtlMemset(d_dbg, 0, (size_t)ncell * 2));
            CK(mtlMemset(d_nsurv, 0, 4));
            CK(MTL_LAUNCH(k_apply_16_1_1_0, nregion, 256, smem, 0,
                          (const uint32_t *)d_out, (const uint32_t *)d_cursor, cap,
                          logI, log_region, (const uint16_t *)d_slice_logp, nslice_pow2,
                          N, CINIT, THRESH, tconst,
                          (uint8_t *)nullptr, d_nsurv, d_dbg, reg,
                          (const uint32_t *)d_dummy, (const uint32_t *)d_dummy,
                          (const uint32_t *)d_dummy, (const uint16_t *)d_dummy16,
                          (const uint32_t *)d_dummy,
                          0u, 0u, 0u, 0xFFFFFFFFu, d_probe,
                          (uint32_t *)nullptr, 0, 0u));
            CK(mtlDeviceSynchronize());
            /* verify_apply_region's OWN return value counts cells >= Cinit,
             * i.e. survivors at BOUND = 0, so it is not comparable with a run
             * at any other bound. Ignore it and derive both counts from the
             * cell arrays at the threshold this run actually used. */
            (void)verify_apply_region(
                (const uint32_t *)(d_out + (size_t)reg * cap * 4), nr, hlogp, &N,
                logI, log_region, reg, NORM_HORNER, CINIT, tconst,
                NULL, NULL, NULL, NULL, 0, href.data());
            for (uint32_t i = 0; i < ncell; i++) {
                if (href[i]   >= THRESH) surv_cpu++;
                if (d_dbg[i]  >= THRESH) surv_gpu_cells++;
            }
            uint32_t rbad = 0;
            for (uint32_t i = 0; i < ncell; i++)
                if (d_dbg[i] != href[i]) {
                    if (!bad) { firstreg = reg; firstcell = i; }
                    rbad++;
                    long d = (long)d_dbg[i] - (long)href[i];
                    if (labs(d) > labs(worst)) worst = d;
                }
            if (rbad) badregions++;
            bad += rbad;
            cells += ncell;
        }
        printf("apply parity: %llu cells over %u regions;"
               " cells over threshold: gpu %llu, cpu %llu\n",
               (unsigned long long)cells, ncheck,
               (unsigned long long)surv_gpu_cells, (unsigned long long)surv_cpu);
        if (surv_gpu_cells != surv_cpu) {
            printf("  FAIL: the two disagree on how many cells pass the threshold\n");
            fails++;
        }
        if (!surv_gpu_cells)
            printf("  WARNING: no cell in the checked regions passes the"
                   " threshold, so this run does not exercise the survivor"
                   " path at all -- raise --bound\n");
        if (bad) {
            printf("  %llu cells differ (%.5f%%) across %u regions;"
                   " first region %u cell %u; largest difference %ld\n",
                   (unsigned long long)bad, 100.0 * (double)bad / (double)cells,
                   badregions, firstreg, firstcell, worst);
            printf("  This is the PHASE 7 MEASUREMENT, not necessarily a bug: the\n"
                   "  GPU uses portable_log2.h's pl_log2f and this CPU reference\n"
                   "  uses libm's log2f, which Phase 2 measured disagreeing on\n"
                   "  ~1%% of inputs by up to 3 ULP.\n");
            fails++;
        } else {
            printf("  PASS: every one of %llu cells matches the CPU reference\n"
                   "        exactly -- including the soft-fp64 norm fallback and,\n"
                   "        at this geometry, pl_log2f against libm's log2f\n",
                   (unsigned long long)cells);
        }
    }

    mtlShutdown();
    printf("\n%s\n", fails ? "PHASE 5 GATE: FAIL" : "PHASE 5 GATE: PASS");
    return fails ? 1 : 0;
}
