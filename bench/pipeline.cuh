/* ======================= the production pipeline ========================= *
 *
 * Both sides in one process. `bench`'s run_bench remains the measurement
 * harness -- every gate closed so far lives there and every command in
 * RESULTS.md still reproduces -- and this is the path that becomes the siever.
 *
 * The difference that matters is capability, not speed. The two survivor
 * bitmaps stay device-resident, so the intersection, the rank scan and the
 * (x,a,b) emission happen ONCE instead of once per side, and the two sides'
 * cofactors are joined in memory. Nothing here produces a candidate the
 * two-process path could not; what it can do is produce a *stream* of them
 * across many special-q, which is what the cofactor queue needs and what a
 * file round trip between two single-side processes cannot supply.
 *
 * Included at the end of bench_kernels.cu so it can use the static helpers
 * already defined there (run_td_stage, td_build_*).
 */
#ifndef CUDA_SIEVE_PIPELINE_CUH
#define CUDA_SIEVE_PIPELINE_CUH

typedef struct {
    uint32_t *primes, *roots;
    plat_t   *plat;
    uint32_t *survbits;
    uint16_t *slice, *slice_logp;
    uint32_t *sp, *srt, *sg;
    uint16_t *slp;
    uint32_t  nslice_pow2, nsmall, nblk, nwrp;
    norm_t    N;
    uint32_t  CINIT, BOUND, tconst, nsurv;
    /* persistent across special-q */
    uint32_t *hsp, *hsrt, *hsg; uint16_t *hslp;
    uint32_t *d_surv, *d_nsurv, *d_nproj;
    unsigned long long *d_nlost;
} pside_t;

static void pside_free(pside_t *S)
{
    cudaFree(S->primes); cudaFree(S->roots); cudaFree(S->plat);
    cudaFree(S->survbits); cudaFree(S->slice); cudaFree(S->slice_logp);
    cudaFree(S->sp); cudaFree(S->srt); cudaFree(S->sg); cudaFree(S->slp);
    cudaFree(S->d_surv); cudaFree(S->d_nsurv);
    cudaFree(S->d_nproj); cudaFree(S->d_nlost);
    if (S->hsp)  cudaFreeHost(S->hsp);
    if (S->hsrt) cudaFreeHost(S->hsrt);
    if (S->hsg)  cudaFreeHost(S->hsg);
    if (S->hslp) cudaFreeHost(S->hslp);
    memset(S, 0, sizeof(*S));
}

/* Records the fill will produce, for sizing the bucket array. */
static uint64_t pipe_est_records(const fb_t *fb, uint32_t xmax)
{
    double acc = 0;
    for (uint32_t i = 0; i < fb->n; i++) acc += (double)xmax / fb->primes[i];
    return (uint64_t)(acc * 1.15) + 4096;
}

/* Everything that does NOT depend on the special-q: the factor base upload,
 * the slice tables, and the pinned staging. Hoisted out of the per-q path so a
 * band of q measures sieving rather than allocation churn. */
static int pipe_side_init(const fb_t *fb, const fb_t *fbs,
                          const bench_cfg_t *cfg, uint32_t bound,
                          pside_t *S)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t maxsurv = 1u << 22;
    uint16_t *hslice = NULL, *hlogp = NULL;
    int rc = -1;

#define SIDE_INIT_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    memset(S, 0, sizeof(*S));
    S->CINIT = 4096u;
    S->BOUND = bound;
    /* Do not rely on every present and future loader remembering the lattice
     * transform's prime-power precondition. Loaders/builders earn this cookie
     * through fb_validate(); subsets preserve it. Refuse before the first
     * host transform or device upload if that proof is absent. */
    if (!fb_is_transform_validated(fb) ||
        (fbs && fbs->n && !fb_is_transform_validated(fbs))) {
        fprintf(stderr,
                "pipe_side_init: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or uploading it\n");
        goto done;
    }
    SIDE_INIT_CK(cudaMalloc(&S->primes, (size_t)fb->n * 4));
    SIDE_INIT_CK(cudaMalloc(&S->roots,  (size_t)fb->n * 4));
    SIDE_INIT_CK(cudaMalloc(&S->plat,   (size_t)fb->n * sizeof(plat_t)));
    SIDE_INIT_CK(cudaMalloc(&S->survbits, (size_t)nbitword * 4));
    SIDE_INIT_CK(cudaMemcpy(S->primes, fb->primes, (size_t)fb->n * 4,
                            cudaMemcpyHostToDevice));
    SIDE_INIT_CK(cudaMemcpy(S->roots, fb->roots, (size_t)fb->n * 4,
                            cudaMemcpyHostToDevice));

    if (fb_build_slices(fb, &hslice, &hlogp, &S->nslice_pow2) < 0) {
        report_slice_build_error();
        goto done;
    }
    {
        const size_t smem = (size_t)(1u << cfg->log_region) * 2 +
                            (size_t)S->nslice_pow2 * sizeof(*hlogp);
        if (smem > 101376u) {
            fprintf(stderr, "  apply needs %zu B of shared memory for %u"
                    " padded slices; device path supports at most 101376 B\n",
                    smem, S->nslice_pow2);
            goto done;
        }
    }
    SIDE_INIT_CK(cudaMalloc(&S->slice, (size_t)fb->n * 2));
    SIDE_INIT_CK(cudaMalloc(&S->slice_logp, (size_t)S->nslice_pow2 * 2));
    SIDE_INIT_CK(cudaMemcpy(S->slice, hslice, (size_t)fb->n * 2,
                            cudaMemcpyHostToDevice));
    SIDE_INIT_CK(cudaMemcpy(S->slice_logp, hlogp,
                            (size_t)S->nslice_pow2 * 2,
                            cudaMemcpyHostToDevice));
    free(hslice); hslice = NULL;
    free(hlogp); hlogp = NULL;

    /* PINNED staging for the per-q small-sieve tables, allocated ONCE. The
     * 2026-08-04 review measured pageable staging here at 6.2 ms per side for
     * 54 KB; cudaHostAlloc per special-q would be worse than either. */
    if (cfg->small_sieve && fbs && fbs->n) {
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsp, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsrt, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hslp, (size_t)fbs->n * 2,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaHostAlloc((void **)&S->hsg, (size_t)fbs->n * 4,
                                   cudaHostAllocDefault));
        SIDE_INIT_CK(cudaMalloc(&S->sp,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->srt, (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->sg,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(cudaMalloc(&S->slp, (size_t)fbs->n * 2));
    }
    SIDE_INIT_CK(cudaMalloc(&S->d_surv, (size_t)maxsurv * 4));
    SIDE_INIT_CK(cudaMalloc(&S->d_nsurv, 4));
    SIDE_INIT_CK(cudaMalloc(&S->d_nproj, 4));
    SIDE_INIT_CK(cudaMalloc(&S->d_nlost, 8));
    (void)nbitword;
    rc = 0;

done:
    free(hslice);
    free(hlogp);
#undef SIDE_INIT_CK
    return rc;
}

/* The per-q work for one side: transform, small-sieve tables, norm setup,
 * fill, apply. The bucket array is passed in and REUSED between sides -- at
 * 1.38 GB it is the largest allocation in the process and nothing needs it
 * after apply. */
static int pipe_side_perq(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                          const poly_t *POLY, const bench_cfg_t *cfg,
                          int side, double scale,
                          uint8_t *d_bucket, uint32_t *d_cursor, uint32_t cap,
                          uint32_t *d_overflow, int blocks, int fblocks,
                          int fthreads,
                          pside_t *S, float t_stage[3], double *t_host)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t ncell = 1u << log_region;
    const uint32_t maxsurv = 1u << 22;
    const size_t smem = (size_t)ncell * 2 +
                        (size_t)S->nslice_pow2 * sizeof(uint16_t);
    const int athr = cfg->apply_threads ? cfg->apply_threads : 512;
    cudaEvent_t e0 = NULL, e1 = NULL, e2 = NULL, e3 = NULL;
    uint32_t *idx = NULL, *tp = NULL, *trt = NULL, *tg = NULL;
    uint16_t *tlp = NULL;
    float t_t = 0, t_f = 0, t_a = 0;
    double h0 = host_ms();
    int rc = -1;

#define PERQ_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t *hsp = S->hsp, *hsrt = S->hsrt, *hsg = S->hsg;
        uint16_t *hslp = S->hslp;
        uint32_t k = 0;
        for (uint32_t i = 0; i < fbs->n; i++) {
            uint32_t rt, g, m = pl_transform_enc(fbs->primes[i], fbs->roots[i],
                                                 L->a0, L->a1, L->b0, L->b1,
                                                 &rt, &g);
            hsp[k] = m;
            hsrt[k] = rt;
            hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        S->nsmall = k;
        if (k) {
            idx = (uint32_t *)malloc((size_t)k * sizeof(*idx));
            tp  = (uint32_t *)malloc((size_t)k * sizeof(*tp));
            trt = (uint32_t *)malloc((size_t)k * sizeof(*trt));
            tg  = (uint32_t *)malloc((size_t)k * sizeof(*tg));
            tlp = (uint16_t *)malloc((size_t)k * sizeof(*tlp));
            if (!idx || !tp || !trt || !tg || !tlp) {
                fprintf(stderr,
                        "pipe_side_perq: out of memory sorting %u small-sieve entries\n",
                        k);
                goto done;
            }
            for (uint32_t i = 0; i < k; i++) idx[i] = i;
            std::stable_sort(idx, idx + k,
                             [hsp](uint32_t x, uint32_t y) {
                                 return hsp[x] < hsp[y];
                             });
            for (uint32_t i = 0; i < k; i++) {
                const uint32_t z = idx[i];
                tp[i] = hsp[z];
                trt[i] = hsrt[z];
                tg[i] = hsg[z];
                tlp[i] = hslp[z];
            }
            memcpy(hsp, tp, (size_t)k * sizeof(*hsp));
            memcpy(hsrt, trt, (size_t)k * sizeof(*hsrt));
            memcpy(hsg, tg, (size_t)k * sizeof(*hsg));
            memcpy(hslp, tlp, (size_t)k * sizeof(*hslp));
            free(idx); idx = NULL;
            free(tp); tp = NULL;
            free(trt); trt = NULL;
            free(tg); tg = NULL;
            free(tlp); tlp = NULL;
        }
        S->nblk = S->nwrp = 0;
        for (uint32_t i = 0; i < k && hsp[i] < SS_BLOCK_CUT; i++) S->nblk = i + 1;
        for (uint32_t i = 0; i < k && hsp[i] < SS_WARP_CUT; i++) S->nwrp = i + 1;
        PERQ_CK(cudaMemcpy(S->sp, hsp, (size_t)k * sizeof(*hsp),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->srt, hsrt, (size_t)k * sizeof(*hsrt),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->sg, hsg, (size_t)k * sizeof(*hsg),
                           cudaMemcpyHostToDevice));
        PERQ_CK(cudaMemcpy(S->slp, hslp, (size_t)k * sizeof(*hslp),
                           cudaMemcpyHostToDevice));
    }

    /* Side 0's norms are G(x) = Y1*x + Y0, a degree-1 form. run_bench gets
     * that by mutating POLY in the caller; the pipeline must not, because
     * run_td_stage still needs the degree-5 coefficients for side 1. Make the
     * variant here and leave POLY alone. */
    {
        poly_t P = *POLY;
        if (side == 0) {
            P.deg = 1;
            P.c[0] = P.y0;
            P.c[1] = P.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) P.c[z] = 0.0;
        }
        norm_setup(&S->N, &P, L, cfg->logI, cfg->J, scale,
                   side == cfg->sq_side);
        {
            const double bits = norm_exact_bound_bits(&S->N);
            if (!norm_fits_exact(&S->N, BN_LIMBS * 32)) {
                fprintf(stderr,
                        "  exact side-%d degree-%d norm may require %.2f bits;"
                        " the trial-division type holds %d\n",
                        side, P.deg, bits, BN_LIMBS * 32);
                goto done;
            }
        }
    }
    /* CINIT and BOUND were checked once, before any allocation, and are
     * invariant across the special-q band. Keeping them in the persistent
     * side context avoids reparsing or floating-point conversion per q. */
    {
        const float t = norm_target_host(&S->N, 0, cfg->J / 2);
        const int ti = (int)(t + 0.5f);
        S->tconst = (ti < 1) ? 1u :
                    ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
    }

    *t_host = host_ms() - h0;
    PERQ_CK(cudaEventCreate(&e0));
    PERQ_CK(cudaEventCreate(&e1));
    PERQ_CK(cudaEventCreate(&e2));
    PERQ_CK(cudaEventCreate(&e3));
    /* k_transform writes these unconditionally; NULL is an illegal access */
    PERQ_CK(cudaMemset(S->d_nproj, 0, 4));
    PERQ_CK(cudaMemset(S->d_nlost, 0, 8));

    PERQ_CK(cudaEventRecord(e0));
    k_transform<<<blocks, cfg->threads>>>(S->primes, S->roots, S->plat, fb->n,
        cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1,
        S->d_nproj, S->d_nlost);
    PERQ_CK(cudaEventRecord(e1));
    PERQ_CK(cudaMemset(d_cursor, 0, (size_t)nregion * 4));
    PERQ_CK(cudaMemset(d_overflow, 0, 4));
    k_fill_atomic<4><<<fblocks, fthreads>>>(S->plat, S->slice, fb->n, xmax,
        cfg->logI, log_region, d_cursor, d_bucket, cap, d_overflow);
    PERQ_CK(cudaEventRecord(e2));
    PERQ_CK(cudaMemset(S->d_nsurv, 0, 4));
    PERQ_CK(cudaMemset(S->survbits, 0, (size_t)nbitword * 4));
    PERQ_CK(cudaFuncSetAttribute(k_apply<16, 1, NORM_HORNER>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    k_apply<16, 1, NORM_HORNER><<<nregion, athr, smem>>>(
        (const uint32_t *)d_bucket, d_cursor, cap, cfg->logI, log_region,
        S->slice_logp, S->nslice_pow2, S->N, S->CINIT, S->CINIT - S->BOUND,
        S->tconst, NULL, S->d_surv, S->d_nsurv, maxsurv, NULL, 0xFFFFFFFFu,
        S->sp, S->srt, S->sg, S->slp, S->nsmall, S->nblk, S->nwrp,
        0xFFFFFFFFu, NULL, S->survbits, cfg->not_both_even);
    PERQ_CK(cudaEventRecord(e3));
    PERQ_CK(cudaEventSynchronize(e3));
    PERQ_CK(cudaGetLastError());
    t_t = time_kernel(e0, e1);
    t_f = time_kernel(e1, e2);
    t_a = time_kernel(e2, e3);
    PERQ_CK(cudaMemcpy(&S->nsurv, S->d_nsurv, 4, cudaMemcpyDeviceToHost));
    {
        uint32_t hov = 0;
        PERQ_CK(cudaMemcpy(&hov, d_overflow, 4, cudaMemcpyDeviceToHost));
        if (hov) {
            fprintf(stderr, "  side %d: bucket array OVERFLOWED by %u records\n",
                    side, hov);
            goto done;
        }
    }
    if (cfg->verbose_q)
        printf("  side %d: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
               " %8u survivors (bound %u)\n",
               side, t_t, t_f, t_a, t_t + t_f + t_a,
               S->nsurv, S->BOUND);
    t_stage[0] = t_t;
    t_stage[1] = t_f;
    t_stage[2] = t_a;
    rc = 0;

done:
    free(idx);
    free(tp);
    free(trt);
    free(tg);
    free(tlp);
    if (e0 && CUDA_CHECKED(cudaEventDestroy(e0)) && rc == 0) rc = -1;
    if (e1 && CUDA_CHECKED(cudaEventDestroy(e1)) && rc == 0) rc = -1;
    if (e2 && CUDA_CHECKED(cudaEventDestroy(e2)) && rc == 0) rc = -1;
    if (e3 && CUDA_CHECKED(cudaEventDestroy(e3)) && rc == 0) rc = -1;
#undef PERQ_CK
    return rc;
}

/* ======================= the production TD path ========================== *
 *
 * run_td_stage is the MEASUREMENT harness. Calling it per special-q, per side,
 * is what made the pipeline's post-sieve cost 222 ms/q: it runs the rank scan,
 * the emission, the resieve and the classification three times each for a
 * best-of-three, plus two diagnostic variants of k_td that exist only to
 * separate the congruence test from the divisions it triggers, plus a dense
 * recording pass over every survivor -- and it rebuilds the rank scan and the
 * emission for the SECOND side even though both are functions of the shared
 * two-sided bitmap alone.
 *
 * This is the same arithmetic with the harness taken out. Four things change:
 *
 *   1. Rank scan, emission and the resieve summary run ONCE per q, not once
 *      per side. They read the two-sided bitmap; neither side owns them.
 *   2. Every buffer is allocated once for the whole band, not once per q.
 *      Only a survivor count larger than anything seen so far reallocates.
 *   3. The two sides' acceptances are intersected ON DEVICE, and the recording
 *      pass runs over that compacted list. Recording is the only pass that
 *      writes 64 words per thread; over ~240,000 survivors that is a 61 MB
 *      matrix per side, and over the ~1,900 joint candidates it is 486 KB.
 *   4. The reconstruction gate is an explicitly separate phase on the first q,
 *      excluded from the band timing rather than hidden inside it.
 *
 * What does NOT change is the arithmetic: same kernels, same production
 * settings (resieve unroll 4, 1 summary bit per 64 positions). The output of
 * this path is diffed byte-for-byte against the harness path, which is the
 * only reason a restructure this large is safe to make.
 */

#define PIPE_K 16          /* large primes kept per survivor, as run_td_stage */

typedef struct {
    double rank, emit, summary, resieve, td, classify, compact, record;
    double readback, join, hostq, cofac;
} pipe_tm_t;

typedef struct {
    uint32_t nbitword, ngroup, nb, nsumword;
    uint32_t scap, ccap;          /* survivors, joint candidates */
    /* shared by both sides: functions of the two-sided bitmap alone */
    uint32_t *d_cnt, *d_gbase, *d_bsum, *d_sum;
    uint32_t *d_x;  int64_t *d_a, *d_b;
    uint32_t *d_flags;  unsigned long long *d_ovf;
    /* joint acceptance */
    uint32_t *d_aflag, *d_aoff, *d_absum, *d_sel, *d_nacc;
    /* per side */
    uint32_t  *d_plist[2], *d_pcnt[2];
    bn_t      *d_cof[2];
    uint8_t   *d_cofbits[2], *d_status[2];
    tdpoly_t  *d_poly[2];
    uint32_t  *d_stats;          /* {already-relations, candidates, overflows} */
    tdsmall_t *d_sm[2], *h_sm[2];
    uint32_t   nsm[2], nsmcap[2];
    tdpoly_t   hpoly[2];
    /* compacted candidate records, device then pinned host */
    int64_t  *d_ca, *d_cb, *h_ca, *h_cb;
    bn_t     *d_ccof[2], *h_ccof[2];
    uint8_t  *d_cbits[2], *h_cbits[2];
    uint32_t *d_cfac[2], *h_cfac[2], *d_cfn[2], *h_cfn[2];
    cudaEvent_t ev[16];
} pipe_td_t;

static void pipe_td_free(pipe_td_t *C)
{
    cudaFree(C->d_cnt); cudaFree(C->d_gbase); cudaFree(C->d_bsum);
    cudaFree(C->d_sum); cudaFree(C->d_x); cudaFree(C->d_a); cudaFree(C->d_b);
    cudaFree(C->d_flags); cudaFree(C->d_ovf);
    cudaFree(C->d_aflag); cudaFree(C->d_aoff); cudaFree(C->d_absum);
    cudaFree(C->d_sel); cudaFree(C->d_nacc);
    cudaFree(C->d_ca); cudaFree(C->d_cb);
    if (C->h_ca) cudaFreeHost(C->h_ca);
    if (C->h_cb) cudaFreeHost(C->h_cb);
    cudaFree(C->d_stats);
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_plist[s]); cudaFree(C->d_pcnt[s]);
        cudaFree(C->d_cof[s]); cudaFree(C->d_cofbits[s]); cudaFree(C->d_status[s]);
        cudaFree(C->d_poly[s]); cudaFree(C->d_sm[s]);
        cudaFree(C->d_ccof[s]); cudaFree(C->d_cbits[s]);
        cudaFree(C->d_cfac[s]); cudaFree(C->d_cfn[s]);
        if (C->h_sm[s])   cudaFreeHost(C->h_sm[s]);
        if (C->h_ccof[s]) cudaFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) cudaFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s]) cudaFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])  cudaFreeHost(C->h_cfn[s]);
    }
    for (int k = 0; k < 16; k++) if (C->ev[k]) cudaEventDestroy(C->ev[k]);
    memset(C, 0, sizeof(*C));
}

/* Survivor-sized buffers. Grown, never shrunk: a band's first q pays for its
 * own count plus half, and in practice nothing after it reallocates. */
static int pipe_td_grow(pipe_td_t *C, uint32_t n)
{
    uint32_t cap, nab;
    if (n <= C->scap) return 0;
    cap = n + n / 2 + 1024;
    nab = (cap + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    cudaFree(C->d_x); cudaFree(C->d_a); cudaFree(C->d_b);
    cudaFree(C->d_aflag); cudaFree(C->d_aoff); cudaFree(C->d_absum);
    C->d_x = NULL; C->d_a = NULL; C->d_b = NULL;
    C->d_aflag = C->d_aoff = C->d_absum = NULL;
    CK(cudaMalloc(&C->d_x, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_a, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_b, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_aflag, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_aoff, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_absum, (size_t)nab * 4));
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_plist[s]); cudaFree(C->d_pcnt[s]); cudaFree(C->d_cof[s]);
        cudaFree(C->d_cofbits[s]); cudaFree(C->d_status[s]);
        C->d_plist[s] = C->d_pcnt[s] = NULL; C->d_cof[s] = NULL;
        C->d_cofbits[s] = C->d_status[s] = NULL;
        CK(cudaMalloc(&C->d_plist[s], (size_t)cap * PIPE_K * 4));
        CK(cudaMalloc(&C->d_pcnt[s], (size_t)cap * 4));
        CK(cudaMalloc(&C->d_cof[s], (size_t)cap * sizeof(bn_t)));
        CK(cudaMalloc(&C->d_cofbits[s], (size_t)cap));
        CK(cudaMalloc(&C->d_status[s], (size_t)cap));
    }
    C->scap = cap;
    return 0;
}

/* Candidate-sized buffers, device and pinned host. */
static int pipe_td_grow_cand(pipe_td_t *C, uint32_t n)
{
    uint32_t cap;
    if (n <= C->ccap) return 0;
    cap = n + n / 2 + 1024;
    cudaFree(C->d_sel); cudaFree(C->d_ca); cudaFree(C->d_cb);
    C->d_sel = NULL; C->d_ca = C->d_cb = NULL;
    if (C->h_ca) cudaFreeHost(C->h_ca);
    if (C->h_cb) cudaFreeHost(C->h_cb);
    C->h_ca = C->h_cb = NULL;
    CK(cudaMalloc(&C->d_sel, (size_t)cap * 4));
    CK(cudaMalloc(&C->d_ca, (size_t)cap * 8));
    CK(cudaMalloc(&C->d_cb, (size_t)cap * 8));
    CK(cudaHostAlloc((void **)&C->h_ca, (size_t)cap * 8, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&C->h_cb, (size_t)cap * 8, cudaHostAllocDefault));
    for (int s = 0; s < 2; s++) {
        cudaFree(C->d_ccof[s]); cudaFree(C->d_cbits[s]);
        cudaFree(C->d_cfac[s]); cudaFree(C->d_cfn[s]);
        C->d_ccof[s] = NULL; C->d_cbits[s] = NULL;
        C->d_cfac[s] = C->d_cfn[s] = NULL;
        if (C->h_ccof[s])  cudaFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) cudaFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s])  cudaFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])   cudaFreeHost(C->h_cfn[s]);
        C->h_ccof[s] = NULL; C->h_cbits[s] = NULL;
        C->h_cfac[s] = C->h_cfn[s] = NULL;
        CK(cudaMalloc(&C->d_ccof[s], (size_t)cap * sizeof(bn_t)));
        CK(cudaMalloc(&C->d_cbits[s], (size_t)cap));
        CK(cudaMalloc(&C->d_cfac[s], (size_t)cap * TD_FMAX * 4));
        CK(cudaMalloc(&C->d_cfn[s], (size_t)cap * 4));
        CK(cudaHostAlloc((void **)&C->h_ccof[s], (size_t)cap * sizeof(bn_t),
                         cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cbits[s], (size_t)cap, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cfac[s], (size_t)cap * TD_FMAX * 4,
                         cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&C->h_cfn[s], (size_t)cap * 4, cudaHostAllocDefault));
    }
    C->ccap = cap;
    return 0;
}

/* q-independent setup: the exact polynomials, the rank tables sized by the
 * bitmap, and the pinned staging for the per-q small-prime table. */
static int pipe_td_init(pipe_td_t *C, const fb_t *fbs1, const fb_t *fbs0,
                        const poly_t *POLY, uint32_t nbitword)
{
    const fb_t *fbs[2];
    memset(C, 0, sizeof(*C));
    fbs[0] = fbs0; fbs[1] = fbs1;
    if (nbitword % TD_GROUP_W) {
        fprintf(stderr, "  pipeline: %u bitmap words is not a multiple of %d\n",
                nbitword, TD_GROUP_W);
        return -1;
    }
    C->nbitword = nbitword;
    C->ngroup = nbitword / TD_GROUP_W;
    C->nb = (C->ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    C->nsumword = ((nbitword / 2) + 31) / 32;
    for (int k = 0; k < 16; k++) CK(cudaEventCreate(&C->ev[k]));
    CK(cudaMalloc(&C->d_cnt, (size_t)C->ngroup * 4));
    CK(cudaMalloc(&C->d_gbase, (size_t)C->ngroup * 4));
    CK(cudaMalloc(&C->d_bsum, (size_t)C->nb * 4));
    CK(cudaMalloc(&C->d_sum, (size_t)C->nsumword * 4));
    CK(cudaMalloc(&C->d_flags, 4));
    CK(cudaMalloc(&C->d_ovf, 8));
    CK(cudaMalloc(&C->d_nacc, 4));
    CK(cudaMalloc(&C->d_stats, 12));
    CK(cudaMemset(C->d_stats, 0, 12));      /* accumulates over the whole band */
    for (int s = 0; s < 2; s++) {
        if (td_build_poly(&C->hpoly[s], POLY, s)) {
            fprintf(stderr, "  pipeline: could not parse side %d coefficients\n", s);
            return -1;
        }
        CK(cudaMalloc(&C->d_poly[s], sizeof(tdpoly_t)));
        CK(cudaMemcpy(C->d_poly[s], &C->hpoly[s], sizeof(tdpoly_t),
                      cudaMemcpyHostToDevice));
        C->nsmcap[s] = (fbs[s] && fbs[s]->n) ? fbs[s]->n : 0;
        if (C->nsmcap[s]) {
            CK(cudaHostAlloc((void **)&C->h_sm[s],
                             (size_t)C->nsmcap[s] * sizeof(tdsmall_t),
                             cudaHostAllocDefault));
            CK(cudaMalloc(&C->d_sm[s], (size_t)C->nsmcap[s] * sizeof(tdsmall_t)));
        }
    }
    return pipe_td_grow_cand(C, 8192);
}

/* Per-q, per-side host work: retransform the small-prime direct-test table
 * into the pinned buffer and upload it. Everything else about the side is
 * already resident. */
static int pipe_td_small(pipe_td_t *C, int side, const fb_t *fbs,
                         const qlat_t *L, int logI)
{
    if (!C->nsmcap[side]) { C->nsm[side] = 0; return 0; }
    C->nsm[side] = td_fill_small(fbs, L, logI, C->h_sm[side]);
    if (!C->nsm[side]) return -1;
    CK(cudaMemcpy(C->d_sm[side], C->h_sm[side],
                  (size_t)C->nsm[side] * sizeof(tdsmall_t), cudaMemcpyHostToDevice));
    return 0;
}

/* ---- the first-q validation phase -------------------------------------- *
 *
 * Reconstruction: the recorded factors times the residual cofactor must
 * rebuild the exact norm. This runs the DENSE recording pass and checks every
 * candidate on the side, which is what run_td_stage did -- deliberately, so
 * the check keeps its old strength rather than narrowing to the joint set the
 * production path records. It reads back 61 MB per side and rebuilds a 224-bit
 * norm per candidate on the host, so it is a separate phase whose cost is
 * reported on its own and excluded from the band average. */
static int pipe_td_verify(pipe_td_t *C, int side, uint32_t n, int logI,
                          uint32_t sq, const bench_cfg_t *cfg,
                          int blocks, int threads)
{
    uint32_t *d_fac = NULL, *d_faccnt = NULL;
    uint32_t *hfac = NULL, *hfn = NULL;
    uint8_t *hstat = NULL;
    bn_t *hcof = NULL;
    int64_t *ha = NULL, *hb = NULL;
    uint32_t checked = 0, bad = 0, overflowed = 0, maxfac = 0;
    const tdpoly_t *P = &C->hpoly[side];
    int rc = 0;

#define VERIFY_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto out; } } while (0)
    VERIFY_CK(cudaMalloc(&d_fac, (size_t)n * TD_FMAX * 4));
    VERIFY_CK(cudaMalloc(&d_faccnt, (size_t)n * 4));
    k_td<1, 1, 0><<<blocks, threads>>>(C->d_a, C->d_b, C->d_x, NULL, n, logI,
                                       C->d_poly[side], sq,
                                       C->d_plist[side], C->d_pcnt[side], PIPE_K,
                                       C->d_sm[side], C->nsm[side],
                                       C->d_cof[side], C->d_cofbits[side],
                                       C->d_flags, NULL, d_fac, d_faccnt, TD_FMAX);
    if (CUDA_CHECKED(cudaDeviceSynchronize()) ||
        CUDA_CHECKED(cudaGetLastError())) {
        fprintf(stderr, "  verify: recording pass failed\n");
        rc = -1;
        goto out;
    }
    hfac = (uint32_t *)malloc((size_t)n * TD_FMAX * 4);
    hfn = (uint32_t *)malloc((size_t)n * 4);
    hstat = (uint8_t *)malloc((size_t)n);
    hcof = (bn_t *)malloc((size_t)n * sizeof(bn_t));
    ha = (int64_t *)malloc((size_t)n * 8);
    hb = (int64_t *)malloc((size_t)n * 8);
    if (!hfac || !hfn || !hstat || !hcof || !ha || !hb) { rc = -1; goto out; }
    VERIFY_CK(cudaMemcpy(hfac, d_fac, (size_t)n * TD_FMAX * 4,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hfn, d_faccnt, (size_t)n * 4,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hstat, C->d_status[side], (size_t)n,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hcof, C->d_cof[side], (size_t)n * sizeof(bn_t),
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(ha, C->d_a, (size_t)n * 8,
                         cudaMemcpyDeviceToHost));
    VERIFY_CK(cudaMemcpy(hb, C->d_b, (size_t)n * 8,
                         cudaMemcpyDeviceToHost));

    for (uint32_t k = 0; k < n; k++) {
        bns_t acc; bn_t t;
        int64_t a, b; uint64_t ua, ub; int sa, sb;
        if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
        if (hfn[k] > maxfac) maxfac = hfn[k];
        if (hfn[k] > TD_FMAX) { overflowed++; continue; }
        checked++;
        a = ha[k]; b = hb[k];
        ua = (uint64_t)(a < 0 ? -a : a); ub = (uint64_t)(b < 0 ? -b : b);
        sa = (a < 0) ? -1 : 1; sb = (b < 0) ? -1 : 1;
        bns_zero(&acc);
        for (int d = 0; d <= P->deg; d++) {
            int sgn = P->sign[d];
            t = P->c[d];
            if (bn_is_zero(&t)) continue;
            for (int e = 0; e < d; e++) { bn_mul_u64(&t, ua); sgn *= sa; }
            for (int e = 0; e < P->deg - d; e++) { bn_mul_u64(&t, ub); sgn *= sb; }
            bns_addmag(&acc, &t, sgn);
        }
        t = hcof[k];
        for (uint32_t z = 0; z < hfn[k]; z++)
            bn_mul_u64(&t, (uint64_t)hfac[(size_t)k * TD_FMAX + z]);
        if (bn_cmp(&t, &acc.m) != 0) bad++;
    }
    printf("    side %d: factors x cofactor == norm  %u of %u  %s"
           "   (most factors %u of %d)\n",
           side, checked - bad, checked, bad ? "FAIL" : "PASS", maxfac, TD_FMAX);
    if (bad) rc = -1;
    if (overflowed) {
        fprintf(stderr, "    side %d: %u candidates had more than %d factors;"
                " raise TD_FMAX\n", side, overflowed, TD_FMAX);
        rc = -1;
    }
    if (cfg->cofgate &&
        td_gate_cofactors(cfg->cofgate, n, ha, hb, hcof, side) != 0) rc = -1;

out:
    free(hfac); free(hfn); free(hstat); free(hcof); free(ha); free(hb);
    cudaFree(d_fac); cudaFree(d_faccnt);
#undef VERIFY_CK
    return rc;
}

/* ---- one special-q of trial division, both sides ----------------------- */

static int pipe_td_perq(pipe_td_t *C, const fb_t *fb1, const fb_t *fbs1,
                        const fb_t *fb0, const fb_t *fbs0,
                        const qlat_t *L, const bench_cfg_t *cfg,
                        const pside_t *S1, const pside_t *S0,
                        const uint32_t *d_two, uint32_t xmax,
                        int blocks, int threads, int verify,
                        uint32_t *n_out, uint32_t *nacc_out, pipe_tm_t *tm,
                        double *t_verify, int want_host, int accumulate_stats)
{
    const fb_t *fb[2];
    const pside_t *S[2];
    const uint32_t lpb[2] = {cfg->lpb0, cfg->lpb};
    const uint32_t mfb[2] = {cfg->mfb0, cfg->mfb};
    const uint32_t lim[2] = {cfg->lim0, cfg->lim};
    cudaEvent_t *E = C->ev;
    uint32_t n = 0, nacc = 0, nab, hflags = 0;
    unsigned long long hovf = 0;
    double h0;

    fb[0] = fb0; fb[1] = fb1;
    S[0] = S0; S[1] = S1;

    /* ---- phase 1: rank over the two-sided bitmap (shared) ---- */
    CK(cudaEventRecord(E[0]));
    k_group_counts<<<blocks, threads>>>(d_two, C->ngroup, C->d_cnt);
    k_scan_pass1<<<C->nb, TD_SCAN_BLK>>>(C->d_cnt, C->ngroup, C->d_gbase, C->d_bsum);
    k_scan_pass2<<<1, 1024>>>(C->d_bsum, C->nb);
    k_scan_pass3<<<C->nb, TD_SCAN_BLK>>>(C->d_gbase, C->ngroup, C->d_bsum);
    CK(cudaEventRecord(E[1]));
    CK(cudaEventSynchronize(E[1])); CK(cudaGetLastError());
    {
        uint32_t base = 0, cnt = 0;
        CK(cudaMemcpy(&base, C->d_gbase + C->ngroup - 1, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&cnt, C->d_cnt + C->ngroup - 1, 4, cudaMemcpyDeviceToHost));
        n = base + cnt;
    }
    *n_out = n; *nacc_out = 0;
    if (!n) { fprintf(stderr, "  pipeline: no survivors at this q\n"); return -1; }
    if (pipe_td_grow(C, n)) return -1;

    /* the per-q, per-side host tables; the only host work left in the TD path */
    h0 = host_ms();
    if (pipe_td_small(C, 1, fbs1, L, cfg->logI) ||
        pipe_td_small(C, 0, fbs0, L, cfg->logI)) return -1;
    tm->hostq += host_ms() - h0;

    /* ---- phase 2: emit, summary, then both sides ---- */
    CK(cudaMemset(C->d_flags, 0, 4));
    CK(cudaMemset(C->d_ovf, 0, 8));
    CK(cudaEventRecord(E[2]));
    k_emit_ranked<<<blocks, threads>>>(d_two, C->d_gbase, C->nbitword, cfg->logI,
                                       L->a0, L->a1, L->b0, L->b1,
                                       C->d_x, C->d_a, C->d_b, n);
    CK(cudaEventRecord(E[3]));
    /* 1 summary bit per 2 bitmap words == per 64 positions, the setting the
     * resieve sweep selected. */
    k_build_summary_g<<<blocks, threads>>>(d_two, C->nbitword, 2u, C->d_sum);
    CK(cudaEventRecord(E[4]));

    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;          /* side 1 first, as before */
        const int e = si ? 7 : 4;
        CK(cudaMemsetAsync(C->d_pcnt[side], 0, (size_t)n * 4));
        k_resieve_scatter<4><<<blocks, threads>>>(
            S[side]->plat, S[side]->primes, NULL, fb[side]->n, xmax, cfg->logI,
            C->d_sum, d_two, C->d_gbase, C->d_plist[side], C->d_pcnt[side],
            PIPE_K, C->d_ovf, 6);
        CK(cudaEventRecord(E[e + 1]));
        k_td<1, 0, 0><<<blocks, threads>>>(
            C->d_a, C->d_b, C->d_x, NULL, n, cfg->logI, C->d_poly[side],
            side == cfg->sq_side ? (uint32_t)L->q : 0u,
            C->d_plist[side], C->d_pcnt[side], PIPE_K,
            C->d_sm[side], C->nsm[side],
            C->d_cof[side], C->d_cofbits[side], C->d_flags, NULL, NULL, NULL, 0);
        CK(cudaEventRecord(E[e + 2]));
        k_classify<<<blocks, threads>>>(C->d_cof[side], C->d_cofbits[side],
                                        C->d_b, n, lpb[side], mfb[side],
                                        (double)lim[side], C->d_status[side]);
        CK(cudaEventRecord(E[e + 3]));
    }

    /* ---- joint acceptance, compacted on device ---- */
    nab = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    k_accept_flags<<<blocks, threads>>>(C->d_status[0], C->d_status[1], n, C->d_aflag);
    k_scan_pass1<<<nab, TD_SCAN_BLK>>>(C->d_aflag, n, C->d_aoff, C->d_absum);
    k_scan_pass2<<<1, 1024>>>(C->d_absum, nab);
    k_scan_pass3<<<nab, TD_SCAN_BLK>>>(C->d_aoff, n, C->d_absum);
    k_scatter_sel<<<blocks, threads>>>(C->d_aflag, C->d_aoff, n, C->d_sel,
                                       C->ccap, C->d_nacc);
    CK(cudaEventRecord(E[11]));
    CK(cudaEventSynchronize(E[11])); CK(cudaGetLastError());
    tm->emit     += time_kernel(E[2], E[3]);
    tm->summary  += time_kernel(E[3], E[4]);
    tm->resieve  += time_kernel(E[4], E[5]) + time_kernel(E[7], E[8]);
    tm->td       += time_kernel(E[5], E[6]) + time_kernel(E[8], E[9]);
    tm->classify += time_kernel(E[6], E[7]) + time_kernel(E[9], E[10]);
    tm->compact  += time_kernel(E[10], E[11]);
    tm->rank     += time_kernel(E[0], E[1]);

    CK(cudaMemcpy(&hflags, C->d_flags, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hovf, C->d_ovf, 8, cudaMemcpyDeviceToHost));
    if (hflags & TDF_NORM_OVERFLOW) {
        fprintf(stderr, "  ** NORM OVERFLOW: a norm exceeded %d bits\n", BN_LIMBS * 32);
        return -1;
    }
    if (hflags & TDF_LIST_TRUNCATED) {
        fprintf(stderr, "  ** %llu large-prime records past the %u/survivor cap\n",
                hovf, PIPE_K);
        return -1;
    }
    CK(cudaMemcpy(&nacc, C->d_nacc, 4, cudaMemcpyDeviceToHost));
    if (nacc > C->ccap) {           /* the scatter clamped; redo it once, larger */
        if (pipe_td_grow_cand(C, nacc)) return -1;
        k_scatter_sel<<<blocks, threads>>>(C->d_aflag, C->d_aoff, n, C->d_sel,
                                           C->ccap, C->d_nacc);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    }
    *nacc_out = nacc;

    /* ---- the first q's reconstruction gate, on its own clock ---- */
    if (verify) {
        double v0 = host_ms();
        printf("\n  --- first-q validation (excluded from the band timing) ---\n");
        /* The special-q goes to whichever side carries it, matching the k_td
         * calls that produced d_cof[]. Hardcoding side 1 here survived
         * --sq-side 0 in practice -- td_divide_out is a no-op when q does not
         * divide, and the gate still passed 15,174 of 15,174 -- but the verify
         * pass and the pass it is checking must be given the same q on the
         * same side, or the agreement is luck rather than evidence. */
        if (pipe_td_verify(C, 1, n, cfg->logI,
                           cfg->sq_side == 1 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads) ||
            pipe_td_verify(C, 0, n, cfg->logI,
                           cfg->sq_side == 0 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads)) return -1;
        *t_verify = host_ms() - v0;
        printf("  --- validation took %.1f ms ---\n\n", *t_verify);
    }

    /* ---- phase 3: record the joint candidates only ---- */
    if (!nacc) return 0;
    CK(cudaEventRecord(E[12]));
    k_gather_ab<<<blocks, threads>>>(C->d_a, C->d_b, C->d_sel, nacc,
                                     C->d_ca, C->d_cb);
    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;
        k_td<1, 1, 1><<<blocks, threads>>>(
            C->d_a, C->d_b, C->d_x, C->d_sel, nacc, cfg->logI, C->d_poly[side],
            side == cfg->sq_side ? (uint32_t)L->q : 0u,
            C->d_plist[side], C->d_pcnt[side], PIPE_K,
            C->d_sm[side], C->nsm[side],
            C->d_ccof[side], C->d_cbits[side], C->d_flags, NULL,
            C->d_cfac[side], C->d_cfn[side], TD_FMAX);
    }
    CK(cudaEventRecord(E[13]));
    CK(cudaEventSynchronize(E[13])); CK(cudaGetLastError());
    tm->record += time_kernel(E[12], E[13]);

    /* Counts the host loop used to produce, computed where the data already is.
     * They ACCUMULATE across the band and are read at flush points, because a
     * blocking 12-byte readback per q costs about as much as the ~618 bytes per
     * candidate this change removes -- a device round trip per q is the thing
     * the rest of this pipeline is built to avoid. */
    if (accumulate_stats) {
        k_cand_stats<<<blocks, threads>>>(nacc, C->d_cbits[0], C->d_cbits[1],
                                          C->d_cfn[0], C->d_cfn[1],
                                          cfg->lpb0, cfg->lpb, TD_FMAX,
                                          C->d_stats);
        CK(cudaGetLastError());
    }

    h0 = host_ms();
    /* `want_host` is false under inline cofactorisation with no candidate file:
     * the queue has already taken everything it needs straight from device
     * memory, so the ~618 bytes per candidate this used to move existed only to
     * be counted. k_cand_stats counts them in place. */
    if (!want_host) { tm->readback += host_ms() - h0; return 0; }
    CK(cudaMemcpy(C->h_ca, C->d_ca, (size_t)nacc * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(C->h_cb, C->d_cb, (size_t)nacc * 8, cudaMemcpyDeviceToHost));
    for (int s = 0; s < 2; s++) {
        CK(cudaMemcpy(C->h_ccof[s], C->d_ccof[s], (size_t)nacc * sizeof(bn_t),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cbits[s], C->d_cbits[s], (size_t)nacc,
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cfn[s], C->d_cfn[s], (size_t)nacc * 4,
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(C->h_cfac[s], C->d_cfac[s], (size_t)nacc * TD_FMAX * 4,
                      cudaMemcpyDeviceToHost));
    }
    tm->readback += host_ms() - h0;
    return 0;
}

/* ---- the whole per-q pipeline ------------------------------------------ */

extern "C" int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                            const fb_t *fb0, const fb_t *fbs0,
                            const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                            const poly_t *POLY, const bench_cfg_t *cfg)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const uint32_t nregion = xmax >> cfg->log_region;
    const uint32_t nbitword = xmax >> 5;
    const int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    /* Fill's grid is absolute, not per-SM -- see FILL_BLOCKS_DEFAULT. */
    const int fblocks = cfg->fill_blocks ? cfg->fill_blocks : FILL_BLOCKS_DEFAULT;
    const int fthreads = cfg->fill_threads ? cfg->fill_threads : FILL_THREADS_DEFAULT;

    pside_t S1, S0;
    pipe_td_t C;
    pipe_tm_t tm;
    cofq_t Q; cofq_out_t QO;
    uint8_t *d_bucket = NULL;
    double vram_prev = 0;
    size_t need = 0;
    uint32_t *d_cursor = NULL, *d_overflow = NULL, *d_two = NULL;
    uint32_t *d_n = NULL;
    unsigned long long *d_pre = NULL;
    uint64_t est1, est0, est;
    uint32_t cap, nqdone = 0, bound1 = 0, bound0 = 0;
    double acc_isect = 0, acc_host = 0, acc_wall = 0;
    /* The three sieve stages, broken out because the band total alone cannot be
     * compared against the standalone bench (no --pipeline), which reports the
     * same three. When the two disagreed on the A100 only the split showed that
     * the whole discrepancy was transform. "sieve, both sides" is their sum --
     * derived, not accumulated separately, so the printed total and its three
     * children cannot drift apart. */
    double acc_tr = 0, acc_fi = 0, acc_ap = 0;
    double acc_td = 0, t_verify = 0, td0 = 0, jn0 = 0, cf0 = 0, cofac_tail = 0;
    unsigned long long acc_surv = 0, acc_cand = 0, acc_rel = 0;
    FILE *fr = NULL, *fc = NULL;
    char rtmp[2048] = "", ctmp[2048] = "";
    int rc = 0, source_exhausted = 0, count_limit_reached = 0;
    int target_reached = 0;
    qsel_t last_q = {0, 0};
    cudaEvent_t ea = NULL, eb = NULL;

    /* This must precede pipe_est_records(): that helper divides by every
     * modulus, so a zero modulus in an unvalidated object is already too late
     * even though the first device upload happens in pipe_side_init(). */
    if (!fb_is_transform_validated(fb1) ||
        !fb_is_transform_validated(fb0) ||
        (fbs1 && fbs1->n && !fb_is_transform_validated(fbs1)) ||
        (fbs0 && fbs0->n && !fb_is_transform_validated(fbs0))) {
        fprintf(stderr,
                "run_pipeline: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or estimating it\n");
        return -1;
    }
    /* Check both thresholds before pipe_est_records(), CUDA event creation,
     * or the multi-gigabyte bucket allocation. This is also the consumer-side
     * gate for callers that bypass bench_main's CLI validation. */
    if (sieve_bound_checked(cfg->scale, cfg->allowance, 4096u, &bound1,
                            "run_pipeline side 1 survivor parameters") ||
        sieve_bound_checked(cfg->scale0, cfg->allowance0, 4096u, &bound0,
                            "run_pipeline side 0 survivor parameters"))
        return -1;

    memset(&S1, 0, sizeof S1); memset(&S0, 0, sizeof S0);
    memset(&C, 0, sizeof C); memset(&tm, 0, sizeof tm);
    memset(&Q, 0, sizeof Q); memset(&QO, 0, sizeof QO);
#define PIPE_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto done; } } while (0)
    PIPE_CK(cudaEventCreate(&ea));
    PIPE_CK(cudaEventCreate(&eb));

    if (qgen)
        printf("\n=== pipeline: both sides in one process, streaming special-q ===\n");
    else
        printf("\n=== pipeline: both sides in one process, %u special-q ===\n", nq);

    est1 = pipe_est_records(fb1, xmax);
    est0 = pipe_est_records(fb0, xmax);
    est = est1 > est0 ? est1 : est0;
    cap = (uint32_t)(est / nregion) + 256;
    {
        size_t freeB = 0, totalB = 0;
        need = (size_t)nregion * cap * 4;
        PIPE_CK(cudaMemGetInfo(&freeB, &totalB));
        printf("  bucket array %u x %u x 4 B = %.2f GB, shared by both sides"
               " (%.2f GB free)\n", nregion, cap, need / 1073741824.0,
               freeB / 1073741824.0);
        if (need + 512u * 1024 * 1024 > freeB) {
            fprintf(stderr, "  pipeline: bucket array does not fit\n");
            rc = -1;
            goto done;
        }
        PIPE_CK(cudaMalloc(&d_bucket, need));
    }
    PIPE_CK(cudaMalloc(&d_cursor, (size_t)nregion * 4));
    PIPE_CK(cudaMalloc(&d_overflow, 4));

    /* ---- one-time setup, hoisted out of the q loop ---- */
    /* Device memory, by stage. The bucket array dominates on a big-area job
     * like c183 and is a minority of the total on a small-area one like the
     * c151, so "the bucket array is the footprint" is only true at one end.
     * Print the actual split rather than inviting anyone to model it. */
#define VRAM_MARK(label)                                                        \
    do {                                                                        \
        size_t fb_ = 0, tb_ = 0;                                                \
        PIPE_CK(cudaMemGetInfo(&fb_, &tb_));                                    \
        printf("    %-28s %6.2f GB   (%.2f GB free)\n", label,                  \
               (vram_prev - (double)fb_) / 1073741824.0, fb_ / 1073741824.0);   \
        vram_prev = (double)fb_;                                                \
    } while (0)
    {
        size_t fb_ = 0, tb_ = 0;
        PIPE_CK(cudaMemGetInfo(&fb_, &tb_));
        vram_prev = (double)fb_;
        printf("  device memory by stage:\n");
        vram_prev += (double)need;   /* the bucket array is already allocated */
        VRAM_MARK("bucket array");
    }
    if (pipe_side_init(fb1, fbs1, cfg, bound1, &S1) ||
        pipe_side_init(fb0, fbs0, cfg, bound0, &S0)) { rc = -1; goto done; }
    VRAM_MARK("factor bases + bitmaps");
    PIPE_CK(cudaMalloc(&d_two, (size_t)nbitword * 4));
    PIPE_CK(cudaMalloc(&d_n, 4));
    PIPE_CK(cudaMalloc(&d_pre, 8));
    /* Untimed warm-up, n = 0 so it touches nothing. Everything above this line
     * is cudaMalloc/cudaMemcpy, so without it k_transform is the first kernel
     * launched and the one-time CUDA cost -- module load, and on a card with
     * no native cubin the PTX JIT -- lands inside the FIRST q's transform
     * window and is then divided by the band length. At 1340 q that is ~0.15 ms
     * on a 0.954 ms figure; on a 50-q band it is ~4 ms on the same figure.
     * run_bench got this fix (finding 48) and this path, which the RUNBOOK
     * points at as the honest source for transform, did not.
     *
     * Lazy module loading is per-kernel, so this warms k_transform only; fill
     * and apply still pay their own first-launch cost inside q0. They are the
     * reps-stable stages and the cost is one-time either way, but a band short
     * enough to care should be treated as a warm-up run, not a measurement. */
    k_transform<<<blocks, cfg->threads>>>(S1.primes, S1.roots, S1.plat, 0u,
        cfg->logI, cfg->J, 1, 0, 0, 1, S1.d_nproj, S1.d_nlost);
    PIPE_CK(cudaDeviceSynchronize());
    PIPE_CK(cudaGetLastError());
    if (pipe_td_init(&C, fbs1, fbs0, POLY, nbitword)) { rc = -1; goto done; }
    VRAM_MARK("trial division context");
    /* The cross-q cofactor queue. Per-q the candidate count is ~1,956, which
     * would run the rho kernel at 3% occupancy; the queue accumulates across
     * special-q and flushes ~67 q worth at a time. With it resident, only the
     * ~2% of records that become relations are read back, and the 295 MB
     * candidate file and its ~3 ms/q of host emission are gone. */
    if (cfg->cofactor && cofq_init(&Q, &QO, CQ_FLUSH, cfg->cof_ecm,
                                   cfg->ecm_b1, cfg->ecm_b2,
                                   cfg->ecm_curves)) { rc = -1; goto done; }
    if (cfg->cofactor) VRAM_MARK("cofactor queue");
#undef VRAM_MARK

    if (cfg->relations && cfg->candidates &&
        !strcmp(cfg->relations, cfg->candidates)) {
        fprintf(stderr, "  --relations and --candidates must differ\n");
        rc = -1; goto done;
    }
    /* Written through temporaries and renamed only on success: a band that
     * fails at q 20 must not leave files that look like a complete run. */
    if (cfg->relations) {
        snprintf(rtmp, sizeof rtmp, "%s.part", cfg->relations);
        if (!(fr = fopen(rtmp, "w"))) { perror(rtmp); rc = -1; goto done; }
    }
    if (cfg->candidates) {
        snprintf(ctmp, sizeof ctmp, "%s.part", cfg->candidates);
        if (!(fc = fopen(ctmp, "w"))) { perror(ctmp); rc = -1; goto done; }
    }

    /* ---- the band ---- */
    {
    const double t_band = host_ms();
    double t_report = t_band;
    for (uint32_t qi = 0;; qi++) {
        qsel_t generated, checked;
        const qsel_t *cur;
        qlat_t Lq;
        float ts1[3] = {0,0,0}, ts0[3] = {0,0,0}, tis = 0;
        double th1 = 0, th0 = 0, qwall = host_ms(), tv = 0;
        uint32_t hn = 0, nacc = 0, ncand = 0, nrel = 0;
        /* The host loop exists to write files. With inline cofactorisation and
         * no candidate file, nothing reads the host mirrors, so neither the
         * copies nor the loop have to happen.
         *
         * cfg->emit_cof is NOT a third case: --emit-cof is in harness_only and
         * is rejected outright under --pipeline, so it is always NULL here.
         * Including it made this look like it had three outcomes when it has
         * two, and the end-of-band stats fold-in depends on this and
         * accumulate_stats being exact complements. */
        const int want_host = !cfg->cofactor || fc;

        if (qi < nq) {
            cur = &qlist[qi];
        } else if (qgen) {
            const int qr = sqgen_next(qgen, &generated);
            if (qr < 0) {
                fprintf(stderr, "  special-q generator failed after %u q\n",
                        nqdone);
                rc = -1; break;
            }
            if (qr == 0) {
                if (cfg->nq_max && nqdone >= cfg->nq_max)
                    count_limit_reached = 1;
                else
                    source_exhausted = 1;
                break;
            }
            cur = &generated;
        } else {
            if (cfg->nq_max && nqdone >= cfg->nq_max)
                count_limit_reached = 1;
            else
                source_exhausted = 1;
            break;
        }
        /* Validate again at the consumer boundary. bench_main validates every
         * input before scale derivation, but run_pipeline is a public entry
         * point and the streaming generator supplies later q directly here. */
        checked = *cur;
        {
            const qsel_validate_result_t qv =
                qsel_validate(&checked, POLY, cfg->sq_side);
            if (qv != QSEL_VALID) {
                const char *why = qv == QSEL_ERR_Q_RANGE ? "q outside [2,2^32)"
                                : qv == QSEL_ERR_Q_COMPOSITE ? "composite q"
                                : qv == QSEL_ERR_NOT_ROOT ? "rho is not a root"
                                : qv == QSEL_ERR_SIDE ? "invalid special-q side"
                                : "invalid exact polynomial";
                fprintf(stderr, "  q=%llu rho=%llu: special-q rejected: %s\n",
                        (unsigned long long)checked.q,
                        (unsigned long long)checked.rho, why);
                rc = -1; break;
            }
        }
        cur = &checked;
        last_q = *cur;
        qlat_build(&Lq, cur->q, cur->rho, POLY->skew);
        if (cfg->verbose_q)
            printf("\n  q = %llu, rho = %llu\n",
                   (unsigned long long)cur->q,
                   (unsigned long long)cur->rho);

        if (pipe_side_perq(fb1, fbs1, &Lq, POLY, cfg, 1, cfg->scale,
                           d_bucket, d_cursor, cap, d_overflow,
                           blocks, fblocks, fthreads, &S1, ts1, &th1) ||
            pipe_side_perq(fb0, fbs0, &Lq, POLY, cfg, 0, cfg->scale0,
                           d_bucket, d_cursor, cap, d_overflow,
                           blocks, fblocks, fthreads, &S0, ts0, &th0)) { rc = -1; break; }

        PIPE_CK(cudaMemset(d_n, 0, 4));
        PIPE_CK(cudaMemset(d_pre, 0, 8));
        PIPE_CK(cudaMemset(d_two, 0, (size_t)nbitword * 4));
        PIPE_CK(cudaEventRecord(ea));
        /* The compacted (x,a,b) output is not used: k_emit_ranked rebuilds it
         * in RANK order, which the resieve scatter needs and this kernel's
         * atomic order cannot give. Passing cap 0 keeps the survivor count --
         * an independent second count of the same bitmap, cross-checked below
         * against the rank scan -- without the writes. */
        k_intersect_compact<1><<<blocks, cfg->threads>>>(
            S1.survbits, S0.survbits, nbitword, cfg->logI,
            Lq.a0, Lq.a1, Lq.b0, Lq.b1, NULL, NULL, NULL, 0u, d_n, d_pre, d_two);
        PIPE_CK(cudaEventRecord(eb));
        PIPE_CK(cudaEventSynchronize(eb));
        PIPE_CK(cudaGetLastError());
        tis = time_kernel(ea, eb);
        PIPE_CK(cudaMemcpy(&hn, d_n, 4, cudaMemcpyDeviceToHost));

        td0 = host_ms();
        {
            uint32_t n = 0;
            if (pipe_td_perq(&C, fb1, fbs1, fb0, fbs0, &Lq, cfg, &S1, &S0,
                             d_two, xmax, blocks, cfg->threads, qi == 0,
                             &n, &nacc, &tm, &tv, want_host,
                             !want_host)) { rc = -1; break; }
            if (n != hn) {
                fprintf(stderr, "  q=%llu: intersect counted %u survivors,"
                        " rank scan %u\n", (unsigned long long)cur->q, hn, n);
                rc = -1; break;
            }
            t_verify += tv;
            acc_td += host_ms() - td0 - tv;   /* the gate is not the siever */
        }

        cf0 = host_ms();
        /* Inline cofactorisation: the joint candidates stay on device and are
         * appended to the queue instead of being formatted into a file. The
         * relations trial division already completed are still emitted here --
         * they need no splitting and would only take a queue slot. */
        if (cfg->cofactor && nacc) {
            /* Flush BEFORE a q that will not fit rather than splitting it.
             * Splitting means enqueuing part of a q, flushing, and enqueuing
             * the rest -- and the obvious way to write that re-enqueues the
             * whole q after the flush, which counts its first rows twice. A
             * special-q is ~1,956 records against a 131,072 slot queue, so
             * never splitting one costs at most 1.5% of a flush's occupancy. */
            if (Q.n + nacc > Q.cap &&
                cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                           cfg->cof_rounds, cfg->cof_budget, blocks,
                           cfg->threads, fr)) { rc = -1; break; }
            if (nacc > Q.cap) {
                fprintf(stderr, "  q=%llu: %u candidates exceeds the %u-slot"
                        " cofactor queue\n", (unsigned long long)cur->q,
                        nacc, Q.cap);
                rc = -1; break;
            }
            k_cof_enqueue<<<blocks, cfg->threads>>>(
                C.d_ccof[0], C.d_cbits[0], C.d_ccof[1], C.d_cbits[1],
                C.d_ca, C.d_cb, C.d_cfac[0], C.d_cfn[0], C.d_cfac[1], C.d_cfn[1],
                nacc, Q.n, cfg->lpb0, cfg->lpb, Q);
            if (CUDA_CHECKED(cudaDeviceSynchronize()) ||
                CUDA_CHECKED(cudaGetLastError())) { rc = -1; break; }
            Q.n += nacc;
        }
        tm.cofac += host_ms() - cf0;
        jn0 = host_ms();
        /* want_host is constant for the whole run, so it gates the loop rather
         * than being retested per candidate. Hoisting it also makes visible
         * what the inline path skips -- including the TD_FMAX overflow check
         * below, which is why that check is repeated after the band. */
        for (uint32_t k = 0; want_host && k < nacc; k++) {
            int64_t a = C.h_ca[k], b = C.h_cb[k];
            const uint32_t c0 = C.h_cfn[0][k], c1 = C.h_cfn[1][k];
            const uint32_t *f0 = C.h_cfac[0] + (size_t)k * TD_FMAX;
            const uint32_t *f1 = C.h_cfac[1] + (size_t)k * TD_FMAX;
            const int b0 = C.h_cbits[0][k], b1 = C.h_cbits[1][k];
            char buf[80];
            /* A truncated factor list leaves factors undivided and every
             * consumer reads faccnt entries, so it fails the run rather than
             * being emitted. Checked on every q, over exactly the records that
             * get written. */
            if (c0 > TD_FMAX || c1 > TD_FMAX) {
                fprintf(stderr, "  q=%llu: a candidate has %u/%u factors,"
                        " more than the %d recorded; raise TD_FMAX\n",
                        (unsigned long long)cur->q, c0, c1, TD_FMAX);
                rc = -1; break;
            }
            /* Canonical order: the resieve scatters through an atomicAdd on
             * the survivor's slot counter, so the list order varies run to run
             * and sorting is what makes the emitted batch reproducible. */
            std::sort((uint32_t *)f0, (uint32_t *)f0 + c0);
            std::sort((uint32_t *)f1, (uint32_t *)f1 + c1);
            if (b < 0) { a = -a; b = -b; }
            if (b0 <= (int)cfg->lpb0 && b1 <= (int)cfg->lpb) {
                nrel++;
                /* Under --cofactor the queue holds EVERY joint candidate,
                 * including the ones trial division already completed -- they
                 * come back from it with both sides CF_OK and no split primes.
                 * Emitting them here as well would duplicate them, so the
                 * queue is the single emitter. */
                if (fr && !cfg->cofactor) {
                    fprintf(fr, "%lld,%lld:", (long long)a, (long long)b);
                    for (uint32_t z = 0; z < c0; z++)
                        fprintf(fr, "%s%x", z ? "," : "", f0[z]);
                    if (b0 > 1)
                        fprintf(fr, "%s%llx", c0 ? "," : "",
                                (unsigned long long)strtoull(
                                    bn_to_dec(&C.h_ccof[0][k], buf), NULL, 10));
                    fputc(':', fr);
                    for (uint32_t z = 0; z < c1; z++)
                        fprintf(fr, "%s%x", z ? "," : "", f1[z]);
                    if (b1 > 1)
                        fprintf(fr, "%s%llx", c1 ? "," : "",
                                (unsigned long long)strtoull(
                                    bn_to_dec(&C.h_ccof[1][k], buf), NULL, 10));
                    fputc('\n', fr);
                }
                continue;
            }
            ncand++;
            if (cfg->cofactor) continue;   /* the queue has it */
            if (fc) {
                fprintf(fc, "%s%s,", b0 > (int)cfg->lpb0 ? "-" : "",
                        bn_to_dec(&C.h_ccof[0][k], buf));
                fprintf(fc, "%s%s:%lld,%lld:", b1 > (int)cfg->lpb ? "-" : "",
                        bn_to_dec(&C.h_ccof[1][k], buf), (long long)a, (long long)b);
                for (uint32_t z = 0; z < c0; z++)
                    fprintf(fc, "%s%x", z ? "," : "", f0[z]);
                fputc(':', fc);
                for (uint32_t z = 0; z < c1; z++)
                    fprintf(fc, "%s%x", z ? "," : "", f1[z]);
                fputc('\n', fc);
            }
        }
        if (rc) break;

        tm.join += host_ms() - jn0;
        acc_tr += ts1[0] + ts0[0];
        acc_fi += ts1[1] + ts0[1];
        acc_ap += ts1[2] + ts0[2];
        acc_isect += tis; acc_host += th1 + th0;
        acc_wall += host_ms() - qwall - tv;
        acc_surv += hn; acc_cand += ncand; acc_rel += nrel;   /* host path only */
        nqdone++;
        norm_verbose = 0;      /* the first q's setup is printed; the rest are not */
        /* --target-rels: stop once enough relations exist. Under inline
         * cofactorisation the only running total the host has without a per-q
         * readback is Q.nrel, which advances at FLUSH boundaries -- so this
         * overshoots by at most one flush (~256 q here, well under 1% of any
         * useful target). Sieving upward until satisfied is what you actually
         * want when the question is "enough for the matrix", since the yield
         * per q falls as q grows and guessing the range wastes either time or
         * relations. */
        if (cfg->target_rels) {
            unsigned long long have = cfg->cofactor ? Q.nrel
                                                    : (unsigned long long)acc_rel;
            if (have >= cfg->target_rels) {
                printf("\n  --target-rels %llu reached after %u q (%llu relations)\n",
                       (unsigned long long)cfg->target_rels, nqdone, have);
                target_reached = 1;
                break;
            }
        }
        /* ncand and nrel are produced by the host join loop, which does not run
         * under inline cofactorisation -- so printing them there reported a
         * flat "0 joint candidates, 0 relations" for every q of the band. The
         * progress line below was fixed for this; this branch was not, and it
         * is the one somebody passes --verbose-q to read. Report what is
         * actually known per q rather than a variable that stayed at its
         * initialiser; the device counters are folded in after the band. */
        if (cfg->verbose_q) {
            if (want_host)
                printf("  q=%llu: %u survivors, %u joint candidates,"
                       " %u relations\n",
                       (unsigned long long)cur->q, hn, ncand, nrel);
            else
                printf("  q=%llu: %u survivors, %u joint candidates"
                       " (relations counted on the device; see band summary)\n",
                       (unsigned long long)cur->q, hn, nacc);
        }
        /* Under inline cofactorisation the host has no per-q relation count --
         * the counters accumulate on the device and Q.nrel only advances at a
         * flush. Reporting acc_rel there showed a flat 0 for the whole run,
         * which on a multi-hour band looks exactly like failure.
         *
         * Q.nrel is not enough on its own either. It advances at a FLUSH, and a
         * flush is 131,072 candidates -- ~67 q on the c183 this was tuned
         * against, but 686 q on the SNFS job, which enqueues 191 records per q
         * instead of 1,956. That is 41 s of "0 rel" before the counter can
         * move at all, and `--relations` stages to NAME.part until the band
         * ends, so `ls` agrees with it. A run that was working normally, with
         * 63,000 candidates queued that went on to yield 39,710 relations, read
         * as a dead run. So carry the queue occupancy alongside -- it advances
         * every q, on every job -- and label it as candidates, which is what it
         * counts.
         *
         * Progress is reported against whichever goal is actually in force: a
         * relation target if one was given, otherwise an explicit --nq count,
         * generated numeric q interval, or list length. Rate-limited to one
         * update every 30 s: it is a single \r line, but a band runs for hours
         * and nobody needs it faster than that.
         */
        if (!cfg->verbose_q &&
            (host_ms() - t_report > 30000.0 || (!qgen && qi + 1 == nq))) {
            const unsigned long long rels = cfg->cofactor
                ? Q.nrel : (unsigned long long)acc_rel;
            const double el = (host_ms() - t_band) / 1000.0;
            const double rps = el > 0 ? rels / el : 0.0;
            double frac, eta;
            t_report = host_ms();
            if (cfg->target_rels) {
                frac = (double)rels / (double)cfg->target_rels;
                eta  = rps > 0 ? ((double)cfg->target_rels - rels) / rps : 0.0;
            } else if (qgen && cfg->nq_max) {
                frac = (double)(qi + 1) / cfg->nq_max;
                eta  = frac > 0 ? el * (1.0 / frac - 1.0) : 0.0;
            } else if (qgen && cfg->qmax) {
                const double span = (double)(cfg->qmax - cfg->qmin) + 1.0;
                frac = ((double)(cur->q - cfg->qmin) + 1.0) / span;
                eta  = frac > 0 ? el * (1.0 / frac - 1.0) : 0.0;
            } else {
                frac = (double)(qi + 1) / nq;
                eta  = frac > 0 ? el * (1.0 / frac - 1.0) : 0.0;
            }
            if (frac > 1.0) frac = 1.0;
            if (eta < 0.0) eta = 0.0;
            {   /* Q.n counts queued CANDIDATE records, not relations -- only
                 * ~2/3 of them survive splitting on this job, and the band
                 * summary keeps nseen and nrel apart for that reason. Labelled
                 * "cand" so the line cannot be read as pending relations.
                 *
                 * Shown only while the queue holds work the relation count
                 * cannot see yet, so a steady-state line stays as it was --
                 * which means the line CHANGES WIDTH when cofq_flush resets
                 * Q.n to 0. A \r without an erase leaves the tail of the longer
                 * line on screen ("...ETA 0h 12m  cand"), so clear to
                 * end-of-line rather than trusting a trailing pad to cover the
                 * widest case. */
                char queued[32] = "";
                if (cfg->cofactor && Q.n)
                    snprintf(queued, sizeof queued, " +%u cand", Q.n);
                /* rps == 0 means "no relation has flushed yet", not "done".
                 * eta is 0 there as a sentinel, and printing it as 0h 00m says
                 * the band is finishing at the exact moment nothing has been
                 * counted -- which is the reading this line exists to prevent. */
                if (rps > 0.0 && isfinite(eta)) {
                    /* Do not cast the complete ETA to an int: immediately
                     * after the first cofactor flush it can legitimately be
                     * millions of hours. Keep the unbounded hour count as a
                     * double and cast only the modulo-60 minute component. */
                    const double eta_hours = floor(eta / 3600.0);
                    const int eta_minutes =
                        (int)fmod(floor(eta / 60.0), 60.0);
                    printf("    q=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA %.0fh %02dm\033[K\r",
                           (unsigned long long)cur->q, qi + 1, rels, queued,
                           rps, 100.0 * frac, eta_hours, eta_minutes);
                } else if (rps > 0.0) {
                    printf("    q=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA inf\033[K\r",
                           (unsigned long long)cur->q, qi + 1, rels, queued,
                           rps, 100.0 * frac);
                } else {
                    printf("    q=%llu  %u q  %llu rel%s  -- rel/s  %.1f%%"
                           "  ETA --h --m\033[K\r",
                           (unsigned long long)cur->q, qi + 1, rels, queued,
                           100.0 * frac);
                }
            }
        }
        fflush(stdout);
    }

    /* A streamed source discovers its end only when the NEXT pull returns 0,
     * so the in-loop 30-second reporter cannot know that the preceding q was
     * terminal. Give it an explicit terminal line instead of leaving the last
     * visible progress sample up to 30 seconds stale. */
    if (!cfg->verbose_q && qgen && !rc && nqdone &&
        (source_exhausted || count_limit_reached)) {
        const unsigned long long rels = cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel;
        char queued[32] = "";
        if (cfg->cofactor && Q.n)
            snprintf(queued, sizeof queued, " +%u cand", Q.n);
        if (count_limit_reached)
            printf("\n    q=%llu  %u q  %llu rel%s  [--nq %u reached]\033[K\n",
                   (unsigned long long)last_q.q, nqdone, rels, queued,
                   cfg->nq_max);
        else
            printf("\n    q=%llu  %u q  %llu rel%s  [q range exhausted]\033[K\n",
                   (unsigned long long)last_q.q, nqdone, rels, queued);
    }

    {   /* Steady-state footprint. The per-q buffers (survivor lists, prime
         * lists, candidate records) grow on demand as q vary, so the band's
         * high-water mark is not knowable from the init sizes -- it is the
         * larger part of the total on a small-area job. */
        size_t fb_ = 0, tb_ = 0;
        PIPE_CK(cudaMemGetInfo(&fb_, &tb_));
        printf("\n  device memory, steady state: %.2f GB in use of %.2f GB"
               " (%.2f GB free)\n",
               (tb_ - fb_) / 1073741824.0, tb_ / 1073741824.0,
               fb_ / 1073741824.0);
    }
    }   /* t_band scope */

    /* Fold in the device-side candidate counters. Under inline cofactorisation
     * the host loop never ran, so acc_cand and acc_rel are still zero and these
     * are the only counts there are. The overflow check lands here rather than
     * per q: the alternative was a blocking readback every q, which cost more
     * than the host loop it replaced. A truncated factor list still fails the
     * run -- just at the end of the band rather than inside it. */
    if (!rc && cfg->cofactor && !fc) {      /* exactly !want_host */
        uint32_t cs[3] = {0, 0, 0};
        if (CUDA_CHECKED(cudaMemcpy(cs, C.d_stats, 12,
                                    cudaMemcpyDeviceToHost)))
            rc = -1;
        else {
            acc_rel += cs[0]; acc_cand += cs[1];
            if (cs[2]) {
                fprintf(stderr, "  %u candidates in this band have more than the"
                        " %d recorded factors; raise TD_FMAX\n", cs[2], TD_FMAX);
                rc = -1;
            }
        }
    }

    /* The last partial flush happens AFTER the band loop, so its cost is not
     * in acc_wall. On a band shorter than one flush that is the ENTIRE
     * cofactorisation, which is why it is carried separately and added back
     * rather than folded into the per-q average. */
    if (rc == 0 && cfg->cofactor && Q.n) {
        cf0 = host_ms();
        if (cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                       cfg->cof_rounds, cfg->cof_budget, blocks, cfg->threads, fr))
            rc = -1;
        cofac_tail = host_ms() - cf0;
    }
    if (rc == 0 && cfg->target_rels && !target_reached) {
        const unsigned long long have = cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel;
        if (have >= cfg->target_rels) {
            printf("\n  --target-rels %llu reached by the final cofactor flush"
                   " after %u q (%llu relations)\n",
                   (unsigned long long)cfg->target_rels, nqdone, have);
            target_reached = 1;
        } else if (count_limit_reached) {
            fprintf(stderr,
                    "\n  note: --nq %u reached after %u q with %llu relations;"
                    " --target-rels %llu was not reached\n",
                    cfg->nq_max, nqdone, have,
                    (unsigned long long)cfg->target_rels);
        } else if (source_exhausted) {
            fprintf(stderr,
                    "\n  WARNING: special-q source exhausted after %u q with"
                    " %llu relations; --target-rels %llu was NOT reached\n",
                    nqdone, have, (unsigned long long)cfg->target_rels);
        }
    }
    if (fr) { if (ferror(fr)) rc = -1; if (fclose(fr)) rc = -1; fr = NULL; }
    if (fc) { if (ferror(fc)) rc = -1; if (fclose(fc)) rc = -1; fc = NULL; }
    if (rc == 0) {
        if ((cfg->relations && rename(rtmp, cfg->relations)) ||
            (cfg->candidates && rename(ctmp, cfg->candidates))) {
            perror("rename"); rc = -1;
        }
    }
    if (rc) {
        if (cfg->relations) remove(rtmp);
        if (cfg->candidates) remove(ctmp);
        if (qgen)
            fprintf(stderr, "  band FAILED after %u generated q; no output kept\n",
                    nqdone);
        else
            fprintf(stderr, "  band FAILED after %u of %u q; no output kept\n",
                    nqdone, nq);
    }

    if (nqdone) {
        const double N = nqdone;
        const double dev = (tm.rank + tm.emit + tm.summary + tm.resieve + tm.td
                            + tm.classify + tm.compact + tm.record) / N;
        const double acc_sieve = acc_tr + acc_fi + acc_ap;
        printf("\n\n  --- band of %u special-q ---\n", nqdone);
        if (t_verify > 0)
            printf("  (first-q reconstruction gate: %.1f ms, excluded below)\n",
                   t_verify);
        printf("  %-34s %8.2f ms\n", "wall clock per q", acc_wall / N);
        printf("  %-34s %8.2f ms\n", "  sieve, both sides", acc_sieve / N);
        printf("  %-34s %8.3f ms\n", "    transform + plattice", acc_tr / N);
        printf("  %-34s %8.3f ms\n", "    fill", acc_fi / N);
        printf("  %-34s %8.3f ms\n", "    apply", acc_ap / N);
        printf("  %-34s %8.3f ms\n", "  intersect + gcd", acc_isect / N);
        printf("  %-34s %8.3f ms\n", "  host per-q (sieve tables, staging)",
               acc_host / N);
        printf("  %-34s %8.2f ms   <- of which, on the device:\n",
               "  TD + classify, wall", acc_td / N);
        printf("      %-30s %8.3f ms\n", "rank scan (shared)", tm.rank / N);
        printf("      %-30s %8.3f ms\n", "emit (x,a,b), rank order (shared)",
               tm.emit / N);
        printf("      %-30s %8.3f ms\n", "survivor filter (shared)", tm.summary / N);
        printf("      %-30s %8.3f ms\n", "resieve + scatter, both sides",
               tm.resieve / N);
        printf("      %-30s %8.3f ms\n", "norms + trial division, both sides",
               tm.td / N);
        printf("      %-30s %8.3f ms\n", "classify, both sides", tm.classify / N);
        printf("      %-30s %8.3f ms\n", "joint accept + compact", tm.compact / N);
        printf("      %-30s %8.3f ms\n", "record candidate factorisations",
               tm.record / N);
        printf("      %-30s %8.3f ms\n", "= device total", dev);
        printf("      %-30s %8.3f ms\n", "host: small-prime tables", tm.hostq / N);
        printf("      %-30s %8.3f ms\n", "host: readback of candidates",
               tm.readback / N);
        printf("      %-30s %8.3f ms\n", "host: unaccounted",
               acc_td / N - dev - tm.hostq / N - tm.readback / N);
        printf("  %-34s %8.2f ms\n", "  join and emit", tm.join / N);
        if (cfg->cofactor)
            printf("  %-34s %8.2f ms\n", "  cofactorisation, in-loop flushes",
                   tm.cofac / N);
        printf("  %-34s %8.2f ms\n", "  unaccounted",
               (acc_wall - acc_sieve - acc_isect - acc_host - acc_td - tm.join
                - tm.cofac) / N);
        /* The one number that makes host starvation visible. Every stage above
         * is either a cudaEvent (blind to what the CPU is doing) or a host
         * clock, so a contended box prints PERFECT kernel times next to a bad
         * wall clock and nothing says why. Finding 53: saturating this box's 16
         * cores left fill and apply flat within 1% while wall went 24.30 ->
         * 31.27 ms/q. No idle/loaded pair is quoted here on purpose: the first
         * one measured (0.824 / 0.661) used an earlier expression that kept the
         * cofactor queue in the denominator only, and no pair has yet been
         * taken with THIS one on a box confirmed idle. Finding 53 is canonical
         * and says so; a number duplicated into a comment is a number that
         * drifts, which is how the 0.824 and a hand-derived 0.85 ended up in
         * the docs and this file describing the same measurement.
         *
         * THE COFACTOR QUEUE IS OUT OF BOTH SIDES. dev covers only the
         * event-timed TD/classify kernels; the queue's flush is measured with a
         * host clock (tm.cofac) and mixes device and host, so it can go in
         * neither term. Leaving it in the denominator alone made the ratio move
         * with SURVIVOR DENSITY -- a band with more candidates flushes more and
         * reads as a contended host on a perfectly idle box, which is the exact
         * misdiagnosis this line exists to prevent.
         *
         * Even so the numerator is a LOWER BOUND on device time: k_cof_enqueue,
         * k_cand_stats and the flush's own kernels are real GPU work that no
         * event times, so the ratio understates utilisation by a small fixed
         * amount. That is tolerable for the comparison it is for and fatal for
         * the one it is not -- hence:
         *
         * Deliberately NO threshold and no warning text. The healthy value is a
         * function of the card and the job -- a faster GPU spends relatively
         * more of its wall on the same host work, so it sits LOWER when
         * perfectly healthy -- and a hardcoded "good" constant here would be
         * the same mistake as the 144-block default: one box's number promoted
         * to a universal one. Compare it against your own idle baseline on the
         * same card, job AND band length; that comparison is valid, an absolute
         * reading is not.
         *
         * Band length matters because acc_wall excludes the final cofactor
         * flush (see the comment on cofac_tail above). On a band shorter than
         * one flush that tail IS the whole cofactorisation, so a short smoke
         * run and a production band are not comparable to each other. */
        printf("  %-34s %8.3f\n", "GPU-accounted / wall (excl cofac)",
               (acc_sieve + acc_isect + dev * N) / (acc_wall - tm.cofac));
        printf("  %-34s %8.1f\n", "two-sided primitive survivors/q",
               (double)acc_surv / N);
        printf("  %-34s %8.2f\n", "cofactorisation candidates/q",
               (double)acc_cand / N);
        printf("  %-34s %8.3f\n", "COMPLETE RELATIONS/q", (double)acc_rel / N);
        printf("  %-34s %8llu\n", "total relations", (unsigned long long)acc_rel);
        printf("  %-34s %8llu\n", "total candidates", (unsigned long long)acc_cand);
        if (cfg->cofactor) {
            printf("\n  --- cofactorisation, cross-q queue ---\n");
            printf("  %-34s %8.2f ms\n", "rational queue", Q.ms_rat / N);
            printf("  %-34s %8.2f ms\n", "algebraic queue", Q.ms_alg / N);
            printf("  %-34s %8.3f ms\n", "readback + emit of relations", Q.ms_host / N);
            printf("  %-34s %8.2f ms\n", "  = device time per q",
                   (Q.ms_rat + Q.ms_alg + Q.ms_host) / N);
            printf("  %-34s %8.2f ms\n", "wall: in-loop flushes", tm.cofac / N);
            printf("  %-34s %8.2f ms\n", "wall: final flush after the band",
                   cofac_tail / N);
            /* Everything joint-accepted is enqueued, INCLUDING the records
             * trial division already completed -- the queue is the single
             * emitter, so they have to travel with the rest. They cost nothing
             * to split (their status is already CF_OK, and compaction drops
             * them before the first round), but they are counted here, so this
             * is a record count and not a splitting-work count. Labelling it
             * "cofactorised" made it look like a 7-record discrepancy against
             * the candidate total at the parity q. */
            printf("  %-34s %8llu  (of which %llu needed no splitting)\n",
                   "records enqueued", (unsigned long long)Q.nseen,
                   (unsigned long long)acc_rel);
            /* Q.nrel is EVERY relation: the queue is the single emitter, so
             * it already holds the ones trial division completed. */
            printf("  %-34s %8.2f\n", "ALL RELATIONS/q", (double)Q.nrel / N);
            printf("  %-34s %8.2f\n", "  of which, from cofactorisation",
                   (double)(Q.nrel - acc_rel) / N);
            printf("  %-34s %8llu\n", "total relations",
                   (unsigned long long)Q.nrel);
            /* acc_wall ALREADY contains the queue: the flush happens inside
             * the q loop. Adding the device times to it would double count,
             * which the first version of this line did -- it reported 132.31
             * ms/q against a 112 ms/q process wall clock. */
            printf("  %-34s %8.2f ms\n", "wall clock per q, COMPLETE",
                   (acc_wall + cofac_tail) / N);
        }
    }

#undef PIPE_CK
done:
    if (fr) {
        const int io_bad = ferror(fr);
        if (fclose(fr) || io_bad) rc = -1;
        fr = NULL;
    }
    if (fc) {
        const int io_bad = ferror(fc);
        if (fclose(fc) || io_bad) rc = -1;
        fc = NULL;
    }
    /* CUDA failures can jump here from any stage. Remove incomplete outputs
     * here as well as in the normal band-finalisation path so no early error
     * leaves a plausible-looking .part file behind. */
    if (rc) {
        if (rtmp[0]) remove(rtmp);
        if (ctmp[0]) remove(ctmp);
    }
    pipe_td_free(&C);
    cofq_free(&Q, &QO);
    pside_free(&S1); pside_free(&S0);
    cudaFree(d_bucket); cudaFree(d_cursor); cudaFree(d_overflow);
    cudaFree(d_two); cudaFree(d_n); cudaFree(d_pre);
    if (ea) cudaEventDestroy(ea);
    if (eb) cudaEventDestroy(eb);
    return rc;
}

#endif  /* CUDA_SIEVE_PIPELINE_CUH */
