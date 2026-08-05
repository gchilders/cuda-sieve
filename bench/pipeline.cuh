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
 * already defined there (build_slices, run_td_stage, td_build_*).
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

/* Is rho really a root of f mod q? The pipeline subtracts log2(q) from the
 * algebraic norm and divides q out of every one, so a wrong root does not
 * degrade the output, it invalidates it. Cheap to check and it also catches a
 * malformed q-list. */
static int pipe_check_root(const poly_t *P, uint64_t q, uint64_t rho)
{
    uint64_t acc = 0;
    if (!q || q >> 32) return 0;
    for (int k = P->deg; k >= 0; k--) {
        bn_t c; int sg = 1;
        uint32_t cm;
        if (P->cs[k][0]) { if (bn_from_dec(&c, P->cs[k], &sg)) return 0; }
        else bn_zero(&c);
        cm = bn_mod_u32(&c, (uint32_t)q);
        if (sg < 0 && cm) cm = (uint32_t)q - cm;
        acc = (acc * (rho % q) + cm) % q;
    }
    return acc == 0;
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
                          const bench_cfg_t *cfg, pside_t *S)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t maxsurv = 1u << 22;
    uint16_t *hslice = NULL, *hlogp = NULL;

    memset(S, 0, sizeof(*S));
    CK(cudaMalloc(&S->primes, (size_t)fb->n * 4));
    CK(cudaMalloc(&S->roots,  (size_t)fb->n * 4));
    CK(cudaMalloc(&S->plat,   (size_t)fb->n * sizeof(plat_t)));
    CK(cudaMalloc(&S->survbits, (size_t)nbitword * 4));
    CK(cudaMemcpy(S->primes, fb->primes, (size_t)fb->n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S->roots,  fb->roots,  (size_t)fb->n * 4, cudaMemcpyHostToDevice));

    hslice = (uint16_t *)malloc((size_t)fb->n * 2);
    if (!hslice) return -1;
    build_slices(fb, hslice, &hlogp, &S->nslice_pow2);
    CK(cudaMalloc(&S->slice, (size_t)fb->n * 2));
    CK(cudaMalloc(&S->slice_logp, (size_t)S->nslice_pow2 * 2));
    CK(cudaMemcpy(S->slice, hslice, (size_t)fb->n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S->slice_logp, hlogp, (size_t)S->nslice_pow2 * 2,
                  cudaMemcpyHostToDevice));
    free(hslice); free(hlogp);

    /* PINNED staging for the per-q small-sieve tables, allocated ONCE. The
     * 2026-08-04 review measured pageable staging here at 6.2 ms per side for
     * 54 KB; cudaHostAlloc per special-q would be worse than either. */
    if (cfg->small_sieve && fbs && fbs->n) {
        CK(cudaHostAlloc((void **)&S->hsp,  (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&S->hsrt, (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&S->hslp, (size_t)fbs->n * 2, cudaHostAllocDefault));
        CK(cudaHostAlloc((void **)&S->hsg,  (size_t)fbs->n * 4, cudaHostAllocDefault));
        CK(cudaMalloc(&S->sp,  (size_t)fbs->n * 4));
        CK(cudaMalloc(&S->srt, (size_t)fbs->n * 4));
        CK(cudaMalloc(&S->sg,  (size_t)fbs->n * 4));
        CK(cudaMalloc(&S->slp, (size_t)fbs->n * 2));
    }
    CK(cudaMalloc(&S->d_surv, (size_t)maxsurv * 4));
    CK(cudaMalloc(&S->d_nsurv, 4));
    CK(cudaMalloc(&S->d_nproj, 4));
    CK(cudaMalloc(&S->d_nlost, 8));
    (void)nbitword;
    return 0;
}

/* The per-q work for one side: transform, small-sieve tables, norm setup,
 * fill, apply. The bucket array is passed in and REUSED between sides -- at
 * 1.38 GB it is the largest allocation in the process and nothing needs it
 * after apply. */
static int pipe_side_perq(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                          const poly_t *POLY, const bench_cfg_t *cfg,
                          int side, double scale, double allowance,
                          uint8_t *d_bucket, uint32_t *d_cursor, uint32_t cap,
                          uint32_t *d_overflow, int blocks, pside_t *S,
                          float *t_side, double *t_host)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t ncell = 1u << log_region;
    const uint32_t maxsurv = 1u << 22;
    const size_t smem = (size_t)ncell * 2 + 128;
    const int athr = cfg->apply_threads ? cfg->apply_threads : 512;
    cudaEvent_t e0, e1, e2, e3;
    float t_t, t_f, t_a;
    double h0 = host_ms();

    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t *hsp = S->hsp, *hsrt = S->hsrt, *hsg = S->hsg;
        uint16_t *hslp = S->hslp;
        uint32_t k = 0;
        for (uint32_t i = 0; i < fbs->n; i++) {
            uint32_t rt, g, m = pl_transform_enc(fbs->primes[i], fbs->roots[i],
                                                 L->a0, L->a1, L->b0, L->b1, &rt, &g);
            hsp[k] = m; hsrt[k] = rt; hsg[k] = g; hslp[k] = fbs->logp[i]; k++;
        }
        S->nsmall = k;
        {
            uint32_t *idx = (uint32_t *)malloc((size_t)k * 4);
            uint32_t *tp  = (uint32_t *)malloc((size_t)k * 4);
            uint32_t *trt = (uint32_t *)malloc((size_t)k * 4);
            uint32_t *tg  = (uint32_t *)malloc((size_t)k * 4);
            uint16_t *tlp = (uint16_t *)malloc((size_t)k * 2);
            for (uint32_t i = 0; i < k; i++) idx[i] = i;
            std::stable_sort(idx, idx + k,
                             [hsp](uint32_t x, uint32_t y) { return hsp[x] < hsp[y]; });
            for (uint32_t i = 0; i < k; i++) {
                uint32_t z = idx[i];
                tp[i] = hsp[z]; trt[i] = hsrt[z]; tg[i] = hsg[z]; tlp[i] = hslp[z];
            }
            memcpy(hsp, tp, (size_t)k * 4); memcpy(hsrt, trt, (size_t)k * 4);
            memcpy(hsg, tg, (size_t)k * 4); memcpy(hslp, tlp, (size_t)k * 2);
            free(idx); free(tp); free(trt); free(tg); free(tlp);
        }
        S->nblk = S->nwrp = 0;
        for (uint32_t i = 0; i < k && hsp[i] < SS_BLOCK_CUT; i++) S->nblk = i + 1;
        for (uint32_t i = 0; i < k && hsp[i] < SS_WARP_CUT;  i++) S->nwrp = i + 1;
        CK(cudaMemcpy(S->sp,  hsp,  (size_t)k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S->srt, hsrt, (size_t)k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S->sg,  hsg,  (size_t)k * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S->slp, hslp, (size_t)k * 2, cudaMemcpyHostToDevice));
    }

    /* Side 0's norms are G(x) = Y1*x + Y0, a degree-1 form. run_bench gets
     * that by mutating POLY in the caller; the pipeline must not, because
     * run_td_stage still needs the degree-5 coefficients for side 1. Make the
     * variant here and leave POLY alone. */
    {
        poly_t P = *POLY;
        if (side == 0) {
            P.deg = 1; P.c[0] = P.y0; P.c[1] = P.y1;
            for (int z = 2; z < 8; z++) P.c[z] = 0.0;
        }
        norm_setup(&S->N, &P, L, cfg->logI, cfg->J, scale, side == 1);
    }
    S->CINIT = 4096u;
    S->BOUND = (uint32_t)(scale * allowance + 1.0);
    {
        float t = norm_target_host(&S->N, 0, cfg->J / 2);
        int ti = (int)(t + 0.5f);
        S->tconst = (ti < 1) ? 1u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
    }

    *t_host = host_ms() - h0;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    /* k_transform writes these unconditionally; NULL is an illegal access */
    CK(cudaMemset(S->d_nproj, 0, 4)); CK(cudaMemset(S->d_nlost, 0, 8));

    cudaEventRecord(e0);
    k_transform<<<blocks, cfg->threads>>>(S->primes, S->roots, S->plat, fb->n,
        cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, S->d_nproj, S->d_nlost);
    cudaEventRecord(e1);
    CK(cudaMemset(d_cursor, 0, (size_t)nregion * 4));
    CK(cudaMemset(d_overflow, 0, 4));
    k_fill_atomic<4><<<blocks, cfg->threads>>>(S->plat, S->slice, fb->n, xmax,
        cfg->logI, log_region, d_cursor, d_bucket, cap, d_overflow);
    cudaEventRecord(e2);
    CK(cudaMemset(S->d_nsurv, 0, 4));
    CK(cudaMemset(S->survbits, 0, (size_t)nbitword * 4));
    CK(cudaFuncSetAttribute(k_apply<16, 1, NORM_HORNER>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    k_apply<16, 1, NORM_HORNER><<<nregion, athr, smem>>>(
        (const uint32_t *)d_bucket, d_cursor, cap, cfg->logI, log_region,
        S->slice_logp, S->nslice_pow2, S->N, S->CINIT, S->CINIT - S->BOUND,
        S->tconst, NULL, S->d_surv, S->d_nsurv, maxsurv, NULL, 0xFFFFFFFFu,
        S->sp, S->srt, S->sg, S->slp, S->nsmall, S->nblk, S->nwrp,
        0xFFFFFFFFu, NULL, S->survbits, cfg->not_both_even);
    cudaEventRecord(e3);
    CK(cudaEventSynchronize(e3)); CK(cudaGetLastError());
    t_t = time_kernel(e0, e1); t_f = time_kernel(e1, e2); t_a = time_kernel(e2, e3);
    CK(cudaMemcpy(&S->nsurv, S->d_nsurv, 4, cudaMemcpyDeviceToHost));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    {
        uint32_t hov = 0;
        CK(cudaMemcpy(&hov, d_overflow, 4, cudaMemcpyDeviceToHost));
        if (hov) {
            fprintf(stderr, "  side %d: bucket array OVERFLOWED by %u records\n",
                    side, hov);
            return -1;
        }
    }
    if (cfg->verbose_q)
        printf("  side %d: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
               " %8u survivors (bound %u)\n",
               side, t_t, t_f, t_a, t_t + t_f + t_a, S->nsurv, S->BOUND);
    *t_side = t_t + t_f + t_a;
    return 0;
}

/* ---- the whole per-q pipeline ------------------------------------------ */

extern "C" int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                            const fb_t *fb0, const fb_t *fbs0,
                            const qsel_t *qlist, uint32_t nq,
                            const poly_t *POLY, const bench_cfg_t *cfg)
{
    const uint32_t xmax = (1u << cfg->logI) * cfg->J;
    const uint32_t nregion = xmax >> cfg->log_region;
    const uint32_t nbitword = xmax >> 5;
    const int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    const uint32_t icap = 1u << 22;

    pside_t S1, S0;
    td_out_t o1, o0;
    uint8_t *d_bucket = NULL;
    uint32_t *d_cursor = NULL, *d_overflow = NULL, *d_two = NULL;
    uint32_t *d_x = NULL, *d_n = NULL;
    int64_t *d_a = NULL, *d_b = NULL;
    unsigned long long *d_pre = NULL;
    uint64_t est1, est0, est;
    uint32_t cap, nqdone = 0;
    double acc_sieve = 0, acc_isect = 0, acc_host = 0, acc_wall = 0;
    double acc_td = 0, acc_join = 0, td0 = 0, jn0 = 0;
    unsigned long long acc_surv = 0, acc_cand = 0, acc_rel = 0;
    FILE *fr = NULL, *fc = NULL;
    char rtmp[2048] = "", ctmp[2048] = "";
    int rc = 0;
    cudaEvent_t ea, eb;

    memset(&S1, 0, sizeof S1); memset(&S0, 0, sizeof S0);
    memset(&o1, 0, sizeof o1); memset(&o0, 0, sizeof o0);
    cudaEventCreate(&ea); cudaEventCreate(&eb);

    printf("\n=== pipeline: both sides in one process, %u special-q ===\n", nq);

    est1 = pipe_est_records(fb1, xmax);
    est0 = pipe_est_records(fb0, xmax);
    est = est1 > est0 ? est1 : est0;
    cap = (uint32_t)(est / nregion) + 256;
    {
        size_t need = (size_t)nregion * cap * 4, freeB = 0, totalB = 0;
        CK(cudaMemGetInfo(&freeB, &totalB));
        printf("  bucket array %u x %u x 4 B = %.2f GB, shared by both sides"
               " (%.2f GB free)\n", nregion, cap, need / 1073741824.0,
               freeB / 1073741824.0);
        if (need + 512u * 1024 * 1024 > freeB) {
            fprintf(stderr, "  pipeline: bucket array does not fit\n");
            return -1;
        }
        CK(cudaMalloc(&d_bucket, need));
    }
    CK(cudaMalloc(&d_cursor, (size_t)nregion * 4));
    CK(cudaMalloc(&d_overflow, 4));

    /* ---- one-time setup, hoisted out of the q loop ---- */
    if (pipe_side_init(fb1, fbs1, cfg, &S1) ||
        pipe_side_init(fb0, fbs0, cfg, &S0)) { rc = -1; goto done; }
    CK(cudaMalloc(&d_two, (size_t)nbitword * 4));
    CK(cudaMalloc(&d_x, (size_t)icap * 4));
    CK(cudaMalloc(&d_a, (size_t)icap * 8));
    CK(cudaMalloc(&d_b, (size_t)icap * 8));
    CK(cudaMalloc(&d_n, 4)); CK(cudaMalloc(&d_pre, 8));

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
    for (uint32_t qi = 0; qi < nq; qi++) {
        qlat_t Lq;
        float ts1 = 0, ts0 = 0, tis = 0;
        double th1 = 0, th0 = 0, qwall = host_ms();
        uint32_t hn = 0, ncand = 0, nrel = 0;

        if (!pipe_check_root(POLY, qlist[qi].q, qlist[qi].rho)) {
            fprintf(stderr, "  q=%llu: rho=%llu is not a root of f mod q\n",
                    (unsigned long long)qlist[qi].q,
                    (unsigned long long)qlist[qi].rho);
            rc = -1; break;
        }
        qlat_build(&Lq, qlist[qi].q, qlist[qi].rho, POLY->skew);
        if (cfg->verbose_q)
            printf("\n  q = %llu, rho = %llu\n",
                   (unsigned long long)qlist[qi].q,
                   (unsigned long long)qlist[qi].rho);

        if (pipe_side_perq(fb1, fbs1, &Lq, POLY, cfg, 1, cfg->scale,
                           cfg->allowance, d_bucket, d_cursor, cap, d_overflow,
                           blocks, &S1, &ts1, &th1) ||
            pipe_side_perq(fb0, fbs0, &Lq, POLY, cfg, 0, cfg->scale0,
                           cfg->allowance0, d_bucket, d_cursor, cap, d_overflow,
                           blocks, &S0, &ts0, &th0)) { rc = -1; break; }

        CK(cudaMemset(d_n, 0, 4)); CK(cudaMemset(d_pre, 0, 8));
        CK(cudaMemset(d_two, 0, (size_t)nbitword * 4));
        cudaEventRecord(ea);
        k_intersect_compact<1><<<blocks, cfg->threads>>>(
            S1.survbits, S0.survbits, nbitword, cfg->logI,
            Lq.a0, Lq.a1, Lq.b0, Lq.b1, d_x, d_a, d_b, icap, d_n, d_pre, d_two);
        cudaEventRecord(eb);
        CK(cudaEventSynchronize(eb)); CK(cudaGetLastError());
        tis = time_kernel(ea, eb);
        CK(cudaMemcpy(&hn, d_n, 4, cudaMemcpyDeviceToHost));
        if (hn > icap) {
            fprintf(stderr, "  q=%llu: survivor list overflowed %u\n",
                    (unsigned long long)qlist[qi].q, icap);
            rc = -1; break;
        }

        td0 = host_ms();
        {
            bench_cfg_t c = *cfg;
            c.resieve_sweep = 0;          /* the sweep is a one-off experiment */
            /* The reconstruction gate rebuilds a 224-bit norm per candidate on
             * the host. It is worth running -- it caught a real bug -- but at
             * ~47K norms per q it would dominate a band measurement, so run it
             * on the first special-q and trust it thereafter. */
            c.td_verify = (qi == 0);
            c.side = 1; c.emit_cof = NULL;
            c.lpb = cfg->lpb; c.mfb = cfg->mfb; c.lim = cfg->lim;
            if (run_td_stage(fb1, fbs1, &Lq, POLY, &c, S1.plat, S1.primes, d_two,
                             nbitword, xmax, blocks, cfg->threads, &o1)) { rc = -1; break; }
            c.side = 0; c.scale = cfg->scale0; c.allowance = cfg->allowance0;
            c.lpb = cfg->lpb0; c.mfb = cfg->mfb0; c.lim = cfg->lim0;
            if (run_td_stage(fb0, fbs0, &Lq, POLY, &c, S0.plat, S0.primes, d_two,
                             nbitword, xmax, blocks, cfg->threads, &o0)) { rc = -1; break; }
        }
        acc_td += host_ms() - td0;
        jn0 = host_ms();
        if (o1.n != o0.n || !o1.cof || !o0.cof) {
            fprintf(stderr, "  q=%llu: sides produced %u and %u survivors\n",
                    (unsigned long long)qlist[qi].q, o1.n, o0.n);
            rc = -1; break;
        }

        for (uint32_t k = 0; k < o1.n; k++) {
            int st1 = o1.status[k], st0 = o0.status[k];
            int64_t a, b;
            char buf[80];
            if ((st1 != COF_ACCEPT && st1 != COF_SPLIT) ||
                (st0 != COF_ACCEPT && st0 != COF_SPLIT)) continue;
            a = o1.a[k]; b = o1.b[k];
            if (b < 0) { a = -a; b = -b; }
            if (o0.bits[k] <= (int)cfg->lpb0 && o1.bits[k] <= (int)cfg->lpb) {
                nrel++;
                if (fr) {
                    fprintf(fr, "%lld,%lld:", (long long)a, (long long)b);
                    for (uint32_t z = 0; o0.faccnt && z < o0.faccnt[k] && z < TD_FMAX; z++)
                        fprintf(fr, "%s%x", z ? "," : "", o0.fac[(size_t)k * TD_FMAX + z]);
                    if (o0.bits[k] > 1)
                        fprintf(fr, "%s%llx", (o0.faccnt && o0.faccnt[k]) ? "," : "",
                                (unsigned long long)strtoull(bn_to_dec(&o0.cof[k], buf), NULL, 10));
                    fputc(':', fr);
                    for (uint32_t z = 0; o1.faccnt && z < o1.faccnt[k] && z < TD_FMAX; z++)
                        fprintf(fr, "%s%x", z ? "," : "", o1.fac[(size_t)k * TD_FMAX + z]);
                    if (o1.bits[k] > 1)
                        fprintf(fr, "%s%llx", (o1.faccnt && o1.faccnt[k]) ? "," : "",
                                (unsigned long long)strtoull(bn_to_dec(&o1.cof[k], buf), NULL, 10));
                    fputc('\n', fr);
                }
                continue;
            }
            ncand++;
            if (fc) {
                fprintf(fc, "%s%s,", o0.bits[k] > (int)cfg->lpb0 ? "-" : "",
                        bn_to_dec(&o0.cof[k], buf));
                fprintf(fc, "%s%s:%lld,%lld:", o1.bits[k] > (int)cfg->lpb ? "-" : "",
                        bn_to_dec(&o1.cof[k], buf), (long long)a, (long long)b);
                for (uint32_t z = 0; o0.faccnt && z < o0.faccnt[k] && z < TD_FMAX; z++)
                    fprintf(fc, "%s%x", z ? "," : "", o0.fac[(size_t)k * TD_FMAX + z]);
                fputc(':', fc);
                for (uint32_t z = 0; o1.faccnt && z < o1.faccnt[k] && z < TD_FMAX; z++)
                    fprintf(fc, "%s%x", z ? "," : "", o1.fac[(size_t)k * TD_FMAX + z]);
                fputc('\n', fc);
            }
        }

        acc_join += host_ms() - jn0;
        acc_sieve += ts1 + ts0; acc_isect += tis; acc_host += th1 + th0;
        acc_wall += host_ms() - qwall;
        {   /* read the count BEFORE freeing the arrays it lives in */
            uint32_t nsurv_q = o1.n;
            acc_surv += nsurv_q; acc_cand += ncand; acc_rel += nrel;
            nqdone++;
            td_out_free(&o1); td_out_free(&o0);
            if (cfg->verbose_q)
                printf("  q=%llu: %u survivors, %u candidates, %u relations\n",
                       (unsigned long long)qlist[qi].q, nsurv_q, ncand, nrel);
        }
        if (!cfg->verbose_q && ((qi % 10) == 9 || qi + 1 == nq))
            printf("    %u/%u q done, %llu relations, %llu candidates\r",
                   qi + 1, nq, (unsigned long long)acc_rel,
                   (unsigned long long)acc_cand);
        fflush(stdout);
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
        fprintf(stderr, "  band FAILED after %u of %u q; no output kept\n",
                nqdone, nq);
    }

    if (nqdone) {
        printf("\n\n  --- band of %u special-q ---\n", nqdone);
        printf("  %-34s %8.2f ms\n", "wall clock per q", acc_wall / nqdone);
        printf("  %-34s %8.2f ms\n", "  sieve, both sides", acc_sieve / nqdone);
        printf("  %-34s %8.3f ms\n", "  intersect + gcd + compact", acc_isect / nqdone);
        printf("  %-34s %8.3f ms\n", "  host per-q (tables, staging)", acc_host / nqdone);
        printf("  %-34s %8.2f ms   <- device kernels + per-q allocation and\n"
               "%-36s               readback, both sides\n",
               "  TD + classify, wall", acc_td / nqdone, "");
        printf("  %-34s %8.2f ms\n", "  join and emit", acc_join / nqdone);
        printf("  %-34s %8.2f ms\n", "  unaccounted",
               (acc_wall - acc_sieve - acc_isect - acc_host - acc_td - acc_join) / nqdone);
        printf("  %-34s %8.1f\n", "two-sided primitive survivors/q",
               (double)acc_surv / nqdone);
        printf("  %-34s %8.2f\n", "cofactorisation candidates/q",
               (double)acc_cand / nqdone);
        printf("  %-34s %8.3f\n", "COMPLETE RELATIONS/q", (double)acc_rel / nqdone);
        printf("  %-34s %8llu\n", "total relations", (unsigned long long)acc_rel);
        printf("  %-34s %8llu\n", "total candidates", (unsigned long long)acc_cand);
    }

done:
    if (fr) fclose(fr);
    if (fc) fclose(fc);
    td_out_free(&o1); td_out_free(&o0);
    pside_free(&S1); pside_free(&S0);
    cudaFree(d_bucket); cudaFree(d_cursor); cudaFree(d_overflow);
    cudaFree(d_two); cudaFree(d_x); cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_n); cudaFree(d_pre);
    cudaEventDestroy(ea); cudaEventDestroy(eb);
    return rc;
}

#endif  /* CUDA_SIEVE_PIPELINE_CUH */
