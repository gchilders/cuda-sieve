/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 6 intermediate gate: the cofactoriser on Metal, driven through the
 * tree's own standalone entry point.
 *
 * run_cofac() (cofac.cuh:2728) reads a batch of cofactor candidates and splits
 * them on the GPU, writing relations. oracle/c183.q120000053.cofac_candidates
 * .txt is the recorded CADO run's own candidate list for the parity special-q,
 * so this exercises the whole cofactor path -- Montgomery arithmetic at 3 and
 * 4 limbs, Pollard-Brent rho, ECM with and without stage 2, the status
 * machine and the compaction rounds -- without needing the sieve pipeline.
 *
 * Exactly as fbgpucheck.sh did for Phase 4, this reaches a real gate long
 * before the full ./bench binary exists.
 *
 * Job parameters are c183's, from oracle/input.job:
 *   rational  lim 67,100,000  lpb 31  mfb 60
 *   algebraic lim 134,200,000 lpb 32  mfb 92
 */
#include "bench.h"
#include "metal/metal_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* The oracle file is a candidate list -- `a b cof0 cof1` per line -- while
 * run_cofac parses mkcofbatch's batch format, `res1,res2:a,b:rfac:afac`.
 * Convert, following the sign convention mkcofbatch.c documents: a residue is
 * written NEGATIVE exactly when its bit length exceeds that side's lpb, which
 * is what tells the cofactoriser "this one still needs splitting".
 *
 * The factor lists are left empty: these candidates carry no trial-division
 * factors, so every prime below lim has to come out of the split. That makes
 * the test harder than the pipeline's own path, not easier. */
static int bits_of_dec(const char *d)
{
    /* decimal digits -> bit length, without bignum arithmetic: log2(10) */
    size_t n = strlen(d);
    while (*d == '0' && n > 1) { d++; n--; }
    return (int)((double)n * 3.321928094887362) ;
}

static int make_batch(const char *in, const char *tmp, uint32_t lpb0, uint32_t lpb1,
                      long *nlines)
{
    FILE *fi = fopen(in, "r"), *fo = fopen(tmp, "w");
    if (!fi || !fo) { fprintf(stderr, "cannot open %s or %s\n", in, tmp); return -1; }
    char line[8192]; *nlines = 0;
    while (fgets(line, sizeof line, fi)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char a[128], b[128], c0[512], c1[512];
        if (sscanf(line, "%127s %127s %511s %511s", a, b, c0, c1) != 4) continue;
        fprintf(fo, "%s%s,%s%s:%s,%s::\n",
                bits_of_dec(c0) > (int)lpb0 ? "-" : "", c0,
                bits_of_dec(c1) > (int)lpb1 ? "-" : "", c1, a, b);
        (*nlines)++;
    }
    fclose(fi); fclose(fo);
    return 0;
}

int main(int argc, char **argv)
{
    const char *in  = "../oracle/c183.q120000053.cofac_candidates.txt";
    const char *out = "/tmp/cofac_metal_relations.txt";
    int meth = COF_METHOD_RHO, rounds = 2, want = -1;
    uint32_t budget = 65536, ecm_b1 = 2000, ecm_b2 = 60000, ecm_curves = 48;
    int limbs = 3;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in")  && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--ecm")) meth = COF_METHOD_ECM;
        else if (!strcmp(argv[i], "--rho")) meth = COF_METHOD_RHO;
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) budget = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--b1") && i + 1 < argc) ecm_b1 = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--b2") && i + 1 < argc) ecm_b2 = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--curves") && i + 1 < argc) ecm_curves = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--limbs") && i + 1 < argc) limbs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expect") && i + 1 < argc) want = atoi(argv[++i]);
    }

    printf("cofactoriser: %s, method %s, rounds %d, budget %u",
           meth == COF_METHOD_ECM ? "ECM" : "rho", meth == COF_METHOD_ECM ? "" : "",
           rounds, budget);
    if (meth == COF_METHOD_ECM) printf(", B1 %u B2 %u curves %u", ecm_b1, ecm_b2, ecm_curves);
    printf(", %d limbs\n", limbs);

    const char *batch = "/tmp/cofac_metal_batch.txt";
    long ncand = 0;
    if (make_batch(in, batch, 31u, 32u, &ncand)) return 1;
    printf("converted %ld candidates to batch format\n", ncand);

    int rc = run_cofac(batch, out,
                       /*lim0*/ 67100000u, /*lpb0*/ 31u,
                       /*lim1*/ 134200000u, /*lpb1*/ 32u,
                       rounds, budget,
                       /*blocks*/ 256, /*threads*/ 128,
                       meth, meth, ecm_b1, ecm_b2, ecm_curves,
                       /*limbs0*/ limbs, /*limbs1*/ limbs, /*chunk*/ 0);
    if (rc) { printf("run_cofac returned %d\n", rc); return 1; }

    /* Count what it wrote. */
    long nrel = 0;
    if (FILE *f = fopen(out, "r")) {
        char line[4096];
        while (fgets(line, sizeof line, f)) if (line[0] != '#' && line[0] != '\n') nrel++;
        fclose(f);
    }
    printf("relations written: %ld\n", nrel);
    if (want >= 0) {
        if (nrel == want) { printf("PHASE 6a GATE: PASS (expected %d)\n", want); return 0; }
        printf("PHASE 6a GATE: FAIL (expected %d, got %ld)\n", want, nrel);
        return 1;
    }
    return 0;
}
