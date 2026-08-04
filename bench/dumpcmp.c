/* Inspect and diff `las -dumpfile` sieve regions.
 *
 * The dump is written in las-process-bucket-region.cpp right after SminusS():
 *   init_norms(side)   -> S[x]  = round(scale * log2|F(a,b)|), capped at 255
 *   apply_buckets + small_sieve -> SS[x] = sum of scaled log p landing on x
 *   SminusS            -> S[x]  = max(S[x] - SS[x], 0)
 * and a position is a survivor when S[x] <= bound (las prints `bound` per side
 * with -v). So the file is exactly one byte per sieve position, in x order,
 * I*J bytes total.
 *
 * Our kernel stores the complement -- cells start at CINIT - T(x) and logs are
 * ADDED, because a GPU 16-bit half-word atomic cannot borrow. To compare,
 * convert ours to las's convention before diffing (see --ours).
 *
 * usage: dumpcmp stat  FILE
 *        dumpcmp diff  FILE_A FILE_B [tolerance]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHUNK (1u << 22)

static long long hist_file(const char *path, unsigned long long *h)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long long total = 0;
    size_t got;
    if (!f) { perror(path); return -1; }
    buf = (unsigned char *)malloc(CHUNK);
    while ((got = fread(buf, 1, CHUNK, f)) > 0) {
        size_t i;
        for (i = 0; i < got; i++) h[buf[i]]++;
        total += (long long)got;
    }
    free(buf); fclose(f);
    return total;
}

static int cmd_stat(const char *path)
{
    unsigned long long h[256], cum = 0;
    long long n;
    int i, lo = 255, hi = 0;
    double mean = 0;
    memset(h, 0, sizeof h);
    n = hist_file(path, h);
    if (n < 0) return 1;
    for (i = 0; i < 256; i++) if (h[i]) { if (i < lo) lo = i; if (i > hi) hi = i; }
    for (i = 0; i < 256; i++) mean += (double)i * h[i];
    mean /= (double)n;
    printf("%s\n  %lld bytes (I*J positions), values in [%d,%d], mean %.2f\n",
           path, n, lo, hi, mean);
    printf("  cumulative counts at candidate thresholds:\n");
    for (i = 0; i < 256; i++) {
        cum += h[i];
        if (i == 0 || i == 10 || i == 50 || i == 100 || i == 140 || i == 141 ||
            i == 143 || i == 150 || i == 200 || i == 255)
            printf("    S <= %3d : %12llu  (1 in %.4g)\n", i, cum,
                   cum ? (double)n / cum : 0.0);
    }
    printf("  top of the histogram:");
    for (i = hi; i > hi - 6 && i >= 0; i--) printf(" [%d]=%llu", i, h[i]);
    printf("\n");
    return 0;
}

static int cmd_diff(const char *pa, const char *pb, int tol)
{
    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    unsigned char *ba, *bb;
    long long pos = 0, ndiff = 0, first = -1;
    long long histd[513];
    size_t ga, gb;
    if (!fa || !fb) { perror("open"); return 1; }
    memset(histd, 0, sizeof histd);
    ba = (unsigned char *)malloc(CHUNK);
    bb = (unsigned char *)malloc(CHUNK);
    for (;;) {
        size_t i, n;
        ga = fread(ba, 1, CHUNK, fa);
        gb = fread(bb, 1, CHUNK, fb);
        n = ga < gb ? ga : gb;
        if (!n) break;
        for (i = 0; i < n; i++) {
            int d = (int)ba[i] - (int)bb[i];
            histd[d + 256]++;
            if (d > tol || d < -tol) {
                if (first < 0) first = pos + (long long)i;
                ndiff++;
            }
        }
        pos += (long long)n;
        if (ga != gb) break;
    }
    printf("compared %lld positions, tolerance +/-%d\n", pos, tol);
    printf("  differing: %lld (%.4g%%)\n", ndiff, 100.0 * ndiff / (pos ? pos : 1));
    if (first >= 0) printf("  first at x = %lld\n", first);
    printf("  delta distribution (a - b):");
    { int d; for (d = -8; d <= 8; d++) if (histd[d + 256])
        printf(" %+d:%lld", d, histd[d + 256]); }
    printf("\n");
    free(ba); free(bb); fclose(fa); fclose(fb);
    return ndiff != 0;
}

/* Two-sided survivor intersection.
 *
 * las reports only `survivors after_sieve`, which is ALREADY the intersection
 * of the two sides -- it never prints a one-sided count, and the byte dump that
 * would have given us one is broken (finding 17). So this is the only survivor
 * figure our sieve can be gated against, and it needs both sides' bitmaps.
 *
 * Inputs are the `bench --survbits` files: one bit per sieve position in x
 * order, x = j*I + (i + I/2), little-endian 32-bit words. */
static uint32_t gcd32(uint32_t a, uint32_t b)
{
    while (b) { uint32_t t = a % b; a = b; b = t; }
    return a;
}

/* CADO's survivor population is the PRIMITIVE points: `after_sieve` has
 * already dropped every (i,j) with gcd(i,j) > 1, not merely the both-even ones
 * (prototype.md's parity-gotcha list says so; RESULTS finding 40 forgot it and
 * compared two different populations). j > 0 throughout, and i = 0 is primitive
 * only at j = 1, so gcd(|i|, j) == 1 is the whole test. */
static int cmd_and(const char *pa, const char *pb, int logI)
{
    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    uint32_t *ba, *bb;
    unsigned long long na = 0, nb = 0, nboth = 0, nword = 0;
    unsigned long long ca = 0, cb = 0, cboth = 0;
    const uint32_t I = 1u << logI;
    size_t ga, gb;
    if (!fa || !fb) { perror("open"); return 1; }
    ba = (uint32_t *)malloc(CHUNK);
    bb = (uint32_t *)malloc(CHUNK);
    for (;;) {
        size_t i, n;
        ga = fread(ba, 1, CHUNK, fa);
        gb = fread(bb, 1, CHUNK, fb);
        n = (ga < gb ? ga : gb) / 4;
        if (!n) break;
        for (i = 0; i < n; i++) {
            uint32_t both = ba[i] & bb[i], m;
            na    += (unsigned)__builtin_popcount(ba[i]);
            nb    += (unsigned)__builtin_popcount(bb[i]);
            nboth += (unsigned)__builtin_popcount(both);
            if (!logI) continue;
            /* primitive-point filter, one gcd per SET bit only */
            for (m = ba[i] | bb[i]; m; m &= m - 1) {
                unsigned bit = (unsigned)__builtin_ctz(m);
                unsigned long long x = (nword + i) * 32ull + bit;
                uint32_t j  = (uint32_t)(x >> logI);
                int32_t  ii = (int32_t)(x & (I - 1)) - (int32_t)(I >> 1);
                uint32_t ai = (uint32_t)(ii < 0 ? -ii : ii);
                if (gcd32(ai, j) != 1) continue;
                if (ba[i] >> bit & 1u) ca++;
                if (bb[i] >> bit & 1u) cb++;
                if (both  >> bit & 1u) cboth++;
            }
        }
        nword += n;
        if (ga != gb) break;
    }
    if (ga != gb)
        fprintf(stderr, "warning: bitmaps differ in length, compared %llu words\n",
                nword);
    printf("%llu positions (%llu words)\n", nword * 32, nword);
    printf("  side A survivors : %12llu  (1 in %.4g)\n", na,
           na ? (double)(nword * 32) / na : 0.0);
    printf("  side B survivors : %12llu  (1 in %.4g)\n", nb,
           nb ? (double)(nword * 32) / nb : 0.0);
    printf("  BOTH sides       : %12llu  (1 in %.4g)\n", nboth,
           nboth ? (double)(nword * 32) / nboth : 0.0);
    /* If the two sides were independent the intersection would be
     * na*nb/total; las's own product estimate overshot its measured count by
     * about 20%, so the correlation is real and worth printing. */
    if (logI) {
        printf("  --- primitive points only (gcd(i,j) == 1), CADO's population ---\n");
        printf("  side A survivors : %12llu\n", ca);
        printf("  side B survivors : %12llu\n", cb);
        printf("  BOTH sides       : %12llu\n", cboth);
    }
    if (nword) {
        double indep = (double)na * (double)nb / (double)(nword * 32);
        printf("  if independent   : %12.0f  (measured / independent = %.3f)\n",
               indep, indep > 0 ? (double)nboth / indep : 0.0);
    }
    free(ba); free(bb); fclose(fa); fclose(fb);
    return 0;
}

/* Joint (S_side1, S_side0) histogram, so the two-sided survivor count can be
 * read off at EVERY pair of bounds from one pass instead of one GPU run per
 * pair. 256x256 counters, one increment per position.
 *
 * The point is to separate two explanations of a two-sided count that is too
 * high: a uniform norm/threshold offset (which shows up as "tighten both bounds
 * by k and the count lands on las's") versus something structural (which does
 * not land at any plausible k). */
static int cmd_joint(const char *pa, const char *pb, int logI)
{
    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    unsigned char *ba, *bb;
    static unsigned long long h[256][256];
    unsigned long long pos = 0;
    int t1, t0;
    if (!fa || !fb) { perror("open"); return 1; }
    ba = (unsigned char *)malloc(CHUNK);
    bb = (unsigned char *)malloc(CHUNK);
    for (;;) {
        size_t i, n, ga = fread(ba, 1, CHUNK, fa), gb = fread(bb, 1, CHUNK, fb);
        n = ga < gb ? ga : gb;
        if (!n) break;
        for (i = 0; i < n; i++) {
            unsigned long long x = pos + i;
            /* las's not_both_even; see k_apply for why this is x's parity */
            if (((x & 1u) == 0u) && (((x >> logI) & 1u) == 0u)) continue;
            h[ba[i]][bb[i]]++;
        }
        pos += n;
        if (ga != gb) break;
    }
    free(ba); free(bb); fclose(fa); fclose(fb);
    /* cumulative in both axes: h[t1][t0] becomes #{S1 <= t1 and S0 <= t0} */
    for (t1 = 0; t1 < 256; t1++)
        for (t0 = 1; t0 < 256; t0++) h[t1][t0] += h[t1][t0 - 1];
    for (t0 = 0; t0 < 256; t0++)
        for (t1 = 1; t1 < 256; t1++) h[t1][t0] += h[t1 - 1][t0];
    printf("%llu positions after not_both_even (of %llu)\n",
           h[255][255], pos);
    printf("two-sided survivors, rows = side-1 bound, cols = side-0 bound\n");
    printf("        ");
    for (t0 = 135; t0 <= 153; t0 += 2) printf("%10d", t0);
    printf("\n");
    for (t1 = 137; t1 <= 155; t1 += 2) {
        printf("  %3d : ", t1);
        for (t0 = 135; t0 <= 153; t0 += 2) printf("%10llu", h[t1][t0]);
        printf("\n");
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "stat"))  return cmd_stat(argv[2]);
    if (argc >= 4 && !strcmp(argv[1], "diff"))
        return cmd_diff(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0);
    if (argc >= 4 && !strcmp(argv[1], "and"))
        return cmd_and(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0);
    if (argc >= 5 && !strcmp(argv[1], "joint"))
        return cmd_joint(argv[2], argv[3], atoi(argv[4]));
    fprintf(stderr, "usage: dumpcmp stat FILE | dumpcmp diff A B [tol]\n"
                    "       dumpcmp and BITMAP_A BITMAP_B   (two-sided survivors)\n"
                    "       dumpcmp joint DUMP_S1 DUMP_S0 logI  (count vs both bounds)\n");
    return 1;
}
