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

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "stat"))  return cmd_stat(argv[2]);
    if (argc >= 4 && !strcmp(argv[1], "diff"))
        return cmd_diff(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 0);
    fprintf(stderr, "usage: dumpcmp stat FILE | dumpcmp diff A B [tol]\n");
    return 1;
}
