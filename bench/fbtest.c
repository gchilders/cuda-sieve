/* CPU-only correctness gates for the factor-base and transform layer.
 *
 * Deliberately a separate binary from `bench`: everything here runs without a
 * GPU, which means these gates can be run on a busy box, in a loop, and by
 * anyone who does not have an sm_120 card. `bench --verify` covers the
 * GPU-vs-CPU cross-check; this covers the layer underneath it, where the two
 * bugs found on 2026-08-02 lived -- both of which were invisible to every gate
 * that compared us against ourselves.
 *
 * Gates, in increasing order of what they can catch:
 *   1. transform vs definition, set equality, synthetic moduli 2..QMAX and
 *      every root in [0,2q). Primes, prime powers, even moduli, affine and
 *      projective, zero and nonzero reciprocal.
 *   2. the same, driven by the real factor bases, so a loader that mangles the
 *      encoding fails even though the algebra is right.
 *   3. sum(logp/q) over the factor base -- the oracle-free density check. It
 *      cannot see placement (that is what gate 1 is for), but it is the one
 *      number that exercises parse, powers, log deltas and scale at once.
 *
 * usage: fbtest [--poly P] [--fb1 F] [--rlim N] [--maxbits N]
 *               [--q N] [--rho N] [--scale S] [--qmax N]
 */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

static int fail = 0;

static void ok(const char *what, int good, const char *fmt, ...)
{
    va_list ap;
    printf("%-6s %-34s ", good ? "PASS" : "FAIL", what);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (!good) fail = 1;
}

/* sum over the factor base of logp/q: the expected sieved log per position. */
static double log_density(const fb_t *fb)
{
    double s = 0;
    uint32_t i;
    for (i = 0; i < fb->n; i++) s += (double)fb->logp[i] / (double)fb->primes[i];
    return s;
}

static uint32_t count_proj(const fb_t *fb, uint32_t *nonzero)
{
    uint32_t i, n = 0;
    *nonzero = 0;
    for (i = 0; i < fb->n; i++)
        if (fb->roots[i] >= fb->primes[i]) {
            n++;
            if (fb->roots[i] > fb->primes[i]) (*nonzero)++;
        }
    return n;
}

/* Trace one position: every factor-base ideal that hits (i,j), and the log
 * sum they contribute. This is the half of gate 5 that lives on our side --
 * `las_tracek -traceab a,b` prints the same thing from las's pipeline, and
 * `-traceab` is basis-independent, so the two are directly comparable without
 * worrying that our basis is las's with the first vector negated. */
static void trace_side(const char *name, const fb_t *fb, const qlat_t *L,
                       const poly_t *P, int32_t i, int32_t j,
                       uint32_t bkthresh, int logI, uint32_t J,
                       double scale, int is_sq, uint64_t sq, int maxlist)
{
    uint64_t sum = 0;
    uint32_t k, nhit = 0;
    norm_t N;
    long T;
    memset(&N, 0, sizeof N);
    norm_setup(&N, P, L, logI, J, scale, is_sq);
    T = (long)floorf(norm_target_host(&N, i, j) + 0.5f);
    if (T < 0) T = 0;

    printf("  %s (scale %.2f): ideals hitting (i=%d, j=%d)\n", name, scale, i, j);
    for (k = 0; k < fb->n; k++) {
        const uint32_t q = fb->primes[k], r = fb->roots[k];
        if ((uint64_t)q == sq) continue;      /* the special-q is divided out */
        if (!hits_def_pub(q, r, L, i, j)) continue;
        sum += fb->logp[k];
        if ((int)nhit < maxlist)
            printf("      q=%-10u %-11s logp=%-4u %s\n", q,
                   r >= q ? "projective" : "affine", fb->logp[k],
                   q < bkthresh || fb_is_proper_power(q) ? "line-sieved" : "bucketed");
        nhit++;
    }
    if ((int)nhit > maxlist) printf("      ... %u more\n", nhit - maxlist);
    printf("  %s: %u ideals, init norm S = %ld, log sum = %llu,"
           "  S - sum = %ld\n\n",
           name, nhit, T, (unsigned long long)sum, T - (long)sum);
}

int main(int argc, char **argv)
{
    const char *polypath = "../oracle/c183.poly";
    const char *cadofb = NULL;
    uint64_t q = 120000053ull, rho = 112625526ull;
    uint32_t rlim = 67100000, bkthresh = 1u << 15;
    int maxbits = 15, qmax = 200;
    int32_t ti = 0, tj = 0; int do_trace = 0;
    /* las's EXACT scales. It prints them rounded to 2 dp ("scale=1.28"), but also
 * prints logbase = 2^(1/scale) to 7 figures, from which scale = 1/log2(logbase)
 * comes out at exactly 1.275 and 1.925. The 2-dp values are wrong by enough to
 * move fb_log by one unit for some primes -- p=25811171 gives 32 at 1.28 and
 * 31, which is what las applies, at 1.275. */
    double scale1 = 1.275, scale0 = 1.925;
    double scale = 1.0;
    poly_t P; qlat_t L; fb_t fb, fbs;
    int i, n;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) polypath = argv[++i];
        else if ((!strcmp(argv[i], "--fb1") || !strcmp(argv[i], "--cadofb")) && i + 1 < argc)
            cadofb = argv[++i];
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) rlim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) maxbits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--rho") && i + 1 < argc) rho = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--scale1") && i + 1 < argc) scale1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) scale0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--qmax") && i + 1 < argc) qmax = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bkthresh") && i + 1 < argc) bkthresh = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d", &ti, &tj) != 2) {
                fprintf(stderr, "--trace wants i,j\n"); return 2;
            }
            do_trace = 1;
        }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    if (poly_load(polypath, &P) != 0) return 1;
    qlat_build(&L, q, rho, P.skew);
    printf("q=%llu rho=%llu  basis (a0,a1,b0,b1)=(%lld,%lld,%lld,%lld)\n",
           (unsigned long long)q, (unsigned long long)rho,
           (long long)L.a0, (long long)L.a1, (long long)L.b0, (long long)L.b1);
    /* b = i*a1 + j*b1, and for a special-q of this size the reduced basis has
     * b-components of order sqrt(q/skew) ~ 1. Print it: it is the fact that
     * decides whether a projective ideal hits one row or every row. */
    printf("  b(i,j) = %lld*i + %lld*j   ->  a projective ideal above p hits"
           " %s\n\n", (long long)L.a1, (long long)L.b1,
           L.b1 == 0 ? "i == 0 (mod p) on EVERY row" : "a full lattice");

    printf("[1] transform vs definition, set equality, q <= %d, all roots"
           " in [0,2q)\n", qmax);
    ok("synthetic moduli", verify_transform(&L, qmax) == 0,
       "primes, prime powers, even moduli, affine + projective");

    if (do_trace) {
        const int64_t a = (int64_t)ti * L.a0 + (int64_t)tj * L.b0;
        const int64_t b = (int64_t)ti * L.a1 + (int64_t)tj * L.b1;
        printf("[trace] (i,j) = (%d,%d)  ->  (a,b) = (%lld,%lld)\n",
               ti, tj, (long long)a, (long long)b);
        printf("        compare with:  las_tracek ... -traceab %lld,%lld\n"
               "        (-traceab is basis-independent, so the i-mirroring\n"
               "         between our basis and las's does not matter here)\n\n",
               (long long)a, (long long)b);
        if (cadofb) {
            if (fb_load_cado(cadofb, scale1, &fb) != 0) return 1;
            trace_side("side 1", &fb, &L, &P, ti, tj, bkthresh, 15, 16384,
                       scale1, 1, q, 40);
            fb_free(&fb);
        }
        {
            poly_t R = P;
            if (rfb_build(&R, rlim, maxbits, scale0, &fb) != 0) return 1;
            R.deg = 1; R.c[0] = R.y0; R.c[1] = R.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) R.c[z] = 0.0;
            trace_side("side 0", &fb, &L, &R, ti, tj, bkthresh, 15, 16384,
                       scale0, 0, 0, 40);
            fb_free(&fb);
        }
        return 0;
    }

    printf("\n[2] transform vs definition, driven by the real factor bases\n");

    if (cadofb) {
        uint32_t np, nzp;
        if (fb_load_cado(cadofb, scale, &fb) != 0) return 1;
        np = count_proj(&fb, &nzp);
        if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
        fb_fill_logp(&fbs, scale);
        n = verify_fb_transform(&fbs, &L, bkthresh);
        ok("side 1 small part", n > 0, "%d entries checked", n);
        ok("side 1 moduli are prime powers", fb_check_prime_powers(&fb) < 0,
           "%u ideals", fb.n);
        ok("side 1 projective entries seen", np > 0,
           "%u projective, %u with a NONZERO reciprocal", np, nzp);
        {   /* The oracle-free density gate, as a PASS/FAIL rather than a
             * print, so a log-parser or scale regression fails `make check`.
             *
             * 42.2913 was computed INDEPENDENTLY from c183.fb1 -- a separate
             * implementation that re-derives each entry's base prime and log
             * delta from the file -- not read off our own loader. It is the
             * post-2026-08-02 value: before the is_power fix it was 42.744, and
             * wiring this gate is what caught that the constant was stale.
             * fb_log's rounding is not linear in scale, so gate only at 1.0. */
            const double got = log_density(&fb);
            if (scale == 1.0)
                ok("side 1 sum(logp/q)", fabs(got - 42.2913) < 0.002,
                   "%.4f (independent expectation 42.2913)", got);
            else
                printf("       side 1 sum(logp/q) = %.4f at scale %.3f"
                       " (gate runs at scale 1.0 only)\n", got, scale);
        }
        fb_free(&fbs); fb_free(&fb);
    } else {
        printf("       (skipped: pass --fb1 ../oracle/c183.fb1)\n");
    }

    {
        uint32_t np, nzp;
        poly_t R = P;
        if (rfb_build(&R, rlim, maxbits, scale, &fb) != 0) return 1;
        np = count_proj(&fb, &nzp);
        if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
        fb_fill_logp(&fbs, scale);
        n = verify_fb_transform(&fbs, &L, bkthresh);
        ok("side 0 small part", n > 0, "%d entries checked", n);
        ok("side 0 moduli are prime powers", fb_check_prime_powers(&fb) < 0,
           "%u ideals", fb.n);
        /* Y1 = 59*101*127*281*1259*38321*5746453, so seven projective primes,
         * and the ladders 59^2, 101^2, 127^2 stay under 2^15. */
        ok("side 0 projective ladder", nzp >= 3,
           "%u projective, %u with a NONZERO reciprocal (expect >= 3: 59^2,"
           " 101^2, 127^2)", np, nzp);
        {   /* likewise independent: a separate sieve to rlim plus the power
             * ladder, which also reproduces the ideal count 3,957,374 exactly */
            const double got = log_density(&fb);
            if (scale == 1.0)
                ok("side 0 sum(logp/q)", fabs(got - 25.2037) < 0.002,
                   "%.4f (independent expectation 25.2037)", got);
            else
                printf("       side 0 sum(logp/q) = %.4f at scale %.3f"
                       " (gate runs at scale 1.0 only)\n", got, scale);
        }
        fb_free(&fbs); fb_free(&fb);
    }

    printf("\n[3] exact-norm width admission\n");
    {
        norm_t W;
        memset(&W, 0, sizeof(W));
        W.deg = 8;
        W.log2M = 252.0f;       /* + log2(9) = 255.17: fits 256 bits */
        ok("octic below 256-bit bound", norm_fits_exact(&W, 256),
           "upper bound %.2f bits", norm_exact_bound_bits(&W));
        W.log2M = 253.0f;       /* + log2(9) = 256.17: must be refused */
        ok("octic above 256-bit bound", !norm_fits_exact(&W, 256),
           "upper bound %.2f bits refused", norm_exact_bound_bits(&W));
    }

    printf("\n%s\n", fail ? "*** FAILED ***" : "all gates passed");
    return fail;
}
