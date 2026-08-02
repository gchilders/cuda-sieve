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
 * usage: fbtest [--poly P] [--cadofb F] [--rlim N] [--maxbits N]
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

int main(int argc, char **argv)
{
    const char *polypath = "../oracle/c183.poly";
    const char *cadofb = NULL;
    uint64_t q = 120000053ull, rho = 112625526ull;
    uint32_t rlim = 67100000, bkthresh = 1u << 15;
    int maxbits = 15, qmax = 200;
    double scale = 1.0;
    poly_t P; qlat_t L; fb_t fb, fbs;
    int i, n;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--poly") && i + 1 < argc) polypath = argv[++i];
        else if (!strcmp(argv[i], "--cadofb") && i + 1 < argc) cadofb = argv[++i];
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) rlim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) maxbits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--rho") && i + 1 < argc) rho = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--qmax") && i + 1 < argc) qmax = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bkthresh") && i + 1 < argc) bkthresh = (uint32_t)strtoul(argv[++i], 0, 10);
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
        printf("       sum(logp/q) over the whole side-1 base: %.3f"
               "  (scale %.3f)\n", log_density(&fb), scale);
        fb_free(&fbs); fb_free(&fb);
    } else {
        printf("       (skipped: pass --cadofb ../oracle/c183.fb1)\n");
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
        printf("       sum(logp/q) over the whole side-0 base: %.3f"
               "  (scale %.3f)\n", log_density(&fb), scale);
        fb_free(&fbs); fb_free(&fb);
    }

    printf("\n%s\n", fail ? "*** FAILED ***" : "all gates passed");
    return fail;
}
