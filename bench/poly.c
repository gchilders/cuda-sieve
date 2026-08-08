/* Polynomial load + norm-init setup for the apply kernel.
 *
 * The apply kernel must start each sieve cell at the log-norm bound, not at
 * zero. Evaluating |F(a,b)| directly is impossible in fp32: a,b reach ~4e8 and
 * c0 is ~2e44, so F is ~1e48 and the coefficients alone span 1e39 -- both
 * outside float range. The fix is to homogenise the whole thing once on the
 * host:
 *
 *     a = i*a0 + j*b0,  b = i*a1 + j*b1      (q-lattice)
 *     u = a/A, v = b/B                        with |u|,|v| <= 1 by construction
 *     F(a,b) = M * sum_k d_k u^k v^(deg-k)    d_k = c_k A^k B^(deg-k) / M
 *
 * so log2|F| = log2(M) + log2|sum d_k u^k v^(deg-k)|, and everything the GPU
 * touches is O(1) in fp32. This works only because a skewed polynomial has
 * balanced terms by construction -- which is exactly what skew is for. The
 * balance is checked and printed; if it ever fails the d_k spread will show it.
 */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int poly_load(const char *path, poly_t *P)
{
    FILE *f = fopen(path, "r");
    char line[4096];
    if (!f) { fprintf(stderr, "poly_load: cannot open %s\n", path); return -1; }
    memset(P, 0, sizeof(*P));
    P->deg = -1;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'c' && line[1] >= '0' && line[1] <= '9' && line[2] == ':') {
            int k = line[1] - '0';
            P->c[k] = strtod(line + 3, NULL);
            /* Keep the exact decimal too. The double is what the norm-init
             * needs (it is taking a logarithm); trial division needs the
             * integer, and c0 here is 147 bits. */
            sscanf(line + 3, " %79s", P->cs[k]);
            if (P->c[k] != 0.0 && k > P->deg) P->deg = k;
        } else if (!strncmp(line, "skew:", 5)) {
            P->skew = strtod(line + 5, NULL);
        } else if (!strncmp(line, "Y0:", 3)) {
            P->y0 = strtod(line + 3, NULL);
            sscanf(line + 3, " %79s", P->y0s);
        } else if (!strncmp(line, "Y1:", 3)) {
            P->y1 = strtod(line + 3, NULL);
            sscanf(line + 3, " %79s", P->y1s);
        }
        /* Sieve parameters. A GGNFS .job file carries all of these; a CADO
         * .poly carries none, and every one stays 0 so the caller can tell
         * "absent" from "zero". Both loaders used to stop at Y1 and the human
         * retyped the rest onto the command line. */
        else if (!strncmp(line, "rlim:", 5))    P->rlim    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "alim:", 5))    P->alim    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "lpbr:", 5))    P->lpbr    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "lpba:", 5))    P->lpba    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "mfbr:", 5))    P->mfbr    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "mfba:", 5))    P->mfba    = strtoul(line + 5, NULL, 10);
        else if (!strncmp(line, "rlambda:", 8)) P->rlambda = strtod(line + 8, NULL);
        else if (!strncmp(line, "alambda:", 8)) P->alambda = strtod(line + 8, NULL);
    }
    fclose(f);
    if (P->deg < 1) { fprintf(stderr, "poly_load: no algebraic coefficients\n"); return -1; }
    return 0;
}

/* Does f have a root mod 2 -- affine (x = 0 or 1) or projective (2 | c_deg)?
 *
 * If it does not, 2 NEVER divides the algebraic norm, and a factor base with
 * no p = 2 entry is correct rather than truncated. That is not a corner case:
 * the SNFS polynomial x^5 + x^4 - 4x^3 - 3x^2 + 3x + 1 has f(0) = 1, f(1) = -1
 * and an odd leading coefficient, so makefb emits no p = 2 and the norms are
 * always odd.
 *
 * Parity is read off the EXACT decimal strings, not the doubles: c0 runs to
 * 147 bits on some jobs and the double is good to 53. The last digit is all
 * that is needed, and a leading '-' does not change it. */
int poly_has_root_mod2(const poly_t *P)
{
    /* Initialised, though the loop always assigns both: k runs 0..deg, so the
     * k == 0 and k == P->deg arms each fire once. The compiler cannot see it. */
    int sum = 0, c0_even = 0, lead_even = 0;
    for (int k = 0; k <= P->deg; k++) {
        const char *s = P->cs[k];
        size_t n = strlen(s);
        int odd = n && (s[n - 1] - '0') % 2;   /* empty string = absent = 0 */
        sum ^= odd;
        if (k == 0)       c0_even  = !odd;
        if (k == P->deg)  lead_even = !odd;
    }
    if (c0_even)   return 1;        /* x = 0 is a root      */
    if (!sum)      return 1;        /* x = 1 is a root      */
    if (lead_even) return 1;        /* projective root at 2 */
    return 0;
}

double job_allowance_bits(double lambda, uint32_t lim)
{
    if (lambda <= 0.0 || lim < 2) return 0.0;
    return lambda * (log((double)lim) / log(2.0));
}

/* allowance_bits: how many bits of unfactored cofactor we tolerate before a
 * position stops being a survivor. las derives this from lambda*lpb. */
/* norm_setup's two diagnostic lines are per side PER SPECIAL-Q. That is fine
 * for a single-q run and ruinous for a band: a 378,000-q job printed over a
 * million lines and shredded the \r progress display. The band loop prints the
 * first q's setup and then clears this. */
int norm_verbose = 1;

double las_scale(double maxlog2)
{
    double s;
    if (maxlog2 <= 0.0) return 0.0;
    s = (255.0 - 1.0) / maxlog2;          /* UCHAR_MAX - LOGNORM_GUARD_BITS */
    return (double)(int)(s * 40.0) * 0.025;
}

double las_allowance(double maxlog2, double scale, double lambda,
                     unsigned lpb, unsigned mfb)
{
    double r;
    if (scale <= 0.0) return 0.0;
    /* CADO's automatic lambda when none is given. */
    if (lambda <= 0.0) lambda = 0.3 + (double)mfb / (double)lpb;
    r = maxlog2 - 1.0 / scale;            /* LOGNORM_GUARD_BITS / scale */
    if (r > lambda * (double)lpb) r = lambda * (double)lpb;
    return r;
}

/* The survivor allowance we actually want, in bits, derived from THIS tool's
 * behaviour rather than from an imported lambda.
 *
 * A survivor is a position whose approximated remaining log looks small enough
 * to be worth trial dividing. The hard gate downstream is mfb: a cofactor
 * above it is rejected outright, so admitting positions far above mfb buys
 * nothing but work. The only reason to exceed mfb at all is that the survivor
 * test is approximate -- a byte-quantised log against an fp32 norm estimate --
 * so a position whose true cofactor is under mfb can read over it.
 *
 * One byte unit is 1/scale bits, which is the natural granularity of that
 * error, and the floor of 1.5 bits keeps a fine scale from producing a slack
 * so tight that ordinary rounding starts costing relations.
 *
 * Measured, SNFS job side 0 (mfb 88, scale 1.60), 200 special-q:
 *     mfb + 3.8 (91.80, the .job file's rlambda 3.4): 120,317 surv  10,313 rel
 *     mfb + 1.5 (89.50, what this returns)          : 104,615 surv  10,312 rel
 *     mfb + 1.0 (89.00)                             :  99,808 surv  10,312 rel
 *     mfb + 0.0 (88.00)                             :  90,782 surv  10,306 rel
 *     mfb - 3.0 (85.00)                             :  74,849 surv   9,668 rel
 * so 13% of the trial-division input goes away for one relation in ten
 * thousand. Below mfb the cliff is real and fast -- hence a floor, not a cap
 * at mfb exactly.
 *
 * WHY NOT THE .job FILE'S LAMBDA. It is calibrated to a different survivor
 * gate. On the same job and q range, gnfs-lasieve4I14e loses 17.3% of its
 * yield going from 91.8 to 87.5 bits where we lose 0.07% going to 88.0 -- so
 * the number does not transfer, and importing it just inherits another tool's
 * calibration. CADO's automatic (0.3 + mfb/lpb) x lpb is no better a source:
 * on this job it gives 97.3 bits, looser still.
 *
 * Capped by the norm itself: a cofactor cannot exceed the norm it came from,
 * less one guard unit. */
double sieve_allowance(double maxlog2, double scale, unsigned mfb)
{
    double slack, r;
    if (scale <= 0.0) return 0.0;
    slack = 2.0 / scale;
    if (slack < 1.5) slack = 1.5;
    r = (double)mfb + slack;
    if (r > maxlog2 - 1.0 / scale) r = maxlog2 - 1.0 / scale;
    return r;
}

void norm_setup(norm_t *N, const poly_t *P, const qlat_t *L,
                int logI, uint32_t J, double scale, int is_sqside)
{
    const double I = (double)(1u << logI);
    /* |a| and |b| are maximised at the corners of the sieve rectangle. Using
     * the max (not the RMS) guarantees |u|,|v| <= 1 everywhere, so no term can
     * overflow no matter where in the region we land. */
    const double A = 0.5 * I * fabs((double)L->a0) + (double)J * fabs((double)L->b0);
    const double B = 0.5 * I * fabs((double)L->a1) + (double)J * fabs((double)L->b1);
    double d[8], M = 0.0;
    int k, nclamp = 0;

    for (k = 0; k <= P->deg; k++) {
        d[k] = P->c[k] * pow(A, (double)k) * pow(B, (double)(P->deg - k));
        if (fabs(d[k]) > M) M = fabs(d[k]);
    }
    N->deg = P->deg;
    for (k = 0; k <= P->deg; k++) {
        double r = d[k] / M;
        /* below fp32 denormal range the term cannot affect the leading bits.
         * If this fires for a leading coefficient the homogeneous terms are
         * not balanced, which means the q-lattice was reduced without the
         * skew -- see qlat_build.
         *
         * A coefficient that is ZERO in the polynomial is not evidence of
         * that. Sparse SNFS forms are all zeros: x^5 - 6032531 has c1..c4 = 0
         * and flushed four terms on every q. Count only terms that were
         * nonzero and lost, which is the case the check was written for. */
        if (fabs(r) < 1e-37) { N->d[k] = 0.0f; nclamp += (d[k] != 0.0); }
        else N->d[k] = (float)r;
    }
    for (; k < 8; k++) N->d[k] = 0.0f;

    for (k = 0; k <= P->deg; k++) N->dd[k] = d[k] / M;
    for (; k < 8; k++) N->dd[k] = 0.0;
    N->A = A; N->B = B;
    N->a0 = L->a0; N->a1 = L->a1; N->b0 = L->b0; N->b1 = L->b1;

    N->ua  = (float)((double)L->a0 / A);   /* u = ua*i + ub*j */
    N->ub  = (float)((double)L->b0 / A);
    N->va  = (float)((double)L->a1 / B);   /* v = va*i + vb*j */
    N->vb  = (float)((double)L->b1 / B);
    N->log2M = (float)(log(M) / log(2.0));
    N->scale = (float)scale;
    /* ONLY the special-q side's norm carries a factor of q to divide out. The
     * other side's norm is untouched by the special-q. Subtracting log2(q) on
     * both sides makes the non-sq side's threshold ~27 bits too generous and
     * floods the survivor list.
     * Cross-check: las prints log2(maxnorm)=131.86 for side 0 and 196.61 for
     * side 1 here. We compute 131.86 for side 0 (undivided) and 223.6 for
     * side 1, and 223.6 - log2(q) = 196.8. Both match, which is what pins the
     * factor of q to the sq side. */
    N->bias = (float)(is_sqside ? log((double)L->q) / log(2.0) : 0.0);

    /* The unbalanced-terms warning means the q-lattice was reduced without the
     * skew, which is a real defect.
     *
     * It does NOT apply to the degree-1 rational form, which has no balance to
     * lose: G = Y1*a + Y0*b has one huge coefficient and one small one, in
     * whichever orientation the poly generator chose -- input.job has |Y0| = 1
     * against a 136-bit Y1, work/snfs236 has it the other way round -- and the
     * ratio of the two terms is |Y0|B/(|Y1|A), which for this job is ~1e-42. No
     * choice of skew moves that: it is the size of the polynomial, not the
     * shape of the lattice, and skew is fitted to the algebraic side anyway.
     *
     * Dropping the small term is safe, but NOT because of the near-cancellation
     * fallback -- that fallback cannot fire here. Both this file (below) and the
     * device build `aabs` from the already-flushed d[], so with one of two
     * deg-1 terms zeroed, |acc| == aabs identically and `s < TOL*aabs` is never
     * true; norm_acc_fp64 is unreachable on the rational side. What makes it
     * safe is that the estimate is only wrong where the SURVIVING term is
     * smaller than the flushed one, and at a 42-order gap that needs a == 0
     * exactly. A reduced q-lattice has gcd(a0,b0) = 1, so i*a0 + j*b0 == 0 only
     * at i = j = 0, which is outside the region (j >= 1). Zero cells, not few.
     *
     * Named by DEGREE, not by is_sqside. This printed is_sqside in a field
     * labelled "side", which was only ever right while is_sqside == (side==1);
     * --sq-side 0 broke that and the warning then pointed at the wrong
     * polynomial. norm_setup does not know the side index, but it knows the
     * degree, and that identifies the form unambiguously.
     *
     * Deliberately NOT gated on norm_verbose, which the band loop clears after
     * the first q: this is a defect report, not a setup dump, and a lattice that
     * reduces without the skew at q=10^8 has to be able to say so. The zero-term
     * and deg-1 filters above are what make it quiet; the gate would have made
     * it silent. */
    if (nclamp && P->deg > 1)
        printf("  ** q=%llu algebraic (deg %d): %d normalised term(s) flushed"
               " to zero -- terms are unbalanced **\n",
               (unsigned long long)L->q, P->deg, nclamp);
    if (!norm_verbose) return;
    printf("  norm setup: deg %d, A=%.4e B=%.4e, log2(maxnorm)=%.2f, scale=%.3f%s\n",
           P->deg, A, B, N->log2M - N->bias, N->scale,
           is_sqside ? "  (sq side: log2(q) divided out)" : "");
    printf("  normalised coeffs d[0..%d] =", P->deg);
    for (k = 0; k <= P->deg; k++) printf(" %.3g", N->d[k]);
    printf("\n");
}

/* Host mirror of the device norm evaluation, used by the correctness gate. */
/* Exact-in-fp64 evaluation of the same normalised form. (a,b) are formed in
 * int64, so they are exact; only the polynomial sum rounds, at 2^-53 instead of
 * fp32's 2^-24. */
double norm_acc_fp64(const norm_t *N, int32_t i, uint32_t j)
{
    const double a = (double)((int64_t)i * N->a0 + (int64_t)j * N->b0);
    const double b = (double)((int64_t)i * N->a1 + (int64_t)j * N->b1);
    const double u = a / N->A, v = b / N->B;
    double acc = N->dd[N->deg], vp = 1.0;
    int k;
    for (k = N->deg - 1; k >= 0; k--) { vp *= v; acc = acc * u + N->dd[k] * vp; }
    return acc;
}

float norm_target_host(const norm_t *N, int32_t i, uint32_t j)
{
    float u = N->ua * (float)i + N->ub * (float)j;
    float v = N->va * (float)i + N->vb * (float)j;
    float acc = N->d[N->deg], vp = 1.0f;
    /* the same Horner on |.|, which bounds the cancellation: the fp32 error is
     * ~deg*2^-24*aabs, so |acc| << aabs means the result is mostly noise */
    float aabs = fabsf(N->d[N->deg]), vpa = 1.0f, av = fabsf(v), au = fabsf(u);
    float s;
    int k;
    for (k = N->deg - 1; k >= 0; k--) {
        vp  *= v;   acc   = acc * u + N->d[k] * vp;
        vpa *= av;  aabs  = aabs * au + fabsf(N->d[k]) * vpa;
    }
    /* Mirror the DEVICE expression exactly -- same float type, same clamp, same
     * log2f -- because this function is the reference the GPU is gated against.
     * Being more accurate here than the kernel would make "0 cells differ"
     * meaningless: it would be comparing two different functions. */
    s = fabsf(acc);
    if (s < NORM_CANCEL_TOL * aabs) s = (float)fabs(norm_acc_fp64(N, i, j));
    if (s < 1e-30f) s = 1e-30f;
    /* las: fb_log(n) = floor(log2(n)*scale + 0.5) */
    return N->scale * (N->log2M + log2f(s) - N->bias);
}

/* Gauss-reduce the lattice generated by (q,0) and (rho,1) to a basis with
 * entries of size ~sqrt(q). This is what las does per special-q. */
void qlat_build(qlat_t *L, uint64_t q, uint64_t rho, double skew)
{
    /* Lagrange/Gauss reduction of <(q,0),(rho,1)>, under the SKEWED norm
     *     |(a,b)|^2 = (a/sqrt(s))^2 + (b*sqrt(s))^2 .
     * The skew is not cosmetic. An unskewed reduction gives |a| ~ |b| ~
     * sqrt(q); the polynomial's coefficients then span 10^39 across the
     * homogeneous terms and the norm is dominated by c0*b^5 alone. With the
     * skew applied, |a| ~ sqrt(q*s) and |b| ~ sqrt(q/s), the terms c_k a^k
     * b^(d-k) come out within an order of magnitude of each other, and
     * log2|F| lands where las expects it. That balance is the entire purpose
     * of the skew parameter.
     *
     * The invariant is that (a0,a1) is the SHORTER vector and (b0,b1) gets
     * reduced against it; getting the swap backwards leaves the basis
     * unreduced, which silently produces a q-lattice of the wrong shape. */
    int64_t a0 = (int64_t)q, a1 = 0, b0 = (int64_t)rho, b1 = 1;
    const double wa = 1.0 / skew, wb = skew;     /* weights on a^2 and b^2 */
    for (int guard = 0; guard < 200; guard++) {
        double na = wa * (double)a0 * a0 + wb * (double)a1 * a1;
        double nb = wa * (double)b0 * b0 + wb * (double)b1 * b1;
        if (nb < na) {                       /* keep a as the shorter vector */
            int64_t t;
            t = a0; a0 = b0; b0 = t; t = a1; a1 = b1; b1 = t;
            na = nb;
        }
        {
            double dot = wa * (double)a0 * b0 + wb * (double)a1 * b1;
            double mu = dot / na;
            int64_t m = (int64_t)(mu < 0 ? mu - 0.5 : mu + 0.5);
            if (m == 0) break;
            b0 -= m * a0; b1 -= m * a1;
        }
    }
    L->a0 = a0; L->a1 = a1; L->b0 = b0; L->b1 = b1; L->q = q;
}
