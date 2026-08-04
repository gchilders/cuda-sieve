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
    }
    fclose(f);
    if (P->deg < 1) { fprintf(stderr, "poly_load: no algebraic coefficients\n"); return -1; }
    return 0;
}

/* allowance_bits: how many bits of unfactored cofactor we tolerate before a
 * position stops being a survivor. las derives this from lambda*lpb. */
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
         * skew -- see qlat_build. */
        if (fabs(r) < 1e-37) { N->d[k] = 0.0f; nclamp++; }
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

    printf("  norm setup: deg %d, A=%.4e B=%.4e, log2(maxnorm)=%.2f, scale=%.3f%s\n",
           P->deg, A, B, N->log2M - N->bias, N->scale,
           is_sqside ? "  (sq side: log2(q) divided out)" : "");
    printf("  normalised coeffs d[0..%d] =", P->deg);
    for (k = 0; k <= P->deg; k++) printf(" %.3g", N->d[k]);
    printf("%s\n", nclamp ? "   ** unbalanced: terms flushed to zero **" : "");
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
