/* Rational-side factor base, built from scratch.
 *
 * There is no rational .fb on disk to load: GGNFS computes it on the fly and
 * CADO rebuilds it every run ("Creating side-0 rational factor base"). It is
 * cheap to make and needs no library -- the rational polynomial is
 * G(x) = Y1*x + Y0, so every prime p has exactly one root,
 *
 *     r = -Y0 * Y1^-1  (mod p),
 *
 * and p | Y1 is the projective case (encoded r == p, matching the .afb.0
 * convention). Y0 and Y1 are 118 and 76 bits, so they are kept as decimal
 * strings in poly_t and reduced mod p by limb-wise Horner in base 2^32 --
 * doubles would lose the low bits and give wrong roots, even though doubles
 * are perfectly adequate for the *norm* (which only needs a logarithm).
 */
#include "bench.h"
#include "plattice.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXLIMB 8

/* las's fb_log_delta: the marginal log of going from p^oldexp to p^nexp. */
static uint8_t fb_log_delta(uint32_t p, int nexp, int oldexp, double scale)
{
    double l2 = log2((double)p);
    long a = (long)floor((double)nexp   * l2 * scale + 0.5);
    long b = (long)floor((double)oldexp * l2 * scale + 0.5);
    long d = a - b;
    return (uint8_t)(d < 0 ? 0 : (d > 255 ? 255 : d));
}

typedef struct { uint32_t l[MAXLIMB]; int n, neg; } bigint_t;

/* decimal string -> base-2^32 limbs, little-endian */
static int big_parse(const char *s, bigint_t *B)
{
    memset(B, 0, sizeof(*B));
    B->n = 1;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') { B->neg = 1; s++; } else if (*s == '+') s++;
    for (; *s >= '0' && *s <= '9'; s++) {
        uint64_t carry = (uint64_t)(*s - '0');
        int i;
        for (i = 0; i < B->n; i++) {
            uint64_t v = (uint64_t)B->l[i] * 10u + carry;
            B->l[i] = (uint32_t)v;
            carry = v >> 32;
        }
        if (carry) {
            if (B->n >= MAXLIMB) return -1;
            B->l[B->n++] = (uint32_t)carry;
        }
    }
    return 0;
}

/* |B| mod p, then signed */
static uint32_t big_mod(const bigint_t *B, uint32_t p)
{
    uint64_t r = 0;
    int i;
    for (i = B->n - 1; i >= 0; i--)
        r = ((r << 32) | (uint64_t)B->l[i]) % (uint64_t)p;
    if (B->neg && r) r = (uint64_t)p - r;
    return (uint32_t)r;
}

int rfb_build(const poly_t *P, uint32_t lim, int maxbits, double scale, fb_t *fb)
{
    bigint_t Y0, Y1;
    unsigned char *sieve;                  /* odd numbers only: bit k <-> 2k+3 */
    uint32_t nodd, i, k = 0, cap;
    uint32_t nproj = 0, npow = 0;
    const uint32_t powmax = (maxbits > 0 && maxbits < 31) ? (1u << maxbits) : 0;

    memset(fb, 0, sizeof(*fb));
    if (big_parse(P->y0s, &Y0) || big_parse(P->y1s, &Y1)) {
        fprintf(stderr, "rfb_build: Y0/Y1 too large for %d limbs\n", MAXLIMB);
        return -1;
    }
    if (lim < 8) return -1;

    nodd  = (lim - 1) / 2;                 /* candidates 3,5,...,<=lim */
    sieve = (unsigned char *)calloc(nodd / 8 + 1, 1);
    if (!sieve) return -1;
    for (i = 0; (uint64_t)(2 * i + 3) * (2 * i + 3) <= (uint64_t)lim; i++) {
        uint32_t p, j;
        if (sieve[i >> 3] & (1u << (i & 7))) continue;
        p = 2 * i + 3;
        for (j = (p * p - 3) / 2; j < nodd; j += p)
            sieve[j >> 3] |= (uint8_t)(1u << (j & 7));
    }

    /* pi(x) ~ x/ln x; 12% headroom is plenty and this is a one-time alloc */
    cap = (uint32_t)((double)lim / (log((double)lim) - 1.1)) + 4096;
    fb->primes = (uint32_t *)malloc((size_t)cap * 4);
    fb->roots  = (uint32_t *)malloc((size_t)cap * 4);
    fb->logp   = (uint8_t  *)malloc((size_t)cap);
    if (!fb->primes || !fb->roots || !fb->logp) { free(sieve); return -1; }

    /* p = 2 first, then the odd primes in order -- fb_split_small and
     * fb_restrict both assume non-decreasing p */
    {
        uint32_t y1 = big_mod(&Y1, 2), y0 = big_mod(&Y0, 2);
        fb->primes[k] = 2;
        fb->roots[k]  = y1 ? (uint32_t)((2 - y0 % 2) % 2) : 2;
        fb->logp[k]   = fb_log_delta(2, 1, 0, scale);
        if (!y1) nproj++;
        k++;
        /* powers of two need a 2-adic inverse, not the binary-Euclid one */
        if (y1) {
            uint32_t qk = 2; int e;
            for (e = 2; e <= 31 && k < cap; e++) {
                uint64_t nq = (uint64_t)qk * 2;
                uint32_t yy1, yy0, inv;
                int t = 0;
                if (!powmax || nq > powmax) break;
                qk = (uint32_t)nq;
                yy1 = big_mod(&Y1, qk);
                if ((yy1 & 1) == 0) break;
                yy0 = big_mod(&Y0, qk);
                (void)t;
                inv = pl_invmod_any(yy1, qk);
                fb->primes[k] = qk;
                fb->roots[k]  = (uint32_t)(((uint64_t)(qk - yy0 % qk) % qk
                                            * (uint64_t)inv) % (uint64_t)qk);
                fb->logp[k]   = fb_log_delta(2, e, e - 1, scale);
                k++; npow++;
            }
        }
    }
    for (i = 0; i < nodd && k < cap; i++) {
        uint32_t p, y0, y1, qk;
        int e;
        if (sieve[i >> 3] & (1u << (i & 7))) continue;
        p  = 2 * i + 3;
        y1 = big_mod(&Y1, p);
        fb->primes[k] = p;
        fb->logp[k] = fb_log_delta(p, 1, 0, scale);
        if (y1 == 0) { fb->roots[k] = p; nproj++; k++; continue; }
        y0 = big_mod(&Y0, p);
        /* r = -Y0/Y1 mod p */
        fb->roots[k] = (uint32_t)(((uint64_t)(p - y0 % p) % p
                                   * pl_invmod(y1, p)) % (uint64_t)p);
        k++;
        /* prime powers, exactly as makefb emits them: modulus p^e, root lifted
         * mod p^e (for a linear polynomial no Hensel step is needed -- the
         * inverse of Y1 mod p^e is enough), and the log increment is the
         * MARGINAL fb_log(p^e) - fb_log(p^(e-1)), not log(p^e). */
        for (e = 2, qk = p; e <= 31 && k < cap; e++) {
            uint64_t nq = (uint64_t)qk * p;
            uint32_t yy1, yy0;
            if (!powmax || nq > powmax) break;
            qk = (uint32_t)nq;
            yy1 = big_mod(&Y1, qk);
            if (yy1 % p == 0) break;           /* p | Y1: no lift */
            yy0 = big_mod(&Y0, qk);
            fb->primes[k] = qk;
            fb->roots[k]  = (uint32_t)(((uint64_t)(qk - yy0 % qk) % qk
                                        * pl_invmod(yy1, qk)) % (uint64_t)qk);
            fb->logp[k]   = fb_log_delta(p, e, e - 1, scale);
            k++; npow++;
        }
    }
    free(sieve);
    fb->n = k;
    /* powers were emitted next to their base prime, so q is not yet ascending */
    {
        uint32_t a;
        for (a = 1; a < fb->n; a++) {
            uint32_t q = fb->primes[a], r = fb->roots[a]; uint8_t l = fb->logp[a];
            int32_t m = (int32_t)a - 1;
            if (fb->primes[m] <= q) continue;
            while (m >= 0 && fb->primes[m] > q) {
                fb->primes[m+1] = fb->primes[m]; fb->roots[m+1] = fb->roots[m];
                fb->logp[m+1] = fb->logp[m]; m--;
            }
            fb->primes[m+1] = q; fb->roots[m+1] = r; fb->logp[m+1] = l;
        }
    }
    printf("rational factor base: %u ideals up to %u (%u prime-power, %u projective)"
           " at scale %.3f\n", k, lim, npow, nproj, scale);
    return 0;
}
