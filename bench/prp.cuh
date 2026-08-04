/* Strong probable-prime test, base 2, for moduli up to 128 bits.
 *
 * This is the test CADO applies to a leftover norm, and its comment explains
 * why base 2 alone is enough: a composite wrongly called prime loses a
 * relation, but never emits a relation with a composite ideal. So the failure
 * mode is a missed relation at a rate around 2^-40, not a wrong answer.
 * `check_leftover_norm` uses modredc*_sprp2 for exactly this.
 *
 * Montgomery arithmetic on four 32-bit limbs. The alternative -- plain modular
 * multiplication -- needs a 256-by-128 division per step, and division by a
 * runtime divisor is precisely what the trial-division work showed to be the
 * expensive thing on this hardware.
 *
 * Only cofactors between lpb and mfb bits ever reach here (92 bits at most on
 * this job), so four limbs is one limb of headroom.
 */
#ifndef CUDA_SIEVE_PRP_CUH
#define CUDA_SIEVE_PRP_CUH

#include <stdint.h>
#include "bigint.cuh"

#define M_LIMBS 4

typedef struct { uint32_t v[M_LIMBS]; } m128_t;

#if defined(__CUDACC__)
#define PRP_FN __host__ __device__ static inline
#else
#define PRP_FN static inline
#endif

PRP_FN int m_cmp(const m128_t *a, const m128_t *b)
{
    for (int i = M_LIMBS - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

PRP_FN uint32_t m_sub(m128_t *a, const m128_t *b)
{
    uint64_t borrow = 0;
    for (int i = 0; i < M_LIMBS; i++) {
        uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow;
        a->v[i] = (uint32_t)d;
        borrow = (d >> 32) & 1u;
    }
    return (uint32_t)borrow;
}

/* a = 2a mod n, given a < n. Returns nothing; the carry out of the top limb
 * is folded in, which matters because 2a can need 129 bits. */
PRP_FN void m_dbl_mod(m128_t *a, const m128_t *n)
{
    uint32_t carry = 0;
    for (int i = 0; i < M_LIMBS; i++) {
        uint32_t hi = a->v[i] >> 31;
        a->v[i] = (a->v[i] << 1) | carry;
        carry = hi;
    }
    if (carry || m_cmp(a, n) >= 0) m_sub(a, n);
}

/* -n^{-1} mod 2^32, by Newton iteration. n must be odd. */
PRP_FN uint32_t m_n0inv(uint32_t n0)
{
    uint32_t inv = n0;                   /* correct to 3 bits for odd n0 */
    for (int k = 0; k < 5; k++) inv *= 2u - n0 * inv;   /* doubles each time */
    return (uint32_t)(0u - inv);
}

/* Montgomery product: r = a*b*R^-1 mod n, R = 2^128. CIOS form. */
PRP_FN void m_mul(m128_t *r, const m128_t *a, const m128_t *b,
                  const m128_t *n, uint32_t n0inv)
{
    uint32_t t[M_LIMBS + 2];
    for (int i = 0; i < M_LIMBS + 2; i++) t[i] = 0;

    for (int i = 0; i < M_LIMBS; i++) {
        uint64_t C = 0;
        for (int j = 0; j < M_LIMBS; j++) {
            uint64_t p = (uint64_t)a->v[j] * b->v[i] + t[j] + C;
            t[j] = (uint32_t)p;
            C = p >> 32;
        }
        {
            uint64_t p = (uint64_t)t[M_LIMBS] + C;
            t[M_LIMBS] = (uint32_t)p;
            t[M_LIMBS + 1] = (uint32_t)(p >> 32);
        }
        {
            uint32_t m = t[0] * n0inv;
            uint64_t p = (uint64_t)m * n->v[0] + t[0];
            C = p >> 32;
            for (int j = 1; j < M_LIMBS; j++) {
                p = (uint64_t)m * n->v[j] + t[j] + C;
                t[j - 1] = (uint32_t)p;
                C = p >> 32;
            }
            p = (uint64_t)t[M_LIMBS] + C;
            t[M_LIMBS - 1] = (uint32_t)p;
            t[M_LIMBS] = t[M_LIMBS + 1] + (uint32_t)(p >> 32);
        }
    }
    for (int i = 0; i < M_LIMBS; i++) r->v[i] = t[i];
    if (t[M_LIMBS] || m_cmp(r, n) >= 0) m_sub(r, n);
}

/* Strong probable prime test, base 2. n must be odd and at least 3.
 * Returns 1 if n is a base-2 strong probable prime. */
PRP_FN int m_sprp2(const m128_t *n)
{
    m128_t one, two, nm1, x, base;
    uint32_t n0inv = m_n0inv(n->v[0]);
    int r = 0, dbits;
    m128_t d;

    /* R mod n by doubling 1 up 128 times: R itself does not fit in four
     * limbs, and a division to reduce it is what this file exists to avoid. */
    for (int i = 0; i < M_LIMBS; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * M_LIMBS; i++) m_dbl_mod(&one, n);
    two = one; m_dbl_mod(&two, n);              /* Montgomery form of 2 */

    nm1 = *n;
    { m128_t t = one; m_sub(&nm1, &t); }        /* (n-1) in Montgomery form */

    /* n - 1 = d * 2^r, in ordinary form */
    d = *n; d.v[0] -= 1u;                        /* n is odd, so no borrow */
    while ((d.v[0] & 1u) == 0u) {
        uint32_t carry = 0;
        for (int i = M_LIMBS - 1; i >= 0; i--) {
            uint32_t lo = d.v[i] & 1u;
            d.v[i] = (d.v[i] >> 1) | (carry << 31);
            carry = lo;
        }
        r++;
    }
    dbits = 0;
    for (int i = M_LIMBS - 1; i >= 0; i--) {
        if (d.v[i]) { uint32_t w = d.v[i]; int b = 0;
                      while (w) { w >>= 1; b++; }
                      dbits = i * 32 + b; break; }
    }

    /* left-to-right square and multiply on 2^d */
    x = one;
    base = two;
    for (int i = dbits - 1; i >= 0; i--) {
        m128_t t;
        m_mul(&t, &x, &x, n, n0inv); x = t;
        if ((d.v[i >> 5] >> (i & 31)) & 1u) { m_mul(&t, &x, &base, n, n0inv); x = t; }
    }

    if (m_cmp(&x, &one) == 0 || m_cmp(&x, &nm1) == 0) return 1;
    for (int i = 1; i < r; i++) {
        m128_t t;
        m_mul(&t, &x, &x, n, n0inv); x = t;
        if (m_cmp(&x, &nm1) == 0) return 1;
    }
    return 0;
}

/* Narrow a bn_t cofactor to four limbs. Returns 0 if it does not fit, which
 * the caller must treat as "too big to be a candidate" rather than ignore. */
PRP_FN int m_from_bn(m128_t *r, const bn_t *x)
{
    for (int i = M_LIMBS; i < BN_LIMBS; i++) if (x->v[i]) return 0;
    for (int i = 0; i < M_LIMBS; i++) r->v[i] = x->v[i];
    return 1;
}

PRP_FN double bn_to_double(const bn_t *x)
{
    double d = 0.0;
    for (int i = BN_LIMBS - 1; i >= 0; i--) d = d * 4294967296.0 + (double)x->v[i];
    return d;
}

/* ---- CADO's check_leftover_norm, sieve/las-cofactor.cpp:118 ------------- */

#define COF_REJECT_MFB   0   /* more than mfb bits                          */
#define COF_REJECT_GAP   1   /* L^k < n < B^(k+1): too few factors possible  */
#define COF_REJECT_PRIME 2   /* a probable prime larger than 2^lpb           */
#define COF_ACCEPT       3   /* carry into cofactorisation                   */
#define COF_SPLIT        4   /* already fully factored (n == 1)              */
#define COF_DEGENERATE   5   /* b == 0: not a relation at any cofactor size  */

/* `lim` is the factor-base bound for the side, as a double, because the gap
 * test below is written against doubles in CADO and matching it matters more
 * than being more careful than it. */
PRP_FN int cof_classify(const bn_t *n, int bits, unsigned lpb, unsigned mfb,
                        double lim)
{
    if (bits <= 1) return COF_SPLIT;                 /* n == 1 */
    if ((unsigned)bits > mfb) return COF_REJECT_MFB;
    if ((unsigned)bits <= lpb) return COF_ACCEPT;    /* CADO's case (a) */

    /* If n < B^(k+1) then n has at most k prime factors above B, so n <= L^k.
     * Anything above L^k in that window cannot be split at all. */
    {
        double nd = bn_to_double(n);
        double kB = lim * lim;
        for (unsigned klpb = lpb; klpb < (unsigned)bits; klpb += lpb, kB *= lim)
            if (nd < kB) return COF_REJECT_GAP;
    }

    {
        m128_t m;
        if (!m_from_bn(&m, n)) return COF_REJECT_MFB;   /* > 128 bits */
        if ((m.v[0] & 1u) == 0u) return COF_ACCEPT;     /* even: 2 was not
                                                         * divided out, so it
                                                         * is certainly not a
                                                         * prime above lpb */
        if (m_sprp2(&m)) return COF_REJECT_PRIME;
    }
    return COF_ACCEPT;
}

#endif  /* CUDA_SIEVE_PRP_CUH */
