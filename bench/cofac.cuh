/* Cofactorisation: split the residual norms into large primes, on device.
 *
 * This is the stage that stands between a survivor list and a relation count.
 * At the parity q trial division alone yields 7 relations of las's 37; the
 * other 30 are inside the cofactors, and over a 67-q band the split is 556 of
 * 3,162 -- 83% of the yield is here.
 *
 * WHAT THE WORK ACTUALLY IS, measured from 1,958,036 emitted candidates rather
 * than assumed from the mfb settings (see prototype.md):
 *
 *   rational  (lpb 31, mfb 60): 1,007,630 splits, ALL 53-60 bits, ALL 2LP.
 *                               Both factors in (2^26, 2^31].
 *   algebraic (lpb 32, mfb 92): 1,949,125 splits, bimodal --
 *                               2.6% at 55-64 bits (2LP),
 *                               97.4% at 81-92 bits (3LP, ~30-bit factors).
 *                               Nothing at 65-80 bits: that is COF_REJECT_GAP.
 *
 * So this is not a general-purpose cofactorizer. It is two narrow shapes, and
 * the arithmetic is sized to them: 2 limbs for the rational side, 3 for the
 * algebraic.
 *
 * WHY POLLARD-BRENT RHO AND NOT ECM. ECM's appeal on a GPU is that its cost is
 * data-independent, which rho's emphatically is not -- rho's iteration count is
 * geometric, and a warp runs at its slowest lane. The answer here is not to
 * abandon rho but to BOUND it: every lane gets the same iteration budget per
 * launch, and whatever has not split is requeued for another launch with a
 * different polynomial constant. Divergence inside a launch is then bounded by
 * the budget rather than by the tail of a geometric distribution, and the tail
 * becomes a scheduling question -- which is what the "reseed, requeue,
 * escalate" note in prototype.md was about. A fixed cap that DROPS work would
 * be wrong: rho is Monte Carlo, so a cofactor that resists one constant is not
 * a cofactor that cannot be split.
 *
 * PRIMALITY IS MOSTLY FREE HERE, and that is worth stating because it removes
 * most of the sprp2 calls a general cofactorizer would make. Trial division
 * already removed every factor below `lim`, so any part below lim^2 is prime
 * BY SIZE -- no test. On this job lim^2 is 2^54 on the algebraic side, so the
 * only parts that need a real primality test are those of 55 bits or more,
 * i.e. the middle of a 3LP split.
 */
#ifndef CUDA_SIEVE_COFAC_CUH
#define CUDA_SIEVE_COFAC_CUH

#include <stdint.h>

#define CF_OK         0   /* fully split, every prime within lpb */
#define CF_DEAD       1   /* a prime factor exceeds 2^lpb: never a relation */
#define CF_INCOMPLETE 2   /* out of rho budget: requeue, do not drop */
#define CF_OVERFLOW   3   /* more parts than the caller made room for */

#define CF_MAXFAC 4       /* mfb/lpb is 2 on the rational side, 3 on the
                           * algebraic; 4 leaves one slot of headroom so an
                           * unexpected shape is reported rather than silently
                           * truncated */

#if defined(__CUDACC__)
#define CF_FN __device__ __forceinline__
#else
#define CF_FN static inline
#endif

/* ---- L-limb arithmetic ------------------------------------------------- *
 *
 * Templated on the limb count so the rational side pays for 64 bits and the
 * algebraic side for 96, rather than both paying prp.cuh's fixed 128. At three
 * limbs the CIOS inner loop is 9 multiply-adds against 16 -- the modmul is the
 * whole cost of this stage, so the limb count is not a detail.
 */

template <int L> struct mz { uint32_t v[L]; };

template <int L> CF_FN int mz_cmp(const mz<L> *a, const mz<L> *b)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

template <int L> CF_FN int mz_is_zero(const mz<L> *a)
{
    uint32_t o = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN int mz_is_one(const mz<L> *a)
{
    uint32_t o = a->v[0] ^ 1u;
    #pragma unroll
    for (int i = 1; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> CF_FN uint32_t mz_sub(mz<L> *a, const mz<L> *b)
{
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow;
        a->v[i] = (uint32_t)d;
        borrow = (d >> 32) & 1u;
    }
    return (uint32_t)borrow;
}

/* a = a - b mod n, given a, b < n */
template <int L> CF_FN void mz_sub_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    if (mz_sub<L>(a, b)) {
        uint64_t c = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {
            uint64_t s = (uint64_t)a->v[i] + n->v[i] + c;
            a->v[i] = (uint32_t)s;
            c = s >> 32;
        }
    }
}

/* a = a + b mod n, given a, b < n */
template <int L> CF_FN void mz_add_mod(mz<L> *a, const mz<L> *b, const mz<L> *n)
{
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t s = (uint64_t)a->v[i] + b->v[i] + c;
        a->v[i] = (uint32_t)s;
        c = s >> 32;
    }
    if (c || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

/* a = 2a mod n */
template <int L> CF_FN void mz_dbl_mod(mz<L> *a, const mz<L> *n)
{
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint32_t hi = a->v[i] >> 31;
        a->v[i] = (a->v[i] << 1) | carry;
        carry = hi;
    }
    if (carry || mz_cmp<L>(a, n) >= 0) mz_sub<L>(a, n);
}

template <int L> CF_FN void mz_rshift1(mz<L> *a)
{
    #pragma unroll
    for (int i = 0; i < L - 1; i++)
        a->v[i] = (a->v[i] >> 1) | (a->v[i + 1] << 31);
    a->v[L - 1] >>= 1;
}

template <int L> CF_FN int mz_bits(const mz<L> *a)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i]) return i * 32 + (32 - __clz(a->v[i]));
    return 0;
}

/* -n^{-1} mod 2^32, Newton. n odd. */
CF_FN uint32_t mz_n0inv(uint32_t n0)
{
    uint32_t inv = n0;
    for (int k = 0; k < 5; k++) inv *= 2u - n0 * inv;
    return (uint32_t)(0u - inv);
}

/* r = a*b*R^-1 mod n, R = 2^(32L). CIOS. */
template <int L>
CF_FN void mz_mul(mz<L> *r, const mz<L> *a, const mz<L> *b,
                  const mz<L> *n, uint32_t n0inv)
{
    uint32_t t[L + 2];
    #pragma unroll
    for (int i = 0; i < L + 2; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < L; i++) {
        uint64_t C = 0;
        #pragma unroll
        for (int j = 0; j < L; j++) {
            uint64_t p = (uint64_t)a->v[j] * b->v[i] + t[j] + C;
            t[j] = (uint32_t)p;
            C = p >> 32;
        }
        {
            uint64_t p = (uint64_t)t[L] + C;
            t[L] = (uint32_t)p;
            t[L + 1] = (uint32_t)(p >> 32);
        }
        {
            uint32_t m = t[0] * n0inv;
            uint64_t p = (uint64_t)m * n->v[0] + t[0];
            C = p >> 32;
            #pragma unroll
            for (int j = 1; j < L; j++) {
                p = (uint64_t)m * n->v[j] + t[j] + C;
                t[j - 1] = (uint32_t)p;
                C = p >> 32;
            }
            p = (uint64_t)t[L] + C;
            t[L - 1] = (uint32_t)p;
            t[L] = t[L + 1] + (uint32_t)(p >> 32);
        }
    }
    #pragma unroll
    for (int i = 0; i < L; i++) r->v[i] = t[i];
    if (t[L] || mz_cmp<L>(r, n) >= 0) mz_sub<L>(r, n);
}

/* Binary GCD. Both arguments are destroyed. n is odd, which removes half the
 * usual bookkeeping: the common power of two is always 1. */
template <int L> CF_FN void mz_gcd(mz<L> *g, mz<L> a, mz<L> b)
{
    if (mz_is_zero<L>(&a)) { *g = b; return; }
    while ((a.v[0] & 1u) == 0u) mz_rshift1<L>(&a);
    for (;;) {
        while ((b.v[0] & 1u) == 0u) mz_rshift1<L>(&b);
        if (mz_cmp<L>(&a, &b) > 0) { mz<L> t = a; a = b; b = t; }
        mz_sub<L>(&b, &a);                       /* b >= a, so no borrow */
        if (mz_is_zero<L>(&b)) { *g = a; return; }
    }
}

/* q = n / d, EXACT. Hensel: d is odd, so d has an inverse mod 2^(32L), and for
 * an exact quotient the low 32L bits of n * d^-1 are the whole answer. This is
 * the only division in the stage, and it is a few multiplies. */
template <int L> CF_FN void mz_divexact(mz<L> *q, const mz<L> *n, const mz<L> *d)
{
    mz<L> inv, t;
    uint32_t i0 = d->v[0];
    uint32_t x = i0;
    for (int k = 0; k < 5; k++) x *= 2u - i0 * x;   /* d^-1 mod 2^32 */
    #pragma unroll
    for (int i = 0; i < L; i++) inv.v[i] = 0;
    inv.v[0] = x;
    /* lift to 2^(32L): inv <- inv * (2 - d*inv), low L limbs only */
    for (int step = 1; step < L; step <<= 1) {
        mz<L> p;
        #pragma unroll
        for (int i = 0; i < L; i++) p.v[i] = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {           /* p = d*inv mod 2^(32L) */
            uint64_t C = 0;
            #pragma unroll
            for (int j = 0; i + j < L; j++) {
                uint64_t s = (uint64_t)d->v[j] * inv.v[i] + p.v[i + j] + C;
                p.v[i + j] = (uint32_t)s;
                C = s >> 32;
            }
        }
        {   /* p = 2 - p, mod 2^(32L) */
            uint64_t borrow = 0;
            mz<L> two;
            #pragma unroll
            for (int i = 0; i < L; i++) two.v[i] = 0;
            two.v[0] = 2u;
            #pragma unroll
            for (int i = 0; i < L; i++) {
                uint64_t s = (uint64_t)two.v[i] - p.v[i] - borrow;
                p.v[i] = (uint32_t)s;
                borrow = (s >> 32) & 1u;
            }
        }
        #pragma unroll
        for (int i = 0; i < L; i++) t.v[i] = 0;
        #pragma unroll
        for (int i = 0; i < L; i++) {           /* t = inv*p mod 2^(32L) */
            uint64_t C = 0;
            #pragma unroll
            for (int j = 0; i + j < L; j++) {
                uint64_t s = (uint64_t)inv.v[j] * p.v[i] + t.v[i + j] + C;
                t.v[i + j] = (uint32_t)s;
                C = s >> 32;
            }
        }
        inv = t;
    }
    #pragma unroll
    for (int i = 0; i < L; i++) q->v[i] = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) {               /* q = n*inv mod 2^(32L) */
        uint64_t C = 0;
        #pragma unroll
        for (int j = 0; i + j < L; j++) {
            uint64_t s = (uint64_t)n->v[j] * inv.v[i] + q->v[i + j] + C;
            q->v[i + j] = (uint32_t)s;
            C = s >> 32;
        }
    }
}

/* Strong probable prime test, base 2, entirely in the Montgomery domain.
 * Same argument as prp.cuh: a composite called prime loses a relation at a rate
 * around 2^-40, it never emits one with a composite ideal. */
template <int L> CF_FN int mz_sprp2(const mz<L> *n)
{
    mz<L> one, two, nm1, x, base, d;
    uint32_t n0inv = mz_n0inv(n->v[0]);
    int r = 0, dbits;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);   /* R mod n */
    two = one; mz_dbl_mod<L>(&two, n);

    nm1 = *n; { mz<L> t = one; mz_sub<L>(&nm1, &t); }

    d = *n; d.v[0] -= 1u;
    while ((d.v[0] & 1u) == 0u) { mz_rshift1<L>(&d); r++; }
    dbits = mz_bits<L>(&d);

    x = one; base = two;
    for (int i = dbits - 1; i >= 0; i--) {
        mz<L> t;
        mz_mul<L>(&t, &x, &x, n, n0inv); x = t;
        if ((d.v[i >> 5] >> (i & 31)) & 1u) { mz_mul<L>(&t, &x, &base, n, n0inv); x = t; }
    }
    if (mz_cmp<L>(&x, &one) == 0 || mz_cmp<L>(&x, &nm1) == 0) return 1;
    for (int i = 1; i < r; i++) {
        mz<L> t;
        mz_mul<L>(&t, &x, &x, n, n0inv); x = t;
        if (mz_cmp<L>(&x, &nm1) == 0) return 1;
    }
    return 0;
}

/* ---- Pollard-Brent rho, with a hard iteration budget ------------------- *
 *
 * The whole loop runs in the Montgomery domain and never leaves it. f(y) =
 * y^2 + c maps Montgomery representatives to Montgomery representatives, and
 * the accumulated product carries a power of R that gcd cannot see: gcd(qR^k,
 * n) = gcd(q, n) because n is odd and R is a power of two. So there is no
 * conversion in or out anywhere in the hot loop.
 *
 * `budget` bounds the iteration count so a warp's cost is bounded by the
 * budget rather than by its unluckiest lane. Returning 0 means "not yet",
 * never "cannot".
 */
#define CF_BATCH 128       /* modmuls between gcds; the gcd is ~L*192 ops, so
                            * batching it under the modmuls keeps it off the
                            * critical path without delaying detection much */

/* `acc` accumulates iterations actually consumed, for the instrumentation that
 * asks where rho's time goes (dead records vs splitting ones). The re-walk in
 * the `found` tail is not counted: it is at most CF_BATCH steps against a
 * budget of tens of thousands. */
template <int L>
CF_FN int mz_rho(mz<L> *fac, const mz<L> *n, uint32_t c0, uint32_t budget,
                 uint64_t *acc)
{
    const uint32_t n0inv = mz_n0inv(n->v[0]);
    mz<L> one, y, x, ys, q, g, c;
    uint32_t steps = 0, r = 1;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);   /* R mod n */

    c = one;
    for (uint32_t k = 1; k < c0; k++) mz_add_mod<L>(&c, &one, n);  /* c0 * R */
    y = one; mz_add_mod<L>(&y, &one, n);                            /* 2 * R */
    q = one;
    x = y; ys = y;

    for (;;) {
        uint32_t k = 0;
        x = y;
        for (uint32_t i = 0; i < r; i++) {
            mz<L> t;
            mz_mul<L>(&t, &y, &y, n, n0inv);
            mz_add_mod<L>(&t, &c, n);
            y = t;
        }
        steps += r;
        while (k < r) {
            uint32_t lim = r - k; if (lim > CF_BATCH) lim = CF_BATCH;
            ys = y;
            for (uint32_t i = 0; i < lim; i++) {
                mz<L> t, d;
                mz_mul<L>(&t, &y, &y, n, n0inv);
                mz_add_mod<L>(&t, &c, n);
                y = t;
                d = x; mz_sub_mod<L>(&d, &y, n);
                if (mz_is_zero<L>(&d)) { d = one; }   /* cycle closed on x */
                mz_mul<L>(&t, &q, &d, n, n0inv);
                q = t;
            }
            k += lim; steps += lim;
            mz_gcd<L>(&g, q, *n);
            if (!mz_is_one<L>(&g)) goto found;
            if (steps >= budget) { *acc += steps; return 0; }
        }
        r <<= 1;
        if (steps >= budget) { *acc += steps; return 0; }
    }

found:
    if (mz_cmp<L>(&g, n) == 0) {
        /* the batch swallowed the factor: re-walk it one step at a time */
        y = ys;
        for (uint32_t i = 0; i < CF_BATCH; i++) {
            mz<L> t, d;
            mz_mul<L>(&t, &y, &y, n, n0inv);
            mz_add_mod<L>(&t, &c, n);
            y = t;
            d = x; mz_sub_mod<L>(&d, &y, n);
            if (mz_is_zero<L>(&d)) { *acc += steps; return 0; }  /* degenerate */
            mz_gcd<L>(&g, d, *n);
            if (!mz_is_one<L>(&g)) break;
        }
        if (mz_is_one<L>(&g) || mz_cmp<L>(&g, n) == 0) { *acc += steps; return 0; }
    }
    *fac = g;
    *acc += steps;
    return 1;
}

/* ---- ECM, stage 1 only -------------------------------------------------- *
 *
 * Montgomery curves By^2 = x^3 + Ax^2 + x in XZ coordinates, which is the form
 * whose differential addition needs no y and no inversion.
 *
 * The curve constant is kept PROJECTIVE, A24 = (an : ad). Suyama's
 * parameterisation puts A24 = (v-u)^3 (3u+v) / (16 u^3 v), and carrying that as
 * a fraction costs one extra multiply per doubling -- 12 modmuls per ladder bit
 * against 11 -- while removing a modular inversion that would otherwise be paid
 * once per curve per record. At tens of curves per record that trade is clearly
 * the right way round, and it keeps a binary extended GCD out of the file.
 *
 * Stage 2 is deliberately absent. It is also where most of ECM's probability
 * per unit cost lives, so the measurement this produces is a floor on ECM, not
 * a verdict on it -- see the write-up.
 */

/* The stage-1 exponent lcm(1..B1), held as its prime powers rather than as one
 * big integer: each is at most B1, so the ladder for it is a dozen bits and the
 * whole schedule is an array of uint32 instead of a 1.4*B1-bit number the device
 * would have to store and shift. Returns the count. */
static uint32_t cf_ecm_plan(uint32_t b1, uint32_t **out)
{
    uint32_t n = 0, cap = 0, *s = NULL;
    unsigned char *sieve;
    if (b1 < 2) { *out = NULL; return 0; }
    sieve = (unsigned char *)calloc(b1 + 1, 1);
    if (!sieve) { *out = NULL; return 0; }
    for (uint32_t p = 2; p <= b1; p++) {
        if (sieve[p]) continue;
        for (uint64_t q = (uint64_t)p * p; q <= b1; q += p) sieve[q] = 1;
        {   /* the largest power of p that still fits under B1 */
            uint64_t pe = p;
            while (pe * p <= b1) pe *= p;
            if (n == cap) {
                cap = cap ? cap * 2 : 256;
                s = (uint32_t *)realloc(s, (size_t)cap * 4);
                if (!s) { free(sieve); *out = NULL; return 0; }
            }
            s[n++] = (uint32_t)pe;
        }
    }
    free(sieve);
    *out = s;
    return n;
}

template <int L> struct mpt { mz<L> X, Z; };

/* r = k * one, for a small k, by double-and-add on the Montgomery form of 1. */
template <int L>
CF_FN void mz_small(mz<L> *r, uint32_t k, const mz<L> *one, const mz<L> *n)
{
    int b = 31;
    #pragma unroll
    for (int i = 0; i < L; i++) r->v[i] = 0;
    if (!k) return;
    while (b > 0 && !((k >> b) & 1u)) b--;
    *r = *one;
    for (b--; b >= 0; b--) {
        mz_dbl_mod<L>(r, n);
        if ((k >> b) & 1u) mz_add_mod<L>(r, one, n);
    }
}

/* 2P. r must not alias p. */
template <int L>
CF_FN void x_dbl(mpt<L> *r, const mpt<L> *p, const mz<L> *an, const mz<L> *ad,
                 const mz<L> *n, uint32_t n0inv)
{
    mz<L> t1, t2, s1, s2, d, w, z;
    t1 = p->X; mz_add_mod<L>(&t1, &p->Z, n);
    t2 = p->X; mz_sub_mod<L>(&t2, &p->Z, n);
    mz_mul<L>(&s1, &t1, &t1, n, n0inv);            /* (X+Z)^2 */
    mz_mul<L>(&s2, &t2, &t2, n, n0inv);            /* (X-Z)^2 */
    d = s1; mz_sub_mod<L>(&d, &s2, n);             /* 4XZ     */
    mz_mul<L>(&t1, ad, &s1, n, n0inv);
    mz_mul<L>(&r->X, &t1, &s2, n, n0inv);          /* ad*(X+Z)^2*(X-Z)^2 */
    mz_mul<L>(&t2, ad, &s2, n, n0inv);
    mz_mul<L>(&w, an, &d, n, n0inv);
    z = t2; mz_add_mod<L>(&z, &w, n);
    mz_mul<L>(&r->Z, &d, &z, n, n0inv);
}

/* P+Q given the difference D = P-Q. r must not alias p, q or d. */
template <int L>
CF_FN void x_add(mpt<L> *r, const mpt<L> *p, const mpt<L> *q, const mpt<L> *dif,
                 const mz<L> *n, uint32_t n0inv)
{
    mz<L> a, b, c, e, t1, t2, s, t;
    a = p->X; mz_add_mod<L>(&a, &p->Z, n);
    b = p->X; mz_sub_mod<L>(&b, &p->Z, n);
    c = q->X; mz_add_mod<L>(&c, &q->Z, n);
    e = q->X; mz_sub_mod<L>(&e, &q->Z, n);
    mz_mul<L>(&t1, &b, &c, n, n0inv);              /* (XP-ZP)(XQ+ZQ) */
    mz_mul<L>(&t2, &a, &e, n, n0inv);              /* (XP+ZP)(XQ-ZQ) */
    s = t1; mz_add_mod<L>(&s, &t2, n);
    t = t1; mz_sub_mod<L>(&t, &t2, n);
    mz_mul<L>(&a, &s, &s, n, n0inv);
    mz_mul<L>(&b, &t, &t, n, n0inv);
    mz_mul<L>(&r->X, &dif->Z, &a, n, n0inv);
    mz_mul<L>(&r->Z, &dif->X, &b, n, n0inv);
}

/* P = [k]P by the Montgomery ladder. k is a prime power <= B1, so this is a
 * dozen bits at most and the ladder's constant-time shape costs nothing. */
template <int L>
CF_FN void x_mul(mpt<L> *p, uint32_t k, const mz<L> *an, const mz<L> *ad,
                 const mz<L> *n, uint32_t n0inv, uint64_t *bits)
{
    mpt<L> r0 = *p, r1, t;
    int b = 31;
    if (k < 2) { if (!k) { for (int i = 0; i < L; i++) { p->X.v[i] = 0; p->Z.v[i] = 0; } } return; }
    while (b > 0 && !((k >> b) & 1u)) b--;
    x_dbl<L>(&r1, p, an, ad, n, n0inv);
    for (b--; b >= 0; b--) {
        if ((k >> b) & 1u) {
            x_add<L>(&t, &r1, &r0, p, n, n0inv); r0 = t;
            x_dbl<L>(&t, &r1, an, ad, n, n0inv);  r1 = t;
        } else {
            x_add<L>(&t, &r1, &r0, p, n, n0inv); r1 = t;
            x_dbl<L>(&t, &r0, an, ad, n, n0inv);  r0 = t;
        }
        (*bits)++;
    }
    *p = r0;
}

/* One curve of stage 1. `s` holds the prime powers p^e <= B1, ascending.
 * Returns 1 and a nontrivial factor when gcd(Z, n) splits n. */
template <int L>
CF_FN int mz_ecm(mz<L> *fac, const mz<L> *n, uint32_t sigma,
                 const uint32_t *__restrict s, uint32_t ns, uint64_t *bits)
{
    const uint32_t n0inv = mz_n0inv(n->v[0]);
    mz<L> one, sig, u, v, t, w, an, ad, g;
    mpt<L> P;

    #pragma unroll
    for (int i = 0; i < L; i++) one.v[i] = 0;
    one.v[0] = 1;
    for (int i = 0; i < 32 * L; i++) mz_dbl_mod<L>(&one, n);      /* R mod n */

    mz_small<L>(&sig, sigma, &one, n);
    mz_mul<L>(&u, &sig, &sig, n, n0inv);                          /* sigma^2 */
    mz_small<L>(&t, 5, &one, n);
    mz_sub_mod<L>(&u, &t, n);                                     /* u = s^2-5 */
    v = sig; mz_dbl_mod<L>(&v, n); mz_dbl_mod<L>(&v, n);          /* v = 4s   */
    if (mz_is_zero<L>(&u) || mz_is_zero<L>(&v)) return 0;

    mz_mul<L>(&t, &u, &u, n, n0inv);
    mz_mul<L>(&P.X, &t, &u, n, n0inv);                            /* x0 = u^3 */
    mz_mul<L>(&t, &v, &v, n, n0inv);
    mz_mul<L>(&P.Z, &t, &v, n, n0inv);                            /* z0 = v^3 */

    t = v; mz_sub_mod<L>(&t, &u, n);                              /* v-u      */
    mz_mul<L>(&w, &t, &t, n, n0inv);
    mz_mul<L>(&an, &w, &t, n, n0inv);                             /* (v-u)^3  */
    mz_small<L>(&t, 3, &one, n);
    mz_mul<L>(&w, &t, &u, n, n0inv);
    mz_add_mod<L>(&w, &v, n);                                     /* 3u+v     */
    mz_mul<L>(&t, &an, &w, n, n0inv); an = t;                     /* numerator*/

    mz_small<L>(&t, 16, &one, n);
    mz_mul<L>(&w, &t, &P.X, n, n0inv);                            /* 16*u^3   */
    mz_mul<L>(&ad, &w, &v, n, n0inv);                             /* *v       */
    if (mz_is_zero<L>(&ad)) return 0;

    for (uint32_t i = 0; i < ns; i++)
        x_mul<L>(&P, s[i], &an, &ad, n, n0inv, bits);

    mz_gcd<L>(&g, P.Z, *n);
    if (mz_is_one<L>(&g) || mz_cmp<L>(&g, n) == 0) return 0;
    *fac = g;
    return 1;
}

/* ---- full split of one cofactor ---------------------------------------- *
 *
 * `lim2` is lim^2 for the side. Anything below it is PRIME by construction --
 * trial division removed every factor below lim, so a composite needs two of
 * them -- which is why this makes far fewer primality calls than a general
 * cofactorizer would. On this job that threshold is 2^54 on the algebraic side,
 * so only the middle part of a 3LP split is ever tested.
 */
/* METHOD 0 is Pollard-Brent rho and `budget` is an iteration count; METHOD 1 is
 * ECM stage 1 and `budget` is a CURVE count. Both are bounded per launch and
 * neither turns an exhausted budget into a proof, so the requeue structure
 * around them is unchanged. */
template <int L, int METHOD>
CF_FN int mz_split(const mz<L> *n0, const mz<L> *lim2, uint32_t lpb,
                   uint32_t c0, uint32_t budget, uint32_t *out, int *nout,
                   uint64_t *acc, const uint32_t *__restrict s, uint32_t ns)
{
    mz<L> st[CF_MAXFAC];
    int sp = 0, nf = 0;
    st[sp++] = *n0;
    while (sp) {
        mz<L> m = st[--sp], g, h;
        if (mz_is_one<L>(&m)) continue;
        if (mz_cmp<L>(&m, lim2) < 0) {              /* prime by size */
            if ((uint32_t)mz_bits<L>(&m) > lpb) return CF_DEAD;
            if (nf >= CF_MAXFAC) return CF_OVERFLOW;
            out[nf++] = m.v[0];
            continue;
        }
        if (mz_sprp2<L>(&m)) return CF_DEAD;        /* prime, and >= lim2 > 2^lpb */
        if (METHOD == 0) {
            if (!mz_rho<L>(&g, &m, c0, budget, acc)) return CF_INCOMPLETE;
        } else {
            uint32_t cv = 0;
            for (; cv < budget; cv++)
                if (mz_ecm<L>(&g, &m, c0 * 1000u + cv + 6u, s, ns, acc)) break;
            if (cv == budget) return CF_INCOMPLETE;
        }
        mz_divexact<L>(&h, &m, &g);
        if (sp + 2 > CF_MAXFAC) return CF_OVERFLOW;
        st[sp++] = g; st[sp++] = h;
    }
    *nout = nf;
    return CF_OK;
}

#if defined(__CUDACC__)

/* ---- the kernel -------------------------------------------------------- *
 *
 * One thread per job, and a job that runs out of budget keeps its slot: the
 * host relaunches with a different rho constant and a larger budget, and a
 * thread whose status is already resolved falls straight through. Compacting
 * between rounds would save the fall-through, but the rounds after the first
 * hold a few percent of the jobs, so the scan is cheap and the compaction is
 * not yet worth its own code. That is a measurement to revisit, not a
 * principle.
 */
/* SELECT runs the launch over a compacted index list instead of every slot.
 * It matters more than a fall-through normally would: a lane that is already
 * resolved costs nothing itself, but its warp still runs for as long as its
 * slowest LIVE lane, so a warp holding one live lane and 31 dead ones pays a
 * full rho budget for one job. Packing the live lanes together is what turns
 * that back into 32 jobs per budget. `njp` is read on the device so the grid
 * never has to be sized from a host-visible count. */
template <int L, int SELECT, int METHOD>
__global__ void k_cofac(const mz<L> *__restrict n, mz<L> lim2, uint32_t lpb,
                        uint32_t c0, uint32_t budget, uint32_t nj,
                        const uint32_t *__restrict sel,
                        const uint32_t *__restrict njp,
                        uint8_t *__restrict status,
                        uint32_t *__restrict fac, uint8_t *__restrict nfac,
                        unsigned long long *__restrict iters,
                        const uint32_t *__restrict s, uint32_t ns)
{
    const uint32_t cnt = SELECT ? *njp : nj;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += stride) {
        const uint32_t t = SELECT ? sel[i] : i;
        uint32_t o[CF_MAXFAC];
        uint64_t acc = 0;
        int k = 0, st;
        mz<L> v;
        if (status[t] != CF_INCOMPLETE) continue;
        v = n[t];
        st = mz_split<L, METHOD>(&v, &lim2, lpb, c0, budget, o, &k, &acc, s, ns);
        if (iters) iters[t] += (unsigned long long)acc;   /* summed over rounds */
        status[t] = (uint8_t)st;
        if (st == CF_OK) {
            nfac[t] = (uint8_t)k;
            for (int i = 0; i < k; i++) fac[(size_t)t * CF_MAXFAC + i] = o[i];
        }
    }
}


/* ---- the cross-q device queue ------------------------------------------ *
 *
 * A special-q produces about 1,956 joint candidates. That is nothing for this
 * GPU -- 73,728 threads at the pipeline's launch geometry -- so cofactorising
 * per special-q would run the rho kernel at 3% occupancy and pay the requeue
 * rounds on an almost empty grid. The queue exists to fix that: candidates
 * accumulate across special-q and are flushed as one batch of ~131,072, which
 * is ~67 q worth and fills the machine.
 *
 * It is also what removes the file round trip. `--candidates` writes 295 MB
 * over a 1,000-q band and the host emission of it costs ~3 ms/q; with the queue
 * resident, only the ~2% of records that become relations are ever read back.
 */

#define CQ_FLUSH  131072u        /* candidates per flush, ~67 special-q */

typedef struct {
    uint32_t cap, n, rcap;
    mz<2>    *d_c0;   mz<3>   *d_c1;      /* the two cofactors, narrowed */
    uint8_t  *d_st0,  *d_st1;             /* CF_* per side                */
    uint32_t *d_sm0,  *d_sm1;             /* residual when no split needed */
    int64_t  *d_a,    *d_b;
    uint32_t *d_f0,   *d_f1;              /* cap * TD_FMAX                */
    uint8_t  *d_fn0,  *d_fn1;
    uint32_t *d_sp0,  *d_sp1;             /* cap * CF_MAXFAC split primes */
    uint8_t  *d_nsp0, *d_nsp1;
    uint32_t *d_flag, *d_off, *d_bsum, *d_idx, *d_nrel;
    uint32_t *d_sel, *d_nsel;             /* compacted job list per round  */
    uint32_t *d_s, ns, ecm_curves; int ecm;   /* ECM stage-1 prime powers  */
    /* pinned mirrors, sized for the relations only */
    int64_t  *h_a, *h_b;
    uint32_t *h_f0, *h_f1, *h_sp0, *h_sp1;
    uint8_t  *h_fn0, *h_fn1, *h_nsp0, *h_nsp1;
    double ms_rat, ms_alg, ms_host;
    unsigned long long nseen, nrel, ndead, nstuck;
} cofq_t;

/* Append one special-q's joint candidates. Everything the relation will need
 * is copied in, because the per-q arrays are overwritten by the next q. */
__global__ void k_cof_enqueue(const bn_t *__restrict cof0, const uint8_t *__restrict bits0,
                              const bn_t *__restrict cof1, const uint8_t *__restrict bits1,
                              const int64_t *__restrict a, const int64_t *__restrict b,
                              const uint32_t *__restrict f0, const uint32_t *__restrict fn0,
                              const uint32_t *__restrict f1, const uint32_t *__restrict fn1,
                              uint32_t n, uint32_t base, uint32_t lpb0, uint32_t lpb1,
                              cofq_t Q)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const uint32_t d = base + t;
        Q.d_a[d] = a[t]; Q.d_b[d] = b[t];
        Q.d_nsp0[d] = 0; Q.d_nsp1[d] = 0;
        {
            uint32_t c = fn0[t]; if (c > TD_FMAX) c = TD_FMAX;
            Q.d_fn0[d] = (uint8_t)c;
            for (uint32_t z = 0; z < c; z++)
                Q.d_f0[(size_t)d * TD_FMAX + z] = f0[(size_t)t * TD_FMAX + z];
            c = fn1[t]; if (c > TD_FMAX) c = TD_FMAX;
            Q.d_fn1[d] = (uint8_t)c;
            for (uint32_t z = 0; z < c; z++)
                Q.d_f1[(size_t)d * TD_FMAX + z] = f1[(size_t)t * TD_FMAX + z];
        }
        /* A side within lpb needs no splitting, but its residual is still a
         * prime of the relation whenever it is not 1. */
        {
            bn_t v = cof0[t];
            if ((uint32_t)bits0[t] > lpb0) {
                Q.d_c0[d].v[0] = v.v[0]; Q.d_c0[d].v[1] = v.v[1];
                Q.d_sm0[d] = 0; Q.d_st0[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm0[d] = (bits0[t] > 1) ? v.v[0] : 0u;
                Q.d_st0[d] = CF_OK;
            }
            v = cof1[t];
            if ((uint32_t)bits1[t] > lpb1) {
                Q.d_c1[d].v[0] = v.v[0]; Q.d_c1[d].v[1] = v.v[1]; Q.d_c1[d].v[2] = v.v[2];
                Q.d_sm1[d] = 0; Q.d_st1[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm1[d] = (bits1[t] > 1) ? v.v[0] : 0u;
                Q.d_st1[d] = CF_OK;
            }
        }
    }
}

/* The class-aware gate, on device. A record whose rational side did not split
 * has its algebraic status forced away from CF_INCOMPLETE, so k_cofac falls
 * straight through it instead of spending a full rho budget on a record that
 * can never be a relation. */
__global__ void k_cof_gate(uint32_t n, uint8_t *__restrict st0, uint8_t *__restrict st1)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
        if (st0[t] != CF_OK && st1[t] == CF_INCOMPLETE) st1[t] = CF_DEAD;
}

/* Flag the jobs a round still has to run, and pack their indices. Rebuilt
 * before every round, so round r+1 sees only what round r left unresolved --
 * which on the algebraic side is where the requeue cost actually lives. */
__global__ void k_cof_selflags(uint32_t n, const uint8_t *__restrict st,
                               uint32_t *__restrict flag)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
        flag[t] = (st[t] == CF_INCOMPLETE) ? 1u : 0u;
}

__global__ void k_cof_selscatter(uint32_t n, const uint32_t *__restrict flag,
                                 const uint32_t *__restrict off,
                                 uint32_t *__restrict sel, uint32_t *__restrict nsel)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        if (flag[t]) sel[off[t]] = t;
        if (t == n - 1) *nsel = off[t] + flag[t];
    }
}

__global__ void k_rel_flags(uint32_t n, const uint8_t *__restrict st0,
                            const uint8_t *__restrict st1, uint32_t *__restrict flag)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
        flag[t] = (st0[t] == CF_OK && st1[t] == CF_OK) ? 1u : 0u;
}

/* Gather the relations into the front of the queue's own arrays. Only these
 * rows are read back -- about 2% of the batch. */
__global__ void k_rel_gather(uint32_t n, const uint32_t *__restrict flag,
                             const uint32_t *__restrict off, uint32_t cap,
                             uint32_t *__restrict idx, uint32_t *__restrict nrel)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        if (flag[t] && off[t] < cap) idx[off[t]] = t;
        if (t == n - 1) *nrel = off[t] + flag[t];
    }
}

__global__ void k_rel_pack(uint32_t nr, const uint32_t *__restrict idx, cofq_t Q,
                           int64_t *__restrict oa, int64_t *__restrict ob,
                           uint32_t *__restrict of0, uint8_t *__restrict ofn0,
                           uint32_t *__restrict of1, uint8_t *__restrict ofn1,
                           uint32_t *__restrict osp0, uint8_t *__restrict onsp0,
                           uint32_t *__restrict osp1, uint8_t *__restrict onsp1)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < nr; t += stride) {
        const uint32_t r = idx[t];
        oa[t] = Q.d_a[r]; ob[t] = Q.d_b[r];
        ofn0[t] = Q.d_fn0[r]; ofn1[t] = Q.d_fn1[r];
        for (uint32_t z = 0; z < Q.d_fn0[r]; z++)
            of0[(size_t)t * TD_FMAX + z] = Q.d_f0[(size_t)r * TD_FMAX + z];
        for (uint32_t z = 0; z < Q.d_fn1[r]; z++)
            of1[(size_t)t * TD_FMAX + z] = Q.d_f1[(size_t)r * TD_FMAX + z];
        /* the split primes, plus the residual of a side that needed no split */
        {
            uint32_t k = Q.d_nsp0[r];
            for (uint32_t z = 0; z < k; z++)
                osp0[(size_t)t * CF_MAXFAC + z] = Q.d_sp0[(size_t)r * CF_MAXFAC + z];
            if (!k && Q.d_sm0[r]) { osp0[(size_t)t * CF_MAXFAC] = Q.d_sm0[r]; k = 1; }
            onsp0[t] = (uint8_t)k;
            k = Q.d_nsp1[r];
            for (uint32_t z = 0; z < k; z++)
                osp1[(size_t)t * CF_MAXFAC + z] = Q.d_sp1[(size_t)r * CF_MAXFAC + z];
            if (!k && Q.d_sm1[r]) { osp1[(size_t)t * CF_MAXFAC] = Q.d_sm1[r]; k = 1; }
            onsp1[t] = (uint8_t)k;
        }
    }
}

/* ---- the queue, host side ---------------------------------------------- */

static int cf_u32cmp(const void *x, const void *y)
{
    uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y;
    return a < b ? -1 : (a > b);
}


typedef struct {
    int64_t  *a, *b;
    uint32_t *f0, *f1, *sp0, *sp1;
    uint8_t  *fn0, *fn1, *nsp0, *nsp1;
    int64_t  *d_a, *d_b;
    uint32_t *d_f0, *d_f1, *d_sp0, *d_sp1;
    uint8_t  *d_fn0, *d_fn1, *d_nsp0, *d_nsp1;
} cofq_out_t;

static void cofq_free(cofq_t *Q, cofq_out_t *O)
{
    cudaFree(Q->d_c0); cudaFree(Q->d_c1); cudaFree(Q->d_st0); cudaFree(Q->d_st1);
    cudaFree(Q->d_sm0); cudaFree(Q->d_sm1); cudaFree(Q->d_a); cudaFree(Q->d_b);
    cudaFree(Q->d_f0); cudaFree(Q->d_f1); cudaFree(Q->d_fn0); cudaFree(Q->d_fn1);
    cudaFree(Q->d_sp0); cudaFree(Q->d_sp1); cudaFree(Q->d_nsp0); cudaFree(Q->d_nsp1);
    cudaFree(Q->d_flag); cudaFree(Q->d_off); cudaFree(Q->d_bsum);
    cudaFree(Q->d_idx); cudaFree(Q->d_nrel);
    cudaFree(Q->d_sel); cudaFree(Q->d_nsel); cudaFree(Q->d_s);
    cudaFree(O->d_a); cudaFree(O->d_b); cudaFree(O->d_f0); cudaFree(O->d_f1);
    cudaFree(O->d_fn0); cudaFree(O->d_fn1); cudaFree(O->d_sp0); cudaFree(O->d_sp1);
    cudaFree(O->d_nsp0); cudaFree(O->d_nsp1);
    if (O->a) cudaFreeHost(O->a);
    if (O->b) cudaFreeHost(O->b);
    if (O->f0) cudaFreeHost(O->f0);
    if (O->f1) cudaFreeHost(O->f1);
    if (O->fn0) cudaFreeHost(O->fn0);
    if (O->fn1) cudaFreeHost(O->fn1);
    if (O->sp0) cudaFreeHost(O->sp0);
    if (O->sp1) cudaFreeHost(O->sp1);
    if (O->nsp0) cudaFreeHost(O->nsp0);
    if (O->nsp1) cudaFreeHost(O->nsp1);
    memset(Q, 0, sizeof(*Q)); memset(O, 0, sizeof(*O));
}

/* `ecm`/`ecm_b1`/`ecm_curves` come from the caller's configuration. They used
 * to be left at the memset's zero, which silently pinned the inline path to rho
 * no matter what the CLI asked for -- the two schedulers had diverged so that
 * only the standalone one could run ECM at all. */
static int cofq_init(cofq_t *Q, cofq_out_t *O, uint32_t cap,
                     int ecm, uint32_t ecm_b1, uint32_t ecm_curves)
{
    const uint32_t nb = (cap + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    memset(Q, 0, sizeof(*Q)); memset(O, 0, sizeof(*O));
    Q->cap = cap;
    Q->ecm = ecm; Q->ecm_curves = ecm_curves;
    if (ecm) {
        uint32_t *h_s = NULL;
        Q->ns = cf_ecm_plan(ecm_b1, &h_s);
        if (!Q->ns) { fprintf(stderr, "  cofac queue: empty ECM plan for B1=%u\n", ecm_b1); return -1; }
        CK(cudaMalloc(&Q->d_s, (size_t)Q->ns * 4));
        CK(cudaMemcpy(Q->d_s, h_s, (size_t)Q->ns * 4, cudaMemcpyHostToDevice));
        free(h_s);
        printf("  cofactor queue: ECM stage 1, B1 = %u, %u prime powers,"
               " %u curves per round\n", ecm_b1, Q->ns, ecm_curves);
    }
    Q->rcap = cap / 4;              /* relations run ~2% of candidates */
    CK(cudaMalloc(&Q->d_c0, (size_t)cap * sizeof(mz<2>)));
    CK(cudaMalloc(&Q->d_c1, (size_t)cap * sizeof(mz<3>)));
    CK(cudaMalloc(&Q->d_st0, cap)); CK(cudaMalloc(&Q->d_st1, cap));
    CK(cudaMalloc(&Q->d_sm0, (size_t)cap * 4)); CK(cudaMalloc(&Q->d_sm1, (size_t)cap * 4));
    CK(cudaMalloc(&Q->d_a, (size_t)cap * 8)); CK(cudaMalloc(&Q->d_b, (size_t)cap * 8));
    CK(cudaMalloc(&Q->d_f0, (size_t)cap * TD_FMAX * 4));
    CK(cudaMalloc(&Q->d_f1, (size_t)cap * TD_FMAX * 4));
    CK(cudaMalloc(&Q->d_fn0, cap)); CK(cudaMalloc(&Q->d_fn1, cap));
    CK(cudaMalloc(&Q->d_sp0, (size_t)cap * CF_MAXFAC * 4));
    CK(cudaMalloc(&Q->d_sp1, (size_t)cap * CF_MAXFAC * 4));
    CK(cudaMalloc(&Q->d_nsp0, cap)); CK(cudaMalloc(&Q->d_nsp1, cap));
    CK(cudaMalloc(&Q->d_flag, (size_t)cap * 4));
    CK(cudaMalloc(&Q->d_off, (size_t)cap * 4));
    CK(cudaMalloc(&Q->d_bsum, (size_t)nb * 4));
    CK(cudaMalloc(&Q->d_idx, (size_t)Q->rcap * 4));
    CK(cudaMalloc(&Q->d_nrel, 4));
    CK(cudaMalloc(&Q->d_sel, (size_t)cap * 4));
    CK(cudaMalloc(&Q->d_nsel, 4));
    CK(cudaMalloc(&O->d_a, (size_t)Q->rcap * 8)); CK(cudaMalloc(&O->d_b, (size_t)Q->rcap * 8));
    CK(cudaMalloc(&O->d_f0, (size_t)Q->rcap * TD_FMAX * 4));
    CK(cudaMalloc(&O->d_f1, (size_t)Q->rcap * TD_FMAX * 4));
    CK(cudaMalloc(&O->d_fn0, Q->rcap)); CK(cudaMalloc(&O->d_fn1, Q->rcap));
    CK(cudaMalloc(&O->d_sp0, (size_t)Q->rcap * CF_MAXFAC * 4));
    CK(cudaMalloc(&O->d_sp1, (size_t)Q->rcap * CF_MAXFAC * 4));
    CK(cudaMalloc(&O->d_nsp0, Q->rcap)); CK(cudaMalloc(&O->d_nsp1, Q->rcap));
    CK(cudaHostAlloc((void **)&O->a, (size_t)Q->rcap * 8, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->b, (size_t)Q->rcap * 8, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->f0, (size_t)Q->rcap * TD_FMAX * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->f1, (size_t)Q->rcap * TD_FMAX * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->fn0, Q->rcap, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->fn1, Q->rcap, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->sp0, (size_t)Q->rcap * CF_MAXFAC * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->sp1, (size_t)Q->rcap * CF_MAXFAC * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->nsp0, Q->rcap, cudaHostAllocDefault));
    CK(cudaHostAlloc((void **)&O->nsp1, Q->rcap, cudaHostAllocDefault));
    return 0;
}

static void cq_emit_side(FILE *o, const uint32_t *f, int nf,
                         const uint32_t *sp, int nsp)
{
    uint32_t v[TD_FMAX + CF_MAXFAC];
    int n = 0;
    for (int i = 0; i < nf && n < TD_FMAX + CF_MAXFAC; i++) v[n++] = f[i];
    for (int i = 0; i < nsp && n < TD_FMAX + CF_MAXFAC; i++) v[n++] = sp[i];
    qsort(v, n, sizeof v[0], cf_u32cmp);
    for (int i = 0; i < n; i++) fprintf(o, "%s%x", i ? "," : "", v[i]);
}

/* Split everything queued, then write only the relations. */
static int cofq_flush(cofq_t *Q, cofq_out_t *O, uint64_t lim0, uint32_t lpb0,
                      uint64_t lim1, uint32_t lpb1, int rounds, uint32_t budget,
                      int blocks, int threads, FILE *fo)
{
    const uint32_t n = Q->n;
    const uint32_t nb = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    mz<2> l0; mz<3> l1;
    uint32_t nr = 0;
    cudaEvent_t e0, e1, e2;
    float t0 = 0, t1 = 0;
    double h0;
    if (!n) return 0;
    { unsigned __int128 v = (unsigned __int128)lim0 * lim0;
      for (int i = 0; i < 2; i++) { l0.v[i] = (uint32_t)v; v >>= 32; } }
    { unsigned __int128 v = (unsigned __int128)lim1 * lim1;
      for (int i = 0; i < 3; i++) { l1.v[i] = (uint32_t)v; v >>= 32; } }

    /* Compact before every round. The scan is three small kernels over 131,072
     * slots -- microseconds against a stage that runs for tens of milliseconds --
     * and it is the ordered prefix scan, not an atomic append, so the job order
     * is a function of the status array alone and the output stays reproducible
     * byte for byte. */
#define CQ_COMPACT(ST)                                                          \
    do {                                                                        \
        k_cof_selflags<<<blocks, threads>>>(n, (ST), Q->d_flag);                \
        k_scan_pass1<<<nb, TD_SCAN_BLK>>>(Q->d_flag, n, Q->d_off, Q->d_bsum);   \
        k_scan_pass2<<<1, 1024>>>(Q->d_bsum, nb);                               \
        k_scan_pass3<<<nb, TD_SCAN_BLK>>>(Q->d_off, n, Q->d_bsum);              \
        k_cof_selscatter<<<blocks, threads>>>(n, Q->d_flag, Q->d_off,           \
                                              Q->d_sel, Q->d_nsel);            \
    } while (0)

    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e0);
    for (int r = 0; r < rounds; r++) {
        CQ_COMPACT(Q->d_st0);
        if (Q->ecm)
            k_cofac<2, 1, 1><<<blocks, threads>>>(Q->d_c0, l0, lpb0, (uint32_t)(r + 1),
                                                  Q->ecm_curves, n, Q->d_sel, Q->d_nsel,
                                                  Q->d_st0, Q->d_sp0, Q->d_nsp0, NULL,
                                                  Q->d_s, Q->ns);
        else
            k_cofac<2, 1, 0><<<blocks, threads>>>(Q->d_c0, l0, lpb0, (uint32_t)(r + 1),
                                                  budget << r, n, Q->d_sel, Q->d_nsel,
                                                  Q->d_st0, Q->d_sp0, Q->d_nsp0, NULL,
                                                  NULL, 0);
    }
    k_cof_gate<<<blocks, threads>>>(n, Q->d_st0, Q->d_st1);
    cudaEventRecord(e1);
    for (int r = 0; r < rounds; r++) {
        CQ_COMPACT(Q->d_st1);
        if (Q->ecm)
            k_cofac<3, 1, 1><<<blocks, threads>>>(Q->d_c1, l1, lpb1, (uint32_t)(r + 1),
                                                  Q->ecm_curves, n, Q->d_sel, Q->d_nsel,
                                                  Q->d_st1, Q->d_sp1, Q->d_nsp1, NULL,
                                                  Q->d_s, Q->ns);
        else
            k_cofac<3, 1, 0><<<blocks, threads>>>(Q->d_c1, l1, lpb1, (uint32_t)(r + 1),
                                                  budget << r, n, Q->d_sel, Q->d_nsel,
                                                  Q->d_st1, Q->d_sp1, Q->d_nsp1, NULL,
                                                  NULL, 0);
    }
    cudaEventRecord(e2);
#undef CQ_COMPACT

    k_rel_flags<<<blocks, threads>>>(n, Q->d_st0, Q->d_st1, Q->d_flag);
    k_scan_pass1<<<nb, TD_SCAN_BLK>>>(Q->d_flag, n, Q->d_off, Q->d_bsum);
    k_scan_pass2<<<1, 1024>>>(Q->d_bsum, nb);
    k_scan_pass3<<<nb, TD_SCAN_BLK>>>(Q->d_off, n, Q->d_bsum);
    k_rel_gather<<<blocks, threads>>>(n, Q->d_flag, Q->d_off, Q->rcap,
                                      Q->d_idx, Q->d_nrel);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEventElapsedTime(&t0, e0, e1); cudaEventElapsedTime(&t1, e1, e2);
    Q->ms_rat += t0; Q->ms_alg += t1;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);

    CK(cudaMemcpy(&nr, Q->d_nrel, 4, cudaMemcpyDeviceToHost));
    if (nr > Q->rcap) {
        fprintf(stderr, "  cofac queue: %u relations in a flush of %u exceeds"
                " the %u readback slots\n", nr, n, Q->rcap);
        return -1;
    }
    Q->nseen += n; Q->nrel += nr;
    if (!nr) { Q->n = 0; return 0; }

    k_rel_pack<<<blocks, threads>>>(nr, Q->d_idx, *Q, O->d_a, O->d_b,
                                    O->d_f0, O->d_fn0, O->d_f1, O->d_fn1,
                                    O->d_sp0, O->d_nsp0, O->d_sp1, O->d_nsp1);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    h0 = host_ms();
    CK(cudaMemcpy(O->a, O->d_a, (size_t)nr * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->b, O->d_b, (size_t)nr * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->f0, O->d_f0, (size_t)nr * TD_FMAX * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->f1, O->d_f1, (size_t)nr * TD_FMAX * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->fn0, O->d_fn0, nr, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->fn1, O->d_fn1, nr, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->sp0, O->d_sp0, (size_t)nr * CF_MAXFAC * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->sp1, O->d_sp1, (size_t)nr * CF_MAXFAC * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->nsp0, O->d_nsp0, nr, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(O->nsp1, O->d_nsp1, nr, cudaMemcpyDeviceToHost));
    if (fo)
        for (uint32_t k = 0; k < nr; k++) {
            int64_t a = O->a[k], b = O->b[k];
            if (b < 0) { a = -a; b = -b; }
            fprintf(fo, "%lld,%lld:", (long long)a, (long long)b);
            cq_emit_side(fo, O->f0 + (size_t)k * TD_FMAX, O->fn0[k],
                         O->sp0 + (size_t)k * CF_MAXFAC, O->nsp0[k]);
            fputc(':', fo);
            cq_emit_side(fo, O->f1 + (size_t)k * TD_FMAX, O->fn1[k],
                         O->sp1 + (size_t)k * CF_MAXFAC, O->nsp1[k]);
            fputc('\n', fo);
        }
    Q->ms_host += host_ms() - h0;
    Q->n = 0;
    return 0;
}

/* ---- host driver: cofactorise an emitted candidate batch ---------------- *
 *
 * Reads the file `--candidates` writes -- `res0,res1:a,b:f0:f1`, res negative
 * when that side still needs splitting -- and emits the relations that come out
 * of it. Deliberately a separate entry point from the pipeline for now: the
 * corpus is 1,958,036 real candidates from a 1,000-q band, which is a far
 * better test of the splitter than one q's worth would be, and it keeps the
 * algorithm measurable before it is wired into the band loop.
 */

/* Decimal to L limbs. Returns 0 if it does not fit. */
template <int L>
static int cf_from_dec(mz<L> *r, const char *s, int *needs_split)
{
    *needs_split = 0;
    if (*s == '-') { *needs_split = 1; s++; }
    for (int i = 0; i < L; i++) r->v[i] = 0;
    for (; *s >= '0' && *s <= '9'; s++) {
        uint64_t c = (uint64_t)(*s - '0');
        for (int i = 0; i < L; i++) {
            uint64_t t = (uint64_t)r->v[i] * 10u + c;
            r->v[i] = (uint32_t)t;
            c = t >> 32;
        }
        if (c) return 0;
    }
    return 1;
}

typedef struct {
    uint32_t nrec, njob[2], nsplit[2], ndead[2], nstuck[2], nrel;
    double ms[2];
} cof_stat_t;

/* One side's queue: launch, then relaunch whatever is still CF_INCOMPLETE with
 * a different rho constant and a bigger budget. The escalation is the point --
 * a cofactor that resists c=1 is not a cofactor that cannot be split, so the
 * only correct way to stop is to run out of ROUNDS, and even then the record is
 * reported as stuck rather than counted as dead. */
template <int L>
static int cf_run_side(mz<L> *h_n, uint32_t nj, uint64_t lim, uint32_t lpb,
                       uint8_t *h_status, uint32_t *h_fac, uint8_t *h_nfac,
                       int rounds, uint32_t budget0, int blocks, int threads,
                       int verbose, double *ms_out,
                       int ecm, uint32_t ecm_curves, const uint32_t *d_s, uint32_t ns)
{
    mz<L> *d_n = NULL; uint8_t *d_status = NULL, *d_nfac = NULL;
    uint32_t *d_fac = NULL;
    unsigned long long *d_iters = NULL, *h_iters = NULL;
    mz<L> lim2;
    cudaEvent_t e0, e1;
    float ms = 0;
    if (!nj) { *ms_out = 0; return 0; }

    {   /* lim^2, in L limbs */
        unsigned __int128 v = (unsigned __int128)lim * lim;
        for (int i = 0; i < L; i++) { lim2.v[i] = (uint32_t)v; v >>= 32; }
        if (v) { fprintf(stderr, "  cofac: lim^2 does not fit %d limbs\n", L); return -1; }
    }
    CK(cudaMalloc(&d_n, (size_t)nj * sizeof(mz<L>)));
    CK(cudaMalloc(&d_status, nj));
    CK(cudaMalloc(&d_nfac, nj));
    CK(cudaMalloc(&d_fac, (size_t)nj * CF_MAXFAC * 4));
    CK(cudaMalloc(&d_iters, (size_t)nj * 8));
    CK(cudaMemcpy(d_n, h_n, (size_t)nj * sizeof(mz<L>), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_status, CF_INCOMPLETE, nj));
    CK(cudaMemset(d_nfac, 0, nj));
    CK(cudaMemset(d_iters, 0, (size_t)nj * 8));
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0);
    for (int r = 0; r < rounds; r++) {
        if (ecm)
            k_cofac<L, 0, 1><<<blocks, threads>>>(d_n, lim2, lpb, (uint32_t)(r + 1),
                                                  ecm_curves, nj, NULL, NULL,
                                                  d_status, d_fac, d_nfac, d_iters,
                                                  d_s, ns);
        else
            k_cofac<L, 0, 0><<<blocks, threads>>>(d_n, lim2, lpb, (uint32_t)(r + 1),
                                                  budget0 << r, nj, NULL, NULL,
                                                  d_status, d_fac, d_nfac, d_iters,
                                                  NULL, 0);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "  cofac: launch failed\n"); return -1; }
    }
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    cudaEventElapsedTime(&ms, e0, e1);
    *ms_out = ms;
    if (verbose) printf("  side %d-limb queue: %u jobs, %d rounds, %.1f ms\n",
                        L, nj, rounds, ms);

    CK(cudaMemcpy(h_status, d_status, nj, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_nfac, d_nfac, nj, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_fac, d_fac, (size_t)nj * CF_MAXFAC * 4, cudaMemcpyDeviceToHost));

    /* Where do the rho iterations actually go? A CF_DEAD verdict is reached by
     * a primality test, not by splitting, so it is cheap; a CF_INCOMPLETE one
     * has burned the whole budget in every round. If the stuck class dominates
     * the iteration total, the cofactor stage is not paying for the relations
     * it finds -- it is paying to fail to reject. That is the number that
     * decides whether ECM is worth building. */
    h_iters = (unsigned long long *)malloc((size_t)nj * 8);
    if (h_iters) {
        unsigned long long it[6] = {0,0,0,0,0,0}, tot = 0;
        uint32_t cnt[6] = {0,0,0,0,0,0};
        CK(cudaMemcpy(h_iters, d_iters, (size_t)nj * 8, cudaMemcpyDeviceToHost));
        for (uint32_t k = 0; k < nj; k++) {
            int s = h_status[k]; if (s < 0 || s > 5) s = 5;
            it[s] += h_iters[k]; cnt[s]++; tot += h_iters[k];
        }
        if (verbose && tot) {
            static const char *nm[6] = {"split ok", "dead", "stuck", "overflow", "?", "?"};
            printf("  %d-limb rho iterations, by outcome (total %.3g):\n", L, (double)tot);
            for (int s = 0; s < 4; s++)
                if (cnt[s]) printf("    %-9s %8u jobs  %6.2f%% of iters  %9.0f mean\n",
                                   nm[s], cnt[s], 100.0 * (double)it[s] / (double)tot,
                                   (double)it[s] / cnt[s]);
        }
        free(h_iters);
    }

    cudaFree(d_n); cudaFree(d_status); cudaFree(d_nfac); cudaFree(d_fac);
    cudaFree(d_iters);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}


/* ---- post-cofactor reconstruction gate --------------------------------- *
 *
 * The existing gate (pipe_td_verify) checks trial division's factors against
 * the norm BEFORE splitting, so it says nothing about the primes the
 * cofactoriser emits. This one runs over the emitted relation file, which both
 * the inline queue and the standalone splitter write in the same format, so one
 * implementation gates both paths.
 *
 * For each relation it recomputes F(a,b) and G(a,b) exactly from the
 * polynomial, divides out every recorded factor, and requires the remainder to
 * be exactly 1 with every prime within its side's lpb. That is the property a
 * relation actually has to have, and nothing upstream establishes it: a
 * truncated 33-bit factor, a dropped power of two, or a split that emitted a
 * composite all reconstruct wrongly here and nowhere else.
 */
static int cf_norm_exact(bn_t *out, const tdpoly_t *P, int64_t a, int64_t b)
{
    bns_t acc; bns_zero(&acc);
    const uint64_t ua = (uint64_t)(a < 0 ? -a : a), ub = (uint64_t)(b < 0 ? -b : b);
    const int sa = a < 0 ? -1 : 1, sb = b < 0 ? -1 : 1;
    int ovf = 0;
    for (int k = 0; k <= P->deg; k++) {
        bn_t term = P->c[k];
        int s = P->sign[k];
        if (bn_is_zero(&term)) continue;
        for (int e = 0; e < k; e++) { if (bn_mul_u64(&term, ua)) ovf = 1; s *= sa; }
        for (int e = 0; e < P->deg - k; e++) { if (bn_mul_u64(&term, ub)) ovf = 1; s *= sb; }
        if (bns_addmag(&acc, &term, s)) ovf = 1;
    }
    *out = acc.m;
    return ovf;
}

/* Returns 0 if every relation reconstructs, else the number that did not. */
static int cf_check_relations(const char *path, const poly_t *poly,
                              uint32_t lpb0, uint32_t lpb1)
{
    tdpoly_t tp[2];
    FILE *f = fopen(path, "rb");
    char *line = NULL;
    size_t cap = 0;
    long nl = 0;
    uint32_t nok = 0, nbad = 0, nprime = 0;
    if (!f) { perror(path); return -1; }
    if (td_build_poly(&tp[1], poly, 1) || td_build_poly(&tp[0], poly, 0)) {
        fprintf(stderr, "check-relations: polynomial does not fit\n");
        fclose(f); return -1;
    }
    while (getline(&line, &cap, f) > 0) {
        int64_t a, b;
        char *p = line, *e;
        nl++;
        if (*p == '#' || *p == '\n') continue;
        a = strtoll(p, &e, 10); if (*e != ',') { nbad++; continue; }
        b = strtoll(e + 1, &e, 10); if (*e != ':') { nbad++; continue; }
        {
            bn_t N[2];
            char *fld[2];
            int side, bad = 0;
            /* fields: "a,b" ":" side-0 primes ":" side-1 primes */
            fld[0] = e + 1;
            fld[1] = strchr(fld[0], ':');
            if (!fld[1]) { nbad++; continue; }
            *fld[1]++ = 0;
            { char *nlp = strchr(fld[1], '\n'); if (nlp) *nlp = 0; }
            if (cf_norm_exact(&N[1], &tp[1], a, b) ||
                cf_norm_exact(&N[0], &tp[0], a, b)) { nbad++; continue; }
            for (side = 0; side < 2 && !bad; side++) {
                const uint32_t lpb = side ? lpb1 : lpb0;
                char *q = fld[side];
                while (*q) {
                    unsigned long v = strtoul(q, &e, 16);
                    if (e == q) { bad = 1; break; }
                    if (!v || (lpb < 32 && v >= (1ul << lpb))) { nprime++; bad = 1; break; }
                    {   /* exact division: a factor that does not divide is a
                         * reconstruction failure, not a rounding question */
                        bn_t t = N[side];
                        int top = bn_top(&t);
                        if (top < 0 || bn_divmod_u32_pre(&t, (uint32_t)v,
                                                        bn_recip_u32((uint32_t)v), top)) {
                            bad = 1; break;
                        }
                        N[side] = t;
                    }
                    q = e;
                    if (*q == ',') q++; else break;
                }
            }
            if (!bad && bn_is_one(&N[0]) && bn_is_one(&N[1])) nok++;
            else nbad++;
        }
    }
    free(line); fclose(f);
    printf("  relation reconstruction gate: %u of %u rebuild both norms exactly",
           nok, nok + nbad);
    if (nprime) printf(", %u carried a prime above lpb", nprime);
    printf("  %s\n", nbad ? "<-- FAIL" : "PASS");
    (void)nl;
    return (int)nbad;
}

extern "C" int check_relations(const char *path, const poly_t *poly,
                               uint32_t lpb0, uint32_t lpb1)
{
    return cf_check_relations(path, poly, lpb0, lpb1);
}

typedef struct { char *ab, *f0, *f1; } cf_rest_t;

/* Append the split primes to a side's trial-division factor list and print it
 * in the relation format: ascending hex, comma separated. */
static void cf_emit_side(FILE *o, const char *tdfac, const uint32_t *extra,
                         int nextra)
{
    uint32_t f[TD_FMAX + CF_MAXFAC + 2];
    int n = 0;
    const char *p = tdfac;
    while (*p && n < TD_FMAX + CF_MAXFAC) {
        char *end;
        unsigned long v = strtoul(p, &end, 16);
        if (end == p) break;
        f[n++] = (uint32_t)v;
        p = end; if (*p == ',') p++;
    }
    for (int i = 0; i < nextra && n < TD_FMAX + CF_MAXFAC; i++) f[n++] = extra[i];
    qsort(f, n, sizeof f[0], cf_u32cmp);
    for (int i = 0; i < n; i++) fprintf(o, "%s%x", i ? "," : "", f[i]);
}

/* Cofactorise a `--candidates` batch and write the relations it yields.
 *
 * CLASS-AWARE DISPATCH: the rational side runs FIRST and alone. It is one
 * 60-bit semiprime split against the algebraic side's 92-bit 3LP one, and 51%
 * of the corpus needs both. A record whose rational side has a prime factor
 * above 2^lpb is dead whatever the algebraic side does, so its algebraic job is
 * never enqueued at all. */
extern "C" int run_cofac(const char *path, const char *out, uint32_t lim0,
                         uint32_t lpb0, uint32_t lim1, uint32_t lpb1,
                         int rounds, uint32_t budget, int blocks, int threads,
                         int ecm, uint32_t ecm_b1, uint32_t ecm_curves)
{
    FILE *f = fopen(path, "rb");
    uint32_t *h_s = NULL, *d_s = NULL, ns = 0;
    char *buf = NULL;
    size_t sz = 0;
    uint32_t nrec = 0, n0 = 0, n1 = 0;
    cf_rest_t *rest = NULL;
    mz<2> *j0 = NULL; mz<3> *j1 = NULL;
    uint32_t *i0 = NULL, *i1 = NULL, *fac0 = NULL, *fac1 = NULL;
    uint8_t *st0 = NULL, *st1 = NULL, *nf0 = NULL, *nf1 = NULL;
    uint8_t *side0ok = NULL;
    uint32_t *small0 = NULL, *small1 = NULL;
    double ms0 = 0, ms1 = 0, thost;
    uint32_t nrel = 0, nrel2 = 0, dead0 = 0, dead1 = 0, stuck0 = 0, stuck1 = 0, novf = 0;
    uint8_t *alg_job = NULL;
    FILE *fo = NULL;
    char tmp[2048];

    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END); sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    buf = (char *)malloc(sz + 1);
    if (!buf || fread(buf, 1, sz, f) != sz) { fprintf(stderr, "cofac: read failed\n"); fclose(f); free(buf); return -1; }
    buf[sz] = 0; fclose(f);
    for (size_t k = 0; k < sz; k++) if (buf[k] == '\n') nrec++;
    /* A final record with no trailing newline is still a record. Counting only
     * newlines silently dropped it, which is the kind of off-by-one that shows
     * up as one missing relation and gets blamed on the splitter. */
    if (sz && buf[sz - 1] != '\n') nrec++;
    printf("\n=== cofactorisation: %u candidate records from %s ===\n", nrec, path);
    if (!nrec) { free(buf); return -1; }

    rest = (cf_rest_t *)malloc((size_t)nrec * sizeof(cf_rest_t));
    j0 = (mz<2> *)malloc((size_t)nrec * sizeof(mz<2>));
    j1 = (mz<3> *)malloc((size_t)nrec * sizeof(mz<3>));
    i0 = (uint32_t *)malloc((size_t)nrec * 4);
    i1 = (uint32_t *)malloc((size_t)nrec * 4);
    small0 = (uint32_t *)malloc((size_t)nrec * 4);
    small1 = (uint32_t *)malloc((size_t)nrec * 4);
    side0ok = (uint8_t *)malloc(nrec);
    alg_job = (uint8_t *)calloc(nrec, 1);
    if (!rest || !j0 || !j1 || !i0 || !i1 || !small0 || !small1 || !side0ok || !alg_job) {
        fprintf(stderr, "cofac: out of memory\n"); return -1;
    }

    thost = host_ms();
    {   /* parse; keep the trial-division part of each line in place */
        char *p = buf;
        for (uint32_t r = 0; r < nrec; r++) {
            char *nl = strchr(p, '\n');
            if (!nl) {                       /* unterminated final line */
                if (!*p) { nrec = r; break; }
                nl = p + strlen(p);
            }
            *nl = 0;
            {
                char *c1 = strchr(p, ',');
                char *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
                char *c3 = c2 ? strchr(c2 + 1, ':') : NULL;
                char *c4 = c3 ? strchr(c3 + 1, ':') : NULL;
                mz<2> v0; mz<3> v1; int ns0 = 0, ns1 = 0;
                if (!c1 || !c2 || !c3 || !c4) { fprintf(stderr, "cofac: line %u malformed\n", r); return -1; }
                *c1 = 0; *c2 = 0; *c3 = 0; *c4 = 0;
                if (!cf_from_dec<2>(&v0, p, &ns0) || !cf_from_dec<3>(&v1, c1 + 1, &ns1)) {
                    fprintf(stderr, "cofac: line %u cofactor does not fit\n", r); return -1;
                }
                rest[r].ab = c2 + 1; rest[r].f0 = c3 + 1; rest[r].f1 = c4 + 1;
                side0ok[r] = 1;
                /* A side that does NOT need splitting still carries a residual
                 * prime, and it is part of the relation. Dropping it emits a
                 * factorisation whose product is not the norm. */
                small0[r] = ns0 ? 0u : (v0.v[0] > 1u ? v0.v[0] : 0u);
                small1[r] = ns1 ? 0u : (v1.v[0] > 1u ? v1.v[0] : 0u);
                if (ns0) { j0[n0] = v0; i0[n0] = r; n0++; }
                if (ns1) { j1[n1] = v1; i1[n1] = r; n1++; alg_job[r] = 1; }
            }
            p = nl + 1;
        }
    }
    printf("  rational jobs %u (%.1f%%), algebraic jobs %u (%.1f%%)\n",
           n0, 100.0 * n0 / nrec, n1, 100.0 * n1 / nrec);
    thost = host_ms() - thost;

    st0 = (uint8_t *)malloc(n0 ? n0 : 1); nf0 = (uint8_t *)calloc(n0 ? n0 : 1, 1);
    fac0 = (uint32_t *)malloc((size_t)(n0 ? n0 : 1) * CF_MAXFAC * 4);
    if (ecm) {
        ns = cf_ecm_plan(ecm_b1, &h_s);
        if (!ns) { fprintf(stderr, "  cofac: empty ECM plan for B1=%u\n", ecm_b1); return -1; }
        CK(cudaMalloc(&d_s, (size_t)ns * 4));
        CK(cudaMemcpy(d_s, h_s, (size_t)ns * 4, cudaMemcpyHostToDevice));
        printf("  ECM stage 1: B1 = %u, %u prime powers, %u curves per round,"
               " %d rounds\n", ecm_b1, ns, ecm_curves, rounds);
    }
    if (cf_run_side<2>(j0, n0, lim0, lpb0, st0, fac0, nf0,
                       rounds, budget, blocks, threads, 1, &ms0,
                       ecm, ecm_curves, d_s, ns)) return -1;
    for (uint32_t k = 0; k < n0; k++) {
        if (st0[k] == CF_OK) continue;
        side0ok[i0[k]] = 0;
        if (st0[k] == CF_DEAD) dead0++; else if (st0[k] == CF_INCOMPLETE) stuck0++;
        else novf++;
    }
    printf("  rational: %u split, %u dead (a prime > 2^%u), %u still stuck\n",
           n0 - dead0 - stuck0 - novf, dead0, lpb0, stuck0);

    /* class-aware: only records whose rational side survived are enqueued */
    {
        uint32_t keep = 0;
        for (uint32_t k = 0; k < n1; k++)
            if (side0ok[i1[k]]) { j1[keep] = j1[k]; i1[keep] = i1[k]; keep++; }
        printf("  algebraic queue after the rational gate: %u of %u (%.1f%% suppressed)\n",
               keep, n1, n1 ? 100.0 * (n1 - keep) / n1 : 0.0);
        n1 = keep;
    }
    st1 = (uint8_t *)malloc(n1 ? n1 : 1); nf1 = (uint8_t *)calloc(n1 ? n1 : 1, 1);
    fac1 = (uint32_t *)malloc((size_t)(n1 ? n1 : 1) * CF_MAXFAC * 4);
    if (cf_run_side<3>(j1, n1, lim1, lpb1, st1, fac1, nf1,
                       rounds, budget, blocks, threads, 1, &ms1,
                       ecm, ecm_curves, d_s, ns)) return -1;
    for (uint32_t k = 0; k < n1; k++) {
        if (st1[k] == CF_DEAD) dead1++;
        else if (st1[k] == CF_INCOMPLETE) stuck1++;
        else if (st1[k] != CF_OK) novf++;
    }
    printf("  algebraic: %u split, %u dead (a prime > 2^%u), %u still stuck\n",
           n1 - dead1 - stuck1, dead1, lpb1, stuck1);

    if (out) {
        snprintf(tmp, sizeof tmp, "%s.part", out);
        fo = fopen(tmp, "w");
        if (!fo) { perror(tmp); return -1; }
        setvbuf(fo, NULL, _IOFBF, 1 << 22);
    }
    {
        uint32_t *ex0 = (uint32_t *)calloc(nrec, CF_MAXFAC * 4);
        uint8_t  *ne0 = (uint8_t *)calloc(nrec, 1);
        uint8_t  *done1 = (uint8_t *)calloc(nrec, 1);
        for (uint32_t r = 0; r < nrec; r++)
            if (small0[r]) { ex0[(size_t)r * CF_MAXFAC] = small0[r]; ne0[r] = 1; }
        for (uint32_t k = 0; k < n0; k++)
            if (st0[k] == CF_OK) {
                ne0[i0[k]] = nf0[k];
                for (int z = 0; z < nf0[k]; z++)
                    ex0[(size_t)i0[k] * CF_MAXFAC + z] = fac0[(size_t)k * CF_MAXFAC + z];
            }
        for (uint32_t k = 0; k < n1; k++) {
            uint32_t r = i1[k];
            if (st1[k] != CF_OK || !side0ok[r]) continue;
            done1[r] = 1; nrel++;
            if (fo) {
                fprintf(fo, "%s:", rest[r].ab);
                cf_emit_side(fo, rest[r].f0, ex0 + (size_t)r * CF_MAXFAC, ne0[r]);
                fputc(':', fo);
                cf_emit_side(fo, rest[r].f1, fac1 + (size_t)k * CF_MAXFAC, nf1[k]);
                fputc('\n', fo);
            }
        }
        /* The records whose algebraic side needed no splitting: they are
         * relations as soon as the rational side splits, and they have no
         * algebraic job to come back from. `small1` is 0 both when the side
         * was queued AND when its cofactor was already 1, so the queued case
         * has to be excluded by `done1`/membership, not by `small1` -- testing
         * `small1` alone silently dropped every record whose algebraic
         * cofactor trial division had already reduced to 1. */
        for (uint32_t r = 0; r < nrec; r++) {
            uint32_t one1 = small1[r];
            if (done1[r] || alg_job[r] || !side0ok[r]) continue;
            nrel2++;
            if (fo) {
                fprintf(fo, "%s:", rest[r].ab);
                cf_emit_side(fo, rest[r].f0, ex0 + (size_t)r * CF_MAXFAC, ne0[r]);
                fputc(':', fo);
                cf_emit_side(fo, rest[r].f1, &one1, one1 ? 1 : 0);
                fputc('\n', fo);
            }
        }
        free(ex0); free(ne0); free(done1);
        printf("  relations: %u from the algebraic queue + %u whose algebraic"
               " side needed none\n", nrel, nrel2);
        nrel += nrel2;
    }
    if (fo) {
        int bad = ferror(fo);
        if (fclose(fo) || bad) { fprintf(stderr, "cofac: write failed\n"); remove(tmp); return -1; }
        if (rename(tmp, out)) { perror("rename"); remove(tmp); return -1; }
    }
    if (novf) fprintf(stderr, "  ** %u records produced more than %d factors\n", novf, CF_MAXFAC);

    printf("\n  --- cofactorisation ---\n");
    printf("  %-34s %8u\n", "candidate records", nrec);
    printf("  %-34s %8.1f ms\n", "rational queue (2 limbs)", ms0);
    printf("  %-34s %8.1f ms\n", "algebraic queue (3 limbs)", ms1);
    printf("  %-34s %8.1f ms\n", "host parse", thost);
    printf("  %-34s %8u   (%.2f%% of candidates)\n", "RELATIONS", nrel,
           100.0 * nrel / nrec);
    printf("  %-34s %8u\n", "still stuck (requeue, not dead)", stuck0 + stuck1);

    free(buf); free(rest); free(j0); free(j1); free(i0); free(i1);
    free(small0); free(small1); free(side0ok); free(alg_job);
    free(st0); free(st1); free(nf0); free(nf1); free(fac0); free(fac1);
    return novf ? -1 : 0;
}

#endif  /* __CUDACC__ */
#endif  /* CUDA_SIEVE_COFAC_CUH */
