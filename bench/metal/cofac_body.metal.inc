/* Optional buffer arguments: metal_rt supplies bit i = "argument i is bound". */
constant uint mtl_bound_mask [[function_constant(0)]];
constant bool mtl_bound_10 = (mtl_bound_mask & (1u << 10)) != 0;

template <int L> struct mz { uint32_t v[L]; };

template <int L> static inline int mz_cmp(const thread mz<L> *a, const thread mz<L> *b)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

template <int L> static inline int mz_is_zero(const thread mz<L> *a)
{
    uint32_t o = 0;
    #pragma unroll
    for (int i = 0; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> static inline int mz_is_one(const thread mz<L> *a)
{
    uint32_t o = a->v[0] ^ 1u;
    #pragma unroll
    for (int i = 1; i < L; i++) o |= a->v[i];
    return o == 0;
}

template <int L> static inline uint32_t mz_sub(thread mz<L> *a, const thread mz<L> *b)
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
template <int L> static inline void mz_sub_mod(thread mz<L> *a, const thread mz<L> *b, const thread mz<L> *n)
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
template <int L> static inline void mz_add_mod(thread mz<L> *a, const thread mz<L> *b, const thread mz<L> *n)
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
template <int L> static inline void mz_dbl_mod(thread mz<L> *a, const thread mz<L> *n)
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

template <int L> static inline void mz_rshift1(thread mz<L> *a)
{
    #pragma unroll
    for (int i = 0; i < L - 1; i++)
        a->v[i] = (a->v[i] >> 1) | (a->v[i + 1] << 31);
    a->v[L - 1] >>= 1;
}

template <int L> static inline int mz_bits(const thread mz<L> *a)
{
    #pragma unroll
    for (int i = L - 1; i >= 0; i--)
        if (a->v[i]) return i * 32 + (32 - __clz(a->v[i]));
    return 0;
}

/* -n^{-1} mod 2^32, Newton. n odd. */
static inline uint32_t mz_n0inv(uint32_t n0)
{
    uint32_t inv = n0;
    for (int k = 0; k < 5; k++) inv *= 2u - n0 * inv;
    return (uint32_t)(0u - inv);
}

/* r = a*b*R^-1 mod n, R = 2^(32L). CIOS. */
template <int L>
static inline void mz_mul(thread mz<L> *r, const thread mz<L> *a, const thread mz<L> *b,
                  const thread mz<L> *n, uint32_t n0inv)
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
template <int L> static inline void mz_gcd(thread mz<L> *g, mz<L> a, mz<L> b)
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
template <int L> static inline void mz_divexact(thread mz<L> *q, const thread mz<L> *n, const thread mz<L> *d)
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
template <int L> static inline int mz_sprp2(const thread mz<L> *n)
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
static inline int mz_rho(thread mz<L> *fac, const thread mz<L> *n, uint32_t c0, uint32_t budget,
                 thread uint64_t *acc)
{
    const uint32_t n0inv = mz_n0inv(n->v[0]);
    mz<L> one, y, x, ys, q, g, c;
    uint32_t steps = 0, r = 1;
    bool rho_found = false;   /* stands in for the original goto */

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
            if (!mz_is_one<L>(&g)) { rho_found = true; break; }
            if (steps >= budget) { *acc += steps; return 0; }
        }
        if (rho_found) break;
        r <<= 1;
        if (steps >= budget) { *acc += steps; return 0; }
    }

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

/* ---- ECM, stages 1 and 2 ------------------------------------------------ *
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
 * Stage 2 is the small-table Montgomery-XZ continuation below. It deliberately
 * uses D=30: the four unsigned baby residues {1,7,11,13} keep one curve's
 * private table small enough for this first GPU experiment. The host supplies
 * one four-bit mask per giant step; bit u is set when vD-u or vD+u is prime in
 * (B1,B2]. Thus every lane follows the same giant-step walk and only the cheap
 * product-of-differences loop is masked.
 */

#define CF_ECM_D       30u
#define CF_ECM_NBABY    4u
#define CF_ECM_MAX_B1   1000000u
#define CF_ECM_MAX_B2  10000000u

/* One definition supplies both the host mask-bit mapping and the device baby
 * scalar. A duplicated host/device table can compile and test cleanly while
 * silently assigning different meanings to the same mask bit. */
static inline uint8_t cf_ecm_baby(uint32_t k)
{
    return k == 0 ? 1u : k == 1 ? 7u : k == 2 ? 11u : 13u;
}

/* Shared composite sieve for both ECM plans. Bounds live here as well as in
 * the CLI: these helpers are also called by tests and must be safe on their
 * own. */

/* The stage-1 exponent lcm(1..B1), held as its prime powers rather than as one
 * big integer: each is at most B1, so the ladder for it is a dozen bits and the
 * whole schedule is an array of uint32 instead of a 1.4*B1-bit number the device
 * would have to store and shift. Returns the count. */

/* Build the stage-2 prime-pair schedule. For every prime q in (B1,B2], write
 * q = vD +/- u with 0 < u < D/2. Since q > B1 >= 30 in the stage-2 path, q is
 * coprime to D and u is one of the four residues below. One x-coordinate
 * comparison covers both signs. Empty giant steps remain in the schedule so
 * the device can advance with one differential addition rather than restart a
 * scalar multiplication at every populated v. */

template <int L> struct mpt { mz<L> X, Z; };

/* r = k * one, for a small k, by double-and-add on the Montgomery form of 1. */
template <int L>
static inline void mz_small(thread mz<L> *r, uint32_t k, const thread mz<L> *one, const thread mz<L> *n)
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
static inline void x_dbl(thread mpt<L> *r, const thread mpt<L> *p, const thread mz<L> *an, const thread mz<L> *ad,
                 const thread mz<L> *n, uint32_t n0inv)
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
static inline void x_add(thread mpt<L> *r, const thread mpt<L> *p, const thread mpt<L> *q, const thread mpt<L> *dif,
                 const thread mz<L> *n, uint32_t n0inv)
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
static inline void x_mul(thread mpt<L> *p, uint32_t k, const thread mz<L> *an, const thread mz<L> *ad,
                 const thread mz<L> *n, uint32_t n0inv, thread uint64_t *work)
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
        (*work)++;
    }
    *p = r0;
}

/* Keep one copy of the ladder behind a call boundary. Force-inlining x_mul at
 * every baby/giant setup site makes nvcc expand seven copies of the bigint
 * arithmetic into this already-large kernel and markedly increases both
 * compile time and register pressure. */
template <int L>
static inline void x_mul_s2(thread mpt<L> *p, uint32_t k, const thread mz<L> *an,
                          const thread mz<L> *ad, const thread mz<L> *n, uint32_t n0inv,
                          thread uint64_t *work)
{
    x_mul<L>(p, k, an, ad, n, n0inv, work);
}

/* One pass over the D=30 baby-step/giant-step continuation. All selected
 * cross-products are accumulated and normally only one gcd is paid. If hits
 * for different prime factors make that gcd equal n, replay the same curve and
 * gcd each selected difference. The replay is rare, preserves the fast common
 * path, and prevents a batched product from swallowing every factor it found. */
template <int L>
static inline int mz_ecm_stage2_pass(thread mz<L> *fac, const thread mpt<L> *Q,
                                    const thread mz<L> *an, const thread mz<L> *ad,
                                    const thread mz<L> *one, const thread mz<L> *n,
                                    uint32_t n0inv,
                                    const device uint8_t * masks,
                                    uint32_t vmin, uint32_t nv,
                                    thread uint64_t *work)
{
    /* The baby steps are held on a SHARED DENOMINATOR: bx[k] = X_k *
     * prod_{j!=k} Z_j against one bz = prod_j Z_j, so (bx[k] : bz) is the
     * same projective point (X_k : Z_k) was. Every cross product below is
     * then the old one scaled by prod_{j!=k} Z_j.
     *
     * That scaling leaves gcd(d, n) alone only when the scale factor is a UNIT
     * mod n. It normally is -- cofcheck.sh's pinned counts match and a 148-q
     * A/B produced identical relations either way -- but that is empirical,
     * NOT a proof, and an earlier version of this comment wrongly claimed the
     * factors were provably identical.
     *
     * Note the BLAST RADIUS when it is not. Every bx[k] except k's own carries
     * Z_k as a factor, so ONE baby step whose Z shares a factor f with n
     * contaminates the cross product of every OTHER k. All of those gcds
     * become multiples of f, which may surface f earlier than the per-point
     * form would, or may grow to exactly n and be discarded by the g != n
     * guard -- losing chances the per-point form kept independent. Z_k == 0
     * mod n is the extreme: every other k's d collapses to 0 and is rejected
     * outright, while k's own term survives. The per-point form has no such
     * coupling between k.
     *
     * It needs a baby-step Z sharing a factor with n (the lucky-hit case
     * stage 1 usually catches), and it can NEVER return a wrong factor: any
     * gcd with n is a true divisor, and g == n is rejected. That is what
     * makes this safe -- "never returns a non-factor", not "same factors".
     *
     * The point is the inner loop: with one denominator for every baby step,
     * the X_G * Z term is common to all k and hoists out, so a selected pair
     * costs ONE multiply instead of two. At B1=200/B2=6000 that is ~540 fewer
     * mz_mul per curve against the rescale's one-off cost -- 4*NBABY mz_mul
     * issued, two per k in each of the forward and backward passes. Three of
     * those are provably redundant (both passes open by multiplying into
     * acc == one, which is the identity in Montgomery form, and the backward
     * pass's final acc update is never read), so 4*NBABY - 3 do real work.
     * Peeling them is left alone deliberately: 3 multiplies per replay is
     * ~1% of the ~540 this change saves, and it is not worth restructuring
     * delicate modular arithmetic for. Measured
     * A/B on gfx1103 (oracle/c183, 148 q, only this function differing):
     * algebraic queue 52.62/52.18 -> 50.19/49.75 ms, ~4.6%, with the rational
     * (rho) queue unchanged as the control and 596 relations either way.
     *
     * Chosen over per-point affine normalisation deliberately: affine needs a
     * modular inverse, there is no mz_inv in this file, and a binary extended
     * GCD in device code is new delicate arithmetic for no extra gain here.
     * The shared denominator needs no inversion at all.
     *
     * NOT a register-pressure fix, and not why RDNA1/2 lost stage 2 -- every
     * stage-2 kernel is already at vgpr=128, so trimming live values relocates
     * spill rather than freeing registers (measured: scratch 336 -> 384 on
     * gfx1030 L=3). That was the gfx10 MUBUF stack lowering; see CLAUDE.md. */
    mz<L> bx[CF_ECM_NBABY], bz;
    mpt<L> step, prev, cur, next;
    mz<L> prod, g;

    for (int replay = 0; replay < 2; replay++) {
        prod = *one;
        {
            /* Each baby step is built in ONE scratch point, then split into
             * bx[k] and bzs[k]; bx[] doubles as the running accumulator, so
             * no separate prefix array is needed.
             *
             * This does NOT reduce peak storage, and an earlier version of
             * this comment wrongly implied it did: an mpt<L> is (X, Z), so
             * the baby[CF_ECM_NBABY] this replaced was already 2*NBABY mz
             * values -- exactly what bx[] + bzs[] is. With p, acc, t and gx
             * live on top, the peak is slightly HIGHER, which is what the
             * measured scratch 336 -> 384 below is. Deliberately NOT unrolled --
             * x_mul_s2 is __noinline__, and unrolling its call sites widens
             * the live range of every argument across all NBABY calls. */
            mz<L> bzs[CF_ECM_NBABY], acc, t;
            for (uint32_t k = 0; k < CF_ECM_NBABY; k++) {
                const uint32_t scalar = cf_ecm_baby(k);
                mpt<L> p = *Q;
                if (scalar != 1)
                    x_mul_s2<L>(&p, scalar, an, ad, n, n0inv, work);
                bx[k] = p.X; bzs[k] = p.Z;
            }
            /* Forward: bx[k] = X_k * prod_{j<k} Z_j, acc ends at prod_j Z_j. */
            acc = *one;
            for (uint32_t k = 0; k < CF_ECM_NBABY; k++) {
                mz_mul<L>(&t, &bx[k], &acc, n, n0inv); bx[k] = t;
                mz_mul<L>(&t, &acc, &bzs[k], n, n0inv); acc = t;
            }
            bz = acc;
            /* Backward: acc holds prod_{j>k} Z_j, so this completes each
             * bx[k] to X_k * prod_{j!=k} Z_j. */
            acc = *one;
            for (int k = (int)CF_ECM_NBABY - 1; k >= 0; k--) {
                mz_mul<L>(&t, &bx[k], &acc, n, n0inv); bx[k] = t;
                mz_mul<L>(&t, &acc, &bzs[k], n, n0inv); acc = t;
            }
        }
        step = *Q; x_mul_s2<L>(&step, CF_ECM_D, an, ad, n, n0inv, work);
        prev = step; x_mul_s2<L>(&prev, vmin, an, ad, n, n0inv, work);
        if (nv > 1) {
            cur = step;
            x_mul_s2<L>(&cur, vmin + 1, an, ad, n, n0inv, work);
        }

        for (uint32_t vi = 0; vi < nv; vi++) {
            const thread mpt<L> *G = vi ? &cur : &prev;
            const uint32_t mask = masks[vi];
            if (mask) {
                /* One denominator for every baby step, so this term is the
                 * same for all k -- hoisted out of the loop below, and not
                 * computed at all on a giant step that selects nothing. */
                mz<L> gx;
                mz_mul<L>(&gx, &G->X, &bz, n, n0inv);
                #pragma unroll
                for (uint32_t k = 0; k < CF_ECM_NBABY; k++)
                    if (mask & (1u << k)) {
                        mz<L> b, d;
                        mz_mul<L>(&b, &bx[k], &G->Z, n, n0inv);
                        d = gx; mz_sub_mod<L>(&d, &b, n);
                        (*work)++;              /* one selected stage-2 pair */
                        if (replay) {
                            mz_gcd<L>(&g, d, *n);
                            if (!mz_is_one<L>(&g) && mz_cmp<L>(&g, n) != 0) {
                                *fac = g; return 1;
                            }
                        } else {
                            mz_mul<L>(&b, &prod, &d, n, n0inv); prod = b;
                        }
                    }
            }
            if (vi + 1 < nv && vi) {
                x_add<L>(&next, &cur, &step, &prev, n, n0inv);
                (*work)++;                       /* one stage-2 giant step */
                prev = cur; cur = next;
            }
        }
        if (replay) return 0;
        mz_gcd<L>(&g, prod, *n);
        if (mz_is_one<L>(&g)) return 0;
        if (mz_cmp<L>(&g, n) != 0) { *fac = g; return 1; }
    }
    return 0;
}

/* One curve of stage 1. `s` holds the prime powers p^e <= B1, ascending.
 * Returns 1 and a nontrivial factor when gcd(Z, n) splits n. */
template <int L, int STAGE2>
static inline int mz_ecm(thread mz<L> *fac, const thread mz<L> *n, uint32_t sigma,
                 const device uint32_t * s, uint32_t ns,
                 const device uint8_t * s2mask, uint32_t s2vmin,
                 uint32_t s2nv, thread uint64_t *work)
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
        x_mul<L>(&P, s[i], &an, &ad, n, n0inv, work);

    mz_gcd<L>(&g, P.Z, *n);
    if (mz_is_one<L>(&g)) {
        if constexpr (STAGE2) {
            if (mz_ecm_stage2_pass<L>(fac, &P, &an, &ad, &one, n, n0inv,
                                      s2mask, s2vmin, s2nv, work)) return 1;
        }
        return 0;
    }
    if (mz_cmp<L>(&g, n) == 0) return 0;
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
 * ECM and `budget` is a CURVE count. Both are bounded per launch and
 * neither turns an exhausted budget into a proof, so the requeue structure
 * around them is unchanged. */
template <int L, int METHOD, int STAGE2>
static inline int mz_split(const thread mz<L> *n0, const thread mz<L> *lim2, uint32_t lpb,
                   uint32_t c0, uint32_t budget, thread uint64_t *out, thread int *nout,
                   thread uint64_t *acc, const device uint32_t * s, uint32_t ns,
                   const device uint8_t * s2mask, uint32_t s2vmin,
                   uint32_t s2nv)
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
            /* Two limbs. The bound just checked is lpb, not 32, so at
             * lpb > 32 this factor legitimately needs 64 bits; m.v[0] alone
             * used to emit its low half. L is 2 or 3 at every instantiation,
             * but the guard costs nothing and makes the assumption explicit. */
            if constexpr (L >= 2) out[nf++] = ((uint64_t)m.v[1] << 32) | m.v[0];
            else                  out[nf++] = (uint64_t)m.v[0];
            continue;
        }
        if (mz_sprp2<L>(&m)) return CF_DEAD;        /* prime, and >= lim2 > 2^lpb */
        if constexpr (METHOD == 0) {
            if (!mz_rho<L>(&g, &m, c0, budget, acc)) return CF_INCOMPLETE;
        } else {
            uint32_t cv = 0;
            for (; cv < budget; cv++)
                if (mz_ecm<L, STAGE2>(&g, &m, c0 * 1000u + cv + 6u, s, ns,
                                      s2mask, s2vmin, s2nv, acc)) break;
            if (cv == budget) return CF_INCOMPLETE;
        }
        mz_divexact<L>(&h, &m, &g);
        if (sp + 2 > CF_MAXFAC) return CF_OVERFLOW;
        st[sp++] = g; st[sp++] = h;
    }
    *nout = nf;
    return CF_OK;
}



template <int L, int METHOD, int STAGE2>
static inline void k_cofac_body(
    device const mz<L> * n,
    constant mz<L> &lim2_v,
    constant uint32_t &lpb,
    constant uint32_t &c0,
    constant uint32_t &budget,
    device const uint32_t * sel,
    device const uint32_t * njp,
    device uint8_t * status,
    device uint64_t * fac,
    device uint8_t * nfac,
    device ulong * iters,
    device const uint32_t * s,
    constant uint32_t &ns,
    device const uint8_t * s2mask,
    constant uint32_t &s2vmin,
    constant uint32_t &s2nv,
    constant uint32_t &ibeg,
    constant uint32_t &iend,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS
    mz<L> lim2 = lim2_v;   /* CUDA passes this by value */

    /* [ibeg, iend) is this launch's slice of the compacted list; cf_sched_t's
     * `chunk` comment says why the caller may split a round this way. The
     * count is still read on the device, so the host never has to know it and
     * the round loop stays free of synchronisation: a slice that starts past
     * the live count simply runs an empty loop, which is what the later rounds
     * (a few percent of the jobs) mostly do. */
    const uint32_t cnt = *njp;
    const uint32_t hi = (iend < cnt) ? iend : cnt;
    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t ii = ibeg + bench_grid_thread_x(); ii < hi; ii += stride) {
        const uint32_t i = (uint32_t)ii;
        const uint32_t t = sel[i];
        uint64_t o[CF_MAXFAC];
        uint64_t acc = 0;
        int k = 0, st;
        mz<L> v;
        if (status[t] != CF_INCOMPLETE) continue;
        v = n[t];
        st = mz_split<L, METHOD, STAGE2>(&v, &lim2, lpb, c0, budget, o, &k,
                                         &acc, s, ns, s2mask, s2vmin, s2nv);
        if (iters) iters[t] += (ulong)acc;   /* summed over rounds */
        status[t] = (uint8_t)st;
        if (st == CF_OK) {
            nfac[t] = (uint8_t)k;
            for (int i = 0; i < k; i++) fac[(size_t)t * CF_MAXFAC + i] = o[i];
        }
    }
}

kernel void k_cofac_3_1_1(
    device const mz<3> * n [[buffer(0)]],
    constant mz<3> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<3, 1, 1>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cofac_3_1_0(
    device const mz<3> * n [[buffer(0)]],
    constant mz<3> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<3, 1, 0>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cofac_3_0_0(
    device const mz<3> * n [[buffer(0)]],
    constant mz<3> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<3, 0, 0>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cofac_4_1_1(
    device const mz<4> * n [[buffer(0)]],
    constant mz<4> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<4, 1, 1>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cofac_4_1_0(
    device const mz<4> * n [[buffer(0)]],
    constant mz<4> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<4, 1, 0>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cofac_4_0_0(
    device const mz<4> * n [[buffer(0)]],
    constant mz<4> &lim2 [[buffer(1)]],
    constant uint32_t &lpb [[buffer(2)]],
    constant uint32_t &c0 [[buffer(3)]],
    constant uint32_t &budget [[buffer(4)]],
    device const uint32_t * sel [[buffer(5)]],
    device const uint32_t * njp [[buffer(6)]],
    device uint8_t * status [[buffer(7)]],
    device uint64_t * fac [[buffer(8)]],
    device uint8_t * nfac [[buffer(9)]],
    device ulong * iters [[buffer(10), function_constant(mtl_bound_10)]],
    device const uint32_t * s [[buffer(11)]],
    constant uint32_t &ns [[buffer(12)]],
    device const uint8_t * s2mask [[buffer(13)]],
    constant uint32_t &s2vmin [[buffer(14)]],
    constant uint32_t &s2nv [[buffer(15)]],
    constant uint32_t &ibeg [[buffer(16)]],
    constant uint32_t &iend [[buffer(17)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    device ulong * iters_opt = nullptr;
    if (mtl_bound_10) iters_opt = iters;
    k_cofac_body<4, 0, 0>(n, lim2, lpb, c0, budget, sel, njp, status, fac, nfac, iters_opt, s, ns, s2mask, s2vmin, s2nv, ibeg, iend, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


kernel void k_cof_selflags(
    constant uint32_t &n [[buffer(0)]],
    device const uint8_t * st [[buffer(1)]],
    device uint32_t * flag [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        flag[t] = (st[t] == CF_INCOMPLETE) ? 1u : 0u;
    }
}


kernel void k_cof_selscatter(
    constant uint32_t &n [[buffer(0)]],
    device const uint32_t * flag [[buffer(1)]],
    device const uint32_t * off [[buffer(2)]],
    device uint32_t * sel [[buffer(3)]],
    device uint32_t * nsel [[buffer(4)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t]) sel[off[t]] = t;
        if (t == n - 1) *nsel = off[t] + flag[t];
    }
}


kernel void k_cof_gate(
    constant uint32_t &n [[buffer(0)]],
    device uint8_t * st0 [[buffer(1)]],
    device uint8_t * st1 [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (st0[t] != CF_OK && st1[t] == CF_INCOMPLETE) st1[t] = CF_DEAD;
    }
}


kernel void k_cof_status_hist(
    constant uint32_t &n [[buffer(0)]],
    device const uint8_t * st0 [[buffer(1)]],
    device const uint8_t * st1 [[buffer(2)]],
    device uint32_t * out [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    /* Relies on blockDim.x >= 8 for both the zeroing and the flush below.
     * Guaranteed: --threads is refused outside [32, 1024] and must be a
     * multiple of 32. Stated because nothing else in this file needs a block
     * floor, so a future launch geometry could break it silently -- bins 4..7
     * are side 1, and they would read uninitialised shared memory and never
     * be flushed, i.e. the instrumentation would lie rather than fail. */
    threadgroup unsigned int bin[8];
    if (threadIdx.x < 8) bin[threadIdx.x] = 0u;
    __syncthreads();
    {
        const uint64_t stride = bench_grid_stride_x();
        for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
            const uint32_t t = (uint32_t)tt;
            uint32_t a = st0[t], b = st1[t];
            atomicAdd(&bin[(a < 4u ? a : 3u)], 1u);
            atomicAdd(&bin[4u + (b < 4u ? b : 3u)], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x < 8 && bin[threadIdx.x])
        atomicAdd64(&out[2 * threadIdx.x], (ulong)bin[threadIdx.x]);
}


kernel void k_rel_flags(
    constant uint32_t &n [[buffer(0)]],
    device const uint8_t * st0 [[buffer(1)]],
    device const uint8_t * st1 [[buffer(2)]],
    device uint32_t * flag [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        flag[t] = (st0[t] == CF_OK && st1[t] == CF_OK) ? 1u : 0u;
    }
}


kernel void k_rel_gather(
    constant uint32_t &n [[buffer(0)]],
    device const uint32_t * flag [[buffer(1)]],
    device const uint32_t * off [[buffer(2)]],
    constant uint32_t &cap [[buffer(3)]],
    device uint32_t * idx [[buffer(4)]],
    device uint32_t * nrel [[buffer(5)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
        if (flag[t] && off[t] < cap) idx[off[t]] = t;
        if (t == n - 1) *nrel = off[t] + flag[t];
    }
}


kernel void k_rel_pack(
    constant uint32_t &nr [[buffer(0)]],
    device const uint32_t * idx [[buffer(1)]],
    constant cofq_dev_t &Q [[buffer(2)]],
    device int64_t * oa [[buffer(3)]],
    device int64_t * ob [[buffer(4)]],
    device uint32_t * of0 [[buffer(5)]],
    device uint8_t * ofn0 [[buffer(6)]],
    device uint32_t * of1 [[buffer(7)]],
    device uint8_t * ofn1 [[buffer(8)]],
    device uint64_t * osp0 [[buffer(9)]],
    device uint8_t * onsp0 [[buffer(10)]],
    device uint64_t * osp1 [[buffer(11)]],
    device uint8_t * onsp1 [[buffer(12)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < nr; tt += stride) {
        const uint32_t t = (uint32_t)tt;
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


template <int L0, int L1>
static inline void k_cof_enqueue_body(
    device const bn_t * cof0,
    device const uint8_t * bits0,
    device const bn_t * cof1,
    device const uint8_t * bits1,
    device const int64_t * a,
    device const int64_t * b,
    device const uint32_t * f0,
    device const uint32_t * fn0,
    device const uint32_t * f1,
    device const uint32_t * fn1,
    constant uint32_t &n,
    constant uint32_t &base,
    constant uint32_t &lpb0,
    constant uint32_t &lpb1,
    constant cofq_dev_t &Q,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    CUDA_KERNEL_IDS

    const uint64_t stride = bench_grid_stride_x();
    for (uint64_t tt = bench_grid_thread_x(); tt < n; tt += stride) {
        const uint32_t t = (uint32_t)tt;
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
            /* bn_fits_u64, not a hand-rolled two-limb pack: this branch is
             * taken when the residual is already within lpb, and at lpb > 32
             * that residual is a prime needing 64 bits. The helper also
             * REJECTS a value with limbs above the low two set, so a future
             * change to lpb or the bits accounting cannot silently reinstate
             * the truncation this whole path exists to remove. A 0 here is a
             * residual that did not fit, which cannot happen while
             * bits <= lpb <= 64 and is emitted as "no residual" if it ever
             * does -- visible as a failed reconstruction, not a wrong one. */
            uint64_t sm = 0;
            bn_t v = cof0[t];
            if ((uint32_t)bits0[t] > lpb0) {
                uint32_t hi = 0;
                #pragma unroll
                for (int i = 0; i < L0; i++) Q.d_c0[(size_t)d * L0 + i] = v.v[i];
                #pragma unroll
                for (int i = L0; i < BN_LIMBS; i++) hi |= v.v[i];
                Q.d_sm0[d] = 0;
                if (hi) { Q.d_st0[d] = CF_OVERFLOW; atomicAdd(Q.d_ovf, 1u); }
                else      Q.d_st0[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm0[d] = (bits0[t] > 1 && bn_fits_u64(&v, &sm)) ? sm : 0ull;
                Q.d_st0[d] = CF_OK;
            }
            v = cof1[t];
            if ((uint32_t)bits1[t] > lpb1) {
                uint32_t hi = 0;
                #pragma unroll
                for (int i = 0; i < L1; i++) Q.d_c1[(size_t)d * L1 + i] = v.v[i];
                #pragma unroll
                for (int i = L1; i < BN_LIMBS; i++) hi |= v.v[i];
                Q.d_sm1[d] = 0;
                if (hi) { Q.d_st1[d] = CF_OVERFLOW; atomicAdd(Q.d_ovf, 1u); }
                else      Q.d_st1[d] = CF_INCOMPLETE;
            } else {
                Q.d_sm1[d] = (bits1[t] > 1 && bn_fits_u64(&v, &sm)) ? sm : 0ull;
                Q.d_st1[d] = CF_OK;
            }
        }
    }
}

kernel void k_cof_enqueue_3_3(
    device const bn_t * cof0 [[buffer(0)]],
    device const uint8_t * bits0 [[buffer(1)]],
    device const bn_t * cof1 [[buffer(2)]],
    device const uint8_t * bits1 [[buffer(3)]],
    device const int64_t * a [[buffer(4)]],
    device const int64_t * b [[buffer(5)]],
    device const uint32_t * f0 [[buffer(6)]],
    device const uint32_t * fn0 [[buffer(7)]],
    device const uint32_t * f1 [[buffer(8)]],
    device const uint32_t * fn1 [[buffer(9)]],
    constant uint32_t &n [[buffer(10)]],
    constant uint32_t &base [[buffer(11)]],
    constant uint32_t &lpb0 [[buffer(12)]],
    constant uint32_t &lpb1 [[buffer(13)]],
    constant cofq_dev_t &Q [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_cof_enqueue_body<3, 3>(cof0, bits0, cof1, bits1, a, b, f0, fn0, f1, fn1, n, base, lpb0, lpb1, Q, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cof_enqueue_3_4(
    device const bn_t * cof0 [[buffer(0)]],
    device const uint8_t * bits0 [[buffer(1)]],
    device const bn_t * cof1 [[buffer(2)]],
    device const uint8_t * bits1 [[buffer(3)]],
    device const int64_t * a [[buffer(4)]],
    device const int64_t * b [[buffer(5)]],
    device const uint32_t * f0 [[buffer(6)]],
    device const uint32_t * fn0 [[buffer(7)]],
    device const uint32_t * f1 [[buffer(8)]],
    device const uint32_t * fn1 [[buffer(9)]],
    constant uint32_t &n [[buffer(10)]],
    constant uint32_t &base [[buffer(11)]],
    constant uint32_t &lpb0 [[buffer(12)]],
    constant uint32_t &lpb1 [[buffer(13)]],
    constant cofq_dev_t &Q [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_cof_enqueue_body<3, 4>(cof0, bits0, cof1, bits1, a, b, f0, fn0, f1, fn1, n, base, lpb0, lpb1, Q, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cof_enqueue_4_3(
    device const bn_t * cof0 [[buffer(0)]],
    device const uint8_t * bits0 [[buffer(1)]],
    device const bn_t * cof1 [[buffer(2)]],
    device const uint8_t * bits1 [[buffer(3)]],
    device const int64_t * a [[buffer(4)]],
    device const int64_t * b [[buffer(5)]],
    device const uint32_t * f0 [[buffer(6)]],
    device const uint32_t * fn0 [[buffer(7)]],
    device const uint32_t * f1 [[buffer(8)]],
    device const uint32_t * fn1 [[buffer(9)]],
    constant uint32_t &n [[buffer(10)]],
    constant uint32_t &base [[buffer(11)]],
    constant uint32_t &lpb0 [[buffer(12)]],
    constant uint32_t &lpb1 [[buffer(13)]],
    constant cofq_dev_t &Q [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_cof_enqueue_body<4, 3>(cof0, bits0, cof1, bits1, a, b, f0, fn0, f1, fn1, n, base, lpb0, lpb1, Q, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_cof_enqueue_4_4(
    device const bn_t * cof0 [[buffer(0)]],
    device const uint8_t * bits0 [[buffer(1)]],
    device const bn_t * cof1 [[buffer(2)]],
    device const uint8_t * bits1 [[buffer(3)]],
    device const int64_t * a [[buffer(4)]],
    device const int64_t * b [[buffer(5)]],
    device const uint32_t * f0 [[buffer(6)]],
    device const uint32_t * fn0 [[buffer(7)]],
    device const uint32_t * f1 [[buffer(8)]],
    device const uint32_t * fn1 [[buffer(9)]],
    constant uint32_t &n [[buffer(10)]],
    constant uint32_t &base [[buffer(11)]],
    constant uint32_t &lpb0 [[buffer(12)]],
    constant uint32_t &lpb1 [[buffer(13)]],
    constant cofq_dev_t &Q [[buffer(14)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_cof_enqueue_body<4, 4>(cof0, bits0, cof1, bits1, a, b, f0, fn0, f1, fn1, n, base, lpb0, lpb1, Q, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

