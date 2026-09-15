static inline uint32_t d_big_mod(const constant gpu_big_t *a, uint32_t p)
{
    uint64_t r = 0;
    for (int i = a->n - 1; i >= 0; i--)
        r = ((r << 32) | a->v[i]) % p;
    if (a->neg && r) r = p - r;
    return (uint32_t)r;
}

static inline uint32_t d_mod_add(uint32_t a, uint32_t b, uint32_t p)
{
    /* a,b < p.  Work entirely in 32 bits: on wrap, subtracting p in the
     * wrapped word is exactly (a+b)-p because the true sum is below 2p. */
    uint32_t z = a + b;
    if (z < a || z >= p) z -= p;
    return z;
}

static inline uint32_t d_mod_sub(uint32_t a, uint32_t b, uint32_t p)
{
    return a >= b ? a - b : p - (b - a);
}

/* Legacy remainder multiply retained for the rational root and for the
 * --alg-backend legacy comparison path. */
static inline uint32_t d_mod_mul(uint32_t a, uint32_t b, uint32_t p)
{
    return (uint32_t)(((uint64_t)a * b) % p);
}

/* Independent deterministic primality gate for ordinary q=p entries emitted
 * by the complete GPU builder.  This deliberately does not share the segmented
 * sieve or Montgomery root arithmetic: for 32-bit n, bases 2,7,61 are a
 * deterministic Miller-Rabin witness set. */
static inline bool d_mr32_pass(uint32_t n, uint32_t d,
                                             unsigned s, uint32_t a)
{
    if (a % n == 0u) return true;
    uint32_t x = 1u, b = a % n;
    uint32_t e = d;
    while (e) {
        if (e & 1u) x = d_mod_mul(x, b, n);
        e >>= 1;
        if (e) b = d_mod_mul(b, b, n);
    }
    if (x == 1u || x == n - 1u) return true;
    for (unsigned r = 1; r < s; r++) {
        x = d_mod_mul(x, x, n);
        if (x == n - 1u) return true;
    }
    return false;
}

static inline bool d_is_prime32_independent(uint32_t n)
{
    if (n < 2u) return false;
    if ((n & 1u) == 0u) return n == 2u;
    if (n % 3u == 0u) return n == 3u;
    uint32_t d = n - 1u;
    unsigned s = 0;
    while ((d & 1u) == 0u) { d >>= 1; s++; }
    return d_mr32_pass(n, d, s, 2u) &&
           d_mr32_pass(n, d, s, 7u) &&
           d_mr32_pass(n, d, s, 61u);
}

/* Per-prime 32-bit Montgomery context, R = 2^32.  `one` is R mod p (the
 * Montgomery representation of 1), and r2 is R^2 mod p in ordinary form.
 *
 * The setup deliberately contains no integer division.  For odd p, Newton's
 * iteration gives -p^-1 mod 2^32 in five multiplies.  Sixty-four modular
 * doublings produce R mod p and R^2 mod p.  That fixed setup cost replaces the
 * hundreds to thousands of uint64 %% p operations in the polynomial powers.
 */
typedef struct {
    uint32_t p;
    uint32_t n0inv;
    uint32_t one;
    uint32_t r2;
} mont32_t;

static inline mont32_t d_mont_setup(uint32_t p)
{
    mont32_t M;
    uint32_t inv = p;
    uint32_t x = 1;
    M.p = p;
    #pragma unroll
    for (int i = 0; i < 5; i++) inv *= 2u - p * inv;
    M.n0inv = 0u - inv;
    M.one = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        x = d_mod_add(x, x, p);
        if (i == 31) M.one = x;
    }
    M.r2 = x;
    return M;
}

/* REDC(a*b), with a allowed to be any 32-bit word and b < p.  The latter is
 * enough to keep a*b < R*p, so the Montgomery result is below 2p and one
 * subtraction suffices.  tlo + (tlo*n0inv mod R)*p is exactly 0 mod R; its
 * carry into the high word is therefore simply (tlo != 0), avoiding another
 * low-half multiply by p. */
static inline uint32_t d_mont_mul(uint32_t a, uint32_t b,
                                               const thread mont32_t *M)
{
    const uint32_t tlo = a * b;
    const uint32_t thi = mulhi(a, b);
    const uint32_t m = tlo * M->n0inv;
    const uint32_t mph = mulhi(m, M->p);
    uint32_t u = thi + mph;
    uint32_t carry = (u < thi);
    const uint32_t before = u;
    u += (tlo != 0u);
    carry |= (u < before);
    if (carry) return u - M->p;  /* represents (2^32 + u) - p */
    if (u >= M->p) u -= M->p;
    return u;
}

static inline uint32_t d_mont_from(uint32_t a, const thread mont32_t *M)
{
    return d_mont_mul(a, M->r2, M);
}

static inline uint32_t d_mont_to(uint32_t a, const thread mont32_t *M)
{
    return d_mont_mul(a, 1u, M);
}

/* Reduce an arbitrary signed base-2^32 integer directly into Montgomery form.
 * If r represents the prefix value times R, appending one 32-bit limb w gives
 * (prefix*R + w)*R.  Multiplication by r2 advances the prefix by one radix and
 * multiplication of w by r2 converts that word to Montgomery form. */
static inline uint32_t d_big_mod_mont(const constant gpu_big_t *a,
                                                   const thread mont32_t *M)
{
    uint32_t r = 0;
    for (int i = a->n - 1; i >= 0; i--) {
        r = d_mont_mul(r, M->r2, M);
        r = d_mod_add(r, d_mont_mul(a->v[i], M->r2, M), M->p);
    }
    if (a->neg && r) r = M->p - r;
    return r;
}

template <bool MONT>
static inline uint32_t d_ctx_mul(uint32_t a, uint32_t b,
                                              const thread mont32_t *M)
{
    if constexpr (MONT) return d_mont_mul(a, b, M);
    else return (uint32_t)(((uint64_t)a * b) % M->p);
}

template <bool MONT>
static inline uint32_t d_ctx_from(uint32_t a, const thread mont32_t *M)
{
    if constexpr (MONT) return d_mont_from(a, M);
    else return a;
}

template <bool MONT>
static inline uint32_t d_ctx_to(uint32_t a, const thread mont32_t *M)
{
    if constexpr (MONT) return d_mont_to(a, M);
    else return a;
}

template <bool MONT>
static inline uint32_t d_ctx_big(const constant gpu_big_t *a,
                                              const thread mont32_t *M)
{
    if constexpr (MONT) return d_big_mod_mont(a, M);
    else return d_big_mod(a, M->p);
}

template <bool MONT>
static inline uint32_t d_ctx_one(const thread mont32_t *M)
{
    if constexpr (MONT) return M->one;
    else return 1u;
}

/* Binary extended GCD for an odd prime modulus.  This avoids a p-2 power just
 * to make a polynomial monic.  All ordinary coefficient state stays in [0,p). */
static inline uint32_t d_mod_inv(uint32_t a, uint32_t p)
{
    if (a == 0) return 0;
    if (p == 2) return 1;
    uint32_t u = a, v = p, x1 = 1, x2 = 0;
    while (u != 1 && v != 1) {
        while (!(u & 1u)) {
            u >>= 1;
            x1 = !(x1 & 1u) ? (x1 >> 1)
                              : (uint32_t)(((uint64_t)x1 + p) >> 1);
        }
        while (!(v & 1u)) {
            v >>= 1;
            x2 = !(x2 & 1u) ? (x2 >> 1)
                              : (uint32_t)(((uint64_t)x2 + p) >> 1);
        }
        if (u >= v) {
            u -= v;
            x1 = x1 >= x2 ? x1 - x2 : (uint32_t)((uint64_t)x1 + p - x2);
        } else {
            v -= u;
            x2 = x2 >= x1 ? x2 - x1 : (uint32_t)((uint64_t)x2 + p - x1);
        }
    }
    return u == 1 ? x1 : x2;
}

static inline void d_poly_zero(thread dpoly_t *a)
{
    a->deg = -1;
    #pragma unroll
    for (int i = 0; i < BENCH_NCOEFF; i++) a->c[i] = 0;
}

static inline void d_poly_norm(thread dpoly_t *a)
{
    while (a->deg >= 0 && a->c[a->deg] == 0) a->deg--;
}

template <bool MONT>
static inline void d_poly_make_monic(thread dpoly_t *a, const thread mont32_t *M)
{
    const uint32_t one = d_ctx_one<MONT>(M);
    if (a->deg < 0 || a->c[a->deg] == one) return;
    uint32_t lead = d_ctx_to<MONT>(a->c[a->deg], M);
    uint32_t z = d_ctx_from<MONT>(d_mod_inv(lead, M->p), M);
    for (int i = 0; i <= a->deg; i++) a->c[i] = d_ctx_mul<MONT>(a->c[i], z, M);
}

/* b must be monic. */
template <bool MONT>
static inline dpoly_t d_poly_rem_monic(dpoly_t a, const thread dpoly_t *b,
                                    const thread mont32_t *M)
{
    while (a.deg >= b->deg) {
        uint32_t q = a.c[a.deg];
        int shift = a.deg - b->deg;
        /* b is monic, so the leading cancellation is known exactly.  Do not
         * spend a modular multiply computing q*1 only to subtract q from q. */
        for (int i = 0; i < b->deg; i++)
            a.c[i + shift] = d_mod_sub(a.c[i + shift],
                                       d_ctx_mul<MONT>(q, b->c[i], M), M->p);
        a.c[a.deg] = 0;
        d_poly_norm(&a);
    }
    return a;
}

/* b must be monic and divide a exactly. */
template <bool MONT>
static inline dpoly_t d_poly_divexact_monic(dpoly_t a, const thread dpoly_t *b,
                                         const thread mont32_t *M, thread uint32_t *failed)
{
    dpoly_t q;
    d_poly_zero(&q);
    while (a.deg >= b->deg) {
        uint32_t z = a.c[a.deg];
        int shift = a.deg - b->deg;
        q.c[shift] = z;
        if (shift > q.deg) q.deg = shift;
        for (int i = 0; i < b->deg; i++)
            a.c[i + shift] = d_mod_sub(a.c[i + shift],
                                       d_ctx_mul<MONT>(z, b->c[i], M), M->p);
        a.c[a.deg] = 0;
        d_poly_norm(&a);
    }
    if (a.deg >= 0) *failed = 1;
    d_poly_norm(&q);
    return q;
}

template <bool MONT>
static inline dpoly_t d_poly_gcd(dpoly_t a, dpoly_t b, const thread mont32_t *M)
{
    if (a.deg >= 0) d_poly_make_monic<MONT>(&a, M);
    if (b.deg >= 0) d_poly_make_monic<MONT>(&b, M);
    while (b.deg >= 0) {
        dpoly_t r = d_poly_rem_monic<MONT>(a, &b, M);
        a = b;
        b = r;
        if (b.deg >= 0) d_poly_make_monic<MONT>(&b, M);
    }
    return a;
}

/* mod must be monic. */
template <bool MONT>
static inline dpoly_t d_poly_mul_rem(const thread dpoly_t *a, const thread dpoly_t *b,
                                  const thread dpoly_t *mod, const thread mont32_t *M)
{
    uint32_t t[2 * BENCH_MAX_DEGREE + 1];
    dpoly_t r;
    #pragma unroll
    for (int i = 0; i < 2 * BENCH_MAX_DEGREE + 1; i++) t[i] = 0;
    int tdeg = -1;
    for (int i = 0; i <= a->deg; i++) for (int j = 0; j <= b->deg; j++) {
        uint32_t z = d_ctx_mul<MONT>(a->c[i], b->c[j], M);
        t[i + j] = d_mod_add(t[i + j], z, M->p);
        if (t[i + j] && i + j > tdeg) tdeg = i + j;
    }
    while (tdeg >= mod->deg) {
        uint32_t q = t[tdeg];
        int shift = tdeg - mod->deg;
        for (int i = 0; i < mod->deg; i++)
            t[i + shift] = d_mod_sub(t[i + shift],
                                     d_ctx_mul<MONT>(q, mod->c[i], M), M->p);
        t[tdeg] = 0;
        while (tdeg >= 0 && !t[tdeg]) tdeg--;
    }
    d_poly_zero(&r);
    r.deg = tdeg;
    for (int i = 0; i <= tdeg; i++) r.c[i] = t[i];
    return r;
}

/* Polynomial squaring is the dominant operation in x^p mod F.  A generic
 * product computes both a_i*a_j and a_j*a_i; exploit symmetry so a degree-d
 * square uses (d+1)(d+2)/2 modular multiplies instead of (d+1)^2. */
template <bool MONT>
static inline dpoly_t d_poly_sqr_rem(const thread dpoly_t *a, const thread dpoly_t *mod,
                                  const thread mont32_t *M)
{
    uint32_t t[2 * BENCH_MAX_DEGREE + 1];
    dpoly_t r;
    #pragma unroll
    for (int i = 0; i < 2 * BENCH_MAX_DEGREE + 1; i++) t[i] = 0;
    int tdeg = -1;
    for (int i = 0; i <= a->deg; i++) {
        uint32_t z = d_ctx_mul<MONT>(a->c[i], a->c[i], M);
        t[2 * i] = d_mod_add(t[2 * i], z, M->p);
        if (t[2 * i] && 2 * i > tdeg) tdeg = 2 * i;
        for (int j = i + 1; j <= a->deg; j++) {
            z = d_ctx_mul<MONT>(a->c[i], a->c[j], M);
            z = d_mod_add(z, z, M->p);
            t[i + j] = d_mod_add(t[i + j], z, M->p);
            if (t[i + j] && i + j > tdeg) tdeg = i + j;
        }
    }
    while (tdeg >= mod->deg) {
        uint32_t q = t[tdeg];
        int shift = tdeg - mod->deg;
        for (int i = 0; i < mod->deg; i++)
            t[i + shift] = d_mod_sub(t[i + shift],
                                     d_ctx_mul<MONT>(q, mod->c[i], M), M->p);
        t[tdeg] = 0;
        while (tdeg >= 0 && !t[tdeg]) tdeg--;
    }
    d_poly_zero(&r);
    r.deg = tdeg;
    for (int i = 0; i <= tdeg; i++) r.c[i] = t[i];
    return r;
}

template <bool MONT, bool SYMMETRIC>
static inline dpoly_t d_poly_pow_rem(dpoly_t a, uint32_t e,
                                  const thread dpoly_t *mod, const thread mont32_t *M)
{
    dpoly_t r;
    d_poly_zero(&r);
    r.deg = 0;
    r.c[0] = d_ctx_one<MONT>(M);
    a = d_poly_rem_monic<MONT>(a, mod, M);
    while (e) {
        if (e & 1u) r = d_poly_mul_rem<MONT>(&r, &a, mod, M);
        e >>= 1;
        if (e) {
            if constexpr (SYMMETRIC) a = d_poly_sqr_rem<MONT>(&a, mod, M);
            else a = d_poly_mul_rem<MONT>(&a, &a, mod, M);
        }
    }
    return r;
}

template <bool MONT>
static inline uint32_t d_poly_eval(const thread dpoly_t *f, uint32_t x,
                                const thread mont32_t *M)
{
    uint32_t r = 0;
    uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 0; i--)
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), f->c[i], M->p);
    return r;
}

/* Split a squarefree product of linear factors without device recursion. */
template <bool MONT, bool SYMMETRIC>
static inline int d_split_linear(const thread dpoly_t *input, const thread mont32_t *M,
                              thread uint32_t *roots, thread uint32_t *failed)
{
    dpoly_t stack[BENCH_MAX_DEGREE];
    int sp = 0, nr = 0;
    if (input->deg <= 0) return 0;
    stack[sp++] = *input;
    while (sp) {
        dpoly_t f = stack[--sp];
        if (f.deg == 1) {
            /* Every factor placed on this stack is monic. */
            uint32_t c0 = d_ctx_to<MONT>(f.c[0], M);
            roots[nr++] = c0 ? M->p - c0 : 0;
            continue;
        }
        dpoly_t g;
        uint32_t a;
        int found = 0;
        for (a = 0; a < M->p; a++) {
            dpoly_t lin, h;
            d_poly_zero(&lin);
            lin.deg = 1;
            lin.c[0] = d_ctx_from<MONT>(a, M);
            lin.c[1] = d_ctx_one<MONT>(M);
            h = d_poly_pow_rem<MONT, SYMMETRIC>(lin, (M->p - 1) / 2, &f, M);
            if (h.deg < 0) { h.deg = 0; h.c[0] = 0; }
            h.c[0] = d_mod_sub(h.c[0], d_ctx_one<MONT>(M), M->p);
            d_poly_norm(&h);
            g = d_poly_gcd<MONT>(f, h, M);
            if (g.deg > 0 && g.deg < f.deg) { found = 1; break; }
        }
        if (!found || sp + 2 > BENCH_MAX_DEGREE) {
            *failed = 1;
            return nr;
        }
        dpoly_t other = d_poly_divexact_monic<MONT>(f, &g, M, failed);
        if (*failed) return nr;
        d_poly_make_monic<MONT>(&g, M);
        d_poly_make_monic<MONT>(&other, M);
        stack[sp++] = g;
        stack[sp++] = other;
    }
    return nr;
}

template <bool MONT, bool SYMMETRIC>
static inline int d_alg_roots_prime(uint32_t p, thread uint32_t *roots, thread uint32_t *failed,
                                          constant const gpu_big_t *c_alg, int c_alg_deg)
{
    dpoly_t f;
    mont32_t M;
    M.p = p;
    M.n0inv = M.one = M.r2 = 0;
    if constexpr (MONT) M = d_mont_setup(p);

    uint32_t lead = d_ctx_big<MONT>(&c_alg[c_alg_deg], &M);
    int nr = 0;
    d_poly_zero(&f);
    f.deg = c_alg_deg;
    for (int i = 0; i <= c_alg_deg; i++) f.c[i] = d_ctx_big<MONT>(&c_alg[i], &M);
    d_poly_norm(&f);

    if (f.deg < 0) {
        /* A primitive input polynomial should never vanish identically mod p. */
        *failed = 1;
        return 0;
    }
    if (f.deg == 1) {
        uint32_t c0 = d_ctx_to<MONT>(f.c[0], &M);
        uint32_t c1 = d_ctx_to<MONT>(f.c[1], &M);
        uint32_t inv = d_mod_inv(c1, p);
        uint32_t z = d_ctx_mul<MONT>(d_ctx_from<MONT>(c0 ? p - c0 : 0, &M),
                                     d_ctx_from<MONT>(inv, &M), &M);
        roots[nr++] = d_ctx_to<MONT>(z, &M);
    } else if (p <= GPU_FB_BRUTE_ROOT_LIMIT) {
        for (uint32_t x = 0; x < p; x++)
            if (d_poly_eval<MONT>(&f, x, &M) == 0) roots[nr++] = x;
    } else {
        dpoly_t x, xp, linear;
        d_poly_make_monic<MONT>(&f, &M);
        d_poly_zero(&x);
        x.deg = 1;
        x.c[1] = d_ctx_one<MONT>(&M);
        xp = d_poly_pow_rem<MONT, SYMMETRIC>(x, p, &f, &M);
        if (xp.deg < 1) xp.deg = 1;
        xp.c[1] = d_mod_sub(xp.c[1], d_ctx_one<MONT>(&M), p);
        d_poly_norm(&xp);
        linear = d_poly_gcd<MONT>(f, xp, &M);
        nr = d_split_linear<MONT, SYMMETRIC>(&linear, &M, roots, failed);
        if (*failed) return nr;
    }

    /* Homogeneous projective root.  At the prime level this exists exactly
     * when the original leading coefficient vanishes mod p. */
    if (lead == 0) roots[nr++] = p;
    if (nr > GPU_FB_MAX_ROOTS) {
        *failed = 1;
        return 0;
    }
    /* Deterministic output, matching fbgen's sorted affine roots and placing
     * the projective encoding p last naturally. */
    for (int i = 1; i < nr; i++) {
        uint32_t v = roots[i];
        int j = i;
        while (j && roots[j - 1] > v) { roots[j] = roots[j - 1]; j--; }
        roots[j] = v;
    }
    return nr;
}


/* Fixed-capacity algebraic root path.
 *
 * The original generic dpoly_t path is intentionally retained below as the
 * reference/fallback implementation.  This path instantiates the polynomial
 * capacity at compile time (currently 6 and 8) and avoids the generic
 * dpoly_t stack[BENCH_MAX_DEGREE] used by d_split_linear().  Instead it isolates
 * one linear factor, emits that root, divides it from the remaining squarefree
 * product, and repeats.  That trades a little repeated splitting in uncommon
 * many-root cases for substantially less per-thread local state.
 *
 * Access helpers deliberately turn dynamic coefficient indices into unrolled
 * constant-index selects.  This gives ptxas a chance to scalarize the small
 * coefficient vectors rather than materializing them as local-memory arrays.
 */
template <int CAP>
struct fpoly_t {
    int deg;
    uint32_t c[CAP + 1];
};

template <int CAP>
static inline uint32_t fp_get(const thread fpoly_t<CAP> *a, int idx)
{
    uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (idx == i) v = a->c[i];
    return v;
}

template <int CAP>
static inline void fp_set(thread fpoly_t<CAP> *a, int idx, uint32_t v)
{
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (idx == i) a->c[i] = v;
}

template <int CAP>
static inline void fp_zero(thread fpoly_t<CAP> *a)
{
    a->deg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) a->c[i] = 0;
}

template <int CAP>
static inline void fp_copy(thread fpoly_t<CAP> *dst, const thread fpoly_t<CAP> *src)
{
    dst->deg = src->deg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) dst->c[i] = src->c[i];
}

template <int CAP>
static inline void fp_norm(thread fpoly_t<CAP> *a)
{
    while (a->deg >= 0 && fp_get(a, a->deg) == 0) a->deg--;
}

template <int CAP, bool MONT>
static inline void fp_make_monic(thread fpoly_t<CAP> *a, const thread mont32_t *M)
{
    const uint32_t one = d_ctx_one<MONT>(M);
    if (a->deg < 0) return;
    const uint32_t lc = fp_get(a, a->deg);
    if (lc == one) return;
    const uint32_t lead = d_ctx_to<MONT>(lc, M);
    const uint32_t z = d_ctx_from<MONT>(d_mod_inv(lead, M->p), M);
    #pragma unroll
    for (int i = 0; i <= CAP; i++)
        if (i <= a->deg) a->c[i] = d_ctx_mul<MONT>(a->c[i], z, M);
}

template <int CAP, bool MONT>
static inline void fp_rem_monic(thread fpoly_t<CAP> *out,
                                              const thread fpoly_t<CAP> *ain,
                                              const thread fpoly_t<CAP> *b,
                                              const thread mont32_t *M)
{
    fpoly_t<CAP> a;
    fp_copy(&a, ain);
    while (a.deg >= b->deg) {
        const uint32_t q = fp_get(&a, a.deg);
        const int shift = a.deg - b->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < b->deg) {
                const int k = i + shift;
                const uint32_t av = fp_get(&a, k);
                const uint32_t bv = b->c[i];
                fp_set(&a, k, d_mod_sub(av, d_ctx_mul<MONT>(q, bv, M), M->p));
            }
        }
        fp_set(&a, a.deg, 0);
        fp_norm(&a);
    }
    fp_copy(out, &a);
}

template <int CAP, bool MONT>
static inline void fp_divexact_monic(thread fpoly_t<CAP> *qout,
                                                   const thread fpoly_t<CAP> *ain,
                                                   const thread fpoly_t<CAP> *b,
                                                   const thread mont32_t *M,
                                                   thread uint32_t *failed)
{
    fpoly_t<CAP> a, q;
    fp_copy(&a, ain);
    fp_zero(&q);
    while (a.deg >= b->deg) {
        const uint32_t z = fp_get(&a, a.deg);
        const int shift = a.deg - b->deg;
        fp_set(&q, shift, z);
        if (shift > q.deg) q.deg = shift;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < b->deg) {
                const int k = i + shift;
                const uint32_t av = fp_get(&a, k);
                fp_set(&a, k, d_mod_sub(av,
                                        d_ctx_mul<MONT>(z, b->c[i], M), M->p));
            }
        }
        fp_set(&a, a.deg, 0);
        fp_norm(&a);
    }
    if (a.deg >= 0) *failed = 1;
    fp_norm(&q);
    fp_copy(qout, &q);
}

template <int CAP, bool MONT>
static inline void fp_gcd(thread fpoly_t<CAP> *out,
                                        const thread fpoly_t<CAP> *ain,
                                        const thread fpoly_t<CAP> *bin,
                                        const thread mont32_t *M)
{
    fpoly_t<CAP> a, b, r;
    fp_copy(&a, ain);
    fp_copy(&b, bin);
    if (a.deg >= 0) fp_make_monic<CAP, MONT>(&a, M);
    if (b.deg >= 0) fp_make_monic<CAP, MONT>(&b, M);
    while (b.deg >= 0) {
        fp_rem_monic<CAP, MONT>(&r, &a, &b, M);
        fp_copy(&a, &b);
        fp_copy(&b, &r);
        if (b.deg >= 0) fp_make_monic<CAP, MONT>(&b, M);
    }
    fp_copy(out, &a);
}

template <int CAP>
struct fwide_t {
    uint32_t c[2 * CAP + 1];
};

template <int CAP>
static inline uint32_t fw_get(const thread fwide_t<CAP> *a, int idx)
{
    uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) if (idx == i) v = a->c[i];
    return v;
}

template <int CAP>
static inline void fw_set(thread fwide_t<CAP> *a, int idx, uint32_t v)
{
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) if (idx == i) a->c[i] = v;
}

template <int CAP, bool MONT>
static inline void fp_mul_rem(thread fpoly_t<CAP> *out,
                                            const thread fpoly_t<CAP> *a,
                                            const thread fpoly_t<CAP> *b,
                                            const thread fpoly_t<CAP> *mod,
                                            const thread mont32_t *M)
{
    fwide_t<CAP> t;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) t.c[i] = 0;

    int tdeg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) {
        if (i <= a->deg) {
            #pragma unroll
            for (int j = 0; j <= CAP; j++) {
                if (j <= b->deg) {
                    const uint32_t z = d_ctx_mul<MONT>(a->c[i], b->c[j], M);
                    t.c[i + j] = d_mod_add(t.c[i + j], z, M->p);
                    if (t.c[i + j] && i + j > tdeg) tdeg = i + j;
                }
            }
        }
    }

    while (tdeg >= mod->deg) {
        const uint32_t q = fw_get(&t, tdeg);
        const int shift = tdeg - mod->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < mod->deg) {
                const int k = i + shift;
                const uint32_t tv = fw_get(&t, k);
                fw_set(&t, k, d_mod_sub(tv,
                                        d_ctx_mul<MONT>(q, mod->c[i], M), M->p));
            }
        }
        fw_set(&t, tdeg, 0);
        do { tdeg--; } while (tdeg >= 0 && fw_get(&t, tdeg) == 0);
    }

    fp_zero(out);
    out->deg = tdeg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (i <= tdeg) out->c[i] = t.c[i];
}

template <int CAP, bool MONT>
static inline void fp_sqr_rem(thread fpoly_t<CAP> *out,
                                            const thread fpoly_t<CAP> *a,
                                            const thread fpoly_t<CAP> *mod,
                                            const thread mont32_t *M)
{
    fwide_t<CAP> t;
    #pragma unroll
    for (int i = 0; i <= 2 * CAP; i++) t.c[i] = 0;

    int tdeg = -1;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) {
        if (i <= a->deg) {
            uint32_t z = d_ctx_mul<MONT>(a->c[i], a->c[i], M);
            t.c[2 * i] = d_mod_add(t.c[2 * i], z, M->p);
            if (t.c[2 * i] && 2 * i > tdeg) tdeg = 2 * i;
            #pragma unroll
            for (int j = i + 1; j <= CAP; j++) {
                if (j <= a->deg) {
                    z = d_ctx_mul<MONT>(a->c[i], a->c[j], M);
                    z = d_mod_add(z, z, M->p);
                    t.c[i + j] = d_mod_add(t.c[i + j], z, M->p);
                    if (t.c[i + j] && i + j > tdeg) tdeg = i + j;
                }
            }
        }
    }

    while (tdeg >= mod->deg) {
        const uint32_t q = fw_get(&t, tdeg);
        const int shift = tdeg - mod->deg;
        #pragma unroll
        for (int i = 0; i <= CAP; i++) {
            if (i < mod->deg) {
                const int k = i + shift;
                const uint32_t tv = fw_get(&t, k);
                fw_set(&t, k, d_mod_sub(tv,
                                        d_ctx_mul<MONT>(q, mod->c[i], M), M->p));
            }
        }
        fw_set(&t, tdeg, 0);
        do { tdeg--; } while (tdeg >= 0 && fw_get(&t, tdeg) == 0);
    }

    fp_zero(out);
    out->deg = tdeg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++) if (i <= tdeg) out->c[i] = t.c[i];
}

template <int CAP, bool MONT, bool SYMMETRIC>
static inline void fp_pow_rem(thread fpoly_t<CAP> *out,
                                            const thread fpoly_t<CAP> *ain,
                                            uint32_t e,
                                            const thread fpoly_t<CAP> *mod,
                                            const thread mont32_t *M)
{
    fpoly_t<CAP> r, a, tmp;
    fp_zero(&r);
    r.deg = 0;
    r.c[0] = d_ctx_one<MONT>(M);
    fp_rem_monic<CAP, MONT>(&a, ain, mod, M);
    while (e) {
        if (e & 1u) {
            fp_mul_rem<CAP, MONT>(&tmp, &r, &a, mod, M);
            fp_copy(&r, &tmp);
        }
        e >>= 1;
        if (e) {
            if constexpr (SYMMETRIC) fp_sqr_rem<CAP, MONT>(&tmp, &a, mod, M);
            else fp_mul_rem<CAP, MONT>(&tmp, &a, &a, mod, M);
            fp_copy(&a, &tmp);
        }
    }
    fp_copy(out, &r);
}

template <int CAP, bool MONT>
static inline uint32_t fp_eval(const thread fpoly_t<CAP> *f, uint32_t x,
                                             const thread mont32_t *M)
{
    uint32_t r = 0;
    const uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 0; i--)
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), fp_get(f, i), M->p);
    return r;
}

template <int CAP, bool MONT>
static inline uint32_t fp_deriv_eval(const thread fpoly_t<CAP> *f,
                                                   uint32_t x,
                                                   const thread mont32_t *M)
{
    uint32_t r = 0;
    const uint32_t xx = d_ctx_from<MONT>(x, M);
    for (int i = f->deg; i >= 1; i--) {
        const uint32_t ii = d_ctx_from<MONT>((uint32_t)i, M);
        const uint32_t term = d_ctx_mul<MONT>(fp_get(f, i), ii, M);
        r = d_mod_add(d_ctx_mul<MONT>(r, xx, M), term, M->p);
    }
    return r;
}

/* Find any proper factor of a squarefree product of linear factors. */
template <int CAP, bool MONT, bool SYMMETRIC>
static inline int fp_find_split(thread fpoly_t<CAP> *gout,
                                              const thread fpoly_t<CAP> *f,
                                              const thread mont32_t *M)
{
    for (uint32_t a = 0; a < M->p; a++) {
        fpoly_t<CAP> lin, h, g;
        fp_zero(&lin);
        lin.deg = 1;
        lin.c[0] = d_ctx_from<MONT>(a, M);
        lin.c[1] = d_ctx_one<MONT>(M);
        fp_pow_rem<CAP, MONT, SYMMETRIC>(&h, &lin, (M->p - 1u) / 2u, f, M);
        if (h.deg < 0) { h.deg = 0; h.c[0] = 0; }
        h.c[0] = d_mod_sub(h.c[0], d_ctx_one<MONT>(M), M->p);
        fp_norm(&h);
        fp_gcd<CAP, MONT>(&g, f, &h, M);
        if (g.deg > 0 && g.deg < f->deg) {
            fp_copy(gout, &g);
            return 1;
        }
    }
    return 0;
}

/* Stack-free root extraction for the fixed-capacity path.  `remaining` always
 * contains every not-yet-emitted root.  To get the next root, repeatedly take
 * one proper factor until a linear factor remains, emit it, and divide that
 * linear factor from `remaining`.  The common case has zero or one root, so it
 * pays no repeated-splitting penalty while avoiding CAP polynomial stack slots
 * for every thread. */
template <int CAP, bool MONT, bool SYMMETRIC>
static inline int fp_split_linear(thread fpoly_t<CAP> *input,
                                                const thread mont32_t *M,
                                                thread uint32_t *roots,
                                                thread uint32_t *failed)
{
    fpoly_t<CAP> remaining, target, g, lin, next;
    int nr = 0;
    if (input->deg <= 0) return 0;
    fp_copy(&remaining, input);
    fp_make_monic<CAP, MONT>(&remaining, M);

    while (remaining.deg > 0) {
        fp_copy(&target, &remaining);
        while (target.deg > 1) {
            if (!fp_find_split<CAP, MONT, SYMMETRIC>(&g, &target, M)) {
                *failed = 1;
                return nr;
            }
            fp_make_monic<CAP, MONT>(&g, M);
            fp_copy(&target, &g);
        }

        const uint32_t c0 = d_ctx_to<MONT>(target.c[0], M);
        const uint32_t root = c0 ? M->p - c0 : 0;
        roots[nr++] = root;
        if (nr > CAP) { *failed = 1; return 0; }

        fp_zero(&lin);
        lin.deg = 1;
        lin.c[0] = d_ctx_from<MONT>(root ? M->p - root : 0, M);
        lin.c[1] = d_ctx_one<MONT>(M);
        fp_divexact_monic<CAP, MONT>(&next, &remaining, &lin, M, failed);
        if (*failed) return nr;
        fp_copy(&remaining, &next);
        if (remaining.deg >= 0) fp_make_monic<CAP, MONT>(&remaining, M);
    }
    return nr;
}

template <int CAP, bool MONT, bool SYMMETRIC>
static inline int d_alg_roots_prime_fixed(uint32_t p,
                                                        thread uint32_t *roots,
                                                        thread uint32_t *failed,
                                          constant const gpu_big_t *c_alg, int c_alg_deg,
                                                        thread uint8_t *special = nullptr)
{
    fpoly_t<CAP> f;
    mont32_t M;
    M.p = p;
    M.n0inv = M.one = M.r2 = 0;
    if constexpr (MONT) M = d_mont_setup(p);

    const uint32_t lead = d_ctx_big<MONT>(&c_alg[c_alg_deg], &M);
    int nr = 0;
    fp_zero(&f);
    f.deg = c_alg_deg;
    #pragma unroll
    for (int i = 0; i <= CAP; i++)
        if (i <= c_alg_deg) f.c[i] = d_ctx_big<MONT>(&c_alg[i], &M);
    fp_norm(&f);

    if (f.deg < 0) {
        *failed = 1;
        return 0;
    }
    if (f.deg == 1) {
        const uint32_t c0 = d_ctx_to<MONT>(f.c[0], &M);
        const uint32_t c1 = d_ctx_to<MONT>(f.c[1], &M);
        const uint32_t inv = d_mod_inv(c1, p);
        const uint32_t z = d_ctx_mul<MONT>(
            d_ctx_from<MONT>(c0 ? p - c0 : 0, &M), d_ctx_from<MONT>(inv, &M), &M);
        roots[nr++] = d_ctx_to<MONT>(z, &M);
    } else if (p <= GPU_FB_BRUTE_ROOT_LIMIT) {
        for (uint32_t x = 0; x < p; x++)
            if (fp_eval<CAP, MONT>(&f, x, &M) == 0) roots[nr++] = x;
    } else {
        fpoly_t<CAP> x, xp, linear;
        fp_make_monic<CAP, MONT>(&f, &M);
        fp_zero(&x);
        x.deg = 1;
        x.c[1] = d_ctx_one<MONT>(&M);
        fp_pow_rem<CAP, MONT, SYMMETRIC>(&xp, &x, p, &f, &M);
        if (xp.deg < 1) xp.deg = 1;
        xp.c[1] = d_mod_sub(xp.c[1], d_ctx_one<MONT>(&M), p);
        fp_norm(&xp);
        fp_gcd<CAP, MONT>(&linear, &f, &xp, &M);
        nr = fp_split_linear<CAP, MONT, SYMMETRIC>(&linear, &M, roots, failed);
        if (*failed) return nr;
    }

    if (lead == 0) roots[nr++] = p;
    if (nr > GPU_FB_MAX_ROOTS) {
        *failed = 1;
        return 0;
    }
    if (special) {
        uint8_t sp = lead == 0;
        /* A repeated affine root is exactly the case where the q=p entry can
         * carry a valuation increment other than (1,0).  Projective roots are
         * exceptional for the same reason.  Mark only those primes; the host
         * then reruns the native exact p-adic code for a tiny exception set. */
        for (int i = 0; i < nr && !sp; i++)
            if (roots[i] < p && fp_deriv_eval<CAP, MONT>(&f, roots[i], &M) == 0)
                sp = 1;
        *special = sp;
    }
    for (int i = 1; i < nr; i++) {
        const uint32_t v = roots[i];
        int j = i;
        while (j && roots[j - 1] > v) { roots[j] = roots[j - 1]; j--; }
        roots[j] = v;
    }
    return nr;
}

template <int CAP, bool MONT, bool SYMMETRIC>
static inline void k_alg_roots_fixed_body(
    device const uint32_t *primes,
    constant uint32_t &n,
    device uint32_t *rootbuf,
    device uint32_t *counts,
    device uint32_t *failures,
    constant const gpu_big_t *c_alg,
    constant int &c_alg_deg,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        const uint32_t p = primes[t];
        const int nr = (MONT && p == 2u)
            ? d_alg_roots_prime_fixed<CAP, false, SYMMETRIC>(p, roots, &bad, c_alg, c_alg_deg)
            : d_alg_roots_prime_fixed<CAP, MONT, SYMMETRIC>(p, roots, &bad, c_alg, c_alg_deg);
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = i < nr ? roots[i] : 0u;
    }
}

kernel void k_alg_roots_fixed_6_1_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_body<6, true, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_fixed_8_1_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_body<8, true, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_fixed_6_0_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_body<6, false, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_fixed_8_0_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_body<8, false, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


template <int CAP, bool MONT>
static inline void k_alg_roots_fixed_mark_body(
    device const uint32_t *primes,
    constant uint32_t &n,
    device uint32_t *rootbuf,
    device uint32_t *counts,
    device uint8_t *special,
    device uint32_t *failures,
    constant const gpu_big_t *c_alg,
    constant int &c_alg_deg,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        uint8_t sp = 0;
        const uint32_t p = primes[t];
        const int nr = (MONT && p == 2u)
            ? d_alg_roots_prime_fixed<CAP, false, false>(p, roots, &bad, c_alg, c_alg_deg, &sp)
            : d_alg_roots_prime_fixed<CAP, MONT, false>(p, roots, &bad, c_alg, c_alg_deg, &sp);
        /* A q=p record is trusted by FB_VALIDATE_GENERATED_PRIME_POWERS, so do
         * not let the same GPU sieve that proposed p be its only primality
         * proof.  Check only candidates that would actually emit an ideal. */
        if (!bad && nr && !d_is_prime32_independent(p)) bad = 1u;
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        special[t] = bad ? 1u : sp;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = (!bad && i < nr) ? roots[i] : 0u;
    }
}

kernel void k_alg_roots_fixed_mark_6_1(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint8_t *special [[buffer(4)]],
    device uint32_t *failures [[buffer(5)]],
    constant const gpu_big_t *c_alg [[buffer(6)]],
    constant int &c_alg_deg [[buffer(7)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_mark_body<6, true>(primes, n, rootbuf, counts, special, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_fixed_mark_8_1(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint8_t *special [[buffer(4)]],
    device uint32_t *failures [[buffer(5)]],
    constant const gpu_big_t *c_alg [[buffer(6)]],
    constant int &c_alg_deg [[buffer(7)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_fixed_mark_body<8, true>(primes, n, rootbuf, counts, special, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


kernel void k_make_odds(
    constant uint32_t &lo [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *values [[buffer(2)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) values[t] = (uint32_t)((uint64_t)lo + 2ull * t);
}

kernel void k_mark_composites(
    constant uint32_t &lo [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device const uint32_t *base [[buffer(2)]],
    constant uint32_t &nbase [[buffer(3)]],
    device uint8_t *isprime [[buffer(4)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    /* One block per base prime.  A one-thread-per-prime sieve makes p=3 one
     * serial thread write roughly a third of the entire segment; spreading a
     * prime's progression across the block is dramatically less pathological. */
    uint32_t k = blockIdx.x;
    if (k >= nbase) return;
    uint32_t p = base[k];
    if (p == 2) return;
    uint64_t hi = (uint64_t)lo + 2ull * (n - 1u);
    uint64_t first = (uint64_t)p * p;
    if (first < lo) first = ((uint64_t)lo + p - 1u) / p * p;
    if (!(first & 1u)) first += p;
    if (first > hi) return;
    uint64_t idx = ((first - lo) >> 1) + (uint64_t)threadIdx.x * p;
    uint64_t step = (uint64_t)p * blockDim.x;
    for (; idx < n; idx += step) isprime[idx] = 0;
}

template <bool MONT, bool SYMMETRIC>
static inline void k_alg_roots_body(
    device const uint32_t *primes,
    constant uint32_t &n,
    device uint32_t *rootbuf,
    device uint32_t *counts,
    device uint32_t *failures,
    constant const gpu_big_t *c_alg,
    constant int &c_alg_deg,
    uint _tid,
    uint _ntid,
    uint _bid,
    uint _lid,
    uint _bdim,
    uint _gdim)
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t roots[GPU_FB_MAX_ROOTS];
        uint32_t bad = 0;
        const uint32_t p = primes[t];
        /* Montgomery arithmetic requires an odd modulus; p=2 is one entry and
         * stays on the legacy path.  The branch is uniform for all but the
         * first lane of the first segment. */
        int nr = (MONT && p == 2u) ? d_alg_roots_prime<false, SYMMETRIC>(p, roots, &bad, c_alg, c_alg_deg)
                                   : d_alg_roots_prime<MONT, SYMMETRIC>(p, roots, &bad, c_alg, c_alg_deg);
        if (bad) atomicAdd(failures, 1u);
        counts[t] = bad ? 0u : (uint32_t)nr;
        #pragma unroll
        for (int i = 0; i < GPU_FB_MAX_ROOTS; i++)
            rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + i] = i < nr ? roots[i] : 0u;
    }
}

kernel void k_alg_roots_1_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_body<true, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_0_0(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_body<false, false>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_1_1(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_body<true, true>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}

kernel void k_alg_roots_0_1(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *rootbuf [[buffer(2)]],
    device uint32_t *counts [[buffer(3)]],
    device uint32_t *failures [[buffer(4)]],
    constant const gpu_big_t *c_alg [[buffer(5)]],
    constant int &c_alg_deg [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    k_alg_roots_body<false, true>(primes, n, rootbuf, counts, failures, c_alg, c_alg_deg, _tid, _ntid, _bid, _lid, _bdim, _gdim);
}


kernel void k_rational_roots(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device uint32_t *roots [[buffer(2)]],
    device uint32_t *failures [[buffer(3)]],
    constant const gpu_big_t *c_y0 [[buffer(4)]],
    constant const gpu_big_t *c_y1 [[buffer(5)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t p = primes[t];
        uint32_t y0 = d_big_mod(c_y0, p);
        uint32_t y1 = d_big_mod(c_y1, p);
        if (!y1) {
            if (!y0) { atomicAdd(failures, 1u); roots[t] = 0; }
            else roots[t] = p;
        } else {
            uint32_t inv = d_mod_inv(y1, p);
            roots[t] = d_mod_mul(y0 ? p - y0 : 0, inv, p);
        }
    }
}

kernel void k_total_roots(
    device const uint32_t *counts [[buffer(0)]],
    device const uint32_t *offsets [[buffer(1)]],
    constant uint32_t &n [[buffer(2)]],
    device uint32_t *total [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    if (blockIdx.x == 0 && threadIdx.x == 0)
        *total = n ? offsets[n - 1] + counts[n - 1] : 0;
}

kernel void k_scatter_alg(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device const uint32_t *rootbuf [[buffer(2)]],
    device const uint32_t *counts [[buffer(3)]],
    device const uint32_t *offsets [[buffer(4)]],
    device uint32_t *out_p [[buffer(5)]],
    device uint32_t *out_r [[buffer(6)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t off = offsets[t], nr = counts[t];
        for (uint32_t k = 0; k < nr; k++) {
            out_p[off + k] = primes[t];
            out_r[off + k] = rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + k];
        }
    }
}

kernel void k_scatter_alg_mark(
    device const uint32_t *primes [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    device const uint32_t *rootbuf [[buffer(2)]],
    device const uint32_t *counts [[buffer(3)]],
    device const uint32_t *offsets [[buffer(4)]],
    device const uint8_t *special [[buffer(5)]],
    device uint32_t *out_p [[buffer(6)]],
    device uint32_t *out_r [[buffer(7)]],
    device uint8_t *out_special [[buffer(8)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (; t < n; t += stride) {
        uint32_t off = offsets[t], nr = counts[t];
        for (uint32_t k = 0; k < nr; k++) {
            out_p[off + k] = primes[t];
            out_r[off + k] = rootbuf[(size_t)t * GPU_FB_MAX_ROOTS + k];
            out_special[off + k] = special[t];
        }
    }
}

kernel void k_upper_bound(
    device const uint32_t *v [[buffer(0)]],
    constant uint32_t &n [[buffer(1)]],
    constant uint32_t &lim [[buffer(2)]],
    device uint32_t *out [[buffer(3)]],
    uint _tid [[thread_position_in_grid]],
    uint _ntid [[threads_per_grid]],
    uint _bid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint _bdim [[threads_per_threadgroup]],
    uint _gdim [[threadgroups_per_grid]])
{
    FB_KERNEL_IDS

    if (blockIdx.x || threadIdx.x) return;
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (v[mid] <= lim) lo = mid + 1;
        else hi = mid;
    }
    *out = lo;
}
