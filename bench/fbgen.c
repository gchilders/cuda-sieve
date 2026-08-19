#define _GNU_SOURCE
/* Standalone algebraic factor-base generator.
 *
 * This emits the text format consumed by fb_load_cado(), but has no CADO or
 * GMP dependency.  The root finder works over uint32 prime fields; the exact
 * integer polynomial used while following ramified Hensel branches is kept in
 * a small fixed-width signed bigint.  With degree <= 8, 80-digit input
 * coefficients, lim < 2^32 and maxbits <= 31, 640 bits is ample:
 * a linear substitution can add at most degree*32 bits.
 *
 * The lifting recursion follows the ideal-valued representation used by
 * makefb.  In particular, an entry is not merely a root modulo p^k: nexp and
 * oldexp describe the marginal p-adic valuation contributed by that root.
 * This distinction matters at ramified roots and at projective prime powers.
 */
#include "bench.h"

#include <errno.h>
#include <limits.h>
#ifndef FBGEN_LIBRARY
#include <pthread.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FBGEN_MAX_DEG BENCH_MAX_DEGREE
#define BIG_LIMBS 20
#define BRUTE_ROOT_LIMIT 67u

typedef struct {
    uint32_t v[BIG_LIMBS];
    int n;                              /* used limbs; zero has n == 0 */
    int neg;
} sbig_t;

typedef struct {
    int deg;
    sbig_t c[FBGEN_MAX_DEG + 1];
} zpoly_t;

typedef struct {
    int deg;
    uint32_t c[2 * FBGEN_MAX_DEG + 2];
} mpoly_t;

typedef struct {
    uint32_t q, r;
    int n1, n0;
} fbgen_entry_t;

typedef struct {
    fbgen_entry_t *v;
    size_t n, cap;
} entry_vec_t;

#ifndef FBGEN_LIBRARY
static void die_oom(void)
{
    fprintf(stderr, "fbgen: out of memory\n");
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die_oom();
    return p;
}

static void *xrealloc(void *p, size_t n)
{
    p = realloc(p, n ? n : 1);
    if (!p) die_oom();
    return p;
}

#endif

/* ---- fixed-width signed integers ----------------------------------- */

static void big_norm(sbig_t *a)
{
    while (a->n && a->v[a->n - 1] == 0) a->n--;
    if (!a->n) a->neg = 0;
}

static int big_parse(sbig_t *a, const char *s)
{
    memset(a, 0, sizeof(*a));
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-' || *s == '+') { a->neg = (*s == '-'); s++; }
    for (; *s >= '0' && *s <= '9'; s++) {
        uint64_t carry = (uint32_t)(*s - '0');
        int i;
        for (i = 0; i < a->n; i++) {
            uint64_t t = (uint64_t)a->v[i] * 10u + carry;
            a->v[i] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) {
            if (a->n == BIG_LIMBS) return -1;
            a->v[a->n++] = (uint32_t)carry;
        }
    }
    big_norm(a);
    return 0;
}

static int big_abs_cmp(const sbig_t *a, const sbig_t *b)
{
    int i;
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (i = a->n - 1; i >= 0; i--)
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    return 0;
}

static void big_abs_add(sbig_t *r, const sbig_t *a, const sbig_t *b)
{
    uint64_t carry = 0;
    int i, n = a->n > b->n ? a->n : b->n;
    sbig_t t;
    memset(&t, 0, sizeof(t));
    for (i = 0; i < n; i++) {
        uint64_t z = carry;
        if (i < a->n) z += a->v[i];
        if (i < b->n) z += b->v[i];
        t.v[i] = (uint32_t)z;
        carry = z >> 32;
    }
    if (carry) {
        if (n == BIG_LIMBS) { fprintf(stderr, "fbgen: bigint overflow\n"); exit(EXIT_FAILURE); }
        t.v[n++] = (uint32_t)carry;
    }
    t.n = n;
    *r = t;
}

/* |r| = |a|-|b|, with |a| >= |b|. */
static void big_abs_sub(sbig_t *r, const sbig_t *a, const sbig_t *b)
{
    uint64_t borrow = 0;
    int i;
    sbig_t t;
    memset(&t, 0, sizeof(t));
    for (i = 0; i < a->n; i++) {
        uint64_t av = a->v[i], bv = (i < b->n ? b->v[i] : 0u) + borrow;
        t.v[i] = (uint32_t)(av - bv);
        borrow = av < bv;
    }
    t.n = a->n;
    big_norm(&t);
    *r = t;
}

static void big_add(sbig_t *r, const sbig_t *a, const sbig_t *b)
{
    sbig_t t;
    if (!a->n) { *r = *b; return; }
    if (!b->n) { *r = *a; return; }
    if (a->neg == b->neg) {
        big_abs_add(&t, a, b); t.neg = a->neg;
    } else {
        int c = big_abs_cmp(a, b);
        if (!c) { memset(&t, 0, sizeof(t)); }
        else if (c > 0) { big_abs_sub(&t, a, b); t.neg = a->neg; }
        else { big_abs_sub(&t, b, a); t.neg = b->neg; }
    }
    *r = t;
}

static void big_mul_ui(sbig_t *a, uint32_t m)
{
    uint64_t carry = 0;
    int i;
    if (!m || !a->n) { memset(a, 0, sizeof(*a)); return; }
    for (i = 0; i < a->n; i++) {
        uint64_t z = (uint64_t)a->v[i] * m + carry;
        a->v[i] = (uint32_t)z;
        carry = z >> 32;
    }
    if (carry) {
        if (a->n == BIG_LIMBS) { fprintf(stderr, "fbgen: bigint overflow\n"); exit(EXIT_FAILURE); }
        a->v[a->n++] = (uint32_t)carry;
    }
}

/* Divide |a| by d, preserving the sign; return the nonnegative remainder. */
static uint32_t big_div_ui(sbig_t *a, uint32_t d)
{
    uint64_t rem = 0;
    int i;
    for (i = a->n - 1; i >= 0; i--) {
        uint64_t z = (rem << 32) | a->v[i];
        a->v[i] = (uint32_t)(z / d);
        rem = z % d;
    }
    big_norm(a);
    return (uint32_t)rem;
}

/* Signed a modulo m, in [0,m). */
static uint32_t big_mod_ui(const sbig_t *a, uint32_t m)
{
    uint64_t r = 0;
    int i;
    for (i = a->n - 1; i >= 0; i--) r = ((r << 32) | a->v[i]) % m;
    if (a->neg && r) r = m - r;
    return (uint32_t)r;
}

static int big_p_val(const sbig_t *a, uint32_t p)
{
    sbig_t t = *a;
    int v = 0;
    if (!t.n) return INT_MAX;
    while (big_div_ui(&t, p) == 0) v++;
    return v;
}

static int zpoly_content_val(const zpoly_t *f, uint32_t p)
{
    int i, v = INT_MAX;
    for (i = 0; i <= f->deg; i++) {
        int w = big_p_val(&f->c[i], p);
        if (w < v) v = w;
        if (!v) break;
    }
    return v;
}

static void zpoly_div_p(zpoly_t *f, uint32_t p, int v)
{
    int i, j;
    for (i = 0; i <= f->deg; i++)
        for (j = 0; j < v; j++)
            if (big_div_ui(&f->c[i], p) != 0) {
                fprintf(stderr, "fbgen: internal non-exact polynomial division\n");
                exit(EXIT_FAILURE);
            }
}

static uint32_t binom_small(int n, int k)
{
    static const uint8_t b[9][9] = {
        {1}, {1,1}, {1,2,1}, {1,3,3,1}, {1,4,6,4,1},
        {1,5,10,10,5,1}, {1,6,15,20,15,6,1},
        {1,7,21,35,35,21,7,1},
        {1,8,28,56,70,56,28,8,1}
    };
    return b[n][k];
}

/* g(x) = f(a*x+b), with small nonnegative a,b. */
static void zpoly_linear_comp(zpoly_t *g, const zpoly_t *f, uint32_t a, uint32_t b)
{
    int i, j, k;
    memset(g, 0, sizeof(*g));
    g->deg = f->deg;
    for (j = 0; j <= f->deg; j++) {
        sbig_t sum;
        memset(&sum, 0, sizeof(sum));
        for (i = j; i <= f->deg; i++) {
            sbig_t term = f->c[i], next;
            big_mul_ui(&term, binom_small(i, j));
            for (k = 0; k < i - j; k++) big_mul_ui(&term, b);
            big_add(&next, &sum, &term);
            sum = next;
        }
        for (k = 0; k < j; k++) big_mul_ui(&sum, a);
        g->c[j] = sum;
    }
}

/* ---- polynomials over F_p ------------------------------------------ */

static uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p)
{
    uint64_t z = (uint64_t)a + b;
    return (uint32_t)(z >= p ? z - p : z);
}

static uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p)
{
    return a >= b ? a - b : (uint32_t)((uint64_t)a + p - b);
}

static uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p)
{
    return (uint32_t)(((uint64_t)a * b) % p);
}

static uint32_t mod_pow(uint32_t a, uint32_t e, uint32_t p)
{
    uint32_t r = 1;
    while (e) {
        if (e & 1u) r = mod_mul(r, a, p);
        a = mod_mul(a, a, p);
        e >>= 1;
    }
    return r;
}

static uint32_t mod_inv(uint32_t a, uint32_t p)
{
    return mod_pow(a, p - 2, p);
}

static void mp_norm(mpoly_t *a)
{
    while (a->deg >= 0 && a->c[a->deg] == 0) a->deg--;
}

static mpoly_t mp_from_z(const zpoly_t *f, uint32_t p)
{
    mpoly_t r;
    int i;
    memset(&r, 0, sizeof(r));
    r.deg = f->deg;
    for (i = 0; i <= f->deg; i++) r.c[i] = big_mod_ui(&f->c[i], p);
    mp_norm(&r);
    return r;
}

static uint32_t mp_eval(const mpoly_t *f, uint32_t x, uint32_t p)
{
    uint32_t r = 0;
    int i;
    for (i = f->deg; i >= 0; i--) r = mod_add(mod_mul(r, x, p), f->c[i], p);
    return r;
}

static void mp_make_monic(mpoly_t *f, uint32_t p)
{
    uint32_t z;
    int i;
    if (f->deg < 0 || f->c[f->deg] == 1) return;
    z = mod_inv(f->c[f->deg], p);
    for (i = 0; i <= f->deg; i++) f->c[i] = mod_mul(f->c[i], z, p);
}

static mpoly_t mp_rem(mpoly_t a, const mpoly_t *b, uint32_t p)
{
    uint32_t ilc;
    int i, shift;
    if (b->deg < 0) { fprintf(stderr, "fbgen: polynomial division by zero\n"); exit(EXIT_FAILURE); }
    ilc = mod_inv(b->c[b->deg], p);
    while (a.deg >= b->deg) {
        uint32_t q = mod_mul(a.c[a.deg], ilc, p);
        shift = a.deg - b->deg;
        for (i = 0; i <= b->deg; i++)
            a.c[i + shift] = mod_sub(a.c[i + shift], mod_mul(q, b->c[i], p), p);
        mp_norm(&a);
    }
    return a;
}

static mpoly_t mp_divexact(mpoly_t a, const mpoly_t *b, uint32_t p)
{
    mpoly_t q;
    uint32_t ilc;
    int i, shift;
    memset(&q, 0, sizeof(q)); q.deg = -1;
    ilc = mod_inv(b->c[b->deg], p);
    while (a.deg >= b->deg) {
        uint32_t z = mod_mul(a.c[a.deg], ilc, p);
        shift = a.deg - b->deg;
        q.c[shift] = z;
        if (shift > q.deg) q.deg = shift;
        for (i = 0; i <= b->deg; i++)
            a.c[i + shift] = mod_sub(a.c[i + shift], mod_mul(z, b->c[i], p), p);
        mp_norm(&a);
    }
    if (a.deg >= 0) { fprintf(stderr, "fbgen: internal inexact modular division\n"); exit(EXIT_FAILURE); }
    mp_norm(&q);
    return q;
}

static mpoly_t mp_gcd(mpoly_t a, mpoly_t b, uint32_t p)
{
    while (b.deg >= 0) {
        mpoly_t r = mp_rem(a, &b, p);
        a = b; b = r;
    }
    mp_make_monic(&a, p);
    return a;
}

static mpoly_t mp_mul_rem(const mpoly_t *a, const mpoly_t *b,
                          const mpoly_t *mod, uint32_t p)
{
    mpoly_t t;
    int i, j;
    memset(&t, 0, sizeof(t)); t.deg = -1;
    for (i = 0; i <= a->deg; i++) for (j = 0; j <= b->deg; j++) {
        uint32_t z = mod_mul(a->c[i], b->c[j], p);
        t.c[i+j] = mod_add(t.c[i+j], z, p);
        if (t.c[i+j] && i+j > t.deg) t.deg = i+j;
    }
    mp_norm(&t);
    return mp_rem(t, mod, p);
}

static mpoly_t mp_pow_rem(mpoly_t a, uint32_t e, const mpoly_t *mod, uint32_t p)
{
    mpoly_t r;
    memset(&r, 0, sizeof(r)); r.deg = 0; r.c[0] = 1;
    a = mp_rem(a, mod, p);
    while (e) {
        if (e & 1u) r = mp_mul_rem(&r, &a, mod, p);
        e >>= 1;
        if (e) a = mp_mul_rem(&a, &a, mod, p);
    }
    return r;
}

static int u32_cmp(const void *aa, const void *bb)
{
    uint32_t a = *(const uint32_t *)aa, b = *(const uint32_t *)bb;
    return a < b ? -1 : a > b;
}

/* Split a squarefree product of linear factors.  Trying successive constants
 * makes this deterministic; the final roots are sorted in any case. */
static int mp_split_linear(const mpoly_t *f, uint32_t p, uint32_t *roots)
{
    mpoly_t lin, h, g, other;
    uint32_t a;
    int n;
    if (f->deg == 0) return 0;
    if (f->deg == 1) {
        roots[0] = mod_mul(p - f->c[0], mod_inv(f->c[1], p), p);
        return 1;
    }
    for (a = 0;; a++) {
        memset(&lin, 0, sizeof(lin)); lin.deg = 1; lin.c[0] = a % p; lin.c[1] = 1;
        h = mp_pow_rem(lin, (p - 1) / 2, f, p);
        h.c[0] = mod_sub(h.c[0], 1, p); mp_norm(&h);
        g = mp_gcd(*f, h, p);
        if (g.deg > 0 && g.deg < f->deg) break;
        if (a == p - 1) {
            fprintf(stderr, "fbgen: failed to split degree-%d polynomial mod %u\n", f->deg, p);
            exit(EXIT_FAILURE);
        }
    }
    other = mp_divexact(*f, &g, p);
    n = mp_split_linear(&g, p, roots);
    n += mp_split_linear(&other, p, roots + n);
    return n;
}

static int zpoly_roots(const zpoly_t *F, uint32_t p, uint32_t *roots)
{
    mpoly_t f = mp_from_z(F, p), x, xp, linear;
    int n;
    if (f.deg <= 0) return 0;
    /* The rational side is linear, so its one affine root costs one modular
     * inverse.  Do this before the general x^p-x root finder: special-q
     * generation calls this once per prime, and making a degree-1 polynomial
     * take the degree-N path would turn the cheapest side into needless work. */
    if (f.deg == 1) {
        roots[0] = mod_mul(p - f.c[0], mod_inv(f.c[1], p), p);
        return 1;
    }
    if (p <= BRUTE_ROOT_LIMIT) {
        uint32_t r;
        n = 0;
        for (r = 0; r < p; r++) if (mp_eval(&f, r, p) == 0) roots[n++] = r;
        return n;
    }
    memset(&x, 0, sizeof(x)); x.deg = 1; x.c[1] = 1;
    xp = mp_pow_rem(x, p, &f, p);
    if (xp.deg < 1) xp.deg = 1;
    xp.c[1] = mod_sub(xp.c[1], 1, p); mp_norm(&xp);
    linear = mp_gcd(f, xp, p);
    n = mp_split_linear(&linear, p, roots);
    qsort(roots, (size_t)n, sizeof(*roots), u32_cmp);
    return n;
}

/* ---- streaming prime special-q generation -------------------------- */

struct sqgen {
    zpoly_t f;
    uint64_t candidate;                 /* next integer eligible for testing */
    uint64_t qmax;                      /* inclusive                         */
    uint32_t current_q;
    uint32_t roots[FBGEN_MAX_DEG + 1];
    uint32_t nroot, iroot;
    uint32_t emitted, nqmax;
};

sqgen_t *sqgen_create(const poly_t *P, int side, uint64_t qmin, uint64_t qmax,
                      uint32_t nqmax)
{
    sqgen_t *G;
    int deg;
    if (!P || (side != 0 && side != 1) || qmin > UINT32_MAX ||
        (qmax && qmax > UINT32_MAX) || (qmax && qmin > qmax)) {
        fprintf(stderr, "sqgen: q range must lie in [2, 2^32)\n");
        return NULL;
    }
    if (!side && (!P->y0s[0] || !P->y1s[0])) {
        fprintf(stderr, "sqgen: rational special-q generation requires both"
                        " Y0 and Y1\n");
        return NULL;
    }
    G = (sqgen_t *)calloc(1, sizeof(*G));
    if (!G) {
        fprintf(stderr, "sqgen: out of memory\n");
        return NULL;
    }
    deg = side ? P->deg : 1;
    G->f.deg = deg;
    for (int k = 0; k <= deg; k++) {
        const char *s = side ? (P->cs[k][0] ? P->cs[k] : "0")
                             : (k ? P->y1s : P->y0s);
        if (!s[0]) s = "0";
        if (big_parse(&G->f.c[k], s)) {
            fprintf(stderr, "sqgen: coefficient %d exceeds %d bits\n",
                    k, BIG_LIMBS * 32);
            free(G);
            return NULL;
        }
    }
    if (!side && !G->f.c[1].n) {
        fprintf(stderr, "sqgen: rational special-q generation requires a"
                        " nonzero Y1\n");
        free(G);
        return NULL;
    }
    G->candidate = qmin < 2 ? 2 : qmin;
    G->qmax = qmax ? qmax : UINT32_MAX;
    G->nqmax = nqmax;
    return G;
}

int sqgen_next(sqgen_t *G, qsel_t *out)
{
    if (!G || !out) return -1;
    if (G->nqmax && G->emitted >= G->nqmax) return 0;
    for (;;) {
        if (G->iroot < G->nroot) {
            out->q = G->current_q;
            out->rho = G->roots[G->iroot++];
            G->emitted++;
            return 1;
        }
        G->nroot = G->iroot = 0;
        while (G->candidate <= G->qmax) {
            uint32_t p = (uint32_t)G->candidate;
            if (p == 2)
                G->candidate = 3;
            else {
                if (!(p & 1u)) p++;
                G->candidate = (uint64_t)p + 2;
            }
            if (p > G->qmax) break;
            if (!bench_is_prime32(p)) continue;
            G->nroot = (uint32_t)zpoly_roots(&G->f, p, G->roots);
            if (G->nroot) {
                G->current_q = p;
                break;
            }
        }
        if (!G->nroot) return 0;
    }
}

void sqgen_free(sqgen_t *G)
{
    free(G);
}

#ifndef FBGEN_LIBRARY
static uint32_t zpoly_eval_mod(const zpoly_t *f, uint32_t x, uint32_t m)
{
    uint64_t r = 0;
    int i;
    for (i = f->deg; i >= 0; i--)
        r = (r * x + big_mod_ui(&f->c[i], m)) % m;
    return (uint32_t)r;
}

static uint32_t zpoly_deriv_mod(const zpoly_t *f, uint32_t x, uint32_t p)
{
    uint32_t r = 0;
    int i;
    for (i = f->deg; i >= 1; i--)
        r = mod_add(mod_mul(r, x, p), mod_mul(big_mod_ui(&f->c[i], p), (uint32_t)i, p), p);
    return r;
}

static uint32_t lift_unramified(const zpoly_t *f, uint32_t r, uint32_t p, int kmax)
{
    uint32_t q = p, df = zpoly_deriv_mod(f, r, p), idf = mod_inv(df, p);
    int k;
    for (k = 1; k < kmax; k++) {
        uint64_t nq64 = (uint64_t)q * p;
        uint32_t nq = (uint32_t)nq64;
        uint32_t fr = zpoly_eval_mod(f, r, nq);
        uint32_t digit = (fr / q) % p;
        uint32_t t = digit ? mod_mul(p - digit, idf, p) : 0;
        r += t * q;
        q = nq;
    }
    return r;
}

/* ---- p-adic branches and output entries ---------------------------- */

static void ev_push(entry_vec_t *v, fbgen_entry_t e)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? 2 * v->cap : 32;
        v->v = (fbgen_entry_t *)xrealloc(v->v, v->cap * sizeof(*v->v));
    }
    v->v[v->n++] = e;
}

static uint32_t pow_u32(uint32_t p, int k)
{
    uint64_t q = 1;
    while (k--) q *= p;
    if (q > UINT32_MAX) { fprintf(stderr, "fbgen: prime power overflow\n"); exit(EXIT_FAILURE); }
    return (uint32_t)q;
}

static void all_roots_affine(entry_vec_t *L, const zpoly_t *f, uint32_t p,
                             int kmax, int k0, int m,
                             uint32_t phi1, uint32_t phi0)
{
    uint32_t roots[FBGEN_MAX_DEG + 1];
    int i, nr;
    if (k0 >= kmax) return;
    nr = zpoly_roots(f, p, roots);
    for (i = 0; i < nr; i++) {
        uint32_t r = roots[i];
        uint32_t dfr = zpoly_deriv_mod(f, r, p);
        if (dfr) {
            int l, klift = kmax - k0;
            uint32_t rr = lift_unramified(f, r, p, klift);
            uint64_t phir = (uint64_t)phi1 * rr + phi0;
            uint32_t pml = pow_u32(p, m);
            for (l = 1; l <= klift; l++) {
                pml = (uint32_t)((uint64_t)pml * p);
                ev_push(L, (fbgen_entry_t){pml, (uint32_t)(phir % pml),
                                           k0 + l, k0 + l - 1});
            }
        } else {
            zpoly_t ff;
            int val;
            uint32_t q = pow_u32(p, m + 1);
            uint64_t phir = (uint64_t)phi1 * r + phi0;
            uint32_t nphi1, nphi0;
            zpoly_linear_comp(&ff, f, p, r);
            val = zpoly_content_val(&ff, p);
            if (val == INT_MAX) {
                fprintf(stderr, "fbgen: zero polynomial on ramified branch mod %u\n", p);
                exit(EXIT_FAILURE);
            }
            ev_push(L, (fbgen_entry_t){q, (uint32_t)(phir % q), k0 + val, k0});
            nphi1 = (uint32_t)((uint64_t)phi1 * p);
            nphi0 = (uint32_t)((uint64_t)phi0 + (uint64_t)phi1 * r);
            zpoly_div_p(&ff, p, val);
            all_roots_affine(L, &ff, p, kmax, k0 + val, m + 1, nphi1, nphi0);
        }
    }
}

static int entry_cmp(const void *aa, const void *bb)
{
    const fbgen_entry_t *a = (const fbgen_entry_t *)aa;
    const fbgen_entry_t *b = (const fbgen_entry_t *)bb;
#define CMP_FIELD(x) do { if (a->x != b->x) return a->x < b->x ? -1 : 1; } while (0)
    CMP_FIELD(q); CMP_FIELD(n1); CMP_FIELD(n0); CMP_FIELD(r);
#undef CMP_FIELD
    return 0;
}

static entry_vec_t all_roots(const zpoly_t *f, uint32_t p, int maxbits)
{
    entry_vec_t L = {0};
    zpoly_t rev;
    int i, v, kmax = 0;
    uint64_t q = 1, bound = 1ull << maxbits;
    while (q * p <= bound) { q *= p; kmax++; }
    if (!kmax) kmax = 1;                 /* primes are always included to lim */

    memset(&rev, 0, sizeof(rev)); rev.deg = f->deg;
    for (i = 0; i <= f->deg; i++) {
        rev.c[i] = f->c[f->deg - i];
        for (int j = 0; j < i; j++) big_mul_ui(&rev.c[i], p);
    }
    v = zpoly_content_val(&rev, p);
    if (v > 0 && v != INT_MAX) {
        size_t first;
        ev_push(&L, (fbgen_entry_t){p, p, v, 0});
        zpoly_div_p(&rev, p, v);
        first = L.n;
        all_roots_affine(&L, &rev, p, kmax - 1, 0, 0, 1, 0);
        for (i = (int)first; i < (int)L.n; i++) {
            L.v[i].q *= p;
            L.v[i].n1 += v;
            L.v[i].n0 += v;
            L.v[i].r = L.v[i].r * p + L.v[i].q;
        }
    }
    all_roots_affine(&L, f, p, kmax, 0, 0, 1, 0);
    if (L.n > 1) qsort(L.v, L.n, sizeof(*L.v), entry_cmp);
    return L;
}

static void print_prime_entries(FILE *out, const entry_vec_t *L)
{
    uint32_t oldq = 0;
    int oldn1 = -1, oldn0 = -1;
    size_t i;
    for (i = 0; i < L->n; i++) {
        const fbgen_entry_t *e = &L->v[i];
        if (e->q == oldq && e->n1 == oldn1 && e->n0 == oldn0) {
            fprintf(out, ",%u", e->r);
        } else {
            if (i) fputc('\n', out);
            oldq = e->q; oldn1 = e->n1; oldn0 = e->n0;
            if (e->n1 == 1 && e->n0 == 0) fprintf(out, "%u: %u", e->q, e->r);
            else fprintf(out, "%u:%d,%d: %u", e->q, e->n1, e->n0, e->r);
        }
    }
    if (L->n) fputc('\n', out);
}

/* ---- prime enumeration and workers --------------------------------- */

typedef struct {
    const zpoly_t *f;
    const uint32_t *primes;
    size_t begin, end;
    int maxbits;
    char *data;
    size_t size;
    int failed;
} worker_t;

static void *worker_main(void *arg)
{
    worker_t *w = (worker_t *)arg;
    FILE *stream;
    size_t i;
    stream = open_memstream(&w->data, &w->size);
    if (!stream) { w->failed = errno ? errno : EIO; return NULL; }
    for (i = w->begin; i < w->end; i++) {
        entry_vec_t L = all_roots(w->f, w->primes[i], w->maxbits);
        print_prime_entries(stream, &L);
        free(L.v);
        if (ferror(stream)) { w->failed = errno ? errno : EIO; break; }
    }
    if (fclose(stream) && !w->failed) w->failed = errno ? errno : EIO;
    return NULL;
}

static void print_poly(FILE *out, const poly_t *P)
{
    int i, any = 0;
    fputs("# Roots for polynomial ", out);
    for (i = 0; i <= P->deg; i++) {
        const char *s = P->cs[i][0] ? P->cs[i] : "0";
        int neg = s[0] == '-';
        const char *mag = neg ? s + 1 : s;
        if (!strcmp(mag, "0")) continue;
        if (!any) {
            if (neg) fputc('-', out);
        } else {
            fputc(neg ? '-' : '+', out);
        }
        {
            int printed_coeff = (i == 0 || strcmp(mag, "1"));
            if (printed_coeff) fputs(mag, out);
            if (i >= 1) {
                fputs(printed_coeff ? "*x" : "x", out);
                if (i >= 2) fprintf(out, "^%d", i);
            }
        }
        any = 1;
    }
    if (!any) fputc('0', out);
    fputc('\n', out);
}

static int generate(FILE *out, const poly_t *P, uint32_t lim,
                                        int maxbits, int nthr)
{
    zpoly_t f;
    uint32_t *primes;
    size_t np;
    worker_t *w;
    pthread_t *tid;
    int started = 0, create_error = 0, rc = -1;
    memset(&f, 0, sizeof(f)); f.deg = P->deg;
    for (int k = 0; k <= P->deg; k++)
        if (big_parse(&f.c[k], P->cs[k][0] ? P->cs[k] : "0")) {
            fprintf(stderr, "fbgen: coefficient c%d exceeds %d bits\n", k, BIG_LIMBS * 32);
            return -1;
        }
    primes = prime_list_build(lim, &np);
    if (!primes) die_oom();
    if ((size_t)nthr > np) nthr = (int)np;
    if (nthr < 1) nthr = 1;
    w = (worker_t *)calloc((size_t)nthr, sizeof(*w));
    tid = (pthread_t *)xmalloc((size_t)nthr * sizeof(*tid));
    if (!w) die_oom();

    for (int t = 0; t < nthr; t++) {
        int err;
        w[t].f = &f; w[t].primes = primes; w[t].maxbits = maxbits;
        w[t].begin = np * (size_t)t / (size_t)nthr;
        w[t].end   = np * (size_t)(t + 1) / (size_t)nthr;
        err = pthread_create(&tid[t], NULL, worker_main, &w[t]);
        if (err) {
            create_error = err;
            break;
        }
        started++;
    }
    for (int t = 0; t < started; t++) pthread_join(tid[t], NULL);
    if (create_error) {
        fprintf(stderr, "fbgen: pthread_create failed: %s\n", strerror(create_error));
        goto cleanup;
    }
    for (int t = 0; t < nthr; t++) {
        if (w[t].failed || !w[t].data) {
            int err = w[t].failed ? w[t].failed : EIO;
            fprintf(stderr, "fbgen: worker %d failed: %s\n", t, strerror(err));
            goto cleanup;
        }
    }
    print_poly(out, P);
    fprintf(out, "# DEGREE: %d\n# lim = %u\n# maxbits = %d\n", P->deg, lim, maxbits);
    for (int t = 0; t < nthr; t++)
        if (fwrite(w[t].data, 1, w[t].size, out) != w[t].size) goto cleanup;
    if (ferror(out)) goto cleanup;
    rc = 0;

cleanup:
    for (int t = 0; t < nthr; t++) free(w[t].data);
    free(tid); free(w); free(primes);
    if (!rc)
        fprintf(stderr, "fbgen: processed %zu primes through %u with %d thread%s\n",
                np, lim, nthr, nthr == 1 ? "" : "s");
    return rc;
}

#endif

#ifndef FBGEN_LIBRARY
static void usage(FILE *f)
{
    fputs("usage: fbgen --poly FILE [--lim N] [--maxbits N] [--threads N] [--out FILE]\n"
          "  --lim defaults to alim from the job file\n"
          "  --maxbits defaults to 15; 1 is legacy prime-only, logI is recommended\n",
          f);
}

int main(int argc, char **argv)
{
    const char *poly_path = NULL, *out_path = NULL;
    uint32_t lim = 0;
    int maxbits = 15, nthr = 1, i, rc;
    poly_t P;
    FILE *out = stdout;
    char *stage_path = NULL;
    for (i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "--poly") || !strcmp(argv[i], "-poly")) && i + 1 < argc)
            poly_path = argv[++i];
        else if ((!strcmp(argv[i], "--lim") || !strcmp(argv[i], "-lim")) && i + 1 < argc)
            lim = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if ((!strcmp(argv[i], "--maxbits") || !strcmp(argv[i], "-maxbits")) && i + 1 < argc)
            maxbits = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t")) && i + 1 < argc)
            nthr = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--out") || !strcmp(argv[i], "-out")) && i + 1 < argc)
            out_path = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(stdout); return 0; }
        else { fprintf(stderr, "fbgen: unknown or incomplete option: %s\n", argv[i]); usage(stderr); return 2; }
    }
    if (!poly_path) { usage(stderr); return 2; }
    if (maxbits < 1 || maxbits > 31) {
        fprintf(stderr, "fbgen: --maxbits must be in [1,31]\n"); return 2;
    }
    if (nthr < 1 || nthr > 256) { fprintf(stderr, "fbgen: invalid thread count %d\n", nthr); return 2; }
    if (poly_load(poly_path, &P)) return 1;
    if (P.deg > FBGEN_MAX_DEG) {
        fprintf(stderr, "fbgen: degree %d exceeds supported maximum %d\n", P.deg, FBGEN_MAX_DEG);
        return 1;
    }
    if (!lim) lim = P.alim;
    if (lim < 2) { fprintf(stderr, "fbgen: --lim is required (or alim in the job file)\n"); return 2; }
    if (out_path && strcmp(out_path, "-")) {
        size_t n = strlen(out_path) + sizeof(".part");
        stage_path = (char *)xmalloc(n);
        snprintf(stage_path, n, "%s.part", out_path);
        out = fopen(stage_path, "wb");
        if (!out) {
            fprintf(stderr, "fbgen: cannot open %s: %s\n", stage_path, strerror(errno));
            free(stage_path);
            return 1;
        }
    }
    rc = generate(out, &P, lim, maxbits, nthr);
    if (out != stdout && fclose(out)) rc = -1;
    if (stage_path) {
        if (!rc && rename(stage_path, out_path)) {
            fprintf(stderr, "fbgen: cannot rename %s to %s: %s\n",
                    stage_path, out_path, strerror(errno));
            rc = -1;
        }
        if (rc) remove(stage_path);
        free(stage_path);
    }
    return rc ? 1 : 0;
}
#endif
