#include "bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t dec_mod(const char *s, uint32_t p)
{
    uint64_t r = 0;
    int neg = 0;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-' || *s == '+') { neg = *s == '-'; s++; }
    while (*s >= '0' && *s <= '9') {
        r = (r * 10 + (uint32_t)(*s - '0')) % p;
        s++;
    }
    if (neg && r) r = p - r;
    return (uint32_t)r;
}

static int check_root(const poly_t *P, int side, const qsel_t *s)
{
    const uint32_t q = (uint32_t)s->q, r = (uint32_t)s->rho;
    uint64_t v = 0;
    int deg = side ? P->deg : 1;
    for (int k = deg; k >= 0; k--) {
        const char *c = side ? (P->cs[k][0] ? P->cs[k] : "0")
                             : (k ? P->y1s : P->y0s);
        v = (v * r + dec_mod(c, q)) % q;
    }
    return v == 0;
}

static int qsel_validation(void)
{
    poly_t P, Z;
    qsel_t s;
    memset(&P, 0, sizeof P);
    P.deg = 2;
    strcpy(P.cs[0], "-1");
    strcpy(P.cs[1], "0");
    strcpy(P.cs[2], "1");          /* f(x) = x^2 - 1 */
    strcpy(P.y0s, "-1");
    strcpy(P.y1s, "1");            /* G(x) = x - 1   */

    s.q = 7; s.rho = ~(uint64_t)0;  /* UINT64_MAX mod 7 == 1 */
    if (qsel_validate(&s, &P, 1) != QSEL_VALID || s.rho != 1) {
        fprintf(stderr, "qsel validation: failed to canonicalise a valid root\n");
        return -1;
    }
    s.q = 7; s.rho = 2;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_NOT_ROOT || s.rho != 2) return -1;
    s.q = 9; s.rho = 1;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_Q_COMPOSITE) return -1;
    s.q = 0; s.rho = 0;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_Q_RANGE) return -1;
    s.q = 1; s.rho = 0;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_Q_RANGE) return -1;
    s.q = UINT64_C(1) << 32; s.rho = 1;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_Q_RANGE) return -1;
    s.q = UINT32_C(4294967291); s.rho = 1; /* largest 32-bit prime */
    if (qsel_validate(&s, &P, 1) != QSEL_VALID || s.rho != 1) return -1;
    s.q = 7; s.rho = 8;
    if (qsel_validate(&s, &P, 0) != QSEL_VALID || s.rho != 1) return -1;

    /* Explicit --rho 0 used to be mistaken for "rho omitted" by truth-value
     * tests in the single-q path. Zero is a legitimate affine root. */
    memset(&Z, 0, sizeof Z);
    Z.deg = 1;
    strcpy(Z.cs[0], "0");
    strcpy(Z.cs[1], "1");            /* z(x) = x */
    s.q = 5; s.rho = 0;
    if (qsel_validate(&s, &Z, 1) != QSEL_VALID || s.rho != 0) return -1;

    if (qsel_validate(&s, &P, 2) != QSEL_ERR_SIDE) return -1;
    P.y1s[0] = 0;
    if (qsel_validate(&s, &P, 0) != QSEL_ERR_POLY) return -1;
    strcpy(P.y1s, "1");
    strcpy(P.cs[1], "1x");
    s.q = 7; s.rho = 1;
    if (qsel_validate(&s, &P, 1) != QSEL_ERR_POLY || s.rho != 1) return -1;

    printf("PASS   central special-q validation\n");
    return 0;
}

static int algebraic_above_lim(void)
{
    poly_t P;
    sqgen_t *G;
    qsel_t s, prev = {0, 0};
    unsigned n = 0;
    if (poly_load("testdata/octic.poly", &P)) return -1;
    G = sqgen_create(&P, 1, P.alim + 1, P.alim + 1000, 0);
    if (!G) return -1;
    for (;;) {
        int rc = sqgen_next(G, &s);
        if (rc < 0) { sqgen_free(G); return -1; }
        if (!rc) break;
        if (s.q <= P.alim || s.q > P.alim + 1000 ||
            !bench_is_prime32((uint32_t)s.q) || !check_root(&P, 1, &s) ||
            (prev.q && (s.q < prev.q || (s.q == prev.q && s.rho < prev.rho)))) {
            fprintf(stderr, "algebraic sqgen: bad pair (%llu, %llu)\n",
                    (unsigned long long)s.q, (unsigned long long)s.rho);
            sqgen_free(G); return -1;
        }
        prev = s; n++;
    }
    sqgen_free(G);
    if (!n) { fprintf(stderr, "algebraic sqgen: empty above-alim range\n"); return -1; }
    printf("PASS   algebraic q above alim       %u roots\n", n);
    return 0;
}

static int rational_open_count(void)
{
    poly_t P;
    sqgen_t *G;
    qsel_t s;
    unsigned n = 0;
    if (poly_load("../oracle/c183.poly", &P)) return -1;
    G = sqgen_create(&P, 0, 1000000, 0, 25);
    if (!G) return -1;
    while (sqgen_next(G, &s) > 0) {
        if (!bench_is_prime32((uint32_t)s.q) || !check_root(&P, 0, &s)) {
            fprintf(stderr, "rational sqgen: bad pair (%llu, %llu)\n",
                    (unsigned long long)s.q, (unsigned long long)s.rho);
            sqgen_free(G); return -1;
        }
        n++;
    }
    sqgen_free(G);
    if (n != 25) {
        fprintf(stderr, "rational sqgen: got %u roots, expected 25\n", n);
        return -1;
    }
    printf("PASS   rational open --nq          %u roots\n", n);
    return 0;
}

static int inclusive_single_prime(void)
{
    poly_t P;
    sqgen_t *G;
    qsel_t s;
    uint64_t first_q;
    int a, b;
    if (poly_load("../oracle/c183.poly", &P)) return -1;
    G = sqgen_create(&P, 0, 1000003, 1000003, 0);
    if (!G) return -1;
    a = sqgen_next(G, &s);
    first_q = s.q;
    b = sqgen_next(G, &s);
    sqgen_free(G);
    if (a != 1 || b != 0 || first_q != 1000003) {
        fprintf(stderr, "sqgen inclusive upper bound failed\n");
        return -1;
    }
    printf("PASS   inclusive qrange upper bound\n");
    return 0;
}

/* Compare directly with the CADO-format oracle in a range that includes the
 * parity q used by cofcheck.sh.  Prime-power lines and projective roots are not
 * special-q lattice ideals, so the comparison deliberately skips both. */
static int oracle_agreement(void)
{
    const uint32_t qmin = 120000000, qmax = 120001000;
    poly_t P;
    sqgen_t *G;
    qsel_t got = {0, 0};
    FILE *f;
    char line[4096];
    unsigned n = 0;

    if (poly_load("../oracle/c183.poly", &P)) return -1;
    f = fopen("../oracle/c183.fb1", "r");
    if (!f) { perror("../oracle/c183.fb1"); return -1; }
    G = sqgen_create(&P, 1, qmin, qmax, 0);
    if (!G) { fclose(f); return -1; }

    while (fgets(line, sizeof line, f)) {
        char *end, *roots, *p = line;
        unsigned long q;
        if (*p == '#') continue;
        q = strtoul(p, &end, 10);
        if (end == p || *end != ':' || q < qmin || q > qmax ||
            !bench_is_prime32((uint32_t)q)) continue;
        /* Prime entries are `q: roots`; only prime-power entries carry the
         * extra `q:nexp,oldexp: roots` field, and those were skipped above. */
        roots = end;
        p = roots + 1;
        for (;;) {
            unsigned long rho;
            while (*p == ' ' || *p == '\t' || *p == ',') p++;
            if (*p == '\0' || *p == '\n' || *p == '\r') break;
            rho = strtoul(p, &end, 10);
            if (end == p) {
                fprintf(stderr, "oracle sqgen: malformed root list\n");
                sqgen_free(G); fclose(f); return -1;
            }
            p = end;
            if (rho >= q) continue;       /* projective root at infinity */
            if (sqgen_next(G, &got) != 1 || got.q != q || got.rho != rho) {
                fprintf(stderr,
                        "oracle sqgen: expected (%lu,%lu), got (%llu,%llu)\n",
                        q, rho, (unsigned long long)got.q,
                        (unsigned long long)got.rho);
                sqgen_free(G); fclose(f); return -1;
            }
            n++;
        }
    }
    fclose(f);
    if (sqgen_next(G, &got) != 0 || n != 67) {
        fprintf(stderr, "oracle sqgen: expected exactly 67 pairs, saw %u\n", n);
        sqgen_free(G); return -1;
    }
    sqgen_free(G);
    printf("PASS   oracle q/rho agreement       %u pairs\n", n);
    return 0;
}

static int missing_rational_coefficient(void)
{
    poly_t P;
    sqgen_t *G;
    memset(&P, 0, sizeof P);
    strcpy(P.y0s, "1");
    G = sqgen_create(&P, 0, 100, 200, 0);
    if (G) {
        fprintf(stderr, "rational sqgen: accepted a missing Y1\n");
        sqgen_free(G); return -1;
    }
    printf("PASS   missing rational Y1 refused\n");
    return 0;
}

int main(void)
{
    if (qsel_validation() || algebraic_above_lim() || rational_open_count() ||
        inclusive_single_prime() || oracle_agreement() ||
        missing_rational_coefficient()) return 1;
    return 0;
}
