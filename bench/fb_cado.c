/* Loader for CADO's text factor base (the output of makefb).
 *
 * Using this instead of GGNFS's .afb.0 buys two things that matter for parity:
 * prime powers (which .afb.0 has none of), and the exact per-entry log
 * increment las will use.
 *
 * Format, from fb_entry_general::parse_line:
 *   q: r1,r2,...              short form; q is prime, nexp=1, oldexp=0
 *   q:nexp,oldexp: r1,r2,...  long form;  q may be a prime power p^k
 * A root value >= q is projective, encoding the actual root as r - q
 * (fb_root_p1::to_old_format), the same convention .afb.0 uses.
 *
 * The log increment is NOT log(q). It is
 *     fb_log_delta = fb_log(p^nexp) - fb_log(p^oldexp),
 *     fb_log(n) = floor(log2(n) * scale + 0.5)
 * i.e. the marginal cost of going from p^oldexp to p^nexp, which is what makes
 * powers compose correctly with their base prime.
 *
 * The file is grouped by base prime with the powers of each listed inside the
 * group, so it is NOT globally sorted by q -- and everything downstream
 * (fb_split_small, fb_restrict, the slice builder) needs ascending q. The k==1
 * entries do arrive in ascending order and the powers are few (makefb was run
 * with -maxbits 15, so every power is <= 32768), so a merge is enough and a
 * full sort of 7.6M entries is not.
 */
#define _GNU_SOURCE
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* If q = p^k with k > 1, return the BASE PRIME p and set *kk = k; else 0.
 *
 * It must be the base prime, i.e. k maximal. The previous version searched k
 * upward from 2 and returned the first that worked, which is k MINIMAL and so
 * the largest base: it read 81 as 9^2 and 729 as 27^2. The exponents in the
 * file are relative to the base prime (`81:5,4: 114` means 3^5 over 3^4), so a
 * wrong base silently inflates the log increment -- 4 instead of 2 for 81, 6
 * instead of 2 for 729, 3 instead of 1 for 16. Placement was unaffected, which
 * is why only a byte-level trace against las found it (gate 5, 2026-08-02).
 *
 * Trial division for the smallest prime factor is exact and, with 97 long-form
 * lines in the file, free. */
static uint32_t is_power(uint32_t q, int *kk)
{
    uint32_t d;
    for (d = 2; (uint64_t)d * d <= (uint64_t)q; d++) {
        uint32_t t;
        int k = 0;
        if (q % d) continue;
        for (t = q; t % d == 0; t /= d) k++;
        if (t == 1 && k > 1) { *kk = k; return d; }
        return 0;                     /* not a pure power of its least factor */
    }
    return 0;                         /* prime */
}

static uint8_t log_delta(uint32_t p, int nexp, int oldexp, double scale)
{
    double l2 = log2((double)p);
    long a = (long)floor((double)nexp   * l2 * scale + 0.5);
    long b = (long)floor((double)oldexp * l2 * scale + 0.5);
    long d = a - b;
    return (uint8_t)(d < 0 ? 0 : (d > 255 ? 255 : d));
}

int fb_load_cado(const char *path, double scale, fb_t *fb)
{
    FILE *f = fopen(path, "r");
    char *line = NULL;
    size_t linecap = 0;
    ssize_t len;
    uint32_t na = 0, nb = 0, capa = 1u << 23, capb = 1u << 16;
    uint32_t *qa, *ra, *qb, *rb, i, j, k;
    uint8_t  *la, *lb;
    unsigned long linenr = 0;

    memset(fb, 0, sizeof(*fb));
    if (!f) { fprintf(stderr, "fb_load_cado: cannot open %s\n", path); return -1; }

    qa = (uint32_t *)malloc((size_t)capa * 4);   /* k == 1, already ascending */
    ra = (uint32_t *)malloc((size_t)capa * 4);
    la = (uint8_t  *)malloc((size_t)capa);
    qb = (uint32_t *)malloc((size_t)capb * 4);   /* k >  1, few hundred       */
    rb = (uint32_t *)malloc((size_t)capb * 4);
    lb = (uint8_t  *)malloc((size_t)capb);
    if (!qa || !ra || !la || !qb || !rb || !lb) { fclose(f); return -1; }

    while ((len = getline(&line, &linecap, f)) > 0) {
        char *s = line, *e;
        uint32_t q, p;
        int nexp = 1, oldexp = 0, kk = 1;
        uint8_t lg;
        linenr++;
        if (*s == '#' || *s == '\n') continue;
        q = (uint32_t)strtoul(s, &e, 10);
        if (e == s || *e != ':') continue;
        s = e + 1;
        p = q;
        if (strchr(s, ':')) {                     /* long form */
            uint32_t base = is_power(q, &kk);
            if (base) p = base; else kk = 1;
            nexp = (int)strtoul(s, &e, 10);
            if (e == s || *e != ',') {
                fprintf(stderr, "fb_load_cado:%lu: bad nexp\n", linenr); goto fail;
            }
            s = e + 1;
            oldexp = (int)strtoul(s, &e, 10);
            if (e == s || *e != ':') {
                fprintf(stderr, "fb_load_cado:%lu: bad oldexp\n", linenr); goto fail;
            }
            s = e + 1;
        }
        lg = log_delta(p, nexp, oldexp, scale);
        for (;;) {                                 /* comma-separated roots */
            uint32_t r;
            while (*s == ' ' || *s == '\t') s++;
            if (*s < '0' || *s > '9') break;
            r = (uint32_t)strtoul(s, &e, 10);
            s = e;
            if (kk > 1) {
                if (nb == capb) {
                    capb *= 2;
                    qb = (uint32_t *)realloc(qb, (size_t)capb * 4);
                    rb = (uint32_t *)realloc(rb, (size_t)capb * 4);
                    lb = (uint8_t  *)realloc(lb, (size_t)capb);
                    if (!qb || !rb || !lb) goto fail;
                }
                qb[nb] = q; rb[nb] = r; lb[nb] = lg; nb++;
            } else {
                if (na == capa) {
                    capa *= 2;
                    qa = (uint32_t *)realloc(qa, (size_t)capa * 4);
                    ra = (uint32_t *)realloc(ra, (size_t)capa * 4);
                    la = (uint8_t  *)realloc(la, (size_t)capa);
                    if (!qa || !ra || !la) goto fail;
                }
                qa[na] = q; ra[na] = r; la[na] = lg; na++;
            }
            if (*s == ',') s++; else break;
        }
    }
    free(line);
    fclose(f);

    /* the k==1 stream must already be ascending for the merge to be valid */
    for (i = 1; i < na; i++)
        if (qa[i] < qa[i - 1]) {
            fprintf(stderr, "fb_load_cado: base primes not ascending at %u\n", i);
            goto fail2;
        }
    /* insertion-sort the power entries (a few hundred) */
    for (i = 1; i < nb; i++) {
        uint32_t q = qb[i], r = rb[i]; uint8_t l = lb[i];
        int32_t m = (int32_t)i - 1;
        while (m >= 0 && qb[m] > q) { qb[m+1] = qb[m]; rb[m+1] = rb[m]; lb[m+1] = lb[m]; m--; }
        qb[m+1] = q; rb[m+1] = r; lb[m+1] = l;
    }

    fb->n      = na + nb;
    fb->primes = (uint32_t *)malloc((size_t)fb->n * 4);
    fb->roots  = (uint32_t *)malloc((size_t)fb->n * 4);
    fb->logp   = (uint8_t  *)malloc((size_t)fb->n);
    if (!fb->primes || !fb->roots || !fb->logp) goto fail2;
    for (i = j = k = 0; k < fb->n; k++) {
        int takea = (i < na) && (j >= nb || qa[i] <= qb[j]);
        if (takea) { fb->primes[k] = qa[i]; fb->roots[k] = ra[i]; fb->logp[k] = la[i]; i++; }
        else       { fb->primes[k] = qb[j]; fb->roots[k] = rb[j]; fb->logp[k] = lb[j]; j++; }
    }
    printf("CADO factor base %s: %u ideals (%u prime, %u prime-power)"
           " at scale %.3f\n", path, fb->n, na, nb, scale);
    free(qa); free(ra); free(la); free(qb); free(rb); free(lb);
    return 0;

fail:
    free(line); fclose(f);
fail2:
    free(qa); free(ra); free(la); free(qb); free(rb); free(lb);
    return -1;
}
