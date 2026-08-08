/* Join the two per-side --emit-cof files into a cofactorisation batch.
 *
 * This is the piece the 2026-08-04 harness work identified as the real blocker
 * for feeding a cofactorizer, and it is NOT "a small input adapter". A GGNFS
 * siever emits rels.raw only after deciding which cofactors are worth
 * cofactorising; CADO's -batch-print-survivors emits them raw, which is why the
 * C183 capture could not be fed. The classification that closes that gap now
 * runs on the GPU (k_classify, CADO's check_leftover_norm), so this program is
 * only the join and the format.
 *
 * Input, per side, one line per two-sided survivor in rank order:
 *     a b cofactor bits status nfac f1 f2 ...
 * Output:
 *   - the batch, in the format nfs_3lp_batch_factor's main.c parses,
 *     `res1,res2:a,b:rfac_hex:afac_hex`, with res NEGATIVE meaning "still
 *     needs splitting". That convention was inferred from the reference file
 *     by histogramming 400,000 of its relations: negative res1 values occupy
 *     48-63 bits and never 24-31. The rule is exactly `negative iff bits>lpb`.
 *   - COMPLETE RELATIONS, separately. A record whose cofactors are both within
 *     lpb needs no cofactorisation and is already a relation; counting it and
 *     dropping it (which this program did until it was reviewed) loses real
 *     output. At the parity q that is 7 relations, and 7 of las's 37 at that q
 *     are exactly these.
 *
 * The two input files are written in rank order over the same two-sided
 * bitmap, so they must agree line for line and end together. Anything else
 * means the join is pairing unrelated positions, and this exits nonzero rather
 * than writing a batch that looks usable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define COF_ACCEPT 3
#define COF_SPLIT  4
#define MAXFAC     64

#define ROW_OK    0
#define ROW_EOF   1
#define ROW_ERROR 2
#define ROW_SKIP  3   /* well-formed, but not a candidate */

typedef struct {
    long long a, b;
    char cof[80];
    int bits, status, nfac;
    unsigned fac[MAXFAC];
} row_t;

static int read_row(FILE *f, row_t *r, const char *what, unsigned long line)
{
    char buf[4096], *p, *end;
    if (!fgets(buf, sizeof buf, f)) return feof(f) ? ROW_EOF : ROW_ERROR;
    if (!strchr(buf, '\n') && !feof(f)) {
        fprintf(stderr, "%s:%lu: line longer than %zu bytes\n",
                what, line, sizeof buf);
        return ROW_ERROR;
    }
    if (sscanf(buf, "%lld %lld %79s %d %d %d",
               &r->a, &r->b, r->cof, &r->bits, &r->status, &r->nfac) != 6) {
        fprintf(stderr, "%s:%lu: malformed record\n", what, line);
        return ROW_ERROR;
    }
    if (r->nfac < 0) {
        fprintf(stderr, "%s:%lu: negative factor count\n", what, line);
        return ROW_ERROR;
    }
    /* Only candidates are used, so only their factor lists have to be intact.
     * A non-candidate may legitimately carry more factors than --emit-cof
     * writes; bench flags that case for candidates itself and fails the run. */
    if (r->status != COF_ACCEPT && r->status != COF_SPLIT) return ROW_SKIP;
    if (r->nfac > MAXFAC) {
        fprintf(stderr, "%s:%lu: candidate has %d factors, more than the %d"
                " --emit-cof writes, so its list is truncated\n",
                what, line, r->nfac, MAXFAC);
        return ROW_ERROR;
    }
    /* skip the six leading fields, then read exactly nfac factors */
    p = buf;
    for (int k = 0; k < 6; k++) {
        while (*p == ' ' || *p == '\t') p++;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
    }
    for (int k = 0; k < r->nfac; k++) {
        unsigned long v;
        errno = 0;
        v = strtoul(p, &end, 10);
        if (end == p || errno == ERANGE || v == 0 || v > 0xFFFFFFFFul) {
            fprintf(stderr, "%s:%lu: factor %d of %d is not a 32-bit prime\n",
                    what, line, k + 1, r->nfac);
            return ROW_ERROR;
        }
        r->fac[k] = (unsigned)v;
        p = end;
    }
    return ROW_OK;
}

static void put_factors(FILE *o, const row_t *r)
{
    for (int k = 0; k < r->nfac; k++)
        fprintf(o, "%s%x", k ? "," : "", r->fac[k]);
}

int main(int argc, char **argv)
{
    FILE *f0, *f1, *o, *ro;
    row_t r0, r1;
    unsigned long nline = 0, nout = 0, nfull = 0;
    int lpbr = 31, lpba = 32, rc = 0, s0, s1;
    char tmp[2048], rtmp[2048], relname[2048];
    size_t outlen;

    if (argc < 4) {
        fprintf(stderr,
            "usage: mkcofbatch SIDE0.cof SIDE1.cof OUT.raw [lpbr] [lpba]\n"
            "  joins the two --emit-cof files into cofactorisation batch input;\n"
            "  records already complete are written to OUT.raw.relations\n");
        return 2;
    }
    if (argc > 4) lpbr = atoi(argv[4]);
    if (argc > 5) lpba = atoi(argv[5]);
    outlen = strlen(argv[3]);
    if (outlen > sizeof rtmp - sizeof(".relations.tmp")) {
        fprintf(stderr, "mkcofbatch: output path too long\n");
        return 2;
    }
    /* Write through temporaries: a failed join must not leave a file that
     * looks like a usable batch. */
    memcpy(tmp, argv[3], outlen);
    memcpy(tmp + outlen, ".tmp", sizeof(".tmp"));
    memcpy(relname, argv[3], outlen);
    memcpy(relname + outlen, ".relations", sizeof(".relations"));
    memcpy(rtmp, argv[3], outlen);
    memcpy(rtmp + outlen, ".relations.tmp", sizeof(".relations.tmp"));

    f0 = fopen(argv[1], "r"); f1 = fopen(argv[2], "r");
    o = fopen(tmp, "w"); ro = fopen(rtmp, "w");
    if (!f0 || !f1 || !o || !ro) { perror("mkcofbatch"); return 1; }

    for (;;) {
        nline++;
        s0 = read_row(f0, &r0, argv[1], nline);
        s1 = read_row(f1, &r1, argv[2], nline);
        if (s0 == ROW_EOF && s1 == ROW_EOF) { nline--; break; }
        if (s0 == ROW_ERROR || s1 == ROW_ERROR) { rc = 1; break; }
        if (s0 == ROW_EOF || s1 == ROW_EOF) {
            fprintf(stderr, "mkcofbatch: %s ended at line %lu but %s did not;"
                    " the two sides are not the same survivor list\n",
                    s0 == ROW_EOF ? argv[1] : argv[2], nline,
                    s0 == ROW_EOF ? argv[2] : argv[1]);
            rc = 1; break;
        }
        if (r0.a != r1.a || r0.b != r1.b) {
            fprintf(stderr, "mkcofbatch: line %lu pairs (%lld,%lld) with"
                    " (%lld,%lld); the join is wrong\n",
                    nline, r0.a, r0.b, r1.a, r1.b);
            rc = 1; break;
        }
        if (s0 == ROW_SKIP || s1 == ROW_SKIP) continue;

        if (r0.bits <= lpbr && r1.bits <= lpba) {
            /* Already a relation: every prime is known and within bounds. The
             * residual cofactor is itself the last prime, and it has to be
             * printed in HEX like the rest of the list -- --emit-cof carries it
             * as decimal, and appending it verbatim silently produced a
             * relation whose factors did not multiply to the norm. It is at
             * most lpb bits here, so it fits a uint64. */
            fprintf(ro, "%lld,%lld:", r1.a, r1.b);
            put_factors(ro, &r0);
            if (r0.bits > 1)
                fprintf(ro, "%s%llx", r0.nfac ? "," : "",
                        (unsigned long long)strtoull(r0.cof, NULL, 10));
            fputc(':', ro);
            put_factors(ro, &r1);
            if (r1.bits > 1)
                fprintf(ro, "%s%llx", r1.nfac ? "," : "",
                        (unsigned long long)strtoull(r1.cof, NULL, 10));
            fputc('\n', ro);
            nfull++;
            continue;
        }

        fprintf(o, "%s%s,%s%s:%lld,%lld:",
                r0.bits > lpbr ? "-" : "", r0.cof,
                r1.bits > lpba ? "-" : "", r1.cof, r1.a, r1.b);
        put_factors(o, &r0);
        fputc(':', o);
        put_factors(o, &r1);
        fputc('\n', o);
        nout++;
    }

    fclose(f0); fclose(f1);
    /* a short write or a full disk shows up at fclose, not before */
    if (ferror(o) | ferror(ro)) rc = 1;
    if (fclose(o)) { perror("mkcofbatch: batch"); rc = 1; }
    if (fclose(ro)) { perror("mkcofbatch: relations"); rc = 1; }

    if (rc) {
        remove(tmp); remove(rtmp);
        fprintf(stderr, "mkcofbatch: FAILED, no output written\n");
        return 1;
    }
    if (rename(tmp, argv[3]) || rename(rtmp, relname)) {
        perror("mkcofbatch: rename"); remove(tmp); remove(rtmp); return 1;
    }
    printf("mkcofbatch: %lu survivors -> %lu to cofactor, %lu complete"
           " relations written to %s\n", nline, nout, nfull, relname);
    return 0;
}
