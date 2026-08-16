/* ckpt.h -- crash-safe checkpoint and resume for a --relations band.
 *
 * WHY A SIDECAR AND NOT THE RELATION FILE ITSELF. The obvious way to resume is
 * to read the tail of the relation file and work out which special-q produced
 * the last line, from the sq-side primes it carries. That is ambiguous under
 * the convention the pipeline actually runs: the FULL factor base is in force,
 * so a relation is re-found at every band q dividing its sq-side norm --
 * measured P(re-found) is 70.2%/72.3%, mean 1.82 sq-side primes per relation.
 * A line typically carries two in-band primes and nothing says which one found
 * it. Guessing high skips special-q and loses relations with no symptom;
 * guessing low only re-sieves an overlap. See STATUS.md item 12a.
 *
 * WHAT MAKES THE SIDECAR EXACT. pipeline.cuh flushes the cofactor queue BEFORE
 * enqueuing a q that would not fit, rather than splitting it, so a flush never
 * straddles a special-q. Immediately after a flush the .part file holds the
 * complete relation set for every q processed so far and nothing partial. That
 * instant -- and under --cofactor, ONLY that instant -- is a valid resume
 * point. Recording it costs an fsync and a rename.
 *
 * The stored byte offset is what makes a torn final line a non-problem: resume
 * truncates to it, so a partial write from a kill -9 is discarded rather than
 * parsed.
 */
#ifndef BENCH_CKPT_H
#define BENCH_CKPT_H

#include "bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CKPT_VERSION 1

typedef struct {
    char     fp[17];              /* job fingerprint, 16 hex digits + NUL      */
    uint64_t next_q, next_rho;    /* the first special-q NOT yet accounted for */
    unsigned long long rel_bytes; /* valid prefix of NAME.part                 */
    unsigned long long cand_bytes;/* valid prefix of CANDIDATES.part           */
    unsigned long long nrel;      /* relations in that prefix                  */
    unsigned long long nqdone;    /* special-q completed, all sessions         */
    /* Derived once from the FIRST q of the original band and held band-wide
     * (bench_main.cu, "params from q="). Re-deriving them from the resume q
     * would move the survivor gate partway through a run, so they are carried
     * rather than recomputed. */
    double   scale, scale0, allowance, allowance0;
} ckpt_t;

/* ---- paths ------------------------------------------------------------- */

static inline void ckpt_part_path(const char *relations, char *out, size_t n)
{
    snprintf(out, n, "%s.part", relations);
}

static inline void ckpt_ckpt_path(const char *relations, char *out, size_t n)
{
    snprintf(out, n, "%s.part.ckpt", relations);
}

static inline void ckpt_lock_path(const char *relations, char *out, size_t n)
{
    snprintf(out, n, "%s.lock", relations);
}

/* ---- fingerprint ------------------------------------------------------- */

/* FNV-1a over a canonical rendering of everything that decides WHICH relations
 * a run emits. Not a security hash: it exists to catch resuming one job's file
 * with another job's parameters, which produces lines that individually verify
 * and collectively mean nothing.
 *
 * The derived scale/allowance are deliberately NOT here. They depend on the
 * first q of the band, so they differ between the original run and a resumed
 * one by construction; they are carried in the checkpoint instead. */
static inline void ckpt_fp_feed(uint64_t *h, const char *s)
{
    for (; *s; s++) { *h ^= (unsigned char)*s; *h *= 1099511628211ULL; }
}

/* Appends and never advances past the buffer. The previous version accumulated
 * `k` from snprintf's would-have-written return value and kept forming
 * `out + k`, which is undefined once k exceeds n -- and, worse, silently
 * dropped the TAIL of the string on overflow. The tail is the sieve-parameter
 * block, so two jobs differing only in logI or lpb would have hashed
 * identically: the exact collision the fingerprint exists to prevent. */
static inline void ckpt_fp_cat(char *out, size_t n, size_t *k,
                               const char *fmt, ...)
{
    va_list ap;
    int w;
    if (*k >= n) return;
    va_start(ap, fmt);
    w = vsnprintf(out + *k, n - *k, fmt, ap);
    va_end(ap);
    *k = (w < 0 || (size_t)w >= n - *k) ? n : *k + (size_t)w;
}

static inline void ckpt_job_text(const poly_t *P, const bench_cfg_t *cfg,
                                 char *out, size_t n)
{
    size_t k = 0;
    int z;
    if (!n) return;
    out[0] = 0;
    ckpt_fp_cat(out, n, &k, "deg=%d skew=%.17g", P->deg, P->skew);
    for (z = 0; z <= P->deg && z < BENCH_NCOEFF; z++)
        ckpt_fp_cat(out, n, &k, " c%d=%s", z, P->cs[z]);
    ckpt_fp_cat(out, n, &k, " Y0=%s Y1=%s", P->y0s, P->y1s);
    ckpt_fp_cat(out, n, &k,
                " rlim=%u alim=%u lpbr=%u lpba=%u mfbr=%u mfba=%u"
                " rlambda=%.17g alambda=%.17g",
                P->rlim, P->alim, P->lpbr, P->lpba, P->mfbr, P->mfba,
                P->rlambda, P->alambda);
    ckpt_fp_cat(out, n, &k,
                " logI=%d J=%u sqside=%d nbe=%d"
                " lim0=%u lpb0=%u mfb0=%u lim1=%u lpb1=%u mfb1=%u",
                cfg->logI, cfg->J, cfg->sq_side, cfg->not_both_even,
                cfg->lim0, cfg->lpb0, cfg->mfb0,
                cfg->lim, cfg->lpb, cfg->mfb);
    /* The factor-base convention and the cofactor configuration belong here
     * too: STATUS.md 12a names the FB convention explicitly, and dropping
     * --cofactor on a resume changes the relation file from queue-emitted to
     * trial-division-only AND swaps the checkpoint discipline from
     * flush-anchored to a timer. The cofactor method and its bounds decide
     * which cofactors split at all, so they change the yield too. */
    ckpt_fp_cat(out, n, &k,
                " maxbits=%d cofac=%d ecm=%d b1=%u b2=%u curves=%u"
                " rounds=%d budget=%u",
                cfg->fb_maxbits, cfg->cofactor, cfg->cof_ecm,
                cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,
                cfg->cof_rounds, cfg->cof_budget);
}

static inline void ckpt_fingerprint(const poly_t *P, const bench_cfg_t *cfg,
                                    char out[17])
{
    char text[2048];
    uint64_t h = 1469598103934665603ULL;
    ckpt_job_text(P, cfg, text, sizeof text);
    ckpt_fp_feed(&h, text);
    snprintf(out, 17, "%016llx", (unsigned long long)h);
}

/* ---- write ------------------------------------------------------------- */

/* Atomic: a full write to a temporary, fsynced, then renamed over the live
 * checkpoint. A crash mid-write leaves the PREVIOUS checkpoint intact, which is
 * always a valid (merely older) resume point. The caller must have fsynced the
 * relation file itself BEFORE calling this -- the checkpoint asserts that those
 * bytes are durable. */
static inline int ckpt_write(const char *relations, const poly_t *P,
                             const bench_cfg_t *cfg, const ckpt_t *ck)
{
    char path[1200], tmp[1216], text[2048];
    FILE *f;
    ckpt_ckpt_path(relations, path, sizeof path);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (!(f = fopen(tmp, "w"))) { perror(tmp); return -1; }
    ckpt_job_text(P, cfg, text, sizeof text);
    fprintf(f, "# cuda-sieve resume checkpoint. Delete this file to force a"
               " fresh run.\n");
    fprintf(f, "# job: %s\n", text);
    fprintf(f, "version = %d\n", CKPT_VERSION);
    fprintf(f, "fingerprint = %s\n", ck->fp);
    fprintf(f, "next_q = %llu\n", (unsigned long long)ck->next_q);
    fprintf(f, "next_rho = %llu\n", (unsigned long long)ck->next_rho);
    fprintf(f, "rel_bytes = %llu\n", ck->rel_bytes);
    fprintf(f, "cand_bytes = %llu\n", ck->cand_bytes);
    fprintf(f, "relations = %llu\n", ck->nrel);
    fprintf(f, "nq_done = %llu\n", ck->nqdone);
    fprintf(f, "scale = %.17g\n", ck->scale);
    fprintf(f, "scale0 = %.17g\n", ck->scale0);
    fprintf(f, "allowance = %.17g\n", ck->allowance);
    fprintf(f, "allowance0 = %.17g\n", ck->allowance0);
    if (fflush(f) || fsync(fileno(f)) || fclose(f)) {
        perror(tmp); remove(tmp); return -1;
    }
    if (rename(tmp, path)) { perror(path); remove(tmp); return -1; }
    return 0;
}

/* ---- read -------------------------------------------------------------- */

/* 0 = loaded, -1 = absent or unreadable, -2 = present but malformed. A
 * malformed checkpoint is NOT treated as absent: silently starting over would
 * discard a file the operator believes is being resumed. */
static inline int ckpt_read(const char *relations, ckpt_t *ck)
{
    char path[1200], line[1024];
    FILE *f;
    int version = 0, got = 0;
    ckpt_ckpt_path(relations, path, sizeof path);
    if (!(f = fopen(path, "r"))) return -1;
    memset(ck, 0, sizeof *ck);
    while (fgets(line, sizeof line, f)) {
        char key[64]; char val[256];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%63s = %255s", key, val) != 2) continue;
        if      (!strcmp(key, "version"))     { version = atoi(val); got |= 1; }
        /* Length-checked rather than truncated: a short or overlong
         * fingerprint is a corrupt sidecar, and silently clipping one to 16
         * characters could make it compare equal to a real job's. */
        else if (!strcmp(key, "fingerprint")) {
            if (strlen(val) != 16) {
                fprintf(stderr, "resume: %s has a malformed fingerprint"
                        " (%zu characters, expected 16)\n", path, strlen(val));
                fclose(f); return -2;
            }
            memcpy(ck->fp, val, 16); ck->fp[16] = 0; got |= 2;
        }
        else if (!strcmp(key, "next_q"))      { ck->next_q = strtoull(val, 0, 10); got |= 4; }
        else if (!strcmp(key, "next_rho"))    { ck->next_rho = strtoull(val, 0, 10); got |= 8; }
        else if (!strcmp(key, "rel_bytes"))   { ck->rel_bytes = strtoull(val, 0, 10); got |= 16; }
        else if (!strcmp(key, "cand_bytes"))  ck->cand_bytes = strtoull(val, 0, 10);
        else if (!strcmp(key, "relations"))   ck->nrel = strtoull(val, 0, 10);
        else if (!strcmp(key, "nq_done"))     ck->nqdone = strtoull(val, 0, 10);
        else if (!strcmp(key, "scale"))       { ck->scale = strtod(val, 0); got |= 32; }
        else if (!strcmp(key, "scale0"))      { ck->scale0 = strtod(val, 0); got |= 64; }
        else if (!strcmp(key, "allowance"))   { ck->allowance = strtod(val, 0); got |= 128; }
        else if (!strcmp(key, "allowance0"))  { ck->allowance0 = strtod(val, 0); got |= 256; }
    }
    fclose(f);
    if (got != 511) {
        fprintf(stderr, "resume: %s is missing required fields\n", path);
        return -2;
    }
    if (version != CKPT_VERSION) {
        fprintf(stderr, "resume: %s is version %d, this build writes %d\n",
                path, version, CKPT_VERSION);
        return -2;
    }
    if (ck->scale <= 0.0 || ck->scale0 <= 0.0) {
        fprintf(stderr, "resume: %s carries a degenerate scale\n", path);
        return -2;
    }
    return 0;
}

/* ---- lock -------------------------------------------------------------- */

/* Two sievers appending to one .part is silent corruption, and a job queue
 * makes that a realistic accident rather than a theoretical one. O_EXCL plus
 * the pid.
 *
 * A stale lock CLEARS ITSELF. The crash cases resume exists for -- SIGKILL,
 * OOM, power loss, and the second-signal _exit -- all skip the unlink, and a
 * lock that then needs a human to remove defeats unattended operation more
 * thoroughly than the race it prevents. So an existing lock naming a pid that
 * no longer exists is taken over. The residual hazard is pid reuse: a lock
 * whose pid has been recycled onto an unrelated process reads as live and is
 * refused, which is the safe direction. */
static inline int ckpt_lock(const char *relations, char *path, size_t n)
{
    int fd, retried = 0;
    char buf[64];
    ckpt_lock_path(relations, path, n);
again:
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        if (errno == EEXIST) {
            FILE *f = fopen(path, "r");
            long other = 0;
            if (f) { if (fscanf(f, "%ld", &other) != 1) other = 0; fclose(f); }
            if (!retried && other > 0 && kill((pid_t)other, 0) &&
                errno == ESRCH) {
                fprintf(stderr, "  %s is stale (pid %ld is gone); taking it"
                        " over\n", path, other);
                remove(path);
                retried = 1;
                goto again;
            }
            fprintf(stderr,
                    "  %s exists: another run (pid %ld) is writing these"
                    " files.\n"
                    "  If it is not running, remove the lock and retry.\n",
                    path, other);
        } else perror(path);
        return -1;
    }
    snprintf(buf, sizeof buf, "%ld\n", (long)getpid());
    if (write(fd, buf, strlen(buf)) < 0) { /* advisory only */ }
    close(fd);
    return 0;
}

/* Held for the process lifetime and released via atexit, so every early
 * `return 1` between acquisition and the band drops it. _exit and fatal
 * signals still skip it -- that is what the staleness check above is for. */
static char ckpt_lock_held[2048];

static inline void ckpt_unlock_atexit(void)
{
    if (ckpt_lock_held[0]) { remove(ckpt_lock_held); ckpt_lock_held[0] = 0; }
}

static inline void ckpt_unlock(const char *path)
{
    if (path && *path) remove(path);
}

#ifdef __cplusplus
}
#endif
#endif
