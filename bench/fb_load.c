/* GGNFS .afb.0 loader. Format decoded in oracle/ggnfs_afb_format.txt. */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int fb_load(const char *path, fb_t *fb)
{
    FILE *f = fopen(path, "rb");
    long sz;
    uint32_t n;
    size_t got;

    memset(fb, 0, sizeof(*fb));
    if (!f) { fprintf(stderr, "fb_load: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);

    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    if (n == 0 || (long)(4 * (1 + 2ull * n)) > sz) {
        fprintf(stderr, "fb_load: header count %u inconsistent with size %ld\n", n, sz);
        fclose(f); return -1;
    }

    fb->n      = n;
    fb->primes = (uint32_t *)malloc((size_t)n * 4);
    fb->roots  = (uint32_t *)malloc((size_t)n * 4);
    if (!fb->primes || !fb->roots) { fclose(f); return -1; }

    got  = fread(fb->primes, 4, n, f);
    got += fread(fb->roots,  4, n, f);
    fclose(f);
    if (got != 2ull * n) { fprintf(stderr, "fb_load: short read\n"); return -1; }
    return 0;
}

void fb_free(fb_t *fb)
{
    free(fb->primes); free(fb->roots); free(fb->logp);
    fb->primes = fb->roots = NULL; fb->logp = NULL; fb->n = 0;
}

void fb_fill_logp(fb_t *fb, double scale)
{
    uint32_t i;
    if (fb->logp) return;
    fb->logp = (uint8_t *)malloc(fb->n ? fb->n : 1);
    if (!fb->logp) return;
    for (i = 0; i < fb->n; i++) {
        long l = (long)floor(log2((double)fb->primes[i]) * scale + 0.5);
        fb->logp[i] = (uint8_t)(l < 0 ? 0 : (l > 255 ? 255 : l));
    }
}

void fb_restrict(fb_t *fb, uint32_t bkthresh, uint32_t fb_bound)
{
    uint32_t i, k = 0;
    for (i = 0; i < fb->n; i++) {
        uint32_t p = fb->primes[i], r = fb->roots[i];
        if (p < bkthresh || p >= fb_bound) continue;
        if (r >= p) continue;              /* projective; encoded r == p */
        fb->primes[k] = p; fb->roots[k] = r;
        if (fb->logp) fb->logp[k] = fb->logp[i];
        k++;
    }
    fb->n = k;
}

/* Copy out the entries below bkthresh -- these are line-sieved per region, not
 * bucketed. Projective entries (r == p) are KEPT here: all four of them on this
 * job have p < 12, and between them they account for ~0.77*A updates, more than
 * the entire bucket-sieve volume. Dropping them is not an option. */
int fb_split_small(const fb_t *fb, uint32_t bkthresh, fb_t *small)
{
    uint32_t i, k = 0;
    memset(small, 0, sizeof(*small));
    for (i = 0; i < fb->n && fb->primes[i] < bkthresh; i++) k++;
    small->n = k;
    small->primes = (uint32_t *)malloc((size_t)(k ? k : 1) * 4);
    small->roots  = (uint32_t *)malloc((size_t)(k ? k : 1) * 4);
    if (!small->primes || !small->roots) return -1;
    memcpy(small->primes, fb->primes, (size_t)k * 4);
    memcpy(small->roots,  fb->roots,  (size_t)k * 4);
    if (fb->logp) {
        small->logp = (uint8_t *)malloc((size_t)(k ? k : 1));
        if (!small->logp) return -1;
        memcpy(small->logp, fb->logp, (size_t)k);
    }
    return 0;
}
