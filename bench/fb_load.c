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

static uint32_t mulmod32(uint32_t a, uint32_t b, uint32_t m)
{
    return (uint32_t)(((uint64_t)a * b) % m);
}

static uint32_t powmod32(uint32_t a, uint32_t e, uint32_t m)
{
    uint32_t r = 1;
    a %= m;
    while (e) { if (e & 1u) r = mulmod32(r, a, m); a = mulmod32(a, a, m); e >>= 1; }
    return r;
}

/* Deterministic Miller-Rabin: bases {2,7,61} decide primality for all 32-bit n. */
static int is_prime32(uint32_t n)
{
    static const uint32_t bases[3] = {2u, 7u, 61u};
    uint32_t d = n - 1, i;
    int s = 0, k;
    if (n < 2) return 0;
    for (i = 0; i < 3; i++) { if (n == bases[i]) return 1; if (n % bases[i] == 0) return 0; }
    while (!(d & 1u)) { d >>= 1; s++; }
    for (k = 0; k < 3; k++) {
        uint32_t x = powmod32(bases[k], d, n);
        int j;
        if (x == 1 || x == n - 1) continue;
        for (j = 1; j < s; j++) { x = mulmod32(x, x, n); if (x == n - 1) break; }
        if (j == s) return 0;
    }
    return 1;
}

/* Is q a prime power (including a prime)?
 *
 * Primality first, by Miller-Rabin: a factor base is almost entirely primes,
 * and trial division to sqrt(q) on 11.5M of them costs minutes. Only the
 * composites -- a couple of hundred -- pay for a factor search, and their base
 * prime is at most sqrt(q), so the search is short. */
int fb_is_prime_power(uint32_t q)
{
    uint32_t d;
    if (q < 2) return 0;
    if (is_prime32(q)) return 1;
    for (d = 2; (uint64_t)d * d <= (uint64_t)q; d++) {
        if (q % d) continue;
        while (q % d == 0) q /= d;
        return q == 1;
    }
    return 0;                    /* composite with no factor <= sqrt: impossible */
}

/* Every modulus in a factor base must be a prime power: the root transform's
 * "solutions exist only when g | j" step depends on it (see plattice.cuh), and
 * a composite modulus with two prime factors would be silently mis-placed
 * rather than rejected. makefb and rfb_build both satisfy this; this is the
 * gate that notices if a hand-edited or third-party file does not. Returns the
 * first offending index, or -1 if the base is clean. */
int32_t fb_check_prime_powers(const fb_t *fb)
{
    uint32_t i;
    for (i = 0; i < fb->n; i++)
        if (!fb_is_prime_power(fb->primes[i])) return (int32_t)i;
    return -1;
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

/* Keep the bucketed window [bkthresh, fb_bound).
 *
 * Projective entries (root >= p) are KEPT. Dropping them used to look safe on
 * the argument that a bucketed p exceeds J, so a projective ideal can only hit
 * row j = 0 -- but that argument confuses the two coordinates. The condition is
 * on b, and b = i*a1 + j*b1; for a special-q of this size the reduced basis has
 * b-components of order sqrt(q/skew) ~ 1, and on the q = 120000053 lattice it
 * is exactly b = i. So a projective ideal hits i == 0 (mod p) on EVERY row:
 * 16,384 positions per entry here, not one row's worth. Side 0 has two of them
 * (38321 and 5746453, both dividing Y1). Let the transform decide -- it now
 * handles the encoding, and reports what it genuinely cannot walk. */
void fb_restrict(fb_t *fb, uint32_t bkthresh, uint32_t fb_bound)
{
    uint32_t i, k = 0;
    for (i = 0; i < fb->n; i++) {
        uint32_t p = fb->primes[i], r = fb->roots[i];
        if (p < bkthresh || p >= fb_bound) continue;
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
