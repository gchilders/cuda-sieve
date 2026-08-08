/* Shared odd-only Eratosthenes sieve for host-side factor-base builders. */
#include "bench.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t *prime_list_build(uint32_t lim, size_t *count)
{
    size_t nodd = lim >= 3 ? ((size_t)lim - 1) / 2 : 0;
    unsigned char *bits = (unsigned char *)calloc(nodd / 8 + 1, 1);
    uint32_t *primes;
    size_t cap, n = 0, i;

    *count = 0;
    if (!bits) return NULL;
    for (i = 0; i < nodd; i++) {
        uint32_t p = (uint32_t)(2 * i + 3);
        size_t j;
        if ((uint64_t)p * p > lim) break;
        if (bits[i >> 3] & (1u << (i & 7))) continue;
        for (j = ((size_t)p * p - 3) / 2; j < nodd; j += p)
            bits[j >> 3] |= (unsigned char)(1u << (j & 7));
    }

    cap = lim > 10 ? (size_t)((double)lim / (log((double)lim) - 1.1)) + 1024 : 16;
    primes = (uint32_t *)malloc(cap * sizeof(*primes));
    if (!primes) { free(bits); return NULL; }
    if (lim >= 2) primes[n++] = 2;
    for (i = 0; i < nodd; i++) {
        uint32_t p = (uint32_t)(2 * i + 3);
        if (p > lim) break;
        if (bits[i >> 3] & (1u << (i & 7))) continue;
        if (n == cap) {
            uint32_t *grown;
            cap *= 2;
            grown = (uint32_t *)realloc(primes, cap * sizeof(*primes));
            if (!grown) { free(primes); free(bits); return NULL; }
            primes = grown;
        }
        primes[n++] = p;
    }
    free(bits);
    *count = n;
    return primes;
}
