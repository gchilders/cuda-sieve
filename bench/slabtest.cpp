/* CPU-only gates for j-slabbing. No CUDA runtime or GPU required. */
#include "bench.h"
#include "plattice.cuh"
#include "td.cuh"
#include <stdio.h>
#include <stdint.h>

static uint32_t mod_magic_host(uint32_t w, uint32_t m, uint32_t magic,
                               uint32_t sh)
{
    const uint32_t q = (uint32_t)(((uint64_t)w * magic) >> 32) >> sh;
    return w - q * m;
}

static int check_plan(void)
{
    slab_plan_t P = {0,0,0};
    const struct {
        int logI; uint32_t J, pmax, forceJ, want_j, want_n;
    } v[] = {
        {15, 16384u, 32767u, 0u,     16384u, 1u},
        {16, 32768u, 65535u, 0u,     32768u, 1u},
        {17, 65536u,131071u, 0u,     16384u, 4u},
        {18,131072u,262143u, 0u,      8192u,16u},
        {19,262144u,524287u, 0u,      4096u,64u},
        {20,524288u,1048575u,0u,      2048u,256u},
        {15, 16384u, 32767u, 123u,      123u,134u},
        {17, 65536u,262143u, 0u,      8192u, 8u},
    };
    for (unsigned k = 0; k < sizeof(v)/sizeof(v[0]); k++) {
        if (slab_make_plan(v[k].logI, v[k].J, v[k].pmax, v[k].forceJ, &P) ||
            P.jmax != v[k].want_j || P.nslab != v[k].want_n ||
            P.enabled != (v[k].want_n > 1u)) {
            fprintf(stderr, "slabtest: plan %u got jmax=%u n=%u enabled=%d;"
                    " want %u/%u/%d\n", k, P.jmax, P.nslab, P.enabled,
                    v[k].want_j, v[k].want_n, v[k].want_n > 1u);
            return -1;
        }
        {
            uint64_t covered = 0;
            for (uint32_t slab = 0; slab < P.nslab; slab++) {
                const uint32_t base = slab_jbase_at(&P, slab);
                const uint32_t rows = slab_rows_at(&P, v[k].J, slab);
                if ((uint64_t)base != covered || !rows || rows > P.jmax) {
                    fprintf(stderr,
                            "slabtest: plan %u coverage error at slab %u:"
                            " base=%u covered=%llu rows=%u jmax=%u\n",
                            k, slab, base, (unsigned long long)covered, rows, P.jmax);
                    return -1;
                }
                covered += rows;
            }
            if (covered != v[k].J) {
                fprintf(stderr, "slabtest: plan %u covered %llu rows, want %u\n",
                        k, (unsigned long long)covered, v[k].J);
                return -1;
            }
        }
    }
    /* A forced slab is allowed to be smaller, never larger than a safety cap. */
    if (!slab_make_plan(17, 65536u, 131071u, 20000u, &P)) {
        fprintf(stderr, "slabtest: unsafe forced slab height was accepted\n");
        return -1;
    }
    return 0;
}

static int check_phase(void)
{
    uint32_t seed = 0x6d2b79f5u;
    for (unsigned t = 0; t < 20000; t++) {
        const int logI = 10 + (int)(seed % 11u);
        const uint32_t I = 1u << logI;
        uint32_t m, rt, cst, base, local, hi, shifted;
        seed = seed * 1664525u + 1013904223u;
        m = 2u + seed % (I - 1u);
        seed = seed * 1664525u + 1013904223u;
        rt = seed % m;
        seed = seed * 1664525u + 1013904223u;
        base = seed & ((1u << 20) - 1u);
        seed = seed * 1664525u + 1013904223u;
        local = seed & 2047u;
        seed = seed * 1664525u + 1013904223u;
        hi = 1u + seed % I;
        cst = (I >> 1) % m;
        shifted = slab_phase_cst(cst, rt, m, base);
        {
            const uint32_t a = (uint32_t)(((uint64_t)rt * (base + local) + hi) % m);
            const uint32_t b = (uint32_t)(((uint64_t)rt * local + hi) % m);
            if ((a == cst) != (b == shifted)) {
                fprintf(stderr,
                        "slabtest: phase mismatch logI=%d m=%u rt=%u base=%u"
                        " local=%u hi=%u cst=%u shifted=%u a=%u b=%u\n",
                        logI, m, rt, base, local, hi, cst, shifted, a, b);
                return -1;
            }
        }
    }
    return 0;
}

static int check_walk_continuation(void)
{
    static const struct { int logI; uint32_t J, slabJ; } v[] = {
        {8, 512u, 127u}, {9, 512u, 129u},
        {10, 1024u, 257u}, {12, 2048u, 511u}
    };
    for (unsigned k = 0; k < sizeof(v)/sizeof(v[0]); k++) {
        const int checked = verify_walk_slabs(v[k].logI, v[k].J,
                                               v[k].slabJ, 24);
        if (checked != 24) {
            fprintf(stderr,
                    "slabtest: walk continuation case %u checked %d/24 primes\n",
                    k, checked);
            return -1;
        }
    }
    return 0;
}

static int check_magic(void)
{
    uint32_t seed = 0x31415926u;
    /* Edge divisors plus randomized values through the largest default I. */
    for (unsigned t = 0; t < 20000; t++) {
        uint32_t m, magic, sh;
        if (t < 20) {
            static const uint32_t edge[] = {2,3,4,5,7,8,15,16,31,32,63,64,
                                             127,128,255,256,1023,32767,131071,1048575};
            m = edge[t];
        } else {
            seed = seed * 1664525u + 1013904223u;
            m = 2u + seed % 1048574u;
        }
        td_magic_build(m, &magic, &sh);
        for (unsigned k = 0; k < 12; k++) {
            uint32_t w;
            if (k == 0) w = 0;
            else if (k == 1) w = 1;
            else if (k == 2) w = m - 1u;
            else if (k == 3) w = m;
            else if (k == 4) w = 0x7fffffffu;
            else {
                seed = seed * 1664525u + 1013904223u;
                w = seed & 0x7fffffffu;
            }
            if (mod_magic_host(w, m, magic, sh) != w % m) {
                fprintf(stderr,
                        "slabtest: magic mismatch m=%u w=%u magic=%u sh=%u"
                        " got=%u ref=%u\n", m, w, magic, sh,
                        mod_magic_host(w, m, magic, sh), w % m);
                return -1;
            }
        }
    }
    return 0;
}

int main(void)
{
    if (check_plan() || check_phase() || check_magic() ||
        check_walk_continuation()) return 1;
    printf("slabtest: plan, phase, reciprocal, and continued-walk gates OK\n");
    return 0;
}
