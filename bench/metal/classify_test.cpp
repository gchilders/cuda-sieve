/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Runs cof_classify on the GPU and on the host, over cofactor-shaped inputs,
 * and compares every verdict. The host side compiles prp.cuh UNMODIFIED, so
 * the reference is the code the CUDA build runs.
 */
#include "prp.cuh"
#include "metal/metal_rt.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static uint64_t rs = 0xC0FFEE1234567ull;
static uint64_t rnd(void)
{ rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1Dull; }

int main(int argc, char **argv)
{
    if (mtlInit(argc > 1 ? argv[1] : "bench.metallib") != mtlSuccess) return 1;
    const uint32_t N = 1u << 16;
    const uint32_t lpb = 32, mfb = 92;
    const double lim = 134200000.0;

    uint32_t *d_limbs; uint8_t *d_bits, *d_out;
    mtlMalloc((void **)&d_limbs, (size_t)N * BN_LIMBS * 4);
    mtlMalloc((void **)&d_bits, N);
    mtlMalloc((void **)&d_out, N);

    std::vector<bn_t> host(N);
    for (uint32_t i = 0; i < N; i++) {
        memset(&host[i], 0, sizeof(bn_t));
        /* cofactor-shaped: 1 to 4 limbs, i.e. up to 128 bits, which is the
         * window cof_classify is written for. */
        int limbs = 1 + (int)(rnd() % 4);
        for (int k = 0; k < limbs; k++) host[i].v[k] = (uint32_t)rnd();
        /* bits = true bit length */
        int b = 0;
        for (int k = BN_LIMBS - 1; k >= 0; k--)
            if (host[i].v[k]) { b = k * 32 + (32 - __builtin_clz(host[i].v[k])); break; }
        d_bits[i] = (uint8_t)b;
        memcpy(&d_limbs[i * BN_LIMBS], &host[i], sizeof(bn_t));
    }

    mtlError_t e = mtl_launch_begin("k_classify_probe", 0, N / 256, 256, 0, 0u);
    if (e != mtlSuccess) { printf("launch: %s\n", mtlGetErrorString(e)); return 1; }
    mtl_bind_buffer(d_out, 0); mtl_bind_buffer(d_limbs, 1); mtl_bind_buffer(d_bits, 2);
    mtl_bind_bytes(&lpb, 4, 3); mtl_bind_bytes(&mfb, 4, 4);
    mtl_bind_bytes(&lim, 8, 5); mtl_bind_bytes(&N, 4, 6);
    if (mtl_launch_end() != mtlSuccess) { printf("launch_end failed\n"); return 1; }
    mtlDeviceSynchronize();

    long bad = 0; long hist[8] = {0};
    for (uint32_t i = 0; i < N; i++) {
        int want = cof_classify(&host[i], (int)d_bits[i], lpb, mfb, lim);
        hist[want & 7]++;
        if (d_out[i] != (uint8_t)want) {
            if (bad < 6)
                printf("  [%u] bits=%u gpu=%u host=%d\n", i, d_bits[i], d_out[i], want);
            bad++;
        }
    }
    static const char *nm[6] = {"REJECT_MFB","REJECT_GAP","REJECT_PRIME",
                                "ACCEPT","SPLIT","DEGENERATE"};
    printf("host verdicts:");
    for (int k = 0; k < 6; k++) printf(" %s=%ld", nm[k], hist[k]);
    printf("\n%ld of %u verdicts differ\n", bad, N);
    printf("%s\n", bad ? "CLASSIFY GATE: FAIL" : "CLASSIFY GATE: PASS");
    mtlShutdown();
    return bad ? 1 : 0;
}
