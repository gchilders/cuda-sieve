/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 4a gate: the CUB replacement against a host reference.
 *
 * Sizes are chosen to hit the places a block scan actually breaks: exactly on
 * and either side of the 256 block boundary, exactly on and either side of
 * the 65536 two-level boundary, and past 8M, which is
 * GPU_FB_DEFAULT_SEG_ODDS and therefore the largest n fbgen_gpu will ask for.
 * Values are large enough to wrap uint32 on the way, because a scan that
 * silently promotes to 64-bit somewhere would pass on small data and produce
 * a different factor base on real data.
 */
#include "metal_scan.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;

static uint64_t rs = 0x243F6A8885A308D3ull;
static uint32_t rnd(void)
{ rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return (uint32_t)(rs * 0x2545F4914F6CDD1Dull >> 32); }

static void check_scan(uint32_t n, bool bigvals)
{
    uint32_t *d_in = nullptr, *d_out = nullptr;
    void *d_temp = nullptr;
    size_t tb = 0;
    if (mtlMalloc((void **)&d_in, (size_t)n * 4 + 4) != mtlSuccess ||
        mtlMalloc((void **)&d_out, (size_t)n * 4 + 4) != mtlSuccess) {
        printf("  scan n=%-9u ALLOC FAILED\n", n); failures++; return;
    }
    for (uint32_t i = 0; i < n; i++) d_in[i] = bigvals ? rnd() : (rnd() & 0xff);

    mtlScanExclusiveSumU32(nullptr, &tb, d_in, d_out, n, 0);
    if (mtlMalloc(&d_temp, tb) != mtlSuccess) { printf("  temp alloc failed\n"); failures++; return; }
    memset(d_out, 0xCD, (size_t)n * 4);
    mtlError_t e = mtlScanExclusiveSumU32(d_temp, &tb, d_in, d_out, n, 0);
    mtlDeviceSynchronize();

    uint32_t acc = 0; bool good = (e == mtlSuccess);
    for (uint32_t i = 0; i < n && good; i++) {
        if (d_out[i] != acc) {
            printf("  scan n=%u FAIL at %u: got %u want %u\n", n, i, d_out[i], acc);
            good = false;
        }
        acc += d_in[i];
    }
    printf("  exclusive scan n=%-9u %-8s %s  (temp %zu B)\n", n,
           bigvals ? "wrapping" : "small", good ? "PASS" : "FAIL", tb);
    if (!good) failures++;
    mtlFree(d_in); mtlFree(d_out); mtlFree(d_temp);
}

static void check_select(uint32_t n, int density)
{
    uint32_t *d_in = nullptr, *d_out = nullptr, *d_nsel = nullptr;
    uint8_t *d_flags = nullptr;
    void *d_temp = nullptr;
    size_t tb = 0;
    mtlMalloc((void **)&d_in, (size_t)n * 4 + 4);
    mtlMalloc((void **)&d_out, (size_t)n * 4 + 4);
    mtlMalloc((void **)&d_flags, (size_t)n + 4);
    mtlMalloc((void **)&d_nsel, 4);

    std::vector<uint32_t> want;
    want.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        d_in[i] = rnd();
        int f = (density == 0) ? 0 : (density == 100) ? 1 : ((int)(rnd() % 100) < density);
        d_flags[i] = (uint8_t)(f ? (1 + (rnd() & 0x7f)) : 0);   /* any nonzero counts */
        if (f) want.push_back(d_in[i]);
    }

    mtlSelectFlaggedU32(nullptr, &tb, d_in, d_flags, d_out, d_nsel, n, 0);
    mtlMalloc(&d_temp, tb);
    memset(d_out, 0xCD, (size_t)n * 4);
    *d_nsel = 0xFFFFFFFFu;
    mtlError_t e = mtlSelectFlaggedU32(d_temp, &tb, d_in, d_flags, d_out, d_nsel, n, 0);
    mtlDeviceSynchronize();

    bool good = (e == mtlSuccess) && (*d_nsel == want.size());
    for (size_t i = 0; i < want.size() && good; i++)
        if (d_out[i] != want[i]) {
            printf("  select n=%u FAIL at %zu: got %u want %u\n", n, i, d_out[i], want[i]);
            good = false;
        }
    printf("  flagged select n=%-9u %3d%% dense  %s  (%u selected)\n",
           n, density, good ? "PASS" : "FAIL", *d_nsel);
    if (!good) failures++;
    mtlFree(d_in); mtlFree(d_out); mtlFree(d_flags); mtlFree(d_nsel); mtlFree(d_temp);
}

int main(int argc, char **argv)
{
    if (mtlInit(argc > 1 ? argv[1] : "scan_test.metallib") != mtlSuccess) return 1;

    printf("== exclusive prefix sum ==\n");
    const uint32_t sizes[] = { 0, 1, 2, 255, 256, 257, 511, 512, 1000,
                               65535, 65536, 65537, 1u << 20, 8u << 20 };
    for (uint32_t n : sizes) if (n) check_scan(n, false);
    check_scan(1u << 20, true);       /* wraps uint32 many times over */
    check_scan(8u << 20, true);

    printf("\n== stable flagged compaction ==\n");
    for (uint32_t n : { 1u, 255u, 256u, 257u, 65536u, 65537u, 1u << 20 }) {
        check_select(n, 50);
    }
    check_select(1u << 16, 0);        /* nothing selected   */
    check_select(1u << 16, 100);      /* everything selected */
    check_select(8u << 20, 7);        /* the sieve's real density */

    mtlShutdown();
    printf("\n%s\n", failures ? "PHASE 4a GATE: FAIL" : "PHASE 4a GATE: PASS");
    return failures ? 1 : 0;
}
