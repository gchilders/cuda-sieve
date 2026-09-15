/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Host driver for scan.metal. Plain C++ -- it goes through metal_rt.h like
 * any other ported orchestration will.
 *
 * The scan is recursive: each level reduces n by SCAN_BLK, so 8M elements
 * (GPU_FB_DEFAULT_SEG_ODDS) is three levels. All scratch comes out of the
 * caller's single temp allocation, carved deterministically so that the
 * size reported by the query call is exactly the size the real call uses.
 */
#include "metal_scan.h"
#include <cstdio>
#include <cstring>

#define SCAN_BLK 256u

namespace {

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/* Scratch for one scan: two uint32 arrays per recursion level. */
size_t scan_scratch(uint32_t n)
{
    size_t total = 0;
    while (n > SCAN_BLK) {
        uint32_t nb = (n + SCAN_BLK - 1) / SCAN_BLK;
        total += 2 * align_up((size_t)nb * 4, 256);
        n = nb;
    }
    return total;
}

mtlError_t scan_rec(uint32_t *out, const uint32_t *in, uint32_t n,
                    char *scratch, mtlStream_t s)
{
    if (!n) return mtlSuccess;
    const uint32_t nb = (n + SCAN_BLK - 1) / SCAN_BLK;
    mtlError_t e;

    if (nb == 1) {
        e = MTL_LAUNCH(k_scan_block, 1, SCAN_BLK, SCAN_BLK * 4, s,
                       out, (uint32_t *)nullptr, in, n);
        return e;
    }

    const size_t stride = align_up((size_t)nb * 4, 256);
    uint32_t *bsum  = (uint32_t *)scratch;
    uint32_t *bscan = (uint32_t *)(scratch + stride);
    char     *rest  = scratch + 2 * stride;

    e = MTL_LAUNCH(k_scan_block, nb, SCAN_BLK, SCAN_BLK * 4, s, out, bsum, in, n);
    if (e != mtlSuccess) return e;
    e = scan_rec(bscan, bsum, nb, rest, s);
    if (e != mtlSuccess) return e;
    return MTL_LAUNCH(k_scan_add, nb, SCAN_BLK, 0, s, out, (const uint32_t *)bscan, n);
}

}  /* namespace */

extern "C" mtlError_t mtlScanExclusiveSumU32(void *temp, size_t *temp_bytes,
                                             const uint32_t *in, uint32_t *out,
                                             uint32_t n, mtlStream_t s)
{
    if (!temp_bytes) return mtlErrorInvalidValue;
    if (!temp) { *temp_bytes = scan_scratch(n); if (!*temp_bytes) *temp_bytes = 256; return mtlSuccess; }
    return scan_rec(out, in, n, (char *)temp, s);
}

extern "C" mtlError_t mtlSelectFlaggedU32(void *temp, size_t *temp_bytes,
                                          const uint32_t *in, const uint8_t *flags,
                                          uint32_t *out, uint32_t *num_selected,
                                          uint32_t n, mtlStream_t s)
{
    if (!temp_bytes) return mtlErrorInvalidValue;
    const size_t words = align_up((size_t)n * 4, 256);
    if (!temp) {
        *temp_bytes = 2 * words + scan_scratch(n);
        if (!*temp_bytes) *temp_bytes = 256;
        return mtlSuccess;
    }
    if (!n) {
        /* CUB writes a zero count for an empty input; match it. */
        return MTL_LAUNCH(k_select_total, 1, 1, 0, s, num_selected,
                          (const uint32_t *)temp, (const uint32_t *)temp, 0u);
    }

    char     *base    = (char *)temp;
    uint32_t *flag01  = (uint32_t *)base;
    uint32_t *offsets = (uint32_t *)(base + words);
    char     *scratch = base + 2 * words;

    const unsigned grid = 256, block = 256;
    mtlError_t e = MTL_LAUNCH(k_select_flags01, grid, block, 0, s, flag01, flags, n);
    if (e != mtlSuccess) return e;
    e = scan_rec(offsets, flag01, n, scratch, s);
    if (e != mtlSuccess) return e;
    e = MTL_LAUNCH(k_select_scatter, grid, block, 0, s,
                   out, in, flags, (const uint32_t *)offsets, n);
    if (e != mtlSuccess) return e;
    return MTL_LAUNCH(k_select_total, 1, 1, 0, s, num_selected,
                      (const uint32_t *)offsets, (const uint32_t *)flag01, n);
}
