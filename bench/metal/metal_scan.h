/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * CUB's two entry points, with CUB's calling convention.
 *
 * fbgen_gpu.cu calls these eight times, always in the query-then-use pattern:
 * pass a NULL temp buffer to learn the size, allocate once, then pass it on
 * every real call. Keeping that convention exactly means each of those eight
 * lines ports by changing the name and nothing else.
 */
#ifndef CUDA_SIEVE_METAL_SCAN_H
#define CUDA_SIEVE_METAL_SCAN_H

#include "metal_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exclusive prefix sum over n uint32. `in` and `out` may not overlap.
 * temp == NULL: set *temp_bytes and return. */
mtlError_t mtlScanExclusiveSumU32(void *temp, size_t *temp_bytes,
                                  const uint32_t *in, uint32_t *out,
                                  uint32_t n, mtlStream_t s);

/* Stable compaction: out[] gets in[i] for every i with flags[i] != 0, in
 * index order; *num_selected gets the count (a device pointer, as CUB's is). */
mtlError_t mtlSelectFlaggedU32(void *temp, size_t *temp_bytes,
                               const uint32_t *in, const uint8_t *flags,
                               uint32_t *out, uint32_t *num_selected,
                               uint32_t n, mtlStream_t s);

#ifdef __cplusplus
}
#endif
#endif
