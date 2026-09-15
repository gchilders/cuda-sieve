/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * The CUB replacement: an exclusive prefix sum and a stable flagged
 * compaction over uint32, which is all fbgen_gpu.cu asks of
 * cub::DeviceScan::ExclusiveSum and cub::DeviceSelect::Flagged.
 *
 * hipCUB was a drop-in for the HIP port. Metal has no equivalent, so these
 * are hand-rolled -- but not invented: td.cuh:365-420 already carries a
 * three-pass block scan for exactly this job, and the tree already does
 * flagged compaction as scan-then-scatter (k_cof_selflags / k_cof_selscatter,
 * k_intersect_compact). td.cuh:455 records WHY that shape rather than
 * atomics: an ordered prefix scan gives every selected element a
 * deterministic slot, so the output is byte-reproducible run to run, and that
 * reproducibility is what makes diffing against the CPU generator a real
 * test. The same reasoning applies here, so the same shape is used.
 *
 * Determinism is free for the arithmetic itself -- these are uint32 adds,
 * which are associative and exact in any order -- but NOT for the output
 * order, which is why nothing here uses an atomic to allocate slots.
 *
 * Stability: k_select_scatter writes element i to offsets[i], and offsets is
 * a prefix sum taken in index order, so selected elements keep their relative
 * order. cub::DeviceSelect::Flagged guarantees the same.
 */
#include <metal_stdlib>
using namespace metal;

#define SCAN_BLK 256

/* One threadgroup's exclusive scan, plus that block's total for the next
 * level up. Hillis-Steele in threadgroup memory, the same structure as
 * td.cuh's k_scan_pass1. */
kernel void k_scan_block(device uint         *out   [[buffer(0)]],
                         device uint         *bsum  [[buffer(1)]],
                         device const uint   *in    [[buffer(2)]],
                         constant uint       &n     [[buffer(3)]],
                         threadgroup uint    *s     [[threadgroup(0)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint tgi [[threadgroup_position_in_grid]])
{
    const uint v = (gid < n) ? in[gid] : 0u;
    s[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < SCAN_BLK; off <<= 1) {
        uint add = (lid >= off) ? s[lid - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s[lid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < n) out[gid] = s[lid] - v;          /* inclusive -> exclusive */
    if (lid == SCAN_BLK - 1 && bsum) bsum[tgi] = s[lid];
}

/* Add each block's exclusive base back into its elements. */
kernel void k_scan_add(device uint       *out  [[buffer(0)]],
                       device const uint *base [[buffer(1)]],
                       constant uint     &n    [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tgi [[threadgroup_position_in_grid]])
{
    if (gid < n) out[gid] += base[tgi];
}

/* flags (bytes, any nonzero meaning selected) -> 0/1 words the scan can sum. */
kernel void k_select_flags01(device uint        *o     [[buffer(0)]],
                             device const uchar *flags [[buffer(1)]],
                             constant uint      &n     [[buffer(2)]],
                             uint gid    [[thread_position_in_grid]],
                             uint stride [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += stride) o[i] = flags[i] ? 1u : 0u;
}

kernel void k_select_scatter(device uint        *out     [[buffer(0)]],
                             device const uint  *in      [[buffer(1)]],
                             device const uchar *flags   [[buffer(2)]],
                             device const uint  *offsets [[buffer(3)]],
                             constant uint      &n       [[buffer(4)]],
                             uint gid    [[thread_position_in_grid]],
                             uint stride [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += stride)
        if (flags[i]) out[offsets[i]] = in[i];
}

/* num_selected = offsets[n-1] + flag01[n-1]. One thread; the alternative is a
 * device-to-host copy of two words, which costs more than this launch. */
kernel void k_select_total(device uint       *nsel    [[buffer(0)]],
                           device const uint *offsets [[buffer(1)]],
                           device const uint *flag01  [[buffer(2)]],
                           constant uint     &n       [[buffer(3)]])
{
    *nsel = n ? offsets[n - 1] + flag01[n - 1] : 0u;
}
