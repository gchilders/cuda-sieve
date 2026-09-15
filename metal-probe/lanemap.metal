#include <metal_stdlib>
using namespace metal;
/* CUDA code computes lane = threadIdx.x & 31 and warp = threadIdx.x >> 5, then
 * synchronises with __syncwarp(). That is only valid if the thread's position
 * in the threadgroup maps to its SIMD lane in exactly that way. Phase 0
 * verified simd_ballot ordering and simd_shuffle semantics but never THIS. */
kernel void k_lanemap(device uint *o [[buffer(0)]],
                      uint lid   [[thread_position_in_threadgroup]],
                      uint lane  [[thread_index_in_simdgroup]],
                      uint sg    [[simdgroup_index_in_threadgroup]])
{
    o[lid * 3 + 0] = lid;
    o[lid * 3 + 1] = lane;
    o[lid * 3 + 2] = sg;
}
