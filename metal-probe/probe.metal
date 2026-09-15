#include <metal_stdlib>
using namespace metal;

kernel void k_warp(device uint *o [[buffer(0)]],
                   uint t [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint w [[threads_per_simdgroup]])
{
    if (t >= 32) return;
    /* ballot over "lane is odd": ascending lane order => bit i set iff lane i odd */
    ulong b = (ulong)((simd_vote::vote_t)simd_ballot((lane & 1u) != 0u));
    o[0] = w;                                   /* simd width            */
    if (t == 0) { o[1] = (uint)(b & 0xffffffffu); o[2] = (uint)(b >> 32); }
    o[3 + lane] = simd_shuffle_up(lane * 7u, 1u);   /* shuffle semantics */
    o[35 + lane] = simd_prefix_exclusive_sum(1u);   /* == lane           */
}

kernel void k_mulhi(device ulong *o [[buffer(0)]], device const ulong *a [[buffer(1)]],
                    uint t [[thread_position_in_grid]])
{ o[t] = mulhi(a[2*t], a[2*t+1]); }

/* log2: default vs precise, on host-supplied inputs */
kernel void k_log2(device float *od [[buffer(0)]], device float *op [[buffer(1)]],
                   device const float *in [[buffer(2)]], uint t [[thread_position_in_grid]])
{ od[t] = log2(in[t]); op[t] = precise::log2(in[t]); }
