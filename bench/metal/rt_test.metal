/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 3 gate kernels. Between them these exercise every mechanism the
 * ported orchestration depends on: grid-stride loops, pointer arithmetic on
 * device pointers (so the registry's offset lookup is really tested),
 * dynamic threadgroup memory, atomics, a templated kernel reached through
 * [[host_name]], a __constant__ symbol, and a struct passed by value.
 *
 * The binding convention is the one that keeps each ported launch site
 * mechanical: CUDA parameter i becomes [[buffer(i)]], in the same order.
 */
#include <metal_stdlib>
using namespace metal;

/* Grid-stride, exactly as bench.h's bench_grid_thread_x/stride_x idiom. */
kernel void k_scale(device uint       *o [[buffer(0)]],
                    device const uint *a [[buffer(1)]],
                    constant uint     &n [[buffer(2)]],
                    constant uint     &k [[buffer(3)]],
                    uint tid    [[thread_position_in_grid]],
                    uint stride [[threads_per_grid]])
{
    for (uint i = tid; i < n; i += stride) o[i] = a[i] * k + 1u;
}

/* Dynamic threadgroup memory plus a device atomic: the k_apply shape. */
kernel void k_sum(device atomic_uint *total [[buffer(0)]],
                  device const uint  *a     [[buffer(1)]],
                  constant uint      &n     [[buffer(2)]],
                  threadgroup uint   *sh    [[threadgroup(0)]],
                  uint tid    [[thread_position_in_grid]],
                  uint lid    [[thread_position_in_threadgroup]],
                  uint tgsz   [[threads_per_threadgroup]],
                  uint stride [[threads_per_grid]])
{
    uint acc = 0;
    for (uint i = tid; i < n; i += stride) acc += a[i];
    sh[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz >> 1; s; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) atomic_fetch_add_explicit(total, sh[0], memory_order_relaxed);
}

/* A templated kernel, instantiated explicitly under the port's naming rule:
 * base name, then each template argument, joined by '_', bools as 0/1. */
template <int K, bool NEG>
kernel void k_tmpl(device int   *o [[buffer(0)]],
                   constant uint &n [[buffer(1)]],
                   uint tid    [[thread_position_in_grid]],
                   uint stride [[threads_per_grid]])
{
    for (uint i = tid; i < n; i += stride) o[i] = NEG ? -(int)(i * K) : (int)(i * K);
}
template [[host_name("k_tmpl_3_1")]] kernel void
k_tmpl<3, true>(device int *, constant uint &, uint, uint);
template [[host_name("k_tmpl_5_0")]] kernel void
k_tmpl<5, false>(device int *, constant uint &, uint, uint);

/* A __constant__ global: on Metal an ordinary constant-address-space buffer,
 * fed by mtlMemcpyToSymbol / mtlGetSymbol. */
kernel void k_const(device uint         *o [[buffer(0)]],
                    constant const uint *c [[buffer(1)]],
                    constant uint       &n [[buffer(2)]],
                    uint tid    [[thread_position_in_grid]],
                    uint stride [[threads_per_grid]])
{
    for (uint i = tid; i < n; i += stride) o[i] = c[i & 7u];
}

/* A struct by value, the norm_t shape. */
struct rt_params { float a; int b; uint c; };
kernel void k_struct(device float       *o [[buffer(0)]],
                     constant rt_params &p [[buffer(1)]],
                     constant uint      &n [[buffer(2)]],
                     uint tid    [[thread_position_in_grid]],
                     uint stride [[threads_per_grid]])
{
    for (uint i = tid; i < n; i += stride) o[i] = p.a * (float)i + (float)p.b + (float)p.c;
}
