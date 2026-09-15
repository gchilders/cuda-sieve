#include <metal_stdlib>
using namespace metal;
kernel void k_fp32(device float *qd [[buffer(0)]], device float *qf [[buffer(1)]],
                   device const float *a [[buffer(2)]], device const float *b [[buffer(3)]],
                   device const float *c [[buffer(4)]], uint t [[thread_position_in_grid]])
{
    qd[t] = a[t] / b[t];                  /* division                */
    qf[t] = fma(a[t], b[t], c[t]);        /* fused multiply-add      */
}
