#include <metal_stdlib>
using namespace metal;
/* A deliberately long compute kernel. macOS is known to abort GPU work that
 * runs too long while the GPU also drives the display; cuda-sieve's
 * --cof-chunk exists for exactly that hazard on CUDA. This measures whether,
 * and at what duration, Metal does the same here. */
kernel void k_spin(device uint *o [[buffer(0)]], constant uint &iters [[buffer(1)]],
                   uint t [[thread_position_in_grid]])
{
    uint x = t | 1u;
    for (uint i = 0; i < iters; i++) { x = x * 1664525u + 1013904223u; x ^= x >> 13; }
    if (x == 0xdeadbeefu) o[0] = x;   /* never; keeps the loop alive */
}
