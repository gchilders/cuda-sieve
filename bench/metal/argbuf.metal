#include <metal_stdlib>
using namespace metal;
/* The cofq_t shape: a struct of device pointers plus scalars, passed as one
 * buffer, with the pointers reached bindlessly through GPU addresses. */
struct Q_t {
    device uint  *a;
    device uint  *b;
    device uchar *st;
    device long  *ab;
    uint cap, n;
};
kernel void k_argbuf(constant Q_t &Q [[buffer(0)]], device uint *out [[buffer(1)]],
                     uint t [[thread_position_in_grid]])
{
    if (t >= Q.n) return;
    out[t] = Q.a[t] + Q.b[t] + (uint)Q.st[t] + (uint)Q.ab[t] + Q.cap;
}
