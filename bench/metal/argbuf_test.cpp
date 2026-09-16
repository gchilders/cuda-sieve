/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 6 groundwork gate: passing a struct full of device pointers.
 *
 * CUDA hands cofq_t to k_cof_enqueue BY VALUE (cofac.cuh:1410) -- roughly 27
 * device pointers in one struct. It is the only kernel in the tree that does
 * this, and it is the one thing in this port with no CUDA-shaped equivalent:
 * a host pointer means nothing to a shader. On Metal the struct carries GPU
 * addresses and the kernel declares its members as `device T*`.
 *
 * Two mechanisms have to hold, and this exercises both:
 *
 *   mtlDeviceAddress() -- a registered allocation pointer (interior pointers
 *   included) to the address the shader dereferences.
 *
 *   mtlUseResource() -- residency. Metal only guarantees a resource is mapped
 *   if it can see it bound; one reached through a raw address is invisible to
 *   it. This is NOT defensive: the negative control below omits the residency
 *   calls and gets 4095 of 4096 values wrong, every run. Forgetting it in the
 *   real port would produce garbage, not a clean error.
 */
#include "metal_rt.h"
#include <cstdio>
#include <cstring>
/* Host mirror of the MSL struct: device pointers become 64-bit GPU addresses.
 * Layout must match exactly -- MSL aligns a device pointer to 8. */
struct Q_host { uint64_t a, b, st, ab; uint32_t cap, n; uint32_t _pad[2]; };

int main(int argc, char **argv)
{
    if (mtlInit(argc > 1 ? argv[1] : "argbuf.metallib") != mtlSuccess) return 1;
    const uint32_t N = 4096;
    uint32_t *a, *b, *out; uint8_t *st; int64_t *ab; Q_host *q;
    mtlMalloc((void**)&a, N*4); mtlMalloc((void**)&b, N*4);
    mtlMalloc((void**)&st, N);  mtlMalloc((void**)&ab, N*8);
    mtlMalloc((void**)&out, N*4); mtlMalloc((void**)&q, sizeof(Q_host));
    for (uint32_t i = 0; i < N; i++) { a[i]=i; b[i]=2*i; st[i]=(uint8_t)(i&7); ab[i]=(int64_t)(i%5); }
    memset(out, 0, N*4);

    q->a  = mtlDeviceAddress(a);
    q->b  = mtlDeviceAddress(b);
    q->st = mtlDeviceAddress(st);
    q->ab = mtlDeviceAddress(ab);
    q->cap = 1000; q->n = N;
    printf("gpu addresses: a=%#llx b=%#llx st=%#llx ab=%#llx\n",
           (unsigned long long)q->a, (unsigned long long)q->b,
           (unsigned long long)q->st, (unsigned long long)q->ab);

    /* NEGATIVE CONTROL FIRST, and the ordering is the point.
     *
     * Residency, once granted for a resource, persists within the process --
     * so a control run AFTER the positive case passes and proves nothing.
     * (Observed exactly that: 0 of 4096 wrong when it followed, 4095 of 4096
     * when it ran in a fresh process.) Run it before anything makes these
     * buffers resident. */
    memset(out, 0, N*4);
    mtlError_t e = mtl_launch_begin("k_argbuf", 0, N/64, 64, 0, 0u);
    if (e != mtlSuccess) { printf("launch_begin: %s\n", mtlGetErrorString(e)); return 1; }
    mtl_bind_buffer(q, 0);
    mtl_bind_buffer(out, 1);
    /* deliberately no mtlUseResource */
    mtl_launch_end();
    mtlDeviceSynchronize();
    uint32_t bad2 = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t want = i + 2*i + (i&7) + (uint32_t)(i%5) + 1000;
        if (out[i] != want) bad2++;
    }
    printf("  control (no residency, run first): %u of %u wrong -- %s\n", bad2, N,
           bad2 ? "residency is load-bearing, as expected"
                : "WARNING: control passed, so the check below proves nothing");

    /* Now the real thing. */
    memset(out, 0, N*4);
    e = mtl_launch_begin("k_argbuf", 0, N/64, 64, 0, 0u);
    if (e != mtlSuccess) { printf("launch_begin: %s\n", mtlGetErrorString(e)); return 1; }
    mtl_bind_buffer(q, 0);
    mtl_bind_buffer(out, 1);
    mtlUseResource(a); mtlUseResource(b); mtlUseResource(st); mtlUseResource(ab);
    e = mtl_launch_end();
    if (e != mtlSuccess) { printf("launch_end: %s\n", mtlGetErrorString(e)); return 1; }
    if (mtlDeviceSynchronize() != mtlSuccess) { printf("sync failed\n"); return 1; }

    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t want = i + 2*i + (i&7) + (uint32_t)(i%5) + 1000;
        if (out[i] != want) { if (!bad) printf("  first bad at %u: got %u want %u\n", i, out[i], want); bad++; }
    }
    printf("  positive: %s (%u of %u wrong)\n",
           bad ? "FAIL" : "PASS", bad, N);

    mtlShutdown();
    printf("\n%s\n", (bad || !bad2) ? "ARGUMENT BUFFER GATE: FAIL"
                                     : "ARGUMENT BUFFER GATE: PASS");
    return (bad || !bad2) ? 1 : 0;
}
