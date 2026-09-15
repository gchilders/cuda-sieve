/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 3 gate. Note the file extension: this is PLAIN C++, not
 * Objective-C++. That is the point being demonstrated as much as anything
 * else -- ported orchestration from pipeline.cuh will compile exactly like
 * this, with metal_rt.mm the only Objective-C++ in the build.
 */
#include "metal_rt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int failures = 0;
static void ok(const char *what, bool cond, const char *detail = "")
{
    printf("  %-46s %s%s%s\n", what, cond ? "PASS" : "FAIL",
           *detail ? "  " : "", detail);
    if (!cond) failures++;
}

#define CK(call) do { mtlError_t _e = (call); if (_e != mtlSuccess) { \
    printf("  %s -> %s\n", #call, mtlGetErrorString(_e)); failures++; } } while (0)

int main(int argc, char **argv)
{
    const char *lib = argc > 1 ? argv[1] : "rt_test.metallib";
    if (mtlInit(lib) != mtlSuccess) { printf("mtlInit failed\n"); return 1; }

    mtlDeviceProp prop;
    CK(mtlGetDeviceProperties(&prop, 0));
    printf("device: %s, %d cores, %.2f GB, %zu B threadgroup memory, warp %d\n\n",
           prop.name, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / 1073741824.0,
           prop.sharedMemPerBlock, prop.warpSize);

    const uint32_t N = 1u << 20;
    uint32_t *d_a = nullptr, *d_o = nullptr, *d_total = nullptr;
    CK(mtlMalloc((void **)&d_a, (size_t)N * 4));
    CK(mtlMalloc((void **)&d_o, (size_t)N * 4));
    CK(mtlMalloc((void **)&d_total, 4));

    std::vector<uint32_t> h(N);
    for (uint32_t i = 0; i < N; i++) h[i] = i * 2654435761u;
    CK(mtlMemcpy(d_a, h.data(), (size_t)N * 4, mtlMemcpyHostToDevice));

    printf("== memory and binding ==\n");
    {
        CK(mtlMemset(d_o, 0, (size_t)N * 4));
        bool z = true; for (uint32_t i = 0; i < N; i++) if (d_o[i]) { z = false; break; }
        ok("mtlMemset via blit clears the buffer", z);
    }
    {
        uint32_t k = 3;
        CK(MTL_LAUNCH(k_scale, 256, 256, 0, 0, d_o, (const uint32_t *)d_a, N, k));
        CK(mtlDeviceSynchronize());
        bool good = true;
        for (uint32_t i = 0; i < N && good; i++) if (d_o[i] != h[i] * 3u + 1u) good = false;
        ok("grid-stride kernel over a whole buffer", good);
    }
    {
        /* THE registry test: hand the kernel an interior pointer, which is
         * what the CUDA code does constantly (bk->bucket + i * stride). */
        const uint32_t off = 12345, n2 = 4096;
        CK(mtlMemset(d_o, 0, (size_t)N * 4));
        uint32_t k = 7;
        CK(MTL_LAUNCH(k_scale, 64, 128, 0, 0, d_o + off,
                      (const uint32_t *)(d_a + off), n2, k));
        CK(mtlDeviceSynchronize());
        bool good = (d_o[off - 1] == 0);
        for (uint32_t i = 0; i < n2 && good; i++)
            if (d_o[off + i] != h[off + i] * 7u + 1u) good = false;
        if (good && d_o[off + n2] != 0) good = false;
        ok("offset device pointers bind at the right offset", good);
    }

    printf("\n== threadgroup memory and atomics ==\n");
    {
        const uint32_t n = 1u << 16, tg = 256;
        CK(mtlMemset(d_total, 0, 4));
        CK(MTL_LAUNCH(k_sum, 64, tg, tg * 4, 0, d_total, (const uint32_t *)d_a, n));
        CK(mtlDeviceSynchronize());
        uint32_t want = 0; for (uint32_t i = 0; i < n; i++) want += h[i];
        char d[96]; snprintf(d, sizeof d, "got %u want %u", *d_total, want);
        ok("dynamic threadgroup reduction + device atomic", *d_total == want, d);
    }
    {
        size_t over = prop.sharedMemPerBlock + 1024;
        mtlError_t e = mtlFuncSetMaxThreadgroupMemory("k_sum", over);
        ok("oversized threadgroup request is refused", e != mtlSuccess);
        ok("in-range threadgroup request is accepted",
           mtlFuncSetMaxThreadgroupMemory("k_sum", prop.sharedMemPerBlock) == mtlSuccess);
        /* Clear the sticky error the deliberate failure above left behind,
         * the way CUDA code calls cudaGetLastError() for the same reason. */
        ok("a later launch is not blamed for an earlier error",
           mtlGetLastError() == mtlErrorLaunchOutOfResources);
    }

    printf("\n== templated kernels and symbols ==\n");
    {
        int32_t *d_i = nullptr; const uint32_t n = 4096;
        CK(mtlMalloc((void **)&d_i, (size_t)n * 4));
        CK(MTL_LAUNCH(k_tmpl_3_1, 16, 128, 0, 0, d_i, n));
        CK(mtlDeviceSynchronize());
        bool good = true;
        for (uint32_t i = 0; i < n && good; i++) if (d_i[i] != -(int32_t)(i * 3)) good = false;
        ok("templated kernel k_tmpl<3,true> via host_name", good);
        CK(MTL_LAUNCH(k_tmpl_5_0, 16, 128, 0, 0, d_i, n));
        CK(mtlDeviceSynchronize());
        good = true;
        for (uint32_t i = 0; i < n && good; i++) if (d_i[i] != (int32_t)(i * 5)) good = false;
        ok("templated kernel k_tmpl<5,false> via host_name", good);
        CK(mtlFree(d_i));
        ok("a kernel name absent from the library is an error",
           mtl_launch("k_does_not_exist", 0, 1, 1, 0) == mtlErrorKernelNotFound);
        (void)mtlGetLastError();
    }
    {
        uint32_t c[8]; for (int i = 0; i < 8; i++) c[i] = 1000u + i;
        CK(mtlMemcpyToSymbol("c_tab", c, sizeof c, 0));
        const uint32_t n = 1024;
        CK(mtlMemset(d_o, 0, (size_t)n * 4));
        CK(MTL_LAUNCH(k_const, 8, 128, 0, 0, d_o, (const uint32_t *)mtlGetSymbol("c_tab"), n));
        CK(mtlDeviceSynchronize());
        bool good = true;
        for (uint32_t i = 0; i < n && good; i++) if (d_o[i] != 1000u + (i & 7u)) good = false;
        ok("__constant__ symbol via mtlMemcpyToSymbol", good);
    }
    {
        struct rt_params { float a; int b; uint32_t c; } p { 1.5f, -3, 10u };
        float *d_f = nullptr; const uint32_t n = 1024;
        CK(mtlMalloc((void **)&d_f, (size_t)n * 4));
        CK(MTL_LAUNCH(k_struct, 8, 128, 0, 0, d_f, p, n));
        CK(mtlDeviceSynchronize());
        bool good = true;
        for (uint32_t i = 0; i < n && good; i++)
            if (d_f[i] != 1.5f * (float)i + (float)(-3) + 10.0f) good = false;
        ok("struct passed by value (the norm_t shape)", good);
        CK(mtlFree(d_f));
    }

    printf("\n== streams and events ==\n");
    {
        mtlStream_t s0 = nullptr, s1 = nullptr;
        CK(mtlStreamCreate(&s0));
        CK(mtlStreamCreate(&s1));
        mtlEvent_t e0, e1, e2;
        CK(mtlEventCreate(&e0)); CK(mtlEventCreate(&e1)); CK(mtlEventCreate(&e2));

        /* Cross-stream ordering: s1 must not read d_o before s0 has written it. */
        CK(mtlMemsetAsync(d_o, 0, (size_t)N * 4, s0));
        uint32_t k = 11;
        CK(MTL_LAUNCH(k_scale, 512, 256, 0, s0, d_o, (const uint32_t *)d_a, N, k));
        CK(mtlEventRecord(e0, s0));
        CK(mtlStreamWaitEvent(s1, e0, 0));
        CK(mtlMemset(d_total, 0, 4));
        CK(MTL_LAUNCH(k_sum, 64, 256, 256 * 4, s1, d_total, (const uint32_t *)d_o, N));
        CK(mtlEventRecord(e1, s1));
        CK(mtlStreamSynchronize(s1));
        uint32_t want = 0; for (uint32_t i = 0; i < N; i++) want += h[i] * 11u + 1u;
        char d[96]; snprintf(d, sizeof d, "got %u want %u", *d_total, want);
        ok("cross-stream sequence produces correct results", *d_total == want, d);

        /* NEGATIVE CONTROL, and it reports something worth knowing: on this
         * device the same sequence with NO wait gives the same answer, every
         * time. So the check above does not actually isolate
         * mtlStreamWaitEvent -- Metal is already ordering the two queues by
         * itself, almost certainly through the automatic hazard tracking that
         * a default (tracked) MTLBuffer gets when two command buffers touch
         * it. That is convenient but it is NOT a guarantee to lean on across
         * queues, so the encodeWaitForEvent path stays.
         *
         * What the pair of checks therefore establishes is: the wait does not
         * deadlock, does not corrupt, and does not reorder. What it does NOT
         * establish is that the wait is load-bearing. Re-test when Phase 5
         * has two sides running genuinely concurrent work on untracked or
         * disjoint buffers. Not counted as a failure either way. */
        {
            CK(mtlMemsetAsync(d_o, 0, (size_t)N * 4, s0));
            uint32_t k2 = 11;
            CK(MTL_LAUNCH(k_scale, 512, 256, 0, s0, d_o, (const uint32_t *)d_a, N, k2));
            CK(mtlMemset(d_total, 0, 4));
            CK(MTL_LAUNCH(k_sum, 64, 256, 256 * 4, s1, d_total, (const uint32_t *)d_o, N));
            CK(mtlStreamSynchronize(s1));
            CK(mtlStreamSynchronize(s0));
            printf("  %-46s %s (got %u, ordered would be %u)\n",
                   "[control] same sequence with no wait",
                   *d_total == want ? "happened to serialise" : "raced, as expected",
                   *d_total, want);
        }

        CK(mtlEventRecord(e2, s1));
        float ms = -1.0f;
        CK(mtlEventElapsedTime(&ms, e1, e2));
        char t[96]; snprintf(t, sizeof t, "%.4f ms", ms);
        ok("mtlEventElapsedTime returns a sane interval", ms >= 0.0f && ms < 5000.0f, t);
        ok("mtlEventQuery reports a completed event", mtlEventQuery(e1) == mtlSuccess);

        CK(mtlEventDestroy(e0)); CK(mtlEventDestroy(e1)); CK(mtlEventDestroy(e2));
        CK(mtlStreamDestroy(s0)); CK(mtlStreamDestroy(s1));
    }

    printf("\n== error reporting ==\n");
    {
        size_t freeB = 0, totalB = 0;
        CK(mtlMemGetInfo(&freeB, &totalB));
        char d[96]; snprintf(d, sizeof d, "%.2f of %.2f GB free",
                             freeB / 1073741824.0, totalB / 1073741824.0);
        ok("mtlMemGetInfo reports a plausible budget", totalB > 0 && freeB <= totalB, d);
        uint32_t stack_local = 0;
        (void)mtl_launch("k_scale", 0, 1, 1, 0, &stack_local, (const uint32_t *)d_a,
                         1u, 1u);
        ok("binding an unregistered pointer is caught",
           mtlGetLastError() == mtlErrorNotMapped);
    }

    CK(mtlFree(d_a)); CK(mtlFree(d_o)); CK(mtlFree(d_total));
    mtlShutdown();

    printf("\n%s\n", failures ? "PHASE 3 GATE: FAIL" : "PHASE 3 GATE: PASS");
    return failures ? 1 : 0;
}
