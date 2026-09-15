#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

static id<MTLComputePipelineState> mk(id<MTLDevice> d, id<MTLLibrary> l, const char *n) {
    NSError *e = nil;
    id<MTLFunction> f = [l newFunctionWithName:[NSString stringWithUTF8String:n]];
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) { printf("pipeline %s failed: %s\n", n, e.description.UTF8String); exit(1); }
    return p;
}

int main() {
@autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    printf("=== DEVICE ===\n");
    printf("name                         : %s\n", d.name.UTF8String);
    printf("hasUnifiedMemory             : %d\n", (int)d.hasUnifiedMemory);
    printf("maxThreadgroupMemoryLength   : %lu B\n", (unsigned long)d.maxThreadgroupMemoryLength);
    printf("maxBufferLength              : %.2f GB\n", d.maxBufferLength/1073741824.0);
    printf("recommendedMaxWorkingSetSize : %.2f GB\n", d.recommendedMaxWorkingSetSize/1073741824.0);
    printf("argumentBuffersSupport       : %ld (2 == tier2)\n", (long)d.argumentBuffersSupport);
    printf("maxThreadsPerThreadgroup     : %lu\n", (unsigned long)d.maxThreadsPerThreadgroup.width);
    for (int f = 1009; f >= 1001; f--)
        if ([d supportsFamily:(MTLGPUFamily)f]) { printf("highest MTLGPUFamilyApple   : %d\n", f-1000); break; }

    NSError *err = nil;
    id<MTLLibrary> lib = [d newLibraryWithURL:[NSURL fileURLWithPath:@"probe.metallib"] error:&err];
    if (!lib) { printf("lib load failed: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [d newCommandQueue];

    // ---- warp semantics ----
    {
        id<MTLComputePipelineState> p = mk(d, lib, "k_warp");
        printf("threadExecutionWidth         : %lu\n", (unsigned long)p.threadExecutionWidth);
        printf("maxTotalThreadsPerThreadgroup: %lu\n", (unsigned long)p.maxTotalThreadsPerThreadgroup);
        id<MTLBuffer> o = [d newBufferWithLength:67*4 options:MTLResourceStorageModeShared];
        memset(o.contents, 0, 67*4);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        uint32_t *r = (uint32_t*)o.contents;
        printf("\n=== WARP SEMANTICS ===\n");
        printf("threads_per_simdgroup        : %u\n", r[0]);
        printf("simd_ballot(lane odd) lo     : 0x%08x  (expect 0xaaaaaaaa == ascending)\n", r[1]);
        printf("simd_ballot hi word          : 0x%08x  (expect 0)\n", r[2]);
        int su_ok = 1; for (uint32_t L=1; L<32; L++) if (r[3+L] != (L-1)*7u) su_ok = 0;
        printf("simd_shuffle_up matches CUDA : %s\n", su_ok ? "YES" : "NO");
        int ps_ok = 1; for (uint32_t L=0; L<32; L++) if (r[35+L] != L) ps_ok = 0;
        printf("simd_prefix_exclusive_sum    : %s\n", ps_ok ? "lane-exact" : "MISMATCH");
    }

    // ---- mulhi(ulong,ulong) vs host __int128 ----
    {
        id<MTLComputePipelineState> p = mk(d, lib, "k_mulhi");
        const int N = 4096;
        id<MTLBuffer> in = [d newBufferWithLength:N*2*8 options:MTLResourceStorageModeShared];
        id<MTLBuffer> o  = [d newBufferWithLength:N*8   options:MTLResourceStorageModeShared];
        uint64_t *a = (uint64_t*)in.contents;
        srandom(12345);
        for (int i=0;i<2*N;i++) a[i] = ((uint64_t)random()<<40) ^ ((uint64_t)random()<<20) ^ random();
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:p]; [e setBuffer:o offset:0 atIndex:0]; [e setBuffer:in offset:0 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(N/32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        uint64_t *g = (uint64_t*)o.contents; int bad=0;
        for (int i=0;i<N;i++) {
            uint64_t want = (uint64_t)(((unsigned __int128)a[2*i] * (unsigned __int128)a[2*i+1]) >> 64);
            if (g[i] != want) bad++;
        }
        printf("\n=== 64x64 MULHI ===\nmulhi(ulong,ulong) vs __int128: %d / %d mismatches\n", bad, N);
    }

    // ---- log2: THE byte-identity question ----
    {
        id<MTLComputePipelineState> p = mk(d, lib, "k_log2");
        const int N = 1<<20;
        id<MTLBuffer> in=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> od=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> op=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
        float *x=(float*)in.contents;
        srandom(999);
        /* magnitudes the sieve actually sees: |F| spread over a wide exponent range */
        for (int i=0;i<N;i++) {
            uint32_t m = (uint32_t)random() & 0x7fffff;
            int ex = 1 + (random() % 250);
            uint32_t bits = ((uint32_t)ex<<23) | m;
            memcpy(&x[i], &bits, 4);
        }
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:od offset:0 atIndex:0]; [e setBuffer:op offset:0 atIndex:1]; [e setBuffer:in offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(N/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        float *gd=(float*)od.contents, *gp=(float*)op.contents;
        long nd=0, np=0; int maxd=0, maxp=0;
        for (int i=0;i<N;i++) {
            float h = log2f(x[i]);
            int32_t hb, gb;
            memcpy(&hb,&h,4); memcpy(&gb,&gd[i],4); int ud = abs(hb-gb);
            memcpy(&gb,&gp[i],4); int up = abs(hb-gb);
            if (ud) { nd++; if (ud>maxd) maxd=ud; }
            if (up) { np++; if (up>maxp) maxp=up; }
        }
        printf("\n=== log2 vs host libm log2f (%d samples) ===\n", N);
        printf("metal::log2          differs : %ld (%.4f%%), max %d ULP\n", nd, 100.0*nd/N, maxd);
        printf("metal::precise::log2 differs : %ld (%.4f%%), max %d ULP\n", np, 100.0*np/N, maxp);
    }
    printf("\n");
}
return 0; }
