/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 2 gate, part 2 (driver): run sf_test.metal on the GPU and compare
 * every result, as a bit pattern, against the host build of the SAME headers.
 *
 * Part 1 (sf_test_host.cpp) established that the host build matches hardware
 * fp64. This establishes that the device build matches the host build. Chained,
 * they say the GPU's soft-fp64 is bit-exact fp64 -- which is the property the
 * two device-side fp64 sites in the siever actually need.
 */
#import <Metal/Metal.h>
#include "softfp64.h"
#include "portable_log2.h"
#include <cstdio>
#include <cstring>
#include <vector>

static sf_u64 rs = 0x9E3779B97F4A7C15ull;
static sf_u64 rnd(void)
{ rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27; return rs * 0x2545F4914F6CDD1Dull; }

static id<MTLBuffer> buf(id<MTLDevice> d, size_t n)
{ return [d newBufferWithLength:n options:MTLResourceStorageModeShared]; }

static long nfail = 0;
static void cmp64(const char *op, int i, sf_u64 host, sf_u64 dev, sf_u64 a, sf_u64 b, sf_u64 c)
{
    if (host == dev) return;
    if (nfail++ < 10)
        printf("  MISMATCH %-6s [%d] a=%016llx b=%016llx c=%016llx host=%016llx dev=%016llx\n",
               op, i, (unsigned long long)a, (unsigned long long)b, (unsigned long long)c,
               (unsigned long long)host, (unsigned long long)dev);
}

int main(void)
{
@autoreleasepool {
    const int N = 1 << 18;
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithURL:[NSURL fileURLWithPath:@"sf_test.metallib"] error:&e];
    if (!lib) { printf("lib load failed: %s\n", e.description.UTF8String); return 1; }
    id<MTLComputePipelineState> p =
        [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_sf"] error:&e];
    if (!p) { printf("pipeline failed: %s\n", e.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [d newCommandQueue];

    id<MTLBuffer> ba = buf(d, N*8), bb = buf(d, N*8), bc = buf(d, N*8);
    id<MTLBuffer> oadd = buf(d, N*8), osub = buf(d, N*8), omul = buf(d, N*8),
                  odiv = buf(d, N*8), ofma = buf(d, N*8), oi64 = buf(d, N*8);
    id<MTLBuffer> of32 = buf(d, N*4), olg2 = buf(d, N*4);

    sf_u64 *A = (sf_u64*)ba.contents, *B = (sf_u64*)bb.contents, *C = (sf_u64*)bc.contents;
    for (int i = 0; i < N; i++) {
        /* A mix: fully random patterns (NaN/inf/subnormal included), bounded
         * normals, subnormals, and integer-valued doubles. */
        int k = i & 3;
        if (k == 0) { A[i] = rnd(); B[i] = rnd(); }
        else if (k == 1) {
            sf_u64 ea = 1 + (rnd() % 2046), eb = 1 + (rnd() % 2046);
            A[i] = ((rnd()&1)<<63) | (ea<<52) | (rnd() & SF_MANT52);
            B[i] = ((rnd()&1)<<63) | (eb<<52) | (rnd() & SF_MANT52);
        } else if (k == 2) {
            A[i] = ((rnd()&1)<<63) | (rnd() & SF_MANT52);   /* subnormal */
            B[i] = ((rnd()&1)<<63) | (rnd() & SF_MANT52);
        } else {
            sf_u64 ea = 1000 + (rnd() % 60);                /* near-equal */
            A[i] = (ea<<52) | (rnd() & SF_MANT52);
            B[i] = (ea<<52) | (rnd() & SF_MANT52);
        }
        C[i] = rnd();
    }

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:p];
    id<MTLBuffer> bufs[] = {oadd,osub,omul,odiv,ofma,of32,oi64,olg2,ba,bb,bc};
    for (int i = 0; i < 11; i++) [en setBuffer:bufs[i] offset:0 atIndex:i];
    [en dispatchThreadgroups:MTLSizeMake(N/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { printf("command buffer error: %s\n", cb.error.description.UTF8String); return 1; }

    sf_u64 *Gadd=(sf_u64*)oadd.contents, *Gsub=(sf_u64*)osub.contents,
           *Gmul=(sf_u64*)omul.contents, *Gdiv=(sf_u64*)odiv.contents,
           *Gfma=(sf_u64*)ofma.contents, *Gi64=(sf_u64*)oi64.contents;
    sf_u32 *Gf32=(sf_u32*)of32.contents, *Glg2=(sf_u32*)olg2.contents;

    for (int i = 0; i < N; i++) {
        sf_u64 a = A[i], b = B[i], c = C[i];
        cmp64("add", i, sf_add(a,b), Gadd[i], a,b,c);
        cmp64("sub", i, sf_sub(a,b), Gsub[i], a,b,c);
        cmp64("mul", i, sf_mul(a,b), Gmul[i], a,b,c);
        cmp64("div", i, sf_div(a,b), Gdiv[i], a,b,c);
        cmp64("fma", i, sf_fma(a,b,c), Gfma[i], a,b,c);
        cmp64("i64", i, sf_from_i64((sf_i64)c), Gi64[i], a,b,c);
        cmp64("tof32", i, (sf_u64)sf_f2b(sf_to_f32(a)), (sf_u64)Gf32[i], a,b,c);
        sf_u32 fb = (sf_u32)(c & 0x7fffffffu);
        if ((fb >> 23) == 0u || (fb >> 23) == 0xffu) fb = 0x3f800000u;
        cmp64("log2", i, (sf_u64)sf_f2b(pl_log2f(sf_b2f(fb))), (sf_u64)Glg2[i], a,b,c);
    }

    printf("compared %d cases x 8 operations = %d results\n", N, N*8);
    if (nfail) { printf("SOFTFP64 DEVICE GATE: FAIL (%ld mismatches)\n", nfail); return 1; }
    printf("SOFTFP64 DEVICE GATE: PASS -- device build is bit-identical to host build\n");
}
return 0; }
