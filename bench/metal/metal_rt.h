/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * A CUDA-runtime-shaped shim over Metal.
 *
 * WHY THIS SHAPE. pipeline.cuh is ~4,000 lines of host orchestration with no
 * kernels in it at all, and cofac.cuh and bench_kernels.cu carry a few
 * thousand more. Rewritten against Metal's object model by hand that is the
 * bulk of this port. Rewritten against an API with CUDA's names and CUDA's
 * semantics it is close to a rename pass -- which is precisely the leverage
 * hipify-perl gave the HIP port, and why three of that port's five files
 * needed zero hand fixes. Plan section 6 estimates this decision at roughly
 * 3x on total effort.
 *
 * So every function here deliberately mirrors a cuda* function's signature
 * and blocking behaviour, including the parts that are redundant on Apple
 * silicon. mtlMemcpy between two unified-memory pointers is a memcpy, but it
 * still synchronises first, because that is what cudaMemcpy on the default
 * stream promises and the ported code is written against that promise.
 *
 * This header is plain C++ and pulls in no Metal headers, so ported
 * orchestration compiles as ordinary C++ and only metal_rt.mm needs
 * Objective-C++.
 *
 * WHAT IS DELIBERATELY NOT HERE. Structs that contain device pointers cannot
 * be passed by value through mtl_launch -- a host pointer is meaningless to a
 * shader. The tree has exactly one such kernel, k_cof_enqueue taking cofq_t
 * by value (cofac.cuh:1410, ~27 pointers), and it needs a Metal argument
 * buffer. That is Phase 6 work; mtl_launch rejects it loudly rather than
 * binding garbage.
 */
#ifndef CUDA_SIEVE_METAL_RT_H
#define CUDA_SIEVE_METAL_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <type_traits>
#endif

typedef int mtlError_t;

enum {
    mtlSuccess = 0,
    mtlErrorInvalidValue,
    mtlErrorMemoryAllocation,
    mtlErrorInitialization,
    mtlErrorLaunchFailure,
    mtlErrorLaunchTimeout,      /* macOS killed the command buffer          */
    mtlErrorLaunchOutOfResources,
    mtlErrorNotReady,
    mtlErrorSymbolNotFound,
    mtlErrorKernelNotFound,
    mtlErrorNotMapped,          /* pointer is not inside any allocation     */
    mtlErrorUnsupported
};

enum {
    mtlMemcpyHostToDevice = 0,
    mtlMemcpyDeviceToHost,
    mtlMemcpyDeviceToDevice,
    mtlMemcpyDefault
};

#define mtlHostAllocDefault 0u

typedef struct mtlStreamOpaque *mtlStream_t;   /* NULL == the default stream */
typedef struct mtlEventOpaque  *mtlEvent_t;

typedef struct {
    char   name[256];
    size_t totalGlobalMem;
    int    multiProcessorCount;   /* GPU core count, via IOKit              */
    int    maxGridSize[3];
    int    pciBusID, pciDeviceID, pciDomainID;   /* always 0; UMA has no bus */
    size_t sharedMemPerBlock;     /* maxThreadgroupMemoryLength: 32 KB on M3 */
    int    maxThreadsPerBlock;
    int    warpSize;              /* threadExecutionWidth: 32               */
    int    unifiedMemory;
} mtlDeviceProp;

#ifdef __cplusplus
extern "C" {
#endif

/* Extra to CUDA: load the compiled shader library. Call once before anything
 * else. A NULL path looks for "bench.metallib" next to the executable. */
mtlError_t  mtlInit(const char *metallib_path);
void        mtlShutdown(void);

mtlError_t  mtlSetDevice(int dev);
mtlError_t  mtlGetDevice(int *dev);
mtlError_t  mtlGetDeviceCount(int *n);
mtlError_t  mtlGetDeviceProperties(mtlDeviceProp *p, int dev);
mtlError_t  mtlMemGetInfo(size_t *freeB, size_t *totalB);

mtlError_t  mtlMalloc(void **p, size_t n);
mtlError_t  mtlFree(void *p);
mtlError_t  mtlHostAlloc(void **p, size_t n, unsigned flags);
mtlError_t  mtlFreeHost(void *p);

mtlError_t  mtlMemcpy(void *dst, const void *src, size_t n, int kind);
mtlError_t  mtlMemcpyAsync(void *dst, const void *src, size_t n, int kind,
                           mtlStream_t s);
mtlError_t  mtlMemset(void *p, int value, size_t n);
mtlError_t  mtlMemsetAsync(void *p, int value, size_t n, mtlStream_t s);

/* __constant__ globals. The MSL side declares an ordinary `constant T *`
 * buffer parameter; the host passes mtlGetSymbol(name) as that argument, so
 * it binds through the same path as any other allocation. */
mtlError_t  mtlMemcpyToSymbol(const char *name, const void *src, size_t n,
                              size_t offset);
void       *mtlGetSymbol(const char *name);

mtlError_t  mtlStreamCreate(mtlStream_t *s);
mtlError_t  mtlStreamDestroy(mtlStream_t s);
mtlError_t  mtlStreamSynchronize(mtlStream_t s);
mtlError_t  mtlStreamWaitEvent(mtlStream_t s, mtlEvent_t e, unsigned flags);
mtlError_t  mtlDeviceSynchronize(void);

mtlError_t  mtlEventCreate(mtlEvent_t *e);
mtlError_t  mtlEventDestroy(mtlEvent_t e);
mtlError_t  mtlEventRecordOn(mtlEvent_t e, mtlStream_t s);
mtlError_t  mtlEventSynchronize(mtlEvent_t e);
mtlError_t  mtlEventQuery(mtlEvent_t e);
mtlError_t  mtlEventElapsedTime(float *ms, mtlEvent_t start, mtlEvent_t end);

mtlError_t  mtlGetLastError(void);
mtlError_t  mtlPeekAtLastError(void);
const char *mtlGetErrorString(mtlError_t e);

/* CUDA's cudaFuncSetAttribute(MaxDynamicSharedMemorySize). On Metal the
 * threadgroup allocation is per-dispatch, so this only validates the request
 * against the device ceiling -- which is the check that matters, since Apple's
 * 32 KB is well under CUDA's opt-in tier. */
mtlError_t  mtlFuncSetMaxThreadgroupMemory(const char *kernel, size_t bytes);

/* ---- launch plumbing (used by the template below, not called directly) --- */
mtlError_t  mtl_launch_begin(const char *kernel, mtlStream_t s,
                             unsigned grid, unsigned block, size_t smem);
void        mtl_bind_buffer(const void *ptr, int index);
void        mtl_bind_bytes(const void *data, size_t sz, int index);
mtlError_t  mtl_launch_end(void);

#ifdef __cplusplus
}   /* extern "C" */

/* One kernel argument. Pointers become buffer bindings; everything else is
 * copied inline. The MSL side must declare parameter i as [[buffer(i)]], in
 * the same order as the CUDA kernel's parameter list -- that one convention
 * is what keeps the port of each launch site mechanical. */
template <class T>
inline void mtl_bind_one(T a, int i)
{
    if constexpr (std::is_pointer<T>::value) {
        mtl_bind_buffer((const void *)a, i);
    } else {
        static_assert(std::is_trivially_copyable<T>::value,
                      "kernel arguments must be trivially copyable");
        static_assert(sizeof(T) <= 4096,
                      "kernel argument too large for setBytes; use a buffer");
        mtl_bind_bytes(&a, sizeof(T), i);
    }
}

/* Templated kernels reach MSL through an explicit [[host_name]] instantiation.
 * The convention is base name, then each template argument, joined by '_',
 * with bools as 0/1: k_td<1,0,0,false> is "k_td_1_0_0_0". */
template <class... A>
inline mtlError_t mtl_launch(const char *kernel, mtlStream_t s, unsigned grid,
                             unsigned block, size_t smem, A... args)
{
    mtlError_t e = mtl_launch_begin(kernel, s, grid, block, smem);
    if (e != mtlSuccess) return e;
    int i = 0;
    (void)i;
    /* Unary left fold over the comma operator: guaranteed left-to-right, so
     * argument k lands at buffer index k. */
    (..., mtl_bind_one(args, i++));
    return mtl_launch_end();
}

/* k_foo<<<g, b, smem, stream>>>(a, b, c)  ->  MTL_LAUNCH(k_foo, g, b, smem, stream, a, b, c) */
#define MTL_LAUNCH(kern, grid, block, smem, stream, ...) \
    mtl_launch(#kern, (stream), (unsigned)(grid), (unsigned)(block), (size_t)(smem), __VA_ARGS__)

/* cudaEventRecord(e) defaults to the legacy default stream. */
inline mtlError_t mtlEventRecord(mtlEvent_t e) { return mtlEventRecordOn(e, 0); }
inline mtlError_t mtlEventRecord(mtlEvent_t e, mtlStream_t s) { return mtlEventRecordOn(e, s); }

#endif  /* __cplusplus */
#endif  /* CUDA_SIEVE_METAL_RT_H */
