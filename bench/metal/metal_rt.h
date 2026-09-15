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
#include <cstddef>
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
    /* Metal publishes no cache size. Reported as 0 so the startup line that
     * prints it stays honest rather than inventing a number; CUDA's value is
     * used for reporting only, never for a sizing decision. */
    size_t l2CacheSize;
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

/* ---- bindless: passing a struct full of device pointers ----------------- *
 *
 * CUDA passes cofq_t (cofac.cuh:1410, ~27 device pointers) to k_cof_enqueue
 * BY VALUE. A host pointer means nothing to a shader, so on Metal the struct
 * must carry GPU addresses and the kernel must declare its members as
 * `device T*`. Two things are needed for that, and BOTH are required:
 *
 *   mtlDeviceAddress() turns a registered allocation pointer (interior
 *   pointers included) into the GPU address the shader will dereference.
 *
 *   mtlUseResource() makes that allocation resident for the current encoder.
 *   Metal only guarantees residency for resources it can see bound; one
 *   reached through a raw address is invisible to it, and skipping this is
 *   the classic way to get a page fault or silent garbage rather than a
 *   clean error. Call it for every pointer inside the struct, after
 *   mtl_launch_begin and before mtl_launch_end.
 */
uint64_t    mtlDeviceAddress(const void *p);
void        mtlUseResource(const void *p);

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
/* A struct-of-device-pointers argument, i.e. what CUDA passes by value. The
 * buffer holds GPU addresses; every pointer reached through it must also be
 * made resident, which is what `refs` is for. Passing one of these through
 * mtl_launch does both, so a call site cannot bind the struct and forget the
 * residency -- the failure mode there is garbage, not an error. */
struct mtl_argbuf_t {
    const void *buf;
    const void *const *refs;
    int nrefs;
};

inline void mtl_bind_one(mtl_argbuf_t a, int i)
{
    mtl_bind_buffer(a.buf, i);
    for (int k = 0; k < a.nrefs; k++) mtlUseResource(a.refs[k]);
}

/* A null kernel argument.
 *
 * THIS OVERLOAD IS LOAD-BEARING, and the bug it prevents is vicious. CUDA
 * code passes NULL for optional buffers (k_apply's `dump`, `dbg_cells`,
 * `probe_out`), and in C++ NULL is 0L -- an INTEGER, not a pointer. Without
 * this, the template below takes its non-pointer branch and binds eight bytes
 * of zeros as a small constant buffer, so the kernel's `if (dump)` sees a
 * perfectly good non-nil address and dereferences it. The result is a GPU
 * page fault with no clue as to which argument caused it.
 *
 * The generators rewrite a bare NULL argument to nullptr so that it lands
 * here; this overload is what makes that rewrite mean something. */
inline void mtl_bind_one(std::nullptr_t, int i) { mtl_bind_buffer(nullptr, i); }

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

/* Same, but the kernel name is an expression rather than a token. Needed
 * where a launch sits inside a function templated on the very parameter that
 * selects the instantiation -- cf_run_rounds<L> launching k_cofac<L,M,S> --
 * so the name cannot be formed textually at all. */
#define MTL_LAUNCH_NAMED(kname, grid, block, smem, stream, ...) \
    mtl_launch((kname), (stream), (unsigned)(grid), (unsigned)(block), (size_t)(smem), __VA_ARGS__)

/* k_foo<<<g, b, smem, stream>>>(a, b, c)  ->  MTL_LAUNCH(k_foo, g, b, smem, stream, a, b, c) */
#define MTL_LAUNCH(kern, grid, block, smem, stream, ...) \
    mtl_launch(#kern, (stream), (unsigned)(grid), (unsigned)(block), (size_t)(smem), __VA_ARGS__)

/* CUDA ships a templated cudaMalloc overload so callers can write
 * cudaMalloc(&typed_ptr, n) without a cast, and the ported code relies on it.
 * Mirror it rather than editing every call site. */
template <class T>
inline mtlError_t mtlMalloc(T **p, size_t n) { return mtlMalloc((void **)p, n); }
template <class T>
inline mtlError_t mtlHostAlloc(T **p, size_t n, unsigned f)
{ return mtlHostAlloc((void **)p, n, f); }

/* CUDA's async calls default their stream argument to the legacy default
 * stream, and the ported code relies on that. extern "C" cannot carry default
 * arguments, so mirror them as C++ overloads. */
inline mtlError_t mtlMemsetAsync(void *p, int v, size_t n)
{ return mtlMemsetAsync(p, v, n, 0); }
inline mtlError_t mtlMemcpyAsync(void *dst, const void *src, size_t n, int kind)
{ return mtlMemcpyAsync(dst, src, n, kind, 0); }

/* cudaEventRecord(e) defaults to the legacy default stream. */
inline mtlError_t mtlEventRecord(mtlEvent_t e) { return mtlEventRecordOn(e, 0); }
inline mtlError_t mtlEventRecord(mtlEvent_t e, mtlStream_t s) { return mtlEventRecordOn(e, s); }

#endif  /* __cplusplus */

/* Threadgroup bytes k_apply needs for one bucket region.
 *
 * CUDA's figure carries a second term, `nslice_pow2 * sizeof(uint16_t)`, for a
 * copy of the slice-log table in shared memory. This build leaves that table
 * in device memory (see metal/gen_bench_kernels.py for why), so the term is
 * gone -- and with it the 128 bytes that put log_region 14 at 32,896 B against
 * Apple's hard 32,768 B ceiling. At 14 the requirement is now exactly 32,768,
 * and every check against the ceiling is `>`, so it fits.
 *
 * It lives here because three places need it -- the pipeline, the bench_kernels
 * harness and the Phase 5 gate -- and a threadgroup length that disagrees with
 * what the kernel indexes is not a compile error, it is a wrong answer. */
static inline size_t mtl_apply_smem(uint32_t ncell, int cellbits)
{
    return (size_t)ncell * (size_t)cellbits / 8;
}

#endif  /* CUDA_SIEVE_METAL_RT_H */
