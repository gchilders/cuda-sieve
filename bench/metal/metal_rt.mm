/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * metal_rt.h's implementation. The only Objective-C++ in the port.
 *
 * THE CENTRAL TRICK is the allocation registry. CUDA code does pointer
 * arithmetic on device pointers freely -- `bk->bucket + i * stride` is passed
 * to a kernel as if it were a base pointer -- but Metal binds an MTLBuffer
 * plus an offset, not an address. So every allocation is a shared-storage
 * MTLBuffer, mtlMalloc hands back its `contents` pointer, and a sorted
 * registry maps any pointer back to (buffer, offset) at bind time. Ported
 * code keeps doing arithmetic on real addresses and never learns that Metal
 * is underneath.
 *
 * STREAMS. A stream owns one MTLCommandQueue and at most one open command
 * buffer with at most one open encoder. Dispatches accumulate into that
 * encoder, which is what keeps a band's many small kernels cheap; the encoder
 * is closed and the buffer committed only when something demands ordering --
 * a sync, an event record, or a switch between compute and blit work. Metal
 * serialises dispatches within a compute encoder and tracks buffer hazards
 * automatically, so this reproduces a CUDA stream's in-order semantics.
 *
 * EVENTS. Recording an event closes and commits the stream's command buffer,
 * because a command buffer's GPUEndTime is the only timestamp available
 * without counter sample buffers. That makes every cudaEventRecord a
 * pipeline flush. It is correct, and it is a known performance cost rather
 * than a hidden one -- see the note at mtlEventRecordOn.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>

#include "metal_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

/* ---------------------------------------------------------------- state -- */

namespace {

struct Alloc { uintptr_t base; size_t len; id<MTLBuffer> buf; };

enum EncKind { ENC_NONE = 0, ENC_COMPUTE, ENC_BLIT };

struct Stream {
    id<MTLCommandQueue>          q   = nil;
    id<MTLCommandBuffer>         cb  = nil;
    id<MTLComputeCommandEncoder> cenc = nil;
    id<MTLBlitCommandEncoder>    benc = nil;
    EncKind                      kind = ENC_NONE;
    id<MTLCommandBuffer>         last = nil;   /* last committed, for waiting */
};

id<MTLDevice>        g_dev = nil;
id<MTLLibrary>       g_lib = nil;
Stream               g_default;
std::vector<Alloc>   g_allocs;           /* sorted by base                   */
std::map<std::string, id<MTLComputePipelineState>> g_psos;
std::map<std::string, void *>                      g_symbols;
std::mutex           g_lock;
mtlError_t           g_last_error = mtlSuccess;
int                  g_core_count = 0;

/* the encoder a launch is currently binding into */
id<MTLComputeCommandEncoder> g_bind_enc = nil;
Stream                      *g_bind_stream = nullptr;
unsigned                     g_bind_grid = 0, g_bind_block = 0;
/* The error belonging to the launch in progress, distinct from the sticky
 * g_last_error. CUDA's launch returns its OWN status and only
 * cudaGetLastError() is sticky; conflating the two makes a launch report a
 * failure that happened somewhere else entirely. */
mtlError_t                   g_bind_error = mtlSuccess;

inline mtlError_t fail(mtlError_t e) { g_last_error = e; return e; }

Stream *resolve(mtlStream_t s)
{ return s ? (Stream *)s : &g_default; }

/* ---- encoder lifecycle -------------------------------------------------- */

void close_encoder(Stream *st)
{
    if (st->kind == ENC_COMPUTE && st->cenc) { [st->cenc endEncoding]; st->cenc = nil; }
    if (st->kind == ENC_BLIT    && st->benc) { [st->benc endEncoding]; st->benc = nil; }
    st->kind = ENC_NONE;
}

void ensure_cb(Stream *st)
{ if (!st->cb) st->cb = [st->q commandBuffer]; }

id<MTLComputeCommandEncoder> ensure_compute(Stream *st)
{
    if (st->kind != ENC_COMPUTE) { close_encoder(st); ensure_cb(st);
        st->cenc = [st->cb computeCommandEncoder]; st->kind = ENC_COMPUTE; }
    return st->cenc;
}

id<MTLBlitCommandEncoder> ensure_blit(Stream *st)
{
    if (st->kind != ENC_BLIT) { close_encoder(st); ensure_cb(st);
        st->benc = [st->cb blitCommandEncoder]; st->kind = ENC_BLIT; }
    return st->benc;
}

/* Close and commit whatever is open. Returns the committed buffer, or the
 * previously committed one if there was nothing new. */
id<MTLCommandBuffer> commit(Stream *st)
{
    close_encoder(st);
    if (st->cb) { [st->cb commit]; st->last = st->cb; st->cb = nil; }
    return st->last;
}

mtlError_t cb_status(id<MTLCommandBuffer> cb)
{
    if (!cb || !cb.error) return mtlSuccess;
    if (cb.error.code == MTLCommandBufferErrorTimeout) return mtlErrorLaunchTimeout;
    if (cb.error.code == MTLCommandBufferErrorOutOfMemory) return mtlErrorLaunchOutOfResources;
    return mtlErrorLaunchFailure;
}

mtlError_t sync(Stream *st)
{
    id<MTLCommandBuffer> cb = commit(st);
    if (!cb) return mtlSuccess;
    [cb waitUntilCompleted];
    mtlError_t e = cb_status(cb);
    return e == mtlSuccess ? mtlSuccess : fail(e);
}

/* ---- allocation registry ------------------------------------------------ */

void reg_add(void *p, size_t n, id<MTLBuffer> b)
{
    Alloc a{ (uintptr_t)p, n, b };
    auto it = std::lower_bound(g_allocs.begin(), g_allocs.end(), a.base,
                               [](const Alloc &x, uintptr_t v){ return x.base < v; });
    g_allocs.insert(it, a);
}

/* Largest base <= p, then a containment check. O(log n) per bound pointer. */
const Alloc *reg_find(const void *p)
{
    uintptr_t v = (uintptr_t)p;
    auto it = std::upper_bound(g_allocs.begin(), g_allocs.end(), v,
                               [](uintptr_t x, const Alloc &a){ return x < a.base; });
    if (it == g_allocs.begin()) return nullptr;
    --it;
    if (v >= it->base && v < it->base + it->len) return &*it;
    return nullptr;
}

bool reg_erase(void *p)
{
    uintptr_t v = (uintptr_t)p;
    auto it = std::lower_bound(g_allocs.begin(), g_allocs.end(), v,
                               [](const Alloc &x, uintptr_t q){ return x.base < q; });
    if (it == g_allocs.end() || it->base != v) return false;
    g_allocs.erase(it);
    return true;
}

mtlError_t alloc_shared(void **out, size_t n)
{
    if (!out) return fail(mtlErrorInvalidValue);
    if (n == 0) n = 1;
    id<MTLBuffer> b = [g_dev newBufferWithLength:n
                                         options:MTLResourceStorageModeShared];
    if (!b) { *out = nullptr; return fail(mtlErrorMemoryAllocation); }
    void *p = b.contents;
    reg_add(p, n, b);
    *out = p;
    return mtlSuccess;
}

/* Apple exposes no GPU core count through Metal. IOKit does, and the value is
 * only ever used as a grid-size cap (fbgen_gpu.cu:1683), so a wrong answer
 * costs occupancy, not correctness -- hence the benign fallback. */
int query_core_count(void)
{
    int cores = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"), &it) == KERN_SUCCESS) {
        io_object_t obj;
        while ((obj = IOIteratorNext(it))) {
            CFTypeRef v = IORegistryEntrySearchCFProperty(
                obj, kIOServicePlane, CFSTR("gpu-core-count"),
                kCFAllocatorDefault, kIORegistryIterateRecursively | kIORegistryIterateParents);
            if (v) {
                if (CFGetTypeID(v) == CFNumberGetTypeID())
                    CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &cores);
                CFRelease(v);
            }
            IOObjectRelease(obj);
            if (cores) break;
        }
        IOObjectRelease(it);
    }
    return cores > 0 ? cores : 8;
}

id<MTLComputePipelineState> pso_for(const char *name)
{
    auto it = g_psos.find(name);
    if (it != g_psos.end()) return it->second;
    id<MTLFunction> f = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) { g_psos[name] = nil; return nil; }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) fprintf(stderr, "metal_rt: pipeline for '%s' failed: %s\n",
                    name, err.description.UTF8String);
    g_psos[name] = p;
    return p;
}

}  /* anonymous namespace */

/* ------------------------------------------------------------ lifecycle -- */

extern "C" mtlError_t mtlInit(const char *metallib_path)
{
    std::lock_guard<std::mutex> lk(g_lock);
    if (g_dev) return mtlSuccess;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return fail(mtlErrorInitialization);

    NSError *err = nil;
    NSString *path = metallib_path ? [NSString stringWithUTF8String:metallib_path]
                                   : @"bench.metallib";
    g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!g_lib) {
        fprintf(stderr, "metal_rt: cannot load shader library '%s': %s\n",
                path.UTF8String, err.description.UTF8String);
        return fail(mtlErrorInitialization);
    }
    g_default.q = [g_dev newCommandQueue];
    g_core_count = query_core_count();
    return mtlSuccess;
}

extern "C" void mtlShutdown(void)
{
    std::lock_guard<std::mutex> lk(g_lock);
    if (!g_dev) return;
    sync(&g_default);
    g_allocs.clear();
    g_psos.clear();
    g_symbols.clear();
    g_default = Stream();
    g_lib = nil; g_dev = nil;
}

extern "C" mtlError_t mtlSetDevice(int dev)   { return dev == 0 ? mtlSuccess : fail(mtlErrorInvalidValue); }
extern "C" mtlError_t mtlGetDevice(int *dev)  { if (dev) *dev = 0; return mtlSuccess; }
extern "C" mtlError_t mtlGetDeviceCount(int *n) { if (n) *n = g_dev ? 1 : 0; return mtlSuccess; }

extern "C" mtlError_t mtlGetDeviceProperties(mtlDeviceProp *p, int dev)
{
    if (!p || dev != 0 || !g_dev) return fail(mtlErrorInvalidValue);
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", g_dev.name.UTF8String);
    p->totalGlobalMem      = (size_t)g_dev.recommendedMaxWorkingSetSize;
    p->multiProcessorCount = g_core_count;
    /* Metal has no published grid-dimension ceiling; dispatchThreadgroups
     * takes NSUInteger. Report a large but honest 32-bit-safe bound, which is
     * how the one caller (bench_kernels.cu:2425) uses it. */
    p->maxGridSize[0] = 0x7fffffff; p->maxGridSize[1] = 0xffff; p->maxGridSize[2] = 0xffff;
    p->sharedMemPerBlock   = (size_t)g_dev.maxThreadgroupMemoryLength;
    p->maxThreadsPerBlock  = (int)g_dev.maxThreadsPerThreadgroup.width;
    p->warpSize            = 32;
    p->unifiedMemory       = g_dev.hasUnifiedMemory ? 1 : 0;
    return mtlSuccess;
}

extern "C" mtlError_t mtlMemGetInfo(size_t *freeB, size_t *totalB)
{
    if (!g_dev) return fail(mtlErrorInitialization);
    size_t total = (size_t)g_dev.recommendedMaxWorkingSetSize;
    size_t used  = (size_t)g_dev.currentAllocatedSize;
    if (totalB) *totalB = total;
    if (freeB)  *freeB  = used < total ? total - used : 0;
    return mtlSuccess;
}

/* --------------------------------------------------------------- memory -- */

extern "C" mtlError_t mtlMalloc(void **p, size_t n)
{ std::lock_guard<std::mutex> lk(g_lock); return alloc_shared(p, n); }

extern "C" mtlError_t mtlFree(void *p)
{
    if (!p) return mtlSuccess;
    std::lock_guard<std::mutex> lk(g_lock);
    return reg_erase(p) ? mtlSuccess : fail(mtlErrorNotMapped);
}

/* On UMA "pinned host memory" and device memory are the same thing. Backing
 * host allocations with buffers too means a staging pointer can be bound
 * directly if a kernel ever wants it, and keeps one free path. */
extern "C" mtlError_t mtlHostAlloc(void **p, size_t n, unsigned)
{ std::lock_guard<std::mutex> lk(g_lock); return alloc_shared(p, n); }

extern "C" mtlError_t mtlFreeHost(void *p) { return mtlFree(p); }

extern "C" mtlError_t mtlMemcpy(void *dst, const void *src, size_t n, int)
{
    if (!n) return mtlSuccess;
    if (!dst || !src) return fail(mtlErrorInvalidValue);
    mtlError_t e = mtlDeviceSynchronize();      /* cudaMemcpy's promise */
    if (e != mtlSuccess) return e;
    memcpy(dst, src, n);
    return mtlSuccess;
}

extern "C" mtlError_t mtlMemcpyAsync(void *dst, const void *src, size_t n,
                                     int kind, mtlStream_t s)
{
    /* Honest about what this is: the copy is a host memcpy, so it must not
     * start before work already queued on the stream has finished. That makes
     * it synchronous with respect to that stream -- weaker than CUDA's async
     * copy, but never wrong, and on UMA the copy itself is the cheap part. */
    if (!n) return mtlSuccess;
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = sync(resolve(s));
    if (e != mtlSuccess) return e;
    memcpy(dst, src, n);
    (void)kind;
    return mtlSuccess;
}

extern "C" mtlError_t mtlMemsetAsync(void *p, int value, size_t n, mtlStream_t s)
{
    if (!n) return mtlSuccess;
    if (!p) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    const Alloc *a = reg_find(p);
    if (!a) { memset(p, value, n); return mtlSuccess; }   /* plain host memory */
    Stream *st = resolve(s);
    id<MTLBlitCommandEncoder> b = ensure_blit(st);
    [b fillBuffer:a->buf
            range:NSMakeRange((uintptr_t)p - a->base, n)
            value:(uint8_t)value];
    return mtlSuccess;
}

extern "C" mtlError_t mtlMemset(void *p, int value, size_t n)
{
    mtlError_t e = mtlMemsetAsync(p, value, n, 0);
    if (e != mtlSuccess) return e;
    return mtlDeviceSynchronize();
}

/* -------------------------------------------------------------- symbols -- */

extern "C" mtlError_t mtlMemcpyToSymbol(const char *name, const void *src,
                                        size_t n, size_t offset)
{
    if (!name || !src) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_symbols.find(name);
    void *dst = nullptr;
    if (it == g_symbols.end()) {
        mtlError_t e = alloc_shared(&dst, n + offset);
        if (e != mtlSuccess) return e;
        g_symbols[name] = dst;
    } else {
        dst = it->second;
        const Alloc *a = reg_find(dst);
        if (!a || a->len < n + offset) return fail(mtlErrorInvalidValue);
    }
    mtlError_t e = sync(&g_default);
    if (e != mtlSuccess) return e;
    memcpy((char *)dst + offset, src, n);
    return mtlSuccess;
}

extern "C" void *mtlGetSymbol(const char *name)
{
    std::lock_guard<std::mutex> lk(g_lock);
    auto it = g_symbols.find(name);
    return it == g_symbols.end() ? nullptr : it->second;
}

/* -------------------------------------------------------------- streams -- */

extern "C" mtlError_t mtlStreamCreate(mtlStream_t *s)
{
    if (!s) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    Stream *st = new Stream();
    st->q = [g_dev newCommandQueue];
    if (!st->q) { delete st; return fail(mtlErrorInitialization); }
    *s = (mtlStream_t)st;
    return mtlSuccess;
}

extern "C" mtlError_t mtlStreamDestroy(mtlStream_t s)
{
    if (!s) return mtlSuccess;
    std::lock_guard<std::mutex> lk(g_lock);
    Stream *st = (Stream *)s;
    sync(st);
    delete st;
    return mtlSuccess;
}

extern "C" mtlError_t mtlStreamSynchronize(mtlStream_t s)
{ std::lock_guard<std::mutex> lk(g_lock); return sync(resolve(s)); }

extern "C" mtlError_t mtlDeviceSynchronize(void)
{ std::lock_guard<std::mutex> lk(g_lock); return sync(&g_default); }

/* --------------------------------------------------------------- events -- */

struct mtlEventOpaque {
    id<MTLEvent>         ev;
    uint64_t             value;
    id<MTLCommandBuffer> cb;      /* for timing and host waits */
};

extern "C" mtlError_t mtlEventCreate(mtlEvent_t *e)
{
    if (!e) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    mtlEventOpaque *ev = new mtlEventOpaque();
    ev->ev = [g_dev newEvent];
    ev->value = 0;
    ev->cb = nil;
    *e = ev;
    return mtlSuccess;
}

extern "C" mtlError_t mtlEventDestroy(mtlEvent_t e)
{
    if (!e) return mtlSuccess;
    std::lock_guard<std::mutex> lk(g_lock);
    delete e;
    return mtlSuccess;
}

/* PERFORMANCE NOTE, recorded rather than buried: a command buffer's
 * GPUEndTime is the only timestamp reachable without MTLCounterSampleBuffer,
 * so recording an event forces a commit and therefore a pipeline flush. The
 * CUDA source records events freely (78 sites, most of them timing), so a
 * port that keeps every one of them will serialise more than CUDA does. That
 * is a Phase 8 question -- either sample counters inside the encoder, or drop
 * the harness-only records -- not a correctness one. */
extern "C" mtlError_t mtlEventRecordOn(mtlEvent_t e, mtlStream_t s)
{
    if (!e) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    Stream *st = resolve(s);
    close_encoder(st);
    ensure_cb(st);
    e->value++;
    [st->cb encodeSignalEvent:e->ev value:e->value];
    e->cb = st->cb;
    [st->cb commit];
    st->last = st->cb;
    st->cb = nil;
    return mtlSuccess;
}

extern "C" mtlError_t mtlEventSynchronize(mtlEvent_t e)
{
    if (!e) return fail(mtlErrorInvalidValue);
    id<MTLCommandBuffer> cb;
    { std::lock_guard<std::mutex> lk(g_lock); cb = e->cb; }
    if (!cb) return mtlSuccess;
    [cb waitUntilCompleted];
    mtlError_t st = cb_status(cb);
    return st == mtlSuccess ? mtlSuccess : fail(st);
}

extern "C" mtlError_t mtlEventQuery(mtlEvent_t e)
{
    if (!e) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    if (!e->cb) return mtlSuccess;
    return e->cb.status == MTLCommandBufferStatusCompleted ? mtlSuccess : mtlErrorNotReady;
}

extern "C" mtlError_t mtlEventElapsedTime(float *ms, mtlEvent_t a, mtlEvent_t b)
{
    if (!ms || !a || !b) return fail(mtlErrorInvalidValue);
    mtlError_t e = mtlEventSynchronize(a); if (e != mtlSuccess) return e;
    e = mtlEventSynchronize(b);            if (e != mtlSuccess) return e;
    std::lock_guard<std::mutex> lk(g_lock);
    double t0 = a->cb ? a->cb.GPUEndTime : 0.0;
    double t1 = b->cb ? b->cb.GPUEndTime : 0.0;
    *ms = (float)((t1 - t0) * 1000.0);
    return mtlSuccess;
}

extern "C" mtlError_t mtlStreamWaitEvent(mtlStream_t s, mtlEvent_t e, unsigned)
{
    if (!e) return fail(mtlErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_lock);
    Stream *st = resolve(s);
    /* encodeWaitForEvent must precede any encoder on the command buffer, so
     * close out whatever is open and start a fresh one. */
    commit(st);
    ensure_cb(st);
    [st->cb encodeWaitForEvent:e->ev value:e->value];
    return mtlSuccess;
}

/* --------------------------------------------------------------- errors -- */

extern "C" mtlError_t mtlGetLastError(void)
{ mtlError_t e = g_last_error; g_last_error = mtlSuccess; return e; }

extern "C" mtlError_t mtlPeekAtLastError(void) { return g_last_error; }

extern "C" const char *mtlGetErrorString(mtlError_t e)
{
    switch (e) {
    case mtlSuccess:                   return "no error";
    case mtlErrorInvalidValue:         return "invalid value";
    case mtlErrorMemoryAllocation:     return "out of memory";
    case mtlErrorInitialization:       return "initialization error";
    case mtlErrorLaunchFailure:        return "kernel launch failed";
    case mtlErrorLaunchTimeout:        return "command buffer timed out (GPU watchdog)";
    case mtlErrorLaunchOutOfResources: return "launch out of resources";
    case mtlErrorNotReady:             return "not ready";
    case mtlErrorSymbolNotFound:       return "symbol not found";
    case mtlErrorKernelNotFound:       return "kernel not found in shader library";
    case mtlErrorNotMapped:            return "pointer is not a device allocation";
    case mtlErrorUnsupported:          return "unsupported on Metal";
    }
    return "unknown error";
}

extern "C" mtlError_t mtlFuncSetMaxThreadgroupMemory(const char *kernel, size_t bytes)
{
    if (!g_dev) return fail(mtlErrorInitialization);
    if (bytes > (size_t)g_dev.maxThreadgroupMemoryLength) {
        fprintf(stderr, "metal_rt: %s wants %zu B of threadgroup memory; this"
                " device allows %lu B\n", kernel ? kernel : "(kernel)", bytes,
                (unsigned long)g_dev.maxThreadgroupMemoryLength);
        return fail(mtlErrorLaunchOutOfResources);
    }
    return mtlSuccess;
}

/* --------------------------------------------------------------- launch -- */

extern "C" mtlError_t mtl_launch_begin(const char *kernel, mtlStream_t s,
                                       unsigned grid, unsigned block, size_t smem)
{
    g_lock.lock();
    id<MTLComputePipelineState> pso = pso_for(kernel);
    if (!pso) {
        fprintf(stderr, "metal_rt: no kernel named '%s' in the shader library\n", kernel);
        g_lock.unlock();
        return fail(mtlErrorKernelNotFound);
    }
    if (block > pso.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "metal_rt: %s launched with %u threads; max is %lu\n",
                kernel, block, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
        g_lock.unlock();
        return fail(mtlErrorLaunchOutOfResources);
    }
    Stream *st = resolve(s);
    id<MTLComputeCommandEncoder> enc = ensure_compute(st);
    [enc setComputePipelineState:pso];
    if (smem) [enc setThreadgroupMemoryLength:smem atIndex:0];
    g_bind_enc = enc; g_bind_stream = st; g_bind_grid = grid; g_bind_block = block;
    g_bind_error = mtlSuccess;
    return mtlSuccess;     /* lock held until mtl_launch_end */
}

extern "C" void mtl_bind_buffer(const void *ptr, int index)
{
    if (!ptr) { [g_bind_enc setBuffer:nil offset:0 atIndex:index]; return; }
    const Alloc *a = reg_find(ptr);
    if (!a) {
        fprintf(stderr, "metal_rt: argument %d (%p) is not inside any device"
                " allocation\n", index, ptr);
        g_bind_error = mtlErrorNotMapped;
        g_last_error = mtlErrorNotMapped;
        return;
    }
    [g_bind_enc setBuffer:a->buf offset:((uintptr_t)ptr - a->base) atIndex:index];
}

extern "C" void mtl_bind_bytes(const void *data, size_t sz, int index)
{ [g_bind_enc setBytes:data length:sz atIndex:index]; }

extern "C" mtlError_t mtl_launch_end(void)
{
    if (g_bind_grid && g_bind_block)
        [g_bind_enc dispatchThreadgroups:MTLSizeMake(g_bind_grid, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(g_bind_block, 1, 1)];
    g_bind_enc = nil; g_bind_stream = nullptr;
    mtlError_t e = g_bind_error;
    g_bind_error = mtlSuccess;
    g_lock.unlock();
    return e;
}
