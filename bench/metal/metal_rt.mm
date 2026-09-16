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
 * COMMAND BUFFER DURATION IS A SHIPPING CONSTRAINT, NOT A PERFORMANCE ONE.
 * macOS kills a command buffer that holds the GPU too long against the UI --
 * kIOGPUCommandBufferCallbackErrorImpactingInteractivity, a soft watchdog that
 * arrives as a plain "internal" error. It judges a COMMAND BUFFER. Because
 * dispatches batch into one buffer until something demands ordering, the unit
 * this port spent two rounds bounding -- the dispatch -- is not the unit that
 * gets killed: 50 cofactor rounds of a 244 ms launch, each comfortably inside
 * the 750 ms policy, were one 3,180 ms submission. Anything that issues a long
 * run of expensive dispatches must call mtlStreamFlush between them.
 *
 * CUDA_SIEVE_METAL_CBTIME=1 prints GPUEndTime - GPUStartTime and the contents
 * of every command buffer at each sync. It is here permanently because its
 * absence is exactly how 9z-j came to be measured, committed and signed while
 * changing nothing the watchdog could see. `make -f Makefile.metal cbtimecheck`
 * is the gate built on it.
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
#include <mach-o/dyld.h>
/* getsectiondata + _mh_execute_header: the embedded metallib, below. */
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>

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
    /* Every command buffer committed since the last sync. `last` alone was
     * enough while a sync followed every commit, but mtlStreamFlush breaks
     * that: buffers 1..n-1 would be replaced and released UNCHECKED, so a
     * failure in an early slice would vanish and the run would carry on with
     * garbage. Drained and checked by sync(). */
    std::vector<id<MTLCommandBuffer> > pending;
    /* What went into the command buffer now open, for the CBTIME report. */
    int         ndisp = 0;
    std::string first_k, last_k;
    std::vector<std::string> plabel;   /* one per pending buffer */
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
bool                 g_width_checked = false;

/* the encoder a launch is currently binding into */
id<MTLComputeCommandEncoder> g_bind_enc = nil;
Stream                      *g_bind_stream = nullptr;
unsigned                     g_bind_grid = 0, g_bind_block = 0;
/* The error belonging to the launch in progress, distinct from the sticky
 * g_last_error. CUDA's launch returns its OWN status and only
 * cudaGetLastError() is sticky; conflating the two makes a launch report a
 * failure that happened somewhere else entirely. */
mtlError_t                   g_bind_error = mtlSuccess;

/* CUDA_SIEVE_METAL_TRACE=1 reports every error at the point it is raised.
 * Ported code threads failures back through CK/PIPE_CK macros that often
 * `goto done` without printing, so a real fault can reach exit(255) with
 * nothing on stderr -- which is exactly how the first pipeline run failed. */
int g_trace = -1;
inline bool tracing(void)
{
    if (g_trace < 0) { const char *e = getenv("CUDA_SIEVE_METAL_TRACE");
                       g_trace = (e && *e && *e != '0') ? 1 : 0; }
    return g_trace != 0;
}

const char *err_name(mtlError_t e);

inline mtlError_t fail(mtlError_t e)
{
    g_last_error = e;
    if (tracing()) fprintf(stderr, "metal_rt: [trace] error raised: %s\n", err_name(e));
    return e;
}

/* Ported code calls mtlSetDevice / mtlMalloc exactly where the CUDA original
 * called cudaSetDevice / cudaMalloc, and never calls an init function -- CUDA
 * has none. So the shim initialises itself on first use, finding the shader
 * library next to the executable (CUDA_SIEVE_METALLIB overrides). Keeping the
 * ported call sequence identical to CUDA's is worth more than an explicit
 * init, because every deviation is a place a port can silently drift. */
mtlError_t ensure_init_locked(void);

struct AutoInit {
    AutoInit() { }
};

Stream *resolve(mtlStream_t s)
{ return s ? (Stream *)s : &g_default; }

/* ---- encoder lifecycle -------------------------------------------------- */

/* OWNERSHIP, because this file is compiled WITHOUT -fobjc-arc.
 *
 * commandBuffer, computeCommandEncoder and blitCommandEncoder all return
 * AUTORELEASED objects, and a Stream holds them across calls. Previously they
 * were stored unretained, which worked only because there is no autorelease
 * pool anywhere in this process: nothing ever drained, so nothing ever died.
 * That is a leak standing in for a lifetime -- and it made adding a pool
 * (below) a use-after-free rather than a fix. Every one of them is now
 * retained on store and released on replace or teardown, which is what lets
 * the pools exist at all.
 *
 * It is not a small leak. A command buffer retains every resource it
 * references, so each leaked one pinned its share of the sieve's buffers too;
 * the port commits one per event record, and cofac.cuh alone records 78. */
void close_encoder(Stream *st)
{
    if (st->kind == ENC_COMPUTE && st->cenc) { [st->cenc endEncoding]; [st->cenc release]; st->cenc = nil; }
    if (st->kind == ENC_BLIT    && st->benc) { [st->benc endEncoding]; [st->benc release]; st->benc = nil; }
    st->kind = ENC_NONE;
}

/* THE POOL GOES AROUND THE CREATION SITE, and it has to.
 *
 * These three factory methods return autoreleased objects: balancing our own
 * retain with a release is NOT enough, because the pending autorelease also
 * has to fire, and this process has no pool anywhere else -- a plain C++
 * main(), no Cocoa run loop. Without one the object is immortal however
 * carefully we balance our own reference.
 *
 * Creating inside the pool and retaining before it drains is the whole fix,
 * and it is local: nothing outside these functions has to know. The wider
 * alternative -- a pool around each entry point -- would be correct too, but
 * only AFTER the ownership work above, and it would put the pool boundary far
 * from the thing it governs. */
void ensure_cb(Stream *st)
{
    if (st->cb) return;
    @autoreleasepool { st->cb = [[st->q commandBuffer] retain]; }
}

id<MTLComputeCommandEncoder> ensure_compute(Stream *st)
{
    if (st->kind != ENC_COMPUTE) { close_encoder(st); ensure_cb(st);
        @autoreleasepool { st->cenc = [[st->cb computeCommandEncoder] retain]; }
        st->kind = ENC_COMPUTE; }
    return st->cenc;
}

id<MTLBlitCommandEncoder> ensure_blit(Stream *st)
{
    if (st->kind != ENC_BLIT) { close_encoder(st); ensure_cb(st);
        @autoreleasepool { st->benc = [[st->cb blitCommandEncoder] retain]; }
        st->kind = ENC_BLIT; }
    return st->benc;
}

/* Close and commit whatever is open. Returns the committed buffer, or the
 * previously committed one if there was nothing new. The stream keeps exactly
 * one committed buffer alive, because sync() and the event queries read it
 * after the fact; `last` takes over cb's reference rather than adding one. */
id<MTLCommandBuffer> commit(Stream *st)
{
    close_encoder(st);
    if (st->cb) {
        [st->cb commit];
        { char lb[256];
          snprintf(lb, sizeof lb, "%d dispatches %s..%s", st->ndisp,
                   st->first_k.empty() ? "-" : st->first_k.c_str(),
                   st->last_k.empty() ? "-" : st->last_k.c_str());
          st->plabel.push_back(lb); }
        st->ndisp = 0; st->first_k.clear(); st->last_k.clear();
        st->pending.push_back([st->cb retain]);   /* pending holds its own */
        if (st->last != st->cb) [st->last release];
        st->last = st->cb;              /* takes cb's +1 */
        st->cb = nil;
    }
    return st->last;
}

/* Release everything a Stream owns. Used by stream destruction and shutdown;
 * both used to drop the queue on the floor as well. */
void stream_teardown(Stream *st)
{
    close_encoder(st);
    for (size_t i = 0; i < st->pending.size(); i++) [st->pending[i] release];
    st->pending.clear();
    st->plabel.clear();
    [st->cb release];   st->cb = nil;
    [st->last release]; st->last = nil;
    [st->q release];    st->q = nil;
}

/* REPORT THE REAL REASON, ALWAYS.
 *
 * This used to print cb.error.localizedDescription only under
 * CUDA_SIEVE_METAL_TRACE and collapse everything that was not Timeout or
 * OutOfMemory into one code, so a field failure reached a volunteer's
 * uploaded stderr.txt as the single word "kernel launch failed" with the
 * diagnosis discarded. A real report looked like this, and nothing in it can
 * be acted on:
 *
 *   fbgen_gpu: mtlMemcpy(...) failed at fbgen_gpu_metal.cpp:568:
 *   kernel launch failed
 *
 * Metal's own description distinguishes the cases that matter -- a page fault
 * (a bad address, i.e. the class 9z-c was about), a GPU hang, the process
 * being a victim of someone else's fault, device removal, a stack overflow --
 * and a failure is fatal anyway, so there is no cost to saying which. The
 * error code goes out too: descriptions are localised, the code is not. */
mtlError_t cb_status(id<MTLCommandBuffer> cb)
{
    if (!cb || !cb.error) return mtlSuccess;
    @autoreleasepool {
        const char *what = "?";
        switch (cb.error.code) {
        case MTLCommandBufferErrorNone:            what = "none"; break;
        case MTLCommandBufferErrorInternal:        what = "internal"; break;
        case MTLCommandBufferErrorTimeout:         what = "timeout (GPU watchdog)"; break;
        case MTLCommandBufferErrorPageFault:       what = "page fault (bad device address)"; break;
        case MTLCommandBufferErrorNotPermitted:    what = "not permitted"; break;
        case MTLCommandBufferErrorOutOfMemory:     what = "out of memory"; break;
        case MTLCommandBufferErrorInvalidResource: what = "invalid resource"; break;
        case MTLCommandBufferErrorMemoryless:      what = "memoryless"; break;
        case MTLCommandBufferErrorDeviceRemoved:   what = "device removed"; break;
        case MTLCommandBufferErrorStackOverflow:   what = "stack overflow"; break;
        default: break;
        }
        fprintf(stderr, "metal_rt: command buffer failed: %s (code %ld): %s\n",
                what, (long)cb.error.code,
                cb.error.localizedDescription
                    ? cb.error.localizedDescription.UTF8String : "(no description)");
    }
    if (cb.error.code == MTLCommandBufferErrorTimeout) return mtlErrorLaunchTimeout;
    if (cb.error.code == MTLCommandBufferErrorOutOfMemory) return mtlErrorLaunchOutOfResources;
    return mtlErrorLaunchFailure;
}

mtlError_t sync(Stream *st)
{
    id<MTLCommandBuffer> cb = commit(st);
    if (!cb) return mtlSuccess;
    [cb waitUntilCompleted];
    { static int cbt = -1;
      if (cbt < 0) { const char *e = getenv("CUDA_SIEVE_METAL_CBTIME"); cbt = (e && *e && *e != '0') ? 1 : 0; }
      if (cbt) for (size_t i = 0; i < st->pending.size(); i++)
          fprintf(stderr, "CBTIME %.2f ms [%s]\n",
                  (st->pending[i].GPUEndTime - st->pending[i].GPUStartTime) * 1000.0,
                  i < st->plabel.size() ? st->plabel[i].c_str() : "?"); }
    /* Command buffers on one queue complete in commit order, so waiting for
     * the last one means every earlier one is done and its status is final.
     * Report the FIRST failure: a later buffer's error is usually a
     * consequence of the first, and the first is the one worth diagnosing. */
    mtlError_t e = mtlSuccess;
    for (size_t i = 0; i < st->pending.size(); i++) {
        mtlError_t ei = cb_status(st->pending[i]);
        if (ei != mtlSuccess && e == mtlSuccess) e = ei;
        [st->pending[i] release];
    }
    st->pending.clear();
    st->plabel.clear();
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

/* RELEASE, not just forget. This file is compiled WITHOUT -fobjc-arc, so the
 * `id<MTLBuffer>` inside Alloc is an unmanaged pointer: newBufferWithLength:
 * returns +1 and the registry entry is the only owner. Erasing the entry
 * without releasing leaked the whole allocation -- measured at 64 MB per
 * malloc/free pair, and ~3.7 GB over a pipeline run, because slab calibration
 * builds and tears down the bucket array and factor bases three times before
 * the real band. Safe to release here for the same reason cudaFree is: a
 * command buffer retains the resources it references, so work already
 * encoded keeps the buffer alive past this call. */
bool reg_erase(void *p)
{
    uintptr_t v = (uintptr_t)p;
    auto it = std::lower_bound(g_allocs.begin(), g_allocs.end(), v,
                               [](const Alloc &x, uintptr_t q){ return x.base < q; });
    if (it == g_allocs.end() || it->base != v) return false;
    [it->buf release];
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

/* Pipelines are cached per (name, nilmask): a kernel with an optional buffer
 * is a DIFFERENT function object depending on whether that argument exists,
 * because the argument is declared under an MSL function constant.
 *
 * ONE constant carries the whole answer -- a uint bitmask at index 0, bit i
 * meaning "argument i is bound" -- and a kernel derives its per-argument
 * booleans from it:
 *
 *     constant uint mtl_bound_mask [[function_constant(0)]];
 *     constant bool bsum_bound = (mtl_bound_mask & (1u << 1)) != 0;
 *
 * A bitmask rather than one Bool per buffer index because the validation
 * layer REJECTS a constant value for an index the function does not declare,
 * so the host cannot set a whole range speculatively -- and it has no way to
 * ask which indices a function declares without first creating it.
 *
 * Discovery is `functionConstantsDictionary`. A plain newFunctionWithName:
 * does NOT return nil for a specialised function -- it returns an object that
 * cannot build a pipeline, and Metal asserts when you try:
 *
 *     function k_scan_block cannot be used to build a pipeline state.
 *     Use newFunctionWithName:constantValues:... to get the specialized function
 *
 * and that assertion fires in ordinary builds, not only under the validation
 * layer. So the plain lookup is a probe: ask the function what constants it
 * declares, and re-create it specialised when there are any. */
id<MTLComputePipelineState> pso_for(const char *name, uint32_t nilmask)
{
    char key[256];
    snprintf(key, sizeof key, "%s#%u", name, (unsigned)nilmask);
    auto it = g_psos.find(key);
    if (it != g_psos.end()) return it->second;
    id<MTLComputePipelineState> p = nil;
    @autoreleasepool {
        NSString *nm = [NSString stringWithUTF8String:name];
        id<MTLFunction> f = [g_lib newFunctionWithName:nm];
        if (f && f.functionConstantsDictionary.count) {
            [f release];
            f = nil;
        }
        if (!f) {
            const uint32_t bound = ~nilmask;
            MTLFunctionConstantValues *cv = [[MTLFunctionConstantValues alloc] init];
            [cv setConstantValue:&bound type:MTLDataTypeUInt atIndex:0];
            NSError *cerr = nil;
            f = [g_lib newFunctionWithName:nm constantValues:cv error:&cerr];
            [cv release];
            if (!f)
                fprintf(stderr, "metal_rt: specialising '%s' failed: %s\n",
                        name, cerr.description.UTF8String);
        }
        if (!f) { g_psos[key] = nil; return nil; }
        NSError *err = nil;
        p = [g_dev newComputePipelineStateWithFunction:f error:&err];
        [f release];   /* +1 from newFunctionWithName:; the PSO holds what it needs */
        if (!p) fprintf(stderr, "metal_rt: pipeline for '%s' failed: %s\n",
                        name, err.description.UTF8String);
    }
    g_psos[key] = p;   /* +1 from newComputePipelineState...; released at shutdown */
    return p;
}

}  /* anonymous namespace */

/* ------------------------------------------------------------ lifecycle -- */

static mtlError_t init_locked(const char *metallib_path)
{
    if (g_dev) return mtlSuccess;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return fail(mtlErrorInitialization);

    /* Four places, in this order:
     *
     *   1. the explicit argument,
     *   2. $CUDA_SIEVE_METALLIB,
     *   3. a metallib EMBEDDED in this executable's __DATA,__metallib,
     *   4. bench.metallib next to the executable.
     *
     * 1 and 2 come before the embedded copy so a developer can still point a
     * shipped binary at a rebuilt library -- every gate here drives the build
     * through $CUDA_SIEVE_METALLIB and must keep doing so. 3 comes before 4
     * because a binary that carries its own shaders should trust itself over
     * whatever file happens to sit beside it; a stale bench.metallib in the
     * working directory is exactly the kind of thing a BOINC slot accumulates.
     *
     * Embedding is what makes the application ONE FILE. A BOINC project ships
     * an executable, and a second file that must land beside it and match it
     * is a class of failure (missing, stale, mismatched) that simply does not
     * exist if the shaders are inside. */
    NSError *err = nil;
    NSString *path = nil;
    if (metallib_path) {
        path = [NSString stringWithUTF8String:metallib_path];
    } else if (const char *env = getenv("CUDA_SIEVE_METALLIB")) {
        path = [NSString stringWithUTF8String:env];
    }

    if (!path) {
        unsigned long n = 0;
        const uint8_t *p = getsectiondata(&_mh_execute_header,
                                          "__DATA", "__metallib", &n);
        if (p && n) {
            /* No copy and no free: the bytes are in our own __DATA and live
             * as long as the process, so the destructor is a no-op block.
             * DISPATCH_DATA_DESTRUCTOR_DEFAULT would copy all of it. */
            dispatch_data_t d = dispatch_data_create(p, (size_t)n, nil, ^{});
            g_lib = [g_dev newLibraryWithData:d error:&err];
            dispatch_release(d);
            if (!g_lib) {
                fprintf(stderr,
                        "metal_rt: the embedded shader library (%lu bytes) did"
                        " not load: %s\n", n, err.description.UTF8String);
                return fail(mtlErrorInitialization);
            }
        }
    }

    if (!g_lib) {
        if (!path) {
            char buf[4096]; uint32_t sz = sizeof buf;
            if (_NSGetExecutablePath(buf, &sz) != 0) buf[0] = 0;
            NSString *dir = [[NSString stringWithUTF8String:buf]
                             stringByDeletingLastPathComponent];
            path = [dir stringByAppendingPathComponent:@"bench.metallib"];
        }
        g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!g_lib) {
            fprintf(stderr, "metal_rt: cannot load shader library '%s': %s\n",
                    path.UTF8String, err.description.UTF8String);
            fprintf(stderr, "metal_rt: and this binary carries no embedded"
                    " library (build with EMBED_METALLIB=1)\n");
            return fail(mtlErrorInitialization);
        }
    }
    /* FAIL CLOSED ON UNSUPPORTED HARDWARE.
     *
     * The sieve assumes a 32-lane warp everywhere -- `>> 5`, `& 31`, lane 31
     * broadcasts, and td.cuh:647's ballot-ordering argument. Apple GPUs are
     * 32-wide and MTLGPUFamilyApple7 (M1) is the floor this port targets, but
     * Metal also runs on Intel Macs with AMD or Intel GPUs, where the SIMD
     * width is 64 or 8. Those would not crash; they would quietly compute a
     * different factor base. Refuse them by name instead.
     *
     * The threadExecutionWidth cross-check happens at the first launch
     * (mtl_launch_begin), because it is a pipeline property, not a device one. */
    if (![g_dev supportsFamily:MTLGPUFamilyApple7]) {
        fprintf(stderr,
                "metal_rt: '%s' is not an Apple silicon GPU of family Apple7 or\n"
                "          later (M1 and up). This build assumes a 32-lane SIMD\n"
                "          group throughout and would produce wrong results on\n"
                "          a 64-lane or 8-lane device, so it refuses to run.\n",
                g_dev.name.UTF8String);
        g_lib = nil; g_dev = nil;
        return fail(mtlErrorUnsupported);
    }

    g_default.q = [g_dev newCommandQueue];
    g_core_count = query_core_count();
    return mtlSuccess;
}

extern "C" mtlError_t mtlInit(const char *metallib_path)
{
    std::lock_guard<std::mutex> lk(g_lock);
    return init_locked(metallib_path);
}

namespace {
mtlError_t ensure_init_locked(void)
{
    if (g_dev) return mtlSuccess;
    return init_locked(nullptr);
}
}

extern "C" void mtlShutdown(void)
{
    std::lock_guard<std::mutex> lk(g_lock);
    if (!g_dev) return;
    sync(&g_default);
    /* Same ownership rule as reg_erase: clear() would drop the references
     * without releasing them. */
    for (Alloc &a : g_allocs) [a.buf release];
    g_allocs.clear();
    for (auto &kv : g_psos) [kv.second release];
    g_psos.clear();
    g_symbols.clear();
    stream_teardown(&g_default);
    g_default = Stream();
    g_lib = nil; g_dev = nil;
}

/* These three are the first calls a ported main() makes, so they are where
 * lazy initialisation actually has to happen. mtlGetDeviceCount in particular
 * must NOT answer 0 merely because nothing has touched the GPU yet -- that
 * reads to the caller as "this process sees no device" and aborts the run. */
extern "C" mtlError_t mtlSetDevice(int dev)
{
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = ensure_init_locked();
    if (e != mtlSuccess) return e;
    return dev == 0 ? mtlSuccess : fail(mtlErrorInvalidValue);
}

extern "C" mtlError_t mtlGetDevice(int *dev)
{
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = ensure_init_locked();
    if (e != mtlSuccess) return e;
    if (dev) *dev = 0;
    return mtlSuccess;
}

extern "C" mtlError_t mtlGetDeviceCount(int *n)
{
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = ensure_init_locked();
    if (n) *n = (e == mtlSuccess && g_dev) ? 1 : 0;
    return mtlSuccess;
}

extern "C" mtlError_t mtlGetDeviceProperties(mtlDeviceProp *p, int dev)
{
    { std::lock_guard<std::mutex> _lk(g_lock); mtlError_t _e = ensure_init_locked(); if (_e != mtlSuccess) return _e; }
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
    { std::lock_guard<std::mutex> _lk(g_lock); mtlError_t _e = ensure_init_locked(); if (_e != mtlSuccess) return _e; }
    if (!g_dev) return fail(mtlErrorInitialization);
    size_t total = (size_t)g_dev.recommendedMaxWorkingSetSize;
    size_t used  = (size_t)g_dev.currentAllocatedSize;
    if (totalB) *totalB = total;
    if (freeB)  *freeB  = used < total ? total - used : 0;
    return mtlSuccess;
}

/* --------------------------------------------------------------- memory -- */

extern "C" mtlError_t mtlMalloc(void **p, size_t n)
{
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = ensure_init_locked();
    return e == mtlSuccess ? alloc_shared(p, n) : e;
}

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
{
    std::lock_guard<std::mutex> lk(g_lock);
    mtlError_t e = ensure_init_locked();
    return e == mtlSuccess ? alloc_shared(p, n) : e;
}

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
    { mtlError_t _e = ensure_init_locked(); if (_e != mtlSuccess) return _e; }
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
    { std::lock_guard<std::mutex> _lk(g_lock); mtlError_t _e = ensure_init_locked(); if (_e != mtlSuccess) return _e; }
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
    stream_teardown(st);
    delete st;
    return mtlSuccess;
}

extern "C" mtlError_t mtlStreamSynchronize(mtlStream_t s)
{ std::lock_guard<std::mutex> lk(g_lock); return sync(resolve(s)); }

/* Submit without waiting. Command buffers on one MTLCommandQueue execute in
 * commit order, so splitting a run of dispatches across several changes
 * nothing about ordering or hazard tracking -- only how long any single
 * submission occupies the GPU, which is what the watchdog measures. The CPU
 * keeps running ahead, so this is not a pipeline stall the way an event
 * record is. */
extern "C" mtlError_t mtlStreamFlush(mtlStream_t s)
{
    std::lock_guard<std::mutex> lk(g_lock);
    commit(resolve(s));
    return mtlSuccess;
}

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
    { std::lock_guard<std::mutex> _lk(g_lock); mtlError_t _e = ensure_init_locked(); if (_e != mtlSuccess) return _e; }
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
    [e->ev release];   /* +1 from newEvent */
    [e->cb release];   /* retained in mtlEventRecordOn */
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
    /* commit() transfers cb's reference to st->last; the event needs its OWN,
     * because it outlives the stream's next commit and is read afterwards by
     * mtlEventSynchronize/Query/ElapsedTime. Retain before release, in case
     * this event is being re-recorded onto the same buffer. */
    id<MTLCommandBuffer> cb = commit(st);
    [cb retain];
    [e->cb release];
    e->cb = cb;
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

namespace { const char *err_name(mtlError_t e) { return mtlGetErrorString(e); } }

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
                                       unsigned grid, unsigned block, size_t smem,
                                       uint32_t nilmask)
{
    g_lock.lock();
    if (ensure_init_locked() != mtlSuccess) { g_lock.unlock(); return fail(mtlErrorInitialization); }
    id<MTLComputePipelineState> pso = pso_for(kernel, nilmask);
    if (pso && !g_width_checked) {
        g_width_checked = true;
        if (pso.threadExecutionWidth != 32) {
            fprintf(stderr, "metal_rt: SIMD width is %lu, not 32 -- this build's"
                    " warp arithmetic is invalid on this device\n",
                    (unsigned long)pso.threadExecutionWidth);
            g_lock.unlock();
            return fail(mtlErrorUnsupported);
        }
    }
    if (!pso) {
        fprintf(stderr, "metal_rt: no kernel named '%s' in the shader library\n", kernel);
        g_lock.unlock();
        return fail(mtlErrorKernelNotFound);
    }
    if (tracing() && block > pso.maxTotalThreadsPerThreadgroup)
        fprintf(stderr, "metal_rt: [trace] %s: %u threads requested, pipeline"
                " allows %lu\n", kernel, block,
                (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    if (block > pso.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "metal_rt: %s launched with %u threads; max is %lu\n",
                kernel, block, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
        g_lock.unlock();
        return fail(mtlErrorLaunchOutOfResources);
    }
    if (tracing()) fprintf(stderr, "metal_rt: [trace] launch %s grid=%u block=%u smem=%zu\n",
                           kernel, grid, block, smem);
    Stream *st = resolve(s);
    id<MTLComputeCommandEncoder> enc = ensure_compute(st);
    /* Accounting for the CBTIME report: which dispatches share this command
     * buffer. ensure_compute may have opened a fresh one, so record after. */
    if (st->ndisp == 0) st->first_k = kernel ? kernel : "?";
    st->last_k = kernel ? kernel : "?";
    st->ndisp++;
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

extern "C" uint64_t mtlDeviceAddress(const void *p)
{
    if (!p) return 0;
    std::lock_guard<std::mutex> lk(g_lock);
    const Alloc *a = reg_find(p);
    if (!a) { g_last_error = mtlErrorNotMapped; return 0; }
    return (uint64_t)a->buf.gpuAddress + ((uintptr_t)p - a->base);
}

/* Must be called while a launch is being bound: the encoder is the scope
 * residency applies to. */
extern "C" void mtlUseResource(const void *p)
{
    if (!p || !g_bind_enc) return;
    const Alloc *a = reg_find(p);
    if (!a) { g_bind_error = mtlErrorNotMapped; g_last_error = mtlErrorNotMapped; return; }
    [g_bind_enc useResource:a->buf usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
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
