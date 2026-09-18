#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces cofac_metal.cpp from cofac.cuh:
# the __global__ kernels are removed (they live in metal/cofac.metal now) and
# the CUDA runtime calls are renamed onto metal_rt.h. The mz<L> arithmetic
# STAYS -- it is CF_FN, i.e. host-and-device, and the host half parses and
# verifies with it.
import re

SRC = 'bench/cofac.cuh'
OUT = 'bench/metal/cofac_metal.cpp'
src = open(SRC).read()

# ---- remove every __global__ definition ---------------------------------
def strip_kernels(text):
    out, i, n = [], 0, 0
    while True:
        # `__global__ __launch_bounds__(N, M)` then `void` on the NEXT line is
        # a shape upstream introduced (2bc1c6e). Without the optional group
        # this pattern silently stopped matching k_cofac, the kernel count went
        # 9 -> 8, and its whole device definition was left sitting in the HOST
        # file. Nothing asserted the count, so the only signal was the number
        # in the generator's own output line.
        m = re.search(r'(?:template\s*<[^>]*>\s*)?__global__\s+'
                      r'(?:__launch_bounds__\([^)]*\)\s*)?void\s+\w+\s*\(', text[i:])
        if not m:
            out.append(text[i:]); break
        s0 = i + m.start()
        b = text.index('{', i + m.end() - 1)
        d = 0; j = b
        while j < len(text):
            if text[j] == '{': d += 1
            elif text[j] == '}':
                d -= 1
                if d == 0: break
            j += 1
        out.append(text[i:s0])
        out.append('/* kernel moved to metal/cofac.metal */\n')
        i = j + 1; n += 1
    return ''.join(out), n

src, nk = strip_kernels(src)

# ---- launches ------------------------------------------------------------
ALLOWED = {'k_cofac', 'k_cof_selflags', 'k_cof_selscatter',
           'k_scan_pass1', 'k_scan_pass2', 'k_scan_pass3',
           'k_cof_enqueue', 'k_cof_gate', 'k_cof_status_hist',
           'k_rel_flags', 'k_rel_gather', 'k_rel_pack'}

def rewrite_launches(text):
    out, i, count, skipped = [], 0, 0, set()
    while True:
        j = text.find('<<<', i)
        if j < 0:
            out.append(text[i:]); break
        k = j; targs = None
        if text[k - 1] == '>':
            d = 0; k -= 1
            while k >= 0:
                if text[k] == '>': d += 1
                elif text[k] == '<':
                    d -= 1
                    if d == 0: break
                k -= 1
            targs = text[k + 1:j - 1]
        e = k
        while e > 0 and (text[e - 1].isalnum() or text[e - 1] == '_'): e -= 1
        base = text[e:k]
        d = 0; m = j + 3
        while m < len(text):
            if text[m] == '<': d += 1
            elif text[m] == '>':
                if d == 0 and text[m:m + 3] == '>>>': break
                d -= 1
            elif text[m] == '(': d += 1
            elif text[m] == ')': d -= 1
            m += 1
        cfg = text[j + 3:m]
        d = 0; parts = []; cur = ''
        for ch in cfg:
            if ch in '(<[': d += 1
            elif ch in ')>]': d -= 1
            if ch == ',' and d == 0: parts.append(cur); cur = ''
            else: cur += ch
        parts.append(cur)
        grid, block = parts[0], parts[1]
        a = text.index('(', m); d = 0; b = a
        while b < len(text):
            if text[b] == '(': d += 1
            elif text[b] == ')':
                d -= 1
                if d == 0: break
            b += 1
        args = text[a + 1:b]
        if base not in ALLOWED:
            skipped.add(base)
        name = base
        if targs:
            tv = [t.strip() for t in targs.split(',')]
            name = base + '_' + '_'.join('1' if t == 'true' else '0' if t == 'false' else t for t in tv)
        out.append(text[i:e])
        out.append('MTL_LAUNCH(%s, %s, %s, 0, 0, %s)'
                   % (name, grid.strip(), block.strip(),
                      ' '.join(args.replace('\\\n', ' ').replace('\\', ' ').split())))
        count += 1
        i = b + 1
    return ''.join(out), count, skipped

src, nl, skipped = rewrite_launches(src)

REN = [('cudaDeviceSynchronize','mtlDeviceSynchronize'), ('cudaGetLastError','mtlGetLastError'),
       ('cudaGetErrorString','mtlGetErrorString'), ('cudaMemcpyDeviceToHost','mtlMemcpyDeviceToHost'),
       ('cudaMemcpyHostToDevice','mtlMemcpyHostToDevice'),
       ('cudaMemcpyDeviceToDevice','mtlMemcpyDeviceToDevice'),
       ('cudaEventElapsedTime','mtlEventElapsedTime'), ('cudaEventSynchronize','mtlEventSynchronize'),
       ('cudaEventDestroy','mtlEventDestroy'), ('cudaEventCreate','mtlEventCreate'),
       ('cudaEventRecord','mtlEventRecord'), ('cudaEvent_t','mtlEvent_t'),
       ('cudaStreamSynchronize','mtlStreamSynchronize'), ('cudaStream_t','mtlStream_t'),
       ('cudaMemcpyAsync','mtlMemcpyAsync'), ('cudaMemsetAsync','mtlMemsetAsync'),
       ('cudaMemcpy','mtlMemcpy'), ('cudaMemset','mtlMemset'),
       ('cudaMalloc','mtlMalloc'), ('cudaFreeHost','mtlFreeHost'), ('cudaHostAlloc','mtlHostAlloc'),
       ('cudaFree','mtlFree'), ('cudaError_t','mtlError_t'), ('cudaSuccess','mtlSuccess'),
       ('cudaErrorLaunchTimeout','mtlErrorLaunchTimeout'), ('cudaErrorNotReady','mtlErrorNotReady')]
for a, b in REN: src = src.replace(a, b)

# ---- the __CUDACC__ guards ----------------------------------------------
# THE TRAP THE HIP PORT'S LEDGER WARNS ABOUT, and it bit here exactly as
# described: cofac.cuh wraps its entire GPU host driver -- run_cofac included
# -- in `#if defined(__CUDACC__)`. Compiled as ordinary C++ that block simply
# vanishes, so the file builds with ZERO errors and then fails to link.
#
# The two guards do opposite things and must be treated differently:
#   the CF_FN / CF_NOINLINE / CF_HD block wants the NON-CUDA branch, because
#   host code needs `static inline`, not `__device__`;
#   every other guard wraps the driver and must be ENABLED.
#
# hipify-perl rewrites __CUDACC__ outright when forking a file, which is
# correct for a fork and is what happens here.
head, sep, rest = src.partition('#else')
assert 'CF_FN __device__' in head, 'CF_FN guard is not the first __CUDACC__ block'
rest = rest.replace('#if defined(__CUDACC__)', '#if 1  /* was __CUDACC__: this fork is the GPU build */')
rest = rest.replace('#endif  /* __CUDACC__ */', '#endif  /* was __CUDACC__ */')
nguard = rest.count('#if 1  /* was __CUDACC__')
src = head + sep + rest
print('  enabled %d __CUDACC__ driver guard(s)' % nguard)

hdr = '''/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * cofac.cuh's HOST half, on metal_rt.h. Generated by
 * metal/gen_cofac_host.py; the generator is a porting aid, not part of the
 * build, and this file is committed and reviewed like any other source.
 *
 * The mz<L> arithmetic stays: it is CF_FN (host and device alike) and the
 * host half parses, narrows and verifies cofactors with it. Only the
 * __global__ definitions are gone -- @NK@ of them, now in metal/cofac.metal.
 */
#include "bench.h"
#include "platform.h"
#include "bigint.cuh"
#include "td.cuh"
#include "metal/metal_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <algorithm>

/* cofac.cuh is a header: bench_kernels.cu supplies these before including it.
 * A standalone translation unit has to supply them itself. */
static int mtl_check_impl(mtlError_t err, const char *expr,
                          const char *file, int line)
{
    if (err == mtlSuccess) return 0;
    fprintf(stderr, "Metal %s: %s at %s:%d\\n",
            expr, mtlGetErrorString(err), file, line);
    return -1;
}
#define CUDA_CHECKED(x) mtl_check_impl((x), #x, __FILE__, __LINE__)
#define CK(x) do { if (CUDA_CHECKED(x)) return -1; } while (0)

/* td.cuh keeps its TD_* bounds inside its own __CUDACC__ guard, so a host
 * translation unit never sees them. td_host.h is generated from td.cuh by
 * metal/gen_td_host.py and lifts them by name -- no hand copies. */
#include "metal/td_host.h"

'''.replace('@NK@', str(nk))
# ---- two host helpers cofac.cuh borrows from its including TU ------------
# cofac.cuh is a header; bench_kernels.cu defines host_ms() and
# td_build_poly() above the point where it includes cofac.cuh. A standalone
# translation unit has to bring them along. EXTRACTED, not hand-copied, so
# regenerating tracks the originals.
bk = open('bench/bench_kernels.cu').read()
borrowed = []
for fn in ('static double host_ms(void)', 'static int td_build_poly('):
    i = bk.index(fn)
    d = 0; j = bk.index('{', i)
    k = j
    while k < len(bk):
        if bk[k] == '{': d += 1
        elif bk[k] == '}':
            d -= 1
            if d == 0: break
        k += 1
    borrowed.append(bk[i:k + 1])
# ---- launches whose instantiation is a template parameter ----------------
# cf_run_rounds<L> launches k_cofac<L, METHOD, STAGE2>. L is the enclosing
# function's own template parameter, so the mangled kernel name cannot be
# formed textually -- the naive rewrite produces "k_cofac_L_0_0". Select it at
# run time from L instead, which is exactly what the CUDA compiler was doing
# at compile time.
HELPER = (
"static const char *cf_kname(int L, int method, int stage2)" + chr(10) +
"{" + chr(10) +
"    if (L == 3) return method ? (stage2 ? \"k_cofac_3_1_1\" : \"k_cofac_3_1_0\")" + chr(10) +
"                              : \"k_cofac_3_0_0\";" + chr(10) +
"    return method ? (stage2 ? \"k_cofac_4_1_1\" : \"k_cofac_4_1_0\")" + chr(10) +
"                  : \"k_cofac_4_0_0\";" + chr(10) +
"}" + chr(10))
for suffix, meth, st2 in (('1_1', '1', '1'), ('1_0', '1', '0'), ('0_0', '0', '0')):
    src = src.replace('MTL_LAUNCH(k_cofac_L_' + suffix + ',',
                      'MTL_LAUNCH_NAMED(cf_kname(L, ' + meth + ', ' + st2 + '),')
ARGBUF = (
"/* ---- cofq_t as an argument buffer --------------------------------------" + chr(10) +
" *" + chr(10) +
" * CUDA passes cofq_t to k_cof_enqueue and k_rel_pack BY VALUE. A host pointer" + chr(10) +
" * means nothing to a shader, so the device gets a struct of GPU ADDRESSES" + chr(10) +
" * instead -- cofq_dev_t in metal/cofac.metal, whose member order and types" + chr(10) +
" * this mirror must match exactly." + chr(10) +
" *" + chr(10) +
" * Residency is not optional: a buffer reached through a raw address is" + chr(10) +
" * invisible to Metal's own tracking, and omitting mtlUseResource gives" + chr(10) +
" * garbage rather than an error (metal/argbuf_test.cpp shows 4095 of 4096" + chr(10) +
" * values wrong in its negative control). Passing the struct as mtl_argbuf_t" + chr(10) +
" * makes mtl_launch do the binding and the residency together, so a call site" + chr(10) +
" * cannot do one and forget the other." + chr(10) +
" */" + chr(10) +
"struct cofq_dev_t {" + chr(10) +
"    uint64_t d_c0, d_c1, d_st0, d_st1, d_sm0, d_sm1, d_a, d_b," + chr(10) +
"             d_f0, d_f1, d_fn0, d_fn1, d_sp0, d_sp1, d_nsp0, d_nsp1, d_ovf;" + chr(10) +
"};" + chr(10) + chr(10) +
"static cofq_dev_t *g_cofq_dev = NULL;" + chr(10) +
"static const void *g_cofq_refs[17];" + chr(10) + chr(10) +
"static mtl_argbuf_t cofq_argbuf(const cofq_t *Q)" + chr(10) +
"{" + chr(10) +
"    if (!g_cofq_dev && mtlMalloc((void **)&g_cofq_dev, sizeof *g_cofq_dev) != mtlSuccess) {" + chr(10) +
"        fprintf(stderr, \"cofq_argbuf: out of memory\\n\");" + chr(10) +
"        mtl_argbuf_t z = { NULL, NULL, 0 }; return z;" + chr(10) +
"    }" + chr(10) +
"    const void *p[17] = { Q->d_c0, Q->d_c1, Q->d_st0, Q->d_st1, Q->d_sm0," + chr(10) +
"                          Q->d_sm1, Q->d_a, Q->d_b, Q->d_f0, Q->d_f1," + chr(10) +
"                          Q->d_fn0, Q->d_fn1, Q->d_sp0, Q->d_sp1," + chr(10) +
"                          Q->d_nsp0, Q->d_nsp1, Q->d_ovf };" + chr(10) +
"    uint64_t *a = (uint64_t *)g_cofq_dev;" + chr(10) +
"    for (int i = 0; i < 17; i++) { a[i] = mtlDeviceAddress(p[i]); g_cofq_refs[i] = p[i]; }" + chr(10) +
"    mtl_argbuf_t r = { g_cofq_dev, g_cofq_refs, 17 };" + chr(10) +
"    return r;" + chr(10) +
"}" + chr(10))
# ARGBUF references cofq_t, so it has to sit AFTER that struct's definition
# rather than at the top of the file.
anchor = '} cofq_t;'
assert anchor in src, 'cofq_t definition not found'
src = src.replace(anchor, anchor + chr(10) + chr(10) + ARGBUF, 1)
src = HELPER + chr(10) + src

# The two by-value cofq_t launches take the argument buffer instead.
src = src.replace(", lpb0, lpb1, *Q);", ", lpb0, lpb1, cofq_argbuf(Q));")
src = src.replace("MTL_LAUNCH(k_rel_pack, blocks, threads, 0, 0, nr, Q->d_idx, *Q,",
                  "MTL_LAUNCH(k_rel_pack, blocks, threads, 0, 0, nr, Q->d_idx, cofq_argbuf(Q),")

# CF_ENQ(A, B) parameterises the template arguments, so the mangled kernel
# name has to be pasted by the preprocessor rather than formed here.
src = src.replace("MTL_LAUNCH(k_cof_enqueue_A_B,",
                  "MTL_LAUNCH(k_cof_enqueue_##A##_##B,")

src = (hdr
       + '/* ---- borrowed verbatim from bench_kernels.cu, which defines these\n'
         ' * above its #include of cofac.cuh. Extracted by the generator so they\n'
         ' * track the originals rather than drifting as hand copies. */\n'
       + '\n\n'.join(borrowed) + '\n\n'
       + src)
# ---- decouple the chunk floor from the launch grid ------------------------
# cof_chunk_floor() is blocks * threads, and the CUDA argument for it is sound:
# a slice of one record per thread is fully loaded, so subdividing costs
# nothing and small devices bound their launch duration for free.
#
# That argument breaks once the grid is sized from the work (see
# gen_bench_main.py). With blocks * threads >= CQ_FLUSH the floor equals the
# whole flush, so auto chunking could never subdivide again -- on ANY Apple
# GPU, by construction. Subdivision is not pointless there: a launch's duration
# is roughly (records / resident threads) x one ECM chain, and a 10-core part
# cannot hold 131,072 threads resident, so it runs them in waves and a full
# flush IS a long launch. Removing the ability to split it would silently
# retire the watchdog protection on exactly the slow parts that need it -- and
# this port has never run on an M1 at all (5.1a).
#
# So the floor keeps its own, core-derived notion of "fully loaded" instead of
# inheriting whatever grid the caller happens to launch. Set once at init.
_floor_old = (
    "static uint32_t cof_chunk_floor(int blocks, int threads)" + chr(10) +
    "{" + chr(10) +
    "    /* The width k_cofac actually launches at (cf_run_rounds clamps it to" + chr(10) +
    "     * COFAC_THREADS_MAX), or the floor would promise one record per thread on" + chr(10) +
    "     * threads that never run. */" + chr(10) +
    "    if (threads > COFAC_THREADS_MAX) threads = COFAC_THREADS_MAX;" + chr(10) +
    "    const uint64_t f = (uint64_t)(blocks > 0 ? blocks : 1)" + chr(10) +
    "                     * (uint64_t)(threads > 0 ? threads : 1);" + chr(10) +
    "    return f > 0xffffffffull ? 0xffffffffu : (uint32_t)f;" + chr(10) +
    "}")
_floor_new = (
    "/* Blocks' worth of records the DEVICE can keep loaded, independent of the" + chr(10) +
    " * grid actually launched. Zero until mtl_set_cof_floor_blocks() runs, in" + chr(10) +
    " * which case the floor falls back to the caller's grid -- the CUDA" + chr(10) +
    " * behaviour, and the safe direction if initialisation order ever moves. */" + chr(10) +
    "static int g_cof_floor_blocks;" + chr(10) +
    chr(10) +
    "void mtl_set_cof_floor_blocks(int b) { g_cof_floor_blocks = b > 0 ? b : 0; }" + chr(10) +
    chr(10) +
    "static uint32_t cof_chunk_floor(int blocks, int threads)" + chr(10) +
    "{" + chr(10) +
    "    /* Upstream's clamp, kept: k_cofac launches at COFAC_THREADS_MAX or" + chr(10) +
    "     * narrower, so a floor computed from a wider --threads would promise" + chr(10) +
    "     * records to threads that never run. */" + chr(10) +
    "    if (threads > COFAC_THREADS_MAX) threads = COFAC_THREADS_MAX;" + chr(10) +
    "    const int fb = g_cof_floor_blocks ? g_cof_floor_blocks : blocks;" + chr(10) +
    "    const uint64_t f = (uint64_t)(fb > 0 ? fb : 1)" + chr(10) +
    "                     * (uint64_t)(threads > 0 ? threads : 1);" + chr(10) +
    "    return f > 0xffffffffull ? 0xffffffffu : (uint32_t)f;" + chr(10) +
    "}")
assert _floor_old in src, 'cof_chunk_floor shape changed'
src = src.replace(_floor_old, _floor_new, 1)
print('  chunk floor decoupled from the launch grid')

# The queue capacity, for bench_main's grid sizing. An ACCESSOR, not a second
# #define: CQ_FLUSH has exactly one definition and the grid must track it.
src = src.replace(
    'static uint32_t cof_chunk_floor(int blocks, int threads)',
    'uint32_t mtl_cof_flush_capacity(void) { return CQ_FLUSH; }' + chr(10) + chr(10) +
    'static uint32_t cof_chunk_floor(int blocks, int threads)', 1)
print('  CQ_FLUSH exposed for grid sizing')

# ---- (b) measure a real per-launch duration, not a whole-side sum ---------
# The auto chunker halves whenever `stage > COF_CHUNK_TARGET_MS`, where stage
# is a SIDE's device time summed over every round and every slice. That is not
# the quantity the target names. A watchdog kills ONE LAUNCH, and on this
# hardware stage runs 39-45x the 250 ms target at every chunk size a 10-core
# part can produce -- so the test is true always, the loop parks at the floor
# forever, and an always-true test is not a controller (plan 8j).
#
# Measure the thing instead. Round 0, slice 0 is the longest launch in a flush:
# every record is still CF_INCOMPLETE, so it carries the most live work of any
# launch in the side. Bracket exactly that one, per side, and steer on the
# larger of the two. The events are read after cofq_flush's EXISTING
# mtlDeviceSynchronize(), so this adds no synchronisation -- only two event
# records on one launch per side per flush.
# ---- the peak-launch mechanism is UPSTREAM now ---------------------------
# cofac.cuh carries cof_peak_t, g_cof_peak, the round-0/slice-0 bracket in
# cf_run_rounds and the pk0/pk1 arming in cofq_flush, because the same field
# failure reached CUDA: a 980 Ti parked at cof_chunk_floor() and its launches
# exceeded the Windows TDR. This generator used to inject all of that and now
# inherits it -- portlib's cuda*->mtl* renames carry it across unchanged.
#
# What is still Metal-only, and stays here:
#   - the per-launch mtlStreamFlush, because the watchdog here judges a COMMAND
#     BUFFER and the stream would batch every round's launch into one (9z-k);
#   - steering the chunk on the measured launch rather than on `stage`. Upstream
#     added a one-way VALVE at COF_LAUNCH_TARGET_MS (1000 ms) and deliberately
#     left its `stage` test alone, so on Metal that test would still be
#     always-true and still park at the floor. Metal keeps its own proportional
#     bidirectional steering at COF_CHUNK_TARGET_MS; upstream's valve sits below
#     it as a backstop and, at 1000 ms against Metal's 400, should never fire.

_lines = src.split(chr(10))
_j = [k for k, l in enumerate(_lines) if 'cf_kname(L, 0, 0)' in l]
assert len(_j) == 1, 'rho launch not unique'
assert _lines[_j[0] + 1].strip() == '}', 'slice-loop else-branch shape changed'
_lines.insert(_j[0] + 2,
    "            /* ONE cofactor launch per command buffer. The chunker bounds a" + chr(10) +
    "             * LAUNCH at COF_CHUNK_TARGET_MS, but macOS's interactivity" + chr(10) +
    "             * watchdog judges a COMMAND BUFFER, and the stream batches" + chr(10) +
    "             * every round's launch into one: measured here, 50 rounds of a" + chr(10) +
    "             * 244 ms launch became a single 3,180 ms submission, 4x the" + chr(10) +
    "             * bound the chunker thought it was holding. Upstream's _peak" + chr(10) +
    "             * bracket hid it -- an event record commits, so the one launch" + chr(10) +
    "             * being MEASURED was the one launch not batched. See 9z-k." + chr(10) +
    "             */" + chr(10) +
    "            mtlStreamFlush(0);   /* cannot fail; sync() reports the work */")
src = chr(10).join(_lines)
print('  per-launch flush added to cf_run_rounds')

# Hoist upstream's launch_ms to function scope so the steering below can read
# it: upstream computes it inside its valve's own block.
_h_old = "    float t0 = 0, t1 = 0;"
_h_new = "    float t0 = 0, t1 = 0, launch_ms = 0;"
assert src.count(_h_old) == 1, 'cofq_flush declarations changed'
src = src.replace(_h_old, _h_new, 1)

# THE MEASUREMENT IS HOISTED OUT OF THE VALVE, and out of auto mode with it.
#
# Upstream computes launch_ms inside its `if (!chunk)` valve block, which sits
# BELOW the over-bound report this generator inserts further down. Inherited as
# it stands, that report read a launch_ms that was still zero -- dead code, and
# measured as such: a build with COF_CHUNK_TARGET_MS at 1 ms printed the valve's
# own line at 238 ms and this report not at all. 9c kept the report because an
# over-bound launch is the condition the bound exists for and is invisible from
# anywhere else; it went dead in the rebase that took upstream's valve.
#
# Deliberately not re-gated on auto mode either, for 9c's stated reason: a
# pinned --cof-chunk that overruns is worth more, not less, since nothing will
# adapt. The excised lines reappear above the report.
_v_old = chr(10).join([
    "        float pm0 = 0.0f, pm1 = 0.0f;",
    "        if (pk0.fired) COF_FLUSH_CK(mtlEventElapsedTime(&pm0, pk0.a, pk0.b));",
    "        if (pk1.fired) COF_FLUSH_CK(mtlEventElapsedTime(&pm1, pk1.a, pk1.b));",
    "        const float launch_ms = pm0 > pm1 ? pm0 : pm1;",
    ""])
assert src.count(_v_old) == 1, "upstream's valve no longer computes launch_ms"
src = src.replace(_v_old, "", 1)

# THE VALVE RECORDS A CEILING AND YIELDS THE FLUSH TO THE STEERING.
#
# Upstream's valve is a backstop for a floor this port does not have.
# cof_chunk_floor() here is one WAVE, not one grid (8j), so the steering below
# can already descend past the point the valve exists to reach. What the valve
# did instead was PRE-EMPT it: `valve_acted` skips the steering for that flush,
# so a launch over the 1000 ms valve bound got a correction aimed at 0.8x1000
# rather than at 0.8x this build's own MTL_INTERACTIVITY_BOUND_MS, and the
# no-progress guard did not run either.
#
# Measured, with the valve bound at 200 ms and the steering bound at 1 ms: two
# consecutive flushes -- about 134 special-q -- were steered by the valve at
# 0.67x per step while the build's primary bound wanted 0.33x. The cases where
# this fires are an M1 at its opening chunk, whose command buffers are then in
# the range macOS has actually been observed killing.
#
# So the valve keeps the one thing only it can say -- that a measured launch
# has disproved the floor -- and the steering does all the arithmetic.
# Upstream's action is three branches now -- descend, park having stopped
# paying, park at the floor -- and ALL THREE go, because on Metal the steering
# owns the descent and has its own park (chunk_parked). Spliced by index rather
# than matched as one literal, because it is 60 lines of upstream prose; the
# asserts below name what is being discarded, so a shape change cannot pass
# quietly. .index() raises on a miss, which is the assert for the bounds.
_va = src.index("            /* DID THE LAST DESCENT PAY?")
_vz = src.index(chr(10) + "        }" + chr(10) + "    }" + chr(10), _va)
_valve_old = src[_va:_vz]
assert 'parking at %u records/launch' in _valve_old, 'no-progress park missing'
assert 'parking here rather than giving' in _valve_old, 'floor park missing'
assert 'Q->chunk_ceiling = next;' in _valve_old, "the valve's ceiling missing"
assert _valve_old.count('valve_acted = 1;') == 3, 'not three valve branches'
_valve_new = chr(10).join([
    "            /* THE CEILING ONLY, AND ONLY EVER DOWNWARD. The steering below",
    "             * measures this same launch against a tighter bound and has a",
    "             * no-progress guard, so it -- not this -- picks the chunk; all",
    "             * that is needed from here is that the floor stop applying. */",
    "            if (!Q->chunk_ceiling || next < Q->chunk_ceiling) {",
    "                fprintf(stderr,",
    "                        \"  cofactor: kernel launch %.0f ms is over this build's\"",
    "                        \" %.0f ms valve bound; the chunk floor no longer holds\"",
    "                        \" above %u records/launch\\n\",",
    "                        (double)launch_ms, (double)COF_LAUNCH_TARGET_MS, next);",
    "                Q->chunk_ceiling = next;",
    "            }"])
src = src[:_va] + _valve_new + src[_vz:]

# ... and with the valve no longer claiming a flush, valve_acted has no reader.
_va_old = "    int valve_acted = 0;   /* declared here: COF_FLUSH_CK hides a goto done */" + chr(10)
assert src.count(_va_old) == 1, 'valve_acted declaration changed'
src = src.replace(_va_old, "", 1)
_vu_old = "    if (!chunk && !valve_acted) {"
assert src.count(_vu_old) == 1, 'the steering gate changed'
src = src.replace(_vu_old, "    if (!chunk) {", 1)
print('  the valve records a ceiling; the steering keeps the flush')

_s_old = "        const float stage = t0 + t1;"
_s_new = ("        /* The MEASURED longest launch, not t0+t1. The sum over every round" + chr(10) +
          "         * and slice of a side is not what a watchdog kills and, on a" + chr(10) +
          "         * 10-core Apple GPU, exceeds the target at every reachable chunk" + chr(10) +
          "         * size -- which made this loop park at its floor unconditionally." + chr(10) +
          "         * Upstream left this test alone on purpose (its valve is a" + chr(10) +
          "         * separate one-way path at a looser bound), so the substitution" + chr(10) +
          "         * still has to happen here. Falls back to the old sum if no" + chr(10) +
          "         * launch was timed, so a flush that never fired steers as before. */" + chr(10) +
          "        const float stage = launch_ms > 0.0f ? launch_ms : (t0 + t1);")
assert _s_old in src, 'steering input changed'
src = src.replace(_s_old, _s_new, 1)
print('  auto chunker steers on the measured launch, not the side sum')

# Report a launch only when it EXCEEDS the bound. The first version of this
# printed every new maximum, which in a healthy run is a stream of lines saying
# nothing is wrong -- noise in a volunteer's uploaded stderr.txt (9c). A launch
# over 750 ms is the opposite: it is the condition 8k set the bound for, the
# one a stalled compositor or a killed task would be explained by, and it is
# invisible from anywhere else.
#
# Still gated on a new maximum, so a device that simply cannot meet the bound
# emits a handful of lines and then goes quiet as the chunker parks, rather
# than one per flush for the whole band. Not gated on auto mode: a pinned
# --cof-chunk that overruns is exactly as worth knowing about, and more so,
# since nothing will adapt.
_rep_old = "    Q->ms_rat += t0; Q->ms_alg += t1;"
_rep_new = _rep_old + chr(10) + chr(10).join([
    "    {   /* The measured launch, hoisted out of the valve block below --",
    "         * which is where upstream computes it, BELOW this report and only",
    "         * in auto mode. Both are wrong for this line: it fired on a",
    "         * launch_ms that was still zero, and a pinned --cof-chunk that",
    "         * overruns is exactly the case worth reporting. */",
    "        float pm0 = 0.0f, pm1 = 0.0f;",
    "        if (pk0.fired) COF_FLUSH_CK(mtlEventElapsedTime(&pm0, pk0.a, pk0.b));",
    "        if (pk1.fired) COF_FLUSH_CK(mtlEventElapsedTime(&pm1, pk1.a, pk1.b));",
    "        launch_ms = pm0 > pm1 ? pm0 : pm1; }",
    "    if (launch_ms > COF_CHUNK_TARGET_MS && launch_ms > Q->ms_launch_max) {",
    "        Q->ms_launch_max = launch_ms;",
    "        fprintf(stderr,",
    "                \"  cofactor: kernel launch %.0f ms is over this build's\"",
    "                \" %.0f ms bound (%u records/launch, %u in flush)\\n\",",
    "                (double)launch_ms, (double)COF_CHUNK_TARGET_MS,",
    "                Q->chunk_cur < n ? Q->chunk_cur : n, n);",
    "    }"])
assert _rep_old in src, 'ms accumulation shape changed'
src = src.replace(_rep_old, _rep_new, 1)

# The high-water field, anchored on cofac.cuh's OWN timing declaration --
# something this generator does not add and cannot delete. 9c broke the
# ecm_rounds anchor precisely by hanging it off a field that was added here and
# later removed; nothing else may hang off this one.
_hw_old = "    double ms_rat, ms_alg, ms_host;"
assert _hw_old in src, 'cofq_t timing fields missing'
src = src.replace(_hw_old, _hw_old + chr(10) +
                  "    float  ms_launch_max;   /* longest OVER-BOUND launch seen, ms       */", 1)
print('  over-bound launches reported; in-bound ones are not')

# ---- (a), done properly: defend a launch-duration bound -------------------
# The target is a UI-responsiveness and watchdog bound on ONE kernel launch:
# 750 ms, chosen by the user, worth paying throughput for. With (b) the
# controller finally measures that quantity, so it can now defend it.
#
# Two things had to change for the bound to be reachable at all.
#
# 1. THE FLOOR. cof_chunk_floor() is a core-derived 15,360 records here, and
#    the control law clamps to it, so auto could never descend to the ~7,680
#    that a 750 ms launch needs. The floor's job is to stop the controller
#    subdividing into uselessly small launches; that job is done better by
#    measuring whether subdividing still helps (below), so the floor drops to
#    one full grid of records and the measurement takes over.
#
# 2. A NO-PROGRESS GUARD, because a bound can be UNREACHABLE. Measured here:
#    launch time falls with chunk size only until one record's ECM chain is
#    the whole launch, then it plateaus -- 3567, 1846, 1641, 1495, 1499, 1498
#    ms at chunks 15360 down to 480, flat below ~1920, while cofac/q explodes
#    from 465 to 4969. Against a target the chain cannot meet, an unguarded
#    controller halves forever and lands exactly there: the same unconditional
#    slowdown 8j removed, arrived at from the other direction. So a halving
#    that does not buy at least 10% is undone and the descent stops.
_g_old = "        if (stage > COF_CHUNK_TARGET_MS) {"
_g_new = chr(10).join([
    "        if (Q->chunk_parked) {",
    "            /* Descent already proved useless at this size; see below. */",
    "        } else if (stage > COF_CHUNK_TARGET_MS) {",
    "            /* Did the LAST halving actually shorten the launch? If not,",
    "             * the bound is below one ECM chain and no chunk can meet it.",
    "             * Undo that halving and stop, rather than subdividing down to",
    "             * the floor for nothing. */",
    "            if (Q->ms_launch_prev > 0.0f && Q->chunk_prev > Q->chunk_cur",
    "                && stage > Q->ms_launch_prev * 0.90f) {",
    "                cof_report_chunk(Q->chunk_prev, n, 0);",
    "                Q->chunk_cur = Q->chunk_prev;",
    "                Q->chunk_parked = 1;",
    "            } else {"])
assert _g_old in src, 'steering halve branch shape changed'
src = src.replace(_g_old, _g_new, 1)

_h_old = chr(10).join([
    "            const uint32_t half = Q->chunk_cur / 2;",
    "            Q->chunk_cur = (half > floor_ch) ? half : floor_ch;",
    "        } else if"])
_h_new = chr(10).join([
    "                const uint32_t half = Q->chunk_cur / 2;",
    "                Q->ms_launch_prev = stage; Q->chunk_prev = Q->chunk_cur;",
    "                Q->chunk_cur = (half > floor_ch) ? half : floor_ch;",
    "            }",
    "        } else if"])
assert _h_old in src, 'halve body shape changed'
src = src.replace(_h_old, _h_new, 1)

# Anchored on cofac.cuh's own timing fields. This used to hang off the
# ms_launch_max field inserted above, which no longer exists -- the
# longest-launch REPORT was dropped, and with it the high-water field nothing
# else read. These three are the chunker's state and are still needed.
_f_old = "    double ms_rat, ms_alg, ms_host;"
_f_new = chr(10).join([
    _f_old,
    "    float  ms_launch_prev;  /* launch time before the last halving      */",
    "    uint32_t chunk_prev;    /* the chunk that produced it               */",
    "    int    chunk_parked;    /* halving stopped paying; descend no more  */"])
assert _f_old in src, 'cofq_t launch field missing'
src = src.replace(_f_old, _f_new, 1)

_t_old = "#define COF_CHUNK_TARGET_MS   250.0f"
assert _t_old in src, 'target shape changed'
src = src.replace(_t_old, chr(10).join([
    "/* The port's interactivity bound, defined ONCE in metal_rt.h and shared",
    " * with the fbgen root finder, which bounds itself against the same number.",
    " * Two constants that must agree is how this port has repeatedly hurt",
    " * itself; see MTL_INTERACTIVITY_BOUND_MS for the value and its evidence.",
    " *",
    " * A UI-responsiveness bound as much as a watchdog one, and explicitly",
    " * worth throughput to hold. CUDA keeps 250 ms against",
    " * a whole-side sum; this is 400 against a measured launch, so the two",
    " * numbers are not comparable. See gen_cofac_host.py.",
    " *",
    " * WAS 750, lowered 2026-09-16 after 9z-k. 750 was set (8k) against what",
    " * was believed to be a launch bound but was in fact a bound on one",
    " * DISPATCH, while macOS kills a COMMAND BUFFER -- and the observed kills",
    " * on M1/M2 were command buffers of roughly 800 ms. 9z-k makes one launch",
    " * one command buffer, so this number is now the real thing the watchdog",
    " * sees, and 400 puts it at half the shortest duration anyone has been",
    " * killed at rather than a hair under it. The true threshold is still",
    " * undocumented and unmeasured. */",
    "#ifndef COF_CHUNK_TARGET_MS",
    "#define COF_CHUNK_TARGET_MS   MTL_INTERACTIVITY_BOUND_MS",
    "#endif"]), 1)

# The floor stops being the policy and becomes a sanity bound: one grid's
# worth of records, so a slice never has fewer records than the grid can hold
# in a single wave.
# The opening value must NOT be the floor any more. It was the same number
# when the floor was a full grid; now that the floor is a wave, opening there
# would start every band at a slice far too small to be efficient and, with
# one flush, never adapt at all -- measured at 722 ms/q against 355 for the
# right slice. Open at ONE GRID of records, which is what the floor used to
# be, and let the measurement walk it down toward the bound.
_op_old = "                            : (Q->chunk_cur ? Q->chunk_cur : floor_ch);"
_op_new = "                            : (Q->chunk_cur ? Q->chunk_cur : cof_chunk_open(blocks, threads));"
assert _op_old in src, 'auto opening value shape changed'
src = src.replace(_op_old, _op_new, 1)

_od_old = "static uint32_t cof_chunk_floor(int blocks, int threads)"
_od_new = chr(10).join([
    "/* Where auto STARTS: one core-derived grid of records, one per thread.",
    " * The floor below is where it may descend TO, which is a different and",
    " * now much smaller number. */",
    "static uint32_t cof_chunk_open(int blocks, int threads)",
    "{",
    "    const int ob = g_cof_floor_blocks ? g_cof_floor_blocks : blocks;",
    "    const uint64_t f = (uint64_t)(ob > 0 ? ob : 1)",
    "                     * (uint64_t)(threads > 0 ? threads : 1);",
    "    return f > 0xffffffffull ? 0xffffffffu : (uint32_t)f;",
    "}",
    "",
    _od_old])
assert _od_old in src, 'cof_chunk_floor declaration missing'
src = src.replace(_od_old, _od_new, 1)
print('  auto opens at one grid, descends toward the bound by measurement')

_fl_old = "    const int fb = g_cof_floor_blocks ? g_cof_floor_blocks : blocks;"
_fl_new = chr(10).join([
    "    /* One WAVE, not one grid: the measurement decides how far to descend",
    "     * (see the no-progress guard), and this only stops the controller",
    "     * asking for fewer records than the device can run at once. */",
    "    const int fbb = g_cof_floor_blocks ? g_cof_floor_blocks : blocks;",
    "    const int fb = fbb > 8 ? fbb / 8 : 1;"])
assert _fl_old in src, 'chunk floor shape changed'
src = src.replace(_fl_old, _fl_new, 1)
print('  target 400 ms on a measured launch; floor lowered; no-progress guard added')

# ---- proportional steering, because the response is LINEAR ----------------
# Measured at a FULL CQ_FLUSH batch (plan 9z-h), which is the regime every
# earlier measurement missed -- 8h and 8i both sampled single-q runs of ~1,852
# records, where the grid is so oversubscribed that the launch is chain-bound
# and the chunk does nothing. At a real 130k-record flush the launch is
# RECORD-bound and very nearly linear in the chunk:
#
#     15,360 -> 242 ms    30,720 -> 483 ms
#     61,440 -> 976 ms   131,072 -> 1713 ms      (~15.8 us/record, this M3)
#
# Against a linear response, halving is the wrong step. It overshoots to half
# the bound or less, the `stage < target/4` branch below then doubles straight
# back, and the controller oscillates instead of converging -- which is
# exactly what the M4 Max field log shows: 61440 -> 30720 -> 61440 within
# three flushes, having measured 4802 ms once.
#
# A proportional step lands in ONE flush and cannot flip-flop: aim at 0.8x the
# bound, so the result sits inside the dead band rather than on its edge.
# The no-progress guard above is untouched and still catches the case the
# proportional model cannot see -- a bound below one ECM chain, where the
# launch does not shrink with the chunk at all.
_ps_old = chr(10).join([
    "            } else {",
    "                const uint32_t half = Q->chunk_cur / 2;",
    "                Q->ms_launch_prev = stage; Q->chunk_prev = Q->chunk_cur;",
    "                Q->chunk_cur = (half > floor_ch) ? half : floor_ch;",
    "            }"])
_ps_new = chr(10).join([
    "            } else {",
    "                /* Proportional, not halved: the response is linear in the",
    "                 * chunk at a full flush, so aim straight at 0.8x the bound",
    "                 * and land inside the dead band. Halving overshoots and the",
    "                 * doubling branch below flips it back (plan 9z-h). */",
    "                double want = (double)Q->chunk_cur",
    "                            * ((double)COF_CHUNK_TARGET_MS * 0.8) / (double)stage;",
    "                uint32_t next = want < 1.0 ? 1u : (uint32_t)want;",
    "                /* Always make progress downward, however bad the estimate. */",
    "                if (next >= Q->chunk_cur) next = Q->chunk_cur / 2;",
    "                if (next < floor_ch) next = floor_ch;",
    "                Q->ms_launch_prev = stage; Q->chunk_prev = Q->chunk_cur;",
    "                Q->chunk_cur = next;",
    "            }"])
assert _ps_old in src, 'halving branch shape changed'
src = src.replace(_ps_old, _ps_new, 1)

_pu_old = chr(10).join([
    "            Q->chunk_cur = (Q->chunk_cur > Q->cap / 2) ? Q->cap",
    "                                                       : Q->chunk_cur * 2;"])
_pu_new = chr(10).join([
    "            /* Same reasoning upward, and the same 0.8x aim point, so a",
    "             * flush that comes in far under the bound climbs to the right",
    "             * size in one step instead of doubling toward it. */",
    "            double want = (double)Q->chunk_cur",
    "                        * ((double)COF_CHUNK_TARGET_MS * 0.8) / (double)stage;",
    "            uint64_t next = want < 1.0 ? 1ull : (uint64_t)want;",
    "            if (next <= Q->chunk_cur) next = (uint64_t)Q->chunk_cur * 2ull;",
    "            if (next > Q->cap) next = Q->cap;",
    "            Q->chunk_cur = (uint32_t)next;"])
assert _pu_old in src, 'doubling branch shape changed'
src = src.replace(_pu_old, _pu_new, 1)
print('  chunk steering is proportional, not halve/double')


# ---- refuse a launch this port cannot bound -------------------------------
# The block above warns when stage 2 grows past what --cof-chunk can divide,
# and says outright that the one configuration known to trip a watchdog is
# --ecm-b1 400000, and that cofcheck.sh skips that case on HIP. This port did
# not skip it, and it took the machine down TWICE -- a WindowServer crash and
# userspace watchdog timeout, both ~1m47s into cofcheckgate, at exactly that
# case, on an otherwise idle Mac. On Apple silicon the GPU also drives the
# display, so a launch that a discrete card merely fails is one the compositor
# does not survive. A warning is not enough here.
#
# The cost of one curve is very close to linear in the work it does. Measured
# on a 10-core M3 at one curve, one round, B2 = 30*B1:
#
#     B1     2000     8000    32000
#     ms      110      370     1461
#
# which is ~0.045 ms per (prime power + giant step) -- Q->ns + Q->s2nv -- and
# predicts ~19 s per curve at B1 400000, so ~5 MINUTES for that case's default
# 16 curves in a single launch. That is what the two crashes were.
#
# REFUSE rather than silently reducing --ecm-curves: curves-per-round chooses
# which sigmas run (`sigma = c0*1000 + cv + 6`, see the record-axis comment
# above), so quietly changing it would change which numbers factor. Tell the
# caller instead, and name the knobs.
_guard_anchor = "    {\n        static const char *nm[2] = { \"rho\", \"ECM\" };"
_guard = chr(10).join([
    "    if (Q->meth[0] || Q->meth[1]) {",
    "        /* ~0.045 ms per prime power + giant step, 10-core M3. A slower or",
    "         * faster Apple GPU moves this; it is a guard rail, not a model. */",
    "        /* MARGINAL cost of a curve: 0.0413 ms per prime power + giant",
    "         * step. The one-curve points (110/370/1461 ms at B1 2000/8000/",
    "         * 32000) include a fixed per-launch overhead and so overstate",
    "         * it; 8 curves at B1 2000 measured 740 ms, i.e. 92.5 ms each",
    "         * over 2,237 units. Calibrated on a 10-core M3. */",
    "        const double ms_one_curve = 0.0413 * (double)(Q->ns + Q->s2nv);",
    "        if (ms_one_curve > COF_LAUNCH_REFUSE_MS) {",
    "            fprintf(stderr,",
    "                    \"  cofactor queue: REFUSED -- one ECM curve is about\"",
    "                    \" %.0f ms of work (%u prime powers + %u giant steps),\"",
    "                    \" and a launch cannot be made shorter than one curve:\"",
    "                    \" --cof-chunk splits RECORDS, never the chain. At\"",
    "                    \" %u curves/round that is one kernel launch of roughly\"",
    "                    \" %.0f s.\\n\",",
    "                    ms_one_curve, Q->ns, Q->s2nv, Q->ecm_curves,",
    "                    ms_one_curve * (double)Q->ecm_curves / 1000.0);",
    "            fprintf(stderr,",
    "                    \"  On Apple silicon the GPU also drives the display,\"",
    "                    \" so this does not merely fail the task -- it hangs\"",
    "                    \" WindowServer and takes the session down. Observed\"",
    "                    \" twice at --ecm-b1 400000. Lower --ecm-b1/--ecm-b2,\"",
    "                    \" or cut --ecm-curves and raise --cof-rounds to keep\"",
    "                    \" the same curve budget in shorter launches.\\n\");",
    "            goto done;",
    "        }",
    "    }",
    ""])
assert _guard_anchor in src, 'cofactor method banner shape changed'
src = src.replace(_guard_anchor, _guard + _guard_anchor, 1)

_c_old = "#define COF_S2NV_WATCHDOG_WARN  20000u"
_c_new = chr(10).join([
    _c_old,
    "",
    "/* Refuse above this much estimated work in ONE curve, because no chunking",
    " * can divide a chain. 10 s: launches of 3.6 s and 6.9 s have both run",
    " * here without incident, and the case that crashed the machine twice",
    " * estimates at ~19 s per curve. Between those, closer to what is known to",
    " * work than to what is known to kill it. */",
    "#define COF_LAUNCH_REFUSE_MS  10000.0"])
assert _c_old in src, 'watchdog warn constant missing'
src = src.replace(_c_old, _c_new, 1)
print('  refuses an ECM chain no chunk size can bound')

# ---- derive curves-per-round from the launch bound ------------------------
# A fixed (curves, rounds) pair is only right for one B1: one curve costs
# ~0.0413 ms per prime power + giant step, so the number that fits 8k's bound
# falls as B1 grows -- ~8 curves at B1 2000, ~2 at B1 8000 (8m). Rather than
# pin a pair that is correct for c183 and progressively wrong elsewhere, derive
# it, and raise rounds to keep the caller's TOTAL curve budget intact.
#
# Only when --ecm-curves was NOT given: an explicit curve count is the user
# choosing which sigmas run, and this does not overrule it -- it advises
# instead. Only when BOTH sides are ECM, because cofq_flush passes one round
# count to both and raising it under rho would walk into the `budget << r`
# overflow that 8m showed is checked against the caller's rounds, not ours.
_adv_old = "        if (ms_one_curve > COF_LAUNCH_REFUSE_MS) {"
_adv_new = chr(10).join([
    "        const int both_ecm = Q->meth[0] && Q->meth[1];",
    "        /* No advisory branch: when this cannot act -- an explicit",
    "         * --ecm-curves, or a side running rho -- it stays silent rather",
    "         * than printing guidance into a volunteer's stderr.txt. */",
    "        if (ms_one_curve <= COF_LAUNCH_REFUSE_MS && Q->ecm_curves > 2u",
    "            && !curves_set && both_ecm) {",
    "            /* NOT the largest count that fits the bound -- that is 8 at",
    "             * B1 2000 and costs 265 ms/q against 180. Measured at --nq 144,",
    "             * 192 curves, B1 2000/B2 60000, cofac ms/q by curves/round:",
    "             *   16c 353.5  8c 265.3  6c 252.3  4c 212.5  3c 192.8",
    "             *   2c 180.0   1c 181.0",
    "             * A bracketed interior minimum at TWO. Every round re-compacts",
    "             * the live list, so splitting the budget finely drops records",
    "             * that have already split before the expensive rounds run; at",
    "             * one curve the five per-round kernels finally cost more than",
    "             * that saves. So aim at 2 and let the bound lower it further. */",
    "            uint32_t fit = 2u;",
    "            while (fit > 1u && ms_one_curve * (double)fit > COF_CHUNK_TARGET_MS)",
    "                fit--;",
    "            if (fit > Q->ecm_curves) fit = Q->ecm_curves;",
    "            const uint64_t budget = (uint64_t)Q->ecm_curves * rounds_in;",
    "            uint64_t r = (budget + fit - 1u) / fit;",
    "            if (r > 1000u) r = 1000u;",
    "            if (r < 1u) r = 1u;",
    "            /* Says what it DID. The line it replaces asserted that the",
    "             * caller's split was 'over the bound', which is false",
    "             * whenever a curve is cheap -- at B1 200 twelve curves are",
    "             * 119 ms against a 750 ms bound, and it said so anyway. The",
    "             * reason to move is that 2/round is the measured optimum;",
    "             * the bound is a ceiling that can lower it further, not the",
    "             * thing being enforced here. */",
    "            printf(\"  cofactor queue: %u curves/round -> %u x %u\"",
    "                   \" (~%.0f ms/launch, %llu curves vs %llu)\\n\",",
    "                   Q->ecm_curves, fit, (unsigned)r,",
    "                   ms_one_curve * (double)fit,",
    "                   (unsigned long long)((uint64_t)fit * r),",
    "                   (unsigned long long)budget);",
    "            Q->ecm_curves = fit;",
    "            Q->ecm_rounds = (uint32_t)r;",
    "        }",
    "        if (ms_one_curve > COF_LAUNCH_REFUSE_MS) {"])
assert _adv_old in src, 'refusal guard shape changed'
src = src.replace(_adv_old, _adv_new, 1)

# The derived round count, and the signature that carries the inputs in.
_sig_old = ("static int cofq_init(cofq_t *Q, cofq_out_t *O, uint32_t cap," + chr(10) +
            "                     int meth0, int meth1, uint32_t ecm_b1, uint32_t ecm_b2," + chr(10) +
            "                     uint32_t ecm_curves, int limbs0, int limbs1)")
_sig_new = ("static int cofq_init(cofq_t *Q, cofq_out_t *O, uint32_t cap," + chr(10) +
            "                     int meth0, int meth1, uint32_t ecm_b1, uint32_t ecm_b2," + chr(10) +
            "                     uint32_t ecm_curves, int limbs0, int limbs1," + chr(10) +
            "                     int curves_set, uint32_t rounds_in)")
assert _sig_old in src, 'cofq_init signature changed'
src = src.replace(_sig_old, _sig_new, 1)

# Anchored on the chunker state inserted above, not on the dropped
# ms_launch_max field -- and ASSERTED. This replace carried no assert, so when
# its anchor went away it silently did nothing and left Q->ecm_rounds assigned
# but never declared. That is a compile error two functions later, and the
# generator reported success.
_fld_old = "    int    chunk_parked;    /* halving stopped paying; descend no more  */"
assert _fld_old in src, 'chunker state fields missing'
src = src.replace(_fld_old, _fld_old + chr(10) +
                  "    uint32_t ecm_rounds;    /* rounds to actually run; derived or the caller's */", 1)

_init_old = "    Q->meth[0] = meth0; Q->meth[1] = meth1; Q->ecm_curves = ecm_curves;"
assert _init_old in src, 'cofq_init field init changed'
src = src.replace(_init_old, _init_old + chr(10) +
                  "    Q->ecm_rounds = rounds_in ? rounds_in : 1u;", 1)
print('  curves/round derived from the launch bound when --ecm-curves is unset')


# ---- report a change in what is PRINTED, not in the internal slice --------
# cof_report_chunk's own comment says "only on change, never per flush", and
# it was defeated by its own arithmetic: step is min(chunk, n), so when the
# old and new chunk BOTH exceed n the internal value changes and the rendered
# line does not. One q printed it twice, verbatim -- the opening choice, then
# the doubling that followed a 93 ms flush. Compare what goes out.
_rc_old = chr(10).join([
    "    fprintf(stderr, \"  cofactor chunk: %u records/launch, %u launch%s per\"",
    "            \" round over %u records (%s)\\n\", step, nl, nl == 1 ? \"\" : \"es\", n,",
    "            pinned ? \"pinned by --cof-chunk\" : \"auto\");"])
_rc_new = chr(10).join([
    "    static uint32_t last_step, last_nl, last_n;",
    "    static int last_pinned = -1;",
    "    if (step == last_step && nl == last_nl && n == last_n",
    "        && pinned == last_pinned) return;",
    "    last_step = step; last_nl = nl; last_n = n; last_pinned = pinned;",
    _rc_old])
assert _rc_old in src, 'cof_report_chunk shape changed'
src = src.replace(_rc_old, _rc_new, 1)
print('  chunk report de-duplicated on rendered content')

open(OUT, 'w').write(src)
assert nk == 9, ('removed %d __global__ definitions, expected 9 -- a kernel '
                 'shape changed upstream and one was left in the HOST file' % nk)
print('wrote %s: %d kernels removed, %d launches rewritten' % (OUT, nk, nl))
if skipped:
    print('  launches of kernels NOT yet in metal/cofac.metal:', ' '.join(sorted(skipped)))


# The same body without the standalone prelude, for the combined host TU
# (bench_host.cpp), which supplies those includes itself -- mirroring the way
# bench_kernels.cu includes cofac.cuh.
_marker = '#ifndef CUDA_SIEVE_COFAC_CUH'
_body = src[src.index(_marker):]
open('bench/metal/cofac_host.inc', 'w').write(
    '/* cofac.cuh host half, for inclusion by bench_host.cpp. Generated by\n'
    ' * metal/gen_cofac_host.py; cofac_metal.cpp is the standalone form used by\n'
    ' * the Phase 6a run_cofac gate. */\n' + HELPER + chr(10) + _body)
print('wrote bench/metal/cofac_host.inc')
