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
        m = re.search(r'(?:template\s*<[^>]*>\s*)?__global__\s+void\s+\w+\s*\(', text[i:])
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
_peak_decl = (
    "/* Bracket for the longest launch in a flush: round 0, slice 0, where every" + chr(10) +
    " * record is still live. cf_run_rounds records it; cofq_flush reads it after" + chr(10) +
    " * the device sync it already performs, so no extra stall is introduced." + chr(10) +
    " * A file-scope pointer rather than another parameter on the template and its" + chr(10) +
    " * dispatcher, both of which are threaded through by width. */" + chr(10) +
    "typedef struct { mtlEvent_t a, b; int armed, fired; } cof_peak_t;" + chr(10) +
    "static cof_peak_t *g_cof_peak;" + chr(10) + chr(10) +
    "template <int L>" + chr(10) +
    "static void cf_run_rounds(")
assert src.count("template <int L>" + chr(10) + "static void cf_run_rounds(") == 1, 'cf_run_rounds shape changed'
src = src.replace("template <int L>" + chr(10) + "static void cf_run_rounds(", _peak_decl, 1)

_lines = src.split(chr(10))
_open = "            const uint32_t e = (n - b > step) ? b + step : n;"
_i = [k for k, l in enumerate(_lines) if l == _open]
assert len(_i) == 1, 'slice-loop head not unique'
_lines.insert(_i[0] + 1,
    "            /* The flush's worst-case launch; see g_cof_peak. */" + chr(10) +
    "            const int _peak = (r == 0 && b == 0 && g_cof_peak" + chr(10) +
    "                               && g_cof_peak->armed);" + chr(10) +
    "            if (_peak) mtlEventRecord(g_cof_peak->a);")

_j = [k for k, l in enumerate(_lines) if 'cf_kname(L, 0, 0)' in l]
assert len(_j) == 1, 'rho launch not unique'
assert _lines[_j[0] + 1].strip() == '}', 'slice-loop else-branch shape changed'
_lines.insert(_j[0] + 2,
    "            if (_peak) {" + chr(10) +
    "                mtlEventRecord(g_cof_peak->b);" + chr(10) +
    "                g_cof_peak->armed = 0; g_cof_peak->fired = 1;" + chr(10) +
    "            }")
src = chr(10).join(_lines)
print('  per-launch bracket added to cf_run_rounds')

# cofq_flush: own the peak events, arm one per side, steer on the larger.
_d_old = "    mtlEvent_t e0 = NULL, e1 = NULL, e2 = NULL;" + chr(10) + "    float t0 = 0, t1 = 0;"
_d_new = ("    mtlEvent_t e0 = NULL, e1 = NULL, e2 = NULL;" + chr(10) +
          "    cof_peak_t pk0 = {NULL, NULL, 0, 0}, pk1 = {NULL, NULL, 0, 0};" + chr(10) +
          "    float t0 = 0, t1 = 0, launch_ms = 0;")
assert _d_old in src, 'cofq_flush declarations changed'
src = src.replace(_d_old, _d_new, 1)

_c_old = ("    COF_FLUSH_CK(mtlEventCreate(&e0));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&e1));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&e2));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventRecord(e0));")
_c_new = ("    COF_FLUSH_CK(mtlEventCreate(&e0));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&e1));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&e2));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&pk0.a)); COF_FLUSH_CK(mtlEventCreate(&pk0.b));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventCreate(&pk1.a)); COF_FLUSH_CK(mtlEventCreate(&pk1.b));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventRecord(e0));" + chr(10) +
          "    pk0.armed = 1; g_cof_peak = &pk0;")
assert _c_old in src, 'cofq_flush event creation changed'
src = src.replace(_c_old, _c_new, 1)

_a_old = "    MTL_LAUNCH(k_cof_gate, blocks, threads, 0, 0, n, Q->d_st0, Q->d_st1);"
_a_new = ("    g_cof_peak = NULL;" + chr(10) + _a_old + chr(10) +
          "    pk1.armed = 1; g_cof_peak = &pk1;")
assert src.count(_a_old) == 1, 'k_cof_gate launch not unique'
src = src.replace(_a_old, _a_new, 1)

_r_old = ("    COF_FLUSH_CK(mtlEventElapsedTime(&t0, e0, e1));" + chr(10) +
          "    COF_FLUSH_CK(mtlEventElapsedTime(&t1, e1, e2));")
_r_new = (_r_old + chr(10) +
          "    g_cof_peak = NULL;" + chr(10) +
          "    /* The longest single launch this flush actually ran, which is what" + chr(10) +
          "     * COF_CHUNK_TARGET_MS is about. A side with no live records never" + chr(10) +
          "     * fires, and contributes nothing. */" + chr(10) +
          "    {" + chr(10) +
          "        float p = 0;" + chr(10) +
          "        if (pk0.fired && !mtlEventElapsedTime(&p, pk0.a, pk0.b)" + chr(10) +
          "            && p > launch_ms) launch_ms = p;" + chr(10) +
          "        if (pk1.fired && !mtlEventElapsedTime(&p, pk1.a, pk1.b)" + chr(10) +
          "            && p > launch_ms) launch_ms = p;" + chr(10) +
          "    }")
assert _r_old in src, 'elapsed-time reads changed'
src = src.replace(_r_old, _r_new, 1)

_s_old = "        const float stage = t0 + t1;"
_s_new = ("        /* The MEASURED longest launch, not t0+t1. The sum over every round" + chr(10) +
          "         * and slice of a side is not what a watchdog kills and, on a" + chr(10) +
          "         * 10-core Apple GPU, exceeds the target at every reachable chunk" + chr(10) +
          "         * size -- which made this loop park at its floor unconditionally." + chr(10) +
          "         * Falls back to the old sum if no launch was timed, so a flush" + chr(10) +
          "         * that never fired steers exactly as it used to. */" + chr(10) +
          "        const float stage = launch_ms > 0.0f ? launch_ms : (t0 + t1);")
assert _s_old in src, 'steering input changed'
src = src.replace(_s_old, _s_new, 1)

# cofq_flush's OWN cleanup label -- the first "done:" in the file belongs to a
# different function, and destroying pk0/pk1 there does not compile.
_k = src.index("const float stage = launch_ms > 0.0f")
_d = src.index(chr(10) + "done:", _k)
src = (src[:_d] + chr(10) + "done:" + chr(10) +
       "    g_cof_peak = NULL;" + chr(10) +
       "    if (pk0.a) mtlEventDestroy(pk0.a);" + chr(10) +
       "    if (pk0.b) mtlEventDestroy(pk0.b);" + chr(10) +
       "    if (pk1.a) mtlEventDestroy(pk1.a);" + chr(10) +
       "    if (pk1.b) mtlEventDestroy(pk1.b);" +
       src[_d + len(chr(10) + "done:"):])
print('  auto chunker now steers on a measured launch duration')

open(OUT, 'w').write(src)
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
