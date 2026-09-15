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
