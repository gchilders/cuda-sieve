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
           'k_scan_pass1', 'k_scan_pass2', 'k_scan_pass3'}

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

/* td.cuh puts these inside its own __CUDACC__ guard, so a HOST translation
 * unit does not see them. Values must track td.cuh (TD_SCAN_BLK at :365,
 * TD_FMAX at :589).
 *
 * Forking td.cuh for the DEVICE side (metal/td_msl.h) did not remove this:
 * that header is MSL. Removing it properly means either lifting these two
 * defines above td.cuh's guard -- a CUDA-side change, and the cleanest fix --
 * or giving the host a generated companion to td_msl.h. Left duplicated and
 * flagged rather than quietly forgotten. */
#ifndef TD_SCAN_BLK
#define TD_SCAN_BLK 256
#endif
#ifndef TD_FMAX
#define TD_FMAX 64
#endif

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
src = HELPER + chr(10) + src

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
