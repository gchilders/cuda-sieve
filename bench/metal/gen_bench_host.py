#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces bench_host.cpp from
# bench_kernels.cu's HOST half -- the TU that, in the CUDA build, also pulls in
# td.cuh, cofac.cuh and pipeline.cuh. Same shape here, with the ported
# versions.
import re, sys
sys.path.insert(0, 'bench/metal')
from portlib import rewrite_launches, apply_renames

SRC = 'bench/bench_kernels.cu'
OUT = 'bench/metal/bench_host.cpp'
lines = open(SRC).read().split('\n')

def upto_close(start):
    d = 0; seen = False
    for i in range(start, len(lines)):
        d += lines[i].count('{') - lines[i].count('}')
        if '{' in lines[i]: seen = True
        if seen and d == 0: return i
    raise SystemExit('unterminated at %d' % start)

def find(pat, after=0):
    for i in range(after, len(lines)):
        if re.search(pat, lines[i]): return i
    raise SystemExit('not found: ' + pat)

# Device spans to remove; the host half is everything else.
spans = []
a = find(r'^template <bool SLABBED>')
spans.append((a, upto_close(find(r'void k_fill_l2\(', a))))
a = find(r'^static inline uint32_t bgcd\(|^__device__ __forceinline__ uint32_t bgcd\(')
spans.append((a, upto_close(find(r'^__global__ void k_intersect_compact', a))))
a = find(r'^__global__ void k_verify_td_mod_cases')
spans.append((a, upto_close(a)))

# build_slices_b is host code that happens to sit between two kernels, so the
# device-span removal would take it with them. Lift it out and re-insert.
bs_a = find(r'^static uint32_t build_slices_b\(')
bs_b = upto_close(bs_a)
build_slices = '\n'.join(lines[bs_a:bs_b + 1])

keep, prev = [], 0
for (s0, e0) in sorted(spans):
    keep += lines[prev:s0]
    keep.append('/* device code: see metal/bench_kernels.metal and metal/td.metal */')
    prev = e0 + 1
keep += lines[prev:]
src = '\n'.join(keep)
# put build_slices_b back, just before its first use
src = src.replace('/* device code: see metal/bench_kernels.metal and metal/td.metal */',
                  build_slices + '\n\n/* device code: see metal/bench_kernels.metal and metal/td.metal */',
                  1) if 'build_slices_b' not in src.split('static uint32_t build_slices_b')[0] else src
if 'static uint32_t build_slices_b' not in src:
    src = src.replace('/* device code: see metal/bench_kernels.metal and metal/td.metal */',
                      build_slices + '\n\n/* device code: see metal/bench_kernels.metal and metal/td.metal */', 1)

src, nl = rewrite_launches(src)
# The apply threadgroup width.
ATHR_NOTE = "/* 192, not CUDA's 512. MEASURED on this box, bracketed interior minimum,\n * three runs per point at logI 14 / J 8192 / region 13 on oracle/c183:\n *\n *   threads   64      128     192     256     512\n *   apply    83.9    56.2    54.6    57.8    77.4  ms\n *\n * Run-to-run spread is under 1%% at each point, so the 29.5%% gap between 192\n * and CUDA's 512 is far outside the noise. 192 is 6 SIMD groups and keeps\n * (athr & 31) == 0, which k_apply's warp-ballot path requires.\n *\n * THE BOX: a 10-core M3 in a fanless MacBook Air that also drives the\n * display. The shape of the curve should carry to other Apple GPUs; the\n * exact optimum may not, and an M3 Max has four times the cores. Re-measure\n * there rather than trusting this number. */\n"
n_athr = src.count('cfg->apply_threads ? cfg->apply_threads : 512;')
src = src.replace('cfg->apply_threads ? cfg->apply_threads : 512;',
                  'cfg->apply_threads ? cfg->apply_threads : 192;')
if n_athr:
    src = src.replace('    const int athr =', ATHR_NOTE + '    const int athr =', 1)
    src = src.replace('            int athr =', ATHR_NOTE + '            int athr =', 1)
    print('  apply threads default 512 -> 192 (%d site)' % n_athr)

src = apply_renames(src)

# the CUDA TU's own includes become the ported ones
src = src.replace('#include <cuda_runtime.h>', '#include "metal/metal_rt.h"\n#include "metal/td_host.h"')
src = src.replace('#include "cofac.cuh"', '#include "cofac_host.inc"')
src = src.replace('#include "pipeline.cuh"', '#include "pipeline_host.inc"')

# error-check helper, renamed with everything else
src = src.replace('static int cuda_check_impl', 'static int mtl_check_impl')
src = src.replace('cuda_check_impl(', 'mtl_check_impl(')

# cuda_optin_smem_limit queries the device ceiling; on Metal that is
# maxThreadgroupMemoryLength, which mtlGetDeviceProperties reports.
src = re.sub(r'static int cuda_optin_smem_limit\(size_t \*out\)\s*\{.*?\n\}\n',
'''static int mtl_optin_smem_limit(size_t *out)
{
    /* CUDA asks for cudaDevAttrMaxSharedMemoryPerBlockOptin. Metal's
     * equivalent is maxThreadgroupMemoryLength, and there is no separate
     * opt-in tier: 32 KB on Apple silicon against CUDA's ~100 KB, which is
     * what forces log_region <= 13 (plan section 5.1). */
    mtlDeviceProp p;
    if (mtlGetDeviceProperties(&p, 0) != mtlSuccess) return -1;
    *out = p.sharedMemPerBlock;
    return 0;
}
''', src, flags=re.S)

# Constants that live in the DEVICE region this file just removed, but which
# the host half reads (launch shapes, buffer sizing). Lifted by name from
# bench_kernels.cu so they track the original instead of being hand-copied --
# the same treatment td_host.h gives td.cuh's guarded bounds.
LIFT = ('SS_BLOCK_CUT', 'SS_WARP_CUT', 'L1_NBUF', 'L1_CAP', 'L2_NBUF', 'L2_CAP')
orig = open(SRC).read()
picked = []
for nm in LIFT:
    m = re.search(r'^#define\s+' + nm + r'\b[^\n]*$', orig, re.M)
    if m: picked.append(m.group(0))
if picked:
    anchor = '#include "pipeline_host.inc"'
    block = ('\n/* Launch-shape constants defined inside the device region above, which\n'
             ' * the host half reads. Lifted by name from bench_kernels.cu. */\n'
             + '\n'.join(picked) + '\n')
    # near the TOP: the host reads these long before the includes at the end
    src = src.replace('#include "metal/td_host.h"',
                      '#include "metal/td_host.h"\n' + block, 1)
    print('  lifted %d device-region #define(s) the host reads' % len(picked))

# cudaDeviceGetAttribute(&v, cudaDevAttrMaxGridDimX, dev) -> the shim reports
# the same thing through mtlGetDeviceProperties.
src = re.sub(r'mtl_check_impl\(cudaDeviceGetAttribute\(&(\w+), cudaDevAttrMaxGridDimX, 0\)[^)]*\)',
             r'mtl_grid_dim_x(&\1)', src)
src = re.sub(r'cudaDeviceGetAttribute\(&(\w+),\s*cudaDevAttrMaxGridDimX,\s*0\)',
             r'mtl_grid_dim_x(&\1)', src)
src = src.replace('static int mtl_check_impl',
"""/* CUDA asks the driver for cudaDevAttrMaxGridDimX. Metal publishes no grid
 * ceiling of its own -- dispatchThreadgroups takes an NSUInteger -- so the
 * shim reports an honest 32-bit-safe bound through mtlGetDeviceProperties. */
static int mtl_grid_dim_x(int *out)
{
    mtlDeviceProp p;
    if (mtlGetDeviceProperties(&p, 0) != mtlSuccess) return -1;
    *out = p.maxGridSize[0];
    return 0;
}

static int mtl_check_impl""", 1)

# run_bench's LAUNCH_APPLY takes the template arguments as MACRO parameters,
# so the mangled kernel name has to be pasted by the preprocessor rather than
# formed here -- the same shape fbgen_gpu's LAUNCH_FIXED needed. The norm mode
# is passed as NORM_CONST/NORM_HORNER, which stringify to their own names, so
# the call sites pass 0/1 instead.
i0 = src.find('CK(cudaFuncSetAttribute(k_apply<CBV, AT, NM, false>')
if i0 >= 0:
    i1 = src.index('(int)smem));', i0) + len('(int)smem));')
    src = (src[:i0]
           + 'CK(mtlFuncSetMaxThreadgroupMemory('
           + '"k_apply_" #CBV "_" #AT "_" #NM "_0", smem));'
           + src[i1:])
src = re.sub(r'MTL_LAUNCH\(k_apply_CBV_AT_NM_0,',
             'MTL_LAUNCH_NAMED("k_apply_" #CBV "_" #AT "_" #NM "_0",', src)
for a, b in (('LAUNCH_APPLY(16, 1, NORM_CONST)', 'LAUNCH_APPLY(16, 1, 0)'),
             ('LAUNCH_APPLY(16, 1, NORM_HORNER)', 'LAUNCH_APPLY(16, 1, 1)'),
             ('LAUNCH_APPLY(16, 0, NORM_CONST)', 'LAUNCH_APPLY(16, 0, 0)'),
             ('LAUNCH_APPLY(16, 0, NORM_HORNER)', 'LAUNCH_APPLY(16, 0, 1)'),
             ('LAUNCH_APPLY(8, 1, NORM_CONST)', 'LAUNCH_APPLY(8, 1, 0)'),
             ('LAUNCH_APPLY(8, 1, NORM_HORNER)', 'LAUNCH_APPLY(8, 1, 1)'),
             ('LAUNCH_APPLY(8, 0, NORM_CONST)', 'LAUNCH_APPLY(8, 0, 0)'),
             ('LAUNCH_APPLY(8, 0, NORM_HORNER)', 'LAUNCH_APPLY(8, 0, 1)')):
    src = src.replace(a, b)

src = src.replace('cuda_optin_smem_limit', 'mtl_optin_smem_limit')

# Same as the pipeline's: the harness binds the same kernel, so it must agree.
_smem_old = "            const size_t smem = (size_t)ncell * CB / 8 + (size_t)nslice_pow2 * 2;"
_smem_new = "            const size_t smem = mtl_apply_smem(ncell, CB);"
assert _smem_old in src, 'harness apply smem shape changed'
src = src.replace(_smem_old, _smem_new, 1)
print('  apply threadgroup length via mtl_apply_smem')

# The harness's own error macro prints "CUDA <expr>: <error>". Benchmark-only,
# but it is the same class and costs one line.
_eh_old = '"CUDA %s: %s at %s:%d\\n"'
_eh_new = '"Metal %s: %s at %s:%d\\n"'
assert _eh_old in src, 'harness error message shape changed'
src = src.replace(_eh_old, _eh_new, 1)
print('  harness error message names Metal')

open(OUT, 'w').write(src)
print('wrote %s (%d lines, %d launches rewritten)' % (OUT, src.count('\n'), nl))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*', src)))
if left: print('  cuda* remaining (check whether comments):', ' '.join(left))
