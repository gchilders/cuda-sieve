#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces bench_host.cpp from
# bench_kernels.cu's HOST half -- the TU that, in the CUDA build, also pulls in
# td.cuh, cofac.cuh and pipeline.cuh. Same shape here, with the ported
# versions.
import re, sys
sys.path.insert(0, 'bench/metal')
from portlib import (rewrite_launches, apply_renames,
                     kernel_param_defaults, pad_launch_defaults)

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

# ss_tiers (upstream 6997d77) is the same case as build_slices_b below: HOST
# code that sits among the device functions, so the span removal takes it with
# them -- and BOTH generated host files call it, bench_host.cpp's run_bench and
# pipeline_host.inc's small-sieve setup. Rescued by name, and `find` raises if
# it ever moves, because the alternative signal is an "undeclared identifier"
# two generated files away with nothing pointing back to here.
ts_a = find(r'^static inline void ss_tiers\(')
ss_tiers_src = '\n'.join(lines[ts_a:upto_close(ts_a) + 1])
assert 'nblk' in ss_tiers_src and 'nwrp' in ss_tiers_src, 'ss_tiers shape changed'

# build_slices_b is host code that happens to sit between two kernels, so the
# device-span removal would take it with them. Lift it out and re-insert.
bs_a = find(r'^static uint32_t build_slices_b\(')
bs_b = upto_close(bs_a)
build_slices = ss_tiers_src + '\n\n' + '\n'.join(lines[bs_a:bs_b + 1])

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
# ASSERTED: missing this is silent. The build would simply keep CUDA's 512
# threads for k_apply, which measured 29.5% slower than 192 on this box -- a
# performance regression with no compile error and no changed output.
assert n_athr >= 1, 'the apply-threads default is no longer spelled `: 512;`'
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
_fsa = 'CK(cudaFuncSetAttribute(k_apply<CBV, AT, NM, false, ST>'
assert src.count(_fsa) == 1, 'LAUNCH_APPLY cudaFuncSetAttribute shape changed'
i0 = src.index(_fsa)
i1 = src.index('(int)smem));', i0) + len('(int)smem));')
src = (src[:i0]
       + 'CK(mtlFuncSetMaxThreadgroupMemory('
       + '"k_apply_" #CBV "_" #AT "_" #NM "_0_" #ST, smem));'
       + src[i1:])
src, _n = re.subn(r'MTL_LAUNCH\(k_apply_CBV_AT_NM_0_ST,',
                  'MTL_LAUNCH_NAMED("k_apply_" #CBV "_" #AT "_" #NM "_0_" #ST,', src)
assert _n == 1, 'LAUNCH_APPLY launch shape changed'

# The macro arguments are PASTED into the kernel name, so every one of them
# has to arrive as the digit the suffix uses. NORM_CONST/NORM_HORNER and
# true/false stringify to their own spelling, so the call sites are rewritten
# to 0/1 -- and the count is asserted, because a call site this missed would
# ask for a kernel named "k_apply_16_1_NORM_HORNER_0_true" and fail only when
# that pricing arm is run.
_sites = re.findall(r'LAUNCH_APPLY\((\d+), (\d+), (NORM_\w+), (true|false)\)', src)
assert len(_sites) == 9, ('expected 9 LAUNCH_APPLY call sites, found %d'
                           % len(_sites))
src = re.sub(r'LAUNCH_APPLY\((\d+), (\d+), NORM_CONST, (true|false)\)',
             lambda m: 'LAUNCH_APPLY(%s, %s, 0, %s)'
                       % (m.group(1), m.group(2), '1' if m.group(3) == 'true' else '0'), src)
src = re.sub(r'LAUNCH_APPLY\((\d+), (\d+), NORM_HORNER, (true|false)\)',
             lambda m: 'LAUNCH_APPLY(%s, %s, 1, %s)'
                       % (m.group(1), m.group(2), '1' if m.group(3) == 'true' else '0'), src)
assert 'LAUNCH_APPLY(16, 1, NORM' not in src and 'LAUNCH_APPLY(8, 0, NORM' not in src, \
    'a LAUNCH_APPLY call site still names the norm mode by macro'
print('  LAUNCH_APPLY: %d call sites mangled by the preprocessor' % len(_sites))

# ---- run_device_cases: a kernel passed as a FUNCTION POINTER -------------
# upstream 2a3dfb0 factored the two --verify device gates onto a template that
# takes the kernel as `void (*kern)(const Case *, uint32_t, uint32_t *)`.
# Metal reaches a kernel by NAME, not by address -- there is no device
# function pointer to take -- so the parameter becomes the name and the two
# callers pass a string. The kernels themselves go to the metallib
# (gen_bench_kernels.py carries both case structs and both kernels).
_rdc_old = """static int run_device_cases(void (*kern)(const Case *, uint32_t, uint32_t *),
                            const Case *h, uint32_t n, uint32_t blocks,
                            uint32_t *first_bad)"""
_rdc_new = """static int run_device_cases(const char *kern,
                            const Case *h, uint32_t n, uint32_t blocks,
                            uint32_t *first_bad)"""
assert src.count(_rdc_old) == 1, 'run_device_cases signature shape changed'
src = src.replace(_rdc_old, _rdc_new, 1)
_rdl_old = "    kern<<<blocks, 256>>>(d, n, d_bad);"
assert src.count(_rdl_old) == 1, 'run_device_cases launch shape changed'
src = src.replace(_rdl_old,
                  "    MTL_LAUNCH_NAMED(kern, blocks, 256, 0, 0, d, n, d_bad);", 1)
for _k in ('k_verify_td_mod_cases', 'k_verify_ss_first_cases'):
    _c = 'run_device_cases(%s,' % _k
    assert src.count(_c) == 1, 'run_device_cases call site for %s' % _k
    src = src.replace(_c, 'run_device_cases("%s",' % _k, 1)
print('  run_device_cases takes a kernel NAME (2 call sites)')

# ...and the kernel DEFINITION that upstream wrote between the two host
# verifiers goes with them. k_verify_td_mod_cases was already outside the host
# half; k_verify_ss_first_cases is new and sits among host code, the ss_tiers
# shape again, so it is dropped by name with the count asserted.
_ssk = '__global__ void k_verify_ss_first_cases('
assert src.count(_ssk) == 1, 'k_verify_ss_first_cases is not unique'
_i = src.index(_ssk); _j = src.index('{', _i); _d = 0
while _j < len(src):
    if src[_j] == '{': _d += 1
    elif src[_j] == '}':
        _d -= 1
        if _d == 0: break
    _j += 1
src = (src[:_i] + '/* device code: see metal/bench_kernels.metal */\n'
       + src[_j + 1:])
assert '__global__' not in src, 'device code survives in bench_host.cpp'

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


# ---- defaulted kernel arguments become explicit ones ----------------------
# See portlib.pad_launch_defaults: Metal has no default argument: every
# parameter a kernel declares must be bound at dispatch, so each omitted
# trailing argument is filled in here from the KERNEL'S own default rather
# than retyped per call site. The assert inside also checks that no launch
# passes more arguments than the kernel declares -- which is how a kernel
# gaining a parameter in the middle would be caught.
_sigs = kernel_param_defaults(['bench/bench_kernels.cu', 'bench/td.cuh',
                               'bench/pipeline.cuh'])
src, _npad, _bad = pad_launch_defaults(src, _sigs, allow_unresolved=('kern',))
assert not _bad, 'launch through a name this cannot resolve: %s' % ', '.join(sorted(_bad))
print('  %d launch(es) had defaulted arguments filled in' % _npad)

open(OUT, 'w').write(src)
print('wrote %s (%d lines, %d launches rewritten)' % (OUT, src.count('\n'), nl))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*', src)))
if left: print('  cuda* remaining (check whether comments):', ' '.join(left))
