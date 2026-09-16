#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build -- the companion to gen_fbgen_metal.py.
# Produces fbgen_gpu_metal.cpp from fbgen_gpu.cu by deleting the device half
# (which now lives in fbgen_gpu.metal) and renaming the CUDA runtime calls
# onto metal_rt.h. That this is mostly renaming is the whole argument for
# building the shim in Phase 3.
import re

SRC = 'bench/fbgen_gpu.cu'
OUT = 'bench/metal/fbgen_gpu_metal.cpp'
lines = open(SRC).read().split('\n')

beg = next(i for i, l in enumerate(lines) if 'uint32_t d_big_mod(' in l)
beg = beg - 1 if lines[beg - 1].startswith('__device__') else beg
end = next(i for i, l in enumerate(lines) if 'k_upper_bound' in l and '__global__' in l)
d = 0; started = False
for i in range(end, len(lines)):
    d += lines[i].count('{') - lines[i].count('}')
    if '{' in lines[i]: started = True
    if started and d == 0: end = i; break

s = '\n'.join(lines[:beg] + [
    '/* ---- device code lives in metal/fbgen_gpu.metal ---------------------- *',
    ' * The %d lines of __device__/__global__ code that were here are compiled' % (end - beg + 1),
    ' * by the Metal shader compiler instead. Kernel names reach this file as',
    ' * strings through MTL_LAUNCH; templated kernels use the port-wide naming',
    ' * rule (base, then template arguments, joined by _, bools as 0/1).',
    ' */',
] + lines[end + 1:])

# ---- headers -------------------------------------------------------------
s = s.replace('#include <cuda_runtime.h>\n#include <cub/cub.cuh>',
              '#include "metal/metal_rt.h"\n#include "metal/metal_scan.h"')

# ---- __constant__ declarations become symbol names -----------------------
s = re.sub(r'__constant__[^;]*;\n', '', s)

# ---- launches ------------------------------------------------------------
# k_name<T,...><<<g,b>>>(args)  and  k_name<<<g,b>>>(args)
def tname(base, targs):
    if not targs: return base
    parts = [a.strip() for a in targs.split(',')]
    return base + '_' + '_'.join('1' if a == 'true' else '0' if a == 'false' else a for a in parts)

ALG_KERNELS = {'k_alg_roots_fixed', 'k_alg_roots_fixed_mark', 'k_alg_roots'}
RAT_KERNELS = {'k_rational_roots'}

def launch(m):
    base, targs, grid, block, args = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    name = tname(base, targs[1:-1] if targs else None)
    extra = ''
    if base in ALG_KERNELS:
        extra = ', (const gpu_big_t *)mtlGetSymbol("c_alg"), *(const int *)mtlGetSymbol("c_alg_deg")'
    if base in RAT_KERNELS:
        extra = ', (const gpu_big_t *)mtlGetSymbol("c_y0"), (const gpu_big_t *)mtlGetSymbol("c_y1")'
    return 'MTL_LAUNCH(%s, %s, %s, 0, 0, %s%s)' % (name, grid.strip(), block.strip(), args.strip(), extra)

# A regex cannot do this: grid expressions contain std::min<uint32_t>(...),
# so <<< ... >>> has to be found by scanning with nesting awareness.
def rewrite_launches(text):
    out = []
    i = 0
    count = 0
    while True:
        j = text.find('<<<', i)
        if j < 0:
            out.append(text[i:]); break
        # kernel name (and optional template arguments) immediately before
        k = j
        targs = None
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
        while e > 0 and (text[e - 1].isalnum() or text[e - 1] == '_'):
            e -= 1
        base = text[e:k]
        if not base.startswith('k_'):
            out.append(text[i:j + 3]); i = j + 3; continue
        # matching >>>
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
        # split grid,block at top level
        d = 0; parts = []; cur = ''
        for ch in cfg:
            if ch in '(<[': d += 1
            elif ch in ')>]': d -= 1
            if ch == ',' and d == 0: parts.append(cur); cur = ''
            else: cur += ch
        parts.append(cur)
        grid, block = parts[0], parts[1]
        # argument list after >>>
        a = text.index('(', m)
        d = 0; b = a
        while b < len(text):
            if text[b] == '(': d += 1
            elif text[b] == ')':
                d -= 1
                if d == 0: break
            b += 1
        args = text[a + 1:b]
        name = tname(base, targs)
        extra = ''
        if base in ALG_KERNELS:
            extra = ', (const gpu_big_t *)mtlGetSymbol("c_alg"), *(const int *)mtlGetSymbol("c_alg_deg")'
        if base in RAT_KERNELS:
            extra = ', (const gpu_big_t *)mtlGetSymbol("c_y0"), (const gpu_big_t *)mtlGetSymbol("c_y1")'
        out.append(text[i:e])
        out.append('MTL_LAUNCH(%s, %s, %s, 0, 0, %s%s)'
                   % (name, grid.strip(), block.strip(),
                      ' '.join(args.replace('\\\n', ' ').replace('\\', ' ').split()), extra))
        count += 1
        i = b + 1
    return ''.join(out), count

s, nl = rewrite_launches(s)

# ---- CUB -> metal_scan ---------------------------------------------------
s = re.sub(r'cub::DeviceSelect::Flagged\(\s*(NULL|\w+)\s*,\s*(\w+)\s*,',
           r'mtlSelectFlaggedU32(\1, &\2,', s)
s = re.sub(r'cub::DeviceScan::ExclusiveSum\(\s*(NULL|\w+)\s*,\s*(\w+)\s*,',
           r'mtlScanExclusiveSumU32(\1, &\2,', s)
# both of ours take a trailing stream argument
s = re.sub(r'(mtlSelectFlaggedU32\([^;]*?)\)\)', r'\1, 0))', s)
s = re.sub(r'(mtlScanExclusiveSumU32\([^;]*?)\)\)', r'\1, 0))', s)

# ---- runtime renames -----------------------------------------------------
REN = [
 ('cudaMemcpyToSymbol', 'mtlMemcpyToSymbol'), ('cudaDeviceProp', 'mtlDeviceProp'),
 ('cudaGetDeviceProperties', 'mtlGetDeviceProperties'), ('cudaGetErrorString', 'mtlGetErrorString'),
 ('cudaDeviceSynchronize', 'mtlDeviceSynchronize'), ('cudaGetLastError', 'mtlGetLastError'),
 ('cudaMemcpyDeviceToHost', 'mtlMemcpyDeviceToHost'), ('cudaMemcpyHostToDevice', 'mtlMemcpyHostToDevice'),
 ('cudaMemcpyDeviceToDevice', 'mtlMemcpyDeviceToDevice'),
 ('cudaEventElapsedTime', 'mtlEventElapsedTime'), ('cudaEventSynchronize', 'mtlEventSynchronize'),
 ('cudaEventDestroy', 'mtlEventDestroy'), ('cudaEventCreate', 'mtlEventCreate'),
 ('cudaEventRecord', 'mtlEventRecord'), ('cudaEvent_t', 'mtlEvent_t'),
 ('cudaSetDevice', 'mtlSetDevice'), ('cudaGetDevice', 'mtlGetDevice'), ('cudaGetDeviceCount', 'mtlGetDeviceCount'),
 ('cudaMemGetInfo', 'mtlMemGetInfo'), ('cudaError_t', 'mtlError_t'), ('cudaSuccess', 'mtlSuccess'),
 ('cudaMemcpyAsync', 'mtlMemcpyAsync'), ('cudaMemsetAsync', 'mtlMemsetAsync'),
 ('cudaMemcpy', 'mtlMemcpy'), ('cudaMemset', 'mtlMemset'),
 ('cudaMalloc', 'mtlMalloc'), ('cudaFreeHost', 'mtlFreeHost'), ('cudaHostAlloc', 'mtlHostAlloc'),
 ('cudaFree', 'mtlFree'), ('CUDA_OR_DIE', 'MTL_OR_DIE'), ('CUDA_CHECKED', 'MTL_CHECKED'),
]
for a, b in REN:
    s = s.replace(a, b)

# mtlMemcpyToSymbol takes a NAME and an offset
# The size argument is usually sizeof(...), so the closing paren must be
# found by counting, not by matching [^)]+.
def fix_symbol(text):
    out = []; i = 0
    while True:
        j = text.find('mtlMemcpyToSymbol(', i)
        if j < 0: out.append(text[i:]); break
        a = j + len('mtlMemcpyToSymbol(')
        d = 1; b = a
        while b < len(text):
            if text[b] == '(': d += 1
            elif text[b] == ')':
                d -= 1
                if d == 0: break
            b += 1
        inner = text[a:b]
        dd = 0; parts = []; cur = ''
        for ch in inner:
            if ch == '(': dd += 1
            elif ch == ')': dd -= 1
            if ch == ',' and dd == 0: parts.append(cur); cur = ''
            else: cur += ch
        parts.append(cur)
        sym = parts[0].strip()
        out.append(text[i:j])
        out.append('mtlMemcpyToSymbol("%s", %s, %s, 0)'
                   % (sym, parts[1].strip(), parts[2].strip()))
        i = b + 1
    return ''.join(out)

s = fix_symbol(s)

# The two A/B dispatch macros parameterise the template arguments, so the
# port-wide name mangling has to happen with token pasting rather than in this
# script. Benchmark-only paths (--alg-backend / --alg-kernel); the production
# generator never reaches them.
s = re.sub(r'#define LAUNCH_FIXED\(CAP, MONT\)[^\n]*\n[^\n]*\n',
 '#define LAUNCH_FIXED(CAP, MONTBIT)                                          \\\n'
 '                MTL_LAUNCH(k_alg_roots_fixed_##CAP##_##MONTBIT##_0, blocks, 128, 0, 0,  \\\n'
 '                    d_primes, nalg, d_rootbuf, d_counts, d_failures,        \\\n'
 '                    (const gpu_big_t *)mtlGetSymbol("c_alg"),               \\\n'
 '                    *(const int *)mtlGetSymbol("c_alg_deg"))\n', s)
s = re.sub(r'#define LAUNCH_GENERIC\(MONT, SYM\)[^\n]*\n[^\n]*\n',
 '#define LAUNCH_GENERIC(MONTBIT, SYMBIT)                                     \\\n'
 '                MTL_LAUNCH(k_alg_roots_##MONTBIT##_##SYMBIT, blocks, 128, 0, 0,        \\\n'
 '                    d_primes, nalg, d_rootbuf, d_counts, d_failures,        \\\n'
 '                    (const gpu_big_t *)mtlGetSymbol("c_alg"),               \\\n'
 '                    *(const int *)mtlGetSymbol("c_alg_deg"))\n', s)
for a, b in [('LAUNCH_FIXED(6, false)', 'LAUNCH_FIXED(6, 0)'), ('LAUNCH_FIXED(6, true)', 'LAUNCH_FIXED(6, 1)'),
             ('LAUNCH_FIXED(8, false)', 'LAUNCH_FIXED(8, 0)'), ('LAUNCH_FIXED(8, true)', 'LAUNCH_FIXED(8, 1)'),
             ('LAUNCH_GENERIC(false, true)', 'LAUNCH_GENERIC(0, 1)'),
             ('LAUNCH_GENERIC(false, false)', 'LAUNCH_GENERIC(0, 0)'),
             ('LAUNCH_GENERIC(true, true)', 'LAUNCH_GENERIC(1, 1)'),
             ('LAUNCH_GENERIC(true, false)', 'LAUNCH_GENERIC(1, 0)')]:
    s = s.replace(a, b)

print('wrote %s (%d lines); %d launches rewritten' % (OUT, s.count('\n'), nl))

# ---- device-facing messages name Metal, not CUDA -------------------------
# fbgen runs on EVERY production task -- a workunit that supplies no --fb1
# generates the factor base here -- so these reach volunteers' stderr.txt.
# Found by the plan 9z-g sweep, after a field log showed the earlier pass had
# only fixed the lines it had seen fire.
for _o, _n in [
    ('"fbgen_gpu: CUDA %s failed at %s:%d: %s\\n"',
     '"fbgen_gpu: Metal %s failed at %s:%d: %s\\n"'),
    ('"%s: cannot query current CUDA device\\n"',
     '"%s: cannot query the current Metal device\\n"'),
    ('"%s: cannot query CUDA device %d\\n"',
     '"%s: cannot query Metal device %d\\n"'),
    ('"fbgen_gpu: cannot create CUDA events\\n"',
     '"fbgen_gpu: cannot create Metal timing events\\n"'),
]:
    assert _o in s, 'fbgen message shape changed: ' + _o[:44]
    s = s.replace(_o, _n, 1)
print('  fbgen device messages name Metal')

# ---- bound the root finder's launch duration -----------------------------
# macOS kills a command buffer that hogs the GPU against the UI. A field task
# on an M1 died with
#
#   command buffer failed: internal (code 1): Impacting Interactivity
#   (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
#
# and the reason is the grid: ablocks is min((n+127)/128, cores*8), so the
# SMALLER the GPU the MORE primes each thread must loop over. Measured on a
# 10-core M3, 80 blocks: ~790 ms per launch, ~105 primes per thread -- already
# past this build's own 750 ms interactivity policy, which until now applied
# only to the cofactoriser. An M1 gets 56-64 blocks and a slower core, so the
# same launch runs into the seconds. That is the whole family correlation:
# M1/M2 fail, M3/M4 Max do not, and an idle machine survives where a machine
# in use does not.
#
# Sliced by ITERATIONS PER THREAD rather than by a fixed prime count, so the
# bound is machine-independent in the right way: every device does K
# grid-strides per launch and only the per-stride cost varies.
#
# No kernel change. The slices are taken with pointer arithmetic on the
# allocation registry -- an interior pointer resolves to (buffer, offset) at
# bind time, which is exactly what the registry is for -- so the device side
# is untouched and fbgen_gpu.cu stays as it is.
_rf_old = chr(10).join([
    "        {",
    "            const uint32_t ablocks = std::min<uint32_t>((nprime + 127u) / 128u,",
    "                                                        (uint32_t)prop.multiProcessorCount * 8u);",
    "            if (P->deg <= 6)",
    "                MTL_LAUNCH(k_alg_roots_fixed_mark_6_1, ablocks, 128, 0, 0, d_primes, nprime, d_rootbuf, d_counts, d_special, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "            else",
    "                MTL_LAUNCH(k_alg_roots_fixed_mark_8_1, ablocks, 128, 0, 0, d_primes, nprime, d_rootbuf, d_counts, d_special, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "        }"])
_rf_new = chr(10).join([
    "        {",
    "            const uint32_t ablocks = std::min<uint32_t>((nprime + 127u) / 128u,",
    "                                                        (uint32_t)prop.multiProcessorCount * 8u);",
    "            /* One grid-stride covers `wave` primes; cap each SUBMISSION at",
    "             * FB_ROOTS_STRIDES_PER_LAUNCH of them. The flush is the load-",
    "             * bearing half: macOS's interactivity watchdog judges a command",
    "             * buffer, and the stream would otherwise batch every slice into",
    "             * one. See metal/gen_fbgen_host.py and plan 9z-k. */",
    "            const uint32_t wave = ablocks * 128u;",
    "            const uint64_t step64 = (uint64_t)wave * FB_ROOTS_STRIDES_PER_LAUNCH;",
    "            const uint32_t step = step64 >= nprime ? nprime : (uint32_t)step64;",
    "            for (uint32_t off = 0; off < nprime; off += step) {",
    "                const uint32_t cnt = (nprime - off) < step ? (nprime - off) : step;",
    "                const uint32_t sb = std::min<uint32_t>((cnt + 127u) / 128u, ablocks);",
    "                uint32_t *r_off = d_rootbuf + (size_t)off * GPU_FB_MAX_ROOTS;",
    "                if (P->deg <= 6)",
    "                    MTL_LAUNCH(k_alg_roots_fixed_mark_6_1, sb, 128, 0, 0, d_primes + off, cnt, r_off, d_counts + off, d_special + off, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "                else",
    "                    MTL_LAUNCH(k_alg_roots_fixed_mark_8_1, sb, 128, 0, 0, d_primes + off, cnt, r_off, d_counts + off, d_special + off, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "                MTL_OR_DIE(mtlStreamFlush(0));",
    "            }",
    "        }"])
assert _rf_old in s, 'root-finder launch shape changed'
s = s.replace(_rf_old, _rf_new, 1)

_k_old = "#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)"
_k_new = chr(10).join([
    "#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)",
    "",
    "/* Grid-strides per root-finder SUBMISSION -- each slice is flushed, so this",
    " * bounds a command buffer rather than a dispatch, which is the unit macOS's",
    " * interactivity watchdog actually judges. Measured on this 10-core M3: the",
    " * unsliced segment is a 790 ms command buffer; at 16 it is ~100 ms, leaving",
    " * ~7x for a device slower per stride (an M1 has 7-8 cores). Raise it only",
    " * with a measurement of GPUEndTime - GPUStartTime, NOT of dispatch time --",
    " * measuring the dispatch is how 9z-j came to report a fix that changed",
    " * nothing. Too large is a workunit killed by macOS, not a slow one. */",
    "#ifndef FB_ROOTS_STRIDES_PER_LAUNCH",
    "#define FB_ROOTS_STRIDES_PER_LAUNCH 16u",
    "#endif"])
assert _k_old in s, 'GPU_FB_MAX_ROOTS define not found'
s = s.replace(_k_old, _k_new, 1)
print('  root finder sliced to bound its launch duration')

open(OUT, 'w').write(s)
print('re-wrote %s with macro dispatch fixed' % OUT)
