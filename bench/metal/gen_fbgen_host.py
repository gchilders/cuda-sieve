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
    "            /* SLICED AND STEERED. macOS's interactivity watchdog judges a",
    "             * COMMAND BUFFER, so each slice is flushed; the flush is half",
    "             * the mechanism and the slice SIZE is the other half.",
    "             *",
    "             * The size is MEASURED, not chosen. A fixed grid-stride count",
    "             * does not transfer between devices: one stride covers `wave`",
    "             * primes and wave scales with core count, so 16 strides was",
    "             * 130 ms on this 10-core M3 and an estimated 560-840 ms across",
    "             * the M1 family -- over the bound, and near the ~800 ms at",
    "             * which field tasks were actually killed. That is 9z-j's error",
    "             * one level up: 9z-j replaced a fixed prime count with a fixed",
    "             * stride count and assumed strides were the device-independent",
    "             * unit. They are not. Nothing here is.",
    "             *",
    "             * So start at a size safe on the slowest device there is field",
    "             * data for, and let the measurement move it -- exactly as",
    "             * cofq_flush steers the cofactor chunk. See plan 9z-o. */",
    "            const uint32_t wave = ablocks * 128u;",
    "            for (uint32_t off = 0; off < nprime; ) {",
    "                const uint64_t step64 = (uint64_t)wave * g_fb_strides;",
    "                const uint32_t step = step64 >= nprime ? nprime : (uint32_t)step64;",
    "                const uint32_t cnt = (nprime - off) < step ? (nprime - off) : step;",
    "                const uint32_t sb = std::min<uint32_t>((cnt + 127u) / 128u, ablocks);",
    "                uint32_t *r_off = d_rootbuf + (size_t)off * GPU_FB_MAX_ROOTS;",
    "                if (P->deg <= 6)",
    "                    MTL_LAUNCH(k_alg_roots_fixed_mark_6_1, sb, 128, 0, 0, d_primes + off, cnt, r_off, d_counts + off, d_special + off, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "                else",
    "                    MTL_LAUNCH(k_alg_roots_fixed_mark_8_1, sb, 128, 0, 0, d_primes + off, cnt, r_off, d_counts + off, d_special + off, d_failures, (const gpu_big_t *)mtlGetSymbol(\"c_alg\"), *(const int *)mtlGetSymbol(\"c_alg_deg\"));",
    "                MTL_OR_DIE(mtlStreamFlush(0));",
    "                off += cnt;",
    "            }",
    "        }"])

assert _rf_old in s, 'root-finder launch shape changed'
s = s.replace(_rf_old, _rf_new, 1)

_k_old = "#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)"
_k_new = chr(10).join([
    "#define GPU_FB_MAX_ROOTS (BENCH_MAX_DEGREE + 1)",
    "",
    "/* Grid-strides per root-finder SUBMISSION -- a STARTING POINT, not a",
    " * constant. 4 is safe on the slowest device this port has field data for:",
    " * an estimated 140-210 ms across the M1 family, where 16 strides measured",
    " * 560-840 ms. A faster part is raised by the controller within one or two",
    " * submissions, so starting low costs almost nothing. */",
    "#ifndef FB_ROOTS_STRIDES_START",
    "#define FB_ROOTS_STRIDES_START 4u",
    "#endif",
    "/* A ceiling, so a mismeasurement cannot run away. ~8x what a 10-core M3",
    " * settles at. */",
    "#ifndef FB_ROOTS_STRIDES_MAX",
    "#define FB_ROOTS_STRIDES_MAX 256u",
    "#endif",
    "/* Attempts at one segment before giving up. The watchdog that motivates",
    " * the retry is contention-dependent, so a couple of halvings is normally",
    " * enough; a bound that is NOT transient (a page fault, say) then costs",
    " * four quick attempts instead of hanging. */",
    "#ifndef FB_ROOTS_MAX_ATTEMPTS",
    "#define FB_ROOTS_MAX_ATTEMPTS 4",
    "#endif",
    "",
    "static uint32_t g_fb_strides = FB_ROOTS_STRIDES_START;",
    "static float    g_fb_launch_max;          /* worst over-bound launch reported */",
    "static unsigned long long g_fb_seen_seq;  /* last submission steered on */",
    "",
    "/* Steer the slice size from the last COMPLETED submission. Proportional,",
    " * aiming at 0.8x the bound, with the dead band cofq_flush uses: act when",
    " * the measurement is over target or under a quarter of it, else leave it.",
    " *",
    " * THE seq TEST IS NOT OPTIONAL. The reading is one or two submissions",
    " * stale, so steering twice on the same measurement applies the same",
    " * correction twice: a 4 -> 38 grow would become 4 -> 38 -> 256, a",
    " * multi-second command buffer, which is exactly what the bound exists to",
    " * prevent. One measurement, one adjustment. */",
    "static void fb_steer_strides(const char *who)",
    "{",
    "    const float bound  = MTL_INTERACTIVITY_BOUND_MS;",
    "    const float target = bound * 0.8f;",
    "    float ms = 0.0f;",
    "    unsigned long long seq = 0;",
    "    if (mtlStreamWorstMs(0, &ms, &seq) != mtlSuccess) return;",
    "    if (ms <= 0.0f || seq <= g_fb_seen_seq) return;   /* nothing new */",
    "    g_fb_seen_seq = seq;",
    "    if (ms > bound && ms > g_fb_launch_max) {",
    "        g_fb_launch_max = ms;",
    "        fprintf(stderr,",
    "                \"%s: root-finder launch %.0f ms is over this build's %.0f ms\"",
    "                \" bound (%u grid-strides)\\n\",",
    "                who, (double)ms, (double)bound, g_fb_strides);",
    "    }",
    "    if (ms <= target && ms >= target * 0.25f) return;  /* dead band */",
    "    double want = (double)g_fb_strides * ((double)target / (double)ms);",
    "    if (want < 1.0) want = 1.0;",
    "    if (want > (double)FB_ROOTS_STRIDES_MAX) want = (double)FB_ROOTS_STRIDES_MAX;",
    "    g_fb_strides = (uint32_t)want;",
    "}"])

assert _k_old in s, 'GPU_FB_MAX_ROOTS define not found'
s = s.replace(_k_old, _k_new, 1)
print('  root finder sliced to bound its launch duration')

# ---- steer the root finder once per segment ------------------------------
_st_old = '        MTL_OR_DIE(mtlMemcpy(&failures, d_failures, sizeof(failures),\n                               mtlMemcpyDeviceToHost));'
_st_new = "        MTL_OR_DIE(mtlMemcpy(&failures, d_failures, sizeof(failures),\n                               mtlMemcpyDeviceToHost));\n        /* ONE adjustment per segment, here and nowhere else: that memcpy is\n         * the sync that drains this segment's submissions, so it is the only\n         * moment their durations exist. Steering inside the slice loop reads\n         * whatever OTHER stage last drained -- the odds sieve, the select, the\n         * scan -- and each of those is a fresh measurement by the seq test's\n         * reckoning, so the controller grows on every one of them. Measured:\n         * it ran to the 256-stride ceiling inside the first segment and put\n         * the whole segment in one 788 ms command buffer, which is worse than\n         * the fixed size it replaced. */\n        fb_steer_strides(who);"
assert _st_old in s, 'segment sync shape changed'
s = s.replace(_st_old, _st_new, 1)
print('  root finder steers on the segment drain')

# ---- an interactivity kill is TRANSIENT: halve and redo the segment -------
# The watchdog that kills these buffers is contention-dependent -- it fires
# when the GPU is wanted elsewhere -- so no slice size, measured or chosen, can
# guarantee it never fires. A field M2 lost a whole workunit to two killed
# buffers at 60% of a 250M factor base. Recovery has to exist alongside sizing.
#
# The root finder is IDEMPOTENT, which is what makes this safe: d_rootbuf,
# d_counts and d_special are indexed by prime and simply rewritten, d_failures
# is re-zeroed, and nothing has reached the sink yet -- the scan, k_total_roots
# and every sink->  call are still below. So re-running the loop re-derives the
# same segment from the same inputs.
#
# Halving on each attempt makes the kill itself the controller's strongest
# input: it is the one measurement that says "too long" without needing a
# timer.
_ra = s.index("        MTL_OR_DIE(mtlMemset(d_failures, 0, sizeof(*d_failures)));")
_rz = s.index("        fb_steer_strides(who);", _ra) + len("        fb_steer_strides(who);")
_region = s[_ra:_rz]

_tail_old = ("        MTL_OR_DIE(mtlGetLastError());" + chr(10) +
             "        MTL_OR_DIE(mtlMemcpy(&failures, d_failures, sizeof(failures)," + chr(10) +
             "                               mtlMemcpyDeviceToHost));")
_tail_new = ("        mtlError_t fb_rc = mtlGetLastError();" + chr(10) +
             "        if (fb_rc == mtlSuccess)" + chr(10) +
             "            fb_rc = mtlMemcpy(&failures, d_failures, sizeof(failures)," + chr(10) +
             "                              mtlMemcpyDeviceToHost);")
assert _tail_old in _region, 'segment sync shape changed (retry wrap)'
_region = _region.replace(_tail_old, _tail_new, 1)

# indent the whole region one level, then wrap it in the attempt loop
_region = chr(10).join(("    " + l) if l.strip() else l for l in _region.split(chr(10)))
# The steer must not run on a FAILED attempt: it reads the drain that just
# died, measures the surviving short buffers as cheap, and grows the slice
# straight back over the halving. Measured: 34 -> halve to 17 -> steer to 256
# -> "halve" to 128, i.e. the recovery made the next attempt four times worse.
assert _region.rstrip().endswith("fb_steer_strides(who);"), 'steer call moved'
_region = _region.rstrip()[:-len("fb_steer_strides(who);")].rstrip(chr(10) + " ")

_wrapped = (
    "        for (int fb_try = 0; ; fb_try++) {" + chr(10) +
    "            /* Consume any error left by the previous attempt. mtlMemcpy's"
    + chr(10) +
    "             * failure also sets the sticky last-error, so without this the"
    + chr(10) +
    "             * next attempt reads it back and 'fails' again having done"
    + chr(10) +
    "             * nothing wrong -- measured: every injected fault produced"
    + chr(10) +
    "             * exactly two retries. */" + chr(10) +
    "            (void)mtlGetLastError();" + chr(10) +
    _region + chr(10) +
    "            if (fb_rc == mtlSuccess) { fb_steer_strides(who); break; }" + chr(10) +
    "            if (fb_try + 1 >= FB_ROOTS_MAX_ATTEMPTS) {" + chr(10) +
    "                fprintf(stderr, \"%s: root finder in [%u,%u] failed %d times: %s\\n\"," + chr(10) +
    "                        who, lo, (uint32_t)hi64, fb_try + 1," + chr(10) +
    "                        mtlGetErrorString(fb_rc));" + chr(10) +
    "                goto fail;" + chr(10) +
    "            }" + chr(10) +
    "            g_fb_strides = g_fb_strides > 1u ? g_fb_strides / 2u : 1u;" + chr(10) +
    "            fprintf(stderr, \"%s: root finder in [%u,%u] did not complete (%s);\"" + chr(10) +
    "                    \" retrying at %u grid-strides\\n\"," + chr(10) +
    "                    who, lo, (uint32_t)hi64, mtlGetErrorString(fb_rc)," + chr(10) +
    "                    g_fb_strides);" + chr(10) +
    "        }")
s = s[:_ra] + _wrapped + s[_rz:]
print('  root finder retries a killed segment at half the slice size')

open(OUT, 'w').write(s)
print('re-wrote %s with macro dispatch fixed' % OUT)
