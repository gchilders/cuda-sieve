#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces bench_main_metal.cpp from
# bench_main.cu, which contains no device code and no launches at all -- only
# CLI parsing, band driving, and the device-selection and version-reporting
# calls this rewrites.
import re, sys
sys.path.insert(0, 'bench/metal')
from portlib import apply_renames

SRC = 'bench/bench_main.cu'
OUT = 'bench/metal/bench_main_metal.cpp'
src = open(SRC).read()

# ---- device initialisation ----------------------------------------------
# CUDA's dance here exists to get cudaDeviceScheduleBlockingSync onto a device
# that is not necessarily 0, across a CUDART_VERSION split. Metal has no
# scheduling-flags concept at all: waitUntilCompleted already blocks the
# calling thread, which is exactly what --blocking-sync asks for. So the whole
# block reduces to selecting the device and saying so.
i0 = src.index('        if (blocking_sync) {\n#if CUDART_VERSION >= 12000')
j0 = src.index('    }\n\n    /* Grid width is 6 resident blocks per SM', i0)
src = src[:i0] + '''        /* CUDA needs cudaInitDevice/cudaSetDeviceFlags here to put
         * cudaDeviceScheduleBlockingSync on a device that is not necessarily
         * 0, across a CUDART_VERSION split. Metal has no scheduling-flags
         * concept: a command buffer's waitUntilCompleted already blocks the
         * calling thread, which is what --blocking-sync is asking for. So
         * there is nothing to configure and nothing to verify -- select the
         * device and carry on. */
        err = mtlSetDevice(selected_device);
        if (err != mtlSuccess) {
            fprintf(stderr, "bench: cannot select Metal device %d: %s\\n",
                    selected_device, mtlGetErrorString(err));
            return 1;
        }
        if (blocking_sync)
            printf("  --blocking-sync: implicit on Metal"
                   " (waitUntilCompleted already blocks)\\n");
''' + src[j0:]

# ---- version reporting ---------------------------------------------------
# There is no driver/runtime version pair to report. Name the GPU instead,
# which is the thing a run log actually wants to identify.
src = src.replace(
    '            const int drv_ok = cudaDriverGetVersion(&drv) == cudaSuccess;\n'
    '            const int rt_ok = cudaRuntimeGetVersion(&rt) == cudaSuccess;',
    '            /* Metal publishes no driver/runtime version pair. The GPU\n'
    '             * name is what identifies the machine in a run log. */\n'
    '            mtlDeviceProp _vp;\n'
    '            const int drv_ok = 0, rt_ok = 0;\n'
    '            (void)drv_ok; (void)rt_ok; (void)drv; (void)rt;\n'
    '            if (mtlGetDeviceProperties(&_vp, 0) == mtlSuccess)\n'
    '                runlog_note("device", "%s", _vp.name);')

# THE THREADGROUP-MEMORY DEFAULT.
#
# k_apply wants (1 << log_region) * 2 bytes for the region plus 2 per padded
# slice. At CUDA's default log_region = 14 that is 32,896 B on the c183 job --
# 128 bytes over Apple's HARD 32,768 B threadgroup ceiling, which has no
# opt-in tier the way CUDA's ~100 KB does. The run fails closed with a clear
# message (pipeline.cuh's own check), but failing closed on the default
# geometry is not a usable default.
#
# 13 is the largest value that fits, exactly as planned in section 5.1. This
# is a Metal-side default only; the CUDA build is untouched, and --region
# still overrides.
old = 'cfg.logI = 15; cfg.J = 16384; cfg.slab_j = 0; cfg.log_region = 14;'
new = ('cfg.logI = 15; cfg.J = 16384; cfg.slab_j = 0; cfg.log_region = 13;'
       '  /* 13, not CUDA\'s 14: Apple\'s threadgroup ceiling is a hard'
       ' 32 KB */')
assert old in src, 'bench_main.cu default geometry line changed'
src = src.replace(old, new)

# --mode twolevel FAILS CLOSED on Metal.
#
# The two-level fill kernels now compile and fit -- L1_CAP/L2_CAP retuned from
# 64 to 61 for Apple's 32 KB threadgroup ceiling -- but they MISPLACE RECORDS
# ACROSS REGIONS: the per-region gate reports ~350-500 of 2048 regions off by
# about one, with the grand total exactly right. That is precisely the failure
# bench_kernels.cu:2617 warns about ("every placement bug this project has hit
# had exactly the right total"), and the cause is not yet found.
#
# It is not the production path -- apply requires single-level 4-byte records
# (bench_kernels.cu:2660), so --pipeline never reaches it -- so refusing costs
# nothing today. Refusing rather than warning is the tree's own convention for
# a path known to compute the wrong thing.
old_mode = """            if (!strcmp(m, "atomic")) cfg.fill_mode = FILL_ATOMIC;
            else if (!strcmp(m, "twolevel")) cfg.fill_mode = FILL_TWOLEVEL;"""
new_mode = (
    '            if (!strcmp(m, "atomic")) cfg.fill_mode = FILL_ATOMIC;' + chr(10) +
    '            else if (!strcmp(m, "twolevel")) {' + chr(10) +
    '                fprintf(stderr, "--mode twolevel is refused on Metal: '
    'the two-level fill kernels compile and fit, but they misplace records '
    'across regions (right total, wrong distribution -- run with --verify to '
    'see it). Not the production path; apply needs single-level 4 B records. '
    'See METAL_PORT_PLAN.md section 8.\\n");' + chr(10) +
    '                return 1;' + chr(10) +
    '            }')
assert old_mode in src, 'bench_main.cu --mode parsing changed'
src = src.replace(old_mode, new_mode)

src = apply_renames(src)

# names with no Metal analogue
src = src.replace('CUDA_VISIBLE_DEVICES', 'CUDA_SIEVE_METAL_DEVICE')
src = src.replace('#include <cuda_runtime.h>', '#include "metal/metal_rt.h"')
if '#include "metal/metal_rt.h"' not in src:
    src = src.replace('#include "bench.h"', '#include "bench.h"\n#include "metal/metal_rt.h"', 1)

open(OUT, 'w').write(src)
print('wrote %s (%d lines)' % (OUT, src.count('\n')))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*|CUDART_VERSION', src)))
if left: print('  cuda* remaining (check whether comments):', ' '.join(left))
