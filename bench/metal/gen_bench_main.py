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
