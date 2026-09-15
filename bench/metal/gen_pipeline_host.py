#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces pipeline_host.inc from
# pipeline.cuh -- ~4,000 lines of host orchestration with no device code in it
# at all. This is the file the Phase 3 shim was built for: with metal_rt.h
# presenting CUDA's names and CUDA's blocking semantics, the port is close to
# a rename pass.
#
# The one thing renaming cannot do is the kernel names. Every launch here sits
# inside a function templated on `bool SLABBED`, so the mangled name cannot be
# formed textually -- the same problem cf_run_rounds<L> had. Those become a
# ternary over the two concrete names, which the compiler folds.
import re, sys
sys.path.insert(0, "bench/metal")
from portlib import rewrite_launches, apply_renames

SRC = 'bench/pipeline.cuh'
OUT = 'bench/metal/pipeline_host.inc'
src = open(SRC).read()

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

# cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n)
# -> the shim validates the request against the device ceiling instead, which
# is the check that matters on Apple (32 KB, well under CUDA's opt-in tier).
src = re.sub(r'cudaFuncSetAttribute\(\s*k_apply<[^>]*>\s*,\s*\n?\s*cudaFuncAttributeMaxDynamicSharedMemorySize,\s*\(int\)(\w+)\)',
             r'mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1" : "k_apply_16_1_1_0", \1)',
             src)
src = src.replace('cuda_optin_smem_limit', 'mtl_optin_smem_limit')

open(OUT, 'w').write(src)
print('wrote %s (%d lines, %d launches rewritten)' % (OUT, src.count('\n'), nl))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*', src)))
if left: print('  UNRENAMED cuda* remaining:', ' '.join(left))
