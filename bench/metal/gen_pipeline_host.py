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
