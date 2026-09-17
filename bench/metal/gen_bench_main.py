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

# ---- cofactor grid: size it from the WORK, not from the core count --------
# multiProcessorCount * 6 is an NVIDIA-shaped rule. It works there because SM
# counts are large -- a 4090 gets 768 blocks, 196,608 threads, which cofac.cuh
# notes "exceed CQ_FLUSH outright". Apple reports 10-40 GPU cores, so the same
# formula gives a 10-core M3 sixty blocks: 15,360 threads against a flush of
# 131,072 records, a grid 8.5x smaller than its own work, with every thread
# walking ~8.5 records end to end.
#
# Plan 8i measured the cost at --nq 72: 465.3 ms/q of cofactor time at 60
# blocks against 357.0 at 512, and 512 is indistinguishable from 576 because
# this is a THRESHOLD -- clear CQ_FLUSH and you are done, since threads past
# the record count have no records to take.
#
# max(), not replacement: the core-count rule stays as a floor and takes over
# on a hypothetical very large Apple GPU. Uses the live cfg.threads, since
# --threads moves and a literal 256 would silently mis-size the grid with it.
_ab_old = "            const uint64_t ab = (uint64_t)prop.multiProcessorCount * 6u;"
_ab_new = (
    "            const uint64_t ab_cores = (uint64_t)prop.multiProcessorCount * 6u;" + chr(10) +
    "            /* One record per thread for a full CQ_FLUSH batch. */" + chr(10) +
    "            const uint64_t ab_work = (mtl_cof_flush_capacity() + (uint64_t)cfg.threads - 1)" + chr(10) +
    "                                     / (uint64_t)cfg.threads;" + chr(10) +
    "            const uint64_t ab = ab_cores > ab_work ? ab_cores : ab_work;")
assert _ab_old in src, 'auto_blocks shape changed'
src = src.replace(_ab_old, _ab_new, 1)

_pr_old = (
    '            printf("grid: %d SMs x 6 = %d blocks (dev %d: %s, %d MB L2)' + chr(92) + 'n",' + chr(10) +
    "                   prop.multiProcessorCount, cfg.blocks, dev, prop.name," + chr(10) +
    "                   prop.l2CacheSize >> 20);")
_pr_new = (
    '            printf("grid: %d blocks = max(%d cores x 6, %u records / %d threads)"' + chr(10) +
    '                   " (dev %d: %s, %d MB L2)' + chr(92) + 'n",' + chr(10) +
    "                   cfg.blocks, prop.multiProcessorCount," + chr(10) +
    "                   (unsigned)mtl_cof_flush_capacity(), cfg.threads, dev, prop.name," + chr(10) +
    "                   prop.l2CacheSize >> 20);")
assert _pr_old in src, 'grid report shape changed'
src = src.replace(_pr_old, _pr_new, 1)
print('  cofactor grid sized from CQ_FLUSH, core count kept as a floor')

# Hand the chunk floor its own, core-derived notion of a loaded device, so it
# does NOT inherit the work-sized grid above. Without this the floor would
# equal a whole flush and auto chunking could never subdivide on any Apple GPU
# -- see gen_cofac_host.py for why that protection has to survive.
_set_old = "            auto_blocks = (int)ab;"
_set_new = (
    "            auto_blocks = (int)ab;" + chr(10) +
    "            mtl_set_cof_floor_blocks((int)ab_cores);")
assert _set_old in src, 'auto_blocks assignment shape changed'
src = src.replace(_set_old, _set_new, 1)

_decl_anchor = '#include "metal/metal_rt.h"'
assert _decl_anchor in src, 'metal_rt.h include not found'
src = src.replace(
    _decl_anchor,
    _decl_anchor + chr(10) +
    "/* metal/cofac_host.inc, compiled into bench_host.cpp. Declared rather than" + chr(10) +
    " * included: this file needs the one setter, not the cofactor host. */" + chr(10) +
    "void mtl_set_cof_floor_blocks(int b);" + chr(10) +
    "uint32_t mtl_cof_flush_capacity(void);", 1)
print('  chunk floor given its core-derived blocks at init')

# A build marker in --help, mirroring the HIP port's "select HIP device".
# cofcheck.sh needs to tell the builds apart to handle the one case that is
# refused here, and grepping for anything less deliberate would be fragile.
_dev_old = '"  --device N       select CUDA device N  [CUDA\'s default device]\\n"'
_dev_new = '"  --device N       select Metal device N  [the system default device]\\n"'
assert _dev_old in src, '--device help line shape changed'
src = src.replace(_dev_old, _dev_new, 1)
print('  --help marks this as the Metal build')

# ---- the SAME marker under HAVE_BOINC ------------------------------------
# The rewrite above touches only the #else branch. A -DHAVE_BOINC build takes
# the other one, so `--help` would say "select CUDA device" and cofcheck.sh
# would classify this binary as the CUDA build -- and then run the
# `--ecm-b1 400000` case that crashed WindowServer twice (plan 8k). The
# detection marker has to hold in BOTH branches or it is not a marker.
_devb_old = chr(10).join([
    '"  --device N       select CUDA device N, used only when the BOINC client did\\n"',
    '"                   not assign one; its assignment wins  [CUDA\'s default]\\n"'])
_devb_new = chr(10).join([
    '"  --device N       select Metal device N, used only when the BOINC client did\\n"',
    '"                   not assign one; its assignment wins  [the system default]\\n"'])
assert _devb_old in src, 'BOINC --device help line shape changed'
src = src.replace(_devb_old, _devb_new, 1)
print('  --help marks this as the Metal build under HAVE_BOINC too')

# ---- the two runtime lines that name the wrong API ------------------------
# Both are stderr, which is what a BOINC client keeps; a volunteer's log
# saying "running on CUDA device 0" out of a Metal binary is a support ticket.
_gpu_old = chr(10).join([
    '                    "BOINC: no usable GPU assignment in init_data.xml; using"',
    '                    " CUDA\'s default device\\n");'])
_gpu_new = chr(10).join([
    '                    "BOINC: no usable GPU assignment in init_data.xml; using"',
    '                    " the system default Metal device\\n");'])
assert _gpu_old in src, 'BOINC no-assignment message shape changed'
src = src.replace(_gpu_old, _gpu_new, 1)

_run_old = '        fprintf(stderr, "BOINC: running on CUDA device %d of %d: %s\\n",'
_run_new = '        fprintf(stderr, "BOINC: running on Metal device %d of %d: %s\\n",'
assert _run_old in src, 'BOINC device-report message shape changed'
src = src.replace(_run_old, _run_new, 1)

# The comment above the first of those explains the field in terms of an
# NVIDIA coprocessor, which is the wrong vendor for this build and the wrong
# advice for a project packaging it.
_nv_old = "version's plan class actually declares an NVIDIA coprocessor."
_nv_new = "version's plan class actually declares a GPU coprocessor at all."
assert _nv_old in src, 'BOINC assignment comment shape changed'
src = src.replace(_nv_old, _nv_new, 1)
print('  BOINC stderr lines name Metal, not CUDA')

# ---- the rest of the device-facing messages ------------------------------
# A field run on an M4 Max (plan 9z-g) showed the earlier pass had fixed only
# the lines it had seen fire. These are the rest of the class in this file:
# every message a volunteer or a project admin can read that names the wrong
# API. Two of them (the "client assigned" pair) could not fire at all until
# the assignment rejection in boinc_support.cpp was fixed, which is exactly
# why grepping for what appears in a log is not the same as grepping for the
# class.
for _old, _new in [
    ('" task CUDA device %d\\n", cuda_device, boinc_device);',
     '" task Metal device %d\\n", cuda_device, boinc_device);'),
    ('"BOINC: client assigned CUDA device %d\\n"',
     '"BOINC: client assigned Metal device %d\\n"'),
    ('"bench: cannot enumerate CUDA devices: %s\\n"',
     '"bench: cannot enumerate Metal devices: %s\\n"'),
    ('"bench: this process sees no CUDA device\\n"',
     '"bench: this process sees no Metal device\\n"'),
    ('"bench: CUDA device %d requested, but this process sees"',
     '"bench: Metal device %d requested, but this process sees"'),
    ('"bench: cannot query the CUDA device -- refusing to"',
     '"bench: cannot query the Metal device -- refusing to"'),
]:
    assert _old in src, 'message shape changed: ' + _old[:48]
    src = src.replace(_old, _new, 1)
print('  six more device messages name Metal')

# ---- a missing GPU is TRANSIENT on macOS, not an error --------------------
# A field task (client 8.2.9) exited 1 seconds after starting with "this
# process sees no Metal device", on a host whose client had already assigned
# an apple_gpu -- so the hardware was there and the PROCESS could not reach
# it. On macOS that is what happens outside a GUI login session, and a retry
# once somebody logs in simply works. boinc_finish(1) instead burns the
# workunit and charges the host with a failure.
#
# mtlGetDeviceCount now returns mtlErrorNoDevice for exactly this case and
# something else for a refused GPU or an unloadable shader library, both of
# which ARE permanent and must keep erroring out. That distinction is the
# whole point: before it, all three shared one message and one exit path.
_nd_old = chr(10).join([
    "        if (ndev < 1) {",
    '            fprintf(stderr, "bench: this process sees no Metal device\\n");',
    "            return 1;",
    "        }"])
_nd_new = chr(10).join([
    "        if (err == mtlErrorNoDevice || ndev < 1) {",
    "            /* metal_rt has already said WHY on stderr. Ask the client to",
    "             * try again later rather than recording a permanent error;",
    "             * this returns only when BOINC is not managing the run. */",
    "            bench_boinc_temporary_exit(BENCH_NO_GPU_RETRY_S,",
    '                                       "no Metal device is visible to this'
    ' process");',
    '            fprintf(stderr, "bench: this process sees no Metal device\\n");',
    "            return 1;",
    "        }"])
assert _nd_old in src, 'no-device block shape changed'
src = src.replace(_nd_old, _nd_new, 1)

# The generic "cannot enumerate" test sits BEFORE the block above and would
# swallow mtlErrorNoDevice, leaving the new branch unreachable. Let that one
# through; it is handled, and handled differently.
_ee_old = chr(10).join([
    "        if (err != mtlSuccess) {",
    '            fprintf(stderr, "bench: cannot enumerate Metal devices: %s\\n",',
    "                    mtlGetErrorString(err));",
    "            return 1;",
    "        }"])
_ee_new = chr(10).join([
    "        if (err != mtlSuccess && err != mtlErrorNoDevice) {",
    "            /* A refused GPU or an unloadable shader library: permanent,",
    "             * so error out. mtlErrorNoDevice is transient and is handled",
    "             * just below -- do not collapse the two again. */",
    '            fprintf(stderr, "bench: cannot enumerate Metal devices: %s\\n",',
    "                    mtlGetErrorString(err));",
    "            return 1;",
    "        }"])
assert _ee_old in src, 'enumerate-error block shape changed'
src = src.replace(_ee_old, _ee_new, 1)

# The retry delay. Long enough not to spin against a host nobody is using,
# short enough that a task is not parked for an evening after a login.
_rt_old = "static int bench_main_impl(int argc, char **argv, enum bench_outcome *outcome)"
_rt_new = chr(10).join([
    "/* Seconds to ask BOINC to wait before retrying a task that found no GPU.",
    " * The cause is usually a login session, which can change at any time, so",
    " * this is a compromise: 10 minutes is six retries an hour rather than a",
    " * busy loop, and does not park a task all evening after somebody logs",
    " * in. Nothing has measured how long the condition typically lasts. */",
    "#define BENCH_NO_GPU_RETRY_S 600",
    "",
    "static int bench_main_impl(int argc, char **argv, enum bench_outcome *outcome)"])
assert src.count(_rt_old) == 1, 'bench_main_impl signature not unique'
src = src.replace(_rt_old, _rt_new, 1)
print('  no Metal device: temporary exit rather than a burnt workunit')

# ---- drop the allowance advisory ------------------------------------------
# Two stderr notes, emitted once per run, saying an --allowance is looser than
# the derived value. Useful at a terminal, noise in a BOINC log: under
# HAVE_BOINC every one of them lands in the volunteer's uploaded stderr.txt,
# and this build's parity settings (--allowance 101.6 --allowance0 68.1) fire
# BOTH of them on every single q. The advice is also unactionable there --
# the allowance comes from the job file the project sent.
#
# The whole braced block goes: sl1/sl0/d1/d0 exist only to produce those two
# notes, so leaving the block would trade two messages for four unused
# variables under -Wall -Wextra. m0/m1 stay live in the printf above it.
# CUDA's own behaviour is unchanged -- this is a Metal-side removal, and the
# stdout line above still reports the allowance in force.
_a0 = src.index("            {" + chr(10) +
                "                const double sl1 = cfg.allowance")
_a1 = src.index("                (void)sl1; (void)sl0;" + chr(10) +
                "            }" + chr(10), _a0)
_a1 += len("                (void)sl1; (void)sl0;" + chr(10) +
           "            }" + chr(10))
src = src[:_a0] + chr(10).join([
    "            /* CUDA warns here when an --allowance is looser than the",
    "             * derived value. Dropped in this build: under HAVE_BOINC it",
    "             * lands in the volunteer's uploaded stderr.txt, where it is",
    "             * noise and unactionable -- the allowance came from the job",
    "             * file the project sent. The stdout line above still reports",
    "             * the value in force. */",
    ""]) + src[_a1:]
print('  allowance advisory dropped (Metal-side only)')

# ---- lift the 24-round cap for ECM ----------------------------------------
# The cap's own message says why it exists: "budget << r overflows beyond
# that". That is RHO's iteration budget. ECM never shifts it -- the ECM
# launches pass S->curves unshifted, and the round index reaches the device
# only as c0 = r+1, which selects a 1000-wide sigma block
# (`sigma = c0*1000 + cv + 6`).
#
# The overflow it guards is also already guarded exactly, and only where it
# applies: bench_main's own check tests `(uint64_t)budget << (rounds-1) >
# 0xFFFFFFFF` against the ACTUAL budget, gated on a side actually using rho.
# That is why `--cof-rho --cof-rounds 20` is refused at the default budget of
# 65536 while `--cof-rounds 16` is accepted. The blanket 24 adds nothing for
# rho and costs ECM the only axis that can hold 8k's launch bound: to keep a
# curve budget while shortening launches you need MORE rounds of FEWER curves,
# and at B1 8000 one curve is ~370 ms, so 750 ms means 2 curves/round and 192
# curves would need 96 rounds.
#
# 1000 rounds, not unbounded: sigma must stay in uint32 (it would take ~4.3M
# rounds to break that) and each round costs five small kernels plus its
# launches, so a cap that is generous without being meaningless is the useful
# shape. Rho keeps 24 and its exact budget check underneath.
_r1_old = chr(10).join([
    "    if (cof_rounds < 1 || cof_rounds > 24) {",
    '        fprintf(stderr, "--cof-rounds %d: must be 1..24 (budget << r overflows"',
    '                " beyond that, and < 1 splits nothing)\\n", cof_rounds);',
    "        bad = 1;",
    "    }",
    "    if (cfg->cof_rounds < 1 || cfg->cof_rounds > 24) {",
    '        fprintf(stderr, "pipeline cof-rounds %d: must be 1..24\\n", cfg->cof_rounds);',
    "        bad = 1;",
    "    }"])
_r1_new = chr(10).join([
    "    {",
    "        /* 24 is rho's bound, not ECM's -- see metal/gen_cofac_host.py. The",
    "         * exact `budget << (rounds-1)` test below still gates rho. */",
    "        const int rho_in_use = (cfg->cof_meth0 == COF_METHOD_RHO ||",
    "                                cfg->cof_meth1 == COF_METHOD_RHO);",
    "        const int rmax = rho_in_use ? 24 : 1000;",
    "        if (cof_rounds < 1 || cof_rounds > rmax) {",
    '            fprintf(stderr, "--cof-rounds %d: must be 1..%d (%s)\\n",',
    "                    cof_rounds, rmax, rho_in_use",
    '                    ? "budget << r overflows beyond that for rho, and < 1'
    ' splits nothing"',
    '                    : "ECM does not shift the budget; this bound is the'
    ' sigma-block space, and < 1 splits nothing");',
    "            bad = 1;",
    "        }",
    "        if (cfg->cof_rounds < 1 || cfg->cof_rounds > rmax) {",
    '            fprintf(stderr, "pipeline cof-rounds %d: must be 1..%d\\n",',
    "                    cfg->cof_rounds, rmax);",
    "            bad = 1;",
    "        }",
    "        /* Sigma blocks are 1000 wide and indexed by the round, so more than",
    "         * 994 curves in a round runs into the NEXT round's sigmas and",
    "         * repeats them. Harmless arithmetically, but it silently spends",
    "         * curves that cannot find anything new -- and it matters more now",
    "         * that many-rounds-of-few-curves is the recommended shape. */",
    "        if (cfg->ecm_curves > 994) {",
    '            fprintf(stderr, "--ecm-curves %u: must be <= 994, or a round\'s"',
    '                    " sigmas (c0*1000 + cv + 6) run into the next round\'s"',
    '                    " and repeat them\\n", cfg->ecm_curves);',
    "            bad = 1;",
    "        }",
    "    }"])
assert _r1_old in src, 'cof-rounds validation shape changed'
src = src.replace(_r1_old, _r1_new, 1)
print('  cof-rounds cap lifted to 1000 for ECM; rho keeps 24; curves bounded to 994')

open(OUT, 'w').write(src)
print('wrote %s (%d lines)' % (OUT, src.count('\n')))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*|CUDART_VERSION', src)))
if left: print('  cuda* remaining (check whether comments):', ' '.join(left))
