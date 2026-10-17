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
from portlib import (_split_args, rewrite_launches, apply_renames,
                     kernel_param_defaults, pad_launch_defaults)

SRC = 'bench/pipeline.cuh'
OUT = 'bench/metal/pipeline_host.inc'
src = open(SRC).read()

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

# ---- pipeline.cuh's one piece of DEVICE code ------------------------------
# upstream 2a3dfb0 put k_resieve_bucket and its rb_mod helper here, in a file
# whose whole premise (see the header above) is that it holds no device code.
# They go to the metallib instead -- gen_td_metal.py appends the same span to
# td_body.metal.inc, because the kernel calls td_rank -- and are dropped here,
# since `__global__`, `__shared__` and __ballot_sync have no meaning in the
# host translation unit. Asserted on both sides: if one generator stops
# finding the span the other says so.
def _drop_fn(text, sig, what):
    assert text.count(sig) == 1, 'pipeline.cuh device code: %s is not unique' % what
    i = text.index(sig)
    j = text.index('{', i); d = 0
    while j < len(text):
        if text[j] == '{': d += 1
        elif text[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    return text[:i] + ('/* %s: device code, see metal/td.metal */\n' % what) + text[j + 1:]

src = _drop_fn(src, '__device__ __forceinline__ uint32_t rb_mod(', 'rb_mod')
src = _drop_fn(src, '__global__ void k_resieve_bucket(', 'k_resieve_bucket')
for _gone in ('__global__', '__shared__', '__ballot_sync(0xffffffffu, hit)'):
    assert _gone not in src, 'device code survives in the host file: ' + _gone
print('  k_resieve_bucket and rb_mod dropped (they are in the metallib)')


# cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n)
# -> the shim validates the request against the device ceiling instead, which
# is the check that matters on Apple (32 KB, well under CUDA's opt-in tier).
src = re.sub(r'cudaFuncSetAttribute\(\s*k_apply<[^>]*>\s*,\s*\n?\s*cudaFuncAttributeMaxDynamicSharedMemorySize,\s*\(int\)(\w+)\)',
             r'mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1" : "k_apply_16_1_1_0", \1)',
             src)
src = src.replace('cuda_optin_smem_limit', 'mtl_optin_smem_limit')

# ---- k_apply's SSTAIL instance, chosen at run time -----------------------
# upstream 2a3dfb0 picks between two specialisations of k_apply with a
# function POINTER (`auto *const apply_kern = cond ? k_apply<...,true> :
# k_apply<...,false>`). Metal reaches a kernel by name, so the pointer becomes
# a name and the launch becomes MTL_LAUNCH_NAMED. SLABBED is a template
# parameter of the enclosing function, so the name is a nested ternary over
# both -- which the compiler folds, since SLABBED is a constant there.
#
# The four names are spelled out rather than built, so that an instantiation
# missing from gen_bench_kernels.py's K table fails as a missing kernel with
# the name in the message instead of as a silently different specialisation.
_ak_old = """        auto *const apply_kern = ss_needs_tail(S->nsplit, S->nsmall)
            ? k_apply<16, 1, NORM_HORNER, SLABBED, true>
            : k_apply<16, 1, NORM_HORNER, SLABBED, false>;"""
_ak_new = """        const char *const apply_kern = ss_needs_tail(S->nsplit, S->nsmall)
            ? (SLABBED ? "k_apply_16_1_1_1_1" : "k_apply_16_1_1_0_1")
            : (SLABBED ? "k_apply_16_1_1_1_0" : "k_apply_16_1_1_0_0");"""
assert src.count(_ak_old) == 1, 'k_apply instance selection shape changed'
src = src.replace(_ak_old, _ak_new, 1)
# rewrite_launches skipped this launch: its callee is a variable, not a
# k_-prefixed name. Done here, NULL -> nullptr included -- NULL is 0L in C++
# and would bind eight bytes of zeros as a constant buffer (see metal_rt.h).
_akl_start = "        apply_kern<<<nregion, athr, S->apply_smem, st>>>("
assert src.count(_akl_start) == 1, 'k_apply launch shape changed'
_i = src.index(_akl_start)
_j = src.index(');', _i) + 2
_args = ' '.join(src[_i + len(_akl_start):_j - 2].split())
_args = re.sub(r'\bNULL\b', 'nullptr', _args)
src = (src[:_i]
       + "        MTL_LAUNCH_NAMED(apply_kern, nregion, athr, S->apply_smem, st,\n"
       + "            " + _args + ");"
       + src[_j:])
print('  k_apply SSTAIL instance selected by name')

# The two cudaFuncSetAttribute calls name the same two instances and were
# rewritten by the regex above WITHOUT the SSTAIL suffix, so they asked about
# kernels that no longer exist. The shim only validates the request against
# the device ceiling, which is the same answer for both tail instances, so one
# per SLABBED value is enough -- but it has to name one that EXISTS.
_fsa_old = 'mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1" : "k_apply_16_1_1_0", apply_smem)'
_nfsa = src.count(_fsa_old)
assert _nfsa == 2, 'expected 2 k_apply threadgroup-memory requests, found %d' % _nfsa
src = src.replace(_fsa_old,
                  'mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1_1"'
                  ' : "k_apply_16_1_1_0_1", apply_smem)')


# k_apply no longer keeps the slice-log table in threadgroup memory, so the
# length the host binds must lose that term too, or the host would reserve
# 128 bytes the kernel never indexes -- and at region 14 those 128 bytes are
# the difference between fitting the ceiling and being refused.
_smem_old = ("        const size_t smem = ((size_t)1 << cfg->log_region) * 2 +\n"
             "                            (size_t)S->nslice_pow2 * sizeof(*hlogp);")
_smem_new = "        const size_t smem = mtl_apply_smem(1u << cfg->log_region, 16);"
assert _smem_old in src, 'pipeline apply smem shape changed'
src = src.replace(_smem_old, _smem_new, 1)
print('  apply threadgroup length via mtl_apply_smem')

# Slab auto-calibration, ported from the HIP port. The FUNCTION is hand-written
# in metal/slab_calib.inc -- splicing a hundred lines of new logic in by regex
# is how this port broke a header once already -- so the generator only has to
# include it and rewrite the one call site.
_inc_anchor = 'extern "C" int run_pipeline(const fb_t *fb1, const fb_t *fbs1,'
assert _inc_anchor in src, 'run_pipeline signature changed'
src = src.replace(_inc_anchor,
                  '#include "slab_calib.inc"\n\n' + _inc_anchor, 1)

_call_old = """    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       cfg->slab_j, &plan)) {"""
_call_new = """    /* Auto-calibration returns 0 when it did not run or was inconclusive, in
     * which case the caller's own --slab-j (itself usually 0, meaning the
     * static default) is what the planner sees -- exactly as before. */
    const uint32_t calibrated_j = calibrate_slab_rows(fb1, fbs1, fb0, fbs0,
                                                      qlist, nq, POLY, cfg,
                                                      pmax);
    const uint32_t forced_j = calibrated_j ? calibrated_j : cfg->slab_j;

    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       forced_j, &plan)) {"""
assert _call_old in src, 'run_pipeline slab_make_plan call shape changed'
src = src.replace(_call_old, _call_new, 1)
print('  slab auto-calibration spliced in')

# The error message below it still names --slab-j; say which number the
# planner was actually given, or a calibrated run reports a request the
# caller never made.
_err_old = """                cfg->logI, cfg->J, pmax, cfg->slab_j);"""
assert src.count(_err_old) == 1, 'slab plan failure message shape changed'
src = src.replace(_err_old, """                cfg->logI, cfg->J, pmax, forced_j);""", 1)

# ---- upstream's VRAM-modelled slab target --------------------------------
# main 74fcba0 picks the largest of 4x / 2x / the default SLAB_PERF_REGIONS
# whose MODELLED footprint fits free VRAM. This port decides the same question
# by MEASUREMENT -- calibrate_slab_rows times the run's own first special-q
# against three slab sizes and keeps the fastest (plan 8g) -- so when the
# calibration fires it owns the decision and the model must not overwrite the
# plan underneath it. The model still runs when calibration did not (below the
# trigger geometry, or --slab-j), which is where it is the only evidence there
# is.
#
# Not a judgement that the model is wrong: on UMA `free VRAM` is free system
# memory and mtlMemGetInfo reports recommendedMaxWorkingSetSize, so the model
# would happily up-size here. Whether that beats the measured 2^26 is a tuning
# question with a measurement attached to it, not something to adopt silently.
_vram_old = "    if (!cfg->slab_j && !cfg->slab_fixed) {"
assert src.count(_vram_old) == 1, 'VRAM slab chooser guard shape changed'
src = src.replace(_vram_old,
                  "    if (!calibrated_j && !cfg->slab_j && !cfg->slab_fixed) {", 1)
print('  VRAM slab model yields to the measured calibration')

# ---- --instances auto never picks 2 on Metal -----------------------------
# upstream 00e2c8a makes `auto` the default, and auto can choose two whole
# pipelines on two host threads. NOTHING in this port has ever run that: the
# only two-thread path here is --cof-async, and the one experiment that let two
# threads into metal_rt more deeply (dropping g_lock across the GPU wait)
# produced a 288-q band with 3,054 of 13,485 relations WRONG while every gate
# passed, because every gate is single-threaded or single-q.
#
# So auto resolves to one here. An explicit `--instances N` is left working --
# this is "untested", not "known broken", and the way to test it is to ask for
# it -- but it warns, because the failure mode demonstrated above is silent
# wrong relations rather than a crash.
_inst_old = "                ninst = c2.fits && c2.idx == c1.idx ? 2 : 1;"
_inst_new = chr(10).join([
    "                /* METAL: one. See metal/gen_pipeline_host.py -- two host",
    "                 * threads against the shim is an untested path whose",
    "                 * demonstrated failure mode is silently wrong relations,",
    "                 * and no gate in this tree exercises it. */",
    "                ninst = 1;",
    "                (void)c2;"])
assert src.count(_inst_old) == 1, '--instances auto choice shape changed'
src = src.replace(_inst_old, _inst_new, 1)

_warn_old = "    if (IP && ninst >= 2) {"
_warn_new = chr(10).join([
    "    if (IP && ninst >= 2)",
    "        fprintf(stderr, \"  warning: --instances %d runs %d pipelines on %d host\"",
    "                        \" threads, which this Metal port has never tested; the\"",
    "                        \" failure mode to watch for is wrong relations, not a\"",
    "                        \" crash. --instances 1 is the tested path.\\n\",",
    "                ninst, ninst, ninst);",
    "    if (IP && ninst >= 2) {"])
assert src.count(_warn_old) == 1, 'instances dispatch shape changed'
src = src.replace(_warn_old, _warn_new, 1)
print('  --instances auto resolves to 1; an explicit N warns')

# The j-slabbing status line reports the STATIC target, which becomes a lie the
# moment calibration overrides it -- the plan printed would not be the one the
# named target implies. Say which decided.
_msg_old = """        if (!cfg->slab_j && perf_jmax != 0xffffffffu)"""
_msg_new = """        if (calibrated_j) {
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto-calibrated against this run's first special-q,"
                   " not the static target)\\n",
                   P.nslab, P.nslab == 1 ? "" : "s", P.jmax);
        } else if (!cfg->slab_j && perf_jmax != 0xffffffffu)"""
assert src.count(_msg_old) == 1, 'j-slabbing status message shape changed'
src = src.replace(_msg_old, _msg_new, 1)

# HIP's fleet-aggregation line, ported. stderr, because stderr.txt is the only
# per-task output a real BOINC client keeps -- the point is to learn, across
# many volunteers' actual Apple GPUs, whether this 10-core M3's 2^26 optimum
# generalises or whether an M3 Max picks something else, without having to
# reproduce every machine by hand. Gated on there having BEEN a slab decision
# worth reporting, so the below-trigger majority does not dilute the aggregate.
_boinc_marker = "    /* The A/B record path leaves no other trace"
assert _boinc_marker in src, 'run_pipeline body shape changed'
src = src.replace(_boinc_marker, """#ifdef HAVE_BOINC
    if (cfg->slab_j ||
        slab_perf_jmax(cfg->logI, cfg->log_region, cfg->J) != 0xffffffffu)
        fprintf(stderr, "BOINC: slab plan: %u rows/slab, %u slab%s%s\\n",
                plan.jmax, plan.nslab, plan.nslab == 1 ? "" : "s",
                calibrated_j ? " (auto-calibrated)"
                : cfg->slab_j ? " (--slab-j forced)" : " (static default)");
#endif
""" + _boinc_marker, 1)
print('  status line reports who decided; BOINC slab-plan line added')

# cofq_init now derives curves-per-round from the launch bound, so it needs to
# know whether the caller chose the curve count and what round count to keep
# the budget against. It reports a derived round count in Q->ecm_rounds, which
# every flush must then use instead of cfg->cof_rounds -- otherwise the derived
# curves would run against the caller's rounds and quietly shrink the budget.
_ci_pairs = [
    ("""                                   cfg->cof_limbs0, cfg->cof_limbs))""",
     """                                   cfg->cof_limbs0, cfg->cof_limbs,
                                   cfg->ecm_curves_set,
                                   (uint32_t)cfg->cof_rounds))"""),
    ("""                        cfg->cof_limbs0, cfg->cof_limbs);""",
     """                        cfg->cof_limbs0, cfg->cof_limbs,
                        cfg->ecm_curves_set, (uint32_t)cfg->cof_rounds);"""),
]
for _a, _b in _ci_pairs:
    assert src.count(_a) == 1, 'cofq_init call site shape changed: ' + _a.strip()
    src = src.replace(_a, _b, 1)
print('  both cofq_init call sites carry the derivation arguments')

# The ASYNC flush runs in a std::thread whose capture list is fixed, and `Q`
# there is a POINTER parameter of cofa_start, not the local the other sites
# see. Read the derived count before the thread is created and capture it by
# value: `Q->ecm_rounds` inside the lambda would be an uncaptured variable
# (it does not compile) and capturing Q itself would hand a second thread a
# pointer the q loop is still writing through.
# upstream dae6cca's three new messages name CUDA, and this binary is not
# CUDA. 9z-g scanned every fprintf(stderr, ...) in the port and got the count
# to zero; these are the first regression of that since, and they arrived
# through shared code rather than through anything written here. Metal-side
# only -- pipeline.cuh keeps its own wording. Anchored on fragments with no
# escapes in them: the message is split across source lines and spelling the
# join needs a backslash-n that has to survive being written BY a generator
# INTO a generator, which it did not the first time.
for _o, _n in [("cannot select CUDA\"", "cannot select Metal\""),
               ("cannot read the current CUDA device",
                "cannot read the current Metal device"),
               ("instance %d: cannot select CUDA device %d",
                "instance %d: cannot select Metal device %d")]:
    assert src.count(_o) == 1, 'device-selection message shape changed: ' + _o
    src = src.replace(_o, _n, 1)
print('  device-selection messages say Metal, not CUDA')

# upstream dae6cca added `dev` to this capture list, which broke the anchor --
# loudly, which is the point of asserting it. The device handover itself is
# INERT here and is kept rather than stripped: Apple silicon exposes one Metal
# device, metal_rt holds it in a process-global set once under g_lock, and
# mtlGetDevice always answers 0 while mtlSetDevice(0) always succeeds. So the
# multi-GPU hazard it fixes cannot arise, the generated code costs one lock
# acquisition per flush thread, and carrying shared code unchanged is worth
# more than deleting a branch that does nothing.
_async_old = """    A->th = new std::thread([A, O, cfg, blocks, fr, cs, dev]() {"""
_async_new = """    const int ecm_rounds = (int)Q->ecm_rounds;
    A->th = new std::thread([A, O, cfg, blocks, fr, cs, dev, ecm_rounds]() {"""
assert src.count(_async_old) == 1, 'cofa_start thread shape changed'

# THE HANDOVER NEEDS A BARRIER ON METAL, and nothing in the shared code says
# so. k_cof_enqueue runs on the q loop's stream and fills exactly the buffers
# the flush thread is about to read on g_cof_stream. CUDA orders those two
# through the runtime's own rules for the legacy default stream; two
# MTLCommandQueues have NO ordering at all, and Metal's hazard tracking is
# per-queue -- so without this the flush reads whatever the enqueue has
# managed to write so far. The failure is silent: the records are simply not
# there, the band exits 0, and only the relation count says anything. Once per
# flush, i.e. about every 70 special-q.
#
# Reported rather than returned: cofa_start is void, and metal_rt's sticky
# last error survives to the COF_FLUSH_CK(mtlGetLastError()) inside the flush,
# which is what turns it into this flush's rc.
_bar_old = """    A->F = *Q;                        /* full buffers, stats, steering   */"""
_bar_new = """    {   /* Metal only: see metal/gen_pipeline_host.py. The q loop's stream
         * and g_cof_stream are separate command queues with no ordering
         * between them, and the enqueue below has just written the buffers
         * this flush is about to read. */
        const mtlError_t _e = mtlDeviceSynchronize();
        if (_e != mtlSuccess)
            fprintf(stderr, "  pipeline: the cofactor handover barrier failed:"
                    " %s; the flush will report it\\n", mtlGetErrorString(_e));
    }
    A->F = *Q;                        /* full buffers, stats, steering   */"""
assert src.count(_bar_old) == 1, 'cofa_start handover shape changed'
src = src.replace(_bar_old, _bar_new, 1)
print('  cofactor handover barrier inserted (Metal has no cross-queue order)')

src = src.replace(_async_old, _async_new, 1)
_ar_old = """                           cfg->cof_rounds, cfg->cof_budget, cfg->cof_chunk,
                           blocks, cfg->threads, fr);"""
assert src.count(_ar_old) == 1, 'async cofq_flush rounds argument shape changed'
src = src.replace(_ar_old, """                           ecm_rounds, cfg->cof_budget, cfg->cof_chunk,
                           blocks, cfg->threads, fr);""", 1)

_n = src.count("cfg->cof_rounds, cfg->cof_budget")
assert _n == 4, 'expected 4 inline cofq_flush sites, found %d' % _n
src = src.replace("cfg->cof_rounds, cfg->cof_budget",
                  "(int)Q.ecm_rounds, cfg->cof_budget")
# Two sites wrap the argument onto its own line and the pattern above misses
# them. Matched separately and ASSERTED, because a flush left on the caller's
# rounds would run the derived curve count against the wrong round count and
# silently shrink the curve budget -- the exact failure this is meant to avoid.
for _wrapped, _fixed in (
        ("cfg->lim, cfg->lpb, cfg->cof_rounds,",
         "cfg->lim, cfg->lpb, (int)Q.ecm_rounds,"),):
    assert src.count(_wrapped) == 1, 'wrapped cofq_flush rounds argument: ' + _wrapped
    src = src.replace(_wrapped, _fixed, 1)
    _n += 1
assert _n == 5, 'expected 5 cofq_flush sites, found %d' % _n
# Nothing may be left asking cofq_flush for the CALLER's round count.
assert 'cof_rounds, cfg->cof_budget' not in src, \
    'a cofq_flush site still passes cfg->cof_rounds'
print('  %d cofq_flush site(s) use the derived round count' % _n)

# cofq_init has TWO call sites now: the queue itself and the async spare
# (upstream a025c32). Both take the derivation's two extra arguments, and the
# spare must agree with the real queue or a flush that runs on it would use a
# different curve schedule.
_ci_sites = src.count("""                                   cfg->cof_limbs0, cfg->cof_limbs))""")
assert _ci_sites == 0, 'cofq_init anchors already applied'



# ---- widen the warm-up launch's integer literals --------------------------
# k_transform declares a0/a1/b0/b1 as int64_t. CUDA's launch converts the
# literals `1, 0, 0, 1` implicitly at the call site, so the CUDA build is
# correct as written; Metal binds by VALUE and takes the literal's own width,
# so it bound 4 bytes for an argument the kernel reads as 8 -- the upper half
# being whatever followed. Harmless in practice only because this warm-up
# passes n = 0u and the loop never runs, but it is a real host/device type
# mismatch and Metal's validation layer refuses to dispatch it:
#
#   Compute Function(k_transform_1): argument a0[0] from Buffer(6) with
#   offset(0) and length(4) has space for 4 bytes, but argument has a
#   length(8).
#
# Metal-side only; pipeline.cuh is untouched.
_w_old = "cfg->logI, cfg->J, 1, 0, 0, 1, S1.d_nproj, S1.d_nlost, S1.walk_cur);"
_w_new = ("cfg->logI, cfg->J, (int64_t)1, (int64_t)0, (int64_t)0, (int64_t)1,"
          " S1.d_nproj, S1.d_nlost, S1.walk_cur);")
assert _w_old in src, 'warm-up launch shape changed'
src = src.replace(_w_old, _w_new, 1)
print('  warm-up k_transform literals widened to int64_t')

# The memory-diagnostic warning names the API too.
_md_old = '"warning: CUDA memory diagnostic \'%s\' unavailable: %s\\n"'
_md_new = '"warning: Metal memory diagnostic \'%s\' unavailable: %s\\n"'
assert _md_old in src, 'memory-diagnostic warning shape changed'
src = src.replace(_md_old, _md_new, 1)
print('  memory-diagnostic warning names Metal')

# The resume fix for pipe_progress_fraction used to live here, as a Metal-only
# rewrite. It is now in pipeline.cuh itself (plan 9z-n) because the bug was
# reported against the CUDA build and belongs to every port, so there is
# nothing left for this generator to do about it.
#
# There is no compile-time control either: this tree treats ANY non-empty DEFS
# as a PRICING build and refuses --relations from it, which a resume gate needs
# -- so a deliberately-broken control binary cannot be built that way, and
# should not be. progresscheck.sh discriminates without one.


# ---- defaulted kernel arguments become explicit ones ----------------------
# See portlib.pad_launch_defaults: Metal has no default argument: every
# parameter a kernel declares must be bound at dispatch, so each omitted
# trailing argument is filled in here from the KERNEL'S own default rather
# than retyped per call site. The assert inside also checks that no launch
# passes more arguments than the kernel declares -- which is how a kernel
# gaining a parameter in the middle would be caught.
_sigs = kernel_param_defaults(['bench/bench_kernels.cu', 'bench/td.cuh',
                               'bench/pipeline.cuh'])
src, _npad, _bad = pad_launch_defaults(src, _sigs, allow_unresolved=('apply_kern',))
assert not _bad, 'launch through a name this cannot resolve: %s' % ', '.join(sorted(_bad))
print('  %d launch(es) had defaulted arguments filled in' % _npad)

# apply_kern is launched through a VARIABLE, so the padder cannot resolve it;
# its argument count is checked here instead. k_apply's own default (lut_shift)
# is passed explicitly at that site by pipeline.cuh itself.
_ak = src.index('MTL_LAUNCH_NAMED(apply_kern,')
_ake = src.index(');', _ak)
assert len(_split_args(src[_ak + len('MTL_LAUNCH_NAMED('):_ake], '([{', ')]}')) - 5 \
       == len(_sigs['k_apply']), 'the apply_kern launch does not pass every k_apply argument'

open(OUT, 'w').write(src)
print('wrote %s (%d lines, %d launches rewritten)' % (OUT, src.count('\n'), nl))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*', src)))
if left: print('  UNRENAMED cuda* remaining:', ' '.join(left))
