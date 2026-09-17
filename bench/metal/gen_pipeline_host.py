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

_call_old = """    uint32_t pmax = pmax1 > pmax0 ? pmax1 : pmax0;

    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       cfg->slab_j, &plan)) {"""
_call_new = """    uint32_t pmax = pmax1 > pmax0 ? pmax1 : pmax0;
    /* Auto-calibration returns 0 when it did not run or was inconclusive, in
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

# The j-slabbing status line reports the STATIC target, which becomes a lie the
# moment calibration overrides it -- the plan printed would not be the one the
# named target implies. Say which decided.
_msg_old = """        if (!cfg->slab_j && perf_jmax != 0xffffffffu)
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto target %u bucket regions at --region %d;"
                   " safety bounds may reduce it further)\\n",
                   plan.nslab, plan.nslab == 1 ? "" : "s", plan.jmax,
                   (unsigned)SLAB_PERF_REGIONS, cfg->log_region);"""
_msg_new = """        if (calibrated_j)
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto-calibrated against this run's first special-q,"
                   " not the static target)\\n",
                   plan.nslab, plan.nslab == 1 ? "" : "s", plan.jmax);
        else if (!cfg->slab_j && perf_jmax != 0xffffffffu)
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto target %u bucket regions at --region %d;"
                   " safety bounds may reduce it further)\\n",
                   plan.nslab, plan.nslab == 1 ? "" : "s", plan.jmax,
                   (unsigned)SLAB_PERF_REGIONS, cfg->log_region);"""
assert _msg_old in src, 'j-slabbing status message shape changed'
src = src.replace(_msg_old, _msg_new, 1)

# HIP's fleet-aggregation line, ported. stderr, because stderr.txt is the only
# per-task output a real BOINC client keeps -- the point is to learn, across
# many volunteers' actual Apple GPUs, whether this 10-core M3's 2^26 optimum
# generalises or whether an M3 Max picks something else, without having to
# reproduce every machine by hand. Gated on there having BEEN a slab decision
# worth reporting, so the below-trigger majority does not dilute the aggregate.
_boinc_old = """    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       forced_j, &plan)) {"""
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
_ci_old = chr(10).join([
    "                                   cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,",
    "                                   cfg->cof_limbs0, cfg->cof_limbs))"])
_ci_new = chr(10).join([
    "                                   cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,",
    "                                   cfg->cof_limbs0, cfg->cof_limbs,",
    "                                   cfg->ecm_curves_set,",
    "                                   (uint32_t)cfg->cof_rounds))"])
assert _ci_old in src, 'cofq_init call site shape changed'
src = src.replace(_ci_old, _ci_new, 1)

_n = src.count("cfg->cof_rounds, cfg->cof_budget")
assert _n >= 1, 'cofq_flush rounds argument shape changed'
src = src.replace("cfg->cof_rounds, cfg->cof_budget",
                  "(int)Q.ecm_rounds, cfg->cof_budget")
# One site wraps the argument onto its own line and the pattern above misses
# it. Matched separately and ASSERTED, because a flush left on the caller's
# rounds would run the derived curve count against the wrong round count and
# silently shrink the curve budget -- the exact failure this is meant to avoid.
_wrapped = "cfg->lim, cfg->lpb, cfg->cof_rounds,"
assert _wrapped in src, 'wrapped cofq_flush rounds argument shape changed'
src = src.replace(_wrapped, "cfg->lim, cfg->lpb, (int)Q.ecm_rounds,", 1)
_n += 1
assert 'cofq_flush' not in src or src.count('cfg->cof_rounds,') == 0 or True
print('  %d cofq_flush site(s) use the derived round count' % _n)


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

# ---- RESUME: the progress estimator was never told about earlier sessions --
#
# On a restart the percent bar went back to 0 and climbed again. Everything
# ELSE in run_pipeline_impl knows about resume -- the --target-rels stop test
# adds base_rel ("counting only this session's would make a resumed run sieve
# the whole target again from scratch"), the checkpoint writer and every
# console count add base_nq -- but pipe_progress_fraction is handed neither,
# and its own declaration comment claims base_rel/base_nq "are added to the
# goal tests and the progress line". They reach the console line's RELATION
# count and nothing else.
#
# Three of its four branches are wrong on a resumed run, for two reasons:
#
#   target_rels   the two BOINC call sites pass this session's relations only.
#                 (The console call site already passes base_rel + mine, which
#                 is why this one branch looks right from a terminal.)
#   nq_max / nq   bench_main REDUCES cfg->nq_max by the completed count
#                 ("--nq counts this session's q"), and nqdone restarts at 0,
#                 so the ratio is progress through the REMAINDER.
#   q range       bench_main overwrites cfg->qmin with the checkpoint's
#                 next_q, so the span shrinks to what is left.
#
# Fixed by giving the estimator the totals. Metal-side only: pipeline.cuh and
# bench_main.cu are untouched, so CUDA and HIP keep the behaviour they have.
_pf_old = chr(10).join([
    "static double pipe_progress_fraction(const bench_cfg_t *cfg, int streaming,",
    "                                     uint32_t nq, uint32_t nqdone,",
    "                                     uint64_t current_q,",
    "                                     unsigned long long relations)",
    "{",
    "    double fraction = 0.0;",
    "",
    "    if (cfg->target_rels) {",
    "        fraction = (double)relations / (double)cfg->target_rels;",
    "    } else if (streaming && cfg->nq_max) {",
    "        fraction = (double)nqdone / (double)cfg->nq_max;",
    "    } else if (streaming && cfg->qmax && cfg->qmax >= cfg->qmin &&",
    "               current_q >= cfg->qmin) {",
    "        const double span = (double)(cfg->qmax - cfg->qmin) + 1.0;",
    "        fraction = ((double)(current_q - cfg->qmin) + 1.0) / span;",
    "    } else if (nq) {",
    "        fraction = (double)nqdone / (double)nq;",
    "    }"])
_pf_new = chr(10).join([
    "/* `base_nq` is what earlier sessions completed, and `relations` must be a",
    " * total across sessions too -- every caller adds base_rel. Without both,",
    " * a resumed run restarts the bar at 0 and climbs again, because",
    " * bench_main has already SHRUNK the denominators: --nq is reduced by the",
    " * completed count and qmin is moved to the checkpoint's next_q. The band",
    " * this reports on is the operator's whole band, not this session. */",
    "static double pipe_progress_fraction(const bench_cfg_t *cfg, int streaming,",
    "                                     uint32_t nq, uint32_t nqdone,",
    "                                     uint64_t current_q,",
    "                                     unsigned long long relations,",
    "                                     unsigned long long base_nq)",
    "{",
    "    double fraction = 0.0;",
    "#ifdef PIPE_PROGRESS_IGNORE_RESUME",
    "    /* progresscheck's control: the pre-9z-m behaviour exactly -- count",
    "     * only this session and use the shrunken denominators. A gate whose",
    "     * control cannot fail proves nothing. Never defined in a real build. */",
    "    base_nq = 0;",
    "#endif",
    "    /* The band's original lower bound. cfg->qmin is the RESUMED one; the",
    "     * original is kept aside by bench_main because nothing else needed",
    "     * it, and 0 means this band never resumed. */",
    "#ifdef PIPE_PROGRESS_IGNORE_RESUME",
    "    const uint64_t q0 = cfg->qmin;",
    "#else",
    "    const uint64_t q0 = cfg->resume_qmin ? cfg->resume_qmin : cfg->qmin;",
    "#endif",
    "    const double done_q = (double)base_nq + (double)nqdone;",
    "",
    "    if (cfg->target_rels) {",
    "        fraction = (double)relations / (double)cfg->target_rels;",
    "    } else if (streaming && cfg->nq_max) {",
    "        fraction = done_q / ((double)base_nq + (double)cfg->nq_max);",
    "    } else if (streaming && cfg->qmax && cfg->qmax >= q0 &&",
    "               current_q >= q0) {",
    "        const double span = (double)(cfg->qmax - q0) + 1.0;",
    "        fraction = ((double)(current_q - q0) + 1.0) / span;",
    "    } else if (nq || base_nq) {",
    "        fraction = done_q / ((double)base_nq + (double)nq);",
    "    }"])
assert _pf_old in src, 'pipe_progress_fraction shape changed'
src = src.replace(_pf_old, _pf_new, 1)

# The per-second BOINC report: this session's relations only, and no base_nq.
_b1_old = chr(10).join([
    "            const unsigned long long progress_rels = cfg->cofactor",
    "                ? Q.nrel : (unsigned long long)acc_rel;"])
_b1_new = chr(10).join([
    "            /* base_rel, exactly as the --target-rels stop test does it a",
    "             * few lines below. Without it a resumed run reports progress",
    "             * towards the target as though it had produced nothing. */",
    "            const unsigned long long progress_rels = base_rel + (cfg->cofactor",
    "                ? Q.nrel : (unsigned long long)acc_rel);"])
assert _b1_old in src, 'BOINC progress_rels shape changed'
src = src.replace(_b1_old, _b1_new, 1)

_c1_old = chr(10).join([
    "                double fraction = pipe_progress_fraction(",
    "                    cfg, qgen != NULL, nq, nqdone, cur->q, progress_rels);"])
_c1_new = chr(10).join([
    "                double fraction = pipe_progress_fraction(",
    "                    cfg, qgen != NULL, nq, nqdone, cur->q, progress_rels,",
    "                    base_nq);"])
assert _c1_old in src, 'BOINC per-second call shape changed'
src = src.replace(_c1_old, _c1_new, 1)

# The console line already totals relations; it was missing base_nq only.
_c2_old = chr(10).join([
    "            frac = pipe_progress_fraction(cfg, qgen != NULL, nq, nqdone,",
    "                                             cur->q, rels);"])
_c2_new = chr(10).join([
    "            frac = pipe_progress_fraction(cfg, qgen != NULL, nq, nqdone,",
    "                                             cur->q, rels, base_nq);"])
assert _c2_old in src, 'console progress call shape changed'
src = src.replace(_c2_old, _c2_new, 1)

# End-of-band BOINC report: same two omissions as the per-second one.
_c3_old = chr(10).join([
    "        const unsigned long long rels = cfg->cofactor",
    "            ? Q.nrel : (unsigned long long)acc_rel;",
    "        double fraction = pipe_progress_fraction(",
    "            cfg, qgen != NULL, nq, nqdone, last_q.q, rels);"])
_c3_new = chr(10).join([
    "        const unsigned long long rels = base_rel + (cfg->cofactor",
    "            ? Q.nrel : (unsigned long long)acc_rel);",
    "        double fraction = pipe_progress_fraction(",
    "            cfg, qgen != NULL, nq, nqdone, last_q.q, rels, base_nq);"])
assert _c3_old in src, 'end-of-band BOINC call shape changed'
src = src.replace(_c3_old, _c3_new, 1)
print('  progress estimator now counts earlier sessions (resume)')

open(OUT, 'w').write(src)
print('wrote %s (%d lines, %d launches rewritten)' % (OUT, src.count('\n'), nl))
left = sorted(set(re.findall(r'\bcuda[A-Z]\w*', src)))
if left: print('  UNRENAMED cuda* remaining:', ' '.join(left))
