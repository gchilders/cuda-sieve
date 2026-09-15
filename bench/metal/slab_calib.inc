/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Startup slab-size auto-calibration, ported from the HIP port's
 * pipeline_hip.cuh (commits d3d9e7d, 3483e17, eb72ede, 3b7498a). Hand-written
 * rather than generated: gen_pipeline_host.py splices only the CALL, because a
 * hundred lines of new logic injected by regex is the kind of thing this port
 * has already been bitten by once.
 *
 * Included from pipeline_host.inc AFTER run_pipeline_impl and BEFORE
 * run_pipeline, since it calls the former and is called by the latter.
 */

/* Candidate slab sizes, as ABSOLUTE position counts.
 *
 * The HIP port probes {2^27, 2^28, 2^29} because gfx1103 measured 2^27. This
 * build probes one octave lower because THIS hardware measured lower: plan 8c
 * swept the production c183 geometry and found a bracketed interior minimum at
 * 2^26 positions -- 558.4 ms against 572.6 at 2^27 and 592.5 at 2^25, with
 * 994.2 ms for the unslabbed 2^29 CUDA was tuned for. So the three candidates
 * bracket this build's own static default rather than sitting above it, and
 * unlike the HIP port's list the measured winner is the INTERIOR point, which
 * is what makes a three-point probe able to confirm or move it.
 *
 * Absolute, not region-relative, matching the HIP port -- and carrying its
 * caveat, which applies here unchanged: these were measured at this build's
 * default --region 13 only, and calibration does not re-derive them against a
 * caller's own --region. A job run at another --region calibrates among the
 * same three absolute sizes. Flagged rather than silently assumed correct.
 * (The STATIC default does track --region, via SLAB_PERF_REGIONS << log_region
 * in slab.h, so the two disagree away from --region 13. Calibration wins when
 * it runs, and it only runs where it is eligible.) */
static constexpr uint32_t SLAB_CAND_LOG2[3] = {25u, 26u, 27u};

/* Metal's ceiling is NOT the HIP port's. gfx1103 had a confirmed
 * kernel-launch-hangs-the-OS boundary near 2^30 (Kernel-Power id 41), and its
 * SLAB_PERF_MAX_SAFE_LOG2 encodes a crash. **No such boundary has been
 * observed on Apple silicon here**, and it would be wrong to imply one: the
 * unslabbed 2^29 case runs to completion on this box and is in plan 8c's own
 * table. What Apple does have is a display watchdog that can kill a long
 * kernel (see plan 8b), so a ceiling is still prudence rather than paranoia --
 * it is just prudence about a timeout, not about a crash. Set to the largest
 * size this port has actually run, and no higher. */
#define SLAB_CALIB_MAX_SAFE_LOG2 29u
static_assert(SLAB_CAND_LOG2[2] <= SLAB_CALIB_MAX_SAFE_LOG2,
              "calibration candidate exceeds the largest slab size this port"
              " has actually run -- see metal/slab_calib.inc");

/* Auto-calibrate the j-slab size against the real first special-q, rather than
 * trusting a single build-wide constant. SLAB_PERF_REGIONS (slab.h) is itself
 * a measurement on ONE piece of hardware -- a 10-core M3 -- and an M3 Max with
 * four times the cores and far more bandwidth is a different point on the same
 * curve. There is no way to know without asking the hardware in front of it,
 * and doing so costs one extra special-q at startup, negligible against a run
 * measured in hours.
 *
 * Skipped entirely when the caller passes an explicit --slab-j (a
 * regression/testing knob, not something auto-tuning should override) or when
 * the geometry is too small to trigger performance slabbing at all --
 * calibrating a sieve that already fits under the trigger in one pass would
 * only add the extra q's cost for no possible benefit. */
static uint32_t calibrate_slab_rows(const fb_t *fb1, const fb_t *fbs1,
                                    const fb_t *fb0, const fb_t *fbs0,
                                    const qsel_t *qlist, uint32_t nq,
                                    const poly_t *POLY,
                                    const bench_cfg_t *cfg, uint32_t pmax)
{
    const uint32_t I = 1u << cfg->logI;
    const uint32_t quantum = slab_row_quantum(cfg->logI);
    double best_ms = -1.0;
    uint32_t best_rows = 0, best_log2 = 0;
    uint32_t tested_jmax[3] = {0, 0, 0};
    int ncand = 0, ntested = 0;

    if (cfg->slab_j || !nq || !quantum ||
        slab_perf_jmax(cfg->logI, cfg->log_region, cfg->J) == 0xffffffffu)
        return 0;

    printf("\n  slab auto-calibration: probing 2^25/2^26/2^27-position slabs"
           " against q=%llu (this run's first special-q)...\n",
           (unsigned long long)qlist[0].q);
    for (int i = 0; i < 3; i++) {
        const uint64_t rows64 = ((uint64_t)1 << SLAB_CAND_LOG2[i]) / I;
        uint32_t rows = rows64 > 0xffffffffull ? 0xffffffffu : (uint32_t)rows64;
        slab_plan_t cplan;
        bench_cfg_t ccfg;
        double t0, t1, ms;
        int k, dup = 0, rc_cal;

        rows -= rows % quantum;
        if (!rows || slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                                    rows, &cplan))
            continue;

        /* slab_make_plan() clamps a forced row count down to J (near the
         * trigger, J itself can be smaller than a candidate's raw target), so
         * two nominally different candidates can land on the identical actual
         * plan. Skip the repeat rather than spending a whole extra GPU pass
         * measuring a config already timed under a different name. */
        for (k = 0; k < ntested; k++)
            if (tested_jmax[k] == cplan.jmax) { dup = 1; break; }
        if (dup) {
            printf("    2^%u: same effective slab size as an earlier"
                   " candidate (%u rows/slab), skipped\n",
                   SLAB_CAND_LOG2[i], cplan.jmax);
            continue;
        }
        tested_jmax[ntested++] = cplan.jmax;

        /* Throwaway pass: no output path means no relation/candidate file is
         * ever opened, so this cannot write, checkpoint, or otherwise disturb
         * real state -- it only spends GPU time. qgen is NULL and nq is 1, so
         * the loop processes exactly qlist[0] and stops.
         *
         * cofactor is forced OFF here, not merely output-less: inline ECM/rho
         * runs regardless of whether a relations path is set, and it is the
         * wrong thing to be timing anyway. Plan 8d measured the cofactor stage
         * FLAT to within 2% across a 16x range of threadgroup sizes and 115x
         * of counts -- it is critical-path bound and does not depend on slab
         * geometry -- so paying for it three extra times would buy nothing but
         * a tripled ECM cost, and in cofcheck.sh it would multiply the
         * cofactor-diagnostic lines a caller greps for exactly once. */
        ccfg = *cfg;
        ccfg.relations = NULL; ccfg.candidates = NULL;
        ccfg.stopfile  = NULL; ccfg.logpath   = NULL;
        ccfg.cofgate   = NULL; ccfg.emit_cof  = NULL;
        ccfg.cofactor  = 0;
        ccfg.td_verify = 0;
        ccfg.nq_max = 1; ccfg.slab_j = rows;

        /* Dispatch on cplan.enabled exactly like run_pipeline() does, not
         * unconditionally <true>: near the trigger a candidate's forced row
         * count can clamp to an unslabbed plan, and the real run would time
         * THAT with the <false> specialization. Timing every candidate as
         * <true> would bias a winner measured with slab overhead the real run
         * for that plan never pays.
         *
         * g_runlog_quiet suppresses only this throwaway pass's mid-band
         * warnings: they are real conditions but tell a reader nothing about
         * the task that actually ran, since this pass's output is discarded
         * either way. Reset unconditionally right after, so the real run is
         * never silenced by a candidate that happened to warn.
         *
         * bench_boinc_progress_suspend is NOT cosmetic. run_pipeline_impl
         * reports fraction-done from its own arguments, and with qgen NULL and
         * nq 1 that is 1/1 the moment the single calibration q retires --
         * clamped to 0.99 and, because BOINC reports must be nondecreasing, it
         * becomes the floor for the REST OF THE WORKUNIT. The HIP port shipped
         * exactly that and the field symptom was a task pinned at 99% within
         * seconds and staying there for hours. Suspending drops the throwaway
         * reports WITHOUT advancing the high-water mark. Gated by
         * metal/boinc_progress_test.cpp, whose control reproduces the bug. */
        t0 = host_ms();
        g_runlog_quiet = 1;
        bench_boinc_progress_suspend(1);
        rc_cal = cplan.enabled
            ? run_pipeline_impl<true>(fb1, fbs1, fb0, fbs0, qlist, 1, NULL,
                                      POLY, &ccfg, &cplan)
            : run_pipeline_impl<false>(fb1, fbs1, fb0, fbs0, qlist, 1, NULL,
                                       POLY, &ccfg, &cplan);
        bench_boinc_progress_suspend(0);
        g_runlog_quiet = 0;
        if (rc_cal)
            continue;
        t1 = host_ms();
        ms = t1 - t0;
        /* cplan.jmax, not the pre-clamp `rows`: near the trigger the two can
         * differ, and it is what was actually measured that must be reported
         * and, if it wins, reused for the real run. */
        printf("    2^%u (%u rows/slab): %.1f ms\n",
               SLAB_CAND_LOG2[i], cplan.jmax, ms);
        ncand++;
        if (best_ms < 0 || ms < best_ms) {
            best_ms = ms; best_rows = cplan.jmax; best_log2 = SLAB_CAND_LOG2[i];
        }
    }
    if (ncand < 2 || !best_rows) {
        printf("  slab auto-calibration: inconclusive, using the built-in"
               " default\n");
        return 0;
    }
    printf("  slab auto-calibration: selected 2^%u (%u rows/slab)\n",
           best_log2, best_rows);
    return best_rows;
}
