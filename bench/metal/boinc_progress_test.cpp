/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Gate for the one BOINC hazard the slab auto-calibration introduces.
 *
 * Calibration runs throwaway single-q bands through the same run_pipeline_impl
 * that reports progress. A one-q band reads as 100% done the moment its single
 * q retires; that is clamped to 0.99, and because BOINC reports must be
 * NONDECREASING, 0.99 then becomes the floor for the entire workunit. The HIP
 * port shipped exactly this and the field symptom was a task pinned at 99%
 * within seconds of starting and staying there for hours.
 *
 * The path is compiled out of every default build (HAVE_BOINC defaults to 0),
 * so nothing else in the tree can reach it. This compiles boinc_support.cpp
 * with -DHAVE_BOINC against metal/boinc_stub/ and drives it directly.
 *
 * The monotonic high-water mark is a static with no reset -- by design, it is
 * a per-task invariant -- so each case gets its OWN PROCESS. Same discipline
 * as argbufcheck's residency control, and for the same reason: a control that
 * shares mutable state with the case it is controlling proves nothing.
 */
#include "bench.h"
#include "metal/boinc_stub/boinc_api.h"
#include <cstdio>
#include <cstring>

static int failures;

static void ck(int cond, const char *what)
{
    printf("%-6s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) failures++;
}

/* What a calibration pass does to the progress estimator: run_pipeline_impl
 * reports fraction-done from its own arguments, and with one q in and one q
 * retired that is 1/1, clamped to 0.99 by the caller. */
static void throwaway_calibration_band(int suspended)
{
    if (suspended) bench_boinc_progress_suspend(1);
    bench_boinc_fraction_done(0.99);
    if (suspended) bench_boinc_progress_suspend(0);
}

int main(int argc, char **argv)
{
    const int suspended = argc > 1 && !strcmp(argv[1], "suspended");

    if (bench_boinc_init()) { printf("FAIL   stub bench_boinc_init\n"); return 1; }
    /* Init reports 0.0 itself; that is the real task's genuine starting point,
     * not something calibration did. Start counting after it. */
    stub_nreported = 0;

    throwaway_calibration_band(suspended);

    if (!suspended) {
        /* CONTROL: this is the shipped-and-broken behaviour. It must fail in
         * the specific way the field saw, or the gate below is vacuous. */
        ck(stub_nreported == 1 && stub_reported[0] == 0.99,
           "control: an unsuspended calibration band reports 0.99");
        stub_nreported = 0;
        bench_boinc_fraction_done(0.004);   /* real band's first real report */
        ck(stub_nreported == 0,
           "control: the real band's 0.4% is then swallowed by the 0.99 floor");
        bench_boinc_fraction_done(0.50);
        ck(stub_nreported == 0,
           "control: so is 50% -- the task is pinned until it truly passes 99%");
        printf("\ncontrol reproduced the field bug, so the gate below is real\n");
        return failures;
    }

    /* THE GATE: same band, suspended. */
    ck(stub_nreported == 0, "suspended calibration reports never reach BOINC");

    bench_boinc_fraction_done(0.004);
    ck(stub_nreported == 1 && stub_reported[0] == 0.004,
       "the real band's first report still starts from 0.4%, not 99%");

    bench_boinc_fraction_done(0.50);
    ck(stub_nreported == 2 && stub_reported[1] == 0.50,
       "and progress keeps advancing normally afterwards");

    /* The invariant the suspend must NOT break: still nondecreasing. */
    bench_boinc_fraction_done(0.25);
    ck(stub_nreported == 2,
       "monotonicity is preserved -- a backwards report is still dropped");

    /* ---- transient failure: reschedule, do not burn the workunit ---------
     *
     * A field task exited 1 with "this process sees no Metal device" on a host
     * whose client had already assigned an apple_gpu: the GPU was there and
     * the process could not reach it, which on macOS means no GUI login
     * session. That is a property of the moment, not of the host.
     *
     * Two halves, and the STANDALONE one matters as much: the wrapper must
     * RETURN there, because a HAVE_BOINC binary run from a terminal must still
     * reach its own error path rather than vanishing. The real
     * boinc_temporary_exit never returns, so this can only be tested against
     * the stub -- which is the entire reason the stub exists. */
    stub_standalone = 1;
    stub_ntempexit = 0;
    bench_boinc_temporary_exit(600, "no GPU");
    ck(stub_ntempexit == 0,
       "standalone: no temporary exit, and the wrapper returns to its caller");

    stub_standalone = 0;                 /* now pretend a client is managing us */
    bench_boinc_temporary_exit(600, "no Metal device is visible to this process");
    ck(stub_ntempexit == 1 && stub_tempexit_delay == 600,
       "managed: the client is asked to retry, with the delay it was given");
    ck(stub_tempexit_reason && strstr(stub_tempexit_reason, "no Metal device"),
       "managed: and the reason reaches BOINC, not just stderr");
    stub_standalone = 1;

    return failures;
}
