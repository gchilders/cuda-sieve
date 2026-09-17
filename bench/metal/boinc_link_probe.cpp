/* Phase 9 link probe: does bench's BOINC integration compile and link against
 * a REAL libboinc_api/libboinc, and does the init path run?
 *
 * This is the real-library counterpart to `make -f Makefile.metal boinccheck`,
 * which drives the same `boinc_support.cpp` against a hand-written stub. The
 * stub proves the progress logic; this proves the link. Neither replaces the
 * other -- the stub gate runs everywhere with no SDK installed, and this one
 * needs a BOINC tree built on the machine.
 *
 * There is deliberately no make target: the library prefix is machine-specific.
 * Build it by hand, with BOINCDIR pointing at a BOINC `make install` prefix:
 *
 *   clang++ -O2 -std=c++17 -Wall -Wextra -mmacosx-version-min=13.0 \
 *       -DHAVE_BOINC -I. -I$BOINCDIR/include/boinc \
 *       metal/boinc_link_probe.cpp boinc_support.cpp -o /tmp/boinc_link_probe \
 *       -L$BOINCDIR/lib -lboinc_api -lboinc -lpthread
 *
 * Then check `otool -L` names no BOINC dylib, `otool -l | grep minos` says
 * 13.0, and run it with and without an argument. See plan section 9a for how
 * that BOINC tree must be configured -- three of the four flags are not
 * optional.
 */
#include "bench.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    (void)argv;
    /* argc-gated rather than `if (0)`, so the optimiser cannot delete the
     * reference and leave boinc_api.o unpulled from the static archive. A
     * link that never had to resolve boinc_init_parallel() proves nothing.
     * Gated at all because bench_boinc_finish() does NOT return: boinc_finish()
     * exits the process, which is the behaviour this probe documents. */
    if (argc > 1) {
        printf("init rc=%d\n", bench_boinc_init());
        /* Same reason as finish: the real boinc_temporary_exit does not
         * return either, so it is referenced from the same dead-at-runtime
         * but live-at-link branch. What is being proved here is that the
         * symbol resolves against the real archive. */
        bench_boinc_temporary_exit(600, "link probe");
        printf("finish rc=%d (NOT REACHED: boinc_finish exits)\n",
               bench_boinc_finish(BENCH_OUTCOME_OK, 0));
    }
    printf("is_managed=%d gpu_device=%d\n",
           bench_boinc_is_managed(), bench_boinc_gpu_device());
    bench_boinc_progress_suspend(1);
    bench_boinc_fraction_done(0.5);
    bench_boinc_progress_suspend(0);
    const char *p = NULL;
    printf("resolve rc=%d (want -1: refused before init)\n",
           bench_boinc_resolve_path("--fb1", "c183.fb1", &p));
    printf("LINK OK\n");
    return 0;
}
