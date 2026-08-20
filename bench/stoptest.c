/* Regression test for the clean-stop hook in the platform layer.
 *
 * The hook replaced a direct signal(SIGINT)/signal(SIGTERM) pair in
 * pipeline.cuh, because signal(SIGTERM, ...) is accepted but never delivered
 * by the Win32 CRT -- the SIGTERM half of that pair was dead code on Windows
 * and there was no way to ask a running band to checkpoint and exit. What is
 * checked here is that moving it did not change the POSIX behaviour the sieve
 * has always had: both signals reach the callback, and removing the hook puts
 * the caller's previous disposition back.
 *
 * The Windows path is a console control handler, which cannot be exercised by
 * raise(); it needs a real CTRL_C_EVENT delivered to a console process group.
 * Rather than assert something untrue there, this skips.
 */
#include "platform.h"

#include <stdio.h>

#ifdef _WIN32
int main(void)
{
    printf("stoptest: skipped on Windows (the console control handler needs a"
           " real CTRL_C_EVENT, not raise())\n");
    return 0;
}
#else
#include <signal.h>

static volatile sig_atomic_t hits = 0;
static volatile sig_atomic_t prev_hit = 0;

static void on_stop(void) { hits++; }
static void prev_handler(int sig) { (void)sig; prev_hit = 1; }

static int fail(const char *what)
{
    fprintf(stderr, "stoptest: %s\n", what);
    return 1;
}

int main(void)
{
    /* Install a disposition first, so "remove() restores what was there"
     * is tested against something distinguishable from SIG_DFL. */
    if (signal(SIGTERM, prev_handler) == SIG_ERR)
        return fail("cannot install the prior SIGTERM handler");

    if (bench_stop_hook_install(on_stop))
        return fail("bench_stop_hook_install failed");

    raise(SIGTERM);
    raise(SIGINT);
    if (hits != 2)
        return fail("both SIGINT and SIGTERM must reach the stop callback");

    bench_stop_hook_remove();

    /* After removal the callback must be inert and the caller's handler live
     * again: run_pipeline restores signal state on every exit path, including
     * the CUDA-failure jump to done:, and a hook that stayed armed would fire
     * against a torn-down pipeline. */
    raise(SIGTERM);
    if (hits != 2)
        return fail("the stop callback still ran after bench_stop_hook_remove");
    if (!prev_hit)
        return fail("the previous SIGTERM disposition was not restored");

    printf("stoptest: SIGINT and SIGTERM reach the stop hook; remove()"
           " restores the prior handler\n");
    return 0;
}
#endif
