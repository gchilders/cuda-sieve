/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * A stub of the BOINC client API, enough to compile boinc_support.cpp with
 * -DHAVE_BOINC on a machine that has no BOINC SDK installed. It exists for one
 * reason: the fraction-done path is COMPILED OUT of every default build
 * (HAVE_BOINC defaults to 0), so the one bug this port most needs to not
 * reintroduce -- a throwaway calibration band pinning a volunteer's task at
 * 99% for hours -- cannot be reached by any ordinary test. The HIP port found
 * that one in the field. This stub lets a gate reach it at desk.
 *
 * NOT a BOINC implementation and never linked into a shipped binary: only
 * metal/boinc_progress_test.cpp compiles against it.
 */
#ifndef CUDA_SIEVE_BOINC_STUB_H
#define CUDA_SIEVE_BOINC_STUB_H

#include <string>

struct APP_INIT_DATA {
    int  gpu_device_num;
    char gpu_type[64];
};
struct BOINC_OPTIONS {
    int normal_thread_priority, main_program, check_heartbeat;
    int handle_process_control, send_status_msgs, direct_process_action;
    int multi_thread, multi_process;
};

/* The recorder the test reads. Every fraction-done value that survives
 * boinc_support.cpp's own filtering lands here, in order. */
extern double stub_reported[256];
extern int    stub_nreported;
extern int    stub_fraction_rc;
extern int    stub_standalone;          /* boinc_is_standalone()'s answer   */
extern int    stub_ntempexit;           /* boinc_temporary_exit() call count */
extern int    stub_tempexit_delay;
extern const char *stub_tempexit_reason;

int  boinc_init(void);
int  boinc_init_options(BOINC_OPTIONS *opt);
void boinc_options_defaults(BOINC_OPTIONS &opt);
int  boinc_get_init_data(APP_INIT_DATA &aid);
int  boinc_is_standalone(void);
int  boinc_resolve_filename_s(const char *virt, std::string &phys);
int  boinc_fraction_done(double f);
int  boinc_finish(int status);
void boinc_exit(int status);
int  boinc_temporary_exit(int delay, const char *reason, bool is_notice);

#endif
