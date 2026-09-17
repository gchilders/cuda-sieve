/* SPDX-License-Identifier: LGPL-2.1-or-later -- see boinc_api.h in this dir. */
#include "boinc_api.h"
#include <cstdio>

double stub_reported[256];
int    stub_nreported;
int    stub_fraction_rc;
/* 1 by default, so every existing case keeps the standalone behaviour it was
 * written against; the transient-exit case flips it to test the managed half. */
int    stub_standalone = 1;
int    stub_ntempexit;
int    stub_tempexit_delay;
const char *stub_tempexit_reason;

int  boinc_init(void) { return 0; }
int  boinc_init_options(BOINC_OPTIONS *) { return 0; }
void boinc_options_defaults(BOINC_OPTIONS &opt) { opt = BOINC_OPTIONS(); }
int  boinc_get_init_data(APP_INIT_DATA &aid)
{
    /* A plausible assignment: device 0, and a gpu_type the caller accepts. */
    aid.gpu_device_num = 0;
    aid.gpu_type[0] = '\0';
    return 0;
}
int  boinc_is_standalone(void) { return stub_standalone; }
int  boinc_resolve_filename_s(const char *virt, std::string &phys)
{ phys = virt ? virt : ""; return 0; }
int  boinc_fraction_done(double f)
{
    if (stub_nreported < 256) stub_reported[stub_nreported++] = f;
    return stub_fraction_rc;
}
int  boinc_finish(int) { return 0; }
void boinc_exit(int) {}
/* The real one does not return; this records the call so the test can assert
 * it happened AND keep running. That difference is why the assertion below is
 * about reaching this function, not about the process dying. */
int  boinc_temporary_exit(int delay, const char *reason, bool)
{
    stub_ntempexit++;
    stub_tempexit_delay = delay;
    stub_tempexit_reason = reason;
    return 0;
}
