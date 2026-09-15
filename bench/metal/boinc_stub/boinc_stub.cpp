/* SPDX-License-Identifier: LGPL-2.1-or-later -- see boinc_api.h in this dir. */
#include "boinc_api.h"
#include <cstdio>

double stub_reported[256];
int    stub_nreported;
int    stub_fraction_rc;

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
int  boinc_is_standalone(void) { return 1; }
int  boinc_resolve_filename_s(const char *virt, std::string &phys)
{ phys = virt ? virt : ""; return 0; }
int  boinc_fraction_done(double f)
{
    if (stub_nreported < 256) stub_reported[stub_nreported++] = f;
    return stub_fraction_rc;
}
int  boinc_finish(int) { return 0; }
void boinc_exit(int) {}
int  boinc_temporary_exit(int, const char *, bool) { return 0; }
