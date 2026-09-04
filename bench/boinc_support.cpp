/* Optional BOINC runtime integration for the CUDA sieve.
 *
 * This translation unit is always linked into the GPU executable. Without
 * HAVE_BOINC it reduces to small no-op wrappers, so the rest of the program
 * does not need preprocessor branches around every call site. With
 * HAVE_BOINC it owns BOINC initialisation/termination, logical filename
 * resolution, and monotonic fraction-done reporting.
 */
#include "bench.h"

#include <errno.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_BOINC
#include <string>
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#endif
/* APP_INIT_DATA comes from app_ipc.h, which boinc_api.h includes for C++
 * translation units. Including it directly here would put it ahead of
 * boinc_win.h on Windows builds, which BOINC's headers require first. */
#include "boinc_api.h"

struct resolved_path_node {
    char *path;
    struct resolved_path_node *next;
};

static struct resolved_path_node *resolved_paths;
enum boinc_runtime_state {
    BOINC_NOT_STARTED = 0,
    BOINC_INIT_ATTEMPTED,
    BOINC_READY
};
static enum boinc_runtime_state boinc_state;
static double last_fraction_done;
/* Non-zero while a throwaway pass is running; see bench_boinc_progress_suspend.
 * Not reset by bench_boinc_init: it is only ever set around a bracketed call
 * that clears it again, and a stale 1 would silence progress for the whole
 * run. */
static int progress_suspended;
static int fraction_warning_printed;

static void free_resolved_paths(void)
{
    struct resolved_path_node *p = resolved_paths;
    while (p) {
        struct resolved_path_node *next = p->next;
        free(p->path);
        free(p);
        p = next;
    }
    resolved_paths = NULL;
}

#ifndef _WIN32
/* boinc_resolve_filename_s() intentionally leaves a native Unix symlink as
 * the returned name because ordinary fopen() follows it.  This application
 * stages output as NAME.part and commits with rename(), however; renaming over
 * NAME would replace the BOINC link rather than its physical result file.
 * Follow any native symlink chain here so both ordinary I/O and the atomic
 * commit operate on the physical file in the project directory. */
static int follow_native_symlinks(const char *option, std::string *path)
{
    for (unsigned depth = 0; depth < 32; depth++) {
        struct stat st;

        if (lstat(path->c_str(), &st) != 0) {
            /* A generated output's physical target normally does not exist
             * yet. Reaching that target is successful resolution. */
            if (errno == ENOENT) return 0;
            fprintf(stderr,
                    "%s: cannot inspect BOINC-resolved path '%s': %s\n",
                    option ? option : "filename", path->c_str(),
                    strerror(errno));
            return -1;
        }
        if (!S_ISLNK(st.st_mode)) return 0;

        size_t cap = 256;
        std::string target;
        for (;;) {
            std::vector<char> buf(cap);
            const ssize_t n = readlink(path->c_str(), buf.data(), buf.size());
            if (n < 0) {
                fprintf(stderr,
                        "%s: cannot read BOINC filename link '%s': %s\n",
                        option ? option : "filename", path->c_str(),
                        strerror(errno));
                return -1;
            }
            if ((size_t)n < buf.size()) {
                target.assign(buf.data(), (size_t)n);
                break;
            }
            if (cap > SIZE_MAX / 2) {
                fprintf(stderr, "%s: BOINC filename link is too long\n",
                        option ? option : "filename");
                return -1;
            }
            cap *= 2;
        }

        if (!target.empty() && target[0] == '/') {
            *path = target;
        } else {
            const size_t slash = path->find_last_of('/');
            if (slash == std::string::npos)
                *path = target;
            else
                *path = path->substr(0, slash + 1) + target;
        }
    }

    fprintf(stderr, "%s: BOINC filename has too many symbolic-link levels\n",
            option ? option : "filename");
    return -1;
}
#endif
#endif

extern "C" int bench_boinc_init(void)
{
#ifdef HAVE_BOINC
    BOINC_OPTIONS options;
    int rc;

    boinc_options_defaults(options);
    /* BOINC runs CPU applications at idle priority on Windows by default.
     * GPU applications need normal host-thread priority so kernel launches and
     * completion handling are not starved while the CPUs are busy. */
    options.normal_thread_priority = 1;

    boinc_state = BOINC_INIT_ATTEMPTED;
    rc = boinc_init_options(&options);
    if (rc) {
        fprintf(stderr, "BOINC: boinc_init_options failed with status %d\n", rc);
        return rc;
    }
    boinc_state = BOINC_READY;
    last_fraction_done = 0.0;
    fraction_warning_printed = 0;
    rc = boinc_fraction_done(0.0);
    if (rc) {
        fprintf(stderr,
                "BOINC: initial fraction-done report failed with status %d\n",
                rc);
        fraction_warning_printed = 1;
    }
    printf("BOINC: API initialised (%s mode)\n",
           boinc_is_standalone() ? "standalone" : "client");
#endif
    return 0;
}

extern "C" int bench_boinc_gpu_device(void)
{
#ifdef HAVE_BOINC
    APP_INIT_DATA aid;

    /* The client's per-task GPU assignment lives in init_data.xml, and this is
     * the only mechanism that is not deprecated. The client ALSO appends
     * "--device N" to the command line, but only when the app version declares
     * an api_version below 7.5 (client/app_start.cpp:
     * `if (!app_version->api_version_at_least(7, 5))`, over a call marked
     * "NOTE: this is deprecated. Use app_init_data instead."). That test is on
     * the APP VERSION, not the client: a current client silently stops
     * supplying the argument as soon as the project declares a modern
     * api_version, and an application reading only the command line then runs
     * every task of a multi-GPU host on device 0.
     *
     * boinc_get_init_data() needs the runtime to have come up; it reads the
     * copy boinc_init_options() already parsed. */
    if (boinc_state != BOINC_READY) return -1;
    /* Every released boinc_get_init_data() returns 0 unconditionally. Checked
     * anyway so this does not read stale data if that contract ever changes. */
    if (boinc_get_init_data(aid)) return -1;

    /* -1 is BOINC's sentinel for "no GPU assigned", and it is set by
     * APP_INIT_DATA::clear() rather than by the parse, so it also holds on the
     * standalone path where init_data.xml does not exist at all. The client
     * writes the same -1 for a task whose app version claims no coprocessor. */
    if (aid.gpu_device_num < 0) return -1;

    /* The index is an ordinal within ONE coprocessor's device list, not a
     * machine-wide GPU number. An index taken from the wrong vendor's list
     * carries no information about this build's own device ordinals: acting
     * on it would run this task on a card the client never reserved for it,
     * colliding with whatever the client did place there -- the exact
     * failure this function exists to prevent. Refuse it and let this
     * build's default device stand, with the misconfiguration on the record
     * in the uploaded stderr.
     *
     * BENCH_HIP_BUILD (defined only by build_windows_hip.bat and
     * build_windows_hip_boinc.bat) selects the vendor string this build
     * expects: "ATI" is BOINC's own standardized vendor name for AMD GPUs
     * (see coproc.h -- kept from BOINC's pre-rebrand naming), "NVIDIA" for
     * the CUDA build. This #if, not a runtime check, because a single
     * binary is always built for one GPU vendor here -- there is no
     * multi-vendor bench.exe that needs to decide this at runtime. */
#if defined(BENCH_HIP_BUILD)
#define BENCH_GPU_VENDOR "ATI"
#else
#define BENCH_GPU_VENDOR "NVIDIA"
#endif
    if (aid.gpu_type[0] && strncmp(aid.gpu_type, BENCH_GPU_VENDOR,
                                    sizeof(BENCH_GPU_VENDOR) - 1) != 0) {
        fprintf(stderr,
                "BOINC: this is a %s application but the client assigned a"
                " '%s' device (index %d); ignoring the assignment. The app"
                " version's plan class is not a %s one.\n",
                BENCH_GPU_VENDOR, aid.gpu_type, aid.gpu_device_num,
                BENCH_GPU_VENDOR);
        return -1;
    }
    return aid.gpu_device_num;
#undef BENCH_GPU_VENDOR
#else
    return -1;
#endif
}

/* Dump the client's raw GPU assignment. Called ONLY when this process cannot
 * honour the assigned ordinal, to make the cause diagnosable from a host's
 * uploaded stderr -- never on a healthy task.
 *
 * gpu_opencl_dev_index is reported alongside gpu_device_num because they are
 * different numbers from different lists, and which one the client picked
 * says where the mismatch comes from. It is NOT a usable mapping into HIP's
 * ordinals either: the client sets it from opencl_prop.opencl_device_index
 * (client/gpu_opencl.cpp), its own index into a list flattened across every
 * OpenCL platform, so consuming it would mean reproducing BOINC's exact
 * enumeration order and filtering. Logged as evidence, not acted on. */
extern "C" void bench_boinc_log_gpu_assignment(void)
{
#ifdef HAVE_BOINC
    APP_INIT_DATA aid;

    if (boinc_state != BOINC_READY) return;
    if (boinc_get_init_data(aid)) return;
    fprintf(stderr,
            "BOINC: assignment was gpu_type='%s' device_num=%d"
            " opencl_dev_index=%d gpu_usage=%.3f\n",
            aid.gpu_type, aid.gpu_device_num, aid.gpu_opencl_dev_index,
            aid.gpu_usage);
#endif
}

extern "C" int bench_boinc_is_managed(void)
{
#ifdef HAVE_BOINC
    return boinc_state == BOINC_READY && !boinc_is_standalone();
#else
    return 0;
#endif
}

extern "C" int bench_boinc_resolve_path(const char *option,
                                         const char *logical_name,
                                         const char **resolved_name)
{
    if (!resolved_name || !logical_name || !*logical_name) {
        fprintf(stderr, "%s: empty filename\n", option ? option : "filename");
        return -1;
    }

#ifdef HAVE_BOINC
    std::string physical_name;
    struct resolved_path_node *node;
    char *copy;
    int rc;

    if (boinc_state != BOINC_READY) {
        fprintf(stderr,
                "%s: BOINC filename resolution requested before initialisation\n",
                option ? option : "filename");
        return -1;
    }
    rc = boinc_resolve_filename_s(logical_name, physical_name);
    if (rc) {
        fprintf(stderr,
                "%s: BOINC could not resolve logical filename '%s'"
                " (status %d)\n",
                option ? option : "filename", logical_name, rc);
        return -1;
    }
    if (physical_name.empty()) {
        fprintf(stderr,
                "%s: BOINC resolved logical filename '%s' to an empty path\n",
                option ? option : "filename", logical_name);
        return -1;
    }
#ifndef _WIN32
    if (follow_native_symlinks(option, &physical_name)) return -1;
#endif

    copy = (char *)malloc(physical_name.size() + 1);
    node = (struct resolved_path_node *)malloc(sizeof(*node));
    if (!copy || !node) {
        free(copy);
        free(node);
        fprintf(stderr,
                "%s: out of memory while retaining resolved filename\n",
                option ? option : "filename");
        return -1;
    }
    memcpy(copy, physical_name.c_str(), physical_name.size() + 1);
    node->path = copy;
    node->next = resolved_paths;
    resolved_paths = node;
    *resolved_name = copy;
#else
    (void)option;
    *resolved_name = logical_name;
#endif
    return 0;
}

extern "C" void bench_boinc_progress_suspend(int on)
{
#ifdef HAVE_BOINC
    progress_suspended = on ? 1 : 0;
#else
    (void)on;
#endif
}

extern "C" void bench_boinc_fraction_done(double fraction_done)
{
#ifdef HAVE_BOINC
    int rc;

    if (boinc_state != BOINC_READY || !std::isfinite(fraction_done)) return;
    /* Dropped before the monotonic high-water mark below, deliberately: a
     * throwaway calibration band must not leave 0.99 behind for the real band
     * to be clamped up to. See bench_boinc_progress_suspend(). */
    if (progress_suspended) return;
    if (fraction_done < 0.0) fraction_done = 0.0;
    if (fraction_done > 1.0) fraction_done = 1.0;
    /* BOINC requires successive values to be nondecreasing. Keep that
     * invariant here rather than relying on every progress estimator and
     * future call site to remember it. */
    if (fraction_done < last_fraction_done)
        fraction_done = last_fraction_done;
    if (fraction_done == last_fraction_done) return;

    rc = boinc_fraction_done(fraction_done);
    /* Preserve monotonicity of the arguments even if the runtime reports an
     * error: it may have consumed the update before returning that status. */
    last_fraction_done = fraction_done;
    if (rc && !fraction_warning_printed) {
        fprintf(stderr,
                "BOINC: fraction-done report failed with status %d;"
                " computation will continue\n",
                rc);
        fraction_warning_printed = 1;
    }
#else
    (void)fraction_done;
#endif
}

extern "C" int bench_boinc_finish(int status)
{
#ifdef HAVE_BOINC
    int rc;

    if (boinc_state == BOINC_READY && status == 0)
        bench_boinc_fraction_done(1.0);
    free_resolved_paths();

    /* Only a runtime that came up may be shut down through the API. On
     * BOINC_INIT_ATTEMPTED, boinc_init_options() failed: no shared-memory
     * segment was attached and no timer thread started, so boinc_finish()
     * would drive an uninitialised runtime. Every BOINC sample application
     * plain-exits this case (`retval = boinc_init(); if (retval) exit(retval);`).
     * BOINC_READY is the only state where boinc_finish(nonzero) is what
     * reports an application error rather than a crash. */
    if (boinc_state != BOINC_READY) {
        boinc_state = BOINC_NOT_STARTED;
        return status;
    }
    boinc_state = BOINC_NOT_STARTED;
    rc = boinc_finish(status);
    /* BOINC normally terminates the application here. Preserve a useful
     * process status if a test double or an unusual runtime returns anyway. */
    return rc ? rc : status;
#else
    return status;
#endif
}
