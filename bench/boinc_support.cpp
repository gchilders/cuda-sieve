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

extern "C" void bench_boinc_fraction_done(double fraction_done)
{
#ifdef HAVE_BOINC
    int rc;

    if (boinc_state != BOINC_READY || !std::isfinite(fraction_done)) return;
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
