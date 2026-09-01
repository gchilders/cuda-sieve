/* runlog.h -- the on-disk half of a run's record (STATUS.md item 12b).
 *
 * WHY THIS IS NOT A TEE. The console reporter is a 30-second \r line, which is
 * the right thing to watch and the wrong thing to keep: it holds q, relations,
 * rate and ETA, and none of the four numbers that say whether those are worth
 * anything. Finding 53 is that host contention costs up to 29% of wall clock
 * while every cudaEvent timer stays flat within 1%, and that no cross-box wall
 * or ETA comparison is valid without knowing host load on both. A log that
 * carries only what the terminal shows therefore records an unattended run
 * that cannot be audited afterwards -- which is the whole point of running it
 * unattended.
 *
 * So each record carries GPU-accounted/wall, GPU utilisation, board watts and
 * the host load average alongside the progress numbers. The first three come
 * from NVML, loaded dynamically at run time: there is no build dependency,
 * and a host without the library logs "n/a" in those columns rather than
 * failing.
 *
 * THE HEADER IS THE OTHER HALF. RESULTS.md has had to grow a warning that
 * findings 48/54 are the c147 while 43/44 are the c183, and finding 55 exists
 * because a geometry lived in the source defaults rather than in the run
 * record. A log whose first lines carry the commit, the argv, the fingerprint,
 * the card, the geometry and the factor-base convention ends that class of
 * confusion at the point where it is free to prevent.
 *
 * FAILURE IS NEVER FATAL, same discipline as the checkpoint: a log that cannot
 * be written warns once and the band continues. Losing the record of a run is
 * bad; losing the run is worse.
 *
 * A MODULE AND NOT A HEADER, unlike ckpt.h next to it, because it owns state:
 * run_pipeline() compiles into bench_kernels.o and the setup into
 * bench_main.o, so a `static` handle in a header would give those two
 * translation units a log each -- the one in the band loop unopened, and the
 * header written to a file nothing else ever appends to.
 */
#ifndef BENCH_RUNLOG_H
#define BENCH_RUNLOG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Appends; a resumed session continues the same file, so a job that was
 * stopped and restarted reads as one run with a second header block in the
 * middle -- which is what happened, and is more useful than either a truncated
 * file or a directory of numbered fragments.
 *
 * The first record falls due one full period after this call rather than
 * immediately. Returns -1 having already warned; the caller continues either
 * way. */
int  runlog_open(const char *path, double period_s);
void runlog_close(void);
int  runlog_active(void);

/* True at most once per period. This owns the clock -- its own monotonic one,
 * so that a log opened during startup and consumed inside the band loop needs
 * no clock passed between two translation units -- and the caller decides what
 * to record, so the console reporter and the log can run at different cadences
 * from the same loop. */
int  runlog_due(void);

/* Header line, commented so `grep -v '^#'` leaves the records alone. */
void runlog_note(const char *key, const char *fmt, ...);

/* One timestamped record: local time with the UTC offset, then the caller's
 * text. */
void runlog_record(const char *fmt, ...);

/* Mid-band warnings -- bucket overflow, the TD factor cap, a target not
 * reached -- go to BOTH streams. stderr is what a terminal shows and what
 * BOINC uploads; the log is what survives to be read afterwards, and a warning
 * missing from it makes the run look clean in the one place anybody will look
 * a week later. Takes a printf format, and unlike fprintf(stderr, ...) it
 * needs no trailing newline. */
void runlog_warn(const char *fmt, ...);

/* Set (and restored) only around a throwaway pass that must not be able to
 * write anything a reader could mistake for the real run's own diagnostics --
 * currently just the HIP pipeline's slab-size calibration probes. Suppresses
 * runlog_warn()'s stderr half only; the log-file half is already inert there
 * since a throwaway pass runs with no --log path configured. Zero by
 * default and never set by any CUDA-side code, so this is a no-op there. */
extern int g_runlog_quiet;

/* Bind the telemetry to the card THIS process selected, by PCI bus ID in
 * NVML's "domain:bus:device.function" form -- from cudaDeviceProp, never from
 * a CUDA ordinal. NVML enumerates by PCI order while CUDA can be reordered by
 * CUDA_DEVICE_ORDER and renumbered by CUDA_VISIBLE_DEVICES, so
 * nvmlDeviceGetHandleByIndex(cuda_ordinal) reads a DIFFERENT CARD's watts
 * whenever those disagree -- silently, and exactly on the multi-GPU hosts
 * where per-task telemetry matters most.
 *
 * Returns 0 if telemetry is available; a failure is normal on a host without
 * the driver library and simply leaves the columns "n/a". */
int  runlog_gpu_bind(const char *pci_bus_id);

/* Utilisation is a percentage over NVML's own sampling window; watts is the
 * BOARD sensor, not whole-box power, which needs the meter (STATUS.md item 6).
 * Either can be unavailable on its own -- WSL2 and some datacentre parts
 * report one and not the other -- so they are queried independently rather
 * than as a pair. */
int  runlog_gpu_util(unsigned int *pct);
int  runlog_gpu_watts(double *w);

/* `git describe --always --dirty` as of the build, or "unknown" outside a
 * checkout. Lives here rather than in bench_main.cu so that only this one
 * cheap object has to be rebuilt when HEAD moves -- the whole point being that
 * the string in the log is the commit that produced the binary, not the one
 * that happened to be checked out when a big CUDA object was last compiled. */
const char *runlog_build_desc(void);

/* The pricing -D's the binary was built with, empty for a shipping build. A
 * non-empty value means the norm was deliberately altered and the run's
 * relations are not production output. */
const char *runlog_build_defs(void);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_RUNLOG_H */
