/* ======================= the production pipeline ========================= *
 *
 * Both sides in one process. `bench`'s run_bench remains the measurement
 * harness -- every gate closed so far lives there and every command in
 * RESULTS.md still reproduces -- and this is the path that becomes the siever.
 *
 * The difference that matters is capability, not speed. The two survivor
 * bitmaps stay device-resident, so the intersection, the rank scan and the
 * (x,a,b) emission happen ONCE instead of once per side, and the two sides'
 * cofactors are joined in memory. Nothing here produces a candidate the
 * two-process path could not; what it can do is produce a *stream* of them
 * across many special-q, which is what the cofactor queue needs and what a
 * file round trip between two single-side processes cannot supply.
 *
 * Included at the end of bench_kernels.cu so it can use the static helpers
 * already defined there (run_td_stage, td_build_*).
 */
#ifndef CUDA_SIEVE_PIPELINE_CUH
#define CUDA_SIEVE_PIPELINE_CUH

#include "td.cuh"      /* ss_magic_build, SS_KSHIFT */
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <utility>
#include "ckpt.h"
#include "runlog.h"
#include "watchdog.h"
#include <signal.h>

/* ---- clean stop -------------------------------------------------------- *
 *
 * A band runs for days and gets interrupted. Without a handler the process
 * dies inside a q with the stdio buffer unwritten and the last line possibly
 * torn, so the first signal only sets a flag: the band loop notices it at the
 * top of the next q -- after the next (q, rho) has been pulled but before any
 * work is done for it -- drains the cofactor queue, fsyncs, and records that
 * pair as the resume point. A planned stop then loses nothing at all.
 *
 * The second signal is the escape hatch for a drain that is itself stuck. It
 * skips the checkpoint, which is safe: the previous one is still a valid, just
 * older, resume point. _exit is used because it is async-signal-safe and
 * flushing stdio from a handler is not. */
static volatile sig_atomic_t g_pipe_stop = 0;

/* run_pipeline_impl -> run_pipeline only: the bucket array did not fit or
 * could not be allocated, before anything was opened or consumed. Negative,
 * so every `rc < 0` test treats it as the failure it is; run_pipeline retries
 * an up-sized slab plan once at the default and maps it to PIPE_RC_FAIL. */
#define PIPE_RC_NOVRAM (-2)

/* ---- --instances: what a pipeline thread knows about its siblings ------ *
 *
 * NULL on an ordinary run, so every hook below is a no-op there. Under
 * --instances each pipeline thread points it at its own context: its index,
 * and the state the instances share -- one lock that lets a single instance at
 * a time run its setup or its end-of-band report (which also serialises their
 * device allocations), and the per-instance progress the BOINC fraction is
 * averaged from. */
struct pipe_inst_shared_t {
    std::mutex print;
    std::mutex boinc;
    int n;
    double frac[PIPE_INSTANCES_MAX];
    /* Under `print`: instances whose setup has not finished, and the modelled
     * footprint each will need. The optional second bucket array and
     * --q-pipeline's transform set reserve that much for them, so the first
     * instance cannot take what a sibling needs to start at all. */
    int setups_left;
    size_t m_inst;
};
struct pipe_inst_ctx_t {
    int k, n;
    uint64_t qmin, qmax;
    pipe_inst_shared_t *sh;
};
static thread_local pipe_inst_ctx_t *g_pinst = NULL;

/* What an optional allocation must leave free for the siblings still to set
 * up: 0 without --instances. Read during setup, under `print`. */
static size_t pipe_inst_reserve()
{
    return g_pinst
        ? (size_t)(g_pinst->sh->setups_left - 1) * g_pinst->sh->m_inst : 0;
}

/* Held across run_pipeline_impl's setup and its end-of-band report, released
 * for the band itself. A no-op without --instances. */
struct pipe_print_guard_t {
    bool held = false;
    void lock()
    {
        if (g_pinst && !held) { g_pinst->sh->print.lock(); held = true; }
    }
    void unlock()
    {
        if (held) { g_pinst->sh->print.unlock(); held = false; }
    }
    ~pipe_print_guard_t() { unlock(); }
};

#ifdef HAVE_BOINC
/* One fraction for the work unit: under --instances the mean of the
 * instances' own, each of which is the existing estimator over its share. */
static void pipe_report_fraction(double f)
{
    if (!g_pinst) { bench_boinc_fraction_done(f); return; }
    pipe_inst_shared_t *sh = g_pinst->sh;
    std::lock_guard<std::mutex> g(sh->boinc);
    double sum = 0.0;
    sh->frac[g_pinst->k] = f;
    for (int i = 0; i < sh->n; i++) sum += sh->frac[i];
    bench_boinc_fraction_done(sum / sh->n);
}
#endif

/* The per-q sync of the q loop. Device-wide by default; under --cof-async the
 * loop's own work is all on the legacy stream, and a device-wide sync would
 * wait for the in-flight cofactor flush (on g_cof_stream) and undo the
 * overlap. See cofa_start. */
static mtlError_t pipe_qsync(void)
{
    /* Stream 0 is this thread's own default stream (the build uses
     * --default-stream per-thread), and never the device-wide form: under
     * --q-pipeline the next q is sieving on another stream (finding 108),
     * under --instances so are the other pipelines, and under --cof-async a
     * flush is running on its own. Every launch this sync stands for is on
     * stream 0; --fill-concurrent's side streams are joined by event first. */
    return mtlStreamSynchronize(0);
}


/* Called from signal context on POSIX and from the console-control thread on
 * Windows, so it touches nothing but the flag. The ++ is not atomic against
 * the sieve thread on Windows, which makes the second-request escape hatch
 * best-effort there rather than exact -- a lost increment costs one extra
 * request, never a missed stop. */
static void pipe_request_stop(void)
{
    if (++g_pipe_stop >= 2) bench_fast_exit(130);
}

/* The widest BN_LIMBS any skipped (q,rho) asked for this session; -1 once some
 * skip needed more than the 16 limbs the build supports, 0 when nothing was
 * skipped. It exists so the PIPE_SKIP_MAX stop can put ONE number on stderr --
 * stderr.txt is what a BOINC project gets back from a failed result, and
 * "rebuild at N limbs" is the whole content of that failure. Leaving it to be
 * reconstructed from a hundred per-skip warnings is not a dispatch rule.
 *
 * A file static because the skip is raised three call levels down inside
 * pipe_side_prepare_q, whose return value is already a three-way code; the
 * alternative was an out-parameter threaded through the per-q path for a value
 * only the cap ever reads. Reset at the top of every band. */
static thread_local int g_pipe_need_limbs = 0;

/* Record a resume point.
 *
 * THE PRECONDITION IS THE WHOLE DESIGN: the caller must guarantee that the
 * relation file holds a whole number of special-q. Under --cofactor that is
 * true only at a whole-q boundary with no unflushed candidates belonging to
 * completed work. The pipeline deliberately records such a point before q0,
 * after a flush performed before any slab of the current q has been emitted,
 * and after an explicit clean-stop drain. A flush after an earlier slab of the
 * current q may leave a partial q in the file and is NOT checkpointable; resume
 * truncates that tail to the previous whole-q checkpoint and replays the q.
 * Without --cofactor the host join loop writes each q as it finishes, so any q
 * boundary will do. */
static int pipe_checkpoint(const poly_t *P, const bench_cfg_t *cfg, ckpt_t *ck,
                           FILE *fr, FILE *fc, const qsel_t *next,
                           unsigned long long nrel, unsigned long long nqdone)
{
    static thread_local int warned_no_id = 0;
    int64_t ro, co = 0;
    bench_file_id_t cid;
    if (!fr || !cfg->relations) return 0;
    /* fsync before the sidecar, never after: the checkpoint asserts these
     * bytes are on the platter, and a sidecar naming bytes that are still in
     * the page cache is worse than no sidecar at all. */
    if (fflush(fr) || bench_sync_stream(fr)) { perror("relations"); return -1; }
    if (fc && (fflush(fc) || bench_sync_stream(fc))) {
        perror("candidates"); return -1;
    }
    if ((ro = bench_tell(fr)) < 0) { perror("relations"); return -1; }
    if (fc && (co = bench_tell(fc)) < 0) { perror("candidates"); return -1; }
    if (fc) {
        /* Identity from the open handle, not from a stat buffer: st_ino is a
         * hardcoded 0 under MSVC, so the (dev, ino) this used to record would
         * later match any peer .part on the same drive -- and the recorded
         * identity is what stands between a checkpoint pathname and an
         * unlink. On POSIX these are still st_dev/st_ino, so an existing
         * Linux checkpoint compares equal exactly as before. */
        if (bench_file_id_stream(fc, &cid)) { perror("candidates"); return -1; }
        if (ckpt_part_path(cfg->candidates, ck->cand_part,
                           sizeof ck->cand_part))
            return -1;
        /* A filesystem that will not supply an identity leaves this 0/0, which
         * the reader treats as "unknown" and refuses to delete on. Recording
         * it is still right: the alternative is failing the checkpoint over a
         * field that only ever gates a discard. */
        if (!bench_file_id_valid(&cid) && !warned_no_id) {
            warned_no_id = 1;   /* once: this runs at every checkpoint */
            fprintf(stderr, "  warning: %s is on a filesystem that supplies no"
                    " file identity; automatic recovery will not be able to"
                    " discard it\n", ck->cand_part);
        }
        ck->cand_dev = (unsigned long long)cid.volume;
        ck->cand_ino = (unsigned long long)cid.index;
    } else {
        ck->cand_part[0] = '\0';
        ck->cand_dev = ck->cand_ino = 0;
    }
    ck->next_q   = next->q;
    ck->next_rho = next->rho;
    ck->rel_bytes  = (unsigned long long)ro;
    ck->cand_bytes = (unsigned long long)co;
    ck->nrel   = nrel;
    ck->nqdone = nqdone;
    return ckpt_write(cfg->relations, P, cfg, ck);
}

/* A missed checkpoint costs at most one flush of re-sieving on the next
 * resume. Aborting the band costs everything produced since the last
 * successful one, so a full disk, a read-only remount or an fsync EIO warns
 * and the sieve keeps going. Only the FIRST failure is reported: a persistent
 * one would otherwise print on every flush for days. */
static void pipe_try_checkpoint(const poly_t *P, const bench_cfg_t *cfg,
                                ckpt_t *ck, FILE *fr, FILE *fc,
                                const qsel_t *next, unsigned long long nrel,
                                unsigned long long nqdone,
                                int *written, int *warned)
{
    /* fsync of the relation file lives under here, which is the one host
     * phase that can legitimately block for seconds on a busy disk -- name it
     * so a stall report can say so. */
    wd_phase("q.checkpoint");
    if (pipe_checkpoint(P, cfg, ck, fr, fc, next, nrel, nqdone) == 0) {
        *written = 1;
        return;
    }
    if (!(*warned)++)
        fprintf(stderr,
                "\n  WARNING: cannot write the resume checkpoint for %s.\n"
                "  Sieving continues, but a crash will resume from an older"
                " point, or leave an\n"
                "  unresumable .part if none has been written yet.\n",
                cfg->relations);
}

typedef struct {
    uint32_t *primes, *roots;
    plat_t   *plat;
    /* Present only in the slabbed specialization. fill advances walk_cur into
     * walk_next; resieve replays walk_cur. Then, single-queued, the host swaps
     * the two pointers after the slab; under --q-pipeline the three buffers
     * (with walk_spare) turn at the issue point instead (pside_rotate_walk). */
    uint64_t *walk_cur, *walk_next;
    /* --fill-rows: the first nrow bucketed entries (q < fill_rows * I) are
     * filled by k_fill_rows from their transformed roots, written here by
     * k_transform; k_fill_atomic walks the rest. nrow == 0 is the old fill. */
    uint32_t *rowrt;
    uint32_t  nrow;
    /* --q-pipeline: the next q's transform output, written while this q's
     * trial division still reads plat/rowrt. pside_swap_alt trades the two
     * sets at the point the next q is issued. NULL when single-queued.
     *
     * walk_spare is the third walk-cursor buffer a slabbed plan needs once
     * the next slab is sieved under this slab's trial division: this slab's
     * TD reads the cursor it started from (walk_cur), the next slab's fill
     * reads this slab's output (walk_next) and writes its own, so three are
     * live. pside_rotate_walk turns them at every issue point. NULL unless
     * the pipeline was granted on a slabbed plan. */
    plat_t   *plat_alt;
    uint32_t *rowrt_alt;
    uint64_t *walk_spare;
    uint32_t *survbits;
    /* Fine slices (pipe_build_fine): per-entry 18-bit id, and the first entry
     * of each id. NULL when log_region > 14 keeps the original slices. */
    uint32_t *fine, *fine_start;
    uint32_t  nfine, fineF;
    int       lut_shift;   /* apply's LUT index: rec >> lut_shift */
    uint16_t *slice, *slice_logp;
    uint32_t *sp, *srt, *sg;
    uint32_t *smag;             /* ss_first reciprocals; see ss_magic_build */
    uint16_t *slp;
    uint32_t  nslice_pow2, nsmall, nblk, nwrp, nsplit, nfb;
    size_t    apply_smem;
    norm_t    N;
    uint32_t  CINIT, BOUND, tconst, nsurv;
    /* persistent across special-q */
    uint32_t *hsp, *hsrt, *hsg; uint16_t *hslp;
    uint32_t *hsmag;
    uint32_t *d_nsurv, *d_nproj;
    unsigned long long *d_nlost;
    /* PINNED readback slot, filled by async copies queued right behind each
     * slab's k_apply. The join reads it after ev[5] instead of making two
     * synchronous mtlMemcpy round trips per side, which cost 35-70 us each
     * under WSL2 and sat on the critical path between q (finding 113). */
    struct { uint32_t nsurv, overflow; } *h_rb;
    /* ev[0..3] are the transform/fill/apply boundaries. ev[4] is the
     * transform END, kept separate because pipe_side_sieve_issue re-records
     * ev[1] once per slab and would clobber it. ev[5] follows the h_rb
     * copies, after ev[3], so the apply timing does not include them; it is
     * only waited on, never timed, so it is created without timing. */
    mtlEvent_t ev[6];
} pside_t;

static void pside_free(pside_t *S)
{
    mtlFree(S->primes); mtlFree(S->roots); mtlFree(S->plat);
    mtlFree(S->walk_cur); mtlFree(S->walk_next);
    mtlFree(S->rowrt);
    mtlFree(S->plat_alt); mtlFree(S->rowrt_alt); mtlFree(S->walk_spare);
    mtlFree(S->survbits); mtlFree(S->slice); mtlFree(S->slice_logp);
    mtlFree(S->fine); mtlFree(S->fine_start);
    mtlFree(S->sp); mtlFree(S->srt); mtlFree(S->sg); mtlFree(S->slp);
    mtlFree(S->smag);
    mtlFree(S->d_nsurv);
    mtlFree(S->d_nproj); mtlFree(S->d_nlost);
    if (S->hsp)  mtlFreeHost(S->hsp);
    if (S->hsrt) mtlFreeHost(S->hsrt);
    if (S->hsg)  mtlFreeHost(S->hsg);
    if (S->hslp) mtlFreeHost(S->hslp);
    if (S->hsmag) mtlFreeHost(S->hsmag);
    if (S->h_rb) mtlFreeHost(S->h_rb);
    for (int k = 0; k < 6; k++) if (S->ev[k]) mtlEventDestroy(S->ev[k]);
    memset(S, 0, sizeof(*S));
}

/* --q-pipeline: point the side at its other transform-output set. The caller
 * keeps a copy of the pside_t from before the swap for the trial division of
 * the q it has not finished; everything else in the struct is shared. */
static void pside_swap_alt(pside_t *S)
{
    std::swap(S->plat, S->plat_alt);
    std::swap(S->rowrt, S->rowrt_alt);
}

/* --q-pipeline on a slabbed plan, at an issue point: the next slab (or the
 * next q's first) starts from this slab's output cursor and writes into the
 * spare, while the cursor this slab started from stays with its trial
 * division. The caller takes TD's view of the side BEFORE turning them. The
 * single-queued advance is the two-buffer swap in run_pipeline_impl. */
static void pside_rotate_walk(pside_t *S)
{
    uint64_t *const started = S->walk_cur;
    S->walk_cur = S->walk_next;
    S->walk_next = S->walk_spare;
    S->walk_spare = started;
}

/* The walk and line-sieve preconditions, checked where the factor bases are
 * CONSUMED rather than only where --bkthresh is parsed, so a caller that
 * builds its own split cannot bypass them (finding 100):
 *   - every BUCKETED modulus must be >= I: the Franke-Kleinjung walk
 *     (plattice.cuh) visits one hit per row, and below I a row holds several,
 *     so the rest are lost silently;
 *   - every LINE-SIEVED modulus must be < 2^30, so sieve_small's even-row
 *     step 2m under --sieve-skip cannot wrap a uint32. */
static int pipe_fb_bounds_ok(const fb_t *fb, uint32_t lo, uint32_t hi,
                             const char *what)
{
    if (!fb) return 1;
    for (uint32_t i = 0; i < fb->n; i++)
        if (fb->primes[i] < lo || fb->primes[i] >= hi) {
            fprintf(stderr, "run_pipeline: %s modulus %u is outside [%u, %u)\n",
                    what, fb->primes[i], lo, hi);
            return 0;
        }
    return 1;
}

/* Records the fill will produce, for sizing the bucket array. */
static uint64_t pipe_est_records(const fb_t *fb, uint32_t xmax)
{
    double acc = 0;
    for (uint32_t i = 0; i < fb->n; i++) acc += (double)xmax / fb->primes[i];
    return (uint64_t)(acc * 1.15) + 4096;
}

/* One progress estimator shared by the terminal reporter and BOINC.  Its
 * priority matches the user-visible progress line: a relation target is the
 * goal when present; otherwise use a bounded q count/range.  Open-ended work
 * is required by bench_main to have either --target-rels or --nq, so every
 * production band has a denominator.
 *
 * IT REPORTS ON THE OPERATOR'S WHOLE BAND, NOT ON THIS SESSION. That is what
 * `base_nq` is for, and why `relations` must be a cross-session total: on a
 * resume this used to restart the bar at 0 and climb again. Everything else
 * here already knew about resume -- the --target-rels stop test adds
 * base_rel, the checkpoint writer and the console counts add base_nq -- and
 * this one function was handed neither.
 *
 * The denominators are the half that is easy to miss, because bench_main has
 * ALREADY shrunk them by the time they arrive: --nq is reduced by the
 * completed count ("--nq counts this session's q") and qmin is moved to the
 * checkpoint's next_q, which is why the band's original lower bound has to be
 * kept aside in cfg->resume_qmin. Measured before the fix, one run-log record
 * read `nq=141 ... pct=18.06` -- 141 of 200 special-q done, bar at 18%. */
static double pipe_progress_fraction(const bench_cfg_t *cfg, int streaming,
                                     uint32_t nq, uint32_t nqdone,
                                     uint64_t current_q,
                                     unsigned long long relations,
                                     unsigned long long base_nq)
{
    double fraction = 0.0;
    /* The band's original lower bound; 0 means this band never resumed, so
     * cfg->qmin is already it. */
    const uint64_t q0 = cfg->resume_qmin ? cfg->resume_qmin : cfg->qmin;
    const double done_q = (double)base_nq + (double)nqdone;

    if (cfg->target_rels) {
        fraction = (double)relations / (double)cfg->target_rels;
    } else if (streaming && cfg->nq_max) {
        fraction = done_q / ((double)base_nq + (double)cfg->nq_max);
    } else if (streaming && cfg->qmax && cfg->qmax >= q0 &&
               current_q >= q0) {
        const double span = (double)(cfg->qmax - q0) + 1.0;
        fraction = ((double)(current_q - q0) + 1.0) / span;
    } else if (nq || base_nq) {
        fraction = done_q / ((double)base_nq + (double)nq);
    }

    if (!isfinite(fraction) || fraction < 0.0) return 0.0;
    return fraction > 1.0 ? 1.0 : fraction;
}

/* Everything that does NOT depend on the special-q: the factor base upload,
 * the slice tables, and the pinned staging. Hoisted out of the per-q path so a
 * band of q measures sieving rather than allocation churn. */
/* ---- fine slices: the bucket record names its factor-base entry -------- *
 *
 * A bucket record is (offset in region | slice << 16), and the slice only told
 * apply which log p to add: up to 262,144 entries shared one. The bucket
 * resieve (k_resieve_bucket) has to get from a record back to its prime, so
 * the pipeline gives each record an 18-bit FINE slice id at bit 14 instead --
 * a run of at most F consecutive entries, F chosen so the ids fit -- and apply
 * reads log p from a 256-entry COARSE table indexed by id >> 10. A coarse
 * group never spans a change of log p (the id skips to the next group when
 * log p changes), so every record still adds exactly its entry's log p; only
 * the index arithmetic moved. Needs log_region <= 14; above that the pipeline
 * keeps the original slices and the walking resieve. */
#define FINE_SHIFT     14u
#define FINE_BITS      18u
#define FINE_PER_GROUP 1024u    /* fine ids per coarse log-p group */

static int pipe_build_fine(const fb_t *fb, uint32_t **fine_out,
                           uint32_t **start_out, uint16_t **coarse_out,
                           uint32_t *nfine_out, uint32_t *F_out)
{
    const uint32_t maxid = 1u << FINE_BITS;
    const uint32_t ngroup = maxid / FINE_PER_GROUP;
    for (uint32_t F = 32; F <= (1u << 26); F <<= 1) {
        uint32_t id = 0, fill = 0;
        int cur = -1, ok = 1;
        for (uint32_t k = 0; k < fb->n && ok; k++) {
            const int lp = fb->logp[k];
            if (cur < 0) cur = lp;
            else if (lp != cur) {
                id = (id / FINE_PER_GROUP + 1) * FINE_PER_GROUP; fill = 0; cur = lp;
            } else if (fill == F) { id++; fill = 0; }
            if (id >= maxid) ok = 0;
            fill++;
        }
        if (!ok) continue;
        const uint32_t nfine = id + 1;
        uint32_t *fine = (uint32_t *)malloc((size_t)fb->n * 4);
        uint32_t *start = (uint32_t *)malloc((size_t)(nfine + 1) * 4);
        uint16_t *coarse = (uint16_t *)calloc(ngroup, 2);
        if (!fine || !start || !coarse) {
            free(fine); free(start); free(coarse);
            fprintf(stderr, "pipe_build_fine: out of memory\n");
            return -1;
        }
        id = 0; fill = 0; cur = -1;
        for (uint32_t k = 0; k < fb->n; k++) {
            const int lp = fb->logp[k];
            if (cur < 0) cur = lp;
            else if (lp != cur) {
                id = (id / FINE_PER_GROUP + 1) * FINE_PER_GROUP; fill = 0; cur = lp;
            } else if (fill == F) { id++; fill = 0; }
            fill++;
            fine[k] = id;
            coarse[id / FINE_PER_GROUP] = (uint16_t)lp;
        }
        for (uint32_t f = 0; f <= nfine; f++) start[f] = fb->n;
        for (uint32_t k = fb->n; k-- > 0; ) start[fine[k]] = k;
        for (uint32_t f = nfine; f-- > 0; )
            if (start[f] > start[f + 1]) start[f] = start[f + 1];
        *fine_out = fine; *start_out = start; *coarse_out = coarse;
        *nfine_out = nfine; *F_out = F;
        return 0;
    }
    fprintf(stderr, "pipe_build_fine: %u entries do not fit %u fine ids\n",
            fb->n, maxid);
    return -1;
}

/* Recover each bucketed prime hitting a two-sided survivor from the side's
 * bucket records rather than by re-walking its lattice (k_resieve_scatter),
 * after glas. One block per region. A record at a survivor names its fine
 * slice; every entry in that slice is tested at the survivor's (a, b) --
 * p | a - r*b for an affine root, p | r'*a - b for a projective one (the
 * convention pl_transform_gen uses) -- skipping the lattice-degenerate entries
 * the walk also skips. Two hitting entries in one slice are each appended once
 * per record, i.e. twice: harmless, k_td divides a listed prime out
 * completely, and the factor list it records is the same. */
/* rb_mod: device code, see metal/td.metal */


/* One block per region. The region's slice of the two-sided survivor bitmap
 * (2 KB at --region 14) is staged in shared memory once, so the test every
 * record faces is a shared-memory read rather than a global load that waits
 * on the record just loaded.
 *
 * A HIT IS HANDLED BY THE WHOLE WARP. Most of this kernel's time was never
 * the record scan: it was the hits, each one walking its fine slice -- ~120
 * factor-base entries on 2,1147+ -- in ONE lane, three dependent random loads
 * per entry, while the rest of the warp waited. The warp now finds its hits
 * with a ballot and takes them one at a time, lane l testing entries
 * start+l, start+l+32, ...: the slice is contiguous, so the loads coalesce,
 * and 120 serial steps become 4. Several primes of one slice can land in the
 * same survivor's list in any order; the emitters sort, so the relations do
 * not change. */
/* k_resieve_bucket: device code, see metal/td.metal */


template <bool SLABBED>
static int pipe_side_init(const fb_t *fb, const fb_t *fbs,
                          const bench_cfg_t *cfg, uint32_t bound,
                          uint32_t xmax_alloc, size_t optin_smem_limit,
                          pside_t *S)
{
    const uint32_t nbitword = xmax_alloc >> 5;
    uint16_t *hslice = NULL, *hlogp = NULL;
    uint32_t *hfine = NULL, *hstart = NULL;
    int rc = -1;

#define SIDE_INIT_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    memset(S, 0, sizeof(*S));
    S->CINIT = 4096u;
    S->BOUND = bound;
    S->nfb = fb->n;
    /* Do not rely on every present and future loader remembering the lattice
     * transform's prime-power precondition. Loaders/builders earn this cookie
     * through fb_validate(); subsets preserve it. Refuse before the first
     * host transform or device upload if that proof is absent. */
    if (!fb_is_transform_validated(fb) ||
        (fbs && fbs->n && !fb_is_transform_validated(fbs))) {
        fprintf(stderr,
                "pipe_side_init: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or uploading it\n");
        goto done;
    }
    SIDE_INIT_CK(mtlMalloc(&S->primes, (size_t)fb->n * 4));
    SIDE_INIT_CK(mtlMalloc(&S->roots,  (size_t)fb->n * 4));
    SIDE_INIT_CK(mtlMalloc(&S->plat,   (size_t)fb->n * sizeof(plat_t)));
    if constexpr (SLABBED) {
        SIDE_INIT_CK(mtlMalloc(&S->walk_cur,  (size_t)fb->n * sizeof(uint64_t)));
        SIDE_INIT_CK(mtlMalloc(&S->walk_next, (size_t)fb->n * sizeof(uint64_t)));
    }
    SIDE_INIT_CK(mtlMalloc(&S->survbits, (size_t)nbitword * 4));
    if (cfg->fill_rows) {
        /* The row tier is a PREFIX of the bucketed entries, which is what lets
         * k_fill_atomic take the rest by pointer offset. Ascending order is
         * fb_validate's (fb_load.c), and the check above refuses a base that
         * has not passed it. */
        const uint64_t rb = (uint64_t)cfg->fill_rows << cfg->logI;
        while (S->nrow < fb->n && fb->primes[S->nrow] < rb) S->nrow++;
        if (S->nrow) SIDE_INIT_CK(mtlMalloc(&S->rowrt, (size_t)S->nrow * 4));
    }
    SIDE_INIT_CK(mtlMemcpy(S->primes, fb->primes, (size_t)fb->n * 4,
                            mtlMemcpyHostToDevice));
    SIDE_INIT_CK(mtlMemcpy(S->roots, fb->roots, (size_t)fb->n * 4,
                            mtlMemcpyHostToDevice));

    S->lut_shift = 16;
    /* Fine slices when they fit; otherwise -- log_region > 14, or a base with
     * more log-p changes than 256 coarse groups hold -- the original slices
     * and the walking resieve, exactly as before. Never a reason to fail. */
    if (cfg->log_region <= (int)FINE_SHIFT &&
        !pipe_build_fine(fb, &hfine, &hstart, &hlogp, &S->nfine, &S->fineF)) {
        S->nslice_pow2 = (1u << FINE_BITS) / FINE_PER_GROUP;
        S->lut_shift = (int)(FINE_SHIFT + 10u);   /* log2(FINE_PER_GROUP) */
        SIDE_INIT_CK(mtlMalloc(&S->fine, (size_t)fb->n * 4));
        SIDE_INIT_CK(mtlMalloc(&S->fine_start, (size_t)(S->nfine + 1) * 4));
        SIDE_INIT_CK(mtlMemcpy(S->fine, hfine, (size_t)fb->n * 4,
                                mtlMemcpyHostToDevice));
        SIDE_INIT_CK(mtlMemcpy(S->fine_start, hstart,
                                (size_t)(S->nfine + 1) * 4,
                                mtlMemcpyHostToDevice));
        free(hfine); hfine = NULL;
        free(hstart); hstart = NULL;
    } else if (fb_build_slices(fb, &hslice, &hlogp, &S->nslice_pow2) < 0) {
        report_slice_build_error();
        goto done;
    }
    {
        const size_t smem = mtl_apply_smem(1u << cfg->log_region, 16);
        if (smem > optin_smem_limit) {
            fprintf(stderr, "  apply needs %zu B of shared memory for %u"
                    " padded slices; selected device supports at most %zu B"
                    " opt-in per block\n",
                    smem, S->nslice_pow2, optin_smem_limit);
            goto done;
        }
        /* cudaFuncSetAttribute is a property of the kernel specialization, not
         * of this side.  Remember the requirement here; run_pipeline configures
         * k_apply once, after both sides are initialized, to the larger value. */
        S->apply_smem = smem;
    }
    if (!S->fine) {
        SIDE_INIT_CK(mtlMalloc(&S->slice, (size_t)fb->n * 2));
        SIDE_INIT_CK(mtlMemcpy(S->slice, hslice, (size_t)fb->n * 2,
                                mtlMemcpyHostToDevice));
    }
    SIDE_INIT_CK(mtlMalloc(&S->slice_logp, (size_t)S->nslice_pow2 * 2));
    SIDE_INIT_CK(mtlMemcpy(S->slice_logp, hlogp,
                            (size_t)S->nslice_pow2 * 2,
                            mtlMemcpyHostToDevice));
    free(hslice); hslice = NULL;
    free(hlogp); hlogp = NULL;

    /* PINNED staging for the per-q small-sieve tables, allocated ONCE. The
     * 2026-08-04 review measured pageable staging here at 6.2 ms per side for
     * 54 KB; mtlHostAlloc per special-q would be worse than either. */
    if (cfg->small_sieve && fbs && fbs->n) {
        SIDE_INIT_CK(mtlHostAlloc((void **)&S->hsp, (size_t)fbs->n * 4,
                                   mtlHostAllocDefault));
        SIDE_INIT_CK(mtlHostAlloc((void **)&S->hsrt, (size_t)fbs->n * 4,
                                   mtlHostAllocDefault));
        SIDE_INIT_CK(mtlHostAlloc((void **)&S->hslp, (size_t)fbs->n * 2,
                                   mtlHostAllocDefault));
        SIDE_INIT_CK(mtlHostAlloc((void **)&S->hsg, (size_t)fbs->n * 4,
                                   mtlHostAllocDefault));
        SIDE_INIT_CK(mtlHostAlloc((void **)&S->hsmag, (size_t)fbs->n * 4,
                                   mtlHostAllocDefault));
        SIDE_INIT_CK(mtlMalloc(&S->sp,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(mtlMalloc(&S->srt, (size_t)fbs->n * 4));
        SIDE_INIT_CK(mtlMalloc(&S->sg,  (size_t)fbs->n * 4));
        SIDE_INIT_CK(mtlMalloc(&S->slp, (size_t)fbs->n * 2));
        SIDE_INIT_CK(mtlMalloc(&S->smag, (size_t)fbs->n * 4));
    }
    SIDE_INIT_CK(mtlMalloc(&S->d_nsurv, 4));
    SIDE_INIT_CK(mtlMalloc(&S->d_nproj, 4));
    SIDE_INIT_CK(mtlMalloc(&S->d_nlost, 8));
    for (int k = 0; k < 5; k++) SIDE_INIT_CK(mtlEventCreate(&S->ev[k]));
    SIDE_INIT_CK(mtlEventCreateWithFlags(&S->ev[5], mtlEventDisableTiming));
    SIDE_INIT_CK(mtlHostAlloc((void **)&S->h_rb, sizeof(*S->h_rb),
                               mtlHostAllocDefault));
    rc = 0;

done:
    free(hslice);
    free(hlogp);
    free(hfine);
    free(hstart);
#undef SIDE_INIT_CK
    return rc;
}

/* pipe_side_prepare_q's third return value: this special-q cannot be sieved at
 * this build's BN_LIMBS, but the band continues. Negative values stay reserved
 * for real failures, so `< 0` remains "the band is over". An enum, not a
 * #define, so it does not leak an unscoped macro out of this header. */
enum { PIPE_Q_SKIP = 1 };

/* Rate-limit budgets for the per-slab soft-skip warnings; see the note at the
 * bucket-overflow site for why these are not static locals. Like the other
 * file statics here they are never reset, so a second band in one process
 * inherits the first band's spend and sees only the end-of-band totals. */
static thread_local int g_warned_bucket_ovf = 0;
static thread_local int g_warned_td_trunc   = 0;

/* Skips that end the run. Past this the build is simply too narrow for the job
 * and the band would emit systematically deficient output; ./normscan says so
 * up front and names the width to rebuild with.
 *
 * An ABSOLUTE count, and deliberately so: a build that normscan has cleared for
 * the band should skip ZERO q, because the projection sits well under the limit.
 * Any skip at all means the width was chosen without the survey or the band
 * moved; a hundred means it was chosen wrongly. Note the two ways this net has
 * holes -- nqskip is not checkpointed, so a resumed band starts counting again,
 * and a work unit of a few hundred q can never reach the cap. Both are
 * acceptable only because normscan is supposed to make the whole situation
 * unreachable; neither should be relied on as the primary defence.
 *
 * REACHING IT IS NOT A CLEAN STOP. The band still drains and checkpoints --
 * throwing away the queued relations would be strictly worse -- but it returns
 * PIPE_RC_UNSUPPORTED, exits nonzero, and reports BENCH_OUTCOME_UNSUPPORTED to
 * BOINC. It used to exit 0, which credited a work unit for a band that emitted
 * a hundred skips and, per finding 93 case D, would emit nothing again on every
 * resume of the same binary. The stop is deliberate; the SUCCESS was not. */
#ifndef PIPE_SKIP_MAX
#define PIPE_SKIP_MAX 100
#endif

/* The same net, for the per-slab SOFT skips (bucket overflow and a truncated
 * factor list; norm overflow is fatal, see pipe_td_perq). Those exist so a
 * one-record shortfall costs a slab's
 * yield instead of a volunteer's whole task, which is right -- but the trade
 * is silent in one direction, and without a ceiling it stays silent all the
 * way down to "emitted almost nothing, exited 0". A job with an undersized
 * bucket array can skip every slab of every q and still report success; a
 * rate computed from that band looks like a slow card rather than a
 * misconfigured job.
 *
 * TWO counters, because either can be pathological while the other looks
 * clean:
 *
 *   PIPE_LOST_MAX       whole special-q that produced nothing because every
 *                       slab of them was skipped. The sharp signal, and
 *                       deliberately the same 100 as PIPE_SKIP_MAX: past a
 *                       hundred, the condition is systemic, not transient.
 *   PIPE_SLAB_SKIP_MAX  slabs LOST, which is not the same as incidents: a
 *                       bucket overflow abandons the rest of its q (see the
 *                       skip site), so one incident on slab 0 of a 4-slab
 *                       plan counts 4, not 1. Counting lost slabs is the
 *                       point -- it is the yield -- but it means the ceiling
 *                       trips after correspondingly fewer incidents. Catches
 *                       the case PIPE_LOST_MAX cannot see: one slab of four
 *                       failing on EVERY q is a permanent yield loss with
 *                       nq_lost stuck at zero.
 *
 * Absolute counts, not rates, matching PIPE_SKIP_MAX's idiom: a healthy band
 * needs a thousand separate incidents to trip this, and a broken one trips it
 * within the first few hundred q. They inherit PIPE_SKIP_MAX's two holes too
 * -- neither is checkpointed, so a resumed band starts counting again, and a
 * short work unit may finish before reaching either. Like that cap, this is a
 * backstop against silent waste, not the primary defence; sizing the bucket
 * array and mfb correctly is. */
#ifndef PIPE_LOST_MAX
#define PIPE_LOST_MAX 100
#endif
#ifndef PIPE_SLAB_SKIP_MAX
#define PIPE_SLAB_SKIP_MAX 1000
#endif

/* Per-special-q work that is independent of the slab: transform the small
 * line-sieve tables, set up the norm over the FULL J range, and transform the
 * bucketed factor base once.  In SLABBED mode k_transform also seeds the first
 * slab's continuation walk. */
template <bool SLABBED>
static int pipe_side_prepare_q(const fb_t *fb, const fb_t *fbs,
                               const qlat_t *L, const poly_t *POLY,
                               const bench_cfg_t *cfg, int side, double scale,
                               int blocks, pside_t *S, double *t_host,
                               mtlStream_t st = 0, int quiet_skip = 0)
{
    uint32_t *idx = NULL, *tp = NULL, *trt = NULL, *tg = NULL;
    uint16_t *tlp = NULL;
    double h0 = host_ms();
    int rc = -1;
    /* PIPE_Q_SKIP is a THIRD outcome, distinct from both success and failure:
     * this one special-q cannot be sieved at this build's width, but the band
     * can continue past it. See the norm-width check below. */

#define PERQ_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t *hsp = S->hsp, *hsrt = S->hsrt, *hsg = S->hsg;
        uint16_t *hslp = S->hslp;
        uint32_t k = 0;
        for (uint32_t i = 0; i < fbs->n; i++) {
            uint32_t rt, g, m = pl_transform_enc(fbs->primes[i], fbs->roots[i],
                                                 L->a0, L->a1, L->b0, L->b1,
                                                 &rt, &g);
            hsp[k] = m;
            hsrt[k] = rt;
            hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        S->nsmall = k;
        if (k) {
            idx = (uint32_t *)malloc((size_t)k * sizeof(*idx));
            tp  = (uint32_t *)malloc((size_t)k * sizeof(*tp));
            trt = (uint32_t *)malloc((size_t)k * sizeof(*trt));
            tg  = (uint32_t *)malloc((size_t)k * sizeof(*tg));
            tlp = (uint16_t *)malloc((size_t)k * sizeof(*tlp));
            if (!idx || !tp || !trt || !tg || !tlp) {
                fprintf(stderr,
                        "pipe_side_prepare_q: out of memory sorting %u small-sieve entries\n",
                        k);
                goto done;
            }
            for (uint32_t i = 0; i < k; i++) idx[i] = i;
            std::stable_sort(idx, idx + k,
                             [hsp](uint32_t x, uint32_t y) {
                                 return hsp[x] < hsp[y];
                             });
            for (uint32_t i = 0; i < k; i++) {
                const uint32_t z = idx[i];
                tp[i] = hsp[z];
                trt[i] = hsrt[z];
                tg[i] = hsg[z];
                tlp[i] = hslp[z];
            }
            memcpy(hsp, tp, (size_t)k * sizeof(*hsp));
            memcpy(hsrt, trt, (size_t)k * sizeof(*hsrt));
            memcpy(hsg, tg, (size_t)k * sizeof(*hsg));
            memcpy(hslp, tlp, (size_t)k * sizeof(*hslp));
            free(idx); idx = NULL;
            free(tp); tp = NULL;
            free(trt); trt = NULL;
            free(tg); tg = NULL;
            free(tlp); tlp = NULL;
        }
        /* Reciprocals AFTER the sort, same as the standalone path: the sort
         * key is the modulus, so building them earlier would only mean
         * permuting them too. And after the tiers: the thread tier's
         * two-step range starts at nsplit, and each magic is proved for the
         * form its index gets (ss_bound_ok in td.cuh). */
        ss_tiers(hsp, k, &S->nblk, &S->nwrp);
        S->nsplit = ss_two_step_from(hsp, k, S->nwrp, cfg->J, cfg->logI);
        for (uint32_t i = 0; i < k; i++)
            ss_magic_build(hsp[i], hsg[i] > 1 ? cfg->J / hsg[i] : cfg->J,
                           cfg->logI, i >= S->nsplit, &S->hsmag[i]);
        /* ASYNC on purpose. A synchronous mtlMemcpy on the legacy default
         * stream cannot begin until prior stream work drains, so once the
         * transform sync was removed from the end of this function these
         * copies became the new serialisation point: side 0's uploads blocked
         * on side 1's still-running transform, and the wait was charged to
         * `host per-q` (measured: 1.179 -> 3.220 ms/q). Async keeps the host
         * running ahead and issuing the next launch.
         *
         * `st` is the legacy stream unless --q-pipeline was granted, when it
         * is the sieve stream; either way these copies precede this q's
         * transform, fill and apply on it.
         *
         * Safe because hsp/hsrt/hsg/hslp/hsmag are PINNED (mtlHostAlloc, see
         * pipe_side_init) and are only rewritten by the NEXT q's call, which
         * must not start before these copies have run. FOUR paths provide
         * that drain and ALL FOUR are load-bearing:
         *
         *   normal            single-queued, the slab loop's join, which waits
         *                     on ev[5] -- later in stream order. Under
         *                     --q-pipeline the next call is made DURING this
         *                     q's last slab, before its join (finding 114),
         *                     and prepare_next_q first reads this q's
         *                     transform time, which waits on ev[4] -- recorded
         *                     after these copies;
         *   prepared, not issued
         *                     a next q prepared that way and then not issued
         *                     (stop, a soft skip of this q, its own skip) is
         *                     prepared again at the loop top: the end-of-q
         *                     drain after the slab loop waits on its ev[4];
         *   PIPE_Q_SKIP       never reaches the slab loop, so it carries its
         *                     own mtlStreamSynchronize for exactly this
         *                     reason (see the skip site below);
         *   every slab soft-skipped
         *                     reaches the slab loop but may never sync side 0
         *                     (a side-1 skip short-circuits side 0), and is
         *                     covered by the post-loop mtlEventSynchronize
         *                     on ev[4] -- which is recorded AFTER these copies
         *                     and so subsumes them.
         *
         * A new early return added between here and that ev[4] synchronise
         * needs the same treatment, or these must go back to synchronous. */
        PERQ_CK(mtlMemcpyAsync(S->sp, hsp, (size_t)k * sizeof(*hsp),
                                mtlMemcpyHostToDevice, st));
        PERQ_CK(mtlMemcpyAsync(S->srt, hsrt, (size_t)k * sizeof(*hsrt),
                                mtlMemcpyHostToDevice, st));
        PERQ_CK(mtlMemcpyAsync(S->sg, hsg, (size_t)k * sizeof(*hsg),
                                mtlMemcpyHostToDevice, st));
        PERQ_CK(mtlMemcpyAsync(S->slp, hslp, (size_t)k * sizeof(*hslp),
                                mtlMemcpyHostToDevice, st));
        PERQ_CK(mtlMemcpyAsync(S->smag, S->hsmag, (size_t)k * sizeof(*S->hsmag),
                                mtlMemcpyHostToDevice, st));
    }

    /* Side 0's norms are G(x) = Y1*x + Y0, a degree-1 form. run_bench gets
     * that by mutating POLY in the caller; the pipeline must not, because
     * run_td_stage still needs the degree-5 coefficients for side 1. Make the
     * variant here and leave POLY alone. */
    {
        poly_t P = *POLY;
        if (side == 0) {
            P.deg = 1;
            P.c[0] = P.y0;
            P.c[1] = P.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) P.c[z] = 0.0;
        }
        norm_setup(&S->N, &P, L, cfg->logI, cfg->J, scale,
                   side == cfg->sq_side);
        {
            const double bits = norm_exact_bound_bits(&S->N);
            if (!norm_fits_exact(&S->N, BN_LIMBS * 32)) {
                /* SKIP THIS q, DO NOT KILL THE BAND.
                 *
                 * Aborting here was strictly the worst option available. It is
                 * not even a clean stop: the checkpoint records the failing
                 * (q,rho), so "rerun to resume" returns to this exact q and the
                 * band can never advance -- and under BOINC the work unit fails,
                 * is reissued, and fails identically on the next client. Against
                 * that, one special-q out of the tens of millions in a band is a
                 * rounding error in yield.
                 *
                 * WHAT THIS COSTS, because it is not free. "A wider build is
                 * byte-identical to a narrower one on any job the narrower one
                 * could run" was true before this change, and it is the property
                 * that qualified BN_LIMBS 12 -- the 8- and 12-limb relation files
                 * were diffed. Skipping weakens it to "byte-identical on any job
                 * normscan clears at the narrower width", which is still
                 * checkable, but only because normscan now exists to certify that
                 * condition up front. Do not weaken it further.
                 *
                 * Loud, counted, and fatal in bulk: every skip prints, the total
                 * lands in the band summary, and PIPE_SKIP_MAX of them ends the
                 * run. A hundred skips is not a tail any more, it is the wrong
                 * build for the job, and quietly emitting a whole band of
                 * deficient output would be worse than stopping. */
                const int need = bn_limbs_for_bits(bits);
                /* --q-pipeline's speculative call stays silent and leaves the
                 * band's limb advice alone: that q may never be counted (a
                 * stop or the target can come first), so the loop top
                 * prepares it again, loudly, if and when it is consumed. */
                if (quiet_skip) {
                    PERQ_CK(mtlStreamSynchronize(st));
                    rc = PIPE_Q_SKIP;
                    goto done;
                }
                /* NAMES rho, NOT JUST q. A q carries up to deg roots and it is
                 * the (q,rho) LATTICE that overflows, not the q: on the 2,1139+
                 * octic, (600002827, 369315816) needs 260.75 bits while
                 * (600002827, 272545945) needs 238.20 and sieves normally.
                 * Printing q alone leaves no way to reproduce or exclude the
                 * pair -- and the log position cannot supply it either, because
                 * this goes to stderr unbuffered while the per-q sieve lines go
                 * to stdout block-buffered, so in a redirected run this line
                 * surfaces above its own iteration's header. */
                runlog_warn("  ** SKIPPED q=%llu rho=%llu: exact side-%d"
                            " degree-%d norm may require %.2f bits; the"
                            " trial-division type holds %d",
                            (unsigned long long)L->q,
                            (unsigned long long)L->rho, side, P.deg, bits,
                            BN_LIMBS * 32);
                if (need)
                    runlog_warn("     rebuild with `make BN_LIMBS=%d` to sieve it"
                                " (run ./normscan over the band first)", need);
                else
                    runlog_warn("     no supported BN_LIMBS is wide enough"
                                " (the maximum is 16, i.e. 512 bits)");
                /* MAX over the band, and -1 wins outright: a single (q,rho)
                 * that no supported width covers makes "rebuild at N" wrong
                 * advice however many others would have fitted at N. */
                if (!need) g_pipe_need_limbs = -1;
                else if (g_pipe_need_limbs >= 0 && need > g_pipe_need_limbs)
                    g_pipe_need_limbs = need;
                /* DRAIN BEFORE SKIPPING. This path does NOT reach the slab
                 * loop -- run_pipeline_impl `continue`s on PIPE_Q_SKIP -- so
                 * nothing downstream will synchronise. Two things are left in
                 * flight without this: the four mtlMemcpyAsync uploads above,
                 * whose pinned source buffers the NEXT q rewrites, and (when
                 * side 0 raises the skip) side 1's k_transform, whose
                 * execution errors would otherwise be reported against the
                 * next q's first CUDA call. Skips are rare and capped at
                 * PIPE_SKIP_MAX, so the cost here is irrelevant. */
                PERQ_CK(mtlStreamSynchronize(st));
                rc = PIPE_Q_SKIP;
                goto done;
            }
        }
    }
    /* CINIT and BOUND were checked once, before any allocation, and are
     * invariant across the special-q band. Keeping them in the persistent
     * side context avoids reparsing or floating-point conversion per q. */
    {
        const float t = norm_target_host(&S->N, 0, cfg->J / 2);
        const int ti = (int)(t + 0.5f);
        S->tconst = (ti < 1) ? 1u :
                    ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
    }

    *t_host = host_ms() - h0;
    /* k_transform writes these unconditionally; NULL is an illegal access */
    /* st is 0 (the legacy default stream) unless --q-pipeline was granted,
     * so the single-queued path issues exactly what it always did. */
    PERQ_CK(mtlMemsetAsync(S->d_nproj, 0, 4, st));
    PERQ_CK(mtlMemsetAsync(S->d_nlost, 0, 8, st));

    PERQ_CK(mtlEventRecord(S->ev[0], st));
    MTL_LAUNCH_NAMED((SLABBED ? "k_transform_1" : "k_transform_0"), blocks, cfg->threads, 0, st, S->primes, S->roots, S->plat, fb->n, cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, S->d_nproj, S->d_nlost, S->walk_cur, S->rowrt, S->nrow);
    PERQ_CK(mtlEventRecord(S->ev[4], st));
    /* NO SYNCHRONIZE HERE. This function used to block on the transform purely
     * to read t_transform, and nothing after that read touches device data --
     * it is mtlGetLastError, an elapsed-time read and five host free()s. The
     * block cost a measured 0.800 ms/q of GPU idle at the
     * k_transform -> k_fill_atomic boundary plus 0.487 at
     * k_transform -> k_transform (RESULTS finding 88's gap ranking), because
     * the card drained while the host prepared the other side and the TD
     * tables. The elapsed time is now read later -- after the slab loop, or
     * under --q-pipeline when the next q is prepared -- by
     * pipe_read_transform, which waits on ev[4] itself.
     *
     * mtlGetLastError still catches launch-configuration failures, which is
     * what it was for. An error raised DURING execution now surfaces at the
     * next sync, inside pipe_side_sieve_join -- normal async CUDA practice. */
    PERQ_CK(mtlGetLastError());
    /* The transform time is NOT produced here any more. ev[0] and ev[4] are
     * left for run_pipeline_impl to subtract after the slab loop, which is
     * the only place they are known to be complete. */
    rc = 0;

done:
    free(idx); free(tp); free(trt); free(tg); free(tlp);
#undef PERQ_CK
    return rc;
}

/* The per-side bucket workspace. The two sides share ONE allocation today
 * precisely because they run back to back; `--fill-concurrent` gives each its
 * own so the two fills can overlap, and that second array is the entire cost
 * of the option. See run_pipeline_impl's admission check.
 *
 * `stream` is 0 -- the legacy default stream -- in serial mode, which is what
 * keeps that path identical to the pre-split code: every launch below stays on
 * the stream it was always on, in the order it was always in. */
typedef struct {
    uint8_t     *bucket;
    uint32_t    *cursor;
    uint32_t    *overflow;
    uint32_t     cap;
    mtlStream_t stream;        /* 0 = legacy default stream */
} pbkt_t;

/* Area-dependent work for one side and one slab, ISSUED but not awaited.
 * SLABBED=false compiles to the pre-slab fill and apply kernels; no walk-state
 * read/write or global-j adjustment survives.
 *
 * Split into issue/join so the caller picks the order:
 *
 *   serial      issue(1) join(1) issue(0) join(0)    -- unchanged behaviour
 *   concurrent  issue(1) issue(0) join(1) join(0)
 *
 * NOTHING MAY TOUCH THE LEGACY DEFAULT STREAM BETWEEN THE TWO ISSUES in the
 * concurrent order. The side streams are created BLOCKING, so a default-stream
 * operation slipped in between would implicitly synchronise with side 1 and
 * quietly serialise the one thing the option exists to do -- while still
 * emitting correct relations, so no output gate would catch it. That is also
 * why the memsets below became mtlMemsetAsync ON THE SIDE'S STREAM: the
 * synchronous form runs on the default stream and would do exactly that. */
/* Row bands per slab for the row tier (k_fill_rows, k_resieve_rows):
 * --fill-row-bands, else one band per FILL_ROW_RUN rows; never more bands
 * than rows, so no band is empty. */
static uint32_t pipe_row_bands(const bench_cfg_t *cfg, uint32_t rows)
{
    uint32_t nb = cfg->fill_row_bands ? (uint32_t)cfg->fill_row_bands
                                      : (rows + FILL_ROW_RUN - 1) / FILL_ROW_RUN;
    if (nb > rows) nb = rows;
    return nb ? nb : 1;
}

template <bool SLABBED>
static int pipe_side_sieve_issue(const fb_t *fb, const bench_cfg_t *cfg,
                                 int side, uint32_t xmax, uint32_t j_base,
                                 const pbkt_t *bk, int fblocks, int fthreads,
                                 pside_t *S, const uint32_t *gate_bits = NULL,
                                 uint32_t *gate_fold = NULL)
{
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;
/* 192, not CUDA's 512. MEASURED on this box, bracketed interior minimum,
 * three runs per point at logI 14 / J 8192 / region 13 on oracle/c183:
 *
 *   threads   64      128     192     256     512
 *   apply    83.9    56.2    54.6    57.8    77.4  ms
 *
 * Run-to-run spread is under 1%% at each point, so the 29.5%% gap between 192
 * and CUDA's 512 is far outside the noise. 192 is 6 SIMD groups and keeps
 * (athr & 31) == 0, which k_apply's warp-ballot path requires.
 *
 * THE BOX: a 10-core M3 in a fanless MacBook Air that also drives the
 * display. The shape of the curve should carry to other Apple GPUs; the
 * exact optimum may not, and an M3 Max has four times the cores. Re-measure
 * there rather than trusting this number. */
    const int athr = cfg->apply_threads ? cfg->apply_threads : 192;
    const mtlStream_t st = bk->stream;
    int rc = -1;

#define SLAB_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    /* Preserve the pre-slab timing boundary: fill includes clearing the bucket
     * cursors/overflow counter, just as pipe_side_perq did. */
    wd_phase(side ? "slab.fill.side1" : "slab.fill.side0");
    /* k_fill_atomic reads S->plat and S->walk_cur, which k_transform wrote on
     * the stream pipe_side_prepare_q was given: the LEGACY DEFAULT STREAM,
     * except under --q-pipeline, where it is this same sieve stream. With the
     * concurrent arm's blocking side streams that ordering is implicit and
     * free -- and it is also ambient: compiling with `--default-stream
     * per-thread`, or ever creating those with mtlStreamNonBlocking, removes
     * it silently and lets fill race the transform for uninitialised
     * plattices. The failure would be wrong log sums in the CONCURRENT arm
     * only, which the serial-vs-concurrent identity gate is the one test that
     * could catch and the serial gates could not. Wait on the transform's own
     * end event instead, so the dependency is stated rather than inherited.
     * Costs nothing on stream 0, or on the stream the transform ran on. */
    if (st) SLAB_CK(mtlStreamWaitEvent(st, S->ev[4], 0));
    SLAB_CK(mtlEventRecord(S->ev[1], st));
    SLAB_CK(mtlMemsetAsync(bk->cursor, 0, (size_t)nregion * 4, st));
    SLAB_CK(mtlMemsetAsync(bk->overflow, 0, 4, st));
    {
        /* --fill-gate: gate_bits is the OTHER side's survivor bitmap for this
         * slab, already complete (the serial arm joins it first). The fold
         * runs inside the fill window so the gate's cost is billed to fill. */
        const uint32_t *gate = gate_bits;
        const int gshift = gate_bits ? cfg->fill_gate_shift : 0;
        if (gate_bits && gshift) {
            /* rounded up: a bitmap tail shorter than 2^gshift words still
             * gets a gate word, which k_gate_fold fills from what exists */
            const uint32_t nbw = xmax >> 5;
            const uint32_t ngword = (nbw + (1u << gshift) - 1) >> gshift;
            MTL_LAUNCH(k_gate_fold, (ngword + 255) / 256, 256, 0, st, gate_bits, gate_fold, gshift, ngword, nbw);
            gate = gate_fold;
        }
        const uint32_t nrow = S->nrow;
        if (nrow) {
            const uint32_t rows = xmax >> cfg->logI;
            const uint32_t nband = pipe_row_bands(cfg, rows);
            const uint64_t nthr = (uint64_t)((nrow + 31u) >> 5) * nband * 32u;
            /* Fine-slice ids when the side has them (the bucket resieve
             * reads them back), the original slices otherwise. */
            if (S->fine)
                MTL_LAUNCH_NAMED((SLABBED ? "k_fill_rows_1_uint32_t" : "k_fill_rows_0_uint32_t"), (unsigned)((nthr + 255) / 256), 256, 0, st, S->primes, S->rowrt, S->fine, nrow, nband, rows, cfg->logI, log_region, bk->cursor, bk->bucket, bk->cap, bk->overflow, cfg->sieve_skip, j_base, gate, gshift, (int)FINE_SHIFT);
            else
                MTL_LAUNCH_NAMED((SLABBED ? "k_fill_rows_1" : "k_fill_rows_0"), (unsigned)((nthr + 255) / 256), 256, 0, st, S->primes, S->rowrt, S->slice, nrow, nband, rows, cfg->logI, log_region, bk->cursor, bk->bucket, bk->cap, bk->overflow, cfg->sieve_skip, j_base, gate, gshift, 16);
            /* Only k_resieve_scatter reads the row tier's walk state, and only
             * when --resieve-rows is off; nothing else needs it. */
            if (SLABBED && !cfg->resieve_rows)
                MTL_LAUNCH(k_rows_walk_next, (nrow + 255) / 256, 256, 0, st, S->primes, S->rowrt, nrow, j_base + rows, cfg->logI, S->walk_cur, S->walk_next);
        }
        /* The rest by pointer offset: k_fill_atomic indexes every per-entry
         * array by the same k. */
        if (fb->n > nrow && S->fine)
            MTL_LAUNCH_NAMED((SLABBED ? "k_fill_atomic_4_1_uint32_t" : "k_fill_atomic_4_0_uint32_t"), fblocks, fthreads, 0, st, S->plat + nrow, S->fine + nrow, fb->n - nrow, xmax, cfg->logI, log_region, bk->cursor, bk->bucket, bk->cap, bk->overflow, SLABBED ? S->walk_cur + nrow : nullptr, SLABBED ? S->walk_next + nrow : nullptr, cfg->sieve_skip, j_base, gate, gshift, (int)FINE_SHIFT);
        else if (fb->n > nrow)
            MTL_LAUNCH_NAMED((SLABBED ? "k_fill_atomic_4_1" : "k_fill_atomic_4_0"), fblocks, fthreads, 0, st, S->plat + nrow, S->slice + nrow, fb->n - nrow, xmax, cfg->logI, log_region, bk->cursor, bk->bucket, bk->cap, bk->overflow, SLABBED ? S->walk_cur + nrow : nullptr, SLABBED ? S->walk_next + nrow : nullptr, cfg->sieve_skip, j_base, gate, gshift, 16);
    }
    SLAB_CK(mtlEventRecord(S->ev[2], st));
    SLAB_CK(mtlMemsetAsync(S->d_nsurv, 0, 4, st));
    /* Provably redundant on every production geometry: k_apply's warp-ballot
     * path stores every word of the slab bitmap unconditionally, and this
     * function has both inputs its guard tests (log_region and athr, above), so
     * `if (cfg->log_region < 5 || (athr & 31))` would skip the clear safely.
     * Measured cost of keeping it: ~512 MB of device writes per q at I16/J32768
     * x 4 slabs x 2 sides, about 0.7 ms against ~440, i.e. 0.16%.
     *
     * It stays anyway, and the honest reason is not "avoiding coupling" -- the
     * coupling already exists implicitly. It is that the guard would restate
     * k_apply's launch-shape condition in a second place, where it can rot out
     * of sync with the kernel's, and the payoff is 0.16%. Revisit if the clear
     * ever shows up in a profile. */
    SLAB_CK(mtlMemsetAsync(S->survbits, 0, (size_t)nbitword * 4, st));
    wd_phase(side ? "slab.apply.side1" : "slab.apply.side0");
    {   /* The instance with the two-step tail only when this q has one: at
         * logI <= 15 it never does, and that instance's thread tier is the
         * one-step loop alone (sieve_small, SSTAIL; finding 111). The choice
         * is ss_needs_tail, never a geometry test: the one-step instance on a
         * table with a tail would give silently wrong residues, and traps.
         * Its own scope, so the SLAB_CK gotos above do not jump past it. */
        const char *const apply_kern = ss_needs_tail(S->nsplit, S->nsmall)
            ? (SLABBED ? "k_apply_16_1_1_1_1" : "k_apply_16_1_1_0_1")
            : (SLABBED ? "k_apply_16_1_1_1_0" : "k_apply_16_1_1_0_0");
        MTL_LAUNCH_NAMED(apply_kern, nregion, athr, S->apply_smem, st,
            (const uint32_t *)bk->bucket, bk->cursor, bk->cap, cfg->logI, log_region, S->slice_logp, S->nslice_pow2, S->N, S->CINIT, S->CINIT - S->BOUND, S->tconst, nullptr, S->d_nsurv, nullptr, 0xFFFFFFFFu, S->sp, S->srt, S->sg, S->slp, S->smag, S->nsmall, S->nblk, S->nwrp, S->nsplit, 0xFFFFFFFFu, nullptr, S->survbits, cfg->not_both_even, cfg->sieve_skip, j_base, S->lut_shift);
    }
    SLAB_CK(mtlEventRecord(S->ev[3], st));
    /* The join's two counts, queued behind apply on the same stream: the
     * join then waits once (ev[5]) and reads host memory. */
    SLAB_CK(mtlMemcpyAsync(&S->h_rb->nsurv, S->d_nsurv, 4,
                            mtlMemcpyDeviceToHost, st));
    SLAB_CK(mtlMemcpyAsync(&S->h_rb->overflow, bk->overflow, 4,
                            mtlMemcpyDeviceToHost, st));
    SLAB_CK(mtlEventRecord(S->ev[5], st));
    rc = 0;
done:
#undef SLAB_CK
    return rc;
}

/* Await a slab issued above and classify it. Returns 0 = usable, 1 = SOFT
 * (skip this slab, the band lives), -1 = a real device/API failure.
 *
 * The two counts arrive in S->h_rb by async copies queued behind apply on
 * this side's stream, so one wait on ev[5] (after ev[3], so ev[1..3] are
 * complete for the timings) delivers them. They used to be two synchronous
 * mtlMemcpy calls after the ev[3] wait: four round trips per q at 35-70 us
 * each under WSL2, on the critical path (finding 113). That also removes a
 * whole-device barrier: a synchronous copy on the legacy stream waited for
 * every blocking stream, so join(1) used to wait for side 0 as well. Now each
 * join waits for its own side only; the callers join both sides before using
 * either, which is all they relied on. */
static int pipe_side_sieve_join(int side, pside_t *S,
                                float *t_fill, float *t_apply,
                                int concurrent_join)
{
    int rc = -1;
#define SLAB_CK(x) do { if (CUDA_CHECKED(x)) goto done; } while (0)
    /* Under --fill-concurrent all four fill/apply phases above are async
     * ISSUES that flash past in microseconds, and the run then parks here for
     * the duration of BOTH sides' work -- so naming this side would tell an
     * operator (and --watchdog-kill) that side 1 is stuck when it may well be
     * side 0's k_apply. Name what is actually being awaited. */
    wd_phase(concurrent_join ? "slab.sync.both"
                             : (side ? "slab.sync.side1" : "slab.sync.side0"));
    SLAB_CK(mtlEventSynchronize(S->ev[5]));
    SLAB_CK(mtlGetLastError());
    *t_fill  = time_kernel(S->ev[1], S->ev[2]);
    *t_apply = time_kernel(S->ev[2], S->ev[3]);
    S->nsurv = S->h_rb->nsurv;
    {
        const uint32_t hov = S->h_rb->overflow;
        if (hov) {
            /* SOFT failure: rc = 1, not -1. Dropped bucket records can only
             * LOWER a position's log sum, so an overflow costs survivors it
             * never invents -- and every relation that does come out is still
             * verified downstream. The caller therefore skips this slab and
             * keeps the band alive, rather than failing a whole BOINC task
             * over a shortfall that was 1 record in the field report that
             * prompted this. Hard SLAB_CK failures above still return -1.
             *
             * Rate-limited: under BOINC stderr is uploaded, and a band that
             * overflows on many q would otherwise ship thousands of identical
             * lines. The end-of-band summary carries the true total. */
            /* The budget is ONE PER PROCESS, which is why the counter is at
             * file scope (line ~352) and not a static local here. It was
             * originally file scope because this code lived in a template and
             * a static local in a template is per-instantiation -- 8 per
             * SLABBED specialisation, 16 in a binary carrying both. That
             * reason is gone now that the overflow branch sits in a
             * non-template function; the budget being process-wide is not.
             * Do not "simplify" it back to a static local. */
            if (g_warned_bucket_ovf < 8)
                runlog_warn("  side %d: bucket array OVERFLOWED by %u records"
                            " -- slab skipped%s", side, hov,
                            ++g_warned_bucket_ovf == 8
                                ? " (further overflows summarised"
                                  " at end of band)" : "");
            else
                g_warned_bucket_ovf++;
            rc = 1;
            goto done;
        }
    }
    rc = 0;

done:
#undef SLAB_CK
    return rc;
}

/* ======================= the production TD path ========================== *
 *
 * run_td_stage is the MEASUREMENT harness. Calling it per special-q, per side,
 * is what made the pipeline's post-sieve cost 222 ms/q: it runs the rank scan,
 * the emission, the resieve and the classification three times each for a
 * best-of-three, plus two diagnostic variants of k_td that exist only to
 * separate the congruence test from the divisions it triggers, plus a dense
 * recording pass over every survivor -- and it rebuilds the rank scan and the
 * emission for the SECOND side even though both are functions of the shared
 * two-sided bitmap alone.
 *
 * This is the same arithmetic with the harness taken out. Four things change:
 *
 *   1. Rank scan, emission and the resieve summary run ONCE per q, not once
 *      per side. They read the two-sided bitmap; neither side owns them.
 *   2. Every buffer is allocated once for the whole band, not once per q.
 *      Only a survivor count larger than anything seen so far reallocates.
 *   3. The two sides' acceptances are intersected ON DEVICE, and the recording
 *      pass runs over that compacted list. Recording is the only pass that
 *      writes 64 words per thread; over ~240,000 survivors that is a 61 MB
 *      matrix per side, and over the ~1,900 joint candidates it is 486 KB.
 *   4. The reconstruction gate is an explicitly separate phase on the first q,
 *      excluded from the band timing rather than hidden inside it.
 *
 * What does NOT change is the arithmetic: same kernels, same production
 * settings (resieve unroll 4, 1 summary bit per 64 positions). The output of
 * this path is diffed byte-for-byte against the harness path, which is the
 * only reason a restructure this large is safe to make.
 */

/* Large primes kept per survivor, as run_td_stage. Overrun is DETECTED, not
 * silently dropped: k_resieve_scatter counts the records it could not place
 * into d_ovf, and k_td/k_td_leader raise TDF_LIST_TRUNCATED when they read a
 * count past K -- at CONSUMPTION, not at scatter -- after which the slab is
 * skipped. So a value that is too small costs yield and never emits a partial
 * record. Passed to every kernel as a runtime `uint32_t K`, and no fixed-size
 * array is sized by it, but it is NOT free above: see the Makefile's PIPE_K
 * for why the accepted range stops at 32 rather than at TD_FMAX's 64.
 *
 * OVERRIDABLE FROM THE BUILD, via its own Makefile variable rather than DEFS,
 * for the reason spelled out for CF_LMAX there: a non-empty DEFS marks a
 * pricing build and bench refuses to emit relations from one.
 * degradecheck.sh needs both -- a small PIPE_K so every slab truncates,
 * and --relations to assert the degraded stop still keeps its .part. */
#ifndef PIPE_K
#define PIPE_K 16
#endif

/* Bucket capacity as a percentage of the estimate's: test builds only (see
 * its use in run_pipeline_impl). */
#ifndef PIPE_CAP_PCT
#define PIPE_CAP_PCT 100
#endif

/* TD's three per-slab counters, as ONE device allocation (pipe_td_t.d_ctr)
 * so the host reads them with one copy. Order is the copy's layout; the
 * 64-bit word is 8-byte aligned in both the device block and the pinned
 * mirror (pipe_td_t.h_rb). */
typedef struct {
    uint32_t flags, nacc;
    unsigned long long ovf;
} td_ctr_t;

typedef struct {
    double rank, emit, summary, resieve, td, classify, compact, record;
    double resieve0;    /* side 0's share of resieve, reported beside it */
    double readback, join, hostq, cofac;
} pipe_tm_t;

typedef struct {
    uint32_t nbitword, ngroup, nb, nsumword;
    uint32_t scap, ccap;          /* survivors, joint candidates */
    /* shared by both sides: functions of the two-sided bitmap alone */
    uint32_t *d_cnt, *d_gbase, *d_bsum, *d_sum;
    uint32_t *d_x;  int64_t *d_a, *d_b;
    uint32_t *d_flags;  unsigned long long *d_ovf;
    /* d_flags, d_nacc and d_ovf are views into ONE device block, d_ctr, so
     * the host reads all three with one copy. */
    td_ctr_t *d_ctr;
    /* PINNED readback slot for the per-slab counts the host needs: the
     * survivor total's two parts (phase 1), the counter block (after
     * k_scatter_sel) and the intersection's count. Filled by async copies
     * behind the kernels and read after one wait each, instead of
     * synchronous mtlMemcpy round trips (35-70 us each under WSL2,
     * finding 113). */
    struct {
        uint32_t base, cnt;     /* last group's scan base and count */
        td_ctr_t ctr;           /* flags, accepted count, overflow word */
        uint32_t isect;         /* the intersection's survivor count */
    } *h_rb;
    /* joint acceptance */
    uint32_t *d_aflag, *d_aoff, *d_absum, *d_sel, *d_nacc;
    /* per side */
    uint32_t  *d_plist[2], *d_pcnt[2];
    bn_t      *d_cof[2];
    uint8_t   *d_cofbits[2], *d_status[2];
    tdpoly_t  *d_poly[2];
    uint32_t  *d_stats;          /* {already-relations, candidates, overflows} */
    tdsmall_t *d_sm[2], *h_sm[2];
    uint32_t   nsm[2], nsmcap[2];
    tdpoly_t   hpoly[2];
    /* compacted candidate records, device then pinned host */
    int64_t  *d_ca, *d_cb, *h_ca, *h_cb;
    bn_t     *d_ccof[2], *h_ccof[2];
    uint8_t  *d_cbits[2], *h_cbits[2];
    uint32_t *d_cfac[2], *h_cfac[2], *d_cfn[2], *h_cfn[2];
    mtlEvent_t ev[16];
} pipe_td_t;

static void pipe_td_free(pipe_td_t *C)
{
    mtlFree(C->d_cnt); mtlFree(C->d_gbase); mtlFree(C->d_bsum);
    mtlFree(C->d_sum); mtlFree(C->d_x); mtlFree(C->d_a); mtlFree(C->d_b);
    mtlFree(C->d_ctr);     /* d_flags, d_nacc, d_ovf point into it */
    mtlFree(C->d_aflag); mtlFree(C->d_aoff); mtlFree(C->d_absum);
    mtlFree(C->d_sel);
    mtlFree(C->d_ca); mtlFree(C->d_cb);
    if (C->h_ca) mtlFreeHost(C->h_ca);
    if (C->h_cb) mtlFreeHost(C->h_cb);
    if (C->h_rb) mtlFreeHost(C->h_rb);
    mtlFree(C->d_stats);
    for (int s = 0; s < 2; s++) {
        mtlFree(C->d_plist[s]); mtlFree(C->d_pcnt[s]);
        mtlFree(C->d_cof[s]); mtlFree(C->d_cofbits[s]); mtlFree(C->d_status[s]);
        mtlFree(C->d_poly[s]); mtlFree(C->d_sm[s]);
        mtlFree(C->d_ccof[s]); mtlFree(C->d_cbits[s]);
        mtlFree(C->d_cfac[s]); mtlFree(C->d_cfn[s]);
        if (C->h_sm[s])   mtlFreeHost(C->h_sm[s]);
        if (C->h_ccof[s]) mtlFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) mtlFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s]) mtlFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])  mtlFreeHost(C->h_cfn[s]);
    }
    for (int k = 0; k < 16; k++) if (C->ev[k]) mtlEventDestroy(C->ev[k]);
    memset(C, 0, sizeof(*C));
}

/* Survivor-sized buffers. Grown, never shrunk: a band's first q pays for its
 * own count plus half, and in practice nothing after it reallocates. */
static int pipe_td_grow(pipe_td_t *C, uint32_t n)
{
    uint32_t cap, nab;
    if (n <= C->scap) return 0;
    cap = n + n / 2 + 1024;
    nab = (cap + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    mtlFree(C->d_x); mtlFree(C->d_a); mtlFree(C->d_b);
    mtlFree(C->d_aflag); mtlFree(C->d_aoff); mtlFree(C->d_absum);
    C->d_x = NULL; C->d_a = NULL; C->d_b = NULL;
    C->d_aflag = C->d_aoff = C->d_absum = NULL;
    CK(mtlMalloc(&C->d_x, (size_t)cap * 4));
    CK(mtlMalloc(&C->d_a, (size_t)cap * 8));
    CK(mtlMalloc(&C->d_b, (size_t)cap * 8));
    CK(mtlMalloc(&C->d_aflag, (size_t)cap * 4));
    CK(mtlMalloc(&C->d_aoff, (size_t)cap * 4));
    CK(mtlMalloc(&C->d_absum, (size_t)nab * 4));
    for (int s = 0; s < 2; s++) {
        mtlFree(C->d_plist[s]); mtlFree(C->d_pcnt[s]); mtlFree(C->d_cof[s]);
        mtlFree(C->d_cofbits[s]); mtlFree(C->d_status[s]);
        C->d_plist[s] = C->d_pcnt[s] = NULL; C->d_cof[s] = NULL;
        C->d_cofbits[s] = C->d_status[s] = NULL;
        CK(mtlMalloc(&C->d_plist[s], (size_t)cap * PIPE_K * 4));
        CK(mtlMalloc(&C->d_pcnt[s], (size_t)cap * 4));
        CK(mtlMalloc(&C->d_cof[s], (size_t)cap * sizeof(bn_t)));
        CK(mtlMalloc(&C->d_cofbits[s], (size_t)cap));
        CK(mtlMalloc(&C->d_status[s], (size_t)cap));
    }
    C->scap = cap;
    return 0;
}

/* Candidate-sized buffers, device and pinned host. */
static int pipe_td_grow_cand(pipe_td_t *C, uint32_t n)
{
    uint32_t cap;
    if (n <= C->ccap) return 0;
    cap = n + n / 2 + 1024;
    mtlFree(C->d_sel); mtlFree(C->d_ca); mtlFree(C->d_cb);
    C->d_sel = NULL; C->d_ca = C->d_cb = NULL;
    if (C->h_ca) mtlFreeHost(C->h_ca);
    if (C->h_cb) mtlFreeHost(C->h_cb);
    C->h_ca = C->h_cb = NULL;
    CK(mtlMalloc(&C->d_sel, (size_t)cap * 4));
    CK(mtlMalloc(&C->d_ca, (size_t)cap * 8));
    CK(mtlMalloc(&C->d_cb, (size_t)cap * 8));
    CK(mtlHostAlloc((void **)&C->h_ca, (size_t)cap * 8, mtlHostAllocDefault));
    CK(mtlHostAlloc((void **)&C->h_cb, (size_t)cap * 8, mtlHostAllocDefault));
    for (int s = 0; s < 2; s++) {
        mtlFree(C->d_ccof[s]); mtlFree(C->d_cbits[s]);
        mtlFree(C->d_cfac[s]); mtlFree(C->d_cfn[s]);
        C->d_ccof[s] = NULL; C->d_cbits[s] = NULL;
        C->d_cfac[s] = C->d_cfn[s] = NULL;
        if (C->h_ccof[s])  mtlFreeHost(C->h_ccof[s]);
        if (C->h_cbits[s]) mtlFreeHost(C->h_cbits[s]);
        if (C->h_cfac[s])  mtlFreeHost(C->h_cfac[s]);
        if (C->h_cfn[s])   mtlFreeHost(C->h_cfn[s]);
        C->h_ccof[s] = NULL; C->h_cbits[s] = NULL;
        C->h_cfac[s] = C->h_cfn[s] = NULL;
        CK(mtlMalloc(&C->d_ccof[s], (size_t)cap * sizeof(bn_t)));
        CK(mtlMalloc(&C->d_cbits[s], (size_t)cap));
        CK(mtlMalloc(&C->d_cfac[s], (size_t)cap * TD_FMAX * 4));
        CK(mtlMalloc(&C->d_cfn[s], (size_t)cap * 4));
        CK(mtlHostAlloc((void **)&C->h_ccof[s], (size_t)cap * sizeof(bn_t),
                         mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&C->h_cbits[s], (size_t)cap, mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&C->h_cfac[s], (size_t)cap * TD_FMAX * 4,
                         mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&C->h_cfn[s], (size_t)cap * 4, mtlHostAllocDefault));
    }
    C->ccap = cap;
    return 0;
}

/* q-independent setup: the exact polynomials, the rank tables sized by the
 * bitmap, and the pinned staging for the per-q small-prime table. */
static int pipe_td_init(pipe_td_t *C, const fb_t *fbs1, const fb_t *fbs0,
                        const poly_t *POLY, uint32_t nbitword)
{
    const fb_t *fbs[2];
    memset(C, 0, sizeof(*C));
    fbs[0] = fbs0; fbs[1] = fbs1;
    if (nbitword % TD_GROUP_W) {
        fprintf(stderr, "  pipeline: %u bitmap words is not a multiple of %d\n",
                nbitword, TD_GROUP_W);
        return -1;
    }
    C->nbitword = nbitword;
    C->ngroup = nbitword / TD_GROUP_W;
    C->nb = (C->ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    C->nsumword = ((nbitword / 2) + 31) / 32;
    for (int k = 0; k < 16; k++) CK(mtlEventCreate(&C->ev[k]));
    CK(mtlHostAlloc((void **)&C->h_rb, sizeof(*C->h_rb), mtlHostAllocDefault));
    CK(mtlMalloc(&C->d_cnt, (size_t)C->ngroup * 4));
    CK(mtlMalloc(&C->d_gbase, (size_t)C->ngroup * 4));
    CK(mtlMalloc(&C->d_bsum, (size_t)C->nb * 4));
    CK(mtlMalloc(&C->d_sum, (size_t)C->nsumword * 4));
    CK(mtlMalloc(&C->d_ctr, sizeof(td_ctr_t)));
    C->d_flags = &C->d_ctr->flags;
    C->d_nacc  = &C->d_ctr->nacc;
    C->d_ovf   = &C->d_ctr->ovf;
    CK(mtlMalloc(&C->d_stats, 12));
    CK(mtlMemset(C->d_stats, 0, 12));      /* accumulates over the whole band */
    for (int s = 0; s < 2; s++) {
        if (td_build_poly(&C->hpoly[s], POLY, s)) {
            fprintf(stderr, "  pipeline: could not parse side %d coefficients\n", s);
            return -1;
        }
        CK(mtlMalloc(&C->d_poly[s], sizeof(tdpoly_t)));
        CK(mtlMemcpy(C->d_poly[s], &C->hpoly[s], sizeof(tdpoly_t),
                      mtlMemcpyHostToDevice));
        C->nsmcap[s] = (fbs[s] && fbs[s]->n) ? fbs[s]->n : 0;
        if (C->nsmcap[s]) {
            CK(mtlHostAlloc((void **)&C->h_sm[s],
                             (size_t)C->nsmcap[s] * sizeof(tdsmall_t),
                             mtlHostAllocDefault));
            CK(mtlMalloc(&C->d_sm[s], (size_t)C->nsmcap[s] * sizeof(tdsmall_t)));
        }
    }
    return pipe_td_grow_cand(C, 8192);
}

/* Per-q, per-side host work: retransform the small-prime direct-test table
 * into the pinned buffer and upload it. Everything else about the side is
 * already resident. */
static int pipe_td_small(pipe_td_t *C, int side, const fb_t *fbs,
                         const qlat_t *L, int logI)
{
    if (!C->nsmcap[side]) { C->nsm[side] = 0; return 0; }
    C->nsm[side] = td_fill_small(fbs, L, logI, C->h_sm[side]);
    if (!C->nsm[side]) return -1;
    /* Async for the same reason as the four uploads in pipe_side_prepare_q,
     * and safe on the same grounds: h_sm is pinned and the stream drains
     * every q. */
    CK(mtlMemcpyAsync(C->d_sm[side], C->h_sm[side],
                       (size_t)C->nsm[side] * sizeof(tdsmall_t),
                       mtlMemcpyHostToDevice, 0));
    return 0;
}

/* Build the two direct-test tables once per special-q.  In slabbed mode their
 * congruence constants are then translated between slabs by the tiny
 * pipe_td_advance_small() kernel, keeping the billions-of-tests inner loop
 * unchanged. */
static int pipe_td_prepare_q(pipe_td_t *C, const fb_t *fbs1,
                             const fb_t *fbs0, const qlat_t *L,
                             const bench_cfg_t *cfg, pipe_tm_t *tm)
{
    const double h0 = host_ms();
    if (pipe_td_small(C, 1, fbs1, L, cfg->logI) ||
        pipe_td_small(C, 0, fbs0, L, cfg->logI)) return -1;
    tm->hostq += host_ms() - h0;
    return 0;
}

static int pipe_td_advance_small(pipe_td_t *C, uint32_t delta_j,
                                 int blocks, int threads)
{
    if (!delta_j) return 0;
    for (int s = 0; s < 2; s++) {
        if (!C->nsm[s]) continue;
        MTL_LAUNCH(k_tdsmall_advance, blocks, threads, 0, 0, C->d_sm[s], C->nsm[s], delta_j);
        CK(mtlGetLastError());
    }
    return 0;
}

/* ---- the first-q validation phase -------------------------------------- *
 *
 * Reconstruction: the recorded factors times the residual cofactor must
 * rebuild the exact norm. This runs the DENSE recording pass and checks every
 * candidate on the side, which is what run_td_stage did -- deliberately, so
 * the check keeps its old strength rather than narrowing to the joint set the
 * production path records. It reads back 61 MB per side and rebuilds a 224-bit
 * norm per candidate on the host, so it is a separate phase whose cost is
 * reported on its own and excluded from the band average. */
template <bool SLABBED>
static int pipe_td_verify(pipe_td_t *C, int side, uint32_t n, int logI,
                          uint32_t sq, const bench_cfg_t *cfg,
                          int blocks, int threads, uint32_t j_base,
                          uint32_t *gate_found)
{
    uint32_t *d_fac = NULL, *d_faccnt = NULL;
    uint32_t *hfac = NULL, *hfn = NULL;
    uint8_t *hstat = NULL;
    bn_t *hcof = NULL;
    int64_t *ha = NULL, *hb = NULL;
    uint32_t checked = 0, bad = 0, overflowed = 0, maxfac = 0;
    const tdpoly_t *P = &C->hpoly[side];
    int rc = 0;

#define VERIFY_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto out; } } while (0)
    VERIFY_CK(mtlMalloc(&d_fac, (size_t)n * TD_FMAX * 4));
    VERIFY_CK(mtlMalloc(&d_faccnt, (size_t)n * 4));
    MTL_LAUNCH_NAMED((SLABBED ? "k_td_1_1_0_1" : "k_td_1_1_0_0"), blocks, threads, 0, 0, C->d_a, C->d_b, C->d_x, nullptr, n, logI, C->d_poly[side], sq, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_sm[side], C->nsm[side], C->d_cof[side], C->d_cofbits[side], C->d_flags, nullptr, d_fac, d_faccnt, TD_FMAX, j_base);
    if (CUDA_CHECKED(mtlDeviceSynchronize()) ||
        CUDA_CHECKED(mtlGetLastError())) {
        fprintf(stderr, "  verify: recording pass failed\n");
        rc = -1;
        goto out;
    }
    hfac = (uint32_t *)malloc((size_t)n * TD_FMAX * 4);
    hfn = (uint32_t *)malloc((size_t)n * 4);
    hstat = (uint8_t *)malloc((size_t)n);
    hcof = (bn_t *)malloc((size_t)n * sizeof(bn_t));
    ha = (int64_t *)malloc((size_t)n * 8);
    hb = (int64_t *)malloc((size_t)n * 8);
    if (!hfac || !hfn || !hstat || !hcof || !ha || !hb) { rc = -1; goto out; }
    VERIFY_CK(mtlMemcpy(hfac, d_fac, (size_t)n * TD_FMAX * 4,
                         mtlMemcpyDeviceToHost));
    VERIFY_CK(mtlMemcpy(hfn, d_faccnt, (size_t)n * 4,
                         mtlMemcpyDeviceToHost));
    VERIFY_CK(mtlMemcpy(hstat, C->d_status[side], (size_t)n,
                         mtlMemcpyDeviceToHost));
    VERIFY_CK(mtlMemcpy(hcof, C->d_cof[side], (size_t)n * sizeof(bn_t),
                         mtlMemcpyDeviceToHost));
    VERIFY_CK(mtlMemcpy(ha, C->d_a, (size_t)n * 8,
                         mtlMemcpyDeviceToHost));
    VERIFY_CK(mtlMemcpy(hb, C->d_b, (size_t)n * 8,
                         mtlMemcpyDeviceToHost));

    for (uint32_t k = 0; k < n; k++) {
        bns_t acc; bn_t t;
        int64_t a, b; uint64_t ua, ub; int sa, sb;
        if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
        if (hfn[k] > maxfac) maxfac = hfn[k];
        if (hfn[k] > TD_FMAX) { overflowed++; continue; }
        checked++;
        a = ha[k]; b = hb[k];
        ua = (uint64_t)(a < 0 ? -a : a); ub = (uint64_t)(b < 0 ? -b : b);
        sa = (a < 0) ? -1 : 1; sb = (b < 0) ? -1 : 1;
        bns_zero(&acc);
        for (int d = 0; d <= P->deg; d++) {
            int sgn = P->sign[d];
            t = P->c[d];
            if (bn_is_zero(&t)) continue;
            for (int e = 0; e < d; e++) { bn_mul_u64(&t, ua); sgn *= sa; }
            for (int e = 0; e < P->deg - d; e++) { bn_mul_u64(&t, ub); sgn *= sb; }
            bns_addmag(&acc, &t, sgn);
        }
        t = hcof[k];
        for (uint32_t z = 0; z < hfn[k]; z++)
            bn_mul_u64(&t, (uint64_t)hfac[(size_t)k * TD_FMAX + z]);
        if (bn_cmp(&t, &acc.m) != 0) bad++;
    }
    printf("    side %d: factors x cofactor == norm  %u of %u  %s"
           "   (most factors %u of %d)\n",
           side, checked - bad, checked, bad ? "FAIL" : "PASS", maxfac, TD_FMAX);
    if (bad) rc = -1;
    if (overflowed) {
        fprintf(stderr, "    side %d: %u candidates had more than %d factors;"
                " raise TD_FMAX\n", side, overflowed, TD_FMAX);
        rc = -1;
    }
    if (cfg->cofgate &&
        td_gate_cofactors_part(cfg->cofgate, n, ha, hb, hcof, side,
                               gate_found) != 0) rc = -1;

out:
    free(hfac); free(hfn); free(hstat); free(hcof); free(ha); free(hb);
    mtlFree(d_fac); mtlFree(d_faccnt);
#undef VERIFY_CK
    return rc;
}

/* ---- one special-q of trial division, both sides ----------------------- */

/* What trial division reads of a side, and nothing else. Under --q-pipeline
 * the next q's sieve is already rewriting the rest of pside_t -- its norm, its
 * small-sieve tables and staging, its events -- while this q is trial-divided,
 * and only plat and rowrt are doubled to allow it (finding 108). A TD change
 * that needs another per-q field has to add it here, and with it decide
 * whether the pipeline must double that field too. primes and nfb are fixed
 * for the band; walk_cur, read only under slabbing, is the cursor this slab
 * started from, which pside_rotate_walk keeps out of the next slab's way. */
typedef struct {
    const plat_t   *plat;
    const uint32_t *primes, *rowrt;
    const uint64_t *walk_cur;
    uint32_t        nfb, nrow;
    /* The bucket resieve's tables: fixed for the band, like primes. fine is
     * NULL when the side keeps the original slices (no bucket resieve). */
    const uint32_t *fine, *fine_start, *roots;
} ptd_side_t;

static inline ptd_side_t pipe_td_side(const pside_t *S)
{
    ptd_side_t v;
    v.plat = S->plat; v.primes = S->primes; v.rowrt = S->rowrt;
    v.walk_cur = S->walk_cur; v.nfb = S->nfb; v.nrow = S->nrow;
    v.fine = S->fine; v.fine_start = S->fine_start; v.roots = S->roots;
    return v;
}

template <bool SLABBED>
static int pipe_td_perq(pipe_td_t *C, const qlat_t *L, const bench_cfg_t *cfg,
                        const ptd_side_t *S1, const ptd_side_t *S0,
                        const uint32_t *d_two, uint32_t xmax,
                        uint32_t j_base, int blocks, int threads, int verify,
                        uint32_t *n_out, uint32_t *nacc_out, pipe_tm_t *tm,
                        double *t_verify, int want_host, int accumulate_stats,
                        uint32_t gate_found[2],
                        const pbkt_t *const bkr[2] = NULL,
                        const std::function<int()> *after_resieve = NULL)
{
    const ptd_side_t *S[2];
    const uint32_t lpb[2] = {cfg->lpb0, cfg->lpb};
    const uint32_t mfb[2] = {cfg->mfb0, cfg->mfb};
    const uint32_t lim[2] = {cfg->lim0, cfg->lim};
    mtlEvent_t *E = C->ev;
    uint32_t n = 0, nacc = 0, nab, hflags = 0;
    unsigned long long hovf = 0;
    const uint32_t nbitword = xmax >> 5;
    const uint32_t ngroup = nbitword / TD_GROUP_W;
    const uint32_t nb = (ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;

    S[0] = S0; S[1] = S1;

    if (!ngroup || nbitword % TD_GROUP_W) {
        fprintf(stderr,
                "  pipeline: slab area %u has %u bitmap words; trial division"
                " requires a nonzero multiple of %u\n",
                xmax, nbitword, (unsigned)TD_GROUP_W);
        return -1;
    }

    /* ---- phase 1: rank over the two-sided bitmap (shared) ---- */
    CK(mtlEventRecord(E[0]));
    MTL_LAUNCH(k_group_counts, blocks, threads, 0, 0, d_two, ngroup, C->d_cnt);
    MTL_LAUNCH(k_scan_pass1, nb, TD_SCAN_BLK, 0, 0, C->d_cnt, ngroup, C->d_gbase, C->d_bsum);
    MTL_LAUNCH(k_scan_pass2, 1, 1024, 0, 0, C->d_bsum, nb);
    MTL_LAUNCH(k_scan_pass3, nb, TD_SCAN_BLK, 0, 0, C->d_gbase, ngroup, C->d_bsum);
    CK(mtlEventRecord(E[1]));
    /* The survivor total, base + count of the last group (two arrays, so two
     * copies), queued after E[1] so the rank-scan timing excludes them, and
     * one wait on the legacy stream, which is where they and every TD
     * kernel run -- not two synchronous round trips. */
    CK(mtlMemcpyAsync(&C->h_rb->base, C->d_gbase + ngroup - 1, 4,
                       mtlMemcpyDeviceToHost, 0));
    CK(mtlMemcpyAsync(&C->h_rb->cnt, C->d_cnt + ngroup - 1, 4,
                       mtlMemcpyDeviceToHost, 0));
    CK(mtlStreamSynchronize(0)); CK(mtlGetLastError());
    n = C->h_rb->base + C->h_rb->cnt;
    *n_out = n; *nacc_out = 0;
    /* A slab can legitimately contain no survivors, especially a short tail.
     * Preserve the original q-level invariant after all slabs have run rather
     * than turning an empty individual slab into a failed q. */
    if (!n) return 0;
    if (pipe_td_grow(C, n)) return -1;

    /* ---- phase 2: emit, summary, then both sides ---- */
    CK(mtlMemset(C->d_flags, 0, 4));
    CK(mtlMemset(C->d_ovf, 0, 8));
    CK(mtlEventRecord(E[2]));
    MTL_LAUNCH_NAMED((SLABBED ? "k_emit_ranked_1" : "k_emit_ranked_0"), blocks, threads, 0, 0, d_two, C->d_gbase, nbitword, cfg->logI, L->a0, L->a1, L->b0, L->b1, C->d_x, C->d_a, C->d_b, n, j_base);
    CK(mtlEventRecord(E[3]));
    /* 1 summary bit per 2 bitmap words == per 64 positions, the setting the
     * resieve sweep selected. */
    MTL_LAUNCH(k_build_summary_g, blocks, threads, 0, 0, d_two, nbitword, 2u, C->d_sum);
    CK(mtlEventRecord(E[4]));

    /* Both sides' resieves first, then the rest of trial division. The
     * resieve is the last reader of this slab's bucket records, so the hook
     * between the two -- --q-pipeline's issue of the next slab or special-q,
     * whose fill rewrites those arrays -- runs only once they are queued, and
     * makes its stream wait on E[14]. Same kernels, same inputs, in a
     * different order: the relations cannot change.
     *
     * Events: resieve side 1 E[4]->E[5], side 0 E[5]->E[14]; trial division
     * side 1 E[15]->E[6], side 0 E[7]->E[9]; classify E[6]->E[7] and
     * E[9]->E[10]. */
    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;          /* side 1 first, as before */
        CK(mtlMemsetAsync(C->d_pcnt[side], 0, (size_t)n * 4));
        if (bkr && bkr[side] && S[side]->fine) {
            /* This side's records for this slab are still in its bucket
             * array -- the --fill-rows tier's included, which writes the same
             * fine-slice ids: read the primes back from them. */
            MTL_LAUNCH(k_resieve_bucket, xmax >> cfg->log_region, 256, 0, 0, (const uint32_t *)bkr[side]->bucket, bkr[side]->cursor, bkr[side]->cap, cfg->log_region, d_two, C->d_gbase, C->d_a, C->d_b, S[side]->fine_start, S[side]->primes, S[side]->roots, S[side]->plat, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_ovf);
        } else {
            /* --resieve-rows: the --fill-rows tier, banded by row; the FK
             * walk takes the rest by pointer offset, as in the fill. */
            const uint32_t nr = cfg->resieve_rows ? S[side]->nrow : 0;
            if (nr) {
                const uint32_t rows = xmax >> cfg->logI;
                const uint32_t nband = pipe_row_bands(cfg, rows);
                const uint64_t nthr = (uint64_t)nr * nband;
                MTL_LAUNCH_NAMED((SLABBED ? "k_resieve_rows_4_1" : "k_resieve_rows_4_0"), (unsigned)((nthr + 255) / 256), 256, 0, 0, S[side]->primes, S[side]->rowrt, nullptr, nr, nband, rows, cfg->logI, C->d_sum, d_two, C->d_gbase, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_ovf, 6, j_base);
            }
            if (S[side]->nfb > nr)
                MTL_LAUNCH_NAMED((SLABBED ? "k_resieve_scatter_4_1" : "k_resieve_scatter_4_0"), blocks, threads, 0, 0, S[side]->plat + nr, S[side]->primes + nr, nullptr, S[side]->nfb - nr, xmax, cfg->logI, C->d_sum, d_two, C->d_gbase, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_ovf, 6, SLABBED ? S[side]->walk_cur + nr : nullptr);
        }
        CK(mtlEventRecord(E[si ? 14 : 5]));
    }
    if (after_resieve && *after_resieve && (*after_resieve)()) return -1;
    CK(mtlEventRecord(E[15]));
    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;
        MTL_LAUNCH_NAMED((SLABBED ? "k_td_1_0_0_1" : "k_td_1_0_0_0"), blocks, threads, 0, 0, C->d_a, C->d_b, C->d_x, nullptr, n, cfg->logI, C->d_poly[side], side == cfg->sq_side ? (uint32_t)L->q : 0u, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_sm[side], C->nsm[side], C->d_cof[side], C->d_cofbits[side], C->d_flags, nullptr, nullptr, nullptr, 0, j_base);
        CK(mtlEventRecord(E[si ? 9 : 6]));
        MTL_LAUNCH(k_classify, blocks, threads, 0, 0, C->d_cof[side], C->d_cofbits[side], C->d_b, n, lpb[side], mfb[side], (double)lim[side], C->d_status[side]);
        CK(mtlEventRecord(E[si ? 10 : 7]));
    }

    /* ---- joint acceptance, compacted on device ---- */
    nab = (n + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    MTL_LAUNCH(k_accept_flags, blocks, threads, 0, 0, C->d_status[0], C->d_status[1], n, C->d_aflag);
    MTL_LAUNCH(k_scan_pass1, nab, TD_SCAN_BLK, 0, 0, C->d_aflag, n, C->d_aoff, C->d_absum);
    MTL_LAUNCH(k_scan_pass2, 1, 1024, 0, 0, C->d_absum, nab);
    MTL_LAUNCH(k_scan_pass3, nab, TD_SCAN_BLK, 0, 0, C->d_aoff, n, C->d_absum);
    MTL_LAUNCH(k_scatter_sel, blocks, threads, 0, 0, C->d_aflag, C->d_aoff, n, C->d_sel, C->ccap, C->d_nacc);
    CK(mtlEventRecord(E[11]));
    /* flags, the accepted count and the overflow word: the one counter
     * block, one async copy queued after E[11] (so the compact timing
     * excludes it) and one wait on the legacy stream -- not three
     * synchronous round trips. The count is read before the flags are
     * judged; harmless, and a slab the flags reject never uses it. */
    CK(mtlMemcpyAsync(&C->h_rb->ctr, C->d_ctr, sizeof(td_ctr_t),
                       mtlMemcpyDeviceToHost, 0));
    CK(mtlStreamSynchronize(0)); CK(mtlGetLastError());
    tm->emit     += time_kernel(E[2], E[3]);
    tm->summary  += time_kernel(E[3], E[4]);
    tm->resieve  += time_kernel(E[4], E[14]);
    tm->resieve0 += time_kernel(E[5], E[14]);
    tm->td       += time_kernel(E[15], E[6]) + time_kernel(E[7], E[9]);
    tm->classify += time_kernel(E[6], E[7]) + time_kernel(E[9], E[10]);
    tm->compact  += time_kernel(E[10], E[11]);
    tm->rank     += time_kernel(E[0], E[1]);

    hflags = C->h_rb->ctr.flags;
    hovf = C->h_rb->ctr.ovf;
    /* The TRUNCATED LIST is SOFT (return 1): it says this slab's trial
     * division is untrustworthy, not that the device or the run is broken.
     * Norm overflow, immediately below, is NOT -- see its own note. It is
     * tested here, before the caller emits anything, so skipping the slab
     * means no incomplete or wrong record ever reaches the relation file --
     * the cost is the slab's yield. Under BOINC that is the difference
     * between a task that returns slightly less and a task that returns
     * nothing at all. Rate-limited for the same reason as the bucket
     * overflow: stderr is uploaded.
     *
     * A truncated list usually means a MISCONFIGURED job (PIPE_K too small)
     * rather than a transient, so a run that trips it on every slab now
     * yields almost nothing instead of erroring. The end-of-band counters are
     * what make that visible; do not remove them. */

    /* NORM OVERFLOW IS FATAL, DELIBERATELY, AND WAS BRIEFLY NOT.
     *
     * It is not a condition a well-formed run can reach, so softening it did
     * not buy robustness -- it converted a broken invariant into a silent
     * yield loss. pipe_side_prepare_q already refuses any (q,rho) whose norm
     * bound does not fit: norm_fits_exact tests
     * norm_exact_bound_bits == log2M + log2(deg+1), evaluated per side per q
     * over the FULL J, and PIPE_Q_SKIP passes the q over before it is ever
     * sieved. So by the time k_td runs, the exact norm of every cell in this
     * q has been PROVEN to fit.
     *
     * That proof reaches the intermediates too, which is what actually
     * matters here, because the flag is raised by bn_mul_u64/bns_addmag
     * mid-Horner rather than by the finished norm. Each term is built as c_k,
     * then multiplied by |a| k times and by |b| (deg-k) times, so with
     * |b| >= 1 every partial product divides the finished term and is <= it;
     * and every magnitude partial sum is a sum of at most deg+1 terms each
     * <= M, hence <= (deg+1)*M, which is the bound itself.
     *
     * THE ONE GAP, stated because it is real: |b| == 0 breaks that chain,
     * since the multiply by zero comes last and the intermediate |c_k|*|a|^k
     * is formed first. It is a single known lattice point, (i,j) == (0,1)
     * with a == q (see k_classify's COF_DEGENERATE comment), so at most one
     * survivor per q; its intermediates are smaller than the |c_deg|*q^deg
     * term the bound already covers by a factor of q per degree; and
     * k_classify discards it on B[t] == 0 without reading the norm. Closing
     * the gap formally would mean skipping the norm block for b == 0, which
     * is not worth leaving `cof` undefined for a downstream reader.
     *
     * So if this fires, the bound is wrong -- and a wrong norm bound is
     * exactly what BN_LIMBS and normscan exist to catch. Skipping the slab
     * would hide it. */
    if (hflags & TDF_NORM_OVERFLOW) {
        runlog_warn("  ** NORM OVERFLOW: a norm exceeded %d bits during trial"
                    " division, but pipe_side_prepare_q proved this q's bound"
                    " fits. That is a BROKEN INVARIANT, not a job that needs"
                    " tuning: the norm width bound is wrong, or this q reached"
                    " k_td without being screened. Run ./normscan over the"
                    " band and report the (q,rho).", BN_LIMBS * 32);
        return -1;
    }
    if (hflags & TDF_LIST_TRUNCATED) {
        if (g_warned_td_trunc++ < 4)      /* file scope: see bucket overflow */
            runlog_warn("  ** %llu large-prime records past the %u/survivor"
                        " cap -- slab skipped", hovf, PIPE_K);
        return 1;
    }
    nacc = C->h_rb->ctr.nacc;
    if (nacc > C->ccap) {           /* the scatter clamped; redo it once, larger */
        if (pipe_td_grow_cand(C, nacc)) return -1;
        MTL_LAUNCH(k_scatter_sel, blocks, threads, 0, 0, C->d_aflag, C->d_aoff, n, C->d_sel, C->ccap, C->d_nacc);
        CK(pipe_qsync()); CK(mtlGetLastError());
    }
    *nacc_out = nacc;

    /* ---- the first q's reconstruction gate, on its own clock ---- */
    if (verify) {
        double v0 = host_ms();
        printf("\n  --- first-q validation (excluded from the band timing) ---\n");
        /* The special-q goes to whichever side carries it, matching the k_td
         * calls that produced d_cof[]. Hardcoding side 1 here survived
         * --sq-side 0 in practice -- td_divide_out is a no-op when q does not
         * divide, and the gate still passed 15,174 of 15,174 -- but the verify
         * pass and the pass it is checking must be given the same q on the
         * same side, or the agreement is luck rather than evidence. */
        if (pipe_td_verify<SLABBED>(C, 1, n, cfg->logI,
                           cfg->sq_side == 1 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads, j_base,
                           gate_found ? &gate_found[1] : NULL) ||
            pipe_td_verify<SLABBED>(C, 0, n, cfg->logI,
                           cfg->sq_side == 0 ? (uint32_t)L->q : 0u,
                           cfg, blocks, threads, j_base,
                           gate_found ? &gate_found[0] : NULL)) return -1;
        {
            const double vms = host_ms() - v0;
            *t_verify += vms;
            printf("  --- validation took %.1f ms ---\n\n", vms);
        }
    }

    /* ---- phase 3: record the joint candidates only ---- */
    if (!nacc) return 0;
    CK(mtlEventRecord(E[12]));
    MTL_LAUNCH(k_gather_ab, blocks, threads, 0, 0, C->d_a, C->d_b, C->d_sel, nacc, C->d_ca, C->d_cb);
    /* One warp per candidate, and a grid sized to the candidate count -- which
     * is what makes the grid sizing worth doing here and not in `k_td`. There
     * the per-thread cost is the whole `nsm` march and more threads cannot
     * shorten it; here more warps do, right up to one warp per candidate.
     * Beyond that point extra blocks only re-stage the same shared tile out of
     * L2, so the grid is capped at the candidates present. */
    static_assert(TD_RECORD_THREADS >= 32 && TD_RECORD_THREADS <= 1024 &&
                  (TD_RECORD_THREADS % 32) == 0,
                  "TD_RECORD_THREADS must be a whole number of warps: "
                  "k_td_record_warp maps one warp to one candidate and its "
                  "__ballot_sync assumes a full, converged warp.");
    const int rthreads = TD_RECORD_THREADS;
    const int rblocks  = (int)((nacc + (uint32_t)(rthreads / 32) - 1u) /
                               (uint32_t)(rthreads / 32));
    for (int si = 0; si < 2; si++) {
        const int side = si ? 0 : 1;
        if (cfg->td_record_scalar)
            MTL_LAUNCH_NAMED((SLABBED ? "k_td_1_1_1_1" : "k_td_1_1_1_0"), blocks, threads, 0, 0, C->d_a, C->d_b, C->d_x, C->d_sel, nacc, cfg->logI, C->d_poly[side], side == cfg->sq_side ? (uint32_t)L->q : 0u, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_sm[side], C->nsm[side], C->d_ccof[side], C->d_cbits[side], C->d_flags, nullptr, C->d_cfac[side], C->d_cfn[side], TD_FMAX, j_base);
        else
            MTL_LAUNCH_NAMED((SLABBED ? "k_td_record_warp_1" : "k_td_record_warp_0"), rblocks, rthreads, 0, 0, C->d_a, C->d_b, C->d_x, C->d_sel, nacc, cfg->logI, C->d_poly[side], side == cfg->sq_side ? (uint32_t)L->q : 0u, C->d_plist[side], C->d_pcnt[side], PIPE_K, C->d_sm[side], C->nsm[side], C->d_ccof[side], C->d_cbits[side], C->d_flags, C->d_cfac[side], C->d_cfn[side], TD_FMAX, j_base);
    }
    CK(mtlEventRecord(E[13]));
    CK(mtlEventSynchronize(E[13])); CK(mtlGetLastError());
    tm->record += time_kernel(E[12], E[13]);

    /* Counts the host loop used to produce, computed where the data already is.
     * They ACCUMULATE across the band and are read at flush points, because a
     * blocking 12-byte readback per q costs about as much as the ~618 bytes per
     * candidate this change removes -- a device round trip per q is the thing
     * the rest of this pipeline is built to avoid. */
    if (accumulate_stats) {
        MTL_LAUNCH(k_cand_stats, blocks, threads, 0, 0, nacc, C->d_cbits[0], C->d_cbits[1], C->d_cfn[0], C->d_cfn[1], cfg->lpb0, cfg->lpb, TD_FMAX, C->d_stats);
        CK(mtlGetLastError());
    }

    const double rb0 = host_ms();
    /* `want_host` is false under inline cofactorisation with no candidate file:
     * the queue has already taken everything it needs straight from device
     * memory, so the ~618 bytes per candidate this used to move existed only to
     * be counted. k_cand_stats counts them in place. */
    if (!want_host) { tm->readback += host_ms() - rb0; return 0; }
    CK(mtlMemcpy(C->h_ca, C->d_ca, (size_t)nacc * 8, mtlMemcpyDeviceToHost));
    CK(mtlMemcpy(C->h_cb, C->d_cb, (size_t)nacc * 8, mtlMemcpyDeviceToHost));
    for (int s = 0; s < 2; s++) {
        CK(mtlMemcpy(C->h_ccof[s], C->d_ccof[s], (size_t)nacc * sizeof(bn_t),
                      mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(C->h_cbits[s], C->d_cbits[s], (size_t)nacc,
                      mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(C->h_cfn[s], C->d_cfn[s], (size_t)nacc * 4,
                      mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(C->h_cfac[s], C->d_cfac[s], (size_t)nacc * TD_FMAX * 4,
                      mtlMemcpyDeviceToHost));
    }
    tm->readback += host_ms() - rb0;
    return 0;
}

/* Memory reports are telemetry, not part of the scientific result. A failed
 * query must never discard a completed band or bypass the final cofactor queue
 * flush. Admission-control queries remain fatal at their call site. */
/* ---- --cof-async: the cofactor queue flushed off the q loop -------------- *
 *
 * glas overlaps its device ECM with the next special-q's sieve; cuda-sieve's
 * queue flush used to stall the q loop for the whole of it (57 ms/q of 315 on
 * the 2,1147+ job, RTX 4090). Under --cof-async the queue is double-buffered:
 * when the filling set is full it is handed, whole, to a host thread that runs
 * the unchanged cofq_flush on g_cof_stream, and the q loop carries on into the
 * spare set. Flushes stay strictly serialised -- the next one waits for this
 * one -- so relation order, and so the output file, is exactly what the
 * synchronous path writes.
 *
 * Only the per-batch DEVICE BUFFERS move between the two sets. The schedule
 * tables, the status histogram d_nst and every host statistic and chunk
 * steering field belong to the canonical queue Q: the flush works on a copy
 * (F) and cofa_wait copies them back, which is safe because nothing else
 * writes them while a flush is in flight. */
static void cofq_swap_buffers(cofq_t *a, cofq_t *b)
{
#define CQ_SW(f) std::swap(a->f, b->f)
    CQ_SW(n);
    CQ_SW(d_c0); CQ_SW(d_c1); CQ_SW(d_st0); CQ_SW(d_st1);
    CQ_SW(d_sm0); CQ_SW(d_sm1); CQ_SW(d_a); CQ_SW(d_b);
    CQ_SW(d_f0); CQ_SW(d_f1); CQ_SW(d_fn0); CQ_SW(d_fn1);
    CQ_SW(d_sp0); CQ_SW(d_sp1); CQ_SW(d_nsp0); CQ_SW(d_nsp1);
    CQ_SW(d_flag); CQ_SW(d_off); CQ_SW(d_bsum); CQ_SW(d_idx); CQ_SW(d_nrel);
    CQ_SW(d_sel); CQ_SW(d_nsel); CQ_SW(d_ovf);
#undef CQ_SW
}

typedef struct {
    cofq_t      spare;      /* the second buffer set; only its buffers are used */
    cofq_out_t  spare_out;  /* allocated by cofq_init alongside it, unused      */
    cofq_t      F;          /* the queue as handed to the in-flight flush       */
    std::thread *th;
    std::atomic<int> *fin;  /* set by the flush thread as its last act         */
    int         busy, rc;
    double      ms_wait;    /* host time the q loop spent waiting on a flush    */
    /* Each async flush's own wall time, and the spacing of their starts: a
     * flush that takes longer than the interval is what ms_wait pays for. */
    double      t_start, t_last, ms_flush, ms_interval;
    int         nflush;
    /* The checkpoint the flush writes when it is done (ck_pending), from the
     * flush thread itself: its relations are then in the file, and the q loop
     * never waits on the fsync. Measured on a network filesystem (CephFS) at
     * ~0.6 s a flush -- 16-19 ms/q, a quarter of c187's wall -- against
     * nothing on local disk; a volunteer's slow disk pays it the same way.
     * Every other checkpoint is written by the q loop only after cofa_wait,
     * so ck and its flags are never touched by both threads at once. */
    const poly_t *ck_P;
    ckpt_t      *ck;
    FILE        *ck_fc;
    unsigned long long ck_base_rel;
    int         *ck_written, *ck_warned;
    /* The resume point as it stood when the in-flight flush started: the file
     * holds exactly that flush's relations once it is joined, so that is the
     * checkpoint to write then -- not the q the loop has since reached. */
    int         ck_pending;
    qsel_t      ck_next;
    unsigned long long ck_nq;
} cofasync_t;

static int cofa_wait(cofasync_t *A, cofq_t *Q)
{
    if (!A->busy) return 0;
    const double t0 = host_ms();
    A->th->join();
    delete A->th;
    A->th = NULL;
    A->busy = 0;
    A->ms_wait += host_ms() - t0;
    Q->ms_rat = A->F.ms_rat; Q->ms_alg = A->F.ms_alg; Q->ms_host = A->F.ms_host;
    Q->nseen = A->F.nseen; Q->nrel = A->F.nrel;
    memcpy(Q->nst, A->F.nst, sizeof Q->nst);
    Q->chunk_cur = A->F.chunk_cur; Q->chunk_ceiling = A->F.chunk_ceiling;
    Q->eff_prev_valve = A->F.eff_prev_valve;
    Q->ms_prev_valve = A->F.ms_prev_valve;
    Q->valve_parked = A->F.valve_parked;
    A->spare.n = 0;
    return A->rc;
}

static void cofa_start(cofasync_t *A, cofq_t *Q, cofq_out_t *O,
                       const bench_cfg_t *cfg, int blocks, FILE *fr)
{
    {   /* Metal only: see metal/gen_pipeline_host.py. The q loop's stream
         * and g_cof_stream are separate command queues with no ordering
         * between them, and the enqueue below has just written the buffers
         * this flush is about to read. */
        const mtlError_t _e = mtlDeviceSynchronize();
        if (_e != mtlSuccess)
            fprintf(stderr, "  pipeline: the cofactor handover barrier failed:"
                    " %s; the flush will report it\n", mtlGetErrorString(_e));
    }
    A->F = *Q;                        /* full buffers, stats, steering   */
    cofq_swap_buffers(Q, &A->spare);  /* Q now fills the spare set, n=0  */
    A->busy = 1;
    A->rc = 0;
    A->fin->store(0, std::memory_order_relaxed);
    A->t_start = host_ms();
    if (A->nflush) A->ms_interval += A->t_start - A->t_last;
    A->t_last = A->t_start;
    A->nflush++;
    /* Full width. Launching the flush narrower so the sieve kept SMs to
     * itself was measured twice: blocks/2, /4 at 278.2, 277.0 ms/q against
     * 276.5 (2,1147+, RTX 4090), and 256/128/64 blocks against 768 at
     * 196.8/195.6/203.5 vs 192.5 (and c187 68.6/70.5/69.8 vs 66.9). No
     * effect, so there is no knob. */
    const mtlStream_t cs = g_cof_stream;   /* thread_local: hand it over */
    /* The CURRENT DEVICE is per host thread too, and a new thread starts on
     * device 0. On a multi-GPU host whose task was given another card, the
     * flush's events were created on device 0 and recorded on this device's
     * stream: "invalid resource handle" at the first flush (a Tesla T4 as
     * device 1 of 4, every work unit). Hand the device over with the stream. */
    int dev = 0;
    if (mtlGetDevice(&dev) != mtlSuccess) dev = -1;
    const int ecm_rounds = (int)Q->ecm_rounds;
    A->th = new std::thread([A, O, cfg, blocks, fr, cs, dev, ecm_rounds]() {
        g_cof_stream = cs;
        wd_bind(-1);    /* the sieve thread's slot stays the sieve thread's */
        if (dev < 0 || mtlSetDevice(dev) != mtlSuccess) {
            fprintf(stderr, "  cofactor flush thread: cannot select Metal"
                    " device %d\n", dev);
            A->rc = -1;
            A->fin->store(1, std::memory_order_release);
            return;
        }
        A->rc = cofq_flush(&A->F, O, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                           ecm_rounds, cfg->cof_budget, cfg->cof_chunk,
                           blocks, cfg->threads, fr);
        if (A->rc == 0 && A->ck_pending) {
            pipe_try_checkpoint(A->ck_P, cfg, A->ck, fr, A->ck_fc, &A->ck_next,
                                A->ck_base_rel + A->F.nrel, A->ck_nq,
                                A->ck_written, A->ck_warned);
            A->ck_pending = 0;
        }
        A->ms_flush += host_ms() - A->t_start;
        A->fin->store(1, std::memory_order_release);
    });
}

static int pipe_mem_info_optional(const char *where,
                                  size_t *free_bytes, size_t *total_bytes)
{
    const mtlError_t err = mtlMemGetInfo(free_bytes, total_bytes);
    if (err == mtlSuccess) return 0;
    fprintf(stderr, "warning: Metal memory diagnostic '%s' unavailable: %s\n",
            where, mtlGetErrorString(err));
    return -1;
}

/* Close the two staged outputs exactly once. `commit` is true only after the
 * whole band and final cofactor flush succeed. Any close or rename failure
 * converts the operation to discard mode and removes whatever temporary files
 * remain. */
/* commit:       claim the final names. A clean stop is NOT a completed band,
 *               so the caller passes 0 and the .part keeps its staging name.
 * keep_partial: a checkpoint sidecar was actually written, so what is on disk
 *               is resumable work rather than debris. Only ever consulted on
 *               the discard path. */
static int pipe_finalize_outputs(FILE **frp, FILE **fcp,
                                 const bench_cfg_t *cfg,
                                 const char *rtmp, const char *ctmp,
                                 int commit, int keep_partial)
{
    int bad = 0;

    if (frp && *frp) {
        const int io_bad = ferror(*frp);
        if (fclose(*frp) || io_bad) bad = 1;
        *frp = NULL;
    }
    if (fcp && *fcp) {
        const int io_bad = ferror(*fcp);
        if (fclose(*fcp) || io_bad) bad = 1;
        *fcp = NULL;
    }

    if (commit && !bad) {
        int relations_claimed = 0;
        if (cfg->relations) {
            if (bench_atomic_replace(rtmp, cfg->relations)) { perror("rename relations"); bad = 1; }
            else relations_claimed = 1;
        }
        if (!bad && cfg->candidates && bench_atomic_replace(ctmp, cfg->candidates)) {
            perror("rename candidates");
            bad = 1;
            /* Put the relations file back under its staging name. Claiming one
             * output and then reporting the band as failed leaves a finished
             * result on disk that the exit status says was discarded, and the
             * discard below cannot reach it -- rtmp no longer exists. Under
             * BOINC that is a compute error raised for a workunit whose result
             * file is present and complete. */
            if (relations_claimed && bench_atomic_replace(cfg->relations, rtmp))
                perror("rename relations back to staging");
        }
    }
    if (!commit || bad) {
        /* A .part is resumable work only if a sidecar was actually written --
         * NOT merely because a relation file was opened. A fresh --cofactor
         * run seeds an initial whole-q checkpoint before q0, specifically so a
         * later mid-q slab flush may leave an uncheckpointed tail without
         * making the .part disposable. Resume truncates that tail to the safe
         * sidecar prefix. If even the initial checkpoint could not be written,
         * keeping the .part would strand every rerun on the startup refusal. */
        if (!keep_partial) {
            if (rtmp && rtmp[0]) remove(rtmp);
            if (ctmp && ctmp[0]) remove(ctmp);
        }
    }
    return bad ? -1 : 0;
}

/* ---- the whole per-q pipeline ------------------------------------------ */

/* The overlap credit --fill-concurrent earns, clamped once and read through
 * here by BOTH consumers. A negative band total would mean the pair took longer
 * than the sum of its parts, which is launch noise and is not credited. The
 * running `acc/wall=` runlog record and the band summary's
 * `GPU-accounted / wall` are the running and final forms of one ratio (see the
 * note at the runlog site); spelling the clamp out at each of them let the two
 * drift the moment the policy changed. */
static inline double pipe_ovl_credit(double acc_ovl)
{
    return acc_ovl > 0 ? acc_ovl : 0;
}

/* Both sides' transform time for the q whose events these are. The explicit
 * ev[4] synchronises are load-bearing (see the read after the slab loop):
 * time_kernel discards mtlEventElapsedTime's status, so a not-ready event
 * would read 0 ms and latch mtlErrorNotReady for the next unrelated check. */
static int pipe_read_transform(const pside_t *S1, const pside_t *S0,
                               float *t1, float *t0)
{
    if (CUDA_CHECKED(mtlEventSynchronize(S1->ev[4])) ||
        CUDA_CHECKED(mtlEventSynchronize(S0->ev[4])))
        return -1;
    *t1 = time_kernel(S1->ev[0], S1->ev[4]);
    *t0 = time_kernel(S0->ev[0], S0->ev[4]);
    return 0;
}

template <bool SLABBED>
static int run_pipeline_impl(const fb_t *fb1, const fb_t *fbs1,
                             const fb_t *fb0, const fb_t *fbs0,
                             const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                             const poly_t *POLY, const bench_cfg_t *cfg,
                             const slab_plan_t *slab_plan)
{
    /* First, ahead of every goto: see pipe_print_guard_t, and cofasync_t's
     * `fin`, which the memset of A below cannot hold by value. */
    pipe_print_guard_t pg;
    std::atomic<int> cofa_fin(0);
    pg.lock();
    if (g_pinst)
        printf("\n=== instance %d of %d: q [%llu, %llu] ===\n", g_pinst->k,
               g_pinst->n, (unsigned long long)g_pinst->qmin,
               (unsigned long long)g_pinst->qmax);
    const uint32_t I = 1u << cfg->logI;
    const uint32_t xmax_alloc = I * slab_plan->jmax;
    const uint32_t nregion_alloc = xmax_alloc >> cfg->log_region;
    const uint32_t nbitword_alloc = xmax_alloc >> 5;
    const int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    /* Fill's grid is absolute, not per-SM -- see FILL_BLOCKS_DEFAULT. */
    const int fblocks = cfg->fill_blocks ? cfg->fill_blocks : FILL_BLOCKS_DEFAULT;
    const int fthreads = cfg->fill_threads ? cfg->fill_threads : FILL_THREADS_DEFAULT;

    pside_t S1, S0;
    pipe_td_t C;
    pipe_tm_t tm;
    cofq_t Q; cofq_out_t QO;
    cofasync_t A;
    uint8_t *d_bucket = NULL;
    double vram_prev = 0;
    size_t need = 0;
    uint32_t *d_cursor = NULL, *d_overflow = NULL, *d_two = NULL;
    /* --fill-gate with --fill-gate-shift > 0: side 1's folded bitmap */
    uint32_t *d_gate = NULL;
    /* --q-pipeline as GRANTED, not as requested: the stream the next q is
     * sieved on. Both sides share the bucket array's overflow counter, as in
     * the serial arm: side 1's count is copied to the host in stream order,
     * behind its apply, before side 0's issue clears the counter (finding
     * 114 retired the separate side-0 counter this needed when the join
     * read the counter after both sides had run). */
    mtlStream_t st_sv = 0;
    int qpipe = 0;
    /* Slabs and q actually issued ahead, for the band summary. */
    unsigned long long n_ahead_slab = 0, n_ahead_q = 0;
    /* Second bucket workspace, allocated only under --fill-concurrent. The
     * two sides share one today because they run back to back; overlapping
     * them is exactly what that sharing forbids. */
    uint8_t  *d_bucket_c = NULL;
    uint32_t *d_cursor_c = NULL, *d_overflow_c = NULL;
    mtlStream_t st_side[2] = { 0, 0 };
    int concurrent = 0;         /* the option as GRANTED, not as requested */
    int dual = 0;               /* serial arm with a second bucket array      */
    pbkt_t bk1, bk0;
    uint32_t *d_n = NULL;
    unsigned long long *d_pre = NULL;
    uint64_t est1, est0, est;
    uint32_t cap, nqdone = 0, bound1 = 0, bound0 = 0;
    /* Slabs abandoned to a soft failure (bucket overflow, or trial division
     * this slab cannot be trusted). Counted rather than fatal: see
     * pipe_side_sieve_join. Reported at end of band so the rate-limited
     * per-slab warnings do not have to carry the total. */
    unsigned long long nslab_skipped = 0;
    /* Of nslab_skipped, those dropped for untrustworthy trial division
     * rather than a bucket overflow. */
    unsigned long long ntd_skipped = 0;
    /* q abandoned because every slab of it was skipped. */
    unsigned long long nq_lost = 0;
    unsigned long long nqskip = 0;   /* special-q passed over on norm width */
    size_t optin_smem_limit = 0;
    double acc_isect = 0, acc_host = 0, acc_wall = 0;
    /* The three sieve stages, broken out because the band total alone cannot be
     * compared against the standalone bench (no --pipeline), which reports the
     * same three. When the two disagreed on the A100 only the split showed that
     * the whole discrepancy was transform. "sieve, both sides" is their sum --
     * derived, not accumulated separately, so the printed total and its three
     * children cannot drift apart. */
    double acc_tr = 0, acc_fi = 0, acc_ap = 0;
    /* --fill-concurrent only: fill and apply are still measured PER SIDE, and
     * under concurrency those two spans OVERLAP, so their sum exceeds the wall
     * time the pair actually took. Left uncorrected the band report showed
     * `sieve, both sides` at 95 ms inside a 79 ms q and `unaccounted` at
     * -41 ms. Accumulate what the overlap was worth and subtract it from the
     * stage total, so the printed sieve figure stays a wall quantity and the
     * per-side children still add up visibly to it plus this. Zero when the
     * flag is off, so the serial report is unchanged to the last digit. */
    double acc_ovl = 0;
    /* --qspan (item 19 step 2): the GPU-timeline span of one special-q, from
     * before its first GPU operation to after its last. `unaccounted` is
     * wall minus the bracketed stages and has twice been mis-read as "GPU
     * idle"; this splits it for real. wall - qspan is host time with no GPU
     * work in flight; qspan - (sum of stage device times) is idle BETWEEN
     * those stages. Off by default: two mtlEventRecords per q behind a
     * branch, so the cost when off is nothing and the shipping binary can
     * produce the number without a rebuild. */
    double acc_qspan = 0, acc_hspan = 0;   /* host clock over the SAME bracket */
    /* Host time from the top of the per-q body to the first call that issues
     * GPU work, i.e. the prologue item 4.1 proposes to hide in the previous
     * q's shadow: q selection, root validation, checkpointing and the
     * q-lattice build. Everything after this point already has its own
     * counter (`host per-q`, `host: small-prime tables`). */
    double acc_prologue = 0;
    /* Wall time of each region of the per-q body, to attribute the ~2 ms that
     * is neither the prologue nor any existing host counter. Compare each
     * against the device work it contains: sieve+TD+cofactor for the slab
     * loop, nothing for prep/tdprep/tail. */
    double acc_prep = 0, acc_tdprep = 0, acc_slab = 0, acc_tail = 0;
    mtlEvent_t qspan0 = NULL, qspan1 = NULL;
    double acc_td = 0, t_verify = 0, cofac_tail = 0;
    unsigned long long acc_surv = 0, acc_cand = 0, acc_rel = 0;
    FILE *fr = NULL, *fc = NULL;
    char rtmp[CKPT_PATH_MAX] = "", ctmp[CKPT_PATH_MAX] = "";
    int rc = 0, source_exhausted = 0, count_limit_reached = 0;
    int outputs_finalized = 0, vram_reporting = 1;
    int target_reached = 0;
    qsel_t last_q = {0, 0};
    /* Initialised: the done: block destroys whatever is non-NULL, and a CUDA
     * failure can jump there from any stage before these are created. */
    mtlEvent_t ea = NULL, eb = NULL;
    /* Resume state. base_rel/base_nq are what earlier sessions already put on
     * disk: they are added to the goal tests and the progress line, and kept
     * OUT of the band summary, whose per-q figures describe this session. */
    ckpt_t ck;
    const unsigned long long base_rel = cfg->resume ? cfg->resume_nrel : 0;
    const unsigned long long base_nq  = cfg->resume ? cfg->resume_nq  : 0;
    /* ckpt_armed: a relation file is open, so checkpointing and the signal
     * handlers are active. ckpt_written: a sidecar has actually been written,
     * which is the ONLY thing that makes the .part resumable. Conflating the
     * two keeps a .part that no rerun can consume and that the startup check
     * then refuses, wedging an unattended queue. */
    /* stopped: a clean stop of either kind, which is what suppresses the
     * commit. capped: that stop was PIPE_SKIP_MAX, which is a different thing
     * to tell a job queue -- rerunning the same binary cannot help. */
    int stopped = 0, capped = 0, ckpt_armed = 0, ckpt_written = 0, ckpt_warned = 0;
    int degraded = 0;               /* PIPE_LOST_MAX/PIPE_SLAB_SKIP_MAX tripped */
    int stop_hooked = 0;

    /* A file static, so a second band in one process would inherit the first
     * one's answer. Nothing does that today; the reset costs nothing and the
     * bug it prevents is a wrong width in a failure report. */
    g_pipe_need_limbs = 0;

    /* This must precede pipe_est_records(): that helper divides by every
     * modulus, so a zero modulus in an unvalidated object is already too late
     * even though the first device upload happens in pipe_side_init(). */
    if (!fb_is_transform_validated(fb1) ||
        !fb_is_transform_validated(fb0) ||
        (fbs1 && fbs1->n && !fb_is_transform_validated(fbs1)) ||
        (fbs0 && fbs0->n && !fb_is_transform_validated(fbs0))) {
        fprintf(stderr,
                "run_pipeline: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or estimating it\n");
        return -1;
    }
    /* Check both thresholds before pipe_est_records(), CUDA event creation,
     * or the multi-gigabyte bucket allocation. This is also the consumer-side
     * gate for callers that bypass bench_main's CLI validation. */
    if (sieve_bound_checked(cfg->scale, cfg->allowance, 4096u, &bound1,
                            "run_pipeline side 1 survivor parameters") ||
        sieve_bound_checked(cfg->scale0, cfg->allowance0, 4096u, &bound0,
                            "run_pipeline side 0 survivor parameters"))
        return -1;
    if (!pipe_fb_bounds_ok(fb1, I, UINT32_MAX, "side 1 bucketed") ||
        !pipe_fb_bounds_ok(fb0, I, UINT32_MAX, "side 0 bucketed") ||
        !pipe_fb_bounds_ok(fbs1, 1u, 1u << 30, "side 1 line-sieved") ||
        !pipe_fb_bounds_ok(fbs0, 1u, 1u << 30, "side 0 line-sieved"))
        return -1;

    memset(&S1, 0, sizeof S1); memset(&S0, 0, sizeof S0);
    memset(&C, 0, sizeof C); memset(&tm, 0, sizeof tm);
    memset(&Q, 0, sizeof Q); memset(&QO, 0, sizeof QO);
    memset(&A, 0, sizeof A);
    A.fin = &cofa_fin;
#define PIPE_CK(x) do { if (CUDA_CHECKED(x)) { rc = -1; goto done; } } while (0)
/* Headroom left for the per-q buffers that grow on demand after all
 * one-time setup -- measured at 0.12 GB at 15e (finding 94). ONE constant:
 * both bucket arrays are checked against it, and a margin that applied to
 * only one of them would reinstate the bug the second check exists to fix. */
#define PIPE_VRAM_MARGIN (512u * 1024u * 1024u)
    /* --qspan's events are created HERE, not with their accumulators above,
     * because both validation gates return -1 without passing through `done:`
     * and would leak them. The comment at those gates states the rule: nothing
     * creates CUDA state before they run. Creating them after PIPE_CK exists
     * also means a failure unwinds through `done:` like every other resource,
     * and mtlEventCreate is no longer the first CUDA call in the process --
     * so a refused factor base prints its refusal without paying for context
     * initialisation first. */
    if (cfg->qspan) {
        PIPE_CK(mtlEventCreate(&qspan0));
        PIPE_CK(mtlEventCreate(&qspan1));
    }
    PIPE_CK(mtlEventCreate(&ea));
    PIPE_CK(mtlEventCreate(&eb));

    if (qgen)
        printf("\n=== pipeline: both sides in one process, streaming special-q ===\n");
    else
        printf("\n=== pipeline: both sides in one process, %u special-q ===\n", nq);

    est1 = pipe_est_records(fb1, xmax_alloc);
    est0 = pipe_est_records(fb0, xmax_alloc);
    est = est1 > est0 ? est1 : est0;
    cap = (uint32_t)(est / nregion_alloc) + 256;
    /* TEST-ONLY (PIPE_CAP_PCT, 100 everywhere else): cut the bucket capacity
     * so that some slabs overflow. No real job reaches the bucket-overflow
     * soft skip (degradecheck.sh), and finding 114's review needed it driven
     * with the next q already prepared. Set through DEFS, so a build that
     * changes it is marked a pricing build. Region loads are tight, so the
     * band where SOME slabs overflow is narrow: such a build also reads a
     * finer setting, in parts per thousand, from BENCH_TEST_CAP_PERMILLE. */
    if (PIPE_CAP_PCT != 100) {
        unsigned long pm = (unsigned long)PIPE_CAP_PCT * 10;
        const char *e = getenv("BENCH_TEST_CAP_PERMILLE");
        if (e && *e) pm = strtoul(e, NULL, 10);
        cap = (uint32_t)((uint64_t)cap * pm / 1000);
        printf("  TEST BUILD: bucket capacity cut to %lu per mille of the"
               " estimate's (%u records per region)\n", pm, cap);
    }
    {
        size_t freeB = 0, totalB = 0;
        need = (size_t)nregion_alloc * cap * 4;
        PIPE_CK(mtlMemGetInfo(&freeB, &totalB));
        printf("  bucket array %u x %u x 4 B = %.2f GB, shared by both sides"
               " (%.2f GB free)\n", nregion_alloc, cap, need / 1073741824.0,
               freeB / 1073741824.0);
        /* PIPE_RC_NOVRAM, not -1: when run_pipeline sized the slab up from
         * free VRAM, another process may have taken that memory since (two
         * tasks starting on one GPU), and the band can still run at the
         * default slab. Nothing is opened or consumed before this point. */
        if (need + PIPE_VRAM_MARGIN > freeB) {
            fprintf(stderr, "  pipeline: bucket array does not fit\n");
            rc = PIPE_RC_NOVRAM;
            goto done;
        }
        {
            const mtlError_t me = mtlMalloc(&d_bucket, need);
            if (me == mtlErrorMemoryAllocation) {
                (void)mtlGetLastError();        /* not sticky; clear it */
                fprintf(stderr, "  pipeline: bucket array allocation of"
                        " %.2f GB failed\n", need / 1073741824.0);
                rc = PIPE_RC_NOVRAM;
                goto done;
            }
            PIPE_CK(me);
        }
    }
    /* The OTHER capacity a job can undersize, reported beside the bucket array
     * because the two are the pair the per-slab soft skips fire on and the pair
     * the degraded stop's remedy message has to choose between. An operator
     * reading "mostly trial-division skips: check mfb and PIPE_K" needs to know
     * what PIPE_K this binary actually carries, and it is a build-time knob, so
     * nothing else in the run says. degradecheck.sh reads it from here to tell
     * "this build cannot trigger the ceiling" apart from "the ceiling broke". */
    printf("  large-prime list %d records per survivor (PIPE_K)\n", PIPE_K);
    PIPE_CK(mtlMalloc(&d_cursor, (size_t)nregion_alloc * 4));
    PIPE_CK(mtlMalloc(&d_overflow, 4));

    /* ---- one-time setup, hoisted out of the q loop ---- */
    /* Device memory, by stage. The bucket array dominates on a big-area job
     * like c183 and is a minority of the total on a small-area one like the
     * c151, so "the bucket array is the footprint" is only true at one end.
     * Print the actual split rather than inviting anyone to model it. */
#define VRAM_MARK(label)                                                        \
    do {                                                                        \
        if (vram_reporting) {                                                   \
            size_t fb_ = 0, tb_ = 0;                                            \
            if (pipe_mem_info_optional((label), &fb_, &tb_) == 0) {            \
                printf("    %-28s %6.2f GB   (%.2f GB free)\n", label,          \
                       (vram_prev - (double)fb_) / 1073741824.0,                \
                       fb_ / 1073741824.0);                                     \
                vram_prev = (double)fb_;                                        \
            } else {                                                            \
                vram_reporting = 0;                                             \
            }                                                                   \
        }                                                                       \
    } while (0)
    {
        size_t fb_ = 0, tb_ = 0;
        if (pipe_mem_info_optional("stage-report baseline", &fb_, &tb_) == 0) {
            vram_prev = (double)fb_;
            printf("  device memory by stage:\n");
            vram_prev += (double)need; /* bucket is already allocated */
            VRAM_MARK("bucket array");
        } else {
            vram_reporting = 0;
        }
    }
    if (mtl_optin_smem_limit(&optin_smem_limit))
        { rc = -1; goto done; }
    if (pipe_side_init<SLABBED>(fb1, fbs1, cfg, bound1, xmax_alloc,
                                optin_smem_limit, &S1) ||
        pipe_side_init<SLABBED>(fb0, fbs0, cfg, bound0, xmax_alloc,
                                optin_smem_limit, &S0))
        { rc = -1; goto done; }
    {
        const size_t apply_smem =
            S1.apply_smem > S0.apply_smem ? S1.apply_smem : S0.apply_smem;
        /* MaxDynamicSharedMemorySize belongs to each k_apply specialization.
         * Setting it independently while initializing the two sides is wrong:
         * the second side can lower the limit required by the first. Both
         * SSTAIL instances, because the launch picks one per q. */
        PIPE_CK(mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1_1" : "k_apply_16_1_1_0_1", apply_smem));
        PIPE_CK(mtlFuncSetMaxThreadgroupMemory(SLABBED ? "k_apply_16_1_1_1_1" : "k_apply_16_1_1_0_1", apply_smem));
    }
    VRAM_MARK("factor bases + bitmaps");
    PIPE_CK(mtlMalloc(&d_two, (size_t)nbitword_alloc * 4));
    if (cfg->fill_gate && cfg->fill_gate_shift)
        PIPE_CK(mtlMalloc(&d_gate, (size_t)((nbitword_alloc +
                           (1u << cfg->fill_gate_shift) - 1) >>
                           cfg->fill_gate_shift) * 4));
    PIPE_CK(mtlMalloc(&d_n, 4));
    PIPE_CK(mtlMalloc(&d_pre, 8));
    /* Untimed warm-up, n = 0 so it touches nothing. Everything above this line
     * is mtlMalloc/mtlMemcpy, so without it k_transform is the first kernel
     * launched and the one-time CUDA cost -- module load, and on a card with
     * no native cubin the PTX JIT -- lands inside the FIRST q's transform
     * window and is then divided by the band length. At 1340 q that is ~0.15 ms
     * on a 0.954 ms figure; on a 50-q band it is ~4 ms on the same figure.
     * run_bench got this fix (finding 48) and this path, which the RUNBOOK
     * points at as the honest source for transform, did not.
     *
     * Lazy module loading is per-kernel, so this warms k_transform only; fill
     * and apply still pay their own first-launch cost inside q0. They are the
     * reps-stable stages and the cost is one-time either way, but a band short
     * enough to care should be treated as a warm-up run, not a measurement. */
    MTL_LAUNCH_NAMED((SLABBED ? "k_transform_1" : "k_transform_0"), blocks, cfg->threads, 0, 0, S1.primes, S1.roots, S1.plat, 0u, cfg->logI, cfg->J, (int64_t)1, (int64_t)0, (int64_t)0, (int64_t)1, S1.d_nproj, S1.d_nlost, S1.walk_cur, nullptr, 0);
    PIPE_CK(mtlDeviceSynchronize());
    PIPE_CK(mtlGetLastError());
    if (pipe_td_init(&C, fbs1, fbs0, POLY, nbitword_alloc)) { rc = -1; goto done; }
    VRAM_MARK("trial division context");
    /* The cross-q cofactor queue. Per-q the candidate count is ~1,956, which
     * would run the rho kernel at 3% occupancy; the queue accumulates across
     * special-q and flushes ~67 q worth at a time. With it resident, only the
     * ~2% of records that become relations are read back, and the 295 MB
     * candidate file and its ~3 ms/q of host emission are gone. */
    if (cfg->cofactor && cofq_init(&Q, &QO, CQ_FLUSH,
                                   cfg->cof_meth0, cfg->cof_meth1,
                                   cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,
                                   cfg->cof_limbs0, cfg->cof_limbs,
                                   cfg->ecm_curves_set,
                                   (uint32_t)cfg->cof_rounds))
        { rc = -1; goto done; }
    if (cfg->cofactor) VRAM_MARK("cofactor queue");
    if (cfg->cofactor && cfg->cof_async) {
        int plo = 0, phi = 0, irc;
        g_cofq_quiet = 1;
        irc = cofq_init(&A.spare, &A.spare_out, CQ_FLUSH,
                        cfg->cof_meth0, cfg->cof_meth1,
                        cfg->ecm_b1, cfg->ecm_b2, cfg->ecm_curves,
                        cfg->cof_limbs0, cfg->cof_limbs,
                        cfg->ecm_curves_set, (uint32_t)cfg->cof_rounds);
        g_cofq_quiet = 0;
        if (irc) { rc = -1; goto done; }
        /* Lowest priority, so the q loop's own kernels win any SM a flush
         * frees -- what glas does with its ECM stream. The legacy stream the
         * loop runs on is itself at the default (lowest) priority today, so
         * on current parts this ties rather than yields. */
        PIPE_CK(mtlDeviceGetStreamPriorityRange(&plo, &phi));
        PIPE_CK(mtlStreamCreateWithPriority(&g_cof_stream,
                                             mtlStreamNonBlocking, plo));
        VRAM_MARK("cofactor queue, second buffer set");
        printf("  cofactor queue: async, double-buffered (--no-cof-async: inline)\n");
    }
    /* --fill-concurrent's admission check, deliberately LAST.
     *
     * It ran beside the first bucket array at first, and that was wrong: the
     * factor bases, bitmaps, trial-division context and cofactor queue had not
     * been allocated yet, so the check passed on free memory that was already
     * spoken for and the run died at the next mtlMalloc instead. Here the
     * remainder is real. The 512 MB margin then only has to cover the per-q
     * buffers that grow on demand afterwards, measured at ~0.12 GB at 15e --
     * the same margin the first bucket array uses, for the same reason.
     *
     * The second array is deliberately the same `cap` rather than one derived
     * from side 0's own est: a smaller cap for the smaller side would change
     * where k_fill_atomic overflows, which changes which records are dropped,
     * which changes the relations. Sizing each side to its own est is a real
     * saving and a SEPARATE change -- it cannot ride along inside one whose
     * gate is output identity.
     *
     * REFUSED, NOT SILENTLY DROPPED. An operator who passed the flag and got a
     * serial run would quote the timings as concurrent. Same rule as the
     * harness_only list in bench_main. */
    if (cfg->fill_concurrent) {
        /* Charge the cursor array too, not just the bucket. It is small
         * against the margin, but this check exists precisely so the run
         * refuses cleanly instead of dying at the next mtlMalloc, and a
         * check that does not count what the next four lines allocate is
         * the same bug in miniature. */
        const size_t need2 = need + (size_t)nregion_alloc * 4 + 4;
        size_t f2 = 0, t2 = 0;
        PIPE_CK(mtlMemGetInfo(&f2, &t2));
        if (need2 + PIPE_VRAM_MARGIN > f2) {
            fprintf(stderr,
                    "  pipeline: --fill-concurrent needs a second bucket array"
                    " + cursors of %.2f GB and only %.2f GB is free after"
                    " setup.\n"
                    "  Drop the flag, or lower the area (J, then logI).\n",
                    need2 / 1073741824.0, f2 / 1073741824.0);
            rc = -1;
            goto done;
        }
        PIPE_CK(mtlMalloc(&d_bucket_c, need));
        PIPE_CK(mtlMalloc(&d_cursor_c, (size_t)nregion_alloc * 4));
        PIPE_CK(mtlMalloc(&d_overflow_c, 4));
        PIPE_CK(mtlStreamCreate(&st_side[0]));
        PIPE_CK(mtlStreamCreate(&st_side[1]));
        concurrent = 1;
        printf("  --fill-concurrent: a SECOND bucket array (%.2f GB), the two"
               " sides on two streams\n", need / 1073741824.0);
        printf("  NOTE: per-side fill/apply times now OVERLAP; the band report"
               " subtracts the overlap\n");
        VRAM_MARK("second bucket array");
    }
    /* The SERIAL arm's second bucket array, when VRAM allows (default on,
     * --no-dual-bucket). Side 0 then fills its own array instead of
     * overwriting side 1's, so side 1's records are still resident at trial
     * division and k_resieve_bucket covers both sides instead of side 1
     * re-walking every lattice (measured under --fill-concurrent: resieve
     * 28.8 -> 21.9 ms/q, RTX 4090). Same launches, same stream, same caps --
     * the relations cannot change. Optional, so never fatal: a card without
     * the room (margin max(1 GiB, 10%), since a volunteer's card may also be
     * driving a display) keeps the one shared array exactly as before. */
    if (!concurrent && cfg->bucket_resieve && cfg->dual_bucket &&
        cfg->log_region <= (int)FINE_SHIFT) {
        const size_t need2 = need + (size_t)nregion_alloc * 4 + 4;
        size_t f2 = 0, t2 = 0;
        PIPE_CK(mtlMemGetInfo(&f2, &t2));
        const size_t gib = (size_t)1 << 30;
        const size_t margin = t2 / 10 > gib ? t2 / 10 : gib;
        if (need2 + margin + pipe_inst_reserve() <= f2) {
            PIPE_CK(mtlMalloc(&d_bucket_c, need));
            PIPE_CK(mtlMalloc(&d_cursor_c, (size_t)nregion_alloc * 4));
            PIPE_CK(mtlMalloc(&d_overflow_c, 4));
            dual = 1;
            printf("  second bucket array (%.2f GB): side 1's records kept for"
                   " the bucket resieve\n", need / 1073741824.0);
            VRAM_MARK("second bucket array (serial)");
        } else {
            printf("  one bucket array: %.2f GB free is short of a second"
                   " (%.2f GB + margin); side 1 resieves by walking\n",
                   f2 / 1073741824.0, need2 / 1073741824.0);
        }
    }
    /* --q-pipeline's admission check, after every other allocation for the
     * same reason as --fill-concurrent's above: free memory is only real once
     * the factor bases, TD context and cofactor queue are resident.
     *
     * It needs a second copy of what the next q's k_transform writes while
     * this q's resieve still reads it: plat for every bucketed entry and
     * rowrt for the row tier. On a slabbed plan, where the next SLAB is also
     * sieved under this slab's TD, a third walk-cursor buffer per entry
     * (walk_spare; see pside_rotate_walk). NOT another bucket array: the
     * next q's fills reuse this q's one or two, and pipe_td_perq issues them
     * only after both sides' bucket resieve has recorded C.ev[14] (st_sv
     * waits on it), so a fill never overwrites records still being read.
     *
     * Under --instances the siblings still to set up are held back
     * (pipe_inst_reserve), as for the second bucket array above; their
     * modelled footprint already prices this transform set.
     *
     * SINGLE-QUEUED, NOT REFUSED, when it does not fit: the flag asks for a
     * speed-up, and a job that runs today must keep running. The banner and
     * the run log say which happened. */
    if (cfg->q_pipeline) {
        char why[160] = "";
        size_t extra = 0;
        for (int side = 0; side < 2; side++) {
            const pside_t *S = side ? &S1 : &S0;
            extra += (size_t)S->nfb * sizeof(plat_t) + (size_t)S->nrow * 4;
            if (SLABBED) extra += (size_t)S->nfb * sizeof(uint64_t);
        }
        /* bench_main refuses these combinations on the command line; a caller
         * reaching this public entry point another way gets the same answer
         * as every other exclusion here, one q at a time. */
        if (concurrent || cfg->qspan) {
            snprintf(why, sizeof why, "%s excludes it",
                     concurrent ? "--fill-concurrent" : "--qspan");
        } else {
            size_t f3 = 0, t3 = 0;
            PIPE_CK(mtlMemGetInfo(&f3, &t3));
            const size_t reserve = pipe_inst_reserve();
            if (extra + PIPE_VRAM_MARGIN + reserve > f3)
                snprintf(why, sizeof why, "it needs %.2f GB more and %.2f GB"
                         " is free%s", extra / 1073741824.0, f3 / 1073741824.0,
                         reserve ? " after the other instances' share" : "");
        }
        if (!why[0]) {
            /* An allocation that fails AFTER the check passed -- another
             * process took the memory, or (WSL2) the driver's free figure was
             * optimistic -- is not a reason to fail a band that runs
             * single-queued. Undo the partial set, clear the error (an
             * out-of-memory status is not sticky) and fall back. */
            mtlError_t e = mtlSuccess;
            for (int side = 0; side < 2 && e == mtlSuccess; side++) {
                pside_t *S = side ? &S1 : &S0;
                e = mtlMalloc(&S->plat_alt, (size_t)S->nfb * sizeof(plat_t));
                if (e == mtlSuccess && S->nrow)
                    e = mtlMalloc(&S->rowrt_alt, (size_t)S->nrow * 4);
                if (e == mtlSuccess && SLABBED)
                    e = mtlMalloc(&S->walk_spare,
                                   (size_t)S->nfb * sizeof(uint64_t));
            }
            if (e == mtlSuccess)
                e = mtlStreamCreateWithFlags(&st_sv, mtlStreamNonBlocking);
            if (e != mtlSuccess) {
                mtlGetLastError();
                for (int side = 0; side < 2; side++) {
                    pside_t *S = side ? &S1 : &S0;
                    mtlFree(S->plat_alt);   S->plat_alt = NULL;
                    mtlFree(S->rowrt_alt);  S->rowrt_alt = NULL;
                    mtlFree(S->walk_spare); S->walk_spare = NULL;
                }
                if (st_sv) { mtlStreamDestroy(st_sv); st_sv = 0; }
                snprintf(why, sizeof why, "allocating it failed (%s)",
                         mtlGetErrorString(e));
            }
        }
        if (why[0]) {
            printf("  --q-pipeline: staying single-queued, because %s\n", why);
            runlog_note("q-pipeline", "single-queued: %s", why);
        } else {
            qpipe = 1;
            if (SLABBED)
                printf("  --q-pipeline: each slab's sieve under the previous"
                       " slab's trial division, across q too; a second"
                       " transform set and a third walk-cursor set (%.2f GB)\n",
                       extra / 1073741824.0);
            else
                printf("  --q-pipeline: two special-q in flight, a second"
                       " transform set (%.2f GB)\n", extra / 1073741824.0);
            printf("  NOTE: stage times now include sharing the card with the"
                   " other q; only wall clock per q is additive, and"
                   " `unaccounted` and GPU-accounted / wall are not reported\n");
            runlog_note("q-pipeline", "%s, +%.2f GB",
                        SLABBED ? "slabs and q overlapped" : "two q in flight",
                        extra / 1073741824.0);
            VRAM_MARK("q-pipeline transform set");
        }
    } else {
        runlog_note("q-pipeline", "off");
    }
#undef VRAM_MARK
    /* Name the fill knobs in the banner: a knob the log does not show is one
     * a reader cannot prove was live. */
    if (cfg->fill_rows)
        printf("  --fill-rows %d: q < %d*I filled by row (side 1: %u entries,"
               " side 0: %u), %s\n", cfg->fill_rows, cfg->fill_rows,
               S1.nrow, S0.nrow,
               cfg->fill_row_bands ? "fixed row bands" : "auto row bands");
    if (cfg->resieve_rows)
        printf("  --resieve-rows: the same tier is resieved by row\n");
    if (cfg->fill_gate)
        printf("  --fill-gate: side 0 stores fill records only under side 1's"
               " survivors, 1 gate bit per %u positions\n",
               1u << cfg->fill_gate_shift);

    /* Serial mode points BOTH sides at the one workspace on the legacy default
     * stream, which is what the two sides have always shared -- so the slab
     * loop below runs the identical sequence of launches on the identical
     * stream as before the issue/join split. Concurrent mode gives side 0 the
     * second array and each side its own stream. */
    bk1.bucket = d_bucket; bk1.cursor = d_cursor;
    bk1.overflow = d_overflow; bk1.cap = cap;
    bk1.stream = concurrent ? st_side[1] : 0;
    if (concurrent) {
        bk0.bucket = d_bucket_c; bk0.cursor = d_cursor_c;
        bk0.overflow = d_overflow_c; bk0.cap = cap;
        bk0.stream = st_side[0];
    } else if (dual) {
        /* Serial, on the legacy stream like the one-array arm; only the
         * workspace differs. */
        bk0 = bk1;
        bk0.bucket = d_bucket_c; bk0.cursor = d_cursor_c;
        bk0.overflow = d_overflow_c;
        bk0.stream = 0;
    } else {
        bk0 = bk1;
        /* Redundant today -- bk1.stream is provably 0 on this branch -- and
         * kept deliberately: it pins the serial arm to the legacy default
         * stream at the one place a reader looks to see which side gets which
         * stream, so a later edit to the ternary above cannot hand the serial
         * path a side stream by accident. */
        bk0.stream = 0;
    }
    /* --q-pipeline: both sides on the one non-blocking sieve stream, so the
     * gate's side-1-before-side-0 order is stream order and a bucket array is
     * never filled by two q at once; TD and cofactorisation stay on the
     * legacy stream, which a non-blocking stream does not wait for. Each
     * side keeps its own array when the serial arm has two (dual): only the
     * stream moves. The next unit's fill waits for this one's resieves on
     * C.ev[14] (see issue_next), the last readers of either array. */
    if (qpipe) {
        bk1.stream = st_sv;
        bk0.stream = st_sv;
    }

    if (cfg->relations && cfg->candidates &&
        !strcmp(cfg->relations, cfg->candidates)) {
        fprintf(stderr, "  --relations and --candidates must differ\n");
        rc = -1; goto done;
    }
    /* Written through temporaries and renamed only on success: a band that
     * fails at q 20 must not leave files that look like a complete run.
     *
     * The .part is nonetheless the DURABLE artifact, not scratch. The rename is
     * only the "band completed" marker; an interrupted run leaves the .part and
     * its sidecar in place to be resumed, and this used to delete both. */
    /* Build BOTH staging names before opening either file. Failing on the
     * candidates name after fr is open would, on a resume, already have
     * ftruncated the relation .part to the checkpointed prefix -- and the
     * jump to done: discards with keep_partial = ckpt_written = 0, deleting
     * a .part that holds every earlier session's work. bench_main gates both
     * paths on ckpt_path_usable() before this, so these cannot fire; they
     * keep run_pipeline safe as the public entry point it is. */
    if ((cfg->relations && ckpt_part_path(cfg->relations, rtmp, sizeof rtmp)) ||
        (cfg->candidates && ckpt_part_path(cfg->candidates, ctmp, sizeof ctmp))) {
        fprintf(stderr, "run_pipeline: output path too long for its .part name\n");
        rc = -1; goto done;
    }
    /* Binary mode, on every staging open, on both platforms. Three things
     * depend on it and all three fail quietly in text mode on Windows:
     * relations are a wire format shipped to msieve/CADO, and a CRLF copy is
     * not the same file the Linux build produces; every reader of these two
     * paths opens "rb", so a line stored as "\r\n" reaches the parser with a
     * stray \r and is counted a bad relation by the resume gate; and the
     * checkpoint records bench_tell() on this stream to compare against a
     * physical stat size, which a translating stream makes a different
     * number, so bench_truncate() would cut mid-line on resume. */
    if (cfg->relations) {
        if (cfg->resume) {
            /* Truncate to the checkpointed prefix before appending. This is
             * what makes a torn final line from a kill -9 a non-problem: the
             * partial write is discarded rather than parsed. */
            if (!(fr = fopen(rtmp, "r+b"))) { perror(rtmp); rc = -1; goto done; }
            if (bench_truncate(fr, cfg->resume_rel_bytes) ||
                bench_seek(fr, cfg->resume_rel_bytes)) {
                perror(rtmp); rc = -1; goto done;
            }
        } else if (!(fr = fopen(rtmp, "wb"))) {
            perror(rtmp); rc = -1; goto done;
        }
    }
    if (cfg->candidates) {
        if (cfg->resume) {
            if (!(fc = fopen(ctmp, "r+b"))) { perror(ctmp); rc = -1; goto done; }
            if (bench_truncate(fc, cfg->resume_cand_bytes) ||
                bench_seek(fc, cfg->resume_cand_bytes)) {
                perror(ctmp); rc = -1; goto done;
            }
        } else if (!(fc = fopen(ctmp, "wb"))) {
            perror(ctmp); rc = -1; goto done;
        }
    }
    /* Only meaningful once there is a relation file to point at. */
    /* Unconditional, NOT under `if (fr)`. pipe_checkpoint returns 0 -- success
     * -- at its own `if (!fr || !cfg->relations)` guard, so pipe_try_checkpoint
     * sets ckpt_written even when nothing was written, and the terminal-status
     * messages below then format ck.nrel/ck.next_q. With ck left indeterminate
     * that is undefined behaviour, reachable by any policy stop in a run with
     * no --relations (i.e. plain benchmarking). Zeroing costs nothing. */
    memset(&ck, 0, sizeof ck);
    if (fr) {
        ckpt_fingerprint(POLY, cfg, ck.fp);
        /* --resume starts from an existing valid sidecar; preserve the .part
         * even if this session fails before advancing the checkpoint. */
        if (cfg->resume) {
            ckpt_written = 1;
            ck.next_q = cfg->resume_q; ck.next_rho = cfg->resume_rho;
            ck.rel_bytes = cfg->resume_rel_bytes;
            ck.cand_bytes = cfg->resume_cand_bytes;
            ck.nrel = cfg->resume_nrel; ck.nqdone = cfg->resume_nq;
        }
        ck.scale = cfg->scale; ck.scale0 = cfg->scale0;
        ck.allowance = cfg->allowance; ck.allowance0 = cfg->allowance0;
        ckpt_armed = 1;
        /* See bench_stop_hook_install: on Windows this is a console control
         * handler, because signal(SIGTERM) is never delivered there. A task
         * stopped with TerminateProcess still runs nothing at all, so
         * --stop-file remains the only reliable managed stop on Windows. */
        /* --instances: run_pipeline_instances owns the hook and the flag for
         * all of them; one instance finishing must not unhook the others. */
        if (!g_pinst) {
            stop_hooked = bench_stop_hook_install(pipe_request_stop) == 0;
            g_pipe_stop = 0;
        }
    }

    /* ---- the band ---- */
    {
    const double t_band = host_ms();
    double t_report = t_band, t_ckpt = t_band;
    /* The \r line is for a human watching a terminal. Redirected to a file it
     * is a single unreadable line that grows a carriage return every 30 s, so
     * a redirect gets whole lines every five minutes instead -- ~864 of them
     * over a three-day band rather than 8,640.
     *
     * FIVE MINUTES IS THIS STREAM'S OWN CONSTANT, not --log-every. Reading the
     * log's period here coupled two unrelated outputs: `--log-every 86400`
     * with no --log at all would have left a redirected run printing one
     * progress line per day, and a plain redirect with no logging flags would
     * have silently moved from 30 s to 300 s because log_every_s defaults to
     * 300 whether or not a log exists. The run log (--log) is a separate,
     * richer stream on its own clock. */
    /* --instances: whole lines, tagged, since several pipelines report. */
    const int reporting_to_tty = bench_stdout_is_tty() && !g_pinst;
    char ipfx[16] = "";
    if (g_pinst) snprintf(ipfx, sizeof ipfx, "[%d] ", g_pinst->k);
    const double report_ms = reporting_to_tty ? 30000.0 : 300000.0;
#ifdef HAVE_BOINC
    double t_boinc_report = t_band - 1000.0; /* report after the first q */
#endif
    /* The side-0 special-q polynomial that used to be built here is gone:
     * qsel_validate() derives G = Y1*x + Y0 itself when sq_side == 0, so the
     * per-q gate below covers what pipe_check_root() did and more. */
    /* From here on a stall is a stall: every phase below is milliseconds, so
     * the watchdog's give-up threshold can be trusted. It is deliberately NOT
     * armed during setup, where factor-base generation and a multi-gigabyte
     * resume scan sit under one coarse label and can legitimately run for
     * minutes. */
    wd_arm_kill();
    if (g_pinst) g_pinst->sh->setups_left--;
    pg.unlock();            /* --instances: the band runs alongside the others */
    /* --q-pipeline: the special-q selected ahead of the loop top that will
     * process it, during the previous q's last slab (prepared before that
     * slab's join, issued after its intersection, finding 114). spec_qr
     * replays the source step's outcome there, and spec_issued says its
     * sieve is already running on st_sv. A q that was prepared but not
     * issued -- the norm-width skip, the stop flag, or a soft skip of the
     * previous q -- is prepared again by the loop top, which only then warns
     * and counts a skip. */
    int spec_have = 0, spec_issued = 0, spec_qr = 0;
    qsel_t spec_raw = {};
    double spec_th1 = 0, spec_th0 = 0;
    /* One step of the special-q source: 1 = a q, 0 = none left, -1 = failure.
     * qlist first, then the streaming generator. The loop top and the
     * --q-pipeline issue point both pull through this, and both ask
     * count_limit_before first. */
    auto pull_q = [&](uint32_t idx, qsel_t *out) -> int {
        if (idx < nq) { *out = qlist[idx]; return 1; }
        if (qgen) {
            const int r = sqgen_next(qgen, out);
            return r < 0 ? -1 : (r > 0 ? 1 : 0);
        }
        return 0;
    };
    /* The generator's budget, checked before pulling position idx with `done`
     * q completed. Enforced here as well as in the generator, because a
     * resumed run hands sqgen no limit at all -- see the sqgen_create comment
     * in bench_main.cu. Without this the budget would not exist. A qlist is
     * bounded by its own length. */
    auto count_limit_before = [&](uint32_t idx, uint32_t done) -> int {
        return idx >= nq && qgen && cfg->nq_max && done >= cfg->nq_max;
    };
    /* Both sides' preparation, short-circuiting on failure or skip like the
     * `||` chain it replaced: 0, PIPE_Q_SKIP, or negative. One side being
     * unfittable is enough to pass over the q, and running side 0 after side
     * 1 failed would launch work against a sticky error and blame side 0. */
    auto prepare_sides = [&](const qlat_t *L, double *h1, double *h0,
                             int quiet_skip) -> int {
        const int p1 = pipe_side_prepare_q<SLABBED>(fb1, fbs1, L, POLY, cfg, 1,
                                                    cfg->scale, blocks, &S1, h1,
                                                    st_sv, quiet_skip);
        if (p1 != 0) return p1;
        return pipe_side_prepare_q<SLABBED>(fb0, fbs0, L, POLY, cfg, 0,
                                            cfg->scale0, blocks, &S0, h0,
                                            st_sv, quiet_skip);
    };
    /* One side's sieve issue for one slab. Side 0 is gated on side 1's bitmap
     * when the gate is on, which it never is under --fill-concurrent. */
    auto issue_side = [&](int side, uint32_t xmax, uint32_t j_base) -> int {
        return side
            ? pipe_side_sieve_issue<SLABBED>(fb1, cfg, 1, xmax, j_base, &bk1,
                                             fblocks, fthreads, &S1)
            : pipe_side_sieve_issue<SLABBED>(fb0, cfg, 0, xmax, j_base, &bk0,
                                             fblocks, fthreads, &S0,
                                             cfg->fill_gate ? S1.survbits : NULL,
                                             d_gate);
    };
    /* --q-pipeline: both sides of slab `sl` (of the q currently prepared)
     * issued back to back on st_sv, with no host wait; the joins come at the
     * loop top that consumes it. A failed issue leaves launches in flight, so
     * the stream is drained before the caller unwinds. Counted (n_ahead_*),
     * so a band shows how often each kind of overlap actually happened. */
    auto issue_ahead = [&](uint32_t sl) -> int {
        const uint32_t xm = I * slab_rows_at(slab_plan, cfg->J, sl);
        const uint32_t jb = slab_jbase_at(slab_plan, sl);
        if (issue_side(1, xm, jb) || issue_side(0, xm, jb)) {
            mtlStreamSynchronize(st_sv);
            return -1;
        }
        return 0;
    };
    for (uint32_t qi = 0;; qi++) {
        qsel_t generated, checked;
        const qsel_t *cur;
        qlat_t Lq;
        float ts1[3] = {0,0,0}, ts0[3] = {0,0,0}, tis = 0;
        double th1 = 0, th0 = 0, qwall = host_ms(), tv = 0;
        double hspan0 = 0, hreg = 0, prologue_q = 0;
        /* Per-q locals for the same reason prologue_q is one: the
         * report divides every region by N == nqdone, so a q that is
         * abandoned before the accumulation block must contribute to
         * NEITHER side of that ratio. Charged straight into the
         * accumulators, they were added by lost q that never reached
         * nqdone++, inflating exactly these three lines against an
         * honest acc_qspan/acc_hspan and silently breaking the
         * region-sum reconciliation -- the hazard the comment at the
         * accumulation block names for the prologue. */
        double prep_q = 0, tdprep_q = 0, slab_q = 0;
        /* Same rule as prep_q/tdprep_q/slab_q, but these two cannot be
         * localised the same way: acc_td is charged per SLAB inside the loop,
         * and pipe_td_perq adds into tm several levels down. Snapshot and roll
         * back instead, on the one path that abandons a q after charging them
         * -- the all-slabs-skipped `continue` below, which does not increment
         * nqdone. Left alone, `acc_td / N` and the tm-derived `dev` divide a
         * numerator that includes lost q by a denominator that excludes them,
         * and the `unaccounted` residual can go negative. */
        const double     acc_td_q0 = acc_td;
        const double     acc_ovl_q0 = acc_ovl;
/* Abandoning a q means every band-level accumulator it charged INSIDE the slab
 * loop has to come back, because `nqdone` is not incremented and the band
 * summary still prints: the totals would then be divided by an N that excludes
 * the q, and `unaccounted` can go negative. That is the un-reconcilable stage
 * total finding 94 exists to forbid.
 *
 * It is a macro because there are THREE abandon sites, not one -- the nq_lost
 * path, and two hard failures further down (`no survivors at this q`, and the
 * cofactor-gate abort). Twice now this rollback has been written out by hand at
 * one site and missed at the others: first acc_ovl was added to nq_lost alone,
 * then acc_td and tm were restored at the hard-failure break and not at the two
 * below it. Open-coding it is what keeps producing that bug.
 *
 * NOT included, deliberately: `nslab_skipped` and `ntd_skipped`. They are raw
 * band totals reported as counts, never divided by N, and a skip that really
 * happened should stay counted. The rule binds the per-q TIME accumulators. */
#define PIPE_Q_ROLLBACK() do { acc_td = acc_td_q0; acc_ovl = acc_ovl_q0; \
                               tm = tm_q0; } while (0)
        const pipe_tm_t  tm_q0     = tm;
        if (cfg->qspan) { hspan0 = host_ms(); PIPE_CK(mtlEventRecord(qspan0)); }
        uint32_t hn = 0, nacc = 0, ncand = 0, nrel = 0;
        uint64_t side_surv1 = 0, side_surv0 = 0;
        uint32_t cofgate_found[2] = {0, 0};
        int td_verified = 0;
        /* The host loop exists to write files. With inline cofactorisation and
         * no candidate file, nothing reads the host mirrors, so neither the
         * copies nor the loop have to happen.
         *
         * cfg->emit_cof is NOT a third case: --emit-cof is in harness_only and
         * is rejected outright under --pipeline, so it is always NULL here.
         * Including it made this look like it had three outcomes when it has
         * two, and the end-of-band stats fold-in depends on this and
         * accumulate_stats being exact complements. */
        const int want_host = !cfg->cofactor || fc;

        /* A BOINC suspend pauses only the thread that called boinc_init. With
         * --instances that thread is parked in a join and the sieving happens
         * here, so each instance pauses itself at the special-q boundary --
         * otherwise a "suspended" task goes on using the GPU. Its watchdog
         * slot reads as idle meanwhile, so a long suspend is not a stall. A
         * quit needs nothing here: the runtime exits the whole process. */
        if (g_pinst && bench_boinc_suspended()) {
            wd_active(0);
            while (bench_boinc_suspended() && !g_pipe_stop)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            wd_active(1);
        }

        wd_phase("q.select");
        int pre_issued = 0, ts_read = 0, qr;
        if (spec_have) {
            /* Pulled during the previous q's last slab. Its outcome is replayed
             * below, so a failed or exhausted source is reported at the same
             * loop top, with the same nqdone, as it would have been. */
            spec_have = 0;
            qr = spec_qr;
            generated = spec_raw;
            pre_issued = spec_issued;
            if (pre_issued) { th1 = spec_th1; th0 = spec_th0; }
        } else if (count_limit_before(qi, nqdone)) {
            count_limit_reached = 1;
            break;
        } else {
            qr = pull_q(qi, &generated);
        }
        if (qr < 0) {
            fprintf(stderr, "  special-q generator failed after %u q\n",
                    nqdone);
            rc = -1; break;
        }
        if (qr == 0) {
            if (cfg->nq_max && nqdone >= cfg->nq_max)
                count_limit_reached = 1;
            else
                source_exhausted = 1;
            break;
        }
        cur = &generated;
        /* Validate again at the consumer boundary. bench_main validates every
         * input before scale derivation, but run_pipeline is a public entry
         * point and the streaming generator supplies later q directly here. */
        checked = *cur;
        {
            const qsel_validate_result_t qv =
                qsel_validate(&checked, POLY, cfg->sq_side);
            if (qv != QSEL_VALID) {
                const char *why = qv == QSEL_ERR_Q_RANGE ? "q outside [2,2^32)"
                                : qv == QSEL_ERR_Q_COMPOSITE ? "composite q"
                                : qv == QSEL_ERR_NOT_ROOT ? "rho is not a root"
                                : qv == QSEL_ERR_SIDE ? "invalid special-q side"
                                : "invalid exact polynomial";
                fprintf(stderr, "  q=%llu rho=%llu: special-q rejected: %s\n",
                        (unsigned long long)checked.q,
                        (unsigned long long)checked.rho, why);
                rc = -1; break;
            }
        }
        cur = &checked;
        /* The context every later report is read against, set HERE and not
         * further down: the seed checkpoint and the 30 s periodic checkpoint
         * below both already work on THIS q, and pipe_try_checkpoint's fsync is
         * the one host phase that can legitimately block for seconds. Set after
         * those, a stall in them was reported against the PREVIOUS special-q --
         * exactly the confusion the placement is supposed to prevent. */
        wd_q((unsigned long long)cur->q, (unsigned long long)cur->rho,
             (unsigned long long)nqdone);

        /* A cofactor queue can first fill in slab 1, 2, ... of q0. Such a
         * flush contains a partial q and is not a resume point. Seed a safe
         * empty/prefix checkpoint before any work so a crash can always keep
         * the .part and replay this q from a whole-q boundary. */
        if (fr && cfg->cofactor && !ckpt_written && !nqdone && Q.n == 0)
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + Q.nrel, base_nq + nqdone,
                                &ckpt_written, &ckpt_warned);

        /* ---- degradation ceiling ----
         *
         * A POLICY STOP, exactly like the norm-width cap above and for the
         * same reason. The per-slab soft skips trade yield for keeping the
         * band alive, which is right for a transient; past a point a band
         * that keeps making that trade has stopped doing the work it was
         * given, and nothing else will say so -- skipping is designed not to
         * fail the task, so the alternative is exiting 0 having emitted
         * almost nothing.
         *
         * Checked HERE, at the top of a q with no work done for it, because
         * that makes `cur` exactly the resume point and because the previous
         * q's ev[4] synchronise has already drained the stream. Checking it
         * mid-slab-loop would checkpoint a partial q.
         *
         * It drains and checkpoints -- every relation already earned is valid
         * and is kept -- but reports PIPE_RC_DEGRADED, so the exit status
         * says the band as a whole is not creditable. Draining is what this
         * shares with a clean stop; "the work unit succeeded" is not. See
         * BENCH_EXIT_DEGRADED in bench.h. */
        if (nq_lost >= (unsigned long long)PIPE_LOST_MAX ||
            nslab_skipped >= (unsigned long long)PIPE_SLAB_SKIP_MAX) {
            runlog_warn("  ** stopping at q=%llu: %llu slab(s) skipped and"
                        " %llu special-q lost entirely (limits %d slab / %d"
                        " q). The band ran, but its yield no longer means"
                        " anything. %s Relations already written are kept and"
                        " are valid.",
                        (unsigned long long)cur->q, nslab_skipped, nq_lost,
                        PIPE_SLAB_SKIP_MAX, PIPE_LOST_MAX,
                        /* Which remedy to name follows the MAJORITY, not
                         * the mere presence of a trial-division skip: one
                         * truncated list among a thousand bucket overflows
                         * used to print the PIPE_K advice and steer the
                         * operator away from the actual cause. Norm overflow
                         * is fatal now, so a TD skip can only be a truncated
                         * list -- a PIPE_K/mfb problem, never a bucket one. */
                        ntd_skipped * 2 >= nslab_skipped
                          ? "Mostly trial-division skips: the large-prime list"
                            " ran past PIPE_K per survivor, so check mfb and"
                            " PIPE_K rather than the bucket array."
                          : "Mostly bucket overflow: check this job's bucket"
                            " array size first (the counts above separate the"
                            " two).");
            if (cofa_wait(&A, &Q)) { rc = -1; break; }
            if (cfg->cofactor && Q.n &&
                cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                           (int)Q.ecm_rounds, cfg->cof_budget, cfg->cof_chunk, blocks,
                           cfg->threads, fr)) { rc = -1; break; }
            /* Guarded on fr, like the clean stop above and like the
             * norm-width cap: without a relation file there is nothing to
             * checkpoint, and letting ckpt_written be set anyway would print a
             * "resume at q=" line naming a .part that does not exist. */
            if (fr)
                pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                    base_rel + (cfg->cofactor ? Q.nrel
                                        : (unsigned long long)acc_rel),
                                    base_nq + nqdone, &ckpt_written,
                                    &ckpt_warned);
            stopped = 1;
            degraded = 1;
            break;
        }

        /* ---- clean stop ----
         *
         * Checked here, with the next (q, rho) in hand but no work done for it,
         * because that pair is exactly the resume point: drain the queue so
         * every q already sieved reaches the file, checkpoint, and leave this
         * one for the next session. A planned stop loses nothing.
         *
         * This runs AFTER the validation gate above so that the pair written to
         * the sidecar is always one a rerun will accept -- and with the rho
         * qsel_validate canonicalised, not whatever width the source supplied.
         *
         * The stop file is for unattended runs, where there is no terminal to
         * press ^C in and a job queue needs a way to ask for the card back. */
        if (fr && !g_pipe_stop && cfg->stopfile && (qi & 63) == 0 &&
            bench_path_exists(cfg->stopfile)) {
            printf("\n  stop file %s exists\n", cfg->stopfile);
            g_pipe_stop = 1;
        }
        if (fr && g_pipe_stop) {
            printf("\n  stopping cleanly at q=%llu; draining the cofactor"
                   " queue\n", (unsigned long long)cur->q);
            wd_phase("stop.cofq_flush");
            if (cofa_wait(&A, &Q)) { rc = -1; break; }
            if (cfg->cofactor && Q.n &&
                cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                           (int)Q.ecm_rounds, cfg->cof_budget, cfg->cof_chunk, blocks,
                           cfg->threads, fr)) { rc = -1; break; }
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + (cfg->cofactor ? Q.nrel
                                                          : (unsigned long long)acc_rel),
                                base_nq + nqdone, &ckpt_written, &ckpt_warned);
            stopped = 1;
            break;
        }
        /* Without --cofactor the host join loop writes each q's relations as it
         * finishes, so every q boundary is a valid resume point and the only
         * question is how often to pay the fsync. Under --cofactor this is
         * handled at the flush instead, and doing it here as well would record
         * a point the queue's contents are not in the file for. */
        if (fr && !cfg->cofactor && qi && host_ms() - t_ckpt > 30000.0) {
            t_ckpt = host_ms();
            pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                base_rel + (unsigned long long)acc_rel,
                                base_nq + nqdone, &ckpt_written, &ckpt_warned);
        }

        wd_phase("q.lattice");
        qlat_build(&Lq, cur->q, cur->rho, POLY->skew);
        if (cfg->verbose_q)
            printf("\n  q = %llu, rho = %llu\n",
                   (unsigned long long)cur->q,
                   (unsigned long long)cur->rho);

        {
            /* SHORT-CIRCUITS ON FAILURE, like the `||` chain this replaced.
             * Running side 0 after side 1 has already failed launches more CUDA
             * work against a sticky error, and CUDA_CHECKED then reports side 0
             * for side 1's fault. A SKIP short-circuits too -- one side being
             * unfittable is enough to pass over the q, and the warning naming
             * the side has already been printed by the side that raised it. */
            if (cfg->qspan) {
                prologue_q = host_ms() - hspan0;
                hreg = host_ms();
            }
            wd_phase("q.prepare");
            /* A q issued ahead was prepared at the issue point. */
            const int pq = pre_issued ? 0 : prepare_sides(&Lq, &th1, &th0, 0);
            if (pq == PIPE_Q_SKIP) {
                if (++nqskip >= (unsigned long long)PIPE_SKIP_MAX) {
                    /* A POLICY STOP, NOT A CRASH. Falling out with rc = -1
                     * skipped the post-loop cofactor flush (guarded on rc == 0)
                     * and threw away every relation queued since the last one,
                     * then reported the band as FAILED with no checkpoint. This
                     * is a deliberate stop on a known condition, so it drains
                     * and checkpoints exactly like the stop-file path.
                     *
                     * IT DOES NOT EXIT LIKE ONE. `capped` carries the reason
                     * out to the return mapping, which reports it as
                     * PIPE_RC_UNSUPPORTED: draining is what the two stops share,
                     * "this work unit succeeded" is not. */
                    if (g_pipe_need_limbs > 0)
                        runlog_warn("  ** %llu special-q skipped for norm width;"
                                    " this build is too narrow for the job."
                                    " Rebuild with `make BN_LIMBS=%d` (this"
                                    " build carries %d) and run ./normscan over"
                                    " the band first.",
                                    nqskip, g_pipe_need_limbs, BN_LIMBS);
                    else
                        runlog_warn("  ** %llu special-q skipped for norm width;"
                                    " this build is too narrow for the job and"
                                    " no supported BN_LIMBS is wide enough for"
                                    " every skipped (q,rho). Run ./normscan"
                                    " over the band.", nqskip);
                    if (cofa_wait(&A, &Q)) { rc = -1; break; }
                    if (cfg->cofactor && Q.n &&
                        cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim,
                                   cfg->lpb, (int)Q.ecm_rounds, cfg->cof_budget, cfg->cof_chunk,
                                   blocks, cfg->threads, fr)) { rc = -1; break; }
                    /* Guarded on fr, like the degradation ceiling above.
                     * pipe_checkpoint returns 0 -- success -- at its own
                     * `if (!fr || !cfg->relations)` guard, i.e. it reports
                     * success for correctly doing nothing, so calling this
                     * unguarded set ckpt_written with no sidecar on disk and
                     * the terminal message then printed "checkpoint written,
                     * resume at q=0" naming a .part that was never created.
                     * Reachable only by plain benchmarking on a too-narrow
                     * build, so it cost a wrong message rather than a wrong
                     * outcome -- but the two policy stops must not disagree
                     * about the same hazard. */
                    if (fr)
                        pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                            base_rel + (cfg->cofactor ? Q.nrel
                                                : (unsigned long long)acc_rel),
                                            base_nq + nqdone, &ckpt_written,
                                            &ckpt_warned);
                    stopped = 1;
                    capped = 1;
                    break;
                }
                continue;              /* next q; nqdone is not incremented */
            }
            if (cfg->qspan) { prep_q   = host_ms() - hreg; hreg = host_ms(); }
            wd_phase("q.td_prepare");
            if (pq < 0 || pipe_td_prepare_q(&C, fbs1, fbs0, &Lq, cfg, &tm)) {
                rc = -1; break;
            }
            /* AFTER the skip check, not before: last_q names the band's final
             * progress point and feeds the BOINC fraction, and a skipped q was
             * never sieved. */
            last_q = *cur;
        }

        /* Only the area-dependent part lives inside this loop. For
         * SLABBED=false slab_plan has exactly one entry and if constexpr
         * removes every continuation-state operation from device code. */
        if (cfg->qspan) { tdprep_q = host_ms() - hreg; hreg = host_ms(); }
        uint32_t nslab_skipped_q = 0;   /* soft-skipped slabs, THIS q */
        /* --q-pipeline: the slab about to be processed was already issued,
         * at the previous slab's issue point (or, for slab 0, the previous
         * q's). Only its joins remain, and its walk cursors were turned
         * there. */
        int ahead = pre_issued;
        /* --q-pipeline: the next q was prepared during this q's last slab
         * (next_prepared), and may be issued (next_ready); its (q, rho) for
         * the watchdog at the issue. q-level because the end-of-q drain
         * below needs them after the slab loop. */
        int next_prepared = 0, next_ready = 0;
        unsigned long long next_wq = 0, next_wrho = 0;
        /* The next slab's cursors were already turned at THIS slab's issue
         * point, so the end-of-slab advance must not swap them again. Set
         * only at that issue point, cleared at the top of every slab: one
         * meaning, one slab. */
        int rotated = 0;
        /* The per-q continuation advance, factored out because THREE paths
         * must run it and one of them is a skip: the ordinary end of the slab
         * loop, and the trial-division skip that keeps going. Returns -1 on a
         * device failure. Skipping it leaves the next slab reading the
         * previous slab's walk positions against its own j_base. When the
         * next slab was issued ahead, its cursors were already rotated at the
         * issue point; only trial division's small-prime origin moves here. */
        auto advance_slab = [&](uint32_t sl, uint32_t rows) -> int {
            if constexpr (SLABBED) {
                if (sl + 1 < slab_plan->nslab) {
                    if (pipe_td_advance_small(&C, rows, blocks, cfg->threads))
                        return -1;
                    if (!rotated) {
                        std::swap(S1.walk_cur, S1.walk_next);
                        std::swap(S0.walk_cur, S0.walk_next);
                    }
                }
            } else {
                (void)sl; (void)rows;
            }
            return 0;
        };
        for (uint32_t slab = 0; slab < slab_plan->nslab; slab++) {
            wd_slab(slab, slab_plan->nslab);
            const uint32_t j_base = slab_jbase_at(slab_plan, slab);
            const uint32_t J_here = slab_rows_at(slab_plan, cfg->J, slab);
            const uint32_t xmax = I * J_here;
            const uint32_t nbitword = xmax >> 5;
            float sf1 = 0, sa1 = 0, sf0 = 0, sa0 = 0;
            uint32_t hn_s = 0, nacc_s = 0;
            double td_start, tv_before, cf_start, join_start;

            /* Split soft from hard. A positive return is "this slab is
             * unusable, skip it"; negative is a real device/API failure that
             * the band cannot continue past. Short-circuit || is kept: if
             * side 1 is unusable there is no point sieving side 0, and the
             * skip below means neither side's survbits are ever read. */
            int sv1, sv0 = 0;
            rotated = 0;
            /* --q-pipeline: preparing the next q. This q's transform time
             * first (preparing the next q records the same two events again,
             * and that wait also guarantees this q's table uploads have left
             * the pinned staging buffers the preparation overwrites); then
             * the plat/rowrt sets swap and, slabbed, the walk cursors turn,
             * because the next q's k_transform writes them while this q's TD
             * still reads its own (tdS1/tdS0, taken before). Pulls the next
             * q once -- the source is a generator -- and leaves spec_* for
             * the loop top. next_ready says it is prepared and may be issued.
             * Shared by both call sites so the two cannot drift. */
            ptd_side_t tdS1 = {}, tdS0 = {};
            auto prepare_next_q = [&]() -> int {
                if (pipe_read_transform(&S1, &S0, &ts1[0], &ts0[0])) return -1;
                ts_read = 1;
                pside_swap_alt(&S1);
                pside_swap_alt(&S0);
                /* Slabbed: the next q's k_transform writes walk_cur, which
                 * this last slab's TD still reads; turn the cursors as for a
                 * next slab. The new walk_cur is this slab's output, which
                 * nothing needs once the q is over. */
                if (SLABBED) {
                    pside_rotate_walk(&S1);
                    pside_rotate_walk(&S0);
                }
                spec_have = 1; spec_issued = 0;
                spec_th1 = spec_th0 = 0;
                wd_phase("q.select_next");
                spec_qr = pull_q(qi + 1, &spec_raw);
                if (spec_qr == 1) {
                    qsel_t vq = spec_raw;
                    if (qsel_validate(&vq, POLY, cfg->sq_side) == QSEL_VALID) {
                        qlat_t Ln;
                        qlat_build(&Ln, vq.q, vq.rho, POLY->skew);
                        next_wq = vq.q; next_wrho = vq.rho;
                        /* The watchdog names the q being worked on: the next
                         * one while it is prepared, this one again after. */
                        wd_q((unsigned long long)vq.q, (unsigned long long)vq.rho,
                             (unsigned long long)nqdone);
                        wd_phase("q.prepare_next");
                        /* Silent here, and the setup print is left as the
                         * single-queued run leaves it: a q prepared ahead is
                         * not yet a q the band has reached. */
                        const int nv = norm_verbose;
                        norm_verbose = 0;
                        const int pn = prepare_sides(&Ln, &spec_th1, &spec_th0, 1);
                        norm_verbose = nv;
                        wd_q((unsigned long long)cur->q, (unsigned long long)cur->rho,
                             (unsigned long long)nqdone);
                        if (pn < 0) return -1;
                        /* A skip is not issued: the loop top prepares the q
                         * again, warns and counts it if it gets that far. */
                        next_ready = pn == 0;
                    }
                    /* An invalid q is not issued; the loop top validates it
                     * again and fails there, exactly as it would have. */
                }
                return 0;
            };
            /* ONE POINT, every arm: once this slab is fully issued and before
             * its last join. TD's views are taken here (nothing between here
             * and TD changes them, except the swap and rotation below, which
             * is why they come first). On --q-pipeline's last slab, the next
             * q is prepared here too, on the host, while this slab's sieve
             * runs -- not after this q's intersect, where the card waited for
             * it (0.39 ms/q at C143, finding 113). Its uploads and k_transform
             * queue on st_sv behind this slab's apply. Same stop and budget
             * conditions as the issue point used, which now only issues.
             *
             * A FAILURE HERE MAY NOT BE THE PREPARATION'S. This slab's sieve
             * is still in flight, so a fault in its fill or apply surfaces
             * at the first CUDA call made here, not at its join. Drain, and
             * say so, rather than let the error name the next q's upload. */
            auto views_and_prep = [&]() -> int {
                tdS1 = pipe_td_side(&S1);
                tdS0 = pipe_td_side(&S0);
                if (!(qpipe && slab + 1 == slab_plan->nslab && !g_pipe_stop &&
                      !count_limit_before(qi + 1, nqdone + 1)))
                    return 0;
                next_prepared = 1;
                if (prepare_next_q()) {
                    const mtlError_t e = mtlStreamSynchronize(st_sv);
                    fprintf(stderr, "  q=%llu: the error above surfaced while"
                            " preparing the next q, with this q's last slab"
                            " still sieving (%s); either may be its source\n",
                            (unsigned long long)cur->q, mtlGetErrorString(e));
                    return -1;
                }
                return 0;
            };
            if (ahead) {
                ahead = 0;
                if (views_and_prep()) { rc = -1; break; }
                /* --q-pipeline: both sides were issued back to back on st_sv
                 * at the previous slab's or q's issue point, so only the joins
                 * remain.
                 * On a soft skip this behaves like the concurrent arm: side 0
                 * ran anyway, and the q is abandoned below either way. */
                sv1 = pipe_side_sieve_join(1, &S1, &sf1, &sa1, 1);
                /* Side 1 failing hard is not a reason to call into a sticky
                 * error again and have it blamed on side 0. Drain the stream
                 * so `done:` frees nothing a kernel is still writing. */
                if (sv1 < 0) mtlStreamSynchronize(st_sv);
                else sv0 = pipe_side_sieve_join(0, &S0, &sf0, &sa0, 1);
            } else if (concurrent) {
                /* Both sides go to the device before either is awaited, and
                 * NOTHING between the two issues may touch the legacy default
                 * stream -- see pipe_side_sieve_issue. The short-circuit below
                 * is therefore gone in this arm: side 0 has already been
                 * issued by the time side 1's overflow is known. That costs a
                 * wasted side-0 slab on the rare soft skip and buys the thing
                 * the option exists for.
                 *
                 * It also makes S0's walk state current on that path, where
                 * the serial arm leaves it stale -- but the `break` below
                 * abandons the whole q either way, so no later slab reads it
                 * and the two arms stay behaviourally identical. Do not use
                 * this to turn that break into a continue: it would be correct
                 * HERE and wrong in the serial arm, which is exactly the kind
                 * of divergence output identity cannot catch. */
                int i1 = issue_side(1, xmax, j_base);
                int i0 = i1 ? -1 : issue_side(0, xmax, j_base);
                /* A failed ISSUE leaves launches in flight with nobody to
                 * await them, and `done:` below frees the bucket array they
                 * are writing into. Drain before unwinding. */
                if (i1 || i0) { mtlDeviceSynchronize(); rc = -1; break; }
                /* qpipe is never granted with --fill-concurrent: views only. */
                if (views_and_prep()) { rc = -1; break; }
                sv1 = pipe_side_sieve_join(1, &S1, &sf1, &sa1, concurrent);
                sv0 = pipe_side_sieve_join(0, &S0, &sf0, &sa0, concurrent);
                /* NOTHING collapses sv0 into sv1 here, and an earlier
                 * `if (sv1 > 0 && sv0 > 0) sv0 = 0;` was dead code: below,
                 * sv0 is read only by `sv1 < 0 || sv0 < 0` and
                 * `sv1 > 0 || sv0 > 0`, and with sv1 > 0 both already take the
                 * branch they would take anyway. Skipped slabs are counted per
                 * q, never per side.
                 *
                 * What DOES differ between the arms is stderr, not control
                 * flow: a slab where both sides overflow calls runlog_warn
                 * twice here and once in the serial arm, so the rate-limit
                 * budget of 8 is reached after 4 such slabs rather than 8 and
                 * the "further overflows summarised" line lands on a different
                 * slab. The end-of-band total is unaffected, and two lines
                 * naming two sides is arguably the more useful report -- but it
                 * is a real divergence that the relations-identity gate cannot
                 * see, so it is recorded rather than papered over. */
            } else {
                sv1 = issue_side(1, xmax, j_base);
                if (!sv1)
                    sv1 = pipe_side_sieve_join(1, &S1, &sf1, &sa1, concurrent);
                if (!sv1) {
                    /* --fill-gate: side 1 has been joined, so its bitmap for
                     * this slab is complete. Refused with --fill-concurrent,
                     * whose side 0 is issued before side 1 finishes. */
                    sv0 = issue_side(0, xmax, j_base);
                    if (!sv0 && views_and_prep()) {
                        mtlDeviceSynchronize();
                        rc = -1; break;
                    }
                    if (!sv0)
                        sv0 = pipe_side_sieve_join(0, &S0, &sf0, &sa0, concurrent);
                }
                /* The SAME hazard as the concurrent arm's, and it is NOT
                 * ruled out by joining side 1 before side 0 is issued: an
                 * issue that fails does so at one of the four API calls
                 * AFTER k_fill_atomic is already launched, so it returns -1
                 * with a kernel writing the bucket array and its join
                 * skipped. Then `done:` mtlFrees that array underneath it.
                 * -1 only; a soft skip (+1) has been through join and has
                 * already synchronised. */
                if (sv1 < 0 || sv0 < 0) mtlDeviceSynchronize();
            }
            if (sv1 < 0 || sv0 < 0) { rc = -1; break; }
            if (sv1 > 0 || sv0 > 0) {
                /* Skipped BEFORE the intersect/TD/emit below, so nothing from
                 * this slab reaches the relation file: never a wrong record.
                 *
                 * BREAK, NOT CONTINUE, and that is a correctness requirement
                 * rather than a policy choice. Under SLABBED the bottom of
                 * this loop advances the per-q continuation state -- the
                 * k_tdsmall_advance origin and both sides' walk_cur/walk_next
                 * swap -- and a `continue` jumps straight over it. Every
                 * LATER slab of this q would then sieve with the previous
                 * slab's walk positions against its own j_base: wrong log
                 * sums, wrong survivors, and a trial division one slab out of
                 * step, which can leave a composite in the cofactor and emit
                 * it as a large prime. Skipping a slab is a yield loss;
                 * carrying stale state into the next one is a bad relation.
                 *
                 * Advancing the state here instead does NOT work IN THE
                 * SERIAL ARM: side 0 is issued only under `if (!sv1)`, so on a
                 * side-1 soft skip its fill never ran and S0.walk_next holds
                 * the PREVIOUS slab's output -- there is nothing correct to
                 * swap in. Under --fill-concurrent side 0 IS issued and its
                 * state IS current, which is why the note at the concurrent
                 * arm forbids using that to turn this break into a continue:
                 * the repair would be right in one arm and wrong in the other.
                 * Abandoning the rest of the q is the cheap, obviously-correct
                 * repair in both; slabs already emitted keep their relations. */
                nslab_skipped   += slab_plan->nslab - slab;
                nslab_skipped_q += slab_plan->nslab - slab;
                break;
            }
            ts1[1] += sf1; ts1[2] += sa1;
            ts0[1] += sf0; ts0[2] += sa0;
            if (concurrent) {
                /* The overlap the two sides actually achieved, measured on one
                 * origin. Both side streams are released by the same
                 * default-stream fence, so WHICH ev[1] the GPU stamps first is
                 * launch noise, not the issue order -- an earlier version
                 * assumed side 1 always started first and then clamped each
                 * slab's result at zero, which keeps every over-estimate and
                 * discards the matching under-estimates. That is a biased
                 * estimator, and it biases the reported sieve stage LOW.
                 *
                 * Take S1.ev[1] as the origin and let d1 carry the sign: side 1
                 * spans [0, sf1+sa1], side 0 spans [d1, d1+sf0+sa0], and the
                 * pair spans from the earlier start to the later end. The one
                 * extra driver read is d1, which is genuinely new information;
                 * side 1's own end is sf1+sa1, already in hand from join. */
                /* Both ev[1] are complete here only because join() synced
                 * each side's ev[5], which is later in stream order -- a
                 * guarantee established in ANOTHER function, which is exactly
                 * what the ev[4] reads after the slab loop refuse to rely on.
                 * time_kernel discards mtlEventElapsedTime's status, so a
                 * not-ready event would report 0 ms AND latch mtlErrorNotReady
                 * for the NEXT slab's mtlGetLastError to report as a fatal
                 * failure of a healthy side. Both syncs are already satisfied,
                 * so they cost microseconds; do not remove them. */
                if (CUDA_CHECKED(mtlEventSynchronize(S1.ev[1])) ||
                    CUDA_CHECKED(mtlEventSynchronize(S0.ev[1]))) {
                    rc = -1; break;
                }
                const double d1 = time_kernel(S1.ev[1], S0.ev[1]);
                const double e1 = (double)sf1 + sa1;
                const double e0 = d1 + sf0 + sa0;
                const double span = (e1 > e0 ? e1 : e0) - (d1 < 0 ? d1 : 0);
                /* Signed, and accumulated signed: the clamp belongs once at
                 * print time, not per slab. */
                acc_ovl += (double)sf1 + sa1 + sf0 + sa0 - span;
            }
            side_surv1 += S1.nsurv; side_surv0 += S0.nsurv;

            wd_phase("slab.intersect");
            PIPE_CK(mtlMemset(d_n, 0, 4));
            PIPE_CK(mtlMemset(d_pre, 0, 8));
            PIPE_CK(mtlMemset(d_two, 0, (size_t)nbitword * 4));
            PIPE_CK(mtlEventRecord(ea));
            MTL_LAUNCH_NAMED((SLABBED ? "k_intersect_compact_1_1" : "k_intersect_compact_1_0"), blocks, cfg->threads, 0, 0, S1.survbits, S0.survbits, nbitword, cfg->logI, Lq.a0, Lq.a1, Lq.b0, Lq.b1, (int64_t)Lq.q, nullptr, nullptr, nullptr, 0u, d_n, d_pre, d_two, nullptr, j_base);
            PIPE_CK(mtlEventRecord(eb));
            /* The count by an async copy queued after eb (so the intersect
             * timing excludes it) and one wait on the legacy stream, where
             * the kernel and the copy run -- not a synchronous round trip
             * after the wait (finding 113). */
            PIPE_CK(mtlMemcpyAsync(&C.h_rb->isect, d_n, 4,
                                    mtlMemcpyDeviceToHost, 0));
            PIPE_CK(mtlStreamSynchronize(0));
            PIPE_CK(mtlGetLastError());
            tis += time_kernel(ea, eb);
            hn_s = C.h_rb->isect;

            /* ---- --q-pipeline issue point ----
             *
             * This slab's sieve and intersection are complete and nothing
             * reads its survbits again, so the next unit of work -- the next
             * slab of this q, or after the last slab the next q -- can be
             * sieved on st_sv while this slab is trial-divided on the legacy
             * stream. TD's view of each side (tdS1/tdS0) was taken before this
             * slab's last join (views_and_prep), before anything moved. A next
             * slab turns the walk cursors here (pside_rotate_walk); a next q
             * was already prepared there, which swapped plat/rowrt (its
             * k_transform would otherwise overwrite them under this q's
             * resieve), so here it is only issued.
             *
             * Everything the loop top checks for the next q -- stop, ceiling,
             * checkpoints -- still runs there, before any of its output is
             * consumed; a q issued here and then not processed is simply
             * drained at the loop exit and resumed from, never counted. */
            /* Called by pipe_td_perq once both sides' resieves are queued
             * (they are the last readers of this slab's bucket records), or
             * below if it returned before reaching them -- a slab with no
             * survivors, or a skip. The next unit's fill rewrites the bucket
             * arrays, so st_sv waits on the resieves' end (C.ev[14]) first. */
            int issue_done = 0;
            const std::function<int()> issue_next = [&]() -> int {
                if (issue_done) return 0;
                issue_done = 1;
                if (qpipe && (slab + 1 < slab_plan->nslab || next_prepared) &&
                    mtlStreamWaitEvent(st_sv, C.ev[14], 0) != mtlSuccess)
                    return -1;
                if (qpipe && slab + 1 < slab_plan->nslab) {
                    /* The next slab of this q. Its fill starts from this slab's
                     * output cursor; the cursor this slab started from stays with
                     * its TD (tdS1/tdS0 above), so the three turn. plat and rowrt
                     * belong to the whole q and are not touched. No stop or
                     * budget check: a q is never left half-sieved, single-queued
                     * or not. */
                    pside_rotate_walk(&S1);
                    pside_rotate_walk(&S0);
                    rotated = 1;
                    wd_phase("slab.issue_next");
                    if (issue_ahead(slab + 1)) return -1;
                    ahead = 1;
                    n_ahead_slab++;
                } else if (qpipe && next_prepared) {
                    /* The next q, after this q's last slab, prepared before its
                     * join (views_and_prep). Issued only if the stop flag is still
                     * clear: a q prepared and not issued is consumed at the loop
                     * top like any other, after the end-of-q drain below. */
                    if (next_ready && !g_pipe_stop) {
                        wd_q(next_wq, next_wrho, (unsigned long long)nqdone);
                        wd_phase("q.issue_next");
                        if (issue_ahead(0)) return -1;
                        wd_q((unsigned long long)cur->q, (unsigned long long)cur->rho,
                             (unsigned long long)nqdone);
                        spec_issued = 1;
                        n_ahead_q++;
                    }
                }
                return 0;
            };
            td_start = host_ms();
            tv_before = tv;
            {
                uint32_t n = 0;
                /* nqdone, NOT qi: these gates run on the first q actually
                 * SIEVED. Keyed on the loop index they were silently disabled
                 * for the whole band by a single skipped q at qi == 0 --
                 * --cofgate would then report success having never found any
                 * reference overlap, and --td-verify would reconstruct nothing
                 * while still reporting a pass. */
                const int do_verify = nqdone == 0 && cfg->td_verify && hn_s != 0 &&
                    (!td_verified || cfg->cofgate);
                wd_phase("slab.trialdiv");
                /* Whose bucket records survive to here: side 0's always in
                 * the serial arm (it filled last into the shared array), both
                 * when each side has its own (--fill-concurrent, or the
                 * serial arm's second array). */
                const pbkt_t *bkr[2] = {
                    cfg->bucket_resieve ? &bk0 : NULL,
                    cfg->bucket_resieve && (concurrent || dual) ? &bk1 : NULL };
                const int tdrc = pipe_td_perq<SLABBED>(&C, &Lq, cfg, &tdS1, &tdS0,
                                          d_two, xmax,
                                          j_base, blocks, cfg->threads, do_verify,
                                          &n, &nacc_s, &tm, &tv, want_host,
                                          !want_host, cofgate_found, bkr,
                                          &issue_next);
                if (tdrc < 0) { rc = -1; break; }
                /* TD returned before its resieves (no survivors) or skipped
                 * them: issue the next unit now. */
                if (issue_next()) { rc = -1; break; }
                if (tdrc > 0) {
                    /* Untrustworthy TD: drop THIS slab only and carry on.
                     *
                     * Unlike the bucket-overflow skip above, the continuation
                     * state here is intact and can simply be advanced. That
                     * skip has to abandon the whole q because in the serial arm
                     * side 0 is issued only under `if (!sv1)`, so its fill
                     * never ran and S0.walk_next holds the PREVIOUS slab's
                     * output -- and because the concurrent arm, where it DID
                     * run, must not diverge from it. Reaching HERE means
                     * sv1 == sv0 == 0: both k_fill_atomic launches completed
                     * and both walk_next buffers are current, so the ordinary
                     * end-of-loop advance is exactly right and the remaining
                     * slabs keep their yield.
                     *
                     * What must NOT happen is a bare `continue`, which would
                     * skip that advance and leave every later slab a step out
                     * of date against its own j_base. */
                    if (advance_slab(slab, J_here)) { rc = -1; break; }
                    ntd_skipped++;
                    nslab_skipped++;
                    nslab_skipped_q++;
                    continue;
                }
                if (nqdone == 0 && cfg->td_verify && !td_verified && hn_s != 0)
                    td_verified = 1;
                if (n != hn_s) {
                    fprintf(stderr,
                            "  q=%llu slab %u: intersect counted %u survivors,"
                            " rank scan %u\n",
                            (unsigned long long)cur->q, slab, hn_s, n);
                    rc = -1; break;
                }
                acc_td += host_ms() - td_start - (tv - tv_before);
            }
            hn += hn_s;
            nacc += nacc_s;

            cf_start = host_ms();
            /* A queue flush is safe inside a q, but it is NOT a new resume
             * point once an earlier slab of this q has been emitted. Resume
             * truncates to the previous whole-q checkpoint, so a crash still
             * replays the q atomically. */
            if (cfg->cofactor && nacc_s) {
                if (Q.n + nacc_s > Q.cap && cfg->cof_async) {
                    wd_phase("slab.cofq_flush_async");
                    if (cofa_wait(&A, &Q)) { rc = -1; break; }
                    A.ck_pending = fr && slab == 0;
                    if (A.ck_pending) {
                        A.ck_next = *cur;
                        A.ck_nq = base_nq + nqdone;
                    }
                    /* fc's offset is safe to take from this thread: under
                     * --cofactor (the only mode with a cofactor queue) the
                     * join loop writes no candidates, so it does not move. */
                    A.ck_P = POLY; A.ck = &ck; A.ck_fc = fc;
                    A.ck_base_rel = base_rel;
                    A.ck_written = &ckpt_written; A.ck_warned = &ckpt_warned;
                    cofa_start(&A, &Q, &QO, cfg, blocks, fr);
                } else if (Q.n + nacc_s > Q.cap) {
                    wd_phase("slab.cofq_flush");
                    if (cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0,
                                   cfg->lim, cfg->lpb, (int)Q.ecm_rounds,
                                   cfg->cof_budget, cfg->cof_chunk,
                                   blocks, cfg->threads, fr)) {
                        rc = -1; break;
                    }
                    if (fr && slab == 0)
                        pipe_try_checkpoint(POLY, cfg, &ck, fr, fc, cur,
                                            base_rel + Q.nrel,
                                            base_nq + nqdone,
                                            &ckpt_written, &ckpt_warned);
                }
                if (nacc_s > Q.cap) {
                    runlog_warn("  q=%llu slab %u: %u candidates exceeds the"
                                " %u-slot cofactor queue",
                                (unsigned long long)cur->q, slab, nacc_s, Q.cap);
                    rc = -1; break;
                }
                wd_phase("slab.cof_enqueue");
                if (cof_enqueue(blocks, cfg->threads,
                                C.d_ccof[0], C.d_cbits[0],
                                C.d_ccof[1], C.d_cbits[1], C.d_ca, C.d_cb,
                                C.d_cfac[0], C.d_cfn[0],
                                C.d_cfac[1], C.d_cfn[1],
                                nacc_s, Q.n, cfg->lpb0, cfg->lpb, &Q)) {
                    rc = -1; break;
                }
                if (CUDA_CHECKED(pipe_qsync()) ||
                    CUDA_CHECKED(mtlGetLastError())) { rc = -1; break; }
                Q.n += nacc_s;
                /* --cof-async: a flush that has finished (and checkpointed
                 * itself) is joined HERE rather than at the next queue-full,
                 * so --target-rels sees its relations a whole flush interval
                 * sooner. The join cannot wait: the thread has returned. */
                if (A.busy && A.fin->load(std::memory_order_acquire))
                    if (cofa_wait(&A, &Q)) { rc = -1; break; }
            }
            tm.cofac += host_ms() - cf_start;

            join_start = host_ms();
            wd_phase("slab.join");
            for (uint32_t k = 0; want_host && k < nacc_s; k++) {
                int64_t a = C.h_ca[k], b = C.h_cb[k];
                const uint32_t c0 = C.h_cfn[0][k], c1 = C.h_cfn[1][k];
                const uint32_t *f0 = C.h_cfac[0] + (size_t)k * TD_FMAX;
                const uint32_t *f1 = C.h_cfac[1] + (size_t)k * TD_FMAX;
                const int b0 = C.h_cbits[0][k], b1 = C.h_cbits[1][k];
                char buf[BN_DEC_MAX];
                if (c0 > TD_FMAX || c1 > TD_FMAX) {
                    fprintf(stderr, "  q=%llu: a candidate has %u/%u factors,"
                            " more than the %d recorded; raise TD_FMAX\n",
                            (unsigned long long)cur->q, c0, c1, TD_FMAX);
                    rc = -1; break;
                }
                std::sort((uint32_t *)f0, (uint32_t *)f0 + c0);
                std::sort((uint32_t *)f1, (uint32_t *)f1 + c1);
                if (b < 0) { a = -a; b = -b; }
                if (b0 <= (int)cfg->lpb0 && b1 <= (int)cfg->lpb) {
                    nrel++;
                    if (fr && !cfg->cofactor) {
                        fprintf(fr, "%lld,%lld:", (long long)a, (long long)b);
                        for (uint32_t z = 0; z < c0; z++)
                            fprintf(fr, "%s%x", z ? "," : "", f0[z]);
                        if (b0 > 1)
                            fprintf(fr, "%s%llx", c0 ? "," : "",
                                    (unsigned long long)strtoull(
                                        bn_to_dec(&C.h_ccof[0][k], buf), NULL, 10));
                        fputc(':', fr);
                        for (uint32_t z = 0; z < c1; z++)
                            fprintf(fr, "%s%x", z ? "," : "", f1[z]);
                        if (b1 > 1)
                            fprintf(fr, "%s%llx", c1 ? "," : "",
                                    (unsigned long long)strtoull(
                                        bn_to_dec(&C.h_ccof[1][k], buf), NULL, 10));
                        fputc('\n', fr);
                    }
                    continue;
                }
                ncand++;
                if (cfg->cofactor) continue;
                if (fc) {
                    fprintf(fc, "%s%s,", b0 > (int)cfg->lpb0 ? "-" : "",
                            bn_to_dec(&C.h_ccof[0][k], buf));
                    fprintf(fc, "%s%s:%lld,%lld:",
                            b1 > (int)cfg->lpb ? "-" : "",
                            bn_to_dec(&C.h_ccof[1][k], buf),
                            (long long)a, (long long)b);
                    for (uint32_t z = 0; z < c0; z++)
                        fprintf(fc, "%s%x", z ? "," : "", f0[z]);
                    fputc(':', fc);
                    for (uint32_t z = 0; z < c1; z++)
                        fprintf(fc, "%s%x", z ? "," : "", f1[z]);
                    fputc('\n', fc);
                }
            }
            tm.join += host_ms() - join_start;
            if (rc) break;

            if (advance_slab(slab, J_here)) { rc = -1; break; }
        }
        /* A next q prepared during this q's last slab and NOT issued -- the
         * stop flag rose, this q was abandoned on a soft skip, or the next
         * q skips -- still has its table uploads in flight from the pinned
         * staging buffers, and the loop top is about to prepare it again
         * into the same buffers. ev[4] is recorded after those uploads:
         * wait for it. This is the fourth drain pipe_side_prepare_q's
         * comment lists. (Finding 114's review: the soft-skip case is not
         * reachable by any gate, as degradecheck.sh explains.) */
        if (next_prepared && !spec_issued) {
            const mtlError_t e1 = mtlEventSynchronize(S1.ev[4]);
            const mtlError_t e0 = mtlEventSynchronize(S0.ev[4]);
            if (!rc && (e1 != mtlSuccess || e0 != mtlSuccess)) {
                CUDA_CHECKED(e1 != mtlSuccess ? e1 : e0);
                rc = -1;
            }
        }
        if (cfg->qspan) { slab_q   = host_ms() - hreg; hreg = host_ms(); }
        if (rc) {
            /* Same rule as the nq_lost rollback below, on the other exit. This
             * q's fill and apply times live in ts1/ts0 and are never banked
             * into acc_fi/acc_ap, but acc_ovl was banked PER SLAB inside the
             * loop -- and the band summary still prints after a hard failure
             * (`if (nqdone)` below), so leaving it in would subtract this dead
             * q's overlap from a sieve total made only of the q that did
             * complete. Un-reconcilable stage total, finding 94's rule.
             *
             * ALL THREE, not just acc_ovl. An earlier version of this rollback
             * restored the accumulator that prompted it and left acc_td and tm
             * charged -- acc_td is banked per slab and tm.join/tm.td/tm.rank
             * inside the slab loop and pipe_td_perq, so a q that dies on slab 3
             * of 4 leaves all of them holding work that nqdone never counts.
             * `TD + classify, wall` and `= device total` then print divided by
             * an N that excludes the q, and `unaccounted` can go negative: the
             * same defect, one accumulator over. The rule is the rule the
             * comment states -- every band-level accumulator written inside the
             * slab loop is restored here -- and it is not satisfied by
             * restoring whichever one was noticed. */
            PIPE_Q_ROLLBACK();
            break;
        }
        /* Deferred from pipe_side_prepare_q, which no longer blocks on the
         * transform. Same events, same subtraction, same reported number --
         * only the moment of reading moved.
         *
         * This USED to argue that the slab loop above always ended with a
         * mtlEventSynchronize on each side's ev[3], so ev[4] (earlier in
         * stream order) was complete for free. That argument is DEAD: a slab
         * that soft-skips on side 1 short-circuits side 0 entirely
         * (the serial arm joins side 1 before side 0 is issued at all), so
         * S0.ev[3] is not recorded for that slab, and a q whose every slab
         * skips that way reaches here having never synchronised side 0.
         * --fill-concurrent does issue both, but that arm must not be the
         * reason this is safe either -- the flag is off by default. The explicit
         * ev[4] synchronises below are what make this safe now -- they are
         * load-bearing, not the belt-and-braces the next comment calls them.
         * Do not reinstate the ev[3] reasoning. */
        /* On the ordinary path the slab loop's ev[5] has already covered
         * these, so they cost microseconds; on the all-slabs-skipped path
         * above they are the ONLY thing that drains side 0. Either way they
         * make the read structurally safe rather than dependent on a
         * guarantee established in another function. time_kernel discards
         * mtlEventElapsedTime's status (see time_kernel in
         * bench_kernels.cu), so a
         * not-ready event would silently report 0 ms AND latch
         * mtlErrorNotReady for the next unrelated mtlGetLastError to
         * report as fatal. Do not remove these to save the microseconds. */
        /* --q-pipeline read them when it prepared the next q */
        if (!ts_read && pipe_read_transform(&S1, &S0, &ts1[0], &ts0[0])) {
            rc = -1; goto done;
        }
        /* This is the original pre-slabbing invariant: an entire special-q
         * with no two-sided survivors is suspicious and remains fatal. Empty
         * individual slabs are allowed; only their sum is tested here. */
        if (!hn) {
            /* Still fatal when the sieve actually RAN: a fully-sieved q with
             * no two-sided survivor means something is wrong, and that
             * invariant predates slabbing. But if slabs were soft-skipped
             * above, zero is explained rather than suspicious -- do not
             * convert a bucket overflow into a dead task by the back door. */
            if (!nslab_skipped_q) {
                fprintf(stderr, "  pipeline: no survivors at this q\n");
                PIPE_Q_ROLLBACK();
                rc = -1; break;
            }
            nq_lost++;
            /* Roll back the accumulators this q already charged. The others
             * are either held in a local above (ts1/ts0, so acc_fi and acc_ap
             * never see this q) or added after this point.
             *
             * acc_ovl BELONGS HERE and was missed when it was added: it is
             * banked per slab inside the loop, while the fill and apply times
             * it corrects are banked per q below and are dropped on this path.
             * Left in, a q that reached here would subtract an overlap from a
             * sieve total that was never charged the matching per-side time --
             * understating `sieve, both sides` and overstating `unaccounted`,
             * which is the un-reconcilable stage total finding 94 exists to
             * forbid. Any future band-level accumulator written inside the
             * slab loop has to be added here too. */
            PIPE_Q_ROLLBACK();
            continue;              /* next q; nqdone is not incremented */
        }
        if (nqdone == 0 && cfg->cofgate &&
            (!cofgate_found[0] || !cofgate_found[1])) {
            fprintf(stderr,
                    "  trial-division cofactor gate found no reference overlap"
                    " on side%s%s across the complete q\n",
                    cofgate_found[0] ? "" : " 0",
                    cofgate_found[1] ? "" : " 1");
            PIPE_Q_ROLLBACK();
            rc = -1; break;
        }

        if (cfg->verbose_q) {
            printf("  side 1: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
                   " %8llu survivors (bound %u)%s\n",
                   ts1[0], ts1[1], ts1[2], ts1[0] + ts1[1] + ts1[2],
                   (unsigned long long)side_surv1, S1.BOUND,
                   SLABBED ? " [slabbed]" : "");
            /* Gated, side 0's own count is not its survivor population: its
             * fill skips every position side 1 rejected. */
            printf("  side 0: transform %.3f + fill %.3f + apply %.3f = %7.3f ms,"
                   " %8llu survivors (bound %u)%s%s\n",
                   ts0[0], ts0[1], ts0[2], ts0[0] + ts0[1] + ts0[2],
                   (unsigned long long)side_surv0, S0.BOUND,
                   cfg->fill_gate ? " [gated by side 1]" : "",
                   SLABBED ? " [slabbed]" : "");
        }

        t_verify += tv;
        acc_tr += ts1[0] + ts0[0];
        acc_fi += ts1[1] + ts0[1];
        acc_ap += ts1[2] + ts0[2];
        if (cfg->qspan) {
            /* Every readback in this q has already synchronised, so this is a
             * no-op drain; it is here so the elapsed read below cannot see a
             * not-ready event (time_kernel discards the status). */
            acc_tail += host_ms() - hreg;   /* already inside if (qspan) */
            PIPE_CK(mtlEventRecord(qspan1));
            acc_hspan += host_ms() - hspan0;   /* host clock, before the sync */
            /* Only now, on a q that actually completed -- every other region
             * timer accumulates here too, and N is nqdone. Charging a SKIPPED
             * q's prologue would break the reconciliation silently. */
            acc_prologue += prologue_q;
            acc_prep     += prep_q;
            acc_tdprep   += tdprep_q;
            acc_slab     += slab_q;
            PIPE_CK(mtlEventSynchronize(qspan1));
            acc_qspan += time_kernel(qspan0, qspan1);
        }
        acc_isect += tis; acc_host += th1 + th0;
        acc_wall += host_ms() - qwall - tv;
        acc_surv += hn; acc_cand += ncand; acc_rel += nrel;   /* host path only */
        nqdone++;
        norm_verbose = 0;      /* the first q's setup is printed; the rest are not */
#ifdef HAVE_BOINC
        {
            /* base_rel, exactly as the --target-rels stop test does it a
             * few lines below: without it a resumed run reports progress
             * towards the target as though it had produced nothing. */
            const unsigned long long progress_rels = base_rel + (cfg->cofactor
                ? Q.nrel : (unsigned long long)acc_rel);
            const double now = host_ms();
            const int terminal_known =
                (!qgen && nqdone == nq) ||
                (cfg->target_rels && progress_rels >= cfg->target_rels);
            if (now - t_boinc_report >= 1000.0 || terminal_known) {
                double fraction = pipe_progress_fraction(
                    cfg, qgen != NULL, nq, nqdone, cur->q, progress_rels,
                    base_nq);
                /* Leave the final one percent for the after-band cofactor flush,
                 * output close/rename, cleanup, and bench_boinc_finish(). A
                 * successful workunit reports exactly 1.0 only after main has
                 * completed successfully. */
                if (fraction > 0.99) fraction = 0.99;
                pipe_report_fraction(fraction);
                t_boinc_report = now;
            }
        }
#endif
        /* --target-rels: stop once enough relations exist. Under inline
         * cofactorisation the only running total the host has without a per-q
         * readback is Q.nrel, which advances at FLUSH boundaries -- so this
         * overshoots by at most one flush (~256 q here; two under
         * --cof-async, where Q.nrel also lags by the flush still in flight),
         * well under 1% of any useful target. Sieving upward until satisfied
         * is what you actually want when the question is "enough for the matrix", since the yield
         * per q falls as q grows and guessing the range wastes either time or
         * relations. */
        if (cfg->target_rels) {
            /* base_rel is what earlier sessions already wrote. Counting only
             * this session's would make a resumed run sieve the whole target
             * again from scratch. */
            unsigned long long have = base_rel + (cfg->cofactor
                                                  ? Q.nrel
                                                  : (unsigned long long)acc_rel);
            if (have >= cfg->target_rels) {
                printf("\n  --target-rels %llu reached after %llu q (%llu relations)\n",
                       (unsigned long long)cfg->target_rels,
                       base_nq + nqdone, have);
                target_reached = 1;
                break;
            }
        }
        /* ncand and nrel are produced by the host join loop, which does not run
         * under inline cofactorisation -- so printing them there reported a
         * flat "0 joint candidates, 0 relations" for every q of the band. The
         * progress line below was fixed for this; this branch was not, and it
         * is the one somebody passes --verbose-q to read. Report what is
         * actually known per q rather than a variable that stayed at its
         * initialiser; the device counters are folded in after the band. */
        if (cfg->verbose_q) {
            if (want_host)
                printf("  q=%llu: %u survivors, %u joint candidates,"
                       " %u relations\n",
                       (unsigned long long)cur->q, hn, ncand, nrel);
            else
                printf("  q=%llu: %u survivors, %u joint candidates"
                       " (relations counted on the device; see band summary)\n",
                       (unsigned long long)cur->q, hn, nacc);
        }
        /* Under inline cofactorisation the host has no per-q relation count --
         * the counters accumulate on the device and Q.nrel only advances at a
         * flush. Reporting acc_rel there showed a flat 0 for the whole run,
         * which on a multi-hour band looks exactly like failure.
         *
         * Q.nrel is not enough on its own either. It advances at a FLUSH, and a
         * flush is 131,072 candidates -- ~67 q on the c183 this was tuned
         * against, but 686 q on the SNFS job, which enqueues 191 records per q
         * instead of 1,956. That is 41 s of "0 rel" before the counter can
         * move at all, and `--relations` stages to NAME.part until the band
         * ends, so `ls` agrees with it. A run that was working normally, with
         * 63,000 candidates queued that went on to yield 39,710 relations, read
         * as a dead run. So carry the queue occupancy alongside -- it advances
         * every q, on every job -- and label it as candidates, which is what it
         * counts.
         *
         * Progress is reported against whichever goal is actually in force: a
         * relation target if one was given, otherwise an explicit --nq count,
         * generated numeric q interval, or list length. Rate-limited to one
         * update per report_ms -- 30 s on a terminal, where it is a single \r
         * line and a band runs for hours, and the log's own period when stdout
         * is redirected and each update costs a line.
         */
        /* Two consumers on two clocks: the console line, and the run log,
         * which is due on its own period and carries the four numbers that say
         * whether the progress ones can be compared to anything (runlog.h,
         * finding 53). Both predicates are evaluated every q -- deliberately,
         * rather than short-circuiting one against the other, which would skip
         * the log's clock on exactly the q where both came due and post the
         * record a special-q late. The shared arithmetic below is done once
         * for whichever is asking. */
        const int console_due = !cfg->verbose_q &&
            (host_ms() - t_report > report_ms || (!qgen && qi + 1 == nq));
        const int log_due = runlog_due();
        if (console_due || log_due) {
            /* Two different numbers, and mixing them is how a resumed run
             * reports a nonsense rate: the goal is measured against everything
             * on disk, the RATE only against what this session produced in the
             * time this session has been running. */
            const unsigned long long mine = cfg->cofactor
                ? Q.nrel : (unsigned long long)acc_rel;
            const unsigned long long rels = base_rel + mine;
            const double el = (host_ms() - t_band) / 1000.0;
            const double rps = el > 0 ? mine / el : 0.0;
            double frac, eta;
            if (console_due) t_report = host_ms();
            frac = pipe_progress_fraction(cfg, qgen != NULL, nq, nqdone,
                                             cur->q, rels, base_nq);
            if (cfg->target_rels)
                eta = rps > 0 ? ((double)cfg->target_rels - rels) / rps : 0.0;
            else
                eta = frac > 0 ? el * (1.0 / frac - 1.0) : 0.0;
            if (eta < 0.0) eta = 0.0;
            if (console_due) {
                /* Q.n counts queued CANDIDATE records, not relations -- only
                 * ~2/3 of them survive splitting on this job, and the band
                 * summary keeps nseen and nrel apart for that reason. Labelled
                 * "cand" so the line cannot be read as pending relations.
                 *
                 * Shown only while the queue holds work the relation count
                 * cannot see yet, so a steady-state line stays as it was --
                 * which means the line CHANGES WIDTH when cofq_flush resets
                 * Q.n to 0. A \r without an erase leaves the tail of the longer
                 * line on screen ("...ETA 0h 12m  cand"), so clear to
                 * end-of-line rather than trusting a trailing pad to cover the
                 * widest case. */
                char queued[32] = "";
                /* The erase-to-end-of-line and the carriage return are a
                 * terminal's, not a file's: redirected, they make one line
                 * carrying an ANSI escape per update. A redirect gets plain
                 * newline-terminated lines instead. */
                const char *end = reporting_to_tty ? "\033[K\r" : "\n";
                if (cfg->cofactor && Q.n)
                    snprintf(queued, sizeof queued, " +%u cand", Q.n);
                /* rps == 0 means "no relation has flushed yet", not "done".
                 * eta is 0 there as a sentinel, and printing it as 0h 00m says
                 * the band is finishing at the exact moment nothing has been
                 * counted -- which is the reading this line exists to prevent. */
                if (rps > 0.0 && isfinite(eta)) {
                    /* Do not cast the complete ETA to an int: immediately
                     * after the first cofactor flush it can legitimately be
                     * millions of hours. Keep the unbounded hour count as a
                     * double and cast only the modulo-60 minute component. */
                    const double eta_hours = floor(eta / 3600.0);
                    const int eta_minutes =
                        (int)fmod(floor(eta / 60.0), 60.0);
                    printf("    %sq=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA %.0fh %02dm%s",
                           ipfx, (unsigned long long)cur->q, nqdone, rels, queued,
                           rps, 100.0 * frac, eta_hours, eta_minutes, end);
                } else if (rps > 0.0) {
                    printf("    %sq=%llu  %u q  %llu rel%s  %.0f rel/s  %.1f%%"
                           "  ETA inf%s",
                           ipfx, (unsigned long long)cur->q, nqdone, rels, queued,
                           rps, 100.0 * frac, end);
                } else {
                    printf("    %sq=%llu  %u q  %llu rel%s  -- rel/s  %.1f%%"
                           "  ETA --h --m%s",
                           ipfx, (unsigned long long)cur->q, nqdone, rels, queued,
                           100.0 * frac, end);
                }
            }
            if (log_due) {
                /* The same progress numbers, plus what makes them auditable.
                 *
                 * GPU-accounted/wall is the running form of the band summary's
                 * line, and identical to it in construction: the cofactor
                 * queue is out of BOTH terms because its flush is host-timed
                 * and mixes device and host work, and leaving it in the
                 * denominator alone would make the ratio move with survivor
                 * density -- a dense band would read as a contended host on an
                 * idle box. Compare it against your own idle baseline on the
                 * same card, job and band length; an absolute reading is not
                 * meaningful (finding 53).
                 *
                 * Utilisation, board watts and load average are sampled here
                 * rather than averaged: this is a spot check that says whether
                 * the box was busy, not a power measurement. The metric of
                 * record is whole-box watts from a meter (STATUS.md item 6),
                 * and a board sensor cannot be promoted to it. */
                const double devsum = tm.rank + tm.emit + tm.summary
                    + tm.resieve + tm.td + tm.classify + tm.compact + tm.record;
                const double denom = acc_wall - tm.cofac;
                const double accwall = denom > 0.0
                    ? (acc_tr + acc_fi + acc_ap - pipe_ovl_credit(acc_ovl)
                       + acc_isect + devsum) / denom
                    : 0.0;
                /* Two q in flight: the next q's sieve time overlaps this q's
                 * host-clocked TD, so the stage sum is no measure of how busy
                 * the card was and can pass 1.0. No credit is computable from
                 * these timers, so say so rather than print a number. */
                char accwall_s[32];
                if (qpipe) snprintf(accwall_s, sizeof accwall_s, "acc/wall=n/a");
                else snprintf(accwall_s, sizeof accwall_s, "acc/wall=%.3f", accwall);
                char gpu[48] = "gpu=n/a", pwr[32] = "board=n/a";
                char eta_s[32] = "eta=--";
                double load[3] = {0.0, 0.0, 0.0};
                unsigned int util = 0;
                double watts = 0.0;
                if (runlog_gpu_util(&util) == 0)
                    snprintf(gpu, sizeof gpu, "gpu=%u%%", util);
                if (runlog_gpu_watts(&watts) == 0)
                    snprintf(pwr, sizeof pwr, "board=%.1fW", watts);
#ifdef _WIN32
                /* Unix load average has no direct Win32/MSVC equivalent.
                 * Keep the status field but mark it unavailable on Windows. */
                load[0] = -1.0;
#else
                if (getloadavg(load, 3) < 1) load[0] = -1.0;
#endif
                if (rps > 0.0 && isfinite(eta))
                    snprintf(eta_s, sizeof eta_s, "eta=%.2fh", eta / 3600.0);
                runlog_record(
                    "q=%llu nq=%llu rel=%llu cand=%u rel/s=%.1f pct=%.2f %s"
                    " ms/q=%.2f %s %s %s load=%.2f",
                    (unsigned long long)cur->q,
                    base_nq + (unsigned long long)nqdone, rels,
                    cfg->cofactor ? Q.n : 0u, rps, 100.0 * frac, eta_s,
                    nqdone ? acc_wall / nqdone : 0.0, accwall_s, gpu, pwr,
                    load[0]);
            }
        }
        fflush(stdout);
    }

    pg.lock();
    if (g_pinst)
        printf("\n=== instance %d of %d: end of band ===\n", g_pinst->k,
               g_pinst->n);
    /* A streamed source discovers its end only when the NEXT pull returns 0,
     * so the in-loop 30-second reporter cannot know that the preceding q was
     * terminal. Give it an explicit terminal line instead of leaving the last
     * visible progress sample up to 30 seconds stale. */
    if (!cfg->verbose_q && qgen && !rc && nqdone &&
        (source_exhausted || count_limit_reached)) {
        const unsigned long long rels = base_rel + (cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel);
        char queued[32] = "";
        if (cfg->cofactor && Q.n)
            snprintf(queued, sizeof queued, " +%u cand", Q.n);
        /* The erase clears the tail of the \r line this one replaces, and the
         * leading newline breaks away from it. Both are a terminal's, not a
         * file's: redirected, the progress line above already ended in a
         * newline, so the lead would only add a blank line. */
        const char *erase = reporting_to_tty ? "\033[K" : "";
        const char *lead = reporting_to_tty ? "\n" : "";
        if (count_limit_reached)
            /* base_nq + nq_max reconstructs what the operator actually typed:
             * a resumed run holds only the REMAINING budget in cfg->nq_max, so
             * echoing it raw prints a number nobody passed, next to a q count
             * that then disagrees with it. */
            printf("%s    q=%llu  %llu q  %llu rel%s  [--nq %llu reached]%s\n",
                   lead, (unsigned long long)last_q.q, base_nq + nqdone, rels,
                   queued, base_nq + cfg->nq_max, erase);
        else
            printf("%s    q=%llu  %llu q  %llu rel%s  [q range exhausted]%s\n",
                   lead, (unsigned long long)last_q.q, base_nq + nqdone, rels,
                   queued, erase);
    }

#ifdef HAVE_BOINC
    if (!rc && nqdone) {
        const unsigned long long rels = base_rel + (cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel);
        double fraction = pipe_progress_fraction(
            cfg, qgen != NULL, nq, nqdone, last_q.q, rels, base_nq);
        if (fraction > 0.99) fraction = 0.99;
        pipe_report_fraction(fraction);
    }
#endif

    }   /* t_band scope */

    /* --q-pipeline: a q issued ahead and never consumed (the band stopped,
     * ran out, or reached its target) is drained here, before anything below
     * reads results. Its relations are never used, but a device fault in it
     * is still a fault, and must not vanish at `done:`. */
    if (st_sv && CUDA_CHECKED(mtlStreamSynchronize(st_sv)) && rc == 0)
        rc = -1;

    /* Fold in the device-side candidate counters. Under inline cofactorisation
     * the host loop never ran, so acc_cand and acc_rel are still zero and these
     * are the only counts there are. The overflow check lands here rather than
     * per q: the alternative was a blocking readback every q, which cost more
     * than the host loop it replaced. A truncated factor list still fails the
     * run -- just at the end of the band rather than inside it. */
    if (!rc && cfg->cofactor && !fc) {      /* exactly !want_host */
        uint32_t cs[3] = {0, 0, 0};
        if (CUDA_CHECKED(mtlMemcpy(cs, C.d_stats, 12,
                                    mtlMemcpyDeviceToHost)))
            rc = -1;
        else {
            acc_rel += cs[0]; acc_cand += cs[1];
            if (cs[2]) {
                runlog_warn("  %u candidates in this band have more than the"
                            " %d recorded factors; raise TD_FMAX", cs[2],
                            TD_FMAX);
                rc = -1;
            }
        }
    }

    /* The last partial flush happens AFTER the band loop, so its cost is not
     * in acc_wall. On a band shorter than one flush that is the ENTIRE
     * cofactorisation, which is why it is carried separately and added back
     * rather than folded into the per-q average. */
    {
        /* Joined whatever rc is: the thread writes fr, which done: closes. */
        const double cfw = host_ms();
        if (cofa_wait(&A, &Q) && rc == 0) rc = -1;
        cofac_tail += host_ms() - cfw;
    }
    if (rc == 0 && cfg->cofactor && Q.n) {
        const double cf0 = host_ms();
        wd_phase("band.final_cofq_flush");
        if (cofq_flush(&Q, &QO, cfg->lim0, cfg->lpb0, cfg->lim, cfg->lpb,
                       (int)Q.ecm_rounds, cfg->cof_budget, cfg->cof_chunk, blocks, cfg->threads, fr))
            rc = -1;
        cofac_tail += host_ms() - cf0;
    }
    if (rc == 0 && !stopped && cfg->target_rels && !target_reached) {
        const unsigned long long have = base_rel + (cfg->cofactor
            ? Q.nrel : (unsigned long long)acc_rel);
        if (have >= cfg->target_rels) {
            printf("\n  --target-rels %llu reached by the final cofactor flush"
                   " after %llu q (%llu relations)\n",
                   (unsigned long long)cfg->target_rels, base_nq + nqdone, have);
            target_reached = 1;
        } else if (count_limit_reached) {
            runlog_warn(
                    "\n  note: --nq %llu reached after %llu q with %llu relations;"
                    " --target-rels %llu was not reached",
                    base_nq + cfg->nq_max, base_nq + nqdone, have,
                    (unsigned long long)cfg->target_rels);
        } else if (source_exhausted) {
            runlog_warn(
                    "\n  WARNING: special-q source exhausted after %llu q with"
                    " %llu relations; --target-rels %llu was NOT reached",
                    base_nq + nqdone, have,
                    (unsigned long long)cfg->target_rels);
        }
    }
    {
        /* A clean stop is not a completed band: the .part and its sidecar stay,
         * and the final name is not claimed. Renaming here would present a
         * partial band as a finished one and strand the rest of the q range. */
        const int commit = (rc == 0 && !stopped);
        if (pipe_finalize_outputs(&fr, &fc, cfg, rtmp, ctmp, commit,
                                  ckpt_written))
            rc = -1;
        outputs_finalized = 1;
        if (commit && rc == 0 && ckpt_armed) {
            char cp[CKPT_PATH_MAX], rp[CKPT_PATH_MAX], rtp[CKPT_PATH_MAX];
            /* Cannot fail here: the same name was built for every checkpoint
             * this band wrote, into a buffer of the same size. Checked anyway
             * because the band is already committed -- a stale sidecar is a
             * better outcome than an ignored return value. */
            if (ckpt_ckpt_path(cfg->relations, cp, sizeof cp) == 0)
                remove(cp);      /* nothing left to resume */
            /* A BOINC recovery budget spans the whole workunit, not merely one
             * checkpoint interval. Clear it only after the final output has
             * committed; otherwise a deterministic resume defect could write
             * one good checkpoint per attempt and restart forever. */
            if (ckpt_recovery_path(cfg->relations, rp, sizeof rp) == 0)
                remove(rp);
            if (ckpt_recovery_tmp_path(cfg->relations, rtp, sizeof rtp) == 0)
                remove(rtp);
        }
    }
#ifdef HAVE_BOINC
    /* !stopped, NOT just rc == 0. rc is still 0 here for a stopped or capped
     * band -- the PIPE_RC_ mapping runs at `done:` -- so this reported 0.99 for
     * a band that took a signal after 30 of 1000 q. The client would show 99%,
     * then the restarted session reports 3% and the progress bar and the
     * remaining-time estimate both jump backwards. The 0.99 means "the band is
     * finished and only the flush, rename and cleanup remain"; a stop has the
     * whole band left. */
    if (rc == 0 && !stopped) pipe_report_fraction(0.99);
#endif
    if (rc == 0) {
        /* Query only after the scientific output is committed. Per-q buffers
         * grow on demand, so this remains useful telemetry, but its failure is
         * never allowed to change a completed workunit into an error. */
        size_t fb_ = 0, tb_ = 0;
        if (pipe_mem_info_optional("steady-state report", &fb_, &tb_) == 0) {
            printf("\n  device memory, steady state: %.2f GB in use of %.2f GB"
                   " (%.2f GB free)\n",
                   (tb_ - fb_) / 1073741824.0, tb_ / 1073741824.0,
                   fb_ / 1073741824.0);
        }
    }
    if (rc) {
        const char *fate = ckpt_written
            ? "the .part is kept; rerun the same command to resume"
            : "nothing was checkpointed, so no output is kept";
        /* Into the log as well as stderr. A band that dies on its FIRST q
         * never reaches the summary below -- that whole block is guarded on
         * nqdone -- so without this the log ends with a header, possibly a few
         * heartbeats, and no indication that anything went wrong. Read a week
         * later, or under BOINC where stderr is a separate upload, that is
         * indistinguishable from a run still in progress. */
        if (qgen)
            runlog_warn("  band FAILED after %u generated q; %s",
                        nqdone, fate);
        else
            runlog_warn("  band FAILED after %u of %u q; %s",
                        nqdone, nq, fate);
    }
    /* THE CHAIN BELOW TESTS cfg->relations, NOT fr, AND MUST.
     *
     * pipe_finalize_outputs above closes fr and sets it to NULL, so every `fr`
     * here is null by construction no matter what the run was asked to do.
     * Written as `!fr` -- which is what it said -- the "no relation file
     * requested" arms matched ALWAYS, printing "nothing to resume" at a run
     * that had just written a perfectly good .part, and shadowing the
     * `ckpt_written` arms behind them into dead code: the whole "FIX THE JOB
     * BEFORE RESUMING" warning could never print. degradecheck.sh case A is
     * what caught it.
     *
     * cfg->relations is the honest predicate: it is const config, it survives
     * finalize, and fr is opened if and only if it is set (a failed open
     * goto done's with rc = -1 rather than continuing with fr null). The
     * in-loop `if (fr)` guards are a different question and stay as they are
     * -- fr is live there. */
    if (stopped && ckpt_written && capped)
        /* runlog_warn, NOT printf: this is the operator-facing half of an
         * exit-3 result, and under BOINC stdout is not uploaded -- stderr.txt
         * is all a project gets back. The width already went to stderr at the
         * cap; the resume point and the "do not just rerun it" warning were
         * going to stdout, where a project triaging the failure cannot see
         * them.
         *
         * Same checkpoint, opposite advice. "Rerun the same command" is what
         * the ordinary stop wants and it is exactly wrong here: finding 93's
         * case D measured a resumed capped band advancing ~PIPE_SKIP_MAX q and
         * emitting nothing, because nqskip is not checkpointed. The resume
         * point is still printed -- it is real, and a WIDER build should start
         * from it. The phrase "resume at q=" is load-bearing for skipcheck.sh,
         * which reads the point back out of this line. */
        runlog_warn("\n  stopped after %u q this session (%llu total, %llu"
                    " relations); checkpoint written, resume at q=%llu."
                    " REBUILD BEFORE RESUMING: this binary hits the same cap"
                    " about %d skips into any rerun and emits nothing.",
                    nqdone, base_nq + nqdone, ck.nrel,
                    (unsigned long long)ck.next_q, PIPE_SKIP_MAX);
    else if (capped && !cfg->relations)
        /* The counterpart of `degraded && !cfg->relations` below, and it
         * exists for the same reason: without it this falls through to the
         * "NO checkpoint
         * could be written" error, which tells the operator to move a .part
         * aside or pass --restart over a file that was never requested. The
         * remedy differs from the degraded one -- the job is fine, the BUILD
         * is too narrow -- and the width to rebuild at already went to stderr
         * at the cap itself. */
        runlog_warn("\n  stopped after %u q with no relation file requested;"
                    " nothing to resume. The timings from this band are NOT"
                    " usable -- it skipped its way to the norm-width cap."
                    " Rebuild wider (see above) before reading any number"
                    " from it.", nqdone);
    else if (degraded && !cfg->relations)
        /* No --relations at all: nothing was written and nothing was meant to
         * be, so this is not the "cannot resume" error below. Said plainly
         * because this is the benchmarking path, where the number the operator
         * came for is the one the skipping just invalidated. */
        runlog_warn("\n  stopped after %u q with no relation file requested;"
                    " nothing to resume. The timings from this band are NOT"
                    " usable -- it skipped its way to the ceiling.", nqdone);
    else if (degraded && ckpt_written)
        /* Same checkpoint, same "opposite advice" trap as the cap above. Both
         * ceilings reset on resume, so "rerun the same command" would sieve
         * another PIPE_LOST_MAX q (or PIPE_SLAB_SKIP_MAX slabs) into the same
         * wall and stop again, forever. The resume point is still real and a
         * rerun with the job FIXED should start from it. */
        runlog_warn("\n  stopped after %u q this session (%llu total, %llu"
                    " relations); checkpoint written, resume at q=%llu. FIX"
                    " THE JOB BEFORE RESUMING: both ceilings reset on resume,"
                    " so an unchanged rerun stops again about %d lost q or %d"
                    " skipped slabs later.",
                    nqdone, base_nq + nqdone, ck.nrel,
                    (unsigned long long)ck.next_q,
                    PIPE_LOST_MAX, PIPE_SLAB_SKIP_MAX);
    else if (stopped && ckpt_written)
        printf("\n  stopped after %u q this session (%llu total, %llu"
               " relations). Rerun the same command to resume at q=%llu.\n",
               nqdone, base_nq + nqdone, ck.nrel,
               (unsigned long long)ck.next_q);
    else if (stopped)
        /* Now an ERROR, not a zero exit. A stop with no resume point is not
         * resumable work, and reporting it as a clean stop is what would turn
         * a BOINC temporary exit into a restart loop over a band that always
         * begins again from nothing. See the PIPE_RC_ mapping at the end. */
        /* DO NOT name rtmp as something to move aside: it is already gone.
         * This branch is reached only when ckpt_written == 0, which is
         * exactly the keep_partial = 0 that made pipe_finalize_outputs
         * remove(rtmp) above -- a .part with no sidecar is not resumable work
         * and keeping it would strand every rerun on the startup refusal. So
         * the old advice pointed at a path that no longer exists, which is
         * the same defect the two branches above were just fixed for. The
         * session simply produced nothing; say that. */
        fprintf(stderr,
                "\n  stopped after %u q, but NO checkpoint could be written,"
                " so this session\n  produced nothing resumable and its"
                " staged output was discarded. Fix what\n  prevented the"
                " checkpoint (the warning above names it) and rerun.\n",
                nqdone);

    /* OUTSIDE the `if (nqdone)` below, because the case that most needs saying
     * is a band where skips left nqdone == 0: guarded, the one run that
     * produced nothing but skips would have reported nothing at all. A band
     * runs for days and scrolls, so this has to survive in the summary and not
     * only in the per-q warnings. Silent is the one thing this must not be. */
    /* runlog_warn for the same reason the capped stop uses it: a band that
     * skipped 99 q never reaches the cap, exits 0, and reports a completed work
     * unit -- so this line is the ONLY signal that its output is deficient, and
     * on stdout a project never sees it. It also names the width now; that
     * number was computed at every skip and thrown away here. */
    if (nqskip && g_pipe_need_limbs > 0)
        runlog_warn("\n  ** %llu special-q SKIPPED: exact norm wider than the"
                    " %d bits this build carries. Run ./normscan over the band"
                    " and rebuild with `make BN_LIMBS=%d`. **",
                    nqskip, BN_LIMBS * 32, g_pipe_need_limbs);
    else if (nqskip)
        runlog_warn("\n  ** %llu special-q SKIPPED: exact norm wider than the"
                    " %d bits this build carries, and no supported BN_LIMBS is"
                    " wide enough for every one of them. Run ./normscan over the"
                    " band. **", nqskip, BN_LIMBS * 32);
    if (nqdone) {
        const double N = nqdone;
        const double dev = (tm.rank + tm.emit + tm.summary + tm.resieve + tm.td
                            + tm.classify + tm.compact + tm.record) / N;
        /* Clamped once, in pipe_ovl_credit -- per-slab clamping biased the
         * estimator (see the accumulate site). */
        const double ovl_pr = pipe_ovl_credit(acc_ovl);
        const double acc_sieve = acc_tr + acc_fi + acc_ap - ovl_pr;
        printf("\n\n  --- band of %u special-q ---\n", nqdone);
        if (t_verify > 0)
            printf("  (first-q reconstruction gate: %.1f ms, excluded below)\n",
                   t_verify);
        printf("  %-34s %8.2f ms\n", "wall clock per q", acc_wall / N);
        if (cfg->qspan) {
            printf("  %-34s %8.2f ms   <- --qspan\n",
                   "  GPU span of one q", acc_qspan / N);
            printf("  %-34s %8.2f ms\n",
                   "    host clock, same bracket", acc_hspan / N);
            printf("  %-34s %8.2f ms\n",
                   "    of which: prologue (pre-GPU)",
                   acc_prologue / N);
            printf("  %-34s %8.2f ms\n", "    region: prepare_q, both sides",
                   acc_prep / N);
            printf("  %-34s %8.2f ms\n", "    region: td_prepare_q",
                   acc_tdprep / N);
            printf("  %-34s %8.2f ms\n", "    region: slab loop, all stages",
                   acc_slab / N);
            printf("  %-34s %8.2f ms\n", "    region: tail to accumulation",
                   acc_tail / N);
            printf("  %-34s %8.2f ms\n",
                   "    host, nothing in flight",
                   acc_wall / N - acc_qspan / N);
        }
        printf("  %-34s %8.2f ms\n", "  sieve, both sides", acc_sieve / N);
        printf("  %-34s %8.3f ms\n", "    transform + plattice", acc_tr / N);
        printf("  %-34s %8.3f ms\n", "    fill", acc_fi / N);
        printf("  %-34s %8.3f ms\n", "    apply", acc_ap / N);
        if (ovl_pr > 0)
            printf("  %-34s %8.3f ms   <- --fill-concurrent\n",
                   "    less: sides overlapped", -ovl_pr / N);
        printf("  %-34s %8.3f ms\n", "  intersect + gcd", acc_isect / N);
        printf("  %-34s %8.3f ms\n", "  host per-q (sieve tables, staging)",
               acc_host / N);
        printf("  %-34s %8.2f ms   <- of which, on the device:\n",
               "  TD + classify, wall", acc_td / N);
        printf("      %-30s %8.3f ms\n", "rank scan (shared)", tm.rank / N);
        printf("      %-30s %8.3f ms\n", "emit (x,a,b), rank order (shared)",
               tm.emit / N);
        printf("      %-30s %8.3f ms\n", "survivor filter (shared)", tm.summary / N);
        printf("      %-30s %8.3f ms\n", "resieve + scatter, both sides",
               tm.resieve / N);
        printf("      %-30s %8.3f ms\n", "  of which side 0", tm.resieve0 / N);
        printf("      %-30s %8.3f ms\n", "norms + trial division, both sides",
               tm.td / N);
        printf("      %-30s %8.3f ms\n", "classify, both sides", tm.classify / N);
        printf("      %-30s %8.3f ms\n", "joint accept + compact", tm.compact / N);
        printf("      %-30s %8.3f ms\n", "record candidate factorisations",
               tm.record / N);
        printf("      %-30s %8.3f ms\n", "= device total", dev);
        printf("      %-30s %8.3f ms\n", "host: small-prime tables", tm.hostq / N);
        printf("      %-30s %8.3f ms\n", "host: readback of candidates",
               tm.readback / N);
        printf("      %-30s %8.3f ms\n", "host: unaccounted",
               acc_td / N - dev - tm.hostq / N - tm.readback / N);
        printf("  %-34s %8.2f ms\n", "  join and emit", tm.join / N);
        if (cfg->cofactor)
            printf("  %-34s %8.2f ms\n", "  cofactorisation, in-loop flushes",
                   tm.cofac / N);
        /* Under --q-pipeline the stages above overlap the other q, so the
         * residual is not a time anything spent (it goes negative). */
        if (qpipe)
            printf("  %-34s %8s\n", "  unaccounted", "n/a");
        else
            printf("  %-34s %8.2f ms\n", "  unaccounted",
                   (acc_wall - acc_sieve - acc_isect - acc_host - acc_td - tm.join
                    - tm.cofac) / N);
        /* The one number that makes host starvation visible. Every stage above
         * is either a cudaEvent (blind to what the CPU is doing) or a host
         * clock, so a contended box prints PERFECT kernel times next to a bad
         * wall clock and nothing says why. Finding 53: saturating this box's 16
         * cores left fill and apply flat within 1% while wall went 24.30 ->
         * 31.27 ms/q. No idle/loaded pair is quoted here on purpose: the first
         * one measured (0.824 / 0.661) used an earlier expression that kept the
         * cofactor queue in the denominator only, and no pair has yet been
         * taken with THIS one on a box confirmed idle. Finding 53 is canonical
         * and says so; a number duplicated into a comment is a number that
         * drifts, which is how the 0.824 and a hand-derived 0.85 ended up in
         * the docs and this file describing the same measurement.
         *
         * THE COFACTOR QUEUE IS OUT OF BOTH SIDES. dev covers only the
         * event-timed TD/classify kernels; the queue's flush is measured with a
         * host clock (tm.cofac) and mixes device and host, so it can go in
         * neither term. Leaving it in the denominator alone made the ratio move
         * with SURVIVOR DENSITY -- a band with more candidates flushes more and
         * reads as a contended host on a perfectly idle box, which is the exact
         * misdiagnosis this line exists to prevent.
         *
         * Even so the numerator is a LOWER BOUND on device time: k_cof_enqueue,
         * k_cand_stats and the flush's own kernels are real GPU work that no
         * event times, so the ratio understates utilisation by a small fixed
         * amount. That is tolerable for the comparison it is for and fatal for
         * the one it is not -- hence:
         *
         * Deliberately NO threshold and no warning text. The healthy value is a
         * function of the card and the job -- a faster GPU spends relatively
         * more of its wall on the same host work, so it sits LOWER when
         * perfectly healthy -- and a hardcoded "good" constant here would be
         * the same mistake as the 144-block default: one box's number promoted
         * to a universal one. Compare it against your own idle baseline on the
         * same card, job AND band length; that comparison is valid, an absolute
         * reading is not.
         *
         * Band length matters because acc_wall excludes the final cofactor
         * flush (see the comment on cofac_tail above). On a band shorter than
         * one flush that tail IS the whole cofactorisation, so a short smoke
         * run and a production band are not comparable to each other. */
        const double gpu_acc = (acc_sieve + acc_isect + dev * N) /
                               (acc_wall - tm.cofac);
        char gpu_acc_s[32];
        if (qpipe) snprintf(gpu_acc_s, sizeof gpu_acc_s, "n/a");
        else snprintf(gpu_acc_s, sizeof gpu_acc_s, "%.3f", gpu_acc);
        printf("  %-34s %8s\n", "GPU-accounted / wall (excl cofac)", gpu_acc_s);
        /* What --q-pipeline actually overlapped, as against what it was
         * granted: fillcheck.sh reads this line to tell a band that issued
         * slabs ahead from one that only ever issued whole q. */
        if (qpipe)
            printf("  %-34s %llu slab(s), %llu q\n", "issued ahead (--q-pipeline)",
                   n_ahead_slab, n_ahead_q);
        printf("  %-34s %8.1f\n", "two-sided primitive survivors/q",
               (double)acc_surv / N);
        printf("  %-34s %8.2f\n", "cofactorisation candidates/q",
               (double)acc_cand / N);
        printf("  %-34s %8.3f\n", "COMPLETE RELATIONS/q", (double)acc_rel / N);
        printf("  %-34s %8llu\n", "total relations", (unsigned long long)acc_rel);
        printf("  %-34s %8llu\n", "total candidates", (unsigned long long)acc_cand);
        /* The band summary in one record, so the log ends with the numbers a
         * reader would otherwise have to reconstruct from the last heartbeat
         * and the console output the session no longer has. Timing terms are
         * the summary's own, so the two cannot disagree.
         *
         * `nq` and `rel` MEAN HERE WHAT THEY MEAN IN A HEARTBEAT: totals
         * across sessions, and `rel` counted the way the heartbeat counts it
         * -- Q.nrel under --cofactor, which is what reached the file, not
         * acc_rel, which is only the subset trial division completed without
         * splitting (111 against 374 on a 3-q c147 smoke run). The
         * session-only aggregates get their own names rather than reusing
         * those two keys for a second quantity: a log where one key means two
         * things is worse than one field short, because nothing in the file
         * says which reading applies to which line. */
        runlog_record("band end  nq=%llu  rel=%llu  nq_session=%u"
                      "  rel_session=%llu  cand_session=%llu  wall=%.2fms/q"
                      "  sieve=%.2fms/q  td=%.2fms/q  acc/wall=%s"
                      "  rel/q=%.3f%s",
                      base_nq + nqdone,
                      base_rel + (cfg->cofactor ? Q.nrel
                                                : (unsigned long long)acc_rel),
                      nqdone,
                      cfg->cofactor ? Q.nrel : (unsigned long long)acc_rel,
                      (unsigned long long)acc_cand,
                      acc_wall / N, acc_sieve / N, acc_td / N, gpu_acc_s,
                      /* rel_session per q, not the summary's COMPLETE
                       * RELATIONS/q, which counts only what trial division
                       * finished -- next to rel_session on the same line the
                       * two would read as a contradiction. */
                      (double)(cfg->cofactor ? Q.nrel
                                             : (unsigned long long)acc_rel) / N,
                      /* `degraded` BEFORE `stopped`, because the ceiling
                       * sets both. Ordered the other way -- which is how this
                       * read until degradecheck went looking -- an exit-5
                       * band recorded "[stopped cleanly]" in the one artifact
                       * that outlives the run, asserting the opposite of its
                       * own exit status to whoever reads the log later. */
                      capped   ? "  [stopped: build too narrow]"
                      : degraded ? "  [DEGRADED: yield not creditable]"
                      : stopped ? "  [stopped cleanly]"
                      : (rc ? "  [FAILED]" : ""));
        if (cfg->cofactor) {
            printf("\n  --- cofactorisation, cross-q queue ---\n");
            printf("  %-34s %8.2f ms\n", "rational queue", Q.ms_rat / N);
            printf("  %-34s %8.2f ms\n", "algebraic queue", Q.ms_alg / N);
            printf("  %-34s %8.3f ms\n", "readback + emit of relations", Q.ms_host / N);
            printf("  %-34s %8.2f ms\n", "  = device time per q",
                   (Q.ms_rat + Q.ms_alg + Q.ms_host) / N);
            printf("  %-34s %8.2f ms\n", "wall: in-loop flushes", tm.cofac / N);
            if (cfg->cof_async && A.nflush) {
                printf("  %-34s %8.2f ms\n", "  of which waiting on a flush",
                       A.ms_wait / N);
                printf("  %-34s %8.0f ms each, one every %.0f ms (%d)\n",
                       "async flushes", A.ms_flush / A.nflush,
                       A.nflush > 1 ? A.ms_interval / (A.nflush - 1) : 0.0,
                       A.nflush);
            }
            printf("  %-34s %8.2f ms\n", "wall: final flush after the band",
                   cofac_tail / N);
            /* Everything joint-accepted is enqueued, INCLUDING the records
             * trial division already completed -- the queue is the single
             * emitter, so they have to travel with the rest. They cost nothing
             * to split (their status is already CF_OK, and compaction drops
             * them before the first round), but they are counted here, so this
             * is a record count and not a splitting-work count. Labelling it
             * "cofactorised" made it look like a 7-record discrepancy against
             * the candidate total at the parity q. */
            printf("  %-34s %8llu  (of which %llu needed no splitting)\n",
                   "records enqueued", (unsigned long long)Q.nseen,
                   (unsigned long long)acc_rel);
            /* Per-side outcome, and in particular the STUCK count.
             *
             * This exists because its absence hid two real yield losses: 13.7%
             * on AS276 and 0.33% on the c183 (RESULTS findings 69 and 70). A
             * record whose splitter ran out of budget is left CF_INCOMPLETE,
             * quietly fails to become a relation, and is indistinguishable in
             * every other printed number from a job that simply had no more
             * relations to give. Only a corpus comparison found it. Now the
             * run says so itself. */
            {
                const unsigned long long st0 = Q.nst[0][CF_INCOMPLETE];
                const unsigned long long st1 = Q.nst[1][CF_INCOMPLETE];
                const unsigned long long stuck = st0 + st1;
                for (int sd = 0; sd < 2; sd++) {
                    char lbl[40];
                    snprintf(lbl, sizeof lbl, "  side %d: split / dead / stuck", sd);
                    printf("  %-34s %8llu / %llu / %llu\n", lbl,
                           Q.nst[sd][CF_OK], Q.nst[sd][CF_DEAD],
                           Q.nst[sd][CF_INCOMPLETE]);
                    /* Two producers, so the label names neither: mz_split
                     * sets CF_OVERFLOW when a split yields more than
                     * CF_MAXFAC parts, and k_cof_enqueue sets it when a
                     * residual is wider than the side's limbs. Saying
                     * "CF_MAXFAC" would point an operator at the splitter's
                     * stack guard when the actual cause was mfb vs CF_LMAX. */
                    if (Q.nst[sd][CF_OVERFLOW])
                        printf("  %-34s %8llu\n",
                               "    OVERFLOW (parts, or cofactor width)",
                               Q.nst[sd][CF_OVERFLOW]);
                }
                /* Warned, not merely printed, because the band otherwise
                 * exits 0 looking healthy.
                 *
                 * "Unresolved", NOT "lost". A stuck record's status is
                 * UNKNOWN, and the two populations behind it are very
                 * different: it may be a cofactor that would have split into a
                 * relation, or a composite whose factors ALL exceed lpb --
                 * which can never be a relation, but which the splitter can
                 * only prove dead by factoring it anyway. At a tight mfb/lpb
                 * ratio the second population dominates, so a large stuck
                 * count is not by itself a large yield loss. Measuring which
                 * is which costs a run at a bigger budget, which is exactly
                 * what this line is telling the operator to do. */
                /* A RATE, not any nonzero count. A correctly tuned 3LP band
                 * still leaves plenty of records unresolved -- AS276 at its
                 * saturated-yield configuration sits at 20.9% -- because most
                 * of them are composites whose factors all exceed lpb, which
                 * can only be proven dead by factoring them. Warning on
                 * `stuck > 0` would therefore fire on nearly every healthy
                 * production band, and under BOINC stderr is uploaded per work
                 * unit. The counts above are always printed; this fires only
                 * when the rate is high enough to be worth a look.
                 *
                 * A third is a heuristic with two data points behind it, not a
                 * derived bound: AS276 sits at 20.9% when correctly tuned and
                 * 55% when its budget was five times too small. Nothing inside
                 * a single run distinguishes the two, which is why the message
                 * names the actual test rather than asserting a fault. */
                if (Q.nseen && stuck * 3ull > Q.nseen)
                    runlog_warn("  %llu of %llu records (%.1f%%) left unresolved by"
                                " the splitting budget. That is high; a healthy 3LP"
                                " band runs ~20%%. These are cofactors of UNKNOWN"
                                " status, so some may have been relations -- re-run"
                                " a short band at a higher --cof-budget /"
                                " --cof-rounds and see whether the yield moves.",
                                stuck, Q.nseen,
                                100.0 * (double)stuck / (double)Q.nseen);
            }
            /* Q.nrel is EVERY relation: the queue is the single emitter, so
             * it already holds the ones trial division completed. */
            printf("  %-34s %8.2f\n", "ALL RELATIONS/q", (double)Q.nrel / N);
            printf("  %-34s %8.2f\n", "  of which, from cofactorisation",
                   (double)(Q.nrel - acc_rel) / N);
            printf("  %-34s %8llu\n", "total relations",
                   (unsigned long long)Q.nrel);
            /* acc_wall ALREADY contains the queue: the flush happens inside
             * the q loop. Adding the device times to it would double count,
             * which the first version of this line did -- it reported 132.31
             * ms/q against a 112 ms/q process wall clock. */
            printf("  %-34s %8.2f ms\n", "wall clock per q, COMPLETE",
                   (acc_wall + cofac_tail) / N);
        }
    }

#undef PIPE_CK
#undef PIPE_VRAM_MARGIN
#undef PIPE_Q_ROLLBACK
done:
    /* BELOW `done:`, not above it, so "always reported" is literally true.
     * Every PIPE_CK failure is `rc = -1; goto done`, so sited above this label
     * the whole summary was skipped by exactly the runs that most need it: a
     * band that soft-skipped 400 slabs and then died on a mtlMemcpy reported
     * "band FAILED" with no hint that the yield had already gone, which is
     * the difference between "transient device fault" and "the bucket array
     * was undersized the whole time".
     *
     * To the log rather than stdout: under BOINC stdout dies with the slot
     * directory, and a task that silently yielded less than it should have is
     * exactly what a validator operator needs to see. */
    if (nslab_skipped)
        runlog_warn("  %llu slab(s) skipped or abandoned -- this band's yield"
                    " is short by whatever those slabs held. (This total"
                    " covers bucket overflow AND the trial-division causes"
                    " below; a bucket overflow also abandons the remaining"
                    " slabs of its special-q, which are counted here too.)",
                    nslab_skipped);
    if (ntd_skipped)
        runlog_warn("  of those, %llu were single slabs dropped for a"
                    " large-prime list truncated past PIPE_K per survivor,"
                    " which usually means mfb or PIPE_K needs attention",
                    ntd_skipped);
    if (nq_lost)
        runlog_warn("  %llu special-q produced nothing because every slab of"
                    " them was skipped", nq_lost);
    if (stop_hooked) bench_stop_hook_remove();
    /* CUDA failures can jump here from any stage. Normal completion already
     * finalized exactly once above; an early jump closes and discards here --
     * but still honours a written checkpoint, since a .part that a sidecar
     * describes is resumable work whatever killed this session. */
    /* An error path can leave a flush in flight; it writes fr, so join it
     * before the outputs are closed below. */
    if (cofa_wait(&A, &Q)) rc = -1;
    if (!outputs_finalized) {
        if (pipe_finalize_outputs(&fr, &fc, cfg, rtmp, ctmp, 0, ckpt_written))
            rc = -1;
    }
    pipe_td_free(&C);
    cofq_free(&Q, &QO);
    /* A special-q issued ahead and never consumed may still be sieving. */
    if (st_sv) mtlStreamSynchronize(st_sv);
    if (A.spare.cap) cofq_free(&A.spare, &A.spare_out);
    if (g_cof_stream) { mtlStreamDestroy(g_cof_stream); g_cof_stream = 0; }
    pside_free(&S1); pside_free(&S0);
    mtlFree(d_bucket); mtlFree(d_cursor); mtlFree(d_overflow);
    if (st_sv) mtlStreamDestroy(st_sv);
    mtlFree(d_bucket_c); mtlFree(d_cursor_c); mtlFree(d_overflow_c);
    if (st_side[0]) mtlStreamDestroy(st_side[0]);
    if (st_side[1]) mtlStreamDestroy(st_side[1]);
    mtlFree(d_two); mtlFree(d_n); mtlFree(d_pre); mtlFree(d_gate);
    if (ea) mtlEventDestroy(ea);
    if (eb) mtlEventDestroy(eb);
    if (qspan0) mtlEventDestroy(qspan0);
    if (qspan1) mtlEventDestroy(qspan1);
    /* ONE place maps the band's four outcomes onto the return code, and it is
     * after every cleanup path that can still turn a success into a failure.
     * Everything above works in the internal 0/-1 convention, so no `rc < 0`
     * test anywhere in this function had to learn about the new values.
     *
     * A stop with no checkpoint is a FAILURE, not a stop: there is nothing to
     * resume, so calling it resumable is what would make a client restart it
     * forever. See `enum bench_outcome` in bench.h. */
    /* The ceiling is deliberately NOT re-tested here. By this point a band
     * that left the loop any way other than the loop-top stop has FINISHED:
     * `commit` above renamed the .part onto the final --relations name and
     * removed the checkpoint sidecars. Reporting DEGRADED then would tell a
     * BOINC client to record an application error for a complete work unit
     * whose resume state has already been deleted, and would contradict the
     * one thing exit 5 promises -- that the relations are waiting in the
     * .part. A band that finishes under its own steam is a success with loud
     * warnings (see the end-of-band summary above), not an error.
     *
     * The cost is the documented hole: a band whose LAST q crosses a limit
     * exits 0. That is the same hole PIPE_SKIP_MAX has, and for the same
     * reason -- these counters are a backstop, not the primary defence.
     *
     * `degraded` is tested before `stopped` because both are set together by
     * the ceiling; it carries ckpt_written for the same reason `stopped`
     * does, since a stop with nothing to resume from is a failure however it
     * was reached. */
    if (rc == 0) {
        /* capped carries the SAME ckpt_written test as degraded below, and
         * for the same reason: with --relations set and no checkpoint
         * written, pipe_finalize_outputs ran with keep_partial = 0 and
         * DELETED the staged relation file. Reporting UNSUPPORTED there
         * ("rebuild with a wider BN_LIMBS") describes a build problem to
         * BOINC for a session whose output was destroyed and which has no
         * resume point -- the operator rebuilds and restarts from q0 with no
         * indication anything was lost. The rebuild advice is still correct
         * and still on stderr; the exit status has to say the session failed.
         * Left unconditional, this was the one hazard the two policy stops
         * still disagreed about after the rest of this was made symmetric. */
        if (capped)         rc = (!cfg->relations || ckpt_written)
                                     ? PIPE_RC_UNSUPPORTED : PIPE_RC_FAIL;
        /* "No relation file was requested" is NOT the no-checkpoint failure:
         * with no --relations there was never any output to resume, which is
         * the plain benchmarking case the ceiling exists to serve -- a
         * degraded band there must still report DEGRADED so the rate is not
         * read as a slow card. Only a run that HAS a relation file and could
         * not checkpoint it is unresumable, and that is the failure.
         *
         * cfg->relations, NOT fr, for the reason spelled out at the terminal
         * messages above: pipe_finalize_outputs has already closed fr and set
         * it to NULL, so `!fr` was unconditionally true here and the
         * PIPE_RC_FAIL arm could never be taken. A degraded band that asked
         * for relations and could not write a single checkpoint reported
         * DEGRADED -- resumable -- when nothing of it is resumable. */
        else if (degraded)  rc = (!cfg->relations || ckpt_written)
                                     ? PIPE_RC_DEGRADED : PIPE_RC_FAIL;
        else if (stopped)   rc = ckpt_written ? PIPE_RC_STOPPED : PIPE_RC_FAIL;
    }
    return rc;
}

/* ---- the slab target from free VRAM ------------------------------------ *
 *
 * SLAB_PERF_REGIONS (2^29 positions at --region 14) was tuned on factor bases
 * of a few million entries. Every slab re-walks every bucketed entry's lattice
 * state, so with a large factor base that per-slab cost dominates: on the
 * 2,1147+ SNFS job (31M entries a side, I16, RTX 4090) going from four 2^29
 * slabs to one 2^31 slab cut fill 120.8 -> 79.9 ms/q and resieve 47.5 ->
 * 33.3 ms/q, byte-identical relations, for 5.5 -> 9.4 GB of device memory.
 *
 * Small cards cannot pay that, so the target is chosen at run time: the
 * largest of 4x, 2x the default whose modelled footprint fits the free memory
 * with headroom, else the default exactly as before. The model mirrors the
 * allocations pipe_side_init and run_pipeline_impl make, and errs high; the
 * per-allocation admission checks there still stand behind it. */
static double pipe_inv_sum(const fb_t *fb)
{
    double acc = 0;
    for (uint32_t i = 0; i < fb->n; i++) acc += 1.0 / fb->primes[i];
    return acc;
}

static size_t pipe_vram_model(const fb_t *fb1, const fb_t *fb0,
                              const bench_cfg_t *cfg, uint32_t jmax,
                              double inv1, double inv0)
{
    const uint64_t X = ((uint64_t)1 << cfg->logI) * jmax;
    const uint64_t nregion = X >> cfg->log_region;
    const double inv = inv1 > inv0 ? inv1 : inv0;
    const uint64_t est = (uint64_t)((double)X * inv * 1.15) + 4096; /* pipe_est_records */
    const uint64_t cap = est / nregion + 256;
    const size_t narr = cfg->fill_concurrent ? 2 : 1;
    size_t b = 0;
    b += narr * (size_t)nregion * cap * 4;              /* bucket array(s)   */
    b += narr * (size_t)nregion * 4;                    /* cursors           */
    b += 3 * (size_t)(X / 8);                           /* S1, S0, d_two     */
    b += 2 * (size_t)(X / 64);                          /* --fill-gate folds */
    b += (size_t)(fb1->n + fb0->n) *
         (4 + 4 + sizeof(plat_t) + 8 + 8 + 4);          /* per-entry state   */
    if (cfg->q_pipeline)        /* plat_alt, rowrt_alt (nrow <= n), walk_spare */
        b += (size_t)(fb1->n + fb0->n) * (sizeof(plat_t) + 4 + 8);
    b += (size_t)768 << 20;   /* TD buffers, cofactor queue(s), small tables */
    return b;
}

/* One run_pipeline_impl, at whichever instantiation the plan needs: a plan
 * whose one slab covers the whole area runs unslabbed. */
static int pipe_impl_any(const slab_plan_t *P, const fb_t *fb1,
                         const fb_t *fbs1, const fb_t *fb0, const fb_t *fbs0,
                         const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                         const poly_t *POLY, const bench_cfg_t *cfg)
{
    return P->enabled
        ? run_pipeline_impl<true>(fb1, fbs1, fb0, fbs0, qlist, nq, qgen, POLY,
                                  cfg, P)
        : run_pipeline_impl<false>(fb1, fbs1, fb0, fbs0, qlist, nq, qgen, POLY,
                                   cfg, P);
}

/* ---- --instances N: N whole pipelines on one GPU ----------------------- *
 *
 * Several special-q in flight without restructuring the q loop: the bounded
 * band is cut into N equal q ranges and each runs its own run_pipeline_impl on
 * its own host thread. The build uses --default-stream per-thread, so each
 * thread's launches go to its own stream and the pipelines' kernels overlap
 * on the device (2,1147+, RTX 4090: two instances 73.5 s of band against
 * 80.5 s for one, byte-identical relations).
 *
 * Each instance is an ordinary band over its share, with its own relation
 * file NAME.inst<k>: its own .part, checkpoint, BOINC recovery counter,
 * resume and commit, all the existing code. What the instances share is set
 * up here -- the stop hook, the watchdog and run-log slots, the BOINC
 * fraction -- and what they produce is merged here, in q order, into NAME:
 * the file one pipeline would have written. NAME.instances (written by
 * run_pipeline when the band starts) pins N and the band for a relaunch. */
static int pipe_inst_merge(const pipe_inst_plan_t *IP, const char *relations)
{
    static char buf[1 << 20];
    char tmp[CKPT_PATH_MAX];
    FILE *out;
    if (ckpt_path_fmt(relations, tmp, sizeof tmp, ".instances.merge")) {
        fprintf(stderr, "--instances: output path too long\n");
        return -1;
    }
    if (!(out = fopen(tmp, "wb"))) { perror(tmp); return -1; }
    for (int k = 0; k < IP->n; k++) {
        FILE *in = fopen(IP->in[k].relations, "rb");
        size_t m;
        int bad = 0;
        if (!in) { perror(IP->in[k].relations); bad = 1; }
        while (!bad && (m = fread(buf, 1, sizeof buf, in)) > 0)
            if (fwrite(buf, 1, m, out) != m) { perror(tmp); bad = 1; }
        if (in) {
            if (!bad && ferror(in)) { perror(IP->in[k].relations); bad = 1; }
            fclose(in);
        }
        if (bad) { fclose(out); remove(tmp); return -1; }
    }
    if (fflush(out) || bench_sync_stream(out) || fclose(out)) {
        perror(tmp); remove(tmp); return -1;
    }
    if (bench_atomic_replace(tmp, relations)) {
        perror(relations); remove(tmp); return -1;
    }
    if (bench_sync_parent(relations)) perror(relations);
    /* The instance files first and the manifest last. Interrupted in
     * between, a relaunch still finds the manifest, sieves again whichever
     * instance's file is gone, and merges again: wasted work, never a lost
     * or doubled special-q. */
    for (int k = 0; k < IP->n; k++)
        if (remove(IP->in[k].relations) && errno != ENOENT)
            perror(IP->in[k].relations);
    if (IP->manifest[0] && remove(IP->manifest) && errno != ENOENT)
        perror(IP->manifest);
    return 0;
}

/* One instance, on its own thread: its configuration is the band's with its
 * own share, relation file and resume state, and its own special-q stream. */
static int pipe_inst_run_one(const fb_t *fb1, const fb_t *fbs1,
                             const fb_t *fb0, const fb_t *fbs0,
                             const poly_t *POLY, const bench_cfg_t *base,
                             const pipe_inst_one_t *I, int k,
                             const slab_plan_t *plan,
                             const slab_plan_t *base_plan, int upsized)
{
    bench_cfg_t c = *base;
    qsel_t first;
    int r, have, rc;
    c.inst = NULL;
    c.relations = I->relations[0] ? I->relations : NULL;
    c.qmin = I->resume ? I->resume_q : I->qmin;
    c.qmax = I->qmax;
    c.nq_max = 0;
    c.restart = 0;
    c.resume = I->resume;
    c.resume_q = I->resume_q;
    c.resume_rho = I->resume_rho;
    c.resume_qmin = I->resume_qmin;
    c.resume_rel_bytes = I->resume_rel_bytes;
    c.resume_cand_bytes = 0;
    c.resume_nrel = I->resume_nrel;
    c.resume_nq = I->resume_nq;

    sqgen_t *G = sqgen_create(POLY, c.sq_side, c.qmin, c.qmax, 0);
    if (!G) return PIPE_RC_FAIL;
    r = sqgen_next(G, &first);
    /* As bench_main does for one pipeline: the checkpoint names the exact
     * root, so wind forward to it rather than re-sieve the q's other roots.
     * An instance whose checkpoint is already past its last q still runs,
     * with nothing to sieve, so that it commits. */
    if (r > 0 && c.resume) {
        int guard = 0;
        while ((first.q != c.resume_q || first.rho != c.resume_rho) &&
               ++guard <= BENCH_NCOEFF)
            if ((r = sqgen_next(G, &first)) <= 0) break;
        if (r > 0 && (first.q != c.resume_q || first.rho != c.resume_rho)) {
            fprintf(stderr, "  instance %d: the generator does not produce"
                    " q=%llu rho=%llu at its checkpoint\n", k,
                    (unsigned long long)c.resume_q,
                    (unsigned long long)c.resume_rho);
            sqgen_free(G);
            return PIPE_RC_FAIL;
        }
    }
    if (r < 0) { sqgen_free(G); return PIPE_RC_FAIL; }
    have = r > 0;
    rc = pipe_impl_any(plan, fb1, fbs1, fb0, fbs0, have ? &first : NULL,
                       have ? 1u : 0u, G, POLY, &c);
    /* Nothing is opened or consumed before the bucket admission, so this
     * instance can retry at the default slab exactly as one pipeline does --
     * whatever its siblings are doing. */
    if (rc == PIPE_RC_NOVRAM && upsized) {
        printf("  [%d] slab size from VRAM: the up-sized plan no longer fits;"
               " retrying at the default %u rows/slab, one bucket array\n",
               k, base_plan->jmax);
        c.dual_bucket = 0;
        rc = pipe_impl_any(base_plan, fb1, fbs1, fb0, fbs0,
                           have ? &first : NULL, have ? 1u : 0u, G, POLY, &c);
    }
    sqgen_free(G);
    return rc == PIPE_RC_NOVRAM ? PIPE_RC_FAIL : rc;
}

/* Worst first: a failure outranks the two policy stops, which outrank a clean
 * stop, which outranks a finished band. */
static int pipe_rc_rank(int rc)
{
    switch (rc) {
    case PIPE_RC_OK:          return 0;
    case PIPE_RC_STOPPED:     return 1;
    case PIPE_RC_DEGRADED:    return 2;
    case PIPE_RC_UNSUPPORTED: return 3;
    default:                  return 4;
    }
}

static int run_pipeline_instances(const fb_t *fb1, const fb_t *fbs1,
                                  const fb_t *fb0, const fb_t *fbs0,
                                  const poly_t *POLY, const bench_cfg_t *cfg,
                                  const pipe_inst_plan_t *IP,
                                  const slab_plan_t *plan,
                                  const slab_plan_t *base_plan, int upsized,
                                  size_t m_inst)
{
    const int n = IP->n;
    pipe_inst_shared_t sh;
    pipe_inst_ctx_t ctx[PIPE_INSTANCES_MAX];
    int rc[PIPE_INSTANCES_MAX] = { 0 };
    std::thread th[PIPE_INSTANCES_MAX];
    int hooked = 0, worst = PIPE_RC_OK;
    const double t0 = host_ms();

    sh.n = n;
    sh.m_inst = m_inst;
    sh.setups_left = 0;
    for (int k = 0; k < n; k++) sh.setups_left += !IP->in[k].done;
    printf("\n=== --instances %d: q [%llu, %llu] ===\n", n,
           (unsigned long long)cfg->qmin, (unsigned long long)cfg->qmax);
    for (int k = 0; k < n; k++) {
        const pipe_inst_one_t *I = &IP->in[k];
        sh.frac[k] = I->done ? 1.0 : 0.0;
        printf("  instance %d: q [%llu, %llu]%s", k,
               (unsigned long long)I->qmin, (unsigned long long)I->qmax,
               I->relations[0] ? ", " : "");
        if (I->relations[0]) printf("%s", I->relations);
        if (I->done)        printf(" (complete)");
        else if (I->resume) printf(" (resuming at q=%llu)",
                                   (unsigned long long)I->resume_q);
        printf("\n");
        runlog_note("instance", "%d of %d: q [%llu, %llu] %s%s", k, n,
                    (unsigned long long)I->qmin, (unsigned long long)I->qmax,
                    I->relations[0] ? I->relations : "(no relations)",
                    I->done ? " complete" : I->resume ? " resumed" : "");
    }
    /* Each instance thread must select the device itself: the current device
     * is per host thread and a new one starts on device 0, so on a multi-GPU
     * host every instance would otherwise sieve on device 0 -- not the card
     * the client assigned, and on top of whatever task owns device 0. */
    int dev = 0;
    if (mtlGetDevice(&dev) != mtlSuccess) {
        fprintf(stderr, "--instances: cannot read the current Metal device\n");
        return PIPE_RC_FAIL;
    }
    /* One stop hook for all of them, owned here: an instance that finishes
     * first must not unhook the others (see the g_pinst test where
     * run_pipeline_impl would install it). */
    if (cfg->relations) {
        hooked = bench_stop_hook_install(pipe_request_stop) == 0;
        g_pipe_stop = 0;
    }
    for (int k = 0; k < n; k++) {
        if (IP->in[k].done) continue;
        ctx[k].k = k;
        ctx[k].n = n;
        ctx[k].qmin = IP->in[k].qmin;
        ctx[k].qmax = IP->in[k].qmax;
        ctx[k].sh = &sh;
        th[k] = std::thread([&, k]() {
            char tag[16];
            g_pinst = &ctx[k];
            wd_bind(k + 1);
            snprintf(tag, sizeof tag, "inst%d", k);
            runlog_thread_tag(tag);
            if (mtlSetDevice(dev) != mtlSuccess) {
                fprintf(stderr, "  instance %d: cannot select Metal device %d\n",
                        k, dev);
                rc[k] = PIPE_RC_FAIL;
            } else {
                rc[k] = pipe_inst_run_one(fb1, fbs1, fb0, fbs0, POLY, cfg,
                                          &IP->in[k], k, plan, base_plan,
                                          upsized);
            }
            /* A failure or a policy stop ends the band, so let the others
             * drain and checkpoint at their next special-q rather than sieve
             * on into a work unit that will not be credited. */
            if (rc[k] != PIPE_RC_OK && rc[k] != PIPE_RC_STOPPED &&
                !g_pipe_stop) {
                fprintf(stderr, "  instance %d ended with status %d;"
                        " stopping the others at their next special-q\n",
                        k, rc[k]);
                g_pipe_stop = 1;
            }
            wd_active(0);
            g_pinst = NULL;
        });
    }
    /* The main thread only waits; a parked join is not a stall. */
    wd_active(0);
    for (int k = 0; k < n; k++)
        if (th[k].joinable()) th[k].join();
    wd_active(1);
    wd_phase("instances.merge");
    if (hooked) bench_stop_hook_remove();

    for (int k = 0; k < n; k++)
        if (pipe_rc_rank(rc[k]) > pipe_rc_rank(worst)) worst = rc[k];
    if (worst == PIPE_RC_OK && cfg->relations) {
        if (pipe_inst_merge(IP, cfg->relations)) worst = PIPE_RC_FAIL;
        else printf("\n  --instances %d: the instances' relations merged into"
                    " %s, in q order\n", n, cfg->relations);
    } else if (worst != PIPE_RC_OK) {
        printf("\n  --instances %d: band not finished (status %d); each"
               " instance's own .part and checkpoint are kept, and the same"
               " command resumes all of them\n", n, worst);
    }
    printf("  --instances %d: %.1f s for the instances' bands\n", n,
           (host_ms() - t0) / 1000.0);
    return worst;
}

#include "slab_calib.inc"

extern "C" int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                            const fb_t *fb0, const fb_t *fbs0,
                            const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                            const poly_t *POLY, const bench_cfg_t *cfg)
{
    slab_plan_t plan;
    uint32_t pmax1 = fb_max_td_prime(fbs1);
    uint32_t pmax0 = fb_max_td_prime(fbs0);
    uint32_t pmax = pmax1 > pmax0 ? pmax1 : pmax0;

    if (cfg->fill_rows < 0) {   /* FILL_ROWS_UNSET / _BY_DEVICE: a caller bug */
        fprintf(stderr, "  pipeline: --fill-rows was not resolved (%d)\n",
                cfg->fill_rows);
        return -1;
    }
    /* Auto-calibration returns 0 when it did not run or was inconclusive, in
     * which case the caller's own --slab-j (itself usually 0, meaning the
     * static default) is what the planner sees -- exactly as before. */
    const uint32_t calibrated_j = calibrate_slab_rows(fb1, fbs1, fb0, fbs0,
                                                      qlist, nq, POLY, cfg,
                                                      pmax);
    const uint32_t forced_j = calibrated_j ? calibrated_j : cfg->slab_j;

    if (slab_make_plan(cfg->logI, cfg->log_region, cfg->J, pmax,
                       forced_j, &plan)) {
        fprintf(stderr,
                "  pipeline: cannot make a safe slab plan for logI=%d J=%u"
                " (largest direct-test prime %u, requested --slab-j %u)\n",
                cfg->logI, cfg->J, pmax, forced_j);
        return -1;
    }
    const slab_plan_t base_plan = plan;
    uint32_t perf_regions = SLAB_PERF_REGIONS;
    /* --instances: a band under way has its N; a fresh one takes the number
     * asked for, or for auto (0) the choice below. Without a plan, one. */
    pipe_inst_plan_t *IP = cfg->inst;
    int ninst = IP ? (IP->n ? IP->n : cfg->instances) : 1;
    /* A fresh band never has more shares than q values: inst_split would
     * hand the extra instances an empty range, which sqgen refuses, and the
     * whole band would fail. */
    const uint64_t qspan_all = cfg->qmax - cfg->qmin + 1;
    if (IP && !IP->n && ninst > 1 && (uint64_t)ninst > qspan_all) {
        printf("  --instances %d: q [%llu, %llu] has only %llu value%s;"
               " running %llu\n", ninst, (unsigned long long)cfg->qmin,
               (unsigned long long)cfg->qmax, (unsigned long long)qspan_all,
               qspan_all == 1 ? "" : "s", (unsigned long long)qspan_all);
        ninst = (int)qspan_all;
    }
    /* auto only: the expected special-q in the band, primes in it at one
     * root each on average. */
    const double nq_est = IP && !IP->n && ninst == 0
        ? (double)qspan_all / log(0.5 * ((double)cfg->qmin + (double)cfg->qmax))
        : 0.0;
    /* The slab target: the largest of 4x, 2x the default whose modelled
     * footprint fits free VRAM with headroom, else the default. N instances
     * need N footprints and share ONE headroom -- the margin is for the
     * display and the model's error, which do not multiply. The serial arm's
     * second bucket array is decided later, in run_pipeline_impl, against the
     * memory actually left once everything else it needs is allocated. */
    struct choice_t {
        slab_plan_t P;
        uint32_t regions;
        int idx;                /* 0 = 4x, 1 = 2x, 2 = the default */
        size_t m;
        int fits;               /* n footprints + headroom fit free VRAM */
    };
    choice_t ch = { plan, SLAB_PERF_REGIONS, 2, 0, 1 };
    if (!calibrated_j && !cfg->slab_j && !cfg->slab_fixed) {
        size_t freeB = 0, totalB = 0;
        if (mtlMemGetInfo(&freeB, &totalB) == mtlSuccess) {
            const size_t gib = (size_t)1 << 30;
            const size_t headroom = totalB / 10 > gib ? totalB / 10 : gib;
            const double inv1 = pipe_inv_sum(fb1), inv0 = pipe_inv_sum(fb0);
            const size_t m0 = pipe_vram_model(fb1, fb0, cfg, plan.jmax,
                                              inv1, inv0);
            auto choose = [&](int n, choice_t *out) {
                /* The default is the fallback whether or not it fits: one
                 * pipeline ran it before any of this existed. `fits` says
                 * whether it does, for auto's N and the warning below. */
                *out = { plan, SLAB_PERF_REGIONS, 2, m0,
                         m0 * (size_t)n + headroom <= freeB };
                for (int k = 2; k >= 1; k--) {
                    slab_plan_t P2;
                    const uint32_t r = (uint32_t)SLAB_PERF_REGIONS << k;
                    if (slab_make_plan_r(cfg->logI, cfg->log_region, cfg->J,
                                         pmax, 0u, r, &P2))
                        continue;
                    if (P2.jmax <= plan.jmax) continue;   /* a safety cap binds */
                    const size_t m = pipe_vram_model(fb1, fb0, cfg, P2.jmax,
                                                     inv1, inv0);
                    if (m * (size_t)n + headroom > freeB) continue;
                    *out = { P2, r, 2 - k, m, 1 };
                    return;
                }
            };
            if (ninst == 0) {
                /* auto: two instances only when each still gets the slab
                 * one would, and two footprints fit. A smaller slab does not
                 * pay for the second instance: on 2,1147+ (RTX 4090) two at
                 * 2^30 took 110.0 s against 108.2 s for one at 2^31, while on
                 * c187 two at the same default slab took 30.1 s against 33.8.
                 * Three measured slower than two, so auto stops at two. */
                choice_t c1, c2;
                choose(1, &c1);
                choose(2, &c2);
                /* METAL: one. See metal/gen_pipeline_host.py -- two host
                 * threads against the shim is an untested path whose
                 * demonstrated failure mode is silently wrong relations,
                 * and no gate in this tree exercises it. */
                ninst = 1;
                (void)c2;
                /* A second instance costs its own setup and pays back a
                 * share of the band, so a short band loses. Whole process,
                 * c187, RTX 4090, one against two: 12 q 5.23/5.76 s, 65 q
                 * 7.24/7.55, 109 q 8.77/8.69, 281 q 14.83/14.13, 570 q
                 * 24.95/23.12 -- even near 100 q. PIPE_INST_AUTO_MIN_Q
                 * leaves margin for a host whose setup costs more. */
                const int short_band = ninst == 2 &&
                                       nq_est < PIPE_INST_AUTO_MIN_Q;
                if (short_band) ninst = 1;
                printf("  --instances auto: %d (%.2f GB free%s)\n", ninst,
                       freeB / 1073741824.0,
                       short_band ? "; the band is too short for two" : "");
            }
            choose(ninst, &ch);
            if (ninst > 1 && !ch.fits)
                fprintf(stderr, "  warning: --instances %d: %d x %.2f GB modelled"
                        " plus %.2f GB headroom exceeds the %.2f GB free; an"
                        " instance may fail to allocate\n", ninst, ninst,
                        ch.m / 1073741824.0, headroom / 1073741824.0,
                        freeB / 1073741824.0);
            if (ch.idx < 2)
                printf("  slab size from VRAM: %.2f GB free, modelled"
                       " %.2f GB at %u rows/slab%s (%.2f GB at the default"
                       " %u); using %u bucket regions/slab\n",
                       freeB / 1073741824.0, ch.m / 1073741824.0, ch.P.jmax,
                       ninst > 1 ? " per instance" : "",
                       m0 / 1073741824.0, plan.jmax, ch.regions);
            else
                printf("  slab size from VRAM: %.2f GB free, modelled %.2f GB"
                       " at the default %u rows/slab%s; keeping it\n",
                       freeB / 1073741824.0, m0 / 1073741824.0, plan.jmax,
                       ninst > 1 ? " per instance" : "");
            plan = ch.P;
            perf_regions = ch.regions;
        }
    }
    if (ninst == 0) ninst = 1;     /* auto without a VRAM reading */
#ifdef HAVE_BOINC
    if (cfg->slab_j ||
        slab_perf_jmax(cfg->logI, cfg->log_region, cfg->J) != 0xffffffffu)
        fprintf(stderr, "BOINC: slab plan: %u rows/slab, %u slab%s%s\n",
                plan.jmax, plan.nslab, plan.nslab == 1 ? "" : "s",
                calibrated_j ? " (auto-calibrated)"
                : cfg->slab_j ? " (--slab-j forced)" : " (static default)");
#endif
    /* The A/B record path leaves no other trace: the stage label is the same on
     * both branches and the checkpoint job text does not carry the flag, so a
     * sweep log that mixes the two kernels would have no tell. Finding 78's
     * method note is what this is for -- an entire sweep was discarded there
     * over a harness mixup caught only by row ordering. */
    if (cfg->td_record_scalar)
        printf("  recording pass: SCALAR (one thread per candidate,"
               " --td-record-scalar) -- the pre-finding-77 path, for A/B only\n");

    /* run_pipeline_impl reads the decision from the flag it already tests;
     * its own free-memory check stays behind it as a second line. */
    bench_cfg_t cfg_run = *cfg;
    cfg_run.inst = NULL;

    auto slab_line = [&](const slab_plan_t &P, uint32_t regions) {
        if (!P.enabled) return;
        const uint32_t perf_jmax = slab_perf_jmax_r(cfg->logI, cfg->log_region,
                                                    cfg->J, regions);
        if (calibrated_j) {
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto-calibrated against this run's first special-q,"
                   " not the static target)\n",
                   P.nslab, P.nslab == 1 ? "" : "s", P.jmax);
        } else if (!cfg->slab_j && perf_jmax != 0xffffffffu)
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (auto target %u bucket regions at --region %d;"
                   " safety bounds may reduce it further)\n",
                   P.nslab, P.nslab == 1 ? "" : "s", P.jmax,
                   (unsigned)regions, cfg->log_region);
        else
            printf("  j-slabbing: %u slab%s, up to %u rows/slab"
                   " (<=2^31 local-position safety bound)\n",
                   P.nslab, P.nslab == 1 ? "" : "s", P.jmax);
    };
    slab_line(plan, perf_regions);

    if (IP && ninst >= 2)
        fprintf(stderr, "  warning: --instances %d runs %d pipelines on %d host"
                        " threads, which this Metal port has never tested; the"
                        " failure mode to watch for is wrong relations, not a"
                        " crash. --instances 1 is the tested path.\n",
                ninst, ninst, ninst);
    if (IP && ninst >= 2) {
        if (!IP->n) {
            /* A fresh band: split it, and pin the split before any instance
             * creates a file -- a relaunch reads N from here. */
            IP->n = ninst;
            for (int k = 0; k < ninst; k++) {
                pipe_inst_one_t *I = &IP->in[k];
                inst_split(cfg->qmin, cfg->qmax, ninst, k, &I->qmin, &I->qmax);
                I->relations[0] = 0;
                if (cfg->relations &&
                    (inst_rel_path(cfg->relations, k, I->relations,
                                   sizeof I->relations) ||
                     ckpt_path_usable(I->relations,
                                      "--relations (instance file)")))
                    return PIPE_RC_FAIL;
            }
            if (IP->manifest[0]) {
                inst_manifest_t m;
                memset(&m, 0, sizeof m);
                m.n = ninst;
                m.qmin = cfg->qmin;
                m.qmax = cfg->qmax;
                ckpt_fingerprint(POLY, cfg, m.fp);
                if (inst_manifest_write(IP->manifest, &m)) return PIPE_RC_FAIL;
            }
        }
        return run_pipeline_instances(fb1, fbs1, fb0, fbs0, POLY, &cfg_run, IP,
                                      &plan, &base_plan,
                                      perf_regions != SLAB_PERF_REGIONS,
                                      ch.m);
    }

    int rc = pipe_impl_any(&plan, fb1, fbs1, fb0, fbs0, qlist, nq, qgen, POLY,
                           &cfg_run);
    /* An up-sized plan whose bucket array did not fit or could not be
     * allocated -- typically another task took the memory after the free-VRAM
     * reading above -- runs once more at the default, which is what this
     * build ran before the slab target became VRAM-driven. */
    if (rc == PIPE_RC_NOVRAM && perf_regions != SLAB_PERF_REGIONS) {
        printf("  slab size from VRAM: the up-sized plan no longer fits;"
               " retrying at the default %u rows/slab, one bucket array\n",
               base_plan.jmax);
        cfg_run.dual_bucket = 0;
        slab_line(base_plan, SLAB_PERF_REGIONS);
        rc = pipe_impl_any(&base_plan, fb1, fbs1, fb0, fbs0, qlist, nq, qgen,
                           POLY, &cfg_run);
    }
    return rc == PIPE_RC_NOVRAM ? PIPE_RC_FAIL : rc;
}

#endif  /* CUDA_SIEVE_PIPELINE_CUH */
