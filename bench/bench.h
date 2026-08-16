/* Shared declarations for the standalone GPU bucket-fill benchmark. */
#ifndef CUDA_SIEVE_BENCH_H
#define CUDA_SIEVE_BENCH_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- factor base ------------------------------------------------------ */

/* GNFS normally uses degree 5 or 6, but octic SNFS polynomials are valid and
 * occur in the local job corpus.  Keep one named bound across parsing, norm
 * setup and exact trial division so c8 cannot overrun a degree-7 array. */
#define BENCH_MAX_DEGREE 8
#define BENCH_NCOEFF (BENCH_MAX_DEGREE + 1)

/* Odd-only Eratosthenes sieve shared by both native factor-base builders.
 * The returned ascending array belongs to the caller. */
uint32_t *prime_list_build(uint32_t lim, size_t *count);

/* GGNFS .afb.0 layout, decoded in oracle/ggnfs_afb_format.txt:
 *   word 0        : uint32 n
 *   words 1..n    : uint32 primes[n]     (non-decreasing)
 *   words n+1..2n : uint32 roots[n]
 *   words 2n+1..  : 6 trailer words
 * Projective roots are encoded as root == p (only 4 such entries on the
 * C183 job, all with p < 12, so all below bkthresh). */
typedef struct {
    uint32_t  n;
    uint32_t *primes;   /* the walk modulus: p, or p^k for a prime power */
    uint32_t *roots;    /* root mod that modulus; >= modulus means projective */
    uint8_t  *logp;     /* las's log increment for this ideal, at a given scale */
    uint8_t  *ispow;    /* 1 if the modulus is p^k with k >= 2; may be NULL */
    int       maxbits;  /* source power bound; 0 if the input did not say */
    /* Set only by fb_validate(). The CUDA entry points reject a factor base
     * without this cookie, so a future loader cannot accidentally bypass the
     * prime-power precondition of pl_transform_enc(). Subsets produced by
     * fb_split_small()/fb_restrict() retain it. */
    uint32_t  validation_cookie;
} fb_t;

/* Is entry i a proper prime power? Uses the flag the loader already set --
 * every loader knows this for free, since it parses or generates the ladder.
 * Re-deriving it with primality tests cost 6.7 s per run over 11.5M ideals,
 * against a 55 ms sieve chain, until 2026-08-02. */
#define FB_ISPOW(fb, i)  ((fb)->ispow ? (fb)->ispow[i] : fb_is_proper_power((fb)->primes[i]))

int  fb_load(const char *path, fb_t *fb);          /* 0 on success */
/* fbgen/CADO-compatible text format: carries prime powers and the exact
 * per-ideal log increment, neither of which GGNFS's .afb.0 has. */
int  fb_load_cado(const char *path, double scale, fb_t *fb);
/* Fill logp with floor(log2(p)*scale + 0.5) for factor bases that did not come
 * with one (i.e. .afb.0). No-op if logp is already set. Returns 0 on success;
 * invalid/nonfinite scales, an unrepresentable 8-bit log, and allocation
 * failures return -1 without exposing a partially filled array. */
int fb_fill_logp(fb_t *fb, double scale);

/* Checked form of las's marginal factor-base log:
 *
 *   round(nexp   * log2(p) * scale)
 * - round(oldexp * log2(p) * scale)
 *
 * The output record is only eight bits. Keep the floating-point operations in
 * double until the complete range check has succeeded; converting NaN, an
 * infinity, or an out-of-range value to an integer is undefined or
 * implementation-dependent. Header-inline because rational factor-base
 * generation calls it once per ideal and should not pay a non-inline function
 * call on top of log2(). */
static inline int fb_log_delta_checked(uint32_t p, int nexp, int oldexp,
                                       double scale, uint8_t *out)
{
    double x, a, b, d;
    if (!out || p < 2 || nexp < 1 || oldexp < 0 || oldexp >= nexp ||
        !isfinite(scale) || scale <= 0.0)
        return -1;

    x = log2((double)p) * scale;
    /* For x >= 256, even one exponent step has an integer rounded delta of at
     * least 256. Reject before forming large nearly equal rounded values, which
     * would otherwise lose the low bits and could make an absurd scale appear
     * to have a small delta through cancellation. */
    if (!isfinite(x) || x < 0.0 || x >= 256.0) return -1;
    a = floor((double)nexp   * x + 0.5);
    b = floor((double)oldexp * x + 0.5);
    d = a - b;
    if (!isfinite(d) || d < 0.0 || d > 255.0) return -1;
    *out = (uint8_t)d;
    return 0;
}

/* Build the bucket-record slice map. Each record carries a 16-bit slice ID,
 * so at most 65,536 slices can be represented. On success, returns the actual
 * slice count and allocates both output arrays for the caller to free. The log
 * table is zero-padded to a power of two for the device-side mask. With valid
 * output arguments, failure returns -1, leaves both pointers NULL and
 * nslice_pow2 zero, and sets errno. */
int32_t fb_build_slices(const fb_t *fb, uint16_t **slice_out,
                        uint16_t **logp_tab_out, uint32_t *nslice_pow2);

/* The root transform is only valid for prime-power moduli; these keep that
 * assumption honest. fb_check_prime_powers returns the first bad index, or -1. */
int     fb_is_prime_power(uint32_t q);
int     fb_is_proper_power(uint32_t q);   /* p^k, k >= 2 */
int32_t fb_check_prime_powers(const fb_t *fb);

/* Validation policy for the modulus stream.
 *
 * EXTERNAL_PRIMES is the strict policy for GGNFS .afb.0: every modulus must
 * be prime, because that format has no prime-power metadata.
 * EXTERNAL_PRIME_POWERS accepts primes and proper powers, but verifies every
 * distinct modulus and checks ispow against the result. This is the policy for
 * text factor bases.
 * GENERATED_PRIME_POWERS is only for in-process builders whose prime entries
 * come directly from prime_list_build(); it still checks structure, roots,
 * ordering, flags, and every proper-power entry without re-sieving the full
 * generated prime range a second time. Never use it for file input.
 * PRECLASSIFIED_PRIME_POWERS trusts the caller's ispow flags for prime entries
 * and independently checks only the flagged proper powers. It has no production
 * caller: the CADO loader used it briefly and now uses EXTERNAL_PRIME_POWERS,
 * because moving the classification into the parser cost a Miller-Rabin per
 * line to save one sieve, and left the file with no witness but itself.
 * Like GENERATED_PRIME_POWERS, never use it for file input -- for a text
 * factor base the loader's reading of the file is the thing under test. */
typedef enum {
    FB_VALIDATE_EXTERNAL_PRIMES = 0,
    FB_VALIDATE_EXTERNAL_PRIME_POWERS = 1,
    FB_VALIDATE_GENERATED_PRIME_POWERS = 2,
    FB_VALIDATE_PRECLASSIFIED_PRIME_POWERS = 3
} fb_validate_policy_t;

/* Validate all invariants needed before a factor base reaches the lattice
 * transform. On success, records a validation cookie in fb and returns 0. On
 * failure, clears the cookie and returns -1. `source == NULL` suppresses the
 * diagnostic, which is useful for negative regression tests. */
int fb_validate(fb_t *fb, fb_validate_policy_t policy, const char *source);
int fb_is_transform_validated(const fb_t *fb);
void fb_free(fb_t *fb);

/* Deterministic primality over the WHOLE 32-bit range -- not probabilistic and
 * not a bound to check before calling. {2, 7, 61} is a verified witness set for
 * every n < 4,759,123,141, which covers uint32 with room to spare.
 *
 * Header-inline because two callers that share nothing else both need it: the
 * relation gate has to reject a composite masquerading as a prime factor, and
 * the --qlist parser has to reject a composite special-q. Both were previously
 * satisfied by weaker checks that a composite passes -- exact division and a
 * root test respectively -- so neither could see the failure that matters. */
static inline int bench_is_prime32(uint32_t n)
{
    static const uint32_t witness[3] = { 2, 7, 61 };
    uint32_t d = n - 1;
    int s = 0, i;
    if (n < 2) return 0;
    if (!(n & 1)) return n == 2;
    if (n % 3 == 0) return n == 3;
    while (!(d & 1)) { d >>= 1; s++; }
    for (i = 0; i < 3; i++) {
        uint64_t a = witness[i] % n, x = 1, p = a;
        uint32_t e = d;
        int r, composite;
        if (!a) continue;                 /* n divides the witness: n <= 61 */
        while (e) { if (e & 1) x = x * p % n; p = p * p % n; e >>= 1; }
        if (x == 1 || x == (uint64_t)n - 1) continue;
        composite = 1;
        for (r = 1; r < s; r++) {
            x = x * x % n;
            if (x == (uint64_t)n - 1) { composite = 0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

/* CADO's byte scale and survivor allowance, derived rather than hardcoded.
 * `maxlog2` is log2 of the largest norm over the sieve rectangle -- what
 * norm_setup already computes as (log2M - bias). See las-norms.cpp:237.
 *   scale = (255 - 1)/maxlog2, quantised to a multiple of 1/40
 *   lambda = given, or CADO's automatic 0.3 + mfb/lpb
 *   r     = min(maxlog2 - 1/scale, lambda*lpb)      <- our "allowance"
 *   bound = (uint32_t)(r*scale + 1)
 *
 * Quantising the scale is what makes it stable across a band, which matters
 * because the factor-base logs are baked at load time and CANNOT track a scale
 * that moves. We derive the scale ONCE, from the first special-q of the band,
 * and reuse it for every q after -- an assumption, not a checked invariant.
 * Nothing re-derives it per q. See prototype.md. */
/* 1 = norm_setup prints its per-q diagnostic; the band loop clears it after
 * the first special-q. */
extern int norm_verbose;

double las_scale(double maxlog2);
/* CADO's rule. Used only when --lambda0/--lambda1 asks for it explicitly. */
double las_allowance(double maxlog2, double scale, double lambda,
                     unsigned lpb, unsigned mfb);
/* OUR rule, and the default: mfb plus the slack our own approximation needs.
 * Neither GGNFS's lambda nor CADO's transfers to this tool's survivor gate --
 * see the comment on the definition for the measurements. */
double sieve_allowance(double maxlog2, double scale, unsigned mfb);

/* Strict decimal parser shared by the production CLI and CPU tests. It
 * accepts the complete strtod grammar, but rejects empty strings, trailing
 * characters, under/overflow, NaN and infinities. The output is unchanged on
 * failure. */
int bench_parse_finite_double(const char *text, double *out);

/* Strict unsigned decimal parser shared by command-line options and tests.
 * Only ASCII digits are accepted: signs, whitespace, prefixes and trailing
 * characters are rejected. The output is unchanged on failure. */
int bench_parse_u64_decimal(const char *text, uint64_t *out);

/* Strict parser for --qrange. Accepts MIN:MAX and the open-ended MIN: form,
 * consumes the complete token, rejects signs/overflow/junk, and requires
 * MIN <= MAX when MAX is present. Outputs are unchanged on failure. */
int bench_parse_qrange(const char *text, uint64_t *qmin, uint64_t *qmax);

/* Build the exact integer survivor bound used by the kernels:
 *
 *     bound = trunc(scale * allowance + 1)
 *
 * Both operands must be finite, scale must be positive and representable by
 * norm_t's float field, allowance must be nonnegative, and the truncated value
 * must not exceed CINIT. On failure, *bound_out is zeroed. `source == NULL`
 * suppresses diagnostics for negative tests. */
int sieve_bound_checked(double scale, double allowance, uint32_t CINIT,
                        uint32_t *bound_out, const char *source);
/* Restrict to bkthresh <= p < fb_bound, compacting in place. GGNFS truncates
 * the algebraic FB at the special-q; pass fb_bound = q to match it. Returns 0,
 * or -1 if the caller bypassed factor-base validation. */
int  fb_restrict(fb_t *fb, uint32_t bkthresh, uint32_t fb_bound);
/* Copy out the entries with p < bkthresh (line-sieved, not bucketed).
 * Call BEFORE fb_restrict, which compacts fb in place. */
int  fb_split_small(const fb_t *fb, uint32_t bkthresh, fb_t *small);

/* ---- q-lattice -------------------------------------------------------- */

/* Reduced basis of the q-lattice: a0*b1 - a1*b0 = +/- q. */
typedef struct { int64_t a0, a1, b0, b1; uint64_t q; } qlat_t;

/* Gauss-reduce the lattice generated by (q,0) and (rho,1). */
void qlat_build(qlat_t *L, uint64_t q, uint64_t rho, double skew);

/* ---- polynomial and norm initialisation ------------------------------- */

typedef struct {
    int    deg;
    double c[BENCH_NCOEFF];   /* algebraic coefficients, as doubles           */
    double skew, y0, y1;      /* doubles are fine for norms (a logarithm)...  */
    char   y0s[80], y1s[80];  /* ...but roots need Y0,Y1 exactly. See rfb.c.  */
    char   cs[BENCH_NCOEFF][80]; /* ...and so does TRIAL DIVISION, which factors */
                              /* the exact integer F(a,b), not its log. c0 is */
                              /* 147 bits on this job, so the double is only  */
                              /* good to 53 of them. See bigint.cuh.          */

    /* Sieve parameters carried by a GGNFS .job file. Zero means ABSENT, which
     * is the normal case for a CADO .poly -- the caller then falls back to a
     * derived value or refuses. These were being parsed by the human and typed
     * back in on the command line, eight flags transcribed from a file already
     * open in this process.
     *
     * lambda is the one parameter that does NOT transfer between conventions:
     * GGNFS's is a multiple of log2(lim), CADO's a multiple of lpb. These
     * fields hold the GGNFS reading, because that is what a .job file means by
     * them; job_allowance_bits() below does the conversion. Getting it
     * backwards TIGHTENS the bound and silently costs relations. */
    uint32_t rlim, alim;        /* factor-base bounds,  side 0 / side 1       */
    uint32_t lpbr, lpba;        /* large-prime bounds,  bits                  */
    uint32_t mfbr, mfba;        /* max cofactor bits                          */
    double   rlambda, alambda;  /* GGNFS units: multiples of log2(lim)        */
} poly_t;

/* GGNFS lambda -> bits of tolerated cofactor. `lambda * log2(lim)`, which is
 * what the GGNFS test-sieve's own "Suggested rlambda: mfbr / log2(rlim)"
 * confirms. Returns 0 when either input is absent. */
double job_allowance_bits(double lambda, uint32_t lim);

/* 1 if f has a root mod 2 (affine or projective), so 2 can divide the
 * algebraic norm and a factor base without p = 2 is truncated. 0 if it cannot,
 * in which case the absence of p = 2 is correct and not a missing --cadofb. */
int poly_has_root_mod2(const poly_t *P);

/* Build the rational factor base: G(x) = Y1*x + Y0 has one root per prime,
 * r = -Y0/Y1 mod p, with p | Y1 encoded as r == p. Nothing on disk holds this
 * -- GGNFS and CADO both regenerate it every run. */
int rfb_build(const poly_t *P, uint32_t lim, int maxbits, double scale, fb_t *fb);

int poly_load(const char *path, poly_t *P);

/* Everything the apply kernel needs to compute log2|F(a,b)| in fp32.
 * See poly.c for why the homogeneous normalisation is mandatory. */
typedef struct {
    float d[BENCH_NCOEFF];       /* c_k A^k B^(deg-k) / M, all O(1)          */
    float ua, ub, va, vb;        /* u = ua*i + ub*j ; v = va*i + vb*j        */
    float log2M;                 /* log2 of the normalisation constant       */
    float bias;                  /* log2(q) on the sq side only, else 0      */
    float scale;                 /* las's per-side byte scale; 1.0 = bits    */
    int   deg;
    /* fp64 fallback state. The fp32 Horner has ample dynamic range (that is
     * what the normalisation buys) but NOT ample precision near a root line of
     * F, where the terms are O(1) and cancel to something tiny. Measured on
     * this job: 144 positions in a 63,497-position band along the three real
     * root lines round to the wrong sieve-log value, error -3.31 to +2.57
     * units, BOTH SIGNS -- so both false survivors and lost relations. Cells
     * that cancel are recomputed from these. */
    double dd[BENCH_NCOEFF];     /* the same normalised coefficients, fp64    */
    double A, B;                 /* u = a/A, v = b/B                          */
    int64_t a0, a1, b0, b1;      /* so (a,b) can be formed exactly in int64   */
} norm_t;

/* Recompute a cell in fp64 when the fp32 Horner cancels below this fraction of
 * the sum of |terms|. 2^-11 leaves ~13 good fp32 bits, i.e. a relative error
 * around 1e-4 -- far tighter than the ~0.3 needed to round the sieve log
 * correctly, and it fires on well under 1 cell in 1000.
 *
 * Overridable at compile time (-DNORM_CANCEL_TOL=0.0f / =1.0f) only to PRICE
 * the hybrid: 0 disables the fp64 path, 1 forces every cell through it, and the
 * two together bracket what the guard costs. Neither is a correct setting --
 * 0 reinstates the fp32 error documented above. */
#ifndef NORM_CANCEL_TOL
#define NORM_CANCEL_TOL 4.9e-4f
#endif

double norm_acc_fp64(const norm_t *N, int32_t i, uint32_t j);

/* Conservative log2 bound for the exact homogeneous norm: every one of the
 * deg+1 terms is at most 2^log2M over the rectangle. Trial division constructs
 * the full norm before removing the special-q, so N->bias is not subtracted. */
double norm_exact_bound_bits(const norm_t *N);
int norm_fits_exact(const norm_t *N, unsigned bits);

void  norm_setup(norm_t *N, const poly_t *P, const qlat_t *L,
                 int logI, uint32_t J, double scale, int is_sqside);
float norm_target_host(const norm_t *N, int32_t i, uint32_t j);

/* ---- CPU reference ---------------------------------------------------- */

/* Brute-force check that the Franke-Kleinjung walk enumerates exactly the
 * lattice points {x in [0,I*J) : i == r*j mod p}, for small p. Returns the
 * number of primes checked; aborts on mismatch. */
int  verify_walk(int logI, uint32_t J, int nprimes);

/* Count updates the walk produces for the given FB, single-threaded.
 * This is the ground truth the GPU fill must reproduce exactly. */
uint64_t verify_count_updates(const fb_t *fb, const qlat_t *L,
                              int logI, uint32_t J,
                              int log_region, uint32_t *per_region);

/* ---- optional BOINC integration -------------------------------------- */

/* These wrappers are no-ops in the normal build.  When HAVE_BOINC is set,
 * bench_boinc_init() configures the BOINC runtime for a GPU application,
 * filename arguments are resolved from BOINC logical names, progress is
 * reported monotonically, and bench_boinc_finish() terminates through the
 * BOINC API. */
int  bench_boinc_init(void);
int  bench_boinc_resolve_path(const char *option, const char *logical_name,
                              const char **resolved_name);
void bench_boinc_fraction_done(double fraction_done);
int  bench_boinc_finish(int status);

/* ---- GPU entry points ------------------------------------------------- */

typedef struct {
    int      logI;          /* 15 for I15e */
    uint32_t J;             /* 16384 for I15e */
    int      log_region;    /* bucket region = 2^log_region positions */
    int      record_bytes;  /* 2, 4 or 8 */
    int      fill_mode;     /* see FILL_* below */
    int      threads;       /* threads per block */
    int      blocks;        /* 0 = auto (6 per SM) */
    /* Fill gets its OWN grid AND its own block width, because its optimum is
     * nothing like the rest of the pipeline's. Measured on 5070, 4090 and 5090:
     * fill wants MANY NARROW blocks -- 1152 x 32 -- while transform, intersect,
     * TD, resieve and the cofactor kernels are tuned at 256. Sharing --threads
     * meant the fill optimum was unreachable without wrecking five other
     * stages, so it stayed at the 144 x 256 that finding 51 found by sweeping
     * blocks alone at a fixed 256 threads.
     *
     * Both are ABSOLUTE, not per-SM: the 1152-block knee is the same on a
     * 48-SM 5070 and a 170-SM 5090, so SM count is the wrong axis. Past the
     * knee it is flat (<=1.7% to 9216 blocks on two of three cards), so
     * overshooting is cheap and undershooting is not. See RESULTS.md
     * finding 52. */
    int      fill_blocks;   /* 0 = auto (FILL_BLOCKS_DEFAULT) */
    int      fill_threads;  /* 0 = auto (FILL_THREADS_DEFAULT) */
    int      reps;          /* timing repetitions */
    int      verify;        /* run CPU cross-check */
    /* ---- Path 2 ---- */
    int      stage;         /* STAGE_* below */
    int      cell_bits;     /* 16 = safe (doc default), 8 = unsafe, for cost   */
    int      norm_mode;     /* NORM_* below */
    int      apply_atomic;  /* 1 = smem atomicAdd, 0 = racy plain += (probe)   */
    int      apply_threads; /* threads per apply block (0 = same as --threads) */
    int      small_sieve;   /* 1 = line-sieve p < bkthresh into the region      */
    int      side;          /* 1 = algebraic (special-q side), 0 = rational     */
    double   allowance;     /* bits of cofactor tolerated (lambda*lpb)         */
    double   scale;         /* las byte scale for this side; 1.0 = raw bits    */
    const char *dump;       /* write the region in las byte convention here     */
    int32_t  probe_i;            /* gate 5: read back this cell after apply  */
    uint32_t probe_j;            /* 0xFFFFFFFF = no probe                    */
    const char *cadofb;     /* fbgen/CADO text factor base (has powers)        */
    const char *survbits;   /* write a 1-bit-per-position survivor bitmap here  */
    int      not_both_even; /* apply las's not_both_even filter (see k_apply)   */
    const char *other_bits; /* the OTHER side's bitmap; enables device intersect */
    const char *emit;       /* write the compacted survivor list here           */
    int      td;            /* run exact norms + trial division on survivors    */
    int      ab_resieve;    /* re-run the settled layout A/B resieve experiment  */
    int      resieve_sweep; /* sweep resieve unroll depth and summary granularity */
    /* ---- both-sides pipeline: side 1 uses scale/allowance/lpb/mfb/lim ---- */
    int      pipeline;      /* run both sides in one process                    */
    /* Which side carries the special-q. 1 (algebraic) for a GNFS job, which is
     * every job this was built against; 0 (rational) for an SNFS job whose
     * rational side is the hard one -- there the algebraic coefficients are
     * tiny and the rational norms carry the difficulty, so the q belongs on
     * the rational side and mfbr is the one asking for 3LP.
     *
     * This is NOT the same thing as `side` above, which selects which single
     * side the benchmark harness measures. The pipeline runs both. */
    int      sq_side;       /* side carrying the special-q  [1 = algebraic]     */
    int      verbose_q;     /* print a line per special-q rather than a summary */
    const char *qlist;      /* file of `q rho` pairs; one special-q per line     */
    uint64_t qmin, qmax;    /* --qrange: generated prime band; qmax=0 is open   */
    int      cofactor;      /* split the cofactors inline, cross-q queue        */
    int      cof_rounds;    /* rho requeue rounds                               */
    uint32_t cof_budget;    /* rho iterations in the first round                */
    uint64_t target_rels;   /* stop the band once this many relations exist     */
    int      cof_ecm;       /* use ECM instead of rho                           */
    uint32_t ecm_b1;        /* ECM stage-1 bound                                */
    uint32_t ecm_b2;        /* ECM stage-2 bound (0 disables stage 2)            */
    uint32_t ecm_curves;    /* ECM curves attempted per round                   */
    uint32_t nq_max;        /* stop after this many q (0 = all)                 */
    int      td_verify;     /* run the factors x cofactor == norm reconstruction */
    double   scale0, allowance0;
    uint32_t lim0, lpb0, mfb0;
    const char *relations;  /* complete relations, needing no cofactorisation   */
    const char *candidates; /* the cofactorisation batch                        */
    uint32_t lim;           /* factor-base bound for this side (rlim / alim)    */
    uint32_t lpb;           /* large-prime bound, in bits                       */
    uint32_t mfb;           /* max cofactor bits carried into cofactorisation   */
    const char *cofgate;    /* CADO cofactor file to gate trial division against */
    const char *emit_cof;   /* write (a, b, cofactor) for every survivor here    */
    /* ---- checkpoint / resume / clean stop; see ckpt.h and STATUS.md 12a ---- */
    /* The factor-base convention, carried here purely so the resume
     * fingerprint can see it: dropping --maxbits under-divides every norm for
     * the rest of a run, which startup refuses outright but a resume could
     * not otherwise detect. */
    int      fb_maxbits;
    int      resume;        /* continue an existing NAME.part from its sidecar  */
    int      restart;       /* --restart: discard that .part instead of resuming */
    const char *stopfile;   /* --stop-file: stop cleanly once this path exists  */
    /* Loaded from the sidecar when resume is set. The scale/allowance the band
     * was originally derived with travel in cfg.scale/scale0/allowance*, which
     * bench_main.cu overwrites from the checkpoint before deriving its own --
     * re-deriving them from a later q would move the survivor gate mid-run. */
    uint64_t resume_q, resume_rho;
    unsigned long long resume_rel_bytes, resume_cand_bytes;
    unsigned long long resume_nrel, resume_nq;
} bench_cfg_t;

#define FILL_ATOMIC   0     /* (a) direct global atomicAdd per record   */
#define FILL_TWOLEVEL 1     /* (c) smem-staged, flush full cache lines  */

/* Fill's grid -- both numbers ABSOLUTE, NOT scaled by SM count.
 *
 * 1152 x 32 is the knee on all three cards swept (5070, 4090, 5090), which is
 * the useful part: one geometry, not a per-card constant. Past 1152 blocks it
 * is FLAT -- 9216 blocks buys a further 1.7% on the 4090, 0.04% on the 5090
 * and 3.8% on the 5070 (whose own run-to-run spread is ~3%). Below it the cost
 * is steep: 288 blocks is 15-38% worse. Overshooting is nearly free,
 * undershooting is not, so this sits at the knee rather than below it.
 *
 * NOTE THE SIGN AGAINST FINDING 51, which recorded the 4090 DEGRADING 38% from
 * 144 to 768 blocks and called it an Ada-specific penalty. Both are real and
 * they are not in conflict: that sweep held threads at 256, this one at 32. At
 * 256 the 4090 gets worse with more blocks; at 32 it gets better. The
 * architecture-specific block response in finding 51 -- Ada degrading,
 * Blackwell flat -- was itself an artifact of the fixed 256, and dissolves at
 * 32 where all three cards behave alike.
 *
 * Against the previous 144 x 256 default: 7.5% (5070), 8.8% (4090), 16.7%
 * (5090) off fill.
 *
 * WHY 32 THREADS, AND WHY A SEPARATE FLAG. Sweeping blocks at a fixed 256
 * threads -- which is all finding 51 did -- found a 144-block minimum and read
 * it as a hard saturation point. That was an artifact of never varying the
 * block WIDTH. Width is NOT a free parameter: at a constant 576 blocks the
 * 5090 measures 2.716 / 2.711 / 3.147 / 4.289 ms at 64 / 128 / 256 / 512
 * threads, so 256 is 16% off the optimum and 512 is 58% off. Between 64 and
 * 128 it flattens (2.716 vs 2.711), which is the only band measured in which
 * thread count is uncritical. NOTE 64, not 32: 576 x 32 was never run, so
 * flatness is established down to 64 and ASSUMED below it. The 1152-block row
 * shows the two axes interact strongly, so do not extrapolate -- if you need
 * 32-thread behaviour at some other block count, measure it.
 *
 * At the SHIPPED 1152 blocks the gap is wider still: the 5090 measures 2.633 ms
 * at 32 threads against 3.239 at 256, a 23% penalty. The width cost grows with
 * block count rather than washing out, and 1152 x 256 (3.239) is even worse
 * than 576 x 256 (3.147) -- at 256 threads more blocks still hurts, which is
 * the finding-51 behaviour, while at 32 threads more blocks helps.
 *
 * That 23% is what --fill-threads buys, and it is why raising
 * FILL_BLOCKS_DEFAULT alone would not do: --threads also drives transform,
 * intersect, TD, resieve and the cofactor kernels, all tuned at 256, so fill's
 * optimum is unreachable without moving theirs. Measured, not inferred -- the
 * 1152 x 256 cell was run specifically to test whether the flag earns its
 * keep.
 *
 * It is a work-granularity result -- fine chunks balance the tail -- and it
 * does NOT support the L2 mechanisms: capacity was already dead (finding 51),
 * write-combining decay was that finding's surviving candidate, and one
 * geometry fitting cards with 48, 72 and 96 MB of L2 argues against both.
 *
 * Measured at logI 14, J 8192, 8192 buckets, 77.4M records, on k_fill_atomic.
 * The optimum plausibly moves with bucket and record count, which is not yet
 * measured -- override with --fill-blocks/--fill-threads for a new job shape.
 *
 * Applies to k_fill_atomic, the shipping path, and to nothing else by default.
 * k_fill_l1 takes an explicit --fill-blocks but defaults to FILL_L1_BLOCKS
 * (never swept, different write pattern). k_fill_segmented is a different
 * algorithm in the experimental section and stays on the per-SM grid; pushing
 * an unmeasured constant onto it would be inventing a number. k_fill_l2 is
 * data-driven (one block per super-bucket) and has no grid to tune. */
#define FILL_BLOCKS_DEFAULT  1152
#define FILL_THREADS_DEFAULT 32

/* k_fill_l1's grid, frozen at what it has always run. See the launch site. */
#define FILL_L1_BLOCKS 144

/* A practical ceiling for user-selected grids. The device kernels use 64-bit
 * grid indices and strides, so this is no longer an arithmetic requirement;
 * it rejects accidental billion-block launches and keeps every accepted value
 * comfortably inside the launch range of the supported devices. */
#define BENCH_BLOCKS_MAX (1 << 20)

/* Keep the widening operation available to CPU-only correctness gates too.
 * Casting either operand after the multiplication would be too late: the
 * 32-bit product may already have wrapped to zero. */
#if defined(__CUDACC__)
static __host__ __device__ __forceinline__ uint64_t
bench_grid_product_u64(uint32_t a, uint32_t b)
#else
static inline uint64_t bench_grid_product_u64(uint32_t a, uint32_t b)
#endif
{
    return (uint64_t)a * (uint64_t)b;
}

#if defined(__CUDACC__)
/* CUDA's grid dimensions are 32-bit, but their product need not fit in 32
 * bits. Keep the cast BEFORE multiplication in one shared helper so a future
 * grid-stride kernel cannot reintroduce a zero or wrapped stride. */
static __device__ __forceinline__ uint64_t bench_grid_thread_x(void)
{
    return bench_grid_product_u64(blockIdx.x, blockDim.x) + threadIdx.x;
}

static __device__ __forceinline__ uint64_t bench_grid_stride_x(void)
{
    return bench_grid_product_u64(gridDim.x, blockDim.x);
}
#endif

#define STAGE_FILL    0
#define STAGE_BOTH    1
#define STAGE_APPLY   2     /* fill still runs; only apply is reported */

#define NORM_CONST    0     /* flat initial value -- isolates apply cost */
#define NORM_HORNER   1     /* real per-position log2|F(a,b)| in fp32    */

/* Runs transform -> plattice -> fill -> apply and prints a timing breakdown.
 * Returns 0 on success. */
/* One special-q of the band: the prime and a root of f mod it. las prints the
 * pair; the oracle captures carry it in their `# q = (q, rho, side)` headers. */
typedef struct { uint64_t q, rho; } qsel_t;

/* Validate and canonicalise one special-q selection before it is allowed to
 * build a q-lattice. Success reduces rho modulo q. The root is checked against
 * f on side 1 and against G = Y1*x + Y0 on side 0, using the exact decimal
 * coefficients rather than their double approximations. */
typedef enum {
    QSEL_VALID = 0,
    QSEL_ERR_Q_RANGE,
    QSEL_ERR_Q_COMPOSITE,
    QSEL_ERR_SIDE,
    QSEL_ERR_POLY,
    QSEL_ERR_NOT_ROOT
} qsel_validate_result_t;

qsel_validate_result_t qsel_validate(qsel_t *sel, const poly_t *P, int side);

/* Streaming special-q generator.  The factor base and special-q stream are
 * deliberately independent: lim bounds the ideals used to sieve and trial
 * divide, while this enumerates prime q and computes roots of the selected
 * side's polynomial on demand.  qmax == 0 means the uint32 representation
 * limit; nqmax == 0 means no count limit. */
typedef struct sqgen sqgen_t;
sqgen_t *sqgen_create(const poly_t *P, int side, uint64_t qmin, uint64_t qmax,
                      uint32_t nqmax);
/* 1 writes the next (q,rho), 0 means exhausted, -1 means failure. */
int sqgen_next(sqgen_t *G, qsel_t *out);
void sqgen_free(sqgen_t *G);

/* Both sides in one process, over a band of special-q: sieve, intersect,
 * trial-divide and classify each side against the shared two-sided bitmap,
 * then join. This is the path that becomes the siever; run_bench stays the
 * measurement harness. */
int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                 const fb_t *fb0, const fb_t *fbs0,
                 const qsel_t *qlist, uint32_t nq, sqgen_t *qgen,
                 const poly_t *P, const bench_cfg_t *cfg);

/* Cofactorise a batch written by --candidates and emit the relations it
 * yields. A separate entry point on purpose: the emitted corpus is a far
 * better test of the splitter than one special-q would be. */
/* Verify an emitted relation file against the polynomial: every recorded factor
 * divides its norm exactly, both norms rebuild to 1, every prime within its lpb.
 * Gates what the cofactoriser EMITTED, which the pre-split gate cannot see. */
int check_relations(const char *path, const poly_t *poly, uint32_t lpb0,
                    uint32_t lpb1);

/* The same gate over the first and last `n` relations only, for resume: a
 * multi-gigabyte .part cannot be rescanned at every restart. Proves the
 * polynomial and nothing else -- the checkpoint fingerprint covers the sieve
 * parameters, which a valid-looking line cannot. Returns bad lines, or -1. */
/* `limit` bounds the sample to the checkpointed prefix, since a torn final line
 * past it is normal after a crash and is about to be truncated away; 0 means
 * the whole file. */
int check_relations_sample(const char *path, const poly_t *poly, uint32_t lpb0,
                           uint32_t lpb1, uint32_t n, unsigned long long limit,
                           uint32_t *checked);

int run_cofac(const char *path, const char *out, uint32_t lim0, uint32_t lpb0,
              uint32_t lim1, uint32_t lpb1, int rounds, uint32_t budget,
              int blocks, int threads, int ecm, uint32_t ecm_b1,
              uint32_t ecm_b2, uint32_t ecm_curves);

int run_bench(const fb_t *fb, const fb_t *small, const qlat_t *L,
              const poly_t *P, const bench_cfg_t *cfg);

/* Replay one region's bucket records on the CPU into a 16-bit cell array.
 * This is the ground truth for the apply kernel, including the norm init and
 * the threshold test. Returns the number of survivors found. */
uint32_t verify_apply_region(const uint32_t *records, uint32_t nrec,
                             const uint16_t *slice_logp,
                             const norm_t *N, int logI, int log_region,
                             uint32_t region, int norm_mode,
                             uint32_t Cinit, uint32_t tconst,
                             const uint32_t *sp, const uint32_t *srt,
                             const uint32_t *sg, const uint16_t *slp,
                             uint32_t nsmall, uint16_t *cells_out);

/* Gate the root transform against its definition (a == r*b mod p), which the
 * walk check and the GPU/CPU cross-check both structurally cannot see. */
int verify_transform(const qlat_t *L, int ncheck);

/* The same set-equality gate driven by a real factor base: checks every entry
 * with q <= maxq against the definition. Returns entries checked, or -1. */
int verify_fb_transform(const fb_t *fb, const qlat_t *L, uint32_t maxq);

/* Does factor-base entry (q, r_enc) hit position (i,j)? Straight from the
 * definition -- no lattice algebra. Ground truth for the gates and for
 * `fbtest --trace`. */
int hits_def_pub(uint32_t q, uint32_t r_enc, const qlat_t *L,
                 int32_t i, int32_t j);

#ifdef __cplusplus
}
#endif
#endif
