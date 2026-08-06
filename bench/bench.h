/* Shared declarations for the standalone GPU bucket-fill benchmark. */
#ifndef CUDA_SIEVE_BENCH_H
#define CUDA_SIEVE_BENCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- factor base ------------------------------------------------------ */

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
} fb_t;

/* Is entry i a proper prime power? Uses the flag the loader already set --
 * every loader knows this for free, since it parses or generates the ladder.
 * Re-deriving it with primality tests cost 6.7 s per run over 11.5M ideals,
 * against a 55 ms sieve chain, until 2026-08-02. */
#define FB_ISPOW(fb, i)  ((fb)->ispow ? (fb)->ispow[i] : fb_is_proper_power((fb)->primes[i]))

int  fb_load(const char *path, fb_t *fb);          /* 0 on success */
/* CADO's makefb text format: carries prime powers and the exact per-ideal log
 * increment, neither of which GGNFS's .afb.0 has. */
int  fb_load_cado(const char *path, double scale, fb_t *fb);
/* Fill logp with floor(log2(p)*scale + 0.5) for factor bases that did not come
 * with one (i.e. .afb.0). No-op if logp is already set. */
void fb_fill_logp(fb_t *fb, double scale);

/* The root transform is only valid for prime-power moduli; these keep that
 * assumption honest. fb_check_prime_powers returns the first bad index, or -1. */
int     fb_is_prime_power(uint32_t q);
int     fb_is_proper_power(uint32_t q);   /* p^k, k >= 2 */
int32_t fb_check_prime_powers(const fb_t *fb);
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
/* Restrict to bkthresh <= p < fb_bound, compacting in place. GGNFS truncates
 * the algebraic FB at the special-q; pass fb_bound = q to match it. */
void fb_restrict(fb_t *fb, uint32_t bkthresh, uint32_t fb_bound);
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
    double c[8];              /* algebraic coefficients, as doubles           */
    double skew, y0, y1;      /* doubles are fine for norms (a logarithm)...  */
    char   y0s[80], y1s[80];  /* ...but roots need Y0,Y1 exactly. See rfb.c.  */
    char   cs[8][80];         /* ...and so does TRIAL DIVISION, which factors */
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
    float d[8];                  /* c_k A^k B^(deg-k) / M, all O(1)          */
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
    double dd[8];                /* the same normalised coefficients, fp64    */
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

/* ---- GPU entry points ------------------------------------------------- */

typedef struct {
    int      logI;          /* 15 for I15e */
    uint32_t J;             /* 16384 for I15e */
    int      log_region;    /* bucket region = 2^log_region positions */
    int      record_bytes;  /* 2, 4 or 8 */
    int      fill_mode;     /* see FILL_* below */
    int      threads;       /* threads per block */
    int      blocks;        /* 0 = auto (6 per SM) */
    /* Fill gets its OWN grid, and it is an absolute block count rather than a
     * per-SM one. Measured minimum is 144 blocks on both a 48-SM 5070 (3/SM)
     * and a 128-SM 4090 (1.1/SM) -- the same work in flight on cards 2.7x
     * apart in width, so SM count is the wrong axis entirely. Overshooting
     * costs 27% on the 4090 at its old 768. See RESULTS.md finding 51. */
    int      fill_blocks;   /* 0 = auto (FILL_BLOCKS_DEFAULT) */
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
    const char *cadofb;     /* CADO makefb text factor base (has powers)       */
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
    uint64_t qmin, qmax;    /* --qrange: band from the SPECIAL-Q SIDE's FB      */
    int      cofactor;      /* split the cofactors inline, cross-q queue        */
    int      cof_rounds;    /* rho requeue rounds                               */
    uint32_t cof_budget;    /* rho iterations in the first round                */
    uint64_t target_rels;   /* stop the band once this many relations exist     */
    int      cof_ecm;       /* use ECM stage 1 instead of rho                   */
    uint32_t ecm_b1;        /* ECM stage-1 bound                                */
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
} bench_cfg_t;

#define FILL_ATOMIC   0     /* (a) direct global atomicAdd per record   */
#define FILL_TWOLEVEL 1     /* (c) smem-staged, flush full cache lines  */

/* Fill's grid, in blocks -- absolute, NOT scaled by SM count. At 256 threads
 * this is 36,864 threads in flight. Both the 5070 and the 4090 minimise here,
 * and both degrade in BOTH directions from it: the 5070 is flat out to 1536
 * and falls off below 96, the 4090 climbs steadily to +38% by 768. Fill is not
 * parallelism-limited past this point; more concurrency thrashes whatever
 * fixed-rate resource it is actually waiting on rather than saturating it.
 *
 * Measured at logI 14, J 8192, 8192 buckets, 77.4M records, on k_fill_atomic.
 * The optimum plausibly moves with bucket and record count, which is not yet
 * measured -- override with --fill-blocks when characterising a new job shape.
 *
 * Applies to k_fill_atomic (the shipping path) and k_fill_l1. NOT to
 * k_fill_segmented, which is a different algorithm in the experimental section
 * and was never swept; pushing an unmeasured constant onto it would be
 * inventing a number. It stays on the per-SM grid. k_fill_l2 is data-driven
 * (one block per super-bucket) and has no grid to tune. */
#define FILL_BLOCKS_DEFAULT 144

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

/* Both sides in one process, over a band of special-q: sieve, intersect,
 * trial-divide and classify each side against the shared two-sided bitmap,
 * then join. This is the path that becomes the siever; run_bench stays the
 * measurement harness. */
int run_pipeline(const fb_t *fb1, const fb_t *fbs1,
                 const fb_t *fb0, const fb_t *fbs0,
                 const qsel_t *qlist, uint32_t nq,
                 const poly_t *P, const bench_cfg_t *cfg);

/* Cofactorise a batch written by --candidates and emit the relations it
 * yields. A separate entry point on purpose: the emitted corpus is a far
 * better test of the splitter than one special-q would be. */
/* Verify an emitted relation file against the polynomial: every recorded factor
 * divides its norm exactly, both norms rebuild to 1, every prime within its lpb.
 * Gates what the cofactoriser EMITTED, which the pre-split gate cannot see. */
int check_relations(const char *path, const poly_t *poly, uint32_t lpb0,
                    uint32_t lpb1);

int run_cofac(const char *path, const char *out, uint32_t lim0, uint32_t lpb0,
              uint32_t lim1, uint32_t lpb1, int rounds, uint32_t budget,
              int blocks, int threads, int ecm, uint32_t ecm_b1,
              uint32_t ecm_curves);

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
