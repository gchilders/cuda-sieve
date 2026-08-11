/* CLI for the standalone bucket-fill benchmark. */
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void usage(void)
{
    printf(
"usage: bench [options]\n"
"\n"
"RUNNING A JOB\n"
"  A typical run needs six flags. Everything else has a right answer that is\n"
"  derived from the polynomial or read from the .job file, and is printed:\n"
"\n"
"    bench --pipeline --cofactor --poly JOB.job --fb1 JOB.roots1 \\\n"
"          --logI 14 --qrange 15000000: --target-rels 65000000 \\\n"
"          --relations msieve.dat\n"
"\n"
"  --pipeline       BOTH SIDES in one process: sieve, intersect, TD, classify,\n"
"                   join, and emit relations + cofactorisation candidates.\n"
"                   Runs one fixed configuration; the harness options below\n"
"                   are refused rather than silently ignored\n"
"  --poly PATH      polynomial. A GGNFS .job works directly, and its rlim,\n"
"                   alim, lpbr/lpba, mfbr/mfba and lambdas are USED -- they do\n"
"                   not need repeating below. A CADO .poly carries none of\n"
"                   those, so state them or let them derive\n"
"  --fb1 PATH       native fbgen factor base. REQUIRED to emit relations: the\n"
"                   GGNFS .afb.0 has neither p = 2 nor prime powers\n"
"  --cadofb PATH    legacy alias for --fb1 (CADO files remain compatible)\n"
"  --logI N         log2 of sieve width I      [15]   (gnfs-lasieve4I14e -> 14)\n"
"  --J N            sieve height J             [2^(logI-1), CADO's convention]\n"
"  --relations F    write complete relations here (GGNFS/msieve format)\n"
"  --cofactor       split the cofactors INLINE, in a cross-q device queue;\n"
"                   --relations then holds every relation, not just TD's\n"
"\n"
"BAND SELECTION\n"
"  --qrange MIN:MAX generate every prime special-q and every affine root of\n"
"                   the selected side's polynomial in that inclusive range.\n"
"                   MIN: generates upward until --target-rels or --nq stops it\n"
"  --sq-side S      which side carries the special-q: 1 = algebraic (GNFS,\n"
"                   the default), 0 = rational. An SNFS job whose algebraic\n"
"                   coefficients are tiny puts the difficulty on the rational\n"
"                   side, and the q and the 3LP mfb go there with it\n"
"  --qlist FILE     band of special-q: `q rho` per line (# comments ok);\n"
"                   q must be prime, rho is reduced mod q, bad lines are fatal\n"
"  --q N / --rho N  a single special-q          [120000011]  (las -v prints rho)\n"
"  --nq N           stop after N special-q from the list or generated range\n"
"  --target-rels N  stop once N relations have been collected. Checked at\n"
"                   flush boundaries, so it overshoots by under a flush.\n"
"                   Pair with --qrange MIN: to sieve upward until satisfied\n"
"\n"
"PARAMETERS  (precedence: this flag > .job file > derived from the poly)\n"
"  The byte scale and survivor allowance are ALWAYS derived, as las does, from\n"
"  the largest norm over the sieve rectangle. Stating one overrides it.\n"
"  --rlim N / --alim N  factor base bounds, side 0 / side 1\n"
"  --lpb N / --lpb0 N   large-prime bound in bits  [32 side 1, 31 side 0]\n"
"  --mfb N / --mfb0 N   max cofactor bits          [92 side 1, 60 side 0]\n"
"  --allowance B / --allowance0 B   survivor cofactor BITS, overriding the\n"
"                   derived default. The default is mfb + the slack our own\n"
"                   byte-quantised survivor test needs (~1.5 bits), NOT the\n"
"                   .job file's lambda -- that is calibrated to GGNFS's gate\n"
"                   and does not transfer. It is reported, not applied\n"
"  --lambda0/1 L    opt back into CADO's rule: lambda in CADO units, i.e.\n"
"                   multiples of lpb [0 = CADO's automatic 0.3 + mfb/lpb].\n"
"                   Looser than the derived default on every job measured\n"
"  --scale S / --scale0 S   las byte scale per side\n"
"  --fbbound N      truncate FB at this p      [alim]  (GGNFS truncates at q)\n"
"  --bkthresh N     bucket-sieve p >= this     [1<<logI]\n"
"  --region N       log2 bucket region size    [14]  (16384 16-bit cells, 32 KB)\n"
"  --maxbits N      prime powers below 2^N      [logI]\n"
"\n"
"COFACTORISATION\n"
"  --cof-rounds N   rho requeue rounds, budget doubling each time\n"
"                   [6 for --cofac; 2 for --pipeline --cofactor]\n"
"  --cof-budget N   rho iterations in the first round\n"
"                   [4096 for --cofac; 65536 for --pipeline --cofactor]\n"
"  --cof-ecm        ECM instead of Pollard-Brent rho; stage 1 alone loses,\n"
"                   while tuned stage 2 is near rho at matched yield\n"
"  --ecm-b1 N       ECM stage-1 bound                              [1000]\n"
"  --ecm-b2 N       ECM D=30 stage-2 bound; 0 disables stage 2         [0]\n"
"  --ecm-curves N   ECM curves attempted per round                 [16]\n"
"  --candidates F   write the cofactorisation batch here\n"
"  --cofac FILE     cofactorise a batch written by --candidates and emit the\n"
"                   relations to --relations; needs no sieving\n"
"\n"
"CHECKING\n"
"  --check-relations F  verify an emitted relation file: every factor divides,\n"
"                   both norms rebuild to 1, every prime within its lpb\n"
"  --cofgate FILE   gate the cofactors against CADO's own (a b cof0 cof1);\n"
"                   under --pipeline this runs in the first-q validation\n"
"  --verbose-q      print a line per special-q instead of a band summary\n"
"\n"
"RUNTIME\n"
"  --threads N      threads per block, multiple of 32  [256]\n"
"  --blocks N       0 = auto (6 per SM)        [0]\n"
"  --fill-blocks N  fill only; 0 = auto (1152, absolute -- NOT per SM) [0]\n"
"  --fill-threads N fill only; 0 = auto (32), else a multiple of 32 in\n"
"                   [32,1024]. Independent of --threads: fill wants many\n"
"                   narrow blocks, the other kernels do not.            [0]\n"
"  --blocking-sync  yield the CPU while waiting on the GPU instead of spinning.\n"
"                   CUDA busy-waits by default, so a host thread that is 90%%\n"
"                   idle still pegs a core; this frees it, at the cost of a\n"
"                   wakeup latency per synchronisation\n"
"\n"
"BENCHMARK HARNESS  (single-side measurement; REFUSED under --pipeline)\n"
"  These reproduce the numbers in RESULTS.md. They select configurations that\n"
"  were measured and rejected, so they are not knobs to tune a real run with.\n"
"  --fb PATH        GGNFS .afb.0 factor base   [../oracle/input.job.afb.0]\n"
"  --side S         1 = algebraic (special-q side), 0 = rational  [1]\n"
"  --record-bytes N 2 | 4 | 8                  [4]\n"
"  --mode M         atomic | twolevel          [atomic]  (twolevel lost by 2.7x)\n"
"  --stage S        fill | both | apply        [both]\n"
"  --cells N        16 | 8 bits per sieve cell [16]  (8 is unsafe; cost only)\n"
"  --norm M         horner | const             [horner]\n"
"  --apply-mode M   atomic | plain             [atomic]  (plain is racy)\n"
"  --apply-threads N  threads per apply block  [512]\n"
"  --reps N         timing repetitions         [3]\n"
"  --verify         run the CPU cross-check (slow)\n"
"  --no-smallsieve  skip the p < bkthresh line sieve\n"
"  --not-both-even  apply las's filter: i,j both even can never survive\n"
"  --survbits FILE  write a survivor bitmap (1 bit/position, x order)\n"
"  --other-bits F   the other side's bitmap -> device intersect+gcd+compact\n"
"  --emit FILE      write the compacted survivor list (x, a, b) here\n"
"  --emit-cof FILE  write (a, b, cofactor, bits) for every survivor here\n"
"  --td             exact norms + trial division on the survivors\n"
"  --ab-resieve     re-run the settled layout A/B resieve experiment (slow)\n"
"  --resieve-sweep  sweep the resieve unroll depth and summary granularity\n"
"  --dump PATH      write the sieve region in las byte convention\n"
"  --probe i,j      read back that cell after apply (gate 5)\n");
}

/* Cofactoriser bounds. The representation is narrow ON PURPOSE -- split
 * factors are emitted as a single uint32 limb and both cofactors are mz<3> --
 * so a bound outside that design does not degrade, it silently truncates: a
 * 33-bit factor is stored as its low 32 bits and the relation stops
 * reconstructing its own norm. Refuse.
 *
 * This MUST run after the .job file has been read. It used to sit inline in
 * main() ahead of poly_load, so it only ever saw command-line values -- and
 * since --lpb/--mfb are already range-checked by the parser, the interesting
 * cases (mfba 120 from a job file, or a 4LP mfb/lpb ratio) were exactly the
 * ones it could not see. The RUNBOOK's own SNFS recipe takes mfbr and lpbr
 * from the file. */
static int check_cofactor_bounds(const bench_cfg_t *cfg, uint32_t alim,
                                 uint32_t rlim, int cof_rounds,
                                 uint32_t cof_budget)
{
    const uint32_t lpb1 = cfg->lpb ? cfg->lpb : 32;
    const uint32_t mfb1 = cfg->mfb ? cfg->mfb : 92;
    int bad = 0;
    if (lpb1 > 32 || cfg->lpb0 > 32) {
        fprintf(stderr, "lpb %u / side-0 %u: split factors are emitted as one"
                " 32-bit limb, so lpb > 32 truncates them\n", lpb1, cfg->lpb0);
        bad = 1;
    }
    if (mfb1 > 96) {
        fprintf(stderr, "side-1 mfb %u: the cofactor is 3 limbs, so a residual"
                " above 96 bits loses its high limbs\n", mfb1);
        bad = 1;
    }
    if (cfg->mfb0 > 96) {
        fprintf(stderr, "side-0 mfb %u: the cofactor is 3 limbs, so a residual"
                " above 96 bits loses its high limbs\n", cfg->mfb0);
        bad = 1;
    }
    /* ceil(mfb/lpb) parts must fit in CF_MAXFAC. 3LP is what an SNFS job with
     * mfbr 88 / lpbr 31 asks for and is supported; 4LP is not, and it would
     * present as CF_OVERFLOW on the records that need it -- a partial yield
     * loss, not a failure -- so it is refused up front. */
    {
        const uint32_t p1 = (mfb1 + lpb1 - 1) / lpb1;
        const uint32_t p0 = (cfg->mfb0 + cfg->lpb0 - 1) / cfg->lpb0;
        if (p1 > 3 || p0 > 3) {
            fprintf(stderr, "mfb/lpb asks for %u parts on side 1 and %u on"
                    " side 0; the splitter handles at most 3\n", p1, p0);
            bad = 1;
        }
    }
    /* "prime by size" in mz_split rests on lim^2 > 2^lpb: below that a
     * composite could sit under lim^2 and be emitted as a prime. */
    {
        const uint64_t l1 = cfg->lim ? cfg->lim : alim;
        const uint64_t l0 = cfg->lim0 ? cfg->lim0 : rlim;
        if ((double)l1 * l1 <= ldexp(1.0, (int)lpb1)) {
            fprintf(stderr, "alim^2 (%.3g) <= 2^lpb: the prime-by-size test in"
                    " mz_split is unsound\n", (double)l1 * l1);
            bad = 1;
        }
        if ((double)l0 * l0 <= ldexp(1.0, (int)cfg->lpb0)) {
            fprintf(stderr, "rlim^2 (%.3g) <= 2^lpb0: the prime-by-size test in"
                    " mz_split is unsound\n", (double)l0 * l0);
            bad = 1;
        }
    }
    /* budget << r must not shift past the width, and a non-positive round
     * count would run no splitting at all yet still exit successfully. */
    if (cof_rounds < 1 || cof_rounds > 24) {
        fprintf(stderr, "--cof-rounds %d: must be 1..24 (budget << r overflows"
                " beyond that, and < 1 splits nothing)\n", cof_rounds);
        bad = 1;
    }
    if (cfg->cof_rounds < 1 || cfg->cof_rounds > 24) {
        fprintf(stderr, "pipeline cof-rounds %d: must be 1..24\n", cfg->cof_rounds);
        bad = 1;
    }
    if (!cfg->cof_ecm) {
        if (cfg->ecm_b2) {
            fprintf(stderr, "--ecm-b2 requires --cof-ecm\n"); bad = 1;
        }
        if (!cof_budget || (uint64_t)cof_budget << (cof_rounds - 1) > 0xFFFFFFFFull) {
            fprintf(stderr, "--cof-budget %u with %d rounds overflows uint32"
                    " (or is zero)\n", cof_budget, cof_rounds);
            bad = 1;
        }
        if (!cfg->cof_budget ||
            (uint64_t)cfg->cof_budget << (cfg->cof_rounds - 1) > 0xFFFFFFFFull) {
            fprintf(stderr, "pipeline cof-budget %u with %d rounds overflows"
                    " uint32 (or is zero)\n", cfg->cof_budget, cfg->cof_rounds);
            bad = 1;
        }
    } else {
        if (!cfg->ecm_curves) {
            fprintf(stderr, "--ecm-curves 0 attempts no curves\n"); bad = 1;
        }
        if (cfg->ecm_b1 < 2 || cfg->ecm_b1 > 1000000u) {
            fprintf(stderr, "--ecm-b1 %u: must be 2..1000000\n", cfg->ecm_b1);
            bad = 1;
        }
        if (cfg->ecm_b2 && (cfg->ecm_b1 < 30u ||
                            cfg->ecm_b2 <= cfg->ecm_b1 ||
                            cfg->ecm_b2 > 10000000u)) {
            fprintf(stderr, "--ecm-b2 %u: want 0 (disabled), or B1 < B2 <="
                    " 10000000 with B1 >= 30\n", cfg->ecm_b2);
            bad = 1;
        }
    }
    return bad;
}

int main(int argc, char **argv)
{
    const char *fbpath = "../oracle/input.job.afb.0";
    const char *polypath = "../oracle/c183.poly";
    bench_cfg_t cfg;
    uint64_t q = 120000011ull;          /* prime, mid-range of [50M,190M] */
    uint32_t bkthresh = 0, fbbound = 0;
    int fbbound_set = 0, scale_set = 0;
    uint64_t rho = 0;   /* 0 = synthetic; pass the real root of f mod q to match las */
    uint32_t rlim = 67100000;   /* side-0 factor base bound (input.job rlim) */
    uint32_t alim = 134200000;  /* side-1 factor base bound (input.job alim) */
    fb_t fb, fbs; qlat_t L; poly_t POLY;

    /* Defaults are the CONFIGURATION OF RECORD -- single-level atomic fill,
     * 4 B records, 2^14 regions -- not the design the doc started from. Two-
     * level lost by 2.7x (RESULTS.md finding 1) and 2^15 regions lost to 2^14
     * (finding 8); leaving those as defaults meant the commands in RESULTS
     * reproduced a path nobody would ship. */
    cfg.logI = 15; cfg.J = 16384; cfg.log_region = 14;
    cfg.record_bytes = 4; cfg.fill_mode = FILL_ATOMIC;
    cfg.threads = 256; cfg.blocks = 0; cfg.fill_blocks = 0; cfg.fill_threads = 0;
    cfg.reps = 3; cfg.verify = 0;
    cfg.stage = STAGE_BOTH; cfg.cell_bits = 16; cfg.norm_mode = NORM_HORNER;
    cfg.apply_atomic = 1; cfg.apply_threads = 0; cfg.allowance = 3.5 * 32.0;
    cfg.small_sieve = 1; cfg.side = 1;
    cfg.scale = 1.0; cfg.dump = NULL; cfg.cadofb = NULL;
    cfg.probe_i = 0; cfg.probe_j = 0xFFFFFFFFu;
    cfg.survbits = NULL; cfg.not_both_even = 0;
    cfg.other_bits = NULL; cfg.emit = NULL;
    cfg.td = 0; cfg.cofgate = NULL; cfg.emit_cof = NULL;
    cfg.lim = 0; cfg.lpb = 0; cfg.mfb = 0;   /* 0 == take the side's default */
    cfg.ab_resieve = 0; cfg.resieve_sweep = 0;
    /* Pipeline defaults are the WORKING configuration established 2026-08-04:
     * survivor bound 128 on side 1 and 132 on side 0 (128 on both loses one of
     * las's 37 relations at the parity q), and the full algebraic factor base
     * to alim, since truncating at the special-q costs relations outright. */
    cfg.pipeline = 0; cfg.sq_side = 1;
    cfg.scale0 = 1.925; cfg.allowance0 = 68.1;
    cfg.lim0 = 0; cfg.lpb0 = 31; cfg.mfb0 = 60;
    cfg.relations = NULL; cfg.candidates = NULL;
    cfg.qlist = NULL; cfg.nq_max = 0; cfg.verbose_q = 0; cfg.td_verify = 1;
    cfg.qmin = 0; cfg.qmax = 0; cfg.target_rels = 0;
    cfg.cofactor = 0; cfg.cof_rounds = 2; cfg.cof_budget = 65536;
    cfg.cof_ecm = 0; cfg.ecm_b1 = 1000; cfg.ecm_b2 = 0; cfg.ecm_curves = 16;
    int maxbits = 0, maxbits_set = 0;
    int allowance_set = 0, allowance0_set = 0, scale0_set = 0;
    const char *cofac_in = NULL;
    const char *check_rel = NULL;
    int blocking_sync = 0;
    /* Which values the COMMAND LINE supplied. Precedence is
     *      explicit flag  >  job file  >  derived  >  refuse
     * and these are what distinguishes the first level from the rest. A
     * default that merely looks like the job file's value is not the same
     * thing: the whole point is that a run says where each number came from. */
    int rlim_set = 0, alim_set = 0, lpb_set = 0, mfb_set = 0;
    int lpb0_set = 0, mfb0_set = 0, J_set = 0;
    double lambda0 = 0.0, lambda1 = 0.0;   /* 0 = CADO's automatic 0.3+mfb/lpb */
    int lambda0_set = 0, lambda1_set = 0;  /* asked for CADO's rule at all?    */
    int cof_rounds = 6;
    uint32_t cof_budget = 4096;
    int qrange_set = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fb") && i + 1 < argc) fbpath = argv[++i];
        else if (!strcmp(argv[i], "--logI") && i + 1 < argc) cfg.logI = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--J") && i + 1 < argc) { cfg.J = (uint32_t)strtoul(argv[++i], 0, 10); J_set = 1; }
        else if (!strcmp(argv[i], "--region") && i + 1 < argc) cfg.log_region = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bkthresh") && i + 1 < argc) bkthresh = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--fbbound") && i + 1 < argc) { fbbound = (uint32_t)strtoul(argv[++i], 0, 10); fbbound_set = 1; }
        else if (!strcmp(argv[i], "--q") && i + 1 < argc) q = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--rho") && i + 1 < argc) rho = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--record-bytes") && i + 1 < argc) cfg.record_bytes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            cfg.fill_mode = !strcmp(m, "atomic") ? FILL_ATOMIC : FILL_TWOLEVEL;
        }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cfg.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) cfg.blocks = atoi(argv[++i]);
        /* strtol, not atoi, for these two ALONE -- not a style preference. Both
         * treat 0 as "auto", and atoi maps any malformed argument to 0, so
         * `--fill-threads 64x` would run at the default, print no [--flag] tag,
         * and be indistinguishable from an unswept run. A sweep over a typo'd
         * list then reports N identical timings, which reads as flatness --
         * precisely the shape of the conclusion these flags exist to test. */
        else if (!strcmp(argv[i], "--fill-blocks") && i + 1 < argc) {
            char *e; long v = strtol(argv[++i], &e, 10);
            if (*e || e == argv[i]) {
                fprintf(stderr, "--fill-blocks: not a number: %s\n", argv[i]);
                return 1;
            }
            cfg.fill_blocks = (int)v;
        }
        else if (!strcmp(argv[i], "--fill-threads") && i + 1 < argc) {
            char *e; long v = strtol(argv[++i], &e, 10);
            if (*e || e == argv[i]) {
                fprintf(stderr, "--fill-threads: not a number: %s\n", argv[i]);
                return 1;
            }
            cfg.fill_threads = (int)v;
        }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) cfg.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) cfg.verify = 1;
        else if (!strcmp(argv[i], "--poly") && i + 1 < argc) polypath = argv[++i];
        else if (!strcmp(argv[i], "--stage") && i + 1 < argc) {
            const char *s = argv[++i];
            cfg.stage = !strcmp(s, "fill") ? STAGE_FILL
                      : !strcmp(s, "apply") ? STAGE_APPLY : STAGE_BOTH;
        }
        else if (!strcmp(argv[i], "--cells") && i + 1 < argc) cfg.cell_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--norm") && i + 1 < argc)
            cfg.norm_mode = !strcmp(argv[++i], "const") ? NORM_CONST : NORM_HORNER;
        else if (!strcmp(argv[i], "--apply-mode") && i + 1 < argc)
            cfg.apply_atomic = strcmp(argv[++i], "plain") ? 1 : 0;
        else if (!strcmp(argv[i], "--apply-threads") && i + 1 < argc)
            cfg.apply_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allowance") && i + 1 < argc)
            { cfg.allowance = atof(argv[++i]); allowance_set = 1; }
        else if (!strcmp(argv[i], "--no-smallsieve")) cfg.small_sieve = 0;
        else if (!strcmp(argv[i], "--side") && i + 1 < argc) cfg.side = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) { rlim = (uint32_t)strtoul(argv[++i], 0, 10); rlim_set = 1; }
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) { cfg.scale = atof(argv[++i]); scale_set = 1; }
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) cfg.dump = argv[++i];
        else if ((!strcmp(argv[i], "--fb1") || !strcmp(argv[i], "--cadofb")) && i + 1 < argc)
            cfg.cadofb = argv[++i];
        else if (!strcmp(argv[i], "--survbits") && i + 1 < argc) cfg.survbits = argv[++i];
        else if (!strcmp(argv[i], "--other-bits") && i + 1 < argc) cfg.other_bits = argv[++i];
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) cfg.emit = argv[++i];
        else if (!strcmp(argv[i], "--td")) cfg.td = 1;
        else if (!strcmp(argv[i], "--ab-resieve")) cfg.ab_resieve = 1;
        else if (!strcmp(argv[i], "--resieve-sweep")) cfg.resieve_sweep = 1;
        else if (!strcmp(argv[i], "--pipeline")) cfg.pipeline = 1;
        /* strtol, not atoi: atoi("rational") is 0, which is a LEGAL value here,
         * so a typo would silently configure the wrong side instead of being
         * rejected. Every other numeric flag in this parser avoids atoi for
         * the same reason. */
        else if (!strcmp(argv[i], "--sq-side") && i + 1 < argc) {
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (!end || *end || (v != 0 && v != 1)) {
                fprintf(stderr, "--sq-side %s: want 1 (algebraic) or 0"
                        " (rational)\n", argv[i]);
                return 1;
            }
            cfg.sq_side = (int)v;
        }
        else if (!strcmp(argv[i], "--cofac") && i + 1 < argc) cofac_in = argv[++i];
        else if (!strcmp(argv[i], "--cofactor")) cfg.cofactor = 1;
        else if (!strcmp(argv[i], "--cof-rounds") && i + 1 < argc) { cof_rounds = atoi(argv[++i]); cfg.cof_rounds = cof_rounds; }
        else if (!strcmp(argv[i], "--cof-budget") && i + 1 < argc) { cof_budget = (uint32_t)strtoul(argv[++i], 0, 10); cfg.cof_budget = cof_budget; }
        else if (!strcmp(argv[i], "--cof-ecm")) cfg.cof_ecm = 1;
        /* Deriving is now unconditional, so this is accepted and ignored
         * rather than rejected: it appears in RUNBOOK.md and in scripts, and
         * breaking those to make a point about a flag that now describes the
         * default helps nobody. The note fires so nobody keeps typing it. */
        else if (!strcmp(argv[i], "--auto-params"))
            fprintf(stderr, "note: --auto-params is the default now and is"
                            " ignored; drop it\n");
        else if (!strcmp(argv[i], "--blocking-sync")) blocking_sync = 1;
        else if (!strcmp(argv[i], "--lpb0") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 32) { fprintf(stderr, "--lpb0 %ld out of range 1..32\n", v); return 1; } cfg.lpb0 = (uint32_t)v; lpb0_set = 1; }
        else if (!strcmp(argv[i], "--mfb0") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 96) { fprintf(stderr, "--mfb0 %ld out of range 1..96\n", v); return 1; } cfg.mfb0 = (uint32_t)v; mfb0_set = 1; }
        /* Range-checked like --lpb0/--mfb0 above, and for the same reason. An
         * unchecked --lambda1 0.01 gives allowance 0.32, bound 1, and a band
         * that runs for hours and emits nothing with no diagnostic.
         *
         * Exactly 0 stays legal: it is the documented sentinel for "use CADO's
         * automatic", and scripts pass it to mean the default. The window
         * refused is (0, 0.5), which no real lambda occupies -- CADO's
         * automatic lands near 2-3 -- and anything above 8. This is a guard
         * against a typo, not a tuning opinion, so the ends are loose. */
        /* `_set` is tracked separately from the value because 0 is a MEANINGFUL
         * value here -- the documented sentinel for "use CADO's automatic" --
         * so `var > 0.0` cannot stand in for "the user asked for CADO's rule".
         * Testing the value instead routed --lambda1 0 to the derived default,
         * silently giving mfb+1.5 where the script asked for 0.3+mfb/lpb. */
        #define LAMBDA_ARG(flag, var, seen)                                    \
            else if (!strcmp(argv[i], flag) && i + 1 < argc) {                 \
                var = atof(argv[++i]); seen = 1;                               \
                if (var < 0.0 || (var > 0.0 && var < 0.5) || var > 8.0) {      \
                    fprintf(stderr, "%s %g: want 0 (CADO's automatic) or"      \
                            " 0.5..8\n", flag, var);                           \
                    return 1;                                                  \
                }                                                              \
            }
        LAMBDA_ARG("--lambda0", lambda0, lambda0_set)
        LAMBDA_ARG("--lambda1", lambda1, lambda1_set)
        #undef LAMBDA_ARG
        else if (!strcmp(argv[i], "--check-relations") && i + 1 < argc) check_rel = argv[++i];
        else if (!strcmp(argv[i], "--ecm-b1") && i + 1 < argc) cfg.ecm_b1 = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ecm-b2") && i + 1 < argc) cfg.ecm_b2 = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ecm-curves") && i + 1 < argc) cfg.ecm_curves = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--qlist") && i + 1 < argc) cfg.qlist = argv[++i];
        else if (!strcmp(argv[i], "--qrange") && i + 1 < argc) {
            unsigned long long lo, hi;
            const char *a = argv[++i];
            int got = sscanf(a, "%llu:%llu", &lo, &hi);
            if (got == 1 && strchr(a, ':')) { hi = 0; got = 2; }   /* "MIN:" = open */
            if (got != 2 || (hi && lo > hi)) {
                fprintf(stderr, "bench: --qrange wants MIN:MAX, or MIN: to"
                        " generate upward until --target-rels/--nq\n"); return 2;
            }
            cfg.qmin = lo; cfg.qmax = hi;   /* 0 = stream until target/nq */
            qrange_set = 1;
        }
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) cfg.nq_max = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-rels") && i + 1 < argc)
            cfg.target_rels = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--verbose-q")) cfg.verbose_q = 1;
        else if (!strcmp(argv[i], "--alim") && i + 1 < argc) { alim = (uint32_t)strtoul(argv[++i], 0, 10); alim_set = 1; }
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) { cfg.scale0 = atof(argv[++i]); scale0_set = 1; }
        else if (!strcmp(argv[i], "--allowance0") && i + 1 < argc) { cfg.allowance0 = atof(argv[++i]); allowance0_set = 1; }
        else if (!strcmp(argv[i], "--relations") && i + 1 < argc) cfg.relations = argv[++i];
        else if (!strcmp(argv[i], "--candidates") && i + 1 < argc) cfg.candidates = argv[++i];
        else if (!strcmp(argv[i], "--lpb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 32) { fprintf(stderr, "--lpb %ld out of range 1..32\n", v); return 1; } cfg.lpb = (uint32_t)v; lpb_set = 1; }
        else if (!strcmp(argv[i], "--mfb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 96) { fprintf(stderr, "--mfb %ld out of range 1..96\n", v); return 1; } cfg.mfb = (uint32_t)v; mfb_set = 1; }
        else if (!strcmp(argv[i], "--cofgate") && i + 1 < argc) cfg.cofgate = argv[++i];
        else if (!strcmp(argv[i], "--emit-cof") && i + 1 < argc) cfg.emit_cof = argv[++i];
        else if (!strcmp(argv[i], "--not-both-even")) cfg.not_both_even = 1;
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) {
            maxbits = atoi(argv[++i]); maxbits_set = 1;
        }
        else if (!strcmp(argv[i], "--probe") && i + 1 < argc) {
            int pi; unsigned pj;
            if (sscanf(argv[++i], "%d,%u", &pi, &pj) != 2) {
                fprintf(stderr, "--probe wants i,j\n"); return 1;
            }
            cfg.probe_i = pi; cfg.probe_j = pj;
        }
        else { usage(); return 1; }
    }
    if (cfg.cell_bits != 8 && cfg.cell_bits != 16) { usage(); return 1; }
    /* logI bounds FIRST: every default below shifts by it (bkthresh, the area
     * check, the probe range), and an out-of-range shift is undefined. */
    if (cfg.logI < 2 || cfg.logI > 20) { usage(); return 1; }
    if (!maxbits_set) maxbits = cfg.logI;
    if (maxbits < 1 || maxbits > 31) {
        fprintf(stderr, "--maxbits must be in [1,31]\n");
        return 1;
    }
    /* las's survivor bound is scale*lambda*lpb per side; ours is the same
     * quantity in unscaled bits (16-bit cells need no scale). */
    if (!allowance_set && cfg.side == 0) cfg.allowance = 2.35 * 31.0;
    if (!bkthresh) bkthresh = 1u << cfg.logI;
    if (!fbbound && cfg.side == 1 && !cfg.pipeline)
        fbbound = (q > 0xFFFFFFFFull) ? 0xFFFFFFFFu : (uint32_t)q;

    /* Bound log_region before ANY 1u << log_region: the shift is undefined for
     * >= 32 and UBSan flags --region 32 on the old ordering. */
    if (cfg.log_region < 1 || cfg.log_region > 30) {
        fprintf(stderr, "--region must be in [1,30] (got %d)\n", cfg.log_region);
        return 1;
    }
    /* An out-of-range probe silently ALIASES another cell -- --probe 16384,0
     * lands on the real (-16384,1) -- so it would certify a coordinate nobody
     * asked about. */
    if (cfg.probe_j != 0xFFFFFFFFu) {
        const int32_t half = 1 << (cfg.logI - 1);
        if (cfg.probe_i < -half || cfg.probe_i >= half || cfg.probe_j >= cfg.J) {
            fprintf(stderr, "--probe i,j out of range: i must be in [%d,%d) and"
                    " j in [0,%u)\n", -half, half, cfg.J);
            return 1;
        }
    }
    if ((uint64_t)(1u << cfg.logI) * cfg.J > 0x80000000ull) {
        fprintf(stderr, "I*J must fit in 31 bits (uint32 positions)\n"); return 1;
    }
    /* Two limits that used to be silent. A 2 B or 4 B record carries the
     * in-region offset in 16 bits, so a region above 2^16 wraps and every
     * record past the wrap lands on the wrong cell; and the fused small sieve
     * derives j from a single shift, which assumes a region lies inside one
     * j-row. Both produced plausible-looking output rather than an error. */
    if (cfg.record_bytes != 2 && cfg.record_bytes != 4 && cfg.record_bytes != 8) {
        fprintf(stderr, "--record-bytes must be 2, 4 or 8 (got %d): any other"
                " value falls through to the 8-byte kernel while allocating the"
                " requested size, which writes out of bounds\n", cfg.record_bytes);
        return 1;
    }
    if (cfg.fill_mode == FILL_TWOLEVEL && cfg.record_bytes != 4) {
        fprintf(stderr, "--mode twolevel only has a 4-byte level-2 kernel;"
                " --record-bytes %d would launch the wrong specialisation\n",
                cfg.record_bytes);
        return 1;
    }
    if (cfg.J == 0 || cfg.reps < 1 || cfg.threads < 32 || cfg.threads > 1024) {
        fprintf(stderr, "--J must be > 0, --reps >= 1, --threads in [32,1024]\n");
        return 1;
    }
    /* Same rule --apply-threads has enforced all along, and for the same class
     * of reason -- it was simply never applied to the block width every OTHER
     * kernel launches with. k_intersect_compact runs a warp-wide inclusive scan
     * under a hardcoded 0xffffffff mask and broadcasts the atomic base from
     * lane 31 (bench_kernels.cu:899-910). A partial final warp has no lane 31,
     * so the base every thread in it reads is undefined.
     *
     * --threads 33 was accepted and ran to completion, reporting "intersect
     * counted 262538 survivors, rank scan 270888". The independent second count
     * turned that into a band FAILED rather than bad relations, which is the
     * gate working -- but it costs a whole band to say what one modulo says
     * here. */
    if (cfg.threads & 31) {
        fprintf(stderr, "--threads must be a multiple of 32 (got %d): the"
                " intersection kernel scans under a full warp mask and reads"
                " its atomic base from lane 31, which a partial warp does not"
                " have\n", cfg.threads);
        return 1;
    }
    /* Fill has no warp-collective code, so a partial warp here would not be
     * wrong -- just wasteful, since the tail lanes are launched and idle. The
     * range matters more: 0 means "auto" so it can never be passed through, and
     * above 1024 every launch fails at runtime with a message that does not
     * mention this flag. */
    if (cfg.fill_threads != 0 &&
        (cfg.fill_threads < 32 || cfg.fill_threads > 1024
         || (cfg.fill_threads & 31))) {
        fprintf(stderr, "--fill-threads must be 0 (auto) or a multiple of 32 in"
                " [32,1024], got %d\n", cfg.fill_threads);
        return 1;
    }
    /* Bounded ABOVE, and the bound is not cosmetic. k_fill_atomic strides by
     * `uint32_t stride = gridDim.x * blockDim.x` (bench_kernels.cu:98). CUDA
     * accepts gridDim.x up to 2^31-1, so --fill-blocks 134217728 at 32 threads
     * launches fine and computes a stride of 2^32 == 0: the grid-stride loop
     * never advances and the device hangs until the watchdog fires. Products
     * that overflow to a nonzero value are worse -- primes get skipped or
     * walked twice and the run completes with a plausible smaller record count.
     * 1<<20 blocks is ~900x the measured knee, so nothing legitimate is lost. */
    if (cfg.fill_blocks < 0 || cfg.fill_blocks > (1 << 20)) {
        fprintf(stderr, "--fill-blocks must be in [0, %d] (0 = auto), got %d:"
                " the fill kernels stride by gridDim.x*blockDim.x in 32 bits\n",
                1 << 20, cfg.fill_blocks);
        return 1;
    }
    /* The grid-width query used to sit HERE. It now runs after the
     * --check-relations return further down, so that gate works on a box with
     * no GPU -- see the comment there. */
    /* The small sieve's warp tier strides by nwarps = threads >> 5. Below 32
     * that is ZERO -- an infinite loop on the device. A non-multiple of 32
     * leaves a partial warp whose lanes re-run warp-tier entries, double-adding
     * their logs. Neither is diagnosable from the output. */
    if (cfg.apply_threads != 0 &&
        (cfg.apply_threads < 32 || cfg.apply_threads > 1024
         || (cfg.apply_threads & 31))) {
        fprintf(stderr, "--apply-threads must be 0 (auto) or a multiple of 32 in"
                " [32,1024]: the small sieve strides by threads/32, so under 32"
                " hangs and a partial warp double-counts\n");
        return 1;
    }
    if ((uint64_t)(1u << cfg.logI) * cfg.J % (1u << cfg.log_region)) {
        fprintf(stderr, "I*J must divide evenly into 2^%d regions\n", cfg.log_region);
        return 1;
    }
    if (cfg.log_region > 16 && cfg.record_bytes < 8) {
        fprintf(stderr, "--region %d needs --record-bytes 8: a %d B record has"
                " only a 16-bit offset field\n", cfg.log_region, cfg.record_bytes);
        return 1;
    }
    if (cfg.log_region > cfg.logI && cfg.small_sieve) {
        fprintf(stderr, "--region %d > --logI %d: the fused small sieve assumes"
                " a region lies within one j-row.\n  Use --no-smallsieve, or a"
                " region <= 2^%d.\n", cfg.log_region, cfg.logI, cfg.logI);
        return 1;
    }
    /* Scale is free with 16-bit cells (see k_apply), but not unbounded: the
     * norm must fit under CINIT and each ideal's log must fit in the uint8
     * that fb_t carries. Refuse rather than saturate. */
    {
        const double cinit = (cfg.cell_bits == 16) ? 4096.0 : 255.0;
        /* c183's norm sizes, and only a sanity bound on a user-supplied
         * --scale in the single-side harness modes. The pipeline rechecks both
         * sides against their REAL derived maxnorm and lim after the scale is
         * derived; this one runs too early to see either. */
        const double maxnorm = (cfg.side == 1) ? 196.61 : 131.86;
        const double maxlogp = 27.0;           /* log2(alim) */
        if (cfg.scale * maxnorm > cinit) {
            fprintf(stderr, "--scale %.3f x log2(maxnorm) %.1f exceeds CINIT %.0f\n",
                    cfg.scale, maxnorm, cinit);
            return 1;
        }
        if (cfg.scale * maxlogp > 255.0) {
            fprintf(stderr, "--scale %.3f x log2(p) %.0f exceeds the 8-bit"
                    " per-ideal log\n", cfg.scale, maxlogp);
            return 1;
        }
    }


    /* Must precede any call that creates the CUDA context. */
    if (blocking_sync && cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) {
        fprintf(stderr, "warning: could not set blocking sync\n");
    }

    printf("=== cuda-sieve bucket-fill benchmark ===\n");



    if (poly_load(polypath, &POLY) != 0) return 1;
    printf("polynomial %s: algebraic degree %d, skew %.4g\n",
           polypath, POLY.deg, POLY.skew);

    /* ---- sieve parameters: explicit flag > job file > derived > refuse ----
     *
     * A GGNFS .job file already carries rlim, alim, lpbr/lpba, mfbr/mfba and
     * the two lambdas. They used to be retyped onto the command line -- eight
     * flags transcribing a file this process has open -- which is both tedious
     * and a place for the two to silently disagree.
     *
     * Whatever is taken from the file is PRINTED, because a parameter that
     * appears from nowhere is worse than one that has to be typed. */
    {
        int used = 0;
        #define JOB_TAKE(cond, dst, src, name)                                 \
            do { if ((cond) && (src)) {                                        \
                     (dst) = (src);                                            \
                     printf("%s%s %u", used++ ? ", " : "  job file: ",         \
                            name, (unsigned)(src));                            \
                 } } while (0)
        JOB_TAKE(!rlim_set, rlim,     POLY.rlim, "rlim");
        JOB_TAKE(!alim_set, alim,     POLY.alim, "alim");
        JOB_TAKE(!lpb0_set, cfg.lpb0, POLY.lpbr, "lpbr");
        JOB_TAKE(!lpb_set,  cfg.lpb,  POLY.lpba, "lpba");
        JOB_TAKE(!mfb0_set, cfg.mfb0, POLY.mfbr, "mfbr");
        JOB_TAKE(!mfb_set,  cfg.mfb,  POLY.mfba, "mfba");
        #undef JOB_TAKE
        if (used) printf("\n");

        /* The .job file's lambdas are REPORTED, not applied.
         *
         * They are calibrated to GGNFS's survivor gate, and that calibration
         * does not transfer: on the SNFS job, same q range, gnfs-lasieve4I14e
         * loses 17.3% of its yield going from 91.8 to 87.5 bits where we lose
         * 0.07% going to 88.0. Importing the number inherits another tool's
         * tuning and, here, 13% of the trial-division input for one relation
         * in ten thousand. CADO's automatic is no better a source -- on this
         * job it is looser still (97.3 bits).
         *
         * So the default comes from sieve_allowance(), derived from our own
         * quantisation. The file's value is printed anyway, because it is
         * useful to see what the job's author intended and how far it sits
         * from what we chose. --allowance / --allowance0 override; so do
         * --lambda0 / --lambda1 if CADO's rule is wanted. */
        if (POLY.alambda > 0.0 && alim)
            printf("  job file: alambda %.3g -> %.2f bits (side 1), reported"
                   " only; the allowance is derived below\n",
                   POLY.alambda, job_allowance_bits(POLY.alambda, alim));
        if (POLY.rlambda > 0.0 && rlim)
            printf("  job file: rlambda %.3g -> %.2f bits (side 0), reported"
                   " only; the allowance is derived below\n",
                   POLY.rlambda, job_allowance_bits(POLY.rlambda, rlim));
    }

    /* CADO's convention, and the only value we have ever wanted. Deriving it
     * removes one more flag whose right answer is a function of another. */
    if (!J_set && cfg.logI > 1) cfg.J = 1u << (cfg.logI - 1);

    /* HERE, not before poly_load: lpb/mfb/lim may all have come from the .job
     * file just read, and validating only the command line was validating the
     * half that the argument parser had already range-checked. */
    if (check_cofactor_bounds(&cfg, alim, rlim, cof_rounds, cof_budget)) return 1;

    /* --check-relations is pure host code -- it re-derives both norms from the
     * polynomial and divides. It cannot run before this point because lpb0/lpb
     * may have just come from the .job file, and it must run before the device
     * query below, because a fatal "cannot query the CUDA device" on a machine
     * with no GPU is exactly what stops an emitted relation file from being
     * verified on the box that has the file rather than the card. */
    if (check_rel)
        return check_relations(check_rel, &POLY, cfg.lpb0,
                              cfg.lpb ? cfg.lpb : 32) ? 1 : 0;

    /* Grid width is 6 resident blocks per SM, so it MUST come from the device.
     * It was hardcoded 48*6 = 288 -- 48 being this box's 5070. On an 82-SM
     * 3090 that is 3.5 blocks/SM, i.e. 58% of the occupancy every tuning
     * decision in RESULTS.md assumed, which reads as the bigger card being
     * slower. Resolve it once here; run_pipeline() and bench_kernels.cu both
     * treat a nonzero cfg.blocks as final, so this reaches every launch.
     *
     * Queried and printed UNCONDITIONALLY, on stdout with every other startup
     * line. Printing only in the default case hid the number from exactly the
     * --blocks A/B that wants it, stderr dropped it from any run redirected
     * with `> log`, and a failed query used to leave cfg.blocks at 0 so the
     * three `48 * 6` fallbacks silently reinstated the 5070's SM count with no
     * diagnostic. There is no useful run on a device we cannot query, so a
     * failure here is fatal rather than a default.
     *
     * Device 0 always. There is no cudaSetDevice and no --device flag; select
     * another GPU with CUDA_VISIBLE_DEVICES and confirm it on this line. */
    {
        cudaDeviceProp prop;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
            fprintf(stderr, "bench: cannot query the CUDA device -- refusing to"
                    " fall back to a hardcoded grid width\n");
            return 1;
        }
        /* L2 size rides on the card name, on BOTH branches. The fill geometry
         * is an absolute block count that is the same on every card measured,
         * which is itself the interesting thing -- L2 capacity does not explain
         * it (finding 51 already killed capacity, and 1152 x 32 fits cards with
         * 48, 72 and 96 MB alike). It is printed so a sweep log carries the
         * number instead of someone reconstructing it later from a card name.
         * Printing it only in the default branch would hide it from exactly the
         * --blocks A/B that wants it -- the defect the paragraph above this
         * block records having already fixed once. */
        if (cfg.blocks == 0) {
            cfg.blocks = prop.multiProcessorCount * 6;
            printf("grid: %d SMs x 6 = %d blocks (%s, %d MB L2)\n",
                   prop.multiProcessorCount, cfg.blocks, prop.name,
                   prop.l2CacheSize >> 20);
        } else {
            printf("grid: %d blocks on %d SMs (%s, %d MB L2)  [--blocks; auto"
                   " would be %d]\n", cfg.blocks, prop.multiProcessorCount,
                   prop.name, prop.l2CacheSize >> 20,
                   prop.multiProcessorCount * 6);
        }
        /* Reported, NOT resolved. cfg.fill_blocks/fill_threads stay 0 for
         * "auto" all the way to the launch sites, because k_fill_atomic and
         * k_fill_l1 have different defaults (the 1152 x 32 sweep was run on the
         * former only) and collapsing 0 here would erase the distinction they
         * need. Every ?: at a launch site is therefore live, not a dead
         * fallback -- and there is exactly one place per kernel that knows its
         * own default. */
        printf("grid: %d x %d for fill (absolute, not per SM)%s%s\n",
               cfg.fill_blocks  ? cfg.fill_blocks  : FILL_BLOCKS_DEFAULT,
               cfg.fill_threads ? cfg.fill_threads : FILL_THREADS_DEFAULT,
               cfg.fill_blocks  ? "  [--fill-blocks]"  : "",
               cfg.fill_threads ? "  [--fill-threads]" : "");
    }

    /* ---- both sides in one process ---- */
    if (cofac_in) {
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.lim)  cfg.lim  = alim;
        if (!cfg.lim0) cfg.lim0 = rlim;
        return run_cofac(cofac_in, cfg.relations, cfg.lim0, cfg.lpb0,
                         cfg.lim, cfg.lpb, cof_rounds, cof_budget,
                         cfg.blocks ? cfg.blocks : 48 * 6, cfg.threads,
                         cfg.cof_ecm, cfg.ecm_b1 ? cfg.ecm_b1 : 1000,
                         cfg.ecm_b2,
                         cfg.ecm_curves ? cfg.ecm_curves : 16) ? 1 : 0;
    }

    /* Only the pipeline reads these, so outside it they were silent no-ops --
     * the very thing the harness_only block below calls an error in the other
     * direction. A run quoted as relation-targeted or lambda-tuned when it was
     * neither is the same defect either way. */
    if (!cfg.pipeline) {
        static const char *pipeline_only[] = {
            "--target-rels", "--lambda0", "--lambda1", "--sq-side", NULL
        };
        int nbad = 0;
        for (int i = 1; i < argc; i++)
            for (int k = 0; pipeline_only[k]; k++)
                if (!strcmp(argv[i], pipeline_only[k])) {
                    fprintf(stderr, "bench: %s applies to --pipeline only\n",
                            argv[i]);
                    nbad++;
                }
        if (nbad) { fprintf(stderr, "  add --pipeline, or drop them.\n"); return 2; }
    }

    if (cfg.pipeline) {
        fb_t fb1, fbs1, fb0, fbs0;
        qsel_t *ql = NULL;
        sqgen_t *qgen = NULL;
        uint32_t nq = 0, capq = 0;
        int prc;
        /* Options that belong to the measurement harness only. The pipeline
         * runs ONE configuration -- the one every gate was closed against:
         * single-level atomic fill, 4 B records, 16-bit cells, the fp32/fp64
         * Horner norm, both stages, no repetitions, side 1 then side 0. Taking
         * a flag that changes none of that and running anyway is how a run gets
         * quoted as something it was not, so it is an error rather than a
         * silent no-op. */
        static const char *harness_only[] = {
            "--record-bytes", "--mode", "--cells", "--norm", "--apply-mode",
            "--stage", "--reps", "--verify", "--side", "--dump", "--probe",
            "--survbits", "--other-bits", "--emit", "--emit-cof", "--td",
            "--ab-resieve", "--resieve-sweep", NULL
        };
        int nbad = 0;
        for (int i = 1; i < argc; i++)
            for (int k = 0; harness_only[k]; k++)
                if (!strcmp(argv[i], harness_only[k])) {
                    fprintf(stderr, "bench --pipeline: %s applies to the"
                            " benchmark harness, not the pipeline\n", argv[i]);
                    nbad++;
                }
        if (nbad) {
            fprintf(stderr, "  drop --pipeline to use them, or drop them.\n");
            return 2;
        }
        /* The pipeline uses the FULL factor base up to each side's lim; the
         * special-q stream is separate and may continue beyond that lim.
         *
         * This comment used to say that truncating at the special-q "costs
         * relations outright -- 30 of 1,851 cofactors at the parity q differ
         * by exactly one prime in (q, alim]". That measurement was right and
         * the conclusion was wrong: those 30 are not lost, they are found
         * again when the special-q reaches that larger prime. Counting what a
         * single q loses, without asking whether a later q recovers it, is the
         * error -- and GGNFS's FB_bound truncation is deliberate duplicate
         * avoidance, not a limitation of its design.
         *
         * Measured on the SNFS job: 1.82 sq-side primes in range per relation
         * and 72% re-found when that prime is later sieved as q, giving 1.34
         * finds per unique relation -- ~25% of raw output is duplicates, and
         * we pay full trial division and cofactorisation on every one.
         * Calibrated against msieve's dedup of the c151: 10,594,292 duplicates
         * in 67,165,877 relations, 15.8%, which back-solves to the same 0.73
         * re-find probability. See task #26. */
        if (cfg.qlist && qrange_set) {
            fprintf(stderr, "bench --pipeline: --qlist and --qrange both give"
                    " the band; pass one\n");
            return 2;
        }
        if (!cfg.qlist && !qrange_set && !rho) {
            fprintf(stderr, "bench --pipeline: needs a real root of %s mod q."
                    " Pass --rho, --qlist, or --qrange for a band.\n"
                    "  A synthetic root is fine for a sieve microbenchmark but"
                    " not for a path that emits relations.\n",
                    cfg.sq_side ? "f" : "G");
            return 2;
        }
        if (!fbbound_set) fbbound = alim;
        if (!cfg.lim)  cfg.lim  = fbbound;
        if (!cfg.lim0) cfg.lim0 = rlim;
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.mfb)  cfg.mfb  = 92;

        /* --qlist is read HERE, before the factor base, for two reasons. It
         * needs nothing from the base, so a missing or malformed list should
         * fail before a 29 MB parse rather than after it. And the scale
         * derivation below reads ql[0] to build its lattice: this block used
         * to sit ~90 lines further down, past that point, so --qlist left nq
         * at 0 and the derivation silently fell back to the hardcoded default
         * q -- a c183-sized lattice for whatever job was actually running,
         * with the wrong scale applied to the entire band. --qrange was
         * unaffected only because it happens to populate ql[] earlier. */
        if (cfg.qlist) {
            FILE *f = fopen(cfg.qlist, "r");
            char line[256];
            if (!f) { perror(cfg.qlist); return 1; }
            unsigned long lno = 0;
            while (fgets(line, sizeof line, f)) {
                unsigned long long qq, rr;
                const char *s = line;
                lno++;
                while (*s == ' ' || *s == '\t') s++;
                if (*s == '#' || *s == '\n' || *s == '\r' || !*s) continue;
                /* Was `continue`. A typo'd or wrongly-columned q-list then ran
                 * as a SHORTER band with no diagnostic at all -- the count
                 * printed below was the only hint, and only if you knew what it
                 * should have been. */
                if (sscanf(s, "%llu %llu", &qq, &rr) != 2) {
                    fprintf(stderr, "%s:%lu: expected `q rho`, got: %s",
                            cfg.qlist, lno, line);
                    fclose(f); free(ql); return 1;
                }
                /* pipe_check_root is NOT a substitute for these two, which is
                 * why they belong here rather than there. Everything is 0 mod
                 * 1, so q = 1 passes the root test outright and then reaches a
                 * full-multiplicity division by 1. A composite q with a genuine
                 * root mod q also passes, gets divided out of every norm on its
                 * side, and is emitted as though it were a prime relation
                 * factor -- a wrong relation that reconstructs.
                 *
                 * --qrange cannot reach either case: its generator admits only
                 * primes. A hand-written --qlist is the only way in, which is
                 * exactly why nothing downstream looks. */
                if (qq < 2 || (qq >> 32)) {
                    fprintf(stderr, "%s:%lu: q = %llu is not in [2, 2^32)\n",
                            cfg.qlist, lno, qq);
                    fclose(f); free(ql); return 1;
                }
                if (!bench_is_prime32((uint32_t)qq)) {
                    fprintf(stderr, "%s:%lu: q = %llu is composite\n",
                            cfg.qlist, lno, qq);
                    fclose(f); free(ql); return 1;
                }
                /* Canonicalise. This is NOT about getting a different lattice:
                 * <(q,0),(rho,1)> and <(q,0),(rho+kq,1)> generate the same one,
                 * and Gauss reduction converges to the same basis either way --
                 * measured, an unreduced rho reproduces the golden 37 relations
                 * exactly. What it is not safe against is SIZE. qlat_build
                 * casts rho straight to int64_t, so a rho at or above 2^63
                 * arrives negative, and its reduction is bounded at 200
                 * iterations. One modulo here removes both. */
                rr %= qq;
                if (nq == capq) {
                    capq = capq ? capq * 2 : 256;
                    ql = (qsel_t *)realloc(ql, (size_t)capq * sizeof(qsel_t));
                    if (!ql) { fclose(f); return 1; }
                }
                ql[nq].q = qq; ql[nq].rho = rr; nq++;
                if (cfg.nq_max && nq >= cfg.nq_max) break;
            }
            fclose(f);
            if (!nq) { fprintf(stderr, "%s: no `q rho` pairs\n", cfg.qlist); return 1; }
            printf("band: %u special-q from %s\n", nq, cfg.qlist);
        }

        /* --qrange is a stream of prime ideals, independent of the factor
         * base.  lim bounds the small ideals used by sieve/TD; it is not an
         * upper bound on q.  Cache only the first generated pair because norm
         * and byte-scale setup needs its lattice before run_pipeline starts;
         * the rest are pulled on demand so MIN: can run until the relation
         * target without allocating a list through 2^32. */
        if (qrange_set) {
            int qr;
            if (!cfg.qmax && !cfg.target_rels && !cfg.nq_max) {
                fprintf(stderr, "bench: open --qrange MIN: needs --target-rels"
                                " or --nq as a stopping condition\n");
                return 2;
            }
            qgen = sqgen_create(&POLY, cfg.sq_side, cfg.qmin, cfg.qmax,
                                cfg.nq_max);
            if (!qgen) return 1;
            ql = (qsel_t *)malloc(sizeof(*ql));
            if (!ql) { sqgen_free(qgen); return 1; }
            qr = sqgen_next(qgen, ql);
            if (qr < 0) {
                fprintf(stderr, "bench: special-q generator failed before its"
                                " first result\n");
                free(ql); sqgen_free(qgen); return 1;
            }
            if (qr == 0) {
                if (cfg.qmax)
                    fprintf(stderr, "bench: no affine special-q roots in"
                            " [%llu, %llu]\n",
                            (unsigned long long)cfg.qmin,
                            (unsigned long long)cfg.qmax);
                else
                    fprintf(stderr, "bench: no affine special-q roots from"
                            " %llu through the 32-bit q range\n",
                            (unsigned long long)cfg.qmin);
                free(ql); sqgen_free(qgen); return 1;
            }
            nq = 1;
            if (cfg.qmax)
                printf("band: generated prime special-q roots on side %d in"
                       " [%llu, %llu]\n", cfg.sq_side,
                       (unsigned long long)cfg.qmin,
                       (unsigned long long)cfg.qmax);
            else
                printf("band: generating prime special-q roots on side %d from"
                       " %llu upward\n", cfg.sq_side,
                       (unsigned long long)cfg.qmin);
        }

        /* Derive the byte scale and survivor allowance from the polynomial,
         * as las does. This is UNCONDITIONAL. It used to sit behind
         * --auto-params, whose "off" state was not a mode but a frozen copy of
         * the c183's derived constants -- correct for exactly one polynomial
         * and quietly wrong for every other. An explicit --scale/--allowance
         * still overrides, which is the override that was actually wanted.
         *
         * The scale depends on the largest norm over the sieve rectangle,
         * which needs a q-lattice. The q list or streaming generator has
         * already supplied its first pair above. Derive the real scale first,
         * then load the factor base once with that scale. */
        {
            qlat_t L0; norm_t N1, N0;
            double m1, m0;
            uint64_t q0  = nq ? ql[0].q   : q;
            uint64_t rh0 = nq ? ql[0].rho : (rho ? rho : 1);
            poly_t P0 = POLY;      /* side 0's norm is G = Y1*x + Y0, degree 1 */
            P0.deg = 1; P0.c[0] = P0.y0; P0.c[1] = P0.y1;
            for (int z = 2; z < BENCH_NCOEFF; z++) P0.c[z] = 0.0;
            qlat_build(&L0, q0, rh0, POLY.skew);
            /* Only the sq side's norm carries a factor of q to divide out.
             * Passing is_sqside=1 for side 1 unconditionally made side 0's
             * maxnorm ~log2(q) too large and side 1's too small whenever the q
             * lives on the rational side -- a ~25-bit error in both scales, in
             * opposite directions. */
            norm_setup(&N1, &POLY, &L0, cfg.logI, cfg.J, 1.0, cfg.sq_side == 1);
            norm_setup(&N0, &P0,   &L0, cfg.logI, cfg.J, 1.0, cfg.sq_side == 0);
            m1 = (double)(N1.log2M - N1.bias);
            m0 = (double)(N0.log2M - N0.bias);
            /* An explicit --scale / --allowance is a deliberate override: it
             * is how a swept operating point, or a bound the job file does not
             * express, gets stated. */
            if (!scale_set)  cfg.scale  = las_scale(m1);
            if (!scale0_set) cfg.scale0 = las_scale(m0);
            if (cfg.scale <= 0.0 || cfg.scale0 <= 0.0) {
                fprintf(stderr, "derived scale: degenerate maxnorm"
                        " (side1 %.2f, side0 %.2f)\n", m1, m0);
                return 1;
            }
            /* The startup guards on --scale ran long before this point and
             * against the c183's norm sizes, so a DERIVED scale never met
             * them. fb_load's logp is a uint8 that saturates silently at 255,
             * which is exactly what those guards exist to prevent. Recheck
             * here, on both sides, against the real bounds. */
            {
                const double cinit = (cfg.cell_bits == 16) ? 4096.0 : 255.0;
                const struct { double sc, mx; uint32_t lim; const char *nm; } chk[2] = {
                    { cfg.scale,  m1, fbbound, "side 1" },
                    { cfg.scale0, m0, rlim,    "side 0" }
                };
                for (int z = 0; z < 2; z++) {
                    const double lg = chk[z].lim > 1
                        ? log((double)chk[z].lim) / log(2.0) : 0.0;
                    if (chk[z].sc * chk[z].mx > cinit) {
                        fprintf(stderr, "%s: scale %.3f x log2(maxnorm) %.1f"
                                " exceeds CINIT %.0f\n",
                                chk[z].nm, chk[z].sc, chk[z].mx, cinit);
                        return 1;
                    }
                    if (chk[z].sc * lg > 255.0) {
                        fprintf(stderr, "%s: scale %.3f x log2(lim) %.1f exceeds"
                                " the 8-bit per-ideal log\n",
                                chk[z].nm, chk[z].sc, lg);
                        return 1;
                    }
                }
            }
            /* Default: our own rule, mfb + the slack our approximation needs.
             * --lambda0/--lambda1 opt back into CADO's rule for anyone who
             * wants it; an explicit --allowance overrides both. */
            if (!allowance_set)
                cfg.allowance  = lambda1_set
                    ? las_allowance(m1, cfg.scale, lambda1,
                                    cfg.lpb ? cfg.lpb : 32,
                                    cfg.mfb ? cfg.mfb : 92)
                    : sieve_allowance(m1, cfg.scale, cfg.mfb ? cfg.mfb : 92);
            if (!allowance0_set)
                cfg.allowance0 = lambda0_set
                    ? las_allowance(m0, cfg.scale0, lambda0, cfg.lpb0, cfg.mfb0)
                    : sieve_allowance(m0, cfg.scale0, cfg.mfb0);
            /* uint32_t, matching pipe_side_init. The (unsigned char) this used
             * to print through wrapped any bound above 255 -- legal against a
             * 16-bit cell with CINIT 4096 -- so the banner disagreed with the
             * bound the sieve actually ran. */
            printf("params from q=%llu: side 1 log2(maxnorm)=%.2f scale=%.3f"
                   " allowance=%.2f bound=%u\n", (unsigned long long)q0, m1,
                   cfg.scale, cfg.allowance,
                   (uint32_t)(cfg.allowance * cfg.scale + 1.0));
            printf("                    side 0 log2(maxnorm)=%.2f scale=%.3f"
                   " allowance=%.2f bound=%u\n", m0, cfg.scale0, cfg.allowance0,
                   (uint32_t)(cfg.allowance0 * cfg.scale0 + 1.0));
            /* An allowance far above mfb admits survivors that classify will
             * then reject: the sieve keeps a position whose cofactor is bigger
             * than the cofactoriser is willing to touch, and the whole resieve
             * and trial division for it is thrown away.
             *
             * SOME slack is right -- the survivor test is a byte-quantised log
             * approximation, so a bound at exactly mfb loses relations to
             * rounding. One byte unit is 1/scale bits, and ~1 bit of slack is
             * what that buys. Beyond that it is pure waste. Measured on the
             * SNFS job (rlambda 3.4 -> 91.80 against mfbr 88):
             *
             *     91.80  120,317 survivors/q   10,313 relations
             *     89.00   99,808 survivors/q   10,312   (-0.01%)
             *     88.00   90,782 survivors/q   10,306   (-0.07%)
             *     85.00   74,849 survivors/q    9,668   (-6.25%)
             *
             * so 17% of the trial-division input was buying one relation in
             * ten thousand. Warned rather than clamped: it is the job file's
             * stated parameter and silently overriding it would make a run
             * something other than what was asked for. */
            {
                const double sl1 = cfg.allowance  - (double)(cfg.mfb ? cfg.mfb : 92);
                const double sl0 = cfg.allowance0 - (double)cfg.mfb0;
                /* Compared against what the derivation WOULD have produced,
                 * not against a fixed number of bits over mfb. A fixed 2.0
                 * threshold fired on the derived default itself whenever
                 * scale < 1 -- which happens for any maxnorm above 254 bits,
                 * since slack is 2/scale -- telling the operator to "drop the
                 * override" and use the very value already in force.
                 *
                 * Per side, too: one overridden side used to print and
                 * "correct" both, naming a side-0 override that was never
                 * passed. */
                const double d1 = sieve_allowance(m1, cfg.scale,
                                                  cfg.mfb ? cfg.mfb : 92);
                const double d0 = sieve_allowance(m0, cfg.scale0, cfg.mfb0);
                if (cfg.allowance > d1 + 0.01)
                    fprintf(stderr,
                        "note: side 1 allowance %.2f is %.2f bits looser than"
                        " the derived %.2f; the surplus\n"
                        "      admits survivors the cofactoriser then rejects"
                        " (mfb %u).\n",
                        cfg.allowance, cfg.allowance - d1, d1,
                        cfg.mfb ? cfg.mfb : 92);
                if (cfg.allowance0 > d0 + 0.01)
                    fprintf(stderr,
                        "note: side 0 allowance %.2f is %.2f bits looser than"
                        " the derived %.2f; the surplus\n"
                        "      admits survivors the cofactoriser then rejects"
                        " (mfb %u).\n",
                        cfg.allowance0, cfg.allowance0 - d0, d0, cfg.mfb0);
                (void)sl1; (void)sl0;
            }
            if (cfg.cadofb) { if (fb_load_cado(cfg.cadofb, cfg.scale, &fb1) != 0) return 1; }
            else if (fb_load(fbpath, &fb1) != 0) return 1;
            fb_fill_logp(&fb1, cfg.scale);
        }
        if (cfg.cadofb && fb1.maxbits > 0 && fb1.maxbits != maxbits)
            fprintf(stderr,
                    "note: algebraic factor base says maxbits=%d, while --maxbits=%d;\n"
                    "      the file controls algebraic powers and the flag controls rational powers.\n"
                    "      maxbits=1 is supported but prime-only and normally lower-yielding;\n"
                    "      regenerate the file or pass its bound explicitly with --maxbits.\n",
                    fb1.maxbits, maxbits);
        if (fb_split_small(&fb1, bkthresh, &fbs1) != 0) return 1;
        /* The GGNFS .afb.0 format carries neither p = 2 nor any prime power, so
         * the default factor base silently under-divides every algebraic norm by
         * its full power of 2 and loses ~2/3 of the relations -- with no error
         * anywhere, because the norms still reconstruct exactly and the leftover
         * 2^k merely makes the cofactor even. Montgomery arithmetic then needs an
         * odd modulus, so those records burn the whole rho budget and report
         * "stuck". Cost me a day; check the invariant instead. */
        {
            uint32_t i, n2 = 0;
            for (i = 0; i < fbs1.n; i++) if (fbs1.primes[i] == 2) n2++;
            /* Absent p = 2 is only a defect if f actually HAS a root mod 2.
             * When it does not -- true of the SNFS polynomial x^5+x^4-4x^3-
             * 3x^2+3x+1, whose f(0), f(1) and leading coefficient are all odd
             * -- the algebraic norm is always odd, fbgen is right to emit no
             * entry, and refusing the run rejects a perfectly good job. This
             * guard used to test only for the entry's presence and did exactly
             * that on the first SNFS job it saw. */
            /* The p = 2 test was, in practice, what forced --cadofb: a GGNFS
             * .afb.0 has neither p = 2 nor prime powers, and the missing 2 was
             * the symptom that got caught. Exempting polynomials with no root
             * mod 2 removed that gate for exactly the SNFS jobs -- which still
             * need the prime powers, and would otherwise run to completion
             * with every algebraic norm under-divided by p^(k-1), exiting 0
             * having quietly lost yield.
             *
             * So the requirement is stated directly rather than inferred from
             * a symptom. */
            if (!cfg.cadofb && (cfg.relations || cfg.candidates || cfg.cofactor)) {
                fprintf(stderr,
                    "ERROR: relation-producing runs need --fb1 <fbgen output>.\n"
                    "         The GGNFS .afb.0 format carries neither p = 2 nor"
                    " prime powers, so\n"
                    "         algebraic norms are under-divided and the yield is"
                    " silently short.\n");
                return 1;
            }
            if (!n2 && !poly_has_root_mod2(&POLY)) {
                printf("note: f has no root mod 2, so the algebraic norm is"
                       " always odd and the factor base correctly has no"
                       " p = 2 entry\n");
            } else if (!n2) {
                const int producing = cfg.relations || cfg.candidates || cfg.cofactor;
                fprintf(stderr,
                        "%s: the algebraic factor base has no entry for p = 2"
                        " (%u small entries, first p = %u).\n"
                        "         Algebraic norms keep their power of 2, so the"
                        " cofactors are EVEN --\n"
                        "         and mz_n0inv requires an odd modulus, so they"
                        " cannot be split at all.\n"
                        "         Pass --fb1 <fbgen output>.\n",
                        producing ? "ERROR" : "WARNING",
                        fbs1.n, fbs1.n ? fbs1.primes[0] : 0);
                /* A sieve-only run may reasonably continue: it never reaches the
                 * cofactoriser, and the survivor counts are still meaningful. A
                 * relation-producing one may not -- it would exit 0 having
                 * quietly lost two thirds of the yield, which is exactly how
                 * this was mistaken for a regression once already. */
                if (producing) return 1;
            }
        }
        fb_restrict(&fb1, bkthresh, fbbound);

        if (rfb_build(&POLY, rlim, maxbits, cfg.scale0, &fb0) != 0) return 1;
        fb_fill_logp(&fb0, cfg.scale0);
        if (fb_split_small(&fb0, bkthresh, &fbs0) != 0) return 1;
        fb_restrict(&fb0, bkthresh, rlim);

        printf("\nside 1 (algebraic): bucketed %u <= p < %u : %u entries,"
               " line-sieved %u\n", bkthresh, fbbound, fb1.n, fbs1.n);
        printf("side 0 (rational) : bucketed %u <= p < %u : %u entries,"
               " line-sieved %u\n", bkthresh, rlim, fb0.n, fbs0.n);

        {
            /* Neither --qlist nor --qrange: the single-q path, read above and
             * at the --qrange block respectively. */
            if (!nq) {
                ql = (qsel_t *)malloc(sizeof(qsel_t));
                if (!ql) return 1;
                ql[0].q = q;
                ql[0].rho = rho ? rho : (uint64_t)(0x9E3779B97F4A7C15ull % q);
                nq = 1;
            }
            prc = run_pipeline(&fb1, &fbs1, &fb0, &fbs0, ql, nq, qgen,
                               &POLY, &cfg);
            free(ql);
            sqgen_free(qgen);
        }
        fb_free(&fb1); fb_free(&fbs1); fb_free(&fb0); fb_free(&fbs0);
        return prc;
    }


    /* The special-q lives on side 1, so only side 1's factor base is truncated
     * at q (the GGNFS convention). Side 0 runs to rlim regardless. */
    if (cfg.side == 1) {
        if (cfg.cadofb) {
            if (fb_load_cado(cfg.cadofb, cfg.scale, &fb) != 0) return 1;
            if (fb.maxbits > 0 && fb.maxbits != maxbits)
                fprintf(stderr,
                        "note: factor base maxbits=%d differs from --maxbits=%d;"
                        " maxbits=1 is valid but normally lower-yielding\n",
                        fb.maxbits, maxbits);
        } else {
            if (fb_load(fbpath, &fb) != 0) return 1;
            printf("\nfactor base %s: %u (p,r) pairs\n", fbpath, fb.n);
        }
    } else {
        if (rfb_build(&POLY, rlim, maxbits, cfg.scale, &fb) != 0) return 1;
        if (!fbbound) fbbound = rlim;
        /* side 0 is G(x) = Y1*x + Y0: degree 1, and its norms are what the
         * apply kernel must initialise cells with */
        POLY.deg = 1; POLY.c[0] = POLY.y0; POLY.c[1] = POLY.y1;
        for (int z = 2; z < BENCH_NCOEFF; z++) POLY.c[z] = 0.0;
        printf("\nside 0 (rational): degree 1, G(x) = Y1*x + Y0\n");
    }
    fb_fill_logp(&fb, cfg.scale);
    if (fb_split_small(&fb, bkthresh, &fbs) != 0) return 1;
    fb_restrict(&fb, bkthresh, fbbound);
    printf("  bucketed  %u <= p < %u : %u entries\n", bkthresh, fbbound, fb.n);

    /* Cofactor-classification parameters, from oracle/input.job. `lim` is the
     * factor-base bound actually sieved, which is what CADO's gap test uses --
     * so truncating the factor base tightens the classification too. */
    /* --td runs inside the two-sided intersection, so without the other
     * side's bitmap it would silently do nothing at all. */
    if (cfg.td && !cfg.other_bits) {
        fprintf(stderr, "bench: --td needs --other-bits FILE (the other side's"
                        " survivor bitmap); trial division runs on the\n"
                        "       two-sided survivor list, which does not exist"
                        " without it.\n");
        return 2;
    }
    if (!cfg.lim) cfg.lim = fbbound;
    if (!cfg.lpb) cfg.lpb = (cfg.side == 1) ? 32u : 31u;
    if (!cfg.mfb) cfg.mfb = (cfg.side == 1) ? 92u : 60u;
    printf("  line-sieved       p < %u (plus every p^k, k>=2) : %u entries\n",
           bkthresh, fbs.n);
    if (!fb.n) { fprintf(stderr, "empty factor base after restriction\n"); return 1; }

    qlat_build(&L, q, rho ? rho : (uint64_t)(0x9E3779B97F4A7C15ull % q), POLY.skew);
    printf("q-lattice q=%llu basis (a0,a1,b0,b1) = (%lld,%lld,%lld,%lld), det=%lld\n",
           (unsigned long long)q, (long long)L.a0, (long long)L.a1,
           (long long)L.b0, (long long)L.b1, (long long)(L.a0 * L.b1 - L.a1 * L.b0));
    if (!rho)
        printf("  NOTE: rho is synthetic (no root of f mod q computed here). The basis is\n"
               "  a genuine reduced q-lattice basis of the right shape, and transformed\n"
               "  roots are uniform mod p either way, so fill volume and distribution are\n"
               "  representative. Pass --rho to match a real las special-q.\n");
    else
        printf("  rho = %llu (real root of f mod q); compare against the a0/b0/a1/b1\n"
               "  that las -v prints, remembering las groups by coordinate and qlat_t\n"
               "  by vector: las (a0,b0,a1,b1) == qlat_t (a0,a1,b0,b1).\n",
               (unsigned long long)rho);

    printf("\nsieve area I=2^%d x J=%u = %.3e positions; region 2^%d; %u regions\n",
           cfg.logI, cfg.J, (double)(1u << cfg.logI) * cfg.J, cfg.log_region,
           ((1u << cfg.logI) * cfg.J) >> cfg.log_region);
    printf("mode=%s record=%dB threads=%d reps=%d\n\n",
           cfg.fill_mode == FILL_ATOMIC ? "atomic" : "twolevel",
           cfg.record_bytes, cfg.threads, cfg.reps);

    if (cfg.verify) {
        int nc;
        printf("[verify] Franke-Kleinjung walk vs brute force (logI=8, J=128)...\n");
        nc = verify_walk(8, 128, 24);
        if (nc < 0) { printf("[verify] FAILED\n"); return 1; }
        printf("[verify] OK: %d primes x 5 roots enumerated identically\n", nc);
        printf("[verify] root transform vs its definition, by set equality...\n");
        if (verify_transform(&L, 200) != 0) { printf("[verify] FAILED\n"); return 1; }
        printf("[verify] OK: q <= 200, every root in [0,2q) -- primes, prime\n"
               "         powers, even moduli, affine and projective\n");
        printf("[verify] the same, over the loaded factor base itself...\n");
        nc = verify_fb_transform(&fbs, &L, bkthresh);
        if (nc < 0) { printf("[verify] FAILED\n"); return 1; }
        printf("[verify] OK: %d small-part entries agree with the definition\n", nc);
    }

    int rc = run_bench(&fb, &fbs, &L, &POLY, &cfg);
    fb_free(&fb);
    return rc;
}
