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
"  --fb PATH        GGNFS .afb.0 factor base   [../oracle/input.job.afb.0]\n"
"  --logI N         log2 of sieve width I      [15]  (I15e)\n"
"  --J N            sieve height J             [16384]\n"
"  --region N       log2 bucket region size    [14]  (16384 16-bit cells, 32 KB)\n"
"  --bkthresh N     bucket-sieve p >= this     [1<<logI]\n"
"  --fbbound N      truncate FB at this p      [q]  (GGNFS truncates at the special-q)\n"
"  --q N            special-q                  [120000011]\n"
"  --rho N          root of f mod q            [synthetic]  (las -v prints it)\n"
"  --record-bytes N 2 | 4 | 8                  [4]\n"
"  --mode M         atomic | twolevel          [atomic]  (twolevel lost by 2.7x)\n"
"  --threads N      threads per block          [256]\n"
"  --blocks N       0 = auto (6 per SM)        [0]\n"
"  --reps N         timing repetitions         [3]\n"
"  --survbits FILE  write a survivor bitmap (1 bit/position, x order)\n"
"  --other-bits F   the other side's bitmap -> device intersect+gcd+compact\n"
"  --emit FILE      write the compacted survivor list (x, a, b) here\n"
"  --td             exact norms + trial division on the survivors\n"
"  --ab-resieve     re-run the settled layout A/B resieve experiment (slow)\n"
"  --resieve-sweep  sweep the resieve unroll depth and summary granularity\n"
"  --pipeline       BOTH SIDES in one process: sieve, intersect, TD, classify,\n"
"                   join, and emit relations + cofactorisation candidates.\n"
"                   Runs one fixed configuration; the harness-only options\n"
"                   above (--mode, --cells, --norm, --stage, ...) are refused\n"
"  --alim N         side-1 factor base bound   [134200000]\n"
"  --scale0 S       side-0 byte scale          [1.925]\n"
"  --allowance0 B   side-0 survivor bits       [68.1 -> bound 132]\n"
"  --relations F    write complete relations here\n"
"  --qlist FILE     band of special-q: `q rho` per line (# comments ok)\n"
"  --qrange MIN:MAX every (q, rho) in that range, taken from the algebraic\n"
"                   factor base -- no root-finding and no q-list file needed\n"
"  --nq N           stop after N special-q from the list\n"
"  --verbose-q      print a line per special-q instead of a band summary\n"
"  --candidates F   write the cofactorisation batch here\n"
"  --cofactor       split the cofactors INLINE, in a cross-q device queue;\n"
"                   --relations then holds every relation, not just TD's\n"
"  --cofac FILE     cofactorise a batch written by --candidates and emit the\n"
"                   relations to --relations; needs no sieving\n"
"  --cof-rounds N   rho requeue rounds, budget doubling each time\n"
"                   [6 for --cofac; 2 for --pipeline --cofactor]\n"
"  --cof-budget N   rho iterations in the first round\n"
"                   [4096 for --cofac; 65536 for --pipeline --cofactor]\n"
"  --cof-ecm        ECM stage 1 instead of Pollard-Brent rho\n"
"  --check-relations F  verify an emitted relation file: every factor divides,\n"
"                   both norms rebuild to 1, every prime within its lpb\n"
"  --ecm-b1 N       ECM stage-1 bound                              [1000]\n"
"  --ecm-curves N   ECM curves attempted per round                 [16]\n"
"  --lpb N          large-prime bound in bits  [32 side 1, 31 side 0]\n"
"  --mfb N          max cofactor bits          [92 side 1, 60 side 0]\n"
"  --cofgate FILE   gate the cofactors against CADO's own (a b cof0 cof1);\n"
"                   under --pipeline this runs in the first-q validation\n"
"  --emit-cof FILE  write (a, b, cofactor, bits) for every survivor here\n"
"  --not-both-even  apply las's filter: i,j both even can never survive\n"
"  --verify         run the CPU cross-check (slow)\n"
"  --poly PATH      algebraic polynomial       [../oracle/c183.poly]\n"
"  --stage S        fill | both | apply        [both]\n"
"  --cells N        16 | 8 bits per sieve cell [16]  (8 is unsafe; cost only)\n"
"  --norm M         horner | const             [horner]\n"
"  --apply-mode M   atomic | plain             [atomic]  (plain is racy)\n"
"  --apply-threads N  threads per apply block  [512]\n"
"  --allowance B    survivor cofactor bits     [112 = alambda*lpba]\n"
"  --no-smallsieve  skip the p < bkthresh line sieve\n"
"  --side S         1 = algebraic (special-q side), 0 = rational  [1]\n"
"  --rlim N         side-0 factor base bound   [67100000]\n"
"  --scale S        las byte scale for the side [1.0 = raw bits]\n"
"  --cadofb PATH    CADO makefb factor base (has prime powers)\n"
"  --maxbits N      prime powers below 2^N      [15]\n"
"  --dump PATH      write the sieve region in las byte convention\n"
"  --probe i,j      read back that cell after apply (gate 5)\n");
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
    cfg.threads = 256; cfg.blocks = 0; cfg.reps = 3; cfg.verify = 0;
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
    cfg.pipeline = 0;
    cfg.scale0 = 1.925; cfg.allowance0 = 68.1;
    cfg.lim0 = 0; cfg.lpb0 = 31; cfg.mfb0 = 60;
    cfg.relations = NULL; cfg.candidates = NULL;
    cfg.qlist = NULL; cfg.nq_max = 0; cfg.verbose_q = 0; cfg.td_verify = 1;
    cfg.qmin = 0; cfg.qmax = 0;
    cfg.cofactor = 0; cfg.cof_rounds = 2; cfg.cof_budget = 65536;
    cfg.cof_ecm = 0; cfg.ecm_b1 = 1000; cfg.ecm_curves = 16;
    int maxbits = 15;
    int allowance_set = 0;
    const char *cofac_in = NULL;
    const char *check_rel = NULL;
    int cof_rounds = 6;
    uint32_t cof_budget = 4096;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fb") && i + 1 < argc) fbpath = argv[++i];
        else if (!strcmp(argv[i], "--logI") && i + 1 < argc) cfg.logI = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--J") && i + 1 < argc) cfg.J = (uint32_t)strtoul(argv[++i], 0, 10);
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
        else if (!strcmp(argv[i], "--rlim") && i + 1 < argc) rlim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) { cfg.scale = atof(argv[++i]); scale_set = 1; }
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) cfg.dump = argv[++i];
        else if (!strcmp(argv[i], "--cadofb") && i + 1 < argc) cfg.cadofb = argv[++i];
        else if (!strcmp(argv[i], "--survbits") && i + 1 < argc) cfg.survbits = argv[++i];
        else if (!strcmp(argv[i], "--other-bits") && i + 1 < argc) cfg.other_bits = argv[++i];
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) cfg.emit = argv[++i];
        else if (!strcmp(argv[i], "--td")) cfg.td = 1;
        else if (!strcmp(argv[i], "--ab-resieve")) cfg.ab_resieve = 1;
        else if (!strcmp(argv[i], "--resieve-sweep")) cfg.resieve_sweep = 1;
        else if (!strcmp(argv[i], "--pipeline")) cfg.pipeline = 1;
        else if (!strcmp(argv[i], "--cofac") && i + 1 < argc) cofac_in = argv[++i];
        else if (!strcmp(argv[i], "--cofactor")) cfg.cofactor = 1;
        else if (!strcmp(argv[i], "--cof-rounds") && i + 1 < argc) { cof_rounds = atoi(argv[++i]); cfg.cof_rounds = cof_rounds; }
        else if (!strcmp(argv[i], "--cof-budget") && i + 1 < argc) { cof_budget = (uint32_t)strtoul(argv[++i], 0, 10); cfg.cof_budget = cof_budget; }
        else if (!strcmp(argv[i], "--cof-ecm")) cfg.cof_ecm = 1;
        else if (!strcmp(argv[i], "--check-relations") && i + 1 < argc) check_rel = argv[++i];
        else if (!strcmp(argv[i], "--ecm-b1") && i + 1 < argc) cfg.ecm_b1 = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--ecm-curves") && i + 1 < argc) cfg.ecm_curves = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--qlist") && i + 1 < argc) cfg.qlist = argv[++i];
        else if (!strcmp(argv[i], "--qrange") && i + 1 < argc) {
            unsigned long long lo, hi;
            if (sscanf(argv[++i], "%llu:%llu", &lo, &hi) != 2 || lo > hi) {
                fprintf(stderr, "bench: --qrange wants MIN:MAX\n"); return 2;
            }
            cfg.qmin = lo; cfg.qmax = hi;
        }
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) cfg.nq_max = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose-q")) cfg.verbose_q = 1;
        else if (!strcmp(argv[i], "--alim") && i + 1 < argc) alim = (uint32_t)strtoul(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--scale0") && i + 1 < argc) cfg.scale0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--allowance0") && i + 1 < argc) cfg.allowance0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--relations") && i + 1 < argc) cfg.relations = argv[++i];
        else if (!strcmp(argv[i], "--candidates") && i + 1 < argc) cfg.candidates = argv[++i];
        else if (!strcmp(argv[i], "--lpb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 32) { fprintf(stderr, "--lpb %ld out of range 1..32\n", v); return 1; } cfg.lpb = (uint32_t)v; }
        else if (!strcmp(argv[i], "--mfb") && i + 1 < argc) { long v = strtol(argv[++i], 0, 10); if (v < 1 || v > 96) { fprintf(stderr, "--mfb %ld out of range 1..96\n", v); return 1; } cfg.mfb = (uint32_t)v; }
        else if (!strcmp(argv[i], "--cofgate") && i + 1 < argc) cfg.cofgate = argv[++i];
        else if (!strcmp(argv[i], "--emit-cof") && i + 1 < argc) cfg.emit_cof = argv[++i];
        else if (!strcmp(argv[i], "--not-both-even")) cfg.not_both_even = 1;
        else if (!strcmp(argv[i], "--maxbits") && i + 1 < argc) maxbits = atoi(argv[++i]);
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

    /* Cofactoriser bounds. The representation is narrow ON PURPOSE -- split
     * factors are emitted as a single uint32 limb, the rational cofactor is
     * mz<2> and the algebraic one mz<3> -- so a bound outside that design does
     * not degrade, it silently truncates: a 33-bit factor is stored as its low
     * 32 bits and the relation stops reconstructing its own norm. Refuse.
     *
     * These are also the values `atoi` would have turned into enormous
     * unsigned bounds if given a negative number, which is why they are parsed
     * into a signed long here and range-checked before the cast. */
    {
        const uint32_t lpb1 = cfg.lpb ? cfg.lpb : 32, mfb1 = cfg.mfb ? cfg.mfb : 92;
        int bad = 0;
        if (lpb1 > 32 || cfg.lpb0 > 32) {
            fprintf(stderr, "--lpb %u / side-0 %u: split factors are emitted as one"
                    " 32-bit limb, so lpb > 32 truncates them\n", lpb1, cfg.lpb0);
            bad = 1;
        }
        if (mfb1 > 96) {
            fprintf(stderr, "--mfb %u: the algebraic cofactor is 3 limbs, so a"
                    " residual above 96 bits loses its high limbs\n", mfb1);
            bad = 1;
        }
        if (cfg.mfb0 > 64) {
            fprintf(stderr, "side-0 mfb %u: the rational cofactor is 2 limbs, so a"
                    " residual above 64 bits loses its high limbs\n", cfg.mfb0);
            bad = 1;
        }
        /* "prime by size" in mz_split rests on lim^2 > 2^lpb: below that a
         * composite could sit under lim^2 and be emitted as a prime. */
        {
            const uint64_t l1 = cfg.lim ? cfg.lim : alim, l0 = cfg.lim0 ? cfg.lim0 : rlim;
            if ((double)l1 * l1 <= ldexp(1.0, (int)lpb1)) {
                fprintf(stderr, "alim^2 (%.3g) <= 2^lpb: the prime-by-size test in"
                        " mz_split is unsound\n", (double)l1 * l1);
                bad = 1;
            }
            if ((double)l0 * l0 <= ldexp(1.0, (int)cfg.lpb0)) {
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
        if (cfg.cof_rounds < 1 || cfg.cof_rounds > 24) {
            fprintf(stderr, "pipeline cof-rounds %d: must be 1..24\n", cfg.cof_rounds);
            bad = 1;
        }
        if (!cfg.cof_ecm) {
            if (!cof_budget || (uint64_t)cof_budget << (cof_rounds - 1) > 0xFFFFFFFFull) {
                fprintf(stderr, "--cof-budget %u with %d rounds overflows uint32"
                        " (or is zero)\n", cof_budget, cof_rounds);
                bad = 1;
            }
            if (!cfg.cof_budget ||
                (uint64_t)cfg.cof_budget << (cfg.cof_rounds - 1) > 0xFFFFFFFFull) {
                fprintf(stderr, "pipeline cof-budget %u with %d rounds overflows"
                        " uint32 (or is zero)\n", cfg.cof_budget, cfg.cof_rounds);
                bad = 1;
            }
        } else {
            if (!cfg.ecm_curves) {
                fprintf(stderr, "--ecm-curves 0 attempts no curves\n"); bad = 1;
            }
            if (cfg.ecm_b1 < 2 || cfg.ecm_b1 > 1000000u) {
                fprintf(stderr, "--ecm-b1 %u: must be 2..1000000\n", cfg.ecm_b1);
                bad = 1;
            }
        }
        if (bad) return 1;
    }

    printf("=== cuda-sieve bucket-fill benchmark ===\n");



    if (poly_load(polypath, &POLY) != 0) return 1;
    printf("polynomial %s: algebraic degree %d, skew %.4g\n",
           polypath, POLY.deg, POLY.skew);

    if (check_rel)
        return check_relations(check_rel, &POLY, cfg.lpb0,
                              cfg.lpb ? cfg.lpb : 32) ? 1 : 0;

    /* ---- both sides in one process ---- */
    if (cofac_in) {
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.lim)  cfg.lim  = alim;
        if (!cfg.lim0) cfg.lim0 = rlim;
        return run_cofac(cofac_in, cfg.relations, cfg.lim0, cfg.lpb0,
                         cfg.lim, cfg.lpb, cof_rounds, cof_budget,
                         cfg.blocks ? cfg.blocks : 48 * 6, cfg.threads,
                         cfg.cof_ecm, cfg.ecm_b1 ? cfg.ecm_b1 : 1000,
                         cfg.ecm_curves ? cfg.ecm_curves : 16) ? 1 : 0;
    }

    if (cfg.pipeline) {
        fb_t fb1, fbs1, fb0, fbs0;
        qsel_t *ql = NULL;
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
        /* The pipeline defaults to the full algebraic factor base. Truncating
         * it at the special-q is a legitimate configuration (GGNFS does it)
         * but it costs relations outright -- 30 of 1,851 cofactors at the
         * parity q differ by exactly one prime in (q, alim]. */
        /* The pipeline's documented configuration is bound 128 on side 1 and
         * 132 on side 0. Side 1's scale and allowance were being left at the
         * single-side benchmark defaults (1.0 and 112), which gives bound 113
         * and a materially larger trial-division input. */
        if (!scale_set) cfg.scale = 1.275;
        if (!allowance_set) cfg.allowance = 100.0;
        if (cfg.qlist && cfg.qmin) {
            fprintf(stderr, "bench --pipeline: --qlist and --qrange both give"
                    " the band; pass one\n");
            return 2;
        }
        if (!cfg.qlist && !cfg.qmin && !rho) {
            fprintf(stderr, "bench --pipeline: needs a real root of f mod q."
                    " Pass --rho, or --qlist for a band.\n"
                    "  A synthetic root is fine for a sieve microbenchmark but"
                    " not for a path that emits relations.\n");
            return 2;
        }
        if (!fbbound_set) fbbound = alim;
        if (!cfg.lim)  cfg.lim  = fbbound;
        if (!cfg.lim0) cfg.lim0 = rlim;
        if (!cfg.lpb)  cfg.lpb  = 32;
        if (!cfg.mfb)  cfg.mfb  = 92;

        if (cfg.cadofb) {
            if (fb_load_cado(cfg.cadofb, cfg.scale, &fb1) != 0) return 1;
        } else if (fb_load(fbpath, &fb1) != 0) return 1;
        fb_fill_logp(&fb1, cfg.scale);
        /* --qrange: every special-q in [qmin, qmax] with all of its roots.
         *
         * No root-finding is needed and none is done. A special-q below alim
         * is by definition an algebraic factor-base prime, and CADO's makefb
         * output already lists every root of f mod it -- so the band is a
         * WINDOW OF THE FACTOR BASE WE ARE ABOUT TO UPLOAD ANYWAY. Read here,
         * before fb_restrict compacts fb1 in place.
         *
         * Proper prime powers are skipped (a special-q is prime) and so are
         * projective roots, encoded as root >= p: those are roots at infinity,
         * which do not give a q-lattice. */
        if (cfg.qmin) {
            if (cfg.qmax >= fbbound)
                printf("note: --qrange runs past the factor-base bound %u;"
                       " q above it have no entry to read a root from\n", fbbound);
            for (uint32_t i = 0; i < fb1.n; i++) {
                uint32_t pp = fb1.primes[i];
                if (pp < cfg.qmin) continue;
                if (pp > cfg.qmax) break;
                if (FB_ISPOW(&fb1, i) || fb1.roots[i] >= pp) continue;
                if (nq == capq) {
                    capq = capq ? capq * 2 : 1024;
                    ql = (qsel_t *)realloc(ql, (size_t)capq * sizeof(qsel_t));
                    if (!ql) return 1;
                }
                ql[nq].q = pp; ql[nq].rho = fb1.roots[i]; nq++;
                if (cfg.nq_max && nq >= cfg.nq_max) break;
            }
            if (!nq) {
                fprintf(stderr, "bench: no special-q in [%llu, %llu]\n",
                        (unsigned long long)cfg.qmin, (unsigned long long)cfg.qmax);
                return 1;
            }
            printf("band: %u special-q (q, rho) from the algebraic factor base"
                   " in [%llu, %llu]\n", nq,
                   (unsigned long long)cfg.qmin, (unsigned long long)cfg.qmax);
        }
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
            if (!n2) {
                const int producing = cfg.relations || cfg.candidates || cfg.cofactor;
                fprintf(stderr,
                        "%s: the algebraic factor base has no entry for p = 2"
                        " (%u small entries, first p = %u).\n"
                        "         Algebraic norms keep their power of 2, so the"
                        " cofactors are EVEN --\n"
                        "         and mz_n0inv requires an odd modulus, so they"
                        " cannot be split at all.\n"
                        "         Pass --cadofb <makefb output>.\n",
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
            if (cfg.qlist) {
                FILE *f = fopen(cfg.qlist, "r");
                char line[256];
                if (!f) { perror(cfg.qlist); return 1; }
                while (fgets(line, sizeof line, f)) {
                    unsigned long long qq, rr;
                    if (line[0] == '#') continue;
                    if (sscanf(line, "%llu %llu", &qq, &rr) != 2) continue;
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
            } else if (!nq) {
                ql = (qsel_t *)malloc(sizeof(qsel_t));
                ql[0].q = q;
                ql[0].rho = rho ? rho : (uint64_t)(0x9E3779B97F4A7C15ull % q);
                nq = 1;
            }
            prc = run_pipeline(&fb1, &fbs1, &fb0, &fbs0, ql, nq, &POLY, &cfg);
            free(ql);
        }
        fb_free(&fb1); fb_free(&fbs1); fb_free(&fb0); fb_free(&fbs0);
        return prc;
    }


    /* The special-q lives on side 1, so only side 1's factor base is truncated
     * at q (the GGNFS convention). Side 0 runs to rlim regardless. */
    if (cfg.side == 1) {
        if (cfg.cadofb) {
            if (fb_load_cado(cfg.cadofb, cfg.scale, &fb) != 0) return 1;
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
        for (int z = 2; z < 8; z++) POLY.c[z] = 0.0;
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
