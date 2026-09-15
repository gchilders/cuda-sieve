/* GPU kernels for the bucket-fill benchmark.
 *
 * Stages, matching the design doc's cost pillars:
 *   (T) transform + plattice : one modular inverse per distinct prime per
 *                              special-q, then the FK reduction. Producer
 *                              only -- deliberately NOT fused with scatter
 *                              (msieve-s experiment #11: fusing cost +25%).
 *   (a) FILL_ATOMIC          : one global atomicAdd per record, 16K-way.
 *   (c) FILL_TWOLEVEL        : level 1 stages records in shared memory and
 *                              flushes full cache lines with one atomic per
 *                              flush; level 2 splits super-buckets into
 *                              regions the same way.
 */
#include "bench.h"
#include "platform.h"
#include "plattice.cuh"
#include "bigint.cuh"
#include "td.cuh"
#include "metal/metal_rt.h"
#include "metal/td_host.h"

/* Launch-shape constants defined inside the device region above, which
 * the host half reads. Lifted by name from bench_kernels.cu. */
#define SS_BLOCK_CUT   64u
#define SS_WARP_CUT  1024u
#define L1_NBUF   128
#define L1_CAP     64        /* uint32 slots = 256 B; 128*64*4 = 32 KB static smem */
#define L2_NBUF 128
#define L2_CAP   64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <errno.h>

/* Host-side wall clock, for the per-q work that runs off the GPU and so is
 * invisible to cudaEvent timing. Goal 1 is about host demand, so this has to
 * be billed rather than assumed small. */
static double host_ms(void)
{
    return bench_monotonic_ms();
}

static void report_slice_build_error(void)
{
    if (errno == EOVERFLOW)
        fprintf(stderr, "fb_build_slices: factor base requires more than"
                " 65,536 slices; bucket records carry only a 16-bit slice ID\n");
    else if (errno == EINVAL)
        fprintf(stderr, "fb_build_slices: empty factor base or missing log table\n");
    else
        perror("fb_build_slices");
}

/* CUDA asks the driver for cudaDevAttrMaxGridDimX. Metal publishes no grid
 * ceiling of its own -- dispatchThreadgroups takes an NSUInteger -- so the
 * shim reports an honest 32-bit-safe bound through mtlGetDeviceProperties. */
static int mtl_grid_dim_x(int *out)
{
    mtlDeviceProp p;
    if (mtlGetDeviceProperties(&p, 0) != mtlSuccess) return -1;
    *out = p.maxGridSize[0];
    return 0;
}

static int mtl_check_impl(mtlError_t err, const char *expr,
                           const char *file, int line)
{
    if (err == mtlSuccess) return 0;
    fprintf(stderr, "CUDA %s: %s at %s:%d\n",
            expr, mtlGetErrorString(err), file, line);
    return -1;
}

#define CUDA_CHECKED(x) mtl_check_impl((x), #x, __FILE__, __LINE__)
#define CK(x) do { if (CUDA_CHECKED(x)) return -1; } while (0)

/* The maximum opt-in dynamic shared memory is a property of the selected
 * device, not of a CUDA architecture family.  Query it at runtime so cards
 * such as A100/H100 can use larger legal regions while cards with a smaller
 * limit are rejected before cudaFuncSetAttribute()/launch. */
static int mtl_optin_smem_limit(size_t *out)
{
    /* CUDA asks for cudaDevAttrMaxSharedMemoryPerBlockOptin. Metal's
     * equivalent is maxThreadgroupMemoryLength, and there is no separate
     * opt-in tier: 32 KB on Apple silicon against CUDA's ~100 KB, which is
     * what forces log_region <= 13 (plan section 5.1). */
    mtlDeviceProp p;
    if (mtlGetDeviceProperties(&p, 0) != mtlSuccess) return -1;
    *out = p.sharedMemPerBlock;
    return 0;
}

/* ---- stage T: root transform + plattice reduction --------------------- */

/* The transform runs through pl_transform_enc, not pl_transform, for three
 * reasons that all bite on CADO's factor base and none of which bite on
 * GGNFS's:
 *   - a projective entry (root >= q) must keep its reciprocal, and the affine
 *     formula would reduce it to a bogus affine root instead;
 *   - q = 2^15 sits exactly at the default bkthresh, so an EVEN modulus
 *     reaches this kernel, and binary-Euclid pl_invmod cannot invert mod 2^k;
 *   - raising -maxbits puts odd prime powers here, where a non-invertible
 *     denominator makes binary Euclid spin forever ON THE DEVICE.
 * pl_transform_enc handles all three; the first two are wrong answers and the
 * third is a hang, so none of them would have shown up as a failed gate.
 *
 * What is still not expressible is g > 1: hits confined to every g-th row,
 * which is not a plat_t walk. Those emit an empty walk, and the kernel
 * accumulates the number of positions thereby dropped so the loss is a printed
 * number rather than a silence. With the default bkthresh = I >= J it is
 * exactly zero: g > 1 needs q | (rows), and every bucketed q exceeds J. */
static uint32_t build_slices_b(const fb_t *fb, uint32_t **starts_out)
{
    uint32_t ns = 0, k, capacity = 256;
    uint32_t *starts = (uint32_t *)malloc(capacity * 4);
    int cur = -1;
    uint32_t cut = 0;
    for (k = 0; k < fb->n; k++) {
        int lp = fb->logp[k];
        if (lp != cur || (ns && (k - cut) >= 65536u)) {
            if (ns + 2 > capacity) { capacity *= 2; starts = (uint32_t *)realloc(starts, capacity * 4); }
            cur = lp; cut = k; starts[ns++] = k;
        }
    }
    starts[ns] = fb->n;          /* sentinel: slice s is [starts[s], starts[s+1]) */
    *starts_out = starts;
    return ns;
}

/* device code: see metal/bench_kernels.metal and metal/td.metal */

/* ---- host driver ------------------------------------------------------ */

struct dev_bufs {
    uint32_t *primes, *roots, *cursor, *overflow, *nproj, *l1, *l1cnt;
    unsigned long long *nlost;
    plat_t   *plat;
    uint8_t  *out;
    uint16_t *slice, *slice_logp;
    uint32_t *nsurv, *probe;
    uint16_t *dbg;
    uint32_t *sp, *srt, *sg;
    uint32_t *smag;
    uint16_t *slp;
    uint8_t  *dumpbuf;
    uint32_t *survbits;
};

static float time_kernel(mtlEvent_t a, mtlEvent_t b)
{ float ms = 0; mtlEventElapsedTime(&ms, a, b); return ms; }

/* ---- intersect + primitive filter + compaction ------------------------- *
 *
 * The two per-side survivor bitmaps are ANDed, positions that cannot give a
 * PRIMITIVE (a,b) are dropped, and what remains is compacted into a dense
 * list.
 *
 * PRIMITIVITY TAKES TWO TESTS, NOT ONE. gcd(i,j) != 1 is the obvious one, and
 * for a long time it was the only one, on the reasoning that (a,b) inherits
 * the primitivity of (i,j). It does not: (a,b) = M(i,j) with
 * det M = +-q, and a non-unimodular map can destroy primitivity. Exactly one
 * way, and it is cheap to test -- q | b (equivalently q | a, since a = rho*b
 * mod q on the lattice), which makes (a,b) = q*(a',b').
 *
 * Any OTHER common prime p is already excluded: p | a and p | b with p != q
 * implies (a/p, b/p) is still on the q-lattice, so (i,j) = p*(i',j') and the
 * gcd test catches it. So gcd(a,b) is 1 or q, and `b % q` decides which.
 *
 * These are not rare curiosities on a small-q job. A point with q | a and q | b
 * lies on the plattice line of EVERY root of q, so the sieve subtracts log(q)
 * once per root; when the algebraic polynomial splits completely mod q, that
 * is deg*log(q), which is precisely the q^deg sitting in F(a,b) = q^deg
 * F(a',b'). Both sides then look perfectly smooth and the position sails
 * through to trial division, which confirms the factorisation -- of a relation
 * that is q times a smaller one. msieve rejects them with "error -6"
 * (relation.c: gcd(a,b) != 1). Measured on an SNFS job with
 * F = (x^7-1)/(x-1), alim 3.5M and q from 400009: 154 of 154,810 emitted
 * relations were non-primitive, and every one had q = 1 (mod 7) -- the
 * condition for that F to split completely.
 *
 * The list carries BOTH the sieve index x and the pair (a,b). Emitting only
 * (a,b) would be the natural-looking choice and it is the wrong one: every
 * downstream stage -- resieve under either layout, trial division, the
 * cofactor queue -- is indexed by x, and recovering x from (a,b) means
 * inverting the lattice basis per survivor. x is 4 bytes; keep it.
 *
 * Bit-order note: bit k of word w is position x = 32*w + k, which is the
 * order k_apply writes one word per warp (atomicOr only in its small-region
 * fallback).
 */
/* device code: see metal/bench_kernels.metal and metal/td.metal */

/* ======================= trial division, host side ======================= */

/* The exact homogeneous form for one side. The rational side is not a special
 * case: G(a,b) = Y1*a + Y0*b is the degree-1 member of the same family, so the
 * norm kernel is shared. */
static int td_build_poly(tdpoly_t *T, const poly_t *P, int side)
{
    memset(T, 0, sizeof(*T));
    for (int k = 0; k < BENCH_NCOEFF; k++) T->sign[k] = 1;
    if (side == 0) {
        int s0 = 1, s1 = 1;
        T->deg = 1;
        if (bn_from_dec(&T->c[0], P->y0s, &s0)) return -1;
        if (bn_from_dec(&T->c[1], P->y1s, &s1)) return -1;
        T->sign[0] = s0; T->sign[1] = s1;
        return 0;
    }
    T->deg = P->deg;
    for (int k = 0; k <= P->deg; k++) {
        int s = 1;
        if (!P->cs[k][0]) continue;                 /* absent == zero */
        if (bn_from_dec(&T->c[k], P->cs[k], &s)) return -1;
        T->sign[k] = s;
    }
    return 0;
}

/* Direct-test table for p < bkthresh.
 *
 * PROPER PRIME POWERS ARE EXCLUDED. fb_split_small puts every power in this
 * table regardless of size, but trial division recovers multiplicity by
 * repeated division by the base prime -- which is always present here too,
 * since a power p^k below the factor-base bound forces p well below bkthresh.
 * Keeping the powers would divide a norm by p^2 as though p^2 were prime.
 *
 * For a PRIME modulus the transform's row divisor g can only be 1 or p (it is
 * gcd(D, p)), so p == m*g always holds and the prime never has to be carried
 * separately. */
/* Fills a caller-provided table of at least fbs->n entries. Split out from
 * td_build_small so the pipeline can refill ONE pinned buffer per special-q
 * instead of malloc/free-ing 85 KB on every q of a band. Returns the entry
 * count, or 0 with a message if the m*g invariant fails. */
static uint32_t td_fill_small(const fb_t *fbs, const qlat_t *L, int logI,
                              tdsmall_t *t)
{
    const uint32_t Ihalf = 1u << (logI - 1);
    uint32_t n = 0;
    for (uint32_t i = 0; i < fbs->n; i++) {
        uint32_t rt, g, m;
        if (FB_ISPOW(fbs, i)) continue;
        m = pl_transform_enc(fbs->primes[i], fbs->roots[i],
                             L->a0, L->a1, L->b0, L->b1, &rt, &g);
        if (m * g != fbs->primes[i]) {          /* the invariant above */
            fprintf(stderr, "td_build_small: m*g != p at entry %u"
                            " (p=%u m=%u g=%u)\n", i, fbs->primes[i], m, g);
            return 0;
        }
        t[n].m = m; t[n].rt = rt; t[n].g = g;
        t[n].cst = Ihalf % m;
        t[n].recip = bn_recip_u32(fbs->primes[i]);
        td_magic_build(m, &t[n].magic, &t[n].sh);
        n++;
    }
    if (getenv("TD_DUMP_SMALL")) {
        uint32_t c2 = 0;
        fprintf(stderr, "td_fill_small: %u entries from %u fb rows; first 6:", n, fbs->n);
        for (uint32_t i = 0; i < n && i < 6; i++)
            fprintf(stderr, " p=%u(m=%u,g=%u,rt=%u,magic=%u)",
                    t[i].m * t[i].g, t[i].m, t[i].g, t[i].rt, t[i].magic);
        for (uint32_t i = 0; i < n; i++) if (t[i].m * t[i].g == 2) c2++;
        fprintf(stderr, "  | entries with p==2: %u\n", c2);
    }
    return n;
}

static uint32_t td_build_small(const fb_t *fbs, const qlat_t *L, int logI,
                               tdsmall_t **out)
{
    uint32_t n;
    tdsmall_t *t;
    if (!fbs || !fbs->n) { *out = NULL; return 0; }
    t = (tdsmall_t *)malloc((size_t)fbs->n * sizeof(tdsmall_t));
    if (!t) { *out = NULL; return 0; }
    n = td_fill_small(fbs, L, logI, t);
    if (!n) { free(t); *out = NULL; return 0; }
    *out = t;
    return n;
}

/* ---- the gate: our cofactors against CADO's ---------------------------- */

typedef struct { int64_t a, b; uint32_t idx; } td_ab_t;

static bool td_ab_less(const td_ab_t &x, const td_ab_t &y)
{ return x.b != y.b ? x.b < y.b : x.a < y.a; }

/* oracle/c183.q*.cofac_candidates.txt holds `a b cofac0 cofac1` for every
 * position CADO carried into cofactoring, i.e. its residual after its own
 * trial division. Ours must agree exactly, which is a far stronger statement
 * than agreeing on a count. */
static int td_gate_cofactors_part(const char *path, uint32_t n,
                                  const int64_t *ha, const int64_t *hb,
                                  const bn_t *hcof, int side,
                                  uint32_t *found_out)
{
    FILE *f = fopen(path, "r");
    char line[512];
    uint32_t nref = 0, found = 0, match = 0, absent = 0;
    td_ab_t *tab;
    if (!f) { perror(path); return -1; }

    /* our list, keyed on (a,b) normalised to b > 0 -- our b is i, and las
     * reports the mirrored point (-a, -i) whenever i < 0. */
    tab = (td_ab_t *)malloc((size_t)n * sizeof(td_ab_t));
    if (!tab) { fclose(f); return -1; }
    for (uint32_t k = 0; k < n; k++) {
        int64_t a = ha[k], b = hb[k];
        if (b < 0) { a = -a; b = -b; }
        tab[k].a = a; tab[k].b = b; tab[k].idx = k;
    }
    std::sort(tab, tab + n, td_ab_less);

    printf("\n  --- trial-division gate vs %s (side %d) ---\n", path, side);
    while (fgets(line, sizeof line, f)) {
        char c0[128], c1[128];
        long long ra, rb;
        td_ab_t key, *lo;
        if (line[0] == '#') continue;
        if (sscanf(line, "%lld %lld %127s %127s", &ra, &rb, c0, c1) != 4) continue;
        nref++;
        key.a = ra; key.b = rb; key.idx = 0;
        lo = std::lower_bound(tab, tab + n, key, td_ab_less);
        if (lo == tab + n || lo->a != key.a || lo->b != key.b) { absent++; continue; }
        found++;
        {
            char buf[BN_DEC_MAX];
            bn_to_dec(&hcof[lo->idx], buf);
            if (!strcmp(buf, side ? c1 : c0)) match++;
            else if (match + 8 > found)      /* show the first few only */
                printf("    MISMATCH (a,b)=(%lld,%lld)  ours %s  CADO %s\n",
                       ra, rb, buf, side ? c1 : c0);
        }
    }
    fclose(f);
    printf("  %-30s %8u\n", "reference records", nref);
    printf("  %-30s %8u  (%u not in our survivor list)\n",
           "matched to a survivor", found, absent);
    printf("  %-30s %8u of %u   %s\n", "cofactors identical", match, found,
           !found ? "NO OVERLAP"
                  : (match == found ? "PASS" : "FAIL"));
    if (found_out) *found_out += found;
    free(tab);
    /* No overlap in one slab is not an error: a reference file covers the
     * complete q. The slabbed pipeline accumulates found across all slabs and
     * rejects the q if the total remains zero. Any actual mismatch is fatal. */
    return (int)(found - match);
}

/* Whole-area/harness semantics retain the historical requirement that the
 * reference overlap at least one survivor. */
static int td_gate_cofactors(const char *path, uint32_t n,
                             const int64_t *ha, const int64_t *hb,
                             const bn_t *hcof, int side)
{
    uint32_t found = 0;
    const int rc = td_gate_cofactors_part(path, n, ha, hb, hcof, side, &found);
    if (rc) return rc;
    return found ? 0 : 1;
}

/* ---- the stage ---------------------------------------------------------- */

/* The MEASUREMENT harness for the trial-division chain: best-of-three on every
 * kernel, plus the two diagnostic variants of k_td that separate the small
 * prime congruence test from the divisions it triggers, plus the reconstruction
 * gate. The pipeline used to call this per q and per side, which is what made
 * its post-sieve cost 222 ms/q; it now runs pipe_td_perq, and this stays what
 * `bench --td` reports and what every command in RESULTS.md reproduces. */
static int run_td_stage(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                        const poly_t *POLY, const bench_cfg_t *cfg,
                        const plat_t *d_plat, const uint32_t *d_primes,
                        const uint32_t *d_two, uint32_t nbitword,
                        uint32_t xmax, int blocks, int threads)
{
    const uint32_t K = 16;          /* large primes kept per survivor */
    const uint32_t ngroup = nbitword / TD_GROUP_W;
    const uint32_t nb = (ngroup + TD_SCAN_BLK - 1) / TD_SCAN_BLK;
    const uint32_t nsum = nbitword / 2;
    const uint32_t nsumword = (nsum + 31) / 32;

    uint32_t *d_cnt = NULL, *d_gbase = NULL, *d_bsum = NULL, *d_sum = NULL;
    uint32_t *d_x = NULL, *d_plist = NULL, *d_pcnt = NULL, *d_flags = NULL;
    uint8_t  *d_status = NULL;
    uint32_t *d_fac = NULL, *d_faccnt = NULL;
    int64_t  *d_a = NULL, *d_b = NULL;
    bn_t     *d_cof = NULL;
    uint8_t  *d_cofbits = NULL;
    tdpoly_t *d_poly = NULL;
    tdsmall_t *d_sm = NULL, *h_sm = NULL;
    unsigned long long *d_ovf = NULL;
    tdpoly_t h_poly;
    int rc = 0;
    uint32_t n = 0, nsm = 0, hflags = 0;

    unsigned long long hovf = 0;
    mtlEvent_t t0, t1;
    float ms_rank = 0, ms_emit = 0, ms_sum = 0, ms_scatter = 1e30f, ms_td = 1e30f;
    float ms_td_nosm = 1e30f, ms_td_nodiv = 1e30f, ms_class = 0;
    unsigned long long hhits = 0;

    if (nbitword % TD_GROUP_W) {
        fprintf(stderr, "  --td: %u bitmap words is not a multiple of %d\n",
                nbitword, TD_GROUP_W);
        return -1;
    }
    if (L->q >> 32) {
        fprintf(stderr, "  --td: special-q %llu exceeds 32 bits; the divide-out"
                " path assumes it fits\n", (unsigned long long)L->q);
        return -1;
    }
    if (td_build_poly(&h_poly, POLY, cfg->side)) {
        fprintf(stderr, "  --td: could not parse exact polynomial coefficients\n");
        return -1;
    }
    nsm = td_build_small(fbs, L, cfg->logI, &h_sm);

    printf("\n  --- exact norms + trial division (side %d) ---\n", cfg->side);
    printf("  form: degree %d, |c| up to %d bits;"
           " small direct-test table %u entries (%.1f KB)\n",
           h_poly.deg, bn_bits(&h_poly.c[0]), nsm,
           nsm * sizeof(tdsmall_t) / 1024.0);

    mtlEventCreate(&t0); mtlEventCreate(&t1);

    /* ---- survivor rank over the two-sided bitmap ---- */
    CK(mtlMalloc(&d_cnt, (size_t)ngroup * 4));
    CK(mtlMalloc(&d_gbase, (size_t)ngroup * 4));
    CK(mtlMalloc(&d_bsum, (size_t)nb * 4));
    /* Best of 3, like every other timed block here: CUDA loads a kernel's code
     * on its FIRST launch (lazy module loading), and these four kernels have
     * never run at this point. Timing the first launch reported 10.7 ms for
     * what steady-state measurement puts an order of magnitude below. */
    ms_rank = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        mtlEventRecord(t0);
        MTL_LAUNCH(k_group_counts, blocks, threads, 0, 0, d_two, ngroup, d_cnt);
        MTL_LAUNCH(k_scan_pass1, nb, TD_SCAN_BLK, 0, 0, d_cnt, ngroup, d_gbase, d_bsum);
        MTL_LAUNCH(k_scan_pass2, 1, 1024, 0, 0, d_bsum, nb);
        MTL_LAUNCH(k_scan_pass3, nb, TD_SCAN_BLK, 0, 0, d_gbase, ngroup, d_bsum);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_rank) ms_rank = t; }
    }
    {
        uint32_t base = 0, cnt = 0;
        CK(mtlMemcpy(&base, d_gbase + ngroup - 1, 4, mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(&cnt,  d_cnt   + ngroup - 1, 4, mtlMemcpyDeviceToHost));
        n = base + cnt;
    }
    printf("  %-30s %8u\n", "survivors (rank scan)", n);
    if (!n) { fprintf(stderr, "  --td: no survivors to divide\n"); rc = -1; goto done; }

    /* ---- rank-ordered (x, a, b) ---- */
    CK(mtlMalloc(&d_x, (size_t)n * 4));
    CK(mtlMalloc(&d_a, (size_t)n * 8));
    CK(mtlMalloc(&d_b, (size_t)n * 8));
    ms_emit = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        mtlEventRecord(t0);
        MTL_LAUNCH(k_emit_ranked_0, blocks, threads, 0, 0, d_two, d_gbase, nbitword, cfg->logI, L->a0, L->a1, L->b0, L->b1, d_x, d_a, d_b, n, 0u);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_emit) ms_emit = t; }
    }

    /* ---- large primes: re-walk, filtered, scattered per survivor ---- */
    CK(mtlMalloc(&d_sum, (size_t)nsumword * 4));
    CK(mtlMalloc(&d_plist, (size_t)n * K * 4));
    CK(mtlMalloc(&d_pcnt, (size_t)n * 4));
    CK(mtlMalloc(&d_ovf, 8));
    mtlEventRecord(t0);
    CK(mtlMemset(d_sum, 0, (size_t)nsumword * 4));
    MTL_LAUNCH(k_build_summary, blocks, threads, 0, 0, d_two, nbitword, d_sum);
    mtlEventRecord(t1);
    CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
    ms_sum = time_kernel(t0, t1);

    /* Sweep BOTH knobs, because they attack the same bottleneck from opposite
     * directions and only measurement separates them.
     *
     *   unroll     -- more probes in flight per warp, hiding the latency
     *   granularity-- a coarser summary is a smaller table, so the probe may
     *                 land in L1 instead of L2, lowering the latency itself.
     *                 Coarser also means more steps reach the 67 MB bitmap.
     *
     * The recovered prime counts must be identical at every setting. */
    {
        uint32_t *h_ref = NULL;
        /* Production setting first, so a normal run times exactly one thing.
         * The rest is the sweep that chose it and only runs on request:
         * unroll 4 at the original granularity, measured 5.21 ms against 6.92
         * at unroll 1. Granularity turned out not to matter at all -- 5.21 to
         * 5.35 ms across a 16x change in table size -- which is itself the
         * evidence that this kernel is latency-bound rather than
         * cache-resident-bound. */
        struct { int lg, u; } trial[] = {
            {6,4},
            {6,1},{6,2},{6,8},{7,4},{8,4},{9,4},{10,4},{9,8},{10,8}
        };
        const unsigned ntrial = cfg->resieve_sweep
                              ? (unsigned)(sizeof trial / sizeof *trial) : 1u;
        for (unsigned ti = 0; ti < ntrial; ti++) {
            const int log_gran = trial[ti].lg, U = trial[ti].u;
            const uint32_t wper = 1u << (log_gran - 5);
            const uint32_t nsw = nbitword / (wper * 32);
            float best = 1e30f;
            uint32_t mism = 0;
            if (!nsw) continue;
            MTL_LAUNCH(k_build_summary_g, blocks, threads, 0, 0, d_two, nbitword, wper, d_sum);
            CK(mtlDeviceSynchronize()); CK(mtlGetLastError());
            for (int rep = 0; rep < 3; rep++) {
                CK(mtlMemset(d_pcnt, 0, (size_t)n * 4));
                CK(mtlMemset(d_ovf, 0, 8));
                mtlEventRecord(t0);
                switch (U) {
                case 1: MTL_LAUNCH(k_resieve_scatter_1_0, blocks, threads, 0, 0, d_plat, d_primes, nullptr, fb->n, xmax, cfg->logI, d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, nullptr); break;
                case 2: MTL_LAUNCH(k_resieve_scatter_2_0, blocks, threads, 0, 0, d_plat, d_primes, nullptr, fb->n, xmax, cfg->logI, d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, nullptr); break;
                case 4: MTL_LAUNCH(k_resieve_scatter_4_0, blocks, threads, 0, 0, d_plat, d_primes, nullptr, fb->n, xmax, cfg->logI, d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, nullptr); break;
                default: MTL_LAUNCH(k_resieve_scatter_8_0, blocks, threads, 0, 0, d_plat, d_primes, nullptr, fb->n, xmax, cfg->logI, d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, log_gran, nullptr); break;
                }
                mtlEventRecord(t1);
                CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
                { float t = time_kernel(t0, t1); if (t < best) best = t; }
            }
            /* Compare the recovered PRIMES, not merely how many there are:
             * two configurations could scatter different primes into a
             * survivor's slot and still agree on the count. The scatter order
             * within a slot is atomic-dependent, so sort each slot first. */
            {
                const size_t sz = (size_t)n * (K + 1) * 4;
                uint32_t *h = (uint32_t *)malloc(sz);
                CK(mtlMemcpy(h, d_pcnt, (size_t)n * 4, mtlMemcpyDeviceToHost));
                CK(mtlMemcpy(h + n, d_plist, (size_t)n * K * 4,
                              mtlMemcpyDeviceToHost));
                for (uint32_t z = 0; z < n; z++) {
                    uint32_t c = h[z] > K ? K : h[z];
                    std::sort(h + n + (size_t)z * K, h + n + (size_t)z * K + c);
                }
                if (!h_ref) h_ref = h;
                else {
                    for (uint32_t z = 0; z < n; z++) {
                        uint32_t c = h[z] > K ? K : h[z];
                        if (h[z] != h_ref[z]) { mism++; continue; }
                        for (uint32_t y = 0; y < c; y++)
                            if (h[n + (size_t)z * K + y] !=
                                h_ref[n + (size_t)z * K + y]) { mism++; break; }
                    }
                    free(h);
                }
            }
            if (cfg->resieve_sweep)
                printf("  %-30s %8.3f ms   (1 bit/%4d positions, %6.1f KB,"
                       " unroll %d)%s\n", "resieve + scatter", best,
                       1 << log_gran, nsw * 4 / 1024.0, U,
                       mism ? "  ** RECOVERY CHANGED" : "");
            else
                printf("  %-30s %8.3f ms   (unroll %d)\n",
                       "resieve + scatter", best, U);
            if (mism) rc = -1;
            /* ti == 0 IS the production configuration (unroll 4, 1 bit/64).
             * Taking the fastest trial instead would report a chain total for
             * a setting the pipeline does not run. */
            if (ti == 0) ms_scatter = best;
        }
        free(h_ref);
        /* restore the reference summary and leave a correct scatter behind */
        MTL_LAUNCH(k_build_summary_g, blocks, threads, 0, 0, d_two, nbitword, 2u, d_sum);
        CK(mtlMemset(d_pcnt, 0, (size_t)n * 4));
        CK(mtlMemset(d_ovf, 0, 8));
        MTL_LAUNCH(k_resieve_scatter_4_0, blocks, threads, 0, 0, d_plat, d_primes, nullptr, fb->n, xmax, cfg->logI, d_sum, d_two, d_gbase, d_plist, d_pcnt, K, d_ovf, 6, nullptr);
        CK(mtlDeviceSynchronize()); CK(mtlGetLastError());
        CK(mtlMemcpy(&hovf, d_ovf, 8, mtlMemcpyDeviceToHost));
    }

    /* ---- the trial division itself ---- */
    CK(mtlMalloc(&d_poly, sizeof(tdpoly_t)));
    CK(mtlMemcpy(d_poly, &h_poly, sizeof(tdpoly_t), mtlMemcpyHostToDevice));
    if (nsm) {
        CK(mtlMalloc(&d_sm, (size_t)nsm * sizeof(tdsmall_t)));
        CK(mtlMemcpy(d_sm, h_sm, (size_t)nsm * sizeof(tdsmall_t),
                      mtlMemcpyHostToDevice));
    }
    CK(mtlMalloc(&d_cof, (size_t)n * sizeof(bn_t)));
    CK(mtlMalloc(&d_cofbits, (size_t)n));
    CK(mtlMalloc(&d_flags, 4));

    /* Same kernel with the small-prime table empty. The direct test is the
     * part with no prior measurement behind it -- 3,500 entries against every
     * survivor -- so it is worth separating from the norm and the recovered
     * large primes rather than reporting one fused number. Run FIRST so the
     * full pass overwrites its output. */
    for (int rep = 0; rep < 3; rep++) {
        mtlEventRecord(t0);
        MTL_LAUNCH(k_td_1_0_0_0, blocks, threads, 0, 0, d_a, d_b, d_x, nullptr, n, cfg->logI, d_poly, cfg->side == 1 ? (uint32_t)L->q : 0u, d_plist, d_pcnt, K, d_sm, 0u, d_cof, d_cofbits, d_flags, nullptr, nullptr, nullptr, 0, 0u);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td_nosm) ms_td_nosm = t; }
    }

    /* the same walk with the divisions removed: separates the 3e9 congruence
     * tests from the big-integer divisions they trigger */
    CK(mtlMemset(d_ovf, 0, 8));
    for (int rep = 0; rep < 3; rep++) {
        mtlEventRecord(t0);
        MTL_LAUNCH(k_td_0_0_0_0, blocks, threads, 0, 0, d_a, d_b, d_x, nullptr, n, cfg->logI, d_poly, cfg->side == 1 ? (uint32_t)L->q : 0u, d_plist, d_pcnt, K, d_sm, nsm, d_cof, d_cofbits, d_flags, d_ovf, nullptr, nullptr, 0, 0u);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td_nodiv) ms_td_nodiv = t; }
    }
    CK(mtlMemcpy(&hhits, d_ovf, 8, mtlMemcpyDeviceToHost));
    hhits /= 3;                      /* three reps accumulated into it */

    for (int rep = 0; rep < 3; rep++) {
        CK(mtlMemset(d_flags, 0, 4));
        mtlEventRecord(t0);
        MTL_LAUNCH(k_td_1_0_0_0, blocks, threads, 0, 0, d_a, d_b, d_x, nullptr, n, cfg->logI, d_poly, cfg->side == 1 ? (uint32_t)L->q : 0u, d_plist, d_pcnt, K, d_sm, nsm, d_cof, d_cofbits, d_flags, nullptr, nullptr, nullptr, 0, 0u);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_td) ms_td = t; }
    }
    CK(mtlMemcpy(&hflags, d_flags, 4, mtlMemcpyDeviceToHost));

    printf("  %-30s %8.3f ms\n", "rank scan", ms_rank);
    printf("  %-30s %8.3f ms\n", "emit (x,a,b) in rank order", ms_emit);
    printf("  %-30s %8.3f ms\n", "build survivor filter", ms_sum);
    if (hovf) printf("  ** resieve list overflow\n");
    printf("  %-30s %8.3f ms   (norm + special-q + %u recovered large primes)\n",
           "  ...without small primes", ms_td_nosm, K);
    printf("  %-30s %8.3f ms   (%llu hits, %.2f per survivor)\n",
           "  ...test only, no division", ms_td_nodiv, hhits, (double)hhits / n);
    printf("  %-30s %8.3f ms   (%u-entry test %.3f + division %.3f)\n",
           "norms + trial division", ms_td, nsm,
           ms_td_nodiv - ms_td_nosm, ms_td - ms_td_nodiv);
    printf("  %-30s %8.3f ms\n", "TD chain total",
           ms_rank + ms_emit + ms_sum + ms_scatter + ms_td);
    /* Both of these silently corrupt factorisations rather than crashing, so
     * they must fail the run. A norm that overflowed 256 bits is wrong, and a
     * truncated prime list leaves factors undivided. */
    if (hflags & TDF_NORM_OVERFLOW) {
        fprintf(stderr, "  ** NORM OVERFLOW: a norm exceeded %d bits\n", BN_LIMBS * 32);
        rc = -1;
    }
    if (hflags & TDF_LIST_TRUNCATED) {
        fprintf(stderr, "  ** %llu large-prime records past the %u/survivor cap\n",
                hovf, K);
        rc = -1;
    }

    /* ---- factorisation record, for relation output ----
     * A separate untimed pass with RECORD=1 rather than stores in the measured
     * kernel: the factors are only wanted when a run is emitting, and the hot
     * path should not carry writes it does not need. */
    if (cfg->emit_cof) {
        CK(mtlMalloc(&d_fac, (size_t)n * TD_FMAX * 4));
        CK(mtlMalloc(&d_faccnt, (size_t)n * 4));
        MTL_LAUNCH(k_td_1_1_0_0, blocks, threads, 0, 0, d_a, d_b, d_x, nullptr, n, cfg->logI, d_poly, cfg->side == 1 ? (uint32_t)L->q : 0u, d_plist, d_pcnt, K, d_sm, nsm, d_cof, d_cofbits, d_flags, nullptr, d_fac, d_faccnt, TD_FMAX, 0u);
        CK(mtlDeviceSynchronize()); CK(mtlGetLastError());
    }

    /* ---- classification: CADO's check_leftover_norm ---- */
    CK(mtlMalloc(&d_status, (size_t)n));
    ms_class = 1e30f;
    for (int rep = 0; rep < 3; rep++) {
        mtlEventRecord(t0);
        MTL_LAUNCH(k_classify, blocks, threads, 0, 0, d_cof, d_cofbits, d_b, n, cfg->lpb, cfg->mfb, (double)cfg->lim, d_status);
        mtlEventRecord(t1);
        CK(mtlEventSynchronize(t1)); CK(mtlGetLastError());
        { float t = time_kernel(t0, t1); if (t < ms_class) ms_class = t; }
    }

    /* ---- readback: cofactor sizes, gate, emission ---- */
    {
        bn_t *hcof = (bn_t *)malloc((size_t)n * sizeof(bn_t));
        uint8_t *hbits = (uint8_t *)malloc((size_t)n);
        uint8_t *hstat = (uint8_t *)malloc((size_t)n);
        int64_t *ha = (int64_t *)malloc((size_t)n * 8);
        int64_t *hb = (int64_t *)malloc((size_t)n * 8);
        uint32_t hist[257]; uint32_t nfully = 0;
        memset(hist, 0, sizeof hist);
        CK(mtlMemcpy(hcof, d_cof, (size_t)n * sizeof(bn_t), mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(hbits, d_cofbits, (size_t)n, mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(hstat, d_status, (size_t)n, mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(ha, d_a, (size_t)n * 8, mtlMemcpyDeviceToHost));
        CK(mtlMemcpy(hb, d_b, (size_t)n * 8, mtlMemcpyDeviceToHost));
        for (uint32_t k = 0; k < n; k++) {
            hist[hbits[k]]++;
            if (hbits[k] <= 1) nfully++;
        }
        printf("  %-30s %8u  (%.2f%% of survivors)\n",
               "cofactor == 1 (fully split)", nfully, 100.0 * nfully / n);
        printf("  cofactor bits:");
        for (int b = 0; b <= 256; b++)
            if (hist[b] && (b % 16 == 0 || hist[b] > n / 64))
                printf(" %d:%u", b, hist[b]);
        printf("\n");

        {
            uint32_t cs[6] = {0, 0, 0, 0, 0, 0};
            static const char *nm[6] = {"rejected: > mfb bits",
                                        "rejected: too few factors possible",
                                        "rejected: prime above 2^lpb",
                                        "ACCEPTED for cofactorisation",
                                        "already fully split",
                                        "rejected: b == 0, not a relation"};
            for (uint32_t k = 0; k < n; k++) if (hstat[k] < 6) cs[hstat[k]]++;
            printf("  %-30s %8.3f ms   (lpb %u, mfb %u, lim %u)\n",
                   "classify", ms_class, cfg->lpb, cfg->mfb, cfg->lim);
            for (int k = 0; k < 6; k++)
                printf("    %-32s %8u  (%.3f%%)\n", nm[k], cs[k],
                       100.0 * cs[k] / n);
            printf("    %-32s %8u\n", "-> this side's candidates",
                   cs[COF_ACCEPT] + cs[COF_SPLIT]);
        }

        if (cfg->cofgate &&
            td_gate_cofactors(cfg->cofgate, n, ha, hb, hcof, cfg->side) != 0)
            rc = -1;

        if (cfg->emit_cof) {
            FILE *fo = cfg->emit_cof ? fopen(cfg->emit_cof, "wb") : NULL;
            uint32_t *hfac = (uint32_t *)malloc((size_t)n * TD_FMAX * 4);
            uint32_t *hfn = (uint32_t *)malloc((size_t)n * 4);
            uint32_t checked = 0, bad = 0, overflowed = 0, maxfac = 0;
            CK(mtlMemcpy(hfac, d_fac, (size_t)n * TD_FMAX * 4, mtlMemcpyDeviceToHost));
            CK(mtlMemcpy(hfn, d_faccnt, (size_t)n * 4, mtlMemcpyDeviceToHost));
            /* Canonical order. The large primes arrive via an atomicAdd on the
             * survivor's slot counter, so their order in the list varies run to
             * run; sorting makes the emitted factorisation byte-reproducible,
             * which is what lets two paths be diffed against each other. */
            /* Unconditional: a candidate whose factor list overflowed TD_FMAX
             * has a TRUNCATED list, and every consumer reads faccnt entries.
             * Detecting that only on the verified q left later q emitting an
             * out-of-bounds read of the factor matrix. */
            for (uint32_t k = 0; k < n; k++) {
                uint32_t c = hfn[k] > TD_FMAX ? TD_FMAX : hfn[k];
                if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
                if (hfn[k] > maxfac) maxfac = hfn[k];
                if (hfn[k] > TD_FMAX) overflowed++;
                std::sort(hfac + (size_t)k * TD_FMAX,
                          hfac + (size_t)k * TD_FMAX + c);
            }

            /* Reconstruction gate: the recorded factors times the residual
             * cofactor must rebuild the exact norm. The CADO gate above checks
             * only the residual, so it would pass even if the factor list were
             * wrong; this checks the list. Run over the candidates, which are
             * the records that will actually be emitted. */
            for (uint32_t k = 0; cfg->td_verify && k < n; k++) {
                bns_t acc; bn_t t;
                int64_t a, b;
                uint64_t ua, ub;
                int sa, sb;
                if (hstat[k] != COF_ACCEPT && hstat[k] != COF_SPLIT) continue;
                if (hfn[k] > TD_FMAX) continue;
                checked++;
                a = ha[k]; b = hb[k];
                ua = (uint64_t)(a < 0 ? -a : a); ub = (uint64_t)(b < 0 ? -b : b);
                sa = (a < 0) ? -1 : 1; sb = (b < 0) ? -1 : 1;
                bns_zero(&acc);
                for (int d = 0; d <= h_poly.deg; d++) {
                    int sgn = h_poly.sign[d];
                    t = h_poly.c[d];
                    if (bn_is_zero(&t)) continue;
                    for (int e = 0; e < d; e++) { bn_mul_u64(&t, ua); sgn *= sa; }
                    for (int e = 0; e < h_poly.deg - d; e++) { bn_mul_u64(&t, ub); sgn *= sb; }
                    bns_addmag(&acc, &t, sgn);
                }
                /* rebuild: cofactor * prod(factors) */
                t = hcof[k];
                for (uint32_t z = 0; z < hfn[k]; z++)
                    bn_mul_u64(&t, (uint64_t)hfac[(size_t)k * TD_FMAX + z]);
                if (bn_cmp(&t, &acc.m) != 0) bad++;
            }
            if (cfg->td_verify)
                printf("  %-30s %8u of %u   %s\n",
                       "factors x cofactor == norm", checked - bad, checked,
                       bad ? "FAIL" : "PASS");
            if (bad) rc = -1;
            if (cfg->td_verify)
                printf("  %-30s %8u of %d\n", "most factors on a candidate",
                       maxfac, TD_FMAX);
            if (overflowed) {
                fprintf(stderr, "  ** %u candidates had more than %d factors;"
                        " raise TD_FMAX\n", overflowed, TD_FMAX);
                rc = -1;
            }

            if (cfg->emit_cof && !fo) { perror(cfg->emit_cof); rc = -1; }
            else if (fo) {
                char buf[BN_DEC_MAX];
                for (uint32_t k = 0; k < n; k++) {
                    int64_t a = ha[k], b = hb[k];
                    if (b < 0) { a = -a; b = -b; }
                    fprintf(fo, "%lld %lld %s %u %u %u", (long long)a, (long long)b,
                            bn_to_dec(&hcof[k], buf), hbits[k], hstat[k], hfn[k]);
                    for (uint32_t z = 0; z < hfn[k] && z < TD_FMAX; z++)
                        fprintf(fo, " %u", hfac[(size_t)k * TD_FMAX + z]);
                    fputc('\n', fo);
                }
                /* a short write shows up at fclose, and this file is now an
                 * input to the next process in the pipeline */
                if (ferror(fo)) rc = -1;
                if (fclose(fo)) { perror(cfg->emit_cof); rc = -1; }
                else if (rc == 0)
                    printf("  wrote %u (a, b, cofactor, bits, status, factors)"
                           " to %s\n", n, cfg->emit_cof);
            }
            free(hfac); free(hfn);
        }
        free(hcof); free(hbits); free(hstat); free(ha); free(hb);
    }

done:
    mtlEventDestroy(t0); mtlEventDestroy(t1);
    free(h_sm);
    mtlFree(d_cnt); mtlFree(d_gbase); mtlFree(d_bsum); mtlFree(d_sum);
    mtlFree(d_x); mtlFree(d_a); mtlFree(d_b);
    mtlFree(d_plist); mtlFree(d_pcnt); mtlFree(d_flags);
    mtlFree(d_cof); mtlFree(d_cofbits); mtlFree(d_poly); mtlFree(d_sm);
    mtlFree(d_ovf); mtlFree(d_status); mtlFree(d_fac); mtlFree(d_faccnt);
    return rc;
}

typedef struct {
    uint32_t w, m, magic, sh, ref;
} td_mod_case_t;

/* device code: see metal/bench_kernels.metal and metal/td.metal */

/* Device half of the reciprocal gate. slabtest calls the same td_mod_magic()
 * source on the CPU; this gate additionally executes the __umulhi branch used
 * by k_td so a CUDA-codegen/device-only regression cannot hide behind the
 * host implementation. It runs only under --verify. */
static int verify_td_mod_device(void)
{
    enum { NCASE = 4096 };
    td_mod_case_t *h = NULL, *d = NULL;
    uint32_t *d_bad = NULL, bad = UINT32_MAX, seed = 0x7f4a7c15u;
    int rc = -1;

    h = (td_mod_case_t *)malloc(sizeof(*h) * NCASE);
    if (!h) return -1;
    for (uint32_t k = 0; k < NCASE; k++) {
        uint32_t m, magic, sh, w;
        if (k < 20) {
            static const uint32_t edge[] = {
                2,3,4,5,7,8,15,16,31,32,63,64,127,128,255,256,
                1023,32767,131071,1048575
            };
            m = edge[k];
        } else {
            seed = seed * 1664525u + 1013904223u;
            m = 2u + seed % 1048574u;
        }
        td_magic_build(m, &magic, &sh);
        seed = seed * 1664525u + 1013904223u;
        w = (k & 3u) == 0 ? 0x7fffffffu : (seed & 0x7fffffffu);
        h[k].w = w; h[k].m = m; h[k].magic = magic; h[k].sh = sh;
        h[k].ref = w % m;
    }
#define MODDEV_CK(x) do { if (CUDA_CHECKED(x)) goto out; } while (0)
    MODDEV_CK(mtlMalloc(&d, sizeof(*h) * NCASE));
    MODDEV_CK(mtlMalloc(&d_bad, sizeof(*d_bad)));
    MODDEV_CK(mtlMemcpy(d, h, sizeof(*h) * NCASE, mtlMemcpyHostToDevice));
    MODDEV_CK(mtlMemcpy(d_bad, &bad, sizeof(bad), mtlMemcpyHostToDevice));
    MTL_LAUNCH(k_verify_td_mod_cases, 16, 256, 0, 0, d, NCASE, d_bad);
    MODDEV_CK(mtlDeviceSynchronize());
    MODDEV_CK(mtlGetLastError());
    MODDEV_CK(mtlMemcpy(&bad, d_bad, sizeof(bad), mtlMemcpyDeviceToHost));
    if (bad != UINT32_MAX) {
        const td_mod_case_t c = h[bad];
        fprintf(stderr,
                "[verify] device td_mod mismatch case %u: w=%u m=%u"
                " magic=%u sh=%u ref=%u\n",
                bad, c.w, c.m, c.magic, c.sh, c.ref);
        goto out;
    }
    rc = 0;
out:
    mtlFree(d); mtlFree(d_bad); free(h);
#undef MODDEV_CK
    return rc;
}

extern "C" int run_bench(const fb_t *fb, const fb_t *fbs, const qlat_t *L,
                         const poly_t *POLY, const bench_cfg_t *cfg)
{
    const uint32_t I = 1u << cfg->logI;
    const uint32_t xmax = I * cfg->J;
    const int log_region = cfg->log_region;
    const uint32_t nregion = xmax >> log_region;
    const uint32_t nbitword = xmax >> 5;   /* survivor bitmap, 1 bit/position */
    const int log_super = log_region + 7;             /* 128 regions/super */
    const uint32_t nsuper = xmax >> log_super;
    const uint32_t CINIT = (cfg->cell_bits == 16) ? 4096u : 255u;
    uint32_t BOUND = 0;
    size_t optin_smem_limit = 0;
    norm_t N;

    if (!fb_is_transform_validated(fb) ||
        (fbs && fbs->n && !fb_is_transform_validated(fbs))) {
        fprintf(stderr,
                "run_bench: refusing an unvalidated factor base;"
                " call fb_validate() before splitting or uploading it\n");
        return -1;
    }
    /* Validate the threshold before norm_setup(), allocations, or launches.
     * run_bench is an exported consumer boundary and must remain safe even
     * when called directly instead of through bench_main's CLI parser. */
    if (sieve_bound_checked(cfg->scale, cfg->allowance, CINIT, &BOUND,
                            cfg->side == 1
                                ? "run_bench side 1 survivor parameters"
                                : "run_bench side 0 survivor parameters"))
        return -1;

    /* Exact trial division constructs the full homogeneous norm. Reject a
     * shape that cannot fit before allocating or launching anything on the
     * GPU; the logarithmic sieve itself remains usable at any supported
     * degree. */
    memset(&N, 0, sizeof(N));
    norm_setup(&N, POLY, L, cfg->logI, cfg->J, cfg->scale, cfg->side == 1);
    if (cfg->td && !norm_fits_exact(&N, BN_LIMBS * 32)) {
        const double bits = norm_exact_bound_bits(&N);
        const int need = bn_limbs_for_bits(bits);
        fprintf(stderr, "  exact degree-%d norm may require %.2f bits;"
                " the trial-division type holds %d\n",
                POLY->deg, bits, BN_LIMBS * 32);
        if (need)
            fprintf(stderr, "  rebuild with `make BN_LIMBS=%d`\n", need);
        else
            fprintf(stderr, "  no supported BN_LIMBS is wide enough"
                    " (the maximum is 16, i.e. 512 bits)\n");
        return -1;
    }

    if (mtl_optin_smem_limit(&optin_smem_limit)) return -1;

    size_t freeB = 0, totalB = 0;
    CK(mtlMemGetInfo(&freeB, &totalB));
    printf("  device memory: %.2f GB free of %.2f GB\n",
           freeB / 1073741824.0, totalB / 1073741824.0);
    if (cfg->verify) {
        printf("[verify] direct-TD reciprocal on device (__umulhi path)...\n");
        if (verify_td_mod_device()) return -1;
        printf("[verify] OK: 4096 device reciprocal cases through w=0x7fffffff\n");
    }

    int td_failed = 0;      /* a failed gate must reach the exit status */
    dev_bufs D; memset(&D, 0, sizeof(D));
    CK(mtlMalloc(&D.primes, (size_t)fb->n * 4));
    CK(mtlMalloc(&D.roots,  (size_t)fb->n * 4));
    CK(mtlMalloc(&D.plat,   (size_t)fb->n * sizeof(plat_t)));
    CK(mtlMalloc(&D.overflow, 4));
    CK(mtlMalloc(&D.nproj, 4));
    CK(mtlMalloc(&D.nlost, 8));
    CK(mtlMemcpy(D.primes, fb->primes, (size_t)fb->n * 4, mtlMemcpyHostToDevice));
    CK(mtlMemcpy(D.roots,  fb->roots,  (size_t)fb->n * 4, mtlMemcpyHostToDevice));

    /* ---- slices: bucket record hint -> log p ---- */
    uint16_t *hslice = NULL, *hlogp = NULL;
    uint32_t nslice_pow2 = 1;
    int32_t nslice_rc = fb_build_slices(fb, &hslice, &hlogp, &nslice_pow2);
    if (nslice_rc < 0) {
        report_slice_build_error();
        mtlFree(D.primes); mtlFree(D.roots); mtlFree(D.plat);
        mtlFree(D.overflow); mtlFree(D.nproj); mtlFree(D.nlost);
        return -1;
    }
    uint32_t nslice = (uint32_t)nslice_rc;
    printf("  factor base cut into %u slices (padded to %u), log p in [%u,%u] bits\n",
           nslice, nslice_pow2, hlogp[0], hlogp[nslice - 1]);
    CK(mtlMalloc(&D.slice, (size_t)fb->n * 2));
    CK(mtlMalloc(&D.slice_logp, (size_t)nslice_pow2 * 2));
    CK(mtlMemcpy(D.slice, hslice, (size_t)fb->n * 2, mtlMemcpyHostToDevice));
    CK(mtlMemcpy(D.slice_logp, hlogp, (size_t)nslice_pow2 * 2, mtlMemcpyHostToDevice));

    /* ---- small primes: transform on the host (a few thousand entries) and
     * split into the three load-balance tiers. Entries arrive sorted by p, so
     * the tier boundaries are just two indices. ---- */
    uint32_t nsmall = 0, nblk = 0, nwrp = 0;
    uint32_t *hsp = NULL, *hsrt = NULL, *hsg = NULL; uint16_t *hslp = NULL;
    uint32_t *hsmag = NULL;
    /* per-q HOST work, billed separately: it is invisible to cudaEvent timing
     * and Goal 1 is a claim about host demand. */
    double h_ms_transform = 0, h_ms_sort = 0, h_ms_xfer = 0;
    if (cfg->small_sieve && fbs && fbs->n) {
        uint32_t i, k = 0, nrow = 0, nprj = 0;
        /* PINNED, not malloc'd. These four are the only per-special-q host->
         * device transfer, and on this box (WSL2) a pageable mtlMemcpy of
         * this size costs ~1.5 ms per call against ~0.1 ms pinned -- measured
         * at 6.0 ms vs 0.3 ms for the four together. That is per side, so
         * ~12 ms per special-q of pure host overhead, which dwarfs everything
         * else in this block and is a Goal-1 cost. */
        CK(mtlHostAlloc((void **)&hsp,  (size_t)fbs->n * 4, mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&hsrt, (size_t)fbs->n * 4, mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&hslp, (size_t)fbs->n * 2, mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&hsg,  (size_t)fbs->n * 4, mtlHostAllocDefault));
        CK(mtlHostAlloc((void **)&hsmag, (size_t)fbs->n * 4, mtlHostAllocDefault));
        h_ms_transform = host_ms();
        for (i = 0; i < fbs->n; i++) {
            uint32_t q = fbs->primes[i], r = fbs->roots[i], rt, g, m;
            m = pl_transform_enc(q, r, L->a0, L->a1, L->b0, L->b1, &rt, &g);
            if (g > 1) nrow++;
            if (r >= q) nprj++;
            hsp[k] = m; hsrt[k] = rt; hsg[k] = g;
            hslp[k] = fbs->logp[i];
            k++;
        }
        nsmall = k;
        h_ms_transform = host_ms() - h_ms_transform;

        /* Tier by the EFFECTIVE modulus m, not by q: an entry with q = 32768
         * and g = 32768 has m = 1 and hits every position in its rows, so
         * leaving it in the thread-per-entry tier would hand one thread the
         * whole region. Sorting by m puts every entry in the tier sized for
         * the number of hits it actually produces.
         *
         * This was an insertion sort, which is O(n^2) and ran ~3.3M
         * comparisons at nsmall ~3.6K. `qsort` is NOT a drop-in replacement:
         * the four arrays below are parallel and must be permuted together,
         * and no C library sort can do that. Sort a permutation of indices by
         * the key, then scatter -- which also keeps the device side SoA, which
         * is what the small-sieve kernel wants. */
        h_ms_sort = host_ms();
        {
            uint32_t *idx = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *tp  = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *trt = (uint32_t *)malloc((size_t)nsmall * 4);
            uint32_t *tg  = (uint32_t *)malloc((size_t)nsmall * 4);
            uint16_t *tlp = (uint16_t *)malloc((size_t)nsmall * 2);
            for (i = 0; i < nsmall; i++) idx[i] = i;
            /* stable_sort, not sort: the insertion sort this replaces was
             * stable, and ties in m are common (many entries share a modulus).
             * Stability keeps the output bit-identical to the old code, which
             * is what makes the bitmap regression test meaningful. */
            std::stable_sort(idx, idx + nsmall,
                             [hsp](uint32_t a, uint32_t b) { return hsp[a] < hsp[b]; });
            for (i = 0; i < nsmall; i++) {
                uint32_t s = idx[i];
                tp[i] = hsp[s]; trt[i] = hsrt[s]; tg[i] = hsg[s]; tlp[i] = hslp[s];
            }
            memcpy(hsp,  tp,  (size_t)nsmall * 4);
            memcpy(hsrt, trt, (size_t)nsmall * 4);
            memcpy(hsg,  tg,  (size_t)nsmall * 4);
            memcpy(hslp, tlp, (size_t)nsmall * 2);
            free(idx); free(tp); free(trt); free(tg); free(tlp);
        }
        h_ms_sort = host_ms() - h_ms_sort;

        /* Reciprocals AFTER the sort: the key is the modulus itself, so
         * building them earlier would mean permuting them too, for no gain. */
        for (i = 0; i < nsmall; i++)
            ss_magic_build(hsp[i],
                           hsg[i] > 1 ? cfg->J / hsg[i] : cfg->J,
                           cfg->logI, &hsmag[i]);

        for (i = 0; i < nsmall && hsp[i] < SS_BLOCK_CUT; i++) nblk = i + 1;
        for (i = 0; i < nsmall && hsp[i] < SS_WARP_CUT;  i++) nwrp = i + 1;
        CK(mtlMalloc(&D.smag, (size_t)nsmall * 4));
        CK(mtlMalloc(&D.sp,  (size_t)nsmall * 4));
        CK(mtlMalloc(&D.srt, (size_t)nsmall * 4));
        CK(mtlMalloc(&D.sg,  (size_t)nsmall * 4));
        CK(mtlMalloc(&D.slp, (size_t)nsmall * 2));
        /* Drain everything queued earlier (the factor-base uploads above are
         * large and asynchronous) BEFORE starting the clock -- otherwise the
         * trailing sync below bills their tail to this transfer and reports
         * milliseconds for 54 KB. */
        CK(mtlDeviceSynchronize());
        h_ms_xfer = host_ms();
        CK(mtlMemcpy(D.sp,  hsp,  (size_t)nsmall * 4, mtlMemcpyHostToDevice));
        CK(mtlMemcpy(D.srt, hsrt, (size_t)nsmall * 4, mtlMemcpyHostToDevice));
        CK(mtlMemcpy(D.sg,  hsg,  (size_t)nsmall * 4, mtlMemcpyHostToDevice));
        CK(mtlMemcpy(D.slp, hslp, (size_t)nsmall * 2, mtlMemcpyHostToDevice));
        CK(mtlMemcpy(D.smag, hsmag, (size_t)nsmall * 4, mtlMemcpyHostToDevice));
        CK(mtlDeviceSynchronize());
        h_ms_xfer = host_ms() - h_ms_xfer;
        {   double upd = 0; uint32_t xm = (1u << cfg->logI) * cfg->J;
            for (i = 0; i < nsmall; i++) upd += (double)xm / hsp[i] / hsg[i];
            printf("  small sieve: %u entries (%u block-tier m<%u, %u warp-tier m<%u,"
                   " %u thread-tier), %u with a row divisor, %u projective,"
                   " %.3e updates\n",
                   nsmall, nblk, SS_BLOCK_CUT, nwrp - nblk, SS_WARP_CUT,
                   nsmall - nwrp, nrow, nprj, upd);
        }
    }

    /* ---- norm initialisation constants ---- */
    /* las's survivor test is S = max(T - sum, 0) <= bound with
     * bound = round(scale * lambda * lpb). Ours holds CINIT - T + sum, so
     * S = CINIT - cell and the test becomes cell >= CINIT - bound. */
    /* las: bound = (unsigned char)(lambda*lpb*scale + LOGNORM_GUARD_BITS),
     * las-norms.cpp:270 -- a TRUNCATING cast plus a guard bit, not a round.
     * With the exact scales (1.275 / 1.925, not the 2-dp values las prints)
     * this reproduces both of las's bounds exactly: 143 and 141. The old
     * round(scale*allowance) matched only because the rounded scales happened
     * to compensate. */
    uint32_t tconst;
    {
        float t = norm_target_host(&N, 0, cfg->J / 2);
        int ti = (int)(t + 0.5f);
        tconst = (ti < 1) ? 1u : ((uint32_t)ti > 255u ? 255u : (uint32_t)ti);
        printf("  init T at (i=0, j=J/2) = %u; survivor bound = %u"
               " (scale %.3f x %.2f bits)\n", tconst, BOUND, cfg->scale, cfg->allowance);
    }

    mtlEvent_t e0, e1, e2, e3, e4;
    mtlEventCreate(&e0); mtlEventCreate(&e1); mtlEventCreate(&e2);
    mtlEventCreate(&e3); mtlEventCreate(&e4);

    int blocks = cfg->blocks ? cfg->blocks : 48 * 6;
    /* Fill's grid is absolute, not per-SM -- see FILL_BLOCKS_DEFAULT. */
    const int fblocks = cfg->fill_blocks ? cfg->fill_blocks : FILL_BLOCKS_DEFAULT;
    const int fthreads = cfg->fill_threads ? cfg->fill_threads : FILL_THREADS_DEFAULT;
    float t_trans = 0, t_fill = 0, t_l1 = 0, t_l2 = 0;

    /* ---- stage T ---- *
     * Untimed warm-up. k_transform is the FIRST kernel of the run, so without
     * this it absorbs the whole one-time CUDA cost -- module load for a
     * four-architecture fatbin, context setup -- and reports it divided by
     * reps. Measured on WSL2 that fixed cost is ~170-220 ms, which put the
     * "transform" line at 71.2 ms at --reps 3 against a true 0.55 ms: a 98x
     * swing across --reps 3..1000 while fill and apply moved under 1%. Two
     * different people compared that number across GPUs before anyone noticed
     * it was measuring startup. The memsets follow the warm-up because nproj
     * and nlost are accumulators divided by reps. */
    MTL_LAUNCH(k_transform_0, blocks, cfg->threads, 0, 0, D.primes, D.roots, D.plat, fb->n, cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, D.nproj, D.nlost, nullptr);
    CK(mtlDeviceSynchronize());
    CK(mtlMemset(D.nproj, 0, 4));
    CK(mtlMemset(D.nlost, 0, 8));
    mtlEventRecord(e0);
    for (int rep = 0; rep < cfg->reps; rep++)
        MTL_LAUNCH(k_transform_0, blocks, cfg->threads, 0, 0, D.primes, D.roots, D.plat, fb->n, cfg->logI, cfg->J, L->a0, L->a1, L->b0, L->b1, D.nproj, D.nlost, nullptr);
    mtlEventRecord(e1);
    CK(mtlEventSynchronize(e1));
    CK(mtlGetLastError());
    t_trans = time_kernel(e0, e1) / cfg->reps;

    uint32_t hproj = 0;
    unsigned long long hlost = 0;
    CK(mtlMemcpy(&hproj, D.nproj, 4, mtlMemcpyDeviceToHost));
    CK(mtlMemcpy(&hlost, D.nlost, 8, mtlMemcpyDeviceToHost));
    hproj /= cfg->reps;
    hlost /= (unsigned)cfg->reps;

    /* ---- expected record count, for sizing ---- */
    double exp_rec = 0;
    for (uint32_t i = 0; i < fb->n; i++) exp_rec += (double)xmax / fb->primes[i];
    uint64_t est = (uint64_t)(exp_rec * 1.15) + 4096;

    printf("  transformed roots: %u row-confined (g > 1), %llu positions lost%s\n",
           hproj, hlost, hlost ? "  ** move these to the small tier **" : "");
    printf("  analytic records : %.3e   (sized with 1.15x margin)\n", exp_rec);

    uint32_t cap = 0;               /* records per region, set by the fill path */
    if (cfg->fill_mode == FILL_ATOMIC) {
        cap = (uint32_t)(est / nregion) + 256;
        size_t need = (size_t)nregion * cap * cfg->record_bytes;
        printf("  single-level: %u buckets x cap %u x %d B = %.2f GB\n",
               nregion, cap, cfg->record_bytes, need / 1073741824.0);
        if (need + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(mtlMalloc(&D.cursor, (size_t)nregion * 4));
        CK(mtlMalloc(&D.out, need));
        CK(mtlMemset(D.overflow, 0, 4));
/* One dispatch for the control fill AND every concurrency arm below. Two copies
 * of this three-way record-width switch would let the experiment silently
 * measure a kernel specialisation the control never runs. */
#define FILL_ONE(GRID, STREAM, PLAT, CUR, OUT, OVF)                          \
    do {                                                                     \
        if (cfg->record_bytes == 2)                                          \
            MTL_LAUNCH(k_fill_atomic_2_0, (GRID), fthreads, 0, (STREAM), (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region, (CUR), (OUT), cap, (OVF), nullptr, nullptr);                       \
        else if (cfg->record_bytes == 4)                                     \
            MTL_LAUNCH(k_fill_atomic_4_0, (GRID), fthreads, 0, (STREAM), (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region, (CUR), (OUT), cap, (OVF), nullptr, nullptr);                       \
        else                                                                 \
            MTL_LAUNCH(k_fill_atomic_8_0, (GRID), fthreads, 0, (STREAM), (PLAT), D.slice, fb->n, xmax, cfg->logI, log_region, (CUR), (OUT), cap, (OVF), nullptr, nullptr);                       \
    } while (0)

        mtlEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
            FILL_ONE(fblocks, 0, D.plat, D.cursor, D.out, D.overflow);
        }
        mtlEventRecord(e3);
        CK(mtlEventSynchronize(e3));
        CK(mtlGetLastError());
        t_fill = time_kernel(e2, e3) / cfg->reps;

        /* ---- item 1: is fill's knee per-KERNEL or per-DEVICE? ----
         *
         * Every card swept so far plateaus at the same ABSOLUTE block count,
         * which is the shape of a device limit; but the 4090 is 1.80x SLOWER
         * at fill than a 5070 with 1.5x its bandwidth, which no device limit
         * explains. The `ncu` profile named the candidate: waves per SM = 1.00,
         * so the whole grid is resident at once and a block that draws a heavy
         * chunk has no queued block to backfill its slot -- SMs idle 26.5% of
         * elapsed cycles. A second INDEPENDENT kernel could occupy those.
         *
         *   CONCURRENT  N workspaces, N streams, fblocks each   -- N workspaces
         *   SERIAL      the same N launches, one stream         -- N workspaces
         *   WIDE        one launch at N * fblocks               -- ONE workspace
         *
         * CONCURRENT vs SERIAL is the comparison: identical work, identical
         * launches, only the stream assignment differs. CONCURRENT ~= SERIAL
         * says the device is saturated and the plateau is real; CONCURRENT <
         * SERIAL says one kernel cannot feed the card.
         *
         * WIDE DOES 1/N THE WORK OF THE OTHER TWO and is not comparable to
         * them in raw ms. k_fill_atomic is a grid-stride loop over fb->n, so
         * N*fblocks blocks still process one factor base into one workspace.
         * It answers a different question -- can a single kernel buy the same
         * capacity just by being wider? -- and is only ever quoted per
         * workspace, which is why every row below is normalised.
         *
         * ARM ORDER AND DRIFT. Boost clocks decay under sustained load, and a
         * fixed arm order would bias whichever arm runs first -- here, in
         * exactly the direction of the conclusion. Arms are therefore
         * interleaved and each keeps its MINIMUM over the outer repeats, the
         * same best-of-N every other A/B in this file uses.
         *
         * WHAT THE WORKSPACES DO AND DO NOT MODEL. Each gets its own plat,
         * cursor and bucket arrays, so the concurrent arm streams N x 42 B per
         * entry from DISTINCT addresses -- sharing one plat would hand it an
         * L2 advantage no pair of real special-q enjoys. `slice` stays shared
         * because two real q on one factor base share it too. But the copies
         * hold IDENTICAL plat values, and the values are the walk: two real q
         * would write different per-region distributions, while these march
         * their bucket frontiers in lockstep. Finding 81 makes that exactly
         * the variable fill is bound by (read-modify-write on partly filled
         * lines), so this measures the SATURATION question honestly and does
         * NOT predict how two real q interleave. That needs the pipeline. */
        if (cfg->fill_streams > 1) {
            int NS = cfg->fill_streams;
            /* run_bench is an exported boundary; the argv clamp lives in a
             * different translation unit and cannot be relied on here. */
            if (NS > FILL_STREAMS_MAX) NS = FILL_STREAMS_MAX;
            const size_t wsz = (size_t)nregion * cap * cfg->record_bytes;
            const size_t psz = (size_t)fb->n * sizeof(plat_t);
            size_t extra = (size_t)(NS - 1) * (wsz + psz + (size_t)nregion * 4);
            size_t fnow = 0, tnow = 0;
            int maxgrid = 0;
            CK(mtlMemGetInfo(&fnow, &tnow));
            CK(mtl_grid_dim_x(&maxgrid));
            printf("\n  --- fill concurrency, %d workspaces (item 1) ---\n", NS);
            printf("  extra workspaces need %.2f GB of %.2f GB free\n",
                   extra / 1073741824.0, fnow / 1073741824.0);
            if (extra + 64u * 1024 * 1024 > fnow) {
                printf("  SKIP: %d workspaces need %.2f GB but only %.2f GB is"
                       " free%s\n", NS, extra / 1073741824.0,
                       fnow / 1073741824.0,
                       NS > 2 ? "; try --fill-streams 2" : "");
            } else if ((double)fblocks * NS > (double)maxgrid) {
                printf("  SKIP: the WIDE arm would need %d x %d = %.0f blocks,"
                       " past this device's grid-x limit %d\n",
                       fblocks, NS, (double)fblocks * NS, maxgrid);
            } else {
                plat_t   *wplat[FILL_STREAMS_MAX];
                uint32_t *wcur [FILL_STREAMS_MAX];
                uint8_t  *wout [FILL_STREAMS_MAX];
                mtlStream_t st[FILL_STREAMS_MAX];
                uint32_t *xovf = NULL;   /* NOT D.overflow -- see below */
                int k;
                wplat[0] = D.plat; wcur[0] = D.cursor; wout[0] = D.out;
                for (k = 1; k < NS; k++) {
                    CK(mtlMalloc(&wplat[k], psz));
                    CK(mtlMalloc(&wcur[k], (size_t)nregion * 4));
                    CK(mtlMalloc(&wout[k], wsz));
                    CK(mtlMemcpy(wplat[k], D.plat, psz, mtlMemcpyDeviceToDevice));
                }
                for (k = 0; k < NS; k++) CK(mtlStreamCreate(&st[k]));
                /* The run's overflow count is REPORTED and gates --verify, and
                 * these arms issue (2*NS+1)*reps more fills into the same
                 * buckets. Accumulating them into D.overflow would inflate the
                 * figure an operator sizes the rerun from, and would fail
                 * --verify on records this harness dropped rather than the
                 * measured configuration. Give the experiment its own counter
                 * and leave D.overflow untouched. */
                CK(mtlMalloc(&xovf, 4));
                CK(mtlMemset(xovf, 0, 4));

                float t_arm[3] = { 1e30f, 1e30f, 1e30f };
                const int outer = 3;
                for (int pass = 0; pass < outer; pass++) {
                    for (int arm = 0; arm < 3; arm++) {
                        const int grid = (arm == 2) ? fblocks * NS : fblocks;
                        const int nws  = (arm == 2) ? 1 : NS;
                        mtlEventRecord(e2);
                        for (int rep = 0; rep < cfg->reps; rep++) {
                            for (k = 0; k < nws; k++) {
                                if (arm == 0)
                                    CK(mtlMemsetAsync(wcur[k], 0,
                                        (size_t)nregion * 4, st[k]));
                                else    /* same synchronous memset the control
                                         * fill uses, so SERIAL and the control
                                         * share one harness */
                                    CK(mtlMemset(wcur[k], 0, (size_t)nregion * 4));
                            }
                            for (k = 0; k < nws; k++)
                                FILL_ONE(grid, arm == 0 ? st[k] : 0,
                                         wplat[k], wcur[k], wout[k], xovf);
                        }
                        /* Do not rely on legacy default-stream semantics to
                         * fence the blocking streams: one --default-stream
                         * per-thread in NVCCFLAGS would silently turn e3 into
                         * a launch-issue timestamp and report a 20x win. */
                        if (arm == 0)
                            for (k = 0; k < NS; k++) CK(mtlStreamSynchronize(st[k]));
                        mtlEventRecord(e3);
                        CK(mtlEventSynchronize(e3));
                        CK(mtlGetLastError());
                        float t = time_kernel(e2, e3) / cfg->reps;
                        if (t < t_arm[arm]) t_arm[arm] = t;
                    }
                }
                {
                    uint32_t xo = 0;
                    CK(mtlMemcpy(&xo, xovf, 4, mtlMemcpyDeviceToHost));
                    printf("  best of %d passes x %d reps, arms interleaved\n",
                           outer, cfg->reps);
                    printf("  CONCURRENT %2d x %5d blocks, %d streams : %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           NS, fblocks, NS, t_arm[0], t_arm[0] / NS);
                    printf("  SERIAL     %2d x %5d blocks, one stream : %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           NS, fblocks, t_arm[1], t_arm[1] / NS);
                    printf("  WIDE        1 x %5d blocks, ONE workspace: %8.3f ms"
                           "  = %7.3f ms per workspace\n",
                           fblocks * NS, t_arm[2], t_arm[2]);
                    printf("  concurrent/serial %.4f   (<1 = one kernel cannot"
                           " feed this card; same work, only the streams differ)\n",
                           t_arm[0] / t_arm[1]);
                    printf("  concurrent vs the best single kernel: %.4f"
                           "  (wide %.3f, single %.3f ms per workspace)\n",
                           (t_arm[0] / NS) / (t_arm[2] < t_fill ? t_arm[2] : t_fill),
                           t_arm[2], t_fill);
                    if (xo) printf("  note: %u records overflowed inside the"
                                   " concurrency arms (own counter; the run's"
                                   " own overflow figure is unaffected)\n", xo);
                }
                mtlFree(xovf);
                for (k = 0; k < NS; k++) CK(mtlStreamDestroy(st[k]));
                for (k = 1; k < NS; k++) {
                    mtlFree(wplat[k]); mtlFree(wcur[k]); mtlFree(wout[k]);
                }
            }
        }
#undef FILL_ONE
    } else {
        /* Both levels stage their fan-out in a fixed number of shared buffers,
         * so the split has to fit: L1 needs nsuper <= L1_NBUF and L2 needs
         * regions_per_super <= L2_NBUF. With 128 each, two-level tops out at
         * 128*128 = 16384 regions -- exactly I15e at region 2^15, and no more.
         * Past that it silently indexed cnt[] out of bounds and livelocked in
         * the retry loop. Refuse instead: the operating point that actually
         * won (region 2^14, 32768 regions) is out of reach for a two-level
         * split at these buffer sizes and would need a third level. */
        if (nsuper > L1_NBUF || (1u << (log_super - log_region)) > L2_NBUF) {
            printf("  two-level cannot express this split: %u super-buckets"
                   " (max %u) x %u regions each (max %u).\n"
                   "  Use --mode atomic, which is 2.7x faster anyway"
                   " (RESULTS.md finding 1).\n",
                   nsuper, L1_NBUF, 1u << (log_super - log_region), L2_NBUF);
            return 1;
        }
        uint32_t l1cap = (uint32_t)(est / nsuper) + 4096;
        uint32_t l2cap = (uint32_t)(est / nregion) + 256;
        size_t need1 = (size_t)nsuper * l1cap * 4;
        size_t need2 = (size_t)nregion * l2cap * cfg->record_bytes;
        printf("  two-level: L1 %u super x cap %u x 4 B = %.2f GB;"
               " L2 %u regions x cap %u x %d B = %.2f GB\n",
               nsuper, l1cap, need1 / 1073741824.0,
               nregion, l2cap, cfg->record_bytes, need2 / 1073741824.0);
        if (need1 + need2 + 64u * 1024 * 1024 > freeB) {
            printf("  SKIP: does not fit in free device memory\n"); return 1;
        }
        CK(mtlMalloc(&D.l1, need1));
        CK(mtlMalloc(&D.l1cnt, (size_t)nsuper * 4));
        CK(mtlMalloc(&D.cursor, (size_t)nregion * 4));
        CK(mtlMalloc(&D.out, need2));
        CK(mtlMemset(D.overflow, 0, 4));

        mtlEventRecord(e2);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(mtlMemset(D.l1cnt, 0, (size_t)nsuper * 4));
            /* Its own DEFAULT, but --fill-blocks still reaches it. k_fill_l1 is
             * the twolevel path's level-1 kernel and has never been swept; the
             * 1152 x 32 geometry was measured on k_fill_atomic, a different
             * kernel with a different write pattern, so inheriting it as a
             * default would move an experimental path nobody remeasured.
             *
             * Pinning it outright was worse: it made --fill-blocks a silent
             * no-op under --mode twolevel while the startup line still printed
             * the requested number, which is how a run gets quoted as a sweep
             * that never happened. An explicit value wins here as everywhere
             * else; only the default differs. Threads stay at 512 -- never
             * swept either, and --fill-threads has no measured meaning for this
             * kernel. */
            MTL_LAUNCH(k_fill_l1, cfg->fill_blocks ? cfg->fill_blocks : FILL_L1_BLOCKS, 512, 0, 0, D.plat, fb->n, xmax, cfg->logI, log_super, D.l1cnt, D.l1, l1cap, D.overflow);
        }
        mtlEventRecord(e3);
        for (int rep = 0; rep < cfg->reps; rep++) {
            CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
            if (cfg->record_bytes == 2)
                MTL_LAUNCH(k_fill_l2_2, nsuper, 512, 0, 0, D.l1, D.l1cnt, l1cap, log_region, log_super, D.cursor, D.out, l2cap, D.overflow);
            else
                MTL_LAUNCH(k_fill_l2_4, nsuper, 512, 0, 0, D.l1, D.l1cnt, l1cap, log_region, log_super, D.cursor, D.out, l2cap, D.overflow);
        }
        mtlEventRecord(e4);
        CK(mtlEventSynchronize(e4));
        CK(mtlGetLastError());
        t_l1 = time_kernel(e2, e3) / cfg->reps;
        t_l2 = time_kernel(e3, e4) / cfg->reps;
        t_fill = t_l1 + t_l2;
    }

    uint32_t ovf = 0;
    CK(mtlMemcpy(&ovf, D.overflow, 4, mtlMemcpyDeviceToHost));

    /* ---- count what actually landed ---- */
    uint64_t landed = 0;
    {
        uint32_t *h = (uint32_t *)malloc((size_t)nregion * 4);
        CK(mtlMemcpy(h, D.cursor, (size_t)nregion * 4, mtlMemcpyDeviceToHost));
        uint32_t mx = 0;
        for (uint32_t i = 0; i < nregion; i++) { landed += h[i]; if (h[i] > mx) mx = h[i]; }
        printf("  records landed   : %llu   (max bucket %u, mean %.0f -> imbalance %.2fx)\n",
               (unsigned long long)landed, mx, (double)landed / nregion,
               mx / ((double)landed / nregion));
        /* PER-REGION assertion, not just the total. A global count cannot tell
         * "right total, wrong region" from "right" -- and every placement bug
         * this project has hit (transposed basis, projective reciprocal) had
         * exactly the right total. This FAILS the run rather than printing. */
        if (cfg->verify) {
            uint32_t *ref = (uint32_t *)malloc((size_t)nregion * 4);
            uint64_t tot;
            uint32_t bad = 0, first = 0;
            printf("  [verify] per-region reference (single-threaded)...\n");
            tot = verify_count_updates(fb, L, cfg->logI, cfg->J, log_region, ref);
            for (uint32_t i = 0; i < nregion; i++)
                if (ref[i] != h[i]) { if (!bad) first = i; bad++; }
            if (bad || tot != landed) {
                printf("  [verify] FAILED: %u of %u regions differ (first: region %u,"
                       " gpu %u ref %u); totals gpu %llu ref %llu\n",
                       bad, nregion, first, h[first], ref[first],
                       (unsigned long long)landed, (unsigned long long)tot);
                free(ref); free(h); return -1;
            }
            printf("  [verify] OK: all %u regions match the CPU reference exactly\n"
                   "           (counts are of records ATTEMPTED; bucket overflow is\n"
                   "            gated separately, above)\n", nregion);
            free(ref);
        }
        free(h);
    }
    if (ovf) {
        printf("  ** OVERFLOW: %u records dropped -- resize and re-run **\n", ovf);
        /* The per-region gate CANNOT see this. cursor[] is incremented before
         * the cap test, so it counts records ATTEMPTED, and the CPU reference
         * counts the same thing -- they agree exactly while apply silently
         * truncates each bucket at cap and drops the excess. Overflow has to be
         * fatal on its own, or verification certifies a sieve that lost data. */
        if (cfg->verify) {
            fprintf(stderr, "[verify] FAILED: %u records overflowed their bucket."
                    " The per-region count gate cannot detect this (cursors count\n"
                    "         attempts, not stores), so it is failed explicitly.\n", ovf);
            return -1;
        }
    }

    /* ---- stage A: apply ---- */
    float t_apply = 0;
    if (cfg->stage != STAGE_FILL) {
        if (cfg->fill_mode != FILL_ATOMIC || cfg->record_bytes != 4) {
            printf("\n  apply needs single-level 4 B records (--mode atomic"
                   " --record-bytes 4); skipping\n");
        } else {
            const int CB = cfg->cell_bits;
            const uint32_t ncell = 1u << log_region;
            const size_t smem = (size_t)ncell * CB / 8 + (size_t)nslice_pow2 * 2;
            const uint32_t maxsurv = 1u << 22;
            /* gate 5: the one position whose pipeline-produced cell we read back */
            const uint32_t probe_x = (cfg->probe_j != 0xFFFFFFFFu)
                ? (uint32_t)((cfg->probe_i + (1 << (cfg->logI - 1)))
                             + ((uint64_t)cfg->probe_j << cfg->logI))
                : 0xFFFFFFFFu;
            int athr = cfg->apply_threads ? cfg->apply_threads : 512;
            /* region 0 is the j=0 row and is legitimately almost empty --
             * gating on it would check nothing. Use a mid-range region. */
            const uint32_t dbgreg = nregion / 2;

            CK(mtlMalloc(&D.probe, 8));
            CK(mtlMemset(D.probe, 0, 8));
            CK(mtlMalloc(&D.nsurv, 4));
            if (cfg->survbits || cfg->other_bits) {
                CK(mtlMalloc(&D.survbits, (size_t)nbitword * 4));
                if (cfg->survbits)
                    printf("  writing a survivor bitmap (1 bit per position, x"
                           " order) to %s (%.0f MB)\n",
                           cfg->survbits, nbitword * 4 / 1048576.0);
            }
            if (cfg->not_both_even)
                printf("  las `not_both_even` filter ON: positions with i and j"
                       " both even cannot survive\n");
            if (cfg->dump) {
                CK(mtlMalloc(&D.dumpbuf, (size_t)xmax));
                printf("  dumping the region in las byte convention to %s"
                       " (%.0f MB)\n", cfg->dump, xmax / 1048576.0);
            }
            if (cfg->verify) CK(mtlMalloc(&D.dbg, (size_t)ncell * 2));

            printf("\n  apply: %u regions x %u cells x %d bit = %zu B smem/block,"
                   " %d threads, %s, norm=%s\n",
                   nregion, ncell, CB, smem, athr,
                   cfg->apply_atomic ? "smem atomicAdd" : "PLAIN (racy probe)",
                   cfg->norm_mode == NORM_CONST ? "const" : "horner");
            if (smem > optin_smem_limit) {
                printf("  SKIP: %zu B exceeds this device's %zu B opt-in"
                       " shared-memory limit\n", smem, optin_smem_limit);
                goto after_apply;
            }

#define LAUNCH_APPLY(CBV, AT, NM)                                              \
            do {                                                               \
                CK(mtlFuncSetMaxThreadgroupMemory("k_apply_" #CBV "_" #AT "_" #NM "_0", smem));  \
                mtlEventRecord(e3);                                           \
                for (int rep = 0; rep < cfg->reps; rep++) {                    \
                    CK(mtlMemset(D.nsurv, 0, 4));                             \
                    if (D.survbits)                                            \
                        CK(mtlMemset(D.survbits, 0, (size_t)nbitword * 4));   \
                    MTL_LAUNCH_NAMED("k_apply_" #CBV "_" #AT "_" #NM "_0", nregion, athr, smem, 0, (const uint32_t *)D.out, D.cursor, cap, cfg->logI, log_region, D.slice_logp, nslice_pow2, N, CINIT, CINIT - BOUND, tconst, D.dumpbuf, D.nsurv, D.dbg, dbgreg, D.sp, D.srt, D.sg, D.slp, D.smag, nsmall, nblk, nwrp, probe_x, D.probe, D.survbits, cfg->not_both_even, 0u);                   \
                }                                                              \
                mtlEventRecord(e4);                                           \
                CK(mtlEventSynchronize(e4));                                  \
                CK(mtlGetLastError());                                        \
                t_apply = time_kernel(e3, e4) / cfg->reps;                      \
            } while (0)

            if (CB == 16) {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 1, 0);
                    else                              LAUNCH_APPLY(16, 1, 1);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(16, 0, 0);
                    else                              LAUNCH_APPLY(16, 0, 1);
                }
            } else {
                if (cfg->apply_atomic) {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 1, 0);
                    else                              LAUNCH_APPLY(8, 1, 1);
                } else {
                    if (cfg->norm_mode == NORM_CONST) LAUNCH_APPLY(8, 0, 0);
                    else                              LAUNCH_APPLY(8, 0, 1);
                }
            }
#undef LAUNCH_APPLY

            if (cfg->survbits) {
                uint32_t *h = (uint32_t *)malloc((size_t)nbitword * 4);
                FILE *fo = fopen(cfg->survbits, "wb");
                CK(mtlMemcpy(h, D.survbits, (size_t)nbitword * 4,
                              mtlMemcpyDeviceToHost));
                if (!fo) { perror(cfg->survbits); }
                else { fwrite(h, 4, (size_t)nbitword, fo); fclose(fo); }
                free(h);
            }

            /* ---- device intersect + primitive filter + compaction -------- */
            if (cfg->other_bits) {
                uint32_t *hother = NULL, *dother = NULL;
                uint32_t *d_x = NULL, *d_n = NULL, hn = 0;
                int64_t *d_a = NULL, *d_b = NULL;
                unsigned long long *d_pre = NULL, hpre = 0, *d_qb = NULL, hqb = 0;
                uint32_t *d_two = NULL;
                /* cap: the two-sided count is ~1 in 400 of the area, but size
                 * it off the one-sided count so a bad pairing cannot overflow
                 * silently. */
                const uint32_t icap = maxsurv;
                FILE *fi = fopen(cfg->other_bits, "rb");
                float t_isect = 0;

                if (!fi) { perror(cfg->other_bits); td_failed = 1; }
                else {
                    CK(mtlHostAlloc((void **)&hother, (size_t)nbitword * 4,
                                     mtlHostAllocDefault));
                    size_t got = fread(hother, 4, (size_t)nbitword, fi);
                    fclose(fi);
                    if (got != (size_t)nbitword) {
                        fprintf(stderr, "  --other-bits: short read (%zu of %u"
                                " words)\n", got, nbitword);
                        td_failed = 1;
                    } else {
                        CK(mtlMalloc(&dother, (size_t)nbitword * 4));
                        CK(mtlMemcpy(dother, hother, (size_t)nbitword * 4,
                                      mtlMemcpyHostToDevice));
                        CK(mtlMalloc(&d_x, (size_t)icap * 4));
                        CK(mtlMalloc(&d_a, (size_t)icap * 8));
                        CK(mtlMalloc(&d_b, (size_t)icap * 8));
                        CK(mtlMalloc(&d_n, 4));   CK(mtlMemset(d_n, 0, 4));
                        CK(mtlMalloc(&d_pre, 8)); CK(mtlMemset(d_pre, 0, 8));
                        CK(mtlMalloc(&d_qb, 8));  CK(mtlMemset(d_qb, 0, 8));
                        CK(mtlMalloc(&d_two, (size_t)nbitword * 4));
                        CK(mtlMemset(d_two, 0, (size_t)nbitword * 4));

                        /* Time it repeatedly: the first launch of any kernel
                         * carries module/context cost that is not part of the
                         * steady-state per-q price. Report the best. */
                        t_isect = 1e30f;
                        for (int rep = 0; rep < 3; rep++) {
                            CK(mtlMemset(d_n, 0, 4));
                            CK(mtlMemset(d_pre, 0, 8));
                            CK(mtlMemset(d_qb, 0, 8));
                            mtlEventRecord(e3);
                            MTL_LAUNCH(k_intersect_compact_1_0, blocks, cfg->threads, 0, 0, D.survbits, dother, nbitword, cfg->logI, L->a0, L->a1, L->b0, L->b1, (int64_t)L->q, d_x, d_a, d_b, icap, d_n, d_pre, d_two, d_qb, 0u);
                            mtlEventRecord(e4);
                            CK(mtlEventSynchronize(e4));
                            CK(mtlGetLastError());
                            float t = time_kernel(e3, e4);

                            if (t < t_isect) t_isect = t;
                        }

                        CK(mtlMemcpy(&hn, d_n, 4, mtlMemcpyDeviceToHost));
                        CK(mtlMemcpy(&hpre, d_pre, 8, mtlMemcpyDeviceToHost));
                        CK(mtlMemcpy(&hqb, d_qb, 8, mtlMemcpyDeviceToHost));

                        printf("\n  intersect+gcd+compact vs %s\n", cfg->other_bits);
                        printf("  %-26s %8llu\n", "two-sided, pre-gcd",
                               (unsigned long long)hpre);
                        /* THIS is the number to compare against dumpcmp --and
                         * and CADO's after_sieve: both apply the gcd(i,j) test
                         * and nothing else. */
                        printf("  %-26s %8llu  (%.1f%% of pre-gcd survive)"
                               "  <- CADO-comparable\n",
                               "primitive gcd(i,j)=1",
                               (unsigned long long)hn + hqb,
                               hpre ? 100.0 * (hn + hqb) / (double)hpre : 0.0);
                        printf("  %-26s %8llu  (q | b: (a,b) = q*(a',b'),"
                               " finding 68)\n",
                               "  of which dropped", (unsigned long long)hqb);
                        printf("  %-26s %8u\n", "emitted", hn);
                        if (hn > icap)
                            printf("  ** OVERFLOW: %u > cap %u, list truncated\n", hn, icap);
                        printf("  %-26s %8.3f ms  (best of 3; the first launch of any kernel\n"
       "                                       carries module-load cost -- 4.6 ms here)\n",
       "intersect+compact", t_isect);

                        if (cfg->emit && hn && hn <= icap) {
                            uint32_t *hx = (uint32_t *)malloc((size_t)hn * 4);
                            int64_t *ha = (int64_t *)malloc((size_t)hn * 8);
                            int64_t *hb = (int64_t *)malloc((size_t)hn * 8);
                            FILE *fo = fopen(cfg->emit, "wb");
                            CK(mtlMemcpy(hx, d_x, (size_t)hn * 4, mtlMemcpyDeviceToHost));
                            CK(mtlMemcpy(ha, d_a, (size_t)hn * 8, mtlMemcpyDeviceToHost));
                            CK(mtlMemcpy(hb, d_b, (size_t)hn * 8, mtlMemcpyDeviceToHost));
                            if (!fo) perror(cfg->emit);
                            else {
                                for (uint32_t z = 0; z < hn; z++)
                                    fprintf(fo, "%u %lld %lld\n", hx[z],
                                            (long long)ha[z], (long long)hb[z]);
                                fclose(fo);
                                printf("  wrote %u (x, a, b) to %s\n", hn, cfg->emit);
                            }
                            free(hx); free(ha); free(hb);
                        }
                    }
                    /* ---- recovery A/B ------------------------------------ */
                    if (d_two && hn && hn <= icap) {
                        const uint32_t nsum = nbitword / 2;          /* bits */
                        const uint32_t nsumword = (nsum + 31) / 32;
                        uint32_t *d_sum = NULL, *d_rx = NULL, *d_rp = NULL, *d_rn = NULL;
                        uint32_t *d_scratch = NULL;   /* so the plain purge cannot
                                                       * clobber A's (x,p) output */
                        unsigned long long *d_probe = NULL, *d_p1 = NULL, *d_rd = NULL;
                        /* recovery output: one (x, p) per survivor-hit. Size it
                         * generously -- the whole point is that it is small. */
                        const uint32_t rcap = 8u * 1024u * 1024u;
                        uint32_t hrn = 0; unsigned long long hprobe = 0, hp1 = 0, hrd = 0;
                        float tA = 1e30f, tB = 1e30f, tsum = 1e30f;
                        uint32_t hrnB = 0;

                        CK(mtlMalloc(&d_sum, (size_t)nsumword * 4));
                        CK(mtlMalloc(&d_rx, (size_t)rcap * 4));
                        CK(mtlMalloc(&d_rp, (size_t)rcap * 4));
                        CK(mtlMalloc(&d_rn, 4));
                        CK(mtlMalloc(&d_scratch, (size_t)rcap * 4));
                        CK(mtlMalloc(&d_probe, 8)); CK(mtlMalloc(&d_p1, 8));
                        CK(mtlMalloc(&d_rd, 8));

                        printf("\n  --- recovery A/B (side %d) ---\n", cfg->side);
                        printf("  survivor filter: %u summary bits (1 per 64 positions),"
                               " %.2f MB\n", nsum, nsumword * 4 / 1048576.0);

                        for (int rep = 0; rep < 3; rep++) {
                            CK(mtlMemset(d_sum, 0, (size_t)nsumword * 4));
                            mtlEventRecord(e3);
                            MTL_LAUNCH(k_build_summary, blocks, cfg->threads, 0, 0, d_two, nbitword, d_sum);
                            mtlEventRecord(e4);
                            CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                            float t = time_kernel(e3, e4); if (t < tsum) tsum = t;
                        }
                        /* occupancy of the summary table, measured not assumed */
                        {
                            uint32_t *hs = (uint32_t *)malloc((size_t)nsumword * 4);
                            CK(mtlMemcpy(hs, d_sum, (size_t)nsumword * 4,
                                          mtlMemcpyDeviceToHost));
                            unsigned long long occ = 0;
                            for (uint32_t z = 0; z < nsumword; z++)
                                occ += bench_popcount32(hs[z]);
                            printf("  %-26s %8llu  (%.2f%% occupied)\n",
                                   "summary bits set", occ, 100.0 * occ / (double)nsum);
                            free(hs);
                        }

                        for (int rep = 0; rep < 3; rep++) {
                            CK(mtlMemset(d_rn, 0, 4)); CK(mtlMemset(d_probe, 0, 8));
                            CK(mtlMemset(d_p1, 0, 8));
                            mtlEventRecord(e3);
                            MTL_LAUNCH(k_resieve_rewalk, blocks, cfg->threads, 0, 0, D.plat, D.primes, fb->n, xmax, cfg->logI, d_sum, d_two, d_rx, d_rp, rcap, d_rn, d_probe, d_p1);
                            mtlEventRecord(e4);
                            CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                            float t = time_kernel(e3, e4); if (t < tA) tA = t;
                        }
                        CK(mtlMemcpy(&hrn, d_rn, 4, mtlMemcpyDeviceToHost));
                        CK(mtlMemcpy(&hprobe, d_probe, 8, mtlMemcpyDeviceToHost));
                        CK(mtlMemcpy(&hp1, d_p1, 8, mtlMemcpyDeviceToHost));

                        if (D.out && D.cursor && cap) {
                            for (int rep = 0; rep < 3; rep++) {
                                CK(mtlMemset(d_rn, 0, 4)); CK(mtlMemset(d_rd, 0, 8));
                                mtlEventRecord(e3);
                                MTL_LAUNCH(k_purge, blocks, cfg->threads, 0, 0, (const uint32_t *)D.out, D.cursor, nregion, cap, log_region, d_two, d_scratch, rcap, d_rn, d_rd);
                                mtlEventRecord(e4);
                                CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                float t = time_kernel(e3, e4); if (t < tB) tB = t;
                            }
                            CK(mtlMemcpy(&hrnB, d_rn, 4, mtlMemcpyDeviceToHost));
                            CK(mtlMemcpy(&hrd, d_rd, 8, mtlMemcpyDeviceToHost));
                        }

                        printf("  %-26s %8.3f ms\n", "build survivor filter", tsum);
                        printf("  A: re-walk %llu hits, %llu passed the summary"
                               " (%.2f%%), %u landed on a survivor\n",
                               hprobe, hp1, hprobe ? 100.0 * hp1 / (double)hprobe : 0.0, hrn);
                        printf("  %-26s %8.3f ms\n", "A: re-walk + filter", tA);
                        if (hrnB || hrd) {
                            printf("  B: purged %llu retained records, %u landed"
                                   " on a survivor\n", hrd, hrnB);
                            printf("  %-26s %8.3f ms\n", "B: purge retained buckets", tB);
                        }
                        if (hrn && hrnB)
                            printf("  %-26s %s\n", "A and B agree on count",
                                   hrn == hrnB ? "YES" : "NO  <-- investigate");

                        /* ---- LAYOUT B, built for real: segmented fill +
                         * purge that names the prime ---------------------- */
                        /* Layout B is a SETTLED experiment -- layout A won by
                         * 2.6x -- so it is opt-in. Two reasons beyond the
                         * wasted seconds. It refills D.out and D.cursor with
                         * segmented records in a different format, and the
                         * --verify replay at the end of this function reads
                         * exactly those buffers, so leaving it on by default
                         * made `--verify --other-bits` report thousands of
                         * differing cells and exit nonzero for a reason that
                         * had nothing to do with the fill under test. */
                        if (cfg->ab_resieve && cfg->verify) {
                            printf("\n  --ab-resieve skipped: it overwrites the"
                                   " bucket array that --verify replays\n");
                        } else if (cfg->ab_resieve &&
                                   D.out && D.cursor && cap && cfg->record_bytes == 4) {
                            uint32_t *hstarts = NULL;
                            uint32_t nsl = build_slices_b(fb, &hstarts);
                            uint32_t *d_starts = NULL, *d_bounds = NULL;
                            uint32_t *d_bx = NULL, *d_bp = NULL;
                            uint32_t hbn = 0, hov = 0;
                            float tfillB = 1e30f, tpurgeB = 1e30f;

                            printf("\n  --- layout B, built ---\n");
                            printf("  slices at the 65536 cap: %u  (the 262144 cut"
                                   " gives %u, but a 16-bit within-slice offset\n"
                                   "    cannot address it -- CADO asserts the same"
                                   " bound at fb.hpp:356)\n", nsl, nslice);
                            CK(mtlMalloc(&d_starts, (size_t)(nsl + 1) * 4));
                            CK(mtlMemcpy(d_starts, hstarts, (size_t)(nsl + 1) * 4,
                                          mtlMemcpyHostToDevice));
                            CK(mtlMalloc(&d_bounds, (size_t)nregion * nsl * 4));
                            CK(mtlMalloc(&d_bx, (size_t)rcap * 4));
                            CK(mtlMalloc(&d_bp, (size_t)rcap * 4));
                            printf("  boundary table: %u buckets x %u slices x 4 B"
                                   " = %.1f MB\n", nregion, nsl,
                                   (double)nregion * nsl * 4 / 1048576.0);

                            for (int rep = 0; rep < 3; rep++) {
                                CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
                                CK(mtlMemset(D.overflow, 0, 4));
                                mtlEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++) {
                                    MTL_LAUNCH(k_fill_segmented, blocks, cfg->threads, 0, 0, D.plat, hstarts[sl], hstarts[sl + 1], xmax, cfg->logI, log_region, D.cursor, (uint32_t *)D.out, cap, D.overflow);
                                    /* snapshot cursor[] -> this slice's boundary.
                                     * Bucket-major so the purge's binary search
                                     * reads contiguously. */
                                    MTL_LAUNCH(k_snapshot_bounds, blocks, cfg->threads, 0, 0, D.cursor, d_bounds, nregion, nsl, sl);
                                }
                                mtlEventRecord(e4);
                                CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillB) tfillB = t;
                            }
                            CK(mtlMemcpy(&hov, D.overflow, 4, mtlMemcpyDeviceToHost));
                            /* Merge slices into G super-segments and refill.
                             * Total work is identical; only the number of
                             * passes over the bucket array changes. If the cost
                             * tracks G, the price of segmentation is write
                             * amplification on the bucket array -- adjacent
                             * slots in a cache line being written by different
                             * passes -- and not anything about launches. */
                            for (uint32_t G : {1u, 2u, 4u, 8u, 16u, 32u, 63u}) {
                                float tg = 1e30f;
                                for (int rep = 0; rep < 3; rep++) {
                                    CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
                                    mtlEventRecord(e3);
                                    for (uint32_t g = 0; g < G; g++) {
                                        uint32_t s0 = (uint32_t)((uint64_t)g * nsl / G);
                                        uint32_t s1 = (uint32_t)((uint64_t)(g + 1) * nsl / G);
                                        if (s1 <= s0) continue;
                                        MTL_LAUNCH(k_fill_segmented, blocks, cfg->threads, 0, 0, D.plat, hstarts[s0], hstarts[s1], xmax, cfg->logI, log_region, D.cursor, (uint32_t *)D.out, cap, D.overflow);
                                    }
                                    mtlEventRecord(e4);
                                    CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                    float t = time_kernel(e3, e4); if (t < tg) tg = t;
                                }
                                printf("  %-26s %8.3f ms   (%u passes over the bucket array)\n",
                                       "B: fill in G passes", tg, G);
                            }
                            /* Split the cost: same 126 fill launches, no
                             * boundary snapshots. The difference is what the
                             * segmentation bookkeeping costs; what remains
                             * above the monolithic fill is the launches
                             * themselves plus per-launch occupancy loss. */
                            float tfillNS = 1e30f;
                            for (int rep = 0; rep < 3; rep++) {
                                CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
                                mtlEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++)
                                    MTL_LAUNCH(k_fill_segmented, blocks, cfg->threads, 0, 0, D.plat, hstarts[sl], hstarts[sl + 1], xmax, cfg->logI, log_region, D.cursor, (uint32_t *)D.out, cap, D.overflow);
                                mtlEventRecord(e4);
                                CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillNS) tfillNS = t;
                            }
                            /* and: one launch per slice but sized to the slice,
                             * to separate launch count from occupancy loss */
                            float tfillSized = 1e30f;
                            for (int rep = 0; rep < 3; rep++) {
                                CK(mtlMemset(D.cursor, 0, (size_t)nregion * 4));
                                mtlEventRecord(e3);
                                for (uint32_t sl = 0; sl < nsl; sl++) {
                                    uint32_t nent = hstarts[sl + 1] - hstarts[sl];
                                    int bl = (int)((nent + cfg->threads - 1) / cfg->threads);
                                    if (bl < 1) bl = 1;
                                    if (bl > blocks) bl = blocks;
                                    MTL_LAUNCH(k_fill_segmented, bl, cfg->threads, 0, 0, D.plat, hstarts[sl], hstarts[sl + 1], xmax, cfg->logI, log_region, D.cursor, (uint32_t *)D.out, cap, D.overflow);
                                }
                                mtlEventRecord(e4);
                                CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tfillSized) tfillSized = t;
                            }
                            printf("  %-26s %8.3f ms   (no boundary snapshots)\n",
                                   "B: segmented fill", tfillNS);
                            printf("  %-26s %8.3f ms   (grid sized per slice)\n",
                                   "B: segmented fill", tfillSized);

                            for (int rep = 0; rep < 3; rep++) {
                                CK(mtlMemset(d_rn, 0, 4));
                                mtlEventRecord(e3);
                                MTL_LAUNCH(k_purge_prime, blocks, cfg->threads, 0, 0, (const uint32_t *)D.out, D.cursor, d_bounds, d_starts, D.primes, nregion, cap, nsl, log_region, d_two, d_bx, d_bp, rcap, d_rn);
                                mtlEventRecord(e4);
                                CK(mtlEventSynchronize(e4)); CK(mtlGetLastError());
                                float t = time_kernel(e3, e4);
                                if (t < tpurgeB) tpurgeB = t;
                            }
                            CK(mtlMemcpy(&hbn, d_rn, 4, mtlMemcpyDeviceToHost));

                            printf("  %-26s %8.3f ms   (%u launches%s)\n",
                                   "B: segmented fill", tfillB, nsl,
                                   hov ? ", OVERFLOWED" : "");
                            printf("  %-26s %8.3f ms\n", "B: purge + name the prime",
                                   tpurgeB);
                            printf("  %-26s %8u  vs A's %u   %s\n",
                                   "B: recovered (x,p)", hbn, hrn,
                                   hbn == hrn ? "MATCH" : "MISMATCH <-- investigate");

                            /* set-equality gate: A and B must recover the same
                             * (x, p) multiset, not merely the same count. */
                            if (hbn == hrn && hrn && hrn <= rcap) {
                                uint32_t *ax = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *ap = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *bx = (uint32_t *)malloc((size_t)hrn * 4);
                                uint32_t *bp = (uint32_t *)malloc((size_t)hrn * 4);
                                CK(mtlMemcpy(ax, d_rx, (size_t)hrn * 4, mtlMemcpyDeviceToHost));
                                CK(mtlMemcpy(ap, d_rp, (size_t)hrn * 4, mtlMemcpyDeviceToHost));
                                CK(mtlMemcpy(bx, d_bx, (size_t)hrn * 4, mtlMemcpyDeviceToHost));
                                CK(mtlMemcpy(bp, d_bp, (size_t)hrn * 4, mtlMemcpyDeviceToHost));
                                /* compare as sorted (x,p) pairs */
                                uint64_t *A64 = (uint64_t *)malloc((size_t)hrn * 8);
                                uint64_t *B64 = (uint64_t *)malloc((size_t)hrn * 8);
                                for (uint32_t z = 0; z < hrn; z++) {
                                    A64[z] = ((uint64_t)ax[z] << 32) | ap[z];
                                    B64[z] = ((uint64_t)bx[z] << 32) | bp[z];
                                }
                                std::sort(A64, A64 + hrn); std::sort(B64, B64 + hrn);
                                uint32_t diff = 0;
                                for (uint32_t z = 0; z < hrn; z++) if (A64[z] != B64[z]) diff++;
                                printf("  %-26s %s (%u of %u pairs differ)\n",
                                       "A vs B (x,p) set equality",
                                       diff ? "FAIL" : "IDENTICAL", diff, hrn);
                                free(ax); free(ap); free(bx); free(bp); free(A64); free(B64);
                            }
                            free(hstarts);
                            mtlFree(d_starts); mtlFree(d_bounds);
                            mtlFree(d_bx); mtlFree(d_bp);
                        }

                        mtlFree(d_sum); mtlFree(d_rx); mtlFree(d_rp); mtlFree(d_rn);
                        mtlFree(d_scratch);
                        mtlFree(d_probe); mtlFree(d_p1); mtlFree(d_rd);
                    }

                    /* ---- exact norms + trial division --------------------- */
                    if (cfg->td && d_two &&
                        run_td_stage(fb, fbs, L, POLY, cfg, D.plat, D.primes,
                                     d_two, nbitword, xmax, blocks, cfg->threads))
                        td_failed = 1;

                    if (hother) mtlFreeHost(hother);
                    mtlFree(dother); mtlFree(d_x); mtlFree(d_a);
                    mtlFree(d_b); mtlFree(d_n); mtlFree(d_pre); mtlFree(d_two);
                    mtlFree(d_qb);
                }
            }
            if (cfg->dump) {
                uint8_t *h = (uint8_t *)malloc((size_t)xmax);
                FILE *fo = fopen(cfg->dump, "wb");
                CK(mtlMemcpy(h, D.dumpbuf, (size_t)xmax, mtlMemcpyDeviceToHost));
                if (!fo) { perror(cfg->dump); }
                else { fwrite(h, 1, (size_t)xmax, fo); fclose(fo); }
                free(h);
            }
            uint32_t hs = 0;
            if (probe_x != 0xFFFFFFFFu) {
                uint32_t pr[2] = {0, 0};
                CK(mtlMemcpy(pr, D.probe, 8, mtlMemcpyDeviceToHost));
                printf("\n  [gate 5] probe (i=%d, j=%u)  x=%u  region %u offset %u\n"
                       "           init norm S   = %u\n"
                       "           final cell    = %u\n"
                       "           SIEVED LOG SUM = %d   <- produced by transform +"
                       " walk + fill + small sieve + apply\n"
                       "           las byte S-sum = %d\n",
                       cfg->probe_i, cfg->probe_j, probe_x,
                       probe_x >> log_region, probe_x & ((1u << log_region) - 1),
                       pr[0], pr[1],
                       (int)pr[1] - ((int)CINIT - (int)pr[0]),
                       (int)CINIT - (int)pr[1]);
            }
            CK(mtlMemcpy(&hs, D.nsurv, 4, mtlMemcpyDeviceToHost));
            printf("  survivors: %u of %u positions (1 in %.3e)\n", hs, xmax,
                   hs ? (double)xmax / hs : 0.0);

            if (cfg->verify && CB == 16) {
                uint32_t *hrec = (uint32_t *)malloc((size_t)cap * 4);
                uint32_t *hcnt = (uint32_t *)malloc((size_t)nregion * 4);
                uint16_t *hgpu = (uint16_t *)malloc((size_t)ncell * 2);
                uint16_t *href = (uint16_t *)malloc((size_t)ncell * 2);
                CK(mtlMemcpy(hcnt, D.cursor, (size_t)nregion * 4, mtlMemcpyDeviceToHost));
                CK(mtlMemcpy(hrec, D.out + (size_t)dbgreg * cap * 4, (size_t)cap * 4,
                              mtlMemcpyDeviceToHost));
                CK(mtlMemcpy(hgpu, D.dbg, (size_t)ncell * 2, mtlMemcpyDeviceToHost));
                uint32_t nr = hcnt[dbgreg] > cap ? cap : hcnt[dbgreg];
                uint32_t rs = verify_apply_region(hrec, nr, hlogp, &N, cfg->logI,
                        log_region, dbgreg, cfg->norm_mode, CINIT, tconst,
                        hsp, hsrt, hsg, hslp, nsmall, href);
                uint32_t bad = 0, first = 0xFFFFFFFFu;
                for (uint32_t i = 0; i < ncell; i++)
                    if (hgpu[i] != href[i]) { if (!bad) first = i; bad++; }
                printf("  [verify] region %u: %u records replayed on CPU, %u cells differ",
                       dbgreg, nr, bad);
                if (bad) printf("  (first at cell %u: gpu %u ref %u)",
                                first, hgpu[first], href[first]);
                printf("\n  [verify] region %u survivors: cpu %u\n", dbgreg, rs);
                free(hrec); free(hcnt); free(hgpu); free(href);
                /* A differing cell is a failed run, not a log line. This used
                 * to print and return 0, so `--verify && echo ok` reported
                 * success on a sieve that disagreed with its own reference. */
                if (bad) {
                    fprintf(stderr, "[verify] FAILED: %u cells differ in region %u\n",
                            bad, dbgreg);
                    return -1;
                }
            }
        }
    }
after_apply:

    printf("\n  %-26s %8.3f ms\n", "transform + plattice (T)", t_trans);
    if (cfg->fill_mode == FILL_ATOMIC)
        printf("  %-26s %8.3f ms\n", "fill: atomic single-level", t_fill);
    else {
        printf("  %-26s %8.3f ms\n", "fill L1: -> super-buckets", t_l1);
        printf("  %-26s %8.3f ms\n", "fill L2: -> regions", t_l2);
        printf("  %-26s %8.3f ms\n", "fill total", t_fill);
    }
    if (t_apply > 0)
        printf("  %-26s %8.3f ms\n", "apply (init+add+scan)", t_apply);
    /* The old "vs ~225 ms replaceable / ~71 ms hybrid-retained" suffix is gone.
     * Those were fixed constants from the GGNFS breakdown at N_eff = 10.24 --
     * a different job at a different logI/J -- printed on every run whatever
     * was actually being sieved, which read as a per-run comparison and is not
     * one. The surviving numbers live in RESULTS.md findings 43 and 45, where
     * the config they belong to is stated. */
    printf("  %-26s %8.3f ms\n",
           "SIEVE CHAIN ms/special-q", t_trans + t_fill + t_apply);
    /* Per-q HOST work. Not part of the sieve chain above -- it runs on the CPU,
     * once per special-q per side, and cudaEvent timing cannot see it. Goal 1
     * is about host demand, so it is billed here rather than left implicit. */
    printf("  %-26s %8.3f ms  (transform %.3f + sort %.3f + H2D %.3f)\n",
           "host per-q work", h_ms_transform + h_ms_sort + h_ms_xfer,
           h_ms_transform, h_ms_sort, h_ms_xfer);
    if (landed) {
        printf("  %-26s %8.2f\n", "ns per record (fill)", t_fill * 1e6 / landed);
        if (t_apply > 0)
            printf("  %-26s %8.2f\n", "ns per record (apply)", t_apply * 1e6 / landed);
    }

    free(hslice); free(hlogp);
    if (hsp)  mtlFreeHost(hsp);
    if (hsrt) mtlFreeHost(hsrt);
    if (hsmag) mtlFreeHost(hsmag);
    if (hsg)  mtlFreeHost(hsg);
    if (hslp) mtlFreeHost(hslp);
    mtlFree(D.primes); mtlFree(D.roots); mtlFree(D.plat); mtlFree(D.cursor);
    mtlFree(D.out); mtlFree(D.overflow); mtlFree(D.nproj); mtlFree(D.nlost);
    mtlFree(D.l1); mtlFree(D.l1cnt); mtlFree(D.slice); mtlFree(D.slice_logp);
    mtlFree(D.nsurv); mtlFree(D.dbg); mtlFree(D.probe);
    mtlFree(D.smag);
    mtlFree(D.sp); mtlFree(D.srt); mtlFree(D.sg); mtlFree(D.slp); mtlFree(D.dumpbuf); mtlFree(D.survbits);
    return td_failed ? -1 : 0;
}

#include "cofac_host.inc"
#include "pipeline_host.inc"
