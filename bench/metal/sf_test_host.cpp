/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Phase 2 gate, part 1: softfp64.h against the host's HARDWARE fp64.
 *
 * Every operation is compared as a 64-bit BIT PATTERN, not as a value, so a
 * wrong sign of zero or a denormal off by one ulp fails here rather than in a
 * relation file three phases later. Inputs deliberately include subnormals,
 * infinities, NaNs, exact powers of two and the exponent-crossing cases that
 * naive soft-float gets wrong, alongside the value ranges the siever actually
 * produces.
 *
 * Build with -ffp-contract=off: the reference expressions must not be
 * contracted into fma by the host compiler, or we would be testing against
 * something the CUDA build never computes.
 */
#include "softfp64.h"
#include "portable_log2.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static double b2d(sf_u64 b) { double d; memcpy(&d, &b, 8); return d; }
static sf_u64 d2b(double d) { sf_u64 b; memcpy(&b, &d, 8); return b; }
static bool isnan_b(sf_u64 b) { return ((b >> 52) & 0x7ff) == 0x7ff && (b & SF_MANT52); }
static bool isnan_f(sf_u64 b) { return ((b >> 23) & 0xff) == 0xff && (b & 0x7fffffu); }

struct Fail { const char *op; sf_u64 a, b, c, got, want; };
static std::vector<Fail> fails;
static long checked = 0;

static void chk(const char *op, sf_u64 a, sf_u64 b, sf_u64 c, sf_u64 got, sf_u64 want)
{
    checked++;
    /* Any NaN result is acceptable for any NaN result: IEEE does not pin the
     * payload, and neither CUDA nor Metal guarantees one. */
    if (isnan_b(got) && isnan_b(want)) return;
    /* Same for 32-bit results: a double->float conversion of a NaN may or may
     * not carry the payload through, and no vendor pins that. */
    if (isnan_f(got) && isnan_f(want)) return;
    if (got == want) return;
    if (fails.size() < 12) fails.push_back({op, a, b, c, got, want});
    else fails.push_back({op, 0, 0, 0, 0, 0});
}

/* xorshift64*, so the corpus is identical on any platform. */
static sf_u64 rs = 0x9E3779B97F4A7C15ull;
static sf_u64 rnd(void)
{
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return rs * 0x2545F4914F6CDD1Dull;
}

static std::vector<sf_u64> corpus(void)
{
    std::vector<sf_u64> v;
    /* hand-picked edges */
    const double edges[] = {
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 3.0, -3.0,
        4294967296.0, 4503599627370496.0 /*2^52*/, 9007199254740992.0 /*2^53*/,
        1e300, 1e-300, 1e308, 1e-308,
        2.2250738585072014e-308 /*min normal*/,
        4.9406564584124654e-324 /*min subnormal*/,
        1.7976931348623157e308 /*max normal*/,
        INFINITY, -INFINITY, NAN,
        1.0000000000000002, 0.9999999999999999,
    };
    for (double d : edges) v.push_back(d2b(d));
    v.push_back(0x000FFFFFFFFFFFFFull);   /* max subnormal */
    v.push_back(0x0010000000000000ull);   /* min normal    */

    for (int i = 0; i < 4000; i++) v.push_back(rnd());               /* any pattern */
    for (int i = 0; i < 4000; i++) {                                  /* normal range */
        sf_u64 e = 1 + (rnd() % 2046);
        v.push_back(((rnd() & 1) << 63) | (e << 52) | (rnd() & SF_MANT52));
    }
    for (int i = 0; i < 2000; i++)                                    /* subnormals */
        v.push_back(((rnd() & 1) << 63) | (rnd() & SF_MANT52));
    for (int i = 0; i < 2000; i++)                                    /* integers */
        v.push_back(d2b((double)(sf_i64)rnd()));
    for (int i = 0; i < 1000; i++) {                                  /* near-equal pairs */
        sf_u64 e = 900 + (rnd() % 250);
        v.push_back((e << 52) | (rnd() & 0xff));
    }
    return v;
}

int main(void)
{
    std::vector<sf_u64> v = corpus();
    printf("corpus: %zu values\n", v.size());

    /* ---- unary ---- */
    for (sf_u64 a : v) {
        chk("to_f32", a, 0, 0,
            (sf_u64)sf_f2b(sf_to_f32(a)), (sf_u64)sf_f2b((float)b2d(a)));
        chk("abs", a, 0, 0, sf_abs(a), d2b(fabs(b2d(a))));
    }
    for (int i = 0; i < 200000; i++) {
        sf_i64 x = (sf_i64)rnd();
        if (i % 7 == 0) x >>= (rnd() % 63);
        chk("from_i64", (sf_u64)x, 0, 0, sf_from_i64(x), d2b((double)x));
        sf_u32 u = (sf_u32)rnd();
        chk("from_u32", u, 0, 0, sf_from_u32(u), d2b((double)u));
        sf_u32 fb = (sf_u32)rnd();
        float f = sf_b2f(fb);
        chk("from_f32", fb, 0, 0, sf_from_f32(f), d2b((double)f));
    }

    /* ---- binary, all ordered pairs over a sampled subset ---- */
    size_t n = v.size();
    for (size_t i = 0; i < n; i++) {
        for (int t = 0; t < 40; t++) {
            size_t j = (size_t)(rnd() % n);
            sf_u64 a = v[i], b = v[j];
            double da = b2d(a), db = b2d(b);
            chk("add", a, b, 0, sf_add(a, b), d2b(da + db));
            chk("sub", a, b, 0, sf_sub(a, b), d2b(da - db));
            chk("mul", a, b, 0, sf_mul(a, b), d2b(da * db));
            chk("div", a, b, 0, sf_div(a, b), d2b(da / db));
            int lt_got = sf_lt(a, b), lt_want = (da < db) ? 1 : 0;
            if (lt_got != lt_want) chk("lt", a, b, 0, (sf_u64)lt_got, (sf_u64)lt_want);
            else checked++;
            size_t k = (size_t)(rnd() % n);
            sf_u64 c = v[k];
            chk("fma", a, b, c, sf_fma(a, b, c), d2b(fma(da, db, b2d(c))));
        }
    }

    /* ---- portable_log2: host self-consistency and accuracy vs libm ---- */
    {
        long worst_ulp = 0; long nz = 0;
        for (int i = 0; i < 1000000; i++) {
            sf_u32 m = (sf_u32)(rnd() & 0x7fffff);
            sf_u32 e = 1 + (sf_u32)(rnd() % 250);
            float x = sf_b2f((e << 23) | m);
            float got = pl_log2f(x), want = log2f(x);
            long d = (long)sf_f2b(got) - (long)sf_f2b(want);
            if (d < 0) d = -d;
            if (d) nz++;
            if (d > worst_ulp) worst_ulp = d;
        }
        printf("pl_log2f vs libm log2f      : %ld/1000000 differ, max %ld ULP\n",
               nz, worst_ulp);
    }

    printf("checked %ld operations\n", checked);
    if (fails.empty()) { printf("SOFTFP64 HOST GATE: PASS\n"); return 0; }
    printf("SOFTFP64 HOST GATE: FAIL (%zu)\n", fails.size());
    for (size_t i = 0; i < fails.size() && i < 12; i++) {
        Fail &f = fails[i];
        printf("  %-8s a=%016llx b=%016llx c=%016llx got=%016llx want=%016llx\n",
               f.op, (unsigned long long)f.a, (unsigned long long)f.b,
               (unsigned long long)f.c, (unsigned long long)f.got,
               (unsigned long long)f.want);
    }
    return 1;
}
