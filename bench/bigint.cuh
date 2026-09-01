/* Fixed-width big integers for exact norms and trial division.
 *
 * WHY 384 BITS, AND WHY IT IS A KNOB. The width is BN_LIMBS 32-bit limbs, set
 * by the Makefile variable of the same name; 12 limbs = 384 bits is the
 * default. Nothing here assumes a particular value beyond an even limb count
 * (cf_bn_divmod_u64 steps the array in 64-bit pairs).
 *
 * 8 limbs was the original width, measured on a quintic: over the corners of
 * the sieve area plus 20K interior points at q=120000053, rho=112625526,
 * skew 1.154e8, max |F(a,b)| was 224 bits and the largest individual
 * homogeneous term c_k a^k b^(5-k) also 224, so a 6-term sum stayed under 227
 * and 256 bits left ~28 spare.
 *
 * That headroom was a property of THAT polynomial. A large octic blows it:
 * for the Cunningham 2,1139+ SNFS form (deg 8, skew 1.256) the exact algebraic
 * norm runs 232-292 bits depending on the q-lattice, and the tail is driven by
 * the SHAPE of the reduced basis rather than by the sieve area -- shrinking
 * logI/J does not escape it, and neither does a tighter estimate, since
 * norm_exact_bound_bits sits only 2-3 bits above the true maximum term.
 * Measured: a 2000-q band at 8 limbs died after 116 q needing 260.75 bits,
 * having written no relations, and could not resume past that q because the
 * checkpoint names it. The same band at 12 limbs completed, and all 4015
 * relations rebuilt both norms exactly.
 *
 * WHAT THE EXTRA LIMBS COST, measured A/B on AS276 (C208, deg 5, logI 15,
 * three runs each): the norm-and-trial-division kernel goes 1.138 -> 1.593 ms
 * (+40%) and the warp recorder 0.234 -> 0.284 ms (+21%), which is +0.45 ms on
 * a 90 ms special-q -- under the +-0.8 ms run-to-run spread of the wall clock.
 * Registers go 58 -> 78 on k_td and 68 -> 80 on k_td_record_warp with NO spill,
 * and k_apply, k_classify, k_cof_enqueue and k_rel_pack are untouched. The
 * divide-down does not scale with BN_LIMBS at all, because td_divide_out loops
 * on bn_top: the added limbs are leading zeros and are skipped.
 *
 * MEMORY, which the timing A/B does not show. sizeof(bn_t) goes 32 -> 48 bytes
 * and it is allocated per survivor (d_cof[2]) and per candidate (d_ccof[2] plus
 * the pinned h_ccof[2]), so the cost scales with the survivor count, not with
 * the band. Measured on AS276 at 39,042 survivors/q the growth is ~2 MB and
 * both widths report the same `trial division context 0.09 GB` and `cofactor
 * queue 0.15 GB`. A job with far more survivors per q would see proportionally
 * more, and the pipeline's steady-state figure is what operators size cards
 * from, so re-measure rather than assume it stays invisible.
 *
 * A WIDER BUILD MUST BE BYTE-IDENTICAL on any job the narrower one could run.
 * Every value that fit is represented identically, so this is a real gate, and
 * it is the one that qualified 12 limbs: same relation file md5, same survivor,
 * candidate and split/dead/stuck counts as the 8-limb binary on AS276.
 *
 * `bn_mul_u64` reports overflow rather than wrapping, so a job too big for
 * whatever width was built cannot corrupt a norm silently.
 *
 * WHY MAGNITUDE + SIGN rather than two's complement. Trial division is the
 * only consumer, and it divides magnitudes; a sign-magnitude representation
 * keeps division and comparison trivial at the cost of a branch in add. The
 * norm's sign is discarded the moment the norm is factored, so it is carried
 * only far enough to get the accumulation right.
 *
 * The whole header is __host__ __device__ so the host reference and the kernel
 * run the SAME arithmetic. A host reference that reimplements the arithmetic
 * only proves the two implementations agree with each other.
 */
#ifndef CUDA_SIEVE_BIGINT_CUH
#define CUDA_SIEVE_BIGINT_CUH

#include <stdint.h>
/* BN_MULHI64's MSVC branch calls bench_mulhi_u64; include its declaration
 * here rather than relying on an includer having pulled platform.h in. */
#include "platform.h"

#ifndef BN_LIMBS
#define BN_LIMBS 12                     /* 12 x 32 = 384 bits */
#endif
#if (BN_LIMBS) < 4 || (BN_LIMBS) > 16 || ((BN_LIMBS) & 1)
#error "BN_LIMBS must be an even limb count in 4..16 (cf_bn_divmod_u64 steps in 64-bit pairs)"
#endif

/* Bytes bn_to_dec needs, including the NUL. 32 bits is 9.633 decimal digits,
 * so ten per limb is a bound with room to spare. NOT a literal 80: that was
 * sized for 256 bits (78 digits) and silently became a stack overrun the
 * moment the width moved -- 384 bits is 116 digits. Only cofactors, which are
 * bounded by mfb, reach it today, which is exactly why it would not have
 * failed in testing. */
#define BN_DEC_MAX (BN_LIMBS * 10 + 2)

/* Narrowest build width that would hold a `bits`-bit magnitude, for the
 * "rebuild with make BN_LIMBS=N" advice on the refusal paths. Returns 0 when
 * no legal width suffices, so a caller does not print an instruction the
 * Makefile would reject.
 *
 * floor(bits/32) + 1, THEN rounded up to even -- not ceil(bits/32). The
 * distinction only shows at an exact multiple of 32, and there ceil is wrong:
 * norm_fits_exact demands a strict `bits < 32*L`, so a 256.0-bit norm does NOT
 * fit 8 limbs and advising 8 would send the operator through a nine-minute
 * rebuild back to the same refusal. Even because cf_bn_divmod_u64 steps the
 * limb array in 64-bit pairs. */
static inline int bn_limbs_for_bits(double bits)
{
    int L;
    if (!(bits >= 0.0)) return 0;               /* NaN included */
    /* MATCH norm_fits_exact's PREDICATE, which is `bound + 1e-3 < 32*L`
     * (poly.c) -- the millibit keeps a downward rounding of the float log2M
     * from certifying a width that does not hold. Sizing against the bare
     * bound instead returned the width the caller already has whenever bits
     * landed in [32L - 1e-3, 32L): the refusal message then said "rebuild with
     * BN_LIMBS=12" to an operator already running 12 limbs. */
    bits += 1e-3;
    if (bits >= 32.0 * 16) return 0;
    L = ((int)(bits / 32.0) + 2) & ~1;
    return (L < 4) ? 4 : L;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BN_FN __host__ __device__ static inline
#else
#define BN_FN static inline
#endif

typedef struct { uint32_t v[BN_LIMBS]; } bn_t;

BN_FN void bn_zero(bn_t *x)
{
    for (int i = 0; i < BN_LIMBS; i++) x->v[i] = 0;
}

BN_FN void bn_set_u64(bn_t *x, uint64_t w)
{
    bn_zero(x);
    x->v[0] = (uint32_t)w;
    x->v[1] = (uint32_t)(w >> 32);
}

BN_FN int bn_is_zero(const bn_t *x)
{
    uint32_t o = 0;
    for (int i = 0; i < BN_LIMBS; i++) o |= x->v[i];
    return o == 0;
}

BN_FN int bn_is_one(const bn_t *x)
{
    uint32_t o = 0;
    for (int i = 1; i < BN_LIMBS; i++) o |= x->v[i];
    return o == 0 && x->v[0] == 1u;
}

/* -1, 0, +1 */
BN_FN int bn_cmp(const bn_t *a, const bn_t *b)
{
    for (int i = BN_LIMBS - 1; i >= 0; i--) {
        if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1;
    }
    return 0;
}

/* a += b. Returns the carry out of the top limb (nonzero == overflow). */
BN_FN uint32_t bn_add(bn_t *a, const bn_t *b)
{
    uint64_t carry = 0;
    for (int i = 0; i < BN_LIMBS; i++) {
        uint64_t s = (uint64_t)a->v[i] + b->v[i] + carry;
        a->v[i] = (uint32_t)s;
        carry = s >> 32;
    }
    return (uint32_t)carry;
}

/* a -= b, and the caller must have established a >= b. Returns the borrow,
 * which is nonzero exactly when that precondition was violated. */
BN_FN uint32_t bn_sub(bn_t *a, const bn_t *b)
{
    uint64_t borrow = 0;
    for (int i = 0; i < BN_LIMBS; i++) {
        uint64_t d = (uint64_t)a->v[i] - b->v[i] - borrow;
        a->v[i] = (uint32_t)d;
        borrow = (d >> 32) & 1u;
    }
    return (uint32_t)borrow;
}

/* x *= m. Returns nonzero if the product did not fit in BN_LIMBS limbs. */
BN_FN uint32_t bn_mul_u32(bn_t *x, uint32_t m)
{
    uint64_t carry = 0;
    for (int i = 0; i < BN_LIMBS; i++) {
        uint64_t p = (uint64_t)x->v[i] * m + carry;
        x->v[i] = (uint32_t)p;
        carry = p >> 32;
    }
    return (uint32_t)carry;
}

/* x *= m for a 64-bit multiplier: two 32-bit schoolbook passes accumulating
 * into one result, the second offset by a limb. Written to keep only two bn_t
 * live at once -- the obvious three-temporary version costs 8 more registers
 * per thread and this runs inside the trial-division kernel, which is already
 * carrying a norm and a scratch value. Returns nonzero on overflow. */
BN_FN uint32_t bn_mul_u64(bn_t *x, uint64_t m)
{
    const uint32_t mlo = (uint32_t)m, mhi = (uint32_t)(m >> 32);
    bn_t r;
    uint64_t carry = 0;
    uint32_t ovf = 0;

    for (int i = 0; i < BN_LIMBS; i++) {
        uint64_t p = (uint64_t)x->v[i] * mlo + carry;
        r.v[i] = (uint32_t)p;
        carry = p >> 32;
    }
    ovf |= (uint32_t)carry;

    if (mhi) {
        /* the top limb of x would land one limb past the end */
        ovf |= x->v[BN_LIMBS - 1];
        carry = 0;
        for (int i = 0; i < BN_LIMBS - 1; i++) {
            uint64_t p = (uint64_t)x->v[i] * mhi + r.v[i + 1] + carry;
            r.v[i + 1] = (uint32_t)p;
            carry = p >> 32;
        }
        ovf |= (uint32_t)carry;
    }
    *x = r;
    return ovf;
}

/* Index of the highest nonzero limb, or -1 for zero. Trial division shrinks a
 * 224-bit norm to ~112 bits, and skipping the limbs that have gone to zero
 * halves the work over the course of a factorisation. */
BN_FN int bn_top(const bn_t *x)
{
    for (int i = BN_LIMBS - 1; i >= 0; i--) if (x->v[i]) return i;
    return -1;
}

/* 64x64 -> high 64. The GPU has an instruction; the host has __int128.
 *
 * The device-compile check MUST come before the _MSC_VER check below: hipcc
 * on Windows defines _MSC_VER in its device compile pass too (confirmed via
 * probe_macros.hip), so without __HIP_DEVICE_COMPILE__ checked first, HIP
 * device code would fall into the _MSC_VER branch and call
 * bench_mulhi_u64 -- a host-only function -- from device code. HIP-clang
 * never defines __CUDA_ARCH__ on the AMD backend; __HIP_DEVICE_COMPILE__ is
 * its analogue. */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define BN_MULHI64(a, b) __umul64hi((a), (b))
#elif defined(_MSC_VER)
#define BN_MULHI64(a, b) bench_mulhi_u64((a), (b))
#else
#define BN_MULHI64(a, b) \
    ((uint64_t)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))
#endif

/* Reciprocal for bn_divmod_u32_pre. d must be at least 2. */
BN_FN uint64_t bn_recip_u32(uint32_t d)
{
    return (uint64_t)0xFFFFFFFFFFFFFFFFull / d;
}

/* x /= d with a PRECOMPUTED reciprocal, returning the remainder.
 *
 * This exists because the obvious `cur / d` on a 64-bit dividend is not an
 * instruction on any GPU -- nvcc expands it into a ~100-instruction software
 * routine, and trial division runs it eight times per pass, several passes per
 * prime factor, several million times per special-q. Measured on this job it
 * was 30 ms of a 43 ms kernel, against 13 ms for the 3e9 congruence tests that
 * find the factors in the first place.
 *
 * Barrett instead: with M = floor((2^64 - 1)/d) and any cur < d * 2^32,
 * floor(cur*M / 2^64) is either floor(cur/d) or one less, so one conditional
 * correction is exact. Two multiplies replace the division. */
BN_FN uint32_t bn_divmod_u32_pre(bn_t *x, uint32_t d, uint64_t M, int top)
{
    uint64_t rem = 0;
    for (int i = top; i >= 0; i--) {
        uint64_t cur = (rem << 32) | x->v[i];
        uint64_t q = BN_MULHI64(cur, M);
        uint64_t r = cur - q * d;
        if (r >= d) { q++; r -= d; }
        x->v[i] = (uint32_t)q;
        rem = r;
    }
    return (uint32_t)rem;
}

/* x /= d, returning the remainder. d must be nonzero. */
BN_FN uint32_t bn_divmod_u32(bn_t *x, uint32_t d)
{
    uint64_t rem = 0;
    for (int i = BN_LIMBS - 1; i >= 0; i--) {
        uint64_t cur = (rem << 32) | x->v[i];
        x->v[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    return (uint32_t)rem;
}

/* x mod d, leaving x alone. */
BN_FN uint32_t bn_mod_u32(const bn_t *x, uint32_t d)
{
    uint64_t rem = 0;
    for (int i = BN_LIMBS - 1; i >= 0; i--)
        rem = (uint64_t)(((rem << 32) | x->v[i]) % d);
    return (uint32_t)rem;
}

/* Number of significant bits; 0 for zero. */
BN_FN int bn_bits(const bn_t *x)
{
    for (int i = BN_LIMBS - 1; i >= 0; i--) {
        if (x->v[i]) {
            int b = 0;
            uint32_t w = x->v[i];
            while (w) { w >>= 1; b++; }
            return i * 32 + b;
        }
    }
    return 0;
}

/* Does x fit in 64 bits, and if so what is it? Cofactor classification asks
 * this constantly -- most residues are small and the 64-bit path is the one
 * worth taking. */
BN_FN int bn_fits_u64(const bn_t *x, uint64_t *out)
{
    for (int i = 2; i < BN_LIMBS; i++) if (x->v[i]) return 0;
    *out = ((uint64_t)x->v[1] << 32) | x->v[0];
    return 1;
}

/* ---- signed accumulator ------------------------------------------------ */

/* Sign is +1 or -1 and is meaningless when the magnitude is zero. */
typedef struct { bn_t m; int sign; } bns_t;

BN_FN void bns_zero(bns_t *x) { bn_zero(&x->m); x->sign = 1; }

/* acc += (sign) * t. Returns nonzero on magnitude overflow. */
BN_FN uint32_t bns_addmag(bns_t *acc, const bn_t *t, int sign)
{
    if (bn_is_zero(t)) return 0;
    if (bn_is_zero(&acc->m)) { acc->m = *t; acc->sign = sign; return 0; }
    if (acc->sign == sign) return bn_add(&acc->m, t);
    /* opposite signs: subtract the smaller from the larger */
    int c = bn_cmp(&acc->m, t);
    if (c == 0) { bn_zero(&acc->m); acc->sign = 1; return 0; }
    if (c > 0) { bn_sub(&acc->m, t); }
    else { bn_t big = *t; bn_sub(&big, &acc->m); acc->m = big; acc->sign = sign; }
    return 0;
}

/* ---- host-only decimal conversion -------------------------------------- */
/* Cofactors are compared against CADO's printed values, which run to 96 bits,
 * so the gate needs decimal in and decimal out. Neither is on any hot path.
 *
 * Deliberately NOT guarded by __CUDA_ARCH__: nvcc parses the whole file in the
 * device pass too, and a host function that calls these still has to resolve
 * their names there. Leaving them as plain host functions is what makes that
 * work -- they simply go unused in the device pass. */

static inline int bn_from_dec(bn_t *x, const char *s, int *sign_out)
{
    int sign = 1;
    bn_zero(x);
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') { sign = -1; s++; }
    else if (*s == '+') s++;
    if (*s < '0' || *s > '9') return -1;
    for (; *s >= '0' && *s <= '9'; s++) {
        if (bn_mul_u32(x, 10u)) return -1;
        bn_t d; bn_set_u64(&d, (uint64_t)(*s - '0'));
        if (bn_add(x, &d)) return -1;
    }
    if (sign_out) *sign_out = sign;
    return 0;
}

/* Writes into buf, which must hold BN_DEC_MAX bytes, and returns it. */
static inline char *bn_to_dec(const bn_t *x, char *buf)
{
    char tmp[BN_DEC_MAX];
    int n = 0;
    bn_t w = *x;
    if (bn_is_zero(&w)) { buf[0] = '0'; buf[1] = 0; return buf; }
    while (!bn_is_zero(&w)) {
        uint32_t r = bn_divmod_u32(&w, 1000000000u);
        if (bn_is_zero(&w)) {
            while (r) { tmp[n++] = (char)('0' + r % 10u); r /= 10u; }
        } else {
            for (int k = 0; k < 9; k++) { tmp[n++] = (char)('0' + r % 10u); r /= 10u; }
        }
    }
    for (int k = 0; k < n; k++) buf[k] = tmp[n - 1 - k];
    buf[n] = 0;
    return buf;
}

#endif  /* CUDA_SIEVE_BIGINT_CUH */
