/* Fixed-width big integers for exact norms and trial division.
 *
 * WHY 256 BITS. Measured on the real lattice for this job (q=120000053,
 * rho=112625526, skew 1.154e8), over the corners of the sieve area plus 20K
 * interior points:
 *
 *     max |a| = 41 bits      max |F(a,b)| = 224 bits
 *     max |b| = 15 bits      max |G(a,b)| = 132 bits
 *
 * and the largest individual homogeneous term c_k a^k b^(5-k) is also 224 bits,
 * so a 6-term sum stays under 227. Eight 32-bit limbs give 256, i.e. ~28 bits
 * of headroom over the worst term. That headroom is a property of THIS
 * polynomial and lattice shape, not a general guarantee -- `bn_mul_u64`
 * reports overflow rather than wrapping so a different job cannot corrupt a
 * norm silently.
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

#define BN_LIMBS 8                      /* 8 x 32 = 256 bits */

#if defined(__CUDACC__)
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

/* 64x64 -> high 64. The GPU has an instruction; the host has __int128. */
#if defined(__CUDA_ARCH__)
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

/* Writes into buf (needs 80 bytes for 256 bits) and returns it. */
static inline char *bn_to_dec(const bn_t *x, char *buf)
{
    char tmp[80];
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
