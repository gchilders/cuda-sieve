/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * IEEE-754 binary64 in integer operations, for Metal.
 *
 * Apple GPUs have no fp64 and MSL rejects the `double` type outright, but two
 * device-side sites in this siever need it and neither is optional:
 *
 *   1. bench_kernels.cu:525 -- the fp64 recompute in k_apply when the fp32
 *      Horner cancels near a root line of F. bench.h:329 records what losing
 *      it costs: 144 of 63,497 positions round to the wrong sieve-log value,
 *      in BOTH directions, so both false survivors and lost relations.
 *   2. td.cuh:981 -> prp.cuh:194 cof_classify, whose gap test is deliberately
 *      written against doubles to match CADO's check_leftover_norm.
 *
 * Both are cold -- the first fires on well under one cell in a thousand, the
 * second is a handful of operations per survivor -- so this file optimises for
 * being *exactly right* and not at all for speed. A double-float (two-float)
 * representation would be far faster and is the usual answer, but it carries
 * ~48 bits against fp64's 53 and would silently change which cells survive.
 * Bit-exactness against the CUDA build is the whole point, so: soft-float.
 *
 * REPRESENTATION. A value is carried as its 64-bit IEEE bit pattern. Working
 * values are unpacked to (sign, exp, sig) where `sig` is a 64-bit significand
 * with bit 63 set -- subnormals are normalised on unpack, so the arithmetic
 * below never has a special case for them -- and `exp` is the biased exponent
 * such that
 *
 *     value = (sig / 2^63) * 2^(exp - 1023)
 *
 * Intermediates use a 128-bit (hi, lo) pair with bit 127 of the pair set, on
 * the same convention:
 *
 *     value = ((hi:lo) / 2^127) * 2^(exp - 1023)
 *
 * which makes sf_add, sf_mul and sf_fma share one align-add-normalise core
 * and keeps a multiply's low 64 bits alive so fma is genuinely single-rounded.
 *
 * Rounding is round-to-nearest-even throughout, via one `sf_round_pack`.
 */
#ifndef CUDA_SIEVE_SOFTFP64_H
#define CUDA_SIEVE_SOFTFP64_H

#include "msl_compat.h"

typedef sf_u64 sf64;

#define SF_ONE_U64  ((sf_u64)1)
#define SF_SIGNBIT  (SF_ONE_U64 << 63)
#define SF_MANT52   ((SF_ONE_U64 << 52) - SF_ONE_U64)

#define SF_CLS_ZERO 0u
#define SF_CLS_NORM 1u
#define SF_CLS_INF  2u
#define SF_CLS_NAN  3u

struct sf_parts { sf_u32 sign; sf_i32 exp; sf_u64 sig; sf_u32 cls; };

SF_FN struct sf_parts sf_unpack(sf64 x)
{
    struct sf_parts p;
    p.sign = (sf_u32)(x >> 63);
    sf_i32 e = (sf_i32)((x >> 52) & 0x7ff);
    sf_u64 m = x & SF_MANT52;
    if (e == 0) {
        if (m == 0) { p.cls = SF_CLS_ZERO; p.exp = 0; p.sig = 0; return p; }
        /* subnormal: normalise so the rest of the file has no special case.
         * value = m * 2^-1074, and we need sig/2^63 * 2^(exp-1023). */
        sf_u32 n = sf_clz64(m);
        p.sig = m << n;
        p.exp = 12 - (sf_i32)n;
        p.cls = SF_CLS_NORM;
        return p;
    }
    if (e == 0x7ff) {
        p.cls = m ? SF_CLS_NAN : SF_CLS_INF;
        p.exp = e; p.sig = m;
        return p;
    }
    p.sig = (m | (SF_ONE_U64 << 52)) << 11;   /* bit 63 set */
    p.exp = e;
    p.cls = SF_CLS_NORM;
    return p;
}

SF_FN sf64 sf_inf(sf_u32 sign)  { return ((sf_u64)sign << 63) | ((sf_u64)0x7ff << 52); }
SF_FN sf64 sf_zero(sf_u32 sign) { return (sf_u64)sign << 63; }
SF_FN sf64 sf_nan(void)         { return ((sf_u64)0x7ff << 52) | (SF_ONE_U64 << 51); }

/* Round (sign, exp, sig with bit 63 set, sticky) to a binary64.
 * `exp` is the biased exponent of the leading bit; it may be out of range on
 * entry and is resolved here to inf or a subnormal. */
SF_FN sf64 sf_round_pack(sf_u32 sign, sf_i32 exp, sf_u64 sig, sf_u32 sticky)
{
    if (sig == 0) return sf_zero(sign);

    if (exp <= 0) {
        /* Subnormal result: shift right by (1 - exp), collecting sticky. */
        sf_i32 sh = 1 - exp;
        if (sh > 63) { sticky |= (sig != 0) ? 1u : 0u; sig = 0; }
        else {
            sticky |= ((sig & ((SF_ONE_U64 << sh) - SF_ONE_U64)) != 0) ? 1u : 0u;
            sig >>= sh;
        }
        exp = 0;
    }

    sf_u64 frac  = sig >> 11;                                  /* 53 bits */
    sf_u32 rnd   = (sf_u32)((sig >> 10) & 1);
    sf_u32 stk   = sticky | (((sig & 0x3ff) != 0) ? 1u : 0u);

    if (rnd && (stk || (frac & 1))) {
        frac++;
        if (frac >> 53) { frac >>= 1; exp++; }
    }

    if (exp >= 0x7ff) return sf_inf(sign);

    if (frac >> 52) {                       /* normal */
        if (exp <= 0) exp = 1;              /* subnormal rounded up to normal */
        return ((sf_u64)sign << 63) | ((sf_u64)exp << 52) | (frac & SF_MANT52);
    }
    return ((sf_u64)sign << 63) | frac;     /* subnormal, exponent field 0 */
}

/* ---- conversions ------------------------------------------------------- */

SF_FN sf64 sf_from_u32(sf_u32 v)
{
    if (v == 0) return 0;
    sf_u32 n = sf_clz64((sf_u64)v);
    return sf_round_pack(0u, 1086 - (sf_i32)n, ((sf_u64)v) << n, 0u);
}

SF_FN sf64 sf_from_i64(sf_i64 v)
{
    if (v == 0) return 0;
    sf_u32 sign = 0u;
    sf_u64 u;
    if (v < 0) { sign = 1u; u = (sf_u64)0 - (sf_u64)v; }   /* INT64_MIN safe */
    else u = (sf_u64)v;
    sf_u32 n = sf_clz64(u);
    return sf_round_pack(sign, 1086 - (sf_i32)n, u << n, 0u);
}

SF_FN sf64 sf_from_f32(float f)
{
    sf_u32 b = sf_f2b(f);
    sf_u32 sign = b >> 31;
    sf_i32 e = (sf_i32)((b >> 23) & 0xff);
    sf_u32 m = b & 0x7fffff;
    if (e == 0) {
        if (m == 0) return sf_zero(sign);
        /* value = m * 2^-149, and sig = m << n has bit 63 set, so
         * sig * 2^(exp-1023-63) = m * 2^-149 gives exp = 937 - n. */
        sf_u32 n = sf_clz64((sf_u64)m);
        return sf_round_pack(sign, 937 - (sf_i32)n, ((sf_u64)m) << n, 0u);
    }
    if (e == 0xff) return m ? sf_nan() : sf_inf(sign);
    return ((sf_u64)sign << 63) | ((sf_u64)(e - 127 + 1023) << 52)
           | (((sf_u64)m) << 29);
}

/* Round a binary64 to a float, round-to-nearest-even. */
SF_FN float sf_to_f32(sf64 x)
{
    struct sf_parts p = sf_unpack(x);
    if (p.cls == SF_CLS_ZERO) return sf_b2f(p.sign << 31);
    if (p.cls == SF_CLS_INF)  return sf_b2f((p.sign << 31) | 0x7f800000u);
    if (p.cls == SF_CLS_NAN)  return sf_b2f(0x7fc00000u);

    sf_i32 fe = p.exp - 1023 + 127;         /* target biased float exponent */
    sf_u64 sig = p.sig;
    sf_u32 sticky = 0u;

    if (fe <= 0) {                           /* subnormal float or zero */
        sf_i32 sh = 1 - fe;
        if (sh > 63) { sticky = (sig != 0) ? 1u : 0u; sig = 0; }
        else {
            sticky = ((sig & ((SF_ONE_U64 << sh) - SF_ONE_U64)) != 0) ? 1u : 0u;
            sig >>= sh;
        }
        fe = 0;
    }
    sf_u32 frac = (sf_u32)(sig >> 40);                       /* 24 bits */
    sf_u32 rnd  = (sf_u32)((sig >> 39) & 1);
    sf_u32 stk  = sticky | (((sig & ((SF_ONE_U64 << 39) - SF_ONE_U64)) != 0) ? 1u : 0u);
    if (rnd && (stk || (frac & 1))) {
        frac++;
        if (frac >> 24) { frac >>= 1; fe++; }
    }
    if (fe >= 0xff) return sf_b2f((p.sign << 31) | 0x7f800000u);
    if (frac >> 23) {
        if (fe <= 0) fe = 1;
        return sf_b2f((p.sign << 31) | ((sf_u32)fe << 23) | (frac & 0x7fffffu));
    }
    return sf_b2f((p.sign << 31) | frac);
}

/* ---- sign and comparison ----------------------------------------------- */

SF_FN sf64 sf_neg(sf64 a) { return a ^ SF_SIGNBIT; }
SF_FN sf64 sf_abs(sf64 a) { return a & ~SF_SIGNBIT; }

SF_FN int sf_isnan(sf64 a)
{ return ((a >> 52) & 0x7ff) == 0x7ff && (a & SF_MANT52) != 0; }

/* a < b, IEEE semantics (NaN compares false, -0 == +0). */
SF_FN int sf_lt(sf64 a, sf64 b)
{
    if (sf_isnan(a) || sf_isnan(b)) return 0;
    if (((a | b) & ~SF_SIGNBIT) == 0) return 0;              /* both zero */
    sf_u32 sa = (sf_u32)(a >> 63), sb = (sf_u32)(b >> 63);
    if (sa != sb) return (int)sa;
    return sa ? (a > b) : (a < b);
}

/* ---- the shared 128-bit align / add / normalise core -------------------- */

/* Shift (hi:lo) right by n in [0,255], OR-ing anything lost into sticky. */
struct sf_w128 { sf_u64 hi, lo; sf_u32 sticky; };

SF_FN struct sf_w128 sf_shr128(sf_u64 hi, sf_u64 lo, sf_i32 n, sf_u32 sticky)
{
    struct sf_w128 r;
    if (n <= 0) { r.hi = hi; r.lo = lo; r.sticky = sticky; return r; }
    if (n >= 128) {
        r.hi = 0; r.lo = 0;
        r.sticky = sticky | (((hi | lo) != 0) ? 1u : 0u);
        return r;
    }
    if (n < 64) {
        sticky |= ((lo & ((SF_ONE_U64 << n) - SF_ONE_U64)) != 0) ? 1u : 0u;
        r.lo = (lo >> n) | (hi << (64 - n));
        r.hi = hi >> n;
    } else if (n == 64) {
        sticky |= (lo != 0) ? 1u : 0u;
        r.lo = hi; r.hi = 0;
    } else {
        sf_i32 m = n - 64;
        sticky |= ((lo != 0) || ((hi & ((SF_ONE_U64 << m) - SF_ONE_U64)) != 0)) ? 1u : 0u;
        r.lo = hi >> m; r.hi = 0;
    }
    r.sticky = sticky;
    return r;
}

/* Add or subtract two 128-bit significands that are each normalised with bit
 * 127 set, at biased exponents ea and eb, and round to a binary64. */
SF_FN sf64 sf_addsub128(sf_u32 sa, sf_i32 ea, sf_u64 ahi, sf_u64 alo,
                        sf_u32 sb, sf_i32 eb, sf_u64 bhi, sf_u64 blo,
                        sf_u32 sticky)
{
    /* Order so that a is the larger magnitude. */
    if (eb > ea || (eb == ea && (bhi > ahi || (bhi == ahi && blo > alo)))) {
        sf_u32 ts = sa; sa = sb; sb = ts;
        sf_i32 te = ea; ea = eb; eb = te;
        sf_u64 th = ahi; ahi = bhi; bhi = th;
        sf_u64 tl = alo; alo = blo; blo = tl;
    }

    struct sf_w128 bs = sf_shr128(bhi, blo, ea - eb, sticky);
    bhi = bs.hi; blo = bs.lo; sticky = bs.sticky;

    sf_u64 hi, lo;
    sf_i32 nshift;

    if (sa == sb) {
        lo = alo + blo;
        sf_u32 c = (lo < alo) ? 1u : 0u;
        hi = ahi + bhi;
        sf_u32 ov = (hi < ahi) ? 1u : 0u;
        sf_u64 hi2 = hi + c;
        if (hi2 < hi) ov = 1u;
        hi = hi2;
        if (ov) {                      /* carried out of bit 127 */
            sticky |= (sf_u32)(lo & 1);
            lo = (lo >> 1) | (hi << 63);
            hi = (hi >> 1) | SF_SIGNBIT;
            nshift = -1;
        } else nshift = 0;
    } else {
        sf_u32 brw = (alo < blo) ? 1u : 0u;
        lo = alo - blo;
        hi = ahi - bhi - brw;
        /* Bits of b were discarded, so the true b is larger than what we
         * subtracted: the exact result lies strictly between (hi:lo)-1 and
         * (hi:lo). Represent it as that value with a nonzero remainder. */
        if (sticky) {
            if (lo == 0) hi--;
            lo--;
        }
        if ((hi | lo) == 0) return sf_zero(0u);     /* exact cancellation */
        nshift = 0;
        if (hi == 0) { hi = lo; lo = 0; nshift += 64; }
        sf_u32 n = sf_clz64(hi);
        if (n) { hi = (hi << n) | (lo >> (64 - n)); lo <<= n; nshift += (sf_i32)n; }
    }

    return sf_round_pack(sa, ea - nshift, hi, sticky | ((lo != 0) ? 1u : 0u));
}

/* ---- add / sub --------------------------------------------------------- */

SF_FN sf64 sf_add(sf64 a, sf64 b)
{
    struct sf_parts pa = sf_unpack(a), pb = sf_unpack(b);
    if (pa.cls == SF_CLS_NAN || pb.cls == SF_CLS_NAN) return sf_nan();
    if (pa.cls == SF_CLS_INF) {
        if (pb.cls == SF_CLS_INF && pa.sign != pb.sign) return sf_nan();
        return sf_inf(pa.sign);
    }
    if (pb.cls == SF_CLS_INF) return sf_inf(pb.sign);
    if (pa.cls == SF_CLS_ZERO) return (pb.cls == SF_CLS_ZERO)
                                    ? sf_zero(pa.sign & pb.sign) : b;
    if (pb.cls == SF_CLS_ZERO) return a;
    return sf_addsub128(pa.sign, pa.exp, pa.sig, 0,
                        pb.sign, pb.exp, pb.sig, 0, 0u);
}

SF_FN sf64 sf_sub(sf64 a, sf64 b) { return sf_add(a, sf_neg(b)); }

/* ---- multiply ---------------------------------------------------------- */

SF_FN sf64 sf_mul(sf64 a, sf64 b)
{
    struct sf_parts pa = sf_unpack(a), pb = sf_unpack(b);
    sf_u32 sign = pa.sign ^ pb.sign;
    if (pa.cls == SF_CLS_NAN || pb.cls == SF_CLS_NAN) return sf_nan();
    if (pa.cls == SF_CLS_INF || pb.cls == SF_CLS_INF) {
        if (pa.cls == SF_CLS_ZERO || pb.cls == SF_CLS_ZERO) return sf_nan();
        return sf_inf(sign);
    }
    if (pa.cls == SF_CLS_ZERO || pb.cls == SF_CLS_ZERO) return sf_zero(sign);

    sf_u64 hi = sf_mulhi64(pa.sig, pb.sig);
    sf_u64 lo = pa.sig * pb.sig;
    sf_i32 exp = pa.exp + pb.exp - 1022;
    if (!(hi & SF_SIGNBIT)) {              /* MSB landed at 126: normalise */
        hi = (hi << 1) | (lo >> 63); lo <<= 1; exp--;
    }
    return sf_round_pack(sign, exp, hi, (lo != 0) ? 1u : 0u);
}

/* ---- fused multiply-add: a*b + c, single rounding ----------------------- */

SF_FN sf64 sf_fma(sf64 a, sf64 b, sf64 c)
{
    struct sf_parts pa = sf_unpack(a), pb = sf_unpack(b), pc = sf_unpack(c);
    sf_u32 psign = pa.sign ^ pb.sign;

    if (pa.cls == SF_CLS_NAN || pb.cls == SF_CLS_NAN || pc.cls == SF_CLS_NAN)
        return sf_nan();
    if (pa.cls == SF_CLS_INF || pb.cls == SF_CLS_INF) {
        if (pa.cls == SF_CLS_ZERO || pb.cls == SF_CLS_ZERO) return sf_nan();
        if (pc.cls == SF_CLS_INF && pc.sign != psign) return sf_nan();
        return sf_inf(psign);
    }
    if (pc.cls == SF_CLS_INF) return sf_inf(pc.sign);
    if (pa.cls == SF_CLS_ZERO || pb.cls == SF_CLS_ZERO) {
        if (pc.cls == SF_CLS_ZERO) return sf_zero(psign & pc.sign);
        return c;
    }

    sf_u64 phi = sf_mulhi64(pa.sig, pb.sig);
    sf_u64 plo = pa.sig * pb.sig;
    sf_i32 pexp = pa.exp + pb.exp - 1022;
    if (!(phi & SF_SIGNBIT)) { phi = (phi << 1) | (plo >> 63); plo <<= 1; pexp--; }

    if (pc.cls == SF_CLS_ZERO)
        return sf_round_pack(psign, pexp, phi, (plo != 0) ? 1u : 0u);

    return sf_addsub128(psign, pexp, phi, plo,
                        pc.sign, pc.exp, pc.sig, 0, 0u);
}

/* ---- divide ------------------------------------------------------------ */

SF_FN sf64 sf_div(sf64 a, sf64 b)
{
    struct sf_parts pa = sf_unpack(a), pb = sf_unpack(b);
    sf_u32 sign = pa.sign ^ pb.sign;
    if (pa.cls == SF_CLS_NAN || pb.cls == SF_CLS_NAN) return sf_nan();
    if (pa.cls == SF_CLS_INF) {
        if (pb.cls == SF_CLS_INF) return sf_nan();
        return sf_inf(sign);
    }
    if (pb.cls == SF_CLS_INF)  return sf_zero(sign);
    if (pb.cls == SF_CLS_ZERO) return (pa.cls == SF_CLS_ZERO) ? sf_nan() : sf_inf(sign);
    if (pa.cls == SF_CLS_ZERO) return sf_zero(sign);

    /* Restoring division, 64 quotient bits. The remainder needs 65 bits, so
     * its top bit is carried separately. Both significands have bit 63 set,
     * so the quotient is in [2^63, 2^64): at most one normalising shift. */
    sf_u64 r = pa.sig, d = pb.sig, q = 0;
    sf_u32 rtop = 0u;
    for (int i = 0; i < 64; i++) {
        q <<= 1;
        if (rtop || r >= d) { r -= d; rtop = 0u; q |= 1; }
        rtop = (sf_u32)(r >> 63);
        r <<= 1;
    }
    sf_u32 sticky = ((r != 0) || rtop) ? 1u : 0u;
    sf_i32 exp = pa.exp - pb.exp + 1023;
    if (!(q & SF_SIGNBIT)) { q <<= 1; exp--; }   /* siga < sigb */
    return sf_round_pack(sign, exp, q, sticky);
}

#endif  /* CUDA_SIEVE_SOFTFP64_H */
