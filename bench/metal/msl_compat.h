/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * One header so that a piece of pure arithmetic can be compiled BOTH by the
 * Metal shader compiler and by the host C++ compiler, from the same source.
 *
 * This matters more here than it looks. The port's correctness argument rests
 * on the device reproducing the host's fp64 and fp32 results bit for bit; if
 * the two builds compiled different source, the test would be proving nothing
 * about the code that actually ships. So: one file, two compilers.
 *
 * MSL is C++14 with no <stdint.h>, no `double`, and address-space qualifiers
 * on every pointer. The primitives that include this header are deliberately
 * value-only (no pointer parameters at all), which sidesteps address spaces
 * entirely and is why they can be shared without a per-address-space overload
 * set.
 */
#ifndef CUDA_SIEVE_MSL_COMPAT_H
#define CUDA_SIEVE_MSL_COMPAT_H

#if defined(__METAL_VERSION__)

#include <metal_stdlib>

typedef ulong  sf_u64;
typedef uint   sf_u32;
typedef long   sf_i64;
typedef int    sf_i32;

#define SF_FN static inline

/* clz on a 64-bit zero is undefined in C; Metal's returns 64. Callers here
 * never pass zero, but keep the shapes identical on both sides anyway. */
SF_FN sf_u32 sf_clz64(sf_u64 x) { return (sf_u32)metal::clz(x); }
SF_FN sf_u64 sf_mulhi64(sf_u64 a, sf_u64 b) { return metal::mulhi(a, b); }

/* Bit-exact float <-> uint punning. */
SF_FN sf_u32 sf_f2b(float f)  { return as_type<uint>(f); }
SF_FN float  sf_b2f(sf_u32 b) { return as_type<float>(b); }

SF_FN float sf_fmaf(float a, float b, float c) { return metal::fma(a, b, c); }

/* The ONE place this family of headers takes a pointer. MSL needs an address
 * space on it; the real kernels copy a bn_t into registers before calling
 * helpers (td.cuh:977 `bn_t c = cof[t];` is the pattern), so `thread` is the
 * right default and `device` callers copy first, exactly as CUDA's do. */
#define SF_BN_PTR thread

#else   /* ---------------------------- host ---------------------------- */

#include <stdint.h>
#include <string.h>

typedef uint64_t sf_u64;
typedef uint32_t sf_u32;
typedef int64_t  sf_i64;
typedef int32_t  sf_i32;

#define SF_FN static inline

SF_FN sf_u32 sf_clz64(sf_u64 x) { return x ? (sf_u32)__builtin_clzll(x) : 64u; }
SF_FN sf_u64 sf_mulhi64(sf_u64 a, sf_u64 b)
{
    return (sf_u64)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
}

SF_FN sf_u32 sf_f2b(float f)  { sf_u32 b; memcpy(&b, &f, 4); return b; }
SF_FN float  sf_b2f(sf_u32 b) { float f;  memcpy(&f, &b, 4); return f; }

/* The host MUST NOT contract this into anything else, or the two builds
 * diverge. Every call site is an explicit fma for exactly that reason; the
 * host build additionally compiles with -ffp-contract=off. */
SF_FN float sf_fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

#define SF_BN_PTR /* no address spaces on the host */

#endif

#endif  /* CUDA_SIEVE_MSL_COMPAT_H */
