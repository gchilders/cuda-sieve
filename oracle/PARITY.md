# CADO oracle for byte-exact parity — captured 2026-08-01

Everything below came from one `las` run on `c183.poly`. This is the reference
Path 3 gate 2 diffs against.

## Reproduce

```
makefb -poly c183.poly -lim 134200000 -maxbits 15 -t 6 -out c183.fb1

las -poly c183.poly -fb1 c183.fb1 \
    -lim0 67100000 -lim1 134200000 -lpb0 31 -lpb1 32 \
    -mfb0 60 -mfb1 92 -lambda0 2.35 -lambda1 3.5 \
    -A 29 -sqside 1 -q0 120000011 -nq 1 -adjust-strategy 0 -B 14 \
    -t 1 -v -dumpfile las_dump
```

`-A 29` is `logI=15, J=16384` (I15e). `-B 14` sets the bucket region to 2^14 to
match the GPU's measured optimum. `-t 1` is mandatory: `-dumpfile` asserts on
more than one thread.

**`-dumpfile` needed a one-line patch to CADO.** In revision `0574bc39d`,
`sieve/las-threads-work-data.cpp:131` has `ASSERT_ALWAYS(!las.dump_filename)`
immediately before the only `dumpfile.open()` call site in the tree, so any use
of `-dumpfile` aborts. The assertion was removed. Original file preserved here
as `las-threads-work-data.cpp.orig`; revert with

```
cp oracle/las-threads-work-data.cpp.orig ~/cado-nfs/sieve/las-threads-work-data.cpp
```

Because `doing` is not yet initialised at that point, the dump is named
`sq0.rho0` regardless of the actual q. With `-nq 1` the contents are still
exactly the one special-q requested; the files here were renamed by hand.

## The special-q

`q0=120000011` has no root of f, so las advanced to the next one that does:

| | |
|---|---|
| q | **120000053** |
| rho | **112625526** |
| las basis (its naming) | `a0=7374527; b0=-1; a1=120000053; b1=0` |
| I, J | 32768, 16384 |

**Naming trap.** las groups the basis by *coordinate* — `a = a0*i + a1*j`,
`b = b0*i + b1*j`. `qlat_t` in `bench/` groups by *vector*: `(a0,a1)` and
`(b0,b1)` are the two basis vectors, so `a = i*a0 + j*b0`, `b = i*a1 + j*b1`.
The translation is **las `(a0,b0,a1,b1)` = qlat_t `(a0,a1,b0,b1)`**. Getting
this wrong transposes the basis and yields a valid lattice of the right density
that is simply not the one the norms describe — nothing but this comparison
catches it.

`bench/qlat_build` with `--q 120000053 --rho 112625526` produces
`(-7374527, 1, 120000053, 0)`, which is las's basis with the first vector
negated: **the same reduced basis. Gate passed.**

## Log scales — the parity-critical constants

| side | log2(maxnorm) | scale | logbase | bound |
|---|---:|---:|---:|---:|
| 0 (rational) | 131.86 | 1.93 | 1.433439 | **141** |
| 1 (algebraic) | 196.61 | 1.28 | 1.722273 | **143** |

**The printed `scale` is rounded to 2 dp and using it is wrong.** `logbase` is
printed to 7 figures and `logbase = 2^(1/scale)`, so the exact scale is
`1/log2(logbase)`:

| side | printed | **exact** |
|---|---:|---:|
| 0 | 1.93 | **1.925** |
| 1 | 1.28 | **1.275** |

The difference moves `fb_log` by one unit for a band of primes — p = 25811171
gives 32 at scale 1.28 and 31, which is what las actually applies, at 1.275.
Found by the gate-5 trace below; every run before 2026-08-02 used the 2-dp
values.

`fb_log(n) = floor(log2(n) * scale + 0.5)` (`fb.cpp:86`), and for a factor-base
entry the increment is `fb_log(p^nexp) - fb_log(p^oldexp)`.

The bound is **not** a round. `las-norms.cpp:270`:

```
bound = (unsigned char) (r * scale + LOGNORM_GUARD_BITS);     r = lambda * lpb
```

a truncating cast plus a guard bit. With the exact scales that is
`trunc(3.5*32*1.275 + 1) = 143` and `trunc(2.35*31*1.925 + 1) = 141`, both
exact. The old `round(scale*lambda*lpb)` agreed only because the rounded scales
happened to compensate.

`scale` exists only so that `scale * log2(maxnorm)` fits in a byte
(`1.28 * 196.61 = 251.7`). **We use 16-bit cells and do not need it** — we can
run at scale 1.0, or higher, for free. las is throwing away 0.39 bits of
resolution per position that we get to keep, which means fewer false survivors
into cofactoring.

Independent check of our norm setup: with the q-lattice above, our A (max |a|)
and B (max |b|) give `log2|Y1*A| = 116.4`, `log2|Y0*B| = 132.2`, so
`log2(maxnorm) ≈ 132.2` on side 0 against las's **131.86**.

## Gate 5 — byte-exact parity via TRACE_K. **PASSED 2026-08-02.**

`las_tracek` is a stock CADO target (`sieve/CMakeLists.txt:111`, built with
`TRACE_K=1`) and needs no patching. It follows one position through the whole
pipeline and prints **every ideal applied to it, with its log** — strictly more
informative than a byte dump, and independent of the broken dumpfile path.

```
cd build/<host> && make las_tracek
las_tracek <the same flags as above, minus -dumpfile> -traceab 946175173703,4999
```

Use **`-traceab`**, not `-traceij`: it is basis-independent, so the i-mirroring
between our basis and las's never enters. (Confirmed by the trace header: our
`(i,j) = (4999,8192)` is las's `(-4999,8192)` for the same `(a,b)`, exactly as
expected from our basis being las's with the first vector negated.)

Output to read:

```
# Tracing relation (a,b)=(...) (i,j)=(-9237,15022), (N,x)=(30044,7147)
# After side 1 init_norms_bucket_region, N=30044 S[7147]=248
# Add log(hint=227,side 1) = 24 to S[7147] = 0, ... -> 24
# Add log(3,side 1) = 4 to S[7147] = 59, ... -> 63
...
# Final value on side 1, N=30044 S[7147]=154
```

so the **sieved log sum is `init - final`**, and the per-ideal lines name every
contributor. `hint=` marks a bucketed entry (the number is the slice index); a
bare number is the line-sieved modulus.

### Result

Four positions, both sides, `q=120000053, rho=112625526`. Our side comes from
`bench/fbtest --trace i,j`.

| (i,j) ours | side | las sum | ours sum | ideals |
|---|---|---:|---:|---:|
| (4999, 8192) | 1 | 22 | **22** | 5 |
| (4999, 8192) | 0 | 58 | **58** | 6 |
| (9237, 15022) | 1 | 94 | **94** | 15 |
| (9237, 15022) | 0 | 65 | **65** | 6 |
| (2978, 8393) | 1 | 46 | **46** | 14 |
| (2978, 8393) | 0 | 94 | **94** | 2 |
| (13198, 9151) | 1 | 51 | **51** | 13 |
| (13198, 9151) | 0 | 63 | **63** | 4 |

**8 of 8 exact**, and not only the totals — the *ideal lists* agree entry by
entry. Two positions were picked because long projective power ladders hit them
(the 3-ladder 3,9,27,81,243,729,2187 and the 2-ladder 2,4,8,16,32,64), which is
the direct gate on the nonzero-reciprocal bug.

Getting here took two real fixes, both found by this trace and by nothing else:
the exact scale above, and a base-prime bug in the makefb parser (RESULTS.md
findings 18–25).

### The one thing that does NOT match: norm initialisation

| (i,j) | side 1 exact / las | side 0 exact / las |
|---|---|---|
| (4999, 8192) | 241 / 243 | 251 / 251 |
| (9237, 15022) | 247 / 248 | 252 / 253 |
| (2978, 8393) | 241 / 242 | 249 / 249 |
| (13198, 9151) | 246 / 246 | 253 / 254 |

las is **0 to +2 above** the exact value, scattered. One unit of that is
`LOGNORM_GUARD_BITS`, which las prints in its own banner (`# las flags:
BUCKET_SIEVE_POWERS TRACE_K LOGNORM_GUARD_BITS=1.00`) and adds deliberately so
the byte cannot wrap below zero. The rest is las's norm *approximation*: it
interpolates log|F| along a row instead of evaluating the polynomial at every
position, and computes the exact norm only later, for survivors.

So our fp32 Horner is the **more accurate** of the two, and the gap is bounded
and one-directional — las over-estimates, so it is slightly more permissive and
passes slightly more false survivors. This is a difference in what las *chooses*
to compute, not an error on either side, and it is why a byte-for-byte region
diff could never have reached zero even with a working dumpfile.

## ⚠ The `-dumpfile` oracle is still not usable (superseded by gate 5)

## What the dump contains

Written in `las-process-bucket-region.cpp` right after `SminusS()`:

```
init_norms(side)              S[x]  = round(scale * log2|F(a,b)|), capped 255
apply_buckets + small_sieve   SS[x] = sum of scaled log p landing on x
SminusS                       S[x]  = max(S[x] - SS[x], 0)
```

One byte per position, in x order, `I*J = 536,870,912` bytes. Survivor when
`S[x] <= bound`.

Note the shape difference: las needs a **second full region array `SS`** plus a
subtract pass. Our kernel starts cells at `CINIT - T(x)` and *adds* (a GPU
16-bit half-word atomic cannot borrow), which fuses init, accumulate and
threshold into one shared-memory pass with no second array.

| file | contents |
|---|---|
| `c183.q120000053.side0.dump` | rational side, 512 MB |
| `c183.q120000053.side1.dump` | algebraic side, 512 MB |
| `las_q120000053.log` | the full `-v` log |

Measured distributions (`bench/dumpcmp stat`):

| | side 0 | side 1 |
|---|---:|---:|
| mean S | 215.66 | 204.16 |
| S ≤ bound | 23,952,829 | 21,650,256 |
| one in | 22.4 | 24.8 |

las's own two-sided survivor count is **797,028** (1 in 674), consistent with
the product of the two one-sided rates (966K if independent).

## Factor base composition — CADO vs GGNFS

| | GGNFS `input.job.afb.0` | CADO |
|---|---:|---:|
| algebraic, p < 32768 | 3,631 entries | 3,839 ideals |
| algebraic, bucketed | 6,843,511 (truncated at q) | 7,601,777 ideals (to `alim`) |
| rational | not stored; built on the fly | 3,954,896 ideals |

Two differences to carry into any comparison:

1. **CADO includes prime powers** (`BUCKET_SIEVE_POWERS`, `powlim=ULONG_MAX`).
   GGNFS's `.afb.0` has none — the 208-ideal gap in the small part is exactly
   this. *(Closed 2026-08-01: we load `c183.fb1` directly.)*

   **Root encoding, and the trap in it (2026-08-02).** A root is stored in
   `[0, 2q)`; at or above `q` it is projective with reciprocal `rr = r - q`,
   meaning `a*rr == b (mod q)`. For a **prime** `q` the only projective root is
   `rr = 0`, the classical `b == 0 (mod q)` — so assuming `rr = 0` is correct
   on a prime-only factor base and wrong as soon as powers arrive. `c183.fb1`
   has **35 entries with a nonzero reciprocal**, the projective ladder above
   the leading coefficient `110880 = 2^5*3^2*5*7*11`, starting at `4:4,3: 6`.
   They are worth 4.7e8 updates per special-q. Treating them as `rr = 0` gives
   a lattice of the *same density* on *different congruences* — see RESULTS.md
   finding 18 for why that defeats every density-based gate.

2. **Side 0's projective ladder.** `Y1 = 59*101*127*281*1259*38321*5746453`,
   so seven projective primes; the ladders `59^2, 101^2, 127^2` stay under
   `2^15`. Those three were the **-3** in side 0's small part (3,586 vs 3,589)
   and are now emitted. Side 0 is 3,957,374 ideals. The remaining -1,114 in the
   bucketed count is a `powlim` difference: las builds side 0 on the fly with
   `powlim = ULONG_MAX`, we cap at `maxbits = 15`. For parity runs, pin las
   down with `-powlim0 32767 -powlim1 32767` rather than chasing it up.
3. **CADO does not truncate the factor base at q**, GGNFS does. Confirmed
   here: largest prime read was 134,199,997 against `alim = 134,200,000`.

## Not a timing reference

This run took 30.5s elapsed for one special-q, single-threaded, on a box with
four other jobs saturating memory. It produced 37 relations. **Do not quote it
as a CADO baseline** — timings get re-taken on a quiet box.
