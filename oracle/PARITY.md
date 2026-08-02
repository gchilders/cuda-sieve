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

`logbase = 2^(1/scale)`, and `log p` in sieve units is `round(log2(p) * scale)`.
The bound is exactly `round(scale * lambda * lpb)`:
`1.28 * 3.5 * 32 = 143.4 → 143`, and `1.93 * 2.35 * 31 = 140.6 → 141`.

`scale` exists only so that `scale * log2(maxnorm)` fits in a byte
(`1.28 * 196.61 = 251.7`). **We use 16-bit cells and do not need it** — we can
run at scale 1.0, or higher, for free. las is throwing away 0.39 bits of
resolution per position that we get to keep, which means fewer false survivors
into cofactoring.

Independent check of our norm setup: with the q-lattice above, our A (max |a|)
and B (max |b|) give `log2|Y1*A| = 116.4`, `log2|Y0*B| = 132.2`, so
`log2(maxnorm) ≈ 132.2` on side 0 against las's **131.86**.

## ⚠ The dumps are NOT usable as a byte-diff oracle

Measured 2026-08-01, after the sieve was brought onto CADO's factor base and
scale. Two independent signs that the dump is wrong, not us:

| | mean sieved log / position |
|---|---:|
| analytic over the whole factor base (from `c183.fb1`) | 54.65 |
| our sieve | 54.2 |
| analytic, small part only | 39.31 |
| **these dumps** | **41.1** |

The dump carries roughly the small-sieve contribution and not the
bucket-sieved one, although `las-process-bucket-region.cpp` calls
`apply_buckets()` before `SminusS()` and the write. And it does not line up
positionally: correlation with our region is **+0.033** direct and **+0.016**
i-mirrored (both orientations checked, since our basis is las's with the first
vector negated), against a **+0.636** control correlating two of our own dumps
at the same positions.

That is consistent with how the flag was found — dead behind an
`ASSERT_ALWAYS` that made its only `open()` call unreachable. Use
`las_tracek`/`TRACE_K` to follow a single `(i,j)` instead, or report upstream.
Everything else in this file comes from the `-v` log and is sound.

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
