# Path 1 — bucket-fill microbenchmark results

**Hardware:** RTX 5070 (sm_120, 48 SM, 48 MB L2, 99 KB opt-in smem, 672.0 GB/s)
**Toolchain:** CUDA 13.2.78, `-gencode arch=compute_120,code=sm_120`
**Date:** 2026-08-01
**Input:** `oracle/input.job.afb.0` — the real C183 algebraic factor base, no synthetic data
**Config unless stated:** I15e (`logI=15`, `J=16384`, A=5.369e8), region 2^15, q=120000011,
FB truncated at q (GGNFS convention) → 6,843,511 entries, **312,265,384 records/special-q**

## Correctness

The Franke-Kleinjung reduction and walk are ported from CADO
(`las-plattice.hpp`, `las-reduce-plattice-simplistic.hpp`) and shared
`__host__ __device__` between `verify_cpu.c` and the kernels.

| gate | result |
|---|---|
| FK walk vs brute-force enumeration (logI=8, J=128, 24 primes × 5 roots) | **exact** |
| GPU records landed vs single-threaded CPU reference | **1,690,293 = 1,690,293** |
| Bucket imbalance (max/mean) | **1.03×** — the doc predicted "mild"; confirmed |

## Headline

| stage | ms / special-q |
|---|---:|
| transform + plattice (T) | **1.78** |
| fill, single-level 4 B (best) | **12.09** |
| **total, algebraic side** | **13.87** |
| *(both sides ≈ 2×, extrapolated)* | *≈ 28* |

Against the CPU lines derived from the GGNFS breakdown at `N_eff ≈ 13`:
**182 ms** to tie the box on sieve work, **56 ms** to reach the trial-division floor.
Apply, small-prime sieve, norm init and the threshold scan are **not yet measured**.

## Finding 1 — two-level fan-out loses, and the reason invalidates its premise

| variant | ms | L2 write sectors | sectors/record | DRAM % | occupancy | barrier stall |
|---|---:|---:|---:|---:|---:|---:|
| single-level atomic, 16K-way | **14.15** | 280,286,706 | 0.898 (**7.2× amplification**) | 17.7% | 85.7% | 0% |
| two-level L1 (128-way, staged) | 12.48 | 51,648,069 | 0.165 (1.3×) | 17.6% | 63.3% | 34.7% |
| two-level L2 (128-way, staged) | 25.38 | 51,025,495 | — | 17.1% | 64.9% | 35.3% |
| **two-level total** | **37.86** | | | | | |

Two-level does exactly what the design doc predicted at the transaction level:
it converts a 7.2× write amplification into 1.3×, a **5.5× reduction in L2 write
sectors**. And it still loses by 2.7×.

**All three kernels sit at the same ~17–18% of DRAM peak.** They are not
differentiated by efficiency — only by how many bytes they must move.
Single-level writes its output once (1.36 GB). Two-level writes it, reads it
back, and writes it again (3.75 GB). The time ratio 37.86/14.15 = 2.68×
is almost exactly the traffic ratio 3.75/1.36 = 2.76×.

**Why the premise fails on this card.** The doc argued: *"A single-level split
into ~16K buckets means ~16K simultaneously-open write streams; you cannot hold
16K partially-filled cache lines, so every record becomes a partial-line write."*
The first half is right — the amplification is real and measured. The second
half does not follow **because 16384 buckets × 128 B = 2 MB of open cache lines,
and this card has 48 MB of L2.** The L2 *is* the write-combining buffer. The
amplification happens between SM and L2 and is absorbed before DRAM, which is
why 280M write sectors and 51M write sectors produce identical DRAM throughput.

Two-level is spending 2.76× the DRAM traffic to solve a problem the L2 already
solves for free. Note per-pass it is genuinely better — L1 alone (12.48 ms)
beats single-level (14.15 ms) — but the second pass costs more than the win.

**This does not generalize.** It is a property of a 48 MB L2 against a 16K-way
fan-out. At I16e with 4× the regions, or on a card with a small L2, the open-line
footprint grows and the conclusion may flip. The rule to carry forward is
*compare open-line footprint to L2 size*, not "two-level wins."

## Finding 2 — record size: 4 B is optimal, and 2 B is worse

| record | ms | vs 4 B |
|---|---:|---|
| 2 B | 16.95 | **+40%** |
| **4 B** | **12.09** | — |
| 8 B | 14.09 | +17% |

The doc calls record size *"the single largest lever… bigger than bandwidth
efficiency"* and schedules 2 B as the measured upgrade. **It is a regression.**

8 B > 4 B is the expected byte-bound behaviour. 2 B > 4 B is not: the fill issues
the same 312M store instructions regardless of width, and a scattered 16-bit
store occupies a full sector transaction just as a 32-bit one does. Below 4 B you
stop saving bytes and only lose store efficiency. 4 B is the minimum record that
still uses a full-width store — which is also, not coincidentally, CADO's
`shorthint` size.

## Finding 3 — run-aggregation works, and is worth 2.3×

The doc's named lever: *"each (p,r) walk generates positions in monotone
increasing order… accumulate a run and emit it as one coalesced burst with a
single slot reservation."*

| L1 variant | ms | barrier stall |
|---|---:|---:|
| one record per thread per barrier | 26.00 | 30.4% |
| **runs of up to 8, one reservation each** | **11.19** | 34.7% |

**2.3×.** The first version was not memory-bound at all — it was barrier-bound at
7.5% of DRAM peak, which is why the profiler mattered more than the wall clock
here. Barrier stalls are still 35%, so more remains.

## Finding 4 — the root transform is a non-issue on GPU

**1.78 ms/special-q** for 6.84M modular inverses plus 6.84M FK reductions.

The CPU spends 373 ms/q on the same work (GGNFS "Sieve-Change", 12% of its wall
time). That is a **~200× speedup**, the largest single-stage win measured, and it
demotes the doc's "second per-q cost pillar" to 13% of the GPU sieve chain.

It is also *unoptimized*: `pl_transform` uses two 64-bit `%` operations and a
binary extended-Euclid inverse. The doc's recommended Hensel/REDC path should cut
it further. Not worth doing — it is already noise next to fill.

---

# Path 2 — apply kernel

**Date:** 2026-08-01. Same card, same factor base, same special-q.

> **Note on the record count.** Path 2 needed real norms, which forced the
> q-lattice reduction to become skew-aware (Finding 10). That changes the
> transformed roots, so the record count moved from 312,265,384 to
> **312,170,605** — 0.03%. Both are exact against their own CPU reference, and
> no Path 1 conclusion depends on the difference; fill volume is a property of
> the primes, and the transformed roots are uniform mod p either way. Path 1's
> timings above were not re-run.

The apply kernel owns one bucket region for its whole life: it initialises the
cells to the log-norm bound *in shared memory*, accumulates every log p that
landed in the region, scans for survivors, and writes back only the survivors.
The region never touches global memory in either direction. Apply's entire DRAM
footprint is the bucket read plus ~5 MB of survivors.

## Correctness

| gate | result |
|---|---|
| GPU records landed vs CPU reference, full I15e | **312,170,605 = 312,170,605** |
| Region 16384 (9,483 records) replayed on CPU: cells differing | **0** |
| Region 256 at I12/J512 (3,286 records): cells differing | **0** |

The cell gate covers the norm init, the shared-memory atomics, the log p
lookup and the threshold test together, against an independent CPU replay that
shares no device code.

## Headline

Best configuration: **region 2^14, 4 B records, 16-bit cells, 256 fill threads,
512 apply threads.**

| stage | ms / special-q |
|---|---:|
| transform + plattice (T) | 1.80 |
| fill, single-level 4 B | 12.30 |
| **apply (init + accumulate + scan)** | **3.08** |
| **total, algebraic side** | **17.4** |
| *(both sides ≈ 2×, extrapolated)* | *≈ 35* |

Against **182 ms** to tie the box and **56 ms** to reach the trial-division
floor. Still missing: the small-prime sieve (p < 2^15) and the rational side.

## Finding 5 — occupancy is the whole ballgame, and the ceiling table was right

| region | apply threads | smem | achieved occupancy | apply ms |
|---|---|---:|---:|---:|
| 2^15 | 512 | 64 KB | **32.4%** | 7.32 |
| 2^15 | 1024 | 64 KB | **64.2%** | 5.17 |
| **2^14** | **512** | **32 KB** | **95.7%** | **3.08** |
| 2^13 | 256 | 16 KB | — | 3.26 |

The doc's occupancy-ceiling table said a 64 KB region gets 1 block/SM, a block
caps at 1024 threads, the SM holds 1536, so 66.7% is a structural ceiling. The
measured numbers are 32.4% (512 threads × 1 block), 64.2% (1024 × 1) and 95.7%
(512 × 3). **A 2.4× swing in apply time from a table that could have been
written down before any code existed.**

Region 2^14 costs **+0.8 ms** in fill (more buckets) and saves **4.2 ms** in
apply. Take the trade. Region 2^13 gives back nothing in apply and costs 5.5 ms
in fill (65536 buckets), so 2^14 is the optimum, not a corner.

## Finding 6 — the safe cell scheme is also the fast one

| cells | region | apply ms |
|---|---|---:|
| **16-bit (safe)** | 2^14 | **3.077** |
| 8-bit (unsafe) | 2^15 | 3.047 |

Byte cells carry a real bug — accumulated logs do exceed 255 and `atomicAdd`
carries into the neighbouring position, which is why the doc rejected them.
Their only advantage is halving the shared memory a region needs. **That
advantage is worth 1%,** because shared memory is recoverable for free by
halving the region instead, and the region size is ours to choose. There is no
tradeoff here to agonise over.

Two related worries also priced out at nothing:

- **Shared-memory `atomicAdd` vs a racy plain `+=`: 3.08 vs 3.00 ms (~1%).**
  The "byte-atomic wrinkle" section spends a page resolving a problem that
  costs nothing to resolve correctly.
- **The sign convention works.** Atomics are add-only and a 16-bit half-word
  subtract would borrow into its neighbour, so cells start at `CINIT − T(x)`
  and *add*, with survivor ⟺ cell ≥ `CINIT`. Identical test to "start at the
  norm and subtract", no borrow, no CAS. Verified byte-exact above.

## Finding 7 — bank conflicts land inside the predicted band, and don't matter

| | shared wavefronts | bank conflicts | replay |
|---|---:|---:|---:|
| whole apply kernel | 66,515,509 | 25,831,468 | **1.63×** |
| init + scan alone (empty FB) | 17,486,146 | 20,839 | 1.001× |
| scatter (by difference) | 49,029,363 | 25,810,629 | ~2.5× |

The doc predicted *"roughly 1.5–2× on 32 banks. Acceptable, but confirm."*
Confirmed at **1.63×**. Init and scan are conflict-free by construction
(consecutive threads touch consecutive words); every conflict belongs to the
scatter, which makes two shared accesses per record — the log p lookup and the
atomic — for ~2.5× replay on its own.

It does not matter, because of the next finding.

## Finding 8 — apply is DRAM-bound at 58% of peak, a different regime from fill

| kernel | DRAM % of peak |
|---|---:|
| fill (all variants) | 17–18% |
| **apply** | **58.3%** |

Apply reads 1.25 GB of bucket records (312M × 4 B) in 3.08 ms = **406 GB/s**.
The floor — that same read at 100% of peak — is **1.86 ms**, so about 1.2 ms of
headroom remains and *no amount of shared-memory work can reach it*. Apply is
already within 1.7× of the only irreducible thing it does.

This also settles the record-size question from the other direction. A 2 B
record would halve apply's DRAM stream, but 2 B fill measured 40% slower
(Finding 2), and fill is 4× the cost of apply. 4 B stays.

## Finding 9 — norm init is cheaper than budgeted, and survivor emit is noise

| component | ms | how measured |
|---|---:|---|
| norm init, full fp32 degree-5 Horner + `__log2f` | **1.76** | `--norm horner` minus `--norm const` |
| accumulate + scan | ~5.2 | remainder at region 2^15 |
| survivor emit, 1.30M survivors | **0.41** | 1.30M vs 406 survivors |

The doc budgeted ~5 ms for norm init; the real per-position evaluation costs
**1.76 ms**. No piecewise approximation is needed on GPU — CADO's whole
piecewise-linear norm machinery exists to avoid a cost this card doesn't have.

## Finding 10 — the q-lattice must be reduced under the *skewed* norm

Not a performance result; a correctness one that Path 3 would have hit later
and harder. An unskewed Gauss reduction gives |a| ~ |b| ~ √q. The homogeneous
terms `c_k a^k b^(5−k)` then span **10³⁹**, `log2|F|` is set by `c₀b⁵` alone,
and the norms are simply wrong — the leading coefficient's term underflows fp32
entirely.

Reducing under `|(a,b)|² = (a/√s)² + (b√s)²` gives A = 2.33e12, B = 1.64e4 and
normalised coefficients `-0.0136, -0.0605, 1, -0.419, -0.0054, 0.381` — all
O(1). That balance is the entire purpose of the skew parameter, and it is also
what makes fp32 norm evaluation possible at all: without it there is no
scaling that keeps both the coefficients and the powers in float range.

---

# Small-prime sieve, rational side, and the CADO oracle

**Date:** 2026-08-01. `q = 120000053`, `rho = 112625526` — a **real** special-q
taken from las, not the synthetic one used above. Region 2^14, 4 B records,
16-bit cells, 128 fill threads, 512 apply threads.

## The whole sieve chain, both sides

| stage | side 1 (algebraic) | side 0 (rational) | both |
|---|---:|---:|---:|
| transform + plattice | 1.75 | 1.01 | 2.76 |
| bucket fill | 12.60 | 11.89 | 24.49 |
| apply — norm init + small sieve + bucket apply + scan | 11.95 | 7.91 | 19.86 |
| **total ms / special-q** | **26.30** | **20.81** | **47.11** |

**47.1 ms for the complete sieve, both sides**, against **182 ms** to tie the
CPU box and **56 ms** to reach its trial-division floor.

Correctness at full I15e, both sides: records landed equal the CPU reference
exactly (312,211,826 and 295,181,761), and a full region replayed independently
on the CPU gives **0 cells differing** on each side.

## Finding 11 — the small-prime sieve is 5× its budget, and it is now 28% of the chain

The doc budgeted *"small-prime sieve (~1.5e9 shared-mem updates) | ~1–3 ms"*.

| | side 1 | side 0 | both |
|---|---:|---:|---:|
| entries below `bkthresh` = 2^15 | 3,631 | 3,512 | 7,143 |
| updates | 2.957e9 | 1.398e9 | **4.36e9** |
| cost (apply with − without) | 8.77 ms | 4.40 ms | **13.17 ms** |
| rate | 337 G/s | 318 G/s | — |

**2.9× the updates and ~5× the time.** The update count is the surprise: those
7,143 entries are 0.1% of the factor base and produce **7.2× the entire
bucket-sieve volume**. On side 1, 84% of it comes from the 52 entries with
p < 64.

That skew is the whole engineering problem — a thread-per-prime loop would put
8,192 serial updates on whichever thread drew p = 2 while its neighbours did
one. Three tiers sized to each entry's hit count:

| tier | side-1 entries | threads per entry |
|---|---:|---|
| p < 64 | 52 (84% of updates) | the whole block |
| 64 ≤ p < 1024 | 165 | one warp |
| p ≥ 1024 (≤16 hits) | 3,414 | one thread |

At ~330 G updates/s against a card ceiling near 3.8 T conflict-free shared
atomics/s, this is ~9% of peak and there is real headroom left.

**Superseded by prime powers (finding 15).** At the time this was written fill
was the larger half (24.5 ms of 47.1) and the small sieve was not the place to
spend effort. Powers took the small sieve from 4.36e9 to 6.84e9 updates, and at
55.5 ms **apply (27.45) has overtaken fill (24.77) and the small-prime sieve is
now the largest single component of the chain.** If sieve milliseconds are ever
worth chasing again, this is where they are.

**Region choice is what makes this cheap.** With `log_region ≤ logI` a region
lies inside one j-row, so the entry point is one multiply and one remainder —
hits within a row are just the progression `i ≡ rt*j (mod p)`. No walk state
crosses regions, so every block is independent. CADO carries per-prime
positions forward between regions because it processes them in sequence; we
cannot, and do not need to.

## Finding 12 — the rational side is cheaper than the algebraic side

Side 0 costs **20.8 ms** against side 1's 26.3 ms, despite comparable bucket
volume (295.2M records vs 312.2M). The difference is entirely in the parts that
scale with the factor base and the polynomial: the transform is degree 1 and
every prime has exactly one root (1.01 vs 1.75 ms), and the small sieve carries
half the updates (1.40e9 vs 2.96e9).

Nothing on disk holds a rational factor base — GGNFS computes it on the fly and
CADO rebuilds it every run — so `rfb.c` builds it: sieve to `rlim`, then
`r = -Y0*Y1^-1 (mod p)`, with `p | Y1` as the projective case. Y0 and Y1 are 118
and 76 bits and are reduced by limb-wise Horner in base 2^32; **doubles are fine
for the norm (it only needs a logarithm) but give wrong roots.** Verified
against exact arithmetic: prime counts exact (1229 / 9592 / 78498 at 10^4 / 10^5
/ 10^6) and every root satisfies `Y1*r + Y0 ≡ 0 (mod p)`.

## Finding 13 — gated against CADO itself

`makefb` + `las -v -dumpfile` on the same polynomial. Full capture and
constants in `oracle/PARITY.md`.

| gate | las | ours | |
|---|---|---|---|
| q-lattice basis | `a0=7374527 b0=-1 a1=120000053 b1=0` | `(-7374527, 1, 120000053, 0)` | **same basis, first vector negated** |
| log2(maxnorm) side 0 | 131.86 | **131.86** | exact |
| log2(maxnorm) side 1 | 196.61 | 196.41 | 0.2 bits |
| one-sided survivors, side 0 | 23,952,829 | 30,043,786 | +25% |
| one-sided survivors, side 1 | 21,650,256 | 18,174,114 | −16% |

The survivor rates are close but not equal, and shouldn't be yet: we are missing
prime powers (CADO's small part has 3,839 ideals to our 3,631 — the gap is
exactly powers), and we run at scale 1.0 where las uses 1.28/1.93.

**We do not need las's `scale`.** It exists only so `scale × log2(maxnorm)` fits
in a byte — `1.28 × 196.61 = 251.7`. With 16-bit cells we can run at scale 1.0
or higher for free. las is discarding 0.39 bits of resolution per position that
we keep, which means fewer false survivors reaching cofactoring.

`-dumpfile` needed a one-line CADO patch to work at all: revision `0574bc39d`
has `ASSERT_ALWAYS(!las.dump_filename)` immediately before the only
`dumpfile.open()` call site in the tree. Original file and revert instructions
are in `oracle/`.

## Three bugs the oracle and the second side exposed

1. **`pl_transform` had CADO's field names on a differently-grouped struct.**
   las groups the basis by coordinate (`a = a0*i + a1*j`); `qlat_t` groups by
   vector (`a = i*a0 + j*b0`). Copying CADO's formula across silently swaps
   `a1` and `b0`, which **transposes the basis**. The result is a perfectly
   valid lattice of the right density — record counts, walk checks and every
   GPU/CPU cross-check still agree, because both sides used the same wrong
   lattice. It is simply not the lattice the norms describe. Nothing but a
   comparison against las would have caught it. `verify_transform()` now gates
   the transform against its definition (`a ≡ r*b mod p`) rather than against
   itself.

2. **The two-level fill livelocks past 16,384 regions.** Both levels stage into
   128 fixed shared buffers, so the split only fits when `nsuper ≤ 128` *and*
   `regions_per_super ≤ 128` — exactly I15e at region 2^15 and no further. At
   region 2^14 it indexed `cnt[]` out of bounds and spun forever in the retry
   loop. Path 1's two-level numbers were all taken at region 2^15, sitting
   exactly on that limit, so **Finding 1 stands**. But it sharpens the verdict:
   two-level cannot even express the operating point that won, without adding a
   third level. It now refuses the configuration instead of hanging.

3. **`log2(q)` was being divided out of both sides' norms.** Only the special-q
   side's norm carries a factor of q. On side 0 this made the threshold ~27 bits
   too generous — 171.7M survivors instead of 30.0M. Caught by comparing against
   las's per-side `log2(maxnorm)`, which is printed *after* the division on the
   sq side and *before* it on the other.

---

# Prime powers, las's log scale, and the attempt at a byte-diff

**Date:** 2026-08-01. Same q. The sieve now runs on **CADO's own factor base**
(`fb_cado.c` parses makefb's text format) at **las's own byte scale**, so both
the ideal set and the log units are las's rather than approximations of them.

## The chain with the correct factor base

| stage | side 1 (scale 1.28) | side 0 (scale 1.93) | both |
|---|---:|---:|---:|
| transform + plattice | 2.15 | 1.14 | 3.29 |
| bucket fill | 12.85 | 11.92 | 24.77 |
| apply (norm + small sieve + bucket apply + scan) | 17.82 | 9.63 | 27.45 |
| **ms / special-q** | **32.82** | **22.69** | **55.51** |

Up from 47.1 ms, entirely because prime powers add small-sieve work: side 1
goes 2.96e9 → **5.03e9** updates, side 0 1.40e9 → **1.81e9**. **6.84e9
small-sieve updates per special-q**, now half the chain. That puts the complete
two-sided sieve at **55.5 ms against the 56 ms trial-division floor**.

Correctness holds: full I15e, both sides, a region replayed independently on
the CPU gives **0 cells differing**.

## Finding 14 — factor base and threshold now match las exactly

| | las | ours |
|---|---:|---:|
| side 1 small part (p < 2^15) | 3,839 ideals | **3,839** |
| side 1 bucketed | 7,601,777 ideals | **7,601,776** |
| side 0 small part | 3,589 | 3,586 |
| side 0 total | 3,958,485 | 3,957,371 |
| side 1 survivor bound | 143 | **143** |
| side 0 survivor bound | 141 | **141** |

The bounds fall out of `round(scale * lambda * lpb)` with no fitting:
`1.28 × 3.5 × 32 = 143.4` and `1.93 × 2.35 × 31 = 140.6`.

Two things had to be right to get here. **`fb_log_delta`, not `log(q)`**: a
prime power's log increment is `fb_log(p^nexp) − fb_log(p^oldexp)`, the
*marginal* cost of the extra valuation, so that powers telescope with their
base prime. And **the general prime-power root transform** (below), without
which the powers cannot be walked at all.

## Finding 15 — prime powers break two assumptions the prime-only code made

Both were silent until CADO's factor base supplied moduli that GGNFS's
`.afb.0` never does.

1. **`pl_invmod` requires an odd modulus.** It is binary extended Euclid, whose
   halving step `(x+p)>>1` is only exact for odd `p`. CADO's factor base
   carries the ladder 2, 4, 8, …, 32768. Feeding one to `pl_invmod` does not
   return a wrong answer — **it never terminates**. Powers of two now go to a
   2-adic Newton iteration (`pl_invmod_any`).

2. **With a prime power the denominator can be non-zero and still have no
   inverse.** Modulo a prime, `D = a0 − r·a1` is either invertible or zero;
   modulo `p^k` it can be divisible by `p` but not by `p^k`, and binary Euclid
   spins forever on `gcd > 1`. This hung on the first `q = 49` in the file.

   The fix generalises and *subsumes* the old whole-rows special case. With
   `g = gcd(D, q)` (D and N cannot both be divisible by p, or p would divide
   the determinant, which is the special-q):

   > solutions exist only when `g | j`, and then, writing `j = g·j'`,
   > `i ≡ rt·j' (mod m)` with `m = q/g` and `rt = N·(D/g)^-1 (mod m)`.

   `g = 1` is the ordinary affine case; `g = q` gives `m = 1`, meaning "every
   `i`, on every `q`-th row" — exactly the old `PL_ROWS`, no longer special.
   The small sieve now also tiers on `m` rather than `q`, because an entry with
   `q = 32768, g = 32768` has `m = 1` and hits every position in its rows;
   leaving it in the thread-per-entry tier would hand one thread a whole region.

## Finding 16 — the sieve is validated against the factor base analytically

This turned out to be a stronger gate than the byte-diff, and it needs no
oracle file. Expected sieved log per position is `Σ logp(ideal) / q(ideal)`
over the whole factor base — computable directly from `c183.fb1`:

| | log units / position |
|---|---:|
| analytic, whole factor base | **54.65** |
| **measured, ours** | **54.2** |
| analytic, small part only | 39.31 |
| measured, ours with the bucketed part removed | 40.4 |

**0.8% agreement.** That exercises the factor-base parse, the power handling,
the general root transform, the small sieve, the bucket fill and apply, and the
`fb_log_delta` scaling, in one number, against a quantity derived independently
from CADO's own file.

## Finding 17 — the `-dumpfile` oracle is not usable as captured

The byte-diff does **not** pass, and the evidence says the problem is the dump,
not the sieve.

| | mean sieved log / position |
|---|---:|
| analytic, whole factor base | 54.65 |
| ours, full | 54.2 |
| analytic, small part only | 39.31 |
| **las's dump** | **41.1** |

las's dump carries roughly the small-sieve contribution and **not** the
bucket-sieved one, even though `las-process-bucket-region.cpp` calls
`apply_buckets()` before `SminusS()` and the dump.

Worse, it does not line up positionally at all:

| | correlation |
|---|---:|
| ours (small-only) vs las, direct | +0.033 |
| ours (small-only) vs las, i-mirrored | +0.016 |
| *control:* ours (small-only) vs ours (full), same positions | **+0.636** |

The control shows the measurement works. Both orientations were checked because
our basis is las's with the first vector negated, which mirrors the i-axis.
Neither correlates.

This is consistent with how the flag was found: `-dumpfile` sat behind an
`ASSERT_ALWAYS(!las.dump_filename)` placed immediately before the only
`dumpfile.open()` call site in the tree, i.e. it has been dead code for long
enough to rot. **Do not treat these two 512 MB files as ground truth.** Options,
in order of expected value: `las_tracek`/`TRACE_K`, which follows one `(i,j)`
through las's pipeline and would localise this in one run; or `las -v`
checksums; or reporting the bug upstream.

Everything else from `oracle/PARITY.md` — the basis, the scales, the bounds,
the factor-base composition — came from the `-v` log and remains sound.

# Review round: two placement bugs, and the gate that would have caught them

*2026-08-02. Two external reviews (fable's, in `prototype.md`; codex's, quoted
in the session log). Everything below is measured or verified on the CPU — the
GPU was busy, so no timing in this section is new.*

## Finding 18 — projective roots with a nonzero reciprocal were placed wrong

**The bug.** Both CADO and GGNFS store a root of `q` in `[0, 2q)`: below `q` it
is affine (`a ≡ r·b mod q`), at or above it is projective with reciprocal
`rr = r − q`, meaning `a·rr ≡ b (mod q)`. For a **prime** `q` the only
projective root is `rr = 0`, the classical `b ≡ 0 (mod q)` — so code that
assumes `rr = 0` is correct on a prime-only factor base and wrong the moment
prime powers arrive. `bench_kernels.cu` passed `0` where the reciprocal
belonged, and `plattice.cuh` only implemented the `rr = 0` case.

`c183.fb1` has **35 such entries** — the projective ladder above the leading
coefficient `110880 = 2^5·3^2·5·7·11`, opening with `4:4,3: 6` (q=4, rr=2).
Between them:

| | |
|---|---:|
| updates per special-q | **469,482,034** |
| scaled log units per cell (scale 1.28) | **1.538** |

**Why every existing gate was blind to it.** The wrong lattice has the *same
density* as the right one — both are index-`q` sublattices. Measured directly
for q=4, rr=2 over a small box: 32 true hits, 32 predicted hits, **16 in
common**. So:

- `Σ logp/q` (finding 16) checks density. Same density → passes.
- the CPU replay (`verify_apply_region`) shares `plattice.cuh` with the GPU, so
  it made the identical error → passes.
- record counts vs the CPU reference → same count → passes.
- `verify_transform` as it stood checked only that each *predicted* hit
  satisfies the congruence, one direction, and it never exercised a nonzero
  reciprocal at all → passes.

This is the same failure class as the transposed-basis bug: **right volume,
wrong placement.** It is the third one this project has hit, which is the real
lesson — volume gates cannot catch placement bugs, and this codebase kept
building volume gates.

**The fix.** `pl_transform_gen` now takes the reciprocal and uses
`D = rr·a0 − a1, N = b1 − rr·b0` for the projective case (at `rr = 0` that is
the negation of the old `(a1, −b1)`, same ratio — a generalisation, not a
replacement). `pl_transform_enc` is the single place the encoding is decoded,
and every consumer now goes through it.

## Finding 19 — bucketed projective entries were dropped, on a wrong argument

`fb_restrict` discarded every entry with `root ≥ p`. Side 0 has two:
**38321** and **5746453**, both factors of
`Y1 = 59·101·127·281·1259·38321·5746453`.

fable's review recommended *keeping* the drop, reasoning that a bucketed
`p > J = 16384` means a projective ideal can only hit row `j = 0`. **That
argument is wrong, and the way it is wrong is worth recording**: the projective
condition constrains `b`, not `j`, and `b = i·a1 + j·b1`. For a special-q of
this size the reduced basis has b-components of order `sqrt(q/skew) ≈ 1`, and
on this lattice it is exactly

```
(a0,a1,b0,b1) = (-7374527, 1, 120000053, 0)   ->   b = i
```

so the condition is `i ≡ 0 (mod p)` — one hit per row, on **every** row.
**16,384 positions per entry, not one row's worth.** `fbtest` prints the
`b(i,j)` form at startup for exactly this reason.

(A consequence worth its own line: `b = i` means the entire `i = 0` column has
`b = 0` and can never yield a relation. That is a property of this q-lattice,
not a bug, but it caps what these two entries are worth in practice.)

`fb_restrict` now keeps them and the transform handles the encoding. What the
bucket walk still cannot express is `g > 1` — hits confined to every `g`-th row
— so `k_transform` emits an empty walk for those and **accumulates the number
of positions dropped**, which is printed. At the default `bkthresh = I ≥ J` it
is exactly zero.

## Finding 20 — the projective power ladder does lift

`rfb.c` broke out of the power loop at `p | Y1` with the comment "no lift".
Projective roots *do* lift: the point `(Y0 : −Y1)` of `P^1` normalises to
reciprocal `rr = −Y1/Y0 (mod p^e)`, which is divisible by `p` but not `p^e`.
Restoring it adds `59^2, 101^2, 127^2` — **exactly the −3 in side 0's small-part
count against las** (3,586 vs 3,589), which fable predicted and which now
closes. Side 0 is 3,957,374 ideals, up 3.

The affine and projective ladders are now the same loop: which of `Y0`, `Y1` is
the unit mod `p` decides the encoding, and exactly one of them is, because
`gcd(Y0,Y1) = 1`.

## Finding 21 — the device transform could hang, and would have

Three latent faults in `k_transform`, all of which CADO's factor base reaches
and GGNFS's does not, all now removed by routing through `pl_transform_enc`:

| | consequence |
|---|---|
| `q = 2^15` sits exactly at the default `bkthresh`, so an **even** modulus reaches the device | `pl_invmod` is binary Euclid; it needs an odd modulus. Silent wrong root. |
| projective entries reaching the affine formula | `r ≡ 0` gives a bogus affine root. Silent. |
| raising `--maxbits` puts odd prime powers in the bucket range | non-invertible denominator → binary Euclid **spins forever on the device**. |

Two silent wrong answers and one hang — none of which any gate would have
reported as a failure.

`pl_transform_gen` has a genuine precondition, now documented and enforced:
**`q` must be a prime power.** The step "solutions exist only when `g | j`"
needs `gcd(N,g) = 1`, which follows from "`p` divides at most one of `D`, `N`"
and holds only for a single prime. For `q = 200` (`D` even, `N` divisible by 5)
the solution set is a CRT combination and this form is wrong — the new
brute-force gate found this immediately, which is how it came to be documented.
`fb_check_prime_powers` keeps the assumption true for any factor base loaded,
including hand-edited or third-party ones.

## Finding 22 — the gates now check placement, not volume

New CPU-only binary `fbtest` (`make check`), 6 seconds, **no GPU** — so it runs
on a busy box and on machines without the card:

| gate | what it catches |
|---|---|
| transform vs definition by **set equality**, every prime power `q ≤ 200` and every root in `[0,2q)` | placement. Both directions, so a proper sublattice fails too. Covers primes, powers, even moduli, affine, projective with zero and nonzero reciprocal. |
| the same, driven by the **real** factor bases | a loader that mangles the encoding, even where the algebra is right |
| every modulus is a prime power | the transform's precondition |
| `Σ logp/q` per side | parse, powers, log deltas, scale — one number |

The first gate is the one that matters. The old `verify_transform` compared
predicted hits against the congruence in **one direction only**; a transform
naming a proper sublattice — right congruence, half the hits — passed it. Set
equality against the definition, with `(a,b)` computed from `(i,j)` and no
lattice algebra involved, is what closes that. Both of this session's bugs fail
it loudly.

`verify_count_updates` also gained an optional per-region count array, so the
fill can be gated on **placement across all 32K regions** rather than on a
global total (fable's R4).

Result:

```
PASS  synthetic moduli              primes, prime powers, even moduli, affine + projective
PASS  side 1 small part             3839 entries checked
PASS  side 1 moduli are prime powers 7605616 ideals
PASS  side 1 projective entries seen 41 projective, 35 with a NONZERO reciprocal
PASS  side 0 small part             3589 entries checked
PASS  side 0 moduli are prime powers 3957374 ideals
PASS  side 0 projective ladder      10 projective, 3 with a NONZERO reciprocal
```

## Finding 23 — the 16-bit cell's precision was being thrown away

`k_apply` clamped the initialised norm at **255**. las clamps there because its
cell *is* a byte — that is the entire reason `scale` exists
(`1.28 × 196.61 = 251.7`). Ours is 16 bits with `CINIT = 4096`, so the ceiling
is `CINIT`, and `scale` is a free parameter rather than a constraint. The clamp
meant we inherited las's 0.39 bits of lost resolution for nothing, and would
have silently flattened every norm above `255/scale` into one bucket the moment
anyone raised `scale`.

Now clamped at `CINIT` (16-bit cells) or 255 (8-bit), with the two real limits
validated up front and refused rather than saturated:

| limit | binds at |
|---|---|
| `scale × log2(maxnorm) ≤ CINIT` | scale ≈ 20 (side 1) |
| `scale × log2(p) ≤ 255` (the uint8 per-ideal log) | **scale ≈ 9** |

So production scales of 2, 4, 8 are all available and free on the GPU side.
Whether they reduce false survivors enough to matter for CPU cofactor work is
**unmeasured** — it needs the GPU, and it is the cheapest remaining experiment.

## Also fixed

- **CLI defaults were the losing configuration.** `--mode twolevel --region 15`
  were still the defaults after two-level lost by 2.7× and 2^15 lost to 2^14,
  so several commands in this file reproduced a path nobody would ship.
  Defaults are now `atomic` / `region 14`.
- **Two silent parameter failures now refuse.** `--region > 16` with 2 B or 4 B
  records overflows the 16-bit offset field (wrong cells, no error);
  `--region > logI` breaks the fused small sieve's one-row assumption. Both
  produced plausible-looking output.
- `qlat_build` moved to `poly.c` so the correctness gates link without CUDA.

## Finding 24 — the makefb parser read 81 as 9², and it cost log accuracy

`fb_cado.c`'s `is_power` searched the exponent **upward** from 2 and returned
the first that worked. That is k *minimal*, i.e. the *largest* base — it read
`81` as `9^2` and `729` as `27^2`, despite its own comment saying "k > 1
maximal".

The exponents in the file are relative to the base prime (`81:5,4: 114` means
3^5 over 3^4), so a wrong base inflates the log increment:

| q | read as | logp | correct | logp |
|---|---|---:|---|---:|
| 16 | 4² | 3 | **2⁴** | **1** |
| 64 | 8² | 4 | **2⁶** | **1** |
| 81 | 9² | 4 | **3⁴** | **2** |
| 729 | 27² | 6 | **3⁶** | **2** |

It hits every q whose exponent is composite. Placement was unaffected, which is
why nothing before gate 5 saw it — and note it is the mirror image of findings
18–19: those were right-value/wrong-place, this is right-place/wrong-value.
Between them they cover both ways a sieve can be wrong while looking healthy.

Fixed by trial-dividing for the least prime factor, which is exact and, with 97
long-form lines in the file, free.

## Finding 25 — las's printed `scale` is rounded, and the bound is a truncation

Two constants taken from las's `-v` log were subtly wrong, and both were
recorded in `oracle/PARITY.md` and used by every run since.

**The scale.** las prints `scale=1.28`, rounded to 2 dp. It also prints
`logbase` to 7 figures, and `logbase = 2^(1/scale)`, so:

| side | printed | exact |
|---|---:|---:|
| 0 | 1.93 | **1.925** |
| 1 | 1.28 | **1.275** |

That is enough to move `fb_log` by one unit for a band of primes: p = 25811171
gives 32 at 1.28 and **31**, which is what las applies, at 1.275. This was the
last surviving discrepancy in the gate-5 trace.

**The bound.** We used `round(scale * lambda * lpb)`. las (`las-norms.cpp:270`)
uses

```
bound = (unsigned char) (r * scale + LOGNORM_GUARD_BITS);     r = lambda * lpb
```

— a **truncating** cast plus a guard bit. With the exact scales,
`trunc(3.5*32*1.275 + 1) = 143` and `trunc(2.35*31*1.925 + 1) = 141`, both
exact. Our old formula agreed on these two only because the rounded scales
happened to compensate; it would have diverged on any other parameter set. The
agreement reported in finding 14 was therefore luckier than it looked.

## Finding 26 — GATE 5 PASSED: byte-exact sieve parity against las

`las_tracek` is a **stock CADO target** (`sieve/CMakeLists.txt:111`,
`TRACE_K=1`) — no patch needed, unlike `-dumpfile`. It follows one position
through the pipeline and prints every ideal applied to it with its log, which is
strictly more informative than a byte dump.

Use `-traceab a,b`, not `-traceij`: it is basis-independent, so the i-mirroring
never enters. It also confirmed the mirroring directly — our `(4999,8192)` is
las's `(-4999,8192)` for the same `(a,b)`.

Four positions, both sides, `q=120000053, rho=112625526`:

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

**8 of 8 exact — and the ideal lists agree entry by entry, not just the
totals.** Two positions were chosen because long projective power ladders hit
them (3,9,27,81,243,729,2187 and 2,4,8,16,32,64), making this the direct gate on
the nonzero-reciprocal bug rather than an aggregate one.

> **Correction, later the same day.** As first written this table came from
> `fbtest --trace`, which enumerates the factor base directly with
> `hits_def_pub` and **does not run the sieve** — not `pl_transform_enc`, not
> the walk, not the tiering, the bucket fill, the small sieve or the GPU apply.
> So it established that our *factor base, roots and logs* agree with las, which
> is what found the bugs above, but it did not establish sieve parity, and
> calling it "byte-exact sieve parity" was an overclaim. Finding 27 closes the
> gap properly; the numbers below stand as the ideal/log half of it.

**Norm initialisation still differs, and should.** las runs 0 to +2 above the
exact value, scattered: one unit is `LOGNORM_GUARD_BITS` (which las prints in
its own banner), and the rest is its norm *approximation* — it interpolates
log|F| along a row rather than evaluating at every position, computing the exact
norm only later for survivors. **Our fp32 Horner is the more accurate of the
two.** The gap is bounded and one-directional (las over-estimates, so it is
slightly more permissive), which also means a byte-for-byte region diff could
never have reached zero even with a working dumpfile. Gate 5 was the right
instrument, not merely an available one.

Reproduce:

```
cd ~/cado-nfs/build/<host> && make las_tracek
cd ~/code/cuda-sieve/oracle && las_tracek -poly c183.poly -fb1 c183.fb1 \
    -lim0 67100000 -lim1 134200000 -lpb0 31 -lpb1 32 -mfb0 60 -mfb1 92 \
    -lambda0 2.35 -lambda1 3.5 -A 29 -sqside 1 -q0 120000011 -nq 1 \
    -adjust-strategy 0 -B 14 -t 1 -traceab 946175173703,4999

cd ~/code/cuda-sieve/bench && ./fbtest --cadofb ../oracle/c183.fb1 --trace 4999,8192
```

## Finding 27 — gate 5, done properly: the *pipeline* reproduces las

Finding 26's table compared las against a direct enumeration of the factor base.
The thing we actually want to gate is the sieve. `bench --probe i,j` now reads
the cell back out of `k_apply` after the run, so the number it prints has been
through the root transform, the Franke-Kleinjung walk, the three-tier split, the
bucket fill, the small-prime line sieve and the GPU apply:

```
[gate 5] probe (i=4999, j=8192)  x=268456839  region 16385 offset 4999
         init norm S   = 241
         final cell    = 3877
         SIEVED LOG SUM = 22   <- produced by transform + walk + fill + small sieve + apply
```

Same four positions, both sides, `q=120000053, rho=112625526`:

| (i,j) | side 1 las / GPU | side 0 las / GPU |
|---|---|---|
| (4999, 8192) | 22 / **22** | 58 / **58** |
| (9237, 15022) | 94 / **94** | 65 / **65** |
| (2978, 8393) | 46 / **46** | 94 / **94** |
| (13198, 9151) | 51 / **51** | 63 / **63** |

**8 of 8, through the whole pipeline.** *This* is what closes Path 3 gate 2.

The probe costs one comparison per cell against a sentinel and is compiled in
unconditionally; it is off unless `--probe` is passed.

## Finding 28 — fp32 norms are wrong near root lines, in both directions

The normalisation gives fp32 ample dynamic *range* — that is what finding 10
bought. It does not give precision. Near a real root line of `F` the Horner
terms are O(1) and cancel to something tiny, so what survives is mostly rounding
noise. Measured against an fp64 evaluation over a band along the three real root
lines:

| | before | after |
|---|---:|---:|
| positions sampled | 63,497 | 63,497 |
| rounded-log mismatches | **144** | **1** |
| error range (sieve units) | **−3.31 .. +2.57** | −0.00 |
| worst relative error on the sum | 2.59 | 4.8e-4 |

A random control over 200,000 positions shows 1 mismatch either way, so this is
specific to the root lines and not a general precision problem.

**Both signs occur**, so it costs relations as well as wasting cofactor time —
and raising `scale` (finding 23) amplifies it in sieve units, which makes it a
blocker for that experiment rather than an independent issue.

The fix is an error bound, not more precision everywhere: run the same Horner on
`|.|` alongside the real one, and when `|acc| < 4.9e-4 * sum|terms|` recompute
that cell in fp64 with `(a,b)` formed exactly in int64. The bound costs one extra
FMA chain per cell; the fp64 path fires on well under one cell in a thousand, so
even at 1/64 double rate it is noise. The one residual mismatch is a genuine tie
at the rounding boundary.

Credit where due: this was codex's finding, and my first attempt to reproduce it
sampled too sparsely and came back with 1 mismatch instead of 144. Sampling every
row at closest approach to each root line reproduced it exactly.

## Finding 29 — factor-base preprocessing cost 6.7 s, 120x the sieve chain

Self-inflicted, on 2026-08-02: `fb_restrict` and `fb_split_small` called
`fb_is_proper_power` per entry, which runs Miller-Rabin, over 11.5M ideals.

| stage | before | after |
|---|---:|---:|
| load | 0.42 s | 0.25 s |
| `fb_split_small` | **4.63 s** | **0.005 s** |
| `fb_restrict` | **2.05 s** | **0.008 s** |

Every loader already knows which entries are powers — `fb_cado.c` literally
separates them into their own stream before merging, and `rfb.c` generates them
in a loop with the exponent in hand. `fb_t` now carries an `ispow` flag set at
load, and nothing downstream re-derives it. The independent primality check
survives where it belongs, in `fbtest`'s prime-power gate, where being
independent is the point.

## Finding 30 — the gates are now assertions, not printouts

Three things that were computed and printed but could not fail a run:

- **Per-region record counts.** `verify_count_updates` gained the array in the
  last round but `bench` passed `NULL`. It now compares all 32,768 regions
  against the GPU cursors and **returns non-zero** on any mismatch. A global
  total cannot distinguish "right total, wrong region" — and every placement bug
  this project has hit had exactly the right total. Currently: *all 32,768
  regions match the CPU reference exactly*, both sides.
- **`sum(logp/q)`.** Now a PASS/FAIL in `make check`, against expectations
  computed by **separate implementations** (a re-parse of `c183.fb1` that
  re-derives base primes independently; a fresh sieve to `rlim` plus the power
  ladder, which also reproduces the 3,957,374 ideal count exactly). Wiring it up
  immediately caught that the constant I had was the stale pre-finding-24 value:
  42.744 against the correct **42.2913**.
- **Parameter validation.** `--record-bytes` outside {2,4,8} used to fall through
  to the 8-byte kernel while allocating the requested size — an out-of-bounds
  write. Two-level mode accepted 8-byte records but launches the 4-byte level-2
  specialisation. Both now refuse, along with zero/negative `J`, `reps`,
  `threads`, and an area that does not divide evenly into regions.

## Finding 31 — the parity profile and the timed profile are not the same set

Worth stating plainly, because it limits what the gate-5 numbers cover:

| | timed `bench` run | las |
|---|---|---|
| side 1 upper bound | truncated at the special-q (GGNFS convention), **6,840,490** bucketed | runs to `alim`, **7,601,777** |
| side 0 powers | capped at `--maxbits 15` | `powlim = ULONG_MAX` |

The eight probes still agree because no ideal in `[q, alim)` happens to hit those
four positions — which the agreement itself demonstrates, since the GPU excludes
that band and would otherwise have come out low. That is evidence for these
positions, **not** a general equivalence.

Before any survivor-set comparison, pin one profile. The cheap direction is to
pin las *down* to ours (`-powlim0 32767 -powlim1 32767`, and a `q0` above `alim`
so no truncation applies) rather than chase it up; `bench --fbbound 134200000`
goes the other way but then re-includes the special-q ideal, which las divides
out.

## Finding 32 — `--verify` could exit 0 on a sieve that was demonstrably wrong

Two independent false-success paths, both of which made the verification suite
decorative rather than load-bearing.

**Cell mismatches printed but did not fail.** `verify_apply_region` counted
differing cells, printed the count, and returned normally. Demonstrated with the
deliberately racy `--apply-mode plain`:

```
[verify] region 16384: 9576 records replayed on CPU, 518 cells differ
                       (first at cell 0: gpu 3946 ref 3966)
exit=255      <- was 0
```

20 sieve-log units lost at cell 0 alone, and the old code reported success.

**Bucket overflow was invisible to the per-region gate, by construction.**
`cursor[b]` is incremented *before* the cap test, so it counts records
**attempted**, not stored — and `verify_count_updates` counts the same thing.
The two therefore agree exactly while `k_apply` truncates each bucket at `cap`
and silently drops the excess. Finding 30's per-region assertion, added
specifically to catch placement errors, cannot see this class at all. Overflow is
now failed explicitly under `--verify`, and the per-region OK line says what its
counts do and do not cover.

The general lesson is the one this project keeps re-learning: a gate that cannot
fail is not a gate. Findings 30 and 32 are the same mistake in two forms —
computing the right comparison and then not acting on it.

## Finding 33 — the CPU reference was not evaluating the kernel's expression

`k_apply` used `__log2f`, the fast intrinsic (~2 ulp, no host equivalent);
`norm_target_host` kept full `double` through `log2` with a `1e-300` clamp
against the device's `1e-30f`. So "0 cells differ" was comparing two slightly
different functions, and an emulation over 127k root-line samples found three
rounded disagreements between the two expressions.

The device now uses accurate `log2f` and the host mirrors it float-for-float,
including the clamp. Norm init was 1.76 ms of a 27 ms apply, so the intrinsic was
not worth the loss of a meaningful reference. **The reference must be the same
function, not a better one** — being more accurate on the host would make the
comparison meaningless in exactly the same way.

Re-measured after the change: still 1 residual mismatch in the 63,497-position
root-line band (the rounding tie from finding 28), and region 16384 still gives
0 cells differ — but now that number is a statement about the device path.

## Finding 34 — the default invocation never got finding 29's fix

`fb_load` (GGNFS `.afb.0`, the **default** `--fb` path) left `ispow` NULL, so
`FB_ISPOW` fell back to a primality test per entry — the exact 7.8 s that
finding 29 removed, still present on the one path not exercised while fixing it.

| stage | before | after |
|---|---:|---:|
| `fb_split_small` | 5.39 s | **0.069 s** |
| `fb_restrict` | 2.39 s | **0.024 s** |

`.afb.0` contains no prime powers at all — that is one of the two reasons the
CADO loader exists — so the fix is a zeroed flag array, stating that fact rather
than rediscovering it 7.6M times.

## Finding 35 — three more parameter combinations that failed silently

- **`--apply-threads` was never validated** (only `--threads` was). The small
  sieve strides its warp tier by `nwarps = threads >> 5`: below 32 that is
  **zero — an infinite loop on the device**, and a non-multiple of 32 leaves a
  partial warp whose lanes re-run warp-tier entries and double-add their logs.
  Now requires 0 or a multiple of 32 in [32,1024].
- **`1u << log_region` happened before `log_region` was bounded**; UBSan flags
  `--region 32`. Both `logI` and `log_region` are now bounded before anything
  shifts by them.
- **Probe coordinates were unchecked**, so `--probe 16384,0` aliased the real
  cell `(-16384,1)` and would have certified a position nobody asked about —
  the worst possible failure mode for a parity instrument.

## Not addressed in this round

- **The survivor-set gate.** Gate 5 is *sampled* pipeline parity: four
  positions, and on profiles that differ (finding 31). The decisive test is a
  profile-pinned survivor-set comparison, or failing that per-region **offset
  hashes** rather than counts — counts cannot see a permutation within a region.
  This is the next correctness step and it is not done.
- **The hybrid norm kernel has not been timed on a quiet GPU.** The extra
  absolute-value FMA chain runs on every cell and the fp64 branch diverges;
  calling that "noise" (finding 28) is reasoning, not measurement.
- **Timings in this session are not comparable.** The GPU was shared with an
  ECM job throughout; every millisecond figure printed on 2026-08-02 is inflated
  and none should be quoted. Correctness results are unaffected.
- **`--maxbits > 15` is now safe but untested**: the transform handles bucketed
  odd prime powers, and `g > 1` losses are reported, but no run has exercised
  it. The reported-loss counter is the thing to watch when someone does.
- The `g > 1` bucketed case emits nothing rather than routing to the small
  tier. Correct-and-reported, not correct-and-complete. Zero at the default
  `bkthresh`.
- Everything under "What is not yet measured" below, unchanged.

## What is not yet measured

**This is a sieve measurement, not a relation-collection measurement, and the
distinction is load-bearing.** What runs end-to-end is: transform → fill →
apply → threshold → survivor list, per side. What does not exist at all is the
two-sided survivor intersection, resieve, factor recovery, host transfer, the
cofactor feed, and unique-relation accounting. The survivor *list* is also
capped at 2^22 entries against one-sided sets of 18–30M — the count is exact
and truncation is reported, but nothing consumes the list yet.

So: **kernel feasibility is demonstrated; relation-collection feasibility is
not.** Any "3–4× whole-box speedup" is a projection from sieve-q throughput
with the cofactor path assumed unchanged, not a measured relation rate. It
should be stated that way everywhere it appears.

The comparison constants are also still assumptions, and every graded claim
inherits their error bars:

| constant | status |
|---|---|
| GGNFS `N_eff` at 14–16 workers | **unmeasured** — needs the quiet box. It is the divisor in every target number. |
| GPU watts during the chain | **never captured.** The Gate-1 metric is updates/sec/**joule** and no joules exist. ~5 s of looping plus `nvidia-smi power.draw` sampling. |
| CADO `f ≈ 0.23` (Gate 0) | assumed to generalise from GGNFS |
| the "200× root-transform speedup" (finding 4) | GGNFS's Sieve-Change timer also covers small-sieve setup, transformed-polynomial work and report-bound setup, so this is an **upper bound**, not a like-for-like stage comparison. |

Also unmeasured: `bkthresh` sweep, I16e slabbing, throughput mode, production
scales 2/4/8 (finding 23), and the per-q host-side small-FB transform and sort
(`bench_kernels.cu`), which runs outside every timed number.

Byte-exact parity is blocked on a trustworthy oracle, not on our side of the
comparison.

**Caveats on the numbers above.** *(Written against the Path-2 configuration;
the sections from "Small-prime sieve, rational side" onward run both sides on
the real `q=120000053, rho=112625526`, so "one side only, synthetic rho" no
longer applies to those. It still applies to findings 1–10.)* The 1-in-413
survivor rate is one-sided with a generous 112-bit allowance and is *not* a
claim about las's survivor count — the real rate comes from intersecting both
sides, which is not implemented.

## Reproduce

**The 55.5 ms configuration of record** — this is the command every number
downstream quotes, and the one the CLI defaults now reproduce:

```
cd bench && make
./bench --cadofb ../oracle/c183.fb1 --side 1 --scale 1.28 \
        --q 120000053 --rho 112625526 --allowance 112     # 32.8 ms
./bench --side 0 --scale 1.93 \
        --q 120000053 --rho 112625526 --allowance 72.85   # 22.7 ms
```

Defaults are `--mode atomic --record-bytes 4 --region 14 --apply-threads 512`
as of 2026-08-02; before that they were `twolevel` and `--region 15`, so
commands below that omit those flags reproduced a path that had already lost.

**CPU-only gates** (no GPU, safe to run on a busy box):

```
make check                                       # == ./fbtest --cadofb ../oracle/c183.fb1
```

```
./bench --verify --logI 12 --J 512 --region 12   # correctness vs CPU reference
./bench --verify --region 14 --mode atomic --record-bytes 4 --apply-threads 512
./bench --mode atomic --record-bytes 4 --region 14 --apply-threads 512   # best
./bench --mode twolevel                          # the fill variant that loses
./bench --region 14 --norm const                 # isolates norm-init cost
./bench --region 14 --apply-mode plain           # isolates smem atomic cost
./bench --region 15 --cells 8                    # prices the unsafe byte cell
```
