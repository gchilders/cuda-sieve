# Path 1 — bucket-fill microbenchmark results

> **This is a lab notebook, in discovery order, including refuted findings.**
> For what the siever does *today* — architecture, validated jobs and cards,
> measured vs projected, open experiments — read **`STATUS.md`** instead.

**Hardware:** RTX 5070 (sm_120, 48 SM, 48 MB L2, 99 KB opt-in smem, 672.0 GB/s)
**Toolchain:** CUDA 13.2.78; sm_120 + sm_89 + sm_86 native, compute_80 PTX (finding 49)
**Also measured:** RTX 3090 (sm_86), A100 80GB PCIe (sm_80) — finding 50
**Date:** 2026-08-01
**Input:** `oracle/input.job.afb.0` — the real C183 algebraic factor base, no synthetic data
**Config unless stated:** I15e (`logI=15`, `J=16384`, A=5.369e8), region 2^15, q=120000011,
FB truncated at q (GGNFS convention) → 6,843,511 entries, **312,265,384 records/special-q**

## Architecture scope

The project's **primary goal is GPU-resident relation collection**, not a
permanent GPU-sieve + CPU-cofactor split. These results first isolate the sieve
because it is the largest regular stage. A completed primary path must also put
survivor intersection/compaction, primitive filtering, resieve/factor recovery,
trial division, and cofactorization on the GPU wherever practical, so sustained
host compute does not scale roughly one-for-one with GPU count.

A hybrid that leaves GGNFS/CADO factor recovery and cofactoring on the CPU is a
useful secondary deployment option, especially for the measured 9800X3D + RTX
5070. Any hybrid throughput or energy number below is labelled as a projection
and does not set the primary roadmap.

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

Against the CPU lines from the GGNFS breakdown. **Superseded 2026-08-03**: at
the *measured* `N_eff = 10.24` (finding 43) these become **~225 ms** to tie the
box on sieve work and **~70 ms** for the optional hybrid's retained CPU TD
stage. The 182/56
figures below assumed `N_eff ≈ 13` and are too generous to the CPU.
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
*(both superseded — measured N_eff gives ~225 / ~71 ms, finding 43, and the
~71 ms is a hybrid's retained stage, not a floor for the GPU-resident target)*
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

**47.1 ms for the complete sieve, both sides**, against the then-used **182 ms**
CPU sieve-work line and **56 ms** optional-hybrid retained-TD line.

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
two-sided sieve at **55.5 ms against the then-used 56 ms hybrid retained-TD
stage**.

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

## Finding 36 — the GPU half of Gate 1 now has joules; the CPU half cannot be measured on this box

**Superseded by finding 44 for the Gate-1 comparison.** WSL2 still cannot read
CPU energy directly, but a later same-session HWiNFO64 log measured CPU PPT,
GPU board power and both DIMM PMICs from Windows while the real CPU and GPU
workloads ran. The GPU-only measurements below remain valid historical data.

Gate 1's metric is updates/sec/**joule** and no joules existed. Board power
sampled at 10 Hz (`nvidia-smi --query-gpu=power.draw,utilization.gpu
--loop-ms=100`) across a 400-rep run of the chain, 2026-08-03:

| | chain ms/q | busy board power | J/q |
|---|---:|---:|---:|
| side 1 | 37.97 | 202.0 W (peak 222.7) | 7.67 |
| side 0 | 26.11 | 188.9 W (peak 220.3) | 4.93 |
| **both sides** | **64.08** | | **12.60** |

Idle is 27.65 W (108 samples, nothing on the card). 147 of 160 side-1 samples
sat at 100% utilisation, so this is a clean plateau, not an average over gaps.
The card's cap is 250 W and the chain draws 76–81% of it: this is a
power-limited, fully-occupied kernel with no headroom story to tell.

Three caveats, all of which push the same way:

- `power.draw` is **board** power, not wall. A 90%-efficient PSU puts this
  nearer **14 J/q** at the plug, and that is the number a relations/watt claim
  has to use.
- The box was running 14 `gnfs-lasieve4I1` workers throughout, so the
  milliseconds are contended and J/q is *over*-stated. The wattage itself is a
  device property and is unaffected.
- A GPU-sieving box still pays for its CPU and its idle draw. Per-device joules
  is the wrong comparison; whole-box wall watts in each configuration is the
  right one.

**The CPU side cannot be measured from inside WSL2.** There is no
`/sys/class/powercap` (no RAPL) and no `/dev/cpu/*/msr`, so package energy for
the 9800X3D is not readable here at all. It needs either a Windows-side tool
(HWiNFO64, LibreHardwareMonitor) or a wall meter. **A wall meter is the better
instrument** — Gate 1's real question is relations/sec/watt for the whole box in
each configuration, and a plug measures exactly that with no attribution
argument. Two readings settle it: the box sieving on CPU at 14 workers, and the
box sieving on GPU. Until one of those exists, Gate 1 has a numerator and no
denominator, and no relations/watt claim should be made.

## Finding 37 — the hybrid norm costs 1.4%; fp64-everywhere would have cost 2.7x

Finding 28 called the fp64 fallback "noise" on reasoning alone, and the
not-addressed list flagged that as reasoning rather than measurement. Measured
now. `NORM_CANCEL_TOL` became a compile-time override (`make
DEFS=-DNORM_CANCEL_TOL=...`) purely for this; the default is unchanged and
neither override is a correct setting. Norm-init cost is
apply(`--norm horner`) − apply(`--norm const`), side 1, 300 reps:

| build | apply ms | norm init | chain ms |
|---|---:|---:|---:|
| `--norm const` (no init at all) | 16.94 | — | 30.91 |
| tol = 0 — fp32 only, fallback dead | 23.32 | 6.34 | 37.21 |
| **tol = 4.9e-4 — default hybrid** | **23.82** | **6.89** | **37.87** |
| tol = 1 — every cell through fp64 | 86.87 | 69.94 | 100.82 |

The guard plus the fallback cost **0.55 ms on a 37.9 ms chain — 1.4%**. That is
an upper bound on the fallback alone, since nvcc may fold the now-dead `|·|`
error-bound chain out of the tol=0 build.

The interesting number is the last row. Fixing finding 28 the obvious way —
evaluate every norm in fp64 — costs **63 ms more per q** and takes the chain
from 37.9 ms to 100.8 ms. Against the then-used 182 ms tie-point (now ~225 ms,
finding 43) that is 1.8x instead of
4.8x. Consumer Blackwell runs fp64 at 1/64 rate so the cliff is expected; what
is worth recording is that the accuracy repair the probe needed turned out to
be nearly free, while the naive version of the same repair would have eaten
most of the margin the probe exists to establish.

## Finding 38 — RETRACTED: there is no measured survivor-count discrepancy

**Originally claimed** that our one-sided survivor counts (33.7M side 1, 32.8M
side 0) ran ~1.5x above "las's" 21,650,256 and 23,952,829. Those two numbers
came from `dumpcmp stat` on the `-dumpfile` capture — the same capture
**finding 17 in this document already ruled unusable**, because the dump
carries the small-sieve contribution and *not* the bucket-sieved one. Missing
bucket logs means less is subtracted, S is too high, and the survivor count
from that file is too low by construction. It was never las's survivor count.
The claim is withdrawn; no comparison was actually made.

Two things worth keeping from the attempt:

**The dump was re-confirmed dead, this time positionally.** Running side 1
pinned (`--scale 1.275 --allowance 112 --fbbound 134200000`) and dumping all
536,870,912 positions gives a `dumpcmp diff` delta histogram that is *flat* —
about 4.5M positions at every delta from -8 to +8, 86% of positions outside
that window entirely. Two arrays that share no position-level structure. Our
side of it is sound: at the four gate-5 positions our dump reads exactly
`init - sieved sum` = 219, 153, 195, 195 at `x = j*I + i + I/2`. las's file
reads none of those under either i orientation.

**Exact scale pins the bound with no fitting.** At the documented `scale =
1.275` (not the printed 1.28) the rule `(unsigned char)(scale * lambda * lpb) +
LOGNORM_GUARD_BITS` gives `(uchar)(142.8) + 1 = 143` — **las's bound exactly**.
At 1.28 it gives 144. Finding 25's exact scales and finding 31's bound rule
agree only at the exact value, which is a second independent check on both.

The real lesson is procedural: this document contained the refutation (finding
17) about 600 lines above the claim, and I quoted the discredited numbers
anyway. Numbers from a source marked unusable have to be deleted, not left
lying in a table where they read as data.

## Finding 39 — the only las survivor number that exists is two-sided

Chasing finding 38 turned up what the oracle can and cannot support. las's `-v`
log reports:

```
# survivors before_sieve: 536870912
# survivors after_sieve: 797028 (ratio 1.48e-03)
# survivors trial_divided_on_side[0]: 33355
# survivors enter_cofactoring: 1851
# survivors smooth: 37          -> 37 relations
```

`after_sieve: 797028` is **after intersecting both sides**. las never prints a
one-sided count, and the byte dump that would have given us one is broken. So
**there is no way to gate our one-sided survivor rate against las at all** —
not with a better comparison, but at all. The gate that does exist is the
two-sided one, and it requires the intersection we have not built.

That collapses three open items into one deliverable. The two-sided
intersection is simultaneously the missing pipeline stage, the only available
survivor parity gate (target: **797,028**), and the input to every
cofactor-cost projection. It should be built next.

Note also the funnel las reports: 797,028 survivors -> 1,851 entering
cofactoring -> 37 relations. Two orders of magnitude are removed by trial
division and leftover-norm checks *before* cofactoring, which is the part of
the CPU cost this probe keeps assuming is unchanged.

## Finding 40 — the two-sided survivor gate PASSES to within 0.25%

`bench --survbits FILE` writes a 1-bit-per-position survivor bitmap (64 MB);
`dumpcmp and A B logI` intersects two of them and applies the primitive-point
filter. Both sides pinned to las:

| | ours | las | |
|---|---:|---:|---|
| side-1 bound | **143** | 143 | exact |
| side-0 bound | **141** | 141 | exact |
| side-1 one-sided | 14,888,741 | 14,139,941 | +5.3% |
| side-0 one-sided | 18,936,923 | 18,663,976 | +1.5% |
| **two-sided** | **841,418** | **797,028** | **+5.6%** |

Both bounds land on las with no fitting, from `(unsigned char)(scale * lambda *
lpb) + LOGNORM_GUARD_BITS` at the **exact** scales 1.275/1.925 — 1.28 would
give 144. A second independent confirmation of findings 25 and 31.

Tightening only the side-1 bound by the already-understood one-unit norm
difference (`--allowance 111` -> bound 142) gives **795,037 against 797,028,
-0.25%**, and side 1 one-sided 14,062,732 against 14,139,941, -0.55%. That
one unit is diagnostic only — it must **not** be hardcoded, it is las's
approximation and ours is the more accurate side (PARITY.md).

las never prints a one-sided count; those two las numbers came from opening the
opposite side's bound to 253, which excludes only the 0.4% of positions whose
byte saturates. The two-sided figure is las's own `survivors after_sieve`
straight from its log, so this gate does not touch the broken `-dumpfile`.

**Bonus result, and it matters more than the parity.** las returns **37
relations at bound, bound+6 and bound+12** (`after_sieve` 797,028 -> 1,408,504
-> 2,438,956). Tripling the survivor count yields zero extra relations, so
every survivor past the tuned bound is pure cofactorisation waste. Survivor
*rate* is therefore a first-order cost input, not a detail.

## Finding 41 — CORRECTED: the "1.74x failure" was comparing two different populations

This finding first reported 1,386,939 against 797,028 and called the gate
failed. It was wrong, and the error was mine in a way worth recording.

**CADO's `after_sieve` counts PRIMITIVE points**: it has already dropped every
`(i,j)` with `gcd(i,j) > 1`, not merely the both-even ones. Our filter
(`--not-both-even`, `bench_kernels.cu`) removed only both-even. Applying the
right filter moves our two-sided count 1,386,939 -> **841,418** and closes
essentially the whole gap.

This convention is written down in `prototype.md`, in a list headed *"Known
parity gotchas, collected so nobody rediscovers them"*: "skip positions with
`gcd(i,j) > 1` conventions (`unsievethresh`)". I rediscovered it, from the
wrong end, after a day of measurement.

Two chains of reasoning built on the bad comparison, both now void:

- **The "clean six-unit shift on both sides".** Real, reproducible, and an
  artifact — two populations differing by ~35% of positions produce a smooth
  CDF offset that looks exactly like a calibration error. It reconciled with
  las at three bound settings across a 3x range of counts, which is precisely
  why it was convincing. *A consistent-looking offset between aggregates is not
  evidence of a constant; it is evidence the aggregates are comparable, which
  was the untested assumption.*
- **The p=2 ladder suspicion.** The "wrong sign" on both-even
  over-representation (33.7% of our survivors against a 25% baseline) is not
  anomalous at all. Removing the common power of two from a nonprimitive point
  leaves the cofactor of a *smaller* primitive point, so nonprimitive points
  are unusually likely to pass until they are explicitly filtered. Expected
  behaviour, not a bug.

The fifteen `las_tracek` traces were right all along: las is **+1**, and only
+1, everywhere. Per-position parity was never in question — the aggregate was
counting a different set. When per-position evidence and aggregate counts
disagree, the population definition is the thing to check first, before any
hypothesis about the arithmetic.

**Implementation note.** The right shape is the one now in `dumpcmp`: intersect
the two device bitmaps first, then compact and gcd-filter the ~1.4M
intersections — **not** 537M gcds over the whole area. `--not-both-even` stays
as the device-side pre-filter because parity is nearly free in the kernel
(`x`'s low bit and `x >> logI`'s low bit) while a gcd is not; the full
primitive-point test belongs on the compacted host-side set.

Still open: the exact survivor-*set* comparison. Matching counts to 0.25% is
not the same as matching membership, and nothing here has compared the sets
element by element.

## Finding 42 — a clean single-thread las number, on the parity profile only

With the CPU actually free (the GGNFS job finished), 31 special-q from
q=120000053, same flags as the parity capture minus `-dumpfile`:

```
# Total elapsed time 321.53s, per special-q 10.372, per relation 0.218581
# Total 1471 reports [0.182s/r, 47.5r/sq] in 322 elapsed s [83.1% CPU]
```

**10.37 s per special-q, 47.5 relations per special-q, single-threaded.**
Against our 66.9 ms for the two-sided sieve chain (39.0 + 27.9).

**This is not a baseline and must not be quoted as one.** `-B 14` forces las
into 2^14 bucket regions to match our region size, and `-adjust-strategy 0`
pins I and J — both are parity settings chosen to make the *comparison* valid,
and both cost las performance. `-t 1` is one thread of sixteen. The honest
CADO Gate 0 number needs production flags and the full box. At the time of
this parity run GGNFS `N_eff` was still unmeasured; finding 43 immediately
below measures it and supersedes the old 182 ms tie-point.

What it does establish: 47.5 relations per special-q, and the funnel
797,028 survivors -> 1,851 into cofactoring -> 37 smooth at the parity q.

## Finding 43 — N_eff measured at last: 10.24, not the assumed 13

Measured by codex on the quiet box, GGNFS `gnfs-lasieve4I15e` scaling sweep:

| workers | steady q/s | N_eff |
|---|---:|---:|
| 1 | 0.331 | 1.00 |
| 8 (physical cores) | 2.120 | 6.40 |
| 14 | 2.974 | 8.98 |
| **16** | **3.392** | **10.24** |

Sixteen workers win — SMT is worth 1.6x over the eight physical cores, which is
why 14 was leaving throughput on the table. `N_eff` is the divisor in every
target number in this document and it had never been measured; the assumed 13
was optimistic.

**Every guidepost moves, and against us.** Production-equivalent CPU time is
**~295 ms/q, not the assumed 238 ms**. The two lines this document has quoted
since the first page become approximately:

| | old (assumed N_eff 13) | **measured (N_eff 10.24)** |
|---|---:|---:|
| tie the box on sieve work | 182 ms | **~225 ms** |
| hybrid retained TD stage (if kept on CPU) | 56 ms | **~70 ms** |

Quiet-machine GPU timing, same session: side 1 38.056 ms, side 0 26.166 ms,
**64.2 ms/q** on the equal-work profile (65.0 ms on full-alim parity).

So the GPU sieve sits just below the **hybrid's retained CPU TD stage** — 64.2
ms against ~70 ms. That makes the 9800X3D + RTX 5070 a promising balanced
hybrid pair, but it does not create an irreducible project floor. Everything
that remains unbuilt—device intersection/compaction, resieve, factor recovery,
GPU cofactorization, and relation output—now determines the primary
GPU-resident result. Pause sieve-kernel optimization until those stages reveal
the actual critical path; do not infer that sieve milliseconds have no value
in an all-GPU or multi-GPU design.

## Finding 44 — Windows-side power closes the component-energy proxy

HWiNFO64 logged the Windows sensors every two seconds while codex drove the
actual workloads from WSL2. The saved run has 466 usable samples over 931 s.
The raw local capture is
`C:\Users\Kyle\OneDrive\Documents\loads.CSV` (not committed to this repo).
The comparison uses

```
P_proxy = CPU PPT + NVIDIA GPU Power + DIMM0 Total Power + DIMM1 Total Power
```

so both configurations pay for the CPU socket, the discrete GPU board and the
two DDR5 modules. It is a substantially better comparison than finding 36's
GPU-only joules, but it is still a **component proxy, not wall power**: it omits
the motherboard/chipset, storage, fans, VRM losses and PSU conversion losses.

Same-session idle and the full 16-worker CPU plateau:

| configuration | CPU PPT | GPU board | two DIMMs | **component proxy** |
|---|---:|---:|---:|---:|
| idle (45 samples) | 37.637 W | 28.669 W | 2.006 W | **68.311 W** |
| 16-worker GGNFS (27 samples, before the first worker exited) | 125.073 W | 28.436 W | 5.222 W | **158.731 W** |

The power batch covered 448 special-q and 20,445 relations. Summing each
persistent worker's steady rate gives **3.55825 q/s** (45.64 relations/q), or
281.0 ms of whole-box time per q. That makes the CPU configuration
**44.609 J/q** on the component proxy, or **25.411 J/q above idle**. This
short power batch does not replace finding 43's broader `N_eff` sweep; it is
the simultaneous throughput denominator for this power capture.

The benchmark times transform, fill and apply in separate repetition blocks,
and the HWiNFO trace resolves the resulting plateaus. Weighting each power
plateau by that stage's measured time is required; a single sample or a simple
mean over the invocation is wrong:

| stage | algebraic proxy | rational proxy |
|---|---:|---:|
| transform + plattice | 221.213 W | 228.283 W |
| fill | 237.330 W | 236.125 W |
| apply + norm + small sieve + scan | 281.775 W | 282.384 W |

| side | chain ms/q | stage-weighted proxy | proxy J/q | J/q above idle |
|---|---:|---:|---:|---:|
| algebraic | 38.177 | 264.183 W | 10.086 | 7.478 |
| rational | 26.194 | 259.172 W | 6.789 | 4.999 |
| **both sides** | **64.371** | **262.144 W** | **16.874** | **12.477** |

Of the two-sided 16.874 J/q, 13.306 J is GPU-board energy, 3.440 J is CPU
package energy and about 0.129 J is DIMM energy. Independent `nvidia-smi`
spot checks agreed with the HWiNFO watt trace: 227.16 W during algebraic apply
and 179.41 W during rational fill. Several unrelated HWiNFO utilisation and
temperature columns were frozen and were not used. The earlier standalone
`idle.CSV` reported about 14 W for the GPU while NVML read about 29 W, so it was
excluded; all baselines and deltas here come from the self-consistent load log,
whose idle GPU value (~28.7 W) agrees with NVML.

For the **measured portions only**, the GPU sieve requires 37.8% as much total
proxy energy per q as the complete CPU GGNFS q (a 2.64x advantage), or 49.1%
as much energy above idle (a 2.04x advantage). That is encouraging, but it is
deliberately **not an end-to-end perf/watt claim**: the CPU number includes its
complete q, while the GPU number stops after the two sieve sides. Device
intersection, primitive filtering, transfer, resieve/factor recovery, the
TD/cofactor feed and unique-relation accounting must be measured before Gate 2
can close. A wall meter is still the final instrument for economics, but CPU
power is no longer the binding unknown in Gate 1.

## Finding 45 — secondary hybrid whole-box projection

Finding 44 compares the GPU *sieve* against a complete CPU *q*, and says so.
This section asks a useful but secondary deployment question: what would the
measured 9800X3D + RTX 5070 do if the GPU sieved while GGNFS's factor recovery,
trial division, and cofactoring remained on the CPU? It is still a projection—the
glue is unbuilt—but its arithmetic inputs come from measurements rather than
the earlier planning assumptions. It is **not** the primary GPU-resident result.

**Cross-check first.** The GGNFS breakdown extrapolated to the measured
`N_eff = 10.24` predicts 302.1 ms/q. Directly measured: 294.8 ms/q (finding 43
sweep) and 281.0 ms/q (finding 44 power batch). Within 5–7%, by two independent
routes. For this hybrid split, **231.1 ms/q moves to the GPU and 71.1 ms/q is
retained on the CPU** for trial division and cofactoring.

A hybrid box pipelines: the GPU sieves q+1 while the CPU cofactors q, so
per-q time is the **max**, not the sum.

| | CPU-only | hybrid (projected) |
|---|---:|---:|
| ms per q | 281–295 | **71.1** (max of GPU 64.4, CPU floor 71.1) |
| component-proxy watts | 158.7 | **337.0** (125.1 CPU + 206.7 GPU board + 5.2 DIMM) |
| **J per q** | **44.6–46.8** | **23.96** |
| throughput | 1.00x | **3.95–4.15x** |
| **energy** | 1.00x | **1.86–1.95x** |

**Three things follow for this deployment mode.**

1. **The projected hybrid energy win is half its throughput win.** ~4x faster,
   ~1.9x more efficient, because the GPU adds 207 W while the CPU remains busy.
   For this hybrid, **1.9x is the relevant projection**, not finding 44's 2.64x
   stage comparison and not the 4x throughput number. The GPU-resident
   relations/watt result remains unmeasured.

2. **One RTX 5070 approximately fills one 9800X3D-class CPU in this split.**
   The ideal max() is pinned by 71.1 ms of retained CPU work versus a 64.4 ms
   sieve. Under perfect overlap and zero glue, further sieve optimization does
   not improve this particular hybrid. That statement does not apply to the
   GPU-resident path, to another CPU/GPU ratio, or if missing glue puts the GPU
   stage back on the critical path.

3. **The hybrid's next lever is its retained side.** Tightening the survivor
   bound did not reduce the 37-relation yield at bound, bound+6, or bound+12,
   but it also did not remove the retained work. Moving resieve/TD and
   cofactorization to the GPU is the primary roadmap; retaining them on the
   CPU remains the optional shortcut.

**Assumptions, stated so they can be attacked.** Perfect pipelining with no
stall; the CPU cofactor path unchanged and fully parallel at `N_eff = 10.24`;
GPU and CPU plateau powers additive; approximately one strong CPU available per
GPU. CPU cofactor-only scaling and power have not been isolated directly.

**For hybrid accounting, resieve is already charged to retained CPU TD.**
`oracle/ggnfs_timing_breakdown.txt` accounts for wall time in four categories
that close to within 0.01% at every q: Sieve (1683 ms), medsched (309) and
Sieve-Change (373) — all replaced — against **TD including MPQS (734 ms),
retained**. There is no separate resieve line because lasieve4 recovers a
survivor's prime factorisation *inside* trial division, which is the retained
phase. The 71.1 ms hybrid stage therefore already pays for it, and the hybrid
GPU need only deliver survivor `(i,j)` pairs. For the primary GPU-resident
design, resieve/factor recovery remains an unimplemented and unmeasured GPU
stage; CPU bookkeeping does not resolve that engineering work.

One caveat for the hybrid: this holds for the **GGNFS** path, which the 182/56 split
descends from. CADO's las does have an explicit resieve inside its sieve time,
so a CADO Gate 0 comparison must re-derive the split rather than reuse this
one. The residual risk is instead our **+5.6% survivor excess** (finding 40),
which raises the retained TD load by the same 5.6%.

The hybrid's survivor transfer is not a bandwidth risk: ~800K survivors/q at
4–8 B is 3–6 MB, and at 14 q/s that is 45–90 MB/s over an x16 link. The primary
path should keep those candidates resident and transfer final relations, so
this bandwidth observation is not an argument for choosing the hybrid.

## Finding 46 — the GPU-resident budget: what Path 4 has to hit

The primary target is GPU-resident, so the hybrid max() model does not apply —
a standalone box pays **sieve + every post-sieve stage, summed**. That makes
the verdict a budget question, and every input is now measured, so the budget
is derivable rather than notional.

Box power in GPU-resident operation is finding 44's stage-weighted proxy,
**262.1 W** (206.7 W GPU board + ~53 W host + ~2 W DIMM) — measured while the
GPU sieves and the CPU only orchestrates, which is exactly the target mode.
Against the CPU-only box at 44.6–46.8 J/q:

| goal | total ms/q allowed | **post-sieve budget** (after the 64.4 ms sieve) |
|---|---:|---:|
| throughput parity | 281–295 | ~220 ms — not the binding constraint |
| **energy parity** | **170–179** | **~106–114 ms** |
| **2x energy win** | **85–89** | **~21–25 ms** |

Read that carefully, because it is the whole project in three rows.
**Throughput parity is nearly free** — the sieve alone is 4.5x faster than the
CPU box, so almost any post-sieve path beats the CPU on relations/sec.
**Energy parity is comfortable**: ~106–114 ms for intersection, compaction,
primitive filtering, resieve/factor recovery, TD and cofactorisation, against
a CPU that needs 71 ms of *whole-box* time for the last three. **A 2x energy
win is tight**: ~21–25 ms for all of it.

So the honest statement of the remaining question is not "can it work" but
**"where in the 21–114 ms band does the post-sieve path land"** — and the
answer decides between a marginal result and a strong one. That is a far
better-posed question than this project had yesterday.

Two required rates follow, at the measured 15.5 q/s sieve pace:

| stage | volume | **required rate** |
|---|---:|---:|
| primitive candidates out of intersection | ~0.80M/q | **12.4M/s** |
| hard cofactors after TD funnel | ~1,900/q | **29,500/s** |

For reference the CPU produces 1.015–1.065 **relations/joule** at 47.5
relations/q. That is the number to beat and it is the first time it has been
written down.

**Caveat on the power figure.** 262.1 W was measured with the GPU running the
*sieve*. Post-sieve stages have different occupancy and instruction mixes —
ECM in particular is integer-heavy and branchy — so their board power may
differ. The budget should be re-derived once any post-sieve stage is measured;
until then treat 262 W as the sieve-plateau estimate, not a pipeline average.

## Finding 47 — preliminary YAFU 3LP cofactor benchmark clears the rate gate; a target run is still required

The owner supplied a 2026-02-05 log from Ben Buhrow's standalone
`nfs_3lp_batch_factor` suite. This was **not rerun in this results session**
because the CPU and GPU were occupied, and the raw log is not yet a frozen
artifact in this repository. It is nevertheless useful as a fail-fast result:
the RTX 5070's recurring 64/96-bit cofactor work is comfortably faster than
Path 4's estimated feed rate.

The source dataset is one million records from a **C164 GNFS job with
`lpbr=lpba=31`**. The project target is C183 with `lpbr=31, lpba=32`, so this is
not a target-equivalent benchmark. The CPU and GPU paths also use different
algorithms—batch GCD on the CPU versus staged ECM on the GPU—so the ratio is a
comparison of complete cofactor strategies, not the speedup of one identical
kernel.

The logged commands were `./cuda_3lp -m 1` on the CPU and
`./cuda_3lp -m 0 -b1 300 -b2 50 -c 100 -s 10` on the GPU. The latter printed
an effective 96-bit ECM run at `b1=300`, `b2=15000`, 100 requested curves and
a ten-curve no-success stopping rule. The CPU's 22.7166 s timer excludes both
its one-time 54,321,530-prime product build and the 0.55 s input parse; the GPU
outer timer excludes its 0.98 s parse but includes CUDA startup and teardown.

| path | reported time / 1M inputs | input rate | vs. 29,500/s Path-4 feed | equivalent time/q at 1,900 inputs/q |
|---|---:|---:|---:|---:|
| Ryzen 7 9800X3D batch solve (`-m 1`) | 22.7166 s | 44,021/s | 1.49x | 43.16 ms/q |
| RTX 5070 outer cold invocation (`-m 0`) | 11.9524 s | 83,665/s | 2.84x | 22.71 ms/q |
| RTX 5070 sum of printed recurring stages | **2.5901 s** | **386,084/s** | **13.09x** | **4.92 ms/q** |

The 2.5901 s estimate is the sum of the timers printed around the three work
stages: 140.4765 ms for 367,684 64-bit inputs, 2,431.7275 ms for 903,301
96-bit inputs, and 17.9066 ms for the final 19,352 64-bit inputs. The 96-bit
stage is 94% of that sum. Ben's correction that the useful GPU work was about
2.4 s is therefore directionally right. The more conservative **2.5901 s is
the provisional steady-work estimate**; it is not yet a clean persistent-mode
wall-clock measurement.

Source inspection explains much of the 11.9524 s cold time: the outer timer
includes GPU discovery, context creation, PTX load/JIT, kernel setup,
allocation/packing, and teardown. A production queue can retain the context,
module, and buffers. It is not valid, however, to label the entire
`11.9524 - 2.5901` s difference one-time until a persistent driver times the
full recurring path, including preparation, compaction, and validation.

The GPU run produced 10,246 complete relations versus 10,257 on the CPU:
**99.893% of the CPU count, 11 fewer**. That is encouraging but not a
correctness proof. The GPU run stopped after 95 of 100 requested curves after
ten curves without a valid factor, and the two strategies need not choose the
same successful subset. The controlled rerun must compare validated relation
sets and report yield as well as speed.

If the C164 stage mix and difficulty transfer to the target workload, the
hard-cofactor stage would consume about **4.9 ms/q**, leaving roughly
**16–20 ms/q** of finding 46's 21–25 ms post-sieve allowance for a 2x energy
win. Even the cold outer timing clears the raw 29,500/s rate by 2.84x, but at
22.7 ms/q it consumes essentially that entire aggressive allowance. The
correct conclusion is therefore:

> **The preliminary cofactor-rate fail-fast gate passes.** It does not yet
> close target coverage, steady-state latency, energy, correctness, or
> host-independence. If the target rerun holds near 4.9 ms/q, GPU
> resieve/factor recovery/TD becomes the largest unknown inside the 2x budget.

One million records represent about 526 target q at 1,900 records/q—roughly
34 seconds of arrivals at the 15.5 q/s sieve rate. The rerun must sweep smaller
batches and overlapping queues; a result that only saturates after buffering
half a minute is not sufficient evidence for the production pipeline.

### Harness audit before the controlled rerun

The freshly downloaded suite has changed since the February log (including
P-1 support and CPU TinySIQS/MPQS tails), and a source audit found several
prototype hazards to fix or explicitly control before trusting new numbers:

- The stack `relation_batch_t` is not zero-initialized. The CPU batch path
  resets its success count, but diagnostic counters are not initialized; this
  explains the nonsensical ~3.49-billion ECM/abort counters in the old CPU
  log. The current GPU path also needs an explicit success-count reset before
  it increments the field.
- The Makefile defines uppercase `TOOLKIT_VERSION`, while the source tests
  lowercase `toolkit_version`; the CUDA-version branch can therefore be the
  wrong one. Its CUDA-13 branch also passes an uninitialized
  `CUctxCreateParams *` to context creation.
- All devices with compute major >= 9 currently select `cuda_ecm90.ptx`; the
  RTX 5070 log consequently loaded sm_90 PTX on an sm_120 card. Add a native
  sm_120 artifact and loader selection before treating the rerun as a fair
  Blackwell result.
- The GPU path is not host-independent as written: the CPU parses and packs
  records, prepares Montgomery constants with GMP, validates and
  primality-checks results, compacts between curves, and in the newer tree may
  run CPU TinySIQS/MPQS tails. Record CPU utilization and component power, and
  distinguish required recurring host work from removable prototype setup.

The target rerun should therefore use a C183 `31/32` survivor/cofactor set,
zeroed accounting, native sm_120 PTX, a persistent context and reusable
buffers, a batch-size/queue-depth sweep, exact output validation, and
simultaneous CPU/GPU power sampling. Until that run exists, use 4.92 ms/q as a
promising sizing datum—not as a Path-5 result.

## Finding 48 — the standalone bench's `transform` line was measuring CUDA startup

k_transform is the first kernel of a standalone run (no `--pipeline`), so it
absorbed the entire one-time CUDA cost — module load for the fatbin, context
setup — and reported it divided by `--reps`. On WSL2 that fixed cost measures
~170–220 ms.

RTX 5070, c147, `--logI 14 --J 8192`, idle GPU:

| `--reps` | transform | fill | apply |
|---:|---:|---:|---:|
| 3 | **71.181** | 3.812 | 5.424 |
| 20 | 11.969 | 3.840 | 5.494 |
| 100 | 3.177 | 3.832 | 5.567 |
| 300 | 1.123 | 3.843 | 5.560 |
| 1000 | **0.728** | 3.846 | 5.583 |

Transform swings **98×** across a 333× range. Fill moves 0.9% (3.812 → 3.846)
and apply 2.9% (5.424 → 5.583) — apply's drift is small but monotone, so it is
not pure noise, and apply is the one stage that should be quoted with its reps
setting rather than treated as reps-free. The true transform cost is ~0.55 ms;
the pipeline, where it runs once per q against a warm context, independently
reports **0.954 ms** for both sides.

The damage was not hypothetical. Three GPUs were compared on this number at
three different `--reps` settings, and the resulting nonsense — an RTX 3090
appearing to transform 10× faster than a 5070 — was taken seriously for two
rounds before anyone checked whether the metric was stable.

**Fixed** by an untimed warm-up launch ahead of the timed loop, with the
`nproj`/`nlost` memsets moved after it (they are accumulators divided by reps).
That removes ~92% of the artifact — reps 3: 71.18 → 6.05, reps 100: 3.18 →
0.645 — but not all of it *on this box*, so **`--reps 100` is the floor for any
cross-machine comparison** and low-reps transform numbers stay untrustworthy.

**The residual is WSL2-specific.** The 5090 in finding 51 ran this same fixed
binary (commit `ad958cc`) on native Linux and reported transform at **0.149 ms
at `--reps 3`**. That is its true cost, not an inflated one: its pipeline
transform is 0.231 ms for both sides (3.13M ideals), and the standalone sieves
side 1 alone (2.06M), so the expected standalone figure is 0.231 × 2.06/3.13 =
**0.152 ms**. Measured 0.149.

So the warm-up fully solves the problem on native Linux, while WSL2 retains
~12× inflation at reps 3 (6.05 against a true ~0.50). The reps floor is a rule
for *this* box. Numbers other people took at low reps on native Linux were
probably fine; what actually broke was comparing across the two platforms.

The general rule this establishes: **a stage whose reported time depends on
`--reps` is not measuring the kernel.** Fill passes that test at reps 3; apply
passes to within 3%; transform fails it by two orders of magnitude.

The same gap existed in `run_pipeline`, which launched no kernel before
`k_transform` and so charged the whole one-time cost to the first q's transform
window before dividing by the band length. Fixed the same way. At 1340 q it was
worth ~0.15 ms on a 0.954 ms figure (+16%); on a 50-q band it would have been
~4 ms. Lazy module loading is per-kernel, so both warm-ups cover transform only
— fill and apply still pay their first-launch cost inside q0.

## Finding 49 — the grid width was hardcoded to this box's SM count

`blocks = cfg->blocks ? cfg->blocks : 48 * 6` appeared in three places, and no
`cudaGetDeviceProperties` call existed anywhere in the tree. The 48 is this
5070's SM count, so every other GPU ran the 6-blocks-per-SM tuning at whatever
occupancy 288 blocks happened to give it: 3.5 blocks/SM on an 82-SM 3090 (58%),
2.25 on a 128-SM 4090 (37%).

It reached `k_transform`, `k_fill_atomic`, `k_td`, `k_classify`,
`k_resieve_scatter` and the cofactor kernels. `k_apply` launches `nregion`
blocks and was never affected.

Now resolved from `multiProcessorCount * 6` and echoed on stdout at startup —
unconditionally, including when `--blocks` overrides it, since that is exactly
the A/B that wants the number. A failed device query is now fatal: it used to
leave `cfg.blocks` at 0, and the three `48 * 6` fallbacks then silently
reinstated this box's SM count with no diagnostic. Those three constants remain
as unreachable fallbacks; they are dead only so long as every entry point
routes through `main()`'s validation.

Unchanged on this box by construction. The reporter's 3090 saw the standalone
fill benchmark move from ~20 ms to ~14 ms, and ~6% end-to-end on the pipeline
(fill being ~21% of wall). **Those two figures are on the reporter's own
config, which was never recorded** — they are not comparable with finding 50's
3090 fill of 7.19 ms at `--logI 14 --J 8192 --reps 100`, and the ~1.4× ratio is
the only thing to take from them.

Related: `NVCC_ARCH` shipped an sm_120 cubin plus **compute_89 PTX**. The driver
only JITs PTX to a target ≥ the virtual arch, so that build could not load on
any Ampere card at all. Now sm_120 + sm_89 + sm_86 native, compute_80 PTX.

## Finding 50 — the design ports across architectures; fill is the whole gap

Three GPUs, c147 at `--logI 14 --J 8192`. Transform excluded per finding 48.

| | SMs × GHz | INT32 | FP32 | `--reps` | fill | apply |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 (sm_120) | 48 × 2.51 | 15.4 T | 15.4 T | 100 | **3.83** | 5.57 |
| RTX 3090 (sm_86) | 82 × 1.70 | 8.9 T | 17.8 T | 100 | 7.19 | **3.92** |
| A100 80GB (sm_80) | 108 × 1.41 | 9.8 T | 9.8 T | **3** | 9.42 | 6.18 |

INT32 is ops/s; FP32 is FMA/s; both are T. Consumer Ampere has 128 FP32 lanes
per SM but only 64 that accept INT32; Blackwell unified all 128. GA100 has 64
of each.

The A100 rows predate finding 48's `--reps 100` floor. Fill is reps-stable so
that row stands; apply carries up to ~3% of reps drift and should be re-taken.
Transform is omitted for all three because no reps setting makes it comparable
across machines.

**Apply's FP32 ranking is exact; its magnitudes are not.** Predicted ranking
3090 > 5070 > A100, measured 3.92 < 5.57 < 6.18 — three for three. But the
3090's predicted ratio (15.4/17.8 = 0.865 → 4.82 ms) misses the measured 3.92
by 19%, which is the same order of error as the fill miss flagged below as
unexplained. The ranking is evidence the stage is FP32-led; the model does not
predict its size. The A100 reaching near-parity on 0.63× the FP32 is consistent
with apply being 58% DRAM-bound (finding 8) on 2.9× the bandwidth, but that is
an explanation offered, not a fit tested.

**Fill tracks INT32 for the 3090** — predicted 1.73× slower than the 5070,
measured 1.88× — **and fails for the A100**, predicted 1.58×, measured 2.47×.
That miss is unexplained.

**REFUTED — see finding 51.** The hypothesis here was cursor contention: fill
scatters into 8192 cursors fixed by `--region` rather than by the GPU, so wider
cards were thought to pile more SMs onto the same contention points. `--region
13` on a 128-SM 4090 made fill *worse* (6.785 → 7.028) and left the 4090/5070
ratio unchanged, so cursor count is not the variable. The right axis is total
concurrency, and the INT32 model above is refuted with it — finding 51 has the
sweep. The 5070 control recorded here stands as data: fill is flat at **3.729 /
3.742 / 3.965 ms** for 8192 / 16384 / 32768 cursors, and apply degrades hard as
regions shrink (5.51 → 8.28 → 13.36 ms over the same sweep), so `--region` is
not a tuning knob in either direction.

**Whole-pipeline consequence.** Same command, same work (1340 q, ~159.8K
relations):

| | 5070 | A100 |
|---|---:|---:|
| wall/q | 25.10 | 37.02 |
| sieve, both sides | 17.74 | 29.73 |
| — transform / fill / apply | 0.954 / 7.459 / 9.322 | — |
| TD + classify, device | 3.121 | 4.420 |
| host per-q | 0.811 | 0.892 |

The sieve gap (11.99 ms) **is** the wall gap (11.92 ms). TD, cofactorisation
and host work contribute nothing net — the Amdahl risk the project was designed
around did not materialise here. Within the sieve, fill accounts for ~11.0 ms,
**92% of the total deficit**.

Two conclusions for the probe. The kernels are portable: nothing
architecture-specific broke, and the ranking is explained by published lane
counts rather than by anything in the design. And **fill is the only lever left
worth pulling** — 42% of sieve time on this box, and essentially the entire
difference against datacenter silicon. Production transform is 0.954 ms, 5% of
sieve; there is nothing there.

Caveat on the A100 rows: they were taken before finding 48's warm-up landed, at
`--reps 3`, so fill and apply are trustworthy (reps-stable) and transform is
not reported. A rerun at `--reps 100` on the current tree would tighten them.

## Finding 51 — fill saturates at 144 blocks on every card and does not scale with the GPU

> **SUPERSEDED by finding 52 on the geometry, 2026-08-06.** Every block sweep
> below holds `--threads` at 256 and varies blocks alone. Varying the block
> *width* moves the optimum to **1152 × 32** and dissolves the
> architecture-specific block response recorded here — the 4090's "+38% by 768"
> degradation reverses sign at 32 threads. `FILL_BLOCKS_DEFAULT` is 1152, not
> the 144 asserted below. What survives is the *scaling* result: fill still
> returns far less than the hardware ratio, on the corrected geometry too.

Adding the RTX 4090 (AD102, 128 SM × 2.52 GHz) and RTX 5090 (GB202, 170 SM ×
2.41 GHz) to the finding 50 set produced a result no hypothesis on the table
predicted. Same `--logI 14 --J 8192`:

| | transform | fill | apply | chain |
|---|---:|---:|---:|---:|
| RTX 5070 | 0.504 | **3.777** | 5.482 | 9.763 |
| RTX 4090 | 0.250 | 6.785 | **2.814** | 9.849 |
| RTX 5090 | 0.149 | 3.250 | 1.908 | 5.307 |

The 4090 tied the 5070 on the total from stages differing ~2× in both
directions. The 5090 rows are `--reps 3`, but on the warm-up-fixed binary
(`ad958cc`) and on native Linux, where that is enough — its 0.149 ms transform
matches the 0.152 predicted from its own pipeline figure. See finding 48 for
why the same reps setting is worthless on the WSL2 box.

A dead heat on the total, from stages that differ by ~2× in both directions.
The full pipeline was the same story: **25.97 ms/q on the 4090 against 25.28
on the 5070** — the wider, hotter, more expensive card losing by 2.7%.

### Three mechanisms, all refuted by measurement

1. **L2 capacity** (the original hypothesis). Dead: the 4090 has 1.5× the
   5070's L2, 1.5× the bandwidth and 2.67× the SMs, and its fill is 1.80×
   slower.
2. **INT32 lane count** (finding 50's model, which fit the 3090 to 8%). Dead:
   the 4090's INT32 peak is 20.6 T against the 5070's 15.4 — **1.34× more**
   integer throughput, 1.80× slower fill.
3. **Bucket-cursor contention** (finding 50's stated hypothesis, with the
   5070's flat `--region` sweep as its control). Dead: doubling cursors on the
   4090 via `--region 13` made fill *worse*, 6.785 → 7.028, and the
   4090/5070 ratio was unchanged at 1.82× vs 1.88×. Finding 50's contention
   note is withdrawn.

### What is actually true: an absolute concurrency optimum

Sweeping `--blocks` with `--stage fill`:

| blocks | threads | 5070 (48 SM) | 4090 (128 SM) | 5090 (170 SM) |
|---:|---:|---:|---:|---:|
| 32 | 8,192 | 5.383 | 6.680 | — |
| 48 | 12,288 | 4.827 | 6.384 | — |
| 64 | 16,384 | 4.828 | 5.392 | — |
| 96 | 24,576 | 4.778 | 6.039 | 4.362 |
| **144** | **36,864** | **3.726** | **4.806** | **3.156** |
| 288 | 73,728 | 3.854 | 5.638 | **3.113** |
| 576 | 147,456 | 3.906 | 6.473 | 3.145 |
| 768 | 196,608 | 3.955 | 6.636 | — |
| 1020 | 261,120 | — | — | 3.238 |
| 1536 | 393,216 | 3.901 | 6.524 | — |

**All three cards saturate at the same absolute 144 blocks** — 3 per SM on the
5070, 1.1 on the 4090, 0.85 on the 5090. Every one of them falls off sharply
below it (96 blocks costs 22–28%) and none gains anything above it. Three cards
spanning **3.5× in SM count** want the same ~37K threads in flight, so SM count
is the wrong axis for this kernel, not merely the wrong constant.

Above saturation the architectures split. **Blackwell is flat** — the 5070
varies 6% out to 1536 blocks, the 5090 4% out to 1020. **Ada degrades**, the
4090 climbing 38% by 768. So the earlier reading that both cards "degrade in
both directions" was wrong: only Ada degrades upward, and the 4090's 27% was a
penalty specific to it rather than a gain available everywhere.

144 is nonetheless the right default on all three: at or within noise of the
best point on every card, and it avoids Ada's penalty entirely.

The Ada/Blackwell split also fits the two fill misses finding 50 could not
explain — the A100 at 648 blocks and the 3090 at 492 were both far past 144 on
pre-Blackwell parts, though neither was swept to confirm it.

### Fill does not scale with the GPU

The 5090 has **3.5× the SMs, 2.7× the bandwidth and 3.4× the FP32/INT32** of
the 5070. Measured at each card's best:

| stage | 5070 | 5090 | speedup |
|---|---:|---:|---:|
| transform | 0.770 | 0.231 | 3.33× |
| apply | 9.385 | 4.681 | 2.01× |
| TD + classify (device) | 3.091 | 2.567 | 1.20× |
| **fill** | **3.726** | **3.113** | **1.20×** |

Every stage scales except fill, which returns 20% for 3.5× the hardware. Across
all five cards measured, fill spans only 3.11–9.42 ms while apply spans
1.91–6.18 tracking FP32 cleanly. **Fill is a near-fixed cost that GPU money
does not buy down**, and it is 40–56% of sieve time. A 5070 lands within 20% of
the best fill any card tested achieves.

The dominant sieve stage is insensitive to everything that makes a GPU
expensive. Note carefully what that does and does not imply: it bounds
relations per **dollar**, not relations per **joule**. A card that cannot use
its width also does not draw for it — the measured-power table below shows the
5090 idling at 47% of nameplate — so flat fill scaling and low draw are the
same fact, and it is only the throughput half of it that costs anything. This
paragraph previously ran on to call the result central *for a probe graded on
relations/sec/watt*, which had the sign backwards.

The surviving candidate mechanism is L2 write-combining decay: more concurrent
walks interleave each bucket's writes further apart, so 32 B sectors evict
before they fill. It is consistent with every observation including the mild
more-buckets-is-worse trend on both cards. **It is a candidate and nothing
more** — three mechanisms that also fit the data at the time have already been
refuted here, and confirming this one needs counters (`lts__t_sectors_op_write`
vs `dram__bytes_write` per card). `ncu` on a rented Vast box hits
ERR_NVGPUCTRPERM, which is a host module parameter and not fixable from inside
the container.

### The fix

`FILL_BLOCKS_DEFAULT = 144` in `bench.h`, an absolute count, with
`--fill-blocks N` to override. Fill's grid is now decoupled from the
`multiProcessorCount * 6` grid the other kernels use, and **`--blocks` no
longer moves fill** — the sweeps in this finding were taken with the old
binary where it did. Both grids are echoed at startup.

Measured at one job shape (8192 buckets, 77.4M records). The optimum plausibly
moves with bucket and record count; that is not measured, so characterise a new
job shape with `--fill-blocks` before trusting the default on it.

### Consequences

Projecting the 4090's 27.6% fill gain onto its pipeline: sieve loses ~3.5 ms,
wall goes 25.97 → **~22.5 ms/q, an 11% win over the 5070** rather than a 2.7%
loss. Projected, not measured. The 5070 and 5090 are unaffected — for them 144
is inside run-to-run noise of what they already ran (the 5070's 144 point
measured 3.726 and 3.850 in two sweeps, ~3%).

**RETRACTED: the efficiency conclusion was an artifact of nameplate TDP.**

This section previously read "the efficiency conclusion holds across four
cards" and concluded **"this design does not want wide expensive GPUs"** from a
rel/J table built on *nameplate* TDP. That table is withdrawn. Measured board
draw, sampled at 5 Hz through a running band with
`nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 200`,
removes the result:

| | ms/q | rel/s | nameplate | measured draw | % plate | old rel/J | **rel/J** |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5070 | 25.15 | 4,742 | 250 W | **168.7 W** (590 samples) | 67% | 19.0 | **28.1** |
| **RTX 5090** | 14.95 | 7,978 | 575 W | **270.4 W** (369 samples) | 47% | 13.9 | **29.5** |
| RTX 4090 | 25.97 (→ ~22.5) | 4,593 | 450 W¹ | **254.2 W** (423 samples) | 64%¹ | 10.2 | **18.1** |
| A100 80GB | 37.02 | 3,222 | 300 W | *unsampled* | — | 10.7 | — |

¹ This particular 4090 is capped at **400 W**, so even the nameplate row was
wrong for the card that produced the number; the percentage is against 400.

**Every card draws far under nameplate, and not by the same factor** — 47% for
the 5090 against 67% for the 5070. TDP was therefore not a constant offset that
cancelled in the ratios, which is the assumption the old table's "treat the
ratios as indicative" hedge silently made. It penalised the widest card
hardest, precisely because the widest card is the one that idles.

**The corrected result is a tie, not a reversal.** The 5090 and 5070 land at
29.5 and 28.1 rel/J — within 5%, inside the run-to-run spread this project has
measured elsewhere. The 5090 is not more efficient in any way worth defending.
What it is, is **equally efficient while being 1.68× faster.**

That is a different claim from the retracted one and it points the opposite
way. If rel/J is flat across a 3.5× SM range, relations/joule stops
discriminating between these cards at all, and the decision falls to relations
per **dollar** and throughput per box.

**The split that does survive is architectural, and it is not width.** Both
Blackwell cards sit at 28–30 rel/J; the Ada 4090 sits at 18.1, worse by 36%
than a 5070 costing a third as much. That is the same division finding 51 found
in fill scaling — Blackwell flat above saturation, Ada degrading 38% — now
visible in the power domain. **Generation, not size, is what the measured data
separates.**

**Still not established.** Board draw excludes the host, and the metric of
record is whole-box relations/sec/watt; the A100 is unsampled; and all four
throughput figures are one-q-at-a-time. See "concurrent-q throughput" in the
open experiments — the 5090's 47% draw is headroom, and if that idleness is
schedulable its rel/J moves up from a tie, not down.

### Correction to finding 49

Finding 49 credits the grid fix with moving the reporter's 3090 standalone fill
from ~20 ms to ~14 ms. Given fill is flat from 144 to 1536 blocks, 288 → 492
cannot have produced that. The likely cause is that the "before" figure was a
`SIEVE CHAIN` total at low `--reps`, dominated by the transform startup artifact
of finding 48, which had not been found yet. The grid fix remains correct on
occupancy grounds for transform, TD, classify, resieve and the cofactor kernels;
**its effect on fill was nil and the ~20 → ~14 ms should not be attributed to
it.**

## Finding 52 — fill wants 1152 x 32, and finding 51's architecture split was an artifact of a fixed 256 threads

Finding 51 swept fill's **block count** at a fixed `--threads 256` and read the
resulting 144-block minimum as a hard saturation point, with an
architecture-specific response above it (Ada degrading 38% by 768 blocks,
Blackwell flat). Block **width** was never varied. It is not a free parameter.

### At constant total threads, narrower blocks always win

36,864 threads throughout, so every row is the same work differently cut up.
Standalone `--logI 14 --J 8192 --reps 100`, fill ms:

| T x B | 5070 | 4090 | 5090 |
|---|---:|---:|---:|
| 192 x 192 | 3.612 | 4.482 | 2.893 |
| 128 x 288 | 3.553 | 4.654 | 2.830 |
| 96 x 384 | 3.455 | 4.407 | 2.760 |
| 64 x 576 | 3.530 | 4.419 | 2.716 |
| **32 x 1152** | **3.454** | **4.384** | **2.636** |

### Thread count matters above 128, and not below it

At a constant 576 blocks the 5090 measures **2.716 / 2.711 / 3.147 / 4.289 ms**
at 64 / 128 / 256 / 512 threads. So 256 is **16% off** and 512 is 58% off,
while **64-128** is flat to 0.2%. Note 64, not 32 — 576 × 32 was never
measured, so flatness is established down to 64 and assumed below it.

At the shipped 1152 blocks the gap is **wider**, not narrower — 2.633 at 32
threads against **3.239 at 256, a 23% penalty**:

| 5090, fill ms | 32 thr | 128 thr | 256 thr |
|---|---:|---:|---:|
| 576 blocks | — | 2.711 | 3.147 |
| **1152 blocks** | **2.633** | — | 3.239 |

Note 1152 × 256 (3.239) is worse than 576 × 256 (3.147): **at 256 threads more
blocks still hurts** — the finding-51 behaviour — while at 32 threads more
blocks helps. The two axes interact strongly and neither can be swept alone.

Two consequences: a sweep pinned at 256 threads was 16-23% off the optimum
before it began, and fill's width cannot be tuned through `--threads`, which
also drives transform, intersect, TD, resieve and the cofactor kernels. Hence
`--fill-threads`. This cell was run specifically to test whether that flag
earns its keep rather than being an over-engineered alternative to raising
`FILL_BLOCKS_DEFAULT`; raising blocks alone would have landed on 3.239.

### The knee is 1152 blocks on all three cards

At `--threads 32`, fill ms:

| blocks | 5070 | 4090 | 5090 |
|---|---:|---:|---:|
| 288 | 3.909 | 5.089 | 3.629 |
| 576 | 3.487 | 4.626 | 2.983 |
| **1152** | **3.444** | **4.360** | **2.633** |
| 2304 | 3.410 | 4.407 | 2.668 |
| 4608 | 3.312 | 4.327 | 2.656 |
| 9216 | 3.312 | 4.285 | 2.632 |

Past 1152 the remaining movement is 1.7% (4090), 0.04% (5090) and 3.8% (5070,
whose own run-to-run spread is ~3%) — flat, not falling. Below it the cost is
steep. Overshooting is nearly free and undershooting is not, so the default
sits **at** the knee. Against the old 144 x 256: **7.5% / 8.8% / 16.7%**.

### The Ada/Blackwell split dissolves

Finding 51's headline architectural claim was that the 4090 *degrades* with
more blocks while Blackwell stays flat. At 32 threads the 4090 **improves**
monotonically over the same range. Both sweeps are correct; they differ only in
the fixed thread count. The block response is not a property of the
architecture, it is a property of the width you happen to hold fixed — and at
32 all three cards behave alike, which is why one geometry serves all of them.

### It is not L2

L2 capacity was already dead (finding 51's own list); write-combining decay was
that finding's surviving candidate. A single 1152 x 32 optimum across cards
with **48, 72 and 96 MB** of L2 argues against both. The behaviour it does fit
is **work granularity**: fine chunks balance the tail, and the effect saturates
once chunks are numerous enough — which is exactly the flat plateau above.
Still a candidate, not a conclusion; `ncu` remains blocked on the rented boxes.

### Verified end to end, not just in the microbenchmark

Same binary, same session, geometry the only variable. c147 band, 1340 q,
RTX 5070:

| | 144 x 256 | 1152 x 32 | delta |
|---|---:|---:|---:|
| wall clock/q | 25.00 ms | 24.47 ms | **-2.1%** |
| sieve, both sides | 17.94 | 17.48 | -2.6% |
| transform | 0.766 | 0.764 | -0.3% |
| **fill** | 7.576 | 7.113 | **-6.1%** |
| **apply** | 9.601 | 9.607 | **+0.06%** |
| relations | 159,837 | 159,837 | **byte-identical** |

Apply was the risk worth checking: it reads the bucket array fill writes, and
the write interleaving changed 32-fold. It did not move. The sorted relation
files compare equal, both reconstruction gates pass with the same 1747/1952
factor counts, and `cofcheck.sh` is 30/30. The pipeline gain (6.1%) is smaller
than the standalone predicted (7.5%), so expect the 5090's 16.7% to land nearer
13-14% in a band.

**Methodological note.** The first version of this A/B compared against a fill
figure captured in an earlier session and appeared to show apply regressing
2.4%. It had not: this box's apply drifts ~2.3% between sessions, and the
controlled same-binary run showed +0.06%. **Do not A/B against a captured
number from another session on this box** — rerun the control.

### Not yet measured

- The pipeline A/B is 5070-only; the 5090 is where the largest gain is claimed.
- `k_fill_l1` (twolevel path) has never been swept at any geometry and keeps its
  own 144 x 512 default.

## Finding 53 — host contention costs 29% of wall clock and every GPU counter we have is blind to it

Reported by the 3090 tester: a saturated CPU leaves the CUDA timings untouched
but moves wall clock. Reproduced here on the 5070, c147 band, 1340 q, same
binary, load applied with N spinning shells:

| | idle | 8/16 cores | 16/16 cores |
|---|---:|---:|---:|
| **wall clock/q** | **24.30 ms** | **29.79 (+22.6%)** | **31.27 (+28.7%)** |
| sieve, both sides | 17.48 | 18.94 | 17.41 |
| — transform | 0.764 | — | 0.822 |
| — fill | 7.113 | — | 7.085 |
| — apply | 9.607 | — | 9.505 |
| intersect + gcd | 0.074 | — | 0.105 |
| host per-q (tables, staging) | 0.699 | 1.249 | **1.994** |
| TD + classify, wall | 4.11 | 5.73 | **7.06** |
| host: small-prime tables | 0.467 | — | 0.833 |
| host: unaccounted | 0.528 | 1.594 | **2.869** |
| cofactorisation flushes | 0.56 | — | 0.97 |
| relations | 159,837 | 159,837 | 159,837 |

The sieve row at half load (18.94) is above both other columns, so it is
run-to-run noise, not a trend — the full-load column (17.41 against 17.48
idle) is the one to read for whether the GPU stages move. They do not.

**Replication status: each column is a single run.** That noise disclaimer
concedes an ~8% spread on the largest term, so the wall deltas carry an
uncertainty this finding does not quantify, and "every kernel within ~1%" rests
on the one column that happened to land close. A replication attempt was made
and had to be discarded: the box turned out to be running a production snfs236
sieve on the same GPU plus a 14-thread msieve, which put wall clock at ~53 ms —
so the contamination signature is unmistakable, and its absence from the 24-25
ms idle runs is the evidence that those were clean. **Re-run all three columns
n>=3 on a confirmed-idle box before quoting these as hardware constants.**

**The kernels are flat and the wall clock is not.** Every `cudaEvent`-timed
stage is within ~1% (`fill` -0.4%, `apply` -1.1%). Everything host-side scales
with contention, worst of all `unaccounted` — wall minus device time inside
TD/classify, i.e. launch and synchronisation overhead — at **+443%**.

### Why this is a measurement hazard and not just a scheduling tip

Our instrumentation is *structurally* unable to see the largest environmental
effect on throughput. A host-starved box reports flawless kernel times next to
a bad ETA, which reads as "the GPU is fine, the ETA is inexplicable" — the exact
shape of the original 3090 report that opened findings 48-52. Any wall-clock or
ETA comparison across boxes is invalid without knowing host load on both, and
the rented boxes used for findings 50-52 are shared machines.

The A100 is the one prior result worth re-examining on these grounds, and it is
**not cleared**. An earlier version of this paragraph said its host per-q of
0.892 ms beat "this box's 1.080", concluding its 37.02 ms wall stands. That
1.080 is unsourced and appears nowhere in this repo. Finding 50's own table —
same job, same 1340 q — records **5070 = 0.811 against A100 = 0.892**: the
A100's host was 10% *slower*, the opposite of what was claimed.

A 10% host gap is well inside what two different host CPUs produce, so it
neither establishes contention nor rules it out. **The A100 wall figure is
unverified** and stays so until a band is run there with the ratio line, on a
host confirmed idle.

### It is far larger for us than the reporter's numbers imply

Their host per-q rose **33%** (0.288 → 0.384 ms); ours rose **185%** (0.699 →
1.994). Quote the two percentages rather than a ratio of them — an earlier
heading here said "~4x", which matches no derivation: the relative growths
differ by 5.6×, the multipliers by 2.1×, the absolute milliseconds by 13.5×.
The difference is which harness was run. The standalone bench does almost no host work — a
transform, a sort and one H2D — while the pipeline carries TD tables, staging
and cofactor flushes. **The standalone structurally understates this effect**,
and the standalone is what testers are usually asked to run.

### There is no free headroom

Half the cores costs most of what all of them cost. State it as **throughput**,
not as percent-of-baseline-wall — an earlier version of this section did the
latter and inflated the decision rule:

| | wall/q | wall vs idle | **relation-rate loss** |
|---|---:|---:|---:|
| idle | 24.30 ms | — | — |
| 8/16 cores | 29.79 | +22.6% | **18.4%** |
| 16/16 cores | 31.27 | +28.7% | **22.3%** |

A +28.7% wall is a 22.3% throughput loss (1/1.287 = 0.777), not a 29% one. So
co-scheduling CPU-NFS work beside the GPU siever pays only if that work is
worth more than ~18% of the GPU's relation output at half load, or ~22% at
full — not the "more than a quarter" this section previously claimed, which
overstated the bar by 27% at half load.

### The pipeline now reports this directly

`GPU-accounted / wall (excl cofac)` prints on every band: event-timed device
time (sieve + intersect + the TD/classify device total) over wall clock, with
the cofactor queue removed from **both** sides.

**Values pending re-measurement.** A first version was measured at 0.824 idle /
0.661 loaded, but with the cofactor queue in the denominator only — which made
the ratio move with survivor density, so a candidate-dense band read as a
contended host on an idle box. That is the exact misdiagnosis the line exists
to prevent, so the expression was corrected and those two numbers no longer
describe what the code prints. Re-taking them needs a confirmed-idle box.

The numerator remains a **lower bound** on device time: `k_cof_enqueue`,
`k_cand_stats` and the flush's own kernels are real GPU work that no event
times. The ratio therefore understates utilisation by a small amount that is
roughly fixed for a given job — tolerable for comparing a box against its own
baseline, which is the only comparison it is for.

It is **not** the most sensitive signal available. An earlier version of this
section claimed it separates the two conditions "where every individual stage
timing does not"; that is false. Two already-printed timings separate them far
more sharply — `host per-q` at +185% and `host: unaccounted` at +443%, against
the ratio's −20%. Its merit is being one scale-free summary that needs no
baseline table to read, not being the sharpest.

No threshold and no warning text is attached, deliberately. The healthy value
depends on card and job — a faster GPU spends relatively more of its wall on
the same host work and therefore reads *lower* while perfectly healthy — so a
hardcoded "good" constant would repeat the 144-block mistake of promoting one
box's number to a universal one. Take an idle baseline per card, job **and band
length** and compare against it: `acc_wall` excludes the final cofactor flush,
so on a band shorter than one flush that tail is the entire cofactorisation and
a smoke run is not comparable to a production band.

### What to do about it: the host work is three problems, not one

The 1.694 ms does not have a single fix, because two thirds of it is *prep* and
one third is *launch overhead*:

| | idle | loaded | what it is |
|---|---:|---:|---|
| host per-q (tables, staging) | 0.699 | 1.994 | prep: tiers, staging, H2D |
| host: small-prime tables | 0.467 | 0.833 | prep: derived from the lattice |
| host: unaccounted | 0.528 | **2.869** | launch + sync overhead |

**Overlap beats threading for the prep, and not narrowly.** Both prep terms
depend only on the q-lattice and on no GPU result, so q+1's host work can be
done during q's kernels — double-buffering, not parallelism. The arithmetic is
lopsided: **1.166 ms of prep against ~20.7 ms of GPU work per q**, so it fits
inside the GPU's shadow with 18x room and perfect overlap takes it to zero on
the critical path. Threading the same work reaches perhaps 0.4 ms on four
threads and leaves it *on* the critical path — strictly worse on the arithmetic
alone.

**But the arithmetic is not the whole cost, and an earlier version of this
paragraph implied it was** by calling threading's downside "a synchronisation
problem that does not currently exist." The host does not sit idle through that
20.7 ms shadow: it blocks at six points per q — `cudaEventSynchronize(e3)` once
per side (`pipeline.cuh:240`), the intersect sync (`:975`), two inside
`pipe_td_perq` (`:671`, `:736`), and `cudaDeviceSynchronize` after
`k_cof_enqueue` (`:1021`). Single-threaded overlap requires splicing q+1's prep
in *ahead* of those syncs, i.e. restructuring them — which is most of the work
sub-item 2's graph capture wants anyway. Overlap is still ranked first because
it and (2) share that restructuring, not because it is free.

**`unaccounted` needs the opposite treatment.** It is wall minus device time
inside TD/classify: the CPU issuing launches and waiting on syncs. It is
interleaved with GPU execution by nature, so overlap cannot hide it, and it is
the term that grew **443%** under contention — it is what makes a box fragile
rather than merely slow. The per-q kernel sequence is fixed, so fewer and
larger launches, fewer sync points, or a captured CUDA graph replayed per q all
attack it directly.

**The micro-optimisation is the least valuable third.** Replacing the per-q
stable sort with a three-way partition and fusing the twice-done small-ideal
transform was previously the whole of open experiment 4. After overlap, that
work is hidden in the GPU shadow and its cost stops mattering. Order: overlap,
then graphs, then the partition.

### Consequences

- Open experiment 4 (host cost) is reordered on the strength of the above, and
  gets more valuable — though not on the idle-case arithmetic: 1.694 ms/q of a
  24.30 ms wall is only **7%**. Reducing it buys robustness on shared hardware
  more than throughput on a quiet box, and against open experiment 3's
  15.8-25% duplicate share it is the smaller prize.
- **It interacts with open experiment 1.** Running two concurrent special-q
  roughly doubles host work per unit time, so the concurrency experiment can
  come back negative for host reasons that have nothing to do with the GPU.
  Run it on a verified-idle box and check `GPU-accounted / wall` first.
- Correctness is unaffected: all three runs emit exactly 159,837 relations.

## Finding 54 — RTX 5060 Ti device timings on c147

**Date:** 2026-08-09. Reported externally on native Linux, RTX 5060 Ti
(36 SM, 32 MB L2), current **1152 x 32** fill geometry. The host was an
i5-2550K and was fully utilised during the pipeline run, so the device-event
timings are the portable result here; the pipeline wall clock is deliberately
not used for a cross-card comparison.

Standalone algebraic side, `--logI 14 --J 8192 --reps 3`, synthetic-root
`q=120000011`, 2,059,531 bucketed entries and 77,389,658 records:

| stage | ms / special-q |
|---|---:|
| transform + plattice | 0.687 |
| fill | 4.121 |
| apply | 7.211 |
| **sieve chain, algebraic side** | **12.019** |

The full pipeline used real algebraic special-q from 15,000,000, both sides,
and ran 1,340 q. Its event-timed device accounting was:

| stage | ms / special-q |
|---|---:|
| transform + plattice, both sides | 1.027 |
| fill, both sides | 7.980 |
| apply, both sides | 12.000 |
| **sieve, both sides** | **21.010** |

TD + classify breakdown (the eight component rows sum to the printed device
total):

| stage | ms / special-q |
|---|---:|
| rank scan | 0.189 |
| emit `(x,a,b)` in rank order | 0.021 |
| survivor filter | 0.099 |
| resieve + scatter, both sides | 2.291 |
| norms + trial division, both sides | 0.429 |
| classify, both sides | 0.159 |
| joint accept + compact | 0.023 |
| record candidate factorisations | 0.525 |
| **TD + classify, device total** | **3.736** |

Pipeline device-accounted totals:

| stage | ms / special-q |
|---|---:|
| sieve, both sides | 21.010 |
| intersect + gcd | 0.114 |
| TD + classify | 3.736 |
| cofactor queues + relation readback/emit, device-accounted | 1.390 |
| **device-accounted total excluding cofactorisation** | **24.860** |
| **device-accounted total including cofactorisation** | **26.250** |

The printed `GPU-accounted / wall (excl cofac)` was **0.743**. That low ratio
is consistent with finding 53: the CUDA stages remain measurable under host
contention while wall time grows. The standalone and pipeline sieve totals are
not a one-side/two-side scaling A/B: they use different q, roots, norm scales,
and the pipeline adds the rational side.

## Not addressed in this round

> **This section is a snapshot of one round, not a current status list**, and it
> sits directly after finding 51 where it reads like one. Items superseded since
> are struck through with the date. For what exists *now*, read `STATUS.md`.

- ~~**The survivor-set gate passes on counts**~~ **DONE — 2026-08-03/04.** This
  item is superseded; see `../prototype.md`, "The gate was built and run" and
  "The gate at scale". Corrections to what is written above:
  - `las -batch-print-survivors` **does not** dump the `after_sieve` set. It
    emitted 1,851 records, not 797,028, because `needs_resieving()`
    (`las-siever-config.hpp:116`) returns false when any side has `lim == 0`,
    and the flag's output is the post-TD cofactor list. The set comparison
    required a 3-line CADO patch instead (`../oracle/cado-after-sieve-survdump.patch`).
  - **Strict set containment fails and that is expected**, not a defect:
    2,162 of las's survivors are absent from ours, attributed to `powlim`
    pinning, a boundary column at `b = I/2` where the two conventions sieve
    different half-open intervals, and an unexplained residue (side 1: 722,
    side 0: 266) that remains open as a diagnostic.
  - **Relation containment is the operative gate and it passes:** over the whole
    frozen band, **3,026 of 3,026** in-region las relations are survivors of
    ours, zero misses. The remaining 136 fall outside our sieve region because
    20 of the band's 67 lattices use a different (equally valid, in fact
    better-reduced) basis — see `../oracle/PARITY.md`.
  - What this establishes is **in-region sieve correctness, not yield
    equivalence.** Final relation yield must come from cofactoring our own
    region.
- ~~**The primitive-point filter is host-side only.**~~ **DONE.** The device
  intersection, compaction and gcd filter are built and gated.
- ~~**The primary post-sieve GPU path is unbuilt.** No GPU resieve/factor
  recovery, regular TD, or hard-cofactor stage consumes the compacted
  candidates.~~ **DONE — 2026-08-05/06.** Intersection, GPU trial division,
  classification, resieve and cofactorisation (rho and ECM) all exist, emit
  relations, and are gated by `cofcheck.sh` (28 cases) plus the post-cofactor
  reconstruction gate. The whole-band runs quoted in findings 48–51 are of that
  complete path. (The struck text went on to weigh borrowing YAFU's CUDA/OpenCL
  ECM kernels; we wrote our own instead, so that option is moot.)
- **Per-region offset hashes** are still not done — counts cannot see a
  permutation within a region, and the verify gate compares counts.
- **`--maxbits > 15` is now safe but untested**: the transform handles bucketed
  odd prime powers, and `g > 1` losses are reported, but no run has exercised
  it. The reported-loss counter is the thing to watch when someone does.
- The `g > 1` bucketed case emits nothing rather than routing to the small
  tier. Correct-and-reported, not correct-and-complete. Zero at the default
  `bkthresh`.
- The remaining items under "What is not yet measured" below.

## What is not yet measured

**This is a sieve measurement, not a relation-collection measurement, and the
distinction is load-bearing.** What runs end-to-end is: transform → fill →
apply → threshold → survivor list, per side. What does not exist at all is the
two-sided survivor intersection/compaction, GPU resieve/factor recovery/TD,
GPU cofactorization, final relation output, and unique-relation accounting.
The survivor *list* is also
capped at 2^22 entries against one-sided sets of 18–30M — the count is exact
and truncation is reported, but nothing consumes the list yet.

So: **kernel feasibility is demonstrated; GPU-resident relation-collection
feasibility is not.** Any "3–4× whole-box speedup" is specifically the optional
hybrid projection with a strong CPU cofactor path assumed unchanged, not a
measured relation rate and not an all-GPU estimate.

The remaining comparison constants and scope limits are:

| constant | status |
|---|---|
| GGNFS `N_eff` at 14–16 workers | **measured 2026-08-03: 10.24 at 16 workers** (finding 43). Was assumed 13. |
| GPU watts during the chain | **measured cleanly** in finding 44: 206.7 W stage-weighted board average, 13.306 board J/q for both sides. |
| CPU and DRAM watts under 16-worker GGNFS | **measured from Windows** in finding 44: 125.073 W CPU PPT + 5.222 W DIMMs; 158.731 W full component proxy including the idle GPU. |
| component energy comparison | **measured** in finding 44: CPU full q 44.609 J/q; GPU two-side sieve 16.874 J/q. Scope differs, so this is not yet end-to-end. |
| resieve/factor recovery | **accounted for only in the hybrid projection** inside GGNFS's retained TD phase; unimplemented and unmeasured on GPU for the primary path. |
| GPU hard cofactor stage | **preliminary only**: finding 47's external C164 `31/31` printed stages imply 386,084 inputs/s and 4.92 ms/q at the projected feed. C183 `31/32`, persistent latency, power, yield and host demand remain unmeasured. |
| whole-box wall watts | **unmeasured** — motherboard, drives, fans, VRM and PSU losses remain outside the HWiNFO component proxy. Required for the final economics verdict. |
| CADO post-sieve share (Gate 0) | unmeasured; useful for oracle workload and a CADO hybrid projection, not a prerequisite for the GPU-resident architecture. |
| the "200× root-transform speedup" (finding 4) | GGNFS's Sieve-Change timer also covers small-sieve setup, transformed-polynomial work and report-bound setup, so this is an **upper bound**, not a like-for-like stage comparison. |

Also unmeasured: target-equivalent persistent YAFU-derived GPU cofactor
throughput, coverage, power and recurring host demand; weak-host and multi-GPU
scaling; `bkthresh` sweep; I16e slabbing; throughput mode; production scales
2/4/8 (finding 23); and the per-q host-side small-FB transform and sort
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

**Current clean equal-work profile** — exact las scales, algebraic factor base
truncated at the special-q (GGNFS convention), and the cheap device parity
filter enabled. This is the profile used by findings 43–44:

```
cd bench && make
./bench --cadofb ../oracle/c183.fb1 --side 1 --scale 1.275 \
        --fbbound 120000053 --q 120000053 --rho 112625526 \
        --allowance 112 --not-both-even                    # 38.177 ms
./bench --side 0 --scale 1.925 \
        --q 120000053 --rho 112625526 --allowance 72.85 \
        --not-both-even                                    # 26.194 ms
```

For finding 44's power plateaus, add `--reps 2500` to side 1 and `--reps
3600` to side 0 while HWiNFO logs at two-second intervals. The benchmark times
transform, fill and apply as separate repetition blocks, so compute joules by
weighting each plateau by its printed stage time.

Defaults are `--mode atomic --record-bytes 4 --region 14 --apply-threads 512`
as of 2026-08-02; before that they were `twolevel` and `--region 15`, so
commands below that omit those flags reproduced a path that had already lost.

**`--reps 100` is the floor for any cross-machine comparison** (finding 48).
Below that the transform line reports amortized CUDA startup rather than
kernel time — it swings 98× between reps 3 and 1000 while fill and apply move
under 1%. The grid width now comes from `multiProcessorCount` and is echoed at
startup, so confirm the `grid: N SMs x 6` line matches the card before
comparing anything (finding 49).

**Cross-GPU profile** — the command all three cards in finding 50 ran:

```
./bench --poly ../oracle/c147.job --cadofb ../oracle/c147.roots1 \
        --logI 14 --J 8192 --reps 100
./bench --pipeline --cofactor --poly ../oracle/c147.job \
        --cadofb ../oracle/c147.roots1 \
        --logI 14 --qrange 15000000: --target-rels 100000 --relations OUT.dat
```

**This finding's workload is the C147, not the C183** — as are findings 48 and
54, the 144×256 vs 1152×32 geometry result, and the host-load experiment. Not
every timing in this file: findings 43–44 above profile the C183 via `--cadofb
../oracle/c183.fb1`. The two jobs are not interchangeable, so check which one a
command names before reusing its numbers.

`../oracle/c147.job` is tracked in git. `../oracle/c147.roots1` is 29 MB and
git-ignored — regenerate it with the `fbgen` command in `../oracle/README.md`,
which needs no CADO and reproduces the manifest-pinned file byte for byte.
**Pass `--maxbits 14`** to match this `--logI 14`; `fbgen` on its own defaults to
15 and produces a different factor base, and `bench` downgrades the mismatch to a
`note:` you will scroll past.

The standalone bench (no `--pipeline`) sieves one side at a fixed
`q=120000011`; the pipeline sieves both sides across a real band. They agree
once transform is excluded: A100/5070 on standalone fill+apply is **1.69×**
comparing like reps (both at `--reps 3`: 15.605/9.236), or 1.66× against the
5070's `--reps 100` row in finding 50's table (15.60/9.40). Pipeline sieve is
**1.68×**. The reconciliation holds either way; the residual spread is apply's
reps drift, not a disagreement between the two harnesses.

**The one CPU-only gate.** `fbtest` is the only thing here that touches
neither the GPU nor `nvcc`:

```
./fbtest --cadofb ../oracle/c183.fb1
```

`make check` is **not** a substitute and is **not safe alongside a running
job**. It used to be exactly the line above; it is now `check: all cofcheck`
(`Makefile:76`), so it compiles the whole CUDA path and then runs `cofcheck`
*on the GPU*. That change was deliberate — the CUDA path could previously fail
to compile while the gates still reported "all gates passed" — but it means
`make check` now contends for the card.

Nor is "CPU-only" the same as "safe on a busy box". Finding 53 measures host
contention at an **18.4–22.3% relation-rate loss** with every `cudaEvent`
timer still flat within 1%, so a parallel `nvcc` build both slows the running
job and silently corrupts any timing that job reports. On a box that is
sieving, run neither.

**GPU gates** (these run kernels — do not use them to check a busy box):

```
./bench --verify --logI 12 --J 512 --region 12   # correctness vs CPU reference
./bench --verify --region 14 --mode atomic --record-bytes 4 --apply-threads 512
./bench --mode atomic --record-bytes 4 --region 14 --apply-threads 512   # best
./bench --mode twolevel                          # the fill variant that loses
./bench --region 14 --norm const                 # isolates norm-init cost
./bench --region 14 --apply-mode plain           # isolates smem atomic cost
./bench --region 15 --cells 8                    # prices the unsafe byte cell
```
