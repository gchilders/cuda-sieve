# Running a job

Every command here was run end to end on a c123 (factored) and a c151 (sieved
and verified), 2026-08-05. Nothing in it is from memory.

## What you need

1. **A polynomial file.** A GGNFS `.job` works directly — `n:`, `skew:`,
   `c0:`..`c5:`, `Y0:`, `Y1:`. The extra keys (`rlim:`, `lpbr:`, `alambda:` …)
   are ignored by both our loader and CADO's `makefb`, so `result.job` needs no
   conversion. A CADO `.poly` works too.

2. **An algebraic factor base from CADO's `makefb`.** This is required, not
   optional: the GGNFS `.afb.0` format carries neither p = 2 nor prime powers,
   and without p = 2 the algebraic cofactors stay even, which Montgomery
   arithmetic cannot split. `bench` refuses to produce relations without it.

   ```sh
   makefb -poly JOB.job -lim ALIM -maxbits LOGI -out JOB.roots1
   ```

   `-maxbits` is the sieve's `logI`. 5 s for a c151.

## Where to put things

`makefb` and `bench` take paths and do not care about the working directory.
**msieve does** — it looks for `cub/`, the lanczos kernels and its default
`msieve.dat` / `msieve.fb` / `worktodo.ini` relative to wherever you launch it.
So: one directory per job, and run msieve from inside it.

```
~/nfs/c151/
  c151.job                 copy of result.job — the poly
  c151.roots1              makefb output
  msieve.dat               relations: sieve STRAIGHT into this, no copy step
  msieve.fb                N, poly and bounds (see below)
  worktodo.ini             N on one line
  cub -> ~/msieve_nfsathome_basement/cub
  lanczos_kernel.ptx    -> ~/msieve_nfsathome_basement/lanczos_kernel.ptx
  lanczos_kernel.fatbin -> ~/msieve_nfsathome_basement/lanczos_kernel.fatbin
  sieve.log
```

Run `makefb` and `bench` from anywhere with `-out` / `--relations` pointing
into the job directory; run `msieve` from inside it.

Two practical notes:

- **`--relations msieve.dat` writes directly**, saving a full duplicate of the
  relation file — ~8 GB on a c151 at 65M relations (~128 bytes each). Budget
  ~15 GB per job including msieve's `.mat` / `.cyc` / `.chk` / `.dep`.
- **`bench` writes `NAME.part` and renames on success.** If a long run is
  interrupted, `mv msieve.dat.part msieve.dat` keeps everything written so far.
  Note it OVERWRITES rather than appends, so if you sieve in several runs, send
  them to separate files and `cat` them together. `--target-rels` is there so
  you can do it in one.

## Sieving

```sh
bench --pipeline --cofactor \
      --poly JOB.job --cadofb JOB.roots1 \
      --logI 14 \
      --qrange 15000000: --target-rels 65000000 \
      --relations ~/nfs/c151/msieve.dat
```

That is the whole command. **`rlim`, `alim`, `lpbr`, `lpba`, `mfbr`, `mfba`
and both lambdas are read from the `.job` file**, and the byte scale and
survivor allowance are derived from the polynomial. Each one is printed with
its source, so a run says where every number came from:

```
  job file: rlim 16700000, alim 33500000, lpbr 29, lpba 30, mfbr 56, mfba 59
  job file: alambda 2.5 -> 62.50 bits (side 1), reported only; the allowance is derived below
  job file: rlambda 2.4 -> 57.60 bits (side 0), reported only; the allowance is derived below
params from q=15000017: side 1 log2(maxnorm)=... scale=... allowance=<derived> bound=...
```

**The lambda lines and the allowance are unrelated numbers** — the example
above used to print `allowance=62.50`, the same value as `alambda`'s, which
read as the job file's lambda flowing through into the bound. It does not; see
"Lambda: we ignore it, on purpose" below. The `reported only` suffix is what
the binary actually prints, and it is there to stop exactly that misreading.

Anything you state explicitly wins over the file; anything the file carries
wins over the derivation. A CADO `.poly` carries none of these keys, so there
you either state them or let them derive.

- **`--logI`** is the siever width: `gnfs-lasieve4I14e` → 14. **`--J` defaults
  to `2^(logI-1)`** (CADO's convention), so it no longer needs passing.
- **`--region`** must be ≤ `--logI`; the default is 14, so only pass it for
  I=13 or smaller (`--region 13`).
- **`--qrange MIN:`** with the upper end omitted means "up to the factor-base
  bound". Pair it with **`--target-rels N`** to stop when you have enough
  rather than guessing a range. Checked at flush boundaries, so it overshoots
  by well under 1% on any real target.
- Side 1 is the algebraic side and carries the special-q. Side 0 is rational.

### Progress output

One `\r` line, updated every 30 s, reported against whichever goal is in force:

```
q=15020857  1222 q  78251 rel +111390 cand  2608 rel/s  26.1%  ETA 0h 01m
```

With `--target-rels` the percentage and ETA track the relation target — which
is what you want, because `--qrange MIN:` makes the q count meaningless (it is
the whole factor base, 1,088,865 special-q for the c151). Without a target they
track the q count instead.

**`+N cand` is the cofactorisation queue's occupancy, not pending relations.**
Roughly two thirds of those records become relations on the SNFS job, and the
band summary reports the two separately. It is on the line because it is the
only field that advances on *every* q: `rel` cannot move until a flush, and a
flush is 131,072 candidates — about 67 q on the c183, but 686 q (~40 s) on the
SNFS job, which enqueues 191 records per q instead of 1,956. Before the first
flush the rate and ETA print as `-- rel/s` and `ETA --h --m`, because they are
unknown rather than zero; `+N cand` climbing is how you tell a healthy run from
a stalled one in that window. `--relations` also stages to `NAME.part` until the
band ends, so `ls` shows nothing during it either.

The relation figure advances at queue flush boundaries rather than per q, so it
moves in steps and the rate is slightly understated early in a run. It settles.

### Special-q on the rational side (SNFS)

An SNFS polynomial often has tiny algebraic coefficients — `x^5 + x^4 - 4x^3 -
3x^2 + 3x + 1` for a `p^11 - 1` job — so the difficulty sits on the *rational*
side, and the special-q and the 3LP `mfb` go there with it. Pass `--sq-side 0`:

```sh
bench --pipeline --cofactor --poly snfs236.job --cadofb snfs236.roots1 \
      --sq-side 0 --logI 14 --qrange 30000000: --relations msieve.dat
```

`--cadofb` is still the **algebraic** base — that side needs prime powers
regardless of where the q lives. `makefb -lim` takes `alim` as usual.

Two things that look like errors and are not:

- **"f has no root mod 2"**. That polynomial has `f(0) = 1`, `f(1) = -1` and an
  odd leading coefficient, so 2 can never divide the algebraic norm and makefb
  correctly emits no `p = 2`. The run says so and continues. The `p = 2` check
  only fires when f *does* have a root mod 2 and the entry is missing anyway.
- **`--qrange MIN:` resolving to an empty band** when you also lower `--rlim`
  below `MIN`. The open upper bound follows the *special-q side's* limit, which
  is `rlim` under `--sq-side 0` and `alim` otherwise.

### Lambda: we ignore it, on purpose

**The survivor allowance is derived, not imported.** A `.job` file's lambda is
reported and *not applied*, because it is calibrated to GGNFS's survivor gate
and that calibration does not transfer — see the measurements below. CADO's
automatic rule is not a better source either: on the SNFS job it gives 97.3
bits where GGNFS's gives 91.8, looser still.

What we use instead is `mfb` plus the slack *our own* approximation needs: one
byte unit is `1/scale` bits, so the slack is `max(1.5, 2/scale)`. Measured:

| job / side | job file says | we derive | survivors | relations |
|---|---:|---:|---:|---:|
| SNFS, both sides | 61.10 / 91.80 | **60.50 / 89.50** | −20% | −0.04% |
| c183, side 0 | 68.1 (CADO rule) | **61.5** | −42% | −0.19% |

A fifth to two fifths of the trial-division input, for two hundredths of a
percent of the relations.

Overrides, in precedence order: `--allowance` / `--allowance0` in bits wins
outright; `--lambda0` / `--lambda1` opts back into CADO's rule; otherwise the
derivation above. `bench` warns if an override is more than 2 bits looser than
what it would have derived.

One trap if you calibrate by hand: **a single special-q is not enough of a
sample.** On the c183's parity q the tighter bound looks like it costs 1
relation in 37 (2.7%); over 120 special-q the real rate is 0.19%.

### For reference: GGNFS and CADO do not mean the same thing either

You should not need this now that the lambda is not applied, but the two
conventions differ by which quantity they multiply:

| | lambda is in units of | cofactor bits allowed |
|---|---|---|
| **GGNFS** | `log2(lim)` | `lambda * log2(lim)` |
| **CADO** | `lpb` | `lambda * lpb` |

The GGNFS test-sieve's own suggestion confirms it: *"Suggested rlambda: 2.32
(mfbr=56 / log2(rlim=16700000))"* — i.e. GGNFS's lambda is `mfb / log2(lim)`.
CADO's automatic is `0.3 + mfb / lpb`.

**To state one yourself, use `--allowance` / `--allowance0`, in BITS.** That is
the quantity both conventions are really expressing, and it is what the
survivor bound is computed from:

```
bound = (uint32_t)(allowance * scale + 1)
```

(CADO writes this as `(unsigned char)`, which is right for its 8-bit cells. We
sieve 16-bit cells with CINIT 4096, so bounds above 255 are legal here and the
cast has to be wider — an earlier version of this file, and the banner, both
printed the 8-bit truncation.)

Worked example, the c151 above (`alambda 2.5`, `alim 33.5M`, `lpba 30`):

```
GGNFS:  2.5 * log2(33500000) = 2.5 * 25.00 = 62.5 bits   -> --allowance 62.5
CADO :  the same 62.5 bits would be lambda = 62.5/30 = 2.083
CADO's automatic (0.3 + 59/30 = 2.267) would give 68.0 bits — more generous
```

and the rational side (`rlambda 2.4`, `rlim 16.7M`, `lpbr 29`):

```
GGNFS:  2.4 * log2(16700000) = 2.4 * 23.99 = 57.6 bits   -> --allowance0 57.6
```

The byte scale is always derived from the polynomial (CADO's formula,
`las-norms.cpp:237`) and **an explicit `--allowance` / `--scale` overrides
it**, so the two compose: derive what you don't know, state what you do.

If you would rather work in CADO's units, `--lambda1` / `--lambda0` take
CADO-style lambdas and 0 means "use CADO's automatic". They override a `.job`
file's GGNFS-unit lambdas, and say so when they do.

> `--auto-params` used to gate the derivation. It is now the default and the
> flag is accepted and ignored, so old scripts keep working; drop it when you
> next touch them.

## Running on a different GPU

The build ships native cubins for sm_120, sm_89, sm_86 and sm_80, plus
compute_80 PTX. Anything from Ampere onward runs without editing the Makefile.

Two cases still want a Makefile line. **Older than sm_80** needs its own
`compute_XX`, because the driver only JITs PTX to a target **at or above** the
virtual arch — a PTX fallback never reaches down. **Newer than sm_120, or
sm_90 (H100)**, runs from PTX, and that JIT costs seconds on a cold
`~/.nv/ComputeCache` and repeats every process start if the cache is disabled
or evicted. It is far larger than the ~200 ms of module load the `--reps 100`
floor below is calibrated against, so add a native `-gencode` for any card you
intend to publish numbers for.

Grid width comes from `multiProcessorCount`. Every run opens with the line it
derived, on stdout, including when `--blocks` overrides it:

```
grid: 48 SMs x 6 = 288 blocks (NVIDIA GeForce RTX 5070, 48 MB L2)
grid: 1152 x 32 for fill (absolute, not per SM)
```

**Check that line first**, and check it names the card you meant. The SM count
is queried at runtime, so it always matches the device actually in use — a
wrong number means the wrong GPU was selected, not a stale binary. A *stale*
binary shows up as the line being **absent**.

There is no `cudaSetDevice` and no `--device` flag: device 0 is always used.
On a multi-GPU box select another card with `CUDA_VISIBLE_DEVICES=1` and
confirm it on this line.

**Fill has its own grid AND its own block width, and neither scales with the
card.** `--blocks` is 6 per SM; fill instead uses a fixed **1152 blocks of 32
threads** (`--fill-blocks` / `--fill-threads` to override). A 48-SM 5070, a
128-SM 4090 and a 170-SM 5090 all reach the knee at that same absolute
geometry. Above it every card is flat, so overshooting is nearly free; below it
the cost is steep (288 blocks is 15–38% worse), so undershoot is the expensive
mistake.

**`--fill-threads` is separate from `--threads` on purpose.** Fill wants many
narrow blocks; transform, intersect, TD, resieve and the cofactor kernels are
all tuned at 256. At a constant 576 blocks the 5090 measures 2.711 ms at 128
threads against 3.147 at 256 — **16%** — so tuning fill through `--threads`
would have cost five other stages to buy one.

Don't expect a faster card to fix a slow fill: a 5090 with 3.5× the 5070's
hardware still returns far less than that ratio on this stage. The geometry was
measured at one job shape (8192 buckets, 77.4M records) and plausibly moves
with bucket and record count, so sweep it when characterising a new job — and
sweep **both** axes, because holding one fixed is exactly how the old 144 × 256
default was arrived at:

```sh
for t in 32 64 128 256; do
  for b in 576 1152 2304 4608; do
    printf "fill %4s x %-3s " $b $t
    bench --poly JOB.job --cadofb JOB.roots1 --logI 14 --J 8192 \
          --reps 100 --stage fill --fill-blocks $b --fill-threads $t | grep "fill:"
  done
done
```

Note `--blocks` and `--threads` do **not** move fill — the grids are fully
separate, and the startup lines print both. See RESULTS.md finding 52 (and 51,
which it supersedes on the geometry).

Three rules for comparing cards, all learned the hard way (RESULTS.md
findings 48–53):

- **Run on an IDLE host, and say whether you did.** GPU kernel times come from
  `cudaEvent`, so they are blind to host contention: saturating this box's 16
  cores left `fill` and `apply` flat within 1% while wall clock went **24.30 →
  31.27 ms/q**. As throughput that is a **22.3% relation-rate loss**, and half
  the cores already costs **18.4%**, so there is no safe headroom. A busy box therefore reports *perfect* kernel numbers and a
  bad ETA — which is exactly what a card looks like when it is fine and the
  host is not. Never compare a wall-clock or ETA figure across boxes without
  knowing the host load on both; rented and shared boxes are the risk.

  **The pipeline** prints `GPU-accounted / wall (excl cofac)` for this — the
  standalone does not, so a wall-clock or ETA claim has to come from a pipeline
  run. There is no universal "good" value: a faster card spends relatively more
  of its wall on the same host work and so reads *lower* while perfectly
  healthy. Take an idle baseline per card, job and band length, and compare
  against that; a drop against your own baseline is the signal, the absolute
  number is not. (Published values are pending a re-measurement — the first
  pair was taken with a formula since corrected. See RESULTS.md finding 53.)

- **Use `--reps 100` or more.** Below that, the standalone bench's `transform`
  line reports amortized CUDA startup rather than kernel time — it moves 98×
  between `--reps 3` and `--reps 1000` on the same hardware, while `fill` moves
  0.9% and `apply` 2.9%. A stage whose time depends on `--reps` is not
  measuring the kernel. Quote `apply` with its reps setting; `fill` is safe
  without one.
- **Compare stages, never the total.** `fill` and `apply` respond to completely
  different limits — fill to integer throughput, apply to FP32 and memory
  bandwidth — and cards rank differently on each. A 3090 loses `fill` to a 5070
  and wins `apply` against it. `SIEVE CHAIN` averages that away into a number
  that cannot be reasoned about.

The pipeline prints the same three stages under `sieve, both sides`, which is
the honest source for transform: there it runs once per q against a warm
context, the way production does.

## Checking the output

```sh
bench --check-relations ~/nfs/c151/msieve.dat --poly JOB.job --lpb 30 --lpb0 29
```

Rebuilds both norms from `(a,b)` and the polynomial for every relation and
requires each recorded factor to divide exactly, both norms to reduce to 1, and
every prime to sit within its side's lpb. Run it before spending hours on
filtering — it catches a truncated factor, a dropped power of two, or a
composite emitted as prime, none of which any other check sees.

## Post-processing with msieve

Relations come out in GGNFS/msieve format already — `a,b:rational:algebraic`,
hex factors, rational side first, special-q included. So:

```sh
cd ~/nfs/c151          # relations are already here, sieved straight into msieve.dat
cat > msieve.fb <<EOF
N <decimal N>
SKEW <skew>
R0 <Y0>
R1 <Y1>
A0 <c0>
...
A5 <c5>
FRMAX <rlim>
FAMAX <alim>
SRLPMAX <2^lpbr>
SALPMAX <2^lpba>
EOF
echo "<decimal N>" > worktodo.ini

# msieve looks for these RELATIVE TO THE WORKING DIRECTORY
ln -s /path/to/msieve/cub .
ln -s /path/to/msieve/lanczos_kernel.ptx .
ln -s /path/to/msieve/lanczos_kernel.fatbin .

msieve -s msieve.dat -l msieve.log -nc1 -v    # filtering
msieve -s msieve.dat -l msieve.log -nc2 -v    # linear algebra (GPU)
msieve -s msieve.dat -l msieve.log -nc3 -v    # square root -> factors
```

There is **no `N` header line** in `msieve.dat`; N comes from `msieve.fb`.

We generate **no free relations**. CADO made 122,390 for the c123 and msieve's
filtering still cleared its target without them, so they are not load-bearing at
that size. Watch for it on larger jobs.

If `-nc1` reports *"filtering wants N more relations"* and singleton removal
collapses the set (e.g. 8.6M relations against 15.0M unique ideals reducing to
279), that is simply not enough relations — sieve more, do not tune.

## Measured

| | c123 | c151 |
|---|---:|---:|
| logI | 13 | 14 |
| lpb (alg / rat) | 29 / 28 | 30 / 29 |
| **ms per special-q** | **10.14** | **24.49** |
| relations / q | 272.1 | 70.62 |
| relations / sec | 26,839 | 2,883 |
| special-q used | — | 949,331 of 1,088,865 |
| relations | — | 67,043,952 |
| reconstruction gate | all pass | 48,420 / 48,420 |

The c151 column is a **complete band**, not a sample. The earlier 24.75 ms and
79.8 rel/q came from the first few thousand special-q; over the full run to
`--target-rels 67000000` the yield falls to 70.62 rel/q, because yield decays
as q climbs and the band consumed 87% of the available special-q. Size a job
off a full-band figure, not off a test sieve at the bottom of the range.

The c123 factored end to end: 1,093 s sieving, 171 s filtering, 61 s linear
algebra, 153 s square root. The c151's relations built a matrix in msieve.

### SNFS, special-q on the rational side

`376364081347875370546831^11 - 1`, difficulty 235.76, `--sq-side 0`, I14,
`mfbr 88 / lpbr 31` (3LP on the rational side). Against `gnfs-lasieve4I14e` at
the same q = 30M, with the rational factor base matched to GGNFS's truncated
one (1.92M vs its 1.86M primes):

| | GGNFS, 1 core | this, GPU |
|---|---:|---:|
| ms per special-q | 919 | 38.84 |
| relations / q | 40.8 | 46.14 |
| **ms per relation** | **22.52** | **0.842** |

**26.7× per relation against one core**, so roughly 3.3× against eight. Same
order as the c151's 2.03×, which is the reassuring part — the rational-side
path is not a special case that happens to look good.

**Relation sets compared directly**, same q range `[30000000, 30001003]`,
GGNFS run locally rather than quoted (`gnfs-lasieve4I14e -v -n0 -c 1000 -f
30000000 -o rels.out -r result.job`, 62 special-q, 2531 relations, 1.127 s/q):

| | count |
|---|---:|
| GGNFS relations | 2,531 |
| ours | 3,357 |
| **in both** | **2,531** |
| **GGNFS found, we missed** | **0** |
| ours only | 826 |

We are a strict superset: **zero misses against a real GGNFS run**, and 3,357
of 3,357 pass `--check-relations`. That is the correctness result and it
stands.

**The 826 extras are mostly duplicates, not extra yield.** GGNFS's `lowering
FB_bound to 29999999` is not a limitation — it is deliberate duplicate
avoidance. A relation whose special-q-side norm has primes p₁ < p₂ is found at
q = p₁ *and* again at q = p₂. Truncating the sq-side base at the current q
means only the largest of those primes ever finds it, so GGNFS collects each
relation exactly once. We do not truncate, so we collect it repeatedly.

Measured, not assumed:

| | |
|---|---:|
| mean sq-side primes in [30M, rlim] per relation | 1.82 |
| P(re-found when that prime is sieved as q) | 71.8% (n = 1,023) |
| **finds per unique relation** | **1.34** |
| **duplicate share of raw output** | **~25%** |

Calibrated against ground truth: msieve's filtering of the **c151** found
10,594,292 duplicates in 67,165,877 relations — **15.8%**, 1.187 finds per
unique. Back-solving that run gives P(re-find) = **0.728**, against the 0.718
measured directly on the SNFS job, so the model holds across two independent
jobs.

**Confirmed directly, 2026-08-07**, by replaying both corpora off disk instead
of back-solving — no GPU. On the c151 the replay finds **10,594,292**
duplicates, digit-for-digit the count above, at 1.1877 finds per unique.

**But over a different total, and that is not yet explained.** The paragraph
above quotes msieve's count over 67,165,877 relations; the file on disk holds
67,043,952, which is also what the run record at the top of this document says.
Those 121,925 lines make the duplicate share 15.77% by msieve's pair of numbers
and 15.80% by the replay's. An exact match on the *count* across different
totals is what you would see if the extra lines were all unique — free
relations being the obvious candidate — which would validate the
duplicate-detection logic and leave only the denominator open. Until someone
checks that, this is a strong agreement, not an exact reproduction:

| | measured here | n | this table |
|---|---:|---:|---:|
| mean sq-side primes, [30M, rlim], SNFS | 1.7918 | 19.6M rel | 1.82 |
| P(re-found), SNFS job | **70.2%** | 1,089,564 | 71.8% (n=1,023) |
| P(re-found), c151 | **72.3%** | 13,709,863 | 0.728 (back-solved) |

Two traps, both of which bit on the way to those numbers. **The raw
slot-take rate is biased low** — a relation is in the corpus only because one
slot already hit, so subtracting that forced hit from both numerator and
denominator understates P; at k = 2 the correction is exact, `P = 2r/(1+r)`,
turning a raw 54.1%/56.6% into 70.2%/72.3%. And **the band must come from the
run log, not from the corpus**: an ordinary prime p divides ~N/p relations, so
on a 67M-line corpus every prime near 1M looks like a special-q. Taking the
c151's band as [1M, 33.5M] rather than [15M, 33.5M] drops P to 18.6%, with
whole separation buckets reading exactly 0.0% — the tell that the low-end
"q" were never sieved at all. The 1,088,865 count above is the cross-check.

P(re-found) also decays with separation, gently: 74% at q₂/q₁ ≈ 1.04 down to
61% by q₂/q₁ ≈ 2 (bias-corrected, c151). Treating it as one constant is fine
for a whole-band estimate and wrong for a narrow A/B.

(An earlier version of this table said 37%, from the naive
`1 + (mean-1) x P`. That is wrong: a relation with k eligible primes still only
needs ONE hit to appear at all, so the correct form is
`P / sum_k (f_k/k)(1-(1-P)^k)` with the population recovered as `n_k ~ f_k/k` —
sampling relation LINES over-weights relations that were found often.)

The re-find probability is 72% rather than 100% because (a,b) must also land
inside the sieve rectangle of the *other* prime's lattice, which is a different
lattice. Confirmed directly: relations found at q ≈ 30.0M reappear when
[31004593, 31107961] is sieved as special-q.

So over a full run we do ~1.34× the trial division and cofactorisation for the
same unique yield, and our raw relation counts are inflated by the same factor.
**Quote unique relations, or quote raw and say so.** GGNFS's 40.8 rel/q are
unique; the 46.14 rel/q in the table above are not, so the honest comparison is
~34.4 unique/q against GGNFS's 40.8 — at matched factor bases GGNFS collects
*more* unique relations per special-q than we do, because it collects each one
only once. What we buy for that is not needing to sieve the full q range to
find a relation whose largest sq-side prime sits above `qmax`.

That last clause is the whole risk in truncating, and it is now measured. Under
truncation a relation is found at its **largest** sq-side factor-base prime, so
if the band stops below that prime it is not deduplicated — it is never found.
On the c147's as-run band ([15.00M, 15.15M], 0.4% of `alim`) the `mfb` ceiling
alone says truncation would find **nothing at all for 22.68%** of unique
relations. Truncation is yield-neutral only over a band reaching `lim`.

And the dedup it buys is capped by `mfb` headroom, because an unsieved prime in
`(q, lim]` does not disappear — it lands in the cofactor. Measured floors over
a band reaching `lim`: **−15.0%** of emissions on the c151 and **−17.3%** on
the c147 (both `mfb` 59), but only **−6.1%** on the SNFS job (`mfbr` 88). The
rest of the way down to 1.00× is the survivor gate's to give, which makes this
one experiment with "our gate is looser than GGNFS's" below, not two.

The `ms per relation` row above is therefore **raw**, not unique. Against
unique relations it is ~1.13 ms, so ~20× one core rather than 26.7×. And the
GPU was shared with another job when the 38.84 ms/q was taken, so treat the
timings here as indicative until re-measured on a quiet card.

### Our survivor bound is looser than GGNFS's at the same lambda

Do not carry GGNFS lambda intuition across. Sweeping the SAME q range on the
SAME job, `gnfs-lasieve4I14e` loses 17.3% of its yield going from 91.8 to 87.5
bits; we lose 0.07% going from 91.8 to 88.0:

| bits | GGNFS | ours |
|---|---:|---:|
| 91.8 | 2,531 | baseline |
| 87.5 / 88.0 | 2,092 (−17.3%) | −0.07% |
| 83.7 / 85.0 | 1,952 (−22.9%) | −6.25% |

GGNFS submits 978 cofactors per special-q (`COF: 60664 tests`, 62 q); we submit
1,426 for comparable unique yield. So at a nominally identical bound we admit
noticeably more, and the surplus is almost all unproductive — which is why
tightening is free for us and expensive for GGNFS. **We are paying in time, not
in relations.** Why the two bounds differ at matched nominal bits is not
understood yet; see the tasks. Until it is, tune `--allowance` by measuring
this tool, not by translating a lambda that worked in GGNFS.

3LP is real and load-bearing here — over a 178-q band at q = 20M, large primes
per relation on the rational side came out 0LP 1,261 / 1LP 3,803 / 2LP 4,009 /
**3LP 1,183**, i.e. 11.5% of relations need the third prime. The algebraic side
tops out at 2LP exactly as `mfba 59 / lpba 30` predicts. All 10,256 relations
passed `--check-relations`.
