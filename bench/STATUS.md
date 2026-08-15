# STATUS — what exists now

`RESULTS.md` and `../prototype.md` are lab notebooks: they record findings in
the order they were discovered, including the ones later refuted, because the
refutations are the most useful part. That makes them bad at answering "what
does this thing do today". This file answers only that, and holds nothing that
is not current. **Last updated 2026-08-14.**

## Architecture

One process, both sides, one special-q at a time:

```
per special-q:
  host    reduce the q-lattice, build the three modulus tiers
  device  k_transform    plattice transform of the factor base
          k_fill_atomic  bucket-sieve into regions
          k_apply        norm init + bucket add + threshold scan  ->  survivor bitmap
  (repeat for the other side)
  device  k_intersect_compact   both-sides bitmap AND, gcd filter, rank scan
          trial division, classification, resieve
          cofactorisation (Pollard rho and ECM)
  host    emit relations
```

Two sides run **sequentially through one shared bucket allocation**. There is
no stream concurrency and no second workspace. Every timing in `RESULTS.md` is
one-q-at-a-time.

## Current size limits, and what lifting them entails

This is a design assessment, not implemented work. The production path still
refuses inputs outside the limits below; there is no unsafe override.

| quantity | current limit | immediate reason |
|---|---:|---|
| sieve area `I*J` | `2^31` positions | the exclusive end `xmax` and the hot plattice walks use `uint32_t` |
| `lpb` | 32 | each split large prime and unsplit small residual is stored in one `uint32_t` |
| `mfb` | 96 | the production cofactor queue narrows residuals to `mz<3>`, three 32-bit limbs |
| large primes per side | 3 | `ceil(mfb/lpb) <= 3` is checked before the run |

The area and cofactor limits are independent even though an A=32 job such as
AS276 needs both lifted (`I=2^16`, `J=2^16`, `lpba=35`, `mfba=101`).

### A=32: representation is the first blocker, memory is the second

Every valid position in a `2^32`-position rectangle still fits in a
`uint32_t`; the value that does not fit is the exclusive endpoint `xmax =
2^32`. The current fill and resieve loops compare a 32-bit walk position with
that endpoint, and their next-position arithmetic wraps at the top of the
range. Merely deleting the front-end check therefore makes an empty or
incorrect sieve.

Two implementation levels are plausible:

1. **Whole-area, large-memory path.** Keep stored positions and bucket-local
   offsets 32-bit, add an explicit full-range/end-of-walk representation, and
   avoid promoting every operation in the hot walk to 64 bits. This is the
   smaller change, but the complete allocation is only comfortable on a card
   with substantially more than 16 GB.
2. **Slabbed path.** Process A=32 as two or more A<=31 slabs, reusing the
   bucket array and survivor bitmaps. A correct version needs persistent
   per-prime walk positions, global row offsets in norm and small-sieve
   calculations, slab-local rank/resieve state, and global coordinates in the
   candidate queue. This is the portable solution for a 12 GB card and the
   more invasive pipeline change.

The measured memory model is linear in area for the two large allocations:

| area | bucket array | three survivor bitmaps | subtotal |
|---:|---:|---:|---:|
| `2^31` | 5.53 GB | 0.81 GB | 6.34 GB |
| `2^32` | 11.06 GB | 1.61 GB | 12.67 GB |

Fixed allocations plus the CUDA/driver footprint put the full A=32 projection
at roughly **14–16 GB**. Work that touches positions or bucket records is
also approximately linear in area: A=32 is about 2x A=31, and 8x the A=29
C183 rectangle, per special-q. Relations per q rise too, so q/s alone is not
a useful comparison. A stateful slab implementation should add only launch
and boundary bookkeeping beyond that intrinsic work; this has not been built
or measured.

### LPB and MFB are separate widths

`lpb` bounds an *individual resulting prime*. `mfb` bounds the *composite
residual sent to cofactorisation*. Raising one does not inherently require
raising the other:

- Raising `lpb` above 32 requires a two-word or `uint64_t` representation for
  split primes, unsplit prime residuals, sorting, emission, primality checking,
  and relation reconstruction. It does **not** require four-limb rho/ECM when
  `mfb <= 96`.
- Raising `mfb` above 96 requires `mz<4>` arithmetic even when the actual
  maximum is only 101 bits. Rho, ECM, probable-prime testing, GCD, and exact
  division all pay the wider modular-arithmetic cost.

The useful dispatch is fixed-width kernels selected on the host, independently
for each side:

| side's `mfb` | cofactor type |
|---:|---|
| `<= 64` | `mz<2>` |
| `<= 96` | `mz<3>` |
| `<= 128` | `mz<4>` |

The underlying `mz<L>` arithmetic is already templated. What is hardcoded
today is the production queue/storage and its launches at `mz<3>`. Runtime
variable-length loops inside a kernel are not the intended design: they would
lose compile-time unrolling and likely increase register pressure. Per-side
host dispatch captures the obvious win because current jobs do not use 3LP on
both sides. The criterion is nevertheless the bit bound, not the label: a 2LP
side with `mfb=65` still needs three limbs.

This makes `lpb=33` with `mfb=64/95` a materially smaller extension than
`lpb=35` with `mfb=64/101`: the former needs 64-bit factor outputs but retains
two- and three-limb cofactor kernels; the latter needs a four-limb kernel on
the 101-bit side. Both ratios still ask for at most three large primes, so
`CF_MAXFAC` is not the blocker.

The output-width change itself should be cheap. The TD factor lists remain
32-bit because factor-base primes remain 32-bit; only the much smaller residual
and split-prime arrays need widening. The expensive part is any extra limb in
the repeatedly executed modular arithmetic.

There is one direct measurement of the per-side dispatch opportunity. Widening
the formerly two-limb rational queue to three limbs changed that queue from
2.15 to 3.94 ms/q (**+83%**) while changing the then-current whole pipeline
from 172.65 to 173.75 ms/q (**+0.64%**). Narrow arithmetic matters greatly to
the stage that uses it, but the easy side was only about 1% of wall time. That
makes per-side dispatch sensible work while the queue is already being changed,
not a high-value standalone optimisation.

### Performance accounting

Three-to-four-limb Montgomery multiplication expands the CIOS inner product
from `3*3 = 9` to `4*4 = 16` limb multiply-adds. That suggests roughly a
1.8–2x cofactor-stage cost from width alone, subject to register pressure. A
quoted **5x** is not a four-limb estimate: it combines the width cost with a
pessimistic rho workload in which the factor being sought is three bits larger
and therefore takes about `sqrt(8) = 2.8x` as many expected iterations. ECM
also pays for four-limb operations, but does not have rho's square-root scaling;
the rho/ECM crossover for the new cofactor population is unmeasured.

On the current measured profile, cofactorisation is 14.37 of 98.26 ms/q
(14.6%). If the *entire* stage changed and everything else remained fixed:

| cofactor-stage multiplier | total ms/q | whole-pipeline slowdown |
|---:|---:|---:|
| 1.8x | 109.8 | 11.7% |
| 2x | 112.6 | 14.6% |
| 3x | 127.0 | 29.3% |
| 5x | 155.7 | 58.5% |

That table is an Amdahl illustration, not a projection for AS276. The larger
job has a different area, cofactor distribution, stage mix, and yield. The 5x
case would normally apply only to its hard four-limb subset, not every queued
record.

The final metric is **time to a filterable matrix**, not raw relations/s:

```
new time / old time
  = (new ms/q / old ms/q)
  * (new required usable relations / old required usable relations)
  / (new usable relations/q / old usable relations/q)
```

LPB directly expands the possible large-prime ideal universe and therefore the
column and singleton burden during filtering. As a scale indication only,
`pi(2^(b+d))/pi(2^b) ~= 2^d*b/(b+d)`: 32 -> 33 is about 1.94x and 32 -> 35
about 7.31x. Those are **not** required-relation multipliers—a run does not
sample the entire prime universe—but they show why extra raw yield cannot be
credited without filtering. MFB does not directly enlarge the ideal universe
at fixed LPB, but admitting more and larger 3LP products can still raise the
required surplus.

### Work status and rough scope

| item | status | rough engineering scope, including GPU validation |
|---|---|---:|
| A=32 whole-area endpoint/walk support | not started | about 1 week |
| A=32 stateful slabbing for 12 GB | not started | about 2–3 weeks |
| 64-bit large-prime outputs and gates | not started | part of the LPB/MFB work |
| per-side `mz<2>` / `mz<3>` / `mz<4>` dispatch | arithmetic templates exist; queue/launch work not started | about 1--2 weeks with LPB widening |
| target cofactor corpus, rho/ECM crossover, and filter test | not captured or measured | required before a performance claim |

Together, a robust slabbed and width-dispatched implementation is roughly a
**3–5 focused engineering-week** change, not counting delays obtaining the
target workload or GPU access. A whole-area path restricted to >=24 GB cards
would be appreciably smaller. These are source-review estimates, not measured
schedules.

## Validated

| | what | where |
|---|---|---|
| **Golden suite** | `./cofcheck.sh` — 30 cases on c183 q=120000053, exact relation counts, not ranges | `cofcheck.sh` |
| **Reconstruction gate** | every emitted factor divides, is prime, is within lpb, and both norms rebuild to exactly 1 | `--check-relations` |
| **Transform gates** | root transform against its definition, over a real factor base | `fbtest` |
| **Band runs** | c147: 1340 q → 159,837 relations, both sides PASS | 5070, 5090, 4090, A100 |
| **Real job** | snfs236, ~20M relations before deliberate interruption | 5070 |

Cards with measured band data: **RTX 5070** (WSL2), **RTX 5090**, **RTX 4090**,
**A100 80GB** (native Linux), and an **RTX 3090** via an external reporter.

`--check-relations` runs on a machine with no GPU.

## Measured, and what it means

- **Fill does not scale with the GPU.** 5090 has 3.5× the SMs of a 5070 and
  returns far less than that on fill, against 3.33× transform and 2.01× apply.
  All three swept cards reach the same **absolute knee at 1152 blocks × 32
  threads**, flat above it. Mechanism unresolved; the leading candidate is now
  **work granularity** (fine chunks balance the tail), which fits the plateau.
  Both L2 stories are out: capacity was already dead, and one geometry fitting
  48/72/96 MB of L2 argues against write-combining decay too. `ncu` is blocked
  on the rented boxes (`ERR_NVGPUCTRPERM`), but **one local 5070 profile now
  exists** — `work/c147/fill_5070.ncu-rep`, a single `k_fill_atomic` launch.
  It corroborates granularity and kills the bandwidth stories outright:

  | | |
  |---|---:|
  | DRAM throughput | 12.83% |
  | L2 hit rate | 96.45% |
  | L2 (max bandwidth) | 52.11% |
  | L1/TEX hit rate | 18.91% |
  | bytes used per 32-byte load sector | 8.7 |
  | bytes used per 32-byte store sector | 4.6 |
  | stall cycles on L1TEX scoreboard | 157.7 of 191.0 (82.6%) |
  | SM busy | 3.75% |
  | waves per SM | 1.00 |
  | SM active / elapsed cycles | 8.00M / 10.88M |

  Read together: it is not DRAM-limited (12.8%), not L2-capacity-limited
  (96.5% hit — independent confirmation that story was dead), and **not at an
  L2 bandwidth ceiling either** (52%). It is latency-bound on uncoalesced
  scatter — 82.6% of stall cycles waiting on L1TEX, at 4.6 useful bytes per
  32-byte store sector. That last number matters for item 11: the prior-art
  claim that scatter tuning here "dies at the L2 transaction ceiling" does not
  describe this kernel, and ~7× store-sector waste is headroom, not a wall.

  **Waves per SM = 1.00 is the granularity mechanism made visible.** The whole
  grid is resident at once, so a block that draws a heavy chunk has no queued
  block to backfill its slot — and SMs sit idle for 26.5% of elapsed cycles,
  which is that tail. More, smaller blocks fix it twice over: finer chunks
  shorten the tail, and more resident warps hide the 157-cycle L1TEX stall.

  **Caveat: this profile is at 288 blocks × 256 threads** — the pre-finding-52
  geometry, and 256 threads is precisely the held-fixed axis that made
  finding 51 an artifact. So it characterises a configuration the project has
  abandoned. The 288-vs-1152 A/B **at 32 threads** is still unrun, and is
  still what would settle the mechanism rather than merely fit it.
- **Finding 51's Ada-vs-Blackwell block response was an artifact.** It held
  `--threads` at 256; at 32 the 4090's degradation reverses to improvement and
  all three cards behave alike. See finding 52.
- **rel/J is flat between the 5070 and 5090** — 28.1 vs 29.5 on sampled board
  power, i.e. a tie, with the 5090 1.68× faster. The Ada 4090 is the outlier at
  18.1. Generation separates these cards; width does not. **This tie is an
  artifact of board-only power and probably does not survive the metric of
  record.** First wall reading on the local box (2026-08-07, UPS, monitors
  excluded): **395–400 W whole-box against 165 W GPU board**, so the card is
  ~42% of the box and host overhead is roughly as large as the GPU's own draw.
  Host overhead is approximately *fixed*, so the faster card amortises it over
  more work: `whole-box rel/J = board rel/J × W_gpu / (W_gpu + W_host)`. With
  the host constant now measured at **~105 W idle / ~115 W with the sieve's own
  core** (item 6), that puts the 5090 at ~23.5 against the 5070's ~17.8 — the
  5090 **ahead by ~1.3×** rather than tied. The tie was an artifact of pricing
  only the card. Caveat: this places both cards in *this* box; the 5090 numbers
  came from a rented machine whose own host draw is unknown and probably
  higher, and per the next bullet that can never be measured.
- **The cross-card rel/J table can never be restated on the metric of record.**
  Whole-box power needs a physical meter on the box, and the 5090/4090/A100
  numbers came from rented machines. Those rows are permanently board-only.
  What *is* answerable with a meter is the thing the project was chartered to
  answer — the local 5070 against the local CPU box, item 0 — because both
  sides of that comparison are on this UPS. Treat cross-card rel/J as an
  architecture note, not as a verdict input.
- **Superseded:** an earlier rel/J table built on *nameplate* TDP concluded the
  design "does not want wide expensive GPUs". It is retracted — see the
  RETRACTED block in `RESULTS.md`. Do not quote the old numbers.

Measured end-to-end on the 5070 (c147 band, 1340 q, same binary, geometry the
only variable): fill −6.1%, apply +0.06%, wall −2.1%, relations byte-identical.
`--fill-threads` is measured to be necessary, not assumed: at the shipped 1152
blocks the 5090 costs **23%** at 256 threads (3.239 ms) against 32 (2.633), so
raising the block default alone would have left that on the table.

Projected but **not** measured end-to-end: the same pipeline A/B on the 5090,
where the standalone gain is largest (16.7%).

## Known defects

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale is a second, unchecked approximation. This is the
  main known yield loss.
- **Duplicate share 15.8–25%** from the full special-q-side factor base, at
  ~1.34× the downstream TD/cofactor work for the same unique yield. Whether
  truncating at `q` wins is untested — it also shortens the q range needed.
  **These are whole-FB-band numbers.** Measured raw inflation runs 1.0021× on
  the c147's 0.4%-wide band to 1.1877× on the c151's full-FB band, so the
  defect scales with band coverage rather than being a fixed property of the
  design; and the share truncation can mechanically remove is capped by `mfb`
  headroom, which the c183's `mfba 92` makes generous. See item 3.
- **`k_fill_l1` (twolevel path) has never been swept** at any geometry. It
  takes an explicit `--fill-blocks` but defaults to its own 144 × 512, because
  the 1152 × 32 result was measured on `k_fill_atomic` — a different kernel
  with a different write pattern.
- **Power is board-only.** The metric of record is whole-box
  relations/sec/watt; host draw is unmeasured. The A100 has no sampled power.
- **Host contention costs up to 29% of wall clock, invisibly.** Saturating the
  CPU leaves every `cudaEvent`-timed kernel flat within 1% while wall goes
  24.30 → 31.27 ms/q; half the cores already costs +22.6%. All GPU counters we
  have are blind to it, so a busy box reports perfect kernel times and a bad
  ETA. Stated as throughput that is an **18.4% relation-rate loss at half load
  and 22.3% at full** — not the 22.6/28.7% wall figures, which are
  percent-of-baseline and overstate the bar for co-scheduling. **Any cross-box
  wall-clock or ETA comparison is invalid without knowing host load on both.**
  The pipeline prints `GPU-accounted / wall (excl cofac)` to expose it; the
  first published pair used a formula since corrected, so current values await
  a confirmed-idle box. Finding 53 is canonical for all of this.

## Open experiments, in order

**This list is the task list of record.** Session-local task trackers do not
survive; if work is worth returning to, it belongs here. Two lists drifted
apart once already — items 7–9 below existed only in a chat session and were
absent from every document in the repo.

0. **The verdict band** *(added 2026-08-06, Fable review)*. The question the
   project was chartered to answer — unique relations/sec/watt on the **c183**
   against the measured CPU box — has never actually been run; every
   end-to-end number so far is a proxy job (c147, c151, snfs236). The CPU side
   is already frozen: 3.12 core-s/q at q=130M and N_eff 10.24 at 16 workers
   give **0.295 s/q for the whole box**, so green is ≤ 0.54 s/q and beating
   the CPU outright on component-proxy perf/watt is **≤ 0.179 s/q end to end**
   (prototype.md, "The GPU's per-special-q budget").

   **Energy bar from measured wall power** *(2026-08-07)*. Those s/q bars were
   derived without a meter. Item 6 now has one, including the host constant
   (~105 W), so the comparison can be made in joules per special-q rather than
   in seconds against an assumed wattage. **Assuming the 16-worker CPU
   baseline is this same box** — it is a 9800X3D, 8 cores / 16 threads, so a
   16-worker sweep fits it, but confirm it:

   Both sides' wattage is now measured (item 6). The GPU side draws ~270 W
   whole-box; the CPU side draws 220 W and its throughput was measured in the
   same session. But **which CPU throughput** turns out to be the whole
   question:

   | CPU comparator | J/q | GPU at 70–90 ms/q × 270 W | **margin** |
   |---|---:|---:|---:|
   | `I14e`, measured | 37.4 | 18.9–24.3 J/q | **1.5–2.0×** |
   | `I15e`, measured | 104.5 | 18.9–24.3 J/q | **4.3–5.5×** |
   | `I15e`, finding 43 | 64.9 | 18.9–24.3 J/q | 2.7–3.4× |

   **The headline result swings 3× on a sieve-geometry choice, and nothing in
   this file currently pins it down.** The GPU runs `--logI 14` with `--J 8192`
   — the `I14e` geometry, `2^14 × 2^13`. Finding 43's CPU baseline used
   `I15e`, `2^15 × 2^14`, four times the sieve area. If the GPU is graded
   against the `I15e` row while sieving an `I14e` rectangle, the verdict
   flatters the GPU by up to 4× and the project's chartered number is wrong in
   the direction nobody would catch by re-reading it. Finding 43 says its GPU
   timing was "on the equal-work profile", so this may already be handled —
   but the words are not enough, and **confirming the geometry match is now
   the first thing item 0 has to do**, ahead of the bands, the meter, or
   anything else here. The 4× is larger than every other effect in this file
   combined.

   Note also that `I14e` and `I15e` are not interchangeable for the CPU: the
   larger area wins 2.2× the relations per special-q (68.9 vs 31.6) while
   costing 2.9× the time, so `I14e` is the better rel/J (0.846 vs 0.659) and
   `I15e` the better relations-per-q-range. A real c183 job picks `I15e` or
   above. Which of those the GPU should be held to is a question about the
   deployment, not about the hardware.

   The c183's own standalone
   sieve measures 38.2 + 26.2 ms for the two sides (RESULTS.md "Reproduce",
   side 1 truncated at q), so with TD and the heavier mfba-92 3LP cofactor
   load the pipeline projects to roughly **70–90 ms/q — about 2× inside the
   harder bar**. That is a projection, not a measurement, and nothing measured
   contradicts it — which is exactly why the run should happen now: it is a
   wall-plug meter and a weekend, not a project. Needs: the meter (item 6);
   bands at q ≈ 50M / 130M / 190M for the yield drift; relations
   **deduplicated before quoting** (raw is inflated 1.19–1.34×, see RUNBOOK);
   an idle host per finding 53. Fold item 3's full-vs-truncated A/B into the
   same bands — the FB convention moves both the yield accounting and the
   downstream load, so it should be settled in the graded configuration.
   **But not in these bands as specified.** Three narrow probes at 50M / 130M
   / 190M are the right shape for yield drift and the wrong shape for the FB
   convention: a band covering a fraction of a percent of `lim` has almost no
   duplicates to remove (measured 1.0021× on the c147), and under truncation
   the relations whose re-finding q lies outside the probe are lost outright
   rather than deduplicated. Grade perf/watt on the three probes; settle the
   FB convention on one contiguous band wide enough to contain both q of a
   duplicate pair, or by replaying the attribution offline as item 3 now does.
1. **Concurrent-q throughput.** Two independent fill workspaces, two q or two
   sides in separate streams, sweep 1/2/4. This is the decisive test for
   whether wide cards are a poor fit or are simply being fed too little
   independent work. **Design it at the new knee** — the "144 blocks each" this
   item used to specify is the pre-finding-52 geometry and would reproduce the
   flaw described next: the old framing
   ("two 144-block fills vs one 288-block fill") is now measured entirely below
   saturation, where any configuration scales, so it would have credited
   streams for reaching a thread count one kernel already reaches. Compare
   2 streams × 1152 against 1 kernel × 2304 instead. The old note here said the
   two architectures predict differently — Blackwell's flatness fitting "idle
   capacity" and Ada's degradation not. That distinction is gone: at 32 threads
   all three cards flatten, so one prediction covers them all.
2. **Startup fill autotune.** `--fill-threads` is done; the autotuner is not.
   Sweep both axes, since holding one fixed is how the old default was reached.
   Opt-in for the standalone benchmark, where reproducibility is the point;
   default-on is defensible for `--pipeline`, where a per-job knee cannot be
   derived and 15 trial fills cost ~60 ms of a multi-hour run.
3. **Full vs q-truncated factor base**, compared at equal *deduplicated* yield.
   `nactive = lower_bound(primes, q)` on the already-sorted base, passed to
   transform/fill/resieve — no rebuild, no re-upload. The standalone harness
   already truncates statically (`--fbbound`, `fb_restrict` at
   `bench_main.cu:1252`); what is missing is making that bound *per-q* in the
   pipeline, where all three kernels still take the full `fb->n`
   (`pipeline.cuh:221`, `:227`, `:644`).

   **Measured 2026-08-07 over relations already on disk, no GPU.** Checked
   against ground truth on the c151: the tool finds **10,594,292 duplicates**,
   digit-for-digit the count msieve's own filtering reported (`RUNBOOK.md:461`).
   **The denominators do not match, though** — msieve quotes that count over
   67,165,877 relations, while the file on disk holds 67,043,952, which is also
   what this repo's own run record says (`RUNBOOK.md:399`). The 121,925 gap
   makes the ratios 15.77% and 15.80% rather than one number. An exact match on
   the duplicate count across different totals is what you would see if those
   extra lines were all unique — free relations are the obvious candidate — so
   the duplicate-detection logic looks validated and the *denominator* is what
   remains unreconciled. Do not quote this as "reproduces msieve exactly" until
   it is.

   | corpus | band as run | band / lim | `mfb` | raw inflation | truncation floor |
   |---|---|---:|---:|---:|---:|
   | c147 | [15.00M, 15.15M] | 0.4% | 59 | **1.0021×** | −17.3% |
   | snfs236 (partial) | [30.0M, 36.97M] | 5.2% | 88 | **1.0391×** | −6.1% |
   | c151 (complete) | [15.0M, 33.5M] | 55% | 59 | **1.1877×** | −15.0% |

   ("truncation floor" is over a band reaching `lim`; see point 3.)

   Three things follow, and two of them cut against doing this first.

   1. **The duplicate share is not a constant of the design — it is set by how
      much of the factor base the q band covers.** The 15.8–25% figure quoted
      under "Known defects" is a *whole-FB-band* number. A narrow band has
      almost nothing to deduplicate, so any A/B run on one will measure noise.
   2. **Truncation is yield-neutral only if the band reaches `lim`.** Under
      truncation a relation is found at its *largest* sq-side FB prime; if the
      band stops below that prime, it is never found at all. On the c147's
      as-run band the mfb ceiling alone says truncation would find **nothing**
      for 22.68% of unique relations. That is outright loss, not deduplication.
   3. **How much truncation actually deduplicates is bounded by `mfb`
      headroom, and on the c183's configuration that headroom is large.** An
      unsieved prime in `(q, lim]` does not vanish — it lands in the cofactor.
      On the snfs236 (`mfbr 88`, 3LP) it fits there comfortably, so over a
      hypothetical full band [30M, rlim] truncation is guaranteed to remove
      only **6.1%** of emissions against the full base's 1.7918× — the rest of
      the way down to 1.0× depends entirely on whether the survivor threshold
      refuses those relations. The c183 carries `mfba 92`, i.e. *more*
      headroom still.

   So the honest range for truncation on a c183-shaped job is "removes between
   6% and ~44% of downstream work", and **where it lands inside that range is
   item 7's question, not this one.** Items 3 and 7 are one experiment: our
   gate is the loose one, and a loose gate is exactly what lets a truncated
   base keep re-finding the duplicates truncation was supposed to remove.
   Run them together, on a band wide enough to have duplicates to remove.

   The 1.7918× on the snfs236 independently reproduces the 1.82 mean sq-side
   primes in `RUNBOOK.md:444`, and **P(re-found) now agrees across both jobs
   measured directly rather than back-solved**: 70.2% on the snfs236
   (n = 1,089,564) and 72.3% on the c151 (n = 13,709,863), against RUNBOOK's
   71.8% at n = 1,023 and its 72.8% back-solve. Two cautions for anyone
   redoing this:

   - The raw slot-take rate is **biased low** — a relation is in the corpus
     only because one slot already hit, so subtracting that forced hit from
     both numerator and denominator understates P. At k = 2 the correction is
     exact (`E[copies | copies ≥ 1] = 2/(2−P)`, so `P = 2r/(1+r)`); it is what
     turns the raw 54.1% / 56.6% into 70.2% / 72.3%.
   - **Detect the band from the run, not from the corpus.** Counting hits per
     prime does not separate band q from ordinary primes at the low end: an
     ordinary p divides ~N/p relations, so on a 67M-relation corpus every
     prime near 1M clears any fixed threshold. Doing that put the c151's band
     at [1M, 33.5M] instead of [15M, 33.5M] and produced a P(re-found) of
     18.6% — the false low-end "q" can never re-find anything, so whole
     buckets read exactly 0.0%. Cross-check against the q count in the run log
     (`RUNBOOK.md:110`: 1,088,865 q for the c151, i.e. π(33.5M) − π(15M)).
4. **Host cost — overlap first, graphs second, micro-optimisation last.**
   Identifiable host work is 1.694 ms/q, only **7%** of a 24.30 ms idle wall,
   but it triples under CPU contention (finding 53). It is three different
   problems and they do not have the same fix:

   Two thirds of it is *prep* (host per-q tables/staging, small-prime tables)
   and one third is *launch + sync overhead* (`host: unaccounted`). **Finding
   53 holds the canonical table** — do not copy it here; it has drifted once
   already.

   1. **Overlap the prep with GPU execution.** Both prep terms depend only on
      the q-lattice, not on any GPU result, so q+1's host work can run during
      q's kernels. This is double-buffering, **not threading**: 1.166 ms of
      prep against ~20.7 ms of GPU work per q fits entirely inside the GPU's
      shadow with 18× room to spare, so perfect overlap takes it to *zero* on
      the critical path. Threading the same work would reach maybe 0.4 ms with
      four threads and leave it *on* the critical path — strictly worse, for
      more code and a synchronisation problem we do not currently have.
   2. **CUDA graphs for the per-q kernel sequence.** `unaccounted` is
      wall-minus-device inside TD/classify: the CPU issuing launches and
      waiting on syncs. Overlap cannot help — it is interleaved with GPU
      execution by nature — and it is the term that grew **443%** under load,
      so it is what makes the box fragile. The per-q sequence is fixed, so it
      can be captured once and replayed.
   3. **Then make the prep cheaper** — three-way partition replacing the per-q
      stable sort, fusing the small-ideal transform that is currently done
      twice, and **hoisting the per-q allocations**: `pipeline.cuh:169-183`
      mallocs and frees five temporaries (`idx/tp/trt/tg/tlp`) per side per q,
      i.e. 10 malloc/free pairs per special-q, plus three memcpys back over the
      pinned arrays. Reusable scratch removes that churn, and it is the part a
      better sort algorithm does *not* fix. **This was previously the whole of this item and it is the least
      valuable third of it**: after (1) the work is hidden in the GPU shadow
      and costs nothing regardless of how fast it is.

   **Priority note, 2026-08-07 (owner).** Sharing the box is a **requirement,
   not a robustness nicety**: a sieve that only performs on an idle host is
   much less useful, because the host is a desktop someone wants to use. That
   moves this item up and changes what "done" means — the target is that the
   GPU does not wait on the CPU, not that host work is small on a quiet box.

   **Do the free experiment before any of the code below.** The host thread is
   not obviously starved of CPU *time* — measured 2026-08-07 it holds 92.9% of
   one core, because CUDA's default sync spins. What it competes for is the
   *moment* it needs to issue the next launch, and at equal priority it loses
   that race as often as it wins. Two zero-code levers:

   - **Renice the competing work.** Raising your own processes' nice value
     needs no privileges. With the sieve at nice 0 and everything else at 19,
     CFS weights the sieve ~1024 to 15 and it wins essentially every contest.
   - **Leave headroom, but know what you are leaving.** This box is 8 physical
     cores / 16 threads, so 14 `ecm` workers plus one spinning sieve thread is
     15 of 16 *threads* and **every physical core is loaded** — there is no
     idle core to fall back on, and the sieve thread shares a core with an ECM
     worker whatever you do. Dropping batch work to `nproc - 2` frees two
     threads, i.e. one core's worth of SMT capacity, not a quiet core. That is
     why the renice lever matters more here than on a wide box: on 8 cores you
     cannot isolate the sieve by subtraction, only by priority.

   **The live instrument is GPU utilisation, not the summary.**
   `GPU-accounted / wall` only prints at the end of a run, but `nvidia-smi`
   reads out continuously, and a host that is starving the GPU shows up as
   utilisation below ~95%. Observed 88–89% during the co-scheduled snfs236 run,
   i.e. the GPU had nothing to do about a tenth of the time. If renicing moves
   that number toward the mid-90s, the contention penalty is a scheduling
   problem and the code below is not urgent. If it does not, the dependency is
   architectural and this item is the top of the list.

   One caution retained: this interacts with item 1: two concurrent q roughly
   **doubles host work per unit time**, so on a contended host that experiment
   can come back negative for host reasons unrelated to the GPU. Run it on a
   verified-idle box and check `GPU-accounted / wall` first.
5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job. Note the payoff is job-shaped: for the snfs236 polynomial the
   special-q side is degree 1 with `d = [1, -3.5e-24]`, so largest-term and
   true-rectangle are the *same number* and this buys nothing there. The ~2-bit
   gap is a GNFS-side problem.
6. **Whole-box power**, to close the metric of record — the meter half of
   item 0. **The meter exists**: the UPS reports real watts. It is not a
   parallel task to item 10, it is a **prerequisite** for it (see there).

   Readings (2026-08-07, UPS, real watts). Monitors measured once at **~45 W**
   and subtracted throughout; the raw readings were taken with them on.

   | state | monitors off | notes |
   |---|---:|---|
   | **everything idle** | **95–115 W** | GPU P8 ~16 W, load avg 1.06 |
   | CPU full (14 `ecm`), GPU idle | 240 W | ≈9.6 W per busy core |
   | CPU full, GPU sieving | 395–400 W | GPU board 165 W |

   | derived | |
   |---|---:|
   | **host constant** (idle box incl. P8 GPU) | **~105 W** |
   | GPU **delta**, at the wall | 155–160 W |
   | GPU **delta**, board sensor (165 − 16) | ~150 W |
   | implied delivery loss | 3–7% |
   | power factor (W ÷ VA) | 0.96 |

   The idle reading is **jumpy, ±7%** — normal for a desktop drifting through
   C-states — so average it rather than taking an instant. Everything derived
   from it inherits that spread.

   **The board sensor is trustworthy.** Compare deltas to deltas: the card
   swings 150 W on the board sensor and 155–160 W at the wall, so the wall
   figure sits a few percent *above* the board figure, which is exactly what
   PSU and delivery losses predict. First independent check this project has
   had on the number every rel/J figure rests on, and it passes.

   **The GPU never reaches zero.** It idles at 15 W in P8 and it drives this
   box's displays (`Disp.A = On`), so there is no configuration of this machine
   where the card is absent. That 15 W belongs in the host constant on *both*
   sides of the item-0 comparison, which is the honest accounting anyway — the
   CPU box needs display output too.

   Cautions. **Read W, not VA** — VA is apparent power, ~4% high here. **Subtract
   the monitors** once and for all. And **none of these is a sieve-only number**:
   the ECM job is in all of them, and per finding 53 it is simultaneously
   costing the sieve throughput, so both sides of rel/J are wrong here. These
   are the *co-scheduled* data, worth having, but not the verdict data.

   **Elastic background load is still a trap**, even though it did not bite
   here. ECM expands to fill whatever cores are free, so stopping the GPU job
   handed it the sieve's spinning core and it drew *more* CPU power — the 240 W
   row therefore contains somewhat more CPU work than the 395–400 W row does,
   which understates the GPU delta by roughly the draw of one core. The
   agreement above survives that because the residual is ~10 W against a 150 W
   signal. It would not survive it on a smaller difference. Pin the background
   worker count across both readings, or stop it.

   **The host constant is now measured, which was the term every projection in
   this file was missing.** At ~105 W it is roughly a third of a sieving box,
   not the 200 W+ the earlier ranges allowed for.

   **The CPU side is measured too**, on the c183 at q=130M, 16 workers, all
   monitors-off and steady-state (2026-08-07):

   The box is a **9800X3D: 8 physical cores, 16 threads, 96 MB L3**, so "16
   workers" is two per physical core and the per-worker column below is per
   *thread*, not per core.

   | load | whole box | per worker |
   |---|---:|---:|
   | idle | ~105 W | — |
   | 14 `ecm` workers | 240 W | 9.6 W |
   | 16 `gnfs-lasieve4I14e` | **225 W** | 7.5 W |
   | 16 `gnfs-lasieve4I15e` | **220 W** | 7.2 W |
   | GPU sieving alone (derived) | ~270 W | — |

   **Power falls as the workload gets more memory-bound**, which is exactly why
   ECM was a bad proxy: it is ALU-bound and drew 33% more per core than the
   siever it was standing in for. The earlier ~260 W estimate scaled from ECM
   was ~18% high. Use 220 W for the CPU side.

   Throughput measured in the same session, so both sides of the CPU figure
   come from one box on one day:

   | siever | ms/q | rel/q | rel/s | J/q | rel/J |
   |---|---:|---:|---:|---:|---:|
   | I14e | 166.2 | 31.6 | 190.4 | **37.4** | 0.846 |
   | I15e | 474.8 | 68.9 | 145.0 | **104.5** | 0.659 |

   **I15e measures 475 ms/q here against finding 43's 295 ms/q** — same siever,
   same 16 workers, same stated q, 1.6× apart. A worker sweep run to chase it
   eliminated every mechanism except one:

   | workers | finding 43 | measured 2026-08-07 |
   |---:|---:|---:|
   | 8 | 2.120 q/s | 1.140 q/s |
   | 16 | 3.392 q/s | 2.106 q/s |
   | SMT gain | 1.60× | **1.85×** |

   Not cache capacity — 16 workers beat 8 by 1.85×, so SMT is working and the
   96 MB L3 is not the binding constraint at I15e (a tempting hypothesis on an
   X3D part, and wrong). Not thermal — cores held 4691 MHz under sustained
   load. Not startup — these are steady-state windows. The slowdown is
   *uniform across worker counts*, and contention or thermal effects would bite
   harder at 16 than at 8, not equally. **That leaves configuration**: finding
   43 does not record which `input.job` it swept, and this run used the c183
   (`alim 134.2M, mfba 92, alambda 3.5, lpba 32`) at 68.9 rel/q. A job yielding
   fewer relations per q would sieve faster per q and prove nothing.

   So the two numbers are probably measuring different work, and **s/q is the
   wrong axis to compare them on — rel/J is the right one**. Resolve by
   recording the job file alongside any future N_eff sweep.
7. **Why our survivor gate is looser than GGNFS's at matched lambda.** Same q
   range, same job, same nominal bits: `gnfs-lasieve4I14e` loses 17.3% of its
   yield going 91.8 → 87.5 bits where we lose 0.07% going to 88.0. We submit
   1,426 cofactors per special-q against GGNFS's 978 (`COF: 60664 tests`, 62 q)
   for comparable *unique* yield. So we admit ~46% more and the surplus is
   almost all unproductive — time, not relations. **This is also the other
   half of item 3**: the dedup a q-truncated base buys is exactly the set of
   relations the gate refuses once their `(q, lim]` primes move into the
   cofactor, so a loose gate spends the truncation prize before it is
   collected. Measured, truncation is guaranteed only 6.1% of the available
   ~44% on an `mfb 88` job; the gate decides the rest. Do not run these two
   apart. Until the cause is known the
   bound can only be set by sweeping, never derived for a new job. Candidates:
   a systematically low per-position norm estimate; the byte scale cancelling
   only if cell init and bound use it consistently; GGNFS's lambda gating more
   than the report threshold. **Do not "fix" it by tightening the default** —
   the current gate finds every relation GGNFS finds (verified, zero misses
   over 2,531), so any trade of yield for speed must be made deliberately.
8. **A = 32 sieve areas and LPB/MFB widening**, for jobs like AS276. All three
   independent blockers remain: the `2^32` exclusive position endpoint, the
   14–16 GB whole-area footprint, and the `lpb 35 / mfb 101` factor/cofactor
   widths. The current design assessment, performance accounting, alternatives,
   and work status are consolidated in **"Current size limits, and what lifting
   them entails"** above; that section is canonical rather than duplicating a
   moving design here.
9. **Dead factor-base parses under `--sq-side 0`.** `fb1` is loaded and
   `fb_fill_logp`'d purely as the throwaway first parse that used to supply the
   q list, but under `sq_side 0` the band comes from the rational base instead
   and `fb1` is untouched until the derivation frees and reloads it. Separately
   `rfb_build` runs twice over `rlim` — a full sieve to 134.2M plus a modular
   inverse per prime, each time. ~15–20 s of startup on snfs236. Irrelevant to
   a multi-day run, worth fixing before anything that restarts the process in a
   loop (parameter sweeps, `cofcheck`).
10. **GPU power-limit sweep** *(added 2026-08-06; numbered out of order —
    items 1–9 keep their numbers because findings cite them; in priority this
    slots directly after item 0)*. Cheap, and it moves the metric of record
    directly: consumer cards ship past their efficiency knee, and a board cap
    at 60–80% of stock often buys 15–30% rel/J for single-digit rel/s — the
    curve is the measurement. The 5070/5090 rel/J tie (28.1 vs 29.5) was taken
    at stock; a capped curve may separate them and should raise both. Sweep
    the cap in ~20 W steps over a fixed c147 band, plot rel/s and rel/J, pick
    the knee; report the item-0 verdict at stock AND at the knee.
    **Grade this on wall power, not board power — item 6 first.** Capping cuts
    board watts but lengthens the run, and the ~115 W host constant (item 6) is
    paid for that whole extra time. It is ~37% of a sieving box, and that is
    enough to change the answer, not merely shade it. Worked example, a cap
    trading 15% throughput for 40% less board power, using item 6's measured
    115 W host and 155 W card:

    | | stock | capped | rel/J |
    |---|---:|---:|---:|
    | board only | 155 W | 93 W | 0.85 / 0.60 = **+42%** |
    | whole box | 270 W | 208 W | 0.85 × 270/208 = **+10%** |

    So the board sensor reports a **4× larger win than the wall meter sees**.
    Both are gains here, so the sign does not flip — but a sweep graded on the
    board will keep capping well past the point where the wall stops improving,
    and will pick a cap that is too low. Under WSL2
    set the cap from the Windows side (`nvidia-smi -pl` as administrator) —
    the WSL-side tool can read power but generally cannot set limits.
11. **Apply breakdown** *(added 2026-08-06)*. Fill got findings 48–52's
    attention because it scales worst across cards, but apply is where the
    milliseconds are: 9.6 ms of the c147's 24.3 ms wall against fill's 7.1,
    and it also scales poorly (2.01× on a 5090 with 2.67× the 5070's
    bandwidth). `--norm const` and `--apply-mode plain` already isolate
    norm-init and smem-atomic cost; one session splitting apply into
    norm / bucket-add / threshold-scan says whether a second material win
    exists or the stage is at its memory-system limit. Do this before any
    further fill tuning — prior art
    (`~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`) says scatter micro-tuning
    on this GPU dies at the L2 transaction ceiling. **The ncu profile under
    "Measured" narrows this**: fill is not at an L2 bandwidth ceiling (52% of
    max) and wastes ~7× on store sectors, so the prior-art warning may not
    transfer. Take the same profile of `k_apply` before assuming it does.
12. **Unattended operation — a job queue** *(added 2026-08-07, owner-stated
    requirement; design not yet scoped)*. The GPU should not idle waiting on
    a human any more than it should idle waiting on the host thread; item 4 is
    the within-job half of this and this is the between-job half. Known
    concrete pieces, pending a decision on what the queue should actually be:

    - A job list consumed without intervention, so finishing one job starts
      the next rather than leaving the card idle overnight.
    - **Resume.** The snfs236 restart was hand-built — the current process was
      launched with `--qrange 36965783:` derived by hand from where
      `msieve.dat.1` stopped. That step has to become automatic before a queue
      means anything, and it is the piece most likely to silently lose or
      duplicate relations if it is got wrong.
    - **Item 9 stops being cosmetic.** Its ~15–20 s of redundant startup is
      noise in a multi-day run and real overhead in anything that restarts the
      process per job — which is exactly what a queue does.
