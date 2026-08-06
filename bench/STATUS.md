# STATUS — what exists now

`RESULTS.md` and `../prototype.md` are lab notebooks: they record findings in
the order they were discovered, including the ones later refuted, because the
refutations are the most useful part. That makes them bad at answering "what
does this thing do today". This file answers only that, and holds nothing that
is not current. **Last updated 2026-08-06.**

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
  on the rented boxes (`ERR_NVGPUCTRPERM`).
- **Finding 51's Ada-vs-Blackwell block response was an artifact.** It held
  `--threads` at 256; at 32 the 4090's degradation reverses to improvement and
  all three cards behave alike. See finding 52.
- **rel/J is flat between the 5070 and 5090** — 28.1 vs 29.5 on sampled board
  power, i.e. a tie, with the 5090 1.68× faster. The Ada 4090 is the outlier at
  18.1. Generation separates these cards; width does not.
- **Superseded:** an earlier rel/J table built on *nameplate* TDP concluded the
  design "does not want wide expensive GPUs". It is retracted — see the
  RETRACTED block in `RESULTS.md`. Do not quote the old numbers.

Measured end-to-end on the 5070 (c147 band, 1340 q, same binary, geometry the
only variable): fill −6.1%, apply +0.06%, wall −2.1%, relations byte-identical.
Projected but **not** measured end-to-end: the same A/B on the 5090, where the
standalone gain is largest (16.7%), and `1152 × 256` — the one cell that would
show whether `--fill-threads` is needed at all.

## Known defects

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale is a second, unchecked approximation. This is the
  main known yield loss.
- **Duplicate share 15.8–25%** from the full special-q-side factor base, at
  ~1.34× the downstream TD/cofactor work for the same unique yield. Whether
  truncating at `q` wins is untested — it also shortens the q range needed.
- **`k_fill_l1` (twolevel path) has never been swept** at any geometry. It
  takes an explicit `--fill-blocks` but defaults to its own 144 × 512, because
  the 1152 × 32 result was measured on `k_fill_atomic` — a different kernel
  with a different write pattern.
- **Power is board-only.** The metric of record is whole-box
  relations/sec/watt; host draw is unmeasured. The A100 has no sampled power.

## Open experiments, in order

1. **Concurrent-q throughput.** Two independent fill workspaces, two q or two
   sides in separate streams, 144 blocks each, sweep 1/2/4. This is the
   decisive test for whether wide cards are a poor fit or are simply being fed
   too little independent work. **Design it at the new knee**: the old framing
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
   transform/fill/resieve — no rebuild, no re-upload.
4. **Host cost.** Each side allocates five temporaries and does a full stable
   sort per q to form three modulus tiers; a three-way partition with reusable
   scratch is sufficient. The same small ideals are then transformed again for
   TD. ~1.7 ms/q is in reach before any worker-thread pipeline is needed.
5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job.
6. **Whole-box power**, to close the metric of record.
