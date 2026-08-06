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
  returns 1.20× on fill, against 3.33× transform and 2.01× apply. All swept
  cards saturate at the same **absolute 144 blocks**. Above saturation
  Blackwell is flat, Ada degrades ~38%. Mechanism unresolved; the surviving
  candidate is L2 write-combining decay, and it is a candidate, not a
  conclusion. `ncu` is blocked on the rented boxes (`ERR_NVGPUCTRPERM`).
- **rel/J is flat between the 5070 and 5090** — 28.1 vs 29.5 on sampled board
  power, i.e. a tie, with the 5090 1.68× faster. The Ada 4090 is the outlier at
  18.1. Generation separates these cards; width does not.
- **Superseded:** an earlier rel/J table built on *nameplate* TDP concluded the
  design "does not want wide expensive GPUs". It is retracted — see the
  RETRACTED block in `RESULTS.md`. Do not quote the old numbers.

Projected but **not** measured end-to-end: the 4090's 27.6% fill gain from
`--fill-blocks 144`, taken from a standalone sweep.

## Known defects

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale is a second, unchecked approximation. This is the
  main known yield loss.
- **Duplicate share 15.8–25%** from the full special-q-side factor base, at
  ~1.34× the downstream TD/cofactor work for the same unique yield. Whether
  truncating at `q` wins is untested — it also shortens the q range needed.
- **Fill geometry is entangled with `--threads`.** The 144-block default was
  swept at 256 threads, so the real invariant may be ~36,864 threads. The same
  144 is applied to `k_fill_l1`, which launches at a hardwired 512 threads and
  was never swept.
- **Power is board-only.** The metric of record is whole-box
  relations/sec/watt; host draw is unmeasured. The A100 has no sampled power.

## Open experiments, in order

1. **Concurrent-q throughput.** Two independent fill workspaces, two q or two
   sides in separate streams, 144 blocks each, sweep 1/2/4. This is the
   decisive test for whether wide cards are a poor fit or are simply being fed
   too little independent work — and it discriminates the write-combining
   hypothesis: if two independent 144-block fills scale while one 288-block
   fill does not, the problem is decomposition within one bucket set, not a
   whole-card ceiling. Note the two architectures predict differently here:
   Blackwell's flatness fits "idle capacity", Ada's *degradation* does not.
2. **Fill geometry independent of `--threads`**, plus an opt-in startup
   autotune over ~(96,144,288) × (128,256,512). Opt-in, not default: a default
   autotuner makes every number in `RESULTS.md` non-reproducible, and
   comparability has caught more errors on this project than anything else.
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
