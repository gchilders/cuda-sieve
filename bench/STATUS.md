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
  on the rented boxes (`ERR_NVGPUCTRPERM`) — but it has never been tried on
  the **local 5070**: recent drivers support Nsight Compute under WSL2, gated
  on the Windows-side permission (NVIDIA Control Panel → Developer →
  Manage GPU Performance Counters → allow all users), which is the same
  permission `ERR_NVGPUCTRPERM` names. One local profile of `k_fill_atomic`
  at 288 vs 1152 blocks would settle granularity vs not.
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
   (prototype.md, "The GPU's per-special-q budget"). The c183's own standalone
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
   transform/fill/resieve — no rebuild, no re-upload.
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

   Two cautions. The idle ceiling for all three is 7% of wall, so measured
   against item 3's 15.8–25% duplicate share this is the smaller prize — the
   argument for it is robustness on shared hardware, not throughput on a quiet
   box. And it interacts with item 1: two concurrent q roughly **doubles host
   work per unit time**, so on a contended host that experiment can come back
   negative for host reasons unrelated to the GPU. Run it on a verified-idle
   box and check `GPU-accounted / wall` first.
5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job. Note the payoff is job-shaped: for the snfs236 polynomial the
   special-q side is degree 1 with `d = [1, -3.5e-24]`, so largest-term and
   true-rectangle are the *same number* and this buys nothing there. The ~2-bit
   gap is a GNFS-side problem.
6. **Whole-box power**, to close the metric of record — the meter half of
   item 0.
7. **Why our survivor gate is looser than GGNFS's at matched lambda.** Same q
   range, same job, same nominal bits: `gnfs-lasieve4I14e` loses 17.3% of its
   yield going 91.8 → 87.5 bits where we lose 0.07% going to 88.0. We submit
   1,426 cofactors per special-q against GGNFS's 978 (`COF: 60664 tests`, 62 q)
   for comparable *unique* yield. So we admit ~46% more and the surplus is
   almost all unproductive — time, not relations. Until the cause is known the
   bound can only be set by sweeping, never derived for a new job. Candidates:
   a systematically low per-position norm estimate; the byte scale cancelling
   only if cell init and bound use it consistently; GGNFS's lambda gating more
   than the report threshold. **Do not "fix" it by tightening the default** —
   the current gate finds every relation GGNFS finds (verified, zero misses
   over 2,531), so any trade of yield for speed must be made deliberately.
8. **A = 32 sieve areas** (`2^16 × 2^16`), for jobs like AS276. Three
   independent blockers, all still present: `xmax` is `uint32_t`
   (`pipeline.cuh:90`, `:146`) and `(1u << 16) * 65536` is exactly 2^32, so it
   wraps to zero; VRAM at that area projects to ~14–16 GB against 11.9 GB
   usable on the 5070; and such jobs carry `lpba 35 > 32` and `mfba 101 > 96`,
   both refused by `check_cofactor_bounds` because the limb widths cannot hold
   them.
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
    the knee; report the item-0 verdict at stock AND at the knee. Under WSL2
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
    on this GPU dies at the L2 transaction ceiling.
