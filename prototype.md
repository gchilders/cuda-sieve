# GPU Bucket Siever for NFS — Prototyping Paths

## What we're actually testing

Not "build a GPU NFS." The single open question:

> **Can a GPU bucket-sieve for relation collection compete with CPU `las` on the economics we care about?**

If yes, the rentable-consumer-GPU case tips and it's worth pursuing. If a well-structured version trails by an order of magnitude with the main levers exhausted, the idea is dead — and we want to find that out cheaply, not after months. **Between those two outcomes there is a wide gray zone, and landing in it is not failure** (see "The verdict" below).

Everything below is scoped to **relation collection only**. Poly-select, filtering, linear algebra, and sqrt come from CADO unchanged.

### Decisions locked (don't relitigate; build against these)

- **Metric:** relations/sec/watt, graded on the green/gray/red zones below — not raw wall-clock, and *not* "% of peak bandwidth."
- **Fill is a multisplit, not a sort.** No `DeviceRadixSort`. Within-bucket order is waste.
- **Sieve cells are 16-bit, two per 32-bit word, native `atomicAdd`, region in shared memory.**
- ~~**Records: 4 B first, 2 B as the measured upgrade.** Record size is the biggest single lever.~~
  **MEASURED 2026-08-01 — 4 B is optimal and 2 B is a 40% regression.** See Path 1 results. Below
  4 B you stop saving bytes and only lose store efficiency; a scattered 16-bit store occupies a
  full sector transaction just as a 32-bit one does. **4 B is final; do not build the 2 B path.**
- ~~**Two-level fan-out is the design to beat**~~ — **MEASURED: single-level wins by 2.7×** on this
  card, because 16K buckets × 128 B = 2 MB of open cache lines fits inside a 48 MB L2, so the L2
  already does the write-combining that two-level exists to provide. Two-level pays 2.76× the DRAM
  traffic to solve a solved problem. **Single-level is the design; revisit only if the open-line
  footprint approaches L2 size (I16e, or a small-L2 card).**
- **Language: C++17 host, CUDA C++ device (sm_120, CUDA 13.2 — verified on this box), Python for the harness only, never in the data path.**
- **Reuse, don't reinvent:** CADO for everything but the sieve. `las` is a *subprocess oracle* — nothing links against CADO before Path 4, and possibly not even then.
- **Baseline ≠ oracle.** **GGNFS `gnfs-lasieve4I15e` is the performance baseline** (AVX-512 lattice-siever asm; ~20% faster than CADO on this owner's jobs). **CADO `las` is the correctness oracle** (`-dumpfile`, `-batch-print-survivors`, `las_tracek`, checksums — GGNFS has none of these). Grade speed against GGNFS; grade correctness against CADO. See "Which siever is the opponent".
- **Grade on *unique* relations**, not raw relations. The two sievers use different factor-base conventions and therefore have different duplicate rates; raw relations/sec is not comparable across them. See the same section.

### Open questions the ladder must answer

1. Cofactorization share of `las` time on this job (Gate 0 — can kill the project in an afternoon).
2. Best achievable **updates/sec/joule** across kernel-structure × record-size, vs. CPU `las` (Gate 1).
3. **Root-transform cost per special-q on GPU** — a second per-q cost pillar the original plan omitted; see its section below.
4. Single- vs. two-level crossover, and where the GPU's `bkthresh` optimum sits (above CADO's).
5. Can batch cofactorization sustain the survivor rate a fast GPU sieve produces?
6. Real rental $/relation at Path 5, with prices written down, not assumed.

---

## The verdict metric, and the actual bar

The headline metric is **relations/sec/watt**, not wall-clock and not raw throughput. A GPU that's 2× faster at 4× the power loses.

But "2–3× of CPU per watt" is an asserted threshold, not a derived one. Derive it from this box:

| | Device | Power under load |
|---|---|---|
| CPU | Ryzen 7 9800X3D (8C/16T, 96 MB L3; 120 W TDP, 162 W PPT) | **~110 W planning estimate** — owner's estimate is "just north of 100 W" at full bore; sieving is integer/memory-bound, not AVX-bound, so it should sit well under PPT. Measure with LibreHardwareMonitor. |
| GPU | RTX 5070 (48 SM, 12 GB GDDR7, 672 GB/s) | **~220 W planning estimate** — 250 W cap, but measured 216–218 W under a compute-saturating ECM load. A bandwidth-bound fill kernel may draw less. Measure with NVML. |

GPU perf-per-watt relative to CPU is `(P_cpu/P_gpu) × speedup`, where *speedup* is measured against the **whole 16-thread box**, not one core. That prefactor is more fragile than it looks:

| Goal | @ `P_cpu/P_gpu = 0.50` (110 W vs 220 W, package vs board) | @ `0.57` (CPU + DRAM ≈ 125 W) |
|---|---|---|
| Within 3× on perf/watt | **0.67×** | 0.59× |
| Within 2× on perf/watt | 1.00× | 0.88× |
| Actually beat CPU on perf/watt | 2.00× | 1.76× |

**The measurement asymmetry is as large as the correction it's competing with.** `nvidia-smi`'s `power.draw` is total *board* power — it includes GDDR7. LibreHardwareMonitor's "CPU Package Power" is socket power — it excludes the DIMMs, and CPU sieving is exactly the workload that hammers them (~12–16 W for two DDR5 sticks under load). So the honest CPU-side number for a fair comparison is package **plus** a DRAM estimate, and that swing (0.50 → 0.57) is bigger than the gap between the owner's ~110 W estimate and this doc's original 120–160 W band.

Consequences: **quote the zone boundaries as a band (0.6×–0.7× for green), not to two significant figures**, and don't let a gray-zone verdict turn on the third digit. A wall-plug meter is the only thing that collapses this uncertainty — it measures both sides inclusive of DRAM and VRM. Not needed for Gate 0 or Gate 1; worth buying before the Path-5 verdict if the result lands in the gray zone.

### The zones — how to read a result

- **Green — ≥ ~0.6–0.7× the full CPU box.** Clears the 3×-perf/watt bar. Go. (The band, not a point, for the power-accounting reason above.)
- **Gray — roughly 0.1× up to the green band.** Not a fail. Decide on evidence, specifically:
  - *Remaining levers:* if we're at 0.3× with 4 B records, single-level fan-out, or unbatched cofactorization still on the table, keep pulling. If every lever in this doc is exhausted, what's left is the economics call.
  - *Rental economics:* the real metric for rented boxes is **$/relation**, and a rented GPU instance includes a CPU — so the marginal question is "does the GPU add relations/hour beyond what the bundled CPU produces, per marginal dollar?" Compute `(R_cpu_bundled + R_gpu) / $·hr_gpu_instance` vs. `R_cpu_box / $·hr_cpu_instance` with actual marketplace prices (vast.ai / RunPod, written down at Path-5 time). This can rescue a result that perf/watt alone wouldn't.
  - *Strength of the baseline:* the 9800X3D's 96 MB V-Cache makes it an unusually strong sieving CPU — bucket regions and much of the working set stay resident. We are grading against a best-case opponent; a near-miss here can still beat a generic rented CPU. Report the ratio *with this caveat attached*, not bare.
- **Red — below ~0.1× with record size, fan-out, and the cofactor path all already exploited.** Orders-of-magnitude territory. Dead, or pivot to offloading only cofactorization / linear algebra (which we already know works).

---

## Concrete target

Grounding everything in the job already staged in `../test-sieve/`:

- **C183**, degree 5, `rlim=67.1M`, `alim=134.2M`, `lpbr=31`, `lpba=32`, `mfbr=60`, `mfba=92`
- Factor base: **~4.0M rational / ~7.6M algebraic primes** — the prepared `input.job.afb.0` is 61 MB
- Oracles available: CADO `las` at `/home/kylea/cado-nfs/build/DESKTOP-3J4UC68/sieve/las`, plus `gnfs-lasieve4I{14,15,16}e`
- **Staged in `../test-sieve/`:** `cado_test_sieve.sh` + configured `cado_test_sieve.ini` (translates the GGNFS job for `las`), `cado_tmp.poly`, `cado_roots1.gz`. **But the staged CADO artifacts are stale — they belong to a different number** (see Step 0). Path 0 is *not* just a matter of running these.

### What the existing test sieve already tells us — read this before Path 1

A `gnfs-lasieve4I15e` test sieve of this exact job was already run on this box (algebraic side, `qintsize=1000`). `test_sieve.sh:187` launches **one** siever with no backgrounding, so these are **single-core** numbers and the script's "ETA: 315d" is a one-core ETA.

| q0 | special-q | yield | rel / q | sec/rel | **core-sec / special-q** |
|---:|---:|---:|---:|---:|---:|
| 50M | 61 | 2919 | 47.9 | 0.056 | 2.68 |
| 90M | 41 | 2025 | 49.4 | 0.063 | 3.11 |
| 130M | 48 | 2138 | 44.5 | 0.070 | 3.12 |
| 170M | 53 | 2304 | 43.5 | 0.072 | 3.13 |
| *(210M — outside job range)* | 54 | 2186 | 40.5 | 0.077 | 3.12 |

**~3.1 core-seconds per special-q, flat to ±1% across the entire q-range.** Yield drifts (48 → 41 rel/q) but per-q cost does not. That is an unusually stable planning constant and everything below is built on it.

Job scale: q ∈ [50M, 190M] → ~7.6M special-q → ~273 core-days → ~340M relations. Consistent with the script's own 391M estimate.

#### The GPU's per-special-q budget

Whole-box CPU per-q is `3.1 s ÷ N_effective`. At 16 workers with realistic 0.7–0.9 scaling (memory and cache contention — this is what the configuration sweep measures), that is **~0.22–0.28 s per special-q**. Against the verdict zones:

| | GPU must deliver one special-q in |
|---|---|
| Green (≥0.67× box) | **≤ 0.32–0.41 s** |
| Beat CPU on perf/watt (2.0×) | **≤ 0.11–0.14 s** |

Now cost the GPU sieve from this doc's own models, at I15e with 4 B records and a pessimistic 30% of peak bandwidth:

| stage | cost |
|---|---:|
| root transform (8.6M modinv) | ~1–5 ms |
| norm init (1.07 GB of 16-bit cells) | ~5 ms |
| small-prime sieve (~1.5e9 shared-mem updates) | ~1–3 ms |
| bucket fill (6.1e8 × 4 B, write + read = 4.6 GB) | ~24 ms single-level, ~48 ms two-level |
| apply (read 2.3 GB into shared memory) | ~12 ms |
| **total** | **~45–75 ms** |

> **MEASURED 2026-08-01, algebraic side, one q.** The model was pessimistic on
> every line that has been measured, and right about the shape.
>
> | stage | modelled | side 1 | side 0 | both |
> |---|---:|---:|---:|---:|
> | root transform | ~1–5 ms | 1.75 | 1.01 | **2.76** |
> | bucket fill, single-level 4 B | ~24 ms | 12.60 | 11.89 | **24.49** |
> | apply = norm init + small sieve + bucket apply + scan | ~18 ms | 11.95 | 7.91 | **19.86** |
> | **total ms / special-q** | ~45–75 | **26.30** | **20.81** | **47.11** |
>
> **Revised 2026-08-01 (later): 55.5 ms** once the sieve runs on CADO's own
> factor base, which includes prime powers. Powers add 2.5e9 small-sieve
> updates (6.84e9 total), taking side 1 to 32.8 ms and side 0 to 22.7 ms. The
> complete two-sided sieve now sits **right at the 56 ms trial-division
> floor**, and the small-prime sieve is half the chain.
>
> **Scope of this number, 2026-08-02.** It is a *sieve* measurement: transform,
> fill, apply, threshold, survivor list. The two-sided survivor intersection,
> resieve, factor recovery, host transfer and the cofactor feed do not exist.
> **Kernel feasibility is demonstrated; relation-collection feasibility is
> not**, and any whole-box speedup quoted from it is a projection with the
> cofactor path assumed unchanged, not a measured relation rate. Two placement
> bugs found in review on 2026-08-02 (findings 18–19) change where updates land
> but not how many, so they do not move this timing.
>
> Measured at `q=120000053, rho=112625526` — a real special-q from las, not a
> synthetic one — with region 2^14. The apply row is a single fused kernel and
> already contains norm init (1.76 ms) and the small-prime sieve (13.17 ms
> across both sides); they are not additional.
>
> **47 ms for the complete two-sided sieve**, against a CPU box that must
> deliver one q in 220–280 ms, and against this document's own 56 ms
> trial-division floor. The 30%-of-peak assumption was wrong in both
> directions: fill runs at 17–18% of peak, apply at 58%.
>
> The one line the model badly underestimated is the **small-prime sieve: 1–3 ms
> budgeted, 13.2 ms measured** (4.36e9 updates, 2.9× the modelled count, and
> 7.2× the entire bucket-sieve volume). It is now 28% of the chain. See
> `bench/RESULTS.md` finding 11.

### Where the time actually goes — GGNFS fine-grain timings (measured, complete accounting)

`gnfs-lasieve4I15e` prints a per-phase millisecond breakdown, and it was captured for all five q points. **The parts sum to wall-clock to within 0.01% at every q** — this is a complete accounting, not a sample. Full data in `oracle/ggnfs_timing_breakdown.txt`.

Share of wall time:

| q0 | Sieve | └ large | └ small | medsched | Sieve-Change | TD | └ MPQS | alg. FB entries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50M | 61.2% | 38.3% | 18.8% | 7.9% | 9.6% | 21.4% | 8.1% | 3,001,128 |
| 90M | 56.0% | 35.6% | 16.7% | 8.8% | 11.2% | 24.1% | 10.7% | 5,218,803 |
| **130M** | **54.3%** | **34.3%** | **16.5%** | **10.0%** | **12.0%** | **23.7%** | **9.9%** | 7,380,860 |
| 170M | 53.9% | 34.0% | 16.3% | 10.0% | 13.2% | 23.0% | 9.3% | 7,605,406 |
| 210M | 54.0% | 34.1% | 16.3% | 10.1% | 13.6% | 22.3% | 9.0% | 7,605,406 |

Per special-q, single core, at q=130M (3100 ms total): Sieve 1683 ms (large 1062, small 510, norm init **18**), medsched 309 ms, Sieve-Change 373 ms, TD 734 ms (MPQS 306 ms).

Four things fall out of this:

1. **`f ≈ 0.23`, not 0.5–0.6.** The fraction that does *not* go away with a GPU sieve is **0.21–0.24** across the whole range if all of trial division stays on the CPU, and **0.08–0.11** if only the hard cofactorization (MPQS) does. Despite `mfba=92` and three large primes, **cofactorization does not dominate this job.** The Amdahl fear this document was built around is, on GGNFS at least, not borne out.
2. **The root-transform cost pillar is confirmed, independently and precisely.** "Sieve-Change" is exactly the per-q q-lattice root transform, and it is **12% of wall** — 373 ms/q over ~11.3M FB entries = ~33 ns/entry ≈ 85 cycles, which is the right cost for one modular inverse. The doc predicted this stage was real and co-equal with fill; the measurement agrees. A GPU doing ~8.6M modinvs in 1–5 ms wins this stage by roughly 100×.
3. **Norm initialization is nearly free** — ~18 ms/q, 0.6% of wall. Path 3 can implement it naively without a performance concern. It remains the main *parity* obstacle, but not a performance one.
4. **The algebraic factor base is truncated at q** (`Warning: lowering FB_bound to <q>`): 3.0M entries at q=50M rising to the full 7.6M at q≥170M. So updates/q, bucket memory, and transform cost all roughly **double across the job's q-range** — the sizing table's numbers are the *high* end. **Check whether CADO does the same truncation**; if it does not, the two sievers are not doing equal work per q and the baseline comparison needs adjusting.

#### What this does to the verdict

Box per-q at `N_eff ≈ 13` is ~238 ms. The CPU-resident floor is `f × 238 ms` = **56 ms** (all TD) or **24 ms** (MPQS only). Modeled GPU sieve is 45–75 ms. Those are the *same size*.

| | per-q | implication |
|---|---:|---|
| CPU box | 238 ms | the thing to beat |
| GPU sieve (modeled) | 45–75 ms | |
| CPU floor, all TD stays | 56 ms | pipeline ≈ `max(60, 56)` ≈ 60 ms → **4.0× box** → perf/watt **1.33** |
| CPU floor, resieve also on GPU | 24 ms | pipeline ≈ 60 ms, sieve-bound → same 4.0× until the sieve improves |
| hard ceiling from `f` | — | `1/0.237` = **4.2×** — and reaching it means the CPU is 100% busy on TD |

At 4× with CPU+GPU drawing ~330 W, perf/watt ≈ **1.33× the CPU box — green, and past the "beat CPU" line.**

**This reverses the previous reading of this section.** With `f` assumed at 0.5–0.6, the cofactor floor sat above the GPU sieve time and sieve optimization was worthless. With `f` measured at 0.23, the floor sits *below* it, and **the GPU sieve is the binding constraint again.** Both stages now matter and they are balanced within ~10%:

- Path 1 matters. If the sieve lands at 150 ms instead of 60 ms, speedup drops to 1.6× and perf/watt to 0.53 — gray. The bandwidth assumption is load-bearing and must be measured.
- Path 4 still matters, but as a *ceiling raiser*, not a rescue: moving resieve/TD to the GPU lifts the cap from 4.2× to ~10×. That is the second lever, not the first.

Caveats: this is **GGNFS lasieve4, not CADO las**, and CADO's cofactorization strategy differs (ECM chains, and `-batch` if enabled). Gate 0 on CADO is still required and may land elsewhere. The 30%-of-peak bandwidth assumption and `N_eff` are both still unmeasured. But `f` is no longer a guess on this job, and the direction it moved is favourable.

**Concrete Path-4 throughput target, now derivable:** ~1,900 survivors reach cofactorization per special-q. At 4× box speedup the pipeline runs ~16.8 special-q/s, so cofactorization must sustain **~32,000 survivors/sec**. At the measured 0.16 ms/survivor/core that needs ~5 of 16 threads — feasible, and now a number to test rather than a hope.

### Which siever is the opponent — and why it isn't CADO

The owner reports **GGNFS is ~20% faster than CADO overall** on his jobs, largely because `lasieve4` carries hand-written AVX-512 lattice-siever code (`asm64,avx-512 lasetup,lasched,sieve1,ecm,tds0,search0,tdsched` in its banner). This document's premise is that we grade against a *best-case* opponent, so:

| role | siever | why |
|---|---|---|
| **Performance baseline** | `gnfs-lasieve4I15e` | the faster of the two here; grading against the slower one would flatter the GPU by ~20% |
| **Correctness oracle** | CADO `las` | `-dumpfile`, `-batch-print-survivors`, `las_tracek`, `-v` checksums. GGNFS offers none of this |

This is a change from the original plan, which used CADO for both. Nothing about Paths 1–3 changes — CADO remains the only practical parity target — but **the number the verdict is computed against comes from GGNFS.**

#### The factor-base convention, duplicates, and the metric

The two sievers differ in a way that makes raw relation counts non-comparable:

- **GGNFS truncates the algebraic factor base at `q`** (`Warning: lowering FB_bound to <q>` in the logs; 3.0M entries at q=50M rising to the full 7.6M at q≥170M).
- **CADO does not.** Consequence, per the owner: CADO has **longer wall-clock at low q, finds more relations at low q, and has a higher duplicate rate**.

The likely mechanism — stated as inference, and testable: a relation whose algebraic norm has a factor `p > q` will also be found when sieving special-q = `p`. Truncating at `q` finds each such relation exactly once, at its largest eligible special-q. CADO instead finds it repeatedly and removes the duplicates afterwards, which is why it ships `-dup`, `dup-qmin`, `dup-qmax`.

Two consequences that must be in the plan:

1. **The verdict metric is *unique* relations/sec/watt.** Comparing raw relations/sec across sievers with different truncation conventions measures the convention, not the siever. Either dedup both outputs or hold the convention fixed and say so.
2. **The GPU sieve should adopt GGNFS's truncation.** It is the convention of the faster siever and it produces less redundant work. Be honest about the size, though: truncating at q=50M removes primes in [50M, 134M], which are ~40% of algebraic *entries* but only ~9% of algebraic *updates* (small primes dominate `A/p`). Net saving at the low end of the q-range is roughly **5–10% of wall**, concentrated in the root-transform stage — worth taking, not transformational. Its real value is the duplicate avoidance.

**Caveat on `f`:** the measured `f ≈ 0.23` is GGNFS's. CADO's non-truncated factor base means more sieve work per q, which would push CADO's `f` *lower* still. So `f ≈ 0.23` is unlikely to be optimistic — if anything it is the pessimistic end.

### Operating point — locked

**`-A 29` (≡ `-I 15`, since CADO defines `-I x` as `-A (2x-1)`), `-adjust-strategy 0`, `-sqside 1`.** C183 is vanilla 15e territory in GGNFS and this is its CADO equivalent. Every graded number in this project is at this operating point; `-A` is not a variable except in an explicitly-labelled sensitivity check.

**q-range: the job is `q ∈ [50M, 190M]`** (it will be sieved for real on NFS@Home). The frozen benchmark band is **`-q0 120000000 -q1 120001000`** — the midpoint, `qintsize=1000` matching the existing test-sieve convention, ~54 special-q, ~170 core-seconds. Per-q cost is flat to ±1% across the range (table above), so one band is genuinely representative for *timing*. Re-run at **50M and 190M** as a sensitivity check for the things that are *not* flat: yield drifts ~10% and cofactorization share may drift with it, and `f` is the number the whole verdict rests on.

`-adjust-strategy 0` (constant `logI`, J capped) rather than the staged script's `2` (dynamic `logI`, skewed basis): strategy 2 varies I per special-q, which makes the sieve area — and therefore the update count, the bucket sizing, and the region count — a per-q variable. That is fine for a yield survey and poison for a fill-throughput benchmark and for byte-exact `-dumpfile` parity. **Do not reuse `cado_test_sieve.sh` for Path 0**; it is interactive and hardcodes `-adjust-strategy 2`.

### Two jobs, and only one of them produces a verdict

The owner has offered any polynomial. Use two, with a hard rule about which is which:

| | **Timing job** | **Parity job** |
|---|---|---|
| Number | the C183 in `../test-sieve/input.job` | a small GNFS composite, ~C120 |
| Params | `lim` 67.1M/134.2M, `-A 29` | `lim` ~2M/4M, `lpb` 26/26, `-A 25` (I=13) |
| Purpose | every graded number: Gate 0, Gate 1, Gate 2 | Path 3 correctness only — `verify_cpu` diffs, `-dumpfile` byte-diffs, survivor-set matching, `las_tracek` chases |
| Why | FB ≈ 11.6M entries ≈ 90+ MB in SoA, plus GB of buckets — comfortably outside the 96 MB V-Cache, i.e. the regime we actually care about | a q sieves in well under a second single-threaded, region dumps are ~33 MB instead of ~537 MB, so parity iterates in seconds |

**Never quote a timing number from the parity job.** At `lim1 = 4M` the whole factor base is L2-resident and you are measuring the cuda-mpqs regime — the exact "cache-residency illusion" listed under Known traps. Its only job is to make correctness debugging fast; it still exercises every code path that matters (bucket-sieved primes from 2^13 to 4M, both sides, projective roots, prime powers).

**The FB is ~61 MB per side, not hundreds of MB.** It fits in L2-adjacent working set comfortably and is a non-issue for a 12 GB card. The memory pressure is entirely from **bucket storage**, and that reframes the whole design (below).

`mfba=92` with `lpba=32` means **three large primes** on the algebraic side. Cofactorization on this job is expensive. See "Amdahl" below — it is the biggest risk in the plan.

---

## Sizing: the numbers that drive every design decision

Updates per special-q, both sides, `bkthresh` = CADO default of `2^I`:

| siever | sieve area A | updates/q | @2 B | @4 B | @8 B |
|---|---:|---:|---:|---:|---:|
| I14e | 1.34e8 | 1.7e8 | 0.3 GB | 0.6 GB | 1.3 GB |
| **I15e** | **5.37e8** | **6.1e8** | **1.1 GB** | **2.3 GB** | **4.6 GB** |
| I16e | 2.15e9 | 2.2e9 | 4.0 GB | 8.1 GB | 16.1 GB |
| I16e `-J 16` | 4.29e9 | 4.3e9 | 8.1 GB | 16.1 GB | 32.3 GB |

Raising `bkthresh` to 2^18 / 2^20 cuts the I15e count to 4.2e8 / 3.0e8 respectively — that's a real tuning knob, not a fixed cost.

**Consequences:**

1. **I15e fits on the card; I16e does not.** At I16e even 4 B/record needs 8.1 GB of bucket storage with no room for double-buffering. **Slab the sieve area** into horizontal strips sized so bucket storage lands at ~2–3 GB. CADO does exactly this; it is normal, but it must be in the design from day one because it sets the batch and streaming structure.
2. **Record size is the single largest lever.** Bigger than bandwidth efficiency, and previously untreated as a design axis.

### Record size — attack this first

Per-q DRAM traffic and time on a 672 GB/s card, I15e, 6.1e8 updates, single-level fill (write + read):

| record | traffic | @100% | @50% | @30% | @20% |
|---|---:|---:|---:|---:|---:|
| 8 B | 9.1 GB | 15 ms | 29 ms | 49 ms | 73 ms |
| 6 B | 6.8 GB | 11 ms | 22 ms | 36 ms | 55 ms |
| 4 B | 4.6 GB | 7 ms | 15 ms | 24 ms | 36 ms |
| **2 B** | **2.3 GB** | **4 ms** | **7 ms** | **12 ms** | **18 ms** |

Two-level fill roughly doubles record traffic (see below for why it can still win).

How to get small:

- **4 B — the safe target, build it first.** 16-bit offset within the region + 16-bit prime hint. This is essentially CADO's `shorthint` bucket update (`bucket.hpp`: 16-bit `x` + 16-bit `slice_offset`). A 16-bit offset covers regions up to 65536 positions, larger than anything we'd use.
- **2 B — offset only, log implicit.** Process the FB in **slices** where every prime in a slice rounds to the same `logp`, and record a per-(bucket, slice) segment start. The update then needs *only* the offset; the log is recovered from which segment it falls in, and the prime identity is recovered later by resieve. CADO buckets are already slice-segmented, so the *layout* is validated — but note the GPU-side bookkeeping cost honestly: per-(bucket, slice) segment boundaries mean either **one kernel launch per logp-slice with a bucket-cursor snapshot after each** (a few tens of launches per level, ~µs each, CUDA-graph-friendly — cursors are 16K×4 B = 64 KB to snapshot), or carrying a slice id in level-1 records and segmenting during the level-2 split. That's why 4 B ships first and 2 B is the measured upgrade, not the starting point.

Budget the microbenchmark to measure both. If 2 B works, the bandwidth-efficiency argument mostly stops mattering.

---

## The core bet

The sieve's dominant cost is bucket fill — routing ~10⁸–10⁹ tiny update records into region-sized buckets, redone from scratch every special-q. The naive GPU version does a per-record `atomicAdd` to reserve a slot → divergent, uncoalesced, atomic-serialized. That's the whole failure mode.

### Correction: this is a multisplit, not a sort

The original framing said "reformulate fill as a radix sort keyed by destination bucket, reuse `cub::DeviceRadixSort`." **Don't.** Apply does `S[offset] -= logp` — addition is commutative, so **order within a bucket is irrelevant**. Sorting produces a property strictly stronger than what's needed, and you pay for it.

On CPU there's a secondary reason to sort within a bucket (write locality in the apply pass). On GPU the region lives in **shared memory**, which is random-access at full rate modulo bank conflicts. That motivation evaporates too. Within-bucket order is pure waste in both directions.

The right primitive is a **single-pass multisplit**: histogram → prefix-sum → scatter. `cub::DeviceScan` and `cub::DeviceHistogram` are useful. `cub::DeviceRadixSort` is not — over a 14-bit bucket key it runs multiple LSD passes plus stability machinery we don't need. Expect ~1.5–2.5× from this correction, not the 2.75× the Gerbicz swap got in msieve-s (that replaced a 7-pass 56-bit sort; this replaces a 2-pass 14-bit one).

### Better: skip the histogram pass entirely

The count per bucket is *predictable*. Prime `p` produces ~`A/p` hits spread near-uniformly across buckets. So pre-size buckets from the analytic estimate times a safety margin, and grow-and-retry on overflow.

This is not speculative on either side:
- CADO has **`bkmult`** — "multiplier to use for taking margin in the bucket allocation" — precisely because its allocation is estimated rather than measured.
- msieve-s already runs this pattern twice over: *"bucket storage grows and retries when the observed max bucket exceeds the estimate,"* and the standalone bench statically sizes at `mean + 6σ + 32` and never overflowed in production runs. Copy both halves: analytic size × margin up front, grow-and-retry as the safety net.

Collapsing count+scan+scatter to just scatter removes a full read pass over the FB.

### Two-level fan-out — and the real reason for it

Two levels, mirroring CADO's hierarchy:
- **Super-buckets** sized to L2 — **48.00 MB measured, but only 30.00 MB can be *pinned*** (`persistingL2CacheMaxSize`, verified 2026-08-01). Size against 30 MB if using a `cudaAccessPolicyWindow` carve-out, or against 48 MB and accept ordinary LRU eviction. Record which.
- **Sub-buckets** sized to per-block shared memory, applied on-chip.

The usual justification is cache sizing. **The sharper reason is write-stream count.** A single-level split into ~16K buckets means ~16K simultaneously-open write streams; you cannot hold 16K partially-filled cache lines, so every record becomes a partial-line write and you burn transactions.

That gives a concrete design rule: **fan-out per pass is bounded by how many open output buffers fit in shared memory.** At 48 KB and 128-byte flush granularity that's ~384 open streams; use 128 for headroom. Two passes of 128 gives 16384 buckets. Each block accumulates into its 128 shared-memory buffers and flushes full cache lines with **one atomic per flush, not per record** — which fixes coalescing and atomic pressure in the same move.

**Evidence this matters:** msieve-s's `scatter_roots_kernel` scattered directly into 16384 buckets, sat at ~77% of the L2 transaction ceiling, was 34% of total GPU time, and **could not be improved** — `__stcs` cache hints and removing warp-aggregation both measured flat. That is a paid-for result telling us single-level 16K-way scatter is transaction-bound at its ceiling. Two-level is the lever msieve-s never got to pull.

The tradeoff is empirical and Path 1 must measure it: two-level costs 2× the record traffic but converts partial-line scattered writes into full-line streaming writes. If single-level lands at 25% of peak and each two-level pass lands at 70%, two-level wins (`4N/0.70 = 5.7N` vs `2N/0.25 = 8N`). Measure both; don't assume.

> ### MEASURED 2026-08-01 — the argument above is wrong on this card
>
> Path 1 built both. Full data in `bench/RESULTS.md`. Two-level **loses by 2.7×**
> (37.9 ms vs 14.2 ms per special-q, algebraic side).
>
> The transaction claim was *correct*: single-level really does amplify writes
> 7.2× (0.898 L2 write sectors per 4 B record), and two-level really does reduce
> that to 1.3× — a 5.5× reduction in L2 write sectors, exactly as predicted.
>
> **It buys nothing, because all three kernels sit at the same 17–18% of DRAM
> peak.** The differentiator is bytes moved, not efficiency: single-level writes
> its output once (1.36 GB), two-level writes-reads-writes (3.75 GB). Time ratio
> 2.68× ≈ traffic ratio 2.76×.
>
> **The premise that failed:** "you cannot hold 16K partially-filled cache lines."
> On this card you can — 16384 × 128 B = **2 MB**, against **48 MB of L2**. The L2
> *is* the write-combining buffer the design wanted to build in shared memory. The
> amplification happens between SM and L2 and is absorbed before DRAM, which is why
> 280M and 51M write sectors produce identical DRAM throughput.
>
> Per *pass* two-level is genuinely better (L1 alone 12.5 ms beats single-level
> 14.2 ms) — the second pass just costs more than the win.
>
> **Carry forward the rule, not the verdict:** compare **open-line footprint vs L2
> size**. At I16e (4× the regions) or on a small-L2 card the footprint grows and
> this can flip. The msieve-s result stands too — it was a *transaction* ceiling,
> and transactions turn out not to be the binding constraint here.

### Threshold to keep in mind

Below some size the entire FB + region fits in one SM's shared memory and you skip bucketing entirely. That's the cuda-mpqs regime, and it's why that problem looked easy. Bucketing only earns its keep above cache-residency — exactly the regime we care about, and exactly the one a naive benchmark will hide from you.

### Where the bucket threshold should sit

Cost of bucketing prime `p` is `A/p` update records; cost of re-walking it per region is `R` reads (once per region). Bucketing wins iff `A/p < R`, i.e. `p > A/R = region_size`. That is exactly CADO's default `bkthresh = 2^I` — the CPU crossover is already at the right place analytically.

But the *cost ratio* differs on GPU: FB re-walk reads are perfectly coalesced and L2-broadcast-friendly across concurrent blocks, while bucket writes are scattered. A coalesced read is worth several scattered writes. **So the GPU crossover sits above CADO's** — direct-sieve (re-walk) a somewhat larger set of primes before switching to bucketing. Sweep `bkthresh` on the GPU independently; do not inherit CADO's value.

(Fully eliminating bucketing by re-walking the whole FB per region is dead — at I15e that's ~11K regions × 7.6M entries ≈ 1.8 TB per special-q, ~1000× worse. Checked, closed, don't revisit.)

---

## The second per-q cost: root transforms (new — the original plan omitted it)

Bucket fill is not the only work redone every special-q. Before any record is scattered, every FB entry's root must be mapped into the q-lattice: for each `(p, r)` pair, compute the transformed root `r' mod p` from the lattice basis `(a0,b0,a1,b1)`. CADO's `fb_root_in_qlattice_31bits` is *"3 calls to `redc_32` and 1 to `invmod_redc_32`"* — a modular inverse per pair.

**Correction (verified in source): the inversion batches across roots of the same prime.** `fb_root_in_qlattice_31bits_batch` computes all denominators first, then does **one** `batchinvredc_u32` for the whole prime (same modulus `p`, so Montgomery's trick applies — it's only *different* moduli that block it), then 3 `redc_32` per root. So the cost is **one modular inverse per distinct prime**, not per `(p, r)` pair. It returns `false` and falls back to per-root if any transformed root is projective.

Sizing, corrected:

| side | FB `(p,r)` pairs | distinct primes with ≥1 root | modinvs/q |
|---|---:|---:|---:|
| rational (`lim0`=67.1M) | 3,957,294 | 3,957,294 (1 root each) | 3,957,294 |
| algebraic (`lim1`=134.2M, deg 5) | **7,605,406** | **4,817,761** | 4,817,761 |
| **total** | **11,562,700** | | **8,775,055** |

**These are measured, not estimated** — counted directly from `oracle/input.job.afb.0` and the `las` log. The S₅ argument that predicted them holds precisely: average roots/prime = 1 by Frobenius, but 44/120 ≈ 37% of primes have *no* root, so pairs-per-distinct-prime should be ≈1.58 — measured **1.579**.

So the transform budget is **~8.78M modular inversions per special-q, not ~11.6M** — a ~25% haircut on this cost pillar, and it transfers to the GPU unchanged because the FB is already grouped by `p` (SoA with a per-prime root count). Still ~2% of the record *volume*, but each op costs tens-of-cycles rather than a byte of scatter — on this card expect **order-of-milliseconds per q, the same order as the fill budget itself**.

The precedent is direct: msieve-s's `sieve_kernel_trans_*` is structurally this exact kernel (per-special-q modular root transformation feeding a scatter), and it was the **single largest kernel at 37% of GPU time, math-chain-bound at its ceiling**. Assume this cost is real until measured otherwise.

Design notes, all inherited from paid-for msieve-s results and CADO:

- **Benchmark it as its own kernel in Path 1 (variant "T")** — it is the producer stage of fill.
- **Do not fuse it into the scatter.** msieve-s experiment #11 fused trans+scatter: registers 56 → 72, lost a block/SM, per-launch time doubled, net +25% GPU time. Keep producer and scatter as separate kernels with an intermediate (or have trans write compact `(bucket, offset)` work items — but check registers before believing anything).
- **Use branch-free 2-adic (Hensel/REDC-style) inversion**, not Euclid — GPU-friendly, and CADO's `fb_root_in_qlattice` (32-bit REDC variants) is the reference implementation, including the projective-root edge cases you will otherwise get wrong.
- Keep the FB in SoA (`primes[]`, `roots[]`) so this kernel reads coalesced; precompute per-prime REDC constants once at startup (amortized across all q).

---

## Resolving the byte-atomic wrinkle

GPU atomics are 32-bit-native; `las` logs are 8-bit. The original options were: pack 4 bytes/word with masked `atomicOr`/`atomicCAS`; sort within each bucket by location; or tolerate races and eat lost hits.

**Recommended default instead: 16-bit cells, two per 32-bit word, native `atomicAdd`.**

- `atomicAdd` on a 32-bit word with the value placed in the correct half-word is *exactly correct* — carry into the neighbouring half-word requires an accumulated log above 65535, which cannot happen.
- The 4-bytes-per-word variant is **not** safe with plain `atomicAdd`: accumulated logs do occasionally exceed 255 (CADO saturates; a GPU `atomicAdd` cannot), and the carry silently corrupts the adjacent position.
- No packing tricks, no CAS loops, no lost hits, no extra sort.

Region size on this card (sm_120; 101376 B = 99.0 KB opt-in per block, 102400 B = 100 KB per SM — both re-verified 2026-08-01):

| smem for cells | blocks/SM | max threads/SM reachable | **occupancy ceiling** | region (16-bit cells) | regions at I15e |
|---|---|---|---|---|---|
| 96 KB | 1 | 1024 | **66.7%** | 49152 | ~10.9K |
| 64 KB (+32 KB staging) | 1 | 1024 | **66.7%** | 32768 | 16384 |
| 48 KB | 2 | 1536 | 100% | 24576 | ~21.8K |
| 32 KB (+16 KB staging) | 2 | 1536 | 100% | 16384 | 32768 |

**The occupancy column is a hard ceiling, not a tuning outcome.** A block caps at 1024 threads and the SM holds 1536, so any configuration with 1 block/SM leaves a third of the thread slots permanently empty. That is a structural argument for the smaller regions that the original table hid.

Prefer the **power-of-two region options** (32768 / 16384): they leave shared memory for fill staging buffers in the same block, and they align with `las` for oracle byte-diffs. Note the fix from the original draft: **CADO's default bucket region is 2^16 = 65536 positions** (`LOG_BUCKET_REGION = 16` in `las-config.cpp`, runtime-settable *downward* via `-B`; values above 16 are rejected). Our regions are smaller than CADO's default — that's fine (more regions, same total work), and for dump-diffing set las to `-B 15` or `-B 14` so region boundaries align. Sweep the occupancy tradeoff — 1 block/SM needs 512–1024 threads/block to hide latency. A 16-bit offset covers every option.

Measure shared-memory bank-conflict rate: with two positions per word, random offsets give an expected conflict degree of roughly 1.5–2× on 32 banks. Acceptable, but confirm.

> **MEASURED 2026-08-01 — this whole section describes a non-problem.**
>
> Bank-conflict rate came in at **1.63×** over the whole apply kernel, inside
> the predicted band. Init and scan are conflict-free (20,839 conflicts of
> 17.5M wavefronts); every conflict belongs to the scatter.
>
> More to the point, **none of it costs anything.** Shared-memory `atomicAdd`
> against a deliberately racy plain `+=` measured 3.08 vs 3.00 ms — ~1%. And
> the 16-bit cell scheme is not a compromise: 8-bit cells (which carry the
> real carry-corruption bug this section identifies) came in **1% faster**, and
> only because they halve the shared memory a region needs — which is
> recoverable for free by halving the region instead. Take the correct scheme;
> it is also the fast one.
>
> One addition the section needs. Atomics are add-only and a 16-bit half-word
> *subtract* would borrow into its neighbour, so "init to the norm and subtract
> logs" cannot be implemented directly. Init to `CINIT − T(x)` and **add**, with
> survivor ⟺ cell ≥ `CINIT`. Identical test, no borrow, no CAS. Verified
> byte-exact against a CPU replay.
>
> The occupancy column of the table above turned out to be the single most
> load-bearing thing in this document: measured 32.4% / 64.2% / 95.7% at
> 2^15@512 / 2^15@1024 / 2^14@512 threads, driving apply from 7.32 to 3.08 ms.
> **Region 2^14 is the operating point**, not 2^15.

---

## Amdahl: cofactorization is co-equal, not a footnote

This is the largest structural risk in the plan and the original ladder under-weighted it. **With the test-sieve data in hand it is no longer just a risk — the arithmetic in "Concrete target" shows the verdict lands on the cofactor share `f` and on essentially nothing else.**

If `las` splits 60/40 sieve/cofactor, then a **10× faster sieve gives 2.2× overall**. A *perfect, free* sieve gives 2.5×. The GPU win is eaten almost entirely, and no amount of kernel tuning recovers it. The modeled GPU sieve time (~45–75 ms/q) is already below the CPU cofactor floor (~0.12 s/q at `f=0.5`), which means we are plausibly *starting* in the regime where this paragraph binds.

On this job — `lpba=32`, `mfba=92`, three large primes — the cofactor share will be large. **Measure it before writing any CUDA.** That single measurement bounds the entire project's ceiling and costs an afternoon.

The mitigation is already built into CADO and is better than the original "use GMP-ECM's CUDA path" plan:

- **`las -batch`** — batch cofactorization via prime product trees. Extract everything below `2^batchlpb` with a batch remainder tree, leaving only hard cofactors for ECM. Remainder trees are bignum multiply chains: highly parallel and genuinely GPU-mappable, unlike divergent per-cofactor ECM.
- **`las -batch-print-survivors <basename>`** — dumps survivors to files for external cofactorization. This is precisely the architecture the GPU project wants (GPU sieves → emits survivors → separate cofactor stage), and CADO supports the split natively.
- **`precompbatch` / `finishbatch` / `finishecm`** are already built in `sieve/ecm/`.
- **`las -stats-cofact <file>`** writes cofactorization statistics directly.
- **lasieve5 (`~/yafu/factor/lasieve5_64/batch_factor.c`)** is a second, independent reference for exactly this 3LP batch-cofactor architecture — the Franke/Kleinjung code that production 3LP jobs actually use.

Plan for cofactorization as a first-class component with its own throughput target: it must sustain the survivor rate the GPU sieve produces, or the sieve speedup is wasted.

---

## Step 0 — Collection and provisioning (before any measurement)

Everything below was audited on this box on **2026-08-01**. Path 0's numbers are only worth what the setup underneath them is worth, so this is the checklist that has to close *before* the first `las` timing run.

### Verified present — nothing to do

| Item | Status |
|---|---|
| `las`, `makefb`, `las_tracek`, `las_descent` | present in `~/cado-nfs/build/DESKTOP-3J4UC68/sieve/` (CADO `0574bc39d`, gcc 13.3, `-O3 -march=native`) |
| `las` flags the plan depends on | **all confirmed in `-help`**: `bkthresh`, `bkthresh1`, `bkthresh2`, `-B`/`log-bucket-region`, `-Bi`/`log-bucket-region-step`, `bkmult`, `dumpfile`, `batch`, `batch-print-survivors`, `stats-cofact`, `-T`, `-v`, `production`, `adjust-strategy`, `fbc`, `-I`/`-A`, `q0`/`q1`/`rho`/`nq`/`todo` |
| Batch-cofactor binaries | `sieve/ecm/{precompbatch,finishbatch,finishecm,issmooth,testbench}` all built |
| `CADO LOG_BUCKET_REGION = 16`, settable downward only | confirmed at `las-config.cpp:14` (`>16` rejected at `:25`); `LOG_BUCKET_REGION_step = 8` |
| 4-byte `shorthint` bucket update | confirmed — `static_assert(sizeof(bucket_update_t<1, shorthint_t>) == 4)`, `bucket.hpp:281` |
| Root-transform reference | `sieve/las-fbroot-qlattice.hpp` (`_31bits`, `_31bits_batch`; `_127bits` exists, `_127bits_batch` is `#if 0`) |
| GGNFS cross-check sievers + prepared job | `../test-sieve/`: `gnfs-lasieve4I{14,15,16}e`, `input.job`, `input.job.afb.0` (61 MB, regenerated 2026-08-01) |
| lasieve5 batch-cofactor reference | `~/yafu/factor/lasieve5_64/batch_factor.c` |
| msieve-s prior results | `~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`, `gerbicz_bench/{Makefile,RESULTS.md}` — the `-gencode arch=compute_120,code=sm_120 -O3 -std=c++17` Makefile is the template to copy verbatim |
| Toolchain | CUDA 13.2.78 (`/usr/local/cuda` → `cuda-13.2`; 12.8 also installed), driver 610.47, `nvcc` for `sm_120`, `ncu` + `nsys` + `compute-sanitizer` present |
| GPU power telemetry | works — `nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv -l 1` returns live watts under WSL passthrough |
| Disk | 623 GB free on `/` |

### Gaps that must be closed — with the fix

1. **The staged CADO oracle is for the wrong number.** `../test-sieve/cado_tmp.poly` is a **C203**; `input.job` is the **C183** target. `cado_roots1.gz` (104 MB, dated Apr 19) was generated from that C203 poly. `.fb_params.sha256` (Apr 19) vs `.afb_params.sha256` (Aug 1) records the drift. `cado_test_sieve.sh` hashes `n + lim1 + poly` and *will* regenerate on its own — but that means **a full `makefb -lim 134200000` run is a Path-0 prerequisite, not a freebie**. Run it deliberately, with `-t 16`, and time it.
2. **No rational-side factor base exists.** `las` recomputes side 0 on the fly and the script only passes `-fb1`. The GPU needs it materialized: `makefb -poly cado_tmp.poly -side 0 -lim 67100000 -out cado_roots0.gz`. Verify makefb accepts the degree-1 side (`-side` is documented but its default is "the unique algebraic side").
3. **Build the FB cache once.** Path 0 is a *sweep* — dozens of `las` invocations over `bkthresh`/`-B`/`-Bi`. Generate `-fbc <file>` on the first run so every subsequent run skips FB parsing. Do this before timing anything, and exclude the cache-building run from the numbers.
4. **The CPU baseline is currently handicapped — fix it or the Gate-1 comparison is unfair in our favour.** `~/cado-nfs/local.sh` sets `-DSIZEOF_P_R_VALUES=8`, so this `las` stores every FB prime and root as 64-bit. Our `lim1` is 134.2M — well under 2³² — so **4 bytes suffice and would halve FB-walk and small-sieve traffic on a CPU whose whole advantage is its 96 MB V-Cache**. Build a second `las` in a separate build dir with `-DSIZEOF_P_R_VALUES=4`, time both on the same q's, and **make the faster one the baseline.** The doc's premise is that we grade against a best-case opponent; a 64-bit-FB build isn't one. (Also confirmed compiled in: `BUCKET_SIEVE_POWERS`, `LOGNORM_GUARD_BITS=1.00` — record both, they affect parity.)
5. ~~`ncu` unproven under WSL2~~ — **tested 2026-08-01, works with one real limitation.** See "Profiler capability" below.
6. **Python deps: use a venv in `harness/`, not `pip install` and not `pipx`.** This box is PEP 668 externally-managed (`/usr/lib/python3.12/EXTERNALLY-MANAGED` present), so plain `pip install` refuses. `pipx` is installed but is the wrong tool — it packages *CLI applications* into isolated venvs and exposes their entry points; `nvidia-ml-py` is an importable library with no CLI, so pipx can't make it importable. `apt install python3-pynvml` exists (12.535.133-1) but mixes project deps into the system and the harness will want matplotlib/pandas too. **`python3 -m venv harness/.venv` is the answer.** Note it may not be needed at all: `nvidia-smi --loop-ms=100 --format=csv` as one long-lived streaming subprocess is already verified working and has zero dependencies — that is the default, and pynvml is only worth it if in-process sampling turns out to matter.
7. **GMP: `/usr/local` is the right one, and it is mistuned.** Correcting an earlier note in this doc: `/usr/include/gmp.h` is absent but **`/usr/local/include/gmp.h` exists**, and `/usr/local/lib/libgmp.so.10.5.0` is **byte-identical (sha256) to `~/gmp-6.3.0/.libs/libgmp.so.10.5.0`** — so `/usr/local` *is* `~/gmp-6.3.0` installed, and it is what `las` and `finishbatch` actually link (`GMP_LIB=/usr/local/lib/libgmp.so` in CMakeCache; confirmed with `ldd`). `~/gmp-fat` is an unrelated `--enable-fat` build (3 `cpuvec` dispatch symbols), installed nowhere and linked by nothing. **Use `/usr/local`; ignore `~/gmp-fat`.**

   **But that build is tuned for AMD K8.** `~/gmp-6.3.0/gmp-mparam.h` is a symlink to `mpn/x86_64/k8/gmp-mparam.h`, and `config.log` records only `build='x86_64-pc-linux-gnu'` — generic. This CPU is family 26 (0x1A, Zen 5); GMP 6.3.0 predates it, `config.guess` didn't recognize it, and configure fell back to the 2003-era K8 path. Meanwhile `mpn/x86_64/zen3/` ships tuned `mul_basecase`, `sqr_basecase`, `addmul_1`, `mul_1`, `sbpi1_bdiv_r` — **exactly the routines that dominate bignum product trees** — and they are sitting unused.

   Why this matters more than it looks: **CADO's batch cofactorization is GMP product trees**, and cofactorization is now the number the entire verdict rests on. (Non-batch `las` is largely unaffected — 92-bit cofactors go through CADO's own `modredc`, not GMP.) The bias runs *against* the project: a slow GMP inflates `f`, and pipeline speedup ≈ `1/f`.

   **Root cause, precisely:** the original configure line was `./configure --prefix=/usr/local --enable-cxx CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native"`. `-march=native` looks like it should cover this — **it does not.** GMP's speed comes from hand-written `.asm` in `mpn/`, and which of those files gets compiled is chosen by the **config triplet**, not by compiler flags. `config.guess` returned bare `x86_64-pc-linux-gnu`, so the assembly came from the generic/K8 path while the surrounding C got Zen 5 codegen.

   **Rebuilt and measured, 2026-08-01.** GMP's name for the target is **`zen3`**, not `znver3` (`config.sub` rejects the latter). Built out-of-tree into `~/gmp-zen3` so `/usr/local` and `~/gmp-6.3.0` are untouched:

   ```
   cp -a ~/gmp-6.3.0 ~/gmp-zen3-src && cd ~/gmp-zen3-src && make distclean
   ./configure --build=zen3-pc-linux-gnu --prefix=$HOME/gmp-zen3 --enable-cxx \
               CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native"
   make -j8 && make install
   ```
   `gmp-mparam.h` now resolves to `mpn/x86_64/zen3/gmp-mparam.h` (was `mpn/x86_64/k8/`).

   Measured on product-/remainder-tree sized operands, three interleaved A/B passes under `LD_LIBRARY_PATH` (box was loaded, so treat as ±3%):

   | operand | `mpz_mul` | `mpz_tdiv_r` |
   |---|---|---|
   | 128 Kbit | 8–15% faster | 8–17% faster |
   | 1 Mbit | 13% faster | 13% faster |
   | 8 Mbit | 10% faster | **15% faster** |

   **Consistent 10–15%, and the remainder-tree primitive (`mpz_tdiv_r`) gains the most** — that is precisely the operation CADO's batch cofactorization is built from. Worth taking. Still **do not overwrite `/usr/local`** (msieve and yafu link it); select it per-run with `LD_LIBRARY_PATH=$HOME/gmp-zen3/lib`.

   **Whether it changes anything is contingent.** GGNFS does its own cofactorization and never touches GMP for it, so this cannot move the GGNFS numbers. It bites only if CADO's Gate-0 run uses `-batch`. Measure `las` both ways before concluding it mattered.
8. ~~CUB moved in CUDA 13~~ — **tested: `#include <cub/cub.cuh>` compiles and links with no `-I` flag** under `nvcc 13.2 -gencode arch=compute_120,code=sm_120 -std=c++17`. `CUB_VERSION 300200`; `cub::DeviceScan::ExclusiveSum` runs. Closed.
9. **`cuda-sieve/` is not a git repo.** `git init` before the first commit of anything measured; every number in this doc should be attributable to a revision.

### Device properties — measured, not assumed (2026-08-01)

| property | value | vs. this doc's assumption |
|---|---|---|
| compute capability / SMs | `sm_120`, **48 SMs** | ✓ |
| global memory | 11.94 GB | ✓ (12 GB) |
| **L2 cache** | **48.00 MB** | ✓ (doc said ~40–48 MB) |
| **L2 persisting-access max** | **30.00 MB** | ✗ **new constraint — see below** |
| smem/block default / opt-in | 48 KB / **101376 B (99.0 KB)** | ✓ (doc said 99 KB opt-in) |
| smem per SM | 102400 B (100 KB) | — |
| **max threads/block vs. threads/SM** | **1024 vs. 1536** | ✗ **new constraint — see below** |
| max blocks/SM, regs/SM | 24, 65536 | — |
| memory | 14001 MHz, 192-bit → **672.0 GB/s** | ✓ exactly |
| SM clock | 2587 MHz base | — |

Two findings that change design choices:

- **The ≥50 KB region options cannot reach full occupancy — ever.** At 96 KB or 64 KB of shared memory you get exactly 1 block/SM, and a block caps at **1024 threads** while the SM holds **1536**. So those rows of the region table are structurally limited to **66.7% occupancy**, no matter how the kernel is written. The doc's note "1 block/SM needs 512–1024 threads/block to hide latency" understates it: 1024 threads/block is not a target, it's a ceiling that still leaves a third of the SM's thread slots empty. The 48 KB (2 blocks × 768) and 32 KB (3 blocks × 512) options can hit 100%. **Weight the region sweep toward the smaller regions**, and treat the 96 KB row as a latency-hiding experiment rather than the default.
- **L2 super-buckets can only be pinned to 30 MB, not 48 MB.** `persistingL2CacheMaxSize` is 30 MB — that is the cap on the `cudaAccessPolicyWindow` carve-out. The two-level design's "super-buckets sized to L2 (~40–48 MB)" is therefore only achievable as *incidental* residency, not as a managed carve-out. Size super-buckets against **30 MB** if you intend to pin them, or against 48 MB and accept normal LRU eviction. Decide explicitly and record which.

### Profiler capability under WSL2 — tested, and it constrains Path 1

`ncu` runs, `--set full` works, and SM/L1/L2 counters are all available. **But absolute DRAM counters are not exposed under WSL2 virtualization:**

| metric | result |
|---|---|
| `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__sectors_read.sum` | **`n/a`** |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ✓ works (86.44% on the probe kernel) |
| `lts__t_sectors.sum` and `_op_read` / `_op_write` / `_op_atom` / `_op_red` splits | ✓ works |
| `lts__t_sector_hit_rate.pct`, `lts__throughput...pct_of_peak` | ✓ works |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | ✓ works — Path 2's key metric |
| `smsp__warp_issue_stalled_*_per_warp_active.pct` (long_scoreboard, membar, lg_throttle) | ✓ works |
| `launch__registers_per_thread`, `sm__warps_active...pct_of_peak`, `gpu__time_duration.sum` | ✓ works |

**Substitution required in the instrument list:** "bytes moved per update" cannot be read directly off a DRAM counter. Use instead:

- **transactions per update** = `lts__t_sectors.sum ÷ n_updates` — *this works, and it is the number that actually matters.* It is also exactly the metric msieve-s's "L2 transaction ceiling" results were expressed in, so the two projects stay directly comparable.
- **DRAM bytes** = `dram__throughput.pct × 672 GB/s × gpu__time_duration.sum` — derived, good to a few percent, adequate for the record-size comparison.
- The read/write split comes from `lts__t_sectors_op_read/op_write`, which is *better* than what the DRAM counters would have given.

Nothing in Path 1 or Path 2 is blocked. Record the substitution so nobody later reports a `dram__bytes` figure that silently came back `n/a` and got treated as zero.

### Resolved with the owner (2026-08-01)

- **CPU power: LibreHardwareMonitor** on the Windows host (Ryzen Master also available as a cross-check). RAPL is confirmed dead inside the guest — `/sys/class/powercap` does not exist, so `turbostat` and `perf` energy counters are out with no software fix. Log "CPU Package Power" to CSV and align by timestamp. Expected ~110 W; see the power section for why the DRAM exclusion matters more than the exact value.
- **Operating point: `-A 29` (= I=15), `-adjust-strategy 0`, `-sqside 1`.** Locked; see "Concrete target".
- **Two jobs:** C183 for every graded number, a ~C120 for Path-3 parity debugging only. See "Concrete target".
- **Quiet box:** GPU frees up ~45 min from the audit (it was at 100% / 216 W running `/ecm`, alongside four `python3` processes at ~95% CPU and ~45 GB RSS). Nothing is measurable until then, and the CPU side needs the python jobs stopped too.

- **q-range: `[50M, 190M]`**, benchmark band frozen at `q0=120M, qintsize=1000`. See "Concrete target".
- **Baseline configuration prior:** the owner runs CADO at **8 workers × 2 threads** as the fastest arrangement on this HT box, and GGNFS at 14–16 single-threaded workers (GGNFS `lasieve4` has no threading at all). Treat 8×2 as the configuration to beat rather than the one to assume — but it is where the sweep should start.

Nothing else is blocking.

### Order of operations — split by what needs the CPU

The box is running long jobs the owner does not want to lose (four `python3` processes, 26+ hours elapsed, ~45 GB resident). **Everything on the GPU track runs fine alongside them**; only the CPU-baseline track has to wait.

**Runs now — GPU free, CPU busy, no CADO needed:**

1. Scaffold `bench/`; copy the `gerbicz_bench` Makefile (`-gencode arch=compute_120,code=sm_120 -O3 -std=c++17`).
2. **Path 1.** Load `oracle/input.job.afb.0` directly (format decoded, already SoA), walk real `(p,r)` progressions, implement variants (a)–(d) and (T), sweep record size. Grade in **ms/q** against the 182 ms / 56 ms lines above.
3. Path 2 (apply kernel, shared-memory region, bank-conflict rate) follows immediately — same inputs, same harness.

**Waits for a quiet box:**

4. CADO Gate 0 — confirm `f ≈ 0.23` on `las`, and check whether it truncates the factor base at `q` (GGNFS does; CADO reportedly does not).
5. `makefb` both sides for `oracle/c183.poly`; `-fbc` cache. *(Only needed for CADO parity work in Path 3, not for Paths 1–2.)*
6. Second `las` build at `SIZEOF_P_R_VALUES=4`; A/B against the Zen3 GMP. *(Baseline-fairness work — and note the baseline is now GGNFS, so this only matters for the CADO oracle's own speed, which demotes it.)*
7. `N_eff` configuration sweep — but for **GGNFS** (14–16 single-threaded workers) now that it is the baseline, not just for CADO.

Item 7 is the highest-value CPU measurement, because `N_eff` is the divisor in every target number on this page and is currently an assumption.

### Superseded ordering (retained for the record)

1. ~~Close the user-blocked items~~ — done.
2. `git init`; scaffold `bench/ harness/ oracle/ engine/`; copy the `gerbicz_bench` Makefile.
3. ~~Smoke-test `ncu` and `cub.cuh`~~ — **done 2026-08-01.** Both pass; see the profiler section for the one substitution required.
4. **Fix the two CPU-baseline handicaps, then pick the baseline.** Second `las` build at `SIZEOF_P_R_VALUES=4`; Zen3-tuned GMP in a scratch prefix. Time the four combinations on the frozen q-band and take the fastest as the opponent.
5. `makefb` both sides for the C183 (`oracle/c183.poly`); build the `-fbc` cache; checksum and freeze into `oracle/`.
6. Then Path 0.

Step 4 is the one that can move the verdict, because both handicaps bias in the project's favour and one of them (GMP) sits directly on the cofactorization path that Gate 0 measures.

**Frozen already (2026-08-01):** `oracle/` holds `input.job`, `result.job`, `c183.poly` (CADO format, generated here — it never existed for this number), `input.job.afb.0`, the GGNFS test sieve and its derived per-q table, the owner's scripts, and `MANIFEST.sha256`. `../test-sieve/` is a live directory whose polynomial is expected to be replaced; nothing in this project should read from it again.

---

## The ladder (fail-fast order)

**Ordering, after the GGNFS timing breakdown:** with `f ≈ 0.23` measured rather than feared, the ladder's original order stands — the GPU sieve is the binding constraint and Path 1 is the right next step. Path 4 is a *ceiling raiser* (moving resieve/TD to the GPU lifts the cap from 4.2× to ~10×), not a rescue, so it stays after Paths 1–3. The one thing worth pulling forward is the **CADO** `f` measurement in Path 0, since the whole ordering rests on `f` being ~0.23 on CADO too and not just on GGNFS.

### Path 0 — Baseline, oracle, and the Amdahl gate — **do this first, it is a decision point**

Prerequisite: **Step 0 above is closed.** In particular the baseline `las` build is the faster of the two `SIZEOF_P_R_VALUES` variants, the FB cache exists, and the box is idle.

This is not just a baseline. It is the cheapest, highest-information measurement in the whole project, and it can kill the idea before a line of CUDA is written.

#### Baseline siever configuration — sweep it, don't assume `-t 16`

The baseline must be *the box's best*, not a default, or Gate 1 is rigged in the GPU's favour. Three facts set the shape of the sweep:

- **hwloc is not compiled into this build** (`cado_config.h:122` — `/* #undef HAVE_HWLOC */`). So `las` cannot place subjobs or bind memory: `job-memory` is documented as "(unused, needs hwloc)", and `las-parallel.cpp:696` refuses every placement specifier except plain integers. **Multi-instance parallelism must be external `las` processes pinned with `taskset`** — which is exactly what CADO's own 8-way workunit split already is.
- The 9800X3D is **single-CCD**: 8 cores, one shared 96 MB L3. No NUMA, no cross-CCD penalty — a genuine simplification. But 16 threads on 8 cores means SMT is a real variable: sieving has memory-latency-bound phases where SMT helps, and cache-resident phases where two threads per core halve the effective private cache.
- Bucket memory scales with concurrent special-q, not with threads. At `-A 29` each in-flight q needs ~2.3 GB of 4-byte bucket records plus `bkmult` margin, so **8 processes ≈ 20+ GB**. The box has 76 GB, but check headroom before launching the wide configurations.

Sweep at fixed total 16 hardware threads, same q-set, same wall-clock budget, and pick the configuration with the highest aggregate relations/sec:

| config | note |
|---|---|
| **8 × `-t 2`** | **the owner's production setting and the fastest he's found on this HT box — start here, it's the one to beat** |
| 1 × `las -t 16` | one process, all threads — max cache sharing, one q in flight, minimum bucket memory |
| 2 × `-t 8`, 4 × `-t 4` | intermediate |
| 16 × `-t 1` | max independence, max bucket memory (~35+ GB) |
| 8 × `-t 1` on physical cores only (`taskset` to one thread per core) | isolates the SMT question |

**The scaling factor from this sweep is a load-bearing number, not a footnote.** Single-core cost is a measured 3.1 s/special-q; whole-box per-q is `3.1 / N_effective`, and `N_effective` sets the GPU's entire budget. A 0.7-vs-0.9 scaling difference moves the green threshold by 25%. Report it explicitly as `N_effective = 3.1 s ÷ (measured whole-box seconds per special-q)`.

Record aggregate relations/sec **and** package power for each — they don't peak at the same point, and the verdict metric is the ratio.

#### The measurements

1. Run CADO `las` on one special-q range at the locked operating point, using the winning configuration from the sweep, `-production` for the timing runs. Record relations/sec and CPU package power (LibreHardwareMonitor on the Windows host — RAPL is *not available* under WSL2).
   - For oracle and parity work, always use **one process, `-t 1`, one q**: deterministic, no thread interleaving in dumps.
2. **Capture the sieve-vs-cofactorization split** from the end-of-run timing report, `-T -T` (fine-grain timings, per-q at two `T`s), and `-stats-cofact`. *Decision:* if cofactorization dominates, the sieve ceiling is low — either commit to solving cofactorization too (see above) or stop here.
3. **Extract the CPU-side Gate-1 comparator while you're here:** updates/sec/joule for `las`'s fill phase ≈ (6.1e8 updates/q) ÷ (fill seconds per q from `-T -T`) ÷ (package watts during the run). Approximate is fine; write down how it was computed.
4. **Sweep the CPU-side cost structure.** `las` exposes `bkthresh`, `bkthresh1`, `bkthresh2` (2- and 3-level bucket sieving), `-B` (log bucket region), `-Bi` (log fan-out per level), `bkmult` — all verified present in the local build. Sweeping these on the CPU teaches the fill/apply cost structure *for this exact job* in a day, and every knob has a direct GPU analogue. Do this before designing the GPU fan-out.
5. Dump the factor base — **both sides**, freshly generated for the C183 (Step 0 items 1–2); the staged `cado_roots1.gz` is for a different number and side 0 has never been materialized at all. Plus a known survivor set for one special-q via `-batch-print-survivors`. This is the correctness oracle.
6. Capture per-q `las -v` output: it prints `# Sieving <q-lattice basis>; I=...; J=...` and `# Checksums over sieve region: after all sieving: ...`. The basis and J are needed to replicate a q exactly; the checksums are a cheap correctness summary.
7. Grab a **byte-exact sieve-region dump** for one or two special-q via `-dumpfile <stem>` (writes `<stem>.<side>.sq<q>.rho<r>.side<N>.dump`). This is a *stronger and more debuggable* oracle than checksums — you can diff to the first divergent byte.

Cross-check against `gnfs-lasieve4I15e` using the existing `test_sieve.sh` tooling, since that's the pipeline you already trust. Use CADO as the primary oracle — it's better instrumented and it's what the reuse map depends on.

### Path 1 — Fill-kernel microbenchmark — highest info per unit effort

**Unblocked as of 2026-08-01: this path needs no CADO, no `makefb`, and no CPU time.** Every prerequisite is closed — `ncu` and CUB verified, device properties measured, and the input factor base is already frozen in `oracle/`.

#### The input is real, not synthetic

`oracle/input.job.afb.0` is the GGNFS algebraic factor base for this exact job, and its format has been decoded (`oracle/ggnfs_afb_format.txt`):

```
word 0        : uint32 n = 7605406        (matches "FBsize (deg 5)" in the las log)
words 1..n    : uint32 primes[n]          non-decreasing, max 134199991
words n+1..2n : uint32 roots[n]
words 2n+1..  : 6 trailer words
```

**It is already in exactly the SoA layout this document asks for** — `primes[]` and `roots[]` as separate contiguous `uint32` arrays. A GPU loader is a `fread` and two `cudaMemcpy`s; no parsing, no conversion, no synthetic generator.

Statistics measured directly from it, which independently confirm the sizing table:

| | measured | previously |
|---|---:|---|
| `(p, r)` pairs | 7,605,406 | doc said ~7.6M ✓ |
| distinct primes | 4,817,761 | S₅ prediction was ~4.75M ✓ |
| pairs per distinct prime | **1.579** | S₅ prediction 1.58 ✓ |
| entries above `bkthresh` 2^15 | 7,601,775 | |
| algebraic updates/q @ A=5.37e8 | **3.152e8** | implies ~2.9e8 rational → **6.1e8 total, the doc's number** ✓ |
| **modular inversions/q** (both sides, per-prime batched) | **8,775,055** | doc originally assumed 11.6M |
| entries with `root ≥ p` (projective) | **4** — `(3,3) (5,5) (7,7) (11,11)`, all `r == p` | |

That last row is a genuine relief: **projective roots are a 4-entry special case on this job**, not a pervasive concern. The parity gotcha is real but tiny, and all four are below `bkthresh` so they never reach the bucket path at all.

**Still walk real `(p, r)` progressions** (positions stepping by `p` from the transformed root) rather than feeding kernels uniform-random `(bucket, offset)` records: real fill has per-thread monotone position order and an `A/p`-shaped volume skew toward small `p`, which is what the run-aggregation lever and the slice structure exploit. A uniform-random benchmark mismeasures both. With the real FB loaded there is no excuse for synthetic input.

Note the FB is truncated at `q` (see "Which siever is the opponent"), so size the benchmark to the **frozen band q=120M ≈ 7.0M algebraic entries**, and sweep 3.0M (q=50M) to 7.6M (q≥170M) to see the range the real job spans.

#### Result, 2026-08-01 — first pass built and measured

Built and run; full data in `bench/RESULTS.md`. Correctness first: the FK walk
matches brute-force enumeration exactly, and the GPU landed **1,690,293 records
against a CPU reference of 1,690,293**. Bucket imbalance **1.03×** — the doc
predicted "mild"; confirmed, and the slowest-block-bound trap does not bite here.

| stage | ms / special-q (algebraic side) |
|---|---:|
| transform + plattice (T) | **1.78** |
| fill, single-level 4 B | **12.09** |
| **total** | **13.87** |
| *both sides, extrapolated* | *≈ 28* |

Against **182 ms** (tie the box) and **56 ms** (TD floor). Apply, small-prime
sieve, norm init and threshold scan are **not yet measured**, so this is not yet
a Gate-1 answer — but the fill and transform stages together are using a quarter
of the floor budget.

Two design decisions were overturned (2 B records, two-level fan-out) and two
predictions confirmed (run-aggregation worth **2.3×**; the transform is a
non-issue on GPU at **1.78 ms vs the CPU's 373 ms — a ~200× win**, the largest
single-stage gain measured). See the Decisions-locked list and the fan-out
section for the corrections.

#### The number Path 1 must produce

Not a bandwidth percentage. **Milliseconds per special-q for the full GPU sieve chain** (transform + fill + apply + small-sieve + threshold), against two lines derived from the GGNFS breakdown at `N_eff ≈ 13`:

| | ms/q | meaning |
|---|---:|---|
| GPU-replaceable CPU work (Sieve + medsched + Sieve-Change) | **182 ms** | match this and the GPU merely ties the box on the sieve |
| CPU-resident floor (TD stays on host) | **56 ms** | get below this and the sieve is no longer the constraint |
| modelled GPU sieve | 45–75 ms | the hypothesis under test |

Land under ~60 ms → 4.0× box, perf/watt ≈ 1.33, green. Land at 150 ms → 1.6×, perf/watt 0.53, gray. **That factor of 2.5 in kernel quality is the whole question**, and it is measurable without a single CPU cycle from the busy box.

Implement fill several ways and measure all of them:

| variant | what it tests |
|---|---|
| (a) atomic-append | the naive baseline to beat |
| (b) single-level multisplit | count/scan/scatter, 16K-way |
| (c) two-level, smem-staged flush | the recommended design |
| (d) (c) with estimated sizing, no histogram pass | removes a full pass |
| (T) root-transform producer | the ~11.6M modinv/q stage feeding all of the above |

Cross each against **record size ∈ {2 B, 4 B, 8 B}** — this axis matters as much as the kernel structure.

Measure: achieved global bandwidth vs. peak, **updates/sec/joule** (the metric that actually maps to the verdict), atomic-stall %, transactions per record. For (T): modinvs/sec and its share of the per-q budget. Use Nsight Compute.

This tests the central hypothesis with no NFS code. If nothing here can beat CPU `las` update throughput per joule, nothing downstream matters.

**Exploit the structure a generic sort throws away:** each `(p, r)` walk generates positions in *monotone increasing* order, so consecutive hits frequently land in the same bucket. Accumulate a run and emit it as one coalesced burst with a single slot reservation — one atomic per *run*, not per record. This is strictly stronger than warp-aggregated atomics because it aggregates along a single thread's work. It pays off most for the smallest bucket-sieved primes, which produce the most records (`A/p` is largest there) — i.e. exactly where the volume is.

### Path 2 — Apply-kernel microbenchmark

Given filled buckets, accumulate logs into a shared-memory region. Use the 16-bit-cell scheme above. Measure bank-conflict rate, throughput, and achieved occupancy at the smem settings in the region table. (Precedent that smem-residency is worth fighting for: the Gerbicz bench's Phase B ran ~3× faster whenever the per-bucket working set fit in shared memory at preserved occupancy.)

> **Result, 2026-08-01 — DONE. Full detail in `bench/RESULTS.md`.**
>
> Best config: **region 2^14, 4 B records, 16-bit cells, 512 apply threads.**
> Apply **3.08 ms/special-q**; algebraic-side chain now **17.4 ms** (T 1.80 +
> fill 12.30 + apply 3.08) against 182 ms to tie and 56 ms to reach the TD floor.
> Correctness: region 16384 at full I15e, 9,483 records replayed on the CPU,
> **0 cells differ** — that gate covers norm init, smem atomics, the log p
> lookup and the threshold test at once.
>
> - **The occupancy-ceiling table is the whole story.** Measured achieved
>   occupancy 32.4% / 64.2% / 95.7% for region 2^15@512 / 2^15@1024 / 2^14@512,
>   with apply at 7.32 / 5.17 / 3.08 ms. A **2.4× swing** predicted by a table
>   that needed no code. Region 2^14 costs +0.8 ms in fill and saves 4.2 ms in
>   apply; 2^13 gives back nothing and costs 5.5 ms more in fill.
> - **The safe scheme is the fast scheme.** 8-bit (unsafe) cells at 2^15 =
>   3.047 ms vs 16-bit at 2^14 = 3.077 ms. Byte cells only buy shared memory,
>   and shared memory is free to buy by halving the region. Likewise smem
>   `atomicAdd` vs a racy plain `+=` is **~1%**. The whole byte-atomic wrinkle
>   costs nothing to resolve correctly — see the strikethrough in that section.
> - **Bank conflicts: 1.63×**, inside the predicted 1.5–2× band. Init and scan
>   are conflict-free (0.1%); all conflicts are in the scatter (~2.5× on its
>   own, two shared accesses per record). Irrelevant given the point below.
> - **Apply is DRAM-bound at 58% of peak** — a different regime from fill's
>   17–18%. It reads 1.25 GB in 3.08 ms = 406 GB/s; that read at 100% of peak
>   is 1.86 ms, so ~1.2 ms of headroom and no shared-memory restructuring can
>   touch it. This is also why 2 B records stay rejected: they would halve
>   apply's stream but fill is 4× the cost and 40% worse at 2 B.
> - **The region never touches global memory** — init, accumulate, scan and
>   threshold all in smem; only survivors are written. Apply's DRAM footprint
>   is the bucket read plus ~5 MB. That is the structural edge over CPU `las`,
>   which must stream the region through cache.

### Path 3 — Single special-q, end-to-end

Wire fill + apply on the real dumped poly/FB. Sieve one special-q and threshold to survivors.

Be explicit about what "end-to-end for one q" requires beyond Paths 1–2 — this is where the hidden scope lives:

1. **q-lattice basis reduction** per special-q (host, microseconds; cross-check against the basis `las -v` prints). **Reduce under the *skewed* norm** `|(a,b)|² = (a/√s)² + (b√s)²`, not the plain one — measured 2026-08-01, this is not a refinement, it is required for the norms to be computable at all. An unskewed reduction gives |a| ~ |b| ~ √q, the homogeneous terms `c_k a^k b^(d−k)` then span **10³⁹**, `log2|F|` is set by `c₀b⁵` alone and the leading term underflows fp32 outright. Skewed: A = 2.33e12, B = 1.64e4, normalised coefficients all O(1) (`-0.0136, -0.0605, 1, -0.419, -0.0054, 0.381`). That balance is what the skew parameter is *for*, and it is the only thing that makes fp32 norm evaluation possible.
2. **Norm initialization** — the sieve array starts at the log-norm bound per position, not zero. A simple float evaluation of `log|F(i,j)|` per position is fine to start, **but exact parity with las requires reproducing its per-q log scale** (las rescales so norms fit a byte) and its piecewise approximation + rounding. Budget this; it's the main obstacle to byte-exact matching.
3. **Small-prime sieve** for `p < bkthresh` (~3.5K primes + prime powers per side) — a per-region shared-memory walk; a third kernel family, small but mandatory for any survivor parity. **BUILT AND MEASURED 2026-08-01 — and it is not small.** 7,143 entries across both sides, 0.1% of the factor base, producing **4.36e9 updates: 7.2× the entire bucket-sieve volume**, at **13.2 ms** against the 1–3 ms budgeted. It is 28% of the sieve chain. 84% of side 1's updates come from the 52 entries with p < 64, so the engineering problem is load balance, not throughput: three tiers (whole block / one warp / one thread per entry, cut at p=64 and p=1024) against a thread-per-prime loop that would hand 8,192 serial updates to whoever drew p=2. Keeping `log_region ≤ logI` puts each region inside one j-row, which reduces the per-region entry point to one multiply and one remainder and makes every block independent — CADO carries per-prime positions forward between regions because it visits them in sequence; we can't, and don't need to. Fused into the apply kernel, so the region is touched once. (The cheap-dodge suggestion is moot.)
4. Bucket fill + apply (Paths 1–2) + threshold test → survivor `(i,j)` list.

**Correctness gates, in pragmatic order:**
1. ~~Fill+apply parity against our own `verify_cpu.c` reference on the real FB (no las semantics needed — catches kernel bugs first).~~ **DONE 2026-08-01**, both sides, full I15e: records landed equal the CPU reference exactly and a full region replayed on the CPU gives 0 cells differing. Note the limit this gate structurally has: it compares us against ourselves, so it cannot see a wrong *lattice*. It didn't — `pl_transform` was transposing the basis and every self-check still passed. Gate the transform against its definition (`a ≡ r*b mod p`) separately; `verify_transform()` now does.
2. **PASSED 2026-08-02, by a different instrument.** The dumpfile route stays dead — las's dump carries ~41 log units per position, matching the *small-sieve-only* expectation (39.31) rather than the whole factor base (54.65), and correlates with our region at **+0.03** in either i-orientation against a **+0.64** control; it sits behind an `ASSERT_ALWAYS` that made its only `open()` call unreachable, so it has rotted. **Gate 5 replaced it outright and is strictly better:** `las_tracek` is a *stock* CADO target needing no patch, and it prints every ideal applied to a position with its log, rather than one byte. Result: **8 of 8 exact log sums**, four positions × two sides, with the *ideal lists* agreeing entry by entry — two of the positions chosen because long projective power ladders hit them. The comparison is made twice: once against a direct enumeration of the factor base (which gates roots and logs, and is what found the bugs), and once against the value read back out of `k_apply` via `bench --probe`, which has been through the transform, the walk, the tiering, the fill, the small sieve and the GPU apply. **The second is the one that closes the gate**; the first alone is ideal/log agreement, and describing it as sieve parity was an overclaim held for about an hour. Full detail in `oracle/PARITY.md` and RESULTS.md finding 26.

   Getting there cost two real fixes that nothing else would have found: las's printed `scale` is rounded to 2 dp and the exact values are **1.275 / 1.925** (enough to move `fb_log` by one for a band of primes), and the survivor bound is a *truncating* cast plus a guard bit, not a round. Norm initialisation still differs by 0..+2 — that is las's own approximation plus `LOGNORM_GUARD_BITS`, and **our fp32 evaluation is the more accurate of the two**, which is also why a byte-for-byte region diff could never have reached zero even with a working dumpfile.
3. Match the `-batch-print-survivors` survivor set (start as superset/subset stats with a threshold tolerance band; tighten to exact).
4. `las -v` checksums — once gate 2 passes these are free; use them as the cheap regression check thereafter.
5. `TRACE_K` / `las_tracek` follows a single `(i,j)` through las's pipeline — the right tool when one position mismatches.

Known parity gotchas, collected so nobody rediscovers them: same J as las (from `-v`; use `-adjust-strategy 0`), i-range is `[-I/2, I/2)`, skip positions with `gcd(i,j) > 1` conventions (`unsievethresh`), projective roots, prime powers (`powlim`), `skipped`/`tdthresh` small primes that las trial-divides instead of sieving, and the per-q log scale.

**Oracle captured 2026-08-01 — `oracle/PARITY.md`.** `makefb` + `las -v -dumpfile` on this polynomial at `-A 29 -B 14 -sqside 1`, giving both 512 MB sieve dumps, the log scales, and a real special-q (`q=120000053, rho=112625526`). Gates already passed against it:

| gate | las | ours |
|---|---|---|
| q-lattice basis | `a0=7374527 b0=-1 a1=120000053 b1=0` | `(-7374527, 1, 120000053, 0)` — same basis, first vector negated |
| log2(maxnorm) side 0 | 131.86 | **131.86** |
| log2(maxnorm) side 1 | 196.61 | 196.41 |
| one-sided survivors | 23.95M / 21.65M | 30.04M / 18.17M |

**Both closed 2026-08-01.** `fb_cado.c` parses makefb's text format, so the ideal set is now CADO's own: side 1 small part **3,839 = 3,839**, bucketed **7,601,776 vs 7,601,777**. Running at las's scale makes the survivor bounds fall out exactly — `round(1.28 × 3.5 × 32) = 143` and `round(1.93 × 2.35 × 31) = 141`, both las's numbers, with nothing fitted.

Powers cost more than they look: they take the small sieve from 4.36e9 to **6.84e9** updates and the chain from 47.1 to 55.5 ms. They also broke two assumptions that only held for prime moduli — `pl_invmod` is binary Euclid and **does not terminate** on the even moduli 2,4,…,32768 that CADO's factor base carries, and modulo `p^k` the denominator can be non-zero yet non-invertible (it hung on the first `q = 49`). The general form, which subsumes the old whole-rows case, is in `pl_transform_gen`: with `g = gcd(D, q)`, hits exist only when `g | j` and then `i ≡ rt·(j/g) (mod q/g)`.

**The strongest gate turned out not to need the oracle at all.** Expected sieved log per position is `Σ logp/q` over the factor base, computable straight from `c183.fb1`: **analytic 54.65, measured 54.2 — 0.8%**. One number exercising the factor-base parse, powers, the root transform, the small sieve, fill, apply and the `fb_log_delta` scaling. **We never need las's `scale`**: it exists only so `scale × log2(maxnorm)` fits in a byte (`1.28 × 196.61 = 251.7`), and 16-bit cells make it free to keep the resolution las throws away — 0.39 bits per position, which is fewer false survivors into cofactoring.

Two traps worth writing down. **las groups the q-lattice basis by coordinate** (`a = a0*i + a1*j`) and `qlat_t` groups by vector (`a = i*a0 + j*b0`); the translation is las `(a0,b0,a1,b1)` = qlat_t `(a0,a1,b0,b1)`, and getting it wrong transposes the basis into a valid lattice that no self-check rejects. And **only the special-q side's norm carries a factor of q** — dividing it out of both sides made side 0's threshold ~27 bits too generous (171.7M survivors instead of 30.0M).

`-dumpfile` also needed a one-line CADO patch: revision `0574bc39d` has `ASSERT_ALWAYS(!las.dump_filename)` immediately before the only `dumpfile.open()` call site in the tree, so the flag aborts unconditionally. Original file and revert command are in `oracle/`.

Then measure GPU sieve time for that q vs `las` for the same q.

### Path 4 — Cofactorization — reuse, don't reinvent (but do invest)

Recover FB primes on survivors via **resieve** (a second pass over survivors only — tiny), not re-division by the whole FB.

Route the large-prime remainder through CADO's batch path (`-batch` / `finishbatch`) rather than per-cofactor ECM — **as subprocesses first**; only link if the process boundary measurably hurts. If batch cofactorization becomes the bottleneck, the product-tree remainder step is the GPU-mappable part; GMP-ECM's CUDA/CGBN path handles the residual hard cofactors. lasieve5's `batch_factor.c` is the second reference implementation.

Per Amdahl above: this needs a throughput target, not just a "confirm it isn't the bottleneck" check — it must sustain the survivor rate Path 5's sieve produces.

### Path 5 — Throughput mode — the real number

Batch many special-q; overlap transform/fill/apply/cofactor with CUDA streams + graph capture (cuda-mpqs does exactly this; the per-slice launch structure from the 2 B record path is graph-friendly too). Slab the sieve area if running I16e. Cofactorization overlaps on the CPU while the GPU sieves the next q's.

Measure aggregate relations/sec/watt vs. the Path-0 baseline, **and compute $/relation with written-down rental prices**. This is the verdict, graded on the zones.

---

## Instrument from day one

- **relations/sec/watt** (verdict) — NVML for GPU power, package power for CPU. Sample continuously, not at endpoints.
- **updates/sec/joule** — the Path-1 proxy for the verdict. Report this instead of bare bandwidth percentages.
- **transactions per update** = `lts__t_sectors.sum ÷ n_updates` — the lever that actually moves the answer, and directly comparable to msieve-s's results. **Bytes per update must be derived, not measured**: absolute `dram__bytes_*` counters return `n/a` under WSL2 (see Step 0).
- Achieved memory bandwidth as % of peak — `dram__throughput.avg.pct_of_peak_sustained_elapsed` works. A diagnostic for "is this kernel well written," *not* a gate.
- Atomic/lock stall %, warp divergence, occupancy, shared-mem bank conflicts.
- **Max bucket occupancy, not mean.** See "slowest-block-bound" below.

### Power measurement on this box (WSL2 — verified, this bites)

- **GPU: works.** `nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=100` (WSL passthrough at `/usr/lib/wsl/lib/nvidia-smi`) or pynvml in the harness.
- **CPU: RAPL is absent under WSL2.** `/sys/class/powercap` doesn't exist here; `turbostat`/`perf` energy counters need MSR access the guest doesn't have. Options, best first:
  1. **HWiNFO64 (or LibreHardwareMonitor) on the Windows host**, logging "CPU Package Power" to CSV during the run; align by timestamp.
  2. **Wall-plug meter** for whole-box draw — arguably the *honest* number for $/relation anyway (it includes DRAM/VRM/fans). Report total draw, not delta-over-idle, for the economics; note idle separately.
  3. Last resort: carry a 120–160 W band as explicit uncertainty. Acceptable for early gates, not for the Path-5 verdict.
- For the final verdict, charge the GPU pipeline for the host cores it keeps busy (cofactor threads included), and the CPU baseline for its whole package. Whole-box wall power for both runs sidesteps the allocation argument entirely.

---

## Known traps

**Cache-residency illusion.** Benchmark at a size where the FB does not fit in L2, or you'll measure the wrong regime and get a falsely rosy result. Single easiest way to fool yourself.

**Uniform-random benchmark inputs.** Fill kernels fed random records instead of real `(p, r)` walks misrepresent run-aggregation, slice volume skew, and locality — in either direction. Walk real progressions (Path 1 note above).

**Slowest-block-bound, not average-bound.** msieve-s measured this twice (experiments #9, #10): improving *average* per-bucket convergence saved literally zero wall time because the long tail dictated kernel duration. If bucket fill is uneven, the apply kernel's duration is set by the fattest bucket. Instrument the max. Sieve positions are near-uniform so this should be mild here — but verify rather than assume, and don't celebrate a mean-case improvement until it shows up in wall time.

**Fusion can lose to register pressure.** msieve-s experiment #11 fused a scatter into its producer kernel: registers went 56 → 72, block_limit dropped 5 → 4, per-launch duration *doubled*, net +25% GPU time despite eliminating an entire kernel. The analogous temptation here is fusing the root transform with the bucket scatter. Check the register count before believing in it.

**Memory capacity is about buckets, not the FB.** The FB is 61 MB. Bucket storage at I16e is 8–16 GB and does not fit. Slab, or stay at I15e.

**Resieve, don't re-divide.** Trial-dividing survivors by the whole FB throws the win away.

**Scope creep.** Dump poly-select, filtering, linear algebra, and sqrt from CADO and use them as-is. Touch only the sieve.

---

## Closed levers (inherited from msieve-s — do not re-run these)

Paid for already on this exact GPU. Full detail in `~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`.

| Lever | Result | Implication here |
|---|---|---|
| `__stcs` cache hint on scatter writes | L2 sectors −3pp, duration flat | Scatter is transaction-ceiling-bound; cache hints won't save you |
| Removing warp-aggregation from scatter | Compute throughput −25pp, duration flat | Aggregation isn't the lever at 16K buckets; fan-out is |
| Fusing scatter into producer kernel | +25% GPU time (register pressure) | Check regs before fusing anything |
| `__launch_bounds__` to raise occupancy | 56 → 40 regs, but +5% duration | Math chains need their registers — applies directly to the root-transform kernel |
| Bucket-count sweep (2^12…2^17) | 2^14 optimal, but a balance point not a coalescing knob | Sweep it here too; expect a different optimum with two-level |
| Hash-salt sweep for bucket balance | Slow-tail count varied only 7.5% across 4 multipliers | Imbalance is structural to the key distribution, not hash-induced |

The pattern across all six: **micro-optimizing the scatter kernel does not work.** The levers that remain untried — and that this project should aim at — are **fewer bytes per record** and **two-level fan-out**.

---

## Kill criteria — zones, not cliffs

Three changes from the original version: bandwidth percentage is demoted from gate to diagnostic, the Amdahl check is promoted to a gate, and **the gates are graded green/gray/red rather than pass/fail** — a shortfall of 2× is a decision point with named options, not a kill. Only order-of-magnitude shortfalls with the levers exhausted kill the project.

**Gate 0 (Path 0) — cofactorization share. Partially answered already, on GGNFS: `f ≈ 0.23`.** The GGNFS fine-grain timings (complete accounting, see "Concrete target") put trial division + cofactorization at 21–24% of wall across the whole q-range, with hard cofactorization (MPQS) at only 8–11%. **This gate is provisionally passed** — the Amdahl ceiling is ~4.2×, not the ~1.7× a 60/40 split would have implied.

What remains is to confirm it on **CADO**, whose cofactorization strategy differs (ECM chains, `-batch` if enabled) and which may not truncate the factor base at `q` the way lasieve4 does. If CADO returns `f ≈ 0.23` too, proceed with the sieve as the primary target. If CADO returns `f > 0.5`, the discrepancy is itself the finding — it would mean CADO is the wrong baseline for this job and GGNFS is the siever to beat.

**Read Gate 0 as a property of the parameter regime, not of NFS.** The C183's cofactor share is driven by `mfba=92` with `lpba=32` — three large primes. A 2LP job (`mfba≈64`) spends far less time there. So while measuring the C183 split, spend one extra sweep point measuring it at `mfb1=64` as well. If the answer comes back "the GPU sieve only pays off in the 2LP regime," that is a **finding about where this technique applies**, not a kill — and it is cheap to learn at the same time as the primary measurement rather than after Path 5.

**Gate 1 (Path 1) — updates/sec/joule, GPU fill+transform vs. CPU `las` fill.** Re-scoped, given the per-q budget derived from the test-sieve data: this gate's job is no longer "is the GPU fast enough" — the model says it has 4–8× of headroom — but **"is the model wrong by 5×?"**
- Modeled GPU sieve time (~45–75 ms/q) confirmed within ~2× → the sieve is off the critical path. **Stop optimizing it** and spend the effort on Path 4. Do *not* pull the 2 B-record or two-level levers just because they're written down; they buy nothing below the cofactor floor.
- 2–5× worse than modeled → gray: now the named levers (2 B records, fan-out shape, `bkthresh` sweep, run-aggregation) actually matter. Pull them and re-grade.
- \>5× worse than modeled, across the entire record-size × fan-out grid → the sieve *is* the constraint after all, and the original framing of this document applies. Re-grade against the zones directly.

The cheapest useful output of Path 1 is therefore a single number — **milliseconds per special-q for the full GPU sieve chain** — not a bandwidth percentage.

**Gate 2 (Path 5) — relations/sec/watt and $/relation, graded on the verdict zones above.** Green: go. Gray: the rental-$/relation calculation and the best-case-opponent caveat decide, in writing. Red: stop, or pivot to offloading only cofactorization / linear algebra (which we already know works).

**Not a kill criterion:** achieved bandwidth as a percentage of peak. If a variant lands at 30% but moves 2 B/record, it beats a variant at 60% moving 8 B. Low bandwidth efficiency means *go optimize*, not *stop the project*. Report it, act on it, don't gate on it.

---

## Language and repo layout

**Decision: C++17 host, CUDA C++ device, Python for the harness only.** (Toolchain on this box, verified: CUDA 13.2, target `sm_120`.)

The reasoning:

- Kernels are CUDA C++ regardless. That's ~95% of the runtime and most of the intellectual work. Host language buys nothing where the risk actually lives.
- The host is never in a per-update or per-survivor inner loop. It does per-special-q lattice reduction (microseconds), buffer/stream orchestration, and calls into GMP. Host language has no measurable performance effect.
- **Through Path 3, nothing links against anything.** `las` is a subprocess oracle driven by the Python harness; GMP enters only for host-side q-lattice/norm setup. The first real integration decision is Path 4, and the cheap answer there is *also* subprocesses (`finishbatch` et al.). "FFI burden" is nearly zero in this plan — which is exactly why it should stay C/C++-shaped: every reuse target (CADO C++, GMP-ECM C, GMP C+asm, lasieve5 C) is a C-ABI-adjacent codebase, and Rust would reintroduce a boundary the plan otherwise never pays for. The safety benefit doesn't reach the kernels, which are CUDA C++ either way.
- Where the CPU *could* become performance-relevant: cofactorization feeding, if it stays host-side and the GPU paces it. That's a threading-design problem (per-thread batches, SPSC queues), routine in C++ and not a language argument.
- **Precedent:** `gerbicz_bench` in msieve-s used exactly this shape — standalone Makefile, nvcc, engine exported behind a 3-function C ABI (`collision_engine_init/free/run`), a C reference implementation as ground truth, and a CLI-driven sweep. It worked well enough to land a 2.75× kernel win and integrate cleanly. Repeat it.
- Python's job: run `las`, parse `-v`/`-T`/`-stats-cofact` output, drive parameter sweeps, sample `nvidia-smi`, diff oracles, plot. It never touches a record.

Suggested layout, mirroring what worked:

```
cuda-sieve/
  bench/          standalone; own Makefile (-arch=sm_120); no CADO dependency
    bench_main.cu   CLI: --fb-size --area --buckets --levels
                         --record-bytes --mode {atomic,multisplit,twolevel,est}
    trans_*.cu      root-transform producer kernel (variant T)
    fill_*.cu
    apply_*.cu
    verify_cpu.c    ground-truth reference implementation
  harness/        Python: parameter sweeps, NVML power sampling + host-side
                  CPU-power ingestion (HWiNFO CSV), plots, oracle diffing
  oracle/         dumped poly / FB / survivors / -dumpfile region dumps /
                  checksums / per-q basis+J lines from las -v
  engine/         later: the .so with a small C ABI, once it's real
```

Keep `bench/` buildable and runnable without CADO or msieve in the loop. That independence is what made `gerbicz_bench` a productive sandbox.

---

## Reuse map

| Need | Source |
|---|---|
| **Performance baseline (the opponent)** | **GGNFS `gnfs-lasieve4I15e`** — AVX-512 asm, ~20% faster than CADO here. Timings in `oracle/ggnfs_timing_breakdown.txt` |
| **Path-1/2 input factor base** | **`oracle/input.job.afb.0`** — already SoA `uint32 primes[]`/`roots[]`; format in `oracle/ggnfs_afb_format.txt`. No CADO or `makefb` needed |
| Algorithm reference, correctness oracle, everything-but-the-sieve | **CADO-NFS** — read `las` for special-q handling, bucket structure, resieve; use poly-select and filtering. **Not** the speed baseline |
| Linear algebra | **msieve** (via YAFU) — the owner reports CADO's block-Wiedemann is much slower. Out of scope for this project, but don't route anyone to CADO's LA on the strength of this doc |
| Factor base generation | CADO `makefb` — **must be re-run for the C183, both sides** (`-side 0 -lim 67100000`, `-side 1 -lim 134200000`); the staged `../test-sieve/cado_roots1.gz` belongs to a different poly |
| Survivor oracle | `las -batch-print-survivors`, `las -v` (checksums, per-q basis + J) |
| **Byte-exact sieve-region oracle** | `las -dumpfile <stem>` (verified in local build; align regions via `-B`) |
| Single-position debugging | `las_tracek` / CADO `TRACE_K` |
| Cofactorization timing | `las -stats-cofact`, `-T -T` fine-grain timings |
| Cost-structure sweep | `las` `bkthresh` / `bkthresh1` / `bkthresh2` / `-B` / `-Bi` / `bkmult` (all verified in local build) |
| Root-transform reference (incl. projective edge cases) | CADO `fb_root_in_qlattice` (32-bit REDC variants) |
| Scan / histogram primitives | CUB (`DeviceScan`, `DeviceHistogram`) — **not** `DeviceRadixSort` |
| Batch cofactorization | CADO `-batch`, `precompbatch` / `finishbatch` / `finishecm`; lasieve5 `batch_factor.c` as second reference |
| Hard-cofactor ECM | GMP-ECM CUDA / CGBN path |
| Strongest CPU-siever cross-check | lasieve5 (`~/yafu/factor/lasieve5_64/`, AVX-512 lattice siever + asm) |
| Async/overlap patterns | cuda-mpqs (streams, CUDA-graph capture, double-buffering) |
| Prior GPU tuning results on this exact card | `~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`, `~/msieve-s/gerbicz_bench/RESULTS.md` |
| Baseline / cross-check tooling | `~/code/test-sieve/` (`test_sieve.sh`, `cado_test_sieve.sh` + configured `.ini`, prepared `input.job` + `.afb.0`) |

---

## What transfers from the msieve-s sort win — and what doesn't

Worth being precise, since it's the most obvious analogy available.

**The algorithm does not transfer.** The Gerbicz engine replaced a full 56-bit radix sort with hash-bucketing plus an iterative in-bucket filter, because the actual requirement was *collision detection*, not order. The sieve has no dedup step, so there is no direct port.

**The sharper structural point:** in the Gerbicz pipeline, the 16K-way scatter (Phase A) was *the cost we paid* to delete the sort — the replacement's own overhead, and profiling showed it promptly became the new wall (34% of GPU time, at the L2 transaction ceiling, immune to micro-optimization). Here, that scatter **is the entire product**. There is no sort to delete; fill *starts* at the wall Gerbicz *ended* at. So expect no order-of-magnitude swap-win here. The available wins are: bytes-per-record (up to ~4× traffic vs. 8 B), two-level fan-out (partial-line → full-line writes), skipping the histogram pass, and run-aggregation. That's the honest shape of the opportunity — several ~1.5–2× levers that compound, not one big kill.

**The meta-lesson transfers exactly:** *you were paying for a sort to obtain a property strictly weaker than sorting.* Same mistake is available here — bucket fill needs a **partition**, and within-bucket order is free waste. That's why the CUB recommendation changed from `DeviceRadixSort` to multisplit.

**The measured results transfer directly**, and they're the real inheritance — six closed levers (table above) measured on this same RTX 5070, plus three patterns worth copying rather than rediscovering: static bucket sizing from an analytic estimate (`mean + 6σ` never overflowed in production), grow-and-retry as the overflow safety net, and keeping per-bucket working sets in shared memory whenever occupancy survives (worth ~3× on the Gerbicz filter phase — the same argument behind our in-smem apply kernel).

---

# Review notes — from Claude (the reviewer), 2026-08-01

Second review pass, at the owner's request. I read this doc end-to-end, `bench/RESULTS.md`, `oracle/PARITY.md`, and all of `bench/` (kernels, plattice port, loaders, verifiers, driver), cross-checked the plattice port against CADO's `las-plattice.hpp`, re-derived several of the measured tables, and factored Y1 to settle the projective question. Everything below is review; I changed no code.

## Overall verdict: not off track — the opposite

This is disciplined work. Two of the design decisions I put in this document (2 B records as the top lever, two-level fan-out as the design to beat) were **overturned by measurement, correctly** — the profiler evidence in Findings 1–2 is sound, the two-level post-mortem (2 MB of open lines vs 48 MB of L2) identifies exactly which premise failed, and the doc annotates the reversals in place instead of hiding them. Finding 10 (skew-aware reduction), Finding 15 (prime-power transforms), and the CINIT−T add-only convention are all catches that would have cost days downstream. Finding 16 — validating the sieve against `Σ logp/q` computed from the factor base itself — is the best kind of gate: oracle-free, one number, exercises the whole chain. Keep it as a permanent regression check.

The one strategic risk I see is *over*-polishing the sieve. At 55.5 ms against a 56 ms TD floor, further sieve milliseconds buy almost nothing until Path 4 exists — the doc's own Gate-1 re-scope says this, and it's right. The highest-value next spends are, in order:

1. **`N_eff` sweep on GGNFS** (needs the quiet box) — it is the divisor in every target number on this page and is still an assumption. Every graded claim ("47 ms vs 182/56") inherits its error bar.
2. **GPU watts during the chain** — the Gate-1 metric is updates/sec/**joule** and no joules have been measured. Loop the best config ~100 reps (~5 s), sample `nvidia-smi power.draw` at 100 ms, report ms/q × W. An hour of work; converts every headline number into the metric of record. I'd guess the chain draws well under the 220 W planning number (fill is latency-bound, not compute-bound), which would *improve* the perf/watt story — measure it.
3. **CADO Gate 0** (`f` on `las`, quiet box) — the 4.2× ceiling rests on GGNFS's `f ≈ 0.23` generalizing.
4. **Path 4 skeleton** (resieve + survivor intersection + batch-cofactor feed) — the ceiling raiser. The two-sided intersection is cheap on GPU: keep per-side survivor bitmaps (I·J/8 = 67 MB each), AND them, resieve only the intersection.
5. Small-prime sieve levers (below) — only after the above, and only if the sieve is ever the constraint again.

## Correctness findings (fix before parity work; none invalidate the timings)

**R1 — Bucketed projective side-0 entries are mishandled, and the CPU reference shares the blind spot.** Y1 = 59 · 101 · 127 · 281 · 1259 · **38321** · **5746453**. The last two are in [2^15, 67.1M], so side 0 has exactly two bucketed projective entries (encoded `root == p` by `rfb.c`). `k_transform` (`bench_kernels.cu:36`) passes `roots[k]` straight into `pl_transform`, which reduces `r = p ≡ 0` and returns a bogus *affine* root — the `rt >= p` check right after it only catches *output*-projective denominators, not *input*-projective roots. Result: ~14,010 (p=38321) + 93 (p=5746453) records per special-q land on a wrong lattice and add 29/43 scaled log units there — pure false-survivor pressure. `verify_count_updates` (`verify_cpu.c:197`) makes the identical mistake, so the "records landed = CPU reference exactly" gate cannot see it — the same failure class as the transposed-basis bug this project already wrote up. Fix in both places: `if (roots[k] >= p) { skip; count; }` *before* the transform — correct because every bucketed p > J = 16384, so a projective entry's only hits are on row j=0 (only i=0 survives the |i| < I/2 < p constraint, and (0,0) isn't a relation). Side 1 is clean for this poly (lc = 110880 = 2^5·3^2·5·7·11, all projective primes < 12), but the fix covers the loader path where `fb_load_cado` keeps `r ≥ q` encodings too.

> **Response, 2026-08-02 — R1's diagnosis is right, its fix is wrong. Do not
> apply it.** The recommendation to skip bucketed projective entries rests on
> "every bucketed p > J = 16384, so a projective entry's only hits are on row
> j = 0". That conflates the two coordinates. The projective condition
> constrains `b`, and `b = i·a1 + j·b1`; for a special-q of this size the
> reduced basis has b-components of order `sqrt(q/skew) ≈ 1`, and on this
> lattice `b = i` exactly. So the condition is `i ≡ 0 (mod p)` — one hit per
> row, on **every** row: 16,384 positions per entry, not one row's worth. (The
> review's own "~14,010" is `A/p`, which is the right order of magnitude for
> the wrong reason.) They are now transformed as projective ideals, not
> skipped. `fbtest` prints the `b(i,j)` form at startup so this cannot be
> mis-argued again.
>
> R1 also states that `k_transform` passes such roots into `pl_transform` and
> gets a bogus affine root. In the code as it stood they never reached
> `k_transform` — `fb_restrict` dropped them first. The bogus-affine-root path
> was real for a *different* input (see R2/R3), and is fixed.
>
> The genuine version of this bug was on the **algebraic** side and larger:
> projective roots with a **nonzero** reciprocal, 35 of them in `c183.fb1`,
> 4.7e8 updates and 1.54 scaled log units per cell placed on the wrong
> congruence. Same density as the right answer, which is why every gate passed.
> RESULTS.md findings 18–20.

**R2 — q = 32768 sits in the bucketed set with an even modulus.** `fb_split_small` cuts at `p < bkthresh`, so the top of the 2-ladder (2^15, present in `c183.fb1`) goes to the *bucket* path, where `pl_transform` → `pl_invmod_any` returns 0 whenever the denominator is even-but-not-invertible (~half of all special-q) and silently produces root 0 — one ideal, 1 log unit, wrong positions on those q. Cheapest fix: send moduli that are not prime (or the boundary power specifically) to the small tier, whose host-side `pl_transform_gen` handles them exactly; alternatively gate `k_transform` through the gen path for non-prime moduli.

**R3 — Latent GPU hang if anyone raises `-maxbits`.** The device path still uses binary-Euclid `pl_invmod` for odd moduli. An odd prime power q = p^k above bkthresh (which regenerating the FB with `maxbits > 15` would create) hits the gcd > 1 case and **loops forever on the device** — the exact Finding-15 failure, one path over. Until the device transform is power-safe, enforce the assumption: assert in the loaders that every bucketed modulus is prime (or 2^k), so the failure is a message rather than a hang.

> **Done, 2026-08-02** — all three faults removed by routing `k_transform`
> through `pl_transform_enc`, which decodes the encoding and is power-safe.
> R3's "assert that every bucketed modulus is prime" became
> `fb_check_prime_powers`, which enforces the transform's actual precondition
> (prime *power*, not prime) over any loaded factor base. RESULTS.md finding 21.

**R4 — Per-region checksums would close the verification gap R1 exposed.** Global record count + one-region cell replay leaves room for "right total, wrong region" errors. Extend `verify_count_updates` to also accumulate per-region (count, XOR/sum of offsets) and compare all 32K regions against the GPU cursors/records. O(same) host time, and it institutionalizes the transpose lesson: gate against *placement*, not just volume.

**R5 — Two-level was flattered, which makes Finding 1 airtight.** `k_fill_l2` writes offset-only records — the slice hint is dropped (`bench_kernels.cu:460`), which is why apply requires single-level records. A real two-level would carry the hint (8 B L1 records, or slice-segmented L1), i.e. *more* traffic than the 37.9 ms measured. Worth one line in RESULTS so nobody reopens the question with "but maybe two-level plus hints…" — it lost from a position of advantage.

## Factor-base reconciliation vs las — all three gaps now explained

For Finding 14's table (counts las vs ours):

> **Confirmed and fixed, 2026-08-02.** The −3 was exactly 59², 101², 127²;
> restoring the projective ladder closes it (RESULTS.md finding 20). The
> "−1,114" and "−1" remain open and are `powlim` differences, as diagnosed.

- **Side-0 small, −3:** the projective *power* ideals 59², 101², 127² (the three factors of Y1 whose squares stay ≤ 2^15). `rfb.c:159` breaks out of the power loop when `p | Y1` ("no lift"), but projective roots do lift — makefb emits them; we don't.
- **Side-0 bucketed, −1,114:** las builds side 0 on the fly with `powlim = ULONG_MAX` (PARITY.md) → 1,112 regular prime-power ideals in [2^15, 67.1M] (I counted them: exactly 1,112) plus the projective squares 281², 1259². Our `rfb` caps powers at `maxbits = 15`.
- **Side-1 bucketed, −1:** still open, but now suspicious by elimination — audit the q = 32768 boundary handling and any `r ≥ q` encodings in `c183.fb1` first. Diff the ideal multisets; it's one line of accounting.

Recommendation: **don't chase las upward — pin las downward.** For parity runs, pass `-powlim0 32767 -powlim1 32767` so both sides use exactly our ideal set (production las keeps its defaults; a parity config differing from the baseline config is normal and should just be recorded). Chasing las's powlim=∞ set instead would require making the device transform power-safe first (R3) for ~0.4% of side-0 updates that are irrelevant to the verdict.

Also reconcile the two side-0 las counts quoted in this repo — PARITY.md says 3,954,896, Finding 14 says 3,958,485. Probably two different counters (primes vs ideals-with-powers); whichever is right, one of the documents is quoting the wrong one.

## Measurement notes

- **Fill is probably near its structural ceiling — stop expecting DRAM% to move.** Single-level fill does ~0.9 L2 write sectors *plus* one L2 atomic per record ≈ 600M L2 operations in ~12 ms. That is the same L2-transaction wall msieve-s's scatter sat at (77% of ceiling, immune to hints and aggregation, and warp-aggregation is a closed lever at 16K-way from that work). The 17–18%-of-DRAM-peak figure is a red herring for this kernel — its budget is L2 ops/record, and 4 B single-level already spends close to the minimum. Corroborates "stop polishing."
- **Unbilled per-q host work.** The small-FB transform + insertion sort by m (`bench_kernels.cu:577–602`) runs per special-q per side on the host, outside every timed number — likely ~1 ms-scale, and the insertion sort is O(n²)-risky if many entries have g > 1. Measure it once, switch to qsort or direct tier-bucketing, and either bill it in the chain or note its exclusion. Path 5 should move it on-GPU (it is `k_transform`-shaped work).
- **Timing hygiene:** reps=3 arithmetic mean with no warmup discard. Add one warmup rep and report median-of-5; the numbers will barely move, but the next 3%-level claim ("2^13 gives back nothing") deserves it.
- **Stale text:** the "Caveats on the numbers above" block in RESULTS still says "One side only… rho is still synthetic," which the section directly above it (real q=120000053/rho, both sides) contradicts. And Finding 11's closing "fill is still 24.5 of the 47.1" predates powers — at 55.5 ms, **apply (27.45) has overtaken fill (24.77) and the small-prime sieve is the largest single component of the chain.** Both worth a one-line edit so the doc doesn't steer the next session at the wrong target.
- **RESULTS' Reproduce block predates the CADO-FB configuration** — add the exact command for the 55.5 ms number (`--cadofb … --side 0/1 --q 120000053 --rho 112625526 --scale 1.28/1.93`), since that's the number everything downstream will quote.

## Small-prime sieve levers, for when they matter (sized from the measurements)

Ranked, all sketch-only, none urgent while the TD floor binds:

1. **Concurrent entries in the block tier.** The 52 p<64 entries are processed one-at-a-time by the whole block; at p=61 that's 268 hits across 512 threads — half the block idle, and 52 serialized loop bodies. Two to four entries in flight (128–256 threads each) cuts the serialization roughly in proportion. The measured 9%-of-peak smem-atomic rate says the ceiling is far away.
2. **Fold power ladders into their base prime via valuation.** Powers are +2.48e9 updates (36% of the small sieve), and 2.07e9 of that is side 1 — this poly's lc (2^5·3^2·5·7·11) makes the tiny primes ramified-rich, so it's structural, not incidental. Instead of walking each rung separately, walk the base prime once and add `fb_log(p^min(v,K))` with v from a short division loop (ctz for p=2): ~1 + 1/p extra ops per base hit replaces ~Σ A/p^k scattered adds. Needs care with the multi-root rungs (this is why makefb's rungs exist), so prototype against `verify_apply_region` first.
3. **Pattern-sieve the tiniest primes into init.** CADO's sieve2357 analogue: for p < ~16, the per-row contribution is periodic in words; compose it during norm init as word-wide adds instead of per-hit atomics. Interacts with lever 2 (the same entries).

## dumpfile — one more round before abandoning the byte-diff

The evidence that the dump is rotten is good. But the project already patches CADO locally (the ASSERT removal), so the marginal cost of *owning* the dump is five lines: write S per region yourself in `las-process-bucket-region.cpp` right after `SminusS()`, with the region index in the record, and the strongest gate comes back. Cheaper probes first: (a) capture at default `-B 16` — the rot may be specific to the `-B 14` interaction nobody has exercised in years; (b) a `las` run with `lim1` tiny enough that nothing is bucketed — if that dump matches the small-only analytic value, the plumbing works and the loss is specifically in bucket application ordering, which pins the upstream bug report. `TRACE_K` remains the right tool for single positions either way. And for any future byte-diff: negate our first basis vector to match las's exactly (same lattice, i-mirrored), so diffs are positional without remapping.

## Housekeeping

- **Still no git repo.** Step 0 item 9 ("every number in this doc should be attributable to a revision") is now overdue by one full results cycle — `git init` and commit the current state before the next measurement session; the 55.5 ms configuration deserves a tag.

> **Done since** — repo exists, one commit, remote `kyleaskine/cuda-sieve`.
> Remaining: `bench/bench` and `bench/dumpcmp` are tracked binaries and should
> be `git rm --cached`'d (they are now in `.gitignore`). A second reviewer
> reported "858 MiB object store with the 512 MB dumps, a 110 MiB factor base
> and object files committed" — that is not this repo: it is 476 KiB across 31
> files, and `.gitignore` already excludes `*.dump`, `*.fb1`, `*.afb.0`, `*.o`.
> The two binaries are the only real instance.
- `oracle/MANIFEST.sha256` predates today's additions (fb1, dumps, PARITY.md) — regenerate.
- This doc has grown to ~1,000 lines with measurement blocks quoted inside design sections. It still reads well because the annotations are dated, but consider making `bench/RESULTS.md` the single source of numbers and keeping only verdict-level callouts here — future sessions will load faster and misquote less.

## What I'd write on the scoreboard

The central hypothesis has survived its first real contact: **the complete two-sided sieve runs at the CPU's trial-division floor (55.5 vs 56 ms), a 3–4× whole-box speedup with the cofactor path untouched, on measured kernels with independent correctness gates.** The remaining risk has moved decisively from "can the GPU sieve fast enough" to (a) the untested `N_eff`/CADO-side baselines and (b) whether Path 4 can raise the 4.2× ceiling. That is a much better place than this document started.
