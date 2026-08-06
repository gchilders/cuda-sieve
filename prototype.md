# GPU-Resident NFS Relation Collection — Prototyping Paths

## What we're actually testing

Not "build a GPU NFS." The primary question is narrower, but it is broader than
the bucket sieve alone:

> **Can NFS relation collection be made GPU-resident end to end, and what are
> its real relations/sec/watt and relations/sec/GPU?**

"GPU-resident" means that the sustained per-special-q data path does not need
CPU compute throughput proportional to GPU count: root transform, both sieve
sides, survivor intersection and compaction, resieve/factor recovery, trial
division, and cofactorization should run on the GPU wherever practical. A CPU
may orchestrate, perform I/O, and handle a genuinely rare fallback tail; a
high-end CPU busy on every q is not part of the primary target. This makes the
result relevant both to dense nodes such as 8-GPU H200 systems and to machines
with a useful GPU behind a weak or old host CPU.

The first experiment was the bucket sieve because it is the largest regular
stage and the cheapest feasibility gate. It is now measured; that does not
turn the sieve-only benchmark into the final architecture.

A **hybrid GPU-sieve + CPU-cofactor pipeline is a secondary deployment option**.
It may be excellent on a balanced machine such as the 9800X3D + RTX 5070, and
it is a useful near-term cross-check, but it must be reported separately. Its
CPU/GPU ratio and power result do not set the roadmap or the ceiling for the
primary GPU-resident design.

If a well-structured GPU-resident version trails by an order of magnitude with
the main levers exhausted, the idea is dead — and we want to find that out
cheaply, not after months. **Between those outcomes there is a wide gray zone,
and landing in it is not failure** (see "The verdict" below).

Everything below is scoped to **relation collection only**. Poly-select,
filtering, linear algebra, and sqrt come from CADO unchanged.

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
- **Reuse algorithms and implementations, not a CPU runtime requirement.** CADO
  and GGNFS are the oracle and baseline. YAFU's existing CUDA/OpenCL NFS
  cofactorization, its 64/96-bit GPU ECM kernels, lasieve5 batch factoring, and
  CADO's batch tools are references and candidate components for Path 4. A CPU
  subprocess may bootstrap correctness, but it is not the primary endpoint.
- **Host-independence is an architectural metric.** Report host cores and host
  watts per GPU, and test whether throughput scales with GPU count without a
  matching increase in CPU compute. A result that requires roughly one strong
  CPU per GPU is the hybrid result, not the GPU-resident result.
- **Baseline ≠ oracle.** **GGNFS `gnfs-lasieve4I15e` is the performance baseline** (AVX-512 lattice-siever asm; ~20% faster than CADO on this owner's jobs). **CADO `las` is the correctness oracle** (`-dumpfile`, `-batch-print-survivors`, `las_tracek`, checksums — GGNFS has none of these). Grade speed against GGNFS; grade correctness against CADO. See "Which siever is the opponent".
- **Grade on *unique* relations**, not raw relations. The two sievers use different factor-base conventions and therefore have different duplicate rates; raw relations/sec is not comparable across them. See the same section.

### Open questions the ladder must answer

1. How much work reaches resieve, trial division, and hard cofactorization on
   this job, and what throughput each GPU stage must sustain (Gate 0).
2. Best achievable **updates/sec/joule** across kernel-structure × record-size, vs. CPU `las` (Gate 1).
3. **Root-transform cost per special-q on GPU** — a second per-q cost pillar the original plan omitted; see its section below.
4. Single- vs. two-level crossover, and where the GPU's `bkthresh` optimum sits (above CADO's).
5. Can GPU-resident resieve/factor recovery sustain the compacted-survivor
   rate, and does the preliminary YAFU C164 cofactor result survive a
   persistent, target-equivalent C183 run?
6. How does the completed path scale across strong-host, weak-host, and
   multi-GPU systems, and what is the real rental $/relation at Path 5?

---

## The verdict metric, and the actual bar

The headline metric is **relations/sec/watt**, not wall-clock and not raw throughput. A GPU that's 2× faster at 4× the power loses.

The Windows-side power run on 2026-08-03 replaces the planning estimates with
a same-session component proxy:

| configuration | measured load power | full component proxy |
|---|---:|---:|
| 16-worker GGNFS | 125.1 W CPU PPT + 5.2 W DIMMs | **158.7 W**, including the idle RTX 5070 |
| two-side GPU sieve | 206.7 W stage-weighted GPU board average | **262.1 W**, including host CPU PPT and DIMMs |

The proxy is `CPU PPT + GPU board + both DIMM PMICs`; same-session idle is
68.3 W. Full method and stage weighting are in `bench/RESULTS.md` finding 44.
GPU perf-per-watt relative to CPU is still `(P_cpu/P_gpu) × speedup`, where
*speedup* is measured against the **whole 16-thread box**, not one core. The
measured **sieve-only** proxy ratio is `158.7 / 262.1 = 0.606`:

| Goal | required speedup at measured component power |
|---|---:|
| Within 3× on perf/watt | **0.55×** |
| Within 2× on perf/watt | **0.83×** |
| Actually beat CPU on perf/watt | **1.65×** |

This is a sieve-stage ratio, not a frozen Path-5 boundary. The primary
GPU-resident path must add the power and time of its GPU post-sieve stages plus
the host power it actually consumes. A future hybrid path must instead add the
power of sustained CPU cofactor workers and report its CPU/GPU pairing
explicitly.

For the measured portions, the complete CPU q costs 44.61 proxy J and the
two-side GPU sieve costs 16.87 proxy J (37.8% as much, a 2.64× advantage), or
25.41 vs 12.48 J above idle (49.1% as much, a 2.04× advantage). **That is not
the Path-5 verdict:** the CPU number is a complete GGNFS q, while the GPU number
still omits intersection, primitive filtering, transfer, resieve/factor
recovery and the cofactor feed.

The proxy also excludes motherboard/chipset, storage, fans, VRM losses and PSU
conversion losses. Keep a modest band around the derived zone boundary and do
not let a gray-zone verdict turn on the third digit. A wall-plug meter remains
the only instrument that collapses the final economics uncertainty.

### The zones — how to read a result

- **Green — ≥ ~0.55× the full CPU box on the measured component proxy.** Clears the 3×-perf/watt bar. Carry a broader ~0.5–0.7 band until whole-box wall power exists.
- **Gray — roughly 0.1× up to the green band.** Not a fail. Decide on evidence, specifically:
  - *Remaining levers:* if we're at 0.3× with 4 B records, single-level fan-out, or unbatched cofactorization still on the table, keep pulling. If every lever in this doc is exhausted, what's left is the economics call.
  - *Rental economics:* the real metric for rented boxes is **$/relation**. For
    the primary result, measure relations produced by the GPU-resident path and
    charge the instance's actual host use. Separately, a hybrid deployment may
    add relations from the bundled CPU while it services the GPU. Compare both
    against `R_cpu_box / $·hr_cpu_instance` using written-down marketplace
    prices (vast.ai / RunPod) at Path-5 time.
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

The scaling sweep now measures `N_effective = 10.24` at 16 workers, so the
production-equivalent whole-box CPU cost at q=130M is **~0.295 s per
special-q**. Against the measured component-power ratio above:

| | GPU must deliver one special-q in |
|---|---|
| Green (≥0.55× box) | **≤ 0.54 s** |
| Beat CPU on perf/watt (≥1.65× box) | **≤ 0.179 s** |

Those are end-to-end pipeline budgets. A sieve-only timing cannot close them.

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
> complete two-sided sieve now sits **right at the then-used 56 ms retained-TD
> line for the optional hybrid**, and the small-prime sieve is half the chain.
>
> **Scope of this number, 2026-08-02.** It is a *sieve* measurement: transform,
> fill, apply, threshold, survivor list. The two-sided survivor intersection,
> GPU resieve/factor recovery/TD, GPU cofactorization, and final relation output
> do not exist.
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
> **Latest clean run, 2026-08-03:** **64.371 ms** for the complete two-sided
> equal-work sieve (38.177 algebraic + 26.194 rational), against a measured
> ~295 ms CPU-box q and ~70 ms retained CPU stage in the optional hybrid.
> Stage-weighted component
> energy is 16.874 J/q. The older 47/55.5 ms figures above predate the later
> correctness and norm work and are retained only as dated development history.
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

1. **`f ≈ 0.23`, not 0.5–0.6.** Trial division plus cofactorization is
   **0.21–0.24** of GGNFS wall time across the range; hard MPQS alone is
   **0.08–0.11**. Those fractions are a CPU Amdahl floor only for architectures
   that retain those stages on the CPU. For the primary GPU-resident design
   they instead size the work that still has to be ported and measured.
2. **The root-transform cost pillar is confirmed, independently and precisely.** "Sieve-Change" is exactly the per-q q-lattice root transform, and it is **12% of wall** — 373 ms/q over ~11.3M FB entries = ~33 ns/entry ≈ 85 cycles, which is the right cost for one modular inverse. The doc predicted this stage was real and co-equal with fill; the measurement agrees. A GPU doing ~8.6M modinvs in 1–5 ms wins this stage by roughly 100×.
3. **Norm initialization is nearly free** — ~18 ms/q, 0.6% of wall. Path 3 can implement it naively without a performance concern. It remains the main *parity* obstacle, but not a performance one.
4. **The algebraic factor base is truncated at q** (`Warning: lowering FB_bound to <q>`): 3.0M entries at q=50M rising to the full 7.6M at q≥170M. So updates/q, bucket memory, and transform cost all roughly **double across the job's q-range** — the sizing table's numbers are the *high* end. **Check whether CADO does the same truncation**; if it does not, the two sievers are not doing equal work per q and the baseline comparison needs adjusting.

#### What this does to the verdict

Box per-q at the measured `N_eff = 10.24` is ~295 ms. If a deployment leaves
all trial division on that CPU, its retained stage is `f × 295 ms` = **~70
ms/q**; if only MPQS remains, it is **~29 ms/q**. The latest clean two-side GPU
sieve is **64.371 ms/q**. These are useful capacity points, not unavoidable
floors for a GPU-resident pipeline.

| | per-q | implication |
|---|---:|---|
| CPU box | 295 ms | measured scaling sweep; the thing to beat |
| GPU sieve | 64.371 ms | measured, both sides, equal-work profile |
| preliminary external GPU cofactor stages | ~4.92 ms | C164 `31/31` stage-timer estimate at the target feed; not yet C183, persistent, powered or integrated |
| hybrid: all TD stays on CPU | ~70 ms | ideal overlap ≈ `max(64.4, 70)` ≈ 70 ms; roughly one strong CPU per GPU |
| GPU-heavy: only MPQS stays on CPU | ~29 ms | CPU capacity might cover about two GPUs; unbuilt and unmeasured |
| primary GPU-resident path | unknown | add measured GPU intersection, recovery/TD, and cofactor stages to 64.4 ms |
| hybrid-only Amdahl ceiling from `f` | — | `1/0.237` = **4.2×** if all TD remains on CPU |

For the measured portions, the component proxy gives a 2.64× total-energy
advantage (2.04× above idle). It is premature to turn that into pipeline
perf/watt: intersection, primitive filtering, transfer, resieve/factor
recovery and the target-equivalent cofactor feed have no integrated time or
power measurement yet.

**This separates two conclusions that were previously conflated.** For the
hybrid option, the measured sieve and retained CPU stage are balanced within
about 10%, so the 9800X3D + RTX 5070 may be an unusually good pairing. For the
primary design, there is no measured end-to-end critical path yet: Path 1 is
ready to feed Path 4, and Path 4 is now the central experiment. Pause sieve
micro-tuning until the missing GPU stages reveal the real bottleneck; do not
declare the CPU floor to be the project's floor.

Caveats: this is **GGNFS lasieve4, not CADO las**, and CADO's cofactorization
strategy differs (ECM chains, and `-batch` if enabled). Gate 0 on CADO is still
required and may land elsewhere. `N_eff`, the sieve timing and the component
power proxy are now measured; end-to-end throughput and whole-box wall power
are not.

**Concrete Path-4 throughput targets, now derivable:** the sieve intersection
contains about 0.8M primitive candidates/q, or roughly **12M candidates/s** at
the measured 15.5 q/s sieve pace. GGNFS's recovery/TD funnel leaves about 1,900
hard-cofactor candidates/q, or roughly **30,000/s**. Those are the two GPU
batch rates to test. CPU capacity remains useful as a hybrid control, not as
the acceptance criterion.

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

## Post-sieve work is co-equal, not a footnote

The original ladder under-weighted resieve, factor recovery, trial division,
and cofactorization. GGNFS measures them at 21–24% of CPU wall time, including
8–11% in hard MPQS. That creates a 4.2× Amdahl ceiling for a **hybrid that
leaves all of TD on the CPU**. It does not create an irreducible floor for the
primary project: it identifies the next GPU workload.

This distinction matters across machines. The measured 9800X3D and RTX 5070
are almost balanced for the hybrid split, so that deployment is worth keeping.
A dense multi-GPU node or an old CPU paired with a modern GPU is not: requiring
CPU TD throughput for every GPU would strand devices or require proportional
host expansion.

There is already substantial implementation evidence to build from:

- **YAFU has an NFS-facing CUDA/OpenCL cofactorization path**, including
  batched 64- and 96-bit ECM kernels in `factor/gpu_cofactorization.c` and a
  call site in `factor/nfs/nfs_sieving.c`. It is a real starting point, not a
  proof for this job; current code explicitly rejects some layouts, including
  3LP cofactors on both sides.
- **lasieve5** (`~/yafu/factor/lasieve5_64/batch_factor.c`) is an independent
  reference for the production 3LP batch-cofactor architecture.
- **CADO `las -batch`** and `precompbatch` / `finishbatch` / `finishecm` expose
  a product-tree implementation and a CPU reference result. Product/remainder
  tree work is itself a candidate for GPU batching.
- **`las -batch-print-survivors <basename>`** dumps the post-sieve population
  and **`las -stats-cofact`** records the funnel, making CADO an oracle for each
  GPU stage without making it the production runtime.

There is now one preliminary rate measurement. An owner-supplied February log
from Ben Buhrow's standalone `nfs_3lp_batch_factor` suite processed one million
C164 `lpbr=lpba=31` inputs on the RTX 5070. Its three printed 64/96-bit stages
sum to **2.5901 s, or 386,084 inputs/s**—13.1x the current 29,500/s Path-4
feed estimate and, if the workload transferred directly, about **4.92 ms/q**.
The cold outer invocation was 11.9524 s; the CPU batch-GCD path on the 9800X3D
was 22.7166 s. GPU yield was 10,246 complete relations versus 10,257 on the
CPU (99.893%). Full provenance, arithmetic, and harness-audit caveats are in
`bench/RESULTS.md`, finding 47.

Treat this as a **positive fail-fast signal, not a closed stage**. The source
data are C164 `31/31`, not the target C183 `31/32`; the GPU path still performs
recurring preparation, validation, compaction, and possible fallback work on
the CPU; and 2.5901 s is the sum of printed stage timers rather than a
persistent end-to-end wall time. If a controlled target run holds near
4.92 ms/q, hard cofactorization leaves about 16–20 ms/q for intersection plus
resieve/recovery/TD inside the aggressive 2x-energy budget. That would move
the largest performance risk to recovery/TD rather than close Path 4.

Plan for every post-sieve stage as a first-class component with a throughput
and energy target. CPU implementations remain valuable for parity and as an
optional hybrid/fallback mode, but the primary acceptance test is that GPU
throughput does not require host compute to scale one-for-one with GPU count.

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

   Why this matters for the CPU reference: **CADO's batch cofactorization is GMP product trees**. (Non-batch `las` is largely unaffected — 92-bit cofactors go through CADO's own `modredc`, not GMP.) A slow GMP inflates the optional hybrid's `f` and distorts the CPU oracle timing; it does not set the GPU-resident architecture's ceiling.

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

- **CPU power: measured 2026-08-03 with HWiNFO64 on Windows.** RAPL remains unavailable inside the guest. The 16-worker plateau was 125.1 W CPU PPT + 5.2 W across the two DIMMs; including the idle GPU gives a 158.7 W component proxy. See `bench/RESULTS.md` finding 44.
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
2. **Path 1.** Load `oracle/input.job.afb.0` directly (format decoded, already SoA), walk real `(p,r)` progressions, implement variants (a)–(d) and (T), sweep record size. Grade in **ms/q** against the measured ~225 ms / ~70 ms lines above.
3. Path 2 (apply kernel, shared-memory region, bank-conflict rate) follows immediately — same inputs, same harness.

**Waits for a quiet box:**

4. CADO Gate 0 — confirm `f ≈ 0.23` on `las`, and check whether it truncates the factor base at `q` (GGNFS does; CADO reportedly does not).
5. `makefb` both sides for `oracle/c183.poly`; `-fbc` cache. *(Only needed for CADO parity work in Path 3, not for Paths 1–2.)*
6. Second `las` build at `SIZEOF_P_R_VALUES=4`; A/B against the Zen3 GMP. *(Baseline-fairness work — and note the baseline is now GGNFS, so this only matters for the CADO oracle's own speed, which demotes it.)*
7. ~~`N_eff` configuration sweep for **GGNFS**~~ — **done 2026-08-03:** 16 workers win, `N_eff = 10.24`; `bench/RESULTS.md` finding 43.

Item 7 was the highest-value CPU measurement because `N_eff` is the divisor in
every target number on this page. It is no longer an assumption.

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

**Current ordering:** Paths 1–3 have established the GPU sieve's timing and
most of its correctness basis. Close exact survivor-set containment, then make
Path 4 GPU-resident stage by stage: device intersection/compaction, primitive
filtering, resieve/factor recovery/TD, and GPU cofactorization. The preliminary
C164 cofactor microbenchmark clears the raw rate gate by 13.1x, but its
target-equivalent persistent rerun remains an independent checkpoint when the
box is idle. Path 5 measures the resulting relation producer and its host
independence. CADO Gate 0 remains useful workload characterization, but the
primary design no longer rests on a CPU cofactor fraction generalising from
GGNFS.

### Path 0 — Baseline, oracle, and post-sieve workload characterization

Prerequisite: **Step 0 above is closed.** In particular the baseline `las` build is the faster of the two `SIZEOF_P_R_VALUES` variants, the FB cache exists, and the box is idle.

This establishes the CPU opponent, correctness oracle, and the volume entering
each post-sieve stage. Its Amdahl fraction determines the ceiling of a hybrid
deployment; for the primary GPU-resident goal it sizes Path 4 rather than
deciding whether Path 4 exists.

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
2. **Capture the sieve-vs-cofactorization split** from the end-of-run timing report, `-T -T` (fine-grain timings, per-q at two `T`s), and `-stats-cofact`. This sizes the GPU post-sieve batches and separately gives the hybrid Amdahl ceiling; a large fraction increases Path 4's importance rather than killing the primary design.
3. **Extract the CPU-side Gate-1 comparator while you're here:** updates/sec/joule for `las`'s fill phase ≈ (6.1e8 updates/q) ÷ (fill seconds per q from `-T -T`) ÷ (package watts during the run). Approximate is fine; write down how it was computed.
4. **Sweep the CPU-side cost structure.** `las` exposes `bkthresh`, `bkthresh1`, `bkthresh2` (2- and 3-level bucket sieving), `-B` (log bucket region), `-Bi` (log fan-out per level), `bkmult` — all verified present in the local build. Sweeping these on the CPU teaches the fill/apply cost structure *for this exact job* in a day, and every knob has a direct GPU analogue. Do this before designing the GPU fan-out.
5. Dump the factor base — **both sides**, freshly generated for the C183 (Step 0 items 1–2); the staged `cado_roots1.gz` is for a different number and side 0 has never been materialized at all. Plus a known survivor set for one special-q via `-batch-print-survivors`. This is the correctness oracle.
6. Capture per-q `las -v` output: it prints `# Sieving <q-lattice basis>; I=...; J=...` and `# Checksums over sieve region: after all sieving: ...`. The basis and J are needed to replicate a q exactly; the checksums are a cheap correctness summary.
7. Grab a **byte-exact sieve-region dump** for one or two special-q via `-dumpfile <stem>` (writes `<stem>.<side>.sq<q>.rho<r>.side<N>.dump`). This is a *stronger and more debuggable* oracle than checksums — you can diff to the first divergent byte.

Cross-check against `gnfs-lasieve4I15e` using the existing `test_sieve.sh`
tooling, since that is the CPU pipeline already trusted. Use CADO as the primary
oracle because it is better instrumented; that does not make it the production
cofactor runtime.

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

At the time this first pass was compared against the old **182 ms / 56 ms**
guideposts. Those are superseded by the measured **~225 ms / ~70 ms** lines;
apply, small-prime sieve, norm init and threshold were added later.

Two design decisions were overturned (2 B records, two-level fan-out) and two
predictions confirmed (run-aggregation worth **2.3×**; the transform is a
non-issue on GPU at **1.78 ms vs the CPU's 373 ms — a ~200× win**, the largest
single-stage gain measured). See the Decisions-locked list and the fan-out
section for the corrections.

#### The number Path 1 must produce

Not a bandwidth percentage. **Milliseconds per special-q for the full GPU sieve chain** (transform + fill + apply + small-sieve + threshold), against two lines derived from the GGNFS breakdown at the measured `N_eff = 10.24`:

| | ms/q | meaning |
|---|---:|---|
| GPU-replaceable CPU work (Sieve + medsched + Sieve-Change) | **~225 ms** | match this and the GPU merely ties the box on the sieve |
| hybrid retained CPU stage (TD stays on host) | **~70 ms** | constrains the optional hybrid, not the GPU-resident design |
| measured GPU sieve | **64.371 ms** | the two stages are well matched for the hybrid option |

The sieve component target is met. The primary question has moved to device
intersection/compaction, primitive filtering, resieve/factor recovery, GPU
cofactorization, and final relation output; bare sieve-kernel speed no longer
decides the result.

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
> fill 12.30 + apply 3.08) against the then-current 182/56 ms guideposts
> (superseded by ~225/~70 ms in finding 43).
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

### Path 3 — Single special-q, complete sieve stage

Wire fill + apply on the real dumped poly/FB. Sieve one special-q and threshold
to survivors. This is end-to-end for the **sieve stage**, not for relation
collection.

Be explicit about what a complete sieve stage for one q requires beyond Paths
1–2—this is where the hidden scope originally lived:

1. **q-lattice basis reduction** per special-q (host, microseconds; cross-check against the basis `las -v` prints). **Reduce under the *skewed* norm** `|(a,b)|² = (a/√s)² + (b√s)²`, not the plain one — measured 2026-08-01, this is not a refinement, it is required for the norms to be computable at all. An unskewed reduction gives |a| ~ |b| ~ √q, the homogeneous terms `c_k a^k b^(d−k)` then span **10³⁹**, `log2|F|` is set by `c₀b⁵` alone and the leading term underflows fp32 outright. Skewed: A = 2.33e12, B = 1.64e4, normalised coefficients all O(1) (`-0.0136, -0.0605, 1, -0.419, -0.0054, 0.381`). That balance is what the skew parameter is *for*, and it is the only thing that makes fp32 norm evaluation possible.
2. **Norm initialization** — the sieve array starts at the log-norm bound per position, not zero. A simple float evaluation of `log|F(i,j)|` per position is fine to start, **but exact parity with las requires reproducing its per-q log scale** (las rescales so norms fit a byte) and its piecewise approximation + rounding. Budget this; it's the main obstacle to byte-exact matching.
3. **Small-prime sieve** for `p < bkthresh` (~3.5K primes + prime powers per side) — a per-region shared-memory walk; a third kernel family, small but mandatory for any survivor parity. **BUILT AND MEASURED 2026-08-01 — and it is not small.** 7,143 entries across both sides, 0.1% of the factor base, producing **4.36e9 updates: 7.2× the entire bucket-sieve volume**, at **13.2 ms** against the 1–3 ms budgeted. It is 28% of the sieve chain. 84% of side 1's updates come from the 52 entries with p < 64, so the engineering problem is load balance, not throughput: three tiers (whole block / one warp / one thread per entry, cut at p=64 and p=1024) against a thread-per-prime loop that would hand 8,192 serial updates to whoever drew p=2. Keeping `log_region ≤ logI` puts each region inside one j-row, which reduces the per-region entry point to one multiply and one remainder and makes every block independent — CADO carries per-prime positions forward between regions because it visits them in sequence; we can't, and don't need to. Fused into the apply kernel, so the region is touched once. (The cheap-dodge suggestion is moot.)
4. Bucket fill + apply (Paths 1–2) + threshold test → survivor `(i,j)` list.

**Correctness gates, in pragmatic order:**
1. ~~Fill+apply parity against our own `verify_cpu.c` reference on the real FB (no las semantics needed — catches kernel bugs first).~~ **DONE 2026-08-01**, both sides, full I15e: records landed equal the CPU reference exactly and a full region replayed on the CPU gives 0 cells differing. Note the limit this gate structurally has: it compares us against ourselves, so it cannot see a wrong *lattice*. It didn't — `pl_transform` was transposing the basis and every self-check still passed. Gate the transform against its definition (`a ≡ r*b mod p`) separately; `verify_transform()` now does.
2. **PASSED 2026-08-02, by a different instrument.** The dumpfile route stays dead — las's dump carries ~41 log units per position, matching the *small-sieve-only* expectation (39.31) rather than the whole factor base (54.65), and correlates with our region at **+0.03** in either i-orientation against a **+0.64** control; it sits behind an `ASSERT_ALWAYS` that made its only `open()` call unreachable, so it has rotted. **Gate 5 replaced it outright and is strictly better:** `las_tracek` is a *stock* CADO target needing no patch, and it prints every ideal applied to a position with its log, rather than one byte. Result: **8 of 8 exact log sums**, four positions × two sides, with the *ideal lists* agreeing entry by entry — two of the positions chosen because long projective power ladders hit them. The comparison is made twice: once against a direct enumeration of the factor base (which gates roots and logs, and is what found the bugs), and once against the value read back out of `k_apply` via `bench --probe`, which has been through the transform, the walk, the tiering, the fill, the small sieve and the GPU apply. **The second is the one that closes the gate**; the first alone is ideal/log agreement, and describing it as sieve parity was an overclaim held for about an hour. Full detail in `oracle/PARITY.md` and RESULTS.md finding 26.

   Getting there cost two real fixes that nothing else would have found: las's printed `scale` is rounded to 2 dp and the exact values are **1.275 / 1.925** (enough to move `fb_log` by one for a band of primes), and the survivor bound is a *truncating* cast plus a guard bit, not a round. Norm initialisation still differs by 0..+2 — that is las's own approximation plus `LOGNORM_GUARD_BITS`, and **our fp32 evaluation is the more accurate of the two**, which is also why a byte-for-byte region diff could never have reached zero even with a working dumpfile.
3. Compare against the `-batch-print-survivors` survivor set on canonical
   `(a,b)` pairs. Require **`CADO \ GPU = 0`** at the locked profile; report
   `GPU \ CADO` separately as false-survivor pressure rather than forcing equal
   counts across different norm approximations.
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

### Path 4 — GPU-resident post-sieve and cofactorization — the primary next path

Build this as measured stages, keeping candidates on device between them:

1. Keep both survivor bitmaps resident, AND them, compact the intersection,
   and apply the full primitive `gcd(i,j) == 1` filter. First close exact CADO
   containment on canonical `(a,b)` pairs; then keep the production path off
   disk and off the host.
2. Recover factor-base primes via **GPU resieve** over compact survivors, then
   recompute norms and perform the regular factor-recovery/trial-division
   checks. This bulk stage is part of the primary GPU deliverable even though
   GGNFS accounts for it inside its CPU TD timer. **Design settled 2026-08-03**
   — re-walk the factor base against a 1 MB hierarchical survivor filter, side 0
   first, direct per-survivor testing for `p < bkthresh`, and no change to the
   4 B record. See the review notes at the end of this document.
3. Batch the remaining large cofactors into the existing YAFU GPU
   cofactorization design. Start from its NFS `relation_batch_t` interface and
   CUDA 64/96-bit ECM kernels; use its unsupported cases as an explicit test
   matrix, not as assumptions. Finding 47's preliminary C164 run is a positive
   rate gate, not the integration measurement. Before the controlled rerun,
   zero all batch accounting, fix the CUDA-version/context setup, select
   native sm_120 PTX, and retain the context, module, and buffers. Then sweep
   batch size and queue depth on the actual C183 `31/32` population, compare
   every batch with CADO/lasieve5 CPU results, and measure recurring host work
   and CPU/GPU power. One million inputs are about 34 seconds of arrivals at
   the projected feed, so a million-record-only result is insufficient.
   **Assume a device-resident cofactor queue spanning many special-q from the
   start**: ~1,851 cofactors/q fills 2.5% of this device's resident threads, so
   per-q lockstep cannot work, and batching across q rather than running curves
   in parallel is what preserves the early-exit stopping rule. This is the one
   hard-to-reverse decision in the post-sieve path. This stage can be developed
   entirely against a `-batch-print-survivors` capture, before stages 1–2 exist.
4. Validate, deduplicate, and emit complete relations. Returning compact final
   relations to the host is expected; returning every sieve survivor for
   sustained CPU processing is the hybrid architecture.

Each stage needs throughput, joules, candidate-in/candidate-out counts, and a
correctness oracle. A temporary CPU fallback is acceptable while bringing up a
stage, but record its frequency and cost. It is compatible with the primary
goal only if it remains rare enough that host demand does not scale materially
with GPU count.

### Path 5 — GPU-resident throughput mode — the real number

Batch many special-q and overlap transform, fill, apply, survivor processing,
resieve/TD, and cofactorization with CUDA streams and graph capture. Slab the
sieve area if running I16e. Keep the steady-state candidate path on the GPU and
transfer final relations plus telemetry.

Measure aggregate unique relations/sec/watt, relations/sec/GPU, required host
cores/GPU, and scaling from one to multiple GPUs where hardware is available.
Include a weak-host test or CPU-throttled proxy so a good result cannot depend
silently on the 9800X3D. Compute $/relation with written-down rental prices.
This is the primary verdict, graded on the zones.

After that result exists, measure the **hybrid deployment option** separately:
GPU sieve with GGNFS/CADO CPU factor recovery and cofactoring, overlapped across
q. Finding 45 projects that this could suit the 9800X3D + RTX 5070; it is a
valuable product mode, not the definition of Path 5.

---

## Instrument from day one

- **relations/sec/watt** (verdict) — log CPU PPT, GPU board power and both DIMM PMICs together; use whole-box wall power for the final economics. Sample continuously, not at endpoints.
- **updates/sec/joule** — the Path-1 proxy for the verdict. Report this instead of bare bandwidth percentages.
- **transactions per update** = `lts__t_sectors.sum ÷ n_updates` — the lever that actually moves the answer, and directly comparable to msieve-s's results. **Bytes per update must be derived, not measured**: absolute `dram__bytes_*` counters return `n/a` under WSL2 (see Step 0).
- Achieved memory bandwidth as % of peak — `dram__throughput.avg.pct_of_peak_sustained_elapsed` works. A diagnostic for "is this kernel well written," *not* a gate.
- Atomic/lock stall %, warp divergence, occupancy, shared-mem bank conflicts.
- **Max bucket occupancy, not mean.** See "slowest-block-bound" below.

### Power measurement on this box (WSL2 — verified, this bites)

- **GPU: works.** `nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=100` (WSL passthrough at `/usr/lib/wsl/lib/nvidia-smi`) or pynvml in the harness.
- **CPU: RAPL is absent under WSL2.** `/sys/class/powercap` doesn't exist here; `turbostat`/`perf` energy counters need MSR access the guest doesn't have. Options, best first:
  1. **HWiNFO64 on the Windows host — done 2026-08-03.** Log CPU PPT, NVIDIA GPU Power and each DIMM's Total Power to one CSV and align by timestamp. The measured proxy is 158.7 W for 16-worker GGNFS and 262.1 W stage-weighted for the two-side GPU sieve; finding 44.
  2. **Wall-plug meter** for whole-box draw — arguably the *honest* number for $/relation anyway (it includes DRAM/VRM/fans). Report total draw, not delta-over-idle, for the economics; note idle separately.
  3. Until then, carry the unmeasured motherboard/VRM/storage/fan/PSU contribution as explicit uncertainty. Acceptable for early gates, not for the Path-5 verdict.
- For the primary verdict, charge the GPU-resident pipeline for every host core
  it actually uses and report host cores/GPU alongside whole-box wall power.
  Repeat with reduced host capacity or multiple GPUs to expose hidden CPU
  scaling. For the optional hybrid result, charge all sustained cofactor
  threads explicitly. The CPU baseline always pays for its whole package.

---

## Known traps

**Cache-residency illusion.** Benchmark at a size where the FB does not fit in L2, or you'll measure the wrong regime and get a falsely rosy result. Single easiest way to fool yourself.

**Uniform-random benchmark inputs.** Fill kernels fed random records instead of real `(p, r)` walks misrepresent run-aggregation, slice volume skew, and locality — in either direction. Walk real progressions (Path 1 note above).

**Slowest-block-bound, not average-bound.** msieve-s measured this twice (experiments #9, #10): improving *average* per-bucket convergence saved literally zero wall time because the long tail dictated kernel duration. If bucket fill is uneven, the apply kernel's duration is set by the fattest bucket. Instrument the max. Sieve positions are near-uniform so this should be mild here — but verify rather than assume, and don't celebrate a mean-case improvement until it shows up in wall time.

**Fusion can lose to register pressure.** msieve-s experiment #11 fused a scatter into its producer kernel: registers went 56 → 72, block_limit dropped 5 → 4, per-launch duration *doubled*, net +25% GPU time despite eliminating an entire kernel. The analogous temptation here is fusing the root transform with the bucket scatter. Check the register count before believing in it.

**Memory capacity is about buckets, not the FB.** The FB is 61 MB. Bucket storage at I16e is 8–16 GB and does not fit. Slab, or stay at I15e.

**Resieve, don't re-divide.** Trial-dividing survivors by the whole FB throws the win away.

**Scope boundary.** Keep poly-select, filtering, linear algebra, and sqrt from
CADO and use them as-is. The GPU project owns **relation collection**, including
the post-sieve factor-recovery and cofactor stages—not merely bucket sieving.

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

**Gate 0 (Path 0) — post-sieve workload. Partially answered already, on
GGNFS: `f ≈ 0.23`.** The fine-grain timings put trial division plus
cofactorization at 21–24% of wall across the q-range, with hard MPQS at 8–11%.
That gives a hybrid retaining all TD on the CPU an Amdahl ceiling of ~4.2×. For
the primary design, Gate 0 records candidate volumes, sizes GPU stages, and
identifies the algorithms that must move; it is not a reason to leave them on
the CPU or to kill the project.

Confirming the split on **CADO** remains useful because its explicit resieve,
ECM chains, batch mode, and factor-base policy differ. A discrepancy changes
the oracle workload and the optional hybrid projection, but GGNFS remains the
performance baseline and the GPU-resident roadmap remains stage-driven.

**Read Gate 0 as a property of the parameter regime, not of NFS.** The C183's
`mfba=92`, `lpba=32` three-large-prime side is deliberately demanding. Also
measure a 2LP point (`mfb1≈64`): the comparison tells us where GPU
cofactorization has the most value and whether one implementation strategy can
cover both regimes.

**Gate 1 (Paths 1–3) — updates/sec/joule and complete GPU sieve time. PASSED as
a component gate.** The measured 64.371 ms/q is within the planned range. Freeze
the current operating point and spend the next effort on Path 4. This is a
sequencing decision: resume sieve optimization if the completed GPU-resident
pipeline shows it on the critical path. It is not based on treating a CPU TD
floor as irreducible.

The cheapest useful output of Path 1 is therefore a single number — **milliseconds per special-q for the full GPU sieve chain** — not a bandwidth percentage.

**Measured status, 2026-08-03:** 64.371 ms/q for the two-side equal-work sieve.
The stage-weighted component proxy is 262.1 W and 16.874 J/q, against 158.7 W,
3.558 q/s and 44.609 J/q for the simultaneous 16-worker GGNFS power run.
That closes the early component-energy proxy, not Gate 2: the GPU path still
stops before intersection, primitive filtering, GPU resieve/factor recovery,
integrated target-equivalent GPU cofactorization, final relation transfer, and
unique-relation accounting. Finding 47's external C164 microbenchmark is a
promising rate datum for the cofactor stage; it does not change that pipeline
status.

**Gate 2 (Path 5) — GPU-resident unique relations/sec/watt, relations/sec/GPU,
host cores/GPU, and $/relation, graded on the verdict zones above.** Green: go.
Gray: rental economics, hardware-shape sensitivity, and the best-case CPU
opponent decide in writing. Red: stop or narrow the useful GPU stages. Report
the optional hybrid mode beside this result, never in place of it.

**Not a kill criterion:** achieved bandwidth as a percentage of peak. If a variant lands at 30% but moves 2 B/record, it beats a variant at 60% moving 8 B. Low bandwidth efficiency means *go optimize*, not *stop the project*. Report it, act on it, don't gate on it.

---

## Language and repo layout

**Decision: C++17 host, CUDA C++ device, Python for the harness only.** (Toolchain on this box, verified: CUDA 13.2, target `sm_120`.)

The reasoning:

- Kernels are CUDA C++ regardless. That's ~95% of the runtime and most of the intellectual work. Host language buys nothing where the risk actually lives.
- In the primary runtime, the host is not in a per-update or per-survivor inner
  loop. It performs control, buffer/stream orchestration, and I/O. Temporary
  CPU parity and fallback paths must be timed so they do not silently become a
  proportional host requirement.
- **Through Path 3, nothing links against anything.** `las` is a subprocess
  oracle driven by the Python harness; GMP enters only for host-side setup.
  Path 4's first integration target is the local YAFU GPU cofactor interface,
  with CADO/lasieve5 subprocesses as result oracles. Every reuse target (YAFU
  C/CUDA, CADO C++, GMP-ECM C, GMP C+asm, lasieve5 C) is C-ABI-adjacent; C++
  remains the low-friction host language.
- A sustained CPU cofactor queue is intentionally a **separate hybrid mode**.
  It still wants routine C++ queueing machinery, but its throughput, power, and
  CPU/GPU ratio must not be attributed to the GPU-resident mode.
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

> **Update 2026-08-03:** items 1 and 2 are complete. Sixteen GGNFS workers
> give `N_eff = 10.24`; the clean two-side sieve is 64.371 ms/q and the
> Windows-side component proxy gives 16.874 GPU-sieve J/q versus 44.609 J/q
> for the complete CPU q. The scope mismatch means this is not yet a Path-5
> relation-per-watt result. With the primary goal clarified as GPU-resident
> relation collection, item 4 means device intersection, recovery/TD, and GPU
> cofactorization—not merely feeding a permanent CPU cofactor pool. Exact
> survivor-set parity is the entry gate.

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

The central hypothesis has survived its first real contact: **the complete
two-sided sieve stage runs in 64.371 ms/q with encouraging component energy and
independent correctness gates.** Kernel feasibility is demonstrated. A
GPU-resident relation collector is not.

> **Primary scoreboard, 2026-08-03:** Goal 1 is a GPU-resident relation path
> whose sustained CPU demand does not grow roughly one-for-one with GPU count.
> Measured: root transform through two-sided thresholding, **64.371 ms/q** and
> **16.874 component-proxy J/q**. Unbuilt: exact device intersection and
> compaction, primitive filtering, GPU resieve/factor recovery/TD, GPU
> cofactor integration, relation emission, and multi-GPU/weak-host scaling.
> Preliminary external evidence: YAFU's C164 `31/31` GPU cofactor stages imply
> **4.92 ms/q at the target feed (13.1x rate headroom)**, but the C183 `31/32`
> persistent, powered, host-accounted run remains open. Exact CADO survivor
> containment is the next correctness gate. These missing stages, not the CPU
> TD timer, define the live roadmap.
>
> **Secondary hybrid projection:** if GGNFS keeps factor recovery, TD, and
> cofactoring on the 9800X3D while the RTX 5070 sieves the next q, the measured
> inputs project **71.1 ms/q, ~4× CPU-only throughput, and ~1.9× component
> energy efficiency** under perfect overlap. This is promising for this
> balanced pair, but it consumes approximately one strong CPU per GPU and is
> therefore not a substitute for Goal 1 on dense-GPU or weak-host systems.
>
> **Optimization rule:** freeze sieve micro-tuning while Path 4 is built. Resume
> it if the measured GPU-resident pipeline makes the sieve critical. The claim
> that further sieve work has "exactly zero" value applies only to the idealized
> hybrid max(64.4, 71.1) model, not to the project as a whole.

---

# Review notes — from Claude (the reviewer), 2026-08-03

Third pass, at the owner's request. Question put to me: *"next couple steps
should be exact survivor containment, then the design for GPU resieve and trial
division"*. Read: `bench/RESULTS.md` findings 38–46, `oracle/PARITY.md`, the
bucket record layout in `bench_kernels.cu`, and CADO's
`las-process-bucket-region.cpp` / `las.cpp` survivor-printing path. Analysis
only; no code changed.

> ## Read first — what a 2026-08-04 cross-review changed in this section
>
> An independent review of these notes found two substantive errors and several
> loose ends. Corrections are inline and dated below; this is the index.
>
> | claim as written | status |
> |---|---|
> | "GPU resieve design is settled: re-walk, no record change" | **wrong — reopened.** I mis-read CADO's shorthint. It is a *within-slice offset* that names the prime exactly, not a slice index, and CADO recovers large primes by **purging retained buckets**, not by resieving. Re-walk is now one of three candidates in an A/B that has to be run. |
> | "bound 131 — adopt unconditionally" | **right conclusion, wrong number.** The sweep's resolution was 4 units and never sampled the edge. The six missing bounds were run: **the zero-loss floor is 128**, and the containment gate was rerun on our own bitmaps at 128 and passes 3,026/3,026. **Adopt 128** — 2.46× on CADO's survivors, 2.35× on ours, zero relations lost. |
> | "`qsort` — a one-line change" | **not one line.** Four parallel arrays; needs an index-permutation sort plus a scatter. |
> | "1,062,811 records over 544 q, 1,957/q" | **548 q, 1,939/q.** Bit-size mix now measured; it is heavier than finding 47's C164 set and moves that projection ~12% slower. |
> | artifacts "Committed to `oracle/`" | **untracked**, and absent from `MANIFEST.sha256`. |
> | `PARITY.md`'s basis gate, `RESULTS.md`'s "set comparison not done" | both **stale**; corrected in place in those files. |
>
> Unchanged by the cross-review: the containment result (3,026/3,026), the
> survivor-filter occupancy table, the `enter_cofactoring` indicator, the
> 2× survivor reduction, and the shape of the budget.

## Short answer: yes, with one reordering and one addition

The proposed order is right, and finding 46 has made it the *only* order that
makes sense — the post-sieve budget is now a number, so every remaining
decision is a cost fit rather than a guess. Two changes:

1. **Containment and the device intersect/compact stage are one deliverable,
   not two.** The set you compare against las is the output of the stage you
   have to build anyway. Doing them separately means writing the compaction
   twice.
2. **Everything that runs on the CPU should run concurrently, and the
   containment oracle needs a CADO patch that `-batch-print-survivors` does not
   substitute for.** I proposed that flag as the oracle, ran it, and it emits
   the 1,851-element `enter_cofactoring` set rather than the 797,028 `after_sieve`
   set — see the next section for why no flag combination fixes that. The
   capture is still valuable, as the target-population cofactor workload finding
   47 needs. Neither it nor the survivor-bound sweep requires any of our GPU
   pipeline to exist.

Sizing below — written before finding 47 landed and revised after — says
resieve and TD clear energy parity comfortably, and that **they, not
cofactorisation, are what now sits on the boundary of a 2× energy win.**

## The containment gate — `-batch-print-survivors` will not do it (measured)

**Ran it; the recipe I first proposed is wrong.** Reading the source I concluded
that `-batch-print-survivors` takes the `else if (batch || batch_print_survivors)`
branch at `las-process-bucket-region.cpp:702`, which does no trial division and
no `check_leftover_norm`, and therefore emits the full `after_sieve` set. It
does not. On the parity q:

```
# survivors after_sieve: 797028
# survivors trial_divided_on_side[0]: 33355
# survivors trial_divided_on_side[1]: 1851
# survivors enter_cofactoring: 1851
```

and the file contains **1,851 lines, not 797,028**.

The branch is guarded. `las.cpp:914` asserts
`batch || batch_print_survivors || needs_resieving()` — a disjunction, not an
override — and `siever_config::needs_resieving()`
(`las-siever-config.hpp:116`) returns false **only if some side has `lim == 0`**.
With both sides sieved normally it is true, so las takes the *first* branch,
does full trial division, and `-batch-print-survivors` prints only what survives
to `enter_cofactoring`. The source comment above `needs_resieving` says this is
deliberate: with two real sides, removing known primes is worth doing even in
batch mode. There is no flag combination that yields the `after_sieve` set while
still sieving both sides.

**So the exact containment gate needs a CADO patch, and it is a small one.**
`rep.survivors.after_sieve++` is at `las-process-bucket-region.cpp:552`; the
`cofac_standalone cur` carrying `(a,b)` is constructed ~15 lines below it,
still before any trial division. Printing `cur.a, cur.b` there is about three
lines and yields exactly the 797,028. The survivor vector reaching that loop is
already primitive-filtered upstream by the unsieve step, which is why finding 41's
gcd filter closed the gap. This repo already carries a local CADO patch (the
`-dumpfile` ASSERT), so the precedent and the revert procedure both exist —
`oracle/PARITY.md` documents the pattern.

**The run was not wasted, and this is worth acting on.** The 1,851-line file is
kept as `oracle/c183.q120000053.cofac_candidates.txt` (log:
`oracle/las_q120000053_batchsurv.log`; both untracked, and `MANIFEST.sha256`
needs regenerating either way). It is
`a b <cofactor…>` on the **real C183 `lpb 31/32` population** — precisely the
target-equivalent cofactor workload finding 47 says its controlled rerun needs,
and which the C164 `31/31` dataset is not. It requires no patch and no GPU
pipeline. At 1,851/q, a `-nq` sweep of ~540 special-q produces the million-record
scale finding 47 benchmarked at; a few dozen q are enough to start the coverage
matrix. **Capture this while the containment patch is being written**, not after.

One consequence of the original recipe does survive:

- **Compare on `(a,b)`, not `(i,j)`.** las prints `(a,b)`, and more usefully,
  `(a,b)` is invariant under the i-mirroring this project already flagged
  ("negate our first basis vector to match las's"). If our basis vector is
  negated relative to las's, our `(i,j)` maps through *our own* basis to the
  same `(a,b)` las gets from `(-i,j)` through its basis. Comparing on `(a,b)`
  makes the gate independent of a convention that has already cost time once.

**The gate should be run at bound 143/141 — our natural setting — and it has a
falsifiable prediction.** Finding 41 established that las is `+1` and only `+1`
at every traced position. If that holds everywhere, then ours = las − 1
pointwise, the test `S ≤ bound` gives us everything las has plus exactly the
positions where las reads `bound+1`, and therefore:

> **|las \ ours| = 0, exactly.** Not "small" — zero. Any element of las's set
> that we do not have is a bug, not a calibration difference.

The reverse difference should be ≈44,000 (841,418 − 797,028), and it should be
*entirely* accounted for by positions whose las-side byte is exactly one over
bound on the binding side. That second check is what turns the gate from
descriptive into diagnostic: if the excess is not concentrated at `bound+1`,
the `+1` model is wrong somewhere the fifteen traced positions did not reach.

## The gate was built and run: strict containment fails, relation containment passes

The patch above was applied, `las` rebuilt, the capture taken, and CADO restored
byte-for-byte (source and binary md5s verified against pre-patch copies; the
only remaining local modification is the pre-existing `-dumpfile` one). The
patched binary is kept outside the CADO tree. The capture is
**797,028 unique `(a,b)` pairs, exactly matching `survivors after_sieve`**, and
the run still produced 37 relations, so the patch does not perturb the sieve.

Intersecting the existing `surv.side{0,1}.bits` bitmaps from the finding-40
session, gcd-filtering, mapping to `(a,b)` through our own basis, and comparing:

| quantity | value |
|---|---:|
| las `after_sieve` | 797,028 |
| ours (primitive, two-sided) | 841,418 |
| \|ours ∩ las\| | 794,866 |
| **\|las \ ours\|** | **2,162** — predicted 0 |
| \|ours \ las\| | 46,552 |

**The prediction was wrong and the `+1` model does not hold universally.**
0.27% of las's survivors are missing from ours. Attributing each missing element
to the side whose bitmap bit is clear:

| | count |
|---|---:|
| fails side 1 only | 722 |
| fails side 0 only | 1,422 |
| fails both | 1 |
| unmappable to an `(i,j)` in our region | 17 |

Chasing each cause, with everything re-measured rather than assumed:

**1. Pinning `powlim` removes more than half of it — and this document had
already prescribed that fix.** The 2026-08-01 review notes say las builds side 0
with `powlim = ULONG_MAX`, giving 1,112 prime-power ideals in [2^15, 67.1M] our
`maxbits = 15` factor base lacks, and then say in bold: *"don't chase las
upward — pin las downward … pass `-powlim0 32767 -powlim1 32767`."* I ran the
first capture without them. Re-running with them:

| | first capture | `-powlim0/1 32767` |
|---|---:|---:|
| las `after_sieve` | 797,028 | 795,845 |
| **\|las \ ours\|** | **2,162** | **1,005** |
| fails side 1 only | 722 | **722 (unchanged)** |
| fails side 0 only | 1,422 | **266** |
| fails both | 1 | 0 |
| unmappable | 17 | 17 |

las subtracts logs for ideals we never had, so those positions survive for it
and not for us — the observed direction and the observed side. Side 1 is
*exactly* unchanged because our `c183.fb1` was already built with
`makefb -maxbits 15`, so `-powlim1` is a no-op there. **Every parity capture
from here on must carry `-powlim0 32767 -powlim1 32767`.**

**2. The 17 unmappable are all at `b = 16384 = I/2` exactly — and this corrects
something I asserted two sections above.** Comparing on `(a,b)` removes the
*remapping* ambiguity from the i-mirroring, but it does **not** make the gate
mirror-independent: the two conventions sieve different half-open intervals.
Our `i ∈ [−I/2, I/2)` maps under the mirror to las's `i ∈ (−I/2, I/2]`, so one
boundary column of 16,384 positions is sieved by las and not by us. A convention
fix, not an arithmetic one, but it is a real gap and it recurs at every q.

**3. The obvious explanation for side 1's 722 is wrong, and disproving it
produced a better result.** `--fbbound` defaults to `q`, so the natural
hypothesis was that the finding-40 bitmaps used the truncated equal-work profile
and were missing the ~720K primes in (q, 134.2M]. Regenerating side 1 with
`--fbbound 134200000` produced a file **byte-identical** to the one already on
disk. So that bitmap was always full-`alim`, the truncation hypothesis is dead,
and side 1's 722 remains unexplained. The by-product is worth more than the
hypothesis was: **the GPU sieve reproduces a 64 MB survivor bitmap bit-exactly
across sessions and rebuilds.** That is a determinism guarantee this project had
not established, and it is what makes bitmap artifacts safe to compare across
sessions at all.

**A documentation defect fell out of the same check.** Popcounting the two
bitmaps gives one-sided survivor counts of **22,499,522 (side 1)** and
**24,301,359 (side 0)**. RESULTS.md finding 40's table reports 14,888,741 and
18,936,923 — while its *two-sided* row, 841,418, and finding 41's pre-gcd
1,386,939 both reproduce from these same two files **exactly**. The two-sided
numbers are trustworthy and the one-sided rows in that table were measured on a
different configuration than the row beneath them. Finding 40's one-sided rows
should be re-derived or struck.

**The `alim` convention is not the explanation, and can be excluded
arithmetically.** Full `alim` adds only 3,256,348 side-1 records, so it can
touch at most ~3.3M positions — while finding 40's one-sided side-1 figure sits
**7.6M** below the bitmap's popcount. The gap is more than twice what the
convention could possibly move, so the difference is in some other parameter
(`--allowance`/bound being the obvious candidate, since one bound unit moves
that count ~5.5%). Whoever re-derives those rows should record the full command,
which is the reason this was ambiguous at all.

## What the residue actually costs: nothing measurable

Strict set containment is a *proxy*. The property it stands in for is that we do
not lose relations. That is directly testable: the patched run emitted las's 37
relations, and every one maps into our region.

> **All 37 las relations at this q are present in our survivor set.**
> 37 of 37, zero missing, zero unmappable.

So the 988 missing survivors are 0.12% of las's set and cost **zero relations**
here. Combined with finding 40's result that tripling the survivor count also
yields zero extra relations, the picture is that las's survivor boundary is
loose relative to what actually factors, in both directions.

**This reframes the gate, and the reframing is the deliverable.** Set
containment is sufficient but not necessary. The operative gate is **relation
containment**, and it passes. Set containment stays valuable as a *diagnostic* —
it is what surfaced the powlim omission, the boundary column, and the finding-40
table defect, none of which counts-to-0.25% could see — but it should not be a
blocker. Two caveats before leaning on this: 37 relations at one special-q is a
small sample, so the relation-containment gate must run over the 31-q band; and
the remaining 988 have no confirmed cause. Finding 28 established that our norms
differ from las's *in both directions* near root lines, which is the leading
candidate and would make the residue a tolerance property rather than a defect.

**Standing recipe for parity captures**, all corrections folded in: patched
`las`, `-powlim0 32767 -powlim1 32767`, our bitmaps generated in the same
session with the command recorded, and las's `|b| = I/2` column either excluded
or covered by extending our region. Gate on relation containment; report set
containment alongside it as a diagnostic.

### The gate at scale: 3,026 of 3,026, and a basis defect it exposed

The single-q result (37/37) was too small a sample to conclude from, so it was
re-run across the whole band: **67 special-q lattices, 3,162 las relations**,
with our survivor bitmaps generated per lattice at bound 143/141 and las's
relations attributed to their lattice exactly by `a ≡ rho·b (mod q)`.

| | |
|---|---:|
| las relations in the band | 3,162 |
| **contained in our survivor set** | **3,026** |
| **in our region but not a survivor** | **0** |
| outside our sieve region | 136 |

> **Relation containment passes: zero misses in 3,026 in-region relations**,
> an 82x larger sample than the single-q check. Every las relation that our
> sieve region covers is in our survivor set.

**The 136 are not a containment failure — they exposed a real defect in the
parity gate.** `oracle/PARITY.md` records that our q-lattice basis equals las's
with the first vector negated, and calls the gate passed. **That was verified at
one special-q and it does not generalise.** Across the band:

| | lattices | relations | contained | outside region |
|---|---:|---:|---:|---:|
| basis matches las (up to mirror) | **47** | 2,195 | **2,195 (100.000%)** | 0 |
| basis differs | **20** | 967 | 831 | 136 |

On those 20, bench's second basis vector is **exactly las's `v2 − v1`**:

| q | las `v2` | bench `v2` |
|---|---|---|
| 120000169 | (76391812, 1) | (−43608357, 1) |
| 120000193 | (64097954, 1) | (−55902239, 1) |
| 120000341 | (80127018, 1) | (−39873323, 1) |

Same lattice, unimodular transform, different basis — so the `(i,j)` rectangle
covers a **different set of `(a,b)`**. Under the skewed norm (skew ≈ 1.15e8,
`sqrt(skew) ≈ 10742`) bench's vector is the *shorter* one — e.g. at q=120000169,
bench 11,484 against las 12,881 — so **our reduction is the better-reduced of
the two, and las's basis is not Gauss-reduced there.** Neither is wrong; they
are different valid choices, and ours may well yield slightly better norms.

Three consequences, in order of importance:

1. **The relation gate is passed and can be closed.** 0 in-region misses over
   3,026 relations.
2. **PARITY.md's basis claim must be restated.** It holds at 47/67 lattices,
   not universally, and the document should say "verified at q=120000053" rather
   than assert it as a property. Any future per-q parity comparison has to check
   basis agreement *first* — otherwise a region difference reads as a
   correctness failure, which is exactly what it did here for an hour.
3. **Per-q yield comparisons against las are only valid on the matching 47.**
   4.3% of las's relations in this band lie outside our region. That does not
   mean we lose them — our region contains points las does not — but it does
   mean "relations per special-q" is not directly comparable at those q without
   running our own cofactorisation.

4. **Added 2026-08-04 — this closes in-region sieve correctness, not yield
   equivalence, and the distinction is load-bearing.** Containment says every
   relation las finds inside our region is a survivor of ours. It says nothing
   about the relations our region contains that las's does not, and on 20 of 67
   lattices those regions genuinely differ. So **the project's final relation
   yield cannot be read off las at all** — it has to be produced by cofactoring
   our own survivors. Until that exists, "relations per special-q" is a borrowed
   number, and every downstream figure that divides by it (the per-q budget, the
   2× energy target, the ETA) inherits the borrowing. That is acceptable now and
   will not be acceptable at the end.

Worth stating plainly: this was found only because the gate compared *sets*
rather than counts, for the third time in this review.

### Artifacts and how to reproduce

**Written to `oracle/`, but still untracked as of 2026-08-04** — `git status`
shows all of these as `??` and none of them are in `oracle/MANIFEST.sha256`.
They are not committed and the manifest is not regenerated; another session was
live in the repo, so this review deliberately did not commit. **Whoever commits
next must regenerate the manifest**, or `oracle/` acquires a set of oracles with
no checksums, which is the one thing that directory exists to prevent.

| file | what |
|---|---|
| `cado-after-sieve-survdump.patch` | the 3-line CADO patch (plus 4 includes); apply, rebuild `las`, keep the binary **outside** the CADO tree, revert the source |
| `c183.q120000053.after_sieve_ab.powlim.txt.gz` | 795,845 `(a,b)`, `-powlim0/1 32767` — **the oracle to gate against** |
| `c183.q120000053.after_sieve_ab.nopowlim.txt.gz` | 797,028 `(a,b)`, default powlim — kept only to reproduce the powlim finding |
| `c183.q120000053.relations_ab.txt` | las's 37 relations at this q, the relation-containment target |
| `c183.q120000053.cofac_candidates.txt` | 1,851 `a b <cofactor…>` from `-batch-print-survivors`, the target cofactor workload |
| `las_q120000053_after_sieve_powlim.log` | the capture log |

The dump is enabled by `CUDASIEVE_SURVDUMP=<path>`; with the variable unset the
patched binary behaves exactly like stock `las`. CADO's working tree was
restored and verified by md5 against pre-patch copies of both the source file
and the `las` binary; the only local modification remaining is the pre-existing
`-dumpfile` one.

Our side of the comparison came from the `surv.side{0,1}.bits` bitmaps, which
this review confirmed are reproducible bit-exactly by

```
./bench --cadofb ../oracle/c183.fb1 --side 1 --scale 1.275 \
        --fbbound 134200000 --q 120000053 --rho 112625526 \
        --allowance 112 --not-both-even --survbits <file>
```

Note `--fbbound 134200000`. **The two profiles exist because the two CPU
sievers disagree, and the rule is simple: CADO always sieves the full `alim`;
GGNFS always caps `alim` at the special-q when sieving `-a`.** `--fbbound`
defaults to `q`, i.e. to the GGNFS convention. So:

- **Timing against GGNFS** — leave the default. It is the equal-work profile
  and it is correct, because GGNFS caps at q too.
- **Any parity or containment comparison against las** — pass
  `--fbbound <alim>` explicitly. CADO never truncates, so a truncated run is
  comparing against a factor base las did not use.

This is the trap finding 31 warns about, and it should be spelled out in every
parity command rather than inherited from a default.

**The convention costs about 1% of the sieve, measured.** Side-1 records are
312,211,826 truncated at q against **315,468,174** at full `alim` — **+3,256,348,
or +1.04%**. (Independently predicted: ~720K primes in (q, 134.2M], ~1 root
each, `A/p ≈ 4.2` hits apiece ≈ 3.0M records. Observed 3.26M.) So the headline
64.371 ms/q is not materially profile-dependent, and a CADO Gate 0 comparison
would pay roughly 1% more side-1 fill than the GGNFS-equivalent number quoted
throughout this document.

## Also next: unbilled per-q host work is a Goal-1 risk

Goal 1 is stated as *"sustained CPU demand does not grow roughly one-for-one
with GPU count."* There is per-special-q host work sitting outside every timed
number in this project, and it is the one measurement that speaks directly to
that criterion rather than to throughput.

`bench_kernels.cu:651–690` transforms the small factor base on the host, then
sorts it with an **O(n²) insertion sort** over four parallel arrays
(`bench_kernels.cu:677`), at n ≈ 3,631 (side 1) and 3,512 (side 0), **once per
special-q per side**. It is sorted by the *effective* modulus `m` produced by
the transform, which is essentially a random permutation of the p-order the
entries arrive in — so it is the worst case, ~n²/4 ≈ 3.3M inner iterations per
side, not the near-sorted best case.

Replicating that exact loop standalone on this box (busy, so treat as an upper
band): **6–27 ms per q for the sort alone, both sides.** Even the fastest
observed run, 6.1 ms, is ~10% of the 64.4 ms two-sided sieve chain, and none of
it appears in any number this document quotes. At 15.5 q/s that is 0.09–0.4
host cores per GPU for a sort of 3,600 elements.

Three things follow, in order of effort:

1. **Bill it.** Time the block and report it in the chain. It is currently
   invisible, and the previous review already flagged it as unmeasured; it is
   still unmeasured.

> **DONE 2026-08-04, and the billing found something the sort never was.**
> `bench` now prints a `host per-q work` line splitting transform / sort / H2D.
> First measurement, q=120000053, both sides:
>
> | | transform | sort | **H2D** | total |
> |---|---:|---:|---:|---:|
> | side 1, before | 0.40 | (insertion) | **6.16** | 6.59 ms |
> | side 0, before | 0.37 | (insertion) | **6.21** | 6.61 ms |
> | side 1, after | 0.47 | 0.035 | **0.43** | **0.94 ms** |
> | side 0, after | 0.44 | 0.046 | **0.29** | **0.78 ms** |
>
> **13.2 ms/q → 1.7 ms/q**, and almost none of it came from the sort. The
> transfer was the cost: four `cudaMemcpy` calls totalling 54 KB were taking
> **6.2 ms** because the staging arrays were `malloc`'d, i.e. pageable. Isolated
> in a standalone microbenchmark on this box (WSL2, 4 x 15 KB H2D):
>
> ```
> rep 0: pageable 4.653 ms   pinned 0.436 ms
> rep 1: pageable 6.029 ms   pinned 0.464 ms
> rep 2: pageable 5.986 ms   pinned 0.218 ms
> ```
>
> ~1.5 ms per pageable call regardless of size — a per-call cost, not
> bandwidth. Switching the four arrays to `cudaHostAlloc` is the whole fix.
> **This is a Goal-1 result, not a micro-optimisation:** 11.5 ms/q of pure host
> overhead was being spent, invisibly, on 54 KB, against a sieve chain of
> 53–83 ms. Anything else in the design that stages small per-q buffers through
> pageable memory will pay the same toll, and the only reason this one was
> found is that the block got a timer.
>
> Both changes are output-neutral: the survivor bitmaps at q=120000053 are
> **bit-identical on both sides** to the pre-change binary (22,499,522 and
> 24,301,359). `std::stable_sort` was used rather than `std::sort` precisely so
> that this test means something — ties in `m` are common and the insertion sort
> it replaces was stable.
2. **Fix the sort.** A comparison sort makes it ~43K comparisons instead of
   3.3M and removes the entire item. There is no design question here — but
   **it is not the one-line `qsort` swap this originally said it was**
   (corrected 2026-08-04). The loop at `bench_kernels.cu:677` sorts **four
   parallel arrays** in lockstep — `hsp` (the key, effective modulus `m`),
   `hsrt`, `hsg`, `hslp` — and `qsort` has no way to permute them together.
   Two workable shapes:
   - **Sort an index permutation** (`std::sort` on `uint32_t idx[]` with a
     comparator reading `hsp`), then scatter into four fresh arrays before the
     `cudaMemcpy`s. Least churn, and it preserves the SoA layout the small-sieve
     kernel wants.
   - **Sort an array of `{m, rt, g, logp}` structs** with `std::sort`, then
     scatter into the four arrays. Marginally better cache behaviour during the
     sort, same scatter afterwards.

   Either way there is a scatter pass, because the device side is SoA and must
   stay SoA. When billing it, **time transform, sort and transfer separately** —
   the current single unmeasured block hides which of the three actually costs,
   and the answer determines whether item 3 below is worth doing at all.
3. **Then decide whether the transform itself belongs on the GPU.** It is
   `k_transform`-shaped work and Path 5 wants it there anyway, but after (2)
   the residual is small enough that this becomes a Path-5 tidy-up rather than
   a Goal-1 risk.

This matters more than its milliseconds because of what it is evidence *of*: the
project's success criterion is about host demand, and the only per-q host work
in the codebase is both quadratic and untimed. Whatever else Path 4 adds, the
rule should be that any per-q host stage is timed and reported from the day it
is written.

## MEASURED: the survivor bound is loose by 2x, and it costs nothing

Finding 40 measured the bound *upward* — 37 relations at bound, bound+6,
bound+12. This review measured it **downward**, over 68 special-q in
[120000000, 120001000] with `-powlim0/1 32767`, `-adjust-strategy 0`, `-t 16`,
varying `lambda1` only (`lambda0` held at 2.35, so the side-0 bound stays 141):

| `lambda1` | side-1 bound | survivors/q | TD input/q | **cofactors/q** | relations | relations lost |
|---:|---:|---:|---:|---:|---:|---:|
| **3.5** (default) | 143 | 781,802 | 32,932 | **1,907** | 3,162 | — |
| 3.4 | 139 | 619,950 | 26,093 | **1,907** | 3,162 | **0** (identical set) |
| 3.3 | 135 | 488,722 | 20,562 | **1,907** | 3,162 | **0** (identical set) |
| **3.2** | **131** | **383,168** | **16,112** | **1,907** | **3,162** | **0** (identical set) |
| 3.1 | 127 | 298,831 | — | 1,906 | 3,161 | 1 (0.03%) |
| 3.0 | 123 | 231,832 | — | 1,903 | 3,159 | 3 (0.09%) |
| 2.9 | 119 | 179,006 | — | 1,899 | 3,158 | 4 (0.13%) |
| 2.8 | 115 | 137,561 | — | **1,458** | 2,949 | **213 (6.7%)** |

**Halving the survivor count costs zero relations, and there is no cliff — only
a very shallow slope.** Not "approximately the same count": the relation *sets*
were extracted and compared element by element, and the symmetric difference is
**0** down to bound 131. Applying the lesson from the containment gate above,
counts were not trusted on their own.

Past 131 the loss is barely measurable, and it stays that way far further down
than expected. **Removing 77% of survivors costs 0.13% of relations**:
bound 143 → 119 drops 602,796 survivors/q to lose 0.059 relations/q. So the operating point is an economic choice, not a correctness
boundary:

- **bound 131 — free**, but this was measured at 4-unit resolution and is not
  the floor. The refinement below finds the true floor at **128** (2.46× rather
  than 2.04×) and recommends **130** as the operating point. Read that section
  before using any number from this list.
- **bound 127 — costs 0.03%** of relations for a further 22% of survivors.
- **bound 123 — costs 0.09%** for a further 40%.
- **bound 119 — costs 0.13%** for a further 53%.

If resieve and TD turn out to be the critical path, trading 0.09% of relations
for 70% of their input is obviously correct. Decide it once those two stages are
measured, not before. **The real cliff is at bound 115**, where 6.7% of
relations vanish at once — so 119 is the practical limit. **The free point
is 128, not 131** (refinement below).

### REFINED (2026-08-04): the sweep's resolution was 4 units — the floor is 128

**The sweep never sampled the edge.** It varied `lambda1` in steps of 0.1, and
the bound is `(unsigned char)(scale1 * lambda1 * lpb1) + 1` with
`scale1 = 1/log2(1.722273) = 1.274995`, i.e. `bound1 = floor(40.7998 * lambda1) + 1`.
**A 0.1 step in `lambda1` is 4.08 units of bound.** The table above therefore has
rows at 143, 139, 135, 131, 127 and nothing between them; 131 was the lowest
*sampled* zero-loss bound, not the lowest one. The six missing bounds were run
(`oracle/bound_sweep_fine.sh`, `-t 8` at `nice 19`; counts are thread- and
load-independent, which is the only reason this was safe on a loaded box):

| `lambda1` | bound | survivors/q | `enter_cofactoring` | relations | lost |
|---:|---:|---:|---:|---:|---:|
| 3.5 | 143 | 781,803 | 129,711 | 3,162 | — |
| 3.28 | 134 | 460,127 | 129,705 | 3,162 | 0 |
| 3.25 | 133 | 433,046 | 129,704 | 3,162 | 0 |
| 3.22 | 132 | 407,412 | 129,704 | 3,162 | 0 |
| 3.2 | 131 | 383,168 | 129,699 | 3,162 | 0 |
| 3.17 | 130 | 360,245 | 129,693 | 3,162 | 0 |
| 3.15 | 129 | 338,589 | 129,681 | 3,162 | 0 |
| **3.12** | **128** | **318,141** | 129,664 | **3,162** | **0 — the floor** |
| 3.1 | 127 | 298,832 | 129,633 | 3,161 | 1 |

**The zero-loss floor is exactly 128**, and 127 is the first bound that costs
anything — one relation in 3,162. So the reduction available for free is larger
than reported: **781,803 → 318,141 survivors/q, 2.46×**, against the 2.04× that
bound 131 was credited with.

**A correction to my own reading of the `enter_cofactoring` column.** The claim
above that it is "pinned at exactly 1,907 for bounds 143 → 131" is an artifact
of rounding the per-q figure. The raw totals drift continuously from 135
downward — 129,711 / 129,711 / 129,707 / 129,705 / 129,704 / 129,704 / 129,699 /
129,693 / 129,681 / 129,664 / 129,633. **There is no plateau in that column.**

It survives as a leading indicator, but for a different reason than I gave: it
starts moving at bound 135, **eight units before the relation count moves at
127**. Losing cofactor candidates is not the same as losing relations — the
marginal candidates are overwhelmingly unproductive — so drift there is an early
warning rather than a cost. Only the collapse at 115 (−23%) tracks a real
relation cliff (−6.7%). Tune on it, but read it as a margin gauge, not as a
threshold.

#### The gate was rerun at bound 128 on our own bitmaps, and it passes

The sweep proves *CADO's* relation set survives at 128 under *CADO's* norms.
That is not the same claim as ours surviving, because our norm approximation is
not bit-identical to las's — findings 40–41 record a one-unit difference and
finding 28 records it as **two-directional**, so at the floor a position that
rounds the wrong way behaves as if the bound were 127, which is exactly where
relations start to go. The conservative reading was to adopt 130 and keep two
units of margin.

**That is not necessary. The gate was rerun and the margin is not needed.**
Regenerating the 67 side-1 bitmaps at `--allowance 100` (which `bench` confirms
gives `survivor bound = 128`) and re-running `relcontain_band` against the same
3,162 las relations reproduces the bound-143 result exactly:

```
lattices: 67  (basis matches las: 47, differs: 20)
relations: 3162   unattributed: 0
  CONTAINED (in our region, is a survivor): 3026
  outside our sieve region (basis differs): 136
  in region but NOT a survivor  <-- real : 0
```

Only side 1 needed regenerating — the side-0 bound is set by `lambda0`, which
the sweep held fixed at 141. Sixty-seven `bench` runs at 0.8 s each.

And this is a real test, not a null one — our survivor set genuinely tightened
by the expected factor (q=120000053, popcounts over the raw bitmaps):

| | side 0 | side 1 | two-sided, pre-gcd |
|---|---:|---:|---:|
| bound 143 | 24,301,359 | 22,499,522 | 1,386,939 |
| bound 128 | 24,301,359 | **9,521,087** | **590,665** |

The 1,386,939 reproduces finding 41's pre-gcd count exactly, which is what says
the two bitmaps are being compared on the same footing. **Our own two-sided
survivor set falls 2.35× and loses zero relations** — measured on our norms,
not inferred from CADO's.

**Operating point: bound 128.**

| bound | CADO survivor reduction | status |
|---:|---:|---|
| **128** | **2.46×** | **adopt** — the CADO floor, and our own gate passes here (3,026/3,026) |
| 130 | 2.17× | the conservative point, no longer needed |
| 131 | 2.04× | what was previously recommended; leaves 1.2× on the table |
| ≤127 | more | below the floor, costs relations: 0.03% at 127, 0.13% at 119, **6.7% at 115** |

The one thing this does *not* license is tightening further on the strength of
the gate. 127 costs a relation in CADO's own sweep; the gate can only tell you
that our norms do not lose something CADO keeps, never that a bound below the
floor is safe.

Full data, including the sensitivity of the operating-point choice, is in
`oracle/bound_sweep_results.txt`.

### `enter_cofactoring` is the leading indicator, and it is the one to tune on

Read the cofactor column down the table. It is pinned at **exactly 1,907** for
bounds 143 → 131 — the whole range where the relation set is identical — then
drifts (1,906 / 1,903 / 1,899) across exactly the range where 1, 3 and 4
relations are lost, then collapses to 1,458 at the cliff. That is not a
coincidence, it is the mechanism: while the bound only removes survivors that
would have failed trial division anyway, the cofactor population cannot move;
the moment it starts falling, the bound is cutting into candidates that could
have factored.

That makes it a better tuning signal than the relation count itself:

- It moves **~3x more** in relative terms (−0.42% vs −0.13% at bound 119), so
  it is far better conditioned than counting 1–4 lost relations out of 3,162.
- It is a single number available from `las -v` without needing to compare
  relation sets element by element.

**Tuning rule: take the smallest bound at which `enter_cofactoring` is still
pinned at its plateau.** On this band that is 131, and it is exactly the free
point found by set comparison. That rule is worth carrying into the q = 50M and
190M sensitivity runs, since it turns each one into a single cheap `las -v`
read rather than a full relation-set diff.

Two facts make this the most useful measurement in this review:

1. **It cuts exactly the stages that were binding.** Survivors −51% and TD input
   −51%. Resieve's filter cost and TD's cost are both linear in survivor count,
   and those two were the ~12–22 ms sitting against a ~16–20 ms allowance.
2. **It does not touch the stage that was already fine.** `enter_cofactoring`
   is **exactly 1,907/q at every bound**, unchanged to the last digit. The hard
   cofactor population is already the irreducible set, so finding 47's 4.92 ms/q
   is insensitive to this lever — and, conversely, no amount of bound tuning
   will improve it.

**Revised post-sieve budget at the free operating point, `lambda1 = 3.2`
(bound 131):**

| stage | at bound 143 | **at bound 131** |
|---|---:|---:|
| intersect + compact | ~0.5 | ~0.25 |
| resieve (fixed re-walk + survivor-scaled filter) | ~8–14 | **~6.5–11** |
| TD (scales with survivors) | ~4–8 | **~2–4** |
| hard cofactors (finding 47, unchanged) | 4.92 | **4.92** |
| **total** | ~18–27 | **~13.7–20.2** |

Against finding 46's ~21–25 ms allowance for a **2x energy win**, the budget
moves from *on the boundary* to *inside it*. That is a materially different
project outlook than this document had yesterday, and it came from four `las`
runs and no code.

**Three caveats, and the first one matters.**

- **This is one q-band.** 68 special-q near q = 120M, against a job range of
  [50M, 190M]. Norm sizes shift with q, so the bound that is free here may not
  be free at the ends. The doc's own operating point already names q0 = 50M and
  q0 = 190M as sensitivity points; **re-run this sweep there before adopting a
  tightened bound in production.**
- **The slope past 131 is shallow enough to be worth re-measuring with more
  relations.** A 1-relation difference in 3,162 is a 0.03% effect measured on a
  single band; it is directionally right but it is one relation. Widen the band
  before treating bound 127 or 123 as characterised.
- **`lambda0` was never swept.** The rational bound stayed at 141 throughout.
  There may be a second, independent lever there, and the same experiment sizes
  it.

### Item 6 delivered: the target cofactor workload, at scale

The same window produced the input finding 47's controlled rerun has been
waiting for: **1,062,811 cofactor candidates over 544 special-q** in
[120000000, 120010000], captured with `-batch-print-survivors` at default
`lambda` with `powlim` pinned (61 MB, six files, 370 s at `-t 16`). That is
1,957 records/q, and it is the **real C183 `lpb 31/32` population** — not the
C164 `31/31` dataset finding 47 had to use. It is a million records, the same
scale that benchmark ran at, so the two are directly comparable.

Note this capture is at bound 143. If the tightened bound is adopted, the
cofactor population moves by less than 0.04% down to the floor (129,711 →
129,664 over the whole band), so this file remains valid
input either way.

A note on scope: this tunes *our* post-sieve input. It is not a criticism of
las's default — `lambda1 = 3.5` is a safe general-purpose value, and CADO pays
much less than we do for a loose bound because its funnel differs. The finding
is that a GPU-resident design gets to choose, and choosing well is worth ~2x on
the two stages that were binding.

#### The mix, measured (2026-08-04), and what it does to finding 47's number

The capture was histogrammed rather than assumed. 548 special-q headers,
1,062,811 records, **1,939 records/q** (the 544/1,957 above are off by the four
q whose headers carry no candidates — corrected here):

| bits | side 0 (rational, `lpb 31`) | side 1 (algebraic, `lpb 32`) |
|---:|---:|---:|
| 1–31 | 517,824 — 48.72% | 7,590 — 0.71% |
| 32–62 | **544,987 — 51.28%** | 17,949 — 1.69% |
| 63–64 | 0 | 11,357 — 1.07% |
| 65–96 | 0 (max = 60 bits) | **1,025,915 — 96.53%** |

**This workload is heavier than the C164 `31/31` set finding 47 measured on.**
Applying finding 47's per-class rates to this mix projects **≈2.99 s for the
1,062,811 records — 355K candidates/s, ≈5.5 ms/q**, about **12% slower** than
the 4.92 ms/q sizing datum, and still roughly **11.7×** the rate the sieve chain
needs to be fed at. The headroom conclusion survives; the number moves.

**But treat even that projection as fragile, for a specific reason.** 96.53% of
the algebraic side lands in the 65–96 bit class, and at `mfba 92` with
`lpba 32` that class is **three large primes**. Finding 47's per-class rate for
it was measured on a 2LP splitting problem. The entire projection therefore
rests on transporting one rate across a change in the *kind* of factorization,
not just its size — and 3LP splitting is where ECM curve counts and the MPQS
fallback behave least like their 2LP counterparts. This is a stage-mix
projection and nothing more; the real `lpba 32` curve behaviour and the actual
yield still require the controlled run.

**Harness gap found while preparing that run:** the captured files are not in a
format the existing YAFU test parser accepts (`a b cofac0 cofac1`, with
`# q = (q, rho, side)` header lines between q blocks). The controlled rerun
needs a small input adapter or a parser mode, on top of the harness fixes
already listed under finding 47.

## GPU resieve — CORRECTED (2026-08-04): not settled, and the A/B has to be run

**I got CADO's shorthint backwards, and it invalidates the conclusion below.**
The claim in the original version of this section — that a 4 B record "loses the
prime by construction, which is why CADO resieves rather than re-reading its own
buckets" — is wrong on both halves. Read the source:

```cpp
/* sieve/bucket.hpp:87 */
class shorthint_t {
  public:
    slice_offset_t hint = 0;          // offset of the prime WITHIN its slice
    shorthint_t(fbprime_t, slice_offset_t slice_offset, slice_index_t)
        : hint(slice_offset) { }
};

/* sieve/bucket.cpp:326 — recovery, in the purge path */
slice_index_t const slice_index = BA.get_slice_index(i_slice);
for (auto const & it : BA.slice_range(i, i_slice))
    if (UNLIKELY(S[it.x] != 255))
        fbprime_t const p = fb[slice_index].get_prime(it.hint);
```

Two facts follow, and both cut against what this section concluded:

1. **CADO's 16 bits are a within-slice offset, not a slice index.** Slice
   identity comes from the enclosing bucket *segment* — `slice_range(i, i_slice)`
   — so `(segment, hint)` names the factor-base entry exactly and
   `get_prime(hint)` returns `p` with no search. Our 16 bits are the slice index
   itself (`bench_kernels.cu:99`, `sl << 16`), which recovers `log p` and
   nothing more. **Ours is not "CADO's shorthint exactly"; it is strictly less
   information in the same 4 bytes.** The comment at `bench_kernels.cu:88–90`
   asserting that this field is "what resieve would use to recover the prime
   itself, exactly as CADO's shorthint does" is false and should be corrected in
   the source.
2. **CADO does not resieve to recover large primes.** `bucket_array_complete::purge`
   (`bucket.cpp:370`) walks the *retained* bucket array, keeps updates whose
   `S[it.x] != 255`, and rewrites each survivor as a 6 B `longhint_t`
   (`slice_index` + `hint`). Resieving is reserved for the small tier below
   `bkthresh` — the primes that were never bucketed. So the option this section
   killed as "insufficient, not merely slow" is the one CADO actually ships; it
   is insufficient *for our record layout*, which is an argument for changing the
   layout, not for abandoning the approach.

Corrected option table:

| option | what it costs | verdict |
|---|---|---|
| widen records to 8 B carrying the FB index | fill is the largest stage at 24.5 ms; ~1.2 GB/q extra write traffic | still **dead** — a permanent sieve cost to save a one-off post-sieve cost |
| **A — re-walk the FB against a hierarchical survivor filter** | measured occupancy below; est. 8–14 ms/side-pair | **candidate**, not the answer |
| **B — slice-segmented 4 B records (16-bit offset + 16-bit within-slice offset) + CADO-style purge** | 2.43 GB second streaming read; segmentation overhead in fill | **candidate — must be benchmarked against A** |
| 14-bit offset + 18-bit hint, unsegmented | 2.43 GB stream, ~26 candidate primes per record | **fallback only** — see below |

The 14/18 split is worth keeping on record because it is the version that needs
no segmentation: region is 2^14, so 14 bits suffice for the offset, and 18 bits
give 262,144 slices of ~26 entries. A re-read then pins the prime to ~26
candidates and brute-forcing 26 modular tests on only the ~880K survivor hits is
free. Its cost lands in *apply*: the `log p` lookup table goes from 2 KB in
shared memory to 512 KB in L2 (`bench_kernels.cu:323` masks a shared-mem LUT),
adding a dependent L2 load per record to a kernel finding 8 already shows
DRAM-bound. Layout B gets the same exact-prime recovery *without* that penalty,
because the segment already names the slice and the LUT stays 2 KB. **B strictly
dominates the 14/18 split.** Keep 14/18 only if segmentation turns out to cost
more than the L2 LUT does.

### Layout B: segmentation without a counting pass

The obvious objection to B is that per-`(bucket, slice)` allocation needs a
counting pass, and a counting pass costs about what the re-walk it replaces
costs. It does not, and the reason is worth writing down:

- Fill **one slice per launch**, in slice order, with a single `cursor[b]` per
  bucket exactly as now. Records from a given slice therefore land contiguously
  within each bucket, because nothing else is writing at the same time.
- After each slice's launch, snapshot `cursor[]` into row `s` of a boundary
  table. That table *is* CADO's per-slice pointer array, obtained for free.
- Size: `nbuckets × nslices × 4 B`. **Measured**, not estimated — `bench --side 1
  --q 120000053 --rho 112625526` prints it at startup:

  ```
    bucketed  32768 <= p < 120000053 : 6840490 entries
    sieve area I=2^15 x J=16384; region 2^14; 32768 regions
    factor base cut into 39 slices (padded to 64), log p in [15,27] bits
    single-level: 32768 buckets x cap 11202 x 4 B = 1.37 GB
    records landed : 312211826
  ```

  So **39 slices on side 1** and 32,768 buckets, giving a boundary table of
  `32768 × 39 × 4 B = 5.1 MB` (8.4 MB if it is padded to 64 like the `logp`
  LUT). Negligible either way.

**Do not assume ~120 segments.** That figure has been quoted in review; the real
count is 39, and it follows mechanically from `build_slices`
(`bench_kernels.cu:591`), which cuts on a `logp` change *or* every 262,144
entries. Note this run truncates the factor base at `q` (the GGNFS convention —
`--fbbound` defaults to `q`); at CADO's full-`alim` convention side 1 carries
all 7,605,406 pairs and the count rises by a few slices.

Total fill *work* is unchanged — every prime is walked exactly once either way.
What changes is the launch structure, and that is where B's real risk lives:

- ~39 launches on side 1 against 1 today. At ~5 µs of launch overhead that is
  ~0.2 ms/q of pure overhead — small against 24.5 ms, but not nothing, and it is
  the kind of thing CUDA graphs or a cooperative-groups grid-wide sync exists to
  remove. Cost it before dismissing it.
- Load imbalance within a launch: the small-`p` slices generate far more hits
  per entry than the large-`p` ones, so per-launch occupancy will vary. The
  262,144-entry cap bounds entries per slice but *not* hits per slice. With 39
  slices over `log p ∈ [15,27]`, the first few slices carry most of the 312M
  records — those launches are fine, and it is the *last* slices, with few hits
  spread over 32,768 buckets, that will underutilise the device.
- `cursor[b]` contention is per-bucket and unchanged; only the number of
  kernel boundaries changes.

### Layout A: re-walk against a hierarchical survivor filter

**The re-walk is cheap because the intersected survivor set is sparse, and it is
only cheap because of that.** This is the structural reason step 1 must precede
step 2, independent of the correctness argument:

- Two-sided survivors: ~800K positions out of A = 536,870,912, i.e. 1 in 671.
- Group positions into 64-position words: 8.39M words, of which **783,415
  actually contain a survivor — 9.34% occupancy, in a 1.05 MB summary bitmap.**
- That table is permanently L2-resident. Each of the 607M bucket-prime hits
  tests one bit in it; **90.7% are rejected there**, and only ~57M proceed to a
  read of the full 67 MB bitmap.

**This table is measured, not assumed.** The `surv.side{0,1}.bits` bitmaps from
the finding-40 session were ANDed and gcd-filtered directly; the intermediate
counts reproduce finding 41 exactly (1,386,939 pre-gcd, 841,418 primitive),
which is what says the measurement is of the right thing.

| granularity | summary table | measured occupancy | 2nd-level DRAM reads |
|---:|---:|---:|---:|
| 32 | 2.10 MB | 4.84% | 29 M |
| **64** | **1.05 MB** | **9.34%** | **57 M** |
| 128 | 0.52 MB | 17.45% | 106 M |
| 256 | 0.26 MB | 30.81% | 187 M |

Survivors turn out to be very nearly Poisson at this scale — 841,418 survivors
in 783,415 occupied words is 1.073 per occupied word against ~1.05 for a random
spread — so the clustering one might hope for near root lines buys almost
nothing at granularity 64, and the assumed 9.5% was right for the wrong reason.
Coarser granularities do show clustering (30.8% at 256 against 38.0% random),
but 64 is the operating point.

Cost estimate, both sides: the walk itself is fill minus the scatter (fill is
12.6/11.9 ms *including* 280M L2 write sectors, so the bare walk is the smaller
part); 607M L2-resident summary probes; ~58M scattered DRAM sector reads ≈
1.85 GB ≈ 3 ms; output ~880K `(survivor, prime)` pairs, which is nothing.
**Estimate ~8–14 ms/q for both sides, with no extra bucket storage and no
change to the record format.** Label this an estimate — it is derived, not
measured.

Now run the same arithmetic against a *one-sided* survivor set (14.9M / 18.9M):
occupancy exceeds 100% at 64-position granularity, the summary rejects nothing,
and every one of the 607M hits goes to DRAM. Resieving before intersecting is
roughly an order of magnitude worse. **Intersect, then resieve** — and the
sequence is forced by cost, not just by taste.

**Order the two sides, cheap one first.** las trial-divides side 0 first and the
funnel shows why: 797,028 → 33,355 after side-0 TD. Restricting the side-1
re-walk to those 33K survivors drops its summary occupancy from 9.5% to 0.4%,
so nearly every side-1 hit is rejected in L2 and the second-level traffic
almost vanishes. It costs a serial dependency between the two sides; take it.

### What the A/B must measure

Recovery cost alone is the wrong metric, because A and B move cost between
stages in opposite directions: A adds a post-sieve pass and touches fill not at
all; B adds launch structure to fill and makes recovery a streaming read.
**Measure `fill + apply + recovery` end to end, per q, for both.** First-order
expectation, to be falsified rather than trusted:

| | A (re-walk) | B (segment + purge) |
|---|---|---|
| fill | unchanged | +launch overhead, +2 MB boundary writes |
| apply | unchanged | unchanged (2 KB LUT survives) |
| recovery | bare FB walk + 607M L2 filter probes + ~58M DRAM sectors | 2.43 GB streamed once ≈ 1.4–1.6 ms at 1.5–1.8 TB/s, + compaction of survivors |
| bucket memory | can be freed after apply | must stay resident through recovery — **1.37 GB for side 1 alone** at the measured `32768 × 11202 × 4 B`, and both sides are needed |
| prime recovery | exact, by construction | exact, `get_prime(hint)` |

The memory row looked like the one most likely to decide it, and **it does not**:
the same bench startup reports **10.76 GB free of 11.94 GB**, against 1.37 GB of
side-1 buckets and a comparable side-0 allocation — ~2.7 GB for both, on top of
two 67 MB bitmaps and the factor base. B's "extra" residency is only extra if
the design was going to reuse the bucket allocation for the cofactor queue, and
there is room not to. **So B is not memory-blocked and the A/B is worth running
on its merits.**

And a third possibility that neither option covers, already flagged in the
post-sieve section and easy to forget once the A/B framing takes hold: **recover
no large primes at all.** CADO's `-batch` mode factors the surviving cofactor
against a product tree of the factor base. That deletes the recovery stage
outright and replaces it with a batch remainder-tree cost, which is a different
shape of expense and is the one that scales with *survivors*, not with *hits*.
It should be on the same chart as A and B.

## Trial division — the arithmetic is free, the small primes are the cost

**Norm recomputation is not a concern and should not be treated as one.** The
log scales pin the sizes: 255/1.275 ≈ 200 bits algebraic, 255/1.925 ≈ 132 bits
rational, so 256-bit and 160-bit integers. A degree-6 Horner in 256-bit
arithmetic is ~150–200 IMADs; at 800K survivors that is ~0.16 Gops/side per q
against a card doing on the order of 15 T lane-ops/s. **Under 0.05 ms.** Exact
division by a recovered 32-bit prime is a multiply-by-inverse chain, similarly
negligible. Nothing in the big-integer path is a throughput risk — the entire
question is memory access patterns, which is a statement worth making
explicitly because "GPU big-int TD" sounds hard and is not.

Note the norms here must be **exact integers recomputed from `(a,b)`**, not the
sieve's hybrid fp32/fp64 approximation. Finding 28 established that the sieve
norm is wrong in both directions near root lines; it is a log approximation and
was only ever meant to be one.

**Small primes (`p < bkthresh = 2^15`) must NOT be resieved.** The small sieve
carries 6.84e9 updates per q. Even at one L2-resident summary probe each, that
is ~40 ms — worse than everything else in the post-sieve budget combined. Test
them directly instead: for a survivor at `(i,j)`, `p` divides the norm iff
`i ≡ rt·j (mod p)` with the transformed root already sitting in the small-sieve
tables. At ~3,600 entries per side and ~4 ops per test, that is **~11.5 Gops
per side per q, ≈1 ms at peak and plausibly 3–5 ms achieved.** Perfectly
regular, fully coalesced over the prime list, no divergence.

So the split inverts relative to the sieve's own reasoning:

- `p ≥ 2^15` (bucketed): **resieve**, because hits are rare and survivors sparse.
- `p < 2^15` (small): **direct test per survivor**, because hits are dense.

The crossover is not at `bkthresh` and there is no reason it should be. Per
prime, direct testing costs `n_survivors × 4` ops while resieving costs `A/p`
probes; they cross near `p ≈ A/(4·n_survivors) ≈ 168`. Between 168 and 2^15
it is close to a wash (≈490M hits to resieve versus ≈11 Gops to test directly),
so this is a tuning knob, not a decision. Take the direct test for the whole
small tier because it needs no second kernel and no second data structure.

## Where the budget stands, with finding 47 folded in

Finding 47 landed while this section was being written and it moves the ranking,
so the ordering below is *not* what I would have written from the cost model
alone. My independent op-count estimate for the hard-cofactor stage was ~7 ms/q
at peak and ~25 ms/q at a realistic fraction of it, on an assumption of B1 in the
low thousands and ~20 curves. Finding 47's provisional recurring-work figure is
**4.92 ms/q**, better than my optimistic end — because the measured harness runs
`b1=300` with a ten-curve no-success stopping rule, which is a much cheaper
strategy than I assumed. The measurement supersedes the model; I record the
model only because it was the basis for a recommendation that the measurement
then changed.

| stage | ms/q | basis |
|---|---:|---|
| intersect two 67 MB bitmaps + compact + gcd filter | ~0.5 | estimate: 134 MB streamed, 1.4M candidates compacted |
| resieve, both sides, re-walk + hierarchical filter | **~8–14** | estimate, above |
| TD: exact norms + small-prime direct tests | **~4–8** | estimate, above |
| hard cofactors, ~1,851/q | **4.92** | finding 47, provisional, non-target C164 `31/31` |
| **total post-sieve** | **~18–27** | |

Against finding 46: energy parity (≤106–114 ms) is comfortably met. **A 2×
energy win (≤21–25 ms) is now genuinely on the boundary**, and the thing sitting
on that boundary is resieve + TD at ~12–22 ms of a ~16–20 ms remaining
allowance. Finding 47 reached the same conclusion from the other side —
*"if the target rerun holds near 4.9 ms/q, GPU resieve/factor recovery/TD
becomes the largest unknown inside the 2× budget"* — and the estimate above is
the first sizing of that unknown. It says the 2× target is reachable but has no
slack, which makes two things load-bearing that would otherwise be tuning:

- **The survivor-bound sweep.** Resieve and TD are both linear in the survivor
  count. If the bound can drop with no relation loss, this is the difference
  between clearing 2× and missing it.
- **The `p < 2^15` direct-test kernel.** It is the larger half of my TD
  estimate and it is a completely regular, coalesced kernel. If it lands at the
  1 ms peak rather than the 5 ms pessimistic figure, the budget stops being
  tight.

Both should be settled before anyone concludes where in the zone this lands.

**One architectural point survives the reranking unchanged.** 1,851 cofactors
per q fills 2.5% of this device's 73,728 resident threads, so the cofactor stage
cannot run in per-q lockstep — it needs a device-resident queue spanning many q.
Finding 47 reaches this from its own direction (one million records ≈ 526 q ≈
34 seconds of arrivals, and its rerun must sweep smaller batches). Two ways to
fill the device: run curves in parallel within a cofactor, which forfeits the
early exit that finding 47's `-s 10` stopping rule depends on; or batch across
q, which keeps it. **Batch across q** — latency is irrelevant for a throughput
job. Assume the decoupled producer/consumer shape from the start; retrofitting
it is expensive, and it is the only hard-to-reverse decision in the post-sieve
path.

## Recommended order

0. **Done in this review.** The CADO patch exists and the oracle is captured;
   `oracle/` holds the `after_sieve` `(a,b)` sets (with and without `powlim`
   pinning), las's 37 relations, and the cofactor-candidate file. CADO itself
   is restored to its pre-review state.
1. **Done in this review.** Relation containment ran over the full 67-lattice
   band: 3,026/3,026 in-region relations contained, zero misses. It also
   exposed that PARITY.md's q-lattice basis claim holds at only 47 of 67
   lattices; restate it and add a basis-agreement check to any per-q parity
   comparison.
2. ~~**Bill and fix the per-q host work**~~ **Done 2026-08-04.** Index-permutation
   `stable_sort` replaced the O(n^2) insertion sort, and transform/sort/H2D are
   now billed separately. The billing is what mattered: the sort was never the
   cost (0.03 ms) — **pageable staging buffers were**, at 6.2 ms/side for 54 KB.
   Pinned host memory takes per-q host work from **13.2 ms to 1.7 ms**, output
   bit-identical. See the block under "Also next" above.
3. ~~**Device intersect + compact + primitive filter**~~ **Done 2026-08-04.**
   `bench --other-bits FILE [--emit FILE]` runs it on the device and emits
   **both** `x` and `(a,b)`. Measured **0.448 ms/q** — see below.
4. **Survivor-bound sweep downward** on las over the band (CPU-side, concurrent
   with 3). It sizes resieve and TD, which are the binding stages.
4b. **Capture the target cofactor population** with `-batch-print-survivors`
   over ~50–500 q. No patch needed; feeds item 6.
5. ~~**Resieve + TD**~~ **A/B RUN 2026-08-04 — layout A wins, 20.8 ms vs 53.8 ms
   for fill+recovery.** Layout B was fully built (segmented fill + prime-naming
   purge) and recovers an identical `(x,p)` set, but segmentation costs ~0.3 ms
   per pass in bucket-array write amplification. See "Item 5 RESOLVED" below.
   Original framing: layout A (re-walk +
   hierarchical survivor filter) against layout B (slice-segmented records with
   an exact within-slice offset + CADO-style purge), measured as
   `fill + apply + recovery` end to end, with batch/product-tree recovery on the
   same chart as a third point. The largest unknown in the 2× budget, and now
   also the largest *open design question*.
5b. **Sweep the small/large split rather than assuming it.** "Direct testing for
   `p < 2^15`" was asserted, not measured; `2^15` is simply the default
   `bkthresh`. CADO's own policy is a mixed direct/resieve split and is the right
   starting point, but the boundary is a free parameter and both A and B are
   sensitive to it in opposite directions. Sweep it inside the A/B.
6. **Finding 47's controlled rerun** on the target `31/32` population, using
   4b's capture. Concurrent with 3–5 whenever the box is free. **Note the
   harness gap:** the captured CADO files are not in a format the existing YAFU
   test parser accepts, so this item needs a small input adapter or a parser
   mode *in addition* to the harness fixes already listed under finding 47.
7. Wire the device-resident cofactor queue and measure the assembled pipeline.

Items 1, 4, 4b and 6 use the CPU; items 2, 3, 5 and 5b use the GPU. They do not
contend.

**Two ordering notes added after the 2026-08-04 cross-review:**

- Item 3 should **retain the compact `x` indices**, not just derive `(a,b)` and
  discard them. Recovery — under either layout — is indexed by `x`; a design
  that throws `x` away at compaction has to reconstruct it, and `(a,b) → x`
  requires inverting the lattice basis per survivor. Emit both.
- Item 4 is **REOPENED — see "The survivor bound is 131 on side 0" in the
  2026-08-04 build log.** Bound 128 loses one of las's 37 relations at
  q=120000053 on our own sieve; the working pair is 128 on side 1 and 132 on
  side 0. The sweep below was run over CADO's bytes, and our bytes differ at
  exactly the positions that matter — a norm carrying a high power of a small
  prime is under-credited by the sieve, because the factor base stops at
  p^k < 2^15 while trial division divides out full multiplicity.
  Original text: The six missing probes were run and the
  true zero-loss floor is 128, not 131. The containment gate was then rerun on
  our own bitmaps at that bound and passes 3,026/3,026 — so the tightening is
  validated on our norms, not just CADO's. Hard-code **128**: 2.46× fewer
  survivors on CADO's count, 2.35× on ours, zero relations lost. Do not go
  below it; 127 costs a relation in CADO's own sweep.

**The loose end, diagnosed 2026-08-04 — and it splits cleanly in two.** Side 1's
722 and side 0's 266 missing survivors turn out to have *different* causes, and
neither is a sieving defect. See "The missing-survivor residue" below.

## The missing-survivor residue — diagnosed, and it is not a sieve defect

Run with `las_tracek -traceab` (the tool `oracle/PARITY.md` already prescribed),
against our own dumps at the same q. The 988 missing positions split into two
populations with unrelated causes.

### Side 1: every one is exactly `bound + 1`

Our sieve byte at all 88 side-1 misses in the sample is **144**, against a bound
of 143. Not one is further out:

```
our side-1 byte at the missing positions (bound 143):
   144 :   88
```

A traced example confirms the other half of the picture — las's own value at the
same `(a,b)` is **exactly 143**, i.e. exactly on the bound, since a survivor is
`S[x] <= bound`:

```
# Final value on side 1, N=998 S[14126]=143
# When entering factor_survivors for bucket 998: S[0][14126]=136, S[1][14126]=143
```

So these are pure boundary cases: our norm rounds one unit high, las lands on
the bound, we land one over. This is finding 28's one-unit difference, and at
the boundary it is the *only* thing that matters. **Benign, and quantified:** it
costs us 722 of 795,845 survivors at this q, 0.09%, and relation containment
says none of them carried a relation.

### Side 0: our bytes are *identical* to las's — the difference is the test

The side-0 misses look nothing like side 1. They spread from 142 to 155:

```
our side-0 byte at the missing positions (bound 141):
   142:12  143:5  144:21  145:12  146:11  147:15  148:10
   149:16  150:4   151:9   152:11  153:12  154:1   155:4
```

That spread is far too wide for a rounding effect, so the obvious reading is
that we under-subtract on the rational side. **That reading is wrong.** Two
traced cases, chosen independently:

| `(a, b)` | las: init − adds | las final | **ours** |
|---|---|---:|---:|
| `(-18587524911, 10201)` | 253 − 103 | **150** | **150** |
| `(144147912218, 10443)` | 253 − 107 | **146** | **146** |

**Our side-0 sieve byte equals las's exactly, in both cases.** The norms agree
and the sieving agrees; there is nothing missing from our rational factor base.

What does not agree is what happens next. Both positions have a side-0 value
well above the bound `-v` prints (141), and **las still lists them as
`after_sieve` survivors** — both are in `c183.q120000053.after_sieve_ab.powlim.txt.gz`,
and las's own trace confirms the coordinates (`i = -10201, j = 472` for the
first, matching our mirrored `i = +10201, j = 472`).

CADO's survivor test is unambiguous (`sieve/las-unsieve.cpp:382`):

```cpp
return SS[0][x] <= bound[0] && SS[1][x] <= bound[1];
```

so the `bound[0]` actually handed to `search_survivors_in_line` is **not** the
141 that `-v` reports. **That is the open question, and it is now a narrow one:**
find where `bound[]` is populated for the survivor search and what it holds for
side 0. Everything upstream of it is confirmed equal.

**Why this matters more than the count suggests.** If las's effective side-0
threshold is looser than ours, then our survivor test is *stricter* than las's,
and we discard positions las keeps. Relation containment says none of those
carried a relation in this band (3,026/3,026), so nothing is lost today — but it
is a yield risk that scales with the band, and it points the opposite way from
the "+5.6% survivor excess" the project has been tracking. Both can be true at
once: looser than las on side 1's boundary, stricter than las on side 0.

**Blocked step, and how to unblock it.** `las_tracek` **aborts** (`SIGABRT`) on
both sampled side-0 points before printing its final value — a TRACE_K-only
consistency assert in the unsieving path (`FAILED test_divisible(p=10201, ...)`,
and the `verify_gcd` `ASSERT_ALWAYS` next to it). The per-add trace still prints,
which is what made the table above possible, but closing the question needs
either an assert-relaxed TRACE_K build or simply reading off where `bound[]` is
filled. That is a source question, not a measurement.

### By-product: finding 17 is stronger than recorded

Finding 17 says the `-dumpfile` oracle "is not usable as captured", which reads
as a capture problem. It is not. A **fresh** dump was taken at exactly the
config of the `(a,b)` capture (`-powlim0/1 32767`, `-bkmult 2.0` so no
`redoing ... buckets are full`, matching `survivors after_sieve: 795845`), and
it is still unusable:

| mapping tried | of 862,171 dump positions, found in las's own list |
|---|---:|
| `a0=+7374527, b=-i` (las's printed convention) | 1,930 (0.22%) |
| `a0=-7374527, b=-i` | 0 |
| `a0=+7374527, b=+i` | 0 |
| `a0=-7374527, b=+i` | 1,775 (0.21%) |

The aggregate statistics are right — 862,171 not-both-even two-sided against
las's 795,845 reported — but **the positions do not correspond under any sign
convention, in either direction.** So the dump cannot be indexed by `x` at all,
and `-dumpfile` is unusable *by construction*, not as captured. Restate finding
17 accordingly, and do not spend time on another capture.

For contrast, our own pipeline is internally exact — the same filters applied to
our dump reproduce the bitmap and finding 41 to the digit:

| | raw two-sided | not-both-even | `gcd(i,j)=1` |
|---|---:|---:|---:|
| **ours** | 2,176,787 | **1,386,939** | **841,418** |
| las (unusable positionally) | 1,314,595 | 862,171 | 706,212 |

## Item 3 delivered: device intersect + compact, and it is ~0.4% of the chain

`k_intersect_compact` ANDs the two per-side survivor bitmaps, drops
`gcd(i,j) != 1`, and compacts what remains into a dense list. Driven by
`bench --other-bits <other side's bitmap> [--emit FILE]`.

**It emits `x` as well as `(a,b)`**, which was the one design note attached to
this item. Every downstream stage — resieve under either layout, trial
division, the cofactor queue — is indexed by `x`, and recovering `x` from
`(a,b)` means inverting the lattice basis per survivor. `x` costs 4 bytes.

**Both correctness gates pass.**

| gate | expected | got |
|---|---:|---:|
| two-sided, pre-gcd | 1,386,939 (finding 41) | **1,386,939** |
| primitive `gcd(i,j)=1` | 841,418 (finding 41) | **841,418** |
| las's relations present in the emitted `(a,b)` | 37 | **37 / 37** |

The first two reproduce finding 41 exactly, so the device path agrees with the
host computation bit for bit; the third confirms the `(a,b)` arithmetic, not
just the count. 60.7% of two-sided survivors are primitive.

### Cost: 0.448 ms/q, and two measurement traps on the way

| variant | steady state |
|---|---:|
| **warp-aggregated allocation** | **0.447 ms** |
| one global atomic per non-empty word | 0.567 ms |

**Trap 1 — first-launch cost.** The first timing was **4.6 ms**, and it is not
real: it is module-load cost on the first launch of this kernel. Repeat and the
same kernel runs in 0.449 ms. A 10x error, and the only thing that exposed it
was timing more than once. Anything in this project that reports a kernel from a
single launch is reporting the same artefact — `bench` now takes the best of 3
here for exactly that reason.

**Trap 2 — I predicted the wrong bottleneck.** 841,418 atomics on one counter
looked like the obvious cost, so warp aggregation went in first. It bought 21%,
not the 10x that a serialised-atomic story would predict. At 0.447 ms the kernel
moves 134 MB of bitmap at ~300 GB/s, so it is bandwidth-shaped, not
atomic-shaped. Keep the aggregation — 21% for ten lines is a good trade — but
the lesson is that the atomic count was never the thing to reason about.

**Against the budget:** 0.448 ms on a sieve chain of 53–84 ms is **~0.5%**.
Intersection and compaction are free, which is what the sparsity argument
assumed and now no longer has to assume. The 67 MB-per-side bitmaps are the
real resource here, not the time.

**Caveat on how this is wired.** `bench` sieves one side per process, so the
other side's bitmap arrives from a file and is uploaded (pinned) before the
kernel runs. That upload is *not* counted in the 0.448 ms, and it should not be:
in the production path both bitmaps are already device-resident and no transfer
happens at all. What the number measures is the kernel, which is the part that
carries over.

## Item 5 RESOLVED: layout B was built, and layout A wins

Both paths are implemented in `bench` and run back to back on the same
special-q. **Layout B is built for real** — segmented fill at the 65,536 cap,
per-(bucket, slice) boundary snapshots, and a purge that names the prime via
`primes[starts[slice] + offset]`. It is not a cost proxy.

### Correctness first: B recovers exactly what A does

```
B: recovered (x,p)          1932951  vs A's 1932951   MATCH
A vs B (x,p) set equality  IDENTICAL (0 of 1932951 pairs differ)
```

Two entirely different traversals — one walking the factor base, one streaming
the retained bucket array and reconstructing the prime from segmentation —
produce the **same 1,932,951 `(x, p)` pairs**. Layout B works. It is simply not
worth what it costs.

### The numbers (quiet GPU, q=120000053, side 1)

| stage | A: re-walk | B: segment + purge |
|---|---:|---:|
| fill | **12.25 ms** | **50.5 ms** |
| recovery | **8.55 ms** | **3.26 ms** |
| **fill + recovery** | **20.8 ms** | **53.8 ms** |

**A wins by 2.6x.** B's recovery really is 2.6x cheaper than A's — that part of
the earlier reasoning held — but it buys that by making fill **4.1x more
expensive**, and fill is the larger stage.

### Why segmentation costs so much, and why it cannot be fixed cheaply

The 38 ms is not launch overhead, not the boundary snapshots, and not
per-launch occupancy. All three were ruled out directly:

| variant | fill |
|---|---:|
| 126 launches, with boundary snapshots | 51.2 ms |
| 126 launches, no snapshots | 50.5 ms |
| 126 launches, grid sized per slice | 50.4 ms |

Merging the slices into `G` super-segments — identical total work, only the
number of passes over the bucket array changes — isolates it:

| passes | 1 | 2 | 4 | 8 | 16 | 32 | 63 | 126 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fill (ms) | 12.26 | 13.24 | 14.92 | 17.67 | 21.81 | 26.98 | 35.12 | 50.5 |

**The cost is linear in the number of passes, at ~0.3 ms per pass.** The
mechanism is write amplification on the bucket array. With one cursor per
bucket shared across launches, consecutive slots in a bucket — the same cache
line — are written by *different* passes, so each line is fetched and dirtied
many times instead of once. The monolithic fill interleaves hits from the whole
factor base, so a line fills in one go.

**This is exactly the idea recorded above as "segmentation without a counting
pass", and it is the part that does not work.** Sharing one cursor per bucket is
what makes the boundary table free, and it is also what destroys write
locality. The fix — per-(bucket, slice) sub-allocation, so each pass writes a
contiguous run — requires knowing the counts in advance, i.e. **the counting
pass this design was constructed to avoid.** That pass costs about a full walk,
which is A's entire recovery cost. There is no cheap version.

### And the slice cap makes it worse, for a reason CADO shares

Layout B needs the within-slice offset to fit the record's high 16 bits, so
slices cap at **65,536**, not the 262,144 the current `build_slices` uses. CADO
has the identical constraint and asserts on it (`sieve/fb.hpp:356`:
`ASSERT(size() <= numeric_limits<slice_offset_t>::max())`). For this factor base
that is **126 slices, not 44** — so the earlier "~64 segments" estimate in this
document was wrong on two counts, and the "~120" figure quoted in review was
right. More segments, more passes, more amplification.

### Standing correction to this section's own history

- **The original conclusion — re-walk, no record format change — was right.**
  It is restored on measurement.
- **The factual error that prompted reopening it was real and the correction
  stands.** CADO's shorthint *is* a within-slice offset, CADO *does* recover
  large primes by purging retained buckets rather than resieving, and the
  comment at `bench_kernels.cu:88` claiming our slice-index field matches it is
  still wrong. Being wrong about the mechanism did not make the conclusion wrong.
- **Reopening it was still the right call.** The original text asserted the
  answer with a supporting argument that did not hold; it is now a measurement
  with a mechanism, and B is eliminated on evidence rather than on assertion.
- **My 8-14 ms estimate for A was accurate** — it measures **8.55 ms**. The
  17.7 ms reported earlier in this section was GPU contention, not a bad
  estimate. See the contamination note below.

### Measurement hazard: a busy GPU silently doubled everything

An earlier pass of this A/B was run while another job held the GPU, and
**every** number was inflated roughly 2x — fill 23-28 ms against 12.25, apply 49
against 23.6, A's recovery 17.7 against 8.55. The *ratios* between A and B
survived, which is exactly why it was not obvious, but the conclusion did not:
that pass reported "layout B wins" and this one, on a quiet GPU, reports the
opposite by 2.6x. Nothing in the output flags contention. **Check GPU
utilisation before trusting any timing in this document**, and prefer
same-run comparisons to cross-run ones.

### The chain as it now stands (quiet GPU, side 1)

| stage | ms |
|---|---:|
| transform + plattice | 2.42 |
| fill | 12.25 |
| apply | 23.62 |
| intersect + compact + primitive filter | 0.46 |
| build survivor filter | 0.11 |
| recovery (layout A) | 8.55 |
| host per-q | 0.84 |
| **total** | **~48.3** |

**apply is now the largest single stage at 23.6 ms — 49% of the chain**, and it
is the one stage that has not been revisited since prime powers were added.
Recovery, the feared unknown, is 8.55 ms. Item 5b (the small/large split sweep)
is untouched and belongs inside layout A.

## Cofactorization harnessed: the baseline runs, our data does not feed yet

`~/code/nfs_3lp_batch_factor` (bbuhrow's 3LP batch cofactorizer) was built and
run on this box. Two results: a **working same-machine baseline**, and a precise
statement of why the C183 capture cannot be fed to it yet.

### Getting it to run at all: a real bug on sm_120

`gpu_cofactorization.c:1567` selects the PTX by compute-capability major, and
Blackwell consumer reports **major 12**, which fell into the `>= 9` branch and
loaded the *shipped* `cuda_ecm90.ptx`. That file predates `cuda_ecm64.cu` and
lacks `gbl_pm196`, so startup died with `CUDA_ERROR_NOT_FOUND` — even though the
Makefile had just built a correct `cuda_ecm120.ptx`. A `major >= 12` branch ahead
of the `>= 9` one fixes it. **Worth sending upstream**: any RTX 50-series user
hits this, and the failure looks like a missing kernel rather than a stale file.

Build settings that worked: `TOOLKIT_VERSION=12`, `CUDA_PATH=/usr/local/cuda-12.8`
(12.8 is the first toolkit with `compute_120`), `SM=120`, GMP from
`~/gmp-zen3`, `make all ICELAKE=1`.

### The baseline, and the number that matters is not the one advertised

1,000,000 bundled GGNFS relations, `-m 0 -b1 205 -b2 50 -c 100 -s 10`:

| | |
|---|---:|
| wall time | **73.18 s** |
| relations produced | 10,257 |
| **sum of all reported GPU kernel times** | **~2.19 s** |

| kernel | ms |
|---|---:|
| 64-bit ECM, r-side 2LP | 75.6 |
| 96-bit P-1 | 386.0 |
| 96-bit ECM (7.86M curves) | 1700.6 |
| 2LP retests | 24.2 |

**The GPU is busy for 3% of the run.** The other ~71 s is host work — batch
setup, validation, primality checks, MPQS tails, output sorting. That is 73 us
of CPU per relation, and at 1,939 records/q it is **~142 ms of host time per
special-q**, against a GPU sieve chain of ~48 ms.

This does not invalidate finding 47's ~4.92 ms/q sizing datum — that datum is
plainly a *GPU-kernel* number, and the kernel times here are consistent with it
(2.19 s for 1M records is 2.2 us/record, ~4.3 ms/q). **What it says is that the
kernel number was never the whole cost**, and the gap is exactly the category
Goal 1 is about. Any deployment that keeps this harness as-is is host-bound, not
GPU-bound, by a factor of 33.

### Why the C183 capture does not feed, stated precisely

The `res1,res2` sign convention was inferred from the reference file rather than
guessed, by histogramming 400,000 of its relations:

| field | sign | sizes present |
|---|---|---|
| `res1` (rational) | positive, 63.2% | value 1, or 24–31 bits |
| `res1` | negative, 36.8% | **48–63 bits only** |
| `res2` (algebraic) | negative, **100%** | 72–95 bits |

So negative means "still needs splitting", which is the rule the adapter used
(`negative iff > 2^lpb`) and it is correct. The mismatch is elsewhere:
**the reference cofactors are already classified, ours are raw.**

A GGNFS siever emits `rels.raw` *after* deciding which cofactors are worth
cofactorizing — negative `res1` values occupy a narrow 48–63 bit band, never
32–47. CADO's `-batch-print-survivors` emits the cofactor as it stands after
trial division, with no primality test and no viability filter. Our rational
cofactors therefore spread across 32–62 bits, and a large share of the 32–47 bit
ones are **single primes above 2^31** — unusable by construction, which CADO
itself would discard later.

The consequence is visible in the run:

| | reference | C183 capture |
|---|---:|---:|
| r-side 2LP inputs | 367,684 | 544,987 |
| **valid 2LP factorizations** | **270,984 (73.7%)** | 379,974 (69.7%) |
| restricted to both-sides-need-work | — | **11,328 of 540,157 (2.1%)** |

and in a hard failure: 34.4% of the 96-bit moduli arrive zero or even, which is
fatal (`mpz_tdiv_r` on a zero modulus raises SIGFPE — the run core-dumped before
a guard was added at `gpu_cofactorization.c:1210`).

### What this actually costs to close

Not "a small input adapter", which is how this document previously described it.
The adapter has to reproduce the classification a siever does before calling
cofactorization: **primality-test each cofactor, drop those that are prime and
above `lpb`, drop those that cannot split within `mfb`, and only then emit.**
That is real work but it is well-defined, and it is work the GPU siever will
have to do anyway — so it belongs in our pipeline, not in a throwaway adapter.

**Until that exists, no C183 `31/32` cofactorization throughput number should be
quoted from this harness**, including any derived from the run above. The
stage-mix projection (~5.5 ms/q, ~11.7x headroom) remains a projection, and it
now has a second caveat: it models GPU kernel time only, and the measured host
overhead around those kernels is 33x larger.

### Standing item

The 34.4% bad-modulus rate is not explained by the classification gap alone —
`gpu_cofactorization.c:1454` fills the 96-bit array indexed by *relation* index
while `residues_a_in` is populated on the 2LP path, and our workload has 48.7%
of relations skipping that path entirely. That is a plausible aliasing bug under
a workload shape the code was not written for, and it is the first thing to
check once the classification adapter exists.

# Build log — Opus, 2026-08-04 (implementation, not review)

## Trial division is built and it agrees with CADO exactly, on both sides

`bench --td` computes the exact integer norm for every survivor, divides out
the special-q, the large primes recovered by the layout-A re-walk, and the
small primes found by direct test, and emits the residual cofactor.
`--cofgate` checks that residual against
`oracle/c183.q120000053.cofac_candidates.txt`, which holds CADO's own
post-trial-division cofactor for each of the 1,851 positions it carried into
cofactoring.

| side | reference records | matched to a survivor | **cofactors identical** |
|---|---:|---:|---:|
| 1 (algebraic) | 1,851 | 1,851 | **1,851 — PASS** |
| 0 (rational) | 1,851 | 1,851 | **1,851 — PASS** |

This is an exact agreement on 96-bit integers, not a count, and it gates the
whole chain at once: the exact norms, the rank map, the resieve scatter, the
small-prime congruence test, and the multiplicity handling. It is also the
first point at which this project has produced a *factorisation* rather than a
survivor.

Implementation notes worth carrying:

- **256-bit sign-magnitude big integers** (`bigint.cuh`), host and device from
  one header so the reference and the kernel run the same arithmetic. Measured
  on the real lattice: |a| 41 bits, |b| 15, |F| 224, |G| 132 — so 256 leaves
  ~28 bits of headroom, and `bn_mul_u64` reports overflow rather than wrapping.
- **The rational side is not a special case.** `G(a,b) = Y1*a + Y0*b` is the
  degree-1 member of the same homogeneous family, so one norm kernel serves
  both sides.
- **Exact coefficients had to be added to `poly_t`.** c0 is 147 bits on this
  job; the existing `double` is good for 53 of them. Norm-init takes a
  logarithm and never noticed.
- **The compaction is now rank-ordered** (prefix sum over the two-sided
  bitmap). That is what lets the resieve scatter a recovered prime straight
  into its survivor's slot — no sort of 1.9M (x,p) pairs, no 2 GB
  position-indexed map — and it makes the emitted list reproducible run to run.
- **Proper prime powers are skipped in both directions.** `fb_restrict`
  already keeps them out of the bucketed factor base, and the small table
  filters them out, because multiplicity is recovered by repeated division by
  the base prime. Dividing by a recorded `p^2` as though it were prime is
  wrong whenever the true multiplicity is odd.

## The special-q factor-base truncation costs relations, and the gate found it

The first gate run failed 30 of 1,781 — and **every one of the 30 differed from
CADO's cofactor by exactly one prime factor in (q, alim]**: 121716281,
123120331, 133854251, 133849943, 128998913, 125436139, 123160313, all prime,
all 27 bits. A further 70 of CADO's records were missing from our survivor list
entirely.

The cause is `--fbbound 120000053`, the GGNFS convention of truncating the
algebraic factor base at the special-q, which the canonical profile in
RESULTS.md uses. CADO sieves to `alim = 134200000`. Rerunning at
`--fbbound 134200000`:

| | truncated at q | full to alim |
|---|---:|---:|
| CADO records absent from our survivors | 70 | **0** |
| cofactors identical | 1,751 / 1,781 | **1,851 / 1,851** |
| two-sided primitive survivors | 818,840 | **841,418** (finding 41) |

**Both failures had the same single cause**, which is worth stating plainly:
the missing 70 were not a sieve defect either — the extra factor base lowers
those norms enough to bring the positions under the bound. **Any parity
comparison against CADO must use `--fbbound 134200000`.** The GGNFS truncation
is a legitimate configuration, but it is a different one, and it costs
relations rather than merely costing cofactor size.

## The survivor bound is 131 on side 0, not 128 — and item 4 is reopened

`prototype.md` records item 4 as **closed at bound 128**: "the true zero-loss
floor is 128, not 131 ... Hard-code 128 ... zero relations lost." Run against
our own sieve at q=120000053, **bound 128 loses one of las's 37 relations.**

| bounds (side 1 / side 0) | survivors | las's 37 recovered |
|---|---:|---:|
| 143 / 141 (the loose profile) | 841,418 | 37 |
| **128 / 128** | 202,670 | **36 — one lost** |
| **128 / 132** | **239,446** | **37** |

The lost relation is `(a,b) = (-372574502414, 14251)`, lattice `(i,j) =
(-14251, 2229)`. Side 1 keeps it at 128; **side 0 drops it**, and `--probe`
gives the reason directly:

```
[gate 5] probe (i=-14251, j=2229)  x=73042005
         init norm S   = 253
         SIEVED LOG SUM = 122
         las byte S-sum = 131
```

**131, against a residual cofactor of 58 bits that implies 1.925 x 58 = 112.**
A 20-unit gap between what the sieve credits and what trial division actually
removes, on a position that carries a real relation.

### Why the gap exists, and why it is not a bug

The factor base contains prime powers only while `p^k < 2^maxbits = 2^15`. The
sieve therefore credits a prime with **at most** the multiplicity its ladder
reaches — `2^14` is the last power of two in the table — while trial division
divides out the **full** multiplicity. Any position whose norm carries a high
power of a small prime is under-credited by the sieve by exactly the excess.

las has the same `maxbits`/`powlim` structure and the same behaviour, so this
is not a defect. What it means is that **the sieve byte is an upper bound on
`scale * log2(cofactor)`, not an estimate of it**, and positions with heavy
prime-power multiplicity sit well above where their cofactor size would put
them. A survivor bound tuned on the cofactor distribution will cut exactly
those positions, and they are not rare enough to ignore — one in 37 relations
at this q.

### What to use, and what to re-run

**Use 128 on side 1 and 132 on side 0** until a band-wide sweep says otherwise.
That recovers 37 of 37 here and still cuts survivors 3.5x against the loose
profile. The recorded "128 both sides" figure came from a sweep of *CADO's* `-v`
bound over CADO's own bytes; the correction is that **our bytes are not las's
bytes at these positions**, which the earlier review had already half-found
from the other direction — its note that las's effective `bound[0]` is looser
than the 141 it prints is the same phenomenon seen from las's side.

**The band-wide bound sweep should be redone on our own sieve**, over the 67
lattices and 3,026 relations the containment gate already uses, rather than
inherited from CADO's sweep. One q is one q; it establishes that 128 is wrong
on side 0, not that 132 is right.

## The post-sieve chain, measured on a quiet GPU

At the working bounds (side 1: 128, side 0: 132), 239,446 two-sided primitive
survivors, best-of-3 per kernel:

| stage | side 1 | side 0 | notes |
|---|---:|---:|---|
| transform + plattice | 2.76 | 1.74 | |
| fill | 12.28 | 11.92 | |
| apply | 23.61 | 15.03 | |
| **sieve subtotal** | **38.06** | **26.94** | |
| intersect + gcd + compact | 0.42 | — | shared, once |
| rank scan + emit + filter | 0.46 | — | shared, once |
| resieve + scatter | 6.68 | 6.00 | |
| norms + trial division | 2.80 | 2.54 | |
| classify | 1.01 | 0.70 | |
| host per-q | 0.54 | 0.55 | |

**Post-sieve total: 21.7 ms/q** of device kernels and per-q host tables,
everything except cofactorisation. (Measured post-sieve *wall clock* over a
348-q band is 27.9 ms/q; see the note under "Against finding 46".)

### Against finding 46

| goal | post-sieve budget | where we are |
|---|---:|---|
| throughput parity | ~220 ms | not binding |
| energy parity | ~106–114 ms | **comfortable, 4x of headroom** |
| **2x energy win** | **~21–25 ms** | **27.9 ms before cofactorisation** |

*(Superseded below: with cofactorisation measured, post-sieve is 41.4 ms/q.)*

*(The 27.9 ms is the 348-q band's measured post-sieve wall clock. This section
originally read 21.7 ms, which was the sum of device kernels and per-q host
table building only, and did not include the intersect, the launch and readback
overhead around the TD kernels, or the host join and file emission. See "Where
the per-q wall time went" below.)*

So the answer to finding 46's question — "where in the 21–114 ms band does the
post-sieve path land" — is **27.9 ms plus cofactorisation**. Energy parity is
not in doubt. The 2x win is out of reach as things stand: finding 47's ~4.9 ms/q
of GPU cofactor kernel time puts the total near 33 ms against a 21–25 ms
allowance, and that ignores the 33x host overhead the harness work measured
around those kernels.

**The lever is resieve + scatter at 12.7 ms, 58% of the post-sieve chain.** It
is now the largest post-sieve stage by a factor of two, and unlike trial
division it does *not* shrink with the survivor bound — it walks the whole
factor base and filters, so its cost is set by the update volume, not by the
survivor count. Item 5b (the small/large split sweep) is the untried lever
that acts directly on it.

### Two estimates settled, one of my own claims withdrawn

**The doc's trial-division estimate was right and my contended measurement of
it was wrong.** I recorded earlier in this log that the small-prime direct test
"does not hold up" against the doc's "~1 ms at peak and plausibly 3–5 ms
achieved". On a quiet GPU it is **1.33 ms (side 1) and 1.49 ms (side 0)** at
the working bound, and 5.6 ms at the loose bound with 3.5x the survivors —
inside the predicted range at both. The 10–11 ms I measured was GPU contention,
and the doc's own standing warning ("**Check GPU utilisation before trusting
any timing in this document**") is there precisely because this keeps
happening. It caught me despite being written down.

**The division finding survives the re-measurement and is the real one.**
Replacing `u64/u32` with a Barrett reciprocal took the division phase from 29.9
to 11.8 ms contended, and it stands at 0.85–0.96 ms quiet. nvcc expands a
64-bit division by a runtime divisor into a ~100-instruction software routine;
trial division ran eight per pass, three passes per factor. Anything in this
project doing that is paying the same toll.

**Two hypotheses that were wrong, recorded so they are not retried:** staging
the small-prime table in shared memory changed nothing (the reads are
warp-uniform and already broadcast), and buffering hits to remove warp
divergence changed nothing. The cost was arithmetic, in the divisions.

## Resieve optimised: 12.68 -> 9.85 ms, and the 2x budget is now met

The resieve walk was 58% of the post-sieve chain and the obvious lever. Three
hypotheses, measured in this order:

| hypothesis | change | result |
|---|---|---|
| L2 **bandwidth** on the summary probe | add a coarse L1-resident pre-filter | **worse at every granularity** |
| the summary table not being **cache-resident** | replace it, 1 MB -> 64 KB | **flat: 5.21 -> 5.35 ms** |
| **latency**: one probe in flight per warp | unroll the walk, probe ahead | **6.92 -> 5.21 ms** |

**The failed experiments are what identified the bottleneck**, so they are
worth more than the successful one. The coarse pre-filter at 1 bit per 256
positions rejected **96.5% of walk steps** before they reached the summary, and
the kernel got *slower* (6.69 -> 7.82 ms). If the probes were a throughput
cost, removing 96.5% of them could not possibly lose. And sweeping the summary
from 1 MB down to 64 KB — a 16x change in table size, straddling any plausible
L1/L2 boundary — moved the time by 2.7%. Neither the number of probes nor where
they land is what this kernel spends its time on.

What it spends its time on is **one dependent scattered load per step, with
nothing else in flight**. It already runs at full occupancy (40 registers, 48
warps/SM, no spills), so the fix is not more warps but more loads per warp:
walk `UNROLL` positions ahead — pure register arithmetic, no speculation
needed — and issue all `UNROLL` summary probes before branching on any of them.

| unroll | 1 | 2 | **4** | 8 |
|---|---:|---:|---:|---:|
| resieve + scatter (side 1) | 6.92 | 5.76 | **5.21** | 5.98 |

4 is the optimum; 8 regresses. Recovered prime counts are bit-identical at
every unroll depth and every granularity — the sweep checks that at each
setting and fails the run if it changes.

### The post-sieve chain as it now stands

| stage | side 1 | side 0 | total |
|---|---:|---:|---:|
| intersect + gcd + compact | 0.42 | shared | 0.42 |
| rank scan + emit + filter | 0.46 | shared | 0.46 |
| resieve + scatter | **5.20** | **4.65** | **9.85** |
| norms + trial division | 2.78 | 2.50 | 5.29 |
| classify | 1.02 | 0.70 | 1.71 |
| host per-q | 0.65 | 0.75 | 1.40 |
| | | | **19.13** |

**19.1 ms against finding 46's 21–25 ms allowance for a 2x energy win.** With
finding 47's ~4.9 ms/q of GPU cofactor kernel the total is ~24.0 ms, which is
**inside the band** rather than 10% over it as the pre-optimisation chain was.
That is a change of verdict, and it rests on one 25% improvement to one kernel,
so it should be treated as *marginally* inside until cofactorisation is
measured rather than projected — the harness work already showed 33x of host
overhead sitting around those kernels, and none of that is in the 4.9 ms.

**Update, after the band ran: this table is device kernels plus per-q host
table building, and that is not the whole post-sieve chain.** The 348-q band
measures the same kernels at **18.53 ms/q**, confirming the 19.13 above, but
post-sieve *wall clock* is **27.93 ms/q** once the intersect (0.41), the
readback and launch overhead around the TD kernels (2.17), and the host join
and file emission (3.00) are counted. Energy is wall clock times power, so
27.93 is the number finding 46's allowance has to be read against, and it is
**above** the 21-25 ms band for a 2x energy win, not marginally inside it.
Adding finding 47's projected ~4.9 ms puts it near 33 ms. Energy parity
(~106-114 ms) remains comfortable. Roughly 3 ms/q of that is writing 1,956
candidate records to a file per q, which the device-resident cofactor queue is
supposed to remove rather than optimise; that is a reason to expect the number
to come down, not a reason to quote a lower one now.

All gates still pass at these settings: 1,850 of 1,850 cofactors identical to
CADO on both sides, factors x cofactor == norm on 32,982 and 14,141 candidates,
1,851 joint candidates, and **37 of 37 of las's relations recovered.**

## Cofactor classification: 1,852 candidates against CADO's 1,851

`k_classify` implements CADO's `check_leftover_norm`
(`sieve/las-cofactor.cpp:118`) verbatim rather than from memory — the mfb cut,
the "too few factors possible" gap test `L^k < n < B^(k+1)`, and a base-2
strong probable prime test. CADO's own comment justifies base 2 alone: calling
a composite prime loses a relation, it never emits a relation with a composite
ideal, so the failure mode is a miss at ~2^-40, not a wrong answer.

Per side, over all 841,418 two-sided primitive survivors:

| outcome | side 1 | side 0 |
|---|---:|---:|
| rejected, more than mfb bits | 674,774 (80.2%) | 555,712 (66.0%) |
| rejected, too few factors possible | 55,797 (6.6%) | 132,442 (15.7%) |
| rejected, probable prime above 2^lpb | 63,296 (7.5%) | 118,578 (14.1%) |
| accepted for cofactorisation | 47,481 | 30,651 |
| already fully split | 70 | 4,035 |
| **this side's candidates** | **47,550** | **34,685** |

A candidate needs BOTH sides, and the joint result is the gate:

| | |
|---|---:|
| our joint candidates | **1,852** |
| CADO's candidates at this q | 1,851 |
| **CADO candidates we drop** | **0** |
| ours CADO does not have | 1 |
| cofactor mismatches on the shared set | **0** |

**And all 37 of las's relations at this q survive the whole chain** — sieve,
intersect, gcd filter, trial division, classification. That is every stage
except cofactorisation itself, gated end to end on the one q where we have
las's own answer.

### The two discrepancies, both understood

The first run produced 1,853, and the two extras had different causes.

**`(a,b) = (120000053, 0)` is the lattice point `(i,j) = (0,1)`.** With this
basis `b = i`, so `i = 0` gives `b = 0` and `a = q`. It passes `gcd(i,j) = 1`,
passes both norm bounds, and its cofactors classify cleanly — `q` on side 0,
which is under `2^31`, and 1 on side 1. **But `(a, 0)` is not a relation at any
cofactor size.** las carries it through `after_sieve` too and drops it later.
It is now rejected explicitly (`COF_DEGENERATE`); leaving it to be caught
downstream would have meant one fake candidate per special-q forever.

**`(-1657184928927, 16384)` is at `i = -I/2`, the extreme edge, and is NOT in
las's `after_sieve` list at all.** Its cofactors are 59 and 90 bits, both
inside mfb, so it is a legitimate candidate that las never tried. This is the
documented +5.6% survivor excess showing up as a *bonus* rather than a cost:
our norms are exact where las's are approximate, and the position las discarded
may well carry a relation. One extra candidate per q against 1,851 is 0.05%.

## The first relation

`(a, b) = (-138913230134, 433)`, at q=120000053, produced by the GPU pipeline
with no cofactorisation at all — both residual cofactors came out of trial
division equal to 1:

```
rational   G(a,b) = 3 . 11 . 53 . 17497 . 74413 . 87443 . 2198473 . 13555187 . 22065019
algebraic  F(a,b) = 2^3 . 5^2 . 7 . 97 . 191 . 211 . 353 . 5821 . 10753 . 245881
                    . 776563 . 1353809 . 9157051 . 92871839 . 120000053
```

Both products were re-derived independently in Python from `(a,b)` and the
polynomial and match the norms exactly; every factor is inside its side's large
prime bound; and the special-q appears in the algebraic factorisation where it
must. **It is one of las's 37 relations at this q.**

That is the end-to-end existence proof this project did not have. The other 36
of las's 37 are all present as candidates with correct partial factorisations —
they need the cofactor stage to finish, which is the next item and is not built
yet.

### The factor list is gated, not just the cofactor

Recording factors during trial division (multiplicity as repetition) allows a
reconstruction gate that the CADO cofactor comparison cannot give: **the
recorded factors times the residual cofactor must rebuild the exact norm.**

| | side 1 | side 0 |
|---|---:|---:|
| candidates checked | 47,550 | 34,685 |
| **factors x cofactor == norm** | **47,550 PASS** | **34,685 PASS** |

and independently in Python over the 1,852 joint candidates, both sides at
once: **1,852 of 1,852**. The CADO gate validates the residual; this validates
the list that produced it. Both were needed — a wrong factor list with a
compensating residual would pass the first and fail the second.

## Cofactorisation: the adapter works, the harness does not yet

`bench/mkcofbatch` joins the two per-side `--emit-cof` files into
`nfs_3lp_batch_factor` input (`res1,res2:a,b:rfac_hex:afac_hex`, negative
meaning "still needs splitting"). **The classification blocker is cleared** —
this is the piece the 2026-08-04 harness work said was not "a small input
adapter", and it is now a 100-line join because `k_classify` does the real
work on the GPU. At q=120000053 it emits 1,844 batch records and 7 already
complete relations, which is the full 1,851 candidate set.

The harness then runs and produces **23 relations from 526,140 records over 548
special-q** = 0.042 relations/q. The batch was filtered to the `r=2,a=3` class
(see below), which at the parity q holds 8 of the 30 batch-visible relations,
so the matched expectation is ~8/q and the shortfall is **191x**. Quoting it
against las's 37/q would give 882x and against all batch-visible relations
715x, but neither is the population that was fed. **No cofactorisation cost
from this run should be quoted**, in either direction. An untimed pipeline that does not produce relations tells us
nothing about the cost of one that does. What the run did establish:

### The predicted aliasing bug is confirmed, and it is worse than recorded

The standing item above says `gpu_cofactorization.c:1454` "fills the 96-bit
array indexed by *relation* index while `residues_a_in` is populated on the 2LP
path". Confirmed, and there are **two** index errors, not one:

```c
t->modulus96_in[3*j + 0] = t->residues_a_in[3*i + 0];   /* line 1454 */
```
`i` runs over `t->rb->num_relations`, but `residues_a_in` is packed at
`residues_a_in[ka++]` (line 1757) **densely and with a variable stride** — the
inner loop runs `c->lp_a_num_words` times, which is 1, 2 or 3. So the read is
wrong in its index space *and* in its stride. Both are invisible on the bundled
reference workload, where every relation has a 3-word a-side residue and takes
the 2LP path, making the dense index equal the relation index and the stride
always 3.

**The classification gap was not the cause**, which settles the open question:
the bad-modulus rate is **33.59%** with correct classification against 34.4%
before, essentially unchanged.

### Second upstream bug: an 80-byte path buffer

`main.c:422` is `char outfile[80]` followed by
`sprintf(outfile, "%s.cu.out", infile)`. Any input path over 72 characters
overflows it and glibc aborts the process. The same unguarded copy of a
user-supplied path into an 80-byte buffer is at `main.c:695`
(`strcpy(fname, options->file)`). The buffer at `main.c:699` is a different,
shadowed one used for the fixed `bgcd_lpb%d` name and is NOT affected. Worth
sending upstream with the PTX fix.

### The finding that actually matters for how this gets fed

Restricting the batch to the shape the harness handles (r-side 2 words, a-side
3 words) makes the aliasing bug harmless — but it throws away most of the
relations. Where las's 30 batch-visible relations at q=120000053 actually live:

| class (r words, a words) | batch records | las relations | density |
|---|---:|---:|---:|
| r=2 a=3 (the shape the harness handles) | 948 | 8 | 0.8% |
| r=0 a=3 | 856 | 6 | 0.7% |
| **r=2 a=2** | **21** | **9** | **43%** |
| **r=0 a=2** | **16** | **4** | **25%** |
| **r=2 a=0** | **3** | **3** | **100%** |

**16 of 30 relations come from 40 records.** The classes where one side is
already finished or small are enormously relation-dense, and they are exactly
the classes the harness cannot process. Feeding it only the `r=2,a=3` class
processes **51.4% of records (948/1,844) to keep 26.7% of the yield (8/30)**.
Taking both `a=3` classes processes **97.8% of records for 46.7% of the yield**
— so the a=3 population is 98% of the records and less than half the relations,
while 40 records carry the other half.

**So the design consequence is not "fix the adapter".** It is that a cofactor
stage for this workload has to be organised by residue class, cheapest and
densest first, and the **2-word-by-3-word** case — the expensive one — is the
least productive per record. (It cannot be 3-word by 3-word: `mfbr = 60`, so a
rational residual never exceeds two words on this job.) That is a scheduling statement about our own queue
(item 7), and it holds regardless of whose ECM eventually runs underneath.

### The rho iteration cap was my error, and it matters

An earlier draft of the plan for our own cofactor stage said that a record not
split within ~2^17 Brent-rho iterations is "provably dead" and can be dropped.
**That is wrong.** Pollard rho and Brent rho are Monte Carlo: expected work is
O(sqrt p) for a factor p, and no unsuccessful finite run establishes that no
admissible factor exists. Implemented as stated it would have silently
discarded relations, with nothing in any gate able to see it — the CADO
cofactor gate and the reconstruction gate both run *upstream* of splitting.

The bounded-work property worth having is real, but it is a **scheduling**
statement, not a rejection proof: 2^17 iterations is one service slice, and on
exhaustion the task must reseed with a different polynomial, requeue with an
attempt count, or escalate to a bounded ECM tier. ECM is bounded the same way,
by finite B1/B2 and a curve count — which is exactly what the external harness
does at `gpu_cofactorization.c:1804`. Neither algorithm turns an exhausted
budget into a proof.

### Where that leaves the budget

Finding 47's ~4.92 ms/q remains **a projection**, now with a third caveat: the
harness that produced its sizing datum cannot process 73% of this job's
candidates without the index fix, and the yield it currently achieves is 191x
short of the class it was fed. The post-sieve chain is measured -- **18.5 ms/q
of device kernels, 27.9 ms/q of wall clock** over a 348-q band; the cofactor
term added to it is not.

## Both sides in one process

`bench --pipeline` sieves side 1 and side 0, keeps both survivor bitmaps
device-resident, intersects once, trial-divides and classifies each side
against the shared two-sided bitmap, joins in memory, and writes complete
relations and the cofactorisation batch directly.

**Its output is byte-identical to the two-process path.** Same 9,521,087 and
16,969,340 one-sided survivors, same 239,446 two-sided primitive survivors,
same 1,844 candidates and 7 complete relations, and `diff` on both output files
against `mkcofbatch`'s is empty. That is the gate for the merge: it is the same
computation, not a reimplementation that happens to agree on counts.

| | |
|---|---:|
| complete relations, products == norms | **7/7** |
| complete relations present in las's 37 | **7/7** |
| candidates + relations | **1,851** (CADO's count at this q) |
| **las's 37 covered** | **37/37** |

Three things had to be got right and are worth recording, because each was a
silent failure rather than a crash:

- **Side 0's norms are a degree-1 form.** `run_bench` obtains that by mutating
  `POLY` in its caller, which the pipeline must not do — `run_td_stage` still
  needs the degree-5 coefficients for side 1's exact norms. The pipeline makes
  a local degree-1 copy for `norm_setup` and leaves `POLY` alone.
- **The factor order was not reproducible.** Large primes reach a survivor's
  slot through an `atomicAdd`, so their order in the emitted factorisation
  varied run to run. The two paths agreed on every multiset and differed on two
  lines by a transposition. Factor lists are now sorted at emission, which is
  what makes a `diff` between two paths meaningful at all.
- **The bucket array is shared between sides.** At 1.38 GB it is the largest
  allocation in the process and nothing needs it after apply, so side 0 reuses
  side 1's rather than doubling the footprint.

### The band: `--qlist`, and the first relations produced at scale

`bench --pipeline --qlist FILE` runs a band of special-q in one process, with
everything q-independent hoisted out of the loop (factor base upload, slice
tables, pinned staging, all device buffers for the sieve). Over the first 8 q
of the captured band:

| | |
|---|---:|
| two-sided primitive survivors / q | 238,820 |
| cofactorisation candidates / q | 1,923 |
| **complete relations / q** | **8.0** |
| relations emitted over 8 q | **64** |
| **factors x cofactor == norm, all primes within lpb** | **64 / 64** |
| also found by las | 63 / 64 |

These are relations produced end to end by the GPU pipeline with no
cofactorisation stage at all -- the 8 per q that fall out of trial division
alone. Every one was re-derived independently from `(a,b)` and the polynomial.

**One of the 64 is a relation las did not find:**

```
(a, b) = (-2282211070036, 15437)
rational : 3 . 7 . 7589 . 40487 . 91529 . 159079 . 591959 . 4099723 . 20472997
algebraic: 2^4 . 5 . 89 . 163 . 383 . 457 . 1987 . 78691 . 175859 . 420557
           . 1395181 . 4123793 . 29617799 . 120000107 . 349460389
```

It is valid: both products equal the norms exactly and every prime is inside
its side's bound. This is the +5.6% survivor excess appearing as a *bonus* for
the second time -- our norms are exact where las's are approximate, so
positions las discards can still carry relations. It is one in 64 here, which
is a yield effect worth quantifying over a longer band rather than a curiosity.

### Where the per-q wall time went -- measured, after the harness was removed

The first version of this section reported **316.2 ms/q**, of which **241.2 ms**
was "TD + classify, wall" against ~19 ms of device kernels, and attributed the
gap to `cudaMalloc`/`cudaFree` and readback. **That was asserted, not measured,
and it was wrong.** The pipeline was calling `run_td_stage`, which is the
*benchmark* entry point: it runs every stage best-of-three plus two diagnostic
variants -- rank x3, emit x3, resieve x3 plus a restoring pass, trial division
x3 with no small primes, x3 with no division, x3 ordinary, one dense recording
pass, classify x3 -- **per side**, and it rebuilt the rank scan, the emission
and the resieve summary for the second side even though all three are functions
of the shared two-sided bitmap alone.

`pipe_td_perq` is the same arithmetic with the harness taken out. Four changes:
the three shared stages run once per q rather than once per side; every buffer
is allocated once for the whole band; the two sides' acceptances are intersected
**on device** and the recording pass runs over that compacted list rather than
over every survivor; and the reconstruction gate is a separate phase whose cost
is reported on its own.

The third of those is the one with real mass. Recording is the only pass that
writes a 64-word factor list per thread. Over 239,446 survivors that is a
**61 MB** matrix per side to allocate, fill and read back; over the 1,844 joint
candidates it is **486 KB**. The joint acceptance is computed by a prefix scan
over the accept flags rather than an atomic, so the compacted order is
deterministic and the emitted batch stays byte-reproducible -- which is the
property that makes diffing two paths a test at all.

**Band of 348 special-q, one process, 35.2 s wall:**

| stage | ms/q |
|---|---:|
| sieve, both sides | 68.77 |
| intersect + gcd | 0.41 |
| host per-q (sieve tables, staging) | 1.14 |
| **TD + classify, wall** | **20.70** |
| join and emit | 3.00 |
| unaccounted | 2.67 |
| **wall clock per q** | **96.70** |

and inside that 20.70 ms, on the device:

| TD stage | ms/q |
|---|---:|
| rank scan (shared) | 0.212 |
| emit (x,a,b) in rank order (shared) | 0.184 |
| survivor filter (shared) | 0.279 |
| resieve + scatter, both sides | 9.902 |
| norms + trial division, both sides | 5.464 |
| classify, both sides | 1.737 |
| joint accept + compact | 0.023 |
| record candidate factorisations | 0.731 |
| **= device total** | **18.53** |
| host: small-prime tables | 0.819 |
| host: readback of candidates | 0.404 |
| host: unaccounted | 0.949 |

**241.2 ms/q becomes 20.70 ms/q, and 18.53 of it is device kernels.** The
prediction that section made from the individual kernel measurements -- "~19
ms/q across both sides" -- was right; what was wrong was the account of the
other 222 ms. It was the harness, and it is gone.

The output is **byte-identical** to the previous path at every scale checked:
the parity q's 1,844 candidates and 7 relations, and the whole 8-q band's
15,382 candidates and 64 relations, `diff`-clean against the files the
`run_td_stage` pipeline wrote.

**Post-sieve is now 27.93 ms/q** (96.70 minus the sieve), against finding 46's
~106-114 ms for energy parity and ~21-25 ms for a 2x energy win -- with no
cofactorisation stage in it yet. The sieve, at 68.77 ms/q, is now 71% of wall
clock, which is where it should be and was not before.

**The lesson is the recurring one.** This was the second gap in this log that I
explained before instrumenting it -- the first was the small-prime direct test,
where a contended GPU produced a 10 ms figure I called a model failure and a
quiet GPU put it inside the predicted range. Both times the honest move was a
timer, and both times the explanation arrived first. The timer has now been
taken, and it agreed with none of the three contributors the earlier draft
guessed at in the proportions it guessed.

### The band at 348 special-q

| | |
|---|---:|
| two-sided primitive survivors / q | 238,786 |
| cofactorisation candidates / q | 1,956 |
| **complete relations / q** | **8.43** |
| relations emitted over 348 q | **2,933** |
| cofactorisation candidates emitted | **680,695** |
| wall clock, whole band | **35.2 s** |

Every one of the 2,933 relations and all 680,695 candidates were re-derived
independently from `(a,b)` and the polynomial: the recorded factors times the
residual cofactor rebuild both exact norms, and every prime in a relation is
inside its side's large-prime bound. **2,933/2,933 and 680,695/680,695.** The
CADO cofactor gate still passes at the parity q on both sides (1,850 of the
1,850 reference records that are in our survivor list).

### A thousand special-q, and where the q-list comes from

**No root-finding is needed, and none was written.** A special-q below `alim`
*is* an algebraic factor-base prime, and CADO's `makefb` output already lists
every root of f mod it -- so the band is a window of the factor base the run
uploads anyway. `--qrange MIN:MAX` reads it there, skipping proper prime powers
and projective roots. It reproduces the hand-extracted `band.qlist` exactly, and
it is worth noting that the captured list held only the **first** root of each
q: the factor base has 543 (q, rho) pairs in the 120000000-120010000 window
where the file had 348. Every root is a separate special-q and all of them are
now enumerated.

**1,000 special-q, one process, 96.5 s wall (95.64 ms/q):**

| | |
|---|---:|
| special-q processed | 1,000 |
| two-sided primitive survivors / q | 238,813 |
| cofactorisation candidates / q | 1,958 |
| **complete relations / q** | **8.28** |
| relations emitted | **8,283**, all distinct `(a,b)` |
| candidates emitted | **1,958,036** (295 MB) |
| **verified from (a,b) and the polynomial** | **8,283 / 8,283** |

Nothing had to be added to make this run. The band loop, the growth policy on
the survivor buffers, the per-q overflow checks on the bucket array and the
factor-list cap all held for 1,000 consecutive special-q without firing.

### Against las, over the same band

`oracle/c183.q120000000-120001000.relations.default.txt` is las's own output over
q in [120000000, 120001000] -- **67 special-q** once every root is counted. Run
against exactly that band:

| | |
|---|---:|
| las's relations | **3,162** |
| ours, trial division only | **556** |
| of ours, also found by las | **527** |
| **of ours, NOT found by las** | **29** |
| las's still locked in our candidates | **2,635** |

Two things follow, and they point in opposite directions.

**The 29 are real and they are ours.** Same effect as the single case recorded
in the 8-q band: our norms are exact where las's are approximate, so positions
las discards can still carry relations. At 29 in 556 that is **+5.2%**, which
lines up with the +5.6% survivor excess PARITY.md attributes to las's
approximation. It is a yield *bonus*, measured now at band scale rather than
inferred from one relation.

**And 83% of the yield is not there yet.** 556 of 3,162 is **17.6%**. The other
2,635 relations are sitting inside the 129,993 cofactorisation candidates this
band emitted and never processed. That is the whole remaining gap, it is a
single missing stage, and it is the last thing between this and a siever whose
relations/sec can be quoted at all. Every timing in this document is a
*post-sieve* timing with no cofactorisation term measured in it.

### What cofactorisation actually has to do, from the 1.96M-record corpus

Sizing it from the real output rather than from the mfb settings, over the
1,000-q band's 1,958,036 candidates:

| | records | share |
|---|---:|---:|
| both sides need splitting | 998,719 | 51.0% |
| algebraic only | 950,406 | 48.5% |
| rational only | 8,911 | 0.5% |

**Rational (lpb 31, mfb 60): 1,007,630 splits, every one of them 53-60 bits and
2LP.** Trial division stripped everything below `rlim` = 67.1M, so both factors
lie in (2^26, 2^31]. There is no 3LP case on this side at all.

**Algebraic (lpb 32, mfb 92): 1,949,125 splits, and they are bimodal.** 51,163
(2.6%) at 55-64 bits are 2LP; 1,897,962 (97.4%) at 81-92 bits are 3LP, three
factors each in (2^27, 2^32]. **Nothing lands in 65-80 bits** -- that band is
exactly `COF_REJECT_GAP`, since a cofactor between L^2 = 2^64 and B^3 = 2^81.05
cannot split at all, and the classifier already threw it away. Seeing the gap
appear in the emitted data is a check on the classifier that no count could give.

Two design consequences follow directly:

- **The rational side is the cheap gate and it must run first.** It is one
  60-bit semiprime split, and 51% of records need it. A record whose rational
  side does not split admissibly is dead regardless of the algebraic side, so
  the 3LP work on it never has to be started. That is the class-aware dispatch,
  and the corpus says it can suppress up to half the expensive work.
- **97.4% of the algebraic work is one shape**: a 81-92 bit number into three
  primes of about 30 bits. That is narrow enough to specialise for, rather than
  building a general cofactorizer.

## Cofactorisation, and the first complete relation rate

`bench --cofac FILE` splits an emitted candidate batch on the GPU and writes the
relations. Built as a separate entry point on purpose: the corpus is 1,958,036
real candidates from the 1,000-q band, which is a far better test of a splitter
than one special-q would be, and it keeps the algorithm measurable before it is
wired into the band loop.

**Pollard-Brent rho, not ECM, and bounded rather than capped.** ECM's appeal on
a GPU is that its cost is data-independent; rho's iteration count is geometric
and a warp runs at its slowest lane. The answer taken here is to bound the
budget per launch and requeue whatever has not split with a different rho
constant, so divergence inside a launch is bounded by the budget rather than by
the tail of a distribution. That is the "reseed, requeue, escalate" structure,
and it is why nothing is ever *dropped*: a cofactor that resists one constant is
not one that cannot be split.

**Primality is nearly free here.** Trial division removed every factor below
`lim`, so any part below lim^2 is prime BY SIZE with no test at all. On the
algebraic side lim^2 is 2^54, so the only parts needing a real sprp2 are those
of 55 bits or more -- the middle of a 3LP split. A general cofactorizer would
test everything.

**Class-aware dispatch, and it is worth 15.6%.** The rational side runs first
and alone: one 60-bit semiprime split, against the algebraic side's 92-bit 3LP
one. A record whose rational side has a prime above 2^31 is dead whatever the
algebraic side does, so its algebraic job is never enqueued. Measured over the
full corpus, that suppresses **303,201 of 1,949,125** algebraic jobs.

### Where the budget knee is

Over las's own 67-q band, sweeping the requeue rounds at budget 65536:

| rounds | relations | las's still missed | GPU ms/q |
|---:|---:|---:|---:|
| 2 | 3,147 | 143 | **14.9** |
| 3 | 3,155 | 136 | 28.2 |
| 4 | 3,155 | 136 | 54.1 |
| 5 | 3,155 | 136 | 103.5 |

**Yield saturates at 3 rounds and each further round costs 2x for nothing.** The
136 that remain are therefore *not* a rho-budget problem -- more rho does not
find them -- which points at the sieve's survivor bound, the one item this
document already has open (bound 128 loses a relation at the parity q). Rounds =
2 is the production setting: it buys 99.7% of the achievable yield at 53% of the
cost of the setting that saturates.

### Against las, end to end

Same 67 special-q, trial division plus cofactorisation:

| | |
|---|---:|
| las's relations | 3,162 |
| **ours** | **3,147 (99.5%)** |
| of ours, las found too | 3,019 |
| **of ours, las did NOT find** | **128** |
| las's we still miss | 143 |

**Every relation was verified independently** from `(a,b)` and the polynomial --
recorded factors times residual cofactor rebuild both exact norms, every prime
inside its side's bound: **2,591/2,591** on this band and **38,440/38,440** over
the full corpus.

### The 1,000-q band, complete

| | |
|---|---:|
| relations from trial division | 8,283 |
| relations from cofactorisation | 38,440 |
| **total** | **46,723** (46,722 distinct `(a,b)`) |
| **relations / q** | **46.7** |
| las, same job | 47.2 / q |
| cofactorisation, GPU | **13.44 ms/q** |
| sieve + TD | 96.70 ms/q |
| **wall clock / q** | **110.1 ms** |

**This is the first end-to-end relation rate in this document, and it is 99% of
las's yield.**

### Against finding 46, with a cofactor term that is measured

| goal | post-sieve budget | where we are |
|---|---:|---|
| throughput parity | ~220 ms | not binding |
| energy parity | ~106-114 ms | **41.4 ms: 2.6x of headroom** |
| **2x energy win** | **~21-25 ms** | **44.1 ms: not met** |

Post-sieve is 44.06 ms/q in the wired pipeline -- 20.71 of TD and
classification, 18.89 of cofactorisation, and the rest join, intersect and
per-q host work. (The batch cofactorizer measures 13.44 ms/q on the same corpus
because its job arrays are compacted; see the queue section.) Finding 47 projected ~4.92 ms/q for the cofactor term; the
measured figure is **2.7x that**, and the projection is now retired -- it was
always a projection, and it was optimistic.

**The lever, if the 2x win is wanted, is ECM.** The evidence is in the round
sweep above: rho's cost doubles per round while its yield is flat after round 3,
which is the signature of a method spending nearly all of its time on cofactors
it will never split. 966,305 of the 1,645,924 algebraic jobs finish "stuck" --
almost all of them are *dead* rather than unsplit, and rho has no cheap way to
prove it. ECM finds a 30-bit factor in far fewer modmuls than rho and its
failure is informative at a fixed price. That is a build, not a tune, and it
should be costed against the 2x win rather than started on principle.

### Wired in: the cross-q device queue

`bench --pipeline --cofactor` now runs sieve to relations in one process with
nothing on disk in between. A special-q produces ~1,956 joint candidates, which
at the pipeline's launch geometry is 3% occupancy -- so cofactorising per q
would run the rho kernel on an almost empty grid and pay every requeue round
for it. Candidates accumulate in a device-resident queue and flush at 131,072,
about 67 q worth. Only the ~2% of records that become relations are ever read
back.

**Band of 1,000 special-q, one process, sieve to relations:**

| stage | ms/q |
|---|---:|
| sieve, both sides | 69.18 |
| intersect + gcd | 0.42 |
| host per-q (sieve tables) | 1.10 |
| TD + classify | 20.71 |
| join and emit | 0.24 |
| **cofactorisation (queue + flush)** | **18.89** |
| unaccounted | 2.36 |
| final flush after the band | 0.34 |
| **wall clock per q, COMPLETE** | **113.24** |
| **relations / q** | **46.72** |

**46,723 relations, 46,722 distinct `(a,b)`, all 46,723 verified** from `(a,b)`
and the polynomial. Over las's own 67-q band the inline path's output is
**byte-for-byte identical** to the two-pass `--candidates` + `--cofac` path.

Two things the wiring bought beyond the file going away: `join and emit` fell
from 3.00 to **0.24 ms/q**, and the 295 MB candidate file for this band is
simply not written.

**It also cost 27% on the cofactor stage, and the reason is worth stating.**
The batch path compacted its job arrays -- 109,446 contiguous algebraic jobs
after the rational gate -- and measured 14.9 ms/q. The inline queue runs all
130,549 slots with the gated-out records falling through, so its live lanes are
scattered: nearly every warp holds a mix and runs at the live lanes' cost.
4,080 warps against 3,420 is 1.19x, and 14.9 x 1.19 is 17.7 against the 18.9
measured. **Compacting the queue before each flush is worth about 20% of this
stage**, which is ~3.7 ms/q of a 113 ms/q wall. It is a real lever, it is
quantified, and it is not built.

### Two accounting errors I made wiring this up, both caught by arithmetic

**The queue received the records trial division had already completed**, and
the host loop emitted them too -- 556 relations written twice over the 67-q
band. The queue is now the single emitter. Caught because "candidates
cofactorised" read 130,549 against 129,993 candidates, a difference of exactly
556.

**Then the summary double-counted the cofactor time**, reporting 132.31 ms/q
against a process that finished in 112 ms/q: the flush happens inside the band
loop, so `acc_wall` already contained it and the summary added the device times
on top. Only the *final* partial flush, after the loop, is outside `acc_wall`,
and it is now the only thing added back. **A per-q figure that exceeds the
process wall clock divided by q is always wrong**, and that check is cheaper
than any of the reasoning that produced the number.

### One bug worth recording

The emitter dropped every record whose **algebraic cofactor was already 1** --
trial division had fully split that side, so there was no algebraic job to come
back from, and the join looked for the record in the queue's output. It cost
**17% of the yield** on the first sample and it was found by the count of
relations disagreeing with the number of lines written by one. The lesson is the
cheap one: emit a count from each path and compare it against the file, because
a stage that silently produces less is invisible in every other check -- all 391
relations it did write verified perfectly.

### What this does and does not measure

These are band averages over 348 special-q in one process, so the first-launch
and one-time costs that made the single-q numbers unusable have amortised.
**They are the honest ones**, with two caveats worth keeping attached:

- the first q's reconstruction gate is excluded and reported separately (137.8
  ms), because it reads back both 61 MB factor matrices and rebuilds a 224-bit
  norm per candidate on the host. It is a validation phase, not the siever.
- there is still **no cofactorisation stage**. The 8.43 relations/q are the ones
  that fall out of trial division alone; the 1,956 candidates/q are what a
  cofactorizer would have to process, and finding 47's term for that is still a
  projection.

## Lazy module loading bit again

The rank scan first reported **10.7 ms**; timed best-of-3 it is **0.17 ms**, a
63x error. Same cause as the intersect kernel's 4.6 ms in the review above —
CUDA loads a kernel's code on its first launch, and these four kernels had
never run. The sieve having already executed dozens of launches does not help;
the loading is per-kernel, not per-module. **Every new kernel added to this
project needs best-of-N timing from the first measurement, not after someone
notices the number is implausible.**

## Two things I would not do yet

- **Do not widen the bucket record to 8 B.** It is the obvious way to make
  resieve trivial and it is a permanent tax on the largest stage in the chain to
  avoid a stage that the sparsity argument already makes cheap. Revisit only if
  the measured re-walk badly misses the ~8–14 ms estimate — and note that if it
  does miss, the 2× budget is gone either way, so the decision will be made on
  evidence rather than on this trade. ~~**Layout B is the thing to try before 8 B
  records**~~ — **superseded: layout B was built and lost by 2.6x** (see "Item 5
  RESOLVED" above). The re-walk did not miss its estimate; it measures 8.55 ms
  against the 8–14 ms predicted, so the condition for revisiting the 8 B record
  has not been met from either direction.
- **Do not chase the remaining +5.6% survivor excess.** It is las's
  approximation, not our error (PARITY.md), and its whole cost is a 5.6% larger
  input to the post-sieve stages — which the bound sweep in item 2 may make
  moot in either direction. Get the containment zero first; the excess is a
  cost question, not a correctness one.

## Costing ECM against rho, and what the rho profile actually says

Task 11 asked for a cost, not a build. The measurement needed first was: where do
rho's iterations *go*? `k_cofac` now accumulates iterations per job and bins them
by final status. Over the 67-q band, 129,993 candidates, rounds = 2, budget =
65536:

| algebraic (3-limb) outcome | jobs | share of iterations | mean iterations |
|---|---:|---:|---:|
| split ok | 2,199 | **0.68%** | 71,617 |
| proven dead | 43,130 | 17.15% | 91,578 |
| **stuck (never resolved)** | **64,117** | **82.17%** | 295,182 |

The rational side by contrast has **zero** stuck jobs and costs 101.8 ms against
the algebraic side's 958.5 ms. The whole problem is the 3LP side.

**82% of the work resolves nothing, and 0.68% of it produces the relations.**

### The stuck records are dead, and that is not a rho failure

Sampling 400 of the 125,919 3LP-sized algebraic cofactors and factoring them on
the CPU:

| | |
|---|---:|
| admissible (every prime ≤ 2^32) | 3 (0.8%) |
| **dead (some prime > 2^32)** | **397 (99.2%)** |

The dead ones' smallest over-`lpb` factor runs 33–65 bits, massed at 33–43. The
admissible ones' smallest factor is 28 bits — trivial for rho. So the
cofactoriser's actual job on this job is **rejection**, and rejection is the one
thing no factoring algorithm can do cheaply: you cannot prove a number has no
factor below 2^32 without searching for one.

### Tuning rho is not the lever — measured, not argued

The earlier sweep varied *rounds* at fixed budget. It never varied the budget.
Sweeping it at rounds = 2, and costing against the whole pipeline rather than
the stage:

| budget | cofac ms/q | relations/q | pipeline ms/q | rel/ms |
|---:|---:|---:|---:|---:|
| 16384 | 5.8 | 29.2 | ~103 | 0.284 |
| 32768 | 10.7 | 43.0 | ~108 | 0.398 |
| **65536** | **15.8** | **47.0** | **~113** | **0.416** |

Yield does not saturate in budget; it tracks cost. Going down loses more yield
than cost, and the rounds sweep already showed going up costs 13.3 ms/q for 8
relations. **Rho is already at its optimum, so any gain must come from a
different algorithm rather than a different parameter.** That is a negative
result and it is the useful half of this task.

### What ECM is and is not worth here

The intuition that ECM beats rho is right in general and needs care here.
*Every* factor this job hunts is 8–10 decimal digits — the rational side splits
53–60 bit semiprimes into two factors in (2^26, 2^31], and 97.4% of the
algebraic work is three factors in (2^27, 2^32]. That is the regime where rho is
competitive; ECM's subexponential advantage opens up well above it.

And ECM cannot buy **yield**: rho at 2 rounds is 65536 + 131072 = 196,608
iterations, against ~1.25·√p ≈ 82,000 expected for a factor at the very top of
the admissible range. Rho's reach already exceeds `lpb`, which is why the rounds
sweep saturates. The 136 relations still missed are the survivor bound, not the
splitter.

So the entire ECM case is **cost of rejection**, and it rests on one asymmetry:
rho's cost to reject is set by √p for a factor it will never find, while ECM's
is set by B1, which can be sized to exactly "no factor ≤ 2^32" and stopped.

An order-of-magnitude cost, in 3-limb modular multiplications per dead record:

| | modmuls | basis |
|---|---:|---|
| rho, as built | ~590,000 | 295,182 mean iterations × ~2 modmuls |
| ECM, B1 = 2000, 3 curves | ~144,000 | ~1.44·B1 ≈ 2,885 ladder bits × ~11 modmuls, +50% for stage 2 |

**~4×, and it applies to the 82% of the stage that is currently pure waste.**
Folding that in: the algebraic stage would go from 14.3 ms/q to roughly
14.3 × (0.18 + 0.82/4) ≈ 5.5 ms/q, saving ~8.8 ms/q.

### Why that still does not close the 2× win, and the arithmetic that says so

| | ms/q |
|---|---:|
| post-sieve today | 44.06 |
| cofactorisation (queue + tail) | 19.23 |
| **post-sieve if cofactorisation were free** | **24.83** |

The 2× energy window is 21–25 ms. **A perfect cofactoriser lands at the worst
edge of it.** With ECM at the estimated 4× (~8.8 ms/q) and the queue compaction
already quantified (~3.7 ms/q), post-sieve reaches ~31.6 ms/q — a real
improvement, comfortably past energy parity, and still short of the 2× win.
**TD at 20.71 ms/q is now the co-equal problem and no amount of ECM touches
it.** Any plan that spends its next effort only on the cofactoriser is planning
to miss.

The ECM figure above is a **model, not a measurement** — the 11-modmuls-per-bit
ladder constant assumes Montgomery curves with PRAC chains, and ECM's register
pressure and stage-2 tables may cost occupancy that rho does not pay. It is good
enough to rank the lever, which is what the task asked for.

## The `--cadofb` trap, which cost 2/3 of the relations and looked like a regression

Found while gathering the profile above, and worth recording as a method
failure rather than a bug. Running the band without `--cadofb` gave 13 relations
at q = 120000053 where las gets 37, and 15.52 relations/q where the band summary
says 46.72. It looked exactly like a regression in the task-10 wiring.

It was not. **The GGNFS `.afb.0` default factor base carries neither p = 2 nor
any prime power.** The chain from there:

1. The algebraic norm keeps its full power of 2 — measured at 2,174 of 2,707
   records, true 2-adic valuation 3–9, recorded 2s **zero**. (`c2` is the only
   odd coefficient of this polynomial, so F(a,b) ≡ a²b³ (mod 2) and 2 has both
   an affine and a projective root.)
2. `cof_classify` *accepts* even cofactors on purpose (`prp.cuh:211`), so they
   inflate the candidate count rather than being rejected — 2,707 against
   CADO's 1,852.
3. `mz_rho` builds `mz_n0inv(n->v[0])`, which requires an **odd** modulus.
   Montgomery arithmetic on an even n is undefined, so those records can never
   split. They burn the entire budget and report `CF_INCOMPLETE`.

**Nothing in the codebase could see this.** The norms still reconstruct exactly,
so every verification passed — factors × cofactor rebuilt both exact norms on
100% of records, because the leftover 2^k is *in* the recorded cofactor. `make
check` passed; it gates the factor-base transform, not end-to-end yield. The
only visible symptom was a relation count, and the only reason it was caught is
that it disagreed with a number already in this document.

Three lessons, in order of how much they would have saved:

- **A yield number is the only gate on a yield bug.** Every structural gate here
  passed. Reproducing a documented relation count is the cheap check and it was
  not run first.
- **"It reconstructs exactly" is not "it is complete."** Verification confirmed
  self-consistency of what we recorded, never that we had recorded everything
  divisible out. Those are different claims and only one was being checked.
- **A 3× discrepancy is a configuration difference until proven otherwise.** I
  spent the first stretch hunting a regression in uncommitted code, when the
  measurement that localised it — dump the small-prime table and look for p = 2 —
  took two minutes.

`bench` now **warns loudly** when the algebraic small factor base has no entry
for p = 2, and `TD_DUMP_SMALL=1` dumps both sides' trial-division tables. The
guard is three lines and would have turned a day into a minute.

## The queue compaction, built and measured

Task 12. `k_cofac` gained a `SELECT` template parameter and the flush now packs
the still-unresolved jobs before every round. The reason it matters is not that
a resolved lane costs anything itself — it costs nothing — but that **its warp
runs for as long as its slowest live lane**, so a warp holding one live lane and
31 dead ones pays a full rho budget to do one job. Packing turns that back into
32 jobs per budget.

The compaction is the ordered prefix scan already used for the survivor rank,
not an atomic append, so the job order is a function of the status array alone.
Clean A/B on the 67-q band, same binary, one template argument apart:

| | rational | algebraic | device/q | wall/q |
|---|---:|---:|---:|---:|
| uncompacted | 2.25 ms | 16.37 ms | 18.69 ms | 111.61 ms |
| **compacted** | **1.38 ms** | **12.75 ms** | **14.20 ms** | **106.70 ms** |

**1.32× on the stage, 4.91 ms/q off the wall** — better than the 1.19× the warp
count predicted, because compacting before *every* round also shrinks the
requeue rounds, which the prediction ignored. Output is byte-for-byte identical
to both the uncompacted queue and the two-pass `--candidates` + `--cofac` path,
3,147 relations all three ways. The device-side count means the grid is never
sized from a host-visible value, so no synchronisation was added.

## Side 0's survivor bound was never swept, and it was worth more than the kernel

Task 13 set out to attack TD's 20.31 ms/q, of which resieve is 9.93 — 54% of
device time. The lever turned out not to be in the kernel.

Side 1's bound has a documented zero-loss floor at 128. **Side 0's was never
swept at all.** It was barely filtering: side 1 alone gives 318,141 survivors and
the two-sided intersection is 238,929, so side 0 was removing 25% of what side 1
had already passed. Sweeping `--allowance0` on the 67-q band:

| `--allowance0` | survivors/q | relations | ms/q | rel/ms |
|---:|---:|---:|---:|---:|
| 68.1 (default) | 238,929 | 3,147 | 106.96 | 0.4391 |
| 62 | 143,998 | 3,139 | 100.22 | 0.4675 |
| **60** | **121,149** | **3,138** | **99.61** | **0.4702** |
| 59 | 111,106 | 3,034 | 98.61 | 0.4592 |
| 58 | 101,910 | 2,861 | 97.31 | 0.4388 |
| 56 | 85,871 | 2,325 | 93.57 | 0.3708 |

**The floor is sharp and it is at 60**: 59 already costs 3.3% of the relations
and 58 costs 9%. At 60 the survivor count halves for **9 relations out of 3,147
(0.29%)**. Confirmed on a wider 300-q band: 13,967 → 13,940 relations (−0.19%),
107.25 → 99.85 ms/q, rate +7.2%.

This is **not** offered as a zero-loss point — it is not one, and side 1's
documented floor is held to a stricter standard on purpose. It is offered as the
best point on relations/sec, which is the metric that decides the project. It
is one flag; the default is unchanged pending that call.

### Resieve is walk-dominated, which bounds what task 13 can ever win

Halving the survivors cut resieve by only 14% (9.93 → 8.52 ms). Fitting the two
points: **~7.1 ms/q is the factor-base walk and is survivor-independent**, and
only ~1.4 ms is the scatter that survivor count touches. So further survivor
reduction cannot move resieve, and resieve is now 65% of TD's device time. Two
points is a line, not a model — but the direction is not in doubt, and the
consequence is: **cutting resieve means cutting the walk, which means the 8 B
bucket record this document has twice refused.** That trade should be re-costed
against the new numbers rather than re-refused from the old ones.

## Where the whole pipeline now stands

67-q band, one process, sieve to relations, `--cadofb --allowance0 60`:

| stage | ms/q |
|---|---:|
| sieve, both sides | 65.74 |
| intersect + gcd | 0.41 |
| host per-q (sieve tables) | 1.03 |
| TD + classify (13.17 device + 1.66 host) | 14.83 |
| join and emit | 0.22 |
| cofactorisation | 14.37 |
| unaccounted | 1.67 |
| **wall clock per q, COMPLETE** | **98.26** |
| **relations / q** | **46.84** |

On absolute wall figures: the same binary and band re-measured later in the
session gave 100.1 ms/q for this configuration and 108.3 for the default -- a
~2% drift with the machine's clock state. Every comparison in this document's
recent sections was run back to back for that reason, and the *deltas* are
stable: compaction 4.91 ms/q, the side-0 bound 8.17 ms/q (repeated). Treat the
absolute numbers as +/-2% and the deltas as real.

Chaining the two back-to-back deltas against the session's starting point on
this band: **113.2 -> 100.1 ms/q at 46.97 -> 46.84 relations/q, a 12.8% gain in
relations per millisecond** for 0.29% of the relations.

**Post-sieve is now ~33 ms/q, against 44.06 at the start.** TD (14.83) and
cofactorisation (14.37) are now equal, which is the first time neither one
dominates.

## ECM stage 1: built, measured, and it loses to rho

Task 11 costed ECM from a model and projected ~4x, which would have put
post-sieve inside the 2x window. **That projection was wrong and the build says
so.** It assumed stage 2; stage 1 alone is a different proposition, because
stage 1 succeeds only when the whole group order is B1-smooth, and stage 2 is
where most of ECM's probability per unit cost lives.

Montgomery curves in XZ coordinates, Suyama parameterisation, binary ladder,
and the curve constant kept PROJECTIVE as A24 = (an : ad) so that no modular
inversion is needed per curve -- one extra multiply per doubling against a
binary extended GCD per curve per record, which at tens of curves is clearly the
right way round. `--cof-ecm --ecm-b1 N --ecm-curves N`. Same corpus, same 67-q
band, 129,993 candidates:

| config | relations | rational ms | algebraic ms | rel/ms |
|---|---:|---:|---:|---:|
| B1=250, 8 curves | 630 | 58.6 | 137.1 | 3.219 |
| B1=250, 32 curves | 1,902 | 126.2 | 593.0 | 2.645 |
| B1=1000, 8 curves | 1,878 | 134.3 | 703.1 | 2.243 |
| B1=1000, 32 curves | 2,589 | 305.3 | 2,071.7 | 1.089 |
| B1=2000, 32 curves | 2,599 | 427.2 | 4,215.5 | 0.560 |
| **rho, rounds=2, budget=65536** | **2,591** | **101.8** | **958.5** | **2.444** |

**At matched yield ECM is 2.3x slower**: B1=1000 with 32 curves takes 2,377 ms
against rho's 1,060 for the same ~2,590 relations. The configurations that beat
rho on rel/ms do so only by doing less work -- B1=250 with 8 curves is 3.219
rel/ms but finds 24% of the relations, and dropping 76% of the cofactor yield
costs far more pipeline throughput than it saves. Recomputing the whole-pipeline
rate, rho is 0.4711 and the best ECM point is 0.393, **17% worse**.

ECM is correct, not broken: it finds 2,589 relations of which **8 are ones rho
does not find**, while missing 10 that rho does. The two methods agree on the
substance and differ only in their stochastic tails.

### The one thing ECM is genuinely better at, and why it still does not pay

The hypothesis behind task 11 was that ECM would reject dead records faster.
**That is confirmed.** On the same corpus:

| | rho | ECM (B1=1000, 32 curves) |
|---|---:|---:|
| algebraic records **proven dead** | 43,130 | **71,846** |
| algebraic records left stuck | 64,117 | 35,391 |

ECM resolves **66% more dead records** and leaves 45% fewer unresolved, exactly
the mechanism predicted: ECM's cost is set by B1 rather than by sqrt(p), so a
dead cofactor whose smallest factor is 40 bits is within reach instead of being
hopeless. It simply does not pay -- the extra rejections cost more than the rho
budget they replace.

### Headroom not built, and the honest projection

Two real optimisations are missing, both deliberately:

- **PRAC addition chains** instead of the binary ladder: ~30% off stage 1.
- **Stage 2**, worth roughly 1.6x in probability per unit cost.

Applying both to the matched-yield point: 2,377 x 0.7 / 1.6 ~= **1,040 ms
against rho's 1,060**. A fully optimised ECM projects to **break-even**, not a
win -- and that is generous to ECM, since it assumes the two optimisations
compose perfectly and ignores stage 2's register pressure and table traffic.

**So the cofactoriser is not the road to the 2x energy win, and the ECM lever is
closed.** Post-sieve stands at ~33 ms/q against a 21-25 ms window, and the ~14.4
of it that cofactorisation costs is not going to fall by a factor that matters.
The remaining levers are TD's resieve -- walk-dominated with a ~7.1 ms/q floor,
so it means revisiting the 8 B bucket record this document has twice refused --
and the sieve itself at 65.7 ms/q, which no work in this session has touched.

The rho path is **byte-for-byte identical** after the refactor that added the
method switch, and `make check` passes.

## Review, 2026-08-05, and what it caught

An external review of everything above. Findings, and what was done.

### The ECM result was measured on a path the production run never takes

**The inline queue never received the ECM configuration.** `cofq_init` memset the
struct and nothing populated `Q.ecm`, `Q.ecm_curves`, `Q.d_s` or `Q.ns`, so the
branch in `cofq_flush` always chose rho no matter what `--cof-ecm` asked for.
Confirmed the way it should be: `--cof-ecm --ecm-b1 2 --ecm-curves 0` still
produced all 37 relations at the parity q, when zero curves can split nothing
and the answer had to be 7.

The rho-vs-ECM *comparison* survives -- both arms ran on the standalone path, so
it was apples to apples -- but the claim that the lever is closed **under the
production compacted scheduler** was never measured. It has been now:

| | ms/q | relations |
|---|---:|---:|
| **rho, rounds=2, budget=65536** | **14.40** | **3,147** |
| ECM B1=1000, 16 curves, 2 rounds | 31.29 | 3,146 |
| ECM B1=1000, 32 curves, 1 round | 36.17 | 3,145 |
| ECM B1=2000, 24 curves, 2 rounds | 65.41 | 3,155 |

**2.2-2.5x slower at matched yield**, slightly worse for ECM than the standalone
measurement suggested. Worth noting: ECM does reach 3,155, the same ceiling rho
reaches at rounds=3 -- but for 65.4 ms/q against rho's 28.2. The conclusion
stands and is now measured where it matters.

### Bounds that truncated silently

`--lpb`/`--mfb` were unrestricted while the representation is deliberately
narrow: split factors are emitted as one uint32 limb, the rational cofactor is
`mz<2>`, the algebraic `mz<3>`. `--lpb 33` therefore emitted relations whose
factors no longer reconstruct the norm, and a negative value through `atoi`
became an enormous unsigned bound. All of it now refuses: lpb <= 32, rational
mfb <= 64, algebraic mfb <= 96, plus `lim^2 > 2^lpb` (without which the
prime-by-size shortcut in `mz_split` is unsound), rounds in 1..24, and
`budget << (rounds-1)` checked against uint32 rather than left to shift past the
width.

### The even-cofactor invariant was a warning, and a warning was not enough

The guard added earlier diagnosed the missing p = 2 correctly and then **exited
0 anyway**, having lost two thirds of the yield -- which is precisely how this
was mistaken for a regression once already. Sieve-only runs still warn, since
they never reach the cofactoriser and their survivor counts are meaningful.
Anything producing relations, candidates, or inline cofactorisation now **fails
fast**: `mz_n0inv` requires an odd modulus and there is no honest way to
continue past a violation of it.

### Gates that could not see the thing they were gating

- **`make check` depended on `fbtest` alone**, so the whole CUDA path could fail
  to compile while the gates reported "all gates passed". It now depends on
  `all`.
- **The reconstruction gate ran before splitting**, so it said nothing about the
  primes the cofactoriser emits. `--check-relations FILE` now rebuilds both
  norms from `(a,b)` and the polynomial over an emitted relation file, requiring
  every recorded factor to divide exactly, both norms to reduce to 1, and every
  prime to sit within its side's lpb. Both emitters write the same format, so
  one implementation gates both paths. 3,147/3,147 on the rho band and
  2,589/2,589 on the ECM one, with a corrupted-factor negative control that
  fails as it must.
- **A new GPU golden test**, `cofcheck.sh`, wired into `make check`: rho and ECM
  at the parity q, the method switch actually taking effect, every bounds
  refusal, the even-cofactor refusal, a candidate file with no trailing newline,
  and inline-equals-two-pass byte equality. Twelve cases, each pinning an
  **exact** relation count -- the bugs in this section all presented as a
  plausible smaller number, never as a crash, so only an exact expectation
  catches them.

### The parser dropped its last record

`run_cofac` counted records by counting newlines and terminated the parse on the
first line without one, so a candidate file with no trailing newline silently
lost its final record. Fixed; a truncated copy of the 129,993-record corpus now
parses identically and yields the same 2,591 relations.

### Accounting

"candidates cofactorised 1,851" against 1,844 needing splitting was a labelling
error, not a double count: every joint-accepted record is enqueued including the
7 trial division had already completed, because the queue is the single emitter
and they have to travel with the rest. They cost nothing -- their status is
already `CF_OK` and compaction drops them before the first round. Now reported
as "records enqueued ... (of which N needed no splitting)". A device readback to
count them exactly was written and then reverted: a synchronisation in the flush
loop is too high a price for a cosmetic number.

### Left undone, deliberately

- **The inline path is not fully device-resident.** Every candidate's
  coordinates, cofactors and two 64-entry factor arrays are still copied to the
  host with a host loop over all of them. Measured cost is ~0.5-0.9 ms/q of a
  ~100 ms/q wall, and the emit path is what every byte-identical guarantee in
  this document rests on, so it is a task and not a same-session change.
- **The two schedulers should become one batch executor.** They diverged in
  exactly the dangerous way -- standalone had ECM but no compaction, inline had
  compaction but never received the ECM configuration. Unifying them is the
  structural fix for the class of bug this review found.

The band output is **byte-for-byte identical to before the review** at 3,147
relations, and the whole gate suite passes.

## The two schedulers became one

Task 15, and the structural fix for the class of bug the review found.
`cf_run_side` (standalone `--cofac`) and the loop inside `cofq_flush` (the
inline cross-q queue) were two independent implementations of the same
round-and-requeue schedule. They diverged exactly as duplicated schedulers do:
the standalone one grew ECM and never got per-round compaction, the inline one
grew compaction and never received the ECM configuration. **Each worked
perfectly when tested on its own**, which is why neither defect was visible
until someone ran the production path with `--cof-ecm` and checked the number.

Now there is one `cf_run_rounds`, parameterised by a `cf_sched_t` holding the
only things that actually differ between callers -- method, rounds, budget,
curves, the ECM plan. Two consequences beyond the obvious:

- **`k_cofac` lost its `SELECT` template parameter.** The uncompacted variant
  existed only for the compaction A/B, whose result is recorded above; keeping
  it meant keeping a second code path for a measurement already made. Always
  compact.
- **The standalone path inherited compaction**, which it never had: its
  algebraic side went 958.5 -> ~865 ms on the 67-q corpus, ~9.7% faster, with
  byte-identical output.

A `dense_first` flag was written to skip the round-0 scan on the standalone path
(where `memset` makes every job live, so the scan can only produce the identity)
and then **reverted**: it did not pay above noise, and it reintroduced a
per-caller behavioural difference, which is the precise thing this task exists
to remove.

## The inline path is now device-resident, and it did not show on the clock

Task 16. Under `--cofactor` the pipeline was copying every candidate's
coordinates, cofactors, factor counts and two 64-entry factor arrays to the host
-- about 618 bytes each, ~1.2 MB per q -- and looping over all of them, to do
three things: validate the factor count, count relations, count candidates. The
queue had already taken what it needed straight from device memory, and
`cq_emit_side` sorts for itself, so even the host-side `std::sort` was sorting
copies nobody read.

`k_cand_stats` replaces all of it with three counters. `readback of candidates`
and `join and emit` both went to **0.000 ms/q** from 0.34 and 0.22.

**The wall clock did not move**: 108.0-109.4 ms/q against a 108.0-108.6
baseline. The removed 0.55 ms/q is 0.5% of the wall, and this machine's
run-to-run spread is ~2%. **The change is below the noise floor and is not
claimed as a speedup.**

It is still the right shape, and one wrong turn inside it is worth recording.
The first version read the three counters back **per q**, and that blocking
12-byte readback cost ~0.45 ms/q -- it consumed nearly the whole saving, because
on this hardware a device round trip per q costs more than the megabyte of PCIe
traffic it was replacing. The counters now accumulate across the band and are
read once at the end. The overflow check moved with them: a truncated factor
list still fails the run, at band granularity rather than per q. **A per-q
synchronisation is what the rest of this pipeline is built to avoid, and adding
one back to save a copy was a bad trade that only measurement caught.**

## The 8 B bucket record, re-costed and closed for good

Task 14. This document refused the 8 B record twice on the argument that it taxes
the largest stage to save a smaller one. Both sides of that trade had moved --
the sieve is ~66-69 ms/q and post-sieve ~33, and resieve is walk-dominated with
a ~7.1 ms/q floor -- so the refusal was due a re-cost against measurements
rather than a third repetition. Measured, `--record-bytes 4` vs `8`, same q,
same factor base:

| | 4 B | 8 B |
|---|---:|---:|
| bucket array | 1.38 GB | 2.76 GB |
| fill | **12.40 ms** | **17.66 ms** |

**The fill tax is +5.26 ms/q, 42%** -- and `apply` is not even implemented for
8 B records, though it reads every record and would see twice the volume.

But the decisive number is on the benefit side, and it is not the one the
earlier arguments were about. Carrying the factor-base index does not make
resieve free; it **replaces a re-walk of the factor base with a re-read of the
bucket array**, and those two are not the same size:

| | objects | bytes |
|---|---:|---:|
| factor base (what resieve re-walks) | 11.56 M x 16 B | **0.185 GB** |
| bucket records (what 8 B would re-read) | 610.7 M x 8 B | **4.89 GB** |

**26x more data.** At an optimistic 600 GB/s the replacement pass alone is
~8.1 ms/q, against the 9.99 ms/q resieve currently costs -- and that is before
the +5.26 ms fill tax, before whatever `apply` gives up, and before the scatter,
which still has to happen either way.

So the trade is not close: **~13+ ms/q of new cost to remove 9.99 ms/q**, with
the memory footprint doubling on a 12 GB card. And the reason resieve is cheap
is now stated properly rather than assumed: **it re-walks 185 MB of factor base
precisely to avoid re-reading 4.9 GB of bucket records.** Its ~7.1 ms/q floor is
not a defect to engineer around; it is close to what re-deriving that
information has to cost on this hardware.

**Closed. The remaining post-sieve levers are not in resieve, and the sieve
itself -- 66-69 ms/q, two thirds of the wall and untouched all session -- is
where the next real work is.**

## A second job: the c123, with parameters derived rather than tuned

Task 17, and the first evidence that any of this generalises. `--auto-params`
implements CADO's own derivation (`las-norms.cpp:237`) instead of the constants
hand-fitted to c183:

```
maxlog2 = log2(largest norm over the sieve rectangle)
scale   = (255 - 1) / maxlog2,  quantised to (int)(scale*40) * 0.025
lambda  = given, or CADO's automatic 0.3 + mfb/lpb
r       = min(maxlog2 - 1/scale, lambda*lpb)          <- our "allowance"
bound   = (unsigned char)(r*scale + 1)
```

**It reproduces both hardcoded constants from the polynomial alone**: scale
1.2750 on side 1 and 1.9250 on side 0, which is what they were tuned to. The
derived allowances (101.60, 69.30) land within 1.5% of the values our own
sweeps found (100.0, 68.1) -- CADO's automatic lambda and our measured optimum
agree, which is reassuring about both.

Run on a live c123 (I=13, lim 6M/3.5M, lpb 29/28, mfb 57/53 -- a 2LP job on
both sides, nothing like c183's 3LP), with no code changes, only flags:

| | las | ours |
|---|---|---|
| side 0 | log2(maxnorm)=93.15, scale=2.73, **bound=145** | 93.15, 2.725, **bound=145** |
| side 1 | 142.97-144.23, scale 1.75-1.78, bound 100-101 | 145.91, 1.725, bound 99 |

Over q in [400000, 401000], 74 special-q:

| | |
|---|---:|
| las's distinct relations | 27,074 |
| **ours** | **27,594** |
| of las's, we found | **27,039 (99.87%)** |
| **found by us, not by las** | **555** |
| las's we missed | 35 |
| **reconstruction gate** | **27,605 / 27,605 exact** |

The 555 extra are the same effect seen on c183: las is running `-ncurves0 11
-ncurves1 13`, a bounded ECM effort, against our rho at rounds 2 / budget 65536.
The 35 missed are our side-1 bound being one unit tight, which has a known
cause.

### The one real inaccuracy this exposed

Our `maxlog2` is the largest single homogeneous term; CADO's is the true maximum
of |F| over the rectangle. That makes us **~2 bits conservative on side 1**
(145.91 against 142.97-144.23), hence bound 99 against 100-101, and it costs
0.13% of las's relations. Side 0, being degree 1, matches exactly.

Also worth recording: **las re-derives the scale per special-q** -- it printed
1.78 for one q and 1.75 for the next -- and re-slices its factor base to match.
We bake the factor-base logs once per run, so a band-wide fixed scale is an
approximation. It held here, but it is an assumption that should be checked per
q rather than trusted.

## The c123, factored end to end

Task 18. GPU sieve -> msieve filtering -> msieve GPU linear algebra -> square
root -> factors, on a job the code had never seen, with survivor bounds derived
from the polynomial rather than tuned.

```
p42 factor: 131980915436137426858863181246116560763169
p82 factor: 1691057659004520172717782898437809080597503666034065786710818800020911277994733649
```

`p * q == N`, verified independently.

| stage | wall |
|---|---:|
| GPU sieve, q in [400000, 1900000], 107,813 special-q | 1,093 s |
| msieve filtering (`-nc1`) | 171 s |
| msieve linear algebra on GPU (`-nc2`) | 61 s |
| square root (`-nc3`) | 153 s |

29,339,495 relations. msieve found 849,784 cycles against 825,717 needed, and
built an 823,418 x 823,845 matrix. **We generate no free relations** -- CADO
generated 122,390 of them for the same job -- and filtering still cleared its
target, so they are not load-bearing at this size.

### Against CADO on the same job

The comparison is close to apples-to-apples: same N, same polynomial, same
factor-base bounds, 107,813 special-q against 103,957, and yields within 1.8%
(272.1 vs 267.4 relations/q). CADO's own log confirms the parallelism:
18,489.9 s CPU over 2,220 s elapsed is 8.33x, matching its 8 sieve instances.

| | CADO (8 instances) | GPU |
|---|---:|---:|
| special-q | 103,957 | 107,813 |
| relations | 27,798,825 | 29,339,495 |
| **sieving wall clock** | **2,220 s** | **1,093 s** |
| ms per special-q | 21.36 | **10.14** |
| relations/sec | 12,522 | **26,839** |

**2.03x on wall clock, 2.11x per special-q.**

### A correction worth recording

An earlier version of this comparison claimed **17x**, from reading CADO's
`Total time: 18489.9s` as wall clock. It is the SUM over sieve instances; eight
ran at once. Kyle caught it. The same mistake then propagated into a 14.7x
perf/watt figure.

The perf/watt number also mixed two configurations: the 206.7 W GPU figure in
the power table above was measured when the GPU side was **sieve only**, before
trial division, classify and cofactorisation existed, and it was being combined
with a 128.5 W reading from the current full pipeline. On the measured-but-stale
proxies the c123 comes out at **~1.85x relations/sec/watt**, which is the right
order but should be treated as provisional until both sides are re-measured with
the current code.

**Two lessons, and the first is the one that keeps recurring in this document:**
a number quoted from a log is not a measurement until you know what the log
means. And a power figure is tied to the configuration that produced it -- the
206.7 W was never wrong, it was answering a different question.

## Device memory, measured rather than modelled

`bench --pipeline` now prints its allocations by stage and the band's steady
state, because a model of them was wrong twice in one session. On the c151
(A=27, alim 33.5M):

```
    bucket array                   0.34 GB
    factor bases + bitmaps         0.15 GB
    trial division context         0.03 GB
    cofactor queue                 0.14 GB
  device memory, steady state: 1.85 GB in use of 11.94 GB
```

**Our named allocations are 0.66 GB**; CUDA reports 1.85 GB in use including its
context; `nvidia-smi` reports ~4.1 GB. The last gap is not an allocation we
make -- under WSL2/WDDM the driver reserves and accounts memory differently, and
the desktop is on the same card. Do not reconcile those three numbers; measure
the one you mean.

The 4000-4100 MB oscillation Kyle observed is `pipe_td_grow` / `pipe_td_grow_cand`
freeing and reallocating as per-q survivor and candidate counts vary. Not a
leak: free memory was unchanged across a 4,000-q band.

### What actually scales with area

Only two things: the bucket array (`4 B x 1.15 x area x sum(1/p)`, formula
validated against c183 at 0.2%) and the survivor bitmaps (`area/8` bytes each,
three of them).

| | 2^27 (c151) | 2^29 (c183) | 2^31 | 2^32 (AS276) |
|---|---:|---:|---:|---:|
| bucket array, 4 B | 0.34 | 1.38 | 5.53 | **11.06** |
| survivor bitmaps | 0.05 | 0.20 | 0.81 | **1.61** |

**An earlier claim that "the bucket array is the whole story" was wrong** -- it
is 8% of the c151's footprint and most of c183's. It is right about *scaling*,
which is what a projection needs, but not about absolute size at small areas.

## The host thread is CPU-saturated, and it is not spin

`bench` sits at ~96-99% of one core for an entire run. The obvious explanation
is CUDA's default busy-wait at synchronisation points. **Measured, that is not
it.** Adding `--blocking-sync` (`cudaDeviceScheduleBlockingSync`):

| | wall | user | CPU |
|---|---:|---:|---:|
| spin (default) | 99.13 s | 97.61 s | 99% |
| `--blocking-sync` | 97.46 s | 92.83 s | 99% |

A 5% reduction, where pure spin would have collapsed it. The host thread is
doing real work: per-q table building (~1.7 ms of a 24.75 ms/q on the c151) and
the launch overhead of 50-100 kernels per q, plus whatever WSL2's GPU
virtualisation layer burns, which no device flag reaches.

**This is a serialisation limit, not a capacity one.** The box has 16 threads
with ~3 busy; the pipeline uses one, and that one issues every launch and does
every per-q host computation, with the GPU idle across it. That is a plausible
share of the 8% utilisation gap at 92%. The fix is overlap -- build the next q's
tables while the current q's kernels run -- not more cores.

## Caveat on everything numeric above

Measured during this review, on a busy box (so the only timing quoted from it
is the host-sort band, explicitly given as an upper band): the
`-batch-print-survivors` population (1,851, not 797,028); the survivor filter
occupancy table (9.34% at granularity 64); the full containment gate
(2,162 → 1,005 missing under `powlim` pinning, with side attribution);
relation containment (37/37); bit-exact bitmap reproducibility; the one-sided
popcounts contradicting finding 40's table; and the host insertion-sort cost.

Added 2026-08-04, also measured on a busy box (load 17–24, so again no timings):
the full-band relation containment (3,026/3,026 over 67 lattices, and the 47/67
basis agreement); the cofactor bit-size histogram over all 1,062,811 captured
records; and the side-1 slice count, bucket geometry and device memory
(39 slices, 32,768 buckets, 1.37 GB, 10.76 GB free) read off `bench`'s startup
output. All of these are **counts and sizes, which are load-independent** —
that is the only reason they were run while the box was busy. ~~Everything in
the resieve A/B remains **derived, not measured**~~ — **superseded: the A/B was
built and run on a quiet GPU** (see "Item 5 RESOLVED" and the 2026-08-04 build
log). Layout A wins on measurement, and the whole post-sieve chain now has
quiet-GPU numbers.

Except for those, the record layout, the CADO source behaviour, the counts
already in RESULTS.md, and finding 47's measured cofactor figure, every number
in this section is **derived from measured inputs, not measured**. The op-count models
are order-of-magnitude tools chosen to rank stages against each other and
against the finding-46 budget. They are good enough to decide what to build next
and in what order — and finding 47 is the demonstration of their limits: a
measurement came in at 0.7× my optimistic end and inverted a recommendation.
The first measurement of any stage should replace its row in the table above
rather than be compared against it.
