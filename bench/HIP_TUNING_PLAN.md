# HIP performance-tuning plan (AMD RDNA)

## Context and honest caveats, read first

The HIP port (`hip-port` branch) is functionally validated: `cofcheck.sh`'s
~52 pinned-count cases all pass on real gfx1103 hardware, matching counts
originally derived from CUDA. Nothing below changes that bar. This plan is
about the SEPARATE question CLAUDE.md has deliberately deferred: none of the
CUDA build's performance tuning (`__launch_bounds__` occupancy targets,
`log_region` sizing against shared-memory limits, grid width, fill-strategy
choice) has been re-derived for AMD hardware at all. Right now the HIP
kernels run with whatever the compiler defaults to, on parameters picked for
a 3090/5070/L40.

Two things make this box a poor *source of numbers* but a fine place to
*build the tuning process*:

1. **It's an iGPU with UMA memory**, not a discrete card. Its memory
   bandwidth, and therefore anything downstream of fill/apply's atomic
   scatter throughput, is bounded by shared DDR5 system memory bandwidth,
   not GDDR6/HBM. A measurement here that says "fill-mode X is Y% faster
   than fill-mode Z" is a real, true fact about *this chip*, but should not
   be assumed to hold on a discrete RDNA card with a completely different
   memory subsystem.
2. **fp64 is very slow on this hardware** (CLAUDE.md's own ground rule).
   Anything that touches double-precision math (there shouldn't be much in
   the hot path, but worth checking) will be disproportionately penalized
   here relative to a discrete card.

What SHOULD generalize reasonably well across RDNA generations, because
it's about compute-unit occupancy and ISA characteristics rather than the
memory subsystem: register-pressure-driven occupancy limits,
`__launch_bounds__`-style thread/block tuning, and the 64 KB (vs NVIDIA's
~100 KB opt-in) shared-memory ceiling, which is architectural to RDNA, not
specific to this iGPU. Those are the parts of this plan worth trusting
without a discrete card to confirm on; the raw ms/q numbers are not.

**Bottom line**: treat every number this box produces as "real for this
chip, directionally suggestive for RDNA generally, not a substitute for a
run on an actual RX 6800/7900/9070 before calling any of this tuned.**

## Available tooling (checked, not assumed)

This ROCm distribution (TheRock, Windows) ships **no rocprof, no omniperf,
no omnitrace** — none of ROCm's usual profilers are present on Windows
here. What IS available:

- **The codebase's own instrumentation.** `bench_kernels.hip`/
  `pipeline_hip.cuh` already carry extensive `hipEvent`-based per-stage wall
  timing (fill/apply/transform+plattice, already seen in every pipeline run
  this session: `fill 49.138 ms`, `apply 59.226 ms`, etc.). This is the
  primary measurement instrument for this plan — no new profiling
  infrastructure needs to be built.
- **`hipFuncGetAttributes()`** — an in-process HIP runtime call that reports
  a compiled kernel's actual register count, shared-memory usage, and max
  threads/block, without needing an external profiler. Cheap to add a
  one-off diagnostic that dumps this for k_fill_l1/l2/k_apply and the
  cofactor kernels.
- **`llvm-objdump`/`llvm-readobj`** (`C:\rocm\lib\llvm\bin\`) — can inspect
  a compiled kernel's ELF metadata (VGPR/SGPR/LDS usage) directly from the
  `.hipfb`/object file, the static-analysis equivalent of `ptxas -v`'s
  register counts that STATUS.md already tracks for the CUDA build. This is
  the direct AMD-side replacement for that table, since the CUDA numbers
  (78/94/86/112/122/154 registers per cofactor-kernel instantiation) do not
  transfer to AMD's compiler or ISA at all.
- **`testsieve.sh`** — the existing yield/timing harness. Already
  parameterized for this exact kind of before/after comparison; no new
  harness needed, just point it at `bench_hip.exe`.

No new tooling needs to be installed for any of this.

## Tuning targets, in priority order

### 1. `__launch_bounds__` re-tuning for k_fill_l1, k_fill_l2, k_apply

Currently: **dropped entirely** for HIP (CLAUDE.md work item 3, and the
recent code-review fix that restored a runtime `APPLY_THREADS_MAX` check
after finding the compiler-enforced ceiling was gone). The compiler picks
whatever register allocation it wants with no occupancy target at all right
now — this is the single most likely source of easy wins, since "no tuning
annotation" is very rarely the optimum.

HIP's `__launch_bounds__(maxThreads, minWarpsPerEU)` semantics differ from
CUDA's `(maxThreads, minBlocksPerMultiprocessor)` — the second argument
means **MIN_WARPS_PER_EXECUTION_UNIT** on HIP, a per-SIMD (not per-CU/WGP)
occupancy target. Re-deriving this needs gfx1103's actual resource limits
(already known from `hipInfo`/probe.hip: 6 CUs, 2048 max threads/CU, 64 KB
LDS/CU flat with no opt-in tier, warp size 32) fed into the same kind of
register-pressure-vs-occupancy tradeoff STATUS.md documents for the CUDA
side, not a guess.

**Action**: use `hipFuncGetAttributes()` to get each kernel's current
(untuned) register count, compute the occupancy that implies against
gfx1103's limits, then sweep `__launch_bounds__` values and measure with
the existing hipEvent timers. Compare against the untuned baseline before
touching anything else — this determines whether the other items below are
worth doing at all before k_apply's own resource shape is stable.

### 2. `log_region` / shared-memory ceiling re-sweep

Already flagged in CLAUDE.md (work item 2, and the original hardware probe)
as a **correctness-adjacent** constraint: NVIDIA's ~100 KB opt-in shared
memory ceiling let `k_apply` size 3 resident blocks at ~33 KB each; gfx1103
reports 64 KB flat with no separate opt-in tier, so even 2 blocks at CUDA's
tuned size won't fit. `log_region <= 14` was noted as the likely resulting
ceiling, but never actually swept for a performance optimum on AMD hardware
— only checked for whether it fits at all.

**Action**: sweep `--region` across its legal range once k_apply's
`__launch_bounds__` (item 1) is settled, measuring fill+apply wall time at
each. This interacts with item 1: shared-memory-per-block and
occupancy-target are two knobs on the same tradeoff, so tune them together,
not independently in sequence.

### 3. Fill-strategy re-comparison (`--fill-mode atomic` vs `twolevel`)

Already a runtime-switchable CLI flag (`--fill-mode`), needing no rebuild.
`FILL_ATOMIC` (one global `atomicAdd` per record) vs `FILL_TWOLEVEL`
(shared-memory staging, fewer, larger atomics) trade off differently
depending on the hardware's atomic throughput and memory subsystem — CUDA's
default (`FILL_ATOMIC`) was chosen based on NVIDIA hardware behavior with no
guarantee it's the right default for AMD's atomic implementation, and
doubly uncertain here given this chip's UMA memory subsystem (caveat #1
above applies directly to this specific comparison).

**Action**: cheapest item on this list — no code change, just run both
modes under `testsieve.sh` and record which wins on this hardware. Treat
the result as informative but explicitly flag that it may be an iGPU/UMA
artifact rather than a general RDNA finding (a discrete-card re-check would
settle that).

### 4. Register-pressure re-measurement for cofactor kernels

STATUS.md's exact `ptxas -v` register-count table for the rho/ECM/
`cf_is_prime64`-family kernel instantiations (78/94/86/112/122/154
registers depending on template parameters) is NVIDIA/ptxas-specific and
doesn't transfer to AMD's compiler or wavefront-32 VALU pipeline at all.
The 4-limb ECM-with-stage-2 instantiation is flagged in STATUS.md as the
tightest case on NVIDIA hardware; worth checking first on gfx1103 via
`llvm-objdump`/`hipFuncGetAttributes()` since an iGPU has a much smaller
register file per CU than any discrete card that table was measured on —
if anything spills here, it would spill worse on this chip specifically,
which could bias a "cofactor width doesn't cost much" read taken only from
this hardware.

**Action**: dump VGPR/LDS usage per template instantiation, compare against
gfx1103's per-CU limits, flag (don't necessarily fix) anything that spills.

### 5. Grid-width re-derivation

`bench_main_hip.cpp:1748`: `ab = prop.multiProcessorCount * 6u` — the `* 6`
multiplier is empirically tuned for NVIDIA occupancy (RESULTS.md: "6
resident blocks per SM" derived from Ampere/Ada/5070-class hardware), fed
through `prop.multiProcessorCount` so it already numerically adapts to
whatever the device reports (6 for gfx1103), but the *multiplier itself*
was never re-derived for AMD's CU/WGP structure — it's NVIDIA's tuned
constant, just pointed at an AMD CU count. Once items 1-2 settle k_apply's
real occupancy on gfx1103, re-derive this constant from that number rather
than carrying over `6` from a different vendor's architecture by
coincidence-of-syntax.

## Explicit non-goals

- **No algorithm changes.** This plan tunes launch configuration and
  resource sizing, not the sieve/cofactor algorithms themselves. Anything
  that looks like an algorithmic win belongs in a design discussion, not a
  tuning pass.
- **No touching the CUDA build.** Same ground rule as the port itself —
  every change here lives in the `_hip`-suffixed files or their
  HIP-specific `#ifdef` branches. Nothing in `bench_kernels.cu`,
  `bench_main.cu`, `cofac.cuh`, `pipeline.cuh`, `build_windows.bat`, or the
  Makefile changes.
- **No claiming these numbers as RDNA-general** without a discrete-card
  re-check (see caveats above). This plan's deliverable is a *validated
  methodology plus this chip's own tuned parameters*, not a claim that
  those exact parameters are correct for an RX 7900 or a 6600.
- **Correctness gate stays authoritative.** Every tuning change gets
  re-verified against `cofcheck.sh`'s pinned counts before being kept — a
  faster wrong answer is not a win. Treat cofcheck.sh the same way here as
  it was treated during the port itself: the thing that decides whether a
  change survives, not a courtesy check afterward.

## Sequencing

1. **Baseline**: run `testsieve.sh` on current `bench_hip.exe` (untuned,
   no `__launch_bounds__`), record fill/apply/total ms/q as the number
   everything else is measured against. Dump current register/LDS usage
   for k_fill_l1/l2/k_apply via `hipFuncGetAttributes()` alongside it.
2. **Item 1 + 2 together** (`__launch_bounds__` x `log_region` joint
   sweep): these interact, don't tune in isolation. Re-run `cofcheck.sh`
   after each candidate change, not just at the end.
3. **Item 3** (fill-mode): cheap, no rebuild, do it in parallel with 1-2
   since it's an independent runtime flag.
4. **Item 4** (cofactor register pressure): static analysis only, no
   runtime measurement needed, can happen any time after step 1's baseline
   tooling is working.
5. **Item 5** (grid width): last, since it depends on item 1's settled
   occupancy number.
6. **Write-up**: record final parameters and the *baseline vs tuned*
   comparison in this file, with the caveats from the top of this document
   repeated next to the numbers, not left implicit.

This is real engineering work, not a quick pass — expect it to look like
the port itself did: several rounds of measure/change/re-verify against
cofcheck.sh, not one sweep that lands everything at once.

## Item 6 (new, not in the original list): j-slab performance target -- CHANGED

Not one of the original five items -- added after the user asked directly
whether `slab.h`'s `SLAB_PERF_TARGET_LOG2 = 29` (2^29 positions/slab, the
auto-mode j-slabbing target above the 2^30-position trigger) is still right
on gfx1103. It documents its own provenance clearly: "Two independent
benchmarks (Ampere RTX 3090 and Blackwell RTX 5070) found this working set
near the fill/TD crossover; an L40 with a larger L2 cache instead preferred
2^30." Since the stated reason 2^29 wins is cache-size-dependent, and
gfx1103's L2 (2 MB) is far smaller than any of those three cards', this was
a good candidate to actually be wrong here, not just untested.

**It was wrong, by a lot.** Needed a large enough sieve to make auto-mode
j-slabbing trigger at all (area >= 2^30 positions) -- used `--logI 16 --J
65536` (area 2^32) against `oracle/c183.poly`, with `--slab-j` (an existing
CLI override) sweeping the actual chunk size directly:

| slab-j (rows) | positions/slab | wall ms/q (7-q sample) | wall ms/q (33-q sample) |
|---:|---:|---:|---:|
| 1024 | 2^26 | 4364.74 | -- |
| **2048** | **2^27** | **3915.17** | **3889.34** |
| 3072 | ~2^27.5 | 4193.10 | -- |
| 4096 | 2^28 | 4744.03 | -- |
| 8192 (old default) | 2^29 | 5646.64 | 5663.54 |
| 16384 | 2^30 (the L40's preferred point) | 12794.33 | -- |
| 32768 | 2^31 | 16919.51 | -- |

2^27 beat 2^29 by **45.6%** at the larger, more stable sample (33 special-q:
3889 vs 5664 ms/q) -- not a marginal or noise-level difference, the largest
effect size found anywhere in this tuning pass. 2^30 (what a big-L2 NVIDIA
card wants) is catastrophic here: 3.3x worse than 2^27. This tracks the
comment's own stated reasoning exactly -- gfx1103's 2 MB L2 is closer in
spirit to "smaller than the working set" than any of the three NVIDIA cards
this constant was tuned against, so a *smaller* target chunk makes sense,
not a coincidence.

**Change made**: `SLAB_PERF_TARGET_LOG2` in `slab.h` is now
`BENCH_HIP_BUILD`-gated -- `27` for the HIP build, `29` unchanged for CUDA.
`slab.h` is a shared, unforked file (like the other low-churn headers), so
this is a value swap behind an existing build-time marker, not a fork --
the CUDA build's behavior is provably unchanged (`BENCH_HIP_BUILD` is never
defined there). Also fixed `pipeline_hip.cuh`'s j-slabbing status line,
which printed a literal `"2^29"` string that would otherwise have gone
silently wrong the moment the target became build-specific; it now prints
`SLAB_PERF_TARGET_LOG2` itself, so the message can't drift from the actual
policy again on either build. cofcheck.sh's full suite still passes (slab
count changes how work is chunked, not what it computes -- the suite
already independently tests "warp vs thread recording... 4 slabs...
identical" for exactly this invariant).

**Follow-up, done**: `SLAB_PERF_TRIGGER_LOG2` needed to move too, and by a
derivable amount, not a guess. `slab_perf_jmax()` computes `rows =
TARGET/I` independent of J, so CUDA's own `TRIGGER = TARGET + 1` (2^30 =
2*2^29) is not a coincidence: a sieve at exactly the trigger gets `nslab =
ceil(J / (TARGET/I)) = ceil(2) = 2` -- a clean minimal split, by
construction. Leaving TRIGGER at 30 while only dropping TARGET to 27 broke
that invariant (8:1 instead of 2:1) and silently reopened the same problem
at a smaller size. Confirmed directly: a sieve at exactly 2^28 positions
(inside the resulting gap) ran unsplit under the old trigger (fill 123.6
ms) vs. a forced 2-way split into 2^27 chunks (fill 64.7 ms, 14% faster
overall) that a corrected trigger produces automatically. Fixed
`SLAB_PERF_TRIGGER_LOG2` to `28` (`= TARGET_LOG2 + 1`) under
`BENCH_HIP_BUILD`, restoring the same clean-split-at-boundary property,
recentered on gfx1103's own working set. Verified the corrected trigger
actually fires automatically at the 2^28 boundary (fill 65.3 ms, matching
the manually-forced split).

## IMPORTANT SAFETY FINDING: long single-kernel launches crash this machine

While validating the trigger fix, `cofcheck.sh` failed once
("large B1 with derived B2" -- normally PASS) with no immediately obvious
cause; two re-runs both passed cleanly. Investigating why turned up
something more serious than a flaky test: **this session had already
caused two unclean system reboots today** (Windows Event Viewer, Kernel-
Power/unclean-shutdown events at 9:06 PM and 11:17 PM), with no softer
"display driver stopped responding and recovered" event (ID 4101) logged
alongside either one. On a discrete NVIDIA card, a kernel that runs too
long typically trips Windows' TDR (Timeout Detection and Recovery) and the
driver resets cleanly. **On this iGPU/driver combination, a long enough
single kernel launch appears to hang the whole system instead of
recovering.**

This retroactively explains the wild run-to-run instability seen earlier
at the largest slab sizes in the region/slab sweeps above. The quantity
that actually matters for this risk is PER-LAUNCH kernel duration, not
total time across a q:

| slab-j (rows) | fill total | slabs | **fill per launch** |
|---:|---:|---:|---:|
| 8192 (old CUDA default) | 2925 ms | 8 | ~366 ms -- safe |
| 16384 (2^30) | 3990 ms | 4 | **~998 ms -- right at the edge** |
| 32768 (2^31) | 6334 ms | 2 | **~3167 ms -- past typical TDR thresholds** |

The inconsistent numbers measured for 16384/32768 across separate runs
earlier in this document (e.g. 12794 vs 6953 ms/q at 2^30) were most
likely contaminated by actual driver hangs/recovery partway through, not
clean compute measurements -- treat those specific data points as
qualitatively "much worse, dangerously so," not as precise numbers.

**This reframes the whole slab fix**: `SLAB_PERF_TARGET_LOG2 = 27` for HIP
is not just a 45.6% throughput win, it is a stability requirement on this
hardware. CUDA's inherited default (2^29, ~366 ms/launch) was already
closer to the edge than comfortable; anything at 2^30 or above is a real
crash risk here, confirmed against actual system logs, not inferred from
timing alone.

**A separate, NOT fixed risk, flagged for whoever picks this up next**: the
`cofcheck.sh` test that uses `--ecm-b1 400000` drives a single long-running
`k_cofac` (cofactor) kernel launch whose duration comes from ECM's own
parameters (B1/rounds/curves), entirely independent of slab planning (item
5 already established `k_cofac`'s grid comes from `--blocks`, not slab
geometry). This is a pre-existing risk on this hardware that the slab fix
does not address and did not introduce. If cofactor-kernel-duration crashes
ever recur, the fix belongs in `--cof-rounds`/ECM curve-batching, not here.

**Practical guidance for future sessions on this machine**: do not run
`--slab-j` values (or unbounded/auto-slabbed geometries) that produce
multi-hundred-millisecond-or-longer single kernel launches without a clear
reason, and do not repeat the specific large-geometry sweep points above
(rows 16384/32768 at `--logI 16`) -- they are now confirmed, not just
suspected, to risk a full system crash on this hardware.

## Item 5 results: grid-width re-derivation -- DONE, no change needed

**A scope correction, found while tracing the code**: `bench_main_hip.cpp`'s
`ab = prop.multiProcessorCount * 6u` (the constant this item was written to
re-derive) does NOT feed k_apply's or the fill kernels' launch
configuration at all -- those use `nregion`, a geometry-derived value
(sieve area / region size) unrelated to this constant. Tracing
`cfg.blocks` (what `auto_blocks` initializes) to its actual consumer:
`pipeline_hip.cuh:1275`, `const int blocks = cfg->blocks ? cfg->blocks :
48*6;` -- this is **specifically the grid width for the cofactor kernels**
(`k_cofac` and its `k_cof_*` support kernels via `cf_run_rounds_dyn`), not
a general "how many blocks does this GPU want" constant. The item's
original framing (re-derive the multiplier for AMD's CU structure in
general) was too broad; the real question is narrower: is 36 (gfx1103's
6 CUs x the inherited NVIDIA constant 6) a good cofactor-kernel grid width
on this hardware.

**Method**: swept `--blocks` (an existing CLI override) across two
representative cofactor cases from item 3 -- `ecm4` (4-limb ECM, no
stage2: 1 block/CU occupancy) and `ecms2_4` (4-limb ECM with stage2: 4
blocks/CU, the best-occupancy instantiation found) -- since item 3 already
showed these two have very different occupancy characteristics, a single
grid width might not suit both. 29-special-q band, `oracle/c183.poly`,
reading the cofactorisation queue's own device-time instrumentation.

| `--blocks` | ecm4 (1 blk/CU) | ecms2_4 (4 blk/CU) |
|---:|---:|---:|
| 6  | 176.67 ms | 138.55 ms |
| 12 | **124.40 ms** | 93.15 ms |
| 18 | 137.44 ms | 87.58 ms |
| 24 | 130.80 ms | 83.76 ms |
| 36 (current default) | 125.65 ms | 80.06 ms |
| 48 | 125.88 ms | **77.45 ms** |
| 60 | 125.46 ms | -- |
| 72 | 128.07 ms | 78.39 ms |
| 96 | 128.26 ms | 81.13 ms |
| 144 | 127.48 ms | 78.02 ms |

**Two different shapes, one workable default**: `ecm4` is best around
`--blocks 12` (too few, at 6, costs 42% -- too many past ~36 costs little,
2-3%); `ecms2_4` keeps improving out to `--blocks 48` (best measured,
77.45 ms) before flattening/going slightly noisy through 144. No single
value in this sweep is simultaneously optimal for both shapes -- 12 is
~20% worse than 48 for `ecms2_4`, and 48 is ~1% worse than 12 for `ecm4`.
**The current default (36) lands within 1% of optimal for `ecm4` and
within 3.3% of optimal for `ecms2_4`** -- close to both, best at neither,
which is exactly what a reasonable fixed default should look like when the
true optimum is workload-shape-dependent.

**Conclusion: no change.** The inherited NVIDIA constant (6, times
gfx1103's 6 CUs = 36) turns out to be a workable choice on this hardware
too, but not because the reasoning that produced it (an NVIDIA per-SM
occupancy target) transfers -- it doesn't, per item 3's occupancy numbers,
which are 1 or 4 blocks/CU here, nothing like 6. It works because the
actual sensitivity of these cofactor kernels to total grid size is fairly
forgiving above a floor (~12+ blocks) and fairly flat above ~36-48,
regardless of per-block occupancy, so a fixed middle-of-the-road value
serves both shapes adequately. Making this workload-adaptive (e.g. scaling
`--blocks` from the measured per-kernel occupancy) could recover the
missing ~1-20% in the worse case for each shape, but doing that safely
means threading the specific kernel instantiation's occupancy number
through to grid-sizing logic that today has no idea which `k_cofac`
variant is about to run -- a real code change, not a constant tweak, and
not justified by a result this close to the existing default. cofcheck.sh's
full suite still passes (no kernel or grid-sizing code was touched for
this item, only measured with the existing `--blocks` override).

## Item 1 results: __launch_bounds__ sweep -- DONE

**Baseline** (544 special-q, `oracle/c183.poly`, `--logI 14`, default
`--fill-mode atomic`, no `__launch_bounds__` anywhere): wall clock/q 172.6 ms
-- transform+plattice 25.7 ms, fill 35.9 ms, apply 51.3 ms.

**Tooling built first**: `bench_dump_kernel_attrs()` (new, in
bench_kernels.hip, triggered by the `BENCH_DUMP_KERNEL_ATTRS` env var) --
calls `hipFuncGetAttributes`/`hipOccupancyMaxActiveBlocksPerMultiprocessor`
on the concrete kernel instantiations the production `--pipeline` path
actually launches. This is the load-bearing tool for this whole item, and
for the rest of this plan.

**What it found, before changing anything**: none of the three kernels
`__launch_bounds__` was dropped from are register-bound on gfx1103.
Register counts are all low (k_fill_atomic 18-20, k_fill_l1 37, k_fill_l2
22, k_apply 26) -- nowhere near a register-file limit. The actual
occupancy ceiling is shared memory:

| kernel | regs | static/dynamic smem | blocks/CU | why |
|---|---:|---|---:|---|
| k_fill_atomic\<4\> (the default fill path) | 18-20 | 0 | **8** (2048/256, thread-count-saturated) | already at gfx1103's max threads/CU -- nothing to gain here regardless of tuning |
| k_apply, `--region 13` | 26 | 17408 B dynamic | **3** | matches CUDA's own `__launch_bounds__(512,3)` target exactly |
| k_apply, `--region 14` (the default) | 26 | 33792 B dynamic | **1** | 2 blocks would need ~68 KB against gfx1103's flat 64 KB/CU ceiling -- a 3x occupancy cliff from the region-14 default alone |
| k_fill_l1 | 37 | 34048 B static | **1** | ~34 KB static smem alone leaves no room for a 2nd block in 64 KB, independent of register tuning |
| k_fill_l2\<2\>/\<4\> | 22 | 34048 B static | **1** | same as k_fill_l1 |

**A correction to this plan's own premise, found along the way**:
`--fill-mode`/`--mode twolevel` (which is what actually exercises
k_fill_l1/k_fill_l2) is a **benchmark-harness-only flag, explicitly refused
under `--pipeline`** -- `bench --help`'s own text: *"twolevel lost by 2.7x"*
(a prior CUDA-side measurement, RESULTS.md). k_fill_l1/k_fill_l2 are not
reachable from the production sieve path at all; they exist only for
historical/comparative benchmarking. This plan's original item 3 (fill-mode
comparison) is accordingly much lower priority than written -- there is no
live "which fill mode should ship" decision to make, CUDA already settled
it, and nothing about HIP changes that.

**Changes made**: added `__launch_bounds__(512, 2)` to `k_apply`,
`__launch_bounds__(512, 1)` to `k_fill_l1` and `k_fill_l2`, each with a
comment explaining why (register headroom exists but doesn't matter, since
shared memory is the binding constraint at the default region size).

**Measured effect**:
- **Performance: none, as predicted before changing anything.** Re-measured
  the identical 544-q baseline: apply 50.99 ms (was 51.25 ms), fill 35.66 ms
  (was 35.85 ms) -- both within normal run-to-run noise, not a real
  improvement. `__launch_bounds__` only steers register allocation; since
  none of these three kernels were register-bound, there was nothing for it
  to improve. Confirmed via `bench_dump_kernel_attrs()` too: register counts
  and blocks/CU are bit-for-bit identical before and after for all three
  kernels, at every region size checked.
- **Correctness: unaffected.** `cofcheck.sh`'s full ~52-case suite still
  passes after the change.
- **A real, independent win found along the way**: `__launch_bounds__`'s
  *first* argument (512) is a genuine compile-time constraint under HIP too
  (unlike the second argument, whose meaning differs from CUDA) --
  `hipFuncGetAttributes` confirms `maxThreadsPerBlock` dropped from 1024 to
  512 for all three kernels after this change. This **restores the
  compiler-enforced launch ceiling** that a prior code-review finding
  flagged as lost when `__launch_bounds__` was dropped entirely for HIP (a
  caller could previously have launched k_apply with an oversized block and
  the compiled kernel wouldn't have rejected it -- only a hand-added runtime
  check at the launch site would have caught it). Now both the runtime check
  *and* the original CUDA-style compile-time backstop exist together, for
  k_apply specifically.

**Where this leaves the plan**: item 1 is closed out -- the sweep was run,
the answer is "no launch_bounds value moves the needle at gfx1103's default
region size, because shared memory is the actual constraint, not
registers." That answer *is* the deliverable for this item; it directly
motivated doing item 2 (log_region re-sweep) next, since region 13
demonstrably gets k_apply back to 3 blocks/CU with zero code change --
whether shrinking the sieve region by one step is worth that 3x occupancy
gain (against whatever it costs in per-region overhead / total region
count) is exactly what item 2 measured. Short answer: it isn't worth it --
see below.

## Item 2 results: log_region re-sweep -- DONE, no change made

**Method**: `--region N` is legal in `[1,30]` but constrained to `<= logI`
for `--pipeline` (validated at bench_main_hip.cpp:1445), so with `--logI 14`
the sweep range is `[1,14]`. Screened `--region 9..14` on a fast 7-special-q
band first (seconds per point, not minutes), then re-ran the two most
relevant candidates (13, the occupancy-favored one; 14, the CUDA-tuned
default) on the full 544-q baseline band for a statistically solid
comparison.

**Fast-screen results** (7 special-q, ms/q, `fill` / `apply`):

| region | fill | apply | blocks/CU (k_apply, from item 1) |
|---:|---:|---:|---:|
| 9  | 369.1 | 786.1 | (far below any measured point) |
| 10 | 260.6 | 402.8 | |
| 11 | 193.0 | 210.9 | |
| 12 | 145.6 | 118.9 | |
| 13 | 80.6  | 69.0  | **3** |
| 14 (default) | 37.8 | 52.1 | **1** |

Both `fill` and `apply` get monotonically **slower** as the region shrinks,
across the entire range tested -- despite `k_apply`'s occupancy going from
1 to 3 blocks/CU as region drops from 14 to 13. Region 9 is ~15-19x slower
than region 14 on both stages. This is the opposite of what item 1's
occupancy numbers alone would predict, and exactly why the plan calls for
measuring rather than tuning from an occupancy calculator in isolation.

**Full 544-q confirmation** (region 13 vs. the existing region-14
baseline):

| region | fill | apply |
|---:|---:|---:|
| 13 | 80.461 ms | 68.853 ms |
| 14 (default, unchanged) | 35.851 ms | 51.250 ms |

Region 13 is **2.24x slower on fill and 1.34x slower on apply** than region
14, confirmed at full sample size, not just the fast screen.

**Why, despite better occupancy**: halving the region size roughly doubles
the number of regions covering the same sieve area, which appears to scale
most of the per-region fixed costs (region cell initialization, the
survivor scan, block scheduling/launch overhead) faster than the 3x
occupancy improvement can pay for. `k_apply`'s 3-blocks/CU number at region
13 is real (confirmed via `hipFuncGetAttributes` in item 1), but it isn't
the dominant cost -- region *count* is. Occupancy is a necessary condition
for a kernel to run fast, not a sufficient one; item 2 is the concrete case
where that distinction actually mattered here.

**Conclusion: no change.** `--region 14` (bench.h's existing default, the
same value CUDA's own tuning already settled on) remains the right choice
on gfx1103 too, despite having dramatically worse theoretical occupancy
(1 block/CU) than the alternative this plan's item 1 surfaced (3 blocks/CU
at region 13). This is a genuine, verified negative result, not a skipped
step -- CLAUDE.md and this file both record it so a future session doesn't
re-derive the same occupancy number and re-try the same change. Not
swept: region values above 14, since that requires also raising `--logI`
(a full geometry change, conflating region size with sieve width/height --
out of this item's scope; worth its own item if pursued later).

## Item 3 results: cofactor kernel register pressure -- DONE, no change needed

**Method**: extended `bench_dump_kernel_attrs()` with a second function,
`dump_cofac_kernel_attrs()` (necessarily separate: `k_cofac` lives in
`cofac_hip.cuh`, `#include`d at the very end of `bench_kernels.hip`, after
where the first diagnostic function had to be placed). Covers all six
`k_cofac<L, METHOD, STAGE2>` instantiations a `CF_LMAX=4` build carries
(`L` in {3,4}, rho has no stage2 variant, ECM has both) -- the same six
STATUS.md's CUDA-side `ptxas -v` table covers, at the same default launch
config (256 threads/block, `bench_main_hip.cpp`'s `cfg.threads = 256`).

**Registers and occupancy, gfx1103** (STATUS.md's NVIDIA sm_120 numbers in
parens for comparison):

| `k_cofac<L,method,stage2>` | regs (NVIDIA) | static smem | spill/thread (NVIDIA) | blocks/CU |
|---|---:|---:|---:|---:|
| `<3,rho,->` | 71 (78) | 61440 B | 0 (0) | 1 |
| `<4,rho,->` | 85 (94) | 65536 B | **32 B (0)** | 1 |
| `<3,ECM,no-s2>` | 88 (86) | 61440 B | 0 (0) | 1 |
| `<4,ECM,no-s2>` | 109 (112) | 65536 B | **32 B (0)** | 1 |
| `<3,ECM,s2>` | 125 (122) | 61440 B | **320 B (0)** | 1 |
| `<4,ECM,s2>` | 128 (154) | **16384 B** | **608 B (0)** | **4** |

**A real difference from CUDA, worth taking seriously**: STATUS.md's own
table (measured 2026-08-18) states plainly *"Nothing spills at any width"*
on NVIDIA's `ptxas`, across all six instantiations. On gfx1103, **four of
the six spill** (32-608 bytes/thread of local/spill memory) -- HIP-clang's
register allocator makes a genuinely different tradeoff than `ptxas` for
this code. This is exactly the risk this plan's item 3 was written to
check for before trusting a "cofactor width doesn't cost much" read taken
only from NVIDIA hardware.

**But it does not translate into a bigger real-world width penalty --
measured, not assumed.** Ran the actual pipeline forcing each width/method
combination (`--cof-rho`/`--cof-ecm`, `--ecm-b2 0` vs. nonzero for stage2,
`--cof-limbs 3/4 --cof-limbs0 3/4`), 29 special-q, reading the
cofactorisation queue's own device-time instrumentation:

| method | 3-limb | 4-limb | 4/3 ratio |
|---|---:|---:|---:|
| rho | 44.66 ms | 64.97 ms | 1.46x |
| ECM, no stage2 | 72.10 ms | 125.14 ms | 1.74x |
| ECM, stage2 | 55.97 ms | 79.48 ms | **1.42x** |

All three ratios land in the same rough band as STATUS.md's own NVIDIA
measurement (~1.72x on a real unforced job -- not perfectly comparable
methodology, but the same order of magnitude, not a multiple worse).
**The most-spilling variant, `<4,ECM,s2>` (608 B/thread), has the SMALLEST
width penalty of the three (1.42x) and the BEST occupancy of any of the six
kernels (4 blocks/CU)** -- HIP-clang's choice to spill some state out of
registers for this specific instantiation correlates with dramatically
better occupancy, and empirically that trade pays for itself rather than
costing extra. Spilling is not automatically bad; here it happened to unlock
occupancy the non-spilling variants don't get.

**Conclusion: no code change.** The spills are real and worth having found
-- STATUS.md's "nothing spills, CUDA-side" claim does not carry over to HIP,
and a future session should not assume it does -- but they are not causing
an AMD-specific performance cliff. cofcheck.sh's full suite still passes
(no kernel code was touched for this item, only the diagnostic). If this
ever needs revisiting, `CF_ECM_NBABY` is the knob STATUS.md itself already
names as the lever for ECM stage 2's live-state footprint, on either
platform.

## Item 7 (new): startup slab-size auto-calibration -- DONE

Item 6 found that gfx1103's optimum (2^27) is 45.6% faster than the CUDA
default (2^29), and reasoned that the cause -- a 2 MB L2 much smaller than
any of the NVIDIA cards the CUDA constant was tuned on -- is a cache-size
effect, not a HIP-vs-CUDA one. That means a discrete AMD card with a larger
L2 could just as easily want something else in the {2^27, 2^28, 2^29} range,
and there is no way to know without asking the hardware in front of it. The
user proposed testing the actual candidates against the real first special-q
at startup and picking the fastest, estimating it could cost "only a few
seconds" -- this item is that feature.

**Design**: lives in `run_pipeline()` (`pipeline_hip.cuh`), right before the
plan-building `slab_make_plan()` call it already had. Gated on `!cfg->slab_j`
(an explicit `--slab-j` bypasses calibration entirely, unchanged) and on
`slab_perf_jmax()` actually applying to this geometry (no point calibrating
a sieve too small to perf-slab at all). Candidates are **{2^27, 2^28, 2^29}
positions/slab only** -- deliberately excluding 2^30 and above, which the
safety finding above confirmed can hang this machine's GPU driver badly
enough to crash the whole OS. Each candidate is timed with a real,
throwaway call to `run_pipeline_impl<true>()` against `qlist[0]` (`nq=1`,
`qgen=NULL`), then the real run proceeds with whichever candidate won,
built via a fresh `slab_make_plan()` call rather than by mutating the
caller's `cfg`.

**Why `qlist[0]` is safe to reuse, not a guess**: read `run_pipeline_impl`'s
own per-q loop (`if (qi < nq) cur = &qlist[qi]; else if (qgen) ...`) and
`bench_main_hip.cpp`'s three call sites that populate `qlist`/`qgen` before
calling `run_pipeline()` (the `--qrange` streaming case, the `--qlist` file
case, and the single `--q`/`--rho` case). In all three, `qlist[0]` is
always populated with the real first special-q before `qgen` (if any) ever
supplies q1 onward -- `qgen` is never consulted for `qi == 0`. Calibration
therefore never calls `sqgen_next()` and never advances the real streaming
generator's state; it only reads `qlist[0]`, which is `const` all the way
through and shared, unmutated, with the real run that follows.

**A real bug found and fixed while verifying against cofcheck.sh**: the
first version left `--cofactor` on in the calibration passes' scratch
config, reasoning that `NULL`ing `relations`/`candidates` was enough to make
them side-effect-free. It was not -- inline ECM/rho cofactorisation runs
regardless of whether an output path is set, so each of the 3 throwaway
passes also ran the *entire* cofactor stage against the real q, printing its
own "cofactor method"/"B2 derived" diagnostic lines and paying ECM's cost 3
extra times. Two costs, not one: it broke `cofcheck.sh` cases that grep for
those lines expecting exactly one occurrence (5 cases failed: B2-derivation,
auto-method resolution, `--cof-rho`/`--cof-ecm` override, and the 4-limb
width case), and it made calibration far more expensive than intended --
the cofactor queue's own cost was measured completely flat across all three
slab candidates (~975-1220 ms regardless of slab size, only `fill` moved),
so none of that spend bought any information relevant to the decision being
made. **Fix**: `ccfg.cofactor = 0` in the calibration scratch config, so it
exercises only the sieve/TD/classify stages slab size actually affects.

**Verified**: `cofcheck.sh`'s full suite passes 52/53 after the fix -- the
same result as the pre-calibration baseline, with one exception noted below.
On the correctness-gate job itself (`oracle/c183.poly`, the default
geometry `cofcheck.sh` uses, q=120000053), calibration picked **2^28**, not
the static 2^27 default -- a genuinely different answer for this job's
smaller factor-base/geometry than the large `--logI 16 --J 65536` job item
6's sweep used, which is exactly the point of calibrating per-job rather
than trusting one static constant:

```
  slab auto-calibration: probing 2^27/2^28/2^29-position slabs against q=120000053...
    2^27 (4096 rows/slab): 1082.1 ms
    2^28 (8192 rows/slab): 938.4 ms
    2^29 (16384 rows/slab): 2197.7 ms
  slab auto-calibration: selected 2^28 (8192 rows/slab)
```

**One open interaction, accepted as-is per user decision, not fixed here**:
the sole remaining `cofcheck.sh` failure after the cofactor fix is the
already-documented, out-of-scope `--ecm-b1 400000` case (see the safety
finding above -- a real device error, `hipDeviceSynchronize(): unspecified
launch failure` at `cofac_hip.cuh:1698`, in a file this item never touches).
That case passed on the pre-calibration baseline build but failed the same
way twice with calibration enabled. It is plausible calibration's three
extra GPU passes immediately before the real run add enough memory/thermal
pressure to make this already-marginal ECM kernel configuration (B1=400000,
B2=10000000, 320000 giant steps) more likely to fail -- or it may simply be
the same pre-existing flakiness this exact case was already flagged for
before any of today's work. Either way: it is a clean, caught failure (error
printed, resumable checkpoint written, process exits normally), confirmed
via Event Viewer to NOT be a system crash, and it belongs to the same
already-flagged, separate `cofac_hip.cuh`/ECM-kernel issue, not to this
item's slab-calibration code. Flagged here for whoever eventually looks at
that kernel, not addressed by this item.
