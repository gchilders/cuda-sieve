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
