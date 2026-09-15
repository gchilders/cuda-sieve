# cuda-sieve — Metal port

Porting the CUDA NFS lattice sieve in `bench/` to Metal Shading Language for
Apple Silicon. The plan of record is **`bench/METAL_PORT_PLAN.md`** — read it
first; it also carries the drift ledger. This file carries the rules.

Branched from `main` at `3e15fec`. `hip-port` is a reference for method, not
a base: it is 23 commits behind `main`.

## This machine
- Apple M3 MacBook Air (fanless), 10-core GPU, `MTLGPUFamilyApple9`, 16 GB
  unified, 8 CPU cores (4P + 4E).
- macOS 26.5, Xcode 26.5 (17F42), Metal Toolchain 17F42.
- `xcode-select -p` points at CommandLineTools, which ships **no** `metal`
  compiler. Do not `sudo xcode-select -s`; export
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` instead.
- **Performance tuning on this box is allowed** (changed 2026-09-14, was
  "correctness vehicle only"). This is a full M3, not a crippled iGPU: fp32
  and fp64-by-emulation behave like the real thing, the GPU is a normal
  10-core Apple GPU, and a measurement here is a real measurement of Apple
  silicon. Tune here, and ship the results.

  What still has to be said out loud with any number from this box:
  - it is a **10-core M3 in a fanless MacBook Air**, so it throttles under
    sustained load and it is the small end of the Apple GPU range (an M3 Max
    has 40 cores);
  - the GPU **also drives the display**, so anything measured while the
    machine is in use has interference in it;
  - memory is a **shared UMA pool**, so working-set effects differ from a
    discrete card with dedicated VRAM.

  So: measure, tune, and ship — but record the machine alongside the number,
  and treat a result as "measured on a 10-core M3" rather than "measured on
  Apple silicon". Where a tuned value becomes a shipped default, say which
  box produced it, exactly as the HIP port's `SLAB_PERF_TARGET_LOG2` entry
  does.

## Ground rules
- The CUDA build stays authoritative. Every CUDA-side edit gets a row in the
  drift ledger below saying what changed and **how far it was actually
  verified**. An unrecorded CUDA-side edit is the failure this rule exists to
  prevent; a stale row is nearly as bad.
- Acceptance gate: `cofcheck.sh` green plus relation comparison against the
  CUDA build on the oracle jobs. Whether that can be *byte*-identical is open
  and belongs to Phase 7 — see below.
- Warp width 32 is assumed throughout the source. Phase 0 confirmed it holds
  on Apple GPUs. Do not "fix" it.
- `grep -rn __CUDACC__` across the **whole** tree before touching a shared
  header. The HIP port learned this the expensive way: reading a file's
  outline is not a substitute for compiling it end to end.

## Confirmed device and language facts (probe, 2026-09-14)
Full tables in `bench/METAL_PORT_PLAN.md` section 5.3. The four that change
design decisions:

1. **No `double`.** The compiler rejects it outright. The two device-side
   fp64 sites (`bench_kernels.cu:525` norm fallback, `td.cuh:981` →
   `prp.cuh:194` `cof_classify`) need soft-fp64.
2. **No 64-bit atomics, at all.** Not add, not min/max.
   `__HAVE_ATOMIC_ULONG__` is never defined under `-std=metal3.2` or
   `metal4.0`. The two `unsigned long long` `atomicAdd` sites
   (`bench_kernels.cu:120`, `cofac.cuh:1554`) must be reworked.
3. **`maxThreadgroupMemoryLength` = 32768 B**, against CUDA's ~100 KB opt-in
   and AMD's 64 KB. `apply_smem` (`pipeline.cuh:285`) forces
   `log_region <= 13`; the shipped default is 14 (`bench_main.cu:971`).
4. **`metal::log2` differs from host `log2f` on 50.03% of inputs, by up to
   3 ULP**, and `precise::log2` is bit-identical to it — the `precise::`
   namespace is not a fix. Measured with `-fno-fast-math` already set.

5. **Apple GPUs flush subnormals to zero.** fp32 `/` and `fma` are otherwise
   correctly rounded and bit-exact against the host — over 2^20 samples each,
   **zero** disagreements where neither side was subnormal, and every
   disagreement that did occur was FTZ. This is what lets `portable_log2.h`
   use `/` and `fma` directly. It is also a live hazard for `k_apply`'s fp32
   Horner, whose `s = fabsf(acc)` is deliberately a cancelled quantity; the
   existing `fmaxf(s, 1e-30f)` clamp and `NORM_CANCEL_TOL` guard look
   sufficient, but Phase 5 must confirm that rather than inherit it.

Everything else checked out clean: SIMD width 32, `simd_ballot` in ascending
lane order (so `td.cuh:647` holds), `simd_shuffle_up` matching CUDA,
`mulhi(ulong,ulong)` exact against `__int128`, templated kernels with
`[[host_name]]`, and `device` pointers inside argument-buffer structs
(Tier 2).

## Support floor: M1 (Apple7) on macOS 13
Set 2026-09-14. `METAL_MIN_MACOS` in `bench/Makefile.metal`, default 13.0,
with `-std=metal3.0`.

MSL 3.0 is this toolchain's floor, not a preference: Xcode 26.5's Metal
compiler lists `metal2.0`-`metal2.4` in its own `-std` help text and then
refuses every one of them. MSL 3.0 requires macOS 13 at runtime. Reaching
macOS 11/12 would mean building the metallib with an older Xcode (14.x emits
MSL 2.4) — a toolchain problem, not a source problem, since nothing here uses
a feature newer than MSL 2.x.

**This costs no hardware coverage.** Every Apple silicon Mac ever shipped runs
macOS 13 or later, so M1 through M4 are all in range; it excludes only an M1
deliberately held back on Big Sur or Monterey.

`metal_rt` now **fails closed** on anything that is not `MTLGPUFamilyApple7`
or later, and cross-checks `threadExecutionWidth == 32` at the first launch.
Metal also runs on Intel Macs with AMD (64-lane) or Intel (8-lane) GPUs; those
would not crash, they would quietly compute a different factor base. Refusing
them by name is the only safe behaviour given the port's warp-32 assumption.

**Untested on real M1 hardware** — the floor is set by what the toolchain
accepts and by `supportsFamily`, not by a run on an M1. Anything family-gated
(`mulhi(ulong,ulong)`, argument buffers) should be re-verified the first time
an M1 or M2 is available.

## Measured performance (10-core M3, 2026-09-14)
Factor-base generation, `oracle/c183.poly`, `--maxbits 15`, output
**byte-identical** between the two at every point:

| `lim` | CPU `fbgen`, 8 threads | Metal `fbgen_gpu` | speedup |
|---|---|---|---|
| 2,000,000 | 0.24 s | 0.16 s | 1.5x |
| 5,000,000 | 0.58 s | 0.33 s | 1.8x |
| 134,200,000 (C183 production `alim`) | 18.1–18.8 s | 7.16–7.25 s | **~2.5x** |

7,605,616 entries, 110 MB, in 7.1 s. GPU fixed startup is 0.04 s, so the
crossover is around `lim` ≈ 1-2M and the advantage grows with the bound.

Context for expectations: this is a 10-core M3 against 8 CPU cores of the same
chip — roughly the ratio you would expect from that hardware, not a
disappointing GPU result. The HIP port's gfx1103 iGPU did comparable work in
8.9 s.

## Status
- **Phase 0 (toolchain + probe): DONE.** Probe sources in `metal-probe/`.
- **Phase 1 (branch + ledger): DONE.**
- **Phase 2 (portability primitives): DONE, gate green.**
  `cd bench && make -f Makefile.metal metalcheck` — 6.6M results compared,
  0 mismatches. `softfp64.h` (IEEE binary64 in integer ops),
  `portable_log2.h`, `msl_compat.h`, `sf_sites.h`.
- **Phase 3 (metal_rt runtime shim): DONE, gate green.**
  `cd bench && make -f Makefile.metal rtcheck` — 17 checks.
  `metal/metal_rt.h` is plain C++; `metal/metal_rt.mm` is the only
  Objective-C++ in the port. Ported orchestration compiles as ordinary C++.
- **Phase 4 (fbgen_gpu + CUB replacement): DONE, gate green.**
  `cd bench && make -f Makefile.metal fbcheck` — `fbgpucheck.sh`, 19 cases,
  all byte-identical to the CPU generator. Also `make -f Makefile.metal
  scancheck` for the scan/select primitives alone.
- **Phase 5 (sieve kernels): DONE, gate green.**
  `cd bench && make -f Makefile.metal sievecheck`. At logI 13 / J 4096 /
  lim 1e6 on c183 with the oracle's special-q: all 4,096 regions of fill and
  all 4,194,304 cells of apply match the CPU reference exactly, with 623,098
  cells over the survivor threshold on both sides. Shared struct layouts are
  cross-checked first.
- **Phase 6 (TD + cofactorisation): cofactoriser DONE and hitting the golden
  number; TD and the pipeline still to do.**
  `make -f Makefile.metal cofaccheck` — `run_cofac()` on the oracle's own
  candidate list for the parity special-q gives **37 relations in all four
  configurations** (rho and ECM x 3-limb and 4-limb). 37 is what `cofcheck.sh`
  pins and what las finds at this q.
  `make -f Makefile.metal argbufcheck` — the `cofq_t` argument-buffer
  mechanism, proven with a negative control.
  **The device side of the port is COMPLETE: 84 kernels in one metallib**
  across `bench_kernels.metal`, `td.metal`, `fbgen_gpu.metal`, `cofac.metal`
  and `scan.metal`, every translation unit compiling with zero errors. MSL
  copies of the shared arithmetic headers are generated from the untouched
  originals: `bigint_msl.h`, `prp_msl.h`, `plattice_msl.h`, `slab_msl.h`,
  `td_msl.h`.
  `pipeline.cuh` and `bench_main.cu` are ported, **`./bench` links, and a
  `--pipeline` band runs end to end on the M3** (`make -f Makefile.metal
  benchbin`). `--verify-only` passes through the real binary.
  **`cofcheck.sh` PASSES: 54 PASS, 0 FAIL, exit 0** —
  `make -f Makefile.metal cofcheckgate`. That is the formal gate the HIP port
  used. **Phase 6 is complete.**

**THE METAL BUILD DEFAULTS `log_region` TO 13, not CUDA's 14.** Apple's
threadgroup ceiling is a hard 32 KB with no opt-in tier, and at 14 `k_apply`
wants 32,896 B — 128 bytes over. `--region` still overrides; the CUDA build is
untouched.

**13 is the measured optimum, and region 14 is no longer refused (plan
8e/8f).** The 128-byte slice-log copy is gone — `k_apply` reads the table from
device memory — so region 14 needs exactly 32,768 B and runs. It is *slower*:
apply ~393 ms against ~342 at region 13. At 14 a threadgroup asks for the
entire 32 KB budget, so one fits per core, and that occupancy cliff costs more
than the halved region count saves. `fill`, which has no region-sized
threadgroup array, does improve at 14 exactly as predicted (136 vs 148 ms) —
the two halves of the sieve want opposite things and apply is the bigger one.
Dropping the copy was worth **-4.2% on apply** on its own (three runs each way,
non-overlapping ranges), and the host-side length now has ONE definition,
`mtl_apply_smem()` in `metal_rt.h`, because three copies of it existed and a
threadgroup length that disagrees with the kernel is a wrong answer, not a
compile error.

**Below 13 everything is worse (plan 8e).** Regions below 13 are monotonically much worse: apply is
349.6 ms at 13 and 1497.8 at 10, fitting `182.3 ms + 20.07 us per region`
with residuals under 4 ms. Each bucket region costs ~20 us of fixed overhead,
so halving the region doubles how often that is paid; freeing threadgroup
memory buys nothing. **Do not lower `--region` looking for occupancy.** The
same fit extrapolates region 14 at ~22% off the sieve stage. That extrapolation was
tested and **refuted** — see above and plan 8f. The per-region cost is real,
but the model was fitted entirely below the occupancy cliff and had no term
for it.

**Two traps this port fell into; do not repeat them.**
1. `NULL` is `0L` in C++, NOT a pointer. Passed to a binding template it takes
   the non-pointer branch and binds eight bytes of zeros as a constant buffer,
   so the kernel's `if (p)` sees a good address and faults. Use `nullptr`,
   which has its own overload in `metal_rt.h`.
2. A `threadgroup` array that was a static `__shared__` array in CUDA must be
   DECLARED IN THE WRAPPER KERNEL, not turned into a `[[threadgroup(n)]]`
   parameter — a parameter is zero-length unless the host sets its length, and
   CUDA's version needed no host involvement at all.

**The thread-to-SIMD-lane mapping is verified** (`metal-probe/lanemap.metal`):
`thread_index_in_simdgroup == tid & 31` and
`simdgroup_index_in_threadgroup == tid >> 5`, so CUDA's `lane`/`warp`
arithmetic and `__syncwarp` scope are valid as written. Phase 0 checked ballot
ordering and shuffle semantics but not this.

**`cof_classify` is verified on device** (`classifycheck`): 0 of 65,536
verdicts differ from `prp.cuh`'s own fp64. The soft-float path is ruled out of
the remaining candidate-count problem.

**The `cofq_t` argument buffer is wired but NOT yet exercised.** `run_cofac`
never reaches `k_cof_enqueue` or `k_rel_pack`; `cofcheck.sh` will be the first
thing that runs them. Pass such a struct as `mtl_argbuf_t` — it binds the
buffer and calls `mtlUseResource` on every pointer inside in one step, so a
call site cannot do one and forget the other.
- **Phase 7 (relation comparison): DONE. The relations are BYTE-IDENTICAL to
  the CUDA build over 288 special-q.** Run on a real **GTX 1080 Ti (sm_61)** in
  an NRP/Nautilus k8s pod (`nvidia/cuda:12.8.1-devel-ubuntu22.04`, nvcc
  12.8.93, `main` at 3e15fec, `GPU_ARCH=61`). The pod regenerated `c183.fb1`
  to the manifest hash first, so both builds sieve identical input.

  Two comparisons, both `cmp`-clean at 13,485 relations: **(A)** matched
  settings, and **(B) each build at its OWN defaults** — CUDA region 14 /
  12c x 4r / 168 blocks against Metal region 13 / derived 2c x 24r / 512
  blocks / 4 auto-calibrated slabs. **B is the one that matters**: every
  default this port changed, all at once, and the output is the same bytes in
  the same order. 564,696 enqueued and side 0 477,071/87,625/0 on both. Side 1
  dead/stuck differs 550,145/260 vs 550,176/229 — the same ±31 8o measured
  between 12x4 and 2x24 on Metal ALONE, so it tracks the curve schedule, not
  the platform, and neither number is a relation.

  **THE log2 DECISION IS SETTLED: do nothing.** Phase 2's 3-ULP divergence
  reaches no relation — 0 of 4.2M cells in Phase 5, now 0 of 13,485 relations
  against real CUDA. `-DNORM_PORTABLE_LOG2` on both builds stays available as
  insurance against a future toolkit or vendor; it fixes nothing currently
  broken. The same run retires the region-13-vs-14 worry (that IS comparison B)
  and soft-fp64 in `cof_classify`.

  **Caveat:** one card, one composite, one band, at B1 2000/B2 60000 and
  lpb 31/32. Does not prove byte-identity at logI 16, which CLAUDE.md already
  flags as differentially untested on CUDA too.

- Phase 7 background (superseded by the above): **largely answered by Phase 6's gate.**
  `cofcheck.sh` pins ~25 relation counts derived from the CUDA build and the
  Metal build matches every one; the 37 relations at the parity special-q are
  the identical (a,b) set as las's. The 3 ULP `log2` divergence has not moved
  a relation. What remains is a full band rather than a single q.
- **Phase 8 (tuning): DONE except the two-level fill defect.** The two-level fill's threadgroup-ceiling
  problem is SOLVED — `L1_CAP`/`L2_CAP` retuned 64 -> 61, measured against the
  driver's own refusal at 33,796 B — but the path **misplaces records across
  regions** (right total, wrong distribution) and therefore **`--mode
  twolevel` refuses on Metal**. Not the production path; apply needs
  single-level 4 B records. Cause not found; cap sweep, vote emulation and
  lane mapping all ruled out. Threadgroup-size tuning, `--cof-chunk`
  and slab sizing are all measured and written up (plan 8a-8d).

  **Slab size is now auto-calibrated at startup (plan 8g), ported from the
  HIP port.** It times the run's real first special-q against 2^25/2^26/2^27
  slabs and keeps the fastest, so the static constant below is a fallback
  rather than the answer. It reproduces 8c's 2^26 from scratch here, and beats
  the static default by 3.8% at `--region 12`, where the region-relative
  target drifts but the hardware's preference does not. Skipped when
  `--slab-j` is given or the geometry is below the trigger — which is also why
  `cofcheck.sh` is undisturbed.

  **THE BOINC HAZARD IS REAL AND IS GATED.** A calibration band is one q, which
  reads as 1/1 = 100% done, clamps to 0.99, and — because BOINC reports must be
  nondecreasing — becomes the floor for the whole workunit. The HIP port
  shipped that; volunteers saw tasks pinned at 99% for hours.
  `bench_boinc_progress_suspend()` drops those reports **before** the
  high-water mark, and `make -f Makefile.metal boinccheck` proves it — with a
  control that reproduces the bug, in its own process, because the mark is a
  static with no reset. `HAVE_BOINC` defaults to 0, so that path is compiled
  out of every ordinary build and NOTHING else in the tree can reach it. That
  is exactly how it reached the field. Do not delete that gate.

  **Slab sizing was the one real win: 8 slabs instead of 2, -44% on the
  sieve.** `slab.h`'s target is in bucket *regions*, so this build's
  `--region 13` silently quartered CUDA's intent; `SLAB_PERF_REGIONS` is now
  overridable (default unchanged) and set to 8192 here.

  **A LAUNCH-DURATION BOUND IS NOW POLICY: 750 ms (plan 8k).** Set by the
  user for UI responsiveness as much as watchdog safety.
  `COF_CHUNK_TARGET_MS` is 750 here, compared against 8k's MEASURED launch
  (CUDA's 250 is against a whole-side sum -- not comparable). Auto opens at one
  core-derived grid and descends, with a **no-progress guard** that stops when
  a halving stops paying, because a bound can be unreachable and chasing one is
  how 8j's unconditional slowdown comes back.

  **`--cof-chunk` CANNOT bound a launch below one ECM chain.** Measured: launch
  time is flat at ~1495 ms below 1920 records while cofac/q rises 10x. The
  lever is `--ecm-curves`, which is PER ROUND: **8 curves x 24 rounds holds
  740 ms and is 24% FASTER than 48 x 4** (354.6 vs 467.2 ms/q), same 192-curve
  budget, identical relations. **VALIDATED over 288 special-q (plan 8l): the
  relation sets are IDENTICAL** — 13,485 each, zero unique to either, identical
  split/dead/stuck, from the identical 564,696 candidates. 160 of the 192
  sigmas differ and it changes nothing, because at B1 2000 the differing sigmas
  never split anything: 8 curves x 1 round already gives 6,723 of the 6,724
  relations that 24 rounds do, so a 192-curve budget does ~8 curves of useful
  work here. **A property of this job's parameters, not a theorem** — at a
  larger B1 the two would diverge. Recommended as job settings, **still not** a
  default, since it changes which sigmas run (`sigma = c0*1000 + cv + 6`).

  **The bound is cheap to meet and expensive to miss.** Over 288 q, 8x24
  settles at 3,840 records/launch and costs 277.4 ms/q — faster than the 465 it
  cost before the bound existed. 48x4 cannot meet it (one curve is ~92 ms, so
  48 is ~4.4 s however the records are sliced), descends, the no-progress guard
  parks it at 3,840, and it pays **835.3 ms/q against 465**. `cofq_init` now
  warns at startup and names the split that would fit; it advises, it does not
  act.

  **`--ecm-b1 400000` CRASHED THIS MACHINE TWICE.** WindowServer crash plus
  userspace watchdog timeout, 1m47s and 1m48s into `cofcheckgate`, same case
  both times, idle machine. B2 derives to 10^7: 320,000 giant steps, ~15.9 s in
  ONE curve, ~190 s in one launch. `cofac.cuh`'s own warning block names this
  configuration and says **cofcheck.sh skips it on HIP** — this port did not,
  and on Apple silicon the GPU drives the display, so it is a dead session and
  not a failed task. `cofq_init` now REFUSES above `COF_LAUNCH_REFUSE_MS`
  (10 s/curve), and `cofcheck.sh` asserts that refusal on Metal. **Do not
  remove either.** The threshold is a judgement: 3.6 s and 6.9 s launches run
  fine here, 15.9 s kills the box, and nobody has measured the real line.

  **CURVES-PER-ROUND IS NOW DERIVED, NOT PINNED (plan 8n).** When
  `--ecm-curves` is not given and both sides are ECM, the build aims at **2
  curves per round** and raises rounds to keep the caller's total curve budget.
  Swept at constant budget, cofac ms/q by curves/round: 16c 353.5, 8c 265.3,
  6c 252.3, 4c 212.5, 3c 192.8, **2c 180.0**, 1c 181.0 — a bracketed interior
  minimum at two, and 8l's recommended 8x24 is 47% worse than it. Every round
  re-compacts the live list, so a fine split drops already-split records before
  the expensive rounds; at one curve the five per-round kernels cost more than
  that saves.

  **The rule is NOT "the largest count that fits the bound"** — that picks 8 at
  B1 2000 and costs 265 against 180. The bound is a ceiling, not an objective:
  aim at 2, let the bound lower it (B1 32000 derives 1 x 48). At the default
  48-curve budget this takes the launch from 1687 to **458 ms**, cofac/q from
  286.9 to **160.1**, wall from 1005 to **860.7**, relations unchanged, six
  gates green including cofcheck's pinned counts.

  **VALIDATED over 288 q (plan 8o): identical relation sets** — 13,485 each,
  zero unique to either, from the identical 564,696 candidates, at **cofac/q
  348.3 -> 153.5, wall 1068.6 -> 856.0, launch 1669 -> 467 ms**. Unlike 8l the
  two configs share only 8 of their 48 sigmas, so this was not forced — and
  side 1's split count is identical (14,291) while dead/stuck move by 31,
  proving the sigmas really did differ and the same cofactors factored anyway.
  **At B1 2000/B2 60000 the relation set is robust to WHICH sigmas are tried**
  — only how many, and how cheaply. That is what makes the re-compaction win
  free. At a B1 where the marginal curve decides relations they would diverge.

  **ECM ON c183 SATURATES AT B1 ~ 500 (plan 8p, §10).** Relations are 6,724 at
  B1 500, 1000, 2000 AND 32000, and 6,719 at 200 — production B1 2000 is 4x
  past the point where effort buys anything, at 2.3x the cofactor cost; B1
  32000 is 64x past it at 20x the cost. **That is why every sigma comparison
  came back identical**: ECM was running far beyond its binding constraint.

  **B1 IS DELIBERATELY NOT CHANGED.** It comes from `cof_auto_b1`, shared by
  every port, so moving it is a CUDA+HIP+Metal decision about the mathematics,
  not a Metal tuning change. Recorded in plan §10 as evidence for that
  decision; this build's B1/B2 handling is byte-identical to `bench_main.cu`.
  **The line: a port may reshape the SCHEDULE to fit its hardware (8n's derived
  curves-per-round — same budget, same B1, identical relation set over 288 q)
  but may not retune the SCIENCE.**

  **Curves matter BELOW saturation, not above** — raising B1 makes the sigma
  choice matter *less*, not more. At B1 200 the derived default loses 24 of
  13,485 relations (0.18%) and gains none, because `mz_split` restarts its
  factor stack every round, so a cofactor needing two factors peeled can be
  split by 12 curves in one round and not by 2 in each of six — exactly the
  hazard `cofac.cuh`'s record-axis comment warns about. It is still the better
  default there (**+28% relations per second of cofactor time**) and loses
  nothing at or above saturation. **Below saturation, pass `--ecm-curves`
  explicitly** to suppress the derivation.

  An explicit `--ecm-curves` is never overruled (it gets the advisory instead),
  and the derivation is gated on both sides being ECM because `cofq_flush`
  passes one round count to both and raising it under rho hits the
  `budget << r` overflow.

  **`--cof-rounds` now allows 1000 for ECM (plan 8m); rho keeps 24.** The cap
  was rho's: `budget << r` appears only in the rho launch, while ECM passes
  `S->curves` unshifted and the round index only picks a 1000-wide sigma block.
  The overflow it named is already guarded exactly and only for rho, by
  `bench_main`'s uint64 test on the ACTUAL budget — at the default 65536 the
  real rho limit is **16**, not 24. Lifting it matters because holding 8k's
  launch bound needs more rounds of fewer curves as B1 grows: at B1 8000,
  2c x 96r against 8c x 24r is **launch 4155 -> 1594 ms, cofac/q 765 -> 596,
  identical relations**. `--ecm-curves` is now bounded at 994 too — sigma
  blocks are 1000 wide, so more repeats the next round's sigmas.

  **The cofactor grid is now sized from the work (plan 8j):**
  `max(multiProcessorCount * 6, ceil(CQ_FLUSH / threads))` = 512 blocks here,
  and `cof_chunk_floor()` is decoupled from it so subdivision stays reachable
  on a slow part. **But `--cof-chunk auto` is NOT adaptive on this hardware and
  never was:** it halves whenever `stage > COF_CHUNK_TARGET_MS`, and `stage` --
  a whole side's device time summed over rounds and slices -- is 39-45x that
  250 ms target at EVERY chunk size, so it parks at the floor forever. An
  always-true test is not a safety mechanism. Harmless on CUDA (there the floor
  is a full grid, so parking is free); costs **25% of the cofactor stage** here.
  `--cof-chunk 131072` gets it back today: 348.5 vs 465.5 ms/q, wall 1085 vs
  1202. Fixing auto means retargeting a watchdog constant for macOS — open
  decision, see 8j.

  **`--threads` 256 is right. `--blocks` was NOT (plan 8i).** Its default is
  `multiProcessorCount * 6` = **60** on a 10-core M3 = 15,360 threads, against
  a `CQ_FLUSH` batch of 131,072 records — a grid 8.5x smaller than the work.
  CUDA's formula works only because NVIDIA SM counts are large (a 4090 gets
  196,608 threads from it). Consequence: `--cof-chunk` auto picks its floor,
  `blocks * threads`, and splits every round into 9 launches — **the worst
  point measured**. `--blocks 576` clears CQ_FLUSH, stops the chunking, and is
  worth **-25.5% on the cofactor stage and -10.9% on wall** at `--nq 72`, with
  identical relations. It is also a SHORTER launch, not a longer one, so the
  watchdog margin improves. Recommended fix: size the grid from the work
  (`blocks * threads >= CQ_FLUSH`), not the core count. NOT yet applied.

  8d's "--blocks is flat, do not hand-tune" was measured at `--nq 1`, where
  1,852 records leave even a 60-block grid oversubscribed. **Flatness measured
  on a starved stage says nothing about the stage when fed.**

  **The cofactor stage is ~39% of wall at a production band size, and is
  SMALLER than the sieve (464 vs 537 ms/q).** The 59% figure below came from a
  one-q band and is an artifact: `CQ_FLUSH` is 131,072 records (~67 special-q)
  and one q hands the queue 1,852, i.e. 1.4% of a batch. Per-q wall is 1183 ms
  at `--nq 72` against 2572 ms at `--nq 1`. **Never quote a queued stage's cost
  from a single-q run** — plan 8h. The sieve is unaffected either way
  (527-546 ms/q at every band size).

  **The cofactor stage is immune to launch geometry** --
  under 2% across a 16x range of threadgroup sizes and 115x of counts. It is
  critical-path bound, not occupancy bound: halving the records in a launch
  leaves the launch's cost unchanged (1508 -> 1540 ms). Grid shape cannot
  shorten a dependent chain, so do not spend effort there; the levers are
  `--ecm-curves`/`--ecm-b1`/`--cof-rounds`, which change the mathematics.
- Phase 9: not started (packaging).

**Candidate counts do not compare across sievers; relation sets do.** Our 1,845
cofactorisation candidates against the oracle's 1,851 is not a defect: the 7
las has that we lack are exactly the relations that need no cofactorisation
(las dumps every post-sieve survivor, we dump only what enters the
cofactoriser), and the 1 we have that it lacks is a marginal survivor at
`i = -16384` under a deliberately looser allowance. Plan section 5b has the
derivation.

**Residency is load-bearing and easy to get wrong.** Any pointer reached
through a GPU address (i.e. inside an argument-buffer struct) must have
`mtlUseResource()` called for it between `mtl_launch_begin` and
`mtl_launch_end`, or the kernel reads garbage rather than failing cleanly —
4,095 of 4,096 wrong in the control. Note also that **residency persists once
granted within a process**, which is why `argbufcheck` runs its negative
control FIRST; a control placed after the positive case passes and proves
nothing.

**`grep -rn __CUDACC__` BEFORE porting any shared header — it bit here exactly
as the HIP port's ledger warns.** `cofac.cuh` wraps its whole GPU host driver,
`run_cofac` included, in `#if defined(__CUDACC__)`. Compiled as ordinary C++
that block vanishes, so the file built with ZERO errors and then failed to
link. Note the two guards want opposite treatment: the `CF_FN`/`CF_HD` block
needs the *non*-CUDA branch, everything else needs enabling.

**A regex over `#if` blocks silently deleted half a header — do not do that.**
Reducing `#if defined(__CUDA_ARCH__)` with a regex assumed a bare `#else`;
`bigint.cuh`'s block has an `#elif defined(_MSC_VER)`, so the regex orphaned
it and turned the rest of the header into a dead branch. It still COMPILED,
and the symbols simply ceased to exist, surfacing two translation units away
as "unknown type name". `gen_msl_headers.py` now walks the conditionals
properly and asserts they balance. Any future guard rewriting must do the
same.

**No hand-copied constants remain.** `metal/td_host.h` and `td_msl.h` are both
generated from `td.cuh` and lift its guarded `TD_*`/`TDF_*` names; the
launch-shape constants `bench_host.cpp` needs are lifted from
`bench_kernels.cu`. Keep it that way — lift by name, never retype.

**`metal/portlib.py` holds the ONE copy of the launch rewriter and the
cuda*->mtl* table.** Two copies of that transformation would drift, and a
drifting transformation produces a file that compiles and computes something
else. Add to portlib, do not fork it.

**Two CUDA-side observations this phase raised** (plan §10, nothing changed):
`verify_count_updates` walks in 32 bits while `k_fill_atomic` walks in 64, and
`pl_add32_sat` saturating ends a walk where `pl_next64` wraps and continues —
they differ by 2.4% on c183, demonstrated on the host with no GPU. And
`verify_apply_region`'s survivor return value is fixed at BOUND = 0.

**Phase 7 is much less risky than Phase 2 implied.** Zero of 4.2M cells differ
between `pl_log2f` on the GPU and libm's `log2f` on the CPU. The expected rate
is ~3e-7 per cell (~1% of log2 results differ at all, each by ~3e-5 after
scaling, and only a boundary crossing changes the integer) — about one cell in
three million, not the tens of thousands the Phase 2 note suggested.

**Known gap, Phase 8:** `k_fill_l1`/`k_fill_l2` are absent. Both want
33,792 B of static threadgroup memory against Apple's 32,768 B ceiling —
over by exactly 1 KB. They are the two-level fill path; apply requires
single-level 4-byte records, so production uses `k_fill_atomic` and never
reaches them. The fix is retuning `L1_CAP`/`L2_CAP` 64 -> 62, which is a
performance change and now permitted to be measured on this box.

**The Phase 4 reference is the CPU generator, not CUDA.** `fbgen.c` builds and
runs natively on macOS unchanged, so `fbgpucheck.sh` compares the Metal build
against an *independent* implementation rather than another GPU build. The HIP
port could not do this — its CPU `fbgen` segfaulted under MinGW — and settled
for a weaker check. Prefer this reference wherever a later phase can use it.

**Porting aids:** `metal/gen_fbgen_metal.py` and `metal/gen_fbgen_host.py`
produced the first drafts of the Phase 4 sources and are committed so a later
CUDA-side change can be re-diffed rather than re-ported from memory. They are
NOT wired into the build. The generated files are committed and reviewed like
any other source; edit the source, and update the generator only if you intend
to regenerate.

**Two conventions the rest of the port depends on** (plan Phase 3):
- CUDA kernel parameter *i* becomes MSL `[[buffer(i)]]`, same order.
- A templated kernel is reached through `[[host_name]]` named *base, then each
  template argument, joined by `_`, bools as 0/1*: `k_td<1,0,0,false>` is
  `"k_td_1_0_0_0"`.

**Known gate limitation, do not mistake for coverage:** the Phase 3
cross-stream check does not isolate `mtlStreamWaitEvent`. Its negative control
shows Metal already orders the two queues without it, almost certainly via
automatic hazard tracking on a shared tracked buffer. Re-test in Phase 5.

**Phase 2 changed a Phase 7 option.** "Portable log2 in the Metal build only"
is not a real choice: `pl_log2f` differs from the host's `log2f` on 1.02% of
inputs by up to 3 ULP, the same order as `metal::log2`. A portable log2 is not
more accurate, it is only *shared* — so it buys nothing for byte-identity
unless the CUDA build uses it too. Phase 7 now decides between "both builds
adopt it" and "accept a divergent relation set". See the plan's Phase 7.

## Drift ledger — CUDA-side changes made for this port

**The ledger lives in `bench/METAL_PORT_PLAN.md` section 9, and only there.**
A copy of it used to sit here and had already drifted to "none yet" while the
plan carried two rows — precisely the failure the ledger rule exists to catch,
committed by the ledger itself. Six rows as of 2026-09-15: `cofcheck.sh`
(`head -c -1`), `fbgpucheck.sh` (`sha256sum`), `slab.h` (`SLAB_PERF_REGIONS`
made overridable, default unchanged), `runlog.c`/`.h` (`g_runlog_quiet`),
`boinc_support.cpp`/`bench.h` (`bench_boinc_progress_suspend`), and
`cofcheck.sh` again (build detection; the `--ecm-b1 400000` case asserts a
refusal on Metal). The last two
are ports from `hip-port` and are inert unless called, which only the Metal
build does. Add new rows there.
