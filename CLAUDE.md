# cuda-sieve — Metal port

Porting the CUDA NFS lattice sieve in `bench/` to Metal Shading Language for
Apple Silicon. The plan of record is **`bench/METAL_PORT_PLAN.md`** — read it
first; it also carries the drift ledger. This file carries the rules.

**Rebased onto `main` at `0acfd8c` on 2026-09-17** (branched at `3e15fec`,
rebased onto `75d4cf7` the day before).
`hip-port` is a reference for method, not a base: it is well behind `main`.

**Rebasing means REGENERATING, and the generators' asserts are what tell you
so.** The rebase brought in upstream's ECM occupancy fix (`2bc1c6e`), which
touches `cofac.cuh` -- a file this port generates from and never edits. Two
generators failed immediately on their anchors, which is the system working;
the dangerous one failed *quietly* underneath. See "Rebase, 2026-09-16" below.

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

  **SPEED, same band, same settings, same relations: 1080 Ti 204.1 s vs M3
  328.0 s — the card 1.61x faster.** Per stage (ms/q, CUDA vs Metal):
  transform 8.5/34.2, **fill 212.1/153.9 — METAL FASTER**, apply 102.5/372.7,
  TD 17.9/125.6 (**7.03x, the weak spot**), algebraic queue 266.8/285.4,
  **cofactor device 318.3/349.7 — 1.10x, essentially parity**. The cofactor
  stage converging confirms 8h from the other side: it is latency-bound on one
  ECM chain, so cores and bandwidth buy almost nothing. `fill` wins on UMA.
  **TD is where the remaining Metal headroom is** — it was never tuned, and it
  is 2.6% of CUDA's wall against 11.2% of Metal's.

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
- **TD tuning (plan 8q): the cause is found, and no knob helps.** TD is the
  port's worst stage at 7.03x CUDA. Three theories measured and REFUTED:
  `BN_LIMBS` 12/8/7 (128.1/130.0/129.0 ms — no effect), `TD_TILE`
  512/256/128/64 (132.0/130.5/131.5/138.0 — no effect), and a TD-specific
  threadgroup width (TD -15% but **classify +106%**, net wall 1.8% WORSE).

  **That third one matters beyond TD.** Classify does not read that knob — the
  stages are coupled through the DATA: `k_td`'s grid-stride mapping decides
  which thread writes which cofactor, and `k_classify` reads those arrays, so
  halving TD's width scatters the writes classify then pays for. **8d's "one
  knob, three stages, three optima" assumed the stages are independent. They
  are not.** No knob shipped; a knob defaulting to the value it already had
  earns nothing.

  **The real cause: 64-bit integer multiply is 4.77x slower than 32-bit on
  this GPU** (probe: 401.54 ms vs 84.12 for the same loop). `bn_divmod_u32_pre`
  runs a `mulhi(ulong,ulong)` plus a 64-bit multiply PER LIMB, and Apple's
  32-bit ALUs emulate both, against a card where `__umul64hi` is an
  instruction.

  **THE 32-BIT DIVISION WAS TRIED AND IS 2x SLOWER (plan 8q).** Knuth D in base
  2^16, exact — verified against `(2^64-1)/d` for **every d in [2, 2^27]**
  (134,217,727 values, the whole range a factor-base prime can occupy) plus 4M
  random — and **4.46x faster as an isolated primitive** (2,231 ms vs 9,958).
  In `k_td` it made TD **254.8 ms/q against 128.9**. Inlined twice per call
  inside two nested loops, its two data-dependent correction loops cost
  register pressure and SIMD divergence a uniform-input probe cannot see. And
  the direction settles the diagnosis: **something 4.46x cheaper cannot double
  a stage it dominates**, so the reciprocal was never the bottleneck. Reverted.

  **Same lesson as 8h in another costume:** there a one-q band made a queued
  stage look like 59% of wall; here a primitive probe made one operation look
  like the bottleneck. Both were precise, reproducible, and about the wrong
  thing. **The next attempt needs a profile of the real kernel, not another
  plausible primitive.**

  **PROFILED AGAINST A GTX 1080 (plan 8q) — and it is neither thing I tried.**
  `--td` decomposes the stage on both platforms for free. Identical work
  (17,625,929 hits): norm 13.6 vs 27.8 ms (2.05x), **congruence test 81.2 vs
  324.2 (3.99x, 80% of Metal's TD)**, **division 31.4 vs 52.9 (1.68x — the
  BEST part, and the one this port tried twice to optimise)**. Had the test
  matched CUDA, TD would be 1.28x rather than 3.21x.

  **FIXED, partly: one struct load instead of six.** The test read six fields
  of `tile[e]` separately from threadgroup memory per prime, all same-address
  broadcasts. Copying the 32-byte `tdsmall_t` once cut the test **324.2 → 223.4
  ms (-31%)**, TD **404.9 → 306.3 standalone (-24%)** and **130.3 → 106.2 ms/q
  in the pipeline (-18%)**, with identical hits and relations. Test is now
  2.75x CUDA, TD 2.43x; same copy applied to `k_td_record_warp`.

  **A smaller tile does NOT help — it is load COUNT, not bytes.** Staging only
  the 24 hot bytes (`recip`, 8 of 32, is read on 0.2% of iterations) and
  fetching `recip` from device memory on a hit gave 221.7/219.9/222.2 ms
  against 223.4 — about 1%, inside noise, and was reverted. Cutting a quarter
  of the bytes changed nothing while cutting six loads to one saved 31%. **The ONE-uint4 packing was then built and is a WASH.** Field
  maxima measured over every entry: m<=32,749, rt<=32,382, cst<=16,384 (15
  bits), g<=19, sh<=14 — so `magic | m,rt | cst,g | sh` fits one `uint4`. The
  test drops 224.1 -> 200.5 ms, **exactly as the load-count theory predicted**,
  and the division rises 54 -> 70.8 because `recip` no longer fits the tile —
  a wash standalone (301 vs 301), and in the PIPELINE `norms + trial division`
  went **106.9 -> 199.5 ms/q** on the RECORD=1 and warp variants the standalone
  path never runs. Reverted.

  **PROFILED: RECORD=1 is 2.7% of the stage, and the pipeline's hot pass is
  RECORD=0.** An earlier note here claimed the opposite; it was wrong.
  `pipe_td_perq` launches `k_td_1_0_0_*` — the same variant the standalone
  `--td` runs. RECORD=1 appears only in `pipe_td_verify` (first q, skippable)
  and the 4.98 ms/q recording pass. Its scattered `fac[t*TD_FMAX+nf]` writes
  are not worth chasing. `SLABBED` is real but small: **105.7 ms/q slabbed vs
  94.1 unslabbed, ~12%** (and unslabbed costs 1358 ms/q of wall vs 981, so it
  is not an option).

  **Device breakdown, --nq 24 (ms/q):** norms+TD 105.3 (57.5%), **resieve +
  scatter 48.8 (26.7%)**, classify 18.3 (10.0%), record 5.0 (2.7%). **resieve+scatter measured (plan 8r): 1.63x — NOT the
  problem.** 30.56 ms on the 1080 against 49.4 on the M3, the same neighbourhood
  as the division's 1.68x and better than the sieve's 1.74x. It looked like a
  target only because it is large in absolute terms. **The gap is `norms +
  trial division` alone, at 4.49x.**

  **Inside that pass (8q's decomposition): norm 2.05x, congruence test 2.75x
  (~80% of the pass), division 1.68x** — the test is the gap at every level.
  But the pass's RATIO does not transfer between scales: **2.39x standalone**
  (one launch, ~16M candidates) against **4.49x in the pipeline** (~270k
  candidates over 8 slabs x 2 sides, ~17k records per launch against a
  131k-thread grid). Slab count explains ~9% (94.6/87.1/88.9/95.3 at 1/2/4/8)
  and `SLABBED` ~12%; **the rest is unexplained**. **SIDE-BY-SIDE PROBE DONE (plan 8r): the composition is
  the SAME on both platforms** — test 75.8% (CUDA) vs 70.8% (Metal), division
  15.9% vs 14.7%, norm 8.3% vs 14.5%. So Metal's 4.49x is a broadly uniform
  slowdown of the whole kernel, **not a hot spot** — there is no single
  sub-step left to attack. Probe = three launches per call (nsm=0, DIVIDE=0,
  real) run BEFORE the real one, which overwrites them.

  **Two traps it hit first, both of which looked like success:**
  `k_td<0,0,0,true>` is NOT instantiated (only the unslabbed one), and
  launching a missing kernel returns in ~0.01 ms — reads as "free", gives a
  NEGATIVE component time. And `nhit=nullptr` makes the congruence loop dead
  code, so the compiler deletes it; the probe must pass a real sink.

  **The candidate-count discrepancy is EXPLAINED (two causes, neither a
  disagreement about work).** (1) **Slab auto-calibration**: 8g runs three
  throwaway single-q bands before the real one, each a full TD pass — 88
  launches with it on, **8 with `--slab-j`**, matching CUDA. (2) **The builds
  slab differently** at the same `--region 13` because `SLAB_PERF_REGIONS` is
  8192 here and 32768 on CUDA — CUDA gets 2 slabs (n=134,755/launch), Metal
  forced to 1 (n=269,360). Exactly the 2x. **The totals agree: two-sided
  survivors/q 269,360 CUDA vs 269,611 Metal, 0.09% apart.**

  **RULE: pin `--slab-j` on BOTH sides before normalising anything per launch
  or per candidate.** Otherwise you are comparing different slab
  decompositions of identical work, against a build that also calibrates where
  Metal does not — and neither difference shows up in the number you compare.

  **`UNROLL` is flat on Metal** (48.7/50.0/51.0/49.9/49.8 at 1/2/4/8/16), so
  this kernel is NOT latency-bound here the way `td.cuh` says it is on NVIDIA.
  Note the slabbed path had **only UNROLL=4 instantiated** — the pipeline is
  always slabbed, so 4 was never a choice; adding 1/2/8/16 and sweeping is what
  showed it does not matter. Reverted rather than ship five unused kernels.

  **The remaining 2.75x is unexplained** and is the whole of TD's gap.

  **Correction:** an earlier note read `bigint.cuh`'s "30 ms division vs 13 ms
  congruence tests" as meaning CUDA's shape was Metal's mirror. It is not — on
  this card the test dominates CUDA too (64% vs 25%). That comment describes a
  different configuration.

  **What remains on the division:** `d` is uint32 and `rem < d`, so
  quotient and remainder both fit in 32 bits — only `cur` is 64-bit. The
  standard 2-word-by-1-word division needs only 32x32→64 products, would live
  in `bigint_msl.h` (Metal-only, no CUDA change), and the gates to verify it
  already exist: `sievecheck`'s 4.2M cells, `cofcheck.sh`, and Phase 7's
  byte-identity against real CUDA.

  **Two latent bugs fixed on the way.** `MSLFLAGS` was not passing
  `-DBN_LIMBS`, so the device kept `bigint_msl.h`'s default of 12 while the
  host took the Makefile value — identical today, a desynced `bn_t` the moment
  anyone changed it. And the define-lifters emitted `TD_TILE`/`TD_FMAX` BARE,
  so a lifted copy would override a `-D` of the same name; both now wrap every
  lifted define in `#ifndef`.

- **Phase 9 (packaging): STARTED. The arm64 BOINC library builds and links
  (plan 9a).** BOINC master `55a5644` / 8.3.0, built here to
  `~/code/boinc-install`. Prereqs came from Homebrew (`autoconf automake
  libtool pkg-config`); `_autosetup` needs
  **`LIBTOOLIZE=/opt/homebrew/bin/glibtoolize`** because Homebrew `g`-prefixes
  GNU libtool and BOINC's checker looks only for `libtoolize`.

  **The bare `--disable-server --disable-client --disable-manager` produces a
  library this port CANNOT SHIP. Three additions are load-bearing, and every
  one of the four configurations builds and links with exit 0 — these were
  found by inspecting the artifact, not by a failure.**
  1. **`--disable-shared`.** Otherwise libtool installs
     `libboinc_api.8.dylib`, `-lboinc_api` prefers it, and the binary carries
     an **absolute build-machine path** to it. Check with `otool -L`: it must
     name only `libSystem.B.dylib` and `libc++.1.dylib`.
  2. **`-mmacosx-version-min=13.0` in CFLAGS/CXXFLAGS.** A default build
     stamps every object `minos 26.0` against this port's macOS 13 floor —
     the support floor silently becoming "whatever this laptop runs".
     Verified 13.0 on all 46 objects in both archives.
  3. **`BOINC_HOST_STATIC=` (empty) on any macOS link.** The Makefile's
     default `-static-libgcc` is **`error: unsupported option`** with Apple
     clang (`-static-libstdc++` is merely ignored). It is also pointless here:
     macOS ships libc++ and libSystem.

  **Verified:** `boinc_support.cpp` compiles against the real 8.3.0 headers
  with `-Wall -Wextra` and zero warnings; `metal/boinc_link_probe.cpp` links
  the whole `bench_boinc_*` surface statically and runs. The probe gates its
  `bench_boinc_init` call on `argc`, **not `if (0)`** — at -O2 a dead call is
  deleted and `boinc_api.o` never gets pulled from the archive, giving a link
  that resolved nothing and looked identical. `bench_boinc_finish` **does not
  return** (`boinc_finish` exits), and the BOINC runtime **redirects stderr to
  `stderr.txt` in cwd** — which Phase 9 must reconcile with `runlog`.

  **`HAVE_BOINC` IS NOW WIRED INTO `Makefile.metal` (plan 9b), gate green.**
  `make -f Makefile.metal benchbin HAVE_BOINC=1 BOINC_DIR=<prefix>`. Default
  stays 0. The ported TUs already carried the whole integration; only the
  build wiring was missing, and neither needs a BOINC header — everything goes
  through `bench.h`.

  **THE `--help` BUILD MARKER HAD ROTTED, AND THE ROT IS MACHINE-CRASHING.**
  `gen_bench_main.py` rewrote CUDA's `--device` line to "select Metal device"
  in the **`#else` branch only**; a `-DHAVE_BOINC` build takes the `#ifdef`
  branch, which still said **"select CUDA device"**. `cofcheck.sh` classifies
  the build from exactly that string, so a BOINC build was detected as CUDA —
  and would then run the `--ecm-b1 400000` case that took WindowServer down
  twice. **A marker that holds in one branch of the `#ifdef` it is printed
  from is not a marker.** Both branches fixed, plus two stderr lines that said
  "running on CUDA device" out of a Metal binary.

  **`.metalflags.stamp`: this Makefile never tracked a flag change.** Flipping
  `HAVE_BOINC` left objects compiled the other way, and a `bench_main.o`
  without the define never calls `bench_boinc_init()` — a silently wrong
  binary, not a link error. The stamp covers the whole
  `HOSTFLAGS|MSLFLAGS|BOINC_LINK|CPUOBJ_TUNE` signature, so changing **any**
  tunable rebuilds what it affects. That also retires 8q's stale-artifact
  trap. `HOSTFLAGS` is split into `HOSTFLAGS_BASE` + BOINC flags because
  include search is left to right: with `BOINC_CPPFLAGS` in `HOSTFLAGS`, the
  stub gate's `-I metal/boinc_stub` would **lose to the real SDK** and
  `boinccheck` would quietly stop testing the stub.

  **`make -f Makefile.metal boinclinkcheck HAVE_BOINC=1 BOINC_DIR=...`** — 7
  checks, all green, **and the control fails** (run against the
  `HAVE_BOINC=0` binary it reports the three BOINC symbols missing). It
  refuses to run at `HAVE_BOINC=0` rather than vacuously pass. Every check is
  for something that builds and links with exit 0 and is still wrong.

  **UNDER BOINC, stderr GOES TO `stderr.txt` IN THE WORKING DIRECTORY**, from
  `boinc_init` onward — which is before argument parsing. **A check that greps
  stderr from a terminal or a `2>&1` pipe finds nothing and "passes" for the
  wrong reason.** Assert on stderr only by reading `stderr.txt`; the gate
  prints the stderr-only assertions to check by hand rather than faking them.
  `--help` is `printf`, i.e. stdout, which is why the marker check and
  `cofcheck.sh`'s detection are sound.

  **End-to-end, done by hand once:** a `HAVE_BOINC=1` binary at the parity
  special-q gives **exit 0 and the golden 37 relations**, with `stderr.txt`
  carrying `BOINC: running on Metal device 0 of 1: Apple M3` — while stdout
  mentioned BOINC once.

  **READING that file cut it from 14 lines to 6 (plan 9c), and one of the 14
  was FALSE.** A volunteer uploads `stderr.txt`; a project reads it when a task
  fails.
  - The curves/round advisory claimed 119 ms was "over this build's 750 ms
    bound". Its guard is `ecm_curves > 2u` with **no test against the bound at
    all** — the behaviour is right (8n: aim at 2, the bound is a ceiling) but
    the message invented a violation. The acting line now says what it did;
    the advisory branch is gone, so when the derivation cannot act (explicit
    `--ecm-curves`, or a rho side — **the default**) it is silent.
  - `cof_report_chunk` printed the SAME line twice against its own "only on
    change" comment: `step` is `min(chunk, n)`, so when both chunks exceed `n`
    the internal value changes and the rendered line does not. It now compares
    **what it is about to print**.
  - The allowance notes are terminal diagnostics that advise changing a
    parameter the project sent in the job file — dropped.
  - **The launch line reports only launches OVER the 750 ms bound.** A
    per-band high-water line is a stream of messages saying nothing is wrong;
    an over-bound launch is the condition 8k set the bound for and is
    invisible elsewhere. Still gated on a new maximum (so a device that cannot
    meet the bound goes quiet once the chunker parks), and **not** gated on
    auto mode — a pinned `--cof-chunk` that overruns matters more, since
    nothing will adapt. Verified both ways on one q: silent at a 401 ms
    launch, and at `--ecm-curves 48` it prints `kernel launch 1508 ms is over
    this build's 750 ms bound`, matching the reported `algebraic queue
    1507.68 ms` from the other side. 37 relations either way.

  **All of it removed in the GENERATORS, never in `bench_main.cu`/`cofac.cuh`**
  — editing those is a CUDA-side *behaviour* change (not the inert kind the
  drift ledger covers) and would reach HIP. The two builds now deliberately
  differ in what they log. No gate reads these strings.

  **AN UNASSERTED `src.replace` IN A GENERATOR IS A SILENT NO-OP.** Dropping
  `ms_launch_max` broke an anchor 200 lines away that had no assert: the
  replace matched nothing, the generator **printed success**, and it emitted a
  `cofq_t` with `Q->ecm_rounds` assigned twice and declared never. A missing
  field fails to compile; a missing *statement* would not have. **Every
  single-site anchor must be asserted** — same failure as the regex that
  deleted half of `bigint.cuh`. Audited: the only other unasserted replaces
  are the bulk `cuda*->mtl*` and `LAUNCH_APPLY` tables (matching nothing is
  legitimate there) plus one with a stronger `assert src.count(...) == 1`.
  **And anchor on the UPSTREAM source, never on this generator's own output** —
  the broken anchor pointed at a line the generator itself had added and later
  removed. `ms_launch_max` now hangs off `cofac.cuh`'s own timing declaration.

  **THE METALLIB IS EMBEDDED; THE APPLICATION IS ONE FILE (plan 9d).**
  `-sectcreate __DATA __metallib` at link, read back in `metal_rt` with
  `getsectiondata(&_mh_execute_header, ...)` and wrapped in a `dispatch_data_t`
  with an **empty destructor block** — the bytes are in our own `__DATA` and
  outlive any use, so there is nothing to free and
  `DISPATCH_DATA_DESTRUCTOR_DEFAULT` would copy the whole megabyte.

  **Four sources, and the order is deliberate:** explicit `mtlInit` path →
  **`$CUDA_SIEVE_METALLIB`** → **embedded section** → `bench.metallib` beside
  the executable. The env var stays ahead of the embedded copy because **every
  gate here drives the library it just built through it**; the embedded copy
  goes ahead of the file because a stale `bench.metallib` is exactly what a
  BOINC slot accumulates. 718 KB → **1,726,104 bytes**. `EMBED_METALLIB=0`
  opts out.

  **`make -f Makefile.metal metallibcheck`**, control first: a
  `EMBED_METALLIB=0` build must have no section AND must fail to run. Both
  runs happen in a temp directory with no `bench.metallib` and with
  `CUDA_SIEVE_METALLIB` unset (`env -u`) — **that negative space is the whole
  check, because every other gate EXPORTS that variable and so none of them
  would notice embedding being broken.** `boinclinkcheck` asserts the section
  too (8 checks now): a distributable binary that needs a file beside it is
  not distributable.

  **End to end:** one 1.7 MB file alone in a directory, `HAVE_BOINC=1`, no
  environment variable → exit 0, 37 relations, the same six-line `stderr.txt`.

  **A `-A2` that should have been `-A4`** — `otool -l` prints `sectname`,
  `segname`, `addr`, *then* `size` — made the gate report "nothing was
  embedded" about a binary that had just sieved 37 relations from its own
  embedded library. **The two checks disagreeing is what caught it.**
  `fbgen_gpu` is deliberately NOT embedded: dev/project-side tool.

  **VALIDATED OVER 288 SPECIAL-Q (plan 9e): the packaged binary's relations are
  BYTE-IDENTICAL to the stock build's** — 13,485 relations, 564,696 records
  enqueued (exactly Phase 7's numbers against real CUDA), `cmp` clean,
  sha256 `8e79762c…`. The packaged binary ran **in a directory containing
  nothing but itself**, `CUDA_SIEVE_METALLIB` unset. Wall 258 s stock vs 280 s
  packaged — that ~8% is this fanless box across back-to-back runs, NOT a
  measurement of packaging overhead.

  **A comparison that agrees can still be measuring almost nothing.** The first
  pair of runs omitted `--cofactor`, emitted 2,381 relations, and were also
  byte-identical — and useless, because without the cofactoriser the band only
  emits relations that need no splitting. The tell was in the output all along:
  `records enqueued 564696 (of which 2381 needed no splitting)`.

  **SHIPPING: ONE executable file, plus the job data.** `otool -L` lists seven
  libraries and every one is part of macOS (Metal, Foundation, IOKit,
  CoreFoundation, libSystem, libc++, libobjc). The tree's one `dlopen` is
  `libnvidia-ml.so.1` for optional NVML telemetry — impossible on macOS,
  `runlog_gpu_bind` returns -1, caller carries on.

  **AND THE 110 MB FACTOR BASE NEED NOT BE SHIPPED EITHER.** With no
  `--fb1`/`--cadofb`, pipeline mode generates the complete algebraic FB on the
  GPU at startup — 7,605,616 ideals in **6.5 s**, once per process.
  **Verified over the same 288 q: 13,485 relations, sha256 `8e79762c…`,
  `cmp` clean against the cached-file run**, from a directory containing
  nothing but the 1.7 MB executable. So a workunit needs the poly (462 B) and
  the parameters, not the factor base.

  **But get `alim` right per job.** `genlim = min(fbbound, alim)`, and `alim`
  comes from a GGNFS `.job`, from `--alim`, or from a **compiled-in default of
  134,200,000** (`bench_main_metal.cpp:989`). A CADO `.poly` carries no alim,
  so the verified run above took that default — which for c183 *is* the
  production alim. **That is a coincidence of this job, not a derivation from
  the polynomial.** For any other composite pass `--alim` or a `.job`, or the
  factor base is silently the wrong size.

  Three things a project must know, none a missing file: **arm64 only,
  non-fat** (an Intel Mac cannot run it — Rosetta goes x86_64→arm64, not the
  reverse — and the port refuses non-Apple GPUs anyway); **`minos 13.0`**; and
  **the signature is ad-hoc/linker-signed, not Developer ID — whether BOINC
  distribution on macOS needs a real signature or notarization is UNTESTED
  here.**

  **SIGNED AND STAGED (plan 9f).** `~/code/dist/` holds `bench` (sha256
  `e5995826…`) and `bench.sig` (256 hex chars, a 1024-bit RSA signature).
  **Re-staged and re-signed 2026-09-16 after 9z-i.** Superseded, none to be
  shipped: `25ab6b65…` (pre-leak-fix), `9e975aa5…` (pre-autorelease-fix),
  `52f6280b…` (pre-nil-binding-fix), `2e1a2d5f…` (pre-mask-guard),
  `1e85fd5e…` (pre-9z-h/9z-i). **The binary in the field is older than all of
  them** — its stderr still says "this is a CUDA application". Revalidated over the full 288 q
  from a one-file directory: **13,485 relations, `cmp` clean, sha256
  `8e79762c…` unchanged**, and free memory ends at **10.84 GB against 8.13
  before** (2.71 GB recovered). A signature is over content, so re-signing was
  mandatory, not optional.
  `crypt_prog` is NOT built by `--disable-server` (`lib/Makefile.am` puts it
  and `libboinc_crypt` under `if ENABLE_SERVER`, and `SSL_LIBS` is empty), so
  it was compiled by hand against the already-built tree with Homebrew
  `openssl@3` — `crypt_prog.cpp` needs `<openssl/encoder.h>`, i.e. OpenSSL 3.x.

  **BOINC's key parser rejects CRLF key files, and that is a real bug.** The
  project key failed with `Error: scan_private_key_hex`. **Diagnosed without
  opening it**, from file size alone: a fresh BOINC 1024-bit key is 1437 bytes
  with 24 newlines, the project key is 1461 — **exactly one extra byte per
  line**. Converting a throwaway key to CRLF gave 1461 bytes and the identical
  error. Two places in `lib/crypt.cpp`: `sscan_key_hex` requires every char
  before `'\n'` to be a digit, and `sscan_hex_data` skips `'\n'` but *breaks*
  on non-hex, truncating the key. Both now skip CR. **Patched in the local
  BOINC tree (`~/code/boinc/lib/crypt.cpp`), NOT in this repo**; original at
  `/tmp/crypt.cpp.orig`. After the patch the same key in LF and CRLF form
  produce an identical signature.

  **`bench.sig` VERIFIES against the project's public key** — `crypt_prog
  -verify` says `signature is valid`, exit 0 — **and the control fails**: one
  bit flipped at byte 863,052 of a copy gives `signature is invalid`, exit 1.
  `sha256(bench) = 25ab6b65…2fdac977`, `sha256(bench.sig) = 77204967…bcb24622`.
  **The private key was never read or copied** — it appears once, as an argv.
  This is BOINC's file signature, **not** a macOS Gatekeeper signature, which
  remains untested.

  **NOT established:** still standalone mode, no `init_data.xml`, so slot
  filename resolution, a real GPU assignment and checkpointing are
  unexercised. **That is the only Phase 9 item left** — it needs a real BOINC
  client, not this machine.

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

## Code review, 2026-09-16 (plan 9z)

**`mtlFree` LEAKED THE WHOLE ALLOCATION — fixed.** `metal_rt.mm` is compiled
**without `-fobjc-arc`**, so `Alloc`'s `id<MTLBuffer>` is unmanaged,
`newBufferWithLength:` is +1, and the registry entry was the only owner;
`reg_erase` erased it without releasing. **64 MB per malloc/free pair**, and it
reached production: slab calibration builds and tears down the pipeline three
times, so a band ended at **8.37 GB free instead of 10.62** — 2.25 GB
recovered by the fix, relations unchanged, twelve gates green. Bounded (21
allocation reports at 24 q and at 288 q alike), but fatal margin on an 8 GB M1.
Same bug fixed in `mtlShutdown` and `pso_for`'s `MTLFunction`.

**Verified with controls, not by reading:**
- **A missing kernel fails CLOSED.** Sabotaging a production launch name gives
  **exit 255, zero relations**, and `mtlGetLastError()` names
  `pipeline_host.inc:1246`. (An older note here claimed the opposite; it was
  about 8q probe code, not the port.)
- **Zero generator drift**: all twelve `gen_*.py` reproduce their committed
  output **byte for byte** — `git status` clean after regenerating everything.
- **42 asserted anchors, 0 unasserted single-site `src.replace`.**
- **No lock recursion**: `mtlUseResource` deliberately skips `g_lock`;
  `mtlDeviceAddress` takes it and is only called outside the bind window.

**FIXED: the autorelease problem, ownership first (plan 9z-b).** There was no
`@autoreleasepool` anywhere in the shim, and **the leak was standing in for a
lifetime**: `commit()` stored `st->last = st->cb` from `[st->q commandBuffer]`
— **autoreleased and never retained** — and `sync()` and every event query read
it later. Nothing drained, so it survived by accident. **A pool alone would
have been a use-after-free, not a fix.**

So ownership came first: `cb`, `cenc`, `benc`, `last`, the stream's
`MTLCommandQueue` and the event's `MTLEvent` are retained on store and released
on replace or teardown (`stream_teardown()` — two places used to drop a queue
on the floor). `mtlEventRecordOn` takes its own reference, retain-before-
release. **Then** the pool, at the creation site — because balancing our own
retain/release is NOT enough, the pending autorelease must also fire, and
without a pool it never does:

```objc
void ensure_cb(Stream *st)
{ if (st->cb) return; @autoreleasepool { st->cb = [[st->q commandBuffer] retain]; } }
```

**Measured, same 144-q band: RSS growth 372→399 MB (~0.36 MB/s) becomes
361.9→367.4 (~0.07) — about 80% gone**, and what remains steps then flattens,
the shape of the cross-q queue filling toward a flush. That also settles 9z's
open question: it was mostly the autoreleased objects, not the queue. A command
buffer retains every resource it references, so each leaked one pinned buffers
too. **Validated as a lifetime change must be — twelve gates, cofcheck 54/0,
and 288 q byte-identical at `sha256 8e79762c…`.**

Lesser, open: a zero-sized launch is silently skipped where CUDA returns
`cudaErrorInvalidConfiguration`; `mtlDeviceAddress` lacks the "do not call
while binding" warning its sibling has; `pso_for` caches `nil` so the
missing-kernel diagnostic repeats per attempt.

## Second review, 2026-09-16 (plan 9z-c)

**FIXED (plan 9z-d/9z-e): the sieve now runs clean under `MTL_DEBUG_LAYER=1`,
and `make -f Makefile.metal validationcheck` keeps it that way.** Optional
buffers are declared under an MSL **function constant** — when the constant is
false the argument does not exist and Metal asks for no binding. metal_rt
computes a nil-mask from the launch arguments (runtime value, not spelling),
`pso_for` caches per (name, mask), and one uint constant at index 0 carries
"argument i is bound". **A bitmask, not one Bool per index, because Metal
rejects a constant value for an index a function does not declare.**
**Discovery is `functionConstantsDictionary`** — a plain `newFunctionWithName:`
does NOT return nil for a specialised function; it returns an object that
cannot build a pipeline, and Metal asserts **in ordinary builds**.

Wrappers forward through a local (`if (mtl_bound_11) dump_opt = dump;`) because
a function-constant argument may be **named** only where it exists — **the
`_body` templates are untouched** and still test the pointer. Converted:
`k_scan_block`, `k_apply` (11,13,24,25), `k_intersect_compact`,
`k_fill_atomic`, `k_transform`, `k_resieve_scatter`, `k_td`, `k_cofac`.

**Two were invisible to a static audit:** `k_apply` 25 (`survbits` — null only
in `phase5_test`, not the pipeline) and `k_cofac` 10 (`iters` — a **runtime**
null variable, not a literal). Grep cannot find those; the runtime mask does.

**AND VALIDATION FOUND A REAL BUG, not just hygiene.** `k_transform` declares
`a0..b1` as `int64_t`; the warm-up launch (`pipeline.cuh:1844`) passes literals
`1, 0, 0, 1`. **CUDA converts at the call site; Metal binds by value and takes
the literal's width** — 4 bytes bound for an 8-byte read. Harmless only because
that warm-up passes `n = 0u`. **It is a class**: any narrower literal or
variable meeting a wider kernel parameter binds the wrong width silently, and
nothing but the validation layer sees it. Fixed Metal-side.

**My 9z-b retain/release work was cleared dynamically, not by reading:**
`OBJC_DEBUG_MISSING_POOLS=YES` → **0** "autoreleased with no pool" (the pools
really do cover every autorelease); `NSZombieEnabled=YES` → **0** messages to a
deallocated object (nothing over-released). Plus a static audit of all 17
ownership sites.

**Reviewed, no findings:** `metal/slab_calib.inc` — `g_runlog_quiet` and the
BOINC suspend are both reset unconditionally, the dispatch matches
`run_pipeline`'s own `cplan.enabled` branch, and post-clamp `cplan.jmax` is what
is reported and reused. Carried minor: `mtlEventSynchronize` uses `e->cb` after
unlocking — safe only because events are single-threaded here.

## Third review, 2026-09-16 (plan 9z-f)

**Fixed: the nil-mask had no width guard.** `mtl_launch` builds it with
`1u << b`; past index 31 that is UB **and silent** — a null above 31 would
report as *bound*, and the port would be back to 9z-c's bug. Widest kernel is
`k_apply` at **29 bound args, max index 27**, so there was headroom but no
protection. Now `static_assert(sizeof...(A) <= 32, ...)`.

**Verified: validation is clean everywhere**, not just the pipeline —
`phase5_test`, `scan_test`, `argbuf_test`, `classify_test`, `cofac_test` all
exit 0 with zero assertions under `MTL_DEBUG_LAYER=1`.

**Verified: MEMORY IS FLAT over a full 288-q band** (closes 9z-b's open
question). RSS 362.8 → 368.1 MB in the first 80 s, then **+0.2 MB over the
next 160 s**. The residual 9z-b could not attribute was the cross-q queue
filling toward its first flush, as hypothesised — **no unbounded component**.
Caveat: four minutes, not four hours; it rules out a per-q or per-flush leak,
not something with a much longer period.

**Verified, no findings:** `fbgen_gpu_metal.cpp` (1,518 lines, production
path) IS generator-reproducible — "NOT wired into the build" is about the
build, not drift. And no emulated 64-bit counter (`nlost`, `nhit`,
`noverflow`, `ntested`, `ndiv`, `nproj`) feeds a host decision, so
`atomicAdd64`'s torn-read hazard stays diagnostic-only as claimed.

**Minor, open:** `argbuf_test.cpp` and `classify_test.cpp` ignore `mtlMalloc`'s
return at six sites (production uses `MTL_OR_DIE`) — gate ergonomics, not a
shipped defect.

## A FIELD RUN, and the GPU assignment it exposed (plan 9z-g)

**The port has now run under a real BOINC client** — a successful workunit on
an **M4 Max**, client 8.2.11, factor base generated on the GPU in **3.0 s**
(6.5 s on this M3), `boinc_finish(0)`. That closes 9a–9f's standing "not
established".

**Its first line was wrong, and not only in wording.**
`bench_boinc_gpu_device()` hardcoded `strncmp(aid.gpu_type, "NVIDIA", 6)`, so
it **rejected the `apple_gpu` device the client correctly assigned**, then
reported "no usable GPU assignment in init_data.xml" — which was false.
**No wrong results and no wrong device** (Apple silicon exposes one Metal
device; the log says `device 0 of 1`, so the fallback IS the assigned one) —
what it costs is diagnosis, and it is exactly the line someone would trust
while debugging a real assignment problem.

Fixed with the port's usual shape, since `boinc_support.cpp` is shared
CUDA-side code: **one selector, `-DBENCH_BOINC_METAL_GPU`**, picks
`apple_gpu`/`Metal`/`an Apple`. **The CUDA build is byte-identical, message
text included** — hence three macros, `KIND` carrying its own article so
`"is not an NVIDIA one."` survives verbatim; verified by compiling both ways
and diffing the strings. One flag rather than three quoted strings because
those must survive make, a sub-make's `BOINC_CPPFLAGS` and two shells, and
they do not (``No rule to make target `Apple"'``). Drift-ledger row added.

**AND 9b's message pass was incomplete — the log is why.** 9b fixed the
messages it had *seen fire*; the two "client assigned CUDA device" lines could
not fire until the rejection was fixed, and `boinc_support.cpp` was never
grepped. A proper multiline-safe scan of every `fprintf(stderr, ...)` across
nine files found **nine** CUDA-named messages: 1 in `boinc_support.cpp`, 6 in
`bench_main_metal.cpp`, 1 in `pipeline_host.inc`, 1 in `bench_host.cpp`, and
**4 in `fbgen_gpu_metal.cpp` — which runs on every production task** that
supplies no `--fb1`. All now say Metal; the scan reports **0 remaining**, the
sole exception being the env-var name `CUDA_SIEVE_METAL_DEVICE`.

**Measuring-rig note:** the first post-sweep 288-q run took **17m48s** against
the usual ~3m35s; re-run immediately, **3m39s with identical relations**. It
had started straight after the full gate suite. This is a fanless MacBook Air —
a number taken right after a long GPU burn is not a measurement.

**TUNED AGAINST A FULL FLUSH (plan 9z-h), and it retracts two earlier
conclusions.** 8h ("halving the records leaves the launch unchanged") and
8i/8j ("chunking costs 25%") were both measured at **single-q, ~1,852
records**, where a 131,072-thread grid is so oversubscribed the launch is
chain-bound. **At a real 130k-record flush the launch is RECORD-bound and
linear in the chunk** — 15.8 µs/record over an 8.5× range (242/483/976/1713 ms
at 15,360/30,720/61,440/131,072). Linearity holds only in that regime: pushed
below ~2k records the response flattens and reverses (1,996 → 94.0 ms, 1,792 →
139.9), which is what the **no-progress guard** is for — it fired and parked.

**Steering is now proportional, not halve/double**: aim at 0.8× the bound.
From a 6× overshoot it converges in **one** flush (15,360 → 1,996) where
halving needs three, each of those running ~67 q at a known-wrong chunk.
`COF_CHUNK_TARGET_MS` is `#ifndef`-guarded so the controller can be exercised
at other bounds without faking hardware.

**Two things this does NOT establish.** (1) **The throughput cost of meeting
the bound is unknown** — wall/q against chunk came back non-monotone (825,
1501, 1658, 1208, 916 ms/q), a 2× spread with no ordering; run-to-run variance
on this fanless box swamps the effect. Relations were identical throughout, so
nothing is wrong — the timing just cannot be read here. (2) **The field
oscillation is NOT a control-law artifact**, contrary to what I first assumed:
for a linear response halving lands at ≥0.5× the bound and can never fall
below the `target/4` doubling threshold, so `61440 → 30720 → 61440` requires a
**25× measurement swing at 2× fewer records** — variance on that host.
Proportional steering does not fix it; that needs hysteresis or averaging, and
there is no data from that host to tune either.

**On this M3 nothing changes** — the opening chunk measures ~240 ms, inside the
dead band, so the controller never steers. The change earns its place on
larger GPUs (an M4 Max opens at 61,440) and at tighter bounds.

## Field failure 9z-i: a command-buffer error the log could not explain

An **M2** task died in fbgen at ~37% with `kernel launch failed` after
`k_alg_roots_fixed_mark_*`. **The memcpy that reported it is not the fault** —
it is the first sync that noticed an asynchronous command-buffer error; the
`mtlGetLastError()` before it passes because that carries only launch-
*configuration* failures.

**Ruled out here:** the `mtlFree` leak (per-segment fbgen allocations are only
~39 MB — `GPU_FB_MAX_ROOTS` 9, 8M-odd segments — so ~150 MB by 37%, nowhere
near an 8 GB M2); a nil binding (fbgen's `scan_rec` buffer is synced and
checked at line 551, which passed); a null `c_alg` (set at line 484 under
`MTL_OR_DIE`). **Not ruled out and not distinguishable from that log:** a real
GPU page fault, a watchdog hang, memory pressure, or eviction.

**THE DEFECT IS THAT THE LOG COULDN'T SAY.** `cb_status` collapsed every
non-Timeout, non-OOM command-buffer error into one code and printed Metal's
`localizedDescription` **only under `CUDA_SIEVE_METAL_TRACE`**, which no
volunteer sets. Now reported unconditionally with the numeric code (
descriptions are localised, codes are not) and named cases for internal,
timeout, **page fault**, not-permitted, OOM, invalid resource, memoryless,
device-removed and stack-overflow. A failure is fatal anyway.

**Reading the next one:** `page fault` → a bad address, the 9z-c class;
`timeout` → the watchdog, argues for splitting the root finder's grid; `out of
memory`/victim → the rest of the volunteer's machine.

**MORE FIELD DATA: FAMILY-CORRELATED AND NON-DETERMINISTIC.** Several
failures, **all M1 and M2**; **a different M2 succeeded**; M3/M4 Max succeed.
The log **positively excludes** timeout and OOM (those map to their own
strings), leaving page fault / internal / invalid resource / device removed /
stack overflow. The root finder is statically clean — grid-stride `t < n`,
`rootbuf` indexed by **prime** not thread (so the machine-dependent grid width,
56 blocks on an M1 vs 320 on an M4 Max, cannot overrun it), and
`fp_split_linear` ≤ `CAP` with one extra root against a 9-element array.

**This is the Phase 0 gap finally showing up.** Every probe, gate and
measurement in this port ran on **one Apple9 device**. M1 is Apple7, M2 is
Apple8, and CLAUDE.md has said since 2026-09-14 that family-gated things
(`mulhi(ulong,ulong)`, argument buffers) "should be re-verified the first time
an M1 or M2 is available". Non-determinism argues against wrong arithmetic
(fbgen depends only on poly and `lim`, so that would kill every M2 at the same
segment) and toward a marginal fault — a spill or a race sensitive to
scheduling or contention.

**The bottleneck is DATA, not analysis.** (1) Deploy 9z-i so the next failure
names its code. (2) **Get `make -f Makefile.metal metalcheck` run on an M1 or
M2** — the Phase 2 gate, 6.6M results against the host, no job data, seconds to
run, and exactly the instrument for the family-gated arithmetic this port has
never verified off Apple9.

**The deployed binary is old** — its stderr still says "this is a CUDA
application", so it pre-dates 9z-g and therefore also the leak fix (9z), the
autorelease fix (9z-b) and the nil-binding fix (9z-e).

## 9z-j: macOS was killing fbgen for IMPACTING INTERACTIVITY

**THE DIAGNOSIS BELOW IS RIGHT AND THE FIX IS WRONG. 9z-j CHANGED NOTHING THE
WATCHDOG CAN SEE, AND IT WAS MEASURED, COMMITTED, SIGNED AND DEPLOYED. Read
9z-k before believing any number in this section.**

9z-i's error reporting answered it on the first failure after deployment. An
**M1**: `command buffer failed: internal (code 1): Impacting Interactivity
(kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`. **macOS killed the
command buffer for hogging the GPU against the UI** — a soft watchdog, not
`MTLCommandBufferErrorTimeout`, which is why the old build could only say
"kernel launch failed".

**The family correlation falls straight out of the grid.** The root finder uses
`min((nprime+127)/128, cores*8)` blocks, so **a smaller GPU means MORE primes
per thread**: M4 Max 320 blocks/~21 each, M3 80/~105 (**790 ms measured**), M1
56–64/~150–190 (seconds). 790 ms already breached this build's own **750 ms
interactivity policy — which had only ever been applied to the cofactoriser;
fbgen was never bounded.** Whether an M1 gets killed depends on what else the
machine is doing, hence one M2 finishing and others not. The root finder is
790 ms of each 819 ms segment, so there is no second offender.

**Fix: slice by GRID-STRIDES (`FB_ROOTS_STRIDES_PER_LAUNCH` = 32)**, so every
device does the same iterations per launch and only per-stride cost varies —
sliced by iterations, not a fixed prime count, because a fixed count is exactly
what made small GPUs loop longer. **No device-side change**: slices are taken
with pointer arithmetic (`d_primes + off`, `d_rootbuf + off*GPU_FB_MAX_ROOTS`,
…) and the allocation registry resolves interior pointers at bind time — what
it exists for. `fbgen_gpu.cu` untouched; `d_failures` accumulates across slices.

**Measured: 790 ms → 238–279 ms per launch, fbgen 6.550 s → 6.616 s** (1% for
4× the launches). Identical output (7,605,616 ideals, 207 prime-power, 38 exact
primes) and a 288-q band **run the field's way with no `--fb1`** is
`cmp`-identical at 13,485.

**`fbcheck` does NOT cover this path** — it exercises the standalone
`fbgen_gpu` tool, which has its own copy of the launch. The no-`--fb1` band is
the check that matters.

## 9z-k: THE WATCHDOG JUDGES A COMMAND BUFFER, NOT A DISPATCH

A **second M2** failed with the identical `Impacting Interactivity` error after
9z-j deployed. 9z-j could not have worked, and the reason was in
`metal_rt.mm`'s own header comment all along:

> the encoder is closed and the buffer committed only when something demands
> ordering -- a sync, an event record, or a switch between compute and blit work

**Slicing one launch into four leaves all four in the same command buffer.**
Measured with `CUDA_SIEVE_METAL_CBTIME=1` (new, permanent): the fbgen segments
were still **787-826 ms command buffers**, exactly their pre-9z-j duration.
9z-j's reported "790 -> 238-279 ms" was **dispatch** time -- a real number
about a quantity nothing kills you for.

**AND THE COFACTORISER WAS FOUR TIMES WORSE THAN fbgen.** With the factor base
loaded from a file, so fbgen never ran:

```
3179.61 ms [50 dispatches k_cofac_3_1_1..k_cofac_3_1_1]
 780.14 ms [51 dispatches k_cofac_3_0_0..k_cof_gate]
 243.79 ms [ 1 dispatch  k_cofac_3_1_1]
```

Fifty ECM **rounds**, each a single launch of ~244 ms and each comfortably
inside the bound, batched into **one 3,180 ms submission**. So **8k's 750 ms
launch bound has been measuring the wrong unit since the day it was set**, and
every conclusion in 8h/8i/8j/9z-h about "launch duration" describes a dispatch.

**The bracket that measured it is what hid it.** `g_cof_peak` records an event
around the `r == 0, b == 0` launch, and an event record forces a commit -- so
the one launch being MEASURED was the one launch that was not batched. The
measurement was correct and unrepresentative, which is the worst combination.

**Fix: `mtlStreamFlush()` -- commit without waiting.** New in `metal_rt`,
because CUDA has no command-buffer object and so nothing to mirror. Command
buffers on one `MTLCommandQueue` execute in commit order, so splitting changes
nothing about ordering or hazard tracking, and the CPU keeps running ahead (it
is not the pipeline stall an event record is). Called per slice in the fbgen
root finder and **per launch in `cf_run_rounds`**.

| | before | after |
|---|---|---|
| fbgen root finder | 826 ms | **156 ms** |
| cofactor round batch | 3,180 ms | **247 ms** |

`FB_ROOTS_STRIDES_PER_LAUNCH` 32 -> **16**, now bounding a submission rather
than a dispatch. fbgen 6.550 -> 6.758 s (~3%); the 288-q band is `cmp`-identical
at **13,485 relations, sha256 `8e79762c…`**, 564,696 enqueued, wall 3m45 against
~3m39 (inside this fanless box's noise).

**THE FLUSH COULD HAVE SWALLOWED ERRORS, AND ALMOST DID.** `sync()` checked
only `st->last`; with flushing, buffers 1..n-1 were replaced and released
**unchecked**, so a failure in an early slice would vanish and the run would
carry on with garbage -- undoing 9z-i. `Stream` now keeps a `pending` vector
and `sync()` checks **every** buffer, reporting the first failure. (The first
attempt at the flush read `st->last` *after* `commit()` had released it:
instant SIGSEGV, the 9z-b class again.)

**GATED: `make -f Makefile.metal cbtimecheck`.** Asserts no command buffer
exceeds `COF_BOUND_MS` (750), runs **without `--fb1`** because that is the path
`fbcheck` does not cover, and **the control fails** -- built with
`METAL_EXTRA_DEFS=-DFB_ROOTS_STRIDES_PER_LAUNCH=4000000u` it reports 804.77 ms
and the gate catches it. This is the gate whose absence let 9z-j ship: the tree
had no instrument for the quantity that was killing tasks.

**Still not measured: the real threshold.** The observed kills were command
buffers of roughly 800 ms on M1/M2; Apple documents no line.

**`COF_CHUNK_TARGET_MS` IS NOW 400, LOWERED FROM 750 BY THE USER AFTER 9z-k.**
750 was set (8k) against what was believed to be a launch bound but was in fact
a bound on one dispatch; 9z-k makes one cofactor launch one command buffer, so
the number finally means what it always claimed to. 400 sits at half the
shortest duration anyone has been killed at rather than a hair under it.
`COF_BOUND_MS` in `Makefile.metal` tracks it so `cbtimecheck` enforces the
policy the build actually holds.

**It changes nothing on this M3, and that is the expected result.** At a full
`CQ_FLUSH` the chunker still settles at 15,360 records/launch and the longest
cofactor command buffer is **250.76 ms** -- already inside 400, so the
controller never steers. 288 q `cmp`-identical at 13,485 / `8e79762c…`, wall
3m39.8 against ~3m39, `cofcheck.sh` 54/0, `cofaccheck` and `validationcheck`
green, `cbtimecheck` passing at the tighter bound with the control still
failing (805 ms). **The bound binds on slower devices, which is its whole
purpose** -- an M1 whose per-launch cost is 1.5-2x this one now gets steered
down instead of sitting at 400-600 ms per submission forever.

## 9z-l: "no Metal device" was three causes wearing one message

A field task (client 8.2.9) exited 1 seconds after starting:

```
BOINC: client assigned Metal device 0
bench: this process sees no Metal device
```

**The client had already assigned an `apple_gpu`, so the hardware was there and
the PROCESS could not reach it.** On macOS that is what happens outside a GUI
login session: `MTLCreateSystemDefaultDevice()` returns nil. It was diagnosable
only by what the log did NOT contain -- the family gate names the GPU it
refuses, and both metallib paths name the library, so their silence ruled them
out. **The nil-device path was the only one in `init_locked` that printed
nothing.** It now prints, and names the login-session cause.

**`mtlGetDeviceCount` computed the real error and then `return mtlSuccess`
unconditionally**, so `bench: cannot enumerate Metal devices: %s` was **dead
code** and a nil device, a refused GPU and an unloadable shader library all
arrived as the same line. New `mtlErrorNoDevice` distinguishes the transient
one; the other two stay permanent and keep erroring out. The generic test had
to be loosened to let the new code reach its own branch -- it sits first and
would otherwise have made the new handler unreachable.

**AND THE TASK NEED NOT HAVE BEEN LOST.** `boinc_finish(1)` marks a task
permanently errored and charges the host with a failure. New
`bench_boinc_temporary_exit()` asks the client to retry after
`BENCH_NO_GPU_RETRY_S` (600 s) instead. **It returns when BOINC is not managing
the run**, so a `HAVE_BOINC` binary launched from a terminal still reaches its
own error path -- `boinccheck` tests that half too, because the real
`boinc_temporary_exit` never returns and only the stub can prove the wrapper
does. Drift-ledger row added; nothing in the CUDA build calls it.

**`boinc_temporary_exit` is not `extern "C"`**, unlike `boinc_finish` --
BOINC's own API is not uniform. `boinclinkcheck` demangles instead of guessing
a symbol name, and caught this by failing when it guessed. 9 checks, control
still failing all four.

**Unverified:** the login-session diagnosis fits every fact in the log but was
not reproduced -- Metal device creation was not actually made to fail here. The
retry delay is a compromise with nothing behind it: nobody has measured how
long the condition lasts.

## 9z-m: the progress bar restarted at 0 on a resume

Reported from the field as a CUDA-side bug; **fixed Metal-side only**, since
`pipeline.cuh` and `bench_main.cu` are shared and a behaviour change there
would reach HIP.

**Everything in `run_pipeline_impl` knows about resume except the thing that
draws the bar.** `base_rel`/`base_nq` are what earlier sessions put on disk;
the `--target-rels` stop test adds `base_rel` ("counting only this session's
would make a resumed run sieve the whole target again from scratch"), the
checkpoint writer and every console q count add `base_nq`. The declaration
comment even claims they "are added to the goal tests and the progress line" —
they reach the console line's RELATION count and nothing else.
`pipe_progress_fraction` was handed neither.

**Three of its four branches were wrong, for two different reasons**, and the
denominators are the subtle half — `bench_main` has ALREADY shrunk them:

| branch | why it restarted |
|---|---|
| `target_rels` | both BOINC call sites passed this session's relations only (the console site already added `base_rel`, which is why this one looked right from a terminal) |
| `nq_max` / `nq` | `cfg->nq_max` is **reduced** by the completed count ("--nq counts this session's q") and `nqdone` restarts at 0 — so the ratio measured progress through the REMAINDER |
| q range | `cfg->qmin` is **overwritten** with the checkpoint's `next_q`, shrinking the span to what is left |

Fixed by passing `base_nq` in and totalling relations at every call site, and
by keeping the band's original lower bound in a new `cfg->resume_qmin` —
inert CUDA-side (the CUDA build zero-initialises `cfg` and never reads it),
drift-ledger row added.

**MEASURED, with the pre-fix binary as the control.** The run log is the right
instrument because ONE record carries both numbers: `nq=` is `base_nq +
nqdone` and has always been right, `pct=` comes from the estimator.

```
control  nq=141  pct=18.06      <- 141 of 200 q done, bar at 18%
control  nq=132  pct= 5.56      <- same checkpoint, 4/72 of the remainder
gate     nq=132  pct=66.00      <- 132/200, exactly
```

Both branches confirmed end to end by stopping a 200-q band at 128 q with
`--stop-file` and rerunning: `--nq` band 62% -> **18%** before, continuous
after; q-range band (no `--nq`, the branch needing `resume_qmin`) 23.29% ->
**3.01%** before, 22.17% -> **27.51%** after, which is `(q-q0+1)/span` to the
digit. Unresumed runs are unchanged — control and fixed both report 62.00% at
`nq=124`.

**GATED: `make -f Makefile.metal progresscheck`.** One expensive run makes the
checkpoint; the gate and its control each resume from a COPY of it, so the
control costs a short resume rather than a whole band. The control is a second
build compiled with `-DPIPE_PROGRESS_IGNORE_RESUME`, which reproduces the
shipped behaviour exactly and **must fail**.

**Two traps while building the gate, both of which looked like a result.**
Comparing against a binary built during a `git stash` dance gave the fixed
build's numbers as the control's — the stale-artifact trap this file already
records. And the gate's wait loop watched the run log GROW, which `--log`'s
per-run header satisfies instantly, so it killed the resumed band before it
reported anything and called that "no record".

## 9z-n: the same fix in the CUDA build, tested on a real NVIDIA card

9z-m fixed the resume bar Metal-side only. Asked to fix the CUDA build too, so
the fix now lives in **`pipeline.cuh` and `bench_main.cu` themselves** and the
Metal generators do nothing about it — one fix, all three ports. **This is a
real behaviour change and reaches HIP**, which shares those files; it is the
only such row in the drift ledger. Nothing about it is platform-specific.

**Verified on the hardware the bug was reported against**: GTX 1080 Ti
(sm_61, driver 580.159.04), nvcc 12.8.93, NRP/Nautilus pod, factor base
regenerated in-pod to the manifest hash `b4534cb6…`. The same
`metal/progresscheck.sh` — it is portable, it drives any build of this tree —
run against a **pre-fix build and the fixed one from the same checkpoint**:

```
pre-fix   resumed at nq=141: pct=18.06   (whole band 70.50)   3 assertions FAIL
fixed     resumed at nq=141: pct=70.50   (whole band 70.50)   all PASS
```

`18.06` is the identical number the Metal build produced pre-fix — same
deterministic band, same bug, both platforms.

**Regression, and it is the strong one: a full 288-q band on that card emits
`sha256 8e79762c…`, 13,485 relations, 564,696 enqueued — byte-identical to the
Metal build and to Phase 7.** Shared-code surgery moved no relation.

**THE GATE HAS NO CONTROL BINARY, AND THAT IS DELIBERATE.** The obvious
control — a build with the fix compiled out — **cannot exist**: this tree marks
any non-empty `DEFS` as a PRICING build and `bench` refuses `--relations` from
it, which a resume gate needs. That guard predates the gate and is right. So
`progresscheck.sh` DISCRIMINATES instead: it computes the pre-fix answer from
the same run-log record and asserts the build does not produce it, so reverting
the fix fails two assertions rather than none. The pre-fix binaries were
measured once, on both ports, above.

**AND THAT GUARD WAS MISSING FROM THE METAL BUILD — my fault, now fixed.**
`METAL_EXTRA_DEFS` (added for `cbtimecheck`'s control) never reached
`-DBENCH_DEFS`, because `Makefile.metal` forwarded no `DEFS` to the CPU
objects at all. So a Metal build with experimental `-D`s was **unmarked and
would happily write relations** — exactly what the CUDA Makefile's own comment
says must not happen ("anything that must stay shippable gets its own
variable, never a DEFS value"). `CPUOBJ_MAKEVARS` now forwards it; verified by
building with a `METAL_EXTRA_DEFS` and watching `--relations` get refused.

## Fourth review, 2026-09-16 (after 9z-g..9z-n and the rebase)

**Two suspicions raised by reading, both DISPROVEN by measurement. Recorded
because the next reader will have the same suspicions.**

1. **`pending` is bounded, not unbounded.** 9z-k made `commit()` push a
   retained command buffer onto a vector drained only by `sync()`, and
   `mtlStreamFlush` commits without syncing -- the shape of an unbounded leak,
   and a command buffer retains every resource it references. Measured
   high-water mark: **54 at `--nq 40`, 80 at 80, 80 at 160.** It plateaus. The
   80-at-80 coincidence is what made it look proportional.
2. **Memory is still flat over a band** (9z-f's property): RSS 370.8 -> 387.3
   MB over the first 40 s, then **+0.13 MB across the next 100 s**. But the
   PLATEAU IS ~19 MB HIGHER than 9z-f measured, which is the honest steady-state
   cost of holding up to 80 command buffers -- about 200 KB each. Immaterial on
   an 8 GB M1, and it is a working set, not a leak.

**FIXED: the CBTIME accounting ran whether or not anyone asked for it.** A
`std::string` assignment per DISPATCH and a 256-byte `snprintf` per COMMIT --
**~70,000 of each per 288-q band** -- building labels for output that is
discarded unless `CUDA_SIEVE_METAL_CBTIME` is set. Now gated on a cached
predicate next to `tracing()`. The instrument added in 9z-k was paying for
itself on every run instead of only when used.

**MEASURED, and it contradicts the shim's own header comment:** that comment
says dispatches "accumulate into that encoder, which is what keeps a band's
many small kernels cheap". They do not. Over an 80-q band: **19,506 dispatches
in 19,454 command buffers** -- essentially one each. The cause is pre-existing
and not 9z-k's flush (a 1-q run was 1039 command buffers BEFORE 9z-k and 1011
after): every blocking `mtlMemcpy` syncs, and every `mtlEventRecord` commits,
and the pipeline does both per q per slab per side. At a plausible ~20 us of
command-buffer overhead that is well under 1% of wall, so this is recorded as a
documentation-accuracy finding and a latent opportunity, **not** a defect worth
chasing -- but the comment should not be read as describing what happens.

**Re-checked, still true:** `mtlGetDeviceCount` has exactly one caller, so
9z-l's new return value cannot be silently dropped; validation is clean
(`validationcheck`); the three 9z/9z-b open items are unchanged and still minor
(zero-sized launch silently skipped, `pso_for` caches `nil`, `mtlDeviceAddress`
lacks its sibling's warning).

Nine gates green after the fix, `cofcheck.sh` **54 PASS / 0 FAIL**, 288-q band
`cmp`-identical at 13,485 relations / `8e79762c…`.

## 9z-o: the fbgen slice size is MEASURED now, not chosen

**Field data, 2026-09-17: M1, M1 Max and M2 workunits now COMPLETE**
(`outcome 1`, `boinc_finish(0)`) — the first ones ever, and they close 9z-k's
open item: the cofactor half is field-proven, and the chunker's steering is
visible in the logs (36864 -> 15369 -> 11930 on an M1 Max).

**But the fbgen wall times say the margin was luck.** This M3 does fbgen in
6.9 s with a measured 130 ms per command buffer at 16 strides. The field:

| device | fbgen wall | cores | slices | per command buffer |
|---|---|---|---|---|
| M3 (Apple9) | 6.9 s | 10 | 50 | **130 ms** (measured) |
| M1 | 41.3 s | 7-8 | ~62-71 | **~560-640 ms** (derived) |
| M1 Max | 13.4 s | 24-32 | ~16-21 | **~640-840 ms** (derived) |

Against a 400 ms policy and an ~800 ms observed kill line. **The M1 Max is the
WORST because more cores means a wider `wave`, so each stride carries more
primes.**

**THAT IS 9z-j's ERROR ONE LEVEL UP.** 9z-j replaced a fixed prime count with a
fixed stride count and assumed strides were the device-independent unit. They
are not: a stride is `ablocks*128` primes and `ablocks` scales with core count,
while per-core throughput does not scale with it the same way. **No fixed unit
is device-independent, and `cbtimecheck` can never catch this because it only
runs on this M3.**

**Fix: start safe and measure.** `FB_ROOTS_STRIDES_START` = 4 (~140-210 ms on
the M1 family), then steer proportionally at 0.8x the bound with cofq_flush's
dead band. On this M3: segment 1 runs at 4 strides, then settles at ~32 —
49 slices, **max 311 ms**, fbgen 6.54 -> 6.74 s (~3%). An M1 should settle near
8 strides.

**Three mistakes on the way, all of which measured as "working":**

1. **The CPU races the GPU, so polling between submissions sees nothing.**
   Eight pending command buffers, all still merely *Committed* — encoding is
   microseconds, execution is tens of ms — and then `sync()` releases them and
   the durations with them. The measurement has to be taken **at the drain**;
   `mtlStreamWorstMs` reports the worst buffer of the last one, worst because
   that is what the watchdog judges.
2. **A stream-wide measurement is not a root-finder measurement.** Steering
   inside the slice loop read whatever stage last drained — the odds sieve, the
   select, the scan — and each was a fresh `seq`, so the controller grew on
   every one of them, hit the 256 ceiling *inside the first segment*, and put
   the whole segment in one **788 ms** command buffer: worse than the constant
   it replaced. It now steers **once per segment**, on the drain that contains
   the slices.
3. **A stale reading applied twice compounds.** Hence `seq`: one measurement,
   one adjustment. Without it a 4 -> 38 grow becomes 4 -> 38 -> 256.

**THE GATE'S CONTROL CAUGHT ITSELF.** `cbtimecheck`'s control still passed
`-DFB_ROOTS_STRIDES_PER_LAUNCH`, a macro this change deleted, so it defeated
nothing and had quietly become a second copy of the gate — and the gate failed
rather than passing, which is the entire argument for a control that must FAIL
over an assertion that must pass. Now `-DFB_ROOTS_STRIDES_START=4000000u
-DFB_ROOTS_STRIDES_MAX=4000000u`: control 803 ms, gate 317 ms.

**ONE DEFINITION OF THE BOUND.** `MTL_INTERACTIVITY_BOUND_MS` (400 ms) now
lives in `metal_rt.h` — the command buffer is the shim's own concept — and
`COF_CHUNK_TARGET_MS` is defined from it. Two stages bound themselves against
this number; two constants that must agree is how this port has repeatedly
hurt itself.

**Not measured:** the M1 figures above are DERIVED from wall time and an
assumed core count, not measured per-buffer. The controller does not depend on
them being right — that is the point of measuring — but the table should not be
quoted as if it were instrumented.

**Also in those logs, unfixed:** an M1 Max task restarted **six times**
(`lock is stale (pid … is gone)`), each restart re-running fbgen (13.5 s) and
re-learning the cofactor chunk from 36864, paying another 744-790 ms
over-bound launch each time. The learned chunk is not carried in the
checkpoint.

## 9z-p: a killed segment is REDONE, not lost -- and the alim was the tell

The persistent M2's log named its own cause in its FIRST line:

```
generating algebraic factor base on GPU through 250000000
... 60% through prime range
metal_rt: command buffer failed: internal (code 1): Impacting Interactivity
metal_rt: command buffer failed: internal (code 1): Impacting Interactivity
```

**`alim` is 250,000,000 there and 134,000,000 in every log that succeeded.**
Not a different device problem -- a bigger job: ~15 segments instead of 8, at
primes up to 250M. 9z-o's measured slice size covers that, because it stops
caring what the job or the device is.

**But TWO buffers were killed in one drain, and that half is not a sizing
problem at all.** The interactivity watchdog is **contention-dependent** -- it
fires when the GPU is wanted elsewhere -- so no slice size, measured or
chosen, can guarantee it never fires. A volunteer opening a game can kill a
correctly-sized buffer. Sizing and recovery are different jobs and the port
only had one of them.

**The root finder is IDEMPOTENT, which is what makes recovery cheap:**
`d_rootbuf`, `d_counts` and `d_special` are indexed by prime and simply
rewritten, `d_failures` is re-zeroed, and **nothing has reached the sink** --
the scan, `k_total_roots` and every `sink->` call are below the failure point.
So a killed segment is re-derived from the same inputs. Up to
`FB_ROOTS_MAX_ATTEMPTS` (4) tries, **halving the slice each time**: the kill is
the controller's strongest input, the one measurement that says "too long"
without a timer. Measured cost of a recovery: ~1 s, output identical.

**FAULT INJECTION, because the watchdog cannot be provoked on demand.**
`CUDA_SIEVE_METAL_FAULT_SYNC=N` fails the Nth sync (N<0: every sync from |N|
on) **without corrupting anything** -- the work really ran, so a correct retry
must produce the same factor base, and that is the property worth asserting.
`make -f Makefile.metal fbretrycheck`: 6 checks, and its control is built in --
a clean run must NOT retry, or the gate is vacuous.

**IT IMMEDIATELY FOUND TWO BUGS IN MY OWN RETRY, both of which "worked":**

1. **Every injected fault produced exactly TWO retries.** `mtlMemcpy`'s failure
   also sets the sticky last-error, so the next attempt read it back and
   "failed" having done nothing wrong. The attempt now consumes it first.
2. **The second halving went UP -- 34 -> 17 -> 256 -> "halve" to 128.** The
   steer ran on the FAILED attempt, measured the surviving short buffers as
   cheap, and grew the slice straight back over the halving. **The recovery was
   making the next attempt four times worse.** It now steers only on success.

Neither would have been visible without executing the path. **A recovery path
that has never run is not a recovery path.**

**Still not exercised:** the 4-attempt exhaustion branch itself. An always-fail
injection dies earlier, in an `MTL_OR_DIE` on the tiny memset, so what is
verified is that an unrecoverable fault TERMINATES (2.1 s, exit 1) rather than
looping -- not the counter.

Eleven gates green, `cofcheck.sh` 54/0, 288-q band `cmp`-identical at 13,485
relations / `8e79762c…`.

## What the interactivity work COSTS a fast, successful machine (measured)

Everything since 9z-j buys not losing workunits. Priced on this 10-core M3,
against the pre-9z-j build (`cbddd0c`) built from a worktree and run
alternately to cancel thermal drift:

| component | cost |
|---|---|
| **fbgen slicing + flush** | 6.479 -> **6.684 s**, +0.205 s, **+3.2% of fbgen** (3 runs each, spread ±0.04) |
| **cofactor per-launch flush** | **below noise** -- 52.89 vs 52.97 ms/q algebraic queue, 733.6 vs 736.3 ms/q wall, against a 12 ms within-variant spread |
| **the 400 ms bound forcing a smaller chunk** | **+23% of the algebraic queue stage, ~1.5% of wall** |
| CBTIME accounting, steering, retry | zero -- gated off, 8 measurements, and nothing unless a kill happens |

**The bound is the only real cost, and it is only paid where the bound
BINDS.** Not on this M3: it opens at 15,360 records/launch, measures ~240 ms,
and never steers. It binds on the fast parts -- the field M1 Max opens at
36,864, measures 768 ms and settles near 11,930 -- so those pay it. Measured by
pinning both chunks here with `--cof-chunk`:

```
chunk 36864   algebraic queue 46.96, 47.21 ms/q
chunk 11930   algebraic queue 57.61, 58.05 ms/q
```

**This retires 9z-h's "the throughput cost of meeting the bound is unknown".**
It was unreadable then because it was measured as WALL, where 10.7 ms/q hides
inside a ±12 ms spread. Measured on the STAGE it resolves cleanly. It also
qualifies 9z-h's linearity result: a single launch is linear in the chunk, but
the STAGE is not, because more launches means more per-launch overhead --
~2 ms each, from 4 launches/round to 11.

**fbgen's 0.205 s is per PROCESS, not per q** -- 0.09% of a 288-q band and
immaterial against a workunit measured in hours, though a host that restarts
repeatedly pays it again each time (the field M1 Max restarted six times).

## Fifth review, 2026-09-17 (after the two rebases and the lock fix)

**THREE FINDINGS, ALL THREE MEASURED RATHER THAN READ, AND ALL THREE WERE
INTRODUCED BY THE TWO REBASES.** Inheriting upstream code is not free: each of
these is a piece of shared code that is correct on CUDA and wrong here, or
correct in isolation and wrong in the order it landed.

### 1. Upstream's launch valve was undone on the very next flush (CUDA)

`8b62c81` let an over-bound launch descend past `cof_chunk_floor()`. It did --
for exactly one flush. The steering below it clamps UP to that same floor every
flush, and its test against `stage` is the always-true one 8b62c81 itself
documents. Measured on a **TITAN RTX** (sm_75, nvcc 12.8.93, NRP) at the
shipped 1000 ms bound:

```
cofactor chunk: 110592 records/launch                  <- cof_chunk_floor()
cofactor: kernel launch 1172 ms ...; 110592 -> 75466 records/launch
cofactor chunk: 110592 records/launch                  <- snapped back
cofactor: kernel launch 1168 ms ...; 110592 -> 75735 records/launch
cofactor chunk: 110592 records/launch                  <- and again
```

**Every second flush ran at the value just measured over the bound**, which on
the 980 Ti this was written for is a TDR kill, and one is enough. Fixed on
`main` (`252fef4`) with a sticky `chunk_ceiling` the floor yields to; this port
inherits it through regeneration and clamps its growth branch with it too.

**8b62c81's own V100 verification could not have seen this.** At the 50 ms bound
it tested, the valve fires on EVERY flush, `valve_acted` suppresses the
steering, and the code that snaps back never runs. The configuration that proves
the descent works is the one that hides the relapse.

### 2. The over-bound launch report went dead in the rebase (Metal)

`cofac_host.inc` read `launch_ms` **above** the point upstream's valve assigns
it. `launch_ms` is a fresh local per call, so the test was always false and 9c's
report -- the line a volunteer's `stderr.txt` carries when a host cannot meet
the bound, kept precisely because it "is invisible from anywhere else" -- had
been dead since `5c7b75c`, **including in the binary that was signed for
deployment**. Same shape as 9z-l's `mtlGetDeviceCount`: the value is computed,
then read in the wrong order.

Measured, not read: a build at `COF_CHUNK_TARGET_MS=1.0f` printed the valve's
own line at 238 ms and this report not at all. After the fix the same build
prints `kernel launch 241 ms is over this build's 1 ms bound`. The measurement
is now hoisted above the report and **out of auto mode**, which 9c also asked
for and upstream's placement had quietly taken away.

### 3. The inherited valve pre-empted this port's own controller (Metal)

`valve_acted` skips the steering for that flush, so a launch over the **1000 ms**
valve bound got a correction aimed at 0.8x1000 instead of 0.8x this build's own
**400 ms**, and the no-progress guard did not run either. Measured at a 200 ms
valve bound against a 1 ms steering bound: **two consecutive flushes, ~134
special-q**, steered by the valve at 0.67x per step while the primary bound
wanted 0.33x. Those are M1-at-opening-chunk flushes -- command buffers in the
range macOS has actually been observed killing.

**Upstream's valve is a backstop for a floor this port does not have**:
`cof_chunk_floor()` here is one WAVE, not one grid (8j), so the steering can
already descend past the point the valve exists to reach. So the valve now
records the CEILING only -- the one thing only it can say, that a measured
launch has disproved the floor -- and the steering keeps the flush. `valve_acted`
is gone. The same probe now reaches the target on flush **1** instead of flush 3.

### GATED: `make chunkcheck` on main -- and its first version passed its own control

`cofcheck.sh` exercises only PINNED `--cof-chunk` values, and both the valve and
the steering are gated on `!chunk`, so **the entire adaptive path -- the one
every production band runs -- had no gate at all.** That is how finding 1
shipped.

**The first `chunkcheck.sh` lowered the valve bound to 200 ms with `DEFS` so the
valve would fire on any card, and the control PASSED.** At a bound the device
cannot reach the valve fires every flush and suppresses the steering, so the
code under test never executes -- **the identical blind spot as 8b62c81's V100
run, reproduced in the gate written to catch it.** The gate now runs the
SHIPPING binary at the SHIPPED bound and uses `--ecm-curves` to lengthen the
launch, and fails loudly if no launch reaches the bound rather than passing on
an assertion that never ran.

```
make chunkcheck                 valve fires ONCE, 110592 -> 74921, and the
                                controller is silent for the remaining 200 q
                                6/6 PASS
DEFS=-DCOF_CHUNK_NO_CEILING     snaps back to 110592
                                3 assertions FAIL, exit 1
```

### Verified

CUDA: `cofcheck.sh` **54 PASS / 0 FAIL**, 288-q band **13,485 relations, sha256
`8e79762c…`**, `c183.fb1` regenerated in-pod to the manifest hash `b4534cb6`.
Metal: fourteen gates green, `cofcheck.sh` **54 PASS / 0 FAIL**, 288-q band run
the field's way with no `--fb1` **`cmp`-identical at 13,485 / `8e79762c…`**,
worst command buffer **190.80 ms**.

### Reviewed with no finding

Zero generator drift -- all twelve `gen_*.py` reproduce their committed output
byte for byte. The lock fix covers every non-returning exit: both
`boinc_temporary_exit` sites are either inside `bench_boinc_finish` (after the
release) or at `bench_main_metal.cpp:1841`, before the lock is taken. The fbgen
slice loop cannot spin (`cnt >= 128` always) and its interior-pointer offsets
stay 128-byte aligned, so the unified-memory `setBuffer:offset:` requirement the
family gate guarantees is met; `d_rootbuf` is freed per segment; and
`MTL_OR_DIE(mtlStreamFlush(0))` inside the retry cannot bypass the retry,
because `mtlStreamFlush` cannot fail.

### Smaller, fixed in the same pass

- **`mtlStreamWorstMs` is gone**, with the `worst_ms`/`worst_seq` bookkeeping
  that fed only it. The second rebase replaced its drain-based measurement with
  upstream's event bracket; CLAUDE.md flagged it for "remove or justify at the
  next review" and this is that review.
- **`sync()` re-implemented `cbtiming()`** with its own cached static, after the
  fourth review added the shared predicate for exactly this. One copy now.
- **`COF_BOUND_MS` in `Makefile.metal` duplicated `MTL_INTERACTIVITY_BOUND_MS`
  by hand.** Lower the header and the gate kept asserting the old, looser number
  -- passing, vacuously. Derived from the header with `sed` now, so there is one
  definition of the bound in the tree and the gate cannot drift loose of it.
- **`fbretrycheck` claimed more than it proved.** Its header says a correct
  retry must produce THE SAME factor base; it compared only the summary counts,
  so a retry with the same counts and different roots passed. It now also pins
  the relation count, which is already in the output it captures.

### Recorded, not changed

**The fault injection ships.** `CUDA_SIEVE_METAL_FAULT_SYNC` is compiled into
the distributed BOINC binary and is settable by anyone who controls the process
environment. The failure mode is a clean error exit rather than a wrong answer,
and a volunteer controls far more than that already -- but it should be a
decision on the record rather than a side effect of `fbretrycheck` needing it.

## Rebase, 2026-09-17 (second): the root-finder SLICING went upstream too

The field reported **interactive stutter at workunit start** on CUDA -- no
crash, just an unresponsive desktop for the several seconds of factor-base
generation, which is the milder relative of the cofactor's TDR problem and
produces no error to diagnose from. Fixed in `fbgen_gpu.cu` itself
(`0acfd8c`), so `FB_ROOTS_STRIDES_START/MAX`, `FB_LAUNCH_TARGET_MS`,
`g_fb_strides`, `fb_steer_strides()` and the event bracket are shared code now.
`gen_fbgen_host.py` shed **227 lines against 160 added**.

**Two things stay Metal-only, for two DIFFERENT reasons:**
- the **per-slice `mtlStreamFlush`** -- CUDA needs nothing, because a kernel
  launch is already the unit its watchdog sees; here the stream batches every
  slice into one command buffer, so without it the slicing bounds nothing;
- the **retry** -- CUDA *cannot* do it, because a TDR reset destroys the
  context. macOS's interactivity kill leaves the device usable.

**Metal's drain-based measurement is gone, and upstream's event bracket suits
it better.** On Metal an event record COMMITS the stream's command buffer, so
bracketing slice 0 puts that slice in a buffer of its own and
`mtlEventElapsedTime` measures exactly its GPU duration -- no need for the
"worst of the last drain" query the CPU/GPU race forced on 9z-o.
**`mtlStreamWorstMs` is now unused in `metal_rt`** and should be removed or
justified at the next review.

Upstream's target is 250 ms against Metal's old 400, so this is MORE
conservative than before: worst root-finder command buffer **311 -> 176 ms**,
fbgen 6.68 -> 6.92 s (+3.6%). The right direction for the M1/M2 hosts this
started with.

**Validated:** eleven gates green including `cbtimecheck` and `fbretrycheck`,
`cofcheck.sh` **54 PASS / 0 FAIL**, 288-q band `cmp`-identical at **13,485
relations / `8e79762c…`**.

## Rebase, 2026-09-17: the peak-launch mechanism went UPSTREAM

A field **980 Ti** hit the same class of failure on CUDA -- it parked at
`cof_chunk_floor()` (22 SMs x 6 x 256 = exactly the 33,792 in its log) and the
launch exceeded the Windows 2 s TDR. Fixed in `cofac.cuh` itself (`8b62c81`),
so **`cof_peak_t`, `g_cof_peak`, the round-0/slice-0 bracket and the pk0/pk1
arming are now shared code** and this port inherits them through portlib's
renames. `gen_cofac_host.py` shed **three whole steps** -- 122 lines removed
against 41 added.

**Two things stay Metal-only, and the second is the subtle one:**

1. The **per-launch `mtlStreamFlush`**, because the watchdog here judges a
   command buffer (9z-k). CUDA has no such object, so upstream needs nothing.
2. **Steering the chunk on the measured launch rather than on `stage`.**
   Upstream added a one-way VALVE at `COF_LAUNCH_TARGET_MS` (1000 ms) and
   **deliberately left its `stage` test alone**, so that test is still
   always-true; inherited unchanged it would park this build at the floor
   exactly as before. So the substitution still has to happen in the generator.
   Upstream's valve then sits below Metal's own steering as a backstop and, at
   1000 ms against Metal's 400, should never fire.

The one adaptation: upstream computes `launch_ms` inside its valve's own block,
so the generator hoists it to function scope.

**Validated:** eleven gates green (including `cbtimecheck` and the new
`fbretrycheck`), `cofcheck.sh` **54 PASS / 0 FAIL**, worst command buffer
**253.73 ms `[1 dispatches]`** -- the 9z-k property survives -- and the 288-q
band `cmp`-identical at **13,485 relations / `8e79762c…`**.

## Rebase, 2026-09-16: upstream's ECM occupancy fix, and a silent kernel leak

`metal-port` was 6 commits behind `main` and is now rebased onto `75d4cf7`
(75 commits ahead, `main` an ancestor). The one that matters is **`2bc1c6e`,
"Fixing a minor regression in some of the recent ECM code"** -- not a
correctness fix but an **occupancy** one, in `cofac.cuh`, which this port
GENERATES from and never edits:

- `__launch_bounds__(256, 2)` on `k_cofac`. At 256 threads, 128 registers is a
  cliff (two blocks per SM), and ECM stage 2's shared denominator had pushed
  the kernel one register over it -- costing a 5070 half its occupancy. Worth
  **-10.3%** of the cofactor stage there, relations md5-identical.
- **`COFAC_THREADS_MAX` 256 as a HARD ceiling** (512 fails outright with
  "invalid argument"), with `cf_run_rounds` and `cof_chunk_floor` clamping.

**What carries to Metal: the clamp. What does not: the bound.**
`__launch_bounds__` is a CUDA register directive with no MSL equivalent, so
`gen_cofac_metal.py` strips it before parsing -- **only the declaration form**,
because a blanket strip also edits the word inside `cofac.cuh`'s own comment.
MSL's nearest relative is `[[max_total_threads_per_threadgroup(n)]]`, which is
a real untried optimisation here and was **deliberately not adopted silently**:
it is a performance change and wants measuring on this box. The clamp does not
bind at this build's default `--threads 256`.

**THE NEAR-MISS: a kernel count went 9 -> 8 and nothing failed.**
`gen_cofac_host.py` strips `__global__` definitions out of the host file with
`(?:template<...>)?__global__\s+void\s+\w+\s*\(`. Upstream's new shape puts
`__launch_bounds__` between `__global__` and a `void` on the NEXT line, so the
pattern stopped matching `k_cofac` and **its entire device definition was left
sitting in the host translation unit**. The only signal was the number in the
generator's own output line. Both fixed, and **the count is now asserted**
(`assert nk == 9`), because an anchor that silently matches nothing is this
port's most-repeated failure.

The two generators that failed LOUDLY (`gen_cofac_metal.py` on its kernel
pattern, `gen_cofac_host.py` on `cof_chunk_floor`) cost minutes. The one that
failed quietly would have shipped.

**Validated after the rebase:** fourteen gates green -- `metalcheck`,
`rtcheck`, `scancheck`, `argbufcheck`, `classifycheck`, `fbcheck`,
`sievecheck`, `cofaccheck`, `validationcheck`, `cbtimecheck`, `boinccheck`,
`metallibcheck`, `progresscheck` -- plus `cofcheck.sh` **54 PASS / 0 FAIL**.

## Drift ledger — CUDA-side changes made for this port

**The ledger lives in `bench/METAL_PORT_PLAN.md` section 9, and only there.**
A copy of it used to sit here and had already drifted to "none yet" while the
plan carried two rows — precisely the failure the ledger rule exists to catch,
committed by the ledger itself. Seven rows as of 2026-09-15: `cofcheck.sh`
(`head -c -1`), `fbgpucheck.sh` (`sha256sum`), `slab.h` (`SLAB_PERF_REGIONS`
made overridable, default unchanged), `runlog.c`/`.h` (`g_runlog_quiet`),
`boinc_support.cpp`/`bench.h` (`bench_boinc_progress_suspend`), and
`cofcheck.sh` again (build detection; the `--ecm-b1 400000` case asserts a
refusal on Metal), and `td.cuh`
(`TD_TILE` made overridable, default unchanged). Two of those
are ports from `hip-port` and are inert unless called, which only the Metal
build does. Add new rows there.
