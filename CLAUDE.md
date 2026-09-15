# cuda-sieve — Metal port

Porting the CUDA NFS lattice sieve in `bench/` to Metal Shading Language for
Apple Silicon. The plan of record is **`bench/METAL_PORT_PLAN.md`** — read it
first; this file carries the running ledger and the rules.

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
- Phases 5-9: not started. Next is Phase 5, the sieve kernels
  (`k_transform`, `k_fill_*`, `k_apply`) and the `pipeline_metal.cpp`
  orchestration, including the soft-fp64 norm fallback.

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

| date | CUDA file(s) | change | verified how |
|---|---|---|---|
| — | — | none yet | — |
