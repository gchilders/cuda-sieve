# cuda-sieve — Metal port

Porting the CUDA NFS lattice sieve in `bench/` to Metal Shading Language for
Apple Silicon. The plan of record is **`bench/METAL_PORT_PLAN.md`** — read it
first; this file carries the running ledger and the rules.

Branched from `main` at `3e15fec`. `hip-port` is a reference for method, not
a base: it is 23 commits behind `main`.

## This machine
- Apple M3 MacBook Air, 10-core GPU, `MTLGPUFamilyApple9`, 16 GB unified.
- macOS 26.5, Xcode 26.5 (17F42), Metal Toolchain 17F42.
- `xcode-select -p` points at CommandLineTools, which ships **no** `metal`
  compiler. Do not `sudo xcode-select -s`; export
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` instead.
- **Correctness vehicle only.** The GPU also drives the display, the memory
  is a shared UMA pool, and there are 10 GPU cores. Never draw a performance
  conclusion from this box. If a number from here ever has to become a
  default, say so explicitly at the point it ships — the HIP port's
  `SLAB_PERF_TARGET_LOG2` entry is the model for how to flag that honestly.

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

## Status
- **Phase 0 (toolchain + probe): DONE.** Probe sources in `metal-probe/`.
- **Phase 1 (branch + ledger): DONE.**
- **Phase 2 (portability primitives): DONE, gate green.**
  `cd bench && make -f Makefile.metal metalcheck` — 6.6M results compared,
  0 mismatches. `softfp64.h` (IEEE binary64 in integer ops),
  `portable_log2.h`, `msl_compat.h`, `sf_sites.h`.
- Phases 3-9: not started. Next is Phase 3, the `metal_rt` runtime shim —
  the decision the rest of the port's effort hinges on (plan section 6).

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
