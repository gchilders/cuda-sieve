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

Everything else checked out clean: SIMD width 32, `simd_ballot` in ascending
lane order (so `td.cuh:647` holds), `simd_shuffle_up` matching CUDA,
`mulhi(ulong,ulong)` exact against `__int128`, templated kernels with
`[[host_name]]`, and `device` pointers inside argument-buffer structs
(Tier 2).

## Status
- **Phase 0 (toolchain + probe): DONE.** Probe sources in `metal-probe/`.
- Phase 1 (branch + ledger): this commit.
- Phases 2-9: not started.

## Drift ledger — CUDA-side changes made for this port

| date | CUDA file(s) | change | verified how |
|---|---|---|---|
| — | — | none yet | — |
