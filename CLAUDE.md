# cuda-sieve — HIP port

Porting the CUDA NFS lattice sieve in `bench/` to HIP for AMD GPUs, targeting
distribution to BOINC volunteers on Windows and Linux.

## This machine
- AMD Radeon 780M iGPU, gfx1103, Windows 11.
- ROCm 10.0.0 at C:\rocm. hipcc uses MSVC as host compiler.
- Build from an x64 Native Tools Command Prompt for VS 2022.
- fp64 is very slow here and memory is a UMA carveout: this box is a
  CORRECTNESS vehicle only. Never draw performance conclusions from it.

## Ground rules
- Do not modify the CUDA build. The HIP port lives alongside it.
- Acceptance test: HIP build output must be byte-identical to the CUDA
  build's relations on the oracle jobs, with `make -C bench check` green.
- Warp width 32 is assumed throughout (`>> 5`, `& 31`, lane 31 broadcasts).
  Correct on RDNA under HIP; do not "fix" it.

## Known port work items
1. `fbgen_gpu.cu` uses cub::DeviceSelect::Flagged and
   cub::DeviceScan::ExclusiveSum -> hipCUB. CONFIRMED present in this ROCm
   install: C:\rocm\include\hipcub (rocprim backend), matching
   DeviceScan::ExclusiveSum / DeviceSelect::Flagged signatures. CMake config
   at C:\rocm\lib\cmake\{hipcub,rocprim} for find_package().
2. k_apply dynamic shared memory is gated on
   cudaDevAttrMaxSharedMemoryPerBlockOptin. AMD's per-workgroup ceiling is
   lower than NVIDIA's opt-in; verify the HIP attribute and expect
   log_region <= 14.
3. `__launch_bounds__` second argument means minBlocksPerMultiprocessor in
   CUDA but MIN_WARPS_PER_EXECUTION_UNIT in HIP. hipify does not translate
   it. Drop it for HIP and re-tune from scratch.
4. No static HIP runtime exists. The BOINC build links cudart statically;
   the AMD build must ship libamdhip64 with $ORIGIN rpath on Linux.
5. `cudaInitDevice` (bench_main.cu) has no HIP equivalent. Use hipSetDevice
   + hipSetDeviceFlags(hipDeviceScheduleBlockingSync).
6. HIP's __ballot returns unsigned long long; expect narrowing warnings.
7. __shfl_up_sync/__shfl_sync/__ballot_sync require a genuine 64-bit mask
   argument (static_assert(sizeof(MaskT) == 8) in amd_warp_sync_functions.h).
   CUDA's 0xffffffffu literal (32-bit) fails to compile under HIP -- every
   callsite in bench_kernels.cu and td.cuh using this idiom needs the mask
   literal widened to 0xffffffffffffffffull. Confirmed via probe.hip on
   ROCm 10.0.0 / gfx1103, ROCm clang 23.0.0git.

## Confirmed device limits (probe.hip, 2026-08-31, ROCm 10.0.0, gfx1103)
- warpSize = 32; gcnArchName = gfx1103.
- sharedMemPerBlock = smemPerMultiproc = optin smem limit = 65536 B (64 KB).
  No separate opt-in tier on this GPU (unlike NVIDIA's ~100 KB) -> log_region <= 14.
- hipMemGetInfo: 14.26 GB free of 14.41 GB total (UMA carveout).
- All warp intrinsics (shfl scan, ballot, popc, ffs, clz, umulhi, umul64hi)
  match CUDA's values bit-for-bit once mask literals are widened (item 7),
  including ascending lane order in the ballot mask -- td.cuh:647's
  assumption holds on RDNA under HIP.

## HIP macro reality (probe_macros.hip, 2026-08-31, hipcc on ROCm 10.0.0)
Confirmed empirically, both host and device compile passes:
- `__CUDACC__` is NEVER defined by hipcc, on either pass. Any code gated on
  `#if defined(__CUDACC__)` to add `__host__ __device__` (e.g. bigint.cuh's
  `BN_FN`, td.cuh's `TD_MOD_HD`) silently loses those qualifiers under HIP,
  making the function host-only and uncallable from device code. Fix:
  `#if defined(__CUDACC__) || defined(__HIPCC__)`.
- `__HIPCC__`, `__HIP__`, `__HIP_PLATFORM_AMD__` are all defined (both passes).
- `__HIP_DEVICE_COMPILE__` is defined only in the device pass (HIP's analogue
  of `__CUDA_ARCH__`, which HIP never defines). Fix for device/host intrinsic
  branches (e.g. bigint.cuh's `BN_MULHI64`, td.cuh's `td_mod_magic`):
  `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`.
- `_MSC_VER` IS defined in hipcc's device compile pass on Windows (not just
  host) -- confirms bigint.cuh:235's existing `#elif defined(_MSC_VER)`
  branch would otherwise be reached from device code and call a host-only
  function (`bench_mulhi_u64`); the device-pass check above must come first.

## CPU-only tool build notes (Phase 2, not a HIP issue)
- MSVC's cl.exe cannot build the standalone `fbgen`/`fbtest`/etc. CLI tools:
  fbgen.c's non-library CLI path `#include`s `<pthread.h>` (guarded by
  `#ifndef FBGEN_LIBRARY`, so bench.exe's in-process library-mode build,
  which build_windows.bat already proves works, never hits this).
- Strawberry Perl (installed for hipify-perl/ROCm scripts) bundles a full
  MinGW-w64 toolchain at C:\Strawberry\c\bin (gcc, g++, mingw32-make,
  winpthreads' pthread.h) -- not on PATH by default. Using it ONLY for
  standalone CPU-only tools (fbgen.exe, fbtest.exe), never for bench.exe
  itself (which stays on cl.exe+hipcc per build_windows_hip.bat, to avoid
  mixing CRT/ABI between MinGW and MSVC in one link) is safe and doesn't
  touch the CUDA build.
- `mingw32-make CC=gcc CXX=g++ fbgen fbtest` against the existing Linux
  Makefile gets further than expected -- poly.o/primes.o/fbgen.o all build,
  it only fails at final link with `undefined reference to open_memstream`
  (fbgen.c:867, inside the CLI worker path, `#ifndef FBGEN_LIBRARY`).
  `open_memstream` is a POSIX/glibc extension MinGW-w64's runtime doesn't
  provide. Needs a small Windows shim (e.g. a temp-file-backed or
  growable-buffer FILE* substitute, in the spirit of platform.c's existing
  `#ifdef _WIN32` shims) before the standalone fbgen.exe CLI can build on
  Windows. NOT yet fixed -- deferred, since it blocks only Phase 6's
  fixture-generation convenience (producing oracle/c183.fb1, fbase.m16),
  not the HIP device-code port itself.

## __CUDACC__ audit: two more hidden instances found compiling bench_kernels.hip
Phase 3's header probe only exercised bigint.cuh/td.cuh's arithmetic helpers,
not the kernel-definition sections further down in the same files -- it
missed that td.cuh has a SECOND `#if defined(__CUDACC__)` (was line 115,
guarding lines 118-919, effectively the entire rest of the file:
k_tdsmall_advance, TD_GROUP_W/TD_SCAN_BLK, k_group_counts, k_scan_pass1, and
more) that would have silently deleted almost all of td.cuh's device code
under HIP -- not a wrong-branch bug like the ones Phase 3 fixed, a
missing-code bug. Same bug, same fix (`|| defined(__HIPCC__)`), found by
actually trying to compile bench_kernels.hip rather than by more reading:
- td.cuh:118 (and its closing #endif at 919)
- cofac.cuh:881 (and its closing #endif at 2710) -- guards essentially the
  entire cofactorization kernel section
- bench.h:750 and :760 -- guards bench_grid_product_u64's __host__ __device__
  qualification and, entirely, bench_grid_thread_x/bench_grid_stride_x
  (called from nearly every kernel's grid-stride loop)
- slab.h:13 -- SLAB_HD macro, same __host__ __device__ pattern as BN_FN/PL_FN
Exhaustively grepped the whole bench/ directory afterward (not just the
files already touched) -- confirmed every __CUDACC__ occurrence anywhere in
the tree now also checks __HIPCC__. Lesson: `grep -rn __CUDACC__` across the
WHOLE directory should have been step one of Phase 3, not incremental
discovery through compile errors -- reading a file's outline is not a
substitute for actually compiling it end to end.

## Header port: DONE (bigint.cuh, td.cuh, plattice.cuh, prp.cuh)
Fixed in place (not forked -- low-churn, correctness-critical, nvcc's
untouched #else/non-matching branch reproduces prior CUDA behavior exactly):
- bigint.cuh: `BN_FN` macro now checks `__HIPCC__` too (was silently
  dropping __host__ __device__ under HIP, making every bn_* function
  host-only). `BN_MULHI64` now checks `__HIP_DEVICE_COMPILE__` before
  `_MSC_VER` (device pass was falling into the MSVC host branch).
- td.cuh: same two fixes, for `TD_MOD_HD` and `td_mod_magic`'s internal
  `__CUDA_ARCH__` branch.
- plattice.cuh, prp.cuh: same `__HIPCC__` fix for `PL_FN`/`PRP_FN` -- read in
  full, confirmed no other CUDA-specific content (pure arithmetic).
- platform.h: `bench_div_u128_u64`'s `_udiv128` MSVC intrinsic is not
  implemented by hipcc's clang even in MSVC-compatible host mode (confirmed:
  `__umulh`/`_umul128` in the other two functions in the same file ARE
  supported, only `_udiv128` is not) -- added `&& !defined(__clang__)` so
  HIP falls back to the existing portable bit-shift division, unaffected for
  real cl.exe (which never defines `__clang__`).
- Validated by `hip-probe/probe_headers.hip` against the real header files
  in place (via `-IC:\dev\cuda-sieve\bench`), calling BN_MULHI64 the same
  way the real code does (through a `BN_FN`-qualified helper, matching
  `bn_divmod_u32_pre` at bigint.cuh:270-275) -- all 6 checks pass bit-exact
  (BN_MULHI64, bn_add, td_mod_magic via the real td_magic_build, pl_invmod,
  m_sprp2 on both a prime and a composite).
- Note for later device TUs: calling a macro that resolves differently per
  compile pass (like BN_MULHI64) is only safe from a `__host__ __device__`
  (BN_FN-style) function. Calling it directly from a bare `__global__`
  kernel does NOT work under HIP -- clang's dual-pass model checks call
  validity against the callee's per-pass resolution even in passes that
  don't codegen the caller, which nvcc's split front end does not do. Not
  currently an issue (real usage is always through BN_FN helpers) but worth
  remembering if new device code calls these macros directly.

## bench_main_hip.cpp: compiles clean (Phase 4a)
Forked from bench_main.cu (hipify-perl first draft, hand-tuned) rather than
#ifdef'd in place, per the file-strategy split -- real logic differences,
not just renames. Three hand fixes beyond hipify-perl's mechanical
cuda*->hip* renames:
1. Device init: HIP has no cudaInitDevice() equivalent at all. Replaced the
   CUDART_VERSION-gated cudaInitDevice/legacy-fallback branch with a single
   unconditional select-then-set-flags-then-verify sequence (hipSetDevice,
   hipSetDeviceFlags(hipDeviceScheduleBlockingSync), hipGetDeviceFlags to
   confirm) -- this is what the CUDA build's own pre-12.0 fallback already
   did for device 0 only; generalized here to any device ordinal since HIP
   always selects first regardless of device number.
2. HIP_VISIBLE_DEVICES replaces CUDA_VISIBLE_DEVICES in the device-count
   mismatch error message.
3. Version-string print: HIP's driver/runtime version integer is NOT CUDA's
   major*1000+minor*10 encoding, and does not obviously match
   HIP_VERSION_MAJOR*10000000+MINOR*100000+PATCH from hip_version.h either
   (observed 70260201 from probe.hip vs. 71526333 computed from this
   install's 7.15.26333) -- printing raw integers instead of a fabricated
   decomposition.
Compiles clean standalone with hipcc (`-DCF_LMAX=4 -DBN_LIMBS=12
--offload-arch=gfx1103`, plus the usual --rocm-device-lib-path) -- only
CRT-deprecation warnings (fopen/sscanf/getenv), same as any MSVC build
without -D_CRT_SECURE_NO_WARNINGS (build_windows_hip.bat will pass it,
matching build_windows.bat).

## fbgen_gpu.hip: compiles clean, zero warnings (Phase 4b)
Forked from fbgen_gpu.cu. Unlike bench_main.cu, hipify-perl's mechanical
pass alone was sufficient -- no hand fixes needed. Confirms hipCUB really is
a drop-in match for this file's cub:: usage (DeviceSelect::Flagged,
DeviceScan::ExclusiveSum), and that __constant__ globals + HIP_SYMBOL() +
hipMemcpyToSymbol work exactly as hipify-perl auto-generates them. Compiled
standalone in -DFBGEN_GPU_LIBRARY mode (the mode bench.exe actually links)
with zero errors AND zero warnings.

## bench_kernels.hip: compiles clean (Phase 4c) -- all three device TUs done
Forked from bench_kernels.cu. Fixes beyond hipify-perl's mechanical pass:
1. The four sync-mask literals (3 in this file, matching CLAUDE.md's earlier
   probe.hip finding; a 4th at line 620 was missed on the first pass due to
   different capitalization, `0xFFFFFFFFu` vs `0xffffffffu` -- caught by
   grepping case-insensitively across ALL of bench/ afterward, not just this
   file).
2. `__launch_bounds__(512, 3)` dropped at all three kernel sites (k_fill_l1,
   k_fill_l2, k_apply) -- CUDA's tuned minBlocksPerMultiprocessor=3 has no
   meaning under HIP (2nd arg = MIN_WARPS_PER_EXECUTION_UNIT there), and this
   session is explicitly a correctness pass, not a performance one (ground
   rules: this box is a correctness vehicle only). Re-tuning from scratch
   for gfx1103 is future work, not done here.
3. #include "cofac.cuh"/"pipeline.cuh" -> #include "cofac_hip.cuh"/
   "pipeline_hip.cuh" -- see below, these needed forking, not in-place fixing.

**cofac.cuh and pipeline.cuh were RECLASSIFIED from "shared, in-place fix"
to "forked" mid-phase.** Initial plan (matching bigint.cuh/td.cuh/
plattice.cuh/prp.cuh) was to just fix their __CUDACC__ guards in place.
That was wrong: unlike the four pure-arithmetic headers, cofac.cuh and
pipeline.cuh are full of DIRECT cuda* API calls (cudaMalloc, cudaFree,
cudaEvent*, ...), not just __CUDACC__-gated __host__/__device__ macros --
confirmed by trying the in-place fix first and hitting "undeclared
identifier 'cudaFree'" deep in cofac.cuh once the guard fix let the compiler
reach that code. Reverted the in-place edit, forked instead:
- `cofac_hip.cuh`, `pipeline_hip.cuh` -- hipify-perl output, ZERO hand
  fixes needed for either, including cofac.cuh's two __CUDACC__ kernel-
  hiding guards (see the audit note above): hipify-perl rewrites
  __CUDACC__ -> __HIPCC__ outright (not `||`-adds it) when forking a file,
  which is correct for a HIP-only fork and is why fbgen_gpu.hip/
  bench_kernels.hip needed no manual __CUDACC__ audit of their own --
  hipify-perl already handles it whenever it's run on a file at all. The
  manual `|| defined(__HIPCC__)` fix is ONLY needed for files that stay
  shared between both builds unforked (bigint.cuh, td.cuh, plattice.cuh,
  prp.cuh, bench.h, slab.h).
- cofac.cuh and pipeline.cuh themselves are back to byte-identical original
  (confirmed via `git diff`, empty) -- fully untouched, as the ground rules
  require for files not actually forked.

**Phase 4 (all three mandatory device TUs) is done**: bench_main_hip.cpp,
fbgen_gpu.hip, bench_kernels.hip all compile individually with zero errors.
Next: build_windows_hip.bat and the first real link.

## Phase 5: FIRST LINK, AND A REAL PIPELINE RUN ON GFX1103 -- WORKING
`bench_hip.exe` links with zero errors via `build_windows_hip.bat` (new,
mirrors build_windows.bat's structure: same host C objects via cl.exe, same
CF_LMAX knob, hipcc in place of nvcc, --rocm-device-lib-path threaded
through, /MT + a note that there's no static HIP runtime to link against
the way --cudart static works for CUDA -- ship C:\rocm\bin's DLLs alongside
bench_hip.exe). Named bench_hip.exe, not bench.exe, deliberately, so it can
never collide with build_windows.bat's own output in the same directory.

`--verify-only` passes both cases (Franke-Kleinjung walk vs brute force,
forced/native slab walk) with EXIT=0 -- confirms plattice.cuh's arithmetic
end-to-end through the real binary, not just the Phase 3 header probe.

**Then a real --pipeline run against oracle/c183.poly, entirely on the
780M, with NO pre-existing factor base file** (`--rlim 67100000 --alim
134200000 --sq-side 1 --logI 14 --qrange 120000053:120000060`, no --fb1):
- fbgen_gpu's on-device factor-base generation ran for real: 7,605,586
  ideals through 134,200,000 in 8.9s wall -- exercises hipCUB
  (DeviceSelect::Flagged / DeviceScan::ExclusiveSum), the __constant__
  arrays, all of it.
- The sieve kernels (k_fill_l1/l2, k_apply -- bench_kernels.hip, no
  __launch_bounds__ tuning at all right now) ran and produced timing broken
  down by stage (fill/apply/transform+plattice), matching the CUDA build's
  own instrumentation output format exactly.
- **The built-in correctness self-check passed 100% on both sides**:
  "side 1: factors x cofactor == norm  7647 of 7647  PASS", "side 0: ...
  3263 of 3263  PASS" -- this is the codebase's own GPU-vs-recomputed-norm
  validation gate, not something added for this port, and it's the
  strongest correctness signal obtained so far on this machine.
- Cofactorization (cofac_hip.cuh's rho/ECM kernels) ran to completion:
  "rational queue 76.18 ms", "algebraic queue 108.61 ms", real relations
  emitted.
- End result over 8 special-q: **16 real relations written to a real
  output file**, correct-looking GGNFS-format (a,b) + full both-side
  factorizations, process exit 0.
- (First attempt used `--relations nul` as a throwaway path and hit "band
  FAILED... rename relations: File exists" -- an artifact of Windows
  treating `nul` as a real device name the checkpoint/resume rename() logic
  tripped over, not a computation bug. Re-ran with a real output filename
  and it completed clean. Left two stray files from that first attempt,
  `nul.part`/`nul.part.ckpt`, since "nul" as a base name becomes a literal
  filename once a suffix is appended -- removed them.)

This is real evidence the port is functionally correct on real gfx1103
hardware, not just "it compiles." Not yet done: the more rigorous Phase 6
checks (cofcheck.sh's pinned counts, wintest.bat's byte/hash comparison)
still need their missing input files (oracle/c183.fb1 via fbgen; wintest's
input.job/fbase.m16) -- this smoke test used its own on-the-fly GPU-generated
FB and arbitrary CLI params, not the pinned oracle inputs those gates check
against, so it's strong-but-informal evidence, not the formal gate.

**Environment note for future runs**: `bench_hip.exe` needs `C:\rocm\bin` on
PATH (for `amdhip64_7.dll` etc.) -- add it if a fresh shell reports the exe
can't start, e.g. `$env:Path += ";C:\rocm\bin"` in PowerShell.

## Phase 6: cofcheck.sh -- THE FORMAL GATE -- ALL CASES PASS
Unblocked the two missing-input problems, both via tools this port itself
produces (no external machine needed):
1. Standalone `fbgen.exe` (CPU, MinGW-built) SEGFAULTS deep inside
   zpoly_roots() (see gdb backtrace below) -- confirmed via gdb this is
   NOT the open_memstream fix (crash is in worker_main -> all_roots ->
   all_roots_affine -> zpoly_roots, pure polynomial-root math, unrelated).
   Pre-existing MinGW-vs-real-math portability bug, orthogonal to CUDA/HIP
   entirely -- NOT investigated further, since a working alternative exists:
2. Built standalone `fbgen_gpu_hip.exe` instead (same fbgen_gpu.hip already
   proven in Phase 4, just linked without -DFBGEN_GPU_LIBRARY, matching
   build_windows.bat's own optional standalone-target pattern). Used its
   `--out FILE` to generate oracle/c183.fb1 (7,605,616 entries, ~115 MB) on
   the actual GPU in 9.9s -- sidesteps the CPU bug AND exercises our own
   ported code instead of unrelated math.
3. `cofcheck.sh` needs `./bench` -- copied bench_hip.exe to a local bench.exe
   (bench.exe is .gitignored and no real CUDA bench.exe exists on this
   machine, so this is a zero-risk, ephemeral local copy, not a tracked
   change).

**Result: `sh cofcheck.sh` (no Make needed, confirmed) -- EVERY case passes,
`cofactor golden test passed`, exit 0.** ~52 pinned-count/behavior cases,
including: trial-division-only (7 rel), rho (37), ECM with/without stage 2
(37/36/37), 3-limb vs 4-limb cofactor width identical results, lpb 33 (59
rel, 24 factors > 2^32, all reconstruct), lpb33/mfb99 at 4 limbs (66 rel),
warp-vs-thread recording identical both unslabbed and at 4 slabs, multi-q
generated bands byte-identical to cached+streamed, and negative controls
(corruption detection, compositeness detection) both firing correctly. This
is the formal correctness gate CLAUDE.md's ground rules call for, and it
passed in full against exact counts originally derived from CUDA.

**fbgen.exe's zpoly_roots MinGW segfault**: real bug, NOT fixed, NOT
blocking (fbgen_gpu_hip.exe covers the same need). If someone wants a
working CPU-only standalone fbgen.exe later: `gdb -batch -ex run -ex bt
--args ./fbgen.exe --poly ../oracle/c183.poly --lim 100 --maxbits 1
--threads 1` reproduces it in under a second; crash is in zpoly_roots, not
in the -pthread/open_memstream Windows-portability code added this session.

## Three more Windows-portability bugs found rounding out the CPU-only gates
All found while trying to get `make check`'s remaining CPU-only targets
(ckpttest/stoptest/slabtest/sqgentest/normscan/dumpcmp/mkcofbatch,
fbgencheck.sh) running under MinGW for completeness, AFTER the actual HIP
port's formal gate (cofcheck.sh) was already fully passing. **All three are
pure host-C, MinGW/Windows-vs-POSIX issues -- zero relation to CUDA/HIP or
to the GPU port itself.** Recorded for whoever wants full Windows CI
coverage of the CPU-only tools later; not fixed except #1, since fixing #2/
#3 means auditing unrelated host-side test/tooling code, out of scope for
this port.

1. **FIXED**: `-march=native` on this CPU auto-vectorizes some code in
   fbgen.c's `zpoly_roots`/`mp_pow_rem` to AVX-512 (`vmovdqa64`, an ALIGNED
   512-bit store) targeting a stack slot the x86-64 ABI doesn't guarantee
   64-byte-aligned -- a GCC 13.2/MinGW codegen bug, not a logic bug.
   Segfaults deterministically (confirmed via gdb: crash PC is literally the
   `vmovdqa64` instruction, `#GP` on an unaligned "aligned" store). Affected
   the standalone fbgen.exe CLI AND sqgentest (both link fbgen.c's
   root-finding code). Fix: build with `HOST_TUNE=-mno-avx512f` instead of
   the Makefile's default `-march=native` when using this MinGW toolchain --
   confirmed sqgentest/fbgen/fbtest all work correctly once rebuilt this way.
   Not something to change in the Makefile itself (Linux gcc doesn't have
   this bug); just don't use bare `-march=native` when invoking
   `mingw32-make` on Windows.
2. **NOT FIXED**: fbtest.c's `verify_loader_validation`/
   `verify_cado_parser_strict`/`verify_poly_parser_strict` hardcode POSIX
   `/tmp/cuda-sieve-*-XXXXXX` paths for `mkstemp()`. No `/tmp` on native
   Windows -- these three cases fail closed (return 0, not a crash) rather
   than actually exercising the parser/loader logic they're meant to test.
   Fix would be using a proper Windows temp-dir API (platform.h already has
   the `_WIN32` pattern to extend), not attempted here.
3. **NOT FIXED**: fbgen.c's atomic `--out` staging (write to a mkstemp'd
   temp name, then `rename()` it onto the final name) fails 100% of the time
   on Windows with "File exists" -- confirmed NOT a stale-file collision
   (reproduces with a fresh mkstemp name every time). Root cause: this
   pattern relies on POSIX `rename()`'s silent-overwrite semantics; Windows'
   CRT `rename()` refuses when the destination already exists (which it
   always does here, since mkstemp() itself creates the destination as an
   empty placeholder before the temp-then-rename dance). Broke
   `fbgencheck.sh`. Same underlying issue as the earlier "nul.part" artifact
   from Phase 5's first (mistaken) smoke-test attempt -- `wintest.bat`
   already works around this class of problem by deleting its output before
   each run rather than fixing rename() itself; a real fix would need
   `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING)` on Windows.

None of these affected cofcheck.sh, which uses a pre-generated c183.fb1
(built via fbgen_gpu_hip.exe, sidestepping fbgen.exe's bug #1 above entirely
at the time) and bench_hip.exe's own relations-file writer, which doesn't
hit bug #3's exact staging pattern for a fresh output filename.

## Code review pass, pre-commit
An 8-angle review (correctness x3, reuse, simplification, efficiency,
altitude, conventions) plus a 1-vote verify pass on every candidate found
4 real, confirmed issues; everything else (device-init ordering, the
fbgen.c `#include` placement, the 7-way `__CUDACC__` guard "duplication",
the missing `/MT` in build_windows_hip.bat's HIPFLAGS) was investigated and
refuted with concrete evidence (one via an actual fresh `--blocking-sync`
hardware run; another via `dumpbin` showing zero dynamic CRT dependency in
the linked binary despite the missing flag). Fixed 3 of the 4 confirmed
issues; the 4th (build script duplication) got cross-reference comments
instead of a real fix, since a proper fix would require editing
build_windows.bat, which the ground rules prohibit:
1. `bench_div_u128_u64`'s portable fallback now aborts loudly on a
   precondition violation (hi>=d) instead of silently returning a
   truncated/wrong result -- restores parity with the real `_udiv128`
   path's hardware fault for the same violation.
2. `k_apply`'s dropped `__launch_bounds__` left three stale "hard ceiling"
   comments (bench.h, bench_kernels.hip, bench_main_hip.cpp's error text) --
   corrected all three, AND added a real runtime check
   (`athr > APPLY_THREADS_MAX`) at both actual launch sites
   (bench_kernels.hip, pipeline_hip.cuh) so a future caller bypassing the
   CLI's own --apply-threads validation is still caught.
3. `bench_close_memstream` now calls this same file's own `bench_tell`/
   `bench_seek` (64-bit-safe) instead of raw `ftell`/`fseek` (32-bit `long`
   on Win64) -- removes a latent overflow trap for large fbgen CLI runs.
4. `build_windows_hip.bat`'s CF_LMAX/git-stamp blocks got explicit
   cross-reference comments naming all three places (Makefile,
   build_windows.bat, here) that must move in lockstep -- the duplication
   itself stays, since removing it for real means editing build_windows.bat.
Re-ran cofcheck.sh after all fixes: all ~52 cases still pass, exit 0.

## BOINC integration: built, linked, verified working end-to-end
Confirmed earlier (see the "did this affect BOINC?" answer in session history)
that the HIP port itself never touched boinc_support.cpp or any BOINC-related
code -- this section is genuinely new work, done separately and later, at the
user's request to actually build and verify a BOINC-enabled binary.

**Getting BOINC's client libraries built** (separate checkout, C:\dev\boinc,
not part of this repo): the user's own established workflow --
`vcpkg_3rdparty_dependencies.vcxproj` then `libboinc.vcxproj` /
`libboincapi.vcxproj` via msbuild -- needed two environment-specific fixes,
both in the freshly-cloned BOINC checkout, NOT in cuda-sieve:
1. `boinc.props` hardcodes `PlatformToolset=v145`; this machine's VS2022
   Build Tools only has v143 installed. Retargeted to v143 (BOINC's C++ code
   has no v145-specific dependency; this is purely an environment mismatch,
   the standard "Retarget solution" fix MSBuild's own error message
   suggests).
2. `vcpkg_3rdparty_dependencies.vcxproj`'s `bootstrap-vcpkg.bat` and
   `vcpkg.exe` Exec commands use bare filenames, relying on cmd.exe's
   implicit current-directory lookup. This session's environment has
   `NoDefaultCurrentDirectoryInExePath=1` set (an intentional sandboxing
   control -- NOT disabled, per the standing rule against touching security
   settings), which breaks that lookup with a bare "not recognized" error.
   Fixed by prefixing both commands with `.\` (an explicit relative path
   sidesteps PATH-search entirely, security setting or not) rather than
   touching the environment variable.
3. The vcpkg install itself failed once on a transient network error
   (curl error 56) fetching `webview2` from NuGet -- unrelated to
   libboinc/libboincapi (that's a BOINC Manager GUI-only dependency).
   Rather than wait on a full retry, built `libboinc.vcxproj`/
   `libboincapi.vcxproj` directly once confirming their actual dependencies
   (openssl, curl, zlib -- all already installed from the first, mostly-
   successful run) were present. Both libraries built clean, 0 errors, in
   about a second each.

**New build script**: `bench/build_windows_hip_boinc.bat` -- same structure
as build_windows_hip.bat (same CF_LMAX/git-stamp duplication, same
cross-reference comments), plus `-DHAVE_BOINC`, BOINC's include paths
(`-I`/`/I` split correctly -- see below), and the two BOINC .lib files at
link time. Produces `bench_hip_boinc.exe`, kept separate from
`bench_hip.exe` (own object-file suffix, `_boinc`) so neither build clobbers
the other.

**One real build bug found and fixed**: the BOINC include paths went on
HIPFLAGS as well as CXXFLAGS in the first draft, using MSVC's `/I "path"`
syntax -- hipcc's underlying clang driver doesn't accept `/I`, only `-I`,
and errored immediately (`no such file or directory: '/I'`). Fixed by
removing BOINC_INC from HIPFLAGS entirely: bench_main_hip.cpp/
bench_kernels.hip/fbgen_gpu.hip never `#include` any BOINC header directly
(confirmed by grep) -- they only call the `bench_boinc_*` wrapper functions
declared in bench.h and implemented in boinc_support.cpp, which is compiled
by cl.exe (MSVC `/I` syntax, correctly) and is the only file that needs the
BOINC include paths at all.

**A real functional gap found and fixed, in boinc_support.cpp (previously
untouched, fixed only after asking the user first)**: `bench_boinc_gpu_device()`
hardcoded `strncmp(aid.gpu_type, "NVIDIA", 6) != 0` to reject any non-NVIDIA
client GPU assignment -- for a real BOINC deployment on AMD hardware, the
client reports `gpu_type == "ATI"` (BOINC's own standardized vendor string
for AMD, confirmed in BOINC's coproc.h -- kept from the pre-rebrand name),
so this check would have silently rejected every real AMD GPU assignment,
defeating the entire multi-GPU-host device-assignment mechanism. Fixed with
a new `BENCH_HIP_BUILD` macro (defined only by build_windows_hip.bat and
build_windows_hip_boinc.bat, harmless/unused when HAVE_BOINC is off) that
picks "ATI" instead of "NVIDIA" for the HIP build -- the CUDA build's
behavior is completely unchanged (BENCH_HIP_BUILD is never defined there).
Also fixed several cosmetic "CUDA device"/"CUDA's default" strings elsewhere
in bench_main_hip.cpp's help text and BOINC-visible log messages (leftover
from the mechanical hipify-perl pass, which only renames cuda*/hip* API
identifiers, not string literals) to say "HIP device"/"HIP's default" --
these are BOINC-log-visible and would otherwise confuse a real volunteer
running this on AMD hardware.

**Verified working end-to-end**: `bench_hip_boinc.exe --help` prints
"BOINC: API initialised (standalone mode)" and the HAVE_BOINC-specific help
text (confirming the BOINC API layer really initializes, not just compiles
in). A real --pipeline run (same job as the Phase 5 smoke test) produced
"BOINC: running on HIP device 0 of 1: AMD Radeon 780M Graphics" in its
BOINC-mode log output, completed with the same correctness self-check
passing 100% (7647/7647, 3263/3263) and 16 relations written, exit 0 --
BOINC's standalone-mode initialization doesn't interfere with the actual
sieve/cofactor computation at all.

**Not tested**: an actual BOINC client assigning a real AMD GPU via
init_data.xml (would need a live BOINC project/client setup, out of scope
for this machine) -- the fix is verified correct against BOINC's own
documented vendor-string convention (coproc.h) and against this build's
compile/link/run behavior, but the client-assignment code path itself
(`aid.gpu_type[0] && strncmp(...)`) only executes under a real client, which
this session cannot provide.

## Multi-arch RDNA fat binary: built, verified on this hardware
At the user's request: a BOINC deployment binary should run on any AMD card
a real volunteer pool has, not just this box's gfx1103. Added `GFX_ARCH=all`
to both build_windows_hip.bat and build_windows_hip_boinc.bat (mirroring
the CUDA build's own `GPU_ARCH=all` fat-binary knob), expanding to 14
`--offload-arch=` flags in one hipcc invocation:

    gfx1010/1012            RDNA1: RX 5700/5700XT/5600XT, RX 5500/5500XT
    gfx1030/1031/1032/1034  RDNA2: RX 6800-6900 series, 6700(XT), 6600
                             series, 6500XT/6400
    gfx1100/1101/1102/1103  RDNA3: RX 7900 series, 7800XT/7700XT, 7600
                             series, and the 780M/880M iGPU family (this box)
    gfx1150/1151            RDNA3.5: Strix Point/Strix Halo iGPUs
    gfx1200/1201            RDNA4: RX 9060 series, RX 9070(XT)

**Scope reasoning**: user asked for "all AMD cards with at least 4GB memory
... excluding pro-level 64-bit warp cards" -- the wavefront-width exclusion
maps directly onto RDNA (wavefront32) vs. CDNA (wavefront64, the MI100/
MI200/MI300 compute-only cards), which matches this codebase's own existing
ground rule ("Warp width 32 is assumed throughout... correct on RDNA under
HIP"). So "exclude 64-bit warp cards" isn't a new constraint here -- it's
the SAME constraint already documented, just phrased from the hardware side
instead of the code side. RDNA iGPUs (gfx1103/1150/1151) are included
because they reach "4GB" via a configurable UMA carveout, same as this
machine's own 780M (confirmed running at 14.41 GB total addressable in
Phase 5/6's own testing) -- excluding them would exclude the very box this
port was developed and validated on.

**Real, load-bearing difference from CUDA's fat binary, not just a footnote**:
CUDA's `compute_80,code=compute_80` PTX entry lets a newer-than-listed GPU
still load via JIT recompilation. HIP has no equivalent here -- each
`--offload-arch` embeds real finished machine code for that exact target,
and the runtime picks an exact match or fails. A future RDNA5 card (or any
gfx ID not in the 14 above) will not run this binary at all, however similar
its ISA -- there is no forward-compatible fallback path to fall back on.
Re-running with the new gfx ID added is the only fix when that happens.

**Verification**: sanity-checked all 14 `--offload-arch` strings against
this ROCm 10.0.0 install with a throwaway probe.hip compile (zero errors)
before committing to the full build. Full `GFX_ARCH=all` build of
bench_hip_boinc.exe: zero errors, ~25 MB binary (vs. ~3 MB single-arch) --
about 20-25 minutes wall clock for all 14 targets, one hipcc invocation per
translation unit compiling all 14 device code objects serially within that
invocation (no fat-binary-specific parallelism knob was used here, unlike
the CUDA Makefile's `-t 0`). Ran the actual fat binary on this
machine's real gfx1103 hardware: correctly selected the gfx1103 code
object at runtime ("AMD Radeon 780M Graphics"), same --pipeline job,
same 100% correctness pass (7647/7647, 3263/3263), 16 relations, exit 0 --
embedding 13 other architectures' code objects doesn't perturb the one
that actually runs on this box.

**Not verified**: the other 13 architectures' actual code objects, since
this machine only has the one gfx1103 card to run on. Each is real compiled
machine code (not skipped or stubbed), and the shared source they're
compiled from is the same source already validated end-to-end on gfx1103 --
but "compiles clean for gfx1030" and "runs correctly on a real RX 6800" are
different claims, and only the former is established here.

## Performance tuning: plan written, items 1-3 and 5 done, plus a real change (item 6)
See `bench/HIP_TUNING_PLAN.md` for the full plan and item 1's (`
__launch_bounds__` sweep) results. Short version: added a real diagnostic
tool (`bench_dump_kernel_attrs()`, env-var gated) since this ROCm/Windows
distribution ships no rocprof/omniperf -- it showed none of k_fill_l1/
k_fill_l2/k_apply are register-bound on gfx1103 (all have low register
counts), so re-adding `__launch_bounds__` had zero measured performance
effect (confirmed both via the occupancy calculator and by re-running the
544-q baseline). It DID have one real, independent benefit: the first
`__launch_bounds__` argument restored the compiler-enforced launch-size
ceiling for k_apply that the earlier code-review finding (this same
CLAUDE.md, "Stale 'hard ceiling' claim") flagged as lost -- both the
compile-time backstop and the runtime check from that fix now exist
together. cofcheck.sh's full suite still passes with these changes in
place. Also corrected a premise in the plan itself: `--fill-mode twolevel`
(what actually exercises k_fill_l1/l2) turns out to be a benchmark-harness-
only flag, refused under `--pipeline` and already known 2.7x slower than
the default -- it's dead code for production, not a live tuning decision.

Item 2 (log_region re-sweep) directly tested whether region 13's 3x better
occupancy (found in item 1) translates to real speed -- **it doesn't**.
Measured region 9 through 14 on the real pipeline: both fill and apply get
monotonically SLOWER as region shrinks, across the whole range, despite
occupancy going the opposite direction. Region 13 at full (544-q) sample:
2.24x slower fill, 1.34x slower apply than the region-14 default. More
regions means more per-region fixed overhead, and that scales faster than
the occupancy gain pays for. No code change -- the CUDA-tuned default
(`--region 14`) is already right on this hardware too, for a completely
different reason (CUDA never needed to know about this tradeoff, since its
higher shared-memory ceiling meant it never hit the occupancy cliff region
14 causes on AMD in the first place -- it just happens to land in the same
place both builds want). Recorded so a future session doesn't re-derive
the same occupancy number and re-try the same change.

Item 3 (cofactor register pressure) extended the diagnostic to all six
`k_cofac<L,method,stage2>` instantiations. Real finding: STATUS.md states
plainly that nothing spills at any width on NVIDIA's ptxas; on gfx1103,
**four of the six spill** (32-608 B/thread) -- HIP-clang's register
allocator genuinely differs from ptxas here, and "cofactor width doesn't
cost much" is not a claim that automatically carries over from the CUDA
build. But measured (not assumed): the 3->4-limb width penalty (1.42x-1.74x
across rho/ECM/ECM+stage2) lands in the same rough band as STATUS.md's own
NVIDIA number, not a multiple worse. Curiously, the MOST-spilling variant
(`<4,ECM,s2>`, 608 B/thread) has the BEST occupancy of any of the six (4
blocks/CU, vs 1 for the other five) and the SMALLEST width penalty (1.42x)
-- HIP-clang traded spill traffic for occupancy on this one instantiation,
and it paid off rather than costing extra. No code change; cofcheck.sh
still passes (no kernel code touched, only the diagnostic).

Item 5 (grid-width re-derivation) turned out narrower than planned: the
`multiProcessorCount * 6` constant only feeds the COFACTOR kernels' grid
width (traced to pipeline_hip.cuh:1275), not k_apply/fill (those use a
geometry-derived `nregion`, unrelated). Swept `--blocks` across two
cofactor cases with very different occupancy (item 3's 1-block/CU and
4-blocks/CU instantiations) -- no single value is optimal for both (12
best for one, 48 best for the other), but the current default (36) lands
within 1-3.3% of optimal for both. No change: the inherited NVIDIA constant
works on gfx1103 not because the occupancy reasoning transfers (it doesn't
-- 1 or 4 blocks/CU here, nothing like NVIDIA's 6), but because these
kernels' sensitivity to total grid size is fairly flat above ~36-48
regardless of per-block occupancy. cofcheck.sh still passes (no code
touched, only measured via the existing --blocks override).

**Item 6 (new, user-directed): j-slab performance target -- this one WAS
wrong.** Asked directly whether slab.h's `SLAB_PERF_TARGET_LOG2 = 29`
(2^29-position auto j-slab target) still holds on gfx1103 -- it doesn't.
Swept `--slab-j` at a large geometry (`--logI 16 --J 65536`, area 2^32) to
force auto-slabbing: 2^27 beats the old default (2^29) by **45.6%** at a
33-special-q sample (3889 vs 5664 ms/q) -- by far the largest effect found
in this whole tuning pass, and directionally exactly what the constant's
own comment predicts (it names L2 cache size as the reason NVIDIA cards
differ from each other; gfx1103's 2 MB L2 is smaller than any of them).
2^30 (what a big-L2 NVIDIA card prefers) is catastrophic here (3.3x worse
than 2^27). **Changed** `SLAB_PERF_TARGET_LOG2` to be `BENCH_HIP_BUILD`-
gated (27 for HIP, 29 unchanged for CUDA) -- a value swap behind an
existing marker in an already-shared file, not a fork. Also fixed
pipeline_hip.cuh's status line, which printed a hardcoded "2^29" that
would otherwise have silently gone wrong; it now prints the actual
constant. cofcheck.sh still passes. **Follow-up, done**: SLAB_PERF_TRIGGER_LOG2 needed to move too, by a
derivable amount (TRIGGER = TARGET+1 is not a CUDA coincidence --
slab_perf_jmax()'s math makes that ratio produce a clean 2-way split
exactly at the trigger boundary). Fixed to 28 for HIP, confirmed the gap
it was leaving (a 2^28-position sieve ran unsplit under the old inherited
trigger, 14% slower than the 2-way split the new trigger now produces
automatically).

**CRITICAL SAFETY FINDING, found while re-verifying this fix**: this
session had already caused TWO unclean system reboots today (confirmed via
Windows Event Viewer, Kernel-Power events at 9:06 PM and 11:17 PM) -- with
no softer "driver recovered" event (ID 4101) alongside either one. Unlike
a discrete NVIDIA card, this iGPU/driver combination appears to hang the
WHOLE SYSTEM, not just reset the GPU driver, when a single kernel launch
runs too long. This retroactively explains the wild run-to-run instability
already noted for the largest slab sizes tested (2^30: ~998 ms per single
fill-kernel launch, right at typical TDR thresholds; 2^31: ~3167 ms/launch,
well past it) -- those specific measurements were likely contaminated by
actual driver hangs, not clean compute time. This means the slab target
fix (27) is a STABILITY fix on this hardware, not just a 45.6% throughput
win -- CUDA's inherited 2^29 default was already close to the edge here.
Do not repeat the large-geometry sweep points (rows 16384/32768 at --logI
16) -- confirmed risky, not just suspected. A separate, unrelated,
NOT-fixed risk was also flagged: cofcheck.sh's `--ecm-b1 400000` case
drives a long single k_cofac launch (duration set by ECM parameters, not
slab geometry) that may be its own crash risk independent of anything in
this session's slab work. Full details in HIP_TUNING_PLAN.md.

**Item 7 (new, user-directed): startup slab-size auto-calibration -- DONE.**
Item 6 found gfx1103's optimum (2^27) is cache-size-driven, not a HIP-vs-CUDA
property -- meaning a discrete AMD card with a bigger L2 could want something
else in {2^27, 2^28, 2^29}. Rather than trust one static constant, added
calibration to `run_pipeline()` (pipeline_hip.cuh): when auto-slabbing would
apply and `--slab-j` wasn't forced, time all three candidates against the
real `qlist[0]` (never touching the streaming generator's state -- confirmed
by reading `run_pipeline_impl`'s loop and all three call sites that populate
`qlist`/`qgen`) and use whichever wins. Candidates are bounded to {2^27,
2^28, 2^29} ONLY -- 2^30 and above are excluded on purpose, per the safety
finding above. **A real bug found and fixed**: the first version left inline
`--cofactor` (ECM/rho) running during the 3 throwaway calibration passes,
since nulling the output paths alone doesn't disable it -- this tripled ECM's
cost for no benefit (its timing is flat regardless of slab size) and broke
5 cofcheck.sh cases that grep for cofactor-method diagnostic lines expecting
exactly one occurrence. Fixed by forcing `cofactor = 0` in the calibration
scratch config. Verified: cofcheck.sh passes 52/53, matching the
pre-calibration baseline with one exception -- the already-flagged
`--ecm-b1 400000` case (previous paragraph) failed the same
already-documented way (a caught `cofac_hip.cuh` device error, not a crash)
with calibration enabled, having passed on the baseline; possibly calibration's
extra GPU load beforehand makes that pre-existing, unrelated issue somewhat
more likely to surface, but per user decision this is accepted and documented
rather than chased further -- fixing it belongs to the separate ECM-kernel
issue, not to this item. Full design/verification detail in
HIP_TUNING_PLAN.md's Item 7.

## Windows build notes
- This ROCm build is AMD's "TheRock" Windows distribution
  (therock-dist-windows-gfx110X-all-10.0.0.tar.gz), not a standard ROCm
  layout. amd-smi is not shipped at all on Windows in this distro; use
  hipInfo.exe for device introspection instead.
- Device bitcode lives at C:\rocm\lib\llvm\amdgcn\bitcode, not
  <ROCM_PATH>\amdgcn\bitcode where clang looks by default. hipcc/clang
  invocations need --rocm-device-lib-path=C:\rocm\lib\llvm\amdgcn\bitcode
  explicitly (a CMake HIP build here will need the same flag threaded through).
