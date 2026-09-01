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

## Windows build notes
- This ROCm build is AMD's "TheRock" Windows distribution
  (therock-dist-windows-gfx110X-all-10.0.0.tar.gz), not a standard ROCm
  layout. amd-smi is not shipped at all on Windows in this distro; use
  hipInfo.exe for device introspection instead.
- Device bitcode lives at C:\rocm\lib\llvm\amdgcn\bitcode, not
  <ROCM_PATH>\amdgcn\bitcode where clang looks by default. hipcc/clang
  invocations need --rocm-device-lib-path=C:\rocm\lib\llvm\amdgcn\bitcode
  explicitly (a CMake HIP build here will need the same flag threaded through).
