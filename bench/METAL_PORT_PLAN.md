# Metal port plan — cuda-sieve on Apple Silicon

Porting the CUDA NFS lattice sieve in `bench/` to Metal Shading Language for
Apple Silicon GPUs. Companion to the HIP port on `hip-port`; that port's
`CLAUDE.md` is the model for how findings get recorded here.

Branched from `main` at `3e15fec`. Note that `hip-port` is 23 commits behind
`main`, so it is a **reference for method, not a base to build on**.

---

## 1. Scope and ground rules

- The CUDA build stays authoritative. Any CUDA-side edit gets a row in the
  drift ledger (section 9) saying what changed and how far it was verified.
  The HIP port's experience is that an unrecorded CUDA-side edit is the
  failure this rule exists to prevent.
- **Performance tuning on this box is allowed** (changed 2026-09-14). A full
  M3 is a real Apple GPU, so measurements here are real. State the machine
  with the number — a fanless 10-core M3 that also drives the display is the
  small end of the range and throttles under sustained load — and say which
  box produced any value that becomes a shipped default.
- Acceptance gate: `cofcheck.sh` green, plus relation comparison against the
  CUDA build on the oracle jobs. Whether that comparison can be *byte*
  identical is an open question — see section 5.1.
- Warp width 32 is assumed throughout the source (`>> 5`, `& 31`, lane 31
  broadcasts). Apple GPUs execute SIMD-groups of 32, so this holds; it is a
  Phase 0 probe item, not an assumption to "fix".

---

## 2. This machine

| | |
|---|---|
| Chip | Apple M3, 10-core GPU, Apple family 9 |
| Memory | 16 GB unified (UMA), shared with display and OS |
| Metal | Metal 4 support, macOS 26.5 |
| Xcode | 26.5 (17F42) at `/Applications/Xcode.app` |
| `xcode-select -p` | `/Library/Developer/CommandLineTools` — **not** Xcode |

`xcode-select` points at Command Line Tools, which ship no `metal` compiler.
Rather than `sudo xcode-select -s`, export `DEVELOPER_DIR` in the build:

```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

The Metal toolchain is a separately downloaded Xcode component:
`xcodebuild -downloadComponent MetalToolchain`.

---

## 3. Why Metal is a bigger jump than HIP

The HIP port was largely mechanical: `hipify-perl` produced near-1:1 forks,
and three of the five ported files needed zero hand fixes. Four structural
differences mean Metal cannot work that way.

### 3.1 Host and device code must physically separate

`bench_kernels.cu` is one translation unit that includes `td.cuh`,
`cofac.cuh` and `pipeline.cuh` (`bench_kernels.cu:14-26`, `:3268-3269`) —
44 `__global__` kernels interleaved with host orchestration, of which
`pipeline.cuh` alone is ~208 KB with **zero** kernels in it. MSL kernels must
live in `.metal` files compiled to a `.metallib`, and every launch becomes
explicit buffer binding on an `MTLComputeCommandEncoder`.

HIP preserved the TU shape. Metal forces a split of every mixed file.

### 3.2 Apple GPUs have no `double`

MSL has no `double` type at all. fp64 appears in **device** code in exactly
two places, both load-bearing:

- `bench_kernels.cu:525-534` — the fp64 recompute in `k_apply` when the fp32
  Horner cancels. `bench.h:329-341` documents the consequence of losing it:
  144 of 63,497 positions in a band along the three real root lines round to
  the wrong sieve-log value, error -3.31 to +2.57 units and **both signs** —
  false survivors *and* lost relations. It fires on well under 1 cell in 1000
  (`bench.h:344-355`), so the path is cold but not optional.
- `td.cuh:981` → `cof_classify` (`prp.cuh:194-210`), whose CADO gap test is
  deliberately written against doubles because matching CADO matters more
  than being more careful than it (`prp.cuh:191-193`). `k_classify` also takes
  `double lim` as a kernel parameter (`td.cuh:963`).

`norm_t` additionally carries `double dd[BENCH_NCOEFF]` and `double A, B`
(`bench.h:338-341`) and is passed **by value** into kernels.

### 3.3 `log2f` is the larger byte-identity risk

`bench_kernels.cu:543-568` already documents that the accurate `log2f`, not
the `__log2f` intrinsic, is required for CPU parity — there is even a
`#warning` guarding the fast path (`bench_kernels.cu:550`). CUDA and glibc
agree here because both can compute a float `log2` with fp64 internally.
MSL cannot.

**Measured on this M3 (Phase 0, section 5.3), not assumed:** over 2^20 float
samples spanning the exponent range, `metal::log2` differs from the host's
`log2f` on **50.03% of inputs, by up to 3 ULP**. `metal::precise::log2`
returns bit-identical results to plain `log2` — the `precise::` namespace
buys nothing here — and this was measured with `-fno-fast-math` already set.

So MSL's built-in `log2` **cannot** reproduce CUDA's sieve bytes. The value
feeds `floor(log2|F| * scale + 0.5)`, so ULP-level differences flip cells by
±1 sieve unit, and cells straddling the survivor threshold change the
survivor set. What remains open is not *whether* the logarithm differs but
whether that difference reaches the relation set — see Phase 7.

### 3.4 No CUB

`fbgen_gpu.cu` uses `cub::DeviceSelect::Flagged` and
`cub::DeviceScan::ExclusiveSum` at 8 sites (`:1660`, `:1662`, `:1693`,
`:1721`, `:2369`, `:2371`, `:2449`, `:2527`). hipCUB was a drop-in for HIP;
Metal has no equivalent.

Mitigating: `td.cuh` already contains a hand-rolled 3-pass exclusive scan
(`k_scan_pass1/2/3`, `td.cuh:365-420`) and the tree already does flagged
compaction by scan-then-scatter (`k_cof_selflags`/`k_cof_selscatter`,
`k_intersect_compact`). The replacement has in-tree precedent and is chosen
deliberately over atomics for reproducibility (`td.cuh:455-465`).

---

## 4. What ports easily

The CUDA API surface is small and old-fashioned. Distinct call sites across
all of `bench/`:

- **No** cooperative groups, CUDA graphs, textures, dynamic parallelism,
  managed/unified memory, or device-side `malloc`/`new`.
- 3 `cudaStreamCreate`, 1 `cudaStreamWaitEvent`. Events are almost entirely
  timing (`cudaEventElapsedTime` x7).
- Intrinsics map 1:1: `__syncthreads` (27) → `threadgroup_barrier`,
  `__syncwarp` (8) → `simdgroup_barrier`, `__ballot_sync` (5) →
  `simd_ballot`, `__shfl_up_sync` (1) → `simd_shuffle_up` (or MSL's native
  `simd_prefix_exclusive_sum`), `__popc` (8) → `popcount`, `__ffs` (8) →
  `ctz`+1, `__clz` (4) → `clz`, `__umulhi` (8) → `mulhi`.
- Only **two** 64-bit atomics, both diagnostic counters
  (`bench_kernels.cu:120` `nlost`, `cofac.cuh:1554` an iteration histogram).
  This matters, because Phase 0 established that **MSL has no 64-bit atomics
  at all** on this toolchain (section 5.3) — not add, not even min/max. Both
  sites must be reworked: accumulate per-threadgroup, then commit through a
  32-bit pair (low word with carry into high) or a `uint2` CAS loop. Neither
  is on a hot path, so the cost is irrelevant.
- Only **three** `__constant__` symbols (`fbgen_gpu.cu:110-112`).
- Widest kernel is 14 pointers (`k_td`), then 13 (`k_apply`,
  `k_td_record_warp`) — all under Metal's 31-buffer binding limit.
- UMA means ~133 `cudaMemcpy` and 30 `cudaHostAlloc` calls collapse into
  `MTLStorageModeShared` buffers with no copy at all.
- No spin-waiting or decoupled-lookback anywhere, so Metal's lack of a
  forward-progress guarantee between threadgroups is not a hazard.

### 4.1 The one hard binding case

`cofq_t Q` is passed **by value** into `k_cof_enqueue` (`cofac.cuh:1410`) and
carries roughly 27 device pointers (`cofac.cuh:1320ff`). Combined with that
kernel's 10 other pointer arguments this exceeds the 31-buffer limit, so it
needs a Metal **argument buffer** (M3 confirmed Tier 2, section 5.3),
not plain bindings. This is
the only kernel in the tree that does.

---

## 5. Constraints this hardware imposes

### 5.1 Threadgroup memory: 32 KB, so `log_region <= 13`

| | CUDA | HIP / gfx1103 | Metal / M3 |
|---|---|---|---|
| Per-block shared memory | ~100 KB opt-in | 64 KB | **32 KB (measured)** |
| Max `log_region` | 15 | 14 | **13** |

`apply_smem = 2^log_region * 2 + nslice_pow2 * sizeof(logp)`
(`pipeline.cuh:285-286`). At the default `log_region = 14`
(`bench_main.cu:971`) the region alone is 32 KB, before slices — it will not
fit. The Metal build must default to 13 and the existing
`cuda_optin_smem_limit` check (`pipeline.cuh:287-292`) must be retargeted at
`MTLDevice.maxThreadgroupMemoryLength`, which already fails closed with a
clear message.

### 5.1a Support floor: M1 (Apple7) on macOS 13

`-std=metal3.0`, `METAL_MIN_MACOS = 13.0`. MSL 3.0 is the toolchain's floor,
not a choice: Xcode 26.5's Metal compiler advertises `metal2.0`-`metal2.4` in
its `-std` help and rejects all of them. No hardware coverage is lost — every
Apple silicon Mac runs macOS 13+ — only an M1 held back on Big Sur or
Monterey. `metal_rt` fails closed below `MTLGPUFamilyApple7` and cross-checks
`threadExecutionWidth == 32`, because Metal also runs on Intel Macs whose AMD
(64-lane) or Intel (8-lane) GPUs would silently compute a different answer
rather than crash.

**The flag only covers the TUs this Makefile compiles, which was not all of
them.** The CPU-side objects (`verify_cpu.o`, `fbgen_lib.o`, `platform.o`,
`watchdog.o`, ...) are built by delegating to the default `Makefile` rather
than duplicating its rules, and its `HOST_TUNE` defaults to `-march=native`.
So those objects were compiled with neither `-mmacosx-version-min` nor a
pinned CPU: `otool -l platform.o` recorded `minos 26.0` inside a binary whose
own load command says `minos 13.0`. The linker says so out loud -- "built for
newer 'macOS' version (26.0) than being linked (13.0)" -- and the warning had
been scrolling past since the first link.

`Makefile.metal` now passes `HOST_TUNE='-mcpu=apple-m1
-mmacosx-version-min=$(METAL_MIN_MACOS)'` to each of the four delegated
builds. Every object in the link is `minos 13.0` and the warnings are gone.
The default `Makefile` folds `HOST_TUNE` into a stamp that every object
depends on, so alternating between a native build and a Metal build rebuilds
them rather than silently reusing the wrong ones -- verified by switching
`HOST_TUNE` back and watching `platform.o` rebuild unprompted.

On *this* machine the ISA half was moot: Apple clang's `-march=native`
resolves to `apple-m1` even on an M3, so no post-M1 instruction was reachable.
That is a fact about this toolchain, not a guarantee, and `-mcpu=apple-m1`
makes it one. All five gates were re-run after the change -- `verify_cpu.o` is
the CPU *reference* the Phase 5 parity gate compares against, so changing its
codegen flags could have moved the thing being compared to, and 4,194,304
cells still match exactly.

### 5.2 Memory and the display watchdog

16 GB unified, shared with the OS and the display the GPU is also driving.
Measured `recommendedMaxWorkingSetSize` is 11.84 GB and `maxBufferLength` is
8.88 GB — the latter is a real ceiling on any single allocation, and the
production geometry's per-slab sieve array must stay under it.

The existing `--cof-chunk` work and `cudaErrorLaunchTimeout` handling
(3 sites) map onto `MTLCommandBufferError.timeout` and are directly relevant
here in a way they were not on a headless card.

### 5.3 Phase 0 probe results — measured 2026-09-14 on this M3

Toolchain: Xcode 26.5 (17F42), Metal Toolchain 17F42, `metal` frontend
32023.883, target `air64-apple-darwin25.5.0`, compiled `-std=metal3.2
-fno-fast-math`. Probe sources in `metal-probe/`.

**Device facts**

| property | value |
|---|---|
| name / family | Apple M3, `MTLGPUFamilyApple9` |
| `hasUnifiedMemory` | yes |
| `maxThreadgroupMemoryLength` | **32768 B** |
| `maxBufferLength` | 8.88 GB |
| `recommendedMaxWorkingSetSize` | 11.84 GB |
| `argumentBuffersSupport` | **Tier 2** (enum value 1) |
| `maxThreadsPerThreadgroup` | 1024 |
| `threadExecutionWidth` | 32 |

**Warp semantics — every CUDA assumption in the source holds**

| check | result |
|---|---|
| `threads_per_simdgroup` | 32 — the `>> 5` / `& 31` idiom is safe |
| `simd_ballot(lane odd)` | `0xaaaaaaaa`, high word 0 — **ascending lane order**, so `td.cuh:647` holds |
| `simd_shuffle_up` | matches CUDA's `__shfl_up_sync` semantics |
| `simd_prefix_exclusive_sum` | lane-exact; a native replacement for the manual shfl scan |
| `mulhi(ulong, ulong)` | exists and is **exact** — 0/4096 mismatches against host `__int128` |

**Language support — compile-tested**

| feature | result |
|---|---|
| `double` | **rejected**: "'double' is not supported in Metal" |
| `mulhi` 32 and 64 bit | OK |
| `popcount` / `clz` / `ctz` on `ulong` | OK |
| templated kernel + `[[host_name]]` explicit instantiation | OK |
| `device` pointers inside a struct (argument buffer) | OK |
| `fma`, `precise::log2` | OK |
| **64-bit atomics — any operation** | **unavailable.** `__HAVE_ATOMIC_ULONG__` and `__HAVE_ATOMIC_ULONG_MIN_MAX__` are never defined on this target, under `-std=metal3.2` or `metal4.0`. `atomic_uint` works normally. |

**log2 accuracy — 2^20 samples against host `log2f`**

| implementation | differs | max error |
|---|---|---|
| `metal::log2` | 524,644 (**50.03%**) | 3 ULP |
| `metal::precise::log2` | 524,644 (**50.03%**) | 3 ULP |

`precise::log2` is bit-identical to plain `log2`; it is not a fix. This is
the measurement that makes `portable_log2.h` load-bearing rather than
speculative — see section 3.3 and Phase 7.

**fp32 divide and fma — 2^20 samples each, against the host**

| operation | differs | of which involve a subnormal or zero | **differs between normals** |
|---|---|---|---|
| `a / b` | 45,238 | 45,238 | **0** |
| `fma(a,b,c)` | 25 | 25 | **0** |

Two things follow, and they point in opposite directions.

The good one: fp32 divide and fma are **correctly rounded and bit-exact
against the host** wherever neither side is subnormal. That is what lets
`portable_log2.h` use `/` and `fma` directly instead of emulating them — its
intermediates are all bounded well inside the normal range.

The one to watch: **Apple GPUs flush subnormals to zero.** Every single
disagreement above is that, and nothing else. It cannot affect `softfp64.h`,
which is integer-only by construction, but it is a live hazard for the fp32
sieve path in `k_apply`, where `s = fabsf(acc)` is *deliberately* a
catastrophically cancelled quantity. The existing `fmaxf(s, 1e-30f)` clamp
(`bench_kernels.cu:566`) lands above the smallest normal float (~1.18e-38), and
a cancellation deep enough to reach subnormal territory would already have
tripped the `NORM_CANCEL_TOL` guard into the fp64 path — so the two existing
guards appear to cover it. **Appear to** is doing real work in that sentence;
Phase 5 should confirm it rather than inherit it.

---

## 6. The load-bearing design decision

**Write a thin CUDA-runtime-shaped shim rather than rewriting the
orchestration.**

`pipeline.cuh` is ~4,000 lines of host orchestration. Ported by hand it is
the bulk of this project; ported through a shim that presents
`mtlMalloc`/`mtlFree`/`mtlMemcpy`/`mtlMemset`/`mtlEventRecord`/`mtlStream_t`
with CUDA's signatures, it becomes a rename pass — the same leverage
`hipify-perl` gave the HIP port, which is why that port's biggest files
needed zero hand fixes.

Paired with it, a variadic launch helper:

```
LAUNCH(kernel_name, grid, block, smem, stream)(args...)
```

that binds pointer arguments as buffers and packs scalars into a single
params buffer, so 44 kernels do not need 44 hand-written encoder wrappers.

This decision is worth roughly 3x on total effort. Everything else in this
plan is downstream of it.

---

## 7. File strategy

Following the HIP port's split: fork files with real logic differences, fix
shared files in place only when the change is invisible to the other build.

**Forked (new):**

| File | From |
|---|---|
| `bench/metal/metal_rt.h` / `.cpp` | new — runtime shim, section 6 |
| `bench/metal/msl_compat.h` | new — types, address-space macros |
| `bench/metal/softfp64.h` | new — IEEE-754 fp64 in integer ops |
| `bench/metal/portable_log2.h` | new — bit-exact fp32 log2, host+device |
| `bench/metal/scan.metal` | CUB replacement, modelled on `td.cuh:365-420` |
| `bench/bench_kernels.metal` | device half of `bench_kernels.cu` + `td.cuh` + `cofac.cuh` |
| `bench/fbgen_gpu.metal` | device half of `fbgen_gpu.cu` |
| `bench/pipeline_metal.cpp` | `pipeline.cuh` |
| `bench/cofac_metal.cpp` | host half of `cofac.cuh` |
| `bench/bench_kernels_metal.cpp` | host half of `bench_kernels.cu` |
| `bench/bench_main_metal.cpp` | `bench_main.cu` |
| `bench/Makefile.metal` | `Makefile` |

**Shared, fixed in place if needed:** `bench.h`, `slab.h`, `platform.h`,
`bigint.cuh`, `plattice.cuh`, `prp.cuh`. The HIP port's lesson applies
directly: `grep -rn __CUDACC__` across the **whole** tree is step one, not
incremental discovery through compile errors. Metal's equivalent hazard is
that `.metal` files get none of these headers' host-side content at all.

**Host language:** plain C++ against `metal-cpp` rather than Objective-C++,
keeping `bench_main_metal.cpp` and `pipeline_metal.cpp` ordinary C++ and
matching the HIP fork's shape. Revisit only if metal-cpp proves to lag the
Metal 4 API in something needed here.

---

## 8. Phases

Each phase has a gate. A phase is not done until its gate is green.

### Phase 0 — Toolchain and device probe — **DONE**, see section 5.3
Install the Metal toolchain component. Write `probe.metal` + host driver
establishing, on real hardware rather than from documentation:
- `threadExecutionWidth` / SIMD-group size = 32
- `maxThreadgroupMemoryLength` (expected 32768)
- `simd_ballot` lane ordering is ascending — `td.cuh:647` depends on it
- whether `mulhi(ulong, ulong)` exists, else a 32x32 decomposition
- int64 buffer atomics support (`supportsBufferInt64Atomics` or equivalent)
- argument buffer tier
- `recommendedMaxWorkingSetSize`, `maxBufferLength`
- command-buffer timeout behaviour while the GPU drives the display
- that every warp intrinsic matches CUDA's value bit-for-bit

**Gate:** a recorded table of device facts in `CLAUDE.md`, the way
`hip-port`'s `probe.hip` section does.

### Phase 1 — Branch and ledger
`metal-port` branch (done), Metal `CLAUDE.md` with ground rules, device
facts, and an empty drift ledger.

**Gate:** committed.

### Phase 2 — Portability primitives, tested standalone — **DONE**
The real work of this port, and it must be bit-exact before a single kernel
is ported:
- `softfp64.h` — add/sub/mul/div/compare/`i64→f64`/`f64→f32`, round-to-
  nearest-even, in `ulong` integer ops. Cold paths only, so cost is
  irrelevant and exactness is everything.
- `portable_log2.h` — one fp32 algorithm (exponent extraction plus a fixed
  minimax polynomial, `fma`-only, no reassociation) compiled for host **and**
  device so they agree by construction. Phase 0 measured MSL's `log2` at
  50.03% disagreement with the host, so this is **load-bearing, not
  speculative**. What Phase 7 still decides is only its *scope*: Metal-only,
  or a `-DNORM_PORTABLE_LOG2` on the CUDA build as well.
- `msl_compat.h` — type and address-space shims.

**Gate: PASSING.** `make -f Makefile.metal metalcheck`, three parts:

| part | what it proves | result |
|---|---|---|
| `sf_test_host` | the host build of `softfp64.h` against **hardware fp64** | 3,752,050 operations, 0 mismatches |
| `sf_test_device` | the **Metal build** of the same headers against the host build | 2,097,152 results, 0 mismatches |
| `sf_sites_test` | `sf_bn_to_double` / `sf_cof_gap_test` against `prp.cuh`'s own fp64, with `prp.cuh` compiled **unmodified** | 800,000 cases, 0 mismatches |

Chained, the first two say the GPU's soft-fp64 *is* fp64, bit for bit. The
corpus is deliberately hostile: subnormals, infinities, NaNs, exact powers of
two, near-equal operands that force massive cancellation, and fully random bit
patterns — not just the value ranges the siever happens to produce. NaN
payloads are the one thing compared loosely, because IEEE does not pin them
and neither does any vendor.

Two implementation notes worth keeping:

- `sf_fma` is genuinely single-rounded: the 128-bit product keeps its low
  word alive into the addition, rather than rounding to fp64 first. This
  matters because nvcc contracts `a*b+c` into fma by default, so the CUDA
  build's norm Horner is almost certainly fused and the Metal build has to be
  able to fuse identically. (Phase 5 must confirm *which* multiply nvcc fuses
  at `bench_kernels.cu:531`; the primitive is ready either way.)
- In `bn_to_double` the contraction question is moot and provably so:
  multiplying by 2^32 is exact, so `fma(d, 2^32, v)` and `d * 2^32 + v` round
  identically. Recorded in `sf_sites.h` rather than left for someone to
  rediscover.

### Phase 3 — Runtime shim — **DONE**
`metal_rt.h` (plain C++, no Metal headers) over `metal_rt.mm` (the only
Objective-C++ in the port), plus `MTL_LAUNCH` and a pipeline-state cache.

**The central trick is the allocation registry.** CUDA code does pointer
arithmetic on device pointers freely — `bk->bucket + i * stride` is handed to
a kernel as though it were a base pointer — but Metal binds an `MTLBuffer`
plus an offset, not an address. So every allocation is a shared-storage
`MTLBuffer`, `mtlMalloc` returns its `contents` pointer, and a sorted registry
maps any pointer back to (buffer, offset) at bind time in O(log n). Ported
code keeps doing arithmetic on real addresses and never learns Metal is
underneath.

**Streams** own one `MTLCommandQueue` and at most one open command buffer with
one open encoder; dispatches accumulate into that encoder and it is closed
only when something demands ordering. **Events** commit the command buffer,
because `GPUEndTime` is the only timestamp reachable without counter sample
buffers — so every `cudaEventRecord` becomes a pipeline flush. With 78 record
sites in the CUDA source, a faithful port will serialise more than CUDA does.
That is a Phase 8 question (sample counters inside the encoder, or drop the
harness-only records), and it is recorded at the function rather than
discovered later.

**Binding convention, the thing that keeps each launch site mechanical:** CUDA
parameter *i* becomes `[[buffer(i)]]`, same order. Templated kernels reach MSL
through `[[host_name]]` under the rule *base, then each template argument,
joined by `_`, bools as 0/1* — `k_td<1,0,0,false>` is `"k_td_1_0_0_0"`.

**Gate: PASSING.** `make -f Makefile.metal rtcheck` — 17 checks covering
grid-stride launches, interior-pointer binding, blit `memset`, dynamic
threadgroup memory with a device atomic, the threadgroup-ceiling refusal,
both templated instantiations, a missing kernel name, a `__constant__` symbol,
a by-value struct, streams, events, elapsed time, and unregistered-pointer
detection.

Two things the gate found or failed to find, both worth carrying forward:

- **A real bug, fixed:** `mtl_launch_end` originally returned the *sticky*
  error, so a launch reported a failure that had happened somewhere else
  earlier. CUDA's launch returns its own status and only `cudaGetLastError()`
  is sticky. Conflating them would have produced spurious failures deep in a
  ported band and cost far more to find later.
- **A limitation, recorded rather than papered over:** the cross-stream
  ordering check does *not* isolate `mtlStreamWaitEvent`. Its negative control
  — the same sequence with the wait removed — gives the same answer every
  time, so Metal is already ordering the two queues itself, almost certainly
  via the automatic hazard tracking a default (tracked) `MTLBuffer` gets. The
  pair establishes that the wait does not deadlock, corrupt or reorder; it
  does **not** establish that the wait is load-bearing. Re-test in Phase 5,
  when two sides run genuinely concurrent work.

**Still deliberately absent:** a struct containing device pointers cannot pass
through `mtl_launch` — a host pointer means nothing to a shader. The tree has
exactly one such kernel, `k_cof_enqueue` taking `cofq_t` by value
(`cofac.cuh:1410`, ~27 pointers); it needs an argument buffer, which is
Phase 6. `mtl_launch` rejects it loudly rather than binding garbage.

### Phase 4 — `fbgen_gpu.metal` + scan/select — **DONE**

**The gate turned out to be stronger than planned, and it is worth saying why.**
The plan assumed the reference had to be a CUDA-generated factor base, which
this machine cannot produce. But `fbgen.c` — the CPU generator — builds and
runs natively on macOS with no changes at all. So the reference is the
*independent CPU implementation*, not another GPU build, and the tree already
had the script that compares them: `fbgpucheck.sh`. The HIP port could not do
this (its CPU `fbgen` segfaulted under MinGW) and fell back to generating with
its own port and running `fbtest` over the result, which is a much weaker
claim.

**Gate: PASSING — `make -f Makefile.metal fbcheck`, 19 cases, all
byte-identical to the CPU generator:** the `lim=2` and `lim=3` boundaries,
every supported polynomial degree 1 through 8 (crossing the CAP=6/CAP=8
dispatch boundary at 6 and 7), GNFS powers/ramification, prime-only, the
octic, C147 degree-5, three in-memory `afb_build_gpu` production-consumer
cases, and two negative controls (an oversized `--segment-odds` is refused; a
failed `--compare-fb` neither publishes over the existing output nor leaves a
`.part` behind).

**4a — the CUB replacement.** `metal/scan.metal` + `metal/metal_scan.cpp`
give an exclusive prefix sum and a stable flagged compaction with *CUB's
calling convention* (NULL temp to query the size, allocate once, pass it
thereafter), so each of the eight `cub::` call sites ported by changing the
name alone. Shape follows `td.cuh:365-420`'s existing three-pass scan and the
tree's existing scan-then-scatter compaction — including its reason
(`td.cuh:455`): an ordered scan gives every selected element a deterministic
slot, so output is byte-reproducible, which is exactly what makes the CPU
diff meaningful. Separately gated by `make -f Makefile.metal scancheck` at
every block/two-level boundary (255/256/257, 65535/65536/65537, 8M =
`GPU_FB_DEFAULT_SEG_ODDS`), with uint32-wrapping values and 0%/50%/100%
densities.

**4b — the device half.** ~1,150 lines. Three things the transformation had
to do, all forced by MSL:

1. *Address spaces.* MSL rejects an unqualified pointer parameter outright, so
   all 51 function heads are qualified. Almost all are `thread` — the original
   builds polynomials in registers and passes their addresses, which maps
   cleanly. Three take `constant` (`d_big_mod`, `d_big_mod_mont`,
   `d_ctx_big`), because their only callers pass `&c_alg[i]`, `&c_y0`, `&c_y1`.
2. *`__constant__` globals become kernel parameters*, bound after the kernel's
   own so the "CUDA parameter i → `[[buffer(i)]]`" rule survives, and threaded
   down into the two device functions that read them.
3. *Kernel bodies are unchanged.* Rather than rewriting every
   `blockIdx.x * blockDim.x + threadIdx.x`, the ids arrive as MSL attributes
   and `FB_KERNEL_IDS` re-exposes them under CUDA's names; `atomicAdd` is a
   one-line shim over `atomic_fetch_add_explicit`. A body textually identical
   to the CUDA original cannot have acquired a transcription bug, which is
   worth more here than elegance.

Templated kernels become `static inline` bodies with thin concrete wrappers
named by the port-wide rule, giving 10 explicit instantiations.

**4c — the host half.** ~1,350 lines, and **this is the Phase 3 bet paying
off**: 15 launches rewritten, every `cuda*` call renamed onto `metal_rt.h`,
and nothing else. The shim also gained lazy initialisation, so the ported
call sequence is identical to CUDA's — which has no init call — rather than
acquiring a Metal-shaped prologue that would be one more place to drift.

**Porting aids, not sources of truth:** `metal/gen_fbgen_metal.py` and
`metal/gen_fbgen_host.py` produced the first drafts and are committed so a
later CUDA-side change to this file can be re-diffed rather than re-ported
from memory. They are deliberately **not** wired into the build; the generated
files are committed and reviewed like any other source.

### Phase 5 — Sieve kernels — **device half DONE, gate not yet run**

**Done: `metal/bench_kernels.metal` compiles clean**, 19 kernel entry points
including `k_transform`, `k_fill_atomic`, `k_apply` (4 instantiations),
`k_resieve_rewalk`, `k_purge`, `k_purge_prime`, `k_intersect_compact`,
`k_build_summary*`, `k_snapshot_bounds`, `k_fill_segmented`. Supporting:
`metal/cuda_msl_compat.h` (CUDA's device vocabulary in MSL) and
`metal/plattice_msl.h` (plattice.cuh's arithmetic, verbatim, with address
spaces added — `plattice.cuh` itself is untouched and still shared by the
CUDA and CPU builds).

**The fp64 norm fallback is ported and is the reason Phase 2 exists.**
`bench_kernels.cu:525`'s eight-line `double` block is rewritten onto
`softfp64.h` — the same computation with the same rounding, not an
approximation, since Phase 2 proved those primitives bit-exact against
hardware fp64.

**`log2f` maps to `pl_log2f`, not `metal::log2`.** Phase 0 measured MSL's
builtin disagreeing with the host on 50.03% of inputs by up to 3 ULP, which
moves sieve cells. This makes the Phase 7 question concrete rather than
hypothetical: the CPU reference (`verify_cpu.c`) still uses libm's `log2f`,
so the parity gate below will *measure* the disagreement directly.

**Two kernels are deliberately absent, and this is a real gap.** `k_fill_l1`
and `k_fill_l2` each declare `128*64` uint32 plus two 128-word counters —
**33,792 B against Apple's hard 32,768 B threadgroup ceiling, over by exactly
1 KB**. MSL additionally forbids threadgroup declarations inside the non-kernel
helper the templated form needs. They are the *two-level* fill path, and apply
requires single-level 4-byte records (`bench_kernels.cu:2660`), so the
production sieve runs `k_fill_atomic` and never reaches them. Closing the gap
means retuning `L1_CAP`/`L2_CAP` from 64 to 62 — a performance change, so it
belongs in Phase 8, measured rather than guessed.

**Also ported:** the five 64-bit diagnostic counters (`nlost`, `nprobe`,
`npass1`, `nread`, `npre`, `nqb`) become pairs of uint32 words with the carry
folded across them, since Metal has no 64-bit atomics at all. Two
little-endian uint32 words at one address *are* a little-endian uint64, so the
host's existing 8-byte readback of each is unchanged — no host edit at all.
Sound only because they are diagnostics: the pair is eventually consistent
rather than atomic as a unit, and nothing computes from them.

**Remaining for this phase: the host harness and the gate.** The tree already
has the ground truth — `verify_count_updates` (per-region fill counts) and
`verify_apply_region` (replays a region's records on the CPU and compares
*every cell*, including the norm init and threshold). Both are pure host C in
`verify_cpu.c`. The harness needs to build a factor base (our own
`fbgen_gpu` can generate it), set up `qlat_t`/`norm_t`, run
transform → fill → apply through `metal_rt`, and compare against both.

**Gate: PASSING — `make -f Makefile.metal sievecheck`.** At logI 13, J 4096,
`lim` 1,000,000 on `oracle/c183.poly` with the oracle's own special-q
(q = 120000053, rho = 112625526):

| check | result |
|---|---|
| shared struct layout (`norm_t`, `plat_t`) host vs device | agree — 192/192 B, matching member offsets |
| fill, per region | **all 4,096 regions match the CPU exactly**, 67,071,278 updates |
| apply, per cell | **all 4,194,304 cells match the CPU exactly** over 512 regions |
| cells over the survivor threshold | 623,098 on both sides |

`norm_t` crosses the boundary by value and its two fp64 members are
*reconstructed* on the device rather than shared, so the layout check runs
first: a silent mismatch would corrupt every norm while still producing
plausible output.

**The fill reference had to be rewritten, and the reason is a CUDA-tree
property worth knowing.** `verify_count_updates` walks with
`pl_first`/`pl_next` — the **32-bit** walk, whose `pl_add32_sat` saturates to
`UINT32_MAX` (ending the walk) whenever an increment's high word is nonzero.
`k_fill_atomic` walks with `pl_first64`/`pl_next64`, where the same increment
wraps and the walk continues. **They are different walks**, and on this factor
base they disagree by 2.4%: 65,519,535 updates against 67,071,278, differing
in 4,095 of 4,096 regions.

This was confirmed **on the host, with no GPU involved** — a standalone
comparison of the two walks over the same lattices differs on 646 moduli,
321 of which disagree on the very first position. So it is not a Metal defect.
The gate therefore compares against a 64-bit CPU walk, which the GPU
reproduces exactly, and reports the 32-bit figure alongside. Whether the CUDA
build's own `--verify` is affected is a question for the CUDA side and is
recorded in §10.

### 5b. Candidate counts do not compare across sievers — relations do

Worth writing down, because the numbers invite a false alarm. On the parity
special-q our pipeline emits **1,845 cofactorisation candidates** where
`oracle/c183.q120000053.cofac_candidates.txt` has **1,851**. The difference
decomposes exactly, and neither half is a defect:

- **The 7 las has that we lack are precisely the 7 relations that need no
  cofactorisation** — proven, not inferred: the set difference
  `oracle − ours` is *identical* to the relation set of a trial-division-only
  run. The two pipelines mean different things by "candidate". las's file comes
  from `-batch-print-survivors`, which dumps every post-sieve survivor;
  `--candidates` dumps only records that actually enter the cofactoriser, and
  a survivor already within `lpb` on both sides is a finished relation. It is
  the same 7 `cofcheck.sh`'s first case pins.
- **The 1 we have that las lacks** decodes to `i = -16384, j = 12803` — the
  leftmost column of the region, a legal interior point — and it does not
  become a relation. The run used `cofcheck.sh`'s pinned allowance, which the
  program's own stderr reports as **8.03 bits looser** than the derived
  policy, against las's `lambda0 2.35 / lambda1 3.5`. A marginal survivor
  landing on one side of a deliberately different bound is expected.

**The invariant is the relation set, and it matches exactly**: our 37 relations
are the *identical* `(a,b)` set as las's 37 — 37 in both, zero either way.
That is why the tree pins relation counts and never candidate counts, and why
`oracle/README.md` cautions from the other direction that containment "does
not establish yield equivalence".

### 5a. The Phase 7 log2 question, measured early — and my earlier estimate corrected

**Zero of 4,194,304 cells differ**, with the GPU running `pl_log2f` and the
CPU reference running libm's `log2f`. Verified that the substitution really
reaches the kernel (the preprocessed device source calls
`pl_log2f(fmax(s, 1e-30f))`), so this is a real result and not a silent
fallback.

Phase 2's note that MSL's `log2` "disagrees on 50.03% of inputs" was correct
about *the logarithm* and misleading about *the sieve*, and the arithmetic is
worth writing down. A cell is an integer, `floor(scale * (log2M + log2(s) -
bias) + 0.5)`. `pl_log2f` differs from libm on ~1% of inputs, and when it does
the difference is 1-3 ULP of a result of magnitude ~100, i.e. ~1e-5 absolute;
scaled by 1.925 that is ~3e-5, and it only matters if it straddles a rounding
boundary. So the expected rate is roughly `0.01 x 3e-5` = **3e-7 per cell** —
about one cell in three million, not the "tens of thousands per q" my Phase 2
framing implied. Observing 0 in 4.2M is exactly consistent with that.

This substantially de-risks Phase 7, but does not settle it: it is one
geometry, the expectation is order-1 rather than 0, and the question that
actually matters is CUDA-vs-Metal relations, not Metal-vs-libm cells.

### Phase 6 — Trial division and cofactorisation — **groundwork done**

**6a. The `cofq_t` argument buffer: SOLVED and gated.** This was the one
mechanism in the whole port with no CUDA-shaped equivalent, so it was
de-risked before writing any of the 2,000 lines that depend on it.
`k_cof_enqueue` takes `cofq_t` **by value** (`cofac.cuh:1410`) — ~27 device
pointers in one struct — and a host pointer means nothing to a shader. On
Metal the struct carries GPU addresses and the kernel declares its members as
`device T*`. Two new `metal_rt` entry points make that work:

- `mtlDeviceAddress(p)` — a registered allocation pointer, interior pointers
  included, to the address the shader dereferences.
- `mtlUseResource(p)` — residency. Metal only guarantees a resource is mapped
  if it can see it bound, and one reached through a raw address is invisible
  to it.

**Gate: PASSING — `make -f Makefile.metal argbufcheck`**, and its negative
control has real teeth: omit the residency calls and 4,095 of 4,096 values
come back wrong.

Getting that control right took two attempts, and the reason is worth
recording. Run *after* the positive case it passes — **residency, once
granted for a resource, persists within the process** — so the control now
runs FIRST, before anything makes those buffers resident. A control that
passes for the wrong reason is worse than none.

**6b. The cofactoriser runs and hits the golden number.**
`make -f Makefile.metal cofaccheck` — `run_cofac()` on the oracle's own
candidate list for the parity special-q, **37 relations in all four
configurations**: rho and ECM, at 3-limb and 4-limb cofactor width. 37 is the
count `cofcheck.sh` pins and the count las itself finds at this q.

`metal/cofac.metal` (11 kernels: six `k_cofac` instantiations, two compaction
kernels, and the three-pass scan `cf_run_rounds` borrows from `td.cuh`) and
`metal/cofac_metal.cpp` (the host driver). Four things had to change beyond
renaming:

- **`goto` is rejected by MSL outright.** `mz_rho`'s `goto found` became a
  flag and two breaks — the same control flow, since the outer loop is
  `for (;;)` and exits only by `return` or that jump.
- **`__CUDACC__` hid the entire host driver**, exactly as the HIP port's
  ledger warns. `cofac.cuh` wraps its GPU host half — `run_cofac` included —
  in `#if defined(__CUDACC__)`; compiled as ordinary C++ that block vanishes,
  so the file built with **zero errors** and then failed to link. The two
  guards want opposite treatment: the `CF_FN`/`CF_HD` block needs the
  *non*-CUDA branch (host wants `static inline`, not `__device__`), every
  other guard needs enabling.
- **A launch whose instantiation is a template parameter.** `cf_run_rounds<L>`
  launches `k_cofac<L, METHOD, STAGE2>`, so the mangled name cannot be formed
  textually at all — the naive rewrite produced `"k_cofac_L_0_0"`. Selected at
  run time from `L` via `MTL_LAUNCH_NAMED`, which is what the CUDA compiler
  was doing at compile time.
- **By-value struct kernel parameters.** `mz<L> lim2` is the kernel's own copy
  in CUDA but arrives as a `constant` reference in MSL, which the body then
  cannot take the address of. The wrapper gives the body back a thread-local
  copy under the original name.

The oracle file is a candidate list (`a b cof0 cof1`) while `run_cofac` parses
`mkcofbatch`'s batch format, so the harness converts, following the sign
convention `mkcofbatch.c` documents (negative exactly when bits exceed that
side's lpb). The factor lists are left **empty**, so every prime below `lim`
has to come out of the split — a harder test than the pipeline's own path,
not an easier one.

**6c. `td.cuh` forked; the whole device side now compiles.**
`metal/td.metal` — 26 kernels: `k_td` (7 instantiations), `k_td_record_warp`,
`k_resieve_scatter` (5), `k_emit_ranked` (2), `k_classify`, `k_cand_stats`,
`k_group_counts`, the three-pass scan, `k_tdsmall_advance`, `k_accept_flags`,
`k_scatter_sel`, `k_gather_ab`. With `bench_kernels.metal` (19),
`cofac.metal` (8), `fbgen_gpu.metal` (16) and `scan.metal` (6) that is
**75 kernels in one metallib**, and every device translation unit compiles
with zero errors.

`k_classify` is where Phase 2 pays off a second time: it calls `cof_classify`,
whose CADO gap test is written against doubles this GPU does not have.
`prp_msl.h` routes it through `softfp64`, and `k_classify`'s `double lim`
parameter is now carried as its 64-bit pattern — **the host passes the same
eight bytes it always did**, so there is no host-side change.

Also generated as MSL, from headers that stay untouched and shared:
`bigint_msl.h`, `prp_msl.h`, `plattice_msl.h`, `slab_msl.h`, and `td_msl.h`
(td.cuh's pre-guard section — the types, the `TD_*` bounds, `td_mod_magic`
and `SS_KSHIFT`). `bench_kernels.metal` now includes `td_msl.h` instead of
carrying its own copies, so the device build states `SS_KSHIFT` exactly once.

**The generator bug worth recording.** Reducing `#if defined(__CUDA_ARCH__)`
blocks with a regex assumed a bare `#else`. `bigint.cuh`'s block is
`#if / #elif defined(_MSC_VER) / #else / #endif`, so the regex ate the `#if`
and `#else` and left the `#elif` **orphaned** — which turned the entire rest
of the header into a dead branch. The file still compiled; every symbol after
that point simply ceased to exist, and the failure surfaced two translation
units away as "unknown type name 'bns_t'". That is a *missing-code* bug, not
a wrong-branch one, exactly the class the HIP port's ledger calls out. Fixed
with a real preprocessor walk, plus an assertion that refuses to emit a header
whose conditionals are unbalanced or whose `#else`/`#elif` is orphaned.

**6d. The inline-queue kernels, and the argument buffer wired end to end.**
`k_cof_enqueue` (4 instantiations), `k_rel_pack`, `k_cof_gate`,
`k_cof_status_hist`, `k_rel_flags`, `k_rel_gather`. **The device side of the
port is now complete: 84 kernels in one metallib**, every translation unit
compiling with zero errors.

`cofq_t` is the payoff from 6a. The device sees `cofq_dev_t`, a struct of
17 GPU addresses — only the fields the kernels actually read; the host-side
bookkeeping, including the `double` members MSL could not express, stays on
the host. The host mirrors it in `cofq_argbuf()` and passes it as an
`mtl_argbuf_t`, a new `metal_rt` type that binds the struct **and** calls
`mtlUseResource` on every pointer inside it in one step — so a call site
cannot bind the buffer and forget the residency, which is the failure that
gives garbage rather than an error.

`k_cof_status_hist`'s eight 64-bit status counters became uint32 pairs, the
last 64-bit atomic in the tree. As everywhere else, two little-endian uint32
words at one address *are* a little-endian uint64, so the host's readback is
unchanged.

`td_msl.h` now also carries the `TD_*` bounds that `td.cuh` keeps inside its
own `__CUDACC__` guard, lifted by name, so every Metal translation unit shares
one statement of each.

**6e. `pipeline.cuh` ported; the combined host TU compiles.**
`metal/pipeline_host.inc` (3,743 lines, 26 launches) and
`metal/bench_host.cpp` (2,056 lines, 38 launches) — the latter mirrors
`bench_kernels.cu`'s role as the single host TU that also pulls in the
cofactor and pipeline code. **Both compile with zero errors.**

**This is the Phase 3 bet settling.** `pipeline.cuh` has no device code and no
`__CUDACC__` guards at all; with `metal_rt.h` presenting CUDA's names and
CUDA's blocking semantics, it ported by renaming plus one real transformation
— every launch there sits inside a function templated on `bool SLABBED`, so
the mangled kernel name cannot be formed textually and becomes a ternary over
the two concrete names, which the compiler folds.

Four host-side details that needed more than a rename:

- `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, n)` becomes
  `mtlFuncSetMaxThreadgroupMemory`, which *validates* against the device
  ceiling rather than raising it — the check that matters on Apple, where
  32 KB is well under CUDA's opt-in tier.
- `cudaDeviceGetAttribute(cudaDevAttrMaxGridDimX)` and
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` become small helpers over
  `mtlGetDeviceProperties`.
- `run_bench`'s `LAUNCH_APPLY` macro parameterises the template arguments, so
  the kernel name is pasted by the preprocessor; all nine `k_apply`
  instantiations (both cell widths x atomic x norm mode) are now built rather
  than failing at run time with a missing-kernel error.
- CUDA's async calls default their stream argument; `extern "C"` cannot carry
  defaults, so `metal_rt.h` mirrors them as C++ overloads.

**The last hand-copied constants are gone.** `metal/td_host.h` is generated
from `td.cuh` and lifts the eight `TD_*`/`TDF_*` names it keeps inside its own
`__CUDACC__` guard; `bench_host.cpp` lifts the six launch-shape constants that
live in the device region it strips. Everything now tracks its original by
name. `metal/portlib.py` holds the one copy of the launch rewriter and the
rename table, because two copies of a transformation this fiddly would drift —
and a drifting transformation produces a file that compiles and computes
something else.

**6f. `bench_main.cu` ported; `./bench` links and the pipeline RUNS.**
`metal/bench_main_metal.cpp` — no device code and no launches at all, only
device selection and version reporting. CUDA's `cudaInitDevice` /
`cudaSetDeviceFlags` dance exists to attach `cudaDeviceScheduleBlockingSync`
across a `CUDART_VERSION` split; Metal has no scheduling-flags concept at all,
because a command buffer's `waitUntilCompleted` already blocks the calling
thread. There is nothing to configure and nothing to verify.

`--verify-only` passes through the real binary: 12 Franke-Kleinjung walk cases
and 7 forced/native slab walk cases, exit 0. A full `--pipeline` band ran end
to end on the M3 at a reduced geometry and exited 0.

**The predicted threadgroup constraint bit exactly as documented.** At CUDA's
default `log_region = 14`, `k_apply` wants 32,896 B — 128 bytes over Apple's
hard 32,768 B ceiling — and the run fails closed with `pipeline.cuh`'s own
message. Failing closed on the *default* geometry is not a usable default, so
the Metal build now defaults `log_region` to **13**, the largest value that
fits, exactly as section 5.1 planned. `--region` still overrides and the CUDA
build is untouched.

**Two real bugs found by running it, both worth recording:**

1. **`NULL` is `0L` in C++, not a pointer.** CUDA code passes `NULL` for
   optional buffers (`k_apply`'s `dump`, `dbg_cells`, `probe_out`). The
   binding template took its non-pointer branch and bound **eight bytes of
   zeros as a constant buffer**, so the kernel's `if (dump)` saw a perfectly
   good non-nil address and dereferenced it — a GPU page fault with no clue
   which argument caused it. `metal_rt.h` now has a `std::nullptr_t` overload
   and the generators rewrite a bare `NULL` argument to `nullptr`.
2. **A threadgroup array must be declared in the wrapper, not made a
   parameter.** `k_td`'s `threadgroup tdsmall_t tile[TD_TILE]` was hoisted to
   a `[[threadgroup(0)]]` parameter, which silently has **zero length** unless
   the host sets it — and these were static `__shared__` arrays in CUDA, with
   no host involvement at all. The wrapper kernel declares them now.

**`cof_classify` is verified on the device**: 0 of 65,536 verdicts differ from
`prp.cuh`'s own fp64, with `prp.cuh` compiled unmodified
(`make -f Makefile.metal classifycheck`). Phase 2 had tested the soft-float
primitives device-vs-host and the call sites host-vs-fp64, but never
`cof_classify` as `prp_msl.h` assembles it on the device. That gap is closed,
and it rules the soft-float path out of the remaining problem.

**`cofcheck.sh` DOES NOT PASS.** Its negative controls and refusal cases pass;
every relation-count case does not. Before the threadgroup fix a band ran to
completion and produced **125 cofactorisation candidates where the oracle has
1,851**, and 0 relations against an expected 7 or 37. After the fix the
production-geometry run is SIGKILLed with no output flushed, which is a
*different* failure and has not been diagnosed.

**Next diagnostic, in order:**
- The SIGKILL. Run under `log stream` or with output unbuffered to see how far
  it gets; check whether it is memory pressure or a GPU fault loop.
- Then the candidate shortfall. Fill and apply are cell-exact (Phase 5) and
  `cof_classify` is verdict-exact, so the untested device stages are `k_td`,
  `k_resieve_scatter`, `k_emit_ranked` and `k_intersect_compact`. `k_td` is
  the largest and the one whose threadgroup handling just changed — extend the
  Phase 5 harness to compare its cofactor output against a CPU replay, the way
  `verify_apply_region` does for apply.

**6g. `cofcheck.sh` PASSES. Phase 6 is done.**

`make -f Makefile.metal cofcheckgate` — **54 PASS, 0 FAIL, `cofactor golden
test passed`, exit 0.** This is the formal gate the HIP port used, and it is
the tree's own script, unmodified except for two BSD/GNU portability
fallbacks recorded in the drift ledger.

It covers trial-division-only, rho and ECM through the inline queue, ECM with
and without stage 2, chunked launches, 3-limb against 4-limb cofactor width,
`lpb 33` with factors above 2^32, the lpb/mfb boundary refusals, the standalone
splitter against the inline queue, multi-q generated bands byte-identical to
cached and streamed, and the negative controls for corruption and
compositeness.

**Two more bugs the gate found, both in the last stretch:**

1. `k_td`'s threadgroup tile was a `[[threadgroup(n)]]` parameter, which is
   zero-length unless the host sets it. The band ran to completion and
   produced **125 cofactorisation candidates where it should produce 1,845** —
   a wrong answer, not a crash. Fixed by declaring the array in the wrapper
   kernel, where MSL allows it and CUDA's static `__shared__` needed no host
   involvement at all.
2. `head -c -1` in `cofcheck.sh` is a GNU extension BSD rejects, which aborted
   the script after 43 passing cases. Portability, not a port defect; ledgered.

**Still to do for this phase:**
- **One duplication remains, on the HOST side only**: `cofac_metal.cpp`'s
  copies of `TD_SCAN_BLK` and `TD_FMAX`. Forking `td.cuh` fixed the device
  side but not this — `td_msl.h` is MSL. The clean fix is a CUDA-side change,
  lifting those two defines above `td.cuh`'s `__CUDACC__` guard, which would
  need a drift-ledger row.
- Nothing. Phase 6 is complete.

**An intermediate gate exists and should be used first:** `run_cofac()`
(`cofac.cuh:2728`) is a standalone cofactorisation entry point that reads a
batch file, and `oracle/c183.q120000053.cofac_candidates.txt` (1,852
candidates) is in the tree. That exercises the whole cofactor path without the
sieve pipeline, exactly as `fbgpucheck.sh` did for Phase 4.

**Final gate:** `cofcheck.sh` — all ~52 pinned cases, `cofactor golden test
passed`, exit 0. It needs `oracle/c183.fb1`, which our own `fbgen_gpu`
generates in 7.1 s.

### Phase 7 — Relation comparison, and the log2 decision
Run the full pipeline on the oracle jobs against the CUDA build. Then
**measure** rather than assume:
1. Dump the sieve-log arrays for a fixed special-q from both builds; count
   cells differing by +/-1.
2. Diff the relation sets.

**Phase 2 has already narrowed this decision from three options to two.**
The original plan held out "portable log2 in the Metal build only" as a
middle path that left CUDA untouched. Building the thing showed that option
is empty: `pl_log2f` disagrees with the host's `log2f` on 1.02% of inputs by
up to 3 ULP — the same order as `metal::log2`'s 3 ULP. A portable log2 is not
*more accurate* than MSL's, and was never meant to be; its whole value is
that **both sides compute the same sequence**. Used on one side only it buys
exactly nothing for byte-identity.

So the real choice is:

1. **Both builds use `pl_log2f`** (`-DNORM_PORTABLE_LOG2` on the CUDA side).
   The sieve bytes then agree by construction on any GPU, any vendor, forever
   — this also retires the same latent risk for the HIP port and for future
   NVIDIA toolkit versions. Costs a CUDA-side change and a drift-ledger row,
   and requires re-pinning `cofcheck.sh`'s counts if any relation moves.
2. **Accept a divergent relation set**, and weaken the gate from
   byte-identical relations to relation-set comparison plus the built-in
   self-check, documenting the expected divergence.

Option 1 is the better engineering and the one to expect; the measurement
decides whether it is *necessary*, and how much moves if it is adopted.

**Gate:** a recorded number — cells flipped, relations gained/lost — and a
decision justified by it, not by this paragraph.

## PHASE 7 IS DONE. The relations are BYTE-IDENTICAL to the CUDA build.

Run against a real NVIDIA GPU, which this project has access to after all: a
**GTX 1080 Ti (sm_61, driver 580.178.04)** in an NRP/Nautilus k8s pod on
`nvidia/cuda:12.8.1-devel-ubuntu22.04`, nvcc 12.8.93, building `main` at
`3e15fec` with `GPU_ARCH=61` -- the same environment and one of the same cards
the HIP port used for its own differential checks.

**The inputs are identical before anything is compared.** The pod generated
`c183.fb1` from `c183.poly` with `fbgen --lim 134200000 --maxbits 15` and it
hashes `b4534cb6a0bbfc218cd8d9e3fd9d9f8f9f8d11227788161fb90563192a3a4cf4` --
the manifest's hash for the canonical file. Both builds sieve the same
7,602,601-prime factor base.

**288 special-q, two comparisons, both byte-identical:**

| | CUDA, GTX 1080 Ti | Metal, M3 | relations |
|---|---|---|---|
| **A. matched settings** | region 13, 12c x 4r | region 13, 12c x 4r | **13,485, `cmp` clean** |
| **B. each at its OWN defaults** | region **14**, 12c x 4r, 168 blocks | region **13**, derived **2c x 24r**, 512 blocks, 4 auto-calibrated slabs | **13,485, `cmp` clean** |

B is the result that matters. The two builds disagree about the bucket region,
the cofactor grid, the slab plan, the launch geometry and the curve schedule --
every default this port changed, all at once -- and they emit the same
13,485 relations in the same order, byte for byte. 564,696 records enqueued on
both; side 0 split/dead/stuck 477,071 / 87,625 / 0 on both.

One diagnostic differs and it is not platform: side 1 `dead`/`stuck` is
550,145 / 260 on CUDA and 550,176 / 229 on Metal -- the same +31/-31 that 8o
measured between 12x4 and 2x24 **on Metal alone**. It tracks the curve
schedule, and neither number is a relation.

### How fast is it, against the card that produced the reference?

Same 288 special-q, same settings, same 13,485 relations, measured end to end
on each platform. **GTX 1080 Ti: 204.1 s. M3: 328.0 s. The card is 1.61x
faster overall** -- and the per-stage split is where it gets interesting:

| stage, ms/q | GTX 1080 Ti | M3 | M3/CUDA |
|---|---|---|---|
| transform + plattice | 8.54 | 34.24 | 4.01x |
| **fill** | **212.09** | **153.92** | **0.73x -- Metal FASTER** |
| apply | 102.49 | 372.69 | 3.64x |
| *sieve, both sides* | *323.12* | *560.85* | *1.74x* |
| **norms + trial division** | **17.86** | **125.60** | **7.03x** |
| rational queue | 51.04 | 64.16 | 1.26x |
| algebraic queue | 266.81 | 285.43 | 1.07x |
| *cofactor, device* | *318.28* | *349.71* | *1.10x* |
| **wall per q** | **696.49** | **1124.79** | **1.61x** |

Three things worth reading off it.

**The cofactor stage is at parity -- 1.10x.** A 250 W discrete card with 28 SMs
and dedicated GDDR5X is 10% faster than a fanless 10-core iGPU at the stage
that is 46% of its own wall. That is 8h's finding confirmed from the other
side: the stage is latency-bound on one long dependent ECM chain, so cores and
bandwidth buy almost nothing, and the two platforms converge. The algebraic
queue alone is 1.07x.

**`fill` is FASTER on the M3, 153.9 against 212.1 ms.** The one stage where
unified memory is an advantage rather than a caveat: fill is a scatter of 4-byte
records through an atomic-heavy path, and it does not have to cross a PCIe bus
to reach memory the host also owns.

**Trial division is the weak spot: 7.03x, the worst ratio here**, and it turns
a stage that is 2.6% of CUDA's wall into 11.2% of Metal's. `apply` at 3.64x is
larger in absolute terms (219 ms/q against 108) but is at least the stage a
discrete card should win. **If anyone wants more Metal performance, TD is where
the headroom is** -- it was never tuned, and Phase 8 spent its effort on the
sieve and the cofactoriser because that is where the one-q benchmark pointed.

Context for the ratio: this is a 250 W discrete card against a fanless laptop
iGPU sharing 16 GB of UMA with the OS and the display, and the gap is 1.61x.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display, against a GTX 1080 Ti in an NRP k8s pod.** Neither machine was
otherwise loaded; the M3 figure carries this port's usual throttling caveat and
the pod's does not.

### The log2 decision, settled by measurement

Phase 2 flagged `metal::log2` as differing from host `log2f` on 50.03% of
inputs by up to 3 ULP and framed a choice between adopting `pl_log2f` on both
builds or accepting a divergent relation set. **Neither is needed.** The
divergence does not reach a relation: 0 of 4,194,304 cells differed in Phase 5,
and now 0 of 13,485 relations differ over 288 special-q against the real CUDA
build. Option 1 (`-DNORM_PORTABLE_LOG2` on both) remains available as insurance
against a future toolkit or vendor, and would cost a CUDA-side change and a
re-pin of `cofcheck.sh`; on this evidence it is not a fix for anything
currently broken. **Recommended: do nothing, and keep this measurement as the
reason.**

The same run retires the other two open worries by construction: the
`log_region 13 vs 14` difference moves no relation (it is comparison B), and
neither does soft-fp64 in `cof_classify`, nor any of the 84 kernels.

**Caveat, stated because the gate is only as wide as what it ran.** One card
(sm_61), one composite (c183), one band (288 q from 120000053), at
B1 2000 / B2 60000 with `lpb 31/32`. The HIP port's experience is that sm_61,
sm_75 and sm_86 agree with each other, so card-to-card risk is low, but this
does not prove byte-identity at another geometry -- notably logI 16, which
CLAUDE.md already flags as differentially untested on CUDA too.

**Largely answered already, by Phase 6's gate.** `cofcheck.sh` pins ~25
relation counts **derived from the CUDA build**, and the Metal build matches
every one of them: 7, 37, 36, 59, 66, and the rest, across rho and ECM, both
cofactor widths, and the lpb/mfb boundary. Separately, the Metal build's 37
relations at the parity special-q are the *identical* `(a,b)` set as las's 37.

So the 3 ULP `log2` divergence Phase 2 flagged **has not moved a single
relation** at this geometry — consistent with the ~3e-7 per-cell estimate in
section 5a and with the 0-of-4.2M cell result Phase 5 measured. Option 2
(accept a divergent relation set) is looking unnecessary and option 1
(`-DNORM_PORTABLE_LOG2` on both builds) is looking like insurance rather than
a fix. What remains is to run a *band* rather than a single q, and on a
geometry that exercises the `log_region <= 13` difference, before calling it
settled.

### Phase 8 — Constraints and tuning — **two-level fill: ceiling solved, path still wrong**

**The threadgroup-ceiling constraint is resolved.** `k_fill_l1` and
`k_fill_l2` declare `L1_NBUF*L1_CAP` uint32 plus two 128-word counters, and on
Metal one more word for the `__syncthreads_or` vote:

```
128 * CAP * 4  +  128*4  +  128*4  +  4
```

At CUDA's `CAP = 64` that is **33,796 B** — the driver's own refusal, quoted
verbatim: *"Threadgroup memory size (33796) exceeds the maximum threadgroup
memory allowed (32768)"*. Note it is 4 bytes worse than the 33,792 predicted
in Phase 5, because of the vote word. **61 is the largest value that fits**
(32,260 B, 508 B spare) and the cap is a pure buffering depth — `L1_FLUSH` is
derived from it and every store is bounds-checked against it — so the change
alters how often a buffer flushes and nothing about what is produced.

`__syncthreads_or` (a barrier *and* a block-wide vote, which MSL has no
primitive for) is emulated in threadgroup memory: idempotent writes of 1
between barriers, so the race between voters is benign.

**But the two-level path computes the wrong answer, and it now fails closed.**
With the kernels running, the per-region gate reports **~350-500 of 2,048
regions off by about one, with the grand total exactly right** — which is
precisely the failure `bench_kernels.cu:2617` exists to catch: *"every
placement bug this project has hit had exactly the right total"*. Ruled out so
far:

- **Not the buffer cap.** Sweeping `CAP` over 61 / 48 / 32 changes the error
  count (345 / 340 / 505) without ever reaching zero.
- **Not the vote emulation.** Adding a third barrier after the read — closing
  the one theoretical race — changed nothing.
- **Not the lane mapping.** A new probe (`metal-probe/lanemap.metal`) confirms
  `thread_index_in_simdgroup == tid & 31` and
  `simdgroup_index_in_threadgroup == tid >> 5` over a 512-thread threadgroup,
  so CUDA's `lane`/`warp` arithmetic and `__syncwarp` scope are valid. Phase 0
  had verified ballot ordering and shuffle semantics but never this.

`--mode twolevel` therefore **refuses** on the Metal build, with a message
saying why. It is not the production path — apply requires single-level 4-byte
records (`bench_kernels.cu:2660`), so `--pipeline` never reaches it — so
refusing costs nothing today, and refusing rather than warning is this tree's
own convention for a path known to compute the wrong thing.

### 8a. Launch-geometry tuning, measured

Permitted now that this box counts as a tuning vehicle, and every number below
was taken on **a 10-core M3 in a fanless MacBook Air that also drives the
display**. Say that wherever these are quoted.

**Fill geometry: leave CUDA's alone.** Both axes are flat here.

| fill threads | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| fill (ms) | 21.85 | 21.44 | 21.43 | 21.47 |

Run-to-run spread at a fixed configuration is **±0.44 ms (~2%)**, so that
entire sweep sits inside the noise band — CUDA's 32, chosen for a 23% win on a
5090 driven by NVIDIA L2 behaviour, is marginally the *worst* point here but
not meaningfully so. The block sweep (60 → 9216) spans 21.25–22.47 ms, about
three times the noise band but with no structure, and the shipped 4608 is
within noise of the best. **No change.** That is a result, not an absence of
one: the NVIDIA tuning does not transfer, and it also does not hurt.

**Apply width: 512 → 192, a 26% win.** This one is real and large.

| apply threads | 64 | 128 | 192 | 256 | 512 (CUDA) |
|---|---|---|---|---|---|
| harness, ms | 83.9 | 56.2 | **54.6** | 57.8 | 77.4 |
| production pipeline, ms | — | 369.1 | **351.5** | 374.0 | 477.9 |

A **bracketed interior minimum** on both, which is the standard `bench.h`
itself demands, with under 1% spread between repeats at each point. 192 is six
SIMD groups and keeps `(athr & 31) == 0`, which `k_apply`'s warp-ballot path
requires. Relations are unchanged — 1,845 candidates and 7 relations either
way — so this is a pure throughput change.

Shipped as the Metal build's default at both sites (`pipeline.cuh` and
`bench_kernels.cu`'s harness copy); `--apply-threads` still overrides and the
CUDA build is untouched. **The shape of the curve should carry to other Apple
GPUs; the exact optimum may not, and an M3 Max has four times the cores.
Re-measure there rather than trusting 192.**

### 8b. `--cof-chunk` costs far more on Metal than on CUDA

Measured on the parity special-q, ECM, 1,852 records, production geometry.
**Relations are 37 at every point**, so this is purely throughput.

| `--cof-chunk` | algebraic queue | wall/q | vs one launch |
|---|---|---|---|
| auto | 1508.6 ms | 2673.6 ms | 1.00x |
| 1852 (= one launch) | 1510.7 ms | 2672.7 ms | 1.00x |
| 926 | 3080.3 ms | 4383.5 ms | **2.04x / 1.64x** |
| 463 | 5044.3 ms | 6556.5 ms | **3.34x / 2.45x** |
| 256 | 7696.1 ms | 9486.5 ms | **5.10x / 3.55x** |
| 128 | refused — below one block | | |

Halving the chunk roughly doubles the cofactor stage. The CUDA-side note
records the same *shape* — "chunk at or above `blocks*threads` is free, below
it costs up to +150%" — and here every chunk below 1,852 is already below
`blocks*threads` (60 x 256 = 15,360), so all of these are in CUDA's expensive
regime. **But the penalty is far steeper on Metal: +410% at chunk 256 against
CUDA's worst case of +150%, and still climbing.**

The practical consequence: **leave `--cof-chunk` on auto.** It exists to keep a
single cofactor launch from running past a GPU watchdog, and on this hardware
buying that insurance costs several times more than it does on CUDA. Reach for
it only if a watchdog abort is actually observed, not prophylactically.

(No watchdog abort has been observed here — a single command buffer ran 10 s
without one — but that was not pushed to a limit and is not a claim that none
exists.)

### Phase 8 (continued) — original scope
`log_region <= 13` default; threadgroup sizing measured from scratch —
`bench.h`'s 32-thread `k_fill_atomic` result is an NVIDIA L2-bound finding
and must not be inherited; `--cof-chunk` retargeted at the macOS display
watchdog; slab sizing for a 16 GB UMA carveout shared with the OS.

**Gate:** documented, with every number labelled as measured on a
correctness vehicle.

### Phase 9 — Packaging
`Makefile.metal`, metallib embedding, arm64 BOINC. **All three done: 9a the
BOINC library, 9b the `HAVE_BOINC` wiring, 9c what the log actually says, 9d
embedding.** What is left is a real BOINC client -- no `init_data.xml` has
ever been in front of this binary.

### 9a. The BOINC library builds on arm64 -- with three flags the instruction did not name

**Done and working**, but the bare `./configure --disable-server
--disable-client --disable-manager` produces a library this port cannot ship.
Three additions are load-bearing, and each was found by inspecting the
artifact rather than by a build failure -- all four configurations above build
and link with exit 0.

**What was built.** BOINC master `55a5644`, `AC_INIT(BOINC, 8.3.0)`, shallow
clone. Prerequisites via Homebrew: `autoconf automake libtool pkg-config`
(the machine had none of them; `/usr/bin/libtool` is Apple's, not GNU's).

**`_autosetup` needs `LIBTOOLIZE` set.** Homebrew installs GNU libtool's
commands under a `g` prefix, and BOINC's version checker looks only for
`libtoolize`:

```
Checking version of 'libtoolize' >= 105... Didn't find application
```

Fix: `export LIBTOOLIZE=/opt/homebrew/bin/glibtoolize`. It then bootstraps
clean and configure reports exactly the intended scope:

```
--- Configuring BOINC 8.3.0 (Release) ---
--- Build Components: ( libraries) ---
```

**The configure line this port actually needs:**

```
export PATH=/opt/homebrew/bin:$PATH
export LIBTOOLIZE=/opt/homebrew/bin/glibtoolize
export MACOSX_DEPLOYMENT_TARGET=13.0
./_autosetup
./configure --disable-server --disable-client --disable-manager \
            --disable-shared --enable-static \
            --prefix=$HOME/code/boinc-install \
            CFLAGS="-O2 -mmacosx-version-min=13.0" \
            CXXFLAGS="-O2 -mmacosx-version-min=13.0"
make -j8 && make install
```

**1. `--disable-shared` is NOT optional.** Without it libtool installs
`libboinc_api.8.dylib` alongside `libboinc_api.a`, and `-lboinc_api` prefers
the dylib. The probe binary came out depending on

```
/Users/gchilders/code/boinc-install/lib/libboinc_api.8.dylib
```

-- an absolute path on the build machine, baked into a binary meant to run on
a volunteer's. It links and runs here, which is exactly why this has to be
checked with `otool -L` rather than by whether the build succeeded. With
`--disable-shared` the binary depends on `/usr/lib/libSystem.B.dylib` and
`/usr/lib/libc++.1.dylib` only, both shipped with macOS. Note only `api` and
`opencl` ever grew a dylib; `libboinc` was static either way.

**2. The deployment target must be set, or the library contradicts
`METAL_MIN_MACOS`.** A default build stamps every object with the host SDK:

| build | `minos` (all 46 objects) |
|---|---|
| default | **26.0** |
| `-mmacosx-version-min=13.0` | **13.0** |

`Makefile.metal` targets macOS 13 (M1/Apple7 floor). A `minos 26.0` archive
linked into it produces a binary that will not start on anything older than
the build host -- the support floor silently becoming "whatever this laptop
runs". Verified across every object in both archives, not just the first.

**3. `BOINC_HOST_STATIC`'s default is a hard error with Apple clang.**
`Makefile` line 348 defaults it to `-static-libstdc++ -static-libgcc`:

| flag | Apple clang |
|---|---|
| `-static-libstdc++` | `warning: argument unused during compilation` |
| `-static-libgcc` | **`error: unsupported option '-static-libgcc'`** |

So any macOS BOINC link must pass `BOINC_HOST_STATIC=` explicitly. The
Makefile already documents that opt-out ("set `BOINC_HOST_STATIC=` to opt out
explicitly") -- on macOS it is mandatory, not a choice. It also costs nothing:
the reason the CUDA build does this is glibc/libstdc++ skew across Linux
distributions, and macOS ships libc++ and libSystem with the OS.

**What was verified, beyond "it compiled".** `boinc_support.cpp` compiles
against the real BOINC 8.3.0 headers with `-Wall -Wextra` and **zero
warnings**, and `metal/boinc_link_probe.cpp` links the whole
`bench_boinc_*` surface against the static archives. The probe references
`bench_boinc_init` behind an `argc` gate rather than `if (0)`, because at -O2
the optimiser deletes a dead call and then `boinc_api.o` is never pulled from
the archive -- a link that resolved nothing and looked identical. With the
gate, `nm` confirms `_boinc_init_parallel`, `_boinc_finish` and
`_boinc_fraction_done` are all in the binary.

Run with an argument, the round trip works:

```
BOINC: API initialised (standalone mode)
init rc=0
```

and then **exits without returning** -- `bench_boinc_finish` calls
`boinc_finish`, which terminates the process. It also leaves
`boinc_finish_called` and `stderr.txt` in the working directory: **the BOINC
runtime redirects stderr to `stderr.txt` in cwd**, which Phase 9 has to
reconcile with `runlog`'s own stderr half (and with `g_runlog_quiet`).

Without an argument it reports `is_managed=0 gpu_device=-1` and refuses
`bench_boinc_resolve_path` before init, which is the correct unmanaged
behaviour.

**What 9a does NOT establish.** The probe is not a BOINC client: nothing here
ran under a real `init_data.xml`, so slot-directory filename resolution,
`boinc_get_init_data`'s GPU device assignment, checkpointing and the
suspend/quit messages are all unexercised. It also says nothing about
`Makefile.metal`, which at this point had no `HAVE_BOINC` path at all -- only
`Makefile` did. Wiring that, with `BOINC_HOST_STATIC=`, is what 9a unblocks
and what 9b does.

### 9b. `HAVE_BOINC` wired into `Makefile.metal` -- and a marker that had rotted

`make -f Makefile.metal benchbin HAVE_BOINC=1 BOINC_DIR=<prefix>` now produces
a BOINC application binary. Default stays 0, so the fraction-done path is
still compiled out of every ordinary build. The ported TUs already carried the
whole integration (`bench_main_metal.cpp` has 26 `bench_boinc_*` call sites,
`pipeline_host.inc` five) -- only the build wiring was missing. Neither needs
a BOINC header: everything goes through `bench.h`, and `boinc_support.cpp` is
the one translation unit that includes `boinc_api.h`.

**The marker in `--help` had rotted, and it is a machine-crashing rot.**
`gen_bench_main.py` rewrote CUDA's `--device` help line to say "select Metal
device"... in the `#else` branch only. A `-DHAVE_BOINC` build takes the
`#ifdef` branch, which still said **"select CUDA device"**. `cofcheck.sh`
classifies the build from exactly that string, so a BOINC build would be
detected as CUDA -- and then run the `--ecm-b1 400000` case that took
WindowServer down twice (8k). A marker that holds in one branch of the
`#ifdef` it is printed from is not a marker. Both branches now say Metal, and
the gate below asserts it on the binary.

Two stderr lines under the same `#ifdef` also named the wrong API -- `BOINC:
running on CUDA device %d of %d` and `using CUDA's default device`, out of a
Metal binary, into the log a volunteer would send to a project. Both fixed in
the generator, with the comment that explained the field in terms of "an
NVIDIA coprocessor".

**A flag stamp, which this Makefile never had.** Flipping `HAVE_BOINC` left
`$(BUILD)` full of objects compiled the other way, and a `bench_main.o` built
without the define simply never calls `bench_boinc_init()` -- a silently wrong
binary, not a link error. `.metalflags.stamp` carries the whole
`HOSTFLAGS|MSLFLAGS|BOINC_LINK|CPUOBJ_TUNE` signature and every Metal-side
object and the metallib depend on it, so changing **any** tunable rebuilds
what it affects. That also retires the stale-artifact trap 8q records costing
a `TD_TILE` measurement. Verified: a no-op rebuild stays a no-op, and
flipping `HAVE_BOINC` either way recompiles.

`HOSTFLAGS` is split into `HOSTFLAGS_BASE` plus the BOINC flags for one
reason: include search is left to right, so with `BOINC_CPPFLAGS` in
`HOSTFLAGS` the stub gate's later `-I metal/boinc_stub` would **lose to the
real SDK** and `boinccheck` would quietly stop testing the stub. And all four
delegations to the default Makefile now pass one `CPUOBJ_MAKEVARS`, including
**`BOINC_HOST_STATIC=`** (9a: `-static-libgcc` is an error with Apple clang),
because that Makefile folds these into a stamp of its own and passing them
inconsistently would have the gates rebuilding each other's CPU objects.

#### The gate: `make -f Makefile.metal boinclinkcheck HAVE_BOINC=1 BOINC_DIR=...`

Every check in it is for something that **builds and links with exit 0 and is
still wrong**. It refuses to run at `HAVE_BOINC=0` rather than vacuously pass.

| check | catches |
|---|---|
| no BOINC dylib in `otool -L` | BOINC built without `--disable-shared` |
| every dependency under `/usr/lib` or `/System` | any build-host path at all |
| `minos` equals `METAL_MIN_MACOS` | BOINC built without `-mmacosx-version-min` |
| `_boinc_init_parallel`, `_boinc_finish`, `_boinc_fraction_done` present | a link that resolved nothing |
| `--help` still says "select Metal device" | the rot above, i.e. `cofcheck.sh` running the machine-crashing case |

All seven pass. **The control fails**: run against the `HAVE_BOINC=0` binary
it reports the three symbols missing and exits 1, so the symbol checks are not
tautologies.

**What the gate cannot check, and says so.** Under `HAVE_BOINC` the runtime
redirects stderr to **`stderr.txt` in the working directory** as soon as
`boinc_init` runs -- which is before argument parsing -- so nothing written to
stderr reaches a terminal or a `2>&1` pipe. **A check that greps stderr from a
pipe finds nothing and "passes" for the wrong reason.** The gate prints the
list of stderr-only assertions to read by hand instead of faking them.
(`--help` itself is `printf`, i.e. stdout, which is why the marker check and
`cofcheck.sh`'s detection are sound.)

**Done by hand, once, and it is the real end-to-end proof.** A `HAVE_BOINC=1`
binary, one q at the parity special-q, in a clean directory:

```
exit 0, total relations 37            <- the golden number, from a BOINC build
stdout.log: 1 mention of "BOINC"      <- everything else went to the file
```

Both rewritten lines name Metal. What `stderr.txt` contained is 9c, which is
where reading it turned into work. **Still not a client**: standalone mode, no
`init_data.xml`, so slot filename resolution, a real GPU assignment and
checkpointing remain unexercised.

### 9c. What was actually in `stderr.txt`, and cutting it to six lines

Reading the file rather than assuming its contents is the whole point of 9b's
rule, and it found 14 lines of which most were noise or wrong. A BOINC
volunteer uploads this file; a project reads it when a task fails.

**Before:**

```
 1  ...: Can't open init data file - running in standalone mode
 2  BOINC: no usable GPU assignment in init_data.xml; using the system default Metal device
 3  BOINC: running on Metal device 0 of 1: Apple M3
 4  note: side 1 allowance 101.60 is 8.03 bits looser than the derived 93.57; the surplus
 5        admits survivors the cofactoriser then rejects (mfb 92).
 6  note: side 0 allowance 68.10 is 6.60 bits looser than the derived 61.50; the surplus
 7        admits survivors the cofactoriser then rejects (mfb 60).
 8  BOINC: slab plan: 2048 rows/slab, 8 slabs (auto-calibrated)
 9    cofactor queue: 12 curves/round is about 119 ms in one launch, over this build's 750 ms bound. ...
10    Shorter launches, same B1/B2: --ecm-curves 2 with more --cof-rounds (ECM allows 1000).
11    cofactor chunk: 1852 records/launch, 1 launch per round over 1852 records (auto)
12    cofactor: longest kernel launch 93 ms (1852 records/launch, 1852 in flush)
13    cofactor chunk: 1852 records/launch, 1 launch per round over 1852 records (auto)
14  ...: called boinc_finish(0)
```

**After** -- same run, same 37 relations, exit 0:

```
 1  ...: Can't open init data file - running in standalone mode
 2  BOINC: no usable GPU assignment in init_data.xml; using the system default Metal device
 3  BOINC: running on Metal device 0 of 1: Apple M3
 4  BOINC: slab plan: 2048 rows/slab, 8 slabs (auto-calibrated)
 5    cofactor chunk: 1852 records/launch, 1 launch per round over 1852 records (auto)
 6  ...: called boinc_finish(0)
```

**Line 9 was false, and its guard explains why.** It said 119 ms was "over
this build's 750 ms bound". The branch is gated on

```c
if (ms_one_curve <= COF_LAUNCH_REFUSE_MS && Q->ecm_curves > 2u)
```

-- curves-per-round above two, with **no test of the launch time against the
bound at all**. The behaviour is right and 8n says so explicitly ("aim at 2
and let the bound lower it further"); the *message* justified it with a bound
violation that had not happened, and at B1 200 a curve is 9.9 ms, so twelve of
them are nowhere near 750. The surviving line in the acting branch now states
what it did (`%u curves/round -> %u x %u (~%.0f ms/launch, ...)`) and mentions
no bound. The advisory branch is gone entirely: when the derivation cannot
act -- an explicit `--ecm-curves`, or a side running rho, which is the
default -- it is silent rather than printing guidance into a volunteer's log.

**Lines 11 and 13 are the same line twice, and `cof_report_chunk`'s own
comment forbids it** ("Only on change, never per flush"). Its arithmetic
defeated it: `step` is `min(chunk, n)`, so when the old and new chunk both
exceed `n`, the internal value changes and the rendered line does not. Here
the opening choice printed it, then a 93 ms flush triggered the doubling
branch and printed it again, identically. It now compares **what it is about
to print**, not the internal slice.

**Lines 4-7 and 12 are terminal diagnostics in the wrong place.** The
allowance notes advise changing a parameter that arrived in the job file the
project sent, and this build's own parity settings fire both of them --
dropped Metal-side.

Line 12 came back, **conditioned**. A per-band high-water line is, in a
healthy run, a stream of messages saying nothing is wrong. A launch *over the
bound* is the opposite: it is the condition 8k set 750 ms for, the thing a
stalled compositor or a killed task would be explained by, and it is invisible
from anywhere else. So it reports only over-bound launches, still gated on a
new maximum so a device that cannot meet the bound emits a few lines and then
goes quiet as the chunker parks -- and **not** gated on auto mode, since a
pinned `--cof-chunk` that overruns is more worth knowing about, not less,
because nothing will adapt.

Demonstrated both ways on the same q. Silent in bounds (the six-line file
above, `--ecm-b1 2000 --ecm-curves 16`, a 401 ms launch: nothing). At
`--ecm-curves 48`:

```
  cofactor: kernel launch 1508 ms is over this build's 750 ms bound (1852 records/launch, 1852 in flush)
```

against a reported `algebraic queue 1507.68 ms` -- the same launch, from the
other side. 37 relations either way.

**Every removal is Metal-side, in the generators.** The strings live in
`bench_main.cu` and `cofac.cuh`; editing those would be a CUDA-side
**behaviour** change, not the inert kind the drift ledger has rows for, and it
would reach HIP too. `gen_bench_main.py` and `gen_cofac_host.py` do the
removal instead, so the CUDA build's stderr is untouched and the two builds
now deliberately differ in what they log. No gate reads any of these strings.

#### A generator bug this uncovered: an unasserted `replace` is a silent no-op

Dropping the `ms_launch_max` field broke an anchor two hundred lines away:

```python
_fld_old = "    float  ms_launch_max;   /* longest single launch seen, ms */"
src = src.replace(_fld_old, _fld_old + ... "uint32_t ecm_rounds; ...", 1)
```

No assert. The anchor was gone, the replace matched nothing, and the generator
**printed success** while emitting a `cofq_t` with `Q->ecm_rounds` assigned in
two places and declared in none. The compiler caught it here, but only because
the missing thing was a field; a missing *statement* would have compiled.
Re-anchored, with an assert. Audited the rest: the only other unasserted
replaces are the bulk `cuda*->mtl*` and `LAUNCH_APPLY` tables, where matching
nothing is legitimate, and the one at `gen_cofac_host.py:402` carries a
stronger `assert src.count(...) == 1`. **Every single-site anchor in these
generators must be asserted** -- it is the same failure as the regex that
deleted half of `bigint.cuh`.

Gates: `cofcheck.sh` 54 PASS / 0 FAIL, `cofaccheck` 37 in all four
configurations, `boinccheck`, `boinclinkcheck` 7/7.

**One more anchor rule out of this.** Restoring `ms_launch_max` hung it off
`cofac.cuh`'s own `double ms_rat, ms_alg, ms_host;` -- a declaration this
generator does not create and cannot delete -- rather than off the chunker
block it adds. 9c's bug was exactly an anchor pointing at a line the generator
had introduced and later removed. **Anchor on the upstream source, never on
this generator's own output.**
### 9d. Metallib embedding: the application is one file

`bench` carried its shaders in a separate `bench.metallib` that had to sit
beside it, be found by `$CUDA_SIEVE_METALLIB`, or be passed by path. For a
BOINC project that is a second file to ship, land in the slot, and keep in
step with the executable -- missing, stale and mismatched are three failure
modes that stop existing once the shaders are inside.

**The mechanism is one linker flag and one lookup.** `ld` writes the library
into a Mach-O section:

```
-sectcreate __DATA __metallib $(BUILD)/bench.metallib
```

and `metal_rt` reads it back with `getsectiondata(&_mh_execute_header,
"__DATA", "__metallib", &n)`, wrapping the bytes in a `dispatch_data_t` with
an **empty destructor block** -- they are in our own `__DATA` and live as long
as the process, so there is nothing to free and
`DISPATCH_DATA_DESTRUCTOR_DEFAULT` would copy the whole megabyte. No generated
byte array, no extra compile step.

**Four places, and the order is deliberate:**

| | source | why here |
|---|---|---|
| 1 | the explicit `mtlInit` path | a caller that names a library means it |
| 2 | `$CUDA_SIEVE_METALLIB` | **every gate in this Makefile drives the library it just built through this** |
| 3 | **embedded `__DATA,__metallib`** | a binary carrying its own shaders should trust itself |
| 4 | `bench.metallib` beside the executable | the old behaviour, unchanged |

3 before 4 because a stale `bench.metallib` in the working directory is
exactly what a BOINC slot accumulates, and the copy inside the binary is the
one that matches it by construction.

**Cost:** 718 KB -> **1,726,104 bytes**, the metallib being 987,730 of them.
`EMBED_METALLIB=0` opts out and is what the gate's control uses.

#### The gate: `make -f Makefile.metal metallibcheck`

**Control first, in its own build.** It builds with `EMBED_METALLIB=0`,
asserts the section is absent, and asserts the binary **fails**:

```
PASS   control: no __metallib section, as intended
PASS   control fails with no library to find (exit 1)
       metal_rt: cannot load shader library '/…/tmp.o51ok885qJ/bench.metallib':
       Error Domain=MTLLibraryErrorDomain Code=6 "library not found"
```

Then the real case:

```
PASS   __DATA,__metallib present (987730 bytes)
PASS   sieved from the embedded library alone: 37 relations at the parity q
```

**Both runs happen in a temporary directory with no `bench.metallib` in it and
with `CUDA_SIEVE_METALLIB` unset (`env -u`).** That negative space is the
whole check: every other gate here *exports* that variable, so **not one of
them would notice embedding being broken**. `boinclinkcheck` now asserts the
section too -- a distributable binary that needs a file beside it is not
distributable.

**End to end, and this is what the phase was for.** One 1.7 MB file alone in a
directory, `HAVE_BOINC=1`, no environment variable:

```
$ ls -l
-rwxr-xr-x  1 gchilders  wheel  1726104  bench
$ env -u CUDA_SIEVE_METALLIB ./bench --pipeline …
exit 0    total relations 37
```

with the same six-line `stderr.txt` from 9c.

**One `-A2` that should have been `-A4`.** `otool -l` prints `sectname`,
`segname`, `addr`, *then* `size`, so the gate's size probe found nothing and
reported "nothing was embedded" about a binary that had, on the very next
line, sieved 37 relations out of its own embedded library. **The two checks
disagreeing is what caught it** -- a gate with only the section probe would
have failed a working build, and one with only the run would have passed a
binary whose section was some other size than intended.

**Not embedded: `fbgen_gpu`.** It is a development and project-side tool, not
the distributed application, and it takes `$CUDA_SIEVE_METALLIB` from the
gates exactly as before.

Gates after the `metal_rt.mm` change -- it is linked by all of them:
`rtcheck`, `scancheck`, `fbcheck`, `sievecheck`, `argbufcheck`,
`classifycheck`, `cofaccheck`, `cofcheck.sh` 54 PASS / 0 FAIL,
`boinccheck`, `boinclinkcheck` 8/8, `metallibcheck` with its control.
### 9e. The packaged binary validated over 288 special-q

The packaged build differs from the stock one in three ways that all touch the
executable: BOINC linked in, the shaders moved inside it, and the CPU objects
rebuilt under `-DHAVE_BOINC`. None of that should move a relation. Measured
rather than assumed, over the same 288-q band Phase 7 used:

| | stock | packaged |
|---|---|---|
| build | `HAVE_BOINC=0` | `HAVE_BOINC=1 BOINC_DIR=…` |
| size | 1,462,664 B | **1,726,104 B** |
| `_boinc_*` symbols | 0 | **46** |
| `__DATA,__metallib` | 987,730 B | 987,730 B |
| **relations** | **13,485** | **13,485** |
| records enqueued | 564,696 | 564,696 |
| `sha256(relations)` | `8e79762c…dafbc002` | `8e79762c…dafbc002` |

**`cmp` clean, byte for byte**, and 13,485 / 564,696 are exactly Phase 7's
numbers against real CUDA on a GTX 1080 Ti. The packaged binary ran **in a
directory containing nothing but itself**, with `CUDA_SIEVE_METALLIB` unset.

Wall: 258 s stock, 280 s packaged. The ~8% is the BOINC build's own work plus
this fanless box between two back-to-back 4-minute runs; it is not a
measurement of packaging overhead and should not be quoted as one.

**A false start worth recording.** The first pair of runs omitted `--cofactor`
and produced 2,381 relations -- also byte-identical between the two builds,
and also useless as a validation, because without the cofactoriser the band
emits only the relations that need no splitting. **A comparison that agrees
can still be measuring almost nothing.** The tell was in the output the whole
time: `records enqueued 564696 (of which 2381 needed no splitting)`.

#### What still has to ship beside it: nothing, and the job data

`otool -L` on the packaged binary lists seven libraries and **every one is
part of macOS**: Metal, Foundation, IOKit and CoreFoundation from
`/System/Library/Frameworks`, and `libSystem.B.dylib`, `libc++.1.dylib`,
`libobjc.A.dylib` from `/usr/lib`. There is one `dlopen` in the tree,
`libnvidia-ml.so.1` for optional NVML telemetry; it cannot exist on macOS,
`runlog_gpu_bind` returns -1, and the caller carries on without it. So the
**application** is one file.

The **job** is not -- but it is much smaller than it looks. `--poly` is 462 B,
and **the 110 MB factor base does not have to be shipped at all**: with no
`--fb1`/`--cadofb`, pipeline mode generates the complete algebraic factor base
on the GPU at startup.

**Verified over the same 288 q, against the same reference.** A directory
containing nothing but the 1.7 MB executable, run with only `--poly`:

```
bench: no --fb1 supplied; generating algebraic factor base on GPU through 134200000
afb_build_gpu: 7605616 ideals through 134200000 (207 prime-power);
               7605407 ordinary GPU roots, 38 exact primes; 6.502 s wall
```

| | relations | sha256 |
|---|---|---|
| `--cadofb c183.fb1` (110 MB on disk) | 13,485 | `8e79762c…dafbc002` |
| **no factor base anywhere** | **13,485** | **`8e79762c…dafbc002`** |

`cmp` clean. 7,605,616 ideals is the canonical `c183.fb1`'s own entry count,
and the 6.5 s is once per process.

**The one thing to get right per job is `alim`.** The generator truncates at
the job's alim (`genlim = min(fbbound, alim)`), and `alim` comes from a GGNFS
`.job`, from `--alim`, or -- failing both -- from a **compiled-in default of
134,200,000** (`bench_main_metal.cpp:989`). A CADO `.poly` carries no alim, so
the run above took that default, which for c183 *is* the production alim. That
is a coincidence of this job, **not a derivation from the polynomial**: for
any other composite, pass `--alim` or a `.job` that carries it, or the factor
base will silently be the wrong size.

So a workunit needs the poly and the parameters, not the factor base. Files
that *are* sent still resolve through `bench_boinc_resolve_path`.

**Three things a project needs to know, none of them a missing file.**

- **arm64 only, non-fat.** An Intel Mac cannot run this at all -- Rosetta
  translates x86_64 to arm64, not the reverse -- and the port refuses
  non-Apple GPUs by name anyway (`MTLGPUFamilyApple7`). The app version's
  platform must say arm64.
- **`minos 13.0`**, so macOS 13 or later. That excludes no Apple silicon Mac
  in practice.
- **The signature is ad-hoc** (`flags=0x20002(adhoc,linker-signed)`) -- what
  ld puts there, not a Developer ID. **Whether BOINC distribution on macOS
  needs a real signature or notarization is untested here** and is a question
  for the project, not a claim this port can make either way.
### 9f. Signed with the project key -- and a CRLF bug in BOINC's key parser

The distributable binary is in **`~/code/dist/`**, outside this repository:

```
bench        1,726,104 B   sha256 25ab6b6550d44b0f…2fdac977
bench.sig          262 B   256 hex chars + "." -- a 1024-bit RSA signature
```

`crypt_prog` is not built by `--disable-server`: `lib/Makefile.am` puts it
under `if ENABLE_SERVER`, and so does `libboinc_crypt`. Rather than reconfigure
the tree that produced the validated library, it was compiled by hand against
it, which is also the whole recipe:

```
brew install openssl@3
clang++ -O2 -std=c++17 -I . -I lib -I api -I $SSL/include \
    lib/crypt_prog.cpp lib/crypt.cpp -L $SSL/lib -lcrypto lib/libboinc.a \
    -o crypt_prog
```

`SSL_LIBS` is empty in the generated Makefile for the same reason, so OpenSSL
has to come from outside; `crypt_prog.cpp` needs `<openssl/encoder.h>`, i.e.
**OpenSSL 3.x**.

#### The key would not parse, and the diagnosis never opened the file

`crypt_prog -sign` failed with `Error: scan_private_key_hex`. The key must not
be read or copied, which rules out looking at it -- so the diagnosis was done
entirely on **file size**, against keys generated locally:

| | bytes |
|---|---|
| fresh BOINC 1024-bit private key (LF) | 1437 |
| newlines in it | **24** |
| the project key | **1461** |
| 1461 - 1437 | **24 -- exactly one extra byte per line** |

One extra byte per line is a CR. Converting the throwaway key to CRLF made it
**1461 bytes**, byte-for-byte the same size, and reproduced the identical
`Error: scan_private_key_hex`. Hypothesis confirmed without opening the file.

**It is a real bug in `lib/crypt.cpp`, in two places.** `sscan_key_hex` reads
the leading bit-count line and requires every character before `'\n'` to be a
digit, so it rejects the `'\r'` outright; and `sscan_hex_data` skips `'\n'`
but *breaks* on anything that is not a hex digit, so a `'\r'` would truncate
the key even if the first line were accepted. Both now skip CR. **Patched in
the local BOINC tree, not here** -- `~/code/boinc/lib/crypt.cpp` -- and the
original is at `/tmp/crypt.cpp.orig`. Anyone reproducing this needs the same
two-line change, or a key with Unix line endings.

**The tool was proven before the key was blamed.** A throwaway
`-genkey 1024` pair signs and verifies (`signature is valid`); after the patch
the same key in LF and CRLF form both verify and produce an **identical**
signature, which is what "same key, different line endings" should mean.

#### Verified against the project's public key, with a negative control

```
$ crypt_prog -verify bench bench.sig code_sign_public
signature is valid                                   exit 0
```

**And the control fails.** One bit flipped at byte 863,052 of a copy
(`0x0f -> 0x0e`), same signature:

```
signature is invalid                                 exit 1
```

so the verification is answering about *these* bytes and not merely about the
file's existence.

| | sha256 |
|---|---|
| `bench` | `1e85fd5e4308d7bde19380e4a42a4bb0a53d069e2d1146ff5895afd7858079eb` |
| `bench.sig` | (re-made; verified with the one-bit-flip control) |

**Re-staged and re-signed 2026-09-16 after 9z's leak fix.** Superseded artifacts, neither of
which is the one to ship: `25ab6b65…2fdac977` (pre-leak-fix) and
`9e975aa5…27422cb0` (pre-autorelease-fix). The binary grew
80 bytes (1,726,104 -> 1,726,184). The signature was re-made from scratch --
a signature is over content, so the old one does not verify the new binary
(and the negative control above is exactly that check).

The public key is the project's 1024-bit `code_sign_public` and is not secret;
it was supplied for this check and is not stored in the repository. Re-signing
is also byte-identical (deterministic PKCS#1 v1.5).

**The key was never read or copied.** It appears exactly once, as an argv to
`crypt_prog`, and nothing in `~/code/dist` refers to it.

**This is BOINC's file signature, not a macOS one.** It is what the client
checks against the project's public key on download; it says nothing about
Gatekeeper, whose requirements for a BOINC-distributed macOS application
remain untested (9e).
## 9z. Code review of the port (2026-09-16)

Review of the ~7,000 hand-written lines; the ~19,000 generated ones are
reviewed through their generators. Every claim below is a measurement or a
control, not a reading.

### Found and fixed: `mtlFree` leaked the entire allocation

**`metal_rt.mm` is compiled WITHOUT `-fobjc-arc`** (only `sf_test_device.mm`
gets it). So the `id<MTLBuffer>` inside `Alloc` is an unmanaged pointer,
`newBufferWithLength:` returns +1, and the registry entry is the only owner.
`reg_erase` did `g_allocs.erase(it)` and nothing else -- **the buffer was never
released.**

Measured with `mtlMemGetInfo`, which reads `currentAllocatedSize`:

| 64 MB malloc+free, repeated | before | after |
|---|---|---|
| after #1 | 64.5 MB used | 0.5 MB |
| after #4 | 256.5 MB used | 0.5 MB |
| after #8 | **512.5 MB used** | **0.5 MB** |

**It reached the production path.** The pipeline's own free-memory reports
fall monotonically through a run, because slab auto-calibration (8g) builds
and tears down the bucket array and both factor bases **three times** before
the real band:

| end of a 24-q band | free memory |
|---|---|
| before the fix | 11.84 -> **8.37 GB** |
| after | 11.84 -> **10.62 GB** |

**2.25 GB recovered**, relations unchanged (1,114), twelve gates green. It is
*bounded* -- 21 allocation reports whether the band is 24 q or 288 q, so it
does not grow per special-q -- but 3.5 GB of dead allocations on an 8 GB M1
is the difference between running and not.

The same ownership bug was in `mtlShutdown` (`g_allocs.clear()`,
`g_psos.clear()`) and in `pso_for`, which leaked an `MTLFunction` per kernel
name. All three fixed.

### Verified sound, each with a control

**A missing kernel fails the run closed.** This was worth testing because the
port reaches 84 kernels **by string name** and templated ones by mangled
`[[host_name]]`, so a missing instantiation is a plausible silent no-op.
Sabotaged a production launch (`k_group_counts` -> `k_group_countsZZ`), rebuilt
and ran:

```
EXIT=255,  no relations emitted
metal_rt: no kernel named 'k_group_countsZZ' in the shader library
CUDA mtlGetLastError(): kernel not found in shader library at metal/pipeline_host.inc:1246
```

`mtl_launch_begin` refuses, and the ported code's `mtlGetLastError()` checks --
CUDA's own idiom, kept -- carry it to an abort naming file and line. **An
earlier note in this plan said a missing kernel "looks like success"; that was
about probe code written for 8q, not about the port.**

**Zero drift between the generators and the committed generated sources.** All
twelve `gen_*.py` run and reproduce their output **byte for byte** -- `git
status` after regenerating everything showed only the hand-edited
`metal_rt.mm`. For a port whose correctness argument rests on "the generated
files are what the generators produce", this is the load-bearing check.

**Generator anchor discipline is now complete**: 42 asserted anchors, **0**
unasserted single-site `src.replace`. The `src.index()` cuts need no assert --
`str.index` raises. This is the discipline 9c's silent no-op established.

**No lock recursion.** `g_lock` is a plain `std::mutex` held from
`mtl_launch_begin` through `mtl_launch_end`. `mtlUseResource` deliberately
does **not** take it (it runs inside that window); `mtlDeviceAddress` does, and
is only ever called outside it.

### Open, not fixed

**1. FIXED: no `@autoreleasepool` anywhere in the shim, and the leak was
holding it together.** See 9z-b below.

**2. A zero-sized launch is silently skipped.** `mtl_launch_end` dispatches
only `if (g_bind_grid && g_bind_block)`; CUDA returns
`cudaErrorInvalidConfiguration` for a zero grid dimension. Ported code must
already guard `n == 0` to have worked on CUDA, so this is benign today -- but
it is a place where a future ported bug would be caught on CUDA and masked
here.

**3. `mtlDeviceAddress` has no "do not call while binding" warning**, unlike
`mtlUseResource` which has the matching one. Calling it between
`mtl_launch_begin` and `mtl_launch_end` would deadlock on the non-recursive
mutex. Nothing does; it is one comment away from being safe by documentation
rather than by luck.

**4. `pso_for` caches `nil`**, so the "no kernel named" diagnostic repeats once
per launch attempt (three times in the control above). Harmless, slightly
noisy.

### 9z-b. The autorelease fix: ownership first, then the pool

`commit()` stored `st->last = st->cb` where `cb` came from `[st->q
commandBuffer]` -- **autoreleased and never retained** -- and `sync()`, plus
every event query, read it afterwards. The same held for the two encoders and
for `mtlEventOpaque::cb`. With no pool anywhere in this process (a plain C++
`main()`, no Cocoa run loop) nothing ever drained, so those objects survived by
accident. **The leak was standing in for a lifetime, which is why a pool alone
would have been a use-after-free rather than a fix.**

**So the ownership came first.** `cb`, `cenc`, `benc`, `last`, the stream's
`MTLCommandQueue` and the event's `MTLEvent` are now retained on store and
released on replace or teardown, with `stream_teardown()` for the two places
that used to drop a queue on the floor (`mtlStreamDestroy`, `mtlShutdown`).
`mtlEventRecordOn` takes its **own** reference, retain-before-release, because
the event outlives the stream's next commit.

**Then the pool, at the creation site.** Balancing our own retain with a
release is *not* sufficient: the pending autorelease also has to fire, and
without a pool it never does -- the object is immortal however carefully we
balance our own reference. So the three factory calls create inside a pool and
retain before it drains:

```objc
void ensure_cb(Stream *st)
{
    if (st->cb) return;
    @autoreleasepool { st->cb = [[st->q commandBuffer] retain]; }
}
```

Local, and the boundary sits on the thing it governs. A pool per entry point
would also have been correct -- but only *after* the ownership work, and it
would have put the boundary far from the object.

**Measured on the same 144-q band:**

| | RSS start | RSS at +75 s | rate |
|---|---|---|---|
| before | 372.0 MB | **399.0 MB** | ~0.36 MB/s |
| after | 361.9 MB | **367.4 MB** | **~0.07 MB/s** |

**About 80% of the growth gone**, and what remains steps and then flattens
(361.9, 361.9, 363.5, 367.3, 367.3, 367.4) rather than rising -- the shape of
the cross-q relation queue filling toward a flush, which is work, not a leak.
That also answers 9z's open question about which of the two causes it was:
mostly the autoreleased objects.

**It is not a small class of object.** A command buffer retains every resource
it references, so each leaked one pinned its share of the sieve's buffers too,
and the port commits one per event record.

**Validated as a lifetime change deserves**, because a premature release is a
wrong answer rather than a leak: twelve gates green, `cofcheck.sh` 54 PASS /
0 FAIL, and the 288-q band **byte-identical** at 13,485 relations,
`sha256 8e79762c…dafbc002`, `cmp` clean.

### 9z-c. Second review: the Metal validation layer cannot run on this port

The first review's leak came from a lifetime assumption nobody had tested, so
the second pass led with the instruments rather than with reading.

#### My own new code, cleared

The retain/release work in 9z-b is where a fresh bug would be most damaging, so
it was checked dynamically, not by inspection:

| instrument | result |
|---|---|
| `OBJC_DEBUG_MISSING_POOLS=YES` | **0** "autoreleased with no pool" |
| `NSZombieEnabled=YES` | **0** messages to a deallocated object |
| static audit of all 17 ownership sites | balanced |

The first confirms the pools actually cover every autorelease; the second that
nothing is over-released. Both on a real 4-q band, relations correct.

#### THE FINDING: `MTL_DEBUG_LAYER=1` aborts, and the cause is systemic

Turning on Metal API validation kills the sieve with SIGABRT. The stack names
it exactly:

```
-[MTLDebugComputeCommandEncoder validateComputeFunctionArgumentsCommon]
  <- dispatchThreadgroups:threadsPerThreadgroup:
  <- mtl_launch_end
  <- scan_rec()            [3 recursive frames]
  <- mtlSelectFlaggedU32
  <- afb_build_gpu
```

**The port binds `nil` for optional buffer arguments, and Metal requires every
declared buffer argument to be bound.** `scan.metal` declares

```metal
kernel void k_scan_block(device uint *out  [[buffer(0)]],
                         device uint *bsum [[buffer(1)]], ...)
```

and `metal_scan.cpp`'s `scan_rec` passes `(uint32_t *)nullptr` for `bsum` at
the deepest recursion level, where there is no next level to accumulate into.
The kernel guards with `if (lid == SCAN_BLK - 1 && bsum)`, so it never
dereferences -- which is why it has always worked.

**It is not one call site.** Twenty launches pass `nullptr`: 14 in
`bench_host.cpp`, 6 in `pipeline_host.inc` -- including the **production**
`k_apply` launch, which passes three (`dump`, `dbg_cells`, `probe_out`) -- and
1 in `metal_scan.cpp`. The idiom is deliberate and documented: `metal_rt.h`'s
`mtl_bind_one(std::nullptr_t)` overload exists precisely to route these to
`setBuffer:nil`, and CLAUDE.md calls that overload load-bearing (it is -- it
is what stops `NULL` being bound as eight bytes of integer zero).

**What it costs.** Not correctness today: Apple's driver tolerates the nil
binding and every such kernel guards the pointer. What it costs is **the
single best tool for catching binding bugs, in the port most exposed to
them** -- 84 kernels reached by *string name*, arguments bound *positionally*
by index, and templated variants reached through mangled `[[host_name]]`
strings. The validation layer is exactly what would catch a wrong index or a
missing argument, and it cannot be switched on.

**A partial fix is worthless**: fixing `scan_rec` alone would move the abort to
the next `nullptr` launch. It is all-or-nothing, and the options are
- **templated variants** per optional argument (`k_scan_block<HAS_BSUM>`), which
  fits how this port already specialises kernels, but multiplies instantiations
  where a kernel has three optional pointers (`k_apply` -> 8); or
- a **sentinel buffer** plus an explicit present-flag argument, which changes
  every affected kernel signature but not their count.

**Deliberately NOT fixed in this pass.** It is an architectural change across
~20 call sites and their kernel signatures, in a port that is validated,
signed and staged; and the defect is invisible to every gate because the gates
do not run under validation. It belongs in its own change, with the validation
layer as its acceptance test -- which is the point of doing it.

**Recorded as a hazard, not a bug:** a future driver, OS, or Apple GPU
generation may stop tolerating it, and the failure mode would be a kernel
reading address zero rather than a clean refusal.

#### Reviewed with no findings

`metal/slab_calib.inc` (184 hand-written lines, production path, ported from
the HIP port -- the source of the BOINC progress bug). It is careful:
`g_runlog_quiet` and `bench_boinc_progress_suspend` are both reset
unconditionally after the throwaway pass, the dispatch matches `run_pipeline`'s
own `cplan.enabled` branch rather than forcing `<true>`, and `cplan.jmax` --
the post-clamp value -- is what gets reported and reused.

Carried, minor: `mtlEventSynchronize` reads `e->cb` under the lock and then
uses it after unlocking. Safe only because events are single-threaded here.

### 9z-e. The nil bindings are gone: the sieve runs under Metal validation

```
$ make -f Makefile.metal validationcheck
== the pipeline under MTL_DEBUG_LAYER=1 ==
  total relations                          37
METAL VALIDATION GATE: PASS
```

Every optional buffer now goes through the 9z-d mechanism. The work was
**driven by the validation layer itself**, which names one kernel and one
buffer index per run, so the loop was: run, read the name, convert, rebuild.
That turned an all-or-nothing refactor into six visible steps.

| kernel | optional buffer indices | generator |
|---|---|---|
| `k_scan_block` | 1 | hand-written `scan.metal` |
| `k_apply` | 11, 13, 24, **25** | `gen_bench_kernels.py` |
| `k_intersect_compact` | 9, 10, 11, 16 | `gen_bench_kernels.py` |
| `k_fill_atomic` | 10, 11 | `gen_bench_kernels.py` |
| `k_transform` | 12 | `gen_bench_kernels.py` |
| `k_resieve_scatter` | 2, 14 | `gen_td_metal.py` |
| `k_td` | 3, 16, 17, 18 | `gen_td_metal.py` |
| `k_cofac` | 10 | `gen_cofac_metal.py` |

**Two of those were invisible to the static audit**, which is why the audit was
not enough on its own:
- `k_apply` **25** (`survbits`): the pipeline passes a real buffer here and
  only `phase5_test` passes null, so scanning the production launcher missed it.
- `k_cofac` **10** (`iters`): passed as a **variable that is null at runtime**,
  not a literal `nullptr`. No grep can see that. It is caught because the mask
  is computed from the argument's *value*, not its spelling.

#### The mechanism, as it ended up

The wrapper/body split was the real design constraint: a function-constant
argument may be **named** only where it exists, and every wrapper forwards its
arguments **by name** into a shared `_body` template. Templating the body on
the optional flags would have multiplied instantiations (`k_apply` alone ->
8x). Instead each wrapper declares a local that is null when the argument is
absent:

```metal
device uint8_t * dump_opt = nullptr;
if (mtl_bound_11) dump_opt = dump;
k_apply_body<16, 1, 1, false>(..., dump_opt, ...);
```

**The `_body` templates are untouched** -- they still receive a
possibly-null pointer and still test it. Only the wrappers changed, and they
are generated.

#### A REAL BUG the validation layer found on the way

Not an API-hygiene issue -- a host/device type mismatch:

```
Compute Function(k_transform_1): argument a0[0] from Buffer(6) with offset(0)
and length(4) has space for 4 bytes, but argument has a length(8).
```

`k_transform` declares `a0/a1/b0/b1` as `int64_t`. The warm-up launch
(`pipeline.cuh:1844`) passes the literals `1, 0, 0, 1`. **CUDA converts them
at the call site; Metal binds by value and takes the literal's own width**, so
4 bytes were bound for an argument the kernel reads as 8 -- the upper half
being whatever followed in the temporary buffer. Harmless in practice only
because that warm-up passes `n = 0u` and the loop never runs.

**This is a class, not an instance**: anywhere a narrower literal or variable
meets a wider kernel parameter, CUDA's implicit conversion is lost and Metal
binds the wrong width silently. The validation layer is the only thing that
sees it. Fixed Metal-side in `gen_pipeline_host.py`; `pipeline.cuh` untouched.

#### Kept honest by a gate

`make -f Makefile.metal validationcheck` runs the pipeline under
`MTL_DEBUG_LAYER=1`. The failure mode is an **abort**, not a diff, so "exit 0
with 37 relations" is the whole check. Without it this property rots the first
time a kernel gains an optional buffer -- which is precisely how the port got
here.

**Validated:** twelve gates green, `cofcheck.sh` 54 PASS / 0 FAIL, and the
288-q band **byte-identical** at 13,485 relations, `sha256 8e79762c…dafbc002`,
`cmp` clean. Re-staged and re-signed.

### 9z-f. Third review: the new mechanism, and memory over a full band

Two rounds each found something real, so this one led with the largest new
surface -- 9z-d/9z-e, which touched three generators, every device wrapper and
the launch path.

#### Found and fixed: the nil-mask had no width guard

`mtl_launch` builds the mask with `1u << b` over its argument pack. Past index
31 that is **undefined behaviour, and the failure would be silent**: a null
argument above 31 would report as *bound*, Metal would then demand a binding
for an argument declared under a false constant, and the port would be back to
9z-c's bug with nothing to show for it.

The widest kernel is `k_apply` at **29 bound arguments, max buffer index 27**,
so there was headroom -- but nothing protecting it, and this port's kernels
have grown before. Now a compile error:

```cpp
static_assert(sizeof...(A) <= 32,
              "kernel has more arguments than the nil-mask has bits; widen "
              "nilmask to uint64_t and the constant in the .metal sources");
```

#### Verified: validation is clean everywhere, not just the pipeline

`validationcheck` covers the pipeline. The other five binaries were run under
`MTL_DEBUG_LAYER=1` by hand, and all are clean -- **exit 0, zero assertions**:
`phase5_test`, `scan_test`, `argbuf_test`, `classify_test`, `cofac_test`.

#### Verified: memory is FLAT over a full band

This closes the question 9z-b left open. RSS across the whole 288-q band, on
the shipping binary:

| t | RSS |
|---|---|
| 40 s | 362.8 MB |
| 80 s | 368.1 MB |
| 120 s | 368.1 MB |
| 180 s | 368.2 MB |
| 240 s | **368.3 MB** |

It rises for the first ~80 s and then **plateaus: +0.2 MB over the next 160
seconds**. So the residual growth 9z-b measured and could not attribute was
the cross-q relation queue filling toward its first flush, exactly as
hypothesised -- **there is no unbounded component**. Free GPU memory ends at
10.84 GB as before, and relations are the usual 13,485.

Caveat, stated rather than glossed: this is four minutes, not four hours. It
rules out a per-flush or per-q leak, which is what the earlier trace looked
like; it cannot rule out something with a much longer period.

#### Verified, no findings

- **`fbgen_gpu_metal.cpp` is generator-reproducible.** 1,518 lines on the
  production path, and `gen_fbgen_metal.py` is described as "NOT wired into the
  build" -- which is about the build, not about drift. Running both fbgen
  generators leaves `git status` clean.
- **No emulated 64-bit counter feeds a host decision.** `atomicAdd64` folds a
  carry across two uint32 words, so a concurrent read can tear; CLAUDE.md
  claims nothing computes from them, and `nlost`, `nhit`, `noverflow`,
  `ntested`, `ndiv` and `nproj` are read only for reporting. The hazard stays
  diagnostic-only.

#### Minor, not fixed

`argbuf_test.cpp` and `classify_test.cpp` ignore `mtlMalloc`'s return at six
call sites. Production code uses `MTL_OR_DIE`. An allocation failure in those
two gates would surface as a confusing fault rather than a message -- gate
ergonomics, not a shipped defect.

### 9z-g. A field run answers 9e's open question -- and finds a rejected GPU

A **successful** workunit on an **M4 Max**, client 8.2.11. The first thing it
ever ran under a real BOINC client, which is the item 9a-9f kept listing as
not established. It worked: factor base generated on the GPU in **3.0 s**
(6.5 s on this M3), slab plan auto-calibrated, `boinc_finish(0)`.

And the very first line was wrong:

```
BOINC: this is a CUDA application but the client assigned a 'apple_gpu'
device (index 0); ignoring the assignment. ...
BOINC: no usable GPU assignment in init_data.xml; using the system default Metal device
```

**The client assigned correctly and the app threw it away.**
`bench_boinc_gpu_device()` hardcoded `strncmp(aid.gpu_type, "NVIDIA", 6)`, so
any non-NVIDIA coprocessor was refused -- including the `apple_gpu` this build
exists to use. The second line then reports that no assignment arrived, which
was **false**.

**Severity, stated honestly: no wrong results, no wrong device.** Apple silicon
exposes exactly one Metal device -- the log says `device 0 of 1` -- so the
fallback IS the assigned device. What it costs is **diagnosis**: a project
reading "this is a CUDA application" out of a Metal app version reasonably
concludes the wrong binary was deployed, and the no-assignment line is exactly
what someone would trust while debugging a real assignment problem on a
multi-GPU host.

**Fixed with the port's usual shape**, because `boinc_support.cpp` is shared
CUDA-side code: one selector, `-DBENCH_BOINC_METAL_GPU`, picks
`apple_gpu`/`Metal`/`an Apple` in place of `NVIDIA`/`CUDA`/`an NVIDIA`. The
**CUDA build is byte-identical, message text included** -- which is why there
are three macros rather than one, `KIND` carrying its own article so
"is not an NVIDIA one" survives verbatim. Verified by compiling the same file
both ways and diffing the strings.

One selector flag rather than three quoted strings on the command line,
because those have to survive make, a sub-make's `BOINC_CPPFLAGS` and two
shells -- and they do not: the first attempt died with ``No rule to make
target `Apple"'``.

#### The message sweep, and why 9b's pass was incomplete

9b fixed the CUDA-named messages it had **seen fire**. This log shows why that
is not the same as fixing the class: the two "client assigned CUDA device"
lines could not fire *at all* until the rejection above was fixed, and
`boinc_support.cpp` was never grepped.

A proper scan -- every `fprintf(stderr, ...)` whose literals mention CUDA or
NVIDIA, across nine files, multiline-safe -- found **nine**. All now say Metal:

| file | messages |
|---|---|
| `boinc_support.cpp` | 1 (the rejection, now macro-selected) |
| `bench_main_metal.cpp` | 6 (device assignment, enumeration, ordinal range, grid query) |
| `pipeline_host.inc` | 1 (memory diagnostic) |
| `bench_host.cpp` | 1 (harness error macro) |
| `fbgen_gpu_metal.cpp` | 4 (error macro, device query x2, timing events) |

The `fbgen` four matter most after the first: **fbgen runs on every production
task** that supplies no `--fb1`, which is this port's recommended shape (9e).

The scan now reports **0 remaining**, with one deliberate exception that is not
a defect: `CUDA_SIEVE_METAL_DEVICE`, an environment variable *name* belonging
to this port.

**Gates:** eleven green including `validationcheck`, `cofcheck.sh` 54 PASS /
0 FAIL, 288-q band `cmp`-identical at 13,485 relations. Re-staged and
re-signed.

**A timing anomaly worth recording, because it is about the measuring rig and
not the code.** The first post-sweep 288-q run took **17m48s** against the
usual ~3m35s. Re-run immediately after: **3m39s, identical relations**. The
slow run started straight after the full gate suite; this is a fanless
MacBook Air and CLAUDE.md's warning about sustained load is not theoretical.
A number from this box taken right after a long GPU burn is not a measurement.

### 9z-h. The launch bound at a FULL flush -- the regime every earlier measurement missed

8h and 8i both sampled **single-q runs of ~1,852 records**, where a
131,072-thread grid is so oversubscribed that the launch is chain-bound and
the chunk does nothing. That produced two conclusions this section has to
retract for the full-flush case:

- 8h: "halving the records in a launch leaves the launch's cost unchanged
  (1508 -> 1540 ms)";
- 8i/8j: "`--cof-chunk` auto ... costs 25% of the cofactor stage".

**At a real 130k-record flush the launch is RECORD-bound and very nearly
linear in the chunk.** Measured twice, back to back:

| chunk | launch (ms) | us/record |
|---|---|---|
| 15,360 | 242.5 | 15.8 |
| 30,720 | 483.4 | 15.7 |
| 61,440 | 975.8 | 15.9 |
| 131,072 | 1713.2 | 13.1 |

So the 750 ms bound **is** controllable at a full flush -- on this M3 it is not
even binding, because the opening chunk of 15,360 measures ~240 ms.

**Linearity holds only in the large-chunk regime.** Pushed down with a
deliberately tight bound, the response flattens and then reverses: 1,996
records -> 94.0 ms, 1,792 -> 139.9 ms. That is the chain reasserting itself,
and it is exactly what the no-progress guard exists for -- it fired, restored
1,996 and parked. The guard is untouched.

#### Proportional steering replaces halve/double

Against a linear response, halving is the wrong step size: it takes one flush
per factor of two, and each of those flushes runs ~67 special-q at a chunk
already known to be wrong. A proportional step aims straight at 0.8x the
bound. Measured, from a 6x overshoot:

```
flush 1: 15,360 records -> 246.2 ms      (over)
         -> 1,996 records               ONE step
flush 2:  1,996 records ->  94.0 ms
```

Halving would have needed **three** flushes (7,680 / 3,840 / 1,920) to reach
the same place. At the 150 ms bound it converged in one step and then held:
137.6, 145.3, 140.7, 136.0 ms over four flushes.

**`COF_CHUNK_TARGET_MS` is now `#ifndef`-guarded**, which is how the controller
was exercised at 150 and 40 ms without faking hardware.

#### What this does NOT establish, and one correction to my own reasoning

**The throughput cost of meeting the bound is not established.** A sweep of
wall/q against chunk came back **non-monotone** -- 825, 1501, 1658, 1208, 916
ms/q at 15,360 / 30,720 / 47,000 / 61,440 / 131,072 -- a 2x spread with no
ordering. Run-to-run variance on this fanless box is larger than the effect.
Relations were 6,724 in every run, so nothing is *wrong*; the timing simply
cannot be read.

**And the field oscillation is NOT a control-law artifact.** I assumed it was.
For a linear response it cannot be: halving lands at >= 0.5x the bound, which
can never fall below the `target/4` threshold that triggers the doubling
branch. The M4 Max log's `61440 -> 30720 -> 61440` requires consecutive
flushes to measure 4802 ms and then under 187 ms -- a **25x swing at 2x fewer
records**. That is measurement variance on that host, not the controller
choosing badly, and proportional steering does not fix it. Fixing it would
need hysteresis or averaging over flushes, and there is no data from that host
to tune either.

**On this M3 nothing changes**: the opening chunk sits inside the dead band at
the shipped 750 ms, so the controller never steers at all. The change earns
its place on larger GPUs -- an M4 Max opens at 61,440, four times this box --
and at tighter bounds.

**Validated:** `cofcheck.sh` 54 PASS / 0 FAIL, `cofaccheck`,
`validationcheck`, `boinccheck` green, and the 288-q band **`cmp`-identical**
at 13,485 relations.

Still open from earlier phases: `--mode twolevel` misplaces records and
refuses (Phase 8), and nothing has run under a real BOINC client (9e).

### The fix is in the shipped artifact, and it was revalidated end to end

`mtlFree` changes **when GPU buffers are deallocated**, so byte-identity is
the check that earns its keep here -- a buffer released while still in flight
would not be a leak, it would be a wrong answer. Rebuilt packaged
(`HAVE_BOINC=1`, embedded metallib), then over the full 288-q band from a
directory containing only the executable:

| | relations | records enqueued | `sha256(relations)` |
|---|---|---|---|
| reference (pre-fix) | 13,485 | 564,696 | `8e79762c…dafbc002` |
| **shipped (post-fix)** | **13,485** | **564,696** | **`8e79762c…dafbc002`** |

`cmp` clean. And on that same 288-q band the leak is gone: free memory ends at
**10.84 GB against 8.13 before**, **2.71 GB recovered**.

`boinclinkcheck` 8/8, signature verified against the project's public key,
negative control fails. **`--slab-j` is not a mitigation for production** --
the production band does not pass it, so the calibration path runs and the
leak was the full 2.7 GB.
---

## 10. Open questions for the CUDA side

### ECM saturates far below the derived B1 on c183 (measured, not acted on)

`cof_auto_b1` derives B1 from `lpb`, in code every port shares. On
`oracle/c183` at the production geometry, 144 special-q, relations against B1:

| B1 | relations | side 1 split | cofactor ms/q |
|---|---|---|---|
| 200 | 6,719 | 7,111 | 53.5 |
| **500** | **6,724** | **7,116** | **75.5** |
| 1,000 | 6,724 | 7,116 | 108.0 |
| 2,000 | 6,724 | 7,116 | 170.9 |
| 32,000 | 6,724 | 7,116 | 1,568.3 |

Every relation this job can reach is reached by B1 500. Beyond it the cofactor
stage grows linearly and the relation set does not move at all -- 4x the cost
at 2,000, 20x at 32,000, for nothing. Below it, B1 200 loses 5 relations, so
the knee is real and sits between 200 and 500.

**Deliberately not acted on.** B1 is shared tuning, and a Metal-only change to
it would diverge the three ports on the mathematics rather than on the
schedule. Raised here as evidence for a decision that belongs to CUDA, HIP and
Metal together.

Two caveats before anyone acts on it. The knee is a property of THIS job's
`lpb`/`mfb` -- another composite saturates elsewhere, and the method (sweep B1,
find where relations stop moving) is what transfers, not the number 500. And
it was measured on one band of one composite on one machine; the relation
counts are machine-independent, but only the counts have been checked.

Also worth noting for whoever picks this up: it explains why the Metal port's
sigma-set comparisons (8l, 8n, 8o) all came back identical. ECM at B1 2000 is
running so far past its binding constraint that which curves are tried cannot
change the outcome. That robustness is a symptom of the same over-provisioning.


Raised by this port, not caused by it. None are Metal bugs and none have been
changed.

1. **`verify_count_updates` walks in 32 bits; `k_fill_atomic` walks in 64.**
   `pl_add32_sat` saturates and ends a walk where `pl_next64` wraps and
   continues, so the two enumerate different position sets — by 2.4% on the
   c183 factor base at logI 13, differing in 4,095 of 4,096 regions.
   `bench_kernels.cu:2619`'s `--verify` gate compares exactly these two, and
   `bench_kernels.cu:2617` says that gate exists because "every placement bug
   this project has hit had exactly the right total". Either that gate is
   failing on CUDA too, or it is only ever run at a geometry where the two
   agree. Worth a look on a box with a card; demonstrated on the host with no
   GPU at all (646 moduli differ, 321 of them at the first position).
2. **`verify_apply_region` returns survivors at `cells[c] >= Cinit`** — that
   is BOUND = 0 — so its return value is not comparable with a run at any
   other bound. The Phase 5 harness derives both counts from the cell arrays
   at the threshold actually used instead.

### 8c. Slab sizing: the target is four times too large on Metal

`slab.h` picks a slab height from a target expressed in *bucket regions*,
`SLAB_PERF_REGIONS << log_region` positions, not in positions directly. CUDA
ships 32768 regions and defaults `--region` to 14, so its target is 2^29. The
Metal build defaults `--region` to 13 (Phase 8a), which silently halves the
target to 2^28 -- the region count was held fixed while the region shrank.

Sweep at the production c183 geometry, `--qrange 120000053:120000053`,
ECM, three runs each. **Relations are 37 at every point**, so this is
throughput only.

| `--slab-j` | slabs | sieve/q, both sides | vs 1 slab | bucket array |
|---|---|---|---|---|
| 16384 | 1 | 994.2 ms | — | 1.41 GB |
| 8192 | 2 | 734.1 ms | -26% | 0.71 GB |
| 4096 | 4 | 572.6 ms | -42% | 0.35 GB |
| **2048** | **8** | **558.4 ms** | **-44%** | **0.18 GB** |
| 1024 | 16 | 592.5 ms | -40% | 0.09 GB |

The minimum is interior and bracketed on both sides, so it is a real optimum
and not a boundary artefact: 1024 is worse than 2048. 2048 rows at
`--region 13` is 8192 regions, i.e. **2^26 positions per slab, a quarter of
CUDA's target**. Smaller slabs cost more launches but shrink the bucket array
by 8x, and on a UMA part that memory is the same pool the sieve is reading
through.

Rather than fork `slab.h`, its `#define` is now `#ifndef`-guarded **with its
default unchanged**, and `Makefile.metal` passes
`-DSLAB_PERF_REGIONS=8192u` to both the host and the Metal compiles -- both,
because `metal/slab_msl.h` carries a generated copy of the same constant and
two copies that disagree are exactly the rot this port has been avoiding.
(Today the device never calls `slab_perf_jmax`; the slab decision is made on
the host and reaches the GPU as parameters. The define keeps it that way by
construction rather than by luck.) See the drift ledger: the CUDA
`make slabcheck` gate still passes, and it is *sensitive* to this constant,
so that pass is evidence rather than a tautology.

Auto mode at the production geometry now reports:

```
j-slabbing: 8 slabs, up to 2048 rows/slab (auto target 8192 bucket regions
at --region 13; safety bounds may reduce it further)
bucket array 8192 x 5786 x 4 B = 0.18 GB, shared by both sides
sieve, both sides                  551.82 ms
total relations                          37
```

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** The 4x gap against CUDA's target is a statement about this part's
cache and memory system, not a universal one; a Max or Ultra has far more
bandwidth and more cores to hide launch latency behind, and should be
re-measured rather than inheriting 8192. `--slab-j` still overrides.


### 8d. Threadgroup-size tuning: the defaults are already right

Two knobs were left after 8a, and both are NVIDIA-shaped: `--threads`
(default 256, governing transform, TD, classify, intersect and both cofactor
queues) and `--blocks`, whose auto value is the literal `48 * 6` -- six blocks
per SM on an assumed 48-SM part -- giving 288 threadgroups on a 10-core GPU.

**Noise band first.** Three repeats of the unchanged configuration: wall
2509.06 / 2528.46 / 2519.27 ms, a spread of **0.77%**. Better still, `fill`
and `apply` are driven by `--fill-threads` and `--apply-threads`, *not* by
either knob under test, so they act as an internal control: across all 14
sweep runs they stayed at 146-150 ms and 349-359 ms, i.e. +/-1.4%. Anything
inside ~20 ms of wall is noise.

**`--threads`:** 256 is already the optimum.

| `--threads` | 32 | 64 | 128 | 192 | 256 | 512 |
|---|---|---|---|---|---|---|
| transform | 92.5 | 63.1 | 58.4 | 62.5 | 55.6 | **50.6** |
| norms + trial division | 512.1 | 291.7 | 159.6 | 106.0 | **92.8** | 122.2 |
| classify | 26.6 | 19.0 | **17.6** | 24.1 | 31.3 | 35.0 |
| algebraic queue | 1504.9 | 1574.4 | 1502.9 | 1570.3 | 1519.9 | 1696.2 |
| **wall** | 3018.7 | 2785.6 | 2559.9 | 2600.6 | **2530.3** | 2761.2 |

**`--blocks`:** flat from 80 upward **at this section's one-q band size --
see 8i, where it is not flat at all.**

> **CORRECTION.** This section said the default was the inherited `48 * 6` =
> 288. It is not: `bench_main_metal.cpp` sets `cfg.blocks` to
> `multiProcessorCount * 6` before the pipeline ever sees it, which on a
> 10-core M3 is **60**, and the `48 * 6` in `pipeline_host.inc` is a fallback
> that never fires. 60 sits just below where the curve flattens -- the table
> puts it between 2537 and 2526 ms against 2517 at 288 -- so the real default
> is worth perhaps 10-20 ms, at the edge of the 0.77% noise band. The
> conclusion "do not hand-tune this" survives; the stated default was wrong.

| `--blocks` | 10 | 20 | 40 | 80 | 144 | 288 | 576 | 1152 |
|---|---|---|---|---|---|---|---|---|
| algebraic queue | 1508 | 1523 | 1505 | 1508 | 1505 | 1509 | 1505 | 1509 |
| **wall** | 2712 | 2582 | 2537 | 2526 | 2520 | **2517** | 2515 | 2519 |

Everything from 80 to 1152 is inside the noise band of everything else, so
288 is kept -- not because it was chosen for this hardware, but because on
this hardware the choice does not matter. Only starving the grid (10 or 20
threadgroups) is measurably bad. **No change to either default.**

**One knob, three stages, three different optima.** TD wants 256, classify
wants 128 (17.6 vs 31.3 ms, far outside the +/-1.6 ms repeat spread), and
transform mildly wants 512. Splitting `--threads` per stage is therefore a
real but *small* win: at each stage's own optimum the saving is about 5 ms on
transform and 14 ms on classify, ~19 ms against a 2520 ms wall, or **0.75% --
the same size as the noise band**. Measured, and declined: a third geometry
knob is not worth 0.75%, and TD, which dominates the three, is already at its
best value.

**Why the cofactor stage ignores all of this.** The algebraic queue is 1500 ms
of a 2520 ms wall here -- and it moves by under 2% across a 16x range of
threadgroup sizes and a 115x range of threadgroup counts.

> **CORRECTION (8h).** That 59% share is an artifact of this section's
> single-q benchmark, not a property of the port. The cross-q queue is built
> to flush 131,072 records and a one-q band hands it 1,852 -- 1.4% of a batch.
> At a production band size the cofactor stage is ~39% of wall and slightly
> *smaller* than the sieve. The geometry-immunity finding below is unaffected
> and its explanation is the same one that explains the inflation: the stage
> is starved of records, and launch geometry cannot invent any. See 8h. It is not occupancy-bound, so no packing fixes it. 8b's
table already says why, read the other way round: dividing the *same* 1,852
records into more launches costs a near-constant amount per launch.

| records per launch | 1852 | 926 | 463 | 256 |
|---|---|---|---|---|
| total algebraic | 1508 | 3080 | 5044 | 7696 |
| **per launch** | **1508** | **1540** | 1261 | 962 |

Halving the records per launch leaves the per-launch cost **unchanged**
(1508 -> 1540 ms, inside noise). A stage whose cost does not fall when you
halve its work is running down a critical path, not a throughput limit: the
ECM round structure is a long dependent chain per record, and the launch ends
when the slowest record does. Grid shape cannot shorten a dependent chain.
The lever for this stage is `--ecm-curves` / `--ecm-b1` / `--cof-rounds`,
which change the mathematics rather than the schedule, and so are out of
scope for tuning.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** Relations were 37 at all 20 configurations. As in 8a, the shape
should carry to other Apple GPUs and the exact values may not -- but "the
inherited NVIDIA geometry is already in the flat region" is the kind of
result that carries further than a specific optimum would.

**Three stale `--help` defaults, found while checking this.** `--help` on the
Metal build still advertised CUDA's numbers for values this port had already
changed: `--apply-threads [512]` when 8a shipped 192, `--region [14]` when
this build defaults to 13, and a `--slab-j` paragraph still working its
example from "the default --region 14" with a 2^29 cap -- arithmetic that
8c had just invalidated twice over. The code was right and the documentation
the user actually reads was wrong. Corrected, with the reason on each line.
(`%` needs escaping as `%%` there; line 296 was already doing it.)


### 8e. Regions below 2^13: measured, and they are all worse

8d tuned the knobs but never asked whether `--region 13` is *good*. It had
only ever been justified as **the largest value that fits** under Apple's
32,768 B threadgroup ceiling (5.1), which is a constraint, not a measurement.
There was a plausible story for going lower: `k_apply`'s threadgroup
requirement is `2^log_region * 2 + nslice_pow2 * 2` bytes, so every step down
halves it and should buy occupancy.

It does not. Swept with **`--slab-j 2048` pinned**, which holds the slab at
the 8c optimum of 2^26 positions regardless of region -- without that pin the
sweep would be moving slab size at the same time, since the auto target is
`SLAB_PERF_REGIONS << log_region`.

| `--region` | 10 | 11 | 12 | **13** |
|---|---|---|---|---|
| threadgroup B | 2,176 | 4,224 | 8,320 | 16,512 |
| regions/slab | 65,536 | 32,768 | 16,384 | 8,192 |
| fill | 518.5 | 311.5 | 175.1 | **154.4** |
| apply | 1497.8 | 841.2 | 507.1 | **349.6** |
| sieve, both sides | 2063.6 | 1212.1 | 739.5 | **562.2** |
| wall | 4036.0 | 3181.2 | 2704.2 | **2544.6** |

Monotonic, large, and in the opposite direction to the occupancy story. 37
relations at every point. **The region wants to be as big as it is allowed to
be**; freeing threadgroup memory buys nothing because the cost is per *region*,
not per byte.

The numbers fit a straight line in the region count almost exactly:

```
apply  =  182.3 ms + 20.07 us per region     residuals +2.8 -4.1 +1.2 +0.1 ms
fill   =   87.6 ms +  6.58 us per region
```

Three independent incremental estimates of the apply slope -- 19.2, 20.4 and
20.0 us -- agree across a 4x range of region counts. So each bucket region
costs about **20 us of fixed overhead** in apply and 6.6 us in fill, against a
constant term that is the actual useful work. Halving the region size does not
halve anything; it doubles the number of times that fixed cost is paid.

**This makes Apple's 32 KB ceiling a measured performance cost, not just a
constraint.** Extrapolating the same fit one step up, to CUDA's default
`--region 14` and its 4,096 regions:

| | region 13 (measured) | region 14 (predicted) |
|---|---|---|
| apply | 349.6 ms | ~265 ms |
| fill | 154.4 ms | ~115 ms |
| sieve | 562.2 ms | ~440 ms, about **-22%** |

**And we miss it by 128 bytes.** At region 14 the requirement is
`32,768 + 128 = 32,896` B against a 32,768 B ceiling -- the overflow this port
documented in Phase 5 and worked around by dropping to 13. Those 128 bytes are
`nslice_pow2 * 2` for 64 slices: a **read-only** lookup table that the kernel
copies in from a `device const uint16_t *` it already has
(`lut[i] = slice_logp[i]`, `bench_kernels_body.metal.inc:538`) and then only
ever reads. It is in threadgroup memory for an NVIDIA reason -- shared memory
beats global for repeated random access there -- that does not obviously apply
to a 128-byte table on an Apple GPU, where it would sit in cache or could be
declared `constant`.

If that copy were dropped on the Metal side, `k_apply` would need exactly
32,768 B at region 14: equal to the ceiling, and the check is `>`, so it fits.
**Attempted, and the extrapolation was wrong -- see 8f.** Region 14 does now
fit and run correctly, and it is *slower*. The per-region model above is
sound but incomplete: it has no term for what happens when a threadgroup asks
for the entire 32 KB budget.

Note also that at region 14 the slab target moves with it -- `8192 << 14` is
2^27 -- so `SLAB_PERF_REGIONS` would want halving to 4096 to keep the 8c
optimum of 2^26 positions per slab. (The 8e sweeps and 8f's all pin
`--slab-j 2048` instead, which fixes 2^26 positions directly.)

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8f. The 128 bytes are gone; region 14 fits, runs, and is still slower

8e predicted region 14 at about -22% on the sieve if `k_apply`'s 128-byte
slice-log copy were dropped. The copy is gone, region 14 fits, every answer is
bit-exact -- **and the prediction was wrong**.

The change is three lines of device code: `k_apply` no longer copies
`slice_logp` into threadgroup memory, it indexes the `device const uint16_t *`
it was already being passed. The table is read-only after the copy and indexed
divergently (`(r >> 16) & (nslice - 1)`), so there is no uniform-access case to
make `constant` the right space, and on a UMA part with a unified cache there
is no faster pool to promote 128 hot bytes into. The host side must agree, or
it reserves bytes the kernel never indexes; that figure had been written out
three times -- the pipeline, the `bench_kernels` harness, and the Phase 5 gate
-- so it is now `mtl_apply_smem()` in `metal_rt.h`, named once. A threadgroup
length that disagrees with what the kernel indexes is not a compile error, it
is a wrong answer.

**Region 14 now runs.** It had been failing closed since Phase 5. At 4,096
regions it needs exactly 32,768 B, the checks are all `>`, and it produces 37
relations. It is also slower than 13:

| `--region`, `--slab-j 2048` | 12 | **13** | 14 |
|---|---|---|---|
| fill | 176.1 | 147.9 / 147.6 / 148.3 | **135.3 / 136.0 / 136.5** |
| apply | 508.8 | **340.0 / 342.0 / 345.4** | 389.8 / 394.4 / 395.1 |
| sieve, both sides | 738.7 | **542.0 - 548.8** | 579.9 - 588.9 |

The decomposition is the interesting part, and it rescues the 8e model rather
than discarding it. **`fill` behaves exactly as predicted** -- 8e's fit said
~115 ms at region 14 against 154 at 13, and it measures 136, the right
direction and the right order. Fill carries no region-sized threadgroup array.
**`apply` does the opposite**: predicted ~265 ms, measured ~393. The per-region
overhead is real and still ~20 us, but at region 14 `k_apply` asks for the
entire 32,768 B threadgroup budget, so a core can hold **one** threadgroup and
nothing else. That occupancy cliff costs more than the halved region count
saves. 8e's model was fitted entirely below the cliff and had no term for it.

So `--region 13` stays the default, now on a measurement rather than on a
constraint. Region 14 is no longer refused, which matters for hardware with a
larger ceiling, where the region-count saving would arrive without the cliff.

**What the change is actually worth, measured back to back.** The 8e numbers
were taken across a long session on a fanless chassis, so the before/after was
re-run against a rebuild of the old code rather than against them:

| region 13, `--slab-j 2048` | before | after | |
|---|---|---|---|
| apply | 357.4 ms [352.3-364.3] | **342.4 ms** [340.0-345.4] | **-4.2%** |
| sieve, both sides | 560.3 ms [557.3-562.6] | **546.2 ms** [542.0-548.8] | **-2.5%** |

Three runs each, and the two apply ranges **do not overlap** -- every run after
is faster than every run before. Small, but real, and it costs nothing: the
threadgroup copy plus its barrier traffic bought nothing on a unified cache.
All five gates green; `sievecheck` compares all 4,194,304 cells exactly.

**One measurement discarded, recorded here because it nearly misled the whole
conclusion.** The first region-14 run reported apply at 905.6 ms, which would
have made the ceiling look catastrophic rather than merely unhelpful. Three
repeats gave 389.8 / 394.4 / 395.1. The outlier was a cold first run on a
fanless machine that had just finished a gate sweep. **On this box a single
timing is not a measurement.**

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**

### 8g. Slab auto-calibration, ported from the HIP port

8c set `SLAB_PERF_REGIONS` to 8192 from a sweep on **one** machine. The HIP
port faced the same problem and solved it better than a constant: probe the
hardware in front of you at startup. Ported here from `pipeline_hip.cuh`
(commits `d3d9e7d`, `3483e17`, `eb72ede`, `3b7498a`).

**What it does.** Before the real band, time the run's actual first special-q
against three candidate slab sizes and keep the fastest. Costs one extra
special-q at startup, negligible against a run measured in hours. Skipped
when `--slab-j` is given (a regression knob auto-tuning must not override) or
when the geometry is below the slabbing trigger -- **both confirmed, and they
are why the existing gates are undisturbed: `cofcheck.sh`'s geometries sit
below the trigger, so calibration never runs there.**

**Candidates are one octave below the HIP port's.** It probes
{2^27, 2^28, 2^29} because gfx1103 measured 2^27; this probes
{2^25, 2^26, 2^27} because 8c measured 2^26 here. The measured winner is the
*interior* point, which is what lets a three-point probe confirm or move it --
the HIP list has its winner at the boundary.

**It reproduces 8c's answer from scratch**, which is the validation that
matters:

| candidate | 2^25 (1024 rows) | **2^26 (2048)** | 2^27 (4096) |
|---|---|---|---|
| probe time | 961.6 ms | **800.8 ms** | 904.4 ms |

Same bracketed interior minimum, same winner, arrived at independently of the
static constant.

**Where it earns its keep is where the static default is wrong.** The static
target is region-relative (`SLAB_PERF_REGIONS << log_region`), so it drifts
with `--region` while the hardware's preference does not:

| `--region 12` | rows/slab | slabs | sieve, both sides |
|---|---|---|---|
| static default (`8192 << 12` = 2^25) | 1024 | 16 | 758.2 ms |
| **auto-calibrated** | **2048** | **8** | **729.6 ms** |

Calibration finds 2^26 again and is **3.8% faster** than the constant. 37
relations both ways.

**The absolute-vs-region-relative caveat is inherited and real.** Candidates
are absolute position counts, measured at `--region 13` only; calibration does
not re-derive them against a caller's own `--region`. The table above is that
caveat working *for* us, but it is the same mechanism that would work against
us on a geometry where the region-relative default happens to be right.
Flagged, as the HIP port flagged it, rather than silently assumed correct.

#### The BOINC counter, which is the part that bit the HIP port in the field

Calibration runs throwaway bands through the **same** `run_pipeline_impl` that
reports progress. With one q in and one q retired that reads as 1/1 -- clamped
to 0.99 and, because BOINC reports must be nondecreasing, **0.99 becomes the
floor for the entire workunit**. The field symptom was a task pinned at 99%
within seconds of starting and staying there for hours.

Three things now stand between this port and that:

1. `bench_boinc_progress_suspend()` (ported into `boinc_support.cpp`). The
   guard sits **before** the monotonic high-water mark, not after -- a
   suspended report must be dropped without advancing the mark, and the
   difference between those two placements is the whole bug.
2. `g_runlog_quiet` (ported into `runlog.c`), so a throwaway pass cannot write
   warnings a reader would attribute to the real run.
3. **`make -f Makefile.metal boinccheck`**, a new gate.

The gate matters more than it looks, because `HAVE_BOINC` defaults to 0: the
entire fraction-done path is **compiled out of every ordinary build**, so
nothing else in the tree can reach it, and this is precisely how the bug
reached the field in the first place. The gate compiles `boinc_support.cpp`
with `-DHAVE_BOINC` against a stub client API in `metal/boinc_stub/` and
drives it directly.

It runs the **control first, in its own process**, and the control must
*reproduce* the bug:

```
== control: WITHOUT the suspend ==
PASS   control: an unsuspended calibration band reports 0.99
PASS   control: the real band's 0.4% is then swallowed by the 0.99 floor
PASS   control: so is 50% -- the task is pinned until it truly passes 99%
== gate: WITH the suspend ==
PASS   suspended calibration reports never reach BOINC
PASS   the real band's first report still starts from 0.4%, not 99%
PASS   and progress keeps advancing normally afterwards
PASS   monotonicity is preserved -- a backwards report is still dropped
```

Own process per case for the same reason `argbufcheck` isolates its residency
control: the high-water mark is a static with no reset -- deliberately, it is
a per-task invariant -- so a control sharing it with the case it controls
would prove nothing.

**The status line was also made to stop lying.** It reported the static target
(`auto target 8192 bucket regions`) even when calibration had overridden it,
naming a target that did not produce the plan being printed. It now says which
decided. HIP's `BOINC: slab plan:` stderr line is ported too, gated on there
having *been* a decision worth reporting, so the below-trigger majority does
not dilute the fleet aggregate it exists to produce.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** The point of calibration is that this sentence stops mattering.


### 8h. The cofactor stage does not dominate. The single-q benchmark does.

Every number in 8a-8g came from a one-special-q band, because that is the
geometry the parity gate and `cofcheck.sh` pin. For the sieve that is
harmless -- sieve/q is 527-546 ms whether the band is 1 q or 72. For the
cofactoriser it is **not**, and it inflated that stage's apparent share by
nearly 4x.

`cofac.cuh` sets `CQ_FLUSH` to **131,072 candidates per flush, "~67
special-q"**. The queue is explicitly cross-q: it exists so the cofactoriser
runs on a batch assembled from many special-q rather than on one q's worth.
A one-q band hands it **1,852 records -- 1.4% of one intended batch** -- and
the kernel assigns *one thread per record*, so the grid runs at a few percent
occupancy and the launch costs what its longest ECM chain costs.

| band | records per flush | cofac/q | sieve/q | wall/q | cofac share |
|---|---|---|---|---|---|
| 1 q | 1,852 | 1753 ms | 546 | 2572 | **68%** |
| 4 q | 7,671 | 535 | 537 | 1273 | 42% |
| 8 q | 15,555 | 589 | 537 | 1304 | 45% |
| 24 q | 46,893 | 463 | 527 | 1191 | 39% |
| **72 q** | **140,363** | **464** | **537** | **1183** | **39%** |

**In steady state the cofactor stage is ~39% of wall and slightly smaller
than the sieve** (464 vs 537 ms/q). It flattens by about 24 q and does not
improve further at 72, which is where a full `CQ_FLUSH` batch is reached
(140,363 records is one in-loop flush plus the final one). Per-q wall is
**1183 ms against the one-q band's 2572** -- the single-q figure overstates
the real cost of a q by 2.2x, essentially all of it in this one stage.

So the ordering to carry forward is sieve first, cofactoriser second, and the
"more than every other stage combined" framing in 8d was measuring the
benchmark rather than the port.

**What this does not change: 8d's geometry-immunity result.** The stage is
insensitive to `--threads` and `--blocks` because the number of *active*
threads is the record count, which no launch geometry can alter. That is the
same fact that makes a one-q band so slow, seen from the other side. It also
means 8b's `--cof-chunk` penalty was measured in the regime where chunking is
most obviously harmful -- chunking 1,852 records below a
`blocks * threads` floor of 15,360 -- and should be re-measured at production
batch size before its 2x figure is quoted as a general property.

**Method note, which is the durable lesson.** A single-q band is the right
gate for correctness and the wrong instrument for throughput, and nothing in
the output says so: the per-q breakdown looks exactly as authoritative at
`--nq 1` as at `--nq 72`. Any future tuning of a *queued* stage must state its
band size alongside the machine, exactly as this port already states the
machine alongside every number.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8i. `--cof-chunk` auto is wrong here, and the cause is `--blocks`

8h predicted that 8b's `--cof-chunk` penalty needed re-measuring at a real band
size. It did, and the answer is worse than "the old number was unrepresentative":
**auto picks the worst available chunk**, and the reason is a grid sized for
NVIDIA's idea of a core count.

At `--nq 24` (46,893 records in the flush), auto selects 15,360 records/launch
-- the floor, `blocks * threads` -- and splits each round into 4 launches:

| `--cof-chunk` | launches/round | algebraic | cofac/q | wall/q |
|---|---|---|---|---|
| **15360 (auto)** | **4** | 423.9 | **466.2** | 1197 |
| 30720 | 2 | 404.0 | 441.0 | 1157 |
| 61440 | 1 | 377.5 | 416.9 | 1130 |
| 131072 | 1 | 372.8 | 409.8 | 1129 |

Monotonic: every subdivision costs. 1,114 relations at every point.

**But turning chunking off is the wrong fix, because the floor is not the
problem -- the grid is.** `cof_chunk_floor()` is `blocks * threads`, and the
CUDA design's argument for it is sound: a slice of exactly one record per
thread is fully loaded, so small devices subdivide for free. That argument
depends on `blocks * threads` being a grid the device can actually fill, and
on Metal it is not. `bench_main_metal.cpp` sets `blocks` to
`multiProcessorCount * 6`, which is **60** on a 10-core M3, giving 15,360
threads. CUDA's multiplier works because NVIDIA SM counts are large -- the
same formula gives a 4090 768 blocks and 196,608 threads, which the comment in
`cofac.cuh` notes "exceed CQ_FLUSH outright". Apple reports ~10-40 GPU cores,
so the identical formula produces a grid **8.5x smaller than the flush it has
to process**, and each thread walks several records serially.

Fix the grid and both problems go away at once. At `--nq 72`, a full
`CQ_FLUSH` batch of 140,363 records:

| `--blocks` | threads | chunking | algebraic | cofac/q | wall/q |
|---|---|---|---|---|---|
| 60 (default) | 15,360 | 9 launches | 423.8 | 465.3 | 1180.7 |
| 288 | 73,728 | 2 launches | 330.5 | 365.3 | 1069.2 |
| **576** | **147,456** | **none** | **312.4** | **346.6** | **1051.9** |
| 1152 | 294,912 | none | 318.7 | 353.4 | 1058.9 |

A bracketed interior minimum at 576, and 3,385 relations at every point.
Against the shipped default that is **-25.5% on the cofactor stage and -10.9%
on wall**. 576 x 256 = 147,456 threads clears `CQ_FLUSH`'s 131,072, so the
build lands on exactly the "large device" path `cofac.cuh` describes: one
slice, one record per thread, no chunking.

**This is not a weaker watchdog margin -- it is a stronger one.** The risk
chunking exists to bound is the duration of a single launch. At 60 blocks an
unchunked launch would make every thread walk ~8.5 records end to end; at 576
each thread handles at most one, so the launch is shorter as well as faster.
Raising the grid shortens the very thing the chunk was protecting.

**And it invalidates 8d's `--blocks` conclusion, for the third time from the
same cause.** 8d swept `--blocks` and found it flat, concluding "do not
hand-tune this". That sweep ran at `--nq 1`, where 1,852 records leave even a
60-block grid oversubscribed, so grid size could not matter. At band size it
is worth 10.9% of wall. Flatness measured on a starved stage says nothing
about the stage when fed.

#### The rule is a threshold, not a slope

`CQ_FLUSH` is the queue's **capacity**, so a flush never exceeds 131,072
records -- `--nq 72`'s 140,363 is the band's cumulative total, one full flush
plus a 9,291-record remainder. That makes the target exact: one record per
thread for a full flush is `131072 / 256` = **512 blocks**. Measured at
`--nq 72`, with every stage `--blocks` feeds:

| `--blocks` | transform | TD | classify | algebraic | cofac/q | wall/q |
|---|---|---|---|---|---|---|
| 60 (default) | 74.1 | 114.7 | 34.7 | 449.6 | 495.0 | 1291.3 |
| 288 | 50.5 | 132.9 | 24.8 | 335.8 | 369.8 | 1080.8 |
| **512** | 60.7 | **131.7** | 19.7 | **322.2** | **357.0** | **1067.5** |
| 576 | 53.2 | 133.4 | 18.1 | 321.8 | 356.5 | 1066.1 |

3,385 relations at every point. **512 and 576 are indistinguishable** (321.8
vs 322.2 ms), and 1152 was slightly *worse* in the previous table. So this is a
threshold and not a curve: clear `CQ_FLUSH` and you are done, and threads
beyond the record count buy nothing because there are no more records to give
them. (These absolute numbers run slightly higher than the previous table's --
same machine, later in a long session, fanless. Compare within a sweep, not
across them.)

**The trade-off it exposes, stated honestly:** `blocks` is a shared knob, and
**TD gets ~15% worse** (114.7 -> 131.7 ms) because TD prefers a smaller grid.
It is swamped -- the cofactor stage gives back 127 ms against TD's 17 -- so
wall falls 17.3%. But it is a real regression in one stage, not a free win.

#### Recommended, not yet applied

A **per-core multiplier is the wrong shape for this**. Reaching 512 on a
10-core M3 needs `x 51`; that same multiplier gives a 40-core M3 Max 2,048
blocks and 524,288 threads, four times a full flush, past the point where 1152
already measured slightly worse. The work is fixed at 131,072 records while
Apple core counts vary by 8x, so no single multiplier fits the range.

Size it from the work, keeping the core-count rule as a floor:

```c
auto_blocks = max(multiProcessorCount * 6,           /* the old rule */
                  (CQ_FLUSH + threads - 1) / threads); /* one record/thread */
```

10-core M3 -> `max(60, 512)` = 512. 40-core M3 Max -> `max(240, 512)` = 512. A
hypothetical 128-core part -> `max(768, 512)` = 768, where the core term takes
over. Must use the live `threads`, not a literal 256, since `--threads` moves.

**The caveat that makes this the user's call, not a tuning pass's.** Auto
chunking clamps to `cof_chunk_floor()` = `blocks * threads` as a hard lower
bound, so a grid at or above `CQ_FLUSH` means **subdivision can never engage
again** -- by construction, on every Apple GPU. On this M3 that is measured to
be right and is also the *shortest* possible launch. On an **M1**, which this
port has never run on at all (5.1a), it removes a watchdog protection whose
value there is unmeasured. Either decouple the chunk floor from the grid so
`COF_CHUNK_TARGET_MS` can still subdivide, or accept that Apple parts take the
"large device, one slice" path the CUDA design already grants a 4090.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8j. Decoupling done. It works, and it proves the budget is the real bug.

Applied, as two independent changes:

**1. The grid is sized from the work.** `bench_main_metal.cpp` now computes
`max(multiProcessorCount * 6, ceil(CQ_FLUSH / threads))`, using the live
`--threads` and reaching `CQ_FLUSH` through an accessor rather than a second
`#define`, since the constant must have one definition and the grid must
track it. On this box: `grid: 512 blocks = max(10 cores x 6, 131072 records /
256 threads)`. The core rule survives as a floor for a hypothetical very large
Apple GPU.

**2. The chunk floor no longer follows the grid.** `cof_chunk_floor()` used
`blocks * threads`, so a work-sized grid would have made the floor equal a
whole flush and retired subdivision on every Apple GPU by construction. It now
uses its own core-derived count (`mtl_set_cof_floor_blocks`, set once at init),
falling back to the caller's grid if never set. Subdivision stays reachable for
a slow part -- and it is not pointless there, because a launch costs roughly
`(records / resident threads) x one ECM chain` and a 10-core part cannot hold
131,072 threads resident.

Both verified. All five gates green, 3,385 relations unchanged.

**And by default it changes nothing, because `--cof-chunk auto` is not
adaptive on this hardware -- it never was.**

| `--nq 72` | chunk | cofac/q | wall/q |
|---|---|---|---|
| auto, decoupled floor | 15,360 | 465.5 | 1201.9 |
| explicit `--cof-chunk 131072` | 130,940 | **348.5** | **1085.0** |

The adaptive loop halves the chunk when `stage > COF_CHUNK_TARGET_MS` and
doubles it when `stage < TARGET/4`, where `stage` is a whole SIDE's device time
for the flush -- summed over every round and every slice. `COF_CHUNK_TARGET_MS`
is 250. Measured here, `stage` is **9.8 s at one slice and 11.2 s at the floor:
39x and 45x the target**. The test is true at every chunk size this hardware
can produce, so the loop halves on every flush and parks at the floor forever.
It is a saturated signal, not a controller. **An always-true test is not a
safety mechanism; it is an unconditional slowdown.**

That saturation was harmless on CUDA and AMD because there the floor *is* a
fully loaded grid -- `cofac.cuh` says so: "over-chunking costs nothing on the
devices that ever reach this path". Parking at the floor is the free point
when the floor is one record per thread for a device-filling grid. Decoupling
the floor on Metal is correct for watchdog reachability and simultaneously
removes that harmlessness: parking now costs 25% of the stage.

**Why no threshold on `stage` can fix this.** Dividing by slices does not help
-- `stage` is dominated by total ECM work across rounds, not by one launch.
Even a correct per-launch figure is about 3 s here (roughly `stage` over ~8
rounds), still 12x a 250 ms budget that was chosen for an 8x margin against
Windows' 2 s TDR. Yet those ~3 s launches have run repeatedly on this machine,
across every one-slice arm in 8i and 8j, with no timeout and no reset. **The
250 ms target is simply not calibrated for macOS**, and the honest options are
about that number, not about the floor.

#### Where this leaves the decision

- **(a) Raise `COF_CHUNK_TARGET_MS` for Metal**, `#ifndef`-guarded like
  `SLAB_PERF_REGIONS`. Auto then stays at one slice on a healthy device and
  still descends on one substantially slower. Needs a value justified by
  observation rather than by a documented macOS limit, because Apple publishes
  none -- ours would be calibrated on a single M3.
- **(b) Fix the signal first**: measure a real per-launch duration with events
  instead of a whole-side sum, then pick a target against a quantity that
  actually means what the target says. More work, and the only option that
  leaves a genuinely adaptive controller.
- **(c) Leave auto alone and document `--cof-chunk 131072` as the production
  setting.** Costs nothing, wins nothing by default, keeps every existing
  safety property.

**Not chosen here.** All three change a watchdog-safety default on hardware
this port has partly never run on -- an M1 remains untested (5.1a) -- and that
is the user's call. The measured facts are above; the 25% is available with
one flag today.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8k. A 750 ms launch bound, and the case that crashed the machine twice

The user set the policy: **no kernel launch longer than about 750 ms**, for UI
responsiveness as much as watchdog safety, and worth paying ~10% of a stage to
hold. Finding out whether that was reachable found something worse first.

#### The record axis cannot do it

`--cof-chunk` splits the RECORD list, so the smallest launch it can make is
one record per thread -- and a launch can never be shorter than **one record's
own ECM chain**. Measured at `--nq 24`:

| `--cof-chunk` | 15360 | 7680 | 3840 | 1920 | 960 | 480 |
|---|---|---|---|---|---|---|
| longest launch | 3567 | 1846 | 1641 | **1495** | 1499 | 1498 |
| cofac/q | 465 | 592 | 884 | 1522 | 2777 | 4969 |

Flat below ~1920 records: that is one chain at 48 curves. Subdividing past it
buys nothing and costs everything -- cofac/q rises 10x.

#### The curve axis can, and is free

`--ecm-curves` is **per round**, and each round takes a disjoint sigma block
(`sigma = c0*1000 + cv + 6`), so the same total curve budget can be spread over
more, shorter rounds. At B1 2000 / B2 60000, 192 curves either way:

| config | longest launch | cofac/q | relations |
|---|---|---|---|
| 48 curves x 4 rounds | 3613 ms | 467.2 | 1114 |
| 24 x 8 | 1498 ms | 476.9 | 1114 |
| 16 x 12 | 1164 ms | 392.1 | 1114 |
| 12 x 16 | 969 ms | 361.2 | 1114 |
| **8 x 24** | **740 ms** | **354.6** | **1114** |

**The bound is met at 8 curves x 24 rounds, and it is not a 10% sacrifice --
it is a 24% GAIN** (467.2 -> 354.6 ms/q). Shorter rounds re-compact the live
list more often, so records that have already split stop occupying threads.
`--cof-rounds` caps at 24, so at a 192-curve budget 8 x 24 is also the floor of
this technique; going below ~740 ms would mean cutting the budget, which is a
yield decision.

**This is a mathematics change, not a tuning one.** `cofac.cuh` says so
directly -- "a curve sub-range makes a later chunk restart the top composite
with sigmas that cannot split it" -- so 48x4 and 8x24 try *different* 192-sigma
sets. Relations were identical at every point above, but that is empirical on
one band, not structural the way record chunking is. **Recommended as job
settings, NOT adopted as defaults.**

#### What the auto chunker now does

`COF_CHUNK_TARGET_MS` is 750 on this build, compared against 8k's measured
launch (not CUDA's 250 against a whole-side sum -- the two are not comparable).
Auto opens at one core-derived grid of records and walks down. At `--nq 144`:

- **bound reachable** (8 x 24): opens 15,360 -> measures 1194 ms -> 7,680 and
  settles. cofac/q 265.0, wall 971.9, 6,724 relations.
- **bound unreachable** (48 x 4): 15,360 -> 7,680 -> 3,840 -> 1,920, each step
  still buying >10%, and a **no-progress guard** stops the descent when a
  halving stops paying. Without it the controller subdivides to the floor
  forever -- the same unconditional slowdown 8j removed, reached from the other
  side.

#### The crash

**Two WindowServer crashes plus userspace watchdog timeouts, on an otherwise
idle machine, 1m47s and 1m48s into `cofcheckgate` -- the same case both times.**
Both gate logs are 14 lines and end identically. The case is:

```
--ecm-b1 400000        # B2 derives to 10^7
```

which is **320,000 giant steps and 33,860 prime powers: ~15.9 s of work in ONE
curve**, and at that case's 12 curves/round a single kernel launch of about
**190 seconds**. Nothing in the port can bound it -- chunking splits records,
never the chain.

`cofac.cuh` already knew. Its warning block names this exact configuration as
"the one configuration observed to actually trip a device watchdog ... and is
why cofcheck.sh skips that case on HIP". **This port did not skip it.** On
gfx1103 it is a caught device failure; on Apple silicon the GPU also drives the
display, so it is not a failed task, it is a dead session.

Two changes, and the second is the one that matters:

1. `cofq_init` now **REFUSES** when one curve's estimated work exceeds
   `COF_LAUNCH_REFUSE_MS` (10 s), naming the numbers and the knobs. The
   estimate is ~0.045 ms per (prime power + giant step), calibrated at one
   curve on this M3: 110 / 370 / 1461 ms at B1 2000 / 8000 / 32000. It refuses
   rather than silently lowering `--ecm-curves`, because curves-per-round
   chooses which sigmas run.
2. `cofcheck.sh` detects the build from `--help` (the Metal build now says
   "select Metal device", mirroring HIP's marker) and **inverts that one case
   on Metal**: it asserts the refusal instead of asserting acceptance. Refusing
   to guess if neither marker matches, exactly as the HIP port does, because a
   silent "assume CUDA" re-enables a machine-crashing case.

`cofcheckgate` now runs to completion for the first time since the crashes:
**54 PASS, 0 FAIL, exit 0**, including `large B1 with derived B2 -> refused, as
Metal must`. All six gates green.

**The 10 s refusal threshold is a judgement, not a measurement**, and is
deliberately closer to what is known to work than to what is known to kill:
launches of 3.6 s and 6.9 s have run here repeatedly without incident, and the
case that took the machine down estimates at 15.9 s per curve. Nobody has
measured where macOS actually draws the line, and finding out means crashing
the machine on purpose.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** That last clause stopped being a disclaimer here and became the
hazard.


### 8l. 8x24 validated over 288 special-q: not "comparable" -- identical

8k recommended `--ecm-curves 8 --cof-rounds 24` but would not adopt it,
because `cofac.cuh` warns the curve axis changes which sigmas run and equal
relations had only been seen on one band. Validated against 48x4 over
**288 special-q**, same B1 2000 / B2 60000, accepting in advance that a few
relations might be gained or lost:

| | 48c x 4r | 8c x 24r |
|---|---|---|
| relations | **13,485** | **13,485** |
| shared (a,b) | 13,485 | 13,485 |
| unique to this run | **0** | **0** |
| records enqueued | 564,696 | 564,696 |
| side 0 split / dead / stuck | 477,071 / 87,625 / 0 | 477,071 / 87,625 / 0 |
| side 1 split / dead / stuck | 14,291 / 550,405 / 0 | 14,291 / 550,405 / 0 |
| cofactor ms/q | 835.3 | **277.4** |
| wall ms/q | 1555.8 | **981.8** |
| longest launch | 3620 ms | 1207 ms |

Not merely comparable yield -- **the identical relation set**, and identical
split/dead/stuck on both sides. The sieve does not depend on ECM parameters,
so both runs hand the cofactoriser the same 564,696 candidates; that number
matching is what makes the rest a like-for-like comparison rather than two
different experiments.

**Why identical, when 160 of the 192 sigmas differ.** Sigma is
`c0*1000 + cv + 6` with `c0 = round + 1`, so 48x4 tries {1006-1053, 2006-2053,
3006-3053, 4006-4053} and 8x24 tries {1006-1013, ..., 24006-24013}; they share
only 32. They agree because **the differing sigmas never split anything**:

| at B1 2000 | relations | side 1 split |
|---|---|---|
| 8 curves x 1 round | 6,723 | 7,115 |
| 8 curves x 4 rounds | 6,724 | 7,116 |
| 8 curves x 24 rounds | 6,724 | 7,116 |

Rounds beyond the first are worth **one relation**, and beyond the fourth,
nothing. At these parameters ECM either splits a cofactor in the first few
curves or does not split it at all, so a 192-curve budget is doing ~8 curves of
useful work. (Later rounds are not idle -- `dead` rises from 246,104 to 274,947
as records are proven unsplittable -- they just do not yield.)

**This is a property of this job's parameters, not a theorem.** On a job where
curves past the eighth do split things -- a larger B1, harder cofactors -- the
differing sigmas would matter and the two configurations would diverge. The
validation is c183 at B1 2000 / B2 60000 over 288 q.

#### What the launch bound costs when it cannot be met

8k set the bound at 750 ms. Over 288 q the two configurations diverge sharply,
and the reason is worth stating plainly:

- **8x24 meets it.** The chunker descends 15,360 -> 7,680 -> 3,840 and settles
  with launches around 400 ms. 277.4 ms/q -- *faster* than the 465 ms/q this
  configuration cost before the bound existed.
- **48x4 cannot.** One curve is ~92 ms, so 48 of them is ~4.4 s no matter how
  the records are sliced. The controller descends 15,360 -> 7,680 -> 3,840 ->
  1,920, the no-progress guard fires and reverts it to 3,840, and it parks
  there at **835.3 ms/q against 465 before**. The guard stops the bleeding; it
  cannot make an unreachable bound reachable.

So the bound is cheap when it can be met and expensive when it cannot, and
which one you get is decided by `--ecm-curves`, not by the chunker. `cofq_init`
now says so at startup rather than leaving it to be discovered:

```
cofactor queue: 48 curves/round is about 4434 ms in one launch, over this
build's 750 ms bound. --cof-chunk splits RECORDS and cannot divide a chain,
so the chunker will subdivide without reaching it and lose throughput doing so.
Shorter launches, same B1/B2: --ecm-curves 7 --cof-rounds 24 (the cap) gives
168 curves in launches of about 705 ms.
```

It advises and does not act: curves-per-round chooses which sigmas run, and
this build does not change that behind the caller's back. The estimate is
**0.0413 ms per (prime power + giant step)** -- the MARGINAL cost of a curve.
The one-curve timings in 8k (110 / 370 / 1461 ms at B1 2000 / 8000 / 32000)
include a fixed per-launch overhead and overstate it by ~9%; 8 curves at
B1 2000 measured 740 ms, i.e. 92.5 ms each over 2,237 units. Calibrated so the
advisory stays quiet on a configuration that does meet the bound.

**Still not adopted as a default.** It is validated, identical and three times
faster on this job, and it remains a mathematics parameter validated on one
composite. Ship it as job settings.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8m. The 24-round cap is rho's, and ECM was paying for it

`--cof-rounds` refused anything above 24, and its own message said why:
"budget << r overflows beyond that". That is **rho's** iteration budget, and
the cap was the only thing stopping ECM from using the one axis that can hold
8k's launch bound.

**ECM never shifts the budget.** The rho launch passes `S->budget << r`; both
ECM launches pass `S->curves`, unshifted. The round index reaches the device
only as `c0 = r + 1`, which selects a 1000-wide sigma block
(`sigma = c0*1000 + cv + 6`). Nothing about ECM overflows with more rounds.

**And the overflow it names is already guarded exactly, where it applies.**
`bench_main` tests `(uint64_t)budget << (rounds-1) > 0xFFFFFFFF` against the
*actual* budget, gated on a side really using rho:

```
$ ./bench ... --cof-rho --cof-rounds 16   # accepted
$ ./bench ... --cof-rho --cof-rounds 20
pipeline cof-budget 65536 with 20 rounds overflows uint32 (or is zero)
```

So at the pipeline's default budget of 65536 the real limit is **16**, not 24 --
the blanket check is simultaneously too loose for rho and irrelevant to ECM,
and the precise check underneath is what actually protects the shift. (An
earlier draft of this section claimed the loose bound let rho silently run
zero-iteration rounds. It does not: the exact check refuses them first. The
claim was wrong and the empirical run is what caught it.)

**Lifted to 1000 rounds for ECM; rho keeps 24 and its exact check.** 1000
because sigma stays in uint32 until ~4.3M rounds, while each round costs five
small kernels plus its launches -- generous without being meaningless.

**What it buys.** To shorten a launch while keeping a curve budget you need
more rounds of fewer curves, and how few depends on B1. At B1 8000 one curve
is ~370 ms, so 24 rounds could not get a 192-curve budget anywhere near the
bound:

| B1 8000, 192 curves | longest launch | cofac/q | relations |
|---|---|---|---|
| 8 curves x 24 rounds (the old ceiling) | 4155 ms | 764.8 | 1114 |
| **2 curves x 96 rounds** | **1594 ms** | **596.2** | 1114 |

Launch down 62%, cofactor stage down 22%, **identical relations** -- the same
result 8l found at B1 2000, for the same reason.

**Also bounded `--ecm-curves` at 994**, which nothing checked before. Sigma
blocks are 1000 wide and indexed by round, so 1000+ curves in a round run into
the *next* round's sigmas and repeat them: arithmetically harmless, silently
wasteful, and easier to hit now that many-rounds-of-few-curves is the
recommended shape.

Metal-side only, through the generator; `bench_main.cu` is untouched, so the
CUDA build keeps 24 and this needs no drift-ledger row. Six gates green.


### 8n. Curves-per-round is derived, not pinned -- and the optimum is 2

8l recommended 8x24 and 8m lifted the round cap that constrained it. Swept
properly, at constant 192-curve budget, `--nq 144`, B1 2000 / B2 60000:

| curves x rounds | longest launch | cofac/q | wall/q | relations |
|---|---|---|---|---|
| 16 x 12 | 2031 ms | 353.5 | 1072.9 | 6,724 |
| 8 x 24 | 1220 ms | 265.3 | 969.4 | 6,724 |
| 6 x 32 | 961 ms | 252.3 | 967.9 | 6,724 |
| 4 x 48 | 756 ms | 212.5 | 912.8 | 6,724 |
| 3 x 64 | 646 ms | 192.8 | 911.6 | 6,724 |
| **2 x 96** | **442 ms** | **180.0** | **881.4** | 6,724 |
| 1 x 192 | 253 ms | 181.0 | 883.4 | 6,724 |

**A bracketed interior minimum at two curves per round**, and 8x24 -- the
configuration 8l validated and recommended -- is 47% worse than it. Identical
relations at all seven points.

The mechanism is 8l's, continued: every round re-compacts the live list, so
splitting the budget finely drops records that have already split before the
expensive later rounds run. At ONE curve the five per-round kernels
(`selflags`, three scan passes, `selscatter`) finally cost more than that
saves, which is what puts the minimum at two rather than at the boundary.

#### The rule, and why it is not "the largest count that fits"

The obvious derivation -- pick the biggest curve count whose launch fits the
750 ms bound -- is **wrong**: at B1 2000 that is 8 curves, and 8 costs 265.3
ms/q against 180.0. The bound is a ceiling, not an objective. So the rule aims
at **2** and lets the bound lower it further, never raise it:

```c
uint32_t fit = 2u;
while (fit > 1u && ms_one_curve * fit > COF_CHUNK_TARGET_MS) fit--;
```

Rounds rise to keep the caller's **total curve budget** unchanged, capped at
8m's 1000. Measured across B1, with the default 48-curve budget:

| | derived | launch estimate | curve budget |
|---|---|---|---|
| B1 2000 | 2 x 24 | ~185 ms | 48 vs 48 |
| B1 8000 | 2 x 24 | ~722 ms | 48 vs 48 |
| B1 32000 | **1** x 48 | ~1419 ms | 48 vs 48 |

At B1 32000 two curves would be ~2.8 s, so it drops to one -- still over the
bound, because one curve is the floor no chunking or round-splitting can go
below (8k). The bound is honoured where it can be and approached where it
cannot.

#### What the default now does

| default budget, 48 curves | launch | cofac/q | wall/q | relations |
|---|---|---|---|---|
| old default, 12 x 4 | 1687 ms | 286.9 | 1005.1 | 6,724 |
| **derived, 2 x 24** | **458 ms** | **160.1** | **860.7** | 6,724 |

Launch **-73%** and under the bound, cofactor stage **-44%**, wall **-14%**,
identical relations.

**It only fires when `--ecm-curves` was not given**, and only when both sides
are ECM. An explicit curve count is the caller choosing which sigmas run and is
never overruled -- that case still gets 8m's advisory instead. The both-ECM
condition is not cosmetic: `cofq_flush` passes one round count to both sides,
so raising it under rho would walk into the `budget << r` overflow that 8m
showed is checked against the caller's rounds, not ours.

All five `cofq_flush` call sites had to move to the derived round count. Four
matched a single pattern and **the fifth wraps its argument onto another line**;
a flush left on `cfg->cof_rounds` would have run the derived curve count
against the caller's rounds and silently shrunk the budget to a twelfth of it.
The generator now asserts that site separately.

**Six gates green, including `cofcheck.sh`'s ~25 pinned relation counts** --
which is the result that matters most here, because the default now runs an
entirely different sigma set and those counts were derived from the CUDA build.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** The optimum at two is a re-compaction effect and should carry; the
exact cost per curve is this machine's and is what the bound is applied to.


### 8o. The derived default validated over 288 q

8n changed what the build does when `--ecm-curves` is not given, so the
comparison that matters is the old default against the new one -- both at the
same 48-curve budget, over the same 288 special-q band as 8l.

| | old default 12c x 4r | derived 2c x 24r |
|---|---|---|
| relations | **13,485** | **13,485** |
| shared (a,b) | 13,485 | 13,485 |
| unique to this run | **0** | **0** |
| records enqueued | 564,696 | 564,696 |
| side 0 split / dead / stuck | 477,071 / 87,625 / 0 | 477,071 / 87,625 / 0 |
| cofactor ms/q | 348.3 | **153.5** |
| wall ms/q | 1068.6 | **856.0** |
| longest launch | 1669 ms | **467 ms** |

**Identical relation sets**, at 56% less cofactor time, 20% less wall, and a
launch inside the 750 ms bound where the old default was more than twice over
it.

#### The interesting part is where they DIFFER

8l's identical result had an easy explanation: both configurations shared the
first 8 sigmas of every round, which is where essentially all splitting happens
at B1 2000. That is **not** true here. Both cover 48 sigmas, but 12x4 takes
`{1006-1017, 2006-2017, 3006-3017, 4006-4017}` and 2x24 takes
`{1006-1007, 2006-2007, ..., 24006-24007}` -- **8 in common, 40 different**. A
cofactor splitting on sigma 1010 is found by the old default and never tried by
the new one. Identical output was not forced this time.

And the side-1 classification shows the sigmas genuinely did differ:

| side 1, 564,696 records | 12c x 4r | 2c x 24r | |
|---|---|---|---|
| split | 14,291 | 14,291 | **identical** |
| dead | 550,145 | 550,176 | +31 |
| stuck | 260 | 229 | -31 |

**The split count is identical while dead/stuck move.** So the two runs tried
genuinely different curves -- this is not a parameter being quietly ignored --
and factored exactly the same cofactors anyway. The 31 records that moved are
ones the old default left unresolved (budget exhausted, `stuck`) and the new
one proved unsplittable (`dead`). Neither is a relation either way, so yield is
untouched.

The reading: **at B1 2000 / B2 60000 the relation set is robust to which
sigmas are tried.** A cofactor with a factor small enough for these bounds is
found by almost any curve; one without is found by none of 48. Which sigmas
you spend the budget on does not matter -- only how many, and how cheaply you
can spend them. That is what makes 8n's re-compaction win free.

**The caveat from 8l stands unchanged and is now more clearly load-bearing:**
this is a property of these parameters. At a B1 where the marginal curve
really does decide relations, the two would diverge, and the argument above
says exactly when -- when a cofactor's smallest factor sits near the edge of
what B1 can reach, so that success depends on the curve rather than on the
bound.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8p. Where curves matter is BELOW B1, not above -- and c183 saturates at 500

8o said the two configurations would diverge "at a B1 where a cofactor's
smallest factor sits near the edge of what B1 can reach". Directionally right,
**and I looked for it in the wrong direction.** Raising B1 moves *further past*
that edge, so every findable factor becomes findable by every curve and the
sigma choice matters *less*. The edge is below.

**B1 32000, sixteen times production, 144 q:** relation sets identical again,
and this time even `dead`/`stuck` match exactly (7,116 / 274,947 / 0 on both),
where at B1 2000 they differed by 31. Sharing only 4 of 48 sigmas changed
nothing whatsoever.

**Because ECM on c183 saturates at B1 ~ 500:**

| B1 | relations | side 1 split | cofac ms/q |
|---|---|---|---|
| 200 | 6,719 | 7,111 | 53.5 |
| **500** | **6,724** | **7,116** | **75.5** |
| 1,000 | 6,724 | 7,116 | 108.0 |
| 2,000 (production) | 6,724 | 7,116 | 170.9 |
| 32,000 | 6,724 | 7,116 | 1,568.3 |

Everything findable is found by B1 500. Production sits 4x past that for 2.3x
the cofactor cost and zero extra relations; B1 32000 is 64x past it for 20x the
cost. **That single fact explains every robustness result in 8l, 8n and 8o** --
ECM was operating so far beyond its binding constraint that which curves ran
could not possibly matter.

**B1 IS NOT BEING CHANGED, and should not be.** It is derived from `lpb` by
`cof_auto_b1` in code shared by every port, so moving it is a tuning decision
for CUDA, HIP and Metal together -- not something a Metal port gets to do on
its own, and not something to discover as a silent divergence later. The
observation is recorded in section 10 for the CUDA side to weigh; this build's
B1/B2 handling is byte-identical to `bench_main.cu`.

**The line this draws is worth stating, because this port already ships one
cofactor default that CUDA does not** (8n's derived curves-per-round). The
difference is what the change can move:

- **Curves-per-round is a SCHEDULE.** The same curve budget, the same B1, the
  same work, in a different launch shape -- and 8n/8o measured the relation set
  identical over 288 q at production parameters, twice. It buys a launch bound
  this hardware needs and CUDA does not.
- **B1 is the SCIENCE.** It decides which cofactors are reachable at all, so
  moving it changes yield, cost and the relation set on every platform.

A port may reshape the schedule to fit its hardware. It may not quietly retune
the mathematics. The saturation measurement is evidence for a decision that
belongs to all three ports at once.

The transferable part is the method -- sweep B1 and find where relations stop
moving -- and the caveat that saturation is tied to this job's `lpb`/`mfb`, so
another composite sits elsewhere.

#### Below saturation the sigmas DO matter, asymmetrically

At **B1 200**, the one regime on this job where the marginal curve decides
relations, 288 q:

| | old 12c x 4r | derived 2c x 24r |
|---|---|---|
| relations | 13,485 | 13,461 |
| shared (a,b) | 13,461 | 13,461 |
| **unique to this run** | **24** | **0** |
| side 1 split | 14,291 | 14,267 |
| side 1 stuck | 115,995 | 116,307 |
| cofac ms/q | 74.0 | **57.7** |

The derived default loses 24 relations (0.18%) and gains none -- a systematic
loss, matching its 24 fewer splits exactly, not a symmetric trade.

**This is the mechanism `cofac.cuh` warned about, measured.** Its record-axis
comment says a curve sub-range "makes a later chunk restart the top composite
with sigmas that cannot split it", because `mz_split` restarts its factor stack
from the original cofactor on every call. A cofactor needing two factors peeled
can be split by 12 curves in one round and not by 2 curves in each of six
rounds: the second round does not resume, it starts over. The warning was
right, and this is where it bites.

**It is still the better default, on the measure that matters:**

| B1 200 | relations/second of cofactor time |
|---|---|
| old 12c x 4r | 182.3 |
| derived | **233.3 (+28%)** |

0.18% fewer relations for 22% less time is more relations per hour, and at or
above saturation there is no loss at all. Production runs at B1 2000, well
above, so the shipped configuration takes the no-loss case. **Anyone running
deliberately below saturation should pass `--ecm-curves` explicitly**, which
suppresses the derivation and gets 8m's advisory instead.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.**


### 8q. Tuning TD: three theories refuted, one cause found, no knob shipped

Phase 7 made TD the port's worst stage -- **7.03x CUDA**, 17.9 ms/q on a
1080 Ti against 125.6 on the M3, turning 2.6% of CUDA's wall into 11.2% of
Metal's. Tuned it. **Nothing shipped, and that is the honest result.**

#### What it is not

| theory | test | result |
|---|---|---|
| `BN_LIMBS` 12 is 384 bits for a 196-bit norm; the loops are 12 long | rebuild at 12 / 8 / 7 | **128.1 / 130.0 / 129.0 ms -- no effect** |
| `TD_TILE` 512 x 32 B = 16 KB of threadgroup memory, half Apple's budget, so ~2 threadgroups per core against a 1080 Ti's six | rebuild at 512 / 256 / 128 / 64 | **132.0 / 130.5 / 131.5 / 138.0 ms -- no effect** |
| TD wants a different threadgroup width from the rest of the pipeline | give `k_td` its own, clean-build A/B | **TD -15%, classify +106%, wall 1.8% WORSE** |

The third is the interesting failure. Swept at band size, TD's own optimum is
**128** threads and classify's is **256**, so 8d's global 256 looked like a
compromise worth splitting. Giving `k_td` its own width does exactly what the
sweep predicted for TD -- 128.9 -> 109.3 ms, -15% -- and **doubles classify**,
18.1 -> 37.3 ms, for a net 1.8% loss. Two clean builds each way, two runs each,
entirely consistent.

Classify does not read that knob. The two stages are **coupled through the
data**: `k_td`'s grid-stride mapping decides which thread writes which
candidate's cofactor, and `k_classify` then reads those arrays. Halving TD's
width doubles each thread's stride and scatters the writes, and classify pays
for it. **8d's "one knob, three stages, three optima" framing assumed the
stages were independent. They are not.** Reverted; no knob is shipped, because
a knob whose default is the value it already had is machinery that earns
nothing.

#### What it is

`bn_divmod_u32_pre` -- the trial-division inner loop -- runs, per limb:

```c
uint64_t cur = (rem << 32) | x->v[i];
uint64_t q   = BN_MULHI64(cur, M);   /* 64x64 -> high 64 */
uint64_t r   = cur - q * d;          /* 64-bit multiply  */
```

That is the only genuinely 64-bit-heavy arithmetic in TD, and **Apple GPUs are
32-bit ALUs that emulate it.** Measured directly with a probe
(`metal-probe`-style, two kernels identical but for width, 65,536 threads x
200,000 iterations, best of 3):

| | ms | vs 32-bit |
|---|---|---|
| 32-bit `mulhi` + multiply | 84.12 | 1.00x |
| **64-bit `mulhi` + multiply** | **401.54** | **4.77x** |

A 4.77x penalty on the operation TD spends its time in, against a card where
`__umul64hi` is an instruction. That is the structural cause, and it is why no
amount of tile sizing or thread counting moved it.

#### The 32-bit division: built, proven exact, and 2x SLOWER

Attempted. It does not work, and the way it fails is the useful part.

Probing the primitives in isolation (65,536 threads x 200,000 iterations, best
of 3) said the target was obvious:

| primitive | ms | vs the cheapest |
|---|---|---|
| Granlund-Moller 2/1 step, 32-bit | 172 | 1.00x |
| plain 32-bit division | 251 | 1.42x |
| the per-limb Barrett (64-bit) | 433 | 2.52x |
| **`bn_recip_u32`'s `(2^64-1)/d`** | **9958** | **57.82x** |

`td_divide_out` calls `bn_recip_u32` **once per prime per candidate**, so a
57x primitive looked like the whole story -- costing more than the division it
exists to accelerate.

Replaced it with Knuth's algorithm D in base 2^16, which needs only 32-bit
division. **The arithmetic is exact**: verified on the host against
`(2^64-1)/d` for **every d in [2, 2^27]** -- 134,217,727 values, the entire
range a factor-base prime can occupy at `alim` 134,200,000 -- plus 4,000,256
random values across the full 32-bit range including the top 256. Zero
mismatches. Probed on the GPU it is **4.46x faster** than the 64-bit division,
2,231 ms against 9,958.

In the actual kernel it made TD **2x slower**: 254.8 ms/q against 128.9.
Relations stayed at 2,282, so it was exact there too -- just slow.

**Why the microbenchmark lied.** Isolated, the routine is 25 lines of 32-bit
work replacing one expensive instruction sequence. Inlined twice per
`bn_recip_u32`, inside `td_divide_out`'s loop, inside `k_td`'s per-candidate
loop, it is a large body with two *data-dependent correction loops* -- so it
costs register pressure and SIMD divergence that a probe with uniform inputs
never sees. And the direction of the result says the reciprocal was **not**
dominant to begin with: something 4.46x cheaper made the stage twice as slow,
which it could not do if it were the bottleneck.

Reverted. `bigint_msl.h` keeps CUDA's `(2^64-1)/d`.

**This is 8h's lesson in a different costume.** There, a one-q band made a
queued stage look like 59% of wall. Here, a primitive probe made one operation
look like the bottleneck. **Both times the isolated measurement was precise,
reproducible and about the wrong thing.** A profile of the real kernel -- which
this port does not have -- is what the next attempt needs, rather than another
plausible primitive.

**What remains, for whoever picks it up:** `d` is a `uint32_t`, `rem < d`, and therefore
both the quotient and the remainder fit in 32 bits -- only `cur` is 64-bit.
This is the standard 2-word-by-1-word division (`udiv_qrnnd` with a 32-bit
inverse), which needs only 32x32->64 products. Done in `bigint_msl.h` it would
be Metal-only and need no CUDA-side change. It is exact arithmetic feeding
relations, so it wants its own care -- but the gates to check it already exist
and are strong: `sievecheck` compares 4,194,304 cells, `cofcheck.sh` pins ~25
relation counts, and Phase 7 can re-run byte-identity against real CUDA.

#### The profile, at last -- and it is neither of the things I tried

`--td` already decomposes the stage on both platforms; nothing had to be
written to get this. Same q, same geometry, **identical work on both --
17,625,929 hits, 7.28 per survivor** -- against a GTX 1080 (sm_61):

| part of `k_td` | GTX 1080 | M3 | M3/CUDA | CUDA share | Metal share |
|---|---|---|---|---|---|
| norm + special-q + 16 large primes | 13.6 | 27.8 | 2.05x | 10.8% | 6.9% |
| **small-prime congruence test** | **81.2** | **324.2** | **3.99x** | 64.4% | **80.1%** |
| division | 31.4 | 52.9 | **1.68x** | 24.9% | 13.1% |
| **norms + trial division** | **126.2** | **404.9** | **3.21x** | | |
| classify | 25.6 | 30.4 | 1.19x | | |

**The division is the best-performing part of the stage at 1.68x**, and this
port tried twice to optimise it. The congruence test is the entire gap: 3.99x,
and 80% of Metal's TD. Had the test matched CUDA, TD would be **1.28x rather
than 3.21x**.

#### The fix: one struct load instead of six field loads

Per prime the test reads **six fields of `tile[e]` separately** -- `m`, `g`,
`magic`, `rt`, `sh`, `cst` -- out of threadgroup memory, and every thread in
the SIMD group reads the same element, so they are broadcasts. Copying the
32-byte `tdsmall_t` once and using the copy lets the compiler issue wide loads
instead of six scalar ones:

| | before | after | |
|---|---|---|---|
| congruence test (standalone `--td`) | 324.2 ms | **223.4 ms** | **-31%** |
| norms + trial division (standalone) | 404.9 ms | **306.3 ms** | **-24%** |
| norms + trial division (pipeline, ms/q) | 130.3 | **106.2 / 106.2 / 107.3** | **-18%** |

17,625,929 hits before and after, 2,282 relations, `sievecheck` /
`cofaccheck` / `cofcheckgate` green. Wall moves about -15 ms/q against a
+/-11 ms band -- real but small, because TD is ~11% of wall.

That takes the test from 3.99x to **2.75x** CUDA and TD from 3.21x to 2.43x.
The same copy was applied to `k_td_record_warp`, which stages the same tile and
read it field-by-field too.

#### A different tile layout: tried, and it is load COUNT, not bytes

If the test were limited by threadgroup *bandwidth*, shrinking the tile entry
should help. `tdsmall_t` is 32 bytes and `recip` is 8 of them, read only when a
prime actually divides -- 7.28 times in 3,633 primes, **0.2% of iterations**.
So the tile was rebuilt to stage only the six hot fields (24 bytes) with
`recip` fetched from device memory on the rare hit: 25% less threadgroup
traffic and 25% less footprint, at the cost of a second struct type and two
code paths for `recip`.

| | test, ms |
|---|---|
| six field reads (original) | 324.2 |
| one 32-byte struct copy | 223.4 |
| **24-byte hot-only tile** | **221.7 / 219.9 / 222.2** |

**About 1%, inside the noise.** Reverted -- the extra type and the split
`recip` path were not earning. And the negative is informative: cutting a
quarter of the bytes changed nothing while cutting six loads to one saved 31%,
so **this loop is limited by the number of threadgroup accesses, not by their
width**. A layout that packs the six fields into a single 16-byte load is
therefore the shape worth trying next, not a smaller one -- `m`, `rt` and `cst`
are all below the 2^15 small-prime bound and `g`, `sh` are tiny, so six fields
plausibly fit in four words. That changes the table the host builds, so it is a
larger change than anything above and is left as the next step.

#### The 16-byte packed layout: built, and it is a wash

The measured field maxima on c183, over every entry of both sides, say the
packing is comfortable: `m <= 32,749`, `rt <= 32,382`, `cst <= 16,384` -- all
15 bits -- with `g <= 19` and `sh <= 14`. So the six hot fields fit one `uint4`:

```
w.x = magic              w.z = cst | (g  << 16)
w.y = m | (rt << 16)     w.w = sh
```

`recip` cannot fit, so it moved out. Built it, with a host-side check that
**refuses rather than truncates** if a job's small-prime bound ever exceeds
16 bits, since those widths are a property of this job and not a theorem.

| | test | division | TD total |
|---|---|---|---|
| struct copy (shipped) | 224.1 | 54 | 301-306 |
| packed, `recip` from device | 201.7 | **76.5** | ~301 |
| packed, `recip` in its own threadgroup array | 200.5 | **70.8** | 299-303 |

**The packing works -- the test drops 10%, exactly as the load-count theory
predicted -- and it is cancelled every time by the division.** Moving `recip`
out of the tile costs the division 54 -> 76.5 ms, and giving it back its own
threadgroup array recovers only a third of that. Standalone TD is 301 ms either
way, against 301-306 for the struct copy: **a wash.**

In the pipeline it is worse than a wash: `norms + trial division` goes
**106.9 -> 199.5 ms/q**, on the `RECORD=1` and warp-recording variants the
standalone path never exercises. Relations stayed at 2,282 and wall did not
move, but a measured stage doubling is not something to ship on the strength of
a standalone number that says "no change".

Reverted. The tile keeps the 32-byte struct copy.

**What that leaves.** The load-count theory survives its own prediction -- one
`uint4` really is ~10% better than one 32-byte struct copy for the test -- but
the win is smaller than what removing `recip` from the tile costs, and this
port has no way to have both without a third layout. The honest summary is that
**the tile layout is now within ~10% of whatever is achievable this way**, and
the remaining gap is elsewhere. ~~Anyone resuming should profile the RECORD=1 variant specifically.~~
**That handoff note was wrong -- see below.**

#### Profiling RECORD=1: the pipeline does not run it

The note above said the pipeline's hot TD pass is `RECORD=1` and that the
standalone `--td` path measures something else. **It is the other way round.**
`pipe_td_perq` -- the per-q, per-side loop the "norms + trial division" timer
brackets -- launches `k_td_1_0_0_*`: **`RECORD=0`, the same variant the
standalone runs.** `RECORD=1` appears only in `pipe_td_verify`, which runs on
the first q of a band and is skipped by `--no-td-verify`, and in the small
recording pass.

Measured, `--nq 24`, Metal, device time per q:

| TD + classify sub-stage | ms/q | share |
|---|---|---|
| rank scan | 2.411 | 1.3% |
| emit (x,a,b) | 2.963 | 1.6% |
| survivor filter | 0.121 | 0.1% |
| resieve + scatter | 48.840 | 26.7% |
| **norms + trial division** -- `k_td_1_0_0_*`, **RECORD=0** | **105.250** | **57.5%** |
| classify | 18.287 | 10.0% |
| joint accept + compact | 0.280 | 0.2% |
| **record candidate factorisations** -- **RECORD=1** | **4.980** | **2.7%** |

**`RECORD=1` is 2.7% of the stage.** It could be free and TD would barely move.
The scattered `fac[t * TD_FMAX + nf]` writes it performs -- 256 bytes of stride
per thread, the thing that made it a plausible suspect -- are not worth chasing.

`SLABBED` was the other candidate, since the pipeline runs `SLABBED=1` and the
standalone `SLABBED=0`. Measured at the same geometry: **105.7 ms/q slabbed
against 94.1 unslabbed, about 12%** -- real, but nothing like the 2x the packed
layout regressed by. (Unslabbed also costs 1358 ms/q of wall against 981, which
is 8c all over again, so it is not an option regardless.)

**So the packed layout's pipeline regression remains unexplained**, and it is
neither of the two things this section proposed. What the profile does settle is
where the remaining time actually is: `norms + trial division` at 57.5% and
**`resieve + scatter` at 26.7%**, which no measurement in this port has ever
looked at.

**The remaining 2.75x is still unexplained** and is now the whole of TD's gap.

#### A correction to this section's own earlier reasoning

An earlier draft cited `bigint.cuh`'s comment -- "30 ms of a 43 ms kernel
[division] against 13 ms for the congruence tests" -- as evidence that CUDA's
shape was the mirror of Metal's. It is not. **On this card and this job the
test dominates CUDA too**, 64.4% against the division's 24.9%. That comment
describes a different configuration and should not have been read as a
statement about this one. The two platforms agree about which part of TD is
expensive; they disagree about by how much.

#### Two latent bugs found while doing this, both fixed

1. **`MSLFLAGS` was not passing `-DBN_LIMBS`.** `bigint_msl.h` declares `bn_t`
   with it and so does the host, but only `HOSTFLAGS` carried it -- so the
   device silently kept the header's `#ifndef` default of 12 while the host
   took the Makefile's value. Identical today at 12; a desynced `bn_t` the
   moment anyone changed it, which is a wrong answer and not a compile error.
2. **Lifted `#define`s were emitted bare.** `gen_msl_headers.py` and
   `gen_td_host.py` copy `TD_TILE`, `TD_FMAX` and friends across from
   `td.cuh` by name, unguarded -- so a bare copy would override a `-D` of the
   same name and desync host from device. Both lifters now wrap every lifted
   define in its own `#ifndef`.

`TD_TILE` is now `#ifndef`-guarded in `td.cuh` with its default unchanged, and
plumbed through both compiles, so the sweep above is reproducible even though
its answer was "no effect".

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display.** Six gates green, plus the CUDA-side `slabcheck`, since `td.cuh` was
touched.


### 8r. resieve + scatter is not the problem: 1.63x

8q ended by pointing at `resieve + scatter` -- 26.7% of the TD stage and never
measured in this port. Measured now, same geometry and settings on both, GTX
1080 against the M3, warm (a cold first run reads 45-67 ms on the card and is
not the number):

| TD sub-stage, `--nq 24` | GTX 1080 | M3 | M3/CUDA |
|---|---|---|---|
| **resieve + scatter** | **30.56** | **49.4** | **1.63x** |
| norms + trial division | 23.61 | 107.1 | **4.49x** |
| classify | 13.7-19.3 | 18.3 | ~1.1x |
| record candidate factorisations | 5.39 | 5.0 | 0.93x |

**1.63x is among the best ratios in this port** -- the same neighbourhood as
the division's 1.68x and better than the sieve's 1.74x. `resieve + scatter`
looked like a target only because it is large in absolute terms; relative to
the card it is already fine. The gap is still `norms + trial division`, alone,
at 4.49x.

#### The unroll knob the slabbed path never had

`td.cuh` documents `k_resieve_scatter` as **latency-bound on dependent summary
probes**, with `UNROLL` existing to multiply memory-level parallelism -- "the
fix for latency is more loads in flight, not fewer loads". It also records the
NVIDIA measurement that established this: making the summary finer, so 96.5% of
probes were rejected earlier, did *not* help, which is what identified latency
rather than probe count as the cost.

The Metal build instantiated `UNROLL` 1, 2, 4 and 8 for the **unslabbed**
kernel and **only 4 for the slabbed one** -- and the pipeline is always slabbed
at production geometry. So 4 was not a choice there; it was the only symbol
that existed. Added slabbed 1/2/8/16, made the depth selectable, and swept:

| `RESIEVE_UNROLL` | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| resieve ms/q | 48.7 | 50.0 | 51.0 | 49.9 | 49.8 |

**Flat across a 16x range**, 1,114 relations throughout. So on Apple this kernel
is *not* latency-bound the way it is on NVIDIA: putting sixteen times more
summary probes in flight changes nothing. Whatever limits it here, more
memory-level parallelism does not relieve it -- and at 1.63x there is little
left to relieve.

**Reverted.** Five extra kernel instantiations and a knob are not worth
shipping for a curve that is flat and a ratio that is already good. Recorded
here so the next person does not rediscover either the lock or the flatness.

#### Inside `norms + trial division`, and the ratio that does not transfer

The pass is already decomposed, by 8q, on the standalone `--td` path:

| inside `norms + trial division` | GTX 1080 | M3 | M3/CUDA |
|---|---|---|---|
| norm + special-q + 16 large primes | 13.6 | 27.8 | 2.05x |
| **small-prime congruence test** | **81.2** | **223** (after 8q's fix) | **2.75x** |
| division | 31.4 | 52.9 | 1.68x |

**The congruence test is the gap inside the pass, as it is inside TD.**

But the pass's ratio is not the same in the two places it gets measured:

| | CUDA | Metal | ratio |
|---|---|---|---|
| standalone `--td`: one q, ~16M candidates, one launch | 126.2 | ~301 | **2.39x** |
| pipeline: ~270k candidates over 8 slabs x 2 sides | 23.6 | 106.5 | **4.49x** |

Same kernel variant, `k_td_1_0_0_*`. **Metal does relatively worse at the
pipeline's scale**, where a launch sees ~17k records against a 131k-thread
grid, than at the standalone's, where one launch sees 16M.

Two candidates tested, neither sufficient:

- **Slab count** -- fewer slabs, bigger launches: 94.6 / 87.1 / 88.9 / 95.3
  ms/q at 1 / 2 / 4 / 8 slabs. **About 9%**, shallow minimum at 2-4.
- **`SLABBED` itself** (8q): 105.7 slabbed against 94.1 unslabbed, **~12%** --
  and unslabbed costs 1358 ms/q of wall against 981 anyway.

**So the scale dependence is real and mostly unexplained.** Decomposing the
pipeline's pass directly -- swapping its timed launch for the `DIVIDE=0`
instantiation, to separate test from division at pipeline scale -- **does not
work**: the pipeline depends on that division downstream and never reaches the
timer. Doing it properly needs instrumentation that runs the variants side by
side rather than substituting one for the other.

#### The side-by-side instrumentation, and what it says

Done. Three launches per TD call, gated on `TD_PROBE`, run **before** the real
one so it overwrites everything they touch and results are unaffected:
`nsm = 0` (norm + special-q + large primes), `DIVIDE = 0` (adds the congruence
test), and the real launch (adds the division). Differences give the split.
Applied to both builds; on CUDA the template instantiates implicitly, on Metal
it needed `k_td<0,0,0,true>` added.

| share of `norms + trial division` | GTX 1080 | M3 |
|---|---|---|
| norm + special-q + large primes | 8.3% | 14.5% |
| **small-prime congruence test** | **75.8%** | **70.8%** |
| division | 15.9% | 14.7% |

**The composition is the same on both platforms.** The test dominates equally,
the division is ~15% on each. So Metal's 4.49x on this pass is **not**
concentrated in one component -- it is a broadly uniform slowdown, and there is
no single sub-step to attack. That is a different answer from "the test is the
gap", which is what the standalone decomposition suggested, and it is the more
useful one: it says the remaining TD gap is the kernel's overall efficiency on
this hardware, not a hot spot.

**Two things this probe got wrong before it got it right**, both worth
recording because both looked like success:

1. **`k_td<0,0,0,true>` does not exist.** The Metal instantiation list has
   `('0','0','0','false')` only, and the pipeline is always slabbed. Launching
   the missing name returned in ~0.01 ms, which read as "the test costs
   nothing" -- a *negative* test time in the first run's arithmetic. A missing
   kernel fails fast and looks like a fast kernel.
2. **`nhit = nullptr` deletes the test.** With no observable sink, the whole
   congruence loop is dead code and the compiler removes it. The probe has to
   pass a real counter or it measures the norm twice.

#### The candidate-count discrepancy, explained

The probe's two runs disagreed about how much they processed -- CUDA 1,078,042
over 8 launches, Metal 3,772,546 over 88, on the same `--nq 4` band. **Two
independent causes, and neither is a disagreement about the work.**

**The launch count is slab auto-calibration.** 8g's calibration runs three
throwaway single-q bands before the real one, and each is a full pipeline pass
with its own TD launches. Disabling it with `--slab-j` collapses the count
exactly:

| | launches |
|---|---|
| Metal, calibration on (default) | 88 |
| Metal, `--slab-j 16384` (calibration off) | **8** |
| CUDA | **8** |

So 80 of the 88 were calibration probes, not the band. Anything that counts
per-launch work on the Metal build **must** pin `--slab-j`, or it is measuring
the calibrator as well as the job.

**The per-launch `n` is the two builds slabbing differently.** At the same
`--region 13` they disagree about the slab plan, because `SLAB_PERF_REGIONS`
is 8192 here (8c) and 32768 on CUDA:

| | slabs | n per launch |
|---|---|---|
| CUDA | **2** | 134,755 |
| Metal, forced to 1 slab | **1** | 269,360 |

Exactly the 2x observed. And the totals agree:

| two-sided primitive survivors/q | CUDA | Metal |
|---|---|---|
| | 269,360 | 269,611 |

**0.09% apart.** The probe was comparing two different slab decompositions of
identical work, which is why per-candidate normalisation from it was
meaningless -- not because the builds disagree about anything. The
within-platform shares stand, and the per-q stage timers (23.6 vs 106.5 ms,
4.49x) remain the sound cross-platform number.

**The general lesson for this port's measurement, and it has now bitten
twice.** A Metal-side per-launch or per-candidate measurement is comparing
against a CUDA build that slabs differently and calibrates where Metal does
not. Neither difference is visible in the number being compared. Pin
`--slab-j` on both sides before normalising anything by launch or candidate.

**Measured on a 10-core M3 in a fanless MacBook Air that also drives the
display, against a GTX 1080 in an NRP k8s pod.**


## 9. Drift ledger — CUDA-side changes made for this port

| date | CUDA file(s) | change | verified how |
|---|---|---|---|
| 2026-09-14 | `bench/cofcheck.sh` | `head -c -1` (all but the last byte) is a GNU coreutils extension that BSD `head` rejects outright; falls back to `dd` where it is unsupported | Ran on macOS: the case it guards ("unterminated candidate file") passes, and the 43 cases before it were already passing when it aborted the script. No behaviour change where GNU `head` exists — the fallback is only taken when `head -c -1 /dev/null` itself fails. **Not run on Linux**, so "no change there" is by inspection of the probe, not by execution. |
| 2026-09-14 | `bench/fbgpucheck.sh` | `sha256sum` (coreutils) falls back to `shasum -a 256` where absent, so the same script is the gate on macOS instead of being forked | Ran on macOS: 19/19 cases pass including the publish-guard case that uses the hash. No behaviour change where `sha256sum` exists, which is every Linux box the CUDA build runs on — the fallback is only taken when the command is missing. **Not run on Linux**, so "no change there" is by inspection of a two-branch `command -v` test, not by execution. |
| 2026-09-14 | `bench/slab.h` | wrapped `#define SLAB_PERF_REGIONS 32768u` in `#ifndef`/`#endif` so a build can override it. **The default is unchanged**: a build that passes no `-D` sees the identical token it saw before. Only the Metal build overrides it, to `8192u`, because its `--region` default of 13 makes CUDA's region *count* mean a quarter of CUDA's slab size in positions -- see section 8c. | `make slabcheck` passes against the edited header. That pass is evidence rather than a tautology because the gate is *sensitive* to this constant: the same `slabtest.cpp` recompiled with `-DSLAB_PERF_REGIONS=8192u` fails on the first pinned row (`plan 0 got jmax=2048 n=2 enabled=1; want 4096/1/0`). The pinned rows do exercise the policy, and they still see 32768. |
| 2026-09-15 | `bench/runlog.c`, `bench/runlog.h` | added `int g_runlog_quiet` (default 0) and a `if (!g_runlog_quiet)` guard around `runlog_warn`'s **stderr half only**; the log-file half is untouched. Verbatim from `hip-port`. **No CUDA-side code sets it**, so the CUDA build sees an unconditional `0` and identical behaviour. | Compiles clean in the default build (`make runlog.o`, `-Wall -Wextra`). Behaviour unchanged by inspection of a one-line guard on a variable no CUDA translation unit writes; **not exercised on a CUDA build**, since there is no nvcc on this machine. |
| 2026-09-15 | `bench/boinc_support.cpp`, `bench/bench.h` | added `bench_boinc_progress_suspend(int)` and a `if (progress_suspended) return;` early-out in `bench_boinc_fraction_done`, placed **before** the monotonic high-water mark. Verbatim from `hip-port`. Nothing in the CUDA build calls the setter, so `progress_suspended` is permanently 0 there and the early-out never fires. | `make -f Makefile.metal boinccheck`: compiles `boinc_support.cpp` with `-DHAVE_BOINC` against a stub client API and drives both cases in separate processes. The control, run first, reproduces the HIP port's field bug (task pinned at 99%); the gate shows the suspend prevents it while preserving monotonicity. Also compiles clean both with and without `HAVE_BOINC` (`-Wall -Wextra`). **Updated 2026-09-15 (plan 9a): the real BOINC SDK is now built here** (8.3.0, arm64, static, `minos 13.0`) and `boinc_support.cpp` compiles against its real headers with zero warnings and links against its real archives. That is a stronger check than the stub for the *link*, but still not a client: no `init_data.xml`, so the stub remains the only thing exercising the progress logic. |
| 2026-09-15 | `bench/cofcheck.sh` | build detection from `--help` (`select Metal device` vs `select CUDA device`, refusing to guess if neither), and the `--ecm-b1 400000` case inverted on Metal to assert a refusal instead of an acceptance. **CUDA's path is unchanged**: `IS_METAL=0` takes the original branch verbatim. | Ran on Metal: 54 PASS / 0 FAIL / exit 0, with `large B1 with derived B2 -> refused, as Metal must`. The case it replaces crashed WindowServer twice on this machine. **Not run on a CUDA build** -- the CUDA branch is unchanged by inspection of a two-way `if`, not by execution. The HIP port skips this case for the same underlying reason (`cofac.cuh`'s own warning block says so). |
| 2026-09-16 | `bench/boinc_support.cpp` | the accepted BOINC coprocessor type, and the two names in its rejection message, became macros selected by `BENCH_BOINC_METAL_GPU`. **Without the flag the CUDA build is byte-identical, message text included** -- `NVIDIA` / `CUDA` / `an NVIDIA`, so `"...is not an NVIDIA one."` survives verbatim. Only the Metal build defines the flag, to accept `apple_gpu`. Fixes a real rejection seen in a field log (plan 9z-g): the client assigned `apple_gpu` and the app refused it as non-NVIDIA, then reported that nothing had been assigned. | Compiled `boinc_support.cpp` both ways and compared the emitted strings: without the flag, `NVIDIA` and the original message; with it, `apple_gpu` and the Metal wording. Twelve Metal gates green. **Not compiled by nvcc** -- the CUDA branch is unchanged by inspection of an `#ifdef` whose `#else` holds the original literals. |
| 2026-09-15 | `bench/td.cuh` | wrapped `#define TD_TILE 512` in `#ifndef`/`#endif`. **Default unchanged**, so a build passing no `-D` sees the identical token; only the Metal build overrides it, and 8q measured that override to be worth nothing, so it does not. | `make slabcheck` passes; six Metal gates green including `sievecheck` (4,194,304 cells) and `cofcheck.sh` (54 cases). Behaviour change is nil by inspection of an `#ifndef` around an unchanged value. **Not compiled by nvcc** — no CUDA build was run against this edit. |
