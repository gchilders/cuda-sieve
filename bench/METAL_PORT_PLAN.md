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

### Phase 8 (continued) — original scope
`log_region <= 13` default; threadgroup sizing measured from scratch —
`bench.h`'s 32-thread `k_fill_atomic` result is an NVIDIA L2-bound finding
and must not be inherited; `--cof-chunk` retargeted at the macOS display
watchdog; slab sizing for a 16 GB UMA carveout shared with the OS.

**Gate:** documented, with every number labelled as measured on a
correctness vehicle.

### Phase 9 — Packaging
`Makefile.metal`, metallib embedding, arm64 BOINC.

---

## 10. Open questions for the CUDA side

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

## 9. Drift ledger — CUDA-side changes made for this port

| date | CUDA file(s) | change | verified how |
|---|---|---|---|
| 2026-09-14 | `bench/cofcheck.sh` | `head -c -1` (all but the last byte) is a GNU coreutils extension that BSD `head` rejects outright; falls back to `dd` where it is unsupported | Ran on macOS: the case it guards ("unterminated candidate file") passes, and the 43 cases before it were already passing when it aborted the script. No behaviour change where GNU `head` exists — the fallback is only taken when `head -c -1 /dev/null` itself fails. **Not run on Linux**, so "no change there" is by inspection of the probe, not by execution. |
| 2026-09-14 | `bench/fbgpucheck.sh` | `sha256sum` (coreutils) falls back to `shasum -a 256` where absent, so the same script is the gate on macOS instead of being forked | Ran on macOS: 19/19 cases pass including the publish-guard case that uses the hash. No behaviour change where `sha256sum` exists, which is every Linux box the CUDA build runs on — the fallback is only taken when the command is missing. **Not run on Linux**, so "no change there" is by inspection of a two-branch `command -v` test, not by execution. |
