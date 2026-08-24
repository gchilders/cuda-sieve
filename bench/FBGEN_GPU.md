# Experimental GPU factor-base generator benchmark

`fbgen_gpu` is both the standalone timing/debug harness and the implementation
behind production in-process algebraic factor-base generation.  In pipeline
mode, `bench` now calls the same generator when `--fb1` is omitted, avoiding a
multi-gigabyte BOINC roots-file download.

## Build

For iteration on the GPU in the machine, use a native-only target so this small
benchmark does not inherit the long multi-architecture build time of the full
sieve:

```sh
cd bench
make GPU_ARCH=native fbgen_gpu
```

A named architecture works as well, for example `GPU_ARCH=89` for Ada.

On native Windows, `build_windows.bat` always compiles the library form needed
by `bench.exe` for in-process generation.  Build the standalone cache utility
only when needed:

```bat
cd bench
build_windows.bat fbgen_gpu
fbgen_gpu.exe --poly JOB.job --lim 600000000 --maxbits 16 --out JOB.roots1
```

The standalone target is intentionally optional because it recompiles the same
CUDA translation unit with the CLI and text serializer enabled.

## Run

The job file supplies `alim` and `rlim` by default:

```sh
./fbgen_gpu --poly test.job
```

To price the large symmetric limits discussed for BOINC:

```sh
./fbgen_gpu --poly test.job --lim 600000000
./fbgen_gpu --poly test.job --lim 1700000000
./fbgen_gpu --poly test.job --lim 4000000000
```

Or set the sides independently:

```sh
./fbgen_gpu --poly test.job --alim 4000000000 --rlim 1700000000
```

The default segment contains 8,388,608 odd candidates (about 16.8 million
integers). `--segment-odds N` changes that working set.  Because compact root
counts/offsets are 32-bit and one degree-8 prime can contribute nine entries,
`N` is limited to 477,218,588; larger values are rejected before allocation.
`--check N` controls how many generated entries are copied back and validated
against the exact decimal polynomial on the host; the default is 1000.

The algebraic arithmetic backend defaults to 32-bit Montgomery reduction.  A
CUDA `uint64 % uint32` backend is retained for same-binary A/B measurement:

```sh
./fbgen_gpu --poly test.job --lim 600000000 --alg-backend montgomery
./fbgen_gpu --poly test.job --lim 600000000 --alg-backend legacy
```

Both backends go through the same root validation and output compaction.

Generic polynomial multiplication is now the default square strategy because
it beat the triangular symmetric square on the RTX A4000 despite doing more
modular products.  The old square remains available for regression:

```sh
./fbgen_gpu --poly test.job --lim 250000000 --alg-square generic
./fbgen_gpu --poly test.job --lim 250000000 --alg-square symmetric --alg-kernel generic
```

The root kernel itself has two fixed-capacity specializations plus the original
generic implementation:

```sh
./fbgen_gpu --poly test.job --lim 250000000 --alg-kernel auto
./fbgen_gpu --poly test.job --lim 250000000 --alg-kernel cap6
./fbgen_gpu --poly test.job --lim 250000000 --alg-kernel cap8
./fbgen_gpu --poly test.job --lim 250000000 --alg-kernel generic
```

`auto` uses cap6 for degree <=6 and cap8 for degree 7-8.  The shared
`bench.h` limit remains `BENCH_MAX_DEGREE=8`; higher degrees are intentionally
not added here.  The fixed-capacity path instantiates the coefficient capacity
at compile time and removes the generic factor stack: it isolates one root at a
time and divides that linear factor from the remaining squarefree polynomial.
`--alg-kernel generic` remains an explicit performance/correctness reference,
not an escape hatch past the degree-8 parser limit.

The report separates:

- odd candidate generation;
- segmented Eratosthenes composite marking;
- prime compaction;
- rational-root generation;
- algebraic root finding;
- algebraic count scan and `(p,r)` compaction;
- whole startup wall time.

## Complete factor-base path

Default benchmark mode still times **ordinary-prime entries only**, so its
stage timings remain directly comparable with the earlier prototype runs.  The
production/verification mode is complete:

```sh
./fbgen_gpu --poly JOB.job --lim ALIM --complete --maxbits 16 --scale SCALE
```

Ordinary roots come from the fixed-capacity GPU kernel (`CAP=6` for degrees
through six and `CAP=8` for degrees seven and eight).  A prime is rerun through
the native exact CPU `fbgen` code when it can contribute a proper power
(`p^2 <= 2^maxbits`) or when the GPU detects a ramified affine root or a
projective root.  Before an ordinary `q=p` entry can leave the production GPU
kernel, `p` is also checked by an independent deterministic 32-bit
Miller-Rabin test (bases 2, 7, and 61).  This keeps the generated-factor-base
validator fail-closed if the segmented GPU prime sieve ever regresses.  The
exact branch supplies the exact `(nexp,oldexp)` marginal-log metadata and all
Hensel rungs; the streams are merged in modulus order directly into `fb_t`.

For an entry-for-entry oracle comparison against text `fbgen` output, use:

```sh
./fbgen --poly JOB.job --lim ALIM --maxbits 16 --out ref.roots1
./fbgen_gpu --poly JOB.job --lim ALIM --complete --maxbits 16 \
    --scale SCALE --compare-fb ref.roots1
```

`make fbgpucheck` now sweeps every supported polynomial degree 1..8 (including
the CAP=6/CAP=8 dispatch boundary), representative GNFS/SNFS cases, and the
legal `lim=2/3` boundaries.  For each equivalence case it requires the GPU and
CPU generators to emit byte-identical roots files and then loads both through
`fb_load_cado()` and compares the resulting `fb_t` arrays.  It also gates
oversized-segment rejection and verifies that a failed `--compare-fb` leaves
the previously published output untouched.  The target is intentionally
separate from the CPU-only `make check` because it requires an NVIDIA device.

### Reusable roots-file output

For a campaign that reuses one polynomial across many `bench` invocations, the
standalone tool can pay the GPU generation cost once and cache the exact native
text representation:

```sh
./fbgen_gpu --poly JOB.job --lim ALIM --maxbits LOGI --out JOB.roots1
./bench --pipeline --cofactor --poly JOB.job --fb1 JOB.roots1 ...
```

`--out` and in-process `afb_build_gpu()` now share one internal complete-GPU
generation engine for device setup, segmented prime sieving, primality gating,
CAP=6/CAP=8 root generation, scan/scatter, exact-prime classification, and
cleanup.  They differ only in the host consumer: production appends the roots to
an in-memory `fb_t`, while `--out` serializes them.  This keeps correctness fixes
from drifting between two near-identical GPU loops without routing the writer
through `afb_build_gpu()` or allocating the full host `fb_t`.

With `--compare-fb`, both the byte comparison and loaded-`fb_t` comparison are
performed on `FILE.part` **before** the atomic replacement, so a failed
comparison cannot overwrite an existing good cache.  The writer streams each
ascending prime's entries directly to a staging file and atomically replaces
`JOB.roots1` from `JOB.roots1.part` only after successful completion.  The
replacement goes through the platform abstraction (`rename` on POSIX,
`MoveFileEx(..., REPLACE_EXISTING | WRITE_THROUGH)` on Windows).  Ordinary
GPU roots and
exact exceptional entries are both serialized by the same formatter used by
native CPU `fbgen`, so the intended contract is byte identity, not merely
loader equivalence.

For a direct oracle check:

```sh
./fbgen --poly JOB.job --lim ALIM --maxbits LOGI --threads 12 --out cpu.roots1
./fbgen_gpu --poly JOB.job --lim ALIM --maxbits LOGI \
    --out gpu.roots1 --compare-fb cpu.roots1
```

With `--out`, `--compare-fb` first compares the raw files byte-for-byte and then
loads both at `--scale` (default 1.0) and compares their complete `fb_t`
contents.  The benchmark-only `--alg-backend`, `--alg-square`, and
`--alg-kernel` A/B switches do not alter the production `--out` generator.

This cached-file path is compiled only into the standalone `fbgen_gpu` binary.
The `FBGEN_GPU_LIBRARY` object linked into production `bench` excludes all text
serialization code, so adding reusable file output does not add a branch or
I/O work to the normal no-`--fb1` in-memory generation path.

The algebraic root finder follows the same mathematical construction as
`fbgen.c`: find the linear part with `gcd(F, x^p-x)`, then deterministically
split that squarefree product.  Its default backend keeps all polynomial
coefficients in a 32-bit Montgomery domain with `R = 2^32`.  Per-prime setup
uses Newton inversion plus 64 modular doublings; repeated products use only
32-bit low/high multiplies and add/subtract correction.  Multi-limb decimal
coefficients are reduced directly into Montgomery form, so the hot algebraic
path contains no integer division or remainder.  The exponentiation path defaults to generic polynomial multiplication for
squaring and skips the known `q*1` multiply when cancelling the leading term of
a monic divisor.  A symmetric-square implementation remains available only as
an A/B/reference option because it was slower on the A4000.

The pre-Montgomery A4000 baseline on the degree-6 `test16_600.poly` was:

| symmetric limit | primes | algebraic-root stage | wall startup |
|---:|---:|---:|---:|
| 250M | 13,679,318 | 7.448 s | 7.519 s |
| 600M | 31,324,703 | 17.99-18.03 s | 18.25-18.28 s |
| 1.7B | 84,163,019 | 51.046 s | 51.577 s |
| 4.0B | 189,961,812 | 120.290 s | 121.394 s |

At 4B the algebraic-root stage was 99.6% of summed GPU-stage time, which is why
the arithmetic backend is the first optimization rather than the segmented
prime sieve.  Use `--alg-backend legacy` to reproduce that baseline on the same
binary and `montgomery` to measure the optimized path.

The A4000 square-strategy A/B at 250M (degree 6) measured:

| arithmetic | square | algebraic-root stage |
|---|---|---:|
| legacy remainder | generic multiply | 7.601 s |
| legacy remainder | symmetric | 8.298 s |
| Montgomery | generic multiply | **6.969 s** |
| Montgomery | symmetric | 7.211 s |

So Montgomery + generic is the default baseline for the fixed-capacity work.
The preceding generic kernel compiled for sm_86 with 55-62 registers, no ptxas
spill loads/stores, but an 800-byte stack frame per thread.  The cap6/cap8
experiment is specifically aimed at that local-memory footprint.

A separate CPU calibration is AS276/C208 near 268.4M, where the existing CPU
generator took about 44 seconds on 12 host threads.  Complete GPU generation
uses that same CPU Hensel implementation only for the tiny exceptional/power
population and avoids writing or reparsing the text factor base.

On the RTX A4000, the final CAP=6 ordinary-root path measured 3.505 s at 250M,
8.433 s at 600M, 24.375 s at 1.7B, and 57.347 s at 4B; complete-generation
startup should therefore remain dominated by the same GPU root stage.
