# STATUS — what exists now

`RESULTS.md` and `../prototype.md` are lab notebooks: they record findings in
the order they were discovered, including the ones later refuted, because the
refutations are the most useful part. That makes them bad at answering "what
does this thing do today". This file answers only that, and holds nothing that
is not current. **Last updated 2026-08-17.**

## Architecture

One process, both sides, one special-q at a time:

```
per special-q:
  host    reduce the q-lattice, build the three modulus tiers
  device  k_transform    plattice transform of the factor base
          k_fill_atomic  bucket-sieve into regions
          k_apply        norm init + bucket add + threshold scan  ->  survivor bitmap
  (repeat for the other side)
  device  k_intersect_compact   both-sides bitmap AND, gcd filter, rank scan
          trial division, classification, resieve
          cofactorisation (Pollard rho and ECM)
  host    emit relations
```

Two sides run **sequentially through one shared bucket allocation**. There is
no stream concurrency and no second workspace. Every timing in `RESULTS.md` is
one-q-at-a-time.

`--relations NAME` stages to `NAME.part` and renames to `NAME` only when the
band completes. The `.part` is the **durable artifact**, not scratch: after
every cofactor flush it is fsynced and `NAME.part.ckpt` records the next
`(q, rho)`, the byte offset, the relation count, the derived scale/allowance
and a job fingerprint. Rerunning the same command resumes there; `--restart`
discards; SIGINT/SIGTERM and `--stop-file` stop cleanly at the next special-q;
`NAME.lock` refuses a second writer. Verified on a GPU 2026-08-16: a stopped
and resumed band reproduces the uninterrupted one byte for byte, `kill -9` and
a torn final line included. See item 12a.

`--log PATH` appends a run log: a header block naming the commit, argv, job
fingerprint, card, geometry and factor-base convention, then a timestamped
record every `--log-every` seconds (300 by default) carrying progress
alongside `GPU-accounted / wall`, GPU utilisation, board watts and host load.
Item 12b.

### Distributed packaging (BOINC), added 2026-08-15/16

Optional and compiled only under `HAVE_BOINC=1`; the ordinary build has no
BOINC dependency and its wrappers are no-ops. The app initialises and finishes
through the BOINC API, resolves every named input and output through
`boinc_resolve_filename_s()`, requests normal host-thread priority for the
feeder thread, and reports a nondecreasing fraction done at special-q
boundaries. `README.md` is the reference for building and for the workunit
command line.

**Which GPU a task runs on** is read from `init_data.xml`
(`<gpu_device_num>`, via `boinc_get_init_data()`) and **outranks `--device`**,
which is the opposite of this codebase's usual precedence and deliberate: a
`--device` in an app version or workunit template is shared by every task on
every host, so honouring it reinstates the all-tasks-on-card-0 bug the
assignment exists to prevent. `--device` selects only when there is no
assignment. Each task's stderr records which of the two happened, the device
count this process can see, and the card actually used. Startup also refuses an
ordinal past `cudaGetDeviceCount()` with both numbers named, rather than
letting it surface as a bare "invalid device ordinal".

**Reviewed and deployed 2026-08-17**: Greg Childers — who reported the
original all-tasks-on-device-0 failure — signed off on the assignment change,
and a BOINC queue is running with it. That closes item 13 as far as the
application is concerned. There are no checkpoint files in the BOINC sense
either: a suspended process resumes,
but a process that exits and restarts begins the current band again (12a's
sidecar is the repo's own mechanism, not BOINC's).

**The application is Linux-only, and that is a bigger gap than it looks for a
volunteer project.** `boinc_support.cpp` guards its POSIX paths with
`#ifndef _WIN32`, which reads as a Windows build being within reach; it is not.
`ckpt.h` uses `fsync`, `ftruncate`, `kill` and `<unistd.h>` unguarded, and
`runlog.c` adds `<dlfcn.h>` and `getloadavg` on the same terms. Windows is most
of the volunteer host population, so this is worth costing before a wider
deployment rather than discovering per-file; the guards in `boinc_support.cpp`
should not be read as evidence that the rest is close.

## Current size limits and j-slabbing

The cofactor-width work landed on 2026-08-18 and the production pipeline now
also supports rectangles larger than `2^31` positions by **j-slabbing** them.
There is still no unsafe override of the local arithmetic bounds.

| quantity | current limit | immediate reason |
|---|---:|---|
| local sieve slab | `2^31` positions | bucket/bitmap/rank positions remain `uint32_t` |
| full pipeline rectangle | default geometries through `logI 20` | host scheduler splits `J` into safe slabs |
| `lpb` | 64 | a resulting prime is stored in one `uint64_t` (was 32 until 2026-08-17) |
| `mfb` | 128 | the cofactor queue narrows residuals to `mz<4>`; 96 in a `CF_LMAX=3` build |
| large primes per side | 3 | `ceil(mfb/lpb) <= 3` is checked before the run |

With default `J=2^(logI-1)` and `bkthresh=I`, I17 uses 4 slabs, I18 16,
I19 64, and I20 256. `--slab-j N` can force a smaller slab for regression or
memory tuning. The actual largest direct-tested prime also constrains slab
height, so raising `bkthresh` can make the planner choose more slabs.

### Why the lattice walk itself is wide

A local sieve position still fits in 31 bits, but the Franke-Kleinjung reduced
**increment** need not fit in 32. For realistic factor-base primes,
`(j0 << logI) - mi0` or `(j1 << logI) + i1` can exceed `2^32` even on I15/I16.
The earlier `uint32_t plat_t` therefore had a latent wrap: it could create
spurious sieve hits even before whole-area slabbing was needed. `plat_t` now
stores 64-bit increments; the bounded local walk terminates instead of wrapping
when its exact next hit leaves the 32-bit coordinate, and slab continuation is
carried exactly in 64 bits between slab origins.

This widening costs 8 additional bytes per uploaded full-FB entry. Slabbed
runs additionally keep two 64-bit continuation values per entry (16 bytes per
entry total); unslabbed runs allocate no continuation arrays.

### Larger rectangles: memory is now the main constraint

The bucket array and survivor bitmaps are allocated for **one slab** and reused,
so a full I17/I18/... rectangle no longer requires a monolithic `2^33`/`2^35`
position workspace. Area-proportional work still scales with the full rectangle,
and every slab re-enters the factor-base walk, so the number of slabs remains a
runtime consideration even though it no longer multiplies peak bucket memory.

The measured pre-slabbing memory model for the two large allocations remains a
useful per-local-area reference:

| local area | bucket array | three survivor bitmaps | subtotal |
|---:|---:|---:|---:|
| `2^31` | 5.53 GB | 0.81 GB | 6.34 GB |
| `2^32` | 11.06 GB | 1.61 GB | 12.67 GB |

Only the `2^31` row is a production slab size now. The `2^32` row is retained
as the old monolithic projection, not as an allocation the slabbed path makes.

### LPB and MFB are separate widths

`lpb` bounds an *individual resulting prime*. `mfb` bounds the *composite
residual sent to cofactorisation*. Raising one does not inherently require
raising the other:

- Raising `lpb` above 32 required a `uint64_t` representation for split
  primes, unsplit prime residuals, sorting, emission, primality checking, and
  relation reconstruction. It did **not** require four-limb rho/ECM at
  `mfb <= 96`. **Done 2026-08-17** — see "64-bit large primes" below.
- Raising `mfb` above 96 requires `mz<4>` arithmetic even when the actual
  maximum is only 101 bits. Rho, ECM, probable-prime testing, GCD, and exact
  division all pay the wider modular-arithmetic cost. **Built 2026-08-18** —
  see "Four-limb cofactors" below.

The useful dispatch is fixed-width kernels selected on the host, independently
for each side:

| side's `mfb` | cofactor type | status |
|---:|---|---|
| `<= 64` | `mz<2>` | **not built, and deliberately so** — see below |
| `<= 96` | `mz<3>` | built |
| `<= 128` | `mz<4>` | built 2026-08-18 |

The underlying `mz<L>` arithmetic was already templated; what was hardcoded was
the production queue/storage and its launches at `mz<3>`. Runtime
variable-length loops inside a kernel are not the intended design: they would
lose compile-time unrolling and likely increase register pressure. The
criterion is the bit bound, not the label: a 2LP side with `mfb=65` still needs
three limbs.

The `mz<2>` tier is left out on purpose. `CF_LMIN` is 3, so a side with
`mfb <= 64` runs three limbs and pays for a limb it does not use. That is the
same trade the rational queue already made when it went from two limbs to
three: the stage nearly doubled (+83%) and wall clock moved +0.64%, because
that queue is ~1% of a special-q. Reinstating a narrow tier means a third
instantiation to keep in step for a fraction of a percent.

### 64-bit large primes — BUILT AND VERIFIED 2026-08-17

`lpb` up to 64 is supported. The widening was confined to values bounded by
`lpb` rather than by `lim`: the split primes and unsplit prime residuals in the
cofactor queue (`d_sp0/1`, `d_sm0/1`), `mz_split`'s output, the relation
emitter on both the inline and `--cofac` paths, and the reconstruction gate.
The trial-division lists stay 32-bit deliberately — those are factor-base
primes, bounded by `lim`, which is well under `2^32` even for a C208. The
cofactor arithmetic did not change at all: `mfb <= 96` is still `mz<3>`.

The gate needed its own 64-bit arithmetic, which is the part that was not a
retype: `bench_is_prime32` cannot test a 33-bit factor and `bn_divmod_u32_pre`
cannot divide one out, so `cf_is_prime64` (deterministic Miller-Rabin, the
seven bases valid below `2^64`) and `cf_bn_divmod_u64` (128-bit long division,
stepping in 64-bit limbs because a divisor above `2^32` overflows a 32-bit
quotient digit) were added host-side. Without them the gate would have skipped
exactly the factors the change introduces.

Verified three ways: the 1500-q c147 band is **byte-identical** to its
pre-change MD5, so nothing moved at `lpb <= 32`; the c183 golden q at `lpb 33`
emits 58 relations carrying 24 factors above `2^32`, all reconstructing; and
that same file is **refused** when re-read at `lpb 32`, so the bound is
load-bearing rather than decorative. All three are now cases in `cofcheck.sh`.
On a 400-q c147 band at `lpb 33`, 8,810 of 2,075,496 emitted factors exceed
`2^32`, the largest `0x1ffd77b3b`.

What this does **not** cover is `mfb > 96`, which is the four-limb change
described next.

This makes `lpb=33` with `mfb=64/95` a materially smaller extension than
`lpb=35` with `mfb=64/101`: the former needed 64-bit factor outputs but retains
two- and three-limb cofactor kernels; the latter needs a four-limb kernel on
the 101-bit side. Both ratios still ask for at most three large primes, so
`CF_MAXFAC` is not the blocker.

The output-width change itself should be cheap. The TD factor lists remain
32-bit because factor-base primes remain 32-bit; only the much smaller residual
and split-prime arrays need widening. The expensive part is any extra limb in
the repeatedly executed modular arithmetic.

There is one direct measurement of the per-side dispatch opportunity. Widening
the formerly two-limb rational queue to three limbs changed that queue from
2.15 to 3.94 ms/q (**+83%**) while changing the then-current whole pipeline
from 172.65 to 173.75 ms/q (**+0.64%**). Narrow arithmetic matters greatly to
the stage that uses it, but the easy side was only about 1% of wall time. That
makes per-side dispatch sensible work while the queue is already being changed,
not a high-value standalone optimisation.

### Four-limb cofactors — BUILT 2026-08-18, NOT YET TIMED

`mfb` up to 128 is supported, and the width is chosen **per side** at run time
as the narrowest instantiation that holds that side's `mfb`. Everything up to
and including a C194's `mfba 95` still gets 3/3 and is unchanged.

**AS276 (`~/code/ggnfs-distributed/AS276.job`, the C208) is the motivating job
and it resolves to 4/3 with no flag.** Its parameters, and what each one does
here:

| | value | consequence |
|---|---|---|
| `lpbr / mfbr` | 33 / 64 | 2 parts, **3 limbs** — 64 bits, but `CF_LMIN` is 3 |
| `lpba / mfba` | 35 / 101 | 3 parts, **4 limbs** — 101 bits does not fit 96 |
| `rlim / alim` | 181.6M / 268.4M | `lim^2` ~ 2^54.7 / 2^56, both well above `2^lpb` |
| derived allowance | 102.63 bits (side 1) | sits just above `mfba 101`, as it should |
| `log2(maxnorm)` | 205.02 at `logI 15, J 16384` | inside `bn_t`'s 256-bit budget, 51 bits spare |

Two things fall out of that table. The **rational** side is the exact case an
`mz<2>` tier would serve — `mfbr 64` is 64 bits carried in 96 — and it is still
not worth building, for the reason given above: that queue is ~1% of a
special-q. And `lpb >= 33` is **necessary but not sufficient** for a fourth
limb; AS276's rational side has `lpbr 33` and still runs 3 limbs, because
`mfbr` is what decides.

**AS276 was sieved end to end on 2026-08-18** at `logI 15, J 16384`, factor
base from `fbgen --maxbits 15` (44 s, 230 MB). Three special-q, `--cofactor`:

- width line reports `side 0 3 limbs (96 bits), side 1 4 limbs (128 bits)` —
  derived, no flag;
- 134 relations, 44.67 rel/q, 39.00 of them from cofactorisation;
- **134 of 134 rebuild both norms exactly** through `--check-relations`;
- 130 emitted factors exceed `2^32`, the largest `2^34.96` — hard against the
  `lpba 35` ceiling.

That is the end-to-end proof the golden test cannot give: a genuine 4-limb
population, split by the 4-limb kernel, emitted through the 64-bit output path,
and verified against the norms. **No timing was taken** — the ECM job had the
GPU, and a 3-q band amortises the final queue flush over nothing.

**Cross-validated 2026-08-19 against the 1.5B-relation GGNFS corpus for this
job** (`~/code/ggnfs-distributed/AS276/`, already filtered): over the identical
30 special-q and an identical sub-rectangle, **recall 3,044 / 3,045 = 99.97%**,
with 64 of our 4,089 relations existing nowhere in their corpus and all 64
reconstructing exactly. Full method and the re-find analysis in RESULTS finding
69. This is the strongest correctness evidence the 4-limb path has: an
independent siever, an independently filtered corpus, and set membership rather
than self-comparison.

**It also surfaced the one thing the width change actually broke, which is not
the width.** At `--cof-rounds 2 --cof-budget 65536` (the default until
2026-08-19) the same run
lost **389 of GGNFS's 2,846 relations (13.7%)**; `6 / 262144` recovered all but
2, for **+19.3% relations**. Rho's iteration count scales as the square root of
the factor sought, so a 35-bit large prime costs 5.7x a 30-bit one, and the
default was calibrated on the c183's `lpba 32`. An exhausted budget returns
`CF_INCOMPLETE` and the relation is silently dropped — it presents as a yield
hole in the siever, not as a cofactoriser problem. **The schedule around the
splitter needs re-deriving per width class; the splitter itself did not.**

Note `A = 29` there, not the `A = 32` NFS@Home's own geometry for this job
would want; the area limit is a separate, still-open blocker and does not stop
the job being sieved at a smaller rectangle.

What moved:

- `CF_LMAX` / `CF_LMIN` / `cf_limbs_for_mfb` in `bench.h` — the build's width
  range and the mfb → limbs rule. `CF_LMAX` defaults to 4;
  `make CF_LMAX=3` compiles the 3-limb splitter alone.
- `cofq_t.d_c0/d_c1` are **raw limb arrays plus `L0`/`L1`**, not `mz<3> *`. The
  stride is a run-time choice and `mz<3>`/`mz<4>` are distinct types, so the
  array cannot carry one of them in its type. `cf_run_rounds_dyn` is the only
  place that casts back.
- `k_cof_enqueue` is templated on `<L0, L1>` (four cheap instantiations) and
  the standalone `--cofac` batch parses at `CF_LMAX` and narrows per side.
- `--cof-limbs N` / `--cof-limbs0 N` force a side **wider** than its `mfb`
  needs. Narrower is refused, in `resolve_and_check_cofactor_config`, which is the one
  place that has seen both the `.job` file and the command line.

The width invariant is asserted rather than trusted. `cof_classify` rejects a
residual above `mfb` and `resolve_and_check_cofactor_config` refuses an `mfb` above
`32*L`, so nothing can reach the queue too wide for its side — and if it does,
`k_cof_enqueue` marks it `CF_OVERFLOW`, counts it, and `cofq_flush` stops the
run. The failure mode being closed is a *silently truncated* cofactor, which
does not crash: it emits a relation that reconstructs to the wrong norm. Both
previous versions of this array (2 limbs, then 3) had exactly that bug.

**The ceiling that binds first is not the width.** `CF_MAXFAC` caps a split at
3 large primes, so `mfb <= 3*lpb` regardless, and at `lpb 32` that is 96 bits —
precisely what 3 limbs already held. A side needs 4 limbs only once its `lpb`
reaches 33. Going past 128 bits means 4LP, which is `CF_MAXFAC` and
`mz_split`'s `sp + 2 > CF_MAXFAC` stack guard, not another limb.

#### What the correctness gate can and cannot prove without a C208

`cofcheck.sh` gained a width block. The load-bearing case is **byte-identical
output between a 3-limb and a 4-limb run of the same job**. That is a real
proof, not a smoke test: rho and ECM are Montgomery-domain algorithms whose
iteration is `y <- y^2 + c` in the *true* domain regardless of `R = 2^(32L)`
(`c = c0*R`, `y0 = 2*R`), and `gcd(qR^k, n) = gcd(q, n)` because `n` is odd. So
widening a side must change the cost and nothing else. The 2 → 3 limb widening
of the rational side produced the same md5, which is the precedent.

Cases added: the build reports its own width range; `mfb` above `32*CF_LMAX` is
refused; rho at 4/4, rho at 4/3, and ECM-with-stage-2 at 4/3 are each
byte-identical to the 3/3 run; `lpb 33 / mfb 99` **derives** 4/3 with nobody
choosing it, emits relations, and every one reconstructs; and that same shape
with `--cof-limbs 3` is refused.

**All 45 cases pass on 2026-08-18**, `sm_120`, including the four new ones. The
`lpb 33 / mfb 99` case derives 4/3 with nobody choosing it and emits 64
relations that all reconstruct, so the 4-limb splitter is not merely compiled —
it has produced verified relations.

What no gate here can reach is a genuine 4-limb *population*. `lpb 33 / mfb 99`
on the c183 admits a few candidates above 96 bits; a C208 is made of them —
which is what AS276 and its GGNFS corpus were used for (finding 69).

#### The three deployment options — RESOLVED 2026-08-19

All three questions this section used to pose are answered; the numbers are in
finding 70 and the decisions are made.

- **Dynamic per-side selector — CHOSEN, and it is what ships.** Forcing the
  wide shape on a job that does not need it costs **+8.8% of wall on the c183
  and +8.2% on the C194** (a widened queue is ×1.72), which is too much to
  give away for the simplicity of always running wide. The selector costs
  nothing at run time — one host-side switch per flush.
- **"Always 4/3" — rejected**, on those same numbers.
- **Separate executables — rejected as a performance measure**, kept as a
  packaging option. The 3-limb kernels are register-identical in both builds
  (78 / 86 / 122, verified against a `CF_LMAX=3` compile), so a wide build does
  not slow the narrow path. What `make CF_LMAX=3` does buy is **binary size**:
  `.nv_fatbin` 8.99 MB → 4.02 MB, and about half the `ptxas` time. That is an
  argument about BOINC distribution, not about sieving.

#### Register cost of the fourth limb — measured 2026-08-18, no GPU needed

The register question was the one genuinely new risk: ECM stage 2 holds
`mpt<L> baby[CF_ECM_NBABY]` plus four more points, which at `L = 4` is 64
registers of live state before any working value. `ptxas -v` answers it without
touching the card. `sm_120`, `-O3`, all six `k_cofac` instantiations:

| `k_cofac<L, method, stage2>` | registers | stack frame | spills | reg-limited warps/SM |
|---|---:|---:|---:|---:|
| `<3, rho, ->` | 78 | 96 B | **0** | 25 |
| `<4, rho, ->` | 94 | 112 B | **0** | 21 |
| `<3, ECM, no s2>` | 86 | 96 B | **0** | 25 |
| `<4, ECM, no s2>` | 112 | 112 B | **0** | 18 |
| `<3, ECM, s2>` | 122 | 368 B | **0** | 16 |
| `<4, ECM, s2>` | 154 | 448 B | **0** | 12 |

(warps/SM from 65,536 registers per SM and 8-register granularity; it is an
upper bound from registers alone, not a measured occupancy.)

**Nothing spills at any width**, which was the failure mode that would have made
the 4-limb ECM path far worse than the CIOS ratio predicts. It does not happen.

What the table does show is that **the cost compounds on the ECM stage-2 path
and only there.** Rho loses 25 → 21 warps (−16%) on top of ~1.8x arithmetic;
ECM with stage 2 loses 16 → 12 (−25%) on top of the same arithmetic.

**That worry did not materialise.** Measured a day later, ECM at 4 limbs is
2.7x *cheaper* than rho on AS276 at matched yield and is now the default on a
3LP side (finding 70) — the occupancy penalty is real but far smaller than
rho's `sqrt(p)` growth over the same width step. `CF_ECM_NBABY` is still the
knob to reach for if 4-limb ECM ever does look disproportionate; it has not
needed touching.

The 3-limb instantiations are **identical in both builds** — 78 / 86 / 122
registers, measured on a `CF_LMAX=3` compile of the same source. They are
separate template instantiations and ptxas allocates registers per kernel, so a
wide build does not make the narrow kernel slower. That closes "separate
executables" as a *performance* option; it survives only as the binary-size
argument above.

### Cofactor width and method — MEASURED 2026-08-19, GPU idle

Both questions the width work opened are now answered. The box's ECM job was
suspended (`kill -STOP`) so these are contention-free.

**1. What the fourth limb costs on work that does not need it.** c183, 200 q,
and C194, 100 q, forced wide with `--cof-limbs`. **All four width combinations
emit byte-identical relation files** (c183 md5 `17d8f1d8…`, 9,363 relations),
so this is a pure width price on unchanged work:

| job | side0/side1 | rational q | algebraic q | stage ms/q | wall ms/q |
|---|---|---:|---:|---:|---:|
| c183 | 3/3 | 2.86 | 13.90 | 16.88 | 112.17 |
| c183 | 4/3 | 4.93 | 14.18 | 19.22 | 116.37 |
| c183 | **3/4** (the C208 shape) | 2.85 | 23.46 | 26.38 | **121.99** |
| c183 | 4/4 | 4.89 | 23.74 | 28.72 | 124.38 |
| C194 | 3/3 | 3.32 | 12.10 | 15.48 | 116.30 |
| C194 | **3/4** | 3.26 | 20.87 | 24.18 | **125.82** |

A side's queue costs **x1.72** at four limbs (rational 2.86 -> 4.93; algebraic
13.90 -> 23.46 and 12.10 -> 20.87), against the 1.8-2x the CIOS ratio
predicted. Wall cost of running the C208 shape on a job that does not need it:
**+8.8% (c183), +8.2% (C194)**.

**Verdict: keep the dynamic selector.** 8-9% of wall is too much to give away
for the simplicity of "always 4/3", and the selector costs nothing at run time
(a host-side switch per flush). Separate executables remain unjustified —
the 3-limb kernels are register-identical in both builds. **This is the
decision the width work was blocking, and it is now made.**

**2. rho vs ECM. SHIPPED as the default 2026-08-19** — chosen per side from
`ceil(mfb/lpb)`, rho at 2LP and ECM at 3LP, with ECM's `B1` derived from `lpb`
and the requeue round default raised 2 -> 4 (ECM escalates in curves per
round). Zero-flag effect: c183 17.21 -> 14.30 ms/q *and* 9,363 -> 9,394
relations; C194 15.48 -> 13.95; AS276 3,443 -> 4,089 relations. `--cof-rho` /
`--cof-ecm` force one method on both sides. Two gate cases pin the automatic
choice and the overrides.

The measurement behind it: **ECM wins 2-4x at 3LP and loses narrowly at 2LP** — the
discriminant is the large-prime count, not `lpb`. Measured cheapest-saturating
config for each, both swept from below: c183 `lpb 32/mfb 92` 30.83 -> 15.38
ms/q (2.00x), AS276 `lpb 35/mfb 101` 332.63 -> 123.22 (2.70x). Full sweep over
`lpb 29-36` in RESULTS finding 70; guidance in RUNBOOK "Method: ECM for 3LP,
rho for 2LP". Making it the default is a shipped-behaviour change and has
**not** been made.

*(An earlier version of finding 70 reported 15-18x. That was wrong: rho had
been priced at an over-large budget rather than swept from below — the same
one-sided-tuning error the finding was written to expose. Corrected in place.)*

### Performance accounting

Three-to-four-limb Montgomery multiplication expands the CIOS inner product
from `3*3 = 9` to `4*4 = 16` limb multiply-adds. That suggests roughly a
1.8–2x cofactor-stage cost from width alone, subject to register pressure. A
quoted **5x** is not a four-limb estimate: it combines the width cost with a
pessimistic rho workload in which the factor being sought is three bits larger
and therefore takes about `sqrt(8) = 2.8x` as many expected iterations. ECM
also pays for four-limb operations, but does not have rho's square-root scaling;
the rho/ECM crossover for the new cofactor population is unmeasured.

On the current measured profile, cofactorisation is 14.37 of 98.26 ms/q
(14.6%). If the *entire* stage changed and everything else remained fixed:

| cofactor-stage multiplier | total ms/q | whole-pipeline slowdown |
|---:|---:|---:|
| 1.8x | 109.8 | 11.7% |
| 2x | 112.6 | 14.6% |
| 3x | 127.0 | 29.3% |
| 5x | 155.7 | 58.5% |

**Measured against this table 2026-08-19:** the widened queue came in at
**x1.72**, inside the 1.8-2x row, and whole-pipeline cost moved **+8.8%**
(c183) and **+8.2%** (C194) — close to the 11.7% the 1.8x row predicts, and
lower because only ONE side widens on a real job. The 5x row never applied: it
priced rho's `sqrt` growth into the width, and the fix for that was to stop
using rho on the side that suffers it.

That table is an Amdahl illustration, not a projection for AS276. The larger
job has a different area, cofactor distribution, stage mix, and yield. The 5x
case would normally apply only to its hard four-limb subset, not every queued
record.

The final metric is **time to a filterable matrix**, not raw relations/s:

```
new time / old time
  = (new ms/q / old ms/q)
  * (new required usable relations / old required usable relations)
  / (new usable relations/q / old usable relations/q)
```

LPB directly expands the possible large-prime ideal universe and therefore the
column and singleton burden during filtering. As a scale indication only,
`pi(2^(b+d))/pi(2^b) ~= 2^d*b/(b+d)`: 32 -> 33 is about 1.94x and 32 -> 35
about 7.31x. Those are **not** required-relation multipliers—a run does not
sample the entire prime universe—but they show why extra raw yield cannot be
credited without filtering. MFB does not directly enlarge the ideal universe
at fixed LPB, but admitting more and larger 3LP products can still raise the
required surplus.

### Work status and rough scope

| item | status | rough engineering scope, including GPU validation |
|---|---|---:|
| A=32 whole-area endpoint/walk support | not started | about 1 week |
| A=32 stateful slabbing for 12 GB | not started | about 2–3 weeks |
| 64-bit large-prime outputs and gates | **done 2026-08-17** | — |
| per-side `mz<3>` / `mz<4>` dispatch | **done 2026-08-18**, gated and timed (+8-9% wall when forced wide) | — |
| per-side rho/ECM dispatch, default | **done 2026-08-19**, gated and timed (finding 70) | — |
| cofactor outcome reporting (`split / dead / stuck`) | **done 2026-08-19** | — |
| C208 validated against a 1.5B-relation GGNFS corpus | **done 2026-08-19**, 99.97% recall (finding 69) | — |
| filter test on a real corpus | not captured | required before a performance claim |

Together, a robust slabbed implementation on top of the width dispatch is
roughly a **3–5 focused engineering-week** change, not counting delays obtaining the
target workload or GPU access. A whole-area path restricted to >=24 GB cards
would be appreciably smaller. These are source-review estimates, not measured
schedules.

## Validated

| | what | where |
|---|---|---|
| **Golden suite** | `./cofcheck.sh` — 30 cases on c183 q=120000053, exact relation counts, not ranges | `cofcheck.sh` |
| **Reconstruction gate** | every emitted factor divides, is prime, is within lpb, and both norms rebuild to exactly 1 | `--check-relations` |
| **Transform gates** | root transform against its definition, over a real factor base | `fbtest` |
| **Band runs** | c147: 1340 q → 159,837 relations, both sides PASS | 5070, 5090, 4090, A100 |
| **Real job** | snfs236, ~20M relations before deliberate interruption | 5070 |

Cards with measured band data: **RTX 5070** (WSL2), **RTX 5090**, **RTX 4090**,
**A100 80GB** (native Linux), and an **RTX 3090** via an external reporter.

`--check-relations` runs on a machine with no GPU.

## Measured, and what it means

- **Fill does not scale with the GPU.** 5090 has 3.5× the SMs of a 5070 and
  returns far less than that on fill, against 3.33× transform and 2.01× apply.
  All three swept cards reach the same **absolute knee at 1152 blocks × 32
  threads**, flat above it. Mechanism unresolved; the leading candidate is now
  **work granularity** (fine chunks balance the tail), which fits the plateau.
  Both L2 stories are out: capacity was already dead, and one geometry fitting
  48/72/96 MB of L2 argues against write-combining decay too. `ncu` is blocked
  on the rented boxes (`ERR_NVGPUCTRPERM`), but **one local 5070 profile now
  exists** — `work/c147/fill_5070.ncu-rep`, a single `k_fill_atomic` launch.
  It corroborates granularity and kills the bandwidth stories outright:

  | | |
  |---|---:|
  | DRAM throughput | 12.83% |
  | L2 hit rate | 96.45% |
  | L2 (max bandwidth) | 52.11% |
  | L1/TEX hit rate | 18.91% |
  | bytes used per 32-byte load sector | 8.7 |
  | bytes used per 32-byte store sector | 4.6 |
  | stall cycles on L1TEX scoreboard | 157.7 of 191.0 (82.6%) |
  | SM busy | 3.75% |
  | waves per SM | 1.00 |
  | SM active / elapsed cycles | 8.00M / 10.88M |

  Read together: it is not DRAM-limited (12.8%), not L2-capacity-limited
  (96.5% hit — independent confirmation that story was dead), and **not at an
  L2 bandwidth ceiling either** (52%). It is latency-bound on uncoalesced
  scatter — 82.6% of stall cycles waiting on L1TEX, at 4.6 useful bytes per
  32-byte store sector. That last number matters for item 11: the prior-art
  claim that scatter tuning here "dies at the L2 transaction ceiling" does not
  describe this kernel, and ~7× store-sector waste is headroom, not a wall.

  **Waves per SM = 1.00 is the granularity mechanism made visible.** The whole
  grid is resident at once, so a block that draws a heavy chunk has no queued
  block to backfill its slot — and SMs sit idle for 26.5% of elapsed cycles,
  which is that tail. More, smaller blocks fix it twice over: finer chunks
  shorten the tail, and more resident warps hide the 157-cycle L1TEX stall.

  **Caveat: this profile is at 288 blocks × 256 threads** — the pre-finding-52
  geometry, and 256 threads is precisely the held-fixed axis that made
  finding 51 an artifact. So it characterises a configuration the project has
  abandoned. The 288-vs-1152 A/B **at 32 threads** is still unrun, and is
  still what would settle the mechanism rather than merely fit it.
- **Finding 51's Ada-vs-Blackwell block response was an artifact.** It held
  `--threads` at 256; at 32 the 4090's degradation reverses to improvement and
  all three cards behave alike. See finding 52.
- **rel/J is flat between the 5070 and 5090** — 28.1 vs 29.5 on sampled board
  power, i.e. a tie, with the 5090 1.68× faster. The Ada 4090 is the outlier at
  18.1. Generation separates these cards; width does not. **This tie is an
  artifact of board-only power and probably does not survive the metric of
  record.** First wall reading on the local box (2026-08-07, UPS, monitors
  excluded): **395–400 W whole-box against 165 W GPU board**, so the card is
  ~42% of the box and host overhead is roughly as large as the GPU's own draw.
  Host overhead is approximately *fixed*, so the faster card amortises it over
  more work: `whole-box rel/J = board rel/J × W_gpu / (W_gpu + W_host)`. With
  the host constant now measured at **~105 W idle / ~115 W with the sieve's own
  core** (item 6), that puts the 5090 at ~23.5 against the 5070's ~17.8 — the
  5090 **ahead by ~1.3×** rather than tied. The tie was an artifact of pricing
  only the card. Caveat: this places both cards in *this* box; the 5090 numbers
  came from a rented machine whose own host draw is unknown and probably
  higher, and per the next bullet that can never be measured.
- **The cross-card rel/J table can never be restated on the metric of record.**
  Whole-box power needs a physical meter on the box, and the 5090/4090/A100
  numbers came from rented machines. Those rows are permanently board-only.
  What *is* answerable with a meter is the thing the project was chartered to
  answer — the local 5070 against the local CPU box, item 0 — because both
  sides of that comparison are on this UPS. Treat cross-card rel/J as an
  architecture note, not as a verdict input.
- **THE BOX IS UNDERVOLTED AS OF 2026-08-17, and every timing taken after that
  date is ~6.7% slower than one taken before it.** The card's V/F curve is
  pinned to ~2900 MHz at 950 mV (stock was 2910 MHz at 1080 mV), which trades
  6.7% of throughput for 28% of board power — **+14.6% whole-box rel/J**,
  finding 61. It is *not* a code change and it is *not* reflected in any
  finding numbered below 61: finding 58's 108.27 ms/pair, finding 60's apply
  breakdown and every c147/c183 timing in this file were measured at stock.
  Comparing a new measurement against an old one without accounting for this
  will read as a 6.7% regression that is not there. Board draw is now ~140 W
  and the card sits at 50–56 °C rather than 69 °C.
- **Superseded:** an earlier rel/J table built on *nameplate* TDP concluded the
  design "does not want wide expensive GPUs". It is retracted — see the
  RETRACTED block in `RESULTS.md`. Do not quote the old numbers.
- **Confirmed on a second job, at every rectangle tested** *(findings 58, 65)*.
  On the NFS@Home C194, sieving above `alim` so GGNFS cannot truncate its base
  either, yield is **0.979–0.981 of GGNFS's at all four matched rectangles**
  (`2^14 × 2^13`, `2^15 × 2^13`, `2^15 × 2^14`, `2^16 × 2^14`) — flat in
  aspect ratio, flat in area, and flat in `j` within a rectangle. The earlier
  "−14.4% at a 1:1 rectangle" was a geometry mismatch: `-J 15` covers our
  `2^16 × 2^14`, so it compared our square against GGNFS's wide rectangle. There is
  no measured cost of the norm approximation (item 5).
  Throughput at that equal-work point is **3.31×** the CPU box and **2.74×** on
  energy, against the c183's 3.11× and 2.53×. The same measurement at q=20M
  reads 4.91×, inflated by the factor-base convention alone.
- **Yield matches GGNFS at matched geometry, and the verdict margin is ~2.5×
  rather than 4.3–5.5×.** A same-session control (finding 57) sieved the c183
  over one q interval with `gnfs-lasieve4I15e` and with this pipeline at the
  identical `2^15 × 2^14` rectangle: **46.09 vs 46.47 relations per (q, rho)
  pair**, a 0.8% agreement, with our side carrying 11% more factor base. The
  correction to the margin is a *unit* error, not a performance one — the CPU
  rows are per **prime q** and ours are per **(q, rho) pair**, and this
  polynomial averages 1.528 roots per prime. **Always state which.**

Measured end-to-end on the 5070 (c147 band, 1340 q, same binary, geometry the
only variable): fill −6.1%, apply +0.06%, wall −2.1%, relations byte-identical.
`--fill-threads` is measured to be necessary, not assumed: at the shipped 1152
blocks the 5090 costs **23%** at 256 threads (3.239 ms) against 32 (2.633), so
raising the block default alone would have left that on the table.

Projected but **not** measured end-to-end: the same pipeline A/B on the 5090,
where the standalone gain is largest (16.7%).

## Known defects

- **FIXED 2026-08-18 — non-primitive relations at small q (finding 68).**
  `k_intersect_compact` filtered on `gcd(i,j) == 1` and assumed that made
  `(a,b)` primitive. The lattice map has determinant q, not 1, so positions
  with `q | b` gave `gcd(a,b) = q`; msieve rejects those with `error -6`. They
  survive the sieve because such a point lies on the plattice line of *every*
  root of q, so the sieve subtracts `deg * log(q)` — exactly the `q^deg` in the
  norm — and the position looks as smooth as the much smaller `(a',b')` it is a
  multiple of. Affected positions per q go as `I*J / q`, so it is invisible at
  production q (~2 per q on a C194 at `2^29`) and conspicuous on a small SNFS
  (~335 per q at `2^27` against q = 400009, giving 0.0995% bad relations).
  Fixed by also rejecting `q | b`; the c147 band is byte-identical.
  **`--check-relations` now tests `gcd(a,b) == 1`** — it previously verified
  every factor and both norms, all of which a non-primitive relation passes,
  which is why this shipped.

- **Remaining-norm approximation.** The largest-term approximation differs from
  the true rectangle maximum by ~2 bits and costs ~0.13% of CADO's relations.
  The band-wide fixed scale is a second, unchecked approximation. This is the
  main known yield loss.
- **Duplicate share 15.8–25%** from the full special-q-side factor base, at
  ~1.34× the downstream TD/cofactor work for the same unique yield.
  **These are whole-FB-band numbers.** Measured raw inflation runs 1.0021× on
  the c147's 0.4%-wide band, 1.1877× on the c151's full-FB band and 1.2325× on
  the snfs2 corpus, so the defect scales with band coverage rather than being a
  fixed property of the design.
  **Truncating at `q` is now measured and does not fix it (finding 67):** on a
  real 275M-relation band it removes only 1.78% of the duplicate finds, and on
  a band stopping below `lim` it destroys more unique relations than it saves
  in duplicate work. On a generous-`mfb` job — the c183's `mfba 92`, the C195's
  95 — it does essentially nothing either way. Item 3 is closed; **this defect
  stands, without that remedy.**
  Note also that a corpus can carry duplication that is not ours at all: the
  snfs2 corpus holds a hand-stitched restart's re-sieved q window (finding 67),
  worth 1.4% of its relations. The `k = 1` stratum detects it.
- **The walk gate tops out at logI 10, and every deployed geometry is 14–16.**
  `--verify`'s eight cases (4:1 through 1:2, `make check` runs them via
  `walkcheck`) establish that the Franke-Kleinjung walk is aspect-ratio-correct,
  which is what finding 63 needed — but at I ≤ 1024. The x packing
  `i + I/2 + (j << logI)` is a uint32 and only gets tight in the untested
  logI 11–16 range, so a `pl_make`/`pl_next` bug specific to a large logI would
  still ship. The blocker is `verify_cpu.c:35`: `check_one` sorts its reference
  with an insertion sort, so at logI 15 the reference is ~16k entries and the
  gate would cost ~2.7e8 comparisons per (p, root). Replacing that sort is what
  would let the gate reach the geometry it is about.
- **`k_fill_l1` (twolevel path) has never been swept** at any geometry. It
  takes an explicit `--fill-blocks` but defaults to its own 144 × 512, because
  the 1152 × 32 result was measured on `k_fill_atomic` — a different kernel
  with a different write pattern.
- **Power is board-only.** The metric of record is whole-box
  relations/sec/watt; host draw is unmeasured. The A100 has no sampled power.
- **Host contention costs up to 29% of wall clock, invisibly.** Saturating the
  CPU leaves every `cudaEvent`-timed kernel flat within 1% while wall goes
  24.30 → 31.27 ms/q; half the cores already costs +22.6%. All GPU counters we
  have are blind to it, so a busy box reports perfect kernel times and a bad
  ETA. Stated as throughput that is an **18.4% relation-rate loss at half load
  and 22.3% at full** — not the 22.6/28.7% wall figures, which are
  percent-of-baseline and overstate the bar for co-scheduling. **Any cross-box
  wall-clock or ETA comparison is invalid without knowing host load on both.**
  The pipeline prints `GPU-accounted / wall (excl cofac)` to expose it; the
  first published pair used a formula since corrected, so current values await
  a confirmed-idle box. Finding 53 is canonical for all of this.

## Open experiments, in order

**This list is the task list of record.** Session-local task trackers do not
survive; if work is worth returning to, it belongs here. Two lists drifted
apart once already — items 7–9 below existed only in a chat session and were
absent from every document in the repo.

0. **The verdict band** *(added 2026-08-06, Fable review)*. The question the
   project was chartered to answer — unique relations/sec/watt on the **c183**
   against the measured CPU box — has never actually been run; every
   end-to-end number so far is a proxy job (c147, c151, snfs236). The CPU side
   is already frozen: 3.12 core-s/q at q=130M and N_eff 10.24 at 16 workers
   give **0.295 s/q for the whole box**, so green is ≤ 0.54 s/q and beating
   the CPU outright on component-proxy perf/watt is **≤ 0.179 s/q end to end**
   (prototype.md, "The GPU's per-special-q budget").

   **Energy bar from measured wall power** *(2026-08-07)*. Those s/q bars were
   derived without a meter. Item 6 now has one, including the host constant
   (~105 W), so the comparison can be made in joules per special-q rather than
   in seconds against an assumed wattage. **Assuming the 16-worker CPU
   baseline is this same box** — it is a 9800X3D, 8 cores / 16 threads, so a
   16-worker sweep fits it, but confirm it:

   Both sides' wattage is now measured (item 6). The GPU side draws ~270 W
   whole-box; the CPU side draws 220 W and its throughput was measured in the
   same session. But **which CPU throughput** turns out to be the whole
   question:

   | CPU comparator | J/q | GPU at 70–90 ms/q × 270 W | **margin** |
   |---|---:|---:|---:|
   | `I14e`, measured | 37.4 | 18.9–24.3 J/q | **1.5–2.0×** |
   | `I15e`, measured | 104.5 | 18.9–24.3 J/q | **4.3–5.5×** |
   | `I15e`, finding 43 | 64.9 | 18.9–24.3 J/q | 2.7–3.4× |

   > **RETRACTED 2026-08-17 — the two columns are in different units.** The
   > CPU J/q above are **per prime q**; the GPU column is **per (q, rho)
   > pair**, which is what `--nq` and the band summary count. GGNFS averages
   > **1.528 roots per prime** on this polynomial, so every margin in the table
   > is inflated by that factor. Finding 57 measured both sides in one session
   > at matched geometry and matched yield (46.47 rel/pair against GGNFS's
   > 46.09, a 0.8% agreement) and the honest figure is:
   >
   > | | per (q, rho) pair | per prime q |
   > |---|---:|---:|
   > | GPU wall / energy | 98.5 ms / 26.6 J | 150.5 ms / 40.6 J |
   > | CPU whole box / energy | 306 ms / 67.4 J | 468 ms / 103.0 J |
   > | **advantage** | **3.11× time, 2.53× energy** | same |
   >
   > **~2.5× on whole-box relations/joule**, which still clears the "beats the
   > CPU outright" bar. Do not quote 4.3–5.5×.
   >
   > **State the unit in every future number.** "Per q" is ambiguous in this
   > project and has now cost two corrections; item 0's own `3.12 core-s/q` is
   > per *pair* while item 6's `474.8 ms/q` is per *prime*, in the same file.

   **Geometry: checked 2026-08-16, and it matches — finding 55.** The concern
   was that the GPU sieved an `I14e` rectangle (`2^14 × 2^13`) while being
   graded against finding 43's `I15e` CPU baseline (`2^15 × 2^14`, four times
   the area), flattering the verdict by up to 4×. It did not: the standalone
   benchmark defaults to `logI 15, J 16384` (`bench_main.cu:426`, unchanged
   since the initial checkin), the finding 43/44 commands pass neither flag,
   and `5.369e8` positions is what this file's own header records. **So
   64.371 ms/q is an `I15e` number, the `I15e` row is its comparator, and the
   margin for it is 4.3–5.5×.**

   What is still unpinned is the *pipeline*: every pipeline run on record uses
   `--logI 14 --J 8192`, so item 0's 70–90 ms/q projection takes standalone
   `I15e` milliseconds and calls them a pipeline number. **A verdict band
   states its geometry in its log header and is graded against the matching CPU
   row** — `I14e` → 37.4 J/q, `I15e` → 104.5 J/q. Mixing them is still worth
   4×; it is now a run-discipline requirement rather than an open question.

   **The mismatch that survives is the factor-base convention, and it is
   q-dependent** (finding 55). Standalone side 1 truncates the base at q, as
   GGNFS does (`bench_main.cu:773-774`); the pipeline does not (`:1360`,
   `fbbound = alim`), because per-q truncation is item 3 and unbuilt. Counted
   on `oracle/c183.fb1`, the pipeline's full base carries **2.54× the bucketed
   entries GGNFS sieves at q ≈ 50M**, 1.11× at 120M, 1.03× at 130M, and 1.00×
   at 190M (above `alim`, where truncation is a no-op). So the 70–90 ms/q
   projection — anchored at q=120M, truncated, standalone — is roughly right at
   the 130M probe and **optimistic at the 50M probe**, the three probes will
   show a q-dependence in ms/q that is not yield drift, and the item-3 A/B has
   no signal at all at 190M.

   Note also that `I14e` and `I15e` are not interchangeable for the CPU: the
   larger area wins 2.2× the relations per special-q (68.9 vs 31.6) while
   costing 2.9× the time, so `I14e` is the better rel/J (0.846 vs 0.659) and
   `I15e` the better relations-per-q-range. A real c183 job picks `I15e` or
   above. Which of those the GPU should be held to is a question about the
   deployment, not about the hardware.

   **The GPU shows the same trade and a worse one** *(finding 57,
   2026-08-17)*. Doubling J at fixed I — `2^15 × 2^14` → `2^15 × 2^15` — costs
   **2.03× the time for 1.47× the relations**, so rel/J falls 27% (2.57 →
   1.87). The CPU's reason for large rectangles is amortising its per-q root
   transform, 12% of its wall and about 1% of ours, so **the GPU should prefer
   smaller areas than the CPU does**. Take the larger J only when the q range,
   not energy, is the binding constraint.

   **But if you do want more area, buy it with I** *(finding 65)*. The two axes
   are not interchangeable: at equal area the wide rectangle beats the square
   by **+11.7% relations at `2^28` and +16.3% at `2^30`, for −0.3% device
   time** — `2^16 × 2^14` returns 10,671 relations against `2^15 × 2^15`'s
   9,176 on the same window. Doubling I costs 1.86 bits of `log2(maxnorm)`
   against doubling J's 3.86, because `i` multiplies the shorter vector of the
   reduced q-lattice and `j` the longer one. Measured rel per device-second
   across the grid: `2^14 × 2^13` 390.6, `2^15 × 2^13` 387.2, `2^14 × 2^14`
   336.0, `2^15 × 2^14` 323.9, `2^16 × 2^14` 260.6, `2^15 × 2^15` 223.4.

   **And it is the cheaper shape too** *(finding 64, row added 2026-08-18)*:
   `2^16 × 2^14` sizes at **4.98 GB** against the square's 5.32 GB, because
   `bkthresh` defaults to `1 << logI` and so a wider I shortens the bucketed
   prime range — 2.60 GB of bucket array against 2.91 GB. Same mechanism as the
   yield win, so there is no trade-off to weigh: at fixed area, wider is better
   on relations, on device time, and on VRAM simultaneously.

   The c183's own standalone
   sieve measures 38.2 + 26.2 ms for the two sides (RESULTS.md "Reproduce",
   `I15e` geometry, side 1 truncated at q, at q=120M), so with TD and the
   heavier mfba-92 3LP cofactor load the pipeline projects to roughly
   **70–90 ms/q — about 2× inside the harder bar**. That is a projection, not a measurement, and nothing measured
   contradicts it — which is exactly why the run should happen now: it is a
   wall-plug meter and a weekend, not a project. Needs: the meter (item 6);
   bands at q ≈ 50M / 130M / 190M for the yield drift; relations
   **deduplicated before quoting** (raw is inflated 1.19–1.34×, see RUNBOOK);
   an idle host per finding 53. Fold item 3's full-vs-truncated A/B into the
   same bands — the FB convention moves both the yield accounting and the
   downstream load, so it should be settled in the graded configuration.
   **But not in these bands as specified.** Three narrow probes at 50M / 130M
   / 190M are the right shape for yield drift and the wrong shape for the FB
   convention: a band covering a fraction of a percent of `lim` has almost no
   duplicates to remove (measured 1.0021× on the c147), and under truncation
   the relations whose re-finding q lies outside the probe are lost outright
   rather than deduplicated. Grade perf/watt on the three probes; settle the
   FB convention on one contiguous band wide enough to contain both q of a
   duplicate pair, or by replaying the attribution offline as item 3 now does.
1. **Concurrent-q throughput.** Two independent fill workspaces, two q or two
   sides in separate streams, sweep 1/2/4. This is the decisive test for
   whether wide cards are a poor fit or are simply being fed too little
   independent work. **Design it at the new knee** — the "144 blocks each" this
   item used to specify is the pre-finding-52 geometry and would reproduce the
   flaw described next: the old framing
   ("two 144-block fills vs one 288-block fill") is now measured entirely below
   saturation, where any configuration scales, so it would have credited
   streams for reaching a thread count one kernel already reaches. Compare
   2 streams × 1152 against 1 kernel × 2304 instead. The old note here said the
   two architectures predict differently — Blackwell's flatness fitting "idle
   capacity" and Ada's degradation not. That distinction is gone: at 32 threads
   all three cards flatten, so one prediction covers them all.
2. **Startup fill autotune.** `--fill-threads` is done; the autotuner is not.
   Sweep both axes, since holding one fixed is how the old default was reached.
   Opt-in for the standalone benchmark, where reproducibility is the point;
   default-on is defensible for `--pipeline`, where a per-job knee cannot be
   derived and 15 trial fills cost ~60 ms of a multi-hour run.
3. **Full vs q-truncated factor base — CLOSED 2026-08-18, do not build it
   (finding 67).** Was: compared at equal *deduplicated* yield.
   **Downgraded 2026-08-17 — worth 10–15%, not the ~25% an entry count
   suggests, and it may simply be the wrong trade (finding 59).** Replayed
   offline over the c151 corpus: truncation removes 18.4% of *potential*
   re-finds and **zero unique relations** — but only because that band ends
   exactly at `alim`, which is the favourable case, and because the potential
   (1.52 finds/relation) is well above what is actually emitted (1.19, msieve's
   own dedup). The sieving prize is small for a reason that generalises: cutting
   the base from 33.5M to 15M drops the **entry count 53%** but the **bucket
   records only 8%**, because they go as `Σ1/p ~ ln ln p`. That model is
   confirmed by the C194 lim sweep, where 4× the lims moved the bucket array
   +12% against +12.6% predicted. Entry count is the right axis only for the
   root transform.

   **ANSWERED AND CLOSED 2026-08-18 on a 275M-relation corpus — finding 67.**
   Replayed over the SNFS `17327^61-1` dataset (41.5 GB, 339,445,456 lines,
   band `q in [40M, 175M)` recovered from the corpus, `rlim` 67.1M):

   - **On the band as run, truncation loses zero unique relations and removes
     1.78% of the duplicate finds.** The zero is structural, not lucky: under
     truncation a relation is found at its largest band prime `q`, and any
     unsieved FB prime above `q` would itself be a band prime larger than `q`.
     So **truncation is lossless exactly when `band_top >= lim`** — which is
     also why the c151 showed zero, its band ending precisely at `alim`. Both
     corpora were the favourable case.
   - **In every configuration an operator would actually run, it is lossless.**
     Only two shapes occur: `lim <= band_bottom` (truncation never binds) and
     `band_bottom < lim <= band_top` (this job). A band whose *top* is below
     `lim` would cap the base at `q < lim` for every q in the run, so the `lim`
     that was set is never the one used — an operator lowers it instead. The
     sweep confirms the boundary sits exactly at `band_top = lim`: below it the
     trade goes negative (9.6% of unique relations lost at `0.70 x lim`, 6.0%
     at 0.80, 2.8% at 0.90, 0 at 1.00), but that regime is a **test band**, not
     a deployment.
   - **This reconciles all four replays.** The two corpora that lost relations
     are the two whose band stops short of `lim`, and neither is a production
     run: the c147's band is 0.15M wide (a probe, `band_top` = 0.45x `alim`,
     22.68% lost) and the snfs236 corpus is a partial slice (0.28x `rlim`). The
     two complete bands — c151 at exactly 1.00x `alim`, snfs2 at 2.61x `rlim` —
     both lose **zero**. So the c147's 22.68%, quoted here as the alarming
     number, is a property of a narrow probe rather than of the design.
   - **Generous `mfb` collapses both sides to under 2.5%.** At `mfb 92` — the
     c183's value, and close to the C195's `mfba 95` — the unsieved primes fit
     in the cofactor, and truncation becomes very nearly a no-op. So the
     C195-shaped job this item was worrying about is the case where truncation
     matters *least*, in either direction.

   **So there is no version of this worth building.** The duplicate prize is
   1.8% where it is safe and is outweighed by outright loss where it is not,
   and the job class we care about is the one where it does nothing at all. The
   separate sieving-work prize (fewer factor-base entries per q) remains what
   finding 59 measured — 10–15%, bucket records only 8% — and is unaffected by
   this; it is also unaffected by *this* item, since it needs the same per-q
   truncation machinery for a return that finding 59 already called small.

   Two by-products worth keeping. **P(re-found) is 0.6968 here** against 0.702
   (snfs236) and 0.723 (c151) — three jobs, one number. And the **`k = 1`
   stratum is a free integrity check on any corpus**: relations with only one
   possible q cannot be re-found by our convention, so that row must read
   exactly 1.0000 copies. Here it read 1.0141, and histogramming those
   duplicates by q localised them to `q` in **[55,469,851, 57,129,679]** — 100%
   duplicated inside that ~1.66M-wide window, exactly zero outside it. That is
   a **stop/restart overlap sieved twice** (this run predates 12a, so the halves
   were stitched by hand), not a property of the siever. Removing it takes the
   duplicate inflation from 1.2325x to **1.2096x**. It is also the clearest
   measurement of what 12a's byte-exact resume is worth: ~1.7M q of duplicate
   sieving avoided.

   `nactive = lower_bound(primes, q)` on the already-sorted base, passed to
   transform/fill/resieve — no rebuild, no re-upload. The standalone harness
   already truncates statically (`--fbbound`, `fb_restrict` at
   `bench_main.cu:1252`); what is missing is making that bound *per-q* in the
   pipeline, where all three kernels still take the full `fb->n`
   (`pipeline.cuh:221`, `:227`, `:644`).

   **Measured 2026-08-07 over relations already on disk, no GPU.** Checked
   against ground truth on the c151: the tool finds **10,594,292 duplicates**,
   digit-for-digit the count msieve's own filtering reported (`RUNBOOK.md:461`).
   **The denominators do not match, though** — msieve quotes that count over
   67,165,877 relations, while the file on disk holds 67,043,952, which is also
   what this repo's own run record says (`RUNBOOK.md:399`). The 121,925 gap
   makes the ratios 15.77% and 15.80% rather than one number. An exact match on
   the duplicate count across different totals is what you would see if those
   extra lines were all unique — free relations are the obvious candidate — so
   the duplicate-detection logic looks validated and the *denominator* is what
   remains unreconciled. Do not quote this as "reproduces msieve exactly" until
   it is.

   | corpus | band as run | band / lim | `mfb` | raw inflation | truncation floor |
   |---|---|---:|---:|---:|---:|
   | c147 | [15.00M, 15.15M] | 0.4% | 59 | **1.0021×** | −17.3% |
   | snfs236 (partial) | [30.0M, 36.97M] | 5.2% | 88 | **1.0391×** | −6.1% |
   | c151 (complete) | [15.0M, 33.5M] | 55% | 59 | **1.1877×** | −15.0% |

   ("truncation floor" is over a band reaching `lim`; see point 3.)

   Three things follow, and two of them cut against doing this first.

   1. **The duplicate share is not a constant of the design — it is set by how
      much of the factor base the q band covers.** The 15.8–25% figure quoted
      under "Known defects" is a *whole-FB-band* number. A narrow band has
      almost nothing to deduplicate, so any A/B run on one will measure noise.
   2. **Truncation is yield-neutral only if the band reaches `lim`.** Under
      truncation a relation is found at its *largest* sq-side FB prime; if the
      band stops below that prime, it is never found at all. On the c147's
      as-run band the mfb ceiling alone says truncation would find **nothing**
      for 22.68% of unique relations. That is outright loss, not deduplication.
   3. **How much truncation actually deduplicates is bounded by `mfb`
      headroom, and on the c183's configuration that headroom is large.** An
      unsieved prime in `(q, lim]` does not vanish — it lands in the cofactor.
      On the snfs236 (`mfbr 88`, 3LP) it fits there comfortably, so over a
      hypothetical full band [30M, rlim] truncation is guaranteed to remove
      only **6.1%** of emissions against the full base's 1.7918× — the rest of
      the way down to 1.0× depends entirely on whether the survivor threshold
      refuses those relations. The c183 carries `mfba 92`, i.e. *more*
      headroom still.

   So the honest range for truncation on a c183-shaped job is "removes between
   6% and ~44% of downstream work", and **where it lands inside that range is
   item 7's question, not this one.** Items 3 and 7 are one experiment: our
   gate is the loose one, and a loose gate is exactly what lets a truncated
   base keep re-finding the duplicates truncation was supposed to remove.
   Run them together, on a band wide enough to have duplicates to remove.

   The 1.7918× on the snfs236 independently reproduces the 1.82 mean sq-side
   primes in `RUNBOOK.md:444`, and **P(re-found) now agrees across both jobs
   measured directly rather than back-solved**: 70.2% on the snfs236
   (n = 1,089,564) and 72.3% on the c151 (n = 13,709,863), against RUNBOOK's
   71.8% at n = 1,023 and its 72.8% back-solve. Two cautions for anyone
   redoing this:

   - The raw slot-take rate is **biased low** — a relation is in the corpus
     only because one slot already hit, so subtracting that forced hit from
     both numerator and denominator understates P. At k = 2 the correction is
     exact (`E[copies | copies ≥ 1] = 2/(2−P)`, so `P = 2r/(1+r)`); it is what
     turns the raw 54.1% / 56.6% into 70.2% / 72.3%.
   - **Detect the band from the run, not from the corpus.** Counting hits per
     prime does not separate band q from ordinary primes at the low end: an
     ordinary p divides ~N/p relations, so on a 67M-relation corpus every
     prime near 1M clears any fixed threshold. Doing that put the c151's band
     at [1M, 33.5M] instead of [15M, 33.5M] and produced a P(re-found) of
     18.6% — the false low-end "q" can never re-find anything, so whole
     buckets read exactly 0.0%. Cross-check against the q count in the run log
     (`RUNBOOK.md:110`: 1,088,865 q for the c151, i.e. π(33.5M) − π(15M)).
4. **Host cost — overlap first, graphs second, micro-optimisation last.**
   Identifiable host work is 1.694 ms/q, only **7%** of a 24.30 ms idle wall,
   but it triples under CPU contention (finding 53). It is three different
   problems and they do not have the same fix:

   **MEASURED 2026-08-17, and it moves this item DOWN — finding 56.** The free
   experiment below was run and **renice buys nothing**: ~1% in every paired
   comparison, at 12 competing workers and at full saturation alike. The reason
   is that at 16 runnable threads on 16 logical CPUs nothing waits for a slot —
   the feeder thread *shares a physical core* over SMT, and nice values do not
   decide who you share a core with.

   **Affinity, `SCHED_FIFO` and `--blocking-sync` were then all tried and all
   fail** (finding 56). `taskset` pins the whole process including the CUDA
   runtime's threads, so pinning *alone* costs 5.8% and reserving the physical
   core only returns to parity; `chrt -f 1` costs 1.6%; `--blocking-sync`
   costs 2.4%; and the two combined — the textbook low-latency recipe — are
   the **worst** arm at 4.5%. **Every zero-code lever is exhausted.**

   **What the contention actually is:** it tracks *oversubscription*, not
   busyness. At one runnable thread per logical CPU it costs ~6%, measured
   twice independently (12 real sievers, 15 spinners); at 1.7x subscribed it
   roughly doubles. No scheduler knob reaches it because it is contention for
   core resources — SMT siblings, cache, memory bandwidth. **The operational
   lever is worker count: keep competing work at or below `nproc - 1`.**

   Note also that `taskset` never keeps *others* off a CPU. Real reservation is
   cgroup v2 `cpuset.cpus.partition`, `isolcpus=` or `nohz_full=`; under WSL2
   all of them still sit above the Windows host's own scheduler.

   Two numbers now bound the prize. On a **verified-idle box** `acc/wall` is
   0.885 and GPU utilisation 89.5%, so the structural gap is **11.5%** on the
   c147 — contention adds ~6% at 12 workers and ~14% at saturation on top.
   But the gap **shrinks with area**, because host work per q is fixed while
   GPU work is not: 11.5% at `2^27`, 5.4% at `2^29`, **3.8% at `2^30`**. At the
   geometry a C195 would deploy at, this whole item is worth under 4%, part of
   which is interleaved launch/sync that overlap cannot recover. Treat it as a
   small-job concern; it is not a prerequisite for the LPB work or item 0.

   Two thirds of it is *prep* (host per-q tables/staging, small-prime tables)
   and one third is *launch + sync overhead* (`host: unaccounted`). **Finding
   53 holds the canonical table** — do not copy it here; it has drifted once
   already.

   1. **Overlap the prep with GPU execution.** Both prep terms depend only on
      the q-lattice, not on any GPU result, so q+1's host work can run during
      q's kernels. This is double-buffering, **not threading**: 1.166 ms of
      prep against ~20.7 ms of GPU work per q fits entirely inside the GPU's
      shadow with 18× room to spare, so perfect overlap takes it to *zero* on
      the critical path. Threading the same work would reach maybe 0.4 ms with
      four threads and leave it *on* the critical path — strictly worse, for
      more code and a synchronisation problem we do not currently have.
   2. **CUDA graphs for the per-q kernel sequence.** `unaccounted` is
      wall-minus-device inside TD/classify: the CPU issuing launches and
      waiting on syncs. Overlap cannot help — it is interleaved with GPU
      execution by nature — and it is the term that grew **443%** under load,
      so it is what makes the box fragile. The per-q sequence is fixed, so it
      can be captured once and replayed.
   3. **Then make the prep cheaper** — three-way partition replacing the per-q
      stable sort, fusing the small-ideal transform that is currently done
      twice, and **hoisting the per-q allocations**: `pipeline.cuh:169-183`
      mallocs and frees five temporaries (`idx/tp/trt/tg/tlp`) per side per q,
      i.e. 10 malloc/free pairs per special-q, plus three memcpys back over the
      pinned arrays. Reusable scratch removes that churn, and it is the part a
      better sort algorithm does *not* fix. **This was previously the whole of this item and it is the least
      valuable third of it**: after (1) the work is hidden in the GPU shadow
      and costs nothing regardless of how fast it is.

   **Priority note, 2026-08-07 (owner).** Sharing the box is a **requirement,
   not a robustness nicety**: a sieve that only performs on an idle host is
   much less useful, because the host is a desktop someone wants to use. That
   moves this item up and changes what "done" means — the target is that the
   GPU does not wait on the CPU, not that host work is small on a quiet box.

   **Do the free experiment before any of the code below.** The host thread is
   not obviously starved of CPU *time* — measured 2026-08-07 it holds 92.9% of
   one core, because CUDA's default sync spins. What it competes for is the
   *moment* it needs to issue the next launch, and at equal priority it loses
   that race as often as it wins. Two zero-code levers:

   - **Renice the competing work.** Raising your own processes' nice value
     needs no privileges. With the sieve at nice 0 and everything else at 19,
     CFS weights the sieve ~1024 to 15 and it wins essentially every contest.
   - **Leave headroom, but know what you are leaving.** This box is 8 physical
     cores / 16 threads, so 14 `ecm` workers plus one spinning sieve thread is
     15 of 16 *threads* and **every physical core is loaded** — there is no
     idle core to fall back on, and the sieve thread shares a core with an ECM
     worker whatever you do. Dropping batch work to `nproc - 2` frees two
     threads, i.e. one core's worth of SMT capacity, not a quiet core. That is
     why the renice lever matters more here than on a wide box: on 8 cores you
     cannot isolate the sieve by subtraction, only by priority.

   **The live instrument is GPU utilisation, and it is now automatic.** Every
   `--log` record carries `gpu=` and `acc/wall` (item 12b), so the pair of runs
   this rule needs is a `grep`. The rule was: if renicing moves utilisation
   toward the mid-90s the penalty is a scheduling problem, and otherwise the
   dependency is architectural. **Resolved 2026-08-17: it does not move, so the
   dependency is architectural** — but see the measured prize above before
   concluding that makes this item urgent. A caution that survives: 88–89%
   during the co-scheduled snfs236 run is *not* comparable to the 89.5% idle
   figure on the c147, because utilisation is a function of the job's area as
   much as of the host.

   One caution retained: this interacts with item 1: two concurrent q roughly
   **doubles host work per unit time**, so on a contended host that experiment
   can come back negative for host reasons unrelated to the GPU. Run it on a
   verified-idle box and check `GPU-accounted / wall` first.
5. **True rectangle maximum** for the norm approximation — one safe band-wide
   scale, or a few scale/slice-table buckets, gated at low/mid/high q on more
   than one job. Note the payoff is job-shaped: for the snfs236 polynomial the
   special-q side is degree 1 with `d = [1, -3.5e-24]`, so largest-term and
   true-rectangle are the *same number* and this buys nothing there. The ~2-bit
   gap is a GNFS-side problem.

   **The 1:1 anomaly that drove this item DOES NOT EXIST — finding 65,
   2026-08-17.** `gnfs-lasieve4I15e -J 15` covers what we call `2^16 x 2^14`,
   while our `--J 32768` sieves `2^15 x 2^15`. (It is a `2^15 x 2^15` square in
   GGNFS's own axes; the two sievers order the reduced basis oppositely, so
   their (i,j) is our (j,i) — and because their j is non-negative while our i
   is signed, the swap also halves one axis and doubles the other rather than
   being a plain transpose. Corrected 2026-08-18 — see finding 65.) Findings 58 and 63 compared a square against a wide
   rectangle. Recovering each run's rectangle from its own relations (invert
   the q-lattice) and re-comparing at **matched** rectangles gives ours/theirs
   = 0.9795, 0.9805, 0.9797, 0.9786 at `2^14 x 2^13`, `2^15 x 2^13`,
   `2^15 x 2^14` and `2^16 x 2^14` — **flat, with no aspect-ratio dependence
   and no j-dependence within a rectangle**. Finding 63 is retracted; its four
   eliminations stand.

   `sieve_allowance` and `sieve_bound_checked` are **exonerated by
   arithmetic**, no GPU needed: the allowance is `mfb + max(2/scale, 1.5)` with
   no geometry term, so the gate in *bits* goes 96.63 → 96.67 between those two
   rectangles — very slightly looser. The bound falling 119 → 117 is
   `las_scale` renormalising for a 3.86-bit larger norm, and both values
   reproduce exactly from the source. Do not start there.

   So this item is back to being justified only by its original ~2-bit argument
   and the ~0.13% of CADO's relations under "Known defects": **low value at the
   geometry we deploy, and no longer blocking anything.** The ~2% we sit below
   GGNFS is uniform across every rectangle and every j, which is the shape of a
   constant downstream residual (finding 62), not of a norm approximation that
   degrades as the rectangle grows.

   **What replaced it is an operational rule — buy area with I, not J.** At
   equal area the wide rectangle yields +11.7% at `2^28` and +16.3% at `2^30`
   for −0.3% device time, because doubling I costs 1.86 bits of
   `log2(maxnorm)` where doubling J costs 3.86: `i` multiplies the shorter
   vector of the reduced q-lattice and `j` the longer one. See item 0.
6. **Whole-box power**, to close the metric of record — the meter half of
   item 0. **The meter exists**: the UPS reports real watts. It is not a
   parallel task to item 10, it is a **prerequisite** for it (see there).

   Readings (2026-08-07, UPS, real watts). Monitors measured once at **~45 W**
   and subtracted throughout; the raw readings were taken with them on.

   | state | monitors off | notes |
   |---|---:|---|
   | **everything idle** | **95–115 W** | GPU P8 ~16 W, load avg 1.06 |
   | CPU full (14 `ecm`), GPU idle | 240 W | ≈9.6 W per busy core |
   | CPU full, GPU sieving | 395–400 W | GPU board 165 W |

   | derived | |
   |---|---:|
   | **host constant** (idle box incl. P8 GPU) | **~105 W** |
   | GPU **delta**, at the wall | 155–160 W |
   | GPU **delta**, board sensor (165 − 16) | ~150 W |
   | implied delivery loss | 3–7% |
   | power factor (W ÷ VA) | 0.96 |

   The idle reading is **jumpy, ±7%** — normal for a desktop drifting through
   C-states — so average it rather than taking an instant. Everything derived
   from it inherits that spread.

   **The board sensor is trustworthy.** Compare deltas to deltas: the card
   swings 150 W on the board sensor and 155–160 W at the wall, so the wall
   figure sits a few percent *above* the board figure, which is exactly what
   PSU and delivery losses predict. First independent check this project has
   had on the number every rel/J figure rests on, and it passes.

   **The GPU never reaches zero.** It idles at 15 W in P8 and it drives this
   box's displays (`Disp.A = On`), so there is no configuration of this machine
   where the card is absent. That 15 W belongs in the host constant on *both*
   sides of the item-0 comparison, which is the honest accounting anyway — the
   CPU box needs display output too.

   Cautions. **Read W, not VA** — VA is apparent power, ~4% high here. **Subtract
   the monitors** once and for all. And **none of these is a sieve-only number**:
   the ECM job is in all of them, and per finding 53 it is simultaneously
   costing the sieve throughput, so both sides of rel/J are wrong here. These
   are the *co-scheduled* data, worth having, but not the verdict data.

   **Elastic background load is still a trap**, even though it did not bite
   here. ECM expands to fill whatever cores are free, so stopping the GPU job
   handed it the sieve's spinning core and it drew *more* CPU power — the 240 W
   row therefore contains somewhat more CPU work than the 395–400 W row does,
   which understates the GPU delta by roughly the draw of one core. The
   agreement above survives that because the residual is ~10 W against a 150 W
   signal. It would not survive it on a smaller difference. Pin the background
   worker count across both readings, or stop it.

   **The host constant is now measured, which was the term every projection in
   this file was missing.** At ~105 W it is roughly a third of a sieving box,
   not the 200 W+ the earlier ranges allowed for.

   **The CPU side is measured too**, on the c183 at q=130M, 16 workers, all
   monitors-off and steady-state (2026-08-07):

   The box is a **9800X3D: 8 physical cores, 16 threads, 96 MB L3**, so "16
   workers" is two per physical core and the per-worker column below is per
   *thread*, not per core.

   | load | whole box | per worker |
   |---|---:|---:|
   | idle | ~105 W | — |
   | 14 `ecm` workers | 240 W | 9.6 W |
   | 16 `gnfs-lasieve4I14e` | **225 W** | 7.5 W |
   | 16 `gnfs-lasieve4I15e` | **220 W** | 7.2 W |
   | GPU sieving alone (derived) | ~270 W | — |

   **Power falls as the workload gets more memory-bound**, which is exactly why
   ECM was a bad proxy: it is ALU-bound and drew 33% more per core than the
   siever it was standing in for. The earlier ~260 W estimate scaled from ECM
   was ~18% high. Use 220 W for the CPU side.

   Throughput measured in the same session, so both sides of the CPU figure
   come from one box on one day:

   | siever | ms/q | rel/q | rel/s | J/q | rel/J |
   |---|---:|---:|---:|---:|---:|
   | I14e | 166.2 | 31.6 | 190.4 | **37.4** | 0.846 |
   | I15e | 474.8 | 68.9 | 145.0 | **104.5** | 0.659 |

   **I15e measures 475 ms/q here against finding 43's 295 ms/q** — same siever,
   same 16 workers, same stated q, 1.6× apart. A worker sweep run to chase it
   eliminated every mechanism except one:

   | workers | finding 43 | measured 2026-08-07 |
   |---:|---:|---:|
   | 8 | 2.120 q/s | 1.140 q/s |
   | 16 | 3.392 q/s | 2.106 q/s |
   | SMT gain | 1.60× | **1.85×** |

   Not cache capacity — 16 workers beat 8 by 1.85×, so SMT is working and the
   96 MB L3 is not the binding constraint at I15e (a tempting hypothesis on an
   X3D part, and wrong). Not thermal — cores held 4691 MHz under sustained
   load. Not startup — these are steady-state windows. The slowdown is
   *uniform across worker counts*, and contention or thermal effects would bite
   harder at 16 than at 8, not equally. **That leaves configuration**: finding
   43 does not record which `input.job` it swept, and this run used the c183
   (`alim 134.2M, mfba 92, alambda 3.5, lpba 32`) at 68.9 rel/q. A job yielding
   fewer relations per q would sieve faster per q and prove nothing.

   So the two numbers are probably measuring different work, and **s/q is the
   wrong axis to compare them on — rel/J is the right one**. Resolve by
   recording the job file alongside any future N_eff sweep.
7. **Why our survivor gate is looser than GGNFS's at matched lambda —
   DOWNGRADED 2026-08-17: the surplus does not exist at shipping defaults
   (finding 62).** Measured at each siever's own gate, our cofactor volume
   matches GGNFS's within **0.4%** on item 7's own config (c183 I14e: 784.85
   against 781.7 per pair) and within **0.03%** on the C194 at 15e (1594.12
   against 1594.8). Relations from those submissions differ by +0.9% and −1.7%
   respectively — opposite signs, so no systematic deficit either.

   The observation below stands as written, because it was made *at matched
   lambda*: a given nominal bit value does mean different things to the two
   gates. What is now measured is that this asymmetry **produces no surplus at
   the defaults we ship** — our derived allowance lands on the same cofactor
   volume GGNFS's lambda rule does, on two jobs and two geometries. The
   operational worry below — that the bound "can only be set by sweeping, never
   derived for a new job" — is answered: it *is* derived, and the derivation
   now has two independent external checks. **No sweep is needed for a new
   job.** Reopen only if a job is found where the volumes diverge.

   The original observation, retained:

   Same q
   range, same job, same nominal bits: `gnfs-lasieve4I14e` loses 17.3% of its
   yield going 91.8 → 87.5 bits where we lose 0.07% going to 88.0. We submit
   1,426 cofactors per special-q against GGNFS's 978 (`COF: 60664 tests`, 62 q)
   for comparable *unique* yield. So we admit ~46% more and the surplus is
   almost all unproductive — time, not relations. **This is also the other
   half of item 3**: the dedup a q-truncated base buys is exactly the set of
   relations the gate refuses once their `(q, lim]` primes move into the
   cofactor, so a loose gate spends the truncation prize before it is
   collected. Measured, truncation is guaranteed only 6.1% of the available
   ~44% on an `mfb 88` job; the gate decides the rest. Do not run these two
   apart. Until the cause is known the
   bound can only be set by sweeping, never derived for a new job. Candidates:
   a systematically low per-position norm estimate; the byte scale cancelling
   only if cell init and bound use it consistently; GGNFS's lambda gating more
   than the report threshold. **Do not "fix" it by tightening the default** —
   the current gate finds every relation GGNFS finds (verified, zero misses
   over 2,531), so any trade of yield for speed must be made deliberately.
8. **A = 32 sieve areas and MFB widening**, for jobs like AS276. **Two of the
   three blockers are gone**: `lpb` now goes to 64 (2026-08-17), so a C194's
   `lpba 33 / mfba 95` runs today; and `mfb` now goes to 128 (2026-08-18) on a
   per-side 3-or-4-limb dispatch, so a C208's 4/3 shape runs without a flag.
   AS276 has since been sieved end to end and validated against its own GGNFS
   corpus at 99.97% recall (finding 69), and the width measured at **×1.72 on
   the widened queue, +8-9% of wall** (finding 70) — the 1.8-2× projection was
   close. What remains is the `2^32` exclusive position endpoint and the
   14-16 GB whole-area footprint, which is what stops a like-for-like
   comparison against NFS@Home's own `I16e -J 16` geometry for this job.

   **Neither remaining blocker binds a C195 at NFS@Home's geometry.** They
   sieve `2^15 × 2^14` and would prefer `2^15 × 2^15` — `2^30`, half the
   current area limit — so no A=32 work is required for that class of job at
   all. The current design assessment, performance accounting, alternatives,
   and work status are consolidated in **"Current size limits, and what lifting
   them entails"** above; that section is canonical rather than duplicating a
   moving design here.
9. **Dead factor-base parses under `--sq-side 0`.** `fb1` is loaded and
   `fb_fill_logp`'d purely as the throwaway first parse that used to supply the
   q list, but under `sq_side 0` the band comes from the rational base instead
   and `fb1` is untouched until the derivation frees and reloads it. Separately
   `rfb_build` runs twice over `rlim` — a full sieve to 134.2M plus a modular
   inverse per prime, each time. ~15–20 s of startup on snfs236. Irrelevant to
   a multi-day run, worth fixing before anything that restarts the process in a
   loop (parameter sweeps, `cofcheck`).
10. **GPU power-limit sweep — MEASURED 2026-08-17 (finding 61); a floor, not
    a knee.** The premise was that consumer cards ship past their efficiency
    knee and a 60–80% cap buys 15–30% rel/J. **The sieve is not power-limited
    at stock**: it draws ~195 W against a 250 W limit, so only 70–77% binds at
    all, and 70% is `power.min_limit`.

    At the floor (175 W), three runs per setting with byte-identical relations:
    **−10.0% board power for +1.83% time**, giving **+9.1% board rel/J and
    +5.0% whole-box rel/J**. Nine percent of the power for half a percent of
    the clock (2917 → 2902 MHz) — past the knee, as predicted.

    **The board sensor overstated the gain 1.8×** (9.1% against 5.0%), the
    mechanism this item warned about: the cap lengthens the run and the ~105 W
    host constant is paid for the extra time. Grading on the board sensor would
    pick a cap that is too low. Robust to the host constant — 115 W gives 4.8%.

    **The undervolt is the real lever, and it is ~3× the cap.** The card was
    never power-limited; it sat far past its efficiency knee at 2910 MHz /
    1080 mV. Re-pinned to ~2900 MHz at **950 mV**: −28.0% board power for
    +6.7% time, i.e. **+30.2% board and +14.6% whole-box rel/J**, and 50–56 °C
    instead of 69 °C. Both correctness gates were run first — `cofcheck`'s 30
    exact counts and a byte-identical c147 band — because an unstable undervolt
    computes *wrong answers* rather than crashing, which is the failure mode
    that actually threatens a sieve.

    **Applied to this box permanently**, so see the note under "Measured": every
    timing after 2026-08-17 is ~6.7% slower than the numbers in findings 43–60.

    Still unmeasured: **where the knee is**. 900 mV was not tried, so 950 mV is
    known to be past the knee and stable, not known to be optimal. Reopen if
    the last few percent are wanted, on this card or a different one.

    Item-0 verdict: **3.10× time and 3.14× energy** against the CPU box at the
    undervolt, from finding 58's equal-work 3.31× / 2.74× at stock.
11. **Apply breakdown — ANSWERED 2026-08-17 (finding 60), and the answer is
    "no cheap win".** Apply is where the milliseconds are — **54% of the sieve
    chain** on the C194 at its deployment geometry, 21.94 ms against fill's
    15.47 — which is why this was ranked ahead of further fill tuning.

    It is **not** at its memory-system limit: DRAM 9.0%, L2 bandwidth 4.7%,
    1.43 sectors per global load request, 341 waves per SM, SM throughput 71%.
    The opposite character to fill, which is a latency-bound scatter with the
    SMs idle — so the prior-art warning about the L2 transaction ceiling
    (`~/msieve-s/POLYSELECT_OPTIMIZATION_NOTES.md`) describes neither kernel.

    **But no single pipe is saturated either** — LSU 32%, ALU 30%, XU 23%, FMA
    19%, issuing 0.70 instructions per cycle, largest stall short-scoreboard on
    the shared-memory cells. Apply is *issue-limited with a balanced mix*.

    Every identifiable lever was priced and all are small: norm init is 29% of
    apply but `__log2f` recovers only 2.8% of it and dropping the fp64
    cancellation guard only 1.8%; smem atomics are 3.4%; halving the cell width
    moves 0.3%. **The accurate-`log2f` and cancellation-guard decisions both
    stand** — each buys correctness for under 3%.

    What remains is algorithmic: fewer instructions per position or per record,
    i.e. a change to *what* apply computes. That is a much larger piece of work
    than this item was scoped as, and nothing in the tuning space is worth
    doing first. A `NORM_FAST_LOG2` build switch was added alongside the
    existing `NORM_CANCEL_TOL` so both prices can be re-measured on another job
    without editing the kernel.
12. **Unattended operation — checkpoint, resume, clean stop, logging**
    *(added 2026-08-07, owner-stated requirement; scoped 2026-08-11)*. The GPU
    should not idle waiting on a human any more than it should idle waiting on
    the host thread; item 4 is the within-job half of this and this is the
    between-job half. Split into three pieces, in dependency order.

    **12a. Checkpoint / resume / clean stop — BUILT 2026-08-11, VERIFIED ON A
    GPU 2026-08-16**, in `bench/ckpt.h` plus changes to `pipeline.cuh`,
    `bench_main.cu` and `cofac.cuh`.

    **The result is stronger than "no relations lost": a stopped and resumed
    band reproduces the uninterrupted one byte for byte.** A 1500-q c147 band
    (`--cofactor`, 178,406 relations, 23.2 MB) was run three ways — straight
    through; `SIGINT` at q 776 and resumed; `kill -9` at q 671, a 25-byte torn
    line appended by hand to the `.part`, and resumed — and all three files
    have the same MD5. The torn tail was truncated away, and all 178,406
    relations pass `--check-relations`. The same comparison without
    `--cofactor` (where the `--candidates` batch is non-empty) gives
    byte-identical relation *and* candidate files, with no NUL run from a
    mis-sized `ftruncate`.

    Each constraint in the checklist below was exercised as a case rather than
    read: the clean stop drained 125k queued candidates and exited 0 in 0.2 s;
    `--stop-file` stopped at q 896 with a valid resume point, and a stop file
    that already exists refuses to start; the lock named the live pid and
    refused a second writer, released itself on a clean exit, and was taken
    over when the recorded pid was gone; a resume with `--lpb 29` was refused
    on the fingerprint; the single-q path refused to continue a resumable band;
    `--restart` discarded 3.5 MB; `--nq 1500` became 724 remaining after 776
    completed, so the two sessions summed to exactly the band that was asked
    for; and resuming a checkpoint holding 9.6 MB of candidates without
    `--candidates` was refused rather than stranding them. `make check` passes
    (cofcheck's 30 cases, fbgencheck, sqgencheck, fbtest).

    Unverified still: **anything on a multi-day timescale** — every case above
    is seconds to a minute long — and resume across a *reboot* or a different
    machine, where the page cache is cold and the fsync ordering is what is
    actually being tested.

    **One unexplained observation, recorded so it is findable if it recurs.**
    The first interrupt attempt of the session (2026-08-16 22:20, `timeout -s
    INT 15s`, another GPU job having just exited) died with no clean-stop
    output at all: empty `.part`, no sidecar, and the lock left behind. Six
    later interrupts — three direct `kill -INT`, `timeout -s INT`, `timeout
    --foreground -s INT`, and `--stop-file` — all drained and checkpointed
    correctly, so it did not reproduce and the verification above stands. The
    signature to watch for is a band that exits on a signal with no "stopping
    cleanly" line: that means the handler did not run, and the run loses
    everything since the last flush rather than nothing.

    The mechanism rests on one property of the existing code:
    `pipeline.cuh:1228` flushes the cofactor queue *before* enqueuing the
    current q rather than splitting it, so `cofq_flush` never straddles a
    special-q. At every flush the `.part` file therefore holds exactly the
    complete relation set for every q processed so far and nothing partial — a
    checkpoint the run already produced and simply did not record.

    So: after each flush, `fflush` + `fsync` the `.part`, then write a sidecar
    `NAME.part.ckpt` recording the next `(q, rho)`, the byte offset, the
    relation count, and a job fingerprint. Resume truncates the `.part` to that
    offset (which also removes any torn final line) and restarts at that
    `(q, rho)`. Exact, and independent of the factor-base convention.
    `--restart` discards; `--stop-file PATH` stops cleanly for unattended runs;
    a `NAME.lock` refuses a second writer and clears itself if the recorded pid
    is gone.

    **Rejected: inferring the resume point from the primes in the relation
    lines.** It is the obvious approach and it is ambiguous under the
    convention the pipeline actually runs. The full factor base is in force —
    item 3's per-q truncation is not built, all three kernels still take the
    full `fb->n` — so a relation is re-found at *every* band q dividing its
    sq-side norm: measured P(re-found) 70.2%/72.3%, mean 1.82 sq-side primes
    per relation. A line typically carries ~2 in-band primes and nothing
    distinguishes the one that produced it. The failure is asymmetric —
    guessing high skips q and loses relations silently, guessing low only
    re-sieves an overlap that msieve deduplicates — so a guess is survivable
    but not auditable. Note it becomes exact *under* truncation, where the
    producing q is the largest sq-side FB prime; that it depends on an open
    experiment (item 3) is itself the reason not to build resume on it. Keep
    prime-scanning as a separate human-read diagnostic for files with no
    sidecar, reporting a conservative (minimum) resume point.

    Constraints the implementation had to meet, each of which is a way to lose
    or corrupt relations. They are listed because they are the review checklist
    for anything that touches this code again, not because any is outstanding:

    - **The `.part` used to be destroyed.** `pipeline.cuh:1588` removed it on
      any failure and `:1030` opens it `"w"`, which would truncate the file
      being resumed from. The reframe is that `.part` is the durable artifact
      and the rename to the final name is only the "band completed" marker.
      It is now kept on failure **iff a sidecar was actually written** — not
      merely because a relation file was opened. Under `--cofactor` the file is
      empty until the first flush and that flush is what writes the first
      checkpoint, so an unresumable `.part` holds nothing; keeping one would
      strand every rerun on the startup refusal, which is the one wedge an
      unattended queue cannot recover from.
    - **A failed checkpoint write must not fail the band.** A missed checkpoint
      costs at most one flush of re-sieving; aborting costs everything since
      the last good one. It warns once and continues.
    - **Never checkpoint between flushes under `--cofactor`.** At the top of an
      arbitrary q the queue holds up to a flush of candidates that are not in
      the file; recording that position as the resume point drops them.
    - **A clean stop must drain.** SIGINT/SIGTERM sets a flag, checked at the
      top of the q loop after the next `(q, rho)` is pulled: flush the queue,
      fsync, checkpoint that pair, exit 0. A planned stop then loses nothing,
      against up to one flush of re-sieving after a crash (~67 q on the c183,
      but **686 q on the SNFS job**, which enqueues 191 records/q instead of
      1,956 — `pipeline.cuh:1385`). Second signal exits immediately; the
      previous checkpoint stays valid.
    - **The derived scale and allowance must be carried in the checkpoint.**
      They are derived once from the *first* q of the band
      (`bench_main.cu:1312`, `q0 = ql[0].q`) and held band-wide — that is the
      fixed-scale approximation under "Known defects". Re-deriving them from
      the resume q would silently change the survivor gate partway through a
      run, which is exactly the kind of inconsistency an item-0 verdict cannot
      absorb. Measured on the c147: the carried gate is scale 0.900/1.100,
      bound 56/65, against a fresh derivation at the resume q of 1.250/2.925,
      bound 76/169. Not a rounding difference — a different sieve.
    - **A relation that verifies proves the polynomial, not the parameters.**
      `--check-relations` (`bench_main.cu:732`) is host-only and cheap, so
      spot-checking a few lines of the `.part` is worth doing — but logI, J,
      lambda, mfb, lpb, sq-side and the FB convention can all differ while
      every line still reconstructs. Hence the fingerprint; refuse a mismatch
      rather than appending incoherent yield to a valid-looking file. The
      fingerprint therefore covers the polynomial, lim/lpb/mfb on both sides,
      logI/J, sq-side, **`--maxbits`** and **the whole cofactor configuration**
      — dropping `--cofactor` on a resume switches the file from queue-emitted
      to trial-division-only *and* the checkpoint discipline from
      flush-anchored to a timer, and the ECM/rho bounds decide which cofactors
      split at all.
    - **Sample only the checkpointed prefix.** A torn final line past the
      offset is the normal shape of a `kill -9` and is about to be truncated
      away; scanning to EOF would refuse a sound file.
    - **Take the lock before anything destructive.** `--restart` unlinks the
      `.part`, so acquiring the lock after that (or after the factor-base
      parse) lets it delete a running siever's output and report the conflict
      afterwards, with the victim still writing to an unlinked inode and its
      final rename doomed to ENOENT.
    - **Resume needs the original band.** The single-q fallback builds `ql[0]`
      from `--q`/`--rho` and never consults the checkpoint, so resuming into it
      would truncate the `.part`, sieve one unrelated q, call the band complete,
      rename over the final name and delete the sidecar. The band source is not
      in the fingerprint, so this is checked separately.
    - **`--target-rels` counts what is already on disk**, or a resumed run
      overshoots by the whole prior amount.
    - **`--candidates` has its own `.part`** and must be truncated to a
      consistent point in the same checkpoint — with the same length guard the
      relation file gets. Unguarded, a short file is *extended* by `ftruncate`
      and the gap is a run of NULs no parser survives; and because
      `--candidates` is not in the fingerprint, a resume that omitted it would
      record `cand_bytes = 0` and the next resume that supplied it again would
      truncate the whole file away.
    - **A lockfile**, because two processes appending to one `.part` is silent
      corruption and a job queue makes that a realistic accident. It must clear
      itself when the recorded pid is gone: `SIGKILL`, OOM, power loss and the
      second-signal `_exit` all skip the unlink, and a lock needing a human
      defeats unattended operation more thoroughly than the race it prevents.
    - **`--nq` is a per-session budget** once a run can resume. It is reduced by
      the completed count, so the messages that echo it must add that count back
      or they print a number nobody typed. The wind-forward to the exact `(q,
      rho)` also must not be charged against the generator's own limit, or the
      band stops early and reports a range exhausted that is nowhere near it.

    **12b. Logging — BUILT 2026-08-16**, in `bench/runlog.{c,h}` plus the
    reporter and warning sites in `pipeline.cuh` and the header block in
    `bench_main.cu`. `--log PATH` appends a run log; `--log-every S` sets the
    record period (default 300 s, ~864 lines over three days, so **no
    rotation**). Both are `--pipeline`-only and resolved through
    `boinc_resolve_filename_s()` like every other named output.

    A tee was the obvious version and worth less than it looks: finding 53 is
    that host contention costs up to 29% of wall while every GPU counter reads
    normal, and that any cross-box wall or ETA comparison is invalid without
    knowing host load on both. So each record carries `GPU-accounted / wall`
    (the running form of the band summary's line, built from the same terms),
    GPU utilisation, board watts and the host load average alongside q,
    relations, candidates, ms/q, percent and ETA. The header block carries the
    commit (`git describe --dirty`, stamped into `runlog.o` alone so a commit
    costs no CUDA recompile), the full argv, the job fingerprint, the card with
    its PCI ID and driver, the **geometry labelled `I14e`/`I15e`**, the
    **factor-base convention** with entry counts and whether the base is
    truncated, the derived gate, and the resume point. Finding 55 is what that
    header exists to prevent.

    Verified on a GPU 2026-08-16, c147: header, heartbeat, band-end record,
    the `isatty(1)` split (a redirected console gets whole lines every five
    minutes — its own constant, *not* `--log-every` — instead of a `\r` line
    and a stray `\033[K`), and NVML binding. The `resume` note was exercised by
    12a's verification the same day — a resumed session appends its own header
    block, carrying the resume point and the narrowed band. **Not yet
    exercised: a BOINC-resolved log path.**

    Three implementation notes worth keeping:

    - **`nq` and `rel` mean the same thing in every record**, including the
      band-end one: totals across sessions, with `rel` counted the way the
      heartbeat counts it (what reached the file, not the subset trial
      division finished — 374 against 111 on one smoke run). Session-only
      aggregates carry their own `_session` keys. A log where one key means two
      things is worse than one field short, because nothing in the file says
      which reading applies to which line.

    - **NVML is bound by PCI bus ID, never by CUDA ordinal.** NVML enumerates
      by PCI order while CUDA can be reordered by `CUDA_DEVICE_ORDER` and
      renumbered by `CUDA_VISIBLE_DEVICES`, so an ordinal reads another card's
      watts whenever those disagree — silently, and precisely on the multi-GPU
      hosts of item 13. It is `dlopen`ed, so there is no build dependency and a
      host without the driver library logs `n/a`.
    - **Every record carries monotonic elapsed seconds as well as a
      timestamp.** Observed on this box during the verification run: WSL2
      resyncs its clock to the Windows host, and one band's records came out
      with a two-second backwards step mid-run, which is enough to make a
      record appear to precede the one above it. Compute rates from the
      elapsed column; the timestamp is for lining a run up against a meter or
      another machine.

    **12c. The queue itself.** A job list consumed without intervention, so
    finishing one job starts the next rather than leaving the card idle
    overnight. Depends on 12a; unscoped beyond that. **Item 9 stops being
    cosmetic here**: its ~15–20 s of redundant startup is noise in a multi-day
    run and real overhead in anything that restarts the process per job.
13. **Validate the BOINC GPU assignment — CLOSED 2026-08-17.** Greg Childers,
    who reported the original failure (every task on a multi-GPU host landing
    on device 0), reviewed and signed off on the assignment change, and a BOINC
    queue is now running with it. Both things this item said could not be
    checked here — that the client emits `<gpu_device_num>` for this app
    version's plan class, and that concurrent tasks land on distinct cards —
    are answered affirmatively by a live queue on a real multi-GPU host.

    What is written below stayed open for one session and is kept only as the
    recipe for re-checking the read path after a change to it, since a silent
    regression still looks exactly like success on a single-card box:

    Testable here, no second card and no client: `boinc_get_init_data()` reads
    the `init_data.xml` in the working directory, so a `HAVE_BOINC=1` build run
    in a directory holding a hand-written one exercises the whole read path.
    Four cases worth pinning, each identified by its stderr line — (a) an
    NVIDIA assignment is read and used; (b) an assignment overrides a
    conflicting `--device`, with the notice printed; (c) a non-NVIDIA
    `gpu_type` is refused rather than acted on; (d) no assignment falls through
    to `--device` and then to CUDA's default. An assignment of 1 on a one-card
    box should also produce the bounded "sees only 1 device" refusal rather
    than an invalid-ordinal error.

    Not testable here, and both are project-side rather than application-side:
    that the client emits `<gpu_device_num>` at all for this app version's plan
    class, and that concurrent tasks land on distinct cards. Those need a
    multi-GPU host running the real client, i.e. the reporter who saw the
    original failure.
14. **Use the survivor cell's real resolution — MEASURED AND CLOSED 2026-08-18
    (finding 66): no win, do not build it.** `las_scale` derives the byte scale
    from las's 255-value cell (`poly.c:584`) although ours is 16-bit with
    `CINIT` 4096, so one byte unit is 0.82 bits and 10–13% of the survivor list
    is admitted on rounding error alone.

    **Removing them buys nothing.** Holding the gate fixed in bits and raising
    the scale to 4.0, on an idle box, alternated: **112.44 against 112.43 ms/q**
    — nothing, against a ~1.3% run-to-run spread — while survivors fall 10.3%
    (71,580 → 64,241). Yield is neutral too (6,743 → 6,740).

    The reason is that **the survivor count is not a cost driver on this
    pipeline.** Fill and apply are ~70 ms of the 112 and are per *position*;
    the whole trial-division block is 14 ms and moves 0.25 ms. Candidates/q is
    flat to 0.04% (1590.68 → 1590.08), so the noise survivors never reach
    cofactorisation — the rank scan and survivor filter kill them for 0.5 ms
    total.

    So `las_scale`'s 8-bit inheritance is harmless rather than a defect.
    Recorded chiefly so the inference "fewer survivors must be faster" is not
    made again: on this pipeline the survivor list is nearly free downstream,
    which also means **survivor counts are a poor proxy for work** in any
    cross-siever comparison (see item 7's funnel numbers).
15. **Older cards, and the shared-memory ceiling that blocks them — code
    read 2026-08-20, NOT tested on any such card.** **Fixed 2026-08-21:** both
    the production pipeline and standalone apply benchmark now query
    `cudaDevAttrMaxSharedMemoryPerBlockOptin` from the selected device at runtime;
    the default remains `--region 14`. `GPU_ARCH_all` starts at sm_80, but the
    apply path no longer assumes one architecture family's opt-in limit.

    Before the fix, both `pipeline.cuh` and `bench_kernels.cu` hardcoded
    `101376` bytes as "the opt-in limit". CUDA does not define one universal
    value; representative limits are:

    | target | max dynamic smem per block | old 101376-byte bound |
    |---|---:|---|
    | sm_70 Volta | 96 KB | too high |
    | sm_75 Turing | 64 KB | **too high — the opt-in call fails** |
    | sm_80 A100 | 163 KB | **too low** |
    | sm_86 / sm_89 / sm_120 | 99 KB | matched |
    | sm_90 Hopper | 227 KB | **too low** |

    Apply sizes its shared memory at `(1 << log_region) * 2 + nslice_pow2 * 2`,
    so `--region 15` needs roughly 66 KB and `--region 16` roughly 131 KB. The
    old literal therefore prevented region 16 on A100/Hopper even though the
    hardware permits it, while it would have admitted sizes that are illegal on
    some older devices. The runtime query fixes both directions.

    `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
    remains the opt-in mechanism; the requested value is checked against the
    selected device's reported limit before that attribute is set or the apply
    kernel is launched. This is capability detection only: `--region 14` is
    still the default because larger regions have not been shown to be faster.

    **What is NOT established.** No card older than the current sm_80 build
    floor has been qualified by this change. Lowering `GPU_ARCH_all` would still
    require compilation and byte-comparison testing on that hardware; fill-kernel
    launch-bounds/shared-memory tuning remains a separate constraint.
16. **Build wall time — MEASURED 2026-08-20, and the cause is `CF_LMAX=4`.**
    A full `make all` is ~12.5 min on this 16-thread box and about 30 on a
    busy Vast.ai 5090. The Makefile's timing table (`Makefile:43-55`) claims
    214 s and blames sm_120's ptxas. **That table was measured 2026-08-11
    (efd7219), before the 4-limb/ECM cofactor dispatch landed 2026-08-19
    (e4a47f2), and it is now wrong by 3.5x.**

    Measured here, `make -j8`, clean tree per arm, one target at a time:

    | target | `CF_LMAX=3` | `CF_LMAX=4` (default) | cost of the 4th limb |
    |---|---:|---:|---:|
    | sm_120 | 277 s | **754 s** | **2.72x** |
    | sm_80 | 15 s | 26 s | 1.73x |

    Three things follow.

    1. **The 4-limb dispatch is the regression, not the target count.** At
       `CF_LMAX=3` sm_120 returns to 277 s, near the table's 215 s; the
       residual is plausibly the ECM rewrite that shipped in the same commit.
    2. **`-t 0` already works, so dropping targets buys nothing.** The full
       six-target fat build measured **750 s** against sm_120 alone at 754 s.
       The fat binary is free and adding sm_90 cost nothing measurable — the
       one target that cannot be dropped is the expensive one. This confirms
       the stale table's *conclusion* even though its numbers are wrong.
    3. **The ptxas asymmetry widened.** sm_120/sm_80 was ~15x when the table
       was written and is **29x** now (754/26). The 4th limb costs sm_120
       disproportionately, so this is a ptxas scaling problem on Blackwell
       rather than a linear code-size effect.

    **For iteration today:** `make GPU_ARCH=120 CF_LMAX=3` is 277 s and is a
    shippable binary — the cofcheck gate pins 3-limb and 4-limb output as
    byte-identical, and `CF_LMAX` is deliberately not a `DEFS` value so such a
    build still emits relations (`Makefile:112-130`). An Ampere or Ada box
    with `GPU_ARCH=native` is 26 s. Neither helps a release build.

    **Not yet investigated:** why the 4th limb triples ptxas time on sm_120.
    `bench_kernels.cu` is a single 138 KB translation unit that includes the
    125 KB `cofac.cuh`, and that is the only site including it
    (`bench_kernels.cu:2682`), so splitting the cofactor kernels into their
    own `.cu` would both parallelise across TUs and stop a sieve-kernel edit
    from recompiling the cofactor templates. That is the first experiment;
    instantiation count and register pressure are the obvious suspects.
