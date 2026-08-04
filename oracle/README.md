# oracle/ — frozen inputs for the cuda-sieve feasibility probe

Everything in this directory is **frozen**. `../test-sieve/` is the owner's live
working directory and its polynomial is expected to be replaced with a different
number; these copies were taken on **2026-08-01** so the target job survives that.

These files are correctness oracles and CPU performance baselines for the
project's primary **GPU-resident relation-collection** goal. Their presence does
not imply that CADO or GGNFS cofactorization remains in the production data
path. A GPU-sieve + CPU-cofactor hybrid is tracked separately as an optional
deployment mode.

## The target job

**C183, GNFS, degree 5.** This is the number every graded result in
`../prototype.md` refers to.

| file | what it is |
|---|---|
| `input.job` | GGNFS-format job: poly + `rlim`/`alim`/`lpbr`/`lpba`/`mfbr`/`mfba`/lambdas |
| `result.job` | same, as written back by `test_sieve.sh` (adds `lss: 0`) |
| `c183.poly` | **CADO-format poly, generated here** — `../test-sieve/cado_tmp.poly` was for a *different* number (a C203) and was never regenerated for this job |
| `input.job.afb.0` | GGNFS algebraic factor base for this poly, `alim=134.2M` (61 MB) |
| `ggnfs_test_sieve_15e.txt` | the `gnfs-lasieve4I15e` test sieve this project's per-q budget is derived from |
| `test_sieve.sh`, `cado_test_sieve.sh`, `estimate_norms.py` | the owner's tooling as it stood, for reproducing the above |
| `MANIFEST.sha256` | checksums for all of it, including the git-ignored large binaries |

## Correctness gates and captures (added 2026-08-03/04)

These are review artifacts, not frozen inputs — they are reproducible, and the
commands that produce them are in `../prototype.md` under "Artifacts and how to
reproduce". They are checksummed here because the gates they support are cited
as passed.

| file | what it is |
|---|---|
| `cado-after-sieve-survdump.patch` | 3-line CADO patch dumping the `after_sieve` survivor `(a,b)` set, which `-batch-print-survivors` does **not** do (it emits the post-TD cofactor list — 1,851 records, not 797,028). Apply, rebuild `las`, keep the binary **outside** the CADO tree, revert the source. Enabled by `CUDASIEVE_SURVDUMP=<path>`; inert when unset. |
| `c183.q120000053.after_sieve_ab.powlim.txt.gz` | 795,845 `(a,b)` at `-powlim0/1 32767` — **the oracle to gate against** |
| `c183.q120000053.after_sieve_ab.nopowlim.txt.gz` | 797,028 `(a,b)` at default `powlim`; kept only to reproduce the `powlim` finding |
| `c183.q120000053.relations_ab.txt` | las's 37 relations at this q |
| `c183.q120000053.cofac_candidates.txt` | 1,851 `a b cofac0 cofac1` from `-batch-print-survivors` |
| `las_q120000053_after_sieve_powlim.log`, `las_q120000053_batchsurv.log` | capture logs |
| `relcontain_band.c` | the relation-containment harness: inverts `(a,b)` through **our** basis, attributes relations to a lattice by `a ≡ rho·b (mod q)`, classifies contained / outside-region / in-region-miss |
| `relation_containment_band.txt` | its result over the frozen band at the default bound: **3,026 of 3,026** in-region relations contained, zero misses, over 67 lattices |
| `relation_containment_bound128.txt` | the same gate rerun at the tightened bound 128 — identical result, which is what licenses adopting it |
| `c183.q120000000-120001000.relations.default.txt` | the 3,162 las relations that harness runs against |
| `bound_sweep.sh` | the coarse survivor-bound sweep, `lambda1` in steps of 0.1. **Note the resolution:** one step is 4.08 units of bound, so it never sampled the cliff edge |
| `bound_sweep_fine.sh` | the 1-unit refinement that did |
| `bound_sweep_results.txt` | both passes, merged. **The zero-loss floor is bound 128** — 2.46× fewer survivors than the default 143 for zero relations lost. Use **130** in the GPU path until the containment gate is rerun at a candidate bound; the floor gives no margin against our two-directional one-unit norm error |
| `c183.q120000000-120010000.cofac_candidates.tar.gz` | **1,062,811 cofactor candidates over 548 special-q** in [120000000, 120010000] — the real C183 `lpb 31/32` post-sieve population, 1,939 records/q, for the controlled cofactorization rerun (31 MB) |

**A caution on the relation artifacts.** Containment establishes that las's
in-region relations are all survivors of ours. It does **not** establish yield
equivalence: 20 of the band's 67 lattices use a different (in fact
better-reduced) basis and therefore cover a different `(a,b)` region. Final
relation yield has to come from cofactoring our own region — see `PARITY.md`.

## Parameters

```
rlim  67100000        lpbr 31        mfbr 60        rlambda 2.35
alim 134200000        lpba 32        mfba 92        alambda 3.5
```

`mfba=92` with `lpba=32` means **three large primes on the algebraic side**. It
makes this a useful stress case for GPU cofactorization and a demanding test of
YAFU's existing GPU path; it does not make CPU cofactorization an architectural
requirement. See the post-sieve section of `../prototype.md`.

## Operating point (locked)

- CADO: `-A 29` (≡ `-I 15`), `-adjust-strategy 0`, `-sqside 1`
- Job q-range: `[50M, 190M]` — to be sieved for real on NFS@Home
- Frozen benchmark band: `-q0 120000000 -q1 120001000`
- Sensitivity points: `q0 = 50M` and `q0 = 190M`

## What is NOT here, and why

- **CADO factor bases.** `../test-sieve/cado_roots1.gz` belongs to the C203, not
  this number. Both sides must be generated fresh:
  ```
  makefb -poly c183.poly -side 0 -lim  67100000 -out cado_roots0.gz -t 16
  makefb -poly c183.poly -side 1 -lim 134200000 -out cado_roots1.gz -t 16
  ```
  Side 0 has never existed at all — `las` computes the rational side on the fly.
- **`gnfs-lasieve4I15e`.** Lives in `../test-sieve/`; the owner is changing the
  poly, not the binaries.

## Provenance of the test sieve

`ggnfs_test_sieve_15e.txt` was produced by `test_sieve.sh`, which launches **one**
siever process with no backgrounding (`test_sieve.sh:187`). All of its rates are
therefore **single-core**, and its "ETA: 315d" is a one-core ETA. The derived
constant that matters — **~3.1 core-seconds per special-q, flat across the whole
q-range** — is per core.
