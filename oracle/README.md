# oracle/ — frozen inputs for the cuda-sieve feasibility probe

Everything in this directory is **frozen**. `../test-sieve/` is the owner's live
working directory and its polynomial is expected to be replaced with a different
number; these copies were taken on **2026-08-01** so the target job survives that.

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
| `MANIFEST.sha256` | checksums for all of it |

## Parameters

```
rlim  67100000        lpbr 31        mfbr 60        rlambda 2.35
alim 134200000        lpba 32        mfba 92        alambda 3.5
```

`mfba=92` with `lpba=32` means **three large primes on the algebraic side** — the
reason cofactorization dominates the verdict. See the Amdahl section of
`../prototype.md`.

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
