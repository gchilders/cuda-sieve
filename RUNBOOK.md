# Running a job

Every command here was run end to end on a c123 (factored) and a c151 (sieved
and verified), 2026-08-05. Nothing in it is from memory.

## What you need

1. **A polynomial file.** A GGNFS `.job` works directly — `n:`, `skew:`,
   `c0:`..`c5:`, `Y0:`, `Y1:`. The extra keys (`rlim:`, `lpbr:`, `alambda:` …)
   are ignored by both our loader and CADO's `makefb`, so `result.job` needs no
   conversion. A CADO `.poly` works too.

2. **An algebraic factor base from CADO's `makefb`.** This is required, not
   optional: the GGNFS `.afb.0` format carries neither p = 2 nor prime powers,
   and without p = 2 the algebraic cofactors stay even, which Montgomery
   arithmetic cannot split. `bench` refuses to produce relations without it.

   ```sh
   makefb -poly JOB.job -lim ALIM -maxbits LOGI -out JOB.roots1
   ```

   `-maxbits` is the sieve's `logI`. 5 s for a c151.

## Where to put things

`makefb` and `bench` take paths and do not care about the working directory.
**msieve does** — it looks for `cub/`, the lanczos kernels and its default
`msieve.dat` / `msieve.fb` / `worktodo.ini` relative to wherever you launch it.
So: one directory per job, and run msieve from inside it.

```
~/nfs/c151/
  c151.job                 copy of result.job — the poly
  c151.roots1              makefb output
  msieve.dat               relations: sieve STRAIGHT into this, no copy step
  msieve.fb                N, poly and bounds (see below)
  worktodo.ini             N on one line
  cub -> ~/msieve_nfsathome_basement/cub
  lanczos_kernel.ptx    -> ~/msieve_nfsathome_basement/lanczos_kernel.ptx
  lanczos_kernel.fatbin -> ~/msieve_nfsathome_basement/lanczos_kernel.fatbin
  sieve.log
```

Run `makefb` and `bench` from anywhere with `-out` / `--relations` pointing
into the job directory; run `msieve` from inside it.

Two practical notes:

- **`--relations msieve.dat` writes directly**, saving a full duplicate of the
  relation file — ~8 GB on a c151 at 65M relations (~128 bytes each). Budget
  ~15 GB per job including msieve's `.mat` / `.cyc` / `.chk` / `.dep`.
- **`bench` writes `NAME.part` and renames on success.** If a long run is
  interrupted, `mv msieve.dat.part msieve.dat` keeps everything written so far.
  Note it OVERWRITES rather than appends, so if you sieve in several runs, send
  them to separate files and `cat` them together. `--target-rels` is there so
  you can do it in one.

## Sieving

```sh
bench --pipeline --cofactor \
      --poly JOB.job --cadofb JOB.roots1 \
      --logI 14 \
      --qrange 15000000: --target-rels 65000000 \
      --relations ~/nfs/c151/msieve.dat
```

That is the whole command. **`rlim`, `alim`, `lpbr`, `lpba`, `mfbr`, `mfba`
and both lambdas are read from the `.job` file**, and the byte scale and
survivor allowance are derived from the polynomial. Each one is printed with
its source, so a run says where every number came from:

```
  job file: rlim 16700000, alim 33500000, lpbr 29, lpba 30, mfbr 56, mfba 59
  job file:  alambda 2.5 x log2(alim 33500000) = 62.50 bits (side 1)
  job file:  rlambda 2.4 x log2(rlim 16700000) = 57.60 bits (side 0)
params from q=15000017: side 1 log2(maxnorm)=... scale=... allowance=62.50 bound=...
```

Anything you state explicitly wins over the file; anything the file carries
wins over the derivation. A CADO `.poly` carries none of these keys, so there
you either state them or let them derive.

- **`--logI`** is the siever width: `gnfs-lasieve4I14e` → 14. **`--J` defaults
  to `2^(logI-1)`** (CADO's convention), so it no longer needs passing.
- **`--region`** must be ≤ `--logI`; the default is 14, so only pass it for
  I=13 or smaller (`--region 13`).
- **`--qrange MIN:`** with the upper end omitted means "up to the factor-base
  bound". Pair it with **`--target-rels N`** to stop when you have enough
  rather than guessing a range. Checked at flush boundaries, so it overshoots
  by well under 1% on any real target.
- Side 1 is the algebraic side and carries the special-q. Side 0 is rational.

### Progress output

One `\r` line, updated every 30 s, reported against whichever goal is in force:

```
q=15020857  1222 q  78251 rel  2608 rel/s  26.1%  ETA 0h 01m
```

With `--target-rels` the percentage and ETA track the relation target — which
is what you want, because `--qrange MIN:` makes the q count meaningless (it is
the whole factor base, 1,088,865 special-q for the c151). Without a target they
track the q count instead.

The relation figure advances at queue flush boundaries rather than per q, so it
moves in steps and the rate is slightly understated early in a run. It settles.

### Lambda: GGNFS and CADO do not mean the same thing

You no longer have to do this conversion by hand — a `.job` file's lambdas are
read and converted, and the conversion is printed. Read this anyway, because
it is the one parameter where a mistake costs relations with nothing failing.

The two conventions differ by which quantity they multiply:

| | lambda is in units of | cofactor bits allowed |
|---|---|---|
| **GGNFS** | `log2(lim)` | `lambda * log2(lim)` |
| **CADO** | `lpb` | `lambda * lpb` |

The GGNFS test-sieve's own suggestion confirms it: *"Suggested rlambda: 2.32
(mfbr=56 / log2(rlim=16700000))"* — i.e. GGNFS's lambda is `mfb / log2(lim)`.
CADO's automatic is `0.3 + mfb / lpb`.

**To state one yourself, use `--allowance` / `--allowance0`, in BITS.** That is
the quantity both conventions are really expressing, and it is what the
survivor bound is computed from:

```
bound = (uint32_t)(allowance * scale + 1)
```

(CADO writes this as `(unsigned char)`, which is right for its 8-bit cells. We
sieve 16-bit cells with CINIT 4096, so bounds above 255 are legal here and the
cast has to be wider — an earlier version of this file, and the banner, both
printed the 8-bit truncation.)

Worked example, the c151 above (`alambda 2.5`, `alim 33.5M`, `lpba 30`):

```
GGNFS:  2.5 * log2(33500000) = 2.5 * 25.00 = 62.5 bits   -> --allowance 62.5
CADO :  the same 62.5 bits would be lambda = 62.5/30 = 2.083
CADO's automatic (0.3 + 59/30 = 2.267) would give 68.0 bits — more generous
```

and the rational side (`rlambda 2.4`, `rlim 16.7M`, `lpbr 29`):

```
GGNFS:  2.4 * log2(16700000) = 2.4 * 23.99 = 57.6 bits   -> --allowance0 57.6
```

The byte scale is always derived from the polynomial (CADO's formula,
`las-norms.cpp:237`) and **an explicit `--allowance` / `--scale` overrides
it**, so the two compose: derive what you don't know, state what you do.

If you would rather work in CADO's units, `--lambda1` / `--lambda0` take
CADO-style lambdas and 0 means "use CADO's automatic". They override a `.job`
file's GGNFS-unit lambdas, and say so when they do.

> `--auto-params` used to gate the derivation. It is now the default and the
> flag is accepted and ignored, so old scripts keep working; drop it when you
> next touch them.

## Checking the output

```sh
bench --check-relations ~/nfs/c151/msieve.dat --poly JOB.job --lpb 30 --lpb0 29
```

Rebuilds both norms from `(a,b)` and the polynomial for every relation and
requires each recorded factor to divide exactly, both norms to reduce to 1, and
every prime to sit within its side's lpb. Run it before spending hours on
filtering — it catches a truncated factor, a dropped power of two, or a
composite emitted as prime, none of which any other check sees.

## Post-processing with msieve

Relations come out in GGNFS/msieve format already — `a,b:rational:algebraic`,
hex factors, rational side first, special-q included. So:

```sh
cd ~/nfs/c151          # relations are already here, sieved straight into msieve.dat
cat > msieve.fb <<EOF
N <decimal N>
SKEW <skew>
R0 <Y0>
R1 <Y1>
A0 <c0>
...
A5 <c5>
FRMAX <rlim>
FAMAX <alim>
SRLPMAX <2^lpbr>
SALPMAX <2^lpba>
EOF
echo "<decimal N>" > worktodo.ini

# msieve looks for these RELATIVE TO THE WORKING DIRECTORY
ln -s /path/to/msieve/cub .
ln -s /path/to/msieve/lanczos_kernel.ptx .
ln -s /path/to/msieve/lanczos_kernel.fatbin .

msieve -s msieve.dat -l msieve.log -nc1 -v    # filtering
msieve -s msieve.dat -l msieve.log -nc2 -v    # linear algebra (GPU)
msieve -s msieve.dat -l msieve.log -nc3 -v    # square root -> factors
```

There is **no `N` header line** in `msieve.dat`; N comes from `msieve.fb`.

We generate **no free relations**. CADO made 122,390 for the c123 and msieve's
filtering still cleared its target without them, so they are not load-bearing at
that size. Watch for it on larger jobs.

If `-nc1` reports *"filtering wants N more relations"* and singleton removal
collapses the set (e.g. 8.6M relations against 15.0M unique ideals reducing to
279), that is simply not enough relations — sieve more, do not tune.

## Measured

| | c123 | c151 |
|---|---:|---:|
| logI | 13 | 14 |
| lpb (alg / rat) | 29 / 28 | 30 / 29 |
| **ms per special-q** | **10.14** | **24.75** |
| relations / q | 272.1 | 79.8 |
| relations / sec | 26,839 | 3,223 |
| reconstruction gate | all pass | 48,420 / 48,420 |

The c123 factored end to end: 1,093 s sieving, 171 s filtering, 61 s linear
algebra, 153 s square root.
