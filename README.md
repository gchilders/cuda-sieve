# cuda-sieve

`cuda-sieve` is an experimental, standalone CUDA implementation of the
lattice-sieving relation-collection pipeline used by the Number Field Sieve.
It builds factor bases, sieves both sides of a special-q lattice, performs
trial division and GPU cofactorisation, and emits relations for msieve.

The implementation has been exercised on NVIDIA Ampere, Ada, and Blackwell
GPUs. It is research software: the correctness gates pass, but the current
limitations and unfinished experiments in [`bench/STATUS.md`](bench/STATUS.md)
are part of its release status.

## Build and test

You need a Linux build environment, GNU Make, a C11 compiler, a C++17
compiler, and an NVIDIA CUDA toolkit. The supplied Makefile targets compute
capability 8.0 and newer.

```sh
make -C bench

# Generate the large factor base used by the GPU golden tests. It is ignored
# by Git and can be regenerated at any time.
./bench/fbgen --poly oracle/c183.poly --lim 134200000 \
    --maxbits 15 --threads 6 --out oracle/c183.fb1

# Same deal for the C147, which is the workload behind the device timings in
# bench/RESULTS.md. --maxbits must match the --logI you sieve at; 14 is what
# the frozen file uses, and --lim defaults to alim from the .job file.
./bench/fbgen --poly oracle/c147.job --maxbits 14 \
    --threads 6 --out oracle/c147.roots1

make -C bench check
```

Both files are pinned in [`oracle/MANIFEST.sha256`](oracle/MANIFEST.sha256);
`cd oracle && sha256sum -c MANIFEST.sha256` confirms a regeneration matched.

The default build produces a portable binary covering compute capability 8.0
through 12.0, which takes a few minutes because `ptxas` is slow on the newest
target. `make -C bench GPU_ARCH=native` builds only for the card in this machine
— much faster on an Ampere or Ada host, and the resulting binary will not run
anywhere else.

For a real job, generate its algebraic factor base and run the complete
pipeline:

```sh
./bench/fbgen --poly JOB.job --lim ALIM --maxbits 14 \
    --threads 6 --out JOB.roots1

./bench/bench --pipeline --cofactor --poly JOB.job --fb1 JOB.roots1 \
    --logI 14 --qrange 15000000: --target-rels 65000000 \
    --relations msieve.dat
```

The job's `rlim`, `alim`, large-prime bounds, cofactor bounds, and polynomial
are read from the `.job` file. See [`RUNBOOK.md`](RUNBOOK.md) before starting
a long run; it documents supported inputs, output handling, parameter
precedence, verification, and msieve handoff.

## Optional BOINC application build

BOINC integration is compiled only when `HAVE_BOINC=1`. The normal build has no
BOINC header or library dependency; its wrappers compile to no-ops and the
recurring per-q progress path is omitted. With distribution-provided BOINC
development files, a typical build is:

```sh
make -C bench bench HAVE_BOINC=1 GPU_ARCH=all \
    BOINC_CPPFLAGS="-I/usr/include/boinc"
```

When building against a BOINC source tree instead, point the compiler and
linker at that tree explicitly:

```sh
make -C bench bench HAVE_BOINC=1 GPU_ARCH=all \
    BOINC_CPPFLAGS="-I/path/to/boinc/api -I/path/to/boinc/lib" \
    BOINC_LIBS="-L/path/to/boinc/api -L/path/to/boinc/lib -lboinc_api -lboinc -lpthread"
```

The BOINC build:

- initialises and finishes through the BOINC API;
- requests normal host-thread priority for the CUDA feeder thread;
- accepts BOINC's `--device N` CUDA-device argument;
- resolves every explicitly supplied input or output filename through
  `boinc_resolve_filename_s()` and follows native BOINC output links before
  staging and renaming result files; and
- reports a nondecreasing fraction done at special-q boundaries, normally no
  more than once per second, with an immediate end-of-band update. The
  denominator is, in priority order, `--target-rels`, `--nq`, a bounded
  generated q range, or the length of `--qlist`.

The sieve reserves the final one percent for final cofactor flushing, output
close/rename, and cleanup. A successful BOINC workunit reports 100 percent just
before `boinc_finish(0)`.

A BOINC workunit command line should therefore name all files using the logical
names declared in its workunit template, for example:

```text
--pipeline --cofactor --poly job_file --fb1 roots_file --logI 14 \
--qrange 15000000:16000000 --relations result_file --device 0
```

This integration does not yet add checkpoint files. BOINC can suspend and
resume a live process, but a process that exits and is restarted begins the
current band again.

## Repository map

- [`bench/`](bench/) contains the implementation, build, and test programs.
- [`bench/STATUS.md`](bench/STATUS.md) is the current architecture, validation,
  known-defect, and open-experiment summary.
- [`bench/RESULTS.md`](bench/RESULTS.md) is the chronological benchmark lab
  notebook, including superseded and refuted findings.
- [`oracle/`](oracle/) contains the small, checked-in CADO/GGNFS parity data and
  a manifest for the larger reproducible artifacts that Git ignores.
- [`prototype.md`](prototype.md) records the design and review history.

## License

Original material in this repository is dedicated to the public domain under
[CC0 1.0 Universal](LICENSE), including its as-is warranty and liability
disclaimers. A few files derived from CADO-NFS remain under
LGPL-2.1-or-later; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

