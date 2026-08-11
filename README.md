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

You need a Linux build environment, GNU Make, a C11 compiler, and an NVIDIA
CUDA toolkit. The supplied Makefile targets compute capability 8.0 and newer.

```sh
make -C bench

# Generate the large factor base used by the GPU golden tests. It is ignored
# by Git and can be regenerated at any time.
./bench/fbgen --poly oracle/c183.poly --lim 134200000 \
    --maxbits 15 --threads 6 --out oracle/c183.fb1

make -C bench check
```

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

