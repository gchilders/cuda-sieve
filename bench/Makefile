# Standalone bucket-fill benchmark for the GPU NFS sieve feasibility probe.
# No CADO, no msieve, no GMP. Builds or loads factor bases directly.
# Pattern copied from ~/msieve-s/gerbicz_bench/Makefile, which worked well.

NVCC      ?= nvcc
CC        ?= gcc

# sm_120 native (RTX 5070, Blackwell consumer), sm_86 native (RTX 3090/A10).
# PTX fallback is compute_80, NOT compute_89: the driver only JITs PTX to a
# target >= the virtual arch, so a compute_89 fallback cannot load on any
# Ampere card at all -- the previous default was unrunnable on a 3090.
# The only sub-sm_80 intrinsic used is __shfl_*_sync (sm_70+).
# sm_80 is native, not PTX: the A100 is a comparison target (RESULTS.md finding
# 50), and JIT-ing the whole module from PTX on first load costs seconds on a
# cold ~/.nv/ComputeCache -- far more than the ~170-220 ms of module load that
# finding 48's --reps 100 floor is calibrated against, and it recurs every
# process start if the cache is disabled or evicted.
NVCC_ARCH ?= -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_80,code=compute_80

# Extra -D's for pricing experiments, e.g. make DEFS=-DNORM_CANCEL_TOL=0.0f.
# Nothing in the default build sets this; it exists so a variant can be timed
# without editing a header and forgetting to put it back.
DEFS      ?=

# -MMD -MP: generate header dependencies. Without them the pattern rules below
# listed only bench.h and plattice.cuh, so editing bigint.cuh, prp.cuh or
# td.cuh left a stale bench_kernels.o and the binary silently did not contain
# the change under test.
NVCC_FLAGS = $(NVCC_ARCH) -O3 -std=c++17 -lineinfo -MMD -MP -Xcompiler "-O3 -Wall" $(DEFS)
CFLAGS     = -O3 -march=native -Wall -Wextra -std=c11 -MMD -MP $(DEFS)

CU_SRCS = bench_main.cu bench_kernels.cu
C_SRCS  = fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c

CU_OBJS = $(CU_SRCS:.cu=.o)
C_OBJS  = $(C_SRCS:.c=.o)
OBJS    = $(CU_OBJS) $(C_OBJS)

all: bench fbtest fbgen dumpcmp mkcofbatch

bench: $(OBJS)
	$(NVCC) $(NVCC_ARCH) -o $@ $^ -lcudart -lm

# CPU-only correctness gates. No CUDA, so these run on a busy box and on
# machines without the card.
fbtest: fbtest.c $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ fbtest.c $(C_OBJS) -lm

# Native algebraic factor-base generator; deliberately CPU-only and free of
# CADO/GMP dependencies. It shares the job/poly parser and prime enumerator.
fbgen: fbgen.c poly.o primes.o
	$(CC) $(CFLAGS) -o $@ fbgen.c poly.o primes.o -lm -pthread

dumpcmp: dumpcmp.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# joins the two per-side --emit-cof files into cofactorisation batch input
mkcofbatch: mkcofbatch.c
	$(CC) $(CFLAGS) -o $@ $< -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(OBJS:.o=.d)

clean:
	rm -f bench fbtest fbgen dumpcmp mkcofbatch *.o *.d

.PHONY: all clean check cofcheck fbgencheck

# `check` depended on fbtest alone, so the whole CUDA path -- kernels, the
# cofactoriser, the pipeline -- could fail to compile while the gates reported
# "all gates passed". Depend on `all`.
check: all cofcheck fbgencheck
	./fbtest --fb1 ../oracle/c183.fb1

fbgencheck: fbgen
	./fbgencheck.sh

# A GPU golden test over the cofactoriser: rho and ECM on the parity special-q,
# the even-cofactor invariant, the bounds refusals, and a candidate file with no
# trailing newline. Each case asserts an exact relation count, so a silent
# change of method or a truncated parse fails here rather than in a band run.
cofcheck: bench
	@./cofcheck.sh
