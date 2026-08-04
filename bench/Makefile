# Standalone bucket-fill benchmark for the GPU NFS sieve feasibility probe.
# No CADO, no msieve, no GMP. Loads a GGNFS .afb.0 factor base directly.
# Pattern copied from ~/msieve-s/gerbicz_bench/Makefile, which worked well.

NVCC      ?= nvcc
CC        ?= gcc

# sm_120 native (RTX 5070, Blackwell consumer). PTX fallback for other cards.
NVCC_ARCH ?= -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_89,code=compute_89

# Extra -D's for pricing experiments, e.g. make DEFS=-DNORM_CANCEL_TOL=0.0f.
# Nothing in the default build sets this; it exists so a variant can be timed
# without editing a header and forgetting to put it back.
DEFS      ?=

NVCC_FLAGS = $(NVCC_ARCH) -O3 -std=c++17 -lineinfo -Xcompiler "-O3 -Wall" $(DEFS)
CFLAGS     = -O3 -march=native -Wall -Wextra -std=c11 $(DEFS)

CU_SRCS = bench_main.cu bench_kernels.cu
C_SRCS  = fb_load.c verify_cpu.c poly.c rfb.c fb_cado.c

CU_OBJS = $(CU_SRCS:.cu=.o)
C_OBJS  = $(C_SRCS:.c=.o)
OBJS    = $(CU_OBJS) $(C_OBJS)

all: bench fbtest dumpcmp

bench: $(OBJS)
	$(NVCC) $(NVCC_ARCH) -o $@ $^ -lcudart -lm

# CPU-only correctness gates. No CUDA, so these run on a busy box and on
# machines without the card.
fbtest: fbtest.c $(C_OBJS) bench.h plattice.cuh
	$(CC) $(CFLAGS) -o $@ fbtest.c $(C_OBJS) -lm

dumpcmp: dumpcmp.c
	$(CC) $(CFLAGS) -o $@ $< -lm

%.o: %.cu bench.h plattice.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c bench.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f bench fbtest dumpcmp *.o

.PHONY: all clean check

check: fbtest
	./fbtest --cadofb ../oracle/c183.fb1
