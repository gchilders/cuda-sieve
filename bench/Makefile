# Standalone bucket-fill benchmark for the GPU NFS sieve feasibility probe.
# No CADO, no msieve, no GMP. Loads a GGNFS .afb.0 factor base directly.
# Pattern copied from ~/msieve-s/gerbicz_bench/Makefile, which worked well.

NVCC      ?= nvcc
CC        ?= gcc

# sm_120 native (RTX 5070, Blackwell consumer). PTX fallback for other cards.
NVCC_ARCH ?= -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_89,code=compute_89

NVCC_FLAGS = $(NVCC_ARCH) -O3 -std=c++17 -lineinfo -Xcompiler "-O3 -Wall"
CFLAGS     = -O3 -march=native -Wall -Wextra -std=c11

CU_SRCS = bench_main.cu bench_kernels.cu
C_SRCS  = fb_load.c verify_cpu.c poly.c rfb.c fb_cado.c

CU_OBJS = $(CU_SRCS:.cu=.o)
C_OBJS  = $(C_SRCS:.c=.o)
OBJS    = $(CU_OBJS) $(C_OBJS)

bench: $(OBJS)
	$(NVCC) $(NVCC_ARCH) -o $@ $^ -lcudart -lm

%.o: %.cu bench.h plattice.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c bench.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f bench *.o

.PHONY: clean
