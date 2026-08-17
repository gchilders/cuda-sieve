# Standalone bucket-fill benchmark for the GPU NFS sieve feasibility probe.
# No CADO, no msieve, no GMP. Builds or loads factor bases directly.
# Pattern copied from ~/msieve-s/gerbicz_bench/Makefile, which worked well.

NVCC      ?= nvcc
CC        ?= gcc
CXX       ?= g++

# sm_120 native (RTX 5070, Blackwell consumer), sm_86 native (RTX 3090/A10).
# PTX fallback is compute_80, NOT compute_89: the driver only JITs PTX to a
# target >= the virtual arch, so a compute_89 fallback cannot load on any
# Ampere card at all -- the previous default was unrunnable on a 3090.
# sm_80 is native, not PTX: the A100 is a comparison target (RESULTS.md finding
# 50), and JIT-ing the whole module from PTX on first load costs seconds on a
# cold ~/.nv/ComputeCache -- far more than the ~170-220 ms of module load that
# finding 48's --reps 100 floor is calibrated against, and it recurs every
# process start if the cache is disabled or evicted.
#
# sm_80 is also the floor, and it is a tuning floor rather than an instruction
# one. The only warp intrinsics used are __syncwarp and __shfl{,_up}_sync, which
# long predate Ampere -- but k_fill_l2's __launch_bounds__(512, 3) targets 1536
# threads/SM and its 3 x 33 KB of shared memory, plus the opt-in shared-memory
# limit in bench_kernels.cu (grep 101376), are all Ampere-and-later numbers.
# sm_75 compiles, with ptxas warning that .minnctapersm is out of range on every
# fill kernel, and then oversubscribes Turing's 64 KB of shared memory at launch.
# Below that nothing is possible anyway: CUDA 13 dropped Volta, so nvcc 13.2
# rejects sm_70 outright.
GPU_ARCH_all := -gencode arch=compute_120,code=sm_120 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_80,code=compute_80

# `make GPU_ARCH=native` builds only for the card in this machine,
# `make GPU_ARCH=86` cross-builds for a named one (bare compute capability, no
# "sm_" and no dot), and the default builds the portable fat binary.
#
# Named GPU_ARCH, not ARCH: ARCH is exported by kernel-build and cross-compile
# shells and by some CI images, and make imports the environment, so an inherited
# ARCH=x86_64 would silently retarget the build to compute_x86_64.
#
# Measured on bench_kernels.cu, which is the entire build -- bench_main.cu is
# 2.5 s and the C files are noise:
#
#     all five targets, serial       309 s
#     all five targets, -t 0         214 s   <- the default
#     sm_120 alone                   215 s
#     sm_80 alone                     14 s
#
# ptxas on sm_120 is ~16x slower than on sm_80 and swamps everything else, so on
# a Blackwell host GPU_ARCH=native saves nothing once the passes run
# concurrently: the one target that cannot be dropped is the expensive one. Keep
# the fat binary here. GPU_ARCH=native is the real win on an Ampere or Ada host,
# where it drops sm_120 and takes the build to well under a minute.
GPU_ARCH ?= all

ifeq ($(GPU_ARCH),all)
NVCC_ARCH := $(GPU_ARCH_all)
else
ifeq ($(GPU_ARCH),native)
# Detected explicitly rather than via nvcc's -arch=native, which does not fail
# when it cannot see a GPU: it warns and silently builds for the toolkit default,
# yielding a binary that will not load on the card you meant to target.
#
# The digit check is not paranoia. nvidia-smi prints its errors to STDOUT, not
# stderr -- an old driver that does not know the compute_cap query field answers
# `Field "compute_cap" is not a valid field to query.` on fd 1, which sails past
# 2>/dev/null and past any test for emptiness, and would land inside the gencode
# flag. Evaluated here rather than at file scope so that `make clean`, `make
# fbgen` and the default fat build never spawn nvidia-smi at all; probing the
# driver is not free while a job is running (finding 53).
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | head -1 | tr -d .)
GPU_CC_VALID := $(shell echo '$(GPU_CC)' | grep -qxE '[0-9]+' && echo yes)
ifneq ($(GPU_CC_VALID),yes)
$(error GPU_ARCH=native: no usable compute capability from nvidia-smi (got "$(GPU_CC)"). Pass GPU_ARCH=<cc>, e.g. GPU_ARCH=86, or GPU_ARCH=all)
endif
NVCC_ARCH := -gencode arch=compute_$(GPU_CC),code=sm_$(GPU_CC)
else
GPU_ARCH_VALID := $(shell echo '$(GPU_ARCH)' | grep -qxE '[0-9]+' && echo yes)
ifneq ($(GPU_ARCH_VALID),yes)
$(error GPU_ARCH must be all, native, or a bare compute capability like 86 or 120 -- got "$(GPU_ARCH)". Not sm_86, not 8.6)
endif
NVCC_ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
endif
endif

# Objects carry no record of which -gencode built them, so switching GPU_ARCH
# leaves them "up to date" and the next link silently mixes or mistargets: build
# once with GPU_ARCH=native on the Ada box, then plain `make` for the portable
# binary, and you get sm_89-only objects in a binary that dies on the A100 with
# "no kernel image is available for execution on the device" -- on the machine
# you cannot debug on, while working fine on the build host. Stamp the arch and
# depend on it. The $(shell) rewrites the file only when the value changed, so
# its mtime moves only then and repeat builds stay incremental.
ARCH_STAMP := .arch.stamp
$(shell [ "$$(cat $(ARCH_STAMP) 2>/dev/null)" = '$(NVCC_ARCH)' ] || printf '%s' '$(NVCC_ARCH)' > $(ARCH_STAMP))

# Extra -D's for pricing experiments, e.g. make DEFS=-DNORM_CANCEL_TOL=0.0f.
# Nothing in the default build sets this; it exists so a variant can be timed
# without editing a header and forgetting to put it back.
DEFS      ?=

# Optional BOINC application build.  Keep BOINC dependencies out of the default
# executable; HAVE_BOINC=1 adds the compile-time API gate and the libraries.
# BOINC_CPPFLAGS should point at directories containing boinc_api.h and its
# companion headers (usually .../include/boinc, or BOINC's api and lib trees).
HAVE_BOINC     ?= 0
BOINC_CPPFLAGS ?=
BOINC_LIBS     ?= -lboinc_api -lboinc -lpthread
# Distribution binaries carry their C++ and GCC support runtimes. These flags
# apply only to the BOINC link; set BOINC_HOST_STATIC= to opt out explicitly.
BOINC_HOST_STATIC ?= -static-libstdc++ -static-libgcc
ifeq ($(HAVE_BOINC),1)
BOINC_DEFS := -DHAVE_BOINC
BENCH_LIBS := $(BOINC_LIBS)
BENCH_LINKFLAGS := $(foreach flag,$(BOINC_HOST_STATIC),-Xcompiler=$(flag))
# A BOINC application binary is distributed to CPUs other than the build host.
# Do not bake the build machine's ISA into the host-only C/C++ objects.
HOST_TUNE ?=
else
BOINC_DEFS :=
BENCH_LIBS :=
BENCH_LINKFLAGS :=
HOST_TUNE ?= -march=native
endif

# The CUDA runtime is linked statically in EVERY build, not only BOINC ones. A
# distributed binary cannot rely on a matching libcudart on the volunteer's
# host, and a local build that links it dynamically fails to start wherever the
# toolkit's lib directory is not on the loader path -- a plain `make && ./bench`
# on a stock CUDA install. It costs ~1 MB against a binary the fat gencode
# already dominates. Pass CUDART_LINK=shared to opt out.
CUDART_LINK ?= static

# -MMD -MP: generate header dependencies. Without them the pattern rules below
# listed only bench.h and plattice.cuh, so editing bigint.cuh, prp.cuh or
# td.cuh left a stale bench_kernels.o and the binary silently did not contain
# the change under test.
# -t 0: run the per-target ptxas passes concurrently, one thread per CPU. The
# gencode list above is otherwise walked serially, which is where the bulk of a
# fat build goes; see the timing table. Harmless when GPU_ARCH picks one target.
NVCC_FLAGS = $(NVCC_ARCH) -t 0 -O3 -std=c++17 -lineinfo -MMD -MP \
             -Xcompiler "-O3 -Wall" $(DEFS) $(BOINC_DEFS) $(BOINC_CPPFLAGS)
CFLAGS     = -O3 $(HOST_TUNE) -Wall -Wextra -std=c11 -MMD -MP \
             $(DEFS) $(BOINC_DEFS)
CXXFLAGS   = -O3 $(HOST_TUNE) -Wall -Wextra -std=c++17 -MMD -MP \
             $(DEFS) $(BOINC_DEFS) $(BOINC_CPPFLAGS)

CU_SRCS  = bench_main.cu bench_kernels.cu
C_SRCS   = fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c
CPP_SRCS = boinc_support.cpp

CU_OBJS  = $(CU_SRCS:.cu=.o)
C_OBJS   = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
# runlog.o is deliberately NOT in C_SRCS: fbtest links C_OBJS, and it is a
# CPU-only gate that must keep building on a machine with no driver library.
OBJS     = $(CU_OBJS) $(C_OBJS) $(CPP_OBJS) runlog.o fbgen_lib.o

# Switching HAVE_BOINC on an existing tree must rebuild boinc_support.o; make
# otherwise sees the same source timestamp and quietly reuses the no-op object.
# Include the BOINC include/link settings and host tuning in the signature so a
# changed deployment build cannot reuse objects produced for another one.
BOINC_STAMP := .boinc.stamp
BOINC_SIGNATURE := HAVE_BOINC=$(HAVE_BOINC)|BOINC_CPPFLAGS=$(BOINC_CPPFLAGS)|BOINC_LIBS=$(BOINC_LIBS)|BOINC_HOST_STATIC=$(BOINC_HOST_STATIC)|HOST_TUNE=$(HOST_TUNE)
$(shell [ "$$(cat $(BOINC_STAMP) 2>/dev/null)" = '$(BOINC_SIGNATURE)' ] || printf '%s' '$(BOINC_SIGNATURE)' > $(BOINC_STAMP))

# The commit the binary was built from, stamped into the run log's header so a
# log can be traced back to a source tree (STATUS.md item 12b). --dirty matters
# more than the hash: an uncommitted tree is exactly the case where the hash
# alone would be a lie. Stamped like the two above so that moving HEAD rebuilds
# it -- and ONLY runlog.o depends on it, which is why the string lives in that
# 200-line object and not in bench_main.cu: a commit must not cost a recompile
# of a CUDA translation unit. Unlike the nvidia-smi probe further up, `git
# describe` is local, cheap, and safe to run while a job is sieving.
#
# --dirty alone is not enough: git considers a tree with only UNTRACKED files
# clean, so a build from commit X plus a new uncommitted source stamps a bare
# X and the log's central claim becomes false in exactly the case the flag was
# added for. The porcelain check covers that; it lists untracked files but not
# ignored ones, so build products do not trip it.
#
# Kept as three plain assignments: a `case` inside $(shell ...) ends the make
# function at its first `*)`, and `wc -l` rather than the file list keeps a
# filename containing `$` or a paren out of a macro definition.
GIT_STAMP := .git.stamp
GIT_RAW  := $(shell git -C . describe --always --abbrev=8 2>/dev/null)
GIT_DIRT := $(shell git -C . status --porcelain 2>/dev/null | wc -l)
GIT_DESC := $(if $(GIT_RAW),$(GIT_RAW)$(if $(filter-out 0,$(GIT_DIRT)),-dirty),unknown)
$(shell [ "$$(cat $(GIT_STAMP) 2>/dev/null)" = '$(GIT_DESC)' ] || printf '%s' '$(GIT_DESC)' > $(GIT_STAMP))

runlog.o: CFLAGS += -DBENCH_GIT_DESC='"$(GIT_DESC)"'
runlog.o: $(GIT_STAMP)

# CUDART_LINK changes the link line and nothing else, so it gets its own stamp.
# Folding it into BOINC_SIGNATURE would make every object depend on it, and
# `make CUDART_LINK=shared` would recompile the whole fat CUDA build (214-309 s
# by the table above) to change one linker argument.
LINK_STAMP := .link.stamp
LINK_SIGNATURE := CUDART_LINK=$(CUDART_LINK)
$(shell [ "$$(cat $(LINK_STAMP) 2>/dev/null)" = '$(LINK_SIGNATURE)' ] || printf '%s' '$(LINK_SIGNATURE)' > $(LINK_STAMP))

all: bench fbtest fbgen sqgentest dumpcmp mkcofbatch

# -ldl for runlog.o, which dlopen()s libnvidia-ml at run time rather than
# linking it: the heartbeat's GPU columns must not become a build dependency,
# and a host with a driver but no toolkit still has the versioned soname. On
# glibc 2.34 and later dlopen is in libc and this resolves to an empty stub.
bench: $(OBJS) $(ARCH_STAMP) $(BOINC_STAMP) $(LINK_STAMP)
	$(NVCC) $(NVCC_ARCH) --cudart $(CUDART_LINK) -Xlinker --gc-sections \
		$(BENCH_LINKFLAGS) -o $@ $(OBJS) $(BENCH_LIBS) -lm -ldl

# Only the CUDA objects care about the arch; the C objects are host-only.
$(CU_OBJS): $(ARCH_STAMP)
# HAVE_BOINC changes boinc_support.cpp and may change host-code portability.
$(OBJS): $(BOINC_STAMP)

# CPU-only correctness gates. No CUDA, so these run on a busy box and on
# machines without the card.
fbtest: fbtest.c $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ fbtest.c $(C_OBJS) -lm

# Native algebraic factor-base generator; deliberately CPU-only and free of
# CADO/GMP dependencies. It shares the job/poly parser and prime enumerator.
fbgen: fbgen.c poly.o primes.o
	$(CC) $(CFLAGS) -o $@ fbgen.c poly.o primes.o -lm -pthread

# The same exact root finder drives on-the-fly special-q generation in bench.
# Keep it in one source: a copied degree-5 root finder would eventually drift
# from the factor-base generator whose output is gated against CADO. Function
# sections let the final links discard fbgen's CLI/thread-pool half, so sharing
# the implementation does not give bench or sqgentest a pthread dependency.
fbgen_lib.o: fbgen.c bench.h
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -DFBGEN_LIBRARY -c $< -o $@

sqgentest: sqgentest.c fbgen_lib.o poly.o primes.o
	$(CC) $(CFLAGS) -Wl,--gc-sections -o $@ $^ -lm

dumpcmp: dumpcmp.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# joins the two per-side --emit-cof files into cofactorisation batch input
mkcofbatch: mkcofbatch.c
	$(CC) $(CFLAGS) -o $@ $< -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJS:.o=.d)

clean:
	rm -f bench fbtest fbgen sqgentest dumpcmp mkcofbatch *.o *.d $(ARCH_STAMP) $(BOINC_STAMP) $(LINK_STAMP) $(GIT_STAMP)

.PHONY: all clean check cofcheck fbgencheck sqgencheck

# `check` depended on fbtest alone, so the whole CUDA path -- kernels, the
# cofactoriser, the pipeline -- could fail to compile while the gates reported
# "all gates passed". Depend on `all`.
check: all cofcheck fbgencheck sqgencheck
	./fbtest --fb1 ../oracle/c183.fb1

fbgencheck: fbgen
	./fbgencheck.sh

sqgencheck: sqgentest
	./sqgentest

# A GPU golden test over the cofactoriser: rho and ECM on the parity special-q,
# the even-cofactor invariant, the bounds refusals, and a candidate file with no
# trailing newline. Each case asserts an exact relation count, so a silent
# change of method or a truncated parse fails here rather than in a band run.
cofcheck: bench
	@./cofcheck.sh
