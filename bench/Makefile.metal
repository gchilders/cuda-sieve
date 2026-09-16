# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Metal port build. Separate from the CUDA Makefile on purpose: the CUDA build
# stays authoritative and untouched (see ../CLAUDE.md's ground rules).
#
# Xcode's command line tools ship no `metal` compiler, so DEVELOPER_DIR must
# point at a real Xcode with the Metal Toolchain component installed:
#     xcodebuild -downloadComponent MetalToolchain
#
# Run the Phase 2 gate with:  make -f Makefile.metal metalcheck

DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR

XCRUN     := xcrun -sdk macosx
METAL     := $(XCRUN) metal
METALLIB  := $(XCRUN) metallib
CXX       ?= clang++

# Slab size target, in bucket regions per slab. CUDA ships 32768; this box
# measures a bracketed interior minimum four times lower -- see
# METAL_PORT_PLAN.md section 8c. slab.h takes the default when unset, so the
# CUDA build and slabtest's pinned expectations are untouched.
SLAB_PERF_REGIONS ?= 8192u

# k_td stages small primes through a threadgroup tile of TD_TILE * 32 B. The
# CUDA default 512 is 16 KB -- half of Apple's whole 32,768 B budget. See
# METAL_PORT_PLAN.md 8q.
TD_TILE   ?= 512

BN_LIMBS  ?= 12
CF_LMAX   ?= 4

# -fno-fast-math is NOT optional. Metal defaults to fast math, which would
# relax the fp32 sequences the sieve's cell values depend on.
#
# -ffp-contract=off on the host side is equally load-bearing: the host must
# not fuse a*b+c into an fma the device build does not, or the two builds of a
# shared header stop agreeing and every bit-exactness gate here becomes a lie.
# TARGET FLOOR: Apple M1 (MTLGPUFamilyApple7) on macOS 13 Ventura.
#
# MSL 3.0 is not a preference, it is this toolchain's floor: Xcode 26.5's
# Metal compiler lists metal2.0-2.4 in its own -std help text but refuses
# every one of them ("invalid value 'metal2.3'"). MSL 3.0 in turn requires
# macOS 13 at runtime. Reaching macOS 11/12 would mean building the metallib
# with an older Xcode (14.x emits MSL 2.4) -- a toolchain problem, not a
# source problem, and nothing in this port uses a feature newer than MSL 2.x
# in the first place.
#
# Every Apple silicon Mac ever shipped runs macOS 13 or later, so this floor
# costs no HARDWARE coverage: M1 through M4 are all in range. It only excludes
# an M1 deliberately held back on Big Sur or Monterey.
#
# -fno-fast-math is NOT optional. Metal defaults to fast math, which would
# relax the fp32 sequences the sieve's cell values depend on.
#
# -Wno-c++17-extensions: MSL 3.0 is nominally C++14 and `if constexpr` warns
# as an extension. The construct works (clang has always accepted it), and
# Phase 4's byte-identical gate is the proof.
METAL_MIN_MACOS ?= 13.0

# The CPU-side objects are built by the DEFAULT Makefile, which we invoke for
# them rather than duplicating its rules. Its HOST_TUNE defaults to
# -march=native, which records the build machine's OS as the minimum and, on a
# machine where "native" resolves above apple-m1, its ISA too. Either one
# quietly breaks the back-to-M1 promise from inside objects this Makefile does
# not compile. Pin both. The default Makefile folds HOST_TUNE into a stamp that
# every object depends on, so switching between builds rebuilds them; we do not
# have to remember to.
CPUOBJ_TUNE := -mcpu=apple-m1 -mmacosx-version-min=$(METAL_MIN_MACOS)
# ---- optional BOINC application build ------------------------------------
#
# Mirrors the CUDA Makefile's switch, with the three macOS-specific
# corrections plan section 9a measured. Default 0, so the fraction-done path
# stays compiled out of every ordinary build and `boinccheck` remains the only
# thing in the tree that reaches it.
#
#   make -f Makefile.metal benchbin HAVE_BOINC=1 BOINC_DIR=$HOME/code/boinc-install
#
# BOINC_DIR is a `make install` prefix and fills in both flag sets. Point
# BOINC_CPPFLAGS/BOINC_LDFLAGS at the api/ and lib/ trees directly for a
# layout that is not a prefix.
#
# THE BOINC TREE MUST BE CONFIGURED FOR THIS TARGET, and neither mistake is a
# build failure -- see plan 9a, and `make -f Makefile.metal boinclinkcheck`,
# which asserts both against the binary this produces:
#   --disable-shared            or -lboinc_api takes an installed .dylib and
#                               bakes the BUILD MACHINE's absolute path into a
#                               binary meant for a volunteer's;
#   -mmacosx-version-min=13.0   in the BOINC CFLAGS/CXXFLAGS, or its objects
#                               are stamped with the build host's SDK and the
#                               METAL_MIN_MACOS floor below becomes a fiction.
# ---- embed the shader library in the executable ---------------------------
#
# ONE FILE. A BOINC project ships an executable; a second file that has to
# land beside it and match it is a class of failure (missing, stale,
# mismatched) that does not exist if the shaders are inside. ld puts the
# metallib in __DATA,__metallib and metal_rt reads it back with
# getsectiondata(), ahead of the bench.metallib-next-to-the-binary path but
# behind $CUDA_SIEVE_METALLIB, so every gate here still drives the library it
# just built.
#
# Costs ~1 MB on a 718 KB binary. EMBED_METALLIB=0 opts out; `metallibcheck`
# uses that as its control.
EMBED_METALLIB ?= 1
ifeq ($(EMBED_METALLIB),1)
METALLIB_EMBED = -sectcreate __DATA __metallib $(BUILD)/bench.metallib
else
METALLIB_EMBED =
endif

HAVE_BOINC     ?= 0
BOINC_DIR      ?=
ifneq ($(BOINC_DIR),)
BOINC_CPPFLAGS ?= -I $(BOINC_DIR)/include/boinc
BOINC_LDFLAGS  ?= -L $(BOINC_DIR)/lib
else
BOINC_CPPFLAGS ?=
BOINC_LDFLAGS  ?=
endif
BOINC_LIBS     ?= -lboinc_api -lboinc

# The CUDA Makefile defaults BOINC_HOST_STATIC to `-static-libstdc++
# -static-libgcc`, against glibc/libstdc++ skew across Linux distributions.
# Apple clang REJECTS `-static-libgcc` outright ("unsupported option") and
# ignores the other, and macOS ships libc++ and libSystem with the OS, so the
# flags are both impossible and unnecessary here. Forced empty in the
# delegation below rather than left to the default.
BOINC_HOST_STATIC :=

ifeq ($(HAVE_BOINC),1)
ifeq ($(strip $(BOINC_CPPFLAGS)),)
$(error HAVE_BOINC=1 needs BOINC_DIR=<prefix> (or BOINC_CPPFLAGS pointing at boinc_api.h))
endif
BOINC_DEFS := -DHAVE_BOINC
BOINC_LINK := $(BOINC_LDFLAGS) $(BOINC_LIBS)
else
BOINC_DEFS :=
BOINC_LINK :=
endif

# BN_LIMBS is used by BOTH sides -- bigint_msl.h declares bn_t with it and the
# host declares the same struct -- but only HOSTFLAGS was passing it, so the
# device silently kept bigint_msl.h's #ifndef default of 12 while the host took
# the Makefile's value. Identical today at 12; a desynced bn_t the moment
# anyone changed it, which is a wrong answer and not a compile error.
MSLFLAGS  := -std=metal3.0 -mmacos-version-min=$(METAL_MIN_MACOS) \
             -DSLAB_PERF_REGIONS=$(SLAB_PERF_REGIONS) \
             -DBN_LIMBS=$(BN_LIMBS) -DTD_TILE=$(TD_TILE) \
             -fno-fast-math -Wno-c++17-extensions -I metal

HOSTFLAGS_BASE := -std=c++17 -O2 -ffp-contract=off -I . -I metal \
             -mmacosx-version-min=$(METAL_MIN_MACOS) \
             -DBN_LIMBS=$(BN_LIMBS) -DCF_LMAX=$(CF_LMAX) -DTD_TILE=$(TD_TILE) \
             -DSLAB_PERF_REGIONS=$(SLAB_PERF_REGIONS)

# Split so `boinccheck` can keep its own boinc_api.h. Include search is
# left to right, so with BOINC_CPPFLAGS already in HOSTFLAGS a later
# `-I metal/boinc_stub` would LOSE to the real SDK and the stub gate would
# quietly stop testing the stub.
HOSTFLAGS := $(HOSTFLAGS_BASE) $(BOINC_DEFS) $(BOINC_CPPFLAGS)

# One setting for every delegation to the default Makefile. It folds these
# into a stamp that all its objects depend on, so passing them inconsistently
# between targets here would make the gates rebuild each other's CPU objects
# on every alternation. HOST_TUNE is passed on the command line and therefore
# beats that Makefile's own `HOST_TUNE ?=` (which empties itself under
# HAVE_BOINC so a distributed binary carries no -march=native): -mcpu=apple-m1
# is this port's declared floor, not the build host's ISA, so it is the right
# value for a distributed binary too.
CPUOBJ_MAKEVARS := HOST_TUNE='$(CPUOBJ_TUNE)' HAVE_BOINC=$(HAVE_BOINC) \
                   BOINC_CPPFLAGS='$(BOINC_CPPFLAGS)' BOINC_HOST_STATIC=

BUILD := .metal-build

# Unlike the CUDA Makefile, nothing here tracked a flag change, so flipping
# HAVE_BOINC left a $(BUILD) full of objects compiled the other way -- and a
# bench_main.o built without the define simply never calls bench_boinc_init(),
# which is a silently wrong binary rather than a link error. The plan records
# the same trap costing a TD measurement (8q). Stamp the whole signature, not
# just HAVE_BOINC: every Metal-side object and the metallib depends on it, so
# changing ANY tunable above rebuilds what it affects.
#
# AFTER `BUILD`, because METALLIB_EMBED names a path under it and this
# assignment is immediate: computed above, the signature would carry
# "/bench.metallib" and read as nonsense in a stamp file people will inspect.
METAL_STAMP := .metalflags.stamp
METAL_SIGNATURE := $(HOSTFLAGS)|$(MSLFLAGS)|$(BOINC_LINK)|$(CPUOBJ_TUNE)|$(METALLIB_EMBED)
$(shell [ "$$(cat $(METAL_STAMP) 2>/dev/null)" = '$(METAL_SIGNATURE)' ] || printf '%s' '$(METAL_SIGNATURE)' > $(METAL_STAMP))

$(BUILD):
	@mkdir -p $(BUILD)

# ---- Phase 2 gate: soft-fp64 and portable log2 --------------------------

$(BUILD)/sf_test.metallib: metal/sf_test.metal metal/softfp64.h metal/portable_log2.h \
                           metal/msl_compat.h | $(BUILD)
	$(METAL) $(MSLFLAGS) -c $< -o $(BUILD)/sf_test.air
	$(METALLIB) $(BUILD)/sf_test.air -o $@

$(BUILD)/sf_test_host: metal/sf_test_host.cpp metal/softfp64.h metal/portable_log2.h \
                       metal/msl_compat.h | $(BUILD)
	$(CXX) $(HOSTFLAGS) $< -o $@

$(BUILD)/sf_test_device: metal/sf_test_device.mm metal/softfp64.h metal/portable_log2.h \
                         metal/msl_compat.h | $(BUILD)
	$(CXX) $(HOSTFLAGS) -fobjc-arc $< -framework Metal -framework Foundation -o $@

$(BUILD)/sf_sites_test: metal/sf_sites_test.cpp metal/sf_sites.h metal/softfp64.h \
                        prp.cuh bigint.cuh | $(BUILD)
	$(CXX) $(HOSTFLAGS) $< -o $@

# sf_sites.h has the port's only address-space-qualified pointer; prove it
# still compiles as MSL, not just as host C++.
$(BUILD)/sites_msl.air: metal/sf_sites.h metal/softfp64.h | $(BUILD)
	@printf '#include "sf_sites.h"\nusing namespace metal;\nkernel void k(device ulong *o [[buffer(0)]], device const uint *l [[buffer(1)]], uint t [[thread_position_in_grid]]) { uint v[12]; for (int i=0;i<12;i++) v[i]=l[t*12+i]; o[t]=sf_bn_to_double(v,12); }\n' > $(BUILD)/sites_msl.metal
	$(METAL) $(MSLFLAGS) -c $(BUILD)/sites_msl.metal -o $@

.PHONY: metalcheck
metalcheck: rtcheck scancheck argbufcheck classifycheck $(BUILD)/sf_test_host $(BUILD)/sf_test.metallib $(BUILD)/sf_test_device \
            $(BUILD)/sf_sites_test $(BUILD)/sites_msl.air
	@echo "== softfp64 vs hardware fp64 (host) =="
	@$(BUILD)/sf_test_host
	@echo "== device build vs host build (same headers) =="
	@cd $(BUILD) && ./sf_test_device
	@echo "== the two real call sites vs prp.cuh's own fp64 =="
	@$(BUILD)/sf_sites_test
	@echo "PHASE 2 GATE: ALL GREEN"

.PHONY: clean
clean:
	rm -rf $(BUILD)

# ---- Phase 3 gate: the metal_rt runtime shim ----------------------------

$(BUILD)/rt_test.metallib: metal/rt_test.metal | $(BUILD)
	$(METAL) $(MSLFLAGS) -c $< -o $(BUILD)/rt_test.air
	$(METALLIB) $(BUILD)/rt_test.air -o $@

$(BUILD)/rt_test: metal/rt_test.cpp metal/metal_rt.mm metal/metal_rt.h | $(BUILD)
	$(CXX) $(HOSTFLAGS) metal/rt_test.cpp metal/metal_rt.mm \
	    -framework Metal -framework Foundation -framework IOKit -o $@

.PHONY: rtcheck
rtcheck: $(BUILD)/rt_test $(BUILD)/rt_test.metallib
	@$(BUILD)/rt_test $(BUILD)/rt_test.metallib

# ---- Phase 4 gate: fbgen_gpu on Metal vs the CPU generator ---------------
#
# Stronger than the HIP port managed for this phase, and for a reason worth
# recording: fbgen.c builds and runs natively on macOS, so the reference is
# the INDEPENDENT CPU implementation rather than another GPU build. That is
# what fbgpucheck.sh was written to compare, so the gate is the tree's own
# script, unmodified except for a sha256sum/shasum portability fix.

METAL_OBJS   := $(BUILD)/metal_rt.o $(BUILD)/metal_scan.o
FBGEN_CPUOBJ := fbgen_lib.o fb_load.o fb_cado.o poly.o primes.o platform.o

$(BUILD)/metal_rt.o: metal/metal_rt.mm metal/metal_rt.h $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) -c $< -o $@

$(BUILD)/metal_scan.o: metal/metal_scan.cpp metal/metal_scan.h $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) -c $< -o $@

$(BUILD)/fbgen_gpu: metal/fbgen_gpu_metal.cpp $(METAL_OBJS) $(BUILD)/bench.metallib $(METAL_STAMP)
	$(MAKE) $(CPUOBJ_MAKEVARS) $(FBGEN_CPUOBJ)
	$(CXX) $(HOSTFLAGS) metal/fbgen_gpu_metal.cpp $(METAL_OBJS) $(FBGEN_CPUOBJ) \
	    -framework Metal -framework Foundation -framework IOKit -lm -o $@

.PHONY: scancheck
scancheck: $(BUILD)/scan_test $(BUILD)/scan_test.metallib
	@$(BUILD)/scan_test $(BUILD)/scan_test.metallib

$(BUILD)/scan_test.metallib: metal/scan.metal | $(BUILD)
	$(METAL) $(MSLFLAGS) -c $< -o $(BUILD)/scan_only.air
	$(METALLIB) $(BUILD)/scan_only.air -o $@

$(BUILD)/scan_test: metal/scan_test.cpp metal/metal_scan.cpp metal/metal_rt.mm | $(BUILD)
	$(CXX) $(HOSTFLAGS) metal/scan_test.cpp metal/metal_scan.cpp metal/metal_rt.mm \
	    -framework Metal -framework Foundation -framework IOKit -o $@

# fbgpucheck.sh wants ./fbgen_gpu and ./fbgen next to it, as the CUDA build
# leaves them. Stage the Metal build under those names for the run only.
.PHONY: fbcheck
fbcheck: $(BUILD)/fbgen_gpu fbgen
	@cp $(BUILD)/fbgen_gpu ./fbgen_gpu
	@CUDA_SIEVE_METALLIB=$(CURDIR)/$(BUILD)/bench.metallib sh fbgpucheck.sh; \
	  rc=$$?; rm -f ./fbgen_gpu; exit $$rc

# ---- BOINC progress gate: the calibration pass must not pin the counter ---
# HAVE_BOINC defaults to 0, so the fraction-done path is compiled OUT of every
# ordinary build and no other test in the tree can reach it. This builds it
# with -DHAVE_BOINC against a stub client API and drives it directly. The
# CONTROL case runs first and must REPRODUCE the field bug -- a gate whose
# control cannot fail proves nothing.
BOINC_GATE_OBJ := $(BUILD)/boinc_progress_test

boinccheck: | $(BUILD)
	$(CXX) $(HOSTFLAGS_BASE) -DHAVE_BOINC -I metal/boinc_stub \
	    metal/boinc_progress_test.cpp metal/boinc_stub/boinc_stub.cpp \
	    boinc_support.cpp -o $(BOINC_GATE_OBJ)
	@echo "== control: WITHOUT the suspend, does the bug still exist? =="
	@$(BOINC_GATE_OBJ) control
	@echo
	@echo "== gate: WITH the suspend =="
	@$(BOINC_GATE_OBJ) suspended
	@echo
	@echo "BOINC PROGRESS GATE: PASS"

# ---- BOINC link gate: is the HAVE_BOINC binary distributable? ------------
# The companion to boinccheck above: that one drives the progress logic
# against a stub, this one inspects the real binary. Everything it checks
# BUILDS AND LINKS with exit 0 and is still wrong -- see plan 9a.
.PHONY: boinclinkcheck
boinclinkcheck:
ifneq ($(HAVE_BOINC),1)
	@echo "boinclinkcheck needs HAVE_BOINC=1, e.g."; \
	 echo "  make -f Makefile.metal boinclinkcheck HAVE_BOINC=1 BOINC_DIR=<prefix>"; \
	 exit 1
else
	@$(MAKE) -f Makefile.metal $(BUILD)/bench HAVE_BOINC=1 \
	    BOINC_DIR='$(BOINC_DIR)' BOINC_CPPFLAGS='$(BOINC_CPPFLAGS)' \
	    BOINC_LDFLAGS='$(BOINC_LDFLAGS)' >/dev/null
	@sh metal/boinclinkcheck.sh $(BUILD)/bench $(METAL_MIN_MACOS)
endif

# ---- metallib embedding gate: one file, and it runs on its own ----------
# Control FIRST and in its own build, because the positive case is only
# meaningful if a non-embedded binary demonstrably cannot run the same way.
# Both runs happen in a temporary directory with no bench.metallib and with
# $CUDA_SIEVE_METALLIB unset -- every other gate here exports it, so none of
# them would notice embedding being broken.
.PHONY: metallibcheck
metallibcheck: ../oracle/c183.fb1
	@echo "== control: the same binary built WITHOUT embedding =="
	@$(MAKE) -f Makefile.metal $(BUILD)/bench EMBED_METALLIB=0 >/dev/null
	@sh metal/metallibcheck.sh $(CURDIR)/$(BUILD)/bench \
	    $(CURDIR)/../oracle/c183.poly $(CURDIR)/../oracle/c183.fb1 control
	@echo
	@echo "== gate: WITH embedding =="
	@$(MAKE) -f Makefile.metal $(BUILD)/bench EMBED_METALLIB=1 >/dev/null
	@sh metal/metallibcheck.sh $(CURDIR)/$(BUILD)/bench \
	    $(CURDIR)/../oracle/c183.poly $(CURDIR)/../oracle/c183.fb1

# ---- Phase 5 gate: the sieve against the tree's own CPU ground truth -----

SIEVE_CPUOBJ := verify_cpu.o fbgen_lib.o fb_load.o fb_cado.o poly.o primes.o \
                platform.o rfb.o

MSL_HEADERS := metal/cuda_msl_compat.h metal/softfp64.h metal/portable_log2.h \
               metal/sf_sites.h metal/bigint_msl.h metal/prp_msl.h \
               metal/plattice_msl.h metal/slab_msl.h metal/td_msl.h

$(BUILD)/bench.metallib: metal/bench_kernels.metal metal/bench_kernels_body.metal.inc \
                         metal/td.metal metal/td_body.metal.inc \
                         metal/fbgen_gpu.metal metal/fbgen_gpu_body.metal.inc \
                         metal/cofac.metal metal/cofac_body.metal.inc \
                         metal/scan.metal $(MSL_HEADERS) $(METAL_STAMP) | $(BUILD)
	$(METAL) $(MSLFLAGS) -c metal/bench_kernels.metal -o $(BUILD)/bench_kernels.air
	$(METAL) $(MSLFLAGS) -c metal/td.metal            -o $(BUILD)/td.air
	$(METAL) $(MSLFLAGS) -c metal/fbgen_gpu.metal     -o $(BUILD)/fbgen_gpu.air
	$(METAL) $(MSLFLAGS) -c metal/cofac.metal         -o $(BUILD)/cofac.air
	$(METAL) $(MSLFLAGS) -c metal/scan.metal          -o $(BUILD)/scan.air
	$(METALLIB) $(BUILD)/bench_kernels.air $(BUILD)/td.air $(BUILD)/fbgen_gpu.air \
	            $(BUILD)/cofac.air $(BUILD)/scan.air -o $@

$(BUILD)/phase5_test: metal/phase5_test.cpp metal/fbgen_gpu_metal.cpp \
                      metal/metal_rt.mm metal/metal_scan.cpp $(BUILD)/bench.metallib $(METAL_STAMP)
	$(MAKE) $(CPUOBJ_MAKEVARS) $(SIEVE_CPUOBJ)
	$(CXX) $(HOSTFLAGS) -DFBGEN_GPU_LIBRARY metal/phase5_test.cpp \
	    metal/fbgen_gpu_metal.cpp metal/metal_rt.mm metal/metal_scan.cpp \
	    $(SIEVE_CPUOBJ) -framework Metal -framework Foundation -framework IOKit \
	    -lm -o $@

# --bound 300 puts a real number of cells over the survivor threshold; a bound
# that leaves none would pass while exercising nothing.
.PHONY: sievecheck
sievecheck: $(BUILD)/phase5_test
	@CUDA_SIEVE_METALLIB=$(CURDIR)/$(BUILD)/bench.metallib $(BUILD)/phase5_test \
	    --poly ../oracle/c183.poly --lim 1000000 --logI 13 --J 4096 \
	    --bound 300 --ncheck 512

# ---- Phase 6 groundwork: bindless struct-of-device-pointers -------------
$(BUILD)/argbuf.metallib: metal/argbuf.metal | $(BUILD)
	$(METAL) $(MSLFLAGS) -c $< -o $(BUILD)/argbuf.air
	$(METALLIB) $(BUILD)/argbuf.air -o $@

$(BUILD)/argbuf_test: metal/argbuf_test.cpp metal/metal_rt.mm | $(BUILD)
	$(CXX) $(HOSTFLAGS) metal/argbuf_test.cpp metal/metal_rt.mm \
	    -framework Metal -framework Foundation -framework IOKit -o $@

.PHONY: argbufcheck
argbufcheck: $(BUILD)/argbuf_test $(BUILD)/argbuf.metallib
	@cd $(BUILD) && ./argbuf_test argbuf.metallib

# ---- Phase 6a gate: the cofactoriser, via run_cofac ---------------------
#
# run_cofac() is the tree's own standalone cofactorisation entry point, and
# oracle/c183.q120000053.cofac_candidates.txt is the recorded CADO run's own
# candidate list for the parity special-q. 37 is the count cofcheck.sh pins
# and the count las itself finds at this q, so this reaches a real golden
# number long before ./bench --pipeline exists.

COFAC_CPUOBJ := verify_cpu.o fb_load.o fb_cado.o poly.o primes.o platform.o \
                rfb.o watchdog.o runlog.o

$(BUILD)/cofac_test: metal/cofac_test.cpp metal/cofac_metal.cpp metal/metal_rt.mm \
                     $(BUILD)/bench.metallib $(METAL_STAMP)
	$(MAKE) $(CPUOBJ_MAKEVARS) $(COFAC_CPUOBJ)
	$(CXX) $(HOSTFLAGS) metal/cofac_test.cpp metal/cofac_metal.cpp \
	    metal/metal_rt.mm $(COFAC_CPUOBJ) \
	    -framework Metal -framework Foundation -framework IOKit -lm -o $@

.PHONY: cofaccheck
cofaccheck: $(BUILD)/cofac_test
	@rc=0; \
	for L in 3 4; do \
	  for M in --rho --ecm; do \
	    printf '%-10s %-6s ' "limbs=$$L" "$$M"; \
	    CUDA_SIEVE_METALLIB=$(CURDIR)/$(BUILD)/bench.metallib $(BUILD)/cofac_test \
	      $$M --limbs $$L --in ../oracle/c183.q120000053.cofac_candidates.txt \
	      --expect 37 2>/dev/null | grep -E 'GATE' || rc=1; \
	  done; \
	done; exit $$rc

# ---- the full binary ----------------------------------------------------
#
# Mirrors the CUDA Makefile's `bench` target: the same host C objects, with
# bench_main_metal.cpp and bench_host.cpp in place of the two .cu files and
# the metal_rt shim in place of the CUDA runtime.

BENCH_CPUOBJ := fb_load.o verify_cpu.o poly.o primes.o rfb.o fb_cado.o \
                platform.o boinc_support.o runlog.o watchdog.o fbgen_lib.o

METAL_TU := $(BUILD)/bench_main.o $(BUILD)/bench_host.o $(BUILD)/metal_rt.o \
            $(BUILD)/metal_scan.o $(BUILD)/fbgen_gpu_lib.o

$(BUILD)/bench_main.o: metal/bench_main_metal.cpp $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) -DPIPE_K=16 -c $< -o $@

$(BUILD)/bench_host.o: metal/bench_host.cpp metal/pipeline_host.inc \
                       metal/cofac_host.inc metal/td_host.h $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) -DPIPE_K=16 -c $< -o $@

$(BUILD)/fbgen_gpu_lib.o: metal/fbgen_gpu_metal.cpp $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) -DFBGEN_GPU_LIBRARY -c $< -o $@

$(BUILD)/bench: $(METAL_TU) $(BUILD)/bench.metallib $(METAL_STAMP)
	$(MAKE) $(CPUOBJ_MAKEVARS) $(BENCH_CPUOBJ)
	$(CXX) $(HOSTFLAGS) $(METAL_TU) $(BENCH_CPUOBJ) $(BOINC_LINK) \
	    $(METALLIB_EMBED) \
	    -framework Metal -framework Foundation -framework IOKit \
	    -lm -ldl -lpthread -o $@

.PHONY: benchbin
benchbin: $(BUILD)/bench
	@echo "built $(BUILD)/bench"

# ---- cof_classify on the device, against prp.cuh's own fp64 --------------
#
# Phase 2 tested the softfp64 PRIMITIVES device-vs-host, and the assembled
# call sites host-vs-fp64 -- but not cof_classify as prp_msl.h assembles it,
# ON the device. This closes that gap: it is the only soft-fp64 consumer in
# the pipeline, so it is the first thing to rule out when a band's candidate
# count looks wrong.
$(BUILD)/classify_test.metallib: metal/classify_test.metal $(MSL_HEADERS) | $(BUILD)
	$(METAL) $(MSLFLAGS) -c $< -o $(BUILD)/classify_test.air
	$(METALLIB) $(BUILD)/classify_test.air -o $@

$(BUILD)/classify_test: metal/classify_test.cpp metal/metal_rt.mm $(METAL_STAMP) | $(BUILD)
	$(CXX) $(HOSTFLAGS) metal/classify_test.cpp metal/metal_rt.mm \
	    -framework Metal -framework Foundation -framework IOKit -o $@

.PHONY: classifycheck
classifycheck: $(BUILD)/classify_test $(BUILD)/classify_test.metallib
	@$(BUILD)/classify_test $(BUILD)/classify_test.metallib

# ---- Phase 6 gate: cofcheck.sh, the formal one -------------------------
#
# The tree's own golden test, unmodified except for two BSD/GNU portability
# fallbacks (sha256sum, head -c -1) that are recorded in the drift ledger.
# It drives ./bench --pipeline, so it needs the binary staged next to it and
# oracle/c183.fb1 present -- which our own fbgen_gpu generates in ~7 s.
../oracle/c183.fb1: $(BUILD)/fbgen_gpu
	CUDA_SIEVE_METALLIB=$(CURDIR)/$(BUILD)/bench.metallib $(BUILD)/fbgen_gpu \
	    --poly ../oracle/c183.poly --lim 134200000 --maxbits 15 \
	    --scale 1.925 --out $@

.PHONY: cofcheckgate
cofcheckgate: $(BUILD)/bench ../oracle/c183.fb1
	@cp $(BUILD)/bench ./bench
	@CUDA_SIEVE_METALLIB=$(CURDIR)/$(BUILD)/bench.metallib sh cofcheck.sh; \
	  rc=$$?; rm -f ./bench; exit $$rc
