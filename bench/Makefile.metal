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

BN_LIMBS  ?= 12
CF_LMAX   ?= 4

# -fno-fast-math is NOT optional. Metal defaults to fast math, which would
# relax the fp32 sequences the sieve's cell values depend on.
#
# -ffp-contract=off on the host side is equally load-bearing: the host must
# not fuse a*b+c into an fma the device build does not, or the two builds of a
# shared header stop agreeing and every bit-exactness gate here becomes a lie.
# -std=metal3.2 rather than metal4.0 for portability: metal4.0 compiles this
# tree warning-free but needs macOS 26, while metal3.2 reaches macOS 15. The
# only cost is a pedantic `if constexpr is a C++17 extension` warning, since
# MSL 3.2 is nominally C++14; the construct itself works (clang treats it as
# the same extension it always has), and Phase 4's byte-identical gate is the
# proof that it does.
MSLFLAGS  := -std=metal3.2 -fno-fast-math -Wno-c++17-extensions -I metal
HOSTFLAGS := -std=c++17 -O2 -ffp-contract=off -I . -I metal \
             -DBN_LIMBS=$(BN_LIMBS) -DCF_LMAX=$(CF_LMAX)

BUILD := .metal-build

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
metalcheck: rtcheck scancheck $(BUILD)/sf_test_host $(BUILD)/sf_test.metallib $(BUILD)/sf_test_device \
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

$(BUILD)/bench.metallib: metal/fbgen_gpu.metal metal/fbgen_gpu_body.metal.inc \
                         metal/scan.metal | $(BUILD)
	$(METAL) $(MSLFLAGS) -c metal/fbgen_gpu.metal -o $(BUILD)/fbgen_gpu.air
	$(METAL) $(MSLFLAGS) -c metal/scan.metal      -o $(BUILD)/scan.air
	$(METALLIB) $(BUILD)/fbgen_gpu.air $(BUILD)/scan.air -o $@

$(BUILD)/metal_rt.o: metal/metal_rt.mm metal/metal_rt.h | $(BUILD)
	$(CXX) $(HOSTFLAGS) -c $< -o $@

$(BUILD)/metal_scan.o: metal/metal_scan.cpp metal/metal_scan.h | $(BUILD)
	$(CXX) $(HOSTFLAGS) -c $< -o $@

$(BUILD)/fbgen_gpu: metal/fbgen_gpu_metal.cpp $(METAL_OBJS) $(BUILD)/bench.metallib
	$(MAKE) $(FBGEN_CPUOBJ)
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
