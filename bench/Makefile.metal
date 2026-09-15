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
MSLFLAGS  := -std=metal3.2 -fno-fast-math -I metal
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
metalcheck: $(BUILD)/sf_test_host $(BUILD)/sf_test.metallib $(BUILD)/sf_test_device \
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
