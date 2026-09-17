# Phase 0 probes. NOT part of the build; run by hand, results in
# bench/METAL_PORT_PLAN.md section 5.3.
#
# There was no recipe here for a long time, and three of the compiled binaries
# were committed instead -- which could not even run from a fresh clone,
# because each one loads its .metallib at runtime by name and those are
# gitignored. Sources plus a recipe is the reproducible form.
#
#   export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
#   make -C metal-probe
#
# CommandLineTools ships no `metal` compiler; DEVELOPER_DIR must point at Xcode.
#
# -std=metal3.2 matches what section 5.3's results were measured with. The PORT
# targets metal3.0 (the macOS 13 floor); a probe may use a newer one, and
# probe.metal deliberately does, to test whether features like
# __HAVE_ATOMIC_ULONG__ appear at a higher level. Override with MSLSTD=.
MSLSTD  ?= metal3.2
PROBES  := probe lanemap fp32 spin

all: $(PROBES)

%.air: %.metal
	xcrun metal -std=$(MSLSTD) -fno-fast-math -c $< -o $@

%.metallib: %.air
	xcrun metallib $< -o $@

%: %.mm %.metallib
	clang++ -std=c++17 -O2 -fobjc-arc $< -framework Metal -framework Foundation -o $@

# Each binary loads ./<name>.metallib by name, so run from this directory.
clean:
	rm -f $(PROBES) *.air *.metallib

.PHONY: all clean
.PRECIOUS: %.metallib %.air
