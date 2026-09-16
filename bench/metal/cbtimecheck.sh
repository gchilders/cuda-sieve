#!/bin/sh
# 9z-k: assert that no COMMAND BUFFER runs longer than the port's own
# interactivity bound.
#
# WHY THIS EXISTS. macOS kills a command buffer that hogs the GPU against the
# UI (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), and it killed
# real volunteer workunits on M1 and M2. Two fixes were attempted before this
# gate existed and the first one, 9z-j, changed NOTHING measurable by the
# watchdog: it sliced the fbgen root finder into four DISPATCHES that the
# stream then batched straight back into one command buffer. It was measured,
# reported as a 790 -> 250 ms improvement, committed and signed -- because
# what was measured was dispatch time, and the watchdog does not judge
# dispatches.
#
# So this gate measures GPUEndTime - GPUStartTime of every command buffer, via
# CUDA_SIEVE_METAL_CBTIME, which is the only number the watchdog cares about.
#
# It deliberately runs WITHOUT --fb1, because factor-base generation on the GPU
# is where the field failures landed and `fbcheck` does not cover that path --
# fbcheck exercises the standalone fbgen_gpu tool, which has its own copy of
# the launch.
set -e
BENCH="$1"; POLY="$2"; BOUND="$3"; MODE="$4"
[ -n "$BOUND" ] || BOUND=750

LOG=$(mktemp -t cbtime)
CUDA_SIEVE_METAL_CBTIME=1 "$BENCH" --pipeline --poly "$POLY" \
    --qrange 120000053:120000053 --allowance 101.6 --allowance0 68.1 \
    --cofactor >"$LOG.out" 2>"$LOG" || { echo "  bench FAILED"; cat "$LOG.out"; exit 1; }

N=$(grep -c CBTIME "$LOG" || true)
if [ "$N" -lt 50 ]; then
    echo "  only $N command buffers seen -- CBTIME instrumentation is not reporting"
    exit 1
fi
MAX=$(grep CBTIME "$LOG" | awk '{print $2}' | sort -gr | head -1)
WORST=$(grep CBTIME "$LOG" | sort -k2 -gr | head -1 | sed 's/^CBTIME //')
echo "  $N command buffers, longest ${MAX} ms (bound ${BOUND} ms)"
echo "  worst: $WORST"

OVER=$(awk -v m="$MAX" -v b="$BOUND" 'BEGIN{print (m>b) ? 1 : 0}')
if [ "$MODE" = "control" ]; then
    [ "$OVER" = "1" ] && { echo "  CONTROL OK: over-bound buffer detected as it should be"; exit 0; }
    echo "  CONTROL FAILED: the unsliced build stayed under the bound, so this"
    echo "  gate would not notice slicing being removed"
    exit 1
fi
[ "$OVER" = "0" ] || { echo "  FAIL: a command buffer exceeded the bound"; exit 1; }
REL=$(grep -E "total relations" "$LOG.out" | tail -1 | awk '{print $NF}')
[ "$REL" = "37" ] || { echo "  FAIL: $REL relations, expected 37"; exit 1; }
echo "  COMMAND BUFFER BOUND GATE: PASS (37 relations)"
