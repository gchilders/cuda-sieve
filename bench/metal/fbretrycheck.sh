#!/bin/sh
# 9z-p: a killed root-finder segment must be REDONE, not lost.
#
# macOS's interactivity watchdog is contention-dependent: it fires when the GPU
# is wanted elsewhere, so no slice size -- measured or chosen -- can guarantee
# it never fires. A field M2 lost a whole workunit to two killed command
# buffers at 60% of a 250M factor base. Sizing and recovery are different jobs.
#
# The watchdog cannot be provoked on demand, so the recovery is driven by
# CUDA_SIEVE_METAL_FAULT_SYNC, which makes a chosen sync report a launch
# failure without corrupting anything. That is the point: the work really ran,
# so a correct retry must produce THE SAME factor base. A recovery path that
# has never executed is not a recovery path.
set -e
BIN="$1"; LIB="$2"; POLY="$3"
export CUDA_SIEVE_METALLIB="$LIB"
fail=0
ck() { if [ "$1" = 1 ]; then echo "PASS   $2"; else echo "FAIL   $2"; fail=1; fi; }
D=$(mktemp -d "${TMPDIR:-/tmp}/fbretry.XXXXXX")

run() {  # $1 = fault spec ("" for none) -> writes $D/$2.err, prints exit status
    if [ -n "$1" ]; then export CUDA_SIEVE_METAL_FAULT_SYNC="$1"; else unset CUDA_SIEVE_METAL_FAULT_SYNC; fi
    "$BIN" --pipeline --poly "$POLY" --qrange 120000053:120000053 \
        --allowance 101.6 --allowance0 68.1 --cofactor \
        > "$D/$2.out" 2> "$D/$2.err"
    echo $?
}

# ---- control: no fault, so no retry, and this is the reference output -----
rc=$(run "" clean)
BASE=$(grep -oE "[0-9]+ ideals through .*exact primes" "$D/clean.err" | head -1)
ck "$([ "$rc" = 0 ] && [ -n "$BASE" ] && echo 1 || echo 0)" \
   "control: a clean run succeeds and emits a factor base"
ck "$(grep -qc "retrying at" "$D/clean.err" 2>/dev/null && echo 0 || echo 1)" \
   "control: a clean run does NOT retry (so the gate below is not vacuous)"

# ---- gate: one synthetic kill inside the root finder ----------------------
rc=$(run 12 hurt)
ck "$([ "$rc" = 0 ] && echo 1 || echo 0)" \
   "a killed segment does not fail the run"
ck "$(grep -q "root finder in .* retrying at" "$D/hurt.err" && echo 1 || echo 0)" \
   "the kill is reported and the segment is retried at a smaller slice"
HURT=$(grep -oE "[0-9]+ ideals through .*exact primes" "$D/hurt.err" | head -1)
ck "$([ -n "$HURT" ] && [ "$HURT" = "$BASE" ] && echo 1 || echo 0)" \
   "and the recovered factor base is IDENTICAL to the clean one"
[ "$HURT" = "$BASE" ] || { echo "       clean: $BASE"; echo "       hurt:  $HURT"; }

# ---- and a kill that never stops must still terminate --------------------
set +e
rc=$(run -12 hopeless)
set -e
ck "$([ "$rc" != 0 ] && echo 1 || echo 0)" \
   "an unrecoverable fault gives up and exits nonzero, rather than looping"

rm -rf "$D"
[ $fail = 0 ] && echo "FBGEN RETRY GATE: PASS" || { echo "FBGEN RETRY GATE: FAIL"; exit 1; }
