#!/bin/sh
# The AUTO cofactor chunk controller, which nothing else tests.
#
# cofcheck.sh exercises only PINNED --cof-chunk values, and both the launch
# valve and the steering below it are gated on `!chunk`, so the whole adaptive
# path -- the one every production band runs -- had no gate at all. That is how
# the defect this gate pins shipped: the valve descended below cof_chunk_floor()
# and the steering, whose test against `stage` is always true, clamped straight
# back UP to that floor on the next flush. Measured on a TITAN RTX before the
# fix, at the shipped 1000 ms bound:
#
#   cofactor chunk: 110592 records/launch          <- cof_chunk_floor()
#   cofactor: kernel launch 1172 ms ...; 110592 -> 75466 records/launch
#   cofactor chunk: 110592 records/launch          <- snapped back
#   cofactor: kernel launch 1168 ms ...; 110592 -> 75735 records/launch
#   cofactor chunk: 110592 records/launch          <- and again
#
# Every second flush ran at the size the valve had just rejected, which on the
# 980 Ti that motivated the valve is a TDR kill.
#
# RUN IT AGAINST THE SHIPPING BINARY, at the shipped bound. An earlier draft
# lowered COF_LAUNCH_TARGET_MS to 200 ms with DEFS so the valve would fire on
# any card -- and its control PASSED, because at a bound the device cannot
# reach the valve fires on EVERY flush, `valve_acted` suppresses the steering,
# and the code under test never executes. That is the same blind spot the
# original V100 verification had. The bug needs the valve to fire and then GO
# QUIET, which only happens at a bound the device can descend under.
#
# So: no DEFS, --ecm-curves 48 to make the launch long. If no launch reaches
# the bound the gate SAYS SO and fails rather than passing on an assertion that
# never ran; raise --ecm-curves (CHUNKCHECK_CURVES) on faster hardware.
#
# THE CONTROL IS A SINGLE MACRO and must FAIL:
#     make DEFS=-DCOF_CHUNK_NO_CEILING bench && make chunkcheck
set -e
BENCH="$1"; POLY="$2"; NQ="${3:-200}"; CURVES="${4:-48}"
[ -x "$BENCH" ] || { echo "usage: chunkcheck.sh ./bench poly [nq] [curves]"; exit 2; }

fail=0
ck() { if [ "$1" = 1 ]; then echo "PASS   $2"; else echo "FAIL   $2"; fail=1; fi; }
D=$(mktemp -d "${TMPDIR:-/tmp}/chunkcheck.XXXXXX")
trap 'rm -rf "$D"' EXIT

set +e
"$BENCH" --pipeline --poly "$POLY" --qrange 120000053:120288000 --nq "$NQ" \
    --allowance 101.6 --allowance0 68.1 --cofactor --ecm-curves "$CURVES" \
    > "$D/out" 2> "$D/err"
rc=$?
set -e
ck "$([ "$rc" = 0 ] && echo 1 || echo 0)" "the band completes (exit $rc)"

# The controller's whole conversation, in order: the valve's descents and the
# steering's reports, which is the only place the two are visible together.
grep -E "cofactor chunk:|cofactor: kernel launch" "$D/err" > "$D/ctl" || true
echo "  --- controller ---"
sed 's/^/  /' "$D/ctl"

OPEN=$(sed -n 's/.*cofactor chunk: \([0-9]*\) records.*/\1/p' "$D/ctl" | head -1)
NVALVE=$(grep -c "cofactor: kernel launch" "$D/ctl" || true)
ck "$([ -n "$OPEN" ] && echo 1 || echo 0)" \
   "an opening chunk is reported (cof_chunk_floor = ${OPEN:-?})"
ck "$([ "$NVALVE" -ge 1 ] && echo 1 || echo 0)" \
   "the launch valve fired at least once (so the rest is not vacuous)"
[ "$NVALVE" -ge 1 ] || echo "       no launch reached the bound on this device --\
 raise CHUNKCHECK_CURVES; do NOT lower the bound with DEFS, that makes the\
 valve fire every flush and the gate stops testing anything"

# Two properties, both violated by the pre-fix build and neither by the fix.
#   over/back: a report above the valve's last choice, or back at the floor.
#   up:        ANY increase at all once the valve has spoken. The controller is
#              one-way downward from that point, so this catches a snapback
#              even if the floor itself were to change.
#
# A PARK IS THE ONE LEGITIMATE INCREASE, and it is distinguishable because it
# says so. There are two, and only the first moves the chunk:
#   "parking at N"  -- the descent bought under 10%, so the launch is not
#                      chunk-bound and the throughput goes back. An explicit,
#                      logged decision, as against the silent relapse this gate
#                      exists to catch, so it resets the ceiling.
#   "parking here"  -- the floor is reached and the bound is still unmet. The
#                      reduction is KEPT, because over the bound is not over the
#                      watchdog and over it by 2.4x is, so nothing moves.
awk '
  function note(n) {
      if (ceil == "") return
      if (n + 0 > ceil + 0) over++
      if (n + 0 == open + 0) back++
      if (last != "" && n + 0 > last + 0) up++
      last = n
  }
  /parking here/ { parked++; next }   # the floor case: nothing moved
  /parking at/ {
      n = $0; sub(/.*parking at /, "", n); sub(/ records.*/, "", n)
      ceil = n; last = n; parked++; next
  }
  /cofactor chunk:/ {
      n = $0; sub(/.*cofactor chunk: /, "", n); sub(/ records.*/, "", n)
      if (open == "") { open = n; next }
      note(n)
  }
  /cofactor: kernel launch/ {
      n = $0; sub(/.*-> /, "", n); sub(/ records.*/, "", n)
      note(n); ceil = n; last = n
  }
  END { printf "%d %d %d %d\n", over + 0, back + 0, up + 0, parked + 0 }' \
    "$D/ctl" > "$D/verdict"
read -r OVER BACK UP PARKED < "$D/verdict"
[ "$PARKED" = 0 ] || echo "  (the valve parked $PARKED time(s): a descent stopped\
 paying, so it gave the throughput back -- a logged increase, not a relapse)"

ck "$([ "$OVER" = 0 ] && echo 1 || echo 0)" \
   "the chunk never climbs back over the valve's choice ($OVER violation(s))"
ck "$([ "$BACK" = 0 ] && echo 1 || echo 0)" \
   "and never returns to the floor the valve descended from ($BACK time(s))"
ck "$([ "$UP" = 0 ] && echo 1 || echo 0)" \
   "and never rises at all once the valve has spoken ($UP increase(s))"

[ $fail = 0 ] && echo "AUTO CHUNK GATE: PASS" || { echo "AUTO CHUNK GATE: FAIL"; exit 1; }
