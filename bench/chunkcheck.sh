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

# THE PRIMARY ASSERTION IS SILENCE, not a comparison of values.
#
# An earlier version compared each reported chunk against the valve's last
# choice, AND ITS CONTROL PASSED: cof_report_chunk renders min(chunk, n), the
# relapse happened to land on the band's final partial flush where n was 1890,
# and the gate read 1890 as "below the ceiling, fine". On another card the same
# control failed only because the relapse landed on a full flush. A gate whose
# discrimination depends on where the band ends is not a gate.
#
# So: after the valve has spoken, a correct build says NOTHING MORE. That is an
# invariant, not a coincidence -- the steering recomputes max(half, floor_ch),
# floor_ch is clamped to the ceiling, chunk_cur already equals the ceiling, so
# it lands on the same value and cof_report_chunk suppresses an unchanged line.
# Any later `cofactor chunk:` line means the steering MOVED the chunk, which
# after a valve descent is the relapse itself, whatever value got rendered.
#
# The two parks are valve lines, not chunk lines, so they do not trip this; and
# they leave chunk_cur where the steering will hold it, so they produce no
# chunk line afterwards either.
AFTER=$(awk '/cofactor: kernel launch|parking at|parking here/ { seen = 1; next }
             /cofactor chunk:/ { if (seen) n++ }
             END { print n + 0 }' "$D/ctl")
ck "$([ "$AFTER" = 0 ] && echo 1 || echo 0)" \
   "the controller is SILENT once the valve has spoken ($AFTER later report(s))"
[ "$AFTER" = 0 ] || sed -n '/cofactor: kernel launch/,$p' "$D/ctl" | \
    grep "cofactor chunk:" | sed 's/^/       relapsed: /'

# Kept as diagnostics, not as the verdict: they are informative when they do
# fire, and worthless when the rendered value is clamped to a small final n.
awk '
  function note(n) {
      if (ceil == "") return
      if (n + 0 > ceil + 0) over++
      if (n + 0 == open + 0) back++
  }
  /parking here/ { next }
  /parking at/ {
      n = $0; sub(/.*parking at /, "", n); sub(/ records.*/, "", n)
      ceil = n; next
  }
  /cofactor chunk:/ {
      n = $0; sub(/.*cofactor chunk: /, "", n); sub(/ records.*/, "", n)
      if (open == "") { open = n; next }
      note(n)
  }
  /cofactor: kernel launch/ {
      n = $0; sub(/.*-> /, "", n); sub(/ records.*/, "", n)
      note(n); ceil = n
  }
  END { printf "%d %d\n", over + 0, back + 0 }' "$D/ctl" > "$D/verdict"
read -r OVER BACK < "$D/verdict"
echo "  (diagnostics: $OVER report(s) over the valve's choice, $BACK back at the floor)"

[ $fail = 0 ] && echo "AUTO CHUNK GATE: PASS" || { echo "AUTO CHUNK GATE: FAIL"; exit 1; }
