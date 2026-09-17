#!/bin/sh
# 9z-m/9z-n: the progress bar must not restart at 0 when a band resumes.
#
# PORTABLE TO ANY BUILD OF THIS TREE. It takes one binary and drives it through
# the run log, so the same script gates the CUDA build -- where the bug was
# reported -- as well as the Metal one. It lives under metal/ only because that
# is where this port's gates are kept.
#
# Everything else in run_pipeline_impl knows about resume -- the --target-rels
# stop test adds base_rel, the checkpoint writer and the console counts add
# base_nq -- but pipe_progress_fraction was handed neither, and bench_main has
# ALREADY shrunk its denominators (--nq is reduced by the completed count, qmin
# is moved to the checkpoint's next_q). So progress was measured against what
# REMAINED.
#
# THE INSTRUMENT is the run log, because one record carries both numbers:
# `nq=` is base_nq + nqdone and has always been right, `pct=` comes from the
# estimator. Before the fix a single line read `nq=141 ... pct=18.06` -- 141 of
# 200 special-q done, bar at 18%.
#
# NO CONTROL BINARY, DELIBERATELY. The obvious control -- a build with the fix
# compiled out -- cannot exist here: this tree marks any non-empty DEFS as a
# PRICING build and refuses --relations from it, which a resume gate needs.
# That guard is right and predates this gate. Instead the check DISCRIMINATES:
# it computes the pre-fix answer from the same record and asserts we do not
# produce it, so reverting the fix fails both assertions rather than neither.
# The real pre-fix binaries were measured once, on both ports; see plan 9z-n.
set -e
BIN="$1"; LIB="$2"; POLY="$3"; FB="$4"
NQ=200
# Portable across BSD and GNU mktemp: `-t NAME` means different things.
D=$(mktemp -d "${TMPDIR:-/tmp}/progresscheck.XXXXXX")
[ -n "$LIB" ] && export CUDA_SIEVE_METALLIB="$LIB"
fail=0
ck() { if [ "$1" = 1 ]; then echo "PASS   $2"; else echo "FAIL   $2"; fail=1; fi; }

ARGS="--pipeline --cadofb $FB --poly $POLY --qrange 120000053:120288000 --nq $NQ
      --allowance 101.6 --allowance0 68.1 --cofactor --relations $D/r.dat
      --stop-file $D/STOP --log $D/run.log --log-every 5"

# ---- run 1: sieve, then stop cleanly so a checkpoint exists ---------------
"$BIN" $ARGS >"$D/out1.log" 2>&1 &
p=$!
i=0
while [ $i -lt 90 ]; do
    grep -qE "nq=(6[4-9]|[7-9][0-9]|1[0-9][0-9])" "$D/run.log" 2>/dev/null && break
    sleep 2; i=$((i+1))
done
touch "$D/STOP"
wait $p || true
grep -q "stopped after" "$D/out1.log" \
  || { echo "FAIL   run 1 did not stop cleanly -- no checkpoint to resume"; exit 1; }
BASE=$(grep -oE "stopped after [0-9]+ q" "$D/out1.log" | grep -oE "[0-9]+")
echo "       run 1 checkpointed after $BASE of $NQ q"
rm -f "$D/STOP"
keep=$(grep -c . "$D/run.log")

# ---- run 2: resume, and read the FIRST new progress record ---------------
"$BIN" $ARGS >"$D/out2.log" 2>&1 &
p=$!
# Wait for a PROGRESS RECORD past the kept prefix, not merely for the log to
# grow: --log appends a header naming the commit and argv on every run, so a
# line-count test is satisfied instantly and would kill the resumed band
# before it reported anything.
j=0
while [ $j -lt 60 ]; do
    tail -n +"$((keep+1))" "$D/run.log" 2>/dev/null \
      | grep -qE "nq=[0-9]+ .*pct=" && break
    sleep 2; j=$((j+1))
done
kill $p 2>/dev/null || true; wait $p 2>/dev/null || true
REC=$(tail -n +"$((keep+1))" "$D/run.log" | grep -oE "nq=[0-9]+ .*pct=[0-9.]+" \
      | tail -1 | sed 's/.*nq=\([0-9]*\) .*pct=\([0-9.]*\)/\1 \2/')
set -- $REC
NQD="$1"; PCT="$2"
[ -n "$NQD" ] || { echo "FAIL   no progress record after the resume"; exit 1; }

WANT=$(awk -v n="$NQD" -v t="$NQ" 'BEGIN{printf "%.2f", 100.0*n/t}')
OLD=$(awk -v n="$NQD" -v b="$BASE" -v t="$NQ" 'BEGIN{printf "%.2f", 100.0*(n-b)/(t-b)}')
echo "       resumed at nq=$NQD: pct=$PCT   (whole band $WANT, pre-fix formula $OLD)"

ck "$(awk -v a="$WANT" -v b="$OLD" 'BEGIN{d=a-b; if(d<0)d=-d; print (d > 5.0) ? 1 : 0}')" \
   "the sample discriminates: the two formulas differ by more than 5 points here"
ck "$(awk -v a="$PCT" -v b="$WANT" 'BEGIN{d=a-b; if(d<0)d=-d; print (d <= 1.0) ? 1 : 0}')" \
   "resumed bar reports the WHOLE band's position, matching its own nq="
ck "$(awk -v a="$PCT" -v b="$OLD" 'BEGIN{d=a-b; if(d<0)d=-d; print (d > 5.0) ? 1 : 0}')" \
   "and is NOT the pre-fix answer (progress through the remainder)"
ck "$(awk -v a="$PCT" -v b="$BASE" -v t="$NQ" 'BEGIN{print (a >= 100.0*b/t) ? 1 : 0}')" \
   "and never goes backwards across the restart"

rm -rf "$D"
[ $fail = 0 ] && echo "PROGRESS RESUME GATE: PASS" || { echo "PROGRESS RESUME GATE: FAIL"; exit 1; }
