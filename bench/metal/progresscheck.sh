#!/bin/sh
# 9z-m: the progress bar must not restart at 0 when a band resumes.
#
# Everything else in run_pipeline_impl knows about resume -- the --target-rels
# stop test adds base_rel, the checkpoint writer and the console counts add
# base_nq -- but pipe_progress_fraction was handed neither, and bench_main has
# ALREADY shrunk its denominators (--nq is reduced by the completed count,
# qmin is moved to the checkpoint's next_q). So progress was measured against
# what REMAINS.
#
# The instrument is the run log, because one record carries both numbers:
# `nq=` is base_nq + nqdone (always correct) and `pct=` comes from the
# estimator. Before the fix a single line read `nq=141 ... pct=18.06` -- 141
# of 200 special-q done, bar at 18%. The check is that those two agree.
#
# One expensive run produces the checkpoint; both the gate and its control
# then resume from a COPY of it, so the control costs a short resume rather
# than a whole band.
set -e
GATE="$1"; CTRL="$2"; LIB="$3"; POLY="$4"; FB="$5"
D=$(mktemp -d -t progresscheck)
NQ=200
export CUDA_SIEVE_METALLIB="$LIB"
fail=0
ck() { if [ "$1" = 1 ]; then echo "PASS   $2"; else echo "FAIL   $2"; fail=1; fi; }

ARGS="--pipeline --cadofb $FB --poly $POLY --qrange 120000053:120288000 --nq $NQ
      --allowance 101.6 --allowance0 68.1 --cofactor --relations $D/r.dat
      --stop-file $D/STOP --log $D/run.log --log-every 5"

# ---- run 1: sieve, then stop cleanly so a checkpoint exists ---------------
"$GATE" $ARGS >"$D/out1.log" 2>&1 &
p=$!
i=0
while [ $i -lt 60 ]; do
    grep -q "nq=6[4-9]\|nq=[7-9][0-9]\|nq=1[0-9][0-9]" "$D/run.log" 2>/dev/null && break
    sleep 2; i=$((i+1))
done
touch "$D/STOP"
wait $p || true
grep -q "stopped after" "$D/out1.log" \
  || { echo "FAIL   run 1 did not stop cleanly -- no checkpoint to resume"; exit 1; }
BASE=$(grep -oE "stopped after [0-9]+ q" "$D/out1.log" | grep -oE "[0-9]+")
echo "       run 1 checkpointed after $BASE of $NQ q"
rm -f "$D/STOP"
cp "$D/run.log" "$D/run1.log.keep"

# resume_and_report <binary> <label>; prints "nq pct" of the first new record
resume_and_report() {
    rm -rf "$D/w"; mkdir "$D/w"
    cp "$D"/r.dat.part "$D"/r.dat.part.ckpt "$D/w/" 2>/dev/null || true
    cp "$D/run1.log.keep" "$D/w/run.log"
    A=$(echo "$ARGS" | sed "s#$D/r.dat#$D/w/r.dat#g; s#$D/run.log#$D/w/run.log#g; s#$D/STOP#$D/w/STOP#g")
    "$1" $A >"$D/w/out.log" 2>&1 &
    q=$!
    # Wait for a PROGRESS RECORD past the kept prefix, not merely for the log
    # to grow: --log appends a header naming the commit and argv on every run,
    # so a line-count test is satisfied instantly and kills the resumed band
    # before it reports anything.
    keep=$(grep -c . "$D/run1.log.keep")
    j=0
    while [ $j -lt 45 ]; do
        tail -n +"$((keep+1))" "$D/w/run.log" 2>/dev/null \
          | grep -qE "nq=[0-9]+ .*pct=" && break
        sleep 2; j=$((j+1))
    done
    kill $q 2>/dev/null || true; wait $q 2>/dev/null || true
    tail -n +"$((keep+1))" "$D/w/run.log" \
      | grep -oE "nq=[0-9]+ .*pct=[0-9.]+" | tail -1 \
      | sed 's/.*nq=\([0-9]*\) .*pct=\([0-9.]*\)/\1 \2/'
}

# ---- control: the estimator built to ignore resume ------------------------
set -- $(resume_and_report "$CTRL" control)
CNQ="$1"; CPCT="$2"
if [ -z "$CNQ" ]; then echo "FAIL   control produced no post-resume record"; exit 1; fi
CWANT=$(awk -v n="$CNQ" -v t="$NQ" 'BEGIN{printf "%.2f", 100.0*n/t}')
echo "       control: nq=$CNQ pct=$CPCT (a correct bar would read $CWANT)"
ck "$(awk -v a="$CPCT" -v b="$CWANT" 'BEGIN{print (b-a > 5.0) ? 1 : 0}')" \
   "control: the unfixed estimator restarts the bar low, as it did in the field"

# ---- gate ----------------------------------------------------------------
set -- $(resume_and_report "$GATE" gate)
GNQ="$1"; GPCT="$2"
if [ -z "$GNQ" ]; then echo "FAIL   gate produced no post-resume record"; exit 1; fi
GWANT=$(awk -v n="$GNQ" -v t="$NQ" 'BEGIN{printf "%.2f", 100.0*n/t}')
echo "       gate:    nq=$GNQ pct=$GPCT (expected $GWANT)"
ck "$(awk -v a="$GPCT" -v b="$GWANT" 'BEGIN{d=a-b; if(d<0)d=-d; print (d <= 1.0) ? 1 : 0}')" \
   "resumed bar reports the WHOLE band's position, matching its own nq="
ck "$(awk -v a="$GPCT" -v b="$BASE" -v t="$NQ" 'BEGIN{print (a >= 100.0*b/t) ? 1 : 0}')" \
   "and it never goes backwards across the restart"

rm -rf "$D"
[ $fail = 0 ] && echo "PROGRESS RESUME GATE: PASS" || { echo "PROGRESS RESUME GATE: FAIL"; exit 1; }
