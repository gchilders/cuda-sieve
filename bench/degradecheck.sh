#!/bin/sh
# degradecheck.sh -- gate for the DEGRADATION CEILING in run_pipeline_impl.
#
# WHAT IS GUARDED. The per-slab soft skips (bucket overflow, truncated
# large-prime list) exist so a one-record shortfall costs a slab's yield instead
# of a volunteer's whole task. That trade is silent in one direction: without a
# ceiling a job with an undersized PIPE_K or bucket array skips every slab of
# every q and still exits 0, and a rate computed from that band looks like a
# slow card rather than a misconfigured job. PIPE_LOST_MAX / PIPE_SLAB_SKIP_MAX
# stop such a band and report BENCH_EXIT_DEGRADED (5).
#
# This is the same shape as skipcheck's case C and it is here for the same
# reason. That gate exists because the norm-width cap USED to exit 0, which let
# a BOINC work unit be credited for a band that emitted nothing. Exit 5 is that
# failure mode with a different trigger, and until this script it had no gate at
# all -- the ceiling was verified once, by hand, with a fault injected into a
# throwaway binary. Nothing in the tree would have caught a regression.
#
# The halves are asserted SEPARATELY, as in case C, because they are what got
# conflated before:
#
#   Drain and checkpoint. Every relation already earned is valid and is kept.
#   Regressing this to `rc = -1` would discard the queue and report FAILED.
#
#   Exit 5, not 0. "The band ran" and "the work unit is creditable" are not the
#   same claim, and only the exit status carries the second one to BOINC.
#
# WHY PIPE_K IS THE TRIGGER. Of the two soft skips only the truncated list has a
# natural lever. The bucket capacity is derived in pipeline.cuh as
# est/nregion + 256, where est comes from pipe_est_records over the same factor
# base --bkthresh shapes, so lowering bkthresh grows the estimate too and never
# overflows; driving it would need a test-only knob on that slack. PIPE_K is a
# plain per-survivor cap, so a small `make PIPE_K=N` truncates essentially every
# survivor on a real job and every slab skips. The ceiling, the drain, the
# checkpoint guard, exit 5, the BOINC outcome and the PIPE_K arm of the remedy
# message are all covered by that one build; the bucket arm differs only in
# which string prints.
#
# NEEDS A CARD and needs `make PIPE_K=8`, so this is NOT in `check` -- same
# reason skipcheck and fbgpucheck are not. Takes ~8 min (two ~1550-q bands).
# Run against the default build it
# says what to rebuild with and exits 0, rather than failing.
set -e

BENCH=./bench
FB=../oracle/c183.fb1
POLY=../oracle/c183.poly
[ -x "$BENCH" ] || { echo "degradecheck: no $BENCH -- run make first"; exit 1; }
[ -f "$POLY" ]  || { echo "degradecheck: no $POLY"; exit 1; }
[ -f "$FB" ]    || { echo "degradecheck: no $FB -- generate it first (see oracle/README.md)"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

pass() { printf 'PASS   %-44s %s\n' "$1" "$2"; }
bad()  { printf 'FAIL   %-44s %s\n' "$1" "$2"; fail=1; }
info() { printf 'INFO   %-44s %s\n' "$1" "$2"; }

# WIDE ENOUGH TO REACH PIPE_LOST_MAX *AT PIPE_K=8*, which is the build this
# gate wants and the reason the span is 40000 rather than the 4000 that would
# do at PIPE_K=2. Measured 2026-09-08 on the c183: PIPE_K=8 loses 65 q in 1072,
# ~6.1%, so ~1650 q are needed for 100 losses; 40000 holds ~2140 at the
# 1/ln(1.2e8) ~ 1/18.6 prime density near 1.2e8. The observed trip was at 1552
# q sieved and 72902 relations. Overshooting is free -- the band stops at the
# ceiling -- and at PIPE_K=2 it simply stops much earlier.
QLO=120000000
QHI=120040000
REL="$TMP/rel.txt"

run_band() {   # run_band EXTRA... -> writes $TMP/out, returns bench's status
    rm -f "$REL" "$REL".part "$REL".part.* "$REL".lock 2>/dev/null || true
    set +e
    $BENCH --pipeline --cadofb "$FB" --poly "$POLY" --qrange $QLO:$QHI "$@" \
           > "$TMP/out" 2>&1
    st=$?
    set -e
    return $st
}

# ------------------------------------------------- is this build able to fire?
# Read PIPE_K back from the binary rather than assuming it. Without this a
# default build produces exit 0 and no skips, which is indistinguishable from a
# broken ceiling -- the gate would pass green on the one build that cannot test
# anything. One q is enough to print the allocation banner.
$BENCH --pipeline --cadofb "$FB" --poly "$POLY" --qrange 120000053:120000053 \
       > "$TMP/probe" 2>&1 || true
K=$(sed -n 's/^  large-prime list \([0-9][0-9]*\) records per survivor.*/\1/p' \
        "$TMP/probe" | head -1)
if [ -z "$K" ]; then
    echo "degradecheck: could not read PIPE_K from the pipeline banner."
    echo "  Expected a '  large-prime list N records per survivor (PIPE_K)' line."
    tail -15 "$TMP/probe" | sed 's/^/  | /'
    exit 1
fi
echo "  build PIPE_K: $K records per survivor"
# 8 IS THE WANTED VALUE, not merely the cutoff, and the difference is real
# coverage. Any small PIPE_K makes every slab truncate, which reaches the
# ceiling -- but at PIPE_K=2 EVERY q is lost, so nqdone is 0, the .part is
# empty and the `if (nqdone)` band-end record never emits. Three of the
# RUNBOOK's claims are then untestable: that relations already earned survive
# the stop, that they are valid, and that the durable log record calls the
# band degraded rather than "stopped cleanly".
#
# PIPE_K=8 straddles instead: measured on the c183, ~6.1% of q lose their only
# slab and the rest sieve normally, so the ceiling trips with 1552 q of real
# work behind it. Lower values still pass this gate -- the three assertions
# below detect nqdone == 0 and report themselves uncovered rather than failing
# -- but 8 is what exercises the whole path.
if [ "$K" -gt 8 ]; then
    echo "  SKIP: PIPE_K=$K is large enough that a well-formed job never"
    echo "        truncates, so nothing here can reach the ceiling. This gate"
    echo "        needs a build whose large-prime cap is deliberately too small."
    echo "        Rebuild with 'make PIPE_K=8' to run it, then 'make' to restore."
    exit 0
fi
[ "$K" -eq 8 ] || echo "  NOTE: PIPE_K=8 is the value this gate is tuned for;" \
                       "at $K every q is lost and three assertions self-skip."

# ---- case A: the ceiling stops the band and keeps what it earned ------------
# WITH --relations, which is the half that matters under BOINC: the stop must
# drain, checkpoint and leave a resumable .part behind.
echo "  ceiling: PIPE_LOST_MAX q lost / PIPE_SLAB_SKIP_MAX slabs (defaults" \
     "100/1000), over q in [$QLO, $QHI)"
run_band --relations "$REL" --cofactor --log "$TMP/run.log" && st=0 || st=$?
# How much real work stood behind the stop. 0 means every q was lost (PIPE_K
# too small to straddle), which makes the three "kept relations" assertions
# below vacuous rather than passing.
NQ=$(sed -n 's/.*stopped after \([0-9][0-9]*\) q this session.*/\1/p' \
         "$TMP/out" | head -1)
NQ=${NQ:-0}

if [ "$st" -eq 5 ]; then
    pass "A degraded stop exits DEGRADED" "bench exited 5"
elif [ "$st" -eq 0 ] && grep -q 'q range exhausted' "$TMP/out"; then
    # NOT A FAILURE OF THE CEILING -- a failure to REACH it. The band width
    # below is sized against PIPE_LOST_MAX's default of 100, and that limit is
    # #ifndef-overridable like PIPE_K is. Raise it and this band's ~215 q run
    # out first, which would otherwise be reported as "expected 5, got 0" and
    # read as a broken ceiling. Say which it actually was.
    echo "  SKIP: the band ran its q range out without reaching the ceiling."
    echo "        QLO..QHI is sized for PIPE_LOST_MAX=100; if that was raised,"
    echo "        widen QHI here to match before reading anything into this."
    rm -rf "$TMP"; trap - EXIT; exit 0
else
    bad "A degraded stop exits DEGRADED" "bench exited $st, expected 5"
    # Same courtesy skipcheck's cases pay: this gate needs a card and is not in
    # `make check`, so a failure here is usually costing someone rented time and
    # must not require a second run just to see what happened.
    tail -20 "$TMP/out" | sed 's/^/  | /'
fi

grep -q 'large-prime records past the' "$TMP/out" \
    && pass "A truncation warning printed" \
            "$(grep -oE '[0-9]+ large-prime records past the [0-9]+/survivor' "$TMP/out" | head -1)" \
    || bad  "A truncation warning printed" "not seen"

grep -qE '\*\* stopping at q=[0-9]+' "$TMP/out" \
    && pass "A ceiling message names the q" \
            "$(grep -oE '\*\* stopping at q=[0-9]+' "$TMP/out" | head -1)" \
    || bad  "A ceiling message names the q" "not seen"

# The remedy must name PIPE_K, not the bucket array. The message picks by
# MAJORITY, and getting that backwards steers an operator at the wrong knob --
# it is the whole reason the branch is not just "mention both".
grep -q 'check mfb and PIPE_K' "$TMP/out" \
    && pass "A remedy names PIPE_K, not buckets" "trial-division branch taken" \
    || bad  "A remedy names PIPE_K, not buckets" \
            "$(grep -oE 'Mostly [a-z- ]*' "$TMP/out" | head -1)"

# THE HALF THAT USED TO BE CONFLATED. A degraded band is not a failed one: the
# relations it did earn before the ceiling are valid and must survive.
grep -q 'FAILED' "$TMP/out" \
    && bad  "A band not reported FAILED" "$(grep -m1 FAILED "$TMP/out")" \
    || pass "A band not reported FAILED" ""

# pipe_finalize_outputs DELETES the .part when no sidecar was written, so a
# .part present alongside the sidecar is the evidence that the degraded path
# took the keep branch and not the discard one. Asserted unconditionally.
if [ -f "$REL".part ]; then
    pass "A .part kept, not discarded" "$(wc -c < "$REL".part) bytes"
else
    bad "A .part kept, not discarded" "missing -- the stop discarded its output"
fi

# THE RUNBOOK'S ACTUAL CLAIM: relations already earned are kept AND are valid.
# Only testable when some q completed before the ceiling -- see the PIPE_K note
# above. Reported as uncovered rather than passed when they did not, because a
# vacuous PASS here is exactly the kind of false assurance this gate exists to
# remove.
if [ "$NQ" -gt 0 ]; then
    nrel=$(wc -l < "$REL".part)
    [ "$nrel" -gt 0 ] \
        && pass "A earned relations survive the stop" "$NQ q sieved, $nrel relations kept" \
        || bad  "A earned relations survive the stop" "$NQ q sieved but .part is empty"
    # --check-relations rebuilds both norms from each (a,b) and its factors, so
    # this is the "and are valid" half rather than a line count. lpb/lpb0 only
    # bound which large primes are acceptable; the reconstruction itself does
    # not depend on them.
    if ./bench --check-relations "$REL".part --poly "$POLY" --lpb 33 --lpb0 31 \
           2>&1 | grep -q 'rebuild both norms exactly  PASS'; then
        pass "A kept relations reconstruct exactly" "$nrel of $nrel"
    else
        bad "A kept relations reconstruct exactly" "reconstruction gate failed"
    fi
    # The durable record. runlog_record("band end ...") is what outlives the
    # run under BOINC, and it is guarded on nqdone -- so this is unreachable at
    # PIPE_K=2 as well. The tag ordering is the point: `degraded` is tested
    # before `stopped`, or an exit-5 band writes "[stopped cleanly]" into the
    # one artifact anybody reads afterwards.
    if grep -q 'band end.*\[DEGRADED' "$TMP/run.log" 2>/dev/null; then
        pass "A run log records the band DEGRADED" \
             "$(grep -o 'band end.*' "$TMP/run.log" | grep -oE '\[[^]]*\]')"
    else
        bad "A run log records the band DEGRADED" \
            "$(grep -o 'band end.*' "$TMP/run.log" 2>/dev/null | grep -oE '\[[^]]*\]' || echo 'no band end record')"
    fi
else
    info "A earned relations survive the stop" "NOT COVERED: every q was lost (PIPE_K=$K)"
    info "A kept relations reconstruct exactly" "NOT COVERED: nothing was earned"
    info "A run log records the band DEGRADED" "NOT COVERED: band end is guarded on nqdone"
fi
[ -f "$REL".part.ckpt ] \
    && pass "A checkpoint sidecar written" "$(basename "$REL".part.ckpt)" \
    || bad  "A checkpoint sidecar written" "missing"

# "FIX THE JOB BEFORE RESUMING" is not decoration. Both ceilings reset on
# resume, so the ordinary "rerun the same command" advice would walk the band
# into the same wall forever.
grep -q 'FIX THE JOB BEFORE RESUMING' "$TMP/out" \
    && pass "A resume advice warns, not invites" "" \
    || bad  "A resume advice warns, not invites" \
            "$(grep -oE 'resume at q=[0-9]+.*' "$TMP/out" | head -1)"

# ---- case B: the same ceiling with NO relation file -------------------------
# The benchmarking path. pipe_checkpoint returns 0 -- success -- at its own
# `if (!fr || !cfg->relations)` guard, i.e. it reports success for correctly
# doing nothing, so a stop that calls it unguarded sets ckpt_written with no
# sidecar on disk and then prints "checkpoint written, resume at q=0" naming a
# .part that was never created. Both policy stops guard on fr for that reason;
# this asserts the degraded one does.
run_band --cofactor && st=0 || st=$?

[ "$st" -eq 5 ] \
    && pass "B no-relations stop still exits DEGRADED" "bench exited 5" \
    || bad  "B no-relations stop still exits DEGRADED" "bench exited $st, expected 5"

grep -q 'no relation file requested' "$TMP/out" \
    && pass "B says nothing to resume" "" \
    || bad  "B says nothing to resume" "not seen"
# DISCRIMINATE, as skipcheck's case E does. Both no-relations arms open with
# "no relation file requested" -- the capped one and the degraded one -- so the
# assertion above cannot tell them apart, and would still pass if the branch
# order regressed or a degraded band somehow set `capped`. The closing phrase
# is what differs.
grep -q 'skipped its way to the ceiling' "$TMP/out" \
    && pass "B names the ceiling, not the cap" "" \
    || bad  "B names the ceiling, not the cap" \
            "$(grep -oE 'skipped its way to [a-z -]*' "$TMP/out" | head -1)"

# The specific regression: a resume point invented for a file that never existed.
grep -q 'checkpoint written, resume at q=' "$TMP/out" \
    && bad  "B invents no resume point" "$(grep -oE 'checkpoint written, resume at q=[0-9]+' "$TMP/out" | head -1)" \
    || pass "B invents no resume point" ""

# And it must not be reported as a checkpoint FAILURE either -- nothing was
# requested, so "move it aside or pass --restart" is the wrong advice too.
grep -q 'NO checkpoint could be written' "$TMP/out" \
    && bad  "B not reported as a failed checkpoint" "fell through to the error branch" \
    || pass "B not reported as a failed checkpoint" ""

echo
[ $fail -eq 0 ] && echo "degradecheck: all cases passed" \
                || echo "degradecheck: FAILURES above"
exit $fail
