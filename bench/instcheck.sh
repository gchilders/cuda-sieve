#!/bin/sh
# Gate for --instances: N pipelines on one GPU, each over its share of the band
# with its own NAME.inst<k> .part and checkpoint, merged into NAME at the end.
#
# The claim under test is that NAME is the file ONE pipeline writes -- byte for
# byte -- however the band was run: split, stopped, killed, resumed with a
# different --instances, or restarted. Every case compares against a single
# pipeline's output over the same band, and checks that nothing of the
# instances' machinery (manifest, instance files, checkpoints) is left behind
# once the band commits, and that it is all there when it has not.
#
# Needs a card and the c183 oracle inputs; about five minutes on an RTX 4090.
set -e

BENCH=./bench
FB=../oracle/c183.fb1
POLY=../oracle/c183.poly
[ -x "$BENCH" ] || { echo "instcheck: no $BENCH -- run make first"; exit 1; }
[ -f "$POLY" ]  || { echo "instcheck: no $POLY"; exit 1; }
[ -f "$FB" ]    || { echo "instcheck: no $FB -- generate it first (see oracle/README.md)"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

pass() { printf 'PASS   %-46s %s\n' "$1" "$2"; }
bad()  { printf 'FAIL   %-46s %s\n' "$1" "$2"; fail=1; }

Q=120000000:120006000
R="$TMP/rel.txt"

clean() { rm -f "$R" "$R".* 2>/dev/null || true; }
run() {   # run OUT ARGS... ; returns bench's status
    out=$1; shift
    set +e
    $BENCH --pipeline --cofactor --cadofb "$FB" --poly "$POLY" "$@" \
           > "$out" 2>&1
    st=$?
    set -e
    return $st
}
leftovers() { ls "$R".* 2>/dev/null | tr '\n' ' '; }
same_as_ref() {   # same_as_ref NAME OUT STATUS
    if [ "$3" -ne 0 ]; then bad "$1" "exit $3 (see below)"; tail -5 "$2"; return; fi
    if ! cmp -s "$TMP/ref.txt" "$R"; then
        bad "$1" "relations differ from one pipeline's"; return
    fi
    l=$(leftovers)
    if [ -n "$l" ]; then bad "$1" "left behind: $l"; return; fi
    pass "$1" "$(wc -l < "$R" | tr -d ' ') relations, identical"
}
# Start a two-instance band in the background and return once BOTH instances
# have flushed relations into their .part -- i.e. genuinely mid-band.
start_mid_band() {   # start_mid_band OUT ARGS...
    out=$1; shift
    $BENCH --pipeline --cofactor --cadofb "$FB" --poly "$POLY" "$@" \
           > "$out" 2>&1 &
    pid=$!
    i=0
    while [ $i -lt 600 ]; do
        if [ -s "$R.inst0.part" ] && [ -s "$R.inst1.part" ]; then return 0; fi
        kill -0 $pid 2>/dev/null || return 1
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

echo
echo "[instances] q $Q, c183"

clean
if run "$TMP/o.ref" --qrange $Q --instances 1 --relations "$R"; then
    cp "$R" "$TMP/ref.txt"
    pass "one pipeline (reference)" "$(wc -l < "$R" | tr -d ' ') relations"
else
    bad "one pipeline (reference)" "exit $?"; tail -5 "$TMP/o.ref"; exit 1
fi

clean
st=0; run "$TMP/o.2" --qrange $Q --instances 2 --relations "$R" || st=$?
same_as_ref "two instances" "$TMP/o.2" $st
grep -q "merged into" "$TMP/o.2" || bad "two instances: merge reported" "no merge line"

clean
st=0; run "$TMP/o.a" --qrange $Q --instances auto --relations "$R" || st=$?
same_as_ref "--instances auto ($(grep -o 'instances auto: [0-9]' "$TMP/o.a" | tail -1))" "$TMP/o.a" $st

# SIGTERM: every instance drains and checkpoints its own .part; the manifest
# stays. The relaunch asks for ONE pipeline and must still resume the band as
# the two instances the manifest records.
clean
if start_mid_band "$TMP/o.t1" --qrange $Q --instances 2 --relations "$R"; then
    kill -TERM $pid; st=0; wait $pid || st=$?
    if [ $st -eq 0 ] && [ -f "$R.instances" ] && [ -f "$R.inst0.part.ckpt" ] &&
       [ -f "$R.inst1.part.ckpt" ] && [ ! -f "$R" ]; then
        pass "SIGTERM mid-band: drained, state kept" "$(leftovers)"
    else
        bad "SIGTERM mid-band: drained, state kept" "exit $st; files: $(leftovers)"
    fi
    st=0; run "$TMP/o.t2" --qrange $Q --relations "$R" || st=$?
    grep -q "resuming the 2-instance band" "$TMP/o.t2" ||
        bad "resume: manifest decides N" "no 'resuming the 2-instance band'"
    same_as_ref "resume after SIGTERM, --instances 1" "$TMP/o.t2" $st
else
    bad "SIGTERM mid-band" "band never reached both .part files"; kill $pid 2>/dev/null || true
fi

# kill -9: no drain at all. Each instance resumes from its last checkpoint,
# truncating its .part to the checkpointed prefix.
clean
if start_mid_band "$TMP/o.k1" --qrange $Q --instances 2 --relations "$R"; then
    kill -KILL $pid; wait $pid 2>/dev/null || true
    # NAME.lock is left behind; the relaunch clears it itself (dead pid).
    st=0; run "$TMP/o.k2" --qrange $Q --instances 2 --relations "$R" || st=$?
    same_as_ref "resume after kill -9" "$TMP/o.k2" $st
else
    bad "kill -9 mid-band" "band never reached both .part files"; kill $pid 2>/dev/null || true
fi

# A band under way belongs to its own --qrange and job: anything else is
# refused rather than appended to, and --restart is the way out.
clean
if start_mid_band "$TMP/o.m1" --qrange $Q --instances 2 --relations "$R"; then
    kill -TERM $pid; wait $pid || true
    if run "$TMP/o.m2" --qrange 120000000:120007000 --instances 2 --relations "$R"; then
        bad "a different band is refused" "accepted"
    elif grep -q "different band or job" "$TMP/o.m2"; then
        pass "a different band is refused" ""
    else
        bad "a different band is refused" "refused, but not for the band"; tail -3 "$TMP/o.m2"
    fi
    st=0; run "$TMP/o.m3" --qrange $Q --instances 2 --restart --relations "$R" || st=$?
    same_as_ref "--restart discards it and reruns" "$TMP/o.m3" $st
else
    bad "a different band is refused" "band never reached both .part files"; kill $pid 2>/dev/null || true
fi

# An instance file with no manifest is some other band's: refused.
clean
: > "$R.inst1.part"
if run "$TMP/o.o" --qrange $Q --instances 2 --relations "$R"; then
    bad "an orphan instance file is refused" "accepted"
else
    pass "an orphan instance file is refused" ""
fi

# Open bands cannot be split ahead of time: one pipeline, said so.
clean
st=0; run "$TMP/o.open" --qrange 120000000: --target-rels 20 --instances 2 \
          --relations "$R" || st=$?
if [ $st -eq 0 ] && grep -q "running one pipeline" "$TMP/o.open" &&
   [ ! -f "$R.instances" ]; then
    pass "--target-rels runs one pipeline" ""
else
    bad "--target-rels runs one pipeline" "exit $st"; tail -3 "$TMP/o.open"
fi

# A band with fewer q values than instances: split into as many as it has.
# Before the cap, instance 0 got the empty range [q, q-1] and the band failed.
Q1=120000053:120000053
clean
st=0; run "$TMP/o.r1" --qrange $Q1 --instances 1 --relations "$R" || st=$?
if [ $st -eq 0 ] && [ -s "$R" ]; then
    cp "$R" "$TMP/ref1.txt"
    clean
    st=0; run "$TMP/o.s1" --qrange $Q1 --instances 2 --relations "$R" || st=$?
    if [ $st -eq 0 ] && cmp -s "$TMP/ref1.txt" "$R" &&
       grep -q "has only 1 value" "$TMP/o.s1" && [ -z "$(leftovers)" ]; then
        pass "--instances 2 on a one-q band" "$(wc -l < "$R" | tr -d ' ') relations, identical"
    else
        bad "--instances 2 on a one-q band" "exit $st"; tail -3 "$TMP/o.s1"
    fi
else
    bad "one-q reference" "exit $st"; tail -3 "$TMP/o.r1"
fi

# The default is auto, and a band too short to pay a second setup runs one.
clean
st=0; run "$TMP/o.d1" --qrange $Q1 --relations "$R" || st=$?
if [ $st -eq 0 ] && cmp -s "$TMP/ref1.txt" "$R" && [ ! -f "$R.instances" ] &&
   ! grep -q "=== --instances" "$TMP/o.d1"; then
    pass "default on a short band: one pipeline" "$(grep -o 'instances auto: 1[^)]*' "$TMP/o.d1" | head -1)"
else
    bad "default on a short band: one pipeline" "exit $st"; tail -3 "$TMP/o.d1"
fi

echo
if [ $fail -eq 0 ]; then echo "instcheck: all cases pass"; else echo "instcheck: FAILURES"; fi
exit $fail
