#!/bin/sh
# Golden test for the cofactor path, on the parity special-q q = 120000053.
#
# Every case pins an exact relation count rather than a range. The bugs this
# guards against -- a method switch that never takes effect, a factor base
# missing p = 2, a parser dropping its last record, a bound that truncates a
# factor -- all present as a *plausible* smaller number, never as a crash, so
# only an exact expectation catches them.
#
# las finds 37 relations at this q; 7 of them need no cofactorisation.
set -e

FB=../oracle/c183.fb1
POLY=../oracle/c183.poly
# Two words on purpose: the parser takes `--qrange VALUE`, not `--qrange=VALUE`,
# and the latter hits the unknown-option branch. Used unquoted so it splits.
Q="--qrange 120000053:120000053"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

run() { ./bench --pipeline --cadofb $FB --poly $POLY --qrange 120000053:120000053 "$@" 2>&1; }

expect_rel() {   # name expected <bench args...>
    name=$1; want=$2; shift 2
    got=$(run "$@" --relations $TMP/r.txt | grep 'total relations' | tail -1 | awk '{print $NF}')
    if [ "$got" = "$want" ]; then
        printf 'PASS   %-34s %s relations\n' "$name" "$got"
    else
        printf 'FAIL   %-34s expected %s, got %s\n' "$name" "$want" "$got"
        fail=1
    fi
}

expect_refused() {  # name <bench args...>
    name=$1; shift
    if ./bench --pipeline --cadofb $FB --poly $POLY --qrange 120000053:120000053 \
               "$@" --relations $TMP/r.txt >$TMP/o 2>&1; then
        printf 'FAIL   %-34s accepted, should have been refused\n' "$name"; fail=1
    else
        printf 'PASS   %-34s refused\n' "$name"
    fi
}

echo
echo "[cofactor] golden test, q = 120000053"

expect_rel "trial division only"            7
expect_rel "rho, inline queue"             37 --cofactor --cof-rounds 2 --cof-budget 65536
expect_rel "ECM, inline queue"             37 --cofactor --cof-ecm --ecm-b1 2000 --ecm-curves 48
# The method switch must actually REACH the queue. B1 = 2 with one curve can
# split essentially nothing, so this must fall back to the trial-division count;
# 37 here would mean rho ran despite --cof-ecm, which is exactly the defect the
# review found (the queue's ECM fields were left at their memset zero).
expect_rel "ECM honoured, not silently rho"  7 --cofactor --cof-ecm --ecm-b1 2 --ecm-curves 1
expect_refused "zero ECM curves"               --cofactor --cof-ecm --ecm-curves 0

# ---- job-file parameters -------------------------------------------------
# ../oracle/input.job is the SAME polynomial as c183.poly, plus the sieve keys.
# So the two must agree once the one thing that differs -- the lambdas, which
# c183.poly does not carry -- is stated explicitly on both.
#
# The counts are compared to EACH OTHER rather than pinned, deliberately. What
# is being tested is that reading the job file changes nothing it should not;
# pinning a number here would also pin the derived allowance, which is a
# separate thing with its own case above.
JOB=../oracle/input.job
got=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 \
      | grep -c 'job file: rlim 67100000, alim 134200000, lpbr 31, lpba 32, mfbr 60, mfba 92')
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s all six keys\n' "job file parsed"
else
    printf 'FAIL   %-34s banner line not found\n' "job file parsed"; fail=1
fi

# GGNFS units: alambda 3.5 * log2(alim 134200000) = 94.5 bits. If this is ever
# read as CADO units (3.5 * lpba 32 = 112) the bound LOOSENS by 17 bits and the
# error is invisible in the relation count -- it just costs time. Pin the
# conversion itself.
got=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 \
      | grep -c 'alambda 3.5 x log2(alim 134200000) = 94.5[0-9]* bits')
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s 3.5 -> 94.5 bits, not 112\n' "GGNFS lambda conversion"
else
    printf 'FAIL   %-34s wrong or missing conversion\n' "GGNFS lambda conversion"; fail=1
fi

# Explicit flags outrank the job file. With both allowances stated, the job
# file and the bare poly must produce the identical relation set.
a=$(run --cofactor --cof-rounds 2 --cof-budget 65536 --allowance 112 --allowance0 68.1 --relations $TMP/p1.txt | grep 'total relations' | tail -1 | awk '{print $NF}')
b=$(./bench --pipeline --cadofb $FB --poly $JOB --qrange 120000053:120000053 \
        --cofactor --cof-rounds 2 --cof-budget 65536 --allowance 112 --allowance0 68.1 \
        --relations $TMP/p2.txt 2>&1 | grep 'total relations' | tail -1 | awk '{print $NF}')
if [ -n "$a" ] && [ "$a" = "$b" ] && cmp -s $TMP/p1.txt $TMP/p2.txt; then
    printf 'PASS   %-34s %s each, byte-identical\n' "explicit flags outrank job file" "$a"
else
    printf 'FAIL   %-34s poly %s vs job %s\n' "explicit flags outrank job file" "$a" "$b"; fail=1
fi

expect_refused "lpb above the 32-bit limb"    --cofactor --lpb 33
expect_refused "mfb above the 3-limb cofactor" --cofactor --mfb 97
expect_refused "zero rho budget"              --cofactor --cof-budget 0
expect_refused "rho rounds overflowing shift" --cofactor --cof-rounds 40
expect_refused "lambda in the typo window"    --cofactor --lambda1 0.01
expect_refused "lambda absurdly large"        --cofactor --lambda1 99
# 0 is the documented "use CADO's automatic" sentinel and must stay legal.
expect_rel     "lambda 0 == automatic"     37 --cofactor --cof-rounds 2 --cof-budget 65536 --lambda1 0

# The GGNFS default factor base has no p = 2, so algebraic cofactors stay even
# and mz_n0inv's odd-modulus contract is broken. Relation-producing runs must
# refuse rather than exit 0 having lost two thirds of the yield.
if ./bench --pipeline --poly $POLY --qrange 120000053:120000053 --cofactor \
           --relations $TMP/r.txt >$TMP/o 2>&1; then
    printf 'FAIL   %-34s accepted an even-cofactor factor base\n' "even cofactors refused"; fail=1
else
    printf 'PASS   %-34s refused\n' "even cofactors refused"
fi

# Standalone splitter: the candidate file must parse identically with and
# without a trailing newline, and must agree with the inline queue.
run --candidates $TMP/c.txt --relations $TMP/td.txt >/dev/null
head -c -1 $TMP/c.txt > $TMP/c_nonl.txt
a=$(./bench --cofac $TMP/c.txt      --poly $POLY --relations $TMP/s1.txt --cof-rounds 2 --cof-budget 65536 2>&1 | grep 'RELATIONS' | awk '{print $2}')
b=$(./bench --cofac $TMP/c_nonl.txt --poly $POLY --relations $TMP/s2.txt --cof-rounds 2 --cof-budget 65536 2>&1 | grep 'RELATIONS' | awk '{print $2}')
if [ "$a" = "$b" ] && [ "$a" = "30" ]; then
    printf 'PASS   %-34s %s each, terminated and not\n' "unterminated candidate file" "$a"
else
    printf 'FAIL   %-34s expected 30/30, got %s/%s\n' "unterminated candidate file" "$a" "$b"
    fail=1
fi

# The inline queue and the two-pass path must emit the same relation set.
run --cofactor --cof-rounds 2 --cof-budget 65536 --relations $TMP/inline.txt >/dev/null
sort $TMP/inline.txt > $TMP/x
sort $TMP/td.txt $TMP/s1.txt > $TMP/y
if cmp -s $TMP/x $TMP/y; then
    printf 'PASS   %-34s byte-identical\n' "inline == two-pass"
else
    printf 'FAIL   %-34s differ\n' "inline == two-pass"; fail=1
fi

# Post-cofactor reconstruction: what was EMITTED must rebuild both norms. The
# pre-split gate cannot see this, and a corrupted factor is the negative control.
for f in inline s1; do
    if ./bench --check-relations $TMP/$f.txt --poly $POLY 2>&1 | grep -q PASS; then
        printf 'PASS   %-34s %s\n' "emitted factors rebuild norms" "$f"
    else
        printf 'FAIL   %-34s %s\n' "emitted factors rebuild norms" "$f"; fail=1
    fi
done
sed '1s/:\([0-9a-f]*\),/:\1f,/' $TMP/inline.txt > $TMP/corrupt.txt
if ./bench --check-relations $TMP/corrupt.txt --poly $POLY 2>&1 | grep -q FAIL; then
    printf 'PASS   %-34s corruption detected\n' "reconstruction negative control"
else
    printf 'FAIL   %-34s corruption NOT detected\n' "reconstruction negative control"; fail=1
fi

[ $fail = 0 ] && echo "cofactor golden test passed" || { echo "cofactor golden test FAILED"; exit 1; }
