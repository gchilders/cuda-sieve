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

# The relation-count cases pin the survivor bound EXPLICITLY, to the values the
# derivation produced before it stopped importing lambda conventions. They are
# here to test the cofactoriser -- a method that silently falls back, a factor
# base missing p = 2, a truncated parse -- and that job is only done if the
# count is held still by everything else. Letting the allowance policy move
# these numbers would mean re-baselining 37 every time the policy is tuned, and
# a re-baselined golden number stops catching the bugs it was written for.
#
# The policy itself is pinned separately, in its own case below.
PIN="--allowance 101.6 --allowance0 68.1"
run() { ./bench --pipeline --cadofb $FB --poly $POLY --qrange 120000053:120000053 $PIN "$@" 2>&1; }

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
# `|| true` on every grep -c: it exits 1 when the count is 0, and `set -e` at
# the top of this script turns that into an abort partway through the suite
# rather than a FAIL line. That cost a confusing debugging round once.
got=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 \
      | grep -c 'job file: rlim 67100000, alim 134200000, lpbr 31, lpba 32, mfbr 60, mfba 92' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s all six keys\n' "job file parsed"
else
    printf 'FAIL   %-34s banner line not found\n' "job file parsed"; fail=1
fi

# The job file's lambda is REPORTED and NOT APPLIED. Two things to pin: that
# the GGNFS-unit conversion is still right (3.5 * log2(134200000) = 94.5 bits,
# not the 112 a CADO reading would give), and that the number does not reach
# the survivor bound -- it is calibrated to GGNFS's gate, which is measurably
# tighter than ours at the same nominal bits.
# `|| true` for the same reason as the greps: a bare command substitution under
# `set -e` takes bench's exit status, so a failing case would abort the suite
# instead of printing FAIL and continuing.
out=$(./bench --pipeline --cadofb $FB --poly $JOB $Q --relations $TMP/j.txt 2>&1 || true)
got=$(printf '%s' "$out" | grep -c 'alambda 3.5 -> 94.5[0-9]* bits (side 1), reported only' || true)
if [ "$got" = "1" ]; then
    printf 'PASS   %-34s 3.5 -> 94.5 bits, reported not applied\n' "GGNFS lambda conversion"
else
    printf 'FAIL   %-34s wrong or missing conversion\n' "GGNFS lambda conversion"; fail=1
fi
# ...and the allowance actually in force is the DERIVED one, not 94.50.
got=$(printf '%s' "$out" | grep -c 'side 1 log2(maxnorm)=.* allowance=94.50' || true)
if [ "$got" = "0" ]; then
    printf 'PASS   %-34s job lambda did not reach the bound\n' "lambda not applied"
else
    printf 'FAIL   %-34s job lambda leaked into the allowance\n' "lambda not applied"; fail=1
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

# ---- cofactor width -------------------------------------------------------
# Side 0 is 3 limbs like side 1, so an SNFS job with mfbr 88 on the rational
# side is representable. 97 is not, and 96 against lpb0 31 asks for 4 parts,
# which mz_split's stack cannot hold -- it would present as CF_OVERFLOW on the
# records needing it, i.e. a silent partial yield loss, so it is refused.
expect_refused "side-0 mfb above 3 limbs"     --cofactor --mfb0 97
expect_refused "side-0 4LP refused"           --cofactor --mfb0 96 --lpb0 31
expect_refused "side-1 4LP refused"           --cofactor --mfb 96 --lpb 24
# $PIN so this measures the cofactor WIDTH and not the allowance policy: with
# --mfb0 88 and no pin, the derived allowance0 follows mfb to 89.5 bits and the
# case would be exercising the survivor bound instead of the thing it is named
# for.
if ./bench --pipeline --cadofb $FB --poly $POLY $Q $PIN --cofactor --mfb0 88 --lpb0 31 \
           --relations $TMP/w.txt >$TMP/o 2>&1; then
    printf 'PASS   %-34s accepted\n' "side-0 3LP width (mfb0 88)"
else
    printf 'FAIL   %-34s refused, should be legal\n' "side-0 3LP width (mfb0 88)"; fail=1
fi

expect_refused "sq-side out of range"         --cofactor --sq-side 2

# ---- survivor allowance policy -------------------------------------------
# The DERIVED default (mfb + ~1.5 bits, from our own quantisation) is tighter
# than the old CADO-rule value, and on this single q that costs one relation:
# 36 rather than the 37 the pinned cases above get. That is a deliberate trade,
# measured over a real band rather than this one q -- c183, 120 special-q:
#
#     allowance0 61.5 (mfb+1.5)   82,969 survivors/q   46.48 rel/q
#     allowance0 68.1 (mfb+8.1)  143,760 survivors/q   46.57 rel/q
#
# 42% of the trial-division input for 0.19% of the relations. The single-q
# figure of 1-in-37 is an unlucky sample, not the rate.
#
# Pinned so that a change to the allowance policy has to be deliberate: if this
# moves, the derivation moved, and the band measurement above needs redoing.
got=$(./bench --pipeline --cadofb $FB --poly $POLY $Q --cofactor --cof-rounds 2 \
      --cof-budget 65536 --relations $TMP/pol.txt 2>&1 \
      | grep 'total relations' | tail -1 | awk '{print $NF}')
if [ "$got" = "36" ]; then
    printf 'PASS   %-34s 36 relations\n' "derived allowance policy"
else
    printf 'FAIL   %-34s expected 36, got %s\n' "derived allowance policy" "$got"; fail=1
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

# The corruption control above only proves the gate notices a factor that does
# not DIVIDE. It says nothing about the failure that is actually likely here: a
# cofactor split that stops early and emits p*q as one "prime". That divides
# perfectly, rebuilds the norm to 1, and passed every check this gate had until
# the primality test was added -- so it needs its own control, built by merging
# a relation's first two side-0 factors into their product.
made=0
while IFS= read -r ln; do
    if [ $made = 0 ]; then
        case ${ln%%:*} in \#*) printf '%s\n' "$ln"; continue;; esac
        ab=${ln%%:*}; rest=${ln#*:}; s0=${rest%%:*}; s1=${rest#*:}
        case $s0 in
        *,*) f1=${s0%%,*}; tl=${s0#*,}; f2=${tl%%,*}; tl2=${tl#*,}
             p=$(( 0x$f1 * 0x$f2 ))
             # Must stay under 2^32 or the range check rejects it first and the
             # control passes without ever exercising primality.
             if [ $p -le 4294967295 ]; then
                 if [ "$tl" = "$f2" ]; then s0=$(printf '%x' $p)
                 else s0="$(printf '%x' $p),$tl2"; fi
                 ln="$ab:$s0:$s1"; made=1
             fi ;;
        esac
    fi
    printf '%s\n' "$ln"
done < $TMP/inline.txt > $TMP/composite.txt
if [ $made = 0 ]; then
    printf 'SKIP   %-34s no relation with two small side-0 factors\n' \
        "primality negative control"
elif ./bench --check-relations $TMP/composite.txt --poly $POLY 2>&1 | grep -q FAIL; then
    printf 'PASS   %-34s composite factor detected\n' "primality negative control"
else
    printf 'FAIL   %-34s composite NOT detected\n' "primality negative control"; fail=1
fi

# ---- generated-band control ---------------------------------------------
# Three ideals exercise all generator handoffs: q0 is cached for norm setup,
# q1 is the first pull inside run_pipeline, and q2 is another root from the
# stream. Compare with the exact oracle pairs, then make the count cap compete
# with an unreachable relation target so its termination message is gated too.
cat >$TMP/q3 <<EOF
120000007 30589397
120000053 112625526
120000103 21725368
EOF
gout=$(./bench --pipeline --cadofb $FB --poly $POLY \
       --qrange 120000000:120000200 --nq 3 --target-rels 999999 $PIN \
       --relations $TMP/qgen.txt 2>&1 || true)
lout=$(./bench --pipeline --cadofb $FB --poly $POLY --qlist $TMP/q3 --nq 10 $PIN \
       --relations $TMP/qlist.txt 2>&1 || true)
if printf '%s' "$gout" | grep -q '\[--nq 3 reached\]' &&
   printf '%s' "$gout" | grep -q 'note: --nq 3 reached.*--target-rels 999999 was not reached' &&
   printf '%s' "$gout" | grep -q -- '--- band of 3 special-q ---' &&
   printf '%s' "$lout" | grep -q -- '--- band of 3 special-q ---' &&
   printf '%s' "$lout" | grep -q '100.0%' &&
   cmp -s $TMP/qgen.txt $TMP/qlist.txt; then
    printf 'PASS   %-34s cached + streamed q, byte-identical\n' \
        "multi-q generated band"
else
    printf 'FAIL   %-34s generator/list mismatch or bad terminal status\n' \
        "multi-q generated band"; fail=1
fi

# qmin == 0 is a valid spelling that normalises to the first prime. It used to
# double as the parser's "--qrange absent" sentinel, silently dropping 0:MAX
# and running the unrelated single-q default instead.
zero=$(./bench --pipeline --cadofb $FB --poly $POLY --qrange 0:1 $PIN 2>&1 || true)
if printf '%s' "$zero" | grep -q 'no affine special-q roots in \[0, 1\]'; then
    printf 'PASS   %-34s parsed as an empty generated band\n' "zero qrange lower bound"
else
    printf 'FAIL   %-34s qrange was dropped or misreported\n' "zero qrange lower bound"; fail=1
fi

# A partial warp has no lane 31 for k_intersect_compact to broadcast its atomic
# base from. This used to be caught only by the survivor/rank cross-check, i.e.
# after a full band had run.
if ./bench --threads 33 --poly $POLY 2>&1 | grep -q 'multiple of 32'; then
    printf 'PASS   %-34s --threads 33 rejected\n' "warp-multiple guard"
else
    printf 'FAIL   %-34s --threads 33 accepted\n' "warp-multiple guard"; fail=1
fi

[ $fail = 0 ] && echo "cofactor golden test passed" || { echo "cofactor golden test FAILED"; exit 1; }
