#!/bin/sh
# Bucket overflow forced at the settings where SOME slabs overflow
# (BENCH_TEST_CAP_PERMILLE on a PIPE_CAP_PCT test build, which refuses
# --relations). Single-queued vs pipelined, compared on the band summary.
R=/home/kylea/code/cuda-sieve; V=$R/work/hostwait/review; X=$V/cap95/bench/bench
cd $R/bench || exit 1
summ() { { grep -oE "[0-9]+ slab\(s\) skipped or abandoned" "$1"; grep -hE "COMPLETE RELATIONS/q|total relations|total candidates" "$1" | tr -s ' '; } | tr '\n' '|'; }
run() {   # name permille args...
    name=$1; pm=$2; shift 2
    BENCH_TEST_CAP_PERMILLE=$pm $X "$@" --no-q-pipeline > $V/ovf2.$name.single.out 2>&1
    BENCH_TEST_CAP_PERMILLE=$pm $X "$@" > $V/ovf2.$name.qp.out 2>&1
    a=$(summ $V/ovf2.$name.single.out); b=$(summ $V/ovf2.$name.qp.out)
    [ "$a" = "$b" ] && r=SAME || r=DIFFER
    echo "$name pm=$pm: $r | qp $(grep -h 'issued ahead' $V/ovf2.$name.qp.out | tr -s ' ')"
    echo "   single: $a"; echo "   qp:     $b"
}
run C143 880 --pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 1000
run c183 890 --pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200
run C194 887 --pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 60
echo done
