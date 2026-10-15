#!/bin/sh
# Gates for the host-wait change (readbacks folded + early next-q prep):
# default fat build checks, make check, fillcheck, stop/resume with work in
# flight on C194 (slabbed) and C143 (one slab, early prep on every q),
# skipslab on BN_LIMBS=4, tdskip on PIPE_K=8.
R=/home/kylea/code/cuda-sieve; G=$R/work/hostwait/gates; H=$R/work/hostwait; L4=$H/B/bench-l4
cd $R/bench || exit 1
echo "fat: $(cat $G/fat.done) | flags $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $G/fat.build.log | sort -u) | archs $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $G/fat.build.log | sort -u | wc -l) | warnings $(grep -i warning $G/fat.build.log | grep -vc minnctapersm)"
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
nice -n 5 make check > $G/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $G/check.log) PASS, $(grep -c '^FAIL' $G/check.log) FAIL) $(tail -1 $G/check.log)"
./fillcheck.sh > $G/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $G/fillcheck.log) ok) $(tail -1 $G/fillcheck.log)"
stoprun() {   # name, ref file, delay s, args...
    name=$1; ref=$2; delay=$3; shift 3
    D=$G/stop-$name; mkdir -p "${D:?}"; rm -f "${D:?}/STOP" "${D:?}/qp.rel" "${D:?}/qp.rel.part"
    ( sleep $delay; touch "$D/STOP" ) &
    ./bench "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp1.out" 2>&1; echo "  $name stop run exit=$?"
    wait
    if grep -q "stopping cleanly" "$D/qp1.out"; then echo "  $name: $(grep -h 'stopped after' "$D/qp1.out" | tr -s ' ' | cut -c1-90)"; else echo "  $name: NOT STOPPED -- vacuous"; fi
    rm -f "${D:?}/STOP"
    ./bench "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp2.out" 2>&1; echo "  $name resume exit=$?"
    cmp -s "$ref" "$D/qp.rel" && echo "  $name stop/resume IDENTICAL ($(wc -l < "$D/qp.rel"))" || echo "  $name stop/resume DIFFER ($(wc -l < "$D/qp.rel") vs $(wc -l < "$ref"))"
}
echo "== stop/resume"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 400"
stoprun c194 $R/work/slabpipe/stop2/base.rel 45 $C194
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 4000"
$H/bench.3e79628 $C143 --relations $G/c143-base.rel > $G/c143-base.out 2>&1
stoprun c143 $G/c143-base.rel 12 $C143
echo "== skipslab (BN_LIMBS=4): $(cat $G/l4.done)"
( cd $L4 && sh $R/work/slabpipe/skipslab.sh > $G/skipslab.log 2>&1; echo "skipslab exit=$?" ); grep -E "^PASS|^FAIL|all cases|FAILED|issued ahead" $G/skipslab.log | tr -s ' ' | head -20
echo "== tdskip (PIPE_K=8)"
( cd $L4 && nice -n 5 make -j2 GPU_ARCH=120 PIPE_K=8 BN_LIMBS=12 bench > $G/pk8.build.log 2>&1; echo "pk8 build exit=$?" )
A="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150"
$L4/bench $A --no-q-pipeline --relations $G/tdskip.single.rel > $G/tdskip.single.out 2>&1; echo "  single exit=$?"
$L4/bench $A --relations $G/tdskip.qp.rel > $G/tdskip.qp.out 2>&1; echo "  qp exit=$?"
echo "  $(grep -h 'slab(s) skipped or abandoned' $G/tdskip.qp.out | cut -c1-40) | $(grep -h 'issued ahead' $G/tdskip.qp.out | tr -s ' ')"
cmp -s $G/tdskip.single.rel $G/tdskip.qp.rel && echo "  tdskip IDENTICAL ($(wc -l < $G/tdskip.qp.rel); finding 109 had 6,875)" || echo "  tdskip DIFFER"
echo done
