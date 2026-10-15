#!/bin/sh
# After the third review: slabbed skip test (BN_LIMBS=4), then the default
# build, the identity gate with the issued-ahead checks, and a stop/resume
# rerun that asserts the band really stopped.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/slabpipe
nice -n 5 make -j2 GPU_ARCH=120 BN_LIMBS=4 bench normscan > $W/r2.build4.log 2>&1 || { echo "limbs4 build failed"; exit 1; }
sh $W/skipslab.sh > $W/skipslab.log 2>&1; echo "skipslab exit=$?"
nice -n 5 make -j2 bench normscan > $W/r2.build.log 2>&1 || { echo "default build failed"; exit 1; }
grep -o "\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*" $W/r2.build.log | sort -u
./bench --verify-only 2>&1 | grep -c "\[verify\] OK" | sed 's/^/verify OK lines: /'
./fillcheck.sh > $W/fillcheck2.log 2>&1; echo "fillcheck exit=$?"
D=$W/stop3; mkdir -p $D
A='--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768 --qrange 80000023: --nq 400'
( sleep 45; touch $D/STOP ) &
./bench $A --stop-file $D/STOP --relations $D/qp.rel > $D/qp1.out 2>&1; echo "stop run exit=$?"
wait
if grep -q "stopping cleanly" $D/qp1.out; then echo "STOPPED: $(grep -h 'stopped after' $D/qp1.out)"; else echo "NOT STOPPED -- test vacuous"; fi
rm -f /home/kylea/code/cuda-sieve/work/slabpipe/stop3/STOP
./bench $A --stop-file $D/STOP --relations $D/qp.rel > $D/qp2.out 2>&1; echo "resume exit=$?"
cmp -s $W/stop2/base.rel $D/qp.rel && echo "stop/resume IDENTICAL ($(wc -l < $D/qp.rel))" || echo "stop/resume DIFFER"
echo done
