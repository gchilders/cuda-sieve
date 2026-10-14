#!/bin/sh
# The slabbed TD-skip path under the pipeline: build PIPE_K=8 (lists truncate,
# slabs are dropped one at a time), compare pipelined vs single-queued on a
# 4-slab c183 band, then restore the default build.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/slabpipe
nice -n 5 make -j2 GPU_ARCH=120 PIPE_K=8 bench > $W/tdskip.build.log 2>&1 || { echo "build failed"; exit 1; }
A='--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150'
./bench $A --no-q-pipeline --relations $W/tdskip.single.rel > $W/tdskip.single.out 2>&1; echo "single exit=$?"
./bench $A --relations $W/tdskip.qp.rel > $W/tdskip.qp.out 2>&1; echo "qp exit=$?"
grep -h -- "--q-pipeline:" $W/tdskip.qp.out
grep -hiE "slab.*skipped|skipped .* slab|TD.*skip|truncat" $W/tdskip.single.out | head -3
grep -hiE "slab.*skipped|skipped .* slab|TD.*skip|truncat" $W/tdskip.qp.out | head -3
cmp -s $W/tdskip.single.rel $W/tdskip.qp.rel && echo "tdskip IDENTICAL ($(wc -l < $W/tdskip.qp.rel))" || echo "tdskip DIFFER"
nice -n 5 make -j2 bench normscan > $W/restore.build.log 2>&1; echo "restore exit=$?"
grep -o "\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*" $W/restore.build.log | sort -u
echo done
