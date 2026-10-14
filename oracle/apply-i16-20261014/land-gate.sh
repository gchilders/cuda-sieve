#!/bin/sh
# Land the thread-tier load hoist (finding 110): default fat build, gates,
# identity against the pre-change relation files from the base copy.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof; L=$W/land
nice -n 5 make -j2 bench normscan > $L/build.log 2>&1 || { echo "build failed"; exit 1; }
echo "flags: $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $L/build.log | sort -u)"
echo "archs: $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $L/build.log | sort -u | tr '\n' ' ')"
echo "ptxas warnings other than minnctapersm: $(grep -i warning $L/build.log | grep -vc minnctapersm)"
echo "k_apply resources (sm_120):"
cuobjdump --dump-resource-usage -arch sm_120 ./bench 2>/dev/null | grep -A1 "k_apply" | grep -oE "REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | paste - - - | sort | uniq -c
echo "verify OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
nice -n 5 make check > $L/check.log 2>&1; echo "make check exit=$?"
./fillcheck.sh > $L/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $L/fillcheck.log) ok) $(tail -1 $L/fillcheck.log)"
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
./bench $NFS --qrange 296847000: --nq 40 --relations $L/id.nfs.rel > $L/id.nfs.out 2>&1
cmp -s $W/id.nfs.base.rel $L/id.nfs.rel && echo "nfs vs pre-change IDENTICAL ($(wc -l < $L/id.nfs.rel))" || echo "nfs vs pre-change DIFFER"
./bench $C183 --qrange 130000000: --nq 200 --relations $L/id.c183.rel > $L/id.c183.out 2>&1
cmp -s $W/id.c183.base.rel $L/id.c183.rel && echo "c183 vs pre-change IDENTICAL ($(wc -l < $L/id.c183.rel))" || echo "c183 vs pre-change DIFFER"
./bench $NFS --qrange 297000000: --nq 60 > $L/t.nfs.out 2>&1
echo "nfs pipelined, landed build: $(grep -h 'wall clock per q, COMPLETE' $L/t.nfs.out | tr -s ' ') load=$(cut -d' ' -f1 /proc/loadavg)"
echo done
