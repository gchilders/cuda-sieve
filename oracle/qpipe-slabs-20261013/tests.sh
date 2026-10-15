#!/bin/sh
# Multi-slab --q-pipeline (finding 109) test plan. Run only with the card free.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/slabpipe
NFS='--poly /tmp/ggnfs-client/files/81864552429a3e62d5b9a480b7c3664aabd048b4a7f5df23094cb6f3bc0c753a --fb1 /tmp/ggnfs-client/files/81864552429a3e62.roots1.m16 --sq-side 1 --logI 16 --J 32768'
C194='--poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768'

echo "== 1. identity gate (pipeline expected on all five jobs)"
./fillcheck.sh > $W/fillcheck.log 2>&1; echo "fillcheck exit=$?"; tail -2 $W/fillcheck.log

echo "== 2. the NFS@Home job (4 slabs): identity, single-queued vs pipelined, 40 q"
./bench --pipeline --cofactor $NFS --qrange 296847000: --nq 40 --no-q-pipeline --relations $W/nfs.single.rel > $W/nfs.single.out 2>&1
./bench --pipeline --cofactor $NFS --qrange 296847000: --nq 40 --relations $W/nfs.qp.rel > $W/nfs.qp.out 2>&1
grep -h -- "--q-pipeline:" $W/nfs.qp.out
cmp -s $W/nfs.single.rel $W/nfs.qp.rel && echo "nfs IDENTICAL ($(wc -l < $W/nfs.qp.rel))" || echo "nfs DIFFER"

echo "== 3. stop and resume on c194 (4 slabs) with work in flight"
mkdir -p $W/stop && rm -f $W/stop/STOP
./bench --pipeline --cofactor $C194 --qrange 80000023: --nq 120 --no-q-pipeline --relations $W/stop/base.rel > $W/stop/base.out 2>&1
( sleep 20; touch $W/stop/STOP ) &
./bench --pipeline --cofactor $C194 --qrange 80000023: --nq 120 --stop-file $W/stop/STOP --relations $W/stop/qp.rel > $W/stop/qp1.out 2>&1
wait
grep -h "stopping cleanly" $W/stop/qp1.out
rm -f $W/stop/STOP
./bench --pipeline --cofactor $C194 --qrange 80000023: --nq 120 --stop-file $W/stop/STOP --relations $W/stop/qp.rel > $W/stop/qp2.out 2>&1
cmp -s $W/stop/base.rel $W/stop/qp.rel && echo "stop/resume IDENTICAL ($(wc -l < $W/stop/qp.rel))" || echo "stop/resume DIFFER"

echo "== 4. timing, interleaved, 3 rounds: c194 and the NFS@Home job"
for r in 0 1 2; do
  arms="single qp"; [ $((r % 2)) -eq 1 ] && arms="qp single"   # alternate order against drift
  for arm in $arms; do
    a=""; [ $arm = single ] && a=--no-q-pipeline
    for job in c194 nfs; do
      if [ $job = c194 ]; then j="$C194 --qrange 80000023: --nq 60"; else j="$NFS --qrange 296847000: --nq 60"; fi
      # shellcheck disable=SC2086
      ./bench --pipeline --cofactor $j $a > $W/t.$job.$arm.$r.out 2>&1
      echo "$job $arm $r $(grep 'wall clock per q, COMPLETE' $W/t.$job.$arm.$r.out | tr -s ' ') load=$(cut -d' ' -f1 /proc/loadavg)"
    done
  done
done
echo done
