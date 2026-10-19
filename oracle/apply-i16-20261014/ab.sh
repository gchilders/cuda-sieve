#!/bin/sh
# A/B of the thread-tier load hoist. base/ and exp/ are `git archive HEAD`
# copies of bench/, both built GPU_ARCH=120; only exp/ carries the change.
# Runs from the real bench/ dir so ../oracle paths resolve; the binaries are
# addressed by absolute path. bench/bench itself is never rebuilt.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof
B=$W/base/bench/bench; E=$W/exp/bench/bench
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
load() { cut -d' ' -f1 /proc/loadavg; }
echo "== identity (pipelined, default flags)"
$B $NFS --qrange 296847000: --nq 40 --relations $W/id.nfs.base.rel > $W/id.nfs.base.out 2>&1
$E $NFS --qrange 296847000: --nq 40 --relations $W/id.nfs.exp.rel > $W/id.nfs.exp.out 2>&1
cmp -s $W/id.nfs.base.rel $W/id.nfs.exp.rel && echo "nfs IDENTICAL ($(wc -l < $W/id.nfs.exp.rel))" || echo "nfs DIFFER"
$B $C183 --qrange 130000000: --nq 200 --relations $W/id.c183.base.rel > $W/id.c183.base.out 2>&1
$E $C183 --qrange 130000000: --nq 200 --relations $W/id.c183.exp.rel > $W/id.c183.exp.out 2>&1
cmp -s $W/id.c183.base.rel $W/id.c183.exp.rel && echo "c183 IDENTICAL ($(wc -l < $W/id.c183.exp.rel))" || echo "c183 DIFFER"
echo "== timing, single-queued, ABBA x2: apply ms/q and wall per q COMPLETE"
n=0
for round in 0 1; do
  for arm in base exp exp base; do
    n=$((n+1))
    [ $arm = base ] && X=$B || X=$E
    $X $NFS --qrange 297000000: --nq 60 --no-q-pipeline > $W/t.nfs.$arm.$round.$n.out 2>&1
    echo "nfs  $arm r$round $(grep -h '^      apply ' $W/t.nfs.$arm.$round.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $W/t.nfs.$arm.$round.$n.out | tr -s ' ') load=$(load)"
    $X $C183 --qrange 131000000: --nq 300 --no-q-pipeline > $W/t.c183.$arm.$round.$n.out 2>&1
    echo "c183 $arm r$round $(grep -h '^      apply ' $W/t.c183.$arm.$round.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $W/t.c183.$arm.$round.$n.out | tr -s ' ') load=$(load)"
  done
done
echo "== production wall, pipelined, NFS job, ABBA"
n=0
for arm in base exp exp base; do
  n=$((n+1))
  [ $arm = base ] && X=$B || X=$E
  $X $NFS --qrange 297000000: --nq 60 > $W/p.nfs.$arm.$n.out 2>&1
  echo "nfs-qp $arm $(grep -h 'wall clock per q, COMPLETE' $W/p.nfs.$arm.$n.out | tr -s ' ') load=$(load)"
done
echo done
