#!/bin/sh
# Host waits (finding 113): base 3e79628 (fat) vs A (readbacks folded) vs
# B (A + early next-q prep), both sm_120 copies. Identity, then interleaved
# pipelined timing on C143 I13, C194 I16 and the NFS@Home lpba-34 I16 job.
cd /home/kylea/code/cuda-sieve/bench || exit 1
H=/home/kylea/code/cuda-sieve/work/hostwait; P=/home/kylea/code/cuda-sieve/work/applyprof
BASE=$H/bench.3e79628; A=$H/A/bench/bench; B=$H/B/bench/bench
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768"
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
load() { cut -d' ' -f1 /proc/loadavg; }
echo "== identity"
$BASE $C143 --qrange 2200000: --nq 1000 --relations $H/id.c143.base.rel > /dev/null 2>&1
$BASE $C194 --qrange 80000023: --nq 60 --relations $H/id.c194.base.rel > /dev/null 2>&1
for X in A B; do eval BIN=\$$X
  $BIN $NFS --qrange 296847000: --nq 40 --relations $H/id.nfs.$X.rel > $H/id.nfs.$X.out 2>&1
  cmp -s $P/id.nfs.base.rel $H/id.nfs.$X.rel && r1="nfs OK($(wc -l < $H/id.nfs.$X.rel))" || r1="nfs DIFFER"
  $BIN $C183 --qrange 130000000: --nq 200 --relations $H/id.c183.$X.rel > $H/id.c183.$X.out 2>&1
  cmp -s $P/id.c183.base.rel $H/id.c183.$X.rel && r2="c183 OK($(wc -l < $H/id.c183.$X.rel))" || r2="c183 DIFFER"
  $BIN $C143 --qrange 2200000: --nq 1000 --relations $H/id.c143.$X.rel > $H/id.c143.$X.out 2>&1
  cmp -s $H/id.c143.base.rel $H/id.c143.$X.rel && r3="c143 OK($(wc -l < $H/id.c143.$X.rel))" || r3="c143 DIFFER"
  $BIN $C194 --qrange 80000023: --nq 60 --relations $H/id.c194.$X.rel > $H/id.c194.$X.out 2>&1
  cmp -s $H/id.c194.base.rel $H/id.c194.$X.rel && r4="c194 OK($(wc -l < $H/id.c194.$X.rel))" || r4="c194 DIFFER"
  echo "$X: $r1 $r2 $r3 $r4 | issued ahead: $(grep -h 'issued ahead' $H/id.c194.$X.out | tr -s ' ')"
done
echo "== timing, pipelined, rounds of base A B / B A base"
n=0
for r in 0 1; do
  for X in BASE A B B A BASE; do n=$((n+1)); eval BIN=\$$X
    $BIN $C143 --qrange 2200000: --nq 3000 > $H/t.c143.$X.$n.out 2>&1
    echo "c143 $X $(grep -h 'wall clock per q, COMPLETE' $H/t.c143.$X.$n.out | tr -s ' ') load=$(load)"
  done
done
for r in 0 1; do
  for X in BASE A B B A BASE; do n=$((n+1)); eval BIN=\$$X
    $BIN $C194 --qrange 80000023: --nq 60 > $H/t.c194.$X.$n.out 2>&1
    echo "c194 $X $(grep -h 'wall clock per q, COMPLETE' $H/t.c194.$X.$n.out | tr -s ' ') load=$(load)"
    $BIN $NFS --qrange 297000000: --nq 60 > $H/t.nfs.$X.$n.out 2>&1
    echo "nfs  $X $(grep -h 'wall clock per q, COMPLETE' $H/t.nfs.$X.$n.out | tr -s ' ') load=$(load)"
  done
done
echo done
