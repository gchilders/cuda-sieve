#!/bin/sh
# Single-loop two-step vs the hoist build, both GPU_ARCH=120 copies.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof; T=$W/two2
H=$W/exp/bench/bench; X=$T/bench/bench
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
load() { cut -d' ' -f1 /proc/loadavg; }
for b in hoist two2; do [ $b = hoist ] && B=$H || B=$X
  ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum --kernel-name k_apply --launch-skip 24 --launch-count 8 --csv $B $NFS --qrange 296847000: --nq 5 --no-q-pipeline 2>/dev/null > $T/ncu.$b.csv
  echo "ncu $b: $(python3 -c "
import csv
rows=[r for r in csv.reader(open('$T/ncu.$b.csv')) if len(r)>10 and r[0].isdigit()]
d={}
for r in rows: d.setdefault(r[0],{})[r[-3]]=float(r[-1].replace(',',''))
v=[d[k] for k in sorted(d,key=int)]
t=[x.get('gpu__time_duration.sum',0) for x in v]; i=[x.get('smsp__inst_executed.sum',0) for x in v]
alg=t[0::2]; rat=t[1::2]
print('alg %.3f ms, rat %.3f ms (means), per q %.2f ms, inst %.4g' % (sum(alg)/len(alg), sum(rat)/len(rat), sum(t), sum(i)))
")"
done
$X $NFS --qrange 296847000: --nq 40 --relations $T/id.nfs.rel > $T/id.nfs.out 2>&1
cmp -s $W/id.nfs.base.rel $T/id.nfs.rel && echo "nfs vs pre-change IDENTICAL ($(wc -l < $T/id.nfs.rel))" || echo "nfs vs pre-change DIFFER"
n=0
for r in 0 1; do for arm in hoist two2 two2 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=$X
  $B $NFS --qrange 297000000: --nq 60 --no-q-pipeline > $T/t.$arm.$n.out 2>&1
  echo "nfs  $arm $(grep -h '^      apply ' $T/t.$arm.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $T/t.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
for r in 0 1; do for arm in hoist two2 two2 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=$X
  $B $NFS --qrange 297000000: --nq 60 > $T/p.$arm.$n.out 2>&1
  echo "nfs-qp $arm $(grep -h 'wall clock per q, COMPLETE' $T/p.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
for arm in hoist two2 two2 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=$X
  $B $C183 --qrange 131000000: --nq 300 --no-q-pipeline > $T/c.$arm.$n.out 2>&1
  echo "c183 $arm $(grep -h '^      apply ' $T/c.$arm.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $T/c.$arm.$n.out | tr -s ' ') load=$(load)"; done
echo done
