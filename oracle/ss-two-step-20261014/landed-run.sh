#!/bin/sh
# The two-step tail, final form: k_apply's SSTAIL instance chosen per launch.
# Gates, apply-parity --verify at geometries with and without a tail,
# identity, then fat-vs-fat against the hoist build (bench.hoist).
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof; V=$W/v4; H=$W/fallback/bench.hoist
echo "flags: $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $V/build.log | sort -u), archs: $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $V/build.log | sort -u | wc -l)"
for a in 120 90 89 86 80 75; do echo "k_apply sm_$a: $(cuobjdump --dump-resource-usage -arch sm_$a ./bench 2>/dev/null | grep -A1 k_apply | grep -oE 'REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+' | paste - - - | sort | uniq -c | tr -s ' \t\n' ' ')"; done
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
for g in "--logI 16 --J 32768" "--logI 17 --J 8192" "--logI 15 --J 16384"; do o=$V/verify.$(echo $g | tr -d ' -').out
  ./bench --verify $g > $o 2>&1; st=$?
  echo "--verify $g: exit=$st | $(grep -h 'ss_first cases\|regions match\|cells differ' $o | tr -s ' ' | tr '\n' '|') $(grep -ci 'mismatch\|FAIL' $o) mismatch/FAIL"
done
nice -n 5 make check > $V/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $V/check.log) PASS, $(grep -c '^FAIL' $V/check.log) FAIL) $(tail -1 $V/check.log)"
./fillcheck.sh > $V/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $V/fillcheck.log) ok) $(tail -1 $V/fillcheck.log)"
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
./bench $NFS --qrange 296847000: --nq 40 --relations $V/id.nfs.rel > $V/id.nfs.out 2>&1
cmp -s $W/id.nfs.base.rel $V/id.nfs.rel && echo "nfs vs pre-change IDENTICAL ($(wc -l < $V/id.nfs.rel))" || echo "nfs vs pre-change DIFFER"
./bench $C183 --qrange 130000000: --nq 200 --relations $V/id.c183.rel > $V/id.c183.out 2>&1
cmp -s $W/id.c183.base.rel $V/id.c183.rel && echo "c183 vs pre-change IDENTICAL ($(wc -l < $V/id.c183.rel))" || echo "c183 vs pre-change DIFFER"
for b in hoist v4; do [ $b = hoist ] && B=$H || B=./bench
  ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum --kernel-name k_apply --launch-skip 24 --launch-count 8 --csv $B $NFS --qrange 296847000: --nq 5 --no-q-pipeline 2>/dev/null > $V/ncu.$b.csv
  echo "ncu I16 $b: $(python3 -c "
import csv
rows=[r for r in csv.reader(open('$V/ncu.$b.csv')) if len(r)>10 and r[0].isdigit()]
d={}
for r in rows: d.setdefault(r[0],{})[r[-3]]=float(r[-1].replace(',',''))
v=[d[k] for k in sorted(d,key=int)]
t=[x.get('gpu__time_duration.sum',0)/1e6 for x in v]; i=[x.get('smsp__inst_executed.sum',0) for x in v]
print('alg %.3f ms, rat %.3f ms (means), per q %.2f ms, inst %.4g' % (sum(t[0::2])/len(t[0::2]), sum(t[1::2])/len(t[1::2]), sum(t), sum(i)))
")"
done
load() { cut -d' ' -f1 /proc/loadavg; }
n=0
for r in 0 1; do for arm in hoist v4 v4 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=./bench
  $B $NFS --qrange 297000000: --nq 60 --no-q-pipeline > $V/t.$arm.$n.out 2>&1
  echo "nfs  $arm $(grep -h '^      apply ' $V/t.$arm.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $V/t.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
for r in 0 1; do for arm in hoist v4 v4 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=./bench
  $B $NFS --qrange 297000000: --nq 60 > $V/p.$arm.$n.out 2>&1
  echo "nfs-qp $arm $(grep -h 'wall clock per q, COMPLETE' $V/p.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
for r in 0 1 2; do for arm in hoist v4 v4 hoist; do n=$((n+1)); [ $arm = hoist ] && B=$H || B=./bench
  $B $C183 --qrange 131000000: --nq 300 --no-q-pipeline > $V/c.$arm.$n.out 2>&1
  echo "c183 $arm $(grep -h '^      apply ' $V/c.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
echo done
