#!/bin/sh
# Finding 111 after the xhigh review: ss_needs_tail + trap, one loop source,
# harness instance choice, shared gate scaffold, slabtest tail gate.
# Build, SASS shape per instance, gates, identity on all four pipeline
# instances against the pre-change 5c1ab93 copy, then fat-vs-fat timing.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof; V=$W/v5; H=$W/fallback/bench.hoist; B0=$W/base/bench/bench
nice -n 5 make -j2 bench normscan > $V/build.log 2>&1 || { echo "build failed"; grep -i error -A3 $V/build.log | head -20; exit 1; }
echo "flags: $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $V/build.log | sort -u), archs: $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $V/build.log | sort -u | wc -l), warnings other than minnctapersm: $(grep -i warning $V/build.log | grep -vc minnctapersm)"
for f in $(cuobjdump -sass ./bench 2>/dev/null | grep -oE "Function : _Z7k_applyILi16ELi1ELi1E[A-Za-z0-9_]*" | cut -d' ' -f3 | sort -u); do
  s=$(cuobjdump -sass -arch sm_120 -fun "$f" ./bench 2>/dev/null)
  echo "sass $(echo "$f" | grep -oE 'ILi16ELi1ELi1E(Lb[01]E)+' | sed 's/ILi16ELi1ELi1E//'): YIELD $(echo "$s" | grep -c YIELD), BSSY.RECONVERGENT $(echo "$s" | grep -cE 'BSSY\.RECONVERGENT'), plain BSSY $(echo "$s" | grep -cE 'BSSY [B]'), BPT.TRAP $(echo "$s" | grep -c 'BPT.TRAP')"
done
for a in 120 90 89 86 80 75; do echo "k_apply sm_$a: $(cuobjdump --dump-resource-usage -arch sm_$a ./bench 2>/dev/null | grep -A1 k_apply | grep -oE 'REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+' | paste - - - | sort | uniq -c | tr -s ' \t\n' ' ')"; done
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
for g in "--logI 16 --J 32768" "--logI 17 --J 8192" "--logI 15 --J 16384" "--logI 17 --J 4096"; do o=$V/verify.$(echo $g | tr -d ' -').out
  ./bench --verify $g > $o 2>&1; st=$?
  echo "--verify $g: exit=$st | $(grep -h 'ss_first cases\|regions match\|cells differ' $o | tr -s ' ' | tr '\n' '|') $(grep -ci 'mismatch\|FAIL' $o) mismatch/FAIL"
done
nice -n 5 make check > $V/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $V/check.log) PASS, $(grep -c '^FAIL' $V/check.log) FAIL) $(grep -h 'small-sieve tail' $V/check.log) | $(tail -1 $V/check.log)"
./fillcheck.sh > $V/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $V/fillcheck.log) ok) $(tail -1 $V/fillcheck.log)"
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1"
./bench $NFS --qrange 296847000: --nq 40 --relations $V/id.nfs.rel > $V/id.nfs.out 2>&1
cmp -s $W/id.nfs.base.rel $V/id.nfs.rel && echo "instance slabbed+tail (NFS I16): IDENTICAL ($(wc -l < $V/id.nfs.rel)) | $(grep -h 'j-slabbing' $V/id.nfs.out | cut -c1-40)" || echo "instance slabbed+tail (NFS I16): DIFFER"
./bench $C --logI 15 --qrange 130000000: --nq 200 --relations $V/id.c183.rel > $V/id.c183.out 2>&1
cmp -s $W/id.c183.base.rel $V/id.c183.rel && echo "instance unslabbed+no tail (c183 I15): IDENTICAL ($(wc -l < $V/id.c183.rel)) | slabbing: $(grep -c 'j-slabbing' $V/id.c183.out)" || echo "instance unslabbed+no tail (c183 I15): DIFFER"
for cfg in "slab|--logI 15 --slab-j 4095 --qrange 130000000: --nq 60|slabbed+no tail (c183 I15 slab-j 4095)" "i17|--logI 17 --J 4096 --qrange 130000000: --nq 30|unslabbed+tail (c183 I17 J4096)"; do
  tag=${cfg%%|*}; rest=${cfg#*|}; args=${rest%%|*}; label=${rest#*|}
  $B0 $C $args --relations $V/id.$tag.base.rel > $V/id.$tag.base.out 2>&1
  ./bench $C $args --relations $V/id.$tag.rel > $V/id.$tag.out 2>&1
  cmp -s $V/id.$tag.base.rel $V/id.$tag.rel && echo "instance $label: IDENTICAL to 5c1ab93 ($(wc -l < $V/id.$tag.rel)) | $(grep -h 'j-slabbing' $V/id.$tag.out | cut -c1-40)" || echo "instance $label: DIFFER from 5c1ab93"
done
for b in hoist v5; do [ $b = hoist ] && X=$H || X=./bench
  ncu --metrics gpu__time_duration.sum,smsp__inst_executed.sum --kernel-name k_apply --launch-skip 24 --launch-count 8 --csv $X $NFS --qrange 296847000: --nq 5 --no-q-pipeline 2>/dev/null > $V/ncu.$b.csv
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
for r in 0 1; do for arm in hoist v5 v5 hoist; do n=$((n+1)); [ $arm = hoist ] && X=$H || X=./bench
  $X $NFS --qrange 297000000: --nq 60 --no-q-pipeline > $V/t.$arm.$n.out 2>&1
  echo "nfs  $arm $(grep -h '^      apply ' $V/t.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
for r in 0 1 2; do for arm in hoist v5 v5 hoist; do n=$((n+1)); [ $arm = hoist ] && X=$H || X=./bench
  $X $C --logI 15 --qrange 131000000: --nq 300 --no-q-pipeline > $V/c.$arm.$n.out 2>&1
  echo "c183 $arm $(grep -h '^      apply ' $V/c.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
echo done
