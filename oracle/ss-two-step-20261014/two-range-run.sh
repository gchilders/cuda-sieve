#!/bin/sh
# Two-step ss_first for the thread tier's refused tail (finding 110's next
# lever): default fat build, gates, apply-parity --verify at geometries that
# reach the two-step range, identity, ncu call count, A/B vs the hoist binary.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof; F=$W/fallback
nice -n 5 make -j2 bench normscan > $F/build.log 2>&1 || { echo "build failed"; tail -20 $F/build.log; exit 1; }
echo "flags: $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $F/build.log | sort -u)"
echo "archs: $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $F/build.log | sort -u | wc -l)"
echo "warnings other than minnctapersm: $(grep -i warning $F/build.log | grep -vc minnctapersm)"
for a in 120 90 89 86 80 75; do echo "k_apply sm_$a: $(cuobjdump --dump-resource-usage -arch sm_$a ./bench 2>/dev/null | grep -A1 k_apply | grep -oE 'REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+' | paste - - - | sort | uniq -c | tr -s ' \t\n' ' ')"; done
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
for g in "--logI 16 --J 32768" "--logI 17 --J 8192" "--logI 15 --J 16384"; do
  ./bench --verify $g > $F/verify.$(echo $g | tr -d ' -').out 2>&1; st=$?
  echo "--verify $g: exit=$st | $(grep -h 'ss_first cases\|small sieve:' $F/verify.$(echo $g | tr -d ' -').out | tr -s ' ' | tr '\n' '|') $(grep -c 'OK' $F/verify.$(echo $g | tr -d ' -').out) OK lines, $(grep -ci 'mismatch\|FAIL' $F/verify.$(echo $g | tr -d ' -').out) mismatch/FAIL"
done
nice -n 5 make check > $F/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $F/check.log) PASS, $(grep -c '^FAIL' $F/check.log) FAIL) $(tail -1 $F/check.log)"
./fillcheck.sh > $F/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $F/fillcheck.log) ok) $(tail -1 $F/fillcheck.log)"
NFS="--pipeline --cofactor --poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
./bench $NFS --qrange 296847000: --nq 40 --relations $F/id.nfs.rel > $F/id.nfs.out 2>&1
cmp -s $W/id.nfs.base.rel $F/id.nfs.rel && echo "nfs vs pre-change IDENTICAL ($(wc -l < $F/id.nfs.rel))" || echo "nfs vs pre-change DIFFER"
./bench $C183 --qrange 130000000: --nq 200 --relations $F/id.c183.rel > $F/id.c183.out 2>&1
cmp -s $W/id.c183.base.rel $F/id.c183.rel && echo "c183 vs pre-change IDENTICAL ($(wc -l < $F/id.c183.rel))" || echo "c183 vs pre-change DIFFER"
for X in $F/bench.hoist ./bench; do
  ncu --section SourceCounters --kernel-name k_apply --launch-skip 24 --launch-count 2 -f -o $F/ncu.$(basename $X) $X $NFS --qrange 296847000: --nq 5 --no-q-pipeline > $F/ncu.$(basename $X).log 2>&1
  for L in 0 1; do ncu --import $F/ncu.$(basename $X).ncu-repz --page source --csv --print-source sass --launch-skip $L --launch-count 1 2>/dev/null > $F/ncu.$(basename $X).L$L.csv; done
  echo "ncu $(basename $X): CALL.REL warp-executions alg/rat: $(for L in 0 1; do python3 -c "
import csv,sys
n=0
for r in csv.reader(open('$F/ncu.$(basename $X).L$L.csv')):
    if len(r)>7 and 'CALL.REL' in ' '.join(r[:4]):
        try: n+=float(r[7] if r[7] else 0)
        except ValueError: pass
print(int(n), end=' ')"; done)"
done
load() { cut -d' ' -f1 /proc/loadavg; }
echo "== A/B single-queued apply, NFS 60 q, ABBA x2"
n=0; for r in 0 1; do for arm in hoist two two hoist; do n=$((n+1)); [ $arm = hoist ] && X=$F/bench.hoist || X=./bench
  $X $NFS --qrange 297000000: --nq 60 --no-q-pipeline > $F/t.$arm.$n.out 2>&1
  echo "nfs  $arm $(grep -h '^      apply ' $F/t.$arm.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $F/t.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
echo "== A/B pipelined wall, NFS 60 q, ABBA x2"
for r in 0 1; do for arm in hoist two two hoist; do n=$((n+1)); [ $arm = hoist ] && X=$F/bench.hoist || X=./bench
  $X $NFS --qrange 297000000: --nq 60 > $F/p.$arm.$n.out 2>&1
  echo "nfs-qp $arm $(grep -h 'wall clock per q, COMPLETE' $F/p.$arm.$n.out | tr -s ' ') load=$(load)"; done; done
echo "== c183 single-queued apply, ABBA (expect no change: no two-step range at I15)"
for arm in hoist two two hoist; do n=$((n+1)); [ $arm = hoist ] && X=$F/bench.hoist || X=./bench
  $X $C183 --qrange 131000000: --nq 300 --no-q-pipeline > $F/c.$arm.$n.out 2>&1
  echo "c183 $arm $(grep -h '^      apply ' $F/c.$arm.$n.out | tr -s ' ') | $(grep -h 'wall clock per q, COMPLETE' $F/c.$arm.$n.out | tr -s ' ') load=$(load)"; done
echo done
