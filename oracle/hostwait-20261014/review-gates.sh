#!/bin/sh
# Finding 114's review round: gates on the default fat build, identity on four
# jobs, stop/resume, fat-vs-fat timing, the bucket-overflow soft skip driven by
# PIPE_CAP_PCT test builds (single-queued vs pipelined), skipslab, tdskip.
R=/home/kylea/code/cuda-sieve; V=$R/work/hostwait/review; H=$R/work/hostwait; P=$R/work/applyprof
BASE=$H/bench.3e79628
cd $R/bench || exit 1
load() { cut -d' ' -f1 /proc/loadavg; }
sm() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits; }
echo "fat: $(cat $V/fat.done) | flags $(grep -o '\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*' $V/fat.build.log | sort -u) | archs $(grep -o 'arch=compute_[0-9]*,code=sm_[0-9]*' $V/fat.build.log | sort -u | wc -l) | warnings $(grep -i warning $V/fat.build.log | grep -vc minnctapersm)"
echo "verify-only OK lines: $(./bench --verify-only 2>&1 | grep -c '\[verify\] OK')"
nice -n 5 make check > $V/check.log 2>&1; echo "make check exit=$? ($(grep -c '^PASS' $V/check.log) PASS, $(grep -c '^FAIL' $V/check.log) FAIL) $(tail -1 $V/check.log)"
./fillcheck.sh > $V/fillcheck.log 2>&1; echo "fillcheck exit=$? ($(grep -c '^  ok' $V/fillcheck.log) ok) $(tail -1 $V/fillcheck.log)"
NFS="--pipeline --cofactor --poly $P/nfs.job --fb1 $P/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15"
C143="--pipeline --cofactor --poly ../oracle/c143.job --logI 13 --J 8192 --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94"
C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 --logI 16 --J 32768"
echo "== identity vs 3e79628"
./bench $NFS --qrange 296847000: --nq 40 --relations $V/id.nfs.rel > /dev/null 2>&1; cmp -s $P/id.nfs.base.rel $V/id.nfs.rel && a="nfs OK($(wc -l < $V/id.nfs.rel))" || a="nfs DIFFER"
./bench $C183 --qrange 130000000: --nq 200 --relations $V/id.c183.rel > /dev/null 2>&1; cmp -s $P/id.c183.base.rel $V/id.c183.rel && b="c183 OK($(wc -l < $V/id.c183.rel))" || b="c183 DIFFER"
./bench $C143 --qrange 2200000: --nq 1000 --relations $V/id.c143.rel > /dev/null 2>&1; cmp -s $H/id.c143.base.rel $V/id.c143.rel && c="c143 OK($(wc -l < $V/id.c143.rel))" || c="c143 DIFFER"
./bench $C194 --qrange 80000023: --nq 60 --relations $V/id.c194.rel > $V/id.c194.out 2>&1; cmp -s $H/id.c194.base.rel $V/id.c194.rel && d="c194 OK($(wc -l < $V/id.c194.rel))" || d="c194 DIFFER"
echo "$a $b $c $d | $(grep -h 'issued ahead' $V/id.c194.out | tr -s ' ')"
stoprun() {   # name, ref, delay, args...
    name=$1; ref=$2; delay=$3; shift 3
    D=$V/stop-$name; mkdir -p "${D:?}"; rm -f "${D:?}/STOP" "${D:?}/qp.rel" "${D:?}/qp.rel.part"
    ( sleep $delay; touch "$D/STOP" ) &
    ./bench "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp1.out" 2>&1
    wait
    if grep -q "stopping cleanly" "$D/qp1.out"; then st="$(grep -h 'stopped after' "$D/qp1.out" | tr -s ' ' | cut -c1-60)"; else st="NOT STOPPED -- vacuous"; fi
    rm -f "${D:?}/STOP"
    ./bench "$@" --stop-file "$D/STOP" --relations "$D/qp.rel" > "$D/qp2.out" 2>&1
    cmp -s "$ref" "$D/qp.rel" && echo "  $name: $st | resumed IDENTICAL ($(wc -l < "$D/qp.rel"))" || echo "  $name: $st | resumed DIFFER"
}
echo "== stop/resume"
stoprun c194 $R/work/slabpipe/stop2/base.rel 45 $C194 --qrange 80000023: --nq 400
stoprun c143 $H/gates/c143-base.rel 12 $C143 --qrange 2200000: --nq 4000
echo "== timing, fat vs fat, SM clock after each run"
n=0
for job in C143 C143 NFS C194; do for arm in base new new base; do n=$((n+1)); [ $arm = base ] && X=$BASE || X=./bench
  case $job in C143) A="$C143 --qrange 2200000: --nq 3000";; NFS) A="$NFS --qrange 297000000: --nq 60";; C194) A="$C194 --qrange 80000023: --nq 60";; esac
  $X $A > $V/t.$job.$arm.$n.out 2>&1
  echo "$job $arm $(grep -h 'wall clock per q, COMPLETE' $V/t.$job.$arm.$n.out | tr -s ' ') load=$(load) sm=$(sm)"
done; done
echo "== bucket overflow forced (PIPE_CAP_PCT test builds), single-queued vs pipelined"
for v in 95 85; do X=$V/cap$v/bench/bench
  for job in c183 C143 C194; do
    case $job in c183) A="$C183 --qrange 130000000: --nq 200";; C143) A="$C143 --qrange 2200000: --nq 1000";; C194) A="$C194 --qrange 80000023: --nq 60";; esac
    $X $A --no-q-pipeline --relations $V/ovf$v.$job.single.rel > $V/ovf$v.$job.single.out 2>&1
    $X $A --relations $V/ovf$v.$job.qp.rel > $V/ovf$v.$job.qp.out 2>&1
    s1=$(grep -h "slab(s) skipped or abandoned" $V/ovf$v.$job.single.out | grep -oE '^ *[0-9]+' | tr -d ' '); s2=$(grep -h "slab(s) skipped or abandoned" $V/ovf$v.$job.qp.out | grep -oE '^ *[0-9]+' | tr -d ' ')
    o2=$(grep -c "OVERFLOWED" $V/ovf$v.$job.qp.out)
    if [ -s $V/ovf$v.$job.qp.rel ]; then cmp -s $V/ovf$v.$job.single.rel $V/ovf$v.$job.qp.rel && r="IDENTICAL ($(wc -l < $V/ovf$v.$job.qp.rel))" || r="DIFFER ($(wc -l < $V/ovf$v.$job.single.rel) vs $(wc -l < $V/ovf$v.$job.qp.rel))"; else r="no relation file (rel=$(grep -h 'total relations' $V/ovf$v.$job.single.out | tr -s ' ') vs $(grep -h 'total relations' $V/ovf$v.$job.qp.out | tr -s ' '))"; fi
    echo "  cap$v $job: skipped/abandoned single ${s1:-0} qp ${s2:-0}, overflow warnings ${o2} | $r | $(grep -h 'issued ahead' $V/ovf$v.$job.qp.out | tr -s ' ')"
  done
done
echo "== skipslab (BN_LIMBS=4) and tdskip (PIPE_K=8), final source"
( cd $V/cap95/bench && nice -n 5 make -j2 GPU_ARCH=120 BN_LIMBS=4 DEFS= bench normscan > $V/l4.build.log 2>&1; echo "l4 build exit=$?"; sh $R/work/slabpipe/skipslab.sh > $V/skipslab.log 2>&1; echo "skipslab exit=$?" ); grep -E "^PASS|^FAIL|all cases|FAILED" $V/skipslab.log | tr -s ' '
( cd $V/cap95/bench && nice -n 5 make -j2 GPU_ARCH=120 PIPE_K=8 BN_LIMBS=12 DEFS= bench > $V/pk8.build.log 2>&1; echo "pk8 build exit=$?" )
A="--pipeline --cofactor --poly $R/oracle/input.job --fb1 $R/oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 120000000: --nq 150"
$V/cap95/bench/bench $A --no-q-pipeline --relations $V/tdskip.single.rel > $V/tdskip.single.out 2>&1
$V/cap95/bench/bench $A --relations $V/tdskip.qp.rel > $V/tdskip.qp.out 2>&1
cmp -s $V/tdskip.single.rel $V/tdskip.qp.rel && echo "  tdskip IDENTICAL ($(wc -l < $V/tdskip.qp.rel)) | $(grep -h 'slab(s) skipped or abandoned' $V/tdskip.qp.out | cut -c1-30)" || echo "  tdskip DIFFER"
echo done
