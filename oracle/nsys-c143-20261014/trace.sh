#!/bin/sh
# Nsight Systems kernel timelines, C143, equal yield (finding 112's next
# step). cuda-sieve I13 pipelined and single-queued, 800 q from the first
# band's start; CADO skip-off with 8 slots and with 1, q in [2200000,
# 2212000). CUDA trace only: no CPU sampling, no context switches.
R=/home/kylea/code/cuda-sieve; W=$R/work/nsys-20261014; CA=$R/work/cado-ablation
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
NS="nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true"
load() { cut -d' ' -f1 /proc/loadavg; }
cd $R || exit 1
for arm in "qp|" "single|--no-q-pipeline"; do
  name=${arm%%|*}; extra=${arm#*|}
  $NS -o $W/cuda-i13-$name ./bench/bench --pipeline --cofactor --poly oracle/c143.job --logI 13 --J 8192 \
      --region 13 --maxbits 13 --sq-side 1 --qrange 2200000: --nq 800 --lambda0 1.905 --lambda1 1.94 $extra \
      > $W/cuda-i13-$name.out 2>&1
  echo "cuda i13 $name: exit=$? $(grep -h 'wall clock per q, COMPLETE' $W/cuda-i13-$name.out | tr -s ' ') load=$(load)"
done
for arm in "slots8|GLAS_SKIP_BOUND=0" "slot1|GLAS_SKIP_BOUND=0 GLAS_GPU_SLOTS=1"; do
  name=${arm%%|*}; envs=${arm#*|}
  env $envs $NS -o $W/cado-$name "$L" -poly oracle/c143.poly -q0 2200000 -q1 2212000 -A 26 \
      -lim0 14000000 -lim1 18000000 -lpb0 30 -lpb1 30 -mfb0 57 -mfb1 58 \
      -lambda0 1.905 -lambda1 1.94 -ncurves0 13 -ncurves1 14 \
      -fb1 "$CA/c143.roots1.gz" -out $W/cado-$name.rel.gz -t 16 -stats-stderr \
      > $W/cado-$name.out 2> $W/cado-$name.err
  echo "cado $name: exit=$? $(grep -h '^# Total' $W/cado-$name.err) load=$(load)"
done
for f in $W/*.nsys-rep; do nsys export --type sqlite --force-overwrite=true -o ${f%.nsys-rep}.sqlite $f > /dev/null 2>&1; echo "exported $(basename $f): $?"; done
echo done
