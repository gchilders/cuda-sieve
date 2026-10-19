#!/bin/sh
# C143 first band [2200000, 2300000), cado-gpu vs cuda-sieve at HEAD + the
# uncommitted findings 110-111 tree, same day. Finding 105/107 protocol:
# CADO through las-glas (16 workers, pinned sievers), cuda-sieve with
# --relations and /usr/bin/time. Two interleaved end-to-end rounds, then
# clean per-kernel arms: CADO one slot (skip on and off) and 8 slots with
# GPU stats; cuda-sieve single-queued at I13 and I14.
R=/home/kylea/code/cuda-sieve; W=$R/work/h2h-20261014; CA=$R/work/cado-ablation
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
POLY=$R/oracle/c143.poly
load() { cut -d' ' -f1 /proc/loadavg; }
cado() {   # name, env assignments...
    name=$1; shift
    d=$W/cado-$name; rm -rf "${d:?}"; mkdir -p "$d"
    env "$@" /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o "$d/time.txt" \
        "$L" -poly "$POLY" -q0 2200000 -q1 2300000 -A 26 \
        -lim0 14000000 -lim1 18000000 -lpb0 30 -lpb1 30 -mfb0 57 -mfb1 58 \
        -lambda0 1.905 -lambda1 1.94 -ncurves0 13 -ncurves1 14 \
        -fb1 "$CA/c143.roots1.gz" -out "$d/relations.gz" -t 16 -stats-stderr \
        > "$d/stdout.txt" 2> "$d/stderr.txt"
    echo "cado $name: $(grep '^# Total' "$d/stderr.txt" | sed 's/^# //') $(cat "$d/time.txt") load=$(load)"
    grep -E '^# GPU (transform|fill|region|tiny|resieve_s|filter|ecm|chunk span):' "$d/stderr.txt" | sed 's/^/    /'
}
cuda() {   # name, extra args...
    name=$1; shift
    d=$W/cuda-$name; rm -rf "${d:?}"; mkdir -p "$d"
    ( cd $R && /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S max_rss_kb=%M' -o "$d/time.txt" \
        ./bench/bench --pipeline --cofactor --poly oracle/c143.job "$@" --region 13 --maxbits 13 \
          --sq-side 1 --qrange 2200000:2299999 --lambda0 1.905 --lambda1 1.94 \
          --relations "$d/relations.dat" --log "$d/run.log" > "$d/output.txt" 2>&1 )
    echo "cuda $name: rel=$(wc -l < "$d/relations.dat") $(cat "$d/time.txt") $(grep -h 'wall clock per q, COMPLETE' "$d/output.txt" | tr -s ' ') load=$(load)"
}
cd $W || exit 1
echo "== round 1"
cado default
cuda i13 --logI 13 --J 8192
cado skip0 GLAS_SKIP_BOUND=0
cuda i14 --logI 14 --J 4096
echo "== round 2"
cuda i14b --logI 14 --J 4096
cado skip0b GLAS_SKIP_BOUND=0
cuda i13b --logI 13 --J 8192
cado defaultb
echo "== per kernel"
cado slot1-skip0 GLAS_GPU_STATS=1 GLAS_GPU_SLOTS=1 GLAS_SKIP_BOUND=0
cado slot1 GLAS_GPU_STATS=1 GLAS_GPU_SLOTS=1
cado stats-skip0 GLAS_GPU_STATS=1 GLAS_SKIP_BOUND=0
cuda i13-single --logI 13 --J 8192 --no-q-pipeline
cuda i14-single --logI 14 --J 4096 --no-q-pipeline
echo done
