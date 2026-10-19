#!/bin/sh
# CADO glas ablations on the C143 first band [2200000, 2300000), through the
# campaign's launcher (16 workers, pinned sievers). One directory per arm:
# relations.gz, stderr, /usr/bin/time, and a mid-run per-thread CPU snapshot.
W=/home/kylea/code/cuda-sieve/work/cado-ablation
L=$HOME/code/cado-nfs-public/gpu-siever/scripts/las-glas
POLY=/home/kylea/code/cuda-sieve/oracle/c143.poly
arm() {   # name, env assignments...
    name=$1; shift
    d=$W/$name; rm -rf "$d"; mkdir -p "$d"
    env "$@" /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S' -o "$d/time.txt" \
        "$L" -poly "$POLY" -q0 2200000 -q1 2300000 -A 26 \
        -lim0 14000000 -lim1 18000000 -lpb0 30 -lpb1 30 -mfb0 57 -mfb1 58 \
        -lambda0 1.905 -lambda1 1.94 -ncurves0 13 -ncurves1 14 \
        -fb1 "$W/c143.roots1.gz" -out "$d/relations.gz" -t 16 -stats-stderr \
        > "$d/stdout.txt" 2> "$d/stderr.txt" &
    pid=$!
    sleep 15
    g=$(pgrep -P "$pid" -x glas-gpu || pgrep -n -x glas-gpu)
    [ -n "$g" ] && ps -L -o tid,pcpu,stat,comm -p "$g" > "$d/threads.txt" 2>&1
    wait "$pid"
    echo "$name: $(grep '^# Total' "$d/stderr.txt") $(cat "$d/time.txt") $(grep 'Cofactor candidates' "$d/stderr.txt")"
}
arm default
arm stats       GLAS_GPU_STATS=1
arm stats-slot1 GLAS_GPU_STATS=1 GLAS_GPU_SLOTS=1
arm skip0       GLAS_SKIP_BOUND=0
arm nogate      GLAS_FILL_GATE=0
arm threads2    GLAS_THREADS=2
echo done
