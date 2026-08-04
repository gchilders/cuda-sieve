#!/bin/bash
# Survivor-bound sweep, downward. CPU only; outputs relation + survivor counts.
# bound = (unsigned char)(scale * lambda * lpb) + 1 ; scale is lambda-independent,
# so lambda1 steps of 0.1 move the side-1 bound by ~4 units.
# Run from anywhere. ~5 min quiet at -t 16 per sweep point over 31 q.
set -u
LAS=~/cado-nfs/build/DESKTOP-3J4UC68/sieve/las
O=~/code/cuda-sieve/oracle
OUT=${1:-$PWD/sweep}
mkdir -p "$OUT"
for L1 in 3.5 3.4 3.3 3.2; do
  for L0 in 2.35; do
    f="$OUT/sweep_l1_${L1}_l0_${L0}.log"
    [ -s "$f" ] && { echo "skip $f"; continue; }
    echo "=== lambda1=$L1 lambda0=$L0 ==="
    "$LAS" -poly $O/c183.poly -fb1 $O/c183.fb1 \
      -lim0 67100000 -lim1 134200000 -lpb0 31 -lpb1 32 \
      -mfb0 60 -mfb1 92 -lambda0 "$L0" -lambda1 "$L1" \
      -powlim0 32767 -powlim1 32767 \
      -A 29 -sqside 1 -q0 120000000 -q1 120001000 -adjust-strategy 0 \
      -t 16 -v > "$f" 2>&1
    printf '  bound1=%s  after_sieve=%s  reports=%s\n' \
      "$(grep -m1 'Side 1:' "$f" | grep -o 'bound=[0-9]*')" \
      "$(grep 'survivors after_sieve' "$f" | tail -1 | awk '{print $4}')" \
      "$(grep -m1 '^# Total [0-9]* reports' "$f" | awk '{print $3}')"
  done
done
echo; echo "=== summary ==="
for f in "$OUT"/sweep_l1_*.log; do
  printf '%-34s %-12s survivors=%-10s relations=%s\n' "$(basename "$f")" \
    "$(grep -m1 'Side 1:' "$f" | grep -o 'bound=[0-9]*')" \
    "$(grep 'survivors after_sieve' "$f" | tail -1 | awk '{print $4}')" \
    "$(grep -m1 '^# Total [0-9]* reports' "$f" | awk '{print $3}')"
done
