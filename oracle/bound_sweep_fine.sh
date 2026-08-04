#!/bin/bash
# Refine the survivor-bound cliff at 1-unit granularity.
# bound1 = (unsigned char)(scale1 * lambda1 * 32) + 1, scale1 = 1/log2(1.722273) = 1.274995
#   => bound1 = floor(40.7998 * lambda1) + 1
# The 0.1-step sweep moved bound1 by 4 and never sampled 128,129,130,132,133,134.
set -u
LAS=~/cado-nfs/build/DESKTOP-3J4UC68/sieve/las
O=~/code/cuda-sieve/oracle
OUT=$CLAUDE_JOB_DIR/tmp/sweepfine
mkdir -p "$OUT"
for L1 in 3.28 3.25 3.22 3.17 3.15 3.12; do
  f="$OUT/f_${L1}.log"
  [ -s "$f" ] && continue
  nice -n 19 "$LAS" -poly $O/c183.poly -fb1 $O/c183.fb1 \
    -lim0 67100000 -lim1 134200000 -lpb0 31 -lpb1 32 \
    -mfb0 60 -mfb1 92 -lambda0 2.35 -lambda1 "$L1" \
    -powlim0 32767 -powlim1 32767 \
    -A 29 -sqside 1 -q0 120000000 -q1 120001000 -adjust-strategy 0 \
    -t 8 -v > "$f" 2>&1
  printf 'l1=%s %s reports=%s\n' "$L1" \
    "$(grep -m1 'Side 1:' "$f" | grep -o 'bound=[0-9]*')" \
    "$(grep -m1 '^# Total [0-9]* reports' "$f" | awk '{print $3}')"
done
