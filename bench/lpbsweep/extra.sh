#!/bin/bash
# Arm 2: saturation checks (did the default cofactor schedule under-yield?)
# Arm 3: mfb capped at 96 -> stays 3-limb at lpb 34/35
set -u
set -o pipefail
B=/home/kylea/code/cuda-sieve/bench; O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep; LOGS=$W/logs; mkdir -p "$LOGS"
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # TSVs live beside this script
TSV=$TSVDIR/extra.tsv
[ -f "$TSV" ] || printf 'arm\tjob\tlpb\tmfb\tflags\twall_ms_q\tcof_ms_q\trel_q\twidth\n' > "$TSV"
run() {
  local arm=$1 tag=$2 poly=$3 fb=$4 logI=$5 J=$6 band=$7 nq=$8 lpb0=$9 mfb0=${10} lpb1=${11} mfb1=${12}; shift 12
  local extra="$*"
  local name="${arm}.${tag}.${lpb0}-${lpb1}.m${mfb1}"
  local log="$LOGS/${name}.log"
  echo "=== $name  $extra ===" >&2
  "$B/bench" --pipeline --cofactor --poly "$poly" --fb1 "$fb" --logI "$logI" --J "$J" --maxbits "$logI" \
     --qrange "$band" --nq "$nq" --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" $extra \
     --relations "$W/${name}.rels" > "$log" 2>&1
  [ $? -ne 0 ] && { echo "FAILED $name" >&2; return; }
  local wall cof relq width
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log"|tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log"|grep -oP '= device time per q\s+\K[0-9.]+'|tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log"|tail -1)
  width=$(grep -oP 'cofactor width: \K.*' "$log"|tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$tag" "$lpb0/$lpb1" "$mfb0/$mfb1" "${extra:-default}" "$wall" "$cof" "$relq" "$width" >> "$TSV"
  echo "  wall=$wall cof=$cof rel/q=$relq" >&2
  rm -f "$W/${name}.rels"
}
NQ=${NQ:-60}
SAT="--ecm-curves 24 --cof-rounds 6"
# Arm 2 -- saturation
run sat c183  "$O/input.job" "$O/c183.fb1" 15 16384 120000000: $NQ 31 60 32 92  $SAT
run sat c183  "$O/input.job" "$O/c183.fb1" 15 16384 120000000: $NQ 32 62 33 95  $SAT
run sat c194  "$O/c194.job" "$O/c194.roots1.m16" 16 32768 120000000: $NQ 32 63 33 95 $SAT
run sat c194  "$O/c194.job" "$O/c194.roots1.m16" 16 32768 120000000: $NQ 33 65 34 98 $SAT
run sat as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 33 64 34 98  $SAT
run sat as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 34 66 35 101 $SAT
run sat as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 35 68 36 104 $SAT
# Arm 3 -- mfb capped at 96, stays 3-limb
run cap96 c194  "$O/c194.job" "$O/c194.roots1.m16" 16 32768 120000000: $NQ 33 65 34 96
run cap96 as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 33 64 34 96
run cap96 as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 34 66 35 96
