#!/bin/bash
# Multi-band arm: same configs, three q bands each, so yield decay can be integrated
set -u
set -o pipefail
B=/home/kylea/code/cuda-sieve/bench
O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep
LOGS=$W/logs; mkdir -p "$LOGS"
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # TSVs live beside this script
TSV=$TSVDIR/bands.tsv
[ -f "$TSV" ] || printf 'job\tband\tlpb\tmfb\twall_ms_q\tcof_ms_q\trel_q\twidth\n' > "$TSV"

run() {
  local tag=$1 poly=$2 fb=$3 logI=$4 J=$5 q=$6 nq=$7 lpb0=$8 mfb0=$9 lpb1=${10} mfb1=${11}
  local name="${tag}.q${q}.${lpb0}-${lpb1}"
  local log="$LOGS/${name}.log"
  grep -qP "^${tag}\t${q}\t${lpb0}/${lpb1}\t[^\\t]*\\t[0-9.]+\\t[0-9.]+\\t[0-9.]+" "$TSV" && { echo "skip $name" >&2; return; }
  echo "=== $name ===" >&2
  "$B/bench" --pipeline --cofactor --poly "$poly" --fb1 "$fb" \
      --logI "$logI" --J "$J" --maxbits "$logI" --qrange "${q}000000:" --nq "$nq" \
      --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" \
      --relations "$W/${name}.rels" > "$log" 2>&1
  [ $? -ne 0 ] && { echo "FAILED $name" >&2; return; }
  local wall cof relq width
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log" | tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log" | grep -oP '= device time per q\s+\K[0-9.]+' | tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log" | tail -1)
  width=$(grep -oP 'cofactor width: \K.*' "$log" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$q" "$lpb0/$lpb1" "$mfb0/$mfb1" "$wall" "$cof" "$relq" "$width" >> "$TSV"
  echo "  q=${q}M wall=$wall cof=$cof rel/q=$relq" >&2
  rm -f "$W/${name}.rels"
}
NQ=${NQ:-60}
for q in 20 60 120; do
  run c183 "$O/input.job" "$O/c183.fb1" 15 16384 $q $NQ 31 60 32 92
  run c183 "$O/input.job" "$O/c183.fb1" 15 16384 $q $NQ 32 62 33 95
done
for q in 40 120 240; do
  run c194 "$O/c194.job" "$O/c194.roots1.m16" 16 32768 $q $NQ 32 63 33 95
  run c194 "$O/c194.job" "$O/c194.roots1.m16" 16 32768 $q $NQ 33 65 34 98
done
for q in 40 120 240; do
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 $q $NQ 33 64 34 98
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 $q $NQ 34 66 35 101
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 $q $NQ 35 68 36 104
done
