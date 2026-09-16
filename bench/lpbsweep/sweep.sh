#!/bin/bash
# lpb lockstep sweep: measure wall ms/q, cofactor stage ms/q, relations/q
set -u
set -o pipefail
B=/home/kylea/code/cuda-sieve/bench
O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep
LOGS=$W/logs
mkdir -p "$LOGS"
# TSVs live beside this script (what integrate.py reads); bulk artifacts stay
# in work/, which is gitignored.
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$W" "$LOGS"
# AS276 factor base: 230 MB, gitignored, so bootstrap it if absent
AS276FB="$W/as276.roots1.m16"
[ -f "$AS276FB" ] || "$B/fbgen" --poly "$O/AS276.job" --maxbits 16 --threads 12 --out "$AS276FB"
TSV=$TSVDIR/results.tsv

run() {
  local tag=$1 poly=$2 fb=$3 logI=$4 J=$5 band=$6 nq=$7 lpb0=$8 mfb0=$9 lpb1=${10} mfb1=${11}
  shift 11
  local extra="$*"
  local name="${tag}.${lpb0}-${lpb1}${extra:+.x}"
  local log="$LOGS/${name}.log"
  echo "=== $name : lpb $lpb0/$lpb1  mfb $mfb0/$mfb1  nq=$nq $extra ===" >&2
  /usr/bin/time -f "%e s wall" "$B/bench" --pipeline --cofactor \
      --poly "$poly" --fb1 "$fb" --logI "$logI" --J "$J" --maxbits "$logI" \
      --qrange "$band" --nq "$nq" \
      --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" \
      $extra \
      --relations "$W/${name}.rels" > "$log" 2>&1
  local rc=$?
  # no row on failure: a ragged 4-field row in an 11-column TSV silently
  # fills six Nones in a DictReader. Absent == not-yet-run, which is correct.
  if [ $rc -ne 0 ]; then echo "FAILED rc=$rc (see $log)" >&2; return; fi
  local wall cof relq limbs meth stuck1 rq aq
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log" | tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log" | grep -oP '= device time per q\s+\K[0-9.]+' | tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log" | tail -1)
  rq=$(grep -oP 'rational queue\s+\K[0-9.]+' "$log" | tail -1)
  aq=$(grep -oP 'algebraic queue\s+\K[0-9.]+' "$log" | tail -1)
  limbs=$(grep -oP 'side 0 \d+ limbs.*' "$log" | tail -1)
  meth=$(grep -oP 'cofactor method: \K.*' "$log" | tail -1)
  stuck1=$(grep -oP 'side 1: split / dead / stuck\s+\K.*' "$log" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$lpb0/$lpb1" "$mfb0/$mfb1" "${extra:-default}" "$wall" "$cof" "$relq" "$rq" "$aq" "$limbs" "$stuck1" >> "$TSV"
  echo "  wall=$wall cof=$cof rel/q=$relq  [$limbs] [$meth]" >&2
  rm -f "$W/${name}.rels"      # multi-GB at 100 q; the TSV row is the artifact
}

if [ ! -f "$TSV" ]; then
  printf 'job\tlpb\tmfb\tflags\twall_ms_q\tcof_ms_q\trel_q\trat_q_ms\talg_q_ms\twidth\tside1_split_dead_stuck\n' > "$TSV"
fi

NQ=${NQ:-100}
case "${ONLY:-all}" in
 c183|all)
  run c183 "$O/input.job" "$O/c183.fb1" 15 16384 120000000: $NQ 31 60 32 92
  run c183 "$O/input.job" "$O/c183.fb1" 15 16384 120000000: $NQ 32 62 33 95
 ;;& 
 c194|all)
  run c194 "$O/c194.job" "$O/c194.roots1.m16" 16 32768 120000000: $NQ 32 63 33 95
  run c194 "$O/c194.job" "$O/c194.roots1.m16" 16 32768 120000000: $NQ 33 65 34 98
 ;;&
 as276|all)
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 33 64 34 98
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 34 66 35 101
  run as276 "$O/AS276.job" "$W/as276.roots1.m16" 16 32768 80000023: $NQ 35 68 36 104
 ;;
esac
