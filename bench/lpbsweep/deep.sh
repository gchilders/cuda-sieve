#!/bin/bash
# A 400M band for AS276: halves the extrapolation behind the headline recommendation
set -u
set -o pipefail
B=/home/kylea/code/cuda-sieve/bench; O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep; LOGS=$W/logs
# TSVs live beside this script (what integrate.py reads); bulk artifacts stay
# in work/, which is gitignored.
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$W" "$LOGS"
# AS276 factor base: 230 MB, gitignored, so bootstrap it if absent
AS276FB="$W/as276.roots1.m16"
[ -f "$AS276FB" ] || "$B/fbgen" --poly "$O/AS276.job" --maxbits 16 --threads 12 --out "$AS276FB"
TSV=$TSVDIR/bands.tsv
[ -f "$TSV" ] || printf 'job\tband\tlpb\tmfb\twall_ms_q\tcof_ms_q\trel_q\twidth\n' > "$TSV"   # bands.sh may not have run first
run() {
  local tag=$1 lpb0=$2 mfb0=$3 lpb1=$4 mfb1=$5 nq=$6 q=400
  local name="as276.q${q}.${lpb0}-${lpb1}.m${mfb1}"; local log="$LOGS/${name}.log"
  grep -qP "^${tag}\t${q}\t${lpb0}/${lpb1}\t${mfb0}/${mfb1}\t[0-9.]+\\t[0-9.]+\\t[0-9.]+" "$TSV" \
    && { echo "skip $name (already in bands.tsv)" >&2; return; }
  echo "=== $name ===" >&2
  "$B/bench" --pipeline --cofactor --poly "$O/AS276.job" --fb1 "$W/as276.roots1.m16" \
     --logI 16 --J 32768 --maxbits 16 --qrange "${q}000000:" --nq "$nq" \
     --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" \
     --relations "$W/${name}.rels" > "$log" 2>&1
  [ $? -ne 0 ] && { echo "FAILED $name" >&2; return; }
  local wall cof relq
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log"|tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log"|grep -oP '= device time per q\s+\K[0-9.]+'|tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log"|tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\n' "$tag" "$q" "$lpb0/$lpb1" "$mfb0/$mfb1" "$wall" "$cof" "$relq" >> "$TSV"
  echo "  q=${q}M wall=$wall cof=$cof rel/q=$relq" >&2
  rm -f "$W/${name}.rels"
}
NQ=${NQ:-60}
run as276    33 64 34 98  $NQ
run as276    34 66 35 101 $NQ
run as276    35 68 36 104 $NQ
run as276cap 33 64 34 96  $NQ
run as276cap 34 66 35 96  $NQ
