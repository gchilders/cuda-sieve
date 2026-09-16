#!/bin/bash
# Validate the AS276 recommendation at NFS@Home's real geometry: A=32 (logI 17, J 32768, j-slabbed)
set -u
set -o pipefail
B=/home/kylea/code/cuda-sieve/bench; O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep; LOGS=$W/logs
# TSVs live beside this script (what integrate.py reads); bulk artifacts stay
# in work/, which is gitignored.
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$W" "$LOGS"
TSV=$TSVDIR/a32.tsv
[ -f "$TSV" ] || printf 'geom\tlpb\tmfb\twall_ms_q\tcof_ms_q\trel_q\twidth\tslabs\n' > "$TSV"
FB=$W/as276.roots1.m17
[ -f "$FB" ] || { echo "### fbgen maxbits 17 ###" >&2; "$B/fbgen" --poly "$O/AS276.job" --maxbits 17 --threads 12 --out "$FB" >&2; }
run() {
  local lpb0=$1 mfb0=$2 lpb1=$3 mfb1=$4 nq=$5
  local name="as276.A32.${lpb0}-${lpb1}.m${mfb1}"; local log="$LOGS/${name}.log"
  echo "=== $name ===" >&2
  "$B/bench" --pipeline --cofactor --poly "$O/AS276.job" --fb1 "$FB" \
     --logI 17 --J 32768 --maxbits 17 --qrange 80000023: --nq "$nq" \
     --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" \
     --relations "$W/${name}.rels" > "$log" 2>&1
  [ $? -ne 0 ] && { echo "FAILED $name (see $log)" >&2; return; }
  local wall cof relq width slabs
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log"|tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log"|grep -oP '= device time per q\s+\K[0-9.]+'|tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log"|tail -1)
  width=$(grep -oP 'cofactor width: \K.*' "$log"|tail -1)
  slabs=$(grep -oiP '\d+ slabs?' "$log"|head -1)
  printf 'A32\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lpb0/$lpb1" "$mfb0/$mfb1" "$wall" "$cof" "$relq" "$width" "$slabs" >> "$TSV"
  echo "  A32 wall=$wall cof=$cof rel/q=$relq [$slabs]" >&2
  rm -f "$W/${name}.rels"
}
NQ=${NQ:-30}   # 30 matches the published a32.tsv; raise it to match a 60-q A=31 arm
run 33 64 34 96 $NQ
run 33 64 34 98 $NQ
run 34 66 35 101 $NQ
