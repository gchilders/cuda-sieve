#!/bin/bash
# (1) A=31 comparators at --nq 30, matching a32.tsv's extent -> settles the
#     withdrawn A=32 penalty.  (2) AS276's SHIPPED 33/35 (gap 2) across the
#     band set -> first gap-2 measurement anywhere in finding 98.
set -u; set -o pipefail
B=/home/kylea/code/cuda-sieve/bench; O=/home/kylea/code/cuda-sieve/oracle
W=/home/kylea/code/cuda-sieve/work/lpbsweep; LOGS=$W/logs
TSVDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$W" "$LOGS"
FB="$W/as276.roots1.m16"
[ -f "$FB" ] || "$B/fbgen" --poly "$O/AS276.job" --maxbits 16 --threads 12 --out "$FB"

# timing runs are worthless under host contention (finding 53): wait it out
echo "waiting for load < 2.0 ..." >&2
for i in $(seq 1 180); do
  L=$(awk '{print int($1)}' /proc/loadavg)
  [ "$L" -lt 2 ] && break
  sleep 10
done
echo "load now $(cut -d' ' -f1-3 /proc/loadavg) -- starting" >&2

run() { # tsv tag q nq lpb0 mfb0 lpb1 mfb1 [col8]
  # col8 is the 8th column, which differs between the two TSVs this writes:
  # a31match.tsv declares it 'nq', bands.tsv declares it 'width'.  Passing the
  # nq count into bands.tsv's width column is what the single shared printf
  # used to do.
  local tsv=$1 tag=$2 q=$3 nq=$4 lpb0=$5 mfb0=$6 lpb1=$7 mfb1=$8 col8=${9:-$4}
  local name="${tag}.q${q}.${lpb0}-${lpb1}.m${mfb1}.n${nq}"; local log="$LOGS/$name.log"
  grep -qP "^${tag}\t${q}\t${lpb0}/${lpb1}\t${mfb0}/${mfb1}\t[0-9.]+\\t[0-9.]+\\t[0-9.]+" "$tsv" \
    && { echo "  skip $name (already recorded)" >&2; return; }
  "$B/bench" --pipeline --cofactor --poly "$O/AS276.job" --fb1 "$FB" \
     --logI 16 --J 32768 --maxbits 16 --qrange "${q}000000:" --nq "$nq" \
     --lpb0 "$lpb0" --mfb0 "$mfb0" --lpb "$lpb1" --mfb "$mfb1" \
     --relations "$W/$name.rels" > "$log" 2>&1 || { echo "FAILED $name" >&2; return; }
  local wall cof relq
  wall=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$log"|tail -1)
  cof=$(awk '/--- cofactorisation, cross-q queue ---/,0' "$log"|grep -oP '= device time per q\s+\K[0-9.]+'|tail -1)
  relq=$(grep -oP 'ALL RELATIONS/q\s+\K[0-9.]+' "$log"|tail -1)
  printf '%s\t%s\t%s/%s\t%s/%s\t%s\t%s\t%s\t%s\n' "$tag" "$q" "$lpb0" "$lpb1" "$mfb0" "$mfb1" "$wall" "$cof" "$relq" "$col8" >> "$tsv"
  echo "  $tag q=${q}M ${lpb0}/${lpb1} mfb$mfb1 nq=$nq  wall=$wall cof=$cof rel/q=$relq  load=$(cut -d' ' -f1 /proc/loadavg)" >&2
  rm -f "$W/$name.rels"
}

T1=$TSVDIR/a31match.tsv
[ -f "$T1" ] || printf 'job\tband\tlpb\tmfb\twall_ms_q\tcof_ms_q\trel_q\tnq\n' > "$T1"
echo "### (1) A=31 at nq=30, matched to a32.tsv ###" >&2
run "$T1" as276 80 30 33 64 34 96
run "$T1" as276 80 30 33 64 34 98
run "$T1" as276 80 30 34 66 35 101

T2=$TSVDIR/bands.tsv
[ -f "$T2" ] || printf 'job\tband\tlpb\tmfb\twall_ms_q\tcof_ms_q\trel_q\twidth\n' > "$T2"   # bands.sh may not have run first
echo "### (2) AS276 shipped 33/35 (gap 2), band set ###" >&2
for q in 40 120 240 400; do run "$T2" as276 $q 60 33 64 35 101 -; done
echo "### RERUN DONE ###" >&2
