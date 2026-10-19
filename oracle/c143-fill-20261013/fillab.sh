#!/bin/sh
# Interleaved A/B of the fill/resieve row tier and the fill gate on C143.
# NQ special-q per run, ROUNDS rounds; each round runs every arm once, in an
# order rotated per round so drift does not land on one arm. One line per run.
cd /home/kylea/code/cuda-sieve/bench || exit 1
NQ=${NQ:-1000}
ROUNDS=${ROUNDS:-3}
OUT=${OUT:-/home/kylea/code/cuda-sieve/work/fillab/runs}
mkdir -p "$OUT"
C143='--poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000:'
for geo in 'i13|--logI 13 --J 8192|64' 'i14|--logI 14 --J 4096|32'; do
    g=${geo%%|*}; rest=${geo#*|}; gargs=${rest%%|*}; b2=${rest#*|}
    ARMS="base|
rows2|--fill-rows 2
rows4|--fill-rows 4
rows8|--fill-rows 8
rows4b2x|--fill-rows 4 --fill-row-bands $b2
rows4rs|--fill-rows 4 --resieve-rows
gate3|--fill-gate
rows4gate3|--fill-rows 4 --fill-gate
rows4gate0|--fill-rows 4 --fill-gate --fill-gate-shift 0
rows4rsgate3|--fill-rows 4 --resieve-rows --fill-gate"
    r=0
    while [ $r -lt "$ROUNDS" ]; do
        echo "$ARMS" | awk -v r=$r '{a[NR]=$0} END{for(i=0;i<NR;i++) print a[(i+r)%NR+1]}' |
        while IFS='|' read -r arm aargs; do
            f="$OUT/$g.$arm.$r.out"
            # shellcheck disable=SC2086
            ./bench --pipeline --cofactor $C143 $gargs --nq "$NQ" $aargs > "$f" 2>&1
            fill=$(awk '/^ +fill +[0-9]/{print $2; exit}' "$f")
            apply=$(awk '/^ +apply +[0-9]/{print $2; exit}' "$f")
            rs=$(awk '/resieve \+ scatter, both sides/{print $6; exit}' "$f")
            td=$(awk '/TD \+ classify, wall/{print $5; exit}' "$f")
            wall=$(awk '/wall clock per q, COMPLETE/{print $6; exit}' "$f")
            relq=$(awk '/ALL RELATIONS\/q/{print $3; exit}' "$f")
            echo "$g $arm $r fill=$fill apply=$apply resieve=$rs td=$td wall=$wall relq=$relq"
        done
        r=$((r + 1))
    done
done
