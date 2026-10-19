#!/bin/sh
# Clean-host confirmation of the proposed defaults: per job, the arms that
# decide that job's default, interleaved over ROUNDS rotated rounds.
cd /home/kylea/code/cuda-sieve/bench || exit 1
ROUNDS=${ROUNDS:-3}
OUT=${OUT:-/home/kylea/code/cuda-sieve/work/qconc/runs-qpipe}
mkdir -p "$OUT"
C143='--poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --nq 1500'
run_job() {   # name, job args, arms (newline separated name|args)
    g=$1; jargs=$2; arms=$3
    r=0
    while [ $r -lt "$ROUNDS" ]; do
        echo "$arms" | awk -v r=$r '{a[NR]=$0} END{for(i=0;i<NR;i++) print a[(i+r)%NR+1]}' |
        while IFS='|' read -r arm aargs; do
            f="$OUT/$g.$arm.$r.out"
            # shellcheck disable=SC2086
            ./bench --pipeline --cofactor $jargs $aargs > "$f" 2>&1
            fill=$(awk '/^ +fill +[0-9]/{print $2; exit}' "$f")
            apply=$(awk '/^ +apply +[0-9]/{print $2; exit}' "$f")
            rs=$(awk '/resieve \+ scatter, both sides/{print $6; exit}' "$f")
            td=$(awk '/TD \+ classify, wall/{print $5; exit}' "$f")
            wall=$(awk '/wall clock per q, COMPLETE/{print $6; exit}' "$f")
            relq=$(awk '/ALL RELATIONS\/q/{print $3; exit}' "$f")
            load=$(cut -d' ' -f1 /proc/loadavg)
            echo "$g $arm $r fill=$fill apply=$apply resieve=$rs td=$td wall=$wall relq=$relq load=$load"
        done
        r=$((r + 1))
    done
}
run_job i13 "$C143 --logI 13 --J 8192" 'base|
qpipe|--q-pipeline'
run_job i14 "$C143 --logI 14 --J 4096" 'base|
qpipe|--q-pipeline'
run_job c183 '--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 200' 'base|
qpipe|--q-pipeline'
echo done
