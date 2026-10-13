#!/bin/sh
# Finding 105's first-band protocol, same day: old fill (all three off) vs the
# new defaults, at I13 and I14, with --relations and /usr/bin/time.
cd /home/kylea/code/cuda-sieve || exit 1
W=work/c143-firstband
for geo in 'i13|--logI 13 --J 8192' 'i14|--logI 14 --J 4096'; do
    g=${geo%%|*}; gargs=${geo#*|}
    for arm in 'off|--fill-rows 0 --no-fill-gate' 'new|'; do
        a=${arm%%|*}; aargs=${arm#*|}
        d=$W/$g-$a; rm -rf "$d"; mkdir -p "$d"
        # shellcheck disable=SC2086
        /usr/bin/time -f 'elapsed_s=%e user_s=%U system_s=%S max_rss_kb=%M' -o "$d/wall-time.txt" \
          ./bench/bench --pipeline --cofactor --poly oracle/c143.job $gargs --region 13 --maxbits 13 \
            --sq-side 1 --qrange 2200000:2299999 --lambda0 1.905 --lambda1 1.94 $aargs \
            --relations "$d/relations.dat" --log "$d/run.log" > "$d/output.txt" 2>&1
        echo "$g $a $(cat $d/wall-time.txt) rel=$(wc -l < $d/relations.dat) $(grep 'wall clock per q, COMPLETE' $d/output.txt | tr -s ' ') load=$(cut -d' ' -f1 /proc/loadavg)"
    done
done
