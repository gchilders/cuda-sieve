#!/bin/sh
# Upper bound on what filling GPU idle gaps is worth: two disjoint 1000-q
# ranges run back to back in one shell, then as two concurrent processes
# (time-sliced on the GPU, no kernel overlap across processes). Interleaved.
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/qconc
C='--pipeline --cofactor --poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --nq 3000'
now() { date +%s.%N; }
for r in 0 1 2; do
  for geo in 'i13|--logI 13 --J 8192' 'i14|--logI 14 --J 4096'; do
    g=${geo%%|*}; ga=${geo#*|}
    seq() {
      rm -f $W/*.rel*; t0=$(now)
      ./bench $C $ga --qrange 2200000: --relations $W/a.rel > $W/$g.seqA.$r.out 2>&1
      ./bench $C $ga --qrange 2700000: --relations $W/b.rel > $W/$g.seqB.$r.out 2>&1
      ts=$(echo "$(now) - $t0" | bc)
    }
    par() {
      rm -f $W/*.rel*; t0=$(now)
      ./bench $C $ga --qrange 2200000: --relations $W/a.rel > $W/$g.parA.$r.out 2>&1 &
      pa=$!
      ./bench $C $ga --qrange 2700000: --relations $W/b.rel > $W/$g.parB.$r.out 2>&1 &
      pb=$!
      wait $pa; wait $pb
      tp=$(echo "$(now) - $t0" | bc)
    }
    if [ $((r % 2)) -eq 0 ]; then seq; par; else par; seq; fi
    echo "$g round $r: sequential $ts s, concurrent $tp s, ratio $(echo "scale=3; $tp / $ts" | bc), load $(cut -d' ' -f1 /proc/loadavg)"
  done
done
