#!/usr/bin/env bash
# rentalcof.sh -- the k_cofac register-cliff protocol, for any rented card.
#
# Measured on an RTX 5070 (sm_120) 2026-09-14: Greg's ECM stage-2 shared-denominator
# change (4ae5308) pushed k_cofac<3,ECM,s2> from 126 to 130 registers per thread.
# Charged in 8s that is 136 against 128, and at the pipeline's 256-thread blocks
# 128 is exactly two blocks per SM -- so the production cofactor kernel dropped
# from 2 blocks per SM to 1 (ncu: theoretical occupancy 33.3% -> 16.7%). On the
# 5070 that made the change a NET LOSS (-1.11 ms/q cofactor device time on c183,
# -3.31 on C194). `__launch_bounds__(256, 2)` on k_cofac recovers both halves:
# -10.3% cofactor device time and -2.4% wall against HEAD, relations md5-identical.
#
# What this script asks of another card: does the cliff sit in the same place on
# ITS ptxas target (sm_86 on a 3060, sm_120 on a 5090), and does the bound still
# win when the SM count and register budget differ?
#
#   usage:  bench/rentalcof.sh [OUTDIR] [phase ...]
#           phases: build fb ident cof3 cof4         (default)
#                   ncu apply c194                    (opt-in)
#
# ncu is opt-in because it does not work inside vast.ai containers (no access
# to the hardware counters it needs), and a CUDA image can ship the binary
# anyway. The build phase's ptxas -v table is what locates the cliff; ncu only
# confirms which block limit binds, which the 5070 already did for sm_120.
#
# Needs a git checkout (it exports HEAD with `git archive`), nvcc, and
# bench/rentalcof/*.patch. c183.fb1 is generated if missing. The card must be
# IDLE, and so should the host -- CPU contention moves wall by ~3% (finding 96).
#
# Card time, roughly: build ~10-15 min of CPU (three single-target builds in
# parallel; PAR=0 serialises them), ident ~2 min, cof3 ~9 x NQ q, cof4 ~6 x NQ q.
# At NQ=500 that is ~15 min on a 5090 and ~35-45 min on a 3060.

set -u
cd "$(dirname "$0")" || exit 1   # bench/: the Makefile, the patches, ../oracle
BENCH_DIR=$(pwd)

ALL_PHASES="build fb ident cof3 cof4 ncu apply c194"
OUT=${1:-rentalcof-$(date +%Y%m%d-%H%M%S)}
case " $ALL_PHASES " in
    *" ${1:-} "*)
        echo "first argument is the OUTPUT DIRECTORY, not a phase."
        echo "you probably meant:  bench/rentalcof.sh <outdir> $*"
        exit 2;;
esac
shift 2>/dev/null
PHASES=${*:-build fb ident cof3 cof4}
for _p in $PHASES; do
    case " $ALL_PHASES " in *" $_p "*) ;;
        *) echo "unknown phase '$_p'; known: $ALL_PHASES"; exit 2;; esac
done
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)          # absolute: the variant binaries live under it
echo "logs -> $OUT"

MANIFEST="$OUT/arms.$(date +%H%M%S).$$"
: > "$MANIFEST"

NQ=${NQ:-500}                    # timing bands; NQ=20 for a dry run
DEV=${DEV:-0}
DEVFLAG="--device $DEV"
WD="--watchdog 120 --watchdog-log"
SRC="$OUT/src"
PATCHES="$BENCH_DIR/rentalcof"

want() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }
run()  { local n=$1; shift; echo "== $n"; echo "\$ $*" > "$OUT/$n.log"
         "$@" >> "$OUT/$n.log" 2>&1; local rc=$?
         printf '%s %d\n' "$n" "$rc" >> "$MANIFEST"
         echo "   rc=$rc  ($OUT/$n.log)"; return $rc; }
bin()  { echo "$SRC/$1/bench/bench"; }

# The variants. Each is HEAD plus one patch -- nothing else differs, so an arm's
# difference is that patch's effect and nothing more.
#   head      HEAD as checked out
#   bound     + __launch_bounds__(256, 2) on k_cofac           (cofbound.patch)
#   nos2      + mz_ecm_stage2_pass reverted to 4ae5308^         (nos2.patch)
#   noabs     + |.| Horner chain and fp64 guard removed         (noabs.patch)  PRICING ONLY
#   fastlog2  + -DNORM_FAST_LOG2                                               PRICING ONLY
# noabs and fastlog2 change sieve cells: never quote their relations, only their time.
COF_ARMS="head bound nos2"
APPLY_ARMS="noabs fastlog2"

# ---------------------------------------------------------------- environment
{ date -u; echo; git -C .. rev-parse HEAD; git -C .. status --short; echo
  nvidia-smi; echo; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
  echo; nvcc --version; echo; nproc; free -g; } > "$OUT/00-env.log" 2>&1
echo "== 00-env  ($OUT/00-env.log)"
nvidia-smi -i "$DEV" --query-gpu=name,compute_cap,memory.total,power.limit --format=csv,noheader

# --------------------------------------------------------------------- build
prep() {  # arm patch|-   -> fresh HEAD export with the patch applied
    local arm=$1 p=$2
    rm -rf "$SRC/$arm"; mkdir -p "$SRC/$arm"
    git -C .. archive HEAD bench | tar -x -C "$SRC/$arm" || return 1
    [ "$p" = - ] && return 0
    # --check first: a patch that no longer matches this checkout must refuse,
    # not half-apply and build an arm that silently measures something else.
    # GIT_CEILING_DIRECTORIES is load-bearing. OUTDIR defaults to a path INSIDE the
    # clone, and inside a repository `git apply` resolves bench/... against the
    # repository root, finds it outside the cwd, and SILENTLY SKIPS it: --check
    # passes, the exit code is 0, and nothing is patched. The first 5090 run
    # (2026-09-14) built three copies of HEAD that way. The ceiling stops git
    # discovering the enclosing clone, so the export is patched as a plain tree.
    ( cd "$SRC/$arm" && export GIT_CEILING_DIRECTORIES="$SRC" &&
      git apply --check "$PATCHES/$p" && git apply "$PATCHES/$p" ) || {
        echo "   *** $p does not apply to this checkout's HEAD -- arm $arm unusable"; return 1; }
    # And prove it rather than trust an exit code: every file the patch names must
    # now differ from HEAD's copy.
    local f
    for f in $(sed -n 's#^+++ b/##p' "$PATCHES/$p"); do
        if git -C .. show "HEAD:$f" | cmp -s - "$SRC/$arm/$f"; then
            echo "   *** $p reported success but $f is unchanged -- arm $arm unusable"; return 1
        fi
    done
    echo "   $arm: $p applied ($(sed -n 's#^+++ b/##p' "$PATCHES/$p" | tr '\n' ' '))"
}
build_arm() {  # arm [make args...]; ptxas -v so the register report comes with it
    local arm=$1; shift
    make -C "$SRC/$arm/bench" GPU_ARCH=native NVCC="nvcc -Xptxas -v" "$@" bench
}
if want build; then
    ok=1
    prep head -              || ok=0
    prep bound cofbound.patch || ok=0
    prep nos2 nos2.patch      || ok=0
    if want apply; then
        prep noabs noabs.patch || ok=0
        prep fastlog2 -        || ok=0
    fi
    [ "$ok" = 1 ] || { echo "STOPPING: a variant could not be prepared"; exit 1; }
    arms="$COF_ARMS"; want apply && arms="$arms $APPLY_ARMS"
    # CF_LMAX stays at its default (4): the cof4 phase needs the 4-limb kernel.
    # Parallel by default -- each is one ptxas target. PAR=0 on a small box.
    pids=""
    for a in $arms; do
        extra=""; [ "$a" = fastlog2 ] && extra="DEFS=-DNORM_FAST_LOG2"
        if [ "${PAR:-1}" = 1 ]; then run "01-build-$a" build_arm "$a" $extra & pids="$pids $!"
        else run "01-build-$a" build_arm "$a" $extra || exit 1; fi
    done
    for p in $pids; do wait "$p" || { echo "STOPPING: a build failed (see 01-build-*.log)"; exit 1; }; done
    [ -x ./fbgen ] || run 02-fbgen make fbgen || exit 1

    echo "   k_cofac registers / spill per variant (the cliff is 128 at 256 threads):"
    for a in $arms; do
        # Portable awk only: a stock rental has mawk, which has no 3-argument
        # match(). The mangled name is k_cofacILi<L>ELi<METHOD>ELi<STAGE2>E.
        awk -v arm="$a" '
          /Compiling entry function .*_Z7k_cofac/ {
              q = sprintf("%c", 39)
              t = $0; sub(/.*k_cofacILi/, "", t)
              L = substr(t, 1, 1); M = substr(t, 5, 1); S2 = substr(t, 9, 1)   # "4ELi0ELi0E"
              tg = $0; sub(".*for " q, "", tg); sub(q ".*", "", tg)
              tag = sprintf("%-8s %-7s k_cofac<%s,%s,%s>", arm, tg, L,
                            (M == "1" ? "ECM" : "rho"), (S2 == "1" ? "s2" : "-")); want = 1; next }
          want && /Function properties for/ { n=2; next }
          want && n>0 { sub(/^ptxas info *: */,""); line=line " | " $0; n--
                        if (n==0) { print "     " tag line; line=""; want=0 } }' "$OUT/01-build-$a.log" |
        sed -E 's/ bytes stack frame/B stack/; s/ bytes spill stores/B spillS/; s/ bytes spill loads/B spillL/; s/Used ([0-9]+) registers.*/\1 reg/'
    done
fi

# ------------------------------------------------------------- factor base
if want fb && [ ! -s ../oracle/c183.fb1 ]; then
    run 03-fb ./fbgen --poly ../oracle/input.job --maxbits 15 \
        --threads "$(nproc)" --out ../oracle/c183.fb1 || exit 1
fi

for a in $COF_ARMS; do
    [ -x "$(bin "$a")" ] || { echo "missing $(bin "$a") -- run the build phase"; exit 1; }
done

C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 \
      --logI 15 --J 16384 --maxbits 15 --restart $DEVFLAG"

# ------------------------------------------------------------ identity gate
# 200 q at 130M, all three variants. THE ABORT: if they differ, nothing after
# this is meaningful. The command matches the 5070 runs, so the md5 doubles as a
# cross-card identity check (not a failure if it differs, but worth knowing).
if want ident; then
    for a in $COF_ARMS; do
        run "10-ident-$a" "$(bin "$a")" $C183 --qrange 130000000: --nq 200 \
            --relations "$OUT/id.$a.rels" $WD "$OUT/id.$a.wd"
    done
    run 14-checkrel "$(bin head)" --poly ../oracle/input.job --check-relations "$OUT/id.bound.rels"
    ok=1; ref=""
    while read -r n rc; do
        case "$n" in 1[04]-*) [ "$rc" = 0 ] || { echo "   *** $n FAILED rc=$rc"; ok=0; };; esac
    done < "$MANIFEST"
    for a in $COF_ARMS; do
        f="$OUT/id.$a.rels"; nl=$(wc -l < "$f" 2>/dev/null || echo 0)
        h=$(md5sum < "$f" 2>/dev/null | cut -d' ' -f1)
        [ "${nl:-0}" -ge 1 ] || { echo "   IDENTITY $a ***FAIL*** empty ($nl)"; ok=0; continue; }
        [ -n "$ref" ] || ref=$h
        if [ "$h" = "$ref" ]; then echo "   IDENTITY $a OK  $h  ($nl relations)"
        else echo "   IDENTITY $a ***FAIL***  $h vs $ref"; ok=0; fi
    done
    echo "   5070 reference: a6545ecf84f411a7192dc30514d57f89  (9053 relations incl. tail)"
    [ "$ok" = 1 ] || { echo "STOPPING: variants disagree"; exit 1; }
fi

# Interleaved rounds: odd rounds A B C, even rounds C B A, so a monotonic drift
# (boost-clock decay, heat) lands on every arm equally.
rounds() {  # prefix nrounds extra-args...
    local pfx=$1 nr=$2; shift 2
    local fwd=($COF_ARMS) rev=() a r
    for ((i=${#fwd[@]}-1; i>=0; i--)); do rev+=("${fwd[$i]}"); done
    for ((r=1; r<=nr; r++)); do
        if (( r % 2 )); then order=("${fwd[@]}"); else order=("${rev[@]}"); fi
        for a in "${order[@]}"; do
            run "$pfx-$a-$r" "$(bin "$a")" $C183 --qrange 130000000: --nq "$NQ" "$@" \
                --relations "$OUT/$pfx.$a.$r.rels" $WD "$OUT/$pfx.$a.$r.wd"
        done
    done
}

# ------------------------------------------------ production 3-limb path
if want cof3; then rounds 20-cof3 3; fi

# ------------------------------------ side 1 forced to 4 limbs (AS276 shape)
# The bound makes k_cofac<4,ECM,s2> spill on sm_120 (88 B stores / 144 B loads);
# on the 5070 two blocks per SM still beat the spill by 8.2% on the algebraic
# queue. This is where a different target could turn that around.
if want cof4; then rounds 30-cof4 2 --cof-limbs 4; fi

# ---------------------------------------------------------------- ncu limits
# The mechanism, not just the timing: which block limit binds for the production
# kernel on this card, per variant. Skipped where ncu is not installed.
if want ncu; then
    if command -v ncu >/dev/null 2>&1; then
        # ncu drops its report files in the cwd, so each variant runs from its own
        # directory -- which means absolute oracle paths, not $C183's ../oracle.
        ORA=$(cd "$BENCH_DIR/../oracle" && pwd)
        C183_ABS="--pipeline --cofactor --poly $ORA/input.job --fb1 $ORA/c183.fb1 \
                  --logI 15 --J 16384 --maxbits 15 --restart $DEVFLAG"
        for a in $COF_ARMS; do
            mkdir -p "$OUT/ncu-$a"
            ( cd "$OUT/ncu-$a" && ncu --kernel-name-base demangled \
                -k 'regex:k_cofac<\(int\)3, \(int\)1, \(int\)1>' -s 2 -c 1 --kill no \
                "$(bin "$a")" $C183_ABS --qrange 130000000: --nq 40 ) \
                > "$OUT/40-ncu-$a.log" 2>&1
            echo "   ncu $a:"
            grep -E "Registers Per Thread|Block Limit (Registers|Shared Mem|Warps|SM)|Theoretical Occupancy|Achieved Occupancy|Compute \(SM\) Throughput|# SMs" \
                "$OUT/40-ncu-$a.log" | sed 's/^ */      /'
        done
    else
        echo "== ncu not installed; skipping the block-limit readout"
    fi
fi

# ------------------------------------------------- norm-init pricing (opt-in)
# Standalone apply microbench. NOTE: uses --fb1 c183.fb1, NOT the GGNFS .afb.0
# the 5070 table used (that file is not in git), so compare arms to each other,
# not to the 5070's absolute milliseconds. 5070 ratios: noabs -5.0%, fastlog2
# -4.2%, --norm const -25%.
if want apply; then
    for a in head $APPLY_ARMS; do
        [ -x "$(bin "$a")" ] || { echo "missing $(bin "$a") -- run: rentalcof.sh $OUT build apply"; exit 1; }
    done
    A="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --J 16384 --stage apply --reps 100 $DEVFLAG"
    for r in 1 2 3 4; do
        if (( r % 2 )); then order="head noabs fastlog2 const"; else order="const fastlog2 noabs head"; fi
        for a in $order; do
            if [ "$a" = const ]; then run "50-apply-const-$r" "$(bin head)" $A --norm const
            else run "50-apply-$a-$r" "$(bin "$a")" $A; fi
        done
    done
fi

# ------------------------------------------------------ C194 (opt-in, staged)
# Only if the 205 MB roots file has been copied in by hand; it is not generated
# here. 4 slabs of 2^29 at I16/J32768.
if want c194; then
    if [ -s ../oracle/c194.roots1.m16 ]; then
        C194="--pipeline --cofactor --poly ../oracle/c194.job --fb1 ../oracle/c194.roots1.m16 \
              --logI 16 --J 32768 --restart $DEVFLAG"
        for r in 1 2; do
            if (( r % 2 )); then order="head bound"; else order="bound head"; fi
            for a in $order; do
                run "60-c194-$a-$r" "$(bin "$a")" $C194 --qrange 80000023: --nq $(( NQ/4 > 0 ? NQ/4 : 1 )) \
                    --relations "$OUT/c194.$a.$r.rels" $WD "$OUT/c194.$a.$r.wd"
            done
        done
    else
        echo "== c194: ../oracle/c194.roots1.m16 not staged; skipping"
    fi
fi

# ------------------------------------------------------------------- summary
echo
echo "=============================== SUMMARY ==============================="
nvidia-smi -i "$DEV" --query-gpu=name,compute_cap --format=csv,noheader
# Per-arm rows, manifest-scoped: a killed arm is reported, never skipped.
TAB="$OUT/.summary.$$"; : > "$TAB"
while read -r n rc; do
    case "$n" in 2*-cof3-*|3*-cof4-*|60-c194-*) ;; *) continue;; esac
    f="$OUT/$n.log"
    printf '%-22s ' "$n"
    if [ "$rc" != 0 ] || ! grep -q "band of" "$f" 2>/dev/null; then
        printf '*** NO RESULT (rc=%s%s)\n' "$rc" "$([ "$rc" = 4 ] && echo ', WATCHDOG KILL')"; continue
    fi
    awk -v nm="$n" -v tab="$TAB" '
        function v(  i){for(i=1;i<=NF;i++) if($i=="ms") return $(i-1); return ""}
        /^  wall clock per q  /         {w=v()}
        /^  wall clock per q, COMPLETE/ {W=v()}
        /^  rational queue/             {r=v()}
        /^  algebraic queue/            {a=v()}
        /^    = device time per q/      {d=v()}
        /^  GPU-accounted . wall/       {g=$NF}
        END{printf "wall %8s cmplt %8s  rat %7s  alg %7s  cofdev %7s  acc %s\n",w,W,r,a,d,g
            grp=nm; sub(/-[0-9]+$/,"",grp); printf "%s %s %s %s %s %s\n",grp,w,W,r,a,d >> tab}' "$f"
done < "$MANIFEST"

echo
echo "means by variant (the comparison that matters is alg and cofdev):"
awk '{n[$1]++; w[$1]+=$2; W[$1]+=$3; r[$1]+=$4; a[$1]+=$5; d[$1]+=$6}
     END{for(g in n) printf "  %-18s n=%d  wall %8.2f  cmplt %8.2f  rat %7.3f  alg %7.3f  cofdev %7.3f\n",
                     g,n[g],w[g]/n[g],W[g]/n[g],r[g]/n[g],a[g]/n[g],d[g]/n[g]}' "$TAB" | sort
echo "  5070 (sm_120), NQ 200, 4 rounds: cof3 alg head 12.83 / nos2 11.78 / bound 11.25;"
echo "                                   cofdev 15.38 / 14.32 / 13.79"
rm -f "$TAB"

if want apply; then
    echo
    echo "apply microbench, mean of 4 (ms):"
    for a in head noabs fastlog2 const; do
        printf '  %-9s' "$a"
        grep -h "apply (init+add+scan)" "$OUT"/50-apply-$a-*.log 2>/dev/null |
            awk '{s+=$(NF-1); n++} END{if(n) printf " %7.3f  (n=%d)\n", s/n, n; else print " no result"}'
    done
fi

echo
echo "relation md5 -- every arm of one geometry must agree (noabs/fastlog2 excluded):"
for pfx in 20-cof3 30-cof4 c194; do
    set --
    while read -r n _; do
        case "$n" in
            "$pfx"-*)   set -- "$@" "$OUT/$pfx.$(echo "$n" | cut -d- -f3).$(echo "$n" | cut -d- -f4).rels";;
            60-c194-*)  [ "$pfx" = c194 ] && set -- "$@" "$OUT/c194.$(echo "$n" | cut -d- -f3).$(echo "$n" | cut -d- -f4).rels";;
        esac
    done < "$MANIFEST"
    [ $# -gt 0 ] && [ -e "$1" ] || continue
    ref=""; bad=0
    for f in "$@"; do
        h=$(md5sum < "$f" | cut -c1-12); nl=$(wc -l < "$f")
        [ -n "$ref" ] || ref=$h
        if [ "$h" = "$ref" ]; then mark="   "; else mark="***"; bad=1; fi
        printf '  %s %-26s %8s  %s\n' "$mark" "$(basename "$f")" "$nl" "$h"
    done
    [ "$bad" = 0 ] || echo "     *** VARIANTS DISAGREE for $pfx -- not usable"
done
echo "======================================================================="
