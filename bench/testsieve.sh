#!/bin/bash
# testsieve.sh -- test-sieve a job at several q, project the whole run from the
# yield curve, and compare geometries.
#
# The GGNFS equivalent (`test_sieve.sh` in a ~/code/test-sieve tree) can only
# sweep q, because its siever is compiled per-I: 14e, 15e and 16e are three
# different binaries and J is fixed at I/2. Ours takes I and J at run time, so
# geometry is a sweepable axis here, and so are the factor-base bounds.
#
# UNITS, STATED ONCE. `bench` processes **(q, rho) pairs** -- one special-q and
# one root of it. Short q intervals contain a noisy number of those pairs, so
# raw relation counts (and projections made directly from them) are misleading.
# As in ~/code/test-sieve, every sample is normalized to the expected number of
# special-q objects in its interval: width / ln(q0). The yield projection, time,
# energy, and target crossing all use those normalized samples.

set -u
usage() {
    cat <<'EOF'
usage: ./testsieve.sh --poly JOB.job [--fb1 FB] [options]
       ./testsieve.sh                       # prompt for parameters

  --poly PATH        the job file (required)
  --fb1 PATH         custom caller-managed algebraic factor base
                     [omit for checked/rebuilt fbase.mLOGI cache]
  --fb-threads N     threads when rebuilding the default factor base [all CPUs]
  --qmin N           band start                          [20000000]
  --qmax N           band end, for the projection        [10 x qmin]
  --points N         sample points across the band       [5]
  --width N          q-interval sieved at each point     [2000]
  --target-rels N    also report where the target is met [job-sized guess off]
  --side a|r         sieve algebraic or rational side              [a]
  --sq-side 1|0      numeric alias for --side (bench convention)
  --geom "logI,J"    geometry to test; repeat for several   [15,16384]
  --rlim N/--alim N  override the job's factor-base bounds
  --extra "FLAGS"    passed through to bench verbatim
  --keep             keep the relation files and logs
  -i, --interactive  prompt for parameters (no arguments does this too)

Each point sieves a WIDTH-wide q interval, so wider costs more and measures
better; the default holds roughly a hundred (q, rho) pairs at q ~ 2e7, which
is a few seconds a point. Startup (factor-base load) is excluded from every
timing: it is a one-off of seconds against a run of days.

Yield normalization follows the reference test sieve:
  expected pairs = WIDTH / ln(q0)
  n-yield        = raw relations * expected pairs / observed pairs
EOF
}

POLY=
FB=fbase
FB_MANAGED=1
FB_THREADS=${FB_THREADS:-}
QMIN=20000000
QMAX=
POINTS=5
WIDTH=2000
TARGET=
SQ_SIDE=1
RLIM=
ALIM=
EXTRA=
KEEP=0
GEOMS=()
INTERACTIVE=0
# Only prompt when there is a human to prompt. With stdin closed or redirected
# -- a Makefile, cron, CI -- every `read` returns EOF instantly, so this used to
# accept ALL the built-in defaults in silence and launch real sieve runs against
# whatever input.job/fbase happened to be lying around. No arguments and no tty
# is a usage error, which is what it was before the interactive mode existed.
if [ $# -eq 0 ]; then
    if [ -t 0 ]; then INTERACTIVE=1; else usage; exit 2; fi
fi
while [ $# -gt 0 ]; do
    case "$1" in
        --poly) POLY=$2; shift 2 ;;
        --fb1) FB=$2; FB_MANAGED=0; shift 2 ;;
        --fb-threads) FB_THREADS=$2; shift 2 ;;
        --qmin) QMIN=$2; shift 2 ;;
        --qmax) QMAX=$2; shift 2 ;;
        --points) POINTS=$2; shift 2 ;;
        --width) WIDTH=$2; shift 2 ;;
        --target-rels) TARGET=$2; shift 2 ;;
        --side)
            case "$2" in
                a|A|1|algebraic) SQ_SIDE=1 ;;
                r|R|0|rational) SQ_SIDE=0 ;;
                *) echo "side must be algebraic/a or rational/r" >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --sq-side) SQ_SIDE=$2; shift 2 ;;
        --geom) GEOMS+=("$2"); shift 2 ;;
        --rlim) RLIM=$2; shift 2 ;;
        --alim) ALIM=$2; shift 2 ;;
        --extra) EXTRA=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -i|--interactive) INTERACTIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage; exit 2 ;;
    esac
done

prompt_value() {
    local label=$1 default=$2 answer
    read -r -p "$label [$default]: " answer
    printf '%s' "${answer:-$default}"
}

default_fb_threads() {
    local n
    n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
    [[ "$n" =~ ^[0-9]+$ ]] || n=1
    [ "$n" -le 256 ] || n=256
    printf '%s' "$n"
}

canonical_decimal() {
    local n=$1
    while [ "${#n}" -gt 1 ] && [ "${n#0}" != "$n" ]; do n=${n#0}; done
    printf '%s' "$n"
}

if [ "$INTERACTIVE" = 1 ]; then
    echo "Interactive test-sieve setup (press Enter to accept a default)."
    POLY=$(prompt_value "Job file" "${POLY:-input.job}")
    read -r -p "Algebraic factor base [$FB]: " answer
    if [ -n "$answer" ]; then
        FB=$answer
        # Typing a path is an explicit choice and must not authorize overwrite.
        FB_MANAGED=0
    fi
    if [ "$FB_MANAGED" = 1 ]; then
        [ -n "$FB_THREADS" ] || FB_THREADS=$(default_fb_threads)
        FB_THREADS=$(prompt_value "Factor-base build threads" "$FB_THREADS")
    fi
    QMIN=$(prompt_value "Band start q" "$QMIN")
    # Same reason as the validation block below: this $(( )) must not see an
    # unvalidated answer.
    if ! [[ "$QMIN" =~ ^[0-9]+$ ]] || [ "$QMIN" -le 1 ]; then
        echo "qmin must be an integer greater than 1" >&2; exit 2
    fi
    QMIN=$(canonical_decimal "$QMIN")
    qmax_default=${QMAX:-$((QMIN * 10))}
    QMAX=$(prompt_value "Band end q" "$qmax_default")
    POINTS=$(prompt_value "Sample points" "$POINTS")
    WIDTH=$(prompt_value "q-interval width per point" "$WIDTH")

    read -r -p "Target relations [none]: " answer
    TARGET=${answer:-$TARGET}

    side_default=a
    [ "$SQ_SIDE" = 0 ] && side_default=r
    read -r -p "Sieve algebraic (a) or rational (r) side? [${side_default}]: " answer
    case "${answer:-$side_default}" in
        a|A|1|algebraic) SQ_SIDE=1 ;;
        r|R|0|rational) SQ_SIDE=0 ;;
        *) echo "side must be algebraic/a or rational/r" >&2; exit 2 ;;
    esac

    geom_default=${GEOMS[*]:-15,16384}
    read -r -p "Geometries, space separated [$geom_default]: " answer
    read -r -a GEOMS <<< "${answer:-$geom_default}"

    read -r -p "Rational factor-base bound [job]: " answer
    RLIM=${answer:-$RLIM}
    read -r -p "Algebraic factor-base bound [job]: " answer
    ALIM=${answer:-$ALIM}
    read -r -p "Extra bench flags [none]: " answer
    EXTRA=${answer:-$EXTRA}
    read -r -p "Keep relation files and logs? [y/N]: " answer
    case "$answer" in y|Y|yes|YES) KEEP=1 ;; esac
    echo
fi

if [ -z "$POLY" ] || [ -z "$FB" ]; then usage; exit 2; fi
[ ${#GEOMS[@]} -gt 0 ] || GEOMS=("15,16384")
# Validate BEFORE the $(( )) below. Bash arithmetic expands array subscripts,
# so `--qmin 'a[$(cmd)]'` executes cmd inside $(( )) -- and the ^[0-9]+$ test
# used to run afterwards, which is too late to matter.
if ! [[ "$QMIN" =~ ^[0-9]+$ ]] || [ "$QMIN" -le 1 ]; then
    echo "qmin must be an integer greater than 1" >&2; exit 2
fi
QMIN=$(canonical_decimal "$QMIN")
if [ -n "$QMAX" ] && ! [[ "$QMAX" =~ ^[0-9]+$ ]]; then
    echo "qmax must be an integer" >&2; exit 2
fi
[ -n "$QMAX" ] || QMAX=$((QMIN * 10))
QMAX=$(canonical_decimal "$QMAX")
if [ "$QMAX" -le "$QMIN" ]; then
    echo "qmax must be greater than qmin" >&2; exit 2
fi
if ! [[ "$POINTS" =~ ^[0-9]+$ && "$WIDTH" =~ ^[0-9]+$ ]] ||
   [ "$POINTS" -eq 0 ] || [ "$WIDTH" -eq 0 ]; then
    echo "points and width must be positive integers" >&2; exit 2
fi
POINTS=$(canonical_decimal "$POINTS")
WIDTH=$(canonical_decimal "$WIDTH")
if [ -n "$TARGET" ] && ! [[ "$TARGET" =~ ^[0-9]+$ ]]; then
    echo "target-rels must be a nonnegative integer" >&2; exit 2
fi
[ -z "$TARGET" ] || TARGET=$(canonical_decimal "$TARGET")
for bound_name in RLIM ALIM; do
    bound=${!bound_name}
    if [ -n "$bound" ]; then
        if ! [[ "$bound" =~ ^[0-9]+$ ]]; then
            echo "${bound_name,,} must be a positive integer" >&2; exit 2
        fi
        bound=$(canonical_decimal "$bound")
        if [ "$bound" -eq 0 ]; then
            echo "${bound_name,,} must be a positive integer" >&2; exit 2
        fi
        printf -v "$bound_name" '%s' "$bound"
    fi
done
if [ "$SQ_SIDE" != 0 ] && [ "$SQ_SIDE" != 1 ]; then
    echo "sq-side must be 1 (algebraic) or 0 (rational)" >&2; exit 2
fi
NORMALIZED_GEOMS=()
for geom in "${GEOMS[@]}"; do
    if ! [[ "$geom" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "geometry must be logI,J with logI in [2,20] and positive J" >&2
        exit 2
    fi
    logI=$(canonical_decimal "${geom%%,*}")
    J=$(canonical_decimal "${geom##*,}")
    if [ "$logI" -lt 2 ] || [ "$logI" -gt 20 ] || [ "$J" = 0 ]; then
        echo "geometry must be logI,J with logI in [2,20] and positive J" >&2
        exit 2
    fi
    NORMALIZED_GEOMS+=("$logI,$J")
done
GEOMS=("${NORMALIZED_GEOMS[@]}")
[ -x ./bench ] || { echo "run me from the bench directory (no ./bench here)" >&2; exit 2; }

# The default factor base is a managed cache, like input.job.afb.0 in the
# GGNFS test sieve. Native fbgen files identify their exact polynomial, lim,
# and maxbits in the first four lines, so inspect those directly instead of
# trusting a filename. Custom --fb1 paths remain caller-managed and are never
# overwritten.
MANAGE_FB=$FB_MANAGED
declare -A FB_BY_LOGI=() SEEN_LOGI=()
for geom in "${GEOMS[@]}"; do SEEN_LOGI[${geom%%,*}]=1; done

if [ "$MANAGE_FB" = 1 ]; then
    [ -n "$FB_THREADS" ] || FB_THREADS=$(default_fb_threads)
    if ! [[ "$FB_THREADS" =~ ^[0-9]+$ ]]; then
        echo "fb-threads must be an integer in [1,256]" >&2; exit 2
    fi
    FB_THREADS=$(canonical_decimal "$FB_THREADS")
    if [ "$FB_THREADS" -eq 0 ] || [ "$FB_THREADS" -gt 256 ]; then
        echo "fb-threads must be an integer in [1,256]" >&2; exit 2
    fi
    [ -x ./fbgen ] || make fbgen
    [ -x ./fbgen ] || { echo "could not build ./fbgen" >&2; exit 1; }

    EFFECTIVE_ALIM=$ALIM
    if [ -z "$EFFECTIVE_ALIM" ]; then
        EFFECTIVE_ALIM=$(sed -nE \
            's/\r$//; s/^[[:space:]]*alim[[:space:]]*:[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' \
            "$POLY" | tail -1)
    fi
    if ! [[ "$EFFECTIVE_ALIM" =~ ^[0-9]+$ ]] || [ "$EFFECTIVE_ALIM" -lt 2 ]; then
        echo "the default factor base needs --alim or an alim: value in $POLY" >&2
        exit 2
    fi
    EFFECTIVE_ALIM=$(canonical_decimal "$EFFECTIVE_ALIM")

    fb_header_err=$(mktemp)
    if ! expected_fb_text=$(./fbgen --poly "$POLY" --lim 2 --maxbits 1 \
            --threads 1 2>"$fb_header_err"); then
        cat "$fb_header_err" >&2
        rm -f "$fb_header_err"
        echo "could not derive factor-base metadata from $POLY" >&2
        exit 1
    fi
    rm -f "$fb_header_err"
    mapfile -t EXPECTED_FB_HEADER < <(printf '%s\n' "$expected_fb_text" | sed -n '1,2p')
    if [ ${#EXPECTED_FB_HEADER[@]} -ne 2 ]; then
        echo "could not derive factor-base metadata from $POLY" >&2; exit 1
    fi

    for logI in "${!SEEN_LOGI[@]}"; do
        cache_fb="${FB}.m${logI}"
        FB_BY_LOGI[$logI]=$cache_fb

        # `-n 1,4p` alone still reads the entire ~200 MB text factor base.
        # Quit after the metadata so cache validation stays constant-time.
        mapfile -t actual_header < <(sed -n '1,4p;4q' "$cache_fb" 2>/dev/null)
        cache_valid=1
        [ ${#actual_header[@]} -eq 4 ] || cache_valid=0
        if [ "$cache_valid" = 1 ]; then
            [ "${actual_header[0]}" = "${EXPECTED_FB_HEADER[0]}" ] || cache_valid=0
            [ "${actual_header[1]}" = "${EXPECTED_FB_HEADER[1]}" ] || cache_valid=0
            [ "${actual_header[2]}" = "# lim = $EFFECTIVE_ALIM" ] || cache_valid=0
            [ "${actual_header[3]}" = "# maxbits = $logI" ] || cache_valid=0
        fi

        if [ "$cache_valid" = 1 ]; then
            echo "[testsieve] factor-base cache valid: $cache_fb"
        else
            echo "[testsieve] rebuilding stale or missing factor base: $cache_fb"
            ./fbgen --poly "$POLY" --lim "$EFFECTIVE_ALIM" \
                --maxbits "$logI" --threads "$FB_THREADS" --out "$cache_fb" || exit 1
        fi
    done
    echo
fi

TMP=$(mktemp -d)
# Announce the path when keeping: mktemp -d names are random, nothing else in
# the output contains one (the per-point messages print basenames), so a kept
# directory was previously findable only by hunting /tmp by mtime.
# shellcheck disable=SC2317  # invoked indirectly by trap
cleanup() {
    if [ "$KEEP" = 1 ]; then echo "[testsieve] kept relation files and logs in $TMP"
    else rm -rf "$TMP"; fi
}
trap cleanup EXIT

echo "[testsieve] $(basename "$POLY")  band [$QMIN, $QMAX)  $POINTS points x ${WIDTH}-wide q intervals"
echo "            yields normalized to WIDTH/ln(q0) expected (q, rho) pairs"
echo "            sieve side: $([ "$SQ_SIDE" = 1 ] && echo algebraic || echo rational)"
[ -n "$RLIM$ALIM" ] && echo "            lim override: rlim=${RLIM:-job} alim=${ALIM:-job}"
echo

ROWS="$TMP/rows.txt"; : > "$ROWS"

for geom in "${GEOMS[@]}"; do
    logI=${geom%%,*}; J=${geom##*,}
    geom_fb=$FB
    [ "$MANAGE_FB" = 0 ] || geom_fb=${FB_BY_LOGI[$logI]}
    area=$(python3 -c "import math;print('2^%.4g'%math.log2((1<<$logI)*$J))")
    printf '  --- logI %s, J %s  (area %s) ---\n' "$logI" "$J" "$area"
    printf '  %-12s %-8s %-9s %-10s %-12s %-9s %-8s %-8s %-8s\n' \
           q0 pairs exp-pairs n-yield exp-rel rel/pair ms/pair rel/s board_W
    PREV="$TMP/prev.txt"
    : > "$PREV"
    for i in $(seq 0 $((POINTS - 1))); do
        # Linear spacing. Yield falls smoothly with q, so the trapezoid rule
        # over evenly spaced points is the right shape; log spacing would
        # oversample the cheap end where the curve is flattest.
        #
        # The span is (QMAX - QMIN - WIDTH) so that the LAST window ends at
        # QMAX rather than starting there: spacing over the full span put the
        # final sample -- a fifth of the evidence, and the lowest-yield one --
        # entirely outside the interval being integrated.
        q0=$(python3 -c "print(int($QMIN + max(0,$QMAX - $QMIN - $WIDTH) * $i / max(1,$POINTS-1)))")
        # bench's qrange upper bound is inclusive, so subtract one to sieve
        # exactly WIDTH integers and keep the normalization denominator exact.
        q1=$((q0 + WIDTH - 1))
        tag="g${logI}_${J}_$i"
        # shellcheck disable=SC2086
        if ! ./bench --pipeline --cofactor --poly "$POLY" --fb1 "$geom_fb" \
            --sq-side "$SQ_SIDE" \
            --logI "$logI" --J "$J" --qrange "$q0:$q1" \
            ${RLIM:+--rlim $RLIM} ${ALIM:+--alim $ALIM} $EXTRA \
            --relations "$TMP/$tag.dat" --log "$TMP/$tag.log" --log-every 1 \
            > "$TMP/$tag.out" 2>&1; then
            printf '  %-12s FAILED: %s\n' "$q0" \
              "$(grep -iE 'error|cannot|refus|does not fit' "$TMP/$tag.out" | head -1 | cut -c1-56)"
            continue
        fi
        pairs=$(grep -oP -- '--- band of \K[0-9]+' "$TMP/$tag.out" | head -1)
        rel=$(grep -oP 'total relations\s+\K[0-9]+' "$TMP/$tag.out" | tail -1)
        # COMPLETE, not the plain 'wall clock per q'. The plain line excludes
        # cofac_tail, the final flush after the band -- and a sample this short
        # never reaches an in-loop flush (CQ_FLUSH is 131072 candidates, ~67
        # special-q), so on these windows that tail IS the whole
        # cofactorisation. Reading the exclusive number understated the C194
        # projection by about 30%. Falls back for a run without --cofactor.
        ms=$(grep -oP 'wall clock per q, COMPLETE\s+\K[0-9.]+' "$TMP/$tag.out" | head -1)
        [ -n "$ms" ] || ms=$(grep -oP 'wall clock per q\s+\K[0-9.]+' "$TMP/$tag.out" | head -1)
        bw=$(grep -oP 'board=\K[0-9.]+' "$TMP/$tag.log" 2>/dev/null |
             awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n}')
        if [ -z "${pairs:-}" ] || [ -z "${rel:-}" ] || [ -z "${ms:-}" ]; then
            # Silence here would drop a point from the integration while the
            # banner still promised N of them, and the flat extrapolation would
            # quietly cover the gap.
            printf '  %-12s UNPARSED: band summary incomplete in %s\n' \
                   "$q0" "$tag.out"
            continue
        fi
        python3 - "$q0" "$WIDTH" "$pairs" "$rel" "$ms" "${bw:-0}" \
            "$logI,$J" "$ROWS" "$PREV" <<'EOF'
import math
import sys
q0,w,pairs,rel,ms,bw,geom,rows,prev_file = sys.argv[1:]
q0,w,pairs,rel,ms,bw = int(q0),int(w),int(pairs),int(rel),float(ms),float(bw)
relpair = rel/pairs if pairs else 0.0
expected = w/math.log(q0)
nrel = relpair*expected
nsecs = expected*ms/1000.0
rels    = relpair/(ms/1000.0) if ms else 0.0
exp_rel = ""
previous = open(prev_file).read().split()
if previous:
    prev_q0, prev_nrel = int(previous[0]), float(previous[1])
    exp_rel = "%.0f" % (((prev_nrel + nrel) / 2.0) * (q0 - prev_q0) / w)
print("  %-12d %-8d %-9.1f %-10.1f %-12s %-9.2f %-8.2f %-8.0f %-8s" %
      (q0, pairs, expected, nrel, exp_rel, relpair, ms, rels,
       ("%.1f"%bw) if bw else "n/a"))
open(rows,"a").write("%s %d %d %d %.9f %.9f %.6f %.3f\n" %
                     (geom,q0,w,pairs,expected,nrel,nsecs,bw))
open(prev_file,"w").write("%d %.9f\n" % (q0,nrel))
EOF
    done
    echo
done

projection_status=0
python3 - "$QMIN" "$QMAX" "${TARGET:-0}" "$ROWS" <<'EOF' || projection_status=$?
import sys
qmin, qmax, target, rows = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
data = {}
for ln in open(rows):
    g,q0,w,pairs,expected,nrel,nsecs,bw = ln.split()
    data.setdefault(g, []).append((int(q0), int(w), int(pairs), float(expected),
                                   float(nrel), float(nsecs), float(bw)))
if not data:
    print("  no usable points"); raise SystemExit(1)

print("  === projection over [%d, %d) ===" % (qmin, qmax))
print("  %-12s %-14s %-12s %-12s %-10s %s" %
      ("geometry", "n-relations", "GPU days", "energy kWh", "rel/kJ", "target reached at q"))
best = None
for g, pts in data.items():
    pts.sort()
    # Trapezoid over normalized relations and seconds per unit q. Normalizing
    # both removes accidental prime/root-density swings from short samples.
    rel_dens  = [(q, nrel / w) for q, w, p, expected, nrel, nsecs, bw in pts]
    time_dens = [(q, nsecs / w) for q, w, p, expected, nrel, nsecs, bw in pts]
    pw = [bw for q, w, p, expected, nrel, nsecs, bw in pts if bw > 0]
    watts = sum(pw) / len(pw) if pw else 0.0

    def integrate(dens, lo, hi):
        """Trapezoid, extrapolating flat outside the sampled range."""
        tot = 0.0
        xs = [d[0] for d in dens]
        for i in range(len(dens) - 1):
            (xa, ya), (xb, yb) = dens[i], dens[i + 1]
            a, b = max(lo, xa), min(hi, xb)
            if b <= a: continue
            fa = ya + (yb - ya) * (a - xa) / (xb - xa)
            fb = ya + (yb - ya) * (b - xa) / (xb - xa)
            tot += (fa + fb) / 2 * (b - a)
        if lo < xs[0]:  tot += dens[0][1]  * (min(hi, xs[0]) - lo)
        if hi > xs[-1]: tot += dens[-1][1] * (hi - max(lo, xs[-1]))
        return tot

    rels = integrate(rel_dens, qmin, qmax)
    secs = integrate(time_dens, qmin, qmax)
    kwh  = watts * secs / 3.6e6
    relkj = rels / (watts * secs / 1000.0) if watts and secs else 0.0

    tq = "-"
    if target:
        lo, hi = qmin, qmax
        if rels < target:            # already integrated over [qmin, qmax)
            tq = "not within band"
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                if integrate(rel_dens, qmin, mid) < target: lo = mid
                else: hi = mid
            d = integrate(time_dens, qmin, hi) / 86400.0
            tq = "%.0f  (%.2f d)" % (hi, d)
    print("  %-12s %-14s %-12.2f %-12.1f %-10.1f %s" %
          (g, "%.3g" % rels, secs / 86400.0, kwh, relkj, tq))
    if watts > 0 and (best is None or relkj > best[1]): best = (g, relkj)

if len(data) > 1:
    if best:
        print("\n  best relations per kilojoule: %s" % best[0])
    else:
        # Every relkj is 0 because no board-power sample arrived (no NVML, no
        # sensor). Picking a winner here would be picking whichever geometry
        # was measured first and calling it an energy result.
        print("\n  no board-power samples: the energy columns are not meaningful")
print("\n  Startup is excluded from every timing, and these are GPU-busy days:")
print("  a real run adds the factor-base load once and whatever the host steals")
print("  (RESULTS.md finding 56: ~6% at one competing thread per core).")
EOF

echo
echo "--- $(basename "$POLY") (source job plus selected sieve side) ---"
cat -- "$POLY" || [ "$projection_status" -ne 0 ] || projection_status=1
if [ "$SQ_SIDE" = 1 ]; then
    # Match the GGNFS test sieve's result.job display without modifying the
    # user's input file or leaving a generated job in the worktree.
    if [ -s "$POLY" ] && [ -n "$(tail -c1 -- "$POLY")" ]; then echo; fi
    echo "lss: 0"
fi
[ -z "$RLIM" ] || echo "# test-sieve override: rlim=$RLIM"
[ -z "$ALIM" ] || echo "# test-sieve override: alim=$ALIM"
[ -z "$EXTRA" ] || echo "# test-sieve extra flags: $EXTRA"
echo
exit "$projection_status"
