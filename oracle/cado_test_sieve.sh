#!/bin/bash

# CADO-NFS test sieve script
# Equivalent of test_sieve.sh (GGNFS) but for CADO-NFS's las siever.
# Reads input.job (GGNFS format) and translates parameters for las.

set -euo pipefail

# --- Configuration ---
INIFILE="cado_test_sieve.ini"
if [ -f "$INIFILE" ]; then
    source "$INIFILE"
else
    echo "Error: $INIFILE not found. Copy cado_test_sieve.ini.example and edit it."
    exit 1
fi
LAS="$CADO_BUILD/sieve/las"
MAKEFB="$CADO_BUILD/sieve/makefb"

if [ ! -x "$LAS" ]; then
    echo "Error: las not found at $LAS"
    echo "Set CADO_BUILD to your cado-nfs build directory."
    exit 1
fi
if [ ! -x "$MAKEFB" ]; then
    echo "Error: makefb not found at $MAKEFB"
    exit 1
fi

# --- Parse input.job (GGNFS format) ---
template="input.job"
if [ ! -f "$template" ]; then
    echo "Error: $template not found!"
    exit 1
fi

get_param() {
    grep "^$1:" "$template" | tr -d '\r' | sed 's/[^[:print:]]//g' | awk '{print $2}' || true
}

# Translate GGNFS → CADO parameter names
lim0=$(get_param rlim)
lim1=$(get_param alim)
lpb0=$(get_param lpbr)
lpb1=$(get_param lpba)
mfb0=$(get_param mfbr)
mfb1=$(get_param mfba)
lambda0="${LAMBDA0:-}"
lambda1="${LAMBDA1:-}"

# Polynomial coefficients
n=$(get_param n)
skew=$(get_param skew)
Y0=$(get_param Y0)
Y1=$(get_param Y1)

# Collect algebraic coefficients c0..c9 (support degree 5 through 6+)
# For SNFS, coefficients can be sparse (e.g. c0 and c6 only), so don't break on gaps.
declare -a ccoeffs=()
max_deg=-1
for i in $(seq 0 9); do
    val=$(get_param "c$i")
    if [ -n "$val" ]; then
        ccoeffs[$i]="$val"
        max_deg=$i
    fi
done
# Fill in missing intermediate coefficients as 0
for i in $(seq 0 "$max_deg"); do
    if [ -z "${ccoeffs[$i]+x}" ]; then
        ccoeffs[$i]=0
    fi
done

echo "--- Parameters from input.job ---"
echo "lim0=$lim0  lim1=$lim1"
echo "lpb0=$lpb0  lpb1=$lpb1"
echo "mfb0=$mfb0  mfb1=$mfb1"
if [ -n "$lambda0" ]; then echo "lambda0=$lambda0 (from env)"; else echo "lambda0: las default"; fi
if [ -n "$lambda1" ]; then echo "lambda1=$lambda1 (from env)"; else echo "lambda1: las default"; fi
echo "degree=$(( ${#ccoeffs[@]} - 1 ))  skew=$skew"
echo ""

# Optional flags: use env overrides or let las pick defaults
lambda0_flag=""
lambda1_flag=""
if [ -n "$lambda0" ]; then lambda0_flag="-lambda0 $lambda0"; fi
if [ -n "$lambda1" ]; then lambda1_flag="-lambda1 $lambda1"; fi

ncurves0_flag=""
ncurves1_flag=""
if [ -n "${NCURVES0:-}" ]; then
    ncurves0_flag="-ncurves0 $NCURVES0"
    echo "ncurves0=$NCURVES0 (from env)"
fi
if [ -n "${NCURVES1:-}" ]; then
    ncurves1_flag="-ncurves1 $NCURVES1"
    echo "ncurves1=$NCURVES1 (from env)"
fi

# --- Step 1: Generate temp poly file ---
polyfile="cado_tmp.poly"
{
    echo "n: $n"
    echo "skew: $skew"
    for i in "${!ccoeffs[@]}"; do
        echo "c${i}: ${ccoeffs[$i]}"
    done
    echo "Y0: $Y0"
    echo "Y1: $Y1"
} > "$polyfile"
echo "Wrote CADO poly file: $polyfile"

# --- Step 2: Generate factor base (algebraic side) ---
rootsfile="cado_roots1.gz"
fb_hashfile=".fb_params.sha256"
fb_hash=$(echo "$n $lim1 $(cat "$polyfile")" | sha256sum | awk '{print $1}')
regen_fb=false
if [ ! -f "$rootsfile" ]; then
    regen_fb=true
elif [ ! -f "$fb_hashfile" ] || [ "$(cat "$fb_hashfile")" != "$fb_hash" ]; then
    echo "Factor base cache is stale (poly or lim1 changed), regenerating..."
    regen_fb=true
fi
if [ "$regen_fb" = true ]; then
    echo "Running makefb (lim=$lim1) ..."
    "$MAKEFB" -poly "$polyfile" -lim "$lim1" -out "$rootsfile"
    echo "$fb_hash" > "$fb_hashfile"
    echo "Factor base written: $rootsfile"
else
    echo "Factor base cache valid: $rootsfile (skipping makefb)"
fi
echo ""

# --- Step 3: Interactive prompts ---
echo "Choose sieve area:"
echo "  1) -A 27  (I=14, ~125MB)"
echo "  2) -A 28  (I=14.5, ~250MB)"
echo "  3) -A 29  (I=15, ~500MB)"
echo "  4) -A 30  (I=15.5, ~1GB)"
echo "  5) -A 31  (I=16, ~2GB)"
echo "  6) -A 32  (I=16.5, ~4GB)"
read -p "Selection [1-6]: " area_choice
case "$area_choice" in
    1) A=27 ;;
    2) A=28 ;;
    3) A=29 ;;
    4) A=30 ;;
    5) A=31 ;;
    6|*) A=32 ;;
esac
echo "Using -A $A"
echo ""

read -p "Sieve algebraic (a) or rational (r) side? [a/r]: " side_choice
case "$side_choice" in
    r|R)
        sqside=0
        side_name="rational"
        ;;
    a|A|*)
        sqside=1
        side_name="algebraic"
        ;;
esac
echo "Sieving $side_name side (sqside=$sqside)"
echo ""

read -p "Two-stage mode? [y/n]: " two_stage_choice
declare -a q_points=()
if [ "$two_stage_choice" = "y" ] || [ "$two_stage_choice" = "Y" ]; then
    read -p "Stage 1 start (millions): " s1_start_m
    read -p "Stage 1 end (millions): " s1_end_m
    read -p "Stage 1 interval (millions): " s1_interval_m
    read -p "Stage 2 end (millions): " s2_end_m
    read -p "Stage 2 interval (millions): " s2_interval_m
    cur=$((s1_start_m * 1000000))
    s1_end=$((s1_end_m * 1000000))
    s1_interval=$((s1_interval_m * 1000000))
    s2_interval=$((s2_interval_m * 1000000))
    while [ "$cur" -le "$s1_end" ]; do
        q_points+=("$cur")
        cur=$((cur + s1_interval))
    done
    cur=$((s1_end_m * 1000000 + s2_interval))
    s2_end=$((s2_end_m * 1000000))
    while [ "$cur" -le "$s2_end" ]; do
        q_points+=("$cur")
        cur=$((cur + s2_interval))
    done
else
    read -p "Enter start q (in millions): " start_m
    read -p "Enter end q (in millions): " end_m
    if [ "$start_m" -eq "$end_m" ]; then
        interval_m=1
    else
        read -p "Enter interval (in millions): " interval_m
    fi
    cur=$((start_m * 1000000))
    end_val=$((end_m * 1000000))
    interval=$((interval_m * 1000000))
    while [ "$cur" -le "$end_val" ]; do
        q_points+=("$cur")
        cur=$((cur + interval))
    done
fi
qintsize=1000

# --- Step 4: Sieve loop ---
declare -a q0_array
declare -a yield_array
declare -a nyld_array
declare -a specq_array
declare -a expq_array
declare -a speed_array
declare -a time_array

index=0
for current in "${q_points[@]}"; do
    current_m=$((current / 1000000))
    q1=$((current + qintsize))
    outfile="cado_out.${current_m}M.txt"
    logfile="cado_log.${current_m}M.txt"

    echo ""
    echo "=========================================="
    echo "  CADO Test Sieve: q0=${current} (${current_m}M)"
    echo "=========================================="

    # Build las command
    las_cmd=("$LAS"
        -poly "$polyfile"
        -fb1 "$rootsfile"
        -lim0 "$lim0" -lim1 "$lim1"
        -lpb0 "$lpb0" -lpb1 "$lpb1"
        -mfb0 "$mfb0" -mfb1 "$mfb1"
        -A "$A" -sqside "$sqside"
        -bkmult "1,1s:1.2"
        -adjust-strategy 2
        -q0 "$current" -q1 "$q1"
        -out "$outfile" -v -stats-stderr
    )
    # Append optional flags
    if [ -n "$lambda0_flag" ]; then las_cmd+=($lambda0_flag); fi
    if [ -n "$lambda1_flag" ]; then las_cmd+=($lambda1_flag); fi
    if [ -n "$ncurves0_flag" ]; then las_cmd+=($ncurves0_flag); fi
    if [ -n "$ncurves1_flag" ]; then las_cmd+=($ncurves1_flag); fi

    echo ""
    echo "Command: ${las_cmd[*]}"
    echo ""

    # Run las, capture all output (las writes comments to stdout with -v)
    start_time=$SECONDS
    "${las_cmd[@]}" 2>&1 | tee "$logfile"
    elapsed=$(( SECONDS - start_time ))

    # Parse results from log
    # Count special-q processed: per-q "# Sieving side-1 q=NNNN;" lines
    specq=$(grep -c '# Sieving side-[01] q=' "$logfile" || true)
    specq=${specq:-0}
    # Fallback: parse from summary "for NN special-q's"
    if [ "$specq" -eq 0 ]; then
        specq=$(grep -oP 'for \K\d+(?= special-q)' "$logfile" | head -1 || true)
        specq=${specq:-0}
    fi

    # Total yield from las summary: "# Total NNN reports"
    total_reports=$(grep -oP '# Total \K\d+(?= reports)' "$logfile" || true)
    if [ -n "$total_reports" ]; then
        yield="$total_reports"
        echo "  Total reports (from las): $total_reports"
    else
        # Fall back to summing per-q yields: "# NNN relation(s)" lines
        yield=$({ grep -oP '# \K\d+(?= relation\(s\))' "$logfile" || true; } | awk '{s+=$1}END{print s+0}')
    fi

    # Parse elapsed time from las "# Total elapsed time NNNs" line
    las_elapsed=$(grep -oP '# Total elapsed time \K[0-9.]+(?=s)' "$logfile" || true)
    # Fallback: "in NNN elapsed s" (handles scientific notation)
    if [ -z "$las_elapsed" ]; then
        las_elapsed=$(grep -oP 'in \K[0-9.eE+\-]+(?= elapsed s)' "$logfile" || true)
    fi

    # Use las-reported time if available, otherwise bash timer
    if [ -n "$las_elapsed" ]; then
        wall_time="$las_elapsed"
    elif [ "$elapsed" -gt 0 ]; then
        wall_time="$elapsed"
    else
        wall_time="1"
    fi

    # Speed: relations per second
    if [ "$yield" -gt 0 ]; then
        speed_relsec=$(echo "scale=3; $yield / $wall_time" | bc -l)
    else
        speed_relsec="0"
    fi

    # expq = qintsize / ln(q0) — expected number of primes in [q0, q0+qintsize]
    expq=$(echo "scale=3; $qintsize / l($current)" | bc -l)

    # n-yld = yield * (expq / specq) — normalized yield per expected special-q
    if [ "$specq" -gt 0 ]; then
        nyld=$(echo "scale=6; $yield * ($expq / $specq)" | bc -l)
    else
        nyld="0"
    fi

    echo ""
    echo "--- Summary for ${current_m}M ---"
    echo "  special-q processed: $specq"
    echo "  yield: $yield"
    echo "  expq: $(printf '%.1f' "$expq")"
    echo "  n-yld: $(printf '%.0f' "$nyld")"
    echo "  wall time: ${wall_time}s"
    echo "  speed: ${speed_relsec} rel/s"

    # Store results
    q0_array[$index]=$current
    yield_array[$index]=$yield
    nyld_array[$index]=$nyld
    specq_array[$index]=$specq
    expq_array[$index]=$expq
    speed_array[$index]=$speed_relsec
    time_array[$index]=$wall_time

    index=$((index + 1))
done

# --- Step 5: Summary table ---
echo ""
echo "=========================================="
echo "  All test sieves completed!"
echo "=========================================="
echo ""

# Suggested lambda values (CADO convention: lambda = mfb / lpb)
echo "--- Suggested Lambda Values (CADO: mfb/lpb) ---"
lambda0_suggested=$(echo "scale=2; $mfb0 / $lpb0" | bc -l)
lambda1_suggested=$(echo "scale=2; $mfb1 / $lpb1" | bc -l)
echo "  lambda0 (rational):  $lambda0_suggested  (mfb0=$mfb0 / lpb0=$lpb0)"
echo "  lambda1 (algebraic): $lambda1_suggested  (mfb1=$mfb1 / lpb1=$lpb1)"
if [ -n "$lambda0" ] || [ -n "$lambda1" ]; then
    echo "  Overrides: lambda0=${lambda0:-default}  lambda1=${lambda1:-default}"
fi
echo ""

# Shortened summary table
printf "%-12s %-8s %-8s %-12s %-15s\n" "q0" "yield" "n-yld" "exp_rel" "speed"
printf "%-12s %-8s %-8s %-12s %-15s\n" "--------" "------" "------" "--------" "-----------"

prev_nyld=""
prev_q0=""
total_exp_rel=0
total_time=0

for i in "${!q0_array[@]}"; do
    q0="${q0_array[$i]}"
    yield="${yield_array[$i]}"
    nyld="${nyld_array[$i]}"
    speed_relsec="${speed_array[$i]}"

    # exp_rel = avg(last two n-yld) * (q0[i] - q0[i-1]) / qintsize
    if [ -n "$prev_nyld" ]; then
        avg_nyld=$(echo "scale=6; ($nyld + $prev_nyld) / 2" | bc -l)
        q0_diff=$((q0 - prev_q0))
        exp_rel=$(echo "scale=0; $avg_nyld * $q0_diff / $qintsize" | bc -l)
        total_exp_rel=$(echo "scale=0; $total_exp_rel + $exp_rel" | bc -l)
        if [ "$(echo "$speed_relsec > 0" | bc -l)" -eq 1 ]; then
            seg_time=$(echo "scale=6; $exp_rel / $speed_relsec" | bc -l)
            total_time=$(echo "scale=6; $total_time + $seg_time" | bc -l)
        fi
    else
        exp_rel=""
    fi

    nyld_fmt=$(printf "%.0f" "$nyld")
    speed_fmt=$(printf "%.3f r/s" "$speed_relsec")

    if [ -n "$exp_rel" ]; then
        exp_rel_fmt=$(printf "%.0f" "$exp_rel")
        printf "%-12s %-8s %-8s %-12s %-15s\n" "$q0" "$yield" "$nyld_fmt" "$exp_rel_fmt" "$speed_fmt"
    else
        printf "%-12s %-8s %-8s %-12s %-15s\n" "$q0" "$yield" "$nyld_fmt" "" "$speed_fmt"
    fi

    prev_nyld="$nyld"
    prev_q0="$q0"
done

# Totals row with ETA
total_exp_rel_fmt=$(printf "%.0f" "$total_exp_rel")
if [ "$(echo "$total_time > 0" | bc -l)" -eq 1 ]; then
    avg_speed=$(echo "scale=3; $total_exp_rel / $total_time" | bc -l)
    avg_speed_fmt=$(printf "%.3f r/s" "$avg_speed")
    total_time_int=$(printf "%.0f" "$total_time")
    days=$((total_time_int / 86400))
    hours=$(( (total_time_int % 86400) / 3600 ))
    mins=$(( (total_time_int % 3600) / 60 ))
    if [ "$days" -gt 0 ]; then
        time_fmt=$(printf "%dd %dh %dm" "$days" "$hours" "$mins")
    elif [ "$hours" -gt 0 ]; then
        time_fmt=$(printf "%dh %dm" "$hours" "$mins")
    else
        time_fmt=$(printf "%dm" "$mins")
    fi
    printf "%-12s %-8s %-8s %-12s %-15s %s\n" "" "" "Total:" "$total_exp_rel_fmt" "$avg_speed_fmt" "ETA: $time_fmt"
else
    printf "%-12s %-8s %-8s %-12s\n" "" "" "Total:" "$total_exp_rel_fmt"
fi
echo ""

# --- Step 6: Test configuration ---
echo "--- Test Configuration ---"
echo "CADO build: $CADO_BUILD"
echo "Sieve area: -A $A"
echo "qintsize: $qintsize"
echo ""

# --- Step 7: Cleanup ---
rm -f cado_log.*.txt cado_out.*.txt
echo "Cleaned up temp log/output files."
echo "Kept: $polyfile, $rootsfile (reusable)"
