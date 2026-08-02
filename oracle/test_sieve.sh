#!/bin/bash

# Prompt for side to sieve
read -p "Sieve algebraic (a), rational (r), or both (b)? [a/r/b]: " side_choice
sieve_both=false
case "$side_choice" in
    r|R)
        side_flag="-r"
        side_name="rational"
        ;;
    b|B)
        sieve_both=true
        side_name="both"
        ;;
    a|A|*)
        side_flag="-a"
        side_name="algebraic"
        ;;
esac

echo "Sieving $side_name side"
echo ""

# Prompt for siever version
read -p "Choose siever version [14/15/16]: " siever_choice
case "$siever_choice" in
    14)
        siever="gnfs-lasieve4I14e"
        ;;
    16)
        siever="gnfs-lasieve4I16e"
        ;;
    15|*)
        siever="gnfs-lasieve4I15e"
        ;;
esac

# If siever 16, ask about expanded sieve size
j_flag=""
if [ "$siever_choice" = "16" ]; then
    read -p "Use expanded sieve size (2^16x2^16 instead of 2^16x2^15)? [y/n]: " expand_choice
    if [ "$expand_choice" = "y" ] || [ "$expand_choice" = "Y" ]; then
        j_flag="-J 16"
    fi
fi

echo "Using $siever"
echo ""

# Prompt for parameters
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
    read -p "Enter start value (in millions): " start_m
    read -p "Enter end value (in millions): " end_m
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

# Base template file
template="input.job"

# Check if template exists
if [ ! -f "$template" ]; then
    echo "Error: $template not found!"
    exit 1
fi

# ── Factor base cache ──
# Current sievers auto-load <job>.afb.0 and trim it to each run's q-dependent
# bound, so build the algebraic factor base once up front instead of once per
# test point. "-c 0" builds and writes the full-alim factor base without
# sieving anything. The cache only depends on the polynomial and alim, so it
# survives lambda/mfb/lpb tweaks; the hash file triggers a rebuild when the
# poly or alim change.
afb_file="${template}.afb.0"
afb_hash_file=".afb_params.sha256"
if ! grep -aq "Trimmed cached aFB" "./$siever"; then
    echo "Note: $siever predates factor-base cache support; removing any cache and proceeding without it."
    rm -f "$afb_file" "$afb_hash_file"
else
    # The "v2-" salt ties the hash to the trailered cache format; bumping it
    # forces regeneration of caches written by older sievers, which current
    # binaries reject.
    afb_hash="v2-$(grep -E '^(c[0-9]+|Y[01]|alim):' "$template" | tr -d '\r' | sha256sum | cut -d' ' -f1)"
    if [ -f "$afb_file" ] && [ "$(cat "$afb_hash_file" 2>/dev/null)" = "$afb_hash" ]; then
        echo "Using existing factor base cache ($afb_file)."
    else
        echo "Building factor base cache ($afb_file)..."
        rm -f "$afb_file" "$afb_hash_file"
        ./$siever -k -F -c 0 -f 1000 -o fbgen.tmp.out -n0 -a "$template"
        rm -f fbgen.tmp.out
        if [ -f "$afb_file" ]; then
            echo "$afb_hash" > "$afb_hash_file"
            echo "Factor base cache ready."
        else
            echo "Warning: cache generation failed; sievers will rebuild the factor base per run."
        fi
    fi
fi
echo ""

# Hardcode qintsize
qintsize=1000

# Arrays to store results (generic, reused per side)
declare -a q0_array expq_array specq_array yield_array nyld_array speed_array
# Side-specific arrays (used when sieve_both=true)
declare -a r_yield_array r_nyld_array r_speed_array r_specq_array r_expq_array
declare -a a_yield_array a_nyld_array a_speed_array a_specq_array a_expq_array

# Determine which sides to run
if [ "$sieve_both" = true ]; then
    sides_to_run=("r" "a")
elif [ "$side_name" = "rational" ]; then
    sides_to_run=("r")
else
    sides_to_run=("a")
fi

for run_side in "${sides_to_run[@]}"; do
    if [ "$run_side" = "r" ]; then
        cur_side_flag="-r"
        cur_side_name="rational"
    else
        cur_side_flag="-a"
        cur_side_name="algebraic"
    fi

    if [ "$sieve_both" = true ]; then
        echo ""
        echo "=========================================="
        echo "  Starting $cur_side_name side sieve"
        echo "=========================================="
        echo ""
    fi

    # Reset generic arrays for this side's run
    q0_array=(); expq_array=(); specq_array=(); yield_array=(); nyld_array=(); speed_array=()
    index=0

    for current in "${q_points[@]}"; do
        current_m=$((current / 1000000))
        if [ "$sieve_both" = true ]; then
            output_file="output.${current_m}M.${run_side}.out"
            log_file="log.${current_m}M.${run_side}.txt"
        else
            output_file="output.${current_m}M.out"
            log_file="log.${current_m}M.txt"
        fi

        echo ""
        echo "=========================================="
        echo "  Test Sieve Run: ${current_m}M"
        echo "=========================================="

        ./$siever -v -n0 $j_flag -c $qintsize -f "$current" -o "$output_file" $cur_side_flag "$template" 2>&1 | tee "$log_file"

        echo ""
        echo "--- Summary for ${current_m}M ---"

        specq=$(grep "Special q," "$log_file" | tail -1 | grep -oE "[0-9]+" | head -1)
        yield=$(grep "^Total yield:" "$log_file" | grep -oE "[0-9]+")
        speed=$(grep "sec/rel" "$log_file" | grep -oE "[0-9.]+ sec/rel" | tail -1)

        echo "$specq Special q"
        echo "Total yield: $yield"
        echo "($speed)"
        echo ""

        expq=$(echo "scale=3; $qintsize / l($current)" | bc -l)

        if [ "$specq" -gt 0 ]; then
            nyld=$(echo "scale=6; $yield * ($expq / $specq)" | bc -l)
        else
            nyld="0"
        fi

        q0_array[$index]=$current
        expq_array[$index]=$expq
        specq_array[$index]=$specq
        yield_array[$index]=$yield
        nyld_array[$index]=$nyld
        speed_array[$index]="$speed"
        index=$((index + 1))
    done

    # Copy results to side-specific arrays when running both
    if [ "$sieve_both" = true ]; then
        if [ "$run_side" = "r" ]; then
            r_yield_array=("${yield_array[@]}")
            r_nyld_array=("${nyld_array[@]}")
            r_speed_array=("${speed_array[@]}")
            r_specq_array=("${specq_array[@]}")
            r_expq_array=("${expq_array[@]}")
        else
            a_yield_array=("${yield_array[@]}")
            a_nyld_array=("${nyld_array[@]}")
            a_speed_array=("${speed_array[@]}")
            a_specq_array=("${specq_array[@]}")
            a_expq_array=("${expq_array[@]}")
        fi
    fi
done

echo ""
echo "=========================================="
echo "All test sieves completed!"
echo "=========================================="
echo ""

# ── Helper: print the long table for current q0_array/yield_array/etc. ──
print_long_table() {
    printf "%-12s %-10s %-8s %-8s %-8s %-8s %-15s %-12s\n" \
        "q0" "qintsize" "expq" "specq" "yield" "n-yld" "speed" "exp_rel"

    local prev_nyld="" prev_q0=""
    for i in "${!q0_array[@]}"; do
        local q0="${q0_array[$i]}" expq="${expq_array[$i]}" specq="${specq_array[$i]}"
        local yield="${yield_array[$i]}" nyld="${nyld_array[$i]}" speed="${speed_array[$i]}"
        local exp_rel=""
        if [ -n "$prev_nyld" ]; then
            local avg_nyld q0_diff
            avg_nyld=$(echo "scale=6; ($nyld + $prev_nyld) / 2" | bc -l)
            q0_diff=$((q0 - prev_q0))
            exp_rel=$(echo "scale=0; $avg_nyld * $q0_diff / $qintsize" | bc -l)
        fi
        local expq_fmt nyld_fmt speed_fmt
        expq_fmt=$(printf "%.1f" "$expq")
        nyld_fmt=$(printf "%.0f" "$nyld")
        speed_fmt=$(printf "%.3f sec/rel" "${speed% sec/rel}")
        if [ -n "$exp_rel" ]; then
            printf "%-12s %-10s %-8s %-8s %-8s %-8s %-15s %-12s\n" \
                "$q0" "$qintsize" "$expq_fmt" "$specq" "$yield" "$nyld_fmt" "$speed_fmt" "$(printf '%.0f' "$exp_rel")"
        else
            printf "%-12s %-10s %-8s %-8s %-8s %-8s %-15s\n" \
                "$q0" "$qintsize" "$expq_fmt" "$specq" "$yield" "$nyld_fmt" "$speed_fmt"
        fi
        prev_nyld="$nyld"; prev_q0="$q0"
    done
}

# ── Helper: print the shortened table for current q0_array/yield_array/etc. ──
print_short_table() {
    printf "%-12s %-8s %-8s %-12s %-15s\n" "q0" "yield" "n-yld" "exp_rel" "speed"
    printf "%-12s %-8s %-8s %-12s %-15s\n" "--------" "------" "------" "--------" "-----------"

    local prev_nyld="" prev_q0="" total_exp_rel=0 total_time=0
    for i in "${!q0_array[@]}"; do
        local q0="${q0_array[$i]}" yield="${yield_array[$i]}" nyld="${nyld_array[$i]}"
        local speed="${speed_array[$i]}"
        local speed_val="${speed% sec/rel}"
        local speed_relsec="0"
        if [ "$(echo "$speed_val > 0" | bc -l)" -eq 1 ]; then
            speed_relsec=$(echo "scale=3; 1 / $speed_val" | bc -l)
        fi
        local exp_rel=""
        if [ -n "$prev_nyld" ]; then
            local avg_nyld q0_diff seg_time
            avg_nyld=$(echo "scale=6; ($nyld + $prev_nyld) / 2" | bc -l)
            q0_diff=$((q0 - prev_q0))
            exp_rel=$(echo "scale=0; $avg_nyld * $q0_diff / $qintsize" | bc -l)
            total_exp_rel=$(echo "scale=0; $total_exp_rel + $exp_rel" | bc -l)
            if [ "$(echo "$speed_relsec > 0" | bc -l)" -eq 1 ]; then
                seg_time=$(echo "scale=6; $exp_rel / $speed_relsec" | bc -l)
                total_time=$(echo "scale=6; $total_time + $seg_time" | bc -l)
            fi
        fi
        local nyld_fmt speed_fmt
        nyld_fmt=$(printf "%.0f" "$nyld")
        speed_fmt=$(printf "%.3f rel/sec" "$speed_relsec")
        if [ -n "$exp_rel" ]; then
            printf "%-12s %-8s %-8s %-12s %-15s\n" "$q0" "$yield" "$nyld_fmt" "$(printf '%.0f' "$exp_rel")" "$speed_fmt"
        else
            printf "%-12s %-8s %-8s %-12s %-15s\n" "$q0" "$yield" "$nyld_fmt" "" "$speed_fmt"
        fi
        prev_nyld="$nyld"; prev_q0="$q0"
    done

    local total_exp_rel_fmt
    total_exp_rel_fmt=$(printf "%.0f" "$total_exp_rel")
    if [ "$(echo "$total_time > 0" | bc -l)" -eq 1 ]; then
        local avg_speed avg_speed_fmt total_time_int days hours mins time_fmt
        avg_speed=$(echo "scale=3; $total_exp_rel / $total_time" | bc -l)
        avg_speed_fmt=$(printf "%.3f rel/sec" "$avg_speed")
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
}

# ── Calculate suggested lambdas ──
mfbr=$(grep "^mfbr:" "$template" | tr -d '\r' | awk '{print $2}')
mfba=$(grep "^mfba:" "$template" | tr -d '\r' | awk '{print $2}')
rlim=$(grep "^rlim:" "$template" | tr -d '\r' | awk '{print $2}')
alim=$(grep "^alim:" "$template" | tr -d '\r' | awk '{print $2}')

if [ "$sieve_both" = true ]; then
    # ── Both-sides path: individual shortened tables, then combined ──

    echo "--- Rational Side ---"
    yield_array=("${r_yield_array[@]}")
    nyld_array=("${r_nyld_array[@]}")
    speed_array=("${r_speed_array[@]}")
    specq_array=("${r_specq_array[@]}")
    expq_array=("${r_expq_array[@]}")
    print_short_table
    echo ""

    echo "--- Algebraic Side ---"
    yield_array=("${a_yield_array[@]}")
    nyld_array=("${a_nyld_array[@]}")
    speed_array=("${a_speed_array[@]}")
    specq_array=("${a_specq_array[@]}")
    expq_array=("${a_expq_array[@]}")
    print_short_table
    echo ""

    # ── Combined summary ──
    echo "--- Combined Summary (rational + algebraic) ---"
    printf "%-12s %-10s %-10s %-14s\n" "q0" "r-nyld" "a-nyld" "comb-exp_rel"
    printf "%-12s %-10s %-10s %-14s\n" "--------" "------" "------" "------------"

    prev_r_nyld=""
    prev_a_nyld=""
    prev_q0_comb=""
    total_comb_exp_rel=0
    total_r_time=0
    total_a_time=0

    for i in "${!q0_array[@]}"; do
        q0="${q0_array[$i]}"
        r_nyld="${r_nyld_array[$i]}"
        a_nyld="${a_nyld_array[$i]}"
        r_speed="${r_speed_array[$i]}"
        a_speed="${a_speed_array[$i]}"

        r_speed_val="${r_speed% sec/rel}"
        a_speed_val="${a_speed% sec/rel}"
        r_relsec="0"; a_relsec="0"
        if [ "$(echo "$r_speed_val > 0" | bc -l)" -eq 1 ]; then
            r_relsec=$(echo "scale=3; 1 / $r_speed_val" | bc -l)
        fi
        if [ "$(echo "$a_speed_val > 0" | bc -l)" -eq 1 ]; then
            a_relsec=$(echo "scale=3; 1 / $a_speed_val" | bc -l)
        fi

        comb_exp_rel=""
        if [ -n "$prev_r_nyld" ] && [ -n "$prev_a_nyld" ]; then
            avg_comb=$(echo "scale=6; ($r_nyld + $a_nyld + $prev_r_nyld + $prev_a_nyld) / 2" | bc -l)
            q0_diff=$((q0 - prev_q0_comb))
            comb_exp_rel=$(echo "scale=0; $avg_comb * $q0_diff / $qintsize" | bc -l)
            total_comb_exp_rel=$(echo "scale=0; $total_comb_exp_rel + $comb_exp_rel" | bc -l)
            # Accumulate sieve time per side
            if [ "$(echo "$r_relsec > 0" | bc -l)" -eq 1 ]; then
                r_seg_exp=$(echo "scale=6; ($r_nyld + $prev_r_nyld) / 2 * $q0_diff / $qintsize" | bc -l)
                total_r_time=$(echo "scale=6; $total_r_time + $r_seg_exp / $r_relsec" | bc -l)
            fi
            if [ "$(echo "$a_relsec > 0" | bc -l)" -eq 1 ]; then
                a_seg_exp=$(echo "scale=6; ($a_nyld + $prev_a_nyld) / 2 * $q0_diff / $qintsize" | bc -l)
                total_a_time=$(echo "scale=6; $total_a_time + $a_seg_exp / $a_relsec" | bc -l)
            fi
        fi

        r_nyld_fmt=$(printf "%.0f" "$r_nyld")
        a_nyld_fmt=$(printf "%.0f" "$a_nyld")
        if [ -n "$comb_exp_rel" ]; then
            printf "%-12s %-10s %-10s %-14s\n" "$q0" "$r_nyld_fmt" "$a_nyld_fmt" "$(printf '%.0f' "$comb_exp_rel")"
        else
            printf "%-12s %-10s %-10s %-14s\n" "$q0" "$r_nyld_fmt" "$a_nyld_fmt" ""
        fi

        prev_r_nyld="$r_nyld"
        prev_a_nyld="$a_nyld"
        prev_q0_comb="$q0"
    done

    total_comb_fmt=$(printf "%.0f" "$total_comb_exp_rel")
    total_both_time=$(echo "scale=6; $total_r_time + $total_a_time" | bc -l)
    if [ "$(echo "$total_both_time > 0" | bc -l)" -eq 1 ]; then
        comb_speed=$(echo "scale=3; $total_comb_exp_rel / $total_both_time" | bc -l)
        comb_speed_fmt=$(printf "%.3f rel/sec" "$comb_speed")
        total_time_int=$(printf "%.0f" "$total_both_time")
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
        printf "%-12s %-10s %-10s %-14s %-18s %s\n" "" "" "Total:" "$total_comb_fmt" "$comb_speed_fmt" "ETA: $time_fmt"
    else
        printf "%-12s %-10s %-10s %-14s\n" "" "" "Total:" "$total_comb_fmt"
    fi
    echo ""

else
    # ── Single-side path: original long table + shortened table ──
    print_long_table
    echo ""
    echo "--- Test Configuration ---"
    echo "Siever: $siever"
    echo "Side: $side_name"
    echo ""

    if [ -n "$mfbr" ] && [ -n "$rlim" ]; then
        rlambda_suggested=$(echo "scale=2; $mfbr / (l($rlim) / l(2))" | bc -l)
        echo "Suggested rlambda: $rlambda_suggested (mfbr=$mfbr / log2(rlim=$rlim))"
    fi
    if [ -n "$mfba" ] && [ -n "$alim" ]; then
        alambda_suggested=$(echo "scale=2; $mfba / (l($alim) / l(2))" | bc -l)
        echo "Suggested alambda: $alambda_suggested (mfba=$mfba / log2(alim=$alim))"
    fi

    echo ""
    echo "--- Shortened Table ---"
    print_short_table
    echo ""
fi

# ── result.job ──
cp input.job result.job
if [ "$side_name" = "algebraic" ]; then
    [ -n "$(tail -c1 result.job)" ] && echo "" >> result.job
    echo "lss: 0" >> result.job
elif [ "$sieve_both" = true ]; then
    # NOTE: remember to manually add "lss: 0" to result.job for the algebraic (-a) side run.
    echo ""
    echo "NOTE: Both sides were tested. Add 'lss: 0' to result.job for the algebraic side job."
fi

echo "--- result.job ---"
cat result.job
echo ""

# Cleanup temporary files
rm -f log.* output.*
