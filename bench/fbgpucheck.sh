#!/bin/sh
# GPU/CPU factor-base equivalence gate.  This is intentionally not part of
# `make check`: it needs a CUDA device.
#
# Each file case gates the strongest serialized representation first: native
# fbgen and fbgen_gpu must emit byte-identical text files, then both are loaded
# through fb_load_cado() and compared.  Separate memory cases below exercise the
# production afb_build_gpu() consumer of the same shared GPU generation engine.
set -eu

scale=1.925
device=${FBGPU_DEVICE:-0}
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/fbgpucheck.XXXXXX")
trap 'rm -rf -- "$tmpdir"' EXIT HUP INT TERM

run_case()
{
    poly=$1
    lim=$2
    maxbits=$3
    slug=$4
    label=$5
    cpu="$tmpdir/$slug.cpu.roots1"
    gpu="$tmpdir/$slug.gpu.roots1"

    ./fbgen --poly "$poly" --lim "$lim" --maxbits "$maxbits" --threads 2 --out "$cpu"
    ./fbgen_gpu --poly "$poly" --lim "$lim" --maxbits "$maxbits" \
        --device "$device" --scale "$scale" --out "$gpu" --compare-fb "$cpu"
    cmp -s "$cpu" "$gpu"
    printf 'PASS   %s (byte-identical + loaded fb_t)\n' "$label"
}

run_memory_case()
{
    poly=$1
    lim=$2
    maxbits=$3
    slug=$4
    label=$5
    cpu="$tmpdir/$slug.cpu.roots1"

    [ -f "$cpu" ] ||
        ./fbgen --poly "$poly" --lim "$lim" --maxbits "$maxbits" --threads 2 --out "$cpu"
    ./fbgen_gpu --poly "$poly" --lim "$lim" --maxbits "$maxbits" \
        --device "$device" --scale "$scale" --complete --compare-fb "$cpu"
    printf 'PASS   %s (in-memory production fb_t)\n' "$label"
}

# Tiny legal bounds exercise the nbase=0 / no-composite-marking boundary.
run_case ../oracle/c183.poly 2 15 lim2 'lim=2 boundary'
run_case ../oracle/c183.poly 3 15 lim3 'lim=3 boundary'

# Exercise every supported degree.  Truncating the synthetic octic preserves a
# valid nonzero leading coefficient at each degree and crosses the CAP=6/CAP=8
# dispatch boundary explicitly at degrees 6 and 7.
for deg in 1 2 3 4 5 6 7 8; do
    dpoly="$tmpdir/degree-$deg.poly"
    awk -v maxdeg="$deg" '
        /^c[0-9][0-9]*:/ {
            k=$1; sub(/^c/, "", k); sub(/:$/, "", k);
            if ((k + 0) > maxdeg) next;
            if ((k + 0) == maxdeg) $0 = "c" maxdeg ": 1";
        }
        { print }
    ' testdata/octic.poly > "$dpoly"
    run_case "$dpoly" 5000 15 "degree-$deg" "degree-$deg kernel dispatch"
done

run_case ../oracle/c183.poly 10000 15 c183-powers 'GNFS powers/ramification'
run_case ../oracle/c183.poly 1000000 1 c183-primes 'GNFS prime-only'
run_case testdata/octic.poly 100000 15 octic 'octic degree-8'
run_case ../oracle/c147.job 1000000 14 c147 'C147 degree-5'

# Exercise both template consumers of gpu_fb_generate_complete().  The degree-6
# and degree-8 cases cross the CAP dispatch boundary; c183 also verifies the
# production sink's exact/Hensel merge rather than only the text serializer.
run_memory_case "$tmpdir/degree-6.poly" 5000 15 degree-6 'degree-6 production consumer'
run_memory_case "$tmpdir/degree-8.poly" 5000 15 degree-8 'degree-8 production consumer'
run_memory_case ../oracle/c183.poly 10000 15 c183-powers 'GNFS powers production consumer'

# The uint32 compact-root representation requires a hard segment cap.  This
# rejection happens before CUDA allocation, and must remain fail-closed.
if ./fbgen_gpu --poly ../oracle/c183.poly --lim 100 --maxbits 15 \
       --segment-odds 477218589 --device "$device" --out "$tmpdir/too-wide.roots1" \
       >/dev/null 2>&1; then
    echo 'FAIL   oversized --segment-odds was accepted' >&2
    exit 1
fi
printf 'PASS   oversized --segment-odds rejected\n'

# A failed --compare-fb must not publish the staged GPU file over an existing
# destination. Use a deliberately different maxbits oracle to force mismatch.
guard="$tmpdir/publish-guard.roots1"
badref="$tmpdir/publish-badref.roots1"
./fbgen --poly ../oracle/c183.poly --lim 10000 --maxbits 15 --threads 2 --out "$guard"
./fbgen --poly ../oracle/c183.poly --lim 10000 --maxbits 1 --threads 2 --out "$badref"
before=$(sha256sum "$guard" | cut -d' ' -f1)
if ./fbgen_gpu --poly ../oracle/c183.poly --lim 10000 --maxbits 15 \
       --device "$device" --out "$guard" --compare-fb "$badref" \
       >/dev/null 2>&1; then
    echo 'FAIL   mismatched --compare-fb unexpectedly passed' >&2
    exit 1
fi
after=$(sha256sum "$guard" | cut -d' ' -f1)
[ "$before" = "$after" ] || {
    echo 'FAIL   failed --compare-fb replaced the existing output' >&2
    exit 1
}
[ ! -e "$guard.part" ] || {
    echo 'FAIL   failed --compare-fb left a .part file' >&2
    exit 1
}
printf 'PASS   failed --compare-fb preserves published output\n'

printf '\nall GPU/CPU factor-base file equivalence gates passed\n'
