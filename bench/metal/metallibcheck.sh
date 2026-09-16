#!/bin/sh
# Phase 9 gate: does the binary carry its own shaders and run on them alone?
#
# The check that matters is NEGATIVE SPACE: run in a directory with no
# bench.metallib, with $CUDA_SIEVE_METALLIB unset, so the only library the
# process can possibly find is the one inside it. Every other gate here exports
# CUDA_SIEVE_METALLIB, so none of them would notice embedding being broken.
#
#   usage: metallibcheck.sh <binary> <poly> <fb> [control]
# "control" asserts the OPPOSITE -- that a non-embedded binary fails the same
# way -- so the positive case is not a tautology.
set -u
BIN=$1; POLY=$2; FB=$3; MODE=${4:-embedded}
fail=0
pass() { printf 'PASS   %s\n' "$1"; }
bad()  { printf 'FAIL   %s\n' "$1"; fail=1; }

# -A4, not -A2: otool prints sectname, segname, addr, THEN size. With -A2
# this found nothing and reported "nothing was embedded" about a binary
# that had just sieved 37 relations out of its own embedded library.
SEC=$(otool -l "$BIN" | grep -A4 'sectname __metallib' | awk '/size/{print $2; exit}')
if [ "$MODE" = control ]; then
    if [ -n "${SEC:-}" ]; then
        bad "control binary has a __metallib section; it was built with EMBED_METALLIB=1"
    else
        pass "control: no __metallib section, as intended"
    fi
else
    if [ -n "${SEC:-}" ]; then
        pass "__DATA,__metallib present ($((SEC)) bytes)"
    else
        bad "no __DATA,__metallib section -- nothing was embedded"
    fi
fi

# A directory with no metallib in it, and no env var. This is the shape of a
# BOINC slot: the project sent one file.
TMP=$(mktemp -d) || exit 2
cp "$BIN" "$TMP/bench"
if [ -e "$TMP/bench.metallib" ]; then bad "test dir is not clean"; fi

out=$(cd "$TMP" && env -u CUDA_SIEVE_METALLIB ./bench --pipeline \
        --cadofb "$FB" --poly "$POLY" --qrange 120000053:120000053 \
        --allowance 101.6 --allowance0 68.1 --cofactor 2>&1)
rc=$?
rel=$(printf '%s' "$out" | awk '/total relations/{n=$NF} END{print n+0}')

if [ "$MODE" = control ]; then
    if [ "$rc" = 0 ]; then
        bad "control RAN with no metallib anywhere -- the gate proves nothing"
    else
        pass "control fails with no library to find (exit $rc)"
    fi
    printf '%s\n' "$out" | grep -i 'shader library' | head -2 | sed 's/^/       /'
else
    if [ "$rc" != 0 ]; then
        bad "run failed (exit $rc) with no metallib on disk"
        printf '%s\n' "$out" | tail -5 | sed 's/^/       /'
    elif [ "$rel" != 37 ]; then
        bad "ran, but $rel relations at the parity q, want 37"
    else
        pass "sieved from the embedded library alone: 37 relations at the parity q"
    fi
fi
rm -rf "$TMP"

echo
if [ "$fail" = 0 ]; then echo "METALLIB EMBED GATE: PASS"; else echo "METALLIB EMBED GATE: FAIL"; fi
exit $fail
