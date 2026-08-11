#!/bin/sh
# Byte-level golden gates captured from CADO makefb revision 0574bc39d.
# Together these cover ordinary roots, ramified/projective power ladders,
# legacy maxbits=1, and the maximum supported polynomial degree (8).
set -eu

tmp1=$(mktemp "${TMPDIR:-/tmp}/fbgen-c183-powers.XXXXXX")
tmp2=$(mktemp "${TMPDIR:-/tmp}/fbgen-c183-primes.XXXXXX")
tmp3=$(mktemp "${TMPDIR:-/tmp}/fbgen-octic.XXXXXX")
tmp4=$(mktemp "${TMPDIR:-/tmp}/fbgen-c147.XXXXXX")
trap 'rm -f -- "$tmp1" "$tmp2" "$tmp3" "$tmp4"' EXIT HUP INT TERM

check_hash()
{
    file=$1
    want=$2
    label=$3
    got=$(sha256sum "$file" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        printf 'FAIL   %-32s got %s want %s\n' "$label" "$got" "$want" >&2
        exit 1
    fi
    printf 'PASS   %-32s %s\n' "$label" "$got"
}

./fbgen --poly ../oracle/c183.poly --lim 10000 --maxbits 15 --threads 2 --out "$tmp1"
check_hash "$tmp1" d5e628b16c6a8f7329c8d474899645693cc8866969ea8f484b630ee10d7e72f1 \
    'GNFS, maxbits=15'

./fbgen --poly ../oracle/c183.poly --lim 1000000 --maxbits 1 --threads 2 --out "$tmp2"
check_hash "$tmp2" 4cf1c1bde338f073d62094d411b1d925e033ce21ad7407cd2a66b630e3cd5fd6 \
    'GNFS, legacy maxbits=1'

./fbgen --poly testdata/octic.poly --lim 100000 --threads 2 --out "$tmp3"
check_hash "$tmp3" 33bfee4610882c5d162d13c7c398461b1c790331c97f9e1fbabdd7a037216882 \
    'octic, default maxbits=15'

# The C147 timing factor base, whole, at the maxbits the frozen file uses.
# The three cases above are maxbits 15, 1 and 15, so none of them exercises the
# ladder at 14 -- a change there could pass every other gate and still silently
# alter oracle/c147.roots1, which is git-ignored and regenerated from this exact
# command. That would leave every device timing in RESULTS.md unreproducible
# with no signal. The hash is the one in oracle/MANIFEST.sha256. ~18 s.
# --lim is omitted on purpose: it must default to alim from the job file.
./fbgen --poly ../oracle/c147.job --maxbits 14 --threads 2 --out "$tmp4"
check_hash "$tmp4" 546a78a12d0310a8e3c221267067d347f338318d9ce01d95170f17a423a9745c \
    'C147 timing FB, maxbits=14'

printf '\nall fbgen gates passed\n'
