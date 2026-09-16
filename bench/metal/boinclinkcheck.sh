#!/bin/sh
# Phase 9 gate: is the HAVE_BOINC binary actually distributable?
#
# Every check here is for something that BUILDS AND LINKS with exit 0 and is
# still wrong (plan 9a). None of these can be caught by compiling.
#
#   usage: sh metal/boinclinkcheck.sh <binary> <expected-minos>
set -u
BIN=${1:?usage: boinclinkcheck.sh <binary> <minos>}
WANT_MINOS=${2:?usage: boinclinkcheck.sh <binary> <minos>}
fail=0
pass() { printf 'PASS   %s\n' "$1"; }
bad()  { printf 'FAIL   %s\n' "$1"; fail=1; }

# 1. No BOINC dylib. Without --disable-shared, -lboinc_api takes an installed
#    .dylib and the binary carries this machine's absolute path to it.
if otool -L "$BIN" | tail -n +2 | grep -qi 'boinc'; then
    bad "links a BOINC dylib -- configure BOINC with --disable-shared"
    otool -L "$BIN" | tail -n +2 | grep -i boinc | sed 's/^/       /'
else
    pass "no BOINC dylib: the library is linked statically"
fi

# 2. Nothing outside /usr/lib and /System. Anything else is a path that exists
#    on the build host and not on a volunteer's.
strays=$(otool -L "$BIN" | tail -n +2 | awk '{print $1}' \
         | grep -v '^/usr/lib/' | grep -v '^/System/')
if [ -n "$strays" ]; then
    bad "depends on libraries outside /usr/lib and /System:"
    printf '%s\n' "$strays" | sed 's/^/       /'
else
    pass "every dependency ships with macOS"
fi

# 3. The deployment target. A default-built BOINC stamps its objects with the
#    build host's SDK; linking those raises the whole binary's floor.
got=$(otool -l "$BIN" | grep -A4 LC_BUILD_VERSION | awk '/minos/{print $2; exit}')
if [ "$got" = "$WANT_MINOS" ]; then
    pass "minos $got matches METAL_MIN_MACOS"
else
    bad "minos $got, want $WANT_MINOS -- build BOINC with -mmacosx-version-min=$WANT_MINOS"
fi

# 4. The shaders are inside the executable. A BOINC project ships ONE file;
#    a bench.metallib that has to land beside it and match it is a failure
#    mode (missing, stale, mismatched) that embedding removes outright.
sec=$(otool -l "$BIN" | grep -A4 'sectname __metallib' | awk '/size/{print $2; exit}')
if [ -n "${sec:-}" ]; then
    pass "carries its own shaders in __DATA,__metallib ($((sec)) bytes)"
else
    bad "no embedded metallib -- this binary needs a bench.metallib beside it"
fi

# 5. The BOINC client API is really in there. A link that resolved nothing
#    looks identical from the outside.
for sym in _boinc_init_parallel _boinc_finish _boinc_fraction_done; do
    if nm "$BIN" 2>/dev/null | grep -q " T $sym\$"; then
        pass "$sym linked in"
    else
        bad "$sym missing -- boinc_api.o was never pulled from the archive"
    fi
done

# 6. cofcheck.sh classifies builds by this exact string, and a HAVE_BOINC
#    build takes the OTHER branch of that #ifdef. Misclassified as CUDA, it
#    runs the --ecm-b1 400000 case that crashed WindowServer twice (plan 8k).
#
#    SAFE TO GREP HERE because usage() is printf(), i.e. STDOUT. Under
#    HAVE_BOINC the runtime redirects stderr to stderr.txt in the working
#    directory as soon as boinc_init runs -- which is before argument parsing
#    -- so a check that greps stderr from a pipe finds nothing and "passes"
#    for the wrong reason. Assert on stderr only by reading stderr.txt.
if "$BIN" --help 2>&1 | grep -q 'select Metal device'; then
    pass "--help still marks this as the Metal build (cofcheck.sh reads this)"
else
    bad "--help does not say 'select Metal device' -- cofcheck.sh would run the"
    printf '       %s\n' "--ecm-b1 400000 case this build must refuse"
fi

echo
# Everything above is checkable from the binary alone. These are not, and the
# gate does not pretend otherwise: under HAVE_BOINC they land in stderr.txt in
# the run's working directory, never on the terminal, so they have to be read
# there by hand after a real band.
cat <<'NOTE'
CHECK BY HAND in the run's stderr.txt (BOINC redirects stderr there, so none
of these reach a terminal or a 2>&1 pipe):
  - "BOINC: running on Metal device N of M: <name>" -- names Metal, not CUDA,
    and carries the ordinal the client assigned
  - "BOINC: no usable GPU assignment in init_data.xml" -- only when the app
    version's plan class declares no GPU coprocessor
  - the cofq_init launch-bound advisory, and the --ecm-b1 refusal
  - runlog_warn's stderr half
NOTE
echo
if [ "$fail" = 0 ]; then echo "BOINC LINK GATE: PASS"; else echo "BOINC LINK GATE: FAIL"; fi
exit $fail
