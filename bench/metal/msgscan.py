#!/usr/bin/env python3
"""Every user-facing string in the Metal port must name Metal, not CUDA.

9z-g got this to zero by hand and nothing kept it there. upstream dae6cca
then added three `cannot select CUDA device` messages to pipeline.cuh -- code
this port GENERATES from, so they arrived without anyone writing them here --
and the scan that would have caught them did not exist as a gate.

Two things this covers that 9z-g's did not. It scans printf and runlog_note
as well as fprintf(stderr, ...): --help is what cofcheck.sh classifies a
build from, and the run log's device record is what a project reads off a
failed task, and neither is stderr. And it joins ADJACENT STRING LITERALS
before matching, because every message long enough to be wrong is split
across source lines -- matching line by line finds nothing and passes.

The allowlist is by exact fragment, not by file and line, so re-wording an
allowed string makes you justify it again rather than silently inheriting
the exemption.
"""
import glob, os, re, sys

ALLOW = [
    ("cuda-sieve",
     "the project's own name"),
    ("CUDA_SIEVE_METAL",
     "environment variable names, which are spelled this way on every port"),
    ("is an NVIDIA measurement and no Apple",
     "deliberate provenance: the threshold was measured on NVIDIA parts"),
    ("a CUDA-tree property",
     "phase5_test naming the CUDA tree it compares this build against"),
]

CALL = re.compile(r"\b(?:fprintf\s*\(\s*stderr\s*,|printf\s*\(|runlog_note\s*\()")
LIT = re.compile(r'"((?:[^"\\]|\\.)*)"')
BAD = re.compile(r"CUDA|NVIDIA|cuda|nvidia")

# The shared .c/.cpp the METAL binary links (Makefile.metal's BENCH_CPUOBJ,
# with fbgen_lib.o built from fbgen.c). These live outside metal/ and so were
# invisible to a metal/-only scan -- including boinc_support.cpp, which is
# where 9z-g found one of its nine and which this build compiles with its own
# -DBENCH_BOINC_METAL_GPU selector. Scoping the gate to metal/ would have
# dropped exactly the file most likely to regress.
SHARED = ("fb_load.c", "verify_cpu.c", "poly.c", "primes.c", "rfb.c",
          "fb_cado.c", "platform.c", "boinc_support.cpp", "runlog.c",
          "watchdog.c", "fbgen.c")

def scan(root):
    pats = ("metal/*.cpp", "metal/*.inc", "metal/*.h", "metal/*.mm")
    files = sorted({f for p in pats for f in glob.glob(os.path.join(root, p))})
    if not files:                       # wrong root: say so before blaming
        print("msgscan: no metal/ sources found under %s" % root)
        return None
    for nm in SHARED:
        f = os.path.join(root, nm)
        # Named individually, not globbed: bench/ also holds CUDA-only sources
        # this binary never links, and flagging their messages would be wrong.
        if not os.path.exists(f):
            print("msgscan: shared source %s is missing -- the object list in"
                  " Makefile.metal has moved; fix SHARED here" % nm)
            return None
        files.append(f)
    files = sorted(set(files))
    bad = []
    for f in files:
        src = open(f, encoding="utf-8", errors="replace").read()
        for m in CALL.finditer(src):
            i, depth, j = m.end(), 1, m.end()
            while j < len(src) and depth:
                depth += (src[j] == "(") - (src[j] == ")")
                j += 1
            lits = "".join(LIT.findall(src[i:j]))     # adjacent literals joined
            spans = [(k, k + len(frag))
                     for frag, _ in ALLOW
                     for k in range(len(lits))
                     if lits.startswith(frag, k)]
            for w in BAD.finditer(lits):
                if any(a <= w.start() < b for a, b in spans):
                    continue
                line = src[: m.start()].count("\n") + 1
                ctx = lits[max(0, w.start() - 30): w.start() + 34]
                bad.append((os.path.relpath(f, root), line, ctx.replace("\\n", " ")))
    # An allowlist entry that matches nothing is dead config, and this tree's
    # most-repeated failure is an anchor that silently matches nothing. Every
    # exemption must still be EARNED by a string that exists.
    seen = set()
    for f in files:
        src = open(f, encoding="utf-8", errors="replace").read()
        for frag, _ in ALLOW:
            if frag in src:
                seen.add(frag)
    stale = [(frag, why) for frag, why in ALLOW if frag not in seen]
    return bad, len(files), stale

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    r = scan(root)
    if r is None:
        sys.exit(1)
    bad, nfiles, stale = r
    for frag, why in stale:
        print("FAIL   stale allowlist entry %r (%s): nothing matches it any"
              " more, so drop it" % (frag, why))
    for f, line, ctx in bad:
        print("FAIL   %-26s:%-5d names CUDA/NVIDIA: ...%s..." % (f, line, ctx))
    if bad or stale:
        print("msgscan: %d CUDA-named user-facing string(s), %d stale"
              " allowlist entr(ies)." % (len(bad), len(stale)))
        print("         Fix it in the GENERATOR, never in the shared .cuh/.cu --"
              " the two builds")
        print("         deliberately differ in what they log (CLAUDE.md, 9c).")
        sys.exit(1)
    print("PASS   %-34s %d files, %d allowed fragments"
          % ("no CUDA-named user output", nfiles, len(ALLOW)))
