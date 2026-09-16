#!/usr/bin/env python3
"""Measure how many relations survive filtering a GGNFS relation set down to a
lower (lpbr, lpba) -- the sampling half of finding 99's filter-down experiment.

Filtering DOWN is exact: a run at the higher lpb/mfb examines a superset of the
survivors a lower run would examine.  Simulating UPWARD is not valid.

GGNFS relation line:  a,b:<rational primes, hex>:<algebraic primes, hex>
Both prime fields carry the FULL factorisation of the norm, factor-base primes
included -- not just the large primes.  That matters for --mfb below.

Usage:  filterdown.py [--mfb MFBR,MFBA] [--lim RLIM,ALIM] [--n N] [DIR]
  --mfb  also drop relations whose per-side COFACTOR exceeds these bit counts.
         Without it the result is a superset of a true lower run, because the
         higher run carried a larger mfb.  Differencing the two measures the
         mfb knob on its own.  Requires --lim: mfb bounds what is left after
         the factor base (primes <= lim) is divided out, so the cofactor is
         the product of the primes ABOVE lim -- not the product of the line.
"""
import argparse, collections, glob, os, subprocess, sys

AP = argparse.ArgumentParser()
AP.add_argument("dir", nargs="?",
                default=os.path.expanduser("~/code/ggnfs-distributed/AS276/rels"))
AP.add_argument("--n", type=int, default=12, help="work units to sample")
AP.add_argument("--mfb", default="", help="MFBR,MFBA bit caps on the cofactor")
AP.add_argument("--lim", default="", help="RLIM,ALIM factor-base bounds; required with --mfb")
AP.add_argument("--sqside", choices=("r", "a"), default=None,
                help="side carrying the special-q ('r' or 'a'); required with --mfb")
AP.add_argument("--lpb-range", default="30,33,30,35",
                help="rmin,rmax,amin,amax to tabulate")
a = AP.parse_args()
rmin, rmax, amin, amax = (int(x) for x in a.lpb_range.split(","))
mfbr, mfba = ((int(x) for x in a.mfb.split(",")) if a.mfb else (0, 0))
if a.mfb and not a.sqside:
    sys.exit("--mfb needs --sqside r|a.  The special-q divides the norm on its "
             "own side and is NOT part of the cofactor mfb bounds, but it IS "
             "listed among that side's primes.  Whenever q exceeds that side's "
             "lim it would otherwise be multiplied into the cofactor and inflate "
             "it by ~28 bits, so most relations above q = lim get rejected as "
             "false mfb violations.  GGNFS job files record this as `lss`: "
             "0 = algebraic, 1 = rational.")
if a.mfb and not a.lim:
    sys.exit("--mfb needs --lim RLIM,ALIM: the cofactor is the product of the "
             "primes above the factor base, and the relation line does not say "
             "where the factor base ended.  Use the job file's rlim/alim.")
rlim, alim = ((int(x) for x in a.lim.split(",")) if a.lim else (0, 0))

files = sorted(glob.glob(f"{a.dir}/*.zst"))
if not files:
    sys.exit(f"no .zst work units under {a.dir}")
step = max(len(files) // a.n, 1)
picked = files[::step][:a.n]

SQ_MIN, SQ_HITS = 10_000_000, 20

def special_qs(lines, side):
    """The special-q values in one file, found by frequency rather than by any
    filename-to-q convention -- the corpora here mix GGNFS work units with
    cuda-sieve blocks 45x larger, and the two do not share a naming scheme.

    Every relation is found BY its special-q, so each q recurs across dozens of
    relations on its own side while an incidental large prime shows up once or
    twice.  The two thresholds are absolute on purpose: an earlier version
    scaled the hit floor with file size, which silently found nothing in the
    big cuda-sieve blocks and let their special-q back into the cofactor."""
    cnt = collections.Counter()
    for rp, ap_ in lines:
        for v in (rp if side == "r" else ap_):
            if v > SQ_MIN:
                cnt[v] += 1
    seed = {v for v, c in cnt.items() if c >= SQ_HITS}
    if not seed:
        return seed
    # The hit count alone misses low-yield special-q: a small GGNFS work unit
    # shows a clean gap (49 values at >=38 hits, none between 3 and 19), but a
    # 180k-relation cuda-sieve block has ~2600 above the threshold and ~180
    # sitting just under it.  Those are real special-q, and leaving them in the
    # cofactor over-rejects their relations.  So use the seed only to locate
    # the file's q WINDOW, then take every prime inside it: the window is
    # narrow (one work unit spans 1000), so an unrelated large prime landing in
    # it is rare, and the seed proves these are the q values being sieved.
    lo, hi = min(seed), max(seed)
    return {v for v in cnt if lo <= v <= hi}

def cofactor_bits(primes, lim, drop):
    """bits in the product of the primes above the factor base, with the
    special-q divided back out -- which is what mfb actually bounds"""
    n = 1
    seen = False
    for v in primes:
        if v > lim:
            if not seen and v in drop:
                seen = True          # remove exactly one copy of the special-q
                continue
            n *= v
    return n.bit_length()

tot = 0
nosq = []
cnt = collections.Counter()
for f in picked:
    parsed = []
    for line in subprocess.run(["zstd", "-dc", f], capture_output=True,
                               text=True).stdout.splitlines():
        p = line.split(":")
        if len(p) != 3:
            continue
        try:
            parsed.append(([int(x, 16) for x in p[1].split(",") if x],
                           [int(x, 16) for x in p[2].split(",") if x]))
        except ValueError:
            continue
    drop = special_qs(parsed, a.sqside) if a.sqside else set()
    if a.sqside and not drop and parsed:
        nosq.append(os.path.basename(f))
    for rp, ap_ in parsed:
        tot += 1
        r, g = max((v.bit_length() for v in rp), default=0), \
               max((v.bit_length() for v in ap_), default=0)
        cr = cofactor_bits(rp, rlim, drop if a.sqside == "r" else ()) if mfbr else 0
        ca = cofactor_bits(ap_, alim, drop if a.sqside == "a" else ()) if mfba else 0
        for br in range(rmin, rmax + 1):
            if r > br or (mfbr and cr > mfbr):
                continue
            # lpba < lpbr is a legal config; honour the amin the caller asked
            # for instead of silently clamping it to br
            for ba in range(amin, amax + 1):
                if g > ba or (mfba and ca > mfba):
                    continue
                cnt[(br, ba)] += 1

print(f"{tot} relations from {len(picked)} files spread over {a.dir}")
if nosq:
    print(f"WARNING: no special-q found in {len(nosq)} of {len(picked)} files "
          f"({', '.join(nosq[:3])}{'...' if len(nosq) > 3 else ''}) -- their "
          f"cofactors still include it and will over-reject.  Check --sqside.")
print(f"mfb caps: {a.mfb or 'none (result is a SUPERSET of a true lower run)'}"
      f"{'  against lim ' + a.lim if a.lim else ''}"
      f"{', special-q on the ' + {'r':'RATIONAL','a':'ALGEBRAIC'}[a.sqside] + ' side' if a.sqside else ''}\n")
if not cnt:
    sys.exit("no relation survives any config in the requested range -- check "
             "--lpb-range, and check --lim against the job file if --mfb is set")
top = max(cnt, key=lambda k: cnt[k])
if a.mfb:
    print(f"NOTE: the --mfb cap is the SAME for every row, so only the row whose own\n"
          f"      job runs at mfb {a.mfb} is a true lower run.  Every other row pairs\n"
          f"      that config's lpb with a different config's mfb.  Real jobs scale mfb\n"
          f"      with lpb, so compare within a column, and re-run per config to get a\n"
          f"      matched pair.\n")
print(f"{'config':<9}{'relations':>12}{'share':>9}{'vs ' + f'{top[0]}/{top[1]}':>11}")
for k in sorted(cnt):
    print(f"{k[0]}/{k[1]:<7}{cnt[k]:>12}{100*cnt[k]/tot:>8.1f}%{cnt[k]/cnt[top]:>11.3f}")
