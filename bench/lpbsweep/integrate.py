#!/usr/bin/env python3
"""Reproduce RESULTS finding 98's chart from bands.tsv.

Integrates relations and time over a q-window as int rel(q)/ln(q) dq, with
rel(q) and wall(q) log-linearly interpolated between measured bands (and
extrapolated on the end slopes).  Scans qmin for the cheapest window that
reaches the relation target under the duplicate-control rule qmax <= 6*qmin,
refusing to extrapolate below the shallowest measured band or past 2x the
deepest one.  Repeated runs of the same (config, band) are averaged.

Relation targets are an INPUT.  The per-step growth R is NOT: finding 99
measures it off 4432 completed NFS@Home jobs.  Controlled on Murphy-E rather
than digits, GNFS and SNFS agree at 1.62-1.74x; Kyle's 1.75-1.80 rule of thumb
sits just above that.  R_LOW and R_RULE bracket the measured range and both
are run, because the level margin varies from under 1% to 9% across it.
"""
import csv, math, os
from collections import defaultdict

W = os.path.dirname(os.path.abspath(__file__))

# Per-config relation targets, in UNIQUE relations.
#
# R = the multiplier on required relations for a +1 lockstep step (BOTH sides).
# Finding 99 measures it off 4432 completed NFS@Home jobs:
# Controlled on Murphy-E -- the difficulty measure that means something for
# BOTH sieve types -- the two arms agree:
#     GNFS   1.739 OLS / 1.623 within-E-band
#     SNFS   1.696 OLS / 1.651 within-E-band
# Controlled on DIGITS instead they disagree wildly (GNFS 1.52, SNFS 1.79),
# because digits is 0.832 collinear with lpb for GNFS and carries no
# information at all for SNFS.  That split, not any physics, produced the
# GNFS-is-lower story earlier revisions told.
#     carry                  1.62-1.74   -- centred ~1.67
#     Kyle's rule of thumb   1.75-1.80   -- just above; mildly conservative
#     ideal-count model      1.94        -- an upper bound only; a job never
#                                           saturates the large-prime universe
# c183 flips at R = 1.632 and c194 at 1.612, both at the BOTTOM edge of the
# range, so the lower rung wins across essentially all of it and finding 98's
# level ranking stands -- though by under 1% at the bottom end.  The balance
# and mfb-cap results do not depend on R at all: balance is a pure ratio, and
# the cap arms share a target.
R_LOW, R_RULE = 1.62, 1.74

# bench counts RAW relations; the targets above are UNIQUE ones.  The unique
# fraction falls with lpb -- GNFS-only fit off Kyle's dedup sheet (finding 99)
# gives -0.98 pt per +1 lpb, which against that sheet's 59.3% level is a 1.017x
# raw premium per rung.  (The pooled GNFS+SNFS fit gives -1.96 pt and 1.035x,
# but SNFS is 6 of its 40 points with a 4x steeper slope and these targets are
# GNFS jobs, so the pooled number does not belong here.)  The level correction
# cancels at the anchor; only the differential survives.
DUP_PER_RUNG = 1.017

def _lpbmax(label):
    """lpb_max of a config label: '33/34@96' -> 34.  This, not ladder position,
    is what sets a config's duplicate rate -- so it is what DUP_PER_RUNG keys
    on.  Keying on ladder index let gap-2 and ideal-count targets, which are
    merged in rather than generated, silently escape the correction."""
    return int(label.split("@")[0].split("/")[1])

def _raw(unique, anchor):
    """convert UNIQUE targets to the RAW counts bench measures, anchored so the
    correction is exactly 1 at lpb_max == anchor"""
    return {lab: n * DUP_PER_RUNG ** (_lpbmax(lab) - anchor)
            for lab, n in unique.items()}

def _rungs(base, ladder, R):
    """base = UNIQUE target at rung 0; ladder = configs per rung, ascending"""
    return {lab: base * R ** k
            for k, labs in enumerate(ladder) for lab in labs}

def _N(x):                       # large-prime ideals below 2^x
    return 2 ** x / (x * math.log(2))

C183 = [["31/32"], ["32/33"]]
C194 = [["32/33"], ["33/34", "33/34@96"]]
# AS276 rung 0 is 33/34.  33/35 is a gap-2 config, not a lockstep rung, so it
# keeps Kyle's own independently derived 1.80B unique under every scenario --
# but it is still an lpb_max 35 config, so it takes the same raw-vs-unique
# correction its lockstep peers at that level take.
AS276 = [["33/34", "33/34@96"], ["34/35", "34/35@96"], ["35/36"]]
A183, A194, A276 = 32, 33, 34        # lpb_max of each ladder's rung 0

TARGETS = {
    "c183": {
        "R = 1.62 (low end)": _raw(_rungs(350e6, C183, R_LOW), A183),
        "R = 1.74 (top of range)": _raw(_rungs(350e6, C183, R_RULE), A183)},
    "c194": {
        "R = 1.62 (low end)": _raw(_rungs(650e6, C194, R_LOW), A194),
        "R = 1.74 (top of range)": _raw(_rungs(650e6, C194, R_RULE), A194)},
    "as276": {
        "R = 1.62 (low end)":
            _raw(dict(_rungs(1.35e9, AS276, R_LOW), **{"33/35": 1.80e9}), A276),
        "R = 1.74 (top of range)":
            _raw(dict(_rungs(1.35e9, AS276, R_RULE), **{"33/35": 1.80e9}), A276),
        "ideal-count model (upper bound)": _raw({
            "33/34": _N(33) + _N(34), "33/34@96": _N(33) + _N(34),
            "33/35": _N(33) + _N(35),
            "34/35": _N(34) + _N(35), "34/35@96": _N(34) + _N(35),
            "35/36": _N(35) + _N(36)}, A276)},
}
# The config each ladder's target is pinned to -- rung 0, which is also the
# lpb_max the dup correction is anchored at (A183/A194/A276).  AS276's was
# left at "34/35" from when its target was 1.85B pinned there; the revised
# sizing pins 1.35e9 to rung 0 "33/34", so the star belonged one row up.
BASELINE = {"c183": "31/32", "c194": "32/33", "as276": "33/34"}
SPAN = 6.0
# How far past the deepest measured band the scan will extrapolate.  2.0 is the
# default and is what every published figure uses.  Raise it with
# `EXTRAP=3.0 python3 integrate.py` to test how much a verdict depends on the
# extrapolation -- which matters for AS276, whose windows run well past its
# deepest band while c183's and c194's do not.
#
# Finding 99 checks the decay model against Kyle's real 1.67-billion-relation
# GGNFS run of this exact job (346,859 work units, q = 80-427M, exact per-q
# counts).  Two things came out of it.  The real yield curve is NOT a power
# law: it is flat to ~250M and then steepens, reaching q^-0.31 over 300-427M,
# against the q^-0.27..-0.28 this harness fits at A=31.  So the extrapolation
# is well calibrated in the deep region, if anything slightly OPTIMISTIC, and
# raising EXTRAP is not free.  And integrating these bands over the real job's
# own q-range projects 1,167M relations at A=31 against its 1,671M at A=32, a
# ratio of 1.432 where finding 98 independently measures A=32 buying ~1.5x.
EXTRAP = float(os.environ.get("EXTRAP", 2.0))

raw = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f"{W}/bands.tsv"), delimiter='\t'):
    if not r.get('rel_q') or not r.get('wall_ms_q'):
        continue                      # guard BOTH numeric fields, not just rel_q
    job, lab = r['job'], r['lpb']
    if job.endswith('cap'):
        job, lab = job[:-3], lab + '@' + r['mfb'].split('/')[1]
    raw[(job, lab)][float(r['band']) * 1e6].append(
        (float(r['wall_ms_q']), float(r['rel_q'])))
# repeats are averaged, not last-write-wins: 9 configs were run twice and the
# published verdict for one of them flips on TSV row order otherwise
d = defaultdict(dict)
for k, bands in raw.items():
    for q, runs in bands.items():
        d[k][q] = (sum(w for w, _ in runs) / len(runs),
                   sum(r for _, r in runs) / len(runs))

def interp(pts, q):
    xs = sorted(pts); L = [math.log(x) for x in xs]; lq = math.log(q)
    for i in range(len(xs) - 1):
        if lq <= L[i + 1] or i == len(xs) - 2:
            t = (lq - L[i]) / (L[i + 1] - L[i]); a, b = pts[xs[i]], pts[xs[i + 1]]
            return (a[0] + t * (b[0] - a[0]), max(a[1] + t * (b[1] - a[1]), 0.05))
    return pts[xs[-1]]

def window(pts, q0, q1):
    rel = ms = 0.0; q = q0
    while q < q1:
        s = min(q * 0.005, q1 - q); w, rl = interp(pts, q); n = s / math.log(q)
        rel += rl * n; ms += w * n; q += s
    return rel, ms

def cheapest(pts, N):
    """cheapest SPAN-ratio window reaching N; (qmin, qmax, days, pin) or None.

    pin is '' for an interior optimum, or 'lo'/'hi' when the best window sits
    against an end of the admissible qmin range rather than at a minimum the
    scan actually found.  A pinned optimum is a measurement-extent artifact:
    the true best window is outside the bands we measured, the reported cost is
    a lower bound on it, and a small target increase flips the config straight
    to NO WINDOW with no intermediate degradation.  Report it, do not rank on
    it."""
    hi, lo_band = max(pts), min(pts)
    best = None
    qmin = lo_band            # symmetric with the qmax guard: never start the
    last = qmin               # window below the shallowest measured band
    while qmin < 600e6:
        qmax = qmin * SPAN
        if qmax > EXTRAP * hi:
            break
        last = qmin
        if window(pts, qmin, qmax)[0] >= N:
            lo, h = qmin, qmax
            for _ in range(50):                    # shrink qmax to exactly hit N
                m = (lo + h) / 2
                if window(pts, qmin, m)[0] < N: lo = m
                else: h = m
            days = window(pts, qmin, h)[1] / 1000 / 86400
            if best is None or days < best[2]:
                best = (qmin, h, days)
        qmin *= 1.02
    if best is None:
        return None
    pin = "hi" if best[0] >= last / 1.05 else "lo" if best[0] <= lo_band * 1.05 else ""
    return best + (pin,)

def report(job, targets, note=""):
    basel = BASELINE[job]
    cfgs = sorted((k for k in d if k[0] == job),
                  key=lambda k: (int(k[1].split('/')[0]), '@' in k[1]))
    rows = {}
    for k in cfgs:
        if len(d[k]) < 2 or k[1] not in targets:
            continue
        rows[k[1]] = (cheapest(d[k], targets[k[1]]), targets[k[1]])
    # cheapest() documents that an edge-pinned optimum is a lower bound and
    # must not be ranked on.  Honour that here: rank among unpinned rows, and
    # only fall back to pinned ones when nothing else has a window -- in which
    # case the winner is flagged rather than presented as a measurement.
    clean = [(l, b[2]) for l, (b, _) in rows.items() if b and b[3] != "hi"]
    feas = [(l, b[2]) for l, (b, _) in rows.items() if b]
    ranked = clean or feas
    winner = min(ranked, key=lambda x: x[1])[0] if ranked else None
    winner_pinned = bool(winner and not clean)
    print(f"--- {job}{note} ---")
    print(f"{'config':<11} {'gap':>3} {'target':>8} {'qmin':>6} {'qmax':>7} {'GPU-days':>9}")
    for lab in sorted(rows, key=lambda l: (int(l.split('/')[0]), '@' in l)):
        b, N = rows[lab]
        gap = int(lab.split('/')[1][:2]) - int(lab.split('/')[0])
        star = "*" if lab == basel else " "
        mark = ("  <-- best (PINNED, lower bound only)" if lab == winner and winner_pinned
                else "  <-- best" if lab == winner else "")
        if b is None:
            print(f"{lab:<11}{star}{gap:>2} {N/1e6:>7.0f}M {'--':>6} {'--':>7} {'NO WINDOW':>9}")
        else:
            flag = {"hi": "!", "lo": "v"}.get(b[3], " ")
            print(f"{lab:<11}{star}{gap:>2} {N/1e6:>7.0f}M {b[0]/1e6:>5.0f}M "
                  f"{b[1]/1e6:>6.0f}M {b[2]:>9.1f}{flag}{mark}")
    pins = {b[3] for b, _ in rows.values() if b and b[3]}
    if any(b is None for b, _ in rows.values()):
        print(f"  NO WINDOW = no {SPAN:.0f}x q-window inside the measured extent"
              f" (<= {EXTRAP:.0f}x the deepest band, {max(max(d[k]) for k in d if k[0]==job)/1e6:.0f}M)"
              " reaches target.\n"
              "              That is a limit of what was benchmarked, NOT a verdict"
              " that the config cannot finish.")
    if "hi" in pins:
        print("  ! optimum pinned against the DEEP edge of the measured q-range:"
              " the real optimum is outside it,\n"
              "    the cost shown is a lower bound, and a small target increase"
              " flips the row straight to NO WINDOW.")
    if "lo" in pins:
        print("  v optimum sits at the shallowest measured band -- sieving lower"
              " could only make it cheaper,\n"
              "    so the cost shown is an upper bound.")
    print()

if __name__ == "__main__":       # window.py imports the loader and the targets
    for job in ("c183", "c194", "as276"):
        for name, t in TARGETS[job].items():
            report(job, t, f"  [{name}]")
