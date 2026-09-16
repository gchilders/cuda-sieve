#!/usr/bin/env python3
"""Finding 99: measure the required-relation multiplier R for a lockstep lpb
step, off NFS@Home's completed-job history instead of assuming it.

Inputs (neither is in this repo, and neither is guaranteed to stay put):
  ~/code/paste-scraper/data/parsed_results.json   4773 parsed msieve logs
  ~/code/paste-scraper/data/results.csv           NFS@Home page metadata
  dedup.tsv (beside this script)                  Kyle's unique-fraction sheet,
      exported from the two tabs of the Google Sheet named in finding 99

The join key is the metadata column `bits`, which is max(lpba, lpbr).  A step
in `bits` is therefore a step in BOTH sides with the gap preserved, which is
exactly the "full step up" the 75-80% rule of thumb is about.
"""
import csv, json, math, os, random, statistics, collections

SRC = os.path.expanduser("~/code/paste-scraper/data")
HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    meta = {r["paste_id"]: r for r in csv.DictReader(open(f"{SRC}/results.csv"))}
    out = []
    for r in json.load(open(f"{SRC}/parsed_results.json")):
        m = meta.get(r["paste_id"])
        if not m or not m.get("bits"):
            continue
        try:
            lpb = int(m["bits"])
        except ValueError:
            continue
        def g(*ks):
            c = r
            for k in ks:
                if not isinstance(c, dict):
                    return None
                c = c.get(k)
            return c
        try:
            e = float(m.get("murphy_e") or 0)
        except ValueError:
            e = 0
        d = dict(name=(m.get("name") or "").strip(), lpb=lpb, e=e,
                 snfs=bool(r.get("is_snfs")), digits=g("input", "digits"),
                 uniq=g("relations", "unique"), total=g("relations", "total"),
                 mrows=g("matrix", "rows"), la=g("lanczos", "time_seconds"),
                 fr=g("filtering", "final_relations"), fi=g("filtering", "final_ideals"))
        if d["uniq"] and d["digits"]:
            out.append(d)
    return out


def ols(data, y, xs):
    """log-OLS; returns (beta, stderr, R2, n).  beta[i+1] is the coef on xs[i],
    so exp(beta[1]) reads as a multiplier per unit of xs[0]."""
    Y, X = [], []
    for r in data:
        v = [r.get(f) for f in xs]
        if not r.get(y) or any(not z for z in v):
            continue
        Y.append(math.log(r[y]))
        X.append([1.0] + [math.log(z) if f == "e" else float(z) for f, z in zip(xs, v)])
    n, k = len(Y), len(xs) + 1
    if n < k + 2:
        return None
    def solve(M):
        for c in range(k):
            p = max(range(c, k), key=lambda i: abs(M[i][c]))
            M[c], M[p] = M[p], M[c]
            pv = M[c][c]
            M[c] = [x / pv for x in M[c]]
            for i in range(k):
                if i != c and M[i][c]:
                    f = M[i][c]
                    M[i] = [a - f * b for a, b in zip(M[i], M[c])]
        return M
    XX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xy = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(k)]
    beta = [row[k] for row in solve([XX[a][:] + [Xy[a]] for a in range(k)])]
    sse = sum((Y[i] - sum(beta[a] * X[i][a] for a in range(k))) ** 2 for i in range(n))
    ybar = statistics.fmean(Y)
    sst = sum((v - ybar) ** 2 for v in Y)
    inv = solve([XX[a][:] + [1.0 if a == c else 0.0 for c in range(k)] for a in range(k)])
    s2 = sse / (n - k)
    se = [math.sqrt(max(s2 * inv[i][k + i], 0)) for i in range(k)]
    return beta, se, 1 - sse / sst, n


def geo(v):
    return math.exp(statistics.fmean(math.log(x) for x in v))


def _bands(data, key, nb):
    """nb equal-count bands of key(r) -- used when the control is continuous.
    Unlike the fixed 5-digit slices this covers the whole sample, which matters
    for SNFS: the digits window below holds only ~36% of the SNFS corpus."""
    v = sorted(key(r) for r in data)
    e = [v[int(i * len(v) / nb)] for i in range(nb)] + [v[-1] + 1e-9]
    return list(zip(e, e[1:]))


def within_band_steps(data, field="uniq", lpb_min=0, lo=150, hi=205, nmin=8,
                      key=None, nb=8):
    """Geometric-mean ratio between adjacent lpb inside a narrow band of the
    difficulty control -- the comparison NFS@Home's own operators ran as a
    natural experiment, and the one estimator that never has to split a
    coefficient with a control that is collinear with lpb.

    Default bands are fixed 5-digit slices.  Pass key=lambda r: log(r['e']) to
    band on Murphy-E instead, which is the control that means something for
    SNFS (and for GNFS too -- digits only proxies it there)."""
    steps = collections.defaultdict(list)
    # Fixed 5-digit slices from `lo` to `hi`.  Anything outside that window
    # contributes to nothing -- fine for GNFS (1425 of 1437 jobs fall inside)
    # and a real restriction for SNFS (1091 of 2995), which is why the
    # Murphy-E-banded variant above is the one quoted for SNFS.
    spans = (_bands(data, key, nb) if key
             else [(d0, d0 + 5) for d0 in range(lo, hi, 5)])
    k = key or (lambda r: r["digits"])
    for a, b_ in spans:
        bys = collections.defaultdict(list)
        for r in data:
            if a <= k(r) < b_ and r.get(field):
                bys[r["lpb"]].append(r[field])
        keep = {b: v for b, v in bys.items() if len(v) >= nmin}
        prev = prevb = None
        for b in sorted(keep):
            g = geo(keep[b])
            if prev and b == prevb + 1 and prevb >= lpb_min:
                steps[(prevb, b)].append(g / prev)
            prev, prevb = g, b
    return steps


def main():
    rows = load()
    gn = [r for r in rows if not r["snfs"]]
    sn = [r for r in rows if r["snfs"]]
    print(f"joined {len(rows)} jobs with an lpb  (GNFS {len(gn)}, SNFS {len(sn)})")

    print("\n== relations COLLECTED stand in for relations REQUIRED ==")
    print("   (only if every operator stopped at the same filtering tightness)")
    print(f"   {'lpb':>4} {'GNFS n':>7} {'GNFS med':>9} {'SNFS n':>7} {'SNFS med':>9}")
    exc = lambda rs, b: [r["fr"] / r["fi"] for r in rs
                         if r["lpb"] == b and r.get("fi") and r.get("fr")]
    for b in range(29, 36):
        g, s = exc(gn, b), exc(sn, b)
        if not g and not s:
            continue
        f = lambda v: statistics.median(v) if v else float("nan")
        print(f"   {b:>4} {len(g):>7} {f(g):>9.4f} {len(s):>7} {f(s):>9.4f}")

    print("\n== R per +1 lockstep step ==")
    # The right difficulty control is NOT the same for the two sieve types.
    # For GNFS, digits is informative (t=+23.7) and Murphy-E adds nothing once
    # it is in (t=+1.8).  For SNFS it is the exact reverse: digits is noise
    # (t=-0.1 with E controlled) because an SNFS number's digit count says
    # nothing about how hard its polynomial is, while ln(E) is strongly
    # informative (t=-11.5).  Controlling SNFS on digits, as this script
    # originally did, therefore controls for nothing and inflates SNFS R.
    cases = [("GNFS all", gn, ["lpb", "digits"]),
             ("GNFS digits>=175", [r for r in gn if r["digits"] >= 175], ["lpb", "digits"]),
             ("GNFS digits>=175 +logE", [r for r in gn if r["digits"] >= 175], ["lpb", "digits", "e"]),
             ("SNFS ~lpb+digits (WRONG)", sn, ["lpb", "digits"]),
             ("SNFS ~lpb+logE", sn, ["lpb", "e"]),
             ("SNFS ~lpb+digits+logE", sn, ["lpb", "digits", "e"])]
    for lab, data, xs in cases:
        fit = ols(data, "uniq", xs)
        if fit is None:
            print(f"   {lab:<24} too few jobs to fit (n={len(data)})")
            continue
        b, se, r2, n = fit
        ctrl = "  ".join(f"per {'ln(E)' if f == 'e' else f} {math.exp(b[i+1]):.4f}"
                         f" (t={b[i+1]/se[i+1]:+.1f})" for i, f in enumerate(xs[1:], 1))
        print(f"   {lab:<26} {math.exp(b[1]):.3f}x "
              f"[{math.exp(b[1]-1.96*se[1]):.3f}-{math.exp(b[1]+1.96*se[1]):.3f}]   "
              f"{ctrl}   R2={r2:.3f}  n={n}")
    print(f"   {'ideal-count model':<24} {2*33/34:.3f}x   (upper bound: a job never saturates")
    print(f"   {'':<24}         the large-prime universe, so it overstates R)")
    def cc(rs, f):
        return statistics.correlation([r["lpb"] for r in rs],
                                      [math.log(r["e"]) if f == "e" else r[f] for r in rs])
    eg = [r for r in gn if r.get("e")]; es = [r for r in sn if r.get("e")]
    print(f"   collinearity with lpb -- digits: GNFS {cc(gn,'digits'):+.3f}  SNFS {cc(sn,'digits'):+.3f}")
    print(f"                            ln(E): GNFS {cc(eg,'e'):+.3f}  SNFS {cc(es,'e'):+.3f}")
    print("   Murphy-E is the difficulty measure that means something for BOTH arms;")
    print("   digits merely proxies it for GNFS (corr -0.68) and means nothing for")
    print("   SNFS, where an SNFS number's digit count says nothing about how hard")
    print("   its polynomial is.  Control on digits and the two arms disagree wildly")
    print("   (GNFS 1.52 vs SNFS 1.79); control both on ln(E) and they agree.")

    print("\n== R controlled on Murphy-E, parametric and nonparametric ==")
    print(f"   {'':<8}{'OLS ~lpb+logE':>16}{'within-E-band':>16}{'steps':>7}")
    for lab, data in (("GNFS", eg), ("SNFS", es)):
        fit = ols(data, "uniq", ["lpb", "e"])
        if fit is None or len(data) < 4:
            print(f"   {lab:<8} too few jobs with a Murphy-E to fit (n={len(data)})")
            continue
        st = within_band_steps(data, key=lambda r: math.log(r["e"]))
        flat = [x for v in st.values() for x in v]
        g = statistics.geometric_mean(flat) if flat else float("nan")
        print(f"   {lab:<8}{math.exp(fit[0][1]):>15.3f}x{g:>15.3f}x{len(flat):>7}"
              f"   (regression n={len(data)})")

    st = within_band_steps(gn, lpb_min=31)
    flat = [x for v in st.values() for x in v]
    random.seed(7)
    boots = []
    for _ in range(2000):
        samp = [gn[random.randrange(len(gn))] for _ in range(len(gn))]
        s = [x for v in within_band_steps(samp, lpb_min=31).values() for x in v]
        if s:
            boots.append(geo(s))
    boots.sort()
    print(f"   {'GNFS within-band lpb>=31':<24} {geo(flat):.3f}x "
          f"[{boots[int(.025*len(boots))]:.3f}-{boots[int(.975*len(boots))]:.3f}] bootstrap   "
          f"n={len(flat)} steps")

    print("\n== the matrix barely notices (GNFS, within digit band, per +1 lpb) ==")
    for field, lab in (("uniq", "unique relations"), ("mrows", "matrix rows"),
                       ("la", "Lanczos wall sec")):
        s = within_band_steps(gn, field, lpb_min=0)
        f = [x for v in s.values() for x in v]
        # geometric throughout: these are ratios, and the pooled figure on the
        # same line is a geometric mean, so an arithmetic per-step column would
        # not reconcile with it
        # The pooled figure is the geometric mean over all INDIVIDUAL step
        # observations, not over the per-step means printed here, so it is
        # weighted by how many digit bands contributed each step and will not
        # equal the naive mean of the columns.  Per-step n is printed for that.
        print(f"   {lab:<18} " + "  ".join(f"{k[0]}->{k[1]}:{geo(v):.2f}x(n{len(v)})"
                                           for k, v in sorted(s.items()))
              + f"   pooled {geo(f):.3f}x over n={len(f)}")
    print("   (the Lanczos row is heterogeneous hardware -- noise, not a result)")

    print("\n== duplicates: unique fraction vs q-span and vs lpb ==")
    path = f"{HERE}/dedup.tsv"
    if not os.path.exists(path):
        print(f"   {path} not found -- skipping (see finding 99 for the numbers)")
    else:
        sheet = [r for r in csv.DictReader(open(path), delimiter="\t")]
        byname = {r["name"]: r for r in rows}
        pts = []
        for r in sheet:
            lo, hi = (float(x) for x in r["range_M"].split("-"))
            mid = (float(r["lo_uniq_pct"]) + float(r["hi_uniq_pct"])) / 2
            j = byname.get(r["poly"])
            pts.append(dict(span=hi / lo, mid=mid, lpb=j["lpb"] if j else None,
                            lo=float(r["lo_uniq_pct"]), hi=float(r["hi_uniq_pct"]),
                            typ=r["type"], name=r["poly"],
                            msieve=100 * j["uniq"] / j["total"] if j and j.get("total") else None))
        for t in ("GNFS", "SNFS"):
            s = [p for p in pts if p["typ"] == t]
            if not s:
                print(f"   {t} n= 0  -- none in dedup.tsv")
                continue
            print(f"   {t} n={len(s):2}  at qmin {statistics.fmean(p['lo'] for p in s):5.1f}%"
                  f"   at qmax {statistics.fmean(p['hi'] for p in s):5.1f}%")
        # Fit the two sieve types SEPARATELY.  Pooling them is what the rest of
        # this script is at pains not to do -- and the GNFS slope is the one
        # integrate.py's DUP_PER_RUNG is built from, so a pooled fit dragged by
        # a handful of steeper SNFS points would double that constant.
        for t in ("GNFS", "SNFS", "pooled"):
            f_ = [p for p in pts if p["lpb"] and (t == "pooled" or p["typ"] == t)]
            fit = ols([dict(y=math.exp(p["mid"]), lnspan=math.log(p["span"]), lpb=p["lpb"])
                       for p in f_], "y", ["lnspan", "lpb"])
            if fit is None:
                print(f"   {t:<6} joined n={len(f_)} -- too thin to fit")
                continue
            b, se, r2, n = fit
            print(f"   {t:<6} unique% = {b[0]:5.1f} {b[1]:+.2f}*ln(span) {b[2]:+.2f}*lpb"
                  f"   R2={r2:.3f} n={n}")
            print(f"            -> 3x to 6x span {b[1]*math.log(2):+.1f} pt;"
                  f"  each +1 lpb {b[2]:+.2f} pt"
                  f"{'   <- DUP_PER_RUNG comes from this row' if t == 'GNFS' else ''}")
        ms = [p for p in pts if p["msieve"]]
        print(f"   LEVEL DISAGREEMENT on the same {len(ms)} jobs: sheet "
              f"{statistics.fmean(p['mid'] for p in ms):.1f}% unique vs msieve "
              f"{statistics.fmean(p['msieve'] for p in ms):.1f}%"
              f" -- only the slope enters the rankings")


if __name__ == "__main__":
    main()
