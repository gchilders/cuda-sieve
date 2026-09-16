#!/usr/bin/env python3
"""AS276 over the fixed production window q = 80M..480M, as a cross-check on
integrate.py's free qmin scan.  Same bands, same targets, same loader -- it
imports them, because the two analyses disagreeing on their inputs is worse
than useless.
"""
from integrate import d, window, TARGETS

QMIN, QMAX = 80e6, 480e6
LABS = ["33/34", "33/34@96", "33/35", "34/35", "34/35@96", "35/36"]

for name, tgt in TARGETS["as276"].items():
    print(f"AS276, production window q = 80M..480M (span 6.0x), A=31  [{name}]\n")
    print(f"{'config':<11} {'rel/q avg':>9} {'relations':>10} {'target':>9} "
          f"{'% of tgt':>9} {'GPU-days':>9} {'verdict':>10}")
    print("-" * 76)
    res = {}
    for lab in LABS:
        k = ("as276", lab)
        if k not in d or len(d[k]) < 2 or lab not in tgt:
            print(f"{lab:<11} (no band data)")
            continue
        rels, ms = window(d[k], QMIN, QMAX)
        nq, _ = window({q: (1.0, 1.0) for q in d[k]}, QMIN, QMAX)
        N = tgt[lab]; days = ms / 1000 / 86400; pct = 100 * rels / N
        verdict = "OK" if rels >= N else "SHORT"
        res[lab] = (rels, N, days, pct, verdict)
        print(f"{lab:<11} {rels/nq:>9.1f} {rels/1e6:>9.0f}M {N/1e6:>8.0f}M "
              f"{pct:>8.0f}% {days:>9.1f} {verdict:>10}")
    base = res.get("34/35")
    if base and base[4] != "OK":
        print(f"\n(no 'x base' column: the 34/35 baseline is itself SHORT at "
              f"{base[3]:.0f}% of target, so scaling it to target would be the same "
              f"extrapolation\n this table refuses to make for the rows it excludes)")
        base = None
    if base:
        print("\nCost to the SAME matrix (scaled to each config's own target):")
        for lab, (rels, N, days, pct, v) in res.items():
            if v == "OK":
                d2 = days * N / rels
                print(f"  {lab:<11} {d2:>6.1f} GPU-days   "
                      f"{d2/(base[2]*base[1]/base[0]):.3f}x base")
    print()
