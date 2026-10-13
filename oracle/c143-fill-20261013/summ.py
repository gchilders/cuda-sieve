"""Summarise fillab.sh output: per geometry and arm, the median of each
stage over rounds, and the change against `base` in the same geometry."""
import statistics
import sys
from collections import defaultdict

FIELDS = ("fill", "apply", "resieve", "td", "wall", "relq")
runs = defaultdict(lambda: defaultdict(list))
order = defaultdict(list)
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 4 or "=" not in p[3]:
        continue
    geo, arm = p[0], p[1]
    if arm not in order[geo]:
        order[geo].append(arm)
    for kv in p[3:]:
        k, v = kv.split("=", 1)
        try:
            runs[(geo, arm)][k].append(float(v))
        except ValueError:
            pass

for geo in order:
    base = {f: statistics.median(runs[(geo, "base")][f]) for f in FIELDS
            if runs[(geo, "base")][f]}
    print(f"\n{geo}  (median of rounds; % vs base)")
    print(f"{'arm':14s}" + "".join(f"{f:>16s}" for f in FIELDS) + "   n")
    for arm in order[geo]:
        r = runs[(geo, arm)]
        cells = []
        for f in FIELDS:
            if not r[f]:
                cells.append(f"{'-':>16s}")
                continue
            m = statistics.median(r[f])
            d = 100.0 * (m / base[f] - 1.0) if base.get(f) else 0.0
            cells.append(f"{m:9.3f} {d:+5.1f}%")
        print(f"{arm:14s}" + "".join(cells) + f"   {len(r['wall'])}")
