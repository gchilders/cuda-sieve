import csv, sys, collections
# stall-reason mix per stage, same nearest-preceding attribution as cat.py
fn = sys.argv[1]
rows = list(csv.reader(open(fn)))
h = rows[2]; SC = list(range(33, 50))
cur = None; line = None; ins = []
for r in rows:
    if not r: continue
    if r[0] == 'File Path': cur = r[1].split('/')[-1]; continue
    if r[0] in ('Function Name', 'Line No'): continue
    if r[0] and r[0].isdigit(): line = (cur, int(r[0])); continue
    if r[2].startswith('0x') and len(r) > 49:
        try: ins.append((int(r[2], 16), line, [float(r[c] or 0) for c in SC], int(r[4])))
        except ValueError: pass
ins.sort()
exec(open('catdef.py').read())
agg = collections.defaultdict(lambda: [0.0]*len(SC)); last = 'entry'
for a, k, st, s in ins:
    c = cat(k)
    if c is None: c = last
    else: last = c
    for i, v in enumerate(st): agg[c][i] += v
for c in ['norm init','small: block tier','small: warp tier','small: thread tier','bucket apply','threshold scan']:
    tot = sum(agg[c]) or 1
    top = sorted(((v, h[SC[i]].replace('stall_','')) for i, v in enumerate(agg[c])), reverse=True)[:5]
    print(f"  {c:20s} " + "  ".join(f"{n} {100*v/tot:.0f}%" for v, n in top))
