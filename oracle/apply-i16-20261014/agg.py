import csv, sys, collections
# usage: agg.py file.csv [N]  -> top source lines by warp-stall samples
fn = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rows = list(csv.reader(open(fn)))
cur = None; S = collections.Counter(); I = collections.Counter(); src = {}
for r in rows:
    if not r: continue
    if r[0] == 'File Path': cur = r[1].split('/')[-1]; continue
    if r[0] in ('Function Name', 'Line No'): continue
    if r[0] and r[0].isdigit() and len(r) > 7:
        k = (cur, int(r[0]))
        try: S[k] += int(r[4]); I[k] += float(r[7])
        except ValueError: pass
        src[k] = r[1].strip()
ts, ti = sum(S.values()), sum(I.values())
print(f"total samples {ts}, warp instructions {ti:.4g}")
for k, v in S.most_common(N):
    print(f"{100*v/ts:5.1f}% {100*I[k]/ti:5.1f}%  {k[0]}:{k[1]:<5} {src[k][:90]}")
