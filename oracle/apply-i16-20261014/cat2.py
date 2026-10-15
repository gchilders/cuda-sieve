import csv, sys, collections
fn = sys.argv[1]
REMAP = len(sys.argv) > 2 and sys.argv[2] == 'exp'
def remap(l):
    if l <= 521: return l
    if l <= 528: return 522
    if l <= 674: return l - 2
    if l <= 684: return 674
    return l - 9
rows = list(csv.reader(open(fn)))
cur = None; line = None; ins = []
for r in rows:
    if not r: continue
    if r[0] == 'File Path': cur = r[1].split('/')[-1]; continue
    if r[0] in ('Function Name', 'Line No'): continue
    if r[0] and r[0].isdigit(): line = (cur, int(r[0])); continue
    if r[2].startswith('0x') and len(r) > 7:
        try: ins.append((int(r[2], 16), line, int(r[4]), float(r[7]), r[3].strip()))
        except ValueError: pass
ins.sort()
def cat(k):
    f, l = k
    if f == 'bench_kernels.cu' and REMAP: l = remap(l)
    if f != 'bench_kernels.cu': return None
    if 784 <= l <= 819: return 'entry'
    if 820 <= l <= 919: return 'norm init'
    if 621 <= l <= 639: return 'small: block tier'
    if 640 <= l <= 670: return 'small: warp tier'
    if 671 <= l <= 676: return 'small: thread tier'
    if 920 <= l <= 926: return 'small: call'
    if 927 <= l <= 940: return 'bucket apply'
    if 941 <= l <= 1041 or 700 <= l <= 725: return 'threshold scan'
    return None
S = collections.Counter(); I = collections.Counter(); last = 'entry'
for a, k, s, i, txt in ins:
    c = cat(k)
    if c is None: c = last
    else: last = c
    S[c] += s; I[c] += i
ts, ti = sum(S.values()), sum(I.values())
for c in ['entry','norm init','small: block tier','small: warp tier','small: thread tier','small: call','bucket apply','threshold scan']:
    print(f"  {c:20s} samples {100*S[c]/ts:5.1f}%   instructions {100*I[c]/ti:5.1f}%")
