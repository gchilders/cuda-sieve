#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces cofac_body.metal.inc from
# cofac.cuh's device half plus the three scan kernels cf_run_rounds borrows
# from td.cuh. Same contract as the other generators: the .inc is committed
# and reviewed like any other source.
import re

OUT = 'bench/metal/cofac_body.metal.inc'
cof = open('bench/cofac.cuh').read().split('\n')
td  = open('bench/td.cuh').read().split('\n')

def upto_close(lines, start):
    d = 0; seen = False
    for i in range(start, len(lines)):
        d += lines[i].count('{') - lines[i].count('}')
        if '{' in lines[i]: seen = True
        if seen and d == 0: return i
    raise SystemExit('unterminated at %d' % start)

def find(lines, pat, after=0):
    for i in range(after, len(lines)):
        if re.search(pat, lines[i]): return i
    raise SystemExit('not found: ' + pat)

# 1. the dual host/device arithmetic: mz<L> through mz_split
a = find(cof, r'^template <int L> struct mz')
b = upto_close(cof, find(cof, r'^CF_FN int mz_split', a))
arith = '\n'.join(cof[a:b + 1])

# 2. the kernels run_cofac reaches
KERNELS = []
for pat in (r'^__global__ void k_cofac', r'^__global__ void k_cof_selflags',
            r'^__global__ void k_cof_selscatter'):
    i = find(cof, pat)
    j = i - 1 if cof[i - 1].startswith('template') else i
    KERNELS.append('\n'.join(cof[j:upto_close(cof, i) + 1]))

# 3. td.cuh's three-pass scan, which cf_run_rounds launches directly
scan = []
for pat in (r'^__global__ void k_scan_pass1', r'^__global__ void k_scan_pass2',
            r'^__global__ void k_scan_pass3'):
    i = find(td, pat)
    scan.append('\n'.join(td[i:upto_close(td, i) + 1]))

body = arith + '\n\n' + '\n\n'.join(scan) + '\n\n' + '\n\n'.join(KERNELS)

# CF_NOINLINE keeps a big ECM helper out of its caller's register footprint
# on NVIDIA. MSL has no equivalent and the decision is the Metal compiler's,
# so it becomes an ordinary device function.
body = body.replace('CF_NOINLINE ', 'static inline ')
body = body.replace('CF_FN ', 'static inline ')
body = body.replace('CF_HD ', 'static inline ')
body = body.replace('__device__ __forceinline__ ', 'static inline ')
body = body.replace('__device__ ', 'static inline ')
body = body.replace('__restrict', '')
body = body.replace('unsigned long long', 'ulong').replace('long long', 'long')
body = body.replace('__umulhi(', 'mulhi(')

# ---- drop the host-only helpers that share this line range ---------------
# cofac.cuh's arithmetic block also holds the ECM plan builders and their
# prime sieve, which are host code (calloc/NULL) and are not CF_FN/CF_HD
# qualified. After the substitution above a device function reads
# 'static inline' and a host one reads plain 'static', so they separate
# cleanly on that.
out, i, dropped = [], 0, []
lines = body.split(chr(10))
while i < len(lines):
    l = lines[i]
    head, j = l, i
    if l.startswith('template') and i + 1 < len(lines):
        head, j = lines[i + 1], i + 1
    if re.match(r'^static\b', head) and 'static inline' not in head:
        d = 0; seen = False; k = j
        while k < len(lines):
            d += lines[k].count('{') - lines[k].count('}')
            if '{' in lines[k]: seen = True
            if seen and d == 0: break
            k += 1
        dropped.append(re.sub(r'.*?(\w+)\s*\(.*', r'\1', head))
        i = k + 1
        continue
    out.append(l); i += 1
body = chr(10).join(out)
if dropped: print('dropped host-only helpers:', ' '.join(dropped))

# ---- mz_rho's goto ------------------------------------------------------
# MSL rejects goto and labels outright. mz_rho's outer loop is `for (;;)`,
# exited only by `return 0` or `goto found`, so the jump becomes a flag and
# two breaks: the same control flow, written the way MSL allows.
OLD_TAIL = (chr(10).join(['        }', '        r <<= 1;',
            '        if (steps >= budget) { *acc += steps; return 0; }',
            '    }', '', 'found:']))
NEW_TAIL = (chr(10).join(['        }', '        if (rho_found) break;',
            '        r <<= 1;',
            '        if (steps >= budget) { *acc += steps; return 0; }',
            '    }', '']))
if 'goto found' in body:
    body = body.replace('    uint32_t steps = 0, r = 1;',
                        '    uint32_t steps = 0, r = 1;' + chr(10) +
                        '    bool rho_found = false;   /* stands in for the original goto */')
    body = body.replace('if (!mz_is_one<L>(&g)) goto found;',
                        'if (!mz_is_one<L>(&g)) { rho_found = true; break; }')
    assert OLD_TAIL in body, 'mz_rho tail not found'
    body = body.replace(OLD_TAIL, NEW_TAIL)
    import re as _re
    assert not _re.search(r'^\s*.*\bgoto\s+\w+;', body, _re.M), 'goto rewrite incomplete'
    print('rewrote mz_rho goto as a flag')

IDS = ['uint _tid [[thread_position_in_grid]]', 'uint _ntid [[threads_per_grid]]',
       'uint _bid [[threadgroup_position_in_grid]]', 'uint _lid [[thread_position_in_threadgroup]]',
       'uint _bdim [[threads_per_threadgroup]]', 'uint _gdim [[threadgroups_per_grid]]']
IDS_PLAIN = [x.split(' [[')[0] for x in IDS]
IDS_ARGS = ['_tid', '_ntid', '_bid', '_lid', '_bdim', '_gdim']

K = {
 'k_cofac': (['int L', 'int METHOD', 'int STAGE2'],
             [(str(L), m, s) for L in (3, 4) for (m, s) in (('1','1'), ('1','0'), ('0','0'))]),
 'k_cof_selflags':   (None, None),
 'k_cof_selscatter': (None, None),
 'k_scan_pass1':     (None, None),
 'k_scan_pass2':     (None, None),
 'k_scan_pass3':     (None, None),
}

def split_top(s):
    out, cur, d = [], '', 0
    for ch in s:
        if ch in '(<[': d += 1
        if ch in ')>]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur); return out

def bufparam(p, i):
    p = p.strip()
    if '*' in p: return 'device %s [[buffer(%d)]]' % (p, i)
    return 'constant %s [[buffer(%d)]]' % (re.sub(r'(\w+)$', r'&\1', p), i)
def plainparam(p):
    p = p.strip()
    return ('device ' + p) if '*' in p else ('constant ' + re.sub(r'(\w+)$', r'&\1', p))
def argname(p):
    return re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))

for name, (tparams, insts) in K.items():
    pre = r'(?:template\s*<[^>]*>\s*)?' if tparams else ''
    m = re.search(pre + r'__global__\s+void\s+' + name + r'\s*\(', body, re.S)
    if not m: print('MISS', name); continue
    a2 = body.index('(', m.end() - 1); d = 0; i = a2
    while i < len(body):
        if body[i] == '(': d += 1
        elif body[i] == ')':
            d -= 1
            if d == 0: break
        i += 1
    params = split_top(body[a2 + 1:i])
    brace = body.index('{', i); d = 0; j = brace
    while j < len(body):
        if body[j] == '{': d += 1
        elif body[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    inner = body[brace + 1:j]
    inner = re.sub(r'\b__shared__\s+', 'threadgroup ', inner)

    if tparams is None:
        head = 'kernel void %s(\n    %s)\n{\n    CUDA_KERNEL_IDS\n' % (
            name, ',\n    '.join([bufparam(p, k) for k, p in enumerate(params)] + IDS))
        newtext = head + inner + '}\n'
    else:
        head = ('template <%s>\nstatic inline void %s_body(\n    %s)\n{\n    CUDA_KERNEL_IDS\n'
                % (', '.join(tparams), name,
                   ',\n    '.join([plainparam(p) for p in params] + IDS_PLAIN)))
        args = [argname(p) for p in params] + IDS_ARGS
        tnames = [t.split()[-1] for t in tparams]
        wr = []
        for inst in insts:
            # A wrapper is a concrete kernel, so its parameter list must not
            # mention L / METHOD / STAGE2 -- substitute the instantiated values.
            cps = []
            for pp in params:
                q = pp
                for tn, tv in zip(tnames, inst):
                    q = re.sub(r'\b' + tn + r'\b', tv, q)
                cps.append(q)
            kps = [bufparam(pp, k) for k, pp in enumerate(cps)] + IDS
            wr.append('kernel void %s_%s(\n    %s)\n{\n    %s_body<%s>(%s);\n}\n'
                      % (name, '_'.join(inst), ',\n    '.join(kps), name,
                         ', '.join(inst), ', '.join(args)))
        # A by-value struct parameter (mz<L> lim2) is the kernel's OWN COPY in
        # CUDA. In MSL it arrives as a constant-space reference, which the body
        # then cannot take the address of, so give the body back a thread-local
        # copy under the original name.
        byval = [pp.strip() for pp in params
                 if '*' not in pp and re.search(r'<[^>]*>', pp)]
        if byval:
            decl = ''
            for pp in byval:
                ty, nm = pp.rsplit(' ', 1)
                decl += '    %s %s = %s;   /* CUDA passes this by value */\n' % (ty, nm, nm + '_v')
            head = head.replace('CUDA_KERNEL_IDS\n', 'CUDA_KERNEL_IDS\n' + decl)
            for pp in byval:
                nm = pp.rsplit(' ', 1)[1]
                head = re.sub(r'(constant [^,\n]*&)' + nm + r'\b', r'\g<1>' + nm + '_v', head)
        newtext = head + inner + '}\n\n' + '\n'.join(wr)
    body = body[:m.start()] + newtext + body[j + 1:]

# address spaces: every pointer here is a thread-local working value except a
# kernel's own buffers, which the parameter builders already qualified.
head_re = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)
# The ECM schedule (`s`, the prime-power list) and the stage-2 giant-step
# masks (`s2mask`) are DEVICE buffers threaded down from the kernel; every
# other pointer in this file is a thread-local working value. Getting this
# wrong is a compile error rather than a silent one, but spell it out anyway.
DEVICE_PARAMS = {'s', 's2mask', 'masks'}   # 'masks' is s2mask's name inside mz_ecm_stage2_pass

def qual(params):
    res = []
    for p in split_top(params):
        if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
            res.append(p); continue
        nm = re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))
        space = ('device ' if nm in DEVICE_PARAMS else 'thread ')
        st = p.lstrip(); pad = p[:len(p) - len(st)]
        res.append(pad + ('const ' + space + st[6:] if st.startswith('const ') else space + st))
    return ','.join(res)
body, nq = head_re.subn(lambda m: m.group(1) + qual(m.group(3)) + m.group(4), body)

# Local pointer VARIABLES need an address space too, not only parameters.
body = body.replace('const mpt<L> *G =', 'const thread mpt<L> *G =')

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d heads qualified)' % (OUT, body.count('\n'), nq))
