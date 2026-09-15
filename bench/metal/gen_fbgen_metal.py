#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Transforms fbgen_gpu.cu's device half
# into fbgen_gpu_body.metal.inc. The .inc is committed and reviewed like any
# other source; this script exists so a later CUDA-side change to that device
# code can be re-diffed rather than re-ported from memory.
#
# Run from the repository root:  python3 bench/metal/gen_fbgen_metal.py
import re

SRC   = 'bench/fbgen_gpu.cu'
OUT   = 'bench/metal/fbgen_gpu_body.metal.inc'
FIRST = 'uint32_t d_big_mod('        # first device function
LAST  = 'k_upper_bound'             # last kernel

src = open(SRC).read()
lines = src.split('\n')
beg = next(i for i, l in enumerate(lines) if FIRST in l)
beg = beg - 1 if lines[beg - 1].startswith('__device__') else beg
end = next(i for i, l in enumerate(lines) if LAST in l and '__global__' in l)
# extend to the closing brace of that kernel
d = 0; started = False
for i in range(end, len(lines)):
    d += lines[i].count('{') - lines[i].count('}')
    if '{' in lines[i]: started = True
    if started and d == 0: end = i; break
body = '\n'.join(lines[beg:end + 1])

body = body.replace('__device__ __forceinline__ ', 'static inline ')
body = body.replace('__device__ ', 'static inline ')
body = body.replace('__umulhi(', 'mulhi(')            # CUDA intrinsic -> MSL

# ---- kernels -------------------------------------------------------------
# name -> (params, template_params_or_None, extra_constant_buffers)
ALG  = [('constant const gpu_big_t *c_alg', 'c_alg'), ('constant int &c_alg_deg', 'c_alg_deg')]
RAT  = [('constant const gpu_big_t *c_y0', 'c_y0'), ('constant const gpu_big_t *c_y1', 'c_y1')]
IDS  = ['uint _tid [[thread_position_in_grid]]', 'uint _ntid [[threads_per_grid]]',
        'uint _bid [[threadgroup_position_in_grid]]', 'uint _lid [[thread_position_in_threadgroup]]',
        'uint _bdim [[threads_per_threadgroup]]', 'uint _gdim [[threadgroups_per_grid]]']
IDS_PLAIN = ['uint _tid', 'uint _ntid', 'uint _bid', 'uint _lid', 'uint _bdim', 'uint _gdim']
IDS_ARGS  = ['_tid', '_ntid', '_bid', '_lid', '_bdim', '_gdim']

K = {
 'k_alg_roots_fixed':      (['const uint32_t *primes','uint32_t n','uint32_t *rootbuf','uint32_t *counts','uint32_t *failures'], ['int CAP','bool MONT','bool SYMMETRIC'], ALG, [('6','true','false'),('8','true','false'),('6','false','false'),('8','false','false')]),
 'k_alg_roots_fixed_mark': (['const uint32_t *primes','uint32_t n','uint32_t *rootbuf','uint32_t *counts','uint8_t *special','uint32_t *failures'], ['int CAP','bool MONT'], ALG, [('6','true'),('8','true')]),
 'k_alg_roots':            (['const uint32_t *primes','uint32_t n','uint32_t *rootbuf','uint32_t *counts','uint32_t *failures'], ['bool MONT','bool SYMMETRIC'], ALG, [('true','false'),('false','false'),('true','true'),('false','true')]),
 'k_make_odds':            (['uint32_t lo','uint32_t n','uint32_t *values'], None, [], None),
 'k_mark_composites':      (['uint32_t lo','uint32_t n','const uint32_t *base','uint32_t nbase','uint8_t *isprime'], None, [], None),
 'k_rational_roots':       (['const uint32_t *primes','uint32_t n','uint32_t *roots','uint32_t *failures'], None, RAT, None),
 'k_total_roots':          (['const uint32_t *counts','const uint32_t *offsets','uint32_t n','uint32_t *total'], None, [], None),
 'k_scatter_alg':          (['const uint32_t *primes','uint32_t n','const uint32_t *rootbuf','const uint32_t *counts','const uint32_t *offsets','uint32_t *out_p','uint32_t *out_r'], None, [], None),
 'k_scatter_alg_mark':     (['const uint32_t *primes','uint32_t n','const uint32_t *rootbuf','const uint32_t *counts','const uint32_t *offsets','const uint8_t *special','uint32_t *out_p','uint32_t *out_r','uint8_t *out_special'], None, [], None),
 'k_upper_bound':          (['const uint32_t *v','uint32_t n','uint32_t lim','uint32_t *out'], None, [], None),
}

def bufparam(p, i):
    p = p.strip()
    if '*' in p: return 'device %s [[buffer(%d)]]' % (p, i)
    return 'constant %s [[buffer(%d)]]' % (re.sub(r'(\w+)$', r'&\1', p), i)

def plainparam(p):
    p = p.strip()
    if '*' in p: return 'device ' + p
    return 'constant ' + re.sub(r'(\w+)$', r'&\1', p)

def argname(p):
    return re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))

def indent(xs):
    return ',\n    '.join(xs)

for name, (params, tparams, extra, insts) in K.items():
    # For a templated kernel the original `template <...>` line sits just
    # above __global__ and must be swallowed with it, or MSL sees two
    # parameter lists on one definition.
    pre = r'(?:template\s*<[^>]*>\s*)?' if tparams else ''
    pat = re.compile(pre + r'__global__ void ' + name + r'\s*\([^{]*?\)\s*\{', re.S)
    m = pat.search(body)
    if not m: print('MISS', name); continue

    if tparams is None:
        ps = [bufparam(p, i) for i, p in enumerate(params)]
        for j, (decl, _) in enumerate(extra):
            ps.append('%s [[buffer(%d)]]' % (decl, len(params) + j))
        head = 'kernel void %s(\n    %s)\n{\n    FB_KERNEL_IDS\n' % (name, indent(ps + IDS))
        body = body[:m.start()] + head + body[m.end():]
    else:
        # templated: body becomes a static inline, wrappers become the kernels
        ps = [plainparam(p) for p in params] + [d for d, _ in extra] + IDS_PLAIN
        head = ('template <%s>\nstatic inline void %s_body(\n    %s)\n{\n    FB_KERNEL_IDS\n'
                % (', '.join(tparams), name, indent(ps)))
        # find the end of this kernel to append wrappers after it
        start = m.start()
        i = m.end() - 1; d = 0
        while i < len(body):
            if body[i] == '{': d += 1
            elif body[i] == '}':
                d -= 1
                if d == 0: break
            i += 1
        tail = i + 1
        wrappers = []
        args = [argname(p) for p in params] + [a for _, a in extra] + IDS_ARGS
        kps = [bufparam(p, j) for j, p in enumerate(params)]
        for j, (decl, _) in enumerate(extra):
            kps.append('%s [[buffer(%d)]]' % (decl, len(params) + j))
        for inst in insts:
            suffix = '_'.join('1' if a == 'true' else '0' if a == 'false' else a for a in inst)
            wrappers.append(
                'kernel void %s_%s(\n    %s)\n{\n    %s_body<%s>(%s);\n}\n'
                % (name, suffix, indent(kps + IDS), name, ', '.join(inst), ', '.join(args)))
        body = (body[:start] + head + body[m.end():tail] + '\n\n' + '\n'.join(wrappers)
                + body[tail:])

# ---- address spaces on every remaining pointer parameter ------------------
CONST_PTR_FNS = {'d_big_mod', 'd_big_mod_mont', 'd_ctx_big'}
head_re = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)

def split_top(s):
    out, cur, d = [], '', 0
    for ch in s:
        if ch in '(<[': d += 1
        if ch in ')>]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur)
    return out

def qual(params, fname):
    res = []
    for p in split_top(params):
        if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
            res.append(p); continue
        space = 'constant ' if (fname in CONST_PTR_FNS and 'gpu_big_t' in p) else 'thread '
        st = p.lstrip(); pad = p[:len(p) - len(st)]
        res.append(pad + ('const ' + space + st[6:] if st.startswith('const ') else space + st))
    return ','.join(res)

body, n = head_re.subn(lambda m: m.group(1) + qual(m.group(3), m.group(2)) + m.group(4), body)

# ---- thread the former __constant__ globals through -----------------------
# CUDA's c_alg / c_alg_deg are file-scope __constant__; MSL has no such thing,
# so the two device functions that read them take them as parameters and the
# kernels pass them down from their own constant buffers.
CARG = 'constant const gpu_big_t *c_alg, int c_alg_deg'
for fn in ('d_alg_roots_prime', 'd_alg_roots_prime_fixed'):
    m = re.search(r'static inline int ' + fn + r'\(', body)
    assert m, fn
    i = m.end(); d = 1
    while d:
        if body[i] == '(': d += 1
        elif body[i] == ')': d -= 1
        i += 1
    close = i - 1
    # C++ forbids a non-defaulted parameter after a defaulted one, and
    # d_alg_roots_prime_fixed's `special` has a default -- so splice in before
    # the first defaulted parameter rather than at the end.
    plist = body[m.end():close]
    eq = plist.find('=')
    if eq >= 0:
        cut = plist.rfind(',', 0, eq)
        ins = m.end() + cut
    else:
        ins = close
    body = body[:ins] + (',\n                                          ' + CARG
                         if eq < 0 else ',\n                                          '
                         + CARG) + body[ins:]

# Every call site gains the same two arguments. d_alg_roots_prime_fixed has a
# trailing defaulted `special` parameter, so for the four-argument calls the
# constants must be spliced in BEFORE it, matching the signature above.
def addargs(m):
    call = m.group(0)
    inner = call[call.index('(') + 1:-1]
    args = [a.strip() for a in inner.split(',')]
    if args and args[-1].startswith('&') and len(args) == 4:
        args = args[:-1] + ['c_alg', 'c_alg_deg', args[-1]]
    else:
        args = args + ['c_alg', 'c_alg_deg']
    return call[:call.index('(') + 1] + ', '.join(args) + ')'

body, ncalls = re.subn(r'\bd_alg_roots_prime(?:_fixed)?<[^>]*>\([^();]*\)', addargs, body)

# c_y0 / c_y1 arrive as constant POINTERS now, so their uses lose the address-of
body = body.replace('d_big_mod(&c_y0,', 'd_big_mod(c_y0,')
body = body.replace('d_big_mod(&c_y1,', 'd_big_mod(c_y1,')
print('threaded constants into 2 functions and %d call sites' % ncalls)

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d function heads qualified)' % (OUT, body.count('\n'), n))
