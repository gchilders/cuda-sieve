#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Produces td_body.metal.inc from td.cuh's
# device half. Forking td.cuh properly is what lets bench_kernels.metal and
# cofac_metal.cpp drop their copies of td_mod_magic, SS_KSHIFT, TD_SCAN_BLK
# and TD_FMAX -- td.cuh calls SS_KSHIFT "the single source of truth for the
# bias shift", and a duplicated single source of truth is a contradiction
# waiting to rot.
import re

SRC = 'bench/td.cuh'
OUT = 'bench/metal/td_body.metal.inc'
lines = open(SRC).read().split('\n')

b = next(i for i, l in enumerate(lines) if l.startswith('#if defined(__CUDACC__)'))
e = next(i for i in range(len(lines) - 1, 0, -1)
         if lines[i].startswith('#endif  /* __CUDACC__ */'))
body = '\n'.join(lines[b + 1:e])
# tdpoly_t and tdsmall_t are declared above td.cuh's __CUDACC__ guard, so
# they come from td_msl.h -- generated from that same section -- rather than
# being prepended here. bench_kernels.metal needs them too, and one
# statement per build is the whole point of forking td.cuh.

body = body.replace('__device__ __forceinline__ ', 'static inline ')
body = body.replace('__device__ ', 'static inline ')
body = body.replace('__restrict', '')
body = body.replace('unsigned long long', 'ulong').replace('long long', 'long')
body = body.replace('__umulhi(', 'mulhi(').replace('__umul64hi(', 'mulhi(')
body = re.sub(r'__global__\s+__launch_bounds__\([^)]*\)\s*\n?', '__global__ ', body)

IDS = ['uint _tid [[thread_position_in_grid]]', 'uint _ntid [[threads_per_grid]]',
       'uint _bid [[threadgroup_position_in_grid]]', 'uint _lid [[thread_position_in_threadgroup]]',
       'uint _bdim [[threads_per_threadgroup]]', 'uint _gdim [[threadgroups_per_grid]]']
IDS_PLAIN = [x.split(' [[')[0] for x in IDS]
IDS_ARGS = ['_tid', '_ntid', '_bid', '_lid', '_bdim', '_gdim']

K = {
 'k_tdsmall_advance': (None, None),
 'k_cand_stats':      (None, None),
 'k_group_counts':    (None, None),
 'k_scan_pass1':      (None, None),
 'k_scan_pass2':      (None, None),
 'k_scan_pass3':      (None, None),
 'k_classify':        (None, None),
 'k_accept_flags':    (None, None),
 'k_scatter_sel':     (None, None),
 'k_gather_ab':       (None, None),
 'k_emit_ranked':     (['bool SLABBED'], [('false',), ('true',)]),
 'k_td_record_warp':  (['bool SLABBED'], [('false',), ('true',)]),
 'k_resieve_scatter': (['int UNROLL', 'bool SLABBED'],
                       [('1','false'),('2','false'),('4','false'),('8','false'),('4','true')]),
 'k_td':              (['int DIVIDE', 'int RECORD', 'int SELECT', 'bool SLABBED'],
                       [('0','0','0','false'),('1','0','0','false'),('1','0','0','true'),
                        ('1','1','0','false'),('1','1','0','true'),
                        ('1','1','1','false'),('1','1','1','true')]),
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
    COMMENT = r'(?:/\*.*?\*/\s*|//[^\n]*\n\s*)*'
    pre = (r'(?:template\s*<[^>]*>\s*' + COMMENT + r')?') if tparams else ''
    m = re.search(pre + r'__global__\s+void\s+' + name + r'\s*\(', body, re.S)
    if not m: print('MISS', name); continue
    a = body.index('(', m.end() - 1); d = 0; i = a
    while i < len(body):
        if body[i] == '(': d += 1
        elif body[i] == ')':
            d -= 1
            if d == 0: break
        i += 1
    params = split_top(body[a + 1:i])
    brace = body.index('{', i); d = 0; j = brace
    while j < len(body):
        if body[j] == '{': d += 1
        elif body[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    inner = re.sub(r'\b__shared__\s+', 'threadgroup ', body[brace + 1:j])

    # MSL forbids a threadgroup declaration inside a non-kernel function, so a
    # templated kernel's static shared arrays have to become parameters. The
    # wrapper kernel declares them; the body receives pointers. Sizes stay in
    # the source as the array bound, and the host binds that many bytes.
    tg_decls = re.findall(r'^\s*threadgroup\s+([\w<>]+)\s+(\w+)\[([^\]]+)\];\s*$',
                          inner, re.M)
    tg_params, tg_args, tg_bufs = [], [], []
    if tparams is not None and tg_decls:
        for n_, (ty, nm, dim) in enumerate(tg_decls):
            inner = re.sub(r'^\s*threadgroup\s+' + ty + r'\s+' + nm + r'\[[^\]]+\];\s*$',
                           '', inner, flags=re.M)
            # The array is DECLARED IN THE WRAPPER, which is kernel-qualified
            # and may therefore hold threadgroup variables, and the body gets a
            # pointer. The obvious alternative -- making it a
            # [[threadgroup(n)]] PARAMETER -- silently gives a zero-length
            # buffer unless the host sets its length, and these were static
            # __shared__ arrays in CUDA with no host involvement at all. That
            # mistake cost a band that ran to completion and produced 125
            # cofactorisation candidates where the oracle has 1,851.
            tg_params.append('threadgroup %s *%s' % (ty, nm))
            tg_bufs.append('    threadgroup %s %s[%s];' % (ty, nm, dim))
            tg_args.append(nm)

    if tparams is None:
        head = 'kernel void %s(\n    %s)\n{\n    CUDA_KERNEL_IDS\n' % (
            name, ',\n    '.join([bufparam(p, k) for k, p in enumerate(params)] + IDS))
        new = head + inner + '}\n'
    else:
        head = ('template <%s>\nstatic inline void %s_body(\n    %s)\n{\n    CUDA_KERNEL_IDS\n'
                % (', '.join(tparams), name,
                   ',\n    '.join([plainparam(p) for p in params] + tg_params + IDS_PLAIN)))
        args = [argname(p) for p in params] + tg_args + IDS_ARGS
        tnames = [t.split()[-1] for t in tparams]
        wr = []
        for inst in insts:
            cps = []
            for pp in params:
                q = pp
                for tn, tv in zip(tnames, inst):
                    q = re.sub(r'\b' + tn + r'\b', tv, q)
                cps.append(q)
            kps = [bufparam(pp, k) for k, pp in enumerate(cps)] + IDS
            suffix = '_'.join('1' if a == 'true' else '0' if a == 'false' else a for a in inst)
            decls = (chr(10).join(tg_bufs) + chr(10)) if tg_bufs else ''
            wr.append('kernel void %s_%s(\n    %s)\n{\n%s    %s_body<%s>(%s);\n}\n'
                      % (name, suffix, ',\n    '.join(kps), decls, name,
                         ', '.join(inst), ', '.join(args)))
        new = head + inner + '}\n\n' + '\n'.join(wr)
    body = body[:m.start()] + new + body[j + 1:]

# td.cuh's device helpers read the survivor bitmap and its group-rank table
# out of device memory; everything else they touch is thread-local.
HELPER_SPACES = {'td_rank': {'bits': 'device', 'gbase': 'device'},
                 'td_divide_out': {'fac': 'device'}}
head_re = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)
def qual(params, fname):
    res = []
    spaces = HELPER_SPACES.get(fname, {})
    for p in split_top(params):
        if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
            res.append(p); continue
        nm = argname(p)
        space = spaces.get(nm, 'thread') + ' '
        st = p.lstrip(); pad = p[:len(p) - len(st)]
        res.append(pad + ('const ' + space + st[6:] if st.startswith('const ') else space + st))
    return ','.join(res)
body, nq = head_re.subn(lambda m: m.group(1) + qual(m.group(3), m.group(2)) + m.group(4), body)

# 64-bit diagnostic counters become uint32 pairs; Metal has no 64-bit atomics
# at all. Two little-endian uint32 words at one address ARE a little-endian
# uint64, so the host's 8-byte readback is unchanged.
for c in ('noverflow', 'nhit', 'ntested', 'ndiv'):
    body = re.sub(r'device ulong \*( ?)' + c + r'\b', r'device uint32_t *\1' + c, body)
    body = re.sub(r'\batomicAdd\(\s*' + c + r'\s*,', 'atomicAdd64(' + c + ',', body)

# k_classify takes the factor-base bound as a `double` because CADO's gap
# test is written against doubles and prp.cuh matches it deliberately. Apple
# GPUs have no double, so it arrives as its 64-bit pattern and cof_classify
# (prp_msl.h) consumes it through softfp64. The HOST passes the same 8 bytes
# it always did -- an IEEE binary64 -- so no host-side change is needed.
n_lim = body.count('constant double &lim')
body = body.replace('constant double &lim', 'constant sf64 &lim')
if n_lim: print('k_classify: lim carried as a binary64 bit pattern (softfp64)')

# Local pointer VARIABLES need an address space too, not only parameters.
# `fac` is the device factor array, so an alias into it is device-qualified.
# (Both sites are the same idiom in k_td and k_td_record_warp.)
n_local = body.count('uint32_t *myfac =')
body = body.replace('uint32_t *myfac =', 'device uint32_t *myfac =')
if n_local: print('qualified %d local device-pointer alias(es)' % n_local)

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d heads qualified)' % (OUT, body.count('\n'), nq))
