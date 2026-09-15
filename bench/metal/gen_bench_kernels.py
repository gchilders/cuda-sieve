#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Transforms bench_kernels.cu's device
# halves into bench_kernels_body.metal.inc. Same contract as
# gen_fbgen_metal.py: the .inc is committed and reviewed like any other
# source; this exists so a later CUDA-side change can be re-diffed.
import re

SRC = 'bench/bench_kernels.cu'
OUT = 'bench/metal/bench_kernels_body.metal.inc'
lines = open(SRC).read().split('\n')

def span(first_marker, last_marker):
    b = next(i for i, l in enumerate(lines) if first_marker in l)
    e = next(i for i, l in enumerate(lines) if last_marker in l)
    d = 0; started = False
    for i in range(e, len(lines)):
        d += lines[i].count('{') - lines[i].count('}')
        if '{' in lines[i]: started = True
        if started and d == 0: return b, i
    raise SystemExit('unterminated ' + last_marker)

b1, e1 = span('template <bool SLABBED>', 'void k_fill_l2(')          # sieve core
b2, e2 = span('uint32_t bgcd(', '__global__ void k_intersect_compact')
# The reciprocal gate's device half. bench_kernels.cu:1992's own comment says
# it exists so "a CUDA-codegen/device-only regression cannot hide behind the"
# CPU check -- which is exactly the class of thing a port should keep.
k3 = next(i for i, l in enumerate(lines)
          if l.startswith('__global__ void k_verify_td_mod_cases'))
_d, _seen, e3 = 0, False, k3
for _i in range(k3, len(lines)):
    _d += lines[_i].count('{') - lines[_i].count('}')
    if '{' in lines[_i]: _seen = True
    if _seen and _d == 0: e3 = _i; break
b3 = k3
while not lines[b3].startswith('typedef struct'): b3 -= 1   # carry td_mod_case_t
body = ('\n'.join(lines[b1:e1 + 1]) + '\n\n' + '\n'.join(lines[b2:e2 + 1])
        + '\n\n' + '\n'.join(lines[b3:e3 + 1]))

body = body.replace('__device__ __forceinline__ ', 'static inline ')
body = body.replace('__device__ ', 'static inline ')
body = body.replace('__restrict', '')
# MSL has no `long long`; its `long` is already 64-bit.
body = body.replace('unsigned long long', 'ulong').replace('long long', 'long')

# ---- the two-level fill buffer depth ------------------------------------
# k_fill_l1 and k_fill_l2 declare L1_NBUF*L1_CAP uint32 plus two NBUF-word
# counters, and on Metal one more word for the __syncthreads_or vote:
#
#     128 * CAP * 4  +  128*4  +  128*4  +  4
#
# At CUDA's CAP = 64 that is 33,796 B against Apple's HARD 32,768 B ceiling,
# which has no opt-in tier to raise the way CUDA's ~100 KB does. 61 is the
# largest value that fits (32,260 B, 508 B spare) -- MEASURED against the
# driver's own refusal, not estimated. The cap is a pure buffering depth:
# L1_FLUSH is derived from it and every store is bounds-checked against it,
# so this changes how often a buffer flushes and nothing about what the
# kernel produces.
#
# Rewritten HERE rather than #defined in the wrapper, because the extracted
# region carries CUDA's own #define and would override anything set earlier.
for _nm in ('L1_CAP', 'L2_CAP'):
    body, _n = re.subn(r'#define ' + _nm + r'(\s+)64\b',
                       '#define ' + _nm + r'\g<1>61', body)
    if _n: print('retuned %s 64 -> 61 for the 32 KB threadgroup ceiling' % _nm)

# __launch_bounds__(N, M): the second argument is CUDA's
# minBlocksPerMultiprocessor and has no Metal analogue; the first becomes
# MSL's own attribute, which the shim applies at the kernel head instead.
body = re.sub(r'__global__\s+__launch_bounds__\([^)]*\)\s*\n?', '__global__ ', body)

IDS = ['uint _tid [[thread_position_in_grid]]', 'uint _ntid [[threads_per_grid]]',
       'uint _bid [[threadgroup_position_in_grid]]', 'uint _lid [[thread_position_in_threadgroup]]',
       'uint _bdim [[threads_per_threadgroup]]', 'uint _gdim [[threadgroups_per_grid]]']
IDS_PLAIN = [x.split(' [[')[0] for x in IDS]
IDS_ARGS = ['_tid', '_ntid', '_bid', '_lid', '_bdim', '_gdim']

# name -> (template params or None, instantiations, wants dynamic threadgroup mem)
K = {
 'k_transform':        (['bool SLABBED'], [('false',), ('true',)], False),
 'k_fill_atomic':      (['int RECBYTES', 'bool SLABBED'],
                        [('2','false'),('4','false'),('8','false'),('4','true')], False),
 # Production uses <16,1,NORM_HORNER,SLABBED>; the rest are run_bench's own
 # A/B pricing arms, which cost nothing to instantiate and would otherwise
 # fail at run time with a missing-kernel error.
 'k_fill_l1':          (None, None, False),
 'k_fill_l2':          (['int RECBYTES'], [('2',), ('4',)], False),
 'k_apply':            (['int CELLBITS','int ATOMIC','int NORMMODE','bool SLABBED'],
                        [(c, a, n, 'false') for c in ('16', '8')
                         for a in ('0', '1') for n in ('0', '1')]
                        + [('16','1','1','true')], True),
 'k_build_summary_g':  (None, None, False),
 'k_build_summary':    (None, None, False),
 'k_resieve_rewalk':   (None, None, False),
 'k_purge':            (None, None, False),
 'k_fill_segmented':   (None, None, False),
 'k_snapshot_bounds':  (None, None, False),
 'k_purge_prime':      (None, None, False),
 'k_intersect_compact':(['int AGG','bool SLABBED'], [('1','false'),('1','true')], False),
 'k_verify_td_mod_cases': (None, None, False),
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

def suffix_of(inst):
    return '_'.join('1' if a == 'true' else '0' if a == 'false' else a for a in inst)

for name, (tparams, insts, dynsmem) in K.items():
    # A comment block can sit between `template <...>` and `__global__`
    # (k_apply has a long one), so the optional prefix has to skip comments.
    COMMENT = r'(?:/\*.*?\*/\s*|//[^\n]*\n\s*)*'
    pre = (r'(?:template\s*<[^>]*>\s*' + COMMENT + r')?') if tparams else ''
    # A comment can sit between __global__ and void (k_fill_l1 carries the
    # occupancy note there), so skip comments on both sides.
    m = re.search(pre + r'__global__\s*' + COMMENT + r'\s*(?:void\s*)?'
                  + COMMENT + name + r'\s*\(', body, re.S)
    if not m:
        print('MISS', name); continue
    a = body.index('(', m.end() - 1)
    d = 0; i = a
    while i < len(body):
        if body[i] == '(': d += 1
        elif body[i] == ')':
            d -= 1
            if d == 0: break
        i += 1
    params = split_top(body[a + 1:i])
    brace = body.index('{', i)
    # find end of the kernel
    d = 0; j = brace
    while j < len(body):
        if body[j] == '{': d += 1
        elif body[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    end = j + 1
    inner = body[brace + 1:j]
    # `extern __shared__ T name[];` becomes a threadgroup buffer parameter
    smem = re.search(r'extern\s+__shared__\s+(\w+)\s+(\w+)\[\];', inner)
    smem_param = []
    if smem:
        inner = inner.replace(smem.group(0), '')
        smem_param = ['threadgroup %s *%s [[threadgroup(0)]]' % (smem.group(1), smem.group(2))]
        smem_plain = ['threadgroup %s *%s' % (smem.group(1), smem.group(2))]
        smem_args  = [smem.group(2)]
    else:
        smem_plain = []; smem_args = []
    # static __shared__ arrays are ordinary threadgroup declarations in MSL
    inner = re.sub(r'\b__shared__\s+', 'threadgroup ', inner)

    # A kernel that votes with __syncthreads_or needs the scratch word the
    # compat macro names. Declared as a one-element array so a plain kernel
    # and a templated one spell the call site identically.
    if '__syncthreads_or' in inner:
        inner = '    threadgroup uint _sync_or_flag[1];\n' + inner

    # MSL forbids a threadgroup declaration inside a non-kernel function, so a
    # templated kernel's static shared arrays move to the WRAPPER, which is
    # kernel-qualified, and the body takes pointers. Making them
    # [[threadgroup(n)]] PARAMETERS instead is the trap: a parameter is
    # zero-length unless the host sets its length, and these were static
    # __shared__ arrays in CUDA with no host involvement at all. That mistake
    # cost a band that ran to completion and produced a wrong answer.
    tg_decls = re.findall(r'^\s*threadgroup\s+([\w<>]+)\s+(\w+)\[([^\]]+)\];\s*$',
                          inner, re.M)
    tg_params, tg_args, tg_decl_lines = [], [], []
    if tparams is not None and tg_decls:
        for ty, nm, dim in tg_decls:
            inner = re.sub(r'^\s*threadgroup\s+' + ty + r'\s+' + nm + r'\[[^\]]+\];\s*$',
                           '', inner, flags=re.M)
            tg_params.append('threadgroup %s *%s' % (ty, nm))
            tg_decl_lines.append('    threadgroup %s %s[%s];' % (ty, nm, dim))
            tg_args.append(nm)

    if tparams is None:
        head = 'kernel void %s(\n    %s)\n{\n    CUDA_KERNEL_IDS\n' % (
            name, ',\n    '.join([bufparam(p, k) for k, p in enumerate(params)]
                                 + smem_param + IDS))
        newtext = head + inner + '}\n'
    else:
        head = ('template <%s>\nstatic inline void %s_body(\n    %s)\n{\n    CUDA_KERNEL_IDS\n'
                % (', '.join(tparams), name,
                   ',\n    '.join([plainparam(p) for p in params] + smem_plain
                                    + tg_params + IDS_PLAIN)))
        wrappers = []
        args = [argname(p) for p in params] + smem_args + tg_args + IDS_ARGS
        kps = [bufparam(p, k) for k, p in enumerate(params)] + smem_param + IDS
        decls = ('\n'.join(tg_decl_lines) + '\n') if tg_decl_lines else ''
        for inst in insts:
            wrappers.append('kernel void %s_%s(\n    %s)\n{\n%s    %s_body<%s>(%s);\n}\n'
                            % (name, suffix_of(inst), ',\n    '.join(kps), decls, name,
                               ', '.join(inst), ', '.join(args)))
        newtext = head + inner + '}\n\n' + '\n'.join(wrappers)
    body = body[:m.start()] + newtext + body[end:]

# build_slices_b is HOST code that happens to sit between two kernels in the
# source, so the device-region extraction sweeps it in. It belongs to the
# host half and is dropped here.
m = re.search(r'static uint32_t build_slices_b\s*\(', body)
if m:
    d = 0; k = body.index('{', m.end())
    j = k
    while j < len(body):
        if body[j] == '{': d += 1
        elif body[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    body = body[:m.start()] + '/* build_slices_b: host-side, see bench_kernels.cu */\n' + body[j + 1:]
    print('dropped host helper build_slices_b')

# k_fill_l1 / k_fill_l2 are BACK. They wanted 33,792 B of static threadgroup
# memory against Apple's 32,768 B ceiling; metal/bench_kernels.metal now
# overrides L1_CAP/L2_CAP from 64 to 62, which brings the footprint to exactly
# 32,768 B. The cap is a pure buffering depth -- L1_FLUSH is derived from it
# and every store is bounds-checked against it -- so the change is
# functionally neutral, not an approximation.

# ---- address spaces ------------------------------------------------------
head_re = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)
# Address spaces on device-helper parameters are a per-parameter decision, not
# a blanket one, and getting it wrong would be silent: `S` is the THREADGROUP
# sieve array, while the small-prime tables passed beside it are device
# buffers. Only three helpers in this region take pointers at all, so they are
# spelled out rather than guessed.
HELPER_SPACES = {
    'ss_add':                  {'S': 'threadgroup'},
    'sieve_small':             {'S': 'threadgroup', 'sp': 'device', 'srt': 'device',
                                'sg': 'device', 'slp': 'device', 'smag': 'device'},
    'apply_cell_side_effects': {'probe_out': 'device', 'dump': 'device'},
}

def qual(params, fname):
    res = []
    spaces = HELPER_SPACES.get(fname, {})
    for p in split_top(params):
        if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
            res.append(p); continue
        nm = re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))
        space = spaces.get(nm, 'thread') + ' '
        st = p.lstrip(); pad = p[:len(p)-len(st)]
        res.append(pad + ('const ' + space + st[6:] if st.startswith('const ') else space + st))
    return ','.join(res)
body, nq = head_re.subn(lambda m: m.group(1) + qual(m.group(3), m.group(2)) + m.group(4), body)

# nlost is this file's only 64-bit atomic counter. Metal has no 64-bit atomics
# at all (Phase 0), so it becomes a PAIR of uint32 words and
# cuda_msl_compat.h's atomicAdd(device uint*, ulong) folds the carry across
# them. Two little-endian uint32 words at one address ARE a little-endian
# uint64, so the host's existing 8-byte readback needs no change at all.
# Every 64-bit counter in this region is a DIAGNOSTIC written with atomicAdd:
# nlost (records the walk could not place), and the resieve/purge/intersect
# probe counters. Metal has no 64-bit atomics at all (Phase 0), so each
# becomes a pair of uint32 words with the carry folded across them by
# cuda_msl_compat.h's atomicAdd64. Because two little-endian uint32 words at
# one address ARE a little-endian uint64, the host's existing 8-byte readback
# of each counter is unchanged -- there is no host-side edit at all.
#
# This is only sound because they are diagnostics: the pair is eventually
# consistent rather than atomic as a unit. Nothing computes from them.
COUNTERS64 = ['nlost', 'nprobe', 'npass1', 'nread', 'npre', 'nqb']
for c in COUNTERS64:
    body = re.sub(r'device ulong \*( ?)' + c + r'\b', r'device uint32_t *\1' + c, body)
    body = re.sub(r'\batomicAdd\(\s*' + c + r'\s*,', 'atomicAdd64(' + c + ',', body)

# ---- pointer casts inside bodies ----------------------------------------
# `*(uint16_t *)(out + at)` needs an address space in MSL. `out` is a device
# buffer everywhere this appears except the one lut cast off shared memory.
body = body.replace('(uint16_t *)(sm +', '(threadgroup uint16_t *)(sm +')
body = re.sub(r'\(\s*(uint16_t|uint32_t|uint64_t)\s*\*\s*\)\s*\(\s*out\s*\+',
              r'(device \1 *)(out +', body)
body = re.sub(r'\b(uint16_t|uint32_t)\s+\*(\w+)\s*=\s*\(threadgroup',
              r'threadgroup \1 *\2 = (threadgroup', body)
# Local pointer VARIABLES need an address space too, not just parameters.
# `uint32_t *S = sm;` aliases the threadgroup sieve array; `const uint32_t
# *row = bounds + ...` aliases a device buffer.
body = body.replace('    uint32_t *S   = sm;', '    threadgroup uint32_t *S   = sm;')
body = body.replace('const uint32_t *row = bounds +', 'device const uint32_t *row = bounds +')

# ---- the fp64 norm fallback ----------------------------------------------
# The one place in the sieve that genuinely needs binary64. Rewritten onto
# softfp64.h, which Phase 2 proved bit-exact against hardware fp64 -- so this
# is the same computation, not an approximation.
#
# CONTRACTION: the CUDA line is `accd = accd * ud + N.dd[k] * vpd;` and nvcc
# contracts a*b+c into an fma by default (-fmad=true), as does clang for C at
# its default -ffp-contract. sf_fma is genuinely single-rounded, so it
# reproduces that. Flip NORM_FP64_NO_FMA if a comparison against a real nvcc
# build ever shows otherwise -- that determination belongs to Phase 7.
OLD_FP64 = """                    const double a = (double)((int64_t)ii * N.a0 + (int64_t)jj * N.b0);
                    const double b = (double)((int64_t)ii * N.a1 + (int64_t)jj * N.b1);
                    const double ud = a / N.A, vd = b / N.B;
                    double accd = N.dd[N.deg], vpd = 1.0;
                    #pragma unroll
                    for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                        if (k < N.deg) { vpd *= vd; accd = accd * ud + N.dd[k] * vpd; }
                    s = (float)fabs(accd);"""
NEW_FP64 = """                    /* Apple GPUs have no `double`; softfp64.h is bit-exact
                     * binary64 in integer ops (Phase 2 gate). Same
                     * computation, same rounding, not an approximation. */
                    const sf64 a = sf_from_i64((long)ii * N.a0 + (long)jj * N.b0);
                    const sf64 b = sf_from_i64((long)ii * N.a1 + (long)jj * N.b1);
                    const sf64 ud = sf_div(a, N.A), vd = sf_div(b, N.B);
                    sf64 accd = N.dd[N.deg], vpd = SF_ONE_D;
                    #pragma unroll
                    for (int k = BENCH_MAX_DEGREE - 1; k >= 0; k--)
                        if (k < N.deg) {
                            vpd = sf_mul(vpd, vd);
#ifdef NORM_FP64_NO_FMA
                            accd = sf_add(sf_mul(accd, ud), sf_mul(N.dd[k], vpd));
#else
                            accd = sf_fma(accd, ud, sf_mul(N.dd[k], vpd));
#endif
                        }
                    s = sf_to_f32(sf_abs(accd));"""
if OLD_FP64 not in body:
    raise SystemExit('fp64 fallback block not found -- did bench_kernels.cu change?')
body = body.replace(OLD_FP64, NEW_FP64)
print('fp64 fallback rewritten onto softfp64')

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d heads qualified)' % (OUT, body.count('\n'), nq))
