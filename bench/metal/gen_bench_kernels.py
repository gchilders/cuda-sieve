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
body = '\n'.join(lines[b1:e1 + 1]) + '\n\n' + '\n'.join(lines[b2:e2 + 1])

body = body.replace('__device__ __forceinline__ ', 'static inline ')
body = body.replace('__device__ ', 'static inline ')
body = body.replace('__restrict', '')
# MSL has no `long long`; its `long` is already 64-bit.
body = body.replace('unsigned long long', 'ulong').replace('long long', 'long')
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
 'k_apply':            (['int CELLBITS','int ATOMIC','int NORMMODE','bool SLABBED'],
                        [('16','1','1','false'),('16','1','1','true'),
                         ('16','1','0','false'),('16','0','1','false')], True),
 'k_build_summary_g':  (None, None, False),
 'k_build_summary':    (None, None, False),
 'k_resieve_rewalk':   (None, None, False),
 'k_purge':            (None, None, False),
 'k_fill_segmented':   (None, None, False),
 'k_snapshot_bounds':  (None, None, False),
 'k_purge_prime':      (None, None, False),
 'k_intersect_compact':(['int AGG','bool SLABBED'], [('1','false'),('1','true')], False),
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
    m = re.search(pre + r'__global__\s+(?:void\s+)?' + name + r'\s*\(', body, re.S)
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

    if tparams is None:
        head = 'kernel void %s(\n    %s)\n{\n    CUDA_KERNEL_IDS\n' % (
            name, ',\n    '.join([bufparam(p, k) for k, p in enumerate(params)]
                                 + smem_param + IDS))
        newtext = head + inner + '}\n'
    else:
        head = ('template <%s>\nstatic inline void %s_body(\n    %s)\n{\n    CUDA_KERNEL_IDS\n'
                % (', '.join(tparams), name,
                   ',\n    '.join([plainparam(p) for p in params] + smem_plain + IDS_PLAIN)))
        wrappers = []
        args = [argname(p) for p in params] + smem_args + IDS_ARGS
        kps = [bufparam(p, k) for k, p in enumerate(params)] + smem_param + IDS
        for inst in insts:
            wrappers.append('kernel void %s_%s(\n    %s)\n{\n    %s_body<%s>(%s);\n}\n'
                            % (name, suffix_of(inst), ',\n    '.join(kps), name,
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

# ---- k_fill_l1 / k_fill_l2: over Apple's threadgroup-memory ceiling ------
# Both declare 128*64 uint32 of static shared memory plus two 128-word
# counters: 33,792 B against Apple's hard 32,768 B limit, over by exactly
# 1 KB. They are the TWO-LEVEL fill path, and apply requires single-level
# 4-byte records (bench_kernels.cu:2660), so the production sieve uses
# k_fill_atomic and never reaches them. Excised rather than silently emitted:
# a kernel whose pipeline cannot be created is worse than one that is absent
# and documented. Re-tuning L1_CAP from 64 to 62 would fit exactly, but that
# is a performance change and belongs in Phase 8, measured, not guessed here.
for _l1 in ('k_fill_l1', 'k_fill_l2'):
  m = re.search(r'(?:template\s*<[^>]*>\s*)?__global__\s+(?:[^\n]*\n\s*)?void\s+' + _l1 + r'\s*\(', body)
  if m:
    d = 0; j = body.index('{', m.end())
    k = j
    while k < len(body):
        if body[k] == '{': d += 1
        elif body[k] == '}':
            d -= 1
            if d == 0: break
        k += 1
    body = (body[:m.start()]
            + '/* ' + _l1 + ' OMITTED. Both two-level fill kernels declare\n'
              ' * 128*64 uint32 plus two 128-word counters: 33,792 B against\n'
              " * Apple's hard 32,768 B threadgroup ceiling, over by exactly 1 KB.\n"
              ' * MSL additionally forbids threadgroup declarations inside the\n'
              ' * non-kernel helper the templated form would need.\n'
              ' *\n'
              ' * Not a blocker: apply requires single-level 4-byte records\n'
              ' * (bench_kernels.cu:2660), so the production sieve runs\n'
              ' * k_fill_atomic and never reaches these. Closing the gap means\n'
              ' * retuning L1_CAP/L2_CAP from 64 to 62, which is a performance\n'
              ' * change and belongs in Phase 8, measured rather than guessed. */\n'
            + body[k + 1:])
    print('excised ' + _l1 + ' (over threadgroup ceiling)')

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
