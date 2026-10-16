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

# Same gate for ss_first's two magic forms (upstream 2a3dfb0). Worth the same
# as the reciprocal gate and for the same reason: the two-step path exists
# because the one-step numerator can overflow, and only the device runs it.
k4 = next(i for i, l in enumerate(lines)
          if l.startswith('__global__ void k_verify_ss_first_cases'))
_d, _seen, e4 = 0, False, k4
for _i in range(k4, len(lines)):
    _d += lines[_i].count('{') - lines[_i].count('}')
    if '{' in lines[_i]: _seen = True
    if _seen and _d == 0: e4 = _i; break
b4 = k4
while not lines[b4].startswith('typedef struct'): b4 -= 1   # carry ss_first_case_t
assert 'ss_first_case_t' in '\n'.join(lines[b4:e4 + 1]), 'ss_first_case_t not carried'
body = ('\n'.join(lines[b1:e1 + 1]) + '\n\n' + '\n'.join(lines[b2:e2 + 1])
        + '\n\n' + '\n'.join(lines[b3:e3 + 1])
        + '\n\n' + '\n'.join(lines[b4:e4 + 1]))

# Every __global__ inside the extracted region must have an entry in K below,
# or it is silently dropped from the metallib and surfaces -- if at all -- as a
# missing-kernel error at run time on whatever path launches it. The 2026-09-16
# rebase lost k_cofac this way in the sibling generator and the only signal was
# a count in a progress line; upstream 2a3dfb0 then added three kernels here at
# once. Checked before any rewriting, against the source's own spelling.
def _kernel_names(text):
    # A comment block and a `void` on its own line can sit between __global__
    # and the name (k_apply and k_fill_l1 both do), so take the identifier that
    # immediately precedes the parameter list instead of pattern-matching the
    # whole head.
    out = []
    for m in re.finditer(r'__global__', text):
        a = text.index('(', m.end())
        out.append(re.findall(r'(\w+)\s*$', text[m.end():a])[-1])
    return sorted(set(out))
_declared = _kernel_names(
    re.sub(r'__launch_bounds__\([^)]*\)\s*', '', body))

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
 # SLICE_T: upstream 2a3dfb0 widened the bucket record's slice field to 32 bits
 # for the fine-slice ids the bucket resieve needs. The DEFAULTED instances
 # (two template arguments, uint16_t) and the explicit uint32_t ones are
 # different kernels and both are launched, so both are named here -- the
 # suffix is the template arguments as WRITTEN at the call site, which is what
 # portlib's mangler produces.
 'k_fill_atomic':      (['int RECBYTES', 'bool SLABBED', 'typename SLICE_T = uint16_t'],
                        [('2','false'),('4','false'),('8','false'),('4','true'),
                         ('4','false','uint32_t'),('4','true','uint32_t')], False),
 'k_fill_rows':        (['bool SLABBED', 'typename SLICE_T = uint16_t'],
                        [('false',),('true',),
                         ('false','uint32_t'),('true','uint32_t')], False),
 'k_rows_walk_next':   (None, None, False),
 'k_gate_fold':        (None, None, False),
 # Production uses <16,1,NORM_HORNER,SLABBED,SSTAIL>; the rest are run_bench's
 # own A/B pricing arms, which cost nothing to instantiate and would otherwise
 # fail at run time with a missing-kernel error. SSTAIL is chosen at run time
 # by ss_needs_tail on both paths, so both values of it are needed for the
 # production configuration and the tail instance alone for the others --
 # exactly what LAUNCH_APPLY's own ST argument says.
 'k_fill_l1':          (None, None, False),
 'k_fill_l2':          (['int RECBYTES'], [('2',), ('4',)], False),
 'k_apply':            (['int CELLBITS','int ATOMIC','int NORMMODE','bool SLABBED',
                         'bool SSTAIL = true'],
                        [(c, a, n, 'false', 'true') for c in ('16', '8')
                         for a in ('0', '1') for n in ('0', '1')]
                        + [('16','1','1','false','false'),
                           ('16','1','1','true','true'),
                           ('16','1','1','true','false')], True),
 'k_build_summary_g':  (None, None, False),
 'k_build_summary':    (None, None, False),
 'k_resieve_rewalk':   (None, None, False),
 'k_purge':            (None, None, False),
 'k_fill_segmented':   (None, None, False),
 'k_snapshot_bounds':  (None, None, False),
 'k_purge_prime':      (None, None, False),
 'k_intersect_compact':(['int AGG','bool SLABBED'], [('1','false'),('1','true')], False),
 'k_verify_td_mod_cases': (None, None, False),
 'k_verify_ss_first_cases': (None, None, False),
}

_missing = [k for k in _declared if k not in K]
assert not _missing, (
    '__global__ kernels in the extracted device region with no K entry: %s.\n'
    'Add each one (template parameters and the instantiations the host\n'
    'launches), or it is dropped from the metallib silently.' % ', '.join(_missing))
_extra = [k for k in K if k not in _declared]
assert not _extra, ('K names a kernel the device region does not declare: %s'
                    % ', '.join(_extra))

def split_top(s):
    out, cur, d = [], '', 0
    for ch in s:
        if ch in '(<[': d += 1
        if ch in ')>]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur); return out

# ---- optional buffer arguments ------------------------------------------
# Buffer indices the HOST sometimes passes nullptr for. Metal has no "unbound
# argument": setBuffer:nil is an UNSET argument and the validation layer
# refuses to dispatch (plan 9z-c). So these are declared under an MSL function
# constant instead -- when the constant is false the argument does not exist
# and Metal asks for nothing.
#
# metal_rt sets ONE constant, a uint bitmask at index 0, bit i meaning
# "argument i is bound"; the per-argument booleans are derived from it below.
#
# The list is the union over every launch site, taken from the audit in plan
# 9z-c. An index missing here that is passed nullptr somewhere shows up as a
# validation abort naming the kernel and the index, which is how these were
# found in the first place.
#
# BY PARAMETER NAME, not by index. They were indices until upstream 2a3dfb0
# inserted `nsplit` in the middle of k_apply's list and shifted the last two by
# one: the names 24 and 25 then meant `probe_x` (a scalar) and `probe_out`.
# That particular shift happened to stop compiling, because a scalar cannot be
# declared `device T *`. Had the inserted parameter landed one place further
# along, the mask would simply have described the wrong two buffers, and the
# only symptom would have been a validation abort on a path that passes null.
OPTIONAL_BUFFERS = {
    'k_apply':             ['dump', 'dbg_cells', 'probe_out', 'survbits'],
    'k_fill_atomic':       ['walk_cur', 'walk_next', 'gate'],
    'k_fill_rows':         ['gate'],
    'k_transform':         ['walk_cur', 'rt_out'],
    'k_resieve_scatter':   ['ispow', 'walk_cur'],
    'k_resieve_rows':      ['ispow'],
    'k_intersect_compact': ['out_x', 'out_a', 'out_b', 'nqb'],
}

# Buffer indices resolved from OPTIONAL_BUFFERS' parameter names by the
# emission loop; the function-constant declarations at the end of this
# file are emitted for exactly these.
OPT_INDICES = set()

MASK_DECL = """/* Optional buffer arguments: see metal/gen_bench_kernels.py.
 * metal_rt supplies bit i = "kernel argument i is bound". */
constant uint mtl_bound_mask [[function_constant(0)]];
"""

def bound_name(i):
    return 'mtl_bound_%d' % i

def bufparam(p, i, opt=()):
    p = p.strip()
    fc = (', function_constant(%s)' % bound_name(i)) if i in opt else ''
    if '*' in p: return 'device %s [[buffer(%d)%s]]' % (p, i, fc)
    return 'constant %s [[buffer(%d)%s]]' % (re.sub(r'(\w+)$', r'&\1', p), i, fc)

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
    # MSL has no default kernel argument -- every parameter is a binding the
    # dispatch must set -- so the default is dropped here and moved to the
    # call site instead (portlib.pad_launch_defaults). Dropping it silently
    # would emit `device uint32_t *rt_out = nullptr` in the body template and
    # `constant uint32_t &n_rt = &0` in the wrapper, neither of which is legal.
    params = [p.split('=', 1)[0].rstrip() for p in split_top(body[a + 1:i])]
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
        assert name not in OPTIONAL_BUFFERS, (
            '%s is not templated, so it has no wrapper to forward an optional'
            ' buffer through; give it one or drop the entry' % name)
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
        _names = [argname(p) for p in params]
        _want = OPTIONAL_BUFFERS.get(name, ())
        for _w in _want:
            assert _w in _names, ('%s has no parameter named %s' % (name, _w))
        opt = set(_names.index(w) for w in _want)
        OPT_INDICES |= opt
        args = [argname(p) for p in params] + smem_args + tg_args + IDS_ARGS
        kps = [bufparam(p, k, opt) for k, p in enumerate(params)] + smem_param + IDS
        decls = ('\n'.join(tg_decl_lines) + '\n') if tg_decl_lines else ''
        # A function-constant argument may be NAMED only where it exists, so
        # the forward goes through a local that is null when it does not. The
        # _body template is untouched: it still receives a possibly-null
        # pointer and still tests it.
        optdecl = ''
        for k in sorted(opt):
            pk = params[k].strip(); an = argname(pk)
            optdecl += ('    device %s = nullptr;\n' % re.sub(r'(\w+)\s*$', an + '_opt', pk))
            optdecl += ('    if (%s) %s_opt = %s;\n' % (bound_name(k), an, an))
            args[k] = an + '_opt'
        # A wrapper is a concrete kernel, so a parameter declared with a
        # template TYPE (upstream's `const SLICE_T *slice`) has to be spelled
        # with that instance's type. Defaulted template parameters an instance
        # does not name take their default, which is what the call site means.
        tnames, tdefaults = [], []
        for tp in tparams:
            _d = tp.split('=', 1)
            tnames.append(_d[0].split()[-1].strip())
            tdefaults.append(_d[1].strip() if len(_d) > 1 else None)
        for inst in insts:
            vals = list(inst) + tdefaults[len(inst):]
            assert all(v is not None for v in vals), (
                '%s<%s>: a template parameter with no default was not given'
                % (name, ', '.join(inst)))
            def subst(t, vals=vals):
                for nm, v in zip(tnames, vals):
                    t = re.sub(r'\b' + nm + r'\b', v, t)
                return t
            wrappers.append('kernel void %s_%s(\n    %s)\n{\n%s%s    %s_body<%s>(%s);\n}\n'
                            % (name, suffix_of(inst),
                               ',\n    '.join(subst(k) for k in kps),
                               subst(decls), subst(optdecl),
                               name, ', '.join(inst), ', '.join(args)))
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

# ---- host code that sits among the device functions ----------------------
# ss_tiers (upstream 6997d77) is `static inline` HOST code -- run_bench and
# pipeline.cuh call it to split the small-prime tiers before any launch -- but
# it is written between two device helpers, so the span this generator takes as
# "the device half" swallows it. It compiled as a device function with `thread`
# pointers and shipped in the metallib as dead code that no kernel calls; the
# only reason it surfaced is that the address-space assert below refuses to
# guess. gen_bench_host.py has the mirror-image bug and rescues it instead.
def drop_fn(text, sig):
    i = text.index(sig)
    j = text.index('{', i); d = 0
    while j < len(text):
        if text[j] == '{': d += 1
        elif text[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    return text[:i] + text[j + 1:].lstrip('\n')

_ss = 'static inline void ss_tiers('
assert body.count(_ss) == 1, 'ss_tiers is not in the device body, or is not unique'
body = drop_fn(body, _ss)
# The remaining mention is k_apply's comment citing ss_tiers as the host-side
# guarantee behind its static_assert -- prose, and worth keeping.
assert 'void ss_tiers(' not in body, 'the ss_tiers definition survives'
assert 'ss_tiers(' not in body.replace('(ss_tiers)', ''), 'a CALL to ss_tiers survives'
print('  dropped ss_tiers from the metallib (host code, no kernel calls it)')

# ---- address spaces ------------------------------------------------------
head_re = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)
# Address spaces on device-helper parameters are a per-parameter decision, not
# a blanket one, and getting it wrong would be silent: `S` is the THREADGROUP
# sieve array, while the small-prime tables passed beside it are device
# buffers. Every helper in this region that takes a pointer is spelled out
# here rather than guessed, and the assert below REFUSES to guess for one that
# is not -- upstream 6997d77 added ss_setup, which the old default sent to
# `thread` and which then failed as an MSL overload error 600 lines into a
# generated file, with nothing pointing back to this table.
HELPER_SPACES = {
    'ss_add':                  {'S': 'threadgroup'},
    'ss_setup':                {'sp': 'device', 'srt': 'device', 'sg': 'device',
                                'smag': 'device', 'mm_out': 'thread'},
    'sieve_small':             {'S': 'threadgroup', 'sp': 'device', 'srt': 'device',
                                'sg': 'device', 'slp': 'device', 'smag': 'device'},
    'apply_cell_side_effects': {'probe_out': 'device', 'dump': 'device'},
    # upstream 2a3dfb0's --fill-gate: the side-1 survivor bitmap, folded, as a
    # device buffer the fill kernels read per candidate record.
    'fill_gate_drops':         {'gate': 'device'},
}
_unspecified = []

def qual(params, fname):
    res = []
    spaces = HELPER_SPACES.get(fname, {})
    for p in split_top(params):
        if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
            res.append(p); continue
        nm = re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))
        if nm not in spaces:
            _unspecified.append('%s(%s)' % (fname, nm))
        space = spaces.get(nm, 'thread') + ' '
        st = p.lstrip(); pad = p[:len(p)-len(st)]
        res.append(pad + ('const ' + space + st[6:] if st.startswith('const ') else space + st))
    return ','.join(res)
body, nq = head_re.subn(lambda m: m.group(1) + qual(m.group(3), m.group(2)) + m.group(4), body)
assert not _unspecified, (
    'device helper pointer parameters with no entry in HELPER_SPACES: %s.\n'
    'Add each one explicitly -- defaulting to `thread` is a guess, and a wrong\n'
    'guess surfaces as an MSL overload error in the generated file rather than\n'
    'here.' % ', '.join(sorted(set(_unspecified))))

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
    # ...and the `_opt` local the optional-buffer wrappers forward through,
    # which the bare \b would not match because `_` is a word character. A
    # local declared `device ulong *` while the body expects uint32_t is a
    # compile error, which is how this was found.
    body = re.sub(r'device ulong \*( ?)' + c + r'(_opt)?\b',
                  lambda m, c=c: 'device uint32_t *' + m.group(1) + c + (m.group(2) or ''),
                  body)
    body = re.sub(r'\batomicAdd\(\s*' + c + r'\s*,', 'atomicAdd64(' + c + ',', body)

# ---- k_apply's slice-log table: keep it in device memory -----------------
# CUDA copies the 64-entry logp table into shared memory and reads it from
# there. That is worth doing on NVIDIA, where shared memory is a distinct and
# much faster pool than global. It is what puts k_apply 128 bytes -- exactly
# nslice_pow2 * sizeof(uint16_t) -- over Apple's hard 32,768 B threadgroup
# ceiling at log_region 14, which is the whole reason this build dropped to
# 13. Plan 8e measured what that costs: each bucket region carries ~20 us of
# fixed overhead, so halving the region doubles how often it is paid, and 13
# is 349.6 ms of apply against a predicted ~265 at 14.
#
# The table is READ-ONLY after the copy and indexed divergently, so on a UMA
# part with a unified cache there is no pool to promote it into: `device`
# reads of 128 hot bytes come out of the same L1. (`constant` would be wrong
# here -- it is tuned for uniform access, and this index varies per thread.)
# Dropping the copy makes the requirement exactly 32,768 B at region 14, and
# the host's check is `>`, so it fits.
_lut_decl = '    uint16_t *lut = (uint16_t *)(sm + nword);\n'
_lut_copy = '    for (uint32_t i = tid; i < nslice; i += nth) lut[i] = slice_logp[i];\n'
assert _lut_decl in body and _lut_copy in body, 'k_apply lut shape changed'
body = body.replace(_lut_decl, '', 1)
body = body.replace(_lut_copy, '', 1)
_n_lut = body.count('lut[')
assert _n_lut == 1, 'expected exactly one lut read, found %d' % _n_lut
body = body.replace('lut[', 'slice_logp[')
print('k_apply: slice-log table left in device memory (-%d B of threadgroup)'
      % (64 * 2))

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

# The function-constant declarations every optional argument derives from.
# One uint from the host, booleans derived here, so a kernel never needs the
# host to know which constant indices it declares -- which matters because
# Metal REJECTS a constant value for an index a function does not declare.
# ---- bench.h constants the device half reads -----------------------------
# LIFTED by name from bench.h, never retyped here: this file has said since
# Phase 4 that a hand-copied constant is how a port computes something else
# while compiling cleanly. Each must be USED by the body, so an entry that
# stops being needed fails here rather than lingering.
DEVICE_DEFINES = ['ROWRT_NONE']
_bh = open('bench/bench.h').read()
_lift = ''
for _nm in DEVICE_DEFINES:
    _m = re.search(r'^#define\s+' + _nm + r'\s+(.+?)\s*(?:/\*.*)?$', _bh, re.M)
    assert _m, 'bench.h does not define ' + _nm
    assert re.search(r'\b' + _nm + r'\b', body), (
        '%s is lifted from bench.h but the device half does not use it' % _nm)
    _lift += '#ifndef %s\n#define %s %s\n#endif\n' % (_nm, _nm, _m.group(1))
body = _lift + '\n' + body
print('  lifted %d bench.h #define(s) the device half reads' % len(DEVICE_DEFINES))

assert OPT_INDICES, 'no optional buffer was resolved -- the loop above did not run'
_idx = sorted(OPT_INDICES)
_decl = MASK_DECL + ''.join(
    'constant bool %s = (mtl_bound_mask & (1u << %d)) != 0;\n' % (bound_name(i), i)
    for i in _idx) + '\n'
body = _decl + body

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d heads qualified)' % (OUT, body.count('\n'), nq))
