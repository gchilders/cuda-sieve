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

# ---- pipeline.cuh's one kernel --------------------------------------------
# upstream 2a3dfb0 put k_resieve_bucket in pipeline.cuh, which until then was
# host orchestration with no device code in it at all -- gen_pipeline_host.py
# says so in its own header. It is pulled in HERE, rather than into a third
# generator, because it calls td_rank and reads plat_t, so td.metal is the
# translation unit that already has everything it needs in scope; and because
# a third copy of the kernel-head machinery below would be a third thing to
# drift. gen_pipeline_host.py drops the same span from the host file, with
# asserts on both sides naming the other.
_pl = open('bench/pipeline.cuh').read().split('\n')
_pb = next(i for i, l in enumerate(_pl) if l.startswith('#define FINE_SHIFT'))
_pe = next(i for i, l in enumerate(_pl) if '__global__ void k_resieve_bucket' in l)
_d, _seen = 0, False
for _i in range(_pe, len(_pl)):
    _d += _pl[_i].count('{') - _pl[_i].count('}')
    if '{' in _pl[_i]: _seen = True
    if _seen and _d == 0: _pe = _i; break
_pspan = '\n'.join(_pl[_pb:_pe + 1])
# The span opens with the three FINE_* defines and must carry rb_mod, the
# kernel's only helper. pipe_build_fine is HOST code written between them --
# the ss_tiers shape from the 2026-09-24 rebase -- so it is dropped by name.
for _need in ('#define FINE_SHIFT', '#define FINE_BITS', '#define FINE_PER_GROUP',
              'uint32_t rb_mod(', '__global__ void k_resieve_bucket'):
    assert _need in _pspan, 'pipeline.cuh device span lost ' + _need
_hb = _pspan.index('static int pipe_build_fine(')
_he = _pspan.index('{', _hb); _d = 0
while _he < len(_pspan):
    if _pspan[_he] == '{': _d += 1
    elif _pspan[_he] == '}':
        _d -= 1
        if _d == 0: break
    _he += 1
_pspan = (_pspan[:_hb] + '/* pipe_build_fine: host-side, see pipeline.cuh */\n'
          + _pspan[_he + 1:])
assert 'pipe_build_fine(' not in _pspan.replace('see pipeline.cuh', ''), \
    'a CALL to pipe_build_fine survives into the metallib'
# A lifted #define must not override a -D of the same name; see td_host.h.
_pspan = re.sub(r'^#define (FINE_\w+)(\s+)(\S+)', r'#ifndef \1\n#define \1\2\3\n#endif',
                _pspan, flags=re.M)
body = body + '\n\n' + _pspan
print('  pulled k_resieve_bucket + rb_mod in from pipeline.cuh')
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
 # upstream 2a3dfb0's --resieve-rows: the row tier of k_resieve_scatter.
 # pipeline.cuh launches <4, SLABBED> only, as it does for the scatter.
 'k_resieve_rows':    (['int UNROLL', 'bool SLABBED'],
                       [('4','false'),('4','true')]),
 'k_resieve_bucket':  (None, None),   # from pipeline.cuh, see above
 'k_td':              (['int DIVIDE', 'int RECORD', 'int SELECT', 'bool SLABBED'],
                       [('0','0','0','false'),('1','0','0','false'),('1','0','0','true'),
                        ('1','1','0','false'),('1','1','0','true'),
                        ('1','1','1','false'),('1','1','1','true')]),
}

# Every __global__ in td.cuh's device half must be named in K, or it is
# dropped from the metallib silently and surfaces -- if at all -- as a
# missing-kernel error on whatever path launches it. upstream 2a3dfb0 added
# k_resieve_rows here and nothing complained until this check existed.
def _kernel_names(text):
    out = []
    for m in re.finditer(r'__global__', text):
        a = text.index('(', m.end())
        out.append(re.findall(r'(\w+)\s*$', text[m.end():a])[-1])
    return sorted(set(out))
_declared = _kernel_names(body)
_missing = [k for k in _declared if k not in K]
assert not _missing, ('__global__ kernels in td.cuh with no K entry: %s'
                      % ', '.join(_missing))
_extra = [k for k in K if k not in _declared]
assert not _extra, ('K names a kernel td.cuh does not declare: %s'
                    % ', '.join(_extra))

def split_top(s):
    out, cur, d = [], '', 0
    for ch in s:
        if ch in '(<[': d += 1
        if ch in ')>]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur); return out

# Parameters the host sometimes passes nullptr for. Metal has no unbound
# argument -- setBuffer:nil is an UNSET argument and dispatch is refused -- so
# these are declared under an MSL function constant. See plan 9z-c/9z-d.
#
# BY NAME, not by index, and k_resieve_rows is here because it was NOT.
# This table was a list of indices; k_resieve_rows was added to K in the
# 2026-10-15 rebase and never added here, so its `ispow` -- which
# pipeline_host.inc passes as nullptr -- was declared as an ordinary buffer.
# Metal said so exactly:
#     Compute Function(k_resieve_rows_4_1): missing Buffer binding at
#     index 2 for ispow[0]
# and NOTHING caught it, because --resieve-rows derives from --fill-rows,
# which this build defaults to 0, so validationcheck never launches the
# kernel. Indices are how that survives: they are correct until upstream
# inserts a parameter, and then they are silently about something else.
OPTIONAL_BUFFERS = {
    'k_td':              ['sel', 'nhit', 'fac', 'faccnt'],
    'k_resieve_scatter': ['ispow', 'walk_cur'],
    'k_resieve_rows':    ['ispow'],
}

_opt_seen = set()
_opt_indices = set()

MASK_DECL = ('/* Optional buffer arguments: metal_rt supplies bit i ='
             ' "argument i is bound". */\n'
             'constant uint mtl_bound_mask [[function_constant(0)]];\n')

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
    # MSL has no default kernel argument -- every parameter is a binding the
    # dispatch must set -- so the default is dropped here and moved to the
    # call site instead (portlib.pad_launch_defaults). Dropping it silently
    # would emit `device uint32_t *rt_out = nullptr` in the body template and
    # `constant uint32_t &n_rt = &0` in the wrapper, neither of which is legal.
    params = [p.split('=', 1)[0].rstrip() for p in split_top(body[a + 1:i])]
    brace = body.index('{', i); d = 0; j = brace
    while j < len(body):
        if body[j] == '{': d += 1
        elif body[j] == '}':
            d -= 1
            if d == 0: break
        j += 1
    inner = re.sub(r'\b__shared__\s+', 'threadgroup ', body[brace + 1:j])

    # A kernel that votes with __syncthreads_or needs the scratch word the
    # compat macro names -- k_resieve_bucket, pulled in from pipeline.cuh,
    # is the first kernel in this half to use it. Declared as a one-element
    # array so a plain kernel and a templated one spell the call site
    # identically. Same shape as gen_bench_kernels.py's.
    if '__syncthreads_or' in inner:
        inner = '    threadgroup uint _sync_or_flag[1];\n' + inner

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
            _want = OPTIONAL_BUFFERS.get(name, ())
            _pnames = [re.sub(r'.*?(\w+)\s*$', r'\1',
                              pp.strip().replace('*', ' ')) for pp in cps]
            for _w in _want:
                assert _w in _pnames, (
                    '%s has no parameter named %s (it has %s)'
                    % (name, _w, ', '.join(_pnames)))
            _opt_seen.add(name)
            opt = set(_pnames.index(w) for w in _want)
            _opt_indices |= opt
            kps = [bufparam(pp, k, opt) for k, pp in enumerate(cps)] + IDS
            suffix = '_'.join('1' if a == 'true' else '0' if a == 'false' else a for a in inst)
            decls = (chr(10).join(tg_bufs) + chr(10)) if tg_bufs else ''
            # A function-constant argument may be NAMED only where it exists,
            # so forward through a local that is null when it is absent. The
            # _body template is untouched and still tests the pointer.
            wargs = list(args)
            optdecl = ''
            for k in sorted(opt):
                pk = cps[k].strip(); an = argname(pk)
                optdecl += '    device %s = nullptr;\n' % re.sub(r'(\w+)\s*$', an + '_opt', pk)
                optdecl += '    if (%s) %s_opt = %s;\n' % (bound_name(k), an, an)
                wargs[k] = an + '_opt'
            wr.append('kernel void %s_%s(\n    %s)\n{\n%s%s    %s_body<%s>(%s);\n}\n'
                      % (name, suffix, ',\n    '.join(kps), decls, optdecl, name,
                         ', '.join(inst), ', '.join(wargs)))
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
    # The `(_opt)?` also covers the local the optional-buffer wrappers
    # forward through: a bare \b will not match it because `_` is a word
    # character, and a local declared `device ulong *` while the body expects
    # uint32_t is a compile error. That is how this was caught.
    body = re.sub(r'device ulong \*( ?)' + c + r'(_opt)?\b',
                  lambda mm, c=c: 'device uint32_t *' + mm.group(1) + c + (mm.group(2) or ''),
                  body)
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
# k_resieve_bucket's alias into its bucket argument, the same idiom.
_rec = 'const uint32_t *rec = bucket +'
assert body.count(_rec) == 1, 'k_resieve_bucket record alias shape changed'
body = body.replace(_rec, 'device const uint32_t *rec = bucket +', 1)
n_local += 1
if n_local: print('qualified %d local device-pointer alias(es)' % n_local)

# ---- bench.h constants this half reads ------------------------------------
# Lifted by name, never retyped; see gen_bench_kernels.py's copy. td.metal is
# its own translation unit, so it needs its own statement of ROWRT_NONE --
# the #ifndef guard is what keeps the two from fighting or from overriding a
# -D of the same name.
DEVICE_DEFINES = ['ROWRT_NONE']
_bh = open('bench/bench.h').read()
_lift = ''
for _nm in DEVICE_DEFINES:
    _m = re.search(r'^#define\s+' + _nm + r'\s+(.+?)\s*(?:/\*.*)?$', _bh, re.M)
    assert _m, 'bench.h does not define ' + _nm
    assert re.search(r'\b' + _nm + r'\b', body), (
        '%s is lifted from bench.h but td.cuh does not use it' % _nm)
    _lift += '#ifndef %s\n#define %s %s\n#endif\n' % (_nm, _nm, _m.group(1))
body = _lift + '\n' + body
print('  lifted %d bench.h #define(s)' % len(DEVICE_DEFINES))

# ---- k_td's congruence test: one struct load, not six field loads ---------
# Profiled against a GTX 1080 at the same q with identical work (17,625,929
# hits): the small-prime congruence test is 324.2 ms on the M3 against 81.2,
# a 3.99x gap and 80% of Metal's TD, while the division this port twice tried
# to optimise is only 1.68x. The test is the whole story.
#
# Per prime the loop reads SIX fields of tile[e] separately -- m, g, magic, rt,
# sh, cst -- from threadgroup memory, and every thread in the SIMD group reads
# the same element, so these are broadcasts. Copy the 32-byte struct once and
# let the compiler issue wide loads instead of six scalar ones.
_old_m = "                    const uint32_t m = tile[e].m, g = tile[e].g;"
_new_m = chr(10).join([
    "                    const tdsmall_t te = tile[e];",
    "                    const uint32_t m = te.m, g = te.g;"])
assert _old_m in body, 'congruence-test tile read shape changed'
body = body.replace(_old_m, _new_m, 1)
for _a, _b in (("if (tile[e].magic) {", "if (te.magic) {"),
               ("uint32_t w = tile[e].rt * jp + hi;", "uint32_t w = te.rt * jp + hi;"),
               ("td_mod_magic(w, m, tile[e].magic, tile[e].sh) != tile[e].cst",
                "td_mod_magic(w, m, te.magic, te.sh) != te.cst"),
               ("hitp[nh] = m * g; hitr[nh] = tile[e].recip; nh++;",
                "hitp[nh] = m * g; hitr[nh] = te.recip; nh++;")):
    assert _a in body, 'tile field read not found: ' + _a
    body = body.replace(_a, _b, 1)
print('  k_td congruence test: tile[e] loaded once per prime, not six times')

# The scan loop's tile[e] is copied once (see the note above); k_td_record_warp
# reads the same tile field-by-field, so give it the same treatment.
_te = body.count("const tdsmall_t te = tile[e];")
_wf_old = "                        const uint32_t m = tile[e].m, g = tile[e].g;"
if _wf_old in body:
    body = body.replace(_wf_old, chr(10).join([
        "                        const tdsmall_t te = tile[e];",
        "                        const uint32_t m = te.m, g = te.g;"]), 1)
    for _a, _b in (("if (tile[e].magic) {", "if (te.magic) {"),
                   ("uint32_t w = tile[e].rt * jp + hi;", "uint32_t w = te.rt * jp + hi;"),
                   ("td_mod_magic(w, m, tile[e].magic, tile[e].sh) != tile[e].cst",
                    "td_mod_magic(w, m, te.magic, te.sh) != te.cst")):
        if _a in body: body = body.replace(_a, _b, 1)
    _te += 1
print('  k_td: tile[e] copied once per prime in both variants (%d sites)' % _te)

_idx = sorted(_opt_indices)
body = (MASK_DECL + ''.join(
    'constant bool %s = (mtl_bound_mask & (1u << %d)) != 0;\n' % (bound_name(i), i)
    for i in _idx) + '\n') + body

open(OUT, 'w').write(body + '\n')
print('wrote %s (%d lines, %d heads qualified)' % (OUT, body.count('\n'), nq))
