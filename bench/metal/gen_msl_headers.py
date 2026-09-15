#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Makes MSL copies of the tree's
# pure-arithmetic headers -- plattice.cuh, bigint.cuh, prp.cuh -- which stay
# UNTOUCHED and shared between the CUDA and CPU builds. The arithmetic is
# verbatim; the only changes are MSL's mandatory address-space qualifiers and
# the handful of host-only constructs MSL has no equivalent for.
import re, sys

def split_top(s):
    out, cur, d = [], '', 0
    for ch in s:
        if ch in '(<[': d += 1
        if ch in ')>]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur); return out

def qualify(body, device_params=()):
    head = re.compile(r'(static inline[^;{()]*?\b(\w+)\s*\()([^{;]*?)(\)\s*\n?\s*\{)', re.S)
    def q(params):
        res = []
        for p in split_top(params):
            if '*' not in p or re.search(r'\b(thread|constant|device|threadgroup)\b', p):
                res.append(p); continue
            nm = re.sub(r'.*?(\w+)\s*$', r'\1', p.strip().replace('*', ' '))
            space = ('device ' if nm in device_params else 'thread ')
            st = p.lstrip(); pad = p[:len(p) - len(st)]
            res.append(pad + ('const ' + space + st[6:] if st.startswith('const ')
                              else space + st))
        return ','.join(res)
    return head.subn(lambda m: m.group(1) + q(m.group(3)) + m.group(4), body)

OLD_BN = 'static inline double bn_to_double(const thread bn_t *x)\n{\n    double d = 0.0;\n    for (int i = BN_LIMBS - 1; i >= 0; i--) d = d * 4294967296.0 + (double)x->v[i];\n    return d;\n}'
NEW_BN = "/* bn_to_double: Apple GPUs have no `double`, so this is sf_bn_to_double in\n * metal/sf_sites.h -- the same accumulation, on softfp64.h, proven bit-exact\n * against prp.cuh's own fp64 by the Phase 2 gate. */"
OLD_GAP = '        double nd = bn_to_double(n);\n        double kB = lim * lim;\n        for (unsigned klpb = lpb; klpb < (unsigned)bits; klpb += lpb, kB *= lim)\n            if (nd < kB) return COF_REJECT_GAP;'
NEW_GAP = '        /* The gap test, on softfp64: same operations, same order, same\n         * rounding. sf_cof_gap_test returns 1 where the original returns\n         * COF_REJECT_GAP. */\n        if (sf_cof_gap_test(sf_bn_to_double(n->v, BN_LIMBS), bits, lpb, lim))\n            return COF_REJECT_GAP;'

def convert(src_path, out_path, guard, fn_macros, first_marker, extra_head='',
            drop_host=True, device_params=(), drop_fns=(), end_marker=None,
            extra_defines_from=None):
    src = open(src_path).read()
    end = src.index(end_marker) if end_marker else src.rindex('#endif')
    body = src[src.index(first_marker):end]
    for m in fn_macros:
        body = body.replace(m + ' ', 'static inline ')
    # some of those macros already expand to `static inline` in the non-CUDA
    # branch, so collapse the doubling rather than emitting a warning
    body = body.replace('static inline static inline ', 'static inline ')
    body = body.replace('__device__ __forceinline__ ', 'static inline ')
    body = body.replace('__host__ __device__ ', '')
    body = body.replace('__device__ ', 'static inline ')
    body = body.replace('__restrict', '')
    body = body.replace('unsigned long long', 'ulong').replace('long long', 'long')
    body = body.replace('__umulhi(', 'mulhi(').replace('__umul64hi(', 'mulhi(')

    # A #if defined(__CUDA_ARCH__) block selects the device intrinsic, and on
    # MSL that branch is always the right one. This MUST be done with a real
    # preprocessor walk, not a regex: bigint.cuh's block is
    #     #if defined(__CUDA_ARCH__) / #elif defined(_MSC_VER) / #else / #endif
    # and a regex that assumes a bare #else leaves the #elif orphaned. That
    # silently turned the whole REST OF THE HEADER into a dead branch -- the
    # file still compiled, and every symbol after it simply ceased to exist.
    # A missing-code bug, not a wrong-branch one, which is exactly the class
    # the HIP port's ledger warns about.
    out, i, depth, taken = [], 0, 0, []
    src_lines = body.split(chr(10))
    while i < len(src_lines):
        l = src_lines[i]
        t = l.strip()
        if t.startswith('#if') and '__CUDA_ARCH__' in t:
            # keep the device branch, drop every alternative through #endif
            d = 1; i += 1
            while i < len(src_lines):
                t2 = src_lines[i].strip()
                if t2.startswith('#if'): d += 1
                elif t2.startswith('#endif'):
                    d -= 1
                    if d == 0: i += 1; break
                elif d == 1 and (t2.startswith('#else') or t2.startswith('#elif')):
                    # skip to the matching #endif
                    d2 = 1; i += 1
                    while i < len(src_lines):
                        t3 = src_lines[i].strip()
                        if t3.startswith('#if'): d2 += 1
                        elif t3.startswith('#endif'):
                            d2 -= 1
                            if d2 == 0: i += 1; break
                        i += 1
                    break
                out.append(src_lines[i]); i += 1
            taken.append(t[:40])
            continue
        out.append(l); i += 1
    body = chr(10).join(out)
    if taken: print('  took the __CUDA_ARCH__ branch in', len(taken), 'block(s)')

    if drop_host:
        out, i, dropped = [], 0, []
        lines = body.split('\n')
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
        body = '\n'.join(out)
        if dropped: print('  dropped host-only:', ' '.join(dropped))

    # A stray #elif/#else would make the rest of the header a dead branch
    # that still compiles. Refuse to emit one.
    d = 0
    for l in body.split(chr(10)):
        t = l.strip()
        if t.startswith('#if'): d += 1
        elif t.startswith('#endif'): d -= 1
        elif t.startswith(('#else', '#elif')):
            assert d > 0, 'orphaned ' + t + ' -- the rest of the header would vanish'
        assert d >= 0, 'unbalanced #endif near: ' + t
    assert d == 0, 'unbalanced preprocessor conditionals in ' + out_path

    # Named functions with no device caller. bn_to_dec is decimal formatting
    # and needs BN_DEC_MAX, a host-side buffer bound.
    for fn in drop_fns:
        m = re.search(r'(/\*[^/]*?\*/\s*)?static inline [^;{]*?\b' + fn + r'\s*\(', body, re.S)
        if m:
            d = 0; k = body.index('{', m.end())
            j = k
            while j < len(body):
                if body[j] == '{': d += 1
                elif body[j] == '}':
                    d -= 1
                    if d == 0: break
                j += 1
            body = body[:m.start()] + body[j + 1:]
            print('  dropped (no device caller):', fn)

    body, n = qualify(body, device_params)
    if 'bn_to_double' in body:
        assert OLD_BN in body, 'prp.cuh bn_to_double changed'
        body = body.replace(OLD_BN, NEW_BN)
        assert OLD_GAP in body, 'prp.cuh gap test changed'
        body = body.replace(OLD_GAP, NEW_GAP)
        # the `lim` parameter is a binary64 value; carry it as one
        body = body.replace('unsigned mfb,' + chr(10) + '                        double lim)',
                            'unsigned mfb,' + chr(10) + '                        sf64 lim)')
        print('  rewrote bn_to_double and the gap test onto softfp64')
    hdr = ('/* SPDX-License-Identifier: LGPL-2.1-or-later\n'
           ' *\n'
           ' * %s as MSL. Generated by metal/gen_msl_headers.py and committed;\n'
           ' * the arithmetic is verbatim and the only changes are MSL address-space\n'
           ' * qualifiers. %s itself is UNTOUCHED and stays shared between the\n'
           ' * CUDA build and the CPU reference.\n'
           ' */\n#ifndef %s\n#define %s\n\n#include <metal_stdlib>\nusing namespace metal;\n'
           '\n/* limits <stdint.h> would provide; MSL has no such header. */\n'
           '#ifndef UINT32_MAX\n#define UINT32_MAX 0xffffffffu\n#endif\n'
           '#ifndef UINT64_MAX\n#define UINT64_MAX 0xfffffffffffffffful\n#endif\n'
           '#ifndef INT64_MAX\n#define INT64_MAX  0x7fffffffffffffffl\n#endif\n%s\n'
           % (src_path.split('/')[-1], src_path.split('/')[-1], guard, guard, extra_head))
    # Some #defines sit INSIDE the source's own __CUDACC__ guard (td.cuh's
    # TD_FMAX and friends), so the extracted section misses them -- yet other
    # Metal translation units need them. Pull them across by name so there is
    # still exactly one statement of each per build.
    if extra_defines_from:
        path, names = extra_defines_from
        src2 = open(path).read()
        picked = []
        for nm in names:
            m = re.search(r'^#define\s+' + nm + r'\b[^\n]*$', src2, re.M)
            if m: picked.append(m.group(0))
        if picked:
            body = ('\n/* Bounds the source keeps inside its own __CUDACC__ guard;\n'
                    ' * lifted here so every Metal translation unit shares one copy. */\n'
                    + '\n'.join(picked) + '\n\n' + body)
            print('  lifted %d #define(s) from inside the guard' % len(picked))

    open(out_path, 'w').write(hdr + body + '\n#endif  /* %s */\n' % guard)
    print('  wrote %s (%d heads qualified)' % (out_path, n))

print('bigint.cuh:')
convert('bench/bigint.cuh', 'bench/metal/bigint_msl.h', 'CUDA_SIEVE_BIGINT_MSL_H',
        ['BN_FN'], 'typedef struct',
        extra_head='\n#ifndef BN_LIMBS\n#define BN_LIMBS 12\n#endif\n',
        drop_fns=('bn_to_dec',))
print('prp.cuh:')
convert('bench/prp.cuh', 'bench/metal/prp_msl.h', 'CUDA_SIEVE_PRP_MSL_H',
        ['PRP_FN'], '#define M_LIMBS',
        extra_head='\n#include "bigint_msl.h"\n#include "sf_sites.h"\n')
print('slab.h:')
convert('bench/slab.h', 'bench/metal/slab_msl.h', 'CUDA_SIEVE_SLAB_MSL_H',
        ['SLAB_HD'], 'typedef struct')
print('td.cuh (shared section):')
# Everything td.cuh defines ABOVE its __CUDACC__ guard: the types, the TD_*
# bounds, td_mod_magic and SS_KSHIFT. Both td.metal and bench_kernels.metal
# include this, which is what lets bench_kernels.metal stop carrying its own
# copies -- td.cuh calls SS_KSHIFT "the single source of truth for the bias
# shift", so there must be exactly one statement of it per build.
convert('bench/td.cuh', 'bench/metal/td_msl.h', 'CUDA_SIEVE_TD_MSL_H',
        ['TD_MOD_HD'], 'typedef struct',
        extra_head='\n#include "bigint_msl.h"\n'
                   '#ifndef BENCH_MAX_DEGREE\n#define BENCH_MAX_DEGREE 8\n#endif\n'
                   '#ifndef BENCH_NCOEFF\n#define BENCH_NCOEFF (BENCH_MAX_DEGREE + 1)\n#endif\n',
        end_marker='#if defined(__CUDACC__)',
        extra_defines_from=('bench/td.cuh',
                            ('TD_GROUP_W', 'TD_GROUP_X', 'TD_SCAN_BLK',
                             'TD_TILE', 'TD_MAXHIT', 'TD_FMAX')))
print('plattice.cuh:')
convert('bench/plattice.cuh', 'bench/metal/plattice_msl.h', 'CUDA_SIEVE_PLATTICE_MSL_H',
        ['PL_FN'], 'typedef struct')
