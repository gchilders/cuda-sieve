#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Shared by the host-side porting aids (gen_pipeline_host.py,
# gen_bench_host.py). NOT part of the build.
#
# One copy of the launch rewriter and the cuda* -> mtl* table, because two
# copies of a transformation this fiddly would drift -- and a drifting
# transformation produces a file that compiles and computes something else.
import re

MACRO_VALUES = {'NORM_CONST': '0', 'NORM_HORNER': '1', 'true': '1', 'false': '0'}

def mangle_parts(targs):
    """-> (name_expr, is_ternary). targs is the list of template arguments."""
    slab = [i for i, t in enumerate(targs) if t.strip() == 'SLABBED']
    def fixed(vals):
        return '_'.join(MACRO_VALUES.get(v.strip(), v.strip()) for v in vals)
    if not slab:
        return None, fixed(targs)
    a = list(targs); a[slab[0]] = 'false'
    b = list(targs); b[slab[0]] = 'true'
    return (fixed(a), fixed(b)), None

def rewrite_launches(text):
    out, i, n = [], 0, 0
    while True:
        j = text.find('<<<', i)
        if j < 0:
            out.append(text[i:]); break
        k = j; targs = None
        if text[k - 1] == '>':
            d = 0; k -= 1
            while k >= 0:
                if text[k] == '>': d += 1
                elif text[k] == '<':
                    d -= 1
                    if d == 0: break
                k -= 1
            targs = text[k + 1:j - 1]
        e = k
        while e > 0 and (text[e - 1].isalnum() or text[e - 1] == '_'): e -= 1
        base = text[e:k]
        # `<<<` also appears in prose inside comments; only a real kernel
        # launch has a k_-prefixed name immediately before it.
        if not base.startswith('k_'):
            out.append(text[i:j + 3]); i = j + 3; continue
        # Scan to the closing >>>. The launch configuration can contain `->`
        # (S->apply_smem) and std::min<T>(...), so a bare `>` is only a bracket
        # when it is not the tail of an arrow.
        d = 0; m = j + 3
        while m < len(text):
            c = text[m]
            if c == '>' and text[m - 1] == '-':
                m += 1; continue
            if c == '<': d += 1
            elif c == '>':
                if d == 0:
                    # Nothing is open, so this `>` closes nothing: it is either
                    # the launch terminator or an OPERATOR. `xmax >> log_region`
                    # in a grid expression (pipeline.cuh's k_resieve_bucket) used
                    # to drive d negative here, after which the real `>>>` never
                    # matched and the scan ran off the end of the file.
                    if text[m:m + 3] == '>>>': break
                else:
                    d -= 1
            elif c == '(': d += 1
            elif c == ')': d -= 1
            m += 1
        assert text[m:m + 3] == '>>>', 'unterminated launch configuration at %d' % j
        cfg = text[j + 3:m]
        d = 0; parts = []; cur = ''; prev = ''
        for ch in cfg:
            if ch == '>' and prev == '-': pass          # an arrow, not a bracket
            elif ch in '(<[': d += 1
            # Same rule as the terminator scan above: at depth 0 a `>` or `]`
            # closes nothing, so it is an operator (a right shift in a grid
            # expression) and must not drive the depth negative -- which would
            # then swallow the comma that separates grid from block.
            elif ch in ')>]': d -= 1 if d > 0 else 0
            if ch == ',' and d == 0: parts.append(cur); cur = ''
            else: cur += ch
            prev = ch
        parts.append(cur)
        grid, block = parts[0].strip(), parts[1].strip()
        smem = parts[2].strip() if len(parts) > 2 else '0'
        stream = parts[3].strip() if len(parts) > 3 else '0'
        a = text.index('(', m); d = 0; b = a
        while b < len(text):
            if text[b] == '(': d += 1
            elif text[b] == ')':
                d -= 1
                if d == 0: break
            b += 1
        args = ' '.join(text[a + 1:b].replace('\\\n', ' ').replace('\\', ' ').split())
        # NULL is 0L in C++, not a pointer, so it would bind as an INTEGER and
        # hand the kernel a non-nil address for an argument it tests with
        # `if (p)`. nullptr has its own overload in metal_rt.h. See the comment
        # there -- this cost a GPU page fault to find.
        args = re.sub(r'\bNULL\b', 'nullptr', args)

        out.append(text[i:e])
        if targs is None:
            out.append('MTL_LAUNCH(%s, %s, %s, %s, %s, %s)'
                       % (base, grid, block, smem, stream, args))
        else:
            tern, fixed = mangle_parts([t for t in targs.split(',')])
            if fixed is not None:
                out.append('MTL_LAUNCH(%s_%s, %s, %s, %s, %s, %s)'
                           % (base, fixed, grid, block, smem, stream, args))
            else:
                name = '(SLABBED ? "%s_%s" : "%s_%s")' % (base, tern[1], base, tern[0])
                out.append('MTL_LAUNCH_NAMED(%s, %s, %s, %s, %s, %s)'
                           % (name, grid, block, smem, stream, args))
        n += 1
        i = b + 1
    return ''.join(out), n


REN = [('cudaDeviceSynchronize','mtlDeviceSynchronize'), ('cudaGetLastError','mtlGetLastError'),
       ('cudaGetErrorString','mtlGetErrorString'), ('cudaMemcpyDeviceToHost','mtlMemcpyDeviceToHost'),
       ('cudaMemcpyHostToDevice','mtlMemcpyHostToDevice'),
       ('cudaMemcpyDeviceToDevice','mtlMemcpyDeviceToDevice'),
       ('cudaEventElapsedTime','mtlEventElapsedTime'), ('cudaEventSynchronize','mtlEventSynchronize'),
       ('cudaEventDestroy','mtlEventDestroy'), ('cudaEventCreate','mtlEventCreate'),
       ('cudaEventRecord','mtlEventRecord'), ('cudaEventQuery','mtlEventQuery'),
       ('cudaEvent_t','mtlEvent_t'),
       ('cudaStreamCreateWithFlags','mtlStreamCreateWithFlags'),
       ('cudaStreamCreateWithPriority','mtlStreamCreateWithPriority'),
       ('cudaDeviceGetStreamPriorityRange','mtlDeviceGetStreamPriorityRange'),
       ('cudaStreamNonBlocking','mtlStreamNonBlocking'),
       ('cudaEventDisableTiming','mtlEventDisableTiming'),
       ('cudaErrorMemoryAllocation','mtlErrorMemoryAllocation'),
       ('cudaStreamCreate','mtlStreamCreate'), ('cudaStreamDestroy','mtlStreamDestroy'),
       ('cudaStreamSynchronize','mtlStreamSynchronize'), ('cudaStreamWaitEvent','mtlStreamWaitEvent'),
       ('cudaStream_t','mtlStream_t'),
       ('cudaMemcpyAsync','mtlMemcpyAsync'), ('cudaMemsetAsync','mtlMemsetAsync'),
       ('cudaMemcpy2D','mtlMemcpy2D'), ('cudaMemcpy','mtlMemcpy'), ('cudaMemset','mtlMemset'),
       ('cudaMalloc','mtlMalloc'), ('cudaFreeHost','mtlFreeHost'),
       ('cudaHostAllocDefault','mtlHostAllocDefault'), ('cudaHostAlloc','mtlHostAlloc'),
       ('cudaFree','mtlFree'), ('cudaError_t','mtlError_t'), ('cudaSuccess','mtlSuccess'),
       ('cudaErrorLaunchTimeout','mtlErrorLaunchTimeout'), ('cudaErrorNotReady','mtlErrorNotReady'),
       ('cudaMemGetInfo','mtlMemGetInfo'), ('cudaDeviceProp','mtlDeviceProp'),
       ('cudaGetDeviceProperties','mtlGetDeviceProperties'),
       ('cudaGetDeviceCount','mtlGetDeviceCount'),
       ('cudaSetDevice','mtlSetDevice'), ('cudaGetDevice','mtlGetDevice')]


def apply_renames(text):
    for a, b in REN: text = text.replace(a, b)
    return text


# ---- default kernel arguments -------------------------------------------
#
# CUDA kernels may declare defaulted trailing parameters (upstream 2a3dfb0
# added them to k_transform, k_fill_atomic, k_fill_rows and k_apply), and a
# launch that omits one is legal. Metal has no such thing: every argument a
# kernel declares is a binding the dispatch must set, and an unset one is a
# validation abort, not a default.
#
# So the DEFAULT IS MOVED TO THE CALL SITE. Each launch's argument list is
# padded from the declaration's own defaults, which keeps one definition of
# each default -- the kernel's -- rather than retyping it per call site, and
# fails loudly if a parameter gains a default this cannot supply.
def _strip_comments(t):
    t = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), t, flags=re.S)
    return re.sub(r'//[^\n]*', '', t)


def _split_args(s, opens='(<[', closes=')>]'):
    out, cur, d, q = [], '', 0, ''
    i = 0
    while i < len(s):
        ch = s[i]
        if q:
            cur += ch
            if ch == '\\': cur += s[i + 1:i + 2]; i += 2; continue
            if ch == q: q = ''
            i += 1; continue
        if ch in '"\'': q = ch; cur += ch; i += 1; continue
        if ch in opens: d += 1
        elif ch in closes: d -= 1 if d > 0 else 0
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
        i += 1
    out.append(cur)
    return out


def kernel_param_defaults(paths):
    """name -> [default expression or None] for every __global__ in paths."""
    out = {}
    for path in paths:
        t = _strip_comments(open(path).read())
        for m in re.finditer(r'__global__', t):
            rest = re.sub(r'^\s*__launch_bounds__\([^)]*\)', '', t[m.end():])
            a = rest.index('(')
            name = re.findall(r'(\w+)\s*$', rest[:a])[-1]
            d, b = 0, a
            while b < len(rest):
                if rest[b] == '(': d += 1
                elif rest[b] == ')':
                    d -= 1
                    if d == 0: break
                b += 1
            out[name] = [p.split('=', 1)[1].strip() if '=' in p else None
                         for p in _split_args(rest[a + 1:b])]
    return out


def pad_launch_defaults(text, sigs, allow_unresolved=()):
    """Append each launch's omitted trailing arguments. Returns (text, n, bad)."""
    names = sorted(sigs, key=len, reverse=True)
    out, i, npad, unresolved = [], 0, 0, set()
    while True:
        m = re.search(r'\bMTL_LAUNCH(_NAMED)?\s*\(', text[i:])
        if not m:
            out.append(text[i:]); break
        s = i + m.start(); a = i + m.end() - 1
        out.append(text[i:a + 1])
        d, b = 0, a
        while b < len(text):
            if text[b] == '(': d += 1
            elif text[b] == ')':
                d -= 1
                if d == 0: break
            b += 1
        parts = _split_args(text[a + 1:b], '([{', ')]}')
        who = parts[0].strip()
        lit = re.search(r'"([A-Za-z_][A-Za-z_0-9]*)', who)
        base = lit.group(1) if lit else who
        hit = next((k for k in names if base.startswith(k)), None)
        if hit is None:
            unresolved.add(who)
        else:
            defs = sigs[hit]
            have = len(parts) - 5          # name, grid, block, smem, stream
            if have < len(defs):
                add = defs[have:]
                assert all(x is not None for x in add), (
                    '%s: launch passes %d of %d arguments and parameter %d has '
                    'no default' % (hit, have, len(defs), have))
                parts += [' ' + x for x in add]
                npad += 1
            assert len(parts) - 5 == len(defs), (
                '%s: launch passes %d arguments, the kernel declares %d'
                % (hit, len(parts) - 5, len(defs)))
        out.append(','.join(parts) + ')')
        i = b + 1
    return ''.join(out), npad, unresolved - set(allow_unresolved)
