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
                if d == 0 and text[m:m + 3] == '>>>': break
                d -= 1
            elif c == '(': d += 1
            elif c == ')': d -= 1
            m += 1
        cfg = text[j + 3:m]
        d = 0; parts = []; cur = ''; prev = ''
        for ch in cfg:
            if ch == '>' and prev == '-': pass          # an arrow, not a bracket
            elif ch in '(<[': d += 1
            elif ch in ')>]': d -= 1
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
       ('cudaGetDeviceProperties','mtlGetDeviceProperties')]


def apply_renames(text):
    for a, b in REN: text = text.replace(a, b)
    return text
