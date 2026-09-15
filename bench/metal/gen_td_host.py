#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Porting aid, NOT part of the build. Emits metal/td_host.h: the bounds td.cuh
# keeps INSIDE its own `#if defined(__CUDACC__)` guard, which a host
# translation unit therefore never sees.
#
# This is the host counterpart of td_msl.h and it exists to kill the last
# hand-copied constants in the port. Lifted BY NAME from td.cuh, so the values
# track the original instead of drifting.
#
# The cleaner fix is a CUDA-side change -- move these defines above the guard,
# where they belong, since they are not device-only in any meaningful sense.
# That needs a drift-ledger row, so it is deliberately not done here.
import re

SRC = 'bench/td.cuh'
OUT = 'bench/metal/td_host.h'
NAMES = ('TD_GROUP_W', 'TD_GROUP_X', 'TD_SCAN_BLK', 'TD_TILE', 'TD_MAXHIT',
         'TD_FMAX', 'TDF_NORM_OVERFLOW', 'TDF_LIST_TRUNCATED')

src = open(SRC).read()
picked = []
for nm in NAMES:
    m = re.search(r'^#define\s+' + nm + r'\b[^\n]*$', src, re.M)
    if not m:
        raise SystemExit('td.cuh no longer defines ' + nm)
    # Guarded, not bare -- see gen_msl_headers.py: an unguarded lifted define
    # overrides a -D of the same name and desyncs host from device.
    picked.append('#ifndef %s\n%s\n#endif' % (nm, m.group(0)))

open(OUT, 'w').write(
'''/* SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Bounds that td.cuh keeps inside its own `#if defined(__CUDACC__)` guard, so
 * a host translation unit never sees them. Generated from td.cuh by
 * metal/gen_td_host.py -- lifted by name, so they track the original rather
 * than drifting as hand copies. The MSL side gets the same values through
 * td_msl.h.
 *
 * The cleaner fix is a CUDA-side change: move these above the guard, since
 * nothing about them is device-only. That needs a drift-ledger row and is
 * deliberately not done here.
 */
#ifndef CUDA_SIEVE_TD_HOST_H
#define CUDA_SIEVE_TD_HOST_H

''' + '\n'.join(picked) + '''

#endif  /* CUDA_SIEVE_TD_HOST_H */
''')
print('wrote %s (%d defines lifted from td.cuh)' % (OUT, len(picked)))
