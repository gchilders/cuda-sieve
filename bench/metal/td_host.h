/* SPDX-License-Identifier: LGPL-2.1-or-later
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

#define TD_GROUP_W 8
#define TD_GROUP_X (TD_GROUP_W * 32)
#define TD_SCAN_BLK 256
#define TD_TILE 512
#define TD_MAXHIT 16      /* buffered small-prime hits; ~7 per survivor typical */
#define TD_FMAX  64
#define TDF_NORM_OVERFLOW  1u
#define TDF_LIST_TRUNCATED 2u

#endif  /* CUDA_SIEVE_TD_HOST_H */
