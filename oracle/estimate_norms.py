#!/usr/bin/env python3
"""
Estimate algebraic and rational norms for a GNFS/SNFS job file.
Helps determine which side to sieve.
"""

import sys
import re
import argparse
from math import log2, sqrt

def parse_job_file(filename):
    """Parse a .job file and return polynomial coefficients and parameters."""
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                # Strip invisible Unicode characters (zero-width spaces, etc.)
                val = re.sub(r'[^\x20-\x7E]', '', val)
                try:
                    # Try to parse as number
                    if '.' in val:
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except ValueError:
                    params[key] = val
    return params

def get_alg_poly(params):
    """Extract algebraic polynomial coefficients (highest degree first)."""
    coeffs = []
    # Find the highest degree
    max_deg = 0
    for key in params:
        if key.startswith('c') and key[1:].isdigit():
            deg = int(key[1:])
            max_deg = max(max_deg, deg)

    # Build coefficient list from highest to lowest degree
    for i in range(max_deg, -1, -1):
        key = f'c{i}'
        coeffs.append(params.get(key, 0))

    return coeffs

def get_rat_poly(params):
    """Extract rational polynomial coefficients (Y1*x + Y0)."""
    y1 = params.get('Y1', 0)
    y0 = params.get('Y0', 0)
    return [y1, y0]

def eval_homogeneous(coeffs, a, b):
    """
    Evaluate homogenized polynomial at (a, b).
    For poly = c_n*x^n + ... + c_0,
    homogenized = c_n*a^n + c_{n-1}*a^{n-1}*b + ... + c_0*b^n
    """
    n = len(coeffs) - 1
    result = 0
    for i, c in enumerate(coeffs):
        result += c * (a ** (n - i)) * (b ** i)
    return result

def algebraic_norm(alg_coeffs, a, b):
    """Calculate algebraic norm = |Res(a - bx, f(x))| = |b^deg * f(a/b)|"""
    return abs(eval_homogeneous(alg_coeffs, a, b))

def rational_norm(rat_coeffs, a, b):
    """Calculate rational norm = |Y1*a + Y0*b|"""
    return abs(eval_homogeneous(rat_coeffs, a, b))

def estimate_norms_at_point(params, a, b):
    """Estimate both norms at a single (a, b) point."""
    alg_coeffs = get_alg_poly(params)
    rat_coeffs = get_rat_poly(params)

    alg_norm = algebraic_norm(alg_coeffs, a, b)
    rat_norm = rational_norm(rat_coeffs, a, b)

    return alg_norm, rat_norm

def estimate_average_norms(params, I_bits=12, q_override=None):
    """
    Estimate average norms across a typical sieve region.
    I_bits: the sieve line size (e.g., 12 for gnfs-lasieve4I12e)
    q_override: if provided, use this special-q instead of the YAFU heuristic

    Uses YAFU's formula for computing representative (a, b) values:
    https://www.mersenneforum.org/showpost.php?p=571762&postcount=3
    """
    skew = params.get('skew', 1.0)

    # YAFU's formula for representative (a, b) at sieve scale
    # From factor/nfs/snfs.c lines 383-385
    if q_override is not None:
        q = float(q_override)
    else:
        q = 10000.0 * (10.0 ** (I_bits - 11))
    scale = (1 << (2 * I_bits - 1)) * q
    a_typical = sqrt(scale * skew)
    b_typical = sqrt(scale / skew)

    alg_coeffs = get_alg_poly(params)
    rat_coeffs = get_rat_poly(params)

    # Sample multiple points around the typical values
    sample_points = [
        (a_typical, b_typical),
        (-a_typical, b_typical),
        (a_typical * 0.5, b_typical),
        (a_typical, b_typical * 2),
    ]

    alg_norms = []
    rat_norms = []

    for a, b in sample_points:
        if b > 0:
            alg_norms.append(algebraic_norm(alg_coeffs, a, b))
            rat_norms.append(rational_norm(rat_coeffs, a, b))

    # Geometric mean
    alg_avg = 1
    rat_avg = 1
    for an, rn in zip(alg_norms, rat_norms):
        alg_avg *= an
        rat_avg *= rn
    alg_avg = alg_avg ** (1/len(alg_norms))
    rat_avg = rat_avg ** (1/len(rat_norms))

    return alg_avg, rat_avg, a_typical, b_typical, q

def main():
    parser = argparse.ArgumentParser(description="Estimate algebraic and rational norms for a GNFS/SNFS job file.")
    parser.add_argument('job_file', nargs='?', default='input.job', help='Job file (default: input.job)')
    parser.add_argument('--qrange', nargs=2, type=int, metavar=('Q_START_M', 'Q_END_M'),
                        help='Special-q range in millions (e.g. --qrange 30 230)')
    parser.add_argument('--siever', type=int, default=16, metavar='I_BITS',
                        help='Siever I value to use with --qrange (default: 16)')
    args = parser.parse_args()

    filename = args.job_file

    # Check if file exists
    import os
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        if filename == 'input.job':
            print("Usage: ./estimate_norms.py [job_file]")
            print("\nCreate an input.job file or specify a different .job file as argument.")
        sys.exit(1)

    print(f"Parsing job file: {filename}")
    print()

    params = parse_job_file(filename)

    # Display polynomials
    alg_coeffs = get_alg_poly(params)
    rat_coeffs = get_rat_poly(params)

    print("Algebraic polynomial (degree {}):"
          .format(len(alg_coeffs) - 1))
    terms = []
    deg = len(alg_coeffs) - 1
    for i, c in enumerate(alg_coeffs):
        if c != 0:
            d = deg - i
            if d == 0:
                terms.append(f"{c}")
            elif d == 1:
                terms.append(f"{c}*x")
            else:
                terms.append(f"{c}*x^{d}")
    print("  f(x) = " + " + ".join(terms).replace("+ -", "- "))

    print(f"\nRational polynomial (linear):")
    print(f"  g(x) = {rat_coeffs[0]}*x + {rat_coeffs[1]}")

    skew = params.get('skew', 1.0)
    print(f"Skewness: {skew}")

    # Large prime bounds
    lpba = params.get('lpba', 0)
    lpbr = params.get('lpbr', 0)
    alim = params.get('alim', 1e6)
    rlim = params.get('rlim', 1e6)
    mfba = params.get('mfba', 0)
    mfbr = params.get('mfbr', 0)
    alambda_current = params.get('alambda', 0)
    rlambda_current = params.get('rlambda', 0)

    print(f"\n--- Sieve Parameters ---")
    print(f"lpba (algebraic): {lpba} bits")
    print(f"lpbr (rational):  {lpbr} bits")
    print(f"alim: {alim:.2e}")
    print(f"rlim: {rlim:.2e}")
    print(f"mfba: {mfba} bits")
    print(f"mfbr: {mfbr} bits")

    # Lambda calculations
    print(f"\n--- Lambda Values ---")
    print(f"Current alambda: {alambda_current}")
    print(f"Current rlambda: {rlambda_current}")

    if mfba > 0 and alim > 0:
        alambda_suggested = mfba / log2(alim)
        print(f"Suggested alambda: {alambda_suggested:.4f} (mfba={mfba} / log2(alim={alim:.2e}))")
    else:
        print(f"Suggested alambda: (need mfba and alim)")

    if mfbr > 0 and rlim > 0:
        rlambda_suggested = mfbr / log2(rlim)
        print(f"Suggested rlambda: {rlambda_suggested:.4f} (mfbr={mfbr} / log2(rlim={rlim:.2e}))")
    else:
        print(f"Suggested rlambda: (need mfbr and rlim)")

    # Show norms for common siever sizes
    print(f"\n--- Norm Estimates by Siever ---")
    print(f"{'Siever':<20} {'q':>10} {'Alg bits':>10} {'Rat bits':>10} {'Diff':>8} {'Sieve side':<12}")
    print("-" * 72)

    for I_bits in [11, 12, 13, 14, 15, 16]:
        alg_avg, rat_avg, a_typ, b_typ, q = estimate_average_norms(params, I_bits)
        alg_bits = log2(alg_avg)
        rat_bits = log2(rat_avg)
        diff = rat_bits - alg_bits

        if diff > 0:
            side = "rational"
        else:
            side = "algebraic"

        siever_name = f"gnfs-lasieve4I{I_bits}e"
        q_exp = int(log2(q) / log2(10))
        q_str = f"1e{q_exp}"
        print(f"{siever_name:<20} {q_str:>10} {alg_bits:>10.1f} {rat_bits:>10.1f} {diff:>+8.1f} {side:<12}")

    # Q range analysis
    if args.qrange:
        q_start = args.qrange[0] * 1_000_000
        q_end = args.qrange[1] * 1_000_000
        I_bits = args.siever
        print(f"\n--- Norm Estimates for Q Range (I={I_bits}) ---")
        print(f"{'q':>14} {'Alg bits':>10} {'Rat bits':>10} {'Diff':>8} {'Sieve side':<12}")
        print("-" * 56)

        for q_val in [q_start, q_end]:
            alg_avg, rat_avg, a_typ, b_typ, q = estimate_average_norms(params, I_bits, q_override=q_val)
            alg_bits = log2(alg_avg)
            rat_bits = log2(rat_avg)
            diff = rat_bits - alg_bits
            side = "rational" if diff > 0 else "algebraic"
            print(f"{q_val:>14,} {alg_bits:>10.1f} {rat_bits:>10.1f} {diff:>+8.1f} {side:<12}")

    # Detailed analysis for a typical choice
    print(f"\n--- Detailed Analysis (I=15) ---")
    alg_avg, rat_avg, a_typ, b_typ, _ = estimate_average_norms(params, 15)
    alg_bits = log2(alg_avg)
    rat_bits = log2(rat_avg)

    print(f"Typical (a,b) region: |a| ~ {a_typ:.2e}, b ~ {b_typ:.2e}")
    print(f"Algebraic norm: ~{alg_bits:.1f} bits ({alg_avg:.2e})")
    print(f"Rational norm:  ~{rat_bits:.1f} bits ({rat_avg:.2e})")

    if alg_bits < rat_bits:
        print("\n=> Algebraic side has SMALLER norms")
        print("   Sieve the RATIONAL side (use -r or ensure rational sieving)")
    else:
        print("\n=> Rational side has SMALLER norms")
        print("   Sieve the ALGEBRAIC side (use -a or ensure algebraic sieving)")

if __name__ == "__main__":
    main()
