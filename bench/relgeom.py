#!/usr/bin/env python3
"""relgeom.py -- recover the sieve rectangle a relation file actually covers,
and compare two files position by position.

WHY THIS EXISTS. Findings 58 and 63 spent two sessions explaining a 14% yield
"deficit" that was a geometry mismatch: `gnfs-lasieve4I15e -J 15` is documented
as "specify J bits" and in fact widens I, sieving 2^16 x 2^14 where our
--J 32768 sieves 2^15 x 2^15. Nothing in either siever's output says which
rectangle it used, so the comparison looked sound and was not. Finding 65 is
the correction.

The fix is to stop trusting flags and ask the relations. A relation (a,b) found
under special-q (q, rho) sits at the lattice point

    [a]   [a0  b0] [i]
    [b] = [a1  b1] [j],   the same reduced basis qlat_build() gives the sieve,

so inverting a 2x2 integer matrix recovers (i,j) exactly. The extent of those
points IS the rectangle, whatever the command line claimed.

Run it before any cross-siever yield claim.

    ./relgeom.py --band QLO:QHI --skew S extent FILE...
    ./relgeom.py --band QLO:QHI --skew S compare OURS THEIRS J LOGI

Both sievers' GGNFS-format output is read directly: `a,b:rat_hex:alg_hex`.
"""
import sys

# The special-q is identified as the algebraic prime lying in the swept q
# window, so the window and the skew must be supplied. Both are REQUIRED
# arguments rather than defaults: a wrong window filters every candidate list
# to empty and the tool then reports a clean-looking zero, and a wrong skew
# silently produces a different lattice reduction -- which is the one thing
# qlat_build must not do. Defaults would make both failures quiet.


def qlat_build(q, rho, skew):
    """Exact mirror of poly.c qlat_build(): Gauss reduction under the skewed
    norm, same fp64 arithmetic, same round-half-away-from-zero. A different
    reduction would put the same relation at a different (i,j), so this must
    not drift from the C."""
    a0, a1, b0, b1 = q, 0, rho, 1
    wa, wb = 1.0 / skew, skew
    for _ in range(200):
        na = wa * float(a0) * a0 + wb * float(a1) * a1
        nb = wa * float(b0) * b0 + wb * float(b1) * b1
        if nb < na:                      # keep (a0,a1) the shorter vector
            a0, b0 = b0, a0
            a1, b1 = b1, a1
            na = nb
        mu = (wa * float(a0) * b0 + wb * float(a1) * b1) / na
        m = int(mu - 0.5) if mu < 0 else int(mu + 0.5)
        if m == 0:
            break
        b0 -= m * a0
        b1 -= m * a1
    return a0, a1, b0, b1


def detect_side(path, band, probe=2000):
    """Which field holds the special-q, 1 = rational or 2 = algebraic.

    Hardcoding the algebraic field made the tool silently map NOTHING on a
    rational special-q run (`--sq-side 0`), and then blame the band for it.
    The band primes are overwhelmingly in the sq-side field, so counting both
    over a short probe decides it without the caller having to know."""
    n = [0, 0, 0]
    seen = 0
    for ln in open(path):
        p = ln.strip().split(':')
        if len(p) != 3:
            continue
        for f in (1, 2):
            try:
                n[f] += sum(1 for x in p[f].split(',')
                            if x and band[0] <= int(x, 16) < band[1])
            except ValueError:
                pass
        seen += 1
        if seen >= probe:
            break
    return 1 if n[1] >= n[2] else 2


def read(path, band, side):
    """-> {(a,b): [candidate special-q]}. A relation carries every band prime
    dividing its sq-side norm, not just the one that produced it (P(re-found)
    is ~70% under the full-base convention), so the producing q is chosen
    later by which one puts the point inside a rectangle."""
    out = {}
    for ln in open(path):
        p = ln.strip().split(':')
        if len(p) != 3:
            continue
        try:
            a, b = (int(x) for x in p[0].split(','))
            fac = [int(x, 16) for x in p[side].split(',') if x]
        except ValueError:
            continue
        out[(a, b)] = [x for x in fac if band[0] <= x < band[1]]
    return out


class Mapper:
    def __init__(self, skew):
        self.skew = skew
        self.lat = {}

    def ij(self, a, b, q):
        """One lattice per (q, RHO), never per q: this polynomial averages 1.53
        roots per prime and each root is a different lattice."""
        try:
            rho = (a * pow(b, -1, q)) % q
        except ValueError:
            return None                  # b == 0 mod q: projective root
        key = (q, rho)
        if key not in self.lat:
            self.lat[key] = qlat_build(q, rho, self.skew)
        a0, a1, b0, b1 = self.lat[key]
        det = a0 * b1 - b0 * a1
        ni, nj = a * b1 - b0 * b, a0 * b - a1 * a
        if det == 0 or ni % det or nj % det:
            return None
        i, j = ni // det, nj // det
        # F(a,b) and F(-a,-b) differ only in sign; the sieve visits j >= 1, so
        # the antipodal point is the one it actually looked at.
        return (-i, -j) if j < 0 else (i, j)

    def best(self, a, b, qs, shape=1.0):
        """Among candidate q, pick the one whose point sits closest to the
        origin -- a re-find at another band q lands far outside its own
        rectangle.

        `shape` is J/(I/2), so the metric is measured in units of the rectangle
        rather than in raw coordinates. A square metric biases the answer
        toward square: on a 2^16 x 2^14 run the true point (30000, 900) loses
        to a re-find at (5000, 6000), and `extent` would then under-report the
        very width it exists to detect. extent() iterates: square metric first,
        then re-picks with the shape that produced."""
        got = gotm = None
        for q in qs:
            p = self.ij(a, b, q)
            if p is None:
                continue
            m = max(abs(p[0]) * shape, float(p[1]))
            if gotm is None or m < gotm:
                got, gotm = p, m
        return got

    def inside(self, a, b, qs, J, logI):
        half = 1 << (logI - 1)
        for q in qs:
            p = self.ij(a, b, q)
            if p and 1 <= p[1] <= J and -half <= p[0] < half:
                return p
        return None


def _pow2_ceil_log2(n):
    return max(0, (n - 1).bit_length()) if n > 1 else 0


def extent(paths, band, skew, side=None):
    for path in paths:
        sd = side or detect_side(path, band)
        M = Mapper(skew)
        rel = read(path, band, sd)
        shape, pts, bad = 1.0, [], 0
        for _ in range(3):                      # converges in one or two
            pts = [M.best(a, b, qs, shape) for (a, b), qs in rel.items()]
            bad = sum(1 for p in pts if p is None)
            pts = [p for p in pts if p]
            if not pts:
                break
            ihalf = max(max(-min(i for i, _ in pts), max(i for i, _ in pts) + 1), 1)
            jmax = max(j for _, j in pts)
            new_shape = jmax / float(ihalf)
            if abs(new_shape - shape) < 1e-9:
                break
            shape = new_shape
        if not pts:
            print("%s: nothing mapped (wrong --band?)" % path)
            continue
        imin, imax = min(i for i, _ in pts), max(i for i, _ in pts)
        jmax = max(j for _, j in pts)
        # i runs over [-I/2, I/2), so the half-width is max(-imin, imax+1) --
        # NOT max|i|, which is I/2 only when the legal i = -I/2 column happens
        # to be occupied and I/2 - 1 otherwise, a factor-of-two swing in the
        # reported I. Round the observed coverage UP to a power of two.
        ihalf = max(-imin, imax + 1)
        frac = bad / float(bad + len(pts))
        print("%s: %d mapped, %d unmapped (sq side = field %d)"
              % (path, len(pts), bad, sd))
        # A band that is merely WRONG maps nothing and is caught below. A band
        # that PARTLY overlaps is worse: a handful of accidental hits still
        # produce a confident-looking rectangle. Say so rather than print it.
        if frac > 0.02:
            print("   ** %.1f%% of relations did not map -- the band or skew is"
                  " probably wrong;" % (100 * frac))
            print("      the rectangle below is derived from the remainder and"
                  " should not be trusted.")
        print("   i in [%d, %d]   j in [1, %d]" % (imin, imax, jmax))
        print("   rectangle: I = 2^%d  x  J = 2^%d   (from observed coverage,"
              " so a LOWER bound)"
              % (_pow2_ceil_log2(ihalf) + 1, _pow2_ceil_log2(jmax)))


def compare(ours_p, theirs_p, J, logI, band, skew, nb=8, side=None):
    M = Mapper(skew)
    so_side = side or detect_side(ours_p, band)
    th_side = side or detect_side(theirs_p, band)
    ours, theirs = read(ours_p, band, so_side), read(theirs_p, band, th_side)

    # Restrict to the special-q BOTH sievers covered FIRST. A partial or
    # differently bounded run must not read as a yield difference -- and the
    # headline ratio is the number a reader quotes, so it is the one that has
    # to be restricted. Computing it over the raw sets and only binning the
    # histogram by common q (which is what this did) reports e.g. 0.80 for two
    # runs that are at exact parity over the q they share.
    qo = {q for v in ours.values() for q in v}
    qt = {q for v in theirs.values() for q in v}
    common = qo & qt
    print("special-q: ours %d theirs %d common %d" % (len(qo), len(qt), len(common)))
    if not common:
        print("NO special-q in common -- is --band right for these files?")
        sys.exit(1)

    ours = {k: v for k, v in ours.items() if any(q in common for q in v)}
    theirs = {k: v for k, v in theirs.items() if any(q in common for q in v)}
    so, st = set(ours), set(theirs)
    both, only_o, only_t = so & st, so - st, st - so
    print("over the common q:  ours %d   theirs %d" % (len(so), len(st)))
    print("both %d   ours-only %d   theirs-only %d" % (len(both), len(only_o), len(only_t)))
    print("recall %.4f   ours/theirs %.4f\n"
          % (len(both) / max(1, len(st)), len(so) / max(1, len(st))))

    hist = {k: [0] * nb for k in ("both", "o", "t")}
    outside = dict.fromkeys(hist, 0)
    for name, s, src in (("both", both, ours), ("o", only_o, ours), ("t", only_t, theirs)):
        for ab in s:
            qs = [q for q in src[ab] if q in common]
            if not qs:
                continue
            p = M.inside(ab[0], ab[1], qs, J, logI)
            if p is None:
                outside[name] += 1
                continue
            hist[name][min(nb - 1, (p[1] - 1) * nb // J)] += 1

    print("  j band (J=%d)          both   ours-only  theirs-only   our share" % J)
    for k in range(nb):
        b_, o_, t_ = hist["both"][k], hist["o"][k], hist["t"][k]
        sh = (b_ + o_) / (b_ + t_) if (b_ + t_) else float("nan")
        print("  %6d - %-8d %8d %10d %12d %11.4f"
              % (k * J // nb, (k + 1) * J // nb, b_, o_, t_, sh))
    tb, to, tt = (sum(hist[k]) for k in ("both", "o", "t"))
    print("  %-17s %8d %10d %12d %11.4f"
          % ("TOTAL mapped", tb, to, tt, (tb + to) / (tb + tt) if (tb + tt) else 0))
    if any(outside.values()):
        print("\n  OUTSIDE the stated rectangle: %s" % outside)
        print("  ^ nonzero here means the rectangle is wrong -- run `extent`.")


def parse_band(t):
    lo, hi = t.split(':')
    return int(lo), int(hi)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", required=True, metavar="QLO:QHI",
                    help="special-q window that was swept, e.g. 250000000:250004000")
    ap.add_argument("--skew", required=True, type=float,
                    help="skew from the .job file the run used")
    ap.add_argument("--side", type=int, choices=(1, 2), default=None,
                    help="field holding the special-q: 1 rational, 2 algebraic"
                         " (default: detect from where the band primes land)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extent", help="what rectangle did each file cover?")
    e.add_argument("files", nargs="+")
    c = sub.add_parser("compare", help="set difference, binned by j")
    c.add_argument("ours")
    c.add_argument("theirs")
    c.add_argument("J", type=int)
    c.add_argument("logI", type=int)
    a = ap.parse_args()
    if a.cmd == "extent":
        extent(a.files, parse_band(a.band), a.skew, a.side)
    else:
        compare(a.ours, a.theirs, a.J, a.logI, parse_band(a.band), a.skew,
                side=a.side)
