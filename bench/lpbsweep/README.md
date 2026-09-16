# lpb / mfb / area sweep — RESULTS finding 98

Harness and committed data for finding 98. Run order, and which table each
arm produces:

| script | arm | q per band | writes |
|---|---|---:|---|
| `sweep.sh` | single-band, all 7 configs | 100 | `results.tsv` |
| `extra.sh` | saturation (`--ecm-curves 24 --cof-rounds 6`) + `mfb` caps | 60 | `extra.tsv` |
| `bands.sh` | three q bands, uncapped configs | 60 | `bands.tsv` |
| `bands2.sh` | three q bands, `mfb`-capped configs | 60 | `bands.tsv` (tags `*cap`) |
| `deep.sh` | the 400M band, AS276 only | 60 | `bands.tsv` |
| `a32.sh` | A=32 (`logI 17`, j-slabbed), AS276 only | 30 | `a32.tsv` |
| `rerun.sh` | A=31 comparators matched to `a32.tsv`, **and** AS276's shipped gap-2 `33/35` | 30 / 60 | `a31match.tsv`, `bands.tsv` |

`bands.tsv` needs all three of `bands.sh`, `bands2.sh` and `deep.sh`, plus
`rerun.sh` for the gap-2 `33/35` rows. Each of the four writes the header if
the file is absent and skips rows already present, so they can run in any
order and be re-run safely.

Analysis:

- `integrate.py` — the finding's main chart. Integrates `∫rel(q)/ln(q)dq` over a
  log-linear fit to the bands, scans for the cheapest window satisfying
  `qmax <= 6*qmin`, and refuses to extrapolate below the shallowest or past 2x
  the deepest measured band. Repeated (config, band) rows are averaged.
- `window.py` — the fixed 80-480M production-window table (fractions of
  target), as a cross-check on `integrate.py`'s free `qmin` scan. It imports
  that module's loader and `TARGETS` rather than keeping its own copies, which
  is what let the two drift apart by 22-32% on the same configs.

## Caveats that bite

- **Band extent is a directional bias, not noise.** ms/rel falls as a band
  lengthens (+4-5% per step measured on c183 60 q vs 100 q). Never compare rows
  from arms with different `--nq`. This is what originally forced finding 98 to
  withdraw its A=32 penalty: `a32.tsv` is 30 q against 60-q and 100-q
  comparators. `rerun.sh` then produced `a31match.tsv`, A=31 comparators at the
  same 30 q, and **the penalty is reinstated at +19.3% / +29.4% / +34.3% per
  relation** across the three configs. Only the whole-window GPU-day cost of
  A=32 stays unquantified, because no A=32 bands were integrated.
- **Host contention is invisible in the GPU counters** (finding 53). Check
  `uptime` before trusting a run; a CADO `las` job on the same box inflated one
  measurement by 64% with device `fill` up 55%. `rerun.sh` gates on
  `/proc/loadavg < 2` and prints the load each run actually saw — do the same
  for anything timed.
- **`NO WINDOW` usually means the target is wrong, not the config.** Its
  deepest AS276 band is 400M and the integrator will not extrapolate past 2x
  that, so configs needing a deeper window print `NO WINDOW`, and edge-pinned
  optima print `!`. Neither is a verdict on the config. AS276 is the worked
  example: five of six rows printed `NO WINDOW`, and when the job actually
  finished (2026-09-16) on 1.671B raw relations — 0.913x the target this
  harness carried — the shipped `33/35` turned out to have a perfectly good
  117M-692M window inside the existing guard. Check the target before
  believing the verdict. c183 and c194 are far from that boundary.
- Bulk artifacts (factor bases, `.rels`, run logs) go to `work/lpbsweep/`, which
  is gitignored. Only the TSVs here are durable.
- Relation targets are inputs, not measurements; edit `TARGETS` in
  `integrate.py` rather than trusting the committed values. They are UNIQUE
  relations and `bench` counts RAW ones, so `_raw()` scales each config by
  `DUP_PER_RUNG ** (lpb_max - anchor)`. Key any new target on its `lpb_max`,
  never on its position in a ladder — gap-2 and merged-in targets escape a
  position-keyed correction silently.

## `nfsathome.py` / `dedup.tsv` — where the relation-growth factor R comes from

`integrate.py` used to take the per-step relation growth on trust (1.75x, from
Kyle). `nfsathome.py` measures it instead, from two sources Kyle supplied on
2026-09-15. Written up as **RESULTS finding 99**.

- **`~/code/paste-scraper/data/`** — 4773 scraped NFS@Home msieve
  post-processing logs. Not in this repo and not a durable location; if it
  disappears, `nfsathome.py` exits on the missing path and the numbers in
  finding 99 are the record. The join key is the NFS@Home page column `bits`,
  which Kyle confirms is `max(lpba, lpbr)` — so a step in `bits` is a lockstep
  step in both sides, which is what makes this corpus usable at all.
- **`dedup.tsv`** — exported here from the two tabs of Kyle's unique-ratio
  Google Sheet (55 jobs, unique fraction at the bottom and top of each
  q-range), so this part at least is reproducible without the sheet.

Headline: **R = 1.62-1.74x per lockstep step, the SAME for GNFS and SNFS**,
once both arms are controlled on **Murphy-E** rather than digits. Digits is a
proxy for difficulty, not difficulty: it carries no information for SNFS
(coefficient 1.0000, t=-0.1) and is 0.832 collinear with `lpb` for GNFS, so
controlling on it splits the arms into GNFS 1.52 / SNFS 1.79 — an artifact
that earlier revisions of finding 99 reported as a real and unexplained gap.
Kyle's 1.75-1.80 rule of thumb sits just above the measured range. The
ideal-count model's 1.94x is an upper bound and should not be used to size a
job.

Consequence: c183's level verdict flips at R = 1.632 and C194's at 1.612, both
at the bottom edge of the range, so the **lower rung wins across almost all of
it** and finding 98's level ranking stands — by under 1% at the bottom and
7-9% at the top. Balance and the `mfb`-96 cap never depended on R at all.

```
python3 nfsathome.py     # the measurement
python3 integrate.py     # the projection, under both R values
```

`filterdown.py` samples a GGNFS relation corpus and reports how many relations
survive filtering down to each lower `(lpbr, lpba)` — the sampling half of the
filter-down experiment described at the end of finding 99. Filtering DOWN is
exact; simulating upward is not.

`--mfb MFBR,MFBA` also caps the per-side **cofactor**, without which the result
is a superset of a true lower run. Two things have to be right, and both were
wrong at first:

- **`--lim RLIM,ALIM`.** A GGNFS relation line carries the full factorisation
  of each norm, factor-base primes included, while `mfb` bounds only what is
  left after the factor base is divided out. The cofactor is the product of the
  primes above `lim`, and the line does not say where the factor base ended.
- **`--sqside r|a`.** The special-q divides the norm on its own side and is
  **not** part of what `mfb` bounds, but it *is* listed among that side's
  primes. Wherever q exceeds that side's `lim` it otherwise gets multiplied
  into the cofactor, inflating it ~28 bits and rejecting most deep-q relations
  as false `mfb` violations. GGNFS job files record the side as `lss`:
  **0 = algebraic, 1 = rational**; this server's ledger stores it as `meta.side`.

**Always sanity-check by filtering a corpus to its own shipped config: it must
keep ~100%.** That check is what caught both bugs.

```
# AS276: algebraic special-q (lss 0), rlim/alim 181.6M/268.4M
python3 filterdown.py --n 12 --mfb 64,101 --lim 181600000,268400000 --sqside a
# snfs301: rational special-q, rlim = alim = 225M
python3 filterdown.py --n 16 --lpb-range 32,33,33,34 \
        --mfb 63,95 --lim 225000000,225000000 --sqside r \
        ~/code/ggnfs-distributed/snfs301/archive
```

Measured lockstep survival, each config at its own `mfb`: **AS276 50.8%**
(a 1.97x yield ratio) and **snfs301 47.9%** (2.09x) — both within ~5% of the
flat 2.0x yield law. The `mfba` 101→96 cap on AS276 costs 16.2%, against
finding 98's independently timed −17.2% in `rel/q`.

Note that `snfs301/archive` mixes GGNFS work units (`wu-*`, ~3.7k relations)
with cuda-sieve blocks (`blk-*`, 134-206k, ~45x bigger), and that the first
three cuda-sieve blocks were run at the top of the q range before the work
assignment was fixed. `filterdown.py` samples whole files evenly, which is
fine, but anything finer should account for both.

## The AS276 work-unit ledger — validating the decay model

`~/code/ggnfs-distributed/AS276/incoming/2026-08-26T09-26-38Z/job.db` is the
complete ledger for Kyle's distributed GGNFS run of AS276: **346,859 verified
work units, q = 80-426.9M, 1,671,198,113 relations**, each a 1000-wide q block
with an exact relation count and its `sieve_seconds`. (The smaller
`AS276/job.db` next to the corpus holds only the first 7,595 work units,
q = 80-87.6M — use the `incoming` copy.) Counts match a line count of the
matching `.zst` exactly.

    sqlite3 job.db "SELECT (w.q_start/10000000)*10, COUNT(*), SUM(s.num_relations)
      FROM workunits w JOIN submissions s ON s.workunit_id=w.id
      WHERE s.verify_status='passed' AND w.state='verified' GROUP BY 1 ORDER BY 1;"

This is what finding 99 uses to check `integrate.py`'s yield extrapolation.
Two traps in reading it:

- **Do not fit one power law over the whole range.** The curve is flat to
  ~250M and steepens after; a single fit gives q^-0.099 over 80-427M but
  q^-0.306 over 300-427M, and only the latter describes the region the
  harness extrapolates into.
- **Do not fit `num_relations` within a `client_id` either.** Clients cover
  different q sub-ranges, so within-client slopes (median q^-0.233) mix the
  flat and steep regions in whatever proportion that client happened to get.
  Yield is deterministic in q, so the band census is exact and is the right
  view. `sieve_seconds` is the opposite case: it *is* hardware-dependent, so
  it needs the within-client treatment and is not usable in aggregate.
