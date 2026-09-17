# Tabero compact v1/v3 action-state continuity comparison

## Result

Dataset v3 removes the stale cross-episode position targets found in v1. Every
v3 episode was verified element-by-element:

- `v3 state[t] == v1 state[t]`
- `v3 action[t] == v1 state[t+1]`
- each v3 episode contains exactly one fewer row than v1

This holds for all 29 episodes. The result is not inferred only from metadata.

| Metric | v1 | v3 |
|---|---:|---:|
| Frames | 6,655 | 6,626 |
| Episodes whose first action/state position gap exceeds 50 mm | 28/29 | 0/29 |
| Episodes whose first action equals the previous episode's final action | 14/29 | 0/29 |
| Same-frame action/state position p50 | 7.484 mm | 1.163 mm |
| Same-frame action/state position p95 | 64.509 mm | 14.718 mm |
| Same-frame action/state position p99 | 423.359 mm | 18.511 mm |
| Same-frame action/state position maximum | 456.515 mm | 45.501 mm |
| Same-frame position gaps above 50 mm | 527 | 0 |
| Physical action/state rotation maximum | 10.429° | 1.225° |

For the first action in every 50-step training chunk (26 training episodes),
v1 contains 482/5,962 position gaps above 50 mm and 253 above 100 mm. Dataset
v3 contains 0/5,936 above 50 mm; its action-0 p50/p95/max are
1.155/14.729/45.501 mm.

## Remaining rotation defect

The absolute axis-angle representation still changes branches near π. In v3,
114/6,626 same-frame targets have an axis-angle component difference near
2π even though the actual SO(3) rotation is small. The largest physical
one-step rotation is only 1.225°. All 29 episodes contain at least one such
branch crossing.

The current `DeltaActions` transform subtracts the three rotation-vector
components directly. It therefore turns these physically smooth transitions
into very large numerical rotation labels. Across complete 50-step training
chunks, 33,311/296,800 v3 targets have a rotation-vector component norm above
6 rad relative to the anchor state, while none has a physical rotation above
20°. This should be changed to a relative SO(3) transform, with the matching
inverse transform at inference, before training the replacement checkpoint.

## Remaining timing/rate concern

V3 preserves the compacted timing policy: 12 episodes contain 18 missing
source sample intervals but are written onto a uniform 10 Hz time axis. The
largest v3 one-step translation is 45.501 mm in episode 17 at row 18. Episode
17 is one of the compacted episodes; without the original source timestamps,
the report cannot prove whether that exact transition spans a missing 0.2 s
interval.

At 10 Hz, 2,726/6,626 v3 targets move more than 2 mm in one step. The current
deployment translation limiter permits 0.02 m/s, or only 2 mm per 100 ms.
Consequently 41.1% of recorded next-state targets are faster than the current
deployment limit. An executing robot would often lag the predicted async
trajectory unless training timing or deployment speed/trajectory handling is
made consistent.

## Interpretation

The v1 label corruption is a plausible major contributor to the checkpoint's
large live position outliers: it explicitly supervises action-0 jumps as large
as 456.5 mm. V3 removes that source. This does not prove causality until a new
checkpoint is trained and evaluated from the same initialization with v3-only
normalization.

The current checkpoint should not be paired with v3 normalization or described
as a v3 model. For a controlled comparison, recompute train-only normalization,
train a new checkpoint from the same upstream initialization and split, run the
same offline evaluation, then repeat shadow deployment with the training prompt
and start pose aligned. Fix the SO(3) rotation transform first so the new run
does not retain the remaining 2π labels.

## Reproduction

The numeric result is stored in `v1_v3_comparison.json` and produced by
`compare_v1_v3.py` in this directory. No source dataset was modified.
