# Systematic Uncertainties

## Regular Systematics (different datasets, `sys_tag_1–3`)

These use different subsets of the same collision data with varied upstream selection cuts applied in `readPicoDst.C` / the KFParticle reconstruction before this pipeline.

| Tag | Varied parameter | Default | Variation |
|-----|-----------------|---------|-----------|
| `sys_tag_1` | Primary vertex z-position cut | \|vz\| < 145 cm | \|vz\| < 70 cm |
| `sys_tag_2` | nHitsFit for primary tracks used in Λ reco | > 15 | > 20 |
| `sys_tag_3` | KFP decay length cut (`SetLCut`) | > 1.0 cm | > 3.0 cm |
|             | KFP max daughter distance (`SetMaxDistanceBetweenParticlesCut`) | < 1.0 cm | < 0.8 cm |

Note: `sys_tag_3` data is currently only available for 11.5 GeV.

## Special Systematics (same dataset, varied analysis choices, `special_sys_tag_5–7`)

These reuse the default dataset (`sys_tag_0`) and vary choices made within this pipeline.

| Tag | Varied parameter | Default | Variation |
|-----|-----------------|---------|-----------|
| `special_sys_tag_5` | Rapidity range for dv1/dy slope fit | \|y\| < 0.6 (full symmetric) | Positive y only (`range = 'half'`) |
| `special_sys_tag_6` | Polynomial order for dv1/dy slope fit | Linear (order 1) | Cubic (order 3) |
| `special_sys_tag_7` | Efficiency correction binning | Per-y-bin (2D in pT×y) | y-integrated (1D in pT only) |

## Combination

All sys tags feed into `combine_sys.py` (rule `combine_sys`):

- **Regular sys** (`sys_tag_1,2,3`): quadrature sum of significant deviations from `sys_tag_0`
- **Special sys** (`special_sys_tag_5,6,7`): same treatment
- Significance criterion: deviation exceeds the quadrature difference of the two stat errors
- Final systematic = `sqrt(sum_of_significant_deltas^2 / sys_divisor)`, where `sys_divisor = 3` (half-width uniform distribution assumption)
- Output: `plots/final/paper_yaml/dv1dy_coal_{energy}.yaml`

## Checks Not Currently Varied

| Parameter | Current value | Notes |
|-----------|--------------|-------|
| pT range | 0.4–1.8 GeV/c | Variation requires rerunning `combine_lambda` (expensive) |
| Invariant mass signal window | ±3σ (σ ≈ 0.0023 GeV/c²) | Hardcoded in `fit_v1.py`; variation requires refit |
| Invariant mass background polynomial order | 2nd order | Hardcoded in `fit_v1.py`; variation requires refit |
| Rapidity fit window width | \|y\| < 0.6 | Could add `special_sys_tag_8` with \|y\| < 0.5 cheaply (fit only) |

## Reference

- Upstream reconstruction cuts: `/mnt/d/Research/kfparticle_v1_19GeV/readPicoDst.C`
- Systematic combination logic: `scripts/combine_sys.py`
- Fit-range / fit-order variations: `scripts/plot_v1.py` (lines 52–56)
