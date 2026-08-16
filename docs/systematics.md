# Systematic Uncertainties

## Regular Systematics (different datasets, `sys_tag_1–3`)

These use different subsets of the same collision data with varied upstream selection cuts applied in `readPicoDst.C` / the KFParticle reconstruction before this pipeline.

| Tag | Varied parameter | Default | Variation |
|-----|-----------------|---------|-----------|
| `sys_tag_1` | Primary vertex z-position cut | \|vz\| < 145 cm | \|vz\| < 70 cm |
| `sys_tag_2` | nHitsFit for primary tracks used in Λ reco | > 15 | > 20 |
| `sys_tag_3` | KFP decay length cut (`SetLCut`) | > 1.0 cm | > 3.0 cm |
|             | KFP max daughter distance (`SetMaxDistanceBetweenParticlesCut`) | < 1.0 cm | < 0.8 cm |

All three regular tags are available for all seven energies.

## Special Systematics (same dataset, varied analysis choices, `special_sys_tag_5,7,8`)

These reuse the default dataset (`sys_tag_0`) and vary choices made within this pipeline. The variation is selected by `sys_tag` inside `plot_v1.py` (lines 52–58).

| Tag | Varied parameter | Default | Variation |
|-----|-----------------|---------|-----------|
| `special_sys_tag_5` | Rapidity range for dv1/dy slope fit | \|y\| < 0.6 (full symmetric) | Positive y only (`range = 'half'`) |
| `special_sys_tag_8` | Rapidity range for dv1/dy slope fit | \|y\| < 0.6 (full symmetric) | Negative y only (`range = 'half_neg'`) |
| `special_sys_tag_7` | Efficiency correction binning | Per-y-bin (2D in pT×y) | y-integrated (1D in pT only) |

### Retired: `special_sys_tag_6` (cubic dv1/dy fit order)

Tag 6 is still implemented in `plot_v1.py` but is **excluded from the combination** (`Snakefile:504–508`). Its deviations (Δ up to ~0.09, 2–9× the slope itself) were driven by cubic overfitting of thin peripheral / low-energy data rather than by a genuine fit-order ambiguity. See `docs/statistical_integrity_scan.md`.

## Combination

All sys tags feed into `combine_sys.py` (rule `combine_sys`):

- **Regular sys** (`sys_tag_1,2,3`) and **special tag 7**: quadrature sum of significant deviations from `sys_tag_0`, divided by `sys_divisor = 3` (half-width uniform distribution assumption)
- Significance criterion: deviation exceeds the quadrature difference of the two stat errors, i.e. only `Δ² − Δ_err²` is accumulated when `|Δ| > Δ_err`
- **Special tags 5 and 8 are paired**, not compared to the default individually. The two half-y fits use disjoint rapidity ranges, so their spread `Δ = slope_pos − slope_neg` is a full-width quantity with statistical scatter `σ = sqrt(err_pos² + err_neg²)`; it contributes `max(Δ² − σ², 0) / 12` (full-width uniform → `/√12`). Implemented in `combine_sys.yrange_paired_unc`.

```
sys² = ( Σ_{significant tags 1,2,3,7} [Δ² − Δ_err²] ) / sys_divisor
       + [ (slope_pos − slope_neg)² − σ² ]_+ / 12
```

- Applied to the per-centrality dv1/dy for Lambda, Lambdabar, Delta-Lambda, net-Lambda and excess-Lambda
- Output: `plots/final/paper_yaml/dv1dy_coal_{energy}.yaml`

The per-point v1(y) systematic used by `plot_v1_cen_y_final.py` follows the same significance-gated rule but includes **only** tags 1, 2, 3 and 7 — tags 5/8 (and 6) change the *fit*, not the v1(y) values themselves.

## Checks Not Currently Varied

| Parameter | Current value | Notes |
|-----------|--------------|-------|
| pT range | 0.4–1.8 GeV/c | Variation requires rerunning `combine_lambda` (expensive) |
| Invariant mass signal window | ±3σ (σ ≈ 0.0023 GeV/c²) | Hardcoded in `fit_v1.py`; variation requires refit |
| Invariant mass background polynomial order | 2nd order | Hardcoded in `fit_v1.py`; variation requires refit |
| Rapidity fit window width | \|y\| < 0.6 | Narrowing (e.g. \|y\| < 0.5) is cheap — fit only, no refit — but tag 8 is now taken by the negative-y half fit |
| Rapidity binning | 20 y bins (`yrebin: 1`) | Explored separately via `scripts/run_10ybin.py` (10 bins), not folded into the systematic |
| Efficiency-correction uncertainty | — | Not propagated; open item in `docs/statistical_integrity_scan.md` §3 |

## Reference

- Upstream reconstruction cuts: `/mnt/d/Research/kfparticle_v1_19GeV/readPicoDst.C`
- Systematic combination logic: `scripts/combine_sys.py`
- Fit-range / fit-order variations: `scripts/plot_v1.py` (lines 52–58)
- Per-point v1(y) systematic: `scripts/plot_v1_cen_y_final.py`
