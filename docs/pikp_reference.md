# pi/K/p Reference Data

## Source

The piKp v1 slope data comes from:

```
Etabins_20_Coalescence/Etabins_20_Coalescence_ymp6top6_withMCeffandTOFcorrection/txtfiles/
  {7GeV,9GeV,11GeV,14GeV,17GeV,19GeV,27GeV}_{pions,kaons,protons}_datapoints.txt
```

This variant applies MC reconstruction efficiency + TOF efficiency corrections and
includes the vz systematic (cf. the earlier `Etabins_20_Coalescence_ymp6top6/` set,
and the `_withoutvzsys` variant which drops the vz systematic).

Energy labels map to pipeline keys: `7GeV→7.7GeV`, `9GeV→9.2GeV`, `11GeV→11.5GeV`, `14GeV→14.6GeV`, `17GeV→17.3GeV`, `19GeV→19.6GeV`, `27GeV→27GeV`.

## Generated File

`scripts/pikp_merged.py` is auto-generated from the txtfiles above. **Do not edit by hand.**

## Updating

When the txtfiles are updated, regenerate `pikp_merged.py`:

```bash
conda run -n lambda_v1 python scripts/gen_pikp_merged.py
```

## Data Contents

Each txtfile contains C-style double arrays for both `linear` and `cubic` fits:

- Per-centrality: `v1_vCent_Selp_{fit}` (positive), `v1_vCent_Seln_{fit}` (negative), and **two** difference variants, each with `_err` and `_systematics`:
  - `ddeltav1dy_vCent_{fit}` — slope of the (v1_pos − v1_neg) difference curve, i.e. **take the difference between v1(y) first, then fit a straight line**.
  - `delta_dv1dy_vCent_{fit}` — difference of the two fitted slopes, i.e. **fit each species first, then subtract**.
- Combined centralities (`0-10`, `10-40`, `40-80`, `50-80`): `v1slopes_{fit}_combinedcent_{pos,neg,deltav1}` with `_err` and `_systematics`. The combined-cent `deltav1` is computed diff-then-fit (it does not equal `pos − neg`).

### Difference method (`delta_*` keys)

`gen_pikp_merged.py` maps the per-centrality `delta_{fit}` keys to **`ddeltav1dy_vCent_{fit}`** (diff-then-fit). This matches the Lambda pipeline, which subtracts Lambda − Lambdabar *before* fitting dv1/dy. The fit-then-diff variant (`delta_dv1dy_vCent_{fit}`) is intentionally **not** used. Earlier txtfiles exposed only a single `deltav1_vCent_{fit}` array, which corresponded to the fit-then-diff (`delta_dv1dy`) method.
