# Architecture Reference

## Pipeline Flow

Driven by `Snakefile` with parameters in `config.yaml`:

```
data/*.root (raw ROOT files)
  → combine_lambda / combine_lambda_with_eff  (C++ ROOT scripts)
      → result/sys_tag_N/combined_{particle}_{flow}_{energy}.root
  → fit_particle (fit_v1.py)
      → result/sys_tag_N/fit_{particle}_{flow}_{energy}.csv
  → plot_v1 (plot_v1.py)
      → result/sys_tag_N/data_{energy}.txt
      → plots/sys_tag_N/paper_yaml/*.yaml
  → combine_sys (combine_sys.py)
      → plots/final/paper_yaml/*.yaml
  → generate_paper_plots (generate_paper_plots.py)
      → plots/paper/report.pdf
```

## Systematic Uncertainty Structure

- `sys_tag_0`: default dataset
- `sys_tag_1,2,3`: regular systematics (different upstream cuts, all seven energies)
- `special_sys_tag_5,8`: positive-/negative-y half-range dv1/dy fits (same dataset), combined as one **paired** systematic with divisor 12 (full-width uniform)
- `special_sys_tag_7`: y-integrated (1D in pT) efficiency correction
- `special_sys_tag_6` (cubic fit order) is implemented but **excluded** from the combination — cubic overfitting, not a real fit-order effect
- `combine_sys` rule aggregates all into `plots/final/paper_yaml/`
- Non-paired tags use `sys_divisor` from `config.yaml` (currently 3, i.e. half-width uniform distribution assumption)
- Full detail: `docs/systematics.md`

## Key Python Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `fit_v1.py` | Core fitting: reads combined ROOT histograms, fits invariant mass spectra (signal + polynomial background), extracts v1 via profile histograms. Uses iminuit. |
| `fit_v1_pt.py` | Same as above but bins in pT instead of rapidity. |
| `plot_v1.py` | Reads fit CSVs, computes dv1/dy slopes, generates comparison plots with piKp reference. |
| `combine_sys.py` | Combines systematic uncertainties from multiple `sys_tag` variants using quadrature sum of significant deviations. |
| `generate_paper_plots.py` | Final paper figure assembly from YAML data files. |
| `simple_profile.py` | `SimpleProfile` class: weighted profile histogram data; supports rebinning and addition. |
| `measurement.py` | `Measurement` class: weighted average of `uncertainties` variables. |
| `data_point.py` | `DataPoint` class: value with separate stat/sys errors; supports arithmetic. |
| `param_storage.py` | `ParamStorage`: fit parameters with optional freezing between fits. |
| `find_bin_center.py` | `BinCenterFinder`: weighted bin centers for non-uniform distributions. |
| `fig3_5080.py` | fig 3 variants (50–80% and/or alternative pT cuts); never writes the unsuffixed `fig_3_*`. |
| `gen_fig2.py` | fig 2 with a swappable piKp reference module (`--pikp_module`). |
| `gen_pikp_merged.py` | Generates `pikp_merged*.py` from the piKp txt datapoint files. |
| `run_10ybin.py` | Standalone driver: re-runs the v1(y) chain at `yrebin=2` into `result/10ybin/`, `plots/10ybin/`. Not in the Snakefile. |
| `plot_v1_y_10bin.py` | Final 10-y-bin v1(y) figures/CSVs from the `run_10ybin.py` output. |
| `plot_v1_cen_y_final.py` | 3×3 raw-centrality v1(y) (or Δv1(y) with `--delta`) with per-point systematics from tags 1,2,3,7. |
| `closure_test_v1.py` | Toy closure test of the v1 extraction fit. |
| `plot_eff_*.py` | Efficiency QA and impact-on-result diagnostics. |
| `quark_v1_bayes.py` | Exploratory Bayesian constituent-quark v1 extraction. |

## C++ ROOT Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `combine_lambda_without_eff.cpp` | Reads raw ROOT files, applies pT and rapidity cuts, combines Lambda/Lambdabar histograms across centralities. |
| `combine_lambda_with_eff.cpp` | Same but applies efficiency corrections from a separate eff ROOT file. |
| `calculate_lambda_eff.cpp` | Computes reconstruction efficiency from MC truth-matched data. |
| `Finish_v1_tof_eff.C` | Processes piKp v1 data with TOF efficiency corrections. |
| `FitSlope.C` | Fits the v1 vs. rapidity slope (dv1/dy) for piKp particles. |

## Data Layout

```
data/
  result*_{energy}.root                        # default dataset per energy
  sys_tag_{1,2,3}/result*.root                 # systematic variation datasets
  eff/result*_{lambda,lambdabar}_exp_{energy}.root  # efficiency files (all 7 energies, both species)
  v1_piKp/{energy}/{particle}/                 # piKp reference data
  model/{urqmd,ampt}/{energy}.root             # model comparisons
```

Efficiency corrections are applied for Lambda and Lambdabar at every energy for which an eff file is present in `data/eff/` — currently all seven, both species. The correction is 2D in (pT, y) by default; the y-integrated 1D variant is `special_sys_tag_7`.

## Config Parameters (`config.yaml`)

- `energies`: collision energies (`7p7GeV` through `27GeV`)
- `particles`: `[Lambda, Lambdabar]`
- `flows`: `[v1]`
- `sys_divisor`: divisor for systematic uncertainty combination (3 = half-width uniform)
- `pt_lo/pt_hi`: pT range for Lambda selection (0.4–1.8 GeV/c)
- `y_cut`: rapidity cut (0.6)
- `yrebin`: per-energy, per-particle rapidity rebinning factor
- `fit_order`: polynomial order for dv1/dy slope fit (always 1 = linear)

## YAML Output Format

Intermediate results in `plots/sys_tag_N/paper_yaml/` contain centrality-binned arrays for: x positions, v1 values, errors, dv1/dy slopes, and stat/sys error breakdown.
