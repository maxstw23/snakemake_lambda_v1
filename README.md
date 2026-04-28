# Lambda Directed Flow (v1) Analysis

Snakemake pipeline for measuring Lambda and anti-Lambda hyperon directed flow (v1) in Au+Au collisions from the RHIC Beam Energy Scan (BES) at sqrt(s_NN) = 7.7--27 GeV.

## Requirements

- [Conda](https://docs.conda.io/) (Miniforge recommended)
- ROOT (for C++ combination scripts)

## Setup

```bash
conda env create -f environment.yaml
conda activate lambda_v1
```

To update the environment after changes to `environment.yaml`:

```bash
sh update_env.sh
```

## Usage

```bash
snakemake -n              # dry run (always do this first)
snakemake --cores all     # run the full pipeline
```

The final output is `plots/paper/report.pdf`.

To visualize the workflow DAG:

```bash
sh create_dag.sh
```

## Pipeline

```
data/*.root
  -> combine_lambda          # merge ROOT histograms (with/without efficiency corrections)
  -> fit_v1.py               # invariant mass fits, v1 extraction via iminuit
  -> plot_v1.py              # v1 vs rapidity plots, dv1/dy slope fits
  -> combine_sys.py          # systematic uncertainty combination
  -> generate_paper_plots.py # final paper figures -> report.pdf
```

## Configuration

All analysis parameters are in `config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `energies` | Collision energies (7p7GeV -- 27GeV) |
| `particles` | Particle species (Lambda, Lambdabar) |
| `pt_lo`, `pt_hi` | Transverse momentum selection (0.4--1.8 GeV/c) |
| `y_cut` | Rapidity cut (0.6) |
| `fit_order` | Polynomial order for dv1/dy slope fit |
| `sys_divisor` | Divisor for systematic uncertainty (3 = half-width uniform) |
| `yrebin` | Per-energy rapidity rebinning factors |

## Systematics

Systematic uncertainties are evaluated by repeating the analysis on independent data subsets (`sys_tag_1,2,3`) and with varied analysis cuts (`special_sys_tag_5,6`). The `combine_sys` rule aggregates them in quadrature.

## Key Dependencies

- **snakemake** -- workflow management
- **uproot** / **awkward** -- ROOT file I/O (no PyROOT)
- **iminuit** -- fitting
- **mplhep** -- HEP-style matplotlib plots
- **uncertainties** -- error propagation
