# Lambda Directed Flow (v1) Analysis

This repository measures the directed flow of $\Lambda$ and $\bar{\Lambda}$ hyperons in Au+Au collisions from the RHIC Beam Energy Scan, at $\sqrt{s_{NN}}$ = 7.7, 9.2, 11.5, 14.6, 17.3, 19.6 and 27 GeV.

The physics goal is the *splitting* between the two species — $\Delta(dv_1/dy) = dv_1/dy(\Lambda) - dv_1/dy(\bar{\Lambda})$ — and how it compares to what quark coalescence predicts from the measured proton and kaon flow. If $\Lambda$ flow is just the sum of its constituent quarks' flow, the splitting should track the proton–kaon reference; a deviation is the interesting result. Everything in the pipeline exists to get that number and its uncertainty right.

The end products are the paper figures in `plots/paper/` (collected into `report.pdf`) and the numbers behind them in `plots/paper/data_points/`.

## What is and isn't in this repository

The code **and the inputs** are here; `result/`, `plots/` and `logs/` are gitignored and get rebuilt. A fresh clone is about 1.2 GB and can run the pipeline as-is.

`data/` is tracked, but it holds *trimmed* copies of the upstream productions, not the originals. The raw files are 3.9 GB and five of them are over GitHub's 100 MB per-file limit, so they cannot be tracked without LFS. `scripts/slim_data_files.py` copies out only the objects the Snakemake rules actually open — for the main productions that is the $\Lambda$/$\bar{\Lambda}$/$\Xi$/$\bar{\Xi}$ mass and $v_1$ histograms inside the analysis $p_T$ window, which is 30% of the bytes and 100% of what any rule reads. That takes `data/` to 1.2 GB with the largest file at 48 MB. Nothing that is kept is altered; see [Trimmed inputs](#trimmed-inputs) for what this costs you.

Three things are committed that look like data and are worth knowing about:

- `scripts/pikp_merged.py` and `scripts/pikp_merged_altcuts.py` — the $\pi$/K/p reference flow, compiled into Python modules. These are **generated** files, but they are committed, so the coalescence comparison works without the raw inputs. Edit `gen_pikp_merged.py`, never the modules; anything you hand-edit gets overwritten. Regenerating them needs the `Etabins_20_Coalescence` dataset, which is *not* committed.
- `docs/` — analysis notes, the statistical audit, and paper prose.
- `data/` — the trimmed inputs described above.

## Getting started

```bash
conda env create -f environment.yaml
conda activate lambda_v1
```

Parts of the analysis are ROOT macros. The Snakefile works this out for you: if a `root` is on your `PATH` it uses that, and otherwise it falls back to the official container, which you would need to pull once:

```bash
docker pull rootproject/root:latest
```

Either way it prints which one it picked when the workflow starts (`[Snakefile] ROOT via ...`) — worth a glance if a C++ step behaves unexpectedly. To point it somewhere specific, for instance a ROOT you source from a module or a non-standard prefix:

```bash
snakemake --config root_cmd='/opt/root/bin/root'   # or set root_cmd in config.yaml
ROOT_CMD='/opt/root/bin/root' snakemake            # equivalent, via the environment
```

The container mounts the repository at `/work` and runs there, so macro arguments are relative paths in both cases and nothing else has to change. Switching between Docker and a native ROOT does **not** invalidate existing results — snakemake tracks the text of the rules, not the command they expand to — so you can move a half-finished analysis between machines without refitting.

The macros are checked against the container's ROOT (currently **6.38.00**), where all three of `combine_lambda_without_eff.cpp`, `combine_lambda_with_eff.cpp` and `calculate_lambda_eff.cpp` load cleanly. They use nothing exotic and should build under any recent ROOT 6, but no version is pinned and native builds are less well travelled — if you go that route, check one combine step against a known-good output before trusting a full run.

Then always look before you leap:

```bash
snakemake -n              # dry run: shows what would rerun
snakemake --cores all     # the whole thing
```

The dry run matters. If it lists hundreds of jobs when you expected a plot update, something upstream got invalidated — find out what before starting the run.

## Reproducing the analysis

### What you need

All inputs live under `data/`, and the Snakefile discovers them by globbing:

| Path | Count | Raw | Tracked | What it is |
|------|-------|-----|---------|------------|
| `data/result*_{energy}.root` | 7 | 685 MB | 287 MB | default dataset, one per energy |
| `data/sys_tag_{1,2,3}/result*_sys_tag_N_{energy}.root` | 21 | 1.76 GB | 811 MB | systematic reprocessings with different upstream cuts |
| `data/eff/result*_{lambda,lambdabar}_exp_{energy}.root` | 14 | 110 MB | 7.9 MB | MC samples for the efficiency correction |
| `data/v1_piKp/{energy}/{particle}/cen{N}.v2_pion.root` | 294 | 1.33 GB | 91 MB | $\pi$/K/p flow, centralities 1–9 plus merged bins |
| `data/model/{urqmd,ampt}/{energy}.root` | 7 | 1 MB | 1 MB | UrQMD and AMPT comparisons (copied whole) |

The **Tracked** column is what is in git; the **Raw** column is the upstream original. If you have the raw productions, drop them in instead — the pipeline reads both identically.

These come out of the upstream KFParticle reconstruction, which lives in a **separate repository** and is not reproducible from here; `docs/upstream_analysis.md` documents the cuts and histograms it produces so you can check that a candidate input file is the right thing.

Beware the discovery rule: for each energy the Snakefile takes the **highest-numbered** matching file. Dropping a newer `resultN` into `data/` silently changes what the analysis runs on, with no error and nothing obvious in the output. The same applies to the systematic and efficiency files.

### Trimmed inputs

`data/` holds trimmed copies. Everything the Snakefile opens is present and bit-identical to the original; what was dropped is:

| Dropped | Why it is safe |
|---------|----------------|
| $p_T$ bins outside 4–17 (0.4–1.8 GeV/c) of the mass and $v_1$ arrays | `combine_lambda_{with,without}_eff.cpp` derives `ptbin_lo`/`ptbin_hi` from `pt_lo`/`pt_hi` in `config.yaml` and never asks for the rest. This is the whole reduction — 30 bins to 14 — and the only one that matters for size |
| `h{Λ,Λ̄,Ξ,Ξ̄}_EPD_a1_pt_*` | `config.yaml` sets `flows: [v1]`, so no rule requests `a1`. Present in only 5 of the 28 productions (14.6 GeV and the 19.6 GeV set), which is too few energies to run an $a_1$ analysis from anyway |
| `hgp*`, `hgKp*` PID QA (dE/dx, 1/β, $m^2$ vs $p$) | never opened by any rule or script |
| 129 of the 147 objects per efficiency file | `calculate_lambda_eff.cpp` reads only `hMCParPtY_*` and `hKFPRecoParPtY_*` |
| ~206 of the ~232 objects per piKp file | `Finish_v1_tof_eff.C` reads 26 of them |

Three things need the raw originals: widening `pt_lo`/`pt_hi` past 0.4–1.8 GeV/c, adding `a1` to `flows`, and `scripts/check_lambda_reco.py` (which scans all 30 $p_T$ bins). Everything else runs unchanged.

To regenerate the trimmed tree from raw productions, or to verify one against the other:

```bash
python scripts/slim_data_files.py --out backup/data_slim --jobs 12
python scripts/slim_data_files.py --out backup/data_slim --jobs 12 --check
```

`--check` compares `values`, `errors`, `fBinEntries`, `fSumw2` and `fEntries` for every retained object, and demands byte equality for the files that are copied whole.

### What it costs

A from-scratch run is **590 jobs**. Roughly 200 of them are invariant-mass/$v_1$ fits, and those dominate everything else:

| Stage | Jobs | Typical time each |
|-------|------|-------------------|
| `prepare_piKp` / `fit_piKp` | 210 | seconds |
| `calculate_efficiency`, `combine_lambda*` | 98 | ~0.5–2 min |
| `fit_particle`, `fit_particle_yint`, `fit_lambda_pt` | 196 | median ~5 min, worst ~2 h |
| plotting, systematics, figures | 86 | seconds to a couple of minutes |

Budget a few tens of CPU-hours overall — overnight on a workstation with `--cores all`. (The per-fit numbers are taken from this repository's recorded run history; snakemake's stored timings are unreliable for jobs whose metadata was later rewritten, so treat them as orders of magnitude, not benchmarks.) You will also want ~2.2 GB of disk beyond the inputs: `result/` runs to about 500 MB and `plots/` to 1.7 GB.

### Reproducing in stages

You do not need everything to check something. The pipeline splits cleanly:

- **Figures only.** Once the YAMLs exist, `snakemake --cores all plots/paper/report.pdf` redraws every figure in minutes without touching a fit. This is the loop to use when working on presentation. Note it reads from both `plots/final/paper_yaml/` (the combined slopes) and `plots/sys_tag_0/paper_yaml/` (the example mass fit, the resolutions, the Xi slopes), plus the model CSVs under `result/model/` — so keep the per-tag YAMLs, not just the combined ones.
- **One energy end to end.** Ask for a single target, for instance `snakemake --cores all plots/sys_tag_0/dv1dy_coal_19p6GeV.pdf`. That pulls in only the fits for that energy — a reasonable first test that your inputs are wired up correctly.
- **Default dataset only.** The systematic tags are independent of each other; `sys_tag_0` alone gives you central values, just without uncertainties. Note that `combine_sys` needs tags 1, 2, 3, 5, 7 and 8 to be present, so the final YAMLs come last.

### Checking you got the same answer

The published numbers are in `plots/paper/data_points/`, one CSV per energy, with statistical and systematic errors split into separate columns:

```
plots/paper/data_points/dv1dy_{energy}.csv          # slopes vs centrality, all five observables
plots/paper/data_points/v1_y_netlambda_{energy}.csv # v1(y) points
```

Diff those against your own run — they are the analysis output in its most comparable form, and far more useful than comparing PDFs. Exact agreement to the last digit is not expected if you are on a different ROOT or iminuit version, since the fits are iterative; disagreement well inside the statistical errors is fine, disagreement at the level of the systematic band is not.

## How the analysis works

Raw input is one ROOT file per energy containing invariant-mass and flow histograms binned in centrality, rapidity and $p_T$.

**Combining.** A C++ ROOT macro (`combine_lambda_without_eff.cpp`, or `..._with_eff.cpp` where an efficiency file exists) applies the $p_T$ and rapidity cuts and merges histograms across centralities into `result/sys_tag_N/combined_*.root`. Efficiency corrections are available for both species at all seven energies; they are two-dimensional in $(p_T, y)$, computed by `calculate_lambda_eff.cpp` from MC truth-matched data.

**Fitting.** `fit_v1.py` does the real work. In each centrality and rapidity bin it fits the $p\pi$ invariant-mass distribution with a double Gaussian plus a second-order polynomial background (extended binned likelihood, via iminuit), then extracts the signal flow from the mass dependence of the raw flow observable:

$$v_1(m) = \frac{B(m)}{S(m)+B(m)} v_1^{\rm bkg}(m) + \frac{S(m)}{S(m)+B(m)} v_1^{\rm sig}$$

The background flow is first constrained on the sidebands, then frozen while $v_1^{\rm sig}$ is fitted over the full mass range. Output is one CSV of $v_1$ per rapidity bin per centrality. `docs/reconstruction_section.md` writes this up in paper prose.

**Slopes.** `plot_v1.py` applies the event-plane resolution correction, drops bad bins, and fits $v_1(y)$ linearly over $|y| < 0.6$ to get $dv_1/dy$ for $\Lambda$, $\bar{\Lambda}$, their difference, net-$\Lambda$ and excess-$\Lambda$. It writes both diagnostic PDFs and the YAML files that everything downstream reads.

**Systematics and figures.** `combine_sys.py` folds the variants together into `plots/final/paper_yaml/`, and `generate_paper_plots.py` turns those into the paper figures.

## The figures

| File | What it shows |
|------|---------------|
| `fig_2.pdf` | $dv_1/dy$ vs centrality for $\Lambda$, $\bar{\Lambda}$ and their difference, against the coalescence references |
| `fig_2_netlambda.pdf`, `fig_2_excesslambda.pdf` | the same for the net- and excess-$\Lambda$ constructions |
| `fig_3_horizontal.pdf`, `fig_3_vertical.pdf` | energy dependence of the splitting in three centrality panels (peripheral bin = 40–80%) |
| `v1_y.pdf` | $v_1$ vs rapidity for all energies |
| `report.pdf` | everything merged |

Variants exist alongside these: `_5080` uses 50–80% as the peripheral bin instead of 40–80%, and `_altcuts` uses a looser proton/kaon $p_T$ selection in the reference (proton $p_T > 0.4$, kaon $p_T > 0.2$, versus the default $0.4 < p_T < 1.8$ and $0.28 < p_T < 1.2$). These are consistency checks, not replacements.

One trap: the *unsuffixed* `fig_3_horizontal.pdf` and `fig_3_vertical.pdf` always come from `generate_paper_plots.py`. `fig3_5080.py` produces only the suffixed variants, even though it can draw the same layout. If you edit `fig3_5080.py` expecting the default figure to change, it won't.

## Configuration

Analysis parameters belong in `config.yaml`, not in the scripts. The ones you're most likely to care about:

| Parameter | Meaning |
|-----------|---------|
| `pt_lo`, `pt_hi` | $\Lambda$ transverse momentum window (0.4–1.8 GeV/c) |
| `y_cut` | rapidity range for the $dv_1/dy$ fit (0.6) |
| `yrebin` | rapidity rebinning; 1 means the full 20 y bins |
| `fit_order` | order of the $dv_1/dy$ fit — linear (1) at every energy, deliberately |
| `sys_divisor` | 3, i.e. treating each systematic spread as a half-width uniform distribution |
| `plotting` | per-energy y-axis ranges for the $v_1(y)$ panels |

## Systematic uncertainties

Two kinds. **Regular** variations (`sys_tag_1,2,3`) are complete reprocessings of the data with different upstream cuts — vertex position, track quality, decay-length topology — and are available for all seven energies. **Special** variations reuse the default dataset and change something inside this pipeline:

- tags **5** and **8**: fit $dv_1/dy$ over positive-$y$ only, and negative-$y$ only
- tag **7**: use a rapidity-integrated (1D in $p_T$) efficiency correction instead of the 2D default

For each variation, `combine_sys.py` takes the deviation from the default and keeps it only if it is statistically significant — larger than the quadrature difference of the two statistical errors — then sums the significant ones in quadrature and divides by `sys_divisor`.

Tags 5 and 8 are handled differently, and this is worth understanding because it's easy to get wrong: they are *not* each compared against the default. The two half-$y$ fits use disjoint rapidity ranges, so their difference is a full-width spread, and it enters as $(\Delta^2 - \sigma^2)/12$ with $\sigma^2$ the combined statistical scatter. Comparing each half against the symmetric fit separately would double-count.

Tag **6** (cubic instead of linear $dv_1/dy$ fit) still exists in the code but has been **removed from the combination**. It was producing deviations up to several times the slope itself, driven by a cubic overfitting sparse peripheral and low-energy data rather than by any real ambiguity in the fit order.

Full details, including exactly what each upstream tag varies, are in `docs/systematics.md`.

## Side studies

These are run by hand rather than through Snakemake. They read existing pipeline output and write into separate directories, so they can't clobber the main results.

- **`run_10ybin.py`** — repeats the whole $v_1(y)$ chain with 10 rapidity bins instead of 20, into `result/10ybin/` and `plots/10ybin/`, to check the binning doesn't drive the result. `--energies 27GeV` for a quick single-energy check, `--skip_fits` to replot without refitting, `--jobs N` to parallelise. `plot_v1_y_10bin.py` then makes the final figures from it.
- **`plot_v1_cen_y_final.py`** — the 3×3 raw-centrality $v_1(y)$ panels with per-point systematics, and with `--delta`, the $\Lambda - \bar{\Lambda}$ version. Note this uses only tags 1, 2, 3 and 7: the half-$y$ tags change the *fit*, not the measured points, so including them would be meaningless here.
- **`closure_test_v1.py`** — toy study checking that the $v_1$ extraction returns what you put in.
- **`test_fit_order.py`** — the diagnostic behind the decision to fit linearly everywhere.
- **`plot_eff_impact.py`, `plot_eff_vs_pt.py`, `plot_eff_y_slices.py`** — how much the efficiency correction moves the answer, and whether the efficiency itself looks sane.
- **`quark_v1_bayes.py`** — exploratory Bayesian extraction of constituent-quark $v_1$. Not used in the paper.

## Further reading

Start with **`docs/statistical_integrity_scan.md`**. It is a line-by-line audit of the statistical treatment through the whole chain, and it lists what is still open — the robust loss function used in the $v_1$ extraction, the fraction of dropped bins, the absence of an efficiency-uncertainty systematic, and the resolution being treated as uncorrelated between centralities. If you are taking this analysis over, those are the things to make a decision about.

The rest:

- `docs/systematics.md` — what each variation changes and how they combine
- `docs/upstream_analysis.md` — the KFParticle reconstruction feeding this repository
- `docs/reconstruction_section.md` — paper-ready description of the reconstruction and fits
- `docs/pikp_reference.md` — provenance of the reference data
- `docs/fig3_model_comparison_statistic.md` — proposal for replacing the two $\chi^2$/ndf values in fig 3 with a single statistic answering the actual question

## Layout

```
data/       inputs: raw, efficiency, pi/K/p reference, models (tracked, trimmed)
result/     combined ROOT files and fit CSVs, one tree per systematic tag
plots/      per-tag diagnostics; plots/final/ combined; plots/paper/ deliverables
logs/       stdout/stderr per rule, mirroring the sys_tag structure
scripts/    all the analysis code, Python and ROOT macros
docs/       analysis notes and paper prose
```

Diagnostic plots belong in the `sys_tag` directory they came from (`sys_tag_0` for the default dataset), not in `plots/paper/` — that directory is reserved for things going into the paper.

To see the workflow structure, `sh create_dag.sh` writes `dag.pdf`.

## Built on

snakemake for the workflow, uproot for reading ROOT from Python (no PyROOT anywhere), iminuit for all fitting, mplhep for plot styling, and the `uncertainties` package for error propagation.
