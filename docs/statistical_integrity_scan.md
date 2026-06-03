# Statistical Integrity Scan — Λ directed-flow pipeline

Scope: full chain traced — raw histograms → efficiency correction (C++) →
invariant-mass + v₁ extraction fit (`fit_v1.py`) → slope fit + resolution
correction (`plot_v1.py`) → systematic combination (`combine_sys.py`).

Findings are ordered by how much they could distort central values or quoted
uncertainties. Each item cites `file:line`. No code was modified in producing
this scan.

Status legend: `[ ]` open / not yet discussed · `[x]` reviewed & resolved ·
`[~]` reviewed, action pending.

## Post-run validation (2026-06-02, full pipeline re-run, exit 0, 0 errors)
All 12 items resolved. Code fixes verified in regenerated outputs:
- **§1** real `fit_v2` now returns hybrid χ²-calibrated error vs raw soft_l1:
  cen7y2 0.00688 vs 0.00887 (×0.78), cen1y19 0.082 vs 0.139 (×0.59) → v₁ errors
  ~25–40% smaller and calibrated, as the closure test predicted.
- **§3** final ΔΛ dv₁/dy now carries a substantial systematic component
  (40–80%: stat 0.0019, sys 0.0023) — efficiency (tag-7) folded in.
- **§5** sys_tag_6 dv₁/dy now genuinely cubic, differs from linear sys_tag_0
  (e.g. ΔΛ 0–10% +0.01923 vs +0.01588) — revert no longer collapses it.
- **§12** v₁(pT) combine regenerated with eff-weighted y-folding (ROOT-verified
  `TProfile::Add` semantics; ~3–4% central-value correction built in).

### Follow-up audit → tag 6 (cubic) REMOVED from the systematic budget
Auditing per-tag contributions to the final dv₁/dy systematic (`/tmp/parse_sys.py`)
showed **tag 6 (cubic) dominated ~10× over every other source** (Σσ² contrib 0.0066 vs
≤0.0005), with Δ up to **0.0897** — concentrated at low-energy peripheral bins
(7.7/9.2 GeV, 40–80%/50–80%). These are cubic *overfitting* of ~12 noisy points (Δ =
2–9× the slope itself; a Λ̄ outlier at v₁=−0.103 drives the y³ term), passing the Barlow
gate with a deceptively small fit error. Decision (user): **drop tag 6 from
`combine_sys`** (`Snakefile` special_sys `[5,6,7]`→`[5,7]`). Effect: 7.7 GeV ΔΛ
50–80% sys 0.0497→0.0085, 0–10% 0.0137→0.0018; tag-6-independent bins unchanged
(19.6 GeV 0–10% 0.0007→0.0007). The §5 genuine-cubic change still stands in the
tag-6 *diagnostic* outputs but no longer feeds the final budget. tag 7 (efficiency)
contribution is appropriately small (Σσ² 0.00005).

---

## 🔴 High concern

### [x] 1. v₁ extracted with a robust loss (`soft_l1`), Hesse error reported as statistical — RESOLVED (hybrid applied)
`fit_v1.py:501-502` (identical `fit_v1_pt.py:505`):
```python
c = cost.LeastSquares(self.x_vals_v2, self.v2_vals, self.v2_errs, func)
c.loss = 'soft_l1'
```
This is the fit that produces v₁ (`m.values['v2']`, `m.errors['v2']`, lines 618-619).

- **Bias toward zero.** `soft_l1` down-weights points deviating from the model.
  In the invariant-mass-binned v₁ fit, robustly down-weighting the signal-region
  points pulls extracted v₁ toward the background line — wrong for a near-zero,
  sign-sensitive observable like the Λ–Λ̄ splitting.
- **Quoted error is not a coverage interval.** With a non-quadratic loss,
  `m.hesse()` gives the curvature of the *modified* cost; iminuit does not rescale
  it to a 1σ statistical error. `m.minos()` is called (`:542`) then discarded; only
  the symmetric Hesse error is returned.

Touches every v₁ point and every dv₁/dy slope downstream.

**Resolution (closure + parametric-bootstrap test, `scripts/closure_test_v1.py`, 600
toys, 7.7 GeV Λ̄, cells from 163 to 24k counts):**

| loss | bias | reported_err / true_err | pull width | converges |
|---|---|---|---|---|
| `soft_l1` (current) | ≤0.04σ | **1.27–1.32** | **0.76–0.80** | robust |
| plain χ² (`linear`) | ≤0.04σ | 1.00–1.02 | 0.98–1.00 | needs good seeds |
| **hybrid** (soft_l1 min → χ² Hesse) | ≤0.07σ | 1.04–1.07 | 0.94–0.96 | robust (0 fails) |

Findings:
- **No bias.** `soft_l1` does *not* miss the signal — the original "bias toward zero"
  concern is disproven. The v₁ point estimate is clean.
- **Error is ~30% over-inflated** by `soft_l1` (rep/boot ≈ 1.3, pull ≈ 0.78), stable
  across 150× in statistics. This is the *conservative* direction, but it discards
  ~30% of the significance of the Λ–Λ̄ splitting.
- **Plain χ² is unbiased and perfectly calibrated** (pull = 1.00) — also validates the
  test framework.
- The toy seeds near truth, so it does not exercise `soft_l1`'s real benefit: robust
  convergence from bad random starts in production.

**Proposed fix (hybrid):** keep `soft_l1` to *find* the minimum (convergence
preserved), then re-evaluate the error with a plain-χ² Hesse at that point. ~3 lines
after the final migrad in `fit_v1.py:532-540` and the mirror in `fit_v1_pt.py`:
`c.loss = 'linear'; m.migrad(); m.hesse()`. Gives unbiased v₁ + calibrated error
(pull 0.95).

**APPLIED** 2026-06-01 to `fit_v1.py` (after `:539`) and `fit_v1_pt.py` (after `:542`).
Pipeline not yet re-run — the fit stage must be re-executed to regenerate the
`fit_*_v1_*.csv` files (user to run; project rule forbids auto-running `fit_v1*`).
Expect v₁ central values ~unchanged and all v₁/dv₁dy errors to shrink by ~25-30%.

### [x] 2. "Refit with random seeds until valid, else drop the bin" — AUDITED, no action needed
`fit_v1.py:930-991`. Failed bins retried up to `max_refit` (500) with randomized
starts (`np.random.uniform(...)`); after that the bin is set to `None` (`:939`) and
silently excluded.

- **Survivorship/selection bias.** Drawing until `m.valid and m.accurate` accepts
  whichever local minimum Minuit blesses; degenerate fits are not fairly sampled.
- **Missing-not-at-random points.** Dropped bins leave the dv₁/dy fit with no flag.
  If low-stat / near-zero-v₁ bins fail more often, the slope is biased.

**Recommendation:** record dropped-bin fraction per (energy, centrality, y); confirm
it is small and uncorrelated with v₁ magnitude.

**Audit result (2026-06-01):** scanned all 70 `fit_*_v1_*.csv` across `sys_tag_0` +
every regular/special systematic dir = **12,600 bins, 0 NaN/dropped (0.00%)**. The
`max_refit=500` loop fits every bin in the default and all systematics, so Hazard B
(silent drops) never occurs in practice. Hazard A (first-valid survivorship) remains
theoretically possible but §1's closure test — which accepts first-valid fits via
random restart — measured v₁ bias ≤0.04σ, so it is not biasing central values.
**No code change.** (`best-of-N` would harden Hazard A if ever desired, but evidence
says it is unnecessary.)

### [x] 3. Efficiency-correction uncertainty never propagated — RESOLVED (wired in special_sys_tag_7)
`calculate_lambda_eff.cpp` fits a 5-parameter ε(pT) per centrality/y-bin;
`combine_lambda_with_eff.cpp:93-95` applies `1/ε(pT)` as a weight. Fit-parameter
covariance and MC stat error of ε are dropped. The "post-merging error correction"
(`combine_lambda_with_eff.cpp:108-125`) resets the bin entry count to the
*uncorrected* count before recomputing the error:
```cpp
float W = entries_new;            // uncorrected entries
float E = (error*error + content*content) * entries_new;
```
So the error bar is forced back to raw-statistics scale — a defensible approximation
(treats ε as exact) but **no efficiency uncertainty enters the final error**, and no
systematic stands in for it (see §6).

Also `combine_lambda_with_eff.cpp:94`: `if (eff_val <= 0) eff_val = 1e-4;` → weight
10⁴. If fitted ε dips near zero in-range, one bin dominates the profile. Safer to
drop the bin than inject a 10⁴ weight.

**Analysis (2026-06-01):**
- **Correct category = systematic**, not statistical: ε is one fixed function applied
  to the whole dataset, so its uncertainty is a coherent, correlated shift that does
  not average down and does not cancel in Λ–Λ̄ (different ε files per species).
- **Key subtlety:** v₁ = ⟨cos(φ−Ψ)⟩ is a mean → a *flat* (pT-independent) ε error has
  ZERO effect. ε enters v₁(y) only through the pT-integration
  (`combine_lambda_with_eff.cpp:100`, profile added with `eff_weight` across pT). So
  the systematic must probe the **pT/y shape** of ε, not its normalization.
- **Sizes (from `result/eff/*.root`, binomial-error hists):** ε is steeply
  pT-dependent (≈0.02→0.38 across 0.4–1.8, ~17×), so the reshaping is a large effect;
  but MC-stat per-bin error is only ~1.2–1.8%, so the MC-stat-driven systematic is
  modest. Dominant uncertainty is shape/model.
- **`1e-4` clamp never fires** (ε>0.02 everywhere in-window) — dormant; add a one-line
  guard for safety but not urgent.

**Recommended implementation:** assess on the FINAL dv₁/dy via the existing
`special_sys` machinery (run a shape-perturbed `..._eff_corrected.root` through
`fit → plot_v1 → combine_sys`); this propagates coherently through the slope and the
Λ–Λ̄ difference for free, and `combine_sys`'s `delta_err≈0` handling for same-dataset
variations is the right behaviour. Cheapest shape handles, both already in code:
(a) **y-binned vs y-integrated ε** (`use_y_binned_eff` flag, zero new fitting);
(b) pT-tilt of ε within the TF1 fit covariance.

**RESOLVED (2026-06-01).** The infrastructure for handle (a) already existed but was
*orphaned*: `combine_lambda_with_eff_yint` → `fit_particle_yint` → `plot_v1_yint`
(Snakefile:274-354) generate `special_sys_tag_7` = y-integrated-ε variant, but
`rule combine_sys` only consumed special tags `[5,6]`, so it was computed and never
used. **Fix = one line:** `sys_tag=[5,6]` → `[5,6,7]` in `combine_sys`'s `special_sys`
input (Snakefile:495). Dry-run confirms the DAG resolves.

Validation/quantification (combine_sys run to a temp file on existing YAMLs): tag-7
ingests correctly; its dv₁/dy contribution ranges ~0.0000–0.0137 and is significant in
most centrality/species bins — a real, non-negligible systematic. NOTE: the
pre-existing `special_sys_tag_7` YAMLs are **stale** (10 ybins vs current 20) and make
`combine_sys` crash in the per-bin v1_y section until regenerated; the re-run
(`plot_v1_yint`) regenerates them with matching binning. Minor follow-up: `combine_sys`
hard-crashes on any ybin-length mismatch between tags — brittle, could guard.

Pipeline re-run required to populate `plots/final/` with the new term (user to run).

---

## 🟠 Moderate concern

### [x] 4. EP resolution correlated factor propagated as per-point independent — QUANTIFIED, negligible, no action
`plot_v1.py:213` `v1_final = v1_unumpy / resolution[cen-1]`. `R` is one number per
centrality (`:62-66`); its uncertainty is 100% correlated across y-bins. It is folded
into each point via `uncertainties`, then the slope is fit with `cost.LeastSquares`
assuming independent errors (`fit_iminuit`, `:1444-1453`). A fully-correlated
multiplicative scale should propagate as σ(slope)/slope = σ_R/R *after* the fit, not
blended per-point. This generally **underestimates** the resolution contribution to
the dv₁/dy error. (Δ = (raw_Λ − raw_Λ̄)/R at `:375` divides by the same R — consistent,
same caveat.)

**Quantified (2026-06-01, `/tmp/quant_res.py` against data ROOT + fit CSVs):**
σ_R/R = 0.02–0.31% per centrality (typically ~0.03%), because R is averaged over the
full event sample. The correlated term `a·σ_R/R` ≈ 3e-4 × slope is swamped by the
statistical fit error (10–50× larger). Current vs stat-only vs fully-correct slope
error agree to **4 sig figs (ratio = 1.000)** at all energies/centralities. The concern
is real in principle but numerically null. **No code change** (user decision): fixing
it would shift error bars <0.1% while adding complexity to working code, and the
merged-centrality case (`plot_v1.py:947+`) would need block-correlated handling of
multiple R_cen. Revisit only if the resolution uncertainty ever becomes
systematics-dominated (currently it is statistics-dominated and tiny).

### [x] 5. Data-driven model selection on the cubic term — RESOLVED (revert commented out)
`plot_v1.py:253-261` (and Λ̄/Δ copies 332-340, ~418-422): a 3rd-order fit `a·y + b·y³`
is run, and if `abs(popt[1]) < perr[1]` (cubic < 1σ) the code reverts to the linear
slope and reports it. Choosing the model from the same data that sets the value, then
quoting the selected model's error, ignores model-selection uncertainty; a 1σ
threshold is loose. Nominal pipeline uses `fit_order: 1` (`config.yaml:20`), so this is
dormant by default but **live in sys_tag 6** (`plot_v1.py:52-54`), where it affects the
quoted systematic spread.

**Confirmed:** sys_tag_6 = cubic fit-order systematic; sys_tag_5 = positive-half-y
range systematic (both reuse default v1(y) points, only the slope fit changes).

**Quantified (2026-06-01, `/tmp/quant_cubic.py`):** 47% of per-centrality bins (29/62)
revert; but they are the *small*-cubic-term bins, so RMS|cubic−linear| barely changes
(0.0150 with revert vs 0.0154 genuine cubic). The principled issue: the revert is a
redundant, *cruder* significance gate (1σ on the cubic term b) that pre-empts
`combine_sys`'s proper Barlow test (on the slope deviation, with the cubic fit's larger
`delta_err`). Wild cubic fits on low-stat bins (e.g. 27 GeV cen1: −0.023→+0.062 sign
flip) are backstopped by `combine_sys` because their inflated `delta_err` fails the
Barlow gate.

**RESOLVED (2026-06-01).** Per user request, the revert `if abs(popt[1]) < perr[1]:`
blocks were *commented out* (not deleted) at all 6 sites in `plot_v1.py`: 3
per-centrality (Λ/Λ̄/Δ) + 3 merged-centrality (Λ/Λ̄/Δ). sys_tag_6 now reports the genuine
cubic slope; significance is delegated to `combine_sys`. File parses OK. Pipeline re-run
required to propagate (user to run). Note: also governs any future nominal `fit_order:3`.

### [x] 6. Barlow significance test for the "special" systematics — NOT AN ISSUE (Barlow generalizes beyond subsets)
`combine_sys.py:37-46`:
```python
delta_err = np.sqrt(np.abs(sys_err**2 - default_err**2))
significance = (delta_err < abs(delta))
if significance: sum_of_unc += delta**2 - delta_err**2
```
**Correction (2026-06-02): the original concern was wrong.** Barlow (*Systematic
Errors: Facts and Fictions*, 2002) does NOT restrict `√|σ_var²−σ_def²|` to subsets. If
the default is the **optimal** (min-variance) estimator `a` and the variation `a'` is a
*less-efficient* analysis on the **same data**, then `δ = a'−a` is uncorrelated with `a`
(else `a` wasn't optimal), giving `V(δ) = σ_var² − σ_def²`. So the formula and the
`δ² − delta_err²` systematic variance are valid for any same-data, sub-optimal-or-equal
variation — exactly the special systematics (half-range, cubic, y-integrated ε), not
just subsets. The earlier "delta_err ≈ 0 → always passes / overestimates" reasoning was
also off: for sys_tag_5 (half range) σ_var is genuinely larger so delta_err is real;
it only nears zero when the variation is nearly as precise as the default, in which case
the deviation legitimately carries little statistical noise.

**No code change.** Regular (1,2,3) and special (5,6,7) keep the **unified** Barlow rule
(user decision). One validity condition worth monitoring: the result assumes
`σ_def ≤ σ_var`; the `np.abs(...)` silently absorbs the reverse — if a special-sys
variation ever comes out *more* precise than nominal, the Barlow premise is violated and
abs() hides it. Shouldn't happen for these variations, but cheap to sanity-check.

### [x] 7. `sys_divisor = 3` convention — ACCEPTED (deliberate conservative prior, no action)
`combine_sys.py:43-46`, `config.yaml:21`. Combined deviation → σ via
`√(Σ(δ²−σ_stat²)/3)`. Dividing by 3 = variance of a uniform half-width δ — correct only
if the true value is uniformly distributed within ±δ (a prior). If a variation reflects
a genuine bias, /3 **understates** the systematic by ~42%. Also sums all sources then
divides by 3 (treats aggregate as one uniform). Highest-leverage convention in the
systematic budget; needs explicit justification.

**ACCEPTED (2026-06-02, user decision).** √3 = half-width-uniform variance is a
deliberate, standard, conservative-leaning convention (more conservative than the
full-width-uniform alternative, /√12). The "correct" alternative — characterizing the
true distribution of the observable under the full ensemble of cut variations — is
computationally prohibitive and rarely done in practice. The choice is a justified
domain-expert prior, not a defect. **No code change.** Recommendation: state the
assumption explicitly in the paper ("half-width uniform → σ = range/√3").

---

## 🟡 Lower priority / latent bugs

### [x] 8. `DataPoint.__mul__` wrong error propagation for DataPoint × DataPoint — FIXED
`data_point.py:75-77`:
```python
new_stat_err = np.sqrt((self.stat_error*other.value)**2 + (self.sys_error*other.value)**2)
new_sys_err  = np.sqrt((self.sys_error*other.stat_error)**2 + (self.stat_error*other.sys_error)**2)
```
For z = x·y the correct form is stat_z = √((y·Δx_stat)² + (x·Δy_stat)²), likewise sys.
The code mixes x's own stat and sys into the stat term; the sys term is a product of
two errors (second-order, ~0). Scalar multiplication (`:79-83`) is fine.

**FIXED (2026-06-02).** Corrected to `√((y·Δx)²+(x·Δy)²)` per error type. The
DataPoint×DataPoint path is **not currently called anywhere** (grep across `scripts/`),
so no downstream numbers change — fixed for correctness. Verified numerically
(x=2±.1±.2, y=3±.3±.1 → z stat=0.6708, sys=0.6325, matching hand-calc).

### [x] 9. `DataPoint.average()` silently drops systematic uncertainty — FIXED
`data_point.py:125-133`: inverse-variance weighting uses only `stat_error`; returned
`ufloat` has no systematic component. Dangerous if a caller expects a total error.

**FIXED (2026-06-02).** Now returns `ufloat(mean, √(stat² + sys²))`, with the systematic
of the weighted mean propagated as independent per-point contributions
(`√Σ(w_i·σ_sys_i)² / Σw_i`). Weighting and value unchanged (still inverse-stat-variance,
per docstring). Backward-compatible when sys=0. Only caller is diagnostic prints
(`generate_paper_plots.py:425-426`), so no paper numbers change. Verified numerically.

### [x] 10. `SimpleProfile.errors()` low-count fallback — NOT AN ISSUE (faithful ROOT TProfile Approximate convention)
`simple_profile.py:60-65`: for bins with `c < 5` (or degenerate spread) the error is
replaced by `2·√|⟨y²⟩−⟨y⟩²|` — ~2× the *global* RMS, independent of the bin count.

**Correction (2026-06-02): this is not ad hoc.** It is a faithful port of ROOT's
`TProfileHelper::GetBinError` *approximate* branch: normally `spread/√N`, and for
zero-spread / `neff==0` bins it returns `2·√(global spread²)`. The factor-of-2 and the
global-spread-over-all-bins are exactly ROOT's algorithm, and the C++ that builds these
profiles explicitly enables it: `Approximate(true)` at
`combine_lambda_with_eff.cpp:79,140,172` (and the no-eff twin). It is *conservative* —
unreliable low-count bins get large errors and are down-weighted in the v₁ fit. Only
deviation from vanilla ROOT: the extra `test < 1e-4` trigger for `c<5` bins (stricter,
same spirit). **No code change.**

### [x] 11. Signal/background fraction treated as exact in the v₁ fit — QUANTIFIED, neglected (bounded)
`fit_v1.py:485-487, 800`: the background fraction `b/(b+s)` from the mass fit is frozen
(`func_return`) and used with no uncertainty when separating signal v₁ from background
v₁. A standard contributor to invariant-mass-method v₁ error, currently neglected as a
statistical term.

**Quantified (2026-06-02, `/tmp/quant_sb.py`; perturb B/S by its mass-fit uncertainty,
refit v₁ — upper bound since perturbation slightly overstates):**

| cell | S/(S+B) | rel.unc B/S | ΔV₁/stat |
|---|---|---|---|
| Λ̄ 7.7 c1y19 (periph) | 0.25 | 25% | 0.15 |
| Λ̄ 7.7 c1y0 (periph)  | 0.33 | 20% | 0.47 |
| Λ̄ 7.7 c7y2 (hi-stat) | 0.33 | 1.8% | 0.02 |
| Λ 19.6 c5y10 (hi-stat)| 0.93 | 0.4% | 0.00 |

≤2% of the per-point stat error for high-stat/high-purity bins (most of the dataset);
up to 15–47% of *that point's* stat error for the noisiest peripheral anti-Λ bins. But
the S/B fraction is fit *independently per (cen,ybin)*, so it averages down in the
dv₁/dy slope fit (~1/√N_bins) → ≤~10% on the worst-case low-energy peripheral anti-Λ
**slope** error, negligible elsewhere, and it affects only the error bar (not the
central value). **Neglected (user decision)**, magnitude documented as a known bound.
Revisit only if a future result hinges on those specific peripheral anti-Λ slope errors.

### [x] 12. East/West efficiency handling inconsistent for v₁(pT) profiles — FIXED
`combine_lambda_with_eff.cpp:160` added the east v₁ profile with weight `sign_flip` (−1,
no ε), while the mass histogram (`:164`) used `eff_weight_e`; west profile (`:191`) used
`1.0`. So pT-differential v₁ profiles were not ε-reweighted across the y-window while the
mass histograms were.

**Quantified (2026-06-02):** ε varies 11–31% with y at fixed pT (`/tmp/quant_effy.py`),
and the missing y-folding correction shifts v₁(pT) by **~3–4% (up to ~8%)** where v₁ is
measurable (mid-peripheral), negligible at central (v₁≈0) (`/tmp/quant_ptshift.py`). A
coherent central-value bias, not just an error effect.

**ROOT behaviour verified** (`/tmp/test_tprofile_add.cpp`): `TProfile::Add(p,c)` scales
Σ(wy) by signed `c` but entries by `|c|`, so weight `−eff_weight_e` yields exactly the
efficiency-weighted, sign-flipped mean (also cross-confirms the existing v₁(y)
weighting). 

**FIXED (2026-06-02):** east weight `sign_flip` → `sign_flip * eff_weight_e`, west
`1.0` → `eff_weight_w` (`combine_lambda_with_eff.cpp`). `puncorrected` profiles kept at
weight 1.0 (raw entry count for the error correction). Full pipeline re-run launched to
regenerate; validate v₁(pT) shifts ~3–4% as predicted once it completes.

### [ ] 13. chi²/ndf uses diagonal stat⊕sys, no systematic covariance — DEFERRED (discuss later)
`generate_paper_plots.py:139` `calculate_chi2_per_ndf` and `fig3_5080.py:86`
`chi2_per_ndf_total`: χ²/ndf = Σ_i (ΔΛ_i−combo_i)² / (σ_stat,i²+σ_sys,i²) / (N−n_p),
n_p=0. Correctly includes both ΔΛ and piKp-combo errors (diff formed via DataPoint
subtraction) and ndf=N for a fixed-model consistency test. **But uses only the diagonal
of the covariance — assumes systematics are uncorrelated across energy points; no
V_sys.** Rigorous χ² = rᵀV⁻¹r with V = diag(σ_stat²)+V_sys.

Materiality (2026-06-02, `/tmp/chi2_cmp.py`): the systematic is ~half the budget —
including it diagonally roughly halves χ² (e.g. 10–40% p: stat-only 35.95 →
stat⊕sys 17.38; 50–80% p: 11.17 → 4.26). So the treatment matters. Likely
cross-energy-correlated components: the **piKp reference** (combo; common published
analysis), and partially the half-y (tag 5) and efficiency-method (tag 7) sources;
subsets (1/2/3) are genuinely independent. If correlated sys are treated as
independent, χ² is **underestimated** (agreement overstated). New fig_3_*_5080.pdf
χ²/ndf (post tag-6 removal): 0–10% p=5.65/p−K=1.61, 10–40% p=17.38/p−K=7.76,
50–80% p=4.26/p−K=1.76. **DEFERRED** per user — revisit treatment (build V vs confirm
independence vs bracket).

---

## ✅ Done correctly (credit where due)

- **Binned Poisson likelihood** for the mass fit: `cost.ExtendedBinnedNLL`
  (`fit_v1.py:325`) — right choice for binned counts.
- **Binomial efficiency errors**: `hEff->Divide(..., "B")`
  (`calculate_lambda_eff.cpp:55,106`).
- **Inverse-variance weighting** in `Measurement` (`measurement.py:21,32`), correct
  error on the weighted mean (1/√Σwᵢ), sensible zero-error fallback.
- **Masking negative-resolution centralities** (`plot_v1.py:64`).
- **NaN bins given infinite error / zero weight** for merging
  (`plot_v1.py:217,295,379`).
- Λ–Λ̄ difference adds species stat errors in quadrature (`plot_v1.py:373`) — the
  conservative/standard choice.

---

## Suggested triage order
1. Replace `soft_l1` with plain χ² in the v₁ extraction (or validate coverage) — §1.
2. Quantify dropped-bin fraction and its correlation with v₁ — §2.
3. Add an efficiency-uncertainty systematic (or propagate it) — §3.
4. Decide/document `sys_divisor`; fix the Barlow test for the correlated "special"
   systematics — §6, §7.
5. Propagate the resolution as a correlated post-fit scale on the slope — §4.
