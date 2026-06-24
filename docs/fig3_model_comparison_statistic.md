# fig_3: replacing two χ²/ndf with a single model-comparison statistic

## Goal

In the final `fig_3`, each centrality panel compares the energy dependence of the
Λ⁰−Λ̄⁰ directed-flow splitting against two coalescence references and currently
prints **two** numbers, `χ²/ndf(p)` and `χ²/ndf(p−K)`. We want a single
statistical quantity that answers the actual physics question: **is Λ closer to
`p−K` than to `p`, and is that preference significant?**

## What fig_3 currently does

Per centrality panel (0–10%, 10–40%, 40–80%), across the 7 BES energies, it
overlays the Λ⁰−Λ̄⁰ data against two references:

- **p** = (p−p̄) — open circles
- **p−K** = (p−p̄) − (K⁺−K⁻) — blue squares

and prints, for each reference, a goodness-of-fit of Λ against that single
reference tested vs the null "they are equal":

```
χ²/ndf = Σ (Λ − model)² / σ²  ÷  ndf       (ndf = n_energies − nparams, nparams = 0)
```

(`calculate_chi2_per_ndf` in `scripts/generate_paper_plots.py`;
`chi2_per_ndf_total` / `_plot_panel` in `scripts/fig3_5080.py`. The σ in the
denominator already includes stat⊕sys of **both** Λ and the reference, via
`DataPoint` error propagation.)

Current displayed values:

| centrality | χ²/ndf (p) | χ²/ndf (p−K) |
|---|---|---|
| 0–10%  | 5.65  | 1.61 |
| 10–40% | 17.38 | 7.76 |
| 40–80% | 5.68  | 1.15 |

## Why two χ²/ndf is the wrong tool for this question

Our question is **model selection** between two non-nested hypotheses, but two
separate goodness-of-fit numbers cannot answer it:

1. **They do not tell you whether the *difference* between the two is
   significant.** `p−K` looks better in every panel, but is that a real
   preference or statistical noise? Two numbers cannot say.
2. **The two references differ *only* by the kaon term.**
   `p − (p−K) = K = (K⁺−K⁻)`. So the *entire* discriminating power is the kaon
   lever arm. If `|K|` is small compared to the errors, the two models are
   statistically **indistinguishable** regardless of the χ² values — and the
   current display hides this.
3. **χ²/ndf ≫ 1 (e.g. 17.4 in 10–40%) conflates two things** — "this model is
   wrong / errors underestimated / residual structure" and "Λ prefers the other
   model" — without separating them.

## Recommended single quantity: a one-parameter mixing fit

Embed both hypotheses in a one-parameter family and fit the mixing fraction `f`:

```
Λ_model = p − f·K ,   f = 0 ⇒ p ,   f = 1 ⇒ p − K
```

Fit `f` by least squares to the 7 Λ points (per centrality). The single result
`f̂ ± σ_f` answers everything:

- **f̂ ≈ 0** → Λ tracks p; **f̂ ≈ 1** → Λ tracks p−K; **f̂ ≈ 0.5** → halfway.
- **Significance of each model:** `f̂/σ_f` = how many σ the data sits from p;
  `(f̂−1)/σ_f` = σ from p−K.
- **σ_f honestly exposes point (2):** `σ_f = 1 / √(Σ Kᵢ²/σᵢ²)`. If the kaon
  lever is weak, σ_f blows up and the honest conclusion is "cannot distinguish"
  — exactly what two χ²/ndf cannot tell you.

A panel would then read e.g. *"f = 0.9 ± 0.2 — consistent with p−K, 4.5σ from p"*
or *"f = 0.4 ± 0.6 — inconclusive."*

### Math intuition (why this is the right statistic)

For two simple, fully specified Gaussian hypotheses A=(p) and B=(p−K) with
`sᵢ = (Bᵢ − Aᵢ)/σᵢ = −Kᵢ/σᵢ`, the log-likelihood-ratio statistic is

```
Δχ² = χ²(A) − χ²(B) = 2 Σ sᵢ zᵢ − Σ sᵢ²     (zᵢ = (dᵢ−Aᵢ)/σᵢ)
```

which is **linear in the data**, hence Gaussian-distributed with separation
`Δ = √(Σ sᵢ²) = √(Σ Kᵢ²/σᵢ²) = 1/σ_f`. So `Δ` is the number of σ separating the
two hypotheses given our errors, and `f̂/σ_f` is the data's observed pull. The
mixing-fit packages this in the most interpretable form.

## Equivalent framings (same information, pick by taste)

- **Δχ² = χ²(p) − χ²(p−K)** (raw, *not* /ndf — same ndf, so a fair head-to-head).
  Gaussian with separation `Δ = 1/σ_f`; data preference `f̂/σ_f`. Minimal code
  change, but less self-explanatory than `f`.
- **Likelihood ratio / Bayes factor** `= exp(Δχ²/2)` — "odds of p−K vs p." One
  number, needs the odds-interpretation caveat.
- **AIC/BIC** reduce to Δχ² here (both models have 0 free parameters), so not
  worth using.

**Primary recommendation: the `f`-fit** — a standard linear least-squares that
reduces two numbers to one, directly answers the question, and is the only
option that surfaces when the comparison is inconclusive.

## Decisions to make before implementing

1. **Error treatment.** The current χ² puts stat⊕sys of both Λ and p in the
   denominator. In the `f`-fit the model itself carries error (the `f·K` term has
   `σ_K`), so strictly it is errors-in-variables: the per-point variance is
   `σ²(Λ) + σ²(p) + f²σ²(K)`. Options: iterate `f`, or linearize. Also: Λ, p, K
   come from the **same events** (shared event plane / centrality) and may be
   correlated; the current treatment assumes independence. Keep that assumption
   or attempt correlations?
2. **The χ²/ndf ≫ 1 problem.** If even the best-fit `f` leaves χ²/ndf ≫ 1 (likely
   in 10–40%), `σ_f` is optimistic. Standard fixes: report the raw fit error and
   *also* a PDG-style inflated `σ_f·√(χ²/ndf)`, or report `χ²_min/ndf` next to
   `f` so the reader sees the model is imperfect.
3. **Scope / display.** Per-centrality `f` (3 numbers) and/or one global `f`
   across all centralities? Replace both χ²/ndf annotations with `f ± σ_f`, or
   show `f` *plus* keep a single combined χ²/ndf for overall goodness-of-fit?

### Suggested starting point

Per-centrality `f̂ ± σ_f` with `χ²_min/ndf` shown alongside; independent-error
treatment to start (matching current behavior); revisit correlations only if
they matter.
