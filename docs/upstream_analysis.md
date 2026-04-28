# Upstream Analysis: KFParticle Lambda Reconstruction

Source: `/mnt/d/Research/kfparticle_v1_19GeV`
Relevant directories: `StRoot/KFParticle`, `StRoot/KFParticlePerformance`, `StRoot/StKFParticleAnalysisMaker`

## Overview

Lambda ($\Lambda^0$) and Lambdabar ($\bar\Lambda^0$) candidates are fully reconstructed using the **KFParticle package** — a Kalman filter-based vertex fitter that simultaneously fits the decay vertex, propagates full covariance matrices through the magnetic field, and assigns PDG mass hypotheses to daughter tracks. The full reconstruction chain (track selection, vertex fitting, candidate selection, histogram filling) is handled within this framework via `StKFParticleAnalysisMaker`.

## Daughter Track Selection

| Cut | Value |
|-----|-------|
| DCA of daughter to primary vertex | < 3.0 cm (3D) |
| Daughter pT range | 0.15–2.0 GeV/c |

Applied to both proton and pion daughters before vertex fitting.

## V0 (Lambda Candidate) Topological Cuts

| Cut | Value | Notes |
|-----|-------|-------|
| Decay length | > 5.0 cm (3D) | Distance from decay vertex to PV |
| Decay length significance (L/σ_L) | > 10 | 3D significance |
| χ²/ndf of Kalman vertex fit | < 10 | KFParticle constrained fit quality |

## Invariant Mass

- **PDG Λ mass**: 1.115683 GeV/c²
- **Signal window**: ±5 MeV/c² around PDG mass (approximately 3σ)
- **Sideband region**: 4σ–6σ from peak (~8–12 MeV/c² away)
- Invariant mass histograms: 400 bins over 1.0–1.2 GeV/c² (1.25 MeV/c² per bin)
- Mass is computed from the KFParticle-fitted 4-momentum (not raw track 4-momenta)

## Background Estimation

- Method: **sideband subtraction** (symmetric sidebands at 4σ–6σ from peak)
- No like-sign pairs, event-mixing, or rotation method used at this stage

## Binning for v1 Analysis

- **Rapidity**: 20 bins (full phase space)
- **pT**: 30 bins (0–3 GeV/c)
- **Centrality**: multiple centrality bins; mass resolution σ varies slightly (2.05–2.20 MeV/c²)

## KFParticle Framework Notes

- Based on: "Reconstruction of decayed particles based on the Kalman filter" (CBM-SOFT note 2007-003)
- Daughters are assigned proton and pion mass hypotheses
- `Construct()`: fits daughters to common secondary vertex
- `SetProductionVertex()`: can constrain to primary vertex
- `SetMassConstraint()`: optional PDG mass constraint
- The fit minimizes χ² of track residuals at the fitted vertex
