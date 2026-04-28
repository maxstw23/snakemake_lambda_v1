"""Plot Lambda efficiency vs pT for each centrality bin from processed efficiency ROOT files.

Each panel shows the y-integrated efficiency (points + fit) plus fit curves for
selected rapidity slices: two most backward, middle, two most forward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


CENTRALITY_LABELS = [
    '70-80%', '60-70%', '50-60%', '40-50%', '30-40%',
    '20-30%', '10-20%', '5-10%', '0-5%',
]

# y-bin numbering: 20 bins from -1.0 to +1.0 (width 0.1)
# Within |y| < 0.6: ybins 4-15
Y_SLICES = [
    (4,  '-0.60 < y < -0.50', 'C1', '--'),
    (5,  '-0.50 < y < -0.40', 'C2', '--'),
    (9,  '-0.10 < y <  0.00', 'C3', '-'),
    (14,  '0.40 < y <  0.50', 'C4', ':'),
    (15,  '0.50 < y <  0.60', 'C5', ':'),
]


def eval_fit(f, key: str, pt_arr: np.ndarray) -> np.ndarray | None:
    """Evaluate a TF1 stored in the ROOT file; return None if missing."""
    keys_bare = [k.split(';')[0] for k in f.keys()]
    if key not in keys_bare:
        return None
    try:
        fit_obj = f[key]
        m = fit_obj.all_members
        xmin = m.get('fXmin', 0.0)
        xmax = m.get('fXmax', 1.8)
        pars = list(m['fParams'].all_members['fParameters'])
        if len(pars) < 5:
            return None
        pt = np.clip(pt_arr, max(xmin, 0.01), xmax)
        return (pars[0] + pars[3] * pt + pars[4] * pt**2) * np.exp(-(pars[1] / pt)**pars[2])
    except Exception:
        return None


def plot_eff(eff_file: str, particle: str, energy: str, output: str) -> None:
    f = uproot.open(eff_file)
    keys_bare = [k.split(';')[0] for k in f.keys()]

    nice_energy = energy.replace('p', '.').replace('GeV', ' GeV')
    particle_label = {'lambda': r'$\Lambda$', 'lambdabar': r'$\bar{\Lambda}$'}[particle]

    pt_curve = np.linspace(0.01, 2.0, 300)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f'{particle_label} Efficiency vs $p_T$  —  {nice_energy}', fontsize=16)

    for cen in range(9):
        ax = axes[cen // 3, cen % 3]
        heff_key = f'hEff_cen{cen}'

        if heff_key not in keys_bare:
            ax.set_visible(False)
            continue

        # --- y-integrated data points ---
        hmc = f[f'hMCPt_cen{cen}']
        hreco = f[f'hRecoPt_cen{cen}']
        edges = hmc.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        mc_vals = hmc.values()
        reco_vals = hreco.values()
        values = np.where(mc_vals > 0, reco_vals / mc_vals, 0.0)
        errors = np.where(mc_vals > 0, np.sqrt(values * (1 - values) / np.maximum(mc_vals, 1)), 0.0)
        mask = (values > 0) & (centers <= 2.0)
        ax.errorbar(centers[mask], values[mask], yerr=errors[mask],
                    fmt='o', ms=3, capsize=2, color='k', alpha=0.4, label='y-integrated data')

        # --- y-integrated fit ---
        fit_int = eval_fit(f, f'fit1D_cen{cen}', pt_curve)
        if fit_int is not None:
            ax.plot(pt_curve, fit_int, 'k-', lw=1.5, label='y-integrated fit')

        # --- y-binned fits ---
        any_ybin = False
        for ybin, ylabel, color, ls in Y_SLICES:
            fit_y = eval_fit(f, f'fit1D_cen{cen}_ybin{ybin}', pt_curve)
            if fit_y is not None:
                ax.plot(pt_curve, fit_y, color=color, ls=ls, lw=1.2, label=ylabel)
                any_ybin = True

        ax.set_title(CENTRALITY_LABELS[cen], fontsize=12)
        ax.set_xlim(0, 2.0)
        ax.set_ylim(-0.02, None)
        ax.tick_params(labelsize=10)
        if cen // 3 == 2:
            ax.set_xlabel(r'$p_T$ (GeV/$c$)', fontsize=12)
        if cen % 3 == 0:
            ax.set_ylabel('Efficiency', fontsize=12)
        if cen == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)
        if not any_ybin and cen == 0:
            ax.text(0.97, 0.95, 'no y-binned fits', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8, color='gray')

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eff_file', required=True)
    parser.add_argument('--particle', required=True, choices=['lambda', 'lambdabar'])
    parser.add_argument('--energy', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot_eff(args.eff_file, args.particle, args.energy, args.output)
