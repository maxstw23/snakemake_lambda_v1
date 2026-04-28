"""Efficiency-correction impact check.

Produces a multi-energy panel figure (same layout as the main dv1dy_coal figure)
with three series per panel:
  1. Delta-Lambda  NO eff correction  (computed from result/no_eff/ CSVs)
  2. Delta-Lambda  eff-corrected       (from final paper YAML)
  3. Delta(proton) - Delta(kaon)       (from final paper YAML)

Nothing in the existing pipeline is touched; this is a standalone diagnostic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml
from iminuit import cost, Minuit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from uncertainties import unumpy

CONFIG = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)
ENERGIES = CONFIG['energies']
Y_CUT = float(CONFIG['y_cut'])

CENTRALITIES = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])

PLOT_CFG = {
    'no_eff':    dict(fmt='s', color='C1', capsize=2, ms=5, label=r'$\Delta\Lambda$ no-eff'),
    'eff':       dict(fmt='o', color='C0', capsize=2, ms=5, label=r'$\Delta\Lambda$ eff-corrected'),
    'combo':     dict(fmt='^', color='C2', capsize=2, ms=5, label=r'$\Delta p - \Delta K$'),
}

SCALING = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}


# ---------------------------------------------------------------------------
# Helpers (same approach as plot_eff_comparison.py)
# ---------------------------------------------------------------------------

def _linear(x, a):
    return a * x


def fit_dv1dy(y_u, v1_u) -> tuple[float, float]:
    mask = np.abs(unumpy.nominal_values(y_u)) < Y_CUT
    if mask.sum() < 2:
        return 0.0, 0.0
    c = cost.LeastSquares(
        unumpy.nominal_values(y_u[mask]),
        unumpy.nominal_values(v1_u[mask]),
        unumpy.std_devs(v1_u[mask]),
        _linear,
    )
    m = Minuit(c, a=0)
    m.migrad()
    if not m.valid:
        return 0.0, 0.0
    return float(m.values['a']), float(m.errors['a'])


def load_csv(path: str) -> dict[int, dict]:
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    return {
        int(cen): {s: df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        for cen in df.columns.levels[0]
    }


def compute_delta(lam_csv: str, lbar_csv: str, fres_path: str,
                  yrebin_lam: int, yrebin_lbar: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_dv1dy, delta_err) arrays over 9 centrality bins."""
    fres = uproot.open(fres_path)
    hres = unumpy.uarray(
        fres['hEPDEP_ew_cos_1'].values(),
        fres['hEPDEP_ew_cos_1'].errors(),
    )
    cen_mask = unumpy.nominal_values(hres) > 0
    resolution = np.where(cen_mask, np.abs(hres) ** 0.5, 1.0)

    lam_data  = load_csv(lam_csv)
    lbar_data = load_csv(lbar_csv)

    delta     = np.zeros(9)
    delta_err = np.zeros(9)

    for cen in range(1, 10):
        if not cen_mask[cen - 1]:
            continue
        res = float(unumpy.nominal_values(resolution[cen - 1]))

        slopes = []
        errs   = []
        for data, yrebin in [(lam_data, yrebin_lam), (lbar_data, yrebin_lbar)]:
            if cen not in data:
                slopes.append(0.0); errs.append(0.0); continue
            d = data[cen]
            n = len(d['values'])
            edges = np.linspace(-1., 1., n + 1)
            yc = 0.5 * (edges[:-1] + edges[1:])
            ye = np.diff(edges) / 2
            good = ~np.isnan(d['values'].astype(float))
            v1_u = unumpy.uarray(d['values'][good], d['errors'][good]) / res
            y_u  = unumpy.uarray(yc[good], ye[good])
            sl, sl_err = fit_dv1dy(y_u, v1_u)
            slopes.append(sl); errs.append(sl_err)

        delta[cen - 1]     = slopes[0] - slopes[1]
        delta_err[cen - 1] = np.sqrt(errs[0]**2 + errs[1]**2)

    return delta, delta_err, cen_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(yaml_files: list[str], no_eff_csvs: list[str],
         data_files: list[str], output: str) -> None:

    # Build dicts keyed by energy
    yaml_by_energy: dict[str, dict] = {}
    for f in yaml_files:
        energy = Path(f).stem.split('_')[-1]
        with open(f) as fh:
            yaml_by_energy[energy] = yaml.load(fh, Loader=yaml.CLoader)

    csv_lam  = {Path(p).stem.split('_')[-1]: p for p in no_eff_csvs if 'Lambda_v1' in p and 'Lambdabar' not in p}
    csv_lbar = {Path(p).stem.split('_')[-1]: p for p in no_eff_csvs if 'Lambdabar_v1' in p}
    res_by_energy = {Path(p).stem.split('_')[-1]: p for p in data_files}

    ncols, nrows = 4, 2
    scaling = {i: SCALING[i] for i in range(len(ENERGIES))}

    fig = plt.figure(figsize=(ncols * 4, nrows * 4))
    gs  = fig.add_gridspec(ncols=ncols, nrows=nrows, hspace=0, wspace=0)
    axes = gs.subplots(sharex='col', sharey='row').flatten()

    for i, energy in enumerate(reversed(ENERGIES)):
        ax = axes[i]
        scale = scaling[i]
        nice_energy = energy.replace('p', '.').replace('GeV', ' GeV')

        # --- corrected Δλ from YAML ---
        data_yaml = yaml_by_energy.get(energy, {})
        x_corr   = np.array(data_yaml.get('x', []))
        y_corr   = np.array(data_yaml.get('y', []))
        ye_corr  = np.array(data_yaml.get('yerr', []))
        combo    = np.array(data_yaml.get('combo', []))
        combo_e  = np.array(data_yaml.get('combo_err', []))

        if len(x_corr):
            ax.errorbar(x_corr,     y_corr  * scale, yerr=ye_corr * scale, **PLOT_CFG['eff'])
            ax.errorbar(x_corr + 1, combo   * scale, yerr=combo_e * scale, **PLOT_CFG['combo'])

        # --- uncorrected Δλ from no-eff CSVs ---
        if energy in csv_lam and energy in csv_lbar and energy in res_by_energy:
            yrebin_lam  = CONFIG['yrebin'][energy]['Lambda']
            yrebin_lbar = CONFIG['yrebin'][energy]['Lambdabar']
            try:
                delta_no, delta_no_err, cen_mask = compute_delta(
                    csv_lam[energy], csv_lbar[energy],
                    res_by_energy[energy],
                    yrebin_lam, yrebin_lbar,
                )
                ax.errorbar(
                    CENTRALITIES[cen_mask] - 1,
                    delta_no[cen_mask]     * scale,
                    yerr=delta_no_err[cen_mask] * scale,
                    **PLOT_CFG['no_eff'],
                )
            except Exception as e:
                print(f'Warning: could not compute no-eff delta for {energy}: {e}')

        ax.hlines(0, 0, 80, linestyles='--', colors='k', lw=0.8)
        ax.set_ylim(-0.259, 0.179)
        ax.annotate(r'$\sqrt{s_{\rm NN}}=$' + nice_energy,
                    xy=(0.97, 0.93), xycoords='axes fraction',
                    fontsize=12, ha='right')
        if scale != 1:
            ax.annotate(f'{scale}×', xy=(0.05, 0.08), xycoords='axes fraction', fontsize=12)

    # legend in the last (empty) panel
    ax_leg = axes[len(ENERGIES)]
    for key in ('no_eff', 'eff', 'combo'):
        ax_leg.errorbar([], [], **PLOT_CFG[key])
    ax_leg.legend(fontsize=13, frameon=False, loc='center')
    ax_leg.axis('off')

    # hide remaining empty panels
    for j in range(len(ENERGIES) + 1, ncols * nrows):
        axes[j].axis('off')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Centrality (%)', fontsize=14)
    plt.ylabel(r'$\Delta\,dv_1/dy$', fontsize=14, labelpad=20)
    fig.suptitle('Efficiency correction impact on ' + r'$\Delta\,dv_1/dy$', fontsize=14, y=1.01)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_files',   nargs='+', required=True,
                        help='Final paper YAML files (dv1dy_coal_*.yaml)')
    parser.add_argument('--no_eff_csvs',  nargs='+', required=True,
                        help='No-eff fit CSVs (result/no_eff/fit_*.csv)')
    parser.add_argument('--data_files',   nargs='+', required=True,
                        help='Raw data ROOT files (for resolution), one per energy')
    parser.add_argument('--output',       required=True)
    args = parser.parse_args()
    main(args.yaml_files, args.no_eff_csvs, args.data_files, args.output)
