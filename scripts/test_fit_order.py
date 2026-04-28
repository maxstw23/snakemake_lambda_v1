"""
Diagnostic: test whether cubic dv1/dy fit is needed within |y| < 0.6.

Criterion (same as plot_v1.py): cubic is needed if |c| > c_err.

Output:
  - stdout: table of (energy, particle, centrality) combinations where cubic is needed
  - PDF:
      Pages 1-3 : |c|/c_err heatmap across all energies × centralities
                  (one per particle: Lambda, Lambdabar, DeltaLambda)
      Pages 4+  : chi2/ndf comparison (linear vs cubic) per energy
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml
from iminuit import cost, Minuit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from uncertainties import unumpy

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)

Y_CUT = float(config['y_cut'])
CENTRALITIES = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
CENTRALITY_BINS = np.array([80, 70, 60, 50, 40, 30, 20, 10, 5, 0])
CEN_LABELS = [f'{CENTRALITY_BINS[c]}-{CENTRALITY_BINS[c-1]}%' for c in range(1, 10)]

ALL_PARTICLES = config['particles'] + ['DeltaLambda']


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def _linear(x, a):
    return a * x

def _cubic(x, a, c):
    return a * x + c * x ** 3

def fit_linear(x, y, ye):
    m = Minuit(cost.LeastSquares(x, y, ye, _linear), a=0)
    m.migrad()
    if not m.valid:
        return None, None, None
    return float(m.values['a']), float(m.errors['a']), float(m.fmin.reduced_chi2)

def fit_cubic(x, y, ye):
    m = Minuit(cost.LeastSquares(x, y, ye, _cubic), a=0, c=0.01)
    m.migrad()
    if not m.valid:
        return None, None, None, None, None
    return (float(m.values['a']), float(m.errors['a']),
            float(m.values['c']), float(m.errors['c']),
            float(m.fmin.reduced_chi2))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_energy_from_csv(path: str) -> str:
    m = re.search(r'_v1_(\w+)\.csv', Path(path).name)
    return m.group(1) if m else ''

def parse_particle_from_csv(path: str) -> str:
    m = re.search(r'fit_(Lambda\w*)_v1_', Path(path).name)
    return m.group(1) if m else ''

def parse_energy_from_root(path: str) -> str:
    m = re.search(r'_(\d+p?\d*GeV)\.root', Path(path).name)
    return m.group(1) if m else ''

def load_csv(path: str) -> dict[int, dict]:
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    return {int(cen): {s: df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
            for cen in df.columns.levels[0]}

def get_v1_in_ycut(data: dict, cen: int, res) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_centers, v1_values, v1_errors) within |y| < Y_CUT, resolution-corrected."""
    d = data[cen]
    n = len(d['values'])
    yc = 0.5 * (np.linspace(-1., 1., n + 1)[:-1] + np.linspace(-1., 1., n + 1)[1:])
    good = ~np.isnan(d['values'].astype(float)) & (np.abs(yc) < Y_CUT)
    v1_u = unumpy.uarray(d['values'][good], d['errors'][good]) / res
    return yc[good], unumpy.nominal_values(v1_u), unumpy.std_devs(v1_u)

def get_delta_v1_in_ycut(data_lam: dict, data_lbar: dict, cen: int,
                          res) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_centers, delta_v1, delta_v1_err) for Lambda - Lambdabar.

    Bins where either particle has NaN are excluded, as in plot_v1.py.
    Resolution correction is applied identically to both before subtraction.
    """
    d_lam  = data_lam[cen]
    d_lbar = data_lbar[cen]
    n = len(d_lam['values'])
    yc = 0.5 * (np.linspace(-1., 1., n + 1)[:-1] + np.linspace(-1., 1., n + 1)[1:])
    nan_mask = (np.isnan(d_lam['values'].astype(float)) |
                np.isnan(d_lbar['values'].astype(float)))
    good = ~nan_mask & (np.abs(yc) < Y_CUT)
    delta_u = (unumpy.uarray(d_lam['values'][good],  d_lam['errors'][good]) -
               unumpy.uarray(d_lbar['values'][good], d_lbar['errors'][good])) / res
    return yc[good], unumpy.nominal_values(delta_u), unumpy.std_devs(delta_u)


# ---------------------------------------------------------------------------
# Fit one (x, y, ye) dataset — returns result dict or None
# ---------------------------------------------------------------------------

def run_fits(x, y, ye) -> dict | None:
    if len(x) < 3 or np.all(ye == 0):
        return None
    lin_a, lin_a_err, lin_chi2 = fit_linear(x, y, ye)
    cub_a, cub_a_err, cub_c, cub_c_err, cub_chi2 = fit_cubic(x, y, ye)
    if lin_a is None or cub_c is None:
        return None
    return dict(
        lin_a=lin_a, lin_a_err=lin_a_err, lin_chi2=lin_chi2,
        cub_a=cub_a, cub_a_err=cub_a_err,
        cub_c=cub_c, cub_c_err=cub_c_err, cub_chi2=cub_chi2,
        needs_cubic=abs(cub_c) > cub_c_err,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fit_csvs: list[str], res_files: list[str], output: str) -> None:
    energy_order = config['energies']

    # Build lookup maps
    res_map = {parse_energy_from_root(f): f for f in res_files}
    csv_map: dict[tuple[str, str], str] = {}
    for f in fit_csvs:
        particle = parse_particle_from_csv(f)
        energy   = parse_energy_from_csv(f)
        if particle and energy:
            csv_map[(particle, energy)] = f

    # Load resolution per energy
    resolution: dict[str, np.ndarray] = {}
    cen_mask:   dict[str, np.ndarray] = {}
    for energy, rf in res_map.items():
        hres = unumpy.uarray(*[uproot.open(rf)['hEPDEP_ew_cos_1'].values(),
                                uproot.open(rf)['hEPDEP_ew_cos_1'].errors()])
        cen_mask[energy]   = unumpy.nominal_values(hres) > 0
        resolution[energy] = np.where(cen_mask[energy], np.abs(hres) ** 0.5, 0)

    # Run fits — individual particles first, then DeltaLambda
    # results[energy][particle][cen] = dict of fit quantities
    results: dict[str, dict[str, dict[int, dict]]] = {}
    for energy in energy_order:
        if energy not in res_map:
            continue
        results[energy] = {}

        # Lambda and Lambdabar
        loaded_data: dict[str, dict] = {}
        for particle in config['particles']:
            if (particle, energy) not in csv_map:
                continue
            data = load_csv(csv_map[(particle, energy)])
            loaded_data[particle] = data
            results[energy][particle] = {}
            for cen in range(1, 10):
                if not cen_mask[energy][cen - 1]:
                    continue
                res = resolution[energy][cen - 1]
                try:
                    x, y, ye = get_v1_in_ycut(data, cen, res)
                except (KeyError, IndexError):
                    continue
                r = run_fits(x, y, ye)
                if r is not None:
                    results[energy][particle][cen] = r

        # DeltaLambda — only when both particles are available
        lam_key, lbar_key = config['particles'][0], config['particles'][1]
        if lam_key in loaded_data and lbar_key in loaded_data:
            results[energy]['DeltaLambda'] = {}
            for cen in range(1, 10):
                if not cen_mask[energy][cen - 1]:
                    continue
                res = resolution[energy][cen - 1]
                try:
                    x, y, ye = get_delta_v1_in_ycut(
                        loaded_data[lam_key], loaded_data[lbar_key], cen, res)
                except (KeyError, IndexError):
                    continue
                r = run_fits(x, y, ye)
                if r is not None:
                    results[energy]['DeltaLambda'][cen] = r

    # ------------------------------------------------------------------
    # Stdout summary
    # ------------------------------------------------------------------
    print('=' * 75)
    print(f'CUBIC FIT NEEDED  (|c| > c_err, |y| < {Y_CUT}):')
    print('=' * 75)
    header = f"{'Energy':10s}  {'Particle':12s}  {'Centrality':10s}  "
    header += f"{'dv1/dy':>10s}  {'c':>10s}  {'c_err':>8s}  {'|c|/c_err':>9s}  chi2: lin / cub"
    print(header)
    print('-' * 75)
    any_found = False
    for energy in energy_order:
        if energy not in results:
            continue
        for particle in ALL_PARTICLES:
            if particle not in results[energy]:
                continue
            for cen in range(1, 10):
                r = results[energy][particle].get(cen)
                if r is None or not r['needs_cubic']:
                    continue
                any_found = True
                sig = abs(r['cub_c']) / r['cub_c_err']
                print(f"{energy:10s}  {particle:12s}  {CEN_LABELS[cen-1]:10s}  "
                      f"{r['lin_a']:>10.4f}  {r['cub_c']:>10.4f}  {r['cub_c_err']:>8.4f}  "
                      f"{sig:>9.2f}  {r['lin_chi2']:.2f} / {r['cub_chi2']:.2f}")
    if not any_found:
        print('  None — linear fit is sufficient at all energies and centralities.')
    print('=' * 75)

    # ------------------------------------------------------------------
    # PDF + SVG
    # ------------------------------------------------------------------
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    svg_stem = Path(output).with_suffix('')
    energies_present = [e for e in energy_order if e in results]
    page_idx = 0

    with PdfPages(output) as pdf:

        # Pages 1-3 : significance heatmap (one per particle incl. DeltaLambda)
        for particle in ALL_PARTICLES:
            sig_matrix = np.zeros((len(energies_present), 9))
            needed_matrix = np.zeros_like(sig_matrix, dtype=bool)
            for i, energy in enumerate(energies_present):
                pdata = results[energy].get(particle, {})
                for cen in range(1, 10):
                    r = pdata.get(cen)
                    if r and r['cub_c_err'] and r['cub_c_err'] > 0:
                        sig_matrix[i, cen - 1] = abs(r['cub_c']) / r['cub_c_err']
                        needed_matrix[i, cen - 1] = r['needs_cubic']

            fig, ax = plt.subplots(figsize=(13, 4))
            im = ax.imshow(sig_matrix, aspect='auto', cmap='RdYlGn_r',
                           vmin=0, vmax=3, origin='upper')
            plt.colorbar(im, ax=ax, label='$|c|/\\sigma_c$')
            ax.contour(needed_matrix.astype(float), levels=[0.5], colors='red', linewidths=1.5)
            ax.set_xticks(range(9))
            ax.set_xticklabels(CEN_LABELS, rotation=45, fontsize=9)
            ax.set_yticks(range(len(energies_present)))
            ax.set_yticklabels(energies_present, fontsize=10)
            label = r'$\Delta\Lambda$' if particle == 'DeltaLambda' else particle
            ax.set_title(f'{label}  —  cubic significance $|c|/\\sigma_c$  '
                         f'(red contour = needed)', fontsize=13)
            for i in range(sig_matrix.shape[0]):
                for j in range(sig_matrix.shape[1]):
                    ax.text(j, i, f'{sig_matrix[i, j]:.1f}', ha='center', va='center',
                            fontsize=8, color='black')
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(f'{svg_stem}_p{page_idx:02d}.svg', format='svg')
            page_idx += 1
            plt.close(fig)

        # Pages 4-6 : dv1/dy comparison (always-linear vs adaptive) per particle
        PARTICLE_LABELS = {
            'Lambda': r'$\Lambda$',
            'Lambdabar': r'$\bar{\Lambda}$',
            'DeltaLambda': r'$\Delta\Lambda$',
        }
        for particle in ALL_PARTICLES:
            fig, axes = plt.subplots(len(energies_present), 1,
                                     figsize=(10, 3.5 * len(energies_present)),
                                     sharex=True, squeeze=False)
            fig.suptitle(f'dv1/dy: always-linear vs adaptive  |  {PARTICLE_LABELS[particle]}',
                         fontsize=14, y=0.995)
            for row, energy in enumerate(energies_present):
                ax = axes[row, 0]
                pdata = results[energy].get(particle, {})
                cens_ok = sorted((c for c in range(1, 10) if c in pdata),
                                 reverse=True)  # most central first
                if not cens_ok:
                    ax.set_visible(False)
                    continue
                xc = CENTRALITIES[np.array(cens_ok) - 1]
                lin_vals = np.array([pdata[c]['lin_a'] for c in cens_ok])
                lin_errs = np.array([pdata[c]['lin_a_err'] for c in cens_ok])
                # Adaptive: use cubic a when needs_cubic, else linear a
                adp_vals = np.array([pdata[c]['cub_a'] if pdata[c]['needs_cubic']
                                     else pdata[c]['lin_a'] for c in cens_ok])
                adp_errs = np.array([pdata[c]['cub_a_err'] if pdata[c]['needs_cubic']
                                     else pdata[c]['lin_a_err'] for c in cens_ok])
                needs = [pdata[c]['needs_cubic'] for c in cens_ok]

                needs_arr = np.array(needs)
                w = 1.0
                ax.errorbar(xc - w, lin_vals, yerr=lin_errs, fmt='s', color='C0',
                            capsize=3, label='always linear')
                # Adaptive: split into linear-chosen and cubic-chosen
                if np.any(~needs_arr):
                    ax.errorbar(xc[~needs_arr] + w, adp_vals[~needs_arr],
                                yerr=adp_errs[~needs_arr], fmt='o', color='C3',
                                capsize=3, label='adaptive (linear)')
                if np.any(needs_arr):
                    ax.errorbar(xc[needs_arr] + w, adp_vals[needs_arr],
                                yerr=adp_errs[needs_arr], fmt='*', color='C3',
                                capsize=3, markersize=10, label='adaptive (cubic)')
                ax.axhline(0, color='k', ls='--', lw=0.7)
                ax.set_xlabel('Centrality (%)', fontsize=11)
                ax.set_ylabel(r'$dv_1/dy$', fontsize=11)
                ax.set_title(energy, fontsize=11, loc='left')
                if row == 0:
                    ax.legend(fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(f'{svg_stem}_p{page_idx:02d}.svg', format='svg')
            page_idx += 1
            plt.close(fig)

        # Pages 7+ : chi2/ndf per energy (3 panels: Lambda, Lambdabar, DeltaLambda)
        for energy in energies_present:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
            fig.suptitle(f'$\\chi^2$/ndf — linear vs cubic  |  {energy}', fontsize=13)
            for ax, particle in zip(axes, ALL_PARTICLES):
                pdata = results[energy].get(particle, {})
                cens_ok = [c for c in range(1, 10) if c in pdata]
                if not cens_ok:
                    ax.set_visible(False)
                    continue
                lin_chi2 = [pdata[c]['lin_chi2'] for c in cens_ok]
                cub_chi2 = [pdata[c]['cub_chi2'] for c in cens_ok]
                needs    = [pdata[c]['needs_cubic'] for c in cens_ok]
                xi = np.arange(len(cens_ok))
                w  = 0.35
                ax.bar(xi - w/2, lin_chi2, w, label='linear', color='C0', alpha=0.85)
                ax.bar(xi + w/2, cub_chi2, w, label='cubic',  color='C1', alpha=0.85)
                ax.axhline(1.0, color='k', linestyle='--', lw=0.8)
                for i, need in enumerate(needs):
                    if need:
                        ymax = max(lin_chi2[i], cub_chi2[i])
                        ax.text(i, ymax + 0.05, '*', ha='center', fontsize=16, color='C3')
                ax.set_xticks(xi)
                ax.set_xticklabels([CEN_LABELS[c - 1] for c in cens_ok],
                                   rotation=45, fontsize=9)
                ax.set_ylabel(r'$\chi^2$/ndf', fontsize=12)
                label = r'$\Delta\Lambda$' if particle == 'DeltaLambda' else particle
                ax.set_title(label, fontsize=12)
                ax.legend(fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(f'{svg_stem}_p{page_idx:02d}.svg', format='svg')
            page_idx += 1
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test cubic vs linear dv1/dy fit within |y| < y_cut')
    parser.add_argument('--fit_csvs', nargs='+', required=True,
                        help='v1(y) fit CSV files (any mix of no_eff / sys_tag_0)')
    parser.add_argument('--res_files', nargs='+', required=True,
                        help='ROOT data files containing hEPDEP_ew_cos_1 per energy')
    parser.add_argument('--output', required=True, help='Output PDF path')
    args = parser.parse_args()
    main(args.fit_csvs, args.res_files, args.output)
