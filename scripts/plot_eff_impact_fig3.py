"""Efficiency-correction impact — energy dependence (fig-3 style).

3-panel figure (0-10%, 10-40%, 40-80%), dv1dy vs sqrt(s_NN):
  1. Delta-Lambda  NO eff correction  (computed from result/no_eff/ CSVs)
  2. Delta-Lambda  eff-corrected       (from final paper YAML)
  3. Delta(proton) - Delta(kaon)       (from PikpMergedSlope)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml
from iminuit import cost, Minuit
from matplotlib import pyplot as plt
from uncertainties import unumpy

sys.path.insert(0, str(Path(__file__).parent))
from pikp_merged import PikpMergedSlope

CONFIG = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)
ENERGIES = CONFIG['energies']
Y_CUT = float(CONFIG['y_cut'])

ENERGY_FLOAT = {
    '7p7GeV': 7.7, '9p2GeV': 9.2, '11p5GeV': 11.5,
    '14p6GeV': 14.6, '17p3GeV': 17.3, '19p6GeV': 19.6, '27GeV': 27.0,
}
ENERGY_PIKP = {e: (f"{int(ENERGY_FLOAT[e])}GeV" if ENERGY_FLOAT[e] == int(ENERGY_FLOAT[e])
               else f"{ENERGY_FLOAT[e]}GeV") for e in ENERGIES}

# centrality bin groups (1-indexed, matching plot_v1.py convention)
BIN_GROUPS = {'010': [8, 9], '1040': [5, 6, 7], '4080': [1, 2, 3, 4], '5080': [1, 2, 3]}
CENT_LABELS = {'010': '0-10%', '1040': '10-40%', '4080': '40-80%', '5080': '50-80%'}
# the third panel can show either 40-80% (default) or 50-80%
HIGH_CENT_CHOICES = ['4080', '5080']

PLOT_CFG = {
    'no_eff': dict(fmt='s', color='C1', capsize=2, ms=6, label=r'$\Delta\Lambda$ no-eff'),
    'eff':    dict(fmt='o', color='C0', capsize=2, ms=6, label=r'$\Delta\Lambda$ eff-corrected'),
    'combo':  dict(fmt='^', color='C2', capsize=2, ms=6, label=r'$\Delta p - \Delta K$'),
}
X_OFFSETS = {'no_eff': +0.2, 'eff': -0.2, 'combo': 0.0}


# ---------------------------------------------------------------------------
# Helpers
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


def per_bin_delta(lam_csv: str, lbar_csv: str, fres_path: str) -> tuple:
    """Return (delta[9], delta_err[9], cen_mask[9]) for centrality bins 1-9."""
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
    delta_err = np.full(9, np.inf)

    for cen in range(1, 10):
        if not cen_mask[cen - 1]:
            continue
        res = float(unumpy.nominal_values(resolution[cen - 1]))
        slopes, errs = [], []
        for data in [lam_data, lbar_data]:
            if cen not in data:
                slopes.append(0.0); errs.append(np.inf); continue
            d = data[cen]
            n = len(d['values'])
            edges = np.linspace(-1., 1., n + 1)
            yc = 0.5 * (edges[:-1] + edges[1:])
            ye = np.diff(edges) / 2
            good = ~np.isnan(d['values'].astype(float))
            v1_u = unumpy.uarray(d['values'][good], d['errors'][good]) / res
            y_u  = unumpy.uarray(yc[good], ye[good])
            sl, sl_err = fit_dv1dy(y_u, v1_u)
            slopes.append(sl)
            errs.append(sl_err if sl_err > 0 else np.inf)
        delta[cen - 1]     = slopes[0] - slopes[1]
        delta_err[cen - 1] = np.sqrt(errs[0]**2 + errs[1]**2)

    return delta, delta_err, cen_mask


def merge_bins(delta: np.ndarray, delta_err: np.ndarray,
               cen_mask: np.ndarray,
               merged_cens: dict[str, list[int]]) -> dict[str, tuple[float, float]]:
    """Inverse-variance weighted average over each merged centrality range."""
    result = {}
    for key, bins in merged_cens.items():
        idxs = [b - 1 for b in bins if cen_mask[b - 1] and delta_err[b - 1] < 1e6]
        if not idxs:
            result[key] = (np.nan, np.nan)
            continue
        w = 1.0 / delta_err[idxs] ** 2
        val = np.sum(w * delta[idxs]) / np.sum(w)
        err = np.sqrt(1.0 / np.sum(w))
        result[key] = (val, err)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(yaml_files: list[str], no_eff_csvs: list[str],
         data_files: list[str], output: str, high_cent: str = '4080') -> None:

    panel_keys = ['010', '1040', high_cent]
    MERGED_CENS = {k: BIN_GROUPS[k] for k in panel_keys}

    yaml_by_energy = {}
    for f in yaml_files:
        energy = Path(f).stem.split('_')[-1]
        with open(f) as fh:
            yaml_by_energy[energy] = yaml.load(fh, Loader=yaml.CLoader)

    csv_lam  = {Path(p).stem.split('_')[-1]: p for p in no_eff_csvs
                if 'Lambda_v1' in p and 'Lambdabar' not in p}
    csv_lbar = {Path(p).stem.split('_')[-1]: p for p in no_eff_csvs
                if 'Lambdabar_v1' in p}
    res_by_energy = {Path(p).stem.split('_')[-1]: p for p in data_files}

    pikp = PikpMergedSlope().get_data()

    # Collect data per energy per centrality range
    data: dict[str, dict] = {key: {'eff': [], 'eff_err': [],
                                    'no_eff': [], 'no_eff_err': [],
                                    'combo': [], 'combo_err': []}
                              for key in MERGED_CENS}

    for energy in ENERGIES:
        # --- corrected ΔΛ from YAML ---
        yaml_d = yaml_by_energy.get(energy, {})
        for key in MERGED_CENS:
            field = f'dv1dy_deltalambda_{key}'
            if field in yaml_d:
                v = yaml_d[field]
                data[key]['eff'].append(v['value'])
                data[key]['eff_err'].append(v['error_stat'])
            else:
                data[key]['eff'].append(np.nan)
                data[key]['eff_err'].append(np.nan)

        # --- no-eff ΔΛ ---
        if energy in csv_lam and energy in csv_lbar and energy in res_by_energy:
            try:
                delta, delta_err, cen_mask = per_bin_delta(
                    csv_lam[energy], csv_lbar[energy], res_by_energy[energy])
                merged = merge_bins(delta, delta_err, cen_mask, MERGED_CENS)
                for key in MERGED_CENS:
                    data[key]['no_eff'].append(merged[key][0])
                    data[key]['no_eff_err'].append(merged[key][1])
            except Exception as e:
                print(f'Warning: no-eff failed for {energy}: {e}')
                for key in MERGED_CENS:
                    data[key]['no_eff'].append(np.nan)
                    data[key]['no_eff_err'].append(np.nan)
        else:
            for key in MERGED_CENS:
                data[key]['no_eff'].append(np.nan)
                data[key]['no_eff_err'].append(np.nan)

        # --- Δp - ΔK from PikpMergedSlope ---
        pikp_e = ENERGY_PIKP[energy]
        if pikp_e in pikp:
            for key in MERGED_CENS:
                fit_key = f'{key}_linear'
                dp  = pikp[pikp_e]['protons'][fit_key]['delta']
                dpe = pikp[pikp_e]['protons'][fit_key]['delta_err']
                dk  = pikp[pikp_e]['kaons'][fit_key]['delta']
                dke = pikp[pikp_e]['kaons'][fit_key]['delta_err']
                data[key]['combo'].append(dp - dk)
                data[key]['combo_err'].append(np.sqrt(dpe**2 + dke**2))
        else:
            for key in MERGED_CENS:
                data[key]['combo'].append(np.nan)
                data[key]['combo_err'].append(np.nan)

    xs = np.array([ENERGY_FLOAT[e] for e in ENERGIES])

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.subplots_adjust(hspace=0)

    for ax, key in zip(axes, panel_keys):
        d = data[key]
        for series, label_key in [('eff', 'eff'), ('no_eff', 'no_eff'), ('combo', 'combo')]:
            vals = np.array(d[series], dtype=float)
            errs = np.array(d[f'{series}_err'], dtype=float)
            x = xs + X_OFFSETS[label_key]
            mask = np.isfinite(vals) & np.isfinite(errs)
            ax.errorbar(x[mask], vals[mask], yerr=errs[mask], **PLOT_CFG[label_key])
            # sys error band for eff-corrected
            if series == 'eff':
                for i, energy in enumerate(ENERGIES):
                    yaml_d = yaml_by_energy.get(energy, {})
                    field = f'dv1dy_deltalambda_{key}'
                    if field in yaml_d and 'error_sys' in yaml_d[field]:
                        sys_e = yaml_d[field]['error_sys']
                        ax.fill_between(
                            [xs[i] + X_OFFSETS['eff'] - 0.15,
                             xs[i] + X_OFFSETS['eff'] + 0.15],
                            vals[i] - sys_e, vals[i] + sys_e,
                            color=PLOT_CFG['eff']['color'], alpha=0.3, linewidth=0)

        ax.axhline(0, linestyle='--', color='k', lw=0.8)
        ax.annotate(CENT_LABELS[key], xy=(0.05, 0.88), xycoords='axes fraction', fontsize=16)
        ax.set_ylabel(r'$\Delta\,dv_1/dy$', fontsize=14)
        ax.tick_params(labelsize=12)

    axes[1].legend(fontsize=13, frameon=False, loc='upper right')
    axes[2].set_xlabel(r'$\sqrt{s_{\rm NN}}$ (GeV)', fontsize=14)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([str(ENERGY_FLOAT[e]) for e in ENERGIES], fontsize=12)
    fig.suptitle('Efficiency correction impact on ' + r'$\Delta\,dv_1/dy$' + ' (energy dependence)',
                 fontsize=13, y=1.01)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_files',  nargs='+', required=True)
    parser.add_argument('--no_eff_csvs', nargs='+', required=True)
    parser.add_argument('--data_files',  nargs='+', required=True)
    parser.add_argument('--output',      required=True)
    parser.add_argument('--high_cent',   default='4080', choices=HIGH_CENT_CHOICES,
                        help='centrality range for the third panel (default: 4080)')
    args = parser.parse_args()
    main(args.yaml_files, args.no_eff_csvs, args.data_files, args.output,
         high_cent=args.high_cent)
