"""Final-paper 9-panel raw-centrality v1(y) plots, with systematic uncertainties.

This reproduces the layout of the per-sys-tag ``plots/sys_tag_0/v1_cen_{energy}.pdf``
(a 3x3 grid of the 9 raw centralities, v1(y) vs y for Lambda and Lambdabar), but adds
the per-point systematic uncertainty.

The per-point v1(y) systematic is built the same way ``combine_sys.py`` builds the
merged-centrality v1(y) systematic: for every raw centrality and y bin we take the
resolution-corrected v1(y) under each sys-tag variation and add, in quadrature, the
significance-gated deviations from the default::

    sys^2 = ( sum_tags  [ Delta^2 - Delta_err^2 ]_{Delta_err < |Delta|} ) / sys_divisor

Only variations that change the *measurement* are included: the regular sys-tags
(1, 2, 3) and the y-integrated efficiency variation (special tag 7). The variations
that only change the v1(y)-vs-y *fit* are excluded, because they do not alter the
per-point v1(y) values:

    * special tags 5 / 8 -- positive / negative half-y fit range
    * tag 6              -- cubic fit function

Statistical errors are drawn as error bars; the systematic as a shaded box of
half-height ``error_sys`` centred on each point.
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml
from matplotlib.patches import Rectangle

# Centrality bin edges (%) shared with plot_v1.py; raw centrality `cen` (1..9) spans
# centralities_bins[cen]..centralities_bins[cen-1].
CENTRALITIES_BINS = np.array([80, 70, 60, 50, 40, 30, 20, 10, 5, 0])

# Variations whose v1(y) points feed the per-point systematic (default is tag 0).
# Tags 5/8 (half-y fit range) and 6 (cubic fit) are deliberately absent: they only
# change the v1(y)-vs-y fit, not the per-point v1(y).
REGULAR_SYS_TAGS = ['1', '2', '3']
SPECIAL_SYS_TAGS = ['7']

# Fit-failure guard. Across all tags the legitimate raw (uncorrected) |v1| tops out at
# ~0.28 (tags 0/2/3/7), whereas every failed fit lands above 0.3 (sys_tag_1's peripheral
# edge bins, plus a stray tag-0 point). So a raw value above V1_RAW_MAX is a failed fit
# and is dropped from every tag -- this keeps it out of both the plot and the per-point
# systematic, where (unlike the merged v1(y)) there is no 1/error^2 averaging to dilute
# it. A loose error guard removes any remaining giant-error garbage.
V1_RAW_MAX = 0.3
V1_RAW_ERR_MAX = 1.0


def _latest_result(pattern: str) -> str:
    """Newest ``result<N>_..._<energy>.root`` for a glob, keyed by the leading number
    in the basename (matches the dataset selection in the Snakefile)."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(pattern)
    return sorted(files, key=lambda x: int(re.search(r'\d+', Path(x).name).group()))[-1]


def _tag_sources(energy: str):
    """Map every sys-tag used in the combination to its (fit-csv dir, resolution file).

    Resolution comes from the same combined dataset that ``plot_v1.py`` uses as ``fres``:
    the regular tags use their own dataset, tag 7 (y-integrated eff) uses the default.
    """
    default_fres = _latest_result(f'data/result*_{energy}.root')
    sources = {'0': ('result/sys_tag_0', default_fres)}
    for t in REGULAR_SYS_TAGS:
        sources[t] = (f'result/sys_tag_{t}',
                      _latest_result(f'data/sys_tag_{t}/result*_{energy}.root'))
    for t in SPECIAL_SYS_TAGS:
        sources[t] = (f'result/special_sys_tag_{t}', default_fres)
    return sources


def _resolution(fres: str) -> np.ndarray:
    """Per-centrality event-plane resolution, resolution[cen-1] = sqrt(|<cos>|).

    Centralities with a non-positive resolution are masked to 0 (as in plot_v1.py),
    which flags the corresponding panels as empty.
    """
    h = uproot.open(fres)['hEPDEP_ew_cos_1']
    vals = h.values()
    return np.where(vals > 0, np.sqrt(np.abs(vals)), 0.0)


def _v1_y(fit_csv: str, fres: str):
    """Resolution-corrected v1(y) for one (particle, sys-tag).

    Returns (y_centers, value[cen, ybin], error[cen, ybin]) on the full y grid with
    centralities indexed 1..9 -> rows 0..8. Bins/centralities without a measurement
    are NaN (value) / inf (error) so they drop out of both plotting and the
    significance-gated systematic.
    """
    df = pd.read_csv(fit_csv, header=[0, 1], index_col=0)
    resolution = _resolution(fres)
    cens = sorted(int(c) for c in df.columns.levels[0])
    num_ybin = len(df.index)
    edges = np.linspace(-1.0, 1.0, num_ybin + 1)
    y = 0.5 * (edges[:-1] + edges[1:])

    value = np.full((9, num_ybin), np.nan)
    error = np.full((9, num_ybin), np.inf)
    for cen in cens:
        res = resolution[cen - 1]
        if res <= 0:
            continue
        v = df.loc[:, (str(cen), 'values')].to_numpy(dtype=float)
        e = df.loc[:, (str(cen), 'errors')].to_numpy(dtype=float)
        good = ~np.isnan(v) & (np.abs(v) < V1_RAW_MAX) & (np.abs(e) < V1_RAW_ERR_MAX)
        value[cen - 1, good] = v[good] / res
        error[cen - 1, good] = e[good] / res
    return y, value, error


def _combine_systematic(default, variations, sys_divisor):
    """Per-point systematic from significance-gated quadrature of tag deviations.

    `default` / each `variations` item is (value[cen, ybin], error[cen, ybin]).
    Returns sys[cen, ybin]; NaN wherever the default point itself is missing.
    """
    def_val, def_err = default
    var = np.zeros_like(def_val)
    # Missing centralities/bins carry inf error; the resulting inf/NaN intermediates are
    # gated out below (significance is False), so ignore the expected numpy warnings.
    with np.errstate(invalid='ignore'):
        for sys_val, sys_err in variations:
            delta = def_val - sys_val
            delta_err = np.sqrt(np.abs(sys_err ** 2 - def_err ** 2))
            significant = delta_err < np.abs(delta)
            var += np.where(significant, delta ** 2 - delta_err ** 2, 0.0)
    sys = np.sqrt(var / sys_divisor)
    sys[np.isnan(def_val)] = np.nan
    return sys


def build_v1_cen_y(energy: str, sys_divisor: float):
    """Resolution-corrected v1(y) for Lambda/Lambdabar at every raw centrality, with
    statistical and (cross-sys-tag) systematic errors. Returns a dict keyed by particle."""
    sources = _tag_sources(energy)
    out = {}
    for particle in ['Lambda', 'Lambdabar']:
        per_tag = {}
        y = None
        for tag, (fitdir, fres) in sources.items():
            fit_csv = f'{fitdir}/fit_{particle}_v1_{energy}.csv'
            if not Path(fit_csv).exists():
                if tag == '0':
                    raise FileNotFoundError(fit_csv)
                print(f'  [warn] {energy} {particle}: missing {fit_csv}, skipping tag {tag}')
                continue
            y, value, error = _v1_y(fit_csv, fres)
            per_tag[tag] = (value, error)
        default = per_tag['0']
        variations = [per_tag[t] for t in (REGULAR_SYS_TAGS + SPECIAL_SYS_TAGS) if t in per_tag]
        sys = _combine_systematic(default, variations, sys_divisor)
        out[particle] = {'y': y, 'value': default[0], 'stat': default[1], 'sys': sys}
    return out


def _linear_slope(y, v, stat, y_cut):
    """Weighted least-squares slope of the odd-through-origin fit v1 = a*y over the fit
    window |y| < y_cut (weights 1/stat^2), matching plot_v1.py's method-1 fit."""
    mask = (~np.isnan(v)) & (np.abs(y) < y_cut) & np.isfinite(stat) & (stat > 0)
    if mask.sum() < 1:
        return None
    w = 1.0 / stat[mask] ** 2
    return np.sum(w * y[mask] * v[mask]) / np.sum(w * y[mask] ** 2)


def plot_energy(energy: str, data, out_dir: str, y_cut: float, sys_box_halfwidth: float = 0.025):
    species = [('Lambda', 'C0', r'$\Lambda$'), ('Lambdabar', 'C1', r'$\bar{\Lambda}$')]
    x_fit = np.linspace(-1.0, 1.0, 101)
    # Shared-border 3x3 grid (no gaps), matching plots/sys_tag_0/v1_cen_*.pdf.
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
    axes = gs.subplots(sharex='col', sharey='row').flatten()
    legend_done = False
    for cen in range(1, 10):
        ax = axes[cen - 1]
        # masked centrality (no measured points): leave the panel empty
        if not any(np.any(~np.isnan(data[p]['value'][cen - 1])) for p, _, _ in species):
            continue
        for particle, color, label in species:
            d = data[particle]
            y = d['y']
            v = d['value'][cen - 1]
            stat = d['stat'][cen - 1]
            sys = d['sys'][cen - 1]
            good = ~np.isnan(v)
            if not np.any(good):
                continue
            in_fit = good & (np.abs(y) < y_cut)
            out_fit = good & ~(np.abs(y) < y_cut)
            # systematic boxes
            for yi, vi, si in zip(y[good], v[good], sys[good]):
                if not np.isfinite(si) or si <= 0:
                    continue
                ax.add_patch(Rectangle((yi - sys_box_halfwidth, vi - si),
                                       2 * sys_box_halfwidth, 2 * si,
                                       facecolor=color, edgecolor='none', alpha=0.3, zorder=1))
            # statistical error bars: filled inside the fit window, open outside
            ax.errorbar(y[in_fit], v[in_fit], yerr=stat[in_fit], fmt='o', color=color,
                        capsize=2, label=label, zorder=3)
            ax.errorbar(y[out_fit], v[out_fit], yerr=stat[out_fit], fmt='o', color=color,
                        markerfacecolor='white', capsize=2, label='_nolegend_', zorder=3)
            # linear odd-through-origin fit over |y| < y_cut (matches plot_v1.py)
            slope = _linear_slope(y, v, stat, y_cut)
            if slope is not None:
                ax.plot(x_fit, slope * x_fit, '-', color=color, zorder=2)
        ax.hlines(0., -1., 1., color='k', linestyle='--')
        ax.tick_params(labelsize=22)
        # nudged inward from the corner so it sits clear of the data points
        ax.annotate(f'{CENTRALITIES_BINS[cen]}-{CENTRALITIES_BINS[cen - 1]}%',
                    xy=(0.30, 0.88), xycoords='axes fraction', fontsize=22)
        if not legend_done:
            # upper-left is the empty corner for odd-in-y data; keeps the legend
            # off both the points and the (centre-top) centrality annotation
            ax.legend(fontsize=24, loc='upper left')
            legend_done = True
    # shared outer axis labels (one set for the whole grid)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$y$', fontsize=48, labelpad=25)
    plt.ylabel(r'$v_1$', fontsize=48, labelpad=35)
    fig.suptitle(r'$\sqrt{s_{\mathrm{NN}}}=$' + energy.replace('p', '.').replace('GeV', ' GeV'),
                 fontsize=30)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out_dir}/v1_cen_y_{energy}.pdf')
    fig.savefig(f'{out_dir}/v1_cen_y_{energy}.svg', format='svg')
    plt.close(fig)


def write_csv(energy: str, data, csv_dir: str):
    Path(csv_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    y = data['Lambda']['y']
    for cen in range(1, 10):
        cen_label = f'{CENTRALITIES_BINS[cen]}-{CENTRALITIES_BINS[cen - 1]}%'
        for j, yj in enumerate(y):
            la, lb = data['Lambda'], data['Lambdabar']
            if np.isnan(la['value'][cen - 1, j]) and np.isnan(lb['value'][cen - 1, j]):
                continue
            rows.append({
                'centrality': cen_label,
                'y': round(float(yj), 2),
                'v1_lambda': la['value'][cen - 1, j],
                'v1_lambda_err': la['stat'][cen - 1, j],
                'v1_lambda_err_sys': la['sys'][cen - 1, j],
                'v1_lambdabar': lb['value'][cen - 1, j],
                'v1_lambdabar_err': lb['stat'][cen - 1, j],
                'v1_lambdabar_err_sys': lb['sys'][cen - 1, j],
            })
    pd.DataFrame(rows).to_csv(f'{csv_dir}/v1_cen_y_{energy}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energies', nargs='+', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out_dir', default='plots/paper')
    parser.add_argument('--csv_dir', default='plots/paper/data_points')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    sys_divisor = config['sys_divisor']
    y_cut = float(config['y_cut'])

    for energy in args.energies:
        print(f'Building v1_cen_y (raw centrality) for {energy} ...')
        data = build_v1_cen_y(energy, sys_divisor)
        plot_energy(energy, data, args.out_dir, y_cut)
        write_csv(energy, data, args.csv_dir)
    print('Done.')


if __name__ == '__main__':
    main()
