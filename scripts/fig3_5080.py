"""Generalized paper Fig. 3: Delta dv1/dy vs sqrt(s_NN), 3 panels (0-10%, 10-40%,
and a configurable high-centrality panel = 40-80% or 50-80%).

Mimics ``plot_dv1dy_energy_dependence`` in ``generate_paper_plots.py``.  The
Lambda-Lambdabar points come from the final ``dv1dy_coal_{energy}.yaml`` files;
the piKp combinations come from a ``PikpMergedSlope``-style module (default
``pikp_merged``; pass ``--pikp_module pikp_merged_altcuts`` for the alternative
proton/kaon pT-cut dataset).  Per-particle pT/p ranges are shown in the legend.

Examples:
    # default 50-80% figure (Coalescence cuts)
    python scripts/fig3_5080.py --input_dv1dy_coal plots/final/paper_yaml/dv1dy_coal_*.yaml
    # alternative-cut dataset, 40-80% layout
    python scripts/fig3_5080.py --input_dv1dy_coal ... --high_cent 4080 \
        --pikp_module pikp_merged_altcuts --cut_set altcuts --output_suffix _altcuts
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data_point import DataPoint
from generate_paper_plots import plot_config, tick_params

PIKP_CORRESPONDENCE = {'pions': ['piplus', 'piminus'],
                       'kaons': ['kplus', 'kminus'],
                       'protons': ['proton', 'antiproton']}

HIGH_LABELS = {'4080': '40-80%', '5080': '50-80%'}

# per-particle kinematic-range labels (pT, p in GeV/c) for the legend second line
CUT_SETS = {
    'default': {'Lambda': r'$0.4<p_{T}<1.8$',
                'proton': r'$0.4<p_{T}<1.8$, $p<2.0$',
                'kaon':   r'$0.28<p_{T}<1.2$, $p<1.6$'},
    'altcuts': {'Lambda': r'$0.4<p_{T}<1.8$',
                'proton': r'$0.4<p_{T}$, $p<2.0$',
                'kaon':   r'$0.2<p_{T}$, $p<1.6$'},
}


def _energy_key_from_path(f: str) -> str:
    return f.split('/')[-1].replace('.yaml', '').split('_')[-1]


def load_datapoints(files, cent_ranges, pikp_slopes, proton_fit='linear'):
    energies = [float(_energy_key_from_path(f).replace('p', '.').replace('GeV', '')) for f in files]

    datapoints: dict[str, DataPoint] = {}
    for particle in ['lambda', 'lambdabar', 'deltalambda', 'piplus', 'piminus',
                     'kplus', 'kminus', 'proton', 'antiproton',
                     'delta_lambdas', 'delta_pions', 'delta_kaons', 'delta_protons']:
        for cent in cent_ranges:
            datapoints[f'{particle}_{cent}'] = DataPoint([], [], [])

    for f in files:
        with open(f, 'r') as fh:
            data_dict = yaml.load(fh, Loader=yaml.CLoader)
        for particle in ['lambda', 'lambdabar', 'deltalambda']:
            for cent in cent_ranges:
                d = data_dict[f'dv1dy_{particle}_{cent}']
                datapoints[f'{particle}_{cent}'].add_point(d['value'], d['error_stat'], d['error_sys'])

        energy = _energy_key_from_path(f).replace('p', '.')
        for particle in ['pions', 'kaons', 'protons']:
            fit = proton_fit if particle == 'protons' else 'linear'
            for cent in cent_ranges:
                src = pikp_slopes[energy][particle][f'{cent}_{fit}']
                pos_key, neg_key = PIKP_CORRESPONDENCE[particle]
                datapoints[f'{pos_key}_{cent}'].add_point(src['pos'], src['pos_err'], src['pos_systematics'])
                datapoints[f'{neg_key}_{cent}'].add_point(src['neg'], src['neg_err'], src['neg_systematics'])
                datapoints[f'delta_{particle}_{cent}'].add_point(src['delta'], src['delta_err'], src['delta_systematics'])

    for cent in cent_ranges:
        datapoints[f'delta_lambdas_{cent}'] = datapoints[f'deltalambda_{cent}']
        datapoints[f'combo1_{cent}'] = datapoints[f'delta_protons_{cent}'] - datapoints[f'delta_kaons_{cent}']
        datapoints[f'combo2_{cent}'] = datapoints[f'delta_protons_{cent}']

    return datapoints, energies


def chi2_per_ndf_total(diff: DataPoint, nparams: int = 0) -> float:
    """chi2/ndf of ``diff`` against zero, using the *total* (stat + sys) error."""
    ndf = len(diff) - nparams
    return float(np.sum(diff.value**2 / diff.total_error()**2) / ndf)


def _plot_panel(ax, datapoints, energies, cent, cut_labels):
    """One panel: Delta-Lambda, p-pbar, (p-pbar)-(K+-K-) vs sqrt(s_NN) with sys bands.
    Legend labels carry the per-particle pT/p range on a second line."""
    e = np.array(energies)
    # (series key, plot_config key, x-offset, cut-label key for the 2nd legend line)
    series = [('delta_lambdas', 'Lambda', +0.2, 'Lambda'),
              ('combo2', 'combo2', -0.2, 'proton'),
              ('combo1', 'combo', 0.0, 'kaon')]
    y_lo, y_hi = np.inf, -np.inf
    for name, cfgkey, dx, cutkey in series:
        cfg = plot_config[cfgkey]
        label = '\n'.join([cfg['label'], cut_labels[cutkey]])
        cfg_nolabel = {k: v for k, v in cfg.items() if k != 'label'}
        dp = datapoints[f'{name}_{cent}']
        ax.errorbar(e + dx, dp.value, yerr=dp.stat_error, label=label, **cfg_nolabel)
        for i in range(len(energies)):
            ax.fill_between([e[i] + dx - 0.15, e[i] + dx + 0.15],
                            y1=dp.value[i] - dp.sys_error[i],
                            y2=dp.value[i] + dp.sys_error[i],
                            color=cfg['color'], alpha=0.4, linewidth=0)
        tot = dp.total_error()
        y_lo = min(y_lo, float(np.min(dp.value - tot)))
        y_hi = max(y_hi, float(np.max(dp.value + tot)))
    chi2_p = chi2_per_ndf_total(datapoints[f'delta_lambdas_{cent}'] - datapoints[f'combo2_{cent}'])
    chi2_pk = chi2_per_ndf_total(datapoints[f'delta_lambdas_{cent}'] - datapoints[f'combo1_{cent}'])
    ax.axhline(0, linestyle='dashed', color='black')
    ax.set_xticks(energies, labels=energies)
    ax.tick_params(**tick_params)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    return chi2_p, chi2_pk, y_lo, y_hi


def make_figure(datapoints, energies, output_dir, high_cent, suffix, cut_labels):
    high_label = HIGH_LABELS[high_cent]
    cent_ranges = ['010', '1040', high_cent]
    cent_title = {'010': '0-10%', '1040': '10-40%', high_cent: high_label}
    for is_horizontal in (False, True):
        fig = plt.figure(figsize=(20, 8) if is_horizontal else (8, 12))
        gs = fig.add_gridspec(ncols=3 if is_horizontal else 1,
                              nrows=1 if is_horizontal else 3, hspace=0, wspace=0)
        axes = gs.subplots(sharex='col', sharey='row').flatten()

        extents = {}
        for ax, cent in zip(axes, cent_ranges):
            chi2_p, chi2_pk, y_lo, y_hi = _plot_panel(ax, datapoints, energies, cent, cut_labels)
            extents[cent] = (y_lo, y_hi)
            if is_horizontal:
                ypos = 0.21 if cent == '1040' else 0.85
                ax.annotate(cent_title[cent], xy=(0.45, ypos), xycoords='axes fraction', fontsize=24)
                ax.annotate(fr'$\chi^2$/ndf (p) = {chi2_p:.2f}', xy=(0.45, ypos - 0.07), xycoords='axes fraction', fontsize=18)
                ax.annotate(fr'$\chi^2$/ndf (p - K) = {chi2_pk:.2f}', xy=(0.45, ypos - 0.14), xycoords='axes fraction', fontsize=18)
            else:
                ypos = 0.55 if cent == '1040' else 0.85
                ax.annotate(cent_title[cent], xy=(0.45, ypos), xycoords='axes fraction', fontsize=20)
                ax.annotate(fr'$\chi^2$/ndf (p) = {chi2_p:.2f}', xy=(0.45, ypos - 0.10), xycoords='axes fraction', fontsize=14)
                ax.annotate(fr'$\chi^2$/ndf (p - K) = {chi2_pk:.2f}', xy=(0.45, ypos - 0.20), xycoords='axes fraction', fontsize=14)

        # the 10-40% panel (axes[1]) carries the legend; title states the units
        axes[1].legend(loc='upper right', fontsize=15 if is_horizontal else 12,
                       frameon=False, title=r'$p_{T}$, $p$ in GeV/$c$', title_fontsize=13)

        def _ylim(lo, hi, pad_lo=0.12, pad_hi=0.40):
            span = hi - lo
            return lo - pad_lo * span, hi + pad_hi * span
        if is_horizontal:
            lo = min(v[0] for v in extents.values())
            hi = max(v[1] for v in extents.values())
            axes[0].set_ylim(*_ylim(lo, hi))
        else:
            for ax, cent in zip(axes, cent_ranges):
                ax.set_ylim(*_ylim(*extents[cent], pad_hi=0.5))

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.tick_params(**tick_params)
        plt.grid(False)
        plt.xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=24 if is_horizontal else 18, labelpad=10)
        plt.ylabel(r'$\Delta dv_1/dy$', fontsize=24 if is_horizontal else 18, labelpad=30 if is_horizontal else 18)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        out_base = output_dir.rstrip('/')
        name = 'fig_3_horizontal' if is_horizontal else 'fig_3_vertical'
        if is_horizontal:
            plt.tight_layout()
        plt.savefig(f'{out_base}/{name}{suffix}.pdf')
        plt.savefig(f'{out_base}/{name}{suffix}.svg', format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Saved: {out_base}/{name}{suffix}.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dv1dy_coal', nargs='+', required=True,
                        help='final dv1dy_coal_{energy}.yaml files')
    parser.add_argument('--output_dir', default='plots/paper')
    parser.add_argument('--high_cent', default='5080', choices=['4080', '5080'])
    parser.add_argument('--pikp_module', default='pikp_merged',
                        help='module providing PikpMergedSlope (e.g. pikp_merged_altcuts)')
    parser.add_argument('--cut_set', default='default', choices=list(CUT_SETS),
                        help='per-particle pT/p range labels shown in the legend')
    parser.add_argument('--output_suffix', default=None,
                        help='filename suffix; default derived from high_cent + cut_set')
    parser.add_argument('--proton_fit', default='linear', choices=['linear', 'cubic'])
    args = parser.parse_args()

    suffix = args.output_suffix
    if suffix is None:
        suffix = ('' if args.high_cent == '4080' else '_5080') + ('_altcuts' if args.cut_set == 'altcuts' else '')

    pikp_slopes = importlib.import_module(args.pikp_module).PikpMergedSlope().get_data()
    cent_ranges = ['010', '1040', args.high_cent]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datapoints, energies = load_datapoints(args.input_dv1dy_coal, cent_ranges, pikp_slopes,
                                           proton_fit=args.proton_fit)
    make_figure(datapoints, energies, args.output_dir, args.high_cent, suffix, CUT_SETS[args.cut_set])


if __name__ == '__main__':
    main()
