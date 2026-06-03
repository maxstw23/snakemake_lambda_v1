"""Generate Fig. 2 (Delta dv1/dy vs centrality grid) for a given piKp dataset.

Thin driver around ``plot_fig_2`` in ``generate_paper_plots.py`` (parametrized to
accept an injectable piKp dataset, output suffix and per-particle cut labels), so an
alternative proton/kaon pT-cut dataset can be plotted without touching the default
``fig_2.pdf``.

    python scripts/gen_fig2.py --input_dv1dy_coal plots/final/paper_yaml/dv1dy_coal_*.yaml \
        --pikp_module pikp_merged_altcuts --cut_set altcuts --output_suffix _altcuts
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_paper_plots import plot_fig_2

CUT_SETS = {
    'default': {'Lambda': r'$0.4<p_{T}<1.8$',
                'proton': r'$0.4<p_{T}<1.8$, $p<2.0$',
                'kaon':   r'$0.28<p_{T}<1.2$, $p<1.6$'},
    'altcuts': {'Lambda': r'$0.4<p_{T}<1.8$',
                'proton': r'$0.4<p_{T}$, $p<2.0$',
                'kaon':   r'$0.2<p_{T}$, $p<1.6$'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dv1dy_coal', nargs='+', required=True,
                    help='final dv1dy_coal_{energy}.yaml files, in config energy order')
    ap.add_argument('--pikp_module', default='pikp_merged')
    ap.add_argument('--cut_set', default='default', choices=list(CUT_SETS))
    ap.add_argument('--output_suffix', default='')
    args = ap.parse_args()

    pikp = importlib.import_module(args.pikp_module).PikpMergedSlope().get_data()
    ncol = 4 if len(args.input_dv1dy_coal) == 7 else 3
    dict_input = {'dv1dy_coal': args.input_dv1dy_coal}
    plot_fig_2(dict_input, [], 'plots/sys_tag_0/paper_yaml', ncols=ncol, nrows=2,
               pikp_data=pikp, output_suffix=args.output_suffix,
               cut_labels=CUT_SETS[args.cut_set])
    print(f'Saved: plots/paper/fig_2{args.output_suffix}.pdf')


if __name__ == '__main__':
    main()
