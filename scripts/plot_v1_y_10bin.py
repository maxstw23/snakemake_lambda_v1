"""Final 10-y-bin v1(y) plots for Lambda, Lambdabar, Delta-Lambda, net-Lambda, excess-Lambda.

Reuses the three bin-count-agnostic v1(y) plotters from ``generate_paper_plots.py``
(``plot_v1_y``, ``plot_v1_y_netlambda``, ``plot_v1_y_excesslambda``), feeding them the
combined 10-y-bin YAMLs produced by ``run_10ybin.py`` (``plots/10ybin/final/paper_yaml/``).

Outputs go under ``plots/10ybin/paper/``:
  * v1_y.pdf/.svg                (Lambda, Lambdabar, Delta-Lambda)
  * v1_y_netlambda.pdf/.svg
  * v1_y_excesslambda.pdf/.svg
  * data_points/v1_y_{energy}.csv, v1_y_netlambda_{energy}.csv, v1_y_excesslambda_{energy}.csv
    (each with value + statistical + systematic errors)

The grid layout assumes the full 7-energy set (4x2), matching generate_paper_plots.
"""
import argparse
from pathlib import Path

import yaml

# scripts/ is on sys.path[0] when run as a script, so these resolve like in generate_paper_plots.
import generate_paper_plots as gpp

REPO = Path(__file__).resolve().parent.parent
FINAL_YAML_DIR = REPO / 'plots' / '10ybin' / 'final' / 'paper_yaml'
OUT_BASE = REPO / 'plots' / '10ybin' / 'paper'
# the plotters derive their output dir as input_path.replace('_yaml','').replace('/sys_tag_0','')
INPUT_PATH = str(REPO / 'plots' / '10ybin' / 'paper_yaml')


def main():
    config = yaml.safe_load(open(REPO / 'config.yaml'))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--energies', nargs='+', default=config['energies'])
    args = parser.parse_args()

    # config energy order (the plotters reverse internally and scale by index)
    files = [FINAL_YAML_DIR / f'dv1dy_coal_{e}.yaml' for e in args.energies]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError('missing combined 10ybin yaml(s):\n  ' + '\n  '.join(missing))

    n = len(files)
    if n != 7:
        print(f'[warn] {n} energies given; the 4x2 grid layout assumes the full 7-energy set.')
    ncol = 4 if n == 7 else 3
    nrow = 2

    (OUT_BASE / 'data_points').mkdir(parents=True, exist_ok=True)

    dict_input = {'dv1dy_coal': [str(f) for f in files]}
    figs = []
    figs = gpp.plot_v1_y(dict_input, figs, INPUT_PATH, ncols=ncol, nrows=nrow)
    figs = gpp.plot_v1_y_netlambda(dict_input, figs, INPUT_PATH, ncols=ncol, nrows=nrow)
    figs = gpp.plot_v1_y_excesslambda(dict_input, figs, INPUT_PATH, ncols=ncol, nrows=nrow)
    print(f'Wrote v1_y / v1_y_netlambda / v1_y_excesslambda under {OUT_BASE}')


if __name__ == '__main__':
    main()
