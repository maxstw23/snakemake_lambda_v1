"""Alternative 10-y-bin Lambda v1(y) pipeline (does NOT touch the default 20-bin outputs).

The default pipeline fits v1(y) with ``fit_v1.py --yrebin 1`` (20 y bins, centers -0.95..0.95).
This driver re-runs the subset of the Snakemake DAG needed for the five v1(y) observables
(Lambda, Lambdabar, Delta-Lambda, net-Lambda, excess-Lambda) with ``--yrebin 2`` (10 y bins,
centers -0.9..0.9), writing everything under parallel ``result/10ybin/`` and ``plots/10ybin/``
trees. The existing ``result/sys_tag_*`` / ``plots/sys_tag_*`` outputs are never written.

It reuses, unchanged:
  * the already-combined ROOT histograms (yrebin does not affect the combine step),
  * the existing piKp reference CSVs and pT-fit CSVs (pT fits feed only the v1_pt panels,
    not the v1(y) arrays we care about).

Stages (per energy):
  1. fit_v1.py --yrebin 2  for sys tags 0,1,2,3 and special tag 7   (~70 fits total)
  2. plot_v1.py --out_root plots/10ybin  for tags 0,1,2,3 (regular), 5,8 (half-y, from the
     tag-0 fits) and 7 (y-integrated eff) -> per-tag dv1dy_coal_{energy}.yaml
  3. combine_sys.py  -> plots/10ybin/final/paper_yaml/dv1dy_coal_{energy}.yaml

The final v1(y) plots/CSVs are produced by ``scripts/plot_v1_y_10bin.py`` (run separately or
via ``--plot``).

Usage::

    python scripts/run_10ybin.py                       # all energies, all stages
    python scripts/run_10ybin.py --energies 27GeV      # single energy (quick check)
    python scripts/run_10ybin.py --skip_fits           # reuse existing 10ybin fits, replot
    python scripts/run_10ybin.py --jobs 4              # parallelise the fit stage
    python scripts/run_10ybin.py --plot                # also run the final v1(y) plotter
"""
import argparse
import glob
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

YREBIN = 2  # 20 raw y bins -> 10 bins

REPO = Path(__file__).resolve().parent.parent
RESULT_ROOT = REPO / 'result' / '10ybin'
PLOTS_ROOT = REPO / 'plots' / '10ybin'   # passed to plot_v1.py as --out_root
LOG_ROOT = REPO / 'logs' / '10ybin'

PYTHON = sys.executable
FIT_SCRIPT = REPO / 'scripts' / 'fit_v1.py'
PLOT_SCRIPT = REPO / 'scripts' / 'plot_v1.py'
COMBINE_SCRIPT = REPO / 'scripts' / 'combine_sys.py'
PLOT_Y_SCRIPT = REPO / 'scripts' / 'plot_v1_y_10bin.py'

# tags that need their own yrebin=2 fit
FIT_TAGS = ['0', '1', '2', '3', '7']
# regular tags that get a normal plot_v1 run (own dataset / own fits)
REGULAR_PLOT_TAGS = ['0', '1', '2', '3']
# special half-y fit-range tags: plotted from the tag-0 fits on the default dataset
SPECIAL_HALF_TAGS = ['5', '8']


def _latest(pattern: str) -> Path:
    """Newest ``result<N>...`` matching a glob, keyed by the leading int in the basename
    (matches the dataset selection in the Snakefile)."""
    files = glob.glob(str(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return Path(sorted(files, key=lambda x: int(re.search(r'\d+', Path(x).name).group()))[-1])


def _tag_dirname(tag: str) -> str:
    """plot_v1.py uses the 'special_' prefix for tag >= 5."""
    return f'special_sys_tag_{tag}' if float(tag) >= 5 else f'sys_tag_{tag}'


def combined_input(tag: str, particle: str, energy: str) -> Path:
    """Combined ROOT histogram file used as the fit input, mirroring the Snakefile.

    Prefers the efficiency-corrected file when present (as ``get_combined_file`` does, since
    eff files exist for both particles at every energy). Tag 7 mirrors
    ``get_combined_file_yint``: its own eff-corrected file, else the sys_tag_0 plain combined.
    """
    src_dir = REPO / 'result' / _tag_dirname(tag) if tag != '7' else REPO / 'result' / 'special_sys_tag_7'
    eff = src_dir / f'combined_{particle}_v1_{energy}_eff_corrected.root'
    plain = src_dir / f'combined_{particle}_v1_{energy}.root'
    if eff.exists():
        return eff
    if plain.exists():
        return plain
    if tag == '7':
        # y-integrated tag falls back to the default dataset's plain combined file
        fallback = REPO / 'result' / 'sys_tag_0' / f'combined_{particle}_v1_{energy}.root'
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f'no combined ROOT file for tag {tag} {particle} {energy} '
                            f'(looked in {src_dir})')


def fres_for_tag(tag: str, energy: str) -> Path:
    """Resolution/dataset ROOT file (``--fres``), matching the Snakefile per tag:
    regular tags 1/2/3 use their own dataset; tags 0/5/7/8 use the default dataset."""
    if tag in ('1', '2', '3'):
        return _latest(REPO / 'data' / f'sys_tag_{tag}' / f'result*_{energy}.root')
    return _latest(REPO / 'data' / f'result*_{energy}.root')


def fit_csv(tag: str, particle: str, energy: str) -> Path:
    return RESULT_ROOT / _tag_dirname(tag) / f'fit_{particle}_v1_{energy}.csv'


def coal_yaml(tag: str, energy: str) -> Path:
    return PLOTS_ROOT / _tag_dirname(tag) / 'paper_yaml' / f'dv1dy_coal_{energy}.yaml'


def _run(cmd, log_path: Path):
    """Run a subprocess, teeing combined stdout/stderr to a log file. Raises on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as log:
        proc = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed (rc={proc.returncode}): {" ".join(str(c) for c in cmd)}\n'
                           f'see log: {log_path}')


# ---------------------------------------------------------------------------- stages

def stage_fit(energies, particles, max_refit: int, jobs: int):
    """Run all yrebin=2 fits (tags 0,1,2,3,7 x particles x energies)."""
    tasks = []
    for tag in FIT_TAGS:
        out_dir = RESULT_ROOT / _tag_dirname(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        for energy in energies:
            for particle in particles:
                inp = combined_input(tag, particle, energy)
                out_csv = fit_csv(tag, particle, energy)
                cmd = [PYTHON, FIT_SCRIPT, inp, out_csv,
                       '--yrebin', YREBIN, '--max_refit', max_refit]
                log = LOG_ROOT / f'fit_{_tag_dirname(tag)}_{particle}_{energy}.log'
                tasks.append((f'fit {_tag_dirname(tag)} {particle} {energy}', cmd, log))

    print(f'[fit] {len(tasks)} fits (yrebin={YREBIN}, max_refit={max_refit}, jobs={jobs})')
    _run_tasks(tasks, jobs)


def stage_plot(energies, particles):
    """plot_v1.py for every tag -> per-tag dv1dy_coal yaml under plots/10ybin/."""
    import yaml as _yaml
    config = _yaml.safe_load(open(REPO / 'config.yaml'))
    # pT fits and piKp references are reused from the default (yrebin=1) results; they only
    # feed the v1_pt panels, not the v1(y) arrays this variant is about.
    def pt_paths(energy):
        return [REPO / 'result' / 'sys_tag_0' / f'pt_fit_{p}_{ew}_v1_{energy}.csv'
                for p in particles for ew in ('east', 'west')]

    def pikp_paths(energy):
        return [REPO / 'result' / 'v1_piKp' / energy / sp / 'result.csv'
                for sp in ('pions', 'kaons', 'protons')]

    for energy in energies:
        method = config['fit_order'][energy]
        ylo = config['plotting'][energy]['v2_lo']
        yhi = config['plotting'][energy]['v2_hi']

        def plot_one(plot_tag: str, fit_tag: str):
            """plot using fit CSVs of `fit_tag`, written under `plot_tag`'s dir."""
            (PLOTS_ROOT / _tag_dirname(plot_tag) / 'paper_yaml').mkdir(parents=True, exist_ok=True)
            (RESULT_ROOT / _tag_dirname(plot_tag)).mkdir(parents=True, exist_ok=True)
            paths = [fit_csv(fit_tag, p, energy) for p in particles]
            out_txt = RESULT_ROOT / _tag_dirname(plot_tag) / f'data_{energy}.txt'
            cmd = [PYTHON, PLOT_SCRIPT,
                   '--fres', fres_for_tag(plot_tag, energy),
                   '--paths', *paths,
                   '--paths_piKp', *pikp_paths(energy),
                   '--paths_pt', *pt_paths(energy),
                   '--energy', energy, '--method', method,
                   '--output', out_txt, '--sys_tag', plot_tag,
                   '--out_root', PLOTS_ROOT,
                   '--yrange', ylo, yhi]
            log = LOG_ROOT / f'plot_v1_{_tag_dirname(plot_tag)}_{energy}.log'
            print(f'[plot] {_tag_dirname(plot_tag)} {energy}')
            _run(cmd, log)

        for tag in REGULAR_PLOT_TAGS:
            plot_one(tag, tag)
        for tag in SPECIAL_HALF_TAGS:      # half-y fit range, from the tag-0 fits
            plot_one(tag, '0')
        plot_one('7', '7')                 # y-integrated eff


def stage_combine(energies):
    """combine_sys.py -> plots/10ybin/final/paper_yaml/dv1dy_coal_{energy}.yaml."""
    config = yaml.safe_load(open(REPO / 'config.yaml'))
    sys_divisor = config['sys_divisor']
    out_dir = PLOTS_ROOT / 'final' / 'paper_yaml'
    out_dir.mkdir(parents=True, exist_ok=True)
    for energy in energies:
        cmd = [PYTHON, COMBINE_SCRIPT,
               '--default', coal_yaml('0', energy),
               '--regular_sys', coal_yaml('1', energy), coal_yaml('2', energy), coal_yaml('3', energy),
               '--special_sys', coal_yaml('5', energy), coal_yaml('7', energy), coal_yaml('8', energy),
               '--output', out_dir / f'dv1dy_coal_{energy}.yaml',
               '--energy', energy, '--sys_divisor', sys_divisor]
        log = LOG_ROOT / f'combine_sys_{energy}.log'
        print(f'[combine_sys] {energy}')
        _run(cmd, log)


def stage_final_plot(energies):
    """Run the final v1(y) plotter on the combined 10ybin yamls."""
    cmd = [PYTHON, PLOT_Y_SCRIPT, '--energies', *energies]
    log = LOG_ROOT / 'plot_v1_y_10bin.log'
    print('[final plot] v1_y / netlambda / excesslambda')
    _run(cmd, log)


def _run_tasks(tasks, jobs):
    if jobs <= 1:
        for name, cmd, log in tasks:
            print(f'  {name}')
            _run(cmd, log)
        return
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run, cmd, log): name for name, cmd, log in tasks}
        for fut in as_completed(futs):
            name = futs[fut]
            fut.result()  # re-raise on failure
            print(f'  done: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    config = yaml.safe_load(open(REPO / 'config.yaml'))
    parser.add_argument('--energies', nargs='+', default=config['energies'])
    parser.add_argument('--particles', nargs='+', default=config['particles'])
    parser.add_argument('--max_refit', type=int, default=500)
    parser.add_argument('--jobs', type=int, default=1, help='parallel workers for the fit stage')
    parser.add_argument('--skip_fits', action='store_true', help='reuse existing 10ybin fits')
    parser.add_argument('--plot', action='store_true', help='also run the final v1(y) plotter')
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'10ybin variant -> result/10ybin , plots/10ybin   energies={args.energies}')

    if not args.skip_fits:
        stage_fit(args.energies, args.particles, args.max_refit, args.jobs)
    else:
        print('[fit] skipped (--skip_fits)')
    stage_plot(args.energies, args.particles)
    stage_combine(args.energies)
    if args.plot:
        stage_final_plot(args.energies)
    print('Done.')


if __name__ == '__main__':
    main()
