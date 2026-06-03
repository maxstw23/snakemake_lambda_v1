"""
Diagnostic (does NOT modify the pipeline): closure + parametric-bootstrap test of
the v1 extraction fit, focused on the soft_l1 vs plain-chi2 question (scan item §1).

Why this works without raw events
----------------------------------
Everything the v1 fit consumes is a per-bin quantity with a known uncertainty:
  - mass bins: counts (kept fixed here; the mass fit only sets the background fraction)
  - profile bins: <cos(phi-Psi)> with error sigma_bin from SimpleProfile.errors()
So we resample each profile bin as Normal(model_mean_bin, sigma_bin) and refit. The
spread of the recovered v1 IS a proper statistical error; injecting a known v1 lets us
measure bias and pull width (= reported_error / true_error coverage).

What it reports, per (cell, loss):
  bias        = mean(v1_fit) - v1_true                 -> is the point estimate biased?
  boot_err    = std(v1_fit)                            -> the *true* statistical error
  rep_err     = mean(reported m.errors['v2'])          -> what the pipeline quotes
  pull_width  = std((v1_fit - v1_true)/reported_err)   -> ~1 means errors are calibrated
                                                          >1 quoted error too big
                                                          <1 quoted error too small

It first validates the reimplemented fit core against the real PolyMinvFit.fit_v2.
"""
import argparse
import sys
import numpy as np
from iminuit import cost, Minuit

sys.path.insert(0, 'scripts')
from fit_v1 import PolyMinvFit  # noqa: E402


def build_fitter(path, particle, cen_bin, ybin):
    """Instantiate and run the (unchanged) invariant-mass fit for one cell."""
    rng_guesses = [5, 5, 1.12, 0.003, 0.002, 0.0, 1.0, 0.0]
    last = None
    for attempt in range(200):
        f = PolyMinvFit(path=path, par_str=particle, poly_deg=2,
                        guesses=list(rng_guesses), monitor=False,
                        signal='double gaussian', bg='poly',
                        cen=[cen_bin], ybin=[ybin], flow_case='v1')
        ok = f.fit_iminuit(masked=True)
        last = f
        if ok:
            return f
        # randomized restart, mirroring production refit strategy
        rng_guesses = [np.random.uniform(2, 8), np.random.uniform(2, 8),
                       np.random.uniform(1.11, 1.13),
                       np.random.uniform(0.002, 0.005), np.random.uniform(0.002, 0.005),
                       np.random.uniform(-5, 5), np.random.uniform(-5, 5),
                       np.random.uniform(-5, 5)]
    raise RuntimeError(f'mass fit never converged for cen={cen_bin} ybin={ybin}')


def fit_v2_core(fitter, v2_vals, loss, min_count=1, v2_start=0.0, p0_start=0.0,
                p1_start=0.0):
    """
    Faithful reimplementation of PolyMinvFit.fit_v2's *fitting* logic (no plotting),
    with the loss function exposed. Uses the fitter's frozen mass model (b_over_total,
    momega, sigma) and the supplied per-bin profile values v2_vals.
    Returns (v2, v2err, p0, p1) or (None, None, None, None) on failure.
    """
    b_over_total = fitter.func_return()
    x = fitter.x_vals_v2
    err = fitter.v2_errs
    counts = fitter.v2_counts

    def func(xx, v2, p0, p1):
        bx = b_over_total(xx)
        return bx * (p0 + p1 * (xx - fitter.hyperon_mass)) + (1 - bx) * v2

    c = cost.LeastSquares(x, v2_vals, err, func)
    # 'hybrid' = find the minimum robustly with soft_l1, then evaluate the error
    # with a plain-chi2 Hesse at that point.
    c.loss = 'soft_l1' if loss == 'hybrid' else loss
    m = Minuit(c, v2=v2_start, p0=p0_start, p1=p1_start)

    # stage 1: fix v2, fit background in sidebands
    m.fixed['v2'] = True
    c.mask = ((x < fitter.momega - 3 * fitter.sigma) |
              (fitter.momega + 3 * fitter.sigma < x)) & (counts > min_count)
    m.migrad()
    # stage 2: fix background, fit v2 over signal+sideband
    c.mask = None
    m.fixed['v2'] = False
    m.fixed['p0', 'p1'] = True
    c.mask = counts > min_count
    m.values['v2'] = v2_start
    m.migrad()
    m.fixed = None
    # final joint
    m.simplex()
    m.migrad()
    if loss == 'hybrid':
        # re-evaluate the error with a calibrated chi2 cost at the robust minimum
        c.loss = 'linear'
        m.migrad()
    m.hesse()
    if not m.valid or not m.accurate:
        return None, None, None, None
    return m.values['v2'], m.errors['v2'], m.values['p0'], m.values['p1']


def run_cell(path, particle, cen_bin, ybin, n_toys, seed=0):
    rng = np.random.default_rng(seed)
    fitter = build_fitter(path, particle, cen_bin, ybin)

    # observed profile and per-bin errors (the toy noise model)
    x = fitter.x_vals_v2
    err = fitter.v2_errs
    counts = fitter.v2_counts
    n_used = int(np.sum(counts > 1))
    mass_integral = float(np.sum(fitter.y_vals))

    # --- validation: reimplemented core vs the real fit_v2 on real data ---
    v2_real, v2err_real = fitter.fit_v2(masked=True, v2=0.0, a=0.0, b=0.0)
    v2_mine, v2err_mine, p0_obs, p1_obs = fit_v2_core(
        fitter, fitter.v2_vals, loss='soft_l1', v2_start=0.0)

    print(f'\n=== cell cen_bin={cen_bin} ybin={ybin}  '
          f'(mass integral={mass_integral:.0f}, profile bins used={n_used}) ===')
    if v2_real is None or v2_mine is None:
        print('  real or reimplemented fit failed on real data; skipping cell')
        return
    print(f'  validation (real fit_v2 vs reimpl, soft_l1): '
          f'v1 {v2_real:+.5f} vs {v2_mine:+.5f}   '
          f'err {v2err_real:.5f} vs {v2err_mine:.5f}')

    # truth model = observed best-fit background + injected v1 (use observed v1)
    def model_mean(v2_true):
        bx = fitter.func_return()(x)
        return bx * (p0_obs + p1_obs * (x - fitter.hyperon_mass)) + (1 - bx) * v2_true

    v2_true = v2_mine
    truth_curve = model_mean(v2_true)

    for loss in ('soft_l1', 'linear', 'hybrid'):
        fits, errs = [], []
        n_fail = 0
        for _ in range(n_toys):
            toy = rng.normal(truth_curve, err)  # resample each profile bin
            v2f, v2e, _, _ = fit_v2_core(fitter, toy, loss=loss, v2_start=0.0,
                                         p0_start=p0_obs, p1_start=p1_obs)
            if v2f is None or v2e <= 0:
                n_fail += 1
                continue
            fits.append(v2f)
            errs.append(v2e)
        fits = np.array(fits)
        errs = np.array(errs)
        bias = fits.mean() - v2_true
        boot_err = fits.std(ddof=1)
        rep_err = errs.mean()
        pull = (fits - v2_true) / errs
        print(f'  [{loss:8s}] n_ok={len(fits):4d} n_fail={n_fail:4d}  '
              f'v1_true={v2_true:+.5f}  bias={bias:+.5f} ({bias/boot_err:+.2f} boot_sig)  '
              f'boot_err={boot_err:.5f}  rep_err={rep_err:.5f}  '
              f'rep/boot={rep_err/boot_err:.2f}  pull_width={pull.std(ddof=1):.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='result/sys_tag_0/combined_Lambdabar_v1_7p7GeV.root')
    ap.add_argument('--particle', default='Lambdabar')
    ap.add_argument('--ntoys', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    # (cen_bin, ybin) pairs: cen_bin is 1-indexed (1 = 70-80%); ybin 0..19
    ap.add_argument('--cells', default='1:19,1:0,7:2',
                    help='comma-list of cen_bin:ybin (low-count first)')
    args = ap.parse_args()
    for cell in args.cells.split(','):
        cb, yb = (int(v) for v in cell.split(':'))
        try:
            run_cell(args.path, args.particle, cb, yb, args.ntoys, seed=args.seed)
        except Exception as e:
            print(f'\ncell {cell} errored: {e}')
