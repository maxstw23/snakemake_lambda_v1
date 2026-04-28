#!/usr/bin/env python3
"""
Bayesian NCQ quark v1(y) extraction — linear mixture + EM model.

Simultaneously fits π±/K±/p/p̄/Λ/Λ̄ v1(y) to extract latent quark v1(y)
slopes under a hydro-mixture + EM-shift decomposition:

  c1_u  = f c1_tr + (1−f) c1_prod   + (+2/3) δ_light
  c1_d  = f c1_tr + (1−f) c1_prod   + (−1/3) δ_light
  c1_ū  =          c1_prod          + (−2/3) δ_light
  c1_d̄  =          c1_prod          + (+1/3) δ_light
  c1_s  =          c1_s                                 # independent
  c1_s̄  =          c1_sbar                              # independent

Light quarks share a charge-proportional EM term δ_light. Strange and
antistrange use independent slopes (no CP linkage) — the data show
that c1_s and c1_s̄ are pulled in different directions by mesons (K±)
vs. baryons (Λ, Λ̄), and a single charge-proportional δ_strange cannot
reconcile both simultaneously.

All quark v1(y) are pure linear (no cubic), matching the project's
`fit_order: 1` convention. Strict |y| ≤ y_max cut applied to all data.

Parameters (6):
  c1_tr, c1_prod, c1_s, c1_sbar, f, δ_light

NCQ coalescence at common rapidity:
  π+ = c1_u  + c1_d̄        # ud̄
  π− = c1_ū  + c1_d         # ūd
  K+ = c1_u  + c1_s̄         # us̄
  K− = c1_ū  + c1_s         # ūs
  p  = 2 c1_u + c1_d        # uud
  p̄  = 2 c1_ū + c1_d̄        # ūūd̄
  Λ  = c1_u  + c1_d + c1_s  # uds
  Λ̄  = c1_ū  + c1_d̄ + c1_s̄ # ūd̄s̄

Notable derived asymmetries:
  Δπ(π+−π−) = 2 δ_light
  Δp(p−p̄)   = 3 f (c1_tr − c1_prod) + 2 δ_light
  ΔK(K+−K−) = f (c1_tr − c1_prod) + (4/3) δ_light + (c1_sbar − c1_s)
  ΔΛ(Λ−Λ̄)  = 2 f (c1_tr − c1_prod) + (2/3) δ_light + (c1_s − c1_sbar)

Likelihood: Gaussian, σ = stat ⊕ sys in quadrature (per bin).

Usage:
  conda activate lambda_v1
  python scripts/quark_v1_bayes.py --energy 19p6GeV --centrality 1040
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    energy            = '19p6GeV',
    centrality        = '1040',
    pikp_data_dir     = 'Etabins_20_Coalescence/Etabins_20_Coalescence_ymp6top6',
    lambda_yaml_dir   = 'plots/final/paper_yaml',
    y_max             = 0.6,
    f_prior_sigma     = 0.5,
    delta_prior_sigma = 0.005,
    thermal_csv       = 'data/chemical_freezeout_GCE_1701.07065.csv',
    err_floor         = 1e-5,
    draws             = 2000,
    tune              = 1000,
    chains            = 4,
    target_accept     = 0.95,
    max_treedepth     = 12,
    random_seed       = 42,
)

ENERGY_TO_SQRTS = {
    '7p7GeV': 7.7, '9p2GeV': 9.2, '11p5GeV': 11.5, '14p6GeV': 14.6,
    '17p3GeV': 17.3, '19p6GeV': 19.6, '27GeV': 27.0,
}

# pi/K/p ROOT filenames use shorthand (truncated) energy labels.
ENERGY_TO_PIKP_TAG = {
    '7p7GeV': '7', '9p2GeV': '9', '11p5GeV': '11', '14p6GeV': '14',
    '17p3GeV': '17', '19p6GeV': '19', '27GeV': '27',
}

CENTRALITY_INFO = {
    '010':  dict(pikp='_89', thermal=['00-05', '05-10'],         label='0-10%'),
    '1040': dict(pikp='_57', thermal=['10-20', '20-30', '30-40'], label='10-40%'),
    '4080': dict(pikp='_14', thermal=['40-60', '60-80'],          label='40-80%'),
    '5080': dict(pikp='_13', thermal=['60-80'],                   label='50-80%'),
}

# species → (charge_tag, root_species_filename)
PIKP_SPECIES_INFO = {
    'piplus':     ('Selp', 'pions'),
    'piminus':    ('Seln', 'pions'),
    'kplus':      ('Selp', 'kaons'),
    'kminus':     ('Seln', 'kaons'),
    'proton':     ('Selp', 'protons'),
    'antiproton': ('Seln', 'protons'),
}

LAMBDA_SPECIES = ['Lambda', 'Lambdabar']
PIKP_SPECIES   = list(PIKP_SPECIES_INFO.keys())
ALL_SPECIES    = PIKP_SPECIES + LAMBDA_SPECIES

# Electric charges (in units of e). Antiquarks have opposite signs.
QUARK_CHARGE = {
    'u': +2/3,  'd': -1/3,  's': -1/3,
    'ubar': -2/3, 'dbar': +1/3, 'sbar': +1/3,
}
QUARK_FLAVORS = ['u', 'd', 'ubar', 'dbar', 's', 'sbar']

LABELS = {
    'piplus':     r'$\pi^+$',
    'piminus':    r'$\pi^-$',
    'kplus':      r'$K^+$',
    'kminus':     r'$K^-$',
    'proton':     r'$p$',
    'antiproton': r'$\bar{p}$',
    'Lambda':     r'$\Lambda$',
    'Lambdabar':  r'$\bar{\Lambda}$',
}
COLORS = {
    'piplus': 'steelblue', 'piminus': 'darkorange',
    'kplus':  'forestgreen', 'kminus':  'firebrick',
    'proton': 'purple',     'antiproton': 'brown',
    'Lambda': 'navy',       'Lambdabar':  'crimson',
}

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Bayesian NCQ quark v1(y) — linear mixture model')
    p.add_argument('--energy',          default=DEFAULTS['energy'])
    p.add_argument('--centrality',      default=DEFAULTS['centrality'],
                   choices=list(CENTRALITY_INFO.keys()))
    p.add_argument('--pikp_data_dir',   default=DEFAULTS['pikp_data_dir'])
    p.add_argument('--lambda_yaml_dir', default=DEFAULTS['lambda_yaml_dir'])
    p.add_argument('--out_dir',         default=None)
    p.add_argument('--y_max',           type=float, default=DEFAULTS['y_max'])
    p.add_argument('--f_prior_sigma',     type=float, default=DEFAULTS['f_prior_sigma'])
    p.add_argument('--delta_prior_sigma', type=float, default=DEFAULTS['delta_prior_sigma'])
    p.add_argument('--thermal_csv',       default=DEFAULTS['thermal_csv'])
    p.add_argument('--err_floor',       type=float, default=DEFAULTS['err_floor'])
    p.add_argument('--draws',           type=int,   default=DEFAULTS['draws'])
    p.add_argument('--tune',            type=int,   default=DEFAULTS['tune'])
    p.add_argument('--chains',          type=int,   default=DEFAULTS['chains'])
    p.add_argument('--target_accept',   type=float, default=DEFAULTS['target_accept'])
    p.add_argument('--max_treedepth',   type=int,   default=DEFAULTS['max_treedepth'])
    p.add_argument('--random_seed',     type=int,   default=DEFAULTS['random_seed'])
    return p.parse_args()

# ── Thermal model prior ────────────────────────────────────────────────────────

def load_thermal_prior(energy, thermal_csv, cen_labels):
    """Compute f_prior = tanh(μ_B / 3T_ch) from GCE thermal model."""
    sqrts = ENERGY_TO_SQRTS.get(energy)
    if sqrts is None:
        print(f'  [thermal prior] Energy {energy} not in lookup; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    try:
        df = pd.read_csv(thermal_csv)
    except FileNotFoundError:
        print(f'  [thermal prior] {thermal_csv} not found; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    rows = df[(df['sqrts_GeV'] == sqrts) & (df['centrality'].isin(cen_labels))]
    if len(rows) == 0:
        print(f'  [thermal prior] No rows for {energy} / {cen_labels}; using f_prior=0.5')
        return 0.5, np.nan, np.nan

    if 'interpolated' in rows.columns and rows['interpolated'].any():
        print(f'  [thermal prior] WARNING: T_ch and μ_B for {energy} are INTERPOLATED.')

    T_ch = rows['Tch_GCEY_MeV'].mean()
    muB  = rows['muB_GCEY_MeV'].mean()
    f    = np.tanh(muB / (3.0 * T_ch))
    return float(f), float(T_ch), float(muB)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_pikp_v1y(pikp_data_dir, energy, centrality):
    """
    Returns dict species → (y, v1, err_total) for π+, π−, K+, K−, p, p̄.
    err_total = sqrt(stat² + sys²); empty bins (zero stat err) are dropped.
    """
    data_dir = Path(pikp_data_dir)
    pikp_tag = ENERGY_TO_PIKP_TAG[energy]
    cen_tag  = CENTRALITY_INFO[centrality]['pikp']

    out = {}
    for sp, (charge, root_sp) in PIKP_SPECIES_INFO.items():
        path = data_dir / f'{pikp_tag}GeV_{root_sp}_withsystematics.root'
        with uproot.open(path) as f:
            base   = f'Flow_vEta_f_{charge}{cen_tag}_rebinned'
            v1     = f[base].values()
            stat   = f[base].errors()
            sys_   = f[f'{base}_systematics'].values()
            yc     = f[f'{base}_bincenters'].values()

        keep = stat > 0
        out[sp] = (yc[keep], v1[keep],
                   np.sqrt(stat[keep] ** 2 + sys_[keep] ** 2))
    return out


def load_lambda_v1y(lambda_yaml_dir, energy, centrality):
    """Returns dict species → (y, v1, err_total) for Lambda, Lambdabar."""
    path = Path(lambda_yaml_dir) / f'dv1dy_coal_{energy}.yaml'
    with open(path) as f:
        d = yaml.unsafe_load(f)

    out = {}
    for sp, key_prefix in [('Lambda', 'v1_y_lambda'), ('Lambdabar', 'v1_y_lambdabar')]:
        rec  = d[f'{key_prefix}_{centrality}']
        y    = np.asarray(rec['y'],          dtype=float)
        v1   = np.asarray(rec['value'],      dtype=float)
        stat = np.asarray(rec['error_stat'], dtype=float)
        sys_ = np.asarray(rec['error_sys'],  dtype=float)
        out[sp] = (y, v1, np.sqrt(stat ** 2 + sys_ ** 2))
    return out


def apply_y_cut(data_dict, y_max):
    """Apply |y| ≤ y_max to all species datasets."""
    return {sp: (y[m := np.abs(y) <= y_max], v1[m], err[m])
            for sp, (y, v1, err) in data_dict.items()}


def print_dataset_summary(data):
    print('\nDataset summary:')
    for sp, (y, v1, err) in data.items():
        if len(y) == 0:
            print(f'  {sp:12s}: 0 bins (empty after cuts)')
            continue
        print(f'  {sp:12s}: {len(y):2d} bins, y∈[{y.min():+.3f},{y.max():+.3f}], '
              f'v1∈[{v1.min():+.4f},{v1.max():+.4f}]')

# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(data, cfg):
    y_obs = {sp: d[0] for sp, d in data.items()}
    v1    = {sp: d[1] for sp, d in data.items()}
    err   = {sp: np.maximum(d[2], cfg['err_floor']) for sp, d in data.items()}

    f_prior       = cfg['f_prior']
    f_sigma       = cfg['f_prior_sigma']
    delta_sigma   = cfg['delta_prior_sigma']
    f_logit_mu    = float(np.log(f_prior / (1.0 - f_prior)))

    with pm.Model() as model:
        c1_tr   = pm.Normal('c1_tr',   mu=0.0, sigma=0.05)
        c1_prod = pm.Normal('c1_prod', mu=0.0, sigma=0.05)
        c1_s    = pm.Normal('c1_s',    mu=0.0, sigma=0.05)
        c1_sbar = pm.Normal('c1_sbar', mu=0.0, sigma=0.05)

        f_logit     = pm.Normal('f_logit', mu=f_logit_mu, sigma=f_sigma)
        f           = pm.Deterministic('f', pt.sigmoid(f_logit))

        delta_light = pm.Normal('delta_light', mu=0.0, sigma=delta_sigma)

        # Per-quark slope (PyTensor scalar): hydro + (charge·δ_light only for u/d/ū/d̄).
        # Strange and antistrange are independent — no CP linkage.
        def quark_slope(flavor):
            if flavor == 'u':
                return f * c1_tr + (1.0 - f) * c1_prod + QUARK_CHARGE['u'] * delta_light
            if flavor == 'd':
                return f * c1_tr + (1.0 - f) * c1_prod + QUARK_CHARGE['d'] * delta_light
            if flavor == 'ubar':
                return c1_prod + QUARK_CHARGE['ubar'] * delta_light
            if flavor == 'dbar':
                return c1_prod + QUARK_CHARGE['dbar'] * delta_light
            if flavor == 's':
                return c1_s
            if flavor == 'sbar':
                return c1_sbar
            raise ValueError(f'Unknown flavor: {flavor}')

        # NCQ-summed hadron mean v1 at observed y array.
        HADRON_QUARK_CONTENT = {
            'piplus':     [('u', 1), ('dbar', 1)],
            'piminus':    [('ubar', 1), ('d', 1)],
            'kplus':      [('u', 1), ('sbar', 1)],
            'kminus':     [('ubar', 1), ('s', 1)],
            'proton':     [('u', 2), ('d', 1)],
            'antiproton': [('ubar', 2), ('dbar', 1)],
            'Lambda':     [('u', 1), ('d', 1), ('s', 1)],
            'Lambdabar':  [('ubar', 1), ('dbar', 1), ('sbar', 1)],
        }

        def hadron(sp):
            y = y_obs[sp]
            slope = sum(n * quark_slope(fl) for fl, n in HADRON_QUARK_CONTENT[sp])
            return slope * y

        for sp in ALL_SPECIES:
            if len(y_obs[sp]) == 0:
                continue
            pm.Normal(f'obs_{sp}', mu=hadron(sp), sigma=err[sp], observed=v1[sp])

    return model

# ── Posterior helpers ──────────────────────────────────────────────────────────

def _flat(post, name):
    return post[name].values.flatten()


def quark_slope_posterior(trace, flavor):
    """Returns (S,) array of c1 (slope) posterior samples for any of the 6 flavors."""
    post = trace.posterior
    if flavor == 's':
        return _flat(post, 'c1_s')
    if flavor == 'sbar':
        return _flat(post, 'c1_sbar')
    if flavor in ('u', 'd'):
        hydro = _flat(post, 'f') * _flat(post, 'c1_tr') + \
                (1.0 - _flat(post, 'f')) * _flat(post, 'c1_prod')
    elif flavor in ('ubar', 'dbar'):
        hydro = _flat(post, 'c1_prod')
    else:
        raise ValueError(f'Unknown flavor: {flavor}')
    return hydro + QUARK_CHARGE[flavor] * _flat(post, 'delta_light')


def eval_quark_v1_posterior(trace, flavor, y_grid):
    """Vectorized over posterior samples: returns (S, N) curves of c1·y."""
    return quark_slope_posterior(trace, flavor)[:, None] * y_grid[None, :]


# Same content map as in build_model — keep in sync.
HADRON_QUARK_CONTENT = {
    'piplus':     [('u', 1), ('dbar', 1)],
    'piminus':    [('ubar', 1), ('d', 1)],
    'kplus':      [('u', 1), ('sbar', 1)],
    'kminus':     [('ubar', 1), ('s', 1)],
    'proton':     [('u', 2), ('d', 1)],
    'antiproton': [('ubar', 2), ('dbar', 1)],
    'Lambda':     [('u', 1), ('d', 1), ('s', 1)],
    'Lambdabar':  [('ubar', 1), ('dbar', 1), ('sbar', 1)],
}


def hadron_slope_posterior(trace, sp):
    return sum(n * quark_slope_posterior(trace, fl) for fl, n in HADRON_QUARK_CONTENT[sp])


def hadron_pred_posterior(trace, sp, y_arr):
    return hadron_slope_posterior(trace, sp)[:, None] * y_arr[None, :]


def pct_band(curves, ax, x, color, zorder=1, alpha_fill=0.25):
    med  = np.median(curves, axis=0)
    lo68 = np.percentile(curves, 16,   axis=0)
    hi68 = np.percentile(curves, 84,   axis=0)
    lo95 = np.percentile(curves, 2.5,  axis=0)
    hi95 = np.percentile(curves, 97.5, axis=0)
    ax.fill_between(x, lo95, hi95, alpha=alpha_fill * 0.6, color=color, zorder=zorder)
    ax.fill_between(x, lo68, hi68, alpha=alpha_fill,       color=color, zorder=zorder)
    ax.plot(x, med, color=color, lw=2, zorder=zorder + 1)
    return med

# ── Plots ──────────────────────────────────────────────────────────────────────

FLAVOR_INFO = [
    ('u',    r'$v_1^u$',         'steelblue'),
    ('d',    r'$v_1^d$',         'darkorange'),
    ('s',    r'$v_1^s$',         'firebrick'),
    ('ubar', r'$v_1^{\bar{u}}$', 'lightblue'),
    ('dbar', r'$v_1^{\bar{d}}$', 'navajowhite'),
    ('sbar', r'$v_1^{\bar{s}}$', 'lightcoral'),
]


def energy_label(energy):
    return energy.replace('p', '.').replace('GeV', ' GeV')


def plot_quark_v1_functions(trace, out_dir, energy, cen_label, y_max):
    y_grid = np.linspace(-y_max, y_max, 200)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    axes = axes.flatten()
    for ax, (fl, lbl, col) in zip(axes, FLAVOR_INFO):
        curves = eval_quark_v1_posterior(trace, fl, y_grid)
        pct_band(curves, ax, y_grid, col)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(r'$y$', fontsize=12)
        ax.set_title(lbl, fontsize=13)
        ax.set_xlim(-y_max, y_max)
    for ax in axes[::3]:
        ax.set_ylabel(r'$v_1^{\rm quark}$', fontsize=12)
    fig.suptitle(f'Quark $v_1$ — Au+Au $\\sqrt{{s_{{NN}}}}$={energy_label(energy)}, {cen_label}',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / 'quark_v1_functions.pdf')
    plt.close(fig)
    print('  Saved quark_v1_functions.pdf')


def plot_em_signal(trace, out_dir, energy, cen_label, cfg):
    """Posteriors of δ_light, c1_s, c1_sbar, and (c1_s − c1_sbar)."""
    post   = trace.posterior
    d_l    = _flat(post, 'delta_light')
    cs     = _flat(post, 'c1_s')
    csb    = _flat(post, 'c1_sbar')
    s_diff = cs - csb

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axes[0,0], d_l,    r'$\delta_{\rm light}$',           'steelblue', cfg['delta_prior_sigma']),
        (axes[0,1], cs,     r'$c_1^{s}$',                       'firebrick', None),
        (axes[1,0], csb,    r'$c_1^{\bar{s}}$',                 'darkorange', None),
        (axes[1,1], s_diff, r'$c_1^{s} - c_1^{\bar{s}}$',       'purple', None),
    ]
    for ax, samples, title, color, prior_sigma in panels:
        ax.hist(samples, bins=60, density=True, color=color, alpha=0.6)
        ax.axvline(0, color='k', ls='--', lw=1)
        med  = np.median(samples)
        lo68 = np.percentile(samples, 16)
        hi68 = np.percentile(samples, 84)
        ax.axvline(med, color='k', lw=1.5)
        ax.set_xlabel(title + f'  (median {med:+.4f}, 68% [{lo68:+.4f}, {hi68:+.4f}])',
                      fontsize=10)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        if prior_sigma is not None:
            ax.axvspan(-prior_sigma, prior_sigma, alpha=0.10, color='grey',
                       label=f'prior 1σ (±{prior_sigma:g})')
            ax.legend(fontsize=9)

    fig.suptitle(
        f'EM signal & strange asymmetry — Au+Au $\\sqrt{{s_{{NN}}}}$={energy_label(energy)}, {cen_label}',
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / 'em_signal.pdf')
    plt.close(fig)
    print('  Saved em_signal.pdf')


def plot_transported_signal(trace, out_dir, energy, cen_label, cfg, y_max):
    """1×3: v1^tr (light hydro) vs v1^prod (light hydro) | f·(tr−prod) | posterior of f.

    Both curves here are EM-free (just the linear hydro slope × y).
    """
    y_grid    = np.linspace(-y_max, y_max, 200)
    post      = trace.posterior
    f_samples = _flat(post, 'f')
    c1_tr_s   = _flat(post, 'c1_tr')
    c1_prod_s = _flat(post, 'c1_prod')
    prod      = c1_prod_s[:, None] * y_grid[None, :]
    tr        = c1_tr_s  [:, None] * y_grid[None, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax0 = axes[0]
    pct_band(prod, ax0, y_grid, 'forestgreen', alpha_fill=0.3)
    pct_band(tr,   ax0, y_grid, 'crimson',     alpha_fill=0.3)
    ax0.plot([], [], color='forestgreen', lw=2, label=r'$v_1^{\rm prod}$ (light)')
    ax0.plot([], [], color='crimson',     lw=2, label=r'$v_1^{\rm tr}$')
    ax0.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax0.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax0.set_xlabel(r'$y$', fontsize=11)
    ax0.set_ylabel(r'$v_1^{\rm quark}$', fontsize=11)
    ax0.set_title(r'$v_1^{\rm tr}$ vs $v_1^{\rm prod}$', fontsize=11)
    ax0.legend(fontsize=10)
    ax0.set_xlim(-y_max, y_max)

    ax1 = axes[1]
    delta = f_samples[:, None] * (tr - prod)
    pct_band(delta, ax1, y_grid, 'steelblue')
    ax1.axhline(0, color='k', lw=1, ls='--')
    ax1.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax1.set_xlabel(r'$y$', fontsize=11)
    ax1.set_ylabel(r'$f \cdot (v_1^{\rm tr} - v_1^{\rm prod})$', fontsize=11)
    ax1.set_title('Transported contribution', fontsize=11)
    ax1.set_xlim(-y_max, y_max)

    ax2 = axes[2]
    ax2.hist(f_samples, bins=50, density=True, color='steelblue', alpha=0.6,
             label='Posterior $f$')
    ax2.axvline(cfg['f_prior'], color='k', ls='--', lw=1.5,
                label=f'Prior mean = {cfg["f_prior"]:.2f}')
    ax2.set_xlabel(r'$f$', fontsize=12)
    ax2.set_ylabel('Density',   fontsize=11)
    ax2.set_title('Posterior of $f$', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=10)

    fig.suptitle(f'Transported quark signal — Au+Au $\\sqrt{{s_{{NN}}}}$={energy_label(energy)}, {cen_label}',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / 'transported_signal.pdf')
    plt.close(fig)
    print('  Saved transported_signal.pdf')


def plot_posterior_predictive(trace, data, out_dir, energy, cen_label, y_max):
    y_fine = np.linspace(-y_max, y_max, 200)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for ax, sp in zip(axes, ALL_SPECIES):
        y_d, v1_d, err_d = data[sp]
        col = COLORS[sp]
        if len(y_d) > 0:
            curves = hadron_pred_posterior(trace, sp, y_fine)
            pct_band(curves, ax, y_fine, col)
            ax.errorbar(y_d, v1_d, yerr=err_d, fmt='o', color='k',
                        ms=4, capsize=2, lw=1, label='Data', zorder=10)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(r'$y$', fontsize=11)
        ax.set_ylabel(r'$v_1$', fontsize=11)
        ax.set_title(LABELS[sp], fontsize=13)
        ax.set_xlim(-y_max, y_max)
        ax.legend(fontsize=9)
    fig.suptitle(f'Posterior predictive — Au+Au $\\sqrt{{s_{{NN}}}}$={energy_label(energy)}, {cen_label}',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / 'posterior_predictive.pdf')
    plt.close(fig)
    print('  Saved posterior_predictive.pdf')


def plot_quark_comparison(trace, out_dir, energy, cen_label, y_max):
    y_grid = np.linspace(-y_max, y_max, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fl, lbl, col in FLAVOR_INFO:
        curves = eval_quark_v1_posterior(trace, fl, y_grid)
        med  = np.median(curves, 0)
        lo68 = np.percentile(curves, 16, 0)
        hi68 = np.percentile(curves, 84, 0)
        ls = '--' if fl.endswith('bar') else '-'
        ax.fill_between(y_grid, lo68, hi68, alpha=0.18, color=col)
        ax.plot(y_grid, med, color=col, lw=2, ls=ls, label=lbl)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(r'$y$', fontsize=13)
    ax.set_ylabel(r'$v_1^{\rm quark}$', fontsize=13)
    ax.set_xlim(-y_max, y_max)
    ax.legend(fontsize=11, ncol=2)
    ax.set_title(f'Au+Au $\\sqrt{{s_{{NN}}}}$={energy_label(energy)}, {cen_label}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'quark_v1_comparison.pdf')
    plt.close(fig)
    print('  Saved quark_v1_comparison.pdf')

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    energy     = args.energy
    centrality = args.centrality
    cen_info   = CENTRALITY_INFO[centrality]
    cen_label  = cen_info['label']

    out_dir = Path(args.out_dir or f'plots/{energy}/quark_v1/{centrality}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Quark v1 mixture model: {energy}, centrality {cen_label} ===')

    print('\n=== Thermal model prior ===')
    f_prior, T_ch, muB = load_thermal_prior(
        energy, args.thermal_csv, cen_info['thermal'])
    print(f'  T_ch = {T_ch:.1f} MeV,  μ_B = {muB:.1f} MeV')
    print(f'  f_prior = tanh(μ_B/3T) = {f_prior:.3f}')
    print(f'  logit(f_prior) = {np.log(f_prior / (1 - f_prior)):.3f},  σ_logit = {args.f_prior_sigma}')

    cfg = dict(
        f_prior           = f_prior,
        f_prior_sigma     = args.f_prior_sigma,
        delta_prior_sigma = args.delta_prior_sigma,
        err_floor         = args.err_floor,
    )

    print('\n=== Loading π/K/p data ===')
    pikp_data   = load_pikp_v1y(args.pikp_data_dir, energy, centrality)
    print('=== Loading Λ/Λ̄ data ===')
    lambda_data = load_lambda_v1y(args.lambda_yaml_dir, energy, centrality)
    data = {**pikp_data, **lambda_data}

    print(f'\n=== Applying |y| ≤ {args.y_max} cut ===')
    data = apply_y_cut(data, args.y_max)
    print_dataset_summary(data)

    print('\n=== Building model ===')
    model = build_model(data, cfg)

    print('\n=== Sampling ===')
    with model:
        trace = pm.sample(
            draws=args.draws, tune=args.tune,
            target_accept=args.target_accept, chains=args.chains,
            max_treedepth=args.max_treedepth,
            random_seed=args.random_seed, progressbar=True,
        )

    print('\n=== Convergence diagnostics ===')
    summary = az.summary(trace, var_names=[
        'c1_tr', 'c1_prod', 'c1_s', 'c1_sbar', 'f', 'delta_light'])
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']].to_string())
    max_rhat = summary['r_hat'].max()
    min_ess  = summary['ess_bulk'].min()
    n_div    = int(trace.sample_stats['diverging'].values.sum())
    print(f'\nMax R-hat:   {max_rhat:.4f}  (target <1.01)')
    print(f'Min ESS:     {min_ess:.0f}   (target >400)')
    print(f'Divergences: {n_div}  (target 0)')

    print('\n=== Saving trace ===')
    trace_path = out_dir / 'trace.nc'
    az.to_netcdf(trace, trace_path)
    print(f'  Saved {trace_path}')

    print('\n=== Generating plots ===')
    plot_quark_v1_functions  (trace, out_dir, energy, cen_label, args.y_max)
    plot_transported_signal  (trace, out_dir, energy, cen_label, cfg, args.y_max)
    plot_em_signal           (trace, out_dir, energy, cen_label, cfg)
    plot_posterior_predictive(trace, data, out_dir, energy, cen_label, args.y_max)
    plot_quark_comparison    (trace, out_dir, energy, cen_label, args.y_max)

    print(f'\nDone. Output in {out_dir}/')


if __name__ == '__main__':
    main()
