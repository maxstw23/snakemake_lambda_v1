"""Diagnostic: decompose chi2/ndf to separate residual vs uncertainty-inflation effects."""
import sys, yaml, numpy as np
import pandas as pd, uproot

sys.path.insert(0, 'scripts')
from pikp_merged import PikpMergedSlope
from iminuit import cost, Minuit
from uncertainties import unumpy

ENERGIES = ['7p7GeV','9p2GeV','11p5GeV','14p6GeV','17p3GeV','19p6GeV','27GeV']
ENERGY_FLOAT = {'7p7GeV':7.7,'9p2GeV':9.2,'11p5GeV':11.5,
                '14p6GeV':14.6,'17p3GeV':17.3,'19p6GeV':19.6,'27GeV':27.0}
ENERGY_PIKP = {e: (str(int(ENERGY_FLOAT[e])) + 'GeV'
                   if ENERGY_FLOAT[e] == int(ENERGY_FLOAT[e])
                   else str(ENERGY_FLOAT[e]) + 'GeV')
               for e in ENERGIES}
CONFIG = yaml.load(open('config.yaml'), Loader=yaml.CLoader)
Y_CUT = float(CONFIG['y_cut'])
MERGED_CENS = {'010': [8,9], '1040': [5,6,7], '4080': [1,2,3,4]}
DATA_ROOTS = {
    '7p7GeV':  'data/result4_7p7GeV.root',
    '9p2GeV':  'data/result5_9p2GeV.root',
    '11p5GeV': 'data/result7_11p5GeV.root',
    '14p6GeV': 'data/result4_14p6GeV.root',
    '17p3GeV': 'data/result6_17p3GeV.root',
    '19p6GeV': 'data/result7_default_19p6GeV.root',
    '27GeV':   'data/result10_27GeV.root',
}


def fit_dv1dy(y_u, v1_u):
    mask = np.abs(unumpy.nominal_values(y_u)) < Y_CUT
    if mask.sum() < 2:
        return 0., 0.
    c = cost.LeastSquares(
        unumpy.nominal_values(y_u[mask]), unumpy.nominal_values(v1_u[mask]),
        unumpy.std_devs(v1_u[mask]), lambda x, a: a * x)
    m = Minuit(c, a=0)
    m.migrad()
    if not m.valid:
        return 0., 0.
    return float(m.values['a']), float(m.errors['a'])


def load_csv(path):
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    return {int(cen): {s: df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
            for cen in df.columns.levels[0]}


def per_bin_delta(lam_csv, lbar_csv, fres_path):
    fres = uproot.open(fres_path)
    hres = unumpy.uarray(fres['hEPDEP_ew_cos_1'].values(), fres['hEPDEP_ew_cos_1'].errors())
    cm = unumpy.nominal_values(hres) > 0
    res = np.where(cm, np.abs(hres) ** 0.5, 1.)
    ld = load_csv(lam_csv)
    lb = load_csv(lbar_csv)
    dlt = np.zeros(9)
    derr = np.full(9, np.inf)
    for cen in range(1, 10):
        if not cm[cen - 1]:
            continue
        r = float(unumpy.nominal_values(res[cen - 1]))
        sl, er = [], []
        for d in [ld, lb]:
            if cen not in d:
                sl.append(0.); er.append(np.inf); continue
            dd = d[cen]
            n = len(dd['values'])
            ed = np.linspace(-1., 1., n + 1)
            yc = .5 * (ed[:-1] + ed[1:])
            ye = np.diff(ed) / 2
            gd = ~np.isnan(dd['values'].astype(float))
            v1 = unumpy.uarray(dd['values'][gd], dd['errors'][gd]) / r
            y = unumpy.uarray(yc[gd], ye[gd])
            s, se = fit_dv1dy(y, v1)
            sl.append(s); er.append(se if se > 0 else np.inf)
        dlt[cen - 1] = sl[0] - sl[1]
        derr[cen - 1] = np.sqrt(er[0] ** 2 + er[1] ** 2)
    return dlt, derr, cm


def merge(dlt, derr, cm):
    result = {}
    for k, bins in MERGED_CENS.items():
        idx = [b - 1 for b in bins if cm[b - 1] and derr[b - 1] < 1e6]
        if not idx:
            result[k] = (np.nan, np.nan); continue
        w = 1. / derr[idx] ** 2
        v = np.sum(w * dlt[idx]) / np.sum(w)
        e = np.sqrt(1. / np.sum(w))
        result[k] = (v, e)
    return result


# Load eff-corrected DeltaLambda from YAMLs
eff_d = {c: {'v': [], 'e': []} for c in MERGED_CENS}
for e in ENERGIES:
    with open(f'plots/final/paper_yaml/dv1dy_coal_{e}.yaml') as fh:
        yd = yaml.load(fh, yaml.CLoader)
    for c in MERGED_CENS:
        field = f'dv1dy_deltalambda_{c}'
        eff_d[c]['v'].append(yd[field]['value'])
        eff_d[c]['e'].append(yd[field]['error_stat'])

# Load no-eff DeltaLambda from CSVs
noe_d = {c: {'v': [], 'e': []} for c in MERGED_CENS}
for e in ENERGIES:
    try:
        d, de, cm = per_bin_delta(
            f'result/no_eff/fit_Lambda_v1_{e}.csv',
            f'result/no_eff/fit_Lambdabar_v1_{e}.csv',
            DATA_ROOTS[e])
        mg = merge(d, de, cm)
        for c in MERGED_CENS:
            noe_d[c]['v'].append(mg[c][0]); noe_d[c]['e'].append(mg[c][1])
    except Exception as ex:
        sys.stderr.write(f'Warning {e}: {ex}\n')
        for c in MERGED_CENS:
            noe_d[c]['v'].append(np.nan); noe_d[c]['e'].append(np.nan)

pikp = PikpMergedSlope().get_data()

# -------------------------------------------------------------------------
print('\n======= PART 1: Dp vs (Dp-DK) decomposition (eff-corrected DeltaLambda) =======\n')
print(f'{"cent":5s}  {"chi2 vs Dp":>12s}  {"chi2 vs Dp-DK":>14s}  {"Dp-DK fixed-denom":>18s}  '
      f'{"RMS(DL-Dp)":>12s}  {"RMS(DL-Dp+DK)":>14s}  {"DK/Dp var%":>10s}')
for cent in ['010', '1040', '4080']:
    dl = np.array(eff_d[cent]['v'])
    sl = np.array(eff_d[cent]['e'])
    dp_v, dp_s, dk_v, dk_s = [], [], [], []
    for e in ENERGIES:
        pe = ENERGY_PIKP[e]
        dp_v.append(pikp[pe]['protons'][f'{cent}_linear']['delta'])
        dp_s.append(pikp[pe]['protons'][f'{cent}_linear']['delta_err'])
        dk_v.append(pikp[pe]['kaons'][f'{cent}_linear']['delta'])
        dk_s.append(pikp[pe]['kaons'][f'{cent}_linear']['delta_err'])
    dp_v = np.array(dp_v); dp_s = np.array(dp_s)
    dk_v = np.array(dk_v); dk_s = np.array(dk_s)
    combo = dp_v - dk_v
    sig_c = np.sqrt(dp_s ** 2 + dk_s ** 2)
    n = len(ENERGIES)
    res_p    = dl - dp_v
    res_comb = dl - combo

    chi2_p    = np.sum(res_p ** 2    / (sl ** 2 + dp_s ** 2)) / n
    chi2_comb = np.sum(res_comb ** 2 / (sl ** 2 + sig_c ** 2)) / n
    # Both with same Dp denominator (isolate kaon-sigma inflation)
    chi2_comb_fix = np.sum(res_comb ** 2 / (sl ** 2 + dp_s ** 2)) / n

    dk_var_pct = 100 * np.mean(dk_s ** 2) / np.mean(dp_s ** 2)
    print(f'{cent:5s}  {chi2_p:12.2f}  {chi2_comb:14.2f}  {chi2_comb_fix:18.2f}  '
          f'{np.sqrt(np.mean(res_p**2)):12.4f}  {np.sqrt(np.mean(res_comb**2)):14.4f}  {dk_var_pct:10.1f}')

print('\n  "Dp-DK fixed-denom" uses sigma(DL+Dp) for both — isolates the residual shift from kaon subtraction.')
print('  "DK/Dp var%" = sigma(DK)^2 / sigma(Dp)^2 — how much kaon uncertainty inflates the denominator.\n')

# -------------------------------------------------------------------------
print('\n======= PART 2: Eff vs no-eff DeltaLambda chi2 decomposition vs (Dp-DK) =======\n')
print(f'{"cent":5s}  {"N":>3s}  {"sig ratio e/n":>13s}  {"RMS eff":>8s}  {"RMS noe":>8s}  '
      f'{"chi2 eff":>9s}  {"chi2 noe":>9s}  '
      f'{"fix-eff-sig: noe":>17s}  {"fix-noe-sig: eff":>17s}')
for cent in ['010', '1040', '4080']:
    dl_eff = np.array(eff_d[cent]['v']); sl_eff = np.array(eff_d[cent]['e'])
    dl_noe = np.array(noe_d[cent]['v']); sl_noe = np.array(noe_d[cent]['e'])
    dp_v, dp_s, dk_v, dk_s = [], [], [], []
    for e in ENERGIES:
        pe = ENERGY_PIKP[e]
        dp_v.append(pikp[pe]['protons'][f'{cent}_linear']['delta'])
        dp_s.append(pikp[pe]['protons'][f'{cent}_linear']['delta_err'])
        dk_v.append(pikp[pe]['kaons'][f'{cent}_linear']['delta'])
        dk_s.append(pikp[pe]['kaons'][f'{cent}_linear']['delta_err'])
    dp_v = np.array(dp_v); dp_s = np.array(dp_s)
    dk_v = np.array(dk_v); dk_s = np.array(dk_s)
    combo = dp_v - dk_v
    sig_c = np.sqrt(dp_s ** 2 + dk_s ** 2)

    mask = np.isfinite(dl_noe) & np.isfinite(sl_noe)
    n = mask.sum()
    re = (dl_eff - combo)[mask]; rn = (dl_noe - combo)[mask]
    se = sl_eff[mask]; sn = sl_noe[mask]; sc = sig_c[mask]

    chi2_eff = np.sum(re ** 2 / (se ** 2 + sc ** 2)) / n
    chi2_noe = np.sum(rn ** 2 / (sn ** 2 + sc ** 2)) / n
    # fixed at eff sigma: only residual change matters
    chi2_noe_fse = np.sum(rn ** 2 / (se ** 2 + sc ** 2)) / n
    # fixed at noe sigma: see inflation effect on eff
    chi2_eff_fsn = np.sum(re ** 2 / (sn ** 2 + sc ** 2)) / n

    sig_ratio = np.mean(se) / np.mean(sn)
    print(f'{cent:5s}  {n:3d}  {sig_ratio:13.2f}  '
          f'{np.sqrt(np.mean(re**2)):8.4f}  {np.sqrt(np.mean(rn**2)):8.4f}  '
          f'{chi2_eff:9.2f}  {chi2_noe:9.2f}  '
          f'{chi2_noe_fse:17.2f}  {chi2_eff_fsn:17.2f}')

print('\n  "fix-eff-sig: noe" = no-eff chi2 computed with eff sigma -> isolates residual change.')
print('  "fix-noe-sig: eff" = eff chi2 computed with no-eff sigma -> see how much larger eff chi2 would be without inflation.\n')
