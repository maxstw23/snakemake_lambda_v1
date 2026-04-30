import numpy as np
from matplotlib import pyplot as plt
import uproot
# import pickle
import yaml
import platform
import scipy.odr as odr
from scipy.optimize import curve_fit
from iminuit import cost, Minuit
from uncertainties import unumpy, ufloat, core
import pandas as pd
import argparse
from matplotlib.patches import Rectangle
from measurement import Measurement
from find_bin_center import BinCenterFinder
from simple_profile import SimpleProfile

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)

def find_csv(particle, flow, paths):
    for path in paths:
        if f'fit_{particle}_{flow}' in path:
            return path
    return None

def find_csv_pt(particle, ew, paths):
    for path in paths:
        if f'pt_fit_{particle}_{ew}' in path:
            return path
    return None


def find_csv_piKp(particle, paths):
    for path in paths:
        if particle in path:
            return path
    return None
        

def main(paths, paths_piKp, paths_pt, fres, output, method, **kwargs):
    no_a1 = True
    print(f'paths: {paths}')
    if method == 1:
        fun = func_wrapper
        range = config['y_cut']
    # elif method == 'curve_fit':
    #     fun = func # need to switch argument, so obselete for now
    elif method == 3:
        fun = func_wrapper_3rd
        range = config['y_cut']

    if kwargs["sys_tag"] == '6':
        method = 3 # use 3rd order polynomial for fitting as a systematic check
        fun = func_wrapper_3rd
    if kwargs["sys_tag"] == '5':
        range = 'half' # use only the positive y range for fitting as a systematic check

    # prefix for output path
    prefix = 'special_' if float(kwargs["sys_tag"]) >= 5 else ""
    fres_root = uproot.open(fres)
    cen_correspondence = {5: '10-40%', 1: '40-80%'}
    hres = unumpy.uarray(fres_root['hEPDEP_ew_cos_1'].values(), fres_root['hEPDEP_ew_cos_1'].errors())
    # mask out centralities where the resolution^2 is negative
    cen_mask = unumpy.nominal_values(hres) > 0
    hres_ct = fres_root['hEPDEP_ew_cos_1'].counts()
    resolution = np.where(cen_mask, abs(hres)**0.5, 0)
    fig_res, ax_res = plt.subplots(figsize=(8, 6))

    # 0.42 - 1.8
    lambda_v1_df = pd.read_csv(find_csv('Lambda', 'v1', paths), header=[0, 1], index_col=0)
    lambdabar_v1_df = pd.read_csv(find_csv('Lambdabar', 'v1', paths), header=[0, 1], index_col=0)
    if not no_a1:
        lambda_a1_df = pd.read_csv(find_csv('Lambda', 'a1', paths), header=[0, 1], index_col=0)
        lambdabar_a1_df = pd.read_csv(find_csv('Lambdabar', 'a1', paths), header=[0, 1], index_col=0)
    lambda_v1_pt_east_df = pd.read_csv(find_csv_pt('Lambda', 'east', paths_pt), header=[0, 1], index_col=0)
    lambda_v1_pt_west_df = pd.read_csv(find_csv_pt('Lambda', 'west', paths_pt), header=[0, 1], index_col=0)
    lambdabar_v1_pt_east_df = pd.read_csv(find_csv_pt('Lambdabar', 'east', paths_pt), header=[0, 1], index_col=0)
    lambdabar_v1_pt_west_df = pd.read_csv(find_csv_pt('Lambdabar', 'west', paths_pt), header=[0, 1], index_col=0)

    lambda_v1 = {}
    lambdabar_v1 = {}
    if not no_a1:
        lambda_a1 = {}
        lambdabar_a1 = {}
    lambda_v1_pt_east = {}
    lambda_v1_pt_west = {}
    lambdabar_v1_pt_east = {}
    lambdabar_v1_pt_west = {}
    for cen in lambda_v1_df.columns.levels[0]:
        lambda_v1[int(cen)] = {s: lambda_v1_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        lambdabar_v1[int(cen)] = {s: lambdabar_v1_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        if not no_a1:
            lambda_a1[int(cen)] = {s: lambda_a1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
            lambdabar_a1[int(cen)] = {s: lambdabar_a1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
    for cen in lambda_v1_pt_east_df.columns.levels[0]:
        lambda_v1_pt_east[int(cen)] = {s: lambda_v1_pt_east_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        lambda_v1_pt_west[int(cen)] = {s: lambda_v1_pt_west_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        lambdabar_v1_pt_east[int(cen)] = {s: lambdabar_v1_pt_east_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
        lambdabar_v1_pt_west[int(cen)] = {s: lambdabar_v1_pt_west_df.loc[:, (cen, s)].values for s in ['values', 'counts', 'errors']}
    
    df_pions = pd.read_csv(find_csv_piKp('pions', paths_piKp))
    df_kaons = pd.read_csv(find_csv_piKp('kaons', paths_piKp))
    df_protons = pd.read_csv(find_csv_piKp('protons', paths_piKp))
    antiproton_v1_slopes = unumpy.uarray(df_protons['v1_n'], df_protons['v1_n_err'])#[:9]
    kminus_v1_slopes = unumpy.uarray(df_kaons['v1_n'], df_kaons['v1_n_err'])#[:9]
    proton_v1_slopes = unumpy.uarray(df_protons['v1_p'], df_protons['v1_p_err'])#[:9]
    kplus_v1_slopes = unumpy.uarray(df_kaons['v1_p'], df_kaons['v1_p_err'])#[:9]
    piminus_v1_slopes = unumpy.uarray(df_pions['v1_n'], df_pions['v1_n_err'])#[:9]
    piplus_v1_slopes = unumpy.uarray(df_pions['v1_p'], df_pions['v1_p_err'])#[:9]
    antiproton_d3v1dy3 = unumpy.uarray(df_protons['d3v1dy3_n'], df_protons['d3v1dy3_n_err'])#[:9]
    kminus_d3v1dy3 = unumpy.uarray(df_kaons['d3v1dy3_n'], df_kaons['d3v1dy3_n_err'])#[:9]
    proton_d3v1dy3 = unumpy.uarray(df_protons['d3v1dy3_p'], df_protons['d3v1dy3_p_err'])#[:9]
    kplus_d3v1dy3 = unumpy.uarray(df_kaons['d3v1dy3_p'], df_kaons['d3v1dy3_p_err'])#[:9]
    piminus_d3v1dy3 = unumpy.uarray(df_pions['d3v1dy3_n'], df_pions['d3v1dy3_n_err'])#[:9]
    piplus_d3v1dy3 = unumpy.uarray(df_pions['d3v1dy3_p'], df_pions['d3v1dy3_p_err'])#[:9]


    # Plotting
    num_ybin = len(lambda_v1[1]['values'])
    ybin_edges = np.linspace(-1., 1., num_ybin + 1)
    ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
    ybin_err = np.diff(ybin_edges) / 2
    ybin_unumpy = unumpy.uarray(ybin, ybin_err)
    centralities = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
    centralities_bins = np.array([80, 70, 60, 50, 40, 30, 20, 10, 5, 0])
    ax_res.errorbar(centralities[cen_mask], unumpy.nominal_values(resolution[cen_mask]), xerr=0, yerr=unumpy.std_devs(resolution[cen_mask]),
                    fmt='d', color='k', capsize=2, ms=8)
    # ax_res.set_ylim(0, 0.5)
    ax_res.set_xlabel('Centrality %', fontsize=18, loc='right')
    ax_res.set_ylabel(r'Resolution', fontsize=18, loc='top')
    data_dict = {'x': centralities[cen_mask], 'y': unumpy.nominal_values(resolution[cen_mask]), 'yerr': unumpy.std_devs(resolution[cen_mask])}
    with open(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/paper_yaml/resolution_{kwargs["energy"]}.yaml', 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)
    # try:
    #     pickle.dump(fig_res, open(f'plots/paper_yaml/resolution_{kwargs["energy"]}.pkl', 'wb'))
    # except KeyError:
    #     pickle.dump(fig_res, open(f'plots/paper_yaml/resolution.pkl', 'wb'))

    ### pi K p
    fig_piKp, ax_piKp = plt.subplots(2, 2, figsize=(12, 9))
    ax_piKp[0,0].errorbar(centralities[cen_mask], unumpy.nominal_values(proton_v1_slopes[cen_mask]), 
                        yerr=unumpy.std_devs(proton_v1_slopes[cen_mask]), fmt='o', color='black', capsize=2, label=r'$p$')
    ax_piKp[0,0].errorbar(centralities[cen_mask], unumpy.nominal_values(antiproton_v1_slopes[cen_mask]), 
                        yerr=unumpy.std_devs(antiproton_v1_slopes[cen_mask]), fmt='o', color='red', capsize=2, label=r'$\bar{p}$')
    ax_piKp[0,0].hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_piKp[0,0].set_xlabel('Centrality (%)')
    ax_piKp[0,0].set_ylabel(r'$dv_1/dy$')
    ax_piKp[0,0].legend()
    ax_piKp[1,0].errorbar(centralities[cen_mask], unumpy.nominal_values(kplus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(kplus_v1_slopes[cen_mask]), fmt='o', color='black', capsize=2, label=r'$K^{+}$')
    ax_piKp[1,0].errorbar(centralities[cen_mask], unumpy.nominal_values(kminus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(kminus_v1_slopes[cen_mask]), fmt='o', color='red', capsize=2, label=r'$K^{-}$')
    ax_piKp[1,0].hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_piKp[1,0].set_xlabel('Centrality (%)')
    ax_piKp[1,0].set_ylabel(r'$dv_1/dy$')
    ax_piKp[1,0].legend()
    ax_piKp[0,1].errorbar(centralities[cen_mask], unumpy.nominal_values(piplus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(piplus_v1_slopes[cen_mask]), fmt='o', color='black', capsize=2, label=r'$\pi^{+}$')
    ax_piKp[0,1].errorbar(centralities[cen_mask], unumpy.nominal_values(piminus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(piminus_v1_slopes[cen_mask]), fmt='o', color='red', capsize=2, label=r'$\pi^{-}$')
    ax_piKp[0,1].hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_piKp[0,1].set_xlabel('Centrality (%)')
    ax_piKp[0,1].set_ylabel(r'$dv_1/dy$')
    ax_piKp[0,1].legend()
    ax_piKp[1,1].errorbar(centralities[cen_mask], unumpy.nominal_values(proton_v1_slopes[cen_mask] - antiproton_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(proton_v1_slopes[cen_mask] - antiproton_v1_slopes[cen_mask]), fmt='o', capsize=2, label=r'$p-\bar{p}$')
    ax_piKp[1,1].errorbar(centralities[cen_mask], unumpy.nominal_values(kplus_v1_slopes[cen_mask] - kminus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(kplus_v1_slopes[cen_mask] - kminus_v1_slopes[cen_mask]), fmt='o', capsize=2, label=r'$K^{+}-K^{-}$')
    ax_piKp[1,1].errorbar(centralities[cen_mask], unumpy.nominal_values(piplus_v1_slopes[cen_mask] - piminus_v1_slopes[cen_mask]),
                        yerr=unumpy.std_devs(piplus_v1_slopes[cen_mask] - piminus_v1_slopes[cen_mask]), fmt='o', capsize=2, label=r'$\pi^{+}-\pi^{-}$')
    ax_piKp[1,1].hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_piKp[1,1].set_xlabel('Centrality (%)')
    ax_piKp[1,1].set_ylabel(r'$\Delta dv_1/dy$')
    ax_piKp[1,1].legend()  

    ### v1
    y_bin_centers = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])
    def initialize_y_merged_dict():
        return {'1234': {'x': y_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(20), np.zeros(20)))},
                '567': {'x': y_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(20), np.zeros(20)))},
                '89': {'x': y_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(20), np.zeros(20)))},
                '123': {'x': y_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(20), np.zeros(20)))},
                }
    merged_cen_labels = {'1234': '40-80%', '567': '10-40%', '89': '0-10%', '123': '50-80%'}
    merged_cen_titles = {'1234': '4080', '567': '1040', '89': '010', '123': '5080'}
    # fig_1, ax_1 = plt.subplots(3, 3, figsize=(24, 18))
    fig_1 = plt.figure(figsize=(24, 18))
    gs_1 = fig_1.add_gridspec(3, 3, hspace=0, wspace=0)
    ax_1 = gs_1.subplots(sharex='col', sharey='row')
    # lambda
    v1_y_lambda_merged = initialize_y_merged_dict()
    dv1dy_lambda = np.zeros(9)
    dv1dy_lambda_err = np.zeros(9)
    d3v1dy3_lambda = np.zeros(9)
    d3v1dy3_lambda_err = np.zeros(9)
    max_ylim = np.zeros(3)
    for cen, data in lambda_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)

        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambda v1(y) data for centrality {cen}.')
        v1 = data['values'][np.invert(bool_nan)]
        v1_count = data['counts'][np.invert(bool_nan)]
        v1_err = data['errors'][np.invert(bool_nan)]
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy[np.invert(bool_nan)]
        # For merging: keep full-length arrays with inf error for NaN bins (zero weight)
        v1_all = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_err_all = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_count_all = np.where(bool_nan, 0.0, data['counts'].astype(float))
        v1_all_unumpy = unumpy.uarray(v1_all, v1_err_all)
        v1_final_all = v1_all_unumpy / resolution[cen - 1]
        for group in v1_y_lambda_merged.keys():
            if str(cen) in group:
                v1_y_lambda_merged[group]['measurement'] += Measurement(v1_final_all)
                v1_y_lambda_merged[group]['count'] += v1_count_all
            if group.endswith(str(cen)):
                v1_y_lambda_merged[group]['x'] = unumpy.nominal_values(ybin_good)

        _ynom = unumpy.nominal_values(ybin_good)
        _fit_mask = np.abs(_ynom) < float(config['y_cut'])
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[_fit_mask], unumpy.nominal_values(v1_final[_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[_fit_mask]),
                                                     fmt='o', color='C0', capsize=2, label=r'$\Lambda$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[~_fit_mask], unumpy.nominal_values(v1_final[~_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[~_fit_mask]),
                                                     fmt='o', color='C0', capsize=2, markerfacecolor='white', label='_nolegend_')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                     xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        top_ylim = np.max(unumpy.nominal_values(v1_final) + unumpy.std_devs(v1_final))
        bot_ylim = np.min(unumpy.nominal_values(v1_final) - unumpy.std_devs(v1_final))
        current_ylim = np.max([top_ylim, -bot_ylim])
        max_ylim[(cen - 1) // 3] = np.max([current_ylim, max_ylim[(cen - 1) // 3]])

        # fitting
        popt, perr, rchi2 = fit(ybin_good, v1_final, method=method, range=range)
        popt_first, perr_first, rchi2_first = fit(ybin_good, v1_final, method=1, range=range)
        dv1dy_lambda[cen - 1] = popt[0]
        dv1dy_lambda_err[cen - 1] = perr[0]
        if method == 3: 
            d3v1dy3_lambda[cen - 1] = popt[1]
            d3v1dy3_lambda_err[cen - 1] = perr[1]
            # if third order term is not significant, revert to 1st order fit
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_first[0]
                perr[0] = perr_first[0]
                popt[1] = 0
                perr[1] = 0
                # d3v1dy3_lambda[cen - 1] = 0
                # d3v1dy3_lambda_err[cen - 1] = 0
                dv1dy_lambda[cen - 1] = popt_first[0]
                dv1dy_lambda_err[cen - 1] = perr_first[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C0')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Lambda$' + r' $d^3 v_1/dy^3$ ' + f'{ufloat(d3v1dy3_lambda[cen - 1], d3v1dy3_lambda_err[cen - 1])}',
                                                     xy=(0.05, 0.15), xycoords='axes fraction', fontsize=15)
        # ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Lambda$' + f' rchi2 = ' + f'{rchi2:.2f}' + ', rchi2 (1st) = ' + f'{rchi2_first:.2f}',
        #                                              xy=(0.55, 0.75), xycoords='axes fraction', fontsize=12)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].legend(fontsize=15)

    # lambdabar
    v1_y_lambdabar_merged = initialize_y_merged_dict()
    dv1dy_lambdabar = np.zeros(9)
    dv1dy_lambdabar_err = np.zeros(9)
    d3v1dy3_lambdabar = np.zeros(9)
    d3v1dy3_lambdabar_err = np.zeros(9)
    for cen, data in lambdabar_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)

        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambdabar v1(y) data for centrality {cen}.')
        v1 = data['values'][np.invert(bool_nan)]
        v1_count = data['counts'][np.invert(bool_nan)]
        v1_err = data['errors'][np.invert(bool_nan)]
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy[np.invert(bool_nan)]
        # For merging: keep full-length arrays with inf error for NaN bins (zero weight)
        v1_all = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_err_all = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_count_all = np.where(bool_nan, 0.0, data['counts'].astype(float))
        v1_all_unumpy = unumpy.uarray(v1_all, v1_err_all)
        v1_final_all = v1_all_unumpy / resolution[cen - 1]
        for group in v1_y_lambdabar_merged.keys():
            if str(cen) in group:
                v1_y_lambdabar_merged[group]['measurement'] += Measurement(v1_final_all)
                v1_y_lambdabar_merged[group]['count'] += v1_count_all
            if group.endswith(str(cen)):
                v1_y_lambdabar_merged[group]['x'] = unumpy.nominal_values(ybin_good)
        _ynom = unumpy.nominal_values(ybin_good)
        _fit_mask = np.abs(_ynom) < float(config['y_cut'])
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[_fit_mask], unumpy.nominal_values(v1_final[_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[_fit_mask]),
                                                     fmt='o', color='C1', capsize=2, label=r'$\bar{\Lambda}$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[~_fit_mask], unumpy.nominal_values(v1_final[~_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[~_fit_mask]),
                                                     fmt='o', color='C1', capsize=2, markerfacecolor='white', label='_nolegend_')
        top_ylim = np.max(unumpy.nominal_values(v1_final) + unumpy.std_devs(v1_final))
        bot_ylim = np.min(unumpy.nominal_values(v1_final) - unumpy.std_devs(v1_final))
        current_ylim = np.max([top_ylim, -bot_ylim])
        max_ylim[(cen - 1) // 3] = np.max([max_ylim[(cen - 1) // 3], current_ylim])
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-max_ylim[(cen - 1) // 3] * 1.2, max_ylim[(cen - 1) // 3] * 1.2)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$', fontsize=18)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$v_1$', fontsize=18)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].tick_params(axis='both', labelsize=15)

        # fitting
        popt, perr, rchi2 = fit(ybin_good, v1_final, method=method, range=range)
        popt_first, perr_first, rchi2_first = fit(ybin_good, v1_final, method=1, range=range)
        dv1dy_lambdabar[cen - 1] = popt[0]
        dv1dy_lambdabar_err[cen - 1] = perr[0]
        if method == 3: 
            d3v1dy3_lambdabar[cen - 1] = popt[1]
            d3v1dy3_lambdabar_err[cen - 1] = perr[1]
            # if third order term is not significant, revert to 1st order fit
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_first[0]
                perr[0] = perr_first[0]
                popt[1] = 0
                perr[1] = 0
                # d3v1dy3_lambdabar[cen - 1] = 0
                # d3v1dy3_lambdabar_err[cen - 1] = 0
                dv1dy_lambdabar[cen - 1] = popt_first[0]
                dv1dy_lambdabar_err[cen - 1] = perr_first[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C1')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\bar{\Lambda}$' + r' $d^3 v_1/dy^3$ ' + f'{ufloat(d3v1dy3_lambdabar[cen - 1], d3v1dy3_lambdabar_err[cen - 1])}',
                                                     xy=(0.05, 0.05), xycoords='axes fraction', fontsize=15)
        # ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\bar{\Lambda}$' + f' rchi2 = ' + f'{rchi2:.2f}' + ', rchi2 (1st) = ' + f'{rchi2_first:.2f}',
        #                                              xy=(0.55, 0.65), xycoords='axes fraction', fontsize=12)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].legend(fontsize=15)

    # delta
    v1_y_deltalambda_merged = initialize_y_merged_dict()
    dv1dy_deltalambda = np.zeros(9)
    dv1dy_deltalambda_err = np.zeros(9)
    d3v1dy3_deltalambda = np.zeros(9)
    d3v1dy3_deltalambda_err = np.zeros(9)
    fig_delta = plt.figure(figsize=(24, 18))
    gs_delta = fig_delta.add_gridspec(3, 3, hspace=0, wspace=0)
    ax_delta = gs_delta.subplots(sharex='col', sharey='row')
    for cen, data in lambda_v1.items():
        data_bar = lambdabar_v1[cen]
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)

        bool_nan = np.isnan(data['values'].astype(float)) | np.isnan(data_bar['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambda or lambdabar v1(y) data for centrality {cen}.')
        v1 = data['values'][np.invert(bool_nan)] - data_bar['values'][np.invert(bool_nan)]
        # v1_count = data['counts'][np.invert(bool_nan)] + data_bar['counts'][np.invert(bool_nan)]
        v1_err = np.sqrt(data['errors'][np.invert(bool_nan)] ** 2 + data_bar['errors'][np.invert(bool_nan)] ** 2)
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy[np.invert(bool_nan)]
        # For merging: keep full-length arrays with inf error for NaN bins (zero weight)
        v1_all = np.where(bool_nan, 0.0, data['values'].astype(float) - data_bar['values'].astype(float))
        v1_err_all = np.where(bool_nan, np.inf, np.sqrt(data['errors'].astype(float) ** 2 + data_bar['errors'].astype(float) ** 2))
        v1_all_unumpy = unumpy.uarray(v1_all, v1_err_all)
        v1_final_all = v1_all_unumpy / resolution[cen - 1]
        for group in v1_y_deltalambda_merged.keys():
            if str(cen) in group:
                v1_y_deltalambda_merged[group]['measurement'] += Measurement(v1_final_all)
                # v1_y_deltalambda_merged[group]['count'] += v1_count_lambda
            if group.endswith(str(cen)):
                v1_y_lambdabar_merged[group]['x'] = unumpy.nominal_values(ybin_good)
        _ynom = unumpy.nominal_values(ybin_good)
        _fit_mask = np.abs(_ynom) < float(config['y_cut'])
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[_fit_mask], unumpy.nominal_values(v1_final[_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[_fit_mask]),
                                                     fmt='o', color='C2', capsize=2, label=r'$\Delta\Lambda$')
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[~_fit_mask], unumpy.nominal_values(v1_final[~_fit_mask]),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final[~_fit_mask]),
                                                     fmt='o', color='C2', capsize=2, markerfacecolor='white', label='_nolegend_')
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$', fontsize=18)
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$v_1$', fontsize=18)
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].tick_params(axis='both', labelsize=15)
        top_ylim = np.max(unumpy.nominal_values(v1_final) + unumpy.std_devs(v1_final))
        bot_ylim = np.min(unumpy.nominal_values(v1_final) - unumpy.std_devs(v1_final))
        current_ylim = np.max([top_ylim, -bot_ylim])
        max_ylim[(cen - 1) // 3] = np.max([current_ylim, max_ylim[(cen - 1) // 3]])
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-max_ylim[(cen - 1) // 3] * 1.2, max_ylim[(cen - 1) // 3] * 1.2)

        # fitting
        popt, perr, rchi2 = fit(ybin_good, v1_final, method=method, range=range)
        popt_first, perr_first, rchi2_first = fit(ybin_good, v1_final, method=1, range=range)
        dv1dy_deltalambda[cen - 1] = popt[0]
        dv1dy_deltalambda_err[cen - 1] = perr[0]
        if method == 3: 
            d3v1dy3_deltalambda[cen - 1] = popt[1]
            d3v1dy3_deltalambda_err[cen - 1] = perr[1]
            # if third order term is not significant, revert to 1st order fit
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_first[0]
                perr[0] = perr_first[0]
                popt[1] = 0
                perr[1] = 0
                # d3v1dy3_deltalambda[cen - 1] = 0
                # d3v1dy3_deltalambda_err[cen - 1] = 0
                dv1dy_deltalambda[cen - 1] = popt_first[0]
                dv1dy_deltalambda_err[cen - 1] = perr_first[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C2')
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                     xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Delta\Lambda$' + r' $d^3 v_1/dy^3$ ' + f'{ufloat(d3v1dy3_deltalambda[cen - 1], d3v1dy3_deltalambda_err[cen - 1])}',
                                                     xy=(0.05, 0.15), xycoords='axes fraction', fontsize=15)
        # ax_delta[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Delta\Lambda$' + f' rchi2 = ' + f'{rchi2:.2f}' + ', rchi2 (1st) = ' + f'{rchi2_first:.2f}',
        #                                              xy=(0.55, 0.55), xycoords='axes fraction', fontsize=12)
        ax_delta[(cen - 1) // 3, (cen - 1) % 3].legend()

    # slopes
    fig_3, ax_3 = plt.subplots(1, 1, figsize=(8, 6))
    ax_3.errorbar(centralities[cen_mask], dv1dy_lambda[cen_mask], yerr=dv1dy_lambda_err[cen_mask], fmt='o', color='black', capsize=2,
                  label=r'$\Lambda$')
    ax_3.errorbar(centralities[cen_mask], dv1dy_lambdabar[cen_mask], yerr=dv1dy_lambdabar_err[cen_mask], fmt='o', color='red', capsize=2,
                  label=r'$\bar{\Lambda}$')
    ax_3.set_xlabel('Centrality (%)')
    ax_3.set_ylabel(r'$dv_1/dy$')
    ax_3.legend()

    if not no_a1:
        # lambda a1
        fig_2 = plt.figure(figsize=(24, 18))
        gs_2 = fig_2.add_gridspec(3, 3, hspace=0, wspace=0)
        ax_2 = gs_2.subplots(sharex='col', sharey='row')
        da1dy_lambda = np.zeros(9)
        da1dy_lambda_err = np.zeros(9)
        d3a1dy3_lambda = np.zeros(9)
        d3a1dy3_lambda_err = np.zeros(9)
        max_ylim_a1 = np.zeros(3)
        for cen, data in lambda_a1.items():
            if cen_mask[cen - 1] == False:
                continue
            num_ybin = len(data['values'])
            ybin_edges = np.linspace(-1., 1., num_ybin + 1)
            ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
            ybin_err = np.diff(ybin_edges) / 2
            ybin_unumpy = unumpy.uarray(ybin, ybin_err)

            bool_nan = np.isnan(data['values'].astype(float))
            a1 = data['values'][np.invert(bool_nan)]
            a1_err = data['errors'][np.invert(bool_nan)]
            a1_unumpy = unumpy.uarray(a1, a1_err)
            a1_final = a1_unumpy / resolution[cen - 1]
            ybin_good = ybin_unumpy[np.invert(bool_nan)]
            _ynom = unumpy.nominal_values(ybin_good)
            _fit_mask = np.abs(_ynom) < float(config['y_cut'])
            ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[_fit_mask], unumpy.nominal_values(a1_final[_fit_mask]),
                                                         xerr=0, yerr=unumpy.std_devs(a1_final[_fit_mask]),
                                                         fmt='o', color='C0', capsize=2, label=r'$\Lambda$')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[~_fit_mask], unumpy.nominal_values(a1_final[~_fit_mask]),
                                                         xerr=0, yerr=unumpy.std_devs(a1_final[~_fit_mask]),
                                                         fmt='o', color='C0', capsize=2, markerfacecolor='white', label='_nolegend_')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                         xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
            top_ylim = np.max(unumpy.nominal_values(a1_final) + unumpy.std_devs(a1_final))
            bot_ylim = np.min(unumpy.nominal_values(a1_final) - unumpy.std_devs(a1_final))
            current_ylim = np.max([top_ylim, -bot_ylim])
            max_ylim_a1[(cen - 1) // 3] = np.max([current_ylim, max_ylim_a1[(cen - 1) // 3]])
            
            # fitting
            popt, perr, rchi2 = fit(ybin_good, a1_final, method=1, range=range)
            # popt_first, perr_first, rchi2_first = fit(ybin_good, a1_final, method=1, range=range)
            da1dy_lambda[cen - 1] = popt[0]
            da1dy_lambda_err[cen - 1] = perr[0]
            # if method == 3: 
            #     d3a1dy3_lambda[cen - 1] = popt[1]
            #     d3a1dy3_lambda_err[cen - 1] = perr[1]
            #     # if third order term is not significant, revert to 1st order fit
            #     if abs(popt[1]) < perr[1]:
            #         popt[0] = popt_first[0]
            #         perr[0] = perr_first[0]
            #         popt[1] = 0
            #         perr[1] = 0
            #         # d3v1dy3_lambdabar[cen - 1] = 0
            #         # d3v1dy3_lambdabar_err[cen - 1] = 0
            #         da1dy_lambda[cen - 1] = popt_first[0]
            #         da1dy_lambda_err[cen - 1] = perr_first[0]

            x_fit = np.linspace(-1, 1, 101)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, func_wrapper(popt, x_fit), '-', color='C0')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Lambda$' + r' $d a_1/dy$ ' + f'{ufloat(da1dy_lambda[cen - 1], da1dy_lambda_err[cen - 1])}',
                                                         xy=(0.05, 0.15), xycoords='axes fraction', fontsize=15)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].legend()

        # lambdabar a1
        da1dy_lambdabar = np.zeros(9)
        da1dy_lambdabar_err = np.zeros(9)
        d3a1dy3_lambdabar = np.zeros(9)
        d3a1dy3_lambdabar_err = np.zeros(9)
        for cen, data in lambdabar_a1.items():
            if cen_mask[cen - 1] == False:
                continue
            num_ybin = len(data['values'])
            ybin_edges = np.linspace(-1., 1., num_ybin + 1)
            ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
            ybin_err = np.diff(ybin_edges) / 2
            ybin_unumpy = unumpy.uarray(ybin, ybin_err)
            
            bool_nan = np.isnan(data['values'].astype(float))
            a1 = data['values'][np.invert(bool_nan)]
            a1_err = data['errors'][np.invert(bool_nan)]
            a1_unumpy = unumpy.uarray(a1, a1_err)
            a1_final = a1_unumpy / resolution[cen - 1]
            ybin_good = ybin_unumpy[np.invert(bool_nan)]
            _ynom = unumpy.nominal_values(ybin_good)
            _fit_mask = np.abs(_ynom) < float(config['y_cut'])
            ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[_fit_mask], unumpy.nominal_values(a1_final[_fit_mask]),
                                                         xerr=0, yerr=unumpy.std_devs(a1_final[_fit_mask]),
                                                         fmt='o', color='C1', capsize=2, label=r'$\bar{\Lambda}$')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(_ynom[~_fit_mask], unumpy.nominal_values(a1_final[~_fit_mask]),
                                                         xerr=0, yerr=unumpy.std_devs(a1_final[~_fit_mask]),
                                                         fmt='o', color='C1', capsize=2, markerfacecolor='white', label='_nolegend_')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$', fontsize=18)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$a_1$', fontsize=18)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].tick_params(axis='both', labelsize=15)
            top_ylim = np.max(unumpy.nominal_values(a1_final) + unumpy.std_devs(a1_final))
            bot_ylim = np.min(unumpy.nominal_values(a1_final) - unumpy.std_devs(a1_final))
            current_ylim = np.max([top_ylim, -bot_ylim])
            max_ylim_a1[(cen - 1) // 3] = np.max([max_ylim_a1[(cen - 1) // 3], current_ylim])
            ax_2[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-max_ylim_a1[(cen - 1) // 3] * 1.2, max_ylim_a1[(cen - 1) // 3] * 1.2)

            # fitting
            popt, perr, rchi2 = fit(ybin_good, a1_final, method=1, range=range)
            # popt_first, perr_first, rchi2_first = fit(ybin_good, a1_final, method=1, range=range)
            da1dy_lambdabar[cen - 1] = popt[0]
            da1dy_lambdabar_err[cen - 1] = perr[0]
            # if method == 3: 
            #     d3a1dy3_lambdabar[cen - 1] = popt[1]
            #     d3a1dy3_lambdabar_err[cen - 1] = perr[1]
            #     # if third order term is not significant, revert to 1st order fit
            #     if abs(popt[1]) < perr[1]:
            #         popt[0] = popt_first[0]
            #         perr[0] = perr_first[0]
            #         popt[1] = 0
            #         perr[1] = 0
            #         # d3v1dy3_lambdabar[cen - 1] = 0
            #         # d3v1dy3_lambdabar_err[cen - 1] = 0
            #         da1dy_lambdabar[cen - 1] = popt_first[0]
            #         da1dy_lambdabar_err[cen - 1] = perr_first[0]

            x_fit = np.linspace(-1, 1, 101)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, func_wrapper(popt, x_fit), '-', color='C1')
            ax_2[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\bar{\Lambda}$' + r' $d a_1/dy$ ' + f'{ufloat(da1dy_lambdabar[cen - 1], da1dy_lambdabar_err[cen - 1])}',
                                                         xy=(0.05, 0.05), xycoords='axes fraction', fontsize=15)
            ax_2[(cen - 1) // 3, (cen - 1) % 3].legend()

        # delta a1
        da1dy_deltalambda = np.zeros(9)
        da1dy_deltalambda_err = np.zeros(9)
        d3a1dy3_deltalambda = np.zeros(9)
        d3a1dy3_deltalambda_err = np.zeros(9)
        fig_delta_a1 = plt.figure(figsize=(24, 18))
        gs_delta_a1 = fig_delta_a1.add_gridspec(3, 3, hspace=0, wspace=0)
        ax_delta_a1 = gs_delta_a1.subplots(sharex='col', sharey='row')
        for cen, data in lambda_a1.items():
            data_bar = lambdabar_a1[cen]
            if cen_mask[cen - 1] == False:
                continue
            num_ybin = len(data['values'])
            ybin_edges = np.linspace(-1., 1., num_ybin + 1)
            ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
            ybin_err = np.diff(ybin_edges) / 2
            ybin_unumpy = unumpy.uarray(ybin, ybin_err)

            bool_nan = np.isnan(data['values'].astype(float)) | np.isnan(data_bar['values'].astype(float))
            a1 = data['values'][np.invert(bool_nan)] - data_bar['values'][np.invert(bool_nan)]
            a1_err = np.sqrt(data['errors'][np.invert(bool_nan)] ** 2 + data_bar['errors'][np.invert(bool_nan)] ** 2)
            a1_unumpy = unumpy.uarray(a1, a1_err)
            a1_final = a1_unumpy / resolution[cen - 1]
            ybin_good = ybin_unumpy[np.invert(bool_nan)]
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ybin_good), unumpy.nominal_values(a1_final),
                                                         xerr=0, yerr=unumpy.std_devs(a1_final),
                                                         fmt='o', color='C2', capsize=2, label=r'$\Delta\Lambda$'
                                                         )
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$', fontsize=18)
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$a_1$', fontsize=18)
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].tick_params(axis='both', labelsize=15)
            top_ylim = np.max(unumpy.nominal_values(a1_final) + unumpy.std_devs(a1_final))
            bot_ylim = np.min(unumpy.nominal_values(a1_final) - unumpy.std_devs(a1_final))
            current_ylim = np.max([top_ylim, -bot_ylim])
            max_ylim_a1[(cen - 1) // 3] = np.max([current_ylim, max_ylim_a1[(cen - 1) // 3]])
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-max_ylim_a1[(cen - 1) // 3] * 1.2, max_ylim_a1[(cen - 1) // 3] * 1.2)
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                         xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)

            # fitting
            print(f'Fitting delta a1 for centrality {cen}...')
            print(f'method: {method}, range: {range}')
            popt, perr, rchi2 = fit(ybin_good, a1_final, method=1, range=range, verbose=True)
            print(f'method: 1, range: {range} (first order fit for comparison)')
            # popt_first, perr_first, rchi2_first = fit(ybin_good, a1_final, method=1, range=range, verbose=True)
            da1dy_deltalambda[cen - 1] = popt[0]
            da1dy_deltalambda_err[cen - 1] = perr[0]
            # if method == 3: 
            #     d3a1dy3_deltalambda[cen - 1] = popt[1]
            #     d3a1dy3_deltalambda_err[cen - 1] = perr[1]
            #     # if third order term is not significant, revert to 1st order fit
            #     if abs(popt[1]) < perr[1]:
            #         popt[0] = popt_first[0]
            #         perr[0] = perr_first[0]
            #         popt[1] = 0
            #         perr[1] = 0
            #         # d3v1dy3_lambdabar[cen - 1] = 0
            #         # d3v1dy3_lambdabar_err[cen - 1] = 0
            #         da1dy_deltalambda[cen - 1] = popt_first[0]
            #         da1dy_deltalambda_err[cen - 1] = perr_first[0]

            x_fit = np.linspace(-1, 1, 101)
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, func_wrapper(popt, x_fit), '-', color='C2')
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].annotate(r'$\Delta\Lambda$' + r' $d a_1/dy$ ' + f'{ufloat(da1dy_deltalambda[cen - 1], da1dy_deltalambda_err[cen - 1])}',
                                                         xy=(0.05, 0.25), xycoords='axes fraction', fontsize=15)
            ax_delta_a1[(cen - 1) // 3, (cen - 1) % 3].legend()

        # slopes of a1
        fig_3_a1, ax_4_a1 = plt.subplots(1, 1, figsize=(8, 6))
        ax_4_a1.errorbar(centralities[cen_mask], da1dy_lambda[cen_mask], yerr=da1dy_lambda_err[cen_mask], fmt='o', color='black', capsize=2,
                    label=r'$\Lambda$')
        ax_4_a1.errorbar(centralities[cen_mask], da1dy_lambdabar[cen_mask], yerr=da1dy_lambdabar_err[cen_mask], fmt='o', color='red', capsize=2,
                    label=r'$\bar{\Lambda}$')
        ax_4_a1.set_xlabel('Centrality (%)')
        ax_4_a1.set_ylabel(r'$da_1/dy$')
        ax_4_a1.legend()

    ### pt dependent, for each centrality
    fig_6 = plt.figure(figsize=(24, 18))
    gs_6 = fig_6.add_gridspec(3, 3, hspace=0, wspace=0)
    ax_6 = gs_6.subplots(sharex='col', sharey='row')
    for cen, data in lambda_v1_pt_east.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ptbin = len(data['values'])
        ptbin_edges = np.linspace(0.4, 1.8, num_ptbin + 1)
        ptbin = 0.5 * (ptbin_edges[:-1] + ptbin_edges[1:])      
        ptbin_err = np.diff(ptbin_edges) / 2
        ptbin_unumpy = unumpy.uarray(ptbin, ptbin_err)

        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambda v1(pt) data for centrality {cen}.')
        v1 = data['values'][np.invert(bool_nan)]
        v1_count = data['counts'][np.invert(bool_nan)]
        v1_err = data['errors'][np.invert(bool_nan)]
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ptbin_good = ptbin_unumpy[np.invert(bool_nan)]
        ax_6[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ptbin_good), unumpy.nominal_values(v1_final),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final),
                                                     fmt='o', color='C0', capsize=2, label=r'$\Lambda$'
                                                     )
        ax_6[(cen - 1) // 3, (cen - 1) % 3].hlines(0., 0.4, 1.8, color='k', linestyle='--')
        ax_6[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$p_T$')
        ax_6[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$v_1$')
        ax_6[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                     xy=(0.5, 0.9), fontsize=15, xycoords='axes fraction')
    ### pt dependent
    fig_5_lambda, ax_5_lambda = plt.subplots(2, 2, figsize=(16, 12))
    fig_5_lambdabar, ax_5_lambdabar = plt.subplots(2, 2, figsize=(16, 12))
    ax_5_lambda = ax_5_lambda.flatten()
    ax_5_lambdabar = ax_5_lambdabar.flatten()
    pt_bin_edges = np.linspace(0.4, 1.8, 15)
    pt_bin_centers = 0.5 * (pt_bin_edges[:-1] + pt_bin_edges[1:])
    pt_bin_widths = pt_bin_edges[1] - pt_bin_edges[0]
    def initialize_merged_dict():
        return {'1234': {'x': pt_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(14), np.zeros(14)))},
                '567': {'x': pt_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(14), np.zeros(14)))},
                '89': {'x': pt_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(14), np.zeros(14)))},
                '123': {'x': pt_bin_centers, 'count': 0, 'measurement':Measurement(unumpy.uarray(np.zeros(14), np.zeros(14)))}
                }
    merged_cen_labels = {'1234': '40-80%', '567': '10-40%', '89': '0-10%', '123': '50-80%'}
    merged_cen_titles = {'1234': '4080', '567': '1040', '89': '010', '123': '5080'}
    pt_lo_forplot = 0.
    pt_hi_forplot = 2.

    v1_pt_merged = initialize_merged_dict()
    v1_pt_lambda_merged_combined = initialize_merged_dict()
    for cen, data in lambda_v1_pt_east.items():
        if cen_mask[cen - 1] == False:
            continue
        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambda_v1_pt_east for centrality {cen}.')
        v1 = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_count = np.where(bool_nan, 0, data['counts'])
        v1_err = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        for group in v1_pt_merged.keys():
            if str(cen) in group:
                v1_pt_merged[group]['measurement'] += Measurement(v1_final)
                v1_pt_lambda_merged_combined[group]['measurement'] += Measurement(v1_final)
                v1_pt_merged[group]['count'] += v1_count
                v1_pt_lambda_merged_combined[group]['count'] += v1_count
            if group.endswith(str(cen)):
                m = v1_pt_merged[group]['measurement'].get_measurement()
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_merged[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_merged[group]['x'] = pt_bin_centers_shifted
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_lambda_merged_combined[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_lambda_merged_combined[group]['x'] = pt_bin_centers_shifted
                ax_5_lambda[0].errorbar(pt_bin_centers_shifted, unumpy.nominal_values(m),
                                xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(m),
                                fmt='o', capsize=2, label=merged_cen_labels[group]
                                )
                ax_5_lambda[0].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
                ax_5_lambda[0].set_xlabel(r'$p_T$')
                ax_5_lambda[0].set_ylabel(r'$v_1$')
                # ax_5_lambda[0].set_ylim(-0.2, 0.2)
                ax_5_lambda[0].set_xlim(pt_lo_forplot, pt_hi_forplot)
                ax_5_lambda[0].annotate(r'y<0', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
                ax_5_lambda[0].annotate(r'$\sqrt{s_{NN}}=$' + f'{kwargs["energy"].replace('p', '.').replace('GeV', ' GeV')}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=20)
                ax_5_lambda[0].legend(loc='upper center')

    v1_pt_merged = initialize_merged_dict()
    v1_pt_lambdabar_merged_combined = initialize_merged_dict()
    for cen, data in lambdabar_v1_pt_east.items():
        if cen_mask[cen - 1] == False:
            continue
        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambdabar_v1_pt_east for centrality {cen}.')
        v1 = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_count = np.where(bool_nan, 0, data['counts'])
        v1_err = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        for group in v1_pt_merged.keys():
            if str(cen) in group:
                v1_pt_merged[group]['measurement'] += Measurement(v1_final)
                v1_pt_lambdabar_merged_combined[group]['measurement'] += Measurement(v1_final)
                v1_pt_merged[group]['count'] += v1_count
                v1_pt_lambdabar_merged_combined[group]['count'] += v1_count
            if group.endswith(str(cen)):
                m = v1_pt_merged[group]['measurement'].get_measurement()
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_merged[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                print(pt_bin_centers_shifted)
                v1_pt_merged[group]['x'] = pt_bin_centers_shifted
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_lambdabar_merged_combined[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_lambdabar_merged_combined[group]['x'] = pt_bin_centers_shifted
                ax_5_lambdabar[0].errorbar(pt_bin_centers, unumpy.nominal_values(m),
                                xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(m),
                                fmt='o', capsize=2, label=merged_cen_labels[group]
                                )
                ax_5_lambdabar[0].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
                ax_5_lambdabar[0].set_xlabel(r'$p_T$')
                ax_5_lambdabar[0].set_ylabel(r'$v_1$')
                # ax_5_lambdabar[0].set_ylim(-0.2, 0.2)
                ax_5_lambdabar[0].set_xlim(pt_lo_forplot, pt_hi_forplot)
                ax_5_lambdabar[0].annotate(r'y<0', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
                ax_5_lambdabar[0].annotate(r'$\sqrt{s_{NN}}=$' + f'{kwargs["energy"].replace('p', '.').replace('GeV', ' GeV')}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=20)
                ax_5_lambdabar[0].legend(loc='upper center')

    v1_pt_merged = initialize_merged_dict()
    for cen, data in lambda_v1_pt_west.items():
        if cen_mask[cen - 1] == False:
            continue
        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambda_v1_pt_west for centrality {cen}.')
        v1 = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_count = np.where(bool_nan, 0, data['counts'])
        v1_err = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        for group in v1_pt_merged.keys():
            if str(cen) in group:
                v1_pt_merged[group]['measurement'] += Measurement(v1_final)
                v1_pt_lambda_merged_combined[group]['measurement'] += Measurement(v1_final)
                v1_pt_merged[group]['count'] += v1_count
                v1_pt_lambda_merged_combined[group]['count'] += v1_count
            if group.endswith(str(cen)):
                m = v1_pt_merged[group]['measurement'].get_measurement()
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_merged[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_merged[group]['x'] = pt_bin_centers_shifted
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_lambda_merged_combined[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_lambda_merged_combined[group]['x'] = pt_bin_centers_shifted
                ax_5_lambda[1].errorbar(pt_bin_centers, unumpy.nominal_values(m),
                                xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(m),
                                fmt='o', capsize=2, label=merged_cen_labels[group]
                                )
                ax_5_lambda[1].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
                ax_5_lambda[1].set_xlabel(r'$p_T$')
                ax_5_lambda[1].set_ylabel(r'$v_1$')
                # ax_5_lambda[1].set_ylim(-0.2, 0.2)
                ax_5_lambda[1].set_xlim(pt_lo_forplot, pt_hi_forplot)
                ax_5_lambda[1].annotate(r'y>0', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
                ax_5_lambda[1].legend(loc='upper center')

    v1_pt_merged = initialize_merged_dict()
    for cen, data in lambdabar_v1_pt_west.items():
        if cen_mask[cen - 1] == False:
            continue
        bool_nan = np.isnan(data['values'].astype(float))
        if np.any(bool_nan):
            print(f'Warning: NaN values found in lambdabar_v1_pt_west for centrality {cen}.')
        v1 = np.where(bool_nan, 0.0, data['values'].astype(float))
        v1_count = np.where(bool_nan, 0, data['counts'])
        v1_err = np.where(bool_nan, np.inf, data['errors'].astype(float))
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        for group in v1_pt_merged.keys():
            if str(cen) in group:
                v1_pt_merged[group]['measurement'] += Measurement(v1_final)
                v1_pt_lambdabar_merged_combined[group]['measurement'] += Measurement(v1_final)
                v1_pt_merged[group]['count'] += v1_count
                v1_pt_lambdabar_merged_combined[group]['count'] += v1_count
            if group.endswith(str(cen)):
                m = v1_pt_merged[group]['measurement'].get_measurement()
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_merged[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_merged[group]['x'] = pt_bin_centers_shifted
                # pt_bin_centers_shifted = BinCenterFinder(pt_bin_edges, v1_pt_lambdabar_merged_combined[group]['count'], fit_order=3)
                pt_bin_centers_shifted = pt_bin_centers
                v1_pt_lambdabar_merged_combined[group]['x'] = pt_bin_centers_shifted
                ax_5_lambdabar[1].errorbar(pt_bin_centers, unumpy.nominal_values(m),
                                xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(m),
                                fmt='o', capsize=2, label=merged_cen_labels[group]
                                )
                ax_5_lambdabar[1].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
                ax_5_lambdabar[1].set_xlabel(r'$p_T$')
                ax_5_lambdabar[1].set_ylabel(r'$v_1$')
                # ax_5_lambdabar[1].set_ylim(-0.2, 0.2)
                ax_5_lambdabar[1].set_xlim(pt_lo_forplot, pt_hi_forplot)
                ax_5_lambdabar[1].annotate(r'y>0', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
                ax_5_lambdabar[1].legend(loc='upper center')
    
    # combined east and west
    for group in v1_pt_merged.keys():
        ml = v1_pt_lambda_merged_combined[group]['measurement'].get_measurement()
        ax_5_lambda[2].errorbar(pt_bin_centers, unumpy.nominal_values(ml),
                        xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(ml),
                        fmt='o', capsize=2, label=merged_cen_labels[group]
                        )
        mlb = v1_pt_lambdabar_merged_combined[group]['measurement'].get_measurement()
        ax_5_lambdabar[2].errorbar(pt_bin_centers, unumpy.nominal_values(mlb),
                        xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(mlb),
                        fmt='o', capsize=2, label=merged_cen_labels[group]
                        )
        delta = ml - mlb
        ax_5_lambda[3].errorbar(pt_bin_centers, unumpy.nominal_values(delta),
                        xerr=pt_bin_widths / 2, yerr=unumpy.std_devs(delta),
                        fmt='o', capsize=2, label=merged_cen_labels[group]
                        )
        
        ax_5_lambda[2].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
        ax_5_lambda[2].set_xlabel(r'$p_T$')
        ax_5_lambda[2].set_ylabel(r'$v_1$')
        # ax_5_lambda[2].set_ylim(-0.2, 0.2)
        ax_5_lambda[2].set_xlim(pt_lo_forplot, pt_hi_forplot)
        ax_5_lambda[2].annotate(r'Combined', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        ax_5_lambda[2].legend(loc='upper center')
        ax_5_lambdabar[2].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
        ax_5_lambdabar[2].set_xlabel(r'$p_T$')
        ax_5_lambdabar[2].set_ylabel(r'$v_1$')
        # ax_5_lambdabar[0].set_ylim(-0.2, 0.2)
        ax_5_lambdabar[2].set_xlim(pt_lo_forplot, pt_hi_forplot)
        ax_5_lambdabar[2].annotate(r'Combined', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        ax_5_lambdabar[2].legend(loc='upper center')
        ax_5_lambda[3].hlines(0., pt_lo_forplot, pt_hi_forplot, color='k', linestyle='--')
        ax_5_lambda[3].set_xlabel(r'$p_T$')
        ax_5_lambda[3].set_ylabel(r'$\Delta v_1$')
        # ax_5_lambda[3].set_ylim(-0.2, 0.2)
        ax_5_lambda[3].set_xlim(pt_lo_forplot, pt_hi_forplot)
        ax_5_lambda[3].annotate(r'Lambda-Lambdabars', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        ax_5_lambda[3].legend(loc='upper center')

    # merged centralities
    fig_4, ax_4 = plt.subplots(2, 2, figsize=(16, 12))
    ax_4 = ax_4.flatten()
    merged_centralities = {'010': [8,9], '1040': [5,6,7], '4080': [1,2,3,4], '5080': [1,2,3]}
    merged_latex = {'010': '0-10%', '1040': '10-40%', '4080': '40-80%', '5080': '50-80%'}
    v1_merged_lambda = {}
    for cen, data in lambda_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)
        
        v1 = np.nan_to_num(data['values'])
        v1_err = np.nan_to_num(data['errors'])
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy
        for key, value in merged_centralities.items():
            if cen in value:
                if key not in v1_merged_lambda.keys():
                    print('v1_final[0]:', v1_final[0])
                    print('type(v1_final[0]):', type(v1_final[0]))
                    v1_merged_lambda[key] = {'x': ybin_good, 'y': Measurement(v1_final)}
                else:
                    v1_merged_lambda[key]['y'] = v1_merged_lambda[key]['y'] + Measurement(v1_final)
    v1_merged_lambdabar = {}
    for cen, data in lambdabar_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)
        
        v1 = np.nan_to_num(data['values'])
        v1_err = np.nan_to_num(data['errors'])
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy
        for key, value in merged_centralities.items():
            if cen in value:
                if key not in v1_merged_lambdabar.keys():
                    v1_merged_lambdabar[key] = {'x': ybin_good, 'y': Measurement(v1_final)}
                else:
                    v1_merged_lambdabar[key]['y'] = v1_merged_lambdabar[key]['y'] + Measurement(v1_final)

    dv1dy_lambda_merged = {}
    dv1dy_lambdabar_merged = {}
    dv1dy_deltalambda_merged = {}
    for i, (key, value) in enumerate(v1_merged_lambda.items()):
        m = value['y'].get_measurement()
        _xnom = unumpy.nominal_values(value['x'])
        _fit_mask = np.abs(_xnom) < float(config['y_cut'])
        ax_4[i].errorbar(_xnom[_fit_mask], unumpy.nominal_values(m[_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(m[_fit_mask]), color='C0',
                         fmt='o', capsize=2, label=r'$\Lambda$')
        ax_4[i].errorbar(_xnom[~_fit_mask], unumpy.nominal_values(m[~_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(m[~_fit_mask]), color='C0',
                         fmt='o', capsize=2, markerfacecolor='white', label='_nolegend_')
        popt, perr, rchi2 = fit(value['x'], m, method=method, range=range)
        popt_linear, perr_linear, rchi2_linear = fit(value['x'], m, method=1, range=range)
        d3 = 0
        d3_err = 0
        if method == 3:
            d3 = popt[1]
            d3_err = perr[1]
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_linear[0]
                perr[0] = perr_linear[0]
                popt[1] = 0
                perr[1] = 0
        if key not in dv1dy_lambda_merged.keys():
            dv1dy_lambda_merged[key] = {}
        dv1dy_lambda_merged[key]['value'] = popt[0]
        dv1dy_lambda_merged[key]['error'] = perr[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_4[i].plot(x_fit, fun(popt, x_fit), '-', color='C0')
        ax_4[i].hlines(0., -1., 1., color='k', linestyle='--')
        ax_4[i].annotate(merged_latex[key], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        if method == 3:
            ax_4[i].annotate(r'$\Lambda$' + f' d3 = {ufloat(d3, d3_err)}', xy=(0.3, 0.35), xycoords='axes fraction', fontsize=12)
            # ax_4[i].annotate(f'rchi2 = {rchi2:.2f}, rchi2_linear = {rchi2_linear:.2f}', xy=(0.05, 0.65), xycoords='axes fraction', fontsize=12)
        ax_4[i].set_xlabel(r'$y$')
        ax_4[i].set_ylabel(r'$v_1$')
        ax_4[i].legend()
    for i, (key, value) in enumerate(v1_merged_lambdabar.items()):
        m = value['y'].get_measurement()
        _xnom = unumpy.nominal_values(value['x'])
        _fit_mask = np.abs(_xnom) < float(config['y_cut'])
        ax_4[i].errorbar(_xnom[_fit_mask], unumpy.nominal_values(m[_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(m[_fit_mask]), color='C1',
                         fmt='o', capsize=2, label=r'$\bar{\Lambda}$')
        ax_4[i].errorbar(_xnom[~_fit_mask], unumpy.nominal_values(m[~_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(m[~_fit_mask]), color='C1',
                         fmt='o', capsize=2, markerfacecolor='white', label='_nolegend_')
        popt, perr, rchi2 = fit(value['x'], m, method=method, range=range)
        popt_linear, perr_linear, rchi2_linear = fit(value['x'], m, method=1, range=range)
        d3 = 0
        d3_err = 0
        if method == 3:
            d3 = popt[1]
            d3_err = perr[1]
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_linear[0]
                perr[0] = perr_linear[0]
                popt[1] = 0
                perr[1] = 0
        if key not in dv1dy_lambdabar_merged.keys():
            dv1dy_lambdabar_merged[key] = {}
        dv1dy_lambdabar_merged[key]['value'] = popt[0]
        dv1dy_lambdabar_merged[key]['error'] = perr[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_4[i].plot(x_fit, fun(popt, x_fit), '-', color='C1')
        ax_4[i].hlines(0., -1., 1., color='k', linestyle='--')
        ax_4[i].annotate(merged_latex[key], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        if method == 3:
            ax_4[i].annotate(r'$\bar{\Lambda}$' + f' d3 = {ufloat(d3, d3_err)}', xy=(0.3, 0.25), xycoords='axes fraction', fontsize=12)
            # ax_4[i].annotate(f'rchi2 = {rchi2:.2f}, rchi2_linear = {rchi2_linear:.2f}', xy=(0.05, 0.65), xycoords='axes fraction', fontsize=12)
        ax_4[i].set_xlabel(r'$y$')
        ax_4[i].set_ylabel(r'$v_1$')
        ax_4[i].legend()
    for i, (key, value) in enumerate(v1_merged_lambda.items()):
        m_l = value['y'].get_measurement()
        m_lb = v1_merged_lambdabar[key]['y'].get_measurement()
        delta = m_l - m_lb
        _xnom = unumpy.nominal_values(value['x'])
        _fit_mask = np.abs(_xnom) < float(config['y_cut'])
        ax_4[i].errorbar(_xnom[_fit_mask], unumpy.nominal_values(delta[_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(delta[_fit_mask]), color='C2',
                         fmt='o', capsize=2, label=r'$\Delta\Lambda$')
        ax_4[i].errorbar(_xnom[~_fit_mask], unumpy.nominal_values(delta[~_fit_mask]),
                         xerr=0, yerr=unumpy.std_devs(delta[~_fit_mask]), color='C2',
                         fmt='o', capsize=2, markerfacecolor='white', label='_nolegend_')
        popt, perr, rchi2 = fit(value['x'], delta, method=method, range=range)
        popt_linear, perr_linear, rchi2_linear = fit(value['x'], delta, method=1, range=range)
        d3 = 0
        d3_err = 0
        if method == 3:
            d3 = popt[1]
            d3_err = perr[1]
            if abs(popt[1]) < perr[1]:
                popt[0] = popt_linear[0]
                perr[0] = perr_linear[0]
                popt[1] = 0
                perr[1] = 0
        if key not in dv1dy_deltalambda_merged.keys():
            dv1dy_deltalambda_merged[key] = {}
        dv1dy_deltalambda_merged[key]['value'] = popt[0]
        dv1dy_deltalambda_merged[key]['error'] = perr[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_4[i].plot(x_fit, fun(popt, x_fit), '-', color='C2')
        ax_4[i].hlines(0., -1., 1., color='k', linestyle='--')
        ax_4[i].annotate(merged_latex[key], xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)
        if method == 3:
            ax_4[i].annotate(r'$\Delta\Lambda$' + f' d3 = {ufloat(d3, d3_err)}', xy=(0.3, 0.15), xycoords='axes fraction', fontsize=12)
            # ax_4[i].annotate(f'rchi2 = {rchi2:.2f}, rchi2_linear = {rchi2_linear:.2f}', xy=(0.05, 0.65), xycoords='axes fraction', fontsize=12)
        ax_4[i].set_xlabel(r'$y$')
        ax_4[i].set_ylabel(r'$v_1$')
        ax_4[i].legend()

    # delta v1 and a1 slope
    # delta_dv1dy = dv1dy_lambda - dv1dy_lambdabar
    # delta_dv1dy_err = np.sqrt(dv1dy_lambda_err ** 2 + dv1dy_lambdabar_err ** 2)
    delta_dv1dy = dv1dy_deltalambda
    delta_dv1dy_err = dv1dy_deltalambda_err
    if not no_a1:
        delta_da1dy = da1dy_lambda - da1dy_lambdabar
        delta_da1dy_err = np.sqrt(da1dy_lambda_err ** 2 + da1dy_lambdabar_err ** 2)
    fig_final, ax_final = plt.subplots(1,1, figsize=(8, 6))
    ax_final.errorbar(centralities[cen_mask] - 0.5, delta_dv1dy[cen_mask], fmt='o', yerr=delta_dv1dy_err[cen_mask], ls='none', label=r'$v_1$')
    if not no_a1:
        ax_final.errorbar(centralities[cen_mask] + 0.5, delta_da1dy[cen_mask], fmt='o', yerr=delta_da1dy_err[cen_mask], ls='none', label=r'$a_1$')
    ax_final.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_final.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    ax_final.set_ylabel(r'$\Delta dv_1/dy$' + r'($\Delta da_1/dy$)', loc='top', fontsize=18)
    if 'yrange' in kwargs.keys():
        ax_final.set_ylim(*kwargs['yrange'])
    else:
        ax_final.set_ylim(-0.2, 0.15)
    ax_final.annotate(r'$\Lambda^{0}-\bar{\Lambda}^{0}$', xy=(0.1, 0.2), xycoords='axes fraction', fontsize=25)
    ax_final.annotate(kwargs['energy'].replace('p', '.'), xy=(0.1, 0.1), xycoords='axes fraction', fontsize=25)
    ax_final.legend(loc='best', fontsize=12)
    plt.tight_layout()
                
    ### convert the output to panda and csv
    data = pd.DataFrame({'centrality': centralities, 
                         'dv1dy_lambda': dv1dy_lambda, 'dv1dy_lambda_err': dv1dy_lambda_err,
                         'dv1dy_lambdabar': dv1dy_lambdabar, 'dv1dy_lambdabar_err': dv1dy_lambdabar_err,
                         'd3v1dy3_lambda': d3v1dy3_lambda, 'd3v1dy3_lambda_err': d3v1dy3_lambda_err,
                         'd3v1dy3_lambdabar': d3v1dy3_lambdabar, 'd3v1dy3_lambdabar_err': d3v1dy3_lambdabar_err,
                         'delta_dv1dy': delta_dv1dy, 'delta_dv1dy_err': delta_dv1dy_err,
                        #  'da1dy_lambda': da1dy_lambda, 'da1dy_lambda_err': da1dy_lambda_err,
                        #  'da1dy_lambdabar': da1dy_lambdabar, 'da1dy_lambdabar_err': da1dy_lambdabar_err,
                        #  'delta_da1dy': delta_da1dy, 'delta_da1dy_err': delta_da1dy_err
                         })
    data.to_csv(output, index=False)

    ### for merged centralities
    data_merged_4080_lambda = pd.DataFrame({'y': unumpy.nominal_values(v1_merged_lambda['4080']['x']).round(decimals=2),
                                            'v1': unumpy.nominal_values(v1_merged_lambda['4080']['y'].get_measurement()),
                                            'v1_err': unumpy.std_devs(v1_merged_lambda['4080']['y'].get_measurement())})
    data_merged_4080_lambdabar = pd.DataFrame({'y': unumpy.nominal_values(v1_merged_lambdabar['4080']['x']).round(decimals=2),
                                            'v1': unumpy.nominal_values(v1_merged_lambdabar['4080']['y'].get_measurement()),
                                            'v1_err': unumpy.std_devs(v1_merged_lambdabar['4080']['y'].get_measurement())})
    data_merged_4080_lambda.to_csv(output.replace('.txt', '_4080_lambda.txt'), index=False)
    data_merged_4080_lambdabar.to_csv(output.replace('.txt', '_4080_lambdabar.txt'), index=False)

    ### for v1 pt
    for group in v1_pt_lambda_merged_combined.keys():
        m = v1_pt_lambda_merged_combined[group]['measurement'].get_measurement()
        data_v1_pt_lambda = pd.DataFrame({'pT': pt_bin_centers.round(decimals=2),
                                          'v1': unumpy.nominal_values(m),
                                          'v1_err': unumpy.std_devs(m)})
        data_v1_pt_lambda.to_csv(output.replace('.txt', f'_{merged_cen_titles[group]}_lambda_pt.txt'), index=False)
    for group in v1_pt_lambdabar_merged_combined.keys():
        m = v1_pt_lambdabar_merged_combined[group]['measurement'].get_measurement()
        data_v1_pt_lambdabar = pd.DataFrame({'pT': pt_bin_centers.round(decimals=2),
                                          'v1': unumpy.nominal_values(m),
                                          'v1_err': unumpy.std_devs(m)})
        data_v1_pt_lambdabar.to_csv(output.replace('.txt', f'_{merged_cen_titles[group]}_lambdabar_pt.txt'), index=False)

    # combine peripheral with error
    # delta_dv1dy_peri = np.sum(delta_dv1dy[:3] * (1 / delta_dv1dy_err[:3] ** 2)) / np.sum(1 / delta_dv1dy_err[:3] ** 2)
    # print(delta_dv1dy_peri)
    # delta_dv1dy_peri_err = 1 / np.sqrt(np.sum(1 / delta_dv1dy_err[:3] ** 2))
    # ax_final.add_patch(Rectangle((50, delta_dv1dy_peri - delta_dv1dy_peri_err),
    #                              30, 2 * delta_dv1dy_peri_err,
    #                              color='C0', alpha=0.5, label=r'$v_1$ 40-80%, statistical only'))

    # comparing with Aditya results
    fig_final_2, ax_final_2 = plt.subplots(1, 1, figsize=(8, 6))
    combo = proton_v1_slopes - antiproton_v1_slopes - kplus_v1_slopes + kminus_v1_slopes
    combo_PT_1 = piplus_v1_slopes + piminus_v1_slopes - antiproton_v1_slopes + kminus_v1_slopes
    combo_PT_2 = piplus_v1_slopes + piminus_v1_slopes - proton_v1_slopes + kplus_v1_slopes
    combo_PT_1_mod = proton_v1_slopes + kminus_v1_slopes - 0.5 * (piplus_v1_slopes + piminus_v1_slopes)
    combo_PT_1_asso = proton_v1_slopes - kplus_v1_slopes
    combo_PT_2_mod = antiproton_v1_slopes + kplus_v1_slopes - 0.5 * (piplus_v1_slopes + piminus_v1_slopes)
    delta_u = 1./3. * (proton_v1_slopes - antiproton_v1_slopes + piplus_v1_slopes - piminus_v1_slopes)
    delta_d = 1./3. * (proton_v1_slopes - antiproton_v1_slopes) - 2./3. * (piplus_v1_slopes - piminus_v1_slopes)
    delta_s = delta_u - (kplus_v1_slopes - kminus_v1_slopes)
    proton = proton_v1_slopes - antiproton_v1_slopes
    kaon = kplus_v1_slopes - kminus_v1_slopes
    ax_final_2.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_final_2.errorbar(centralities[cen_mask] - 0.5, delta_dv1dy[cen_mask], fmt='o', yerr=delta_dv1dy_err[cen_mask], ls='none',
                        label=r'$\Lambda^{0}-\bar{\Lambda}^{0}$')
    ax_final_2.errorbar(centralities[cen_mask] + 0.5, unumpy.nominal_values(combo[cen_mask]), fmt='o', yerr=unumpy.std_devs(combo[cen_mask]), ls='none',
                        label=r'$(p-\bar{p})-(K^{+}-K^{-})$')
    ax_final_2.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    ax_final_2.set_ylabel(r'$\Delta dv_1/dy$', loc='top', fontsize=18)
    if 'yrange' in kwargs.keys():
        ax_final_2.set_ylim(*kwargs['yrange'])
    else:
        ax_final_2.set_ylim(-0.2, 0.15)
    ax_final_2.annotate(kwargs['energy'].replace('p', '.'), xy=(0.1, 0.1), xycoords='axes fraction', fontsize=25)
    ax_final_2.legend(loc='upper right', fontsize=12)

    # rebin pt and y dict
    for d in [v1_pt_lambda_merged_combined, v1_pt_lambdabar_merged_combined]:
        for cen_group in d.keys():
            temp = SimpleProfile(unumpy.nominal_values(d[cen_group]['measurement'].get_measurement()),
                                 d[cen_group]['count'],
                                 unumpy.std_devs(d[cen_group]['measurement'].get_measurement())*np.sqrt(d[cen_group]['count']), # convert standard error of the mean to standard deviation
                                 d[cen_group]['x'], use_edges=False)
            print(len(temp.bin_centers()))
            temp.Rebin(2)
            d[cen_group]['x'] = temp.bin_centers()
            d[cen_group]['measurement'] = Measurement(unumpy.uarray(temp.values(), temp.errors()))
            d[cen_group]['count'] = temp.counts()

    data_dict = {
                # default stuff
                'x': centralities[cen_mask], 'y': delta_dv1dy[cen_mask], 'yerr': delta_dv1dy_err[cen_mask],
                 #'a1': da1dy_lambda[cen_mask], 'a1_err': da1dy_lambda_err[cen_mask],
                'lambda': dv1dy_lambda[cen_mask], 'lambda_err': dv1dy_lambda_err[cen_mask],
                'lambdabar': dv1dy_lambdabar[cen_mask], 'lambdabar_err': dv1dy_lambdabar_err[cen_mask],

                # merged centrality
                'y_merged': unumpy.nominal_values(v1_merged_lambda['4080']['x']).round(decimals=2),
                'v1_merged_lambda_4080': unumpy.nominal_values(v1_merged_lambda['4080']['y'].get_measurement()),
                'v1_merged_lambda_4080_err': unumpy.std_devs(v1_merged_lambda['4080']['y'].get_measurement()),
                'v1_merged_lambdabar_4080': unumpy.nominal_values(v1_merged_lambdabar['4080']['y'].get_measurement()),
                'v1_merged_lambdabar_4080_err': unumpy.std_devs(v1_merged_lambdabar['4080']['y'].get_measurement()),

                # v1 pt
                'v1_pt_lambda_4080': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['1234']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambda_merged_combined['1234']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['1234']['x'])
                },
                'v1_pt_lambda_1040': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['567']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambda_merged_combined['567']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['567']['x'])
                },
                'v1_pt_lambda_010': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['89']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambda_merged_combined['89']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['89']['x'])
                },
                'v1_pt_lambda_5080': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['123']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambda_merged_combined['123']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['123']['x'])
                },
                'v1_pt_lambdabar_4080': {
                    'value': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['1234']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambdabar_merged_combined['1234']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['1234']['x'])
                },
                'v1_pt_lambdabar_1040': {
                    'value': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['567']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambdabar_merged_combined['567']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['567']['x'])
                },
                'v1_pt_lambdabar_010': {
                    'value': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['89']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambdabar_merged_combined['89']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['89']['x'])
                },
                'v1_pt_lambdabar_5080': {
                    'value': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['123']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_pt_lambdabar_merged_combined['123']['measurement'].get_measurement()),
                    'pT': unumpy.nominal_values(v1_pt_lambdabar_merged_combined['123']['x'])
                },
                'v1_pt_delta_4080': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['1234']['measurement'].get_measurement()) - unumpy.nominal_values(v1_pt_lambdabar_merged_combined['1234']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_pt_lambda_merged_combined['1234']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_pt_lambdabar_merged_combined['1234']['measurement'].get_measurement())**2),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['1234']['x'])
                },
                'v1_pt_delta_1040': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['567']['measurement'].get_measurement()) - unumpy.nominal_values(v1_pt_lambdabar_merged_combined['567']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_pt_lambda_merged_combined['567']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_pt_lambdabar_merged_combined['567']['measurement'].get_measurement())**2),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['567']['x'])
                },
                'v1_pt_delta_010': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['89']['measurement'].get_measurement()) - unumpy.nominal_values(v1_pt_lambdabar_merged_combined['89']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_pt_lambda_merged_combined['89']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_pt_lambdabar_merged_combined['89']['measurement'].get_measurement())**2),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['89']['x'])
                },
                'v1_pt_delta_5080': {
                    'value': unumpy.nominal_values(v1_pt_lambda_merged_combined['123']['measurement'].get_measurement()) - unumpy.nominal_values(v1_pt_lambdabar_merged_combined['123']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_pt_lambda_merged_combined['123']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_pt_lambdabar_merged_combined['123']['measurement'].get_measurement())**2),
                    'pT': unumpy.nominal_values(v1_pt_lambda_merged_combined['123']['x'])
                },

                # v1 y
                'v1_y_lambda_4080': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['1234']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambda_merged['1234']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['1234']['x'])
                },
                'v1_y_lambdabar_4080': {
                    'value': unumpy.nominal_values(v1_y_lambdabar_merged['1234']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambdabar_merged['1234']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambdabar_merged['1234']['x'])
                },
                'v1_y_delta_4080': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['1234']['measurement'].get_measurement()) - unumpy.nominal_values(v1_y_lambdabar_merged['1234']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_y_lambda_merged['1234']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_y_lambdabar_merged['1234']['measurement'].get_measurement())**2),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['1234']['x'])
                },
                'v1_y_lambda_1040': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['567']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambda_merged['567']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['567']['x'])
                },
                'v1_y_lambdabar_1040': {
                    'value': unumpy.nominal_values(v1_y_lambdabar_merged['567']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambdabar_merged['567']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambdabar_merged['567']['x'])
                },
                'v1_y_delta_1040': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['567']['measurement'].get_measurement()) - unumpy.nominal_values(v1_y_lambdabar_merged['567']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_y_lambda_merged['567']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_y_lambdabar_merged['567']['measurement'].get_measurement())**2),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['567']['x'])
                },
                'v1_y_lambda_010': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['89']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambda_merged['89']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['89']['x'])
                },
                'v1_y_lambdabar_010': {
                    'value': unumpy.nominal_values(v1_y_lambdabar_merged['89']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambdabar_merged['89']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambdabar_merged['89']['x'])
                },
                'v1_y_delta_010': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['89']['measurement'].get_measurement()) - unumpy.nominal_values(v1_y_lambdabar_merged['89']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_y_lambda_merged['89']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_y_lambdabar_merged['89']['measurement'].get_measurement())**2),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['89']['x'])
                },
                'v1_y_lambda_5080': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['123']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambda_merged['123']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['123']['x'])
                },
                'v1_y_lambdabar_5080': {
                    'value': unumpy.nominal_values(v1_y_lambdabar_merged['123']['measurement'].get_measurement()),
                    'error': unumpy.std_devs(v1_y_lambdabar_merged['123']['measurement'].get_measurement()),
                    'y': unumpy.nominal_values(v1_y_lambdabar_merged['123']['x'])
                },
                'v1_y_delta_5080': {
                    'value': unumpy.nominal_values(v1_y_lambda_merged['123']['measurement'].get_measurement()) - unumpy.nominal_values(v1_y_lambdabar_merged['123']['measurement'].get_measurement()),
                    'error': np.sqrt(unumpy.std_devs(v1_y_lambda_merged['123']['measurement'].get_measurement())**2 + unumpy.std_devs(v1_y_lambdabar_merged['123']['measurement'].get_measurement())**2),
                    'y': unumpy.nominal_values(v1_y_lambda_merged['123']['x'])
                },

                # proton
                'p': unumpy.nominal_values(proton_v1_slopes[cen_mask]), 'p_err': unumpy.std_devs(proton_v1_slopes[cen_mask]),
                'pbar': unumpy.nominal_values(antiproton_v1_slopes[cen_mask]), 'pbar_err': unumpy.std_devs(antiproton_v1_slopes[cen_mask]),

                # thir order component
                'd3v1dy3_lambda': d3v1dy3_lambda[cen_mask], 'd3v1dy3_lambda_err': d3v1dy3_lambda_err[cen_mask],
                'd3v1dy3_lambdabar': d3v1dy3_lambdabar[cen_mask], 'd3v1dy3_lambdabar_err': d3v1dy3_lambdabar_err[cen_mask],
                'd3v1dy3_antiproton': unumpy.nominal_values(antiproton_d3v1dy3[cen_mask]), 'd3v1dy3_antiproton_err': unumpy.std_devs(antiproton_d3v1dy3[cen_mask]),
                'd3v1dy3_proton': unumpy.nominal_values(proton_d3v1dy3[cen_mask]), 'd3v1dy3_proton_err': unumpy.std_devs(proton_d3v1dy3[cen_mask]),
                'd3v1dy3_kminus': unumpy.nominal_values(kminus_d3v1dy3[cen_mask]), 'd3v1dy3_kminus_err': unumpy.std_devs(kminus_d3v1dy3[cen_mask]),
                'd3v1dy3_kplus': unumpy.nominal_values(kplus_d3v1dy3[cen_mask]), 'd3v1dy3_kplus_err': unumpy.std_devs(kplus_d3v1dy3[cen_mask]),
                'd3v1dy3_piplus': unumpy.nominal_values(piplus_d3v1dy3[cen_mask]), 'd3v1dy3_piplus_err': unumpy.std_devs(piplus_d3v1dy3[cen_mask]),
                'd3v1dy3_piminus': unumpy.nominal_values(piminus_d3v1dy3[cen_mask]), 'd3v1dy3_piminus_err': unumpy.std_devs(piminus_d3v1dy3[cen_mask]),
                
                # coalescence
                'combo_PT_1': unumpy.nominal_values(combo_PT_1[cen_mask]), 'combo_PT_1_err': unumpy.std_devs(combo_PT_1[cen_mask]),
                'combo_PT_2': unumpy.nominal_values(combo_PT_2[cen_mask]), 'combo_PT_2_err': unumpy.std_devs(combo_PT_2[cen_mask]),
                'combo_PT_1_mod': unumpy.nominal_values(combo_PT_1_mod[cen_mask]), 'combo_PT_1_mod_err': unumpy.std_devs(combo_PT_1_mod[cen_mask]),
                'combo_PT_1_asso': unumpy.nominal_values(combo_PT_1_asso[cen_mask]), 'combo_PT_1_asso_err': unumpy.std_devs(combo_PT_1_asso[cen_mask]),
                'combo_PT_2_mod': unumpy.nominal_values(combo_PT_2_mod[cen_mask]), 'combo_PT_2_mod_err': unumpy.std_devs(combo_PT_2_mod[cen_mask]),
                'proton': unumpy.nominal_values(proton[cen_mask]), 'proton_err': unumpy.std_devs(proton[cen_mask]),
                'kaon': unumpy.nominal_values(kaon[cen_mask]), 'kaon_err': unumpy.std_devs(kaon[cen_mask]),
                'dv1dy_lambda_010': dv1dy_lambda_merged['010'], 'dv1dy_lambda_1040': dv1dy_lambda_merged['1040'], 'dv1dy_lambda_4080': dv1dy_lambda_merged['4080'],
                'dv1dy_lambdabar_010': dv1dy_lambdabar_merged['010'], 'dv1dy_lambdabar_1040': dv1dy_lambdabar_merged['1040'], 'dv1dy_lambdabar_4080': dv1dy_lambdabar_merged['4080'],
                'dv1dy_deltalambda_010': dv1dy_deltalambda_merged['010'], 'dv1dy_deltalambda_1040': dv1dy_deltalambda_merged['1040'], 'dv1dy_deltalambda_4080': dv1dy_deltalambda_merged['4080'],
                'dv1dy_lambda_5080': dv1dy_lambda_merged['5080'], 'dv1dy_lambdabar_5080': dv1dy_lambdabar_merged['5080'], 'dv1dy_deltalambda_5080': dv1dy_deltalambda_merged['5080'],
                # 'dv1dy_proton_1040': ufloat(df_protons['delta_v1'][10], df_protons['delta_v1_err'][10]),
                # 'dv1dy_proton_4080': ufloat(df_protons['delta_v1'][9], df_protons['delta_v1_err'][9]),
                # 'dv1dy_kaon_1040': ufloat(df_kaons['delta_v1'][10], df_kaons['delta_v1_err'][10]),
                # 'dv1dy_kaon_4080': ufloat(df_kaons['delta_v1'][9], df_kaons['delta_v1_err'][9]),
                'combo': unumpy.nominal_values(combo[cen_mask]), 'combo_err': unumpy.std_devs(combo[cen_mask]),
                'combo2': unumpy.nominal_values(proton[cen_mask]), 'combo2_err': unumpy.std_devs(proton[cen_mask]),
                'delta_u': unumpy.nominal_values(delta_u[cen_mask]), 'delta_u_err': unumpy.std_devs(delta_u[cen_mask]),
                'delta_d': unumpy.nominal_values(delta_d[cen_mask]), 'delta_d_err': unumpy.std_devs(delta_d[cen_mask]),
                'delta_s': unumpy.nominal_values(delta_s[cen_mask]), 'delta_s_err': unumpy.std_devs(delta_s[cen_mask])}
    with open(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/paper_yaml/dv1dy_coal_{kwargs["energy"]}.yaml', 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)

    # comparing with Prithwish suggestion
    # fig_comp_1, ax_comp_1 = plt.subplots(1, 1, figsize=(8, 6))
    # # combo = piplus_v1_slopes + piminus_v1_slopes + kplus_v1_slopes - proton_v1_slopes
    # combo = kminus_v1_slopes + 1./3. * antiproton_v1_slopes
    # ax_comp_1.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    # ax_comp_1.errorbar(centralities - 0.5, dv1dy_lambdabar, fmt='o', yerr=dv1dy_lambdabar_err, ls='none',
    #                     label=r'$\bar{\Lambda}^{0}$')
    # ax_comp_1.errorbar(centralities + 0.5, unumpy.nominal_values(combo), fmt='o', yerr=unumpy.std_devs(combo), ls='none',
    #                     label=r'$K^{-}+rac{1}{3}\bar{p}$')
    # ax_comp_1.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    # ax_comp_1.set_ylabel(r'$dv_1/dy$', loc='top', fontsize=18)
    # ax_comp_1.legend(loc='upper right', fontsize=12)
    #
    # fig_comp_2, ax_comp_2 = plt.subplots(1, 1, figsize=(8, 6))
    # combo = piplus_v1_slopes + piminus_v1_slopes + kminus_v1_slopes - antiproton_v1_slopes
    # ax_comp_2.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    # ax_comp_2.errorbar(centralities - 0.5, dv1dy_lambda, fmt='o', yerr=dv1dy_lambda_err, ls='none',
    #                     label=r'$\Lambda^{0}$')
    # ax_comp_2.errorbar(centralities + 0.5, unumpy.nominal_values(combo), fmt='o', yerr=unumpy.std_devs(combo), ls='none',
    #                     label=r'$\pi^{+}+\pi^{-}+K^{-}-\bar{p}$')
    # ax_comp_2.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    # ax_comp_2.set_ylabel(r'$dv_1/dy$', loc='top', fontsize=18)
    # ax_comp_2.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    # plt.show()
    fig_res.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/resolution_{kwargs["energy"]}.pdf')
    fig_piKp.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_piKp_{kwargs["energy"]}.pdf')
    fig_1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_{kwargs["energy"]}.pdf')
    fig_delta.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/delta_v1_cen_{kwargs["energy"]}.pdf')
    if not no_a1:
        fig_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/a1_cen_{kwargs["energy"]}.pdf')
        fig_3_a1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/da1dy_cen_{kwargs["energy"]}.pdf')
        fig_delta_a1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/delta_a1_cen_{kwargs["energy"]}.pdf')
    else:
        # if a1 is not available, save empty plots with just the title for the sake of uniformity in the paper
        fig_blank, ax_blank = plt.subplots(figsize=(8, 6))
        ax_blank.text(0.5, 0.5, 'a1 not available', horizontalalignment='center', verticalalignment='center', fontsize=20)
        ax_blank.axis('off')
        fig_blank.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/a1_cen_{kwargs["energy"]}.pdf')
        fig_blank.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/da1dy_cen_{kwargs["energy"]}.pdf')
        fig_blank.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/delta_a1_cen_{kwargs["energy"]}.pdf')
    fig_4.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_merged_{kwargs["energy"]}.pdf')
    fig_5_lambda.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_pt_{kwargs["energy"]}_lambda.pdf')
    fig_5_lambdabar.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_pt_{kwargs["energy"]}_lambdabar.pdf')
    fig_final.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1a1dy_{kwargs["energy"]}.pdf')
    fig_final_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1dy_coal_{kwargs["energy"]}.pdf')

    # also png
    fig_res.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/resolution_{kwargs["energy"]}.png')
    fig_piKp.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_piKp_{kwargs["energy"]}.png')
    fig_1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_{kwargs["energy"]}.png')
    fig_delta.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/delta_v1_cen_{kwargs["energy"]}.png')
    if not no_a1:
        fig_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/a1_cen_{kwargs["energy"]}.png')
        fig_3_a1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/da1dy_cen_{kwargs["energy"]}.png')
        fig_delta_a1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/delta_a1_cen_{kwargs["energy"]}.png')
    fig_4.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_merged_{kwargs["energy"]}.png')
    fig_5_lambda.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_pt_{kwargs["energy"]}_lambda.png')
    fig_5_lambdabar.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_pt_{kwargs["energy"]}_lambdabar.png')
    fig_final.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1a1dy_{kwargs["energy"]}.png')
    fig_final_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1dy_coal_{kwargs["energy"]}.png')


def func_wrapper(params, x):
    return params[0] * x


def fit_odr(x, y):
    model = odr.Model(func_wrapper)
    data = odr.RealData(unumpy.nominal_values(x), unumpy.nominal_values(y),
                        sx=unumpy.std_devs(x), sy=unumpy.std_devs(y))
    odr_instance = odr.ODR(data, model, beta0=[0.], maxit=100)
    odr_result = odr_instance.run()
    return odr_result.beta, odr_result.sd_beta[0]


def func_wrapper_3rd(params, x):
    return params[0] * x + params[1] * x ** 3


def fit_odr_3rd(x, y):
    model = odr.Model(func_wrapper_3rd)
    data = odr.RealData(unumpy.nominal_values(x), unumpy.nominal_values(y),
                        sx=unumpy.std_devs(x), sy=unumpy.std_devs(y))
    odr_instance = odr.ODR(data, model, beta0=[0., 1.], maxit=100)
    odr_result = odr_instance.run()
    return odr_result.beta, odr_result.sd_beta[0]


def func(x, a):
    return a * x


def func_3rd(x, a, b):
    return a * x + b * x ** 3


def fit_iminuit(x, y, verbose=False):
    c = cost.LeastSquares(unumpy.nominal_values(x), unumpy.nominal_values(y), unumpy.std_devs(y), func)
    m = Minuit(c, a=0)
    if verbose:
        print(m.migrad())
    else:
        m.migrad()
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    return [m.values['a']], [m.errors['a']], m.fmin.reduced_chi2


def fit_iminuit_3rd(x, y, verbose=False):
    c = cost.LeastSquares(unumpy.nominal_values(x), unumpy.nominal_values(y), unumpy.std_devs(y), func_3rd)
    m = Minuit(c, a=0, b=0.01)
    if verbose:
        print(m.migrad())
    else:
        m.migrad()
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    return [m.values['a'],m.values['b']], [m.errors['a'], m.errors['b']], m.fmin.reduced_chi2


def fit_curve_fit(x, y):
    popt, pcov = curve_fit(func, xdata=unumpy.nominal_values(x), ydata=unumpy.nominal_values(y),
                           p0=[0.], sigma=unumpy.std_devs(y))
    return popt, np.sqrt(pcov[0][0])


def fit_curve_fit_3rd(x, y):
    popt, pcov = curve_fit(func_3rd, xdata=unumpy.nominal_values(x), ydata=unumpy.nominal_values(y),
                           p0=[0, 0.01], sigma=unumpy.std_devs(y))
    # check if fit converges, if not throw an error
    if np.isinf(pcov).any():
        raise RuntimeError('Fit did not converge')    
    return popt, np.sqrt(pcov[0][0])


def fit(x, y, method, range, verbose=False):
    fit = fit_iminuit
    if method == 3:
        fit = fit_iminuit_3rd
    if range == 'half': # positive half
        mask = (x > 0) & (x < float(config['y_cut']))
        x = x[mask]
        y = y[mask]
    if range == config['y_cut']:
        mask = abs(x) < float(config['y_cut'])
        x = x[mask]
        y = y[mask]
    return fit(x, y, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', help='Paths to the four csv files containing the v1 and a1 val', required=True)
    parser.add_argument('--paths_piKp', type=str, nargs='+', help='Paths to the four csv files containing the v1 and a1 val', required=True)
    parser.add_argument('--paths_pt', type=str, nargs='+', help='Path to the file containing the PT histograms', required=True)
    parser.add_argument('--fres', type=str, help='Path to the file containing the resolution histograms', required=True)
    parser.add_argument('--output', type=str, help='Path to the output txt file', required=True)
    parser.add_argument('--sys_tag', type=str, help='Systematic tag. "0" is default cut', default=0)
    parser.add_argument('--method', type=int, help='Method to use for fitting', default=1)
    parser.add_argument('--energy', type=str, help='Energy of the collision, as a string', default='7p7GeV')
    parser.add_argument('--yrange', type=float, nargs=2, help='Y range for the final plot', default=None)
    args = parser.parse_args()
    # select only non-required arguments and group them in a dictionnary
    kwd = {k: v for k, v in vars(args).items() if k not in ['paths', 'paths_piKp', 'paths_pt', 'fres', 'output']}
    main(paths=args.paths, paths_piKp=args.paths_piKp, paths_pt=args.paths_pt, fres=args.fres, output=args.output, **kwd)
