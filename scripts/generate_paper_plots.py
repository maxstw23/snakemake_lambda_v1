import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf
from uncertainties import unumpy, ufloat
import numpy as np
import pandas as pd
# import pickle
import os 
import argparse
import yaml
from pikp_merged import PikpMergedSlope
from data_point import DataPoint

# print(matplotlib.font_manager.get_font_names())
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['font.weight'] = 'bold'

# some plot parameters
tick_params = {'labelsize': 15}
plot_config = {
    'Lambda':
    {
        'marker': '*',
        'color': 'C3',
        'label': r'$\Lambda^0-\bar{\Lambda}^0$',
        'markersize': 10,
        'zorder': 1,
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    },
    'combo':
    {
        'marker': 's',
        'color': 'C0',
        'label': r'$(p-\bar{p})-(K^+-K^-)$',
        'markersize': 8,
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    },
    'combo2':
    {
        'marker': 'o',
        'color': 'black',
        'label': r'$p-\bar{p}$',
        'markersize': 8,
        'markerfacecolor': 'white',
        'ls': 'none',
        'capsize': 2,
        'alpha': 0.8
    },
    'proton':
    {
        'marker': 'o',
        'color': 'black',
        'label': r'$p-\bar{p}$',
        'markersize': 8,
        'markerfacecolor': 'white',
        'ls': 'none',
        'capsize': 2,
        'zorder': 2,
        'alpha': 0.8
    },
    'kaon':
    {
        'marker': 'd',
        'color': 'C0',
        'label': r'$K^+-K^-$',
        'markersize': 8,
        'ls': 'none',
        'markerfacecolor': 'white',
        'capsize': 2,
        'zorder': 3,
        'alpha': 0.8
    },
    'pion':
    {
        'marker': 'v',
        'color': 'C5',
        'label': r'$\pi^+-\pi^-$',
        'markersize': 8,
        'ls': 'none',
        'markerfacecolor': 'white',
        'capsize': 2
    },
    'urqmd': ### use fill_between
    {
        'color': 'C4',
        'label': 'UrQMD',
        'alpha': 0.5
    },
    'delta_s':
    {
        'marker': '*',
        'color': 'C0',
        'label': r'$\Delta s$',
        'markersize': 10,
        'zorder': 3,
        'ls': 'none',
        'capsize': 2
    },
    'delta_d':
    {
        'marker': 's',
        'color': 'C1',
        'label': r'$\Delta d$',
        'markersize': 8,
        'ls': 'none',
        'capsize': 2
    },
    'delta_u':
    {
        'marker': 'o',
        'color': 'C2',
        'label': r'$\Delta u$',
        'markersize': 8,
        'ls': 'none',
        'capsize': 2
    }
}


def find_files(input_files, key):
    for f in input_files:
        if key in f:
            return f
    return None


def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def calculate_chi2_per_ndf(data_points, model_points, nparams):
    """
    Calculate chi2 per ndf for the given data points and model points. Use total errors. 
    """
    chi2_array = (data_points - model_points).value**2 / (data_points.total_error()**2)
    ndf = len(data_points) - nparams
    chi2 = np.sum(chi2_array) / ndf
    return chi2


def plot_invmass_v1fit(dict_input, figs, input_path):
    # load data from input pickle files
    ### inv mass
    fig = plt.figure(figsize=(6.4, 9.6))
    gs = fig.add_gridspec(ncols=1, nrows=2, hspace=0, wspace=0)
    axes = gs.subplots(sharex='col', sharey='row')
    f = dict_input['invmass']
    with open(f, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.CLoader)
        ax = axes[0]

        energy = f.split('/')[-1].replace('.yaml', '').split('_')[-4].replace('p', '.')
        x = np.array(data_dict['x'])
        y = np.array(data_dict['y'])
        xfit = np.array(data_dict['xfit'])
        yerr = np.array(data_dict['yerr'])
        background = np.array(data_dict['background'])
        signal = np.array(data_dict['signal'])
        total = np.array(data_dict['total'])
        cen_string = data_dict['cen_string']
        y_string = data_dict['y_string']
        S = data_dict['S']
        S_err = data_dict['S_error']
        signal_uncertainty = ufloat(S, S_err)
        SpB = data_dict['S+B']
        sigma = data_dict['sigma']

        ax.errorbar(x, y, yerr, 0, ls='none')
        ax.plot(x, y, 'o', c='C0', markersize=3)
        ax.plot(xfit, total, c='C2', label='Total Fit')
        ax.plot(xfit, background, c='C5', label='2-Poly Background')
        ax.plot(x, signal, c='red', ls='--', label=r'$\Lambda$ Signal')
        ax.annotate('S = ' + r'${{{:.2ueL}}}$'.format(signal_uncertainty), xy=(0.5, 0.66), xycoords='axes fraction')
        ax.annotate(rf'S/$\sqrt{{S+B}}$ = {S / SpB ** 0.5:.2f}', xy=(0.5, 0.6), xycoords='axes fraction')
        ax.annotate(rf'$\sigma$ = {sigma :.5f}', xy=(0.5, 0.54), xycoords='axes fraction')
        ax.annotate(cen_string, xy=(0.03, 0.9), fontsize=15, xycoords='axes fraction')
        ax.annotate(y_string, xy=(0.03, 0.8), fontsize=15, xycoords='axes fraction')
        ax.annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.03, 0.7), fontsize=15, xycoords='axes fraction')
        ax.annotate('(a)', xy=(0.05, 0.1), fontsize=15, xycoords='axes fraction')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\text{Counts}$', fontsize=15)
        # ax.set_xlabel(r'$M_{\text{inv}}(\Lambda^0\rightarrow p+\pi^-) (\text{GeV}/c^2)$', fontsize=15)
        ax.set_xlabel('')
        ax.legend(fontsize=15, frameon=False)
        
        # plt.tight_layout()
        # plt.savefig(f.replace('.yaml', '.pdf').replace('_yaml', ''))   
        # plt.savefig(f.replace('.yaml', '.eps').replace('_yaml', ''), format='eps')
        # figs.append(fig)
    # plt.close()

    ### v1 fit
    f = dict_input['v1fit']
    with open(f, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.CLoader)
        ax = axes[1]

        x = np.array(data_dict['x'])
        y = np.array(data_dict['y'])
        yerr = np.array(data_dict['yerr'])
        background = np.array(data_dict['background'])
        signal = np.array(data_dict['signal'])
        total = np.array(data_dict['total'])
        cen_string = data_dict['cen_string']
        y_string = data_dict['y_string']

        # show_figure(fit)
        ax.errorbar(x, y, yerr=yerr, ls='none')
        ax.scatter(x, y, s=12)
        ax.plot(x, background, c='C5', label=r'$\frac{B}{S+B}(M_{\text{inv}})v_{1,\text{B}}(M_{\text{inv}})$')
        ax.plot(x, signal, c='red', label=r'$\frac{S}{S+B}(M_{\text{inv}})v_{1,\text{S}}$')
        ax.plot(x, total, c='C2', label='Total fit')
        ax.plot(x, np.zeros_like(x), ls='--', c='k')
        ax.set_ylabel(r'$v_{1,\text{raw}}(M_{\text{inv}})$', fontsize=15)
        ax.set_xlabel(r'$M_{\text{inv}}(\Lambda^0\rightarrow p+\pi^-) (\text{GeV}/c^2)$', fontsize=15)
        ax.annotate('(b)', xy=(0.05, 0.1), fontsize=15, xycoords='axes fraction')
        bin_size = x[1] - x[0]
        ax.set_xlim(x[0] - bin_size / 2, x[-1] + bin_size / 2)
        ax.set_ylim(-0.059, 0.079)
        # ax.annotate(cen_string, xy=(0.3, 0.9),fontsize=15, xycoords='axes fraction')
        # ax.annotate(y_string, xy=(0.3, 0.8), fontsize=15, xycoords='axes fraction')
        ax.legend(fontsize=15, frameon=False, loc='upper center')

        # plt.tight_layout()
        # plt.savefig(f.replace('.yaml', '.pdf').replace('_yaml', ''))   
        # plt.savefig(f.replace('.yaml', '.eps').replace('_yaml', ''), format='eps')
        # figs.append(fig)
    # plt.close()
    plt.tight_layout()
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/invmass_v1fit.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/invmass_v1fit.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/invmass_v1fit.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig)
    plt.close()
    return figs


def plot_res(dict_input, figs, input_path):
    ### resolution
    files = dict_input['res']
    fig_res, ax_res = plt.subplots()
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'H', 'D', 'P', '*', 'X']
    shift_range = 2 # shift the markers to avoid overlap, this means +- 0.5% shift
    for i, f in enumerate(files):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            x = x - 0.5 * shift_range + i / (len(files) - 1) * shift_range
            ax_res.errorbar(x, y, xerr=0, yerr=yerr, label=energy,
                            fmt=markers[i], capsize=2, ms=8)
    
    ax_res.set_xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    ax_res.set_ylabel(r'$\text{Res}(\Psi_{EP})$', fontsize=15)
    ax_res.set_ylim(0.0, 1.0)
    ax_res.legend(fontsize=12, frameon=False)
    plt.figure(fig_res.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/resolution.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/resolution.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/resolution.svg', format='svg')
    figs.append(fig_res)
    plt.close()
    return figs


def plot_dv1dy_coal_one_energy(dict_input, figs, input_path):
    ### v1 coal comparison for one energy
    f = find_files(dict_input['dv1dy_coal'], '19p6GeV')
    fig_one, ax_one = plt.subplots(figsize=(6.4,4.8))
    scaling = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 1}
    with open(f, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.CLoader)
        x = np.array(data_dict['x'])
        y = np.array(data_dict['y'])
        yerr = np.array(data_dict['yerr'])
        combo = data_dict['combo'] # delta proton - delta kaon
        combo_err = data_dict['combo_err']
        combo2 = data_dict['combo2'] # delta proton
        combo2_err = data_dict['combo2_err']
        # a1 = data_dict['a1']
        # a1_err = data_dict['a1_err']

        energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
        scale = 1 # don't scale
        ax_one.errorbar(x-1, y*scale, yerr=yerr*scale, **plot_config['Lambda'])
        ax_one.errorbar(x, combo*scale, yerr=combo_err*scale, **plot_config['combo'])
        ax_one.errorbar(x+1, combo2*scale, yerr=combo2_err*scale, **plot_config['combo2'])
        # ax_one.errorbar(x, a1*scale, yerr=a1_err*scale, fmt='o', capsize=2, ms=8)
        ax_one.annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
        if scale != 1:
            ax_one.annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
        ax_one.hlines(0, 0, 80, linestyles='--', colors='k')
        ax_one.set_ylim(-0.099, 0.059)
    ax_one.set_xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    ax_one.set_ylabel(r'$\Delta dv_1/dy$', fontsize=15)
    ax_one.legend(fontsize=15, frameon=False, loc='lower left')
    plt.figure(fig_one.number)
    plt.text(0.01, 1.02, '(a)', transform=ax_one.transAxes, fontsize=15)
    plt.tight_layout()
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_one_energy.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_one_energy.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_one_energy.svg', format='svg')
    figs.append(fig_one)
    plt.close()
    return figs


def plot_dv1dy_KP(dict_input, figs, input_path, **kwargs):
    ### v1 coal comparison for all energies
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 4, 1: 4, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}
    if kwargs['ncols'] == 3:
        scaling = {0: 4, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])
            proton = data_dict['proton'] # delta proton
            proton_err = data_dict['proton_err']
            kaon = data_dict['kaon'] # delta kaon
            kaon_err = data_dict['kaon_err']
            # a1 = data_dict['a1']
            # a1_err = data_dict['a1_err']

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x+1, proton*scale, yerr=proton_err*scale, **plot_config['combo2'])
            ax_coal[i].errorbar(x, kaon*scale, yerr=kaon_err*scale, **plot_config['kaon'])
            ax_coal[i].errorbar(x-1, y*scale, yerr=yerr*scale, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'})
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=14, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k') 
            ax_coal[i].set_ylim(-0.259, 0.179)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['kaon'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo2'])
        # ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
        ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='center')    
        ax_coal[index_legend].annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.15, 0.8), xycoords='axes fraction', fontsize=20)
    else:
        index_legend = 0
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend+1].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$K^+-K^-$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo2'])
        ax_coal[index_legend].legend(fontsize=14, frameon=False, loc='lower left') 
        ax_coal[index_legend+1].legend(fontsize=14, frameon=False, loc='lower left')

    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_KP.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_KP.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_KP.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def _eff_corr_label(eff_energies: list[str]) -> str:
    """Format eff-corrected energy list for annotation, e.g. '14.6, 19.6, 27 GeV'."""
    if not eff_energies:
        return ''
    nice = [e.replace('p', '.').replace('GeV', '') for e in eff_energies]
    return 'Eff-corr. applied for\n' + ', '.join(nice) + ' GeV'


def plot_fig_2(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0.5, 6: 0.5}
    if kwargs['ncols'] == 3:
        scaling = {0: 4, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    # piKp source / output naming / per-particle cut labels are injectable so an
    # alternative-cut dataset can be plotted without touching the default figure.
    pikp_slopes = kwargs['pikp_data'] if kwargs.get('pikp_data') is not None else PikpMergedSlope().get_data()
    output_suffix = kwargs.get('output_suffix', '')
    cut_labels = kwargs.get('cut_labels', {
        'Lambda': r'$0.4<p_{T}<1.8$',
        'proton': r'$0.4<p_{T}<1.8$, $p<2.0$',
        'kaon':   r'$0.28<p_{T}<1.2$, $p<1.6$'})
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        datapoints = {}
        for particle in ['lambdas', 'pions', 'kaons', 'protons']: datapoints[particle] = DataPoint([], [], [])
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            x = np.array(data_dict['x'])
            value = data_dict['y']
            error_stat = data_dict['yerr_stat']
            error_sys = data_dict['yerr_sys']
            datapoints['lambdas'].add_point(value, error_stat, error_sys)
        
        for particle in ['pions', 'kaons', 'protons']:
            fit = 'linear'
            value = pikp_slopes[energy][particle][f'delta_{fit}']
            error_stat = pikp_slopes[energy][particle][f'delta_{fit}_err']
            error_sys = pikp_slopes[energy][particle][f'delta_{fit}_systematics']
            datapoints[particle].add_point(value, error_stat, error_sys)
        
        if energy == '19.6GeV':
            # print average proton and kaon delta v1
            print(f'Proton average (50-80%): {datapoints['protons'][:3].average()}')
            print(f'Kaon average (50-80%): {datapoints['kaons'][:3].average()}')

        x_pikp = np.array([75., 65., 55., 45., 35., 25., 15., 7.5, 2.5])
        ax_coal[i].errorbar(x-0.75, datapoints['lambdas'].value*scale, yerr=datapoints[f'lambdas'].stat_error*scale, **plot_config['Lambda'])
        ax_coal[i].errorbar(x_pikp-0.25, datapoints['protons'].value*scale, yerr=datapoints[f'protons'].stat_error*scale, **plot_config['proton'])
        ax_coal[i].errorbar(x_pikp+0.25, datapoints['kaons'].value*scale, yerr=datapoints[f'kaons'].stat_error*scale, **plot_config['kaon'])
        ax_coal[i].set_yticks(np.linspace(-0.2, 0.2, 5))
        ax_coal[i].tick_params(**tick_params)
        # ax_coal[i].errorbar(x_pikp+0.75, datapoints['pions'].value*scale, yerr=datapoints[f'pions'].stat_error*scale, **plot_config['pion'], zorder=1)

        # systematic error bars
        for j, cent in enumerate(x):
            ax_coal[i].fill_between(np.array([cent-0.75-1.0, cent-0.75+1.0]),
                                    y1=datapoints['lambdas'].value[j]*scale-datapoints[f'lambdas'].sys_error[j]*scale,
                                    y2=datapoints['lambdas'].value[j]*scale+datapoints[f'lambdas'].sys_error[j]*scale,
                                    color=plot_config['Lambda']['color'], alpha=0.4, linewidth=0)
        for j, cent in enumerate(x_pikp):
            ax_coal[i].fill_between(np.array([cent-0.25-1.0, cent-0.25+1.0]),
                                    y1=datapoints['protons'].value[j]*scale-datapoints[f'protons'].sys_error[j]*scale,
                                    y2=datapoints['protons'].value[j]*scale+datapoints[f'protons'].sys_error[j]*scale,
                                    color=plot_config['proton']['color'], alpha=0.4, linewidth=0)
            ax_coal[i].fill_between(np.array([cent+0.25-1.0, cent+0.25+1.0]),
                                    y1=datapoints['kaons'].value[j]*scale-datapoints[f'kaons'].sys_error[j]*scale,
                                    y2=datapoints['kaons'].value[j]*scale+datapoints[f'kaons'].sys_error[j]*scale,
                                    color=plot_config['kaon']['color'], alpha=0.4, linewidth=0)
            # ax_coal[i].fill_between(np.array([cent+0.75-1.0, cent+0.75+1.0]),
            #                         y1=datapoints['pions'].value[j]*scale-datapoints[f'pions'].sys_error[j]*scale,
            #                         y2=datapoints['pions'].value[j]*scale+datapoints[f'pions'].sys_error[j]*scale,
            #                         color=plot_config['pion']['color'], alpha=0.4, linewidth=0)

        ax_coal[i].annotate(energy.split('GeV')[0] + ' GeV', xy=(0.85, 0.9), fontsize=18, xycoords='axes fraction', horizontalalignment='right')
        if scale != 1:
            ax_coal[i].annotate(fr'$\times$ {scale}', xy=(0.25, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
        ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k') 
        ax_coal[i].set_ylim(-0.149, 0.099)

    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=18, labelpad=10)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=18, labelpad=25)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].annotate('Au+Au', xy=(0.35, 0.88), fontsize=22, xycoords='axes fraction')
        ax_coal[index_legend].annotate(r'$p_{T}$, $p$ in GeV/$c$', xy=(0.35, 0.80), fontsize=15, xycoords='axes fraction')
        # legend labels: particle on line 1, kinematic range on line 2 (pT, p in GeV/c)
        ax_coal[index_legend].errorbar([], [], yerr=[], **{k: v for k, v in plot_config['Lambda'].items() if k != 'label'}, label='\n'.join([r'$\Lambda^0-\bar{\Lambda}^0$', cut_labels['Lambda']]))
        ax_coal[index_legend].errorbar([], [], yerr=[], **{k: v for k, v in plot_config['proton'].items() if k != 'label'}, label='\n'.join([r'$p-\bar{p}$', cut_labels['proton']]))
        ax_coal[index_legend].errorbar([], [], yerr=[], **{k: v for k, v in plot_config['kaon'].items() if k != 'label'}, label='\n'.join([r'$K^+-K^-$', cut_labels['kaon']]))
        # ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['pion'])
        # ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
        ax_coal[index_legend].legend(fontsize=14, frameon=False, loc='center')
        ax_coal[index_legend].tick_params(**tick_params)
        # ax_coal[index_legend].annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.15, 0.8), xycoords='axes fraction', fontsize=20)

    else:
        index_legend = 0
        ax_coal[index_legend].annotate('Au+Au', xy=(0.4, 0.7), fontsize=18, xycoords='axes fraction')
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['proton'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['kaon'])
        # ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['pion'])
        ax_coal[index_legend].legend(fontsize=18, frameon=False, loc='lower left') 
        ax_coal[index_legend+1].legend(fontsize=18, frameon=False, loc='lower left')
        ax_coal[index_legend+2].legend(fontsize=148, frameon=False, loc='lower left')
        ax_coal[index_legend].tick_params(**tick_params)

    plt.figure(fig_coal.number)
    _fig2_base = input_path.replace('_yaml', '').replace('/sys_tag_0', '') + f'/fig_2{output_suffix}'
    plt.savefig(_fig2_base + '.pdf')
    # plt.savefig(_fig2_base + '.eps', format='eps')
    plt.savefig(_fig2_base + '.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_coal)
    plt.close()

    return figs


def plot_dv1dy_coal(dict_input, figs, input_path, **kwargs):
    ### v1 coal comparison for all energies
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 4, 1: 4, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}
    if kwargs['ncols'] == 3:
        scaling = {0: 4, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])
            combo = data_dict['combo'] # delta proton - delta kaon
            combo_err = data_dict['combo_err']
            combo2 = data_dict['combo2'] # delta proton
            combo2_err = data_dict['combo2_err']
            # a1 = data_dict['a1']
            # a1_err = data_dict['a1_err']

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]

            ### if we are using Zhuo's result, uncomment the following lines
            # if energy.replace('.', 'p').replace('GeV', '') != '27':
            #     f_lambda = pd.read_csv(f'data/Zhuo_result/{energy.replace('.', 'p').replace('GeV', '')}/lam_Slope_9Cent.txt', sep=' ')
            #     f_lambdabar = pd.read_csv(f'data/Zhuo_result/{energy.replace('.', 'p').replace('GeV', '')}/antiLam_Slope_9Cent.txt', sep=' ')
            #     f_lambda.rename(columns={'#': 'Cent', 'Cent': 'Slope', 'Slope': 'Slope_err', 'Slope_err': ' '}, inplace=True)
            #     f_lambdabar.rename(columns={'#': 'Cent', 'Cent': 'Slope', 'Slope': 'Slope_err', 'Slope_err': ' '}, inplace=True)
            #     x_zhou = f_lambda['Cent'].to_numpy()
            #     y_lambda = f_lambda['Slope'].to_numpy()
            #     y_lambdabar = f_lambdabar['Slope'].to_numpy()
            #     yerr_lambda = f_lambda['Slope_err'].to_numpy()
            #     yerr_lambdabar = f_lambdabar['Slope_err'].to_numpy()
            #     y = y_lambda - y_lambdabar
            #     yerr = np.sqrt(yerr_lambda**2 + yerr_lambdabar**2)
            #     # mask corresponding y and yerr if there are elements in x_zhou that are not in x
            #     mask = np.isin(x_zhou, x)
            #     y = y[mask]
            #     yerr = yerr[mask]
                
            # find urqmd
            urqmd = find_files(dict_input['model_sim_urqmd'], energy.replace('.', 'p'))
            if urqmd is not None:
                with open(urqmd, 'r') as file:
                    df = pd.read_csv(file)
                    x_urqmd = df['centralities']
                    y_urqmd = df['dv1dy_lambda'] - df['dv1dy_lambdabar']
                    yerr_urqmd = np.sqrt(df['dv1dy_lambda_err']**2 + df['dv1dy_lambdabar_err']**2)
                    # ax_coal[i].fill_between(x_urqmd, (y_urqmd - yerr_urqmd)*scale, (y_urqmd + yerr_urqmd)*scale, **plot_config['urqmd'])
                
            ax_coal[i].errorbar(x+1, combo2*scale, yerr=combo2_err*scale, **plot_config['combo2'])
            ax_coal[i].errorbar(x, combo*scale, yerr=combo_err*scale, **{key:val for key, val in plot_config['combo'].items() if key != 'label'})
            ax_coal[i].errorbar(x-1, y*scale, yerr=yerr*scale, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'})
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=14, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k') 
            ax_coal[i].set_ylim(-0.259, 0.179)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo'])
        # ax_coal[index_legend].fill_between([], [], [], **plot_config['urqmd'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo2'])
        # ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
        ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='center')    
        # ax_coal[index_legend].annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.15, 0.8), xycoords='axes fraction', fontsize=20)
    else:
        index_legend = 0
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend+1].errorbar([], [], yerr=[], **plot_config['combo'])
        # ax_coal[index_legend+2].fill_between([], [], [], **plot_config['urqmd'])
        ax_coal[index_legend].legend(fontsize=14, frameon=False, loc='lower left') 
        ax_coal[index_legend+1].legend(fontsize=14, frameon=False, loc='lower left')
        ax_coal[index_legend+2].legend(fontsize=14, frameon=False, loc='lower left')

    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_coal)
    plt.close()

    ### output data points to multiple csvs
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])
            l = np.array(data_dict['lambda'])
            l_err = np.array(data_dict['lambda_err'])
            lb = np.array(data_dict['lambdabar'])
            lb_err = np.array(data_dict['lambdabar_err'])
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1]
            
            df = pd.DataFrame({
                'centrality': x, 
                'dv1dy_lambda': l, 
                'dv1dy_lambda_err': l_err, 
                'dv1dy_lambdabar': lb, 
                'dv1dy_lambdabar_err': lb_err, 
                'delta_dv1dy': y, 
                'delta_dv1dy_err': yerr
                })
            ### if this is final conversion, also contains sys and stats separately
            if 'yerr_sys' in data_dict:
                yerr_sys = np.array(data_dict['yerr_sys'])
                yerr_stat = np.array(data_dict['yerr_stat'])
                lambda_err_sys = np.array(data_dict['lambda_err_sys'])
                lambda_err_stat = np.array(data_dict['lambda_err_stat'])
                lambdabar_err_sys = np.array(data_dict['lambdabar_err_sys'])
                lambdabar_err_stat = np.array(data_dict['lambdabar_err_stat'])
                df = pd.DataFrame({
                    'centrality': x, 
                    'dv1dy_lambda': l, 
                    'dv1dy_lambda_err_sys': lambda_err_sys, 
                    'dv1dy_lambda_err_stat': lambda_err_stat, 
                    'dv1dy_lambdabar': lb, 
                    'dv1dy_lambdabar_err_sys': lambdabar_err_sys, 
                    'dv1dy_lambdabar_err_stat': lambdabar_err_stat, 
                    'delta_dv1dy': y, 
                    'delta_dv1dy_err_sys': yerr_sys, 
                    'delta_dv1dy_err_stat': yerr_stat})
            df.to_csv(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + f'/data_points/dv1dy_{energy}.csv', index=False)
    return figs


def plot_dv1dy_model(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_model = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_model = fig_model.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_model = gs_model.subplots(sharex='col', sharey='row')
    ax_model = ax_model.flatten()
    scaling = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4}
    scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
        scale = scaling[i]
        urqmd = find_files(dict_input['model_sim_ampt'], energy.replace('.', 'p'))
        if urqmd is not None:
            with open(urqmd, 'r') as file:
                df = pd.read_csv(file)
                x_urqmd = np.array(df['centralities'])
                y_urqmd = np.array(df['dv1dy_lambda']) - np.array(df['dv1dy_lambdabar'])
                yerr_urqmd = np.sqrt(df['dv1dy_lambda_err']**2 + df['dv1dy_lambdabar_err']**2)
                ax_model[i].errorbar(x_urqmd, y_urqmd*scale, yerr=yerr_urqmd*scale, **plot_config['Lambda'])
            with open(urqmd.replace('_lambda', '_proton'), 'r') as file_proton, open(urqmd.replace('_lambda', '_kaon'), 'r') as file_kaon:
                df_proton = pd.read_csv(file_proton)
                df_kaon = pd.read_csv(file_kaon)
                x_urqmd = np.array(df_proton['centralities'])
                y_urqmd_proton = np.array(df_proton['dv1dy_proton']) - np.array(df_proton['dv1dy_antiproton'])
                yerr_urqmd_proton = np.sqrt(df_proton['dv1dy_proton_err']**2 + df_proton['dv1dy_antiproton_err']**2)
                y_urqmd_kaon = np.array(df_kaon['dv1dy_kplus']) - np.array(df_kaon['dv1dy_kminus'])
                yerr_urqmd_kaon = np.sqrt(df_kaon['dv1dy_kplus_err']**2 + df_kaon['dv1dy_kminus_err']**2)
                y2_urqmd = y_urqmd_proton - y_urqmd_kaon
                yerr2_urqmd = np.sqrt(yerr_urqmd_proton**2 + yerr_urqmd_kaon**2)
                ax_model[i].errorbar(x_urqmd, y2_urqmd*scale, yerr=yerr2_urqmd*scale, **plot_config['combo'])
                ax_model[i].errorbar(x_urqmd, y_urqmd_proton*scale, yerr=yerr_urqmd_proton*scale, **plot_config['combo2'])
        
        ax_model[i].annotate(r'AuAu, $\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
        if scale != 1:
            ax_model[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=14, xycoords='axes fraction', horizontalalignment='right')
        ax_model[i].hlines(0, 0, 80, linestyles='--', colors='k') 
        ax_model[i].set_ylim(-0.259, 0.329)

    fig_model.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_model[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_model[index_legend].errorbar([], [], yerr=[], **plot_config['combo'])
        ax_model[index_legend].errorbar([], [], yerr=[], **plot_config['combo2'])
        ax_model[index_legend].legend(fontsize=15, frameon=False, loc='center')    
        
    plt.figure(fig_model.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_model.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_model.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_model.svg', format='svg')
    figs.append(fig_model)
    return figs


def plot_v1_merged_centrality(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4}
    scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['y_merged'])
            y = np.array(data_dict['v1_merged_lambda_4080'])
            yerr = np.array(data_dict['v1_merged_lambda_4080_err'])
            y2 = np.array(data_dict['v1_merged_lambdabar_4080'])
            yerr2 = np.array(data_dict['v1_merged_lambdabar_4080_err'])

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x, y*scale, yerr=yerr*scale, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'})
            ax_coal[i].errorbar(x, y2*scale, yerr=yerr2*scale, **{key:val for key, val in plot_config['combo'].items() if key != 'label'})
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, -1.05, 1.05, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.249, 0.249)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$y$', fontsize=15)
    plt.ylabel(r'$v_1(y)$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$')
        ax_coal[index_legend].annotate('40-80%', xy=(0.35, 0.3), fontsize=15, xycoords='axes fraction')
        ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
        ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='center')
    else:
        index_legend = 0
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$')
        ax_coal[index_legend+1].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$')
        ax_coal[index_legend].annotate('40-80%', xy=(0.35, 0.8), fontsize=15, xycoords='axes fraction')
        ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='lower center')
        ax_coal[index_legend+1].legend(fontsize=15, frameon=False, loc='lower center')
    
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_merged_centrality.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_merged_centrality.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_merged_centrality.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_v1_pt(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 6, 6: 6}
    scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['v1_pt_lambda_4080']['pT'])
            l = unumpy.uarray(np.array(data_dict['v1_pt_lambda_4080']['value']), np.array(data_dict['v1_pt_lambda_4080']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_pt_lambdabar_4080']['value']), np.array(data_dict['v1_pt_lambdabar_4080']['error_stat']))
            y = unumpy.uarray(np.array(data_dict['v1_pt_delta_4080']['value']), np.array(data_dict['v1_pt_delta_4080']['error_stat']))
            l = unumpy.uarray(np.array(data_dict['v1_pt_lambda_1040']['value']), np.array(data_dict['v1_pt_lambda_1040']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_pt_lambdabar_1040']['value']), np.array(data_dict['v1_pt_lambdabar_1040']['error_stat']))
            y2 = unumpy.uarray(np.array(data_dict['v1_pt_delta_1040']['value']), np.array(data_dict['v1_pt_delta_1040']['error_stat']))
            l = unumpy.uarray(np.array(data_dict['v1_pt_lambda_010']['value']), np.array(data_dict['v1_pt_lambda_010']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_pt_lambdabar_010']['value']), np.array(data_dict['v1_pt_lambdabar_010']['error_stat']))
            y3 = unumpy.uarray(np.array(data_dict['v1_pt_delta_010']['value']), np.array(data_dict['v1_pt_delta_010']['error_stat']))

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            print(f'x: {x})')
            ax_coal[i].errorbar(x, unumpy.nominal_values(y)*scale, yerr=unumpy.std_devs(y)*scale, **plot_config['Lambda'])
            ax_coal[i].errorbar(x, unumpy.nominal_values(y2)*scale, yerr=unumpy.std_devs(y2)*scale, **plot_config['combo'])
            ax_coal[i].errorbar(x, unumpy.nominal_values(y3)*scale, yerr=unumpy.std_devs(y3)*scale, **plot_config['combo2'])
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 2., linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.109, 0.109)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$p_T$', fontsize=15)
    plt.ylabel(r'$\Delta (v_1(p_T))$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$40-80\%$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$10-40\%$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$0-10\%$')
        ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='center' )
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_pt.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_pt.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_pt.svg', format='svg')
    figs.append(fig_coal)
    plt.close()

    ### output data points to multiple csvs
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            # x = np.array(data_dict['v1_pt_lambda_4080']['pT']) # two digits precision
            # two digits precision for pT
            x = np.array([round(pt, 2) for pt in data_dict['v1_pt_lambda_4080']['pT']])
            v1_pt_lambda_4080 = np.array(data_dict['v1_pt_lambda_4080']['value'])
            v1_pt_lambda_4080_err = np.array(data_dict['v1_pt_lambda_4080']['error_stat'])
            v1_pt_lambda_4080_sys = np.array(data_dict['v1_pt_lambda_4080']['error_sys'])
            v1_pt_lambdabar_4080 = np.array(data_dict['v1_pt_lambdabar_4080']['value'])
            v1_pt_lambdabar_4080_err = np.array(data_dict['v1_pt_lambdabar_4080']['error_stat'])
            v1_pt_lambdabar_4080_sys = np.array(data_dict['v1_pt_lambdabar_4080']['error_sys'])
            delta_v1_pt_4080 = np.array(data_dict['v1_pt_delta_4080']['value'])
            delta_v1_pt_4080_err = np.array(data_dict['v1_pt_delta_4080']['error_stat'])
            delta_v1_pt_4080_sys = np.array(data_dict['v1_pt_delta_4080']['error_sys'])
            v1_pt_lambda_1040 = np.array(data_dict['v1_pt_lambda_1040']['value'])
            v1_pt_lambda_1040_err = np.array(data_dict['v1_pt_lambda_1040']['error_stat'])
            v1_pt_lambda_1040_sys = np.array(data_dict['v1_pt_lambda_1040']['error_sys'])
            v1_pt_lambdabar_1040 = np.array(data_dict['v1_pt_lambdabar_1040']['value'])
            v1_pt_lambdabar_1040_err = np.array(data_dict['v1_pt_lambdabar_1040']['error_stat'])
            v1_pt_lambdabar_1040_sys = np.array(data_dict['v1_pt_lambdabar_1040']['error_sys'])
            delta_v1_pt_1040 = np.array(data_dict['v1_pt_delta_1040']['value'])
            delta_v1_pt_1040_err = np.array(data_dict['v1_pt_delta_1040']['error_stat'])
            delta_v1_pt_1040_sys = np.array(data_dict['v1_pt_delta_1040']['error_sys'])
            v1_pt_lambda_010 = np.array(data_dict['v1_pt_lambda_010']['value'])
            v1_pt_lambda_010_err = np.array(data_dict['v1_pt_lambda_010']['error_stat'])
            v1_pt_lambda_010_sys = np.array(data_dict['v1_pt_lambda_010']['error_sys'])
            v1_pt_lambdabar_010 = np.array(data_dict['v1_pt_lambdabar_010']['value'])
            v1_pt_lambdabar_010_err = np.array(data_dict['v1_pt_lambdabar_010']['error_stat'])
            v1_pt_lambdabar_010_sys = np.array(data_dict['v1_pt_lambdabar_010']['error_sys'])
            delta_v1_pt_010 = np.array(data_dict['v1_pt_delta_010']['value'])
            delta_v1_pt_010_err = np.array(data_dict['v1_pt_delta_010']['error_stat'])
            delta_v1_pt_010_sys = np.array(data_dict['v1_pt_delta_010']['error_sys'])
            v1_pt_lambda_5080 = np.array(data_dict['v1_pt_lambda_5080']['value'])
            v1_pt_lambda_5080_err = np.array(data_dict['v1_pt_lambda_5080']['error_stat'])
            v1_pt_lambda_5080_sys = np.array(data_dict['v1_pt_lambda_5080']['error_sys'])
            v1_pt_lambdabar_5080 = np.array(data_dict['v1_pt_lambdabar_5080']['value'])
            v1_pt_lambdabar_5080_err = np.array(data_dict['v1_pt_lambdabar_5080']['error_stat'])
            v1_pt_lambdabar_5080_sys = np.array(data_dict['v1_pt_lambdabar_5080']['error_sys'])
            delta_v1_pt_5080 = np.array(data_dict['v1_pt_delta_5080']['value'])
            delta_v1_pt_5080_err = np.array(data_dict['v1_pt_delta_5080']['error_stat'])
            delta_v1_pt_5080_sys = np.array(data_dict['v1_pt_delta_5080']['error_sys'])
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1]

            df = pd.DataFrame({
                'pT': x, 
                'v1_pt_lambda_4080': v1_pt_lambda_4080, 
                'v1_pt_lambda_4080_err': v1_pt_lambda_4080_err, 
                'v1_pt_lambda_4080_err_sys': v1_pt_lambda_4080_sys,
                'v1_pt_lambdabar_4080': v1_pt_lambdabar_4080, 
                'v1_pt_lambdabar_4080_err': v1_pt_lambdabar_4080_err, 
                'v1_pt_lambdabar_4080_err_sys': v1_pt_lambdabar_4080_sys,
                'delta_v1_pt_4080': delta_v1_pt_4080, 
                'delta_v1_pt_4080_err': delta_v1_pt_4080_err, 
                'delta_v1_pt_4080_err_sys': delta_v1_pt_4080_sys,
                'v1_pt_lambda_1040': v1_pt_lambda_1040, 
                'v1_pt_lambda_1040_err': v1_pt_lambda_1040_err, 
                'v1_pt_lambda_1040_err_sys': v1_pt_lambda_1040_sys,
                'v1_pt_lambdabar_1040': v1_pt_lambdabar_1040, 
                'v1_pt_lambdabar_1040_err': v1_pt_lambdabar_1040_err, 
                'v1_pt_lambdabar_1040_err_sys': v1_pt_lambdabar_1040_sys,
                'delta_v1_pt_1040': delta_v1_pt_1040, 
                'delta_v1_pt_1040_err': delta_v1_pt_1040_err, 
                'delta_v1_pt_1040_err_sys': delta_v1_pt_1040_sys,
                'v1_pt_lambda_010': v1_pt_lambda_010, 
                'v1_pt_lambda_010_err': v1_pt_lambda_010_err, 
                'v1_pt_lambda_010_err_sys': v1_pt_lambda_010_sys,
                'v1_pt_lambdabar_010': v1_pt_lambdabar_010, 
                'v1_pt_lambdabar_010_err': v1_pt_lambdabar_010_err, 
                'v1_pt_lambdabar_010_err_sys': v1_pt_lambdabar_010_sys,
                'delta_v1_pt_010': delta_v1_pt_010,
                'delta_v1_pt_010_err': delta_v1_pt_010_err,
                'delta_v1_pt_010_err_sys': delta_v1_pt_010_sys,
                'v1_pt_lambda_5080': v1_pt_lambda_5080,
                'v1_pt_lambda_5080_err': v1_pt_lambda_5080_err,
                'v1_pt_lambda_5080_err_sys': v1_pt_lambda_5080_sys,
                'v1_pt_lambdabar_5080': v1_pt_lambdabar_5080,
                'v1_pt_lambdabar_5080_err': v1_pt_lambdabar_5080_err,
                'v1_pt_lambdabar_5080_err_sys': v1_pt_lambdabar_5080_sys,
                'delta_v1_pt_5080': delta_v1_pt_5080,
                'delta_v1_pt_5080_err': delta_v1_pt_5080_err,
                'delta_v1_pt_5080_err_sys': delta_v1_pt_5080_sys
            })
            df.to_csv(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + f'/data_points/v1_pt_{energy}.csv', index=False)
    return figs
                         

def plot_v1_y(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 6, 6: 6}
    scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['v1_y_lambda_4080']['y'])
            l = unumpy.uarray(np.array(data_dict['v1_y_lambda_4080']['value']), np.array(data_dict['v1_y_lambda_4080']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_y_lambdabar_4080']['value']), np.array(data_dict['v1_y_lambdabar_4080']['error_stat']))
            y = unumpy.uarray(np.array(data_dict['v1_y_delta_4080']['value']), np.array(data_dict['v1_y_delta_4080']['error_stat']))
            l = unumpy.uarray(np.array(data_dict['v1_y_lambda_1040']['value']), np.array(data_dict['v1_y_lambda_1040']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_y_lambdabar_1040']['value']), np.array(data_dict['v1_y_lambdabar_1040']['error_stat']))
            y2 = unumpy.uarray(np.array(data_dict['v1_y_delta_1040']['value']), np.array(data_dict['v1_y_delta_1040']['error_stat']))
            l = unumpy.uarray(np.array(data_dict['v1_y_lambda_010']['value']), np.array(data_dict['v1_y_lambda_010']['error_stat']))
            lb = unumpy.uarray(np.array(data_dict['v1_y_lambdabar_010']['value']), np.array(data_dict['v1_y_lambdabar_010']['error_stat']))
            y3 = unumpy.uarray(np.array(data_dict['v1_y_delta_010']['value']), np.array(data_dict['v1_y_delta_010']['error_stat']))
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x, unumpy.nominal_values(y)*scale, yerr=unumpy.std_devs(y)*scale, **plot_config['Lambda'])
            ax_coal[i].errorbar(x, unumpy.nominal_values(y2)*scale, yerr=unumpy.std_devs(y2)*scale, **plot_config['combo'])
            ax_coal[i].errorbar(x, unumpy.nominal_values(y3)*scale, yerr=unumpy.std_devs(y3)*scale, **plot_config['combo2'])
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, -1.05, 1.05, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.249, 0.249)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$y$', fontsize=15)
    plt.ylabel(r'$\Delta (v_1(y))$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$40-80\%$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$10-40\%$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$0-10\%$')
        ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='center' )
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_y.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_y.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v1_y.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    
    #### output data points to multiple csvs
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array([round(y, 2) for y in data_dict['v1_y_lambda_4080']['y']])  # two digits precision for y
            v1_y_lambda_4080 = np.array(data_dict['v1_y_lambda_4080']['value'])
            v1_y_lambda_4080_err = np.array(data_dict['v1_y_lambda_4080']['error_stat'])
            v1_y_lambda_4080_sys = np.array(data_dict['v1_y_lambda_4080']['error_sys'])
            v1_y_lambdabar_4080 = np.array(data_dict['v1_y_lambdabar_4080']['value'])
            v1_y_lambdabar_4080_err = np.array(data_dict['v1_y_lambdabar_4080']['error_stat'])
            v1_y_lambdabar_4080_sys = np.array(data_dict['v1_y_lambdabar_4080']['error_sys'])
            delta_v1_y_4080 = np.array(data_dict['v1_y_delta_4080']['value'])
            delta_v1_y_4080_err = np.array(data_dict['v1_y_delta_4080']['error_stat'])
            delta_v1_y_4080_sys = np.array(data_dict['v1_y_delta_4080']['error_sys'])
            v1_y_lambda_1040 = np.array(data_dict['v1_y_lambda_1040']['value'])
            v1_y_lambda_1040_err = np.array(data_dict['v1_y_lambda_1040']['error_stat'])
            v1_y_lambda_1040_sys = np.array(data_dict['v1_y_lambda_1040']['error_sys'])
            v1_y_lambdabar_1040 = np.array(data_dict['v1_y_lambdabar_1040']['value'])
            v1_y_lambdabar_1040_err = np.array(data_dict['v1_y_lambdabar_1040']['error_stat'])
            v1_y_lambdabar_1040_sys = np.array(data_dict['v1_y_lambdabar_1040']['error_sys'])
            delta_v1_y_1040 = np.array(data_dict['v1_y_delta_1040']['value'])
            delta_v1_y_1040_err = np.array(data_dict['v1_y_delta_1040']['error_stat'])
            delta_v1_y_1040_sys = np.array(data_dict['v1_y_delta_1040']['error_sys'])
            v1_y_lambda_010 = np.array(data_dict['v1_y_lambda_010']['value'])
            v1_y_lambda_010_err = np.array(data_dict['v1_y_lambda_010']['error_stat'])
            v1_y_lambda_010_sys = np.array(data_dict['v1_y_lambda_010']['error_sys'])
            v1_y_lambdabar_010 = np.array(data_dict['v1_y_lambdabar_010']['value'])
            v1_y_lambdabar_010_err = np.array(data_dict['v1_y_lambdabar_010']['error_stat'])
            v1_y_lambdabar_010_sys = np.array(data_dict['v1_y_lambdabar_010']['error_sys'])
            delta_v1_y_010 = np.array(data_dict['v1_y_delta_010']['value'])
            delta_v1_y_010_err = np.array(data_dict['v1_y_delta_010']['error_stat'])
            delta_v1_y_010_sys = np.array(data_dict['v1_y_delta_010']['error_sys'])
            v1_y_lambda_5080 = np.array(data_dict['v1_y_lambda_5080']['value'])
            v1_y_lambda_5080_err = np.array(data_dict['v1_y_lambda_5080']['error_stat'])
            v1_y_lambda_5080_sys = np.array(data_dict['v1_y_lambda_5080']['error_sys'])
            v1_y_lambdabar_5080 = np.array(data_dict['v1_y_lambdabar_5080']['value'])
            v1_y_lambdabar_5080_err = np.array(data_dict['v1_y_lambdabar_5080']['error_stat'])
            v1_y_lambdabar_5080_sys = np.array(data_dict['v1_y_lambdabar_5080']['error_sys'])
            delta_v1_y_5080 = np.array(data_dict['v1_y_delta_5080']['value'])
            delta_v1_y_5080_err = np.array(data_dict['v1_y_delta_5080']['error_stat'])
            delta_v1_y_5080_sys = np.array(data_dict['v1_y_delta_5080']['error_sys'])
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1]

            df = pd.DataFrame({
                'y': x,
                'v1_y_lambda_4080': v1_y_lambda_4080,
                'v1_y_lambda_4080_err': v1_y_lambda_4080_err,
                'v1_y_lambda_4080_err_sys': v1_y_lambda_4080_sys,
                'v1_y_lambdabar_4080': v1_y_lambdabar_4080,
                'v1_y_lambdabar_4080_err': v1_y_lambdabar_4080_err,
                'v1_y_lambdabar_4080_err_sys': v1_y_lambdabar_4080_sys,
                'delta_v1_y_4080': delta_v1_y_4080,
                'delta_v1_y_4080_err': delta_v1_y_4080_err,
                'delta_v1_y_4080_err_sys': delta_v1_y_4080_sys,
                'v1_y_lambda_1040': v1_y_lambda_1040,
                'v1_y_lambda_1040_err': v1_y_lambda_1040_err,
                'v1_y_lambda_1040_err_sys': v1_y_lambda_1040_sys,
                'v1_y_lambdabar_1040': v1_y_lambdabar_1040,
                'v1_y_lambdabar_1040_err': v1_y_lambdabar_1040_err,
                'v1_y_lambdabar_1040_err_sys': v1_y_lambdabar_1040_sys,
                'delta_v1_y_1040': delta_v1_y_1040,
                'delta_v1_y_1040_err': delta_v1_y_1040_err,
                'delta_v1_y_1040_err_sys': delta_v1_y_1040_sys,
                'v1_y_lambda_010': v1_y_lambda_010,
                'v1_y_lambda_010_err': v1_y_lambda_010_err,
                'v1_y_lambda_010_err_sys': v1_y_lambda_010_sys,
                'v1_y_lambdabar_010': v1_y_lambdabar_010,
                'v1_y_lambdabar_010_err': v1_y_lambdabar_010_err,
                'v1_y_lambdabar_010_err_sys': v1_y_lambdabar_010_sys,
                'delta_v1_y_010': delta_v1_y_010,
                'delta_v1_y_010_err': delta_v1_y_010_err,
                'delta_v1_y_010_err_sys': delta_v1_y_010_sys,
                'v1_y_lambda_5080': v1_y_lambda_5080,
                'v1_y_lambda_5080_err': v1_y_lambda_5080_err,
                'v1_y_lambda_5080_err_sys': v1_y_lambda_5080_sys,
                'v1_y_lambdabar_5080': v1_y_lambdabar_5080,
                'v1_y_lambdabar_5080_err': v1_y_lambdabar_5080_err,
                'v1_y_lambdabar_5080_err_sys': v1_y_lambdabar_5080_sys,
                'delta_v1_y_5080': delta_v1_y_5080,
                'delta_v1_y_5080_err': delta_v1_y_5080_err,
                'delta_v1_y_5080_err_sys': delta_v1_y_5080_sys
            })

            df.to_csv(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + f'/data_points/v1_y_{energy}.csv', index=False)

    return figs

def plot_dv1dy_baryons(dict_input, figs, input_path, **kwargs):
    ### v1 coal comparison for all energies
    files = dict_input['dv1dy_coal']
    files_xi = dict_input['dv1dy_coal_xi']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 6, 1: 6, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, (f, f_xi) in enumerate(zip(reversed(files), reversed(files_xi))):
        with open(f, 'r') as file, open(f_xi, 'r') as file_xi:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            data_dict_xi = yaml.load(file_xi, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])
            y_proton = np.array(data_dict['proton'])
            yerr_proton = np.array(data_dict['proton_err'])
            x_xi = np.array(data_dict_xi['x'])
            y_xi = np.array(data_dict_xi['y'])
            yerr_xi = np.array(data_dict_xi['yerr'])

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x_xi+1, y_xi*scale, yerr=yerr_xi*scale, **plot_config['combo2'])
            ax_coal[i].errorbar(x, y_proton*scale, yerr=yerr_proton*scale, **plot_config['combo'])
            ax_coal[i].errorbar(x-1, y*scale, yerr=yerr*scale, **plot_config['Lambda'])
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.559, 0.499)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$p-\bar{p}$')
        ax_coal[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$\Xi^- - \bar{\Xi}^+$')
        ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )    
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_baryon.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_baryon.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_baryon.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_PT_test(dict_input, figs, input_path, **kwargs):
    # Test Prithwish's suggestion
    # i.e., Lambda (uds) = piplus (u\bar{d}) + piminus (d\bar{u}) - AntiProton (\bar{u}\bar{u}\bar{d}) + kminus (\bar{u}s)
    #       Lambdabar (\bar{u}\bar{d}\bar{s}) = piminus (d\bar{u}) + piplus (u\bar{d}) - Proton (uud) + kplus (u\bar{s})
    files = dict_input['dv1dy_coal']
    fig_coal_lambda = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal_lambda = fig_coal_lambda.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal_lambda = gs_coal_lambda.subplots(sharex='col', sharey='row')
    ax_coal_lambda = ax_coal_lambda.flatten()
    fig_coal_lambdabar = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal_lambdabar = fig_coal_lambdabar.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal_lambdabar = gs_coal_lambdabar.subplots(sharex='col', sharey='row')
    ax_coal_lambdabar = ax_coal_lambdabar.flatten()
    scaling = {0: 6, 1: 6, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            lambda_dv1dy = np.array(data_dict['lambda'])
            lambda_dv1dy_err = np.array(data_dict['lambda_err'])
            lambdabar_dv1dy = np.array(data_dict['lambdabar'])
            lambdabar_dv1dy_err = np.array(data_dict['lambdabar_err'])
            combo_PT_1 = np.array(data_dict['combo_PT_1'])
            combo_PT_1_err = np.array(data_dict['combo_PT_1_err'])
            combo_PT_2 = np.array(data_dict['combo_PT_2'])
            combo_PT_2_err = np.array(data_dict['combo_PT_2_err'])

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal_lambda[i].errorbar(x-1, lambda_dv1dy*scale, yerr=lambda_dv1dy_err*scale, **plot_config['Lambda'])
            ax_coal_lambda[i].errorbar(x+1, combo_PT_1*scale, yerr=combo_PT_1_err*scale, **plot_config['combo'])
            ax_coal_lambdabar[i].errorbar(x-1, lambdabar_dv1dy*scale, yerr=lambdabar_dv1dy_err*scale, **plot_config['Lambda'])
            ax_coal_lambdabar[i].errorbar(x+1, combo_PT_2*scale, yerr=combo_PT_2_err*scale, **plot_config['combo'])
            ax_coal_lambda[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal_lambdabar[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal_lambda[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
                ax_coal_lambdabar[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal_lambda[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal_lambdabar[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal_lambda[i].set_ylim(-0.419, 0.359)
            ax_coal_lambdabar[i].set_ylim(-0.419, 0.359)
    fig_coal_lambda.add_subplot(111, frameon=False)
    plt.figure(fig_coal_lambda.number)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$' + '\n' + r'$(uds)$')
        ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\pi^+ + \pi^- - \bar{p} + K^-$' + '\n' + r'$(uds)$')
        ax_coal_lambda[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal_lambda[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1.svg', format='svg')
    figs.append(fig_coal_lambda)
    plt.close()

    plt.figure(fig_coal_lambdabar.number)
    fig_coal_lambdabar.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal_lambdabar[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$' + '\n' + r'$(\bar{u}\bar{d}\bar{s})$')
        ax_coal_lambdabar[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\pi^- + \pi^+ - p + K^+$' + '\n' + r'$(\bar{u}\bar{d}\bar{s})$')
        ax_coal_lambdabar[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal_lambdabar[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2.svg', format='svg')
    figs.append(fig_coal_lambdabar)
    plt.close()

    return figs


def plot_PT_test_mod(dict_input, figs, input_path, **kwargs):
    # Test Modified Prithwish's suggestion
    files = dict_input['dv1dy_coal']
    fig_coal_lambda = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal_lambda = fig_coal_lambda.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal_lambda = gs_coal_lambda.subplots(sharex='col', sharey='row')
    ax_coal_lambda = ax_coal_lambda.flatten()
    fig_coal_lambdabar = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal_lambdabar = fig_coal_lambdabar.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal_lambdabar = gs_coal_lambdabar.subplots(sharex='col', sharey='row')
    ax_coal_lambdabar = ax_coal_lambdabar.flatten()
    scaling = {0: 6, 1: 6, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            lambda_dv1dy = np.array(data_dict['lambda'])
            lambda_dv1dy_err = np.array(data_dict['lambda_err'])
            lambdabar_dv1dy = np.array(data_dict['lambdabar'])
            lambdabar_dv1dy_err = np.array(data_dict['lambdabar_err'])
            proton_dv1dy = np.array(data_dict['p'])
            proton_dv1dy_err = np.array(data_dict['p_err'])
            antiproton_dv1dy = np.array(data_dict['pbar'])
            antiproton_dv1dy_err = np.array(data_dict['pbar_err'])
            combo_PT_1 = np.array(data_dict['combo_PT_1_mod'])
            combo_PT_1_err = np.array(data_dict['combo_PT_1_mod_err'])
            combo_PT_1_asso = np.array(data_dict['combo_PT_1_asso'])
            combo_PT_1_asso_err = np.array(data_dict['combo_PT_1_asso_err'])
            combo_PT_2 = np.array(data_dict['combo_PT_2_mod'])
            combo_PT_2_err = np.array(data_dict['combo_PT_2_mod_err'])

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal_lambda[i].errorbar(x, proton_dv1dy*scale, yerr=proton_dv1dy_err*scale, **plot_config['combo2'])
            ax_coal_lambda[i].errorbar(x-1, lambda_dv1dy*scale, yerr=lambda_dv1dy_err*scale, **plot_config['Lambda'])
            ax_coal_lambda[i].errorbar(x+1, combo_PT_1*scale, yerr=combo_PT_1_err*scale, **plot_config['combo'])
            # ax_coal_lambda[i].errorbar(x, combo_PT_1_asso*scale, yerr=combo_PT_1_asso_err*scale, **plot_config['combo2'])
            ax_coal_lambdabar[i].errorbar(x, antiproton_dv1dy*scale, yerr=antiproton_dv1dy_err*scale, **plot_config['combo2'])
            ax_coal_lambdabar[i].errorbar(x-1, lambdabar_dv1dy*scale, yerr=lambdabar_dv1dy_err*scale, **plot_config['Lambda'])
            ax_coal_lambdabar[i].errorbar(x+1, combo_PT_2*scale, yerr=combo_PT_2_err*scale, **plot_config['combo'])
            ax_coal_lambda[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal_lambdabar[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal_lambda[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
                ax_coal_lambdabar[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal_lambda[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal_lambdabar[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal_lambda[i].set_ylim(-0.419, 0.359)
            ax_coal_lambdabar[i].set_ylim(-0.419, 0.359)
    fig_coal_lambda.add_subplot(111, frameon=False)
    plt.figure(fig_coal_lambda.number)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$' + '\n' + r'$(uds)$')
        ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$p + K^- - 0.5(\pi^+ + \pi^-)$' + '\n' + r'$(uds+0.5(u\bar{u}-d\bar{d}))$')
        ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$p$' + '\n' + r'$(uud)$')
        # ax_coal_lambda[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$p - K^+$')
        ax_coal_lambda[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal_lambda[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1_mod.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1_mod.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_1_mod.svg', format='svg')
    figs.append(fig_coal_lambda)
    plt.close()

    plt.figure(fig_coal_lambdabar.number)
    fig_coal_lambdabar.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$dv_1/dy$', fontsize=15, labelpad=20)
    ax_coal_lambdabar[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$' + '\n' + r'$(\bar{u}\bar{d}\bar{s})$')
    ax_coal_lambdabar[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-)$' + '\n' + r'$(\bar{u}\bar{d}\bar{s}+0.5(u\bar{u}-d\bar{d}))$')
    ax_coal_lambdabar[index_legend].errorbar([], [], yerr=[], **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$\bar{p}$' + '\n' + r'$(\bar{u}\bar{u}\bar{d})$')
    ax_coal_lambdabar[index_legend].tick_params(axis='x', which='both', length=0)
    ax_coal_lambdabar[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2_mod.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2_mod.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_2_mod.svg', format='svg')
    figs.append(fig_coal_lambdabar)
    plt.close()

    return figs


def plot_d3v1dy3_coal(dict_input, figs, input_path, **kwargs):
    ### v1 coal comparison for all energies
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 6, 1: 6, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            lambda_d3v1dy3 = np.array(data_dict['d3v1dy3_lambda'])
            lambda_d3v1dy3_err = np.array(data_dict['d3v1dy3_lambda_err'])
            lambdabar_d3v1dy3 = np.array(data_dict['d3v1dy3_lambdabar'])
            lambdabar_d3v1dy3_err = np.array(data_dict['d3v1dy3_lambdabar_err'])
            y = lambda_d3v1dy3 - lambdabar_d3v1dy3
            yerr = np.sqrt(lambda_d3v1dy3_err**2 + lambdabar_d3v1dy3_err**2)
            proton_d3v1dy3 = np.array(data_dict['d3v1dy3_proton'])
            proton_d3v1dy3_err = np.array(data_dict['d3v1dy3_proton_err'])
            antiproton_d3v1dy3 = np.array(data_dict['d3v1dy3_antiproton'])
            antiproton_d3v1dy3_err = np.array(data_dict['d3v1dy3_antiproton_err'])
            kplus_d3v1dy3 = np.array(data_dict['d3v1dy3_kplus'])
            kplus_d3v1dy3_err = np.array(data_dict['d3v1dy3_kplus_err'])
            kminus_d3v1dy3 = np.array(data_dict['d3v1dy3_kminus'])
            kminus_d3v1dy3_err = np.array(data_dict['d3v1dy3_kminus_err'])
            combo = proton_d3v1dy3 - antiproton_d3v1dy3 - kplus_d3v1dy3 + kminus_d3v1dy3
            combo_err = np.sqrt(proton_d3v1dy3_err**2 + antiproton_d3v1dy3_err**2 + kplus_d3v1dy3_err**2 + kminus_d3v1dy3_err**2)
            combo2 = proton_d3v1dy3 - antiproton_d3v1dy3
            combo2_err = np.sqrt(proton_d3v1dy3_err**2 + antiproton_d3v1dy3_err**2)

            # a1 = data_dict['a1']
            # a1_err = data_dict['a1_err']

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x+1, combo2*scale, yerr=combo2_err*scale, **plot_config['combo2'])
            ax_coal[i].errorbar(x, combo*scale, yerr=combo_err*scale, **plot_config['combo'])
            ax_coal[i].errorbar(x-1, y*scale, yerr=yerr*scale, **plot_config['Lambda'])
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.419, 0.359)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta d^3v_1/dy^3$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['Lambda'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo'])
        ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['combo2'])
        ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    else:
        index_legend = 0
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )    
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/d3v1dy3_coal.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/d3v1dy3_coal.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/d3v1dy3_coal.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_dv1dy_coal_xi(dict_input, figs, input_path, **kwargs):
    ### v1 coal comparison for all energies
    files = dict_input['dv1dy_coal_xi']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 8, 1: 8, 2: 8, 3: 5, 4: 5, 5: 1, 6: 1}
    # scaling = {ind: scaling[len(scaling) - 1 - ind] for ind in scaling}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            y = np.array(data_dict['y'])
            yerr = np.array(data_dict['yerr'])
            # mask y values if yerr is 10 times larger than median yerr
            mask = yerr > 10 * np.median(yerr)
            x = x[~mask]
            y = y[~mask]
            yerr = yerr[~mask]
            combo = data_dict['combo'][~mask] # delta Lambda - delta kaon
            combo_err = data_dict['combo_err'][~mask]
            combo2 = data_dict['combo2'][~mask] # delta proton - 2 * delta kaon
            combo2_err = data_dict['combo2_err'][~mask]

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-2].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x+1, combo*scale, yerr=combo_err*scale, **plot_config['combo'])
            # ax_coal[i].errorbar(x, combo2*scale, yerr=combo2_err*scale, **plot_config['combo2'])
            ax_coal[i].errorbar(x-1, y*scale, yerr=yerr*scale, **plot_config['Lambda'])
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.699, 0.699)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
    else:
        index_legend = 0
    ax_coal[index_legend].errorbar([], [], yerr=[], label=r'$\Xi^- - \bar{\Xi}^+$', **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'})
    ax_coal[index_legend].errorbar([], [], yerr=[], label=r'$(\Lambda^0-\bar{\Lambda}^0)-(K^+-K^-)$',**{key:val for key, val in plot_config['combo'].items() if key != 'label'})
    # ax_coal[index_legend].errorbar([], [], yerr=[], label=r'$(p-\bar{p})-2(K^+-K^-)$', **{key:val for key, val in plot_config['combo2'].items() if key != 'label'})
    ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )    
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_xi.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_xi.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_coal_xi.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs


def plot_ds_comparison(dict_input, figs, input_path, **kwargs):
    files = dict_input['dv1dy_coal']
    fig_coal = plt.figure(figsize=(kwargs['ncols']*4, kwargs['nrows']*4))
    gs_coal = fig_coal.add_gridspec(ncols=kwargs['ncols'], nrows=kwargs['nrows'], hspace=0, wspace=0)
    ax_coal = gs_coal.subplots(sharex='col', sharey='row')
    ax_coal = ax_coal.flatten()
    scaling = {0: 4, 1: 4, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}
    if kwargs['ncols'] == 3:
        scaling = {0: 4, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
    for i, f in enumerate(reversed(files)):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            x = np.array(data_dict['x'])
            delta_u = np.array(data_dict['delta_u'])
            delta_u_err = np.array(data_dict['delta_u_err'])
            delta_d = np.array(data_dict['delta_d'])
            delta_d_err = np.array(data_dict['delta_d_err'])
            delta_s = np.array(data_dict['delta_s'])
            delta_s_err = np.array(data_dict['delta_s_err'])
            # a1 = data_dict['a1']
            # a1_err = data_dict['a1_err']

            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
            scale = scaling[i]
            ax_coal[i].errorbar(x+1, delta_s*scale, yerr=delta_s_err*scale, **plot_config['delta_s'])
            ax_coal[i].errorbar(x, delta_d*scale, yerr=delta_d_err*scale, **plot_config['delta_d'])
            ax_coal[i].errorbar(x-1, delta_u*scale, yerr=delta_u_err*scale, **plot_config['delta_u'])
            # ax_coal[i].errorbar(x, a1*scale, yerr=a1_err*scale, ls='none', capsize=2, ms=8)
            ax_coal[i].annotate(r'$\sqrt{s_{\text{NN}}}=$' + energy, xy=(0.85, 0.9), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            if scale != 1:
                ax_coal[i].annotate(f'{scale}x', xy=(0.2, 0.2), fontsize=15, xycoords='axes fraction', horizontalalignment='right')
            ax_coal[i].hlines(0, 0, 80, linestyles='--', colors='k')
            ax_coal[i].set_ylim(-0.119, 0.109)
    fig_coal.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r'$\text{Centrality (%)}$', fontsize=15)
    plt.ylabel(r'$\Delta dv_1/dy$', fontsize=15, labelpad=20)
    if kwargs['ncols'] == 4:
        index_legend = 7
    else:
        index_legend = 0
    ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['delta_s'])
    ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['delta_d'])
    ax_coal[index_legend].errorbar([], [], yerr=[], **plot_config['delta_u'])
    ax_coal[index_legend].tick_params(axis='x', which='both', length=0)
    ax_coal[index_legend].legend(fontsize=15, frameon=False, loc='lower center' )    
    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_ds_comparison.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_ds_comparison.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_ds_comparison.svg', format='svg')
    figs.append(fig_coal)
    plt.close()
    return figs    


def plot_dv1dy_energy_dependence(dict_input, figs, input_path, proton_fit='linear'):
    files = dict_input['dv1dy_coal']
    energies = [float(f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.').replace('GeV', '')) for f in files]
    pikp_correspondence = {'pions': ['piplus', 'piminus'], 'kaons': ['kplus', 'kminus'], 'protons': ['proton', 'antiproton']}
    datapoints = {}
    pikp_slopes = PikpMergedSlope().get_data()
    cent_ranges = ['1040', '4080', '5080', '010']
    # initialize
    for particle in ['lambda', 'lambdabar', 'deltalambda', 'piplus', 'piminus', 'kplus', 'kminus', 'proton', 'antiproton', 'delta_lambdas', 'delta_pions', 'delta_kaons', 'delta_protons']:
        for cent in cent_ranges:
            key = f'{particle}_{cent}'
            datapoints[key] = DataPoint([], [], [])

    # extract data
    for i, f in enumerate(files):
        # lambda and lambdabar
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            for particle in ['lambda', 'lambdabar', 'deltalambda']:
                for cent in cent_ranges:
                    key = f'{particle}_{cent}'
                    value = data_dict[f'dv1dy_{particle}_{cent}']['value']
                    error_stat = data_dict[f'dv1dy_{particle}_{cent}']['error_stat']
                    error_sys = data_dict[f'dv1dy_{particle}_{cent}']['error_sys']
                    datapoints[key].add_point(value, error_stat, error_sys)
        
        # pions, kaons, protons from Aditya
        energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')
        for particle in ['pions', 'kaons', 'protons']:
            fit = proton_fit if particle == 'protons' else 'linear'
            for cent in cent_ranges:
                # positive
                key = f'{pikp_correspondence[particle][0]}_{cent}'
                value = pikp_slopes[energy][particle][f'{cent}_{fit}']['pos']
                error_stat = pikp_slopes[energy][particle][f'{cent}_{fit}']['pos_err']
                error_sys = pikp_slopes[energy][particle][f'{cent}_{fit}']['pos_systematics']
                datapoints[key].add_point(value, error_stat, error_sys)
                # negative
                key = f'{pikp_correspondence[particle][1]}_{cent}'
                value = pikp_slopes[energy][particle][f'{cent}_{fit}']['neg']
                error_stat = pikp_slopes[energy][particle][f'{cent}_{fit}']['neg_err']
                error_sys = pikp_slopes[energy][particle][f'{cent}_{fit}']['neg_systematics']
                datapoints[key].add_point(value, error_stat, error_sys)
                # delta
                key = f'delta_{particle}_{cent}'
                value = pikp_slopes[energy][particle][f'{cent}_{fit}']['delta']
                error_stat = pikp_slopes[energy][particle][f'{cent}_{fit}']['delta_err']
                error_sys = pikp_slopes[energy][particle][f'{cent}_{fit}']['delta_systematics']
                datapoints[key].add_point(value, error_stat, error_sys)
    
    # calculate combos
    for cent in cent_ranges:
        # delta lambdas
        key = f'delta_lambdas_{cent}'
        # datapoints[key] = datapoints['lambda_' + cent] - datapoints['lambdabar_' + cent]
        datapoints[key] = datapoints['deltalambda_' + cent]

        # combo1: delta_protons - delta_kaons
        key = f'combo1_{cent}'
        datapoints[key] = datapoints['delta_protons_' + cent] - datapoints['delta_kaons_' + cent]

        # combo2: delta_protons
        key = f'combo2_{cent}'
        datapoints[key] = datapoints['delta_protons_' + cent]

        # combo3: proton + kminus - 0.5(piplus + piminus)
        # or piplus + piminus + kminus - antiproton
        key = f'combo3_{cent}'
        datapoints[key] = datapoints['proton_' + cent] + datapoints['kminus_' + cent] - 0.5 * (datapoints['piplus_' + cent] + datapoints['piminus_' + cent])
        # datapoints[key] = datapoints['piplus_' + cent] + datapoints['piminus_' + cent] + datapoints['kminus_' + cent] - datapoints['antiproton_' + cent]

        # combo4: antiproton + kplus - 0.5(piplus + piminus)
        # or piplus + piminus + kplus - proton
        key = f'combo4_{cent}'
        datapoints[key] = datapoints['antiproton_' + cent] + datapoints['kplus_' + cent] - 0.5 * (datapoints['piplus_' + cent] + datapoints['piminus_' + cent])
        # datapoints[key] = datapoints['piplus_' + cent] + datapoints['piminus_' + cent] + datapoints['kplus_' + cent] - datapoints['proton_' + cent]

    # fig_dep_010, ax_dep_010 = plt.subplots()
    for is_horizontal in [False, True]:
        fig_dep = plt.figure(figsize=(8, 12)) if not is_horizontal else plt.figure(figsize=(20, 8))
        gs_dep = fig_dep.add_gridspec(ncols=1, nrows=3, hspace=0, wspace=0) if not is_horizontal else fig_dep.add_gridspec(ncols=3, nrows=1, hspace=0, wspace=0)
        ax_dep = gs_dep.subplots(sharex='col', sharey='row')
        ax_dep = ax_dep.flatten()

        ax_dep_010 = ax_dep[0]
        ax_dep_010.errorbar(np.array(energies)+0.2, datapoints['delta_lambdas_010'].value,
                                yerr=datapoints['delta_lambdas_010'].stat_error, **plot_config['Lambda'])
        for i, energy in enumerate(energies):
            ax_dep_010.fill_between(np.array([energies[i]+0.2-0.15, energies[i]+0.2+0.15]),
                                    y1=datapoints['delta_lambdas_010'].value[i] - datapoints['delta_lambdas_010'].sys_error[i],
                                    y2=datapoints['delta_lambdas_010'].value[i] + datapoints['delta_lambdas_010'].sys_error[i],
                                    color=plot_config['Lambda']['color'], alpha=0.4, linewidth=0)
        ax_dep_010.errorbar(np.array(energies)-0.2, datapoints['combo2_010'].value,
                                yerr=datapoints['combo2_010'].stat_error, **plot_config['combo2'])
        for i, energy in enumerate(energies):
            ax_dep_010.fill_between(np.array([energies[i]-0.2-0.15, energies[i]-0.2+0.15]),
                                    y1=datapoints['combo2_010'].value[i] - datapoints['combo2_010'].sys_error[i],
                                    y2=datapoints['combo2_010'].value[i] + datapoints['combo2_010'].sys_error[i],
                                    color=plot_config['combo2']['color'], alpha=0.4, linewidth=0)
        ax_dep_010.errorbar(np.array(energies), datapoints['combo1_010'].value,
                                yerr=datapoints['combo1_010'].stat_error, **plot_config['combo'])
        for i, energy in enumerate(energies):
            ax_dep_010.fill_between(np.array([energies[i]-0.15, energies[i]+0.15]),
                                    y1=datapoints['combo1_010'].value[i] - datapoints['combo1_010'].sys_error[i],
                                    y2=datapoints['combo1_010'].value[i] + datapoints['combo1_010'].sys_error[i],
                                    color=plot_config['combo']['color'], alpha=0.4, linewidth=0)
        # calculate chi2/ndf
        chi2ndf_2 = calculate_chi2_per_ndf(datapoints['delta_lambdas_010']-datapoints['combo2_010'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        chi2ndf_1 = calculate_chi2_per_ndf(datapoints['delta_lambdas_010']-datapoints['combo1_010'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        ax_dep_010.axhline(0, linestyle='dashed', color='black')
        # ax_dep_010.legend(loc='upper right', fontsize=15, frameon=False)
        # ax_dep_010.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
        if is_horizontal:
            ax_dep_010.annotate('0-10%', xy=(0.45, 0.85), xycoords='axes fraction', fontsize=24)
            ax_dep_010.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.78), xycoords='axes fraction', fontsize=18)
            ax_dep_010.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.71), xycoords='axes fraction', fontsize=18)
            ax_dep_010.set_xticks(energies, labels=energies)
        else:
            ax_dep_010.annotate('0-10%', xy=(0.45, 0.85), xycoords='axes fraction', fontsize=20)
            ax_dep_010.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.75), xycoords='axes fraction', fontsize=14)
            ax_dep_010.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.65), xycoords='axes fraction', fontsize=14)
        ax_dep_010.tick_params(**tick_params)
        ax_dep_010.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

        ax_dep_1040 = ax_dep[1]
        ax_dep_1040.errorbar(np.array(energies)+0.2, datapoints['delta_lambdas_1040'].value, 
                            yerr=datapoints['delta_lambdas_1040'].stat_error, **{k: v for k, v in plot_config['Lambda'].items() if k != 'label'}, label='\n'.join([r'$\Lambda^0-\bar{\Lambda}^0$', r'$0.4<p_{T}<1.8$']))
        for i, energy in enumerate(energies):
            ax_dep_1040.fill_between(np.array([energies[i]+0.2-0.15, energies[i]+0.2+0.15]),
                                    y1=datapoints['delta_lambdas_1040'].value[i] - datapoints['delta_lambdas_1040'].sys_error[i],
                                    y2=datapoints['delta_lambdas_1040'].value[i] + datapoints['delta_lambdas_1040'].sys_error[i],
                                    color=plot_config['Lambda']['color'], alpha=0.4, linewidth=0)
        ax_dep_1040.errorbar(np.array(energies)-0.2, datapoints['combo2_1040'].value,
                                yerr=datapoints['combo2_1040'].stat_error, **{k: v for k, v in plot_config['combo2'].items() if k != 'label'}, label='\n'.join([r'$p-\bar{p}$', r'$0.4<p_{T}<1.8$, $p<2.0$']))    
        for i, energy in enumerate(energies):
            ax_dep_1040.fill_between(np.array([energies[i]-0.2-0.15, energies[i]-0.2+0.15]),
                                    y1=datapoints['combo2_1040'].value[i] - datapoints['combo2_1040'].sys_error[i],
                                    y2=datapoints['combo2_1040'].value[i] + datapoints['combo2_1040'].sys_error[i],
                                    color=plot_config['combo2']['color'], alpha=0.4, linewidth=0)
        ax_dep_1040.errorbar(np.array(energies), datapoints['combo1_1040'].value, 
                            yerr=datapoints['combo1_1040'].stat_error, **{k: v for k, v in plot_config['combo'].items() if k != 'label'}, label='\n'.join([r'$(p-\bar{p})-(K^+-K^-)$', r'$0.28<p_{T}<1.2$, $p<1.6$']))
        for i, energy in enumerate(energies):
            ax_dep_1040.fill_between(np.array([energies[i]-0.15, energies[i]+0.15]),
                                    y1=datapoints['combo1_1040'].value[i] - datapoints['combo1_1040'].sys_error[i],
                                    y2=datapoints['combo1_1040'].value[i] + datapoints['combo1_1040'].sys_error[i],
                                    color=plot_config['combo']['color'], alpha=0.4, linewidth=0)
        # calculate chi2/ndf
        chi2ndf_2 = calculate_chi2_per_ndf(datapoints['delta_lambdas_1040']-datapoints['combo2_1040'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        chi2ndf_1 = calculate_chi2_per_ndf(datapoints['delta_lambdas_1040']-datapoints['combo1_1040'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        ax_dep_1040.axhline(0, linestyle='dashed', color='black')
        
        # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
        if is_horizontal:
            ax_dep_1040.set_xticks(energies, labels=energies)
            ax_dep_1040.annotate('10-40%', xy=(0.45, 0.21), xycoords='axes fraction', fontsize=24)
            ax_dep_1040.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.14), xycoords='axes fraction', fontsize=18)
            ax_dep_1040.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.07), xycoords='axes fraction', fontsize=18)
            ax_dep_1040.legend(loc='upper right', fontsize=18, frameon=False, title=r'$p_{T}$, $p$ in GeV/$c$', title_fontsize=15)
        else:
            ax_dep_1040.annotate('10-40%', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
            ax_dep_1040.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=14)
            ax_dep_1040.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.35), xycoords='axes fraction', fontsize=14)
            ax_dep_1040.legend(loc='upper right', fontsize=13, frameon=False, title=r'$p_{T}$, $p$ in GeV/$c$', title_fontsize=13)
        ax_dep_1040.tick_params(**tick_params)
        ax_dep_1040.set_ylim(ax_dep_1040.get_ylim()[0], 0.0849)
        ax_dep_1040.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        
        ax_dep_4080 = ax_dep[2]
        ax_dep_4080.errorbar(np.array(energies)+0.2, datapoints['delta_lambdas_4080'].value, 
                            yerr=datapoints['delta_lambdas_4080'].stat_error, **plot_config['Lambda'])
        for i, energy in enumerate(energies):
            ax_dep_4080.fill_between(np.array([energies[i]+0.2-0.15, energies[i]+0.2+0.15]),
                                    y1=datapoints['delta_lambdas_4080'].value[i] - datapoints['delta_lambdas_4080'].sys_error[i],
                                    y2=datapoints['delta_lambdas_4080'].value[i] + datapoints['delta_lambdas_4080'].sys_error[i],
                                    color=plot_config['Lambda']['color'], alpha=0.4, linewidth=0)
        ax_dep_4080.errorbar(np.array(energies)-0.2, datapoints['combo2_4080'].value,
                                yerr=datapoints['combo2_4080'].stat_error, **plot_config['combo2'])
        for i, energy in enumerate(energies):
            ax_dep_4080.fill_between(np.array([energies[i]-0.2-0.15, energies[i]-0.2+0.15]),
                                    y1=datapoints['combo2_4080'].value[i] - datapoints['combo2_4080'].sys_error[i],
                                    y2=datapoints['combo2_4080'].value[i] + datapoints['combo2_4080'].sys_error[i],
                                    color=plot_config['combo2']['color'], alpha=0.4, linewidth=0)
        ax_dep_4080.errorbar(np.array(energies), datapoints['combo1_4080'].value, 
                            yerr=datapoints['combo1_4080'].stat_error, **plot_config['combo'])
        for i, energy in enumerate(energies):
            ax_dep_4080.fill_between(np.array([energies[i]-0.15, energies[i]+0.15]),
                                    y1=datapoints['combo1_4080'].value[i] - datapoints['combo1_4080'].sys_error[i],
                                    y2=datapoints['combo1_4080'].value[i] + datapoints['combo1_4080'].sys_error[i],
                                    color=plot_config['combo']['color'], alpha=0.4, linewidth=0)
            
        # calculate chi2/ndf
        chi2ndf_2 = calculate_chi2_per_ndf(datapoints['delta_lambdas_4080']-datapoints['combo2_4080'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        chi2ndf_1 = calculate_chi2_per_ndf(datapoints['delta_lambdas_4080']-datapoints['combo1_4080'], DataPoint(np.zeros(len(energies))), nparams=0) # compare with zero
        ax_dep_4080.axhline(0, linestyle='dashed', color='black')
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
        if is_horizontal:
            ax_dep_4080.set_xticks(energies, labels=energies)
            ax_dep_4080.annotate('40-80%', xy=(0.45, 0.85), xycoords='axes fraction', fontsize=24)
            ax_dep_4080.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.78), xycoords='axes fraction', fontsize=18)
            ax_dep_4080.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.71), xycoords='axes fraction', fontsize=18)
            ax_dep_4080.set_ylim(-0.0399, 0.0849)
        else:
            ax_dep_4080.annotate('40-80%', xy=(0.45, 0.85), xycoords='axes fraction', fontsize=20)
            ax_dep_4080.annotate(fr'$\chi^2$/ndf (p) = {chi2ndf_2:.2f}', xy=(0.45, 0.75), xycoords='axes fraction', fontsize=14)
            ax_dep_4080.annotate(fr'$\chi^2$/ndf (p - K) = {chi2ndf_1:.2f}', xy=(0.45, 0.65), xycoords='axes fraction', fontsize=14)
        ax_dep_4080.tick_params(**tick_params)
        ax_dep_4080.set_xticks(energies, labels=energies)
        ax_dep_4080.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

        # labels
        fig_dep.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.tick_params(**tick_params)
        plt.grid(False)
        if is_horizontal:
            plt.xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=24, labelpad=10)
            plt.ylabel(r'$\Delta dv_1/dy$', fontsize=24, labelpad=30)
        plt.xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=18, labelpad=10)
        plt.ylabel(r'$\Delta dv_1/dy$', fontsize=18, labelpad=18)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        # saving
        _suffix = f'_cubic_proton' if proton_fit == 'cubic' else ''
        out_base = input_path.replace('_yaml', '').replace('/sys_tag_0', '')
        if not is_horizontal:
            plt.savefig(out_base + f'/fig_3_vertical{_suffix}.pdf')
            plt.savefig(out_base + f'/fig_3_vertical{_suffix}.svg', format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.tight_layout()
            plt.savefig(out_base + f'/fig_3_horizontal{_suffix}.pdf')
            plt.savefig(out_base + f'/fig_3_horizontal{_suffix}.svg', format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()  

    fig_dep_5080, ax_dep_5080 = plt.subplots()
    ax_dep_5080.errorbar(np.array(energies)+0.2, datapoints['delta_lambdas_5080'].value,
                         yerr=datapoints['delta_lambdas_5080'].stat_error, **plot_config['Lambda'])
    for i, energy in enumerate(energies):
        ax_dep_5080.fill_between(np.array([energies[i]+0.2-0.15, energies[i]+0.2+0.15]),
                                 y1=datapoints['delta_lambdas_5080'].value[i] - datapoints['delta_lambdas_5080'].sys_error[i],
                                 y2=datapoints['delta_lambdas_5080'].value[i] + datapoints['delta_lambdas_5080'].sys_error[i],
                                 color=plot_config['Lambda']['color'], alpha=0.4, linewidth=0)
    ax_dep_5080.errorbar(np.array(energies)-0.2, datapoints['combo2_5080'].value,
                            yerr=datapoints['combo2_5080'].stat_error, **plot_config['combo2'])
    for i, energy in enumerate(energies):
        ax_dep_5080.fill_between(np.array([energies[i]-0.2-0.15, energies[i]-0.2+0.15]),
                                 y1=datapoints['combo2_5080'].value[i] - datapoints['combo2_5080'].sys_error[i],
                                 y2=datapoints['combo2_5080'].value[i] + datapoints['combo2_5080'].sys_error[i],
                                 color=plot_config['combo2']['color'], alpha=0.4, linewidth=0)
    ax_dep_5080.errorbar(np.array(energies), datapoints['combo1_5080'].value,
                            yerr=datapoints['combo1_5080'].stat_error, **plot_config['combo'])
    for i, energy in enumerate(energies):
        ax_dep_5080.fill_between(np.array([energies[i]-0.15, energies[i]+0.15]),
                                 y1=datapoints['combo1_5080'].value[i] - datapoints['combo1_5080'].sys_error[i],
                                 y2=datapoints['combo1_5080'].value[i] + datapoints['combo1_5080'].sys_error[i],
                                 color=plot_config['combo']['color'], alpha=0.4, linewidth=0)

    # calculate sigma deviations
    ax_dep_5080.axhline(0, linestyle='dashed', color='black')
    ax_dep_5080.legend(loc='lower right', fontsize=15, frameon=False)
    ax_dep_5080.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_5080.set_ylabel(r'$\Delta dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_5080.annotate('50-80%', xy=(0.45, 0.6), xycoords='axes fraction', fontsize=20)
    ax_dep_5080.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=20)
    # ax_dep_5080.set_xscale('log')
    ax_dep_5080.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_5080.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_5080)
    plt.close()

    fig_dep_lambda = plt.figure(figsize=(6, 12))
    gs_lambda = fig_dep_lambda.add_gridspec(ncols=1, nrows=3, hspace=0, wspace=0)
    ax_dep_lambda = gs_lambda.subplots(sharex='col', sharey='row')
    ax_dep_lambda = ax_dep_lambda.flatten()
    ax_dep_010_lambda = ax_dep_lambda[0]
    ax_dep_010_lambda.errorbar(np.array(energies)+0.2, datapoints['lambda_010'].value,
                               yerr=datapoints['lambda_010'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                               label=r'$\Lambda^0$')
    ax_dep_010_lambda.errorbar(np.array(energies), datapoints['combo3_010'].value,
                               yerr=datapoints['combo3_010'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                               label=r'$p + K^- - 0.5(\pi^+ + \pi^-)$')
    ax_dep_010_lambda.errorbar(np.array(energies)-0.2, datapoints['proton_010'].value,
                               yerr=datapoints['proton_010'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                               label=r'$p$')
    ax_dep_010_lambda.axhline(0, linestyle='dashed', color='black')    
    ax_dep_010_lambda.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_010_lambda.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_010_lambda.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_010_lambda.annotate('0-10%', xy=(0.45, 0.6), xycoords='axes fraction', fontsize=20)
    # ax_dep_010_lambda.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_010_lambda.set_xscale('log')
    ax_dep_010_lambda.set_xticks(energies, labels=energies)
    # plt.figure(fig_dep_010_lambda.number)
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambda.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambda.eps', format='eps')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambda.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    # figs.append(fig_dep_010_lambda)
    # plt.close()

    # fig_dep_1040_lambda, ax_dep_1040_lambda = plt.subplots()
    ax_dep_1040_lambda = ax_dep_lambda[1]
    ax_dep_1040_lambda.errorbar(np.array(energies)+0.2, datapoints['lambda_1040'].value,
                                yerr=datapoints['lambda_1040'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                label=r'$\Lambda^0$')
    ax_dep_1040_lambda.errorbar(np.array(energies), datapoints['combo3_1040'].value,
                                yerr=datapoints['combo3_1040'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                label=r'$p + K^- - 0.5(\pi^+ + \pi^-)$')
    ax_dep_1040_lambda.errorbar(np.array(energies)-0.2, datapoints['proton_1040'].value,
                                yerr=datapoints['proton_1040'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                label=r'$p$')
    ax_dep_1040_lambda.axhline(0, linestyle='dashed', color='black')
    # ax_dep_1040_lambda.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_1040_lambda.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_1040_lambda.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_1040_lambda.annotate('10-40%', xy=(0.45, 0.8), xycoords='axes fraction', fontsize=20)
    # ax_dep_1040_lambda.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_1040_lambda.set_xscale('log')
    ax_dep_1040_lambda.set_xticks(energies, labels=energies)
    # plt.figure(fig_dep_1040_lambda.number)
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambda.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambda.eps', format='eps')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambda.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    # figs.append(fig_dep_1040_lambda)
    # plt.close()

    # fig_dep_4080_lambda, ax_dep_4080_lambda = plt.subplots()
    ax_dep_4080_lambda = ax_dep_lambda[2]
    ax_dep_4080_lambda.errorbar(np.array(energies)+0.2, datapoints['lambda_4080'].value,
                                yerr=datapoints['lambda_4080'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                label=r'$\Lambda^0$')
    ax_dep_4080_lambda.errorbar(np.array(energies), datapoints['combo3_4080'].value,
                                yerr=datapoints['combo3_4080'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                label=r'$p + K^- - 0.5(\pi^+ + \pi^-)$')
    ax_dep_4080_lambda.errorbar(np.array(energies)-0.2, datapoints['proton_4080'].value,
                                yerr=datapoints['proton_4080'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                label=r'$p$')
    ax_dep_4080_lambda.axhline(0, linestyle='dashed', color='black')
    # ax_dep_4080_lambda.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_4080_lambda.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_4080_lambda.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_4080_lambda.annotate('40-80%', xy=(0.45, 0.8), xycoords='axes fraction', fontsize=20)
    # ax_dep_4080_lambda.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_4080_lambda.set_xscale('log')
    ax_dep_4080_lambda.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_lambda.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambda.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambda.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambda.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_lambda)
    plt.close()

    # fig_dep_010_lambdabar, ax_dep_010_lambdabar = plt.subplots()
    fig_dep_lambdabar = plt.figure(figsize=(6, 12))
    gs_dep_lambdabar = fig_dep_lambdabar.add_gridspec(ncols=1, nrows=3, hspace=0, wspace=0)
    ax_dep_lambdabar = gs_dep_lambdabar.subplots(sharex='col', sharey='row')
    ax_dep_lambdabar = ax_dep_lambdabar.flatten()
    ax_dep_010_lambdabar = ax_dep_lambdabar[0]
    ax_dep_010_lambdabar.errorbar(np.array(energies)+0.2, datapoints['lambdabar_010'].value,
                                  yerr=datapoints['lambdabar_010'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                  label=r'$\bar{\Lambda}^0$')
    ax_dep_010_lambdabar.errorbar(np.array(energies), datapoints['combo4_010'].value,
                                  yerr=datapoints['combo4_010'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                  label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-)$')
    ax_dep_010_lambdabar.errorbar(np.array(energies)-0.2, datapoints['antiproton_010'].value,
                                  yerr=datapoints['antiproton_010'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                  label=r'$\bar{p}$')
    ax_dep_010_lambdabar.axhline(0, linestyle='dashed', color='black')
    ax_dep_010_lambdabar.legend(loc='lower right', fontsize=15, frameon=False)
    ax_dep_010_lambdabar.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_010_lambdabar.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_010_lambdabar.annotate('0-10%', xy=(0.75, 0.85), xycoords='axes fraction', fontsize=20)
    # ax_dep_010_lambdabar.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_010_lambdabar.set_xscale('log')
    ax_dep_010_lambdabar.set_xticks(energies, labels=energies)
    # plt.figure(fig_dep_010_lambdabar.number)
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambdabar.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambdabar.eps', format='eps')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_010_lambdabar.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    # figs.append(fig_dep_010_lambdabar)
    # plt.close()

    # fig_dep_1040_lambdabar, ax_dep_1040_lambdabar = plt.subplots()
    ax_dep_1040_lambdabar = ax_dep_lambdabar[1]
    ax_dep_1040_lambdabar.errorbar(np.array(energies)+0.2, datapoints['lambdabar_1040'].value,
                                   yerr=datapoints['lambdabar_1040'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                   label=r'$\bar{\Lambda}^0$')
    ax_dep_1040_lambdabar.errorbar(np.array(energies), datapoints['combo4_1040'].value,
                                   yerr=datapoints['combo4_1040'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                   label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-)$')
    ax_dep_1040_lambdabar.errorbar(np.array(energies)-0.2, datapoints['antiproton_1040'].value,
                                   yerr=datapoints['antiproton_1040'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                   label=r'$\bar{p}$')
    ax_dep_1040_lambdabar.axhline(0, linestyle='dashed', color='black')
    # ax_dep_1040_lambdabar.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_1040_lambdabar.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_1040_lambdabar.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_1040_lambdabar.annotate('10-40%', xy=(0.75, 0.85), xycoords='axes fraction', fontsize=20)
    # ax_dep_1040_lambdabar.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_1040_lambdabar.set_xscale('log')
    ax_dep_1040_lambdabar.set_xticks(energies, labels=energies)
    # plt.figure(fig_dep_1040_lambdabar.number)
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambdabar.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambdabar.eps', format='eps')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_1040_lambdabar.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    # figs.append(fig_dep_1040_lambdabar)
    # plt.close()

    # fig_dep_4080_lambdabar, ax_dep_4080_lambdabar = plt.subplots()
    ax_dep_4080_lambdabar = ax_dep_lambdabar[2]
    ax_dep_4080_lambdabar.errorbar(np.array(energies)+0.2, datapoints['lambdabar_4080'].value,
                                   yerr=datapoints['lambdabar_4080'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                   label=r'$\bar{\Lambda}^0$')
    ax_dep_4080_lambdabar.errorbar(np.array(energies), datapoints['combo4_4080'].value, 
                                   yerr=datapoints['combo4_4080'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                   label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-)$')
    ax_dep_4080_lambdabar.errorbar(np.array(energies)-0.2, datapoints['antiproton_4080'].value,
                                   yerr=datapoints['antiproton_4080'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                   label=r'$\bar{p}$')
    ax_dep_4080_lambdabar.axhline(0, linestyle='dashed', color='black')
    # ax_dep_4080_lambdabar.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_4080_lambdabar.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_4080_lambdabar.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_4080_lambdabar.annotate('40-80%', xy=(0.75, 0.8), xycoords='axes fraction', fontsize=20)
    # ax_dep_4080_lambdabar.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_4080_lambdabar.set_xscale('log')
    ax_dep_4080_lambdabar.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_lambdabar.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambdabar.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambdabar.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_lambdabar.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_lambdabar)
    plt.close()

    fig_dep_5080_lambda, ax_dep_5080_lambda = plt.subplots()
    ax_dep_5080_lambda.errorbar(np.array(energies)+0.2, datapoints['lambda_5080'].value,
                                yerr=datapoints['lambda_5080'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                label=r'$\Lambda^0$')
    ax_dep_5080_lambda.errorbar(np.array(energies), datapoints['combo3_5080'].value,
                                yerr=datapoints['combo3_5080'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                label=r'$p + K^- - 0.5(\pi^+ + \pi^-)$')
    ax_dep_5080_lambda.errorbar(np.array(energies)-0.2, datapoints['proton_5080'].value,
                                yerr=datapoints['proton_5080'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                label=r'$p$')
    ax_dep_5080_lambda.axhline(0, linestyle='dashed', color='black')
    ax_dep_5080_lambda.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_5080_lambda.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_5080_lambda.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_5080_lambda.annotate('50-80%', xy=(0.45, 0.6), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_lambda.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_5080_lambda.set_xscale('log')
    ax_dep_5080_lambda.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_5080_lambda.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambda.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambda.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambda.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_5080_lambda)
    plt.close()

    fig_dep_5080_lambdabar, ax_dep_5080_lambdabar = plt.subplots()
    ax_dep_5080_lambdabar.errorbar(np.array(energies)+0.2, datapoints['lambdabar_5080'].value,
                                      yerr=datapoints['lambdabar_5080'].stat_error, **{key: val for key, val in plot_config['Lambda'].items() if key != 'label'},
                                      label=r'$\bar{\Lambda}^0$')
    ax_dep_5080_lambdabar.errorbar(np.array(energies), datapoints['combo4_5080'].value,
                                        yerr=datapoints['combo4_5080'].stat_error, **{key: val for key, val in plot_config['combo'].items() if key != 'label'},
                                        label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-)$')
    ax_dep_5080_lambdabar.errorbar(np.array(energies)-0.2, datapoints['antiproton_5080'].value,
                                      yerr=datapoints['antiproton_5080'].stat_error, **{key: val for key, val in plot_config['combo2'].items() if key != 'label'},
                                      label=r'$\bar{p}$')
    ax_dep_5080_lambdabar.axhline(0, linestyle='dashed', color='black')
    ax_dep_5080_lambdabar.legend(loc='upper right', fontsize=15, frameon=False)
    ax_dep_5080_lambdabar.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_5080_lambdabar.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_5080_lambdabar.annotate('50-80%', xy=(0.45, 0.6), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_lambdabar.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.55), xycoords='axes fraction', fontsize=20)
    # ax_dep_5080_lambdabar.set_xscale('log')
    ax_dep_5080_lambdabar.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_5080_lambdabar.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambdabar.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambdabar.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_energy_dependence_5080_lambdabar.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_5080_lambdabar)
    plt.close()

    ### output data points to multiple csvs
    for i, f in enumerate(files):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            centrality = ['0-10%', '10-40%', '40-80%', '50-80%']
            dv1dy_lambda = []
            dv1dy_lambda_err = []
            dv1dy_lambdabar = []
            dv1dy_lambdabar_err = []
            dv1dy_deltalambda = []
            dv1dy_deltalambda_err = []
            for cent in ['010', '1040', '4080', '5080']:
                dv1dy_lambda.append(data_dict['dv1dy_lambda_' + cent]['value'])
                dv1dy_lambda_err.append(data_dict['dv1dy_lambda_' + cent]['error'])
                dv1dy_lambdabar.append(data_dict['dv1dy_lambdabar_' + cent]['value'])
                dv1dy_lambdabar_err.append(data_dict['dv1dy_lambdabar_' + cent]['error'])
                dv1dy_deltalambda.append(data_dict['dv1dy_deltalambda_' + cent]['value'])
                dv1dy_deltalambda_err.append(data_dict['dv1dy_deltalambda_' + cent]['error'])
            dv1dy_lambda = np.array(dv1dy_lambda)
            dv1dy_lambda_err = np.array(dv1dy_lambda_err)
            dv1dy_lambdabar = np.array(dv1dy_lambdabar)
            dv1dy_lambdabar_err = np.array(dv1dy_lambdabar_err)
            dv1dy_deltalambda = np.array(dv1dy_deltalambda)
            dv1dy_deltalambda_err = np.array(dv1dy_deltalambda_err)
            energy = f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.')

            # piKp combo data (matches fig_3: combo2 = delta_proton, combo1 = delta_proton - delta_kaon)
            delta_proton = []
            delta_proton_stat = []
            delta_proton_sys = []
            combo1 = []
            combo1_stat = []
            combo1_sys = []
            for cent in ['010', '1040', '4080', '5080']:
                dp = pikp_slopes[energy]['protons'][f'{cent}_linear']['delta']
                dp_stat = pikp_slopes[energy]['protons'][f'{cent}_linear']['delta_err']
                dp_sys = pikp_slopes[energy]['protons'][f'{cent}_linear']['delta_systematics']
                dk = pikp_slopes[energy]['kaons'][f'{cent}_linear']['delta']
                dk_stat = pikp_slopes[energy]['kaons'][f'{cent}_linear']['delta_err']
                dk_sys = pikp_slopes[energy]['kaons'][f'{cent}_linear']['delta_systematics']
                delta_proton.append(dp)
                delta_proton_stat.append(dp_stat)
                delta_proton_sys.append(dp_sys)
                combo1.append(dp - dk)
                combo1_stat.append(np.sqrt(dp_stat**2 + dk_stat**2))
                combo1_sys.append(np.sqrt(dp_sys**2 + dk_sys**2))
            delta_proton = np.array(delta_proton)
            delta_proton_stat = np.array(delta_proton_stat)
            delta_proton_sys = np.array(delta_proton_sys)
            combo1 = np.array(combo1)
            combo1_stat = np.array(combo1_stat)
            combo1_sys = np.array(combo1_sys)

            df = pd.DataFrame({
                'centrality': centrality,
                'dv1dy_lambda': dv1dy_lambda,
                'dv1dy_lambda_err': dv1dy_lambda_err,
                'dv1dy_lambdabar': dv1dy_lambdabar,
                'dv1dy_lambdabar_err': dv1dy_lambdabar_err,
                'delta_dv1dy': dv1dy_deltalambda,
                'delta_dv1dy_err': dv1dy_deltalambda_err,
                'delta_dv1dy_proton': delta_proton,
                'delta_dv1dy_proton_err': delta_proton_stat,
                'delta_dv1dy_combo': combo1,
                'delta_dv1dy_combo_err': combo1_stat
            })

            if 'yerr_sys' in data_dict:
                dv1dy_lambda_sys_err = []
                dv1dy_lambdabar_sys_err = []
                dv1dy_lambda_stat_err = []
                dv1dy_lambdabar_stat_err = []
                dv1dy_deltalambda_sys_err = []
                dv1dy_deltalambda_stat_err = []
                for cent in ['010', '1040', '4080', '5080']:
                    dv1dy_lambda_sys_err.append(data_dict['dv1dy_lambda_' + cent]['error_sys'])
                    dv1dy_lambda_stat_err.append(data_dict['dv1dy_lambda_' + cent]['error_stat'])
                    dv1dy_lambdabar_sys_err.append(data_dict['dv1dy_lambdabar_' + cent]['error_sys'])
                    dv1dy_lambdabar_stat_err.append(data_dict['dv1dy_lambdabar_' + cent]['error_stat'])
                    dv1dy_deltalambda_sys_err.append(data_dict['dv1dy_deltalambda_' + cent]['error_sys'])
                    dv1dy_deltalambda_stat_err.append(data_dict['dv1dy_deltalambda_' + cent]['error_stat'])
                dv1dy_lambda_sys_err = np.array(dv1dy_lambda_sys_err)
                dv1dy_lambda_stat_err = np.array(dv1dy_lambda_stat_err)
                dv1dy_lambdabar_sys_err = np.array(dv1dy_lambdabar_sys_err)
                dv1dy_lambdabar_stat_err = np.array(dv1dy_lambdabar_stat_err)
                dv1dy_deltalambda_sys_err = np.array(dv1dy_deltalambda_sys_err)
                dv1dy_deltalambda_stat_err = np.array(dv1dy_deltalambda_stat_err)

                df = pd.DataFrame({
                    'centrality': centrality,
                    'dv1dy_lambda': dv1dy_lambda,
                    'dv1dy_lambda_err_stat': dv1dy_lambda_stat_err,
                    'dv1dy_lambda_err_sys': dv1dy_lambda_sys_err,
                    'dv1dy_lambdabar': dv1dy_lambdabar,
                    'dv1dy_lambdabar_err_stat': dv1dy_lambdabar_stat_err,
                    'dv1dy_lambdabar_err_sys': dv1dy_lambdabar_sys_err,
                    'delta_dv1dy': dv1dy_deltalambda,
                    'delta_dv1dy_err_stat': dv1dy_deltalambda_stat_err,
                    'delta_dv1dy_err_sys': dv1dy_deltalambda_sys_err,
                    'delta_dv1dy_proton': delta_proton,
                    'delta_dv1dy_proton_err_stat': delta_proton_stat,
                    'delta_dv1dy_proton_err_sys': delta_proton_sys,
                    'delta_dv1dy_combo': combo1,
                    'delta_dv1dy_combo_err_stat': combo1_stat,
                    'delta_dv1dy_combo_err_sys': combo1_sys
                })
            df.to_csv(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + f'/data_points/dv1dy_{energy}_merged.csv', index=False)

    return figs


def plot_PT_energy_dependence(dict_input, figs, input_path):
    files = dict_input['dv1dy_coal']
    energies = [float(f.split('/')[-1].replace('.yaml', '').split('_')[-1].replace('p', '.').replace('GeV', '')) for f in files]
    lambda_1040 = []
    lambda_1040_err = []
    lambdabar_1040 = []
    lambdabar_1040_err = []
    lambda_5080 = []
    lambda_5080_err = []
    lambdabar_5080 = []
    lambdabar_5080_err = []

    combo_PT_1_1040 = []
    combo_PT_1_1040_err = []
    combo_PT_2_1040 = []
    combo_PT_2_1040_err = []
    combo_PT_1_5080 = []
    combo_PT_1_5080_err = []
    combo_PT_2_5080 = []
    combo_PT_2_5080_err = []

    combo_proton_1040 = []
    combo_proton_1040_err = []
    combo_proton_5080 = []
    combo_proton_5080_err = []
    combo_antiproton_1040 = []
    combo_antiproton_1040_err = []
    combo_antiproton_5080 = []
    combo_antiproton_5080_err = []

    for i, f in enumerate(files):
        with open(f, 'r') as file:
            data_dict = yaml.load(file, Loader=yaml.CLoader)
            
            lambda_merged_slope = ufloat(data_dict['dv1dy_lambda_1040']['value'], data_dict['dv1dy_lambda_1040']['error'])
            lambdabar_merged_slope = ufloat(data_dict['dv1dy_lambdabar_1040']['value'], data_dict['dv1dy_lambdabar_1040']['error'])
            lambda_1040.append(lambda_merged_slope.nominal_value)
            lambda_1040_err.append(lambda_merged_slope.std_dev)
            lambdabar_1040.append(lambdabar_merged_slope.nominal_value)
            lambdabar_1040_err.append(lambdabar_merged_slope.std_dev)
            lambda_merged_slope = ufloat(data_dict['dv1dy_lambda_4080']['value'], data_dict['dv1dy_lambda_4080']['error'])
            lambdabar_merged_slope = ufloat(data_dict['dv1dy_lambdabar_4080']['value'], data_dict['dv1dy_lambdabar_4080']['error'])
            lambda_5080.append(lambda_merged_slope.nominal_value)
            lambda_5080_err.append(lambda_merged_slope.std_dev)
            lambdabar_5080.append(lambdabar_merged_slope.nominal_value)
            lambdabar_5080_err.append(lambdabar_merged_slope.std_dev)

            combo_PT_1 = unumpy.uarray(np.array(data_dict['combo_PT_1_mod']), np.array(data_dict['combo_PT_1_mod_err']))
            combo_PT_2 = unumpy.uarray(np.array(data_dict['combo_PT_2_mod']), np.array(data_dict['combo_PT_2_mod_err']))
            combo_proton = unumpy.uarray(np.array(data_dict['p']), np.array(data_dict['p_err']))
            combo_antiproton = unumpy.uarray(np.array(data_dict['pbar']), np.array(data_dict['pbar_err']))
            temp = np.average(combo_PT_1[4:7], weights=1/unumpy.std_devs(combo_PT_1[4:7])**2)
            combo_PT_1_1040.append(temp.nominal_value)
            combo_PT_1_1040_err.append(temp.std_dev)
            temp = np.average(combo_PT_2[4:7], weights=1/unumpy.std_devs(combo_PT_2[4:7])**2)
            combo_PT_2_1040.append(temp.nominal_value)
            combo_PT_2_1040_err.append(temp.std_dev)
            temp = np.average(combo_PT_1[0:3], weights=1/unumpy.std_devs(combo_PT_1[0:3])**2)
            combo_PT_1_5080.append(temp.nominal_value)
            combo_PT_1_5080_err.append(temp.std_dev)
            temp = np.average(combo_PT_2[0:3], weights=1/unumpy.std_devs(combo_PT_2[0:3])**2)
            combo_PT_2_5080.append(temp.nominal_value)
            combo_PT_2_5080_err.append(temp.std_dev)
            temp = np.average(combo_proton[4:7], weights=1/unumpy.std_devs(combo_proton[4:7])**2)
            combo_proton_1040.append(temp.nominal_value)
            combo_proton_1040_err.append(temp.std_dev)
            temp = np.average(combo_proton[0:3], weights=1/unumpy.std_devs(combo_proton[0:3])**2)
            combo_proton_5080.append(temp.nominal_value)
            combo_proton_5080_err.append(temp.std_dev)
            temp = np.average(combo_antiproton[4:7], weights=1/unumpy.std_devs(combo_antiproton[4:7])**2)
            combo_antiproton_1040.append(temp.nominal_value)
            combo_antiproton_1040_err.append(temp.std_dev)
            temp = np.average(combo_antiproton[0:3], weights=1/unumpy.std_devs(combo_antiproton[0:3])**2)
            combo_antiproton_5080.append(temp.nominal_value)
            combo_antiproton_5080_err.append(temp.std_dev)

    lambda_1040 = np.array(lambda_1040)
    lambda_1040_err = np.array(lambda_1040_err)
    lambdabar_1040 = np.array(lambdabar_1040)
    lambdabar_1040_err = np.array(lambdabar_1040_err)
    lambda_5080 = np.array(lambda_5080)
    lambda_5080_err = np.array(lambda_5080_err)
    lambdabar_5080 = np.array(lambdabar_5080)
    lambdabar_5080_err = np.array(lambdabar_5080_err)

    combo_PT_1_1040 = np.array(combo_PT_1_1040)
    combo_PT_1_1040_err = np.array(combo_PT_1_1040_err)
    combo_PT_2_1040 = np.array(combo_PT_2_1040)
    combo_PT_2_1040_err = np.array(combo_PT_2_1040_err)
    combo_PT_1_5080 = np.array(combo_PT_1_5080)
    combo_PT_1_5080_err = np.array(combo_PT_1_5080_err)
    combo_PT_2_5080 = np.array(combo_PT_2_5080)
    combo_PT_2_5080_err = np.array(combo_PT_2_5080_err)
    combo_proton_1040 = np.array(combo_proton_1040)
    combo_proton_1040_err = np.array(combo_proton_1040_err)
    combo_proton_5080 = np.array(combo_proton_5080)
    combo_proton_5080_err = np.array(combo_proton_5080_err)
    combo_antiproton_1040 = np.array(combo_antiproton_1040)
    combo_antiproton_1040_err = np.array(combo_antiproton_1040_err)
    combo_antiproton_5080 = np.array(combo_antiproton_5080)
    combo_antiproton_5080_err = np.array(combo_antiproton_5080_err)

    fig_dep_1040_l, ax_dep_1040_l = plt.subplots()
    ax_dep_1040_l.errorbar(np.array(energies)+0.2, lambda_1040, yerr=lambda_1040_err, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$')
    ax_dep_1040_l.errorbar(np.array(energies), combo_PT_1_1040, yerr=combo_PT_1_1040_err, **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$p + K^- - 0.5(\pi^+ + \pi^-$')
    ax_dep_1040_l.errorbar(np.array(energies)-0.2, combo_proton_1040, yerr=combo_proton_1040_err, **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$p$')
    ax_dep_1040_l.hlines(0, *ax_dep_1040_l.get_xlim(), linestyles='dashed', color='black')
    ax_dep_1040_l.legend(fontsize=15, frameon=False)
    ax_dep_1040_l.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_1040_l.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_1040_l.annotate('10-40%', xy=(0.5, 0.6), xycoords='axes fraction', fontsize=20)
    ax_dep_1040_l.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=20)
    ax_dep_1040_l.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_1040_l.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_1040.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_1040.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_1040.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_1040_l)

    fig_dep_1040_lb, ax_dep_1040_lb = plt.subplots()
    ax_dep_1040_lb.errorbar(np.array(energies)+0.2, lambdabar_1040, yerr=lambdabar_1040_err, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$')
    ax_dep_1040_lb.errorbar(np.array(energies), combo_PT_2_1040, yerr=combo_PT_2_1040_err, **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-$')
    ax_dep_1040_lb.errorbar(np.array(energies)-0.2, combo_antiproton_1040, yerr=combo_antiproton_1040_err, **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$\bar{p}$')
    ax_dep_1040_lb.hlines(0, *ax_dep_1040_lb.get_xlim(), linestyles='dashed', color='black')
    ax_dep_1040_lb.legend(fontsize=15, frameon=False)
    ax_dep_1040_lb.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_1040_lb.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_1040_lb.annotate('10-40%', xy=(0.5, 0.6), xycoords='axes fraction', fontsize=20)
    ax_dep_1040_lb.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=20)
    ax_dep_1040_lb.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_1040_lb.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_1040.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_1040.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_1040.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_1040_lb)

    fig_dep_5080_l, ax_dep_5080_l = plt.subplots()
    ax_dep_5080_l.errorbar(np.array(energies)+0.2, lambda_5080, yerr=lambda_5080_err, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\Lambda^0$')
    ax_dep_5080_l.errorbar(np.array(energies), combo_PT_1_5080, yerr=combo_PT_1_5080_err, **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$p + K^- - 0.5(\pi^+ + \pi^-$')
    ax_dep_5080_l.errorbar(np.array(energies)-0.2, combo_proton_5080, yerr=combo_proton_5080_err, **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$p$')
    ax_dep_5080_l.hlines(0, *ax_dep_5080_l.get_xlim(), linestyles='dashed', color='black')
    ax_dep_5080_l.legend(fontsize=15, frameon=False)
    ax_dep_5080_l.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_5080_l.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_5080_l.annotate('40-80%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_l.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_l.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_5080_l.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_5080.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_5080.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambda_energy_dependence_5080.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_5080_l)

    fig_dep_5080_lb, ax_dep_5080_lb = plt.subplots()
    ax_dep_5080_lb.errorbar(np.array(energies)+0.2, lambdabar_5080, yerr=lambdabar_5080_err, **{key:val for key, val in plot_config['Lambda'].items() if key != 'label'}, label=r'$\bar{\Lambda}^0$')
    ax_dep_5080_lb.errorbar(np.array(energies), combo_PT_2_5080, yerr=combo_PT_2_5080_err, **{key:val for key, val in plot_config['combo'].items() if key != 'label'}, label=r'$\bar{p} + K^+ - 0.5(\pi^+ + \pi^-$')
    ax_dep_5080_lb.errorbar(np.array(energies)-0.2, combo_antiproton_5080, yerr=combo_antiproton_5080_err, **{key:val for key, val in plot_config['combo2'].items() if key != 'label'}, label=r'$\bar{p}$')
    ax_dep_5080_lb.hlines(0, *ax_dep_5080_lb.get_xlim(), linestyles='dashed', color='black')
    ax_dep_5080_lb.legend(fontsize=15, frameon=False)
    ax_dep_5080_lb.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_dep_5080_lb.set_ylabel(r'$dv_1/dy$', fontsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax_dep_5080_lb.annotate('40-80%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_lb.annotate(r'$\bf{STAR}\;\it{Preliminary}$', xy=(0.45, 0.45), xycoords='axes fraction', fontsize=20)
    ax_dep_5080_lb.set_xticks(energies, labels=energies)
    plt.figure(fig_dep_5080_lb.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_5080.pdf')
    # plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_5080.eps', format='eps')
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/dv1dy_PT_lambdabar_energy_dependence_5080.svg', format='svg', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    figs.append(fig_dep_5080_lb)

    plt.close()
    return figs


def plot_BESI_coal(dict_input, figs, input_path):
    delta_xi_v2 = np.array([None, 0.033, 0.021, 0.0076, 0.0063, 0.00509])
    delta_xi_v2_err = np.array([None, 0.020, 0.007, 0.0043, 0.0026, 0.00301])
    delta_lambda_v2 = np.array([0.042, 0.0257, 0.0153, 0.0126, 0.00829, 0.00597])
    delta_lambda_v2_err = np.array([0.014, 0.0040, 0.0013, 0.0007, 0.00047, 0.00056])
    delta_proton_v2 = np.array([0.0464, 0.0273, 0.0165, 0.0144, 0.0089, 0.0068])
    delta_proton_v2_err = np.array([0.0065, 0.0017, 0.0007, 0.0005, 0.0004, 0.0005])
    delta_kaon_v2 = np.array([0.0091, 0.0044, 0.00270, 0.00178, 0.00068, 0.00026])
    delta_kaon_v2_err = np.array([0.0018, 0.0007, 0.00031, 0.00019, 0.00013, 0.00017])

    energies = np.array([7.7, 11.5, 19.6, 27, 39, 62.4])
    fig_coal, ax_coal = plt.subplots()
    ax_coal.errorbar(energies, delta_lambda_v2, yerr=delta_lambda_v2_err,
                     fmt='o', color='blue', label=r'$\Delta \Lambda^0$')
    ax_coal.errorbar(energies-0.5, delta_proton_v2 - delta_kaon_v2, yerr=np.sqrt(delta_proton_v2_err**2 + delta_kaon_v2_err**2),
                     fmt='o', color='red', label=r'$\Delta p - \Delta K^-$')
    ax_coal.errorbar(energies+0.5, delta_proton_v2, yerr=delta_proton_v2_err,
                     fmt='o', color='green', label=r'$\Delta p$')
    chi2_pminusk = calculate_chi2_per_ndf(DataPoint(delta_lambda_v2, delta_lambda_v2_err) - DataPoint(delta_proton_v2, delta_proton_v2_err) + DataPoint(delta_kaon_v2, delta_kaon_v2_err), 
                                          DataPoint(np.zeros(len(energies))), nparams=0)
    chi2_p = calculate_chi2_per_ndf(DataPoint(delta_lambda_v2, delta_lambda_v2_err) - DataPoint(delta_proton_v2, delta_proton_v2_err), 
                                     DataPoint(np.zeros(len(energies))), nparams=0)
    ax_coal.annotate('BES-I', xy=(0.1, 0.9), xycoords='axes fraction', fontsize=16)
    ax_coal.annotate('10-40%', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=16)
    ax_coal.annotate('Statistical only', xy=(0.1, 0.7), xycoords='axes fraction', fontsize=16)
    ax_coal.annotate(r'$\chi^2/\text{ndf} (p-K) = %.2f$' % chi2_pminusk, xy=(0.4, 0.6), xycoords='axes fraction', fontsize=16)
    ax_coal.annotate(r'$\chi^2/\text{ndf} (p) = %.2f$' % chi2_p, xy=(0.4, 0.5), xycoords='axes fraction', fontsize=16)
    ax_coal.legend(loc='upper right', fontsize=15, frameon=False)
    ax_coal.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_coal.set_ylabel(r'$\Delta v_2$', fontsize=16)

    fig_coal_xi, ax_coal_xi = plt.subplots()
    ax_coal_xi.errorbar(energies[delta_xi_v2 != None], delta_xi_v2[delta_xi_v2 != None], yerr=delta_xi_v2_err[delta_xi_v2 != None],
                        fmt='o', color='blue', label=r'$\Delta \Xi^-$')
    ax_coal_xi.errorbar(energies-0.5, delta_lambda_v2 - delta_kaon_v2, yerr=np.sqrt(delta_lambda_v2_err**2 + delta_kaon_v2_err**2),
                        fmt='o', color='red', label=r'$\Delta \Lambda^0 - \Delta K^-$')
    ax_coal_xi.errorbar(energies+0.5, delta_lambda_v2, yerr=delta_lambda_v2_err,
                        fmt='o', color='green', label=r'$\Delta \Lambda^0$')
    ax_coal_xi.legend(loc='upper right', fontsize=15, frameon=False)
    ax_coal_xi.set_xlabel(r'$\sqrt{s_{\text{NN}}}$ (GeV)', fontsize=16)
    ax_coal_xi.set_ylabel(r'$\Delta v_2$', fontsize=16)

    plt.figure(fig_coal.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v2_coal_BESI.pdf')
    figs.append(fig_coal)
    plt.figure(fig_coal_xi.number)
    plt.savefig(input_path.replace('_yaml', '').replace('/sys_tag_0', '') + '/v2_coal_xi_BESI.pdf')
    figs.append(fig_coal_xi)
    
    plt.close()
    return figs

def main(dict_input, output_file=None):
    figs = []

    num_energies = len(dict_input['res'])
    ncol = 4 if num_energies == 7 else 3
    nrow = 2

    input_path = os.path.dirname(dict_input['invmass']).replace('/invmass', '')
    figs = plot_invmass_v1fit(dict_input, figs, input_path)
    figs = plot_res(dict_input, figs, input_path)
    # figs = plot_dv1dy_coal_one_energy(dict_input, figs, input_path)
    # figs = plot_dv1dy_KP(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_fig_2(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_dv1dy_coal(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_dv1dy_model(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_v1_merged_centrality(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_v1_pt(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_v1_y(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    # figs = plot_d3v1dy3_coal(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    # figs = plot_dv1dy_coal_xi(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    # figs = plot_dv1dy_baryons(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_PT_test(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_PT_test_mod(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_ds_comparison(dict_input, figs, input_path, ncols=ncol, nrows=nrow)
    figs = plot_dv1dy_energy_dependence(dict_input, figs, input_path)
    figs = plot_dv1dy_energy_dependence(dict_input, figs, input_path, proton_fit='cubic')
    figs = plot_PT_energy_dependence(dict_input, figs, input_path)
    figs = plot_BESI_coal(dict_input, figs, input_path)

    if output_file is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
        for fig in figs:
            pdf.savefig(fig)
        pdf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_invmass', type=str)
    parser.add_argument('--input_v1fit', type=str)
    parser.add_argument('--input_res', type=str, nargs='+')
    parser.add_argument('--input_dv1dy_coal', type=str, nargs='+')
    parser.add_argument('--input_dv1dy_coal_xi', type=str, nargs='+')
    parser.add_argument('--model_sim_urqmd', type=str, nargs='+')
    parser.add_argument('--model_sim_ampt', type=str, nargs='+')
    parser.add_argument('--output', type=str, help='a pdf report that includes all generated plots')
    parser.add_argument('--eff_energies', type=str, nargs='*', default=[], help='energies with efficiency correction applied')
    args = parser.parse_args()

    dict_input = {'invmass': args.input_invmass, 'v1fit': args.input_v1fit, 'res': args.input_res,
                  'dv1dy_coal': args.input_dv1dy_coal, 'dv1dy_coal_xi': args.input_dv1dy_coal_xi,
                  'model_sim_urqmd': args.model_sim_urqmd, 'model_sim_ampt': args.model_sim_ampt,
                  'eff_energies': args.eff_energies}
    main(dict_input, args.output)

