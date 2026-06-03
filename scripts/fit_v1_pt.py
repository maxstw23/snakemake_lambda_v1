import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from iminuit import cost, Minuit
from numba_stats import truncnorm
import scipy.odr as odr
import os
import sys
import pickle
import argparse
import time

import numpy as np
import pandas as pd
import uproot
import yaml

from uncertainties import unumpy, ufloat
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
import contextlib

from param_storage import ParamStorage
from simple_profile import SimpleProfile

def poly(x, *coeff):
    result = 0
    for order in range(len(coeff)):
        result += coeff[order] * x ** order
    return result


def poly_cdf(x, lb, rb, *coeff):
    result = 0
    norm = 0
    for order in range(len(coeff)):
        result += coeff[order] * x ** (order + 1) / (order + 1) - \
            coeff[order] * lb ** (order + 1) / (order + 1)
        norm += coeff[order] * rb ** (order + 1) / (order + 1) - \
            coeff[order] * lb ** (order + 1) / (order + 1)
    return result / norm


def poly_pdf(x, lb, rb, *coeff):
    result = 0
    norm = 0
    for order in range(len(coeff)):
        result += coeff[order] * x ** order
        norm += coeff[order] * rb ** (order + 1) / (order + 1) - \
            coeff[order] * lb ** (order + 1) / (order + 1)
    return result / norm


def breit_wigner(x, scale, pos, gamma):
    gamma24 = gamma ** 2 / 4
    return scale * gamma24 / ((x - pos) ** 2 + gamma24)


def gaus(x, scale, pos, sigma):
    # return abs(scale) * np.exp(-(x - pos) ** 2 / 2 / sigma ** 2)
    return 1. / (2 * np.pi) ** 0.5 / abs(sigma) * abs(scale) * np.exp(-(x - pos) ** 2 / 2 / sigma ** 2)


def double_gaus(x, scale1, scale2, pos, sigma1, sigma2):
    return gaus(x, scale1, pos, sigma1) + gaus(x, scale2, pos, sigma2)


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def hist_to_func(x, vals, edges):
    if len(vals) != len(edges) - 1:
        raise ValueError("Edges must have n+1 length!")
    return vals[np.digitize(x, edges) - 1]


def composite_sigma(scale_1, scale_2, sigma_1, sigma_2):
    """
    Find the composite sigma for a double gaussian where the two gaussian components
    share the same mean but may have difference spreads and weights. Here the definition
    for each gaussian is A*exp(-(x-mu)^2/2/sigma^2).
    :param scale_1: A1
    :param scale_2: A2
    :param sigma_1: sigma_1
    :param sigma_2: sigma_2
    :return: the composite sigma
    """

    scale_1 = abs(scale_1)
    scale_2 = abs(scale_2)
    sigma_1 = abs(sigma_1)
    sigma_2 = abs(sigma_2)

    norm = scale_1 * sigma_1 * (2 * np.pi) ** 0.5 + \
        scale_2 * sigma_2 * (2 * np.pi) ** 0.5
    A_reduced = scale_1 / norm
    B_reduced = scale_2 / norm
    variance = A_reduced * (2 * np.pi) ** 0.5 * sigma_1 ** 3 + \
        B_reduced * (2 * np.pi) ** 0.5 * sigma_2 ** 3
    return variance ** 0.5


def percentage_sigma(scale_1, scale_2, sigma_1, sigma_2, opt=3):
    """
    Find the effective"sigma" consistent with the "68-95-99" rule for a double gaussian
    where the two gaussian components share the same mean but may have difference spreads
    and weights. Here the definition for each gaussian is A*exp(-(x-mu)^2/2/sigma^2).
    :param scale_1: A1
    :param scale_2: A2
    :param sigma_1: sigma_1
    :param sigma_2: sigma_2
    :param opt: option 1, 2 or 3. Corresponds to matching the 68, 95, or 99.7
    :return: the effective "sigma"
    """

    num_div = 2000000
    perc = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    x = np.linspace(-0.1, 0.1, num_div + 1)
    y = double_gaus(x, abs(scale_1), abs(scale_2),
                    0, abs(sigma_1), abs(sigma_2))

    # area_all = abs(scale_1) * abs(sigma_1) * (2*np.pi)**0.5 + abs(scale_2) * abs(sigma_2) * (2*np.pi)**0.5
    area_all = abs(scale_1) + abs(scale_2)
    ind_left = num_div // 2
    ind_right = num_div // 2
    cur_perc = 0
    while cur_perc < perc[opt]:
        if ind_left == 0:
            break
        if perc[opt] - cur_perc > 0.05:
            ind_left -= num_div // 100
            ind_right += num_div // 100
        elif perc[opt] - cur_perc > 0.01:
            ind_left -= num_div // 500
            ind_right += num_div // 500
        elif perc[opt] - cur_perc > 0.001:
            ind_left -= num_div // 5000
            ind_right += num_div // 5000
        elif perc[opt] - cur_perc > 0.0001:
            ind_left -= 10
            ind_right += 10
        else:
            ind_left -= 1
            ind_right += 1
        area_signal = np.trapezoid(y[ind_left:ind_right], dx=0.2 / num_div)
        cur_perc = area_signal / area_all
    return (x[ind_right] - x[ind_left]) / opt / 2


# def add_profiles(profiles):
#     W = np.sum(np.array([p.counts(flow=False) for p in profiles]), axis=0)
#     h = np.sum(np.array([p.values() * p.counts(flow=False) for p in profiles]), axis=0) / W
#
#     w = np.array([p.counts(flow=False) for p in profiles])
#     ss = np.array([p.errors(error_mode='s') for p in profiles]) ** 2
#     hh = np.array([p.values(flow=False) for p in profiles]) ** 2
#     E = np.sum(w*(ss + hh), axis=0)
#     print(np.shape(E))
#
#     s_new = (E / W - h ** 2) ** 0.5
#     e = s_new / W ** 0.5
#     return h, e

# def add_profiles(profiles):
#     result = bh.Histogram(profiles[0], storage=bh.storage.WeightedMean())
#     for profile in profiles[1:]:
#         result += bh.Histogram(profile, storage=bh.storage.WeightedMean())
#     return result.values(), result.variances() ** 0.5


class PolyMinvFit:
    # lb = 1.115683 - 6 * 0.0023  # 1.105
    # rb = 1.115683 + 10 * 0.0023  # 1.74
    # lb_for_s = 1.115683 - 3 * 0.0023
    # rb_for_s = 1.115683 + 3 * 0.0023
    ratio = 1
    b_ratio = 1  # for plotting background
    min_count = 1
    hyperon_masses = {'Lambda': 1.115683, 'Xi': 1.32171, 'Omega': 1.67245,
                      'Lambdabar': 1.115683, 'Xibar': 1.32171, 'Omegabar': 1.67245}
    data_slices_hyperons = {'Lambda': [1.0, 1.2], 'Xi': [1.2, 1.4], 'Omega': [1.6, 1.8],
                            'Lambdabar': [1.0, 1.2], 'Xibar': [1.2, 1.4], 'Omegabar': [1.6, 1.8]} # only keep the data in this range

    signal = -1
    sigma = -1
    momega = -1
    signal_error = -1
    bkg = -1
    bkg_error = -1

    ptbin = -1
    cen = -1
    cen_map = {1: [70, 80], 2: [60, 70], 3: [50, 60], 4: [40, 50], 5: [30, 40],
               6: [20, 30], 7: [10, 20], 8: [5, 10], 9: [0, 5]}
    pt_map = [0.5, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.4, 4.0]

    def __init__(self, path, par_str, poly_deg, guesses, monitor=True, **kwargs):
        file = uproot.open(path)
        self.par_str = par_str
        self.hyperon_mass = self.hyperon_masses[par_str]

        if self.par_str.endswith('bar'):
            self.self_lambda = False
        else:
            self.self_lambda = True

        if self.par_str.startswith('Lambda'):
            self.lb = self.hyperon_mass - 6 * 0.0023
        else:
            self.lb = self.hyperon_mass - 8 * 0.0023
        self.rb = self.hyperon_mass + 10 * 0.0023
        self.lb_for_s = self.hyperon_mass - 3 * 0.0023
        self.rb_for_s = self.hyperon_mass + 3 * 0.0023
        self.monitor = monitor

        hist = file[f'h{self.par_str}M_cen_y_1_0']

        bin_width = hist.axis().edges()[1] - hist.axis().edges()[0]
        x_vals = hist.axis().edges()[:-1] + bin_width / 2
        # for iminuit
        y_vals = hist.values()
        x_errs = np.ones(np.shape(x_vals)) * bin_width / 2
        y_errs = hist.errors()
        v2_vals = np.zeros(len(y_vals))
        v2_errs = np.zeros(len(y_errs))

        self.east_west = 0 # 0 for east, 1 for west
        if 'ew' in kwargs.keys():
            self.east_west = kwargs['ew']
        east_west_str = 'east' if self.east_west == 0 else 'west'
        if 'flow_case' in kwargs.keys():
            self.flow_case = kwargs['flow_case']
        else:
            self.flow_case = 'v1'
        if 'ptbin' in kwargs.keys() and 'cen' in kwargs.keys():
            self.ptbin = kwargs['ptbin']
            self.cen = kwargs['cen']
            if self.ptbin == [1]:
                self.ratio = 1
            y_vals = np.sum(np.array([file[f'h{self.par_str}M_cen_pt_{east_west_str}_{i - 1}_{j}'].values()
                                        for i in self.cen for j in self.ptbin]), axis=0)
            y_errs = np.sum(np.array([file[f'h{self.par_str}M_cen_pt_{east_west_str}_{i - 1}_{j}'].errors()
                                        for i in self.cen for j in self.ptbin]) ** 2, axis=0) ** 0.5
            v2_hists = [
                file[f"h{self.par_str}_EPD_{self.flow_case}_pt_{east_west_str}_{i - 1}_{j}"] for i in self.cen for j in self.ptbin]
            v2_profiles = [SimpleProfile(p.values(flow=False), p.counts(flow=False), p.errors(error_mode='s'),
                                            p.axis().edges()) for p in v2_hists]
            self.v2_sum = sum(v2_profiles)

        data_slices = np.logical_and(x_vals > self.data_slices_hyperons[self.par_str][0],
                                     x_vals < self.data_slices_hyperons[self.par_str][1])
        slices = np.logical_and(x_vals > self.lb, x_vals < self.rb)
        slices_signal = np.logical_and(
            x_vals > self.lb_for_s, x_vals < self.rb_for_s)

        self.x_vals_data = x_vals[data_slices]
        self.y_vals_data = y_vals[data_slices]
        self.x_errs_data = x_errs[data_slices]
        self.y_errs_data = y_errs[data_slices]
        self.x_vals = x_vals[slices]
        self.xe = self.x_vals - bin_width / 2
        self.xe = np.append(self.xe, self.xe[-1] + bin_width)
        self.x_errs = x_errs[slices]
        self.y_vals = y_vals[slices] / self.ratio
        self.y_vals_signal = y_vals[slices_signal]
        self.y_errs = y_errs[slices] / self.ratio

        # for v_2 vs Minv
        self.v2_sum.Rebin(1)
        self.x_vals_v2 = self.v2_sum.bin_centers()
        slices_v2 = np.logical_and(
            self.x_vals_v2 > self.lb, self.x_vals_v2 < self.rb)
        self.x_vals_v2 = self.x_vals_v2[slices_v2][:-1]
        self.v2_vals = self.v2_sum.values()[slices_v2][:-1]
        self.v2_counts = self.v2_sum.counts()[slices_v2][:-1]
        self.v2_errs = self.v2_sum.errors()[slices_v2][:-1]
        # self.v2_vals = v2_vals[slices]
        # self.v2_errs = v2_errs[slices]

        self.params = guesses
        self.poly_deg = poly_deg
        self.cost = 0
        self.signal_func = breit_wigner
        self.bg_func = poly
        if 'signal' in kwargs.keys():
            if kwargs['signal'] == 'breit-wigner':
                self.signal_func = breit_wigner
            if kwargs['signal'] == 'gaussian':
                self.signal_func = gaus
            if kwargs['signal'] == "double gaussian":
                self.signal_func = double_gaus

        if 'bg' in kwargs.keys():
            if kwargs['bg'] == 'poly':
                self.bg_func = poly

    def func(self, params, x, y):
        signal = np.sum(self.y_vals_signal)
        bkg = np.sum(self.y_vals) - signal
        nbins = len(self.y_vals)
        ratio = self.params[0] / (self.params[0] + self.params[1])
        return self.bg_func(x, *params[-self.poly_deg - 1:]) \
            + self.signal_func(x, signal / nbins * ratio, signal / nbins * (1 - ratio),
                               *params[2:-self.poly_deg - 1]) - y

    def fit(self):
        res = least_squares(self.func, self.params, x_scale='jac', xtol=None, ftol=1e-10, gtol=None,
                            args=(self.x_vals, self.y_vals), loss='soft_l1')
        self.params = res.x
        self.cost = res.cost
        print(self.params)

        if not res.success:
            print(res.message)

    def fit_iminuit(self, masked=False, fixed_z=None):
        def integral_double(xe, s, z, b, mu, sigma_1, sigma_2, p0, p1, p2):
            return s * z * truncnorm.cdf(xe, self.lb, self.rb, mu, sigma_1) + \
                s * (1 - z) * truncnorm.cdf(xe, self.lb, self.rb, mu, sigma_2) + \
                b * poly_cdf(xe, self.lb, self.rb, p0, p1, p2)

        def integral(xe, s, b, mu, sigma, p0, p1, p2):
            return s * truncnorm.cdf(xe, self.lb, self.rb, mu, sigma) + \
                b * poly_cdf(xe, self.lb, self.rb, p0, p1, p2)

        func = integral_double if self.signal_func == double_gaus else integral
        c = cost.ExtendedBinnedNLL(self.y_vals, self.xe, func)
        signal = np.sum(self.y_vals_signal)
        bkg = (np.sum(self.y_vals) - signal) * 20. / 14.
        if self.par_str.startswith('Lambda'):
            signal = signal - bkg * 6. / 20.
        
        # print(f's = {(abs(self.params[0]) + abs(self.params[1])) * 1000}')
        # print(f's = {signal}')
        # print(f'b = {bkg}')

        # save initial parameters
        if self.signal_func == double_gaus:
            z_init = abs(self.params[0]) / (abs(self.params[0]) + abs(self.params[1]))
            if self.monitor and fixed_z is not None:
                print(f'z_init = {z_init}')
            m = Minuit(c, s=0, z=z_init, b=bkg,
                       mu=self.params[2], sigma_1=abs(self.params[3]), sigma_2=abs(self.params[4]),
                       p0=self.params[-self.poly_deg -
                                      1], p1=self.params[-self.poly_deg],
                       p2=self.params[-self.poly_deg + 1])
        
            if fixed_z is not None:
                m.fixed['z'] = True
            else:
                m.limits['z'] = (0, 1)  
            m.limits['s', 'b'] = (0, None)
        else:
            m = Minuit(c, s=0, b=bkg,
                       mu=self.params[1], sigma=abs(self.params[2]),
                       p0=self.params[-self.poly_deg -
                                      1], p1=self.params[-self.poly_deg],
                       p2=self.params[-self.poly_deg + 1])

            m.limits['s', 'b'] = (0, None)
        # m.limits['sigma_1'] = (0.001, 0.01)
        # m.limits['sigma_2'] = (0.001, 0.01)

        if masked:
            m.fixed['s'] = True
            c.mask = (self.x_vals < self.hyperon_mass - 3 *
                      0.003) | (self.hyperon_mass + 3 * 0.003 < self.x_vals)
            m.migrad()

            if self.monitor:
                fig_m, ax_m = plt.subplots()
                for ma, co in ((c.mask, "k"), (~c.mask, "w")):
                    ax_m.errorbar(self.x_vals[ma], self.y_vals[ma], self.y_vals[ma] ** 0.5, fmt="o", color=co, mec="k",
                                  ecolor="k")
                ax_m.stairs(np.diff(func(self.xe, *[p.value for p in m.init_params])), self.xe,
                            ls=":", label="init")
                ax_m.stairs(np.diff(func(self.xe, *m.values)),
                            self.xe, label="fit")
                ax_m.legend()
                plt.show()
                plt.close(fig_m)

            # release s fix b
            c.mask = None
            m.fixed['s'] = False
            m.fixed['b', 'p0', 'p1', 'p2'] = True
            m.values['s'] = signal
            m.migrad()
            m.fixed = None
            if fixed_z is not None:
                m.fixed['z'] = True
        m.simplex()
        if self.monitor:
            print(m.migrad())
        else:
            m.migrad()
        if not m.accurate or not m.valid:
            # print('Fit failed!!!!!!!!!!!!!!!!!!')
            return False
            
        m.hesse()
        # we need to deal with the case where the background is negative
        # which means that one of the gaussians has a RMS that is too large
        # This also means that the histogram should be fitted with just one gaussian
        # Force the fit to be with one gaussian by fixing z=1 when either sigma_1
        # is at least 5 times larger than sigma_2 or vice versa
        if fixed_z is None:
            if abs(m.values['sigma_1']) > 5 * abs(m.values['sigma_2']) or abs(m.values['sigma_2']) > 5 * abs(m.values['sigma_1']):
                self.params = [self.params[0]+self.params[1], 0, self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7]]
                if self.monitor:
                    print('Switching to single gaussian fit ...')
                return self.fit_iminuit(masked, fixed_z=1)

        # m.draw_mnprofile('s')
        if self.signal_func == double_gaus:
            self.params[0] = m.values['s'] * m.values['z'] / 2000
            # divide by bin counts per unity interval
            self.params[1] = m.values['s'] * (1 - m.values['z']) / 2000
            self.params[2] = m.values['mu']
            self.params[3] = m.values['sigma_1']
            self.params[4] = m.values['sigma_2']
            self.params[5] = m.values['p0']
            self.params[6] = m.values['p1']
            self.params[7] = m.values['p2']
            self.b_ratio = m.values['b']
            self.signal = m.values['s']
            self.signal_error = m.errors['s']
            self.bkg = m.values['b']
            self.bkg_error = m.errors['b']
        else:
            self.params[0] = m.values['s'] / 2000
            self.params[1] = m.values['mu']
            self.params[2] = m.values['sigma']
            self.params[3] = m.values['p0']
            self.params[4] = m.values['p1']
            self.params[5] = m.values['p2']
            self.b_ratio = m.values['b']
            self.signal = m.values['s']
            self.signal_error = m.errors['s']
            self.bkg = m.values['b']
            self.bkg_error = m.errors['b']

        self.sigma = abs(self.params[2]) if self.signal_func == gaus \
            else percentage_sigma(self.params[0], self.params[1], self.params[3], self.params[4], 3)
        self.momega = self.params[1] if self.signal_func == gaus else self.params[2]
        # sum = 0
        # for order in range(self.poly_deg + 1):
        #     sum += self.params[-order] * self.rb ** (order + 1) / (order + 1) - self.params[-order] * self.lb ** (order + 1) / (order + 1)
        # print(f'Integral: {sum}')
        # for order in range(self.poly_deg + 1):
        #     self.params[-order] = m.values[f'p{2-order}'] / sum
        if self.monitor:
            fig, ax = plt.subplots()
            m.visualize()
            fit_info = [
                f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(self.y_vals) - m.nfit} = {m.fval / (len(self.y_vals) - m.nfit)}",
            ]
            for p, v, e in zip(m.parameters, m.values, m.errors):
                # print(f"{p} = {v:.5f} +- {e:.5f}")
                fit_info.append(f"{p} = ${v:.5f} \\pm {e:.5f}$")

            ax.legend(title="\n".join(fit_info))
            plt.show()

        return True

    def get_iminuit_params(self):
        return self.params

    def fit_v2(self, masked=False, v2=0.05, a=0, b=0, save_to=None, cen=None, ybin=None):
        if self.monitor or save_to is not None:
            fig, ax = plt.subplots()
            ax.errorbar(self.x_vals_v2, self.v2_vals,
                        yerr=self.v2_errs, ls='none')
            ax.scatter(self.x_vals_v2, self.v2_vals, s=12)

        b_over_total = self.func_return()

        # s_over_total = 1 - b_over_total
        #
        # def func(x, v2, p0, p1):
        #     return b_over_total * (p0 + p1 * x) + s_over_total * v2
        #
        # def func_masked(x, v2, p0, p1, mask):
        #     return b_over_total[mask] * (p0 + p1 * x) + s_over_total[mask] * v2

        def func(x, v2, p0, p1):
            bx = b_over_total(x)
            return bx * (p0 + p1 * (x - self.hyperon_mass)) + (1 - bx) * v2
            # bin_size_adjustment_ratio = 0.0005 / (self.rb - self.lb)
            # b = bin_size_adjustment_ratio * self.b_ratio * poly_pdf(x, self.lb, self.rb, *self.params[-self.poly_deg - 1:])
            # t = self.signal_func(x, *self.params[:-self.poly_deg - 1])
            # bx = b / (b + t)
            # return bx * (p0 + p1 * x) + (1 - bx) * v2

        if self.monitor:
            fig_bx, ax_bx = plt.subplots()
            ax_bx.errorbar(self.x_vals_v2, b_over_total(self.x_vals_v2))
            ax_bx.set_xlabel(r'$M_{p\pi}$', fontsize=12, loc='right')
            ax_bx.set_ylabel(r'$\frac{b}{b + s}$', fontsize=12, loc='top')
            plt.close(fig_bx)

        c = cost.LeastSquares(self.x_vals_v2, self.v2_vals, self.v2_errs, func)
        c.loss = 'soft_l1'
        m = Minuit(c, v2=0, p0=b, p1=a)
        if masked:
            m.fixed['v2'] = True
            c.mask = ((self.x_vals_v2 < self.momega - 3 * self.sigma) | (self.momega + 3 * self.sigma < self.x_vals_v2)) \
                & (self.v2_counts > self.min_count)
            m.migrad()

            if self.monitor:
                fig_m, ax_m = plt.subplots()
                for ma, co in ((c.mask, "k"), (~c.mask, "w")):
                    ax_m.errorbar(self.x_vals_v2[ma], self.v2_vals[ma], self.v2_errs[ma], fmt="o", color=co, mec="k",
                                  ecolor="k")
                ax_m.plot(self.x_vals_v2, func(
                    self.x_vals_v2, *[p.value for p in m.init_params]), ls=':', label="init")
                ax_m.plot(self.x_vals_v2, func(
                    self.x_vals_v2, *m.values), label="fit")
                ax_m.legend()
                plt.close(fig_m)

            c.mask = None
            m.fixed['v2'] = False
            m.fixed['p0', 'p1'] = True
            c.mask = self.v2_counts > self.min_count
            m.values['v2'] = v2
            m.migrad()
            m.fixed = None
        else:
            c.mask = self.v2_counts > self.min_count

        m.simplex()
        if self.monitor:
            print(m.migrad())
            # m.draw_mnprofile("v2")
            # m.draw_mnprofile("p0")
            # m.draw_mnprofile("p1")
        else:
            m.migrad()
        # soft_l1 robustly locates the minimum; re-evaluate the uncertainty with a
        # calibrated plain-chi2 Hesse at that point. Closure test (closure_test_v1.py)
        # shows this removes soft_l1's ~30% error over-inflation (pull width 0.78 -> 0.95)
        # while preserving convergence and leaving the v1 value unbiased.
        c.loss = 'linear'
        m.migrad()
        m.hesse()
        if m.valid:
            m.minos()

        if self.monitor or save_to is not None:
            ax.plot(self.x_vals_v2, b_over_total(self.x_vals_v2) * (m.values['p0'] + m.values['p1'] * (self.x_vals_v2 - self.hyperon_mass)), label='Background')
            ax.plot(self.x_vals_v2, (1 - b_over_total(self.x_vals_v2)) * m.values['v2'], label='Signal')
            ax.plot(self.x_vals_v2, func(self.x_vals_v2, *m.values), label='Total')
            ax.plot(self.x_vals_v2, np.zeros_like(
                self.x_vals_v2), ls='--', c='k')
            ax.set_xlabel(r'$M_{p\pi}$', fontsize=12, loc='right')
            ax.set_ylabel(r'cos($\phi-\Psi_{EPD}$)', fontsize=12, loc='top')
            if self.cen != -1:
                cen_string = str(self.cen_map[self.cen[-1]][0]) + '-' + str(self.cen_map[self.cen[0]][1]) + '%'
                plt.annotate(cen_string, xy=(0.1, 0.9),
                             fontsize=15, xycoords='axes fraction')
                
            pt_lb, pt_rb = -999, -999
            if type(self.ptbin) is not int:
                pt_lb = self.ptbin[0] * 0.1 - 1
                pt_rb = (self.ptbin[-1]+1) * 0.1 - 1
            elif self.ptbin != -1:
                pt_lb = self.ptbin * 0.1 - 1
                pt_rb = (self.ptbin+1) * 0.1 - 1
            ax.annotate(fr'{pt_lb:.1f}<y<{pt_rb:.1f}', xy=(0.1, 0.8),
                            fontsize=15, xycoords='axes fraction')
            if m.valid and m.accurate:
                ax.annotate('Valid and Accurate', xy=(0.1, 0.7),
                            fontsize=15, xycoords='axes fraction')
            ax.legend()
            if self.monitor:
                plt.show()
                plt.close(fig)

        v2, v2err = None, None
        if not m.valid or not m.accurate:
            return None, None
            # print("Fit v_2 failed!!!!!!!!!!!!!!!!!!")
        else:
            v2 = m.values['v2']
            v2err = m.errors['v2']
        return v2, v2err

    def fit_odr(self):
        deg = self.poly_deg

        def func_wrapper(params, x):
            width = x[1] - x[0]
            xe = np.append(x - 0.5 * width, x[-1] + 0.5 * width)
            return self.b_ratio * np.diff(poly_cdf(xe, self.lb, self.rb, *params[-deg - 1:])) \
                + self.signal_func(x, *params[:-deg - 1])
            # return self.bg_func(x, *params[-deg - 1:]) \
            #        + self.signal_func(x, *params[:-deg - 1])

        model = odr.Model(func_wrapper)
        mydata = odr.RealData(self.x_vals, self.y_vals,
                              sy=self.y_errs, sx=self.x_errs)
        myodr = odr.ODR(mydata, model, beta0=self.params, maxit=100)
        myodr.run()
        myodr.output.pprint()

        self.params = myodr.output.beta
        # print(myodr.output.stopreason)

    def plot(self, save_to=None, cen=None, ybin=None, ybin_total=None):
        if self.signal_func == gaus:
            sigma = abs(self.params[2])
            mean = self.params[1]
        elif self.signal_func == double_gaus:
            # sigma = composite_sigma(self.params[0], self.params[1], self.params[3], self.params[4])
            # sigma = composite_sigma(self.params[0]/self.params[3]/(2*np.pi)**0.5, self.params[1]/self.params[4]/(2*np.pi)**0.5, self.params[3], self.params[4])
            sigma = percentage_sigma(
                self.params[0], self.params[1], self.params[3], self.params[4], 3)
            # sigma = percentage_sigma(self.params[0]*self.params[3]*(2*np.pi)**0.5, self.params[1]*self.params[4]*(2*np.pi)**0.5, self.params[3], self.params[4])
            # sigma = 0.00234 # fixed sigma
            mean = self.params[2]
        else:
            sigma = 0.0021
            mean = 1.67245

        x_smooth = np.linspace(self.lb, self.rb, 10000)
        bin_size_adjustment_ratio = 0.0005 / (self.rb - self.lb) * 10000
        # print(poly_cdf(x_smooth, self.lb, self.rb, *self.params[-self.poly_deg - 1:]))
        # y_exp = np.diff(poly_cdf(x_smooth, self.lb, self.rb, *self.params[-self.poly_deg - 1:])) \
        #         + self.signal_func(x_smooth[:-1], *self.params[:-self.poly_deg - 1])
        y_bg = bin_size_adjustment_ratio * self.b_ratio * np.diff(
            poly_cdf(x_smooth, self.lb, self.rb, *self.params[-self.poly_deg - 1:]))
        y_signal = self.signal_func(
            x_smooth[:-1], *self.params[:-self.poly_deg - 1])
        y_exp = y_bg + y_signal
        # print(f'cost = {self.cost}')
        # print(f'm = {mean}')
        # print(f'stdev = {sigma}')

        fig, ax = plt.subplots()
        ax.errorbar(self.x_vals, self.y_vals,
                     self.y_errs, 0, ls='none')
        ax.plot(self.x_vals, self.y_vals, 'o', c='C0', markersize=3)
        ax.plot(x_smooth[:-1], y_exp * self.ratio, c='C5', label='Total Fit')
        ax.plot(x_smooth[:-1], y_bg * self.ratio, c='C2',
                 label=f'{self.poly_deg}-Poly Background')
        # if self.self_lambda:
        #     plt.plot(x_smooth[:-1], y_signal * self.ratio, c='red', ls='--', label=r'$\Omega$ Signal')
        # else:
        #     plt.plot(x_smooth[:-1], y_signal * self.ratio, c='red', ls='--', label=r'$\bar{\Omega}$ Signal')

        # plt.vlines([mean - 3 * sigma, mean + 3 * sigma],
        #            [0., 0.], [1.5 * np.max(self.y_vals_data + self.y_errs_data),
        #                       1.5 * np.max(self.y_vals_data + self.y_errs_data)], ls='--', colors='black')
        # plt.vlines([mean - 6 * sigma, mean - 10 * sigma],
        #            [0., 0.], [1000, 1000], ls='--', colors='C3')
        # plt.vlines([mean + 6 * sigma, mean + 10 * sigma],
        #            [0., 0.], [1000, 1000], ls='--', colors='C3')
        ax.set_xlabel(r'$M_{p \pi}$ ' + r'(GeV/$c^{2}$)',
                   fontsize=12, loc='right')
        ax.set_ylabel('Counts', fontsize=12, loc='top')
        lb_idx = find_nearest(self.x_vals, mean - 3 * sigma)
        rb_idx = find_nearest(self.x_vals, mean + 3 * sigma)
        y_bg_sparse = self.b_ratio * \
            np.diff(poly_cdf(self.xe, self.lb, self.rb,
                    *self.params[-self.poly_deg - 1:]))
        y_fitted_vals = self.b_ratio * np.diff(poly_cdf(self.xe, self.lb, self.rb, *self.params[-self.poly_deg - 1:])) + \
            self.signal_func(self.x_vals, *self.params[:-self.poly_deg - 1])
        if self.self_lambda:
            ax.plot(self.x_vals, y_fitted_vals - y_bg_sparse,
                     c='red', ls='--', label=r'$\Lambda$ Signal')
        else:
            ax.plot(self.x_vals, y_fitted_vals - y_bg_sparse,
                     c='red', ls='--', label=r'$\bar{\Lambda}$ Signal')

        # plt.plot(self.x_vals, y_fitted_vals)

        # total = np.sum(self.y_vals[lb_idx:rb_idx + 1]) * self.ratio
        total = np.sum(y_fitted_vals[lb_idx:rb_idx + 1]) * self.ratio

        # print(f'A = {self.params[0] * self.ratio}, B = {self.params[1] * self.ratio}')
        # print(f'sigma_A = {abs(self.params[3])}, sigma_B = {abs(self.params[4])}')
        # print(f'A*sigma_A = {self.params[0] * self.ratio * abs(self.params[3])}')
        # print(f'B*sigma_B = {self.params[1] * self.ratio * abs(self.params[4])}')
        # print(f'Background fit = {self.params[-self.poly_deg - 1:]}')

        bg = np.sum(y_bg_sparse[lb_idx:rb_idx + 1]) * self.ratio
        signal_uncertainty = ufloat(self.signal, abs(self.signal_error))
        ax.annotate('S = ' + r'${{{:.2ueL}}}$'.format(signal_uncertainty), xy=(
            0.5, 0.7), xycoords='axes fraction')
        ax.annotate(rf'S/$\sqrt{{S+B}}$ = {self.signal / total ** 0.5:.2f}',
                     xy=(0.5, 0.6), xycoords='axes fraction')
        # plt.annotate(f'S/B = {(total - bg) / bg   :.2f}', xy=(0.7, 0.5), xycoords='axes fraction')
        # plt.annotate(f'S = {total - bg:.4f}', xy=(0.7, 0.7), xycoords='axes fraction')
        # plt.annotate(rf'S/$\sqrt{{S+B}}$ = {(total - bg) / total ** 0.5:.4f}', xy=(0.7, 0.6), xycoords='axes fraction')
        # plt.annotate(f'S/B = {(total - bg) / bg   :.4f}', xy=(0.7, 0.5), xycoords='axes fraction')
        ax.annotate(f'$\sigma$ = {sigma :.5f}', xy=(
            0.5, 0.5), xycoords='axes fraction')
        if self.cen != -1:
            cen_string = str(
                self.cen_map[self.cen[-1]][0]) + '-' + str(self.cen_map[self.cen[0]][1]) + '%'
            ax.annotate(cen_string, xy=(0.05, 0.9),
                         fontsize=15, xycoords='axes fraction')
            
        if self.ptbin != -1:
            ax.annotate(fr'$p_{{T}}: {self.pt_map[self.ptbin[0] - 1]}-{self.pt_map[self.ptbin[-1]]}$ GeV/c',
                         xy=(0.05, 0.8), fontsize=15, xycoords='axes fraction')

        ax.legend()
        pt_lb, pt_rb = -999, -999
        if type(self.ptbin) is not int:
            pt_lb = self.ptbin[0] * 0.1 - 1
            pt_rb = (self.ptbin[-1]+1) * 0.1 - 1
        elif self.ptbin != -1:
            pt_lb = self.ptbin * 0.1 - 1
            pt_rb = (self.ptbin+1) * 0.1 - 1
        if self.monitor:
            plt.show()


    def func_return(self):
        bg = self.b_ratio * \
            np.diff(poly_cdf(self.xe, self.lb, self.rb,
                    *self.params[-self.poly_deg - 1:]))
        sn = self.signal_func(self.x_vals, *self.params[:-self.poly_deg - 1])

        def func(x):
            return hist_to_func(x, bg / (sn + bg), self.xe)

        return func


def main(path, outputDir, particle_str, flow_case, yrebin=1, ew=0, max_refit=500, paper_plot_path=None):
    start = time.time()
    monitor = 0
    paper_plot_cen = 4
    paper_plot_y = 2
    energy_str = path.split('/')[-1].split('_')[-1].split('.')[0]
    cen = np.arange(1, 10).reshape(9, 1)  # regular
    # cen = [[1,2,3]] #, 8, 9] # comparison with Ashik
    initial_masses = {'Lambda': 1.12, 'Xi': 1.32, 'Omega': 1.67,
                      'Lambdabar': 1.12, 'Xibar': 1.32, 'Omegabar': 1.67}
    initial_mass = initial_masses[particle_str]
    poly_deg = 2

    ptbin_lo = 4
    ptbin_hi = 17
    
    num_ybin = ptbin_hi - ptbin_lo + 1
    num_yrebin = yrebin
    v2_cen_y = {c[0]: np.array([None for _ in range(
        ptbin_lo, ptbin_hi + 1, num_yrebin)]) for c in cen}
    count_cen_y = {c[0]: np.array([None for _ in range(
        ptbin_lo, ptbin_hi + 1, num_yrebin)]) for c in cen}
    v2_cen_y_err = {c[0]: np.array([None for _ in range(
        ptbin_lo, ptbin_hi + 1, num_yrebin)]) for c in cen}
    param_drawers = {c[0]: [ParamStorage() for _ in range(
        ptbin_lo, ptbin_hi + 1, num_yrebin)] for c in cen}

    fail_reasons = {} # 1: inv mass, 2: v2
    with std_out_err_redirect_tqdm() as orig_stdout:
        pbar = tqdm(total=len(cen) * num_ybin // num_yrebin,
                    file=orig_stdout, dynamic_ncols=True)
        pbar.set_description('Initial fitting ...')
        for c in cen:
            for i, ptbin in enumerate(np.arange(ptbin_lo, ptbin_hi + 1).reshape(num_ybin // num_yrebin, num_yrebin)):
                guesses = [  # .05, 1.67272, 0.002, # gaus
                    5, 5, initial_mass, 0.003, 0.002,  # double gaus
                    # 0.1, 1.59, 0, 1.0, 0.1] # breit-wigner
                    -0, 1, 0]
                param_drawers[c[0]][i].set_params(guesses)
                # 1.61, 0.01, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01] # tenth poly
                fitter = PolyMinvFit(path=path, par_str=particle_str, poly_deg=poly_deg,
                                     guesses=param_drawers[c[0]
                                                           ][i].get_params(),
                                     monitor=monitor, signal='double gaussian', bg='poly',
                                     cen=c, ptbin=ptbin, ew=ew, flow_case=flow_case)
                invm_fitted = fitter.fit_iminuit(masked=True)
                save_path = None

                if invm_fitted:
                    # print('inv mass fitted!')
                    # param_drawers[c[0]][i].set_params(fitter.get_iminuit_params())
                    # param_drawers[c[0]][i].freeze()
                    # print(f'frozen params: {param_drawers[c[0]][i].get_params()}')
                    count_cen_y[c[0]][i] = fitter.signal
                    v2_cen_y[c[0]][i], v2_cen_y_err[c[0]][i] = fitter.fit_v2(masked=True, v2=np.random.normal(0.0, 0.10),
                                                                             a=np.random.normal(
                                                                                 0, 0.1),
                                                                             b=np.random.normal(0, 0.1),
                                                                             save_to=save_path,
                                                                             cen=c)
                    # if v2_cen_y[c[0]][i] is not None:
                    #     print('v2 fitted!')
                else:
                    if c[0] not in fail_reasons:
                        fail_reasons[c[0]] = {}
                    fail_reasons[c[0]][i] = 1
                    v2_cen_y[c[0]][i], v2_cen_y_err[c[0]][i] = None, None

                pbar.update(1)
        pbar.close()

        # print missing ptbin
        missing_bin = {}
        for c in cen:
            missing_bin[c[0]] = []
            if c[0] not in fail_reasons:
                fail_reasons[c[0]] = {}
            for i, ptbin in enumerate(np.arange(ptbin_lo, ptbin_hi + 1).reshape(num_ybin // num_yrebin, num_yrebin)):
                if v2_cen_y[c[0]][i] is None:
                    missing_bin[c[0]].append(ptbin)
                    fail_reasons[c[0]][i] = 2
                else:
                    missing_bin[c[0]].append(None)
        print(missing_bin)
        print(fail_reasons)
        # pprint.pprint(v2_y_pt)
        # pprint.pprint(v2_y_pt_err)
        has_badfit = not check_none(missing_bin)

        stage_1_time = time.time() - start
        print('-----------------')
        print(f'Stage 1 time: {stage_1_time:.2f} s')
        print('-----------------')

        # Start refitting
        num_trial = 1
        pbar2 = tqdm(total=sum([len([y for y in ybins if y is not None]) for ybins in missing_bin.values()]),
                     file=orig_stdout, dynamic_ncols=True)
        pbar2.set_description('Refitting ...')
        refit_counter = {c[0]: [0 for _ in range(
            (ptbin_hi+1-ptbin_lo) // num_yrebin)] for c in cen}
        strategies = {c[0]: [None for _ in range(
            (ptbin_hi+1-ptbin_lo) // num_yrebin)] for c in cen} # 1 if fixed z, None if not
        while has_badfit:
            for c in cen:
                ptbins = missing_bin[c[0]]
                for i, ptbin in enumerate(ptbins):
                    if ptbin is None:
                        continue
                    if refit_counter[c[0]][i] > max_refit // 2:
                        strategies[c[0]][i] = 1
                    if refit_counter[c[0]][i] > max_refit:
                        missing_bin[c[0]][i] = None
                        print(
                            f'cen = {c}, ptbin = {ptbin} is not fitted after {max_refit} trials, skipping ...')
                        pbar2.update(1)
                    guesses = [  # .05, 1.67272, 0.002, # gaus
                        np.random.uniform(
                            5-3, 5+3), np.random.uniform(5-3, 5+3),
                        np.random.uniform(initial_mass-0.01, initial_mass+0.01),
                        np.random.uniform(0.002, 0.005),
                        np.random.uniform(0.002, 0.005),  # double gaus
                        # 0.1, 1.59, 0, 1.0, 0.1] # breit-wigner
                        np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
                    param_drawers[c[0]][i].set_params(guesses)
                    fitter = PolyMinvFit(path=path, par_str=particle_str, poly_deg=poly_deg,
                                         guesses=param_drawers[c[0]][i].get_params(
                                         ),
                                         monitor=monitor, signal='double gaussian', bg='poly',
                                         cen=c, ptbin=ptbin, ew=ew, flow_case=flow_case)

                    # fitter.plot()
                    invm_fitted = fitter.fit_iminuit(masked=True, fixed_z=strategies[c[0]][i])
                    # if monitor and invm_fitted:
                    #     fitter.plot()
                    save_path = None

                    if invm_fitted:
                        # print('inv mass fitted!')
                        # param_drawers[c[0]][i].set_params(fitter.get_iminuit_params())
                        param_drawers[c[0]][i].freeze()
                        count_cen_y[c[0]][i] = fitter.signal
                        # print(f'frozen params: {param_drawers[c[0]][i].get_params()}')
                        v2_cen_y[c[0]][i], v2_cen_y_err[c[0]][i] = fitter.fit_v2(masked=True,
                                                                                 v2=np.random.uniform(
                                                                                     -0.05, 0.05),
                                                                                 a=np.random.uniform(
                                                                                     -0.1, 0.1),
                                                                                 b=np.random.uniform(
                                                                                     -0.1, 0.1),
                                                                                 save_to=save_path,
                                                                                 cen=c)
                        # if v2_cen_y[c[0]][i] is not None:
                        # print('v2 fitted!')
                    else:
                        v2_cen_y[c[0]][i], v2_cen_y_err[c[0]][i] = None, None
                    if v2_cen_y[c[0]][i] is not None:
                        missing_bin[c[0]][i] = None
                        pbar2.update(1)
                    refit_counter[c[0]][i] += 1
            num_trial += 1
            has_badfit = not check_none(missing_bin)
        pbar2.close()

    # store result to pandas csv
    dict_combined = {key: {
        'values': v2_cen_y[key], 'counts': count_cen_y[key], 'errors': v2_cen_y_err[key]} for key in v2_cen_y.keys()}
    df = pd.DataFrame(dict_combined)
    df = df.stack().apply(pd.Series)
    df.index = df.index.set_names(['First_Level', 'Second_Level'])
    df = df.sort_index(level=[1, 0], ascending=[True, False])
    df = df.swaplevel().T
    print(df)
    print(f'Writing to {outputDir}')
    df.to_csv(outputDir)
    # write_dict(v2_cen_y, v2_cen_y_err, result+f'{particle_str}_fit_{flow_case}.txt')
    print('Num of trials: ', num_trial)

    print('-----------------')
    print('Total time: ', time.time() - start)
    print('-----------------')

    # plt.show()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def check_none(values):
    all_none = True
    for key, val in values.items():
        if any(v is not None for v in val):
            all_none = False
            break
    return all_none


def print_dict(dv, de):
    print('{')
    for key, val in dv.items():
        print(f'\t{key}: ')
        print('\t\t{')
        print(f'\t\t\t"values": ', end='')
        print_numpy(val)
        print(',')
        print(f'\t\t\t"errors": ', end='')
        print_numpy(de[key])
        print('\n\t\t},')
    print('}')

# write a similar function as above but write to a txt file


def write_dict(dv, de, path):
    with open(path, 'w') as f:
        f.write('{\n')
        for key, val in dv.items():
            f.write(f'\t{key}: \n')
            f.write('\t\t{\n')
            f.write(f'\t\t\t"values": ')
            f.write(write_numpy(val))
            f.write(',\n')
            f.write(f'\t\t\t"errors": ')
            f.write(write_numpy(de[key]))
            f.write('\n\t\t},\n')
        f.write('}')


def print_numpy(arr):
    out = repr(arr).replace('None', 'np.nan')
    out = out.replace(', dtype=object', '')
    print('np.', out, sep='', end='')


def write_numpy(arr):
    out = repr(arr).replace('None', 'np.nan')
    out = out.replace(', dtype=object', '')
    return 'np.' + out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputDir', type=str, help='input directory')
    parser.add_argument('outputDir', type=str, help='output directory')
    parser.add_argument('--yrebin', type=int, default=1,
                       help='combine every yrebin rapidity bins')
    parser.add_argument('--ew', type=str, default=0, help='east or west, 0 for east, 1 for west')
    parser.add_argument('--max_refit', type=int, default=500,
                       help='maximum number of refit for failed fits')
    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir

    filename = os.path.basename(inputDir).replace('.root', '')
    # Split by underscore: ['combined', '{particle}', '{flow}', '{energy}', ...]
    parts = filename.split('_')
    particle_str = parts[1]
    flow_case = parts[2]
    
    result = outputDir.split('/')[0] + '/'
    if args.ew == 'west':
        ew = 1
    else:
        ew = 0
    
    main(inputDir, outputDir, particle_str, flow_case,
         yrebin=args.yrebin, ew=ew, max_refit=args.max_refit)
