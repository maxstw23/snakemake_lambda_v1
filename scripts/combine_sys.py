import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt

# Special tags 5 (positive-y half) and 8 (negative-y half) are the two one-sided
# v1(y) slope fits. They are combined into a single paired "y-range" systematic
# rather than each being compared to the default cut.
YRANGE_POS_TAG = '5'
YRANGE_NEG_TAG = '8'
YRANGE_DIVISOR = 12.0  # full-width uniform: std = width / sqrt(12)


def yrange_paired_unc(pos_val, pos_err, neg_val, neg_err):
    """Variance contribution of the paired positive/negative-y half-fit systematic,
    BEFORE the full-width-uniform /12 divisor (the caller applies it).

    Delta = slope_pos - slope_neg; the two halves use disjoint y-ranges, so the
    statistical scatter on Delta is sigma = sqrt(pos_err^2 + neg_err^2). The term is
    significance-gated: it contributes max(Delta^2 - sigma^2, 0) only when |Delta| > sigma.
    """
    delta = pos_val - neg_val
    sigma2 = pos_err**2 + neg_err**2
    if delta**2 > sigma2:
        return delta**2 - sigma2
    return 0.0


def main(default, regular_sys, special_sys, output, energy, sys_divisor):
    with open (default, 'r') as f:
        default_cut = yaml.load(f, Loader=yaml.CLoader)
    sys_cut = {}
    if not regular_sys[0].startswith('result/blank/'):
        for sys in regular_sys:
            with open (sys, 'r') as f:
                sys_tag = sys.split('/')[1].split('_')[-1]
                sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)
    for sys in special_sys:
        with open (sys, 'r') as f:
            sys_tag = sys.split('/')[1].split('_')[-1]
            sys_cut[sys_tag] = yaml.load(f, Loader=yaml.CLoader)
    
    # iterate thru the centralities
    data_pairs = {'y': 'yerr', 'lambda': 'lambda_err', 'lambdabar': 'lambdabar_err'}
    new_yerr = {}
    yerr_stat = {}
    yerr_sys = {}
    for val, err in data_pairs.items():
        new_yerr[val] = np.zeros_like(default_cut['yerr'])
        yerr_stat[val] = np.zeros_like(default_cut['yerr'])
        yerr_sys[val] = np.zeros_like(default_cut['yerr'])
        for cent in range(len(default_cut['x'])):
            print(f'Centrality: {default_cut["x"][cent]}')
            print('\tSys_tag    Delta      Delta_err  Significance')
            sum_of_unc = 0

            for sys_tag in sys_cut.keys():
                if sys_tag in (YRANGE_POS_TAG, YRANGE_NEG_TAG):
                    continue  # combined below as the paired y-range systematic
                delta = default_cut[val][cent] - sys_cut[sys_tag][val][cent]
                delta_err = np.sqrt(np.abs(sys_cut[sys_tag][err][cent]**2 - default_cut[err][cent]**2))
                # significance = abs(delta) / delta_err if delta_err != 0 else 0
                significance = (delta_err < abs(delta))
                print(f'\t{int(sys_tag):<10} {delta:<10.4f} {delta_err:<10.4f} {significance}')
                if significance:
                    sum_of_unc += delta**2 - delta_err**2
            # paired positive/negative-y half-fit systematic (full-width uniform -> /12)
            yrange_unc = 0
            if YRANGE_POS_TAG in sys_cut and YRANGE_NEG_TAG in sys_cut:
                yrange_unc = yrange_paired_unc(
                    sys_cut[YRANGE_POS_TAG][val][cent], sys_cut[YRANGE_POS_TAG][err][cent],
                    sys_cut[YRANGE_NEG_TAG][val][cent], sys_cut[YRANGE_NEG_TAG][err][cent])
                d_pn = sys_cut[YRANGE_POS_TAG][val][cent] - sys_cut[YRANGE_NEG_TAG][val][cent]
                print(f'\t{"5-8 yrange":<10} {d_pn:<10.4f} {"":<10} {yrange_unc > 0}')
            total_var = sum_of_unc / sys_divisor + yrange_unc / YRANGE_DIVISOR
            print(f'\tTotal systematic uncertainty: {np.sqrt(total_var):.4f}')
            yerr_stat[val][cent] = default_cut['yerr'][cent]
            yerr_sys[val][cent] = np.sqrt(total_var)
            new_yerr[val][cent] = np.sqrt(default_cut[err][cent]**2 + total_var)
        
        # basically use default_cut, but replace yerr with new_yerr
        default_cut[err] = new_yerr[val]
        default_cut[err+'_stat'] = yerr_stat[val]
        default_cut[err+'_sys'] = yerr_sys[val]

    
    print('----------------------------------')
    merged_centralities = ['010', '1040', '4080', '5080']
    new_yerr = {}
    yerr_stat = {}
    yerr_sys = {}
    for part in ['lambda', 'lambdabar', 'deltalambda']:
        for cent in merged_centralities:
            pair_name = f'dv1dy_{part}_{cent}'
            new_yerr = 0
            yerr_stat = 0
            yerr_sys = 0
            print(f'Centrality: {cent}')
            print('\tSys_tag    Delta      Delta_err  Significance')
            sum_of_unc = 0

            for sys_tag in sys_cut.keys():
                if sys_tag in (YRANGE_POS_TAG, YRANGE_NEG_TAG):
                    continue  # combined below as the paired y-range systematic
                delta = default_cut[pair_name]['value'] - sys_cut[sys_tag][pair_name]['value']
                delta_err = np.sqrt(np.abs(sys_cut[sys_tag][pair_name]['error']**2 - default_cut[pair_name]['error']**2))
                # significance = abs(delta) / delta_err if delta_err != 0 else 0
                significance = (delta_err < abs(delta))
                print(f'\t{int(sys_tag):<10} {delta:<10.4f} {delta_err:<10.4f} {significance}')
                if significance:
                    sum_of_unc += delta**2 - delta_err**2
            # paired positive/negative-y half-fit systematic (full-width uniform -> /12)
            yrange_unc = 0
            if YRANGE_POS_TAG in sys_cut and YRANGE_NEG_TAG in sys_cut:
                yrange_unc = yrange_paired_unc(
                    sys_cut[YRANGE_POS_TAG][pair_name]['value'], sys_cut[YRANGE_POS_TAG][pair_name]['error'],
                    sys_cut[YRANGE_NEG_TAG][pair_name]['value'], sys_cut[YRANGE_NEG_TAG][pair_name]['error'])
                d_pn = sys_cut[YRANGE_POS_TAG][pair_name]['value'] - sys_cut[YRANGE_NEG_TAG][pair_name]['value']
                print(f'\t{"5-8 yrange":<10} {d_pn:<10.4f} {"":<10} {yrange_unc > 0}')
            total_var = sum_of_unc / sys_divisor + yrange_unc / YRANGE_DIVISOR
            print(f'\tTotal systematic uncertainty: {np.sqrt(total_var):.4f}')
            yerr_stat = default_cut[pair_name]['error']
            yerr_sys = np.sqrt(total_var)
            new_yerr = np.sqrt(default_cut[pair_name]['error']**2 + total_var)

            default_cut[pair_name]['error'] = new_yerr
            default_cut[pair_name]['error_stat'] = yerr_stat
            default_cut[pair_name]['error_sys'] = yerr_sys
    
    print('----------------------------------')
    print('===============v1_pt==============')
    print('----------------------------------')
    merged_centralities = ['010', '1040', '4080', '5080']
    for part in ['lambda', 'lambdabar', 'delta']:
        print(f'Part: {part}')
        for cent in merged_centralities:
            print(f'Centrality: {cent}')
            pair_name = 'v1_pt_' + part + '_' + cent
            new_yerr = 0
            print('\tSys_tag    Delta[5]   Default_err[5]  Varied_err[5]  Delta_err[5]  Significance[5]')
            for sys_tag in sys_cut.keys():
                delta = default_cut[pair_name]['value'] - sys_cut[sys_tag][pair_name]['value']
                delta_err = np.sqrt(np.abs(sys_cut[sys_tag][pair_name]['error']**2 - default_cut[pair_name]['error']**2))
                significance = (delta_err < abs(delta))
                print(f'\t{int(sys_tag):<10} {float(delta[5]):<10.6f} {default_cut[pair_name]["error"][5]:<15.6f} {sys_cut[sys_tag][pair_name]["error"][5]:<14.6f} {delta_err[5]:<10.6f} {significance[5]}')
                # if significance:
                #     new_yerr += delta**2 - delta_err**2
                # significance is an boolean array
                new_yerr += np.where(significance, delta**2 - delta_err**2, 0)
            yerr_stat = default_cut[pair_name]['error']
            yerr_sys = np.sqrt(new_yerr / sys_divisor)
            default_cut[pair_name]['error'] = np.sqrt(yerr_stat**2 + yerr_sys**2)
            default_cut[pair_name]['error_stat'] = yerr_stat
            default_cut[pair_name]['error_sys'] = yerr_sys

            pair_name = 'v1_y_' + part + '_' + cent
            new_yerr = 0
            for sys_tag in sys_cut.keys():
                delta = default_cut[pair_name]['value'] - sys_cut[sys_tag][pair_name]['value']
                delta_err = np.sqrt(np.abs(sys_cut[sys_tag][pair_name]['error']**2 - default_cut[pair_name]['error']**2))
                significance = (delta_err < abs(delta))
                # if significance:
                #     new_yerr += delta**2 - delta_err**2
                # significance is an boolean array
                new_yerr += np.where(significance, delta**2 - delta_err**2, 0)
            yerr_stat = default_cut[pair_name]['error']
            yerr_sys = np.sqrt(new_yerr / sys_divisor)
            default_cut[pair_name]['error'] = np.sqrt(yerr_stat**2 + yerr_sys**2)
            default_cut[pair_name]['error_stat'] = yerr_stat
            default_cut[pair_name]['error_sys'] = yerr_sys

    with open(output, 'w') as f:        
        yaml.dump(default_cut, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--default', type=str, help='Default cut', required=True)
    parser.add_argument('--regular_sys', type=str, help='Regular systematic cut (subsets of default)', nargs='+', required=True)
    parser.add_argument('--special_sys', type=str, help='Special systematic cut (use the same dataset as default)', nargs='+', required=True)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    parser.add_argument('--energy', type=str, help='Energy', required=True)
    parser.add_argument('--sys_divisor', type=float, help='Divisor for systematic uncertainty (3 for half-width uniform, 12 for full-width uniform)', required=True)
    args = parser.parse_args()
    main(args.default, args.regular_sys, args.special_sys, args.output, args.energy, args.sys_divisor)