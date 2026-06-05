configfile: 'config.yaml'

### find the latest result file
import glob
import os
import re
import numpy as np

# CERN ROOT via Docker (not installed locally)
DOCKER_ROOT = 'docker run --rm -v "$(pwd)":/work -w /work rootproject/root:latest root'

# to be changed
energies = config['energies']

# generate a dict of the latest result for each energy
data_files = {'0': {energy: sorted(glob.glob(f'data/result*_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in energies}}
# systematic files, right now only one energy and one systematic!!!
data_files.update({str(sys_tag): {energy: sorted(glob.glob(f'data/sys_tag_{sys_tag}/result*_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in energies} for sys_tag in [1,2,3]})

eff_files = {energy: sorted(glob.glob(f'data/eff/result*_lambda_exp_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']}
eff_files_lambdabar = {energy: sorted(glob.glob(f'data/eff/result*_lambdabar_exp_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV'] if glob.glob(f'data/eff/result*_lambdabar_exp_{energy}.root')}
eff_files_all = {'lambda': eff_files, 'lambdabar': eff_files_lambdabar}

wildcard_constraints:
    particle='[A-Za-z]+',
    flow='[a-z0-9]+',
    energy='[0-9p]+GeV',
    sys_tag='[0-9]+',

_eff_plot_targets = (
    [f'plots/eff_qa/eff_vs_pt_lambda_{e}.png'     for e in eff_files] +
    [f'plots/eff_qa/eff_vs_pt_lambdabar_{e}.png'  for e in eff_files_lambdabar] +
    [f'plots/eff_qa/eff_yslices_lambda_{e}.png'    for e in eff_files] +
    [f'plots/eff_qa/eff_yslices_lambdabar_{e}.png' for e in eff_files_lambdabar]
)
# Alternative-cut (different proton/kaon pT ranges) versions of fig 2 & 3
_alt_fig_targets = [
    'plots/paper/fig_2_altcuts.pdf',
    'plots/paper/fig_3_horizontal_5080.pdf',           # default cuts, 50-80%
    'plots/paper/fig_3_horizontal_altcuts.pdf',        # alt cuts, 40-80%
    'plots/paper/fig_3_horizontal_5080_altcuts.pdf',   # alt cuts, 50-80%
]

rule all:
    input:
        'plots/paper/report.pdf',
        _eff_plot_targets,
        _alt_fig_targets

# rule generate_report:
#     input: expand('plots/dv1a1dy_{energy}.pdf', energy=energies),
#            expand('plots/dv1dy_coal_{energy}.pdf', energy=energies),
#            'plots/dv1dy_1040.pdf',
#            'plots/dv1dy_5080.pdf'
#     output: 'plots/lambda_v1_report.pdf'
#     log: stdout='logs/generate_report.log', stderr='logs/generate_report.err'
#     shell:
#         'python scripts/merge_pdf.py {input} {output} > {log.stdout} 2> {log.stderr}'

rule combine_lambda:
    input: data_file=lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy],
           script='scripts/combine_lambda_without_eff.cpp'
    output: 'result/sys_tag_{sys_tag}/combined_{particle}_{flow}_{energy}.root'
    params:
        pt_lo = config['pt_lo'],
        pt_hi = config['pt_hi'],
        y_cut = config['y_cut']
    log: stdout='logs/sys_tag_{sys_tag}/combine_{particle}_{flow}_{energy}.log', stderr='logs/sys_tag_{sys_tag}/combine_{particle}_{flow}_{energy}.err'
    shell: 
        """
        {DOCKER_ROOT} -b -q -l '{input.script}("{input.data_file}", "./result/sys_tag_{wildcards.sys_tag}/", {params.pt_lo}, {params.pt_hi}, "{wildcards.particle}", "{wildcards.flow}", "{wildcards.energy}", {params.y_cut})' > {log.stdout} 2> {log.stderr}
        """

rule combine_lambda_with_eff:
    input: 
        data_file=lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy],
        eff_file=lambda wildcards: f'result/eff/efficiency_{wildcards.particle.lower()}_{wildcards.energy}.root',
        script='scripts/combine_lambda_with_eff.cpp'
    output: 'result/sys_tag_{sys_tag}/combined_{particle}_{flow}_{energy}_eff_corrected.root'
    params:
        pt_lo = config['pt_lo'],
        pt_hi = config['pt_hi'],
        y_cut = config['y_cut']
    log: 
        stdout='logs/sys_tag_{sys_tag}/combine_eff_{particle}_{flow}_{energy}.log', 
        stderr='logs/sys_tag_{sys_tag}/combine_eff_{particle}_{flow}_{energy}.err'
    shell: 
        """
        {DOCKER_ROOT} -b -q -l '{input.script}("{input.data_file}", "./result/sys_tag_{wildcards.sys_tag}/", "{input.eff_file}", {params.pt_lo}, {params.pt_hi}, "{wildcards.particle}", "{wildcards.flow}", "{wildcards.energy}", {params.y_cut}, 1)' > {log.stdout} 2> {log.stderr}
        """

rule calculate_efficiency:
    input:
        script = 'scripts/calculate_lambda_eff.cpp',
        data = lambda wildcards: eff_files_all[wildcards.particle][wildcards.energy]
    output: 'result/eff/efficiency_{particle}_{energy}.root'
    params:
        y_cut = config["y_cut"]
    shell:
        """
        {DOCKER_ROOT} -l -b -q '{input.script}("{input.data}", "{output}", {params.y_cut})'
        """

rule plot_efficiency:
    input:
        eff_file='result/eff/efficiency_{particle}_{energy}.root',
        script='scripts/plot_eff_vs_pt.py'
    output:
        integrated='plots/eff_qa/eff_vs_pt_{particle}_{energy}.png',
        yslices='plots/eff_qa/eff_yslices_{particle}_{energy}.png'
    shell:
        """
        python {input.script} --eff_file {input.eff_file} --particle {wildcards.particle} --energy {wildcards.energy} --output {output.integrated} --output_yslices {output.yslices}
        """

def get_combined_file(wildcards):
    # Energies for which efficiency corrections are available, per particle
    particle_eff_energies = {
        'lambda':    list(eff_files.keys()),
        'lambdabar': list(eff_files_lambdabar.keys()),
    }
    available = particle_eff_energies.get(wildcards.particle.lower(), [])
    if wildcards.energy in available:
        return f"result/sys_tag_{wildcards.sys_tag}/combined_{wildcards.particle}_{wildcards.flow}_{wildcards.energy}_eff_corrected.root"
    else:
        return f"result/sys_tag_{wildcards.sys_tag}/combined_{wildcards.particle}_{wildcards.flow}_{wildcards.energy}.root"

rule fit_particle:
    input: 
        data_file=get_combined_file,
        script='scripts/fit_v1.py',
    output: 
        data_points='result/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.csv',
        invmass_plot='plots/sys_tag_{sys_tag}/paper_yaml/invmass/{particle}_fit_{flow}_{energy}_invmass_cen4_y0.7.yaml',
        v1_fit_plot='plots/sys_tag_{sys_tag}/paper_yaml/v1fit/{particle}_fit_{flow}_{energy}_v1fit_cen4_y0.7.yaml'
    params: 
        energy=lambda wildcards: wildcards.energy,
        yrebin=lambda wildcards: config['yrebin'][wildcards.energy][wildcards.particle]
    log: 
        stdout='logs/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.log', 
        stderr='logs/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.err'
    shell: 
        """
        python {input.script} {input.data_file} {output.data_points} \
            --yrebin {params.yrebin} \
            --max_refit 500 \
            --paper_plot_path {output.invmass_plot} \
            > {log.stdout} 2> {log.stderr}
        """

rule fit_no_eff:
    """Fit v1(y) on the non-efficiency-corrected combined ROOT file.
    Used to compare against the eff-corrected fit_particle outputs."""
    input:
        data_file='result/sys_tag_0/combined_{particle}_{flow}_{energy}.root',
        script='scripts/fit_v1.py'
    output:
        data_points='result/no_eff/fit_{particle}_{flow}_{energy}.csv'
    params:
        yrebin=lambda wildcards: config['yrebin'][wildcards.energy][wildcards.particle]
    log:
        stdout='logs/no_eff/fit_{particle}_{flow}_{energy}.log',
        stderr='logs/no_eff/fit_{particle}_{flow}_{energy}.err'
    shell:
        """
        python {input.script} {input.data_file} {output.data_points} \
            --yrebin {params.yrebin} \
            --max_refit 500 \
            > {log.stdout} 2> {log.stderr}
        """

rule plot_eff_comparison:
    """Compare eff-corrected vs no-eff v1 results for a given energy.
    Produces a 3-page PDF: v1(y) grids for Lambda/Lambdabar + dv1/dy vs centrality."""
    input:
        paths_eff=lambda wildcards: expand(
            'result/sys_tag_0/fit_{particle}_v1_{energy}.csv',
            particle=config['particles'], energy=wildcards.energy),
        paths_no_eff=lambda wildcards: expand(
            'result/no_eff/fit_{particle}_v1_{energy}.csv',
            particle=config['particles'], energy=wildcards.energy),
        data_file=lambda wildcards: data_files['0'][wildcards.energy],
        script='scripts/plot_eff_comparison.py'
    output:
        'plots/sys_tag_0/eff_comparison_{energy}.pdf'
    log:
        stdout='logs/eff_comparison_{energy}.log',
        stderr='logs/eff_comparison_{energy}.err'
    shell:
        'python {input.script} '
        '--paths_eff {input.paths_eff} '
        '--paths_no_eff {input.paths_no_eff} '
        '--fres {input.data_file} '
        '--output {output} '
        '--energy {wildcards.energy} '
        '> {log.stdout} 2> {log.stderr}'

rule fit_lambda_pt:
    input: data_file=get_combined_file,
           script='scripts/fit_v1_pt.py'
    output: data_points='result/sys_tag_{sys_tag}/pt_fit_{particle}_{ew}_{flow}_{energy}.csv',    
    params: 
        energy=lambda wildcards: wildcards.energy,
        yrebin = lambda wildcards: config['yrebin'][wildcards.energy][wildcards.particle]
    log: stdout='logs/sys_tag_{sys_tag}/pt_fit_{particle}_{ew}_{flow}_{energy}.log', stderr='logs/sys_tag_{sys_tag}/pt_fit_{particle}_{ew}_{flow}_{energy}.err'
    shell: 
        """
        python {input.script} {input.data_file} {output.data_points} --yrebin {params.yrebin} --ew {wildcards.ew} --max_refit 500 > {log.stdout} 2> {log.stderr}
        """

rule fit_lambda_debug:
    input: data_file='result/sys_tag_{sys_tag}/combined_{particle}_{flow}_{energy}.root',
           script='scripts/fit_v1.py'
    output: data_points='debug/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.csv'    
    params:
        energy=lambda wildcards: wildcards.energy,
        yrebin = lambda wildcards: config['yrebin'][wildcards.energy][wildcards.particle],
        cen_bin=2,
        y_bin=1
    log: stdout='logs/debug/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.log', stderr='logs/debug/sys_tag_{sys_tag}/fit_{particle}_{flow}_{energy}.err'
    shell: 
        """
        python {input.script} {input.data_file} {output.data_points} --yrebin {params.yrebin} --max_refit 500 --debug --cen_bin {params.cen_bin} --y_bin {params.y_bin} > {log.stdout} 2> {log.stderr}
        """

rule plot_v1:
    input: data_file=lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy],
           piKp_v1=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/result.csv', energy=wildcards.energy, particle=['pions', 'kaons', 'protons']),
           lambda_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/fit_{particle}_{flow}_{e}.csv', sys_tag=wildcards.sys_tag, particle=config['particles'], flow=config['flows'], e=wildcards.energy),
           lambda_v1_pt=lambda wildcards: expand('result/sys_tag_{sys_tag}/pt_fit_{particle}_{ew}_{flow}_{energy}.csv', sys_tag=wildcards.sys_tag, particle=config['particles'], ew=['east', 'west'], flow='v1', energy=wildcards.energy),
           script='scripts/plot_v1.py'
    params:
        energy=lambda wildcards: wildcards.energy,
        plot_v2_lo = lambda wildcards: config['plotting'][wildcards.energy]['v2_lo'],
        plot_v2_hi = lambda wildcards: config['plotting'][wildcards.energy]['v2_hi'],
        order = lambda wildcards: config['fit_order'][wildcards.energy],
    output: 'plots/sys_tag_{sys_tag}/v1_cen_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/delta_v1_cen_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/v1_pt_{energy}_lambda.pdf',
            'plots/sys_tag_{sys_tag}/v1_pt_{energy}_lambdabar.pdf',
            'plots/sys_tag_{sys_tag}/a1_cen_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/dv1a1dy_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/resolution_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/paper_yaml/resolution_{energy}.yaml',
            'plots/sys_tag_{sys_tag}/dv1dy_coal_{energy}.pdf',
            'plots/sys_tag_{sys_tag}/paper_yaml/dv1dy_coal_{energy}.yaml',
            data_points='result/sys_tag_{sys_tag}/data_{energy}.txt'
    log: stdout='logs/sys_tag_{sys_tag}/plot_v1_{energy}.log', stderr='logs/sys_tag_{sys_tag}/plot_v1_{energy}.err'
    shell: 
        'python {input.script} --fres {input.data_file} --paths {input.lambda_v1} --energy {params.energy} --method {params.order}'
        ' --paths_piKp {input.piKp_v1} --paths_pt {input.lambda_v1_pt} --output {output.data_points} --sys_tag {wildcards.sys_tag} --yrange {params.plot_v2_lo} {params.plot_v2_hi}'
        ' > {log.stdout} 2> {log.stderr}'

# for systematic checks that still use the default dataset
rule plot_v1_special_sys:
    input: data_file=lambda wildcards: data_files['0'][wildcards.energy],
           piKp_v1=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/result.csv', energy=wildcards.energy, particle=['pions', 'kaons', 'protons']),
           lambda_v1=lambda wildcards: expand('result/sys_tag_0/fit_{particle}_{flow}_{e}.csv', particle=config['particles'], flow=config['flows'], e=wildcards.energy),
           lambda_v1_pt=lambda wildcards: expand('result/sys_tag_0/pt_fit_{particle}_{ew}_{flow}_{energy}.csv', particle=config['particles'], ew=['east', 'west'], flow=config['flows'], energy=wildcards.energy),
           script='scripts/plot_v1.py'
    params:
        energy=lambda wildcards: wildcards.energy,
        sys_tag=lambda wildcards: wildcards.sys_tag,
        plot_v2_lo = lambda wildcards: config['plotting'][wildcards.energy]['v2_lo'],
        plot_v2_hi = lambda wildcards: config['plotting'][wildcards.energy]['v2_hi'],
        order = lambda wildcards: config['fit_order'][wildcards.energy]
    wildcard_constraints:
        sys_tag='[568]'
    output: 'plots/special_sys_tag_{sys_tag}/v1_cen_{energy}.pdf',
            # 'plots/special_sys_tag_{sys_tag}/a1_cen_{energy}.pdf',
            # 'plots/special_sys_tag_{sys_tag}/dv1a1dy_{energy}.pdf',
            'plots/special_sys_tag_{sys_tag}/resolution_{energy}.pdf',
            'plots/special_sys_tag_{sys_tag}/paper_yaml/resolution_{energy}.yaml',
            'plots/special_sys_tag_{sys_tag}/dv1dy_coal_{energy}.pdf',
            'plots/special_sys_tag_{sys_tag}/paper_yaml/dv1dy_coal_{energy}.yaml',
            data_points='result/special_sys_tag_{sys_tag}/data_{energy}.txt'
    log: stdout='logs/special_sys_tag_{sys_tag}/plot_v1_{energy}.log', stderr='logs/special_sys_tag_{sys_tag}/plot_v1_{energy}.err'
    shell: 
        'python {input.script} --fres {input.data_file} --paths {input.lambda_v1} --energy {params.energy} --method {params.order}'
        ' --paths_piKp {input.piKp_v1} --paths_pt {input.lambda_v1_pt} --output {output.data_points} --sys_tag {wildcards.sys_tag} --yrange {params.plot_v2_lo} {params.plot_v2_hi}'
        ' > {log.stdout} 2> {log.stderr}'

rule combine_lambda_with_eff_yint:
    """special_sys_tag_7: y-integrated efficiency correction (vs per-y-bin default)."""
    input:
        data_file=lambda wildcards: data_files['0'][wildcards.energy],
        eff_file=lambda wildcards: f'result/eff/efficiency_{wildcards.particle.lower()}_{wildcards.energy}.root',
        script='scripts/combine_lambda_with_eff.cpp'
    output: 'result/special_sys_tag_7/combined_{particle}_{flow}_{energy}_eff_corrected.root'
    params:
        pt_lo = config['pt_lo'],
        pt_hi = config['pt_hi'],
        y_cut = config['y_cut']
    log:
        stdout='logs/special_sys_tag_7/combine_eff_{particle}_{flow}_{energy}.log',
        stderr='logs/special_sys_tag_7/combine_eff_{particle}_{flow}_{energy}.err'
    shell:
        """
        {DOCKER_ROOT} -b -q -l '{input.script}("{input.data_file}", "./result/special_sys_tag_7/", "{input.eff_file}", {params.pt_lo}, {params.pt_hi}, "{wildcards.particle}", "{wildcards.flow}", "{wildcards.energy}", {params.y_cut}, 0)' > {log.stdout} 2> {log.stderr}
        """

def get_combined_file_yint(wildcards):
    particle_eff_energies = {
        'lambda':    list(eff_files.keys()),
        'lambdabar': list(eff_files_lambdabar.keys()),
    }
    available = particle_eff_energies.get(wildcards.particle.lower(), [])
    if wildcards.energy in available:
        return f"result/special_sys_tag_7/combined_{wildcards.particle}_{wildcards.flow}_{wildcards.energy}_eff_corrected.root"
    else:
        return f"result/sys_tag_0/combined_{wildcards.particle}_{wildcards.flow}_{wildcards.energy}.root"

rule fit_particle_yint:
    """Fit v1(y) on y-integrated eff-corrected files for special_sys_tag_7."""
    input:
        data_file=get_combined_file_yint,
        script='scripts/fit_v1.py'
    output:
        data_points='result/special_sys_tag_7/fit_{particle}_{flow}_{energy}.csv',
        invmass_plot='plots/special_sys_tag_7/paper_yaml/invmass/{particle}_fit_{flow}_{energy}_invmass_cen4_y0.7.yaml',
        v1_fit_plot='plots/special_sys_tag_7/paper_yaml/v1fit/{particle}_fit_{flow}_{energy}_v1fit_cen4_y0.7.yaml'
    params:
        energy=lambda wildcards: wildcards.energy,
        yrebin=lambda wildcards: config['yrebin'][wildcards.energy][wildcards.particle]
    log:
        stdout='logs/special_sys_tag_7/fit_{particle}_{flow}_{energy}.log',
        stderr='logs/special_sys_tag_7/fit_{particle}_{flow}_{energy}.err'
    shell:
        """
        python {input.script} {input.data_file} {output.data_points} \
            --yrebin {params.yrebin} \
            --max_refit 500 \
            --paper_plot_path {output.invmass_plot} \
            > {log.stdout} 2> {log.stderr}
        """

rule plot_v1_yint:
    """Plot v1 for special_sys_tag_7 (y-integrated eff) using default dataset."""
    input:
        data_file=lambda wildcards: data_files['0'][wildcards.energy],
        piKp_v1=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/result.csv', energy=wildcards.energy, particle=['pions', 'kaons', 'protons']),
        lambda_v1=lambda wildcards: expand('result/special_sys_tag_7/fit_{particle}_{flow}_{e}.csv', particle=config['particles'], flow=config['flows'], e=wildcards.energy),
        lambda_v1_pt=lambda wildcards: expand('result/sys_tag_0/pt_fit_{particle}_{ew}_{flow}_{energy}.csv', particle=config['particles'], ew=['east', 'west'], flow=config['flows'], energy=wildcards.energy),
        script='scripts/plot_v1.py'
    params:
        energy=lambda wildcards: wildcards.energy,
        plot_v2_lo=lambda wildcards: config['plotting'][wildcards.energy]['v2_lo'],
        plot_v2_hi=lambda wildcards: config['plotting'][wildcards.energy]['v2_hi'],
        order=lambda wildcards: config['fit_order'][wildcards.energy]
    output:
        'plots/special_sys_tag_7/v1_cen_{energy}.pdf',
        'plots/special_sys_tag_7/resolution_{energy}.pdf',
        'plots/special_sys_tag_7/paper_yaml/resolution_{energy}.yaml',
        'plots/special_sys_tag_7/dv1dy_coal_{energy}.pdf',
        'plots/special_sys_tag_7/paper_yaml/dv1dy_coal_{energy}.yaml',
        data_points='result/special_sys_tag_7/data_{energy}.txt'
    log:
        stdout='logs/special_sys_tag_7/plot_v1_{energy}.log',
        stderr='logs/special_sys_tag_7/plot_v1_{energy}.err'
    shell:
        'python {input.script} --fres {input.data_file} --paths {input.lambda_v1} --energy {params.energy} --method {params.order}'
        ' --paths_piKp {input.piKp_v1} --paths_pt {input.lambda_v1_pt} --output {output.data_points} --sys_tag 7 --yrange {params.plot_v2_lo} {params.plot_v2_hi}'
        ' > {log.stdout} 2> {log.stderr}'

rule plot_v1_Xi:
    input: data_file=lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy],
           piKp_v1=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/result.csv', energy=wildcards.energy, particle=['pions', 'kaons', 'protons']),
           lambda_v1='result/sys_tag_{sys_tag}/data_{energy}.txt',
           xi_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/fit_{particle}_{flow}_{e}.csv', sys_tag=wildcards.sys_tag, particle=['Xi', 'Xibar'], flow=config['flows'], e=wildcards.energy),
           script='scripts/plot_v1_Xi.py'
    params:
        energy=lambda wildcards: wildcards.energy,
        plot_v2_lo = lambda wildcards: config['plotting'][wildcards.energy]['v2_lo'],
        plot_v2_hi = lambda wildcards: config['plotting'][wildcards.energy]['v2_hi']
    output: 'plots/sys_tag_{sys_tag}/v1_cen_{energy}_xi.pdf',
            # 'plots/sys_tag_{sys_tag}/a1_cen_{energy}_xi.pdf',
            # 'plots/sys_tag_{sys_tag}/dv1a1dy_{energy}_xi.pdf',
            'plots/sys_tag_{sys_tag}/dv1dy_coal_{energy}_xi.pdf',
            'plots/sys_tag_{sys_tag}/paper_yaml/dv1dy_coal_{energy}_xi.yaml',
            data_points='result/sys_tag_{sys_tag}/data_{energy}_xi.txt'
    log: stdout='logs/sys_tag_{sys_tag}/plot_v1_{energy}_xi.log', stderr='logs/sys_tag_{sys_tag}/plot_v1_{energy}_xi.err'
    shell: 
        'python {input.script} --fres {input.data_file} --paths {input.xi_v1} --energy {params.energy}'
        ' --paths_piKp {input.piKp_v1} --path_lambda {input.lambda_v1}'
        ' --output {output.data_points} --yrange {params.plot_v2_lo} {params.plot_v2_hi}'
        ' > {log.stdout} 2> {log.stderr}'

rule prepare_piKp:
    input: data_file=lambda wildcards: expand('data/v1_piKp/{energy}/{particle}/cen{cen}.v2_pion.root', energy=wildcards.energy, particle=wildcards.particle, cen=wildcards.cen),
           script='scripts/Finish_v1_tof_eff.C'
    output: v1_file='result/v1_piKp/{energy}/{particle}/cen{cen,[1-9]}.v1_pion.root',
            a1_file='result/v1_piKp/{energy}/{particle}/cen{cen,[1-9]}.a1_pion.root'
    params: 
        energy=lambda wildcards: wildcards.energy,
        particle=lambda wildcards: wildcards.particle,
        cen=lambda wildcards: wildcards.cen,
        nquark = lambda wildcards: config['nquark'][wildcards.particle],
        pt_lo = lambda wildcards: config['piKp_ptnq_lo'] * config['nquark'][wildcards.particle],
        pt_hi = lambda wildcards: config['piKp_ptnq_hi'] * config['nquark'][wildcards.particle]
    log: stdout='logs/prepare_piKp_{energy}_{particle}_{cen}.log', stderr='logs/prepare_piKp_{energy}_{particle}_{cen}.err'
    shell:
        """
        {DOCKER_ROOT} -b -q '{input.script}({params.cen},"{input.data_file}","{output.v1_file}",0,{params.pt_lo},{params.pt_hi})' > {log.stdout} 2> {log.stderr}
        {DOCKER_ROOT} -b -q '{input.script}({params.cen},"{input.data_file}","{output.a1_file}",1,{params.pt_lo},{params.pt_hi})' > {log.stdout} 2> {log.stderr}
        """

rule fit_piKp:
    input: v1_file=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/cen{cen}.v1_pion.root', cen=np.append(np.arange(1,10), [13,57,89]), energy=wildcards.energy, particle=wildcards.particle),
           a1_file=lambda wildcards: expand('result/v1_piKp/{energy}/{particle}/cen{cen}.a1_pion.root', cen=np.append(np.arange(1,10), [13,57,89]), energy=wildcards.energy, particle=wildcards.particle),
           script='scripts/FitSlope.C'
    output: data_points='result/v1_piKp/{energy}/{particle}/result.csv',
            plot='plots/v1_piKp/{energy}/v1_{particle}.png'
    params: 
        script_base = 'FitSlope.C',
        output_base = 'result.csv',
        plot_base = 'v1_{particle}.png',
        energy=lambda wildcards: wildcards.energy,
        particle=lambda wildcards: wildcards.particle,
        order = lambda wildcards: config['fit_order'][wildcards.energy],
    log: stdout='logs/fit_piKp_{energy}_{particle}.log', stderr='logs/fit_piKp_{energy}_{particle}.err'
    shell: 
        """
        mkdir -p temp_{params.energy}_{params.particle}
        cp {input.v1_file} temp_{params.energy}_{params.particle}
        cp {input.a1_file} temp_{params.energy}_{params.particle}
        cp {input.script} temp_{params.energy}_{params.particle}
        docker run --rm -v "$(pwd)":/work -w /work/temp_{params.energy}_{params.particle} rootproject/root:latest root -b -q '{params.script_base}("{params.output_base}", "{params.plot_base}",{params.order})' > {log.stdout} 2> {log.stderr}
        mv temp_{params.energy}_{params.particle}/{params.output_base} {output.data_points}
        mv temp_{params.energy}_{params.particle}/{params.plot_base} {output.plot}
        rm -r temp_{params.energy}_{params.particle}
        """

rule plot_other_coal:
    input: lambda_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/data_{energy}.txt', energy=energies, sys_tag=wildcards.sys_tag),
           xi_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/data_{energy}_xi.txt', energy=energies, sys_tag=wildcards.sys_tag),
           proton_v1=expand('result/v1_piKp/{energy}/protons/result.csv', energy=energies),
           pion_v1=expand('result/v1_piKp/{energy}/pions/result.csv', energy=energies),
           kaon_v1=expand('result/v1_piKp/{energy}/kaons/result.csv', energy=energies),
           script='scripts/plot_other_coal.py'
    output: 'plots/sys_tag_{sys_tag}/other_coal.pdf'
    params:
        energies = energies,
        sys_tag=lambda wildcards: wildcards.sys_tag
    log: stdout='logs/sys_tag_{sys_tag}/plot_other_coal.log', stderr='logs/sys_tag_{sys_tag}/plot_other_coal.err'
    shell: 
        'python {input.script} --lambda_v1 {input.lambda_v1} --xi_v1 {input.xi_v1} --proton_v1 {input.proton_v1} --kaon_v1 {input.kaon_v1} --pion_v1 {input.pion_v1}'
        ' --energies {params.energies} --sys_tag {params.sys_tag} > {log.stdout} 2> {log.stderr}'

rule testing_coalescence:
    input: lambda_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/data_{energy}.txt', energy=energies, sys_tag=wildcards.sys_tag),
           xi_v1=lambda wildcards: expand('result/sys_tag_{sys_tag}/data_{energy}_xi.txt', energy=energies, sys_tag=wildcards.sys_tag),
           proton_v1=expand('result/v1_piKp/{energy}/protons/result.csv', energy=energies),
           pion_v1=expand('result/v1_piKp/{energy}/pions/result.csv', energy=energies),
           kaon_v1=expand('result/v1_piKp/{energy}/kaons/result.csv', energy=energies),
           dv1dy_coal=expand('plots/sys_tag_0/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies),
           script='scripts/coal.py'
    output: 'plots/sys_tag_{sys_tag}/comparison_1040_1.png'
    params:
        energies = energies,
        sys_tag=lambda wildcards: wildcards.sys_tag
    log: stdout='logs/sys_tag_{sys_tag}/coal.log', stderr='logs/sys_tag_{sys_tag}/coal.err'
    shell: 
        'python {input.script} --lambda_v1 {input.lambda_v1} --xi_v1 {input.xi_v1} --proton_v1 {input.proton_v1} --kaon_v1 {input.kaon_v1} --pion_v1 {input.pion_v1}'
        ' --input_dv1dy_coal {input.dv1dy_coal} --energies {params.energies} --sys_tag {params.sys_tag} > {log.stdout} 2> {log.stderr}'

rule checking_lambda_reco:
    input: data_file=lambda wildcards: [data_files[str(sys_tag)][wildcards.energy] for sys_tag in [0,1,2]],
           script='scripts/check_lambda_reco.py'
    output: 'debug/lambda_reco_{energy}.pdf'
    log: stdout='logs/debug/check_lambda_reco_{energy}.log', stderr='logs/debug/check_lambda_reco_{energy}.err'
    shell:
        'python {input.script} --data_files {input.data_file} --output_file {output} > {log.stdout} 2> {log.stderr}'

rule plot_model:
    input: model_file='data/model/{model}/{energy}.root',
           script='scripts/read_model.cpp'
    output: 'result/model/{model}/data_{energy}_lambda.csv'
    log: stdout='logs/model/{model}_{energy}.log', stderr='logs/model/{model}_{energy}.err'
    shell:
        """
        {DOCKER_ROOT} -b -q -l '{input.script}("{input.model_file}", "{output}")' > {log.stdout} 2> {log.stderr}
        """

# rule plot_all:
#     input: lambda_v1=expand('result/data_{energy}.txt', energy=energies),
#            proton_v1=expand('result/v1_piKp/{energy}/protons/result.csv', energy=energies),
#            kaon_v1=expand('result/v1_piKp/{energy}/kaons/result.csv', energy=energies),
#            script='scripts/plot_all.py'
#     output: 'plots/dv1dy_1040.pdf',
#             'plots/dv1dy_5080.pdf'
#     params:
#         energies = energies
#     log: stdout='logs/plot_all.log', stderr='logs/plot_all.err'
#     shell: 
#         'python {input.script} --lambda_v1 {input.lambda_v1} --proton_v1 {input.proton_v1} --kaon_v1 {input.kaon_v1}'
#         ' --energies {params.energies} > {log.stdout} 2> {log.stderr}'


rule combine_sys:
    input: script='scripts/combine_sys.py',
           default='plots/sys_tag_0/paper_yaml/dv1dy_coal_{energy}.yaml',
           # regular_sys='result/blank/{energy}.txt',
           regular_sys=lambda wildcards: expand('plots/sys_tag_{sys_tag}/paper_yaml/dv1dy_coal_{energy}.yaml', sys_tag=[1,2,3], energy=wildcards.energy),
           # tag 6 (cubic fit-order) removed: dominated by cubic overfitting on thin
           # peripheral/low-energy data (Δ up to ~0.09, 2-9x the slope) rather than a
           # genuine fit-order systematic. tags 5+8 = positive/negative half-y range
           # (combined as a single paired y-range systematic), tag 7 = y-integrated eff.
           special_sys=lambda wildcards: expand('plots/special_sys_tag_{sys_tag}/paper_yaml/dv1dy_coal_{energy}.yaml', sys_tag=[5,7,8], energy=wildcards.energy)
    output: 'plots/final/paper_yaml/dv1dy_coal_{energy}.yaml'
    params:
        sys_divisor = config['sys_divisor']
    log: stdout='logs/combine_sys_{energy}.log', stderr='logs/combine_sys_{energy}.err'
    shell:
        'python {input.script} --default {input.default} --regular_sys {input.regular_sys} --special_sys {input.special_sys} --output {output} --energy {wildcards.energy} --sys_divisor {params.sys_divisor} > {log.stdout} 2> {log.stderr}'
        
rule blank:
    # generate a blank file
    output: 'result/blank/{energy}.txt'
    shell: 'touch {output}'

rule generate_paper_plots:
    input: script='scripts/generate_paper_plots.py',
           script_clean='scripts/rm_unused_plots.py',
           invmass='plots/sys_tag_0/paper_yaml/invmass/Lambda_fit_v1_19p6GeV_invmass_cen4_y0.7.yaml',
           v1fit='plots/sys_tag_0/paper_yaml/v1fit/Lambda_fit_v1_19p6GeV_v1fit_cen4_y0.7.yaml',
           res=expand('plots/sys_tag_0/paper_yaml/resolution_{energy}.yaml', energy=energies),
           dv1dy_coal=expand('plots/final/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies), # ['7p7GeV', '19p6GeV']), # for final, use the combined systematic
           # dv1dy_coal=expand('plots/sys_tag_0/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies),
           dv1dy_coal_xi=expand('plots/sys_tag_0/paper_yaml/dv1dy_coal_{energy}_xi.yaml', energy=energies),
           model_sim_urqmd=expand('result/model/urqmd/data_{energy}_lambda.csv', energy=['7p7GeV', '17p3GeV', '19p6GeV']),
           model_sim_ampt=expand('result/model/ampt/data_{energy}_lambda.csv', energy=['14p6GeV'])
    output: report='plots/paper/report.pdf',
            data_points=expand('plots/paper/data_points/dv1dy_{energy}.csv', energy=energies)
    params:
        eff_energies=sorted(set(list(eff_files.keys()) + list(eff_files_lambdabar.keys())))
    # log: stdout='logs/sys_tag_0/generate_paper_plots.log', stderr='logs/sys_tag_0/generate_paper_plots.err'
    log: stdout='logs/final/generate_paper_plots.log', stderr='logs/final/generate_paper_plots.err'
    shell:
        """
        python {input.script} --input_invmass {input.invmass} --input_v1fit {input.v1fit} --input_res {input.res} --input_dv1dy_coal {input.dv1dy_coal} --input_dv1dy_coal_xi {input.dv1dy_coal_xi} --model_sim_urqmd {input.model_sim_urqmd} --model_sim_ampt {input.model_sim_ampt} --eff_energies {params.eff_energies} --output {output.report}
        python {input.script_clean} {input.invmass} {input.v1fit} > {log.stdout} 2> {log.stderr}
        """

# ---- Alternative proton/kaon pT-cut dataset (fig 2 & 3); does NOT touch defaults ----
ALT_PIKP_TXTDIR = 'Etabins_20_Coalescence/Etabins_20_ymp6top6_withMCeffandTOFcorrection/txtfiles'

rule gen_pikp_altcuts:
    """Build the alternative-cut piKp data module from the new dataset's txt files."""
    input:
        script='scripts/gen_pikp_merged.py',
        txt=glob.glob(f'{ALT_PIKP_TXTDIR}/*_datapoints.txt')
    output: 'scripts/pikp_merged_altcuts.py'
    shell:
        'python {input.script} --txt_dir "%s" --out {output}' % ALT_PIKP_TXTDIR

rule fig2_altcuts:
    input:
        script='scripts/gen_fig2.py',
        pikp='scripts/pikp_merged_altcuts.py',
        gpp='scripts/generate_paper_plots.py',
        yamls=expand('plots/final/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies)
    output: 'plots/paper/fig_2_altcuts.pdf'
    log: stdout='logs/final/fig2_altcuts.log', stderr='logs/final/fig2_altcuts.err'
    shell:
        'python {input.script} --input_dv1dy_coal {input.yamls}'
        ' --pikp_module pikp_merged_altcuts --cut_set altcuts --output_suffix _altcuts'
        ' > {log.stdout} 2> {log.stderr}'

rule fig3_variants:
    """fig 3 variants produced by the unified generator (consistent layout/ylim):
    default 50-80%, and alternative-cut 40-80% & 50-80%. The default 40-80%
    (fig_3_horizontal.pdf) is produced by generate_paper_plots with matching layout."""
    input:
        script='scripts/fig3_5080.py',
        pikp='scripts/pikp_merged_altcuts.py',
        gpp='scripts/generate_paper_plots.py',
        yamls=expand('plots/final/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies)
    output:
        'plots/paper/fig_3_horizontal_5080.pdf',
        'plots/paper/fig_3_horizontal_altcuts.pdf',
        'plots/paper/fig_3_horizontal_5080_altcuts.pdf'
    log: stdout='logs/final/fig3_variants.log', stderr='logs/final/fig3_variants.err'
    shell:
        'python {input.script} --input_dv1dy_coal {input.yamls} --high_cent 5080'
        ' --cut_set default > {log.stdout} 2> {log.stderr}\n'
        'python {input.script} --input_dv1dy_coal {input.yamls} --high_cent 4080'
        ' --pikp_module pikp_merged_altcuts --cut_set altcuts >> {log.stdout} 2>> {log.stderr}\n'
        'python {input.script} --input_dv1dy_coal {input.yamls} --high_cent 5080'
        ' --pikp_module pikp_merged_altcuts --cut_set altcuts >> {log.stdout} 2>> {log.stderr}'

rule test_fit_order:
    """Diagnostic: check whether cubic dv1/dy fit is needed within |y| < y_cut.
    Uses pre-existing fit CSVs for the given sys_tag (no new fitting triggered)."""
    input:
        fit_csvs=expand('result/sys_tag_{{sys_tag}}/fit_{particle}_v1_{energy}.csv',
                        particle=config['particles'], energy=energies),
        res_files=[data_files['0'][e] for e in energies],
        script='scripts/test_fit_order.py'
    output:
        'plots/sys_tag_{sys_tag}/fit_order_test.pdf'
    log:
        stdout='logs/sys_tag_{sys_tag}/fit_order_test.log',
        stderr='logs/sys_tag_{sys_tag}/fit_order_test.err'
    shell:
        'python {input.script} '
        '--fit_csvs {input.fit_csvs} '
        '--res_files {input.res_files} '
        '--output {output} '
        '> {log.stdout} 2> {log.stderr}'

rule generate_spectrum:
    input: script='scripts/generate_spectrum.py',
           input_file=lambda wildcards: expand('plots/sys_tag_{sys_tag}/paper_yaml/invmass/{particle}_fit_v1_{energy}_invmass_cen4_y0.7.yaml', energy=['19p6GeV'], sys_tag=wildcards.sys_tag, particle=wildcards.particle)
    output: 'plots/sys_tag_{sys_tag}/{particle}_spectrum.pdf'
    log: stdout='logs/sys_tag_{sys_tag}/{particle}_spectrum.log', stderr='logs/sys_tag_{sys_tag}/{particle}_spectrum.err'
    shell:
        'python {input.script} --input_path {input.input_file} --output_path {output} > {log.stdout} 2> {log.stderr}'

rule plot_eff_impact_fig3:
    """Diagnostic: energy-dependence version of eff impact — 3 centrality panels vs sqrt(s_NN)."""
    input:
        script='scripts/plot_eff_impact_fig3.py',
        yaml_files=expand('plots/final/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies),
        no_eff_csvs=expand('result/no_eff/fit_{particle}_v1_{energy}.csv',
                           particle=config['particles'], energy=energies),
        data_files=[data_files['0'][e] for e in energies]
    output:
        'plots/eff_impact/delta_lambda_eff_comparison_fig3.pdf'
    log:
        stdout='logs/eff_impact/plot_eff_impact_fig3.log',
        stderr='logs/eff_impact/plot_eff_impact_fig3.err'
    shell:
        'python {input.script} '
        '--yaml_files {input.yaml_files} '
        '--no_eff_csvs {input.no_eff_csvs} '
        '--data_files {input.data_files} '
        '--output {output} '
        '> {log.stdout} 2> {log.stderr}'

rule plot_eff_impact:
    """Diagnostic: compare eff-corrected vs uncorrected Delta-Lambda alongside Delta(p)-Delta(K)."""
    input:
        script='scripts/plot_eff_impact.py',
        yaml_files=expand('plots/final/paper_yaml/dv1dy_coal_{energy}.yaml', energy=energies),
        no_eff_csvs=expand('result/no_eff/fit_{particle}_v1_{energy}.csv',
                           particle=config['particles'], energy=energies),
        data_files=[data_files['0'][e] for e in energies]
    output:
        'plots/eff_impact/delta_lambda_eff_comparison.pdf'
    log:
        stdout='logs/eff_impact/plot_eff_impact.log',
        stderr='logs/eff_impact/plot_eff_impact.err'
    shell:
        'python {input.script} '
        '--yaml_files {input.yaml_files} '
        '--no_eff_csvs {input.no_eff_csvs} '
        '--data_files {input.data_files} '
        '--output {output} '
        '> {log.stdout} 2> {log.stderr}'