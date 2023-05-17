import parsl
import os
from parsl.app.app import python_app, bash_app
from parsl.data_provider.files import File
from path import Path
from parslpw import pwconfig, pwargs

parsl.clear()
parsl.load(pwconfig)

@bash_app
def run_dlvo(inputs=[], outputs=[], stdout="run.out", stderr="run.err"):
    run_dir = inputs[0]
    nsteps = inputs[1]
    return f"cd {run_dir};bash run_simulation.bash {nsteps}"


if __name__=="__main__":
    import sys
    import argparse
    import shutil
    from path import Path
    from yaml_sweep import update_yaml
    print("running main")
    lattice_repeats = [ int(a) for a in str(pwargs.lattice_repeats).split('_')]
    lattice_spacing = [float(ls) for ls in str(pwargs.lattice_spacing).split('_')]
    fraction_positive = [float(fp) for fp in str(pwargs.fraction_positive).split()]
    radiusN = [float(rn) for rn in str(pwargs.radiusN).split()]
    radiusP = [float(rp) for rp in str(pwargs.radiusP).split()]
    debye_length = [float(dl) for dl in str(pwargs.debye_length).split()]
    seed = [int(sd) for sd in str(pwargs.seed).split()]
    nsteps = [int(steps) for steps in str(pwargs.nsteps).split()]
    input_file = 'inputs.yaml'
    output_directory = str(pwargs.output_dir)

    arg_dict = {'lattice_repeats': lattice_repeats, 'lattice_spacing': lattice_spacing, 'fraction_positive':fraction_positive, 'radiusN':radiusN, 'radiusP': radiusP, \
                'debye_length': debye_length, 'seed': seed, 'yaml_file': input_file, 'output_dir': output_directory, 'nsteps':nsteps}

    run_dir = os.getcwd()
    source_dir = os.path.join('.', output_directory)

    # print(arg_dict)
    print('output at: ', source_dir)
    print('current directory: ', run_dir)
    rows = ['in:a,in:ls,in:fp,in:rn,in:rp,in:dl,in:seed,img:cluster_time'] #, out:cluster_num']
    input_rows = update_yaml(arg_dict)
    input_yaml_files = input_rows[0]
    result_list = []
    full_paths = []
    for i,run_file in enumerate(input_yaml_files):
        print(f"=============== run: {i} ===============")
        run_file_dir = os.path.dirname(run_file)
        outputdir = os.path.join('.', run_file_dir)
        output = Path(outputdir)
        print(F"output: {output}")
        full_paths.append(output)
        # inputs = [output, Path('run_simulation.bash'),Path('run_dlvo_spheres_metal.py') ]
        inputs = [output, nsteps[0]]
        shutil.copy('run_dlvo_spheres_metal.py', output, follow_symlinks=True)
        shutil.copy('run_simulation.bash', output, follow_symlinks=True)
        shutil.copy('clustering_analysis.py', output, follow_symlinks=True)
        shutil.copy('render_frame.py', output, follow_symlinks=True)

        r = run_dlvo(inputs=inputs, outputs=[output])
        result_list.append(r)

    [r.result() for r in result_list]
    print("=======Jobs finished========")

    for i,param_dict in enumerate(input_rows[1]):
        in_a = param_dict['lattice_repeats']
        in_ls = param_dict['lattice_spacing']
        in_fp = param_dict['fraction_positive']
        in_rn = param_dict['radiusN']
        in_rp = param_dict['radiusP']
        in_dl = param_dict['debye_length']
        in_seed = param_dict['seed']
        img_cluster = F"{run_dir}/{full_paths[i]}/{param_dict['outputprefix']}.plot.png"

        rows.append(F"{in_a},{in_ls},{in_fp},{in_rn},{in_rp},{in_dl},{in_seed},{img_cluster}")

    with open(F'{source_dir}/out.csv','w') as csvfile:
        csvfile.write("\n".join(rows))
    with open(F'{source_dir}/out.html', 'w') as html:
        csvpath=F'{run_dir}/{source_dir}/out.csv'
        html.write("""\
        <html style="overflow-y:hidden;background:white">\
        <a style="font-family:sans-serif;z-index:1000;position:absolute;top:15px;right:0px;margin-right:20px;font-style:italic;font-size:10px"\
        href="/preview/DesignExplorer/index.html?datafile={}&colorby={}" target="_blank">Open in New Window</a>\
        <iframe width="100%" height="100%" src="/preview/DesignExplorer/index.html?datafile={}&colorby={}" frameborder="0"></iframe>\
        </html>""".format(csvpath,'dl', csvpath, 'dl'))
