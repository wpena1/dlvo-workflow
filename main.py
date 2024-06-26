import parsl
from parsl.app.app import python_app, bash_app
print(parsl.__version__, flush = True)

# import parsl_utils stuff below
import parsl_utils
from parsl_utils.config import config, resource_labels, form_inputs
from parsl_utils.data_provider import PWFile

print("MAIN.py:...Configuring Parsl...")
parsl.load(config)
print("MAIN.py:...Parsl config loaded...")


@parsl_utils.parsl_wrappers.log_app
@bash_app
def run_dlvo(inputs=[], outputs=[], stdout="run.out", stderr="run.err"):
    run_dir = inputs[0]
    nsteps = inputs[1]
    return f"cd {run_dir}; bash run_simulation.bash {nsteps}"


if __name__=="__main__":
    import os
    import shutil
    from utils.yaml_sweep import update_yaml

    print(F"MAIN.py:...starting workflow setup...\n Resource list: {resource_labels}")

    lattice_repeats = [ int(a) for a in str(form_inputs['parameters']['lattice_repeats']).split('_')]
    lattice_spacing = [float(ls) for ls in str(form_inputs['parameters']['lattice_spacing']).split('_')]
    fraction_positive = [float(fp) for fp in str(form_inputs['parameters']['fraction_positive']).split()]
    radiusN = [float(rn) for rn in str(form_inputs['parameters']['radiusN']).split()]
    radiusP = [float(rp) for rp in str(form_inputs['parameters']['radiusP']).split()]
    debye_length = [float(dl) for dl in str(form_inputs['parameters']['debye_length']).split()]
    seed = [int(sd) for sd in str(form_inputs['parameters']['seed']).split()]
    nsteps = [int(steps) for steps in str(form_inputs['parameters']['nsteps']).split()]
    brush_density = [int(p) for p in str(form_inputs['otherparameters']['brush_density']).split()]
    brush_length= [int(p) for p in str(form_inputs['otherparameters']['brush_length']).split()]
    chargewall = [int(p) for p in str(form_inputs['otherparameters']['chargewall']).split()]
    closedwalls = [int(p) for p in str(form_inputs['otherparameters']['closedwalls']).split()]
    dielectric_constant = [int(p) for p in str(form_inputs['otherparameters']['dielectric_constant']).split()]
    dt = [int(p) for p in str(form_inputs['otherparameters']['dt']).split()]
    gamma = [int(p) for p in str(form_inputs['otherparameters']['gamma']).split()]
    gravity = [int(p) for p in str(form_inputs['otherparameters']['gravity']).split()]
    lattice_type = [int(p) for p in str(form_inputs['otherparameters']['lattice_type']).split()]
    massN = [int(p) for p in str(form_inputs['otherparameters']['massN']).split()]
    massP = [int(p) for p in str(form_inputs['otherparameters']['massP']).split()]
    mode = [int(p) for p in str(form_inputs['otherparameters']['mode']).split()]
    orbit_factor = [int(p) for p in str(form_inputs['otherparameters']['orbit_factor']).split()]
    surface_potentialN = [int(p) for p in str(form_inputs['otherparameters']['surface_potentialN']).split()]
    surface_potentialP = [int(p) for p in str(form_inputs['otherparameters']['surface_potentialP']).split()]
    temperature = [int(p) for p in str(form_inputs['otherparameters']['temperature']).split()]
    output_directory = str(form_inputs['parameters']['output_dir'])
    input_file = 'inputs/inputs.yaml'

    arg_dict = {'lattice_repeats': lattice_repeats, 'lattice_spacing': lattice_spacing, 'fraction_positive':fraction_positive, 'radiusN':radiusN, 'radiusP': radiusP, \
            'debye_length': debye_length, 'seed': seed, 'yaml_file': input_file, 'output_dir': output_directory, 'nsteps':nsteps, \
            'brush_density':brush_density, 'brush_length':brush_length, 'chargewall':chargewall, 'closedwalls':closedwalls, 'dielectric_constant':dielectric_constant, \
            'dt':dt, 'gamma':gamma, 'gravity':gravity, 'lattice_type':lattice_type, 'massN':massN, 'massP':massP, 'mode':mode, 'orbit_factor':orbit_factor, \
            'surface_potentialN':surface_potentialN, 'surface_potentialP':surface_potentialP, 'temperature':temperature}

    run_dir = os.getcwd()
    output_dir = os.path.join(run_dir, output_directory)
    remote_dir = os.path.join(config.executors[0].working_dir , output_directory)

    # print(arg_dict)
    print('MAIN.py: output at: ', output_dir)
    print('MAIN.py: current directory: ', run_dir)
    print('MAIN.py: remote directory: ', remote_dir)

    input_rows = update_yaml(arg_dict)
    input_yaml_files = input_rows[0]  #input yaml files are retrieved here [1] contains all parameters in a list of dictionaries
    result_list = []
    full_paths = []
    
    print("MAIN.py:.......Going in main loop to set up......")
    print(f"MAIN.py:......number of input files {len(input_yaml_files)}")

    for i,run_file in enumerate(input_yaml_files):
        print(f"=============== run: {i} ===============")
        run_file_dir = os.path.dirname(run_file)
        output_dir_remote = run_file_dir
        output_dir_local = output_dir
        output_dir_PW = PWFile(url='file://usercontainer/' + output_dir_local, local_path=remote_dir + output_dir_remote)
        print(F"MAIN.py:....PW output_dir: {output_dir_PW}")
        full_paths.append(output_dir_PW)
        # inputs = [output, Path('run_simulation.bash'),Path('run_dlvo_spheres_metal.py') ]
        inputs = [output_dir, nsteps[0]]
        shutil.copy('utils/run_dlvo_spheres_metal.py', output_dir_PW, follow_symlinks=True)
        shutil.copy('utils/run_simulation.bash', output_dir_PW, follow_symlinks=True)
        shutil.copy('utils/clustering_analysis.py', output_dir_PW, follow_symlinks=True)
        shutil.copy('utils/render_frame.py', output_dir_PW, follow_symlinks=True)

        r = run_dlvo(inputs=inputs, outputs=[output_dir_PW], stdout=remote_dir+output_dir_remote+"/run.out", stderr=remote_dir+output_dir_remote+"/run.err")
        result_list.append(r)

    [r.result() for r in result_list]
    print("MAIN.py: =======Jobs finished========")

    print("MAIN.py: ......setting up cvs and html files......")
    rows = ['in:a,in:ls,in:fp,in:rn,in:rp,in:dl,in:seed,img:cluster_time'] #, out:cluster_num']
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

    with open(F'{output_dir}/out.csv','w') as csvfile:
        csvfile.write("\n".join(rows))
    with open(F'{output_dir}/out.html', 'w') as html:
        csvpath=F'{output_dir}/out.csv'
        html.write("""\
        <html style="overflow-y:hidden;background:white">\
        <a style="font-family:sans-serif;z-index:1000;position:absolute;top:15px;right:0px;margin-right:20px;font-style:italic;font-size:10px"\
        href="/preview/DesignExplorer/index.html?datafile={}&colorby={}" target="_blank">Open in New Window</a>\
        <iframe width="100%" height="100%" src="/DesignExplorer/index.html?datafile={}&colorby={}" frameborder="0"></iframe>\
        </html>""".format(csvpath,'dl', csvpath, 'dl'))
