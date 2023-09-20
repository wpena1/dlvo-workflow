#from amd_parsl_cfg import *
#from cpu_parsl_cfg import *
# PARSL STUFF
import parsl
from parsl.app.app import python_app, bash_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.data_provider.files import File
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
#from parsl.addresses import address_by_hostname

config = Config(
    executors=[
         HighThroughputExecutor(
              label='Greene_HTEX',
              worker_debug=True,
              max_workers=1,
              cores_per_worker=int(1),
              working_dir = './',
              worker_logdir_root = './logs',
              provider = SlurmProvider(
                    nodes_per_block = int(1),
                    min_blocks = int(0),
                    max_blocks = int(10),
                    cores_per_node = int(24),
                    parallelism = 1.0,
                    partition = '',
                    scheduler_options = '',
                    mem_per_node = int(4),
                    launcher=SrunLauncher(),
                    exclusive = False,
                    walltime='48:00:00'
            )
        ),
      ]
)
parsl.clear()
parsl.load(config)

@bash_app
def run_dlvo(inputs=[],outputs=[]):
    import os
    run_dir = os.path.dirname(inputs[0])
    return f"cd {run_dir};bash run_simulation.bash"


if __name__=="__main__":
    import os
    import sys
    import argparse
    import shutil
    from path import Path
    from yaml_sweep import update_yaml
    print("running main")
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action= "store",nargs ="*", help="lattice repeats",default=10, dest = "lattice_repeats", type = int)
    parser.add_argument('-LS', action= "store",nargs ="*", help="lattice spacing",default=5.0, dest = "lattice_spacing", type = float)
    parser.add_argument('-FP', action= "store", nargs="*", help="fraction of positive particles in a unit cell",default=0.5, dest="fraction_positive", type=float)
    parser.add_argument('-RN', action="store", nargs="*", default=None, dest="radiusN", type=float)
    parser.add_argument('-RP', action="store", nargs="*", default=None, dest="radiusP", type=float)
    parser.add_argument('-DL',action="store", nargs="*", help="debye length value",default=5.0, dest="debye_length",type=float)
    parser.add_argument('-seed',action="store", nargs="*", help="seed value",default=10, dest="seed",type=int)
    parser.add_argument('-yaml_file',help="yaml file name",default='inputs.yaml')
    parser.add_argument('-output_dir',help="output directory",default='binary_dlvo')
    arg_dict = parser.parse_args().__dict__
    input_yaml_files = update_yaml(arg_dict)
    result_list = []
    for i,run_file in enumerate(input_yaml_files):
        # print("Input file: ", run_file)
        print(f"run: {i}\n==============")
        inputs = [Path(run_file),Path('run_simulation.bash'),Path('run_dlvo_spheres_metal.py') ]
        shutil.copy('run_dlvo_spheres_metal.py', os.path.dirname(run_file), follow_symlinks=True)
        shutil.copy('run_simulation.bash', os.path.dirname(run_file), follow_symlinks=True)
        r = run_dlvo(inputs=inputs)
        result_list.append(r)

    [r.result() for r in result_list]
    # print('Result: {}'.format(r.result()))
    print("done")
