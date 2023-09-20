import os
import parsl
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.config import Config
config = Config(
    executors=[
         HighThroughputExecutor(
              label='slurm',
              worker_debug=True,             # Default False for shorter logs
              cores_per_worker=int(1),       # DOES NOT correspond to --cpus-per-task 1 per Parsl docs.  Rather BYPASSES SLURM opts and is process_pool.py -c cores_per_worker, but IS NOT CORES ON SLURM - sets number of workers
              working_dir = '/tmp/pworks/',
              worker_logdir_root = os. getcwd() + '/parsllogs',
              provider = SlurmProvider(
                        #========GPU RUNS=============
                        #partition = 'rtx8000',         # For GPU runs! Or v100.  gr,gv,cr,cv on NYU HPC site does not work
                        #scheduler_options = '#SBATCH --gres=gpu:1', # For GPU runs!
                        #========CPU RUNS============
                        #scheduler_options = '#SBATCH --ntasks-per-node=40',  # DO NOT USE! Conflicts with cores_per_worker where Parsl sets --ntasks-per-node on separate SBATCH command, see note above.
              partition = 'mi50',          # Cluster specific! Needs to match GPU availability, and RAM per CPU limits specified for partion.
                        #===========================
              scheduler_options = '#SBATCH --gres=gpu:mi50:1',
              mem_per_node = int(20),
              nodes_per_block = int(1),
              cores_per_node = int(1),   # Corresponds to --cpus-per-task
              min_blocks = int(0),
              max_blocks = int(10),
              parallelism = 1,           # Was 0.80, 1 is "use everything you can NOW"
              exclusive = False,         # Default is T, hard to get workers on shared cluster
              walltime='02:00:00',       # Will limit job to this run time, 10 min default Parsl
              launcher=SrunLauncher() # defaults to SingleNodeLauncher() which seems to work, experiment later?
              )
          )
      ]
)
parsl.load(config)
