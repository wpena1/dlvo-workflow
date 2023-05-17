#!/bin/bash

host=$(env - hostname -s)
wrapper=/scratch/work/public/singularity/run-hoomd-2.9.6.bash

exe=run_dlvo_spheres_metal.py

outprefix=$(grep outputprefix inputs_new.yaml  |cut -f 2 -d ":" |xargs)

# if [[ $host =~ ^gm ]]; then
  #  wrapper=/scratch/work/public/hudson/images/run-hoomd.bash
# fi
nsteps=${1}
$wrapper python -u ${exe} ${simulation_options}  > ${outprefix}.hoomd.log && touch ${outprefix}.done.txt

wrapper=/home/wpc252/misc-scripts/run-render-frame.bash
exe = render_frame.py
$wrapper python -u ${exe} ${outprefix}.run.${nsteps}