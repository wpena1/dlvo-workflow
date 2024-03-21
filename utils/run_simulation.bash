#!/bin/bash

nsteps=${1}
outprefix=$(grep outputprefix inputs_new.yaml  |cut -f 2 -d ":" |xargs)

python -u run_dlvo_spheres_metal.py > ${outprefix}.hoomd.log && touch ${outprefix}.done.txt

python -u render_frame.py ${outprefix}.run.${nsteps}