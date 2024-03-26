# DLVO-WORKFLOW: Molecular Dynamics of DLVO Ions

DLVO-Workflow runs MD simulations of DLVO ions using the HOOMD Blue MD engine. 
Workflow takes input from user and generates input files for various simulations
having different parameters which results in different crystalization behavior.
After simulations are over the predicted trajectories are analyzed to determine clustering.

The workflow is orchestrated with the
[Parsl parallel scripting library](https://parsl-project.org/) 

Key parameters editable in the xml form include the list below.

## Parameter Options:

	lattice_repeats: Times to replicate the system in each direction
	lattice_spacing: Lattice spacing in terms of positive particle diameter 	
	fraction_positive: Fraction positive charge 
	radiusN: Radius of negative colloids 
	radiusP: Radius of positive colloids
    debye_length: Debye length 
    seed: Random seed 
    yaml_file: template yaml file
	output_dir: parent location of simulation outputs
	nsteps: number of steps for simulation

## Contents

+ **`./thumb`:** contains thumbnail for workflow.
+ **`./parsl_utils`:** contains scripts necessay to set up local and remote environments.
+ **`./inputs`:** contains a general input file in yaml format which will be edited and copied to corresponding running directories.
+ **`./utils`:** contains input preparation code and core functionality code that runs MD simulations.
+ **`./requirements`:** contains yaml files specifying packages needed by this workflow.  These dependencies are installed automatically by `parsl_utils`.
+ **`parsl_wrapper.sh`:** bash script to run scripts in parsl_utils.
+ **`main.py`:** is the the workflow code submitting and managing jobs.
+ **`workflow.xml`:** is the file that takes user input in xml format for the workflow