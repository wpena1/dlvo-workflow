
To run: 
        python3 -u run_parameter_sweep_dlvo.py [OPTIONS]

Parameter Options:


        lattice_repeats:
                command line name: '-a'
                nargs="*"
                default=None
                type=int

        lattice_spacing:
                command line name: '-LS'
                nargs="*"
                default=None
                type=float
	
    	fraction_positive:
		command line name:'-FP'
		nargs="*" 
		default=None
		type=float

	radiusN:
		command line name: '-RN'
    		nargs="*"
		default=None
 		type=float
	
	radiusP:
    		command line name: '-RP'
		nargs="*"
		default=None
		dest="radiusP"
		type=float

        debye_length:
                command line: '-DL'
                 nargs="*"
                 default=None
                 type=float
	
    	seed:
		command line name: '-seed'
		nargs="*"
		default=None
		type=int

    	yaml_file: 
		command line name: '-yaml_file'
		default='input_general.yaml'

	output_dir:
		command line name: '-output_dir'
		default='binary_dlvo' 
