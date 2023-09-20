
def combine_arguments_dicts(input_dict, default_arguments):
    import itertools
    list_argument_dicts = []
    non_empty_keys = [ key for key in input_dict.keys() if input_dict[key] is not None and key in default_arguments]

    list_keys = [key for key in non_empty_keys if type(input_dict[key])==type([])]
    single_value_keys = [key for key in non_empty_keys if type(input_dict[key])!=type([])]

    list_key_values = [ input_dict[key] for key in non_empty_keys if type(input_dict[key])==type([])]
    all_combo_list = list(itertools.product(*list_key_values))

    for values in all_combo_list:
        param_dict = { list_keys[vidx]: values[vidx] for vidx in range(len(values)) }
        for key in single_value_keys:
            param_dict[key] = input_dict[key]
        list_argument_dicts.append(param_dict)
    return list_argument_dicts

def update_yaml(arg_dict, only_updated=False, new_yaml="inputs_new.yaml"):
    import yaml
    import os
    import copy
    from path import Path
    with open(arg_dict['yaml_file']) as f:
        default_arguments = yaml.load(f, Loader=yaml.FullLoader)

    list_argument_dicts = combine_arguments_dicts(arg_dict, default_arguments)
    run_list = []
    all_params = []
    #do this to only use the ones being looped over:
    if only_updated is True:
        sweep_keys = sorted(list_argument_dicts[0].keys())
    else:
        sweep_keys = sorted([key for key in arg_dict.keys() if type(arg_dict[key]) is list or arg_dict[key] is None])

    sweep_path = ""
    for key in sweep_keys:
        sweep_path = os.path.join(sweep_path,key+"{"+f"{key}"+"}")

    for param_dict in list_argument_dicts:
        new_params = copy.copy(default_arguments)
        new_params.update(param_dict)
        new_params['outputprefix'] = new_params['outputprefix'].format_map(new_params)
        run_file = os.path.join(arg_dict['output_dir'],sweep_path.format_map(new_params),new_yaml)
        run_file_dir = os.path.dirname(run_file)
        outputdir = os.path.join('.',run_file_dir)
        # print(F"YS run_file_dir: {run_file_dir} \n output_directory: {outputdir} \n path output dir: {Path(outputdir)}")
        output = Path(outputdir)
        os.makedirs(output, exist_ok=True)
        # print(f"YS output: {output}")
        run_list.append(run_file)
        # os.makedirs(os.path.dirname(run_file),exist_ok=True)
        with open(run_file,'w') as fh:
            yaml.dump(new_params,fh, default_flow_style=False)
        all_params.append(new_params)
    return [run_list, all_params]
