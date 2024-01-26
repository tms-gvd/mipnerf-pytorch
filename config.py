import argparse
import yaml
import torch
import os

class DatasetName:
    def __init__(self, name) -> None:
        self.name = name.lower()
    
    def get_path(self, data_dir):
        if self.name == "blender_lego":
            return os.path.join(os.path.abspath(data_dir), "nerf_synthetic", "lego")
        else:
            raise NotImplementedError("LLFF dataset is not supported yet")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def update_params(params, args):
    def _nested_params(params, key, value, base_key):
        if "." in key:
            first, rest = key.split(".", 1)
            _nested_params(params[first], rest, value, base_key)
        else:
            if key not in params.keys():
                print(f"WARNING: {base_key} is not in the config file.")
                params[key] = value
            else:
                print(f"Overwriting {base_key} with {value}")
                params[key] = type(params[key])(value)

    for arg in args.opts:
        key, value = arg.split("=")
        _nested_params(params, key, value, key)


def get_config():
    # load config file and update parameters from command line
    print("LOADING CONFIG FILE")
    args = get_parser()
    with open(args.config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    update_params(config_dict, args)
    
    # convert config_dict to config namespace
    config = argparse.Namespace(**config_dict)        

    # default configs for llff, automatically set if dataset is llff and not override_defaults
    if config.dataset_name == "llff" and config.use_defaults:
        print("Using default parameters for LLFF dataset")
        config.factor = 4
        config.ray_shape = "cylinder"
        config.white_bkgd = False
        config.density_noise = 1.0
    elif config.dataset_name == "blender" and config.use_defaults:
        print("Using default parameters for Blender dataset")
        config.factor = 2
        config.ray_shape = "cone"
        config.white_bkgd = False
        config.density_noise = 0.0
    else:
        raise NotImplementedError(
            f"Invalid value for dataset_name {config.dataset_name}. Now support either 'llff' or 'blender'"
        )

    # set device
    # config.device = torch.device(config.device)

    # set base directory for dataset
    base_data_path = "data/nerf_llff_data/"
    if config.dataset_name == "blender":
        base_data_path = "data/nerf_synthetic/"
    elif config.dataset_name == "multicam":
        base_data_path = "data/nerf_multiscale/"
    config.base_dir = os.path.join(base_data_path, config.scene)

    # set log directory and create if not exist
    config.log_dir = os.path.join(config.log_dir, config.exp_name)
    if os.path.exists(config.log_dir) and not config.exp_name == "default":
        raise ValueError(
            f"Log directory {config.log_path} already exists. Please use a experiment name."
        )
    else:
        os.makedirs(config.log_dir, exist_ok=True)
    
    # save config file
    with open(os.path.join(config.log_dir, "config.yaml"), "w") as file:
        yaml.dump(vars(config), file, default_flow_style=False)
    
    print()
    
    return config
