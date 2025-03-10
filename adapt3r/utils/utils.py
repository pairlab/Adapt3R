import copy
import json
import os
import random
from pathlib import Path
import adapt3r.utils.tensor_utils as TensorUtils
import numpy as np
import torch
import torch.nn as nn
import warnings
from natsort import natsorted
from tqdm import tqdm

def get_experiment_dir(cfg, evaluate=False, allow_overlap=False):
    # if eval_flag:
    #     prefix = "evaluations"
    # else:
    #     prefix = "experiments"
    #     if cfg.pretrain_model_path != "":
    #         prefix += "_finetune"

    prefix = cfg.output_prefix
    if evaluate:
        prefix = os.path.join(prefix, 'evaluate')

    experiment_dir = (
        f"{prefix}/{cfg.task.suite_name}/{cfg.task.benchmark_name}/"
        + f"{cfg.algo.name}/{cfg.exp_name}"
    )
    if cfg.variant_name is not None:
        experiment_dir += f'/{cfg.variant_name}'
    
    if cfg.seed != 10000:
        experiment_dir += f'/{cfg.seed}'

    if cfg.make_unique_experiment_dir:
        # look for the most recent run
        experiment_id = 0
        if os.path.exists(experiment_dir):
            for path in Path(experiment_dir).glob("run_*"):
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split("run_")[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    pass
            experiment_id += 1

        experiment_dir += f"/run_{experiment_id:03d}"
        
        if not allow_overlap and not cfg.training.resume:
            assert not os.path.exists(experiment_dir), \
                f'cfg.make_unique_experiment_dir=false but {experiment_dir} is already occupied'

    experiment_name = "_".join(experiment_dir.split("/")[len(cfg.output_prefix.split('/')):])
    return experiment_dir, experiment_name

def compute_norm_stats(dataset, normalize_action=True, normalize_obs=True, do_tqdm=True):
    def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
        """
        Helper function to aggregate trajectory statistics.
        See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        for more information.
        """
        merged_stats = {}
        for k in traj_stats_a:
            n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
            n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
            n = n_a + n_b
            mean = (n_a * avg_a + n_b * avg_b) / n
            delta = (avg_b - avg_a)
            M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n

            merged_max = np.maximum(traj_stats_a[k]['max'], traj_stats_b[k]['max'])
            merged_min = np.minimum(traj_stats_a[k]['min'], traj_stats_b[k]['min'])
            merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2, max=merged_max, min=merged_min)
        return merged_stats

    merged_stats = {}
    if normalize_action:
        merged_stats_action = dataset.datasets[0].sequence_dataset.normalize_action()
        merged_stats.update(merged_stats_action)
    if normalize_obs:
        merged_stats_obs = dataset.datasets[0].sequence_dataset.normalize_obs()
        merged_stats.update(merged_stats_obs)
    for sub_dataset in tqdm(dataset.datasets[1:], disable=not do_tqdm):
        new_stats = {}
        if normalize_action:
            new_stats_action = dataset.datasets[0].sequence_dataset.normalize_action()
            new_stats.update(new_stats_action)
        if normalize_obs:
            new_stats_obs = sub_dataset.sequence_dataset.normalize_obs()
            new_stats.update(new_stats_obs)
        merged_stats = _aggregate_traj_stats(merged_stats, new_stats)

    for k in merged_stats:
        # note we add a small tolerance of 1e-3 for std
        merged_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3

    return merged_stats

def get_latest_checkpoint(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    onlyfiles = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
    onlyfiles = natsorted(onlyfiles)
    best_file = onlyfiles[-1]
    return os.path.join(checkpoint_dir, best_file)

def soft_load_state_dict(model, loaded_state_dict):
    # loaded_state_dict['task_encoder.weight'] = loaded_state_dict['task_encodings.weight']
    
    current_model_dict = model.state_dict()
    new_state_dict = {}

    for k in current_model_dict.keys():
        if k in loaded_state_dict:
            v = loaded_state_dict[k]
            if not hasattr(v, 'size') or v.size() == current_model_dict[k].size():
                new_state_dict[k] = v
            else:
                warnings.warn(f'Cannot load checkpoint parameter {k} with shape {loaded_state_dict[k].shape}'
                            f'into model with corresponding parameter shape {current_model_dict[k].shape}. Skipping')
                new_state_dict[k] = current_model_dict[k]
        else:
            new_state_dict[k] = current_model_dict[k]
            warnings.warn(f'Model parameter {k} does not exist in checkpoint. Skipping')
    for k in loaded_state_dict.keys():
        if k not in current_model_dict:
            warnings.warn(f'Loaded checkpoint parameter {k} does not exist in model. Skipping')
    
    model.load_state_dict(new_state_dict)

def map_tensor_to_device(data, device):
    """Move data to the device specified by device."""
    return TensorUtils.map_tensor(
        data, lambda x: safe_device(x, device=device)
    )

def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

def extract_state_dicts(inp):

    if not (isinstance(inp, dict) or isinstance(inp, list)):
        if hasattr(inp, 'state_dict'):
            return inp.state_dict()
        else:
            return inp
    elif isinstance(inp, list):
        out_list = []
        for value in inp:
            out_list.append(extract_state_dicts(value))
        return out_list
    else:
        out_dict = {}
        for key, value in inp.items():
            out_dict[key] = extract_state_dicts(value)
        return out_dict
        
def save_state(state_dict, path):
    save_dict = extract_state_dicts(state_dict)
    torch.save(save_dict, path)

def load_state(path):
    return torch.load(path)

def torch_save_model(model, optimizer, scheduler, model_path, cfg=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg,
        },
        model_path,
    )

def torch_load_model(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["scheduler_state_dict"], checkpoint["cfg"]

def recursive_update(base_dict, update_dict):
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict (dict): The dictionary to update.
        update_dict (dict): The dictionary containing updates.
        
    Returns:
        dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

