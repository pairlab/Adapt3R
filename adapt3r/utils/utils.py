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
import adapt3r.utils.pytorch3d_transforms as pt
import logging
from typing import Optional
import hashlib
import pickle
from hydra.utils import instantiate
from omegaconf import OmegaConf
import adapt3r.dataset.utils as data_utils


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
        f"{prefix}/{cfg.task.suite_name}/{cfg.task.benchmark_name}/{cfg.exp_name}"
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
    else:
        if not allow_overlap and not cfg.training.resume:
            assert not os.path.exists(experiment_dir), \
                f'cfg.make_unique_experiment_dir=false but {experiment_dir} is already occupied'

    experiment_name = "_".join(experiment_dir.split("/")[len(cfg.output_prefix.split('/')):])
    return experiment_dir, experiment_name

def make_dataset(cfg, stats_mode=False):
    dataset = instantiate(cfg.task.dataset, stats_mode=stats_mode)
    return dataset

def compute_norm_stats(cfg, policy):
    hash_input = {
        'dataset': OmegaConf.to_container(cfg.task.dataset, resolve=True),
        'eecf': cfg.algo.eecf,
        'abs_action': cfg.algo.abs_action,
    }
    hash = hash_dict(hash_input)
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f'{hash}.pkl'
    if os.path.exists(cache_file):
        print(f'Loading cached norm stats from {cache_file}')
        return pickle.load(open(cache_file, 'rb'))
    
    print(f'Computing norm stats and caching to {cache_file}')
    dataset = make_dataset(cfg, stats_mode=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=0)
    lowdim_keys = cfg.task.shape_meta.observation.lowdim.keys()
    data = {
        'actions': []
    }
    data.update({key: [] for key in lowdim_keys})
    for entry in tqdm(dataloader):
        entry = map_tensor_to_device(entry, 'cuda')
        entry = policy.preprocess_actions(entry)
        data['actions'].append(entry['actions'])
        for key in lowdim_keys:
            data[key].append(entry['obs'][key])
        
    for key in data:
        data[key] = torch.cat(data[key], dim=0)
    
    stats = {}
    for key in data:
        stats[key] = {
            'mean': data[key].mean(dim=0).cpu().numpy(),
            'std': data[key].std(dim=0).cpu().numpy(),
            'max': data[key].max(dim=0)[0].cpu().numpy(),
            'min': data[key].min(dim=0)[0].cpu().numpy(),
            'p1': torch.quantile(data[key], 0.01, dim=0).cpu().numpy(),
            'p99': torch.quantile(data[key], 0.99, dim=0).cpu().numpy()
        }
    pickle.dump(stats, open(cache_file, 'wb'))
    return stats

# def compute_norm_stats_old(
#         dataset, 
#                        normalize_action=True, 
#                        normalize_obs=False, 
#                        do_tqdm=True, 
#                        transform=None):
#     def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
#         """
#         Helper function to aggregate trajectory statistics.
#         See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#         for more information.
#         """
#         merged_stats = {}
#         for k in traj_stats_a:
#             n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
#             n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
#             n = n_a + n_b
#             mean = (n_a * avg_a + n_b * avg_b) / n
#             delta = (avg_b - avg_a)
#             M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n

#             merged_max = np.maximum(traj_stats_a[k]['max'], traj_stats_b[k]['max'])
#             merged_min = np.minimum(traj_stats_a[k]['min'], traj_stats_b[k]['min'])
#             merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2, max=merged_max, min=merged_min)
#         return merged_stats

#     merged_stats = {}
#     if normalize_action:
#         merged_stats_action = dataset.datasets[0].sequence_dataset.normalize_action()
#         merged_stats.update(merged_stats_action)
#     if normalize_obs:
#         merged_stats_obs = dataset.datasets[0].sequence_dataset.normalize_obs()
#         merged_stats.update(merged_stats_obs)
#     for sub_dataset in tqdm(dataset.datasets[1:], disable=not do_tqdm):
#         new_stats = {}
#         if normalize_action:
#             new_stats_action = sub_dataset.sequence_dataset.normalize_action()
#             new_stats.update(new_stats_action)
#         if normalize_obs:
#             new_stats_obs = sub_dataset.sequence_dataset.normalize_obs()
#             new_stats.update(new_stats_obs)
#         merged_stats = _aggregate_traj_stats(merged_stats, new_stats)

#     for k in merged_stats:
#         # note we add a small tolerance of 1e-3 for std
#         merged_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3

#     return merged_stats

def get_latest_checkpoint(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    onlyfiles = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f)) and 'pth' in f]
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
    return torch.load(path, weights_only=False)

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

def load_checkpoint(checkpoint_path, logger=None):
    """
    Load a checkpoint and return the state dictionary and checkpoint path.
    This function handles both automatic checkpoint detection and manual checkpoint loading.
    
    Args:
        checkpoint_path: Path to the checkpoint to load
        logger: Optional logger for logging checkpoint loading status
        
    Returns:
        state_dict: The loaded state dictionary or None if no checkpoint
    """
    try:
        checkpoint_path = get_latest_checkpoint(checkpoint_path)
        if logger:
            logger.info(f"Using provided checkpoint path: {checkpoint_path}")
        
        state_dict = load_state(checkpoint_path)
        if logger:
            logger.info(f"Successfully loaded checkpoint from: {checkpoint_path}")
        
        return state_dict
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        raise e

def setup_logger(log_file: str, experiment_dir: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration for a script.
    
    Args:
        log_file: Name of the log file (e.g. 'training.log', 'evaluate.log')
        experiment_dir: Optional directory to store experiment-specific logs
        
    Returns:
        Configured logger instance
    """
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Add experiment-specific file handler if directory provided
    if experiment_dir is not None:
        experiment_log_file = Path(experiment_dir) / log_file
        file_handler = logging.FileHandler(experiment_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger

def hash(obj):
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()

def hash_dict(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def get_cache_dir():
    """Get the cache directory path.
    
    Returns:
        Path: Path to the cache directory in the main script directory
    """
    script_dir = Path(__file__).resolve().parent.parent.parent
    cache_dir = script_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir
