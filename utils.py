"""
Utility functions for test-time matching.
"""

import torch
import numpy as np
import yaml
import random
from zoneinfo import ZoneInfo
from datetime import datetime

with open("path_config.yaml", "r") as f:
    path_config = yaml.safe_load(f)
    MODEL_CACHE_DIR = path_config["model_cache_dir"]
    DATASET_CACHE_DIR = path_config["dataset_cache_dir"]
    RESULTS_DIR = path_config["results_dir"]

def get_lr_schedule_lambda(total_steps, schedule_type="linear", min_lr_factor=0.1):
    """Create learning rate schedule lambda function."""
    def lr_lambda_batch(current_step):
        if schedule_type == "linear":
            # Linear decay from 1.0 to min_lr_factor
            return max(1.0 - float(current_step) / float(total_steps), min_lr_factor)
        elif schedule_type == "cosine":
            # Cosine annealing: starts at 1.0, decays to min_lr_factor following cosine curve
            progress = min(float(current_step) / float(total_steps), 1.0)
            return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + np.cos(np.pi * progress))
        elif schedule_type == "constant":
            # Constant learning rate
            return 1.0
        else:
            raise ValueError(f"Unknown learning rate schedule: {schedule_type}")

    return lr_lambda_batch

def get_threshold_schedule(current_step, args):
    """Create a threshold for various decay schedules."""
    if args.threshold_schedule == "linear":
        thresholds = np.linspace(args.threshold_start, args.threshold_end, args.iterations)
        return thresholds[current_step]
    elif args.threshold_schedule == "cosine":
        grid = np.linspace(0, np.pi, args.iterations)
        return args.threshold_end + 0.5 * (args.threshold_start - args.threshold_end) * (1 + np.cos(grid[current_step]))
    elif args.threshold_schedule == "constant":
        return args.threshold_start
    else:
        raise ValueError(f"Unknown threshold_schedule: {args.threshold_schedule}")

def get_device(device_id=None):
    if device_id is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA device: {device_id}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Set random seed to {seed}")

def generate_wandb_tags(args):
    """Generate a wandb tags with model and dataset."""

    if args.matching_method == 'group':
        name_parts = [args.model, args.dataset, args.tag]
    elif args.matching_method == 'global':
        name_parts = [args.model, args.dataset, args.matching_method, args.tag]
    else:
        raise ValueError(f"Invalid matching method: {args.matching_method}")

    return ["-".join(name_parts)]

def generate_wandb_run_name(args):
    """Generate a descriptive wandb run name with method, parameters, and timestamp."""
    threshold_start = args.threshold_start
    threshold_end = args.threshold_end
    epochs = args.epochs
    iterations = args.iterations
    
    name_parts = []

    name_parts.append(f"seed{args.random_seed}")

    if args.oracle_matching:
        name_parts.append("oracle")
    
    schedule = f"th{threshold_start:.2f}-{threshold_end:.2f}-{args.threshold_schedule}"
    name_parts.append(schedule)

    name_parts.append(f"lr{args.lr}-{args.lr_schedule}")
    if args.lr_restart != -1:
        name_parts.append(f"lr_restart{args.lr_restart}")
    if args.keep_opt_states:
        name_parts.append("keep_opt_states")
    name_parts.append(f"min_lr_factor{args.min_lr_factor}")

    name_parts.append(f"wd{args.weight_decay}")

    name_parts.append(f"betas{args.betas}")
    
    name_parts.append(f"bs{args.batch_size}")

    name_parts.append(f"train_aug{args.train_augmentation}")

    if args.train_augmentation == 1:
        name_parts.append(f"aug_resize{args.augmentation_resize_factor}")

    name_parts.append(f"{iterations}x{epochs}ep")
    
    if args.no_shuffle:
        name_parts.append("no_shuffle")
    else:
        name_parts.append("shuffle")
    
    # Add timestamp in PT time (YYYYMMDD-HHMM format)
    pt_time = datetime.now(ZoneInfo("America/Los_Angeles"))
    name_parts.append(pt_time.strftime("%Y%m%d-%H%M"))
    
    return "-".join(name_parts)

def generate_wandb_names(args):
    """Generate both project and run names for wandb."""
    tags = generate_wandb_tags(args)
    run_name = generate_wandb_run_name(args)
    return tags, run_name

def get_save_results_path(args):

    tags, run_name = generate_wandb_names(args)
    folder_path = f"{RESULTS_DIR}/{tags[0]}"
    file_name = run_name
    
    return folder_path, file_name
