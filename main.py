"""
Main script for running test-time matching.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
import argparse
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb
import pickle

from dataset import get_grouped_dataset, get_eval_metrics, collate_fn_group
from model import download_and_cache_model, get_image_transform
from match import group_match, get_matched_dataset, global_match
from evaluate import evaluate_model
from train import train_model
from utils import DATASET_CACHE_DIR, MODEL_CACHE_DIR, generate_wandb_names, get_device, set_random_seed, get_lr_schedule_lambda, get_threshold_schedule, get_save_results_path

try:
    profile
except NameError:
    def profile(func): return func

def get_model_name(model_short_name):
    model_dict = {
        'siglip-b16-224': 'google/siglip-base-patch16-224',
        'siglip-l16-256': 'google/siglip-large-patch16-256',
        'clip-b32': 'openai/clip-vit-base-patch32',
        'clip-b16': 'openai/clip-vit-base-patch16',
    }
    return model_dict[model_short_name]

def get_dataset_name(dataset_short_name):
    dataset_dict = {
        'winoground': 'facebook/winoground',
        'mmvp_vlm': 'mmvp_vlm',
        'colorswap': 'stanfordnlp/colorswap',
        'sugarcrepe_replace_rel': 'sugarcrepe/replace_rel',
        'sugarcrepe_swap_att': 'sugarcrepe/swap_att',
        'sugarcrepe_swap_obj': 'sugarcrepe/swap_obj',
        'sugarcrepe_add_att': 'sugarcrepe/add_att',
        'whatsup_a_1x4': 'whatsup_a_1x4',
        'whatsup_b_1x4': 'whatsup_b_1x4',
        'whatsup_a_left_right': 'whatsup_a_left_right',
        'whatsup_a_on_under': 'whatsup_a_on_under',
        'whatsup_b_left_right': 'whatsup_b_left_right',
        'whatsup_b_front_behind': 'whatsup_b_front_behind',
    }
    if dataset_short_name in dataset_dict.keys():
        return dataset_dict[dataset_short_name]
    else:
        return dataset_short_name

def create_parser():
    parser = argparse.ArgumentParser(description="Test-Time Matching")

    # Device and model/dataset selection
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device ID to use')
    parser.add_argument('--model', type=str, default='siglip-b16-224',
                        choices=['siglip-b16-224', 'siglip-l16-256', 'clip-b32', 'clip-b16'],
                        help='CLIP model to use for training and evaluation')
    parser.add_argument('--dataset', type=str, default='colorswap',
                        choices=['winoground', 'mmvp_vlm', 'colorswap', 'sugarcrepe_replace_rel', 'sugarcrepe_swap_att', 'sugarcrepe_swap_obj', 'sugarcrepe_add_att', 'whatsup_a_1x4', 'whatsup_b_1x4', 'whatsup_a_left_right', 'whatsup_a_on_under', 'whatsup_b_left_right', 'whatsup_b_front_behind'],
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--tag', type=str, default='test-run', help='Tag for Weights & Biases')
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1], help='Save results to a file')

    # Training and evaluation parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs per iteration')
    parser.add_argument('--iterations', type=int, default=10, help='Number of match-train iterations')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Evaluation batch size')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval inside training loop')
    parser.add_argument('--batch_size', type=int, default=50, help='Training batch size')
    parser.add_argument('--no_shuffle', action='store_true', help='Do not shuffle training data')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for data loading')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch factor for data loading')
    parser.add_argument('--train_augmentation', type=int, default=0, choices=[0, 1], help='0 means no augmentation, 1 means augmentation')
    parser.add_argument('--augmentation_resize_factor', type=float, default=1.1, help='Resize factor >= 1.0 for augmentation (before cropping)')

    # Optimizer and scheduler options
    parser.add_argument('--lr', type=float, default=4e-5, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'constant'],
                        help='Learning rate schedule type: linear decay, cosine annealing, or constant')
    parser.add_argument('--min_lr_factor', type=float, default=0.1, help='Minimum learning rate factor')
    parser.add_argument('--lr_restart', type=float, default=0.95, help='Restart learning rate with this factor from the last iteration, values within (0,1] means restart; -1 means no restart.')
    parser.add_argument('--keep_opt_states', type=int, default=0, choices=[0, 1], help='Keep optimizer states across iterations after lr_restart')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for optimizer')

    # Matching method selection
    parser.add_argument('--matching_method', type=str, default='group', choices=['group', 'global'], help='Matching method: group (group-based) or global (non-grouped matching)')
    parser.add_argument('--threshold_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'constant'], help='Threshold decaying schedule for filtering matches')
    parser.add_argument('--threshold_start', type=float, default=1.0, help='Start threshold for linear decay')
    parser.add_argument('--threshold_end', type=float, default=0.0, help='End threshold for linear decay')
    parser.add_argument('--oracle_matching', action='store_true', help='Use oracle matching')
    parser.add_argument('--use_ratio_threshold', type=int, default=1, choices=[0, 1], help='Use ratio threshold for global matching')

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.save_results:
        print(f"Saving results to {get_save_results_path(args)[0]}")
    
    set_random_seed(args.random_seed)
    device = get_device(args.cuda_device)
    
    # Model and dataset configuration
    model_name = get_model_name(args.model)
    dataset_name = get_dataset_name(args.dataset)

    tags, run_name = generate_wandb_names(args)
    config = vars(args)

    # Initialize wandb if requested
    use_wandb = args.use_wandb
    if use_wandb:
        # Initialize wandb
        wandb.init(
            project="test-time-matching",
            name=run_name,
            config=config,
            tags=tags
        )

    print(f"==== TTM with model {model_name} and dataset {dataset_name} on device {device} ====")
    print(f"Initialized wandb run: {run_name}")
    print(f"Tags: {tags}")
    print(f"Config: {config}")

    # Record start time
    training_start_time = time.time()
    training_start_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    print(f"Experiment started at: {training_start_pt.strftime('%Y-%m-%d %H:%M:%S PT')}")
    
    # Download and load model and processor
    print("==== Loading Model ====")
    model, processor = download_and_cache_model(model_name, MODEL_CACHE_DIR)
    model.to(device)
    
    # Download and load dataset
    print("==== Loading Dataset ====")
    eval_transform = get_image_transform(model_name, if_augment=False)
    if_augment = True if args.train_augmentation == 1 else False
    train_transform = get_image_transform(model_name, if_augment=if_augment, augmentation_resize_factor=args.augmentation_resize_factor)

    dataset = get_grouped_dataset(dataset_name, DATASET_CACHE_DIR, eval_transform)
    tokenizer = processor.tokenizer

    group_shape = dataset.group_shape
    num_groups = dataset.num_groups

    eval_batch_size = args.eval_batch_size
    train_batch_size = args.batch_size
    train_num_groups = num_groups

    if args.matching_method == 'global':
        eval_batch_size = len(dataset)
        train_batch_size = group_shape[0] * args.batch_size
        train_num_groups = group_shape[0] * num_groups
        # for global matching, we set the batch size to be the number of data in the dataset for simplicity. 
        # If memory is not enough, one can change this function to properly load data in smaller batches

    if train_batch_size < 1 or eval_batch_size < 1:
        raise ValueError(f"train_batch_size or eval_batch_size is less than 1: train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}")
        
    print(f"Using {args.matching_method} matching, eval batch size: {eval_batch_size}, train batch size: {train_batch_size}, train num groups: {train_num_groups}")

    eval_dataloader = DataLoader(
        dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=lambda x: collate_fn_group(x, tokenizer), 
        pin_memory=True, 
        prefetch_factor=args.prefetch_factor)

    all_results = get_eval_metrics(group_shape, matching_method=args.matching_method)
    all_train_logs = {}

    eval_results = evaluate_model(model, eval_dataloader, group_shape, matching_method=args.matching_method)

    for k, v in eval_results.items():
        if k in all_results:
            all_results[k].append(v)
    
    print(f"Initial evaluation results: {all_results}")

    if use_wandb:
        eval_results["others/epoch"] = 0
        eval_results["others/iteration"] = 0
        eval_results["others/global_epoch"] = 0
        print(f"Initial evaluation results logged to wandb: {eval_results}")
        wandb.log(eval_results)

    lr_restart = False if args.lr_restart == -1 else True
    keep_opt_states = False if args.keep_opt_states == 0 else True
    lr = args.lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=args.betas)
    # Setup initial optimizer and learning rate scheduler (if not restarting per iteration)
    if not lr_restart:
        total_steps = args.iterations * args.epochs * ((train_num_groups + train_batch_size - 1) // train_batch_size)
        # The total number of steps is an overestimate since we don't know the number of matched pairs in advance
        total_steps = max(total_steps, 1)
        print(f"with lr_restart={lr_restart}, total number of optimization steps: {total_steps}")
        lr_scheduler = LambdaLR(optimizer, get_lr_schedule_lambda(total_steps, args.lr_schedule, args.min_lr_factor))
        print("Learning without restarting learning rate across iterations")
    else:
        lr_scheduler = None
        print("Will restart learning rate for each iteration")
    
    print(f"==== Starting {args.iterations} Test-Time Matching Iterations ====")
    
    # Main test-time matching loop
    for iteration in range(args.iterations):
        print(f"\n{'='*20} Iteration {iteration + 1}/{args.iterations} {'='*20}")
        
        # Step 1: Perform matching using selected method
        print(f"Step 1: Performing {args.matching_method} matching...")

        if args.matching_method == 'group':

            match_results, selected_indices = group_match(
                model=model,
                dataloader=eval_dataloader,
                group_shape=group_shape,
                num_groups=num_groups,
                threshold=get_threshold_schedule(iteration, args),
                oracle_match=args.oracle_matching)

        elif args.matching_method == 'global':

            use_ratio_threshold = True if args.use_ratio_threshold == 1 else False

            match_results, selected_indices = global_match(
                model=model,
                dataloader=eval_dataloader,
                group_shape=group_shape,
                threshold=get_threshold_schedule(iteration, args),
                use_ratio_threshold=use_ratio_threshold)
        else:
            raise ValueError(f"Invalid matching method: {args.matching_method}")
        
        for k, v in match_results.items():
            if k not in all_results:
                all_results[k] = []
            all_results[k].append(v)

        if use_wandb:
            match_results["others/iteration"] = iteration + 1
            match_results["others/global_epoch"] = iteration * args.epochs
            wandb.log(match_results)
        
        
        # Step 2: Train on matched pairs
        if len(selected_indices) == 0:
            print(f"No matched pairs, skipping training for iteration {iteration + 1}")
            continue
        else:
            print(f"Step 2: Training on {len(selected_indices)} matched groups...")
            matched_dataset = get_matched_dataset(dataset, selected_indices, train_transform, args.matching_method)

            # Restart scheduler and optimizer states if requested
            if lr_restart:
                lr = args.lr * (args.lr_restart ** iteration)
                if keep_opt_states:
                    print(f"Keeping optimizer states across iterations with restart learning rate: {lr}")
                    for g in optimizer.param_groups:
                        g['lr'] = lr
                        g['initial_lr'] = lr
                    
                else:
                    print(f"Creating fresh optimizer and lr_scheduler with restart learning rate: {lr}")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=args.betas)

                # For fresh restart, calculate steps per iteration
                steps_per_iteration = args.epochs * ((len(matched_dataset) + train_batch_size - 1) // train_batch_size)
                lr_scheduler = LambdaLR(optimizer, get_lr_schedule_lambda(steps_per_iteration, args.lr_schedule, args.min_lr_factor))
                print(f"with lr_restart={lr_restart}, steps per iteration: {steps_per_iteration}, optimizer.param_groups[0]['lr'] = {optimizer.param_groups[0]['lr']}")
            
            g = torch.Generator()
            g.manual_seed(args.random_seed + iteration)

            if_shuffle = False if args.no_shuffle else True
            train_dataloader = DataLoader(
                matched_dataset,
                batch_size=train_batch_size,
                shuffle=if_shuffle,
                num_workers=args.num_workers,
                collate_fn=lambda x: collate_fn_group(x, tokenizer),
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                generator=g,
            )

            model, eval_results, iteration_train_logs = train_model(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                eval_dataloader=eval_dataloader,
                group_shape=group_shape,
                args=args,
                iteration_id=iteration,
            )

            all_train_logs[iteration] = iteration_train_logs

            for k, v in eval_results.items():
                if len(v) > 0:
                    all_results[k].append(v[-1])

    
    # Record training end time and calculate duration
    training_end_time = time.time()
    training_end_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    training_duration_seconds = training_end_time - training_start_time
    training_duration_minutes = training_duration_seconds / (60)
    
    print(f"Experiment ended at: {training_end_pt.strftime('%Y-%m-%d %H:%M:%S PT')}")

    if args.save_results:
        # Add config and other metadata to results
        results_to_save = {
            'results': all_results,
            'train_logs': all_train_logs,
            'config': config,
            'training_duration_minutes': training_duration_minutes,
            'training_start_time': training_start_pt.isoformat(),
            'training_end_time': training_end_pt.isoformat()
        }

        print(f"results_to_save: {results_to_save}")

        folder_path, file_name = get_save_results_path(args)
        os.makedirs(folder_path, exist_ok=True)
        file_path = f"{folder_path}/{file_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(results_to_save, f)
    
    # Final results summary
    print(f"\n{'='*60}")
    print(f"                 FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Training time   - Started: {training_start_pt.strftime('%Y-%m-%d %H:%M:%S PT')}")
    print(f"                - Ended:   {training_end_pt.strftime('%Y-%m-%d %H:%M:%S PT')}")
    print(f"Training duration: {training_duration_minutes:.2f} minutes")
    print(f"")
    print(f"Initial evaluation results:", {key: f"{value[0]:.4f}" for key, value in all_results.items() if "eval" in key})
    print(f"")
    print(f"Final evaluation results:", {key: f"{value[-1]:.4f}" for key, value in all_results.items() if "eval" in key})
    print(f"Final improvement results:", {key: f"{(value[-1] - value[0]):.4f}" for key, value in all_results.items() if "eval" in key}) 
    print(f"")
    print(f"Best evaluation results:", {key: f"{max(value):.4f}" for key, value in all_results.items() if "eval" in key}) 
    print(f"Best improvement results:", {key: f"{(max(value) - value[0]):.4f}" for key, value in all_results.items() if "eval" in key})
    print(f"{'='*60}")
        
    if use_wandb:
        wandb.summary["training_duration_minutes"] = training_duration_minutes
        if optimizer is not None:
            wandb.summary["train/final_learning_rate"] = optimizer.param_groups[0]['lr']
        print("Results logged to wandb dashboard")
        try:
            wandb.finish()
        except FileNotFoundError as e:
            print(f"Warning: wandb cleanup failed: {e}")

        
if __name__ == '__main__':
    main()
