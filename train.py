"""
Training-related functions for test-time matching.
"""

from tqdm import tqdm
import torch
import wandb
from collections import defaultdict

from evaluate import evaluate_model
from dataset import get_eval_metrics

try:
    profile
except NameError:
    def profile(func): return func

@profile
def train_model(model, train_dataloader, optimizer, lr_scheduler, eval_dataloader, group_shape, args, iteration_id=None):

    results = get_eval_metrics(group_shape, matching_method=args.matching_method)

    train_logs = defaultdict(list)

    num_epochs = args.epochs
    eval_interval = args.eval_interval
    use_wandb = args.use_wandb

    print(f"Training model for {num_epochs} epochs...")

    device = model.device
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        
        model.train()
        total_epoch_loss = torch.tensor(0.0, device=device)
        total_grad_norm = torch.tensor(0.0, device=device)
        num_batches = 0
        for batch, _ in train_dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                total_epoch_loss += loss.detach()
                total_grad_norm += grad_norm
            num_batches += 1

        avg_epoch_loss = (total_epoch_loss / num_batches).item()
        avg_grad_norm = (total_grad_norm / num_batches).item()
        current_lr = optimizer.param_groups[0]['lr']

        train_logs_epoch = {
            "others/epoch": epoch + 1,
            "others/iteration": iteration_id + 1,
            "others/global_epoch": iteration_id * num_epochs + epoch + 1,
            "train/learning_rate": current_lr,
            "train/avg_epoch_loss": avg_epoch_loss,
            "train/avg_grad_norm": avg_grad_norm,
        }
        for k, v in train_logs_epoch.items():
            train_logs[k].append(v)

        if use_wandb:
            wandb.log(train_logs_epoch)
        
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            # Evaluate after each epoch
            model.eval()
            eval_results = evaluate_model(model, eval_dataloader, group_shape, matching_method=args.matching_method)
            for k, v in eval_results.items():
                results[k].append(v)
            eval_results["others/epoch"] = epoch + 1
            eval_results["others/iteration"] = iteration_id + 1
            eval_results["others/global_epoch"] = iteration_id * num_epochs + epoch + 1
            if use_wandb:
                wandb.log(eval_results)

    print("Evaluation results after training:", {k: f"{(v):.4f}" if isinstance(v, float) else v for k, v in results.items()})
            
    return model, results, train_logs
