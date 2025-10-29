"""
Evaluation-related functions for test-time matching.
"""

import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset import get_eval_metrics

try:
    profile
except NameError:
    def profile(func): return func

def evaluate_model_global(similarity_scores):

    # please modify this function if you want to support non-square group shapes for global matching
    assert similarity_scores.shape[0] == similarity_scores.shape[1]

    num_data = similarity_scores.shape[0]

    labeled_caption_idx = similarity_scores.argmax(dim=-1)
    correct_labels = torch.arange(num_data) == labeled_caption_idx
    num_correct_labels= correct_labels.sum().item()
    global_pseudo_label_accuracy = num_correct_labels / num_data

    img_indices, caption_indices = linear_sum_assignment(similarity_scores.numpy(), maximize=True)
    correct_match = img_indices == caption_indices
    num_correct_match = sum(correct_match)
    global_match_accuracy = num_correct_match / num_data

    return global_pseudo_label_accuracy, global_match_accuracy


@profile
def evaluate_model(model, dataloader, group_shape, matching_method="group"):

    device = model.device

    results = get_eval_metrics(group_shape, matching_method=matching_method)
    
    M, N = group_shape[0], group_shape[1]

    model.eval()
    with torch.no_grad():
        for batch, num_groups_in_batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            similarity_scores = outputs.logits_per_image.cpu()

            for i in range(num_groups_in_batch):
                block = similarity_scores[i*M:i*M+M, i*N:i*N+N]

                if M==2 and N==2:
                    text_correct = (block[0, 0] > block[0, 1]).item() and \
                                (block[1, 1] > block[1, 0]).item()
                    image_correct = (block[0, 0] > block[1, 0]).item() and \
                                (block[1, 1] > block[0, 1]).item()
                    group_correct = text_correct and image_correct

                    individual_match_correct = [float((block[0, 0] > block[0, 1]).item()), float((block[1, 1] > block[1, 0]).item())] 
                    oracle_margin = (block[0, 0] + block[1, 1] - block[0, 1] - block[1, 0]).item()
                    group_match_correct = oracle_margin > 0

                    results["eval/text_score"].append(float(text_correct))
                    results["eval/image_score"].append(float(image_correct))
                    results["eval/group_score"].append(float(group_correct))
                    results["eval/individual_match_score"].extend(individual_match_correct)
                    results["eval/group_match_score"].append(float(group_match_correct))
                    
                elif M==1:
                    oracle_margin = (block[0, 0] - block[0, 1:].max()).item()
                    individual_match_correct = oracle_margin > 0

                    results["eval/individual_match_score"].append(float(individual_match_correct))
                    
                else:
                    raise ValueError(f"Unsupported group shape: {group_shape}")
        
        if matching_method == "global":
            # if global matching, we set the batch size to be the number of data in the dataset, so similarity_scores contains all the similarity scores
            global_pseudo_label_score , global_match_score = evaluate_model_global(similarity_scores)
            results["eval/global_pseudo_label_score"].append(global_pseudo_label_score)
            results["eval/global_match_score"].append(global_match_score)
    
    for k, v in results.items():
        results[k] = np.mean(v) * 100

    print("Evaluation results:", {k: f"{(v):.4f}" if isinstance(v, float) else v for k, v in results.items()})

    return results
