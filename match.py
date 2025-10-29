"""
Matching-related functions for test-time matching.
"""

import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset import GroupedDataset 

try:
    profile
except NameError:
    def profile(func): return func

def get_matched_dataset(original_dataset, selected_indices, transform, matching_method='group'):

    if matching_method == 'group':
        return get_group_matched_dataset(original_dataset, selected_indices, transform)
    elif matching_method == 'global':
        return get_global_matched_dataset(original_dataset, selected_indices, transform)
    else:
        raise ValueError(f"Invalid matching method: {matching_method}")

def get_group_matched_dataset(original_dataset, selected_indices, transform):

    original_images = original_dataset.images
    original_captions = original_dataset.captions

    matched_images = []
    matched_captions = []

    for group_idx, image_indices, caption_indices in selected_indices:
        selected_image = tuple(original_images[group_idx][idx] for idx in image_indices)
        selected_caption = tuple(original_captions[group_idx][idx] for idx in caption_indices)
        matched_images.append(selected_image)
        matched_captions.append(selected_caption)
    
    assert len(matched_images) == len(matched_captions)

    matched_dataset = GroupedDataset(matched_images, matched_captions, transform, original_dataset.dataset_name)

    return matched_dataset

def get_global_matched_dataset(original_dataset, selected_indices, transform):
    original_images = original_dataset.images
    original_captions = original_dataset.captions

    non_grouped_images = [image for image_group in original_images for image in image_group]
    non_grouped_captions = [caption for caption_group in original_captions for caption in caption_group]

    matched_images = []
    matched_captions = []

    for i in range(len(selected_indices)):
        matched_images.append( (non_grouped_images[selected_indices[i][0]],) )
        matched_captions.append( (non_grouped_captions[selected_indices[i][1]],) )
        # we keep the image and caption as a tuple to match the format of GroupedDataset
    
    matched_dataset = GroupedDataset(matched_images, matched_captions, transform, original_dataset.dataset_name)

    return matched_dataset

@profile
def group_match(model, dataloader, group_shape, num_groups, threshold=0, oracle_match=False):

    device = model.device

    if oracle_match:
        print("=== Using oracle matching ===")
    else:
        print(f"=== Using threshold {threshold} for matching ===")
    selected_indices = []
    """
    selected indices of the form (k, (i_1, i_2), (c_1, c_2)). 
    k is the group index, (i_1, i_2) are the indices of the images, and (c_1, c_2) are the corresponding caption indices. 
    For example, (0, (0, 1), (0, 1)) means the first group, the first image and the second image, and the first caption (matched with the first image) and the second caption (matched with the second image).
    """

    match_results = {}
    match_results["match/oracle_margin"] = []
    # oracle margin is calculated as the score of correct match minus the score of incorrect match
    match_results["match/pseudo_margin"] = []
    # pseudo margin is margin perceived by the learner, which is used to construct pseudo matches
    
    M, N = group_shape[0], group_shape[1]

    group_idx = -1
    num_correct_groups = 0

    model.eval()
    with torch.no_grad():
        for batch, num_groups_in_batch in tqdm(dataloader, desc="Matching"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            similarity_scores = outputs.logits_per_image.cpu()

            for i in range(num_groups_in_batch):
                group_idx += 1
                block = similarity_scores[i*M:i*M+M, i*N:i*N+N]

                if M==2 and N==2:
                    oracle_margin = (block[0, 0] + block[1, 1] - block[0, 1] - block[1, 0]).item()
                    pseudo_margin = abs(oracle_margin)
                    # since only two type of matches exist for 2x2 group, we can directly set pseudo margin as the absolute value of oracle margin
                    if oracle_match:
                        if oracle_margin > 0:
                            num_correct_groups += 1
                            selected_indices.append((group_idx, (0, 1), (0, 1)))
                    else:
                        # since only two type of matches exist for 2x2 group, so we can use oracle margin for selecting both correct and incorrect matches
                        if oracle_margin >= threshold:
                            num_correct_groups += 1
                            selected_indices.append((group_idx, (0, 1), (0, 1)))
                        elif oracle_margin <= -threshold:
                            selected_indices.append((group_idx, (0, 1), (1, 0)))

                    match_results["match/oracle_margin"].append(oracle_margin)
                    match_results["match/pseudo_margin"].append(pseudo_margin)
                    
                elif M==1:
                    oracle_margin = (block[0, 0] - block[0, 1:].max()).item()
                    # margin here is calculated as the score of correct match minus the largest score of incorrect match

                    top2 = torch.topk(block[0, :], 2)
                    pseudo_margin = (top2.values[0] - top2.values[1]).item()

                    if oracle_match:
                        if oracle_margin > 0:
                            selected_indices.append((group_idx, (0,), (0,)))
                    else:
                        if pseudo_margin >= threshold:
                            pseudo_caption_idx = (top2.indices[0]).item()
                            selected_indices.append((group_idx, (0,), (pseudo_caption_idx,)))
                            if pseudo_caption_idx == 0:
                                num_correct_groups += 1

                    match_results["match/oracle_margin"].append(oracle_margin)
                    match_results["match/pseudo_margin"].append(pseudo_margin)
                    
                else:
                    raise ValueError(f"Unsupported group shape: {group_shape}")
    

    match_results["match/oracle_margin_mean"] = np.mean(match_results["match/oracle_margin"])
    match_results["match/oracle_margin_median"] = np.median(match_results["match/oracle_margin"])
    match_results["match/oracle_margin_std"] = np.std(match_results["match/oracle_margin"])
    match_results["match/oracle_margin_min"] = np.min(match_results["match/oracle_margin"])
    match_results["match/oracle_margin_max"] = np.max(match_results["match/oracle_margin"])

    match_results["match/pseudo_margin_mean"] = np.mean(match_results["match/pseudo_margin"])
    match_results["match/pseudo_margin_median"] = np.median(match_results["match/pseudo_margin"])
    match_results["match/pseudo_margin_std"] = np.std(match_results["match/pseudo_margin"])
    match_results["match/pseudo_margin_min"] = np.min(match_results["match/pseudo_margin"])
    match_results["match/pseudo_margin_max"] = np.max(match_results["match/pseudo_margin"])

    match_results["match/threshold"] = threshold
    match_results["match/num_selected_groups"] = len(selected_indices)
    match_results["match/num_correct_groups"] = num_correct_groups
    match_results["match/matching_accuracy"] = num_correct_groups/len(selected_indices) * 100 if len(selected_indices) > 0 else 0
    match_results["match/total_accuracy"] = num_correct_groups/num_groups * 100

    del match_results["match/oracle_margin"]
    del match_results["match/pseudo_margin"]

    print("Evaluation and matching results:", {k: f"{(v):.4f}" if isinstance(v, float) else v for k, v in match_results.items()})

    return match_results, selected_indices


def global_match(model, dataloader, group_shape, threshold=0, use_ratio_threshold=True):

    if group_shape[0] != group_shape[1]:
        raise ValueError(f"Please modify the global_match function to support non-square group shapes")
    
    if use_ratio_threshold:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError(f"With use_ratio_threshold, threshold must be between 0.0 and 1.0, but got {threshold}")
    
    print(f"=== Using threshold {threshold} for global matching with use_ratio_threshold={use_ratio_threshold} ===")
    
    match_results = {}

    device = model.device
    model.eval()
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Global Matching"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            similarity_scores = outputs.logits_per_image.cpu().numpy()
            break
            # we adjust batch size to make sure to load all data in one batch; 
            # if memory is not enough, one need to change this function to load data in a smaller batch

    img_indices, caption_indices = linear_sum_assignment(similarity_scores, maximize=True)

    num_data = similarity_scores.shape[0]

    all_similarity_scores = []

    for i in range(num_data):
        all_similarity_scores.append(similarity_scores[img_indices[i], caption_indices[i]])

    if use_ratio_threshold:

        sorted_similarity_scores = sorted(all_similarity_scores, reverse=True)
        num_selection = int(len(sorted_similarity_scores) * (1.0 - threshold + 1e-10))
        # we add 1e-10 to avoid floating point precision issue, which may cause num_selection to be 1 less than expected

        if num_selection == 0:
            threshold = sorted_similarity_scores[0] + 1
        else:
            threshold = sorted_similarity_scores[num_selection - 1]
    else:
        threshold = threshold
    
    print(f"use_ratio_threshold: {use_ratio_threshold}, threshold: {threshold}")

    selected_indices = []
    selected_similarity_scores = []

    for i in range(num_data):
        if similarity_scores[img_indices[i], caption_indices[i]] >= threshold:
            selected_indices.append((img_indices[i], caption_indices[i]))
            selected_similarity_scores.append(similarity_scores[img_indices[i], caption_indices[i]])
    
    num_selection = len(selected_indices)
    correct_selection = [i == j for (i, j) in selected_indices]
    num_correct_selection = sum(correct_selection)

    match_results["match/mean_similarity_score"] = np.mean(all_similarity_scores)
    match_results["match/median_similarity_score"] = np.median(all_similarity_scores)
    match_results["match/std_similarity_score"] = np.std(all_similarity_scores)
    match_results["match/min_similarity_score"] = np.min(all_similarity_scores)
    match_results["match/max_similarity_score"] = np.max(all_similarity_scores)

    if len(selected_similarity_scores) > 0:
        match_results["match/mean_selected_similarity_score"] = np.mean(selected_similarity_scores)
        match_results["match/median_selected_similarity_score"] = np.median(selected_similarity_scores)
        match_results["match/std_selected_similarity_score"] = np.std(selected_similarity_scores)
        match_results["match/min_selected_similarity_score"] = np.min(selected_similarity_scores)
        match_results["match/max_selected_similarity_score"] = np.max(selected_similarity_scores)
    else:
        match_results["match/mean_selected_similarity_score"] = 0
        match_results["match/median_selected_similarity_score"] = 0
        match_results["match/std_selected_similarity_score"] = 0
        match_results["match/min_selected_similarity_score"] = 0
        match_results["match/max_selected_similarity_score"] = 0

    match_results["match/threshold"] = threshold
    match_results["match/num_selected_data"] = num_selection
    match_results["match/num_correct_data"] = num_correct_selection
    match_results["match/matching_accuracy"] = num_correct_selection/num_selection * 100 if num_selection > 0 else 0
    match_results["match/total_accuracy"] = num_correct_selection/num_data * 100
    
    print("Evaluation and matching results:", {k: f"{(v):.4f}" if isinstance(v, float) else v for k, v in match_results.items()})

    return match_results, selected_indices
