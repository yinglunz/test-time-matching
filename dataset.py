"""
Dataset handling for test-time matching.
"""

import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from collections import defaultdict
import torch
import json
import os
import subprocess
import yaml


try:
    profile
except NameError:
    def profile(func): return func

with open("path_config.yaml", "r") as f:
    path_config = yaml.safe_load(f)
    MODEL_CACHE_DIR = path_config["model_cache_dir"]
    DATASET_CACHE_DIR = path_config["dataset_cache_dir"]


def get_eval_metrics(group_shape, matching_method="group"):
    if group_shape == (2, 2):
        metrics = {
            "eval/text_score": [],
            "eval/image_score": [],
            "eval/group_score": [],
            "eval/individual_match_score": [],
            "eval/group_match_score": [],
        }
    elif group_shape == (1, 2) or group_shape == (1, 4):
        metrics = {
            "eval/individual_match_score": [],
        }
    else:
        raise ValueError(f"Unsupported group shape: {group_shape}")
    
    if matching_method == "global":
        metrics["eval/global_pseudo_label_score"] = []
        metrics["eval/global_match_score"] = []
    
    return metrics

class GroupedDataset(Dataset):
    """
    Dataset with group structures. 
    num_groups = len(images) = len(captions) = len(group_labels)

    images: list of grouped images, shape: (num_groups, num_images_in_group)
    captions: list of grouped captions, shape: (num_groups, num_captions_in_group)
    transform: transform to apply to the images
    dataset_name: name of the dataset

    group_shape (inferred from inputs): [num_images_in_group, num_captions_in_group]
    """
    def __init__(self, images, captions, transform=None, dataset_name=None):
        assert len(images) == len(captions)
        self.num_groups = len(images)
        self.images = images
        self.captions = captions
        self.transform = transform
        self.dataset_name = dataset_name
        self.group_shape = (len(images[0]), len(captions[0])) # [num_images_in_group, num_captions_in_group]
    
    def __len__(self):
        return self.num_groups

    def __getitem__(self, idx):
        if self.transform is not None:
            image_group = tuple(self.transform(image) for image in self.images[idx])
        else:
            image_group = self.images[idx]
        caption_group = self.captions[idx]
        return image_group, caption_group 

def collate_fn_group(batch, tokenizer):
    image_groups, caption_groups = zip(*batch)
    num_groups = len(image_groups)
    images = [image for image_group in image_groups for image in image_group]
    images = torch.stack(images, dim=0)
    captions = [caption for caption_group in caption_groups for caption in caption_group]
    captions = tokenizer(captions, return_tensors="pt", padding="max_length", truncation=True)
    batch_inputs = {
        "pixel_values": images,
        "input_ids": captions['input_ids'],
    }
    if "attention_mask" in captions:
        batch_inputs["attention_mask"] = captions['attention_mask']

    return batch_inputs, num_groups

def download_and_cache_dataset(dataset_name, datasets_cache_dir):
    """Download and cache dataset locally."""
    # Replace slashes with underscores to avoid filesystem issues
    safe_dataset_name = dataset_name.replace('/', '_')
    dataset_cache_path = os.path.join(datasets_cache_dir, safe_dataset_name)
    
    if os.path.exists(dataset_cache_path):
        print(f"Loading cached dataset from {dataset_cache_path}")
        dataset = load_from_disk(dataset_cache_path)
    else:
        print(f"Downloading dataset {dataset_name} to {dataset_cache_path}")
        os.makedirs(datasets_cache_dir, exist_ok=True)
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_cache_path)

    return dataset

@profile
def get_grouped_dataset(dataset_name, datasets_cache_dir, transform):

    if dataset_name == "stanfordnlp/colorswap":
        dataset = download_and_cache_dataset(dataset_name, datasets_cache_dir)
        dataset = dataset["test"]
        images = [(image_1.convert("RGB").copy(), image_2.convert("RGB").copy()) for (image_1, image_2) in zip(dataset['image_1'], dataset['image_2'])]
        captions = list(zip(dataset['caption_1'], dataset['caption_2']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "facebook/winoground":
        dataset = download_and_cache_dataset(dataset_name, datasets_cache_dir)
        dataset = dataset["test"]
        images = [(image_0.convert("RGB").copy(), image_1.convert("RGB").copy()) for (image_0, image_1) in zip(dataset['image_0'], dataset['image_1'])]
        captions = list(zip(dataset['caption_0'], dataset['caption_1']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)
    
    elif dataset_name == "mmvp_vlm":
        images, captions = preprocess_mmvp_vlm_dataset(datasets_cache_dir)
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)
    
    elif "sugarcrepe" in dataset_name:
        subset = dataset_name.split('/')[-1]
        json_path, image_dir = download_and_cache_sugarcrepe_dataset(subset, datasets_cache_dir)
        with open(json_path, "r") as f:
            data = json.load(f)
        images = []
        captions = []
        for _, item in data.items():
            image = Image.open(os.path.join(image_dir, item['filename'])).convert('RGB').copy()
            images.append((image,))
            captions.append((item['caption'], item['negative_caption']))

        print(f"len(images): {len(images)}; len(captions): {len(captions)}")
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_a_left_right":
        json_path, image_root = download_whatsup_dataset("whatsup_a", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_a", datasets_cache_dir, json_path, image_root)["left_right"]
        images = list(zip(dataset['image_0'], dataset['image_1']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_a_on_under":
        json_path, image_root = download_whatsup_dataset("whatsup_a", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_a", datasets_cache_dir, json_path, image_root)["other_relations"]
        images = list(zip(dataset['image_0'], dataset['image_1']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_a_1x4":
        json_path, image_root = download_whatsup_dataset("whatsup_a", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_a", datasets_cache_dir, json_path, image_root)["1x4"]
        images = list(zip(dataset['image_0']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1'], dataset['caption_2'], dataset['caption_3']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_b_left_right":
        json_path, image_root = download_whatsup_dataset("whatsup_b", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_b", datasets_cache_dir, json_path, image_root)["left_right"]
        images = list(zip(dataset['image_0'], dataset['image_1']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_b_front_behind":
        json_path, image_root = download_whatsup_dataset("whatsup_b", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_b", datasets_cache_dir, json_path, image_root)["other_relations"]
        images = list(zip(dataset['image_0'], dataset['image_1']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    elif dataset_name == "whatsup_b_1x4":
        json_path, image_root = download_whatsup_dataset("whatsup_b", datasets_cache_dir)
        dataset = preprocess_raw_whatsup_dataset("whatsup_b", datasets_cache_dir, json_path, image_root)["1x4"]
        images = list(zip(dataset['image_0']))
        captions = list(zip(dataset['caption_0'], dataset['caption_1'], dataset['caption_2'], dataset['caption_3']))
        grouped_dataset = GroupedDataset(images, captions, transform, dataset_name)

    else:
        raise ValueError(f"Dataset name {dataset_name} not supported")

    return grouped_dataset

def download_and_cache_mmvp_vlm_dataset(datasets_cache_dir):
    """Download and cache MMVP_VLM dataset locally."""
    dataset_cache_dir = os.path.join(datasets_cache_dir, "mmvp_vlm")

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir, exist_ok=True)

    image_dir = os.path.join(dataset_cache_dir, "MLLM_VLM Images")
    annotation_file = os.path.join(dataset_cache_dir, "Questions.csv")

    if not os.path.exists(image_dir) or not os.path.exists(annotation_file):
        print(f"Image directory {image_dir} or annotation file {annotation_file} not found! Downloading...")
        snapshot_download(repo_id="MMVP/MMVP_VLM", repo_type="dataset", local_dir=dataset_cache_dir)

    return annotation_file, image_dir

def preprocess_mmvp_vlm_dataset(datasets_cache_dir):
    """Preprocess MMVP_VLM dataset."""
    annotation_file, image_dir = download_and_cache_mmvp_vlm_dataset(datasets_cache_dir)
    df = pd.read_csv(annotation_file)

    images = []
    captions = []

    image_pair = []
    caption_pair = []

    for _, row in df.iterrows():
        image_file = str(row["Question ID"]) + ".jpg"
        subfolder = row["Type"]
        caption = row["Statement"]

        image = Image.open(os.path.join(image_dir, subfolder, image_file)).convert("RGB").copy()

        image_pair.append(image)
        caption_pair.append(caption)

        if len(image_pair) == 2:
            images.append(tuple(image_pair))
            captions.append(tuple(caption_pair))
            image_pair = []
            caption_pair = []
    
    return images, captions

def download_and_cache_sugarcrepe_dataset(subset, datasets_cache_dir):
    """Download and cache SugarCrepe dataset locally."""

    dataset_cache_dir = os.path.join(datasets_cache_dir, "sugarcrepe")
    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir, exist_ok=True)

    image_dir = os.path.join(dataset_cache_dir, "val2017")
    annotation_file = os.path.join(dataset_cache_dir, subset + ".json")

    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} not found! Downloading...")
        image_zip = os.path.join(dataset_cache_dir, "val2017.zip")
        if not os.path.exists(image_zip):
            subprocess.call(["wget", "-O", image_zip, "http://images.cocodataset.org/zips/val2017.zip"])
        subprocess.call(["unzip", image_zip], cwd=dataset_cache_dir)

    if not os.path.exists(annotation_file):
        print(f"Annotation file {annotation_file} not found! Downloading...")
        if subset == 'swap_obj':
            subprocess.call(["wget", "-O", annotation_file, "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/refs/heads/main/data/swap_obj.json"])
        elif subset == 'swap_att':
            subprocess.call(["wget", "-O", annotation_file, "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/refs/heads/main/data/swap_att.json"])
        elif subset == 'add_att':
            subprocess.call(["wget", "-O", annotation_file, "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/refs/heads/main/data/add_att.json"])
        elif subset == 'replace_rel':
            subprocess.call(["wget", "-O", annotation_file, "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/refs/heads/main/data/replace_rel.json"])
        else:
            raise ValueError(f"Subset {subset} not supported")
    
    return annotation_file, image_dir

def download_whatsup_dataset(dataset_name, datasets_cache_dir, download=True):
    """
    Download and cache whatsup dataset locally.
    Code adapted from https://github.com/amitakamath/whatsup_vlms/blob/main/dataset_zoo/aro_datasets.py
    """
    datasets_cache_dir = datasets_cache_dir
    if dataset_name == 'whatsup_a':
        datasets_cache_dir = os.path.join(datasets_cache_dir, "whatsup_a_raw")

        annotation_file = os.path.join(datasets_cache_dir, "controlled_images_dataset.json")
        image_dir = os.path.join(datasets_cache_dir, "controlled_images")

        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} for Controlled Images A is not found!")
            if download:
                # Download and extract controlled_images dataset
                os.makedirs(datasets_cache_dir, exist_ok=True)
                images_tar = os.path.join(datasets_cache_dir, "controlled_images.tar.gz")
                if not os.path.exists(images_tar):
                    subprocess.call(["gdown", "--no-cookies", "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", images_tar])
                subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=datasets_cache_dir)
            else:
                raise RuntimeError("Please download dataset (set --download to True) or specify correct directory.")

        if not os.path.exists(annotation_file):
            print(f"Annotation file {annotation_file} not found! Downloading...")
            subprocess.call(["gdown", "--id", "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm", "--output", annotation_file])

    elif dataset_name == 'whatsup_b':
        datasets_cache_dir = os.path.join(datasets_cache_dir, "whatsup_b_raw")

        annotation_file = os.path.join(datasets_cache_dir, "controlled_clevr_dataset.json")
        image_dir = os.path.join(datasets_cache_dir, "controlled_clevr")

        if not os.path.exists(image_dir):
            print(f"Image directory {image_dir} for Controlled Images B is not found!")
            if download:
                # Download and extract controlled_clevr dataset
                os.makedirs(datasets_cache_dir, exist_ok=True)
                images_tar = os.path.join(datasets_cache_dir, "controlled_clevr.tar.gz")
                if not os.path.exists(images_tar):
                    subprocess.call(["gdown", "--no-cookies", "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", images_tar])
                subprocess.call(["tar", "-xvf",  "controlled_clevr.tar.gz"], cwd=datasets_cache_dir)
            else:
                raise RuntimeError("Please download dataset (set --download to True) or specify correct directory.")

        if not os.path.exists(annotation_file):
            print(f"Annotation file {annotation_file} not found! Downloading...")
            subprocess.call(["gdown", "--id", "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z", "--output", annotation_file])
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported")
    
    json_path = annotation_file 
    image_root = image_dir

    return json_path, image_root

def preprocess_raw_whatsup_dataset(dataset_name, datasets_cache_dir, json_path, image_root, reprocessing=False):

    def strip_relation(path):
        """
        Remove spatial relation keyword from filename.
        E.g., 'beer-bottle_on_armchair.jpeg' -> 'beer-bottle__armchair.jpeg'
        """
        filename = os.path.basename(path)
        for rel in ['_on_', '_under_', '_left_of_', '_right_of_', '_in-front_of_', '_behind_']:
            if rel in filename:
                return filename.replace(rel, '__')
        return filename  # fallback if no known pattern found

    safe_dataset_name = dataset_name.replace('/', '_')
    dataset_cache_path = os.path.join(datasets_cache_dir, safe_dataset_name)

    # If reprocessing=True, delete existing cached dataset
    if reprocessing and os.path.exists(dataset_cache_path):
        import shutil
        print(f"Reprocessing requested. Removing existing cached dataset at {dataset_cache_path}")
        shutil.rmtree(dataset_cache_path)

    if not os.path.exists(dataset_cache_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        pair_groups = defaultdict(list)
        for item in raw_data:
            key = strip_relation(item["image_path"])
            pair_groups[key].append(item)
        
        data_pairs = []
        left_right_pairs = []
        other_relation_pairs = []
        data_1x4 = []


        for key, items in pair_groups.items():
            # Separate into relationship buckets
            relation_buckets = {
                'on': [],
                'under': [],
                'left_of': [],
                'right_of': [],
                'in-front_of': [],
                'behind': []
            }
            
            for item in items:
                path = item["image_path"]
                basename = os.path.basename(path)
                if '_on_' in basename:
                    relation_buckets['on'].append(item)
                elif '_under_' in basename:
                    relation_buckets['under'].append(item)
                elif 'left_of' in basename:
                    relation_buckets['left_of'].append(item)
                elif 'right_of' in basename:
                    relation_buckets['right_of'].append(item)
                elif 'in-front_of' in basename:
                    relation_buckets['in-front_of'].append(item)
                elif 'behind' in basename:
                    relation_buckets['behind'].append(item)

            # Pairing (on, under)
            assert len(relation_buckets['on']) == len(relation_buckets['under'])

            for i in range(len(relation_buckets['on'])):
                item_0 = relation_buckets['on'][i]
                item_1 = relation_buckets['under'][i]

                try:
                    image_0 = Image.open(os.path.join(image_root, os.path.basename(item_0["image_path"]))).convert("RGB").copy()
                    image_1 = Image.open(os.path.join(image_root, os.path.basename(item_1["image_path"]))).convert("RGB").copy()

                    data_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],
                    })
                    other_relation_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],                       
                    })
                    data_1x4.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "caption_1": item_0["caption_options"][1],
                        "caption_2": item_0["caption_options"][2],
                        "caption_3": item_0["caption_options"][3],
                    })
                    data_1x4.append({
                        "image_0": image_1,
                        "caption_0": item_1["caption_options"][0],
                        "caption_1": item_1["caption_options"][1],
                        "caption_2": item_1["caption_options"][2],
                        "caption_3": item_1["caption_options"][3],
                    
                    })
                except Exception as e:
                    print(f"[Error] on-under pairing failed for {key}: {e}")

            # Pairing (left_of, right_of)
            assert len(relation_buckets['left_of']) == len(relation_buckets['right_of'])

            for i in range(len(relation_buckets['left_of'])):
                item_0 = relation_buckets['left_of'][i]
                item_1 = relation_buckets['right_of'][i]

                try:
                    if dataset_name == "whatsup_a" and os.path.basename(item_0["image_path"]) == "pillow_left_of_chair.jpeg" and os.path.basename(item_1["image_path"]) == "pillow_right_of_chair.jpeg":

                        print(f"correcting label errors for {os.path.basename(item_0['image_path'])} and {os.path.basename(item_1['image_path'])}")

                        """
                        correct label errors based on GitHub issue #4
                        https://github.com/amitakamath/whatsup_vlms/issues/4
                        one should remove this part if future versions of the dataset fix the label errors
                        """

                        image_0 = Image.open(os.path.join(image_root, os.path.basename(item_1["image_path"]))).convert("RGB").copy()
                        image_1 = Image.open(os.path.join(image_root, os.path.basename(item_0["image_path"]))).convert("RGB").copy()

                    else:
                        image_0 = Image.open(os.path.join(image_root, os.path.basename(item_0["image_path"]))).convert("RGB").copy()
                        image_1 = Image.open(os.path.join(image_root, os.path.basename(item_1["image_path"]))).convert("RGB").copy()

                    data_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],
                    })
                    left_right_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],                       
                    })
                    data_1x4.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "caption_1": item_0["caption_options"][1],
                        "caption_2": item_0["caption_options"][2],
                        "caption_3": item_0["caption_options"][3],
                    })
                    data_1x4.append({
                        "image_0": image_1,
                        "caption_0": item_1["caption_options"][0],
                        "caption_1": item_1["caption_options"][1],
                        "caption_2": item_1["caption_options"][2],
                        "caption_3": item_1["caption_options"][3],
                    
                    })
                except Exception as e:
                    print(f"[Error] left-right pairing failed for {key}: {e}")

            # Pairing (in-front_of, behind)
            assert len(relation_buckets['in-front_of']) == len(relation_buckets['behind'])

            for i in range(len(relation_buckets['in-front_of'])):
                item_0 = relation_buckets['in-front_of'][i]
                item_1 = relation_buckets['behind'][i]

                try:
                    image_0 = Image.open(os.path.join(image_root, os.path.basename(item_0["image_path"]))).convert("RGB").copy()
                    image_1 = Image.open(os.path.join(image_root, os.path.basename(item_1["image_path"]))).convert("RGB").copy()

                    data_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],
                    })
                    other_relation_pairs.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "image_1": image_1,
                        "caption_1": item_1["caption_options"][0],
                    })
                    data_1x4.append({
                        "image_0": image_0,
                        "caption_0": item_0["caption_options"][0],
                        "caption_1": item_0["caption_options"][1],
                        "caption_2": item_0["caption_options"][2],
                        "caption_3": item_0["caption_options"][3],
                    })
                    data_1x4.append({
                        "image_0": image_1,
                        "caption_0": item_1["caption_options"][0],
                        "caption_1": item_1["caption_options"][1],
                        "caption_2": item_1["caption_options"][2],
                        "caption_3": item_1["caption_options"][3],
                    
                    })
                except Exception as e:
                    print(f"[Error] front-behind pairing failed for {key}: {e}")
        

        dataset_all = datasets.Dataset.from_list(data_pairs)
        dataset_left_right = datasets.Dataset.from_list(left_right_pairs)
        dataset_other = datasets.Dataset.from_list(other_relation_pairs) 
        dataset_1x4 = datasets.Dataset.from_list(data_1x4)

        dataset = datasets.DatasetDict({
            "test": dataset_all,
            "left_right": dataset_left_right,
            "other_relations": dataset_other,
            "1x4":dataset_1x4
        })

        os.makedirs(dataset_cache_path, exist_ok=True)
        dataset.save_to_disk(dataset_cache_path)
    else:
        dataset = load_from_disk(dataset_cache_path)

    return dataset
