"""
Model handling for CLIP and SigLIP models.
"""

from torchvision import transforms
from transformers import AutoModel, AutoProcessor
import os
import yaml

with open("path_config.yaml", "r") as f:
    path_config = yaml.safe_load(f)
    MODEL_CACHE_DIR = path_config["model_cache_dir"]

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class CLIPCenterCrop:
    """
    Center crop that matches OpenAI/CLIP's floor-based coordinate calculation.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
            
        Returns:
            PIL Image: Cropped image.
        """
        crop_h, crop_w = self.size
        w, h = img.size
        
        # CLIP uses floor division for center crop coordinates
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        
        return img.crop((left, top, right, bottom))

def get_image_transform(model_name, if_augment=False, augmentation_resize_factor=1.1):
    if if_augment:
        if augmentation_resize_factor < 1.0:
            raise ValueError(f"Augmentation resize factor {augmentation_resize_factor} must be >= 1.0")
        print(f"Use train augmentation with resize factor: {augmentation_resize_factor}")
    

    if model_name == "openai/clip-vit-base-patch32" or model_name == "openai/clip-vit-base-patch16":
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        if if_augment:
            first_resize_size = int(224 * augmentation_resize_factor)
            transform = transforms.Compose([
                transforms.Resize(first_resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                CLIPCenterCrop(224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])

    elif model_name == "google/siglip-base-patch16-224" or model_name == "google/siglip-large-patch16-256":
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        resize_size = 224
        if "256" in model_name:
            resize_size = 256
        if if_augment:
            first_resize_size = int(resize_size * augmentation_resize_factor)
            transform = transforms.Compose([
                transforms.Resize((first_resize_size, first_resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(resize_size),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
    else:
        raise ValueError(f"Model name {model_name} not supported")

    return transform

def download_and_cache_model(model_name, models_cache_dir, download=False):
    """Download and cache CLIP model locally."""
    # Replace slashes with underscores to avoid filesystem issues
    safe_model_name = model_name.replace('/', '_')
    model_cache_path = os.path.join(models_cache_dir, safe_model_name)

    if os.path.exists(model_cache_path) and not download:
        print(f"Loading cached model from {model_cache_path}")
        model = AutoModel.from_pretrained(model_cache_path)
        processor = AutoProcessor.from_pretrained(model_cache_path)
    else:
        print(f"Downloading model {model_name} to {model_cache_path}")
        os.makedirs(model_cache_path, exist_ok=True)
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        # Save to cache
        model.save_pretrained(model_cache_path)
        processor.save_pretrained(model_cache_path)

    return model, processor 
