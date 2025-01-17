import os
import json
from collections import Counter

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
from torchvision import transforms
from PIL import Image

from torch.utils.data import Subset

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Splits the dataset into train, validation, and test partitions.

    :param dataset: The BDD100KDataset dataset object.
    :param train_ratio: Proportion of the dataset to allocate to the training set.
    :param val_ratio: Proportion of the dataset to allocate to the validation set.
    :param test_ratio: Proportion of the dataset to allocate to the test set.
    :param random_seed: Seed for reproducibility of the split.
    :return: A tuple of (train_dataset, val_dataset, test_dataset).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    np.random.seed(random_seed)
    
    total_size = len(dataset)
    indices = np.arange(total_size)
    
    np.random.shuffle(indices)
    
    train_split = int(train_ratio * total_size)
    val_split = train_split + int(val_ratio * total_size)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset
    
def check_dataset_overlap(train_filenames, val_filenames, test_filenames):
    train_set = set(train_filenames)
    val_set = set(val_filenames)
    test_set = set(test_filenames)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    print("--- Overlap Report ---")
    if train_val_overlap:
        print(f"🚨 Overlap detected between train and validation sets! {len(train_val_overlap)} samples.")
        print(f"Overlapping samples: {list(train_val_overlap)}")
    else:
        print("✔️ No overlap detected between train and validation sets.")

    if train_test_overlap:
        print(f"🚨 Overlap detected between train and test sets! {len(train_test_overlap)} samples.")
        print(f"Overlapping samples: {list(train_test_overlap)}")
    else:
        print("✔️ No overlap detected between train and test sets.")

    if val_test_overlap:
        print(f"🚨 Overlap detected between validation and test sets! {len(val_test_overlap)} samples.")
        print(f"Overlapping samples: {list(val_test_overlap)}")
    else:
        print("✔️ No overlap detected between validation and test sets.")
    print()

def map_class_names_and_order(class_distribution, class_dict):
    ordered_classes = sorted(class_dict.keys())  # Ensure consistent class order
    class_names = [class_dict[class_id] for class_id in ordered_classes if class_id in class_distribution]
    proportions = [class_distribution[class_id] for class_id in ordered_classes if class_id in class_distribution]
    return class_names, proportions

def analyze_class_distribution(dataset, num_classes, dataset_name):
    class_counts = Counter()
    
    for idx in trange(len(dataset), desc=f"Analyzing {dataset_name}"):
        try:
            _, mask, _ = dataset[idx]  # Access dataset item
            mask_array = np.array(mask)  # Convert mask to numpy array
            unique, counts = np.unique(mask_array, return_counts=True)
            class_counts.update(dict(zip(unique, counts)))
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

    total_pixels = sum(class_counts.values())
    class_distribution = {cls: count / total_pixels for cls, count in class_counts.items()}

    return class_counts, class_distribution

def calculate_normalization_stats(image_filenames, base_data_path: str):
    total_sum = torch.zeros(3)
    total_squared_sum = torch.zeros(3)
    total_pixel_count = 0

    transform = transforms.ToTensor()

    for image_path in image_filenames:
        image_path = os.path.join(base_data_path, image_path)

        image = Image.open(image_path).convert('RGB')

        tensor_image = transform(image)

        pixels = tensor_image.numel() / 3

        total_sum += tensor_image.sum(dim=(1, 2))
        total_squared_sum += (tensor_image ** 2).sum(dim=(1, 2))

        total_pixel_count += pixels

    mean = total_sum / total_pixel_count
    std = torch.sqrt(total_squared_sum / total_pixel_count - mean ** 2)

    return mean, std

class RemapClasses:
    def __init__(self, old_to_new):
        """
        old_to_new: dict mapping old class indices to new class indices.
                    Any class index not in old_to_new will be set to 255.
        """
        self.old_to_new = old_to_new

    def __call__(self, mask):
        new_mask = torch.full_like(mask, 255)
        for old_class, new_class in self.old_to_new.items():
            new_mask[mask == old_class] = new_class
        return new_mask
    
class ConvertToLongTensor:
    def __call__(self, x):
        return torch.tensor(np.array(x), dtype=torch.long)
    
def unnormalize(img, mean, std):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)

    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = std * img + mean
    img = torch.clip(img, 0, 1)
    return img
