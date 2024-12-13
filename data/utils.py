import os
import json
from collections import Counter

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
from torchvision import transforms
from PIL import Image

def custom_split_dataset_with_det(
    base_data_path,
    base_labels_path,
    det_train_path,
    det_val_path,
    val_subfolder='val',
    train_subfolder='train',
    test_size=454,
    random_state=1337
):
    with open(det_train_path) as f:
        det_train_data = json.load(f)
    with open(det_val_path) as f:
        det_val_data = json.load(f)

    train_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_train_data}
    val_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_val_data}

    train_images_dir = os.path.join(base_data_path)
    train_labels_dir = os.path.join(base_labels_path)

    train_image_filenames = [f for f in os.listdir(train_images_dir) if f in train_scene_map]
    val_image_filenames = [f for f in os.listdir(train_images_dir) if f in val_scene_map]

    train_image_filenames, test_image_filenames = train_test_split(
        train_image_filenames, test_size=test_size, random_state=random_state
    )
    
    print()
    print("--- Split Sizes ---")
    print(f"- Train Images: {len(train_image_filenames)}")
    print(f"- Val Images: {len(val_image_filenames)}")
    print(f"- Test Images: {len(test_image_filenames)}")
    print()
    
    return {
        'train': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': train_image_filenames,
            'scene_map': {k: train_scene_map[k] for k in train_image_filenames},
        },
        'val': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': val_image_filenames,
            'scene_map': {k: val_scene_map[k] for k in val_image_filenames},
        },
        'test': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': test_image_filenames,
            'scene_map': {k: train_scene_map[k] for k in test_image_filenames},
        }
    }
    
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

class ConvertToLongTensor:
    def __call__(self, x):
        return torch.tensor(np.array(x), dtype=torch.long)

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