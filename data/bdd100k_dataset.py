import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from pathlib import Path

class BDD100KDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        transform: transforms.Compose = None,
        target_transform: transforms.Compose = None
    ):
        """
        Initializes the BDD100K Dataset.

        Args:
            base_path (str): Base directory containing 'images/10k' and 'labels/sem_seg/masks'.
            transform (transforms.Compose, optional): Transformations to apply to the images.
            target_transform (transforms.Compose, optional): Transformations to apply to the masks.
        """
        self.base_path = Path(base_path)
        self.images_base_dir = self.base_path / 'images' / '10k'
        self.labels_base_dir = self.base_path / 'labels' / 'sem_seg' / 'masks'
        self.det_dir = self.base_path / 'labels' / 'det_20'
        self.transform = transform
        self.target_transform = target_transform

        self.det_train_path = self.det_dir / 'det_train.json'
        self.det_val_path = self.det_dir / 'det_val.json'

        self.scene_info = self._load_scene_info()

        self.image_filenames, self.labels_dirs = self._gather_filenames()

    def _load_scene_info(self) -> dict:
        """
        Loads scene information from detection JSON files.

        Returns:
            dict: A mapping from image filename to scene information.
        """
        scene_map = {}
        
        if self.det_train_path.exists():
            with open(self.det_train_path, 'r') as f:
                det_train_data = json.load(f)
            train_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_train_data}
            scene_map.update(train_scene_map)
        else:
            print(f"Warning: {self.det_train_path} does not exist.")

        if self.det_val_path.exists():
            with open(self.det_val_path, 'r') as f:
                det_val_data = json.load(f)
            val_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_val_data}
            scene_map.update(val_scene_map)
        else:
            print(f"Warning: {self.det_val_path} does not exist.")

        return scene_map

    def _gather_filenames(self) -> tuple:
        """
        Gathers image filenames from 'train' and 'val' directories that have scene information.

        Returns:
            tuple: A tuple containing:
                - List of image file paths.
                - List of corresponding label directories.
        """
        image_filenames = []
        labels_dirs = []

        for split in ['train', 'val']:
            images_dir = self.images_base_dir / split
            labels_dir = self.labels_base_dir / split

            if not images_dir.exists():
                print(f"Warning: Images directory {images_dir} does not exist.")
                continue
            if not labels_dir.exists():
                print(f"Warning: Labels directory {labels_dir} does not exist.")
                continue

            split_image_filenames = [f for f in os.listdir(images_dir) if f in self.scene_info]

            image_filenames.extend([images_dir / f for f in split_image_filenames])
            labels_dirs.extend([labels_dir] * len(split_image_filenames))

        return image_filenames, labels_dirs

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            idx = idx.tolist() if isinstance(idx, np.ndarray) else idx
            return [self[i] for i in idx]
        else:
            image_path = self.image_filenames[idx]
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            label_path = self.labels_dirs[idx] / image_path.name.replace('.jpg', '.png')
            if not label_path.exists():
                raise FileNotFoundError(f"Label file {label_path} does not exist.")

            label = Image.open(label_path)

            if self.target_transform:
                label = self.target_transform(label)
            else:
                label = torch.tensor(np.array(label), dtype=torch.long)

            scene = self.scene_info.get(image_path.name, None)

            return image, label, scene
