import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, filenames=None, transform=None, target_transform=None, scene_info=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform
        self.scene_info = scene_info if scene_info else {}

        self.image_filenames = filenames if filenames else os.listdir(images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        
        try:
            image_path = os.path.join(self.images_dir, self.image_filenames[idx])
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            if self.labels_dir:
                label_filename = self.image_filenames[idx].replace('.jpg', '.png')
                label_path = os.path.join(self.labels_dir, label_filename)
                label = Image.open(label_path)

                if self.target_transform:
                    label = self.target_transform(label)
                else:
                    label = torch.tensor(np.array(label), dtype=torch.long)

            scene = self.scene_info.get(self.image_filenames[idx], None)

            if self.labels_dir:
                return image, label, scene
            return image, scene

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise e
