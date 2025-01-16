import copy
import os
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from data.utils import unnormalize, RemapClasses
from data import class_dict_remapped

mean = [0.3654, 0.4002, 0.4055]
std = [0.2526, 0.2644, 0.2755]

class Analyzer:
    def __init__(self, model: nn.Module, device: str="cpu", project_name: str="dlbs", entity_name: str="okaynils"):
        self.model = model
        self.device = device
        self.project_name = project_name
        self.entity_name = entity_name
        self.history = None
        self.run_name = None
        self.elapsed_time = None
    
    def model_receipt(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        device = next(self.model.parameters()).device
        
        print(f"--- Model Receipt for {self.model.__class__.__name__} ---")
        if self.elapsed_time:
            print(f"\nTraining Time: {self.elapsed_time/60**2:.2f} hours")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")
        print(f"Device: {device}")
        
        print("\nModel Architecture:\n")
        print(self.model)
        
    def plot(self, run_id: str):
        self._fetch_data(run_id)

        history_list = list(self.history)
        
        train_loss = [entry['train_loss'] for entry in history_list if 'train_loss' in entry and entry['train_loss'] is not None]
        val_loss = [entry['val_loss'] for entry in history_list if 'val_loss' in entry and entry['val_loss'] is not None]
        val_global_iou = [entry['val_global_iou'] for entry in history_list if 'val_global_iou' in entry and entry['val_global_iou'] is not None]
        
        contains_test_metrics = any('test_city_iou' in entry for entry in history_list)
        
        test_city_iou = None
        test_non_city_iou = None
        test_class_ious = []
        if contains_test_metrics:
            last_test_entry = next(entry for entry in reversed(history_list) if 'test_city_iou' in entry)
            test_city_iou = last_test_entry['test_city_iou']
            test_non_city_iou = last_test_entry['test_non_city_iou']
            for i in range(5):
                test_class_iou = last_test_entry.get(f'test_iou_class_{i}', None)
                if test_class_iou is not None:
                    test_class_ious.append(test_class_iou)

        val_class_ious = []
        for i in range(5):
            val_class_iou = [entry[f'val_iou_class_{i}'] for entry in history_list if f'val_iou_class_{i}' in entry and entry[f'val_iou_class_{i}'] is not None]
            val_class_ious.append(val_class_iou)

        colors = list(mcolors.TABLEAU_COLORS.values())
        class_colors = {i: colors[i % len(colors)] for i in range(len(val_class_ious))}

        if contains_test_metrics:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        else: 
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].plot(train_loss, label='Train Loss', zorder=3)
        axes[0].plot(val_loss, label='Validation Loss', zorder=2)
        axes[0].grid(True, zorder=1)
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        
        axes[1].plot(val_global_iou, label='Validation Global IoU', zorder=3)
        for i, val_class_iou in enumerate(val_class_ious):
            axes[1].plot(val_class_iou, label=f'{class_dict_remapped[i].capitalize()} IoU', linestyle='--', color=class_colors[i], zorder=2)
        axes[1].grid(True, zorder=1)
        axes[1].set_title("Validation Global IoU")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU")
        axes[1].legend()
        
        if contains_test_metrics:
            test_labels = ['City IoU', 'Non-City IoU'] + [f'{class_dict_remapped[i].capitalize()} IoU' for i in range(len(test_class_ious))]
            test_values = [test_city_iou, test_non_city_iou] + test_class_ious
            bar_colors = ['gray', 'lightgray'] + [class_colors[i] for i in range(len(test_class_ious))]
            
            bars = axes[2].bar(test_labels, test_values, color=bar_colors, zorder=3)
            axes[2].grid(True, axis='y', zorder=1)
            axes[2].set_title("Test IoU Metrics")
            axes[2].set_ylabel("IoU")
            axes[2].set_xticks(range(len(test_labels)))
            axes[2].set_xticklabels(test_labels, rotation=45, ha='right')
            
            for bar, value in zip(bars, test_values):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{value:.2%}', 
                            ha='center', va='bottom', fontsize=10, color='black')

        plt.tight_layout()
        plt.show()

    def sample(self, run_id, data_pairs: list):
        """
        Predict and visualize model predictions for a list of data pairs.

        Args:
            run_id (str): The run ID for loading model weights.
            data_pairs (list): A list of tuples (image, ground_truth, scene_label).
                image: Tensor of the input image.
                ground_truth: Ground truth segmentation mask (optional, can be None if not used).
                scene_label: Scene label string.
        """
        model_path = self._get_model_path(run_id)
        
        self.model.load_state_dict(torch.load(f'models/{model_path}', map_location=self.device, weights_only=True))
        self.model.eval()

        colors = list(mcolors.TABLEAU_COLORS.values())
        class_colors = {i: colors[i % len(colors)] for i in range(5)}

        images = [pair[0] for pair in data_pairs]
        ground_truths = [pair[1] for pair in data_pairs]
        scene_labels = [pair[2] for pair in data_pairs]

        with torch.no_grad():
            predictions = [self.model(image.unsqueeze(0).to(self.device)) for image in images]

        print(torch.unique(torch.argmax(predictions[0].squeeze(0), dim=0).cpu()))
        
        fig, axes = plt.subplots(3, len(images), figsize=(15, 10))
        for idx, (pair, pred) in enumerate(zip(data_pairs, predictions)):
            image, ground_truth, scene_label = pair
            
            remapping_transform = RemapClasses(old_to_new={0: 0,
                                                           2: 1,
                                                           8: 2,
                                                           10: 3,
                                                           13: 4})
                                                        
            ground_truth = remapping_transform(ground_truth)
            ground_truth[ground_truth == 255] = 4
            
            unnormalized_image = unnormalize(image, mean, std)
            pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy()

            axes[0, idx].imshow(unnormalized_image.permute(1, 2, 0))
            axes[0, idx].set_title(f"Image {idx+1}")
            axes[0, idx].text(5, 5, scene_label, fontsize=10, color='white', 
                            bbox=dict(facecolor='black', alpha=0.8, pad=2), va='top', ha='left')
            axes[0, idx].axis("off")

            cmap = mcolors.ListedColormap([class_colors[i] for i in range(5)])
            axes[1, idx].imshow(pred_mask, cmap=cmap)
            axes[1, idx].set_title(f"Prediction {idx+1}")
            axes[1, idx].axis("off")

            if ground_truth is not None:
                ground_truth_mask = ground_truth.cpu().numpy()
                overlap = (ground_truth_mask == pred_mask).astype(float)
                overlap_percentage = 100 * overlap.sum() / ground_truth_mask.size
                axes[2, idx].imshow(overlap, cmap="Greens", alpha=0.7)
                axes[2, idx].set_title(f"Ground Truth Overlap {idx+1}: {overlap_percentage:.2f}%", fontsize=10)
            else:
                axes[2, idx].set_title("No Ground Truth")
            axes[2, idx].axis("off")

        legend_patches = [plt.Line2D([0], [0], color=class_colors[i], lw=4, label=f"{class_dict_remapped[i]}")
                        for i in range(5)]
        fig.legend(handles=legend_patches, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))

        plt.tight_layout()
        plt.show()

    def _get_model_path(self, run_id: str):
        print(f"\nSearching for model weights for run {run_id}...")
        models = os.listdir('models')
        model_path = None
        for model in models:
            run_name = model.split('_')
            run_id_in_name = run_name[-1].split('.')[0]
            if run_id_in_name == run_id:
                model_path = model
                print(f'Found model: {model_path}!')
        return model_path
        
    def _load_model_weights(self, run_id: str):
        model_path = self._get_model_path(run_id)
        
        self.model.load_state_dict(torch.load(f'models/{model_path}', map_location=self.device, weights_only=True))
        self.model.to(self.device)
    
    def _fetch_data(self, run_id: str):
        api = wandb.Api()
        try:
            run = api.run(f"{self.entity_name}/{self.project_name}/{run_id}")
            self.run_name = run.name
            self.elapsed_time = run.summary.get('_runtime', None)
            self.history = run.scan_history()
        except wandb.errors.CommError as e:
            raise ValueError(f"Error fetching run: {e}")
