import os
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from data.utils import unnormalize, RemapClasses
from data import class_dict_remapped
from core import AttentionUNet
from core.modules import AttentionBlock

mean = [0.3654, 0.4002, 0.4055]
std = [0.2526, 0.2644, 0.2755]

class Analyzer:
    def __init__(
        self,
        model: nn.Module = None,
        device: str = "cpu",
        project_name: str = "dlbs",
        entity_name: str = "okaynils"
    ):
        self.model = model
        self.device = device
        self.project_name = project_name
        self.entity_name = entity_name
        
        self.history = None
        self.run_name = None
        self.elapsed_time = None

    def model_receipt(self):
        if self.model is None:
            print("No model was provided.")
            return
        
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
        if not self.model:
            print("No model instance is set. Only W&B data will be plotted.")
        
        self._fetch_data(run_id)
        history_list = list(self.history)
        
        train_loss = [
            entry['train_loss']
            for entry in history_list 
            if 'train_loss' in entry and entry['train_loss'] is not None
        ]
        val_loss = [
            entry['val_loss']
            for entry in history_list 
            if 'val_loss' in entry and entry['val_loss'] is not None
        ]
        
        val_global_iou = [
            entry['val_global_iou']
            for entry in history_list 
            if 'val_global_iou' in entry and entry['val_global_iou'] is not None
        ]
        
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
            val_class_iou = [
                entry[f'val_iou_class_{i}'] 
                for entry in history_list 
                if f'val_iou_class_{i}' in entry and entry[f'val_iou_class_{i}'] is not None
            ]
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
            axes[1].plot(
                val_class_iou,
                label=f'{class_dict_remapped[i].capitalize()} IoU',
                linestyle='--',
                color=class_colors[i],
                zorder=2
            )
        axes[1].grid(True, zorder=1)
        axes[1].set_title("Validation Global IoU")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU")
        axes[1].legend()
        
        if contains_test_metrics:
            test_labels = (
                ['City IoU', 'Non-City IoU']
                + [f'{class_dict_remapped[i].capitalize()} IoU' for i in range(len(test_class_ious))]
            )
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
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f'{value:.2%}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black'
                )

        plt.tight_layout()
        plt.show()

    def sample(self, run_id: str, data_pairs: list):
        """
        Visualize samples by displaying the input image, predicted mask, and overlap with ground truth.
        
        Each row corresponds to a single sample with three columns:
            1. Input Image
            2. Predicted Mask
            3. Overlap with Ground Truth (including global IoU metric)
        
        Args:
            run_id (str): The W&B run ID to load model weights from.
            data_pairs (list): A list of tuples, each containing:
                            (image_tensor, ground_truth_tensor, scene_label)
        """
        if self.model is None:
            print("No model was provided, cannot sample predictions.")
            return

        model_path = self._get_model_path(run_id)
        if not model_path:
            print(f"No local model file found for run_id {run_id}.")
            return

        try:
            self.model.load_state_dict(
                torch.load(f'models/{model_path}', map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return

        colors = list(mcolors.TABLEAU_COLORS.values())
        class_colors = {i: colors[i % len(colors)] for i in range(5)}
        cmap_pred = mcolors.ListedColormap([class_colors[i] for i in range(5)])

        images = [pair[0] for pair in data_pairs]
        ground_truths = [pair[1] for pair in data_pairs]
        scene_labels = [pair[2] for pair in data_pairs]

        with torch.no_grad():
            predictions = [
                self.model(image.unsqueeze(0).to(self.device)) for image in images
            ]

        num_samples = len(data_pairs)
        fig, axes = plt.subplots(num_samples, 3, figsize=(9, 4 * num_samples))
        
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        legend_patches = [
            mcolors.to_rgba(mcolors.to_rgb(class_colors[i]), alpha=1.0)
            for i in range(5)
        ]
        legend_labels = [f"{class_dict_remapped[i].capitalize()}" for i in range(5)]
        handles = [plt.Line2D([0], [0], marker='s', color='w', label=label,
                            markersize=10, markerfacecolor=legend_patches[i])
                for i, label in enumerate(legend_labels)]
        
        fig.suptitle("Sample Predictions", fontsize=20, y=.92)
        
        fig.legend(handles=handles, title="Classes", loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.9))

        for idx, (pair, pred) in enumerate(zip(data_pairs, predictions)):
            image, ground_truth, scene_label = pair

            unnormalized_image = unnormalize(image, mean, std).permute(1, 2, 0).cpu().numpy()
            
            pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy()

            iou_dict = {}
            for cls in range(5):
                intersection = np.logical_and(pred_mask == cls, ground_truth.cpu().numpy() == cls).sum()
                union = np.logical_or(pred_mask == cls, ground_truth.cpu().numpy() == cls).sum()
                iou = intersection / union if union != 0 else 0.0
                iou_dict[cls] = iou
            global_iou = np.mean(list(iou_dict.values())) * 100

            overlap_mask = np.zeros((*ground_truth.shape, 4))
            for cls in range(5):
                mask = (pred_mask == cls) & (ground_truth.cpu().numpy() == cls)
                overlap_mask[mask] = mcolors.to_rgba(class_colors[cls], alpha=0.5)

            ax_input = axes[idx, 0]
            ax_input.imshow(unnormalized_image)
            ax_input.set_title("Input Image", fontsize=10)
            ax_input.axis("off")
            ax_input.set_ylabel(f"Sample {idx+1}", fontsize=12, rotation=0, labelpad=50, va='center')
            ax_input.text(5, 5, scene_label, fontsize=10, color='white',
                        bbox=dict(facecolor='black', alpha=0.7, pad=2),
                        va='top', ha='left')

            ax_pred = axes[idx, 1]
            im_pred = ax_pred.imshow(pred_mask, cmap=cmap_pred, vmin=0, vmax=4)
            ax_pred.set_title("Predicted Mask", fontsize=10)
            ax_pred.axis("off")

            ax_overlap = axes[idx, 2]
            ax_overlap.imshow(unnormalized_image)
            ax_overlap.imshow(overlap_mask)
            ax_overlap.set_title(f"Overlap with Ground Truth\nGlobal IoU: {global_iou:.2f}%", fontsize=8)
            ax_overlap.axis("off")
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.show()

    def plot_grid_results(self, tuning_grid):
        cost_funcs = sorted(list({v[1] for v in tuning_grid.values()}))
        lrs = sorted(list({v[0] for v in tuning_grid.values()}))
        
        test_iou_matrix = np.zeros((len(cost_funcs), len(lrs)))
        val_iou_matrix = np.zeros((len(cost_funcs), len(lrs)))
        
        api = wandb.Api()
        
        for run_id, (lr, cost_func) in tuning_grid.items():
            row_idx = cost_funcs.index(cost_func)
            col_idx = lrs.index(lr)
            
            try:
                run = api.run(f"{self.entity_name}/{self.project_name}/{run_id}")
                test_iou = run.summary.get("test_global_iou", 0.0)
                val_iou = run.summary.get("val_global_iou", 0.0)
                
                test_iou_matrix[row_idx, col_idx] = test_iou
                val_iou_matrix[row_idx, col_idx] = val_iou
                
            except wandb.errors.CommError as e:
                print(f"Warning: Could not fetch run {run_id}. Error: {e}")
        
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        
        fig.suptitle("Grid Search Results", fontsize=16)
        
        im_test = axes[0].imshow(test_iou_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title("Test Global IoU")
        axes[0].set_yticks(range(len(cost_funcs)))
        axes[0].set_yticklabels(cost_funcs)
        axes[0].set_xticks(range(len(lrs)))
        axes[0].set_xticklabels([str(x) for x in lrs], rotation=45, ha='right')
        axes[0].set_xlabel("Learning Rate")
        axes[0].set_ylabel("Cost Function")
        
        for i in range(len(cost_funcs)):
            for j in range(len(lrs)):
                value = test_iou_matrix[i, j]
                axes[0].text(
                    j, i, f"{value:.2f}",
                    ha='center', va='center',
                    color="white" if value > 0.5 else "black"
                )
        
        fig.colorbar(im_test, ax=axes[0])

        im_val = axes[1].imshow(val_iou_matrix, cmap='Greens', vmin=0, vmax=1)
        axes[1].set_title("Validation Global IoU")
        axes[1].set_yticks(range(len(cost_funcs)))
        axes[1].set_yticklabels(cost_funcs)
        axes[1].set_xticks(range(len(lrs)))
        axes[1].set_xticklabels([str(x) for x in lrs], rotation=45, ha='right')
        axes[1].set_xlabel("Learning Rate")
        axes[1].set_ylabel("Cost Function")
        
        for i in range(len(cost_funcs)):
            for j in range(len(lrs)):
                value = val_iou_matrix[i, j]
                axes[1].text(
                    j, i, f"{value:.2f}",
                    ha='center', va='center',
                    color="white" if value > 0.5 else "black"
                )

        fig.colorbar(im_val, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

    def compare_runs(self, run_ids):
        """
        Compare final test metrics across multiple runs.

        Args:
            run_ids (list of str): W&B run IDs to compare.

        This function retrieves each run, extracts final test IoU metrics, and
        creates a side-by-side comparison. It displays:
          - A bar plot of test_global_iou for each run.
          - A grouped bar plot of test_city_iou vs. test_non_city_iou.
          - A grouped bar plot of all test_iou_class_{i} if available.

        Modify as needed for deeper comparisons.
        """
        api = wandb.Api()
        runs_data = []

        for rid in run_ids:
            try:
                run = api.run(f"{self.entity_name}/{self.project_name}/{rid}")
                summary = run.summary

                test_global_iou = summary.get("test_global_iou", None)
                test_city_iou = summary.get("test_city_iou", None)
                test_non_city_iou = summary.get("test_non_city_iou", None)

                class_ious = []
                for i in range(5):
                    ciou = summary.get(f"test_iou_class_{i}", None)
                    class_ious.append(ciou)

                runs_data.append({
                    "run_id": rid,
                    "run_name": run.name,
                    "test_global_iou": test_global_iou,
                    "test_city_iou": test_city_iou,
                    "test_non_city_iou": test_non_city_iou,
                    "class_ious": class_ious
                })
            except wandb.errors.CommError as e:
                print(f"Warning: Could not fetch run {rid}. Error: {e}")

        x_labels = [d["run_name"] or d["run_id"] for d in runs_data]
        num_runs = len(runs_data)

        global_ious = [d["test_global_iou"] if d["test_global_iou"] is not None else 0.0 
                       for d in runs_data]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Comparison of Multiple Runs", fontsize=16)

        axes[0].bar(x_labels, global_ious, color='skyblue', zorder=3)
        axes[0].set_title("Global Test IoU")
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, axis='y', zorder=0)
        axes[0].set_xticks(range(num_runs))
        axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
        for i, v in enumerate(global_ious):
            axes[0].text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

        city_ious = [d["test_city_iou"] if d["test_city_iou"] is not None else 0.0 
                     for d in runs_data]
        non_city_ious = [d["test_non_city_iou"] if d["test_non_city_iou"] is not None else 0.0 
                         for d in runs_data]
        width = 0.35
        x = np.arange(num_runs)

        axes[1].bar(x - width/2, city_ious, width, label='City IoU', color='gray', zorder=3)
        axes[1].bar(x + width/2, non_city_ious, width, label='Non-City IoU', color='lightgray', zorder=3)
        axes[1].set_title("Test City vs. Non-City IoU")
        axes[1].set_ylim([0, 1])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[1].grid(True, axis='y', zorder=0)
        axes[1].legend()

        for i, (c, n) in enumerate(zip(city_ious, non_city_ious)):
            axes[1].text(i - width/2, c + 0.01, f"{c:.2f}", ha='center', va='bottom', fontsize=9)
            axes[1].text(i + width/2, n + 0.01, f"{n:.2f}", ha='center', va='bottom', fontsize=9)

        class_iou_arrays = [d["class_ious"] for d in runs_data]
        all_colors = list(mcolors.TABLEAU_COLORS.values())
        class_colors = all_colors[:5]

        bar_width = 0.1
        for cls_idx in range(5):
            c_iou = [
                ci[cls_idx] if ci[cls_idx] is not None else 0.0
                for ci in class_iou_arrays
            ]
            offset = (cls_idx - 2) * bar_width
            axes[2].bar(
                x + offset, c_iou, bar_width,
                label=class_dict_remapped[cls_idx],
                color=class_colors[cls_idx],
                zorder=3
            )
            for i, val in enumerate(c_iou):
                axes[2].text(
                    i + offset, val + 0.01, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=8
                )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, axis='y', zorder=0)
        axes[2].set_title("Test IoU per Class")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
        
    def plot_attention_maps(self, data_samples, figsize=(20, 4)):
        """
        Plot the attention maps for each sample in data_samples.
        Each row will display:
        1) the original image (unnormalized), 
        2) up to four attention maps from the AttentionUNet.

        Args:
            data_samples (list): A list of tuples (image_tensor, label_tensor, scene_label)
                                from your BDD100KDataset or a slice of it. 
                                e.g. data_samples = dataset[0:5].
        """
        if not isinstance(self.model, AttentionUNet):
            print("The model is not an AttentionUNet. Aborting attention visualization.")
            return

        num_samples = len(data_samples)
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(figsize[0], figsize[1] * num_samples))

        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        self.model.eval()

        for row_idx, (image_tensor, _, scene_label) in enumerate(data_samples):
            self._attention_maps = []
            self._register_attention_hooks()

            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                _ = self.model(image_tensor)

            unnormalized = unnormalize(
                image_tensor[0].cpu(),
                mean,
                std
            ).permute(1, 2, 0).numpy()

            axes[row_idx, 0].imshow(unnormalized)
            axes[row_idx, 0].set_title("Input Image")
            axes[row_idx, 0].axis("off")
            
            if scene_label is not None:
                axes[row_idx, 0].text(
                    5, 5, scene_label, fontsize=10, color='white',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2),
                    va='top', ha='left'
                )

           
            for col_idx, psi in enumerate(self._attention_maps[:4]):
                psi_upsampled = nn.functional.interpolate(
                    psi[0].unsqueeze(0),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().cpu().numpy()

                im_plot = axes[row_idx, col_idx + 1].imshow(psi_upsampled, cmap='jet')
                axes[row_idx, col_idx + 1].set_title(f"Attention Map {col_idx + 1}")
                axes[row_idx, col_idx + 1].axis("off")

            self._detach_attention_hooks()

        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
        fig.colorbar(im_plot, cax=cbar_ax, orientation='horizontal').set_label("Attention Coefficients")

        fig.suptitle("Attention Maps from Selected Samples", fontsize=16)
        fig.subplots_adjust(top=0.90, bottom=0.15, wspace=0.3)
        plt.show()


    def _register_attention_hooks(self):
        """
        Goes through the model and attaches a forward hook to each AttentionBlock.
        """
        def _attention_hook(module, input, output):
            x, g = input
            g1 = module.W_g(g)
            x1 = module.W_x(x)
            if g1.shape[2:] != x1.shape[2:]:
                g1 = nn.functional.interpolate(g1, size=x1.shape[2:], 
                                               mode="bilinear", align_corners=False)
            psi = module.relu(g1 + x1)
            psi = module.psi(psi)
            
            module.analyzer._attention_maps.append(psi)

        self._attention_hooks = []
        
        for module in self.model.modules():
            if isinstance(module, AttentionBlock):
                module.analyzer = self
                hook = module.register_forward_hook(_attention_hook)
                self._attention_hooks.append(hook)

    def _detach_attention_hooks(self):
        """
        Remove hooks and remove references to self in each AttentionBlock
        (to avoid potential memory leaks if you do this repeatedly).
        """
        for hook in self._attention_hooks:
            hook.remove()

        for module in self.model.modules():
            if isinstance(module, AttentionBlock):
                del module.analyzer
        self._attention_hooks = []

    def _plot_collected_attention_maps(self, image_tensor):
        """
        Once we have a list of attention maps in self._attention_maps,
        display them in a figure. Each map is shape [B, 1, H, W].
        We can optionally upsample them to the original input size.
        """
        num_maps = len(self._attention_maps)
        if num_maps == 0:
            print("No attention maps found.")
            return

        fig, axes = plt.subplots(1, num_maps, figsize=(4 * num_maps, 4), gridspec_kw={"wspace": 0.3})
        if num_maps == 1:
            axes = [axes]

        _, _, H, W = image_tensor.shape
        im = None

        for i, (psi, ax) in enumerate(zip(self._attention_maps, axes)):
            psi_upsampled = nn.functional.interpolate(
                psi[0].unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            )
            psi_upsampled = psi_upsampled.squeeze().cpu().numpy()
            im = ax.imshow(psi_upsampled, cmap='jet')
            ax.set_title(f"Attention Map {i+1}")
            ax.axis('off')

        cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Attention Coefficients", fontsize=12)

        fig.suptitle("Attention Maps from Attention Blocks", fontsize=16)

        fig.subplots_adjust(top=0.85, bottom=0.2, left=0.05, right=0.95, wspace=0.3)
        plt.show()

    def _get_model_path(self, run_id: str):
        print(f"\nSearching for model weights for run {run_id}...")
        models = os.listdir('models')
        model_path = None
        for model in models:
            run_name_parts = model.split('_')
            run_id_in_name = run_name_parts[-1].split('.')[0]
            if run_id_in_name == run_id:
                model_path = model
                print(f'Found model: {model_path}!')
                break
        return model_path
        
    def _load_model_weights(self, run_id: str):
        if self.model is None:
            print("No model to load weights into.")
            return
        model_path = self._get_model_path(run_id)
        if not model_path:
            print(f"No local model file found for run_id {run_id}.")
            return
        
        self.model.load_state_dict(
            torch.load(f'models/{model_path}', map_location=self.device, weights_only=True)
        )
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