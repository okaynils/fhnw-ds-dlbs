import os

import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

MODELS_PATH = "models"

class Analyzer:
    def __init__(self, run_id, project_name, entity_name, test_dataset, model, device="cpu"):
        """
        Initializes the Analyzer object with run details.

        Parameters:
            run_id (str): The ID of the wandb run.
            project_name (str): The name of the wandb project.
            entity_name (str): The name of the wandb entity.
        """
        self.run_id = run_id
        self.project_name = project_name
        self.entity_name = entity_name
        self.history = None
        self.test_dataset = test_dataset
        self.device = device
        self.model = model.to(device)

    def _load_model(self):
        models = os.listdir(MODELS_PATH)
        
        for model in models:
            if self.run_id in model:
                model_state = torch.load(os.path.join(MODELS_PATH, model))
                
                self.model.load_state_dict(model_state)
                
                return self.model

    def _test_n_samples(self, test_sample_ids: list = None):
        model = self._load_model()
        for idx in test_sample_ids:
            image, label, scene = self.test_dataset[idx]
            
            print(f"Image shape: {image.shape}")
            print(f"Label shape: {label.shape}")
            print(f"Scene: {scene}")
            
            model.eval()
            with torch.no_grad():
                image = image.unsqueeze(0).to(self.device)
                output = model(image)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                
                cmap = plt.get_cmap('tab20')
                unique_classes = list(range(19))  # Adjust if there are fewer/more classes
                legend_patches = [mpatches.Patch(color=cmap(i / len(unique_classes)), label=f'Class {i}') for i in unique_classes]

                print(pred, label)

                plt.figure(figsize=(12, 6))

                # Ground Truth subplot
                plt.subplot(1, 2, 1)
                plt.imshow(label, cmap='tab20')
                plt.title("Ground Truth")
                plt.axis('off')
                plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1.0), title="Classes")

                # Prediction subplot
                plt.subplot(1, 2, 2)
                plt.imshow(pred, cmap='tab20')
                plt.title("Prediction")
                plt.axis('off')
                plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1.0), title="Classes")

                plt.tight_layout()
                plt.show()

    def _fetch_data(self):
        """
        Fetches the run data from wandb using the API.
        """
        # Initialize API
        api = wandb.Api()

        # Fetch the run
        try:
            run = api.run(f"{self.entity_name}/{self.project_name}/{self.run_id}")
            self.history = run.history()
        except wandb.errors.CommError as e:
            raise ValueError(f"Error fetching run: {e}")

    def plot(self):
        """
        Plots the train and validation loss for the specified wandb run.
        """
        if self.history is None:
            self._fetch_data()

        # Check if 'train_loss' and 'val_loss' exist in the history
        if 'train_loss' not in self.history or 'val_loss' not in self.history:
            raise ValueError("The run does not have 'train_loss' and/or 'val_loss' logged.")

        # Plot train and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss', linestyle='--')
        plt.title(f"Train and Validation Loss for Run {self.run_id}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()
