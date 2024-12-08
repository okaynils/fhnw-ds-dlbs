import os

import wandb
import matplotlib.pyplot as plt
import torch

MODELS_PATH = "models"

class Analyzer:
    def __init__(self, run_id, project_name, entity_name, test_dataset):
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

    def _load_model(self):
        models = os.listdir(MODELS_PATH)
        
        for model in models:
            if self.run_id in model:
                model = torch.load(os.path.join(MODELS_PATH, model))
                return model

    def _test_n_samples(self, test_sample_ids: list = None):
        model = self._load_model()
        
        samples = self.test_dataset[test_sample_ids]
        
        for image, label in samples:
            print(f"Image shape: {image.shape}")
            print(f"Label shape: {label.shape}")
        

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
