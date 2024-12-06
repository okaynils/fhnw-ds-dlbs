import os
import torch
import wandb
import torch.nn as nn
from torchmetrics import JaccardIndex

class Trainer:
    def __init__(self, model, criterion, optimizer, epochs, seed, device, verbose, run_name, weight_init=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.weight_init = weight_init
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.run_name = run_name

        self.model.to(self.device)

        self.iou_metric = JaccardIndex(num_classes=19, task="multiclass").to(self.device)

        self._set_seed(self.seed)
        
        if not self._precheck():
            wandb.init(project="dlbs", name=self.run_name)        
            self.run_id = wandb.run.id
        else:
            print(f'Model trainer was already initialized. Skipping wandb initialization.')

        self.best_val_loss = float('inf')

        os.makedirs("models", exist_ok=True)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _initialize_weights(self):
        if self.weight_init:
            self.model.apply(self.weight_init)
        else:
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _save_model(self, val_loss):
        """
        Save the model if the current validation loss is lower than the best validation loss.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model_name = f"{self.run_name}_{self.run_id}.pth"
            save_path = os.path.join("models", model_name)
            
            torch.save(self.model.state_dict(), save_path)
            
            wandb.save(model_name)

            if self.verbose:
                print(f"Model saved to {save_path} with val_loss {val_loss:.4f}")

    def _train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        self.iou_metric.reset()

        for images, labels, _ in train_loader:
            images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Calculate IoU, ignoring 255-labeled pixels
            _, predicted = outputs.max(1)
            mask = labels != 255  # Create a mask for valid pixels
            self.iou_metric.update(predicted[mask], labels[mask])

            # Clear cache to manage GPU memory
            del images, labels, outputs
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = self.iou_metric.compute().item()

        return epoch_loss, epoch_iou

    def _validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        self.iou_metric.reset()

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Calculate IoU, ignoring 255-labeled pixels
                _, predicted = outputs.max(1)
                mask = labels != 255  # Create a mask for valid pixels
                self.iou_metric.update(predicted[mask], labels[mask])

                # Clear cache to manage GPU memory
                del images, labels, outputs
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(val_loader)
        epoch_iou = self.iou_metric.compute().item()

        return epoch_loss, epoch_iou

    def _precheck(self):
        if not os.path.exists("models"):
            os.makedirs("models")
        models = os.listdir("models")
        for model in models:
            if self.run_name in model:
                return True
        return False

    def run(self, train_loader, val_loader):
        if not self._precheck():
            self._initialize_weights()

            for epoch in range(self.epochs):
                train_loss, train_iou = self._train_epoch(train_loader)
                val_loss, val_iou = self._validate_epoch(val_loader)

                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} - "
                        f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "val_loss": val_loss,
                    "val_iou": val_iou
                })

                # Save the model if it's the best one so far
                self._save_model(val_loss)
        else:
            print(f'Model {self.run_name} already exists! Skipping training.')

    def test(self, test_loader):
        test_loss, test_iou = self._validate_epoch(test_loader)

        print(f"Test Loss: {test_loss:.4f} - Test IoU: {test_iou:.4f}")
        wandb.log({
            "test_loss": test_loss,
            "test_iou": test_iou
        })