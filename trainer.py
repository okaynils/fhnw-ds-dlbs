import os
import logging

import torch
import wandb
import torch.nn as nn
from torchmetrics import JaccardIndex
from core.focal_loss import FocalLoss

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, optimizer, epochs, seed, device, verbose, run_name, weight_init=None, early_stopping_patience=5, n_classes=19):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.weight_init = weight_init
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.run_name = run_name
        self.early_stopping_patience = early_stopping_patience
        self.n_classes = n_classes

        self.model.to(self.device)

        self.iou_metric_global = JaccardIndex(num_classes=self.n_classes, task="multiclass", average="macro").to(self.device)
        self.iou_metric_per_class = JaccardIndex(num_classes=self.n_classes, task="multiclass", average="none").to(self.device)

        self._set_seed(self.seed)

        if not self._precheck():
            wandb.init(project="dlbs", name=self.run_name)
            self.run_id = wandb.run.id
        else:
            logger.info(f'Model trainer was already initialized. Skipping wandb initialization.')

        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

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
                logger.info(f"Model saved to {save_path} with val_loss {val_loss:.4f}")

    def _prepare_inputs(self, outputs, labels):
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            return outputs, labels
        elif isinstance(self.criterion, FocalLoss):
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # Reshape outputs and labels for FocalLoss
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])
            labels = labels.view(-1)
            return outputs, labels
        else:
            raise ValueError(f"Unsupported criterion type: {type(self.criterion)}")

    def _train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        self.iou_metric_global.reset()
        self.iou_metric_per_class.reset()

        for images, labels, _ in train_loader:
            images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            outputs, labels = self._prepare_inputs(outputs, labels)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            mask = labels != 255
            self.iou_metric_global.update(predicted[mask], labels[mask])
            self.iou_metric_per_class.update(predicted[mask], labels[mask])

            del images, labels, outputs
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        global_iou = self.iou_metric_global.compute().item()
        per_class_iou = self.iou_metric_per_class.compute()

        return epoch_loss, global_iou, per_class_iou

    def _validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        self.iou_metric_global.reset()
        self.iou_metric_per_class.reset()

        with torch.no_grad():
            for images, labels, scene in val_loader:
                images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)

                outputs = self.model(images)
                outputs, labels = self._prepare_inputs(outputs, labels)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                mask = labels != 255
                self.iou_metric_global.update(predicted[mask], labels[mask])
                self.iou_metric_per_class.update(predicted[mask], labels[mask])

                del images, labels, outputs
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(val_loader)
        global_iou = self.iou_metric_global.compute().item()
        per_class_iou = self.iou_metric_per_class.compute()

        return epoch_loss, global_iou, per_class_iou

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
                train_loss, train_global_iou, train_per_class_iou = self._train_epoch(train_loader)
                val_loss, val_global_iou, val_per_class_iou = self._validate_epoch(val_loader)

                if self.verbose:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Global IoU: {train_global_iou:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Global IoU: {val_global_iou:.4f}")

                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_global_iou": train_global_iou,
                    "val_loss": val_loss,
                    "val_global_iou": val_global_iou,
                    **{f"train_iou_class_{i}": train_per_class_iou[i].item() for i in range(len(train_per_class_iou))},
                    **{f"val_iou_class_{i}": val_per_class_iou[i].item() for i in range(len(val_per_class_iou))}
                })

                self._save_model(val_loss)

                if val_loss <= self.best_val_loss:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                        break

        else:
            logger.info(f'Model {self.run_name} already exists! Skipping training.')

    def test(self, test_loader):
        if not wandb.run:
            wandb.init(project="dlbs", name=f"{self.run_name}_test", reinit=True)

        self.model.eval()
        running_loss = 0.0
        self.iou_metric_global.reset()
        self.iou_metric_per_class.reset()

        city_iou_metric = JaccardIndex(num_classes=self.n_classes, task="multiclass", average="macro").to(self.device)
        non_city_iou_metric = JaccardIndex(num_classes=self.n_classes, task="multiclass", average="macro").to(self.device)

        with torch.no_grad():
            for images, labels, scene in test_loader:
                images, labels = images.to(self.device), labels.to(self.device, dtype=torch.long)

                outputs = self.model(images)
                outputs, labels = self._prepare_inputs(outputs, labels)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                mask = labels != 255

                self.iou_metric_global.update(predicted[mask], labels[mask])
                self.iou_metric_per_class.update(predicted[mask], labels[mask])

                for i, scene_label in enumerate(scene):
                    scene_mask = mask[i]
                    if "city street" in scene_label:
                        city_iou_metric.update(predicted[i][scene_mask], labels[i][scene_mask])
                    else:
                        non_city_iou_metric.update(predicted[i][scene_mask], labels[i][scene_mask])

                del images, labels, outputs
                torch.cuda.empty_cache()

        epoch_loss = running_loss / len(test_loader)
        global_iou = self.iou_metric_global.compute().item()
        per_class_iou = self.iou_metric_per_class.compute()
        city_iou = city_iou_metric.compute().item()
        non_city_iou = non_city_iou_metric.compute().item()

        logger.info(f"Test Loss: {epoch_loss:.4f} - Test Global IoU: {global_iou:.4f} - "
                    f"City IoU: {city_iou:.4f} - Non-City IoU: {non_city_iou:.4f}")

        wandb.log({
            "test_loss": epoch_loss,
            "test_global_iou": global_iou,
            "test_city_iou": city_iou,
            "test_non_city_iou": non_city_iou,
            **{f"test_iou_class_{i}": per_class_iou[i].item() for i in range(len(per_class_iou))}
        })
