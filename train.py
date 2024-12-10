import os

import hydra
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from data.bdd100k_dataset import BDD100KDataset
from data.utils import *
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from core.focal_loss import FocalLoss

import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg), flush=True)
    
    logger.info('Test')

    dataset_splits = custom_split_dataset_with_det(
        base_data_path=cfg.dataset.images_dir,
        base_labels_path=cfg.dataset.labels_dir,
        det_train_path=cfg.dataset.det_train_path,
        det_val_path=cfg.dataset.det_val_path,
        test_size=cfg.dataset.test_size,
        random_state=cfg.seed
    )

    train_dataset = BDD100KDataset(
        images_dir=dataset_splits['train']['data_folder'],
        labels_dir=dataset_splits['train']['labels_folder'],
        filenames=dataset_splits['train']['image_filenames'],
        transform=hydra.utils.instantiate(cfg.dataset.transform),
        target_transform=hydra.utils.instantiate(cfg.dataset.target_transform),
        scene_info=dataset_splits['train']['scene_map']
    )

    val_dataset = BDD100KDataset(
        images_dir=dataset_splits['val']['data_folder'],
        labels_dir=dataset_splits['val']['labels_folder'],
        filenames=dataset_splits['val']['image_filenames'],
        transform=hydra.utils.instantiate(cfg.dataset.transform),
        target_transform=hydra.utils.instantiate(cfg.dataset.target_transform),
        scene_info=dataset_splits['val']['scene_map']
    )

    test_dataset = BDD100KDataset(
        images_dir=dataset_splits['test']['data_folder'],
        labels_dir=dataset_splits['test']['labels_folder'],
        filenames=dataset_splits['test']['image_filenames'],
        transform=hydra.utils.instantiate(cfg.dataset.transform),
        target_transform=hydra.utils.instantiate(cfg.dataset.target_transform),
        scene_info=dataset_splits['test']['scene_map']
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size)

    class_weights_file = cfg.dataset.class_weights_file
    num_classes = cfg.model.num_classes
    class_weights = None

    if os.path.exists(class_weights_file):
        print(f"Loading class weights from {class_weights_file}", flush=True)
        class_weights = torch.load(class_weights_file, map_location=cfg.device)
    else:
        print("Calculating class weights...")
        all_masks = [sample[1] for sample in train_dataset]

        flat_labels = np.concatenate([np.array(mask).flatten() for mask in all_masks])

        classes = list(range(0, 19))
        classes.append(255)

        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.array(classes), y=flat_labels)[:-1], dtype=torch.float32, device=cfg.device
        )
        
        torch.save(class_weights, class_weights_file)
        print(f"Class weights saved to {class_weights_file}")

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    if cfg.criterion._target_ == 'torch.nn.CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    if cfg.criterion._target_ == 'core.FocalLoss':
        criterion = FocalLoss(gamma=cfg.criterion.gamma, weights=class_weights, ignore_index=255)

    print(f'--- Model Configuration of {cfg.model._target_} ---', flush=True)
    print(model)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.epochs,
        seed=cfg.seed,
        device=cfg.device,
        verbose=cfg.verbose,
        run_name=cfg.run_name
    )

    trainer.run(train_loader, val_loader)
    trainer.test(test_loader)

if __name__ == "__main__":
    main()
    