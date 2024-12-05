import os

import hydra
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from data.bdd100k_dataset import BDD100KDataset
from data.utils import custom_split_dataset_with_det

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))
    
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

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    criterion = hydra.utils.instantiate(cfg.criterion)

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
