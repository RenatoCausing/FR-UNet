from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
from bunch import Bunch
from ruamel.yaml import YAML
import models
from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch, get_torch_device
from torch.utils.data import DataLoader, ConcatDataset
from utils.metrics import save_confusion_matrix
from test import main as run_test


def _load_config(config_path: Path) -> Bunch:
    yaml = YAML(typ='safe')
    yaml.pure = True
    with config_path.open('r', encoding='utf-8') as fp:
        data = yaml.load(fp)
    return Bunch(data)


def _build_loaders(dataset_path: Path, batch_size: int, num_workers: int,
                   with_val: bool, val_split: float, include_holdout: bool,
                   pin_memory: bool
                   ) -> Tuple[DataLoader, DataLoader | None, Dict[str, int]]:
    base_train = vessel_dataset(str(dataset_path), mode="training",
                                split=val_split if with_val else None)
    val_dataset = vessel_dataset(str(dataset_path), mode="training",
                                 split=val_split, is_val=True) if with_val else None
    holdout_patches = 0
    if include_holdout:
        holdout_dir = dataset_path / "holdout_pro"
        if not holdout_dir.exists():
            raise FileNotFoundError(
                f"holdout_pro directory not found at {holdout_dir}. Run data_process.py with --modes holdout first.")
        if not any(holdout_dir.iterdir()):
            raise RuntimeError(f"holdout_pro at {holdout_dir} is empty – run the holdout preprocessing step.")
        holdout_dataset = vessel_dataset(str(dataset_path), mode="holdout")
        train_dataset = ConcatDataset([base_train, holdout_dataset])
        holdout_patches = len(holdout_dataset)
    else:
        train_dataset = base_train
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
        'persistent_workers': num_workers > 0
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers // 2),
            pin_memory=loader_kwargs['pin_memory'],
            drop_last=False,
            persistent_workers=(num_workers // 2) > 0
        )
    counts = {
        "train_patches": len(train_dataset),
        "base_training_patches": len(base_train),
        "holdout_patches": holdout_patches,
        "val_patches": len(val_dataset) if val_dataset is not None else 0
    }
    return train_loader, val_loader, counts


def _resolve_checkpoint(trainer: Trainer) -> Path:
    if trainer.latest_checkpoint and os.path.exists(trainer.latest_checkpoint):
        return Path(trainer.latest_checkpoint)
    checkpoint_dir = Path(trainer.checkpoint_dir)
    candidates = sorted(checkpoint_dir.glob('checkpoint-epoch*.pth'))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints saved under {checkpoint_dir}")
    return candidates[-1]


def run_training_pipeline(*, include_holdout: bool, dataset_path: str, batch_size: int,
                          num_workers: int, with_val: bool, val_split: float,
                          config_path: str, output_root: str, show_preds: bool = False,
                          device_preference: str | None = None,
                          pin_memory: bool = False
                          ) -> Dict[str, str | int | float]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    CFG = _load_config(config_file)
    seed_torch()
    device = get_torch_device(device_preference)
    train_loader, val_loader, counts = _build_loaders(
        dataset_root, batch_size, num_workers, with_val, val_split, include_holdout, pin_memory)
    logger.info(f"Training patches: {counts['train_patches']} (base={counts['base_training_patches']}, "
                f"holdout={counts['holdout_patches']})")
    if counts['val_patches']:
        logger.info(f"Validation patches: {counts['val_patches']}")
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG, device=device)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    trainer.train()
    checkpoint_path = _resolve_checkpoint(trainer)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_root = Path(output_root)
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_copy = run_dir / 'config.yaml'
    shutil.copy2(config_file, cfg_copy)
    checkpoint_copy = run_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, checkpoint_copy)
    checkpoints_dir = run_dir / 'checkpoints'
    shutil.copytree(trainer.checkpoint_dir, checkpoints_dir)
    logger.info(f"Copied latest checkpoint to {checkpoint_copy}")
    logger.info(f"Mirrored full checkpoint directory to {checkpoints_dir}")
    test_results = run_test(str(dataset_root), str(checkpoint_copy), CFG, show_preds, device=device)
    metrics_path = run_dir / 'test_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as fp:
        json.dump(test_results, fp, indent=2)
    confusion_paths = {}
    confusion = test_results.get('confusion')
    if confusion:
        raw_conf_path = run_dir / 'confusion_matrix.png'
        norm_conf_path = run_dir / 'confusion_matrix_normalized.png'
        save_confusion_matrix(confusion, raw_conf_path)
        save_confusion_matrix(confusion, norm_conf_path, normalize=True)
        confusion_paths = {
            'confusion_matrix': str(raw_conf_path),
            'confusion_matrix_normalized': str(norm_conf_path)
        }
    summary = {
        'regime': 'train_100' if include_holdout else 'train_85',
        'dataset_root': str(dataset_root.resolve()),
        'total_training_patches': counts['train_patches'],
        'base_training_patches': counts['base_training_patches'],
        'holdout_patches': counts['holdout_patches'],
        'validation_patches': counts['val_patches'],
        'epochs': CFG.epochs,
        'checkpoint_dir': str(Path(trainer.checkpoint_dir).resolve()),
        'copied_checkpoint_dir': str(checkpoints_dir.resolve()),
        'copied_checkpoint': str(checkpoint_copy.resolve()),
        'device': device.type,
        'metrics_file': str(metrics_path.resolve()),
        **confusion_paths
    }
    summary_path = run_dir / 'run_summary.json'
    with summary_path.open('w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)
    summary['run_summary'] = str(summary_path.resolve())
    logger.info(f"Pipeline artifacts written to {run_dir}")
    return summary

__all__ = ['run_training_pipeline']
