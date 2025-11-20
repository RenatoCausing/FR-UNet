"""Legacy-compatible preprocessing pipeline for FR-UNet datasets.

This module mirrors the original data_process.py script from the paper while
still playing nicely with the current repo layout/utilities. It keeps the
legacy preprocessing decisions (per-dataset paths, normalization, patching,
and test image exports) so we can reproduce paper numbers side-by-side with
newer pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import YAML
from torchvision.transforms import Grayscale, Normalize, ToTensor

from utils.helpers import dir_exists, remove_files


# ------------------------------
# Shared helpers
# ------------------------------


def _reset_dir(path: Path) -> None:
    path = Path(path)
    if path.exists():
        remove_files(str(path))
    else:
        path.mkdir(parents=True, exist_ok=True)


_GRAYSCALE = Grayscale(num_output_channels=1)
_TO_TENSOR = ToTensor()
_SAVE_PREVIEW_DIR = Path("save_picture")
_SAVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _build_lookup_with_suffix(folder: Path, suffix: str) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if not folder.exists():
        return lookup
    for file in folder.iterdir():
        if not file.is_file() or not _is_image_file(file):
            continue
        stem = file.stem
        if not stem.endswith(suffix):
            continue
        lookup[stem[: -len(suffix)]] = file
    return lookup


def _dataset_prefix(stem: str) -> str:
    return stem.split("_", 1)[0].upper()


def _sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    prefix, _, remainder = stem.partition("_")
    digits = "".join(ch for ch in remainder if ch.isdigit())
    numeric = int(digits) if digits else 0
    return (prefix, numeric, stem)


def data_process(dataset_path: Path | str, dataset_name: str,
                 patch_size: int, stride: int, mode: str) -> None:
    """Run the legacy preprocessing routine for a specific dataset/mode."""
    dataset_path = Path(dataset_path)
    save_path = dataset_path / f"{mode}_pro"
    dir_exists(str(save_path))
    remove_files(str(save_path))

    if dataset_name == "DRIVE":
        img_path = dataset_path / mode / "images"
        gt_path = dataset_path / mode / "1st_manual"
        file_list = sorted(os.listdir(img_path))
    elif dataset_name == "CHASEDB1":
        img_path = dataset_path
        file_list = sorted(os.listdir(img_path))
    elif dataset_name == "STARE":
        img_path = dataset_path / "stare-images"
        gt_path = dataset_path / "labels-ah"
        file_list = sorted(os.listdir(img_path))
    elif dataset_name == "DCA1":
        img_path = dataset_path / "Database_134_Angiograms"
        file_list = sorted(os.listdir(img_path))
    elif dataset_name == "CHUAC":
        img_path = dataset_path / "Original"
        gt_path = dataset_path / "Photoshop"
        file_list = sorted(os.listdir(img_path))
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")

    img_list: List[torch.Tensor] = []
    gt_list: List[torch.Tensor] = []

    for idx, file in enumerate(file_list):
        if dataset_name == "DRIVE":
            img = Image.open(img_path / file)
            gt = Image.open(gt_path / f"{file[:2]}_manual1.gif")
            img_list.append(_image_to_tensor(img))
            gt_list.append(_image_to_tensor(gt))

        elif dataset_name == "CHASEDB1":
            if len(file) == 13:
                sample_id = int(file[6:8])
                if mode == "training" and sample_id <= 10:
                    img = Image.open(img_path / file)
                    gt = Image.open(img_path / f"{file[:9]}_1stHO.png")
                    img_list.append(_image_to_tensor(img))
                    gt_list.append(_image_to_tensor(gt))
                elif mode == "test" and sample_id > 10:
                    img = Image.open(img_path / file)
                    gt = Image.open(img_path / f"{file[:9]}_1stHO.png")
                    img_list.append(_image_to_tensor(img))
                    gt_list.append(_image_to_tensor(gt))

        elif dataset_name == "DCA1":
            if len(file) <= 7 and file.lower().endswith((".png", ".pgm", ".jpg", ".bmp")):
                numeric = int(Path(file).stem)
                if mode == "training" and numeric <= 100:
                    img, gt = _load_dca_pair(img_path, file)
                    img_list.append(_array_to_tensor(img))
                    gt_list.append(_array_to_tensor(gt))
                elif mode == "test" and numeric > 100:
                    img, gt = _load_dca_pair(img_path, file)
                    img_list.append(_array_to_tensor(img))
                    gt_list.append(_array_to_tensor(gt))

        elif dataset_name == "CHUAC":
            numeric = int(Path(file).stem)
            if mode == "training" and numeric <= 20:
                img, gt = _load_chuac_pair(img_path, gt_path, file)
                img_list.append(_array_to_tensor(img))
                gt_list.append(_array_to_tensor(gt))
            elif mode == "test" and numeric > 20:
                img, gt = _load_chuac_pair(img_path, gt_path, file)
                img_list.append(_array_to_tensor(img))
                gt_list.append(_array_to_tensor(gt))

        elif dataset_name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(img_path / file)
                gt = Image.open(gt_path / f"{file[:6]}.ah.ppm")
                cv2.imwrite(str(_SAVE_PREVIEW_DIR / f"{idx}img.png"), np.array(img))
                cv2.imwrite(str(_SAVE_PREVIEW_DIR / f"{idx}gt.png"), np.array(gt))
                img_list.append(_image_to_tensor(img))
                gt_list.append(_image_to_tensor(gt))

    if not img_list:
        raise RuntimeError(f"No samples collected for {dataset_name} ({mode}).")

    img_list = normalization(img_list)
    if mode == "training":
        img_patches = get_patch(img_list, patch_size, stride)
        gt_patches = get_patch(gt_list, patch_size, stride)
        save_patch(img_patches, save_path, "img_patch", dataset_name)
        save_patch(gt_patches, save_path, "gt_patch", dataset_name)
    elif mode == "test":
        if dataset_name != "CHUAC":
            img_list = get_square(img_list, dataset_name)
            gt_list = get_square(gt_list, dataset_name)
        save_each_image(img_list, save_path, "img", dataset_name)
        save_each_image(gt_list, save_path, "gt", dataset_name)
    else:
        raise ValueError("mode must be 'training' or 'test'")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = _GRAYSCALE(image)
    return _TO_TENSOR(image)


def _array_to_tensor(array: np.ndarray) -> torch.Tensor:
    return _TO_TENSOR(array)


def _load_dca_pair(img_path: Path, file: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(img_path / file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing DCA image {file}")
    gt = cv2.imread(str(img_path / f"{Path(file).stem}_gt.pgm"), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Missing DCA mask for {file}")
    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
    return img, gt


def _load_chuac_pair(img_path: Path, gt_path: Path, file_name: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(img_path / file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing CHUAC image {file_name}")
    numeric = int(Path(file_name).stem)
    suffix = "PNG" if 11 <= numeric <= 17 else "png"
    gt_file = gt_path / f"angio{Path(file_name).stem}ok.{suffix}"
    gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Missing CHUAC mask {gt_file.name}")
    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(_SAVE_PREVIEW_DIR / f"{numeric}img.png"), img)
    cv2.imwrite(str(_SAVE_PREVIEW_DIR / f"{numeric}gt.png"), gt)
    return img, gt


def get_square(img_list: Sequence[torch.Tensor], name: str) -> List[torch.Tensor]:
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    else:
        shape = img_list[0].shape[-1]
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape - w, 0, shape - h), 0)
    return [pad(img) for img in img_list]


def get_patch(imgs_list: Sequence[torch.Tensor], patch_size: int, stride: int) -> List[torch.Tensor]:
    patch_size = int(patch_size)
    stride = int(stride)
    image_list: List[torch.Tensor] = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for tensor in imgs_list:
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
        patches = padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        patches = patches.permute(1, 2, 0, 3, 4)
        patches = patches.contiguous().view(-1, patches.shape[2], patch_size, patch_size)
        image_list.extend(patches)
    return image_list


def _extract_patches_single(tensor: torch.Tensor, patch_size: int, stride: int) -> List[torch.Tensor]:
    patch_size = int(patch_size)
    stride = int(stride)
    c, h, w = tensor.shape
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    patches = padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    num_patches = patches.shape[0] * patches.shape[1]
    patches = patches.view(num_patches, c, patch_size, patch_size)
    return [patch for patch in patches]


def save_patch(imgs_list: Iterable[torch.Tensor], path: Path, prefix: str, name: str) -> None:
    for i, tensor in enumerate(imgs_list):
        with (path / f"{prefix}_{i}.pkl").open("wb") as file:
            pickle.dump(_tensor_to_numpy(tensor), file)
        print(f"save {name} {prefix} : {prefix}_{i}.pkl")


def save_each_image(imgs_list: Sequence[torch.Tensor], path: Path, prefix: str, name: str) -> None:
    for i, tensor in enumerate(imgs_list):
        with (path / f"{prefix}_{i}.pkl").open("wb") as file:
            pickle.dump(_tensor_to_numpy(tensor), file)
        print(f"save {name} {prefix} : {prefix}_{i}.pkl")


def normalization(imgs_list: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    stacked = torch.cat(imgs_list, dim=0)
    mean = stacked.mean().item()
    std = stacked.std().item()
    normal_list: List[torch.Tensor] = []
    for tensor in imgs_list:
        norm_tensor = Normalize([mean], [std])(tensor)
        norm_tensor = (norm_tensor - norm_tensor.min()) / (norm_tensor.max() - norm_tensor.min() + 1e-8)
        normal_list.append(norm_tensor)
    return normal_list


def _compute_mean_std(tensors: Sequence[torch.Tensor]) -> Tuple[float, float]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_pixels = 0
    for tensor in tensors:
        tensor_float = tensor.float()
        total_sum += tensor_float.sum().item()
        total_sumsq += torch.square(tensor_float).sum().item()
        total_pixels += tensor_float.numel()
    if total_pixels == 0:
        return 0.0, 1.0
    mean = total_sum / total_pixels
    variance = max((total_sumsq / total_pixels) - mean ** 2, 1e-8)
    std = variance ** 0.5
    return mean, std


def _normalize_with_stats(imgs_list: Sequence[torch.Tensor], stats: Tuple[float, float] | None = None) -> Tuple[List[torch.Tensor], Tuple[float, float]]:
    if not imgs_list:
        return [], (0.0, 1.0)
    if stats is None:
        mean, std = _compute_mean_std(imgs_list)
    else:
        mean, std = stats
    std = max(std, 1e-8)
    normalized: List[torch.Tensor] = []
    for tensor in imgs_list:
        norm_tensor = Normalize([mean], [std])(tensor)
        tensor_min = torch.min(norm_tensor)
        tensor_max = torch.max(norm_tensor)
        norm_tensor = (norm_tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        normalized.append(norm_tensor)
    return normalized, (mean, std)


FINAL_PREFIX_STRIDES = {"DRIVE": 12, "STARE": 14}


def _load_final_combined_samples(final_path: Path) -> List[dict]:
    original_dir = final_path / "Original"
    segmented_dir = final_path / "Segmented"
    mask_dir = final_path / "Mask"
    if not original_dir.exists() or not segmented_dir.exists():
        raise FileNotFoundError("Final dataset must contain Original and Segmented directories.")
    seg_lookup = _build_lookup_with_suffix(segmented_dir, "_segment")
    mask_lookup = _build_lookup_with_suffix(mask_dir, "_mask") if mask_dir.exists() else {}
    samples: List[dict] = []
    for img_path in sorted(original_dir.iterdir(), key=_sort_key):
        if not img_path.is_file() or not _is_image_file(img_path):
            continue
        stem = img_path.stem
        dataset = _dataset_prefix(stem)
        seg_path = seg_lookup.get(stem)
        if seg_path is None:
            continue
        image = Image.open(img_path).convert("L")
        img_tensor = _TO_TENSOR(image)
        mask = np.array(Image.open(seg_path).convert("L"))
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask_tensor = _array_to_tensor(mask)
        fov_mask = None
        mask_path = mask_lookup.get(stem)
        if mask_path is not None:
            fov_mask = np.array(Image.open(mask_path).convert("L"))
            fov_tensor = _array_to_tensor(np.where(fov_mask > 0, 255, 0).astype(np.uint8))
            img_tensor = img_tensor * fov_tensor
            mask_tensor = mask_tensor * fov_tensor
        samples.append({
            "id": stem,
            "dataset": dataset,
            "image": img_tensor,
            "gt": mask_tensor,
            "stride": FINAL_PREFIX_STRIDES.get(dataset, 6),
        })
    if not samples:
        raise RuntimeError(f"No samples discovered under {final_path}")
    return samples


def _split_combined_train_test(samples: Sequence[dict], test_per_dataset: int) -> Tuple[List[dict], List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        buckets[sample["dataset"]].append(sample)
    train: List[dict] = []
    test: List[dict] = []
    for dataset, group in buckets.items():
        group_sorted = sorted(group, key=lambda s: s["id"])
        quota = min(test_per_dataset, len(group_sorted))
        test.extend(group_sorted[:quota])
        train.extend(group_sorted[quota:])
    if not train:
        raise RuntimeError("No training samples available after reserving test allocations.")
    return train, test


def _apply_padding_to_test(samples: List[dict]) -> None:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        buckets[sample["dataset"]].append(sample)
    for dataset, group in buckets.items():
        imgs = [sample["image"] for sample in group]
        gts = [sample["gt"] for sample in group]
        padded_imgs = get_square(imgs, dataset)
        padded_gts = get_square(gts, dataset)
        for sample, img_pad, gt_pad in zip(group, padded_imgs, padded_gts):
            sample["image"] = img_pad
            sample["gt"] = gt_pad


def _save_combined_training(train_samples: Sequence[dict], output_dir: Path, patch_size: int) -> None:
    train_dir = output_dir / "training_pro"
    _reset_dir(train_dir)
    idx = 0
    for sample in train_samples:
        stride = int(sample.get("stride", 6))
        img_patches = _extract_patches_single(sample["image"], patch_size, stride)
        gt_patches = _extract_patches_single(sample["gt"], patch_size, stride)
        for img_patch, gt_patch in zip(img_patches, gt_patches):
            with (train_dir / f"img_patch_{idx}.pkl").open("wb") as fp:
                pickle.dump(_tensor_to_numpy(img_patch), fp)
            with (train_dir / f"gt_patch_{idx}.pkl").open("wb") as fp:
                pickle.dump(_tensor_to_numpy(gt_patch), fp)
            if idx % 1000 == 0:
                print(f"save COMBINED patch pair : idx {idx}")
            idx += 1


def _save_combined_test(test_samples: Sequence[dict], output_dir: Path) -> None:
    test_dir = output_dir / "test_pro"
    _reset_dir(test_dir)
    img_dir = test_dir / "img"
    gt_dir = test_dir / "gt"
    _reset_dir(img_dir)
    _reset_dir(gt_dir)
    metadata: List[dict] = []
    for sample in test_samples:
        suffix = f"{sample['dataset']}_{sample['id']}"
        img_array = _tensor_to_numpy(sample["image"])
        gt_array = _tensor_to_numpy(sample["gt"])
        with (img_dir / f"img_{suffix}.pkl").open("wb") as fp:
            pickle.dump(img_array, fp)
        with (gt_dir / f"gt_{suffix}.pkl").open("wb") as fp:
            pickle.dump(gt_array, fp)
        metadata.append({
            "id": sample["id"],
            "dataset": sample["dataset"],
            "image_file": f"img_{suffix}.pkl",
            "gt_file": f"gt_{suffix}.pkl",
            "shape": list(img_array.shape[-2:]),
        })
    with (test_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def process_final_combined(final_path: Path, patch_size: int, test_per_dataset: int) -> None:
    samples = _load_final_combined_samples(final_path)
    train_samples, test_samples = _split_combined_train_test(samples, test_per_dataset)
    train_imgs = [s["image"] for s in train_samples]
    normalized_train, stats = _normalize_with_stats(train_imgs)
    for sample, tensor in zip(train_samples, normalized_train):
        sample["image"] = tensor
    test_imgs = [s["image"] for s in test_samples]
    normalized_test, _ = _normalize_with_stats(test_imgs, stats)
    for sample, tensor in zip(test_samples, normalized_test):
        sample["image"] = tensor
    _apply_padding_to_test(test_samples)
    _save_combined_training(train_samples, final_path, patch_size)
    _save_combined_test(test_samples, final_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy FR-UNet preprocessing pipeline")
    parser.add_argument('-dp', '--dataset_path', type=Path, required=True,
                        help='Path to the dataset root (matches original script expectation).')
    parser.add_argument('-dn', '--dataset_name', type=str, required=True,
                        choices=['DRIVE', 'CHASEDB1', 'STARE', 'CHUAC', 'DCA1', 'FINAL'],
                        help='Dataset identifier (same as original paper).')
    parser.add_argument('-ps', '--patch_size', type=int, default=48,
                        help='Patch size for training patch extraction (default: 48).')
    parser.add_argument('-s', '--stride', type=int, default=6,
                        help='Stride for training patch extraction (default: 6).')
    parser.add_argument('-m', '--modes', nargs='+', default=['training', 'test'],
                        choices=['training', 'test'],
                        help='Which splits to generate (default: both training and test).')
    parser.add_argument('--test-per-dataset', type=int, default=4,
                        help='(FINAL mode) number of samples per dataset reserved for test set (default: 4).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep parity with the legacy script that eagerly loaded config.yaml even if unused.
    if Path('config.yaml').exists():
        yaml = YAML(typ='safe', pure=True)
        with open('config.yaml', encoding='utf-8') as file:
            _ = yaml.load(file)
    if args.dataset_name == 'FINAL':
        process_final_combined(args.dataset_path, args.patch_size, args.test_per_dataset)
    else:
        for mode in args.modes:
            data_process(args.dataset_path, args.dataset_name, args.patch_size, args.stride, mode)


if __name__ == '__main__':
    main()
