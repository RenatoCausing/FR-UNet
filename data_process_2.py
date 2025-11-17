import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from utils.helpers import dir_exists, remove_files

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FINAL_DATASET_DEFAULT = Path(r"D:/FR-UNet/FR-UNet/Final Dataset")
PREVIEW_DIR_DEFAULT = Path("final_previews")
PREFIX_RULES: Dict[str, Dict[str, int]] = {
    "DRIVE": {"stride": 6},
    "STARE": {"stride": 7},
}
TEST_SET_ALLOCATION: Dict[str, int] = {
    "DRIVE": 4,
    "STARE": 4,
}
NORMALIZATION_STATS_FILE = "normalization_stats.json"
def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS

def _build_lookup_with_suffix(folder: Path, suffix: str) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if not folder.exists():
        return lookup
    for path in folder.iterdir():
        if not path.is_file() or not _is_image_file(path):
            continue
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        lookup[base] = path
    return lookup


def _extract_green_channel(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] >= 2:
        return image[:, :, 1]
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def _morphological_open(image: np.ndarray, radius: int = 2) -> np.ndarray:
    if image is None:
        return None
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def _shade_correct(enhanced_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
    if enhanced_image is None or background_image is None:
        return enhanced_image
    float_img = enhanced_image.astype(np.float32)
    background = background_image.astype(np.float32)
    corrected = float_img - background
    corrected[corrected > 0] = 0  # keep only pixels darker than estimated background
    corrected = -corrected  # flip sign so brighter means stronger vessels
    max_val = float(np.max(corrected))
    if max_val <= 0:
        return enhanced_image
    corrected = (corrected / max_val) * 255.0
    return corrected.astype(np.uint8)


def _apply_fov_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return image
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if image.dtype == np.uint8:
        return cv2.bitwise_and(image, image, mask=(mask > 0).astype(np.uint8) * 255)
    return image * ((mask > 0).astype(image.dtype))


def _stack_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    width_left = left.shape[1]
    width_right = right.shape[1]
    spacer = np.full((height, 4), 0, dtype=np.uint8)
    left_resized = cv2.resize(left, (width_left, height)) if left.shape[0] != height else left
    right_resized = cv2.resize(right, (width_right, height)) if right.shape[0] != height else right
    return np.concatenate([left_resized, spacer, right_resized], axis=1)


def _save_preview(sample_id: str, clahe_img: np.ndarray, shade_img: np.ndarray, preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    stacked = _stack_preview(clahe_img, shade_img)
    cv2.imwrite(str(preview_dir / f"{sample_id}_preview.png"), stacked)


def _dataset_prefix(stem: str) -> str:
    return stem.split("_", 1)[0].upper()


def _stride_for_sample(stem: str) -> int:
    prefix = _dataset_prefix(stem)
    return PREFIX_RULES.get(prefix, {"stride": 6})["stride"]
def _median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def _sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    prefix, _, remainder = stem.partition("_")
    digits = "".join(ch for ch in remainder if ch.isdigit())
    numeric = int(digits) if digits else 0
    return (prefix, numeric, stem)


def _preprocess_image(image: np.ndarray, mask_image: np.ndarray | None,
                      clip_limit: float, tile_grid_size: tuple,
                      shade_kernel: int, shade_correction: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    green = _extract_green_channel(image)
    if green is None:
        return None, None
    opened = _morphological_open(green, radius=2)
    enhanced = _apply_clahe(opened, clip_limit, tile_grid_size)
    processed = enhanced
    if shade_correction:
        shade_kernel = max(3, abs(int(shade_kernel)))
        if shade_kernel % 2 == 0:
            shade_kernel += 1
        background = _median_blur(enhanced, shade_kernel)
        processed = _shade_correct(enhanced, background)
    return _apply_fov_mask(processed, mask_image), _apply_fov_mask(enhanced, mask_image)


def _load_final_samples(dataset_path: Path, clip_limit: float = 2.0,
                        tile_grid_size: tuple = (8, 8),
                        shade_correction: bool = False, shade_kernel: int = 25,
                        preview_dir: Path | None = None) -> List[dict]:
    original_dir = dataset_path / "Original"
    mask_dir = dataset_path / "Mask"
    segmented_dir = dataset_path / "Segmented"
    if not original_dir.exists() or not segmented_dir.exists():
        raise FileNotFoundError("Final Dataset must contain Original and Segmented directories.")

    mask_lookup = _build_lookup_with_suffix(mask_dir, "_mask")
    seg_lookup = _build_lookup_with_suffix(segmented_dir, "_segment")
    to_tensor = ToTensor()
    samples: List[dict] = []

    for img_path in sorted((p for p in original_dir.iterdir() if p.is_file() and _is_image_file(p)), key=_sort_key):
        stem = img_path.stem
        seg_path = seg_lookup.get(stem)
        if seg_path is None:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask_img = None
        mask_path = mask_lookup.get(stem)
        if mask_path is not None:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        processed, clahe_only = _preprocess_image(
            image, mask_img, clip_limit, tile_grid_size, shade_kernel, shade_correction
        )
        if processed is None:
            continue
        if preview_dir is not None and clahe_only is not None:
            _save_preview(stem, clahe_only, processed, preview_dir)
        gt = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = np.where(gt > 0, 255, 0).astype(np.uint8)
        dataset_name = _dataset_prefix(stem)
        samples.append({
            "id": stem,
            "dataset": dataset_name,
            "image": to_tensor(processed),
            "gt": to_tensor(gt),
            "stride": _stride_for_sample(stem),
        })
    if not samples:
        raise RuntimeError(f"No usable samples found under {dataset_path}")
    return samples


def _compute_normalization_stats(tensors: List[torch.Tensor]) -> Dict[str, float]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_pixels = 0
    for tensor in tensors:
        tensor_float = tensor.float()
        total_sum += tensor_float.sum().item()
        total_sumsq += torch.square(tensor_float).sum().item()
        total_pixels += tensor_float.numel()
    if total_pixels == 0:
        raise RuntimeError("No pixels available for normalization")
    mean = total_sum / total_pixels
    variance = max((total_sumsq / total_pixels) - mean ** 2, 1e-8)
    std = variance ** 0.5
    z_min = float("inf")
    z_max = float("-inf")
    for tensor in tensors:
        tensor_float = tensor.float()
        z_tensor = (tensor_float - mean) / std
        current_min = z_tensor.min().item()
        current_max = z_tensor.max().item()
        z_min = min(z_min, current_min)
        z_max = max(z_max, current_max)
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        raise RuntimeError("Failed to compute finite normalization range")
    return {
        "mean": float(mean),
        "std": float(std),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "total_pixels": int(total_pixels),
        "num_samples": len(tensors),
    }


def _apply_normalization(tensors: List[torch.Tensor], stats: Dict[str, float]) -> List[torch.Tensor]:
    mean = stats["mean"]
    std = max(stats["std"], 1e-8)
    z_min = stats["z_min"]
    z_max = stats["z_max"]
    denom = max(z_max - z_min, 1e-8)
    normalized_list: List[torch.Tensor] = []
    for tensor in tensors:
        tensor_float = tensor.float()
        z_tensor = (tensor_float - mean) / std
        scaled = (z_tensor - z_min) / denom
        normalized_list.append(torch.clamp(scaled, 0.0, 1.0))
    return normalized_list


def _save_normalization_stats(dataset_path: Path, stats: Dict[str, float], metadata: Dict[str, object]) -> Path:
    payload = {
        "stats": stats,
        "metadata": metadata,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    target = dataset_path / NORMALIZATION_STATS_FILE
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return target


def _extract_patches(tensor: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    _, h, w = tensor.shape
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), "constant", 0)
    patches = padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride).permute(1, 2, 0, 3, 4)
    return patches.contiguous().view(-1, patches.shape[2], patch_size, patch_size)


def save_patch_pairs_variable_stride(samples: List[dict], path: Path, patch_size: int) -> None:
    dir_exists(str(path))
    remove_files(str(path))
    idx = 0
    for sample in samples:
        img_patches = _extract_patches(sample["image"], patch_size, sample["stride"])
        gt_patches = _extract_patches(sample["gt"], patch_size, sample["stride"])
        for img_patch, gt_patch in zip(img_patches, gt_patches):
            img_file = path / f"img_patch_{idx}.pkl"
            gt_file = path / f"gt_patch_{idx}.pkl"
            with img_file.open("wb") as fp:
                pickle.dump(img_patch.cpu().numpy(), fp)
            with gt_file.open("wb") as fp:
                pickle.dump(gt_patch.cpu().numpy(), fp)
            if idx % 1000 == 0:
                print(f"Saved patch pair {idx}")
            idx += 1


def _select_test_samples(samples: List[dict], allocation: Dict[str, int]) -> tuple[List[dict], List[dict]]:
    if not allocation:
        return samples, []
    dataset_buckets: Dict[str, List[dict]] = {}
    for sample in samples:
        dataset_buckets.setdefault(sample["dataset"], []).append(sample)
    selected_ids: set[str] = set()
    for dataset, quota in allocation.items():
        if quota <= 0:
            continue
        bucket = dataset_buckets.get(dataset.upper(), [])
        if len(bucket) < quota:
            raise RuntimeError(
                f"Requested {quota} samples for {dataset} but only {len(bucket)} present."
            )
        for sample in bucket[:quota]:
            selected_ids.add(sample["id"])
    train_samples = [s for s in samples if s["id"] not in selected_ids]
    test_samples = [s for s in samples if s["id"] in selected_ids]
    if not test_samples:
        raise RuntimeError("Test allocation produced zero samples; check configuration.")
    return train_samples, test_samples


def _serialize_tensor(sample: dict, key: str) -> bytes:
    return pickle.dumps(sample[key].cpu().numpy())


def _save_test_dataset(samples: List[dict], path: Path) -> None:
    dir_exists(str(path))
    img_dir = path / "img"
    gt_dir = path / "gt"
    dir_exists(str(img_dir))
    dir_exists(str(gt_dir))
    remove_files(str(img_dir))
    remove_files(str(gt_dir))
    metadata: List[dict] = []
    for sample in samples:
        suffix = f"{sample['dataset'].upper()}_{sample['id']}"
        img_name = f"img_{suffix}.pkl"
        gt_name = f"gt_{suffix}.pkl"
        with (img_dir / img_name).open("wb") as fp:
            fp.write(_serialize_tensor(sample, "image"))
        with (gt_dir / gt_name).open("wb") as fp:
            fp.write(_serialize_tensor(sample, "gt"))
        metadata.append({
            "id": sample["id"],
            "dataset": sample["dataset"],
            "image_file": img_name,
            "gt_file": gt_name,
            "shape": list(sample["image"].shape[-2:]),
        })
    with (path / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def data_process_final(dataset_path: Path, patch_size: int, train_ratio: float,
                       modes: List[str], clip_limit: float = 2.0,
                       tile_grid_size: tuple = (8, 8), shade_correction: bool = False,
                       shade_kernel: int = 25, dry_run: bool = False,
                       preview_dir: Path | None = None) -> None:
    samples = _load_final_samples(
        dataset_path, clip_limit=clip_limit, tile_grid_size=tile_grid_size,
        shade_correction=shade_correction, shade_kernel=shade_kernel,
        preview_dir=preview_dir if not dry_run else None)
    train_pool, test_samples = _select_test_samples(samples, TEST_SET_ALLOCATION)
    if not train_pool:
        raise RuntimeError("No training samples left after reserving the test set.")

    ratio = min(max(train_ratio, 0.0), 1.0)
    if ratio == 0.0:
        raise RuntimeError("train_ratio of 0 would drop every training sample.")
    if ratio < 1.0:
        keep = max(1, int(len(train_pool) * ratio))
        train_pool = train_pool[:keep]

    train_images = [s["image"] for s in train_pool]
    norm_stats = _compute_normalization_stats(train_images)
    normalized_train = _apply_normalization(train_images, norm_stats)
    for norm, sample in zip(normalized_train, train_pool):
        sample["image"] = norm

    if test_samples:
        test_images = [s["image"] for s in test_samples]
        normalized_test = _apply_normalization(test_images, norm_stats)
        for norm, sample in zip(normalized_test, test_samples):
            sample["image"] = norm

    metadata = {
        "clip_limit": clip_limit,
        "tile_grid_size": list(tile_grid_size),
        "patch_size": patch_size,
        "train_ratio_kept": ratio,
        "train_sample_count": len(train_pool),
        "test_sample_count": len(test_samples),
        "test_allocation": TEST_SET_ALLOCATION,
        "shade_correction": shade_correction,
        "shade_kernel": shade_kernel,
        "preview_dir": str(preview_dir) if preview_dir else None,
    }
    stats_path = dataset_path / NORMALIZATION_STATS_FILE

    if dry_run:
        print(f"[DRY RUN] Would process {len(train_pool)} training samples + {len(test_samples)} test samples")
        print(f"[DRY RUN] Test allocation: {TEST_SET_ALLOCATION}")
        print(f"[DRY RUN] Training keep ratio: {ratio}")
        print(f"[DRY RUN] CLAHE: clipLimit={clip_limit}, tileGridSize={tile_grid_size}")
        print(f"[DRY RUN] Shade correction: {shade_correction} (kernel={shade_kernel})")
        print(f"[DRY RUN] Normalization stats preview: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}, z-range=({norm_stats['z_min']:.4f}, {norm_stats['z_max']:.4f})")
        print(f"[DRY RUN] Stats file would be saved to: {stats_path}")
        print(f"[DRY RUN] Modes: {modes}")
        if preview_dir:
            print(f"[DRY RUN] Previews saved to: {preview_dir}")
        else:
            print("[DRY RUN] Previews disabled")
        return

    _save_normalization_stats(dataset_path, norm_stats, metadata)

    for mode in modes:
        output_dir = dataset_path / f"{mode}_pro"
        if mode == "training":
            save_patch_pairs_variable_stride(train_pool, output_dir, patch_size)
        elif mode == "test":
            _save_test_dataset(test_samples, output_dir)
        else:
            raise ValueError(f"Unsupported mode '{mode}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Final Dataset with prefix-specific strides and MO→CLAHE→shade pipeline."
    )
    parser.add_argument("-dp", "--dataset-path", type=Path, default=FINAL_DATASET_DEFAULT,
                        help="Path to the Final Dataset root (default: D:/FR-UNet/FR-UNet/Final Dataset).")
    parser.add_argument("-ps", "--patch-size", type=int, default=48,
                        help="Patch size for extraction (default: 48).")
    parser.add_argument("-tr", "--train-ratio", type=float, default=1.0,
                        help="Fraction of remaining (non-test) samples kept for training (default: 1.0).")
    parser.add_argument("-m", "--modes", nargs="+", default=["training", "test"],
                        choices=["training", "test"],
                        help="Outputs to generate (default: training + test).")
    parser.add_argument("-pv", "--preview-dir", type=str, default=str(PREVIEW_DIR_DEFAULT),
                        help="Directory for CLAHE vs shade previews (empty string disables previews).")
    parser.add_argument("-cl", "--clip-limit", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0).")
    parser.add_argument("-tg", "--tile-grid", type=int, nargs=2, default=[8, 8],
                        metavar=("WIDTH", "HEIGHT"),
                        help="CLAHE tile grid size (default: 8 8).")
    parser.add_argument("--shade-correction", action="store_true",
                        help="Apply shade correction (median background subtraction) after CLAHE.")
    parser.add_argument("--shade-kernel", type=int, default=25,
                        help="Median filter kernel size (odd) for shade correction background (default: 25).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview preprocessing settings without saving patches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_grid_size = tuple(args.tile_grid)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    data_process_final(
        args.dataset_path, args.patch_size, args.train_ratio, args.modes,
        args.clip_limit, tile_grid_size,
        args.shade_correction, args.shade_kernel, args.dry_run, preview_dir
    )


if __name__ == "__main__":
    main()
