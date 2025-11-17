import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize, ToTensor

from utils.helpers import dir_exists, remove_files

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FINAL_DATASET_DEFAULT = Path(r"D:/FR-UNet/FR-UNet/Final Dataset")
PREVIEW_DIR_DEFAULT = Path("final_previews")
PREFIX_RULES: Dict[str, Dict[str, int]] = {
    "DRIVE": {"stride": 6},
    "STARE": {"stride": 8},
}
IMAGE_MODES = {
    "clahe": "clahe_only",
    "clahe_blur": "clahe_then_blur",
    "blur_clahe": "blur_then_clahe",
}
VARIANT_DISPLAY_ORDER: List[Tuple[str, str]] = [
    ("clahe_only", "CLAHE"),
    ("clahe_then_blur", "CLAHE→Blur"),
    ("blur_then_clahe", "Blur→CLAHE"),
]


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


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), 0)


def _apply_fov_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return image
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=binary_mask)


def _build_variants(gray_image: np.ndarray) -> Dict[str, np.ndarray]:
    clahe_only = _apply_clahe(gray_image)
    clahe_then_blur = _gaussian_blur(clahe_only)
    blur_then_clahe = _apply_clahe(_gaussian_blur(gray_image))
    return {
        "clahe_only": clahe_only,
        "clahe_then_blur": clahe_then_blur,
        "blur_then_clahe": blur_then_clahe,
    }


def _stride_for_sample(stem: str) -> int:
    prefix = stem.split("_", 1)[0].upper()
    return PREFIX_RULES.get(prefix, {"stride": 6})["stride"]


def _save_variant_preview(sample_id: str, variants: Dict[str, np.ndarray], preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    columns = []
    spacer = np.full((next(iter(variants.values())).shape[0], 4), 0, dtype=np.uint8)
    for key, _label in VARIANT_DISPLAY_ORDER:
        if key not in variants:
            continue
        if columns:
            columns.append(spacer)
        columns.append(variants[key])
    if not columns:
        return
    stacked = np.concatenate(columns, axis=1)
    cv2.imwrite(str(preview_dir / f"{sample_id}_preview.png"), stacked)


def _sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    prefix, _, remainder = stem.partition("_")
    digits = "".join(ch for ch in remainder if ch.isdigit())
    numeric = int(digits) if digits else 0
    return (prefix, numeric, stem)


def _load_final_samples(dataset_path: Path, preview_dir: Path | None, image_mode: str) -> List[dict]:
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
        gray = _to_grayscale(image)
        if gray is None:
            continue
        variants = _build_variants(gray)
        mask_img = None
        mask_path = mask_lookup.get(stem)
        if mask_path is not None:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masked_variants = {name: _apply_fov_mask(img, mask_img) for name, img in variants.items()}
        if preview_dir is not None:
            _save_variant_preview(stem, masked_variants, preview_dir)
        gt = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = np.where(gt > 0, 255, 0).astype(np.uint8)
        variant_key = IMAGE_MODES[image_mode]
        if variant_key not in masked_variants:
            continue
        samples.append({
            "id": stem,
            "image": to_tensor(masked_variants[variant_key]),
            "gt": to_tensor(gt),
            "stride": _stride_for_sample(stem),
        })
    if not samples:
        raise RuntimeError(f"No usable samples found under {dataset_path}")
    return samples


def normalization(imgs_list: List[torch.Tensor]) -> List[torch.Tensor]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_pixels = 0
    float_tensors: List[torch.Tensor] = []
    for tensor in imgs_list:
        tensor_float = tensor.float()
        float_tensors.append(tensor_float)
        total_sum += tensor_float.sum().item()
        total_sumsq += torch.square(tensor_float).sum().item()
        total_pixels += tensor_float.numel()
    if total_pixels == 0:
        raise RuntimeError("No pixels available for normalization")
    mean = total_sum / total_pixels
    variance = max((total_sumsq / total_pixels) - mean ** 2, 1e-8)
    std = variance ** 0.5
    normal_list: List[torch.Tensor] = []
    for tensor_float in float_tensors:
        normalized = (tensor_float - mean) / std
        min_val = torch.min(normalized)
        max_val = torch.max(normalized)
        denom = (max_val - min_val).clamp_min(1e-8)
        normalized = (normalized - min_val) / denom
        normal_list.append(normalized)
    return normal_list


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


def save_each_image(samples: List[dict], path: Path, key: str) -> None:
    dir_exists(str(path))
    remove_files(str(path))
    for i, sample in enumerate(samples):
        file_path = path / f"{key}_{i}.pkl"
        with file_path.open("wb") as fp:
            pickle.dump(sample[key].cpu().numpy(), fp)


def _split_samples(samples: List[dict], train_ratio: float) -> tuple[List[dict], List[dict]]:
    ratio = min(max(train_ratio, 0.0), 1.0)
    if len(samples) == 1:
        return samples, samples
    split_index = int(len(samples) * ratio)
    split_index = min(max(split_index, 1), len(samples) - 1)
    return samples[:split_index], samples[split_index:]


def data_process_final(dataset_path: Path, patch_size: int, train_ratio: float,
                       modes: List[str], preview_dir: Path | None, image_mode: str) -> None:
    samples = _load_final_samples(dataset_path, preview_dir, image_mode)
    normalized_images = normalization([sample["image"] for sample in samples])
    for norm, sample in zip(normalized_images, samples):
        sample["image"] = norm

    train_samples, holdout_samples = _split_samples(samples, train_ratio)

    for mode in modes:
        output_dir = dataset_path / f"{mode}_pro"
        if mode == "training":
            save_patch_pairs_variable_stride(train_samples, output_dir, patch_size)
        elif mode == "holdout":
            save_patch_pairs_variable_stride(holdout_samples, output_dir, patch_size)
        elif mode == "test":
            save_each_image(samples, output_dir / "img", "image")
            save_each_image(samples, output_dir / "gt", "gt")
        else:
            raise ValueError(f"Unsupported mode '{mode}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Final Dataset with prefix-specific strides and previews.")
    parser.add_argument("-dp", "--dataset-path", type=Path, default=FINAL_DATASET_DEFAULT,
                        help="Path to the Final Dataset root (default: D:/FR-UNet/FR-UNet/Final Dataset).")
    parser.add_argument("-ps", "--patch-size", type=int, default=48,
                        help="Patch size for extraction (default: 48).")
    parser.add_argument("-tr", "--train-ratio", type=float, default=0.85,
                        help="Fraction of samples reserved for training (default: 0.85).")
    parser.add_argument("-m", "--modes", nargs="+", default=["training", "test"],
                        choices=["training", "test", "holdout"],
                        help="Outputs to generate (default: training + test).")
    parser.add_argument("-pv", "--preview-dir", type=str, default=str(PREVIEW_DIR_DEFAULT),
                        help="Directory for blur/no-blur previews. Use empty string to skip.")
    parser.add_argument("-im", "--image-mode", choices=list(IMAGE_MODES.keys()), default="clahe",
                        help=(
                            "Processing style for patch extraction: "
                            "'clahe' (CLAHE only), 'clahe_blur' (CLAHE then blur), "
                            "'blur_clahe' (blur then CLAHE)."
                        ))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    data_process_final(args.dataset_path, args.patch_size, args.train_ratio, args.modes, preview_dir, args.image_mode)


if __name__ == "__main__":
    main()
