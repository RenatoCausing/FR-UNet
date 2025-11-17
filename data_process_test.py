import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

from data_process import (
    _apply_fov_mask,
    _build_lookup,
    _extract_sample_id,
    _is_image_file,
    _preprocess_retinal_image,
    normalization,
)


# Edit this list to adjust the default CLAHE clip limits and kernel sizes used by the sampler.
DEFAULT_CLAHE_CONFIGS: list[tuple[float, int]] = [
    (2.0, 8),
    (4.0, 16),
]


def _collect_originals(original_dir: Path) -> dict[str, Path]:
    originals: dict[str, Path] = {}
    for file_name in sorted(original_dir.iterdir()):
        if not _is_image_file(file_name.name):
            continue
        sample_id = _extract_sample_id(file_name.name)
        if not sample_id.isdigit():
            continue
        originals[sample_id] = file_name
    return originals


def _select_samples(originals: dict[str, Path], segmented_lookup: dict[str, str], sample_size: int,
                    seed: int | None) -> list[str]:
    shared_ids = sorted(set(originals.keys()) & set(segmented_lookup.keys()))
    if not shared_ids:
        raise RuntimeError("No overlapping Original/Segmented pairs were found.")
    rng = random.Random(seed)
    if sample_size >= len(shared_ids):
        return shared_ids
    return rng.sample(shared_ids, sample_size)


def _weighted_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _apply_custom_clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image)
    return enhanced


def _prepare_sample(sample_id: str, paths: dict[str, Path], segmented_lookup: dict[str, str],
                    mask_lookup: dict[str, str], tensor_transform: ToTensor,
                    clahe_configs: list[tuple[float, int]]) -> dict:
    img_path = paths[sample_id]
    seg_path = Path(segmented_lookup[sample_id])
    mask_path = Path(mask_lookup[sample_id]) if sample_id in mask_lookup else None

    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Failed to read {img_path}")
    if raw.ndim == 3:
        grayscale = _weighted_grayscale(raw)
    else:
        grayscale = raw.copy()

    processed_variants = [_apply_custom_clahe(grayscale, clip, tile) for clip, tile in clahe_configs]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
    masked_variants = [_apply_fov_mask(proc, mask) for proc in processed_variants]

    gt = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise RuntimeError(f"Failed to read segmented map {seg_path}")

    return {
        "sample_id": sample_id,
        "original_gray": grayscale,
        "processed_variants": masked_variants,
        "tensors": [tensor_transform(masked) for masked in masked_variants],
        "segmented": gt,
        "mask": mask,
        "paths": {
            "original": str(img_path),
            "segmented": str(seg_path),
            "mask": str(mask_path) if mask_path else None,
        },
    }


def _stack_columns(columns: list[torch.Tensor | None]) -> torch.Tensor:
    tensors = [col for col in columns if col is not None]
    if not tensors:
        raise RuntimeError("No tensors provided for concatenation")
    return torch.cat(tensors, dim=-1)


def _save_outputs(samples: list[dict], normalized_groups: list[list[torch.Tensor]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] = []
    for idx, (sample, norm_group) in enumerate(zip(samples, normalized_groups), start=1):
        sample_dir = output_dir / f"{idx:02d}_{sample['sample_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        original_tensor = torch.from_numpy(sample["original_gray"]).unsqueeze(0).float() / 255.0
        segmented_tensor = torch.from_numpy(sample["segmented"]).unsqueeze(0).float() / 255.0

        concatenated = _stack_columns([
            original_tensor,
            norm_group[0],
            norm_group[1] if len(norm_group) > 1 else None,
            segmented_tensor,
        ])
        concat_img = (concatenated.squeeze().numpy() * 255.0).clip(0, 255).astype("uint8")
        cv2.imwrite(str(sample_dir / "preview.png"), concat_img)

        metadata.append({
            "sample_id": sample["sample_id"],
            "paths": sample["paths"],
            "output_dir": str(sample_dir),
            "has_mask": sample["mask"] is not None,
        })

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a handful of ALL dataset images, apply CLAHE + masking + normalization, and "
            "write intermediate products for visual inspection."
        )
    )
    default_clahe_str = " ".join(f"{clip} {tile}" for clip, tile in DEFAULT_CLAHE_CONFIGS)
    parser.add_argument(
        "-dp",
        "--dataset-path",
        default=r"D:\\DRIVE\\2nd RUN\\ALL_2",
        help="Root folder containing Original/Mask/Segmented subfolders.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="sampled_previews",
        help="Directory where preview images will be written (default: sampled_previews).",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=10,
        help="Number of random samples to export (default: 10).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--clahe",
        nargs=4,
        type=float,
        metavar=("clip1", "tile1", "clip2", "tile2"),
        default=None,
        help=(
            "Override CLAHE settings as clip/tile pairs ("
            f"default: {default_clahe_str})."
        ),
    )
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    base_path = Path(args.dataset_path)
    original_dir = base_path / "Original"
    segmented_dir = base_path / "Segmented"
    mask_dir = base_path / "Mask"

    if not original_dir.exists() or not segmented_dir.exists():
        raise FileNotFoundError("Original and Segmented folders must exist under the dataset path.")

    originals = _collect_originals(original_dir)
    segmented_lookup = _build_lookup(str(segmented_dir), "_segmented")
    mask_lookup = _build_lookup(str(mask_dir), "_mask") if mask_dir.exists() else {}

    if args.clahe is not None:
        clip1, tile1, clip2, tile2 = args.clahe
        clahe_configs = [(clip1, int(tile1)), (clip2, int(tile2))]
    else:
        clahe_configs = list(DEFAULT_CLAHE_CONFIGS)

    selected_ids = _select_samples(originals, segmented_lookup, args.sample_size, args.seed)
    to_tensor = ToTensor()
    samples: list[dict] = []
    for sample_id in selected_ids:
        samples.append(_prepare_sample(sample_id, originals, segmented_lookup, mask_lookup, to_tensor, clahe_configs))

    normalized_groups: list[list[torch.Tensor]] = []
    for sample in samples:
        normalized = normalization(sample["tensors"])
        normalized_groups.append(normalized)
    _save_outputs(samples, normalized_groups, Path(args.output_dir))

    print(f"Wrote {len(samples)} previews to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
