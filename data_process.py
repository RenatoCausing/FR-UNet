import os
import argparse
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import YAML
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _is_image_file(file_name):
    return Path(file_name).suffix.lower() in IMAGE_EXTENSIONS


def _extract_sample_id(file_name):
    stem = Path(file_name).stem
    return stem.split('_')[0]


def _build_lookup(folder, suffix_keyword):
    lookup = {}
    if not os.path.isdir(folder):
        return lookup
    for file_name in os.listdir(folder):
        if not _is_image_file(file_name):
            continue
        sample_id = _extract_sample_id(file_name.replace(suffix_keyword, ''))
        lookup[sample_id] = os.path.join(folder, file_name)
    return lookup


def _preprocess_retinal_image(image):
    if image is None:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def _apply_fov_mask(image, mask):
    if mask is None:
        return image
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=binary_mask)


def _load_all_samples(data_path):
    original_dir = os.path.join(data_path, "Original")
    segmented_dir = os.path.join(data_path, "Segmented")
    mask_dir = os.path.join(data_path, "Mask")
    if not os.path.isdir(original_dir):
        raise FileNotFoundError(f"Original directory not found: {original_dir}")
    if not os.path.isdir(segmented_dir):
        raise FileNotFoundError(f"Segmented directory not found: {segmented_dir}")
    seg_lookup = _build_lookup(segmented_dir, '_segmented')
    mask_lookup = _build_lookup(mask_dir, '_mask')
    to_tensor = ToTensor()
    samples = []
    original_files = sorted([
        file_name for file_name in os.listdir(original_dir)
        if _is_image_file(file_name) and _extract_sample_id(file_name).isdigit()
    ], key=lambda f: int(_extract_sample_id(f)))
    for file_name in original_files:
        sample_id = _extract_sample_id(file_name)
        img_path = os.path.join(original_dir, file_name)
        seg_path = seg_lookup.get(sample_id)
        if seg_path is None:
            continue
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = _preprocess_retinal_image(image)
        if image is None:
            continue
        mask_path = mask_lookup.get(sample_id)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        image = _apply_fov_mask(image, mask)
        gt = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = np.where(gt > 0, 255, 0).astype(np.uint8)
        samples.append((to_tensor(image), to_tensor(gt)))
    if not samples:
        raise RuntimeError(f"No samples found in {data_path}")
    return samples


def data_process(data_path, name, patch_size, stride, mode, train_ratio=0.85):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "ALL":
        samples = _load_all_samples(data_path)
        if len(samples) == 1:
            subset = samples
        else:
            ratio = min(max(train_ratio, 0.0), 1.0)
            split_index = int(len(samples) * ratio)
            split_index = min(max(split_index, 1), len(samples) - 1)
            if mode == "training":
                subset = samples[:split_index]
            elif mode in ("test", "holdout"):
                subset = samples[split_index:]
                if not subset:
                    subset = samples[-1:]
            else:
                raise ValueError(f"Unsupported mode '{mode}' for ALL dataset")
        img_list = [img for img, _ in subset]
        gt_list = [gt for _, gt in subset]
    else:
        if name == "DRIVE":
            img_path = os.path.join(data_path, mode, "images")
            gt_path = os.path.join(data_path, mode, "1st_manual")
            file_list = list(sorted(os.listdir(img_path)))
        elif name == "CHASEDB1":
            file_list = list(sorted(os.listdir(data_path)))
        elif name == "STARE":
            img_path = os.path.join(data_path, "stare-images")
            gt_path = os.path.join(data_path, "labels-ah")
            file_list = list(sorted(os.listdir(img_path)))
        elif name == "DCA1":
            data_path = os.path.join(data_path, "Database_134_Angiograms")
            file_list = list(sorted(os.listdir(data_path)))
        elif name == "CHUAC":
            img_path = os.path.join(data_path, "Original")
            gt_path = os.path.join(data_path, "Photoshop")
            file_list = list(sorted(os.listdir(img_path)))
        else:
            raise ValueError(f"Unsupported dataset name: {name}")
        img_list = []
        gt_list = []
        for i, file in enumerate(file_list):
            if name == "DRIVE":
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))

            elif name == "CHASEDB1":
                if len(file) == 13:
                    if mode == "training" and int(file[6:8]) <= 10:
                        img = Image.open(os.path.join(data_path, file))
                        gt = Image.open(os.path.join(
                            data_path, file[0:9] + '_1stHO.png'))
                        img = Grayscale(1)(img)
                        img_list.append(ToTensor()(img))
                        gt_list.append(ToTensor()(gt))
                    elif mode == "test" and int(file[6:8]) > 10:
                        img = Image.open(os.path.join(data_path, file))
                        gt = Image.open(os.path.join(
                            data_path, file[0:9] + '_1stHO.png'))
                        img = Grayscale(1)(img)
                        img_list.append(ToTensor()(img))
                        gt_list.append(ToTensor()(gt))
            elif name == "DCA1":
                if len(file) <= 7:
                    if mode == "training" and int(file[:-4]) <= 100:
                        img = cv2.imread(os.path.join(data_path, file), 0)
                        gt = cv2.imread(os.path.join(
                            data_path, file[:-4] + '_gt.pgm'), 0)
                        gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                        img_list.append(ToTensor()(img))
                        gt_list.append(ToTensor()(gt))
                    elif mode == "test" and int(file[:-4]) > 100:
                        img = cv2.imread(os.path.join(data_path, file), 0)
                        gt = cv2.imread(os.path.join(
                            data_path, file[:-4] + '_gt.pgm'), 0)
                        gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                        img_list.append(ToTensor()(img))
                        gt_list.append(ToTensor()(gt))
            elif name == "CHUAC":
                if mode == "training" and int(file[:-4]) <= 20:
                    img = cv2.imread(os.path.join(img_path, file), 0)
                    if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                        tail = "PNG"
                    else:
                        tail = "png"
                    gt = cv2.imread(os.path.join(
                        gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img = cv2.resize(
                        img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(f"save_picture/{i}img.png", img)
                    cv2.imwrite(f"save_picture/{i}gt.png", gt)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 20:
                    img = cv2.imread(os.path.join(img_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        gt_path, "angio"+file[:-4] + "ok.png"), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img = cv2.resize(
                        img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(f"save_picture/{i}img.png", img)
                    cv2.imwrite(f"save_picture/{i}gt.png", gt)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
            elif name == "STARE":
                if not file.endswith("gz"):
                    img = Image.open(os.path.join(img_path, file))
                    gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                    cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                    cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
    img_list = normalization(img_list)
    if mode == "training" or (name == "ALL" and mode == "holdout"):
        save_patch_pairs(img_list, gt_list, save_path, name, patch_size, stride)
    elif mode == "test":
        if name not in ("CHUAC", "ALL"):
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)


def get_square(img_list, name):
    img_s = []
    if not img_list:
        return img_s
    _, h, w = img_list[0].shape
    shape_lookup = {
        "DRIVE": 592,
        "CHASEDB1": 1008,
        "DCA1": 320
    }
    shape = shape_lookup.get(name, max(h, w))
    pad = nn.ConstantPad2d((0, max(shape - w, 0), 0, max(shape - h, 0)), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    for sub1 in imgs_list:
        for sub2 in _extract_patches(sub1, patch_size, stride):
            image_list.append(sub2)
    return image_list


def _extract_patches(tensor, patch_size, stride):
    _, h, w = tensor.shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), "constant", 0)
    patches = padded.unfold(1, patch_size, stride).unfold(
        2, patch_size, stride).permute(1, 2, 0, 3, 4)
    patches = patches.contiguous().view(
        patches.shape[0] * patches.shape[1], patches.shape[2], patch_size, patch_size)
    return patches


def save_patch_pairs(imgs_list, gts_list, path, name, patch_size, stride):
    idx = 0
    for img_tensor, gt_tensor in zip(imgs_list, gts_list):
        img_patches = _extract_patches(img_tensor, patch_size, stride)
        gt_patches = _extract_patches(gt_tensor, patch_size, stride)
        for img_patch, gt_patch in zip(img_patches, gt_patches):
            with open(file=os.path.join(path, f'img_patch_{idx}.pkl'), mode='wb') as file:
                 pickle.dump(img_patch.cpu().numpy(), file)
            with open(file=os.path.join(path, f'gt_patch_{idx}.pkl'), mode='wb') as file:
                 pickle.dump(gt_patch.cpu().numpy(), file)
            if idx % 1000 == 0:
                print(f'save {name} img_patch : img_patch_{idx}.pkl')
            idx += 1


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(sub.cpu().numpy(), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(sub.cpu().numpy(), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/ALL", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="ALL", type=str,
                        help='the name of dataset',choices=['ALL','DRIVE','CHASEDB1','STARE','CHUAC','DCA1'],required=True)
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    parser.add_argument('-tr', '--train_ratio', default=0.85, type=float,
                        help='ratio of samples reserved for training when using the consolidated ALL dataset')
    parser.add_argument(
        '-m', '--modes', nargs='+', default=['training', 'test'],
        choices=['training', 'test', 'holdout'],
        help=('subset of dataset splits to process (default: training + test). Use "holdout" to '
              'patchify the final 15% (ALL dataset only).')
    )
    args = parser.parse_args()
    yaml = YAML(typ='safe')
    yaml.pure = True
    with open('config.yaml', encoding='utf-8') as file:
        CFG = yaml.load(file)  # 为列表类型
    for mode in args.modes:
        data_process(args.dataset_path, args.dataset_name,
                     args.patch_size, args.stride, mode, args.train_ratio)
