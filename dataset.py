import math
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):
        if mode not in {"training", "test", "holdout"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected training, test, or holdout.")
        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self._pad_factor = 32  # guarantees compatibility with the encoder/decoder strides
        # test/holdout exports keep tensors under img/gt folders while training keeps them flat
        self._uses_subdirs = all(
            os.path.isdir(os.path.join(self.data_path, folder)) for folder in ("img", "gt")
        )
        self.img_dir = os.path.join(self.data_path, "img") if self._uses_subdirs else self.data_path
        self.gt_dir = os.path.join(self.data_path, "gt") if self._uses_subdirs else self.data_path

        self.data_file = os.listdir(self.img_dir)
        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]
        self._augment = (self.mode in {"training", "holdout"}) and not self.is_val
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        img = self._load_tensor(img_file, self.img_dir)
        gt_file = "gt" + img_file[3:]
        gt = self._load_tensor(gt_file, self.gt_dir)

        if self._augment:
            seed = torch.seed()
            # Reseed before each transform call so image/label undergo identical augmentations
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)

    def _load_tensor(self, file_name, base_path):
        with open(file=os.path.join(base_path, file_name), mode='rb') as file:
            tensor = torch.from_numpy(pickle.load(file)).float()
        return self._pad_tensor(tensor)

    def _pad_tensor(self, tensor):
        if tensor.ndim < 3:
            return tensor
        h, w = tensor.shape[-2:]
        target_h = math.ceil(h / self._pad_factor) * self._pad_factor
        target_w = math.ceil(w / self._pad_factor) * self._pad_factor
        if target_h == h and target_w == w:
            return tensor
        padded = torch.zeros((tensor.shape[0], target_h, target_w), dtype=tensor.dtype)
        padded[:, :h, :w] = tensor
        return padded
