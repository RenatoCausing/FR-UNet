import os
import pickle
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from utils.helpers import Fix_RandomRotation


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None, cache_in_memory=False):
        if mode not in {"training", "test", "holdout"}:
            raise ValueError(f"Unsupported mode '{mode}'. Expected training, test, or holdout.")
        self.mode = mode
        self.is_val = is_val
        self.cache_in_memory = cache_in_memory
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
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
        self._cached_samples = None
        self._share_memory_failed = False
        if self.cache_in_memory:
            try:
                self._cached_samples = self._preload_samples()
            except MemoryError as exc:
                logger.error("RAM cache: ran out of host memory while preloading patches ({}). Disabling cache and streaming from disk.", exc)
                self._cached_samples = None
                self.cache_in_memory = False

    def __getitem__(self, idx):
        if self.cache_in_memory and self._cached_samples is not None:
            img, gt = self._cached_samples[idx]
            # clone to keep cached tensors pristine when augmentations run
            img = img.clone()
            gt = gt.clone()
        else:
            img_file = self.img_file[idx]
            img = self._load_tensor(img_file)
            gt_file = "gt" + img_file[3:]
            gt = self._load_tensor(gt_file)

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

    def _preload_samples(self):
        samples = []
        try:
            for img_file in self.img_file:
                img = self._load_tensor(img_file)
                gt = self._load_tensor("gt" + img_file[3:])
                if not self._share_memory_failed:
                    try:
                        img.share_memory_()
                        gt.share_memory_()
                    except RuntimeError as exc:
                        self._share_memory_failed = True
                        logger.warning("RAM cache: unable to share tensors via shared memory ({}). Continuing without shared pages.", exc)
                samples.append((img, gt))
        except MemoryError:
            samples.clear()
            raise
        return samples

    def _load_tensor(self, file_name):
        with open(file=os.path.join(self.data_path, file_name), mode='rb') as file:
            tensor = torch.from_numpy(pickle.load(file)).float()
        return tensor
