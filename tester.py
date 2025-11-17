import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration, double_threshold_iteration_fast, get_torch_device
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component, confusion_counts
import ttach as tta


class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False, device=None, max_save=100):
        self.device = device or get_torch_device()
        self.loss = loss.to(self.device)
        self.CFG = CFG
        self.test_loader = test_loader
        model = model.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.dataset_path = dataset_path
        self.show = show
        self.max_save = max_save
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True
        self.confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        non_blocking = self.device.type == 'cuda'
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.to(self.device, non_blocking=non_blocking)
                gt = gt.to(self.device, non_blocking=non_blocking)
                pre = self.model(img)
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)

                img = img[0, 0, ...]
                gt = gt[0, 0, ...]
                pre = pre[0, 0, ...]
                if self.show and i < self.max_save:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    cv2.imwrite(
                        f"save_picture/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"save_picture/pre{i}.png", np.uint8(predict*255))
                    cv2.imwrite(
                        f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))

                if self.CFG.DTI:
                    if self.CFG.fast_DTI:
                        pre_DTI = double_threshold_iteration_fast(
                            i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    else:
                        pre_DTI = double_threshold_iteration(
                            i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    counts = confusion_counts(pre, gt, predict_b=pre_DTI)
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    counts = confusion_counts(pre, gt, threshold=self.CFG.threshold)
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                for key in self.confusion:
                    self.confusion[key] += counts[key]
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
        results = {
            'loss': float(self.total_loss.average),
            **{k: float(v) for k, v in self._metrics_ave().items()},
            'confusion': {k: int(v) for k, v in self.confusion.items()}
        }
        if self.CFG.CCC:
            results['CCC'] = float(self.CCC.average)
        return results

    def _reset_metrics(self):
        super()._reset_metrics()
        self.confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        