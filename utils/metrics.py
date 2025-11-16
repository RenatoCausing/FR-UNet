import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target, threshold=None, predict_b=None):
    probs, predict_b, target = _prepare_predictions(
        predict, target, threshold, predict_b)
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    try:
        auc = roc_auc_score(target, probs)
    except ValueError:
        auc = 0.5
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def confusion_counts(predict, target, threshold=None, predict_b=None):
    _, predict_b, target = _prepare_predictions(
        predict, target, threshold, predict_b)
    tp = int((predict_b * target).sum())
    tn = int(((1 - predict_b) * (1 - target)).sum())
    fp = int(((1 - target) * predict_b).sum())
    fn = int(((1 - predict_b) * target).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def save_confusion_matrix(confusion, output_path, normalize=False):
    matrix = np.array([
        [confusion.get("TP", 0), confusion.get("FN", 0)],
        [confusion.get("FP", 0), confusion.get("TN", 0)]
    ], dtype=np.float64)
    labels = ["Positive", "Negative"]
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(2), labels)
    ax.set_yticks(range(2), labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            display = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, display, ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n


def _prepare_predictions(predict, target, threshold=None, predict_b=None):
    probs = torch.sigmoid(predict).detach().cpu().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()
    elif threshold is not None:
        predict_b = np.where(probs >= threshold, 1, 0)
    else:
        raise ValueError("Either threshold or predict_b must be provided")
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy().flatten()
    else:
        target = target.flatten()
    return probs, predict_b, target
