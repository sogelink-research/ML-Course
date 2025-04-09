from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


def accuracy(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the accuracy metric (TP + TN) / (TP + FP + TN + FN).
    """
    return (tp + tn) / (tp + fp + tn + fn)


def precision(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the precision metric TP / (TP + FP).
    If the denominator is 0, return -1.0.
    """
    return np.where(tp + fp == 0, 1.0, tp / (tp + fp))


def recall(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the recall metric TP / (TP + FN).
    If the denominator is 0, return -1.0.
    """
    return np.where(tp + fn == 0, 1.0, tp / (tp + fn))


def f1(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """
    Calculate the F1 score metric 2 * (precision * recall) / (precision + recall).
    If the denominator is 0, return -1.0.
    """
    precision_value = precision(tp, fp, tn, fn)
    recall_value = recall(tp, fp, tn, fn)
    return np.where(
        precision_value + recall_value == 0,
        0.0,
        2 * (precision_value * recall_value) / (precision_value + recall_value),
    )


def dice_loss(logits, targets, reduction: str, smooth: float = 1.0):
    """
    Compute the Dice loss.
    Args:
        logits: Predicted logits.
        targets: Target labels.
        reduction: Reduction method ('none', 'mean', 'sum').
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice loss value.
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss_value = 1 - dice_score

    if reduction == "mean":
        return dice_loss_value.mean()
    elif reduction == "sum":
        return dice_loss_value.sum()
    elif reduction == "none":
        return dice_loss_value
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def full_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Compute the full loss function.
    This function combines Binary Cross-Entropy (BCE), Dice loss, and Focal loss.
    The loss values are averaged over the batch.
    Args:
        logits: Predicted logits.
        targets: Target labels.
    Returns:
        total: Total loss value.
        (bce, dice, focal): Individual loss components.
    """
    alpha = 1
    beta = 1
    gamma = 1
    sum_weights = alpha + beta + gamma
    alpha /= sum_weights
    beta /= sum_weights
    gamma /= sum_weights

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    dice = dice_loss(logits, targets, reduction="mean")
    focal = sigmoid_focal_loss(logits, targets, reduction="mean")

    # Combine the losses
    total = alpha * bce + beta * dice + gamma * focal
    return total, (bce, dice, focal)


class EvaluationMetrics:

    def __init__(
        self,
    ):
        self.tps = []
        self.fps = []
        self.tns = []
        self.fns = []
        self.updated_metrics = False
        self.all_metrics_functions = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }
        self.individual_values = {
            name: None for name in self.all_metrics_functions.keys()
        }
        self.global_values = {name: None for name in self.all_metrics_functions.keys()}

    def add_tp_fp_tn_fn(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ):
        """
        Add metrics to the class.
        """
        self.tps.append(tp)
        self.fps.append(fp)
        self.tns.append(tn)
        self.fns.append(fn)
        self.updated_metrics = False

    def add_logits_and_targets(
        self,
        logits: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        threshold: float = 0.5,
    ):
        """
        Add logits and targets to the class.
        """
        if isinstance(logits, torch.Tensor):
            preds = (logits > threshold).to(torch.int)
        elif isinstance(logits, np.ndarray):
            preds = (logits > threshold).astype(int)
        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        tn = ((1 - preds) * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()

        self.add_tp_fp_tn_fn(tp, fp, tn, fn)

    def _compute_metrics(self):
        """
        Compute all metrics.
        """
        if self.updated_metrics:
            return
        tps = np.array(self.tps)
        fps = np.array(self.fps)
        tns = np.array(self.tns)
        fns = np.array(self.fns)
        global_tps = np.sum(tps)
        global_fps = np.sum(fps)
        global_tns = np.sum(tns)
        global_fns = np.sum(fns)
        for metric_name, metric_func in self.all_metrics_functions.items():
            self.individual_values[metric_name] = metric_func(tps, fps, tns, fns)
            self.global_values[metric_name] = metric_func(
                global_tps, global_fps, global_tns, global_fns
            ).item()
        self.updated_metrics = True

    def get_individual_metrics(self):
        self._compute_metrics()
        return self.individual_values

    def get_global_metrics(self):
        self._compute_metrics()
        return self.global_values
