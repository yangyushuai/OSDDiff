import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_dice_score(pred, targs):
    pred = (pred > 0).float()
    dice_socre = 2. * (pred * targs).sum() / (pred + targs).sum()
    return dice_socre


def compute_precision(pred, label):
    TP = (pred * label).sum(dim=[2, 3])
    FN = ((1 - pred) * label).sum(dim=[2, 3])
    smooth = 1e-5
    com = TP / (TP + FN + smooth)
    com = torch.mean(com, dim=0).item()
    return com


def compute_hd95(pred, label):
    hd95_values = []
    for i in range(pred.shape[0]):
        pred_mask = pred[i, 0].detach().cpu().numpy()
        label_mask = label[i, 0].detach().cpu().numpy()
        hd95_pred_to_label = directed_hausdorff(pred_mask, label_mask)[0]
        hd95_label_to_pred = directed_hausdorff(label_mask, pred_mask)[0]
        hd95_values.append(hd95_pred_to_label)
        hd95_values.append(hd95_label_to_pred)
    hd95 = np.percentile(hd95_values, 95)
    return hd95


def compute_acc(pred, label):
    label = label.squeeze(1)
    pred = pred.squeeze(1)
    correct_pixels = torch.sum(label == pred).item()
    total_pixels = label.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy


def calculate_metrics(img, label):
    diceScore = compute_dice_score(img, label)
    hd95 = compute_hd95(img, label)
    precision = compute_precision(img, label)
    acc = compute_acc(img, label)
    return diceScore, hd95, precision, acc
