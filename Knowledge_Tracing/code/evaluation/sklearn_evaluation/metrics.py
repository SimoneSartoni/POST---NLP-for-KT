from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import torch


def binary_accuracy(y_true, y_pred):
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    return accuracy_score(y_true, predictions)


def cold_start_binary_accuracy(y_true, y_pred, window_size=30):
    y_true = y_true[0:window_size]
    y_pred = y_pred[0:window_size]
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    return accuracy_score(y_true, predictions)


def window_binary_accuracy(y_true, y_pred, low_index=0, high_index=30):
    y_true = y_true[low_index:high_index]
    y_pred = y_pred[low_index:high_index]
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    return accuracy_score(y_true, predictions)


def area_under_curve(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def cold_start_auc(y_true, y_pred, window_size=30):
    y_true = y_true[0:window_size]
    y_pred = y_pred[0:window_size]
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        x = torch.tensor([0.0, 1.0])
        y_true = torch.cat(y_true, x)
        y_pred = torch.cat(y_pred, x)
        print("error")
        return roc_auc_score(y_true, y_pred)


def window_auc(y_true, y_pred, low_index=0, high_index=30):
    y_true = y_true[low_index:high_index]
    y_pred = y_pred[low_index:high_index]
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        x = torch.tensor([0.0, 1.0])
        y_true = torch.cat(y_true, x)
        y_pred = torch.cat(y_pred, x)
        print("error")
        return roc_auc_score(y_true, y_pred)


def precision(y_true, y_pred):
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    return precision_score(y_true, predictions)


def recall(y_true, y_pred):
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    return recall_score(y_true, predictions)


def confusion_matrix_score(y_true, y_pred):
    predictions = [1.0 if output >= 0.5 else 0.0 for output in y_pred]
    confusion_m = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = confusion_m.ravel()
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    acc, auc,  = binary_accuracy(y_true, y_pred), area_under_curve(y_true, y_pred)
    acc_10, auc_10 = cold_start_binary_accuracy(y_true, y_pred, 10), cold_start_auc(y_true, y_pred, 10)
    acc_30, auc_30 = cold_start_binary_accuracy(y_true, y_pred, 30), cold_start_auc(y_true, y_pred, 30)
    acc_50, auc_50 = cold_start_binary_accuracy(y_true, y_pred, 50), cold_start_auc(y_true, y_pred, 50)
    prec, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    tp, tn, fp, fn = confusion_matrix_score(y_true, y_pred)
    return acc, auc, acc_10, auc_10, acc_30, auc_30, acc_50, auc_50, prec, rec, tp, tn, fp, fn


def get_metrics_names():
    return ["acc", "auc", "acc_10", "auc_10", "acc_30", "auc_30", "acc_50", "auc_50", "prec", "rec", "tp", "tn", "fp", "fn"]