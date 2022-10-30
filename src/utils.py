from sklearn import metrics
import numpy as np
import gc
import torch


def free_gpu_memory(data):
    del data
    data = None
    gc.collect()
    torch.cuda.empty_cache()


def confusion_matrix(predicted_mask, true_mask, threshold=0.5):
    # count the number of pixels that were correctly classified
    # true mask is binary

    # threshold the predicted mask
    predicted_mask = predicted_mask > threshold

    # construct a confusion matrix
    # TN top left
    return metrics.confusion_matrix(true_mask, predicted_mask)


def accuracy(confusion_matrix):
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def precision(confusion_matrix):
    # TP / (TP + FP)
    return confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])


def recall(confusion_matrix):
    # TP / (TP + FN)
    return confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])


def false_positive_rate(confusion_matrix):
    # FP / (FP + TN)
    return confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[0, 0])


def f1_score(confusion_matrix):
    prec = precision(confusion_matrix)
    rec = recall(confusion_matrix)

    return 2 * prec * rec / (prec + rec)
