import torch

# For eigval operations
def eigval_MAE(pred, x):
    e = (pred - x).abs().mean()
    return e


def eigval_L1(pred, x):
    return relative_L1_evaluation_error(pred, x).mean()


def relative_L1_evaluation_error(pred, x):
    return (pred - x).abs().sum(-1) / (x.abs().sum(-1))

