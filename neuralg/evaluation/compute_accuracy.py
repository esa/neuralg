import torch


def compute_accuracy(tol, results):
    """ Compute the accuracy given some tolerance and prediction errors on test set 

    Args:
        tol (float): Tolerance level, should be between zero and one
        results (tensor): Array of prediction errors

    Returns:
        tensor: Resulting prediction accuracy on the test set
    """

    x = torch.zeros(results.shape[0])
    y = (results - tol).squeeze()
    x[y > 0] = 1
    percent = 1 - x.sum() / results.shape[0]
    return percent

