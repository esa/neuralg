def eigval_MAE(pred, x):
    """Mean absolute error loss

    Args:
        pred (tensor): Predictions from model
        x (tensor): Correct responses

    Returns:
        tensor: Mean absolute difference between model prediction and responses
    """
    e = (pred - x).abs().mean()
    return e


def eigval_L1(pred, x):
    """Mean relative L1 error loss function

    Args:
        pred (tensor): Predictions from model
        x (tensor): Correct responses

    Returns:
        tensor: Mean relative L1 norm of difference between model prediction and responses
    """
    return relative_L1_evaluation_error(pred, x).mean()


def relative_L1_evaluation_error(pred, x):
    """Relative L1 error function

    Args:
        pred (tensor): Predictions from model
        x (tensor): Correct responses

    Returns:
        tensor: Relative L1 norm of difference between model prediction and responses
    """
    return (pred - x).abs().sum(-1) / (x.abs().sum(-1))


def eigval_L2(pred, x):
    """Mean relative L2 error loss function

    Args:
        pred (tensor): Predictions from model
        x (tensor): Correct responses

    Returns:
        tensor: Mean relative L2 norm of difference between model prediction and responses
    """
    return relative_L2_evaluation_error(pred, x).mean()


def relative_L2_evaluation_error(pred, x):
    """Relative L2 error function

    Args:
        pred (tensor): Predictions from model
        x (tensor): Correct responses

    Returns:
        tensor: Relative L2 norm of difference between model prediction and responses
    """
    return (pred - x).abs().square().sum(-1) / (x.abs().square().sum(-1))
