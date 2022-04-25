from dotmap import DotMap
from .run_training import run_training


def train_model(model, batch_parameters, run_parameters):
    """ Does a training run characterized by passed run parameters, on matrix batches  with the passed mo

    Args:
        model (torch.nn): Model to be trained
        batch_parameters (DotMap): Specifying matrix batchparameters for the training data, e.g. the distribution of  matrix elements.
        run_parameters (DotMap): Specifying training related parameters, e.g. learning rate and # of epochs.

    Returns:
        DotMap: Containing trained model alongsid training results e.g. loss logs.
    """
    train_cfg = DotMap()  # Create a training config from the user input
    train_cfg.model = model
    train_cfg.batch_parameters = batch_parameters
    train_cfg.run_params = run_parameters

    if "loss" not in run_parameters:
        train_cfg.run_params.loss_fcn = "eigval_L1"
    return run_training(train_cfg)
