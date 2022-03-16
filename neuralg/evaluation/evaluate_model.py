import torch
from neuralg.training.losses import relative_L1_evaluation_error
from neuralg.training.get_sample import get_sample


def evaluate_model(model, test_parameters, test_set=None):
    """Evaluate model on matrix test set for a set of matrix parameters
    Args:
        model (torch.nn): trained model
        test_parameters (dict): parameters specifying what matrices to generate
        test_set (torch.Tensor, optional): Predefined test set of matrices
    Returns:
        dict : Resulting errors from model predictions on test set
    """
    assert model is not None, "Model can not be None type"
    if test_parameters["operation"] == "eig":
        test_parameters["operation"] = torch.linalg.eig
        test_set = _get_test_set(test_parameters)
        test_set.compute_labels()
        results = _evaluate_eigval_model(model, test_set)
    else:
        raise ValueError("Evaluation for this operation is not supported")
    return results


def _get_test_set(test_parameters):
    """Samples a batch of random matrices with a fixed seed

    Args:
        test_parameters (DotMap): Parameters characterizing the requested matrix data set

    Returns:
        RandomMatrixDataSet: Data set of random matrices 
    """
    torch.manual_seed(458)
    return get_sample(test_parameters)


def _evaluate_eigval_model(model, test_set):
    """Computes relative L1 norm of evaluation error for eigenvalue predictions

    Args:
        model (nn.torch): Model to evaluate
        test_set (RandomMatrixDataSet): Data set to evaluate model on

    Returns:
        tensor: Relative L1 norm of the difference between predicted and true eigenvalues of the test set matrices
    """
    assert test_set is not None, "Test is None"
    assert model is not None, "Model is None"
    eigvals = torch.sort(torch.real(test_set.Y[0]), 2)[0]
    predicted_eigvals = model(test_set.X)
    assert predicted_eigvals is not None, "Evaluation non-succesful"
    return relative_L1_evaluation_error(predicted_eigvals, eigvals)
