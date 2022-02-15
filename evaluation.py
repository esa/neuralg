from RandomMatrixDataSet import *
import torch

def model_evaluation(model, test_parameters, test_set=None):
    """ Evaluate model on matrix test set for a set of matrix parameters

    Args:
        model (torch.nn): trained model
        test_parameters (dict): parameters specifying what matrices to generate
        test_set (torch.Tensor, optional): Predefined test set of matrices
    Returns:
        dict : Resulting errors from model predictions on test set
    """
    if not test_set:
        test_set = get_test_set(test_parameters)

    results = evaluate_on_test_set(model, test_set)
    return results

def get_test_set(test_parameters):
    return get_sample(test_parameters)


def evaluate_on_test_set(model, test_set):
    results = {"errors": None,
              "mean_error": None}
    # Get test errors
    if isinstance(test_set, RandomMatrixDataSet):
        errors = test_set.get_error(model)
        results["errors"] = errors
        results["mean_error"] = np.mean(errors)
    else:
        None
        # To have option to test other data sets not of RandomMatrixDataSet type.
        # error = (torch.matmul(model(test_set[features]), test_set[labels]) - id).square().sum((2, 3)).detach().numpy()
    return results