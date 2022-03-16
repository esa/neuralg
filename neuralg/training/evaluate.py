import torch
from dotmap import DotMap
from neuralg.training.losses import relative_L1_evaluation_error
from neuralg.training.get_sample import get_sample
from copy import deepcopy


def evaluate(run_cfg):
    """ Evaluate a training run with the passed configurations

    Args:
        run_cfg (DotMap): Post-training configurations

    Returns:
        DotMap: Passed run configuration with added evaluation results 
    """

    ms = run_cfg.matrix_sizes

    no_models = len(ms)
    # print("models = " + str(no_models))

    for j in range(no_models):
        d = ms[j]  # Matrix dimension
        model = run_cfg[str(d)].model
        run_cfg[str(d)] = deepcopy(_load_test_settings(run_cfg[str(d)]))
        # Initialize a DotMap for every different test set setting
        run_cfg[str(d)].test_results = DotMap()
        temp_test_settings = deepcopy(run_cfg[str(d)].test_cfg.test_settings)
        # print("test sets = " + str(no_test_sets))
        for i, k in enumerate(temp_test_settings):
            test_parameters = temp_test_settings[k]
            test_parameters["d"] = d
            # print(test_parameters)
            results = _evaluate_model(model, test_parameters)
            acc = []
            for tol in run_cfg[str(d)].test_cfg.tolerances:
                acc.append(_compute_accuracy(tol, results).item())
            # print(acc)
            run_cfg[str(d)].test_results[k] = acc

    return run_cfg


def _load_test_settings(run_cfg):
    run_cfg.test_cfg.test_settings = {}
    test_parameters = {
        "N": run_cfg.test_cfg["N"],
        "operation": run_cfg.batch_parameters.operation,
    }
    if run_cfg.test_cfg.wigner:
        test_parameters["wigner"] = True
        run_cfg.test_cfg.test_settings["wigner"] = deepcopy(test_parameters)

    test_parameters["wigner"] = False
    for dist in run_cfg.test_cfg.distributions:
        test_parameters["dist"] = dist
        run_cfg.test_cfg.test_settings[str(dist)] = deepcopy(test_parameters)
    return run_cfg


def _evaluate_model(model, test_parameters, test_set=None):
    """Evaluate model on matrix test set for a set of matrix parameters
    Args:
        model (torch.nn): trained model
        test_parameters (dict): parameters specifying what matrices to generate
        test_set (torch.Tensor, optional): Predefined test set of matrices
    Returns:
        dict : Resulting errors from model predictions on test set
    """
    if test_parameters["operation"] == "eig":
        test_parameters["operation"] = torch.linalg.eig
        test_set = _get_test_set(test_parameters)
        test_set.compute_labels()
        results = _evaluate_eigval_model(model, test_set)
    else:
        raise ValueError("Evaluation for this operation is not supported")
    return results


def _get_test_set(test_parameters):
    torch.manual_seed(458)
    return get_sample(test_parameters)


def _evaluate_eigval_model(model, test_set):
    eigvals = torch.sort(torch.real(test_set.Y[0]), 2)[0]
    predicted_eigvals = model(test_set.X)
    return relative_L1_evaluation_error(predicted_eigvals, eigvals)


def _compute_accuracy(tol, results):
    """ Compute the accuracy given some tolerance and prediction errors on test set 

    Args:
        TOL (_type_): Tolerance level, should be between zero and one
        results (_type_): Array of prediction errors

    Returns:
        _type_: Resulting prediction accuracy on the test set
    """

    x = torch.zeros(results.shape[0])
    y = (results - tol).squeeze()
    x[y > 0] = 1
    percent = 1 - x.sum() / results.shape[0]
    return percent

