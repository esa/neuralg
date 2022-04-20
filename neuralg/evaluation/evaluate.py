from dotmap import DotMap
from copy import deepcopy
from ..evaluation.evaluate_model import evaluate_model
from ..evaluation.compute_accuracy import compute_accuracy


def evaluate(run_cfg):
    """ Evaluate a training run with the passed configurations

    Args:
        run_cfg (DotMap): Post-training configurations

    Returns:
        DotMap: Passed run configuration with added evaluation results 
    """

    ms = run_cfg.matrix_sizes

    no_models = len(ms)

    for j in range(no_models):
        d = ms[j]  # Matrix dimension
        model = run_cfg[d].model
        run_cfg[d] = deepcopy(_load_test_settings(run_cfg[d]))
        # Initialize a DotMap for every different test set setting
        run_cfg[d].test_results = DotMap()
        temp_test_settings = deepcopy(run_cfg[d].test_cfg.test_settings)

        for i, k in enumerate(temp_test_settings):
            test_parameters = temp_test_settings[k]
            test_parameters["d"] = d

            results = evaluate_model(model, test_parameters)
            acc = []
            for tol in run_cfg[d].test_cfg.tolerances:
                acc.append(compute_accuracy(tol, results).item())

            run_cfg[d].test_results[k] = acc

    return run_cfg


def _load_test_settings(run_cfg):
    """ Loads the matrix parameters for test sets to the run configuration given test configurations

    Args:
        run_cfg (DotMap): Run configuration containing test setup configurations

    Returns:
        DotMap: Run configuration containing test matrix parameters to enable generating test sets
    """
    run_cfg.test_cfg.test_settings = {}
    test_parameters = {
        "N": run_cfg.test_cfg["N"],
        "operation": run_cfg.batch_parameters.operation,
    }
    if run_cfg.test_cfg.wigner:
        test_parameters["wigner"] = True
        run_cfg.test_cfg.test_settings["wigner"] = deepcopy(test_parameters)

    test_parameters["wigner"] = False

    if "distributions" in run_cfg.test_cfg:
        for dist in run_cfg.test_cfg.distributions:
            test_parameters["dist"] = dist
            if "symmetric" in run_cfg.test_cfg:
                test_parameters["symmetric"] = deepcopy(run_cfg.test_cfg["symmetric"])
            run_cfg.test_cfg.test_settings[str(dist)] = deepcopy(test_parameters)
    return run_cfg

