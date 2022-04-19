from .evaluate_model import evaluate_model
from .compute_accuracy import compute_accuracy
from ..ops.eigvals import eigvals
from ..ops.svd import svd
from dotmap import DotMap
from itertools import compress


def compare_eig_run(run_cfg, symmetric, real, tol=0.05):
    """Compare training run results with existing neuralg eigvals() models. Prints wether the run outperforms the module or not.

    Args:
        run_cfg (DotMap): Post-training run configuration
        symmetric (bool): Specifies if models in run configuration are specalized to symmetric matrices
        real (bool): Specifies if models in run configuration are specalized to only predict real eigenvalues.
        tol (float, optional): Tolerance to compute accuracy with. Defaults to 0.05.

    Returns:
        DotMap: Results containing computed accuracies and list of bools for eigvals run. True corresponds to improved model perfomance.
    """
    test_parameters = {"N": 10000, "operation": "eig"}
    if symmetric:
        test_parameters[
            "wigner"
        ] = "True"  # Test performance on Wigner matrices with eigvals with variance as in training.
    elif real:
        test_parameters[
            "dist"
        ] = "gaussian"  # Test performance on random matrices with normally distributed eigenvalues
        test_parameters["symmetric"] = False
    # Test performance on random matrices with uniformly distributed elements

    op = lambda x: eigvals(x, symmetric=symmetric, real=real)

    return _compare_with_neuralg(run_cfg, op, test_parameters, tol)


def compare_svd_run(run_cfg, tol=0.05):
    """Compare training run results with existing neuralg svd() models. Prints wether the run outperforms the module or not.
    Args:
        run_cfg (DotMap): Post-training run configuration
        tol (float, optional): Tolerance to compute accuracy with. Defaults to 0.05.

    Returns:
        DotMap: Results containing computed accuracies and list of bools for svd run. True corresponds to improved model perfomance.

    """
    test_parameters = {"N": 10000, "operation": "svd"}
    op = svd
    return _compare_with_neuralg(run_cfg, op, test_parameters, tol)


def _compare_with_neuralg(run_cfg, op, test_parameters, tol):
    """Compare training run results with existing neuralg operation with passed test_parameters.
     Prints wether the run outperforms the module or not.

    Args:
        run_cfg (DotMap): Post-training run configuration
        op (function): Neuralg operator to compare with, e.g. svd or eigvals
        test_parameters (dict): Parameters for test set generation
        tol (float): Tolerance to compute accuracy with.

    Returns:
    DotMap: Results containing computed accuracies and list of bools. True corresponds to improved model perfomance.

    """
    results = DotMap()
    results.is_improved = []
    results.accuracy = []

    for size in run_cfg.matrix_sizes:
        test_parameters["d"] = size  # Matrix dimension
        model = run_cfg[size].model
        trained_acc, neuralg_acc = _compare_model(model, op, test_parameters, tol)
        print(trained_acc, neuralg_acc)
        results.is_improved.append(bool(trained_acc - neuralg_acc > 0))
        results.accuracy.append([trained_acc, neuralg_acc])

    if any(results.is_improved):
        print(
            "Performance improved for matrix sizes {} with accuracies {}, resepectively. Test parameters: {}".format(
                list(compress(run_cfg.matrix_sizes, results.is_improved)),
                list(compress(results.accuracy, results.failed)),
                test_parameters,
            )
        )
    else:
        print("No trained models improved neuralg performance on test set")

    return results


def _compare_model(model, op, test_parameters, tol):
    """Compare a model to neuralg operator with passed test set parameters.

    Args:
        model (torch.nn): Model to compare to neuralg model
        op (function): Neuralg operator to compare with, e.g. svd or eigvals
        test_parameters (dict): Parameters for test set generation
        tol (float): Tolerance to compute accuracy with.

    Returns:
        tensor: two tensors containing accuracy results of passed model an neuralg operator.
    """
    trained_acc = round(
        compute_accuracy(tol, evaluate_model(model, test_parameters)).item(), 3
    )
    neuralg_acc = round(
        compute_accuracy(tol, evaluate_model(op, test_parameters)).item(), 3
    )
    return trained_acc, neuralg_acc
