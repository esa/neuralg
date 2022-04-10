from .evaluate_model import evaluate_model
from .compute_accuracy import compute_accuracy
from ..ops.eig import eig
from ..ops.svd import svd
from dotmap import DotMap
from itertools import compress


def compare_eig_run(run_cfg, symmetric, real, tol=0.05):

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

    op = lambda x: eig(x, symmetric=symmetric, real=real)

    return _compare_with_neuralg(run_cfg, op, test_parameters, tol)


def compare_svd_run(run_cfg):
    test_parameters = {"N": 10000, "operation": "svd"}
    op = svd
    return _compare_with_neuralg(run_cfg, op, test_parameters)


def _compare_with_neuralg(run_cfg, op, test_parameters, tol):
    ms = run_cfg.matrix_sizes
    no_models = len(ms)

    results = DotMap()
    results.is_improved = []
    results.accuracy = []

    for j in range(no_models):
        d = ms[j]  # Matrix dimension
        test_parameters["d"] = d
        model = run_cfg[d].model
        trained_acc, neuralg_acc = _compare_model(model, op, test_parameters, tol)
        print(trained_acc, neuralg_acc)
        results.is_improved.append(bool(trained_acc - neuralg_acc > 0))
        results.accuracy.append([trained_acc, neuralg_acc])

    if True in results.is_improved:
        print(
            "Performance improved for matrix sizes {} with accuracies {}, resepectively. Test parameters: {}".format(
                list(compress(ms, results.is_improved)),
                list(compress(results.accuracy, results.failed)),
                test_parameters,
            )
        )
    else:
        print("No trained models improved neuralg performance on test set")

    return results


def _compare_model(model, op, test_parameters, tol):
    trained_acc = round(
        compute_accuracy(tol, evaluate_model(model, test_parameters)).item(), 3
    )
    neuralg_acc = round(
        compute_accuracy(tol, evaluate_model(op, test_parameters)).item(), 3
    )
    return trained_acc, neuralg_acc

