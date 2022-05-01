from dotmap import DotMap
from ..utils.set_up_torch import set_up_torch
from ..utils.count_parameters import count_parameters
from ..io.get_model import get_model

import torch
from loguru import logger
import time
from datetime import datetime


def run_time_study(d, operation, symmetric=False):
    N = 1000
    set_up_torch(torch_enable_cuda=True)
    neuralg_model = get_model(operation, d, symmetric)
    if operation == "eig":
        torch_model = lambda x: torch.linalg.eigvals(x)
    elif operation == "svd":
        torch_model = lambda x: torch.linalg.svdvals(x)
    results = DotMap()
    results.neuralg = [
        estimate_runtime(neuralg_model, d, N),
        runtime_per_batch(neuralg_model, d),
    ]
    results.torch = [
        estimate_runtime(torch_model, d, N),
        runtime_per_batch(torch_model, d),
    ]

    return results


def estimate_runtime(model, d, N):
    matrices = torch.rand(N, 1, 1, d, d)
    start_time = time.time()
    for i in range(N):
        model(matrices[i])
    ms_per_matrix = time.time() - start_time
    return ms_per_matrix


def runtime_per_batch(model, d):
    mean = 0
    k = 100
    for i in range(k):
        batch = torch.rand(500, 1, d, d)
        start_time = time.time()
        model(batch)
        ms_per_batch = 1000 * (time.time() - start_time)
        mean += ms_per_batch
    return mean / k


if __name__ == "__main__":
    results = run_time_study(10, "eig", symmetric=True)
    # now = datetime.now()
    # dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    # torch.save(results, "timeresults" + dt_string + ".pt")
    print("results for eig, 10" + str(results))

    results = run_time_study(5, "eig")
    print("results for c_eig, 5" + str(results))
    results = run_time_study(20, "eig", symmetric=True)
    print("results for eig, 20" + str(results))
    results = run_time_study(20, "svd")
    print("results for svd, 20" + str(results))
