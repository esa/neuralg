from dotmap import DotMap
from ..utils.set_up_torch import set_up_torch
from ..utils.count_parameters import count_parameters
from ..io.get_model import get_model
import matplotlib.pyplot as plt
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


def run_batch_study():
    set_up_torch(torch_enable_cuda=True)
    torch_model = lambda x: torch.linalg.eigvals(x)

    results = DotMap()
    results.neuralg = []
    results.torch = []
    for d in torch.arange(3, 20 + 1, 2):
        neuralg_model = get_model("eig", d, symmetric=True)
        results.neuralg.append(runtime_per_batch(neuralg_model, d))
        results.torch.append(runtime_per_batch(torch_model, d))
    return results


def estimate_runtime(model, d, N):
    matrices = torch.rand(N, 1, 1, d, d)
    start_time = time.time()
    for i in range(N):
        model(matrices[i])
    ms_per_matrix = time.time() - start_time
    return ms_per_matrix


def runtime_per_batch(model, d, k=100):
    mean = 0
    for i in range(k):
        batch = torch.rand(100, 1, d, d)
        start_time = time.time()
        model(batch)
        ms_per_batch = 1000 * (time.time() - start_time)
        mean += ms_per_batch
    return mean / k


if __name__ == "__main__":
    # results = run_time_study(10, "eig", symmetric=True)
    # now = datetime.now()
    # dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    # torch.save(results, "timeresults" + dt_string + ".pt")
    # print("results for eig, 10" + str(results))

    # results = run_time_study(5, "eig")
    # print("results for c_eig, 5" + str(results))
    # results = run_time_study(20, "eig", symmetric=True)
    # print("results for eig, 20" + str(results))
    # results = run_time_study(20, "svd")
    # print("results for svd, 20" + str(results))

    batch_results = run_batch_study()
    print(batch_results)
    fig = plt.figure(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("white")

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        }
    )
    ax = fig.add_subplot()
    ax.plot(torch.arange(3, 20 + 1, 2), batch_results.neuralg, label="neuralg")
    ax.plot(torch.arange(3, 20 + 1, 2), batch_results.torch, label="torch")
    ax.set_title("Comutation time per batch, batch size = 100", fontsize=25, c="black")
    ax.set_xticks(torch.arange(3, 5 + 1, 2))
    ax.set_xlabel("Matrix size", fontsize=18)
    ax.set_ylabel("Time [ms]", fontsize=16)
    ax.legend()
    plt.show()
