from dotmap import DotMap
from ..utils.set_up_torch import set_up_torch
from ..utils.count_parameters import count_parameters
from ..models.nerf import EigNERF
from ..models.siren import EigSiren
from ..training.train_model import train_model
from ..training.save_run import save_run
from ..utils.set_log_level import set_log_level
import torch
from loguru import logger


def run_accuracy_study(d, model_type, nerf_activation=torch.nn.ReLU()):
    set_log_level("TRACE")
    set_up_torch(torch_enable_cuda=True)
    batch_size = 64
    train_matrix_parameters = DotMap(
        {"N": batch_size, "operation": "eig", "d": d, "wigner": True}
    )

    run_params = DotMap(
        {
            "epoch": 10,  # Number of epochs
            "iterations": 10000,  # Batches per epoch
            "lr": 3e-4,
        }
    )  # Learning rate

    training_runs = DotMap(
        {"models": [], "no_parameters": [], "loss_logs": [], "accuracy": []}
    )

    for hidden_layers in torch.arange(5, 11, 2):
        skip = torch.arange(2, hidden_layers - 1, 2)
        for n_neurons in [25, 50, 100, 200]:
            if model_type == "nerf":
                model = EigNERF(
                    d,
                    d ** 2,
                    n_neurons=n_neurons,
                    skip=skip,
                    hidden_layers=hidden_layers,
                    activation=nerf_activation,
                )
            elif model_type == "siren":
                model = EigSiren(
                    d, hidden_features=n_neurons, hidden_layers=hidden_layers
                )

            logger.trace(
                f"Initializing training model with {hidden_layers} hidden layers, {n_neurons} neurons, skipping layer(s) {skip}"
            )
            training_run = train_model(
                model, train_matrix_parameters, run_parameters=run_params
            )
            training_runs.no_parameters.append(count_parameters(model))
            training_runs.models.append(training_run.model)
            training_runs.loss_logs.append(training_run.results)

    return training_runs


if __name__ == "__main__":
    save_run(run_accuracy_study(d=5, model_type="nerf"), subfolder="nerf")
    save_run(
        run_accuracy_study(d=5, model_type="nerf", nerf_activation=torch.nn.Sigmoid()),
        subfolder="nerf",
    )
    save_run(run_accuracy_study(d=10, model_type="nerf"), subfolder="nerf")

    save_run(run_accuracy_study(d=5, model_type="siren"), subfolder="siren")
    save_run(run_accuracy_study(d=10, model_type="siren"), subfolder="siren")
