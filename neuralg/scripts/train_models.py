from cProfile import run
from copy import deepcopy
from neuralg.training.run_training import run_training  # Maybe save run?
import torch
from loguru import logger
from dotmap import DotMap
from neuralg.models.nerf import EigNERF
from neuralg.training.losses import eigval_L1
from neuralg.training.save_run import save_run
from neuralg.utils.load_default_cfg import load_default_cfg
from neuralg.utils.set_log_level import set_log_level


""" TODO
[] Implement loading of global training configurations
[] Decide and implement how training results are saved and shipped to notebook

[] 
"""


def load_config():
    return None


def train_models(cfg, save_training_run=False):
    """ Train models with settings in passed configurations

    Args:
        cfg (DotMap): _description_

    Returns:
        DotMap: _description_
    """
    set_log_level("TRACE")

    # For now, loop over requested fixed matrix sizes
    run_results = DotMap()
    for d in cfg.matrix_sizes:
        temp_config = deepcopy(cfg)
        temp_config.batch_parameters["d"] = d
        # Initialize network
        if temp_config.model_type == "nerf":
            model = EigNERF(
                d,
                d ** 2,
                n_neurons=temp_config.n_neurons,
                hidden_layers=temp_config.hidden_layers,
            )
        else:
            raise ValueError("Model type is not available for training")
        temp_config.model = model
        logger.trace("Calling training for matrix size {}".format(d))
        run_results[str(d)] = run_training(temp_config)
        # print(run_results[str(d)].model) here it is fine but it seems like its overwriting
    logger.trace("Finalized training all requested models...")
    # For plotting
    run_results.matrix_sizes = deepcopy(cfg.matrix_sizes)
    if save_training_run:
        save_run(run_results)
    return run_results


def save_models():
    return None


# If we want this as a script. Right now Im calling it as a function
# if __name__ == "__main__":
#    cfg = load_default_cfg()
#    run_results = train_models(cfg)
