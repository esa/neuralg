from copy import deepcopy
from loguru import logger
from dotmap import DotMap
from .run_training import run_training
from ..models.nerf import EigNERF, CEigNERF
from .save_run import save_run


def train_models(cfg, save_training_run=False):
    """Train models with settings in passed configurations

    Args:
        cfg (DotMap): Training configurations for run

    Returns:
        DotMap: Post-training configurations with added trained models and loss results
    """

    # For now, loop over requested fixed matrix sizes
    run_results = DotMap()
    for d in cfg.matrix_sizes:
        temp_config = deepcopy(cfg)
        temp_config.batch_parameters["d"] = d
        # Initialize network
        if temp_config.model_type == "nerf":
            model = EigNERF(
                d,
                d**2,
                n_neurons=temp_config.n_neurons,
                hidden_layers=temp_config.hidden_layers,
            )
        elif temp_config.model_type == "complex_nerf":
            model = CEigNERF(
                d,
                d**2,
                n_neurons=temp_config.n_neurons,
                hidden_layers=temp_config.hidden_layers,
            )
        else:
            raise NotImplementedError("Model type is not available for training")
        temp_config.model = model
        logger.trace("Calling training for matrix size {}".format(d))
        run_results[d] = run_training(temp_config)

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
