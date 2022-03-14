from matplotlib import set_loglevel
import torch
from dotmap import DotMap
import numpy as np
from collections import deque
from loguru import logger
from copy import deepcopy
import os
from neuralg.training.losses import eigval_L1
from neuralg.training.RandomMatrixDataSet import RandomMatrixDataSet
from neuralg.training.get_sample import get_sample
from neuralg.utils.set_log_level import set_log_level


import neuralg.models

"""  TODO
[] Figure out why importing only works with neuralg.
[] Improve train_cfg dotmap argument to contain:
    - Option for optimizer and scheduler
[] Improve logging
[] Finish Docstrings
[] Clean up
"""


def _train_on_batch(batch, model, loss_fcn, optimizer):
    """ Update model parameters from forward and backward pass on a batch

    Args:
        batch (RandomMatrixDataSet): Batch of matrices to backpropagate loss from
        model (torch.nn): Model to train on batch
        loss_fcn (function): Function to compute loss on batch
        optimizer (torch.opt): _description_

    Returns:
        _type_: loss from model forward pass on batch 
    """
    pred = model(batch.X)

    if loss_fcn == eigval_L1:
        batch.compute_labels()
        sorted_eigvals = torch.sort(torch.real(batch.Y[0]), 2)[0]
        loss = loss_fcn(pred, sorted_eigvals)
    else:
        batch.compute_labels()
        loss = loss_fcn(pred, batch.Y)

    # Zero the gradient
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss


def _init_training(train_cfg):
    """ Initializes necessary 

    Args:
        train_cfg (DotMap): Configurations for training, must include a model

    Returns:
       DotMap, torch.opt, torch.scheduler: Run config with  optimizer and scheduler
    """
    # Initialize optimizer and scheduler.
    # Maybe this should also be optional so we can try different ones
    optimizer = torch.optim.Adam(
        train_cfg.model.parameters(), lr=train_cfg.run_params.lr
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    return train_cfg, optimizer, scheduler


def run_training(train_cfg):
    """ Does a full training run given a model an training configurations. 

    Args:
        train_cfg (DotMap): Training run configurations, including matrix characteristics and training parameters

    Returns:
        DotMap: Results from training run, including loss logs, trained model and best model state dict
    """
    logger.trace(
        "Initializing training..."  # Maybe specify the model and config etc...?
    )

    loss_fcn = train_cfg.run_params.loss_fcn
    if loss_fcn == "eigval_L1":
        loss_fcn = eigval_L1
    matrix_parameters = deepcopy(train_cfg.batch_parameters)
    if matrix_parameters["operation"] == "eig":
        matrix_parameters["operation"] = torch.linalg.eig

    train_cfg, optimizer, scheduler = _init_training(train_cfg)

    # Initialize empty training logs
    train_cfg.results = DotMap()
    train_cfg.results.loss_log = []
    train_cfg.results.eval_loss_log = []
    train_cfg.results.weighted_average_log = []
    weighted_average = deque([], maxlen=100)

    # And store the best results
    best_loss = np.inf
    train_cfg.best_model_state_dict = train_cfg.model.state_dict()

    # We sample some data to do evaluation during training
    eval_set = get_sample(matrix_parameters)

    supported_sizes = [3, 4, 5, 6, 7, 8, 9, 10]

    distributions = ["gaussian", "laplace", "uniform"]
    logger.trace("Starting training...")
    for e in range(1, train_cfg.run_params.epoch + 1):
        for i in range(train_cfg.run_params.iterations):

            # If we want batches to swicth between different eigenvalue distributions
            if train_cfg.run_params.mixed_eigval_distributions:
                matrix_parameters["dist"] = np.random.choice(distributions)

            # Sample random matrices
            batch = get_sample(matrix_parameters)

            # Compute loss
            loss = _train_on_batch(batch, train_cfg.model, loss_fcn, optimizer)

            # Store the model if it decreased the lowest loss
            if loss < best_loss:
                train_cfg.best_model_state_dict = deepcopy(train_cfg.model.state_dict())
                best_loss = loss
                # print('New Best: ', loss.item())

            # Update the loss trend indicators
            weighted_average.append(loss.item())

            # Update the logs
            train_cfg.results.weighted_average_log.append(np.mean(weighted_average))
            train_cfg.results.loss_log.append(loss.item())

            # Compute evaluation
            if i % 100 == 0:
                pred_on_eval = train_cfg.model(eval_set.X)
                eval_set.compute_labels()
                if loss_fcn == eigval_L1:
                    sorted_eigvals = torch.sort(torch.real(eval_set.Y[0]), 2)[0]
                    eval_loss = loss_fcn(pred_on_eval, sorted_eigvals)
                else:
                    eval_loss = loss_fcn(pred_on_eval, eval_set.Y)
                train_cfg.results.eval_loss_log.append(eval_loss.item())

            # Print every 1000 iterations
            if i % 1000 == 0 and i > 0:
                lr = scheduler.get_last_lr()[0]
                wa_out = np.mean(weighted_average)
                print(
                    f"epoch={e} \t It={i}\t loss={loss.item():.3e}\t lr={lr:0.3e} \t weighted_average={wa_out:.3e} eval_loss={eval_loss:.3e}\t"
                )
        if scheduler is not None:
            scheduler.step()

    logger.trace("Reloading best model state...")

    # Return best model in the end
    train_cfg.model.load_state_dict(train_cfg.best_model_state_dict)

    return train_cfg
