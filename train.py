from losses import *
import numpy as np
from collections import deque
from RandomMatrixDataSet import get_sample

def train_on_batch(batch, model, loss_fcn, optimizer, scheduler=None):
    if batch.X_with_det is not None: 
        pred = model(batch.X_with_det)
    elif batch.X_with_permutations is not None:
        pred = model(batch.X_with_permutations)
    else:
        pred = model(batch.X)
    if loss_fcn == eigval_error:
        batch.compute_labels()
        max_eigvals = torch.max(torch.real(batch.Y[0]),2)[0] #Need to cast as real  
        sorted_eigvals = torch.sort(torch.real(batch.Y[0]),2)[0] #For full eigenvalue decomposition
        loss = loss_fcn(pred, sorted_eigvals)
    elif loss_fcn == inv_MSE or loss_fcn == inv_RMSE or loss_fcn == inv_frobenius or loss_fcn == inv_MAE or loss_fcn == relative_inv_MSE or loss_fcn == cond_scaled_inv_MSE:
        loss = loss_fcn(pred, batch.X)
    else:
        loss = loss_fcn(pred, batch.Y)

    # Zero the gradient
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # Perform a step in LR scheduler to update LR
    if scheduler:
        scheduler.step(loss.item())
    return loss


def run_training(k,model,loss_fcn,optimizer,matrix_parameters):
    # When a new network is created we init empty training logs
    loss_log = []
    eval_loss_log = []
    weighted_average_log = []
    weighted_average = deque([], maxlen=100)

    # And store the best results
    best_loss = np.inf
    best_model_state_dict = model.state_dict()

    # We sample some data to do evaluation during training

    eval_set = get_sample(matrix_parameters)
  

    for i in range(k):

        # Sample random matrices

        batch = get_sample(matrix_parameters)

        # Compute loss
        loss = train_on_batch(batch, model, loss_fcn, optimizer)

        # We store the model if it has the lowest fitness
        # (this is to avoid losing good results during a run that goes wild)
        if loss < best_loss:
            best_model_state_dict = model.state_dict()
            best_loss = loss
            # print('New Best: ', loss.item())

        # Update the loss trend indicators
        weighted_average.append(loss.item())

        # Update the logs
        weighted_average_log.append(np.mean(weighted_average))
        loss_log.append(loss.item())
        if i % 100 == 0:
            if matrix_parameters["det"] or matrix_parameters["det_channel"] is True:
                pred_on_eval = model(eval_set.X_with_det)
            elif "permutations" in matrix_parameters: 
                pred_on_eval = model(eval_set.X_with_permutations)
            else:   
                pred_on_eval = model(eval_set.X)
            if loss_fcn == eigval_error:
                eval_set.compute_labels()
                sorted_eigvals = torch.sort(torch.real(eval_set.Y[0]),2)[0]
                eval_loss = loss_fcn(pred_on_eval,sorted_eigvals)
            else:
                eval_loss = loss_fcn(pred_on_eval, eval_set.X)
            eval_loss_log.append(eval_loss)

        # Print every i iterations
        if i % 1000 == 0:
            wa_out = np.mean(weighted_average)
            print(f"It={i}\t loss={loss.item():.3e}\t  weighted_average={wa_out:.3e}\t eval_loss={eval_loss:.3e}\t")

    return model, loss_log, weighted_average_log, eval_loss_log, eval_set