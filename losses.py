import torch
## Different loss functions

def inv_MSE(predicted,x):
    id = torch.eye(x.shape[2])
    return (torch.matmul(predicted,x) - id).square().mean()

def inv_RMSE(predicted,x):
    id = torch.eye(x.shape[2])
    return torch.sqrt(inv_MSE(predicted,x))

def inv_MAE(predicted,x):
    id = torch.eye(x.shape[2])
    return (torch.matmul(predicted, x) - id).abs().mean()

def relative_inv_MSE(predicted,x):
    #Normalize with batch square mean? Not sure if the best way
    id = torch.eye(x.shape[2])
    id_approx = torch.matmul(predicted,x)
    return((id_approx - id).square()/id_approx.square().mean()).mean()  #((id_approx - id).square()/x.square().mean()).mean()#

def cond_scaled_inv_MSE(predicted,x):
    #Normalize with mean batch condition no
    id = torch.eye(x.shape[2])
    id_approx = torch.matmul(predicted, x)
    return ((id_approx - id).square()/torch.linalg.cond(x).mean()).mean()

def MSE(predicted,y):
    return (predicted-y).square().mean()

def RMSE(predicted,y):
    id = torch.eye(y.shape[2])
    return torch.sqrt(MSE(predicted,y))

def inv_frobenius(predicted,x):
    id = torch.eye(x.shape[2])
    return torch.linalg.matrix_norm(torch.matmul(predicted,x)-id, ord = 'fro').mean()

def frobenius(predicted,y):
    id = torch.eye(y.shape[2])
    return torch.linalg.matrix_norm(predicted-y, ord = 'fro').mean()

##### For only eigenvalue problems.  
def eigval_error(pred,x):
    e = (pred-x).abs().mean()
    return e 
def relative_L1_evaluation_error(pred,x):
    return (pred-x).abs().sum(-1)/(x.abs().sum(-1))

def eigval_L1(pred,x):
    return relative_L1_evaluation_error(pred,x).mean()

def max_eigval_error(pred,x): 
    return eigval_error(pred,x) #Just for the train module to know how to evaluate



