import torch
## Different loss functions

def inv_MSE(predicted,x):
    id = torch.eye(x.shape[2])
    return (torch.matmul(predicted,x) - id).square().mean()
def inv_RMSE(predicted,x):
    id = torch.eye(x.shape[2], x.shape[2])
    return torch.sqrt(inv_MSE(predicted,x))

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
# Add som abs loss and relative/scaled losses


