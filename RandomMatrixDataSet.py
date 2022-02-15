import torch
import numpy as np

def get_sample(matrix_parameters):
    # Instantiate test set
    M = RandomMatrixDataSet(N=matrix_parameters["N"], d=matrix_parameters["d"])

    # If the condition number is specified
    if "cond" in matrix_parameters:
        M.from_condition_number(matrix_parameters["cond"])
    # If the eigenvalues are specified
    elif "eigenvalues" in matrix_parameters:
        M.from_eigenvalues(eigenvalues=matrix_parameters["eigenvalues"], diagonal=matrix_parameters["diagonal"])
    # Otherwise eigenvalues are drawn from IID normal distributions  N(mu,sigma^2)
    else:
        mu, sigma = matrix_parameters["mu"], matrix_parameters["sigma"]
        M.from_eigenvalues(mu=mu, sigma=mu, similar=matrix_parameters["similar"], diagonal=matrix_parameters["diagonal"])
    return M

class RandomMatrixDataSet:
    def __init__(self, N, d=3, operation=torch.linalg.inv):
        self.N = N
        self.d = d
        self.X = None
        self.Y = None
        self.operation = operation
        self.cond = None
        self.det = None
    
    def compute_labels(self):
        self.Y = self.operation(self.X)
    
    def from_condition_number(self, cond):
        self.X = SingularvalueMatrix(self.N, self.d, cond).X
        self.cond = cond

    def from_eigenvalues(self, eigenvalues=None, mu=1, sigma=0.2, diagonal=False, similar=True):
        self.X = EigenMatrix(self.N, self.d, eigenvalues, mu, sigma, diagonal, similar).X

    def get_error(self, model):
        id = torch.eye(self.d)
        #This could be generalized to other test errors
        return (torch.matmul(model(self.X), self.X) - id).square().mean((2, 3)).detach().numpy()
    
    def compute_determinant(self):
        self.det = torch.linalg.det(self.X)
    
    def compute_cond(self):
        self.cond = torch.linalg.cond(self.X).detach().numpy()
    


class SingularvalueMatrix():
    def __init__(self, N, d=3, cond=10):
        self.N = N
        self.d = d
        self.cond = cond
        self.X = None
        self.matrix_from_singular_values(cond=self.cond)

    def matrix_from_singular_values(self, cond):
        log_cond = np.log(self.cond)
        log_s = torch.arange(-log_cond / 4., log_cond * (self.d + 1) / (4 * (self.d - 1)),
                             log_cond / (2. * (self.d - 1)))
        log_s = log_s.repeat(self.N, 1)
        log_s = log_s[:, :, None]
        s = torch.exp(log_s)
        S = torch.eye(s.shape[1]) * s[:, None]
        U, _ = torch.linalg.qr((torch.rand(self.N, 1, self.d, self.d) - 5.) * 200)
        V, _ = torch.linalg.qr((torch.rand(self.N, 1, self.d, self.d) - 5.) * 200)
        X = torch.matmul(U, torch.matmul(S, V.transpose(2, 3)))
        X = torch.matmul(X, X.transpose(2, 3))
        self.X = X


class EigenMatrix():

    def __init__(self, N, d=3, eigenvalues=None, mu=1, sigma=0.2, diagonal=False, triangular=False, similar=True):
        self.N = N
        self.d = d
        self.mu = mu
        self.sigma = sigma
        self.eigenvalues = eigenvalues
        self.X = None
        
        if self.eigenvalues != None:
            self.matrix_from_eigenvalues(eigenvalues=self.eigenvalues, diagonal=diagonal, similar=similar)
        else:
            self.matrix_from_eigenvalues(mu=self.mu, sigma=self.sigma, diagonal=diagonal, similar=similar)

    def matrix_from_eigenvalues(self, eigenvalues= None, mu=None, sigma=None, similar=True, diagonal=False):
        #Generate N invertible matrices of dimension d. 
        #Eigenvalues for each matrix are sampled from a d IID normal distributions with mean mu and std sigma
        #Resulting matrices are generated via similarity transformatins using random matrices. 
        
        if eigenvalues != None:
            x = eigenvalues.repeat(self.N, 1)
            x = x[:, :, None]
        else:
            # Set default values if not specified
            if mu == None:
                mu = 1
            elif sigma == None:
                sigma = 0.2
            x = mu * torch.ones((self.N, self.d, 1)) + sigma * torch.randn(self.N, self.d, 1)

            # Create diagonal matrices
        diag = torch.eye(x.shape[1]) * x[:, None]

        if not diagonal:
            # Transformation matrix for similarity transform
            if not similar:
                # Creates matrices with different basis.
                M = torch.randn(self.N, 1, self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
            # Do similarity transform with same basis
            else:
                M = torch.randn(self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
        else:
            X = diag

        self.X = X
