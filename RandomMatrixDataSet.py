import torch
import numpy as np
class RandomMatrixDataSet:
    def __init__(self, N, d=3, operation=torch.linalg.inv):
        self.N = N
        self.d = d
        self.X = None
        self.Y = None
        self.operation = operation

    def from_condition_number(self, cond):
        self.X = SingularvalueMatrix(self.N, self.d, cond).X
        self.Y = self.operation(self.X)

    def from_eigenvalues(self, eigenvalues=None, mu=1, sigma=0.2, diagonal=False, similar=True, ):
        self.X = EigenMatrix(self.N, self.d, eigenvalues, mu, sigma, diagonal, similar).X
        self.Y = self.operation(self.X)

    def get_error(self, model):
        id = torch.eye(self.d)
        return (torch.matmul(model(self.X), self.Y) - id).square().sum((2, 3)).detach().numpy()

    def get_cond(self):
        return torch.linalg.cond(self.X).detach().numpy()


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
        # torch.random.seed = 101
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

    def matrix_from_eigenvalues(self, eigenvalues=None, mu=None, sigma=None, similar=True, diagonal=False):
        # Generate N invertible matrices of dimension d
        # Generate eigenvalues of matrices of reasonable size, close to eachother
        # Allows specifying the eigenvalues of the matrices by passing
        if eigenvalues != None:
            x = eig.repeat(self.N, 1)
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
                M = torch.rand(self.N, 1, self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
            # Do similarity transform with same basis
            else:
                M = torch.rand(self.d, self.d)
                X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
        else:
            X = diag

        self.X = X