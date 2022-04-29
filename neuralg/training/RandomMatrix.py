import torch
import math


class RandomMatrix:
    """N-sized batch of random dxd-sized matrices with eigenvalues from a passed probability distribution.
    Supports generating symmetric matrices.
    """

    def __init__(
        self, N, d, sigma=10 / math.sqrt(3), dist="gaussian", is_symmetric=True
    ):
        self.N = N
        self.d = d
        self.sigma = math.sqrt(self.d) * sigma  # Std of Wigner matrices
        self.dist = dist
        self.X = None
        self.is_symmetric = is_symmetric
        self.from_distribution(dist=self.dist, is_symmetric=self.is_symmetric)

    def from_distribution(self, dist="gaussian", is_symmetric=True):
        """

        Args:
            dist (str, optional): Distribution of eigenvalues.  Defaults to "gaussian".
            is_symmetric (bool, optional): If true, sampled matrices will be symmetric. Defaults to True.
        """
        # Sample new eigenvalues
        if self.dist == "gaussian":
            x = self.sigma * torch.randn(self.N, self.d, 1)
        elif self.dist == "uniform":
            x = (
                -2 * math.sqrt(self.d) * 10 * torch.rand(self.N, self.d, 1)
                + math.sqrt(self.d) * 10
            )
        elif self.dist == "laplace":
            m = torch.distributions.Laplace(
                torch.tensor([0.0]), torch.tensor([self.sigma / math.sqrt(2)])
            )
            x = m.rsample(sample_shape=torch.Size([self.N, self.d]))
        diag = torch.eye(x.shape[1]) * x[:, None]

        M = torch.randn(self.N, 1, self.d, self.d)
        if is_symmetric:
            # Create symmetric matrices
            M = torch.triu(M, 0) + torch.transpose(torch.triu(M, 1), 2, 3)
            # Compute eigenvectors of hermitian matrix
            P = torch.linalg.eigh(M)[1]
            # Finally, construct the resulting matrix batch with specified eigenvalues
            self.X = torch.matmul(torch.matmul(P, diag), torch.transpose(P, 2, 3))
        else:
            self.X = torch.matmul(torch.matmul(M, diag), torch.linalg.inv(M))
