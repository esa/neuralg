from math import sqrt
import torch
from .RandomMatrix import RandomMatrix


class RandomMatrixDataSet:
    """Data set of random matrix samples and corresponding labels according to passed linear algebra operation, e.g. inverses, eigenvalues.
    Sampling includes matrices with uniform and gaussian elements, drawing eigenvalues from probability distributions and/or random symmetric matrices.
    """

    def __init__(self, N, d, operation):
        self.N = N
        self.d = d
        self.X = None
        self.Y = None
        self.operation = operation
        self.cond = None
        self.det = None

    def from_rand(self, r1=-10, r2=10):
        """Sample matrices with uniformly distributed elements

        Args:
            r1 (int, optional): Lower interval bound. Defaults to -10.
            r2 (int, optional): Upper interval bound. Defaults to 10.
        """
        self.X = (r1 - r2) * torch.rand(self.N, 1, self.d, self.d) + r2

    def from_randn(self, sigma=10 / sqrt(3)):
        """Sample matrices with normally distributed elements

        Args:
            sigma (float, optional): Standard deviation of elements. Defaults to 10/sqrt(3).
        """
        self.X = sigma * torch.randn(self.N, 1, self.d, self.d)

    def from_dist(self, dist, symmetric=True):
        """Sample matrices with eigenvalues according to passed distribution. Supports symmetric and matrix generation.

        Args:
            dist (str): Eigenvalue probability distribution, supports "gaussian", "laplace" and "uniform".
            symmetric (bool, optional): If true, generated matrices are symmetric. Defaults to True.
        """

        self.X = RandomMatrix(N=self.N, d=self.d, dist=dist, is_symmetric=symmetric).X

    def compute_labels(self):
        """Compute labels according to passed operation"""
        self.Y = self.operation(self.X)

    def compute_determinant(self):
        """Compute determinant of matrix examples"""
        self.det = torch.linalg.det(self.X)

    def compute_cond(self):
        """Compute condition number of matrix examples"""
        self.cond = torch.linalg.cond(self.X).detach().numpy()
