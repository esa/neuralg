import torch
from neuralg.training.RandomSymmetricMatrix import RandomSymmetricMatrix


class RandomMatrixDataSet:
    def __init__(self, N, d, operation):
        self.N = N
        self.d = d
        self.X = None
        self.Y = None
        self.operation = operation
        self.cond = None
        self.det = None

    def from_rand(self, r1=-10, r2=10):
        self.X = (r1 - r2) * torch.rand(self.N, 1, self.d, self.d) + r2

    def from_randn(self):
        self.X = torch.randn(self.N, 1, self.d, self.d)

    def from_dist(self, dist):
        self.X = RandomSymmetricMatrix(N=self.N, d=self.d, dist=dist).X

    def compute_labels(self):
        self.Y = self.operation(self.X)

    def compute_determinant(self):
        self.det = torch.linalg.det(self.X)

    def compute_cond(self):
        self.cond = torch.linalg.cond(self.X).detach().numpy()
