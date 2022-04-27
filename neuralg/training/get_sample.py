import torch
from .RandomMatrixDataSet import RandomMatrixDataSet


def get_sample(matrix_parameters):
    """Sample a batch of random matrices from passed configuration

    Args:
        matrix_parameters (dict): Parameters characterizing the batch of generated matrices

    Returns:
        RandomMatrixDataSet: Batch of random matrices with given matrix parameters
    """
    # Instantiate batch
    N, dim = matrix_parameters["N"], matrix_parameters["d"]
    if "operation" in matrix_parameters:
        op = matrix_parameters["operation"]
        M = RandomMatrixDataSet(N, dim, op)
    else:
        M = RandomMatrixDataSet(N, dim)

    if "dist" in matrix_parameters:
        if "symmetric" in matrix_parameters:
            M.from_dist(
                matrix_parameters["dist"], symmetric=matrix_parameters["symmetric"]
            )
        else:
            M.from_dist(matrix_parameters["dist"])
    else:
        if "normal" in matrix_parameters:
            # Create matrices with iid gaussian entries
            if matrix_parameters["normal"]:
                if "sigma" in matrix_parameters:
                    M.from_randn(sigma=matrix_parameters["sigma"])
                else:
                    M.from_randn()
        else:
            # Otherwise just sample a matrix with uniform iid entries
            M.from_rand()  # M.from_randn() #fix so one can choose between these
        if "wigner" in matrix_parameters:
            # Create Wigner matrix
            if matrix_parameters["wigner"]:
                M.X = torch.triu(M.X, 0) + torch.transpose(torch.triu(M.X, 1), 2, 3)

    return M
