from dotmap import DotMap

NEURALG_MATRIX_SIZES = DotMap()

NEURALG_MATRIX_SIZES.eig.sym.lower_bound = 2
NEURALG_MATRIX_SIZES.eig.sym.upper_bound = 20

NEURALG_MATRIX_SIZES.eig.real.lower_bound = 2
NEURALG_MATRIX_SIZES.eig.real.upper_bound = 10

NEURALG_MATRIX_SIZES.eig.complex.lower_bound = 2
NEURALG_MATRIX_SIZES.eig.complex.upper_bound = 5

NEURALG_MATRIX_SIZES.svd.lower_bound = 2
NEURALG_MATRIX_SIZES.svd.upper_bound = 20


NEURALG_SUPPORTED_OPERATIONS = ["eig", "svd"]
