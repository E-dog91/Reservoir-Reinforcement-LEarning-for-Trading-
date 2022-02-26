import warnings

# Import
import torch

from rc_class import util
from matrix_generator.matrix_generator import Matrix_generator


class Matrix_gauss_gen(Matrix_generator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate matrix according to multivariate gaussian.

        Args:
            size (tuple): size of each dimension of the matrix.
            dtype:

        Returns:

        """
        return torch.randn(size, dtype=dtype)

