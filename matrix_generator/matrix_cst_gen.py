import warnings

# Import
import torch

from src.rc_class import util
from src.rc_class.matrix_gen.matrix_generator import Matrix_generator


class Matrix_cst_gen(Matrix_generator):

    def __init__(self, scale=0.99, *args, **kwargs):
        super().__init__(scale=scale, *args, **kwargs)

    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate matrix according to multivariate gaussian.

        Args:
            size (tuple): size of each dimension of the matrix.
            dtype:

        Returns:

        """
        return torch.ones(size, dtype=dtype)

