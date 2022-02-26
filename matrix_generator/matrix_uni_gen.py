# Import
import torch

from src.rc_class.matrix_gen.matrix_generator import Matrix_generator


class Matrix_uni_gen(Matrix_generator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_matrix(self, size, dtype=torch.float64):
        """
        Generate matrix according to multivariate gaussian.

        Args:
            size (tuple): size of each dimension of the matrix.
            dtype:

        Returns:

        """
        return torch.rand(size, dtype=dtype) * 2 - 1  # \in [-1,1]
