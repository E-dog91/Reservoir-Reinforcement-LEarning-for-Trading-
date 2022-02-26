import warnings

# Import
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from rc_class import util

# wip MAKE THIS CLASS ABSTRACT.
class Matrix_generator(object, metaclass=ABCMeta):

    def __init__(self, scale = 0.99, sparse = 0.7,  *args, **kwargs):
        super().__init__()
        self._scale = scale
        self._sparse = sparse

    def generate(self, size, dtype = torch.float64):
        """

        Args:
            size (tuple):
            dtype:

        Returns:

        """
        w = self._generate_matrix(size, dtype)
        # Set spectral radius
        if w.ndimension() == 2 and w.size(0) == w.size(1):
            if util.spectral_radius(w) > 0.0:
                w = (w / util.spectral_radius(w)) * self._scale
            else:
                warnings.warn("Spectral radius of W is zero (due to small size), spectral radius not changed")
        else:
            # Set norm of w to scaling factor
            w *= self._scale * w / (w.norm())

        return Matrix_generator.to_sparse(w, self._sparse)


    @abstractmethod
    def _generate_matrix(self, size, dtype=torch.float64):
        pass

    @staticmethod
    def to_sparse(w, p):
        """
        sparse a matrix, keeping a regular torch.tensor format, since some methods aren't
        available for the SparseTensor format
        param:
            m: matrix or vector to sparse
            p: probability that an entry is zero.
            warning: m is assumed to not have many zero entries
        """
        assert 0 <= p < 1, "p should be between 0 and 1 (excluded). p = {}.".format(p)
        return nn.Dropout(p=p)(w) # dropout the entries.
