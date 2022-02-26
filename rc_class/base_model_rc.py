from abc import ABCMeta, abstractmethod

import torch
from corai import Savable_net
from corai_error import Error_type_setter

from matrix_generator.matrix_generator import Matrix_generator


class Base_model_rc(Savable_net, metaclass=ABCMeta):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 w_generator, win_generator, wbias_generator, h0_Generator, h0_params,
                 washout=0, dtype=torch.float32, **kwargs):
        super().__init__(predict_fct=None)

        # dimensions
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

        # methods to generate the parameter matrices
        self._w_generator = w_generator
        self._win_generator = win_generator
        self._wbias_generator = wbias_generator

        # to generate the initial hidden state
        self._h0_Generator = h0_Generator
        self._h0_params = h0_params

        # data type for the whole process
        self._dtype = dtype

        # the number of time steps that are discarded in the optimization step
        self._washout = washout

    @abstractmethod
    def _hidden_state_init(self, batch_size):
        pass

    @abstractmethod
    def forward(self, input, targets=None):
        pass

    def _generate_matrices(self, w_generator, win_generator, wbias_generator, dtype, dim_in, dim_rec, dim_rec_bis):
        def branching_type_input(matrix, size_a, size_b):
            if size_b != 1:
                size = (size_a, size_b)
            else:
                size = (size_a)
            if isinstance(matrix, Matrix_generator):
                matrix_out = matrix.generate(size=size, dtype=dtype)
            elif callable(matrix):
                matrix_out = matrix(size=size, dtype=dtype)
            else:
                matrix_out = matrix
            return matrix_out

        W_ih = branching_type_input(w_generator, dim_rec_bis, dim_in)
        W_hh = branching_type_input(win_generator, dim_rec_bis, dim_rec)
        W_bias_ih = branching_type_input(wbias_generator, dim_rec_bis, 1)
        W_bias_hh = branching_type_input(wbias_generator, dim_rec_bis, 1)

        return W_ih, W_hh, W_bias_ih, W_bias_hh

    # section ######################################################################
    #  #############################################################################
    # setters/getters

    @property
    def washout(self):
        return self._washout

    @washout.setter
    def washout(self, new_washout):
        if isinstance(new_washout, int):
            self._washout = new_washout
        else:
            raise Error_type_setter(f"washout is not an {str(int)}.")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._dtype = new_dtype

    # abstract rec cell, needs to be redefined in lower level.
    @property
    @abstractmethod
    def rec_cell(self):
        return self._rec_cell

    @rec_cell.setter
    def rec_cell(self, new_rec_cell):
        self._rec_cell = new_rec_cell
