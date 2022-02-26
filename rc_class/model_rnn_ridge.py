from collections import OrderedDict

import torch
import torch.nn as nn

from rc_class.base_model_rc_ridge import Base_model_rc_ridge


class Model_rnn_ridge(Base_model_rc_ridge):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 w_generator, win_generator, wbias_generator, h0_Generator, h0_params,
                 learning_algo='inv', ridge_param=0.0, washout=0, dtype=torch.float32):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                         w_generator=w_generator, win_generator=win_generator,
                         wbias_generator=wbias_generator,
                         h0_Generator=h0_Generator, h0_params=h0_params,
                         learning_algo=learning_algo, ridge_param=ridge_param, washout=washout,
                         dtype=dtype)  # net.nn_predict will return only the output.

        # intializes the hidden cell
        self.rec_cell = nn.RNN(input_size=self._input_dim, hidden_size=self._hidden_dim, batch_first=True,
                               nonlinearity='relu')
        for param in self.rec_cell.parameters():  # unabling the gradient descent on the Rec Cell
            param.requires_grad = False

        # Generate matrices
        # creates the hidden cell parameters
        # the gru cells needs 3 matrices for each the in_matrix, in_bias, recurrent_matrix and recurrent_bias
        # we choose to simplify the expressions as they are initialized the same way
        W_ih, W_hh, b_ih, b_hh = self._generate_matrices(self._w_generator, self._win_generator, self._wbias_generator,
                                                         self.dtype,
                                                         self._input_dim, self._hidden_dim, self._hidden_dim)

        # past parameters into the hidden cell
        new_state_dict = OrderedDict({'weight_ih_l0': W_ih, 'weight_hh_l0': W_hh,
                                      'bias_ih_l0': b_ih, 'bias_hh_l0': b_hh})
        self.rec_cell.load_state_dict(new_state_dict, strict=False)

    def _hidden_state_init(self, batch_size):
        h0 = self._h0_Generator(**self._h0_params).generate(size=(1, batch_size, self._hidden_dim), dtype=self.dtype)
        return h0

    @property
    def rec_cell(self):
        return self._rec_cell

    @rec_cell.setter
    def rec_cell(self, new_rec_cell):
        self._rec_cell = new_rec_cell
