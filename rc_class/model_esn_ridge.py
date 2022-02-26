import torch

from rc_class.leaky_echo_cell import Leaky_echo_cell
from rc_class.base_model_rc_ridge import Base_model_rc_ridge


class Model_esn_ridge(Base_model_rc_ridge):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 w_generator, win_generator, wbias_generator, h0_Generator, h0_params,
                 learning_algo='inv', ridge_param=0.0, washout=0, dtype=torch.float32,
                 nonlin_fct=torch.tanh, leak_rate=0.1):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                         w_generator=w_generator, win_generator=win_generator,
                         wbias_generator=wbias_generator,
                         h0_Generator=h0_Generator, h0_params=h0_params,
                         learning_algo=learning_algo, ridge_param=ridge_param, washout=washout,
                         dtype=dtype)

        self._nonlin_fct = nonlin_fct  # activation function is the transition between states

        # Generate matrices
        # no need for two biases (mathematically).
        W_ih, W_hh, b_ih, _ = self._generate_matrices(self._w_generator, self._win_generator, self._wbias_generator,
                                                      self.dtype,
                                                      self._input_dim, self._hidden_dim, self._hidden_dim)

        self.rec_cell = Leaky_echo_cell(input_dim=self._input_dim, hidden_dim=self._hidden_dim,
                                        W=W_hh, W_in=W_ih, W_bias=b_ih,
                                        nonlin_fct=self._nonlin_fct, leak_rate=leak_rate)


    def _hidden_state_init(self, batch_size):
        h0 = self._h0_Generator(**self._h0_params).generate(size=(batch_size, self._hidden_dim), dtype=self.dtype)
        return h0

    @property
    def rec_cell(self):
        return self._rec_cell

    @rec_cell.setter
    def rec_cell(self, new_rec_cell):
        self._rec_cell = new_rec_cell
