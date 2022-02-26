# Echo State Network layer
# Basis cell for ESN.
import torch

from corai import Savable_net


######################## wip
########################  WORK IN PROGRESS
########################################################

class Leaky_echo_cell(Savable_net):
    """
    Echo State Network layer
    Basis cell for ESN
    """

    def __init__(self, input_dim, hidden_dim,
                 W, W_in, W_bias,
                 nonlin_fct=torch.tanh, leak_rate=0.1, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input dimension
        :param output_dim: Reservoir size
        :param input_scaling: Input scaling
        :param w: Internal weight matrix W
        :param w_in: Input-internal weight matrix Win
        :param w_bias: Internal units bias vector Wbias
        :param nonlin_func: Non-linear function applied to the units
        :param washout: Period to ignore in training at the beginning
        :param debug: Debug mode
        :param dtype: Data type used for vectors/matrices.
        """
        super().__init__(predict_fct=None)

        # Params
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self._nonlin_fct = nonlin_fct

        # data type for the whole process
        self.dtype = dtype
        self._leak_rate = leak_rate

        # not returned by model.parameters but are saved!
        self.register_buffer('W_in', W_in)  # Initialize input weights, as a buffer where parameters are not updated.
        self.register_buffer('W', W)  # Initialize reservoir_info weights, as a buffer where parameters are not updated.
        self.register_buffer('W_bias', W_bias)  # Initialize bias, as a buffer where parameters are not updated.

    def forward(self, input, hidden_states):
        # the notation is u is the entry,
        # h the hidden state,

        # input is shape: (batch_size, time series length, input size)
        assert input.shape[2] == self._input_dim, \
            "The shape of input must be (batch size, input_dim). " \
            "But got: {} while input dim should be {}.".format(input.shape, self._input_dim)

        batch_size = input.shape[0]
        time_length = input.shape[1]
        outputs = torch.zeros(batch_size, time_length, self._hidden_dim,
                              dtype=self.dtype)

        # operation of multiplication depending on the dimension of the input
        if input.shape[-1] > 1:
            matrix_mul = torch.matmul
        else:
            matrix_mul = torch.mul

        # casting into the right size each weight
        W_in = self.W_in.unsqueeze(0).repeat(batch_size, 1, 1)
        W = self.W.unsqueeze(0).repeat(batch_size, 1, 1)
        W_bias = self.W_bias.unsqueeze(0).repeat(batch_size, 1)

        for t in range(time_length):
            ut = input[:, t].unsqueeze(2)
            u_in = matrix_mul(W_in, ut).view(batch_size, -1)
            # the unsqueeze(2) is for the matrix multiplication.
            h_in = torch.matmul(W, hidden_states.unsqueeze(2)).squeeze(-1)

            hidden_states = (1. - self._leak_rate) * hidden_states + \
                            self._leak_rate * self._nonlin_fct(u_in + h_in + W_bias)
            outputs[:,t] = hidden_states

        return outputs, hidden_states  # hidden states for all time,
