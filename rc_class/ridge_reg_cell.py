import torch

from corai import Savable_net


######################## wip
########################  WORK IN PROGRESS
########################################################
# Ridge Regression node


class Ridge_reg_cell(Savable_net):
    """
    Once the reservoir_info has been projecting the data, we put the parameters into this cell. This is the out-matrix.
    Then, we call fit, which trains the parameters of the out-matrix to be mapped towards the targets.
    """

    def __init__(self, input_dim, output_dim,
                 ridge_param=0.0, learning_algo='inv', param=None):
        """
        Constructor
        :param input_dim: Feature space dimension
        :param output_dim: Output space dimension
        :param ridge_param: Ridge parameter
        :param with_bias: Add a bias to the linear layer
        :param learning_algo: Inverse (inv) or pseudo-inverse (pinv)
        :param softmax_output: Add a softmax output (normalize outputs) ?
        :param normalize_output: Normalize outputs to sum to one ?
        :param averaged: Covariance matrix divided by the number of samples ?
        :param debug: Debug mode
        :param test_case: Test case to call for test.
        :param dtype: Data type
        """
        super().__init__(predict_fct=None)

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._ridge_param = ridge_param  # todo rename for ridge_reg
        self._learning_algo = learning_algo

        w, b = _detect_parameters(param)

        self.weights = torch.nn.parameter.Parameter(w)  # savable
        self.bias = torch.nn.parameter.Parameter(b)  # savable

    # Forward
    def forward(self, input, targets=None):
        """Updates the parameters with the solution of the optimization problem.

        Args:
            input: input tensor of shape (batch, n_features)
            expected: target tensor of shape:
                - (batch, n_targets) in case of regression
                - (batch,) in case of classification, with dtype=torch.Long

        """
        batch_size = input.shape[0]
        length_ts = input.shape[1]

        #if self.training:
        #    assert targets is not None, "Targets are None, but we are in training mode."
        #    assert len(input.shape) == len(targets.shape), "Inputs and targets size mismatch."
        #    w, b = _direct_ridge(input, targets, self._ridge_param)
        #    self.weights.data = w
        #    self.bias.data = b

        bias = self.bias.data.unsqueeze(0).repeat(batch_size, length_ts, 1)
        weights = self.weights.data.unsqueeze(0).unsqueeze(0).repeat(batch_size, length_ts, 1, 1)
        output = (bias + (weights @ input.unsqueeze(-1)).squeeze(-1)).transpose(0, 1)
        return output

    def _get_params(self):
        params = {}
        params['w'] = self.weights
        params['b'] = self.bias
        return params

    def _set_params(self, params):
        self.weights = params['w']
        self.bias = params['b']
        return params


@torch.jit.script
def _incremental_ridge_end(mat_a, mat_b, l2_reg: float, ide):
    # Compute A @ (B + l2_reg * I)^{-1}
    weights = torch.linalg.solve(mat_b + l2_reg * ide, mat_a.t()).t()  # (ny, nr+1)
    w, b = weights[:, :-1], weights[:, -1]
    return w, b


@torch.jit.script
def _direct_ridge(input, expected, l2_reg: float):
    """

    Args:
        input_size:
        output_size:
        input:
        expected:
        l2_reg:
        classification: True to perform classification, False to perform regression

    Returns:

    """
    ide = torch.eye(input.shape[-1] + 1, device=input.device)

    input_dim = input.shape[2]
    output_dim = expected.shape[2]
    # Add bias
    s = torch.cat([input.reshape(-1, input_dim),
                   torch.ones(input.shape[0] * input.shape[1], 1, device=input.device, dtype=input.dtype)], dim=1)

    y = expected.reshape(-1, output_dim)

    # s: (nb, nr+1)
    # y: (nb, ny)
    mat_a = torch.einsum('br,by->yr', s, y)
    mat_b = torch.einsum('br,bz->rz', s, s)

    return _incremental_ridge_end(mat_a, mat_b, l2_reg, ide)


def _detect_parameters(params):
    if isinstance(params, torch.Tensor):
        raise TypeError("params argument given to the optimizer should be "
                        "an iterable of Tensors, but got " +
                        torch.typename(params))

    plist = list(params)
    if len(plist) == 0:
        raise ValueError("optimizer got an empty parameter list")

    if len(plist) != 2:
        raise ValueError(f"optimizer expected a list of 2 parameters, but got {len(plist)}")

    # Find which one is the weight and which one is the bias
    if len(plist[0].shape) > len(plist[1].shape):
        w, b = plist[0], plist[1]
    else:
        w, b = plist[1], plist[0]

    if w.shape[0] != b.shape[0]:
        raise ValueError(f"the first dimension of the tensors should match, but got {tuple(w.shape[0])} "
                         f"and {tuple(b.shape[0])}")

    return w, b
