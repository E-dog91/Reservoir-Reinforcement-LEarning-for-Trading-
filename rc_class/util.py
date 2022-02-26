import torch


def spectral_radius(m):
    """
    Compute spectral radius of a square 2-D tensor
    :param m: squared 2D tensor
    :return:
    """
    return torch.max(torch.abs(torch.linalg.eigvals(m)[0])).item()


def r2_score(y_true, y_pred):
    return 1 - (torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - y_true.mean()) ** 2))


def nrmse(y_true, y_pred):
    return torch.sqrt((torch.sum(y_true - y_pred) ** 2) / len(y_true)) / (y_true.max() - y_true.min())


def predict_function_slice_tuple(tpl):
    # for the models returning output, hidden_states, but we only need the ouput.
    return tpl[0]  # add a dimension, the time length that we removed at the entrance.


def flatten_batches_timeseries4scaling_back_normal(scaler, data, flagfit):
    # data form (N,L, D_in) or (L,N,D_in). Then, returns in the same format the data.
    if flagfit:
        fct = scaler.fit_transform
    else:
        fct = scaler.transform
    return torch.FloatTensor(fct(data.flatten(0, 1))).reshape(data.shape)
