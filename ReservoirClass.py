import abc
import pandas as pd
import numpy as np
from typing import Union, List
import torch
import torch.nn as nn
import timeit
from rc_class.model_esn_ridge import Model_esn_ridge
from rc_class.leaky_echo_cell import Leaky_echo_cell
from matrix_generator.matrix_generator import Matrix_generator

# setting the seeds.
torch.manual_seed(0)
np.random.seed(0)

class Reservoir():
    """Creates observation space from data frame of asset information"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def observation(self, df):
        raise NotImplementedError

    @abc.abstractmethod
    def observation_space_size(self, num_assets:int = 1, window_size: int = 1) -> int:
        raise NotImplementedError


class RNNAdvisor(Reservoir):

    def __init__(self, num_features: int, num_layers:int, num_assets: int = 16, nonlinearity = 'relu'):
        super(RNNAdvisor, self).__init__()
        self.num_features = num_features
        self.net = nn.RNN(
            input_size = 5*num_assets,
            hidden_size = num_features,
            num_layers = num_layers,
            nonlinearity = nonlinearity,
            batch_first = True,

        )
        self.num_layers = num_layers

    def observation_space_size(self, num_assets: int = 1, window_size: int = 1) -> int:
        return self.num_features*self.num_layers
    
    def observation(self, data: Union[pd.DataFrame, List[pd.DataFrame]]):
        if isinstance(data, pd.DataFrame):
            data = [data]
        num_assets = len(data)

        #Drop last row to exclude current values from following observations
        for i in range(len(data)):
            data[i] = data[i][:-1]

        #Concatenate different asset dataframes to form one tensor
        input = torch.concat([torch.tensor(asset_data.to_numpy()) for asset_data in data], axis = 1)
        input = torch.unsqueeze(input, 0)

        #Run network and return hidden layer
        output, hidden = self.net(input.float())
        hidden = hidden.flatten()
        return hidden.detach().numpy()


class ReservoirRNN(Reservoir):
    def __init__(self, num_features: int, num_layers: int, num_assets: int = 16, nonlinearity='relu'):
        super(ReservoirRNN, self).__init__()
        self.num_features = num_features
        self._reservoir= nn.RNN(
            input_size=6*num_assets,
            hidden_size=num_features,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )

        self.num_layers = num_layers
        self.hidden_dim = num_features


    def observation_space_size(self, num_assets: int = 1, window_size: int = 1) -> int:
        return self.num_features * self.num_layers

    def observation(self, data: Union[pd.DataFrame, List[pd.DataFrame]]):
        if isinstance(data, pd.DataFrame):
            data = [data]
        num_assets = len(data)

        #Drop last row to exclude current values from following observations
        #for i in range(len(data)):
        #    data[i] = data[i][:-1]

        #Concatenate different asset dataframes to form one tensor
        #list_tensor = [torch.tensor(df) for df in data]
        input = torch.concat([torch.tensor(asset_data.to_numpy()) for asset_data in data], axis = 1)
        input.reshape((1,input.shape[0],input.shape[1]))


        input = torch.unsqueeze(input, 0)
        h0 = torch.zeros((self.num_layers,1,self.hidden_dim))

        #Run network and return hidden layer
        hidden,out = self._reservoir(input.float(), h0.float())
        #hidden = hidden.squeeze(0)
        #hidden_keep = hidden[:][-2:]
        #hidden_keep = hidden_keep.flatten()
        out = out.flatten()
        #state = np.concatenate((out.detach().numpy(),hidden_keep.detach().numpy()))
        return out.detach().numpy()

class ReservoirLSTM(Reservoir):
    def __init__(self, num_features: int, num_layers: int, num_assets: int = 16,dropout=0.3):
        super(ReservoirLSTM, self).__init__()
        self.num_features = num_features
        self._reservoir= nn.LSTM(
            input_size=6*num_assets,
            hidden_size=num_features,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.num_layers = num_layers
        self.hidden_dim = num_features

    def observation_space_size(self, num_assets: int = 1, window_size: int = 1) -> int:
        return self.num_features * self.num_layers

    def observation(self, data: Union[pd.DataFrame, List[pd.DataFrame]]):
        if isinstance(data, pd.DataFrame):
            data = [data]
        num_assets = len(data)

        #Drop last row to exclude current values from following observations
        #for i in range(len(data)):
        #    data[i] = data[i][:-1]

        #Concatenate different asset dataframes to form one tensor
        #list_tensor = [torch.tensor(df) for df in data]
        input = torch.concat([torch.tensor(asset_data.to_numpy()) for asset_data in data], axis = 1)
        input.reshape((1,input.shape[1],input.shape[0]))


        input = torch.unsqueeze(input, 0)
        #h0 = torch.zeros((self.num_layers,1,self.hidden_dim))

        #Run network and return hidden layer
        out, (h,c) = self._reservoir(input.float())
        #hidden = hidden.squeeze(0)
        #hidden_keep = hidden[:][-2:]
        #hidden_keep = hidden_keep.flatten()
        h = h.flatten()
        #state = np.concatenate((out.detach().numpy(),hidden_keep.detach().numpy()))
        return h.detach().numpy()

class ReservoirLeakyESN(Reservoir):
    def __init__(self, num_features: int, w_generator, win_generator,
                 wbias_generator, h0_Generator, h0_params, hidden_dim = 126,
                 num_assets = 16,
                 learning_algo='inv', ridge_param=0.0, washout=0, dtype=torch.float32,
                 nonlin_fct=torch.relu, leak_rate=0.1):
        super(ReservoirLeakyESN, self).__init__()

        # dim rec bis = numfeature
        # dim rec = dim hidden
        W_ih, W_hh, b_ih, _ = self._generate_matrices(w_generator=w_generator,win_generator=win_generator,
                                                      wbias_generator= wbias_generator,
                                                      dtype=dtype,
                                                      dim_in=6*num_assets,
                                                      dim_rec=hidden_dim,
                                                      dim_rec_bis=num_features)
        self._reservoir_ = Leaky_echo_cell(input_dim=6*num_assets,hidden_dim=hidden_dim,W=W_hh, W_in=W_ih, W_bias=b_ih,
                                        nonlin_fct=nonlin_fct, leak_rate=leak_rate)
        self.h0_Generator = h0_Generator
        self._h0_params = h0_params
        self._hidden_dim = hidden_dim
        self._num_features = num_features

    def observation_space_size(self, num_assets:int = 1, window_size: int = 1) -> int:
        return self._num_features

    def observation(self, data: Union[pd.DataFrame, List[pd.DataFrame]]):
        if isinstance(data, pd.DataFrame):
            data = [data]
        num_assets = len(data)

        #Drop last row to exclude current values from following observations
        #for i in range(len(data)):
        #    data[i] = data[i][:-1]

        #Concatenate different asset dataframes to form one tensor

        input = torch.concat([torch.tensor(asset_data.to_numpy()) for asset_data in data], axis = 1)
        #input = torch.transpose(input,0,1)

        input = torch.unsqueeze(input, 0)
        #input.reshape((1,input.shape[2],input.shape[1]))

        #h0 = torch.zeros((self.num_layers,1,self.hidden_dim))

        #Run reservoir_info and return hidden layer
        h0 = self._hidden_state_init()
        _,h = self._reservoir_(input.float(),h0)
        h = h.flatten()
        return h.detach().numpy()

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

    def _hidden_state_init(self):
        h0 = self.h0_Generator(**self._h0_params).generate(size=(1, self._hidden_dim), dtype=torch.float32)
        return h0


class StateReservoirLeakyESN(Reservoir):
    def __init__(self, window_size: int,num_features: int, w_generator, win_generator,
                 wbias_generator, h0_Generator, h0_params, hidden_dim = 126,
                 num_assets = 16,
                 learning_algo='inv', ridge_param=0.0, washout=0, dtype=torch.float32,
                 nonlin_fct=torch.relu, leak_rate=0.1):
        super(StateReservoirLeakyESN, self).__init__()

        # dim rec bis = numfeature
        # dim rec = dim hidden
        # (self.window_size-1)*self.number_of_assets + 3 + 4*(self.number_of_assets)
        W_ih, W_hh, b_ih, _ = self._generate_matrices(w_generator=w_generator,win_generator=win_generator,
                                                      wbias_generator= wbias_generator,
                                                      dtype=dtype,
                                                      dim_in=(window_size+3)*num_assets+3,
                                                      dim_rec=hidden_dim,
                                                      dim_rec_bis=num_features)
        self._reservoir_ = Leaky_echo_cell(input_dim=(window_size+3)*num_assets+3,hidden_dim=hidden_dim,W=W_hh, W_in=W_ih, W_bias=b_ih,
                                        nonlin_fct=nonlin_fct, leak_rate=leak_rate)
        self.h0_Generator = h0_Generator
        self._h0_params = h0_params
        self._hidden_dim = hidden_dim
        self._num_features = num_features

    def observation_space_size(self, num_assets:int = 1, window_size: int = 1) -> int:
        return self._num_features

    def observation(self, state: np.array):

        input = torch.tensor(state)
        input = torch.unsqueeze(input, 0)

        #Run reservoir_info and return hidden layer
        h0 = self._hidden_state_init()
        _,h = self._reservoir_(input.float(),h0)
        h = h.flatten()
        return h.detach().numpy()

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

    def _hidden_state_init(self):
        h0 = self.h0_Generator(**self._h0_params).generate(size=(1, self._hidden_dim), dtype=torch.float32)
        return h0










