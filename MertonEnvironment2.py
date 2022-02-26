import pandas as pd
import numpy as np
from gym import spaces, Env
from typing import List, Optional, Tuple, Dict
from ReservoirClass import *

# TODO: make the bid-asks more modular
# TODO: make the scale coherent and modular
# TODO: implement cost function
# TODO: calculate some statistics along
# TODO: scale the observations

pd.options.mode.chained_assignment = None  # default='warn'

### GLOBAL VARIABLES ###
MAX_ACCOUNT_BALANCE = 100000
MAX_VOLUME = 500
MAX_PRICE = 80000
VOL_SCALE = 50
BID_ASK_SPREAD_RANGE = (0.0065, 0.012)
INITIAL_FORTUNE = 10000
MAX_STEPS = 1000
MAX_NET_WORTH = 30000
MAX_HOLDING = 50

# fix the seed.
np.random.seed(0)


class MultiAssetTradingEnv(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    _scaling_functions = {
        'tanh': np.tanh,
        'sigmoid': lambda x: 1/(1+np.exp(-x)),
        'softsign': lambda x: x/(1 + np.abs(x)),
        'arctan': np.arctan,
        'identity': lambda x: x,
    }
    _reservoir_types = ['rnn','lstm','gru','esn_infos','esn_states']
    def __init__(self,
                 assets: List[pd.DataFrame],
                 delta = np.float64,
                 window_size: int = 25,
                 max_steps: int = 1000,
                 initial_fortune: float = 10000,
                 bid_ask_spread_range: Tuple[float, float] = BID_ASK_SPREAD_RANGE,
                 transaction_cost = 0.001,
                 ):
        super(MultiAssetTradingEnv, self).__init__()
        assert (delta >= 0 and delta <= 1), 'Impossible to construct utility'
        self.number_of_assets = len(assets)
        self.delta = delta
        self.initial_fortune = initial_fortune
        self.window_size = window_size
        self.max_steps = max_steps
        self.data = assets
        self._active_data = list()
        self.bid_ask_spread_min = bid_ask_spread_range[0]
        self.bid_ask_spread_max = bid_ask_spread_range[1]
        self.transaction_cost = transaction_cost
        self.action_space = spaces.Box(low=  -np.inf*np.ones(self.number_of_assets), high=np.inf*np.ones(self.number_of_assets))
        self.obs_size = 4*self.number_of_assets + 3 + (self.window_size-1)*6*self.number_of_assets


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64)

        self.scale_price,self.scale_volume = self.get_scale_values()

    def get_scale_values(self):
        scale_price = 0
        scale_volume = 0
        for asset in self.data:
            max_price = np.max(asset['High'])
            max_volume = np.max(asset['Volume'])
            scale_price = max(scale_price,max_price)
            scale_volume = max(scale_volume,max_volume)
        return scale_price,scale_volume

    def scale_reward(self,x):
        return np.tanh(x)

    def utility(self,x):
        if (self.delta ==0):
            if x < 0:
                return -self.initial_fortune
            else:
                return np.log(x)
        else:
            if x < 0:
                return -self.initial_fortune
            else:
                return (np.power(x,self.delta) - 1)/self.delta

    def _make_trading_period(self) -> None:
        """
        Sets the currently active trading episode of the environment
        """
        start_episode = np.random.randint(1 + self.window_size, len(self.data[0]) - self._steps_left)
        self._active_data = list()
        for asset in self.data:
            #data = asset[start_episode - self.window_size:start_episode + self._steps_left].reset_index(drop=True)
            #data = data.reset_index(drop=True)
            dr = asset.loc[start_episode - self.window_size:start_episode + self._steps_left -1].reset_index(drop=True)
            self._active_data.append(dr)

    def reset(self):
        """
        Reset the environment, and return the next observation.
        :return: next_observation
        """
        self._current_step = self.window_size
        self._steps_left = self.max_steps
        self._make_trading_period()
        self.balance = self.initial_fortune
        self.portfolio = np.zeros(self.number_of_assets)
        self.net_worth = self.initial_fortune
        self.max_net_worth = self.initial_fortune
        self.fees = 0
        self.std_portfolio = []

        return self.next_observation()

    def next_observation(self):
        """
        Use the active dataset active_df to compile the state space fed to the agent.
        Scale the data to fit into the range of observation_space.
        Differentiation of data has to be implemented.
        :return: obs: observation given to the agent. Modify observation_space accordingly.
        """
        #Calculate open prices.
        open_prices_t = []
        for asset in self._active_data:
            open_prices_t.append(asset['Open'][self._current_step])

        open_prices_t = np.array(open_prices_t)/self.scale_price

        # TODO: create window_data with a reservoir_data function
        return_scaled = []
        mean_price = []
        mean_volume = []
        window_data = list()
        for asset_data in self._active_data:
            window_asset_data = asset_data[self._current_step - self.window_size:self._current_step-1]
            window_asset_data.Open /= self.scale_price
            window_asset_data.Close /= self.scale_price
            window_asset_data.High /= self.scale_price
            window_asset_data.Low /= self.scale_price
            window_asset_data.Volume /= self.scale_volume
            return_window = asset_data['Close'][self._current_step - self.window_size:self._current_step - 1]\
                            - asset_data['Open'][self._current_step - self.window_size:self._current_step - 1]
            difference_window = asset_data['High'][self._current_step - self.window_size:self._current_step - 1]\
                                - asset_data['Low'][self._current_step - self.window_size:self._current_step - 1]
            dt = return_window / (difference_window + 1)
            window_asset_data['Return'] = dt
            mean_price.append(np.array(asset_data['Close'][self._current_step - self.window_size:self._current_step-1].values).mean())
            mean_volume.append(np.array(asset_data['Volume'][self._current_step - self.window_size:self._current_step-1].values).mean())
            return_scaled.append(dt)
            window_data.append(window_asset_data)


        #data_pos = np.array((self.balance,self.net_worth))
        obs = np.concatenate((open_prices_t,mean_price,mean_volume,
                              self.utility(self.net_worth),
                              self.portfolio,window_data,
                              (self._current_step - self.window_size)/self.max_steps,
                              self.net_worth)
                             ,axis=None)
        return obs

    def bid_ask_spreads(self):
        """
        Returns the bid ask spread, currently a list of random values sampled uniformly at random from the
        specified bid ask spread range
        """
        return np.random.uniform(low = self.bid_ask_spread_min, high = self.bid_ask_spread_max, size = (self.number_of_assets))

    def take_action(self, action):
        """
        :param: action: action the agent takes. A value in [-1,1]^{num_assets}.
        :return: None
        """
        price_open = []
        for asset in self._active_data:
            price_open.append(asset.loc[self._current_step,'Open'])
        price_open = np.array(price_open)

        price_close = []
        for asset in self._active_data:
            price_close.append(asset.loc[self._current_step, 'Close'])
        price_close = np.array(price_close)

        #creating bid ask-spreads vector
        bid_ask_spreads = self.bid_ask_spreads()

        assets_sold = np.sum((price_open - bid_ask_spreads)[np.where(self.portfolio >= action)]*(action-self.portfolio)[np.where(self.portfolio >= action)])
        asset_bought = np.sum((price_open+ bid_ask_spreads)[np.where(self.portfolio <= action)]*(action -self.portfolio)[np.where(self.portfolio <= action)])
        cost = np.sum(np.abs(self.portfolio - action)*self.transaction_cost)
        self.fees -= cost
        self.balance = self.balance +  asset_bought + assets_sold - cost
        self.portfolio = action
        # Update book keeping. Calculate new net worth
        self.std_portfolio.append((self.net_worth - np.sum(self.portfolio * price_close)) ** 2)

        self.net_worth = self.net_worth + np.sum(self.portfolio * (price_close - price_open))
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth


    def step(self, action):
        """
        Define the evolution of the environment after the action:action has been taken by the agent.
        This is the action took at the begining of the session.
        Override from the gym.Env class' method.
        :param action:
        :return: obs,reward,done, {}
        """
        self._current_step +=1
        self._steps_left -=1
        self.take_action(action)

        if self._steps_left == 1:
            done = True
            obs = self.reset()
            reward = self.utility(self.net_worth)
        else:
            done = False
            obs = self.next_observation()
            reward = 0

        return obs, reward, done, {}

    def render(self, mode='human'):
        """
        Method overriding render method in gym.Env
        Stands for information of the environment to the human in front.
        Display the performance of the agent
        :param mode:
        :return:
        """
        current_profit = self.net_worth - self.initial_fortune
        print(f'Step: {1 + self._current_step - self.window_size} over {self.max_steps}')
        print(f'Balance: {self.balance}')
        print(f'Portfolio: {self.portfolio}')
        print(f'Trading fees: {self.fees}')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {current_profit}')

        return 1 + self._current_step - self.window_size, current_profit




