import pandas as pd
import numpy as np
from gym import spaces, Env
from typing import List, Optional, Tuple, Dict
from AdvisorClass import Advisor, SimpleAdvisor
import timeit


def scale(df,cols,scaling):
    df[cols]/= scaling
    return df

MAX_STEPS = 1400
WINDOW_SIZE = 65
INITIAL_FORTUNE = 10000
BID_ASK_SPREAD_RANGE = (0.0065, 0.012)
TRANSACTION_COST = 0.001

delta = 0.5
# fix the seed.
np.random.seed(0)


class MultiAssetTradingEnv(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    _reward_types = ['profit','sharpe','discounted_profit','return','take_gain','log_return','step_profit','profit+step']
    _scaling_functions = {
        'tanh': np.tanh,
        'sigmoid': lambda x: 1/(1+np.exp(-x)),
        'softsign': lambda x: x/(1 + np.abs(x)),
        'arctan': np.arctan,
        'identity': lambda x: x,
    }

    def __init__(self,
        assets: List[pd.DataFrame],
        reward_type = 'profit',
        reward_scaling = 'softsign',
        discrete_actions: bool = False, 
        scale_observations: bool = True,
        advisors: Optional[List[Advisor]] = None, 
        window_size: int = WINDOW_SIZE,
        max_steps: int = MAX_STEPS,
        initial_fortune: float = INITIAL_FORTUNE,
        bid_ask_spread_range: Tuple[float, float] = BID_ASK_SPREAD_RANGE,
        transaction_cost: float = TRANSACTION_COST
        ):

        super(MultiAssetTradingEnv, self).__init__()
        """"
        Variables important for the environment: 
        initial_fortune: starting initial_fortune for each episodes 
        train_data: data_frame containing all the values 
        bid_ask_spread: difference of price from buying-selling
        transaction_cost: cost from doing a trade 
        window: allows a lookback to calculate indicators.
        """
        assert reward_type in self._reward_types, 'Reward function unknown'
        self.reward_type = reward_type
        if reward_scaling in self._scaling_functions.keys():
            self._scale_reward = self._scaling_functions[reward_scaling]
        else:
            self._scale_reward = lambda x: x
        self.scale_observations = scale_observations
        self.number_of_assets = len(assets)
        self.initial_fortune = initial_fortune
        self.old_net = initial_fortune
        self.window_size = window_size
        self.max_steps = max_steps
        self.data = assets
        self._active_data = list()
        self.bid_ask_spread_min = bid_ask_spread_range[0]
        self.bid_ask_spread_max = bid_ask_spread_range[1]
        self.transaction_cost = transaction_cost
        if advisors is None:
            advisors = [SimpleAdvisor()]
        self.advisors = advisors

        #Set action and observation space
        if discrete_actions:
            self.action_space = spaces.MultiDiscrete([3]*self.number_of_assets)
        else:
            self.action_space = spaces.Box(low=-1*np.ones(self.number_of_assets), high=np.ones(self.number_of_assets))
        self.obs_size = sum([advisor.observation_space_size(num_assets = self.number_of_assets, window_size= self.window_size) for advisor in self.advisors]) + 4 + 2*self.number_of_assets
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.obs_size,), dtype=np.float64)

        self.reset_session()


    def _reward_function(self) -> float:
        """
        Returns current reward of the environment
        """
        if (self.reward_type == 'profit'):
            return  self._scale_reward(float((self.net_worth - self.initial_fortune)))
        if (self.reward_type == 'return'):
            return (self.net_worth - self.initial_fortune)/(self.initial_fortune)
        if (self.reward_type == 'log_return'):
            return self._scale_reward(np.log(self.net_worth/self.initial_fortune))
        if (self.reward_type == 'sharpe'):
            std = np.sqrt(np.array(self.std_portfolio).mean() + delta)
            return self._scale_reward((self.net_worth - self.initial_fortune)/(self.initial_fortune*std))
        if (self.reward_type == 'discounted_profit'):
            return ((self._current_step - self.window_size)/MAX_STEPS)*self._scale_reward(float((self.net_worth - self.initial_fortune)))
        if (self.reward_type == 'take_gain'):
            return -1*self._scale_reward(np.max((self._current_step - self.window_size - MAX_STEPS)*(self.max_net_worth - self.net_worth)/self.initial_fortune,0))\
                   + self._scale_reward(float((self.net_worth - self.initial_fortune)/self.initial_fortune))
        if (self.reward_type == 'step_profit'):
            return self._scale_reward((self.net_worth - self.old_net)/self.old_net)
        if (self.reward_type == 'profit+step'):
            return .5 * self._scale_reward((self.net_worth - self.initial_fortune)/self.initial_fortune) + 0.5 * self._scale_reward((self.net_worth -self.old_net)/self.old_net)



    def _make_trading_period(self) -> None:
        """
        Sets the currently active trading episode of the environment
        """
        start_episode = np.random.randint(1 + self.window_size, len(self.data[0]) - self._steps_left)
        self._active_data = list()
        for asset in self.data:
            #data = asset[start_episode - self.window_size:start_episode + self._steps_left].reset_index(drop=True)
            #data = data.reset_index(drop=True)
            dr = asset.loc[start_episode - self.window_size:start_episode + self._steps_left].reset_index(drop=True)
            dr = dr.reset_index(drop=True)
            self._active_data.append(dr)
        #print(self._active_data)

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

        if self.net_worth <= 0:
            done = True
            obs = self.next_observation()
            reward = -1

        elif np.any(self.portfolio < -10e-3):
            done = True
            obs = self.next_observation()
            reward = -1

        elif self._steps_left == 1:
            done = True
            obs = self.reset()
            reward = self._reward_function()
        else:
            done = False
            obs = self.next_observation()
            reward = self._reward_function()

        return obs, reward, done, {}

    def reset_session(self) -> None: 
        """
        Reset the environment.
        """
        self._current_step = self.window_size
        self._steps_left = self.max_steps
        self._make_trading_period()
        self.balance = self.initial_fortune
        self.last_action = np.zeros(self.number_of_assets)
        self.portfolio = np.zeros(self.number_of_assets)
        self.net_worth = self.initial_fortune
        self.max_net_worth = self.initial_fortune
        self.fees = 0
        self.std_portfolio = []
        self.old_net = self.initial_fortune

    def reset(self):
        """
        Reset the environment, and return the next observation.
        :return:
        """
        self.reset_session()
        return self.next_observation()


    def next_observation(self):
        """
        Use the active dataset active_df to compile the state space fed to the agent.
        Scale the data to fit into the range of observation_space.
        Differentiation of data has to be implemented.

        :return: obs: observation given to the agent. Modify observation_space accordingly.
        """
        # to think about
        max_initial_volume = max([asset_data['Volume'][0] for asset_data in self.data])
        max_initial_open_price = max([asset_data['Open'][0] for asset_data in self.data])
        window_data = list()
        for asset_data in self._active_data:
            window_asset_data = asset_data[self._current_step-self.window_size:self._current_step]
            if self.scale_observations:
                window_asset_data.Volume /= max_initial_volume
                #window_asset_data.apply(lambda x: x/max_initial_volume, index=['Volume'],axis =1)
                #window_asset_data[['Open', 'Close', 'High', 'Low']] = window_asset_data[['Open', 'Close', 'High', 'Low']]/max_initial_open_price
                window_asset_data.Open /= max_initial_open_price
                window_asset_data.Close /= max_initial_open_price
                window_asset_data.High /= max_initial_open_price
                window_asset_data.Low /= max_initial_open_price
            window_data.append(window_asset_data)
        
        observed_features = []
        for advisor in self.advisors:
            advisor_features = advisor.observation(window_data)
            observed_features = np.concatenate((observed_features, advisor_features))

        if self.scale_observations:
            observed_features = np.concatenate((
                observed_features, 
                self.portfolio,
                self.last_action,
                [
                    self.balance/self.initial_fortune,
                    self._current_step/self.max_steps, 
                    self.net_worth/self.initial_fortune, 
                    (self.net_worth - self.initial_fortune)/self.initial_fortune]
            ))
        else:
            observed_features = np.concatenate((
                observed_features,
                self.portfolio,
                self.last_action, 
                [
                    self.balance,
                    self._current_step, 
                    self.net_worth, 
                    self.net_worth - self.initial_fortune
                ]
            ))
        observed_features = observed_features.reshape((self.obs_size,))

        return observed_features

    def bid_ask_spreads(self):
        """
        Returns the bid ask spread, currently a list of random values sampled uniformly at random from the
        specified bid ask spread range
        """
        return np.random.uniform(low = self.bid_ask_spread_min, high = self.bid_ask_spread_max, size = (self.number_of_assets))

    def take_action(self,action):
        """
        SEE CONSTRUCTOR FOR FULL DETAILS
        Important remark: when the agent takes action, it has not seen the closing value.
        The closing value determines the portfolio value after the action is took.
        Translate to the action made by the agent.
        It modifies the holding of btc, namely btc_holdings.
        Update the btc_holdings.
        Update the portfolio value.
        :param: action: action the agent takes. A value in [-1,1].
        :return:
        """
        self.old_net = self.net_worth
        price_open = []
        for asset in self._active_data:
            price_open.append(asset.loc[self._current_step,'Open'])
        price_open = np.array(price_open)

        price_close = []
        for asset in self._active_data:
            price_close.append(asset.loc[self._current_step,'Close'])
        price_close = np.array(price_close)
        bid_ask_spreads = self.bid_ask_spreads()
        idx_buy = []
        
        #Sell before buying
        for j in range(self.number_of_assets):
            bid_ask_spread = bid_ask_spreads[j]

            #This corresponds to a sell action.
            if action[j] < -0.25:
                current_price_ask = price_open[j] - bid_ask_spread
                size = (np.abs(action[j]) - 0.25) / 0.75
                asset_sold = max(self.portfolio[j]*size, self.balance/(self.transaction_cost-current_price_ask))
                transaction_fees = asset_sold*self.transaction_cost
                self.fees += transaction_fees
                self.balance += (asset_sold * current_price_ask -transaction_fees)
                self.portfolio[j] -= asset_sold

            elif action[j] > 0.25:
                idx_buy.append(j)



        for idx in idx_buy:
            bid_ask_spread = bid_ask_spreads[idx]

            # This corresponds to a buy action.
            current_price_bid = price_open[idx] + bid_ask_spread
            size = (action[idx] - 0.25)/0.75
            total_possible = np.max(self.balance/(current_price_bid +self.transaction_cost), 0)
            asset_bought = total_possible * size
            transaction_fees = asset_bought * self.transaction_cost
            cost = asset_bought * current_price_bid
            self.fees += transaction_fees
            self.balance -= (cost + transaction_fees)
            self.portfolio[idx] += asset_bought


        # Update book keeping. Calculate new net worth
        self.std_portfolio.append((self.net_worth - np.sum(self.portfolio * price_close))**2)

        self.net_worth = self.balance + np.sum(self.portfolio * price_close)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        self.last_action = action

        #return price_open


    def render(self, mode='human'):
        """
        Method overriding render method in gym.Env
        Stands for information of the environment to the human in front.
        Display the performance of the agent
        :param mode:
        :return:
        """
        profit = self.net_worth - self.initial_fortune
        reward = self._reward_function()
        print(f'Step: {self._current_step - self.window_size} over {self.max_steps -1}')
        print(f'Balance: {self.balance}')
        print(f'Portfolio: {self.portfolio}')
        print(f'Trading fees: {self.fees}')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        print(f'Reward: {reward}')
        return self._current_step - self.window_size, profit, self.max_net_worth

