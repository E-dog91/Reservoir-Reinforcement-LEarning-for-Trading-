import random
import json
import sys
import gym
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MertonEnvironment import MultiAssetTradingEnv
from stable_baselines3 import PPO,DDPG,SAC,TD3,A2C
import torch
from matrix_generator import matrix_gauss_gen
pd.options.mode.chained_assignment = None  # default='warn'

np.random.seed(0)
##### LOAD BTC #####
btc_data = pd.read_csv('DATA/Binance1hr/BTCUSDT_1h.csv')
btc_data = btc_data.sort_values('date')
btc_test = btc_data[-700:]
btc_train = btc_data[-4600:-700]

btc_train = btc_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
btc_test = btc_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

btc_train = btc_train.reset_index(drop=True)
btc_test = btc_test.reset_index(drop=True)

btc_train.rename(columns={"open": "Open", "low": "Low","Volume BTC": "Volume", "high": "High", "close":"Close"},inplace=True)
btc_test.rename(columns={"open": "Open", "low": "Low","Volume BTC": "Volume", "high": "High", "close":"Close"},inplace=True)

##################
bnb_data = pd.read_csv('DATA/Binance1hr/BNBUSDT_1h.csv')
bnb_data = bnb_data.sort_values('date')
bnb_test = bnb_data[-700:]
bnb_train = bnb_data[-4600:-700]

bnb_train = bnb_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
bnb_test = bnb_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

bnb_train = bnb_train.reset_index(drop=True)
bnb_test = bnb_test.reset_index(drop=True)

bnb_train.rename(columns={"open": "Open", "low": "Low","Volume BNB": "Volume", "high": "High","close":"Close"},inplace=True)
bnb_test.rename(columns={"open":"Open", "low":"Low","Volume BNB":"Volume","high":"High","close":"Close"},inplace=True)

#################

eth_data = pd.read_csv('DATA/Binance1hr/ETHUSDT_1h.csv')
eth_data = eth_data.sort_values('date')
eth_test = eth_data[-700:]
eth_train = eth_data[-4600:-700]

eth_train = eth_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
eth_test = eth_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

eth_train = eth_train.reset_index(drop=True)
eth_test = eth_test.reset_index(drop=True)

eth_train.rename(columns={"open": "Open", "low": "Low","Volume ETH": "Volume", "high": "High", "close":"Close"},inplace=True)
eth_test.rename(columns={"open": "Open", "low": "Low","Volume ETH": "Volume", "high": "High", "close":"Close"},inplace=True)

################

ltc_data = pd.read_csv('DATA/Binance1hr/LTCUSDT_1h.csv')
ltc_data = ltc_data.sort_values('date')
ltc_test = ltc_data[-700:]
ltc_train = ltc_data[-4600:-700]

ltc_train = ltc_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
ltc_test = ltc_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

ltc_train = ltc_train.reset_index(drop=True)
ltc_test = ltc_test.reset_index(drop=True)

ltc_train.rename(columns={"open": "Open", "low": "Low","Volume LTC": "Volume", "high": "High", "close":"Close"},inplace=True)
ltc_test.rename(columns={"open": "Open", "low": "Low","Volume LTC": "Volume", "high": "High", "close":"Close"},inplace=True)

###############
sol_data = pd.read_csv('DATA/Binance1hr/SOLUSDT_1h.csv')
sol_data = sol_data.sort_values('date')
sol_test = sol_data[-700:]
sol_train = sol_data[-4600:-700]

sol_train = sol_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
sol_test = sol_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

sol_train = sol_train.reset_index(drop=True)
sol_test = sol_test.reset_index(drop=True)

sol_train.rename(columns={"open": "Open", "low": "Low","Volume SOL": "Volume", "high": "High", "close":"Close"},inplace=True)
sol_test.rename(columns={"open": "Open", "low": "Low","Volume SOL": "Volume", "high": "High", "close":"Close"},inplace=True)

#################

zec_data = pd.read_csv('DATA/Binance1hr/ZECUSDT_1h.csv')
zec_data = zec_data.sort_values('date')
zec_test = zec_data[-700:]
zec_train = zec_data[-4600:-700]

zec_train = zec_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
zec_test = zec_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

zec_train = zec_train.reset_index(drop=True)
zec_test = zec_test.reset_index(drop=True)

zec_train.rename(columns={"open": "Open", "low": "Low","Volume ZEC": "Volume", "high": "High", "close":"Close"},inplace=True)
zec_test.rename(columns={"open": "Open", "low": "Low","Volume ZEC": "Volume", "high": "High", "close":"Close"},inplace=True)

################

etc_data = pd.read_csv('DATA/Binance1hr/ETCUSDT_1h.csv')
etc_data = etc_data.sort_values('date')
etc_test = etc_data[-700:]
etc_train = etc_data[-4600:-700]

etc_train = etc_train.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})
etc_test = etc_test.drop(columns={'unix','date','symbol','tradecount','Volume USDT'})

etc_train = etc_train.reset_index(drop=True)
etc_test = etc_test.reset_index(drop=True)

etc_train.rename(columns={"open": "Open", "low": "Low","Volume ETC": "Volume", "high": "High", "close":"Close"},inplace=True)
etc_test.rename(columns={"open": "Open", "low": "Low","Volume ETC": "Volume", "high": "High", "close":"Close"},inplace=True)

#############

# train set
train_data = [bnb_train, btc_train, eth_train,ltc_train,etc_train,sol_train,zec_train]

#test set
test_data = [bnb_test, btc_test,eth_test,ltc_test,etc_test,sol_test,zec_test]

timesteps = 350000
delta = 1
reservoir_type_infos='esn_infos'
reservoir_type_states = 'esn_states'
### num_feature = hidden_dim
num_features_reservoir = 512
window_size = 50
leak_rate = 0.15
max_steps = 500
initial_fortune = 10000
transaction_cost=0.01
dtype = torch.float32
washout = 0
W = matrix_gauss_gen.Matrix_gauss_gen(scale=0.95, sparse=0.95)
h0_params = {'scale': 0.8, 'sparse': 0.35}



env = MultiAssetTradingEnv(assets=train_data, delta=delta,reservoir_type_info=reservoir_type_infos,reservoir_type_state= reservoir_type_states,
                           window_size= window_size,
                           max_steps=max_steps,initial_fortune=initial_fortune,
                           num_features_reservoir_states = num_features_reservoir,num_features_reservoir_info=num_features_reservoir,
                           activation_fun='tanh',
                           transaction_cost=transaction_cost,win_generator=W,
                           w_bias_generator=W,w_generator=W,h0_params=h0_params,
                           leak_rate_info=leak_rate,leak_rate_states=leak_rate,h0_Generator=matrix_gauss_gen.Matrix_gauss_gen)
#md = A2C('MlpPolicy', env=env,verbose=1)
#md.learn(total_timesteps=timesteps)
#md.save('A2CSUN')
md = A2C.load('A2CSUN.zip')

test_env = MultiAssetTradingEnv(assets=test_data, delta=delta,reservoir_type_info=reservoir_type_infos,reservoir_type_state= reservoir_type_states,
                           window_size= window_size,
                           max_steps=max_steps,initial_fortune=initial_fortune,
                           num_features_reservoir_states = num_features_reservoir,num_features_reservoir_info=num_features_reservoir,
                           activation_fun='tanh',
                           transaction_cost=transaction_cost,win_generator=W,
                           w_bias_generator=W,w_generator=W,h0_params=h0_params,
                           leak_rate_info=leak_rate,leak_rate_states=leak_rate,h0_Generator=matrix_gauss_gen.Matrix_gauss_gen)

obs = test_env.reset()
d = True
steps = []
profit_ppo = []
while d:
    action, _states = md.predict(obs)
    obs, rewards, dones, info = test_env.step(action)
    s,p = test_env.render()
    steps.append(s)
    profit_ppo.append(p)
    d = not dones
del obs



plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.get_cmap('Pastel1').colors)
plt.plot(range(len(profit_ppo[:-1])),np.array(profit_ppo[:-1]), label='PPO')
plt.xlabel("Number of steps")
plt.ylabel("Profit")
plt.title("Performance on an new episode for the A2C agent")
plt.legend()
plt.savefig('a2c.png',dpi=300)
plt.show()

print(f'Return A2C: {100*profit_ppo[-2]/initial_fortune} %')
print(f'Mean return:{100* np.array(profit_ppo[:-2]).mean()/initial_fortune}, standard deviation: {np.std(np.array(profit_ppo[:-2])/initial_fortune)}')

