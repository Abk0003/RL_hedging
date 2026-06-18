import torch
import pandas as pd
import numpy as np
from market_data import X_test,X_train, X_valid, y_train, y_valid, y_test
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt

window = 20
n_features = 23
a_max = 2.0
rebalance_freq = 5
class HedgeEnv(gym.Env):
    def __init__(self,window,n_features,features,y,a_max,lam=0,psi=1e-4,phi=0.0002):
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.obs = torch.zeros((self.window,self.n_features),dtype=torch.float)
        self.t = window - 1
        self.prev_action = 0
        self.features = features
        self.y = y
        self.lam = lam
        self.psi = psi
        self.phi = phi
        self.a_max = a_max
        self.action_space = spaces.Box(low=-a_max,high=a_max,shape=(1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(window,n_features+1),dtype=np.float32)
        self.episode_length = 252
        self.episode_end = None
        self.rebalance_freq = rebalance_freq
        self.days_held = 0
    def update_obs(self):
        self.obs = self.features[self.t-self.window+1:self.t+1].clone()
        act = torch.full((self.window,1),self.prev_action,dtype=torch.float32)
        self.obs = torch.concat([self.obs,act],dim=1)
        return self.obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = 0.0
        self.days_held = 0
        max_start = len(self.y) - self.episode_length
        self.t = np.random.randint(self.window, max_start)

        self.episode_end = self.t + self.episode_length
        self.obs = self.update_obs()

        return self.obs.numpy(), {}
    def step(self, action):
        action = np.clip(action.item(), -self.a_max, self.a_max)
        trade = action - self.prev_action
        ct = self.phi*abs(trade) + 0.5*self.psi*trade**2
        lev = 0.00005*action**2
        reward = - ct - self.lam*trade*action-lev
        for _ in range(self.rebalance_freq):
            reward += action * self.y[self.t].item()
            self.t += 1
            if self.t >= self.episode_end:
                break
        self.prev_action = action
        terminated = self.t >= len(self.y) - 1
        self.obs = self.update_obs()
        return self.obs.numpy(), reward * 1e4, terminated, False, {}

"""env = HedgeEnv(window,n_features,X_train,y_train,a_max)
policy_kwargs = dict(lstm_hidden_size=64,n_lstm_layers=1,net_arch=dict(vf=[64],pf=[64]))
model = RecurrentPPO("MlpLstmPolicy",env,device="cpu",tensorboard_log="./tensorboard/",policy_kwargs = policy_kwargs,verbose=1,learning_rate=1e-4,n_steps=4096,gae_lambda=0.95,batch_size=64,ent_coef=0.01)
model.learn(total_timesteps=75_000,tb_log_name="lstm_hedging")
model.save("lstm_hedging")"""
model = RecurrentPPO.load("lstm_hedging")
class HedgeEnvEval(HedgeEnv):
    def reset(self, seed=None, options=None):
        super(HedgeEnv, self).reset(seed=seed)
        self.t = self.window - 1
        self.prev_action = 0.0
        self.obs = self.update_obs()
        return self.obs, {}

    def step(self, action):
        action = np.clip(action.item(), -self.a_max, self.a_max)
        trade = action - self.prev_action
        ct = self.phi * abs(trade) + 0.5*self.psi*trade**2 + self.lam * action * trade
        period_rew = []
        pen_paid = False
        for _ in range(self.rebalance_freq):
            if self.t >= len(self.y) - 1:
                break
            day_ct = ct if not pen_paid else 0.0
            day_lam = (self.lam * trade * action) if not pen_paid else 0.0
            pen_paid = True
            day_reward = (action * self.y[self.t].item()) - day_ct - day_lam
            period_rew.append(day_reward)
            self.t += 1
        self.prev_action = action
        obs = self.update_obs()
        terminated = self.t >= len(self.y) - 1
        return obs, period_rew, terminated, False, {}

test_env = HedgeEnvEval(window, n_features, X_valid, y_valid, a_max)
obs, _ = test_env.reset()
done = False
rewards, actions = [], []
lstm_states = None
while not done:
    action, lstm_states = model.predict(obs,state=lstm_states, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    rewards.extend(reward)
    actions.extend([test_env.prev_action]*len(reward))

r = np.array(rewards)
sharpe = np.sqrt(252) * r.mean() / (r.std() + 1e-8)
downside = r[r < 0]
sortino = np.sqrt(252) * r.mean() / (downside.std() + 1e-8)
equity = np.cumprod(1 + r)
peak = np.maximum.accumulate(equity)
drawdown = (equity - peak) / peak
max_drawdown = drawdown.min()
years = len(r) / 252
cagr = equity[-1] ** (1 / years) - 1
actions = np.array(actions)
turnover = np.mean(np.abs(np.diff(actions)))
print(f"Sharpe ratio: {sharpe} ; Sortino : {sortino} ; Maximum Drawdown: {max_drawdown} ; CAGR: {cagr} ; Turnover: {turnover}")
# Buy and hold equivalent
bh_returns = y_test.numpy()  # raw forward returns, unhedged
bh_sharpe = np.sqrt(252) * bh_returns.mean() / (bh_returns.std() + 1e-8)
print("Buy & hold Sharpe (test):", bh_sharpe)

plt.figure(figsize=(12,4))
plt.plot(actions)
plt.title("Action / hedge ratio over test period")
plt.savefig("action_hedge.png")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(np.cumprod(1+r))
plt.title("Equity curve, test period")
plt.savefig("equity.png")
plt.show()

actions = np.array(actions)
y_test_realized = y_valid.numpy()[window - 1: window - 1 + len(actions)]
corr = np.corrcoef(actions, y_test_realized)[0,1]
print("Correlation(action, forward return):", corr)

import matplotlib.pyplot as plt

plt.scatter(actions, y_test_realized, alpha=0.4, s=10)
plt.xlabel("action")
plt.ylabel("forward return")
plt.title("Action vs forward return, test period")
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.savefig("action_vs_forward_return.png")
plt.show()

from scipy.stats import spearmanr
ic_spearman, _ = spearmanr(actions, y_test_realized)
print("Spearman IC:", ic_spearman)

gross_pnl = actions * y_test_realized
print("Gross Sharpe (no costs):", np.sqrt(252) * gross_pnl.mean() / (gross_pnl.std() + 1e-8))











