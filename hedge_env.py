import torch
import pandas as pd
import numpy as np
from market_data import X_test,X_train, X_valid, y_train, y_valid, y_test
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

window = 20
n_features = 23
a_max = 2.0
rebalance_freq = 25
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
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(window*n_features+1,),dtype=np.float32)
        self.episode_length = 252
        self.episode_end = None
        self.rebalance_freq = rebalance_freq
        self.days_held = 0
    def update_obs(self):
        self.obs = self.features[self.t-self.window+1:self.t+1].reshape(-1)
        current_pos = torch.tensor([self.prev_action],dtype=torch.float32)
        self.obs = torch.concat([self.obs,current_pos])
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
    def step(self,action):
        if self.days_held % self.rebalance_freq != 0:
            action = self.prev_action
        else:
            action = np.clip(action.item(), -self.a_max, self.a_max)
        trade = action - self.prev_action
        self.prev_action = action
        ct = self.phi*abs(trade) + 0.5*self.psi*trade**2
        reward = action*self.y[self.t] - ct - self.lam*trade*action
        terminated = self.t >= self.episode_end
        if not terminated:
            self.t += 1
            self.obs = self.update_obs()
        return self.obs.numpy(), reward * 1e4,  terminated, False, {}

"""env = HedgeEnv(window,n_features,X_train,y_train,a_max)
policy_kwargs = dict(net_arch = dict(pi=[256,256],vf=[256,256]))
model = PPO("MlpPolicy",env,device="cpu",tensorboard_log="./tensorboard/",policy_kwargs = policy_kwargs,verbose=1,learning_rate=1e-4,n_steps=4096,gae_lambda=0.95,batch_size=64)
model.learn(total_timesteps=750_000,tb_log_name="hedging")
model.save("hedging")"""
model = PPO.load("hedging")
class HedgeEnvEval(HedgeEnv):
    def reset(self, seed=None, options=None):
        super(HedgeEnv, self).reset(seed=seed)
        self.t = self.window - 1
        self.prev_action = 0.0
        self.obs = self.update_obs()
        return self.obs.numpy(), {}

    def step(self, action):
        action = np.clip(action.item(), -self.a_max, self.a_max)
        trade = action - self.prev_action
        self.prev_action = action
        ct = self.phi*abs(trade) + 0.5*self.psi*trade**2
        reward = action*self.y[self.t] - ct - self.lam*trade*action
        terminated = self.t >= len(self.y) - 1
        if not terminated:
            self.t += 1
            self.obs = self.update_obs()
        return self.obs.numpy(), reward, terminated, False, {}
test_env = HedgeEnvEval(window, n_features, X_valid, y_valid , a_max)
obs, _ = test_env.reset()
done = False
rewards, actions = [], []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    rewards.append(reward)
    actions.append(action.item())

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
y_test_realized = y_test.numpy()[:len(actions)]  # align lengths
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











