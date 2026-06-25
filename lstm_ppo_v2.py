import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt

from market_data import X_test, X_train, X_valid, y_train, y_valid, y_test


window         = 20
n_features     = 23
a_max          = 2.0
rebalance_freq = 5

CVAR_ALPHA          = 0.10
CVAR_WEIGHT         = 0.05
DD_COST_MIN         = 1.0
DD_COST_MAX         = 2.0
DD_COST_SENSITIVITY = 5.0


class HedgeEnv(gym.Env):
    def __init__(self, window, n_features, features, y, a_max,
                 lam=0, psi=1e-4, phi=0.0002):
        super().__init__()
        self.window      = window
        self.n_features  = n_features
        self.features    = features
        self.y           = y
        self.a_max       = a_max
        self.lam         = lam
        self.psi         = psi
        self.phi         = phi


        obs_dim = n_features + 2
        self.action_space      = spaces.Box(low=-a_max, high=a_max,
                                            shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        self.episode_length = 252
        self.episode_end    = None
        self.rebalance_freq = rebalance_freq

        # running state — reset every episode
        self.t             = window - 1
        self.prev_action   = 0.0
        self.cum_reward    = 0.0
        self.peak_reward   = 0.0
        self.drawdown      = 0.0
        self.reward_buffer = []


    def _update_drawdown(self, step_reward: float):
        self.cum_reward  += step_reward
        self.peak_reward  = max(self.peak_reward, self.cum_reward)
        denom             = abs(self.peak_reward) + 1e-8
        self.drawdown     = (self.cum_reward - self.peak_reward) / denom

    def _cost_multiplier(self) -> float:
        raw = DD_COST_MIN + DD_COST_SENSITIVITY * abs(self.drawdown)
        return float(np.clip(raw, DD_COST_MIN, DD_COST_MAX))

    def _cvar_penalty(self) -> float:
        min_samples = max(10, int(1 / CVAR_ALPHA))
        if len(self.reward_buffer) < min_samples:
            return 0.0
        arr     = np.array(self.reward_buffer)
        cutoff  = max(1, int(np.floor(CVAR_ALPHA * len(arr))))
        tail    = np.sort(arr)[:cutoff]
        return float(abs(tail.mean()))

    def update_obs(self) -> np.ndarray:
        feat    = self.features[self.t].numpy()
        extras  = np.array([self.prev_action, self.drawdown], dtype=np.float32)
        return np.concatenate([feat, extras])


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action   = 0.0
        self.cum_reward    = 0.0
        self.peak_reward   = 0.0
        self.drawdown      = 0.0
        self.reward_buffer = []

        max_start        = len(self.y) - self.episode_length
        self.t           = np.random.randint(self.window, max_start)
        self.episode_end = self.t + self.episode_length

        return self.update_obs(), {}

    def step(self, action):
        action  = float(np.clip(action.item(), -self.a_max, self.a_max))
        trade   = action - self.prev_action

        # asymmetric transaction cost
        base_ct = self.phi * abs(trade) + 0.5 * self.psi * trade ** 2
        ct      = self._cost_multiplier() * base_ct

        reward  = -ct - self.lam * trade * action
        for _ in range(self.rebalance_freq):
            reward += action * self.y[self.t].item()
            self.t += 1
            if self.t >= self.episode_end:
                break

        self._update_drawdown(reward)

        # CVaR penalty
        self.reward_buffer.append(reward)
        reward -= CVAR_WEIGHT * self._cvar_penalty()

        self.prev_action = action
        terminated = self.t >= len(self.y) - 1
        return self.update_obs(), reward * 1e4, terminated, False, {}


class HedgeEnvEval(HedgeEnv):

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.t             = self.window - 1
        self.prev_action   = 0.0
        self.cum_reward    = 0.0
        self.peak_reward   = 0.0
        self.drawdown      = 0.0
        self.reward_buffer = []
        return self.update_obs(), {}

    def step(self, action):
        action  = float(np.clip(action.item(), -self.a_max, self.a_max))
        trade   = action - self.prev_action

        base_ct  = self.phi * abs(trade) + 0.5 * self.psi * trade ** 2
        ct       = self._cost_multiplier() * base_ct

        period_rew = []
        pen_paid   = False
        for _ in range(self.rebalance_freq):
            if self.t >= len(self.y) - 1:
                break
            day_ct   = ct if not pen_paid else 0.0
            pen_paid = True
            day_rew  = (action * self.y[self.t].item()) - day_ct
            period_rew.append(day_rew)
            self.t  += 1

        if period_rew:
            self._update_drawdown(sum(period_rew))
            self.reward_buffer.extend(period_rew)

        self.prev_action = action
        terminated = self.t >= len(self.y) - 1
        return self.update_obs(), period_rew, terminated, False, {}



env = HedgeEnv(window, n_features, X_train, y_train, a_max)

policy_kwargs = dict(
    lstm_hidden_size = 128,
    n_lstm_layers    = 1,
    net_arch         = dict(pi=[128, 128], vf=[128, 128]),
)

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    device            = "cpu",
    tensorboard_log   = "./tensorboard/",
    policy_kwargs     = policy_kwargs,
    verbose           = 1,
    learning_rate     = 1e-4,
    n_steps           = 4096,
    gae_lambda        = 0.95,
    batch_size        = 64,
    ent_coef          = 0.01,
)
model.learn(total_timesteps=500_000, tb_log_name="hedging_lstm")
model.save("hedging_lstm")


model    = RecurrentPPO.load("hedging_lstm")
test_env = HedgeEnvEval(window, n_features, X_valid, y_valid, a_max)
obs, _   = test_env.reset()

done, rewards, actions, lstm_states = False, [], [], None
while not done:
    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    rewards.extend(reward)
    actions.extend([test_env.prev_action] * len(reward))

r       = np.array(rewards)
actions = np.array(actions)

sharpe       = np.sqrt(252) * r.mean() / (r.std() + 1e-8)
downside     = r[r < 0]
sortino      = np.sqrt(252) * r.mean() / (downside.std() + 1e-8)
equity       = np.cumprod(1 + r)
peak         = np.maximum.accumulate(equity)
drawdown_arr = (equity - peak) / peak
max_drawdown = drawdown_arr.min()
years        = len(r) / 252
cagr         = equity[-1] ** (1 / years) - 1
turnover     = np.mean(np.abs(np.diff(actions)))

print(f"Sharpe: {sharpe:.3f}  Sortino: {sortino:.3f}  "
      f"Max DD: {max_drawdown:.3f}  CAGR: {cagr:.3f}  Turnover: {turnover:.4f}")

bh_returns = y_valid.numpy()
print("Buy & hold Sharpe:",
      np.sqrt(252) * bh_returns.mean() / (bh_returns.std() + 1e-8))

plt.figure(figsize=(12, 4))
plt.plot(actions)
plt.title("Action / hedge ratio over test period")
plt.savefig("action_hedge.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(equity)
plt.title("Equity curve, test period")
plt.savefig("equity.png")
plt.show()

y_test_realized = y_valid.numpy()[window - 1 : window - 1 + len(actions)]
corr = np.corrcoef(actions, y_test_realized)[0, 1]
print("Correlation(action, forward return):", corr)

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
print("Gross Sharpe (no costs):",
      np.sqrt(252) * gross_pnl.mean() / (gross_pnl.std() + 1e-8))