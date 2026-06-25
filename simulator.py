import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from pyarrow.lib import float32


class QEHestonHedging(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self,ticker="SPY",steps_per_ep=100,dt=1/252,risk_av=1.0,fee_rate=0.001,
                 mu = 0.05, theta = 0.04, initial_price=500.0,epsilon=0.5,rho=-0.7, kappa=2.0,initial_variance=0.04):
        super(QEHestonHedging,self).__init__()
        self.ticker = ticker
        self.steps_per_ep = steps_per_ep
        self.dt = dt
        self.risk_av = risk_av
        self.fee_rate = fee_rate

        self.initial_price = initial_price
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.rho = rho
        self.initial_variance = initial_variance

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 1e-5, -1.0],dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0],dtype=np.float32),
            dtype=np.float32
        )
        self.max_steps = 252  # 1 trading year limit (change this to your preferred horizon)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.hedge_ratio = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.S_t = self.initial_price
        self.v_t = self.initial_variance
        self.portfolio =  self.cash
        self.prev_portfolio = self.portfolio
        self.state = np.array([0.0,np.sqrt(self.v_t),self.hedge_ratio])
        return self.state, {}
    def step(self, action):
        target_hedge = float(action[0])
        prev_S = self.S_t
        prev_v = self.v_t
        m = prev_v * np.exp(-self.kappa*self.dt) + self.theta*(1-np.exp(-self.kappa*self.dt))
        s2 = (self.epsilon**2/self.kappa) * prev_v * np.exp(-self.kappa*self.dt)*(1-np.exp(-self.kappa*self.dt)) + (self.theta * self.epsilon**2 / (2 * self.kappa)) * (1 - np.exp(-self.kappa * self.dt))**2
        phi = s2 / (m**2)

        Zv = np.random.normal(0, 1)
        Zp = np.random.normal(0, 1)
        Zs = Zv * self.rho + (1 - self.rho ** 2) ** 0.5 * Zp

        if phi <= 1.5:
            b2 = 2 / phi - 1 + np.sqrt(2 / phi * (2 / phi - 1))
            a = m / (1 + b2)
            self.v_t = max(a * (np.sqrt(b2) + Zv) ** 2, 1e-10)
        else:
            p = (phi - 1) / (phi + 1)
            beta = (1 - p) / m
            U = np.random.uniform(1e-7, 1 - 1e-7)
            self.v_t = max(0.0 if U <= p else np.log((1 - p) / (1 - U)) / beta, 1e-10)

        self.S_t = prev_S * np.exp((self.mu - 0.5 * prev_v) * self.dt + np.sqrt(prev_v * self.dt) * Zs)
        hedge_change = target_hedge - self.hedge_ratio
        trade_volume = abs(hedge_change * prev_S)
        transaction_cost = trade_volume * self.fee_rate

        self.cash -= (hedge_change * prev_S) + transaction_cost
        self.hedge_ratio = target_hedge

        self.portfolio = self.cash + (self.hedge_ratio * self.S_t)
        pnl = self.portfolio - self.prev_portfolio
        pnl_normalized = pnl / self.initial_price
        reward = pnl_normalized - self.risk_av * (pnl_normalized ** 2)

        self.prev_portfolio = self.portfolio
        self.current_step += 1

        log_return = np.log(self.S_t / prev_S)
        next_state = np.array([log_return, np.sqrt(self.v_t), self.hedge_ratio], dtype=np.float32)

        terminated = self.portfolio < self.initial_cash * 0.5
        truncated = self.current_step >= self.max_steps

        info = {
            "underlying_price": self.S_t,
            "annual_vol": np.sqrt(self.v_t),
            "portfolio_value": self.portfolio
        }

        return next_state, reward, terminated, truncated, info


if __name__ == "__main__":
    # Initialize your new environment
    env = QEHestonHedging(steps_per_ep=100)
    state, info = env.reset()

    print("--- Testing Wrapped Gym Environment ---")
    print(f"Starting Asset Price: {env.S_t:.2f}")
    print(f"Starting Annualized Vol: {np.sqrt(env.v_t) * 100:.2f}%\n")

    done = False
    while not done:
        random_action = np.array([-0.5], dtype=np.float32)
        state, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated

        if env.current_step % 20 == 0:
            print(f"Step {env.current_step:03d} | Price: {info['underlying_price']:.2f} | "
                  f"Vol: {info['annual_vol'] * 100:.2f}% | Portfolio Value: {info['portfolio_value']:.2f} | "
                  f"Step Reward: {reward:.4f}")
