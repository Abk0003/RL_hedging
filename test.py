import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO


# Assuming the classes HedgeEnv, HedgeEnvEval, window, etc., are in scope

def run_stress_test(model_path, norm_path, X_data, y_data, asset_name="SPY", cost_multiplier=1.0):
    print(f"\n--- Running Stress Test: {asset_name} | Cost Mult: {cost_multiplier} ---")

    # 1. Custom Eval Environment with injected costs
    class StressTestEnv(HedgeEnvEval):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.phi *= cost_multiplier  # Stressing transaction costs

    env_dummy = DummyVecEnv([lambda: StressTestEnv(window, n_features, X_data, y_data, a_max)])
    vec_norm = VecNormalize.load(norm_path, env_dummy)
    vec_norm.training = False

    model = RecurrentPPO.load(model_path)
    obs, _ = env_dummy.reset()

    done = False
    rewards = []

    # We must manually manage the state for RecurrentPPO
    lstm_states = None
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, _ = env_dummy.step(action)
        rewards.append(reward[0])

    # Metrics
    r = np.array(rewards)
    sharpe = np.sqrt(252) * r.mean() / (r.std() + 1e-8)
    print(f"Result for {asset_name} -> Sharpe: {sharpe:.3f}")
    return sharpe


# --- EXECUTIONS ---

# 1. THE LOOK-AHEAD TEST
# Shift features by 1. If the model still performs well, it was reading 'tomorrow'
# in your original feature set.
run_stress_test("hedging_lstm", "vec_normalize.pkl", X_test[1:], y_test[:-1], "Look-Ahead Shifted")

# 2. THE ASSET TRANSFER TEST (QQQ / DIA)
# You need to generate X_test_QQQ and y_test_QQQ using the same market_data.py logic
# for a different ticker. If this fails, your model only learned 'SPY' patterns.
# run_stress_test("hedging_lstm", "vec_normalize.pkl", X_test_QQQ, y_test_QQQ, "QQQ Transfer")

# 3. THE LIQUIDITY CRUNCH TEST
# If performance collapses here, your model is not 'Alpha', it is 'Liquidity Arbitrage'.
run_stress_test("hedging_lstm", "vec_normalize.pkl", X_test, y_test, "SPY High-Cost", cost_multiplier=5.0)

# 4. RANDOM ACTION BASELINE
# Compare against a 'Random Agent'. If your model's Sharpe is close to this, it's garbage.
random_r = np.random.normal(0, 0.01, len(y_test))
print(f"Random Baseline Sharpe: {np.sqrt(252) * random_r.mean() / (random_r.std() + 1e-8):.3f}")