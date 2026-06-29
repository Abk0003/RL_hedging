"""
Binomial Option Pricer + Black-Scholes Pricer
Module 4, Week 5 — Summer of Quant

Implements:
1. Single-step binomial model — full derivation
2. CRR multi-step binomial model — European and American
3. Black-Scholes formula — calls and puts
4. Convergence demonstration: binomial → Black-Scholes as n → ∞

Maps directly to:
- Hull (Ch. 13): Binomial Trees
- Hull (Ch. 15): The Black-Scholes-Merton Model
- CRR (1979): original paper parameterization
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal

# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_d1_d2(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0):
    """Compute d1 and d2 for Black-Scholes formula."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0,   # continuous dividend yield
) -> float:
    """
    Black-Scholes option price.

    Parameters
    ----------
    S     : Current stock price
    K     : Strike price
    r     : Risk-free rate (continuously compounded)
    sigma : Volatility (annual)
    T     : Time to expiry (years)
    q     : Continuous dividend yield (default 0)

    Returns the option price.
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1, d2 = bs_d1_d2(S, K, r, sigma, T, q)

    if option_type == 'call':
        # C = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        # P = K·e^{-rT}·N(-d2) - S·e^{-qT}·N(-d1)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_greeks(S, K, r, sigma, T, option_type='call', q=0.0) -> dict:
    """Compute the main Black-Scholes Greeks."""
    d1, d2 = bs_d1_d2(S, K, r, sigma, T, q)
    N_prime_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (- S * N_prime_d1 * sigma * np.exp(-q * T) / (2 * sqrt_T)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # per calendar day
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (- S * N_prime_d1 * sigma * np.exp(-q * T) / (2 * sqrt_T)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    gamma = N_prime_d1 * np.exp(-q * T) / (S * sigma * sqrt_T)
    vega  = S * np.exp(-q * T) * N_prime_d1 * sqrt_T / 100  # per 1% vol move

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


# ── Single-Step Binomial ──────────────────────────────────────────────────────

def single_step_binomial(S0, K, U, D, r, option_type='call'):
    """
    Single-step binomial model — full derivation.
    Maps directly to Section 4.1.
    """
    # Terminal payoffs
    H_U = max(S0 * U - K, 0) if option_type == 'call' else max(K - S0 * U, 0)
    H_D = max(S0 * D - K, 0) if option_type == 'call' else max(K - S0 * D, 0)

    # Risk-neutral probability
    p_star = ((1 + r) - D) / (U - D)

    # Check no-arbitrage condition
    assert 0 < p_star < 1, f"No-arbitrage violated: p* = {p_star:.4f} not in (0,1)"

    # Replicating portfolio
    theta = (H_U - H_D) / (S0 * (U - D))
    phi = (U * H_D - D * H_U) / ((1 + r) * (U - D))

    # Option price
    V0 = phi + theta * S0
    V0_check = (p_star * H_U + (1 - p_star) * H_D) / (1 + r)

    print(f"\n{'='*50}")
    print(f"  SINGLE-STEP BINOMIAL MODEL")
    print(f"  S0={S0}, K={K}, U={U}, D={D}, r={r}")
    print(f"{'='*50}")
    print(f"  Up payoff H_U:        {H_U:.4f}")
    print(f"  Down payoff H_D:      {H_D:.4f}")
    print(f"  Risk-neutral p*:      {p_star:.4f}")
    print(f"  Delta (theta):        {theta:.4f}  shares")
    print(f"  Bond (phi):           {phi:.4f}  dollars")
    print(f"  Option price V0:      {V0:.4f}")
    print(f"  Verify (risk-neutral):{V0_check:.4f}")
    print(f"{'='*50}\n")
    return V0


# ── CRR Multi-Step Binomial ───────────────────────────────────────────────────

def crr_binomial(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    option_type: Literal['call', 'put'] = 'call',
    american: bool = False,
) -> float:
    """
    CRR multi-step binomial model.
    Handles European and American options.

    Parameters
    ----------
    n        : Number of time steps
    american : If True, prices an American option via backward induction
    """
    h = T / n                        # length of each time step
    U = np.exp(sigma * np.sqrt(h))   # CRR up factor
    D = 1 / U                        # CRR down factor (U × D = 1)
    p_star = (np.exp(r * h) - D) / (U - D)   # risk-neutral probability
    discount = np.exp(-r * h)        # one-step discount factor

    # ── Build terminal node prices ────────────────────────────────────────────
    # At step n, the stock has had j up-moves and (n-j) down-moves
    j = np.arange(n + 1)
    S_T = S0 * (U ** j) * (D ** (n - j))

    # ── Terminal payoffs ──────────────────────────────────────────────────────
    if option_type == 'call':
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)

    # ── Backward induction ────────────────────────────────────────────────────
    for step in range(n - 1, -1, -1):
        # Stock prices at this step
        j_step = np.arange(step + 1)
        S_step = S0 * (U ** j_step) * (D ** (step - j_step))

        # Continuation value (risk-neutral discounted expectation)
        CV = discount * (p_star * V[1:step + 2] + (1 - p_star) * V[0:step + 1])

        if american:
            # For American option: take max of continuation and intrinsic
            if option_type == 'call':
                intrinsic = np.maximum(S_step - K, 0)
            else:
                intrinsic = np.maximum(K - S_step, 0)
            V = np.maximum(CV, intrinsic)
        else:
            V = CV

    return float(V[0])


# ── Implied Volatility via Newton-Raphson ─────────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0,
) -> float:
    """
    Extract implied volatility from an observed market price.
    Uses Brent's method for robustness.

    This is the Black-Scholes inversion problem — a standard
    quant interview task. The IV is the sigma that makes
    BS(sigma) = market_price.
    """
    def objective(sigma):
        return black_scholes(S, K, r, sigma, T, option_type, q) - market_price

    # IV must be in (0, 5) = (0%, 500%) — a very wide range
    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=1e-8)
        return iv
    except ValueError:
        return np.nan


# ── Convergence Demonstration ─────────────────────────────────────────────────

def convergence_demo(S0=100, K=100, r=0.05, sigma=0.20, T=1.0):
    """
    Demonstrate that CRR binomial → Black-Scholes as n → ∞.
    This is the visual proof that binomial and BS are the same model.
    """
    bs_call = black_scholes(S0, K, r, sigma, T, 'call')
    bs_put  = black_scholes(S0, K, r, sigma, T, 'put')

    n_values = list(range(1, 201, 2))
    crr_calls = [crr_binomial(S0, K, r, sigma, T, n, 'call') for n in n_values]
    crr_puts  = [crr_binomial(S0, K, r, sigma, T, n, 'put')  for n in n_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, crr_vals, bs_val, title in [
        (ax1, crr_calls, bs_call, 'European Call'),
        (ax2, crr_puts,  bs_put,  'European Put'),
    ]:
        ax.plot(n_values, crr_vals, 'b-', alpha=0.7, linewidth=1.5, label='CRR Binomial')
        ax.axhline(bs_val, color='r', linestyle='--', linewidth=2,
                   label=f'Black-Scholes: {bs_val:.4f}')
        ax.set_xlabel('Number of Steps n', fontsize=12)
        ax.set_ylabel('Option Price', fontsize=12)
        ax.set_title(f'{title} — CRR Convergence to Black-Scholes', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print(f"\nBlack-Scholes Call: {bs_call:.6f}")
    print(f"CRR Call (n=200):  {crr_calls[-1]:.6f}")
    print(f"Difference:        {abs(bs_call - crr_calls[-1]):.6f}")


# ── EXAMPLES ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Single-step binomial — Section 4.1 worked example
    single_step_binomial(S0=100, K=100, U=1.10, D=0.90, r=0.05, option_type='call')

    # 2. CRR binomial — European call
    S0, K, r, sigma, T = 42, 40, 0.10, 0.20, 0.5
    euro_call = crr_binomial(S0, K, r, sigma, T, n=100, option_type='call', american=False)
    bs_call   = black_scholes(S0, K, r, sigma, T, 'call')
    print(f"\nHull Example (Ch.13): S=42, K=40, r=10%, σ=20%, T=0.5")
    print(f"CRR (n=100) call:  {euro_call:.4f}")
    print(f"Black-Scholes call:{bs_call:.4f}  (Hull gets 4.76)")

    # 3. American put — where early exercise matters
    amer_put = crr_binomial(S0, K, r, sigma, T, n=100, option_type='put', american=True)
    euro_put = crr_binomial(S0, K, r, sigma, T, n=100, option_type='put', american=False)
    print(f"\nAmerican put:  {amer_put:.4f}")
    print(f"European put:  {euro_put:.4f}")
    print(f"Early exercise premium: {amer_put - euro_put:.4f}")

    # 4. Black-Scholes with Greeks
    price = black_scholes(100, 100, 0.05, 0.20, 1.0, 'call')
    greeks = bs_greeks(100, 100, 0.05, 0.20, 1.0, 'call')
    print(f"\nATM Call: S=100, K=100, r=5%, σ=20%, T=1y")
    print(f"Price:  {price:.4f}")
    for name, val in greeks.items():
        print(f"  {name:<8} {val:+.4f}")

    # 5. Implied volatility extraction
    market_price = 10.45   # observed market price
    iv = implied_volatility(market_price, S=100, K=100, r=0.05, T=0.5, option_type='call')
    print(f"\nMarket call price: {market_price}")
    print(f"Implied volatility: {iv:.4%}")
    print(f"Verify (BS at IV): {black_scholes(100, 100, 0.05, iv, 0.5, 'call'):.4f}")

    # 6. Convergence demonstration
    convergence_demo()