import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import all modules
from buffer import SequenceReplayBuffer
from encoder import DreamerDiscreteEncoder, DreamerTransitionGRU
from model import DreamerSACActor, DreamerSACCritic
from training import DreamerPredictors
from simulator import QEHestonHedging
import yfinance as yf


def get_heston_params(ticker="SPY"):
    print("Downloading market data for parameter calibration...")
    data = yf.download(ticker, start="2005-01-01", end="2025-12-31", progress=False)
    price = data["Close"].dropna()
    returns = np.log(price / price.shift(1)).dropna().values.flatten()
    idx = int(0.8 * len(returns))
    r_train = returns[:idx]
    return {
        "initial_price": float(price.values.flatten()[-1]),
        "mu":            float(np.mean(r_train) * 252),
        "theta":         float(np.var(returns) * 252),
        "initial_variance": float(np.var(returns) * 252),
    }


def main():
    # ── 1. Hardware & Hyperparameters ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying training pipeline on device: {device}")

    obs_dim     = 3
    action_dim  = 1
    deter_dim   = 200
    num_groups  = 32
    num_classes = 32

    batch_size = 16
    horizon    = 16
    # FIX — burn-in window. SequenceReplayBuffer.sample() returns windows that
    # start at an ARBITRARY point inside an episode, not necessarily t=0. The
    # original code initialized h_wm/z_wm (and h_enc/z_enc) to zero at the
    # start of every sampled window regardless of where that window actually
    # falls in the episode. That told the GRU "you have no history" at a
    # timestep that might be 50+ real steps into a hedging trajectory, which
    # corrupted the world model, the re-encoded SAC states, and therefore the
    # critic/actor/imagination pipeline downstream of it. Burn-in steps are
    # unrolled with no gradient and no loss, purely to re-derive the correct
    # (h, z) belief state before any loss-bearing step.
    burn_in    = 8
    wm_lr      = 3e-4
    gamma      = 0.99
    tau        = 0.02
    beta       = 0.1
    FREE_BITS  = 0.3   # KL free-bits threshold (nats) — prevents posterior collapse
    N_CRITICS  = 5     # RedQ ensemble size
    M_TARGET   = 2     # RedQ: subsample M of N for target to break critic correlation

    # Dreamer imagination parameters
    imag_horizon    = 15    # Steps to roll forward in latent space
    imag_lambda     = 0.95  # TD-lambda mixing coefficient for lambda-returns
    n_imag_starts   = 64    # Starting states sampled per imagination rollout
    imag_crit_coef  = 0.5   # Weight of imagination critic loss vs real-buffer loss

    # FIX — target_entropy for a 1-D continuous action should be ~ -action_dim
    # (standard SAC heuristic). -0.5 was tighter than that, which pushes alpha
    # to decay toward near-deterministic behavior faster than the policy has
    # earned through reliable Q-estimates. Tie it to action_dim explicitly so
    # it scales correctly if action_dim ever changes.
    target_entropy = -float(action_dim)
    log_alpha      = torch.tensor([-1.0], requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)

    total_episodes = 10000

    # ── 2. Environment & Replay Buffer ─────────────────────────────────────────
    params = get_heston_params()
    env    = QEHestonHedging(**params)
    # FIX — pass burn_in through so buffer.sample() returns (burn_in + horizon)
    # length windows instead of bare horizon-length windows.
    buffer = SequenceReplayBuffer(max_episodes=1000, horizon=horizon, burn_in=burn_in)

    # FIX 1 — Proper Welford running statistics.
    # The old code accumulated `reward_rms_var += delta * delta2`, which is the
    # cumulative sum of squared deviations, not the running variance. Dividing
    # that sum by count gives a value that grows without bound, inflating
    # `current_std` and shrinking `scaled_reward` toward zero over time —
    # a progressively weaker reward signal for the critic.
    # The correct update stores the running *sample variance* directly:
    #   var_new = var_old + (delta * delta2 - var_old) / count
    reward_rms_mean  = 0.0
    reward_rms_var   = 1.0   # Running sample variance (bounded; not cumulative sum)
    reward_rms_count = 0

    reward_history = []

    # ── 3. World Model ──────────────────────────────────────────────────────────
    wm_components = {
        "encoder":    DreamerDiscreteEncoder(obs_dim, deter_dim, num_groups, num_classes).to(device),
        "transition": DreamerTransitionGRU(action_dim, deter_dim, num_groups, num_classes).to(device),
        "predictors": DreamerPredictors(obs_dim, deter_dim, num_groups, num_classes).to(device),
    }
    wm_params = (
        list(wm_components["encoder"].parameters()) +
        list(wm_components["transition"].parameters()) +
        list(wm_components["predictors"].parameters())
    )
    wm_optimizer = optim.Adam(wm_params, lr=wm_lr)

    # ── 4. Actor & RedQ Critic Ensemble ────────────────────────────────────────
    critics = nn.ModuleList([
        DreamerSACCritic(action_dim, deter_dim, num_groups, num_classes).to(device)
        for _ in range(N_CRITICS)
    ])
    critics_target = nn.ModuleList([
        DreamerSACCritic(action_dim, deter_dim, num_groups, num_classes).to(device)
        for _ in range(N_CRITICS)
    ])
    for ct, c in zip(critics_target, critics):
        ct.load_state_dict(c.state_dict())

    actor = DreamerSACActor(action_dim, deter_dim, num_groups, num_classes).to(device)

    # FIX 4 — Critic learns faster than actor; mild L2 slows weight magnitude growth.
    # Old code had critic lr=1e-4 (same as actor) with weight_decay=1e-4 (too strong).
    critic_optimizer = optim.Adam(critics.parameters(), lr=3e-4, weight_decay=1e-5)
    actor_optimizer  = optim.Adam(actor.parameters(),  lr=1e-4)

    # ── 5. Training Loop ────────────────────────────────────────────────────────
    print("\nStarting Phase 1: Environment Warm-up & Trajectory Engine...")

    for episode in range(total_episodes):

        # ── 5a. Environment Interaction ─────────────────────────────────────────
        obs, _ = env.reset()
        h_t = torch.zeros(1, deter_dim,              device=device)
        z_t = torch.zeros(1, num_groups, num_classes, device=device)

        ep_obs, ep_actions, ep_rewards = [obs], [], []
        ep_raw_cumulative = 0.0
        done = False

        while not done:
            with torch.no_grad():
                action, _ = actor.sample_action(h_t, z_t, deterministic=False)
                action_np  = action.cpu().squeeze(0).numpy()

            next_obs, raw_reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Fixed Welford: reward_rms_var holds the running sample variance
            reward_rms_count += 1
            delta             = raw_reward - reward_rms_mean
            reward_rms_mean  += delta / reward_rms_count
            delta2            = raw_reward - reward_rms_mean
            reward_rms_var    = reward_rms_var + (delta * delta2 - reward_rms_var) / reward_rms_count

            current_std   = max(np.sqrt(reward_rms_var), 1e-6)
            scaled_reward = float(np.clip(
                (raw_reward - reward_rms_mean) / current_std, -5.0, 5.0
            ))

            ep_actions.append(action_np)
            ep_rewards.append(scaled_reward)
            ep_obs.append(next_obs)
            ep_raw_cumulative += raw_reward

            with torch.no_grad():
                obs_t  = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                h_t, _ = wm_components["transition"](z_t, action.to(device), h_t)
                post_l = wm_components["encoder"](obs_t, h_t)
                z_t    = wm_components["encoder"].sample_latent(post_l, training=False)

            obs = next_obs

        buffer.add_episode(ep_obs[:-1], ep_actions, ep_rewards)
        reward_history.append(ep_raw_cumulative)

        # FIX — the old guard `len(buffer) < 5` only checks episode COUNT, not
        # episode LENGTH. SequenceReplayBuffer.sample() needs episodes longer
        # than (burn_in + horizon) or it raises / spins. Skip training updates
        # until we actually have at least one usable episode, so early
        # training doesn't crash once burn-in is introduced.
        usable_episodes = sum(
            1 for ep in buffer.obs_buffer if len(ep) > (burn_in + horizon)
        )
        if usable_episodes < 5:
            continue

        # ── 5b. Sample Real Batch ───────────────────────────────────────────────
        # obs_seq/action_seq/reward_seq now have length (burn_in + horizon).
        obs_seq, action_seq, reward_seq = buffer.sample(batch_size)
        obs_seq    = obs_seq.to(device)
        action_seq = action_seq.to(device)
        reward_seq = reward_seq.to(device)

        # ── 5c. Burn-in: warm up recurrent state with no loss ───────────────────
        # Re-derive the true (h, z) belief state at the window's start_idx by
        # unrolling the first `burn_in` steps through the transition + encoder.
        # No gradients, no loss — this step exists purely to avoid telling the
        # GRU "you have zero history" at a timestep that isn't actually t=0.
        h_wm = torch.zeros(batch_size, deter_dim,              device=device)
        z_wm = torch.zeros(batch_size, num_groups, num_classes, device=device)
        with torch.no_grad():
            for t in range(burn_in):
                o_t = obs_seq[:, t]
                a_t = action_seq[:, t]
                h_wm, prior_logits = wm_components["transition"](z_wm, a_t, h_wm)
                post_logits        = wm_components["encoder"](o_t, h_wm)
                z_wm               = wm_components["encoder"].sample_latent(post_logits, training=False)

        # ── 5c (cont). World Model Update (real data, loss-bearing steps only) ──
        # h_wm/z_wm now hold the warmed-up state at the start of the loss
        # window.
        total_obs_loss = total_rew_loss = total_kl_loss = 0.0

        wm_optimizer.zero_grad()
        for t in range(burn_in, burn_in + horizon):
            o_t = obs_seq[:, t]
            r_t = reward_seq[:, t]
            a_t = action_seq[:, t]

            h_wm, prior_logits = wm_components["transition"](z_wm, a_t, h_wm)
            post_logits        = wm_components["encoder"](o_t, h_wm)
            z_wm               = wm_components["encoder"].sample_latent(post_logits, training=True)
            pred_obs, pred_rew = wm_components["predictors"](h_wm, z_wm)

            total_obs_loss += F.smooth_l1_loss(pred_obs, o_t)
            total_rew_loss += F.smooth_l1_loss(pred_rew, r_t)

            # FIX 5 — KL balance with free bits (prevents posterior collapse).
            # Without a free-bits floor the KL penalty is active even when the
            # posterior is already close to the prior, collapsing the latent space.
            post_d     = torch.distributions.Categorical(logits=post_logits)
            prior_d    = torch.distributions.Categorical(logits=prior_logits)
            post_d_sg  = torch.distributions.Categorical(logits=post_logits.detach())
            prior_d_sg = torch.distributions.Categorical(logits=prior_logits.detach())

            kl1 = torch.clamp(
                torch.distributions.kl_divergence(post_d_sg, prior_d).sum(-1).mean(),
                min=FREE_BITS,
            )
            kl2 = torch.clamp(
                torch.distributions.kl_divergence(post_d, prior_d_sg).sum(-1).mean(),
                min=FREE_BITS,
            )
            total_kl_loss += 0.8 * kl1 + 0.2 * kl2

        wm_loss = (
            total_obs_loss / horizon +
            total_rew_loss / horizon +
            beta * total_kl_loss / horizon
        )
        wm_loss.backward()
        nn.utils.clip_grad_norm_(wm_params, 2.0)
        wm_optimizer.step()

        # ── 5d. Re-encode Latent States for SAC (no grad) ──────────────────────
        # Same burn-in treatment: warm up h_enc/z_enc over the first `burn_in`
        # steps, THEN start collecting (h, z) pairs only from the loss window.
        # This keeps the SAC critic/actor training on states whose recurrent
        # history is actually consistent with where they sit in the episode.
        h_list, z_list = [], []
        h_enc = torch.zeros(batch_size, deter_dim,              device=device)
        z_enc = torch.zeros(batch_size, num_groups, num_classes, device=device)
        with torch.no_grad():
            for t in range(burn_in):
                h_enc, _ = wm_components["transition"](z_enc, action_seq[:, t], h_enc)
                post_l   = wm_components["encoder"](obs_seq[:, t], h_enc)
                z_enc    = wm_components["encoder"].sample_latent(post_l, training=False)

            for t in range(burn_in, burn_in + horizon):
                h_enc, _ = wm_components["transition"](z_enc, action_seq[:, t], h_enc)
                post_l   = wm_components["encoder"](obs_seq[:, t], h_enc)
                z_enc    = wm_components["encoder"].sample_latent(post_l, training=False)
                h_list.append(h_enc)
                z_list.append(z_enc)

        h_states = torch.stack(h_list, dim=1)   # [B, horizon, deter_dim]
        z_states = torch.stack(z_list, dim=1)   # [B, horizon, G, C]

        # NOTE: action_seq/reward_seq must be sliced to the same loss window
        # [burn_in : burn_in+horizon] wherever they're paired with
        # h_states/z_states, since those tensors only cover that window now.
        action_window = action_seq[:, burn_in:burn_in + horizon]
        reward_window = reward_seq[:, burn_in:burn_in + horizon]

        current_alpha = log_alpha.exp().detach()

        # ── 5e. Imagination Rollout — DreamerV3 Option A ────────────────────────
        #
        # Starting from a random subset of encoded real states, the world model
        # is unrolled forward for `imag_horizon` steps under the current policy.
        # The prior (transition model output) is used as the next latent state
        # since no real observation is available.
        #
        # This generates (state, action, predicted_reward) tuples that are used for:
        #   (i)  Computing TD-lambda returns as critic training targets — giving
        #        the critic a long-horizon, low-bias value signal that is richer
        #        than one-step real-data TD targets alone.
        #   (ii) Providing diverse starting states for the actor update, which
        #        improves policy coverage beyond what the real buffer offers.
        #
        flat_h = h_states.reshape(-1, deter_dim).detach()
        flat_z = z_states.reshape(-1, num_groups, num_classes).detach()
        n_starts  = min(n_imag_starts, flat_h.shape[0])
        start_idx = torch.randperm(flat_h.shape[0], device=device)[:n_starts]

        h_cur = flat_h[start_idx]   # [n_starts, deter_dim]
        z_cur = flat_z[start_idx]   # [n_starts, G, C]

        # Buffers — we also store the final bootstrap state so all_h has H+1 entries
        imag_h_buf = []   # states:  H+1 entries
        imag_z_buf = []
        imag_r_buf = []   # rewards: H entries
        imag_a_buf = []   # actions: H entries

        with torch.no_grad():
            for t in range(imag_horizon):
                imag_h_buf.append(h_cur)
                imag_z_buf.append(z_cur)

                a_imag, _ = actor.sample_action(h_cur, z_cur, deterministic=False)
                _, r_pred = wm_components["predictors"](h_cur, z_cur)

                imag_r_buf.append(r_pred)
                imag_a_buf.append(a_imag)

                # Use prior as next latent state (no real observation in imagination)
                h_next, prior_logits = wm_components["transition"](z_cur, a_imag, h_cur)
                z_next = wm_components["encoder"].sample_latent(prior_logits, training=False)
                h_cur, z_cur = h_next, z_next

            # Append the final bootstrap state
            imag_h_buf.append(h_cur)
            imag_z_buf.append(z_cur)

        all_h = torch.stack(imag_h_buf, dim=0)  # [H+1, B, deter_dim]
        all_z = torch.stack(imag_z_buf, dim=0)  # [H+1, B, G, C]
        all_r = torch.stack(imag_r_buf, dim=0)  # [H,   B, 1]
        all_a = torch.stack(imag_a_buf, dim=0)  # [H,   B, action_dim]

        # Value estimates V(s) ≈ min_Q(s, pi(s)) at all H+1 states (for bootstrapping)
        with torch.no_grad():
            h_flat_v = all_h.reshape(-1, deter_dim)
            z_flat_v = all_z.reshape(-1, num_groups, num_classes)
            v_a, _   = actor.sample_action(h_flat_v, z_flat_v, deterministic=True)
            v_all_q  = torch.stack([c(h_flat_v, z_flat_v, v_a) for c in critics], dim=0)
            v_flat   = v_all_q.min(dim=0).values                         # [(H+1)*B, 1]
            v_states_imag = v_flat.reshape(imag_horizon + 1, n_starts, 1)

        # TD-lambda returns — back-fill from t = H-1 down to t = 0.
        # V_lambda(t) = r(t) + gamma * [(1-lambda)*V(t+1) + lambda*V_lambda(t+1)]
        lambda_returns = torch.zeros(imag_horizon, n_starts, 1, device=device)
        v_lambda = v_states_imag[-1]   # Bootstrap from final imagined state
        for t in reversed(range(imag_horizon)):
            v_lambda = (
                all_r[t] +
                gamma * ((1.0 - imag_lambda) * v_states_imag[t + 1] +
                          imag_lambda        * v_lambda)
            )
            lambda_returns[t] = v_lambda
        lambda_returns = lambda_returns.detach()

        # ── 5f. Critic Update (real TD + imagination lambda-returns) ────────────
        # Real-buffer one-step TD targets
        h_s  = h_states[:, :-1].reshape(-1, deter_dim)
        z_s  = z_states[:, :-1].reshape(-1, num_groups, num_classes)
        a_s  = action_window[:, :-1].reshape(-1, action_dim)
        r_s  = reward_window[:, :-1].reshape(-1, 1)
        h_ns = h_states[:, 1:].reshape(-1, deter_dim)
        z_ns = z_states[:, 1:].reshape(-1, num_groups, num_classes)

        with torch.no_grad():
            next_a, next_lp = actor.sample_action(h_ns, z_ns, deterministic=False)
            # FIX 2 — RedQ proper: randomly subsample M_TARGET of N_CRITICS target
            # critics per update step. All-N targets are still correlated because
            # they share training targets; random M-of-N subsampling breaks that
            # correlation and produces a more conservative (less overestimated) target.
            subset = np.random.choice(N_CRITICS, M_TARGET, replace=False)
            tq = torch.stack([
                critics_target[i](h_ns, z_ns, next_a) for i in subset
            ], dim=0).min(0).values
            expected_q = r_s + gamma * (tq - current_alpha * next_lp)

        real_crit_loss = sum(
            F.smooth_l1_loss(c(h_s, z_s, a_s), expected_q) for c in critics
        )

        # Imagination lambda-return targets: Q(s_imag, a_imag) → lambda_return(t)
        h_im = all_h[:-1].reshape(-1, deter_dim).detach()
        z_im = all_z[:-1].reshape(-1, num_groups, num_classes).detach()
        a_im = all_a.reshape(-1, action_dim).detach()
        r_im = lambda_returns.reshape(-1, 1)

        imag_crit_loss = sum(
            F.smooth_l1_loss(c(h_im, z_im, a_im), r_im) for c in critics
        )

        # Combined: real data grounds the critic; imagination extends its horizon
        critic_loss = real_crit_loss + imag_crit_coef * imag_crit_loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 2.0)
        critic_optimizer.step()

        # ── 5g. Actor Update (imagination starting states, Q-gradient) ──────────
        #
        # The actor is trained at the imagined states via the SAC reparameterization
        # gradient through the Q-function. This sidesteps the need for straight-
        # through gradients through the discrete latent transitions while still
        # giving the actor a richer, more diverse set of starting states than the
        # real buffer alone. The critic's lambda-return training (5f) means the
        # Q-function accurately reflects long-horizon returns at these states.
        #
        h_actor = all_h[:-1].reshape(-1, deter_dim).detach()
        z_actor = all_z[:-1].reshape(-1, num_groups, num_classes).detach()

        sampled_a, log_p = actor.sample_action(h_actor, z_actor, deterministic=False)
        q_actor = torch.stack([
            c(h_actor, z_actor, sampled_a) for c in critics
        ], dim=0).min(0).values   # Conservative: min over all live critics

        actor_loss = (current_alpha * log_p - q_actor).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        actor_optimizer.step()

        # ── 5h. Alpha Update ────────────────────────────────────────────────────
        alpha_loss = -(log_alpha * (log_p + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # ── 5i. Soft-Update All Target Critics ──────────────────────────────────
        for c, ct in zip(critics, critics_target):
            for p, tp in zip(c.parameters(), ct.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

        # ── 5j. Logging ─────────────────────────────────────────────────────────
        if episode % 10 == 0:
            with torch.no_grad():
                all_q  = torch.stack([c(h_s, z_s, a_s) for c in critics], dim=0)
                q_mean = all_q.mean().item()
                q_bias = (all_q.mean(0) - expected_q).mean().item()

            avg_pnl = (np.mean(reward_history[-50:])
                       if len(reward_history) >= 5
                       else ep_raw_cumulative)

            print(
                f"Episode: {episode:<4} | Avg PnL (50-ep): {avg_pnl:<8.4f} | "
                f"WM Loss: {wm_loss.item():<7.4f} | Actor Loss: {actor_loss.item():<7.4f} | "
                f"Mean Q: {q_mean:<6.4f} | Q Bias: {q_bias:+.5f} | "
                f"Alpha: {log_alpha.exp().item():.4f}"
            )


if __name__ == "__main__":
    main()