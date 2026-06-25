import torch
import torch.nn as nn


class DreamerPredictors(nn.Module):
    """
    The 'Eyes' and 'Sensors' of the World Model.
    Takes the internal world state (h_t combined with z_t) and attempts to
    reconstruct the true market features and step rewards.
    """

    def __init__(self, obs_dim=3, deter_dim=200, num_groups=32, num_classes=32):
        super(DreamerPredictors, self).__init__()

        # Combined size of deterministic state and flattened discrete latents
        self.feature_dim = deter_dim + (num_groups * num_classes)

        # 1. Observation Decoder: Reconstructs [log_return, vol, hedge_ratio]
        self.obs_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_dim)
        )

        # 2. Reward Decoder: Predicts the scalar scale-invariant reward
        self.reward_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1)  # Outputs a single scalar value for the reward
        )

    def forward(self, deter_state, latent_sample):
        """
        Args:
            deter_state:   The GRU memory state h_t -> (Batch, deter_dim)
            latent_sample: The one-hot discrete sample z_t -> (Batch, num_groups, num_classes)
        Returns:
            pred_obs:      Predicted market state -> (Batch, obs_dim)
            pred_reward:   Predicted step reward -> (Batch, 1)
        """
        batch_size = deter_state.size(0)

        # Flatten the categorical blocks into a singular feature vector
        latent_flat = latent_sample.view(batch_size, -1)

        # Fuse the memory state and the stochastic state together
        world_features = torch.cat([deter_state, latent_flat], dim=-1)

        # Generate predictions
        pred_obs = self.obs_decoder(world_features)
        pred_reward = self.reward_decoder(world_features)

        return pred_obs, pred_reward


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist


def calculate_kl_loss(posterior_logits, prior_logits):
    """
    Computes the KL Divergence between the analytical Posterior distribution q(z|h,x)
    and the imagined Prior distribution p(z|h) over a grid of Categorical variables.
    """
    # Create categorical distributions from raw logits
    post_dist = dist.Independent(dist.Categorical(logits=posterior_logits), reinterpreted_batch_ndims=1)
    prior_dist = dist.Independent(dist.Categorical(logits=prior_logits), reinterpreted_batch_ndims=1)

    # Calculate the KL Divergence per batch element
    kl = dist.kl_divergence(post_dist, prior_dist)
    return kl.mean()


def train_world_model_step(models, optimizer, obs_seq, action_seq, reward_seq, beta=0.1):
    """
    Args:
        models:       Dict containing {'encoder': ..., 'transition': ..., 'predictors': ...}
        optimizer:    Unified PyTorch optimizer for all world model parameters
        obs_seq:      Tensor of shape (Batch, Horizon, Obs_Dim)
        action_seq:   Tensor of shape (Batch, Horizon, Action_Dim) -> Actions taken at t-1
        reward_seq:   Tensor of shape (Batch, Horizon, 1) -> Rewards earned at t
        beta:         KL divergence penalty scaling factor (balancing parameter)
    """
    encoder = models['encoder']
    transition = models['transition']
    predictors = models['predictors']

    batch_size, horizon, obs_dim = obs_seq.size()
    deter_dim = transition.gru_cell.hidden_size

    # 1. Initialize recurrent memory state h_0 to zeros for the start of the sequence
    h_t = torch.zeros(batch_size, deter_dim).to(obs_seq.device)

    # Initialize mock previous latent state z_0 to start the temporal chain
    # We use a flat zero matrix matching our total discrete latent dimensions
    z_t = torch.zeros(batch_size, encoder.num_groups, encoder.num_classes).to(obs_seq.device)

    # Containers to collect sequence predictions for loss matching
    total_loss_obs = 0.0
    total_loss_reward = 0.0
    total_loss_kl = 0.0

    # 2. Unroll the world model sequentially across the sequence horizon
    for t in range(horizon):
        # Current true items at timestep t
        true_obs = obs_seq[:, t, :]
        true_reward = reward_seq[:, t, :]
        prev_action = action_seq[:, t, :]  # Action that led to this state

        # --- Step A: Transition Step (Predict the PRIOR from past state memory) ---
        h_t, prior_logits = transition(z_t, prev_action, h_t)

        # --- Step B: Inference Step (Calculate the POSTERIOR using current observation) ---
        posterior_logits = encoder(true_obs, h_t)
        z_t = encoder.sample_latent(posterior_logits, training=True)

        # --- Step C: Prediction Step (Map internal features back to concrete world values) ---
        pred_obs, pred_reward = predictors(h_t, z_t)

        # --- Step D: Accumulate Step Metrics ---
        total_loss_obs += nn.functional.mse_loss(pred_obs, true_obs)
        total_loss_reward += nn.functional.mse_loss(pred_reward, true_reward)
        total_loss_kl += calculate_kl_loss(posterior_logits, prior_logits)

    # 3. Average accumulated sequence metrics across the entire sequence length
    loss_obs = total_loss_obs / horizon
    loss_reward = total_loss_reward / horizon
    loss_kl = total_loss_kl / horizon

    # 4. Compute composite Loss Function
    # Free-bits or KL balancing can be added here if the latent space collapses early
    total_loss = loss_obs + loss_reward + (beta * loss_kl)

    # 5. Execute Optimization Optimization Step
    optimizer.zero_grad()
    total_loss.backward()
    # Clip gradients to prevent explosion caused by structural chain rolls
    nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(transition.parameters()) + list(predictors.parameters()),
        max_norm=2.0
    )
    optimizer.step()

    return {
        "loss_total": total_loss.item(),
        "loss_obs": loss_obs.item(),
        "loss_reward": loss_reward.item(),
        "loss_kl": loss_kl.item()
    }


if __name__ == "__main__":
    from encoder import DreamerDiscreteEncoder, DreamerTransitionGRU

    # Hyperparameters
    B, H, O, A = 8, 16, 3, 1  # Batch Size=8, Horizon=16 steps, Obs=3, Action=1

    # Instantiate global pipeline components
    models = {
        'encoder': DreamerDiscreteEncoder(obs_dim=O),
        'transition': DreamerTransitionGRU(action_dim=A),
        'predictors': DreamerPredictors(obs_dim=O)
    }

    # Build a collective optimizer spanning all three network components
    all_params = (list(models['encoder'].parameters()) +
                  list(models['transition'].parameters()) +
                  list(models['predictors'].parameters()))
    optimizer = optim.Adam(all_params, lr=3e-4)

    # Create synthetic trajectory tracking block tensors
    mock_obs_seq = torch.randn(B, H, O)
    mock_action_seq = torch.randn(B, H, A)
    mock_reward_seq = torch.randn(B, H, 1)

    # Execute structural forward-backward pass iteration
    metrics = train_world_model_step(models, optimizer, mock_obs_seq, mock_action_seq, mock_reward_seq)

    print("--- World Model Training Loop Verification Passed ---")
    for k, v in metrics.items():
        print(f"{k.upper():<12}: {v:.6f}")