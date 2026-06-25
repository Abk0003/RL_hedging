import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class DreamerDiscreteEncoder(nn.Module):
    def __init__(self, obs_dim=3, deter_dim=200, num_groups=32, num_classes=32):
        super(DreamerDiscreteEncoder, self).__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        # 1. Observation Embedding Network (Dense alternative to Dreamer's CNN)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
        )

        # 2. Posterior Fusion Layer
        # Combines the raw market feature map and the internal GRU memory (h_t)
        self.fc_fusion = nn.Linear(128 + deter_dim, 256)
        self.fc_logits = nn.Linear(256, num_groups * num_classes)

    def forward(self, obs, deter_state):
        # obs shape: (Batch, obs_dim)
        # deter_state (h_t) shape: (Batch, deter_dim)

        # Embed the market observation
        obs_embed = self.obs_encoder(obs)  # (Batch, 128)

        # Concatenate observation features with recurrent memory
        fused = torch.cat([obs_embed, deter_state], dim=-1)  # (Batch, 128 + deter_dim)
        x = F.elu(self.fc_fusion(fused))

        # Generate raw unnormalized logits for our discrete categories
        logits = self.fc_logits(x)  # (Batch, num_groups * num_classes)

        # Reshape to isolate each individual categorical group
        # Final shape: (Batch, num_groups, num_classes)
        logits = logits.view(-1, self.num_groups, self.num_classes)

        return logits

    def sample_latent(self, logits, training=True):
        """
        Samples from the categorical distributions using the Straight-Through
        Gumbel-Softmax estimator to keep the graph fully differentiable.
        """
        if training:
            return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            argmax = torch.argmax(logits, dim=-1)
            return F.one_hot(argmax, num_classes=self.num_classes).float()
        return sample  # Shape: (Batch, num_groups, num_classes)

class DreamerTransitionGRU(nn.Module):
    """
    The Deterministic Transition Component of the RSSM.
    Updates the recurrent memory state (h_t) based on the past latent state (z_t-1)
    and past action (a_t-1), then predicts the structural PRIOR logits for the next step.
    """

    def __init__(self, action_dim=1, deter_dim=200, num_groups=32, num_classes=32):
        super(DreamerTransitionGRU, self).__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        # Calculate the flattened size of our discrete latent space grid (32 * 32 = 1024)
        self.latent_flat_dim = num_groups * num_classes

        # 1. Action & Latent Feature Pre-processing Layer
        # Compresses the combined input down before feeding it to the GRU
        self.fc_input = nn.Sequential(
            nn.Linear(self.latent_flat_dim + action_dim, 256),
            nn.ELU()
        )

        # 2. Core Recurrent Unit (Tracks the temporal trajectory of the market)
        self.gru_cell = nn.GRUCell(input_size=256, hidden_size=deter_dim)

        # 3. Prior Predictor Network
        # Generates the 'imagined' network logits purely from internal GRU memory (h_t)
        self.fc_prior = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.ELU(),
            nn.Linear(256, self.latent_flat_dim)
        )

    def forward(self, prev_latent, prev_action, prev_deter):
        """
        Args:
            prev_latent:  Stochastic sample from t-1 (Batch, num_groups, num_classes)
            prev_action:  Continuous hedge action from t-1 (Batch, action_dim)
            prev_deter:   Deterministic GRU hidden state from t-1 (Batch, deter_dim)
        Returns:
            next_deter:   Updated deterministic state h_t (Batch, deter_dim)
            prior_logits: Predicted prior distributions for z_t (Batch, num_groups, num_classes)
        """
        batch_size = prev_latent.size(0)

        # Flatten the discrete latent matrix into a 1D vector per batch element
        prev_latent_flat = prev_latent.view(batch_size, self.latent_flat_dim)

        # Combine the past state and action into a unified input feature map
        combined_input = torch.cat([prev_latent_flat, prev_action], dim=-1)
        x = self.fc_input(combined_input)

        # Compute the next deterministic hidden state h_t
        next_deter = self.gru_cell(x, prev_deter)

        # Project the updated memory forward to generate the next prior logits
        prior_logits_flat = self.fc_prior(next_deter)
        prior_logits = prior_logits_flat.view(batch_size, self.num_groups, self.num_classes)

        return next_deter, prior_logits


if __name__ == "__main__":
    # Structural Setup
    batch_size = 4
    obs_dim = 3  # [log_return, vol, hedge]
    action_dim = 1  # Continuous SAC hedge ratio
    deter_dim = 200  # GRU memory units

    # Instantiate both networks
    encoder = DreamerDiscreteEncoder(obs_dim=obs_dim, deter_dim=deter_dim)
    transition = DreamerTransitionGRU(action_dim=action_dim, deter_dim=deter_dim)

    # 1. Initialize empty mock states for step t=0
    h_t = torch.zeros(batch_size, deter_dim)

    # 2. Simulate receiving an initial step observation from your environment
    mock_obs = torch.randn(batch_size, obs_dim)

    # --- STEP A: Posterior Inference (Looking at the market data) ---
    posterior_logits = encoder(mock_obs, h_t)
    z_t = encoder.sample_latent(posterior_logits, training=True)

    # 3. Simulate your SAC agent outputting a continuous trade execution action
    mock_action = torch.tanh(torch.randn(batch_size, action_dim))  # Bound between -1 and 1

    # --- STEP B: The Transition Update (Stepping time forward) ---
    h_next, prior_logits = transition(z_t, mock_action, h_t)

    print("--- RSSM Core Loop Verification ---")
    print(f"Current Latent State Sample (z_t) Shape:      {list(z_t.shape)}")
    print(f"Executed Agent Action State Shape:            {list(mock_action.shape)}")
    print(f"Next Deterministic Memory Shape (h_t+1):     {list(h_next.shape)}")
    print(f"Predicted Next Prior Logits Shape:            {list(prior_logits.shape)}")


