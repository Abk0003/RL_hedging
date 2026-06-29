import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class DreamerSACActor(nn.Module):
    """
    The SAC Policy Network adapted for Dreamer Latent Space.
    Takes the combined world state (h_t, z_t) and outputs a continuous,
    reparameterized hedging action bounded between [-1, 1].
    """

    def __init__(self, action_dim=1, deter_dim=200, num_groups=32, num_classes=32, log_std_min=-20, log_std_max=2):
        super(DreamerSACActor, self).__init__()
        self.latent_flat_dim = num_groups * num_classes
        self.feature_dim = deter_dim + self.latent_flat_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extraction network
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU()
        )

        # Decoupled heads for Gaussian Policy parameters
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, deter_state, latent_sample):
        """
        Processes features and returns the parameters of the action distribution.
        """
        batch_size = deter_state.size(0)
        latent_flat = latent_sample.view(batch_size, -1)

        # Fuse internal world model states
        world_features = torch.cat([deter_state, latent_flat], dim=-1)
        x = self.net(world_features)

        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        # Clamp log_std to prevent exploding variance / division by zero
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample_action(self, deter_state, latent_sample, deterministic=False):
        """
        Samples an action using the reparameterization trick:
        a_t = tanh(mu + sigma * epsilon)
        """
        mu, log_std = self.forward(deter_state, latent_sample)
        std = torch.exp(log_std)

        if deterministic:
            return torch.tanh(mu), None

        # Create normal distribution for reparameterization sampling
        normal = dist.Normal(mu, std)
        x_t = normal.rsample()  # rsample uses the gradient-friendly trick
        action = torch.tanh(x_t)

        # Enforce SAC entropy log-likelihood adjustment for the Tanh mapping
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class DreamerSACCritic(nn.Module):
    """
    The SAC Q-Network adapted for Dreamer Latent Space.
    Evaluates the value of a specific action given the world state: Q(s_t, a_t)
    """

    def __init__(self, action_dim=1, deter_dim=200, num_groups=32, num_classes=32):
        super(DreamerSACCritic, self).__init__()
        self.feature_dim = deter_dim + (num_groups * num_classes)

        # Multi-Layer Perceptron evaluating state + action pairs
        self.q_net = nn.Sequential(
            nn.Linear(self.feature_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, deter_state, latent_sample, action):
        batch_size = deter_state.size(0)
        latent_flat = latent_sample.view(batch_size, -1)

        # Combine state features and action vector
        inputs = torch.cat([deter_state, latent_flat, action], dim=-1)
        return self.q_net(inputs)


if __name__ == "__main__":
    # Test Dimensions
    batch_size = 4
    deter_dim = 200
    num_groups = 32
    num_classes = 32

    # Instantiate Networks
    actor = DreamerSACActor(action_dim=1, deter_dim=deter_dim, num_groups=num_groups, num_classes=num_classes)
    critic = DreamerSACCritic(action_dim=1, deter_dim=deter_dim, num_groups=num_groups, num_classes=num_classes)

    # Generate mock inputs reflecting outputs from your RSSM world model
    mock_h_t = torch.randn(batch_size, deter_dim)
    mock_z_t = torch.randn(batch_size, num_groups, num_classes)

    # 1. Test Actor Sampling (Stochastic execution for data collection)
    stochastic_action, log_prob = actor.sample_action(mock_h_t, mock_z_t, deterministic=False)

    # 2. Test Actor Inference (Deterministic execution for real-world trading deployment)
    deterministic_action, _ = actor.sample_action(mock_h_t, mock_z_t, deterministic=True)

    # 3. Test Critic Evaluation
    q_val = critic(mock_h_t, mock_z_t, stochastic_action)

    print("--- SAC Actor-Critic Latent Check ---")
    print(f"Stochastic Action Shape: {list(stochastic_action.shape)} -> values: {stochastic_action.squeeze().tolist()}")
    print(f"Log Probability Shape:   {list(log_prob.shape)}")
    print(
        f"Deterministic Action:    {list(deterministic_action.shape)} -> values: {deterministic_action.squeeze().tolist()}")
    print(f"Critic Q-Value Shape:    {list(q_val.shape)}")