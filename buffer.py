import numpy as np
import torch


class SequenceReplayBuffer:
    def __init__(self, max_episodes=1000, horizon=16, burn_in=8):
        self.max_episodes = max_episodes
        self.horizon = horizon
        # FIX — burn-in length. We sample `burn_in + horizon` steps; the first
        # `burn_in` steps are used ONLY to warm up the recurrent (h, z) state
        # from a non-zero starting point, with no loss computed on them. This
        # prevents feeding h=0,z=0 to the GRU at an arbitrary mid-episode
        # timestep, which previously corrupted every downstream WM/SAC signal.
        self.burn_in = burn_in
        self.total_len = burn_in + horizon

        # Storage lists for full episodes
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def add_episode(self, obs_list, action_list, reward_list):
        """
        Appends a complete completed episode trajectory to the buffer.
        Expects lists or arrays of steps.
        """
        if len(self.obs_buffer) >= self.max_episodes:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)

        # Convert inputs to float32 arrays for safety
        self.obs_buffer.append(np.array(obs_list, dtype=np.float32))
        self.action_buffer.append(np.array(action_list, dtype=np.float32))
        self.reward_buffer.append(np.array(reward_list, dtype=np.float32))

    def sample(self, batch_size):
        """
        Samples a synchronized batch of continuous sequence blocks, each of
        length `burn_in + horizon`. The caller is responsible for using the
        first `burn_in` steps only to warm up recurrent state (no loss),
        and computing WM/SAC losses only on the trailing `horizon` steps.

        Returns:
            obs_batch:    (Batch, burn_in + Horizon, Obs_Dim)
            action_batch: (Batch, burn_in + Horizon, Action_Dim)
            reward_batch: (Batch, burn_in + Horizon, 1)
        """
        num_episodes = len(self.obs_buffer)
        if num_episodes == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        obs_seqs, action_seqs, reward_seqs = [], [], []

        # Guard against an infinite loop if NO episode in the buffer is long
        # enough yet (e.g. very early training). Caller should ensure episodes
        # exceed total_len before calling sample(); main.py's `len(buffer) < 5`
        # guard does not check episode length, so this safety valve avoids a
        # silent hang.
        max_attempts = batch_size * 200
        attempts = 0

        while len(obs_seqs) < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Could not find enough episodes with length > {self.total_len} "
                    f"(burn_in={self.burn_in} + horizon={self.horizon}). "
                    "Let the buffer accumulate longer episodes before sampling."
                )

            # 1. Pick a random episode from our history
            ep_idx = np.random.randint(0, num_episodes)
            ep_len = len(self.obs_buffer[ep_idx])

            # If the episode is too short for burn-in + lookback window, skip it
            if ep_len <= self.total_len:
                continue

            # 2. Pick a random valid starting point inside that episode timeline
            #    (start_idx can still be > 0 — that's fine now, because the
            #    burn_in steps will re-derive the correct (h, z) belief state
            #    before any loss-bearing step is reached)
            start_idx = np.random.randint(0, ep_len - self.total_len)
            end_idx = start_idx + self.total_len

            # 3. Slice out the synchronized timeline chunk
            obs_seqs.append(self.obs_buffer[ep_idx][start_idx:end_idx])
            action_seqs.append(self.action_buffer[ep_idx][start_idx:end_idx])
            reward_seqs.append(self.reward_buffer[ep_idx][start_idx:end_idx])

        # Convert arrays to PyTorch tensors
        obs_tensor = torch.tensor(np.array(obs_seqs))
        action_tensor = torch.tensor(np.array(action_seqs))
        reward_tensor = torch.tensor(np.array(reward_seqs)).unsqueeze(-1)  # Ensure (B, H, 1)

        return obs_tensor, action_tensor, reward_tensor

    def __len__(self):
        return len(self.obs_buffer)