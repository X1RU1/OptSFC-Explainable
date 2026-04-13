import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class PPOQNet(nn.Module):
    """
    Scalar Q network for PPO.
    Estimates Q(s,a) for all actions simultaneously.
    Trained on PPO's own (s, a, r_scalar, s') transitions via TD,
    independent of PPO's policy training.
    Q(s,a) here is the scalarized action-value, consistent with
    PPO's single-objective reward signal.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            Q: (batch, n_actions)
        """
        return self.net(obs)


class PPOQTrainer:
    """
    Trains PPOQNet with scalar TD targets using PPO's own transitions.
    Does not modify PPO's policy training.

    TD target:
      target(s,a) = r + gamma * max_a' Q(s',a')
    """
    def __init__(self, q_net: PPOQNet, lr: float = 3e-4,
                 gamma: float = 0.99, buffer_size: int = 10_000,
                 batch_size: int = 256):
        self.q_net      = q_net
        self.gamma      = gamma
        self.batch_size = batch_size
        self.optimizer  = optim.Adam(q_net.parameters(), lr=lr)
        self.buffer     = deque(maxlen=buffer_size)

    def store(self, obs: np.ndarray, action: int,
              reward_scalar: float, next_obs: np.ndarray, done: bool):
        self.buffer.append((
            obs.astype(np.float32),
            int(action),
            float(reward_scalar),
            next_obs.astype(np.float32),
            float(done),
        ))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        obs_b, act_b, rew_b, nobs_b, done_b = map(np.array, zip(*batch))

        obs_t  = torch.tensor(obs_b)
        act_t  = torch.tensor(act_b, dtype=torch.long)
        rew_t  = torch.tensor(rew_b, dtype=torch.float32)  # (batch,)
        nobs_t = torch.tensor(nobs_b)
        done_t = torch.tensor(done_b, dtype=torch.float32)  # (batch,)

        with torch.no_grad():
            q_next  = self.q_net(nobs_t)              # (batch, n_actions)
            q_max   = q_next.max(dim=1).values         # (batch,)
            target  = rew_t + self.gamma * q_max * (1 - done_t)

        q_pred_all = self.q_net(obs_t)                # (batch, n_actions)
        q_pred     = q_pred_all[
            torch.arange(len(act_t)), act_t
        ]                                              # (batch,)

        loss = ((q_pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()