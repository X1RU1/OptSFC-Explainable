import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

N_OBJ = 3

class DecomposedCritic(nn.Module):
    def __init__(self, obs_dim: int, n_obj: int = N_OBJ, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(n_obj)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.shared(obs)
        return torch.cat([h(feat) for h in self.heads], dim=-1)


class EUPGCriticTrainer:
    def __init__(self, critic, lr=3e-4, gamma=0.99,
                 buffer_size=10_000, batch_size=256):
        self.critic     = critic
        self.gamma      = gamma
        self.batch_size = batch_size
        self.optimizer  = optim.Adam(critic.parameters(), lr=lr)
        self.buffer     = deque(maxlen=buffer_size)

    def store(self, obs, reward_vec, next_obs, done):
        self.buffer.append((
            obs.astype(np.float32),
            reward_vec.astype(np.float32),
            next_obs.astype(np.float32),
            float(done),
        ))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        obs_b, rew_b, nobs_b, done_b = map(np.array, zip(*batch))

        obs_t  = torch.tensor(obs_b)
        rew_t  = torch.tensor(rew_b)
        nobs_t = torch.tensor(nobs_b)
        done_t = torch.tensor(done_b).unsqueeze(1)

        with torch.no_grad():
            v_next = self.critic(nobs_t)
            target = rew_t + self.gamma * v_next * (1 - done_t)

        v_pred = self.critic(obs_t)
        loss   = ((v_pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()