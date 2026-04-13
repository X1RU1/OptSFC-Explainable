import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

N_OBJ = 3


class DecomposedQNet(nn.Module):
    """
    Per-objective Q network: estimates Q(s, a) for all actions simultaneously.
    Output shape: (batch, n_actions, N_OBJ)
    Q[b, a, i] = expected discounted return of objective i
                 when taking action a in state obs[b],
                 following the current EUPG policy thereafter.
    Trained on EUPG's own (s, a, r_vec, s') transitions via TD,
    so Q values reflect EUPG's policy, not any other algorithm's.
    """
    def __init__(self, obs_dim: int, n_actions: int,
                 n_obj: int = N_OBJ, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Each head outputs Q values for all actions on one objective
        self.heads = nn.ModuleList([
            nn.Linear(hidden, n_actions) for _ in range(n_obj)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
        Returns:
            Q_vec: (batch, n_actions, N_OBJ)
        """
        feat = self.shared(obs)
        per_obj = [h(feat) for h in self.heads]      # N_OBJ x (batch, n_actions)
        return torch.stack(per_obj, dim=2)            # (batch, n_actions, N_OBJ)


class DecomposedQTrainer:
    """
    Trains DecomposedQNet with per-objective TD targets using EUPG's
    own environment transitions. Does not modify EUPG's policy training.

    TD target for state s, action a, objective i:
      target(s,a,i) = r_i + gamma * Q(s', a*', i)
      where a*' = argmax_a' [ Q(s', a') @ weights ]
    This uses the scalarized-greedy next action, consistent with
    EUPG's scalarized optimization objective.
    """
    def __init__(self, q_net: DecomposedQNet, weights: np.ndarray,
                 lr: float = 3e-4, gamma: float = 0.99,
                 buffer_size: int = 10_000, batch_size: int = 256):
        self.q_net      = q_net
        self.weights    = torch.tensor(weights, dtype=torch.float32)
        self.gamma      = gamma
        self.batch_size = batch_size
        self.optimizer  = optim.Adam(q_net.parameters(), lr=lr)
        self.buffer     = deque(maxlen=buffer_size)

    def store(self, obs: np.ndarray, action: int,
              reward_vec: np.ndarray, next_obs: np.ndarray, done: bool):
        self.buffer.append((
            obs.astype(np.float32),
            int(action),
            reward_vec.astype(np.float32),
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
        rew_t  = torch.tensor(rew_b)                        # (batch, N_OBJ)
        nobs_t = torch.tensor(nobs_b)
        done_t = torch.tensor(done_b, dtype=torch.float32).unsqueeze(1)  # (batch, 1)

        with torch.no_grad():
            q_next = self.q_net(nobs_t)                     # (batch, n_actions, N_OBJ)
            scalar_next = q_next @ self.weights             # (batch, n_actions)
            a_next = scalar_next.argmax(dim=1)              # (batch,)
            q_next_best = q_next[
                torch.arange(len(a_next)), a_next, :
            ]                                               # (batch, N_OBJ)
            target = rew_t + self.gamma * q_next_best * (1 - done_t)

        q_pred_all = self.q_net(obs_t)                      # (batch, n_actions, N_OBJ)
        q_pred = q_pred_all[
            torch.arange(len(act_t)), act_t, :
        ]                                                    # (batch, N_OBJ)

        loss = ((q_pred - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()