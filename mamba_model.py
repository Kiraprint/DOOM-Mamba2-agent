from typing import Sequence

import torch
import torch.nn as nn
from mamba_ssm import Mamba2  # Required: pip install mamba-ssm
from torch.distributions.categorical import Categorical


class SSDMamba2Combatant(nn.Module):
    def __init__(self, action_dims: Sequence[int] = [3, 3, 3, 2], d_model=256, n_layers=3):
        super().__init__()
        self.action_dims = action_dims
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 9 * 9, d_model)
        )

        # Multiple Mamba layers with LayerNorm for stability
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # Multi-Discrete Heads
        self.actor_heads = nn.ModuleList([nn.Linear(d_model, dim) for dim in action_dims])
        self.critic = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, Stack, H, W, C)
        b, s, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b * s, c, h, w)
        tokens = self.cnn(x).view(b, s, -1)

        # Process through Mamba layers with residual connections
        for mamba, ln in zip(self.mamba_layers, self.layer_norms):
            tokens = tokens + mamba(ln(tokens))  # Pre-norm residual

        latent = tokens.mean(dim=1)

        logits = [head(latent) for head in self.actor_heads]
        return logits, self.critic(latent)

    def get_action_and_value(self, x, action=None):
        logits_list, value = self.forward(x)
        probs_list = [Categorical(logits=l) for l in logits_list]

        if action is None:
            action = torch.stack([p.sample() for p in probs_list], dim=-1)

        # Joint log_prob is the sum of log_probs of individual heads
        logprobs = torch.stack([p.log_prob(action[:, i]) for i, p in enumerate(probs_list)], dim=-1).sum(-1)
        entropy = torch.stack([p.entropy() for p in probs_list], dim=-1).sum(-1)

        return action, logprobs, entropy, value
