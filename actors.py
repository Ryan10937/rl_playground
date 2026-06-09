from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import pickle
import random

import gymnasium as gym
from gymnasium import spaces
import ant_foraging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

from rubiks_env import RubiksCube


class RNNPolicy(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.input_proj = nn.Linear(obs_size, hidden_size,dtype=torch.float32)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True,num_layers=HIDDEN_LAYERS,dropout=0.1)
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        inpt = self.input_proj(x)
        x = torch.tanh(inpt)
        x = x.unsqueeze(1)              # [batch, 1, hidden]
        out, h = self.gru(x, h)         # out: [batch, 1, hidden]
        out = out.squeeze(1)            # [batch, hidden]
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)
        return logits, value, h

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.gru.num_layers, batch_size, HIDDEN_SIZE, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
class MLPPolicy(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden_size, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions, dtype=torch.float32)
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, x, h=None):
        x = self.backbone(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value, None

    def init_hidden(self, batch_size=1):
        return None