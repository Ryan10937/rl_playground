#!/usr/bin/env python
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

from pathlib import Path
import torch


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
import pandas as pd

import ant_foraging

# In[3]:




# ## Model

# In[8]:

####Here
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
        return torch.zeros(self.gru.num_layers, batch_size, HIDDEN_SIZE, device=DEVICE)

# ### Train Model

# In[9]:


def run_episode(env, model, episode_number, save_dir=None, save_format="csv"):
    """
    Run a single episode and build a pandas DataFrame with one row per step.

    Columns:
      - episode
      - step
      - action
      - reward
      - terminated   (bool from env.step)
      - truncated    (bool from env.step)
      - solved       (bool: terminated & success condition)
      - next_obs_*   (flattened next observation)
    """
    action_hist = np.array([1e-8] * env.action_space.n)
    action_int_hist = []

    obs,_ = env.reset()
    h = model.init_hidden(batch_size=1)

    log_probs = []
    values = []
    rewards = []
    rows = []

    done = False
    total_reward = 0.0
    step_idx = 0

    while not done:
        obs = obs.flatten()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        #get action
        logits, value, h = model(obs_t, h)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        next_obs_arr = np.array(next_obs, copy=True)
        next_obs_flat = next_obs_arr.ravel()

        log_probs.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)
        action_hist[action] += 1
        action_int_hist.append(action.item())

        row = {
            "episode": int(episode_number),
            "step": int(step_idx),
            "action": int(action.item()),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        for i, v in enumerate(next_obs_flat):
            row[f"next_obs_{i}"] = float(v)
        rows.append(row)
        obs = next_obs
        total_reward += reward
        step_idx += 1

    df = pd.DataFrame(rows)

    if save_dir is not None and terminated:
        save_path = Path(save_dir).expanduser()
        save_path.mkdir(parents=True, exist_ok=True)

        if save_format == "csv":
            df.to_csv(save_path / f"episode_{episode_number:06d}.csv", index=False)
        elif save_format == "parquet":
            df.to_parquet(save_path / f"episode_{episode_number:06d}.parquet", index=False)
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")

    return log_probs, values, rewards, total_reward, action_hist,action_int_hist, df

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R # add discounted future rewards to current reward step
        returns.append(R) # log reward
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE) #convert to tensor
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8) # normalize
    return returns
    
def load_or_create_model(model_class, some_path, device="cuda", *model_args, **model_kwargs):
    model = model_class(*model_args, **model_kwargs)
    path = Path(some_path)

    if path.exists():
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model



# In[10]:

env = gym.make("CartPole-v1")
# env = gym.make("AntForaging-v0")
# env = gym.make("Pendulum-v1")

obs_size = env.observation_space.shape
# obs_size = (49)
print(env.action_space)
# n_actions = env.action_space.n
n_actions = (1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.999
LR = 1e-3
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 2
NUM_EPISODES = 100_000
run_id = 1


save_path = f'models/forager_rnn_6-5_{run_id}.pth'
model = load_or_create_model(RNNPolicy,save_path,obs_size=obs_size, hidden_size=HIDDEN_SIZE, n_actions=n_actions)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20_000, gamma=0.1)
####Here
# In[15]:
env.reset()

reward_history = []
loss_history = []
action_history_history = []
model.train()

for episode in range(1, NUM_EPISODES + 1):
    log_probs, values, rewards, total_reward,action_history,_,_ = run_episode(env, model,episode_number=episode)
    returns = compute_returns(rewards, GAMMA)

    log_probs = torch.cat(log_probs)
    values = torch.cat(values)

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    policy_loss = -(log_probs * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    

    loss = policy_loss + 0.5 * value_loss 
    loss_history.append(loss)

    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    # scheduler.step()
    
    reward_history.append(total_reward)
    action_history_history.append(action_history)
    if episode % 1000 == 0:
        avg_reward = np.mean(reward_history)
        print(f"Episode {episode:4d} | avg reward {avg_reward:7.2f}")

        plt.figure()
        plt.subplot(1,2,1)
        plt.title(f'Loss per Episode: {episode} | Avg Reward {avg_reward:7.2f}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.plot(np.linspace(0,len(loss_history)),[hist.cpu().detach().numpy() for hist in loss_history])
        plt.plot([hist.item() for hist in loss_history])
        plt.subplot(1,2,2)
        plt.title('Action Frequency')
        plt.ylabel('count')
        plt.xlabel('action')
        plt.bar(np.arange(len(action_history_history[0])),sum(np.array(action_history_history)))
        
        
        plt.savefig(f'graphs/latest{run_id}.png')
        plt.close()
        torch.save(model.state_dict(), save_path)

        reward_history = []
        loss_history = []
        action_history_history = []


        env.close()


