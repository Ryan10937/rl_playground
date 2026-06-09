#!/usr/bin/env python
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
from utils import StepPairDataset, make_observation_reward_generator, to_numpy_obs, to_serializable_action,infer_action_info,collect_random_episode,save_random_episodes,run_episode,compute_returns,load_or_create_model

from actors import MLPPolicy, RNNPolicy





# ### Train Model

##################################################################################################################
##################################################################################################################
############################## Generate Pretraining Warmup Data ##################################################
##################################################################################################################
##################################################################################################################

# env = gym.make("CartPole-v1")
# env = gym.make("AntForaging-v0")
# env = gym.make("Pendulum-v1")
env = RubiksCube()
env.n_scramble = 5
# obs_size = env.observation_space.shape[0]
obs_size = 54*6 #rubiks


PATH = "random_episodes"          # folder to save warmup episodes
N_RANDOM_EPISODES = 10           # number of random episodes to collect
MAX_STEPS_PER_EPISODE = 10      # set an int if you want a hard cap
env.max_steps = MAX_STEPS_PER_EPISODE
num_warmup_epochs = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.999
LR = 1e-3
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 2
NUM_EPISODES = 100_000
run_id = 1


action_info = infer_action_info(env.action_space)

print("Action space info:", action_info)

if action_info["policy_kind"] == "discrete":
    n_actions = action_info["n_actions"]
elif action_info["policy_kind"] == "continuous_or_vector":
    n_actions = action_info["n_actions"]
elif action_info["policy_kind"] == "multidiscrete":
    n_actions = action_info["n_actions"]
else:
    raise ValueError(f"Unsupported action space: {env.action_space}")

# Collect random episodes before online training
save_random_episodes(
    env=env,
    path=PATH,
    n_episodes=N_RANDOM_EPISODES,
    max_steps=MAX_STEPS_PER_EPISODE,
)

# Reset once before training continues
env.reset()


##################################################################################################################
##################################################################################################################
####################################### Train on Warmup Data #####################################################
##################################################################################################################
##################################################################################################################

n_actions = env.action_space.n
print('obs_size:',obs_size)
print('env.action_space:',env.action_space)
print('n_actions:',n_actions)

save_path = f'models/rubiks_mlp_6-9_{run_id}.pth'
model = load_or_create_model(MLPPolicy,save_path,obs_size=obs_size, hidden_size=HIDDEN_SIZE, n_actions=n_actions)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20_000, gamma=0.1)

env.reset()

train_loader = make_observation_reward_generator(
    PATH,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print('here')

reward_history = []
loss_history = []
action_history_history = []
for epoch in range(num_warmup_epochs):
    for obs_batch, reward_batch in train_loader:
        obs_batch = obs_batch.to(DEVICE)          # [B, obs_dim]
        reward_batch = reward_batch.to(DEVICE)    # [B]
    
        # logits,pred_reward,_ = model(obs_batch).squeeze(-1)
        logits,pred_reward,_ = model(obs_batch)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        
        
        advantages = reward_batch - pred_reward.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(pred_reward, reward_batch)
    

        loss = policy_loss + 0.5 * value_loss 
        # loss = torch.nn.functional.mse_loss(pred_reward, reward_batch)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.detach())
        reward_history.append(reward_batch.detach().cpu().mean())

    plt.figure()
    plt.title(f'Loss per Episode: {epoch} | Avg Reward: {np.mean(reward_history)}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot([hist.item() for hist in loss_history])
    plt.savefig(f'graphs/latest_warmup_{run_id}.png')
    plt.close()
    torch.save(model.state_dict(), save_path)
    

assert 1==0



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


