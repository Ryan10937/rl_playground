
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
class StepPairDataset(Dataset):
    def __init__(self, path, file_pattern="episode_*.pkl", obs_dtype=np.float32, reward_dtype=np.float32):
        self.path = Path(path)
        self.files = sorted(self.path.glob(file_pattern))
        self.obs_dtype = obs_dtype
        self.reward_dtype = reward_dtype

        if not self.files:
            raise FileNotFoundError(f"No episode files found in {self.path} matching {file_pattern}")

        self.index = []
        for file_idx, file_path in enumerate(self.files):
            with open(file_path, "rb") as f:
                episode = pickle.load(f)

            n_steps = len(episode["rewards"])
            for step_idx in range(n_steps):
                self.index.append((file_idx, step_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, step_idx = self.index[idx]
        file_path = self.files[file_idx]

        with open(file_path, "rb") as f:
            episode = pickle.load(f)

        obs = np.asarray(episode["observations"][step_idx], dtype=self.obs_dtype)
        reward = np.asarray(episode["rewards"][step_idx], dtype=self.reward_dtype)
        ep_return = np.asarray(episode["rewards"][step_idx], dtype=self.reward_dtype)

        obs = torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        ep_return = torch.as_tensor(ep_return, dtype=torch.float32)

        return obs, ep_return
def make_observation_reward_generator(
    path,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    file_pattern="episode_*.pkl",
):
    dataset = StepPairDataset(path=path, file_pattern=file_pattern)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader
    
def to_numpy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32)

def to_serializable_action(action):
    if isinstance(action, np.ndarray):
        return action.copy()
    if isinstance(action, (np.integer, np.floating)):
        return action.item()
    if isinstance(action, (list, tuple)):
        return np.asarray(action)
    return action


def infer_action_info(action_space):
    info = {"space_type": type(action_space).__name__}

    if hasattr(action_space, "n"):          # Discrete
        info["policy_kind"] = "discrete"
        info["n_actions"] = action_space.n
    elif hasattr(action_space, "nvec"):     # MultiDiscrete
        info["policy_kind"] = "multidiscrete"
        info["n_actions"] = tuple(np.asarray(action_space.nvec).tolist())
    elif hasattr(action_space, "shape"):    # Box / MultiBinary (fallback)
        info["policy_kind"] = "continuous_or_vector"
        info["n_actions"] = action_space.shape
    else:
        info["policy_kind"] = "unknown"
        info["n_actions"] = None

    return info


def collect_random_episode(env, episode_idx, max_steps=None, seed=None,GAMMA=0.999,reverse_episode=False):
    if seed is not None:
        if reverse_episode:
            original_scramble = env.n_scramble 
            env.n_scramble = 0
        reset_out = env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    else:
        if reverse_episode:
            original_scramble = env.n_scramble 
            env.n_scramble = 0
        reset_out = env.reset()

    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}

    observations = []
    actions = []
    rewards = []
    dones = []
    truncations = []
    infos = []

    total_reward = 0.0
    step_count = 0

    while True:
        observations.append(to_numpy_obs(obs))

        action = env.action_space.sample()
        actions.append(to_serializable_action(action))

        step_out = env.step(action)

        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, step_info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, step_info = step_out
            terminated = done
            truncated = False

        rewards.append(float(reward))
        dones.append(bool(terminated))
        truncations.append(bool(truncated))
        infos.append(step_info if step_info is not None else {})

        total_reward += float(reward)
        step_count += 1
        obs = next_obs

        if done:
            break
        if step_count >= max_steps:
            break
    if reverse_episode:
        observations=observations[::-1]
        actions=actions[::-1]
        rewards=rewards[::-1]
        dones=dones[::-1]
        truncations=truncations[::-1]
        infos=infos[::-1]

        env.n_scramble = original_scramble
    
    returns = compute_returns(rewards, GAMMA)

    episode = {
        "episode_idx": episode_idx,
        "total_reward": total_reward,
        "length": step_count,
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=object),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "returns": np.asarray(returns.cpu(), dtype=np.float32),
        "terminated": np.asarray(dones, dtype=np.bool_),
        "truncated": np.asarray(truncations, dtype=np.bool_),
        "infos": infos,
        "action_space_info": infer_action_info(env.action_space),
        "observation_space_shape": getattr(env.observation_space, "shape", None),
    }
    return episode


def save_random_episodes(env, path, n_episodes, max_steps=None, base_seed=1234,GAMMA=0.999,completed_count=100):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "n_episodes": n_episodes,
        "action_space": repr(env.action_space),
        "observation_space": repr(env.observation_space),
        "action_info": infer_action_info(env.action_space),
        "files": [],
    }

    for ep in range(1, n_episodes + 1):
        if ep < completed_count:
            episode = collect_random_episode(
                env,
                episode_idx=ep,
                max_steps=max_steps,
                seed=base_seed + ep,
                GAMMA = GAMMA,
                reverse_episode=True
            )
        else:
            episode = collect_random_episode(
                env,
                episode_idx=ep,
                max_steps=max_steps,
                seed=base_seed + ep,
                GAMMA = GAMMA,
                reverse_episode=False
                )

        file_path = path / f"episode_{ep:06d}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(episode, f)

        manifest["files"].append(str(file_path))
        print(
            f"[random] episode {ep:4d}/{n_episodes} | "
            f"len {episode['length']:4d} | reward {episode['total_reward']:8.3f}"
        )

    with open(path / "manifest.pkl", "wb") as f:
        pickle.dump(manifest, f)


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
        obs_t = torch.tensor(obs, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
        
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
    returns = torch.tensor(returns, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) #convert to tensor
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8) # normalize
    return returns
    
def load_or_create_model(model_class, some_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), *model_args, **model_kwargs):
    model = model_class(*model_args, **model_kwargs)
    path = Path(some_path)

    if path.exists():
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
