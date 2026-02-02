import genesis as gs
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stand_ENV_batch import StandENV
from ppo_batch_train_code import PPOClip


env = StandENV(
    n_batch = 200
)

print(env.action_space.shape[0])
print(env.observation_space)

ppo = PPOClip(
    env=env,
    device="cuda",
    gamma=0.99,
    gamma_gae=0.95,
    batch_size=64,
    epochs=10,
    
)

ppo.train(10_000)