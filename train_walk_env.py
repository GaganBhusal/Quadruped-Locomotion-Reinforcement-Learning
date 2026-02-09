import genesis as gs
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
from walk_env_batch import WalkENV
from ppo_batch_train_code import PPOClip

gs.init(backend=gs.gpu, logging_level="warning")



# previous_model = torch.load("Checkpoints/checkpoint1100.pth")
N_BATCH = 3000
ROLLOUTS = 96000//N_BATCH

env = WalkENV(
    n_batch =N_BATCH, 
    render=False
)

eval_env = WalkENV(
    n_batch = 1, 
    render=False
)

# print(env.action_space.shape[0])
# print(env.observation_space)

ppo = PPOClip(
    env=env,
    eval_env=eval_env,
    n_batch = N_BATCH,
    rollouts=ROLLOUTS,
    device="cuda",
    gamma=0.99,
    gamma_gae=0.95,
    batch_size=64,
    epochs=10,
    # previous_weights=previous_model
)

ppo.train(1000_0000)
