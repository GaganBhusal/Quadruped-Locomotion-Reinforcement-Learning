
import torch.nn as nn
import torch
from tqdm.notebook import trange
from walk_env_batch import WalkENV
from ppo_batch_train_code import Actor, Critic
import genesis as gs

i = 25

gs.init(backend=gs.gpu, logging_level="warning")

env = WalkENV(render=True, backend=gs.gpu, n_batch=1)

actor = Actor(env.observation_space.shape, env.action_space.shape)
actor.load_state_dict(torch.load(f"Checkpoints/checkpoint{i}.pth")["actor"])
device = "cuda"

obs, _ = env.reset()
for i in range(5):
    state, info = env.reset()
    total_reward = 0
    while True:
        state = torch.tensor(state, dtype=torch.float).to(device)
        mean, std = actor(state)
        dist = torch.distributions.Normal(
            mean, std
        )
        action = dist.sample()
        action = action.detach().cpu().numpy()
        state, reward, dones, _ = env.step(action)
        total_reward += reward
        if dones:
            print(f"The total reward is {total_reward}")