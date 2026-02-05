import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym
from tqdm import tqdm
from tqdm.notebook import trange
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
import os
import math
from walk_env_batch import WalkENV
import genesis as gs


os.makedirs("Checkpoints", exist_ok=True)



class Actor(nn.Module):

    def __init__(self, in_shape, out_shape):
        # print(in_shape, out_shape)
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_shape[0], 128)
        self.fc2 = nn.Linear(128, 256)
        self.mean = nn.Linear(256, out_shape[0])
        self.sd = nn.Linear(256, out_shape[0])

    def forward(self, x):
        # print(x.shape)/
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))

        mean = self.mean(x)
        log_sd = self.sd(x)
        sd = torch.exp(log_sd)     
           
        return mean, sd

    
    
class Critic(nn.Module):

    def __init__(self, in_shape):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)
 

class   PPOClip:
    
    def __init__(self, 
                env,
                eval_env,
                actor = None,
                critic = None,

                device = "cpu",
                rollouts = 2048,

                gamma = 0.9,
                gamma_gae = 0.99,
                _lambda = 0.5,
                batch_size = 32,
                epsilon = 0.1,
                beta = 0.001,
                actor_max_norm = 0.5,
                critic_max_norm = 0.5,
                epochs = 5
                ):
        
        self.env = env
        self.eval_env = eval_env
        # gs.init(backend=backend, logging_level = "warning")
        self.actor = Actor(env.observation_space.shape, env.action_space.shape)
        self.critic = Critic(env.observation_space.shape)

        self.device = device

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = 1e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = 1e-4)

        self.episode_rollouts = rollouts
        self.gamma = gamma
        self.gamma_gae = gamma_gae
        self._lambda = _lambda
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.beta = beta

        self.actor_max_norm = actor_max_norm
        self.critic_max_norm = critic_max_norm

        self.epochs = epochs

        self.writer = SummaryWriter('experiments/PPO_Clip1')


    def _sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(
            mean, std
        )
        
        action = dist.sample()
        # print(action)
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob
    

    def collect_rollout(self):

        states_buffer = []
        actions_buffer = []
        reward_buffer = []
        log_prob_buffer = []
        done_buffer = []
        current_rollout = 0

        
        episode_reward = 0
        
        state, _ = self.env.reset()
        for current_rollout in range(self.episode_rollouts):
            # state, info = self.env.reset()/
            # done = False
            # while not done:            
            action, log_prob = self._sample_action(state)
            next_state, reward, done, _ = self.env.step(action)
            # print(done)
            # done = ter or trun
            episode_reward += reward

            states_buffer.append(state)
            actions_buffer.append(action)
            reward_buffer.append(reward)
            log_prob_buffer.append(log_prob.detach().cpu().numpy())
            done_buffer.append(np.bitwise_not(done))

            

            state = next_state
            # if done:
            #     episode_reward = 0
            #     state, _ = self.env.reset()
            
        

        states_buffer = torch.tensor(np.array(states_buffer), dtype=torch.float).to(self.device)
        actions_buffer = torch.tensor(actions_buffer, dtype=torch.float).to(self.device)
        done_buffer = torch.tensor(done_buffer).to(self.device)
        reward_buffer = torch.tensor(reward_buffer).to(self.device)
        # print(type(log_prob_buffer))
        log_prob_buffer = torch.tensor(log_prob_buffer, dtype=torch.float).to(self.device)
        actions_buffer = torch.tensor(actions_buffer, dtype=torch.float).to(self.device)

        with torch.no_grad():
            current_state_value = self.critic(states_buffer).squeeze(-1)
            last_value = self.critic(torch.FloatTensor(state).to(self.device)).squeeze(1)
            

        # next_state_values = torch.zeros_like(current_state_value)
        # next_state_values[:-1] = current_state_value[1:]
        # next_state_values[done_buffer] = 0

        # print(states_buffer.shape, done_buffer.shape, log_prob_buffer.shape, actions_buffer.shape, reward_buffer.shape)
        # print(current_state_value.shape, next_state_values.shape)
        # td_error = reward_buffer + self.gamma * next_state_values - current_state_value

        advantage_buffer = torch.zeros_like(reward_buffer)
        total_return_buffer = torch.zeros_like(reward_buffer)
        gae = torch.zeros(self.batch_size, device=self.device)

        i = len(states_buffer) -1

        total_returns=0
        gae = 0

        # for t in reversed(range(self.n_steps)):
        #     if t == self.n_steps - 1:


        while i>=0:

            mask = done_buffer[i]

            if i == len(states_buffer) -1:
                next_val = last_value
            else:
                next_val = current_state_value[i + 1]

            delta = reward_buffer[i] + self.gamma * next_val * mask - current_state_value[i]
            gae = delta + self.gamma * self.gamma_gae * mask * gae
            ret = gae + current_state_value[i]

            # print(gae.shape)
            # print(mask.shape, next_val.shape, reward_buffer.shape, delta.shape, current_state_value[i].shape ,gae.shape, ret.shape)
            # print(f"\n\n\n{gae}")
            advantage_buffer[i] = gae
            total_return_buffer[i] = ret

            i -= 1


        advantage_buffer = (advantage_buffer - advantage_buffer.mean())/(advantage_buffer.std() + 1e-8)
        
        mainBuffer = list(zip(
            states_buffer.tolist(),
            actions_buffer.tolist(),
            log_prob_buffer.tolist(),
            total_return_buffer.tolist(),
            advantage_buffer.tolist(),
            current_state_value.tolist()
        ))


        return mainBuffer
    

    def get_reward__(self):


        state, info = self.eval_env.reset()
        total_reward = 0
        # while True:
        #     action = self.predict(state)
        #     state, reward, ter, trun, info = self.eval_env.step(action)
        #     state, _ = self.env.reset()
        #     total_reward = 0
        while True:
            action = self.predict(state)
            state, reward, dones, _ = self.eval_env.step(action)
            total_reward += reward
            # print(dones)
            if dones:
                return total_reward
            
    def predict(self, states):
        state = torch.tensor(states, dtype=torch.float).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(
            mean, std
        )
        action = dist.sample()
        return action.detach().cpu().numpy()
        
    def train(self, timesteps):
        current_timestep = 0
        while current_timestep <= timesteps:

            train_data = self.collect_rollout()
            np.random.shuffle(train_data)
            
            actor_loss_per_epoch = 0
            critic_loss_per_epoch = 0

            for epoch in range(self.epochs):
                actor_loss = 0
                critic_loss = 0
                batch_count = 0


                for idx in range(0, len(train_data), self.batch_size):
                    data = train_data[idx:idx + self.batch_size]
                    states, actions, log_probs, returns, gae, state_values = zip(*data)
                    # print(type(states), type(actions), type(log_probs), type(returns), type(gae))

                    states = torch.tensor(states, dtype=torch.float).to(self.device)
                    actions = torch.tensor(np.array(actions), dtype= torch.float).to(self.device)
                    log_probs = torch.tensor(log_probs, dtype=torch.float).to(self.device)
                    gae = torch.tensor(gae, dtype=torch.float).to(self.device)
                    returns = torch.tensor(returns, dtype=torch.float).to(self.device)
                    state_values = torch.tensor(state_values, dtype=torch.float).to(self.device)
                    # print(type(states), type(actions), type(log_probs), type(returns), type(gae))

                    mean, std = self.actor(states)
                    dist = torch.distributions.Normal(
                        mean, std
                    )
                    log_distribution = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    critic_output_state_values = self.critic(states).squeeze(-1)

                    # I divided log probs haha (DUMB...........)

                    importance_sampling_ratio = torch.exp(log_distribution.sum(dim=-1) - log_probs.sum(dim=-1))
                    # print(importance_sampling_ratio.shape, gae.shape)

                    clip1 = importance_sampling_ratio * gae
                    clip2 = torch.clip(importance_sampling_ratio, 1 - self.epsilon, 1 + self.epsilon) * gae
                    actor_loss = -torch.minimum(clip1, clip2).mean() - self.beta * entropy
                    # print(critic_output_state_values.shape, returns.shape)
                    critic_loss = nn.functional.mse_loss(critic_output_state_values, returns, reduction="mean")

                    actor_loss_per_epoch += actor_loss.item()
                    critic_loss_per_epoch += critic_loss.item()

                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=self.actor_max_norm
                    )
                    self.optimizer_actor.step()


                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=self.critic_max_norm
                    )
                    self.optimizer_critic.step()

                    batch_count += 1


                avg_actor_loss = actor_loss_per_epoch/batch_count
                avg_critic_loss = critic_loss_per_epoch/batch_count

                reward_after_current_epoch = self.get_reward__()
                self.writer.add_scalars(
                    main_tag="Training....",
                    tag_scalar_dict={
                        "Actor Loss" : avg_actor_loss,
                        "Critic Loss" : avg_critic_loss,
                        "Total Reward" : reward_after_current_epoch
                        },     
                    global_step=current_timestep * epoch + 1          
                )

            current_timestep += self.episode_rollouts
            print(f"\n\nActor Loss : {avg_actor_loss:.4f}\nCritic Loss : {avg_critic_loss:.4f}\nTotal Reward : {reward_after_current_epoch}\n\n\n")
            print(f"Total Timestep : {current_timestep}")
            self.save_models((current_timestep+1)//2048)

    def save_models(self, i):
        checkpoint = {
            "actor" : self.actor.state_dict(),
            "critic" : self.critic.state_dict()
        }
        torch.save(checkpoint, f"Checkpoints/checkpoint{i}.pth")

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    actor = Actor()
    critic = Critic()
    ppo = PPOClip(env, actor, critic, )
    ppo.train(500000)