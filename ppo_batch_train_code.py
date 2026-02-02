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


os.makedirs("Checkpoints", exist_ok=True)



class Actor(nn.Module):

    def __init__(self, in_shape, out_shape):
        print(in_shape, out_shape)
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_shape[0], 128)
        self.fc2 = nn.Linear(128, 256)
        self.mean = nn.Linear(256, out_shape[0])
        self.sd = nn.Linear(256, out_shape[0])

    def forward(self, x):
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
 

class PPOClip:
    
    def __init__(self, 
                env,
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
        print(action)
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
            next_state, reward, ter, trun, _ = self.env.step(action)
            done = ter or trun
            episode_reward += reward

            states_buffer.append(state)
            actions_buffer.append(np.array(action, dtype=np.float64))
            reward_buffer.append(reward)
            log_prob_buffer.append(log_prob)
            done_buffer.append(int(not(done)))

            

            state = next_state
            if done:
                episode_reward = 0
                state, _ = self.env.reset()
            
        

        states_buffer = torch.tensor(np.array(states_buffer), dtype=torch.float).to(self.device)
        done_buffer = torch.tensor(done_buffer).to(self.device)
        reward_buffer = torch.tensor(reward_buffer).to(self.device)
        log_prob_buffer = torch.tensor(log_prob_buffer, dtype=torch.float).to(self.device)
        actions_buffer = torch.tensor(actions_buffer, dtype=torch.float).to(self.device)

        with torch.no_grad():
            current_state_value = self.critic(states_buffer).squeeze(1)
            

        next_state_values = torch.zeros_like(current_state_value)
        next_state_values[:-1] = current_state_value[1:]
        next_state_values[done_buffer] = 0


        td_error = reward_buffer+ self.gamma * next_state_values - current_state_value

        advantage_buffer = torch.zeros_like(reward_buffer)
        total_return_buffer = torch.zeros_like(reward_buffer)

        i = len(states_buffer) -1

        total_returns=0
        gae = 0


        while i>=0:

            if done_buffer[i] == 0:
                total_returns = 0
                gae = 0

            total_returns = reward_buffer[i] + self.gamma * total_returns
            gae = td_error[i] + self._lambda * self.gamma_gae * gae

            total_return_buffer[i] = total_returns
            advantage_buffer[i] = gae

            i -= 1


        advantage_buffer = (advantage_buffer - advantage_buffer.mean())/(advantage_buffer.std() + 1e-8)
        
        mainBuffer = list(zip(
            states_buffer.tolist(),
            actions_buffer,
            log_prob_buffer,
            total_return_buffer,
            advantage_buffer,
            current_state_value
        ))


        return mainBuffer
    

    def get_reward__(self):
        state, _ = self.env.reset()
        total_reward = 0
        while True:
            action = self.predict(state)
            state, reward, ter, trun, _ = self.env.step(action)
            total_reward += reward
            if ter or trun:
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

                    states = torch.tensor(states, dtype=torch.float).to(self.device)
                    actions = torch.tensor(actions).to(self.device)
                    log_probs = torch.tensor(log_probs, dtype=torch.float).to(self.device)
                    gae = torch.tensor(gae, dtype=torch.float).to(self.device)
                    returns = torch.tensor(returns, dtype=torch.float).to(self.device)
                    state_values = torch.tensor(state_values, dtype=torch.float).to(self.device)

                    mean, std = self.actor(states)
                    dist = torch.distributions.Normal(
                        mean, std
                    )
                    log_distribution = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    critic_output_state_values = self.critic(states).squeeze(1)

                    # I divided log probs haha (DUMB...........)
                    importance_sampling_ratio = torch.exp(log_distribution - log_probs)

                    clip1 = importance_sampling_ratio * gae
                    clip2 = torch.clip(importance_sampling_ratio, 1 - self.epsilon, 1 + self.epsilon) * gae
                    actor_loss = -torch.minimum(clip1, clip2).mean() - self.beta * entropy
                
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
            print(f"Total Timestep : {current_timestep}")



if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    actor = Actor()
    critic = Critic()
    ppo = PPOClip(env, actor, critic, )
    ppo.train(500000)