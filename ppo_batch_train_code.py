import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym
from tqdm.notebook import trange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os


os.makedirs("Checkpoints", exist_ok=True)



class Actor(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Actor, self).__init__()
        # print(in_shape, out_shape)
        self.net = nn.Sequential(
            nn.Linear(in_shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        # print(out_shape)
        self.mean = nn.Linear(128, out_shape[0])
        self.log_std = nn.Linear(128, out_shape[0])

    def forward(self, x):
        x = self.net(x)
        
        mean = self.mean(x)
        log_sd = self.log_std(x)
        

        log_sd = torch.clamp(log_sd, min=-20, max=2) 
        
        sd = torch.exp(log_sd)
        
        return mean, sd

    
    
class Critic(nn.Module):
    def __init__(self, in_shape):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class   PPOClip:
    
    def __init__(self, 
                env,
                n_batch = 1,
                actor = None,
                critic = None,

                device = "cpu",
                rollouts = 2048,

                gamma = 0.99,
                gamma_gae = 0.95,
                _lambda = 0.5,
                batch_size = 32,
                epsilon = 0.2,
                beta = 0.001,
                actor_max_norm = 1,
                critic_max_norm = 1,
                epochs = 5,
                previous_weights = None
                ):
        
        self.env = env
        self.n_batch = n_batch
        # gs.init(backend=backend, logging_level = "warning")
        self.actor = Actor(env.observation_space.shape, env.action_space.shape)
        self.critic = Critic(env.observation_space.shape)

        if previous_weights:
            self.actor.load_state_dict(previous_weights['actor'])
            self.critic.load_state_dict(previous_weights['critic'])

        self.device = device

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = 5e-4)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = 5e-4)

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

        self.writer = SummaryWriter('experiments/BAck')

    @torch.no_grad()
    def _sample_action(self, state):
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # print(state.shape)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(
            mean, std
        )
        
        action = dist.sample()
        # print(action.shape)
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
            # print(state.shape)
            action, log_prob = self._sample_action(state)
            next_state, reward, done, _ = self.env.step(action)

            episode_reward += reward

            states_buffer.append(state)
            actions_buffer.append(action)
            reward_buffer.append(reward)
            log_prob_buffer.append(log_prob.detach().cpu().numpy())
            done_buffer.append(1 - done.astype(int))

            state = next_state

        

        states_buffer = torch.tensor(np.array(states_buffer), dtype=torch.float).to(self.device)
        actions_buffer = torch.tensor(np.array(actions_buffer), dtype=torch.float).to(self.device)
        done_buffer = torch.tensor(np.array(done_buffer)).to(self.device)
        reward_buffer = torch.tensor(np.array(reward_buffer)).to(self.device)
        log_prob_buffer = torch.tensor(log_prob_buffer, dtype=torch.float).to(self.device)

        with torch.no_grad():
            current_state_value = self.critic(states_buffer).squeeze(-1)
            last_value = self.critic(torch.FloatTensor(state).to(self.device)).squeeze(1)
            

        advantage_buffer = torch.zeros_like(reward_buffer)
        total_return_buffer = torch.zeros_like(reward_buffer)
        i = len(states_buffer) -1
        gae = 0

        while i>=0:

            mask = done_buffer[i]

            if i == len(states_buffer) -1:
                next_val = last_value
            else:
                next_val = current_state_value[i + 1]

            delta = reward_buffer[i] + self.gamma * next_val * mask - current_state_value[i]
            gae = delta + self.gamma * self.gamma_gae * mask * gae
            ret = gae + current_state_value[i]

            advantage_buffer[i] = gae
            total_return_buffer[i] = ret

            i -= 1


        advantage_buffer = (advantage_buffer - advantage_buffer.mean())/(advantage_buffer.std() + 1e-8)
        # Returning tensors directly
        return states_buffer, actions_buffer, log_prob_buffer, total_return_buffer, advantage_buffer, current_state_value
        mainBuffer = list(zip(
            states_buffer.tolist(),
            actions_buffer.tolist(),
            log_prob_buffer.tolist(),
            total_return_buffer.tolist(),
            advantage_buffer.tolist(),
            current_state_value.tolist()
        ))


        return mainBuffer
    

    # def get_reward__(self):

    #     a = 0
    #     state, info = self.eval_env.reset()
    #     total_reward = 0
    #     total_dones = 0
    #     total_steps = 0
    #     # while True:
    #     #     action = self.predict(state)
    #     #     state, reward, ter, trun, info = self.eval_env.step(action)
    #     #     state, _ = self.env.reset()
    #     #     total_reward = 0
    #     while True:
    #         action = self.predict(state)
    #         state, reward, dones, _ = self.eval_env.step(action)
    #         total_reward += reward
    #         total_steps += 1
    #         if dones:
    #             total_dones += 1
    #             print(total_reward)
    #             total_steps = 0
    #             total_reward = 0
    #         if total_dones >= 2 and total_steps >= 200:
    #             return total_reward
    
    @torch.no_grad()      
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

            b_states, b_actions, b_log_probs, b_returns, b_advantage, b_state_values = self.collect_rollout()
            dataset_size = len(b_states)
            indices = torch.arange(dataset_size)
            

            for epoch in range(self.epochs):
                actor_loss_per_epoch = 0
                critic_loss_per_epoch = 0

                batch_count = 0

                for idx in range(0, dataset_size, self.batch_size):
                    batch_idx = indices[idx: idx + self.batch_size]
                    states = b_states[batch_idx]
                    actions = b_actions[batch_idx]
                    log_probs = b_log_probs[batch_idx]
                    gae = b_advantage[batch_idx]
                    returns = b_returns[batch_idx]
                    state_values = b_state_values[batch_idx]

                    mean, std = self.actor(states)
                    dist = torch.distributions.Normal(
                        mean, std
                    )
                    log_distribution = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    critic_output_state_values = self.critic(states).squeeze(-1)

                    # I divided log probs haha (DUMB...........)

                    importance_sampling_ratio = torch.exp(torch.clamp(log_distribution.sum(dim=-1) - log_probs.sum(dim=-1), -10, 10))
                    # print(importance_sampling_ratio.shape, gae.shape)

                    clip1 = importance_sampling_ratio * gae
                    clip2 = torch.clip(importance_sampling_ratio, 1 - self.epsilon, 1 + self.epsilon) * gae
                    actor_loss = -torch.minimum(clip1, clip2).mean() - self.beta * entropy
                    # print(critic_output_state_values.shape, returns.shape)
                    # print(returns, critic_output_state_values/)
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

            self.writer.add_scalar("Loss/Actor", avg_actor_loss, current_timestep)
            self.writer.add_scalar("Loss/Critic", avg_critic_loss, current_timestep)
            self.writer.add_scalar("Reward/Average Return From Critic", b_returns.mean().item(), current_timestep)
            self.writer.flush()
                
            
            current_timestep += self.episode_rollouts
            total_iterations = (current_timestep)//self.episode_rollouts
            print(f"\n\nActor Loss : {avg_actor_loss:.4f}\nCritic Loss : {avg_critic_loss:.4f}\nActual Return : {returns.mean()}\nPredicted Return :{critic_output_state_values.mean()}\n")
            print(f"Iteration Number : {total_iterations}\t Total Timesteps : {total_iterations * self.episode_rollouts}\t Totak Data Collected : {self.n_batch * total_iterations * self.episode_rollouts}")
            self.save_models(total_iterations)
            print(f"\n\n")
            print("==" * 50)

    def save_models(self, i):
        checkpoint = {
            "actor" : self.actor.state_dict(),
            "critic" : self.critic.state_dict()
        }
        torch.save(checkpoint, f"Checkpoints/back_{i}.pth")
        print(f"Saved checkpoint {i} !!!")


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
 
    ppo = PPOClip(env)
    ppo.train(500000)