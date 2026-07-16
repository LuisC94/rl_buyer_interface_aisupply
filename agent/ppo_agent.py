import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

try:
    from agent.actor_critic_v2 import ActorMLP, CriticMLP
except ImportError:
    from actor_critic_v2 import ActorMLP, CriticMLP

class RunningStat:
    """ Welford's online algorithm """
    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.S = 0.0
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)

    @property
    def std(self):
        variance = self.S / (self.n - 1) if self.n > 1 else np.square(self.mean)
        return np.sqrt(variance)

class ParallelRolloutBuffer:
    """ Buffer adapted for parallel N-environments. """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ParallelPPOAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=30, eps_clip=0.2, batch_size=1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_action = max_action
        self.batch_size = batch_size
        
        self.buffer = ParallelRolloutBuffer()
        self.reward_scaler = RunningStat()

        self.policy_actor = ActorMLP(state_dim, action_dim, max_action).to(self.device)
        self.policy_critic = CriticMLP(state_dim).to(self.device)
        
        self.optimizer_actor = optim.Adam(self.policy_actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.policy_critic.parameters(), lr=lr_critic)

        self.policy_old_actor = ActorMLP(state_dim, action_dim, max_action).to(self.device)
        self.policy_old_critic = CriticMLP(state_dim).to(self.device)
        self.policy_old_actor.load_state_dict(self.policy_actor.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action_batched(self, states_matrix):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states_matrix).to(self.device)
            
            action_mean_percent, log_std = self.policy_old_actor(state_tensor)
            
            clamped_log_std = torch.clamp(log_std, min=-2.3, max=1.5)
            std_tensor = torch.exp(clamped_log_std)
            
            dist = Normal(action_mean_percent, std_tensor)
            action_percent = dist.sample()
            
            action_logprob = dist.log_prob(action_percent)
            
            physical_action = torch.round(torch.clamp(action_percent * self.max_action, 0, self.max_action))
            
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action_percent)
        self.buffer.logprobs.append(action_logprob)
        
        return physical_action.cpu().numpy().flatten()

    def evaluate(self, state, action_percent):
        action_mean_percent, log_std = self.policy_actor(state)
        
        clamped_log_std = torch.clamp(log_std, min=-2.3, max=1.5)
        std_tensor = torch.exp(clamped_log_std)
        
        dist = Normal(action_mean_percent, std_tensor)
        
        action_logprobs = dist.log_prob(action_percent)
        dist_entropy = dist.entropy()
        
        state_values = self.policy_critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def update(self):
        all_states_tensor = torch.stack(self.buffer.states, dim=0).to(self.device)
        
        with torch.no_grad():
            all_state_values = self.policy_old_critic(all_states_tensor).squeeze(-1) 
            
        rewards = []
        discounted_reward = np.zeros(len(self.buffer.rewards[0])) # vector of size NUM_ENVS
        
        if len(self.buffer.states) == len(self.buffer.rewards) + 1:
            discounted_reward = all_state_values[-1].detach().cpu().numpy()
        
        for step_t in reversed(range(len(self.buffer.rewards))):
            reward_t = np.array(self.buffer.rewards[step_t])
            is_terminal_t = np.array(self.buffer.is_terminals[step_t])
            
            discounted_reward[is_terminal_t] = 0
            
            discounted_reward = reward_t + (self.gamma * discounted_reward)
            
            for val in discounted_reward:
                self.reward_scaler.push(val)
                
            normalized_r = (discounted_reward - self.reward_scaler.mean) / (self.reward_scaler.std + 1e-8)
            clipped_r = np.clip(normalized_r, -3.0, 3.0)
            
            rewards.insert(0, clipped_r)
            
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)

        T = len(self.buffer.rewards)
        old_states = all_states_tensor[:T].view(-1, all_states_tensor.size(-1)).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).view(-1, 1).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).view(-1, 1).detach()
        
        rewards_flat = rewards_tensor.view(-1, 1)
        old_state_values_flat = all_state_values[:T].view(-1, 1).detach()

        advantages = rewards_flat - old_state_values_flat
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        buffer_size = len(old_states)
        actual_batch_size = min(self.batch_size, buffer_size)
        
        epoch_total_loss = 0.0
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        num_updates = 0
        
        for _ in range(self.K_epochs):
            indices = torch.randperm(buffer_size).to(self.device)
            
            for start in range(0, buffer_size, actual_batch_size):
                end = start + actual_batch_size
                batch_indices = indices[start:end]
                
                b_old_states = old_states[batch_indices]
                b_old_actions = old_actions[batch_indices]
                b_old_logprobs = old_logprobs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_rewards_flat = rewards_flat[batch_indices]

                logprobs, state_values, dist_entropy = self.evaluate(b_old_states, b_old_actions)

                ratios = torch.exp(logprobs - b_old_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = 0.5 * self.MseLoss(state_values, b_rewards_flat)
                entropy_bonus = 0.05 * dist_entropy
                
                loss = actor_loss + critic_loss - entropy_bonus

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.mean().backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                epoch_total_loss += loss.mean().item()
                epoch_actor_loss += actor_loss.mean().item()
                epoch_critic_loss += critic_loss.mean().item()
                num_updates += 1
            
        self.policy_old_actor.load_state_dict(self.policy_actor.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())

        self.buffer.clear()
        
        mean_total = epoch_total_loss / num_updates if num_updates > 0 else 0.0
        mean_actor = epoch_actor_loss / num_updates if num_updates > 0 else 0.0
        mean_critic = epoch_critic_loss / num_updates if num_updates > 0 else 0.0
        return mean_total, mean_actor, mean_critic
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old_actor.state_dict(), checkpoint_path + '_actor.pth')
        torch.save(self.policy_old_critic.state_dict(), checkpoint_path + '_critic.pth')
        
        scaler_state = {
            'n': self.reward_scaler.n,
            'mean': self.reward_scaler.mean,
            'S': self.reward_scaler.S
        }
        torch.save(scaler_state, checkpoint_path + '_scaler.pth')
        
    def load(self, checkpoint_path):
        import os
        self.policy_actor.load_state_dict(torch.load(checkpoint_path + '_actor.pth', map_location=self.device, weights_only=False))
        self.policy_critic.load_state_dict(torch.load(checkpoint_path + '_critic.pth', map_location=self.device, weights_only=False))
        
        self.policy_old_actor.load_state_dict(self.policy_actor.state_dict())
        self.policy_old_critic.load_state_dict(self.policy_critic.state_dict())
        
        scaler_path = checkpoint_path + '_scaler.pth'
        if os.path.exists(scaler_path):
            scaler_state = torch.load(scaler_path, map_location='cpu', weights_only=False)
            self.reward_scaler.n = scaler_state['n']
            self.reward_scaler.mean = scaler_state['mean']
            self.reward_scaler.S = scaler_state['S']
