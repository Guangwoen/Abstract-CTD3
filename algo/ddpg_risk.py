import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, cost):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, cost)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, cost = zip(*batch)
        return state, action, reward, next_state, done, cost

    def __len__(self):
        return len(self.buffer)


class DotProductSimilarity(nn.Module):

    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # a = a.flatten()
        cat = torch.cat([x, a], 1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
    
class Risk(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Risk, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(state_dim+action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        risk_value = self.fc(state_action)
        return risk_value


class DDPG_risk:
    def __init__(self, cfg, action_bound, device, writer=None):
        state_dim = cfg["model"]["observation_dim"]
        hidden_dim = cfg['model']['hidden_dim']
        action_dim = cfg["model"]["action_dim"]
        self.model_config = cfg['model']

        if device == '-1':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.action_dim = action_dim

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(self.device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(self.device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['trainer']['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['trainer']['critic_lr'])

        self.batch_size = cfg["dataloader"]["batch_size"]
        self.gamma = cfg['trainer']['gamma']
        self.sigma = cfg['trainer']['sigma']
        self.tau = cfg['trainer']['soft_tau']
        
        self.risk = Risk(state_dim, action_dim, hidden_dim).to(self.device)
        self.risk_target = Risk(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.risk_optimizer = torch.optim.Adam(self.risk.parameters(), lr=cfg['trainer']['critic_lr'])
        self.risk_target.load_state_dict(self.risk.state_dict())
        
        print(self.risk)

        self.memory = ReplayBuffer(cfg["dataloader"]["memory_capacity"])
        self.danger_scenario = ReplayBuffer(cfg["dataloader"]["memory_capacity"])

        self.writer = writer

        print(self.actor)
        print(self.critic)

    def choose_action(self, state):
        relative = DotProductSimilarity()
        state = torch.tensor(state.reshape(1, -1)).float().to(self.device)
        for data in self.danger_scenario.buffer:
            fal_state, fal_action, fal_reward, fal_next_state, is_done, exist_cost = data
            fal_state = torch.tensor(fal_state.reshape(1, -1)).float().to(self.device)
            relative_degree = relative(state, fal_state)
            if relative_degree.item() > 0.75:
                action = self.actor(state).cpu().data.numpy().flatten()  # [-1,1]
                action = np.clip(action, a_min=None, a_max=fal_action[0])
                return action
        return self.actor(state).cpu().data.numpy().flatten()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, num_iteration=1):
        
        if len(self.memory) < self.batch_size:
            return
        
        for i in range(num_iteration):
            
            states, actions, rewards, next_states, dones, cost = self.memory.sample(self.batch_size)
        
            states = torch.tensor(states, dtype=torch.float).to(self.device)
            actions = torch.tensor(actions, dtype=torch.float).view(-1, 1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
            
            cost = torch.tensor(cost, dtype=torch.float).view(-1, 1).to(self.device)
            
            next_action = self.target_actor(next_states)

            next_q_values = self.target_critic(next_states, next_action)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

            critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -torch.mean(self.critic(states, self.actor(states)))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)
            
            target_risk = self.risk_target(next_states, next_action)
            target_risk = cost + ((1 - dones) * self.gamma * target_risk).detach()
            current_risk = self.risk(states, actions)
            loss_risk = F.mse_loss(current_risk, target_risk)
            self.risk_optimizer.zero_grad()
            loss_risk.backward()
            self.risk_optimizer.step()

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.target_actor.state_dict(), os.path.join(path, "target_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.target_critic.state_dict(), os.path.join(path, "target_critic.pth"))
        print("model has been saved...")

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.target_actor.load_state_dict(torch.load(os.path.join(path, "target_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
        self.target_critic.load_state_dict(torch.load(os.path.join(path, "target_critic.pth")))
        print("model has been loaded...")
