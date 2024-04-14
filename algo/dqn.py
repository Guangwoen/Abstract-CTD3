import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state.flatten(), action, reward, next_state.flatten(), done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, cfg, state_dim, action_dim, device, writer=None):
        self.model_config = cfg['model']

        if device == '-1':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        hidden_dim = cfg['model']['hidden_dim']
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)  # Qnet
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(self.device)  # Target Qnet
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=float(cfg['trainer']['lr']))  # Adam optimizer
        self.gamma = cfg['trainer']['gamma']
        self.epsilon = cfg['trainer']['epsilon']
        self.epsilon_decay = cfg['trainer']['epsilon_decay']
        self.epsilon_min = cfg['trainer']['epsilon_min']
        self.update_every = cfg['trainer']['update_every']  # Update frequency
        self.count = 0  # Count updates

        print(self.q_net)

        self.memory = ReplayBuffer(cfg['dataloader']['memory_capacity'])
        self.writer = writer

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return action

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q value
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # Max Q values of next states
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD error

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.update_every == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )  # Update
        self.count += 1

    def save(self, path):
        torch.save(self.q_net.state_dict(), os.path.join(path, 'q_net.pth'))
        torch.save(self.target_q_net.state_dict(), os.path.join(path, 'target_q_net.pth'))
        print('model has been saved...')

    def load(self, path):
        self.q_net.load_state_dict(torch.load(os.path.join(path, 'q_net.pth')))
        self.target_q_net.load_state_dict(torch.load(os.path.join(path, 'target_q_net.pth')))
        print('model has been loaded...')
