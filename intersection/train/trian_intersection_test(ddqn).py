import sys
sys.path.append('/home/akihi/data1/Abstract-CTD3-main-master')

import os
import copy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import highway_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the environment
env = gym.make("intersection-v0")
env.reset()


class Replay:
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
        """
        A function to initialize the replay buffer.

        : param init_length: int, initial number of transitions to collect
        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        : param env: gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        self._storage = []
        self._init_buffer(init_length)

    def _init_buffer(self, n):
        """
        Init buffer with n samples with state-transitions taken from random actions

        : param n: int, number of samples
        """
        state = self.env.reset()
        for _ in range(n):
            action = self.env.action_space.sample()
            state_next, reward, done, truncated, _ = self.env.step(action)
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self._storage.append(exp)
            state = state_next

            if done:
                state = self.env.reset()
                done = False

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        : param exp: a dictionary consisting of state, action, reward , next state and done flag
        """
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        : param N: int, number of samples to obtain from the buffer
        """
        return random.sample(self._storage, N)


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network

        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        """
        super(Net, self).__init__()

        hidden_nodes1 = 1024
        hidden_nodes2 = 512
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)

    def forward(self, state):
        """
        Define the forward pass of the actor

        : param state: ndarray, the state of the environment
        """
        x = state

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DOUBLEDQN(nn.Module):
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        batch_size=5,
        timestamp="",
    ):
        """
        : param env: object, a gym environment
        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        : param lr: float, learning rate
        : param gamma: float, discount factor
        : param batch_size: int, batch size for training
        """
        super(DOUBLEDQN, self).__init__()

        self.env = env
        self.env.reset()
        self.timestamp = timestamp

        self.test_env = copy.deepcopy(env)  # for evaluation purpose
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_step_counter = 0

        self.target_net = Net(self.state_dim, self.action_dim).to(device)
        self.estimate_net = Net(self.state_dim, self.action_dim).to(device)
        self.ReplayBuffer = Replay(1000, 100, self.state_dim, self.action_dim, env)

        self.optimizer = torch.optim.Adam(self.estimate_net.parameters(), lr=lr)

    def choose_action(self, state, epsilon=0.9):
        """
        Select action using epsilon greedy method

        : param state: ndarray, the state of the environment
        : param epsilon: float, between 0 and 1
        : return: ndarray, chosen action
        """
        state = torch.FloatTensor(state).to(device).reshape(-1)  # get a 1D array
        if np.random.randn() <= epsilon:
            action_value = self.estimate_net(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def train(self, num_epochs):
        """
        Train the policy for the given number of iterations

        :param num_epochs: int, number of epochs to train the policy for
        """
        loss_list = []
        avg_reward_list = []
        epoch_reward = 0

        for epoch in tqdm(range(int(num_epochs))):
            done = False
            state = self.env.reset()
            avg_loss = 0
            step = 0
            while not done:
                step += 1
                action = self.choose_action(state)
                state_next, reward, done, truncated, _ = self.env.step(action)
                # self.env.render()
                # store experience to replay memory
                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "state_next": state_next,
                    "done": done,
                }
                self.ReplayBuffer.buffer_add(exp)
                state = state_next

                # sample random batch from replay memory
                exp_batch = self.ReplayBuffer.buffer_sample(self.batch_size)

                # extract batch data
                state_batch = torch.FloatTensor([exp["state"] for exp in exp_batch]).to(
                    device
                )
                action_batch = torch.LongTensor(
                    [exp["action"] for exp in exp_batch]
                ).to(device)
                reward_batch = torch.FloatTensor(
                    [exp["reward"] for exp in exp_batch]
                ).to(device)
                state_next_batch = torch.FloatTensor(
                    [exp["state_next"] for exp in exp_batch]
                ).to(device)
                done_batch = torch.FloatTensor(
                    [1 - exp["done"] for exp in exp_batch]
                ).to(device)

                # reshape
                state_batch = state_batch.reshape(self.batch_size, -1)
                action_batch = action_batch.reshape(self.batch_size, -1)
                reward_batch = reward_batch.reshape(self.batch_size, -1)
                state_next_batch = state_next_batch.reshape(self.batch_size, -1)
                done_batch = done_batch.reshape(self.batch_size, -1)

                # get estimate Q value
                estimate_Q = self.estimate_net(state_batch).gather(1, action_batch)

                # get target Q value
                max_action_idx = self.estimate_net(state_next_batch).detach().argmax(1)
                target_Q = reward_batch + done_batch * self.gamma * self.target_net(
                    state_next_batch
                ).gather(1, max_action_idx.unsqueeze(1))

                # compute mse loss
                loss = F.mse_loss(estimate_Q, target_Q)
                avg_loss += loss.item()

                # update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update target network
                if self.learn_step_counter % 10 == 0:
                    self.target_net.load_state_dict(self.estimate_net.state_dict())
                self.learn_step_counter += 1

            reward, count = self.eval()
            epoch_reward += reward

            # save
            period = 40
            if epoch % period == 0:
                # log
                avg_loss /= step
                epoch_reward /= period
                avg_reward_list.append(epoch_reward)
                loss_list.append(avg_loss)

                print(
                    "\nepoch: [{}/{}], \tavg loss: {:.4f}, \tavg reward: {:.3f}, \tsteps: {}".format(
                        epoch + 1, num_epochs, avg_loss, epoch_reward, count
                    )
                )

                epoch_reward = 0
                # create a new directory for saving
                save_path = 'intersection/{}'.format(self.timestamp)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
                np.save(save_path + "/double_dqn_loss.npy", loss_list)
                np.save(save_path + "/double_dqn_avg_reward.npy", avg_reward_list)
                torch.save(
                    self.estimate_net.state_dict(), save_path + "/double_dqn.pkl"
                )

        self.env.close()
        return loss_list, avg_reward_list

    def eval(self):
        """
        Evaluate the policy
        """
        count = 0
        total_reward = 0
        done = False
        state = self.test_env.reset()

        while not done:
            action = self.choose_action(state, epsilon=1)
            state_next, reward, done, truncated, _ = self.test_env.step(action)
            total_reward += reward
            count += 1
            state = state_next

        return total_reward, count


if __name__ == "__main__":

    # timestamp for saving
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime(
        "%m%d_%H_%M", named_tuple
    )  # have a folder of "date+time ex: 1209_20_36 -> December 12th, 20:36"

    double_dqn_object = DOUBLEDQN(
        env,
        state_dim=105,
        action_dim=3,
        lr=0.001,
        gamma=0.99,
        batch_size=64,
        timestamp=time_string,
    )

    # Train the policy
    iterations = 4000
    avg_loss, avg_reward_list = double_dqn_object.train(iterations)
    save_path = 'intersection/{}'.format(time_string)
    np.save(save_path + "/double_dqn_loss.npy", avg_loss)
    np.save(save_path + "/double_dqn_avg_reward.npy", avg_reward_list)

    # save the dqn network
    torch.save(
        double_dqn_object.estimate_net.state_dict(), save_path + "/double_dqn.pkl"
    )

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(avg_loss)
    plt.grid()
    plt.title("Double DQN Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(save_path + "double_dqn_loss.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward_list)
    plt.grid()
    plt.title("Double DQN Training Reward")
    plt.xlabel("*40 epochs")
    plt.ylabel("reward")
    plt.savefig(save_path + "/double_dqn_train_reward.png", dpi=150)
    plt.show()