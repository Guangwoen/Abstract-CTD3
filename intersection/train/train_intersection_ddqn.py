import logging
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



class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        # super class
        super(Net, self).__init__()
        # hidden nodes define
        hidden_nodes1 = 1024
        hidden_nodes2 = 512
        self.fc1 = nn.Linear(state_dim, hidden_nodes1)
        self.fc2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.fc3 = nn.Linear(hidden_nodes2, action_dim)
 
    def forward(self, state):
        # define forward pass of the actor
        x = state # state
        # Relu function double
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    


class Replay: # learning
    def __init__(self,
                 buffer_size, init_length, state_dim, action_dim, env):
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
 
        self._storage = []
        self._init_buffer(init_length)
    def _init_buffer(self, n):
        # choose n samples state taken from random actions
        state = self.env.reset()
        state = state.flatten()
        state = state[-10:]
        for i in range(n):
            action = self.env.action_space.sample()
            observation, reward, done, truncated, info = self.env.step(action)
            observation = observation.flatten()
            observation = observation[-10:]
            # gym.env.step(action): tuple (obversation, reward, terminated, truncated, info) can edit
            # observation: numpy array [location]
            # reward: reward for *action
            # terminated: bool whether end
            # truncated: bool whether overflow (from done)
            # info: help/log/information
            if type(state) == type((1,)):
                state = state[0]
            # if state is tuple (ndarray[[],[],...,[]],{"speed":Float,"cashed":Bool,"action":Int,"reward":dict,"agent-reward":Float[],"agent-done":Bool}),we take its first item
            # because after run env.reset(), the state stores the environmental data and it can not be edited
            # we only need the state data -- the first ndarray
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": observation,
                "done": done,
            }
            self._storage.append(exp)
            state = observation
 
            if done:
                state = self.env.reset()
                state = state.flatten()
                state = state[-10:]
                done = False
 
    def buffer_add(self, exp):
        # exp buffer: {exp}=={
        #                 "state": state,
        #                 "action": action,
        #                 "reward": reward,
        #                 "state_next": observation,
        #                 "done": terminated,}
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)  # remove the last one in dict
 
    def buffer_sample(self, n):
        # random n samples from exp buffer
        return random.sample(self._storage, n)

class DOUBLEDQN(nn.Module):
    def __init__(
        self,
            env, # gym environment
            state_dim, # state size
            action_dim, # action size
        lr = 0.001, # learning rate
        gamma = 0.99, # discount factor
        batch_size = 5, # batch size for each training
        timestamp = "",):
        # super class
        super(DOUBLEDQN, self).__init__()
        self.env = env
        self.env.reset()
        self.timestamp = timestamp
        # for evaluation purpose
        self.test_env = copy.deepcopy(env)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.is_rend = True

        self.cnt = 0
        self.epsilon = 0
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.6

        self.target_net = Net(self.state_dim, self.action_dim).to(device)#TODO
        self.estimate_net = Net(self.state_dim, self.action_dim).to(device)#TODO
        self.ReplayBuffer = Replay(1000, 100, self.state_dim, self.action_dim, env)#TODO
        self.optimizer = torch.optim.Adam(self.estimate_net.parameters(), lr=lr)
 
    def choose_our_action(self, state):
        # greedy strategy for choosing action
        # state: ndarray environment state
        # epsilon: float in [0,1]
        # return: action we chosen
        # turn to 1D float tensor -> [[a1,a2,a3,...,an]]
        # we have to increase the speed of transformation ndarray to tensor if not it will spend a long time to train the model
        # ndarray[[ndarray],...[ndarray]] => list[[ndarray],...[ndarray]] => ndarray[...] => tensor[...]
        if type(state) == type((1,)):
            state = state[0]
        temp = [exp for exp in state]
        target = []
        target = np.array(target)
        # n dimension to 1 dimension ndarray
        for i in temp:
            target = np.append(target,i)
        state = torch.FloatTensor(target).to(device)
        # randn() return a set of samples which are Gaussian distribution
        # no argments -> return a float number
        if np.random.randn() > self.epsilon:
            # when random number smaller than epsilon: do these things
            # put a state array into estimate net to obtain their value array
            # choose max values in value array -> obtain action
            action_value = self.estimate_net(state)
            action = torch.argmax(action_value).item()
        else:
            # when random number bigger than epsilon: randomly choose a action
            action = np.random.randint(0, self.action_dim)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.cnt += 1
 
        return action
 
    def train(self, num_episode):
        # num_eposide: total turn number for train
        loss_list = [] # loss set
        avg_reward_list = [] # reward set
        step = 0
        rend = 0
        np.set_printoptions(suppress=True, linewidth=2000)
        # tqdm : a model for showing process bar
        for episode in tqdm(range(1,int(num_episode)+1)):
            episode_reward = 0
            done = False
            truncated = False
            state = self.env.reset()
            state = state.flatten()
            state = state[-10:]
            each_loss = 0
            if type(state) == type((1,)):
                state = state[0]
            while not done and not truncated:
                if self.is_rend:
                    self.env.render()
                step +=1
                action = self.choose_our_action(state)
                observation, reward, done, truncated, info = self.env.step(action)

                observation = observation.flatten()
                observation = observation[-10:]

                is_crash = 1 if info["crashed"] else 0
                cost = 100 if is_crash else 0

                logging.info('%s, %s, %s, %s, %s, %s, %s', episode, state, action, reward, observation, done, cost)

                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "state_next": observation,
                    "done": done,
                }
                self.ReplayBuffer.buffer_add(exp)
                state = observation
 
                # sample random batch in replay memory
                exp_batch = self.ReplayBuffer.buffer_sample(self.batch_size)
                # extract batch data
                action_batch = torch.LongTensor(
                    [exp["action"] for exp in exp_batch]
                ).to(device)
                reward_batch = torch.FloatTensor(
                    [exp["reward"] for exp in exp_batch]
                ).to(device)
                done_batch = torch.FloatTensor(
                    [1 - exp["done"] for exp in exp_batch]
                ).to(device)
                # Slow method -> Fast method when having more data
                state_next_temp = [exp["state_next"] for exp in exp_batch]
                state_temp = [exp["state"] for exp in exp_batch]
                state_temp_list = np.array(state_temp)
                state_next_temp_list = np.array(state_next_temp)
 
                state_next_batch = torch.FloatTensor(state_next_temp_list).to(device)
                state_batch = torch.FloatTensor(state_temp_list).to(device)
 
                episode_reward += reward

                # reshape
                state_batch = state_batch.reshape(self.batch_size, -1)
                action_batch = action_batch.reshape(self.batch_size, -1)
                reward_batch = reward_batch.reshape(self.batch_size, -1)
                state_next_batch = state_next_batch.reshape(self.batch_size, -1)
                done_batch = done_batch.reshape(self.batch_size, -1)
 
                # obtain estimate Q value gather(dim, index) dim==1:column index
                estimate_Q_value = self.estimate_net(state_batch).gather(1, action_batch)
                # obtain target Q value detach:remove the matched element
                max_action_index = self.estimate_net(state_next_batch).detach().argmax(1)
                target_Q_value = reward_batch + done_batch * self.gamma * self.target_net(
                    state_next_batch
                ).gather(1, max_action_index.unsqueeze(1))# squeeze(1) n*1->1*1, unsqueeze(1) 1*1->n*1
 
                # mse_loss: mean loss
                loss = F.mse_loss(estimate_Q_value, target_Q_value)
                each_loss += loss.item()
 
                # update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
 
                # update target network
                # load parameters into model
                if self.learn_step_counter % 10 == 0:
                    self.target_net.load_state_dict(self.estimate_net.state_dict())
                self.learn_step_counter +=1
 
            
 
            # # you can update these variables
            # if episode_reward % 100 == 0:
            #     rend += 1
            #     if rend % 5 == 0:
            #         self.is_rend = True
            #     else:
            #         self.is_rend = False
            # save
            period = 1
            if episode % period == 0:
                each_loss /= step
                episode_reward /= period
                avg_reward_list.append(episode_reward)
                loss_list.append(each_loss)
                logging.info('Step:%s, Episode Num:%s, Ave Reward:%s', step, episode, episode_reward)
                
 

 
        self.env.close()
        return loss_list, avg_reward_list


    def save(self, path):
        torch.save(self.target_net.state_dict(), os.path.join(path, 'target_net.pth'))
        torch.save(self.estimate_net.state_dict(), os.path.join(path, 'estimate_net.pth'))
        print('model has been saved...')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(os.path.join(path, 'target_net.pth')))
        self.estimate_net.load_state_dict(torch.load(os.path.join(path, 'estimate_net.pth')))
        print('model has been loaded...')
 
    # def eval(self):
    #     # evaluate the policy
    #     count = 0
    #     total_reward = 0
    #     done = False
    #     state = self.test_env.reset()
    #     state = state.flatten()
    #     state = state[-12:]
    #     if type(state) == type((1,)):
    #         state = state[0]
     
    #     while not done:
    #         action = self.choose_our_action(state)
    #         observation, reward, done, truncated, info = self.test_env.step(action)
    #         observation = observation.flatten()
    #         observation = observation[-12:]
    #         total_reward += reward
    #         count += 1
    #         state = observation
 
    #     return total_reward, count

if __name__ == "__main__":
    # timestamp
    # Define the environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler('./my_log_ICA.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logging.info('episode x1 y1 vx1 vy1 cos_h1 x2 y2 vx2 vy2 cos_h2 action reward\
                  next_x1 next_y1 next_vx1 next_vy1 next_cos_h1 next_x2 next_y2 next_vx2 next_vy2 next_cos_h2 done cost')

    env = gym.make("intersection-v0")
    # details
    env.config["duration"] = 10
    env.config["observation"]["vehicles_count"] = 2
    env.config["observation"]["features"] = ['x', 'y', 'vx', 'vy', 'cos_h']
    env.config["observation"]["absolute"] = False
    env.config["observation"]["normalize"] = True
    env.config["vehicles_density"] = 1.3
    env.config["reward_speed_range"] = [7.0, 10.0]
    env.config["initial_vehicle_count"] = 1
    env.config["simulation_frequency"] = 10
    env.config["policy_frequency"] = 5
    env.config["arrived_reward"] = 2
    env.config["spawn_probability"] = 0
    print(env.config)
    env.reset()
    # for i in range(100):
    #     state = env.reset()
    #     print(state)
    
    # create a doubledqn object
    double_dqn_object = DOUBLEDQN(
        env,
        state_dim=10,
        action_dim=3,
        lr=0.001,
        gamma=0.99,
        batch_size=32
    )
    # your chosen train times
    iteration = 500
    # start training
    avg_loss, avg_reward_list = double_dqn_object.train(iteration)

    double_dqn_object.save("./intersection/checkpoints")

    episodes_list = list(range(len(avg_reward_list)))
    plt.plot(episodes_list, avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig("./fuck2.png") 
