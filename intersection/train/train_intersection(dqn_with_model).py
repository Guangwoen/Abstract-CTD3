import sys
sys.path.append('/home/akihi/data1/Abstract-CTD3-main-master')

import os
import random
import math
import time

import logging
import gym
import numpy as np
import joblib
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import SpatioTemporalKMeans

from algo import *
import utils
import highway_env


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--project', type=str,
                        help='the path to save project',
                        default='project/20240109-DQN-intersection-with-model')
    parser.add_argument('--model_config', type=str,
                        help='path of rl algorithm configuration',
                        default='conf/algorithm/DQN_intersection.yaml')
    parser.add_argument('--env_config', type=str,
                        help='path of highway env',
                        default='conf/env/highway_intersection.yaml')
    parser.add_argument('--model', type=str, help='the path of model to use', 
                        default='data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/spatio_temporal_elbow_Kmeans.pkl')
    parser.add_argument('--weight', type=str, help='the weight of agent', 
                        default='0.5')
    parser.add_argument('--gpu', type=str,
                        help='[0, 1, 2, 3 | -1] the id of gpu to train/test, -1 means using cpu',
                        default='-1')
    args = parser.parse_args()
    return args


def build_env(env_name, configure):
    env = gym.make(env_name, render_mode='human')
    env.configure(configure)
    env.reset()

    return env


def build_agent(config, gpu, writer):
    if config['algo_name'] == 'DQN':
        agent = DQN(config, gpu, writer=writer)

    return agent


def get_reward(vehicle):
    
    reward = 0.

    speed = vehicle.speed
    distance_to_goal = np.linalg.norm(vehicle.position - vehicle.destination)  # 离目标的距离
    is_collision = vehicle.crashed

    reward += 0.05 * speed - 0.0005 * distance_to_goal

    return reward


def choose_action_with_model(state, agent, eval_config, mdl, centers, agent_weight, model_weight):
     agent_choice = agent.choose_action(np.array(state))
     
     state = np.array(state, dtype=float).reshape(1, -1)
     label = mdl.predict(state)[0]
     state_predict = centers[label]
     
     model_choice = agent.choose_action(np.array(state_predict))
     
     return round(agent_weight * agent_choice + model_weight * model_choice)
     


def train(env, agent, env_config, algo_config, eval_config, mdl, mdl_name, weight, project_dir, writer):

    episode_reward_list = list()  # episode 累计奖励

    episodes = algo_config['trainer']['episodes']
    minimal_size = algo_config['trainer']['minimal_size']

    batch_size = algo_config['dataloader']['batch_size']
    
    reward_history = list()
    
    # 获取中心
    if mdl_name == 'spatio_temporal':
        centers = mdl.get_centers()
    else:
        centers = mdl.cluster_centers_
    
    # 网络和模型权重
    
    agent_weight = float(weight)
    model_weight = 1 - agent_weight
    
    steps = list()
    values = list()
    
    print(agent_weight, model_weight, mdl_name)

    for k_episode in range(episodes):
        state = env.reset(seed=1024)

        episode_reward = 0
        episode_time_steps = 0

        done, truncated = False, False
        while not done and not truncated:
            episode_time_steps += 1
            
            # action = agent.choose_action(state.flatten())
            action = choose_action_with_model(state.flatten(), agent, eval_config, mdl, centers, agent_weight, model_weight)
            next_state, reward, done, truncated, info = env.step(action)
            
            reward = get_reward(env.vehicle)
            
            ego_state = state[0]
            ego_next_state = next_state[0]
            
            # logging.info('%s, [%s, %s, %s, %s], %s, [%s, %s, %s, %s]', 
            #              k_episode, 
            #              ego_state[0], ego_state[1], math.sqrt(ego_state[2]**2 + ego_state[3]**2), ego_state[4], 
            #              action, 
            #              reward, 
            #              ego_next_state[0], ego_next_state[1], math.sqrt(ego_next_state[2]**2 + ego_next_state[3]**2), ego_next_state[4], 
            #              done)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            env.render()

            if len(agent.memory) > minimal_size:
                states, actions, rewards, next_states, dones = agent.memory.sample(batch_size)
                agent.update(states, actions, rewards, next_states, dones)

        reward_history.append(episode_reward)
        
        avg_reward = np.mean(reward_history[-100:])

        logging.info('%s, %s, %s, %s', k_episode, episode_time_steps, episode_reward, avg_reward)

        episode_reward_list.append(episode_reward)

        writer.add_scalar('Reward/avg_reward', avg_reward, global_step=k_episode)
        writer.add_scalar('Reward/epoch_reward', episode_reward, global_step=k_episode)
        
        steps.append(k_episode)
        values.append(avg_reward)

    save_dir = os.path.join(project_dir, 'checkpoint')
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)
    
    # 保存数据
    data_save_dir = 'intersection/data/run_with_model/{}/{}-{}/data'.format(mdl_name, agent_weight, model_weight)
    plot_save_dir = 'intersection/data/run_with_model/{}/{}-{}/plot'.format(mdl_name, agent_weight, model_weight)
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)
    
    format_time = time.strftime("%Y%m%d%H%M", time.localtime())
    utils.save_csv_on_models({ 'Step': steps, 'Value': values }, data_save_dir + '/{}.csv'.format(format_time))
    utils.single_plot(steps, values, plot_save_dir + '/{}.png'.format(format_time), xlabel='episodes', ylabel='return')

    return episode_reward_list


def main():
    config = parse_args()
    utils.make_dir(config.project, config.model_config, config.env_config)

    tensorboard_dir = os.path.join(config.project, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)

    env_config = utils.load_yml(config.env_config)
    env = build_env(env_config['env_name'], env_config['config'])

    algo_config = utils.load_yml(config.model_config)
    # agent = build_agent(algo_config, config.gpu, writer)
    agent = DQN(algo_config, env.observation_space.shape[0]*algo_config['model']['observation_dim'],
                env.action_space.n, config.gpu, writer)
    
    """Model configs"""
    
    # 加载一阶段参数
    eval_config_path = "conf/eval/highway_ica_eval.yaml"
    eval_config = utils.load_yml(eval_config_path)
    
    # 加载聚类模型
    mdl_path = config.model
    mdl = joblib.load(mdl_path)
    mdl_name = utils.get_model_name_from_path(mdl_path)

    episode_reward_list = train(env, agent, env_config, algo_config, eval_config, mdl, mdl_name, config.weight, config.project, writer)

    np.save(os.path.join(config.project, 'results', 'train_epoch_return.npy'), episode_reward_list)

    utils.single_plot(None, episode_reward_list,
                      path=os.path.join(config.project, 'plots', 'train_epoch_return'),
                      xlabel='Epoch', ylabel='Return', title='', label=algo_config['algo_name'])

    env.close()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For OpenMP problem in OSX
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler('intersection/data/trained_intersection_data.log'),
            logging.StreamHandler()
        ],
    )
    logging.info('episode steps episode_reward avg_reward')
    main()
