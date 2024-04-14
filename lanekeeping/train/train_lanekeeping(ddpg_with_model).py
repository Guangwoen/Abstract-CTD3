import sys
sys.path.append('/home/akihi/data1/Abstract-CTD3-main-master')

import os
import tqdm
import logging
import math
import time

import joblib
import numpy as np
from argparse import ArgumentParser

import gym
import highway_env

import torch
from tensorboardX import SummaryWriter
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import SpatioTemporalKMeans

import utils

from algo import *
import time

from algo.td3_risk import TD3_risk
from algo.td3_risk_disturbance import TD3_risk_disturbance


safe_distance = 0.8


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, help="the path to save project", 
                        
                        default="project/20240105-DDPG_lanekeeping-with-model")
    parser.add_argument("--model_config", type=str, help="path of rl algorithm configuration", 
                        
                        default="conf/algorithm/DDPG_lanekeeping.yaml")
    
    parser.add_argument("--env_config", type=str, help="path of highway env", 
                        default="conf/env/highway_lane_keeping_continuous_steering.yaml")
    parser.add_argument("--model", type=str, help="the path of model to use", 
                        default="data_analysis/mdp/second_stage_abstract/kmeans_model/lka_td3_risk/raw_elbow_Kmeans.pkl")
    parser.add_argument("--weight", type=str, help="the weight of agent", 
                        default="0.5")
    parser.add_argument("--gpu", type=str, help="[0,1,2,3 | -1] the id of gpu to train/test, -1 means using cpu", 
                        default="-1")
    
    args = parser.parse_args()
    return args


def build_env(env_name, configure):
    """build the highway environment"""
    
    env = gym.make(env_name, render_mode='human')
    env.configure(configure)
    env.reset()
    
    
    return env


def build_agent(config, env, gpu, writer):
    """choose reinforcement learning algorithm"""
    if config["algo_name"] == "TD3":
        agent = TD3(config, gpu, writer=writer)
    elif config["algo_name"] == "TD3_risk":
        agent = TD3_risk(config, gpu, writer=writer)
    elif config["algo_name"] == "TD3_risk_disturbance":
        agent = TD3_risk_disturbance(config, gpu, writer=writer)
    elif config["algo_name"] == "DDPG":
        agent = DDPG(config, env.action_space.high[1], gpu, writer=writer)
    
    return agent

def get_reward(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your reward function here! """
    
    """PCPO中的方案"""


    y, vy, cos_h = state[0]

    lateral_distance = abs(y)
    lane_vehicle_angle = math.acos(cos_h)
    
    reward = 1 - (lateral_distance**2) - (lane_vehicle_angle**2)
    
    return reward

def get_cost(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your cost function here! """
    cost = 0
    if done and not info["crashed"] and not truncated:
        cost = 100
    return cost


def choose_action_with_model(state, agent, eval_config, mdl, centers, agent_weight, model_weight):
    
    # Agent's choice based on neural network model
    agent_choice = agent.choose_action(np.array(state))

    # Model's choice based on mdp
    state = np.array(state, dtype=float).reshape(1, -1)
    label = mdl.predict(state)[0]
    state_predict = centers[label]

    model_choice = agent.choose_action(np.array(state_predict))
    print(agent_choice, model_choice)

    return agent_weight * agent_choice + model_weight * model_choice


def eval_policy(eval_config, mdl, centers, agent_weight, model_weight, agent, eval_env, env_config, algo_config, eval_episodes=10, seed=1024):
    """evalurate the agent during training"""
    avg_reward = 0.
    num_outoflane = 0 
    
    for _ in range(eval_episodes):
        state = eval_env.reset()
        state = utils.normalize_observation(state, 
                                    env_config["config"]["observation"]["features"], 
                                    env_config["config"]["observation"]["features_range"], 
                                    clip=False)
        state = state.flatten()
        
        done, truncated = False, False
        while not done and not truncated:
            
            # action = agent.choose_action(np.array(state))
            action = choose_action_with_model(np.array(state), agent, eval_config, mdl, centers, agent_weight, model_weight)
            
            if algo_config["model"]["action_dim"] == 1: 
                real_action = [algo_config["model"]["action_config"]["acceleration"], action[0]]
                real_action = np.array(real_action).clip(eval_env.action_space.low, eval_env.action_space.high)
            
            
            state, reward, done, truncated, info = eval_env.step(real_action)
            if done and not info["crashed"] and not truncated: 
                num_outoflane += 1
            
            
            reward = get_reward(info, state, action, done, truncated)
            avg_reward += reward
            
            state = utils.normalize_observation(state, 
                                    env_config["config"]["observation"]["features"], 
                                    env_config["config"]["observation"]["features_range"], 
                                    clip=False)
            state = state.flatten()

            
    avg_reward /= eval_episodes
    
    return avg_reward, num_outoflane


def train(env, eval_env, agent, max_timesteps, max_updates, print_freq, save_freq, 
          env_config, algo_config, eval_config, mdl, mdl_name, weight, project_dir, writer):
    """train the agent"""


    episode_outoflane_list = list() 
    episode_reward_list = list()
    episode_step_list = list()
    
    relative_distance_of_center_line = list()
    
    episode_is_too_far_list = list()
    
    average_time_per_step = 0.
    num_steps = 0
    
    
    ave_reward_list = list()
    num_outoflane_list = list() 
    eval_step_list = list()
    
    state = env.reset(seed=1024) 
    state = utils.normalize_observation(state, 
                            env_config["config"]["observation"]["features"], 
                            env_config["config"]["observation"]["features_range"], 
                            clip=False)
    state = state.flatten()
    
    episode_reward = 0
    episode_timesteps = 0 
    episode_num = 0
    episode_is_too_far = 0
    
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
    outoflanes = list()
    distances = list()
    
    for t_step in range(max_timesteps):
        t_step += 1 
        episode_timesteps += 1
        
        if algo_config["render"]: env.render()
        
        """Select action according to policy"""
        beg_time = time.time()
        action = choose_action_with_model(state, agent, eval_config, mdl, centers, agent_weight, model_weight)
        # action = agent.choose_action(state) 
        end_time = time.time()
        average_time_per_step += (end_time-beg_time)
        num_steps += 1
        
        
        if "exploration_noise" in algo_config["trainer"]:
            noise = np.random.normal(0, algo_config["trainer"]["exploration_noise"], 
                                            size=algo_config["model"]["action_dim"])
            noise = noise.clip(-algo_config["trainer"]["noise_clip"], algo_config["trainer"]["noise_clip"])
            action = action + noise 
            
        if algo_config["model"]["action_dim"] == 1: 
            real_action = [algo_config["model"]["action_config"]["acceleration"], action[0]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        """Perform action"""
        next_state, reward, done, truncated, info = env.step(real_action)
        
        """Define the reward composition"""
        reward = get_reward(info, next_state, action, done, truncated)
        
        """Define the cost"""
        cost = get_cost(info, next_state, action, done, truncated)
        is_outoflane = 1 if done and not info["crashed"] and not truncated else 0
            
        relative_distance = abs(next_state[0][0])
        is_too_far = 1 if relative_distance > safe_distance else 0
        relative_distance_of_center_line.append(relative_distance)
        writer.add_scalar('Distance/step_distance', relative_distance, global_step=t_step)
        
        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state, 
                                env_config["config"]["observation"]["features"], 
                                env_config["config"]["observation"]["features_range"], 
                                clip=False)
        next_state = next_state.flatten()
        
        if "risk" in algo_config["algo_name"]:
            agent.memory.push(state, action, reward, next_state, done, cost)
            logging.info('%s, %s, %s, %s, %s, %s, %s', episode_num, state, action, reward, next_state, done, cost)
        else:
            agent.memory.push(state, action, reward, next_state, done)
            logging.info('%s, %s, %s, %s, %s, %s', episode_num, state, action, reward, next_state, done)

        """Train agent"""
        agent.update(num_iteration=max_updates) 
        state = next_state 
        
        """some information"""
        episode_reward += reward
        episode_is_too_far += is_too_far
        
        if done or truncated:
            episode_num += 1
            
            episode_outoflane_list.append(is_outoflane)
            episode_reward_list.append(episode_reward)
            episode_step_list.append(t_step)
            episode_is_too_far_list.append(episode_is_too_far)
            
            writer.add_scalar('Reward/epoch_reward', episode_reward, global_step=episode_num)
            writer.add_scalar('OutOfLanes/count', is_outoflane, global_step=episode_num)
            writer.add_scalar('relative_distance/too_far_count', episode_is_too_far, global_step=episode_num)
            
            steps.append(episode_num)
            values.append(episode_reward)
            outoflanes.append(sum(episode_outoflane_list))
            distances.append(episode_is_too_far)
            
            # now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
            # print(f"[{now_time}][Train] Step: {t_step}, Episode: {episode_num},",
            #       f"Episode Steps: {episode_timesteps}, Reward: {episode_reward:.2f}",
            #       f"OutOfLane Num: {sum(episode_outoflane_list)}")
            logging.info('驶出道路：Step:%s, Episode: %s, Episode Step：%s, Reward：%s, crash Num:%s', t_step, episode_num,
                         episode_timesteps, episode_reward, sum(episode_outoflane_list))

            episode_reward = 0
            episode_timesteps = 0
            episode_is_too_far = 0
            is_outoflane = 0
            
            state = env.reset(seed=1024) 
            state = utils.normalize_observation(state, 
                                    env_config["config"]["observation"]["features"], 
                                    env_config["config"]["observation"]["features_range"], 
                                    clip=False)
            state = state.flatten()
        
        if t_step % algo_config["trainer"]["eval_freq"] == 0:
            with torch.no_grad():
                eval_episodes = algo_config["trainer"]["eval_episodes"]
                ave_reward, num_outoflane = eval_policy(eval_config, mdl, centers, agent_weight, model_weight, agent, 
                                                        eval_env, env_config, algo_config, eval_episodes=eval_episodes)

                ave_reward_list.append(ave_reward)
                num_outoflane_list.append(num_outoflane)
                eval_step_list.append(t_step)
                # now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
                # print(f"[{now_time}][Eval] Step: {t_step}, Episode Num: {eval_episodes},",
                #     f"Ave Reward: {ave_reward:.2f}",
                #     f"OutOfLane Num: {num_outoflane}")

                logging.info('Step:%s, Episode Num:%s, Ave Reward:%s, Out of Line Num:%s', t_step, eval_episodes,
                             ave_reward, num_outoflane)

    
    save_dir = os.path.join(project_dir, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)
    
    # 保存数据
    data_save_dir = 'lanekeeping/data/risk/ddpg/with_model_data/data'
    plot_save_dir = 'lanekeeping/data/risk/ddpg/with_model_data/plot'
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)
    
    format_time = time.strftime("%Y%m%d%H%M", time.localtime())
    utils.save_csv_on_models({ 'Step': steps, 'Value': values, 'Outoflane': outoflanes, 'Distance': distances }, data_save_dir + '/{}.csv'.format(format_time))
    utils.single_plot(steps, values, plot_save_dir + '/{}.png'.format(format_time), xlabel="episodes", ylabel="return")
    utils.single_plot(steps, outoflanes, plot_save_dir + '/outoflane_{}.png'.format(format_time), xlabel="episodes", ylabel="out of lane")
    utils.single_plot(steps, distances, plot_save_dir + '/distance_{}.png'.format(format_time), xlabel="episodes", ylabel="distances")

    # print(f"Average decision time per step: {average_time_per_step/num_steps} s")
    logging.info('Average decision time per step: %s', average_time_per_step / num_steps)

    return episode_reward_list, episode_outoflane_list, episode_step_list, relative_distance_of_center_line


def main():
    config = parse_args()
    utils.make_dir(config.project, config.model_config, config.env_config)
    
    
    tensorboard_dir = os.path.join(config.project, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    
    
    env_config = utils.load_yml(config.env_config)
    env = build_env(env_config["env_name"], env_config["config"])
    
    eval_env = build_env(env_config["env_name"], env_config["config"])

    
    algo_config = utils.load_yml(config.model_config)
    agent = build_agent(algo_config, env, config.gpu, writer)
    
    """Model configs"""
    
    # 加载一阶段参数
    eval_config_path = "conf/eval/highway_lka_eval.yaml"
    eval_config = utils.load_yml(eval_config_path)
    
    # 加载聚类模型
    mdl_path = config.model
    mdl = joblib.load(mdl_path)
    mdl_name = utils.get_model_name_from_path(mdl_path)
    
    episode_reward_list, episode_outoflane_list, episode_step_list, relative_distance_of_center_line = train(env, eval_env, agent, 
                                                                                                             algo_config["trainer"]["max_timesteps"], 
                                                                                                             algo_config["trainer"]["max_updates"], 
                                                                                                             algo_config["trainer"]["print_freq"], 
                                                                                                             algo_config["trainer"]["save_freq"], 
                                                                                                             env_config, 
                                                                                                             algo_config, 
                                                                                                             eval_config, 
                                                                                                             mdl, 
                                                                                                             mdl_name, 
                                                                                                             config.weight, 
                                                                                                             config.project, 
                                                                                                             writer)
    
    
    
    np.save(os.path.join(config.project, "results", "train_epoch_return.npy"), episode_reward_list)
    np.save(os.path.join(config.project, "results", "train_epoch_outoflane.npy"), episode_outoflane_list)
    np.save(os.path.join(config.project, "results", "train_epoch_step.npy"), episode_step_list)
    np.save(os.path.join(config.project, "results", "train_step_distance.npy"), relative_distance_of_center_line)
    
    
    utils.single_plot(episode_step_list, episode_reward_list, 
                      path=os.path.join(config.project, "plots", "train_step_return"), 
                      xlabel="Step", ylabel="Return", title="", label=algo_config["algo_name"])
    utils.single_plot(None, episode_reward_list, 
                      path=os.path.join(config.project, "plots", "train_epoch_return"), 
                      xlabel="Epoch", ylabel="Return", title="", label=algo_config["algo_name"])
    
    utils.single_plot(episode_step_list, episode_outoflane_list, 
                      path=os.path.join(config.project, "plots", "train_step_outoflane"), 
                      xlabel="Step", ylabel="Num Of OutOfLane", title="", label=algo_config["algo_name"])
    utils.single_plot(None, episode_outoflane_list, 
                      path=os.path.join(config.project, "plots", "train_epoch_outoflane"), 
                      xlabel="Epoch", ylabel="Num Of OutOfLane", title="", label=algo_config["algo_name"])
    
    utils.single_plot(None, relative_distance_of_center_line, 
                      path=os.path.join(config.project, "plots", "train_step_distance"), 
                      xlabel="Step", ylabel="Distance", title="", label=algo_config["algo_name"])
    
    
    
    env.close()
    eval_env.close()


if __name__ == '__main__':
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler('lanekeeping/data/my_log_LKA.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logging.info('episode y_dis v_y cos_h acc reward next_y_dis next_v_y next_cos_h done cost')
    main()
