import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
import logging
import os
import copy
from tqdm import *

import gym
import joblib
import numpy as np

from algo import *
import utils
import highway_env
from sklearn.cluster import KMeans
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import SpatioTemporalKMeans

def get_reward(state):

    reward = 0

    ego_presence, ego_x, ego_vx = state[0]

    reward += (0.05 * ego_vx)

    return reward

def eval_acc_euc(env, agent, env_config, algo_config, mdl):
    mae = 0
    #   初始化状态
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.cluster_centers_

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
            
        #   找模型预测的状态
        state_ = [state]
        label = mdl.predict(state_)
        state_predict = centers[label]

        #   利用聚类中心状态选择动作并处理格式
        action_predict = agent.choose_action(state_predict)
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict = [action_predict[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict = np.array(real_action_predict).clip(env.action_space.low, env.action_space.high)
        #   算mae
        mae += np.sum(np.abs(real_action_predict - real_action))

        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)
        next_state_predict, _, done, truncated, info = env_predict.step(real_action_predict)

        #   考虑奖励
        reward = get_reward(next_state)
        reward_predict = get_reward(next_state_predict)
        #   算mae
        mae += abs(reward_predict - reward)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]
    return mae

def eval_acc_multi(env, agent, env_config, algo_config, mdl):
    mae = 0
    #   初始化状态
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.cluster_centers_

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
            
        #   找模型预测的状态
        state_ = [state]
        label = mdl.predict(state_)
        state_predict = centers[label]

        #   利用聚类中心状态选择动作并处理格式
        action_predict = agent.choose_action(state_predict)
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict = [action_predict[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict = np.array(real_action_predict).clip(env.action_space.low, env.action_space.high)
        #   算mae
        mae += np.sum(np.abs(real_action_predict - real_action))

        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)
        next_state_predict, _, done, truncated, info = env_predict.step(real_action_predict)

        #   考虑奖励
        reward = get_reward(next_state)
        reward_predict = get_reward(next_state_predict)
        #   算mae
        mae += abs(reward_predict - reward)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]
    return mae
    

def eval_acc_spio(env, agent, env_config, algo_config, eval_config, mdl):
    #   初始化状态
    first_mae, second_mae = 0, 0
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.get_centers()

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)
        
        #   找一阶段模型预测的状态
        state_predict1 = utils.intervalize_state(state, eval_config)

        #   预测状态获得动作
        action_predict1 = agent.choose_action(np.array(state_predict1, dtype=float))
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict1 = [action_predict1[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict1 = np.array(real_action_predict1).clip(env.action_space.low, env.action_space.high)
        first_mae += np.sum(np.abs(real_action_predict1 - real_action))

        #   找二阶段模型预测的状态
        state_ = [state]
        label = mdl.predict(state_)[0]
        state_predict2 = centers[label]

        #   预测状态获得动作
        action_predict2 = agent.choose_action(np.array(state_predict2))
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict2 = [action_predict2[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict2 = np.array(real_action_predict2).clip(env.action_space.low, env.action_space.high)
        second_mae += np.sum(np.abs(real_action_predict2 - real_action)) / 2

        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]

    return first_mae, second_mae

if __name__ == "__main__":
    # #   创建日志，要求可重写
    # logging.basicConfig(
    #     filename='./data_analysis/eva/acc_eva/acc_eval.log',
    #     level=logging.INFO,
    #     filemode='w',
    #     format='%(message)s',
    # )
    # logger = logging.getLogger('log')
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # logger.addHandler(console_handler)

    # #   格式时epi, 状态真实值, 状态预测值, 簇标签
    # logging.info('Episode rel_dis_true rel_speed_true rel_dis_predict rel_speed_predict label')

    #   创建环境
    env_config_path = "./conf/env/highway_acc_continuous_acceleration.yaml"
    env_config = utils.load_yml(env_config_path)

    env = gym.make(env_config["env_name"], render_mode='human')
    env.configure(env_config["config"])

    #   创建agent
    algo_config_path = "./conf/algorithm/TD3_risk_acc.yaml"
    algo_config = utils.load_yml(algo_config_path)
    device = 0
    agent = TD3_risk_disturbance(algo_config, device, writer=None)

    #   加载参数
    checkpoint_path = "./project/20220931-TD3_risk-acc/checkpoint"
    agent.load(checkpoint_path)

    #   加载一阶段参数
    eval_config_path = "./conf/eval/highway_acc_eval.yaml"
    eval_config = utils.load_yml(eval_config_path)

    #   加载聚类模型
    mdl_spio_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_gap_Kmeans.pkl"
    mdl_spio = joblib.load(mdl_spio_path)

    print(mdl_spio.get_centers())

    mdl_mulit_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/euclidean_gap_Kmeans.pkl"
    mdl_mulit = joblib.load(mdl_mulit_path)

    mdl_euc_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/raw_gap_Kmeans.pkl"
    mdl_euc = joblib.load(mdl_euc_path)
    
    #   算mae
    print(mdl_euc_path)
    first_mae, second_mae = eval_acc_spio(env, agent, env_config, algo_config, eval_config, mdl_spio)
    print("spio第一阶段的MAE:", first_mae)
    print("spio第二阶段的MAE:", second_mae)

    mae = eval_acc_multi(env, agent, env_config, algo_config, mdl_mulit)
    print("mulit的MAE:", mae)

    mae = eval_acc_euc(env, agent, env_config, algo_config, mdl_euc)
    print("euc的MAE:", mae)
    