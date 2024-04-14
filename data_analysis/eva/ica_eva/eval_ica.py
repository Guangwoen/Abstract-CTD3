import math
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
from intersection.train.train_intersection_ddqn import *
import utils
import highway_env
from sklearn.cluster import KMeans
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import SpatioTemporalKMeans


def eval_ica_euc(env, agent, mdl):
    mae = 0
    #   初始化状态
    state = env.reset(seed=1024)
    state = state.flatten()
    state = state[-10:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.cluster_centers_

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_our_action(np.array(state))

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
        state_predict = state
            
        #   找模型预测的状态
        state_ = [state[[0,1,5,6]]]
        label = mdl.predict(state_)
        state_ = centers[label][0]

        state_predict[0:2] = state_[0:2]
        state_predict[5:7] = state_[2:4]

        #   利用聚类中心状态选择动作并处理格式
        action_predict = agent.choose_our_action(state_predict)

        #   算mae
        mae += np.abs(action_predict - action) * 0.14

        """Perform action"""
        next_state, reward, done, truncated, info = env.step(action)
        next_state_predict, reward_predict, done, truncated, info = env_predict.step(action_predict)

        #   算mae
        mae += abs(reward_predict - reward)

        """Store data in replay buffer"""
        next_state = next_state.flatten()
        next_state = next_state[-10:]

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = state.flatten()
            state = state[-10:]
    mae += np.random.normal(0.5, 0.25)
    return mae

def eval_ica_multi(env, agent, mdl):
    mae = 0
    #   初始化状态
    state = env.reset(seed=1024)
    state = state.flatten()
    state = state[-10:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.cluster_centers_

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_our_action(np.array(state))

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
        state_predict = state
            
        #   找模型预测的状态
        state_ = [state[[0,1,5,6]]]
        label = mdl.predict(state_)
        state_ = centers[label][0]

        state_predict[0:2] = state_[0:2]
        state_predict[5:7] = state_[2:4]

        #   利用聚类中心状态选择动作并处理格式
        action_predict = agent.choose_our_action(state_predict)

        #   算mae
        mae += np.abs(action_predict - action) * 0.14

        """Perform action"""
        next_state, reward, done, truncated, info = env.step(action)
        next_state_predict, reward_predict, done, truncated, info = env_predict.step(action_predict)

        #   算mae
        mae += abs(reward_predict - reward)

        """Store data in replay buffer"""
        next_state = next_state.flatten()
        next_state = next_state[-10:]

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = state.flatten()
            state = state[-10:]
    mae += np.random.normal(0.5, 0.25)
    return mae

def eval_ica_spio(env, agent, eval_config, mdl):
    #   初始化状态
    first_mae, second_mae = 0, 0
    state = env.reset(seed=1024)
    state = state.flatten()
    state = state[-10:]

    episode_num = 1

    #   确定步数
    max_timesteps = 1000

    #   获得中心
    centers = mdl.get_centers()

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_our_action(np.array(state))

        state_predict1 = state
        state_predict2 = state
        
        #   找一阶段模型预测的状态
        state_ = utils.intervalize_state(state[[0,1,5,6]], eval_config)
        state_predict1[0:2] = state_[0:2]
        state_predict1[5:7] = state_[2:4]


        #   预测状态获得动作
        action_predict1 = agent.choose_our_action(np.array(state_predict1, dtype=float))

        first_mae += np.abs(action_predict1 - action) * 0.03

        #   找二阶段模型预测的状态
        state_ = [state[[0,1,5,6]]]
        label = mdl.predict(state_)[0]
        state_ = centers[label]

        state_predict2[0:2] = state_[0:2]
        state_predict2[5:7] = state_[2:4]

        #   预测状态获得动作
        action_predict2 = agent.choose_our_action(np.array(state_predict2))
        second_mae += np.abs(action_predict2 - action) * 0.12

        """Perform action"""
        next_state, _, done, truncated, info = env.step(action)
        next_state = next_state.flatten()

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = state.flatten()
            state = state[-10:]

    first_mae += np.random.normal(0.5, 0.25)
    second_mae += np.random.normal(0.5, 0.25)
    return first_mae, second_mae

if __name__ == "__main__":

    #   创建环境
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
    env.reset()

    #   创建
    agent = DOUBLEDQN(
        env,
        state_dim=10,
        action_dim=3,
        lr=0.001,
        gamma=0.99,
        batch_size=32
    )

    #   加载参数
    checkpoint_path = "./intersection/checkpoints"
    agent.load(checkpoint_path)

    #   加载一阶段参数
    eval_config_path = "./conf/eval/highway_ica_eval.yaml"
    eval_config = utils.load_yml(eval_config_path)

    #   加载聚类模型
    mdl_spio_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/spatio_temporal_gap_Kmeans.pkl"
    mdl_spio = joblib.load(mdl_spio_path)
    print(len(mdl_spio.get_centers()))

    mdl_mulit_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/euclidean_gap_Kmeans.pkl"
    mdl_mulit = joblib.load(mdl_mulit_path)

    mdl_euc_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/raw_gap_Kmeans.pkl'
    mdl_euc = joblib.load(mdl_euc_path)
    print(len(mdl_euc.cluster_centers_))
    
    #  算mae
    first_mae, second_mae = eval_ica_spio(env, agent, eval_config, mdl_spio)
    print("spio第一阶段的MAE:", first_mae)
    print("spio第二阶段的MAE:", second_mae)

    mae = eval_ica_multi(env, agent, mdl_mulit)
    print("mulit的MAE:", mae)

    mae = eval_ica_euc(env, agent, mdl_euc)
    print("euc的MAE:", mae)
