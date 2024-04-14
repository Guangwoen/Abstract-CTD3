import logging
import os
import copy
from tqdm import *

import gym
import joblib
import numpy as np

import highway_env
import utils
from algo import *

def eval(env, agent,  env_config, algo_config, mdl):
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    max_timesteps = algo_config["trainer"]["max_timesteps"]
    max_timesteps = max_timesteps // 10
    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        env_predict = copy.deepcopy(env)
        label = mdl.predict(np.array(state, dtype='float').reshape(1, -1))
        state_predict = mdl.cluster_centers_[label]

        action_predict = agent.choose_action(state_predict)
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict = [action_predict[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict = np.array(real_action_predict).clip(env.action_space.low, env.action_space.high)


        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)

        next_state_predict, _, done, truncated, info = env_predict.step(real_action_predict)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        next_state_predict = utils.normalize_observation(next_state_predict,
                                                    env_config["config"]["observation"]["features"],
                                                    env_config["config"]["observation"]["features_range"],
                                                    clip=False)
        next_state_predict = next_state_predict.flatten()
        next_state_predict = next_state_predict[-2:]

        logging.info('%s, %s, %s %s', episode_num, next_state, next_state_predict, label)

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


def eval_acc(config):
    logging.basicConfig(
        filename='./../../trained_acc_data.log',
        level=logging.INFO,
        filemode='w',
        # format='%(asctime)s [%(levelname)s] %(message)s',
        format='%(message)s',
        # handlers=[
        #     logging.FileHandler('./../../trained_acc_data.log'),  # 输出到文件
        #     logging.StreamHandler()  # 输出到控制台
        # ]
    )
    logger = logging.getLogger('log')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logging.info('Episode rel_dis_true rel_speed_true rel_dis_predict rel_speed_predict label')
    path = os.path.join("./../../", config.env_config)
    env_config = utils.load_yml(path)
    env = gym.make(env_config["env_name"], render_mode='human')
    env.configure(env_config["config"])
    path = os.path.join("./../../", config.model_config)


    algo_config = utils.load_yml(path)
    if algo_config["algo_name"] == "TD3":
        agent = TD3(algo_config, config.gpu, writer=None)
    elif algo_config["algo_name"] == "TD3_risk_disturbance":
        agent = TD3_risk_disturbance(algo_config, config.gpu, writer=None)

    agent.load(os.path.join("./../../", config.checkpoint))

    save_mdl_path = "./mdls/"
    if config.mode == 'gap':
        mdl = joblib.load(save_mdl_path + 'trad_kmeans_gap.pkl')
    elif config.mode == 'canopy':
        mdl = joblib.load(save_mdl_path + 'trad_kmeans_canopy.pkl')
    elif config.mode == 'elbow':
        mdl = joblib.load(save_mdl_path + 'trad_kmeans_elbow.pkl')
    elif config.mode == 'silhouette':
        mdl = joblib.load(save_mdl_path + 'trad_kmeans_silhouette.pkl')
    else:
        print("ERROR: check the value of parameter mode")
        exit(0)
    eval(env, agent, env_config, algo_config, mdl)
