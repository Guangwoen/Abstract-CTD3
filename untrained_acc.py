import logging

import gym
import highway_env

import numpy as np

import utils

def get_reward(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your reward function here! """

    reward = 0.

    ego_presence, ego_x, ego_vx = state[0]
    lead_presence, lead_relative_x, lead_relative_vx = state[1]

    reward += (0.05 * ego_vx)

    return reward


def get_cost(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your risk function here! """
    cost = 0
    if info["crashed"]:
        cost = 100

    return cost





if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s [%(levelname)s] %(message)s',
        format='%(message)s',
        handlers=[
            logging.FileHandler('raw_acc_data.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logging.info('episode, rel_dis, rel_speed, acc, reward, next_rel_dis, next_rel_speed, done, cost')

    env_config = "conf/env/highway_acc_continuous_acceleration.yaml"
    env_config = utils.load_yml(env_config)

    algo_config = "conf/algorithm/TD3_risk_acc.yaml"
    algo_config = utils.load_yml(algo_config)

    env = gym.make(env_config["env_name"], render_mode='human')
    print(env.action_space)
    env.configure(env_config["config"])
    print(env.action_space)

    #   获取状态
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    max_timesteps = algo_config["trainer"]["max_timesteps"]
    max_timesteps =max_timesteps // 5
    for t_step in range(max_timesteps):
        env.render()
        t_step += 1

        action = 2 * np.random.random(size=(1,)) - 1
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
        print(env.action_space)
        real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        """Perform action"""
        next_state, reward, done, truncated, info = env.step(real_action)

        """Define the reward composition"""
        reward = get_reward(info, next_state, action, done, truncated)

        """Define the cost"""
        cost = get_cost(info, next_state, action, done, truncated)
        # print("cost:", cost)
        is_crash = 1 if info["crashed"] else 0

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        logging.info('%s, %s, %s, %s, %s, %s, %s', episode_num, state, action, reward, next_state, done, cost)

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

