import sys
sys.path.append('/home/akihi/data1/Abstract-CTD3-main-master')

import numpy as np
import gym
import highway_env
import torch
import argparse
import logging
from copy import deepcopy
from tensorboardX import SummaryWriter

import utils
from algo.option_critic import OptionCritic, ReplayBuffer
from algo.option_critic import actor_loss as actor_loss_fn
from algo.option_critic import critic_loss as critic_loss_fn

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='highway-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')

parser.add_argument("--env_config", type=str, help="path of highway env",
                        default="conf/env/highway_acc_continuous_acceleration.yaml")
parser.add_argument("--project", type=str, help="the path to save project",
                        default="project/20240130-Option-Critic-acc")


def get_reward(state):
    reward = 0.
    presence, x, ego_vx = state[0]
    reward += (0.05 * ego_vx)
    return reward


def train(env, args, writer):
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    agent = OptionCritic(
        in_features=env.observation_space.shape[0] * env.observation_space.shape[1],  # 25
        num_actions=env.action_space.n,  # 5
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )

    agent_prime = deepcopy(agent)

    optim = torch.optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)

    steps = 0
    ep_num = 0
    while steps < args.max_steps_total:

        rewards = 0
        option_lengths = {opt: [] for opt in range(args.num_options)}

        obs = env.reset(seed=args.seed)[0]
        obs = obs.flatten()
        state = agent.get_state(utils.to_tensor(obs))
        greedy_option = agent.greedy_option(state)
        current_option = 0

        done = False
        ep_steps = 0
        option_termination = True
        curr_op_len = 0

        ep_reward_list = list()

        while not done and ep_steps < args.max_steps_ep:
            epsilon = agent.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0

            action, logp, entropy = agent.get_action(state, current_option)

            next_obs, reward, done, truncated, info = env.step(action)

            reward = get_reward(next_obs)

            next_obs = next_obs.flatten()
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            # env.render()

            actor_loss, critic_loss = None, None

            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs, current_option, logp, entropy,
                    reward, done, next_obs, agent, agent_prime, args
                )
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(
                        agent, agent_prime, data_batch, args
                    )
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    agent_prime.load_state_dict(agent.state_dict())  # update prime agent
            state = agent.get_state(utils.to_tensor(next_obs))
            option_termination, greedy_option = agent.predict_option_termination(state, current_option)

            # update global params
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            ep_reward_list.append(rewards)

            logging.info('steps: %s, rewards: %s, actor_loss: %s, critic_loss: %s, entropy: %s, epsilon: %s',
                         steps, rewards, actor_loss, critic_loss, entropy.item(), epsilon)

        ep_num += 1
        writer.add_scalar('Reward/episode_reward', rewards, global_step=ep_num)
        logging.info('steps: %s, rewards: %s, option_lengths: %s, ep_steps: %s, epsilon: %s',
                     steps, rewards, option_lengths, ep_steps, epsilon)


def main():
    args = parser.parse_args()
    env = gym.make(args.env, render_mode='rgb_array')
    env.configure(utils.load_yml(args.env_config))
    writer = SummaryWriter('./')
    train(env, args, writer)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler('trained_acc_data.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ],
        # filemode='w'
    )
    main()
