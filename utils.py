# coding=utf-8
# File      :   utils.py
# Time      :   2022/09/18 21:44:23
# Author    :   Jinghan Peng
# Desciption:   all kinds of functions
import math
import os, sys
import yaml
import csv
import time
import shutil
import numpy as np
import matplotlib
import torch
import pandas as pd

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties  # 导入字体模块
import matplotlib.image as mpimg


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


def make_plots_on_models(model_paths: list, save_path='', title='', 
                         do_smooth=False, fill_shadow=False, 
                         smooth_weight=0.9, limit=-1, target_field='Value', ylabel='return'):
    matplotlib.rcParams.update({'font.size': 18})
    plt.figure(figsize=(15, 8), dpi=80)
    plt.subplot(1, 1, 1)
    plt.title(title)
    plt.xlabel('episodes')
    plt.ylabel(ylabel)

    for i in range(len(model_paths)):
        path = model_paths[i]['data']
        steps = list()
        values = list()
        count = 0
        with open(path, 'r') as model_csv:
            reader = csv.DictReader(model_csv)
            for row in reader:
                steps.append(int(row['Step']))
                values.append(float(row[target_field]))
                count += 1
                if limit != -1 and count >= limit:
                    break
                
        model_csv.close()
        
        if do_smooth:
            values = smooth(values, smooth_weight)
            
        if fill_shadow:
            # mu = np.array(values).mean()
            sigma = np.array(values).std()
            plt.plot(steps, values, linestyle='-', linewidth=1.5, label=model_paths[i]['label'])
            plt.fill_between(steps, values-sigma, values+sigma, alpha=0.3)
        else:
            plt.plot(steps, values, linestyle='-', linewidth=1.5, label=model_paths[i]['label'])

    if len(model_paths) > 1:
        plt.legend(loc='upper left')
    plt.savefig(save_path)
    
    # save plot in history cache
    format_time = time.strftime("%Y%m%d%H%M", time.localtime())
    plt.savefig('/home/akihi/data1/Abstract-CTD3-main-master/plot_history/{}.png'.format(format_time))

    plt.show()
    
    
def save_csv_on_models(data, save_path=''):
    with open(save_path, 'w') as save_csv:
        writer = csv.writer(save_csv)
        keys = list(data.keys())
        writer.writerow(keys)
        writer.writerows([data[key][i] for key in keys] for i in range(len(data[keys[0]])))
        

def get_model_name_from_path(path):
    return path.split('/')[-1].split('_elbow')[0]


def load_yml(path):
    """load yaml file"""
    with open(path, 'r', encoding='utf-8') as rf:
        config = yaml.load(rf.read(), Loader=yaml.FullLoader)
    return config


def lmap(v, x, y):  # -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def normalize_observation(observations, features, features_range, clip=False):
    """ 对观察值进行正则化到 [-1,1]之间
    Args:
        observations (np): 观测值
        features (list): 观测值的类型
        features_range (dict): 需要正则化的观察值的上下界
    Returns:
        np: 正则化后的观察值
    """
    for index, (feature) in enumerate(features):  # 遍历每一种观测值
        if feature in features_range:
            f_range = features_range[feature]

            observations[:, index] = lmap(observations[:, index], [f_range[0], f_range[1]], [-1, 1])
            if clip:
                observations[:, index] = np.clip(observations[:, index], -1, 1)

    return observations


def make_dir(project_path, model_config=None, env_config=None):
    """create the dir of project to save the models and results"""
    if os.path.exists(project_path):
        print("Project already existed! Do you want to create it? [Y/N]")
        if input() in ('N', 'n'):
            exit(0)

    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "trajectory"), exist_ok=True)

    if os.path.exists(model_config):
        shutil.copy(model_config, project_path)
    if os.path.exists(env_config):
        shutil.copy(env_config, project_path)


#   把状态变成算法第一阶段后的区间形式
def intervalize_state(state, config):
    dim = config["dim"]["state_dim"]
    gran = config["granularity"]["state_gran"]
    prec = config["prec"]

    ret = []
    #   每个维度找到对应的区间
    for i in range(dim):
        base = math.floor(state[i] / gran[i]) * gran[i]
        base += gran[i] / 2
        
        ret.append("{:.3f}".format(base))
    return ret


def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(plot_cfg.env_name,
                                       plot_cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if plot_cfg.save:
        plt.savefig(os.path.join(plot_cfg.result_path, f"{tag}_rewards_curve_cn"))
    # plt.show()


def plot_rewards(rewards, plot_cfg, step_list, tag='train', xlabel="Step", ylabel="Reward"):
    """plot the reward of each epoch"""
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"

    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(step_list, rewards, label='epoch_rewards')
    # plt.plot(ma_rewards, label='ma_rewards')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"], "{}_rewards_curve".format(tag)))
    # plt.show()
    plt.close()


def plot_outoflanes(ep_outoflanes, plot_cfg, step_list, tag='train', xlabel="Step", ylabel="Number of Out of Lane"):
    """plot the count of out of lane for each epoch"""
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"

    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(step_list, ep_outoflanes, label='epoch_outoflanes')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"], "{}_outoflanes_curve".format(tag)))
    # plt.show()
    plt.close()


def plot_relative_distance_of_center_line(ep_outoflanes, plot_cfg, step_list, tag='train', xlabel="Step",
                                          ylabel="Distance"):
    """plot the """
    if plot_cfg["device"] == "-1":
        device = "cpu"
    else:
        device = f"cuda:{plot_cfg['device']}"

    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("distance curve on {} of {} for {}".format(
        device, plot_cfg["algo_name"], plot_cfg["env_name"]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if step_list:
        plt.plot(step_list, ep_outoflanes, label='distance')
    else:
        plt.plot(ep_outoflanes, label='distance')
    plt.legend()
    if plot_cfg["save"]:
        plt.savefig(os.path.join(plot_cfg["result_path"], "{}_distance_curve".format(tag)))
    # plt.show()
    plt.close()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def single_plot(x, y, path, xlabel="", ylabel="", title="", label=None):
    """simple plot for any data"""
    sns.set()
    plt.figure()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if x:
        plt.plot(x, y, label=label)
    else:
        plt.plot(y, label=label)

    plt.legend()
    plt.savefig(path)
    # plt.show()
    plt.close()


def save_results(rewards, costs, episode_steps, tag='train', path='./results'):
    ''' 保存奖励，平滑奖励，驶出车道次数
    '''
    np.save(os.path.join(path, '{}_rewards.npy'.format(tag)), rewards)
    # np.save(os.path.join(path,'{}_ma_rewards.npy'.format(tag)), ma_rewards)
    np.save(os.path.join(path, '{}_costs.npy'.format(tag)), costs)

    np.save(os.path.join(path, '{}_episode_steps.npy'.format(tag)), episode_steps)

    print('结果保存完毕！')
    
    
if __name__ == '__main__':
    
    # make_plots_on_models([
    #     {
    #         'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/ddpg/risk_with_model_data/data/202403011554.csv',
    #         'label': 'DDPG'
    #     },
    #     {
    #         'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/ddpg/risk_data/data/202403011752.csv',
    #         'label': 'MDP-Guided DDPG'
    #     },
    #     {
    #         'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/ddpg/data/data/202403011508.csv',
    #         'label': 'ARC-DDPG'
    #     },
    #     {
    #         'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/ddpg/with_model_data/data/202403011518.csv',
    #         'label': 'MDP-Guided ARC-DDPG'
    #     }
    # ], '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/ddpg/compared_result.png', 
    #                     do_smooth=True, fill_shadow=True, smooth_weight=0.9, limit=450, target_field='Distance', ylabel='too far distance count')
    
    """ACC"""
    make_plots_on_models([{
        'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/td3/risk_data/data/202402292358.csv',
        'label': 'TD3'
    }, {
        'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/td3/risk_with_model_data/data/202403010034.csv',
        'label': 'MDP-Guided TD3'
    }, {
        'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/td3/with_model_data/data/202402292107.csv',
        'label': 'ARC-TD3'
    }, {
        'data': '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/td3/data/data/202402292021.csv',
        'label': 'MDP-Guided ARC-TD3'
    }], '/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/data/risk/td3/compared_result.png', do_smooth=True, fill_shadow=True, smooth_weight=0.99, limit=500, target_field='Distance', ylabel='too far distance count')
    