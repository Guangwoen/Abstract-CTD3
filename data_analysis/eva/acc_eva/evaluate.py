import copy
import os
from argparse import ArgumentParser

import gym

import highway_env
import joblib
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import utils
from algo import *
from data_analysis import utils_data
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import get_mdp_map
# from data_analysis.gap_statistic import compute_gap, gap_statistic
from data_analysis.mdp.second_stage_abstract.findOptimalK import find_optimal_k_elbow, find_optimal_k_gap, find_optimal_k_silhouette

save_fig_path = "./imgs/"
save_mdl_path = "./mdls/"


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, help="the method of kmeans",
                        default="gap")

    parser.add_argument("--cluster_again", type=bool, help='cluster again or not',
                        default=True)

    parser.add_argument("--val_again", type=bool, help="generate val data again or not",
                        default=False)

    parser.add_argument("--train_again", type=bool, help="generate train data again or not",
                        default=False)

    parser.add_argument("--checkpoint", type=str, help="the path to save project",
                        default="project/20220931-TD3_risk-acc/checkpoint")

    parser.add_argument("--model_config", type=str, help="path of rl algorithm configuration",
                        default="conf/algorithm/TD3_risk_acc.yaml")

    parser.add_argument("--eval_config", type=str, help="path of paramaters of intervalize algorithm configuration",
                        default="conf/eval/highway_acc_eval.yaml")

    parser.add_argument("--env_config", type=str, help="path of highway env",
                        default="conf/env/highway_acc_continuous_acceleration.yaml")

    parser.add_argument("--gpu", type=str, help="[0,1,2,3 | -1] the id of gpu to train/test, -1 means using cpu",
                        default="-1")

    args = parser.parse_args()
    return args


def compute_mae(y_true, y_predict, config):
    if not config.algo:
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

    mae = 0

    cluster_centers = mdl.cluster_centers_
    mae += sum(sum(abs(y_predict - y_true)))
    ret = mae / len(cluster_centers)
    print("the value of MAE is: %s" % ret)


if __name__ == '__main__':

    config = parse_args()

    log_file_train = '../../../my_log_ACC.log'
    csv_file_train = './td3_risk_acc_logs.csv'
    csv_file_phase = './result.csv'
    # csv_file_phase = './first_phase_result.csv'

    #   训练好的模型里面的
    log_file_val = '../../../trained_acc_data.log'
    csv_file_val = './td3_risk_acc_val_logs.csv'

    utils_data.log2csv(log_file_train, csv_file_train)



    #   构建MDP模型
    graph = get_mdp_map(csv_file_phase)

    #   生成聚类用数据
    train_data = []
    for tup in graph.keys():
        train_data.append([ele for inner_tuple in tup for ele in inner_tuple])
    train_data = np.array(train_data)

    if config.cluster_again:
        draw(train_data, graph, config)

    if config.val_again:
        eval_acc(config)

    utils_data.log2csv(log_file_val, csv_file_val)
    val_data = utils_data.get_csv_info(csv_file_val, 1, 'rel_dis_true', 'rel_speed_true',
                                       'rel_dis_predict', 'rel_speed_predict')
    val_data = np.array(val_data)

    y_true = val_data[:, 0:2]
    y_predict = val_data[:, 2:4]

    compute_mae(y_true, y_predict, config)
    # TODO：查找节点时多线程，