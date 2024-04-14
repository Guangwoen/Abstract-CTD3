
import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
import csv
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import findOptimalK as fOK
from data_analysis.mdp.MDP import Node, Edge, Attr
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import *
import utils

def get_second_stage_mdp(data, kmeans_model, output_file):
    # 计算概率
    transition_probabilities = {}
    for row in data.itertuples(index=False):
        state = np.array(row.State).reshape(1, 2)
        next_state = np.array(row.NextState).reshape(1, 2)
        action = row.Act
        reward = row.Reward
        done = row.Done
        cost = row.Cost
        state_id_np = kmeans_model.predict(state)
        state_id = state_id_np[0]
        next_state_id_np = kmeans_model.predict(next_state)
        next_state_id = next_state_id_np[0]

        weight = row.Weight

        # 更新或添加概率
        if state_id in transition_probabilities:
            if next_state_id in transition_probabilities[state_id]:
                transition_probabilities[state_id][next_state_id]['Weight'] += weight
                transition_probabilities[state_id][next_state_id]['action'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['action'], action)]
                transition_probabilities[state_id][next_state_id]['reward'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['reward'], reward)]
                transition_probabilities[state_id][next_state_id]['cost'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['cost'], cost)]
                transition_probabilities[state_id][next_state_id]['count'] += 1
                # TODO done 是否要进行或运算 False or True
            else:
                transition_probabilities[state_id][next_state_id] = {
                    'Weight': weight,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'cost': cost,
                    'count': 1
                }
        else:
            transition_probabilities[state_id] = {next_state_id: {'Weight': weight,
                                                                  'action': action,
                                                                  'reward': reward,
                                                                  'done': done,
                                                                  'cost': cost,
                                                                  'count': 1}}

    cluster_centers = kmeans_model.cluster_centers_
    # 归一化权重，计算概率
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['State_ID', 'Next_State_ID', 'Probability', 'Act_Center', 'Reward_Center', 'Cost_Center', 'State_Center', 'NextState_Center', 'Weight', 'Done']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        csv_writer.writeheader()

        # 遍历每一行并写入 CSV 文件
        for state_id, successors in transition_probabilities.items():
            for next_state_id, weight_info in successors.items():
                weight = weight_info['Weight']
                total_weight = sum(weight_info['Weight'] for weight_info in successors.values())
                probability = weight / total_weight
                count = weight_info['count']
                act = [x / count for x in weight_info['action']]
                reward = [x / count for x in weight_info['reward']]
                cost = [x / count for x in weight_info['cost']]
                done = weight_info['done']

                # 将数据写入 CSV 文件
                csv_writer.writerow({
                    'State_ID': state_id,
                    'Next_State_ID': next_state_id,
                    'Probability': probability,
                    'Act_Center': act,
                    'Reward_Center': reward,
                    'Cost_Center': cost,
                    'State_Center': cluster_centers[[state_id]].tolist()[0],
                    'NextState_Center': cluster_centers[[next_state_id]].tolist()[0],
                    'Weight': weight,
                    'Done': done
                })

    return transition_probabilities


def perform_kmeans(data, optimal_k):
    # 曼哈顿距离
    # kmeans = KMeans(n_clusters=optimal_k, algorithm='full')
    # 欧式距离（即L2范数）
    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(data)
    return kmeans


if __name__ == '__main__':    
    # # Load the dataset: Raw data文件
    # file_path = './data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_raw_data.csv'
    # data = pd.read_csv(file_path)

    # # Convert string representations of lists into actual lists
    # data['State'] = data['State'].apply(eval)
    # data['NextState'] = data['NextState'].apply(eval)
    # data['Act'] = data['Act'].apply(eval)
    # data['Reward'] = data['Reward'].apply(eval)
    # data['Cost'] = data['Cost'].apply(eval)
    # # Combine 'State' and 'NextState' into a single list and remove duplicates
    # combined_states = data['State'].tolist() + data['NextState'].tolist()
    # combined_states = [list(x) for x in set(tuple(x) for x in combined_states)]

    """ ACC part """
    # path = "./conf/eval/highway_acc_eval.yaml"
    # eval_config = utils.load_yml(path)

    # graph, attr_dic, _ = get_mdp_map("./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv", eval_config)
    # states = get_mdp_states(graph)
    # print(states.shape)

    # Find optimal k using elbow method
    # optimal_k = fOK.find_optimal_k_canopy(states, t1=0.015, t2=0.005,
    #                                            save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/acc_td3_clusters/canopy')
    
    # Perform KMeans clustering with the optimal number of clusters
    # kmeans_model = perform_kmeans(states, 1260)

    # model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/euclidean_gap_Kmeans.pkl'
    # joblib.dump(kmeans_model, model_save_path)

    """ LKA part """
    # path = "./conf/eval/highway_lka_eval.yaml"
    # eval_config = utils.load_yml(path)

    # graph, attr_dic, _ = get_mdp_map("./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_pro_center_data.csv", eval_config)
    # states = get_mdp_states(graph)
    # print(states.shape)

    # # Find optimal k using elbow method
    # optimal_k = fOK.find_optimal_k_canopy(states, t1=0.015, t2=0.003,
    #                                            save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/lka_td3_risk/lka_td3_clusters/canopy')
    # print(optimal_k)
    # # Perform KMeans clustering with the optimal number of clusters
    # kmeans_model = perform_kmeans(states, optimal_k)
    
    # model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/lka_td3_risk/euclidean_canopy_Kmeans.pkl'
    # joblib.dump(kmeans_model, model_save_path)

    """ ICA part """
    path = "./conf/eval/highway_ica_eval.yaml"
    eval_config = utils.load_yml(path)

    graph, attr_dic, _ = get_mdp_map("./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_pro_center_data.csv", eval_config)
    states = get_mdp_states(graph)
    print(states.shape)

    # Find optimal k using elbow method
    optimal_k = fOK.find_optimal_k_gap(states, max_clusters=4000, min_clusters=2100, 
                                               save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/ica_dqn_clusters/gap')
    print(optimal_k)
    # Perform KMeans clustering with the optimal number of clusters
    kmeans_model = perform_kmeans(states, optimal_k)

    model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/euclidean_gap_Kmeans.pkl'
    joblib.dump(kmeans_model, model_save_path)