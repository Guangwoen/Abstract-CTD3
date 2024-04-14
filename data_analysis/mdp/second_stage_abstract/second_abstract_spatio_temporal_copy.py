import sys
import time
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
import copy
import math
import csv
import ast
import os
import json

import joblib
import numpy as np
import tqdm
import pickle
import concurrent.futures

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.samples.definitions import SIMPLE_SAMPLES

from scipy.spatial.distance import jensenshannon
import networkx as nx
import matplotlib.pyplot as plt
import sys
# import findOptimalK as fOK
import utils
from data_analysis.mdp.MDP import Node, Edge, Attr
from data_analysis.mdp.second_stage_abstract.findOptimalK import * 
from data_analysis.mdp.second_stage_abstract.kmeans import kmeans

#   sys.path.append("F:\桌面\Abstract-CTD3-main-master\data_analysis\\acc_td3")


#   把图还原成2darray
def graph2arr(graph):
    lists = []
    for key, value in graph.items():
        ls = []
        #   处理state
        for it in value.state:
            ls = ls + [*it]

        for edge in value.edges:
            #   处理next_state
            ls1 = copy.deepcopy(ls)
            for it in edge.next_node.state:
                ls1 = ls1 + [*it]
            #   处理action reward cost done prob
            for i in range(len(edge.action)):
                ls2 = copy.deepcopy(ls1)
                ls2 = ls2 + [*edge.action[i]]
                ls2 = ls2 + [*edge.reward[i]]
                ls2 = ls2 + [*edge.cost[i]]
                ls2.append(1 if edge.done is True else 0)
                ls2.append(edge.prob)

                lists.append(ls2)

    ret = np.array(lists)
    return ret

def get_mdp_map(input_file, config):
    mdp_dic = {}
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            state_center = ast.literal_eval(row['State_Center'])
            act_center = ast.literal_eval(row['Act_Center'])
            reward_center = ast.literal_eval(row['Reward_Center'])
            next_state_center = ast.literal_eval(row['NextState_Center'])
            done = ast.literal_eval(row['Done'])
            cost_center = ast.literal_eval(row['Cost_Center'])
            weight = float(row['Weight'])
            probability = float(row['Probability'])

            current_state_key = tuple(state_center)
            next_state_key = tuple(next_state_center)

            if current_state_key in mdp_dic:
                current_state = Node.from_dict(mdp_dic[current_state_key])
                if next_state_key in mdp_dic:
                    next_state = Node.from_dict(mdp_dic[next_state_key])
                else:
                    next_state = Node(next_state_center)
                    mdp_dic[next_state_key] = next_state.to_dict()
                current_state.add_edge(next_state, act_center, reward_center, done, cost_center, weight, probability)
                mdp_dic[current_state_key] = current_state.to_dict()
            else:
                current_state = Node(state_center)
                if next_state_key in mdp_dic:
                    next_state = Node.from_dict(mdp_dic[next_state_key])
                else:
                    next_state = Node(next_state_center)
                # if current_state == next_state:
                #     print(True)
                current_state.add_edge(next_state, act_center, reward_center, done, cost_center, weight, probability)
                mdp_dic[current_state_key] = current_state.to_dict()
                if next_state_key not in mdp_dic:
                    mdp_dic[next_state_key] = next_state.to_dict()

    attr_dic = {}
    for state_key, state in mdp_dic.items():
        attr_dic[state_key] = Attr(Node.from_dict(state)).to_dict()
    
    attr_dic_str = {}
    for state_key, state in mdp_dic.items():
        state_key_str = utils.intervalize_state(np.array(state_key), config)
        state_key_str = tuple(state_key_str)
        attr_dic_str[state_key_str] = Attr(Node.from_dict(state)).to_dict()

    return mdp_dic, attr_dic, attr_dic_str


def visualize_mdp(mdp_dic, save_path=None):
    G = nx.DiGraph()

    for state_key, state_data in mdp_dic.items():
        state_node = Node.from_dict(state_data)
        G.add_node(str(state_key))

        for edge in state_node.edges:
            next_state_key = tuple(edge.next_node.state)
            G.add_node(str(next_state_key))
            G.add_edge(str(state_key), str(next_state_key), action=edge.action, reward=edge.reward,
                       done=edge.done, cost=edge.cost, weight=edge.weight, prob=edge.prob)

    pos = nx.spring_layout(G, seed=42)  # You can choose a different layout if needed
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black',
            font_size=1, edge_color='gray', width=1, alpha=0.7, arrowsize=10)

    edge_labels = {(state,
                    next_state): f"{edge['action']}, R={edge['reward']}, Done={edge['done']}, Cost={edge['cost']}, Weight={edge['weight']}, Prob={edge['prob']:.2f}"
                   for state, next_state, edge in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def convert_to_list(input_data):
    result = []
    for item in input_data:
        if isinstance(item, str):
            # 如果元素是字符串形式的元组，使用 ast.literal_eval 解析
            item = ast.literal_eval(item)
        if isinstance(item, tuple):
            # 如果元素是元组，直接转换为列表
            result.append(list(item))
        else:
            raise ValueError("Invalid element format in the input list")
    return result


def get_mdp_states(mdp_dic, decimal_places=4):
    states = list(mdp_dic.keys())
    datas = convert_to_list(states)
    # datas = [list(ast.literal_eval(item)) for item in states]
    return np.array(datas)


def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return abs((vec1 - vec2)).sum()


class SpatioTemporalKMeans_copy(kmeans):
    def __init__(self, config, graph, graph_str, datas, initial_centers=None, k=8,
                 tolerance=0.001, ccore=True):
        """
        # config 一阶段抽象的参数conf/eval/...yaml
        # graph dict一阶段抽象后生成的mdp图 key:((upperbound, lowerbound), ...) value:node
        # data 2d-array 把graph.keys()里面的tuple变成list 然后2d列表转数组
        """
        # 初始化中心点和距离函数
        if not initial_centers:
            initial_centers = kmeans_plusplus_initializer(datas, k).initialize()

        metric = distance_metric(type_metric.USER_DEFINED, func=self.distance)

        super().__init__(datas, initial_centers, config, tolerance, ccore, metric=metric)
        self.cr = config["cr"]
        self.cd = config["cd"]
        self.cp = config["cp"]
        self.cs = config["cs"]
        self.config = config
        self.state_dim = config["dim"]["state_dim"]
        self.graph = graph
        self.graph_str = graph_str
        self.precs = config["prec"]
        self.ranges = [2.45122418, 1.85,      0.82279176, 2.16      ]
        # self.ranges = self.calculate_ranges(datas)
        # print(self.ranges)

    #   获取保留几位小数的精度
    def get_prec(self, config):
        """
        输入config是一阶段抽象里面的参数，用里面的粒度来计算保留精度
        如果gran[0] 是 0.1，那么 math.ceil(math.log10(0.1) * -1) 将得到 1，表示这个维度上的状态可以精确到小数点后 1 位。
        """
        ret = []
        gran = config["granularity"]["state_gran"]
        for i in range(self.state_dim):
            ret.append(math.ceil(math.log10(gran[i]) * -1))
        return ret


    #   原始数据点在graph里面对应的节点
    def get_node1(self, state):
        """
        data 待匹配状态 1darray 从init的data里面提取出来的一行
        return 匹配到的节点 node
        """
        data_tup = tuple(state)

        #   state是图中的网格中心点的情况
        if data_tup in self.graph.keys():
            return self.graph[data_tup]
        #   上面情况不满足，说明state所在的网格在原始MDP中没有数据，理论上讲不会有这种情况
        return None
        # return 0
    
    #   原始数据点在graph_str里面对应的节点，用str格式化，解决精度问题
    def get_node2(self, state):
        """
        data 待匹配状态 1darray 从init的data里面提取出来的一行
        return 匹配到的节点 node
        """
        data_tup = tuple(state)

        #   state是图中的网格中心点的情况
        if data_tup in self.graph_str.keys():
            return self.graph_str[data_tup]
        #   上面情况不满足，说明聚类中心代表的网格在graph中没有数据

        return None

    #   输入两个概率分布，返回差异
    def compute_distribution_difference(self, prob_x, prob_y):
        max_len = max(len(prob_x), len(prob_y))

        #   保证长度一样，不一样补0到一样
        if len(prob_x) == max_len:
            for _ in range(max_len - len(prob_y)):
                prob_y.append(0)
        else:
            for _ in range(max_len - len(prob_x)):
                prob_x.append(0)

        # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
        return jensenshannon(prob_x, prob_y)

    #   输入两个attr奖励的差值
    def get_reward_distance(self, attr_x, attr_y):
        sum_x = 0
        sum_y = 0
        for i in range(len(attr_x['rewards'])):
            sum_x += attr_x['probs'][i] * attr_x['rewards'][i][0]

        for i in range(len(attr_y['rewards'])):
            sum_y += attr_y['probs'][i] * attr_y['rewards'][i][0]

        return abs(sum_x-sum_y)

    def get_action_distance(self, attr_x, attr_y):
        sum_x = 0
        sum_y = 0
        for i in range(len(attr_x['actions'])):
            for j in range(len(attr_x['actions'][i])):
                sum_x += attr_x['probs'][i] * attr_x['actions'][i][j]

        for i in range(len(attr_y['actions'])):
            for j in range(len(attr_y['actions'][i])):
                sum_y += attr_y['probs'][i] * attr_y['actions'][i][j]

        return abs(sum_x - sum_y)

    #   计算mdp中两个节点距离
    #   距离最小值是0,，把[0, upbound]归一到[0, 1]， 除以upbound即可
    def distance(self, data1, data2):
        """
        data1, data2: 1d-array 两个状态
        return: float 两个状态之间距离
        """
        # 判断两个状态是否相同
        x = self.get_node1(data1)
        y = self.get_node2(data2)
        # print(x['state'])
        #   xy其中之一所在的网格在原始MDP中没有数据
        if x is None or y is None:
            return self.cs * np.linalg.norm(np.array(data1) - np.array(data2)) / self.ranges[0]

        if x['state'] == y['state']:
            return 0

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x['state']['state']) - np.array(y['state']['state'])) / self.ranges[0]

        #   如果有一个是终止节点
        if not x['next_states'] or not y['next_states']:
            return self.cs * state_distance

        #   动作区间交集奖励的最大差异
        max_reward_difference = self.get_reward_distance(x, y) / self.ranges[1]  # 动作区间交集

        #   后继状态分布的差异
        distribution_difference = self.compute_distribution_difference(x['probs'], y['probs']) / self.ranges[2]

        #   动作的最大差异
        max_action_difference = self.get_action_distance(x, y) / self.ranges[3]

        return self.cs * state_distance + self.cr * max_reward_difference + self.cd * max_action_difference + self.cp * distribution_difference

    #   以列表形式返回两个点之间的四种距离
    def calculate_distances(self, x, y):
        """
        x,y: graph中的两个节点
        ret: list[float] 长度为4 代表两点之间4种距离
        """
        if x['state'] == y['state']:
            return [0, 0, 0, 0]

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x['state']['state']) - np.array(y['state']['state']))

        #   如果有一个是终止节点
        if not x['next_states'] or not y['next_states']:
            return [state_distance, 0, 0, 0]

        #   动作区间交集奖励的最大差异
        max_reward_difference = self.get_reward_distance(x, y)

        #   后继状态分布的差异
        distribution_difference = self.compute_distribution_difference(x['probs'], y['probs'])

        #   动作的最大差异
        max_action_difference = self.get_action_distance(x, y)

        return [state_distance, max_reward_difference, distribution_difference, max_action_difference]

    #   获得4个距离的上界
    # def calculate_ranges(self, datas):
    #     """
    #     datas: 2d-array 传入的状态数组
    #     return: 1d-array 长度为4 每种距离最大值
    #     """
    #     # 初始化一个长度为4的数组，用于存储每种距离的最大值
    #     ranges = np.array([0 for i in range(4)])
    #     ranges_list = []

    #     # 定义计算每个数据点的函数
    #     def calculate_distances_for_data(data1):
    #         x = self.get_node1(data1)
    #         values = np.array([self.calculate_distances(x, self.get_node1(data2)) for data2 in datas])
    #         return np.max(values, axis=0)

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #         ranges_list = list(executor.map(calculate_distances_for_data, datas))

    #     # 堆叠所有的最大值
    #     ranges = np.vstack(ranges_list)

    #     # 从堆叠的ranges找到每种距离的最大值，并返回
    #     ret = np.max(ranges, axis=0)
    #     return ret

    #   获得4个距离的上界
    def calculate_ranges(self, datas):
        """
        datas: 2d-array 传入的状态数组
        return: 1d-array 长度为4 每种距离最大值
        """
        ranges = np.array([0 for i in range(4)])
        ranges_list = []

        #   每一次取datas中元素，计算与其它元素的四个距离，找最大值并堆叠到ranges
        for data1 in tqdm(datas):
            x = self.get_node1(data1)
            values = np.array([self.calculate_distances(x, self.get_node1(data2)) for data2 in datas])
            ranges_list.append(np.max(values, axis=0))

        ranges = np.vstack(ranges_list)
        #   从堆叠的ranges找最大值 返回
        ret = np.max(ranges, axis=0)
        return ret
        
    def compute_inertia(self, datas):
        """
        datas: 原始数据 2D 数组，与 'initialize' 输入相同。
        """
        inertia = 0
        centroids = self.get_centers()
        clusters = self.get_clusters()

        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - centroids[clusters], axis=2)

        # 使用矢量化操作计算惯性
        for k, item in enumerate(clusters):
            inertia += np.sum(distances[k, item])

        return inertia

    def pairwise_distance(self, datas):
        """
        datas: 输入数据 2D 数组。
        ret: 具有对称性质的距离矩阵，2D 数组 (n_sample, n_sample)。
        """
        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - datas, axis=2)

        # 返回对称距离矩阵
        return distances

    # 可以使用矢量化操作优化其他方法

    def transform(self, datas):
        """
        datas: 输入数据 2D 数组。
        ret: 距离矩阵 2D 数组 (n_sample, n_center)，表示每个输入数据到中心点的距离。
        """
        centroids = self.get_centers()

        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - centroids, axis=2)

        return distances


# 并行计算 pairwise_distance 方法
def parallel_pairwise_distance(args):
    i, datas, obj = args
    return obj.pairwise_distance(datas[i:i + 1, :])


if __name__ == '__main__':
    #   获得config
    path = "./conf/eval/highway_acc_eval.yaml"
    # path = "../../../conf/eval/highway_acc_eval.yaml"
    eval_config = utils.load_yml(path)

    #   获得图和状态
    """
    这些路径是不是有问题应该是相对于第二行的相对路径，即项目根目录相对路径
    比如"first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv"
    应该是"./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv"
    我记得之前改过 不知道是没保存还是版本回退了
    """
    graph, attr_dic, attr_dic_str = get_mdp_map("./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv", eval_config)
    # graph, attr_dic, attr_dic_str = get_mdp_map("../first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv", eval_config)
    
    # visualize_mdp(graph, 'mdp_pic/acc_td3/3_Node.png')

    graph_str_keys = {str(key): value for key, value in graph.items()}
    # 将字典保存到文件
    with open('./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_graph_k1520.json', 'w') as file:
        json.dump(graph_str_keys, file, indent=2)

    attr_dic_str_keys = {str(key): value for key, value in attr_dic.items()}
    # 将字典保存到文件
    with open('./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_attr_dic_k1520.json', 'w') as file:
        json.dump(attr_dic_str_keys, file, indent = 2)

    # # 保存字典到文件
    # with open('./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/Spatio_temporal_Kmeans.pkl', 'wb') as file:
    #     pickle.dump(graph, file)

    # # 保存字典到文件
    # with open('./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/Spatio_temporal_attr_dic.pkl', 'wb') as file:
    #     pickle.dump(attr_dic, file)

    # visualize_mdp(graph)
    states = get_mdp_states(graph)
    
    # optimal_k = find_optimal_k_elbow(states, min_clusters=1200, max_clusters=1500, 
    #                                  save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/acc_td3_clusters/elbow')

    kmeans_instance = SpatioTemporalKMeans_copy(config=eval_config, datas=states, graph=attr_dic, graph_str=attr_dic_str ,k=1520)
    #   进行聚类

    kmeans_instance.process()

    #   模型保存
    # joblib.dump(kmeans_instance, 'kmeans_model/acc_td3_risk/spatio_temporal_Kmeans.pkl')
    joblib.dump(kmeans_instance, './data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_Kmeans_k1520.pkl')
