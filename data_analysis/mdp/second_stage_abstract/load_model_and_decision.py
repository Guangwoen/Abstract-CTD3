import joblib
from second_abstract_spatio_temporal import SpatioTemporalKMeans
from data_analysis.mdp.MDP import Node, Edge, Attr
import pickle
from second_abstract_spatio_temporal import get_mdp_states
import numpy as np


def get_action(state, Spatio_temporal_attr_dic):
    attr = Spatio_temporal_attr_dic[str(tuple(state))]
    chosen_index = np.random.choice(len(attr['actions']), p=attr['probs'])
    chosen_action = attr['actions'][chosen_index]
    return chosen_action


if __name__ == '__main__':
    #   加载模型
    mdl = joblib.load('kmeans_model/acc_td3_risk/Spatio_temporal_Kmeans.pkl')
    # 加载文件中的字典
    with open('kmeans_model/acc_td3_risk/Spatio_temporal_attr_dic.pkl', 'rb') as file:
        Spatio_temporal_attr_dic = pickle.load(file)

    with open('kmeans_model/acc_td3_risk/Spatio_temporal_graph.pkl', 'rb') as file:
        Spatio_temporal_graph = pickle.load(file)

    #   获取中心点，但此时是没有修正过的
    centers = mdl.get_centers()
    print("中心：", centers)
    #   中心点修正
    revised_centers = mdl.revised_centers(centers)
    print("修正后的中心：", revised_centers)

    states = get_mdp_states(Spatio_temporal_graph)
    label = mdl.predict(states)
    print("预测的簇标签：", label)
    centroid = revised_centers[label[0]]
    print("预测的中心点：", centroid)
    # 使用模型进行动作的选择
    chosen_action = get_action(centroid, Spatio_temporal_attr_dic)
    print("Chosen Action:", chosen_action)



