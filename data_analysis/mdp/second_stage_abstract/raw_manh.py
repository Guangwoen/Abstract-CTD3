import sys
import joblib
from sklearn.cluster import KMeans
import numpy as np
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")

from data_analysis import utils_data
from data_analysis.mdp.second_stage_abstract.findOptimalK import * 

def perform_kmeans(data, optimal_k):
    # 曼哈顿距离
    # 欧式距离（即L2范数）
    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(data)
    return kmeans

if __name__ =="__main__":
    """ ACC part  """
    # file_path = "./data_analysis/acc_td3/dataset/td3_risk_acc_logs.csv"
    # data1 = utils_data.get_csv_info(file_path, 1, 'rel_dis', 'rel_speed')
    # data1 = np.array(data1)
    # data2 = utils_data.get_csv_info(file_path, 1, 'next_rel_dis', 'next_rel_speed')
    # data2 = np.array(data2)

    # data = np.vstack((data1, data2))
    # print(data.shape)

    # # optimal_k = find_optimal_k_canopy(data, t1=0.006, t2=0.003, 
    # #                                 save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/acc_td3_clusters/gap_raw')

    # kmeans_model = perform_kmeans(data, 3400)

    # # # Save the KMeans model
    # model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/raw_gap_Kmeans.pkl'
    # joblib.dump(kmeans_model, model_save_path)


    """ LKA part  """
    # file_path = "./data_analysis/datasets/lka_td3/td3_risk_lka_logs.csv"
    # data1 = utils_data.get_csv_info(file_path, 1, 'y_dis', 'v_y', 'cos_h')
    # data1 = np.array(data1)
    # data2 = utils_data.get_csv_info(file_path, 1, 'next_y_dis', 'next_v_y', 'next_cos_h')
    # data2 = np.array(data2)

    # data = np.vstack((data1, data2))
    # print(data.shape)

    # optimal_k = find_optimal_k_canopy(data, t1=0.006, t2=0.003, 
    #                                 save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/lka_td3_risk/lka_td3_clusters/canopy_raw')
    # print(optimal_k)
    # kmeans_model = perform_kmeans(data, optimal_k)

    # # # Save the KMeans model
    # model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/lka_td3_risk/raw_canopy_Kmeans.pkl'
    # joblib.dump(kmeans_model, model_save_path)

    """ ICA part  """
    file_path = "./data_analysis/datasets/ica_dqn/dqn_risk_ica_logs.csv"
    data1 = utils_data.get_csv_info(file_path, 1, 'x1', 'y1', 'x2', 'y2')
    data1 = np.array(data1)
    data2 = utils_data.get_csv_info(file_path, 1, 'next_x1', 'next_y1', 'next_x2', 'next_y2')
    data2 = np.array(data2)

    #   堆叠并去重
    data = np.vstack((data1, data2))
    data = np.unique(data, axis=0)
    print(data.shape)

    optimal_k = find_optimal_k_gap(data, max_clusters=6000, min_clusters=2100, 
                                    save_path='./data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/ica_dqn_clusters/gap_raw')
    print(optimal_k)
    kmeans_model = perform_kmeans(data, optimal_k)

    # # Save the KMeans model
    model_save_path = './data_analysis/mdp/second_stage_abstract/kmeans_model/ica_dqn_risk/raw_gap_Kmeans.pkl'
    joblib.dump(kmeans_model, model_save_path)


    