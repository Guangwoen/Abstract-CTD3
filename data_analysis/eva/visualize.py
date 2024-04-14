import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import utils
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal import SpatioTemporalKMeans
from data_analysis.mdp.second_stage_abstract.second_abstract_spatio_temporal_copy import SpatioTemporalKMeans_copy
import pandas as pd

# from ..mdp.second_stage_abstract import findOptimalK


#   这一种有网格，但是刻度很密
def plt_grid_font1(eval_path, mdl_path, save_path=None):
    """
    eval_path:一阶段参数eval_config路径
    mdl_path:载入的pkl模型的路径
    save_path:保存的图片路径
    """
    eval_config = utils.load_yml(eval_path)
    mdl = joblib.load(mdl_path)

    #   eval信息
    dim = eval_config["dim"]["state_dim"]
    gran = eval_config["granularity"]["state_gran"]
    upperbound_ls = eval_config["upperbound"]["state_upperbound"]
    lowerbound_ls = eval_config["lowerbound"]["state_lowerbound"]

    #   每个维度画出来多少个区间
    num = [int((upperbound_ls[i] - lowerbound_ls[i]) / gran[i]) for i in range(dim)]

    #   画格子用的数组，因为pcolormesh应该输入的是网格中心点
    x_grid = np.linspace(lowerbound_ls[0] + (gran[0]/2), upperbound_ls[0] - (gran[0]/2), num[0])
    y_grid = np.linspace(lowerbound_ls[1] + (gran[1]/2), upperbound_ls[1] - (gran[1]/2), num[1])

    #   创建一个网格
    X, Y = np.meshgrid(x_grid, y_grid)

    #   生成状态用的数组
    x = np.linspace(lowerbound_ls[0], upperbound_ls[0], num[0]+1)
    y = np.linspace(lowerbound_ls[1], upperbound_ls[1], num[1]+1)

    # 记录每个网格预测的标签 涂色用
    Z = np.zeros_like(X)

    #   生成所有状态 并预测
    for i in range(num[0]):
        for j in range(num[1]):
            state = [x_grid[i], y_grid[j]]
            Z[j][i] = mdl.predict([state])[0]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(8, 8))

    #   生成画格子用的刻度
    ax.set_xticks(np.arange(min(x), max(x) + gran[0], gran[0]*10))
    ax.set_yticks(np.arange(min(y), max(y) + gran[0], gran[1]*5))
    ax.tick_params(axis='both', which='minor', bottom=False, left=False, labelbottom=False, labelleft=False)

    #   绘制格子上色
    c = ax.pcolormesh(X, Y, Z, cmap='viridis')
    plt.grid(True, linestyle='-', linewidth=0.5, color='black')
    plt.xlabel('rel_dis')
    plt.ylabel('rel_speed')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


#   这种用到系数一些的刻度，但是没有网格
def plt_grid_font2(eval_path, mdl_path):
    eval_config = utils.load_yml(eval_path)
    mdl = joblib.load(mdl_path)

    #   eval信息
    dim = eval_config["dim"]["state_dim"]
    gran = eval_config["granularity"]["state_gran"]
    upperbound_ls = eval_config["upperbound"]["state_upperbound"]
    lowerbound_ls = eval_config["lowerbound"]["state_lowerbound"]

    #   每个维度画出来多少个区间
    num = [int((upperbound_ls[i] - lowerbound_ls[i]) / gran[i]) for i in range(dim)]

    #   画格子用的数组，因为pcolormesh应该输入的是网格中心点
    x_grid = np.linspace(lowerbound_ls[0] + (gran[0]/2), upperbound_ls[0] - (gran[0]/2), num[0])
    y_grid = np.linspace(lowerbound_ls[1] + (gran[1]/2), upperbound_ls[1] - (gran[1]/2), num[1])

    #   创建一个网格
    X, Y = np.meshgrid(x_grid, y_grid)
    # print(X)

    #   生成状态用的数组
    x = np.linspace(lowerbound_ls[0], upperbound_ls[0], num[0]+1)
    y = np.linspace(lowerbound_ls[1], upperbound_ls[1], num[1]+1)

    # 记录每个网格预测的标签 涂色用
    Z = np.zeros_like(X)

    #   生成所有状态 并预测
    for i in range(num[0]):
        for j in range(num[1]):
            state = [x_grid[i], y_grid[j]]
            Z[i][j] = mdl.predict([state])[0]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(8, 8))

    #   显示出来的图的刻度
    grid_interval = 0.1
    ax.set_xticks(np.arange(min(x), max(x) + grid_interval, grid_interval))
    ax.set_yticks(np.arange(min(y), max(y) + grid_interval, grid_interval))
    ax.tick_params(axis='both', which='major', bottom=True, left=True, labelbottom=True, labelleft=True)

    #   绘制格子上色
    c = ax.pcolormesh(X, Y, Z, cmap='viridis')
    plt.grid(True, linestyle='-', which='minor', linewidth=0.5, color='black')

    plt.savefig("./font2.png")
    plt.show()

#   这种想要在有网格的同时还能保证刻度系数，但是有点线不显示，未知原因
def plt_grid_experiment(eval_path, mdl_path):
    eval_config = utils.load_yml(eval_path)
    mdl = joblib.load(mdl_path)

    #   eval信息
    dim = eval_config["dim"]["state_dim"]
    gran = eval_config["granularity"]["state_gran"]
    upperbound_ls = eval_config["upperbound"]["state_upperbound"]
    lowerbound_ls = eval_config["lowerbound"]["state_lowerbound"]

    #   每个维度画出来多少个区间
    num = [int((upperbound_ls[i] - lowerbound_ls[i]) / gran[i]) for i in range(dim)]

    #   画格子用的数组，因为pcolormesh应该输入的是网格中心点
    x_grid = np.linspace(lowerbound_ls[0] + (gran[0]/2), upperbound_ls[0] - (gran[0]/2), num[0])
    y_grid = np.linspace(lowerbound_ls[1] + (gran[1]/2), upperbound_ls[1] - (gran[1]/2), num[1])

    #   创建一个网格
    X, Y = np.meshgrid(x_grid, y_grid)
    # print(X)

    #   生成状态用的数组
    x = np.linspace(lowerbound_ls[0], upperbound_ls[0], num[0]+1)
    y = np.linspace(lowerbound_ls[1], upperbound_ls[1], num[1]+1)

    # 记录每个网格预测的标签 涂色用
    Z = np.zeros_like(X)

    #   生成所有状态 并预测
    for i in range(num[0]):
        for j in range(num[1]):
            state = [x_grid[i], y_grid[j]]
            Z[i][j] = mdl.predict([state])[0]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(8, 8))

    #   显示出来的图的刻度
    grid_interval = 0.1
    ax.set_xticks(np.arange(min(x), max(x) + grid_interval, grid_interval))
    ax.set_yticks(np.arange(min(y), max(y) + grid_interval, grid_interval))
    ax.tick_params(axis='both', which='major', bottom=True, left=True, labelbottom=True, labelleft=True)

    #   画网格的刻度
    ax.set_xticks(np.arange(min(x), max(x) + gran[0], gran[0]), minor=True)
    ax.set_yticks(np.arange(min(y), max(y) + gran[0], gran[1]), minor=True)
    ax.tick_params(axis='both', which='minor', bottom=False, left=False, labelbottom=False, labelleft=False)

    #   绘制格子上色
    c = ax.pcolormesh(X, Y, Z, cmap='viridis')
    plt.grid(True, linestyle='--', which='major', linewidth=0.1, color='white')
    plt.grid(True, linestyle='-', which='minor', linewidth=0.5, color='black')

    plt.savefig("./experiment.png")
    plt.show()


def render_raw_data(csv_path, save_path):
    df = pd.read_csv(csv_path)

    # 提取rel_dis和rel_speed列
    rel_dis = df['rel_dis']
    rel_speed = df['rel_speed']

    # 绘制散点图
    plt.scatter(rel_dis, rel_speed)
    plt.xlabel('rel_dis')
    plt.ylabel('rel_speed')
    # plt.title('Scatter Plot of rel_dis vs rel_speed')

    # 保存图形到文件（例如，PNG格式）
    plt.savefig(save_path)

if __name__ == '__main__':
    # eval_path = "../../conf/eval/highway_acc_eval.yaml"
    # mdl_path = "../../data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_elbow_Kmeans_k1520.pkl"
    # eval_path = "./conf/eval/highway_acc_eval.yaml"
    # mdl_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/spatio_temporal_Kmeans_k1520.pkl"

    # plt_grid_font1(eval_path, mdl_path, "acc_eva/abstract_rendering/acc_td3_spatio_gra_0.01_elbow.pdf")
    # plt_grid_font1(eval_path, mdl_path, "data_analysis/eva/acc_eva/abstract_rendering/acc_td3_spatio_gra_0.01_k_1520.png")

    # mdl_path = "./data_analysis/mdp/second_stage_abstract/kmeans_model/acc_td3_risk/Euclidean_kmeans_model_1520.pkl"
    # plt_grid_font1(eval_path, mdl_path, "data_analysis/eva/acc_eva/abstract_rendering/acc_td3_Euclidean_gra_0.01_k_1520.png")

    render_raw_data("../../data_analysis/acc_td3/dataset/td3_risk_acc_logs.csv", "../../data_analysis/eva/acc_eva/abstract_rendering/acc_raw_data.pdf")


    
