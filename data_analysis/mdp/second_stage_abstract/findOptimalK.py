import math
import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class Canopy:
    def __init__(self, data):
        self.data = data
        self.t1 = 0
        self.t2 = 0

    # 设置初始阈值t1 和 t2
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print("t1 needs to be larger than t2!")

    # 欧式距离
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum())

    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return np.random.randint(len(self.data))

    # 核心算法
    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold t1 and t2!')
        else:
            canopies = []  # 用于存放最终归类的结果
            while len(self.data) != 0:
                # 获取一个随机下标
                rand_index = self.getRandIndex()
                # 随机获取一个中心点，定为P点
                current_center = self.data[rand_index]
                # 初始化P点的canopy类容器
                current_center_list = []
                # 初始化P点的删除容器
                delete_list = []
                # 删除随机选择的中心点P
                self.data = np.delete(self.data, rand_index, 0)

                for datum_j in range(len(self.data)):
                    datum = self.data[datum_j]
                    # 计算选取的中心点P到每个点之间的距离
                    distance = self.euclideanDistance(current_center, datum)
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        # 若小于t2则归入删除容器
                        delete_list.append(datum_j)
                self.data = np.delete(self.data, delete_list, 0)
                canopies.append((current_center, current_center_list))

                #   删除空的簇
                canopies = [cluster for cluster in canopies if len(cluster[1]) > 1]
            return canopies


def find_optimal_k_elbow(data, min_clusters=2, max_clusters=10, save_path=None):
    distortions = []
    for i in tqdm(range(min_clusters, max_clusters + 1, 100), desc="Finding optimal k"):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.plot(range(min_clusters, max_clusters + 1, 100), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on the elbow graph): "))
    return optimal_k

def find_optimal_k_silhouette(data, min_clusters=2, max_clusters=10, save_path=None):
    silhouette_scores = []
    for i in tqdm(range(min_clusters, max_clusters + 1, 50), desc="Finding optimal k"):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

    plt.plot(range(min_clusters, max_clusters + 1, 50), silhouette_scores, marker='o')
    plt.title('Silhouette Score Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on silhouette score): "))
    return optimal_k


def calculate_gap_statistic(data, min_clusters, k):
    reference_datasets = []
    for _ in range(5):  # 生成10个随机数据集作为参考
        random_data = np.random.rand(*data.shape)
        reference_datasets.append(random_data)

    gap_values = []
    for i in tqdm(range(min_clusters, k + 1, 100), desc="Finding optimal k"):
    # for i in range(min_clusters, k + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        log_inertia = np.log(kmeans.inertia_)

        reference_log_inertias = []
        for ref_data in reference_datasets:
            ref_kmeans = KMeans(n_clusters=i)
            ref_kmeans.fit(ref_data)
            reference_log_inertias.append(np.log(ref_kmeans.inertia_))

        gap = np.mean(reference_log_inertias) - log_inertia
        gap_values.append(gap)

    return gap_values


def find_optimal_k_gap(data, min_clusters=2, max_clusters=10, save_path=None):
    gap_values = calculate_gap_statistic(data, min_clusters, max_clusters)

    plt.plot(range(min_clusters, max_clusters + 1, 100), gap_values, marker='o')
    plt.title('Gap Statistic Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap Value')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on gap statistic): "))
    return optimal_k


def calculate_gap_statistic_new(data, min_clusters, k):
    reference_datasets = []
    for _ in range(5):  # 生成10个随机数据集作为参考
        random_data = np.random.rand(*data.shape)
        reference_datasets.append(random_data)

    gap_values = []
    for i in tqdm(range(min_clusters, k + 1, 100), desc="Finding optimal k"):
    # for i in range(min_clusters, k + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        log_inertia = np.log(kmeans.inertia_)

        reference_log_inertias = []
        for ref_data in reference_datasets:
            ref_kmeans = KMeans(n_clusters=i)
            ref_kmeans.fit(ref_data)
            reference_log_inertias.append(np.log(ref_kmeans.inertia_))

        gap = np.mean(reference_log_inertias) - log_inertia
        gap_values.append(gap)

    return gap_values


def find_optimal_k_gap_new(data, min_clusters=2, max_clusters=10, save_path=None):
    gap_values = calculate_gap_statistic_new(data, min_clusters, max_clusters)

    plt.plot(range(min_clusters, max_clusters + 1, 100), gap_values, marker='o')
    plt.title('Gap Statistic Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap Value')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    optimal_k = int(input("Enter the optimal number of clusters (based on gap statistic): "))
    return optimal_k

def find_optimal_k_canopy(data, t1, t2, save_path=None):
    gc = Canopy(data)
    gc.setThreshold(t1, t2)
    canopies = gc.clustering()
    return len(canopies)