# import numpy as np
# from sklearn.cluster import KMeans
#
# # 自定义距离函数（曼哈顿距离）
# from sklearn.metrics import pairwise_distances
#
#
# def custom_distance(x, y):
#     return np.sum(np.abs(x - y))
#
#
# # 自定义初始化方法
# class CustomInitializer:
#     def __init__(self, X):
#         self.X = X
#
#     def __call__(self, X, n_clusters, random_state=None):
#         # 计算样本之间的距离
#         distances = pairwise_distances(self.X, metric=custom_distance)
#         # 选择距离最大的样本作为初始聚类中心
#         max_distance_idx = np.argmax(np.sum(distances, axis=1))
#         centers = [self.X[max_distance_idx]]
#         # 随机选择其他聚类中心
#         for _ in range(1, n_clusters):
#             center = self.X[np.random.choice(self.X.shape[0])]
#             centers.append(center)
#         return np.array(centers)
#
#
# # 设置随机种子，以确保结果的可重复性
# np.random.seed(0)
#
# # 设置数据集的参数
# num_samples = 100  # 样本数量
# num_features = 2  # 特征数量
# num_clusters = 3  # 聚类数量
#
# # 生成数据集
# data = np.random.randn(num_samples, num_features)
#
# # 创建 K-means 模型并进行训练
# kmeans = KMeans(n_clusters=num_clusters, init=CustomInitializer(data), algorithm='auto')
# kmeans.fit(data)
#
# # 获取聚类结果
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# print("数据集示例：")
# print(data)
# print("\n聚类标签：")
# print(labels)
# print("\n聚类中心点：")
# print(centroids)


import numpy as np
from sklearn.cluster import KMeans


class CustomKMeans(KMeans):
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None):
        """
        n_clusters：聚类的数量，默认为8。
        init：初始化聚类中心的方法，默认为'k-means++'，表示使用k-means++算法。
        max_iter：最大迭代次数，默认为300。
        tol：收敛阈值，默认为1e-4，表示算法的迭代在误差小于该阈值时停止。
        random_state：随机种子，默认为None。用于控制随机数生成过程的随机性。
        super().__init__语句用于调用父类（即KMeans类）的构造函数，并传递相应的参数。通过这种方式，我们可以继承KMeans类的属性和方法，并对其进行自定义扩展。
        """
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, tol=tol, random_state=random_state)

    # 用于计算数据集中每个样本与各个聚类中心之间的自定义距离。
    def _transform(self, X):
        """
        X：表示数据集，是一个二维数组，形状为(n_samples, n_features)。
        """
        # 自定义距离函数
        distances = np.zeros((X.shape[0], self.n_clusters))
        """
        对每个聚类中心，使用列表推导式遍历数据集中的每个样本 x，并调用自定义的距离计算方法 self.distance(x, self.cluster_centers_[i])，
        将计算出的距离存储在 distances 数组中的相应位置。
        """
        for i in range(self.n_clusters):
            distances[:, i] = np.array([self.distance(x, self.cluster_centers_[i]) for x in X])
        return distances

    def distance(self, x, y):
        # 自定义距离计算方法
        return np.sqrt((x.x - y.x) ** 2 + (x.y - y.y) ** 2)


# 自定义数据点类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 创建数据点
points = [Point(1, 2), Point(3, 4), Point(5, 6), Point(7, 8), Point(9, 10)]
data = np.array([[p.x, p.y] for p in points])

# 创建自定义的K-means模型并进行训练
num_clusters = 2
kmeans = CustomKMeans(n_clusters=num_clusters)
kmeans.fit(data)

# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("数据点集合：")
for point in points:
    print(f"({point.x}, {point.y})")

print("\n聚类标签：")
print(labels)

print("\n聚类中心点：")
for centroid in centroids:
    print(f"({centroid[0]}, {centroid[1]})")
