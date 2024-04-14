import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import joblib
import matplotlib.pyplot as plt

# 自定义距离度量函数
def custom_distance(x, y):
    # 计算自定义距离度量，这里按照 |x1+x2+x3 - (y1+y2+y3)| 的方式计算
    return np.abs(np.sum(x) - np.sum(y))

# 创建示例三维数据
data = np.array([[1, 2, 3], [1.5, 1.8, 2.9], [5, 8, 7], [8, 8, 1], [1, 0.6, 2], [9, 11, 6]])

# 定义要聚类的簇数
num_clusters = 3

# 计算自定义距离矩阵
distances = pairwise_distances(data, metric=custom_distance)

# 创建 K-means 模型并进行训练
kmeans = KMeans(n_clusters=num_clusters, init='random', algorithm='auto')
kmeans.fit(data)

# 保存模型
joblib.dump(kmeans, 'kmeans_model.pkl')

# 预测数据点的簇标签
labels = kmeans.labels_

# 获取聚类中心点的坐标
centroids = kmeans.cluster_centers_

# 绘制数据点和聚类中心
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red')
plt.show()

# 预测新数据点的簇
new_data = np.array([[2.3, 4.5, 1.2], [1.0, 3.5, 2.0], [7.8, 9.0, 0.5]])
predicted_labels = kmeans.predict(new_data)

print(predicted_labels)
