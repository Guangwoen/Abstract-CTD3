import joblib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from data_analysis import utils_data


# 自定义距离度量函数
from sklearn.metrics import pairwise_distances


def custom_distance(x, y):
    # 计算自定义距离度量，这里以示例的曼哈顿距离为例
    return np.abs(x - y).sum()

# 创建示例二维数据
# data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
state = utils_data.get_state("td3_risk_acc_logs.csv")
data = np.array(state)

# 定义要聚类的簇数
num_clusters = 19727
# 计算自定义距离矩阵
distances = pairwise_distances(data, metric=custom_distance)
# 创建 K-means 模型并进行训练
kmeans = KMeans(n_clusters=num_clusters, init='random', algorithm='auto')
kmeans.fit(data)
# 保存模型
joblib.dump(kmeans, '../acc_td3/kmeans_model.pkl')

# 预测数据点的簇标签
labels = kmeans.labels_

# 获取聚类中心点的坐标
centroids = kmeans.cluster_centers_

# 绘制数据点和聚类中心
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red')
plt.show()

# 预测新数据点的簇
# new_data = np.array([[2.3, 4.5], [1.0, 3.5], [7.8, 9.0]])
# predicted_labels = kmeans.predict(new_data)
#
# print(predicted_labels)