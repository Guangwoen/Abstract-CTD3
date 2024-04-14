import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from data_analysis import utils_data

# 数据集为 X
X = np.array(utils_data.get_csv_info('../acc_td3/td3_risk_acc_logs.csv', 15, 'rel_dis', 'rel_speed'))
# 设置聚类个数的范围
min_clusters = 2
max_clusters = 20

# 初始化列表来保存每个聚类个数对应的轮廓系数
silhouette_scores = []

# 计算每个聚类个数对应的轮廓系数
for n_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# 绘制肘方法图像
plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Elbow Method')
plt.show()

# 根据肘方法选择最佳聚类个数
best_n_clusters = np.argmax(silhouette_scores) + min_clusters

# 使用最佳聚类个数重新训练模型
best_kmeans = KMeans(n_clusters=best_n_clusters)
best_kmeans.fit(X)

# 保存模型
model_path = 'kmeans_model.pkl'
joblib.dump(best_kmeans, model_path)

# 预测数据点的簇标签
labels = best_kmeans.labels_

# 获取聚类中心点的坐标
centroids = best_kmeans.cluster_centers_

# 绘制数据点和聚类中心
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red')
plt.show()