import cupy as cp
from cupyx.sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# 自定义距离计算函数
def custom_distance(x, y):
    # 自定义距离计算方式
    return cp.abs(x - y)  # 以绝对差作为距离度量

# 创建数据
X = cp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=cp.float32)

# 使用肘方法确定最佳聚类个数
max_clusters = 10
losses = []
for n_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    loss = kmeans.inertia_
    losses.append(loss)

# 绘制聚类个数与损失函数值的关系图
plt.plot(range(2, max_clusters + 1), losses, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Loss')
plt.title('Elbow Method')
plt.show()

# 选择最佳聚类个数
best_n_clusters = int(input("Enter the best number of clusters: "))

# 使用GPU进行K-means聚类
kmeans = KMeans(n_clusters=best_n_clusters)
kmeans.fit(X)

# 保存最佳聚类个数模型
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# 对新数据进行预测
new_data = cp.array([[2, 3], [6, 7]], dtype=cp.float32)
predictions = kmeans.predict(new_data)

print("Predictions:", predictions)
