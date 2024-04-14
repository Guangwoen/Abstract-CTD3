import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# 创建示例二维数据
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# 执行层次聚类
Z = hierarchy.linkage(data, method='ward', metric='euclidean')

# 绘制层次聚类结果的树状图
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()

# 预测新数据的类别
new_data = [[1, 2], [2.3, 4.5], [1.0, 3.5], [5.6, 6.7]]  # 新数据点
threshold = 3 # 阈值
labels = hierarchy.fcluster(Z, t=threshold, criterion='distance')
print(labels)
