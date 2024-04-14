import joblib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
import sys

# 自定义距离度量函数
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import jensenshannon

from data_analysis.utils_data import get_csv_info
from tool.tools import data_prcss

path1 = os.path.abspath('../tool/tools.py')
path2 = os.path.abspath('../data_analysis/utils_data.py')


# 用于计算交集
class MySet:
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def getMin(self):
        return self.mins

    def getMax(self):
        return self.maxs


def getIntersection(l1, l2):  # 求交集方法
    result = []  # 用来存储l1和l2的所有交集

    # 对输入的两个列表进行排序
    l1.sort(key=lambda x: x.getMin())
    l2.sort(key=lambda x: x.getMin())

    i = 0
    j = 0
    while i < len(l1) and j < len(l2):
        s1 = l1[i]  # 在方法里调用MySet类
        s2 = l2[j]
        if s1.getMin() < s2.getMin():
            if s1.getMax() < s2.getMin():  # 第一种时刻，交集为空，不返回
                i += 1
            elif s1.getMax() <= s2.getMax():  # 第二种时刻
                result.append(MySet(s2.getMin(), s1.getMax()))
                i += 1
            else:  # 第三种时刻第二种情况
                result.append(MySet(s2.getMin(), s2.getMax()))
                j += 1
        elif s1.getMin() <= s2.getMax():
            if s1.getMax() <= s2.getMax():  # 第三种时刻第一种情况
                result.append(MySet(s1.getMin(), s1.getMax()))
                i += 1
            else:  # 第四种时刻
                result.append(MySet(s1.getMin(), s2.getMax()))
                j += 1
        else:  # 第五种时刻
            j += 1
    return result


def is_equal(a, b, tol=None):
    if tol is None:
        tol = sys.float_info.epsilon
    return abs(a - b) < tol


"""
传入的数据类型：前8位表示状态和次态区间
接下来按照7位一组；分别表示动作区间，奖励区间，done，cost区间
直到出现整数，表示一共出现了多少次，以及一个概率
"""


def custom_distance(x, y):
    # 如果两个状态相同，返回0
    if (x == y)[0]:
        return 0
    cd = 0.5
    cp = 0.5
    lx, ly = [], []  # 统计动作区间
    reward_x, reward_y = [], []  # 统计奖励区间
    prob_x, prob_y = [], []  # 统计概率
    """获取x，y的动作区间，求交集"""
    temp_i = 0
    max_reward_difference = 0

    # 从一维列表中，提取动作，奖励
    for i in range(len(x)):
        if i - 1 - temp_i != 0 and (i - 1 - temp_i) % 7 == 0 and x[i] != 100:
            if (i < len(x) and i + 1 < len(x) and not is_equal(x[i + 1], x[i] + 0.01, tol=1e-8)) or (
                    i + 3 < len(x) and i + 2 < len(x) and not is_equal(x[i + 3], x[i + 2] + 0.01, tol=1e-8)):
                prob_x.append(x[i + 1])
                temp_i = i + 2
                continue
            if i < len(x) and i + 1 < len(x):
                lx.append(MySet(x[i], x[i + 1]))
                reward_x.append((x[i + 2] + x[i + 3]) / 2)
    temp_i = 0
    for i in range(len(y)):
        if i - 1 - temp_i != 0 and (i - 1 - temp_i) % 7 == 0 and y[i] != 100:
            if (i + 1 < len(y) and not is_equal(y[i + 1], y[i] + 0.01, tol=1e-8)) or (
                    i + 3 < len(y) and not is_equal(y[i + 3], y[i + 2] + 0.01, tol=1e-8)):
                prob_y.append(y[i + 1])
                temp_i = i + 2
                continue
            if i < len(y) and i + 1 < len(y):
                ly.append(MySet(y[i], y[i + 1]))
                reward_y.append((y[i + 2] + y[i + 3]) / 2)
    result = getIntersection(lx, ly)  # 动作区间交集

    actions = []  # 动作区间交集中点
    actions_x = []  # x动作区间的中点
    actions_y = []  # y动作区间的中点
    # 获得所有的动作
    for i in range(len(lx)):
        actions_x.append((lx[i].getMin() + lx[i].getMax()) / 2)
    for i in range(len(ly)):
        actions_y.append((ly[i].getMin() + ly[i].getMax()) / 2)
    for i in range(len(result)):
        actions.append((result[i].getMax() + result[i].getMin()) / 2)

    # 奖励之间的距离——根据并集中的动作寻找x和y中对应的区间以及对应的奖励
    for action in actions:
        for j in range(len(lx)):
            if lx[j].getMin() < action < lx[j].getMax():
                for k in range(len(ly)):
                    if ly[k].getMin() < action < ly[k].getMax():
                        max_reward_difference = max((reward_x[j] + reward_y[k]) / 2, max_reward_difference)

    # 状态本身之间的距离
    state_distance = np.sqrt(np.square((y[0] + y[1] - x[0] - x[1]) / 2) + np.square((y[2] + y[3] - x[2] - x[3]) / 2))

    # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
    max_len = max(len(prob_x), len(prob_y))
    if len(prob_x) == max_len:
        for _ in range(max_len - len(prob_x)):
            prob_x.append(0)
    else:
        for _ in range(max_len - len(prob_y)):
            prob_y.append(0)
    distribution_difference = jensenshannon(prob_x, prob_y)

    # 动作之间的距离——寻找动作之间最大的差值
    max_action_difference = 0
    for dist_x in actions_x:
        for dist_y in actions_y:
            max_action_difference = max(max_action_difference, abs(dist_x - dist_y))

    return state_distance + max_reward_difference + cd * max_action_difference + cp * distribution_difference


# 创建示例二维数据
# data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
data_processor = data_prcss(9, 2, 1, 1, 1, (1e-3, 1e-3), (1e-3,), (1e-3,), (1e-3,), (1e-1, 1e-1), (1e-2,), (1e-2,),
                            (1e-2,),
                            (1, 1), (-1, -1), (1,), (-1,), (2,), (0,), (100,), (0,))
data_processor.read_in("F:\桌面\Abstract-CTD3-main-master\\tool\\test.csv")
data_processor.process()
data = data_processor.get_data()
max_len = 0
data_using = []

"""获取最大的长度"""
for i in range(len(data)):
    temp = []
    for line in data[i]:
        temp += line
    data_using.append(temp)

"""创建新的data，供custom_distance调用"""
for i in range(len(data_using)):
    max_length = max(len(row) for row in data_using)  # 获取最大长度
    # 使用列表推导式对每一行进行填充
    data_filled = [row + [100] * (max_length - len(row)) for row in data_using]

# 定义要聚类的簇数
num_clusters = 3
# 计算自定义距离矩阵
distances = pairwise_distances(data_filled, metric=custom_distance)
print("distances\n", distances)
# 创建 K-means 模型并进行训练
kmeans = KMeans(n_clusters=num_clusters, init='random', algorithm='auto')
kmeans.fit(data_filled)
# 保存模型
joblib.dump(kmeans, 'kmeans_model.pkl')

# 预测数据点的簇标签
labels = kmeans.labels_

# 获取聚类中心点的坐标
centroids = kmeans.cluster_centers_

print(data_filled)
data_filled = np.array(data_filled)
# 绘制数据点和聚类中心
plt.scatter(data_filled[:, 0], data_filled[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red')
plt.show()

print("show()")

# 预测新数据点的簇
new_data = np.array([[2.3, 4.5], [1.0, 3.5], [7.8, 9.0]])
predicted_labels = kmeans.predict(new_data)

print(predicted_labels)
