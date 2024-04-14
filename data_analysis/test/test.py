acc = []
reward = []
cost = []
acc_reward_cost = [(-0.08, -0.07), (1.17, 1.18), (0.0, 0.01), (-0.21, -0.2), (1.16, 1.17), (0.0, 0.01)]
num = 2

# 按周期将数据添加到列表中
cycle = len(acc_reward_cost) / num
print(cycle)
for index, data in enumerate(acc_reward_cost):
    if index % cycle == 0:
        acc.append(data)
    if index % cycle == 1:
        reward.append(data)
    if index % cycle == 2:
        cost.append(data)

print(acc)
print(reward)
print(cost)
