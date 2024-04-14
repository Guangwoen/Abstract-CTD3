import csv

combined_data = []

with open('/data_analysis/td3_risk_acc_logs.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        rel_dis = float(row['rel_dis'])
        rel_speed = float(row['rel_speed'])
        next_rel_dis = float(row['next_rel_dis'])
        next_rel_speed = float(row['next_rel_speed'])

        combined_data.append([rel_dis, rel_speed])
        combined_data.append([next_rel_dis, next_rel_speed])

# 打印提取的值
for sublist in combined_data:
    print(sublist)
    # for values in sublist:
    #     print(values)
