import csv
import re
import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")

def log2csv(log_file, csv_file):
    lines = 0
    def is_valid_string(input_string):
        pattern = r'^[-0-9 .]+$'
        return re.match(pattern, input_string) is not None
    
    with open(log_file, 'r', encoding='utf8') as file:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=',')

            # 写入CSV文件的标题行
            headline = file.readline()
            headline = headline.strip('\n').split(' ')
            writer.writerow(headline)
            # 逐行读取日志文件并写入CSV文件
            for line in file:
                # 解析日志文件中的数据
                data = line.strip().replace('[', '').replace(']', '').split(', ')
                if len(data) >= 2 and is_valid_string(data[1]):
                    lines += 1
                    write_line = []
                    for item in data:
                        item = item.strip(' ')
                        if ' ' in item:
                            ls = item.split(' ')
                            for it in ls:
                                if it != '':
                                    write_line.append(it)
                        else:
                            write_line.append(item)
                    print(write_line)
                    writer.writerow(write_line)
                # rel_state_values = data[0].split(' ')
                # rel_next_state_values = data[3].split(' ')
                #
                # # 将数据写入CSV文件
                # writer.writerow(rel_state_values + data[1:3] + rel_next_state_values + data[5:])
    return lines


if __name__ == '__main__':
    log_file_train = './my_log_ICA.log'
    csv_file_train = "./data_analysis/datasets/ica_dqn/dqn_risk_ica_logs.csv"
    

    lines = log2csv(log_file_train, csv_file_train)
    print(lines)
