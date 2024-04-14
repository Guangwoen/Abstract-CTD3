import sys
sys.path.append("/hdd1/akihi/Abstract-CTD3-main-master")
import numpy as np
import csv
import math
import ast
from data_analysis import utils_data as ud
import utils

def check_dimensions(dim1, dim2, dim3, dim4):
    """
        检查四个维度是否一致，如果不一致则抛出 ValueError 异常。

        Parameters:
        - dim1, dim2, dim3, dim4 (int): 四个维度的值。

        Raises:
        - ValueError: 如果维度不一致。
    """
    if dim1 != dim2 or dim1 != dim3 or dim1 != dim4:
        raise ValueError("维度不一致")


def get_abstract_id(state, gra, low_bound, up_bound):
    """
    得到状态id/动作id的值，给定具体数据，得到id。根据id可以判断属于哪个状态，id（1，2，3） 和状态的维度一致。

    Parameters:
    - state (List[float]): 具体状态值。
    - gra (List[float]): 状态粒度。
    - low_bound (List[float]): 状态的下界。
    - up_bound (List[float]): 状态的上界。

    Returns:
    - Tuple[List[int], List[float]]: 返回状态id和mod。
    """
    dim_state = len(state)
    check_dimensions(dim_state, len(gra), len(low_bound), len(up_bound))

    abstract_id = []
    mod = []

    for x_i in range(dim_state):
        if state[x_i] < low_bound[x_i]:
            abstract_id.append(0)
            mod.append(state[x_i] - low_bound[x_i])
        elif state[x_i] > up_bound[x_i]:
            abstract_id.append(int((up_bound[x_i] - low_bound[x_i]) / gra[x_i]) - 1)
            mod.append(state[x_i] - up_bound[x_i] + gra[x_i])
        else:
            abstract_id.append(int((state[x_i] - low_bound[x_i]) / gra[x_i]))
            mod.append((state[x_i] - low_bound[x_i]) % gra[x_i])

    return abstract_id, mod


def get_raw_data_by_id(abstract_id, mod, gra, low_bound):
    """
        根据状态id和mod，得到具体数据。

        Parameters:
        - abstract_id (List[int]): 状态id。
        - mod (List[float]): mod。
        - gra (List[float]): 粒度。
        - low_bound (List[float]): 下界。

        Returns:
        - List[float]: 返回具体数据。
    """
    dim_abstract_id = len(abstract_id)
    check_dimensions(dim_abstract_id, len(mod), len(gra), len(low_bound))

    raw_data = [gra[x_i] * abstract_id[x_i] + low_bound[x_i] + mod[x_i] for x_i in range(dim_abstract_id)]
    return raw_data


def get_center_data_by_id(abstract_id, mod, gra, low_bound):
    """
        根据状态id和mod，得到簇的中心数据。

        Parameters:
        - abstract_id (List[int]): 状态id。
        - mod (List[float]): mod。
        - gra (List[float]): 粒度。
        - low_bound (List[float]): 下界。

        Returns:
        - List[float]: 返回簇的中心数据。
    """
    dim_abstract_id = len(abstract_id)
    check_dimensions(dim_abstract_id, len(mod), len(gra), len(low_bound))

    center_data = [gra[x_i] * abstract_id[x_i] + low_bound[x_i] + gra[x_i] / 2 for x_i in range(dim_abstract_id)]
    return center_data


class FirstAbstract:
    def __init__(self, col_num,
                 state, act, reward, next_state, done, cost,
                 state_gra, act_gra, rwd_gra, cost_gra,
                 state_low_bound, state_up_bound, act_low_bound, act_up_bound,
                 rwd_low_bound, rwd_up_bound, cost_low_bound, cost_up_bound):
        """
        初始化对象。

        Parameters:
        - col_num (int): 列数。
        - state (List[str]): 状态的不同维度的名称。
        - act (List[str]): 动作的不同维度的名称。
        - reward (List[str]): 奖励的名称。
        - next_state (List[str]): 下一个状态的不同维度的名称。
        - done (List[str]): 表示该情节是否结束的标志。
        - cost (List[str]): 成本的名称。
        - state_gra (List[float]): 状态的粒度。
        - act_gra (List[float]): 动作的粒度。
        - rwd_gra (List[float]): 奖励的粒度。
        - cost_gra (List[float]): 成本的粒度。
        - state_up_bound (List[float]): 状态的上界。
        - state_low_bound (List[float]): 状态的下界。
        - act_up_bound (List[float]): 动作的上界。
        - act_low_bound (List[float]): 动作的下界。
        - rwd_up_bound (List[float]): 奖励的上界。
        - rwd_low_bound (List[float]): 奖励的下界。
        - cost_up_bound (List[float]): 成本的上界。
        - cost_low_bound (List[float]): 成本的下界。
        """
        self.col_num = col_num
        self.state = state
        self.act = act
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.cost = cost
        self.state_gra = state_gra
        self.act_gra = act_gra
        self.rwd_gra = rwd_gra
        self.cost_gra = cost_gra
        self.state_up_bound = state_up_bound
        self.state_low_bound = state_low_bound
        self.act_up_bound = act_up_bound
        self.act_low_bound = act_low_bound
        self.rwd_up_bound = rwd_up_bound
        self.rwd_low_bound = rwd_low_bound
        self.cost_up_bound = cost_up_bound
        self.cost_low_bound = cost_low_bound

    def interval_mdp(self, file_name, output_file):
        """
                对MDP文件进行区间抽象，并将结果写入输出文件。

                Parameters:
                - file_name (str): 输入MDP文件的文件名。
                - output_file (str): 输出文件的文件名。
        """
        with open(file_name, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)

            with open(output_file, 'w', newline='') as output_csv:
                csv_writer = csv.writer(output_csv)
                csv_writer.writerow(['StateID', 'ActID', 'RewardID', 'NextStateID', 'Done', 'CostID',
                                     'StateIdMod', 'ActIdMod', 'RewardIdMod', 'NextStateIdMod', 'CostIdMod'])

                for row in csv_reader:
                    abstract_row = []
                    state = ud.get_designated_data(row, header, self.state)
                    state_id, state_mod = get_abstract_id(state, self.state_gra, self.state_low_bound,
                                                          self.state_up_bound)
                    abstract_row.append(state_id)

                    act = ud.get_designated_data(row, header, self.act)
                    act_id, act_mod = get_abstract_id(act, self.act_gra, self.act_low_bound, self.act_up_bound)
                    abstract_row.append(act_id)

                    reward = ud.get_designated_data(row, header, self.reward)
                    reward_id, reward_mod = get_abstract_id(reward, self.rwd_gra, self.rwd_low_bound, self.rwd_up_bound)
                    abstract_row.append(reward_id)

                    next_state = ud.get_designated_data(row, header, self.next_state)
                    next_state_id, next_state_mod = get_abstract_id(next_state, self.state_gra, self.state_low_bound,
                                                                    self.state_up_bound)
                    abstract_row.append(next_state_id)

                    done = ud.get_designated_data(row, header, self.done)
                    abstract_row.append(done)

                    cost = ud.get_designated_data(row, header, self.cost)
                    cost_id, cost_mod = get_abstract_id(cost, self.cost_gra, self.cost_low_bound, self.cost_up_bound)
                    abstract_row.append(cost_id)

                    abstract_row.append(state_mod)
                    abstract_row.append(act_mod)
                    abstract_row.append(reward_mod)
                    abstract_row.append(next_state_mod)
                    abstract_row.append(cost_mod)

                    csv_writer.writerow(abstract_row)

    def del_duplicate_row_and_cal_pro(self, input_file, output_file, output_file_raw_data, output_file_center_data):
        """
            从MDP文件中删除重复的行，并计算概率。

            Parameters:
            - input_file (str): 输入MDP文件的文件名。
            - output_file (str): 输出文件的文件名。
        """
        data_input = []
        data_to_save = []
        duplicate_indices = {}
        need_to_del = []

        # 读取CSV文件内容
        with open(input_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for index, row in enumerate(reader):
                state_id = row['StateID']
                next_state_id = row['NextStateID']
                key = f"{state_id}_{next_state_id}"

                if key in duplicate_indices:
                    duplicate_indices[key].append(index)
                else:
                    duplicate_indices[key] = [index]

                data_input.append(row)

        # 删除相同的行，并插入新数据，计算概率
        for indices in duplicate_indices.values():
            # indices.sort()
            if len(indices) >= 2:
                state_new = []
                act_new = []
                reward_new = []
                next_state_new = []
                cost_new = []
                done = []

                for index in range(len(indices)):
                    need_to_del.append(indices[index])
                    # del data[index]
                    state_id = [float(element) for element in ast.literal_eval(data_input[indices[index]]['StateID'])]
                    act_id = [float(element) for element in ast.literal_eval(data_input[indices[index]]['ActID'])]
                    reward_id = [float(element) for element in ast.literal_eval(data_input[indices[index]]['RewardID'])]
                    next_state_id = [float(element) for element in
                                     ast.literal_eval(data_input[indices[index]]['NextStateID'])]
                    done = ast.literal_eval(data_input[indices[index]]['Done'])
                    cost_id = [float(element) for element in ast.literal_eval(data_input[indices[index]]['CostID'])]
                    state_mod = [float(element) for element in
                                 ast.literal_eval(data_input[indices[index]]['StateIdMod'])]
                    act_mod = [float(element) for element in ast.literal_eval(data_input[indices[index]]['ActIdMod'])]
                    reward_mod = [float(element) for element in
                                  ast.literal_eval(data_input[indices[index]]['RewardIdMod'])]
                    next_state_mod = [float(element) for element in
                                      ast.literal_eval(data_input[indices[index]]['NextStateIdMod'])]
                    cost_mod = [float(element) for element in ast.literal_eval(data_input[indices[index]]['CostIdMod'])]

                    # 原始数据
                    state = get_raw_data_by_id(state_id, state_mod, self.state_gra, self.state_low_bound)
                    act = get_raw_data_by_id(act_id, act_mod, self.act_gra, self.act_low_bound)
                    reward = get_raw_data_by_id(reward_id, reward_mod, self.rwd_gra, self.rwd_low_bound)
                    next_state = get_raw_data_by_id(next_state_id, next_state_mod, self.state_gra, self.state_low_bound)
                    cost = get_raw_data_by_id(cost_id, cost_mod, self.cost_gra, self.cost_low_bound)

                    if index == 0:
                        state_new = state
                        act_new = act
                        reward_new = reward
                        next_state_new = next_state
                        cost_new = cost
                    else:
                        for idx in range(len(state)):
                            state_new[idx] = ((state_new[idx] + state[idx]) / 2)

                        for idx in range(len(act)):
                            act_new[idx] = (act_new[idx] + act[idx]) / 2

                        for idx in range(len(reward)):
                            reward_new[idx] = (reward_new[idx] + reward[idx]) / 2

                        for idx in range(len(next_state)):
                            next_state_new[idx] = (next_state_new[idx] + next_state[idx]) / 2

                        for idx in range(len(cost)):
                            cost_new[idx] = (cost_new[idx] + cost[idx]) / 2

                state_new_id, state_new_mod = get_abstract_id(state_new, self.state_gra, self.state_low_bound,
                                                              self.state_up_bound)
                act_new_id, act_new_mod = get_abstract_id(act_new, self.act_gra, self.act_low_bound, self.act_up_bound)
                reward_new_id, reward_new_mod = get_abstract_id(reward_new, self.rwd_gra, self.rwd_low_bound,
                                                                self.rwd_up_bound)
                next_state_new_id, next_state_new_mod = get_abstract_id(next_state_new, self.state_gra,
                                                                        self.state_low_bound, self.state_up_bound)
                cost_new_id, cost_new_mod = get_abstract_id(cost_new, self.cost_gra, self.cost_low_bound,
                                                            self.cost_up_bound)

                new_row = {
                    'StateID': str(state_new_id),
                    'ActID': str(act_new_id),
                    'RewardID': str(reward_new_id),
                    'NextStateID': str(next_state_new_id),
                    'Done': str(done),
                    'CostID': str(cost_new_id),
                    'StateIdMod': str(state_new_mod),
                    'ActIdMod': str(act_new_mod),
                    'RewardIdMod': str(reward_new_mod),
                    'NextStateIdMod': str(next_state_new_mod),
                    'CostIdMod': str(cost_new_mod),
                    'Weight': len(indices)
                }
                data_to_save.insert(len(data_to_save) + 1, new_row)
            else:
                new_row = {
                    'StateID': str(data_input[indices[0]]['StateID']),
                    'ActID': str(data_input[indices[0]]['ActID']),
                    'RewardID': str(data_input[indices[0]]['RewardID']),
                    'NextStateID': str(data_input[indices[0]]['NextStateID']),
                    'Done': str(data_input[indices[0]]['Done']),
                    'CostID': str(data_input[indices[0]]['CostID']),
                    'StateIdMod': str(data_input[indices[0]]['StateIdMod']),
                    'ActIdMod': str(data_input[indices[0]]['ActIdMod']),
                    'RewardIdMod': str(data_input[indices[0]]['RewardIdMod']),
                    'NextStateIdMod': str(data_input[indices[0]]['NextStateIdMod']),
                    'CostIdMod': str(data_input[indices[0]]['CostIdMod']),
                    'Weight': len(indices)
                }

                data_to_save.insert(len(data_to_save) + 1, new_row)

        # need_to_del.sort(reverse=True)
        # data_result = [row for i, row in enumerate(data_to_save) if i not in need_to_del]

        # 计算概率
        transition_probabilities = {}
        for row in data_to_save:
            state_id = row['StateID']
            next_state_id = row['NextStateID']
            weight = row['Weight']

            # 更新或添加概率
            if state_id in transition_probabilities:
                if next_state_id in transition_probabilities[state_id]:
                    transition_probabilities[state_id][next_state_id] += weight
                else:
                    transition_probabilities[state_id][next_state_id] = weight
            else:
                transition_probabilities[state_id] = {next_state_id: weight}

        # 归一化权重，计算概率
        for state_id, successors in transition_probabilities.items():
            total_weight = sum(successors.values())
            for next_state_id, weight in successors.items():
                transition_probabilities[state_id][next_state_id] = weight / total_weight

        for row in data_to_save:
            state_id = row['StateID']
            next_state_id = row['NextStateID']
            row['Probability'] = transition_probabilities[state_id][next_state_id]

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['StateID', 'ActID', 'RewardID', 'NextStateID', 'Done', 'CostID',
                          'StateIdMod', 'ActIdMod', 'RewardIdMod', 'NextStateIdMod', 'CostIdMod', 'Weight',
                          'Probability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data_to_save:
                writer.writerow(row)

        # 将id恢复为原始数据
        fieldnames = ['State', 'Act', 'Reward', 'NextState', 'Done', 'Cost', 'Weight', 'Probability']

        # 将id修改为中心数据
        fieldnames_center = ['State_Center', 'Act_Center', 'Reward_Center', 'NextState_Center', 'Done', 'Cost_Center',
                      'Weight', 'Probability']

        with open(output_file_raw_data, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in data_to_save:
                state_id = ast.literal_eval(row['StateID'])
                act_id = ast.literal_eval(row['ActID'])
                reward_id = ast.literal_eval(row['RewardID'])
                next_state_id = ast.literal_eval(row['NextStateID'])
                done = ast.literal_eval(row['Done'])
                cost_id = ast.literal_eval(row['CostID'])
                state_id_mod = ast.literal_eval(row['StateIdMod'])
                act_id_mod = ast.literal_eval(row['ActIdMod'])
                reward_id_mod = ast.literal_eval(row['RewardIdMod'])
                next_state_id_mod = ast.literal_eval(row['NextStateIdMod'])
                cost_id_mod = ast.literal_eval(row['CostIdMod'])
                weight = row['Weight']
                probability = row['Probability']

                state_val = get_raw_data_by_id(state_id, state_id_mod, self.state_gra, self.state_low_bound)
                act_val = get_raw_data_by_id(act_id, act_id_mod, self.act_gra, self.act_low_bound)
                reward_val = get_raw_data_by_id(reward_id, reward_id_mod, self.rwd_gra, self.rwd_low_bound)
                next_state_val = get_raw_data_by_id(next_state_id, next_state_id_mod, self.state_gra,
                                                    self.state_low_bound)
                done_val = bool(done[0])
                cost_val = get_raw_data_by_id(cost_id, cost_id_mod, self.cost_gra, self.cost_low_bound)

                # 将数据写入CSV文件
                writer.writerow({
                    'State': state_val,
                    'Act': act_val,
                    'Reward': reward_val,
                    'NextState': next_state_val,
                    'Done': done_val,
                    'Cost': cost_val,
                    'Weight': weight,
                    'Probability': probability
                })

        with open(output_file_center_data, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_center)
            writer.writeheader()

            for row in data_to_save:
                state_id = ast.literal_eval(row['StateID'])
                act_id = ast.literal_eval(row['ActID'])
                reward_id = ast.literal_eval(row['RewardID'])
                next_state_id = ast.literal_eval(row['NextStateID'])
                done = ast.literal_eval(row['Done'])
                cost_id = ast.literal_eval(row['CostID'])
                state_id_mod = ast.literal_eval(row['StateIdMod'])
                act_id_mod = ast.literal_eval(row['ActIdMod'])
                reward_id_mod = ast.literal_eval(row['RewardIdMod'])
                next_state_id_mod = ast.literal_eval(row['NextStateIdMod'])
                cost_id_mod = ast.literal_eval(row['CostIdMod'])
                weight = row['Weight']
                probability = row['Probability']

                state_center = get_center_data_by_id(state_id, state_id_mod, self.state_gra, self.state_low_bound)
                act_center = get_center_data_by_id(act_id, act_id_mod, self.act_gra, self.act_low_bound)
                reward_center = get_center_data_by_id(reward_id, reward_id_mod, self.rwd_gra, self.rwd_low_bound)
                next_state_center = get_center_data_by_id(next_state_id, next_state_id_mod, self.state_gra,
                                                          self.state_low_bound)
                done_center = bool(done[0])
                cost_center = get_center_data_by_id(cost_id, cost_id_mod, self.cost_gra, self.cost_low_bound)

                # 将数据写入CSV文件
                writer.writerow({
                    'State_Center': state_center,
                    'Act_Center': act_center,
                    'Reward_Center': reward_center,
                    'NextState_Center': next_state_center,
                    'Done': done_center,
                    'Cost_Center': cost_center,
                    'Weight': weight,
                    'Probability': probability
                })

        return data_to_save


if __name__ == '__main__':
    """ ACC part """
    # eval_path = "./conf/eval/highway_acc_eval.yaml"
    # eval_config = utils.load_yml(eval_path)

    # # rel_dis,rel_speed,acc,reward,next_rel_dis,next_rel_speed,done,cost
    # data = FirstAbstract(eval_config["dim"]["state_dim"],
    #                     ['rel_dis', 'rel_speed'], ['acc'], ['reward'], ['next_rel_dis', 'next_rel_speed'], ['done'],
    #                     ['cost'],
    #                     eval_config["granularity"]["state_gran"], eval_config["granularity"]["action_gran"],
    #                     eval_config["granularity"]["reward_gran"], eval_config["granularity"]["cost_gran"],
    #                     eval_config["lowerbound"]["state_lowerbound"], eval_config["upperbound"]["state_upperbound"],
    #                     eval_config["lowerbound"]["action_lowerbound"], eval_config["upperbound"]["action_upperbound"],
    #                     eval_config["lowerbound"]["reward_lowerbound"], eval_config["upperbound"]["reward_upperbound"],
    #                     eval_config["lowerbound"]["cost_lowerbound"], eval_config["upperbound"]["cost_upperbound"])
    
    # data.interval_mdp(
    #     "./data_analysis/acc_td3/dataset/td3_risk_acc_logs.csv",
    #     "./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_data_with_duplicate_row.csv")
    # data.del_duplicate_row_and_cal_pro("./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_data_with_duplicate_row.csv",
    #                                    "./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro.csv",
    #                                    "./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_raw_data.csv", 
    #                                    "./data_analysis/mdp/first_stage_abstract/acc_datasets/first_abstract_pro_center_data.csv")


    # """ LKA part """
    # eval_path = "./conf/eval/highway_lka_eval.yaml"
    # eval_config = utils.load_yml(eval_path)

    # # y_dis,v_y,cos_h,acc,reward,next_y_dis,next_v_y,next_cos_h,done,cost
    # data = FirstAbstract(eval_config["dim"]["state_dim"],
    #                     ['y_dis', 'v_y', 'cos_h'], ['acc'], ['reward'], 
    #                     ['next_y_dis', 'next_v_y', 'next_cos_h'], ['done'], ['cost'],
    #                     eval_config["granularity"]["state_gran"], eval_config["granularity"]["action_gran"],
    #                     eval_config["granularity"]["reward_gran"], eval_config["granularity"]["cost_gran"],
    #                     eval_config["lowerbound"]["state_lowerbound"], eval_config["upperbound"]["state_upperbound"],
    #                     eval_config["lowerbound"]["action_lowerbound"], eval_config["upperbound"]["action_upperbound"],
    #                     eval_config["lowerbound"]["reward_lowerbound"], eval_config["upperbound"]["reward_upperbound"],
    #                     eval_config["lowerbound"]["cost_lowerbound"], eval_config["upperbound"]["cost_upperbound"])
    
    # data.interval_mdp(
    #     "./data_analysis/datasets/lka_td3/td3_risk_lka_logs.csv",
    #     "./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_data_with_duplicate_row.csv")
    # data.del_duplicate_row_and_cal_pro("./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_data_with_duplicate_row.csv",
    #                                    "./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_pro.csv",
    #                                    "./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_pro_raw_data.csv", 
    #                                    "./data_analysis/mdp/first_stage_abstract/lka_datasets/first_abstract_pro_center_data.csv")
    
    """ ICA part """
    eval_path = "./conf/eval/highway_ica_eval.yaml"
    eval_config = utils.load_yml(eval_path)

    # x1,y1,x2,y2,reward,next_x1,next_y1,next_x2,next_y2,done,cost
    data = FirstAbstract(eval_config["dim"]["state_dim"],
                        ['x1', 'y1', 'x2', 'y2'], ['action'], ['reward'], 
                        ['next_x1', 'next_y1', 'next_x2', 'next_y2'], ['done'], ['cost'],
                        eval_config["granularity"]["state_gran"], eval_config["granularity"]["action_gran"],
                        eval_config["granularity"]["reward_gran"], eval_config["granularity"]["cost_gran"],
                        eval_config["lowerbound"]["state_lowerbound"], eval_config["upperbound"]["state_upperbound"],
                        eval_config["lowerbound"]["action_lowerbound"], eval_config["upperbound"]["action_upperbound"],
                        eval_config["lowerbound"]["reward_lowerbound"], eval_config["upperbound"]["reward_upperbound"],
                        eval_config["lowerbound"]["cost_lowerbound"], eval_config["upperbound"]["cost_upperbound"])
    
    data.interval_mdp(
        "./data_analysis/datasets/ica_dqn/dqn_risk_ica_logs.csv",
        "./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_data_with_duplicate_row.csv")
    data.del_duplicate_row_and_cal_pro("./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_data_with_duplicate_row.csv",
                                       "./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_pro.csv",
                                       "./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_pro_raw_data.csv", 
                                       "./data_analysis/mdp/first_stage_abstract/ica_datasets/first_abstract_pro_center_data.csv")