import pandas as pd
import os
import numpy as np


def modify_file(file_name, rate):
    """
    averages the ridi dataset according to a known rate
    :param file_name: name of csv file to average
    :param rate: the sampling rate desired
    """
    data = pd.read_csv("C:/Users\liorb\Documents\ProjectB\data_publish_v2/" + file_name + "/processed/data.csv", index_col=0)
    names = list(data.columns.values)
    win_size = int(np.floor(200/rate))
    size = data.shape[0]
    num_times = int(np.floor(size/win_size))
    modified_data = pd.DataFrame(columns=names)

    for i in range(num_times):
        tmp = data.iloc[win_size*i:win_size*(i+1)]
        modified_data = modified_data.append(tmp.mean(axis=0), ignore_index=True)

    modified_data.to_csv("C:/Users\liorb\Documents\ProjectB\modified_ridi/" + file_name + "_modified.csv")

    return


def pre_pick(file_number, imu_num):
    """
    pre processing function
    supposed to run before preforming any combination between the different imus data
    :param file_number: number of csv input file
    :param imu_num: number of imu units been in use
    :return: x, y, z acceleration for each imu
    """
    acce_names = ['acce_x', 'acce_y', 'acce_z']
    names = []
    [names.extend([f"acce_x_imu_{cnt}", f"acce_y_imu_{cnt}", f"acce_z_imu_{cnt}"]) for cnt in range(imu_num)]
    file_path = os.path.join(os.getcwd(), "Recordings/second", f"acce_data{file_number}.csv")
    data = pd.read_csv(file_path, header=None, names=names, dtype=np.float64)

    # fixing the upside down axis in some imu ( arbitrary choice to change the y axis with the z axis)
    #upside_down = [col for col in data.columns if any(x in col for x in ['z_imu_1', 'z_imu_3', 'y_imu_1', 'y_imu_3'])]
    #upside_down = [col for col in data.columns if any(x in col for x in ['z_imu_1', 'z_imu_3', 'x_imu_1', 'x_imu_3'])]
    #data[upside_down] = data[upside_down] * -1

    #upside_down2 = [col for col in data.columns if any(x in col for x in ['y_imu_0', 'y_imu_2'])]
    #data[upside_down2] = data[upside_down2] * -1
    return data


def average_method(file_number, imu_num):
    """
    averaging all imus accelerations in axis x, y, and z
    :param file_number: number of csv input file
    :param imu_num: number of imu units been in use
    :return: x, y, z acceleration
    """
    data = pre_pick(file_number, imu_num).copy()
    acce_names = ['acce_x', 'acce_y', 'acce_z']
    for item in acce_names:
        filter_col = [col for col in data if col.startswith(item)]
        data[item + "_avg"] = data[filter_col].sum(axis=1)/imu_num

    # saving into file
    avg_method = data[["acce_x_avg", "acce_y_avg", "acce_z_avg"]]
    avg_method.to_csv(f"Processed/avg_method_file{file_number}.csv", header=0, index=False)
    return avg_method


def voting_method(file_number, imu_num, gap):
    """
    each time stamp, acceleration and axis gets data from one specific imu.
    the chosen imu has the maximal number of votes.
    imu gets a vote if another imu's value is in his range.

    :param file_number: number of csv input file
    :param imu_num: number of imu units been in use
    :param gap: size of region around data point to achieve a vote
    """
    acce_names = ['acce_x', 'acce_y', 'acce_z']
    data = pre_pick(file_number, imu_num).copy()
    data_og = data.copy()

    final_data = pd.read_csv(r'C:/Users\liorb\Documents\ProjectB\Recordings\second\acce_data1.csv') # for final count
    final_data = final_data.iloc[:, 0:3]


    for col in data.columns:
        data[col + "_upper_bound"] = data[col].copy() + gap
        data[col + "_lower_bound"] = data[col].copy() - gap

    idx = 0
    # loop for x, y, z accelerations
    for item in acce_names:
        filter_col = [col for col in data if col.startswith(item)]
        # contains columns of original data without bounds
        data_col = filter_col[:imu_num]
        # contains columns of upper bound
        upper_bound_col = [col for col in filter_col if 'upper_bound' in col]
        # contains columns of lower bounds
        lower_bound_col = [col for col in filter_col if 'lower_bound' in col]

        # loop to find votes for each imu
        for i in range(imu_num):
            # loop to find votes for imu number i
            for cnt in range(imu_num):
                lower_vote = data[lower_bound_col[i]] <= data[data_col[cnt]]
                upper_vote = data[upper_bound_col[i]] >= data[data_col[cnt]]
                vote = (lower_vote & upper_vote).array.astype(int)
                if item + f'_vote_{i}' in data.columns:
                    data[item + f'_vote_{i}'] += vote
                else:
                    data[item + f'_vote_{i}'] = vote

        # data frame that contains votes alone
        votes = data[data.columns[-4:]].copy()
        # amount of votes given to the chosen one
        votes[item + "_final_vote"] = votes.max(axis=1).array
        # name of the chosen imu
        votes[item + "_chosen_imu"] = votes.idxmax(axis=1).array

        best = votes[item + "_chosen_imu"].values
        for i in range(votes.shape[0]):
            if best[i] == (item + '_vote_{0}'):
                final_data[i, idx] = data_og[i, idx]
            elif best[i] == (item + '_vote_{1}'):
                final_data[i, idx] = data_og[i, 3 + idx]
            elif best[i] == (item + '_vote_{2}'):
                final_data[i, idx] = data_og[i, 6 + idx]
            elif best[i] == (item + '_vote_{3}'):
                final_data[i, idx] = data_og[i, 9 + idx]

        idx = idx + 1

    final_data.to_csv(f"Processed/voting_method_file{file_number}.csv", header=1, index=False)
    #print(final_data)
    #print(votes.columns)
    #print(votes)
    return votes


'''
# 
# executing average method for all csv files
_ = [average_method(file_number=cnt, imu_num=4) for cnt in range(1, 11)]
'''
#modify_file("hao_handheld1", 31.25)
#modify_file("hao_handheld2", 31.25)

# executing average method for single csv file

file_number = 2
imu_num = 4
gap = 0.2
average_method(file_number, imu_num)

voting_method(file_number, imu_num, gap)

for i in range(imu_num):
    tmp_data = pd.read_csv(fr'C:/Users\liorb\Documents\ProjectB\Recordings\second\acce_data{file_number}.csv')
    tmp_data = tmp_data.iloc[:, 3*i:3*(i+1)]
    #if (i%2) == 1:
        #tmp_data.iloc[:, 0] = tmp_data.iloc[:, 0] * -1  # x axis
        #tmp_data.iloc[:, 2] = tmp_data.iloc[:, 2] * -1  # z axis
    #else:
        #tmp_data.iloc[:, 1] = tmp_data.iloc[:, 1] * -1  # y axis
    tmp_data.to_csv(f"Processed/acce_data{file_number}_imu{i+1}.csv", header=0, index=False)

