import pandas as pd
import os
import numpy as np


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
    file_path = os.path.join(os.getcwd(), "acce_data", f"acce_data{file_number}.csv")
    data = pd.read_csv(file_path, header=None, names=names, dtype=np.float64)

    # fixing the upside down axis in some imu
    upside_down = [col for col in data.columns if any(x in col for x in ['imu_1', 'imu_3'])]
    data[upside_down] = data[upside_down] * -1
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
    avg_method.to_pickle(f"avg_method_file{file_number}.pkl")
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
    for col in data.columns:
        data[col + "_upper_bound"] = data[col].copy() + gap
        data[col + "_lower_bound"] = data[col].copy() - gap

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

    print(votes.columns)
    print(votes)
    return votes


'''
# 
# executing average method for all csv files
_ = [average_method(file_number=cnt, imu_num=4) for cnt in range(1, 11)]

# executing average method for single csv file
file_number = 1
imu_num = 4
res = average_method(file_number=1, imu_num=4)
print(res)
'''

file_number = 1
imu_num = 4
gap = 0.2
voting_method(file_number, imu_num, gap)




