import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# Identify what is the first and final real steps given beginning step array-ups and ending step array-downs
def identify_walk(ups, downs):
    ups_tuple = [(i, "ups") for i in ups]
    downs_tuple = [(i, "downs") for i in downs]
    sorted_tuple = sorted(ups_tuple + downs_tuple, key=lambda x: x[0])  # sort arrays by first field
    count = 0
    while count < len(sorted_tuple) - 1 and (sorted_tuple[count + 1][1] == sorted_tuple[count][1] or sorted_tuple[count][1] == "downs"):
        count += 1
    begin = count
    #if sorted_tuple[begin][1] == "downs":
    #    begin += 1

    while count < len(sorted_tuple) - 1 and sorted_tuple[count + 1][1] != sorted_tuple[count][1]:
        count += 1
    end = count
    if sorted_tuple[end][1] == "ups":
        end -= 1

    sorted_tuple = sorted_tuple[begin:end + 1]
    print(sorted_tuple)
    return sorted_tuple

# Estimate step lengths by Kim's and Weinberg's approach
def estimate_step(f_force_amp, steps):
    steps_num = int(len(steps)/2)
    s_k = [0 for i in range(steps_num)]
    s_w = [0 for i in range(steps_num)]
    for i in range(steps_num):
        curr_step = f_force_amp[steps[2*i][0]:steps[2*i + 1][0]]
        n = len(curr_step)
        f_max = max(curr_step)
        f_min = min(curr_step)
        s_w[i] = (f_max-f_min)**(1/4)
        s_k[i] = (sum(curr_step)/n)**(1/3)

    return [s_k, s_w]

def main():
    df = pd.read_csv('3meter_5steps7.csv')
    #df = df.iloc[500:, :]
    #df = df.iloc[:-500, :]
    az = df['az (m/s^2)']
    aT = df['aT (m/s^2)']
    at = df['time']

    f_force_amp = np.array(aT)
    z_array = np.array(az)
    z_array -= np.average(z_array)

    # convolution between window and z acceleration
    step_func = np.hstack((np.ones(len(z_array)), -1*np.ones(len(z_array))))
    z_array_step = np.convolve(z_array, step_func, mode='valid')

    # find peaks which represent fastest inclines and declines
    step_down = signal.find_peaks(z_array_step, width=20, distance=40)[0]
    step_up = signal.find_peaks(-1*z_array_step, width=20, distance=40)[0]
    print("This is the start of a step {}.\n", format(step_up))
    print("This is the end of a step {}.\n", format(step_down))

    plt.figure()

    for cnt in range(len(step_down)):
        plt.plot((step_down[cnt], step_down[cnt]), (-10, 10), 'r')

    for cnt in range(len(step_up)):
        plt.plot((step_up[cnt], step_up[cnt]), (-10, 10), 'r')

    walk = identify_walk(step_up, step_down)
    steps_num = int(len(walk)/2)
    kim_step = [0 for i in range(steps_num)]
    weinberg_step = [0 for i in range(steps_num)]
    [kim_step, weinberg_step] = estimate_step(f_force_amp, walk)

    g_k = 300 / sum(kim_step)
    print("constant in kim approach: g_k = {}".format(g_k))
    kim_step = [element * g_k for element in kim_step]
    g_w = 300 / sum(weinberg_step)
    print("constant in weinberg approach: g_w = {}".format(g_w))
    weinberg_step = [element * g_w for element in weinberg_step]

    print("Steps sizes by kim approach: {}".format(kim_step))
    print("Steps sizes by Weinberg approach: {}".format(weinberg_step))

    print("Sum of steps in kims approach: {}".format(sum(kim_step)))
    print("Sum of steps in weinberg approach: {}".format(sum(weinberg_step)))

    # plt.plot(z_array)
    original, = plt.plot(z_array, label='before conv')
    conv, = plt.plot(z_array_step/10, label='after conv')
    conv1, = plt.plot(-1*z_array_step/10, label='after conv1')
    plt.legend(handles=[original, conv, conv1])
    # plt.subplot(1, 2, 1)
    # plt.plot(at, az)
    plt.xlabel('time (s)')
    plt.ylabel('Z acceleration')

    #plt.subplot(1, 2, 2)
    #plt.plot(at, aT)
    #plt.xlabel('time (s)')
    #plt.ylabel('total acceleration')

    plt.show()


if __name__ == "__main__":
    main()