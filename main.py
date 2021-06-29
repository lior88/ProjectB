import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



df = pd.read_csv('3meter_5steps5.csv')
#df = df.iloc[500:, :]
#df = df.iloc[:-500, :]
az = df['az (m/s^2)']
aT = df['aT (m/s^2)']
at = df['time']
a = np.array(az)
#a = a - np.mean(a)
amp = np.array(aT)
t = np.zeros(int(len(a)))
j = 0
for i in range(len(a)-2):
    if a[i] < 0 and a[i+2] > 0:
        t[j] = i+1
        j = j+1
    if a[i] > 0 and a[i+2] < 0:
        t[j] = i+1
        j = j+1

non_zero = np.sum(t != 0)
t = t[:non_zero]
#t = np.argwhere(np.abs(a) < 0.07)
print(t)

steps = np.zeros(int(len(t)))
j = 0
for i, value in enumerate(t):
    if i == 0:
        steps[j] = int(value)
        j = j + 1
    if i != len(t) - 1 and i != 0:
        if int(value) > (steps[j-1] + 10):
            steps[j] = int(value)
            j = j + 1

print(steps)
non_zero = np.sum(steps != 0)
steps = steps[:non_zero]
steps2 = np.zeros(int(len(steps)))
j=0
for i, value in enumerate(steps):
    if i % 3 == 0 or i % 3 == 2:
        steps2[j] = int(value)
        j = j + 1
non_zero = np.sum(steps2 != 0)
steps2 = steps2[:non_zero]
if non_zero % 2 != 0:
    steps2 = steps2[:-1]
print(steps2)


steps_num = int(len(steps2)/2)
#kim_step = [0 for i in range(steps_num)]
#weinberg_step = [0 for i in range(steps_num)]

s_k = [0 for i in range(steps_num)]
s_w = [0 for i in range(steps_num)]
for i in range(steps_num):
    curr_step = amp[int(steps2[2*i]):int(steps2[2*i + 1])]
    n = len(curr_step)
    f_max = max(curr_step)
    f_min = min(curr_step)
    s_w[i] = (f_max - f_min) ** (1 / 4)
    s_k[i] = (sum(curr_step) / n) ** (1 / 3)

g_k = 300 / sum(s_k)
print("constant in kim approach: g_k = {}".format(g_k))
kim_step = [element * g_k for element in s_k]
g_w = 300 / sum(s_w)
print("constant in weinberg approach: g_w = {}".format(g_w))
weinberg_step = [element * g_w for element in s_w]

print("Steps sizes by kim approach: {}".format(kim_step))
print("Steps sizes by Weinberg approach: {}".format(weinberg_step))

print("Sum of steps in kims approach: {}".format(sum(kim_step)))
print("Sum of steps in weinberg approach: {}".format(sum(weinberg_step)))

original, = plt.plot(a, label='before conv')
plt.xlabel('time (s)')
plt.ylabel('Z acceleration')
plt.show()
